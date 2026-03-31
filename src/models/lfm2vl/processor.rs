use crate::params::chat::ChatCompletionParameters;
use crate::{
    models::lfm2vl::config::{Lfm2ImageConfig, Lfm2ProcessorConfig},
    utils::{
        img_utils::{
            crop_img, extract_images, find_closest_aspect_ratio, generate_target_ratios_sorted,
            img_smart_resize, img_transform,
        },
        round_by_factor,
    },
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::DynamicImage;

pub struct Lfm2VLProcessor {
    dtype: DType,
    device: Device,
    image_config: Lfm2ImageConfig,
    max_num_patches: usize,
    total_factor: u32,
    max_pixel_num: usize,
    smart_resize_min_pixels: usize,
    smart_resize_max_pixels: usize,
    target_ratios: Vec<(u32, u32)>,
    img_mean: Tensor,
    img_std: Tensor,
    tokens_per_tile: usize,
    image_token: String,
    // image_token_id: u32,
    image_start_token: String,
    image_end_token: String,
    image_thumbnail_token: String,
}

#[allow(clippy::type_complexity)]
impl Lfm2VLProcessor {
    pub fn new(path: &str, dtype: DType, device: &Device) -> Result<Self> {
        assert!(
            std::path::Path::new(path).exists(),
            "model path file not exists"
        );
        let processor_cfg_path = path.to_string() + "/processor_config.json";
        let processor_cfg =
            serde_json::from_slice::<Lfm2ProcessorConfig>(&std::fs::read(processor_cfg_path)?);

        let image_config = match processor_cfg {
            Ok(cfg) => cfg.image_processor,
            Err(_) => {
                let processor_cfg_path = path.to_string() + "/preprocessor_config.json";
                serde_json::from_slice::<Lfm2ImageConfig>(&std::fs::read(processor_cfg_path)?)?
            }
        };
        // 256
        let max_thumbnail_image_patches =
            image_config.max_image_tokens * image_config.downsample_factor.pow(2);
        // 1024
        let tile_size_patches = if image_config.do_image_splitting {
            (image_config.tile_size / image_config.encoder_patch_size).pow(2)
        } else {
            1
        };
        // 1024
        let max_num_patches = max_thumbnail_image_patches.max(tile_size_patches);
        let total_factor =
            (image_config.encoder_patch_size * image_config.downsample_factor) as u32;
        let token_pixels =
            image_config.encoder_patch_size.pow(2) * image_config.downsample_factor.pow(2);
        let max_pixel_num = ((image_config.max_image_tokens * token_pixels) as f64
            * image_config.max_pixels_tolerance) as usize;

        let smart_resize_min_pixels = image_config.min_image_tokens * token_pixels;
        let smart_resize_max_pixels = image_config.max_image_tokens * token_pixels;
        let target_ratios = generate_target_ratios_sorted(
            image_config.min_tiles as u32,
            image_config.max_tiles as u32,
        );
        let img_mean =
            Tensor::from_slice(&image_config.image_mean, (3, 1, 1), device)?.to_dtype(dtype)?;
        let img_std =
            Tensor::from_slice(&image_config.image_std, (3, 1, 1), device)?.to_dtype(dtype)?;

        let tokens_per_tile = (image_config.tile_size
            / image_config.encoder_patch_size
            / image_config.downsample_factor)
            .pow(2);

        Ok(Self {
            dtype,
            device: device.clone(),
            image_config,
            max_num_patches,
            total_factor,
            max_pixel_num,
            smart_resize_min_pixels,
            smart_resize_max_pixels,
            target_ratios,
            img_mean,
            img_std,
            tokens_per_tile,
            image_token: "<image>".to_string(),
            // image_token_id: 396,
            image_start_token: "<|image_start|>".to_string(),
            image_end_token: "<|image_end|>".to_string(),
            image_thumbnail_token: "<|img_thumbnail|>".to_string(),
        })
    }

    fn is_image_too_large(&self, height: u32, width: u32) -> bool {
        let h_bar = self
            .image_config
            .encoder_patch_size
            .max(round_by_factor(height, self.total_factor) as usize);
        let w_bar = self
            .image_config
            .encoder_patch_size
            .max(round_by_factor(width, self.total_factor) as usize);
        h_bar * w_bar > self.max_pixel_num
    }

    fn get_grid_layout(&self, height: u32, width: u32) -> (u32, u32) {
        let aspect_ratio = width as f64 / height as f64;
        let (grid_width, grid_height) = find_closest_aspect_ratio(
            aspect_ratio,
            &self.target_ratios,
            width,
            height,
            self.image_config.tile_size as u32,
        );
        (grid_width, grid_height)
    }

    fn crop_image_to_patches(
        &self,
        img: &DynamicImage,
        height: u32,
        width: u32,
        new_height: u32,
        new_width: u32,
    ) -> Result<(Vec<DynamicImage>, usize, usize)> {
        let (grid_width, grid_height) = self.get_grid_layout(height, width);
        let mut processed_images = crop_img(
            img,
            grid_height,
            grid_width,
            self.image_config.tile_size as u32,
        );
        if self.image_config.use_thumbnail && processed_images.len() != 1 {
            let thumbnail_img = img.resize_exact(
                new_width,
                new_height,
                image::imageops::FilterType::CatmullRom,
            );
            processed_images.push(thumbnail_img);
        }
        Ok((processed_images, grid_width as usize, grid_height as usize))
    }

    fn resize_and_split(
        &self,
        img: &DynamicImage,
    ) -> Result<(Vec<DynamicImage>, usize, usize, u32, u32)> {
        let height = img.height();
        let width = img.width();
        let is_image_large = self.is_image_too_large(height, width);

        let (new_height, new_width) = img_smart_resize(
            height,
            width,
            self.total_factor,
            self.smart_resize_min_pixels as u32,
            self.smart_resize_max_pixels as u32,
        )?;
        let (images, num_cols, num_rows) = if is_image_large && self.image_config.do_image_splitting
        {
            self.crop_image_to_patches(img, height, width, new_height, new_width)?
        } else {
            let img = img.resize_exact(
                new_width,
                new_height,
                image::imageops::FilterType::CatmullRom,
            );
            (vec![img], 1, 1)
        };

        Ok((images, num_cols, num_rows, new_height, new_width))
    }

    pub fn process_imgs(
        &self,
        imgs: Vec<DynamicImage>,
    ) -> Result<(
        Tensor,
        Tensor,
        Tensor,
        Vec<usize>,
        Vec<usize>,
        Vec<(u32, u32)>,
    )> {
        let patch_size = self.image_config.encoder_patch_size;
        let mut images_list = vec![];
        let mut images_mask_list = vec![];
        let mut processed_spatial_shapes = vec![];
        let mut num_cols_list = vec![];
        let mut num_rows_list = vec![];
        let mut image_size_list = vec![];
        for img in &imgs {
            // img: 过大的切分，返回切块图像 + reshape图像
            // 小的直接reshape
            let (imgs, num_cols, num_rows, new_height, new_width) = self.resize_and_split(img)?;
            num_cols_list.push(num_cols);
            num_rows_list.push(num_rows);
            image_size_list.push((new_height, new_width));
            for img in imgs {
                // img-> tensor
                let img_t = img_transform(
                    &img,
                    &self.img_mean,
                    &self.img_std,
                    &self.device,
                    self.dtype,
                )?;

                // 图像嵌入 -> (seq_len, embedding_size)
                let (c, h, w) = img_t.dims3()?;
                let num_patches_height = h / patch_size;
                let num_patches_width = w / patch_size;
                let patched_image = img_t.reshape((
                    c,
                    num_patches_height,
                    patch_size,
                    num_patches_width,
                    patch_size,
                ))?;
                // (c, num_patches_height, patch_size, num_patches_width, patch_size)
                // -> (num_patches_height, num_patches_width, patch_size, patch_size, c)
                let patched_image = patched_image.permute((1, 3, 2, 4, 0))?;
                let patched_image =
                    patched_image.reshape((num_patches_height * num_patches_width, ()))?;

                // padding
                let curren_length = patched_image.dim(0)?;
                let padding_length = self.max_num_patches - curren_length;
                let (patched_image, pixel_mask) = if self.image_config.do_pad && padding_length > 0
                {
                    let mut pixel_mask = Tensor::ones(curren_length, DType::U32, &self.device)?;
                    let padding_image = patched_image.pad_with_zeros(0, 0, padding_length)?;
                    let pad = Tensor::zeros(padding_length, DType::U32, &self.device)?;
                    pixel_mask = Tensor::cat(&[&pixel_mask, &pad], 0)?;
                    (padding_image, pixel_mask)
                } else {
                    let pixel_mask = Tensor::ones(curren_length, DType::U32, &self.device)?;
                    (patched_image, pixel_mask)
                };
                images_list.push(patched_image);
                images_mask_list.push(pixel_mask);
                processed_spatial_shapes
                    .push(vec![num_patches_height as u32, num_patches_width as u32]);
            }
        }
        let pixel_values = Tensor::stack(&images_list, 0)?;
        let pixel_attention_mask = Tensor::stack(&images_mask_list, 0)?;
        let spatial_shapes = Tensor::new(processed_spatial_shapes, &self.device)?;
        Ok((
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
            num_cols_list,
            num_rows_list,
            image_size_list,
        ))
    }

    fn build_image_tokens(&self, rows: usize, cols: usize, tokens_for_image: usize) -> String {
        let mut parts = "".to_string();
        parts += &self.image_start_token;
        if rows > 1 && cols > 1 {
            for row in 0..rows {
                for col in 0..cols {
                    parts += &format!("<|img_row_{}_col_{}|>", row + 1, col + 1);
                    parts += &(self.image_token.repeat(self.tokens_per_tile));
                }
            }
            if self.image_config.use_thumbnail {
                parts += &self.image_thumbnail_token;
                parts += &(self.image_token.repeat(tokens_for_image));
            }
        } else {
            parts += &(self.image_token.repeat(tokens_for_image));
        }
        parts += &self.image_end_token;
        parts
    }

    fn expand_text_with_placeholders(
        &self,
        text: &str,
        num_cols_list: Vec<usize>,
        num_rows_list: Vec<usize>,
        image_size_list: Vec<(u32, u32)>,
    ) -> String {
        let text_parts: Vec<&str> = text.split(&self.image_token).collect();
        let mut result_parts = "".to_string();
        for i in 0..num_cols_list.len() {
            result_parts += text_parts[i];
            let rows = num_rows_list[i];
            let cols = num_cols_list[i];
            let image_size = image_size_list[i];
            let (h, w) = image_size;
            let tokens_for_image = (h as usize
                / self.image_config.encoder_patch_size
                / self.image_config.downsample_factor)
                * (w as usize
                    / self.image_config.encoder_patch_size
                    / self.image_config.downsample_factor);
            let sub_str = self.build_image_tokens(rows, cols, tokens_for_image);
            result_parts += &sub_str;
        }
        if text_parts.len() > num_cols_list.len() {
            result_parts += text_parts[text_parts.len() - 1];
        }
        result_parts
    }

    pub fn process_info(
        &self,
        messages: &ChatCompletionParameters,
        text: &str,
    ) -> Result<(Tensor, Tensor, Tensor, String)> {
        let imgs = extract_images(messages)?;
        let (
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
            num_cols_list,
            num_rows_list,
            image_size_list,
        ) = self.process_imgs(imgs)?;
        let text =
            self.expand_text_with_placeholders(text, num_cols_list, num_rows_list, image_size_list);
        Ok((pixel_values, pixel_attention_mask, spatial_shapes, text))
    }
}
