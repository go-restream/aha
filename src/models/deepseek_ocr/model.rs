use anyhow::{Ok, Result};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{
    Activation, Conv2d, Conv2dConfig, Init, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder,
    conv2d, layer_norm, linear, linear_no_bias,
};

use crate::{
    models::{common::eager_attention_forward, deepseek_ocr::config::DeepseekOCRConfig},
    utils::tensor_utils::{index_select_2d, interpolate_linear},
};

pub struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    pub fn new(
        vb: VarBuilder,
        in_chans: usize,
        embed_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            stride,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let proj = conv2d(in_chans, embed_dim, kernel_size, cfg, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.proj.forward(xs)?;
        let xs = xs.permute((0, 2, 3, 1))?;
        Ok(xs)
    }
}

pub struct Attention {
    num_heads: usize,
    head_dim: usize,
    qkv: Linear,
    proj: Linear,
    scaling: f64,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        input_size: Option<(usize, usize)>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = if qkv_bias {
            linear(dim, dim * 3, vb.pp("qkv"))?
        } else {
            linear_no_bias(dim, dim * 3, vb.pp("qkv"))?
        };
        let proj = linear(dim, dim, vb.pp("proj"))?;
        let mut rel_pos_h = None;
        let mut rel_pos_w = None;
        if use_rel_pos {
            if input_size.is_none() {
                return Err(anyhow::anyhow!(
                    "Input size must be provided if using relative positional encoding."
                ));
            }
            let input_size = input_size.unwrap();
            let h_len = 2 * input_size.0 - 1;
            let w_len = 2 * input_size.1 - 1;
            rel_pos_h = Some(vb.get_with_hints((h_len, head_dim), "rel_pos_h", Init::Const(0.))?);
            rel_pos_w = Some(vb.get_with_hints((w_len, head_dim), "rel_pos_w", Init::Const(0.))?);
        }

        Ok(Self {
            num_heads,
            head_dim,
            qkv,
            proj,
            scaling,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        })
    }

    fn get_rel_pos(&self, q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
        let max_rel_dist = 2 * std::cmp::max(q_size, k_size) - 1;
        let rel_pos_resized = if rel_pos.dim(0)? != max_rel_dist {
            let rel_pos = rel_pos
                .to_dtype(candle_core::DType::F32)?
                .t()?
                .unsqueeze(0)?
                .contiguous()?;
            let rel_pos_resized = interpolate_linear(&rel_pos, max_rel_dist, None)?;
            let rel_pos_resized = rel_pos_resized
                .squeeze(0)?
                .t()?
                .contiguous()?
                .to_dtype(rel_pos.dtype())?;
            rel_pos_resized
        } else {
            rel_pos.clone()
        };
        let q_coords = Tensor::arange(0 as f32, q_size as f32, rel_pos.device())?
            .unsqueeze(D::Minus1)?
            .affine((k_size as f64 / q_size as f64).max(1.0), 0.0)?;
        let k_coords = Tensor::arange(0 as f32, k_size as f32, rel_pos.device())?
            .unsqueeze(0)?
            .affine((q_size as f64 / k_size as f64).max(1.0), 0.0)?;
        let relative_coords = q_coords
            .broadcast_sub(&k_coords)?
            .affine(1.0, (k_size - 1) as f64)?
            .affine((q_size as f64 / k_size as f64).max(1.0), 0.0)?;
        let relative_coords = relative_coords
            .to_dtype(candle_core::DType::U32)?
            .contiguous()?;
        let rel_pos_resized = rel_pos_resized.contiguous()?;
        let res = index_select_2d(&rel_pos_resized, &relative_coords)?;
        Ok(res)
    }

    fn add_decomposed_rel_pos(
        &self,
        q: &Tensor,
        rel_pos_h: &Tensor,
        rel_pos_w: &Tensor,
        q_size: (usize, usize),
        k_size: (usize, usize),
    ) -> Result<(Tensor, Tensor)> {
        let (q_h, q_w) = q_size;
        let (k_h, k_w) = k_size;
        let rh = self.get_rel_pos(q_h, k_h, rel_pos_h)?; // (h, w, dim)
        let rh = rh.t()?; // (h, dim, w)        
        let rw = self.get_rel_pos(q_w, k_w, rel_pos_w)?;
        let rw = rw.t()?;
        let (b, _, dim) = q.dims3()?;
        let r_q = q.reshape((b, q_h, q_w, dim))?;
        let rel_h = r_q.broadcast_matmul(&rh)?;
        let rel_w = r_q.broadcast_matmul(&rw)?;
        let rel_h = rel_h
            .unsqueeze(D::Minus1)?
            .reshape((b, q_h * q_w, k_h, 1))?;
        let rel_w = rel_w
            .unsqueeze(D::Minus2)?
            .reshape((b, q_h * q_w, 1, k_w))?;
        Ok((rel_h, rel_w))
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = xs.dims4()?;
        // (3, B, n_head, h*w, head_dim)
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, h * w, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let xs = if self.use_rel_pos {
            let q_reshape = query_states.reshape((b * self.num_heads, h * w, ()))?;
            let (rel_h, rel_w) = self.add_decomposed_rel_pos(
                &q_reshape,
                self.rel_pos_h.as_ref().unwrap(),
                self.rel_pos_w.as_ref().unwrap(),
                (h, w),
                (h, w),
            )?;
            let (_, rel_h_dim1, rel_h_dim2, rel_h_dim3) = rel_h.dims4()?;
            let rel_h = rel_h.reshape((b, self.num_heads, rel_h_dim1, rel_h_dim2, rel_h_dim3))?;
            let (_, rel_w_dim1, rel_w_dim2, rel_w_dim3) = rel_w.dims4()?;
            let rel_w = rel_w.reshape((b, self.num_heads, rel_w_dim1, rel_w_dim2, rel_w_dim3))?;
            let attn_bias = rel_h.broadcast_add(&rel_w)?.reshape((
                b,
                self.num_heads,
                rel_h_dim1,
                rel_h_dim2 * rel_w_dim3,
            ))?;
            let xs = eager_attention_forward(
                &query_states,
                &key_states,
                &value_states,
                None,
                Some(&attn_bias),
                self.scaling,
            )?;
            xs
        } else {
            eager_attention_forward(
                &query_states,
                &key_states,
                &value_states,
                None,
                None,
                self.scaling,
            )?
        };
        // (b, h*w, n_head, dim)
        let xs = xs.reshape((b, h * w, ()))?.reshape((b, h, w, ()))?;
        let xs = self.proj.forward(&xs)?;
        Ok(xs)
    }
}

pub struct MLPBlock {
    linear1: Linear,
    linear2: Linear,
    act: Activation,
}

impl MLPBlock {
    pub fn new(
        vb: VarBuilder,
        embedding_dim: usize,
        mlp_dim: usize,
        act: Activation,
    ) -> Result<Self> {
        let linear1 = linear(embedding_dim, mlp_dim, vb.pp("lin1"))?;
        let linear2 = linear(mlp_dim, embedding_dim, vb.pp("lin2"))?;
        Ok(Self {
            linear1,
            linear2,
            act,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)?;
        Ok(xs)
    }
}

pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: MLPBlock,
    window_size: usize,
}

impl Block {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        eps: f64,
        act: Activation,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        window_size: usize,
        input_size: Option<(usize, usize)>,
    ) -> Result<Self> {
        let ln_config = LayerNormConfig {
            eps,
            remove_mean: true, // true for layernorm, false for RMSNorm
            affine: true,      // true for with bias, false for without bias
        };
        let norm1 = layer_norm(dim, ln_config, vb.pp("norm1"))?;
        let input_size = if window_size == 0 {
            input_size
        } else {
            Some((window_size, window_size))
        };
        let attn = Attention::new(
            vb.pp("attn"),
            dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            input_size,
        )?;
        let norm2 = layer_norm(dim, ln_config, vb.pp("norm2"))?;
        let mlp_dim = (dim as f32 * mlp_ratio) as usize;
        let mlp = MLPBlock::new(vb.pp("mlp"), dim, mlp_dim, act)?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }

    pub fn window_partition(
        &self,
        x: &Tensor,
        window_size: usize,
    ) -> Result<(Tensor, (usize, usize))> {
        let (b, h, w, c) = x.dims4()?;
        let pad_h = (window_size - h % window_size) % window_size;
        let pad_w = (window_size - w % window_size) % window_size;
        let x = if pad_h > 0 || pad_w > 0 {
            let x = x.pad_with_zeros(1, 0, pad_h)?;
            let x = x.pad_with_zeros(2, 0, pad_w)?;
            x
        } else {
            x.clone()
        };
        let hp = h + pad_h;
        let wp = w + pad_w;
        let x = x.reshape((
            b,
            hp / window_size,
            window_size,
            wp / window_size,
            window_size,
            c,
        ))?;
        let windows = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((
            (),
            window_size,
            window_size,
            c,
        ))?;
        Ok((windows, (hp, wp)))
    }

    // pub fn window_unpartition(
    //     &self,
    //     x: &Tensor,
    //     window_size: usize,
    //     pad_hw: (usize, usize),
    //     hw: (usize, usize),
    // ) -> Result<Tensor> {

    // }

    // pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    //     let shortcut = xs.clone();
    //     let xs = self.norm1.forward(xs)?;
    //     let xs = if self.window_size > 0 {
    //         let h = xs.dim(1)?;
    //         let w = xs.dim(2)?;
    //         let (x, (hp, wp)) = self.window_partition(&xs, self.window_size)?;
    //         let x = self.attn.forward(&x)?;
    //     } else {
    //         self.attn.forward(&xs)?
    //     };
    // }
}

pub struct ImageEncoderViT {
    img_size: usize,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<Block>,
}

pub struct VitModel {}

pub struct DeepseekV2Model {}

pub struct MlpProjector {}

pub struct DeepseekOCRModel {
    config: DeepseekOCRConfig,
    sam_model: ImageEncoderViT,
    vision_model: VitModel,
    language_model: DeepseekV2Model,
    projector: MlpProjector,
    embed_std: f64,
    image_newline: Tensor,
    view_seperator: Tensor,
    lm_head: Linear,
}
