use crate::{
    models::common::{
        MultiModalData,
        generate::{GenerationContext, generate_generic, generate_stream_generic},
    },
    params::chat::{ChatCompletionParameters, ChatCompletionResponse},
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        lfm2::config::Lfm2GenerateConfig,
        lfm2vl::{config::Lfm2VLConfig, model::Lfm2VLModel, processor::Lfm2VLProcessor},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct Lfm2VLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    device: Device,
    model: Lfm2VLModel,
    processor: Lfm2VLProcessor,
    model_name: String,
}
impl<'a> Lfm2VLGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let gen_cfg_path = path.to_string() + "/generation_config.json";
        let gen_cfg: Lfm2GenerateConfig = serde_json::from_slice(&std::fs::read(gen_cfg_path)?)?;
        let cfg_path = path.to_string() + "/config.json";
        let cfg: Lfm2VLConfig = serde_json::from_slice(&std::fs::read(cfg_path)?)?;

        let model_path = find_type_files(path, "safetensors")?;
        let dtype = get_dtype(dtype, &cfg.dtype);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_path, dtype, &device)? };
        let eos_ids = vec![gen_cfg.eos_token_id];
        let model = Lfm2VLModel::new(vb, &cfg, eos_ids)?;
        let processor = Lfm2VLProcessor::new(path, dtype, &device)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("lfm2.5-vl")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            device,
            model,
            processor,
            model_name,
        })
    }
}

impl<'a> GenerateModel for Lfm2VLGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (pixel_values, pixel_attention_mask, spatial_shapes, text) =
            self.processor.process_info(&mes, &mes_render)?;
        let input_ids = self.tokenizer.text_encode(text, &self.device)?;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let mut ctx = GenerationContext::new(
            mes.temperature,
            mes.top_p,
            None,
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );

        let data_vec = vec![
            pixel_values.into(),
            pixel_attention_mask.into(),
            spatial_shapes.into(),
        ];
        let data = MultiModalData::new(data_vec);
        generate_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            &mut ctx,
            &self.model_name,
        )
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn rocket::futures::Stream<
                    Item = Result<crate::params::chat::ChatCompletionChunkResponse, anyhow::Error>,
                > + Send
                + Unpin
                + '_,
        >,
    > {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (pixel_values, pixel_attention_mask, spatial_shapes, text) =
            self.processor.process_info(&mes, &mes_render)?;
        let input_ids = self.tokenizer.text_encode(text, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let data_vec = vec![
            pixel_values.into(),
            pixel_attention_mask.into(),
            spatial_shapes.into(),
        ];
        let data = MultiModalData::new(data_vec);
        let seed = mes.seed.unwrap_or(34562) as u64;
        let stream = generate_stream_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            mes.temperature,
            mes.top_p,
            None,
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            sample_len,
            false,
            &self.device,
            &self.model_name,
        )?;
        Ok(Box::new(Box::pin(stream)))
    }
}
