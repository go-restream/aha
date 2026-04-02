use crate::models::common::MultiModalData;
use crate::models::common::generate::{
    GenerationContext, generate_generic, generate_stream_generic,
};
use crate::params::chat::{ChatCompletionParameters, ChatCompletionResponse};
use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        lfm2::{
            config::{Lfm2Config, Lfm2GenerateConfig},
            model::Lfm2Model,
        },
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub struct Lfm2GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    device: Device,
    model: Lfm2Model,
    model_name: String,
}
impl<'a> Lfm2GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let gen_cfg_path = path.to_string() + "/generation_config.json";
        let gen_cfg: Lfm2GenerateConfig = serde_json::from_slice(&std::fs::read(gen_cfg_path)?)?;
        let cfg_path = path.to_string() + "/config.json";
        let cfg: Lfm2Config = serde_json::from_slice(&std::fs::read(cfg_path)?)?;
        let model_path = find_type_files(path, "safetensors")?;
        let cfg_dtype = if let Some(dtype) = &cfg.dtype {
            dtype.clone()
        } else if let Some(dtype) = &cfg.torch_dtype {
            dtype.clone()
        } else {
            "bfloat16".to_string()
        };
        let dtype = get_dtype(dtype, &cfg_dtype);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_path, dtype, &device)? };
        let eos_ids = vec![gen_cfg.eos_token_id];
        let model = Lfm2Model::new(vb, &cfg, eos_ids)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("lfm2")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            device,
            model,
            model_name,
        })
    }
}

impl<'a> GenerateModel for Lfm2GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut ctx = GenerationContext::new(
            mes.temperature,
            mes.top_p,
            None,
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );

        let data = MultiModalData::new(vec![]);
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
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let data = MultiModalData::new(vec![]);
        let seed = mes.seed.unwrap_or(34562) as u64;
        let stream = generate_stream_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            mes.temperature,
            mes.top_p,
            None,
            seed,
            sample_len,
            false,
            &self.device,
            &self.model_name,
        )?;
        Ok(Box::new(Box::pin(stream)))
    }
}
