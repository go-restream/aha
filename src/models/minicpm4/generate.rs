use crate::models::common::MultiModalData;
use crate::models::common::generate::{
    GenerationContext, generate_generic, generate_stream_generic,
};
use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use rocket::futures::Stream;

use crate::models::minicpm4::config::MiniCPM4Config;
use crate::models::minicpm4::model::MiniCPMModel;
// use crate::models::GenerateStream;
use crate::utils::{find_type_files, get_device, get_dtype};
use crate::{chat_template::ChatTemplate, models::GenerateModel, tokenizer::TokenizerModel};

pub struct MiniCPMGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    minicpm: MiniCPMModel,
    device: Device,
    model_name: String,
}

impl<'a> MiniCPMGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: MiniCPM4Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let endoftext_id = cfg.eos_token_id[0];
        let im_end_id = cfg.eos_token_id[1];
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let eos_ids = vec![endoftext_id, im_end_id];
        let minicpm = MiniCPMModel::new(vb, cfg, eos_ids)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("minicpm4")
            .to_string();
        Ok(MiniCPMGenerateModel {
            chat_template,
            tokenizer,
            minicpm,
            device: device.clone(),
            model_name,
        })
    }
}

impl<'a> GenerateModel for MiniCPMGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let sample_len = mes.max_tokens.unwrap_or(2048);
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
            &mut self.minicpm,
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
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let data = MultiModalData::new(vec![]);
        let sample_len = mes.max_tokens.unwrap_or(512);
        let stream = generate_stream_generic(
            &mut self.minicpm,
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
