use std::io::{Read, Seek};

use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, quantized::gguf_file};
use candle_nn::Embedding;

use crate::{
    chat_template::ChatTemplate, gguf_models::common::Gguf, tokenizer::TokenizerModel,
    utils::get_device,
};

pub struct GgufQwen3_5<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    embed_tokens: Embedding,
    device: Device,
    dtype: DType,
}

impl<'a> GgufQwen3_5<'a> {
    pub fn new(file_path: &str, device: Option<&Device>) -> Result<Self> {
        if !file_path.ends_with("gguf") {
            return Err(anyhow!("model file suffix must be gguf: {file_path}"));
        }
        let mut reader = std::fs::File::open(file_path)?;
        let content = gguf_file::Content::read(&mut reader)?;
        let device = get_device(device);
        Self::from_gguf(content, &mut reader, &device)
    }
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gguf = Gguf::new(content, reader, device.clone());

        let chat_template_str = gguf
            .get_matedata("tokenizer.chat_template")?
            .to_string()?
            .clone();
        let chat_template = ChatTemplate::str_init(&chat_template_str)?;
        let tokenizer = gguf.build_tokenizer(Some(false), Some(false), Some(false))?;

        let num_attention_heads =
            gguf.get_matedata("qwen35.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = gguf
            .get_matedata("qwen35.attention.head_count_kv")?
            .to_u32()? as usize;
        let head_dim = gguf.get_matedata("qwen35.attention.key_length")?.to_u32()? as usize;
        let num_layers = gguf.get_matedata("qwen35.block_count")?.to_u32()? as usize;
        let hidden_size = gguf.get_matedata("qwen35.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings =
            gguf.get_matedata("qwen35.context_length")?.to_u32()? as usize;
        let rms_norm_eps = gguf
            .get_matedata("qwen35.attention.layer_norm_rms_epsilon")?
            .to_f32()? as f64;
        let rope_freq_base = gguf.get_matedata("qwen35.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gguf.get_matedata("general.type") {
            Ok(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            Err(_) => DType::F16,
        };

        let embed_tensor = gguf.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);
        Ok(Self {
            chat_template,
            tokenizer,
            embed_tokens,
            device: device.clone(),
            dtype,
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<()> {
        let render = self.chat_template.apply_chat_template(&mes)?;
        println!("render: {}", render);
        let input_ids = self.tokenizer.text_encode(render, &self.device)?;
        println!("input_ids: {}", input_ids);
        Ok(())
    }
}
