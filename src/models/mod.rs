pub mod bigvgan;
pub mod campplus;
pub mod common;
pub mod deepseek_ocr;
pub mod feature_extractor;
pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod glm_ocr;
pub mod hunyuan_ocr;
pub mod lfm2;
pub mod lfm2vl;
pub mod mask_gct;
pub mod minicpm4;
pub mod paddleocr_vl;
pub mod qwen2;
pub mod qwen2_5vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_asr;
pub mod qwen3vl;
pub mod rmbg2_0;
pub mod voxcpm;
pub mod w2v_bert_2_0;

use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use rocket::futures::Stream;

use crate::models::{
    deepseek_ocr::generate::DeepseekOCRGenerateModel,
    fun_asr_nano::generate::FunAsrNanoGenerateModel,
    glm_asr_nano::generate::GlmAsrNanoGenerateModel, glm_ocr::generate::GlmOcrGenerateModel,
    hunyuan_ocr::generate::HunyuanOCRGenerateModel, lfm2::generate::Lfm2GenerateModel,
    lfm2vl::generate::Lfm2VLGenerateModel, minicpm4::generate::MiniCPMGenerateModel,
    paddleocr_vl::generate::PaddleOCRVLGenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel,
    qwen3::generate::Qwen3GenerateModel, qwen3_5::generate::Qwen3_5GenerateModel,
    qwen3_asr::generate::Qwen3AsrGenerateModel, qwen3vl::generate::Qwen3VLGenerateModel,
    rmbg2_0::generate::RMBG2_0Model, voxcpm::generate::VoxCPMGenerate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WhichModel {
    #[value(name = "lfm2-1.2b")]
    LFM2_1_2B,
    #[value(name = "lfm2.5-1.2b-instruct")]
    LFM2_5_1_2BInstruct,
    #[value(name = "lfm2.5-vl-1.6b")]
    LFM2_5VL1_6B,
    #[value(name = "lfm2-vl-1.6b")]
    LFM2VL1_6B,
    #[value(name = "minicpm4-0.5b")]
    MiniCPM4_0_5B,
    #[value(name = "qwen2.5vl-3b")]
    Qwen2_5VL3B,
    #[value(name = "qwen2.5vl-7b")]
    Qwen2_5VL7B,
    #[value(name = "qwen3-0.6b")]
    Qwen3_0_6B,
    #[value(name = "qwen3.5-0.8b")]
    Qwen3_5_0_8B,
    #[value(name = "qwen3.5-2b")]
    Qwen3_5_2B,
    #[value(name = "qwen3.5-4b")]
    Qwen3_5_4B,
    #[value(name = "qwen3.5-9b")]
    Qwen3_5_9B,
    #[value(name = "qwen3.5-gguf")]
    Qwen3_5Gguf,
    #[value(name = "qwen3asr-0.6b")]
    Qwen3ASR0_6B,
    #[value(name = "qwen3asr-1.7b")]
    Qwen3ASR1_7B,
    #[value(name = "qwen3vl-2b")]
    Qwen3VL2B,
    #[value(name = "qwen3vl-4b")]
    Qwen3VL4B,
    #[value(name = "qwen3vl-8b")]
    Qwen3VL8B,
    #[value(name = "qwen3vl-32b")]
    Qwen3VL32B,
    #[value(name = "deepseek-ocr")]
    DeepSeekOCR,
    #[value(name = "deepseek-ocr2")]
    DeepSeekOCR2,
    #[value(name = "hunyuan-ocr")]
    HunyuanOCR,
    #[value(name = "paddleocr-vl")]
    PaddleOCRVL,
    #[value(name = "paddleocr-vl1.5")]
    PaddleOCRVL1_5,
    #[value(name = "rmbg2.0")]
    RMBG2_0,
    #[value(name = "voxcpm")]
    VoxCPM,
    #[value(name = "voxcpm1.5")]
    VoxCPM1_5,
    #[value(name = "glm-asr-nano-2512")]
    GlmASRNano2512,
    #[value(name = "fun-asr-nano-2512")]
    FunASRNano2512,
    #[value(name = "glm-ocr")]
    GlmOCR,
}

impl WhichModel {
    /// Get the ModelScope model ID for this model variant
    pub fn model_id(self) -> &'static str {
        match self {
            WhichModel::LFM2_1_2B => "LiquidAI/LFM2-1.2B",
            WhichModel::LFM2_5_1_2BInstruct => "LiquidAI/LFM2.5-1.2B-Instruct",
            WhichModel::LFM2_5VL1_6B => "LiquidAI/LFM2.5-VL-1.6B",
            WhichModel::LFM2VL1_6B => "LiquidAI/LFM2-VL-1.6B",
            WhichModel::MiniCPM4_0_5B => "OpenBMB/MiniCPM4-0.5B",
            WhichModel::Qwen2_5VL3B => "Qwen/Qwen2.5-VL-3B-Instruct",
            WhichModel::Qwen2_5VL7B => "Qwen/Qwen2.5-VL-7B-Instruct",
            WhichModel::Qwen3_0_6B => "Qwen/Qwen3-0.6B",
            WhichModel::Qwen3_5_0_8B => "Qwen/Qwen3.5-0.8B",
            WhichModel::Qwen3_5_2B => "Qwen/Qwen3.5-2B",
            WhichModel::Qwen3_5_4B => "Qwen/Qwen3.5-4B",
            WhichModel::Qwen3_5_9B => "Qwen/Qwen3.5-9B",
            WhichModel::Qwen3_5Gguf => "GGUF",
            WhichModel::Qwen3ASR0_6B => "Qwen/Qwen3-ASR-0.6B",
            WhichModel::Qwen3ASR1_7B => "Qwen/Qwen3-ASR-1.7B",
            WhichModel::Qwen3VL2B => "Qwen/Qwen3-VL-2B-Instruct",
            WhichModel::Qwen3VL4B => "Qwen/Qwen3-VL-4B-Instruct",
            WhichModel::Qwen3VL8B => "Qwen/Qwen3-VL-8B-Instruct",
            WhichModel::Qwen3VL32B => "Qwen/Qwen3-VL-32B-Instruct",
            WhichModel::DeepSeekOCR => "deepseek-ai/DeepSeek-OCR",
            WhichModel::DeepSeekOCR2 => "deepseek-ai/DeepSeek-OCR-2",
            WhichModel::HunyuanOCR => "Tencent-Hunyuan/HunyuanOCR",
            WhichModel::PaddleOCRVL => "PaddlePaddle/PaddleOCR-VL",
            WhichModel::PaddleOCRVL1_5 => "PaddlePaddle/PaddleOCR-VL-1.5",
            WhichModel::RMBG2_0 => "AI-ModelScope/RMBG-2.0",
            WhichModel::VoxCPM => "OpenBMB/VoxCPM-0.5B",
            WhichModel::VoxCPM1_5 => "OpenBMB/VoxCPM1.5",
            WhichModel::GlmASRNano2512 => "ZhipuAI/GLM-ASR-Nano-2512",
            WhichModel::FunASRNano2512 => "FunAudioLLM/Fun-ASR-Nano-2512",
            WhichModel::GlmOCR => "ZhipuAI/GLM-OCR",
        }
    }

    /// Get the model type category for this model variant
    pub fn model_type(self) -> &'static str {
        match self {
            // LLM models
            WhichModel::MiniCPM4_0_5B
            | WhichModel::Qwen3_0_6B
            | WhichModel::LFM2_1_2B
            | WhichModel::LFM2_5_1_2BInstruct => "llm",
            WhichModel::Qwen2_5VL3B
            | WhichModel::Qwen2_5VL7B
            | WhichModel::Qwen3VL2B
            | WhichModel::Qwen3VL4B
            | WhichModel::Qwen3VL8B
            | WhichModel::Qwen3VL32B
            | WhichModel::Qwen3_5_0_8B
            | WhichModel::Qwen3_5_2B
            | WhichModel::Qwen3_5_4B
            | WhichModel::Qwen3_5_9B
            | WhichModel::Qwen3_5Gguf
            | WhichModel::LFM2_5VL1_6B
            | WhichModel::LFM2VL1_6B => "vlm",
            // OCR models
            WhichModel::DeepSeekOCR
            | WhichModel::DeepSeekOCR2
            | WhichModel::HunyuanOCR
            | WhichModel::GlmOCR
            | WhichModel::PaddleOCRVL
            | WhichModel::PaddleOCRVL1_5 => "ocr",
            // ASR models
            WhichModel::Qwen3ASR0_6B
            | WhichModel::Qwen3ASR1_7B
            | WhichModel::GlmASRNano2512
            | WhichModel::FunASRNano2512 => "asr",
            // Image models
            WhichModel::RMBG2_0 => "image",
            WhichModel::VoxCPM | WhichModel::VoxCPM1_5 => "tts",
        }
    }
}

pub trait GenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
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
    >;
}

pub enum ModelInstance<'a> {
    MiniCPM4(MiniCPMGenerateModel<'a>),
    Lfm2(Lfm2GenerateModel<'a>),
    Lfm2VL(Lfm2VLGenerateModel<'a>),
    Qwen2_5VL(Qwen2_5VLGenerateModel<'a>),
    Qwen3(Qwen3GenerateModel<'a>),
    Qwen3_5(Qwen3_5GenerateModel<'a>),
    Qwen3ASR(Qwen3AsrGenerateModel<'a>),
    Qwen3VL(Box<Qwen3VLGenerateModel<'a>>),
    DeepSeekOCR(DeepseekOCRGenerateModel),
    HunyuanOCR(HunyuanOCRGenerateModel<'a>),
    PaddleOCRVL(Box<PaddleOCRVLGenerateModel<'a>>),
    RMBG2_0(Box<RMBG2_0Model>),
    VoxCPM(Box<VoxCPMGenerate>),
    GlmASRNano(GlmAsrNanoGenerateModel<'a>),
    FunASRNano(FunAsrNanoGenerateModel),
    GlmOCR(GlmOcrGenerateModel),
}

impl<'a> GenerateModel for ModelInstance<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        match self {
            ModelInstance::MiniCPM4(model) => model.generate(mes),
            ModelInstance::Lfm2(model) => model.generate(mes),
            ModelInstance::Lfm2VL(model) => model.generate(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate(mes),
            ModelInstance::Qwen3(model) => model.generate(mes),
            ModelInstance::Qwen3_5(model) => model.generate(mes),
            ModelInstance::Qwen3ASR(model) => model.generate(mes),
            ModelInstance::Qwen3VL(model) => model.generate(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate(mes),
            ModelInstance::HunyuanOCR(model) => model.generate(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate(mes),
            ModelInstance::RMBG2_0(model) => model.generate(mes),
            ModelInstance::VoxCPM(model) => model.generate(mes),
            ModelInstance::GlmASRNano(model) => model.generate(mes),
            ModelInstance::FunASRNano(model) => model.generate(mes),
            ModelInstance::GlmOCR(model) => model.generate(mes),
        }
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
        match self {
            ModelInstance::MiniCPM4(model) => model.generate_stream(mes),
            ModelInstance::Lfm2(model) => model.generate_stream(mes),
            ModelInstance::Lfm2VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen3(model) => model.generate_stream(mes),
            ModelInstance::Qwen3_5(model) => model.generate_stream(mes),
            ModelInstance::Qwen3VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen3ASR(model) => model.generate_stream(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate_stream(mes),
            ModelInstance::HunyuanOCR(model) => model.generate_stream(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate_stream(mes),
            ModelInstance::RMBG2_0(model) => model.generate_stream(mes),
            ModelInstance::VoxCPM(model) => model.generate_stream(mes),
            ModelInstance::GlmASRNano(model) => model.generate_stream(mes),
            ModelInstance::FunASRNano(model) => model.generate_stream(mes),
            ModelInstance::GlmOCR(model) => model.generate_stream(mes),
        }
    }
}

pub fn load_model<'a>(
    model_type: WhichModel,
    path: &str,
    gguf: Option<&str>,
    mmproj: Option<&str>,
) -> Result<ModelInstance<'a>> {
    let model = match model_type {
        WhichModel::MiniCPM4_0_5B => {
            let model = MiniCPMGenerateModel::init(path, None, None)?;
            ModelInstance::MiniCPM4(model)
        }
        WhichModel::LFM2_1_2B => {
            let model = Lfm2GenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2(model)
        }
        WhichModel::LFM2_5_1_2BInstruct => {
            let model = Lfm2GenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2(model)
        }
        WhichModel::LFM2_5VL1_6B => {
            let model = Lfm2VLGenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2VL(model)
        }
        WhichModel::LFM2VL1_6B => {
            let model = Lfm2VLGenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2VL(model)
        }
        WhichModel::Qwen2_5VL3B => {
            let model = Qwen2_5VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen2_5VL(model)
        }
        WhichModel::Qwen2_5VL7B => {
            let model = Qwen2_5VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen2_5VL(model)
        }
        WhichModel::Qwen3_0_6B => {
            let model = Qwen3GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3(model)
        }
        WhichModel::Qwen3_5_0_8B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5_2B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5_4B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5_9B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5Gguf => {
            if gguf.is_none() {
                return Err(anyhow!("Qwen3_5Gguf gguf model path is required"));
            }
            let gguf = gguf.unwrap();
            let model = Qwen3_5GenerateModel::init_from_gguf(gguf, mmproj, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3ASR0_6B => {
            let model = Qwen3AsrGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3ASR(model)
        }
        WhichModel::Qwen3ASR1_7B => {
            let model = Qwen3AsrGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3ASR(model)
        }
        WhichModel::Qwen3VL2B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3VL4B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3VL8B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3VL32B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::DeepSeekOCR => {
            let model = DeepseekOCRGenerateModel::init(path, None, None)?;
            ModelInstance::DeepSeekOCR(model)
        }
        WhichModel::DeepSeekOCR2 => {
            let model = DeepseekOCRGenerateModel::init(path, None, None)?;
            ModelInstance::DeepSeekOCR(model)
        }
        WhichModel::HunyuanOCR => {
            let model = HunyuanOCRGenerateModel::init(path, None, None)?;
            ModelInstance::HunyuanOCR(model)
        }
        WhichModel::PaddleOCRVL => {
            let model = PaddleOCRVLGenerateModel::init(path, None, None)?;
            ModelInstance::PaddleOCRVL(Box::new(model))
        }
        WhichModel::PaddleOCRVL1_5 => {
            let model = PaddleOCRVLGenerateModel::init(path, None, None)?;
            ModelInstance::PaddleOCRVL(Box::new(model))
        }
        WhichModel::RMBG2_0 => {
            let model = RMBG2_0Model::init(path, None, None)?;
            ModelInstance::RMBG2_0(Box::new(model))
        }
        WhichModel::VoxCPM => {
            let model = VoxCPMGenerate::init(path, None, None)?;
            ModelInstance::VoxCPM(Box::new(model))
        }
        WhichModel::VoxCPM1_5 => {
            let model = VoxCPMGenerate::init(path, None, None)?;
            ModelInstance::VoxCPM(Box::new(model))
        }
        WhichModel::GlmASRNano2512 => {
            let model = GlmAsrNanoGenerateModel::init(path, None, None)?;
            ModelInstance::GlmASRNano(model)
        }
        WhichModel::FunASRNano2512 => {
            let model = FunAsrNanoGenerateModel::init(path, None, None)?;
            ModelInstance::FunASRNano(model)
        }
        WhichModel::GlmOCR => {
            let model = GlmOcrGenerateModel::init(path, None, None)?;
            ModelInstance::GlmOCR(model)
        }
    };
    Ok(model)
}
