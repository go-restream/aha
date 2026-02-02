use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{Device, Tensor};
use serde_json::{Value, json};

use crate::utils::{
    audio_utils::{extract_audios, split_audio_into_chunks},
    capitalize_first_letter, extract_user_text_vec,
    tensor_utils::float_range_normalize,
};

pub struct Qwen3AsrProcessor {
    device: Device,
    sample_rate: usize,
    support_language: Vec<String>,
    max_asr_input_seconds: f32,
}

impl Qwen3AsrProcessor {
    pub fn new(device: &Device) -> Result<Self> {
        let support_language: Vec<String> = vec![
            "Chinese",
            "English",
            "Cantonese",
            "Arabic",
            "German",
            "French",
            "Spanish",
            "Portuguese",
            "Indonesian",
            "Italian",
            "Korean",
            "Russian",
            "Thai",
            "Vietnamese",
            "Japanese",
            "Turkish",
            "Hindi",
            "Malay",
            "Dutch",
            "Swedish",
            "Danish",
            "Finnish",
            "Polish",
            "Czech",
            "Filipino",
            "Persian",
            "Greek",
            "Romanian",
            "Hungarian",
            "Macedonian",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Ok(Self {
            device: device.clone(),
            sample_rate: 16000,
            support_language,
            max_asr_input_seconds: 1200.0,
        })
    }

    pub fn process_audio(&self, mes: &ChatCompletionParameters) -> Result<Vec<Tensor>> {
        let audio_tensors = extract_audios(mes, &self.device, Some(self.sample_rate))?;
        audio_tensors
            .iter()
            .map(|audio| float_range_normalize(&audio))
            .collect()
    }

    pub fn validate_language(&self, lang: &String) -> bool {
        self.support_language.contains(lang)
    }

    pub fn process_info(&self, mes: &ChatCompletionParameters, render: &str) -> Result<()> {
        let audio_count = render
            .matches("<|audio_start|><|audio_pad|><|audio_end|>")
            .count();
        let mut render = if audio_count > 1 {
            render.replace(
                &"<|audio_start|><|audio_pad|><|audio_end|>".repeat(audio_count),
                "<|audio_start|><|audio_pad|><|audio_end|>",
            )
        } else {
            render.to_string()
        };
        if let Some(map) = &mes.metadata
            && map.contains_key("language")
        {
            let lang = map.get("language").unwrap();
            let lang = capitalize_first_letter(lang);
            if self.validate_language(&lang) {
                render = format!("{}language {}'<asr_text>'", render, lang);
            }
        }
        let audio_tensors = self.process_audio(mes)?;
        let audio_len = audio_tensors.len();
        if audio_len != audio_count {
            return Err(anyhow::anyhow!("audio_pad num != audio num"));
        }
        let mut split_wavs = vec![];
        for wav in audio_tensors.iter() {
            let wavs = split_audio_into_chunks(wav, self.sample_rate, self.max_asr_input_seconds)?;
            split_wavs.extend_from_slice(&wavs);
        }
        

        // let mut audio_datas = vec![];
        // for (i, wav) in audio_tensors.iter().enumerate() {
        //     let wavs = split_audio_into_chunks(wav, self.sample_rate, self.max_asr_input_seconds)?;
        //     for i_w in wavs {
        //         let audio_data = AudioData {
        //             wav: i_w,
        //             language: langs[i].clone(),
        //         };
        //         audio_datas.push(audio_data);
        //     }
        // }
        Ok(())
    }
}

pub struct AudioData {
    pub wav: Tensor,
    pub language: Option<String>,
}
