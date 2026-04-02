use crate::{
    params::{
        chat::{
            AudioUrlType, ChatCompletionChoice, ChatCompletionChunkChoice,
            ChatCompletionChunkResponse, ChatCompletionResponse, ChatMessage,
            ChatMessageAudioContentPart, ChatMessageContent, ChatMessageContentPart,
            ChatMessageImageContentPart, DeltaChatMessage, DeltaFunction, DeltaToolCall, Function,
            ImageUrlType, ToolCall,
        },
        shared::{FinishReason, Usage},
    },
    utils::timestamp,
};

pub fn build_img_completion_response(
    base64vec: &Vec<String>,
    model_name: &str,
) -> ChatCompletionResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        // created: chrono::Utc::now().timestamp() as u32,
        created: timestamp() as u32,
        model: model_name.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage: None,
    };
    let mut conten_part_vec = vec![];
    for img_bas64 in base64vec {
        let img_base64_prefix = "data:image/png;base64,".to_string() + img_bas64;
        let part = ChatMessageContentPart::Image(ChatMessageImageContentPart {
            r#type: "image".to_string(),
            image_url: ImageUrlType {
                url: img_base64_prefix,
                detail: None,
            },
        });
        conten_part_vec.push(part);
    }
    let choice = ChatCompletionChoice {
        index: 0,
        message: ChatMessage::Assistant {
            content: Some(ChatMessageContent::ContentPart(conten_part_vec)),
            reasoning_content: None,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: None,
        },
        finish_reason: Some(FinishReason::StopSequenceReached),
        logprobs: None,
    };
    response.choices.push(choice);
    response
}

pub fn build_audio_completion_response(
    base64_audio: &String,
    model_name: &str,
) -> ChatCompletionResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        created: timestamp() as u32,
        model: model_name.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage: None,
    };

    let base64_audio = format!("data:audio/wav;base64,{}", base64_audio);
    let conten_part_vec = vec![ChatMessageContentPart::Audio(ChatMessageAudioContentPart {
        r#type: "audio".to_string(),
        audio_url: AudioUrlType {
            url: base64_audio.to_string(),
        },
    })];
    let choice = ChatCompletionChoice {
        index: 0,
        message: ChatMessage::Assistant {
            content: Some(ChatMessageContent::ContentPart(conten_part_vec)),
            reasoning_content: None,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: None,
        },
        finish_reason: Some(FinishReason::StopSequenceReached),
        logprobs: None,
    };
    response.choices.push(choice);
    response
}

/// Builds a chat completion response from the model's output string.
///
/// This function handles two special formatting patterns in the input:
/// 1. Tool call formatting using the delimiter "<tool_call>" where the content before the first
///    delimiter is treated as the main content, and subsequent parts (separated by "</tool_call>")
///    are parsed as JSON tool call definitions.
/// 2. Reasoning content formatting where content between <think> and </think> tags is
///    extracted as reasoning_content, and the content after </think> becomes the main content.
///
/// # Arguments
/// * `res` - The raw response string from the model that may contain special formatting
/// * `model_name` - Name of the model generating the response  
/// * `usage` - Optional usage statistics to include in the response
///
/// # Returns
/// A [ChatCompletionResponse](aha/src/params/chat.rs#L11-L32) with processed content, tool calls, and/or reasoning content
///
/// # Format specifications
/// - Tool call format: Content followed by "<tool_call>" and then JSON-formatted tool call data separated by "</tool_call>"
/// - Reasoning format: Content wrapped in <think> and </think> tags followed by actual response
fn build_response(res: String, model_name: &str, usage: Option<Usage>) -> ChatCompletionResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        created: timestamp() as u32,
        model: model_name.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage,
    };
    let (content, tool_calls) = if res.contains("<tool_call>") {
        let mes: Vec<&str> = res.split("<tool_call>").collect();
        let content = mes[0].to_string();
        let mut tool_vec = Vec::new();
        for (i, m) in mes.iter().enumerate().skip(1) {
            let tool_mes = m.replace("</tool_call>", "");
            let function = match serde_json::from_str::<serde_json::Value>(&tool_mes) {
                Ok(json_value) => {
                    let name = json_value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default();

                    let arguments = json_value
                        .get("arguments")
                        .map(|v| v.to_string())
                        .unwrap_or_default();

                    Function { name, arguments }
                }
                Err(_) => Function {
                    name: "".to_string(),
                    arguments: "".to_string(),
                },
            };
            let tool_call = ToolCall {
                id: (i - 1).to_string(),
                r#type: "function".to_string(),
                function,
            };
            tool_vec.push(tool_call);
        }
        (content, Some(tool_vec))
    } else {
        (res, None)
    };
    let (content, reasoning_content) = if content.contains("</think>") {
        let contents: Vec<&str> = content.split("</think>").collect();
        let reasoning_content = contents[0].to_string().replace("<think>", "");
        let content = contents[1].to_string();
        (content, Some(reasoning_content))
    } else {
        (content, None)
    };
    let finish_reason = if tool_calls.is_some() {
        Some(FinishReason::ToolCalls)
    } else {
        Some(FinishReason::StopSequenceReached)
    };
    let choice = ChatCompletionChoice {
        index: 0,
        message: ChatMessage::Assistant {
            content: Some(ChatMessageContent::Text(content)),
            reasoning_content,
            refusal: None,
            name: None,
            audio: None,
            tool_calls,
        },
        finish_reason,
        logprobs: None,
    };
    response.choices.push(choice);
    response
}

pub fn build_completion_response(
    res: String,
    model_name: &str,
    completion_tokens: Option<u32>,
    prompt_tokens: Option<u32>,
) -> ChatCompletionResponse {
    let usage = if prompt_tokens.is_none() && completion_tokens.is_none() {
        None
    } else {
        Some(Usage {
            prompt_tokens,
            prompt_secs: None,
            completion_tokens,
            completion_secs: None,
            completion_per_token_secs: None,
            completion_tps: None,
            total_tokens: prompt_tokens.unwrap_or(0) + completion_tokens.unwrap_or(0),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        })
    };

    build_response(res, model_name, usage)
}

pub fn build_completion_response_with_time(
    res: String,
    model_name: &str,
    completion_tokens: Option<u32>,
    completion_secs: Option<f64>,
    prompt_tokens: Option<u32>,
    prompt_secs: Option<f64>,
) -> ChatCompletionResponse {
    let usage = if prompt_tokens.is_none() && completion_tokens.is_none() {
        None
    } else {
        let (completion_per_token_secs, completion_tps) = if let Some(completion_tokens) =
            completion_tokens
            && let Some(completion_secs) = completion_secs
        {
            let per_token_secs = completion_secs / completion_tokens as f64;
            let tps = completion_tokens as f64 / completion_secs;
            (Some(per_token_secs), Some(tps))
        } else {
            (None, None)
        };
        Some(Usage {
            prompt_tokens,
            prompt_secs,
            completion_tokens,
            completion_secs,
            completion_per_token_secs,
            completion_tps,
            total_tokens: prompt_tokens.unwrap_or(0) + completion_tokens.unwrap_or(0),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        })
    };

    build_response(res, model_name, usage)
}

pub fn build_chunk_response_with_usage(
    model_name: &str,
    completion_tokens: Option<u32>,
    completion_secs: Option<f64>,
    prompt_tokens: Option<u32>,
    prompt_secs: Option<f64>,
) -> ChatCompletionChunkResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let usage = if prompt_tokens.is_none() && completion_tokens.is_none() {
        None
    } else {
        let (completion_per_token_secs, completion_tps) = if let Some(completion_tokens) =
            completion_tokens
            && let Some(completion_secs) = completion_secs
        {
            let per_token_secs = completion_secs / completion_tokens as f64;
            let tps = if (completion_secs - 0.0).abs() > 0.0001 {
                completion_tokens as f64 / completion_secs
            } else {
                0.0
            };
            (Some(per_token_secs), Some(tps))
        } else {
            (None, None)
        };
        Some(Usage {
            prompt_tokens,
            prompt_secs,
            completion_tokens,
            completion_secs,
            completion_per_token_secs,
            completion_tps,
            total_tokens: prompt_tokens.unwrap_or(0) + completion_tokens.unwrap_or(0),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        })
    };
    let mut response = ChatCompletionChunkResponse {
        id: Some(id),
        choices: vec![],
        created: timestamp() as u32,
        model: model_name.to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage,
    };
    let choice = ChatCompletionChunkChoice {
        index: Some(0),
        delta: DeltaChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            refusal: None,
            name: None,
            tool_calls: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    response.choices.push(choice);
    response
}

pub fn build_chunk_response_with_reasoning(
    reasoning: String,
    model_name: &str,
) -> ChatCompletionChunkResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionChunkResponse {
        id: Some(id),
        choices: vec![],
        created: timestamp() as u32,
        model: model_name.to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };
    let choice = ChatCompletionChunkChoice {
        index: Some(0),
        delta: DeltaChatMessage::Assistant {
            content: None,
            reasoning_content: Some(reasoning),
            refusal: None,
            name: None,
            tool_calls: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    response.choices.push(choice);
    response
}

pub fn build_completion_chunk_response(
    res: String,
    model_name: &str,
    tool_call_id: Option<String>,
    tool_call_content: Option<String>,
) -> ChatCompletionChunkResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionChunkResponse {
        id: Some(id),
        choices: vec![],
        created: timestamp() as u32,
        model: model_name.to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };
    let choice = if let Some(tool_call_id) = tool_call_id {
        let function = if let Some(content) = tool_call_content {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(json_value) => {
                    let name = json_value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    let arguments = json_value.get("arguments").map(|v| v.to_string());

                    DeltaFunction { name, arguments }
                }
                Err(_) => DeltaFunction {
                    name: None,
                    arguments: Some(content),
                },
            }
        } else {
            DeltaFunction {
                name: None,
                arguments: None,
            }
        };
        ChatCompletionChunkChoice {
            index: Some(0),
            delta: DeltaChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                refusal: None,
                name: None,
                tool_calls: Some(vec![DeltaToolCall {
                    index: Some(0),
                    id: Some(tool_call_id),
                    r#type: Some("function".to_string()),
                    function,
                }]),
            },
            finish_reason: None,
            logprobs: None,
        }
    } else {
        ChatCompletionChunkChoice {
            index: Some(0),
            delta: DeltaChatMessage::Assistant {
                content: Some(ChatMessageContent::Text(res)),
                reasoning_content: None,
                refusal: None,
                name: None,
                tool_calls: None,
            },
            finish_reason: None,
            logprobs: None,
        }
    };
    response.choices.push(choice);
    response
}
