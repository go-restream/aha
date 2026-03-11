use aha::{chat::ChatCompletionParameters, gguf_models::qwen3_5::GgufQwen3_5};
use anyhow::Result;
use candle_core::{Device, quantized::gguf_file};
#[test]
fn gguf_test() -> Result<()> {
    // cargo test -r -F cuda --test test_gguf_qwen3_5 gguf_test -- --nocapture
    let path = "/home/jhq/.aha/Qwen/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf";
    let mut file = std::fs::File::open(path)?;
    let model = gguf_file::Content::read(&mut file)?;
    let device = Device::new_cuda(0)?;
    // println!("model: {:?}", model.magic);
    // println!("generat.type: {:#?}", model.metadata.keys()); 
    // println!("tokenizer.ggml.model: {:#?}", model.metadata.get("tokenizer.ggml.model"));  // gpt2
    // // println!("model: {:?}", model.tensor_infos);

    let message = r#"
    {
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": [        
                    {
                        "type": "text", 
                        "text": "你好啊"
                    }
                ]
            }
        ]
    }
    "#;
// render: <|im_start|>user
// 你好啊<|im_end|>
// <|im_start|>assistant
// <think>

// </think>


// input_ids: [[248045,    846,    198, 109266,  98710, 248046,    198, 248045,  74455,    198,
//   248068,    271, 248069,    271]]
// Tensor[[1, 14], u32, cuda:0]
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let mut gguf_qwen3_5 = GgufQwen3_5::new(&path, None)?;
    let _ = gguf_qwen3_5.generate(mes)?;
    Ok(())
}