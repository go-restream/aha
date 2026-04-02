use std::time::Instant;

use aha::models::{GenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;

#[test]
fn robo_brain_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda robo_brain_generate -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/BAAI/RoboBrain2.0-3B/", save_dir);

    let message = r#"
    {
        "model": "qwen2.5vl",
        "messages": [
            {
                "role": "user",
                "content": [           
                    {
                        "type": "text", 
                        "text": "hello RoboBrain"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = Qwen2_5VLGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let res = model.generate(mes)?;
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        println!("usage: \n {:?}", usage);
    }
    Ok(())
}
