use aha::utils::tensor_utils::{index_select_2d, interpolate_linear};
use anyhow::Result;
use candle_core::{IndexOp, Tensor};

#[test]
fn messy_test() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda messy_test -- --nocapture
    let device = &candle_core::Device::Cpu;
    let t1 = Tensor::rand(0.0, 1.0, (1, 5, 5, 10), device)?;
    let t2 = Tensor::rand(0.0, 1.0, (5, 8, 10), device)?;
    let t2 = t2.t()?;
    println!("t2: {:?}", t2);
    let re = t1.broadcast_matmul(&t2)?;
    println!("re: {:?}", re);
    // let index = Tensor::arange(0u32, 10u32, device)?;
    // let index_2d_vec = vec![index;5];
    // let index_2d = Tensor::stack(&index_2d_vec, 0)?;
    // println!("index_2d: {}", index_2d);
    // let t = Tensor::rand(0.0, 1.0, (20, 8), device)?;
    // println!("t: {}", t);
    // let res = index_select_2d(&t, &index_2d)?;
    // println!("res: {}", res);
    // let t = Tensor::arange(0.0, 10.0, device)?
    //     .unsqueeze(0)?
    //     .unsqueeze(0)?;
    // println!("t: {}", t);
    // let t_resized = interpolate_linear(&t, 20, None)?;
    // println!("t_resized: {}", t_resized);

    // let grid_thw = Tensor::new(vec![vec![3u32, 12, 20], vec![5, 30, 25]], device)?;
    // let cu_seqlens = grid_thw.i((.., 1))?.mul(&grid_thw.i((.., 2))?)?;
    // let grid_t = grid_thw.i((.., 0))?.to_vec1::<u32>()?;
    // println!("cu_seqlens: {}", cu_seqlens);
    // println!("cu_seqlens rank: {}", cu_seqlens.rank());
    // println!("grid_t: {:?}", grid_t);
    // let image_mask = Tensor::new(vec![0u32, 0, 0, 1, 0, 1], device)?;
    // let video_mask = Tensor::new(vec![0u32, 1, 0, 1, 0, 1], device)?;
    // let visual_mask = bitor_tensor(&image_mask, &video_mask)?;
    // println!("visual_mask: {}", visual_mask);
    // let x = Tensor::arange_step(0.0_f32, 5., 0.5, &device)?;
    // let x_int = x.to_dtype(candle_core::DType::U32)?;
    // println!("x: {}", x);
    // println!("x_int: {}", x_int);
    // let x_affine = x_int.affine(1.0, 1.0)?;
    // println!("x_affine: {}", x_affine);
    // let x_clamp = x_affine.clamp(0u32, 3u32)?;
    // println!("x_clamp: {}", x_clamp);
    // let wav_path = "./assets/audio/voice_01.wav";
    // let audio_tensor = load_audio_with_resample(wav_path, device, Some(16000))?;
    // println!("audio_tensor: {}", audio_tensor);
    // let string = "你好啊".to_string();
    // let vec_str: Vec<String>= string.chars().map(|c| c.to_string()).collect();
    // println!("vec_str: {:?}", vec_str);
    // let t = Tensor::rand(-1.0, 1.0, (2, 2), &device)?;
    // println!("t: {}", t);
    // let re_t = t.recip()?;
    // println!("re_t: {}", re_t);
    Ok(())
}
