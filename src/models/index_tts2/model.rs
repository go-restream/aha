use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{
    Conv1d, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding, linear, linear_b,
    ops::sigmoid, rms_norm,
};

use crate::{
    models::{
        common::{
            GateUpDownMLP, QKVCatAttention, TwoLinearMLP, WNConv1d, WNLinear, get_conv1d,
            get_layer_norm,
        },
        index_tts2::config::{DiTModelArgs, S2MelConfig},
    },
    position_embed::rope::RoPE,
    utils::tensor_utils::{pad_reflect_last_dim, split_tensor_with_size},
};
pub struct AdaptiveLayerNorm {
    project_layer: Linear,
    norm: RmsNorm,
    d_model: usize,
}

impl AdaptiveLayerNorm {
    pub fn new(vb: VarBuilder, d_model: usize, eps: f64) -> Result<Self> {
        let project_layer = linear(d_model, d_model * 2, vb.pp("project_layer"))?;
        let norm = rms_norm(d_model, eps, vb.pp("norm"))?;
        Ok(Self {
            project_layer,
            norm,
            d_model,
        })
    }

    pub fn forward(&self, xs: &Tensor, embedding: Option<&Tensor>) -> Result<Tensor> {
        if let Some(embedding) = embedding {
            let emb = self.project_layer.forward(embedding)?;
            let emb_split = split_tensor_with_size(&emb, 2, D::Minus1)?;
            let weight = &emb_split[0];
            let bias = &emb_split[1];
            Ok(self
                .norm
                .forward(xs)?
                .broadcast_mul(weight)?
                .broadcast_add(bias)?)
        } else {
            Ok(self.norm.forward(xs)?)
        }
    }
}

pub struct DiTTransformerBlock {
    attention: QKVCatAttention,
    feed_forward: GateUpDownMLP,
    ffn_norm: AdaptiveLayerNorm,
    attention_norm: AdaptiveLayerNorm,
    skip_in_linear: Option<Linear>,
    uvit_skip_connection: bool,
    time_as_token: bool,
}

impl DiTTransformerBlock {
    pub fn new(vb: VarBuilder, config: &DiTModelArgs) -> Result<Self> {
        let attention = QKVCatAttention::new(
            vb.pp("attention"),
            config.dim,
            config.n_head,
            Some(config.head_dim),
            false,
            Some("wqkv"),
            Some("wo"),
        )?;

        let feed_forward = GateUpDownMLP::new(
            vb.pp("feed_forward"),
            config.dim,
            config.intermediate_size,
            candle_nn::Activation::Silu,
            false,
            Some("w1"),
            Some("w3"),
            Some("w2"),
        )?;
        let ffn_norm = AdaptiveLayerNorm::new(vb.pp("ffn_norm"), config.dim, config.norm_eps)?;
        let attention_norm =
            AdaptiveLayerNorm::new(vb.pp("attention_norm"), config.dim, config.norm_eps)?;
        let (skip_in_linear, uvit_skip_connection) = if config.uvit_skip_connection {
            let skip_in_linear = linear(config.dim * 2, config.dim, vb.pp("skip_in_linear"))?;
            (Some(skip_in_linear), config.uvit_skip_connection)
        } else {
            (None, false)
        };
        Ok(Self {
            attention,
            feed_forward,
            ffn_norm,
            attention_norm,
            skip_in_linear,
            uvit_skip_connection,
            time_as_token: config.time_as_token,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        c: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        mask: Option<&Tensor>,
        skip_in_x: Option<&Tensor>,
    ) -> Result<Tensor> {
        let c = if self.time_as_token { None } else { Some(c) };
        let mut xs = xs.clone();
        if self.uvit_skip_connection
            && let Some(skip_in_x) = skip_in_x
            && let Some(skip_in_linear) = &self.skip_in_linear
        {
            let cat = Tensor::cat(&[&xs, skip_in_x], D::Minus1)?;
            xs = skip_in_linear.forward(&cat)?;
        }
        let xs = self
            .attention
            .forward(
                &self.attention_norm.forward(&xs, c)?,
                cos,
                sin,
                mask,
                false,
                true,
            )?
            .add(&xs)?;
        let out = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&xs, c)?)?
            .add(&xs)?;
        Ok(out)
    }
}

pub struct DiTTransformer {
    layers: Vec<DiTTransformerBlock>,
    norm: AdaptiveLayerNorm,
    rope: RoPE,
    uvit_skip_connection: bool,
    layers_emit_skip: Vec<usize>,
    layers_receive_skip: Vec<usize>,
}

impl DiTTransformer {
    pub fn new(vb: VarBuilder, config: &DiTModelArgs) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = vec![];
        for i in 0..config.n_layer {
            let layer = DiTTransformerBlock::new(vb_layers.pp(i), config)?;
            layers.push(layer);
        }
        let norm = AdaptiveLayerNorm::new(vb.pp("norm"), config.dim, config.norm_eps)?;
        let rope = RoPE::new(config.dim, 10000.0, vb.device())?;
        let mut layers_emit_skip: Vec<usize> = vec![];
        let mut layers_receive_skip: Vec<usize> = vec![];
        if config.uvit_skip_connection {
            layers_emit_skip = (0..config.n_layer)
                .filter(|&x| x < config.n_layer / 2)
                .collect();
            layers_receive_skip = (0..config.n_layer)
                .filter(|&x| x > config.n_layer / 2)
                .collect();
        }
        Ok(Self {
            layers,
            norm,
            rope,
            uvit_skip_connection: config.uvit_skip_connection,
            layers_emit_skip,
            layers_receive_skip,
        })
    }
    pub fn forward(&self, xs: &Tensor, c: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let (cos, sin) = self.rope.forward(0, seq_len, xs.device())?;
        let mut skip_in_x_list = vec![];
        let mut xs = xs.clone();
        for (i, layer) in (&self.layers).iter().enumerate() {
            let skip_in_x = if self.uvit_skip_connection && self.layers_receive_skip.contains(&i) {
                skip_in_x_list.pop()
            } else {
                None
            };
            xs = layer.forward(&xs, c, Some(&cos), Some(&sin), mask, skip_in_x.as_ref())?;
            if self.uvit_skip_connection && self.layers_emit_skip.contains(&i) {
                skip_in_x_list.push(xs.clone());
            }
        }
        xs = self.norm.forward(&xs, Some(c))?;
        Ok(xs)
    }
}

pub struct TimestepEmbedder {
    mlp: TwoLinearMLP,
    freqs: Tensor,
    scale: f64,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        frequency_embedding_size: usize,
    ) -> Result<Self> {
        let mlp = TwoLinearMLP::new(
            vb.pp("mlp"),
            frequency_embedding_size,
            hidden_size,
            hidden_size,
            candle_nn::Activation::Silu,
            true,
            "0",
            "1",
        )?;
        let scale = 1000.0;
        let half = frequency_embedding_size / 2;
        let freqs = Tensor::arange(0f32, half as f32, vb.device())?
            .affine(-(10000.0f64.ln()), 0.0)?
            .exp()?;
        Ok(Self {
            mlp,
            freqs,
            scale,
            frequency_embedding_size,
        })
    }
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let args = t
            .affine(self.scale, 0.0)?
            .unsqueeze(D::Minus1)?
            .broadcast_matmul(&self.freqs.unsqueeze(0)?)?;
        let mut embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        if self.frequency_embedding_size % 2 > 0 {
            embedding = embedding.pad_with_zeros(D::Minus1, 0, 1)?;
        }
        embedding = self.mlp.forward(&embedding)?;
        Ok(embedding)
    }
}

pub struct SConv1d {
    conv: WNConv1d,
    ks: usize,
    stride: usize,
    dilation: usize,
}

impl SConv1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv = WNConv1d::new(
            vb.pp("conv.conv"),
            in_c,
            out_c,
            ks,
            dilation,
            0,
            groups,
            stride,
            bias,
        )?;
        Ok(Self {
            conv,
            ks,
            stride,
            dilation,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let length = xs.dim(D::Minus1)?;
        let ks = (self.ks - 1) * self.dilation + 1;
        let padding_total = ks - self.stride;
        let n_frames = (length - ks + padding_total) as f32 / self.stride as f32 + 1.0;
        let idea_length = (n_frames.ceil() as usize - 1) * self.stride + (ks - padding_total);
        let extra_padding = idea_length - length;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let xs = pad_reflect_last_dim(xs, (padding_left, padding_right + extra_padding))?;
        let xs = self.conv.forward(&xs)?;
        Ok(xs)
    }
}

pub struct Wavenet {
    cond_layer: Option<SConv1d>,
    in_layers: Vec<SConv1d>,
    res_skip_layers: Vec<SConv1d>,
    hidden_c: usize,
    n_layers: usize,
}

impl Wavenet {
    pub fn new(
        vb: VarBuilder,
        hidden_c: usize,
        ks: usize,
        dilation_rate: usize,
        n_layers: usize,
        gin_channels: usize,
    ) -> Result<Self> {
        let cond_layer = if gin_channels != 0 {
            Some(SConv1d::new(
                vb.pp("cond_layer"),
                gin_channels,
                2 * hidden_c * n_layers,
                1,
                1,
                1,
                1,
                true,
            )?)
        } else {
            None
        };
        let mut in_layers = vec![];
        let vb_layers = vb.pp("in_layers");
        let mut res_skip_layers = vec![];
        let vb_res_skip_layers = vb.pp("res_skip_layers");
        for i in 0..n_layers {
            let dilation = dilation_rate.pow(i as u32);
            let in_layer = SConv1d::new(
                vb_layers.pp(i),
                hidden_c,
                1 * hidden_c,
                ks,
                1,
                dilation,
                1,
                true,
            )?;
            in_layers.push(in_layer);
            let res_skip_c = if i < n_layers - 1 {
                2 * hidden_c
            } else {
                hidden_c
            };
            let res_skip_layer = SConv1d::new(
                vb_res_skip_layers.pp(i),
                hidden_c,
                res_skip_c,
                1,
                1,
                1,
                1,
                true,
            )?;
            res_skip_layers.push(res_skip_layer);
        }
        Ok(Self {
            cond_layer,
            in_layers,
            res_skip_layers,
            hidden_c,
            n_layers,
        })
    }

    pub fn fused_add_tanh_sigmoid_multiply(
        &self,
        input_a: &Tensor,
        input_b: &Tensor,
    ) -> Result<Tensor> {
        let in_act = input_a.add(&input_b)?;
        let parts = split_tensor_with_size(&in_act, 2, 1)?;
        let t_act = (&parts[0]).tanh()?;
        let s_act = sigmoid(&parts[1])?;
        let acts = t_act.mul(&s_act)?;
        Ok(acts)
    }

    pub fn forward(self, xs: &Tensor, x_mask: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut output = xs.zeros_like()?;
        let g = if let Some(g) = g
            && let Some(cond_layer) = &self.cond_layer
        {
            Some(cond_layer.forward(g)?)
        } else {
            None
        };
        let mut xs = xs.clone();
        for i in 0..self.n_layers {
            let xs_in = &self.in_layers[i].forward(&xs)?;
            let g_l = if let Some(g) = &g {
                let cond_offset = i * 2 * self.hidden_c;
                g.narrow(1, cond_offset, 2 * self.hidden_c)?
            } else {
                xs_in.zeros_like()?
            };
            let acts = self.fused_add_tanh_sigmoid_multiply(&xs_in, &g_l)?;
            let res_skip_act = &self.res_skip_layers[i].forward(&acts)?;
            if i < self.n_layers - 1 {
                let res_acts = res_skip_act.narrow(1, 0, self.hidden_c)?;
                let out_acts = res_skip_act.narrow(1, self.hidden_c, self.hidden_c)?;
                xs = xs.add(&res_acts)?.mul(x_mask)?;
                output = output.add(&out_acts)?;
            } else {
                output = output.add(&res_skip_act)?;
            }
        }
        output = output.mul(x_mask)?;
        Ok(output)
    }
}

pub struct FinalLayer {
    norm_final: LayerNorm,
    linear: WNLinear,
    ada_ln_modulation: Linear, // (silu+linear)
}

impl FinalLayer {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        patch_size: usize,
        out_c: usize,
    ) -> Result<Self> {
        let norm_final = get_layer_norm(vb.pp("norm_final"), 1e-6, hidden_size)?;
        let linear = WNLinear::new(
            vb.pp("linear"),
            hidden_size,
            patch_size * patch_size * out_c,
            true,
        )?;
        let ada_ln_modulation = linear_b(
            hidden_size,
            2 * hidden_size,
            true,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    pub fn forward(&self, xs: &Tensor, c: &Tensor) -> Result<Tensor> {
        let linear_c = self.ada_ln_modulation.forward(c)?.chunk(2, 1)?;
        let xs = self.norm_final.forward(xs)?;
        let xs = linear_c[1]
            .unsqueeze(1)?
            .affine(1.0, 1.0)?
            .broadcast_mul(&xs)?
            .add(&linear_c[0].unsqueeze(1)?)?;
        let xs = self.linear.forward(&xs)?;
        Ok(xs)
    }
}

pub struct DiT {
    transformer: DiTTransformer,
    x_embedder: WNLinear,
    cond_embedder: Embedding,
    cond_projection: Linear,
    t_embedder: TimestepEmbedder,
    input_pos: Tensor,
    t_embedder2: TimestepEmbedder,
    conv1: Linear,
    conv2: Conv1d,
    wavenet: Wavenet,
    final_layer: FinalLayer,
    res_projection: Linear,
    content_mask_embedder: Embedding,
    skip_linear: Linear,
    cond_x_merge_linear: Linear,
    style_in: Option<Linear>,
    time_as_token: bool,
    style_as_token: bool,
    uvit_skip_connection: bool,
}

impl DiT {
    pub fn new(vb: VarBuilder, config: &S2MelConfig) -> Result<Self> {
        let time_as_token = config.di_t.time_as_token;
        let style_as_token = config.di_t.style_as_token;
        let uvit_skip_connection = config.di_t.uvit_skip_connection;
        let transformer_config = DiTModelArgs::new_from_dit_config(&config.di_t);
        let transformer = DiTTransformer::new(vb.pp("transformer"), &transformer_config)?;
        let x_embedder = WNLinear::new(
            vb.pp("x_embedder"),
            config.di_t.in_channels,
            config.di_t.hidden_dim,
            true,
        )?;
        let cond_embedder = embedding(
            config.di_t.content_codebook_size,
            config.di_t.hidden_dim,
            vb.pp("cond_embedder"),
        )?;
        let cond_projection = linear_b(
            config.di_t.content_dim,
            config.di_t.hidden_dim,
            true,
            vb.pp("cond_projection"),
        )?;
        let t_embedder = TimestepEmbedder::new(vb.pp("t_embedder"), config.di_t.hidden_dim, 256)?;
        let input_pos = Tensor::arange(0u32, 16384, vb.device())?;
        let t_embedder2 =
            TimestepEmbedder::new(vb.pp("t_embedder2"), config.wavenet.hidden_dim, 256)?;
        let conv1 = linear_b(
            config.di_t.hidden_dim,
            config.wavenet.hidden_dim,
            true,
            vb.pp("conv1"),
        )?;
        let conv2 = get_conv1d(
            vb.pp("conv2"),
            config.wavenet.hidden_dim,
            config.di_t.in_channels,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let wavenet = Wavenet::new(
            vb.pp("wavenet"),
            config.wavenet.hidden_dim,
            config.wavenet.kernel_size,
            config.wavenet.dilation_rate,
            config.wavenet.num_layers,
            config.wavenet.hidden_dim,
        )?;
        let final_layer = FinalLayer::new(
            vb.pp("final_layer"),
            config.wavenet.hidden_dim,
            1,
            config.wavenet.hidden_dim,
        )?;
        let res_projection = linear(
            config.di_t.hidden_dim,
            config.wavenet.hidden_dim,
            vb.pp("res_projection"),
        )?;
        let content_mask_embedder =
            embedding(1, config.di_t.hidden_dim, vb.pp("content_mask_embedder"))?;
        let skip_linear = linear(
            config.di_t.hidden_dim + config.di_t.in_channels,
            config.di_t.hidden_dim,
            vb.pp("skip_linear"),
        )?;
        let in_dim = if config.di_t.style_condition && !config.di_t.style_as_token {
            config.di_t.hidden_dim + config.di_t.in_channels * 2 + config.style_encoder.dim
        } else {
            config.di_t.hidden_dim + config.di_t.in_channels * 2
        };
        let cond_x_merge_linear =
            linear(in_dim, config.di_t.hidden_dim, vb.pp("cond_x_merge_linear"))?;
        let style_in = if config.di_t.style_as_token {
            Some(linear(
                config.style_encoder.dim,
                config.di_t.hidden_dim,
                vb.pp("style_in"),
            )?)
        } else {
            None
        };
        Ok(Self {
            transformer,
            x_embedder,
            cond_embedder,
            cond_projection,
            t_embedder,
            input_pos,
            t_embedder2,
            conv1,
            conv2,
            wavenet,
            final_layer,
            res_projection,
            content_mask_embedder,
            skip_linear,
            cond_x_merge_linear,
            style_in,
            time_as_token,
            style_as_token,
            uvit_skip_connection,
        })
    }
}

pub struct CFM {
    estimator: DiT,
}

pub struct MyModel {
    cfm: CFM,
}

pub struct IndexTTS2 {
    cache_spk_cond: Option<Tensor>,
    cache_s2mel_style: Option<Tensor>,
    cache_s2mel_prompt: Option<Tensor>,
    cache_spk_audio_prompt: Option<String>,
    cache_emo_cond: Option<Tensor>,
    cache_emo_audio_prompt: Option<Tensor>,
    cache_mel: Option<Tensor>,
}
