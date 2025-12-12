use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{Activation, Conv2d, Dropout, LayerNorm, Module, VarBuilder};

use crate::models::common::{TwoLinearMLP, get_conv2d, get_layer_norm};

struct PatchEmbed {
    proj: Conv2d,
    norm: Option<LayerNorm>,
    patch_size: usize,
    embed_dim: usize,
}

impl PatchEmbed {
    pub fn new(
        vb: VarBuilder,
        in_chans: usize,
        embed_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        patch_norm: bool,
    ) -> Result<Self> {
        let patch_size = kernel_size;
        let proj = get_conv2d(
            vb.pp("proj"),
            in_chans,
            embed_dim,
            kernel_size,
            padding,
            stride,
            1,
            1,
            true,
        )?;
        let norm = if patch_norm {
            Some(get_layer_norm(vb.pp("norm"), 1e-5, embed_dim)?)
        } else {
            None
        };
        Ok(Self {
            patch_size,
            proj,
            norm,
            embed_dim,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, _, h, w) = xs.dims4()?;
        let mut xs = xs.clone();
        if w % self.patch_size != 0 {
            xs = xs.pad_with_zeros(3, 0, self.patch_size - w % self.patch_size)?;
        }
        if h % self.patch_size != 0 {
            xs = xs.pad_with_zeros(2, 0, self.patch_size - h % self.patch_size)?;
        }
        xs = self.proj.forward(&xs)?;
        if self.norm.is_some() {
            let (_, _, ph, pw) = xs.dims4()?;
            xs = xs.flatten(2, 3)?.transpose(1, 2)?;
            xs = self.norm.as_ref().unwrap().forward(&xs)?;
            xs = xs.transpose(1, 2)?.reshape((bs, self.embed_dim, ph, pw))?;
        }
        Ok(xs)
    }
}

pub struct WindowAttention {
    num_heads: usize,
    // head_dim: usize,
    qkv: Linear,
    proj: Linear,
    scaling: f64,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl WindowAttention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        window_size: usize, /* use_rel_pos: bool,
                             * input_size: Option<(usize, usize)>, */
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = if qkv_bias {
            linear(dim, dim * 3, vb.pp("qkv"))?
        } else {
            linear_no_bias(dim, dim * 3, vb.pp("qkv"))?
        };
        let proj = linear(dim, dim, vb.pp("proj"))?;
        let mut rel_pos_h = None;
        let mut rel_pos_w = None;
        if use_rel_pos {
            if input_size.is_none() {
                return Err(anyhow::anyhow!(
                    "Input size must be provided if using relative positional encoding."
                ));
            }
            let input_size = input_size.unwrap();
            let h_len = 2 * input_size.0 - 1;
            let w_len = 2 * input_size.1 - 1;
            rel_pos_h = Some(vb.get_with_hints((h_len, head_dim), "rel_pos_h", Init::Const(0.))?);
            rel_pos_w = Some(vb.get_with_hints((w_len, head_dim), "rel_pos_w", Init::Const(0.))?);
        }

        Ok(Self {
            num_heads,
            // head_dim,
            qkv,
            proj,
            scaling,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        })
    }

    fn get_rel_pos(&self, q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
        let max_rel_dist = 2 * std::cmp::max(q_size, k_size) - 1;
        let rel_pos_resized = if rel_pos.dim(0)? != max_rel_dist {
            let rel_pos_t = rel_pos
                .to_dtype(candle_core::DType::F32)?
                .t()?
                .unsqueeze(0)?
                .contiguous()?;
            let rel_pos_resized = interpolate_linear_1d(&rel_pos_t, max_rel_dist, None)?;
            rel_pos_resized
                .squeeze(0)?
                .t()?
                .contiguous()?
                .to_dtype(rel_pos.dtype())?
        } else {
            rel_pos.clone()
        };
        let q_coords = Tensor::arange(0 as f32, q_size as f32, rel_pos.device())?
            .unsqueeze(D::Minus1)?
            .affine((k_size as f64 / q_size as f64).max(1.0), 0.0)?;
        let k_coords = Tensor::arange(0 as f32, k_size as f32, rel_pos.device())?
            .unsqueeze(0)?
            .affine((q_size as f64 / k_size as f64).max(1.0), 0.0)?;
        let relative_coords = q_coords
            .broadcast_sub(&k_coords)?
            .affine(1.0, (k_size - 1) as f64)?
            .affine((q_size as f64 / k_size as f64).max(1.0), 0.0)?;
        let relative_coords = relative_coords
            .to_dtype(candle_core::DType::U32)?
            .contiguous()?;
        let rel_pos_resized = rel_pos_resized.contiguous()?;
        let res = index_select_2d(&rel_pos_resized, &relative_coords)?;
        Ok(res)
    }

    fn add_decomposed_rel_pos(
        &self,
        q: &Tensor,
        rel_pos_h: &Tensor,
        rel_pos_w: &Tensor,
        q_size: (usize, usize),
        k_size: (usize, usize),
    ) -> Result<(Tensor, Tensor)> {
        let (q_h, q_w) = q_size;
        let (k_h, k_w) = k_size;
        let rh = self.get_rel_pos(q_h, k_h, rel_pos_h)?; // (q_h, k_h, dim)
        let rw = self.get_rel_pos(q_w, k_w, rel_pos_w)?; // (q_w, k_w, dim)
        let (b, _, dim) = q.dims3()?;
        let r_q = q.reshape((b, q_h, q_w, dim))?.contiguous()?;
        let r_q_ = r_q.unsqueeze(D::Minus2)?; // (b, q_h, q_w, 1, dim)
        // rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        // rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        let rh_ = rh.unsqueeze(1)?.unsqueeze(0)?; // (1, h, 1, k, dim)
        let rel_h = r_q_.broadcast_mul(&rh_)?.sum(D::Minus1)?;
        let rw_ = rw.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, w, k, dim)
        let rel_w = r_q_.broadcast_mul(&rw_)?.sum(D::Minus1)?;
        let rel_h = rel_h
            .unsqueeze(D::Minus1)?
            .reshape((b, q_h * q_w, k_h, 1))?;
        let rel_w = rel_w
            .unsqueeze(D::Minus2)?
            .reshape((b, q_h * q_w, 1, k_w))?;
        Ok((rel_h, rel_w))
    }

    pub fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, h, w, _) = xs.dims4()?;
        // (3, B, n_head, h*w, head_dim)
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, h * w, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let xs = if self.use_rel_pos {
            let q_reshape = query_states.reshape((b * self.num_heads, h * w, ()))?;
            let (rel_h, rel_w) = self.add_decomposed_rel_pos(
                &q_reshape,
                self.rel_pos_h.as_ref().unwrap(),
                self.rel_pos_w.as_ref().unwrap(),
                (h, w),
                (h, w),
            )?;
            let (_, rel_h_dim1, rel_h_dim2, rel_h_dim3) = rel_h.dims4()?;
            let rel_h = rel_h.reshape((b, self.num_heads, rel_h_dim1, rel_h_dim2, rel_h_dim3))?;
            let (_, rel_w_dim1, rel_w_dim2, rel_w_dim3) = rel_w.dims4()?;
            let rel_w = rel_w.reshape((b, self.num_heads, rel_w_dim1, rel_w_dim2, rel_w_dim3))?;
            let attn_bias = rel_h.broadcast_add(&rel_w)?.reshape((
                b,
                self.num_heads,
                rel_h_dim1,
                rel_h_dim2 * rel_w_dim3,
            ))?;
            eager_attention_forward(
                &query_states,
                &key_states,
                &value_states,
                None,
                Some(&attn_bias),
                self.scaling,
            )?
        } else {
            eager_attention_forward(
                &query_states,
                &key_states,
                &value_states,
                None,
                None,
                self.scaling,
            )?
        };
        // (b, h*w, n_head, dim)
        let xs = xs.reshape((b, h * w, ()))?.reshape((b, h, w, ()))?;
        let xs = self.proj.forward(&xs)?;
        Ok(xs)
    }
}

struct SwinTransformerBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: TwoLinearMLP,
    window_size: usize,
    shift_size: usize,
}

impl SwinTransformerBlock {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        act: Activation,
        window_size: usize,
        shift_size: usize,
    ) -> Result<Self> {
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-5, dim)?;

        let attn = WindowAttention::new(vb.pp("attn"), dim, num_heads, qkv_bias, window_size)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-5, dim)?;
        let mlp_dim = (dim as f32 * mlp_ratio) as usize;
        let mlp = TwoLinearMLP::new(vb.pp("mlp"), dim, mlp_dim, act, true, "fc1", "fc2")?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
            shift_size,
        })
    }

    pub fn window_partition(&self, x: &Tensor, window_size: usize) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;

        let x = x.reshape((
            b,
            h / window_size,
            window_size,
            w / window_size,
            window_size,
            c,
        ))?;
        let windows = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((
            (),
            window_size,
            window_size,
            c,
        ))?;
        Ok(windows)
    }

    pub fn window_unpartition(
        &self,
        windows: &Tensor,
        window_size: usize,
        pad_hw: (usize, usize),
    ) -> Result<Tensor> {
        let (hp, wp) = pad_hw;
        let b = windows.dim(0)? / (hp * wp / window_size / window_size);
        let last_dim = windows.dim(D::Minus1)?;
        let x = windows.reshape(&[
            b,
            hp / window_size,
            wp / window_size,
            window_size,
            window_size,
            last_dim,
        ])?;
        let x = x
            .permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape((b, hp, wp, ()))?;
        Ok(x)
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask_matrix: Option<&Tensor>,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, c) = xs.dims3()?;
        let shortcut = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = xs.reshape((b, h, w, c))?;
        let pad_h = (self.window_size - h % self.window_size) % self.window_size;
        let pad_w = (self.window_size - w % self.window_size) % self.window_size;
        let xs = xs.pad_with_zeros(1, 0, pad_h)?;
        let xs = xs.pad_with_zeros(2, 0, pad_w)?;
        let (_, hp, wp, _) = xs.dims4()?;

        let (shifted_x, attn_mask) = if self.shift_size > 0 {
            (
                xs.roll(-(self.shift_size as i32), 1)?
                    .roll(-(self.shift_size as i32), 2)?,
                mask_matrix,
            )
        } else {
            (xs, None)
        };
        let xs = self.window_partition(&shifted_x, self.window_size)?;
        let xs = xs.reshape(((), self.window_size * self.window_size, c))?;
        let xs = self.attn.forward(&xs, attn_mask)?;
        let xs = self.window_unpartition(&xs, self.window_size, (hp, wp))?;
        let mut xs = if self.shift_size > 0 {
            xs.roll(self.shift_size as i32, 1)?
                .roll(self.shift_size as i32, 2)?
        } else {
            xs
        };
        if pad_h > 0 || pad_w > 0 {
            xs = xs.i((.., 0..h, 0..w, ..))?
        }
        let x = shortcut.add(&xs)?;
        let x = x.add(&self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

struct PatchMerging {}

struct BasicLayer {
    windows_size: usize,
    shift_size: usize,
    blocks: Vec<SwinTransformerBlock>,
    downsample: Option<PatchMerging>,
}
pub struct SwinTransformer {
    patch_embed: PatchEmbed,
    pos_drop: Dropout,
    layers: Vec<BasicLayer>,
    norms: Vec<LayerNorm>,
}
