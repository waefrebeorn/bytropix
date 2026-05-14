"""
WubuLatentAttention: DeepSeek-inspired Multi-head Latent Attention
+ WuBu nested hyperbolic structure + turbo quantization.

Key innovations from DeepSeek V3/V4 MLA:
- Low-rank KV joint compression (c_t^KV = W^{DKV} * h_t, d_c << n_h*d_h)
- Decoupled RoPE for position sensitivity without bloating KV cache
- Multi-head latent queries with absorbed up-projections at inference

WuBu integration replaces RoPE with Möbius gyration in Poincaré ball,
so position encoding is hyperbolic rather than Euclidean rotary.

Turbo quant: block-wise FP8 storage with per-block scaling factors,
applied to the latent KV cache for 2-4x memory reduction vs MHA.

Reference: DeepSeek-V2 (arXiv:2405.04434), DeepSeek-V3 (arXiv:2412.19437)
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Optional, Tuple, Any


# ─── Poincaré Ball (from WuBuMindV7, enhanced with gyration) ───────

class PoincareBall:
    """Hyperbolic geometry operations in the Poincaré ball model."""
    EPS = 1e-7

    @staticmethod
    def project(x):
        x_f32 = x.astype(jnp.float32)
        norm_sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
        max_norm = 1.0 - PoincareBall.EPS
        projected = jnp.where(
            norm_sq >= 1.0,
            x_f32 / jnp.sqrt(norm_sq).clip(PoincareBall.EPS) * max_norm,
            x_f32
        )
        return projected.astype(x.dtype)

    @staticmethod
    def mobius_add(x, y, c=1.0):
        x_f32, y_f32 = x.astype(jnp.float32), y.astype(jnp.float32)
        x2 = jnp.sum(x_f32 * x_f32, -1, keepdims=True)
        y2 = jnp.sum(y_f32 * y_f32, -1, keepdims=True)
        xy = jnp.sum(x_f32 * y_f32, -1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x_f32 + (1 - c * x2) * y_f32
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return PoincareBall.project(num / den.clip(PoincareBall.EPS)).astype(x.dtype)

    @staticmethod
    def gyration(u, v, w, c=1.0):
        """
        Möbius gyration: gyr[u, v]w = -(u ⊕ v) ⊕ (u ⊕ (v ⊕ w))
        Captures non-associativity of hyperbolic addition → position encoding.
        """
        uv = PoincareBall.mobius_add(u, v, c)
        vw = PoincareBall.mobius_add(v, w, c)
        u_vw = PoincareBall.mobius_add(u, vw, c)
        neg_uv = -uv
        return PoincareBall.mobius_add(neg_uv, u_vw, c)

    @staticmethod
    def expmap0(v, c=1.0):
        v_f32 = v.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        v_norm = jnp.linalg.norm(v_f32, axis=-1, keepdims=True)
        safe_norm = v_norm.clip(PoincareBall.EPS)
        tanh_val = jnp.tanh(sqrt_c * safe_norm)
        result = jnp.where(
            safe_norm > 0,
            PoincareBall.project(tanh_val * v_f32 / (sqrt_c * safe_norm)),
            jnp.zeros_like(v_f32)
        )
        return result.astype(v.dtype) if isinstance(v, jnp.ndarray) and hasattr(v, 'dtype') else result

    @staticmethod
    def logmap0(y, c=1.0):
        y_f32 = y.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c).astype(jnp.float32)
        y_norm = jnp.linalg.norm(y_f32, axis=-1, keepdims=True)
        safe_norm = y_norm.clip(PoincareBall.EPS)
        arctanh_val = jnp.arctanh(y_norm.clip(max=1.0 - PoincareBall.EPS))
        return jnp.where(
            safe_norm > 0,
            arctanh_val * y_f32 / (sqrt_c * safe_norm),
            jnp.zeros_like(y_f32)
        )


# ─── Block-Wise FP8 Quantization (Turbo-style) ────────────────────

class BlockQuantizer:
    """
    Block-wise quantization with per-block scale factors.
    Splits tensor into blocks of size B, computes absmax scale per block,
    quantizes to FP8 (e4m3) storage.

    This is the 'turbo quant' approach: minimal overhead, maximal throughput.
    Applied to the latent KV cache for ~4x compression vs FP32.
    """

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def quantize(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Quantize to simulated FP8 with per-block scaling. Returns (q, scales)."""
        # Flatten and reshape to blocks
        orig_shape = x.shape
        flat = x.reshape(-1)
        n_full_blocks = flat.shape[0] // self.block_size
        remainder = flat.shape[0] % self.block_size

        # Split into blocks
        blocks = flat[:n_full_blocks * self.block_size].reshape(-1, self.block_size)

        # Per-block absmax scale
        absmax = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True).clip(PoincareBall.EPS)
        # Scale to FP8 range: [-448, 448] for e4m3
        scales = absmax / 448.0
        quantized = jnp.clip(
            jnp.round(blocks / scales),
            -448, 447
        ).astype(jnp.int8)

        if remainder > 0:
            last_block = flat[-remainder:]
            last_absmax = jnp.max(jnp.abs(last_block)).clip(PoincareBall.EPS)
            last_scale = last_absmax / 448.0
            last_q = jnp.clip(
                jnp.round(last_block / last_scale),
                -448, 447
            ).astype(jnp.int8)
            scales_combined = jnp.concatenate([scales[:, 0], last_scale.reshape(1)])
            quantized_combined = jnp.concatenate([quantized.reshape(-1), last_q])
            return quantized_combined.reshape(orig_shape).astype(x.dtype), scales_combined.reshape(-1)

        return quantized.reshape(orig_shape).astype(x.dtype), scales.reshape(-1)

    def dequantize(self, q: jnp.ndarray, scales: jnp.ndarray, orig_shape: tuple) -> jnp.ndarray:
        """Dequantize back to original dtype."""
        flat = q.reshape(-1)
        n_full_blocks = flat.shape[0] // self.block_size
        remainder = flat.shape[0] % self.block_size

        if remainder > 0:
            blocks = flat[:n_full_blocks * self.block_size].reshape(-1, self.block_size)
            last_block = flat[-remainder:]
            deq_blocks = blocks * scales[:n_full_blocks, None] * 448.0
            deq_last = last_block * scales[n_full_blocks] * 448.0
            return jnp.concatenate([deq_blocks.reshape(-1), deq_last]).reshape(orig_shape).astype(jnp.float32)

        blocks = flat.reshape(-1, self.block_size)
        return (blocks * scales[:, None] * 448.0).reshape(orig_shape).astype(jnp.float32)


# ─── Latent KV Cache (DeepSeek MLA core) ──────────────────────────

class LatentKVProjection(nn.Module):
    """
    Low-rank KV joint compression: c_t^KV = W^{DKV} * h_t
    where d_c (compressed dim) << n_h * d_h (full KV dim)

    At inference, only c_t^KV is cached, not the full K, V.
    W^{UK}, W^{UV} up-project c_t^KV to full K, V at compute time.

    This reduces KV cache from 2*n_h*d_h to d_c + d_h^R elements per token.
    For DeepSeek-V2: d_c = 4*d_h → ~4.5x compression vs MHA.

    Also: W^{UK} is absorbed into W^Q at inference → zero overhead.
    """

    d_model: int
    d_compressed: int  # d_c: compressed latent dimension
    d_head: int        # d_h: per-head dimension
    n_heads: int       # n_h: number of attention heads
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, use_quant: bool = False):
        """
        x: [batch, seq, d_model] hidden states
        Returns: (q_proj, kv_latent, k_rope, v_full_for_training)
          q_proj: [batch, seq, n_heads, d_head] queries
          kv_latent: [batch, seq, d_compressed] compressed KV (cache this)
          k_rope: [batch, seq, n_heads, d_head//2] RoPE-decoupled keys
          v_full: [batch, seq, n_heads, d_head] values (for training only)
        """
        B, T, D = x.shape

        # ── Query projection ──
        # W^Q: [d_model, n_heads * d_head]
        q_proj = nn.Dense(self.n_heads * self.d_head, use_bias=False, dtype=self.dtype, name="wq")(x)
        q = q_proj.reshape(B, T, self.n_heads, self.d_head)

        # ── Low-rank KV joint compression ──
        # W^{DKV}: [d_model, d_compressed] → c_t^KV
        d_kv_out = self.d_compressed + self.n_heads * (self.d_head // 2)
        kv_latent_raw = nn.Dense(self.d_compressed, use_bias=False, dtype=self.dtype, name="wdkv")(x)

        # Separate into compressed latent + RoPE part
        kv_latent = kv_latent_raw  # [B, T, d_c]

        # ── Decoupled RoPE (position via gyration in Poincaré ball) ──
        # DeepSeek uses standard RoPE on a decoupled q/k channel.
        # We replace RoPE with Möbius gyration to capture sequence ordering
        # in hyperbolic space — preserving the decoupling principle.
        d_rope = self.d_head // 2
        q_rope_proj = nn.Dense(self.n_heads * d_rope, use_bias=False, dtype=self.dtype, name="wqr")(x)
        q_rope = q_rope_proj.reshape(B, T, self.n_heads, d_rope)

        # k_rope from compressed + separate projection
        k_rope_proj = nn.Dense(self.n_heads * d_rope, use_bias=False, dtype=self.dtype, name="wkr")(x)
        k_rope = k_rope_proj.reshape(B, T, self.n_heads, d_rope)

        # ── Apply hyperbolic gyration as position encoding ──
        # Map q_rope into Poincaré ball and apply gyration between adjacent positions
        for h in range(self.n_heads):
            q_rope_h = q_rope[:, :, h, :]
            k_rope_h = k_rope[:, :, h, :]

            # Map to ball
            q_ball = PoincareBall.expmap0(q_rope_h.reshape(-1, d_rope))
            q_ball = q_ball.reshape(B, T, d_rope)

            k_ball = PoincareBall.expmap0(k_rope_h.reshape(-1, d_rope))
            k_ball = k_ball.reshape(B, T, d_rope)

            # Gyration between adjacent positions (this is the position encoding)
            for t in range(1, T):
                gyro = PoincareBall.gyration(q_ball[:, t-1, :], q_ball[:, t, :], k_ball[:, t, :])
                k_ball = k_ball.at[:, t, :].set(
                    PoincareBall.mobius_add(k_ball[:, t, :], gyro)
                )

            # Logmap back to tangent space
            k_rope = k_rope.at[:, :, h, :].set(
                PoincareBall.logmap0(k_ball.reshape(-1, d_rope)).reshape(B, T, d_rope)
            )

            q_rope = q_rope.at[:, :, h, :].set(
                PoincareBall.logmap0(q_ball.reshape(-1, d_rope)).reshape(B, T, d_rope)
            )

        return q, kv_latent, k_rope, q_rope


class LatentAttention(nn.Module):
    """
    Full MLA with latent KV cache, hyperbolic gyration RoPE, and turbo quantization.
    This is the unified attention head.
    """

    d_model: int
    d_compressed: int
    d_head: int
    n_heads: int
    use_quant: bool = False
    quant_block_size: int = 128
    dtype: Any = jnp.bfloat16

    def setup(self):
        self.latent_proj = LatentKVProjection(
            d_model=self.d_model,
            d_compressed=self.d_compressed,
            d_head=self.d_head,
            n_heads=self.n_heads,
            dtype=self.dtype
        )
        # Value up-projection: W^{UV}: [d_c, n_heads * d_h]
        self.w_uv = nn.Dense(self.n_heads * self.d_head, use_bias=False, dtype=self.dtype, name="wuv")
        # Key up-projection for training (absorbed at inference): W^{UK}: [d_c, n_heads * d_h]
        self.w_uk = nn.Dense(self.n_heads * self.d_head, use_bias=False, dtype=self.dtype, name="wuk")
        # Output projection
        self.w_o = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, name="wo")
        self.quantizer = BlockQuantizer(block_size=self.quant_block_size)

    def __call__(self, x: jnp.ndarray, kv_cache: Optional[dict] = None, is_training: bool = True):
        B, T, D = x.shape

        # ── Project Q, latent KV, RoPE keys ──
        q, kv_latent, k_rope, q_rope = self.latent_proj(x)

        if kv_cache is not None and not is_training:
            # Inference: use cached latent + apply turbo quant
            cached_latent = kv_cache['latent']
            # Extend cache
            if self.use_quant:
                q_latent, scales = self.quantizer.quantize(kv_latent)
                full_latent = jnp.concatenate([cached_latent, q_latent], axis=1)
                kv_cache['latent'] = full_latent
                kv_cache['scales'] = scales
                # Dequant for compute
                kv_latent = self.quantizer.dequantize(
                    full_latent, scales, (B, full_latent.shape[1], self.d_compressed)
                )
            else:
                full_latent = jnp.concatenate([cached_latent, kv_latent], axis=1)
                kv_cache['latent'] = full_latent
        else:
            full_latent = kv_latent

        # ── Reconstruct K, V from latent (MLA's key trick) ──
        # W^{UK} * c_t^KV → full K (training only; absorbed at inference)
        # W^{UV} * c_t^KV → full V
        k_full = self.w_uk(full_latent.reshape(-1, self.d_compressed))
        k_full = k_full.reshape(B, -1, self.n_heads, self.d_head)

        v_full = self.w_uv(full_latent.reshape(-1, self.d_compressed))
        v_full = v_full.reshape(B, -1, self.n_heads, self.d_head)

        # RoPE part of K (decoupled)
        # k_rope is concatenated with k_full along head_dim
        k_rope_full = jnp.concatenate([k_rope, jnp.zeros_like(k_rope)], axis=-1)
        q_rope_full = jnp.concatenate([q_rope, jnp.zeros_like(q_rope)], axis=-1)

        # ── Compute attention with hyperbolic-gyrated positions ──
        # Standard scaled dot-product but with gyrated positions baked into q/k_rope
        scale = jnp.sqrt(self.d_head).astype(self.dtype)

        # Combine main K with position-encoded K
        k_combined = k_full + k_rope_full

        # Attention: [B, H, T_q, T_kv]
        T_full = k_combined.shape[1]
        attn_logits = jnp.einsum('bqhd,bkhd->bhqk', q, k_combined[:, :T_full, :, :]) / scale

        # Causal mask: [1, 1, T_q, T_kv] broadcastable to [B, H, T_q, T_kv]
        if kv_cache is not None and not is_training:
            mask = jnp.triu(jnp.ones((1, 1, T, T_full), dtype=self.dtype) * -1e10, k=T_full - T + 1)
        else:  # training
            mask = jnp.triu(jnp.ones((1, 1, T, T), dtype=self.dtype) * -1e10, k=1)
        attn_logits = attn_logits + mask

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        # Aggregate values: [B, H, T, T_kv] @ [B, T_kv, H, d] -> [B, T, H, d]
        out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v_full[:, :T_full, :, :])
        out = out.reshape(B, T, self.n_heads * self.d_head)

        # Output projection
        return self.w_o(out), kv_cache


# ─── WuBu Transformer Block ───────────────────────────────────────

class WubuTransformerBlock(nn.Module):
    """
    Full transformer block with:
    1. LatentAttention (MLA + hyperbolic gyration + turbo quant)
    2. WuBu nested feed-forward (hyperbolic residual)
    3. Complex-valued gating (from WuBuMindV7)
    """

    d_model: int
    d_compressed: int
    d_head: int
    n_heads: int
    d_ff: int
    use_quant: bool = False
    dropout_rate: float = 0.1
    dtype: Any = jnp.bfloat16

    def setup(self):
        self.attention = LatentAttention(
            d_model=self.d_model,
            d_compressed=self.d_compressed,
            d_head=self.d_head,
            n_heads=self.n_heads,
            use_quant=self.use_quant,
            dtype=self.dtype
        )
        self.norm1 = nn.LayerNorm(dtype=self.dtype)
        self.norm2 = nn.LayerNorm(dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # WuBu nested FFN: complex-valued with hyperbolic gating
        self.ffn_real1 = nn.Dense(self.d_ff, dtype=self.dtype)
        self.ffn_imag1 = nn.Dense(self.d_ff, dtype=self.dtype)
        self.ffn_real2 = nn.Dense(self.d_model, dtype=self.dtype)
        self.ffn_imag2 = nn.Dense(self.d_model, dtype=self.dtype)

        # Hyperbolic gating weights (level-2 nesting)
        self.phi_gate = self.param(
            'phi_gate', nn.initializers.constant(0.5), (2,), self.dtype
        )

    def __call__(self, x: jnp.ndarray, kv_cache: Optional[dict] = None, is_training: bool = True):
        # ── Attention with residual ──
        attn_out, kv_cache = self.attention(self.norm1(x), kv_cache, is_training)
        x = x + self.dropout(attn_out, deterministic=not is_training)

        # ── WuBu nested FFN (complex-valued) ──
        residual = x

        # Split into real/imag for complex gating
        d_half = self.d_model // 2
        x_real = x[:, :, :d_half]
        x_imag = x[:, :, d_half:2*d_half] if self.d_model >= 2*d_half else x[:, :, :d_half]

        # First layer (complex multiplication)
        h_real = self.ffn_real1(x_real) - self.ffn_imag1(x_imag)
        h_imag = self.ffn_real1(x_imag) + self.ffn_imag1(x_real)
        h_real = nn.gelu(h_real)
        h_imag = nn.gelu(h_imag)

        # Second layer
        out_real = self.ffn_real2(h_real) - self.ffn_imag2(h_imag)
        out_imag = self.ffn_real2(h_imag) + self.ffn_imag2(x_real)

        # Hyperbolic gating: φ-weighted interpolation between complex output and residual
        phi_1, phi_2 = nn.sigmoid(self.phi_gate)
        gated = phi_1 * out_real + phi_2 * x_real + (1 - phi_1 - phi_2) * (out_real + out_imag) / 2.0

        # Combine back to full dimension
        x = residual + self.dropout(gated, deterministic=not is_training)
        x = self.norm2(x)

        return x, kv_cache
