"""
Simplified FFN for WubuBlocks. Replaces the MoE with a single FFN per layer
until GPU compilation is verified working. The MoE module is preserved
but can be swapped via config flag.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Any, Optional


class SimpleFFN(nn.Module):
    """Standard 2-layer FFN with GELU activation."""
    
    d_model: int
    d_ff: int
    dtype: Any = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.d_ff, dtype=self.dtype)(x)
        h = nn.gelu(h)
        return nn.Dense(self.d_model, dtype=self.dtype)(h)


# Re-export MoE for when it's stable
from wubu_moe import SparseMoERouter, WubuMoELayer


class WubuBlock(nn.Module):
    """Complete WuBuNest block: LatentAttention + FFN + Dropout."""
    
    d_model: int
    d_compressed: int
    d_head: int
    n_heads: int
    d_ff: int
    n_shared: int = 0
    n_routed: int = 0
    top_k: int = 0
    use_moe: bool = False  # Set True to enable MoE
    use_quant: bool = False
    dropout_rate: float = 0.1
    dtype: Any = jnp.bfloat16
    
    def setup(self):
        from wubu_latent_attention import LatentAttention
        
        self.norm1 = nn.LayerNorm(dtype=self.dtype)
        self.norm2 = nn.LayerNorm(dtype=self.dtype)
        
        self.attention = LatentAttention(
            d_model=self.d_model,
            d_compressed=self.d_compressed,
            d_head=self.d_head,
            n_heads=self.n_heads,
            use_quant=self.use_quant,
            dtype=self.dtype
        )
        
        if self.use_moe and self.n_routed > 0:
            self.ffn = WubuMoELayer(
                n_shared=self.n_shared,
                n_routed=self.n_routed,
                top_k=self.top_k,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dtype=self.dtype
            )
        else:
            self.ffn = SimpleFFN(d_model=self.d_model, d_ff=self.d_ff, dtype=self.dtype)

    @nn.compact
    def __call__(self, x, kv_cache=None, expert_bias=None, is_training=True):
        # Attention
        attn_out, kv_cache = self.attention(self.norm1(x), kv_cache, is_training)
        if is_training:
            attn_out = nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=False)
        x = x + attn_out
        
        # FFN
        if self.use_moe and self.n_routed > 0:
            ffn_out, bias_update = self.ffn(self.norm2(x), expert_bias, is_training)
        else:
            ffn_out = self.ffn(self.norm2(x))
            bias_update = None
        
        if is_training:
            ffn_out = nn.Dropout(rate=self.dropout_rate)(ffn_out, deterministic=False)
        x = x + ffn_out
        
        return x, kv_cache, bias_update
