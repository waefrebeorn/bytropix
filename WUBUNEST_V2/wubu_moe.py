"""
WubuMoE v2: DeepSeek-style Mixture of Experts with Sparse Routing
+ Auxiliary-loss-free load balancing (DeepSeek-V3 innovation)
+ Proper Flax module management (no list-of-modules anti-pattern)

Instead of list-of-MLP-expert, uses nn.Module with a single large Linear
and sliced parameters — proper Flax pattern that JIT's correctly.

Reference: DeepSeek-V3 (arXiv:2412.19437), DeepSeekMoE (arXiv:2401.06066)
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Any, Optional


class SparseMoERouter(nn.Module):
    """
    DeepSeek-style router with auxiliary-loss-free load balancing.
    
    Instead of an auxiliary loss that penalizes imbalance (which hurts model quality),
    DeepSeek-V3 adds a per-expert bias term that is updated based on load.
    """

    n_shared: int
    n_routed: int
    top_k: int
    d_model: int
    dtype: Any = jnp.bfloat16

    def setup(self):
        self.shared_gate = nn.Dense(self.n_shared, dtype=self.dtype, name="shared_gate")
        self.router_weight = nn.Dense(self.n_routed, use_bias=False, dtype=self.dtype, name="router_weight")
        self.routing_bias = self.param(
            'routing_bias', nn.initializers.zeros, (self.n_routed,), jnp.float32
        )

    def __call__(self, x, expert_bias=None, is_training=True):
        B, T, D = x.shape
        n_tokens = B * T
        
        shared_logits = self.shared_gate(x)
        raw_scores = self.router_weight(x)
        
        if expert_bias is not None:
            biased_scores = raw_scores + expert_bias[None, None, :]
        else:
            biased_scores = raw_scores + self.routing_bias[None, None, :]
        
        routing_weights = jax.nn.softmax(biased_scores, axis=-1)
        top_k_weights, top_k_indices = jax.lax.top_k(routing_weights, self.top_k)
        top_k_weights = top_k_weights / (jnp.sum(top_k_weights, axis=-1, keepdims=True) + 1e-10)
        
        if is_training:
            flat_indices = top_k_indices.reshape(-1)
            flat_one_hot = jax.nn.one_hot(flat_indices, self.n_routed)
            expert_counts = jnp.sum(flat_one_hot, axis=0)
            target_per_expert = n_tokens * self.top_k / self.n_routed
            load_diff = target_per_expert - expert_counts
            bias_update = jnp.sign(load_diff) * 0.01
        else:
            bias_update = jnp.zeros(self.n_routed)
        
        return shared_logits, top_k_indices, top_k_weights, bias_update


class WubuMoELayer(nn.Module):
    """
    MoE FFN layer using fused Linear for all experts.
    We store one big W1: [d_model, n_shared + n_routed, d_ff] and
    one big W2: [d_ff, n_shared + n_routed, d_model].
    This avoids the list-of-modules problem in Flax.
    
    Experts 0..n_shared-1 are shared (always activated).
    Experts n_shared..n_shared+n_routed-1 are routed (top-k sparse).
    """

    n_shared: int
    n_routed: int
    top_k: int
    d_model: int
    d_ff: int
    dtype: Any = jnp.bfloat16

    def setup(self):
        self.router = SparseMoERouter(
            n_shared=self.n_shared,
            n_routed=self.n_routed,
            top_k=self.top_k,
            d_model=self.d_model,
            dtype=self.dtype
        )
        
        n_total = self.n_shared + self.n_routed
        
        # Fused expert weight: each expert has 2-layer MLP
        # W1: [d_model, n_total, d_ff]
        self.expert_w1 = self.param(
            'expert_w1', 
            nn.initializers.lecun_normal(), 
            (self.d_model, n_total, self.d_ff), 
            self.dtype
        )
        self.expert_b1 = self.param(
            'expert_b1',
            nn.initializers.zeros,
            (n_total, self.d_ff),
            self.dtype
        )
        
        # W2: [d_ff, n_total, d_model]
        self.expert_w2 = self.param(
            'expert_w2',
            nn.initializers.lecun_normal(),
            (self.d_ff, n_total, self.d_model),
            self.dtype
        )
        self.expert_b2 = self.param(
            'expert_b2',
            nn.initializers.zeros,
            (n_total, self.d_model),
            self.dtype
        )

    def _expert_forward(self, x, expert_ids, weights):
        """
        Expert forward using jnp.take for efficient gather.
        x: [B, d_model]
        expert_ids: [B, k] expert indices
        weights: [B, k]
        Returns: [B, d_model]
        """
        B, D = x.shape
        k = expert_ids.shape[-1]
        
        output = jnp.zeros_like(x)
        
        for ki in range(k):
            e_ids = expert_ids[:, ki]  # [B]
            w = weights[:, ki:ki+1]    # [B, 1]
            
            # Gather with jnp.take: [n_total, d_model, d_ff] with indices [B] -> [B, d_model, d_ff]
            w1 = jnp.take(self.expert_w1, e_ids, axis=1)       # [d_model, B, d_ff] -> transpose
            w1 = jnp.transpose(w1, (1, 0, 2))                  # [B, d_model, d_ff]
            b1 = jnp.take(self.expert_b1, e_ids, axis=0)        # [B, d_ff]
            w2 = jnp.take(self.expert_w2, e_ids, axis=1)       # [d_ff, B, d_model]
            w2 = jnp.transpose(w2, (1, 0, 2))                  # [B, d_ff, d_model]
            b2 = jnp.take(self.expert_b2, e_ids, axis=0)        # [B, d_model]
            
            # h = x @ W1 + b1: [B, d_model] @ [B, d_model, d_ff] -> [B, d_ff]
            h = jnp.einsum('bd,bdf->bf', x, w1) + b1
            h = nn.gelu(h)
            
            out = jnp.einsum('bf,bfd->bd', h, w2) + b2
            output = output + out * w
        
        return output

    def __call__(self, x, expert_bias=None, is_training=True):
        B, T, D = x.shape
        n_total = self.n_shared + self.n_routed
        
        # Route
        shared_logits, routed_idxs, routed_weights, bias_update = self.router(
            x, expert_bias, is_training
        )
        
        # ── Shared experts (always activate first n_shared experts) ──
        flat_x = x.reshape(-1, D)  # [B*T, d_model]
        
        # Weighted combination of shared experts
        shared_out = jnp.zeros_like(flat_x)
        for ei in range(self.n_shared):
            gate = nn.sigmoid(shared_logits[:, :, ei:ei+1])  # [B, T, 1]
            # Expert ei is at global index ei
            w1_e = self.expert_w1[:, ei, :]   # [d_model, d_ff]
            b1_e = self.expert_b1[ei, :]      # [d_ff]
            h = jnp.dot(flat_x, w1_e) + b1_e[None, :]
            h = nn.gelu(h)
            w2_e = self.expert_w2[:, ei, :]   # [d_ff, d_model]
            b2_e = self.expert_b2[ei, :]      # [d_model]
            e_out = jnp.dot(h, w2_e) + b2_e[None, :]
            shared_out = shared_out + e_out * gate.reshape(-1, 1)
        
        # ── Routed experts (sparse, top-k) ──
        # Shift routed indices by n_shared to get global expert IDs
        global_idxs = routed_idxs.reshape(-1, self.top_k) + self.n_shared
        global_weights = routed_weights.reshape(-1, self.top_k)
        
        routed_out = self._expert_forward(flat_x, global_idxs, global_weights)
        
        result = shared_out + routed_out
        return result.reshape(B, T, D), bias_update


class WubuBlock(nn.Module):
    """Complete WuBuNest block: LatentAttention + WubuMoE + Dropout."""

    d_model: int
    d_compressed: int
    d_head: int
    n_heads: int
    d_ff: int
    n_shared: int
    n_routed: int
    top_k: int
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
        
        self.moe = WubuMoELayer(
            n_shared=self.n_shared,
            n_routed=self.n_routed,
            top_k=self.top_k,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dtype=self.dtype
        )

    @nn.compact
    def __call__(self, x, kv_cache=None, expert_bias=None, is_training=True):
        # Attention
        attn_out, kv_cache = self.attention(self.norm1(x), kv_cache, is_training)
        if is_training:
            attn_out = nn.Dropout(rate=self.dropout_rate)(attn_out, deterministic=False)
        x = x + attn_out
        
        # MoE FFN
        moe_out, bias_update = self.moe(self.norm2(x), expert_bias, is_training)
        if is_training:
            moe_out = nn.Dropout(rate=self.dropout_rate)(moe_out, deterministic=False)
        x = x + moe_out
        
        return x, kv_cache, bias_update
