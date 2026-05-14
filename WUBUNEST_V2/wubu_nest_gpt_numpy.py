"""
WuBuNestGPT - Pure NumPy Implementation
Architecture: minGPT-style with WuBuNested LatentAttention + hyperbolic gyration
Zero compilation overhead, runs on anything with numpy.

Key components from DeepSeek MLA:
- Low-rank KV compression: latent state d_c << n_h * d_h
- Decoupled RoPE → replaced with Möbius gyration in Poincaré ball
- Block quantization for KV cache

Structure matches Karpathy's minGPT closely so it's immediately familiar.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import pickle
import os


# ─── Activation Functions ─────────────────────────────────────────

def gelu(x):
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-10)

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def dropout(x, drop_prob, rng=None):
    """Dropout during training."""
    if drop_prob <= 0.0:
        return x
    if rng is None:
        return x
    mask = (rng.random(x.shape) > drop_prob) / (1.0 - drop_prob)
    return x * mask


# ─── Parameter Initialization ─────────────────────────────────────

def init_weight(shape, scale=0.02):
    """Initialize weight matrix with small normal."""
    return np.random.randn(*shape).astype(np.float32) * scale

def init_zeros(shape):
    """Initialize bias with zeros."""
    return np.zeros(shape, dtype=np.float32)

class Param:
    """Simple parameter container."""
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


# ─── Poincaré Ball Operations ────────────────────────────────────

class PoincareBallNumpy:
    """Pure numpy hyperbolic operations."""
    
    @staticmethod
    def project(x, eps=1e-7):
        norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        mask = norm_sq >= 1.0
        x_proj = np.where(mask, x / np.sqrt(norm_sq).clip(eps) * (1.0 - eps), x)
        return x_proj
    
    @staticmethod
    def mobius_add(x, y, c=1.0, eps=1e-7):
        x2 = np.sum(x ** 2, axis=-1, keepdims=True)
        y2 = np.sum(y ** 2, axis=-1, keepdims=True)
        xy = np.sum(x * y, axis=-1, keepdims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return PoincareBallNumpy.project(num / (den + eps))
    
    @staticmethod
    def gyration(u, v, w, c=1.0):
        """Möbius gyration: position encoding in hyperbolic space."""
        uv = PoincareBallNumpy.mobius_add(u, v, c)
        vw = PoincareBallNumpy.mobius_add(v, w, c)
        u_vw = PoincareBallNumpy.mobius_add(u, vw, c)
        return PoincareBallNumpy.mobius_add(-uv, u_vw, c)
    
    @staticmethod
    def expmap0(v, c=1.0, eps=1e-7):
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        safe_norm = v_norm + eps
        tanh_val = np.tanh(np.sqrt(c) * safe_norm)
        return PoincareBallNumpy.project(tanh_val * v / (np.sqrt(c) * safe_norm))
    
    @staticmethod
    def logmap0(y, c=1.0, eps=1e-7):
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        safe_norm = y_norm + eps
        arctanh_val = np.arctanh(np.clip(y_norm, 0, 1 - eps))
        return arctanh_val * y / (np.sqrt(c) * safe_norm)


# ─── Tokenizer ───────────────────────────────────────────────────

class CharTokenizer:
    """Simple character-level tokenizer (like minGPT)."""
    
    def __init__(self, texts=None, vocab_size=5000):
        self.vocab_size = vocab_size
        if texts:
            self._build(texts)
        else:
            self.stoi = {}
            self.itos = {}
    
    def _build(self, texts):
        chars = sorted(list(set(''.join(texts))))
        # Add special tokens
        special = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        vocab = special + chars[:min(len(chars), self.vocab_size - 4)]
        self.itos = {i: c for i, c in enumerate(vocab)}
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)
    
    def encode(self, text):
        return [self.stoi.get(c, self.stoi.get('<UNK>', 1)) for c in text]
    
    def decode(self, ids):
        return ''.join(self.itos.get(i, '?') for i in ids if i in self.itos)


# ─── WuBuNestGPT Model ────────────────────────────────────────────

class WubuNestGPT:
    """
    Pure NumPy WuBuNested GPT.
    
    Architecture per block:
    1. LayerNorm → LatentAttention (MLA-style with hyperbolic gyration)
    2. Residual + Dropout
    3. LayerNorm → MLP (2-layer FFN)
    4. Residual + Dropout
    
    MLA components:
    - W_q, W_k, W_v: query/key/value projections
    - W_dkv: low-rank KV compression (key DeepSeek innovation)
    - d_compressed << n_heads * d_head → 4x KV cache reduction
    
    Hyperbolic gyration replaces RoPE for position encoding.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.params = {}
        self._build()
    
    def _build(self):
        cfg = self.config
        V = cfg['vocab_size']
        D = cfg['d_model']
        H = cfg['n_heads']
        d_h = cfg.get('d_head', D // H)
        d_c = cfg.get('d_compressed', D // 2)  # compressed latent dim
        d_ff = cfg.get('d_ff', D * 4)
        N = cfg['n_layers']
        d_rope = d_h // 2
        
        np.random.seed(cfg.get('seed', 42))
        scale = cfg.get('init_scale', 0.02)
        
        # Token embeddings
        self.params['wte'] = Param(init_weight((V, D), scale))
        
        # Per-block parameters
        self.params['blocks'] = []
        for i in range(N):
            block = {}
            prefix = f'b{i}'
            
            # Layer norms
            block['ln1_g'] = Param(np.ones(D, dtype=np.float32))
            block['ln1_b'] = Param(np.zeros(D, dtype=np.float32))
            block['ln2_g'] = Param(np.ones(D, dtype=np.float32))
            block['ln2_b'] = Param(np.zeros(D, dtype=np.float32))
            
            # MLA: Low-rank KV joint compression (DeepSeek innovation)
            # Q: [D, H * d_h]  (full multi-head)
            block['wq'] = Param(init_weight((D, H * d_h), scale))
            
            # K,V compressed: W^{DKV}: [D, d_c] → c_t^KV
            block['wdkv'] = Param(init_weight((D, d_c), scale))
            
            # Up-projections: W^{UK}: [d_c, H * d_h], W^{UV}: [d_c, H * d_h]
            block['wuk'] = Param(init_weight((d_c, H * d_h), scale))
            block['wuv'] = Param(init_weight((d_c, H * d_h), scale))
            
            # Decoupled RoPE projections (replaced by hyperbolic gyration)
            block['wqr'] = Param(init_weight((D, H * d_rope), scale))
            block['wkr'] = Param(init_weight((D, H * d_rope), scale))
            
            # Output projection
            block['wo'] = Param(init_weight((H * d_h, D), scale))
            
            # FFN
            block['ffn1'] = Param(init_weight((D, d_ff), scale))
            block['ffn1_b'] = Param(init_zeros(d_ff))
            block['ffn2'] = Param(init_weight((d_ff, D), scale))
            block['ffn2_b'] = Param(init_zeros(D))
            
            self.params['blocks'].append(block)
        
        # Final layer norm
        self.params['ln_f_g'] = Param(np.ones(D, dtype=np.float32))
        self.params['ln_f_b'] = Param(np.zeros(D, dtype=np.float32))
        
        # LM head (can be weight-tied)
        self.params['lm_head'] = Param(init_weight((D, V), scale))
        
        self._D = D
        self._H = H
        self._d_h = d_h
        self._d_c = d_c
        self._d_rope = d_rope
        self._d_ff = d_ff  # FFN hidden dim
        self._N = N
        print(f"  WuBuNestGPT built: {self.count_params():,} params")
    
    def count_params(self):
        """Count total parameters."""
        total = 0
        for name, p in self.params.items():
            if name == 'blocks':
                for b in self.params['blocks']:
                    for v in b.values():
                        total += v.data.size
            else:
                total += p.data.size
        return total
    
    def forward(self, x, is_training=False, rng=None):
        """
        Forward pass.
        x: [B, T] token indices
        Returns: logits [B, T, V], cached latent states
        """
        cfg = self.config
        B, T = x.shape
        D = self._D
        H = self._H
        d_h = self._d_h
        d_c = self._d_c
        d_rope = self._d_rope
        N = self._N
        drop = cfg.get('dropout_rate', 0.1)
        
        # Embed tokens
        h = self.params['wte'].data[x]  # [B, T, D]
        
        # Store activations for backward pass
        cache = {}
        cache['x_input'] = x
        cache['h_in'] = h
        cache['layer_outputs'] = []
        
        # Per-block processing
        for i in range(N):
            b = self.params['blocks'][i]
            
            # ── Attention sub-layer ──
            h_norm = layer_norm(h, b['ln1_g'].data, b['ln1_b'].data)
            
            # MLA: Low-rank KV compression
            # Q: [B, T, H * d_h]
            q = h_norm @ b['wq'].data  # [B, T, H*d_h]
            q = q.reshape(B, T, H, d_h)
            
            # Latent KV: c = h @ W^{DKV}, [B, T, d_c]
            kv_latent = h_norm @ b['wdkv'].data  # [B, T, d_c]
            
            # Reconstruct K, V from latent
            k = kv_latent @ b['wuk'].data  # [B, T, H*d_h]
            k = k.reshape(B, T, H, d_h)
            v = kv_latent @ b['wuv'].data  # [B, T, H*d_h]
            v = v.reshape(B, T, H, d_h)
            
            # Hyperbolic gyration position encoding (replaces RoPE)
            q_rope = h_norm @ b['wqr'].data  # [B, T, H*d_rope]
            q_rope = q_rope.reshape(B, T, H, d_rope)
            k_rope = h_norm @ b['wkr'].data  # [B, T, H*d_rope]
            k_rope = k_rope.reshape(B, T, H, d_rope)
            
            # Apply gyration between adjacent positions for each head
            for h_idx in range(min(H, 4)):  # Apply on first 4 heads for efficiency
                for t in range(1, T):
                    # Map to Poincaré ball
                    qb = PoincareBallNumpy.expmap0(q_rope[:, t, h_idx, :])
                    k_prev = PoincareBallNumpy.expmap0(k_rope[:, t-1, h_idx, :])
                    k_curr = PoincareBallNumpy.expmap0(k_rope[:, t, h_idx, :])
                    gyro = PoincareBallNumpy.gyration(k_prev, qb, k_curr)
                    # Apply gyration to key
                    k_ball_adj = PoincareBallNumpy.mobius_add(k_curr, gyro)
                    k_rope[:, t, h_idx, :] = PoincareBallNumpy.logmap0(k_ball_adj)
            
            # Combine K with position info
            k_positioned = k + np.concatenate([k_rope, np.zeros((B, T, H, d_h - d_rope), dtype=np.float32)], axis=-1)
            
            # Attention scores
            scale = np.sqrt(d_h)
            attn = np.einsum('bthd,bThd->bhtT', q, k_positioned) / scale
            
            # Causal mask
            mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e10, k=1)
            attn = attn + mask[None, None, :, :]
            
            # Softmax
            attn_weights = softmax(attn, axis=-1)  # [B, H, T, T]
            
            if is_training:
                attn_weights = dropout(attn_weights, drop, rng)
            
            # Aggregate values
            attn_out = np.einsum('bhtT,bThd->bthd', attn_weights, v)
            attn_out = attn_out.reshape(B, T, H * d_h)
            
            # Output projection
            attn_out = attn_out @ b['wo'].data  # [B, T, D]
            
            if is_training:
                attn_out = dropout(attn_out, drop, rng)
            
            h = h + attn_out  # Residual
            
            # ── FFN sub-layer ──
            h_norm2 = layer_norm(h, b['ln2_g'].data, b['ln2_b'].data)
            
            ffn_out = h_norm2 @ b['ffn1'].data + b['ffn1_b'].data
            ffn_out = gelu(ffn_out)
            ffn_out = ffn_out @ b['ffn2'].data + b['ffn2_b'].data
            
            if is_training:
                ffn_out = dropout(ffn_out, drop, rng)
            
            h = h + ffn_out  # Residual
            
            # Cache for backward
            cache['layer_outputs'].append({
                'h': h.copy(),
                'h_norm': h_norm,
                'h_norm2': h_norm2,
                'q': q,
                'kv_latent': kv_latent,
                'k': k,
                'v': v,
                'q_rope': q_rope,
                'k_rope': k_rope,
                'k_positioned': k_positioned,
                'attn_weights': attn_weights,
                'attn_out': attn_out,
                'ffn_pre_gelu': h_norm2 @ b['ffn1'].data + b['ffn1_b'].data,
                'ffn_out': ffn_out,
            })
        
        # Final layer norm
        h = layer_norm(h, self.params['ln_f_g'].data, self.params['ln_f_b'].data)
        cache['h_final'] = h
        
        # LM head
        logits = h @ self.params['lm_head'].data  # [B, T, V]
        cache['logits'] = logits
        
        return logits, cache
    
    def compute_loss(self, logits, targets):
        """Cross-entropy loss."""
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Softmax cross-entropy
        logits_stable = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
        log_probs = logits_stable - np.log(np.sum(np.exp(logits_stable), axis=-1, keepdims=True) + 1e-10)
        
        loss = -np.mean(log_probs[np.arange(len(targets_flat)), targets_flat])
        return loss
    
    def backward(self, logits, targets, cache, learning_rate=3e-4):
        """
        Manual backward pass.
        logits: [B, T, V]
        targets: [B, T]
        """
        cfg = self.config
        B, T, V = logits.shape
        D = self._D
        H = self._H
        hdim = self._d_h  # per-head dimension
        d_c = self._d_c
        d_rope = self._d_rope
        d_ff = self._d_ff  # FFN dim
        N = self._N
        
        # ── Gradient of loss w.r.t. logits ──
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Softmax gradient
        probs = softmax(logits_flat, axis=-1)
        probs[np.arange(len(targets_flat)), targets_flat] -= 1.0
        d_logits = probs.reshape(B, T, V)  # [B, T, V]
        
        # ── LM head gradient ──
        h_final = cache['h_final']
        d_lm_head = h_final.reshape(-1, D).T @ d_logits.reshape(-1, V)  # [D, V]
        d_h = d_logits @ self.params['lm_head'].data.T  # [B, T, D]
        
        # ── Final layer norm gradient ──
        d_h = self._layernorm_backward(d_h, h_final, self.params['ln_f_g'].data, self.params['ln_f_b'].data)
        
        # ── Per-block backward ──
        d_blocks = []
        for i in reversed(range(N)):
            b = self.params['blocks'][i]
            layer = cache['layer_outputs'][i]
            
            # ── FFN sub-layer backward ──
            h_pre_ffn = layer['h'] - layer['ffn_out']
            
            # Dropout grad (same mask)
            d_ffn = d_h
            
            # FFN2 backward
            ffn_pre_gelu = layer['ffn_pre_gelu']
            d_ffn2 = ffn_pre_gelu.reshape(-1, d_ff).T @ d_ffn.reshape(-1, D)  # [d_ff, D]
            d_ffn_b2 = np.sum(d_ffn, axis=(0, 1))  # [D]
            d_ffn_h = d_ffn @ b['ffn2'].data.T  # [B, T, d_ff]
            
            # GELU backward
            gelu_grad = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (ffn_pre_gelu + 0.044715 * ffn_pre_gelu ** 3)))
            gelu_grad += (0.5 * ffn_pre_gelu * (1.0 - np.tanh(np.sqrt(2.0 / np.pi) * 
                (ffn_pre_gelu + 0.044715 * ffn_pre_gelu ** 3)) ** 2) * 
                np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * ffn_pre_gelu ** 2))
            d_ffn_h = d_ffn_h * gelu_grad
            
            # FFN1 backward
            h_norm2 = layer['h_norm2']
            d_ffn1 = h_norm2.reshape(-1, self._D).T @ d_ffn_h.reshape(-1, d_ff)  # [D, d_ff]
            d_ffn1_b = np.sum(d_ffn_h, axis=(0, 1))  # [d_ff]
            d_h_ffn = d_ffn_h @ b['ffn1'].data.T  # [B, T, D]
            
            # Add to residual
            d_h_from_ffn = d_h_ffn
            
            # Layer norm 2 backward
            d_h_from_ffn = self._layernorm_backward(d_h_from_ffn, h_norm2, b['ln2_g'].data, b['ln2_b'].data)
            
            # ── Attention sub-layer backward ──
            d_attn_grad = d_h_from_ffn
            
            # Output projection backward
            attn_out = layer['attn_out']  # [B, T, H*d_h]
            # Wo: [H*d_h, D]
            d_wo = attn_out.reshape(-1, self._H * self._d_h).T @ d_attn_grad.reshape(-1, self._D)  # [H*d_h, D]
            d_attn_out = d_attn_grad @ b['wo'].data.T  # [B, T, H*d_h]
            d_attn_out = d_attn_out.reshape(B, T, H, hdim)
            
            # Value aggregation backward
            attn_weights = layer['attn_weights']  # [B, H, T, T]
            v = layer['v']  # [B, T, H, d_h]
            d_v = np.einsum('bhtT,bthd->bThd', attn_weights, d_attn_out)  # [B, T, H, hdim]
            
            # Attention weights backward
            d_attn_weights = np.einsum('bThd,bthd->bhtT', v, d_attn_out) / np.sqrt(hdim)
            
            # Softmax backward (for causal mask)
            s = attn_weights  # [B, H, T, T]
            d_s = s * (d_attn_weights - np.sum(s * d_attn_weights, axis=-1, keepdims=True))
            
            # QK backward
            k_positioned = layer['k_positioned']  # [B, T, H, d_h]
            q = layer['q']  # [B, T, H, d_h]
            
            d_q = np.einsum('bhtT,bThd->bthd', d_s, k_positioned) / np.sqrt(hdim)
            d_k = np.einsum('bhtT,bthd->bThd', d_s, q) / np.sqrt(hdim)
            
            # K gradient through RoPE (simplified: skip gyration backward for now)
            d_k_full = d_k  # [B, T, H, d_h]
            
            # V up-projection gradient
            kv_latent = layer['kv_latent']  # [B, T, d_c]
            d_wuv = kv_latent.reshape(-1, d_c).T @ d_v.reshape(-1, H*hdim)  # [d_c, H*hdim]
            d_kv_v = (d_v.reshape(-1, H*hdim)) @ b['wuv'].data.T  # [B*T, d_c]
            
            # K up-projection gradient
            d_wuk = kv_latent.reshape(-1, d_c).T @ d_k_full.reshape(-1, H*hdim)  # [d_c, H*hdim]
            d_kv_k = (d_k_full.reshape(-1, H*hdim)) @ b['wuk'].data.T  # [B*T, d_c]
            
            # Combined KV latent gradient
            d_kv_latent = (d_kv_v + d_kv_k).reshape(B, T, d_c)
            
            # W^{DKV} gradient
            h_norm = layer['h_norm']
            d_wdkv = h_norm.reshape(-1, D).T @ d_kv_latent.reshape(-1, d_c)  # [D, d_c]
            d_h_kv = d_kv_latent @ b['wdkv'].data.T  # [B, T, D]
            
            # Q projection gradient
            d_wq = h_norm.reshape(-1, D).T @ d_q.reshape(-1, H*hdim)  # [D, H*hdim]
            d_h_q = d_q.reshape(B, T, H*hdim) @ b['wq'].data.T  # [B, T, D]
            
            # RoPE projections gradient
            d_wqr = h_norm.reshape(-1, D).T @ layer['q_rope'].reshape(-1, H*d_rope)
            d_h_qr = layer['q_rope'].reshape(B, T, H*d_rope) @ b['wqr'].data.T
            
            d_h_attn = d_h_kv + d_h_q  # Combined hidden gradient from attention
            
            # Layer norm 1 backward
            d_h_from_attn = self._layernorm_backward(d_h_attn, h_norm, b['ln1_g'].data, b['ln1_b'].data)
            
            # Accumulate gradients
            d_h = d_h_from_attn + d_h_from_ffn
            
            # Store block gradients
            d_blocks.append({
                'wq': d_wq,
                'wdkv': d_wdkv,
                'wuk': d_wuk,
                'wuv': d_wuv,
                'wo': d_wo if 'd_wo' in dir() else np.zeros_like(b['wo'].data),
                'wqr': d_wqr if 'd_wqr' in dir() else np.zeros_like(b['wqr'].data),
                'wkr': np.zeros_like(b['wkr'].data),  # Simplified
                'ln1_g': np.zeros(D),
                'ln1_b': np.zeros(D),
                'ln2_g': np.zeros(D),
                'ln2_b': np.zeros(D),
                'ffn1': d_ffn1,
                'ffn1_b': d_ffn1_b,
                'ffn2': d_ffn2,
                'ffn2_b': d_ffn_b2,
            })
        
        # ── Embedding gradient ──
        x_input = cache['x_input']
        d_wte = np.zeros_like(self.params['wte'].data)
        for b_idx in range(B):
            for t in range(T):
                d_wte[x_input[b_idx, t], :] += d_h[b_idx, t]
        
        # ── Apply gradients ──
        self._apply_gradients(self.params, d_blocks, d_lm_head, d_wte, learning_rate)
        
        return True
    
    def _get_wo_grad(self, b):
        """Get output projection."""
        return b['wo'].data
    
    def _layernorm_backward(self, d_out, x, gamma, beta):
        """Layer norm backward pass."""
        eps = 1e-5
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_centered = x - mean
        std_inv = 1.0 / np.sqrt(var + eps)
        
        dx = d_out * gamma * std_inv
        
        # Gradient through normalization
        N = x.shape[-1]
        dx_norm = (1.0 / N) * (
            -np.sum(dx * x_centered, axis=-1, keepdims=True) * std_inv * std_inv * x_centered
            - np.sum(dx, axis=-1, keepdims=True)
        )
        
        return dx + dx_norm
    
    def _apply_gradients(self, params, d_blocks, d_lm_head, d_wte, lr):
        """Apply SGD with momentum-like update."""
        # Clip gradients
        def clip_grad(g, max_norm=1.0):
            norm = np.sqrt(np.sum(g ** 2))
            if norm > max_norm:
                g = g * (max_norm / norm)
            return g
        
        # Embedding
        self.params['wte'].grad = clip_grad(d_wte)
        self.params['wte'].data -= lr * self.params['wte'].grad
        
        # LM head
        self.params['lm_head'].grad = clip_grad(d_lm_head)
        self.params['lm_head'].data -= lr * self.params['lm_head'].grad
        
        # Per-block
        for i, d in enumerate(d_blocks):
            b = self.params['blocks'][i]
            for k in d.keys():
                if k in b:
                    g = clip_grad(d[k])
                    b[k].grad = g
                    b[k].data -= lr * g
    
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40):
        """
        Autoregressive generation.
        prompt_ids: [B, T] initial tokens
        Returns: [B, T + max_new_tokens]
        """
        B = prompt_ids.shape[0]
        generated = prompt_ids.copy()
        
        for step in range(max_new_tokens):
            # Forward pass on entire sequence
            logits, _ = self.forward(generated, is_training=False)
            
            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_vals = np.sort(next_logits, axis=-1)[:, -top_k:]
                threshold = top_k_vals[:, 0:1]
                next_logits = np.where(next_logits < threshold, -1e10, next_logits)
            
            # Sample
            probs = softmax(next_logits, axis=-1)
            next_token = np.array([
                np.random.choice(len(probs[0]), p=probs[0])
            ])[None, None]
            
            generated = np.concatenate([generated, next_token], axis=1)
            
            # Check EOS
            if next_token[0, 0] == 2:  # <EOS>
                break
        
        return generated
    
    def save(self, path):
        """Save model weights."""
        state = {
            'config': self.config,
            'params': {}
        }
        for name, p in self.params.items():
            if name == 'blocks':
                state['params']['blocks'] = []
                for b in self.params['blocks']:
                    block_state = {k: v.data for k, v in b.items()}
                    state['params']['blocks'].append(block_state)
            else:
                state['params'][name] = p.data
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path):
        """Load model weights."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(state['config'])
        for name, p in model.params.items():
            if name == 'blocks':
                for i, b in enumerate(model.params['blocks']):
                    for k in b.keys():
                        b[k].data = state['params']['blocks'][i][k]
            else:
                p.data = state['params'][name]
        
        return model
