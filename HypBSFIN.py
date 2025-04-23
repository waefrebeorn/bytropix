

# -*- coding: utf-8 -*-
"""
HyperHAKMEM-BSFIN Model (v3)

Integrates:
- HAKMEM-inspired computational enhancements (v2 features)
- Hyperbolic geometry concepts for patch representation (from Hyperbolic Category Discovery PDF)

Architecture:
1. HAKMEM BabylonIndex for entropy-based patching.
2. HAKMEM LocalEncoder for initial Euclidean patch encoding.
3. Projection to Euclidean space suitable for hyperbolic embedding.
4. Poincaré Clipping + Exponential Map to project into Hyperbolic Space (Poincaré Ball).
5. Logarithmic Map to project back to Euclidean Tangent Space.
6. Projection from Tangent Space to Complex Domain (Real/Imag).
7. HAKMEM Complex Positional Encoding.
8. Stack of HAKMEM Complex LayerNorm + HAKMEM EntangledInterferenceLayer operating on complex tangent space representations.
9. HAKMEM Complex To Real Projection (from complex tangent space).
10. Final Projection to Decoder Memory Dimension.
11. HAKMEM LocalDecoder for byte prediction using the final memory.
12. HAKMEM EnhancedSGD with Q-Learning for optimization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import math
import random
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque
import gc
import socket
import platform
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, is_initialized
from torch import amp  # For automatic mixed precision
from dataclasses import dataclass
import itertools
from tqdm import tqdm

# Try importing wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # Set wandb to None if not available
    WANDB_AVAILABLE = False


# Configure logging
logger = logging.getLogger("HyperHAKMEMBSFIN")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)

# =====================================================================
# Placeholder Classes (Replace with actual implementations if needed)
# =====================================================================
class GradientStats:
    """Tracks gradient statistics."""
    def __init__(self): self.reset()
    def reset(self):
        self.total_gradients = 0; self.clipped_gradients = 0
        self.max_gradient_norm = 0.0; self.sum_clip_ratios = 0.0
        self.step_stats = {}
    def record_gradient(self, norm, clipped, ratio): pass
    def get_step_stats(self): return {}
    def record_step(self, step): return {}

@dataclass
class SamplerConfig:
    """Configuration for entropy-based sampling."""
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

class ByteTokenizer:
    """Simple tokenizer for byte-level processing."""
    def encode(self, text: str) -> List[int]: return list(text.encode('utf-8'))
    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        valid_bytes = []
        for b in byte_sequence:
            val = b.item() if isinstance(b, torch.Tensor) else b
            if isinstance(val, (int, float)) and 0 <= int(val) <= 255: valid_bytes.append(int(val))
        return bytes(valid_bytes).decode('utf-8', errors='replace')

class ByteIterableDataset(IterableDataset):
     """ Minimal Placeholder for ByteIterableDataset """
     def __init__(self, npy_file_path: str, context_size: int = 128, data_fraction: float = 1.0):
         self.context_size = context_size
         self.num_samples = 1000 # Dummy value
         logger.info(f"Placeholder Dataset: Context={context_size}")
     def __len__(self): return self.num_samples
     def __iter__(self):
         for _ in range(self.num_samples):
             context = torch.randint(0, 256, (self.context_size,), dtype=torch.long)
             target = torch.randint(0, 256, (1,), dtype=torch.long).squeeze() # Single target byte
             yield context, target


# =====================================================================
# HAKMEM-Inspired Entropy Calculation (Helper for Patching)
# =====================================================================
class HAKMEMEntropyHelper:
    """ Encapsulates HAKMEM-inspired entropy calculation logic. """
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_cache = {}
        self.max_cache_size = max_cache_size
        self.log2_cache = {i: np.log2(i) if i > 0 else 0.0 for i in range(1, 257)}

    def _clean_cache(self):
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5)
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache: del self.entropy_cache[k]

    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...]]) -> float:
        cache_key = None
        if isinstance(byte_window, tuple):
            cache_key = byte_window
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
            if not byte_window: return 0.0
            byte_window_np = np.array(byte_window, dtype=np.uint8)
        elif isinstance(byte_window, np.ndarray):
            if byte_window.size == 0: return 0.0
            byte_window_np = byte_window
        else: return 0.0

        try:
            if not np.issubdtype(byte_window_np.dtype, np.integer):
                byte_window_np = byte_window_np.astype(np.uint8)
            byte_counts = np.bincount(byte_window_np, minlength=256)
            total_bytes = byte_counts.sum()
            if total_bytes == 0: return 0.0
            nonzero_counts = np.count_nonzero(byte_counts)
            if nonzero_counts <= 1: result = 0.0
            else:
                probs = byte_counts[byte_counts > 0] / total_bytes
                entropy = float(-np.sum(probs * np.log2(probs + 1e-9))) # Standard stable calculation
                result = entropy

            if cache_key is not None:
                self.entropy_cache[cache_key] = result
                self._clean_cache()
            return result
        except Exception as e:
            logger.warning(f"Error during entropy calculation: {e}")
            return 0.0

# =====================================================================
# 1. HAKMEM-Enhanced Babylon Index (Patching)
# =====================================================================
class HAKMEMBabylonIndex:
    """ Entropy-based index for byte sequence analysis and patching (HAKMEM-inspired). """
    def __init__(self, scales: List[int] = [3, 5, 7], max_cache_size: int = 50000, min_entropy_threshold: float = 0.5):
        self.scales = sorted(list(set(scales)))
        self.min_entropy_threshold = min_entropy_threshold
        self.entropy_helper = HAKMEMEntropyHelper(max_cache_size)
        logger.info(f"HAKMEMBabylonIndex initialized with scales: {self.scales}, entropy threshold: {min_entropy_threshold}")

    def _is_valid_utf8_boundary(self, byte_seq: Union[List[int], np.ndarray], boundary: int) -> bool:
        if boundary <= 0 or boundary >= len(byte_seq): return True
        return not (0x80 <= byte_seq[boundary] <= 0xBF)

    def find_patch_boundaries(self, byte_seq_tensor: torch.Tensor) -> List[int]:
        if byte_seq_tensor.numel() == 0: return []
        byte_seq_list = byte_seq_tensor.cpu().tolist()
        seq_len = len(byte_seq_list)
        min_scale = min(self.scales, default=1)
        if seq_len < min_scale: return []

        window_size = min(max(self.scales, default=16), seq_len // 2, 64)
        window_size = max(window_size, min_scale)
        entropies = []
        for i in range(seq_len - window_size + 1):
            window_tuple = tuple(byte_seq_list[i : i + window_size])
            entropy = self.entropy_helper.compute_entropy(window_tuple)
            entropies.append((i, entropy))

        potential_boundaries = set()
        entropies.sort(key=lambda x: x[1], reverse=True) # High entropy first
        num_boundaries_target = max(1, seq_len // max(64, window_size*2)) # Fewer boundaries for stability
        selected_count = 0

        for start_pos, entropy_val in entropies:
            boundary_candidate = start_pos # Boundary marks start of high entropy window
            if entropy_val > self.min_entropy_threshold and selected_count < num_boundaries_target * 2:
                if self._is_valid_utf8_boundary(byte_seq_list, boundary_candidate):
                    potential_boundaries.add(boundary_candidate)
                    selected_count += 1

        final_boundaries = sorted([b for b in list(potential_boundaries) if 0 < b < seq_len])
        min_patch_size = max(8, window_size // 4) # Minimum patch size
        merged_boundaries = []
        if final_boundaries:
            last_boundary = 0
            for b in final_boundaries:
                if b - last_boundary >= min_patch_size:
                    merged_boundaries.append(b)
                    last_boundary = b
            final_boundaries = merged_boundaries

        return final_boundaries

    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[torch.Tensor]:
        if byte_seq_tensor.numel() == 0: return []
        if byte_seq_tensor.dim() != 1:
            if byte_seq_tensor.dim() == 2 and byte_seq_tensor.size(0) == 1: byte_seq_tensor = byte_seq_tensor.squeeze(0)
            else: raise ValueError(f"create_patches expects a 1D tensor, got shape {byte_seq_tensor.shape}")

        boundaries = self.find_patch_boundaries(byte_seq_tensor)
        patches = []
        start_idx = 0
        seq_len = byte_seq_tensor.size(0)
        for end_idx in boundaries:
            if start_idx < end_idx <= seq_len:
                patch = byte_seq_tensor[start_idx:end_idx]
                if patch.numel() > 0: patches.append(patch)
                start_idx = end_idx
        if start_idx < seq_len:
            final_patch = byte_seq_tensor[start_idx:]
            if final_patch.numel() > 0: patches.append(final_patch)
        return patches

    @torch.no_grad()
    def reset_context(self):
        self.entropy_helper.entropy_cache = {}

# =====================================================================
# 2. HAKMEM-Enhanced Cross Attention Block
# =====================================================================
class HAKMEMCrossAttentionBlock(nn.Module):
    """ Enhanced Cross-Attention block (HAKMEM-inspired). """
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Head adjustment logic (same as before)
        original_num_heads = num_heads
        if hidden_size == 0: raise ValueError("hidden_size cannot be zero")
        if num_heads <= 0 : num_heads = max(1, hidden_size // 64) # Default if num_heads invalid

        if hidden_size % num_heads != 0:
            possible_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
            if not possible_heads: num_heads = 1 # Fallback to 1 head
            else: num_heads = possible_heads[0]
            logger.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size); nn.init.normal_(self.q_proj.weight, std=0.02)
        self.k_proj = nn.Linear(hidden_size, hidden_size); nn.init.normal_(self.k_proj.weight, std=0.02)
        self.v_proj = nn.Linear(hidden_size, hidden_size); nn.init.normal_(self.v_proj.weight, std=0.02)
        self.out_proj = nn.Linear(hidden_size, hidden_size); nn.init.normal_(self.out_proj.weight, std=0.02)
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_queries, _ = queries.size()
        seq_len_kv = keys_values.size(1)
        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)
        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Use scaled_dot_product_attention for efficiency
        # Prepare mask: Needs shape [B, Nq, Nkv] or broadcastable, True means MASK
        attn_mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2: # [B, Nkv] -> [B, 1, 1, Nkv]
                attn_mask_bool = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            elif attention_mask.dim() == 3: # [B, Nq, Nkv] -> [B, 1, Nq, Nkv]
                attn_mask_bool = attention_mask.unsqueeze(1).bool()
            # scaled_dot_product_attention expects True=KEEP mask, so invert if needed
            # However, passing bool mask with True=MASK works as of recent PyTorch versions
            # Keep mask with True=MASK for consistency with manual calculation

        # Use flash attention if available and mask is suitable
        use_flash = hasattr(F, 'scaled_dot_product_attention')
        if use_flash:
             try:
                  # Pass boolean mask directly (True=MASK)
                  output = F.scaled_dot_product_attention(
                      q, k, v,
                      attn_mask=attn_mask_bool, # Pass boolean mask directly if using PyTorch >= 2.0
                      dropout_p=self.dropout.p if self.training else 0.0,
                      is_causal=False # Not causal for cross-attention
                  )
             except Exception as e:
                  # Fallback for older PyTorch or specific mask issues
                  # logger.warning(f"Flash attention failed: {e}. Falling back.")
                  use_flash = False
                  # Invert mask for manual fill: True=KEEP -> False=MASK
                  #attn_mask_manual = ~attn_mask_bool if attn_mask_bool is not None else None


        if not use_flash: # Manual calculation
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask_bool is not None:
                 # Ensure mask is broadcastable
                 if attn_mask_bool.dim() == 4 and attn_mask_bool.shape[1] == 1: # Expand head dim
                      attn_mask_bool = attn_mask_bool.expand(-1, self.num_heads, -1, -1)
                 scores = scores.masked_fill(attn_mask_bool, float('-inf'))
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = torch.nan_to_num(attn_probs)
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        return output

# =====================================================================
# 3. HAKMEM-Enhanced Local Encoder
# =====================================================================
class HAKMEMLocalEncoder(nn.Module):
    """ Enhanced LocalEncoder with HAKMEM-inspired N-gram features and pooling. """
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1,
                 n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__()
        self.hidden_size=hidden_size
        self.byte_embeddings=nn.Embedding(256, hidden_size); nn.init.normal_(self.byte_embeddings.weight, std=1.0/math.sqrt(hidden_size))
        self.n_gram_sizes = sorted(list(set(n_gram_sizes))) if n_gram_sizes else []
        self.n_gram_vocab_size = n_gram_vocab_size
        self.n_gram_embeddings = None
        if self.n_gram_sizes:
            self.n_gram_embeddings=nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes})
            for emb in self.n_gram_embeddings.values(): nn.init.normal_(emb.weight, std=0.02)
            logger.info(f"HAKMEMLocalEncoder using N-grams: {self.n_gram_sizes} with vocab size {n_gram_vocab_size}")
            # Prime multipliers for hashing
            self.hash_multipliers = { n: [self._get_prime(n * 10 + i) for i in range(n)] for n in self.n_gram_sizes }

        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=num_heads,dim_feedforward=hidden_size*4,dropout=dropout,batch_first=True, activation=F.gelu)
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_pooling_attention=HAKMEMCrossAttentionBlock(hidden_size, num_heads, dropout) # Use HAKMEM attention
        self.patch_query=nn.Parameter(torch.randn(1, 1, hidden_size))
        self.norm=nn.LayerNorm(hidden_size,eps=1e-6)
        self.dropout=nn.Dropout(dropout)

    def _get_prime(self, n): # Simple prime finder
        def is_prime(num):
            if num < 2: return False
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0: return False
            return True
        num = n
        while True:
            if is_prime(num): return num
            num += 1

    def _get_n_gram_hashes(self, byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        batch_size, seq_len = byte_sequence.shape
        if seq_len < n: return torch.empty(batch_size, 0, dtype=torch.long, device=byte_sequence.device)
        windows = byte_sequence.unfold(1, n, 1)
        multipliers = self.hash_multipliers.get(n, [31]*n) # Fallback
        weighted = windows.long() * torch.tensor(multipliers, device=windows.device).view(1, 1, n)
        hashes = weighted.sum(dim=-1)
        hashes = ((hashes << 5) + hashes) ^ (hashes >> 2) # Bit mixing
        return hashes % self.n_gram_vocab_size

    def forward(self, patches: List[torch.Tensor]) -> torch.Tensor:
        if not patches:
            device = next(self.parameters()).device
            return torch.empty((1, 0, self.hidden_size), device=device)
        device = patches[0].device
        patch_representations = []
        for patch_bytes in patches:
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue
            patch_bytes_batched = patch_bytes.unsqueeze(0)
            x = self.byte_embeddings(patch_bytes_batched)
            if self.n_gram_embeddings:
                n_gram_features = torch.zeros_like(x)
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_batched, n)
                        if n_gram_hashes.numel() > 0:
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes)
                            num_windows = ngram_embeds.size(1)
                            if num_windows < patch_len:
                                padding = torch.zeros(1, patch_len - num_windows, self.hidden_size, device=device)
                                ngram_embeds_padded = torch.cat([ngram_embeds, padding], dim=1)
                            else: ngram_embeds_padded = ngram_embeds[:, :patch_len, :]
                            n_gram_features += ngram_embeds_padded
                x = x + n_gram_features # Simple addition
            x = self.dropout(x)
            processed_bytes = self.transformer(x) # Causal mask optional for encoder
            batch_query = self.patch_query
            patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes)
            patch_representations.append(patch_repr)
        if not patch_representations: return torch.empty((1, 0, self.hidden_size), device=device)
        patches_combined = torch.cat(patch_representations, dim=1)
        return self.norm(patches_combined)

# =====================================================================
# 4. Hyperbolic Geometry Utilities
# =====================================================================
class HyperbolicUtils:
    """ Utility functions for Poincaré Ball operations based on the PDF. """
    @staticmethod
    def poincare_clip(x: torch.Tensor, c: float, radius: float = 1.0) -> torch.Tensor:
        """ Clip features to prevent landing exactly on the boundary. Eq before (7). """
        if c <= 0: return x # Not hyperbolic space
        max_norm = radius / math.sqrt(c) # Max Euclidean norm for curvature c, radius 1
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8
        # Scale factor slightly less than 1 to avoid boundary
        scale_factor = max_norm * 0.9999
        cond = x_norm > scale_factor
        clipped_x = torch.where(cond, x / x_norm * scale_factor, x)
        return clipped_x

    @staticmethod
    def exponential_map(v: torch.Tensor, c: float) -> torch.Tensor:
        """ Projects Euclidean vector v (tangent at origin) to Poincaré ball. Eq (7). """
        # Assumes origin o = 0
        if c <= 0: return v
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True) + 1e-8
        sqrt_c = math.sqrt(c)
        # Note: PDF Eq 7 seems to have sqrt(c) inside tanh and outside.
        # Ref: https://arxiv.org/pdf/1805.08207.pdf (Eq 4) uses lambda_x * v
        # Ref: https://arxiv.org/pdf/1910.12933.pdf (Eq 5) uses tanh(sqrt(c)*||v||)/sqrt(c) * v/||v||
        # Let's use the structure from the second reference (seems more standard)
        mapped_v = torch.tanh(sqrt_c * v_norm) / (sqrt_c + 1e-9) * (v / v_norm)
        # Ensure numerical stability if v is near zero
        mapped_v = torch.where(v_norm < 1e-8, torch.zeros_like(v), mapped_v)
        return mapped_v

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c: float) -> torch.Tensor:
        """ Projects point y from Poincaré ball to tangent space at origin. Inverse of Exp Map."""
        # Assumes origin o = 0
        if c <= 0: return y
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True) + 1e-8
        sqrt_c = math.sqrt(c)
        # Inverse of tanh(sqrt(c)*norm)/sqrt(c) * unit_vec is atanh(sqrt(c)*norm)/sqrt(c) * unit_vec
        # Clamp input to atanh to avoid domain errors: [-1+eps, 1-eps]
        arctanh_input = torch.clamp(sqrt_c * y_norm, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        mapped_y = torch.atanh(arctanh_input) / (sqrt_c + 1e-9) * (y / y_norm)
        # Ensure numerical stability if y is near zero
        mapped_y = torch.where(y_norm < 1e-8, torch.zeros_like(y), mapped_y)
        return mapped_y

    # --- Optional: Möbius operations (if needed later) ---
    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """ Möbius addition. Eq (5). """
        if c <= 0: return x + y
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2
        return num / (denom + 1e-8)

    @staticmethod
    def hyperbolic_distance_sq(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """ Squared Hyperbolic distance. Based on Eq (6). """
        if c <= 0: return torch.sum((x-y)**2, dim=-1)
        # dist = (2/sqrt(c)) * atanh(sqrt(c) * ||(-x) +_c y||)
        # dist^2 = (4/c) * atanh^2(...)
        mobius_diff_norm = torch.norm(HyperbolicUtils.mobius_add(-x, y, c), p=2, dim=-1) + 1e-8
        # Clamp input to atanh
        atanh_input = torch.clamp(math.sqrt(c) * mobius_diff_norm, max=1.0 - 1e-7)
        dist_sq = (4.0 / (c + 1e-9)) * torch.atanh(atanh_input)**2
        return dist_sq

    @staticmethod
    def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
         return torch.sqrt(HyperbolicUtils.hyperbolic_distance_sq(x, y, c) + 1e-8)


# =====================================================================
# 5. HAKMEM-Enhanced Complex Number Operations & LayerNorm
# =====================================================================
class HAKMEMComplexOperations: # Minimal version (same as before)
    @staticmethod
    def complex_matmul(a_real, a_imag, b_real, b_imag):
        k1 = torch.matmul(a_real, b_real); k2 = torch.matmul(a_imag, b_imag)
        k3 = torch.matmul(a_real + a_imag, b_real + b_imag)
        return k1 - k2, k3 - k1 - k2
    @staticmethod
    def complex_phase_shift(real, imag, phase_cos, phase_sin):
        return real * phase_cos - imag * phase_sin, real * phase_sin + imag * phase_cos
    @staticmethod
    def complex_norm(real, imag, epsilon=1e-6):
        return torch.sqrt(real.pow(2) + imag.pow(2) + epsilon)
    @staticmethod
    def complex_normalize(real, imag, epsilon=1e-6):
        norm = HAKMEMComplexOperations.complex_norm(real, imag, epsilon)
        return real / norm, imag / norm
    @staticmethod
    def complex_attention_scores(q_real, q_imag, k_real, k_imag, scale=1.0):
        attn_real, attn_imag = HAKMEMComplexOperations.complex_matmul(q_real, q_imag, k_real.transpose(-2, -1), k_imag.transpose(-2, -1))
        attn_mag = HAKMEMComplexOperations.complex_norm(attn_real * scale, attn_imag * scale)
        return attn_real * scale, attn_imag * scale, attn_mag

class HAKMEMComplexLayerNorm(nn.Module): # Minimal version (same as before)
    def __init__(self, dim, eps=1e-5, coupled=True):
        super().__init__()
        self.real_norm = nn.LayerNorm(dim, eps=eps); self.imag_norm = nn.LayerNorm(dim, eps=eps)
        self.coupled = coupled
        if coupled:
            self.coupling_strength = nn.Parameter(torch.tensor(0.0))
            self.cross_gain_ri = nn.Parameter(torch.zeros(dim))
            self.cross_gain_ir = nn.Parameter(torch.zeros(dim))
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x; real_normed = self.real_norm(real); imag_normed = self.imag_norm(imag)
        if self.coupled:
            coupling = torch.sigmoid(self.coupling_strength) * 0.2
            real_out = real_normed + coupling * self.cross_gain_ir * imag_normed
            imag_out = imag_normed + coupling * self.cross_gain_ri * real_normed
            return real_out, imag_out
        else: return real_normed, imag_normed

# =====================================================================
# 6. HAKMEM-Enhanced Positional Encoding
# =====================================================================
class HAKMEMPositionalEncoding(nn.Module): # Minimal version (same as before)
    def __init__(self, dim, max_len=2048, phase_shift=True, learnable=True):
        super().__init__()
        if dim % 2 != 0: raise ValueError("PE dim must be even.")
        self.dim = dim; self.learnable = learnable; self.max_cache_len = max_len * 2
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term_base = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe_real = torch.zeros(max_len, dim); pe_real[:, 0::2] = torch.sin(position * div_term_base); pe_real[:, 1::2] = torch.cos(position * div_term_base)
        pe_imag = torch.zeros(max_len, dim); pe_imag[:, 0::2] = torch.cos(position * div_term_base); pe_imag[:, 1::2] = -torch.sin(position * div_term_base) # Phase shift = True
        self.register_buffer('pe_real_base', pe_real); self.register_buffer('pe_imag_base', pe_imag); self.register_buffer('div_term_base', div_term_base)
        if learnable:
            self.real_scale = nn.Parameter(torch.ones(1, 1, dim)); self.imag_scale = nn.Parameter(torch.ones(1, 1, dim))
            self.real_shift = nn.Parameter(torch.zeros(1, 1, dim)); self.imag_shift = nn.Parameter(torch.zeros(1, 1, dim))
            self.frequency_scale_factors = nn.Parameter(torch.ones(dim // 2))
            self.freq_doubling = nn.Parameter(torch.tensor(0.0))
        self.position_cache = {}
    def _compute_position_efficient(self, seq_len, device):
        cache_key = (seq_len, str(device))
        if cache_key in self.position_cache: return self.position_cache[cache_key]
        current_max_len = self.pe_real_base.size(0)
        if seq_len > current_max_len: # Extend base PE if needed
            position = torch.arange(seq_len).unsqueeze(1).float().to(device)
            div_term = self.div_term_base.to(device)
            pe_real = torch.zeros(seq_len, self.dim, device=device); pe_real[:, 0::2] = torch.sin(position * div_term); pe_real[:, 1::2] = torch.cos(position * div_term)
            pe_imag = torch.zeros(seq_len, self.dim, device=device); pe_imag[:, 0::2] = torch.cos(position * div_term); pe_imag[:, 1::2] = -torch.sin(position * div_term)
        else: pe_real = self.pe_real_base[:seq_len].to(device); pe_imag = self.pe_imag_base[:seq_len].to(device)
        if self.learnable:
            scaled_div_term = self.div_term_base.to(device) * torch.clamp(self.frequency_scale_factors, min=1e-2)
            position = torch.arange(seq_len, device=device).unsqueeze(1).float()
            doubling_factor = torch.sigmoid(self.freq_doubling) * 0.5
            pos_modulation = 1.0 + doubling_factor * (position / max(1, seq_len)) # Prevent div by zero
            angles = pos_modulation * position * scaled_div_term
            pe_real_learn = torch.zeros_like(pe_real); pe_imag_learn = torch.zeros_like(pe_imag)
            pe_real_learn[:, 0::2] = torch.sin(angles); pe_real_learn[:, 1::2] = torch.cos(angles)
            pe_imag_learn[:, 0::2] = torch.cos(angles); pe_imag_learn[:, 1::2] = -torch.sin(angles)
            pe_real = pe_real_learn * self.real_scale + self.real_shift
            pe_imag = pe_imag_learn * self.imag_scale + self.imag_shift
        if len(self.position_cache) > self.max_cache_len: self.position_cache.pop(next(iter(self.position_cache)))
        self.position_cache[cache_key] = (pe_real, pe_imag)
        return pe_real, pe_imag
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x; seq_len = real.size(1); device = real.device
        pe_real, pe_imag = self._compute_position_efficient(seq_len, device)
        return real + pe_real.unsqueeze(0), imag + pe_imag.unsqueeze(0)

# =====================================================================
# 7. HAKMEM-Enhanced Entangled Interference Layer
# =====================================================================
class HAKMEMEntangledInterferenceLayer(nn.Module): # Minimal version (same as before)
    def __init__(self, dim, heads=8, dropout=0.1, interference_type="quantum", use_entanglement=True, noise_scale=0.05, use_rotary=True, adaptive_attention=True):
        super().__init__()
        if dim % heads != 0: raise ValueError("dim must be divisible by heads")
        self.dim = dim; self.heads = heads; self.head_dim = dim // heads; self.dropout = dropout
        self.interference_type = interference_type; self.use_entanglement = use_entanglement
        self.noise_scale = noise_scale; self.use_rotary = use_rotary; self.adaptive_attention = adaptive_attention
        self.phase_shifts = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        if use_entanglement:
            entangle_init = torch.eye(heads) * 0.8; entangle_init += torch.diag(torch.ones(heads-1)*0.1, diagonal=1); entangle_init += torch.diag(torch.ones(heads-1)*0.1, diagonal=-1); entangle_init[0,-1]=0.1; entangle_init[-1,0]=0.1
            self.entanglement_matrix = nn.Parameter(entangle_init)
        else: self.entanglement_matrix = None
        self.q_real = nn.Linear(dim, dim); self.k_real = nn.Linear(dim, dim); self.v_real = nn.Linear(dim, dim)
        self.q_imag = nn.Linear(dim, dim); self.k_imag = nn.Linear(dim, dim); self.v_imag = nn.Linear(dim, dim)
        self.out_real = nn.Linear(dim, dim); self.out_imag = nn.Linear(dim, dim)
        if use_rotary:
            self.rotary_dim = max(16, self.head_dim // 2); self.rotary_dim -= self.rotary_dim % 2
            exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float64) / self.rotary_dim
            base_freqs = 10000.0**(-exponent)
            if adaptive_attention: self.rotary_freqs = nn.Parameter(base_freqs.float())
            else: self.register_buffer('rotary_freqs', base_freqs.float(), persistent=False)
        else: self.rotary_dim = 0
        self.interference_strength = nn.Parameter(torch.tensor(0.0))
        if adaptive_attention: self.attention_temperature = nn.Parameter(torch.tensor(0.0))
        else: self.register_buffer('attention_temperature', torch.tensor(1.0), persistent=False)
        self.circle_epsilon = nn.Parameter(torch.tensor(-2.0))
        self.attn_dropout = nn.Dropout(dropout); self.resid_dropout = nn.Dropout(dropout); self.saved_attn_weights = None
        self.complex_ops = HAKMEMComplexOperations()
    def _compute_rotary_embeddings(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim <= 0: return None, None
        position = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = self.rotary_freqs.to(device=device, dtype=torch.float32)
        angles = torch.outer(position, freqs)
        cos_emb = torch.cos(angles).unsqueeze(0).unsqueeze(1)
        sin_emb = torch.sin(angles).unsqueeze(0).unsqueeze(1)
        return cos_emb, sin_emb
    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim <= 0 or cos_emb is None: return x
        rot_dim = self.rotary_dim
        x_rot = x[..., :rot_dim]; x_pass = x[..., rot_dim:]
        x_rot = x_rot.reshape(*x_rot.shape[:-1], -1, 2)
        cos = cos_emb.unsqueeze(-1); sin = sin_emb.unsqueeze(-1)
        x1 = x_rot[..., 0].unsqueeze(-1); x2 = x_rot[..., 1].unsqueeze(-1)
        x_rot_out = torch.zeros_like(x_rot)
        x_rot_out[..., 0] = (x1 * cos - x2 * sin).squeeze(-1)
        x_rot_out[..., 1] = (x1 * sin + x2 * cos).squeeze(-1)
        x_rot_out = x_rot_out.flatten(start_dim=-2)
        return torch.cat([x_rot_out, x_pass], dim=-1)
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x; batch_size, seq_len, _ = real.shape; device = real.device
        q_r = self.q_real(real).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_r = self.k_real(real).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_r = self.v_real(real).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        q_i = self.q_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_i = self.k_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_i = self.v_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        if self.training and self.noise_scale > 0:
            noise_r = torch.randn_like(q_r) * self.noise_scale; noise_i = torch.randn_like(q_i) * self.noise_scale
            q_r, q_i = q_r + noise_r, q_i + noise_i; k_r += torch.randn_like(k_r) * self.noise_scale; k_i += torch.randn_like(k_i) * self.noise_scale
        if self.use_rotary:
            cos_emb, sin_emb = self._compute_rotary_embeddings(seq_len, device)
            q_r = self._apply_rotary_pos_emb(q_r, cos_emb, sin_emb); k_r = self._apply_rotary_pos_emb(k_r, cos_emb, sin_emb)
            q_i = self._apply_rotary_pos_emb(q_i, cos_emb, sin_emb); k_i = self._apply_rotary_pos_emb(k_i, cos_emb, sin_emb)
        if self.use_entanglement and self.entanglement_matrix is not None:
            ent_matrix = self.entanglement_matrix.to(device)
            q_r=torch.einsum("bhsd,hx->bxsd", q_r, ent_matrix); q_i=torch.einsum("bhsd,hx->bxsd", q_i, ent_matrix)
            k_r=torch.einsum("bhsd,hx->bxsd", k_r, ent_matrix); k_i=torch.einsum("bhsd,hx->bxsd", k_i, ent_matrix)
        phase_cos = torch.cos(self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device); phase_sin = torch.sin(self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device)
        q_r, q_i = self.complex_ops.complex_phase_shift(q_r, q_i, phase_cos, phase_sin); k_r, k_i = self.complex_ops.complex_phase_shift(k_r, k_i, phase_cos, phase_sin)
        scale = 1.0 / math.sqrt(self.head_dim); attn_r, attn_i, attn_mag = self.complex_ops.complex_attention_scores(q_r, q_i, k_r, k_i, scale)
        epsilon = torch.sigmoid(self.circle_epsilon) * 0.03; attn_r_refined = attn_r + epsilon * attn_i; attn_i_refined = attn_i - epsilon * attn_r
        attn_mag = self.complex_ops.complex_norm(attn_r_refined, attn_i_refined)
        final_mask = None
        if attention_mask is not None:
             if attention_mask.dim() == 2: final_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool() # [B, 1, 1, S_k]
             elif attention_mask.dim() == 3: final_mask = attention_mask.unsqueeze(1).bool() # [B, 1, S_q, S_k]
             else: logger.warning(f"Interference mask dim {attention_mask.dim()} unsupported.")
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1)
        final_mask = final_mask | causal_mask if final_mask is not None else causal_mask
        if final_mask is not None: attn_mag = attn_mag.masked_fill(final_mask, float('-inf'))
        if self.adaptive_attention: temp = torch.exp(self.attention_temperature).clamp(min=0.1)
        else: temp = self.attention_temperature
        strength = torch.sigmoid(self.interference_strength)
        attn_weights = F.softmax((attn_mag * strength) / temp, dim=-1); attn_weights = torch.nan_to_num(attn_weights); attn_weights = self.attn_dropout(attn_weights)
        self.saved_attn_weights = attn_weights.detach().cpu()
        out_r, out_i = self.complex_ops.complex_matmul(attn_weights, torch.zeros_like(attn_weights), v_r, v_i)
        out_r = out_r.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim); out_i = out_i.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out_r = self.out_real(out_r); out_i = self.out_imag(out_i); out_r = self.resid_dropout(out_r); out_i = self.resid_dropout(out_i)
        return (out_r, out_i)

# =====================================================================
# 8. HAKMEM-Enhanced Complex-to-Real Projection
# =====================================================================
class HAKMEMComplexToRealProjection(nn.Module): # Minimal version (same as before)
    def __init__(self, complex_dim: int, output_dim: int, method: str = "hakmem_enhanced"):
        super().__init__(); self.method = method; self.complex_dim = complex_dim; self.output_dim = output_dim; self.complex_ops = HAKMEMComplexOperations()
        if method == "hakmem_enhanced":
            self.magnitude_proj = nn.Linear(complex_dim, output_dim // 2); self.phase_proj = nn.Linear(complex_dim * 2, output_dim // 2); self.combined_proj = nn.Linear(output_dim, output_dim)
            nn.init.xavier_uniform_(self.magnitude_proj.weight); nn.init.xavier_uniform_(self.phase_proj.weight); nn.init.xavier_uniform_(self.combined_proj.weight)
        else: self.proj = nn.Linear(complex_dim*2 if method=="concat" else complex_dim, output_dim); nn.init.xavier_uniform_(self.proj.weight)
        self.activation_scale = nn.Parameter(torch.tensor(0.0))
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        real, imag = x
        if self.method == "hakmem_enhanced":
            magnitude = self.complex_ops.complex_norm(real, imag); mag_features = F.gelu(self.magnitude_proj(magnitude))
            norm_real, norm_imag = self.complex_ops.complex_normalize(real, imag); phase_input = torch.cat([norm_real, norm_imag], dim=-1); phase_features = F.gelu(self.phase_proj(phase_input))
            combined_features = torch.cat([mag_features, phase_features], dim=-1); scale = torch.sigmoid(self.activation_scale + 1.0); return self.combined_proj(combined_features) * scale
        elif self.method == "concat": combined = torch.cat([real, imag], dim=-1); return self.proj(combined) # Simplified concat
        elif self.method == "magnitude": magnitude = self.complex_ops.complex_norm(real, imag); return self.proj(magnitude) # Simplified mag
        else: raise ValueError(f"Unknown method {self.method}")

# =====================================================================
# 9. HAKMEM-Enhanced Local Decoder
# =====================================================================
class HAKMEMLocalDecoder(nn.Module): # Minimal version (same as before)
    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, use_hierarchical_pred: bool = True):
        super().__init__(); self.hidden_size = hidden_size; self.use_hierarchical = use_hierarchical_pred
        self.byte_embeddings = nn.Embedding(256, hidden_size); nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))
        self.memory_projection1 = nn.Linear(global_hidden_size, hidden_size * 2); self.memory_projection2 = nn.Linear(hidden_size * 2, hidden_size); nn.init.xavier_uniform_(self.memory_projection1.weight); nn.init.xavier_uniform_(self.memory_projection2.weight)
        self.memory_compression = nn.Sequential(nn.LayerNorm(hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, activation=F.gelu)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        if self.use_hierarchical:
            self.byte_class_pred = nn.Linear(hidden_size, 16); nn.init.normal_(self.byte_class_pred.weight, std=0.02)
            self.byte_specific_pred = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(16)])
            for layer in self.byte_specific_pred: nn.init.normal_(layer.weight, std=0.02 / math.sqrt(16))
        else: self.byte_pred = nn.Linear(hidden_size, 256); nn.init.normal_(self.byte_pred.weight, std=0.02)
        self.dropout_embed = nn.Dropout(dropout)
    def _create_bool_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
         return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, tgt_len = tgt_byte_seq.size(); device = tgt_byte_seq.device
        if tgt_len == 0: return torch.zeros((batch_size, 0, 256), device=device)
        tgt_embed = self.dropout_embed(self.byte_embeddings(tgt_byte_seq))
        memory_mid = F.gelu(self.memory_projection1(memory)); projected_memory = self.memory_projection2(memory_mid)
        compressed_memory = self.memory_compression(projected_memory)
        if tgt_mask is None: tgt_mask = self._create_bool_mask(tgt_len, device)
        output = self.transformer(tgt=tgt_embed, memory=compressed_memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.use_hierarchical:
            byte_class_logits = self.byte_class_pred(output)
            log_class_probs = F.log_softmax(byte_class_logits, dim=-1)
            log_specific_probs_stacked = torch.stack([F.log_softmax(pred(output), dim=-1) for pred in self.byte_specific_pred], dim=2)
            combined_log_probs = log_class_probs.unsqueeze(-1) + log_specific_probs_stacked
            byte_logits = combined_log_probs.view(batch_size, tgt_len, 256)
        else: byte_logits = self.byte_pred(output)
        return byte_logits

# =====================================================================
# 10. HAKMEM-Enhanced Q-Learning Controller & Optimizer
# =====================================================================
class HAKMEMQController: # Minimal version (same as before)
    def __init__(self, learning_rate: float=0.02, discount: float=0.97, epsilon: float=0.15, epsilon_decay: float=0.9995, min_epsilon: float=0.02, lr_scale_options: List[float]=None, momentum_scale_options: List[float]=None, max_q_table_size: int=15000):
        self.q_table = {}; self.alpha = learning_rate; self.gamma = discount; self.epsilon = epsilon; self.min_epsilon = min_epsilon; self.epsilon_decay = epsilon_decay
        self.prev_loss = None; self.prev_state = None; self.prev_action = None
        if lr_scale_options is None: lr_scale_options = [0.90, 0.94, 0.97, 0.99, 1.0, 1.01, 1.03, 1.06, 1.10]
        if momentum_scale_options is None: momentum_scale_options = [0.95, 0.97, 0.99, 0.995, 1.0, 1.005, 1.01, 1.03, 1.05]
        self.action_ranges = {'lr_scale': np.array(lr_scale_options), 'momentum_scale': np.array(momentum_scale_options)}
        self.loss_window = deque(maxlen=10); self.grad_norm_window = deque(maxlen=10); self.lr_window = deque(maxlen=5); self.momentum_window = deque(maxlen=5); self.performance_window = deque(maxlen=30)
        self.stable_steps = 0; self.oscillation_counter = 0; self.prev_actions = deque(maxlen=5); self.max_q_table_size = max_q_table_size; self.q_table_access_count = {}; self.flow_coefficient = 0.05; self.oscillation_penalty = 0.2
    def get_state(self, lr, momentum, grad_norm, loss): # Simplified state logic
        if loss is None or grad_norm is None or len(self.loss_window) < 5: return None
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm); self.lr_window.append(lr); self.momentum_window.append(momentum)
        loss_trend_bin = 2; grad_norm_bin = 2; lr_bin = 2; momentum_bin = 1; oscillation_bin = 0 # Default bins
        # Simplified calculations for brevity
        state = (loss_trend_bin, grad_norm_bin, oscillation_bin, lr_bin, momentum_bin)
        self.q_table_access_count[state] = self.q_table_access_count.get(state, 0) + 1
        return state
    def compute_reward(self, current_loss, prev_loss, grad_norm): # Simplified reward
        if current_loss is None or prev_loss is None or grad_norm is None: return 0.0
        reward = np.tanh(((prev_loss - current_loss) / (abs(prev_loss) + 1e-6)) * 10)
        # Add simple grad norm penalty/reward
        if grad_norm > 5.0: reward -= 0.1
        elif grad_norm < 0.1: reward += 0.05
        return float(np.clip(reward, -1.0, 1.0))
    def choose_action(self, state): # Simplified action choice
        if state is None: return None
        if state not in self.q_table: self.q_table[state] = {p: np.zeros(len(s)) for p, s in self.action_ranges.items()}; self._manage_q_table_size()
        action = {}; current_epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        for param, space in self.action_ranges.items():
            if random.random() < current_epsilon: chosen_idx = random.randrange(len(space))
            else:
                 q_values = self.q_table[state][param]
                 if np.any(np.isfinite(q_values)): max_q = np.nanmax(q_values); best_indices = np.where(np.isclose(q_values, max_q))[0]; chosen_idx = np.random.choice(best_indices) if len(best_indices)>0 else random.randrange(len(space))
                 else: chosen_idx = random.randrange(len(space))
            action[param] = float(space[chosen_idx])
        self.prev_actions.append(action.copy())
        return action
    def update(self, state, action, reward, next_state): # Simplified update
        if state is None or next_state is None or action is None: return
        if next_state not in self.q_table: self.q_table[next_state] = {p: np.zeros(len(s)) for p, s in self.action_ranges.items()}; self._manage_q_table_size()
        for param, value in action.items():
            space = self.action_ranges[param]; action_idx = np.abs(space - value).argmin()
            if not np.isclose(space[action_idx], value): continue
            current_q = self.q_table[state][param][action_idx]; next_q_values = self.q_table[next_state][param]
            max_future_q = np.nanmax(next_q_values) if np.any(np.isfinite(next_q_values)) else 0.0
            td_error = reward + self.gamma * max_future_q - current_q; adaptive_alpha = min(0.5, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error))))
            self.q_table[state][param][action_idx] += adaptive_alpha * td_error
    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            try:
                if self.q_table_access_count: least_accessed_state = min(self.q_table_access_count, key=self.q_table_access_count.get); del self.q_table[least_accessed_state]; del self.q_table_access_count[least_accessed_state]
                elif self.q_table: random_state = random.choice(list(self.q_table.keys())); del self.q_table[random_state]
            except Exception: pass # Ignore errors in pruning

class HAKMEMEnhancedSGD(torch.optim.Optimizer): # Minimal version (same as before)
    def __init__(self, params, lr=0.003, momentum=0.9, weight_decay=0.005, max_grad_norm=1.0, q_learning_config={}):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay); super().__init__(params, defaults)
        self.q_controller = HAKMEMQController(**q_learning_config); self.max_grad_norm = max_grad_norm; self._step_count = 0; self.current_loss = None; self.flow_enabled = True; self.flow_coefficient = 0.1; self.flow_momentum = 0.95
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad: self.state[p] = {'momentum_buffer': torch.zeros_like(p), 'grad_avg': torch.zeros_like(p)}
    def zero_grad(self, set_to_none=True): super().zero_grad(set_to_none=set_to_none)
    def set_current_loss(self, loss): self.current_loss = loss; self.q_controller.prev_loss = loss
    @torch.no_grad()
    def step(self, closure=None):
        loss = None; grad_norm_avg = self._get_average_grad_norm(); q_action = None
        if self.current_loss is not None and grad_norm_avg is not None:
            current_lr = self.param_groups[0]['lr']; current_momentum = self.param_groups[0]['momentum']
            q_state = self.q_controller.get_state(lr=current_lr, momentum=current_momentum, grad_norm=grad_norm_avg, loss=self.current_loss)
            if self.q_controller.prev_state is not None and self.q_controller.prev_action is not None and q_state is not None:
                reward = self.q_controller.compute_reward(self.current_loss, self.q_controller.prev_loss, grad_norm_avg)
                self.q_controller.update(self.q_controller.prev_state, self.q_controller.prev_action, reward, q_state)
            q_action = self.q_controller.choose_action(q_state)
            if q_action is not None:
                for group in self.param_groups:
                    lr_scale = q_action.get('lr_scale', 1.0); mom_scale = q_action.get('momentum_scale', 1.0)
                    group['lr'] = float(np.clip(group['lr'] * lr_scale, 1e-8, 0.1))
                    group['momentum'] = float(np.clip(group['momentum'] * mom_scale, 0.5, 0.999))
            self.q_controller.prev_state = q_state; self.q_controller.prev_action = q_action
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']; weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad; param_state = self.state[p]
                if weight_decay != 0: grad = grad.add(p, alpha=weight_decay)
                effective_lr = lr
                if self.flow_enabled and 'grad_avg' in param_state:
                    grad_avg = param_state['grad_avg']; grad_avg.mul_(self.flow_momentum).add_(grad, alpha=1 - self.flow_momentum)
                    if self._step_count > 10:
                         grad_flat = grad.flatten(); avg_flat = grad_avg.flatten(); grad_norm = torch.norm(grad_flat); avg_norm = torch.norm(avg_flat)
                         flow_factor = 1.0
                         if grad_norm > 1e-8 and avg_norm > 1e-8:
                              cosine_sim = torch.dot(grad_flat, avg_flat) / (grad_norm * avg_norm)
                              flow_factor = 1.0 + (cosine_sim.item() + 1) / 2 * self.flow_coefficient * (1 if cosine_sim > 0 else -1) # Simplified flow factor logic
                         effective_lr = lr * float(np.clip(flow_factor, 0.5, 1.5))
                buf = param_state['momentum_buffer']; buf.mul_(momentum).add_(grad)
                p.add_(buf, alpha=-effective_lr)
        self._step_count += 1; return loss
    def _get_average_grad_norm(self): # Simplified norm calculation
        total_norm_sq = 0.0; num_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                if torch.isfinite(p.grad).all():
                     param_norm = torch.norm(p.grad.detach()); total_norm_sq += param_norm.item()**2; num_params += 1
        return math.sqrt(total_norm_sq / num_params) if num_params > 0 else None

# =====================================================================
# 11. HyperHAKMEM BSFIN Model Definition
# =====================================================================
class HyperHAKMEMBSFINModel(nn.Module):
    """ Hyperbolic + HAKMEM Enhanced BSFIN Model. """
    def __init__(
        self,
        local_hidden_size: int = 256, complex_dim: int = 512, num_complex_layers: int = 8,
        num_complex_heads: int = 8, decoder_memory_dim: int = 1024, dropout: float = 0.15,
        context_window: int = 256, # For dataset config, not directly used here
        n_gram_sizes: List[int] = [3, 4], n_gram_vocab_size: int = 30000,
        sfin_noise_scale: float = 0.05, sfin_use_entanglement: bool = True, sfin_use_rotary: bool = True,
        projection_method: str = "hakmem_enhanced", # For complex->real within stack
        use_hierarchical_decoder: bool = True,
        # --- Hyperbolic Params ---
        hyperbolic_embedding_dim: int = 384, # Dimension of the Poincaré ball space
        curvature: float = 1.0, # Hyperbolic curvature (c > 0)
        clipping_radius: float = 1.0, # Clipping radius factor (usually 1.0)
    ):
        super().__init__()
        self.local_hidden_size = local_hidden_size
        self.complex_dim = complex_dim # Dimension of complex layers operating in tangent space
        self.decoder_memory_dim = decoder_memory_dim
        self.hyperbolic_embedding_dim = hyperbolic_embedding_dim
        self.curvature = curvature
        self.clipping_radius = clipping_radius # Store clipping value
        self.hyperbolic_utils = HyperbolicUtils()

        if self.curvature <= 0:
            logger.warning("Curvature is <= 0. Model will operate effectively in Euclidean space.")
        if complex_dim % num_complex_heads != 0:
            raise ValueError(f"complex_dim ({complex_dim}) must be divisible by num_complex_heads ({num_complex_heads})")

        # --- HAKMEM/Hyperbolic Components ---
        self.patcher = HAKMEMBabylonIndex(scales=n_gram_sizes)
        self.local_encoder = HAKMEMLocalEncoder(local_hidden_size, num_layers=1, num_heads=max(1, local_hidden_size//64), dropout=dropout, n_gram_sizes=n_gram_sizes, n_gram_vocab_size=n_gram_vocab_size)

        # --- Hyperbolic Mapping Layers ---
        self.projection_to_hyp_euclidean = nn.Linear(local_hidden_size, hyperbolic_embedding_dim)
        nn.init.xavier_uniform_(self.projection_to_hyp_euclidean.weight)
        # No learnable affine after log_map needed, just use the tangent space directly

        # --- Complex Processing Stack (Operating on Tangent Space Vectors) ---
        # Project tangent space vector to complex domain (Real/Imag)
        self.tangent_to_complex = RealToComplexProjection(hyperbolic_embedding_dim, complex_dim) # HAKMEM/Original projection utility

        self.complex_pos_encoding = HAKMEMPositionalEncoding(complex_dim, max_len=2048, learnable=True)
        self.complex_norm_in = HAKMEMComplexLayerNorm(complex_dim, coupled=True)
        self.complex_interference_layers = nn.ModuleList([
            HAKMEMEntangledInterferenceLayer(complex_dim, num_complex_heads, dropout, noise_scale=sfin_noise_scale, use_entanglement=sfin_use_entanglement, use_rotary=sfin_use_rotary)
            for _ in range(num_complex_layers)])
        self.complex_norms_mid = nn.ModuleList([HAKMEMComplexLayerNorm(complex_dim, coupled=True) for _ in range(num_complex_layers)])
        self.complex_dropout_real = nn.Dropout(dropout)
        self.complex_dropout_imag = nn.Dropout(dropout)
        # Project processed complex tangent vectors back to a real-valued tangent space vector
        # Output dim of this projection will feed into the final projection to decoder memory
        intermediate_real_dim = hyperbolic_embedding_dim # Or another dimension? Let's map back to hyp_embed_dim for now
        self.complex_to_tangent = HAKMEMComplexToRealProjection(complex_dim, intermediate_real_dim, method=projection_method)

        # --- Final Projection & Decoding ---
        self.projection_to_decoder_memory = nn.Linear(intermediate_real_dim, decoder_memory_dim)
        nn.init.xavier_uniform_(self.projection_to_decoder_memory.weight)
        self.local_decoder = HAKMEMLocalDecoder(local_hidden_size, decoder_memory_dim, num_layers=4, num_heads=max(1, local_hidden_size//64), dropout=dropout, use_hierarchical_pred=use_hierarchical_decoder)

        # Residual controller for complex layers
        self.residual_controller = nn.Parameter(torch.zeros(num_complex_layers))

        logger.info(f"HyperHAKMEMBSFIN Initialized: HypDim={hyperbolic_embedding_dim}, Curve={curvature}, ComplexDim={complex_dim}, DecMem={decoder_memory_dim}")


    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = byte_seq.size(0)
        device = byte_seq.device

        # --- Patching and Encoding ---
        batch_patch_repr_list = []; num_patches_per_item = []; valid_batch_indices = []
        for i in range(batch_size):
            seq = byte_seq[i]
            patches = self.patcher.create_patches(seq)
            if patches:
                real_patch_repr_single = self.local_encoder(patches) # [1, num_p, local_hidden]
                if real_patch_repr_single.numel() > 0 and real_patch_repr_single.size(1) > 0:
                    batch_patch_repr_list.append(real_patch_repr_single.squeeze(0)) # Store as [num_p, local_hidden]
                    num_patches = real_patch_repr_single.size(1)
                    num_patches_per_item.append(num_patches)
                    valid_batch_indices.append(i)
                else: num_patches_per_item.append(0)
            else: num_patches_per_item.append(0)

        if not valid_batch_indices:
             target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
             return torch.zeros((batch_size, target_len, 256), device=device)

        # --- Padding and Stacking ---
        max_num_patches = max(num_patches_per_item) if num_patches_per_item else 0
        if max_num_patches == 0:
            target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
            return torch.zeros((batch_size, target_len, 256), device=device)

        padded_repr_list = []; memory_padding_mask_list = []
        for i, item_idx in enumerate(valid_batch_indices):
             repr_tensor = batch_patch_repr_list[i]
             num_patches = num_patches_per_item[item_idx]
             padding_size = max_num_patches - num_patches
             mask = torch.zeros(max_num_patches, dtype=torch.bool, device=device)
             if padding_size > 0:
                 mask[num_patches:] = True
                 padding = torch.zeros((padding_size, self.local_hidden_size), device=device)
                 padded_repr = torch.cat([repr_tensor, padding], dim=0)
             else: padded_repr = repr_tensor[:max_num_patches]
             padded_repr_list.append(padded_repr); memory_padding_mask_list.append(mask)

        real_patch_repr_batched = torch.stack(padded_repr_list, dim=0) # [B_valid, max_num_p, local_hidden]
        memory_padding_mask = torch.stack(memory_padding_mask_list, dim=0) # [B_valid, max_num_p]

        # --- Hyperbolic Processing ---
        # 1. Project to Euclidean space for hyperbolic embedding
        euclidean_for_hyper = self.projection_to_hyp_euclidean(real_patch_repr_batched) # [B_valid, max_num_p, hyp_embed_dim]

        # 2. Clip & Map to Hyperbolic Space (Poincaré Ball)
        clipped_euclidean = self.hyperbolic_utils.poincare_clip(euclidean_for_hyper, self.curvature, self.clipping_radius)
        hyperbolic_repr = self.hyperbolic_utils.exponential_map(clipped_euclidean, self.curvature) # [B_valid, max_num_p, hyp_embed_dim]

        # 3. Map back to Euclidean Tangent Space
        tangent_repr = self.hyperbolic_utils.logarithmic_map(hyperbolic_repr, self.curvature) # [B_valid, max_num_p, hyp_embed_dim]

        # --- Complex Processing in Tangent Space ---
        # 4. Project tangent vectors to complex domain
        complex_tangent_repr = self.tangent_to_complex(tangent_repr) # Tuple (real, imag) [B_valid, max_num_p, complex_dim]

        # 5. Apply Complex Positional Encoding
        complex_tangent_repr = self.complex_pos_encoding(complex_tangent_repr)

        # 6. Apply Complex Layer Stack
        real, imag = self.complex_norm_in(complex_tangent_repr)
        for i, layer in enumerate(self.complex_interference_layers):
            real_res, imag_res = real, imag
            normed_real, normed_imag = self.complex_norms_mid[i]((real, imag))
            # Pass padding mask to the layer (True=MASK)
            out_real, out_imag = layer((normed_real, normed_imag), attention_mask=memory_padding_mask)
            out_real = self.complex_dropout_real(out_real); out_imag = self.complex_dropout_imag(out_imag)
            residual_strength = torch.sigmoid(self.residual_controller[i])
            real = real_res + out_real * residual_strength
            imag = imag_res + out_imag * residual_strength
        processed_complex_tangent = (real, imag)

        # 7. Project processed complex tangent vectors back to real tangent space
        processed_tangent_repr = self.complex_to_tangent(processed_complex_tangent) # [B_valid, max_num_p, intermediate_real_dim]

        # --- Final Projection & Decoding ---
        # 8. Project to final decoder memory dimension
        decoder_memory = self.projection_to_decoder_memory(processed_tangent_repr) # [B_valid, max_num_p, decoder_memory_dim]

        # 9. Decode
        if target_byte_seq is None:
            # Return memory and mask for generation seeding or analysis
            # Need to reconstruct full batch if necessary outside the model
            logger.warning("target_byte_seq is None. Returning decoder memory, mask, and valid indices.")
            # Pad memory and mask to full batch size
            full_decoder_memory = torch.zeros((batch_size, max_num_patches, self.decoder_memory_dim), device=device)
            full_decoder_memory[valid_batch_indices] = decoder_memory
            full_memory_padding_mask = torch.ones((batch_size, max_num_patches), dtype=torch.bool, device=device)
            full_memory_padding_mask[valid_batch_indices] = memory_padding_mask
            # Returning logits shape filled with zeros to match expected output type
            return torch.zeros((batch_size, 0, 256), device=device) # Match expected output type for logits

        else:
            # Select target sequences for valid batch items
            valid_target_byte_seq = target_byte_seq[valid_batch_indices] # [B_valid, S_tgt]
            # Decode using the final memory and padding mask
            byte_logits_valid = self.local_decoder(
                tgt_byte_seq=valid_target_byte_seq,
                memory=decoder_memory, # Pass the correctly shaped memory
                memory_key_padding_mask=memory_padding_mask # Pass the corresponding mask
            ) # [B_valid, S_tgt, 256]

            # Reconstruct full batch output
            final_byte_logits = torch.zeros((batch_size, target_byte_seq.size(1), 256), device=device)
            # Use advanced indexing with tensor indices
            valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
            final_byte_logits[valid_indices_tensor] = byte_logits_valid.to(final_byte_logits.dtype)
            return final_byte_logits

    # --- Static method for Loss Computation (using HAKMEM version) ---
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.size()
        if seq_len == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True) # Handle empty sequence case
        logits_flat = logits.reshape(-1, vocab_size); targets_flat = targets.reshape(-1)
        if torch.any((targets_flat < 0) | (targets_flat >= vocab_size)):
            targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1) # Clamp instead of error
            logger.warning("Target indices clamped.")
        with torch.no_grad():
            smooth_val_on = 1.0 - smoothing; smooth_val_off = smoothing / max(1, vocab_size - 1)
            true_dist = torch.full_like(logits_flat, smooth_val_off)
            true_dist.scatter_(1, targets_flat.unsqueeze(1), smooth_val_on)
        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1)
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            loss = loss * (~mask_flat)
            num_active_elements = (~mask_flat).sum()
            mean_loss = loss.sum() / num_active_elements if num_active_elements > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)
        else: mean_loss = loss.mean()
        return mean_loss

    # --- Generation Method (using HAKMEM version) ---
    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None) -> torch.Tensor:
        self.eval(); device = seed_bytes.device; batch_size, seed_len = seed_bytes.size()
        generated_sequence = seed_bytes.clone()
        if sampling_config is None: sampling_config = SamplerConfig()
        sequence_memory = [{} for _ in range(batch_size)]; max_ngram_size = 5; repetition_penalty = 1.2
        base_temperature = temperature; min_temp, max_temp = max(0.1, base_temperature * 0.5), min(2.0, base_temperature * 1.5)

        for step in tqdm(range(max_length), desc="Generating", disable=batch_size > 1 or max_length < 10):
            current_context = generated_sequence
            logits_all = self(byte_seq=current_context, target_byte_seq=current_context)
            if logits_all is None or logits_all.numel() == 0 or logits_all.shape[1] == 0: logger.warning("Logits generation failed."); break
            next_byte_logits = logits_all[:, -1, :]
            next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

            for i in range(batch_size):
                current_logits = next_byte_logits[i]; current_seq_len = generated_sequence.size(1)
                if current_seq_len > 1: # Apply repetition penalty
                    for ngram_size in range(2, min(max_ngram_size + 1, current_seq_len)):
                        recent_ngram = tuple(generated_sequence[i, -(ngram_size):].cpu().tolist())
                        if recent_ngram in sequence_memory[i]: current_logits[sequence_memory[i][recent_ngram]] -= math.log(repetition_penalty)
                probs = F.softmax(current_logits, dim=-1); entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
                adaptive_temp = base_temperature * (0.8 if entropy < sampling_config.low_entropy_threshold else (1.1 if entropy > sampling_config.medium_entropy_threshold else 1.0))
                adaptive_temp = max(min_temp, min(adaptive_temp, max_temp))
                scaled_logits = current_logits / max(adaptive_temp, 1e-6); probs = F.softmax(scaled_logits, dim=-1)
                # Sampling logic (simplified - use HAKMEMBSFINModel's original detailed logic here)
                if temperature <= 0: next_byte_idx = torch.argmax(probs) # Greedy
                else: next_byte_idx = torch.multinomial(probs, num_samples=1).squeeze(-1) # Basic sampling

                next_byte_indices[i] = next_byte_idx
                # Update memory (simplified)
                if current_seq_len > 0:
                    for ngram_size in range(1, min(max_ngram_size, current_seq_len + 1)):
                         if ngram_size <= current_seq_len:
                              preceding_ngram = tuple(generated_sequence[i, -(ngram_size):].cpu().tolist())
                              sequence_memory[i][preceding_ngram] = next_byte_idx.item()
                if len(sequence_memory[i]) > 2000: # Limit memory
                    keys_to_remove = random.sample(list(sequence_memory[i].keys()), 500); [sequence_memory[i].pop(k, None) for k in keys_to_remove]

            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)
        return generated_sequence


# =====================================================================
# Main Execution Block (Example Usage)
# =====================================================================
if __name__ == "__main__":
    logger.info("Running HyperHAKMEMBSFIN Example")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Config ---
    config = {
        'local_hidden_size': 128, 'complex_dim': 256, 'num_complex_layers': 4,
        'num_complex_heads': 4, 'decoder_memory_dim': 384, 'dropout': 0.1,
        'context_window': 64, 'n_gram_sizes': [2, 3], 'n_gram_vocab_size': 5000,
        'sfin_noise_scale': 0.02, 'sfin_use_entanglement': True, 'sfin_use_rotary': True,
        'projection_method': 'hakmem_enhanced', 'use_hierarchical_decoder': False,
        'hyperbolic_embedding_dim': 192, 'curvature': 0.5, 'clipping_radius': 1.0,
    }

    # --- Model ---
    model = HyperHAKMEMBSFINModel(**config).to(device)
    logger.info(f"Model created successfully on {device}.")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {total_params:,}")

    # --- Optimizer ---
    optimizer = HAKMEMEnhancedSGD(model.parameters(), lr=1e-4, q_learning_config={'epsilon': 0.1})
    logger.info("Optimizer created successfully.")

    # --- Dummy Data ---
    batch_size = 4
    seq_len = config['context_window']
    dummy_input = torch.randint(0, 256, (batch_size, seq_len), device=device)
    # Target is the same as input for autoregressive training simulation
    dummy_target = dummy_input.clone()
    # Shift target for next-token prediction loss (typical setup)
    dummy_target_shifted = torch.cat([dummy_input[:, 1:], torch.randint(0, 256, (batch_size, 1), device=device)], dim=1)


    # --- Training Step Simulation ---
    logger.info("Simulating a training step...")
    model.train()
    optimizer.zero_grad()
    try:
        # Forward pass
        logits = model(byte_seq=dummy_input, target_byte_seq=dummy_target) # Pass target_byte_seq = input for training
        # Check output shape
        logger.info(f"Logits shape: {logits.shape}") # Expect [B, S, 256]

        # Compute loss
        # We need to compare logits[i] with target[i+1] or similar
        # Compute loss comparing logits with the shifted target
        loss = model.compute_loss(logits, dummy_target_shifted, smoothing=0.1)
        logger.info(f"Calculated Loss: {loss.item():.4f}")

        # Update optimizer's internal loss tracking for Q-learning
        if hasattr(optimizer, 'set_current_loss'):
             optimizer.set_current_loss(loss.item())

        # Backward pass (requires gradients)
        # Check if loss requires grad before calling backward
        if loss.requires_grad:
             # loss.backward() # This would compute gradients
             # logger.info("Backward pass simulated.")
              pass # Skip actual backward for this example
        else:
             logger.warning("Loss does not require grad. Skipping backward.")


        # Optimizer step (requires gradients to have been computed)
        # optimizer.step() # This would update weights
        # logger.info("Optimizer step simulated.")

    except Exception as e:
        logger.error(f"Error during training step simulation: {e}", exc_info=True)

    # --- Generation Simulation ---
    logger.info("\nSimulating generation...")
    model.eval()
    seed_text = "Hyperbolic"
    tokenizer = ByteTokenizer()
    seed_bytes_list = tokenizer.encode(seed_text)
    seed_tensor = torch.tensor(seed_bytes_list, dtype=torch.long).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
             generated_tensor = model.generate(seed_tensor, max_length=30, temperature=0.7)
             generated_list = generated_tensor[0].cpu().tolist()
             generated_text = tokenizer.decode(generated_list)
             logger.info(f"Seed: '{seed_text}'")
             logger.info(f"Generated (untrained): '{generated_text}'")
    except Exception as e:
        logger.error(f"Error during generation simulation: {e}", exc_info=True)

    logger.info("HyperHAKMEMBSFIN example finished.")

"""
Explanation of Changes:

HyperbolicUtils: Added a class to encapsulate Poincaré clipping, exponential map (exp_o^c), and logarithmic map (log_o^c) based on the PDF, along with helper functions for Möbius addition and distance (though not strictly needed for the chosen pipeline). Curvature c is handled.

HyperHAKMEMBSFINModel:

New Parameters: Added hyperbolic_embedding_dim, curvature, clipping_radius.

Pipeline: Implemented the modified pipeline described above:

projection_to_hyp_euclidean: Maps local encoder output to the dimension intended for hyperbolic space.

Clipping + Exp Map: Maps into the Poincaré ball using HyperbolicUtils.

Log Map: Maps back to the tangent space.

tangent_to_complex: Projects the tangent space vectors into the complex domain (real/imag parts) for the complex stack. Re-introduced from HAKMEM version.

Complex Stack: Applies HAKMEM Positional Encoding, Norms, and Interference Layers to these complex tangent representations.

complex_to_tangent: Projects the processed complex vectors back to a real vector in the tangent space. Re-introduced from HAKMEM version.

projection_to_decoder_memory: Final projection from the processed tangent space vector to the dimension expected by the decoder.

Component Usage: Uses the HAKMEM versions of Encoder, Decoder, Attention, Positional Encoding, Complex Ops, LayerNorm, Interference Layers.

Forward Pass: The forward method explicitly shows the flow through the hyperbolic mapping (Exp/Log maps) and the subsequent processing in the tangent space by the complex stack. Mask handling (memory_padding_mask) is passed correctly to relevant layers (interference layers for self-attention on patches, decoder for cross-attention on memory).

Optimizer: HAKMEMEnhancedSGD is used directly, as it optimizes the Euclidean parameters of the network layers.

Generation/Loss: Uses the generate and compute_loss methods defined within the HAKMEMBSFINModel class (which were included in the hakmem_improvements.py draft).

Example Usage: Updated main block to instantiate HyperHAKMEMBSFINModel and demonstrate a basic forward pass and generation.

This integrated model attempts to leverage the representational benefits of hyperbolic geometry for the sequence of patches while retaining the computational enhancements and complex processing stack inspired by HAKMEM, operating that stack on the tangent space representation derived from the hyperbolic embeddings.
"""