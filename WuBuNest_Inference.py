import torch
import os
import argparse
import logging
from typing import List, Optional, Dict, Tuple, Union, Iterable, Any
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import numpy as np
import random
import time
import contextlib
from datetime import datetime, timedelta
import gc
import socket
import platform
from torch import amp # Use torch.amp
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
import inspect
import string
import hashlib
import functools # Added for worker_init_fn fix


# Set up logging
logger = logging.getLogger("WuBuNestInference")
# Basic config will be overridden if run via Trainer, but useful for standalone script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)

# Constants
EPS = 1e-7 # Small epsilon for numerical stability

# =====================================================================
# Hyperbolic Geometry Utilities (Self-Contained)
# =====================================================================
class HyperbolicUtils:
    """Utility functions for Poincare ball model of hyperbolic geometry."""
    @staticmethod
    def poincare_clip(x: torch.Tensor, c: float, radius: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """Clips points to stay strictly inside the Poincare ball boundary."""
        if c <= 0: return x # Not hyperbolic if curvature is non-positive
        sqrt_c = math.sqrt(max(c, eps)) # Ensure c is positive for sqrt
        max_norm = (radius / sqrt_c) * (1.0 - eps) # Max Euclidean norm allowed

        # Use float32 for norm calculation for stability, then cast back if needed
        original_dtype = x.dtype
        x_norm_sq = torch.sum(x.float().pow(2), dim=-1, keepdim=True)
        norm = torch.sqrt(torch.clamp(x_norm_sq, min=0) + eps)

        cond = norm > max_norm
        # Ensure scale_factor is calculated and applied using the same dtype as x
        scale_factor = torch.where(cond, max_norm / (norm + eps), torch.ones_like(norm)).to(original_dtype)
        clipped_x = x * scale_factor

        # Final sanity check
        if not torch.isfinite(clipped_x).all():
             logger.warning("NaN/Inf detected *after* poincare_clip. Replacing.")
             clipped_x = torch.nan_to_num(clipped_x, nan=0.0) # Replace with 0
        return clipped_x

    @staticmethod
    def exponential_map(v: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
        """Maps a tangent vector v at the origin to the Poincare ball (exp_0^c(v))."""
        if c <= 0: return v # No mapping needed for Euclidean space
        original_dtype = v.dtype
        # Compute norm in float32 for stability
        v_norm_sq = torch.sum(v.float().pow(2), dim=-1, keepdim=True)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=0) + eps)
        sqrt_c = math.sqrt(max(c, eps))

        tanh_term = torch.tanh(sqrt_c * v_norm).to(original_dtype) # Calculate tanh in float32, cast back
        # Ensure lambda calculation uses consistent dtype
        lambda_v = torch.where(v_norm > eps, tanh_term / (sqrt_c * v_norm + eps).to(original_dtype), torch.ones_like(v_norm).to(original_dtype))

        mapped_v = lambda_v * v
        # Clip result using original curvature and return clipped value
        return HyperbolicUtils.poincare_clip(mapped_v, c)

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
        """Maps a point y in the Poincare ball back to the tangent space at the origin (log_0^c(y))."""
        if c <= 0: return y # No mapping needed for Euclidean space
        original_dtype = y.dtype
        # Clip input first to ensure it's inside the ball
        y_clipped = HyperbolicUtils.poincare_clip(y, c)

        # Compute norm in float32 for stability
        y_norm_sq = torch.sum(y_clipped.float().pow(2), dim=-1, keepdim=True)
        y_norm = torch.sqrt(torch.clamp(y_norm_sq, min=0) + eps)
        sqrt_c = math.sqrt(max(c, eps))

        # Clamp input to atanh carefully
        arctanh_input = torch.clamp(sqrt_c * y_norm, min=-1.0 + eps, max=1.0 - eps)
        atanh_term = torch.atanh(arctanh_input).to(original_dtype) # Calculate atanh in float32, cast back

        # Ensure lambda calculation uses consistent dtype
        lambda_y = torch.where(y_norm > eps, atanh_term / (sqrt_c * y_norm + eps).to(original_dtype), torch.ones_like(y_norm).to(original_dtype))

        mapped_y = lambda_y * y_clipped
        if not torch.isfinite(mapped_y).all():
             logger.warning("NaN/Inf detected *in* logarithmic_map output. Replacing.")
             mapped_y = torch.nan_to_num(mapped_y, nan=0.0) # Replace with 0
        return mapped_y

    @staticmethod
    def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
        """Computes the hyperbolic distance between points x and y in the Poincare ball."""
        if c <= 0:
            # Using Euclidean distance for c<=0
            return torch.norm(x - y, dim=-1)

        sqrt_c = math.sqrt(max(c, eps))
        radius_clip = 0.999 # Use a slightly tighter radius for distance calculation stability
        # Clip points before calculation
        x_clipped = HyperbolicUtils.poincare_clip(x, c, radius=radius_clip, eps=eps)
        y_clipped = HyperbolicUtils.poincare_clip(y, c, radius=radius_clip, eps=eps)

        original_dtype = x.dtype
        # Calculate intermediate terms in float32 for stability
        x_norm_sq = torch.sum(x_clipped.float().pow(2), dim=-1)
        y_norm_sq = torch.sum(y_clipped.float().pow(2), dim=-1)
        diff_norm_sq = torch.sum((x_clipped.float() - y_clipped.float()).pow(2), dim=-1)

        denom_x = torch.clamp(1.0 - c * x_norm_sq, min=eps)
        denom_y = torch.clamp(1.0 - c * y_norm_sq, min=eps)

        arcosh_arg = 1.0 + 2.0 * c * diff_norm_sq / (denom_x * denom_y + eps)
        # Ensure arcosh argument is >= 1.0
        arcosh_arg_clamped = torch.clamp(arcosh_arg, min=1.0 + eps)

        # Calculate acosh in float32 and cast back
        distance = (1.0 / sqrt_c) * torch.acosh(arcosh_arg_clamped)
        distance = distance.to(original_dtype)

        if not torch.isfinite(distance).all():
            logger.warning(f"NaN/Inf detected in poincare_distance. Replacing with 100.")
            # Replace non-finite distances with a large but finite number
            distance = torch.nan_to_num(distance, nan=100.0, posinf=100.0, neginf=-100.0)
        return distance

# =====================================================================
# Data Structures and Configuration Classes (from Trainer)
# =====================================================================
@dataclass
class SamplerConfig:
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# =====================================================================
# HAKMEM-Inspired Entropy Calculation Helper (from Trainer)
# =====================================================================
class HAKMEMEntropyHelper:
    """Calculates Shannon entropy for byte sequences with caching."""
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_cache = {}
        self.max_cache_size = max_cache_size

    def _clean_cache(self):
        """Removes older entries if cache exceeds max size."""
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5)
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache:
                    del self.entropy_cache[k]

    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...], List[int], bytes, torch.Tensor]) -> float:
        """Computes entropy, using cache if possible."""
        cache_key = None
        byte_list = []
        if isinstance(byte_window, tuple):
            cache_key = byte_window
            byte_list = list(byte_window)
        elif isinstance(byte_window, list):
            cache_key = tuple(byte_window)
            byte_list = byte_window
        elif isinstance(byte_window, bytes):
            cache_key = byte_window
            byte_list = list(byte_window)
        elif isinstance(byte_window, np.ndarray):
            byte_list = byte_window.tolist()
            cache_key = tuple(byte_list)
        elif isinstance(byte_window, torch.Tensor):
            byte_list = byte_window.cpu().byte().tolist()
            cache_key = tuple(byte_list)
        else:
            logger.warning(f"Unsupported type for entropy calculation: {type(byte_window)}")
            return 0.0

        if cache_key is not None and cache_key in self.entropy_cache:
            return self.entropy_cache[cache_key]
        if not byte_list:
            return 0.0
        try:
            byte_counts = np.bincount(np.array(byte_list, dtype=np.uint8), minlength=256)
            total_bytes = byte_counts.sum()
            if total_bytes == 0:
                return 0.0
            probs = byte_counts[byte_counts > 0] / total_bytes
            entropy = float(-np.sum(probs * np.log2(probs + EPS)))
            result = max(0.0, entropy)
            if cache_key is not None:
                self.entropy_cache[cache_key] = result
                self._clean_cache()
            return result
        except Exception as e:
            logger.warning(f"Error during entropy calculation: {e}")
            return 0.0

# =====================================================================
# HAKMEM Babylon Index (Word/Punctuation Based) (from Trainer)
# =====================================================================
class HAKMEMBabylonIndex:
    """Splits byte sequences into 'word' and 'delimiter' patches based on text decoding."""
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_helper = HAKMEMEntropyHelper(max_cache_size)
        self.whitespace_chars = set(string.whitespace)
        self.punctuation_chars = set(string.punctuation)
        logger.info("HAKMEMBabylonIndex initialized (Word/Punctuation Patching with Entropy).")

    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Creates patches from a byte tensor, attempting UTF-8 decoding."""
        if byte_seq_tensor.numel() == 0: return []
        if byte_seq_tensor.dim() != 1: byte_seq_tensor = byte_seq_tensor.flatten()
        if byte_seq_tensor.numel() == 0: return []
        device = byte_seq_tensor.device
        try:
            text = byte_seq_tensor.cpu().numpy().tobytes().decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error decoding byte tensor (len {byte_seq_tensor.numel()}): {e}. Returning no patches.")
            return []

        patches_with_entropy = []
        current_patch_start = 0
        in_word = False
        for i, char in enumerate(text):
            is_delimiter = char in self.whitespace_chars or char in self.punctuation_chars
            if is_delimiter:
                if in_word:
                    word_str = text[current_patch_start:i]
                    try:
                        word_bytes = torch.tensor(list(word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                        if word_bytes.numel() > 0:
                            entropy = self.entropy_helper.compute_entropy(word_bytes)
                            patches_with_entropy.append((word_bytes, entropy))
                    except Exception as enc_e:
                        logger.warning(f"Error encoding word patch '{word_str[:20]}...': {enc_e}")
                    in_word = False
                try:
                    delim_bytes = torch.tensor(list(char.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                    if delim_bytes.numel() > 0:
                        entropy = self.entropy_helper.compute_entropy(delim_bytes)
                        patches_with_entropy.append((delim_bytes, entropy))
                except Exception as enc_e:
                    logger.warning(f"Error encoding delimiter patch '{char}': {enc_e}")
                current_patch_start = i + 1
            else:
                if not in_word:
                    in_word = True
                    current_patch_start = i
        if in_word and current_patch_start < len(text):
            trailing_word_str = text[current_patch_start:]
            try:
                trailing_word_bytes = torch.tensor(list(trailing_word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                if trailing_word_bytes.numel() > 0:
                    entropy = self.entropy_helper.compute_entropy(trailing_word_bytes)
                    patches_with_entropy.append((trailing_word_bytes, entropy))
            except Exception as enc_e:
                logger.warning(f"Error encoding trailing word patch '{trailing_word_str[:20]}...': {enc_e}")
        patches_with_entropy = [(p, e) for p, e in patches_with_entropy if p.numel() > 0]
        return patches_with_entropy

    @torch.no_grad()
    def reset_context(self):
        self.entropy_helper.entropy_cache = {}

# =====================================================================
# ByteTokenizer (Self-Contained)
# =====================================================================
class ByteTokenizer:
    """Simple stateless tokenizer for converting between text and utf-8 byte sequences."""
    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of UTF-8 byte values."""
        return list(text.encode('utf-8', errors='replace'))

    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        """Decodes a sequence of bytes (int or tensor) back to text."""
        valid_bytes = []
        for b in byte_sequence:
            try:
                val = b.item() if isinstance(b, torch.Tensor) else int(b)
            except Exception:
                continue
            if 0 <= val <= 255:
                valid_bytes.append(val)
        return bytes(valid_bytes).decode('utf-8', errors='replace')

# =====================================================================
# HAKMEM-Enhanced Cross Attention Block (Self-Contained)
# =====================================================================
class HAKMEMCrossAttentionBlock(nn.Module):
    """A standard cross-attention block with LayerNorm and optional Flash Attention."""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size <= 0: raise ValueError("hidden_size must be positive")
        if num_heads <= 0 :
            num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads:
            num_heads = 1
            logger.warning(f"Using 1 head for CrossAttention size {hidden_size} (no divisors found).")
        elif hidden_size % num_heads != 0:
            num_heads = valid_heads[0]
            logger.warning(f"Adjusted CrossAttention heads: {original_num_heads} -> {num_heads} for hidden_size {hidden_size}.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(max(1, self.head_dim))

        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_queries, _ = queries.size()
        _, seq_len_kv, kv_hidden_size = keys_values.size()

        if seq_len_kv == 0:
            return torch.zeros_like(queries)
        if kv_hidden_size != self.hidden_size:
            raise ValueError(f"Key/Value hidden size mismatch: K/V has {kv_hidden_size}, Q expects {self.hidden_size}")

        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)
        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask_sdpa = None
        if attention_mask is not None:
            mask_dtype = torch.bool
            if attention_mask.dtype != torch.bool:
                 attention_mask = attention_mask > 0
            if attention_mask.dim() == 2:
                attn_mask_sdpa = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=mask_dtype)
            elif attention_mask.dim() == 3:
                attn_mask_sdpa = attention_mask.unsqueeze(1).to(dtype=mask_dtype)
            elif attention_mask.dim() == 4:
                attn_mask_sdpa = attention_mask.to(dtype=mask_dtype)
            else:
                logger.warning(f"Ignoring unsupported attention mask shape {attention_mask.shape}. Expected 2D, 3D, or 4D.")

            if attn_mask_sdpa is not None:
                attn_mask_sdpa = attn_mask_sdpa.to(device=queries.device)
                expected_shape = (batch_size, self.num_heads, num_queries, seq_len_kv)
                try:
                    torch.broadcast_shapes(attn_mask_sdpa.shape, expected_shape)
                except RuntimeError:
                    logger.warning(f"Attention mask shape {attn_mask_sdpa.shape} not broadcastable to target shape {expected_shape}. Ignoring mask.")
                    attn_mask_sdpa = None

        use_flash = hasattr(F, 'scaled_dot_product_attention')
        output = None
        if use_flash:
            try:
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_sdpa, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
                if not torch.isfinite(output).all():
                    raise ValueError("Flash Attention produced NaN/Inf values.")
            except Exception as e:
                logger.warning(f"Flash Attention failed: {e}. Falling back to standard attention.", exc_info=False)
                use_flash = False
                output = None

        if output is None:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = torch.clamp(scores, min=-30.0, max=30.0)
            if attn_mask_sdpa is not None:
                scores = scores.masked_fill(attn_mask_sdpa, float('-inf'))
            attn_probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn_probs = torch.nan_to_num(attn_probs)
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in CrossAttention final output. Replacing with zeros.")
            output = torch.nan_to_num(output, nan=0.0)
        return output

# =====================================================================
# HAKMEM-Enhanced Local Encoder (Self-Contained)
# =====================================================================
class HAKMEMLocalEncoder(nn.Module):
    """Encodes individual patches using byte embeddings, optional N-grams, and a Transformer."""
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1, n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Local Encoder hidden_size must be positive.")
        self.hidden_size = hidden_size

        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))

        self.n_gram_sizes = sorted(list(set(s for s in n_gram_sizes if isinstance(s, int) and s > 0)))
        self.n_gram_vocab_size = n_gram_vocab_size
        self.n_gram_embeddings = None
        self.hash_multipliers = {}
        if self.n_gram_sizes:
            if n_gram_vocab_size <= 0:
                logger.warning("Disabling N-gram features: n_gram_vocab_size <= 0.")
                self.n_gram_sizes = []
            else:
                self.n_gram_embeddings = nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes})
                for emb in self.n_gram_embeddings.values():
                    nn.init.normal_(emb.weight, std=0.02)
                self.hash_multipliers = {n: torch.tensor([self._get_prime(n * 10 + i + 1) for i in range(n)], dtype=torch.long) for n in self.n_gram_sizes}
                logger.info(f"HAKMEMLocalEncoder N-grams enabled: Sizes={self.n_gram_sizes}, Vocab={self.n_gram_vocab_size}")
        else:
            logger.info("HAKMEMLocalEncoder: N-gram features disabled.")

        if num_heads <= 0:
            num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads:
            num_heads = 1
        elif hidden_size % num_heads != 0:
            num_heads = valid_heads[0]
        if num_heads != original_num_heads:
            logger.warning(f"Local Encoder Transformer adjusted heads: {original_num_heads} -> {num_heads} for hidden_size {hidden_size}.")

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.patch_pooling_attention = HAKMEMCrossAttentionBlock(hidden_size, num_heads, dropout)
        self.patch_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)

        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _get_prime(self, n):
        def is_prime(num):
            if num < 2: return False
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0: return False
            return True
        num = n
        # *** Syntax Fix Start ***
        while True:
            if is_prime(num): return num
            num += 1
        # *** Syntax Fix End ***

    def _get_n_gram_hashes(self, patch_byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        patch_len = patch_byte_sequence.size(0)
        device = patch_byte_sequence.device
        if patch_len < n: return torch.empty(0, dtype=torch.long, device=device)
        windows = patch_byte_sequence.long().unsqueeze(0).unfold(dimension=1, size=n, step=1)
        multipliers = self.hash_multipliers.get(n)
        if multipliers is None:
            logger.warning(f"Hash multipliers not found for n={n}. Using defaults.")
            multipliers = torch.tensor([31**i for i in range(n)], device=device, dtype=torch.long)
        else:
            multipliers = multipliers.to(device=device, dtype=torch.long)
        multipliers = multipliers.view(1, 1, n)
        hashes = (windows * multipliers).sum(dim=-1)
        return (hashes % self.n_gram_vocab_size).squeeze(0)

    def forward(self, patches_with_entropy: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        if not patches_with_entropy:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)
        device = patches_with_entropy[0][0].device
        model_dtype = next(self.parameters()).dtype
        patch_representations = []
        for patch_bytes, patch_entropy in patches_with_entropy:
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue
            patch_bytes_long = patch_bytes.long()
            x = self.byte_embeddings(patch_bytes_long).to(model_dtype)
            x = x.unsqueeze(0)
            if self.n_gram_embeddings and self.n_gram_sizes:
                n_gram_features = torch.zeros_like(x)
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_long, n)
                        if n_gram_hashes.numel() > 0:
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes).to(model_dtype)
                            ngram_embeds = ngram_embeds.unsqueeze(0)
                            num_windows = ngram_embeds.size(1)
                            indices = torch.arange(n - 1, n - 1 + num_windows, device=device, dtype=torch.long)
                            valid_mask = indices < patch_len
                            valid_indices = indices[valid_mask]
                            valid_embeds = ngram_embeds[:, valid_mask, :]
                            if valid_indices.numel() > 0:
                                index_reshaped = valid_indices.view(1, -1, 1)
                                index_expanded = index_reshaped.expand(1, valid_indices.size(0), self.hidden_size)
                                n_gram_features.scatter_add_(1, index_expanded, valid_embeds)
                x = x + n_gram_features
            if not torch.isfinite(x).all():
                logger.warning(f"NaN/Inf in LocalEncoder input pre-Transformer (PatchLen={patch_len}). Replacing.")
                x = torch.nan_to_num(x, nan=0.0)
            x = self.dropout(x)
            processed_bytes = self.transformer(x)
            if not torch.isfinite(processed_bytes).all():
                logger.warning(f"NaN/Inf in LocalEncoder output post-Transformer (PatchLen={patch_len}). Replacing.")
                processed_bytes = torch.nan_to_num(processed_bytes, nan=0.0)
            batch_query = self.patch_query.expand(1, -1, -1).to(dtype=model_dtype)
            patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes)
            if not torch.isfinite(patch_repr).all():
                logger.warning(f"NaN/Inf in LocalEncoder output post-pooling (PatchLen={patch_len}). Replacing.")
                patch_repr = torch.nan_to_num(patch_repr, nan=0.0)
            patch_representations.append(patch_repr.squeeze(1))
        if not patch_representations:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)
        patches_combined = torch.cat(patch_representations, dim=0)
        patches_combined = patches_combined.unsqueeze(0)
        normed_output = self.norm(patches_combined)
        if not torch.isfinite(normed_output).all():
            logger.warning("NaN/Inf in LocalEncoder final output. Replacing.")
            normed_output = torch.nan_to_num(normed_output, nan=0.0)
        return normed_output

# =====================================================================
# HAKMEM-Enhanced Local Decoder (Self-Contained)
# =====================================================================
class HAKMEMLocalDecoder(nn.Module):
    """Decodes byte sequences using embeddings, positional encodings, and cross-attention to memory."""
    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, use_hierarchical_pred: bool = True, max_decode_len: int = 2048):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Decoder hidden size must be positive.")
        if global_hidden_size <= 0: raise ValueError("Decoder global_hidden_size must be positive.")
        self.hidden_size = hidden_size
        self.use_hierarchical = use_hierarchical_pred
        self.max_decode_len = max_decode_len
        self.vocab_size = 256

        if num_heads <= 0:
            num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads:
            num_heads = 1
        elif hidden_size % num_heads != 0:
            num_heads = valid_heads[0]
        if num_heads != original_num_heads:
             logger.warning(f"HAKMEMLocalDecoder adjusted heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        self.byte_embeddings = nn.Embedding(self.vocab_size, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))
        self.positional_encoding = nn.Embedding(max_decode_len, hidden_size)
        nn.init.normal_(self.positional_encoding.weight, std=0.02)

        self.memory_projection = nn.Sequential(nn.Linear(global_hidden_size, hidden_size * 2, bias=True), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size, bias=True), nn.LayerNorm(hidden_size, eps=1e-6))
        for layer in self.memory_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.zeros_(layer.bias)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=self.decoder_norm)

        if self.use_hierarchical:
            self.byte_class_pred = nn.Linear(hidden_size, 16)
            self.byte_specific_pred = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(16)])
            nn.init.normal_(self.byte_class_pred.weight, std=0.02)
            if self.byte_class_pred.bias is not None:
                nn.init.zeros_(self.byte_class_pred.bias)
            for layer in self.byte_specific_pred:
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(16))
            for layer in self.byte_specific_pred:
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            logger.info("HAKMEMLocalDecoder using Hierarchical Prediction Head.")
        else:
            self.byte_pred = nn.Linear(hidden_size, self.vocab_size)
            nn.init.normal_(self.byte_pred.weight, std=0.02)
            if self.byte_pred.bias is not None:
                nn.init.zeros_(self.byte_pred.bias)
            logger.info("HAKMEMLocalDecoder using Flat Prediction Head.")

        self.dropout_embed = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
         mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
         return mask

    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, tgt_len = tgt_byte_seq.size()
        device = tgt_byte_seq.device
        mem_batch_size, mem_len, mem_dim_in = memory.size()
        model_dtype = next(self.parameters()).dtype

        if tgt_len == 0:
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)

        projected_memory: torch.Tensor
        if mem_len == 0:
            logger.debug("HAKMEMLocalDecoder received empty memory.")
            projected_memory = torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=model_dtype)
            memory_key_padding_mask = None
        else:
            if not torch.isfinite(memory).all():
                logger.warning("NaN/Inf detected in memory input to LocalDecoder. Replacing.")
                memory = torch.nan_to_num(memory, nan=0.0)
            projected_memory = self.memory_projection(memory.to(model_dtype))
            if not torch.isfinite(projected_memory).all():
                 logger.warning("NaN/Inf detected after memory projection in LocalDecoder. Replacing.")
                 projected_memory = torch.nan_to_num(projected_memory, nan=0.0)

        tgt_embed = self.byte_embeddings(tgt_byte_seq.long()).to(model_dtype)
        positions = torch.arange(0, tgt_len, device=device).unsqueeze(0)
        positions = torch.clamp(positions, max=self.positional_encoding.num_embeddings - 1)
        pos_embed = self.positional_encoding(positions).to(model_dtype)
        tgt_prepared = self.dropout_embed(tgt_embed + pos_embed)
        if not torch.isfinite(tgt_prepared).all():
            logger.warning("NaN/Inf detected in prepared target sequence input to Decoder Transformer. Replacing.")
            tgt_prepared = torch.nan_to_num(tgt_prepared, nan=0.0)

        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device=device, dtype=torch.bool)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(device=device, dtype=torch.bool)
            if memory_key_padding_mask.shape != (batch_size, mem_len):
                 logger.warning(f"memory_key_padding_mask shape mismatch ({memory_key_padding_mask.shape}) with memory ({batch_size, mem_len}). Ignoring mask.")
                 memory_key_padding_mask = None

        output = self.transformer(tgt=tgt_prepared, memory=projected_memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=memory_key_padding_mask)
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in output of Decoder Transformer. Replacing.")
            output = torch.nan_to_num(output, nan=0.0)

        byte_logits: torch.Tensor
        if self.use_hierarchical:
            byte_class_logits = self.byte_class_pred(output)
            if not torch.isfinite(byte_class_logits).all():
                logger.warning("NaN/Inf in hierarchical class logits. Replacing.")
                byte_class_logits = torch.nan_to_num(byte_class_logits, nan=0.0)
            log_class_probs = F.log_softmax(byte_class_logits, dim=-1)
            log_specific_probs_list = []
            for i in range(16):
                specific_logits = self.byte_specific_pred[i](output)
                if not torch.isfinite(specific_logits).all():
                     logger.warning(f"NaN/Inf in hierarchical specific logits head {i}. Replacing.")
                     specific_logits = torch.nan_to_num(specific_logits, nan=0.0)
                log_specific_probs_list.append(F.log_softmax(specific_logits, dim=-1))
            log_specific_probs_stacked = torch.stack(log_specific_probs_list, dim=2)
            combined_log_probs = log_class_probs.unsqueeze(-1) + log_specific_probs_stacked
            byte_logits = combined_log_probs.view(batch_size, tgt_len, self.vocab_size)
        else:
            byte_logits = self.byte_pred(output)

        byte_logits = byte_logits.float()
        if not torch.isfinite(byte_logits).all():
            logger.warning("NaN/Inf detected in final decoder logits. Replacing with zeros.")
            byte_logits = torch.nan_to_num(byte_logits, nan=0.0, posinf=0.0, neginf=0.0)
        return byte_logits

# =====================================================================
# WuBu Nesting Components (Self-Contained Geometry & Design Doc)
# =====================================================================
def check_quat_dim(dim: int, layer_name: str = "Layer"):
    if dim % 4 != 0:
        raise ValueError(f"{layer_name} dimension must be divisible by 4 for quaternion operations, but got {dim}")
def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1_shape = list(q1.shape)
    q2_shape = list(q2.shape)
    while len(q1_shape) < len(q2_shape): q1_shape.insert(0, 1)
    while len(q2_shape) < len(q1_shape): q2_shape.insert(0, 1)
    q1 = q1.view(q1_shape)
    q2 = q2.view(q2_shape)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)
def quat_rotate_via_pvq(v: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if v.shape[-1] != 4 or p.shape[-1] != 4 or q.shape[-1] != 4:
        raise ValueError(f"Inputs must be 4D for quat_rotate_via_pvq, shapes: v={v.shape}, p={p.shape}, q={q.shape}")
    p = p.expand_as(v)
    q = q.expand_as(v)
    pv = hamilton_product(p, v)
    pvq = hamilton_product(pv, q)
    return pvq
class QuaternionLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        check_quat_dim(in_features, "QuaternionLinear Input")
        check_quat_dim(out_features, "QuaternionLinear Output")
        self.in_features_quat = in_features // 4
        self.out_features_quat = out_features // 4
        self.r_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.i_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.j_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.k_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features_quat)
        gain = 1.0
        scale = gain * stdv
        nn.init.uniform_(self.r_weight, -scale, scale)
        nn.init.uniform_(self.i_weight, -scale, scale)
        nn.init.uniform_(self.j_weight, -scale, scale)
        nn.init.uniform_(self.k_weight, -scale, scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features_quat * 4:
            raise ValueError(f"Input feature dim {x.shape[-1]} != expected {self.in_features_quat * 4}")
        batch_dims = x.shape[:-1]
        x_reshaped = x.view(*batch_dims, self.in_features_quat, 4)
        r_x, i_x, j_x, k_x = x_reshaped[..., 0], x_reshaped[..., 1], x_reshaped[..., 2], x_reshaped[..., 3]
        out_r = F.linear(r_x, self.r_weight) - F.linear(i_x, self.i_weight) - F.linear(j_x, self.j_weight) - F.linear(k_x, self.k_weight)
        out_i = F.linear(r_x, self.i_weight) + F.linear(i_x, self.r_weight) + F.linear(j_x, self.k_weight) - F.linear(k_x, self.j_weight)
        out_j = F.linear(r_x, self.j_weight) - F.linear(i_x, self.k_weight) + F.linear(j_x, self.r_weight) + F.linear(k_x, self.i_weight)
        out_k = F.linear(r_x, self.k_weight) + F.linear(i_x, self.j_weight) - F.linear(j_x, self.i_weight) + F.linear(k_x, self.r_weight)
        output = torch.stack([out_r, out_i, out_j, out_k], dim=-1)
        output = output.view(*batch_dims, self.out_features_quat * 4)
        if self.bias is not None:
            output = output + self.bias
        return output
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, QuaternionLinear):
        m.reset_parameters()
def get_constrained_params(param: torch.Tensor, min_val: float = EPS) -> torch.Tensor:
    return F.softplus(param) + min_val
class TangentFlow(nn.Module):
    def __init__(self, dim: int, flow_type: str = 'mlp', hidden_dim_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.flow_type = flow_type
        if flow_type == 'linear':
            self.flow_map = nn.Linear(dim, dim)
        elif flow_type == 'mlp':
            hidden_dim = max(16, int(dim * hidden_dim_ratio))
            self.flow_map = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim))
        elif flow_type == 'none':
            self.flow_map = nn.Identity()
        else:
            raise ValueError(f"Unsupported tangent_flow_type: {flow_type}")
        self.flow_map.apply(init_weights)
    def forward(self, v_tangent: torch.Tensor) -> torch.Tensor:
        if self.flow_type == 'none':
            return torch.zeros_like(v_tangent)
        flow_displacement = self.flow_map(v_tangent)
        if not torch.isfinite(flow_displacement).all():
            logger.warning("NaN/Inf detected in TangentFlow output. Replacing with zeros.")
            flow_displacement = torch.nan_to_num(flow_displacement, nan=0.0)
        return flow_displacement
class BoundaryManifold(nn.Module):
    def __init__(self, level_idx: int, num_points: int, point_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim
        if num_points > 0 and point_dim > 0:
             tangent_points = torch.randn(num_points, point_dim) * init_scale
             self.tangent_points = nn.Parameter(tangent_points)
        else:
             logger.warning(f"BoundaryManifold L{level_idx} initialized with zero points/dim ({num_points}/{point_dim}). No parameters created.")
             self.register_parameter('tangent_points', None)
    def get_tangent_vectors_at_origin(self) -> Optional[torch.Tensor]:
        if self.tangent_points is None:
            return None
        if not torch.isfinite(self.tangent_points).all():
            logger.warning(f"NaN/Inf detected in BoundaryManifold tangent_points (Level {self.level_idx}). Re-initializing gently.")
            init_scale = 0.01
            self.tangent_points.data.normal_(0, init_scale)
        return self.tangent_points
class TangentSpaceRotation(nn.Module):
    def __init__(self, dim: int, rotation_type: str = 'so_n'):
        super().__init__()
        self.dim = dim
        self.rotation_type = rotation_type
        if rotation_type == 'so_n':
            self.skew_symmetric_params = nn.Parameter(torch.randn(dim, dim) * 0.01)
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using SO(n) via matrix exponential.")
        elif rotation_type == 'quat':
            if dim != 4:
                raise ValueError("Quaternion rotation requires dim=4.")
            init_p = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            init_q = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            self.quat_params_p = nn.Parameter(init_p)
            self.quat_params_q = nn.Parameter(init_q)
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using SO(4) via Quaternions (p*v*q).")
        elif rotation_type == 'identity':
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using Identity (no rotation).")
        else:
            raise ValueError(f"Unsupported rotation type: {rotation_type}")
    def _get_rotation_operator(self, device: torch.device):
        if self.rotation_type == 'so_n':
            params = self.skew_symmetric_params.to(device)
            skew_matrix = params - params.T
            R = torch.matrix_exp(skew_matrix)
            return R
        elif self.rotation_type == 'quat':
            p = self.quat_params_p.to(device)
            q = self.quat_params_q.to(device)
            p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            unit_p = p / p_norm
            q_norm = torch.norm(q, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            unit_q = q / q_norm
            if not torch.isfinite(unit_p).all() or not torch.isfinite(unit_q).all():
                logger.warning("NaN/Inf detected during quaternion normalization. Resetting p/q.")
                self.quat_params_p.data.normal_(0, 0.01).add_(torch.tensor([1.,0,0,0]))
                self.quat_params_q.data.normal_(0, 0.01).add_(torch.tensor([1.,0,0,0]))
                p = self.quat_params_p.to(device)
                q = self.quat_params_q.to(device)
                p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp(min=EPS)
                unit_p = p / p_norm
                q_norm = torch.norm(q, p=2, dim=-1, keepdim=True).clamp(min=EPS)
                unit_q = q / q_norm
            return unit_p, unit_q
        elif self.rotation_type == 'identity':
            return None
        else:
            raise RuntimeError("Invalid internal rotation type state.")
    def forward(self, v_main: torch.Tensor, v_boundaries_tangent: Optional[torch.Tensor], v_descriptor: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        has_boundaries = v_boundaries_tangent is not None and v_boundaries_tangent.numel() > 0
        batch_size, seq_len, dim = v_main.shape
        if dim != self.dim:
            raise ValueError(f"Rotation input dim mismatch: expected {self.dim}, got {dim} for v_main")
        if has_boundaries and v_boundaries_tangent.shape[-1] != self.dim:
            raise ValueError(f"Rotation input dim mismatch: expected {self.dim}, got {v_boundaries_tangent.shape[-1]} for v_boundaries")
        if v_descriptor.shape[-1] != self.dim:
            if v_descriptor.ndim == 1 and v_descriptor.shape[0] == self.dim:
                v_descriptor = v_descriptor.view(1, 1, self.dim)
            elif v_descriptor.shape != (1, 1, self.dim):
                raise ValueError(f"Rotation input dim mismatch: expected {self.dim}, got {v_descriptor.shape[-1]} for v_descriptor with shape {v_descriptor.shape}")
        device = v_main.device
        v_descriptor = v_descriptor.to(device)
        v_main_rotated, v_boundaries_rotated, v_descriptor_rotated = v_main, v_boundaries_tangent, v_descriptor
        operator = self._get_rotation_operator(device)
        if self.rotation_type == 'identity' or operator is None:
            pass
        elif self.rotation_type == 'so_n':
            # *** Syntax Fix Start ***
            R = operator
            v_main_rotated = torch.matmul(v_main, R)
            if has_boundaries:
                 v_boundaries_rotated = torch.matmul(v_boundaries_tangent, R)
            v_descriptor_rotated = torch.matmul(v_descriptor, R)
            # *** Syntax Fix End ***
        elif self.rotation_type == 'quat':
            unit_p, unit_q = operator
            p_b = unit_p.view(1, 1, 4)
            q_b = unit_q.view(1, 1, 4)
            if has_boundaries:
                p_nb = unit_p.expand(v_boundaries_tangent.shape[:-1] + (4,))
                q_nb = unit_q.expand(v_boundaries_tangent.shape[:-1] + (4,))
                v_boundaries_rotated = quat_rotate_via_pvq(v_boundaries_tangent, p_nb, q_nb)
            else:
                p_nb, q_nb = None, None
            v_main_rotated = quat_rotate_via_pvq(v_main, p_b, q_b)
            v_descriptor_rotated = quat_rotate_via_pvq(v_descriptor, p_b, q_b)
        outputs = [v_main_rotated, v_boundaries_rotated, v_descriptor_rotated]
        names = ["main", "boundaries", "descriptor"]
        final_outputs = []
        for i, out in enumerate(outputs):
            if i == 1 and not has_boundaries:
                final_outputs.append(None)
                continue
            if out is None:
                 raise ValueError(f"Rotation output '{names[i]}' is None unexpectedly.")
            if not torch.isfinite(out).all():
                logger.warning(f"NaN/Inf detected in Rotation output ({names[i]}). Replacing with zeros.")
                out = torch.nan_to_num(out, nan=0.0)
            final_outputs.append(out)
        return tuple(final_outputs)
class InterLevelTransform(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.transform_type = transform_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        if transform_type == 'mlp':
            if hidden_dim is None or hidden_dim <= 0:
                h_dim = max(16, (in_dim + out_dim) // 2)
            else:
                h_dim = hidden_dim
            self.transform = nn.Sequential(nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(h_dim, out_dim))
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using MLP (Hidden Dim: {h_dim})")
        elif transform_type == 'linear':
            self.transform = nn.Linear(in_dim, out_dim)
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using Linear")
        elif transform_type == 'quat':
            check_quat_dim(in_dim, "InterLevelTransform Quat Input")
            check_quat_dim(out_dim, "InterLevelTransform Quat Output")
            self.transform = QuaternionLinear(in_dim, out_dim, bias=True)
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using QuaternionLinear")
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")
        self.transform.apply(init_weights)
    def forward(self, v_rotated: torch.Tensor, v_boundaries_rotated: Optional[torch.Tensor], v_descriptor_rotated: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        has_boundaries = v_boundaries_rotated is not None and v_boundaries_rotated.numel() > 0
        # *** Syntax Fix Start ***
        v_transformed = self.transform(v_rotated)
        v_boundaries_transformed = None
        # *** Syntax Fix End ***
        if has_boundaries:
            v_boundaries_transformed = self.transform(v_boundaries_rotated)
        v_descriptor_transformed = self.transform(v_descriptor_rotated)
        outputs = [v_transformed, v_boundaries_transformed, v_descriptor_transformed]
        names = ["main", "boundaries", "descriptor"]
        final_outputs = []
        for i, out in enumerate(outputs):
            if i == 1 and not has_boundaries:
                final_outputs.append(None)
                continue
            if out is None:
                 raise ValueError(f"Transform output '{names[i]}' is None unexpectedly.")
            if not torch.isfinite(out).all():
                logger.warning(f"NaN/Inf detected in Transform output ({names[i]}). Replacing with zeros.")
                out = torch.nan_to_num(out, nan=0.0)
            final_outputs.append(out)
        return tuple(final_outputs)
class WuBuNestingLevel(nn.Module):
    def __init__(self, level_idx: int, config: Dict):
        super().__init__()
        self.level_idx = level_idx
        self.dim = config["hyperbolic_dims"][level_idx]
        self.relative_vector_aggregation = config["relative_vector_aggregation"]
        self.dropout = config["dropout"]
        self.use_ld = config["use_level_descriptors"]
        self.use_spread = config["use_level_spread"]
        self.use_flow = config["use_tangent_flow"]
        self.curvature_min = config.get("curvature_min_value", EPS)
        self.scale_min = config.get("scale_min_value", EPS)
        self.spread_min = config.get("spread_min_value", EPS)
        self.ld_init_scale = config.get("level_descriptor_init_scale", 0.01)
        if len(config["initial_curvatures"]) <= level_idx:
            raise ValueError(f"Config 'initial_curvatures' list too short for level {level_idx}")
        init_c = config["initial_curvatures"][level_idx]
        if init_c <= self.curvature_min:
            logger.warning(f"Initial curvature {init_c} <= min {self.curvature_min} for Level {level_idx}. Clamping init.")
            init_c = self.curvature_min + EPS
        init_c_unconstrained = torch.tensor(math.log(init_c - self.curvature_min), dtype=torch.float)
        if config["learnable_curvature"]:
            self.unconstrained_curvature = nn.Parameter(init_c_unconstrained)
        else:
            self.register_buffer('unconstrained_curvature', init_c_unconstrained)
        if len(config["initial_scales"]) <= level_idx:
            raise ValueError(f"Config 'initial_scales' list too short for level {level_idx}")
        init_s = config["initial_scales"][level_idx]
        if init_s <= self.scale_min:
            logger.warning(f"Initial scale {init_s} <= min {self.scale_min} for Level {level_idx}. Clamping init.")
            init_s = self.scale_min + EPS
        init_s_unconstrained = torch.tensor(math.log(init_s - self.scale_min), dtype=torch.float)
        if config["learnable_scales"]:
            self.unconstrained_scale = nn.Parameter(init_s_unconstrained)
        else:
            self.register_buffer('unconstrained_scale', init_s_unconstrained)
        if self.use_ld:
            self.level_descriptor_param = nn.Parameter(torch.randn(self.dim) * self.ld_init_scale)
        else:
            self.register_buffer('level_descriptor_param', torch.zeros(self.dim))
        if self.use_spread:
            initial_spread_values_list = config.get("initial_spread_values")
            if initial_spread_values_list is None or len(initial_spread_values_list) <= level_idx:
                logger.debug(f"L{level_idx}: Using initial_scale as initial_spread_value.")
                initial_spread_values_list = config["initial_scales"]
            if len(initial_spread_values_list) <= level_idx:
                raise ValueError(f"Insufficient scale/spread values for level {level_idx} after defaults.")
            init_spread = initial_spread_values_list[level_idx]
            if init_spread <= self.spread_min:
                logger.warning(f"Initial spread {init_spread} <= min {self.spread_min} for Level {level_idx}. Clamping init.")
                init_spread = self.spread_min + EPS
            init_spread_unconstrained = torch.tensor(math.log(init_spread - self.spread_min), dtype=torch.float)
            if config["learnable_spread"]:
                self.unconstrained_spread = nn.Parameter(init_spread_unconstrained)
            else:
                self.register_buffer('unconstrained_spread', init_spread_unconstrained)
        else:
            self.register_buffer('unconstrained_spread', torch.tensor(math.log(EPS)))
        combiner_input_dim = self.dim
        if config["relative_vector_aggregation"] != 'none':
            combiner_input_dim += self.dim
        if self.use_ld:
            combiner_input_dim += self.dim
        if self.use_spread:
            combiner_input_dim += 1
        combiner_hidden_dims = config.get("tangent_input_combination_dims", [max(16, combiner_input_dim // 2)])
        layers = []
        in_d = combiner_input_dim
        for h_dim in combiner_hidden_dims:
            layers.extend([nn.Linear(in_d, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(self.dropout)])
            in_d = h_dim
        layers.append(nn.Linear(in_d, self.dim))
        self.tangent_combiner = nn.Sequential(*layers)
        self.tangent_combiner.apply(init_weights)
        if self.use_flow:
            self.tangent_flow = TangentFlow(dim=self.dim, flow_type=config.get("tangent_flow_type", "mlp"), hidden_dim_ratio=config.get("tangent_flow_hidden_dim_ratio", 0.5), dropout=self.dropout)
            self.tangent_flow_scale = config.get("tangent_flow_scale", 1.0)
        else:
            self.tangent_flow = None
            self.tangent_flow_scale = 0.0
        self.intra_ball_processor = nn.Identity()
        logger.info(f"WuBuNestingLevel {level_idx} (Dim {self.dim}) Init. Learn(c/s/sprd): {config['learnable_curvature']}/{config['learnable_scales']}/{config['learnable_spread']}, Use(LD/Sprd/Flow): {self.use_ld}/{self.use_spread}/{self.use_flow}, CombinerInDim: {combiner_input_dim}")
    def get_current_geometry(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        current_c = get_constrained_params(self.unconstrained_curvature, self.curvature_min).to(device)
        current_s = get_constrained_params(self.unconstrained_scale, self.scale_min).to(device)
        return current_c, current_s
    def get_current_spread(self, device: torch.device) -> torch.Tensor:
        if not self.use_spread:
            return torch.tensor(self.spread_min, device=device, dtype=torch.float)
        elif hasattr(self, 'unconstrained_spread'):
            return get_constrained_params(self.unconstrained_spread, self.spread_min).to(device)
        else:
            logger.error(f"L{self.level_idx}: Spread requested but 'unconstrained_spread' not found!")
            return torch.tensor(self.spread_min, device=device, dtype=torch.float)
    def forward(self, v_tangent_in: torch.Tensor, relative_vectors_in: Optional[torch.Tensor], ld_tangent_in: Optional[torch.Tensor], sigma_in: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_in = v_tangent_in.shape
        device = v_tangent_in.device
        model_dtype = next(self.parameters()).dtype
        if d_in != self.dim:
            raise ValueError(f"L{self.level_idx} input dim mismatch: expected {self.dim}, got {d_in} for v_tangent_in")
        current_c, current_scale = self.get_current_geometry(device)
        current_spread = self.get_current_spread(device)
        inputs_to_combine = [v_tangent_in]
        if self.relative_vector_aggregation != 'none':
            if relative_vectors_in is not None and relative_vectors_in.numel() > 0:
                 if relative_vectors_in.shape == (batch_size, seq_len, self.dim):
                      inputs_to_combine.append(relative_vectors_in.to(model_dtype))
                 else:
                      logger.warning(f"L{self.level_idx}: Unexpected relative_vectors_in shape {relative_vectors_in.shape}. Using zeros.")
                      inputs_to_combine.append(torch.zeros_like(v_tangent_in))
            else:
                 inputs_to_combine.append(torch.zeros_like(v_tangent_in))
        if self.use_ld:
            if ld_tangent_in is None:
                if self.level_idx == 0:
                    logger.debug(f"L0: ld_tangent_in is None, as expected for the first level (no preceding level). Using zeros.")
                else:
                    logger.warning(f"L{self.level_idx}: Expected ld_tangent_in from previous level but received None. Using zeros.")
                inputs_to_combine.append(torch.zeros_like(v_tangent_in))
            elif ld_tangent_in.shape == (batch_size, seq_len, self.dim):
                inputs_to_combine.append(ld_tangent_in.to(model_dtype))
            else:
                logger.error(f"L{self.level_idx}: ld_tangent_in shape mismatch {ld_tangent_in.shape} vs expected {(batch_size, seq_len, self.dim)}. Using zeros.")
                inputs_to_combine.append(torch.zeros_like(v_tangent_in))
        if self.use_spread:
            if sigma_in is None:
                if self.level_idx == 0:
                    logger.debug(f"L0: sigma_in is None, as expected for the first level (no preceding level). Using zeros.")
                else:
                    logger.warning(f"L{self.level_idx}: Expected sigma_in from previous level but received None. Using zeros.")
                sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device, dtype=model_dtype)
            elif sigma_in.numel() == 1:
                sigma_in_tensor = sigma_in.expand(batch_size, seq_len, 1).to(model_dtype)
            elif sigma_in.shape == (batch_size, seq_len):
                sigma_in_tensor = sigma_in.unsqueeze(-1).to(model_dtype)
            elif sigma_in.shape == (batch_size, seq_len, 1):
                sigma_in_tensor = sigma_in.to(model_dtype)
            else:
                logger.error(f"L{self.level_idx}: sigma_in shape mismatch {sigma_in.shape}. Using zeros.")
                sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device, dtype=model_dtype)
            inputs_to_combine.append(sigma_in_tensor)
        combined_tangent_inputs = torch.cat(inputs_to_combine, dim=-1)
        if not torch.isfinite(combined_tangent_inputs).all():
            logger.warning(f"NaN/Inf detected in L{self.level_idx} combined tangent inputs before combiner MLP. Replacing.")
            combined_tangent_inputs = torch.nan_to_num(combined_tangent_inputs, nan=0.0)
        expected_combiner_dim = self.tangent_combiner[0].in_features
        if combined_tangent_inputs.shape[-1] != expected_combiner_dim:
            raise ValueError(f"L{self.level_idx} TangentCombiner input dimension mismatch: expected {expected_combiner_dim}, got {combined_tangent_inputs.shape[-1]}")
        v_tangent_combined = self.tangent_combiner(combined_tangent_inputs)
        flow_displacement = torch.zeros_like(v_tangent_combined)
        if self.use_flow and self.tangent_flow is not None:
            flow_displacement = self.tangent_flow(v_tangent_combined)
        v_tangent_flowed = v_tangent_combined + flow_displacement * self.tangent_flow_scale
        if not torch.isfinite(v_tangent_flowed).all():
            logger.warning(f"NaN/Inf detected in L{self.level_idx} tangent vector post-flow. Replacing.")
            v_tangent_flowed = torch.nan_to_num(v_tangent_flowed, nan=0.0)
        # *** Syntax Fix Start ***
        v_for_exp_map = v_tangent_flowed
        x_hyp = HyperbolicUtils.exponential_map(v_for_exp_map, current_c)
        x_hyp_processed = self.intra_ball_processor(x_hyp)
        x_hyp_processed = HyperbolicUtils.poincare_clip(x_hyp_processed, current_c)
        # *** Syntax Fix End ***
        v_tangent_out = HyperbolicUtils.logarithmic_map(x_hyp_processed, current_c)
        if not torch.isfinite(v_tangent_out).all():
            logger.warning(f"NaN/Inf detected in L{self.level_idx} output tangent vector (v_tangent_out). Replacing.")
            v_tangent_out = torch.nan_to_num(v_tangent_out, nan=0.0)
        # *** Syntax Fix Start ***
        ld_param_out = self.level_descriptor_param
        sigma_param_out = current_spread
        # *** Syntax Fix End ***
        v_tangent_out = v_tangent_out.to(model_dtype)
        ld_param_out = ld_param_out.to(model_dtype)
        sigma_param_out = sigma_param_out.to(model_dtype)
        return x_hyp_processed, v_tangent_out, ld_param_out, sigma_param_out

# =====================================================================
# WuBuNestingSequenceModel (Self-Contained)
# =====================================================================
class WuBuNestingSequenceModel(nn.Module):
    """WuBu Nesting model adapted for sequence modeling."""
    def __init__(self, wubu_config: Dict, sequence_config: Dict):
        super().__init__()
        self.wubu_config = wubu_config
        self.sequence_config = sequence_config

        num_levels = wubu_config["num_levels"]
        hyperbolic_dims = wubu_config["hyperbolic_dims"]
        boundary_points = wubu_config["boundary_points_per_level"]
        rotation_types = wubu_config["rotation_types"]
        transform_types = wubu_config["transform_types"]
        transform_hdims = wubu_config["transform_hidden_dims"]
        if len(hyperbolic_dims) != num_levels: raise ValueError("Length of hyperbolic_dims must match num_levels")
        if len(boundary_points) != num_levels: raise ValueError("Length of boundary_points_per_level must match num_levels")
        num_transitions = max(0, num_levels - 1)
        if len(rotation_types) != num_transitions: raise ValueError(f"Length of rotation_types ({len(rotation_types)}) must be num_levels - 1 ({num_transitions})")
        if len(transform_types) != num_transitions: raise ValueError(f"Length of transform_types ({len(transform_types)}) must be num_levels - 1 ({num_transitions})")
        if transform_hdims is None:
            transform_hdims = [None] * num_transitions
        elif len(transform_hdims) != num_transitions:
            raise ValueError(f"Length of transform_hidden_dims ({len(transform_hdims)}) must be num_levels - 1 ({num_transitions})")
        self.transform_hdims = [d if d is not None and d > 0 else None for d in transform_hdims]

        self.local_hidden_size = sequence_config["local_hidden_size"]
        self.decoder_memory_dim = sequence_config["decoder_memory_dim"]
        self.context_window = sequence_config["context_window"]
        self.vocab_size = sequence_config.get("vocab_size", 256)
        if self.vocab_size != 256: logger.warning(f"Sequence vocab_size set to {self.vocab_size}, but model typically uses 256 for bytes.")

        # *** CRITICAL FIX: Use the correct patcher class ***
        self.patcher = HAKMEMBabylonIndex()
        # **************************************************

        self.local_encoder = HAKMEMLocalEncoder(
            hidden_size=self.local_hidden_size,
            num_layers=sequence_config.get("num_local_encoder_layers", 2),
            num_heads=sequence_config.get("num_local_encoder_heads", 8),
            dropout=wubu_config.get("dropout", 0.1),
            n_gram_sizes=sequence_config.get("n_gram_sizes", [3, 4]),
            n_gram_vocab_size=sequence_config.get("n_gram_vocab_size", 30000)
        )

        self.to_first_tangent = nn.Linear(self.local_hidden_size, hyperbolic_dims[0])
        self.to_first_tangent.apply(init_weights)
        self.levels = nn.ModuleList([WuBuNestingLevel(i, wubu_config) for i in range(num_levels)])
        self.boundaries = nn.ModuleList([BoundaryManifold(i, boundary_points[i], hyperbolic_dims[i], init_scale=wubu_config.get('level_descriptor_init_scale', 0.01)) for i in range(num_levels)])
        self.rotations = nn.ModuleList()
        self.transforms = nn.ModuleList()
        if num_levels > 1:
            self.rotations = nn.ModuleList([TangentSpaceRotation(hyperbolic_dims[i], rotation_types[i]) for i in range(num_transitions)])
            self.transforms = nn.ModuleList([InterLevelTransform(hyperbolic_dims[i], hyperbolic_dims[i+1], transform_types[i], self.transform_hdims[i], wubu_config.get("dropout", 0.1)) for i in range(num_transitions)])

        self.aggregation_method = wubu_config["aggregation_method"]
        if self.aggregation_method == "concat_tangent":
            total_tangent_dim = sum(hyperbolic_dims)
            self.projection_to_decoder_memory = nn.Linear(total_tangent_dim, self.decoder_memory_dim)
            self.projection_to_decoder_memory.apply(init_weights)
            logger.info(f"Aggregation: Concatenating tangent outputs (Total Dim: {total_tangent_dim}) -> Decoder Memory Dim: {self.decoder_memory_dim}")
        else:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not yet supported.")

        self.local_decoder = HAKMEMLocalDecoder(
            hidden_size=self.local_hidden_size,
            global_hidden_size=self.decoder_memory_dim,
            num_layers=sequence_config.get("num_local_decoder_layers", 4),
            num_heads=sequence_config.get("num_local_decoder_heads", 8),
            dropout=wubu_config.get("dropout", 0.1),
            use_hierarchical_pred=sequence_config.get("use_hierarchical_decoder", False),
            max_decode_len=max(self.context_window * 2, 2048)
        )
        logger.info("WuBuNestingSequenceModel Initialized.") # Removed "Full Decoder/Optimizer"


    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = byte_seq.shape
        device = byte_seq.device
        model_dtype = next(self.parameters()).dtype
        batch_patch_repr_list = []
        num_patches_per_item = []
        valid_batch_indices = []
        max_num_patches = 0

        for i in range(batch_size):
            seq = byte_seq[i]
            patches_with_entropy = self.patcher.create_patches(seq) # Use the instantiated patcher
            if patches_with_entropy:
                patches_on_device = [(p.to(device), e) for p, e in patches_with_entropy]
                patch_repr_single = self.local_encoder(patches_on_device)
                if not torch.isfinite(patch_repr_single).all():
                    logger.warning(f"NaN/Inf detected in local encoder output for batch item {i}. Replacing.")
                    patch_repr_single = torch.nan_to_num(patch_repr_single, nan=0.0)
                num_p = patch_repr_single.size(1)
                if num_p > 0:
                    batch_patch_repr_list.append(patch_repr_single.squeeze(0))
                    num_patches_per_item.append(num_p)
                    valid_batch_indices.append(i)
                    max_num_patches = max(max_num_patches, num_p)
                else:
                    num_patches_per_item.append(0)
            else:
                num_patches_per_item.append(0)

        target_len = 0
        if target_byte_seq is not None:
            target_len = target_byte_seq.size(1)
        if not valid_batch_indices:
            logger.warning(f"No valid patches found for any item in the batch (Batch Size: {batch_size}). Returning zero logits.")
            return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)
        num_valid = len(valid_batch_indices)
        if max_num_patches == 0:
            logger.warning(f"Valid batch items exist, but max_num_patches is zero. Returning zero logits.")
            return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)

        padded_patch_repr = torch.zeros(num_valid, max_num_patches, self.local_hidden_size, device=device, dtype=model_dtype)
        memory_padding_mask = torch.ones(num_valid, max_num_patches, dtype=torch.bool, device=device)
        valid_item_counter = 0
        for i in range(batch_size):
            if i in valid_batch_indices:
                patch_repr_tensor = batch_patch_repr_list[valid_item_counter]
                num_p = num_patches_per_item[i]
                if num_p > 0:
                    padded_patch_repr[valid_item_counter, :num_p, :] = patch_repr_tensor
                    memory_padding_mask[valid_item_counter, :num_p] = False
                valid_item_counter += 1

        current_tangent_main = self.to_first_tangent(padded_patch_repr)
        level_tangent_outputs = []
        current_ld_tangent = None
        current_sigma = None
        relative_vectors = None

        for i in range(self.wubu_config["num_levels"]):
            level_module = self.levels[i]
            boundary_module = self.boundaries[i]
            v_boundaries_tangent_origin = boundary_module.get_tangent_vectors_at_origin()
            if v_boundaries_tangent_origin is not None:
                v_boundaries_tangent_origin = v_boundaries_tangent_origin.to(device=device, dtype=model_dtype)
            _, v_tangent_i_out, ld_i_param, sigma_i_param = level_module(v_tangent_in=current_tangent_main, relative_vectors_in=relative_vectors, ld_tangent_in=current_ld_tangent, sigma_in=current_sigma)
            level_tangent_outputs.append(v_tangent_i_out)
            if i < self.wubu_config["num_levels"] - 1:
                rotation_module = self.rotations[i]
                transform_module = self.transforms[i]
                ld_i_param_dev = ld_i_param.to(device=device, dtype=model_dtype).view(1, 1, -1)
                v_main_rotated, v_boundaries_rotated, ld_rotated = rotation_module(v_main=v_tangent_i_out, v_boundaries_tangent=v_boundaries_tangent_origin, v_descriptor=ld_i_param_dev)
                v_next_tangent_main, v_boundaries_transformed, ld_next_tangent = transform_module(v_rotated=v_main_rotated, v_boundaries_rotated=v_boundaries_rotated, v_descriptor_rotated=ld_rotated)
                relative_vectors = None # Simplified path
                current_tangent_main = v_next_tangent_main
                current_ld_tangent = ld_next_tangent.expand(num_valid, max_num_patches, -1)
                current_sigma = sigma_i_param

        if self.aggregation_method == "concat_tangent":
            try:
                aggregated_repr = torch.cat(level_tangent_outputs, dim=-1)
            except RuntimeError as e:
                logger.error(f"Error concatenating level outputs: {e}")
                logger.error(f"Shapes: {[t.shape for t in level_tangent_outputs]}")
                return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not implemented.")

        decoder_memory = self.projection_to_decoder_memory(aggregated_repr)
        if not torch.isfinite(decoder_memory).all():
            logger.warning("NaN/Inf detected in projected decoder memory. Replacing.")
            decoder_memory = torch.nan_to_num(decoder_memory, nan=0.0)

        if target_byte_seq is None:
            logger.debug("Forward called without target_byte_seq for decoding. Returning zeros.")
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)
        if target_byte_seq.size(1) == 0:
            logger.debug("Forward called with empty target_byte_seq. Returning zeros.")
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)

        valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
        if num_valid > 0:
            valid_target_byte_seq = torch.index_select(target_byte_seq, 0, valid_indices_tensor).to(device).long()
        else:
            valid_target_byte_seq = torch.empty(0, target_len, dtype=torch.long, device=device)

        if num_valid > 0:
            byte_logits_valid = self.local_decoder(tgt_byte_seq=valid_target_byte_seq, memory=decoder_memory, memory_key_padding_mask=memory_padding_mask)
            if not torch.isfinite(byte_logits_valid).all():
                logger.warning("NaN/Inf detected in decoder logits output. Replacing.")
                byte_logits_valid = torch.nan_to_num(byte_logits_valid, nan=0.0)
        else:
            byte_logits_valid = torch.empty((0, target_len, self.vocab_size), device=device, dtype=torch.float32)

        final_byte_logits = torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)
        if num_valid > 0 and byte_logits_valid.numel() > 0:
            try:
                if byte_logits_valid.shape[0] == valid_indices_tensor.shape[0]:
                    final_byte_logits.index_copy_(0, valid_indices_tensor, byte_logits_valid)
                else:
                    logger.error(f"Shape mismatch during output reconstruction. Target Batch={batch_size}, Valid Items={num_valid}, Logits Shape={byte_logits_valid.shape}, Indices Shape={valid_indices_tensor.shape}. Cannot perform index_copy.")
            except IndexError as e:
                logger.error(f"Index error during logit reconstruction: {e}. NumValid: {num_valid}, IdxShape: {valid_indices_tensor.shape}, LogitsShape: {byte_logits_valid.shape}")
            except Exception as e_scatter:
                logger.error(f"Unexpected error scattering logits: {e_scatter}")

        if not torch.isfinite(final_byte_logits).all():
            logger.error("NaN/Inf detected in final scattered logits! Replacing with zeros.")
            final_byte_logits = torch.nan_to_num(final_byte_logits, nan=0.0)
        return final_byte_logits


    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        if seq_len <= 1:
            return torch.tensor(0.0, device=logits.device, requires_grad=True, dtype=logits.dtype)
        logits_shifted = logits[:, :-1, :].contiguous()
        targets_shifted = targets[:, 1:].contiguous()
        logits_flat = logits_shifted.view(-1, vocab_size)
        targets_flat = targets_shifted.view(-1)
        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)
        if not torch.isfinite(logits_flat).all():
            num_nan = torch.isnan(logits_flat).sum().item()
            num_inf = torch.isinf(logits_flat).sum().item()
            logger.error(f"NaN/Inf detected in logits passed to compute_loss (NaNs: {num_nan}, Infs: {num_inf}). Returning high loss.")
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        loss: torch.Tensor
        if smoothing > 0.0 and 0.0 < smoothing < 1.0:
            with torch.no_grad():
                smooth_val_on = 1.0 - smoothing
                smooth_val_off = smoothing / max(1, vocab_size - 1)
                true_dist = torch.full_like(logits_flat, smooth_val_off)
                true_dist.scatter_(1, targets_flat.unsqueeze(1), smooth_val_on)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -(true_dist * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(logits_flat.float(), targets_flat.long(), reduction='none')
        if not torch.isfinite(loss).all():
             logger.error(f"NaN/Inf loss calculated by cross_entropy/smoothing. Returning high loss.")
             loss = torch.nan_to_num(loss, nan=100.0, posinf=100.0, neginf=-100.0)
        mean_loss: torch.Tensor
        if mask is not None:
            mask_shifted = mask[:, 1:].contiguous()
            if mask_shifted.shape == targets_shifted.shape:
                 mask_flat = mask_shifted.view(-1)
                 mask_flat_bool = mask_flat.bool() if mask_flat.dtype == torch.bool else mask_flat > 0
                 loss = loss.masked_fill(mask_flat_bool, 0.0)
                 num_active_elements = (~mask_flat_bool).sum()
                 if num_active_elements.item() > 0:
                    mean_loss = loss.sum() / num_active_elements
                 else:
                    logger.warning("All target elements masked in loss calculation. Returning zero loss.")
                    mean_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            else:
                 logger.warning(f"Mask shape mismatch ({mask.shape[-1]}) vs target sequence length ({targets.shape[-1]}) for loss calculation. Ignoring mask.")
                 mean_loss = loss.mean()
        else:
             mean_loss = loss.mean()
        if not torch.isfinite(mean_loss):
            logger.error(f"NaN/Inf final mean loss detected. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        return mean_loss


    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None, repetition_penalty: float = 1.1, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        if seed_bytes.device != device:
            seed_bytes = seed_bytes.to(device)
        seed_bytes = seed_bytes.long()
        batch_size, seed_len = seed_bytes.size()
        if seed_len == 0:
            logger.warning("Empty seed provided for generation. Returning empty tensor.")
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        generated_sequence = seed_bytes.clone()
        if sampling_config is None:
            sampling_config = SamplerConfig()
        # *** Syntax Fix Start ***
        base_temperature = max(temperature, EPS)
        min_temp = max(0.1, base_temperature * 0.5)
        max_temp = min(2.0, base_temperature * 1.5)
        # *** Syntax Fix End ***
        disable_tqdm = batch_size > 8 or max_length < 20
        gen_iterator = tqdm(range(max_length), desc="Generating", disable=disable_tqdm, total=max_length, unit="byte", leave=False)

        for step in gen_iterator:
            current_context = generated_sequence.long()
            context_len = current_context.size(1)
            amp_context = amp.autocast(device_type=device.type, enabled=False)
            with torch.no_grad(), amp_context:
                logits_all = self(byte_seq=current_context, target_byte_seq=current_context)
            if logits_all is None or logits_all.numel() == 0 or logits_all.shape[1] == 0:
                 logger.warning(f"Logits generation failed at step {step}. Stopping generation.")
                 break
            if not torch.isfinite(logits_all).all():
                logger.warning(f"NaN/Inf detected in generated logits at step {step}. Using uniform distribution for this step.")
                logits_all = torch.zeros_like(logits_all)
            next_byte_logits = logits_all[:, -1, :].float()
            next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                current_logits = next_byte_logits[i].clone()
                current_seq = generated_sequence[i]
                current_seq_len = current_seq.size(0)
                if repetition_penalty > 1.0 and current_seq_len > 0:
                    seen_bytes = torch.unique(current_seq)
                    for byte_val_tensor in seen_bytes:
                        byte_val = byte_val_tensor.item()
                        if 0 <= byte_val < self.vocab_size:
                            if current_logits[byte_val] > 0:
                                current_logits[byte_val] /= repetition_penalty
                            else:
                                current_logits[byte_val] *= repetition_penalty
                adaptive_temp = base_temperature
                try:
                    probs_orig = F.softmax(current_logits, dim=-1)
                    if torch.isnan(probs_orig).any():
                         entropy = math.log2(self.vocab_size)
                         logger.debug(f"Item {i} step {step}: NaN in probs_orig, using max entropy.")
                    else:
                         entropy = -torch.sum(probs_orig * torch.log2(probs_orig + EPS)).item()
                    if entropy < sampling_config.low_entropy_threshold:
                        adaptive_temp *= 0.8
                    elif entropy > sampling_config.medium_entropy_threshold:
                        adaptive_temp *= 1.1
                    adaptive_temp = max(min_temp, min(adaptive_temp, max_temp))
                except Exception as e_entropy:
                    logger.warning(f"Error calculating entropy/adaptive temp for item {i} step {step}: {e_entropy}. Using base temperature.")
                    adaptive_temp = base_temperature
                # *** Syntax Fix Start ***
                scaled_logits = current_logits / adaptive_temp
                filtered_logits = scaled_logits
                # *** Syntax Fix End ***
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    # *** Syntax Fix Start ***
                    if k > 0:
                        top_k_threshold = torch.topk(filtered_logits, k)[0][..., -1, None]
                        indices_to_remove = filtered_logits < top_k_threshold
                        filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))
                    # *** Syntax Fix End ***
                    else:
                         filtered_logits.fill_(-float('Inf'))
                if top_p is not None and 0.0 < top_p < 1.0:
                    # *** Syntax Fix Start ***
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # *** Syntax Fix End ***
                    # *** Syntax Fix Start ***
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    # *** Syntax Fix End ***
                    # *** Syntax Fix Start ***
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))
                    # *** Syntax Fix End ***
                probs_final = F.softmax(filtered_logits, dim=-1)
                if torch.isnan(probs_final).any() or torch.isinf(probs_final).any() or probs_final.sum() < EPS:
                    logger.warning(f"Invalid final probability distribution for item {i} step {step}. Sampling uniformly.")
                    probs_final = torch.ones_like(current_logits) / current_logits.size(-1)
                if temperature <= EPS:
                    next_byte_idx = torch.argmax(probs_final)
                else:
                    next_byte_idx = torch.multinomial(probs_final, num_samples=1).squeeze(-1)
                next_byte_indices[i] = next_byte_idx.item()
            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)
            if not disable_tqdm:
                gen_iterator.set_description(f"Generating (Len {generated_sequence.size(1)})")
        if not disable_tqdm:
            gen_iterator.close()
        return generated_sequence


# =====================================================================
# Loading and Generation Functions
# =====================================================================

DEFAULT_CONFIG_WUBU = { # Copied from Trainer for reconstruction
    "num_levels": 3, "hyperbolic_dims": [128, 64, 32], "boundary_points_per_level": [5, 5, 5],
    "initial_curvatures": [1.0, 1.0, 1.0], "initial_scales": [1.0, 1.0, 1.0], "initial_spread_values": None,
    "learnable_curvature": True, "learnable_scales": True, "learnable_spread": True,
    "curvature_min_value": 1e-5, "scale_min_value": 1e-5, "spread_min_value": 1e-5,
    "use_level_descriptors": True, "level_descriptor_init_scale": 0.01, "use_level_spread": True,
    "rotation_types": ["so_n", "so_n"], "transform_types": ["mlp", "mlp"], "transform_hidden_dims": [None, None],
    "use_tangent_flow": True, "tangent_flow_type": "mlp", "tangent_flow_hidden_dim_ratio": 0.5, "tangent_flow_scale": 1.0,
    "relative_vector_aggregation": "mean", "tangent_input_combination_dims": [64],
    "aggregation_method": "concat_tangent", "dropout": 0.1,
}

def reconstruct_configs_from_args(loaded_args) -> Tuple[Dict, Dict]:
    """Reconstructs WuBu and Sequence configs from loaded argparse Namespace."""
    # WuBu Config
    wubu_config = DEFAULT_CONFIG_WUBU.copy() # Start with defaults
    wubu_arg_keys = [
        "num_levels", "hyperbolic_dims", "initial_curvatures", "initial_scales",
        "initial_spread_values", "boundary_points_per_level", "rotation_types",
        "transform_types", "transform_hidden_dims", "tangent_flow_type",
        "tangent_flow_scale", "aggregation_method", "relative_vector_aggregation",
        "dropout", "level_descriptor_init_scale", "curvature_min_value",
        "scale_min_value", "spread_min_value", "tangent_input_combination_dims"
    ]
    # Override defaults with values from loaded_args if they exist
    for key in wubu_arg_keys:
        if hasattr(loaded_args, key) and getattr(loaded_args, key) is not None:
             wubu_config[key] = getattr(loaded_args, key)
    # Handle boolean flags (saved args store False if flag was present, True if absent -> so need `not getattr`)
    wubu_config["learnable_curvature"] = not getattr(loaded_args, 'no_learnable_curvature', False)
    wubu_config["learnable_scales"] = not getattr(loaded_args, 'no_learnable_scales', False)
    wubu_config["learnable_spread"] = not getattr(loaded_args, 'no_learnable_spread', False)
    wubu_config["use_level_descriptors"] = not getattr(loaded_args, 'no_level_descriptors', False)
    wubu_config["use_level_spread"] = not getattr(loaded_args, 'no_level_spread', False)
    wubu_config["use_tangent_flow"] = not getattr(loaded_args, 'no_tangent_flow', False)

    # Sequence Model Config
    sequence_config = {}
    seq_arg_keys = [
        "local_hidden_size", "decoder_memory_dim", "context_window",
        "n_gram_sizes", "n_gram_vocab_size",
        "num_encoder_layers", "num_decoder_layers",
        "num_encoder_heads", "num_decoder_heads"
    ]
    # Provide defaults for keys potentially missing in older checkpoints
    sequence_config_defaults = {
        "local_hidden_size": 256, "decoder_memory_dim": 512, "context_window": 256,
        "n_gram_sizes": [3, 4], "n_gram_vocab_size": 30000,
        "num_encoder_layers": 2, "num_decoder_layers": 4,
        "num_encoder_heads": 8, "num_decoder_heads": 8
    }
    for key in seq_arg_keys:
        sequence_config[key] = getattr(loaded_args, key, sequence_config_defaults.get(key)) # Use default if not in args

    sequence_config["use_hierarchical_decoder"] = not getattr(loaded_args, 'no_hierarchical_decoder', False)
    sequence_config["vocab_size"] = 256 # Fixed for byte-level

    logger.info("Reconstructed configurations from checkpoint args.")
    return wubu_config, sequence_config


def load_model(checkpoint_path: str, device: torch.device) -> WuBuNestingSequenceModel:
    """Loads the WuBuNestingSequenceModel from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        # Load onto CPU first to prevent GPU memory issues if checkpoint device differs
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'args' not in checkpoint or checkpoint['args'] is None:
            logger.error("Checkpoint is missing required 'args' field for model reconstruction.")
            raise ValueError("Checkpoint does not contain necessary configuration arguments ('args'). Cannot reconstruct model.")

        loaded_args = checkpoint['args']
        # Reconstruct the necessary config dictionaries from the loaded args
        wubu_config, sequence_config = reconstruct_configs_from_args(loaded_args)

        logger.info("Instantiating WuBuNestingSequenceModel...")
        model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config)

        if 'model_state_dict' not in checkpoint:
             raise KeyError("Checkpoint is missing 'model_state_dict'.")

        logger.info("Loading model state dictionary...")
        # Use strict=False for more robustness during inference loading
        # If loading a DDP-saved model, keys might have 'module.' prefix. Handle this.
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present
        if all(key.startswith('module.') for key in state_dict.keys()):
            logger.info("Removing 'module.' prefix from state dictionary keys.")
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys:
             logger.warning(f"Missing keys when loading model state: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
             logger.warning(f"Unexpected keys when loading model state: {incompatible_keys.unexpected_keys}")

        model.to(device)
        model.eval() # Set model to evaluation mode
        logger.info(f"Model loaded successfully and moved to {device}. Evaluation mode set.")

        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded Model Parameters: Total={total_params:,}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model from checkpoint '{checkpoint_path}': {e}", exc_info=True)
        raise

def generate_text(model: WuBuNestingSequenceModel, seed_text: str, max_length: int, temperature: float, device: torch.device, repetition_penalty: float, top_k: Optional[int], top_p: Optional[float]) -> str:
    """Generates text using the loaded model."""
    tokenizer = ByteTokenizer() # Used for text <-> byte conversion

    logger.info(f"Encoding seed text: '{seed_text}'")
    seed_byte_list = tokenizer.encode(seed_text)
    if not seed_byte_list:
        logger.warning("Seed text resulted in empty byte sequence. Using a default seed byte (newline).")
        # Use a default byte like newline if seed is empty
        seed_byte_list = [10] # ASCII for newline

    seed_tensor = torch.tensor(seed_byte_list, dtype=torch.long, device=device).unsqueeze(0) # Shape: [1, seed_len]

    logger.info(f"Starting generation (Max new bytes: {max_length}, Temp: {temperature}, RepPen: {repetition_penalty}, TopK: {top_k}, TopP: {top_p})")
    start_time = time.time()

    # Prepare generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "temperature": temperature,
        "sampling_config": SamplerConfig(), # Use default adaptive sampling config
        "repetition_penalty": repetition_penalty,
        "top_k": top_k if top_k is not None and top_k > 0 else None,
        "top_p": top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
    }

    # Call the model's generate method
    with torch.no_grad():
        generated_tensor = model.generate(seed_bytes=seed_tensor, **gen_kwargs)

    end_time = time.time()
    duration = end_time - start_time
    num_generated_bytes = generated_tensor.size(1) - seed_tensor.size(1)
    logger.info(f"Generation complete. Generated {num_generated_bytes} bytes in {duration:.2f} seconds.")

    # Decode the full output tensor (seed + generated) back to text
    generated_byte_sequence = generated_tensor.squeeze(0).cpu().tolist() # Get list of ints
    generated_text_output = tokenizer.decode(generated_byte_sequence)

    return generated_text_output

# =====================================================================
# Main Function
# =====================================================================
def main():
    """Main function to parse arguments, load the model, and generate text."""
    parser = argparse.ArgumentParser(description="WuBu Nesting Sequence Model Inference")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--seed_text', type=str, default="Hello world,", help='Initial text to seed the generation')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum number of *new* bytes to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (higher = more random, 0 = greedy)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (cuda or cpu)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Repetition penalty (1.0 = no penalty)')
    parser.add_argument('--top_k', type=int, default=0, help='Top-K filtering (0 or None = disable)')
    parser.add_argument('--top_p', type=float, default=0.0, help='Top-P (nucleus) filtering (0.0 or None = disable)')

    args = parser.parse_args()

    # Select device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA specified but not available, using CPU instead.")
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
        try:
            gpu_name = torch.cuda.get_device_name(device)
            logger.info(f"Using CUDA device: {gpu_name}")
        except Exception as e:
            logger.warning(f"Could not get CUDA device name: {e}. Using CUDA device {device}.")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device.")

    try:
        # Load the model
        model = load_model(args.checkpoint_path, device)

        # Generate text
        generated_text = generate_text(model=model, seed_text=args.seed_text, max_length=args.max_length,
                                     temperature=args.temperature, device=device,
                                     repetition_penalty=args.repetition_penalty,
                                     top_k=args.top_k, top_p=args.top_p)

        # Print the output
        print("\n" + "="*50)
        print(f"Seed Text:    '{args.seed_text}'")
        print("-" * 50)
        print(f"Generated Text:\n{generated_text}") # Print with newline for better readability
        print("="*50)
        logger.info("Generation process finished.")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except ValueError as e:
        logger.error(f"Configuration or Value Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    print("\n****************************")
    print("* WuBu Nesting Inference *")
    print("****************************\n")
    main()