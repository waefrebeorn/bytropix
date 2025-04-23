# -*- coding: utf-8 -*-
"""
Integrated HyperHAKMEM-BSFIN Model 

Combines:
- HAKMEM-inspired computational enhancements (from hackmemclaude.py concepts)
- Hyperbolic geometry concepts (from HypBSFIN.py concepts)
- Base Trainer and Data Loading infrastructure (from bsfin_main.py concepts)

Architecture:
1. HAKMEM BabylonIndex Patching (Word/Punctuation based, with Patch Entropy)
2. HAKMEM LocalEncoder (Euclidean Patch Encoding with N-grams per patch)
3. Projection to Euclidean space for Hyperbolic Embedding
4. Poincaré Clipping + Exponential Map (to Hyperbolic Space)
5. Logarithmic Map (back to Euclidean Tangent Space)
6. Projection from Tangent Space to Complex Domain (Real/Imag)
7. HAKMEM Complex Positional Encoding (Learnable RoPE-like)
8. Stack of HAKMEM Complex LayerNorm + HAKMEM EntangledInterferenceLayer (Complex Attention with RoPE + Stability Fixes)
9. HAKMEM Complex To Real Projection (from complex tangent space)
10. Final Projection to Decoder Memory Dimension
11. HAKMEM LocalDecoder for Byte Prediction (with Cross-Attention)
12. HAKMEM EnhancedSGD Optimizer (with Q-Learning Controller)

"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler
import numpy as np
import math
import random
import argparse
import logging
import time
import contextlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque
import gc
import socket
import platform
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp  # For automatic mixed precision
from dataclasses import dataclass
import itertools
from tqdm import tqdm
import inspect
import string

# Try importing wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # Set wandb to None if not available
    WANDB_AVAILABLE = False

# --- Central Logger Setup ---
logger = logging.getLogger("IntegratedHyperHAKMEM")
# Ensure basicConfig is only called once if module is reloaded
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)

# =====================================================================
# Data Structures and Configuration Classes
# =====================================================================

@dataclass
class SamplerConfig:
    """Configuration for entropy-based sampling (used in generate)."""
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

class GradientStats:
    """Tracks gradient statistics."""
    def __init__(self): self.reset()
    def reset(self):
        self.total_gradients = 0; self.clipped_gradients = 0
        self.max_gradient_norm = 0.0; self.sum_clip_ratios = 0.0
        self.step_stats = {}
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        # Only record finite norms
        if np.isfinite(original_norm):
            self.total_gradients += 1
            self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
            if clipped:
                self.clipped_gradients += 1
                self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)
    def get_step_stats(self) -> dict:
        if self.total_gradients == 0:
            return {"gradients_clipped": 0, "total_gradients": 0, "clip_ratio_avg": 0.0, "max_gradient": 0.0, "clip_percentage": 0.0}
        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100 if self.total_gradients > 0 else 0.0
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0
        return {"gradients_clipped": self.clipped_gradients, "total_gradients": self.total_gradients, "clip_ratio_avg": avg_clip_ratio, "max_gradient": self.max_gradient_norm, "clip_percentage": clip_percentage}
    def record_step(self, step: int):
        stats = self.get_step_stats()
        self.step_stats[step] = stats
        self.reset() # Reset stats after recording for a step
        return stats

# =====================================================================
# HAKMEM-Inspired Entropy Calculation Helper
# =====================================================================
class HAKMEMEntropyHelper:
    """ Encapsulates HAKMEM-inspired Shannon entropy calculation logic. """
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_cache = {}
        self.max_cache_size = max_cache_size

    def _clean_cache(self):
        # Simple cache eviction: remove oldest items if over size limit
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5) # Remove 20% overshoot
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache: del self.entropy_cache[k]

    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...], List[int], bytes, torch.Tensor]) -> float:
        """ Computes Shannon entropy in bits for a window/sequence of bytes. """
        cache_key = None
        byte_list = []

        if isinstance(byte_window, tuple):
            cache_key = byte_window
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
            if not byte_window: return 0.0
            byte_list = list(byte_window)
        elif isinstance(byte_window, list):
             if not byte_window: return 0.0
             cache_key = tuple(byte_window) # Use tuple as key for lists
             if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
             byte_list = byte_window
        elif isinstance(byte_window, bytes):
            if not byte_window: return 0.0
            cache_key = byte_window # bytes are hashable
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
            byte_list = list(byte_window)
        elif isinstance(byte_window, np.ndarray):
            if byte_window.size == 0: return 0.0
            byte_list = byte_window.tolist()
            cache_key = tuple(byte_list) # Convert numpy to tuple for key
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        elif isinstance(byte_window, torch.Tensor):
            if byte_window.numel() == 0: return 0.0
            byte_list = byte_window.cpu().byte().tolist() # Ensure bytes
            cache_key = tuple(byte_list) # Convert tensor to tuple for key
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        else:
            logger.warning(f"compute_entropy received unsupported type: {type(byte_window)}. Returning 0.")
            return 0.0

        try:
            if not byte_list: return 0.0
            # Use numpy for efficient calculation
            byte_counts = np.bincount(np.array(byte_list, dtype=np.uint8), minlength=256)
            total_bytes = byte_counts.sum()
            if total_bytes == 0: return 0.0

            probs = byte_counts[byte_counts > 0] / total_bytes
            # Use log base 2 for entropy in bits
            entropy = float(-np.sum(probs * np.log2(probs + 1e-9))) # Add epsilon for stability
            result = max(0.0, entropy) # Ensure non-negative entropy

            if cache_key is not None:
                self.entropy_cache[cache_key] = result
                self._clean_cache() # Clean cache after adding
            return result
        except Exception as e:
            logger.warning(f"Error during entropy calculation for window (size {len(byte_list)}): {e}", exc_info=False)
            return 0.0

# =====================================================================
# HAKMEM Babylon Index (Patching - Word/Punctuation Based V3)
# =====================================================================
class HAKMEMBabylonIndex:
    """
    Splits byte sequences into patches based on whitespace and punctuation delimiters.
    Calculates Shannon entropy for each generated patch. (V3)
    """
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_helper = HAKMEMEntropyHelper(max_cache_size)
        # Use pre-defined string constants for whitespace and punctuation
        self.whitespace_chars = set(string.whitespace)
        self.punctuation_chars = set(string.punctuation)
        # Add common unicode punctuation/separators if needed
        # self.punctuation_chars.update(['’', '‘', '“', '”', '—', '–'])
        logger.info("HAKMEMBabylonIndex V3 initialized (Word/Punctuation Patching with Entropy).")

    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """
        Splits a byte sequence tensor into patches based on whitespace/punctuation.
        Returns a list of tuples: (patch_tensor, patch_entropy).
        """
        if byte_seq_tensor.numel() == 0: return []
        if byte_seq_tensor.dim() != 1:
            logger.warning(f"BabylonIndex expected 1D tensor, got {byte_seq_tensor.shape}. Flattening.")
            byte_seq_tensor = byte_seq_tensor.flatten()
        if byte_seq_tensor.numel() == 0: return []

        device = byte_seq_tensor.device
        try:
            # Decode bytes to string for easier splitting
            # Using tobytes() is generally more reliable than list() for byte conversion
            text = byte_seq_tensor.cpu().numpy().tobytes().decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error decoding byte tensor (shape {byte_seq_tensor.shape}, numel {byte_seq_tensor.numel()}) to string: {e}. Returning no patches.")
            return []

        patches_with_entropy = []
        current_patch_start = 0
        in_word = False # Are we currently accumulating a word/token?

        for i, char in enumerate(text):
            is_delimiter = char in self.whitespace_chars or char in self.punctuation_chars

            if is_delimiter:
                if in_word:
                    # End of a word patch before the delimiter
                    word_str = text[current_patch_start:i]
                    try:
                        word_bytes = torch.tensor(list(word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                        if word_bytes.numel() > 0:
                            entropy = self.entropy_helper.compute_entropy(word_bytes)
                            patches_with_entropy.append((word_bytes, entropy))
                    except Exception as enc_e:
                        logger.warning(f"Error encoding word patch '{word_str[:20]}...': {enc_e}")
                    in_word = False # Word ended

                # Create patch for the delimiter character(s)
                try:
                    # Potentially handle multi-char delimiters if needed, currently single char
                    delim_bytes = torch.tensor(list(char.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                    if delim_bytes.numel() > 0:
                         entropy = self.entropy_helper.compute_entropy(delim_bytes)
                         patches_with_entropy.append((delim_bytes, entropy))
                except Exception as enc_e:
                    logger.warning(f"Error encoding delimiter patch '{char}': {enc_e}")

                current_patch_start = i + 1 # Start next potential patch after delimiter
            else: # Regular character (part of a word)
                if not in_word:
                    # Start of a new word/token patch
                    in_word = True
                    current_patch_start = i # Mark start index

        # Handle trailing word patch if sequence doesn't end with a delimiter
        if in_word and current_patch_start < len(text):
            trailing_word_str = text[current_patch_start:]
            try:
                trailing_word_bytes = torch.tensor(list(trailing_word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                if trailing_word_bytes.numel() > 0:
                    entropy = self.entropy_helper.compute_entropy(trailing_word_bytes)
                    patches_with_entropy.append((trailing_word_bytes, entropy))
            except Exception as enc_e:
                logger.warning(f"Error encoding trailing word patch '{trailing_word_str[:20]}...': {enc_e}")

        # Final filter for empty patches (should be rare now)
        patches_with_entropy = [(p, e) for p, e in patches_with_entropy if p.numel() > 0]

        # Optional: Log patch info
        # if patches_with_entropy: logger.debug(f"Created {len(patches_with_entropy)} patches. Example entropies: {[f'{e:.2f}' for _, e in patches_with_entropy[:5]]}")

        return patches_with_entropy # List[Tuple[Tensor, float]]

    @torch.no_grad()
    def reset_context(self):
        """Resets the entropy cache."""
        self.entropy_helper.entropy_cache = {}
        logger.debug("HAKMEMBabylonIndex context (entropy cache) reset.")

# =====================================================================
# HAKMEM-Enhanced Cross Attention Block
# =====================================================================
class HAKMEMCrossAttentionBlock(nn.Module):
    """ Enhanced Cross-Attention block (HAKMEM-inspired). """
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size <= 0: raise ValueError("hidden_size must be positive")
        if num_heads <= 0 : num_heads = max(1, hidden_size // 64)

        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads:
            num_heads = 1
            logger.warning(f"Could not find valid head count for hidden_size {hidden_size}. Using 1 head.")
        elif hidden_size % num_heads != 0:
            num_heads = valid_heads[0]
            logger.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size} in CrossAttention.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
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

        if seq_len_kv == 0: return torch.zeros_like(queries)
        if kv_hidden_size != self.hidden_size:
             raise ValueError(f"Keys/Values hidden size ({kv_hidden_size}) does not match block hidden size ({self.hidden_size})")

        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)

        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nq, d]
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nkv, d]
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nkv, d]

        # Prepare mask for scaled_dot_product_attention (expects True where masked)
        attn_mask_sdpa = None
        if attention_mask is not None:
            # Ensure mask is boolean and broadcastable to [B, h, Nq, Nkv]
            if attention_mask.dim() == 2: # [B, Nkv] -> [B, 1, 1, Nkv]
                attn_mask_sdpa = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            elif attention_mask.dim() == 3: # [B, Nq, Nkv] -> [B, 1, Nq, Nkv]
                 attn_mask_sdpa = attention_mask.unsqueeze(1).to(torch.bool)
            elif attention_mask.dim() == 4: # Assume [B, N_heads, Nq, Nkv] or [B, 1, Nq, Nkv]
                 attn_mask_sdpa = attention_mask.to(torch.bool)
            else: logger.warning(f"Unsupported attention mask shape {attention_mask.shape}. Ignoring mask.")

            if attn_mask_sdpa is not None:
                 attn_mask_sdpa = attn_mask_sdpa.to(device=queries.device)
                 # Check for broadcasting compatibility
                 try:
                     # Attempt a dummy broadcast to check compatibility before passing to SDPA
                     _ = torch.empty(batch_size, self.num_heads, num_queries, seq_len_kv, device=queries.device).masked_fill(attn_mask_sdpa, 0.0)
                 except RuntimeError:
                      logger.warning(f"Mask shape {attn_mask_sdpa.shape} not broadcastable to target [B, h, Nq, Nkv] ({batch_size}, {self.num_heads}, {num_queries}, {seq_len_kv}). Ignoring mask.")
                      attn_mask_sdpa = None


        use_flash = hasattr(F, 'scaled_dot_product_attention')
        output = None

        if use_flash:
             try:
                  # Flash attention mask should be True where attention is *prevented*
                  output = F.scaled_dot_product_attention(
                      q, k, v, attn_mask=attn_mask_sdpa, # Pass boolean mask directly
                      dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
                  )
                  if not torch.isfinite(output).all(): raise ValueError("Flash Attention produced NaN/Inf")
             except Exception as e:
                  # logger.debug(f"Flash attention failed in CrossAttentionBlock: {e}. Falling back to manual.", exc_info=False) # DEBUG
                  use_flash = False; output = None

        if output is None: # Manual calculation fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B, h, Nq, Nkv]
            if attn_mask_sdpa is not None:
                 try:
                     # Manual mask needs float('-inf') where mask is True
                     # Ensure mask is broadcastable before filling
                     mask_broadcast, _ = torch.broadcast_tensors(attn_mask_sdpa, scores)
                     scores = scores.masked_fill(mask_broadcast, float('-inf'))
                 except RuntimeError as e: logger.error(f"Mask fill error. Scores: {scores.shape}, Mask: {attn_mask_sdpa.shape}. Error: {e}")
            # Add clamping before softmax for stability
            scores = torch.clamp(scores, min=-30.0, max=30.0)
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = torch.nan_to_num(attn_probs) # Handle potential NaNs after softmax
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v) # [B, h, Nq, d]

        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        # Final check
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in HAKMEMCrossAttentionBlock output.")
            output = torch.nan_to_num(output)
        return output

# =====================================================================
# HAKMEM-Enhanced Local Encoder (Adapted for Patches V3)
# =====================================================================
class HAKMEMLocalEncoder(nn.Module):
    """
    Enhanced LocalEncoder using Transformer layers on word/punctuation patches,
    with HAKMEM-inspired N-gram features per patch and attention pooling. (V3)
    """
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1,
                 n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Local Encoder hidden_size must be positive.")
        self.hidden_size=hidden_size

        self.byte_embeddings=nn.Embedding(256, hidden_size); nn.init.normal_(self.byte_embeddings.weight, std=1.0/math.sqrt(hidden_size))

        self.n_gram_sizes = sorted(list(set(s for s in n_gram_sizes if isinstance(s, int) and s > 0)))
        self.n_gram_vocab_size = n_gram_vocab_size
        self.n_gram_embeddings = None
        if self.n_gram_sizes:
            if n_gram_vocab_size <= 0:
                logger.warning("n_gram_vocab_size <= 0, disabling N-gram features in LocalEncoder.")
                self.n_gram_sizes = []
            else:
                self.n_gram_embeddings=nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes})
                for emb in self.n_gram_embeddings.values(): nn.init.normal_(emb.weight, std=0.02)
                logger.info(f"HAKMEMLocalEncoder using N-grams: {self.n_gram_sizes} with vocab size {n_gram_vocab_size}")
                # Precompute prime multipliers for hashing
                self.hash_multipliers = { n: torch.tensor([self._get_prime(n * 10 + i + 1) for i in range(n)], dtype=torch.long) for n in self.n_gram_sizes }
        else:
            logger.info("HAKMEMLocalEncoder: N-gram features disabled.")

        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads:
             logger.warning(f"HAKMEMLocalEncoder adjusted Transformer heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        # Use standard TransformerEncoderLayer
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling over byte representations within a patch
        self.patch_pooling_attention=HAKMEMCrossAttentionBlock(hidden_size, num_heads, dropout)
        # Learnable query vector for pooling
        self.patch_query=nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        self.norm=nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout=nn.Dropout(dropout)

    def _get_prime(self, n):
        """ Simple utility to find the next prime number >= n. """
        def is_prime(num):
            if num < 2: return False
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0: return False
            return True
        num = n
        while True:
            if is_prime(num): return num
            num += 1

    def _get_n_gram_hashes(self, patch_byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        """Calculates polynomial rolling hashes for n-grams within a single patch."""
        # patch_byte_sequence: [patch_len]
        patch_len = patch_byte_sequence.size(0)
        device = patch_byte_sequence.device
        if patch_len < n: return torch.empty(0, dtype=torch.long, device=device)

        # Unsqueeze to add batch dim [1, patch_len] for unfold
        windows = patch_byte_sequence.long().unsqueeze(0).unfold(dimension=1, size=n, step=1) # [1, NumWindows, n]
        # Ensure multipliers are on the correct device
        multipliers = self.hash_multipliers.get(n)
        if multipliers is None: multipliers = torch.tensor([31]*n, device=device, dtype=torch.long) # Fallback
        else: multipliers = multipliers.to(device=device, dtype=torch.long)

        # Reshape multipliers for broadcasting: [1, 1, n]
        multipliers = multipliers.view(1, 1, n)
        # Perform hashing: sum(byte * multiplier) for each window
        # windows is [1, NumWindows, n], multipliers is [1, 1, n]
        hashes = (windows.long() * multipliers).sum(dim=-1) # [1, NumWindows]
        # Modulo by vocab size and squeeze the batch dim
        return (hashes % self.n_gram_vocab_size).squeeze(0) # [NumWindows]

    def forward(self, patches_with_entropy: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """
        Processes a list of patch tensors (words/punctuation) with their entropies.
        Returns a tensor of aggregated representations for each patch.
        """
        if not patches_with_entropy:
            # Return an empty tensor with the correct shape if no patches are provided
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            logger.debug("LocalEncoder received no patches.")
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        device = patches_with_entropy[0][0].device # Get device from the first patch
        model_dtype = next(self.parameters()).dtype

        patch_representations = []
        for patch_bytes, patch_entropy in patches_with_entropy:
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue

            # Ensure patch_bytes is long for embedding lookup and hashing
            patch_bytes_long = patch_bytes.long()

            # 1. Get Byte Embeddings
            x = self.byte_embeddings(patch_bytes_long).to(model_dtype) # [patch_len, H]
            # Add batch dimension for Transformer: [1, patch_len, H]
            x = x.unsqueeze(0)

            # 2. Add N-gram Features (if enabled)
            if self.n_gram_embeddings and self.n_gram_sizes:
                n_gram_features = torch.zeros_like(x) # Initialize features [1, patch_len, H]
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        # Get hashes for this specific patch
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_long, n) # [NumWin]
                        if n_gram_hashes.numel() > 0:
                            # Get embeddings for these hashes
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes).to(model_dtype) # [NumWin, H]
                            # Add batch dimension: [1, NumWin, H]
                            ngram_embeds = ngram_embeds.unsqueeze(0)

                            # Add these embeddings back to the corresponding positions in x.
                            # The k-th N-gram embedding corresponds to the byte at index k + n - 1.
                            num_windows = ngram_embeds.size(1)
                            indices = torch.arange(n - 1, n - 1 + num_windows, device=device, dtype=torch.long)
                            index_reshaped = indices.view(1, -1, 1)
                            index_expanded = index_reshaped.expand(1, num_windows, self.hidden_size)

                            # Ensure indices are within bounds (should be by construction)
                            if torch.max(indices) < patch_len:
                                n_gram_features.scatter_add_(1, index_expanded, ngram_embeds)
                            else:
                                logger.warning(f"N-gram index mismatch detected (max index {torch.max(indices)} >= patch_len {patch_len}). PatchLen={patch_len}, n={n}, NumWin={num_windows}")
                                # Filter valid indices and embeds if out of bounds occurs
                                valid_mask = indices < patch_len
                                valid_indices = indices[valid_mask]
                                valid_embeds = ngram_embeds[:, valid_mask, :]
                                if valid_indices.numel() > 0:
                                    index_reshaped_valid = valid_indices.view(1, -1, 1)
                                    index_expanded_valid = index_reshaped_valid.expand(1, valid_indices.size(0), self.hidden_size)
                                    n_gram_features.scatter_add_(1, index_expanded_valid, valid_embeds)
                # Add N-gram features to byte embeddings
                x = x + n_gram_features

            # Check stability before transformer
            if not torch.isfinite(x).all():
                logger.warning(f"NaN/Inf detected in LocalEncoder input before Transformer (patch size {patch_len}). Replacing with zeros.")
                x = torch.nan_to_num(x)

            # 3. Apply Dropout and Transformer Layers
            x = self.dropout(x)
            processed_bytes = self.transformer(x) # Output: [1, patch_len, H]

            # Check stability after transformer
            if not torch.isfinite(processed_bytes).all():
                logger.warning(f"NaN/Inf detected in LocalEncoder output after Transformer (patch size {patch_len}). Replacing with zeros.")
                processed_bytes = torch.nan_to_num(processed_bytes)

            # 4. Pool patch representations using Attention
            batch_query = self.patch_query.expand(1, -1, -1).to(dtype=model_dtype) # [1, 1, H]
            patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes) # [1, 1, H]

            # Check stability after pooling
            if not torch.isfinite(patch_repr).all():
                logger.warning(f"NaN/Inf detected in LocalEncoder after patch pooling (patch size {patch_len}). Replacing with zeros.")
                patch_repr = torch.nan_to_num(patch_repr)

            patch_representations.append(patch_repr) # Append the [1, 1, H] tensor

        if not patch_representations:
             device = next(self.parameters()).device
             dtype = next(self.parameters()).dtype
             logger.debug("No valid patch representations generated after encoding.")
             return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        # Concatenate the representations along the sequence dimension (dim 1)
        patches_combined = torch.cat(patch_representations, dim=1) # [1, num_patches, H]

        # Final normalization
        normed_output = self.norm(patches_combined)
        if not torch.isfinite(normed_output).all():
             logger.warning("NaN/Inf detected in LocalEncoder final output after norm. Replacing with zeros.")
             normed_output = torch.nan_to_num(normed_output)

        return normed_output


# =====================================================================
# Hyperbolic Geometry Utilities
# =====================================================================
class HyperbolicUtils:
    """ Utility functions for Poincaré Ball operations with numerical stability. """
    @staticmethod
    def poincare_clip(x: torch.Tensor, c: float, radius: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        # Ensure curvature is positive for geometric meaning, fallback for c=0
        if c <= 0: return x
        sqrt_c = math.sqrt(max(c, eps))
        max_norm = (radius / sqrt_c) * (1.0 - eps) # Stay slightly inside the boundary
        x_norm_sq = torch.sum(x.pow(2), dim=-1, keepdim=True)
        # Use slightly larger eps inside sqrt for norm calculation stability
        norm = torch.sqrt(torch.clamp(x_norm_sq, min=0) + 1e-7)
        # Condition where clipping is needed (compare norms directly for clarity)
        cond = norm > max_norm
        # Compute scale factor: max_norm / norm if clipping, 1 otherwise
        # Add eps to norm in denominator to avoid division by zero if norm is exactly zero
        scale_factor = torch.where(cond, max_norm / (norm + eps), torch.ones_like(norm))
        clipped_x = x * scale_factor
        return clipped_x

    @staticmethod
    def exponential_map(v: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
        """ Maps a tangent vector v at the origin to the Poincaré ball. """
        if c <= 0: return v # Map is identity if curvature is non-positive
        v_norm_sq = torch.sum(v.pow(2), dim=-1, keepdim=True)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=0) + eps) # Stable norm
        sqrt_c = math.sqrt(max(c, eps))
        tanh_term = torch.tanh(sqrt_c * v_norm)
        # Calculate lambda_v safely, handling norm close to zero
        lambda_v = torch.where(v_norm > eps, tanh_term / (sqrt_c * v_norm), torch.ones_like(v_norm)) # Limit case is 1
        mapped_v = lambda_v * v
        # Clip the result to ensure it stays strictly within the ball
        return HyperbolicUtils.poincare_clip(mapped_v, c)

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
        """ Maps a point y in the Poincaré ball back to the tangent space at the origin. """
        if c <= 0: return y # Map is identity if curvature is non-positive
        # Ensure the point y is strictly inside the ball before mapping
        y_clipped = HyperbolicUtils.poincare_clip(y, c)
        y_norm_sq = torch.sum(y_clipped.pow(2), dim=-1, keepdim=True)
        y_norm = torch.sqrt(torch.clamp(y_norm_sq, min=0) + eps) # Stable norm
        sqrt_c = math.sqrt(max(c, eps))
        # Clamp input to atanh to prevent NaN/Inf, slightly away from +/-1
        arctanh_input = torch.clamp(sqrt_c * y_norm, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        atanh_term = torch.atanh(arctanh_input)
        # Calculate lambda_y safely, handling norm close to zero
        lambda_y = torch.where(y_norm > eps, atanh_term / (sqrt_c * y_norm), torch.ones_like(y_norm)) # Limit case is 1
        mapped_y = lambda_y * y_clipped
        return mapped_y

# =====================================================================
# HAKMEM-Enhanced Complex Number Operations & LayerNorm (V3.7 - Stability Fixes)
# =====================================================================
class HAKMEMComplexOperations:
    """ Static methods for complex number operations using real/imag pairs. (V3.7 Stability) """
    @staticmethod
    def complex_matmul(a_real, a_imag, b_real, b_imag):
        """ (a+ib)(c+id) = (ac-bd) + i(ad+bc) """
        try:
            out_real = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
            out_imag = torch.matmul(a_real, b_imag) + torch.matmul(a_imag, b_real)
            return out_real, out_imag
        except RuntimeError as e:
            logger.error(f"Complex matmul error: {e}. Shapes: a_real={a_real.shape}, a_imag={a_imag.shape}, b_real={b_real.shape}, b_imag={b_imag.shape}", exc_info=False)
            raise e

    @staticmethod
    def complex_phase_shift(real, imag, phase_cos, phase_sin):
        """ Multiplies (real + i*imag) by (phase_cos + i*phase_sin) """
        return real * phase_cos - imag * phase_sin, real * phase_sin + imag * phase_cos

    @staticmethod
    def complex_norm(real, imag, epsilon=1e-6): # Default epsilon consistent
        """ Calculates the magnitude |z| = sqrt(real^2 + imag^2) """
        squared_norm = real.pow(2) + imag.pow(2)
        # Clamp minimum before sqrt, add epsilon inside sqrt for gradient stability near zero
        return torch.sqrt(torch.clamp(squared_norm, min=1e-9) + epsilon)

    @staticmethod
    def complex_normalize(real, imag, epsilon=1e-6): # Default epsilon consistent
        """ Normalizes complex number z / |z| """
        norm = HAKMEMComplexOperations.complex_norm(real, imag, epsilon)
        norm_stable = norm + epsilon # Add epsilon to denominator to avoid division by zero
        return real / norm_stable, imag / norm_stable

    @staticmethod
    def complex_attention_scores(q_real, q_imag, k_real, k_imag, scale=1.0):
        """ Calculates complex attention scores q * k^H (k conjugate transpose).
            Returns real part, imaginary part, and magnitude of the scores.
        """
        # k^H = (k_real - i*k_imag)^T = k_real^T - i*k_imag^T
        k_real_t = k_real.transpose(-2, -1)
        k_imag_t = k_imag.transpose(-2, -1) # Transpose only, negation handles conjugate

        # (q_r + i*q_i) * (k_r_t - i*k_i_t) = (q_r*k_r_t + q_i*k_i_t) + i*(q_i*k_r_t - q_r*k_i_t)
        attn_real, attn_imag = HAKMEMComplexOperations.complex_matmul(
            q_real, q_imag, k_real_t, -k_imag_t # Use negative k_imag_t for conjugate
        )
        # Apply scaling factor (optional, often 1/sqrt(d_k))
        attn_real_scaled = attn_real * scale
        attn_imag_scaled = attn_imag * scale

        # --- V3.7 Stability Check ---
        attn_magnitude = torch.zeros_like(attn_real_scaled) # Default to zero if inputs are bad
        if torch.isfinite(attn_real_scaled).all() and torch.isfinite(attn_imag_scaled).all():
            # Calculate magnitude of the complex scores using the stable norm
            attn_magnitude = HAKMEMComplexOperations.complex_norm(attn_real_scaled, attn_imag_scaled)
        else:
            logger.warning("NaN/Inf detected in attn_real_scaled or attn_imag_scaled BEFORE complex_norm. Setting magnitude to 0.")
            attn_real_scaled = torch.nan_to_num(attn_real_scaled) # Clean up for return if needed
            attn_imag_scaled = torch.nan_to_num(attn_imag_scaled)

        # Return all components as they might be useful
        return attn_real_scaled, attn_imag_scaled, attn_magnitude

class HAKMEMComplexLayerNorm(nn.Module):
    """ Layer Normalization for complex numbers represented as (real, imag) pairs, with optional coupling. """
    def __init__(self, dim, eps=1e-5, coupled=True, coupling_strength_init=0.0):
        super().__init__()
        self.dim = dim; self.eps = eps; self.coupled = coupled
        # Standard LayerNorm for real and imaginary parts separately
        self.real_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.imag_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        if coupled:
             # Parameters for coupling between real and imaginary parts after normalization
             self.coupling_strength = nn.Parameter(torch.tensor(coupling_strength_init))
             self.cross_gain_ri = nn.Parameter(torch.zeros(dim)) # Gain from real to imaginary
             self.cross_gain_ir = nn.Parameter(torch.zeros(dim)) # Gain from imaginary to real
             nn.init.normal_(self.cross_gain_ri, std=0.01); nn.init.normal_(self.cross_gain_ir, std=0.01)
             logger.debug(f"ComplexLayerNorm initialized with coupled=True (dim={dim})")
        else: logger.debug(f"ComplexLayerNorm initialized with coupled=False (dim={dim})")

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x
        # Apply standard LayerNorm to each part
        real_normed_affine = self.real_norm(real)
        imag_normed_affine = self.imag_norm(imag)

        if self.coupled:
            # Apply coupling: real_out = normed_r + f(strength) * gain_ir * normed_i
            # Apply coupling: imag_out = normed_i + f(strength) * gain_ri * normed_r
            # Use tanh for a bounded coupling factor, scaled down
            # Clamp the raw strength parameter for stability before tanh
            clamped_strength = torch.clamp(self.coupling_strength, -5.0, 5.0)
            coupling_factor = torch.tanh(clamped_strength) * 0.1
            real_out = real_normed_affine + coupling_factor * self.cross_gain_ir * imag_normed_affine
            imag_out = imag_normed_affine + coupling_factor * self.cross_gain_ri * real_normed_affine
            return real_out, imag_out
        else:
            # No coupling, return independently normalized parts
            return real_normed_affine, imag_normed_affine

# =====================================================================
# HAKMEM-Enhanced Positional Encoding (Complex RoPE-like)
# =====================================================================
class HAKMEMPositionalEncoding(nn.Module):
    """ Learnable complex positional encoding, inspired by RoPE, acting as a factory for cos/sin embeddings. """
    def __init__(self, dim, max_len=4096, learnable=True, base=10000.0):
        super().__init__()
        if dim % 2 != 0: raise ValueError(f"PE dimension must be even for RoPE pairing, got dim={dim}")
        self.dim = dim; self.learnable = learnable
        self.max_cache_len = max(max_len * 2, 40); self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq_base', inv_freq, persistent=True)
        if learnable:
            # Learnable scaling factors for frequencies, initialized near 1
            self.frequency_scale_factors = nn.Parameter(torch.ones(dim // 2))
            logger.info("Using HAKMEM Learnable Positional Encoding (Frequency Scaling).")
        else:
             logger.info("Using HAKMEM Fixed Positional Encoding.")
             self.frequency_scale_factors = None # Explicitly None if not learnable
        self.position_cache = {}

    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes or retrieves cached cos/sin embeddings. """
        compute_dtype = torch.float32
        cache_key = (seq_len, str(device), str(dtype))
        if cache_key in self.position_cache: return self.position_cache[cache_key]

        base_freqs = self.inv_freq_base.to(device=device, dtype=compute_dtype)
        current_freqs = base_freqs
        if self.learnable and self.frequency_scale_factors is not None:
            # Clamp learnable scale factors for stability before multiplication
            scale_factors = torch.clamp(self.frequency_scale_factors, min=0.1, max=10.0).to(device=device, dtype=compute_dtype)
            current_freqs = base_freqs * scale_factors

        position = torch.arange(seq_len, device=device, dtype=compute_dtype).unsqueeze(1) # [S, 1]
        angles = position * current_freqs # [S, dim/2]
        pe_cos = torch.cos(angles); pe_sin = torch.sin(angles) # [S, dim/2]
        pe_cos_cached = pe_cos.to(dtype=dtype); pe_sin_cached = pe_sin.to(dtype=dtype)

        if len(self.position_cache) >= self.max_cache_len:
            try: self.position_cache.pop(next(iter(self.position_cache)))
            except StopIteration: pass
        self.position_cache[cache_key] = (pe_cos_cached, pe_sin_cached)
        return pe_cos_cached, pe_sin_cached

    def get_cos_sin_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Gets cos/sin embeddings suitable for broadcasting with input tensor x. """
        seq_len = x.size(2); device = x.device; dtype = x.dtype # Assumes [B, h, S, d]
        cos_emb, sin_emb = self._compute_cos_sin(seq_len, device, dtype) # [S, dim/2]
        # Reshape for broadcasting: [S, dim/2] -> [1, 1, S, dim/2]
        return cos_emb.unsqueeze(0).unsqueeze(1), sin_emb.unsqueeze(0).unsqueeze(1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.warning("HAKMEMPositionalEncoding.forward() is deprecated. Use get_cos_sin_embeddings().")
        return x

# =====================================================================
# HAKMEM-Enhanced Entangled Interference Layer (Complex Attention) (V3.9 - Stability Fixes)
# =====================================================================
class HAKMEMEntangledInterferenceLayer(nn.Module):
    """ Complex Attention Layer with RoPE, optional Entanglement, and adaptive dynamics. (V3.9 Stability Fixes) """
    def __init__(self, dim, heads=8, dropout=0.1, interference_type="quantum", use_entanglement=True, noise_scale=0.05, use_rotary=True, adaptive_attention=True, rotary_base=10000.0, max_seq_len=4096):
        super().__init__()
        if dim <= 0: raise ValueError("Dimension must be positive.")
        if heads <= 0: heads = max(1, dim // 64) # Default heads if invalid
        if dim % heads != 0:
             original_heads = heads
             valid_heads = [h for h in range(heads, 0, -1) if dim % h == 0]
             heads = valid_heads[0] if valid_heads else 1
             logger.warning(f"Adjusted heads from {original_heads} to {heads} for complex dim {dim} in InterferenceLayer.")

        self.dim = dim; self.heads = heads; self.head_dim = dim // heads; self.dropout_rate = dropout
        self.interference_type = interference_type # Currently unused parameter
        self.use_entanglement = use_entanglement
        self.noise_scale = noise_scale # Currently unused parameter
        self.use_rotary = use_rotary; self.adaptive_attention = adaptive_attention

        if self.use_rotary and self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, but got head_dim={self.head_dim} (dim={dim}, heads={heads})")

        if self.use_rotary:
             self.positional_encoding = HAKMEMPositionalEncoding(self.head_dim, max_len=max_seq_len, learnable=True, base=rotary_base)
             self.rotary_dim = self.head_dim
        else: self.positional_encoding = None

        if use_entanglement:
            # Initialize entanglement matrix near identity
            entangle_init = torch.eye(heads) * 0.9 + torch.randn(heads, heads) * 0.01
            entangle_init.fill_diagonal_(0.9)
            self.entanglement_matrix = nn.Parameter(entangle_init)
            logger.info("EntangledInterferenceLayer: Using learnable entanglement matrix.")
        else:
            self.register_buffer('entanglement_matrix', torch.eye(heads), persistent=False)
            logger.info("EntangledInterferenceLayer: Entanglement disabled (using identity matrix).")

        # Linear projections for Q, K, V (real and imaginary separately)
        self.q_real = nn.Linear(dim, dim, bias=False); self.k_real = nn.Linear(dim, dim, bias=False); self.v_real = nn.Linear(dim, dim, bias=False)
        self.q_imag = nn.Linear(dim, dim, bias=False); self.k_imag = nn.Linear(dim, dim, bias=False); self.v_imag = nn.Linear(dim, dim, bias=False)
        self.out_real = nn.Linear(dim, dim, bias=False); self.out_imag = nn.Linear(dim, dim, bias=False)
        for layer in [self.q_real, self.k_real, self.v_real, self.q_imag, self.k_imag, self.v_imag, self.out_real, self.out_imag]:
             nn.init.xavier_uniform_(layer.weight)

        # Learnable parameter for attention temperature (initialized to 0 for exp(0)=1 initially)
        # interference_strength parameter removed from V3.7
        if adaptive_attention: self.attention_temperature = nn.Parameter(torch.tensor(0.0))
        else: self.register_buffer('attention_temperature', torch.tensor(0.0), persistent=False) # Fixed temperature (log(1)=0)

        self.attn_dropout = nn.Dropout(dropout); self.resid_dropout = nn.Dropout(dropout)
        self.complex_ops = HAKMEMComplexOperations()
        logger.info(f"EntangledInterferenceLayer: RoPE {'Enabled' if use_rotary else 'Disabled'}, Adaptive Temp: {adaptive_attention}")

    def _apply_rotary_pos_emb(self, x_real: torch.Tensor, x_imag: torch.Tensor, cos_emb: torch.Tensor, sin_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies RoPE rotation to complex input (x_real, x_imag)."""
        if not self.use_rotary or self.positional_encoding is None: return x_real, x_imag
        d = x_real.shape[-1]
        if d % 2 != 0: raise ValueError(f"RoPE requires an even head dimension, got {d}")
        if cos_emb.shape[-1] * 2 != d: raise ValueError(f"RoPE dim mismatch: head={d}, cos/sin={cos_emb.shape[-1]*2}")

        d_rope = cos_emb.shape[-1] # This is d/2
        x_r_paired = x_real.reshape(*x_real.shape[:-1], d_rope, 2)
        x_i_paired = x_imag.reshape(*x_imag.shape[:-1], d_rope, 2)
        x_r1, x_r2 = x_r_paired[..., 0], x_r_paired[..., 1] # [B, h, S, d/2]
        x_i1, x_i2 = x_i_paired[..., 0], x_i_paired[..., 1] # [B, h, S, d/2]

        rot_r1 = x_r1 * cos_emb - x_r2 * sin_emb; rot_r2 = x_r1 * sin_emb + x_r2 * cos_emb
        rotated_real = torch.stack([rot_r1, rot_r2], dim=-1).flatten(start_dim=-2)
        rot_i1 = x_i1 * cos_emb - x_i2 * sin_emb; rot_i2 = x_i1 * sin_emb + x_i2 * cos_emb
        rotated_imag = torch.stack([rot_i1, rot_i2], dim=-1).flatten(start_dim=-2)

        # --- V3.7 Stability Check ---
        if not (torch.isfinite(rotated_real).all() and torch.isfinite(rotated_imag).all()):
            logger.warning("NaN/Inf detected within RoPE application. Replacing.")
            rotated_real = torch.nan_to_num(rotated_real)
            rotated_imag = torch.nan_to_num(rotated_imag)
        # --- End V3.7 ---

        return rotated_real, rotated_imag

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = (real, imag), where real/imag are [B, S, D]
        real, imag = x; batch_size, seq_len, _ = real.shape; device = real.device; dtype=real.dtype

        # 1. Project Q, K, V
        q_r_proj = self.q_real(real); q_i_proj = self.q_imag(imag)
        k_r_proj = self.k_real(real); k_i_proj = self.k_imag(imag)
        v_r_proj = self.v_real(real); v_i_proj = self.v_imag(imag)

        # 2. Reshape for multi-head attention: [B, S, D] -> [B, h, S, d]
        q_r = q_r_proj.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_r = k_r_proj.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_r = v_r_proj.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        q_i = q_i_proj.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_i = k_i_proj.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_i = v_i_proj.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # --- Check for NaN/Inf in inputs to attention ---
        input_finite = (torch.isfinite(q_r).all() and torch.isfinite(q_i).all() and
                        torch.isfinite(k_r).all() and torch.isfinite(k_i).all() and
                        torch.isfinite(v_r).all() and torch.isfinite(v_i).all())
        if not input_finite:
            logger.warning(f"NaN/Inf detected in Q/K/V inputs to EntangledInterferenceLayer BEFORE RoPE/Entanglement. Replacing with zeros.")
            # Zero out inputs to prevent propagation
            q_r, q_i = torch.zeros_like(q_r), torch.zeros_like(q_i)
            k_r, k_i = torch.zeros_like(k_r), torch.zeros_like(k_i)
            v_r, v_i = torch.zeros_like(v_r), torch.zeros_like(v_i)

        # 3. Apply Rotary Positional Encoding (if enabled)
        if self.use_rotary and self.positional_encoding is not None:
            cos_emb, sin_emb = self.positional_encoding.get_cos_sin_embeddings(q_r)
            q_r, q_i = self._apply_rotary_pos_emb(q_r, q_i, cos_emb, sin_emb)
            # Check stability after RoPE (already done inside _apply_rotary_pos_emb in V3.7)


        # 4. Apply Head Entanglement (if enabled)
        if self.use_entanglement and hasattr(self, 'entanglement_matrix') and self.entanglement_matrix is not None:
            ent_matrix = self.entanglement_matrix.to(device=device, dtype=dtype) # [h, h]
            try:
                # Mix heads using einsum: 'b h s d, h k -> b k s d' -> need transpose: [B, h, S, d] -> [B, S, h, d]
                q_r_t = q_r.transpose(1, 2); q_i_t = q_i.transpose(1, 2)
                k_r_t = k_r.transpose(1, 2); k_i_t = k_i.transpose(1, 2)
                q_r = torch.einsum('bshd,hk->bskd', q_r_t, ent_matrix).transpose(1, 2).contiguous()
                q_i = torch.einsum('bshd,hk->bskd', q_i_t, ent_matrix).transpose(1, 2).contiguous()
                k_r = torch.einsum('bshd,hk->bskd', k_r_t, ent_matrix).transpose(1, 2).contiguous()
                k_i = torch.einsum('bshd,hk->bskd', k_i_t, ent_matrix).transpose(1, 2).contiguous()
                # Check stability after Entanglement
                if not (torch.isfinite(q_r).all() and torch.isfinite(q_i).all() and \
                        torch.isfinite(k_r).all() and torch.isfinite(k_i).all()):
                    logger.warning("NaN/Inf detected after Entanglement application. Replacing.")
                    q_r, q_i = torch.nan_to_num(q_r), torch.nan_to_num(q_i)
                    k_r, k_i = torch.nan_to_num(k_r), torch.nan_to_num(k_i)
            except Exception as e:
                logger.error(f"Error during entanglement: {e}. Shapes: q={q_r.shape}, ent={ent_matrix.shape}")
                # If entanglement fails, just proceed without it
                pass

        # 5. Calculate Complex Attention Scores (Magnitude used for Softmax)
        scale = 1.0 / math.sqrt(max(1, self.head_dim)) # Standard attention scaling
        # Pass scale=1.0, apply scaling later to magnitude only
        _, _, attn_mag = self.complex_ops.complex_attention_scores(q_r, q_i, k_r, k_i, scale=1.0) # V3.7: attn_mag is checked inside
        attn_mag_scaled = attn_mag * scale # Apply scaling to magnitude [B, h, S, S]

        # --- STABILITY FIX: Check and Clamp attn_mag_scaled ---
        if not torch.isfinite(attn_mag_scaled).all():
            num_nan = torch.isnan(attn_mag_scaled).sum().item()
            num_inf = torch.isinf(attn_mag_scaled).sum().item()
            max_val = attn_mag_scaled[torch.isfinite(attn_mag_scaled)].max().item() if torch.isfinite(attn_mag_scaled).any() else 0.0
            logger.warning(f"NaN/Inf detected in attn_mag_scaled (shape {attn_mag_scaled.shape}, NaN:{num_nan}, Inf:{num_inf}, max finite: {max_val:.2f}) BEFORE clamp and softmax.")
            # Replace NaN/Inf with 0 or a reasonable value (e.g., -10) - 0 might be safest for magnitudes
            attn_mag_scaled = torch.nan_to_num(attn_mag_scaled, nan=0.0, posinf=10.0, neginf=-10.0) # Use tighter clamp target for inf

        # V3.9: Tighter clamp range
        attn_mag_scaled_clamped = torch.clamp(attn_mag_scaled, min=-10.0, max=10.0) # Was [-15.0, 15.0] in V3.8
        # --- END STABILITY FIX ---

        # 6. Apply Attention Mask (Causal + Padding)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1) # [1, 1, S, S]
        final_mask = causal_mask
        if attention_mask is not None:
             padding_mask = None
             # Ensure padding mask is boolean and broadcastable from [B, S_kv] or [B, S_q, S_kv] etc.
             if attention_mask.dim() == 2: padding_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
             elif attention_mask.dim() == 3: padding_mask = attention_mask.unsqueeze(1) # [B, 1, S, S]
             elif attention_mask.dim() == 4: padding_mask = attention_mask # Assume [B, h, S, S] or [B, 1, S, S]
             else: logger.warning(f"Interference layer unexpected mask shape {attention_mask.shape}. Ignoring padding mask.")

             if padding_mask is not None:
                 padding_mask = padding_mask.to(device=final_mask.device, dtype=torch.bool)
                 try:
                     # Ensure correct broadcasting from [B, 1, 1, S] to [B, h, S, S] or [1, 1, S, S]
                     # The boolean mask should be broadcast to the score shape [B, h, S, S]
                     # If padding_mask is [B,1,1,S], it needs to expand for heads and query sequence length
                     # The final_mask combines causal [1,1,S,S] and padding [B,1,1,S] (or similar)
                     # OR operation handles broadcasting correctly if dimensions match or are 1
                     final_mask = final_mask | padding_mask # Combine masks: mask if EITHER causal OR padding is True
                 except RuntimeError as e:
                     logger.error(f"Mask broadcast error: causal={causal_mask.shape}, padding={padding_mask.shape}. Error: {e}")
                     # Fallback to only causal mask if broadcast fails
                     final_mask = causal_mask

        # Apply final mask to attention magnitude scores (use the clamped version)
        attn_scores_masked = attn_mag_scaled_clamped # Start with clamped values
        # Check mask shape compatibility before applying
        mask_target_shape = attn_scores_masked.shape # [B, h, S, S]
        try:
            # masked_fill will try to broadcast the mask
            mask_broadcast, _ = torch.broadcast_tensors(final_mask, attn_scores_masked)
            attn_scores_masked = attn_scores_masked.masked_fill(mask_broadcast, -torch.finfo(attn_scores_masked.dtype).max)
        except RuntimeError as e:
            logger.error(f"Error applying mask. Scores: {mask_target_shape}, Mask: {final_mask.shape}. Error: {e}")


        # 7. Apply Temperature, then Softmax (V3.9: Tighter temp clamping)
        # V3.8: Tighter clamp range for raw parameter
        clamped_temp_param = torch.clamp(self.attention_temperature, -3.0, 3.0) # Was [-5.0, 5.0]
        # V3.9: Tighter clamp range for the temperature itself
        temp = torch.exp(clamped_temp_param).clamp(min=0.2, max=5.0) # Was min=0.5, max=10.0 in V3.8

        # V3.9: Add logging before division with more detail
        if not torch.isfinite(attn_scores_masked).all() or not torch.isfinite(temp):
            scores_max_finite = attn_scores_masked[torch.isfinite(attn_scores_masked)].max().item() if torch.isfinite(attn_scores_masked).any() else 'None'
            scores_min_finite = attn_scores_masked[torch.isfinite(attn_scores_masked)].min().item() if torch.isfinite(attn_scores_masked).any() else 'None'
            logger.warning(f"NaN/Inf detected BEFORE division by temp. Temp={temp.item() if torch.isfinite(temp) else 'NaN/Inf'}. Scores Max/Min Finite={scores_max_finite}/{scores_min_finite}")
        # else: # Optional: Log normal operation range (can be noisy)
        #    logger.debug(f"Temp value: {temp.item():.3f}. Max score before div: {attn_scores_masked.max().item():.2f}, Min score before div: {attn_scores_masked.min().item():.2f}")


        # V3.7 Change: Apply temperature directly, removed interference_strength
        softmax_input = attn_scores_masked / temp

        # --- STABILITY FIX: Check softmax input ---
        if not torch.isfinite(softmax_input).all():
            num_nan = torch.isnan(softmax_input).sum().item()
            num_inf = torch.isinf(softmax_input).sum().item()
            max_val_finite = softmax_input[torch.isfinite(softmax_input)].max().item() if torch.isfinite(softmax_input).any() else 'None'
            min_val_finite = softmax_input[torch.isfinite(softmax_input)].min().item() if torch.isfinite(softmax_input).any() else 'None'
            # V3.9: Log more details
            logger.warning(f"NaN/Inf detected in softmax_input (shape {softmax_input.shape}, NaN:{num_nan}, Inf:{num_inf}, max/min finite: {max_val_finite}/{min_val_finite}) right BEFORE softmax. Temp was {temp.item():.3f}.")
            # V3.9: Log raw scores that caused issue
            max_raw_score = attn_scores_masked.max().item() if torch.isfinite(attn_scores_masked).any() else 'None'
            logger.warning(f"Max raw score before division: {max_raw_score}")

            softmax_input = torch.nan_to_num(softmax_input, nan=-torch.finfo(softmax_input.dtype).max,
                                            posinf=10.0, # Clamp positive inf to a reasonable max (matching score clamp)
                                            neginf=-torch.finfo(softmax_input.dtype).max)

        # V3.9: Force FP32 before softmax for stability
        attn_weights = F.softmax(softmax_input.float(), dim=-1)
        attn_weights = torch.nan_to_num(attn_weights) # Replace potential remaining NaNs with 0
        attn_weights = self.attn_dropout(attn_weights) # Apply dropout [B, h, S, S]
        # --- END STABILITY FIX ---


        # 8. Apply Attention Weights to Value Vectors (Complex)
        # Use complex_matmul helper with zero imaginary part for weights
        try:
             # Ensure weights are compatible dtype with V
             attn_weights_compat = attn_weights.to(v_r.dtype)
             out_r, out_i = self.complex_ops.complex_matmul(attn_weights_compat, torch.zeros_like(attn_weights_compat), v_r, v_i)
        except Exception as mm_err:
             logger.error(f"Error during complex matmul of attn_weights and values: {mm_err}")
             out_r, out_i = torch.zeros_like(v_r), torch.zeros_like(v_i) # Fallback to zeros

        # 9. Reshape Output and Project
        out_r = out_r.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out_i = out_i.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out_r = self.resid_dropout(self.out_real(out_r))
        out_i = self.resid_dropout(self.out_imag(out_i))

        # --- Final Check ---
        if not (torch.isfinite(out_r).all() and torch.isfinite(out_i).all()):
            logger.warning("NaN/Inf detected in the final output of EntangledInterferenceLayer. Replacing with zeros.")
            out_r = torch.nan_to_num(out_r)
            out_i = torch.nan_to_num(out_i)

        return (out_r, out_i)


# =====================================================================
# HAKMEM-Enhanced Complex-to-Real Projection
# =====================================================================
class HAKMEMComplexToRealProjection(nn.Module):
    """ Projects complex representation (real, imag) back to a real space using various methods. """
    def __init__(self, complex_dim: int, output_dim: int, method: str = "hakmem_enhanced", activation_fn = nn.GELU):
        super().__init__()
        self.method = method; self.complex_dim = complex_dim; self.output_dim = output_dim
        self.complex_ops = HAKMEMComplexOperations()
        self.activation = activation_fn() if activation_fn is not None else nn.Identity()

        if method == "hakmem_enhanced":
             if output_dim <= 0:
                 logger.warning(f"Output dim {output_dim} <= 0 for hakmem_enhanced projection. No projection layers created.")
                 mag_out_dim, phase_out_dim = 0, 0
                 self.magnitude_proj, self.phase_proj, self.combined_proj = None, None, None
             else:
                 if output_dim % 2 != 0: logger.warning(f"Output dim {output_dim} is odd for hakmem_enhanced projection. Splitting as floor/ceil.")
                 mag_out_dim = max(0, output_dim // 2); phase_out_dim = max(0, output_dim - mag_out_dim)
                 self.magnitude_proj = nn.Linear(complex_dim, mag_out_dim, bias=True) if mag_out_dim > 0 else None
                 self.phase_proj = nn.Linear(complex_dim * 2, phase_out_dim, bias=True) if phase_out_dim > 0 else None
                 self.combined_proj = nn.Linear(output_dim, output_dim, bias=True)
                 for layer in [self.magnitude_proj, self.phase_proj, self.combined_proj]:
                    if layer is not None: nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        elif method == "concat":
            self.proj = nn.Linear(complex_dim * 2, output_dim, bias=True)
            nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        elif method == "magnitude":
            self.proj = nn.Linear(complex_dim, output_dim, bias=True)
            nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        else: raise ValueError(f"Unknown ComplexToRealProjection method: {method}")
        logger.debug(f"ComplexToRealProjection initialized with method: {method}, output_dim={output_dim}")

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        real, imag = x
        if not (torch.isfinite(real).all() and torch.isfinite(imag).all()):
            logger.warning("NaN/Inf detected in input to ComplexToRealProjection. Replacing.")
            real = torch.nan_to_num(real); imag = torch.nan_to_num(imag)

        if self.method == "hakmem_enhanced":
            features_list = []
            # Calculate magnitude and project if layer exists
            if self.magnitude_proj is not None:
                magnitude = self.complex_ops.complex_norm(real, imag)
                if not torch.isfinite(magnitude).all():
                     logger.warning("NaN/Inf in magnitude calculation. Replacing."); magnitude = torch.nan_to_num(magnitude)
                mag_features = self.activation(self.magnitude_proj(magnitude))
                features_list.append(mag_features)
            # Calculate normalized components for phase and project if layer exists
            if self.phase_proj is not None:
                norm_real, norm_imag = self.complex_ops.complex_normalize(real, imag)
                if not (torch.isfinite(norm_real).all() and torch.isfinite(norm_imag).all()):
                     logger.warning("NaN/Inf in norm_real/imag calculation. Replacing.")
                     norm_real = torch.nan_to_num(norm_real); norm_imag = torch.nan_to_num(norm_imag)
                phase_input = torch.cat([norm_real, norm_imag], dim=-1)
                phase_features = self.activation(self.phase_proj(phase_input))
                features_list.append(phase_features)

            if not features_list:
                 return torch.empty(*real.shape[:-1], 0, device=real.device, dtype=real.dtype)

            combined_features = torch.cat(features_list, dim=-1)
            if not torch.isfinite(combined_features).all():
                logger.warning("NaN/Inf in combined features before final proj. Replacing.")
                combined_features = torch.nan_to_num(combined_features)

            if self.combined_proj is not None:
                 output = self.activation(self.combined_proj(combined_features))
            else: output = combined_features
            if not torch.isfinite(output).all():
                logger.warning("NaN/Inf detected in HAKMEMComplexToRealProjection (hakmem_enhanced) output. Replacing.")
                output = torch.nan_to_num(output)
            return output
        elif self.method == "concat":
            combined = torch.cat([real, imag], dim=-1)
            if not torch.isfinite(combined).all(): combined = torch.nan_to_num(combined)
            output = self.activation(self.proj(combined))
            if not torch.isfinite(output).all():
                logger.warning("NaN/Inf detected in HAKMEMComplexToRealProjection (concat) output. Replacing.")
                output = torch.nan_to_num(output)
            return output
        elif self.method == "magnitude":
            magnitude = self.complex_ops.complex_norm(real, imag)
            if not torch.isfinite(magnitude).all():
                logger.warning("NaN/Inf in magnitude calculation. Replacing."); magnitude = torch.nan_to_num(magnitude)
            output = self.activation(self.proj(magnitude))
            if not torch.isfinite(output).all():
                logger.warning("NaN/Inf detected in HAKMEMComplexToRealProjection (magnitude) output. Replacing.")
                output = torch.nan_to_num(output)
            return output
        else: raise ValueError(f"Invalid method {self.method} in forward pass.")


# =====================================================================
# HAKMEM-Enhanced Local Decoder (Transformer Decoder)
# =====================================================================
class HAKMEMLocalDecoder(nn.Module):
    """ Transformer-based decoder attending to global memory for byte prediction. """
    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, use_hierarchical_pred: bool = True, max_decode_len: int = 2048):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Decoder hidden size must be positive.")
        self.hidden_size = hidden_size
        self.use_hierarchical = use_hierarchical_pred
        self.max_decode_len = max_decode_len # Max length for positional embeddings

        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads:
             logger.warning(f"HAKMEMLocalDecoder adjusted heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))
        self.positional_encoding = nn.Embedding(max_decode_len, hidden_size)
        nn.init.normal_(self.positional_encoding.weight, std=0.02)

        # Projection layer for the memory
        self.memory_projection = nn.Sequential(
            nn.Linear(global_hidden_size, hidden_size * 2, bias=True), nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=1e-6)
        )
        for layer in self.memory_projection:
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

        # Standard Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-6) # Final norm after decoder stack
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=self.decoder_norm)

        # Prediction Head
        if self.use_hierarchical:
            self.byte_class_pred = nn.Linear(hidden_size, 16)
            self.byte_specific_pred = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(16)])
            nn.init.normal_(self.byte_class_pred.weight, std=0.02); nn.init.zeros_(self.byte_class_pred.bias)
            for layer in self.byte_specific_pred: nn.init.normal_(layer.weight, std=0.02 / math.sqrt(16)); nn.init.zeros_(layer.bias)
            logger.info("HAKMEMLocalDecoder using Hierarchical Prediction Head.")
        else:
            self.byte_pred = nn.Linear(hidden_size, 256)
            nn.init.normal_(self.byte_pred.weight, std=0.02); nn.init.zeros_(self.byte_pred.bias)
            logger.info("HAKMEMLocalDecoder using Flat Prediction Head.")
        self.dropout_embed = nn.Dropout(dropout) # Dropout after embeddings + PE

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
         """Generates a square causal mask for Transformer decoder (True means MASKED)."""
         mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
         return mask

    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt_byte_seq: Target sequence for decoding. [B, T]
            memory: Encoded representation from upstream. [B, M, global_hidden_size]
            tgt_mask: Causal mask for target sequence. [T, T] (optional, generated if None)
            memory_key_padding_mask: Mask indicating padding in memory. [B, M] (True means PAD) (optional)
        Returns:
            Logits for next byte prediction. [B, T, 256] (float32)
        """
        batch_size, tgt_len = tgt_byte_seq.size(); device = tgt_byte_seq.device
        mem_batch_size, mem_len, mem_dim_in = memory.size()
        model_dtype = next(self.parameters()).dtype

        if tgt_len == 0: return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)

        # 1. Project Memory
        if mem_len == 0:
             logger.debug("HAKMEMLocalDecoder received empty memory.")
             projected_memory = torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=model_dtype)
             # If memory is empty, padding mask should also be empty or None
             memory_key_padding_mask = None # Or torch.ones(batch_size, 0, dtype=torch.bool, device=device)
        else:
            if not torch.isfinite(memory).all():
                logger.warning("NaN/Inf detected in memory input to LocalDecoder. Replacing.")
                memory = torch.nan_to_num(memory)
            projected_memory = self.memory_projection(memory.to(model_dtype)) # [B, M, H_local]
            if not torch.isfinite(projected_memory).all():
                 logger.warning("NaN/Inf detected after memory projection in LocalDecoder. Replacing.")
                 projected_memory = torch.nan_to_num(projected_memory)

        # 2. Prepare Target Sequence Embeddings + Positional Encoding
        tgt_embed = self.byte_embeddings(tgt_byte_seq.long()).to(model_dtype) # [B, T, H_local]
        positions = torch.arange(0, tgt_len, device=device).unsqueeze(0) # [1, T]
        positions = torch.clamp(positions, max=self.positional_encoding.num_embeddings - 1)
        pos_embed = self.positional_encoding(positions).to(model_dtype) # [1, T, H_local] -> broadcasts
        tgt_prepared = self.dropout_embed(tgt_embed + pos_embed) # [B, T, H_local]

        if not torch.isfinite(tgt_prepared).all():
            logger.warning("NaN/Inf detected in prepared target sequence input to Decoder Transformer. Replacing.")
            tgt_prepared = torch.nan_to_num(tgt_prepared)

        # 3. Generate Causal Mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device) # [T, T], True = masked

        # 4. Pass through Transformer Decoder
        # Ensure masks have correct dtype and device
        if tgt_mask is not None: tgt_mask = tgt_mask.to(device=device, dtype=torch.bool)
        if memory_key_padding_mask is not None: memory_key_padding_mask = memory_key_padding_mask.to(device=device, dtype=torch.bool)

        output = self.transformer(
            tgt=tgt_prepared,                   # [B, T, H_local]
            memory=projected_memory,            # [B, M, H_local]
            tgt_mask=tgt_mask,                  # [T, T] (Causal mask)
            memory_mask=None,                   # Not typically used in standard decoder
            tgt_key_padding_mask=None,          # Assuming target sequence is not padded here
            memory_key_padding_mask=memory_key_padding_mask # [B, M] (Padding mask for memory)
        ) # Output: [B, T, H_local]

        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in output of Decoder Transformer. Replacing.")
            output = torch.nan_to_num(output)

        # 5. Prediction Head
        if self.use_hierarchical:
            byte_class_logits = self.byte_class_pred(output) # [B, T, 16]
            log_class_probs = F.log_softmax(byte_class_logits, dim=-1) # [B, T, 16]

            log_specific_probs_list = []
            for i in range(16):
                specific_logits = self.byte_specific_pred[i](output) # [B, T, 16]
                log_specific_probs_list.append(F.log_softmax(specific_logits, dim=-1)) # [B, T, 16]

            log_specific_probs_stacked = torch.stack(log_specific_probs_list, dim=2) # [B, T, 16, 16]
            # Add log probabilities: log P(byte) = log P(high) + log P(low | high)
            combined_log_probs = log_class_probs.unsqueeze(-1) + log_specific_probs_stacked # [B, T, 16, 16]
            byte_logits = combined_log_probs.view(batch_size, tgt_len, 256) # [B, T, 256]
        else:
            byte_logits = self.byte_pred(output) # [B, T, 256]

        # Ensure final logits are float32 and finite
        byte_logits = byte_logits.float()
        if not torch.isfinite(byte_logits).all():
            logger.warning("NaN/Inf detected in final decoder logits. Replacing with zeros.")
            byte_logits = torch.nan_to_num(byte_logits, nan=0.0, posinf=0.0, neginf=0.0)

        return byte_logits


# =====================================================================
# HAKMEM-Enhanced Q-Learning Controller & Optimizer (V3.7 - Stability Fixes)
# =====================================================================
class HAKMEMQController:
    """ Q-Learning controller to dynamically adjust optimizer hyperparameters. (V3.7 Stability) """
    def __init__(self, learning_rate: float=0.01, discount: float=0.95, epsilon: float=0.2, epsilon_decay: float=0.9998, min_epsilon: float=0.01, lr_scale_options: Optional[List[float]]=None, momentum_scale_options: Optional[List[float]]=None, max_q_table_size: int=10000):
        self.q_table = {}; self.alpha = learning_rate; self.gamma = discount
        self.epsilon = epsilon; self.min_epsilon = min_epsilon; self.epsilon_decay = epsilon_decay
        self.prev_loss = None; self.prev_state = None; self.prev_action = None
        # V3.7: Tighter default ranges
        if lr_scale_options is None: lr_scale_options = [0.9, 0.95, 1.0, 1.05, 1.1]
        if momentum_scale_options is None: momentum_scale_options = [0.95, 0.98, 1.0, 1.01, 1.02]
        self.action_ranges = {'lr_scale': np.array(lr_scale_options), 'momentum_scale': np.array(momentum_scale_options)}
        self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}
        self.loss_window = deque(maxlen=10); self.grad_norm_window = deque(maxlen=10)
        self.lr_window = deque(maxlen=5); self.momentum_window = deque(maxlen=5)
        self.performance_window = deque(maxlen=20); self.stable_steps = 0; self.oscillation_counter = 0
        self.prev_actions_log = deque(maxlen=5)
        self.max_q_table_size = max_q_table_size; self.q_table_access_count = {}
        self.q_table_creation_time = {}
        self.flow_coefficient = 0.05; self.oscillation_penalty = 0.15; self.stability_reward_bonus = 0.05
        logger.info(f"QController initialized: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}, decay={self.epsilon_decay}")
        logger.info(f"QController action ranges: LR={self.action_ranges['lr_scale']}, Momentum={self.action_ranges['momentum_scale']}")

    def get_state(self, lr, momentum, grad_norm, loss):
        # Use default state if inputs are invalid
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm):
            logger.debug(f"QController: Invalid loss ({loss}) or grad_norm ({grad_norm}) for state calculation. Returning default state.")
            return (2, 2, 0, 2, 1) # Return a neutral default state tuple
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm)
        self.lr_window.append(lr); self.momentum_window.append(momentum)
        # Require sufficient history for meaningful state calculation
        if len(self.loss_window) < 3 or len(self.grad_norm_window) < 3:
            logger.debug("QController: Not enough history for state calculation. Returning default state.")
            return (2, 2, 0, 2, 1) # Return a neutral default state tuple

        loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0
        try:
            # Loss Trend (robust calculation)
            y = np.array(list(self.loss_window)[-5:]); x = np.arange(len(y))
            if len(y) >= 2 and len(np.unique(y)) >= 2:
                coeffs = np.polyfit(x, y, 1); slope = coeffs[0]
            else: slope = 0.0
            avg_loss = np.mean(y); normalized_slope = slope / (abs(avg_loss) + 1e-6)
            loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05])

            # Grad Norm Level (using median for robustness to outliers)
            avg_grad_norm = np.median(list(self.grad_norm_window))
            grad_norm_level_bin = np.digitize(avg_grad_norm, bins=[0.1, 0.5, 1.5, 5.0])

            # LR and Momentum Levels (using current values)
            lr_level_bin = np.digitize(lr / 1e-4, bins=[0.5, 2.0, 10.0, 50.0])
            momentum_level_bin = np.digitize(momentum, bins=[0.85, 0.92, 0.97])

            # Oscillation Detection
            if len(self.performance_window) >= 2:
                 if (self.performance_window[-1] > 1e-3 and self.performance_window[-2] < -1e-3) or \
                    (self.performance_window[-1] < -1e-3 and self.performance_window[-2] > 1e-3):
                      self.oscillation_counter = min(self.oscillation_counter + 1, 5) # Cap counter
                 else: self.oscillation_counter = max(0, self.oscillation_counter - 1)
            oscillation_bin = 1 if self.oscillation_counter >= 3 else 0

        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
            logger.warning(f"Q-state calculation numerical error: {e}. Using default bins.")
            loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0

        state = (loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin)
        self.q_table_access_count[state] = self.q_table_access_count.get(state, 0) + 1
        return state

    def compute_reward(self, current_loss, prev_loss, grad_norm):
        if current_loss is None or prev_loss is None or grad_norm is None or \
           not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm):
            logger.debug("QController: Invalid inputs for reward. Returning 0."); return 0.0
        # Primary reward: scaled relative loss change
        loss_change = prev_loss - current_loss
        loss_change_ratio = loss_change / (abs(prev_loss) + 1e-6)
        reward = np.tanh(loss_change_ratio * 10.0) # Scale and squash

        # Penalties/Bonuses
        if grad_norm > 5.0: reward -= 0.1 * min(1.0, max(0.0, (grad_norm - 5.0) / 10.0))
        elif grad_norm < 0.05: reward += 0.02
        if self.oscillation_counter >= 3: reward -= self.oscillation_penalty

        self.performance_window.append(reward)
        if reward > 0.0:
             self.stable_steps += 1
             reward += min(0.1, self.stability_reward_bonus * (self.stable_steps // 5))
        else: self.stable_steps = 0

        return float(np.clip(reward, -1.0, 1.0)) # Clip final reward

    def choose_action(self, state):
        if state is None:
            logger.debug("QController: State is None, returning default actions.")
            return {'lr_scale': 1.0, 'momentum_scale': 1.0}
        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[state] = time.time(); self.q_table_access_count[state] = 1
            logger.debug(f"QController: Initialized Q for new state {state}. Table size: {len(self.q_table)}")
            self._manage_q_table_size()

        # Epsilon-greedy action selection
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        chosen_actions = {}
        for param, q_values in self.q_table[state].items():
            action_space = self.action_ranges[param]
            if random.random() < self.epsilon: # Explore
                chosen_idx = random.randrange(len(action_space))
            else: # Exploit
                # Handle non-finite Q-values robustly
                q_values_finite = q_values[np.isfinite(q_values)]
                if len(q_values_finite) == 0: # All non-finite or empty
                    chosen_idx = random.randrange(len(action_space)) # Explore if no valid Q
                else:
                    max_q = np.max(q_values_finite)
                    # Find all indices matching the max finite Q-value (handle ties)
                    best_indices = np.where(np.isclose(q_values, max_q) & np.isfinite(q_values))[0]
                    if len(best_indices) > 0: chosen_idx = np.random.choice(best_indices)
                    else: chosen_idx = random.randrange(len(action_space)) # Should not happen if max_q was finite
            chosen_actions[param] = float(action_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy())
        return chosen_actions

    def update(self, state, action, reward, next_state):
        if state is None or next_state is None or action is None:
            logger.debug("QController: Skipping Q-update due to None state/action/next_state."); return
        if state not in self.q_table: logger.warning(f"QController: State {state} not in Q-table during update. Skipping."); return
        # Ensure next state exists in Q-table
        if next_state not in self.q_table:
             self.q_table[next_state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
             self.q_table_creation_time[next_state] = time.time(); self.q_table_access_count[next_state] = 0
             logger.debug(f"QController: Initialized Q for next_state {next_state} during update.")
             self._manage_q_table_size()

        # Update Q-value for each hyperparameter
        for param, chosen_value in action.items():
             action_space = self.action_ranges[param]
             action_indices = np.where(np.isclose(action_space, chosen_value))[0]
             if len(action_indices) == 0: logger.warning(f"QController ({param}): Could not find index for action {chosen_value:.4f}. Skipping update."); continue
             action_idx = action_indices[0]

             current_q = self.q_table[state][param][action_idx]
             next_q_values = self.q_table[next_state][param]
             # Find max Q-value in the next state robustly
             finite_next_q = next_q_values[np.isfinite(next_q_values)]
             max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0
             if not np.isfinite(max_future_q): max_future_q = 0.0

             td_target = reward + self.gamma * max_future_q; td_error = td_target - current_q
             adaptive_alpha = min(0.5, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error))))
             new_q = current_q + adaptive_alpha * td_error
             # Clip Q-value to prevent explosion and handle potential NaNs from update
             self.q_table[state][param][action_idx] = np.clip(new_q, -1e5, 1e5) if np.isfinite(new_q) else 0.0

    def _manage_q_table_size(self):
         """ Prunes the Q-table if it exceeds the maximum size using LRU strategy. """
         if len(self.q_table) > self.max_q_table_size:
            num_to_remove = len(self.q_table) - self.max_q_table_size
            logger.debug(f"Q-table ({len(self.q_table)}) exceeds limit ({self.max_q_table_size}). Pruning {num_to_remove}.")
            try:
                if not self.q_table_access_count or not self.q_table_creation_time: # Fallback if metadata missing
                     states_to_remove = random.sample(list(self.q_table.keys()), min(num_to_remove, len(self.q_table)))
                     logger.warning("Q-table metadata incomplete during pruning. Removing random states.")
                else:
                    # Sort states: primary key = access count (lower is worse), secondary = creation time (older is worse)
                    sorted_states = sorted(self.q_table.keys(), key=lambda s: (
                        self.q_table_access_count.get(s, 0), self.q_table_creation_time.get(s, float('inf'))))
                    states_to_remove = sorted_states[:num_to_remove]

                # Remove the selected states
                for state_to_remove in states_to_remove:
                    self.q_table.pop(state_to_remove, None)
                    self.q_table_access_count.pop(state_to_remove, None)
                    self.q_table_creation_time.pop(state_to_remove, None)
                logger.debug(f"Pruned {len(states_to_remove)} states. New Q-table size: {len(self.q_table)}")
            except Exception as e:
                logger.warning(f"Error during Q-table pruning: {e}. Attempting random removal.", exc_info=False)
                # Fallback: Random removal
                current_keys = list(self.q_table.keys())
                num_to_remove = len(current_keys) - self.max_q_table_size
                if num_to_remove > 0:
                    states_to_remove = random.sample(current_keys, min(num_to_remove, len(current_keys)))
                    for state_to_remove in states_to_remove:
                        self.q_table.pop(state_to_remove, None)
                        self.q_table_access_count.pop(state_to_remove, None)
                        self.q_table_creation_time.pop(state_to_remove, None)

    def get_info(self) -> Dict:
        """ Returns current status information about the Q-controller. """
        last_action = self.prev_actions_log[-1] if self.prev_actions_log else None
        avg_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.0
        return {"epsilon": self.epsilon, "stable_steps": self.stable_steps, "oscillation_counter": self.oscillation_counter,
                "q_table_size": len(self.q_table), "last_action": last_action,
                "avg_reward_last_20": avg_reward}


class HAKMEMEnhancedSGD(torch.optim.Optimizer):
    """
    SGD optimizer enhanced with Momentum, Weight Decay, Q-Learning Controller. (V3.7 Stability)
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm=1.0, q_learning_config={}, enable_flow=False, flow_coefficient=0.05, flow_momentum=0.95):
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.q_controller = HAKMEMQController(**q_learning_config)
        self.max_grad_norm = max_grad_norm # Max grad norm for clipping (used by Trainer)
        self._step_count = 0; self.current_loss = None
        self.flow_enabled = False # Flow disabled as it wasn't active in V3.2
        # self.flow_enabled = enable_flow; self.flow_coefficient = flow_coefficient; self.flow_momentum = flow_momentum
        self.gradient_stats = GradientStats() # For tracking clipping etc.
        # Initialize optimizer state
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # if self.flow_enabled: state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        # Reset gradient stats tracker at the beginning of each accumulation cycle
        if hasattr(self, 'gradient_stats') and self.gradient_stats:
            self.gradient_stats.reset()

    def set_current_loss(self, loss: Optional[float]):
         """ Stores the loss for the current step, used by the Q-controller. """
         if loss is not None and np.isfinite(loss):
            self.current_loss = loss
            # Update Q-controller's prev_loss for the *next* step's reward calculation
            if hasattr(self, 'q_controller') and self.q_controller:
                 self.q_controller.prev_loss = loss
         else:
             self.current_loss = None
             if hasattr(self, 'q_controller') and self.q_controller:
                  self.q_controller.prev_loss = None # Mark prev_loss as invalid
             logger.debug("Optimizer received non-finite or None loss.")


    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step. """
        loss_val = None
        if closure is not None:
            with torch.enable_grad(): loss_val = closure()
            self.set_current_loss(loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val)

        q_action = None
        # Calculate average grad norm *before* Q-update uses it
        # This norm reflects the gradients accumulated for the step about to be taken.
        grad_norm_avg = self._get_average_grad_norm()

        # Q-Learning Update and Action Selection
        if hasattr(self, 'q_controller') and self.q_controller:
            group = self.param_groups[0]; current_lr = group['lr']; current_momentum = group['momentum']
            # Determine Q-state based on outcome of the previous step/action
            # Use the loss passed via set_current_loss (which reflects the cycle just completed)
            q_state = self.q_controller.get_state(lr=current_lr, momentum=current_momentum, grad_norm=grad_norm_avg, loss=self.current_loss)

            # If we have previous state/action, update Q-table using the reward from the *previous* action
            if self.q_controller.prev_state is not None and self.q_controller.prev_action is not None and q_state is not None:
                # Compute reward based on loss change (prev_loss -> current_loss) and current grad norm
                # self.q_controller.prev_loss holds the loss *before* the action was taken
                reward = self.q_controller.compute_reward(self.current_loss, self.q_controller.prev_loss, grad_norm_avg)
                if np.isfinite(reward): # Only update if reward is valid
                    self.q_controller.update(self.q_controller.prev_state, self.q_controller.prev_action, reward, q_state)
                    logger.debug(f"Q-Update: PrevState={self.q_controller.prev_state}, Action={self.q_controller.prev_action}, Reward={reward:.3f}, CurrentState={q_state}")
                else:
                    logger.warning(f"QController: Skipping Q-update due to invalid reward ({reward}). Loss(prev={self.q_controller.prev_loss}, cur={self.current_loss}), GradNorm={grad_norm_avg}")

            # Choose next action based on the *current* state (result of previous step)
            if q_state is not None:
                q_action = self.q_controller.choose_action(q_state)
                # Store current state and action for the *next* update step
                self.q_controller.prev_state = q_state
                self.q_controller.prev_action = q_action
            else: # Handle case where state could not be determined
                 logger.debug("QController: Current state is None, cannot choose action or update Q-table properly.")
                 self.q_controller.prev_state = None # Reset prev state if current is invalid
                 self.q_controller.prev_action = None

        # Apply Q-Learning action (adjust LR and Momentum for the *upcoming* step)
        if q_action is not None:
            for group in self.param_groups:
                base_lr = group['base_lr']; base_momentum = group['base_momentum']
                new_lr = base_lr * q_action['lr_scale']; new_momentum = base_momentum * q_action['momentum_scale']
                group['lr'] = float(np.clip(new_lr, 1e-8, 0.1)) # Clamp LR
                group['momentum'] = float(np.clip(new_momentum, 0.5, 0.999)) # Clamp momentum

        # --- Standard SGD with Momentum Update ---
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']; weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue # Skip params without grads (already checked by Trainer before clipping)
                if not p.requires_grad: continue # Skip frozen params
                grad = p.grad
                if grad.is_sparse: raise RuntimeError('HAKMEMEnhancedSGD does not support sparse gradients')
                # Ensure grad is finite (should be guaranteed by Trainer checks before step)
                if not torch.isfinite(grad).all():
                     logger.error(f"Optimizer Error: Non-finite gradient encountered for param shape {p.shape} during step! This should have been caught earlier. Skipping update for this param.")
                     continue

                # Use float32 for update stability
                grad_float = grad.float()
                param_data_float = p.data.float()
                param_state = self.state[p]

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad_float = grad_float.add(param_data_float, alpha=weight_decay) # Use add_ for potential in-place if safe, otherwise out-of-place

                # Momentum buffer update
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf = param_state['momentum_buffer']
                # buf.mul_(momentum).add_(grad_float) # In-place operations
                buf = buf * momentum + grad_float # Out-of-place operations
                param_state['momentum_buffer'] = buf # Store updated buffer

                # Parameter update: p.data = p.data - lr * buf
                # Update parameter data using the correct data type
                update_step = buf * lr
                # p.data.add_(update_step, alpha=-1) # In-place update
                p.data = p.data - update_step.to(p.data.dtype) # Out-of-place update

        self._step_count += 1
        return loss_val # Return loss if closure was provided

    def _get_average_grad_norm(self):
        """ Calculates the average L2 norm of gradients across all parameters. Returns float('inf') if any grad is non-finite. """
        total_norm_sq = 0.0; num_params = 0; any_non_finite = False
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                grad_float = p.grad.detach().float()
                if not torch.isfinite(grad_float).all():
                    # logger.warning(f"Non-finite grad detected in param (shape {p.shape}) during norm calculation.") # Less noisy log
                    any_non_finite = True; break # Stop calculation if non-finite found
                param_norm_sq = torch.sum(grad_float.pow(2))
                total_norm_sq += param_norm_sq.item(); num_params += 1
            if any_non_finite: break
        # Return inf if non-finite grads found, otherwise calculate average norm
        if any_non_finite: return float('inf')
        if num_params == 0: return 0.0
        return math.sqrt(total_norm_sq / num_params) if num_params > 0 else 0.0

    def get_q_info(self) -> Dict:
        """ Retrieves status information from the Q-controller. """
        if hasattr(self, 'q_controller') and self.q_controller: return self.q_controller.get_info()
        return {}


# =====================================================================
# Utilities
# =====================================================================

def is_main_process():
    """Checks if the current process is the main process (rank 0 or not distributed)."""
    return not is_initialized() or get_rank() == 0

class RealToComplexProjection(nn.Module):
     """ Simple projection from a real vector to complex components (real, imag). """
     def __init__(self, input_dim: int, complex_dim: int):
         super().__init__()
         self.input_dim = input_dim; self.complex_dim = complex_dim
         self.projection = nn.Linear(input_dim, complex_dim * 2)
         nn.init.xavier_uniform_(self.projection.weight); nn.init.zeros_(self.projection.bias)
         logger.debug(f"RealToComplexProjection: Input {input_dim} -> Complex {complex_dim}")

     def forward(self, x_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.isfinite(x_real).all():
            logger.warning("NaN/Inf in RealToComplexProjection input. Replacing.")
            x_real = torch.nan_to_num(x_real)
        x_real = x_real.float() # Ensure float for linear layer
        projected = self.projection(x_real) # [..., input_dim] -> [..., complex_dim * 2]
        real_part, imag_part = torch.chunk(projected, 2, dim=-1)
        # Check output stability
        if not (torch.isfinite(real_part).all() and torch.isfinite(imag_part).all()):
            logger.warning("NaN/Inf detected in RealToComplexProjection output. Replacing.")
            real_part = torch.nan_to_num(real_part)
            imag_part = torch.nan_to_num(imag_part)
        return real_part, imag_part # Both [..., complex_dim]

# =====================================================================
# Integrated Model Definition (V3.9 - Stability Fixes)
# =====================================================================

class IntegratedHyperHAKMEMModel(nn.Module):
    """
    Integrated Model combining HAKMEM word/punct patching, Hyperbolic Geometry,
    and Complex Attention layers. (V3.9 Stability Fixes)
    """
    def __init__(
        self,
        local_hidden_size: int = 256, complex_dim: int = 512, num_complex_layers: int = 8,
        num_complex_heads: int = 8, decoder_memory_dim: int = 768, dropout: float = 0.15,
        context_window: int = 256,
        n_gram_sizes: List[int] = [3, 4], n_gram_vocab_size: int = 30000,
        sfin_noise_scale: float = 0.05, # Currently inactive param
        sfin_use_entanglement: bool = True, sfin_use_rotary: bool = True,
        projection_method: str = "hakmem_enhanced", use_hierarchical_decoder: bool = True,
        hyperbolic_embedding_dim: int = 384, curvature: float = 0.8, clipping_radius: float = 1.0,
        use_amp: bool = True
    ):
        super().__init__()
        self.local_hidden_size = local_hidden_size; self.complex_dim = complex_dim
        self.decoder_memory_dim = decoder_memory_dim; self.hyperbolic_embedding_dim = hyperbolic_embedding_dim
        # Ensure curvature is positive and small enough to be stable
        self.curvature = max(min(curvature, 5.0), 1e-6) # Clamp curvature
        self.clipping_radius = max(min(clipping_radius, 1.0), 0.1) # Clamp radius
        self.use_hierarchical_decoder = use_hierarchical_decoder
        self.context_window = context_window
        self.use_amp_in_generate = use_amp

        # --- Input Validation ---
        if complex_dim <= 0 or local_hidden_size <= 0 or decoder_memory_dim <= 0 or hyperbolic_embedding_dim <= 0: raise ValueError("All dimension sizes must be positive")
        if complex_dim % num_complex_heads != 0:
             logger.warning(f"complex_dim ({complex_dim}) not divisible by heads ({num_complex_heads}). Adjusting.")
             valid_heads = [h for h in range(num_complex_heads, 0, -1) if complex_dim % h == 0]
             num_complex_heads = valid_heads[0] if valid_heads else 1
             logger.warning(f"Set num_complex_heads to {num_complex_heads}")
        if num_complex_layers <= 0: raise ValueError("num_complex_layers must be positive")

        # --- Model Components ---
        self.hyperbolic_utils = HyperbolicUtils()
        self.patcher = HAKMEMBabylonIndex()

        local_encoder_heads = max(1, local_hidden_size // 64)
        self.local_encoder = HAKMEMLocalEncoder(
            hidden_size=local_hidden_size, num_layers=2, num_heads=local_encoder_heads,
            dropout=dropout, n_gram_sizes=n_gram_sizes, n_gram_vocab_size=n_gram_vocab_size
        )

        self.projection_to_hyp_euclidean = nn.Linear(local_hidden_size, hyperbolic_embedding_dim)
        nn.init.xavier_uniform_(self.projection_to_hyp_euclidean.weight); nn.init.zeros_(self.projection_to_hyp_euclidean.bias)

        self.tangent_to_complex = RealToComplexProjection(hyperbolic_embedding_dim, complex_dim)
        self.complex_norm_in = HAKMEMComplexLayerNorm(complex_dim, coupled=True)

        self.complex_interference_layers = nn.ModuleList()
        self.complex_norms_mid = nn.ModuleList()
        max_seq_len_for_rope = context_window * 2 # Heuristic max patch sequence length
        for _ in range(num_complex_layers):
             self.complex_norms_mid.append(HAKMEMComplexLayerNorm(complex_dim, coupled=True)) # Pre-LN
             self.complex_interference_layers.append(
                 HAKMEMEntangledInterferenceLayer( # V3.9 - Stability Fixes applied within this layer
                     dim=complex_dim, heads=num_complex_heads, dropout=dropout, noise_scale=sfin_noise_scale,
                     use_entanglement=sfin_use_entanglement, use_rotary=sfin_use_rotary,
                     adaptive_attention=True, max_seq_len=max_seq_len_for_rope
                 )
             )

        self.complex_to_tangent = HAKMEMComplexToRealProjection(complex_dim, hyperbolic_embedding_dim, method=projection_method)
        self.tangent_norm_out = nn.LayerNorm(hyperbolic_embedding_dim, eps=1e-6)

        self.projection_to_decoder_memory = nn.Linear(hyperbolic_embedding_dim, decoder_memory_dim)
        nn.init.xavier_uniform_(self.projection_to_decoder_memory.weight); nn.init.zeros_(self.projection_to_decoder_memory.bias)

        decoder_heads = max(1, local_hidden_size // 64)
        decoder_max_len = context_window * 2
        self.local_decoder = HAKMEMLocalDecoder(
            hidden_size=local_hidden_size, global_hidden_size=decoder_memory_dim, num_layers=4,
            num_heads=decoder_heads, dropout=dropout, use_hierarchical_pred=use_hierarchical_decoder,
            max_decode_len=decoder_max_len
        )
        # Learnable scaling factors for residual connections, initialized to 0 (sigmoid(0)=0.5)
        self.residual_controllers = nn.Parameter(torch.zeros(num_complex_layers))
        # --- End Model Components ---

        logger.info(f"IntegratedHyperHAKMEM V3.9 Initialized: HypDim={hyperbolic_embedding_dim}, Curve={self.curvature:.2f}, ComplexDim={complex_dim}, DecMem={decoder_memory_dim}") # Version Bump
        logger.info(f"Patching: Word/Punctuation | LocalEnc/Dec Hidden: {local_hidden_size}, Complex Layers: {num_complex_layers}, Heads: {num_complex_heads}")
        logger.info(f"Entanglement: {sfin_use_entanglement}, RoPE: {sfin_use_rotary}, Decoder Hierarchical: {use_hierarchical_decoder}")

    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = byte_seq.size(0); device = byte_seq.device; model_dtype = next(self.parameters()).dtype

        # --- 1. Patching and Local Encoding -> Memory Generation Path ---
        batch_patch_repr_list = []; num_patches_per_item = []; valid_batch_indices = []
        max_num_patches = 0

        for i in range(batch_size):
            seq = byte_seq[i]
            patches_with_entropy = self.patcher.create_patches(seq)

            if patches_with_entropy:
                patches_on_device = [(p.to(device), e) for p, e in patches_with_entropy] # Ensure device
                patch_repr_single = self.local_encoder(patches_on_device) # [1, num_p, local_hidden]

                if not torch.isfinite(patch_repr_single).all():
                     logger.warning(f"NaN/Inf detected from local_encoder for batch item {i}. Replacing.")
                     patch_repr_single = torch.nan_to_num(patch_repr_single)

                num_p = patch_repr_single.size(1)
                if num_p > 0:
                    batch_patch_repr_list.append(patch_repr_single.squeeze(0))
                    num_patches_per_item.append(num_p); valid_batch_indices.append(i)
                    max_num_patches = max(max_num_patches, num_p)
                else: num_patches_per_item.append(0)
            else: num_patches_per_item.append(0)

        # --- Handle case where NO valid patches were generated ---
        if not valid_batch_indices:
             target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
             logger.warning("No valid patches generated for any item in the batch. Decoder memory will be empty.")
             if target_byte_seq is not None: return torch.zeros((batch_size, target_len, 256), device=device, dtype=torch.float32)
             else: return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)

        # --- 2. Padding and Stacking Valid Patch Representations ---
        num_valid = len(valid_batch_indices)
        padded_patch_repr = torch.zeros(num_valid, max_num_patches, self.local_hidden_size, device=device, dtype=model_dtype)
        memory_padding_mask = torch.ones(num_valid, max_num_patches, dtype=torch.bool, device=device) # True=MASK
        for idx, patch_repr_tensor in enumerate(batch_patch_repr_list):
             num_p = patch_repr_tensor.size(0)
             padded_patch_repr[idx, :num_p, :] = patch_repr_tensor
             memory_padding_mask[idx, :num_p] = False # False where valid data exists

        # --- 3. Hyperbolic Processing Path ---
        euclidean_for_hyper = self.projection_to_hyp_euclidean(padded_patch_repr)
        if not torch.isfinite(euclidean_for_hyper).all(): logger.warning("NaN/Inf after projection_to_hyp_euclidean."); euclidean_for_hyper = torch.nan_to_num(euclidean_for_hyper)

        clipped_euclidean = self.hyperbolic_utils.poincare_clip(euclidean_for_hyper, self.curvature, self.clipping_radius)
        clipped_euclidean = torch.nan_to_num(clipped_euclidean) # V3.7 Add check
        if not torch.isfinite(clipped_euclidean).all(): logger.warning("NaN/Inf after poincare_clip."); clipped_euclidean = torch.nan_to_num(clipped_euclidean) # Redundant but safe

        hyperbolic_repr = self.hyperbolic_utils.exponential_map(clipped_euclidean, self.curvature)
        hyperbolic_repr = torch.nan_to_num(hyperbolic_repr) # V3.7 Add check
        if not torch.isfinite(hyperbolic_repr).all(): logger.warning("NaN/Inf after exponential_map."); hyperbolic_repr = torch.nan_to_num(hyperbolic_repr) # Redundant but safe

        tangent_repr = self.hyperbolic_utils.logarithmic_map(hyperbolic_repr, self.curvature)
        tangent_repr = torch.nan_to_num(tangent_repr) # V3.7 Add check
        if not torch.isfinite(tangent_repr).all(): logger.warning("NaN/Inf after logarithmic_map."); tangent_repr = torch.nan_to_num(tangent_repr) # Redundant but safe

        # --- 4. Complex Processing in Tangent Space ---
        complex_tangent_repr_real, complex_tangent_repr_imag = self.tangent_to_complex(tangent_repr)
        if not (torch.isfinite(complex_tangent_repr_real).all() and torch.isfinite(complex_tangent_repr_imag).all()):
            logger.warning("NaN/Inf after tangent_to_complex.")
            complex_tangent_repr_real = torch.nan_to_num(complex_tangent_repr_real)
            complex_tangent_repr_imag = torch.nan_to_num(complex_tangent_repr_imag)

        real, imag = self.complex_norm_in((complex_tangent_repr_real, complex_tangent_repr_imag))
        if not (torch.isfinite(real).all() and torch.isfinite(imag).all()):
            logger.warning("NaN/Inf after complex_norm_in.")
            real = torch.nan_to_num(real); imag = torch.nan_to_num(imag)

        # Pass through the stack of complex interference layers
        for i, layer in enumerate(self.complex_interference_layers):
            real_res, imag_res = real, imag
            normed_real, normed_imag = self.complex_norms_mid[i]((real, imag))
            if not (torch.isfinite(normed_real).all() and torch.isfinite(normed_imag).all()):
                logger.warning(f"NaN/Inf detected BEFORE complex layer {i}. Replacing.")
                normed_real = torch.nan_to_num(normed_real); normed_imag = torch.nan_to_num(normed_imag)

            # Apply complex attention layer
            # The mask needs to be broadcastable from [B_valid, M] to [B_valid, h, M, M] or similar
            # The attention layer should handle broadcasting padding_mask [B, M] -> [B, 1, 1, M]
            out_real, out_imag = layer((normed_real, normed_imag), attention_mask=memory_padding_mask)
            if not (torch.isfinite(out_real).all() and torch.isfinite(out_imag).all()):
                 logger.warning(f"NaN/Inf detected AFTER complex layer {i}. Replacing.")
                 out_real = torch.nan_to_num(out_real); out_imag = torch.nan_to_num(out_imag)

            # Apply learnable residual scaling (clamp raw controller param before sigmoid)
            clamped_res_ctrl = torch.clamp(self.residual_controllers[i], -5.0, 5.0)
            residual_strength = torch.sigmoid(clamped_res_ctrl)
            real = real_res + out_real * residual_strength
            imag = imag_res + out_imag * residual_strength
            # Check stability after residual connection
            if not (torch.isfinite(real).all() and torch.isfinite(imag).all()):
                 logger.warning(f"NaN/Inf detected AFTER residual connection in layer {i}. Replacing.")
                 real = torch.nan_to_num(real); imag = torch.nan_to_num(imag)
        processed_complex_tangent = (real, imag)

        # --- 5. Projection Back and Final Memory ---
        processed_tangent_repr = self.complex_to_tangent(processed_complex_tangent)
        if not torch.isfinite(processed_tangent_repr).all():
            logger.warning("NaN/Inf after complex_to_tangent."); processed_tangent_repr = torch.nan_to_num(processed_tangent_repr)

        processed_tangent_repr_normed = self.tangent_norm_out(processed_tangent_repr)
        if not torch.isfinite(processed_tangent_repr_normed).all():
             logger.warning("NaN/Inf after tangent_norm_out."); processed_tangent_repr_normed = torch.nan_to_num(processed_tangent_repr_normed)

        decoder_memory = self.projection_to_decoder_memory(processed_tangent_repr_normed)
        if not torch.isfinite(decoder_memory).all():
            logger.warning("NaN/Inf detected in final decoder_memory. Replacing."); decoder_memory = torch.nan_to_num(decoder_memory)

        # --- 6. Local Decoding Path ---
        if target_byte_seq is None: # Generation case
            logger.debug("target_byte_seq is None in forward pass (likely initial gen step).")
            return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)

        target_len = target_byte_seq.size(1)
        if target_len == 0:
             logger.debug("Received empty target_byte_seq for decoder.")
             return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)

        # Select target sequences for valid items
        valid_target_byte_seq = target_byte_seq[valid_batch_indices].to(device) if valid_batch_indices else torch.empty(0, target_len, dtype=torch.long, device=device)

        # Call Local Decoder only if there are valid items
        if num_valid > 0:
            byte_logits_valid = self.local_decoder(
                tgt_byte_seq=valid_target_byte_seq,
                memory=decoder_memory, # Has shape [B_valid, max_num_patches, dec_mem_dim]
                memory_key_padding_mask=memory_padding_mask # Has shape [B_valid, max_num_patches]
            ) # Output: [B_valid, S_tgt, 256]
        else:
            # Create empty logits tensor if no valid items processed
             byte_logits_valid = torch.empty((0, target_len, 256), device=device, dtype=torch.float32)


        # --- 7. Reconstruct Full Batch Output ---
        final_byte_logits = torch.zeros((batch_size, target_len, 256), device=device, dtype=torch.float32)

        if num_valid > 0 and byte_logits_valid.numel() > 0:
            if not torch.isfinite(byte_logits_valid).all():
                logger.warning("NaN/Inf detected in byte_logits_valid from decoder before final reconstruction. Replacing.")
                byte_logits_valid = torch.nan_to_num(byte_logits_valid)

            valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
            try:
                # Ensure tensor shapes match for index_copy_
                if byte_logits_valid.shape[0] == valid_indices_tensor.shape[0]:
                    final_byte_logits.index_copy_(0, valid_indices_tensor, byte_logits_valid)
                elif byte_logits_valid.numel() > 0: # Check if there's something to copy but shapes mismatch
                    logger.error(f"Shape mismatch for index_copy_: final={final_byte_logits.shape}, valid={byte_logits_valid.shape}, indices={valid_indices_tensor.shape}. Skipping copy.")
            except IndexError as e:
                logger.error(f"Error during final logit reconstruction: {e}. Shapes: final={final_byte_logits.shape}, valid={byte_logits_valid.shape}, indices={valid_indices_tensor.shape}")
                # Fallback: Fill with zeros or handle differently? Keeping zeros for now.

        # Final check on the reconstructed logits
        if not torch.isfinite(final_byte_logits).all():
             logger.error("NaN/Inf detected in final reconstructed logits! Replacing with zeros.")
             final_byte_logits = torch.nan_to_num(final_byte_logits, nan=0.0, posinf=0.0, neginf=0.0)

        return final_byte_logits


    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        """ Computes cross-entropy loss with optional label smoothing, ignoring masked positions. """
        batch_size, seq_len, vocab_size = logits.shape
        if seq_len == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Ensure logits are float32 for stability
        logits = logits.float()
        targets = targets.long()
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

        # Check for NaNs in logits before loss
        if not torch.isfinite(logits_flat).all():
            logger.error("NaN/Inf detected in logits passed to compute_loss. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, requires_grad=True) # Return high loss

        # V3.9: Check for NaNs in targets (extra safety)
        if not torch.isfinite(targets_flat).all():
            logger.error("NaN/Inf detected in targets passed to compute_loss. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, requires_grad=True)

        if smoothing > 0.0:
            with torch.no_grad():
                smooth_val_on = 1.0 - smoothing
                smooth_val_off = smoothing / max(1, vocab_size - 1)
                true_dist = torch.full_like(logits_flat, smooth_val_off)
                true_dist.scatter_(1, targets_flat.unsqueeze(1), smooth_val_on)

            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -(true_dist * log_probs).sum(dim=-1) # Shape: [B*S]
        else:
            # Use reduction='none' to handle masking manually
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none') # Shape: [B*S]

        # Check for NaNs in loss *before* masking
        if not torch.isfinite(loss).all():
            logger.error("NaN/Inf detected in loss calculation before masking. Returning high loss.")
            # Identify where NaNs occurred if possible
            nan_mask = torch.isnan(loss)
            # nan_logits = logits_flat[nan_mask]
            # nan_targets = targets_flat[nan_mask]
            # logger.debug(f"NaN loss occurred for targets: {nan_targets[:10]}") # Log some info
            return torch.tensor(100.0, device=logits.device, requires_grad=True) # Return high loss


        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1).bool() # True=IGNORE
            loss = loss.masked_fill(mask_flat, 0.0)
            num_active_elements = (~mask_flat).sum()
            if num_active_elements.item() == 0: # Avoid division by zero if all elements are masked
                 mean_loss = torch.tensor(0.0, device=logits.device)
            else:
                 mean_loss = loss.sum() / num_active_elements
        else:
            mean_loss = loss.mean()

        # Final check on mean loss
        if not torch.isfinite(mean_loss):
            logger.error(f"NaN/Inf detected in final computed mean loss. Original loss sum: {loss.sum()}, Num active: {num_active_elements if mask is not None else loss.numel()}. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, requires_grad=True) # Return high loss

        return mean_loss

    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None, repetition_penalty: float = 1.1, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """ Generates byte sequences autoregressively with advanced sampling options. """
        self.eval(); device = next(self.parameters()).device; model_dtype = next(self.parameters()).dtype

        if seed_bytes.device != device: seed_bytes = seed_bytes.to(device)
        seed_bytes = seed_bytes.long() # Ensure input is long tensor
        batch_size, seed_len = seed_bytes.size()
        if seed_len == 0: logger.warning("Generate called with empty seed."); return torch.empty((batch_size, 0), dtype=torch.long, device=device)

        generated_sequence = seed_bytes.clone()
        if sampling_config is None: sampling_config = SamplerConfig()

        sequence_memory = [{} for _ in range(batch_size)]
        max_ngram_size = 5

        base_temperature = max(temperature, 1e-6)
        min_temp = max(0.1, base_temperature * 0.5)
        max_temp = min(2.0, base_temperature * 1.5)

        disable_tqdm = not is_main_process() or batch_size > 8 or max_length < 20
        gen_iterator = tqdm(range(max_length), desc="Generating", disable=disable_tqdm, total=max_length, unit="byte")

        for step in gen_iterator:
            current_context = generated_sequence.long()
            context_len = current_context.size(1)

            amp_context = amp.autocast(device_type=device.type, enabled=self.use_amp_in_generate)
            with torch.no_grad(), amp_context:
                 # During generation, target_byte_seq is the same as input context
                 logits_all = self(byte_seq=current_context, target_byte_seq=current_context) # [B, current_len, 256]

            if logits_all is None or logits_all.numel() == 0 or logits_all.shape[1] == 0:
                 logger.warning(f"Logits generation failed or returned empty at step {step}. Stopping generation.")
                 break
            if not torch.isfinite(logits_all).all():
                 logger.warning(f"NaN/Inf detected in model output logits at generation step {step}. Replacing with zeros.")
                 logits_all = torch.nan_to_num(logits_all, nan=0.0, posinf=0.0, neginf=0.0)

            next_byte_logits = logits_all[:, -1, :].float() # [B, 256]
            next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

            for i in range(batch_size):
                current_logits = next_byte_logits[i].clone()
                current_seq = generated_sequence[i]; current_seq_len = current_seq.size(0)

                # 1. Repetition Penalty
                if repetition_penalty > 1.0 and current_seq_len > 0:
                    for ngram_size in range(2, min(max_ngram_size + 1, current_seq_len + 1)):
                        recent_ngram_tuple = tuple(current_seq[-(ngram_size-1):].cpu().tolist())
                        if recent_ngram_tuple in sequence_memory[i]:
                             prev_next_byte = sequence_memory[i][recent_ngram_tuple]
                             if 0 <= prev_next_byte < 256:
                                 if current_logits[prev_next_byte] > 0: current_logits[prev_next_byte] /= repetition_penalty
                                 else: current_logits[prev_next_byte] *= repetition_penalty

                # 2. Adaptive Temperature
                probs_orig = F.softmax(current_logits, dim=-1)
                if torch.isnan(probs_orig).any():
                    logger.warning(f"NaN detected in pre-temperature probs item {i} step {step}. Using uniform.")
                    entropy = math.log2(256.0)
                else:
                    entropy = -torch.sum(probs_orig * torch.log2(probs_orig + 1e-10)).item()

                adaptive_temp = base_temperature
                if entropy < sampling_config.low_entropy_threshold: adaptive_temp *= 0.8
                elif entropy > sampling_config.medium_entropy_threshold: adaptive_temp *= 1.1
                adaptive_temp = max(min_temp, min(adaptive_temp, max_temp))
                scaled_logits = current_logits / adaptive_temp

                # 3. Top-K / Top-P
                filtered_logits = scaled_logits
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    top_k_threshold = torch.topk(filtered_logits, k)[0][..., -1, None]
                    indices_to_remove = filtered_logits < top_k_threshold
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))

                # 4. Sample
                probs_final = F.softmax(filtered_logits, dim=-1)
                if torch.isnan(probs_final).any() or torch.isinf(probs_final).any() or probs_final.sum() < 1e-6:
                    logger.warning(f"Invalid sampling probabilities item {i} step {step}. Using uniform.")
                    probs_final = torch.ones_like(current_logits) / current_logits.size(-1)

                if temperature <= 1e-6: next_byte_idx = torch.argmax(probs_final)
                else: next_byte_idx = torch.multinomial(probs_final, num_samples=1).squeeze(-1)

                next_byte_indices[i] = next_byte_idx.item()

                # Update Repetition Memory
                if current_seq_len > 0:
                    for ngram_size in range(1, min(max_ngram_size, current_seq_len) + 1):
                         ngram_key = tuple(current_seq[-(ngram_size):].cpu().tolist())
                         sequence_memory[i][ngram_key] = next_byte_idx.item()
                if len(sequence_memory[i]) > 2000: # Cache eviction
                     keys_to_remove = random.sample(list(sequence_memory[i].keys()), 500)
                     for k in keys_to_remove: sequence_memory[i].pop(k, None)

            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)
            if not disable_tqdm: gen_iterator.set_description(f"Generating (Len {generated_sequence.size(1)})")

        if not disable_tqdm: gen_iterator.close()
        return generated_sequence


# =====================================================================
# ByteTokenizer
# =====================================================================
class ByteTokenizer:
    """Simple stateless tokenizer for byte-level processing."""
    def encode(self, text: str) -> List[int]:
        try: return list(text.encode('utf-8'))
        except Exception as e: logger.error(f"Error encoding text: {e}"); return []

    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        valid_bytes = []
        for b in byte_sequence:
            try:
                val = b.item() if isinstance(b, torch.Tensor) else int(b)
                if 0 <= val <= 255: valid_bytes.append(val)
                else: logger.warning(f"Invalid byte value {val} during decoding. Skipping.")
            except Exception as e: logger.warning(f"Error processing byte {b} during decoding: {e}. Skipping."); continue
        try:
            return bytes(valid_bytes).decode('utf-8', errors='replace')
        except Exception as e: logger.error(f"Error decoding bytes: {e}"); return ""


# =====================================================================
# ByteIterableDataset (Corrected for Windows Multiprocessing V3.4)
# =====================================================================
class ByteIterableDataset(IterableDataset):
    """
    Iterable dataset for loading sequences of bytes from a numpy file using memory mapping.
    Corrected for Windows multiprocessing compatibility (V3.4).
    Yields (context, next_byte_target) for standard autoregressive setup.
    """
    def __init__(self, npy_file_path: str, context_size: int = 256, data_fraction: float = 1.0):
        if not os.path.exists(npy_file_path): raise FileNotFoundError(f"Dataset not found: {npy_file_path}")
        if context_size <= 0: raise ValueError("context_size must be positive")
        if not (0.0 < data_fraction <= 1.0): raise ValueError("data_fraction must be between 0 (exclusive) and 1 (inclusive)")

        self.npy_file_path = npy_file_path  # Store the path, NOT the mmap handle
        self.context_size = context_size
        self.data_fraction = data_fraction
        self.full_data_size = 0
        self.data_size = 0
        self.num_possible_samples = 0
        self.data_dtype = np.uint8 # Assume uint8
        self.seed = None # For worker shuffling
        self.epoch = 0   # For worker shuffling

        # --- Calculate metadata without loading data via mmap in __init__ ---
        try:
            # Get array header info without loading full data if possible
            with open(self.npy_file_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                # Use internal function carefully, might change across numpy versions
                shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
                self.full_data_size = shape[0]
                # Basic check for dtype, might need refinement
                # Check if dtype is uint8 or byte ('|u1' or '|S1')
                if dtype.itemsize != 1 or dtype.kind not in ('u', 'S'):
                     logger.warning(f"Dataset warning: Read dtype {dtype} might not be uint8/byte. Assuming uint8 for calculations.")
                self.data_dtype = np.uint8 # Store expected dtype

            if self.full_data_size == 0:
                 raise ValueError("Dataset file appears empty or header reading failed.")

            self.data_size = int(self.full_data_size * self.data_fraction)
            if self.data_size <= self.context_size:
                 raise ValueError(f"Dataset size after fraction ({self.data_size}) not larger than context size ({self.context_size})")

            # Last possible start idx is data_size - context_size - 1
            self.num_possible_samples = max(0, self.data_size - self.context_size)
            if self.num_possible_samples == 0:
                 raise ValueError(f"Dataset size {self.data_size} too small for context {self.context_size}, no samples possible.")

            logger.info(f"Dataset Initialized (Metadata): Using {self.data_size:,}/{self.full_data_size:,} bytes ({self.data_fraction:.1%}) from {npy_file_path}")
            logger.info(f"Dataset Context: {self.context_size}, Possible start indices: {self.num_possible_samples:,}")

        except AttributeError:
            logger.warning("Failed to use numpy internal header reading (likely older numpy version). Falling back to np.load for metadata.")
            temp_data = None
            try:
                temp_data = np.load(self.npy_file_path, mmap_mode='r')
                if len(temp_data.shape) != 1: raise ValueError(f"Dataset error: Expected 1D array, got shape {temp_data.shape}")
                if not np.issubdtype(temp_data.dtype, np.uint8): logger.warning(f"Dataset warning: Expected uint8 data, got {temp_data.dtype}. Casting.")
                self.full_data_size = temp_data.shape[0]; self.data_dtype = np.uint8

                self.data_size = int(self.full_data_size * self.data_fraction)
                if self.data_size <= self.context_size:
                    raise ValueError(f"Dataset size after fraction ({self.data_size}) not larger than context size ({self.context_size})")

                self.num_possible_samples = max(0, self.data_size - self.context_size)
                if self.num_possible_samples == 0:
                    raise ValueError(f"Dataset size {self.data_size} too small for context {self.context_size}, no samples possible.")

                logger.info(f"Dataset Initialized (Fallback Load): Using {self.data_size:,}/{self.full_data_size:,} bytes ({self.data_fraction:.1%}) from {npy_file_path}")
                logger.info(f"Dataset Context: {self.context_size}, Possible start indices: {self.num_possible_samples:,}")

            finally:
                if temp_data is not None and hasattr(temp_data, '_mmap') and temp_data._mmap is not None:
                    try: temp_data._mmap.close()
                    except Exception: pass
                del temp_data; gc.collect()

        except Exception as e:
            logger.error(f"Error reading dataset metadata from {self.npy_file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        # Provides an estimate for progress bars etc.
        return self.num_possible_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # --- Calculate indices for this worker ---
        num_samples_total = self.num_possible_samples
        if num_samples_total == 0: # Handle edge case where calculation failed
             return iter([])
        num_samples_per_worker = num_samples_total // num_workers
        start_sample_idx = worker_id * num_samples_per_worker
        end_sample_idx = (worker_id + 1) * num_samples_per_worker
        remainder = num_samples_total % num_workers
        if worker_id < remainder:
            start_sample_idx += worker_id; end_sample_idx += worker_id + 1
        else:
            start_sample_idx += remainder; end_sample_idx += remainder
        end_sample_idx = min(end_sample_idx, num_samples_total)

        bytes_data = None # Initialize variable holding the mmap array
        mmap_handle = None # Initialize variable holding the raw mmap handle

        try:
            # --- Load data using mmap *INSIDE* the worker process ---
            bytes_data = np.load(self.npy_file_path, mmap_mode='r')
            # Check dtype consistency upon actual loading
            if not np.issubdtype(bytes_data.dtype, self.data_dtype):
                 logger.warning(f"Worker {worker_id}: Loaded data dtype {bytes_data.dtype} differs from expected {self.data_dtype}. Proceeding.")
            # Get the actual mmap handle for cleanup
            if hasattr(bytes_data, '_mmap') and bytes_data._mmap is not None:
                 mmap_handle = bytes_data._mmap
            else: logger.debug(f"Worker {worker_id}: Could not get _mmap attribute from loaded numpy array (might not be memory-mapped).")


            # --- Shuffle and Iterate ---
            if start_sample_idx >= end_sample_idx: # No samples for this worker
                return iter([])

            worker_indices = np.arange(start_sample_idx, end_sample_idx, dtype=np.int64)
            # Ensure different seed per worker and per epoch
            # Use the seed set via set_seed() if available
            base_seed = self.seed if self.seed is not None else 42
            current_epoch = self.epoch # Use epoch set by sampler/trainer
            seed = (base_seed + worker_id + current_epoch + int(time.time() * 1000) + os.getpid()) % (2**32) # More robust seeding
            rng = np.random.default_rng(seed=seed); rng.shuffle(worker_indices)

            for idx in worker_indices:
                start_ctx = idx; end_ctx = idx + self.context_size; end_tgt = end_ctx + 1
                # Boundary check (using self.data_size calculated in __init__)
                if end_tgt > self.data_size:
                    logger.warning(f"Worker {worker_id}: Calculated index {idx} leads to out-of-bounds access ({end_tgt} > {self.data_size}). Skipping.")
                    continue
                try:
                    # Extract slices AND copy immediately to read from mmap into worker memory
                    context_slice = bytes_data[start_ctx : end_ctx].copy()
                    target_slice = bytes_data[start_ctx + 1 : end_tgt].copy()
                    # Convert copied slices to tensors
                    context_tensor = torch.tensor(context_slice, dtype=torch.long)
                    target_tensor = torch.tensor(target_slice, dtype=torch.long)
                    yield context_tensor, target_tensor
                except IndexError:
                     logger.warning(f"Worker {worker_id}: IndexError accessing data at index {idx} (Context: {start_ctx}-{end_ctx}, Target: {start_ctx+1}-{end_tgt}, Data Size: {self.data_size}). Skipping.")
                     continue
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error processing index {idx}: {e}", exc_info=False)
                    continue

        except FileNotFoundError:
            logger.error(f"Worker {worker_id}: Dataset file not found: {self.npy_file_path}")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed dataset iteration setup or execution: {e}", exc_info=True)
        finally:
            # Crucial: Ensure mmap file handle is closed when worker iterator finishes/errors
            if mmap_handle is not None:
                 try: mmap_handle.close()
                 except Exception as close_ex: logger.warning(f"Worker {worker_id}: Error closing mmap handle: {close_ex}")
            # Delete the reference to the numpy mmap array
            del bytes_data
            gc.collect()
            # logger.debug(f"Worker {worker_id}: Iterator finished and mmap closed.")

    # Add seed attribute for reproducibility in shuffling within workers
    def set_seed(self, seed):
        logger.debug(f"Dataset seed set to {seed}")
        self.seed = seed

    # Add epoch attribute for shuffling within workers based on epoch (used by DistributedSampler)
    def set_epoch(self, epoch):
        logger.debug(f"Dataset epoch set to {epoch}")
        self.epoch = epoch


# =====================================================================
# Trainer Class (V3.9 - Stability Fix)
# =====================================================================
class Trainer:
    """ Trainer class with DDP, AMP, Gradient Accumulation, Q-Learning Optimizer, Checkpointing, and Anomaly Detection. (V3.9) """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10, save_interval: int = 1000, checkpoint_dir: str = "checkpoints", wandb_enabled: bool = False, max_grad_norm: float = 1.0, rank: int = 0, world_size: int = 1, detect_anomaly: bool = False):
        self.model = model; self.optimizer = optimizer; self.device = device
        self.train_loader = train_loader; self.val_loader = val_loader
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch, "amp")
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.log_interval = log_interval; self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir; self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE
        self.max_grad_norm = max_grad_norm; self.global_step = 0; self.current_epoch = 0
        self.last_val_metrics = None; self.rank = rank; self.world_size = world_size
        self.is_main = is_main_process()
        self.detect_anomaly = detect_anomaly
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(self.optimizer.gradient_stats, GradientStats)
        self.has_q_controller = hasattr(self.optimizer, 'q_controller') and isinstance(self.optimizer.q_controller, HAKMEMQController)
        logger.info(f"Trainer initialized Rank {rank}. AMP: {self.use_amp}, Grad Accum: {self.grad_accum_steps}, Max Norm: {self.max_grad_norm}, Opt: {type(self.optimizer).__name__}, Detect Anomaly: {self.detect_anomaly}")
        if self.has_q_controller: logger.info("Optimizer has Q-Learning Controller.")
        if self.has_grad_stats: logger.info("Optimizer has Gradient Stats Tracking.")

    def _train_epoch(self):
        """ Trains the model for one epoch using standard autoregressive loss. """
        self.model.train()
        epoch_loss = 0.0; optimizer_steps_in_epoch = 0; micro_step_count_cycle = 0; total_loss_accum_cycle = 0.0
        approx_total_micro_batches = -1
        try:
            # Estimate length for TQDM. Might fail for some IterableDatasets.
            num_samples_total = len(self.train_loader.dataset)
            # Correctly get batch size (use batch_size attribute, handle None case)
            loader_batch_size = self.train_loader.batch_size if self.train_loader.batch_size is not None else 1
            batch_size_global = loader_batch_size * self.world_size
            if batch_size_global > 0 and num_samples_total > 0:
                 # TQDM total should reflect optimizer steps
                 approx_total_optim_steps = num_samples_total // (batch_size_global * self.grad_accum_steps)
                 if approx_total_optim_steps <= 0: approx_total_optim_steps = -1 # Handle small dataset case
                 logger.debug(f"Estimated total optimizer steps per epoch: {approx_total_optim_steps}")
            else:
                approx_total_optim_steps = None # Cannot estimate length reliably
        except TypeError:
            logger.warning("DataLoader dataset lacks __len__. Progress bar will be infinite.")
            approx_total_optim_steps = None # Signify unknown length

        disable_tqdm = not self.is_main
        batch_iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}", disable=disable_tqdm, total=approx_total_optim_steps, unit="opt_step")

        # Ensure optimizer grads are zeroed at the start of the epoch's accumulation cycle
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch_data in enumerate(batch_iterator if isinstance(batch_iterator, Iterable) else self.train_loader): # Use iterator source more carefully
            micro_step_count_cycle += 1
            is_last_micro_step = (micro_step_count_cycle % self.grad_accum_steps == 0)
            should_optimizer_step = is_last_micro_step
            retain_graph_flag = not should_optimizer_step

            # Context manager for DDP gradient synchronization
            sync_context = contextlib.nullcontext()
            if self.world_size > 1 and isinstance(self.model, DistributedDataParallel):
                 if not should_optimizer_step: sync_context = self.model.no_sync()

            anomaly_context = torch.autograd.detect_anomaly(check_nan=True) if self.detect_anomaly else contextlib.nullcontext() # V3.8 Added check_nan=True

            # --- Forward and Backward Pass ---
            loss_value = None # Initialize loss value to None
            loss = None
            logits = None
            try:
                with sync_context, anomaly_context:
                    # --- Data Handling ---
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        context, targets = batch_data
                    else:
                        logger.warning(f"Rank {self.rank}: Skipping unexpected batch format at micro_batch index {i}. Type: {type(batch_data)}")
                        if is_last_micro_step: micro_step_count_cycle = 0 # Reset if skipping on last step
                        continue

                    context = context.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    batch_size, ctx_len = context.shape
                    if ctx_len == 0:
                        if is_last_micro_step: micro_step_count_cycle = 0 # Reset if skipping on last step
                        continue

                    decoder_input_seq = context

                    # --- Forward pass ---
                    with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                        logits = self.model(byte_seq=context, target_byte_seq=decoder_input_seq)
                        if logits.shape[0] != batch_size or logits.shape[1] != ctx_len or logits.shape[2] != 256:
                             logger.warning(f"Rank {self.rank}: Logits shape mismatch batch {i}. Inp: {context.shape}, Tgt: {targets.shape}, Got: {logits.shape}. Skip.")
                             raise ValueError("Logits shape mismatch") # Trigger exception handling
                        if not torch.isfinite(logits).all():
                             logger.error(f"Rank {self.rank}: NaN/Inf detected in model logits batch {i}, step {self.global_step}. Skip backward.")
                             raise ValueError("NaN/Inf detected in model logits")

                        # Compute loss
                        loss = self.model.compute_loss(logits, targets, smoothing=0.1)
                        if not torch.isfinite(loss): # V3.7 Check returned loss
                             logger.error(f"Rank {self.rank}: NaN/Inf loss ({loss.item()}) computed batch {i}, step {self.global_step}. Skip backward.")
                             raise ValueError("NaN/Inf loss computed")

                        # Normalize loss for accumulation
                        loss_value = loss / self.grad_accum_steps # Store normalized loss

                    # --- Backward pass ---
                    # Use retain_graph=retain_graph_flag
                    self.scaler.scale(loss_value).backward(retain_graph=retain_graph_flag)

                    # Accumulate loss for logging (use un-normalized loss)
                    current_step_loss = loss_value.item() * self.grad_accum_steps
                    if not np.isfinite(current_step_loss): current_step_loss = 0.0
                    total_loss_accum_cycle += current_step_loss
                    epoch_loss += current_step_loss # Accumulate total epoch loss using micro-batch losses

            except Exception as batch_ex:
                # Log with micro-batch index 'i' and micro_step_count_cycle
                logger.error(f"Error in micro-step {micro_step_count_cycle} (Batch Index {i}, Global {self.global_step}) Rank {self.rank}: {batch_ex}", exc_info=True) # Log full traceback on error
                # If an error occurs during forward/backward of a micro-step:
                logger.warning(f"Rank {self.rank}: Zeroing gradients due to error in micro-step {micro_step_count_cycle}.")
                try:
                    self.optimizer.zero_grad(set_to_none=True) # Crucial fix
                except Exception as zero_ex:
                     logger.error(f"Rank {self.rank}: Failed to zero grads after error: {zero_ex}")


                # Attempt to free memory/references (V3.6 Change)
                try: del loss_value, loss, logits
                except NameError: pass # Ignore if they weren't defined yet
                if torch.cuda.is_available(): torch.cuda.empty_cache() # Be more aggressive with cache clearing on error

                total_loss_accum_cycle = 0.0 # Reset loss accumulator
                micro_step_count_cycle = 0 # Reset counter for this cycle
                should_optimizer_step = False # Prevent optimizer step for this cycle
                # If an error occurred, we skip the rest of the loop for this micro-batch
                # The optimizer step logic below will be skipped if should_optimizer_step is False
                continue # Skip to the next micro-batch

            # --- Optimizer Step (Conditionally) ---
            if should_optimizer_step:
                num_steps_in_cycle = micro_step_count_cycle # Actual micro-steps completed
                avg_loss_cycle = total_loss_accum_cycle / num_steps_in_cycle if num_steps_in_cycle > 0 else 0.0
                total_loss_accum_cycle = 0.0 # Reset loss accumulator for next cycle
                micro_step_count_cycle = 0 # Reset micro-step counter for next cycle

                step_was_skipped = False # Initialize flag
                grad_norm_item = float('inf') # Default value
                optimizer_step_successful = False # Track successful step (V3.6 Change)
                scaler_skipped_step = False # Track if scaler specifically skipped (V3.6 Change)

                try:
                    # 1. Unscale gradients (before clipping)
                    self.scaler.unscale_(self.optimizer)

                    # 2. Check for NaN/Inf in gradients AFTER unscaling
                    has_invalid_grad = False
                    params_with_grad = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                    if params_with_grad: # Only check if there are grads
                        for p_idx, p in enumerate(params_with_grad):
                            if p.grad is not None and not torch.isfinite(p.grad).all(): # Check grad is not None again just in case
                                # Find parameter name if possible (V3.8)
                                param_name = f"Param_{p_idx}"
                                for name, param in self.model.named_parameters():
                                     if param is p:
                                         param_name = name
                                         break
                                logger.warning(f"Rank {self.rank}: NaN/Inf detected in unscaled gradient of '{param_name}' (shape {p.shape}) at step {self.global_step}.")
                                has_invalid_grad = True
                                break
                    if has_invalid_grad:
                        logger.warning(f"Rank {self.rank}: Invalid unscaled gradients detected. Skipping optimizer step and zeroing gradients.")
                        step_was_skipped = True
                        self.optimizer.zero_grad(set_to_none=True) # Zero grads immediately
                        # Skip to finally block to ensure scaler.update() is called

                    # 3. Clip gradients (only if grads are valid)
                    if not step_was_skipped and params_with_grad:
                        grad_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm)
                        grad_norm_item = grad_norm.item() if torch.isfinite(grad_norm) else float('inf')
                        if not np.isfinite(grad_norm_item):
                             logger.warning(f"Rank {self.rank}: Grad norm became non-finite ({grad_norm_item}) AFTER clipping. Skipping optimizer step.")
                             step_was_skipped = True
                             self.optimizer.zero_grad(set_to_none=True) # Zero grads if clip results in NaN/Inf norm
                             # Skip to finally block

                        # Record Gradient Stats *before* potential step skip by scaler
                        if self.has_grad_stats:
                            clipped = grad_norm_item > self.max_grad_norm and np.isfinite(grad_norm_item)
                            clip_ratio = self.max_grad_norm / grad_norm_item if clipped and grad_norm_item > 1e-6 else None
                            self.optimizer.gradient_stats.record_gradient(grad_norm_item, clipped, clip_ratio)
                    elif not step_was_skipped: # No params with grads found
                         grad_norm_item = 0.0
                         if self.has_grad_stats: self.optimizer.gradient_stats.record_gradient(0.0, False, None)


                    # 4. Pass loss to Q-controller (if step not skipped so far)
                    if not step_was_skipped and hasattr(self.optimizer, 'set_current_loss'):
                        loss_to_set = avg_loss_cycle if np.isfinite(avg_loss_cycle) else None
                        self.optimizer.set_current_loss(loss_to_set)

                    # 5. Optimizer Step (using scaler, only if step not skipped so far)
                    if not step_was_skipped:
                        found_inf = self.scaler.step(self.optimizer) # Returns found_inf_per_device, None if no inf
                        # scaler.step internally checks for inf/nan in grads and skips if found
                        if found_inf is not None: # step checks for infs/nans in scaled grads
                            logger.warning(f"Rank {self.rank}: Optimizer step skipped BY SCALER due to invalid gradients detected by scaler.")
                            step_was_skipped = True
                            scaler_skipped_step = True
                        else:
                            optimizer_step_successful = True # Mark step as successful if scaler didn't skip

                except Exception as optim_phase_err:
                     # Catch errors during unscale, grad check, clip, or step
                     logger.error(f"Rank {self.rank}: Error during optimizer/scaler phase: {optim_phase_err}", exc_info=True)
                     step_was_skipped = True
                     # Ensure gradients are zeroed if an error happens here
                     try: self.optimizer.zero_grad(set_to_none=True)
                     except Exception: pass # Ignore errors during zeroing in this context

                finally:
                     # 6. Scaler Update (Always call update after step/skip attempt) (V3.6 Change)
                     try:
                         if self.scaler is not None and self.scaler.is_enabled(): # Ensure scaler exists and is enabled
                             self.scaler.update()
                             # Check scale validity after update (V3.8 Add check for scale < 1.0)
                             scale_after_update = self.scaler.get_scale()
                             if not np.isfinite(scale_after_update):
                                 logger.error(f"Rank {self.rank}: AMP Scaler became non-finite ({scale_after_update}) after update. Resetting scaler.")
                                 self.scaler = amp.GradScaler(enabled=self.use_amp) # Reinitialize scaler
                                 step_was_skipped = True # Mark step as skipped if scale became invalid
                             elif scale_after_update < 1.0: # Check if scale dropped below 1 (can happen)
                                 logger.warning(f"Rank {self.rank}: AMP Scaler dropped below 1.0 ({scale_after_update}). Resetting to default.")
                                 self.scaler = amp.GradScaler(enabled=self.use_amp) # Reinitialize scaler
                                 step_was_skipped = True # Mark step as skipped if scale became invalid


                     except RuntimeError as update_err:
                         # Catch potential "update() must be called after step()" - this shouldn't happen with the finally block
                         logger.error(f"Rank {self.rank}: Error during scaler.update(): {update_err}. May indicate logic error.")
                         step_was_skipped = True # Mark step as skipped if update fails
                     except Exception as final_update_ex:
                         logger.error(f"Rank {self.rank}: Unexpected error during scaler.update(): {final_update_ex}")
                         step_was_skipped = True # Mark step as skipped


                # 7. Zero Gradients **AFTER** step attempt and scaler update (V3.6 Change)
                if optimizer_step_successful:
                    self.optimizer.zero_grad(set_to_none=True) # Zero grads after successful step
                elif step_was_skipped and not scaler_skipped_step:
                    # If skipped due to reasons BEFORE scaler.step (invalid grad, clip error, other exception)
                    # and grads weren't already zeroed in the specific error handling, ensure they are zeroed.
                    self.optimizer.zero_grad(set_to_none=True)
                # If scaler_skipped_step is True, scaler handles grad state, no need to zero here.

                # --- Logging & Counters ---
                if should_optimizer_step: # Log only when a step was attempted
                    if optimizer_step_successful: # Use the success flag (V3.6 Change)
                        # Increment counters only if step was successful
                        optimizer_steps_in_epoch += 1
                        self.global_step += 1
                        if not disable_tqdm: batch_iterator.update(1) # Update tqdm progress for optimizer steps

                    # Log info on main process periodically or if step was skipped
                    if self.is_main and (self.global_step % self.log_interval == 0 or step_was_skipped):
                        lr_val = self.optimizer.param_groups[0]['lr']; mom_val = self.optimizer.param_groups[0]['momentum']
                        # V3.9: Added scaler.is_enabled() check
                        current_amp_scale = self.scaler.get_scale() if self.scaler.is_enabled() else -1.0 # Get current scale (float) or -1 if disabled
                        log_data = {"train/loss_step_avg": avg_loss_cycle if np.isfinite(avg_loss_cycle) else -1.0,
                                    "train/learning_rate": lr_val, "train/momentum": mom_val,
                                    "train/epoch": self.current_epoch + 1, "train/global_step": self.global_step,
                                    "train/grad_norm_clipped": grad_norm_item if np.isfinite(grad_norm_item) else -1.0,
                                    "train/amp_scale": current_amp_scale if np.isfinite(current_amp_scale) else -1.0, # Checked scale validity above
                                    "train/step_skipped": float(step_was_skipped),
                                    "train/scaler_skipped": float(scaler_skipped_step)}
                        if self.has_q_controller:
                            q_info = self.optimizer.get_q_info()
                            log_data.update({f"train/q_{k}": v for k, v in q_info.items() if k != 'last_action'})
                            if q_info.get("last_action"):
                               log_data["train/q_lr_scale"] = q_info["last_action"].get('lr_scale', 1.0)
                               log_data["train/q_mom_scale"] = q_info["last_action"].get('momentum_scale', 1.0)
                        # Use recorded step stats if available and step wasn't skipped *before* recording
                        if self.has_grad_stats and (optimizer_step_successful or (not step_was_skipped and not scaler_skipped_step)):
                            opt_stats = self.optimizer.gradient_stats.record_step(self.global_step)
                            log_data.update({f"train/opt/{k}": v for k, v in opt_stats.items()})

                        log_str = f"Step {self.global_step} | Ep {self.current_epoch + 1} | Loss(avg): {log_data['train/loss_step_avg']:.4f} | LR: {lr_val:.3e} | Mom: {mom_val:.3f} | GradNorm: {log_data['train/grad_norm_clipped']:.2f}"
                        if log_data.get('train/opt/clip_percentage', 0) > 0: log_str += f" | Clip%: {log_data['train/opt/clip_percentage']:.1f}"
                        if step_was_skipped: log_str += f" (Step Skipped{' by Scaler' if scaler_skipped_step else ''})"
                        if self.has_q_controller and log_data.get("train/q_lr_scale") is not None: log_str += f" | QScale(LR/M): {log_data['train/q_lr_scale']:.2f}/{log_data['train/q_mom_scale']:.2f}"

                        logger.info(log_str)
                        if self.wandb_enabled: wandb.log(log_data, step=self.global_step)
                        if not disable_tqdm and isinstance(batch_iterator, tqdm): # Check if tqdm object
                            batch_iterator.set_postfix({"Loss": f"{log_data['train/loss_step_avg']:.3f}", "LR": f"{lr_val:.3e}", "Grad": f"{log_data['train/grad_norm_clipped']:.2f}"})

                    # --- Checkpointing ---
                    if self.is_main and self.save_interval > 0 and self.global_step % self.save_interval == 0 and optimizer_step_successful: # V3.6 Change
                        self._save_checkpoint(is_intermediate=True)

                # Reset micro-step counter moved to after accumulation cycle block

        # --- End of Epoch ---
        # Calculate average loss per *successful* optimizer step
        avg_epoch_loss_per_step = epoch_loss / (optimizer_steps_in_epoch * self.grad_accum_steps) if optimizer_steps_in_epoch > 0 else float('inf')
        if self.is_main:
            logger.info(f"Epoch {self.current_epoch + 1} finished. Avg Train Loss (per micro-batch): {avg_epoch_loss_per_step:.4f} ({optimizer_steps_in_epoch} successful optim steps)")
            if self.wandb_enabled: wandb.log({"train/epoch_loss_avg_per_micro_batch": avg_epoch_loss_per_step, "epoch": self.current_epoch + 1}, step=self.global_step)

        if hasattr(batch_iterator, 'close'): batch_iterator.close()
        return avg_epoch_loss_per_step

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """ Validates the model using standard autoregressive loss. """
        if self.val_loader is None: return {}
        if not self.is_main: return {} # Only validate on main process

        self.model.eval()
        total_loss = 0.0; num_batches = 0
        approx_total_val_batches = -1
        try:
            num_samples_total_val = len(self.val_loader.dataset)
            # Correctly get batch size for val loader
            val_loader_batch_size = self.val_loader.batch_size if self.val_loader.batch_size is not None else 1
            # Estimate total batches, don't divide by world_size as main process runs through all val data
            if val_loader_batch_size > 0 and num_samples_total_val > 0:
                approx_total_val_batches = (num_samples_total_val + val_loader_batch_size - 1) // val_loader_batch_size
                if approx_total_val_batches <= 0: approx_total_val_batches = -1
            else:
                 approx_total_val_batches = -1
            logger.debug(f"Estimated total validation batches: {approx_total_val_batches}")
        except TypeError:
             approx_total_val_batches = None # Cannot estimate length

        val_iterator = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch + 1}", disable=not self.is_main, total=approx_total_val_batches, unit="batch")

        for batch_data in val_iterator:
            try:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    context, targets = batch_data
                else: logger.warning(f"Skipping unexpected validation batch format: {type(batch_data)}"); continue

                context = context.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size, ctx_len = context.shape
                if batch_size == 0 or ctx_len == 0: continue

                decoder_input_seq = context

                with torch.no_grad(), amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # Use model.module if DDP wrapped, otherwise use self.model
                    model_to_eval = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
                    logits = model_to_eval(byte_seq=context, target_byte_seq=decoder_input_seq)

                    if logits.shape[0] != batch_size or logits.shape[1] != ctx_len or logits.shape[2] != 256:
                        logger.warning(f"Validation logits shape mismatch. Skip.")
                        continue
                    if not torch.isfinite(logits).all():
                        logger.warning(f"NaN/Inf detected in validation logits. Skip loss.")
                        continue

                    # Compute loss (no smoothing for validation)
                    loss = model_to_eval.compute_loss(logits, targets, smoothing=0.0)

                loss_item = loss.item()
                if np.isfinite(loss_item):
                     total_loss += loss_item
                     num_batches += 1
                else: logger.warning(f"Non-finite validation loss encountered: {loss_item}")

            except Exception as val_ex: logger.error(f"Validation Error: {val_ex}", exc_info=True); continue

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = float('inf')
        if np.isfinite(avg_loss):
            try:
                clamped_loss = min(avg_loss, 700) # Prevent overflow
                perplexity = math.exp(clamped_loss)
            except (OverflowError, ValueError): perplexity = float('inf')

        metrics = {'val_loss': avg_loss, 'val_perplexity': perplexity}
        self.last_val_metrics = metrics

        logger.info(f"Validation Epoch {self.current_epoch + 1} | Loss: {metrics['val_loss']:.4f} | Perplexity: {metrics['val_perplexity']:.2f}")
        if self.wandb_enabled: wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": self.current_epoch + 1}, step=self.global_step)

        if hasattr(val_iterator, 'close'): val_iterator.close()
        return metrics

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict] = None):
        """Saves model checkpoint (only called from main process)."""
        if not self.is_main: return
        filename_prefix = "checkpoint"
        if is_intermediate: state_indicator = f"step_{self.global_step}"
        else: state_indicator = f"epoch_{self.current_epoch+1}_final"
        if metrics and 'val_loss' in metrics and np.isfinite(metrics['val_loss']):
            state_indicator += f"_loss{metrics['val_loss']:.3f}"
        filename = f"{filename_prefix}_{state_indicator}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        model_state = self.model.module.state_dict() if isinstance(self.model, DistributedDataParallel) else self.model.state_dict()
        checkpoint = {
            'epoch': self.current_epoch, 'global_step': self.global_step,
            'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'metrics': metrics if metrics is not None else self.last_val_metrics,
            'amp_enabled': self.use_amp, 'args': getattr(self, 'args', None)
        }
        if self.has_q_controller:
            q_state_data = {
                'q_table': self.optimizer.q_controller.q_table, 'epsilon': self.optimizer.q_controller.epsilon,
                'access_count': self.optimizer.q_controller.q_table_access_count,
                'creation_time': self.optimizer.q_controller.q_table_creation_time
            }
            checkpoint['q_controller_state'] = q_state_data

        try:
            temp_filepath = filepath + ".tmp"
            torch.save(checkpoint, temp_filepath)
            os.replace(temp_filepath, filepath)
            logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}", exc_info=True)
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError: pass

    def load_checkpoint(self, filepath: str):
        """Loads checkpoint from filepath."""
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return 0

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            model_to_load = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible_keys.missing_keys: logger.warning(f"Missing keys loading model: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys: logger.warning(f"Unexpected keys loading model: {incompatible_keys.unexpected_keys}")

            if 'optimizer_state_dict' in checkpoint:
                 try: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 except Exception as optim_ex: logger.warning(f"Could not load optimizer state: {optim_ex}. Reset.")
            else: logger.warning("Optimizer state not in checkpoint. Using fresh state.")

            saved_amp_enabled = checkpoint.get('amp_enabled', False)
            if self.use_amp:
                if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                     if saved_amp_enabled:
                         try: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                         except Exception as scaler_ex: logger.warning(f"Could not load AMP scaler state: {scaler_ex}. Reset.")
                     else: logger.warning("Ckpt no AMP, but AMP on now. Scaler not loaded.")
                else:
                    logger.warning("AMP scaler state not found in checkpoint. Using fresh state.")
                    # Re-initialize scaler if loading failed or state missing
                    self.scaler = amp.GradScaler(enabled=self.use_amp)
            elif saved_amp_enabled: logger.warning("Ckpt has AMP state, but AMP off now. Ignored.")

            start_epoch = checkpoint.get('epoch', -1) + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.current_epoch = checkpoint.get('epoch', 0)
            self.last_val_metrics = checkpoint.get('metrics')

            if self.has_q_controller and 'q_controller_state' in checkpoint:
                q_state = checkpoint['q_controller_state']
                try:
                    self.optimizer.q_controller.q_table = q_state.get('q_table', {})
                    self.optimizer.q_controller.epsilon = q_state.get('epsilon', self.optimizer.q_controller.epsilon)
                    self.optimizer.q_controller.q_table_access_count = q_state.get('access_count', {})
                    self.optimizer.q_controller.q_table_creation_time = q_state.get('creation_time', {})
                    logger.info("Q-Controller state loaded.")
                except Exception as q_load_err: logger.warning(f"Could not load Q-Controller state: {q_load_err}. Reset.")
            elif self.has_q_controller: logger.warning("Q-Controller active, state not in ckpt. Using fresh state.")

            logger.info(f"Loaded checkpoint '{filepath}'. Resuming from Epoch {start_epoch}, Global Step {self.global_step}")
            return start_epoch
        except FileNotFoundError: logger.error(f"Checkpoint file not found during load: {filepath}"); return 0
        except Exception as e: logger.error(f"Failed loading checkpoint '{filepath}': {e}", exc_info=True); return 0

    def train(self, epochs: int, start_epoch: int = 0):
        """ Main training loop over epochs. """
        self.current_epoch = start_epoch
        logger.info(f"Starting training from epoch {start_epoch + 1}/{epochs} (Global Step: {self.global_step}).")
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            logger.info(f"--- Starting Epoch {epoch + 1}/{epochs} ---")

            # Set epoch for distributed samplers AND dataset internal shuffling
            if self.world_size > 1:
                 if hasattr(self.train_loader.sampler, 'set_epoch'):
                     self.train_loader.sampler.set_epoch(epoch)
                     logger.debug(f"Set train sampler epoch {epoch} rank {self.rank}")
                 if self.val_loader and hasattr(self.val_loader.sampler, 'set_epoch'):
                     self.val_loader.sampler.set_epoch(epoch)
                     logger.debug(f"Set val sampler epoch {epoch} rank {self.rank}")
            # Set epoch in dataset for internal shuffling consistency
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader.dataset, 'set_epoch'):
                self.val_loader.dataset.set_epoch(epoch)


            avg_train_loss = self._train_epoch()
            val_metrics = self._validate()

            if self.is_main: self._save_checkpoint(is_intermediate=False, metrics=val_metrics)
            if self.world_size > 1: torch.distributed.barrier() # Sync after epoch end

        logger.info("Training finished.")


# =====================================================================
# Argument Parsing and Main Execution Logic (V3.9 - Stability Fixes)
# =====================================================================

def setup_distributed(local_rank):
    """Initializes distributed training environment using environment variables."""
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ws = 1; rk = 0; is_distributed = False
        if torch.cuda.is_available(): logger.info(f"CUDA available. Using device: {torch.cuda.get_device_name(device)}")
        else: logger.info("CUDA not available. Using CPU.")
    else:
        if not torch.cuda.is_available(): logger.error("DDP requested but CUDA not available."); sys.exit(1)
        if torch.cuda.device_count() <= local_rank: logger.error(f"Invalid local_rank {local_rank}. GPUs available: {torch.cuda.device_count()}."); sys.exit(1)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        required_env_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        if not all(env_var in os.environ for env_var in required_env_vars):
            logger.warning(f"DDP env vars ({required_env_vars}) not fully set. Use 'torchrun'. Trying init...")
        try:
            init_process_group(backend="nccl", init_method="env://")
            ws = get_world_size(); rk = get_rank(); is_distributed = True
            logger.info(f"DDP Initialized via env:// | Rank: {rk}/{ws} | Device: {device} ({torch.cuda.get_device_name(device)})")
            torch.distributed.barrier()
        except Exception as e:
            logger.error(f"DDP init failed: {e}", exc_info=True)
            if is_initialized(): destroy_process_group()
            sys.exit(1)
    return is_distributed, device, rk, ws

def parse_arguments():
    parser = argparse.ArgumentParser(description="Integrated HyperHAKMEM Model Training Script (V3.9)") # Version Bump
    # --- Data Args ---
    dgrp = parser.add_argument_group('Data Configuration')
    dgrp.add_argument("--data_path", type=str, required=True, help="Path to train data (.npy file).")
    dgrp.add_argument("--val_data_path", type=str, default=None, help="Path to validation data (.npy, optional).")
    dgrp.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of training data to use (0.0 to 1.0).")
    dgrp.add_argument("--context_window", type=int, default=256, help="Sequence length for training and decoder PE.")

    # --- Model Args ---
    mgrp = parser.add_argument_group('Model Architecture')
    mgrp.add_argument("--local_hidden_size", type=int, default=256, help="Hidden size for Local Encoder/Decoder.")
    mgrp.add_argument("--complex_dim", type=int, default=512, help="Dimension of the complex tangent space.")
    mgrp.add_argument("--num_complex_layers", type=int, default=8, help="Number of complex interference layers.")
    mgrp.add_argument("--num_complex_heads", type=int, default=8, help="Number of heads in complex attention.")
    mgrp.add_argument("--decoder_memory_dim", type=int, default=768, help="Dimension of the final memory fed to the decoder.")
    mgrp.add_argument("--n_gram_sizes", type=int, nargs='+', default=[3, 4], help="N-gram sizes for Local Encoder features.")
    mgrp.add_argument("--n_gram_vocab_size", type=int, default=30000, help="Vocabulary size for N-gram hashing.")
    mgrp.add_argument("--dropout", type=float, default=0.15, help="Dropout rate for various layers.")
    mgrp.add_argument("--no_entanglement", action="store_true", help="Disable head entanglement in complex layers.")
    mgrp.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Encoding (RoPE).")
    mgrp.add_argument("--projection_method", type=str, default="hakmem_enhanced", choices=["concat", "magnitude", "hakmem_enhanced"], help="Method for Complex-to-Real projection.")
    mgrp.add_argument("--no_hierarchical_decoder", action="store_true", help="Use flat (256 output) decoder head instead of hierarchical.")
    mgrp.add_argument("--hyperbolic_embedding_dim", type=int, default=384, help="Dimension of the hyperbolic space (and tangent space).")
    mgrp.add_argument("--curvature", type=float, default=0.8, help="Hyperbolic curvature 'c' (must be > 0).")
    mgrp.add_argument("--clipping_radius", type=float, default=1.0, help="Clipping radius within Poincare ball (should be <= 1.0).")

    # --- Training Args ---
    tgrp = parser.add_argument_group('Training Parameters')
    tgrp.add_argument("--batch_size", type=int, default=16, help="Global batch size (distributed across GPUs).")
    tgrp.add_argument("--learning_rate", type=float, default=8e-4, help="Base learning rate for the optimizer.")
    tgrp.add_argument("--epochs", type=int, default=12, help="Total number of training epochs.")
    tgrp.add_argument("--grad_accum_steps", type=int, default=2, help="Number of steps to accumulate gradients before optimizer step.")
    tgrp.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    tgrp.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization).")
    tgrp.add_argument("--checkpoint_dir", type=str, default="./integrated_checkpoints_v3.9", help="Directory to save checkpoints.") # Default dir updated
    tgrp.add_argument("--log_interval", type=int, default=10, help="Log training stats every N optimizer steps.")
    tgrp.add_argument("--save_interval", type=int, default=1000, help="Save intermediate checkpoint every N optimizer steps (0=disable).")
    tgrp.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")

    # --- Q-Learning Args ---
    qgrp = parser.add_argument_group('Optimizer Q-Learning Control')
    qgrp.add_argument("--q_learning_rate", type=float, default=0.01, help="Learning rate (alpha) for the Q-table updates.")
    qgrp.add_argument("--q_discount", type=float, default=0.95, help="Discount factor (gamma) for future rewards in Q-learning.")
    qgrp.add_argument("--q_epsilon", type=float, default=0.2, help="Initial epsilon for epsilon-greedy exploration in Q-learning.")
    qgrp.add_argument("--q_epsilon_decay", type=float, default=0.9998, help="Decay rate for epsilon.")
    qgrp.add_argument("--q_min_epsilon", type=float, default=0.01, help="Minimum value for epsilon.")

    # --- Misc Args ---
    msgrp = parser.add_argument_group('Miscellaneous')
    msgrp.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    msgrp.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    msgrp.add_argument("--wandb_project", type=str, default="bytropix-integrated-v3.9", help="WandB project name.") # Default project updated
    msgrp.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")
    msgrp.add_argument("--num_workers", type=int, default=2, help="Number of dataloader worker processes (set 0 for main process loading, avoids pickling issues).")
    msgrp.add_argument("--detect_anomaly", action="store_true", help="Enable torch.autograd.detect_anomaly() for debugging.")
    msgrp.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), help=argparse.SUPPRESS)

    args = parser.parse_args()
    # Recommend num_workers=0 on Windows if > 0 is used
    if platform.system() == "Windows" and args.num_workers > 0:
         logger.warning("Using num_workers > 0 on Windows can sometimes cause issues with multiprocessing/pickling. If errors occur, try setting --num_workers 0.")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args

# =====================================================================
# Main Execution Logic (V3.9)
# =====================================================================
def main():
    args = parse_arguments()

    # --- Initial Logging ---
    temp_is_main = args.local_rank == -1 or args.local_rank == 0
    log_level = logging.INFO if temp_is_main else logging.WARNING
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)
    logger.info("Initial logging configuration set.")

    # --- Distributed Setup ---
    ddp_active, device, rank, world_size = setup_distributed(args.local_rank)

    # --- Refined Logging ---
    am_main_process = is_main_process()
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.getLogger().setLevel(log_level)
    if am_main_process:
        log_filename = os.path.join(args.checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_r{rank}.log")
        try:
            # Use 'w' mode to overwrite old logs for clarity if script restarts
            fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
            log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
            fh.setFormatter(log_formatter)
            logging.getLogger().addHandler(fh)
            logger.info(f"File logging enabled: {log_filename}")
        except Exception as e: logger.error(f"Failed to configure file logging: {e}")

    logger.info("="*60 + f"\n--- Integrated HyperHAKMEM Run (V3.9 - Stability Fixes III) ---") # Updated Title
    logger.info(f"Rank: {rank}/{world_size} | Device: {device} | DDP Active: {ddp_active} | Is Main Process: {am_main_process}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"System Info: OS={platform.system()}/{platform.release()}, Python={sys.version.split()[0]}, Torch={torch.__version__}, CUDA Available={torch.cuda.is_available()}, CUDA Version={torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n" + "="*60)

    # --- Seeding ---
    seed = args.seed # Use base seed for dataset, add rank later for other things if needed
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + rank)
    logger.info(f"Set random seed to {seed + rank} for Rank {rank}")

    # --- WandB ---
    use_wandb = args.wandb and am_main_process and WANDB_AVAILABLE
    if use_wandb:
        try:
            cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in vars(args).items()}
            run_name = f"HAKMEMv3.9_H{args.local_hidden_size}_C{args.complex_dim}_L{args.num_complex_layers}_B{args.batch_size}_LR{args.learning_rate:.1e}" # Updated name
            wandb.init(project=args.wandb_project, config=cfg, name=run_name, resume="allow")
            logger.info(f"WandB Initialized: Project='{args.wandb_project}', Run Name='{run_name}'")
        except Exception as e: logger.warning(f"Wandb init failed: {e}. Disabling."); use_wandb = False

    # --- Datasets ---
    train_dataset = None
    val_dataset = None
    try:
        logger.info(f"Initializing training dataset: {args.data_path}")
        train_dataset = ByteIterableDataset(args.data_path, context_size=args.context_window, data_fraction=args.data_fraction)
        train_dataset.set_seed(seed) # Set base seed for dataset internal shuffling

        if args.val_data_path:
            if os.path.exists(args.val_data_path):
                logger.info(f"Initializing validation dataset: {args.val_data_path}")
                val_dataset = ByteIterableDataset(args.val_data_path, context_size=args.context_window, data_fraction=1.0)
                val_dataset.set_seed(seed) # Set base seed for val dataset too
            else: logger.warning(f"Validation data path specified but not found: {args.val_data_path}")
        else: logger.info("No validation dataset specified.")

    except Exception as e:
        logger.error(f"Fatal Error: Failed to initialize datasets: {e}", exc_info=True)
        if ddp_active: destroy_process_group()
        sys.exit(1)

    # --- DataLoaders ---
    if world_size <= 0: logger.error("World size <= 0. Exiting."); sys.exit(1)
    if args.batch_size % world_size != 0: logger.warning(f"Global BS {args.batch_size} not divisible by world size {world_size}.")
    batch_size_per_gpu = max(1, args.batch_size // world_size)
    effective_bs = batch_size_per_gpu * world_size * args.grad_accum_steps
    logger.info(f"Batch Config: Global Effective BS={effective_bs}, Per GPU Micro BS={batch_size_per_gpu}, Accum Steps={args.grad_accum_steps}")

    # Create Samplers for DDP. Note: shuffle=True for DistributedSampler shuffles the order of partitions among workers.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
    ) if ddp_active else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False # Usually don't shuffle or drop val
    ) if ddp_active and val_dataset else None

    # DataLoader: shuffle argument MUST be False when using a sampler (especially DistributedSampler)
    # The sampler handles distribution and shuffling order. Internal dataset shuffling handles item order per worker.
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              shuffle=False, # MUST BE FALSE when sampler is used
                              persistent_workers=(args.num_workers > 0) and (platform.system() != 'Windows') # Persistent workers often problematic on Windows
                              )
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False,
                            shuffle=False, # MUST BE FALSE when sampler is used
                            persistent_workers=(args.num_workers > 0) and (platform.system() != 'Windows')
                           ) if val_dataset else None

    # --- Model ---
    try:
        sig = inspect.signature(IntegratedHyperHAKMEMModel.__init__)
        model_args_names = [p.name for p in sig.parameters.values() if p.name != 'self']
        model_cfg = {k: v for k, v in vars(args).items() if k in model_args_names}
        model_cfg['sfin_use_entanglement'] = not args.no_entanglement
        model_cfg['sfin_use_rotary'] = not args.no_rope
        model_cfg['use_hierarchical_decoder'] = not args.no_hierarchical_decoder
        model_cfg['use_amp'] = not args.no_amp

        model = IntegratedHyperHAKMEMModel(**model_cfg).to(device)
        if am_main_process:
            tp = sum(p.numel() for p in model.parameters())
            ttp = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model Parameter Count: Total={tp:,}, Trainable={ttp:,}")
    except Exception as model_ex:
        logger.error(f"Fatal Error: Model initialization failed: {model_ex}", exc_info=True)
        if ddp_active: destroy_process_group()
        sys.exit(1)

    # --- DDP Wrapping ---
    if ddp_active:
        # find_unused_parameters=False assumes parameters are used consistently. Set True if errors occur.
        # Consider static_graph=True if model structure doesn't change, might improve performance.
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                        find_unused_parameters=False, static_graph=False)
        logger.info(f"Model wrapped with DDP on Rank {rank} (find_unused=False, static_graph=False).")
        # Barrier ensures all processes have DDP model before optimizer init uses model.parameters()
        torch.distributed.barrier()

    # --- Optimizer ---
    q_cfg = {"learning_rate": args.q_learning_rate, "discount": args.q_discount, "epsilon": args.q_epsilon,
             "epsilon_decay": args.q_epsilon_decay, "min_epsilon": args.q_min_epsilon}
    optimizer = HAKMEMEnhancedSGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                  weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
                                  q_learning_config=q_cfg, enable_flow=False)
    logger.info(f"Optimizer '{type(optimizer).__name__}' initialized Rank {rank}, LR={args.learning_rate}, WD={args.weight_decay}")

    # --- Trainer ---
    trainer = Trainer(model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader,
                      grad_accum_steps=args.grad_accum_steps, use_amp=(not args.no_amp), log_interval=args.log_interval,
                      save_interval=args.save_interval, checkpoint_dir=args.checkpoint_dir, wandb_enabled=use_wandb,
                      max_grad_norm=args.max_grad_norm, rank=rank, world_size=world_size,
                      detect_anomaly=args.detect_anomaly)
    trainer.args = args # Store args for checkpointing

    # --- Resume ---
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Attempting resume from checkpoint: {args.resume} Rank {rank}")
            start_epoch = trainer.load_checkpoint(args.resume)
            # Barrier ensures all processes load checkpoint before starting training
            if ddp_active: torch.distributed.barrier()
        else:
             logger.warning(f"Resume checkpoint not found: {args.resume}. Starting fresh.")
             # Barrier ensures all processes know ckpt wasn't loaded before proceeding
             if ddp_active: torch.distributed.barrier()
    else:
        logger.info("No checkpoint specified. Starting fresh.")
        if ddp_active: torch.distributed.barrier() # Ensure sync even when not resuming


    # --- Training ---
    save_final = False # Flag to ensure final save
    try:
        trainer.train(args.epochs, start_epoch=start_epoch)
        save_final = True # Mark for saving after successful completion
    except KeyboardInterrupt:
        logger.info(f"Training interrupted by user (KeyboardInterrupt) Rank {rank}.")
        save_final = True # Save progress on interrupt
    except Exception as train_ex:
        logger.error(f"Unhandled error during training Rank {rank}: {train_ex}", exc_info=True)
        save_final = True # Attempt to save progress even after error
    finally:
        # --- Final Actions ---
        # Save final checkpoint on main process if training finished or was interrupted/errored
        if save_final and am_main_process:
            logger.info("Attempting to save final checkpoint...")
            metrics = getattr(trainer, 'last_val_metrics', None) # Get last validation metrics
            trainer._save_checkpoint(is_intermediate=False, metrics=metrics)

        # Clean up DDP
        if ddp_active:
            destroy_process_group()
            logger.info(f"DDP group destroyed Rank {rank}.")

        # Finish WandB run
        if use_wandb and wandb is not None and wandb.run:
            wandb.finish()
            logger.info("WandB run finished.")
    logger.info(f"Script execution finished Rank {rank}.")

if __name__ == "__main__":
    main()
