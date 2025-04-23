# -*- coding: utf-8 -*-
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
from datetime import datetime, timedelta # Added timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque
import gc
import os
import socket
import platform
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp # Use torch.amp instead of torch.cuda.amp
from dataclasses import dataclass
import itertools
from tqdm import tqdm
import inspect
import string
import functools # Added for worker_init_fn fix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Setup logger - ensuring it's configured early
logger = logging.getLogger("IntegratedHyperHAKMEM")
# Basic config for initial setup and potential early errors
# This might be reconfigured later in main based on rank
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)


# =====================================================================
# Data Structures and Configuration Classes
# =====================================================================

@dataclass
class SamplerConfig:
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

class GradientStats:
    """Tracks gradient statistics like clipping counts and norms."""
    def __init__(self): self.reset()
    def reset(self):
        self.total_gradients = 0; self.clipped_gradients = 0
        self.max_gradient_norm = 0.0; self.sum_clip_ratios = 0.0
        self.non_finite_grads_in_step = 0 # Count non-finite grads encountered WITHIN a step calc
        self.step_stats = {}
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        """Records information about a gradient calculation."""
        if np.isfinite(original_norm):
            self.total_gradients += 1
            self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
            if clipped:
                self.clipped_gradients += 1
                self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)
        else:
            # Count non-finite gradients encountered during norm calculation
            self.non_finite_grads_in_step += 1
            # logger.debug(f"GradientStats skipping non-finite gradient norm: {original_norm}") # Less noisy
    def get_step_stats(self) -> dict:
        """Returns statistics aggregated since the last reset."""
        if self.total_gradients == 0 and self.non_finite_grads_in_step == 0:
            return {"gradients_clipped": 0, "total_gradients": 0, "clip_ratio_avg": 0.0, "max_gradient": 0.0, "clip_percentage": 0.0, "non_finite_grads": 0}
        total_attempts = self.total_gradients + self.non_finite_grads_in_step
        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100 if self.total_gradients > 0 else 0.0
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0
        return {"gradients_clipped": self.clipped_gradients,
                "total_gradients": self.total_gradients,
                "non_finite_grads": self.non_finite_grads_in_step,
                "clip_ratio_avg": avg_clip_ratio,
                "max_gradient": self.max_gradient_norm,
                "clip_percentage": clip_percentage}
    def record_step(self, step: int, skipped: bool = False) -> dict:
        """Records stats for a completed optimizer step and resets for the next."""
        stats = self.get_step_stats()
        stats['step_skipped'] = skipped # Add info about whether the step was skipped
        self.step_stats[step] = stats
        self.reset() # Reset after recording step stats
        return stats

# =====================================================================
# HAKMEM-Inspired Entropy Calculation Helper
# =====================================================================
class HAKMEMEntropyHelper:
    """Calculates Shannon entropy for byte sequences with caching."""
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_cache = {}
        self.max_cache_size = max_cache_size

    def _clean_cache(self):
        """Removes oldest entries if cache exceeds max size."""
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5) # Remove oldest 1/5th
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache: del self.entropy_cache[k]

    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...], List[int], bytes, torch.Tensor]) -> float:
        """Computes the Shannon entropy of a byte sequence."""
        cache_key = None
        byte_list = []

        # Convert input to a consistent format (list of ints) and prepare cache key
        if isinstance(byte_window, tuple):
            cache_key = byte_window
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
            if not byte_window: return 0.0
            byte_list = list(byte_window)
        elif isinstance(byte_window, list):
             if not byte_window: return 0.0
             cache_key = tuple(byte_window)
             if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
             byte_list = byte_window
        elif isinstance(byte_window, bytes):
            if not byte_window: return 0.0
            cache_key = byte_window # Bytes are hashable
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
            byte_list = list(byte_window)
        elif isinstance(byte_window, np.ndarray):
            if byte_window.size == 0: return 0.0
            byte_list = byte_window.tolist()
            cache_key = tuple(byte_list)
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        elif isinstance(byte_window, torch.Tensor):
            if byte_window.numel() == 0: return 0.0
            # Ensure tensor is on CPU and converted to bytes before list conversion
            byte_list = byte_window.cpu().byte().tolist()
            cache_key = tuple(byte_list)
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        else:
            # logger.warning(f"compute_entropy received unsupported type: {type(byte_window)}. Returning 0.") # Less noisy
            return 0.0

        # Actual entropy calculation
        try:
            if not byte_list: return 0.0
            # Use np.bincount for efficient frequency counting of byte values (0-255)
            byte_counts = np.bincount(np.array(byte_list, dtype=np.uint8), minlength=256)
            total_bytes = byte_counts.sum()
            if total_bytes == 0: return 0.0

            # Calculate probabilities only for bytes that actually appear
            probs = byte_counts[byte_counts > 0] / total_bytes
            # Shannon entropy formula: -sum(p * log2(p))
            entropy = float(-np.sum(probs * np.log2(probs + 1e-9))) # Add epsilon for numerical stability
            result = max(0.0, entropy) # Ensure entropy is non-negative

            # Cache the result if a key was generated
            if cache_key is not None:
                self.entropy_cache[cache_key] = result
                self._clean_cache() # Maintain cache size limit
            return result
        except Exception as e:
            logger.warning(f"Error during entropy calculation for window (size {len(byte_list)}): {e}", exc_info=False)
            return 0.0 # Return 0 entropy on error

# =====================================================================
# HAKMEM Babylon Index (Patching - Word/Punctuation Based)
# =====================================================================
class HAKMEMBabylonIndex:
    """Splits byte sequences into 'word' and 'delimiter' patches based on text decoding."""
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_helper = HAKMEMEntropyHelper(max_cache_size)
        self.whitespace_chars = set(string.whitespace)
        self.punctuation_chars = set(string.punctuation)
        logger.info("HAKMEMBabylonIndex initialized (Word/Punctuation Patching with Entropy).")

    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Creates patches from a byte tensor, calculating entropy for each."""
        if byte_seq_tensor.numel() == 0: return []
        # Ensure input is a 1D tensor
        if byte_seq_tensor.dim() != 1:
            # logger.warning(f"BabylonIndex expected 1D tensor, got {byte_seq_tensor.shape}. Flattening.") # Less noisy
            byte_seq_tensor = byte_seq_tensor.flatten()
        if byte_seq_tensor.numel() == 0: return []

        device = byte_seq_tensor.device
        try:
            # Decode byte tensor to string for splitting (best effort)
            text = byte_seq_tensor.cpu().numpy().tobytes().decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error decoding byte tensor (shape {byte_seq_tensor.shape}, numel {byte_seq_tensor.numel()}) to string: {e}. Returning no patches.")
            return []

        patches_with_entropy = []
        current_patch_start = 0
        in_word = False # Track if currently inside a word or delimiter sequence

        for i, char in enumerate(text):
            is_delimiter = char in self.whitespace_chars or char in self.punctuation_chars

            if is_delimiter:
                # If we were in a word, process the word patch
                if in_word:
                    word_str = text[current_patch_start:i]
                    try:
                        # Re-encode the word substring back to bytes
                        word_bytes = torch.tensor(list(word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                        if word_bytes.numel() > 0:
                            entropy = self.entropy_helper.compute_entropy(word_bytes)
                            patches_with_entropy.append((word_bytes, entropy))
                    except Exception as enc_e:
                        logger.warning(f"Error encoding word patch '{word_str[:20]}...': {enc_e}")
                    in_word = False # No longer in a word

                # Process the current delimiter character as its own patch
                try:
                    delim_bytes = torch.tensor(list(char.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                    if delim_bytes.numel() > 0:
                         entropy = self.entropy_helper.compute_entropy(delim_bytes)
                         patches_with_entropy.append((delim_bytes, entropy))
                except Exception as enc_e:
                    logger.warning(f"Error encoding delimiter patch '{char}': {enc_e}")

                current_patch_start = i + 1 # Move start index past the delimiter
            else: # Character is part of a word
                if not in_word:
                    # Start of a new word
                    in_word = True
                    current_patch_start = i # Mark the start of the word

        # Handle any trailing word after the loop finishes
        if in_word and current_patch_start < len(text):
            trailing_word_str = text[current_patch_start:]
            try:
                trailing_word_bytes = torch.tensor(list(trailing_word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                if trailing_word_bytes.numel() > 0:
                    entropy = self.entropy_helper.compute_entropy(trailing_word_bytes)
                    patches_with_entropy.append((trailing_word_bytes, entropy))
            except Exception as enc_e:
                logger.warning(f"Error encoding trailing word patch '{trailing_word_str[:20]}...': {enc_e}")

        # Filter out any potentially empty patches that might have slipped through
        patches_with_entropy = [(p, e) for p, e in patches_with_entropy if p.numel() > 0]
        return patches_with_entropy

    @torch.no_grad()
    def reset_context(self):
        """Resets the internal entropy cache."""
        self.entropy_helper.entropy_cache = {}
        logger.debug("HAKMEMBabylonIndex context (entropy cache) reset.")

# =====================================================================
# HAKMEM-Enhanced Cross Attention Block
# =====================================================================
class HAKMEMCrossAttentionBlock(nn.Module):
    """A standard cross-attention block with LayerNorm and optional Flash Attention."""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size <= 0: raise ValueError("hidden_size must be positive")
        # Ensure num_heads is valid for hidden_size
        if num_heads <= 0 : num_heads = max(1, hidden_size // 64) # Default if invalid
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads:
            num_heads = 1 # Fallback if no divisor found
            logger.warning(f"Could not find valid head count for hidden_size {hidden_size}. Using 1 head.")
        elif hidden_size % num_heads != 0:
            num_heads = valid_heads[0] # Pick the largest valid divisor <= original request
            logger.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size} in CrossAttention.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(max(1, self.head_dim)) # Attention scaling factor

        # LayerNorm for queries and keys/values separately
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)

        # Linear projections for Q, K, V and output
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize projection weights
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
             nn.init.xavier_uniform_(layer.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs cross-attention from queries to keys/values."""
        batch_size, num_queries, _ = queries.size()
        _, seq_len_kv, kv_hidden_size = keys_values.size()

        # Handle edge case of empty keys/values
        if seq_len_kv == 0: return torch.zeros_like(queries)
        # Ensure dimensions match
        if kv_hidden_size != self.hidden_size:
             raise ValueError(f"Keys/Values hidden size ({kv_hidden_size}) does not match block hidden size ({self.hidden_size})")

        # Normalize inputs
        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)

        # Project and reshape Q, K, V for multi-head attention
        # (B, Nq, H) -> (B, Nq, h, D) -> (B, h, Nq, D)
        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, Nkv, H) -> (B, Nkv, h, D) -> (B, h, Nkv, D)
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Prepare attention mask for scaled_dot_product_attention format (B, h, Nq, Nkv)
        attn_mask_sdpa = None
        if attention_mask is not None:
            # Ensure mask is boolean type
            mask_dtype = torch.bool if attention_mask.dtype != torch.bool else attention_mask.dtype
            # Add head and query dimensions if necessary
            if attention_mask.dim() == 2: # (B, Nkv) -> (B, 1, 1, Nkv)
                attn_mask_sdpa = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=mask_dtype)
            elif attention_mask.dim() == 3: # (B, Nq, Nkv) -> (B, 1, Nq, Nkv)
                 attn_mask_sdpa = attention_mask.unsqueeze(1).to(dtype=mask_dtype)
            elif attention_mask.dim() == 4: # (B, h, Nq, Nkv)
                 attn_mask_sdpa = attention_mask.to(dtype=mask_dtype)
            else: logger.warning(f"Unsupported attention mask shape {attention_mask.shape}. Ignoring mask.")

            # Validate mask shape compatibility before use
            if attn_mask_sdpa is not None:
                 attn_mask_sdpa = attn_mask_sdpa.to(device=queries.device)
                 expected_shape = (batch_size, self.num_heads, num_queries, seq_len_kv)
                 try:
                     # Attempt a dummy broadcast to check compatibility
                     torch.broadcast_shapes(attn_mask_sdpa.shape, expected_shape)
                 except RuntimeError:
                      logger.warning(f"Mask shape {attn_mask_sdpa.shape} not broadcastable to target [B, h, Nq, Nkv] {expected_shape}. Ignoring mask.")
                      attn_mask_sdpa = None

        # Use Flash Attention (scaled_dot_product_attention) if available
        # Note: scaled_dot_product_attention expects bool mask where True means "mask out"
        use_flash = hasattr(F, 'scaled_dot_product_attention')
        output = None

        if use_flash:
             try:
                  # Ensure mask is boolean if provided
                  # Ensure mask is True where values should be IGNORED (masked out).
                  # Common mistake: padding mask is often True where values EXIST.
                  # The input `attention_mask` here should follow the convention: True == MASK OUT.
                  sdpa_mask = attn_mask_sdpa.bool() if attn_mask_sdpa is not None else None

                  output = F.scaled_dot_product_attention(
                      q, k, v, attn_mask=sdpa_mask,
                      dropout_p=self.dropout.p if self.training else 0.0,
                      is_causal=False # Cross-attention is not causal by default
                  )
                  # Check for NaNs/Infs after Flash Attention
                  if not torch.isfinite(output).all(): raise ValueError("Flash Attention produced NaN/Inf")
             except Exception as e:
                  logger.warning(f"Flash Attention failed: {e}. Falling back to manual attention.", exc_info=False)
                  use_flash = False; output = None

        # Manual attention implementation (fallback)
        if output is None:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask_sdpa is not None:
                 try:
                     # Ensure mask is correctly broadcast and applied
                     # True in mask means mask out -> fill with -inf
                     scores = scores.masked_fill(attn_mask_sdpa, float('-inf'))
                 except RuntimeError as e: logger.error(f"Mask fill error. Scores: {scores.shape}, Mask: {attn_mask_sdpa.shape}. Error: {e}")
            # Clamp scores to prevent extreme values before softmax
            scores = torch.clamp(scores, min=-30.0, max=30.0)
            attn_probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype) # Use float32 for softmax stability
            attn_probs = torch.nan_to_num(attn_probs) # Replace potential NaNs from exp(-inf)
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v)

        # Reshape output back to (B, Nq, H)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)

        # Final check for stability
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in HAKMEMCrossAttentionBlock output. Replacing with zeros.")
            output = torch.nan_to_num(output)
        return output


# =====================================================================
# HAKMEM-Enhanced Local Encoder
# =====================================================================
class HAKMEMLocalEncoder(nn.Module):
    """Encodes individual patches using byte embeddings, optional N-grams, and a Transformer."""
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1,
                 n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Local Encoder hidden_size must be positive.")
        self.hidden_size=hidden_size

        # Embeddings for individual bytes
        self.byte_embeddings=nn.Embedding(256, hidden_size); nn.init.normal_(self.byte_embeddings.weight, std=1.0/math.sqrt(hidden_size))

        # Setup N-gram features if enabled
        self.n_gram_sizes = sorted(list(set(s for s in n_gram_sizes if isinstance(s, int) and s > 0)))
        self.n_gram_vocab_size = n_gram_vocab_size
        self.n_gram_embeddings = None
        if self.n_gram_sizes:
            if n_gram_vocab_size <= 0:
                logger.warning("n_gram_vocab_size <= 0, disabling N-gram features in LocalEncoder.")
                self.n_gram_sizes = []
            else:
                # Create separate embedding layers for each N-gram size
                self.n_gram_embeddings=nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes})
                for emb in self.n_gram_embeddings.values(): nn.init.normal_(emb.weight, std=0.02)
                logger.info(f"HAKMEMLocalEncoder using N-grams: {self.n_gram_sizes} with vocab size {n_gram_vocab_size}")
                # Simple hash multipliers (could be improved, e.g., random primes)
                # Precompute multipliers for hashing N-grams
                self.hash_multipliers = { n: torch.tensor([self._get_prime(n * 10 + i + 1) for i in range(n)], dtype=torch.long) for n in self.n_gram_sizes }

        else:
            logger.info("HAKMEMLocalEncoder: N-gram features disabled.")

        # Adjust transformer heads if needed
        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads:
             logger.warning(f"HAKMEMLocalEncoder adjusted Transformer heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        # Standard Transformer Encoder
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True # Use norm_first for stability
        )
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention to pool patch representations into a single vector per patch
        self.patch_pooling_attention=HAKMEMCrossAttentionBlock(hidden_size, num_heads, dropout)
        # Learnable query vector for the pooling attention
        # Initialize small to avoid early instability
        self.patch_query=nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)

        self.norm=nn.LayerNorm(hidden_size, eps=1e-6) # Final normalization
        self.dropout=nn.Dropout(dropout)

    def _get_prime(self, n):
        """Helper to find the smallest prime >= n."""
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
        """Computes rolling hashes for N-grams within a patch."""
        patch_len = patch_byte_sequence.size(0)
        device = patch_byte_sequence.device
        if patch_len < n: return torch.empty(0, dtype=torch.long, device=device) # Not enough bytes for N-gram

        # Create sliding windows of size n
        # unfold: (dim, size, step) -> creates views of size 'size' along 'dim' with 'step'
        # Ensure input tensor is on the correct device before unfold
        windows = patch_byte_sequence.long().to(device).unsqueeze(0).unfold(dimension=1, size=n, step=1) # Shape: (1, num_windows, n)

        # Get precomputed multipliers for this N
        multipliers = self.hash_multipliers.get(n)
        if multipliers is None: # Fallback if not precomputed (shouldn't happen)
            multipliers = torch.tensor([31]*n, device=device, dtype=torch.long)
        else: multipliers = multipliers.to(device=device, dtype=torch.long)

        multipliers = multipliers.view(1, 1, n) # Reshape for broadcasting: (1, 1, n)
        # Simple weighted sum hash: sum(byte_value * multiplier)
        # Ensure windows is long type for multiplication with long multipliers
        hashes = (windows.long() * multipliers).sum(dim=-1)
        # Modulo by vocab size to get embedding index
        return (hashes % self.n_gram_vocab_size).squeeze(0) # Shape: (num_windows,)

    def forward(self, patches_with_entropy: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Encodes a list of patches and returns their combined representations."""
        if not patches_with_entropy:
            # Handle case where no patches were generated for an input sequence
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            # logger.debug("LocalEncoder received no patches.") # Less noisy
            # Return empty tensor with correct batch and hidden dimensions, but 0 sequence length
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        device = patches_with_entropy[0][0].device # Get device from the first patch
        model_dtype = next(self.parameters()).dtype # Get expected dtype from model parameters

        patch_representations = []
        for patch_bytes, patch_entropy in patches_with_entropy:
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue # Skip empty patches

            # Ensure input is long for embedding lookup
            patch_bytes_long = patch_bytes.long()

            # 1. Get byte embeddings
            x = self.byte_embeddings(patch_bytes_long).to(model_dtype)
            x = x.unsqueeze(0) # Add batch dimension: (SeqLen, Hidden) -> (1, SeqLen, Hidden)

            # 2. Add N-gram features if enabled
            if self.n_gram_embeddings and self.n_gram_sizes:
                n_gram_features = torch.zeros_like(x) # Initialize features to zero
                for n in self.n_gram_sizes:
                    if patch_len >= n: # Check if patch is long enough for this N
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_long, n)
                        if n_gram_hashes.numel() > 0:
                            # Get embeddings for the hashes
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes).to(model_dtype)
                            ngram_embeds = ngram_embeds.unsqueeze(0) # Add batch dim: (NumWindows, H) -> (1, NumWin, H)

                            num_windows = ngram_embeds.size(1)
                            # N-gram window starting at index i corresponds to bytes i to i+n-1.
                            # We align the embedding to the *last* byte of the window (index i+n-1).
                            indices = torch.arange(n - 1, n - 1 + num_windows, device=device, dtype=torch.long)

                            # Prepare indices for scatter_add_
                            index_reshaped = indices.view(1, -1, 1)
                            index_expanded = index_reshaped.expand(1, num_windows, self.hidden_size) # (1, NumWin, Hidden)

                            # Check if indices are within the bounds of the original sequence length
                            if torch.max(indices) < patch_len:
                                n_gram_features.scatter_add_(1, index_expanded, ngram_embeds)
                            else:
                                # Handle cases where indices might go out of bounds (should be rare with correct logic)
                                logger.warning(f"N-gram index mismatch detected (max index {torch.max(indices)} >= patch_len {patch_len}). PatchLen={patch_len}, n={n}, NumWin={num_windows}")
                                # Filter embeddings and indices that are valid
                                valid_mask = indices < patch_len
                                valid_indices = indices[valid_mask]
                                valid_embeds = ngram_embeds[:, valid_mask, :]
                                if valid_indices.numel() > 0:
                                    index_reshaped_valid = valid_indices.view(1, -1, 1)
                                    index_expanded_valid = index_reshaped_valid.expand(1, valid_indices.size(0), self.hidden_size)
                                    n_gram_features.scatter_add_(1, index_expanded_valid, valid_embeds)
                # Add N-gram features to byte embeddings
                x = x + n_gram_features

            # Stability check before transformer
            if not torch.isfinite(x).all():
                logger.warning(f"NaN/Inf detected in LocalEncoder input before Transformer (patch size {patch_len}). Replacing with zeros.")
                x = torch.nan_to_num(x)

            # 3. Pass through Transformer Encoder
            x = self.dropout(x)
            processed_bytes = self.transformer(x) # Shape: (1, SeqLen, Hidden)

            # Stability check after transformer
            if not torch.isfinite(processed_bytes).all():
                logger.warning(f"NaN/Inf detected in LocalEncoder output after Transformer (patch size {patch_len}). Replacing with zeros.")
                processed_bytes = torch.nan_to_num(processed_bytes)

            # 4. Pool byte representations using cross-attention with a learned query
            batch_query = self.patch_query.expand(1, -1, -1).to(dtype=model_dtype) # Use the single learned query
            # Attend from the query to the processed byte sequence
            patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes) # Shape: (1, 1, Hidden)

            # Stability check after pooling
            if not torch.isfinite(patch_repr).all():
                logger.warning(f"NaN/Inf detected in LocalEncoder after patch pooling (patch size {patch_len}). Replacing with zeros.")
                patch_repr = torch.nan_to_num(patch_repr)

            patch_representations.append(patch_repr)

        if not patch_representations:
             # This case might occur if all input patches were empty or failed processing
             device = next(self.parameters()).device
             dtype = next(self.parameters()).dtype
             # logger.debug("No valid patch representations generated after encoding.") # Less noisy
             return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        # Concatenate representations of all patches along the sequence dimension
        patches_combined = torch.cat(patch_representations, dim=1) # Shape: (1, NumPatches, Hidden)

        # Final normalization and stability check
        normed_output = self.norm(patches_combined)
        if not torch.isfinite(normed_output).all():
             logger.warning("NaN/Inf detected in LocalEncoder final output after norm. Replacing with zeros.")
             normed_output = torch.nan_to_num(normed_output)

        return normed_output


# =====================================================================
# Hyperbolic Geometry Utilities (Updated)
# =====================================================================
class HyperbolicUtils:
    """Utility functions for Poincare ball model of hyperbolic geometry."""
    @staticmethod
    def poincare_clip(x: torch.Tensor, c: float, radius: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """Clips points to stay strictly inside the Poincare ball boundary."""
        if c <= 0: return x # Not hyperbolic if curvature is non-positive
        sqrt_c = math.sqrt(max(c, eps)) # Ensure c is positive for sqrt
        # Calculate the maximum allowed norm in the Euclidean space corresponding to the Poincare ball
        # Points on the boundary have norm 1/sqrt(c). We stay slightly inside.
        max_norm = (radius / sqrt_c) * (1.0 - eps)

        # Calculate squared norm and norm safely
        x_norm_sq = torch.sum(x.pow(2), dim=-1, keepdim=True)
        norm = torch.sqrt(torch.clamp(x_norm_sq, min=0) + eps) # Clamp before sqrt, add eps

        # If norm exceeds max_norm, scale it down
        cond = norm > max_norm
        # Calculate scaling factor: max_norm / norm if needs clipping, 1 otherwise
        scale_factor = torch.where(cond, max_norm / (norm + eps), torch.ones_like(norm))
        clipped_x = x * scale_factor
        # Stability check after clipping
        if not torch.isfinite(clipped_x).all():
             logger.warning("NaN/Inf detected *after* poincare_clip. Replacing.")
             clipped_x = torch.nan_to_num(clipped_x)
        return clipped_x

    @staticmethod
    def exponential_map(v: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
        """Maps a tangent vector v at the origin to the Poincare ball (exp_0^c(v))."""
        if c <= 0: return v # No mapping needed for Euclidean space
        # Calculate norm of the tangent vector v
        v_norm_sq = torch.sum(v.pow(2), dim=-1, keepdim=True)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=0) + eps)
        sqrt_c = math.sqrt(max(c, eps))

        # Formula: tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v
        # Use float32 for intermediate tanh calculation for stability
        tanh_term = torch.tanh(sqrt_c * v_norm.float()).to(v.dtype)
        # Handle ||v|| -> 0 case to avoid division by zero (limit is 1)
        lambda_v = torch.where(v_norm > eps, tanh_term / (sqrt_c * v_norm + eps), torch.ones_like(v_norm))

        mapped_v = lambda_v * v
        # Ensure the mapped point is strictly inside the ball
        return HyperbolicUtils.poincare_clip(mapped_v, c)

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
        """Maps a point y in the Poincare ball back to the tangent space at the origin (log_0^c(y))."""
        if c <= 0: return y # No mapping needed for Euclidean space
        # Ensure point y is strictly inside the ball before mapping
        y_clipped = HyperbolicUtils.poincare_clip(y, c)

        # Calculate norm of the point y
        y_norm_sq = torch.sum(y_clipped.pow(2), dim=-1, keepdim=True)
        y_norm = torch.sqrt(torch.clamp(y_norm_sq, min=0) + eps)
        sqrt_c = math.sqrt(max(c, eps))

        # Formula: atanh(sqrt(c) * ||y||) / (sqrt(c) * ||y||) * y
        # Clamp input to atanh to avoid domain errors slightly outside [-1, 1]
        arctanh_input = torch.clamp(sqrt_c * y_norm, min=-1.0 + eps, max=1.0 - eps)
        # Use float32 for stability
        atanh_term = torch.atanh(arctanh_input.float()).to(y.dtype)

        # Handle ||y|| -> 0 case (limit is 1)
        lambda_y = torch.where(y_norm > eps, atanh_term / (sqrt_c * y_norm + eps), torch.ones_like(y_norm))

        mapped_y = lambda_y * y_clipped
        if not torch.isfinite(mapped_y).all():
             logger.warning("NaN/Inf detected *in* logarithmic_map output. Replacing.")
             mapped_y = torch.nan_to_num(mapped_y)
        return mapped_y # Result is a vector in the tangent space (Euclidean)

    @staticmethod
    def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
        """Computes the hyperbolic distance between points x and y in the Poincare ball."""
        if c <= 0: # Fallback to Euclidean distance if curvature is non-positive
            logger.warning("Curvature <= 0 in poincare_distance. Using Euclidean distance.")
            return torch.norm(x - y, dim=-1)

        sqrt_c = math.sqrt(max(c, eps))

        # Ensure inputs are strictly inside the ball
        # Use a slightly tighter radius for clipping within distance calculation
        radius_clip = 0.999
        x_clipped = HyperbolicUtils.poincare_clip(x, c, radius=radius_clip, eps=eps)
        y_clipped = HyperbolicUtils.poincare_clip(y, c, radius=radius_clip, eps=eps)

        # Compute squared norms
        x_norm_sq = torch.sum(x_clipped.pow(2), dim=-1) # Shape: (...,)
        y_norm_sq = torch.sum(y_clipped.pow(2), dim=-1) # Shape: (...,)
        diff_norm_sq = torch.sum((x_clipped - y_clipped).pow(2), dim=-1) # Shape: (...,)

        # Denominator terms (should be > 0 due to clipping)
        denom_x = torch.clamp(1.0 - c * x_norm_sq, min=eps)
        denom_y = torch.clamp(1.0 - c * y_norm_sq, min=eps)

        # Argument for arccosh (must be >= 1)
        arcosh_arg = 1.0 + 2.0 * c * diff_norm_sq / (denom_x * denom_y + eps)
        arcosh_arg_clamped = torch.clamp(arcosh_arg, min=1.0 + eps) # Ensure >= 1

        # Compute distance using arccosh
        # Use float32 for stability if input is lower precision
        original_dtype = x.dtype
        distance = (1.0 / sqrt_c) * torch.acosh(arcosh_arg_clamped.float())
        distance = distance.to(original_dtype)

        # Final stability check
        if not torch.isfinite(distance).all():
            max_arg = arcosh_arg.max().item() if torch.isfinite(arcosh_arg).any() else 'NaN/Inf'
            logger.warning(f"NaN/Inf detected in poincare_distance. Replacing. Max Arg was {max_arg}")
            # Identify potential source
            if (denom_x <= 0).any() or (denom_y <= 0).any():
                 logger.warning("Denominator <= 0 detected in poincare_distance despite clipping.")
            distance = torch.nan_to_num(distance, nan=100.0, posinf=100.0) # Replace with large finite distance

        return distance


# =====================================================================
# Hyperbolic Attention Layer (Replaces Complex Layer)
# =====================================================================
class HyperbolicAttentionLayer(nn.Module):
    """Attention mechanism based on hyperbolic distances in the Poincare ball."""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1, curvature: float = 1.0, ffn_mult: int = 4):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads != 0:
            raise ValueError(f"Invalid dimensions for HyperbolicAttentionLayer: dim={dim}, heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_rate = dropout
        self.curvature = max(curvature, 1e-6) # Ensure positive curvature
        self.hyperbolic_utils = HyperbolicUtils()

        # --- Attention Components ---
        # LayerNorm before attention
        self.norm_attn = nn.LayerNorm(dim, eps=1e-6)
        # Projections for Q, K, V (operate in tangent space)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # Output projection after attention
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # Initialize weights
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
        # Learnable scaling factor for distance->score conversion exp(-gamma * dist)
        # Initialize gamma near 1, log space for positivity constraint
        self.log_gamma = nn.Parameter(torch.zeros(1, heads, 1, 1)) # Broadcasts over heads
        self.attn_dropout = nn.Dropout(dropout)

        # --- Feed-Forward Network Components ---
        # LayerNorm before FFN
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        ffn_hidden_dim = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout), # Dropout within FFN
            nn.Linear(ffn_hidden_dim, dim),
        )
        # Initialize FFN weights
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
        self.ffn_dropout = nn.Dropout(dropout) # Dropout after FFN + residual

        # logger.debug(f"HyperbolicAttentionLayer initialized: dim={dim}, heads={heads}, curvature={self.curvature:.3f}") # Less noisy

    def forward(self, x_tangent: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for hyperbolic attention block (Attention + FFN)."""
        batch_size, seq_len, _ = x_tangent.shape
        residual_attn = x_tangent # Store for first residual connection

        # --- 1. Pre-Normalization for Attention ---
        x_norm_attn = self.norm_attn(x_tangent)
        if not torch.isfinite(x_norm_attn).all():
             logger.warning("NaN/Inf in HyperbolicAttentionLayer input norm. Replacing.")
             x_norm_attn = torch.nan_to_num(x_norm_attn)

        # --- 2. Multi-Head Hyperbolic Attention ---
        # Project Q, K, V in tangent space
        q_t = self.q_proj(x_norm_attn)
        k_t = self.k_proj(x_norm_attn)
        v_t = self.v_proj(x_norm_attn)

        # Reshape for multi-head: (B, S, D) -> (B, h, S, D_head)
        q_t = q_t.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_t = k_t.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_t = v_t.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Map Q, K from tangent space to Poincare ball
        # Use model's dtype for potentially better precision if needed
        model_dtype = next(self.parameters()).dtype
        q_p = self.hyperbolic_utils.exponential_map(q_t.to(torch.float32), self.curvature).to(model_dtype)
        k_p = self.hyperbolic_utils.exponential_map(k_t.to(torch.float32), self.curvature).to(model_dtype)

        # Calculate pairwise hyperbolic distances
        # q_p: (B, h, Sq, D_head), k_p: (B, h, Sk, D_head)
        # Expand dims for broadcasting distance calc: (B, h, Sq, 1, D) vs (B, h, 1, Sk, D)
        q_p_expanded = q_p.unsqueeze(3)
        k_p_expanded = k_p.unsqueeze(2)
        # distance shape: (B, h, Sq, Sk)
        distances = self.hyperbolic_utils.poincare_distance(q_p_expanded, k_p_expanded, self.curvature)

        # Convert distances to similarity scores: exp(-gamma * distance)
        # Ensure gamma is positive and clamp for stability
        gamma = torch.exp(torch.clamp(self.log_gamma, -3.0, 3.0)) # Clamp log_gamma before exp
        scores = torch.exp(-gamma * distances)

        # --- Stability Check (Scores) ---
        if not torch.isfinite(scores).all():
            logger.warning("NaN/Inf detected in hyperbolic attention scores (pre-mask). Replacing with zeros.")
            scores = torch.nan_to_num(scores, nan=0.0)

        # Apply causal mask (standard for decoder-like self-attention)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x_tangent.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # Shape: (1, 1, Sq, Sk)

        # Combine with padding mask if provided
        # The input `attention_mask` should have True where values are INVALID/PADDED.
        final_mask = causal_mask # Start with causal mask
        if attention_mask is not None:
             padding_mask_unsqueezed = None
             # Reshape padding mask to (B, 1, 1, Sk) or (B, 1, Sq, Sk) or (B, h, Sq, Sk)
             if attention_mask.dim() == 2: padding_mask_unsqueezed = attention_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, Sk)
             elif attention_mask.dim() == 3: padding_mask_unsqueezed = attention_mask.unsqueeze(1) # (B, 1, Sq, Sk)
             elif attention_mask.dim() == 4: padding_mask_unsqueezed = attention_mask # (B, h, Sq, Sk)
             else: logger.warning(f"HyperbolicAttention unexpected mask shape {attention_mask.shape}. Ignoring padding mask.")

             if padding_mask_unsqueezed is not None:
                 padding_mask_bool = padding_mask_unsqueezed.to(device=final_mask.device, dtype=torch.bool)
                 try:
                     # OR combines masks: if causal OR padding says mask, then mask.
                     final_mask = final_mask | padding_mask_bool
                 except RuntimeError as e:
                     logger.error(f"Mask broadcast error: causal={causal_mask.shape}, padding={padding_mask_bool.shape}. Error: {e}")
                     final_mask = causal_mask # Fallback to only causal if combined mask failed

                 # Ensure final_mask is broadcastable to scores shape
                 try: torch.broadcast_shapes(final_mask.shape, scores.shape)
                 except RuntimeError:
                      logger.error(f"Combined mask shape {final_mask.shape} not compatible with scores {scores.shape}. Using causal only.")
                      final_mask = causal_mask # Fallback

        # Apply final mask to scores (set masked scores to 0, as exp(-large_dist) -> 0)
        # Note: We mask AFTER exp, setting to 0 is appropriate here.
        scores = scores.masked_fill(final_mask, 0.0)

        # --- Softmax and Value Combination ---
        # Normalize scores to get attention weights
        attn_weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
        attn_weights = torch.nan_to_num(attn_weights) # Handle potential division by zero if all scores were 0
        attn_weights = self.attn_dropout(attn_weights)

        # Combine V vectors (which are still in the tangent space)
        # (B, h, Sq, Sk) @ (B, h, Sk, D_head) -> (B, h, Sq, D_head)
        attn_output_t = torch.matmul(attn_weights.to(v_t.dtype), v_t) # Ensure dtype match

        # Reshape back to (B, S, D)
        attn_output_t = attn_output_t.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        # Final output projection
        attn_output = self.out_proj(attn_output_t)

        # --- 3. First Residual Connection & Dropout ---
        x = residual_attn + self.ffn_dropout(attn_output) # Use ffn_dropout here for consistency

        # --- 4. Pre-Normalization for FFN ---
        residual_ffn = x # Store for second residual connection
        x_norm_ffn = self.norm_ffn(x)
        if not torch.isfinite(x_norm_ffn).all():
             logger.warning("NaN/Inf in HyperbolicAttentionLayer FFN norm. Replacing.")
             x_norm_ffn = torch.nan_to_num(x_norm_ffn)

        # --- 5. Feed-Forward Network ---
        ffn_output = self.ffn(x_norm_ffn)

        # --- 6. Second Residual Connection & Dropout ---
        output = residual_ffn + self.ffn_dropout(ffn_output) # Apply dropout after residual add

        # --- Final Stability Check ---
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in final output of HyperbolicAttentionLayer. Replacing.")
            output = torch.nan_to_num(output)

        return output


# =====================================================================
# HAKMEM-Enhanced Local Decoder (Transformer Decoder)
# =====================================================================
class HAKMEMLocalDecoder(nn.Module):
    """Decodes byte sequences using embeddings, positional encodings, and cross-attention to memory."""
    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, use_hierarchical_pred: bool = True, max_decode_len: int = 2048):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Decoder hidden size must be positive.")
        self.hidden_size = hidden_size
        self.use_hierarchical = use_hierarchical_pred
        self.max_decode_len = max_decode_len

        # Adjust head count if necessary
        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads:
             logger.warning(f"HAKMEMLocalDecoder adjusted heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        # Embeddings for target bytes and positions
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))
        self.positional_encoding = nn.Embedding(max_decode_len, hidden_size)
        nn.init.normal_(self.positional_encoding.weight, std=0.02)

        # Projection layer to adapt memory dimension to decoder dimension
        self.memory_projection = nn.Sequential(
            nn.Linear(global_hidden_size, hidden_size * 2, bias=True), nn.GELU(), # Intermediate expansion
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=1e-6) # Normalize projected memory
        )
        for layer in self.memory_projection:
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

        # Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True # Norm first for stability
        )
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-6) # Final normalization after decoder layers
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=self.decoder_norm)

        # Output prediction head
        if self.use_hierarchical:
            # Predict 16 coarse classes (e.g., 0-15, 16-31, ...)
            self.byte_class_pred = nn.Linear(hidden_size, 16)
            # Predict specific byte within each class (16 heads, each predicting 16 possibilities)
            self.byte_specific_pred = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(16)])
            # Initialize prediction heads
            nn.init.normal_(self.byte_class_pred.weight, std=0.02); nn.init.zeros_(self.byte_class_pred.bias)
            # Smaller init for specific preds as they combine
            for layer in self.byte_specific_pred: nn.init.normal_(layer.weight, std=0.02 / math.sqrt(16)); nn.init.zeros_(layer.bias)
            logger.info("HAKMEMLocalDecoder using Hierarchical Prediction Head.")
        else:
            # Standard flat prediction head
            self.byte_pred = nn.Linear(hidden_size, 256)
            nn.init.normal_(self.byte_pred.weight, std=0.02); nn.init.zeros_(self.byte_pred.bias)
            logger.info("HAKMEMLocalDecoder using Flat Prediction Head.")
        self.dropout_embed = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
         """Generates a causal mask for self-attention."""
         # True values indicate positions to be masked
         mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
         return mask

    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs decoding step."""
        batch_size, tgt_len = tgt_byte_seq.size(); device = tgt_byte_seq.device
        mem_batch_size, mem_len, mem_dim_in = memory.size()
        model_dtype = next(self.parameters()).dtype

        # Handle empty target sequence
        if tgt_len == 0: return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)

        # Project and normalize memory from encoder/complex layers
        if mem_len == 0:
             logger.debug("HAKMEMLocalDecoder received empty memory.")
             projected_memory = torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=model_dtype)
             memory_key_padding_mask = None # No memory to pad
        else:
            # --- Stability Check (Memory Input) ---
            if not torch.isfinite(memory).all():
                logger.warning("NaN/Inf detected in memory input to LocalDecoder. Replacing.")
                memory = torch.nan_to_num(memory)
            projected_memory = self.memory_projection(memory.to(model_dtype))
            # --- Stability Check (Projected Memory) ---
            if not torch.isfinite(projected_memory).all():
                 logger.warning("NaN/Inf detected after memory projection in LocalDecoder. Replacing.")
                 projected_memory = torch.nan_to_num(projected_memory)

        # Prepare target sequence embeddings + positional encodings
        tgt_embed = self.byte_embeddings(tgt_byte_seq.long()).to(model_dtype)
        # Create position indices, clamping to max embedding length
        positions = torch.arange(0, tgt_len, device=device).unsqueeze(0)
        positions = torch.clamp(positions, max=self.positional_encoding.num_embeddings - 1)
        pos_embed = self.positional_encoding(positions).to(model_dtype)
        tgt_prepared = self.dropout_embed(tgt_embed + pos_embed)

        # --- Stability Check (Target Input) ---
        if not torch.isfinite(tgt_prepared).all():
            logger.warning("NaN/Inf detected in prepared target sequence input to Decoder Transformer. Replacing.")
            tgt_prepared = torch.nan_to_num(tgt_prepared)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)

        # Ensure masks are boolean on the correct device
        if tgt_mask is not None: tgt_mask = tgt_mask.to(device=device, dtype=torch.bool)
        # `memory_key_padding_mask` should have True where memory elements are PADDED (invalid).
        if memory_key_padding_mask is not None: memory_key_padding_mask = memory_key_padding_mask.to(device=device, dtype=torch.bool)

        # Pass through Transformer Decoder
        # Note: memory_mask is for masking specific memory elements *from* specific target elements (rarely used)
        # memory_key_padding_mask masks entire memory elements (e.g., padding)
        output = self.transformer(
            tgt=tgt_prepared,
            memory=projected_memory,
            tgt_mask=tgt_mask, # Causal mask for self-attention
            memory_mask=None, # Mask for cross-attention (query pos -> key pos) - usually not needed
            tgt_key_padding_mask=None, # Padding mask for target sequence (if any)
            memory_key_padding_mask=memory_key_padding_mask # Padding mask for memory sequence
        )

        # --- Stability Check (Decoder Output) ---
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in output of Decoder Transformer. Replacing.")
            output = torch.nan_to_num(output)

        # Generate final byte logits using the prediction head
        if self.use_hierarchical:
            byte_class_logits = self.byte_class_pred(output)
            # --- Stability Check ---
            if not torch.isfinite(byte_class_logits).all():
                logger.warning("NaN/Inf in hierarchical class logits. Replacing.")
                byte_class_logits = torch.nan_to_num(byte_class_logits)
            # Use log_softmax for numerical stability when combining probabilities
            log_class_probs = F.log_softmax(byte_class_logits, dim=-1) # Shape (B, S, 16)

            log_specific_probs_list = []
            for i in range(16):
                specific_logits = self.byte_specific_pred[i](output)
                # --- Stability Check ---
                if not torch.isfinite(specific_logits).all():
                     logger.warning(f"NaN/Inf in hierarchical specific logits head {i}. Replacing.")
                     specific_logits = torch.nan_to_num(specific_logits)
                log_specific_probs_list.append(F.log_softmax(specific_logits, dim=-1)) # Shape (B, S, 16)

            log_specific_probs_stacked = torch.stack(log_specific_probs_list, dim=2) # Shape (B, S, 16, 16)

            # Combine log probabilities: log P(byte) = log P(class) + log P(specific | class)
            # Unsqueeze log_class_probs to broadcast: (B, S, 16, 1) + (B, S, 16, 16) -> (B, S, 16, 16)
            combined_log_probs = log_class_probs.unsqueeze(-1) + log_specific_probs_stacked
            # Reshape to final logits: (B, S, 256)
            byte_logits = combined_log_probs.view(batch_size, tgt_len, 256)
        else:
            # Flat prediction
            byte_logits = self.byte_pred(output)

        # Ensure final output is float32 for loss calculation and handle any remaining NaNs
        byte_logits = byte_logits.float()
        if not torch.isfinite(byte_logits).all():
            logger.warning("NaN/Inf detected in final decoder logits. Replacing with zeros.")
            byte_logits = torch.nan_to_num(byte_logits, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0

        return byte_logits


# =====================================================================
# HAKMEM-Enhanced Q-Learning Controller & Optimizer
# =====================================================================
class HAKMEMQController:
    """A Q-learning agent to dynamically adjust optimizer hyperparameters."""
    def __init__(self, learning_rate: float=0.01, discount: float=0.95, epsilon: float=0.2, epsilon_decay: float=0.9998, min_epsilon: float=0.01, lr_scale_options: Optional[List[float]]=None, momentum_scale_options: Optional[List[float]]=None, max_q_table_size: int=10000):
        self.q_table = {}; self.alpha = learning_rate; self.gamma = discount # Q-learning params
        self.epsilon = epsilon; self.min_epsilon = min_epsilon; self.epsilon_decay = epsilon_decay # Exploration params
        self.prev_loss = None; self.prev_state = None; self.prev_action = None # Tracking previous step

        # Define the discrete action space (scaling factors for LR and Momentum)
        if lr_scale_options is None: lr_scale_options = [0.9, 0.95, 1.0, 1.05, 1.1]
        if momentum_scale_options is None: momentum_scale_options = [0.95, 0.98, 1.0, 1.01, 1.02]
        self.action_ranges = {'lr_scale': np.array(lr_scale_options), 'momentum_scale': np.array(momentum_scale_options)}
        self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}

        # State tracking windows
        self.loss_window = deque(maxlen=10)
        self.grad_norm_window = deque(maxlen=10)
        self.lr_window = deque(maxlen=5)
        self.momentum_window = deque(maxlen=5)
        self.performance_window = deque(maxlen=20) # Tracks recent rewards

        # State variables for reward shaping
        self.stable_steps = 0; self.oscillation_counter = 0
        self.prev_actions_log = deque(maxlen=5) # Track recent actions

        # Q-table management
        self.max_q_table_size = max_q_table_size; self.q_table_access_count = {}
        self.q_table_creation_time = {}

        # Reward shaping parameters
        self.flow_coefficient = 0.05 # Modulates alpha based on TD error
        self.oscillation_penalty = 0.15
        self.stability_reward_bonus = 0.05

        logger.info(f"QController initialized: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}, decay={self.epsilon_decay}")
        logger.info(f"QController action ranges: LR={self.action_ranges['lr_scale']}, Momentum={self.action_ranges['momentum_scale']}")

    def get_state(self, lr, momentum, grad_norm, loss):
        """Discretizes the current training status into a state tuple."""
        # Handle invalid inputs gracefully
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm):
            # logger.debug(f"QController: Invalid loss ({loss}) or grad_norm ({grad_norm}) for state calculation. Returning default state.") # Less noisy
            # Return a default/neutral state representation
            return (2, 2, 0, 2, 1) # Mid-range bins

        # Update state tracking windows
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm)
        self.lr_window.append(lr); self.momentum_window.append(momentum)
        # Need sufficient history to calculate trends/levels
        if len(self.loss_window) < 3 or len(self.grad_norm_window) < 3:
            # logger.debug("QController: Not enough history for state calculation. Returning default state.") # Less noisy
            return (2, 2, 0, 2, 1)

        # Default state bins
        loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0

        try:
            # Loss Trend: Calculate slope of recent loss values
            y = np.array(list(self.loss_window)[-5:]); x = np.arange(len(y))
            if len(y) >= 2 and len(np.unique(y)) >= 2: # Need at least 2 points, avoid singular matrix
                coeffs = np.polyfit(x, y, 1); slope = coeffs[0]
            else: slope = 0.0
            avg_loss = np.mean(y); normalized_slope = slope / (abs(avg_loss) + 1e-6) # Normalize slope by avg loss
            # Bins: Steep decrease, decrease, flat, increase, steep increase
            loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05])

            # Gradient Norm Level: Use median for robustness to outliers
            avg_grad_norm = np.median(list(self.grad_norm_window))
            # Bins: Very low, low, medium, high, very high
            grad_norm_level_bin = np.digitize(avg_grad_norm, bins=[0.1, 0.5, 1.5, 5.0])

            # Learning Rate Level: Based on magnitude relative to typical starting LR
            lr_level_bin = np.digitize(lr / 1e-4, bins=[0.5, 2.0, 10.0, 50.0]) # e.g., <0.5e-4, <2e-4, <10e-4, <50e-4, >=50e-4

            # Momentum Level: Standard momentum ranges
            momentum_level_bin = np.digitize(momentum, bins=[0.85, 0.92, 0.97]) # <0.85, <0.92, <0.97, >=0.97

            # Oscillation Detection: Check if recent reward signs are alternating
            if len(self.performance_window) >= 2:
                 # If reward flips sign significantly
                 if (self.performance_window[-1] > 1e-3 and self.performance_window[-2] < -1e-3) or \
                    (self.performance_window[-1] < -1e-3 and self.performance_window[-2] > 1e-3):
                      self.oscillation_counter = min(self.oscillation_counter + 1, 5) # Increase counter (capped)
                 else: self.oscillation_counter = max(0, self.oscillation_counter - 1) # Decrease if stable
            # Binary bin: oscillating or not
            oscillation_bin = 1 if self.oscillation_counter >= 3 else 0

        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
            # Catch numerical errors during state calculation
            logger.warning(f"Q-state calculation numerical error: {e}. Using default bins.")
            loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0

        # Combine bins into state tuple
        state = (loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin)
        # Track state usage for potential pruning
        self.q_table_access_count[state] = self.q_table_access_count.get(state, 0) + 1
        return state

    def compute_reward(self, current_loss, prev_loss, grad_norm):
        """Calculates the reward based on loss change and stability metrics."""
        # Handle invalid inputs
        if current_loss is None or prev_loss is None or grad_norm is None or \
           not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm):
            # logger.debug("QController: Invalid inputs for reward. Returning 0."); return 0.0 # Less noisy
            return 0.0

        # Primary reward signal: normalized loss change
        loss_change = prev_loss - current_loss # Positive change is good (loss decreased)
        loss_change_ratio = loss_change / (abs(prev_loss) + 1e-6) # Normalize by previous loss magnitude
        reward = np.tanh(loss_change_ratio * 10.0) # Squash into [-1, 1], scale sensitivity

        # Penalties/Bonuses
        if grad_norm > 5.0: reward -= 0.1 * min(1.0, max(0.0, (grad_norm - 5.0) / 10.0)) # Penalize large gradients
        elif grad_norm < 0.05: reward += 0.02 # Small bonus for very small gradients (convergence)
        if self.oscillation_counter >= 3: reward -= self.oscillation_penalty # Penalize detected oscillation

        # Track reward for oscillation detection and stability bonus
        self.performance_window.append(reward)
        if reward > 0.0: # If loss improved
             self.stable_steps += 1
             # Bonus for consecutive steps with improvement
             reward += min(0.1, self.stability_reward_bonus * (self.stable_steps // 5))
        else: self.stable_steps = 0 # Reset stability counter if loss didn't improve

        # Clip final reward to [-1, 1]
        return float(np.clip(reward, -1.0, 1.0))

    def choose_action(self, state):
        """Selects actions (LR scale, Momentum scale) based on state using epsilon-greedy."""
        if state is None:
            # logger.debug("QController: State is None, returning default actions (no scaling).") # Less noisy
            return {'lr_scale': 1.0, 'momentum_scale': 1.0} # Default: no change

        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[state] = time.time(); self.q_table_access_count[state] = 1
            # logger.debug(f"QController: Initialized Q for new state {state}. Table size: {len(self.q_table)}") # Less noisy
            self._manage_q_table_size() # Prune if table is too large

        # Epsilon decay for exploration vs exploitation
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        chosen_actions = {}
        for param, q_values in self.q_table[state].items():
            action_space = self.action_ranges[param]
            if random.random() < self.epsilon: # Explore: choose random action
                chosen_idx = random.randrange(len(action_space))
            else: # Exploit: choose action with highest Q-value
                # Handle potential NaN/Inf Q-values gracefully
                q_values_finite = q_values[np.isfinite(q_values)]
                if len(q_values_finite) == 0: # If all Q-values are invalid, choose randomly
                    chosen_idx = random.randrange(len(action_space))
                else:
                    max_q = np.max(q_values_finite)
                    # Find all indices with Q-value close to max_q (handles ties)
                    best_indices = np.where(np.isclose(q_values, max_q) & np.isfinite(q_values))[0]
                    if len(best_indices) > 0: chosen_idx = np.random.choice(best_indices) # Randomly choose among best
                    else: chosen_idx = random.randrange(len(action_space)) # Fallback if no finite max found

            chosen_actions[param] = float(action_space[chosen_idx]) # Get the actual scaling factor

        self.prev_actions_log.append(chosen_actions.copy()) # Log chosen action
        return chosen_actions

    def update(self, state, action, reward, next_state):
        """Updates the Q-table using the Bellman equation."""
        if state is None or next_state is None or action is None:
            # logger.debug("QController: Skipping Q-update due to None state/action/next_state."); return # Less noisy
            return
        # Ensure states exist in the table
        if state not in self.q_table: logger.warning(f"QController: State {state} not in Q-table during update. Skipping."); return
        if next_state not in self.q_table:
             # Initialize Q-values for the next state if encountered for the first time during update
             self.q_table[next_state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
             self.q_table_creation_time[next_state] = time.time(); self.q_table_access_count[next_state] = 0
             # logger.debug(f"QController: Initialized Q for next_state {next_state} during update.") # Less noisy
             self._manage_q_table_size()

        # Update Q-value for each parameter's action
        for param, chosen_value in action.items():
             action_space = self.action_ranges[param]
             # Find the index corresponding to the chosen action value
             action_indices = np.where(np.isclose(action_space, chosen_value))[0]
             if len(action_indices) == 0:
                 logger.warning(f"QController ({param}): Could not find index for action {chosen_value:.4f}. Skipping update."); continue
             action_idx = action_indices[0]

             # Q-learning update rule: Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a'(Q(s',a')) - Q(s,a)]
             current_q = self.q_table[state][param][action_idx]

             # Find the best Q-value for the next state (max_a' Q(s', a'))
             next_q_values = self.q_table[next_state][param]
             finite_next_q = next_q_values[np.isfinite(next_q_values)]
             max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0
             if not np.isfinite(max_future_q): max_future_q = 0.0 # Handle if max is still inf

             # Calculate TD target and TD error
             td_target = reward + self.gamma * max_future_q; td_error = td_target - current_q

             # Use adaptive learning rate (alpha) based on TD error magnitude
             adaptive_alpha = min(0.5, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error))))

             # Update Q-value
             new_q = current_q + adaptive_alpha * td_error
             # Clip Q-value to prevent explosion and handle potential NaNs
             self.q_table[state][param][action_idx] = np.clip(new_q, -1e5, 1e5) if np.isfinite(new_q) else 0.0

    def _manage_q_table_size(self):
         """Prunes the Q-table if it exceeds the maximum size, removing least used/oldest states."""
         if len(self.q_table) > self.max_q_table_size:
            num_to_remove = len(self.q_table) - self.max_q_table_size
            # logger.debug(f"Q-table ({len(self.q_table)}) exceeds limit ({self.max_q_table_size}). Pruning {num_to_remove}.") # Less noisy
            try:
                # Check if metadata for pruning exists
                if not self.q_table_access_count or not self.q_table_creation_time:
                     # Fallback: Remove random states if metadata is missing
                     states_to_remove = random.sample(list(self.q_table.keys()), min(num_to_remove, len(self.q_table)))
                     logger.warning("Q-table metadata incomplete during pruning. Removing random states.")
                else:
                    # Sort states primarily by access count (ascending), then by creation time (ascending)
                    sorted_states = sorted(self.q_table.keys(), key=lambda s: (
                        self.q_table_access_count.get(s, 0), self.q_table_creation_time.get(s, float('inf'))))
                    # Select the states with lowest access count / oldest creation time
                    states_to_remove = sorted_states[:num_to_remove]

                # Remove selected states from Q-table and metadata dictionaries
                for state_to_remove in states_to_remove:
                    self.q_table.pop(state_to_remove, None)
                    self.q_table_access_count.pop(state_to_remove, None)
                    self.q_table_creation_time.pop(state_to_remove, None)
                # logger.debug(f"Pruned {len(states_to_remove)} states. New Q-table size: {len(self.q_table)}") # Less noisy
            except Exception as e:
                # Catch any error during pruning and attempt random removal as a last resort
                logger.warning(f"Error during Q-table pruning: {e}. Attempting random removal.", exc_info=False)
                current_keys = list(self.q_table.keys())
                num_to_remove = len(current_keys) - self.max_q_table_size
                if num_to_remove > 0:
                    states_to_remove = random.sample(current_keys, min(num_to_remove, len(current_keys)))
                    for state_to_remove in states_to_remove:
                        self.q_table.pop(state_to_remove, None)
                        self.q_table_access_count.pop(state_to_remove, None)
                        self.q_table_creation_time.pop(state_to_remove, None)

    def get_info(self) -> Dict:
        """Returns current status information about the Q-controller."""
        last_action = self.prev_actions_log[-1] if self.prev_actions_log else None
        avg_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.0
        return {"epsilon": self.epsilon, "stable_steps": self.stable_steps, "oscillation_counter": self.oscillation_counter,
                "q_table_size": len(self.q_table), "last_action": last_action,
                "avg_reward_last_20": avg_reward}


class HAKMEMEnhancedSGD(torch.optim.Optimizer):
    """SGD optimizer enhanced with momentum, weight decay, gradient clipping, and Q-learning control."""
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm=1.0, q_learning_config={}, enable_flow=False, flow_coefficient=0.05, flow_momentum=0.95):
        # Validate inputs
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        # Store base LR/Momentum separately for Q-controller scaling
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize Q-controller if config is provided
        self.q_controller = HAKMEMQController(**q_learning_config) if q_learning_config else None
        self.max_grad_norm = max_grad_norm # For gradient clipping (used by Trainer)
        self._step_count = 0
        self.current_loss = None # Track loss for Q-controller (set via set_current_loss)
        self.flow_enabled = enable_flow # Placeholder for potential future 'flow' feature
        self.gradient_stats = GradientStats() # Track gradient clipping stats

        # Initialize momentum buffer for each parameter
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def zero_grad(self, set_to_none=True):
        """Zeros gradients."""
        super().zero_grad(set_to_none=set_to_none)
        # Note: GradientStats reset is handled by Trainer after logging

    def set_current_loss(self, loss: Optional[float]):
         """Sets the loss for the current step, used by the Q-controller."""
         if loss is not None and np.isfinite(loss):
            # Store loss for Q-controller reward calculation
            self.current_loss = loss
         else:
             # Handle non-finite loss
             self.current_loss = None
             logger.debug("Optimizer received non-finite or None loss.")

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step (including Q-control and gradient updates)."""
        # Closure execution is not standard with GradScaler/AMP and grad accum loops.
        # Loss should be calculated in the Trainer's loop and passed via set_current_loss.
        if closure is not None:
            logger.warning("HAKMEMEnhancedSGD.step received a closure, but it's not used in standard AMP/accum loop.")

        # Apply Q-learning chosen actions to hyperparameters
        # This action was chosen based on the state *before* this step
        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action
            for group in self.param_groups:
                base_lr = group['base_lr']; base_momentum = group['base_momentum']
                new_lr = base_lr * q_action['lr_scale']; new_momentum = base_momentum * q_action['momentum_scale']
                group['lr'] = float(np.clip(new_lr, 1e-8, 0.1))
                group['momentum'] = float(np.clip(new_momentum, 0.5, 0.999))
        # else: # First step or no Q-controller action available yet
             # Use initial/default hyperparams

        # === Parameter Update ===
        # NOTE: Gradient clipping (unscale_ + clip_grad_norm_) should have been done *before* this step
        # by the Trainer, using the scaler. This step applies the update using the clipped grads.
        # Gradient stats are also recorded by the Trainer.

        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']; weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                if not p.requires_grad: continue

                grad = p.grad # Gradients should already be unscaled and clipped here by Trainer
                if not torch.isfinite(grad).all():
                     logger.error(f"Optimizer Error: Non-finite gradient found for param shape {p.shape} AT THE POINT OF UPDATE. Skipping update for this param.")
                     continue # Skip update for this parameter

                param_data_float = p.data.float() # Use float for update calculation
                param_state = self.state[p]

                # Apply weight decay (L2 penalty) directly to the gradient before momentum
                if weight_decay != 0:
                    grad = grad.add(param_data_float, alpha=weight_decay)

                # Momentum buffer update
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = param_state['momentum_buffer']
                # Update buffer: buf = momentum*buf + grad
                buf.mul_(momentum).add_(grad.to(buf.dtype)) # Ensure grad is same dtype as buffer for add_
                param_state['momentum_buffer'] = buf # Update state

                # Parameter update: p.data = p.data - lr * buf
                update_step = buf * lr
                p.data.add_(-update_step.to(p.data.dtype)) # Update in-place, cast back to original dtype

        self._step_count += 1
        return None # Explicitly return None

    def get_q_info(self) -> Dict:
        """Retrieves information from the Q-controller."""
        if hasattr(self, 'q_controller') and self.q_controller: return self.q_controller.get_info()
        return {}


# =====================================================================
# Utilities
# =====================================================================

def is_main_process():
    """Checks if the current process is the main process (rank 0 in DDP or single process)."""
    return not is_initialized() or get_rank() == 0


# =====================================================================
# Integrated Model Definition (Using Hyperbolic Attention)
# =====================================================================

class IntegratedHyperHAKMEMModel(nn.Module):
    """The main integrated model combining patching, local encoding, hyperbolic space,
       hyperbolic attention layers, and local decoding."""
    def __init__(
        self,
        local_hidden_size: int = 256,
        hyperbolic_embedding_dim: int = 384, # Dimension used for hyperbolic processing
        num_hyperbolic_layers: int = 8,      # Number of hyperbolic attention layers
        num_hyperbolic_heads: int = 8,       # Heads for hyperbolic attention
        decoder_memory_dim: int = 768,
        dropout: float = 0.15,
        context_window: int = 256,
        n_gram_sizes: List[int] = [3, 4], n_gram_vocab_size: int = 30000,
        # projection_method: str = "hakmem_enhanced", # Removed as complex layers are gone
        use_hierarchical_decoder: bool = True,
        curvature: float = 0.8, clipping_radius: float = 1.0,
        use_amp: bool = True # Argument passed but AMP context managed by Trainer
    ):
        super().__init__()
        # Store configuration
        self.local_hidden_size = local_hidden_size
        self.hyperbolic_embedding_dim = hyperbolic_embedding_dim
        self.decoder_memory_dim = decoder_memory_dim
        # Clamp curvature and radius to sensible ranges
        self.curvature = max(min(curvature, 5.0), 1e-6) # Ensure positive and not excessively large
        self.clipping_radius = max(min(clipping_radius, 1.0 - 1e-5), 0.1) # Ensure < 1.0 and not too small
        self.use_hierarchical_decoder = use_hierarchical_decoder
        self.context_window = context_window
        self.use_amp_in_generate = use_amp # Control AMP specifically during generation

        # --- Input Validation ---
        if local_hidden_size <= 0 or hyperbolic_embedding_dim <= 0 or decoder_memory_dim <= 0:
             raise ValueError("All dimension sizes must be positive")
        if num_hyperbolic_layers <= 0: raise ValueError("num_hyperbolic_layers must be positive")
        # Adjust hyperbolic heads if needed
        if num_hyperbolic_heads <= 0: num_hyperbolic_heads = max(1, hyperbolic_embedding_dim // 64)
        if hyperbolic_embedding_dim % num_hyperbolic_heads != 0:
             logger.warning(f"hyperbolic_embedding_dim ({hyperbolic_embedding_dim}) not divisible by heads ({num_hyperbolic_heads}). Adjusting.")
             valid_heads = [h for h in range(num_hyperbolic_heads, 0, -1) if hyperbolic_embedding_dim % h == 0]
             num_hyperbolic_heads = valid_heads[0] if valid_heads else 1
             logger.warning(f"Set num_hyperbolic_heads to {num_hyperbolic_heads}")

        # --- Module Initialization ---
        self.hyperbolic_utils = HyperbolicUtils()
        self.patcher = HAKMEMBabylonIndex()

        # 1. Local Encoder
        local_encoder_heads = max(1, local_hidden_size // 64) # Auto-determine heads
        self.local_encoder = HAKMEMLocalEncoder(
            hidden_size=local_hidden_size, num_layers=2, num_heads=local_encoder_heads,
            dropout=dropout, n_gram_sizes=n_gram_sizes, n_gram_vocab_size=n_gram_vocab_size
        )

        # 2. Projection to Euclidean space (for mapping to hyperbolic)
        self.projection_to_hyp_euclidean = nn.Linear(local_hidden_size, hyperbolic_embedding_dim)
        nn.init.xavier_uniform_(self.projection_to_hyp_euclidean.weight); nn.init.zeros_(self.projection_to_hyp_euclidean.bias)

        # 3. Positional Encoding for Tangent Space (Applied before Hyperbolic Attention layers)
        # Use standard learned embeddings for simplicity
        self.tangent_positional_encoding = nn.Embedding(context_window * 2, hyperbolic_embedding_dim) # Max length assumption
        nn.init.normal_(self.tangent_positional_encoding.weight, std=0.02)
        self.dropout_pe = nn.Dropout(dropout)

        # 4. Input Normalization for Hyperbolic Attention Layers (applied to tangent space + PE)
        self.tangent_norm_in = nn.LayerNorm(hyperbolic_embedding_dim, eps=1e-6)

        # 5. Hyperbolic Attention Layers Stack
        self.hyperbolic_attention_layers = nn.ModuleList()
        for _ in range(num_hyperbolic_layers):
             self.hyperbolic_attention_layers.append(
                 HyperbolicAttentionLayer(
                     dim=hyperbolic_embedding_dim, heads=num_hyperbolic_heads, dropout=dropout,
                     curvature=self.curvature
                 )
             )

        # 6. Output Normalization after Hyperbolic Layers (Included within HyperbolicAttentionLayer)

        # 7. Projection from Tangent Space to Decoder Memory Dimension
        self.projection_to_decoder_memory = nn.Linear(hyperbolic_embedding_dim, decoder_memory_dim)
        nn.init.xavier_uniform_(self.projection_to_decoder_memory.weight); nn.init.zeros_(self.projection_to_decoder_memory.bias)

        # 8. Local Decoder
        decoder_heads = max(1, local_hidden_size // 64) # Auto-determine heads
        decoder_max_len = context_window * 2 # Max sequence length decoder can handle
        self.local_decoder = HAKMEMLocalDecoder(
            hidden_size=local_hidden_size, global_hidden_size=decoder_memory_dim, num_layers=4,
            num_heads=decoder_heads, dropout=dropout, use_hierarchical_pred=use_hierarchical_decoder,
            max_decode_len=decoder_max_len
        )

        logger.info(f"IntegratedHyperHAKMEM Initialized with Hyperbolic Attention: HypDim={hyperbolic_embedding_dim}, Curve={self.curvature:.2f}, Hyp Layers: {num_hyperbolic_layers}, Heads: {num_hyperbolic_heads}, DecMem={decoder_memory_dim}")
        logger.info(f"Patching: Word/Punctuation | LocalEnc/Dec Hidden: {local_hidden_size}")
        logger.info(f"Decoder Hierarchical: {use_hierarchical_decoder}")


    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full forward pass of the model using Hyperbolic Attention."""
        batch_size = byte_seq.size(0); device = byte_seq.device; model_dtype = next(self.parameters()).dtype

        # --- 1. Patching and Local Encoding (Per Batch Item) ---
        batch_patch_repr_list = []; num_patches_per_item = []; valid_batch_indices = []
        max_num_patches = 0
        for i in range(batch_size):
            seq = byte_seq[i]
            patches_with_entropy = self.patcher.create_patches(seq)
            if patches_with_entropy:
                patches_on_device = [(p.to(device), e) for p, e in patches_with_entropy]
                patch_repr_single = self.local_encoder(patches_on_device)
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

        if not valid_batch_indices:
             target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
             logger.warning("No valid patches generated for any item in the batch. Decoder memory will be empty.")
             if target_byte_seq is not None: return torch.zeros((batch_size, target_len, 256), device=device, dtype=torch.float32)
             else: return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)

        # --- 2. Pad and Combine Patch Representations ---
        num_valid = len(valid_batch_indices)
        padded_patch_repr = torch.zeros(num_valid, max_num_patches, self.local_hidden_size, device=device, dtype=model_dtype)
        memory_padding_mask = torch.ones(num_valid, max_num_patches, dtype=torch.bool, device=device) # True where padded
        for idx, patch_repr_tensor in enumerate(batch_patch_repr_list):
             num_p = patch_repr_tensor.size(0)
             padded_patch_repr[idx, :num_p, :] = patch_repr_tensor
             memory_padding_mask[idx, :num_p] = False # Mark non-padded elements as False

        # --- 3. Hyperbolic Processing (Mapping to Tangent Space) ---
        euclidean_for_hyper = self.projection_to_hyp_euclidean(padded_patch_repr)
        if not torch.isfinite(euclidean_for_hyper).all():
             logger.warning("NaN/Inf after projection_to_hyp_euclidean."); euclidean_for_hyper = torch.nan_to_num(euclidean_for_hyper)
        clipped_euclidean = self.hyperbolic_utils.poincare_clip(euclidean_for_hyper, self.curvature, self.clipping_radius)
        clipped_euclidean = torch.nan_to_num(clipped_euclidean)
        hyperbolic_repr = self.hyperbolic_utils.exponential_map(clipped_euclidean, self.curvature)
        hyperbolic_repr = torch.nan_to_num(hyperbolic_repr)
        tangent_repr = self.hyperbolic_utils.logarithmic_map(hyperbolic_repr, self.curvature)
        tangent_repr = torch.nan_to_num(tangent_repr)

        # --- 4. Hyperbolic Attention Processing ---
        current_seq_len = tangent_repr.size(1)
        positions = torch.arange(0, current_seq_len, device=device).unsqueeze(0)
        positions = torch.clamp(positions, max=self.tangent_positional_encoding.num_embeddings - 1)
        pos_embed = self.tangent_positional_encoding(positions)
        x = tangent_repr + pos_embed
        x = self.dropout_pe(x)
        x = self.tangent_norm_in(x)
        if not torch.isfinite(x).all():
             logger.warning("NaN/Inf after initial tangent norm + PE. Replacing.")
             x = torch.nan_to_num(x)

        # The `memory_padding_mask` needs to be passed to the attention layers.
        # It should have shape (B, N_patches) -> becomes (B, 1, 1, N_patches) inside attention.
        # True means MASK OUT.
        for layer in self.hyperbolic_attention_layers:
            x = layer(x, attention_mask=memory_padding_mask)

        processed_tangent_repr = x

        # --- 5. Projection to Decoder Memory and Decoding ---
        decoder_memory = self.projection_to_decoder_memory(processed_tangent_repr)
        if not torch.isfinite(decoder_memory).all():
            logger.warning("NaN/Inf detected in final decoder_memory. Replacing."); decoder_memory = torch.nan_to_num(decoder_memory)

        if target_byte_seq is None:
            logger.debug("target_byte_seq is None in forward pass (likely initial gen step).")
            return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)

        # --- 6. Decoding ---
        target_len = target_byte_seq.size(1)
        if target_len == 0:
             logger.debug("Received empty target_byte_seq for decoder.")
             return torch.zeros((batch_size, 0, 256), device=device, dtype=torch.float32)
        valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
        valid_target_byte_seq = torch.index_select(target_byte_seq, 0, valid_indices_tensor).to(device) if num_valid > 0 else torch.empty(0, target_len, dtype=torch.long, device=device)

        if num_valid > 0:
            # Pass the memory padding mask to the decoder
            byte_logits_valid = self.local_decoder(
                tgt_byte_seq=valid_target_byte_seq,
                memory=decoder_memory,
                memory_key_padding_mask=memory_padding_mask # Pass the mask here
            )
            if not torch.isfinite(byte_logits_valid).all():
                logger.warning("NaN/Inf detected in byte_logits_valid from decoder before final reconstruction. Replacing.")
                byte_logits_valid = torch.nan_to_num(byte_logits_valid, nan=0.0, posinf=0.0, neginf=0.0)
        else:
             byte_logits_valid = torch.empty((0, target_len, 256), device=device, dtype=torch.float32)

        # --- 7. Reconstruct Full Batch Output ---
        final_byte_logits = torch.zeros((batch_size, target_len, 256), device=device, dtype=torch.float32)
        if num_valid > 0 and byte_logits_valid.numel() > 0:
            try:
                if byte_logits_valid.shape[0] == valid_indices_tensor.shape[0]:
                    final_byte_logits.index_copy_(0, valid_indices_tensor, byte_logits_valid)
                else:
                    logger.error(f"Shape mismatch for index_copy_: final={final_byte_logits.shape}, valid={byte_logits_valid.shape}, indices={valid_indices_tensor.shape}. Skipping copy.")
            except IndexError as e:
                logger.error(f"Error during final logit reconstruction: {e}. Shapes: final={final_byte_logits.shape}, valid={byte_logits_valid.shape}, indices={valid_indices_tensor.shape}")
        if not torch.isfinite(final_byte_logits).all():
             logger.error("NaN/Inf detected in final reconstructed logits! Replacing with zeros.")
             final_byte_logits = torch.nan_to_num(final_byte_logits, nan=0.0, posinf=0.0, neginf=0.0)

        return final_byte_logits

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        if seq_len == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logits = logits.float()
        targets = targets.long()
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)
        if not torch.isfinite(logits_flat).all():
            num_nan = torch.isnan(logits_flat).sum().item(); num_inf = torch.isinf(logits_flat).sum().item()
            logger.error(f"NaN/Inf detected in logits passed to compute_loss (NaN:{num_nan}, Inf:{num_inf}). Returning high loss.")
            return torch.tensor(100.0, device=logits.device, requires_grad=True)
        if not torch.isfinite(targets_flat).all():
            logger.error("NaN/Inf detected in targets passed to compute_loss. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, requires_grad=True)
        if smoothing > 0.0 and 0.0 < smoothing < 1.0:
            with torch.no_grad():
                smooth_val_on = 1.0 - smoothing
                smooth_val_off = smoothing / max(1, vocab_size - 1)
                true_dist = torch.full_like(logits_flat, smooth_val_off)
                true_dist.scatter_(1, targets_flat.unsqueeze(1), smooth_val_on)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -(true_dist * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        if not torch.isfinite(loss).all():
            logger.error("NaN/Inf detected in loss calculation before masking. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, requires_grad=True)
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            loss = loss.masked_fill(mask_flat, 0.0)
            num_active_elements = (~mask_flat).sum()
            if num_active_elements.item() == 0:
                 mean_loss = torch.tensor(0.0, device=logits.device)
            else:
                 mean_loss = loss.sum() / num_active_elements
        else:
            mean_loss = loss.mean()
        if not torch.isfinite(mean_loss):
            logger.error(f"NaN/Inf detected in final computed mean loss. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, requires_grad=True)
        return mean_loss

    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None, repetition_penalty: float = 1.1, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        self.eval(); device = next(self.parameters()).device; model_dtype = next(self.parameters()).dtype
        if seed_bytes.device != device: seed_bytes = seed_bytes.to(device)
        seed_bytes = seed_bytes.long()
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
                 # Pass context as target for autoregressive generation
                 logits_all = self(byte_seq=current_context, target_byte_seq=current_context)
            if logits_all is None or logits_all.numel() == 0 or logits_all.shape[1] == 0:
                 logger.warning(f"Logits generation failed or returned empty at step {step}. Stopping generation.")
                 break
            if not torch.isfinite(logits_all).all():
                 logger.warning(f"NaN/Inf detected in model output logits at generation step {step}. Replacing with zeros.")
                 logits_all = torch.nan_to_num(logits_all, nan=0.0, posinf=0.0, neginf=0.0)
            next_byte_logits = logits_all[:, -1, :].float()
            next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                current_logits = next_byte_logits[i].clone()
                current_seq = generated_sequence[i]; current_seq_len = current_seq.size(0)
                if repetition_penalty > 1.0 and current_seq_len > 0:
                    for ngram_size in range(2, min(max_ngram_size + 1, current_seq_len + 1)):
                        recent_ngram_tuple = tuple(current_seq[-(ngram_size-1):].cpu().tolist())
                        if recent_ngram_tuple in sequence_memory[i]:
                             prev_next_byte = sequence_memory[i][recent_ngram_tuple]
                             if 0 <= prev_next_byte < 256:
                                 if current_logits[prev_next_byte] > 0: current_logits[prev_next_byte] /= repetition_penalty
                                 else: current_logits[prev_next_byte] *= repetition_penalty
                probs_orig = F.softmax(current_logits, dim=-1)
                if torch.isnan(probs_orig).any():
                    entropy = math.log2(256.0)
                    probs_orig = torch.ones_like(current_logits) / current_logits.size(-1)
                else:
                    entropy = -torch.sum(probs_orig * torch.log2(probs_orig + 1e-10)).item()
                adaptive_temp = base_temperature
                if entropy < sampling_config.low_entropy_threshold: adaptive_temp *= 0.8
                elif entropy > sampling_config.medium_entropy_threshold: adaptive_temp *= 1.1
                adaptive_temp = max(min_temp, min(adaptive_temp, max_temp))
                scaled_logits = current_logits / adaptive_temp
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
                probs_final = F.softmax(filtered_logits, dim=-1)
                if torch.isnan(probs_final).any() or torch.isinf(probs_final).any() or probs_final.sum() < 1e-6:
                    logger.warning(f"Invalid sampling probabilities item {i} step {step}. Using uniform.")
                    probs_final = torch.ones_like(current_logits) / current_logits.size(-1)
                if temperature <= 1e-6: next_byte_idx = torch.argmax(probs_final)
                else: next_byte_idx = torch.multinomial(probs_final, num_samples=1).squeeze(-1)
                next_byte_indices[i] = next_byte_idx.item()
                if current_seq_len > 0:
                    for ngram_size in range(1, min(max_ngram_size, current_seq_len) + 1):
                         ngram_key = tuple(current_seq[-(ngram_size):].cpu().tolist())
                         sequence_memory[i][ngram_key] = next_byte_idx.item()
                if len(sequence_memory[i]) > 2000:
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
    """Simple tokenizer that encodes/decodes text to/from UTF-8 bytes."""
    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of UTF-8 byte values."""
        try: return list(text.encode('utf-8'))
        except Exception as e: logger.error(f"Error encoding text: {e}"); return []

    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        """Decodes a sequence of byte values (or tensors) back to text."""
        valid_bytes = []
        # Iterate through sequence, converting tensors/ints to valid bytes
        for b in byte_sequence:
            try:
                # Handle both integer and tensor inputs
                val = b.item() if isinstance(b, torch.Tensor) else int(b)
                # Ensure value is a valid byte
                if 0 <= val <= 255: valid_bytes.append(val)
                else: logger.warning(f"Invalid byte value {val} during decoding. Skipping.")
            except Exception as e: logger.warning(f"Error processing byte {b} during decoding: {e}. Skipping."); continue
        try:
            # Decode byte list using UTF-8, replacing errors
            return bytes(valid_bytes).decode('utf-8', errors='replace')
        except Exception as e: logger.error(f"Error decoding bytes: {e}"); return ""


# =====================================================================
# ByteIterableDataset
# =====================================================================
class ByteIterableDataset(IterableDataset):
    """An iterable dataset that loads chunks from a large .npy file of bytes."""
    def __init__(self, npy_file_path: str, context_size: int = 256, data_fraction: float = 1.0):
        if not os.path.exists(npy_file_path): raise FileNotFoundError(f"Dataset not found: {npy_file_path}")
        if context_size <= 0: raise ValueError("context_size must be positive")
        if not (0.0 < data_fraction <= 1.0): raise ValueError("data_fraction must be between 0 (exclusive) and 1 (inclusive)")

        self.npy_file_path = npy_file_path
        self.context_size = context_size
        self.data_fraction = data_fraction
        self.full_data_size = 0
        self.data_size = 0 # Size after applying fraction
        self.num_possible_samples = 0 # Number of valid starting indices
        self.data_dtype = np.uint8 # Assume byte data
        self.seed = None # For shuffling indices per worker/epoch
        self.epoch = 0 # For shuffling indices

        # --- Metadata Loading ---
        # Try reading header efficiently first
        try:
            with open(self.npy_file_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                # Read shape, order, dtype from header without loading data
                shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
                if len(shape) != 1:
                    raise ValueError(f"Dataset error: Expected 1D array, header indicates shape {shape}")
                self.full_data_size = shape[0]
                # Check dtype compatibility (should ideally be uint8 or similar byte type)
                if dtype.itemsize != 1 or dtype.kind not in ('u', 'i', 'S'): # Allow int8, uint8, bytes
                     logger.warning(f"Dataset warning: Read dtype {dtype} might not be uint8/byte. Assuming byte-compatible for size calculations.")
                self.data_dtype = dtype # Store actual dtype

            if self.full_data_size == 0:
                 raise ValueError("Dataset file appears empty or header reading failed.")

            # Calculate size after applying data fraction
            self.data_size = int(self.full_data_size * self.data_fraction)
            # Validate sizes
            if self.data_size <= self.context_size:
                 raise ValueError(f"Dataset size after fraction ({self.data_size}) not larger than context size ({self.context_size})")

            # Calculate number of possible samples (start indices)
            # A sample requires context_size + 1 bytes (context + target)
            self.num_possible_samples = max(0, self.data_size - self.context_size)
            if self.num_possible_samples == 0:
                 raise ValueError(f"Dataset size {self.data_size} too small for context {self.context_size}, no samples possible.")

            logger.info(f"Dataset Initialized (Metadata): Using {self.data_size:,}/{self.full_data_size:,} bytes ({self.data_fraction:.1%}) from {npy_file_path}")
            logger.info(f"Dataset Context: {self.context_size}, Possible start indices: {self.num_possible_samples:,}")

        except AttributeError: # Fallback for older numpy versions lacking internal functions
            logger.warning("Failed to use numpy internal header reading (likely older numpy version). Falling back to np.load for metadata.")
            temp_data = None
            try:
                # Load with mmap_mode='r' to avoid loading full data into memory just for shape/dtype
                temp_data = np.load(self.npy_file_path, mmap_mode='r')
                if len(temp_data.shape) != 1: raise ValueError(f"Dataset error: Expected 1D array, got shape {temp_data.shape}")
                # Check dtype
                if not np.issubdtype(temp_data.dtype, np.integer) and not np.issubdtype(temp_data.dtype, np.character):
                    logger.warning(f"Dataset warning: Expected integer or byte data, got {temp_data.dtype}.")
                if temp_data.itemsize != 1:
                    logger.warning(f"Dataset warning: Expected byte data (itemsize 1), got {temp_data.dtype} with itemsize {temp_data.itemsize}.")

                self.full_data_size = temp_data.shape[0]; self.data_dtype = temp_data.dtype

                # Repeat size calculations
                self.data_size = int(self.full_data_size * self.data_fraction)
                if self.data_size <= self.context_size:
                    raise ValueError(f"Dataset size after fraction ({self.data_size}) not larger than context size ({self.context_size})")
                self.num_possible_samples = max(0, self.data_size - self.context_size)
                if self.num_possible_samples == 0:
                    raise ValueError(f"Dataset size {self.data_size} too small for context {self.context_size}, no samples possible.")

                logger.info(f"Dataset Initialized (Fallback Load): Using {self.data_size:,}/{self.full_data_size:,} bytes ({self.data_fraction:.1%}) from {npy_file_path}")
                logger.info(f"Dataset Context: {self.context_size}, Possible start indices: {self.num_possible_samples:,}")

            finally:
                # Ensure memory map is closed if opened
                if temp_data is not None and hasattr(temp_data, '_mmap') and temp_data._mmap is not None:
                    try: temp_data._mmap.close()
                    except Exception: pass
                del temp_data; gc.collect()

        except Exception as e:
            logger.error(f"Error reading dataset metadata from {self.npy_file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        """Returns the number of possible start indices (samples)."""
        # Note: For IterableDataset, __len__ might not be strictly required or fully accurate,
        # but it's useful for progress bars.
        return self.num_possible_samples

    def __iter__(self):
        """Iterator yielding (context, target) pairs for the current worker."""
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # --- Determine indices for this worker ---
        num_samples_total = self.num_possible_samples
        if num_samples_total == 0: return iter([]) # No samples to yield

        # Split total samples among workers
        num_samples_per_worker = num_samples_total // num_workers
        start_sample_idx = worker_id * num_samples_per_worker
        end_sample_idx = (worker_id + 1) * num_samples_per_worker
        # Distribute remainder samples
        remainder = num_samples_total % num_workers
        if worker_id < remainder:
            start_sample_idx += worker_id; end_sample_idx += worker_id + 1
        else:
            start_sample_idx += remainder; end_sample_idx += remainder
        # Ensure end index doesn't exceed total samples
        end_sample_idx = min(end_sample_idx, num_samples_total)

        # --- Data Loading and Iteration ---
        bytes_data = None
        mmap_handle = None # To explicitly close later if needed

        try:
            # Load data using memory mapping for efficient access
            bytes_data = np.load(self.npy_file_path, mmap_mode='r')
            # Check dtype consistency
            if bytes_data.dtype != self.data_dtype:
                 logger.warning(f"Worker {worker_id}: Loaded data dtype {bytes_data.dtype} differs from expected {self.data_dtype} during iteration.")
            # Get mmap handle if available
            if hasattr(bytes_data, '_mmap') and bytes_data._mmap is not None:
                 mmap_handle = bytes_data._mmap
            else: logger.debug(f"Worker {worker_id}: Could not get _mmap attribute from loaded numpy array (might not be memory-mapped).")


            # Return empty iterator if no samples assigned to this worker
            if start_sample_idx >= end_sample_idx:
                return iter([])

            # Generate indices for this worker
            worker_indices = np.arange(start_sample_idx, end_sample_idx, dtype=np.int64)

            # --- Shuffling ---
            # Create a unique seed for this worker and epoch for reproducible shuffling
            base_seed = self.seed if self.seed is not None else 42
            current_epoch = self.epoch
            # Combine multiple sources for better randomness across workers/epochs/runs
            # Using simple addition for seed components
            seed = (base_seed + worker_id + current_epoch) % (2**32)
            rng = np.random.default_rng(seed=seed); rng.shuffle(worker_indices)

            # Iterate through shuffled indices assigned to this worker
            for idx in worker_indices:
                # Calculate start/end indices for context and target
                # Context: [idx, idx + context_size)
                # Target:  [idx + 1, idx + context_size + 1)
                start_ctx = idx; end_ctx = idx + self.context_size; end_tgt = end_ctx + 1

                # Check if the indices would go out of bounds of the usable data size
                if end_tgt > self.data_size:
                    # This might happen near the end of the data fraction
                    # logger.warning(f"Worker {worker_id}: Calculated index {idx} leads to out-of-bounds access ({end_tgt} > {self.data_size}). Skipping.")
                    continue
                try:
                    # Slice data, making copies to avoid issues with mmap lifetime
                    context_slice = bytes_data[start_ctx : end_ctx].copy()
                    target_slice = bytes_data[start_ctx + 1 : end_tgt].copy()
                    # Convert to PyTorch tensors (long type for embedding lookups)
                    context_tensor = torch.tensor(context_slice, dtype=torch.long)
                    target_tensor = torch.tensor(target_slice, dtype=torch.long)
                    yield context_tensor, target_tensor
                except IndexError:
                     # Catch potential race conditions or calculation errors
                     logger.warning(f"Worker {worker_id}: IndexError accessing data at index {idx} (Context: {start_ctx}-{end_ctx}, Target: {start_ctx+1}-{end_tgt}, Data Size: {self.data_size}). Skipping.")
                     continue
                except Exception as e:
                    # Catch any other errors during data processing for this index
                    logger.error(f"Worker {worker_id}: Error processing index {idx}: {e}", exc_info=False)
                    continue

        except FileNotFoundError:
            logger.error(f"Worker {worker_id}: Dataset file not found: {self.npy_file_path}")
        except Exception as e:
            # Catch errors during worker setup (e.g., file loading)
            logger.error(f"Worker {worker_id}: Failed dataset iteration setup or execution: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            # Explicitly close the memory map handle if it was obtained
            if mmap_handle is not None:
                 try: mmap_handle.close()
                 except Exception as close_ex: logger.warning(f"Worker {worker_id}: Error closing mmap handle: {close_ex}")
            # Delete reference to the potentially large numpy array
            del bytes_data
            # Explicitly call garbage collector (might help release mmap resources sooner)
            gc.collect()

    def set_seed(self, seed):
        """Sets the base seed for shuffling."""
        logger.debug(f"Dataset seed set to {seed}")
        self.seed = seed

    def set_epoch(self, epoch):
        """Sets the current epoch number (used for shuffling seed)."""
        logger.debug(f"Dataset epoch set to {epoch}")
        self.epoch = epoch


# =====================================================================
# Trainer Class (Updated with Improved TQDM)
# =====================================================================
class Trainer:
    """Manages the training and validation loops, checkpointing, and logging."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10, save_interval: int = 1000, checkpoint_dir: str = "checkpoints", wandb_enabled: bool = False, max_grad_norm: float = 1.0, rank: int = 0, world_size: int = 1, detect_anomaly: bool = False):
        self.model = model; self.optimizer = optimizer; self.device = device
        self.train_loader = train_loader; self.val_loader = val_loader
        self.grad_accum_steps = max(1, grad_accum_steps) # Ensure at least 1
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch, "amp")
        self.scaler = amp.GradScaler(enabled=self.use_amp) # AMP GradScaler
        self.log_interval = log_interval; self.save_interval = max(1, save_interval) if save_interval > 0 else 0
        self.checkpoint_dir = checkpoint_dir; self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE and is_main_process() # Log only on main process
        self.max_grad_norm = max_grad_norm; self.global_step = 0; self.current_epoch = 0
        self.last_val_metrics = None; self.rank = rank; self.world_size = world_size
        self.is_main = is_main_process() # Cache if this is the main process
        self.detect_anomaly = detect_anomaly # Enable anomaly detection for debugging
        os.makedirs(self.checkpoint_dir, exist_ok=True) # Ensure checkpoint dir exists

        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(self.optimizer.gradient_stats, GradientStats)
        self.has_q_controller = hasattr(self.optimizer, 'q_controller') and isinstance(self.optimizer.q_controller, HAKMEMQController)
        self.wandb_run = wandb.run if self.wandb_enabled else None # Store wandb run object if enabled

        logger.info(f"Trainer initialized Rank {rank}. AMP: {self.use_amp}, Grad Accum: {self.grad_accum_steps}, Max Norm: {self.max_grad_norm}, Opt: {type(self.optimizer).__name__}, Detect Anomaly: {self.detect_anomaly}")
        if self.has_q_controller: logger.info("Optimizer has Q-Learning Controller.")
        if self.has_grad_stats: logger.info("Optimizer has Gradient Stats Tracking.")


    def _train_epoch(self):
        """Runs a single training epoch."""
        self.model.train() # Set model to training mode
        epoch_loss = 0.0 # Sum of unscaled losses for the epoch average calculation
        optimizer_steps_in_epoch = 0
        micro_step_count_cycle = 0 # Tracks micro-steps within the current accumulation cycle
        total_loss_accum_cycle = 0.0 # Tracks loss accumulated in the current cycle for logging
        total_micro_batches_processed_epoch = 0 # Count total micro-batches processed

        approx_total_optim_steps = None
        total_micro_batches_estimate = None # Estimate of micro-batches in the epoch
        try:
            # Estimate total micro-batches (might be inaccurate for IterableDataset)
            if hasattr(self.train_loader.dataset, '__len__') and len(self.train_loader.dataset) > 0:
                 num_samples_total = len(self.train_loader.dataset)
                 loader_batch_size = self.train_loader.batch_size if self.train_loader.batch_size is not None else 1
                 # Calculate total micro-batches expected across all ranks
                 # Note: DistributedSampler with drop_last=True will handle uneven division.
                 # The effective number of samples considered is floor(total / world_size) * world_size
                 effective_samples_per_rank = num_samples_total // self.world_size
                 total_micro_batches_estimate = effective_samples_per_rank // loader_batch_size
                 if total_micro_batches_estimate > 0 and self.grad_accum_steps > 0:
                     # Calculate expected optimizer steps
                     approx_total_optim_steps = total_micro_batches_estimate // self.grad_accum_steps
                     if approx_total_optim_steps <= 0: approx_total_optim_steps = 1
                     logger.debug(f"Rank {self.rank}: Estimated total micro-batches per epoch/rank: {total_micro_batches_estimate}")
                     logger.debug(f"Rank {self.rank}: Estimated total optimizer steps per epoch: {approx_total_optim_steps}")
                 else:
                     logger.warning(f"Rank {self.rank}: Could not reliably estimate optimizer/micro steps based on sample/batch size. Setting estimates to None.")
                     approx_total_optim_steps = None
                     total_micro_batches_estimate = None
            else:
                logger.warning(f"Rank {self.rank}: DataLoader dataset lacks __len__ or is empty. Progress bar total may be inaccurate.")
                approx_total_optim_steps = None
                total_micro_batches_estimate = None
        except TypeError:
            logger.warning(f"Rank {self.rank}: TypeError getting dataset len. Progress bar total may be inaccurate.")
            approx_total_optim_steps = None
            total_micro_batches_estimate = None
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error calculating approx steps: {e}")
            approx_total_optim_steps = None
            total_micro_batches_estimate = None

        disable_tqdm = not self.is_main
        # TQDM wraps the dataloader, iterates micro-batches. Total = estimated micro-batches.
        batch_iterator = tqdm(self.train_loader,
                              # Description still shows optimizer step progress for context
                              desc=f"Epoch {self.current_epoch + 1} | Opt Step 0/{(approx_total_optim_steps if approx_total_optim_steps else '?')}",
                              disable=disable_tqdm,
                              total=total_micro_batches_estimate, # Total is micro-batches now
                              unit="batch", # Unit is micro-batches
                              dynamic_ncols=True,
                              ncols=100,
                              leave=False # Keep bar on screen until epoch finishes
                             )

        self.optimizer.zero_grad(set_to_none=True) # Initial zero grad

        for i, batch_data in enumerate(batch_iterator):
            total_micro_batches_processed_epoch = i + 1 # Count every micro-batch attempt
            micro_step_count_cycle += 1
            is_last_micro_step = (micro_step_count_cycle % self.grad_accum_steps == 0)
            should_optimizer_step = is_last_micro_step

            # DDP gradient sync context
            sync_context = contextlib.nullcontext()
            if self.world_size > 1 and isinstance(self.model, DistributedDataParallel) and not should_optimizer_step:
                 sync_context = self.model.no_sync()

            # Anomaly detection context
            anomaly_context = torch.autograd.detect_anomaly(check_nan=True) if self.detect_anomaly else contextlib.nullcontext()

            loss_value = None; loss = None; logits = None

            try:
                with sync_context, anomaly_context:
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        context, targets = batch_data
                    else:
                        logger.warning(f"Rank {self.rank}: Skipping unexpected batch format at micro_batch index {i}. Type: {type(batch_data)}")
                        if is_last_micro_step: micro_step_count_cycle = 0 # Reset cycle if last step was skipped
                        continue

                    context = context.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    batch_size, ctx_len = context.shape
                    if ctx_len == 0 or batch_size == 0:
                        logger.warning(f"Rank {self.rank}: Skipping empty batch context shape {context.shape} at micro_batch index {i}")
                        if is_last_micro_step: micro_step_count_cycle = 0
                        continue

                    decoder_input_seq = context

                    # --- Forward Pass ---
                    with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                        logits = self.model(byte_seq=context, target_byte_seq=decoder_input_seq)

                        if logits is None: raise ValueError("Model returned None logits")
                        if logits.shape[0]!=batch_size or logits.shape[1]!=ctx_len or logits.shape[2]!=256:
                             raise ValueError(f"Logits shape mismatch: expected B={batch_size}, L={ctx_len}, V=256, got {logits.shape}")
                        if not torch.isfinite(logits).all():
                             num_nan = torch.isnan(logits).sum().item(); num_inf = torch.isinf(logits).sum().item()
                             logger.warning(f"NaN/Inf detected in model logits (NaN:{num_nan}, Inf:{num_inf}). Replacing.")
                             logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0) # Replace with reasonable values

                        # --- Loss Computation ---
                        loss = IntegratedHyperHAKMEMModel.compute_loss(logits, targets, smoothing=0.1)
                        if loss is None or not torch.isfinite(loss):
                             raise ValueError(f"NaN/Inf/None loss computed: {loss}")

                        # Scale loss for backward pass
                        loss_value = loss / self.grad_accum_steps

                    # --- Backward Pass ---
                    # The scaler automatically handles the scaling of the loss
                    self.scaler.scale(loss_value).backward()

                # --- Accumulate Loss (Use Unscaled) ---
                current_step_loss = loss.item()
                if not np.isfinite(current_step_loss):
                    logger.warning(f"Rank {self.rank}: Non-finite loss ({current_step_loss}) recorded for accumulation at micro-step {micro_step_count_cycle}. Replacing with 0.")
                    current_step_loss = 0.0
                total_loss_accum_cycle += current_step_loss
                epoch_loss += current_step_loss # Add unscaled loss to epoch total

            except Exception as batch_ex:
                logger.error(f"Error in micro-step {micro_step_count_cycle} (Batch Index {i}, Global Step {self.global_step}) Rank {self.rank}: {batch_ex}", exc_info=False) # Log short traceback
                logger.warning(f"Rank {self.rank}: Zeroing gradients and resetting accumulation cycle due to error.")
                try: self.optimizer.zero_grad(set_to_none=True)
                except Exception as zero_ex: logger.error(f"Rank {self.rank}: Failed to zero grads after error: {zero_ex}")
                try: del loss_value, loss, logits, context, targets, decoder_input_seq # Clean up memory
                except NameError: pass
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                total_loss_accum_cycle = 0.0; micro_step_count_cycle = 0
                should_optimizer_step = False # Prevent optimizer step after this micro-batch error
                # Important: Still need to let the loop reach the end of the cycle if needed
                continue


            # --- Optimizer Step Phase ---
            if should_optimizer_step:
                avg_loss_cycle = total_loss_accum_cycle / self.grad_accum_steps if self.grad_accum_steps > 0 else total_loss_accum_cycle

                optimizer_step_skipped = False
                unclipped_norm_val = 0.0
                has_nonfinite_grad = False
                is_clipped = False
                clip_ratio = None

                # --- Unscale Gradients ---
                try:
                    self.scaler.unscale_(self.optimizer)
                except RuntimeError as unscale_err:
                    logger.error(f"Error during scaler.unscale_ at step {self.global_step}: {unscale_err}. Skipping optimizer step.")
                    has_nonfinite_grad = True
                    unclipped_norm_val = float('inf')
                    optimizer_step_skipped = True
                except Exception as e:
                     logger.error(f"Unexpected error during scaler.unscale_ at step {self.global_step}: {e}. Skipping optimizer step.")
                     has_nonfinite_grad = True
                     unclipped_norm_val = float('inf')
                     optimizer_step_skipped = True

                # --- Calculate Pre-Clip Norm and Clip (only if unscale was successful) ---
                if not optimizer_step_skipped:
                    params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
                    if params_with_grad:
                        total_norm_unclipped_sq = torch.tensor(0.0, device=self.device)
                        param_with_nan_inf_grad = None
                        for p in params_with_grad:
                            try:
                                grad_float = p.grad.detach().float()
                                if not torch.isfinite(grad_float).all():
                                     logger.warning(f"Non-finite values detected in gradient for param shape {p.shape}.")
                                     param_with_nan_inf_grad = p # Store ref to param
                                     total_norm_unclipped_sq = torch.tensor(float('inf'), device=self.device)
                                     break # Stop if any grad is non-finite

                                param_norm_sq = torch.sum(grad_float.pow(2))
                                if torch.isfinite(param_norm_sq):
                                    total_norm_unclipped_sq += param_norm_sq
                                else:
                                    logger.warning(f"Non-finite norm component calculated for param shape {p.shape} grad.")
                                    param_with_nan_inf_grad = p
                                    total_norm_unclipped_sq = torch.tensor(float('inf'), device=self.device)
                                    break # Stop if any component is non-finite
                            except Exception as norm_ex:
                                logger.error(f"Error calculating norm component for param grad (shape {p.shape}): {norm_ex}")
                                total_norm_unclipped_sq = torch.tensor(float('inf'), device=self.device)
                                break

                        if torch.isfinite(total_norm_unclipped_sq):
                            total_norm_unclipped = torch.sqrt(total_norm_unclipped_sq)
                            unclipped_norm_val = total_norm_unclipped.item()
                        else:
                            total_norm_unclipped = torch.tensor(float('inf'), device=self.device)
                            unclipped_norm_val = float('inf')

                        has_nonfinite_grad = not torch.isfinite(total_norm_unclipped).item()
                    else:
                        logger.warning(f"Rank {self.rank}: No gradients found for norm calculation at optimizer step {self.global_step}.")
                        unclipped_norm_val = 0.0
                        has_nonfinite_grad = False # No grads is not a non-finite grad scenario

                    # --- Check for non-finite gradients AFTER unscale ---
                    if has_nonfinite_grad:
                        logger.warning(f"Rank {self.rank}: Non-finite total gradient norm ({unclipped_norm_val}) detected AFTER unscale_ at step {self.global_step}. Skipping optimizer step.")
                        optimizer_step_skipped = True
                        if self.has_grad_stats:
                            self.optimizer.gradient_stats.record_gradient(unclipped_norm_val, clipped=False) # Record the non-finite attempt
                    else:
                        # --- Actual Gradient Clipping (if norm is finite and exceeds max) ---
                        if self.max_grad_norm > 0 and unclipped_norm_val > self.max_grad_norm:
                            is_clipped = True
                            clip_ratio = self.max_grad_norm / (unclipped_norm_val + 1e-6)
                            for p in params_with_grad:
                                if p.grad is not None: # Re-check grad exists before clipping
                                    p.grad.detach().mul_(clip_ratio)
                        # Record gradient stats (finite grad path)
                        if self.has_grad_stats:
                            self.optimizer.gradient_stats.record_gradient(unclipped_norm_val, is_clipped, clip_ratio)

                        # --- Provide loss to optimizer for Q-controller (finite grad path) ---
                        if self.has_q_controller:
                            self.optimizer.set_current_loss(avg_loss_cycle)
                            group = self.optimizer.param_groups[0]
                            q_state = self.optimizer.q_controller.get_state(lr=group['lr'], momentum=group['momentum'], grad_norm=unclipped_norm_val, loss=avg_loss_cycle)
                            if self.optimizer.q_controller.prev_state is not None and self.optimizer.q_controller.prev_action is not None and q_state is not None:
                                reward = self.optimizer.q_controller.compute_reward(avg_loss_cycle, self.optimizer.q_controller.prev_loss, unclipped_norm_val)
                                if np.isfinite(reward):
                                    self.optimizer.q_controller.update(self.optimizer.q_controller.prev_state, self.optimizer.q_controller.prev_action, reward, q_state)
                                self.optimizer.q_controller.prev_loss = avg_loss_cycle
                            elif self.optimizer.q_controller.prev_state is None: # First step
                                self.optimizer.q_controller.prev_loss = avg_loss_cycle
                            # Choose action for *next* step now
                            if q_state is not None:
                                next_q_action = self.optimizer.q_controller.choose_action(q_state)
                                self.optimizer.q_controller.prev_state = q_state
                                self.optimizer.q_controller.prev_action = next_q_action # Store for next optimizer.step()
                            else: # If state calculation failed
                                self.optimizer.q_controller.prev_state = None; self.optimizer.q_controller.prev_action = None

                        # --- Optimizer Step (finite grad path) ---
                        self.scaler.step(self.optimizer)
                        # scaler.step() returns None if it skipped the update due to inf/nan grads.

                # --- Update Scaler (CRUCIAL: Called AFTER potential step/skip) ---
                self.scaler.update()

                # --- Zero Gradients for Next Cycle ---
                self.optimizer.zero_grad(set_to_none=True)

                # --- Record Step Stats (including skip info) ---
                grad_stats = {}
                if self.has_grad_stats:
                    grad_stats = self.optimizer.gradient_stats.record_step(self.global_step, skipped=optimizer_step_skipped)

                # --- Post-Step Actions (Logging, Checkpointing, Incrementing) ---
                if optimizer_step_skipped:
                    if self.is_main: logger.warning(f"Optimizer step {self.global_step} skipped due to non-finite gradients.")
                    # Note: global_step is NOT incremented for skipped steps
                else:
                    # Increment global step only if optimizer step was successful
                    optimizer_steps_in_epoch += 1 # Increment counter for *this* epoch
                    self.global_step += 1

                    # --- Update TQDM Description & Postfix ---
                    if self.is_main:
                        # Update description with the current optimizer step count
                        batch_iterator.set_description(f"Epoch {self.current_epoch + 1} | Opt Step {optimizer_steps_in_epoch}/{(approx_total_optim_steps if approx_total_optim_steps else '?')}")
                        current_lr = self.optimizer.param_groups[0]['lr']
                        clipped_norm_val = min(unclipped_norm_val, self.max_grad_norm) if self.max_grad_norm > 0 and is_clipped else unclipped_norm_val
                        batch_iterator.set_postfix(Loss=f"{avg_loss_cycle:.3f}", LR=f"{current_lr:.3e}", Grad=f"{clipped_norm_val:.2f}", refresh=False) # Update less frequently

                    # --- Logging (only if step wasn't skipped) ---
                    if self.is_main and self.global_step % self.log_interval == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        current_mom = self.optimizer.param_groups[0].get('momentum', 0.0)
                        q_info = {}
                        if self.has_q_controller: q_info = self.optimizer.get_q_info()
                        clipped_norm_val = min(unclipped_norm_val, self.max_grad_norm) if self.max_grad_norm > 0 and is_clipped else unclipped_norm_val

                        log_data = {
                            "Epoch": self.current_epoch + 1, "Step": self.global_step,
                            "Train Loss (Cycle Avg)": avg_loss_cycle,
                            "Learning Rate": current_lr, "Momentum": current_mom,
                            "Grad Norm (Actual Before Step)": clipped_norm_val, # Actual norm value after clipping, before step
                            "Grad Norm Max Recorded (Unclipped)": grad_stats.get('max_gradient', 0.0), # Max unclipped norm seen in step
                            "Clip %": grad_stats.get('clip_percentage', 0.0),
                            "Avg Clip Ratio": grad_stats.get('clip_ratio_avg', 0.0),
                            "NonFinite Grads": grad_stats.get('non_finite_grads', 0),
                            "AMP Scale": self.scaler.get_scale(),
                        }
                        log_data.update({f"Q_{k}": v for k, v in q_info.items() if k != 'last_action'})
                        if 'last_action' in q_info and q_info['last_action'] is not None:
                             log_data["Q_LR_Scale"] = q_info['last_action'].get('lr_scale', 1.0)
                             log_data["Q_Mom_Scale"] = q_info['last_action'].get('momentum_scale', 1.0)

                        log_msg = (f"Step {self.global_step} | Ep {self.current_epoch + 1} Opt {optimizer_steps_in_epoch} | Loss(avg): {avg_loss_cycle:.4f} | "
                                   f"LR: {current_lr:.3e} | Mom: {current_mom:.3f} | "
                                   f"GradNorm(Actual): {log_data['Grad Norm (Actual Before Step)']:.2f} | MaxUnclip: {log_data['Grad Norm Max Recorded (Unclipped)']:.2f} | Clip%: {log_data['Clip %']:.1f} | "
                                   f"NF Grads: {log_data['NonFinite Grads']} | Scale: {log_data['AMP Scale']:.0f}")
                        if self.has_q_controller:
                             log_msg += f" | QScale(LR/M): {log_data.get('Q_LR_Scale', 0):.2f}/{log_data.get('Q_Mom_Scale', 0):.2f}"
                             log_msg += f" | QEps: {log_data.get('Q_epsilon', 0):.3f}"
                        logger.info(log_msg)

                        if self.wandb_run:
                            try: wandb.log({"train": log_data}, step=self.global_step)
                            except Exception as wb_err: logger.error(f"WandB logging failed: {wb_err}")

                    # --- Save Checkpoint (only if step wasn't skipped) ---
                    if self.is_main and self.save_interval > 0 and self.global_step % self.save_interval == 0:
                        self._save_checkpoint(is_intermediate=True, metrics={'loss': avg_loss_cycle})

                # --- Reset accumulation cycle state AFTER handling the step ---
                total_loss_accum_cycle = 0.0
                micro_step_count_cycle = 0

        # --- End of Epoch ---
        if self.is_main and hasattr(batch_iterator, 'close'):
            batch_iterator.close()
        # Use the total number of micro-batches processed for epoch loss average
        avg_epoch_loss = epoch_loss / total_micro_batches_processed_epoch if total_micro_batches_processed_epoch > 0 else 0.0
        return avg_epoch_loss


    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Runs validation loop and returns metrics."""
        if self.val_loader is None: return {}
        # Validation can run on all ranks if needed, but only main logs/saves
        # if self.world_size > 1: torch.distributed.barrier() # Sync before validation

        self.model.eval()
        total_loss = 0.0; num_batches = 0; num_samples = 0
        approx_total_val_batches = None
        try:
            if hasattr(self.val_loader.dataset, '__len__') and len(self.val_loader.dataset) > 0:
                num_samples_total_val = len(self.val_loader.dataset)
                # Adjust for drop_last=False in validation loader if using DistributedSampler
                val_loader_batch_size = self.val_loader.batch_size if self.val_loader.batch_size is not None else 1
                if isinstance(self.val_loader.sampler, DistributedSampler):
                    # Estimate batches per rank
                    approx_total_val_batches = math.ceil((num_samples_total_val / self.world_size) / val_loader_batch_size)
                elif val_loader_batch_size > 0:
                    approx_total_val_batches = math.ceil(num_samples_total_val / val_loader_batch_size)
                else: approx_total_val_batches = None
            else: approx_total_val_batches = None
        except TypeError: approx_total_val_batches = None

        val_iterator = tqdm(self.val_loader,
                            desc=f"Validation Epoch {self.current_epoch + 1} Rank {self.rank}",
                            disable=not self.is_main, # Only show on main process
                            total=approx_total_val_batches,
                            unit="batch",
                            leave=False)

        batch_losses = []
        for batch_data in val_iterator:
            try:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    context, targets = batch_data
                else: logger.warning(f"Rank {self.rank}: Skipping unexpected validation batch format: {type(batch_data)}"); continue

                context = context.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size, ctx_len = context.shape
                if batch_size == 0 or ctx_len == 0: continue
                decoder_input_seq = context

                with torch.no_grad(), amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # Use model.module if DDP wrapped, otherwise model directly
                    model_to_eval = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
                    logits = model_to_eval(byte_seq=context, target_byte_seq=decoder_input_seq)

                    if logits.shape[0] != batch_size or logits.shape[1] != ctx_len or logits.shape[2] != 256:
                        logger.warning(f"Rank {self.rank}: Validation logits shape mismatch. Expected ({batch_size},{ctx_len},256), got {logits.shape}. Skip.")
                        continue
                    if not torch.isfinite(logits).all():
                        logger.warning(f"Rank {self.rank}: NaN/Inf detected in validation logits. Skip loss.")
                        continue

                    loss = IntegratedHyperHAKMEMModel.compute_loss(logits, targets, smoothing=0.0) # No smoothing for val

                loss_item = loss.item()
                if np.isfinite(loss_item):
                     batch_losses.append(loss_item)
                else: logger.warning(f"Rank {self.rank}: Non-finite validation loss encountered: {loss_item}")

            except Exception as val_ex:
                 logger.error(f"Rank {self.rank} Validation Error: {val_ex}", exc_info=False)
                 continue

        # Aggregate results across DDP ranks
        if self.world_size > 1:
             losses_tensor = torch.tensor(batch_losses, dtype=torch.float64, device=self.device)
             gathered_losses_list = [torch.zeros_like(losses_tensor) for _ in range(self.world_size)] # Pre-allocate list
             try:
                 # all_gather requires a list of tensors as input
                 torch.distributed.all_gather(gathered_losses_list, losses_tensor)
                 all_losses = torch.cat(gathered_losses_list).cpu().tolist()
                 if self.is_main: logger.debug(f"Gathered {len(all_losses)} validation losses from {self.world_size} ranks.")
             except Exception as gather_err:
                 logger.error(f"Rank {self.rank}: Error during validation loss gather: {gather_err}. Using local losses only.")
                 all_losses = batch_losses # Use local data as fallback
        else:
            all_losses = batch_losses

        # Compute final metrics only on the main process
        metrics = {}
        if self.is_main:
            if not all_losses:
                logger.warning("No valid validation losses recorded.")
                avg_loss = float('inf'); perplexity = float('inf')
            else:
                avg_loss = sum(all_losses) / len(all_losses)
                perplexity = float('inf')
                if np.isfinite(avg_loss):
                    try:
                        # Clamp loss before exponentiation to avoid OverflowError
                        clamped_loss = min(avg_loss, 700) # exp(700) is already huge
                        perplexity = math.exp(clamped_loss)
                    except (OverflowError, ValueError): perplexity = float('inf')

            metrics = {'val_loss': avg_loss, 'val_perplexity': perplexity}
            self.last_val_metrics = metrics

            logger.info(f"Validation Epoch {self.current_epoch + 1} | Loss: {metrics['val_loss']:.4f} | Perplexity: {metrics['val_perplexity']:.2f}")
            if self.wandb_enabled and self.wandb_run:
                try: wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": self.current_epoch + 1}, step=self.global_step)
                except Exception as wb_err: logger.error(f"WandB validation logging failed: {wb_err}")

        if hasattr(val_iterator, 'close'): val_iterator.close()
        # if self.world_size > 1: torch.distributed.barrier() # Sync after validation
        return metrics

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict] = None):
        """Saves model, optimizer, scaler, and Q-controller state."""
        if not self.is_main: return # Only main process saves checkpoints
        filename_prefix = "checkpoint"
        if is_intermediate: state_indicator = f"step_{self.global_step}"
        else: state_indicator = f"epoch_{self.current_epoch+1}_final"

        current_metrics = metrics if metrics is not None else self.last_val_metrics
        # Use val_loss if available and finite, otherwise train loss if available and finite
        metric_str = ""
        if current_metrics:
            val_loss = current_metrics.get('val_loss')
            train_loss = current_metrics.get('loss') # Train loss passed for intermediate saves

            if val_loss is not None and np.isfinite(val_loss):
                metric_key = 'val_loss'; metric_val = val_loss
            elif train_loss is not None and np.isfinite(train_loss):
                metric_key = 'loss'; metric_val = train_loss
            else: metric_key = None

            if metric_key:
                if abs(metric_val) < 1e-3 and metric_val != 0.0 : mf = f"{metric_val:.2e}"
                else: mf = f"{metric_val:.3f}"
                metric_str = f"_{metric_key}{mf}"

        state_indicator += metric_str
        filename = f"{filename_prefix}_{state_indicator}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        model_state = self.model.module.state_dict() if isinstance(self.model, DistributedDataParallel) else self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch, 'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'metrics': current_metrics, # Save the metrics dict used for naming
            'amp_enabled': self.use_amp,
            'args': getattr(self, 'args', None) # Save command line args used
        }
        if self.has_q_controller:
            q_state_data = {
                'q_table': self.optimizer.q_controller.q_table,
                'epsilon': self.optimizer.q_controller.epsilon,
                'access_count': self.optimizer.q_controller.q_table_access_count,
                'creation_time': self.optimizer.q_controller.q_table_creation_time,
                'loss_window': list(self.optimizer.q_controller.loss_window), # Save history if needed
                'grad_norm_window': list(self.optimizer.q_controller.grad_norm_window),
                'prev_loss': self.optimizer.q_controller.prev_loss,
                'prev_state': self.optimizer.q_controller.prev_state,
                'prev_action': self.optimizer.q_controller.prev_action,
            }
            checkpoint['q_controller_state'] = q_state_data

        try:
            temp_filepath = filepath + ".tmp"
            torch.save(checkpoint, temp_filepath)
            # Verify save integrity (optional, can slow down saving)
            # _ = torch.load(temp_filepath)
            os.replace(temp_filepath, filepath)
            logger.info(f"Checkpoint saved to {filepath}")

            # Upload to WandB if enabled
            if self.wandb_enabled and self.wandb_run:
                try:
                    # Use policy='live' to upload intermediate checkpoints as well
                    wandb.save(filepath, base_path=self.checkpoint_dir, policy="live")
                    logger.info(f"Checkpoint {filepath} uploaded to WandB.")
                except Exception as wb_save_err:
                    logger.error(f"Failed to save checkpoint to WandB: {wb_save_err}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}", exc_info=True)
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError: pass

    def load_checkpoint(self, filepath: str):
        """Loads state from a checkpoint file."""
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return 0

        try:
            # Load checkpoint to CPU first to avoid GPU memory spike if model is large
            checkpoint = torch.load(filepath, map_location='cpu')
            model_to_load = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

            # Handle potential mismatch between saved state and current model def
            # Allow partial loading by setting strict=False
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible_keys.missing_keys: logger.warning(f"Missing keys loading model state_dict: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys: logger.warning(f"Unexpected keys loading model state_dict: {incompatible_keys.unexpected_keys}")

            if 'optimizer_state_dict' in checkpoint:
                 try:
                     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                     # Move optimizer states back to the correct device
                     for state in self.optimizer.state.values():
                         for k, v in state.items():
                             if isinstance(v, torch.Tensor):
                                 try: state[k] = v.to(self.device)
                                 except Exception as e_state: logger.warning(f"Could not move optimizer state key {k} to device {self.device}: {e_state}")
                 except Exception as optim_ex: logger.warning(f"Could not load optimizer state properly: {optim_ex}. Optimizer state may be reset.")
            else: logger.warning("Optimizer state not found in checkpoint.")

            saved_amp_enabled = checkpoint.get('amp_enabled', False)
            if self.use_amp:
                if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and saved_amp_enabled:
                     try: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                     except Exception as scaler_ex: logger.warning(f"Could not load AMP scaler state: {scaler_ex}. Scaler state reset.")
                elif saved_amp_enabled: logger.warning("AMP scaler state missing but expected from checkpoint. Using fresh state.")
                else: logger.info("AMP scaler state not found or not applicable. Using fresh scaler state.")
            elif saved_amp_enabled: logger.warning("Checkpoint has AMP state, but AMP is currently disabled.")

            start_epoch = checkpoint.get('epoch', -1) + 1 # Start from next epoch
            self.global_step = checkpoint.get('global_step', 0)
            self.current_epoch = start_epoch -1 if start_epoch > 0 else 0 # Store the completed epoch index
            self.last_val_metrics = checkpoint.get('metrics')
            logger.info(f"Metrics loaded from checkpoint: {self.last_val_metrics}")

            if self.has_q_controller and 'q_controller_state' in checkpoint:
                q_state = checkpoint['q_controller_state']
                try:
                    self.optimizer.q_controller.q_table = q_state.get('q_table', {})
                    self.optimizer.q_controller.epsilon = q_state.get('epsilon', self.optimizer.q_controller.epsilon)
                    # Ensure keys are tuples if needed (JSON might convert them to strings)
                    q_access = q_state.get('access_count', {})
                    q_create = q_state.get('creation_time', {})
                    self.optimizer.q_controller.q_table_access_count = {eval(k) if isinstance(k, str) and k.startswith('(') else k: v for k, v in q_access.items()}
                    self.optimizer.q_controller.q_table_creation_time = {eval(k) if isinstance(k, str) and k.startswith('(') else k: v for k, v in q_create.items()}
                    # Restore history/state
                    self.optimizer.q_controller.loss_window = deque(q_state.get('loss_window', []), maxlen=self.optimizer.q_controller.loss_window.maxlen)
                    self.optimizer.q_controller.grad_norm_window = deque(q_state.get('grad_norm_window', []), maxlen=self.optimizer.q_controller.grad_norm_window.maxlen)
                    self.optimizer.q_controller.prev_loss = q_state.get('prev_loss')
                    self.optimizer.q_controller.prev_state = q_state.get('prev_state')
                    self.optimizer.q_controller.prev_action = q_state.get('prev_action')
                    logger.info("Q-Controller state loaded.")
                except Exception as q_load_err: logger.warning(f"Could not load Q-Controller state: {q_load_err}. Q-Controller state may be reset.")
            elif self.has_q_controller: logger.warning("Q-Controller active, but state not found in checkpoint.")

            # Load args from checkpoint if available, compare with current args
            if 'args' in checkpoint and checkpoint['args']:
                loaded_args_dict = vars(checkpoint['args'])
                current_args_dict = vars(getattr(self, 'args', argparse.Namespace())) # Use current args if available
                if current_args_dict:
                    mismatched_args = {}
                    # Check keys present in loaded args
                    for key, loaded_val in loaded_args_dict.items():
                        current_val = current_args_dict.get(key, '<<Missing Current>>')
                        # Check for significant differences, ignore some volatile args like resume path or local_rank
                        if key not in ['resume', 'local_rank', 'wandb'] and current_val != loaded_val:
                            mismatched_args[key] = {'loaded': loaded_val, 'current': current_val}
                    # Check for keys present only in current args
                    for key, current_val in current_args_dict.items():
                         if key not in loaded_args_dict and key not in ['resume', 'local_rank', 'wandb']:
                              mismatched_args[key] = {'loaded': '<<Missing Loaded>>', 'current': current_val}

                    if mismatched_args:
                         logger.warning(f"Argument mismatch between checkpoint and current run: {mismatched_args}")
                    else: logger.info("Checkpoint arguments match current arguments.")
                else: logger.warning("Could not compare checkpoint args: current args not found in trainer.")

            logger.info(f"Successfully loaded checkpoint '{filepath}'. Resuming training from Epoch {start_epoch} (Global Step {self.global_step})")
            # Move model back to device *after* loading state dict
            self.model.to(self.device)
            return start_epoch
        except FileNotFoundError: logger.error(f"Checkpoint file not found during load: {filepath}"); return 0
        except Exception as e: logger.error(f"Failed loading checkpoint '{filepath}': {e}", exc_info=True); return 0


    def train(self, epochs: int, start_epoch: int = 0):
        """Main training loop over multiple epochs."""
        self.current_epoch = start_epoch # Set starting epoch
        if self.is_main: logger.info(f"Starting training from epoch {start_epoch + 1}/{epochs} (Global Step: {self.global_step}).")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            if self.is_main: logger.info(f"--- Starting Epoch {epoch + 1}/{epochs} ---")

            # Set Epoch for Samplers/Datasets (Important for DDP Sampler and shuffling)
            if self.world_size > 1:
                 if hasattr(self.train_loader.sampler, 'set_epoch'):
                     self.train_loader.sampler.set_epoch(epoch)
                 if self.val_loader and hasattr(self.val_loader.sampler, 'set_epoch'):
                     self.val_loader.sampler.set_epoch(epoch)
            # Set epoch for dataset if it implements the method (for internal shuffling logic)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader.dataset, 'set_epoch'):
                self.val_loader.dataset.set_epoch(epoch)

            # --- Train one epoch ---
            avg_train_loss = self._train_epoch()
            if self.is_main: logger.info(f"Epoch {epoch + 1} Train Avg Loss: {avg_train_loss:.4f}")

            # --- Validate (only on main process) ---
            val_metrics = self._validate() # Returns metrics dict or {}

            # --- Save Checkpoint (only on main process) ---
            if self.is_main:
                # Pass the validation metrics (or last train loss if no val) for filename
                save_metrics = val_metrics if val_metrics else {'loss': avg_train_loss}
                self._save_checkpoint(is_intermediate=False, metrics=save_metrics)

            # Synchronize all processes before starting the next epoch
            if self.world_size > 1:
                logger.debug(f"Rank {self.rank} entering barrier after epoch {epoch+1}.")
                torch.distributed.barrier()
                logger.debug(f"Rank {self.rank} exited barrier after epoch {epoch+1}.")

        if self.is_main: logger.info("Training finished.")


# =====================================================================
# Argument Parsing and Main Execution Logic (Updated Args)
# =====================================================================

def setup_distributed(local_rank):
    """Initializes Distributed Data Parallel (DDP) if local_rank is provided."""
    if local_rank == -1: # DDP not requested or LOCAL_RANK not set
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ws = 1; rk = 0; is_distributed = False
        if torch.cuda.is_available(): logger.info(f"CUDA available. Using device: {torch.cuda.get_device_name(device)}")
        else: logger.info("CUDA not available. Using CPU.")
    else: # DDP requested
        if not torch.cuda.is_available(): logger.error("DDP requested but CUDA not available."); sys.exit(1)
        if torch.cuda.device_count() <= local_rank:
            logger.error(f"Invalid local_rank {local_rank}. GPUs available: {torch.cuda.device_count()}."); sys.exit(1)

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        required_env_vars = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        if not all(env_var in os.environ for env_var in required_env_vars):
            logger.warning(f"DDP env vars ({required_env_vars}) not fully set. 'torchrun' recommended. Trying 'env://'...")

        try:
            # Default timeout can be short, increase if needed
            timeout = timedelta(minutes=30)
            init_process_group(backend="nccl", init_method="env://", timeout=timeout)
            ws = get_world_size(); rk = get_rank(); is_distributed = True
            logger.info(f"DDP Initialized via env:// | Rank: {rk}/{ws} | Device: {device} ({torch.cuda.get_device_name(device)}) | Timeout: {timeout}")
            torch.distributed.barrier()
        except Exception as e:
            logger.error(f"DDP initialization failed: {e}", exc_info=True)
            if is_initialized(): destroy_process_group()
            sys.exit(1)
    return is_distributed, device, rk, ws


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Integrated HyperHAKMEM Model Training Script (Hyperbolic Attention Version)")

    # --- Data Configuration ---
    dgrp = parser.add_argument_group('Data Configuration')
    dgrp.add_argument("--data_path", type=str, required=True, help="Path to train data (.npy file).")
    dgrp.add_argument("--val_data_path", type=str, default=None, help="Path to validation data (.npy, optional).")
    dgrp.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of training data to use (0.0 to 1.0).")
    dgrp.add_argument("--context_window", type=int, default=256, help="Sequence length for training and decoder PE.")

    # --- Model Architecture (Hyperbolic Attention Focus) ---
    mgrp = parser.add_argument_group('Model Architecture')
    mgrp.add_argument("--local_hidden_size", type=int, default=256, help="Hidden size for Local Encoder/Decoder.")
    mgrp.add_argument("--hyperbolic_embedding_dim", type=int, default=384, help="Dimension of the hyperbolic space & attention.")
    mgrp.add_argument("--num_hyperbolic_layers", type=int, default=8, help="Number of hyperbolic attention layers.")
    mgrp.add_argument("--num_hyperbolic_heads", type=int, default=8, help="Number of heads in hyperbolic attention.")
    mgrp.add_argument("--decoder_memory_dim", type=int, default=768, help="Dimension of the final memory fed to the decoder.")
    mgrp.add_argument("--n_gram_sizes", type=int, nargs='+', default=[3, 4], help="N-gram sizes for Local Encoder features.")
    mgrp.add_argument("--n_gram_vocab_size", type=int, default=30000, help="Vocabulary size for N-gram hashing.")
    mgrp.add_argument("--dropout", type=float, default=0.15, help="Dropout rate for various layers.")
    mgrp.add_argument("--no_hierarchical_decoder", action="store_true", help="Use flat (256 output) decoder head instead of hierarchical.")
    mgrp.add_argument("--curvature", type=float, default=0.8, help="Hyperbolic curvature 'c' (must be > 0).")
    mgrp.add_argument("--clipping_radius", type=float, default=0.99, help="Clipping radius within Poincare ball (should be < 1.0).") # Default slightly inside

    # --- Training Parameters ---
    tgrp = parser.add_argument_group('Training Parameters')
    tgrp.add_argument("--batch_size", type=int, default=16, help="Global batch size (distributed across GPUs).")
    tgrp.add_argument("--learning_rate", type=float, default=8e-4, help="Base learning rate for the optimizer.")
    tgrp.add_argument("--epochs", type=int, default=12, help="Total number of training epochs.")
    tgrp.add_argument("--grad_accum_steps", type=int, default=2, help="Number of steps to accumulate gradients before optimizer step.")
    tgrp.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    tgrp.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization).")
    tgrp.add_argument("--checkpoint_dir", type=str, default="./integrated_checkpoints_hyperbolic", help="Directory to save checkpoints.")
    tgrp.add_argument("--log_interval", type=int, default=10, help="Log training stats every N optimizer steps.")
    tgrp.add_argument("--save_interval", type=int, default=1000, help="Save intermediate checkpoint every N optimizer steps (0=disable).")
    tgrp.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")

    # --- Optimizer Q-Learning Control ---
    qgrp = parser.add_argument_group('Optimizer Q-Learning Control')
    qgrp.add_argument("--q_learning_rate", type=float, default=0.01, help="Learning rate (alpha) for the Q-table updates.")
    qgrp.add_argument("--q_discount", type=float, default=0.95, help="Discount factor (gamma) for future rewards in Q-learning.")
    qgrp.add_argument("--q_epsilon", type=float, default=0.2, help="Initial epsilon for epsilon-greedy exploration in Q-learning.")
    qgrp.add_argument("--q_epsilon_decay", type=float, default=0.9998, help="Decay rate for epsilon.")
    qgrp.add_argument("--q_min_epsilon", type=float, default=0.01, help="Minimum value for epsilon.")

    # --- Miscellaneous ---
    msgrp = parser.add_argument_group('Miscellaneous')
    msgrp.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    msgrp.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    msgrp.add_argument("--wandb_project", type=str, default="bytropix-integrated-hyperbolic", help="WandB project name.")
    msgrp.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")
    msgrp.add_argument("--num_workers", type=int, default=2, help="Number of dataloader worker processes (set 0 for main process loading).")
    msgrp.add_argument("--detect_anomaly", action="store_true", help="Enable torch.autograd.detect_anomaly() for debugging.")
    # Automatically detect local rank from environment variable set by torchrun/slurm
    msgrp.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), help=argparse.SUPPRESS)

    args = parser.parse_args()

    if platform.system() == "Windows" and args.num_workers > 0:
         logger.warning("Using num_workers > 0 on Windows can sometimes cause issues. If errors occur, try setting --num_workers 0.")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args

# =====================================================================
# Worker Initialization Function (Top Level for Pickling)
# =====================================================================
def seed_worker(worker_id, base_seed, rank_offset):
    """Sets the random seed for a DataLoader worker."""
    worker_seed = base_seed + rank_offset + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # logger.debug(f"Worker {worker_id} seeded with {worker_seed}")


# =====================================================================
# Main Execution Logic
# =====================================================================
def main():
    """Main function to parse args, setup, and run training."""
    args = parse_arguments()

    # Initial minimal logging config until rank is known
    temp_is_main = args.local_rank == -1 or args.local_rank == 0
    initial_log_level = logging.INFO if temp_is_main else logging.WARNING
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    logging.basicConfig(level=initial_log_level, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)
    logger.info("Initial logging configuration set.")

    # Setup DDP or single device
    ddp_active, device, rank, world_size = setup_distributed(args.local_rank)

    # Reconfigure logging based on actual rank
    am_main_process = is_main_process() # Re-check after DDP init
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.getLogger().setLevel(log_level)
    if am_main_process:
        log_filename = os.path.join(args.checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_r{rank}.log")
        try:
            fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
            log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
            fh.setFormatter(log_formatter)
            logging.getLogger().addHandler(fh)
            logger.info(f"File logging enabled: {log_filename}")
        except Exception as e: logger.error(f"Failed to configure file logging: {e}")

    logger.info("="*60 + f"\n--- Integrated HyperHAKMEM Run (Hyperbolic Attention) ---")
    logger.info(f"Rank: {rank}/{world_size} | Device: {device} | DDP Active: {ddp_active} | Is Main Process: {am_main_process}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"System Info: OS={platform.system()}/{platform.release()}, Python={sys.version.split()[0]}, Torch={torch.__version__}, CUDA Available={torch.cuda.is_available()}, CUDA Version={torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n" + "="*60)

    # Set seeds for reproducibility
    seed = args.seed # Base seed
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + rank)
    logger.info(f"Set random seed to {seed + rank} for Rank {rank}")

    # Initialize WandB if requested
    use_wandb = args.wandb and am_main_process and WANDB_AVAILABLE
    if use_wandb:
        try:
            # Sanitize config for wandb (convert lists to tuples)
            cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in vars(args).items()}
            run_name = f"HypAttn_H{args.hyperbolic_embedding_dim}_L{args.num_hyperbolic_layers}_B{args.batch_size}_LR{args.learning_rate:.1e}"
            wandb.init(project=args.wandb_project, config=cfg, name=run_name, resume="allow")
            logger.info(f"WandB Initialized: Project='{args.wandb_project}', Run Name='{run_name}'")
        except Exception as e: logger.warning(f"Wandb init failed: {e}. Disabling."); use_wandb = False

    # Initialize Datasets
    train_dataset = None; val_dataset = None
    try:
        logger.info(f"Initializing training dataset: {args.data_path}")
        train_dataset = ByteIterableDataset(args.data_path, context_size=args.context_window, data_fraction=args.data_fraction)
        train_dataset.set_seed(seed)

        if args.val_data_path:
            if os.path.exists(args.val_data_path):
                logger.info(f"Initializing validation dataset: {args.val_data_path}")
                val_dataset = ByteIterableDataset(args.val_data_path, context_size=args.context_window, data_fraction=1.0)
                val_dataset.set_seed(seed)
            else: logger.warning(f"Validation data path specified but not found: {args.val_data_path}")
        else: logger.info("No validation dataset specified.")
    except Exception as e:
        logger.error(f"Fatal Error: Failed to initialize datasets: {e}", exc_info=True)
        if ddp_active: destroy_process_group()
        sys.exit(1)

    # Configure DataLoader
    if world_size <= 0: logger.error("World size <= 0. Exiting."); sys.exit(1)
    if args.batch_size % world_size != 0: logger.warning(f"Global BS {args.batch_size} not divisible by world size {world_size}.")
    batch_size_per_gpu = max(1, args.batch_size // world_size)
    effective_bs = batch_size_per_gpu * world_size * args.grad_accum_steps
    logger.info(f"Batch Config: Global Effective BS={effective_bs}, Per GPU Micro BS={batch_size_per_gpu}, Accum Steps={args.grad_accum_steps}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=seed, drop_last=True) if ddp_active else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if ddp_active and val_dataset else None

    use_persistent_workers = (args.num_workers > 0) and (platform.system() != 'Windows')

    # Create partial functions for worker init (Fix for Windows pickling)
    train_worker_init_fn = functools.partial(seed_worker, base_seed=seed, rank_offset=rank * 100)
    val_worker_init_fn = functools.partial(seed_worker, base_seed=seed, rank_offset=rank * 100 + 1) # Slightly different offset for validation

    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=train_worker_init_fn, # Use partial function
                              persistent_workers=use_persistent_workers,
                              shuffle=False) # Shuffle MUST be False with IterableDataset & DistributedSampler
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False,
                            worker_init_fn=val_worker_init_fn, # Use partial function
                            persistent_workers=use_persistent_workers,
                            shuffle=False) if val_dataset else None

    # Initialize Model
    try:
        # Extract args relevant to the (updated) model signature
        sig = inspect.signature(IntegratedHyperHAKMEMModel.__init__)
        model_args_names = [p.name for p in sig.parameters.values() if p.name != 'self']
        model_cfg = {k: v for k, v in vars(args).items() if k in model_args_names}
        # Set boolean flags based on args
        model_cfg['use_hierarchical_decoder'] = not args.no_hierarchical_decoder
        model_cfg['use_amp'] = not args.no_amp
        # Remove irrelevant args if they sneak in
        model_cfg.pop('projection_method', None)

        model = IntegratedHyperHAKMEMModel(**model_cfg).to(device)
        if am_main_process:
            tp = sum(p.numel() for p in model.parameters()); ttp = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model Parameter Count: Total={tp:,}, Trainable={ttp:,}")
    except Exception as model_ex:
        logger.error(f"Fatal Error: Model initialization failed: {model_ex}", exc_info=True)
        if ddp_active: destroy_process_group()
        sys.exit(1)

    # Wrap model with DDP if needed
    if ddp_active:
        # find_unused_parameters can be expensive, set False if sure no parameters are unused
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        logger.info(f"Model wrapped with DDP on Rank {rank} (find_unused=False).")
        torch.distributed.barrier()

    # Initialize Optimizer
    q_cfg = {"learning_rate": args.q_learning_rate, "discount": args.q_discount, "epsilon": args.q_epsilon,
             "epsilon_decay": args.q_epsilon_decay, "min_epsilon": args.q_min_epsilon}
    optimizer = HAKMEMEnhancedSGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                  weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
                                  q_learning_config=q_cfg, enable_flow=False)
    logger.info(f"Optimizer '{type(optimizer).__name__}' initialized Rank {rank}, LR={args.learning_rate}, WD={args.weight_decay}")

    # Initialize Trainer
    trainer = Trainer(model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader,
                      grad_accum_steps=args.grad_accum_steps, use_amp=(not args.no_amp), log_interval=args.log_interval,
                      save_interval=args.save_interval, checkpoint_dir=args.checkpoint_dir, wandb_enabled=use_wandb,
                      max_grad_norm=args.max_grad_norm, rank=rank, world_size=world_size,
                      detect_anomaly=args.detect_anomaly)
    trainer.args = args # Store args for saving in checkpoint

    # Load Checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Attempting resume from checkpoint: {args.resume} Rank {rank}")
            start_epoch = trainer.load_checkpoint(args.resume)
            if ddp_active: torch.distributed.barrier() # Ensure all ranks load before proceeding
        else:
             logger.warning(f"Resume checkpoint not found: {args.resume}. Starting fresh.")
             if ddp_active: torch.distributed.barrier()
    else:
        logger.info("No checkpoint specified. Starting fresh training.")
        if ddp_active: torch.distributed.barrier()

    # --- Training Loop ---
    save_final = False
    try:
        trainer.train(args.epochs, start_epoch=start_epoch)
        save_final = True # Mark for final save if training completes normally
    except KeyboardInterrupt:
        logger.info(f"Training interrupted by user (KeyboardInterrupt) Rank {rank}.")
        save_final = True # Attempt to save on interruption
    except Exception as train_ex:
        logger.error(f"Unhandled error during training Rank {rank}: {train_ex}", exc_info=True)
        save_final = True # Attempt to save on error
    finally:
        # Cleanup
        if save_final and am_main_process:
            logger.info("Attempting to save final checkpoint...")
            metrics = getattr(trainer, 'last_val_metrics', None)
            trainer._save_checkpoint(is_intermediate=False, metrics=metrics)

        if ddp_active:
            try:
                destroy_process_group()
                logger.info(f"DDP group destroyed Rank {rank}.")
            except Exception as ddp_destroy_err:
                 logger.error(f"Error destroying DDP group Rank {rank}: {ddp_destroy_err}")

        if use_wandb and wandb is not None and wandb.run:
            try:
                wandb.finish()
                logger.info("WandB run finished.")
            except Exception as wb_finish_err:
                 logger.error(f"Error finishing WandB run: {wb_finish_err}")

    logger.info(f"Script execution finished Rank {rank}.")

if __name__ == "__main__":
    # This check prevents the worker processes spawned by DataLoader on Windows
    # from re-running the main script block.
    # The processes need to import the main module to unpickle the worker_init_fn,
    # but they shouldn't execute main() again.
    # It's generally good practice for multi-processing safety.
    main()
