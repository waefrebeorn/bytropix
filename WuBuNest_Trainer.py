
# -*- coding: utf-8 -*-
"""
WuBu Nesting Model Trainer (v0.02 - Relative Vectors Implemented)

Integrates the WuBu Nesting architecture (from WuBuNesting.py v3.0 described
in the provided design document) into the training framework.

Version 0.02 Changes:
- Implemented relative vector calculation *after* inter-level transform.
- Passed aggregated relative vectors to the next WuBuNestingLevel.
- Corrected syntax issues identified in v0.01.
- Added comments clarifying the current state of geometric nesting vs. info flow.
- Includes full HAKMEM components (Optimizer, Decoder, etc.).

Model Flow Adaptation (v0.02):
1. Input bytes -> HAKMEMBabylonIndex -> Patches
2. Patches -> HAKMEMLocalEncoder -> Initial Euclidean Patch Embeddings
3. Euclidean Embeddings -> Project to Tangent Space Level 0 (using HyperbolicUtils)
4. Process through WuBu Nesting Levels:
    - **Level Input:** Combine v_tangent_in, aggregated relative_vectors_in (from *previous* transition),
                      ld_tangent_in (from *previous* transition), sigma_in (from *previous* level).
    - Intra-level processing (tangent combiner, optional flow F_i). Output: v_tangent_out, ld_param, sigma_param.
    - Store v_tangent_out for final aggregation.
    - **If not last level:**
        - Get boundary tangent points for *this* level.
        - Rotate v_tangent_out, boundary_points, ld_param using R_i.
        - Map rotated vectors using non-rotational T~_i -> v_next_main, v_next_boundaries, ld_next.
        - **Calculate Relative Vectors:** d_{i+1} = v_next_main - v_next_boundaries.
        - Aggregate d_{i+1} based on config -> aggregated_relative_vectors_for_next_level.
        - Prepare inputs for next level:
            - current_tangent_main = v_next_main
            - current_relative_vectors = aggregated_relative_vectors_for_next_level
            - current_ld_tangent = ld_next (expanded)
            - current_sigma = sigma_param (from this level)
5. Aggregate Tangent Space outputs (v_tangent_out) from all levels.
6. Project Aggregated Tangent -> Decoder Memory Dimension.
7. HAKMEMLocalDecoder(Target Sequence, Decoder Memory) -> Output Byte Logits

**Note on Nesting:** While levels process sequentially and pass information (including geometric parameters like curvature, scale, spread, and now relative vectors), the strict mathematical definition of H_i+1 being a sub-manifold *within* H_i is not explicitly enforced by this implementation. The "nesting" is primarily conceptual and informational via the tangent space transitions.
"""

# =====================================================================
# Python Imports and Setup
# =====================================================================
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
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque, defaultdict
import gc
import os
import socket
import platform
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp # Use torch.amp instead of torch.cuda.amp for broader compatibility check
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
import inspect
import string
import hashlib
import functools # Added for worker_init_fn fix

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Setup logger
logger = logging.getLogger("WuBuNestTrainer")
# Basic config for initial setup and potential early errors
# This will be reconfigured later in main based on rank
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
        self.non_finite_grads_in_step = 0
        self.step_stats = {} # Store stats per step if needed
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        """Records info about a gradient before potential clipping."""
        if np.isfinite(original_norm):
            self.total_gradients += 1
            self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
            if clipped:
                self.clipped_gradients += 1
                self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)
        else:
            self.non_finite_grads_in_step += 1 # Count non-finite grads encountered *before* clipping attempt
    def get_step_stats(self) -> dict:
        """Calculates summary statistics for the gradients processed since the last reset."""
        if self.total_gradients == 0 and self.non_finite_grads_in_step == 0:
            # Handle case with no gradients processed
            return {"gradients_clipped": 0, "total_gradients": 0, "clip_ratio_avg": 0.0,
                    "max_gradient": 0.0, "clip_percentage": 0.0, "non_finite_grads": 0}

        total_attempts = self.total_gradients + self.non_finite_grads_in_step
        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100 if self.total_gradients > 0 else 0.0
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0
        return {"gradients_clipped": self.clipped_gradients,
                "total_gradients": self.total_gradients, # Count of finite gradients processed
                "non_finite_grads": self.non_finite_grads_in_step, # Count of non-finite grads *encountered*
                "clip_ratio_avg": avg_clip_ratio,
                "max_gradient": self.max_gradient_norm, # Max norm among finite gradients
                "clip_percentage": clip_percentage}
    def record_step(self, step: int, skipped: bool = False) -> dict:
        """Finalizes stats for a step, stores them, and resets for the next step."""
        stats = self.get_step_stats(); stats['step_skipped'] = skipped
        self.step_stats[step] = stats; self.reset() # Reset counters for the next optimizer step
        return stats

# =====================================================================
# HAKMEM-Inspired Entropy Calculation Helper
# =====================================================================
class HAKMEMEntropyHelper:
    """Calculates Shannon entropy for byte sequences with caching."""
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_cache = {}; self.max_cache_size = max_cache_size
    def _clean_cache(self):
        """Removes older entries if cache exceeds max size."""
        if len(self.entropy_cache) > self.max_cache_size:
            # Remove roughly 20% of the oldest entries
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5)
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                # Check existence before deleting (might have been removed concurrently)
                if k in self.entropy_cache: del self.entropy_cache[k]
    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...], List[int], bytes, torch.Tensor]) -> float:
        """Computes entropy, using cache if possible."""
        cache_key = None; byte_list = []
        # Convert input to a canonical list and cache key (tuple/bytes)
        if isinstance(byte_window, tuple): cache_key = byte_window; byte_list = list(byte_window)
        elif isinstance(byte_window, list): cache_key = tuple(byte_window); byte_list = byte_window
        elif isinstance(byte_window, bytes): cache_key = byte_window; byte_list = list(byte_window)
        elif isinstance(byte_window, np.ndarray): byte_list = byte_window.tolist(); cache_key = tuple(byte_list)
        elif isinstance(byte_window, torch.Tensor): byte_list = byte_window.cpu().byte().tolist(); cache_key = tuple(byte_list)
        else:
            logger.warning(f"Unsupported type for entropy calculation: {type(byte_window)}")
            return 0.0

        # Use cache if available
        if cache_key is not None and cache_key in self.entropy_cache:
            return self.entropy_cache[cache_key]

        # Handle empty sequence
        if not byte_list: return 0.0

        try:
            # Calculate counts and probabilities
            byte_counts = np.bincount(np.array(byte_list, dtype=np.uint8), minlength=256)
            total_bytes = byte_counts.sum()
            if total_bytes == 0: return 0.0 # Should not happen if byte_list is not empty, but safety check
            probs = byte_counts[byte_counts > 0] / total_bytes
            # Calculate entropy using log base 2
            entropy = float(-np.sum(probs * np.log2(probs + EPS))) # Add EPS for log stability
            result = max(0.0, entropy) # Ensure entropy is non-negative

            # Cache the result and clean if necessary
            if cache_key is not None:
                self.entropy_cache[cache_key] = result
                self._clean_cache()
            return result
        except Exception as e:
            logger.warning(f"Error during entropy calculation: {e}"); return 0.0

# =====================================================================
# HAKMEM Babylon Index (Word/Punctuation Based)
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
        # Ensure 1D tensor
        if byte_seq_tensor.dim() != 1: byte_seq_tensor = byte_seq_tensor.flatten()
        if byte_seq_tensor.numel() == 0: return [] # Check again after flatten

        device = byte_seq_tensor.device

        try:
            # Decode bytes to text using UTF-8 with replacement for errors
            text = byte_seq_tensor.cpu().numpy().tobytes().decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error decoding byte tensor (len {byte_seq_tensor.numel()}): {e}. Returning no patches.")
            return []

        patches_with_entropy = []; current_patch_start = 0; in_word = False

        for i, char in enumerate(text):
            is_delimiter = char in self.whitespace_chars or char in self.punctuation_chars

            if is_delimiter:
                # If we were in a word, process the completed word patch
                if in_word:
                    word_str = text[current_patch_start:i]
                    try:
                        # Re-encode the word patch and calculate entropy
                        word_bytes = torch.tensor(list(word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                        if word_bytes.numel() > 0:
                            entropy = self.entropy_helper.compute_entropy(word_bytes)
                            patches_with_entropy.append((word_bytes, entropy))
                    except Exception as enc_e:
                        logger.warning(f"Error encoding word patch '{word_str[:20]}...': {enc_e}")
                    in_word = False # We are no longer in a word

                # Process the delimiter patch
                try:
                    # Re-encode the delimiter character and calculate entropy
                    delim_bytes = torch.tensor(list(char.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                    if delim_bytes.numel() > 0:
                        entropy = self.entropy_helper.compute_entropy(delim_bytes)
                        patches_with_entropy.append((delim_bytes, entropy))
                except Exception as enc_e:
                    logger.warning(f"Error encoding delimiter patch '{char}': {enc_e}")
                current_patch_start = i + 1 # Move start to after the delimiter

            else: # Character is not a delimiter
                # If we weren't in a word before, start a new word patch
                if not in_word:
                    in_word = True
                    current_patch_start = i

        # Handle any trailing word patch after the loop finishes
        if in_word and current_patch_start < len(text):
            trailing_word_str = text[current_patch_start:]
            try:
                trailing_word_bytes = torch.tensor(list(trailing_word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                if trailing_word_bytes.numel() > 0:
                    entropy = self.entropy_helper.compute_entropy(trailing_word_bytes)
                    patches_with_entropy.append((trailing_word_bytes, entropy))
            except Exception as enc_e:
                logger.warning(f"Error encoding trailing word patch '{trailing_word_str[:20]}...': {enc_e}")

        # Filter out any empty patches that might have slipped through
        patches_with_entropy = [(p, e) for p, e in patches_with_entropy if p.numel() > 0]
        return patches_with_entropy

    @torch.no_grad()
    def reset_context(self):
        """Resets the entropy cache."""
        self.entropy_helper.entropy_cache = {}

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
        if not valid_heads: # Should not happen if hidden_size > 0
            num_heads = 1
            logger.warning(f"Using 1 head for CrossAttention size {hidden_size} (no divisors found).")
        elif hidden_size % num_heads != 0:
            num_heads = valid_heads[0] # Choose the largest divisor <= original num_heads
            logger.warning(f"Adjusted CrossAttention heads: {original_num_heads} -> {num_heads} for hidden_size {hidden_size}.")

        self.hidden_size = hidden_size; self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(max(1, self.head_dim)) # Avoid sqrt(0)

        # Layer Norms for stability
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialization
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries (Tensor): Shape (Batch, NumQueries, HiddenSize)
            keys_values (Tensor): Shape (Batch, NumKV, HiddenSize)
            attention_mask (Optional[Tensor]): Boolean mask. Shape (Batch, NumKV) or (Batch, NumQueries, NumKV).
                                              True indicates position should be MASKED (ignored).
        Returns:
            Tensor: Output of cross-attention. Shape (Batch, NumQueries, HiddenSize)
        """
        batch_size, num_queries, _ = queries.size()
        _, seq_len_kv, kv_hidden_size = keys_values.size()

        # Handle empty key/value sequence
        if seq_len_kv == 0:
            return torch.zeros_like(queries)

        if kv_hidden_size != self.hidden_size:
            raise ValueError(f"Key/Value hidden size mismatch: K/V has {kv_hidden_size}, Q expects {self.hidden_size}")

        # Normalize inputs
        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)

        # Project and reshape for multi-head attention
        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v shapes: (Batch, NumHeads, SeqLen, HeadDim)

        # Prepare attention mask for scaled_dot_product_attention (expects boolean, True=Mask)
        attn_mask_sdpa = None
        if attention_mask is not None:
            mask_dtype = torch.bool
            # Ensure mask is boolean
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask > 0 # Assuming non-zero means mask
            # Reshape mask to be broadcastable to (Batch, NumHeads, NumQueries, NumKV)
            if attention_mask.dim() == 2: # Shape (B, Nkv) -> (B, 1, 1, Nkv)
                attn_mask_sdpa = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=mask_dtype)
            elif attention_mask.dim() == 3: # Shape (B, Nq, Nkv) -> (B, 1, Nq, Nkv)
                attn_mask_sdpa = attention_mask.unsqueeze(1).to(dtype=mask_dtype)
            # Dim 4 mask (B, h, Nq, Nkv) can be used directly if provided
            elif attention_mask.dim() == 4:
                attn_mask_sdpa = attention_mask.to(dtype=mask_dtype)
            else:
                logger.warning(f"Ignoring unsupported attention mask shape {attention_mask.shape}. Expected 2D, 3D, or 4D.")

            # Ensure mask is on the correct device and check broadcast compatibility
            if attn_mask_sdpa is not None:
                attn_mask_sdpa = attn_mask_sdpa.to(device=queries.device)
                expected_shape = (batch_size, self.num_heads, num_queries, seq_len_kv)
                try:
                    # Test if shapes are broadcastable
                    torch.broadcast_shapes(attn_mask_sdpa.shape, expected_shape)
                except RuntimeError:
                    logger.warning(f"Attention mask shape {attn_mask_sdpa.shape} not broadcastable to target shape {expected_shape}. Ignoring mask.")
                    attn_mask_sdpa = None # Ignore incompatible mask

        # Use Flash Attention (scaled_dot_product_attention) if available
        use_flash = hasattr(F, 'scaled_dot_product_attention')
        output = None
        if use_flash:
            try:
                # sdpa mask must be boolean, True means mask OUT. Dropout applied internally.
                output = F.scaled_dot_product_attention(q, k, v,
                                                        attn_mask=attn_mask_sdpa,
                                                        dropout_p=self.dropout.p if self.training else 0.0,
                                                        is_causal=False) # Not causal for cross-attention
                # Check for NaNs/Infs immediately after Flash Attention
                if not torch.isfinite(output).all():
                    raise ValueError("Flash Attention produced NaN/Inf values.")
            except Exception as e:
                logger.warning(f"Flash Attention failed: {e}. Falling back to standard attention.", exc_info=False)
                use_flash = False # Disable flash for fallback
                output = None # Ensure output is reset

        # Fallback to standard attention mechanism
        if output is None:
            # Calculate attention scores: (B, h, Nq, Nk)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # Clamp scores for stability before softmax
            scores = torch.clamp(scores, min=-30.0, max=30.0)

            if attn_mask_sdpa is not None:
                # Apply mask: fill masked positions with large negative value
                scores = scores.masked_fill(attn_mask_sdpa, float('-inf'))

            # Calculate probabilities (use float32 for softmax stability)
            attn_probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn_probs = torch.nan_to_num(attn_probs) # Handle potential NaNs after softmax if all scores were -inf
            attn_probs = self.dropout(attn_probs)

            # Calculate weighted sum of values: (B, h, Nq, Vd)
            output = torch.matmul(attn_probs, v)

        # Reshape output back to (Batch, NumQueries, HiddenSize)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        # Final output projection
        output = self.out_proj(output)

        # Final stability check
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in CrossAttention final output. Replacing with zeros.")
            output = torch.nan_to_num(output, nan=0.0)

        return output

# =====================================================================
# HAKMEM-Enhanced Local Encoder
# =====================================================================
class HAKMEMLocalEncoder(nn.Module):
    """Encodes individual patches using byte embeddings, optional N-grams, and a Transformer."""
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1, n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Local Encoder hidden_size must be positive.")
        self.hidden_size = hidden_size

        # Byte Embeddings
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))

        # N-gram Embeddings (Optional)
        self.n_gram_sizes = sorted(list(set(s for s in n_gram_sizes if isinstance(s, int) and s > 0)))
        self.n_gram_vocab_size = n_gram_vocab_size
        self.n_gram_embeddings = None
        self.hash_multipliers = {} # Store multipliers for hashing
        if self.n_gram_sizes:
            if n_gram_vocab_size <= 0:
                logger.warning("Disabling N-gram features: n_gram_vocab_size <= 0.")
                self.n_gram_sizes = []
            else:
                self.n_gram_embeddings = nn.ModuleDict(
                    {f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes}
                )
                # Initialize N-gram embeddings
                for emb in self.n_gram_embeddings.values():
                    nn.init.normal_(emb.weight, std=0.02)
                # Generate hash multipliers (simple prime-based approach)
                self.hash_multipliers = {n: torch.tensor([self._get_prime(n * 10 + i + 1) for i in range(n)], dtype=torch.long) for n in self.n_gram_sizes}
                logger.info(f"HAKMEMLocalEncoder N-grams enabled: Sizes={self.n_gram_sizes}, Vocab={self.n_gram_vocab_size}")
        else:
            logger.info("HAKMEMLocalEncoder: N-gram features disabled.")

        # Transformer Encoder
        # Adjust head count if necessary
        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads:
            logger.warning(f"Local Encoder Transformer adjusted heads: {original_num_heads} -> {num_heads} for hidden_size {hidden_size}.")

        # NormFirst is generally more stable
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Patch Pooling using Cross-Attention with a learnable query
        self.patch_pooling_attention = HAKMEMCrossAttentionBlock(hidden_size, num_heads, dropout)
        self.patch_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01) # Single query for pooling

        # Final LayerNorm and Dropout
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _get_prime(self, n):
        """Helper to find the smallest prime >= n."""
        def is_prime(num):
            if num < 2: return False
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0: return False
            return True
        num = n
        # --- Corrected Indentation ---
        while True:
            if is_prime(num): return num
            num += 1
        # --- End Correction ---

    def _get_n_gram_hashes(self, patch_byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        """Calculates rolling hashes for n-grams within a patch."""
        patch_len = patch_byte_sequence.size(0)
        device = patch_byte_sequence.device
        if patch_len < n: return torch.empty(0, dtype=torch.long, device=device) # Not enough bytes for n-gram

        # Create sliding windows of size n
        windows = patch_byte_sequence.long().unsqueeze(0).unfold(dimension=1, size=n, step=1) # Shape: (1, num_windows, n)

        # Get multipliers for this n-gram size
        multipliers = self.hash_multipliers.get(n)
        if multipliers is None: # Fallback if not precomputed (should not happen)
            logger.warning(f"Hash multipliers not found for n={n}. Using defaults.")
            multipliers = torch.tensor([31**i for i in range(n)], device=device, dtype=torch.long)
        else:
            multipliers = multipliers.to(device=device, dtype=torch.long)

        # Simple polynomial rolling hash: sum(byte * multiplier) mod vocab_size
        multipliers = multipliers.view(1, 1, n) # Reshape for broadcasting
        hashes = (windows * multipliers).sum(dim=-1) # Shape: (1, num_windows)
        # Modulo to fit within vocab size
        return (hashes % self.n_gram_vocab_size).squeeze(0) # Shape: (num_windows,)

    def forward(self, patches_with_entropy: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Encodes a list of patches for a single sequence item. Returns shape [1, NumPatches, HiddenSize]."""
        if not patches_with_entropy:
            # Return empty tensor if no patches provided
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        device = patches_with_entropy[0][0].device # Get device from the first patch
        model_dtype = next(self.parameters()).dtype
        patch_representations = []

        for patch_bytes, patch_entropy in patches_with_entropy:
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue # Skip empty patches

            patch_bytes_long = patch_bytes.long() # Ensure indices are long type

            # 1. Get byte embeddings
            x = self.byte_embeddings(patch_bytes_long).to(model_dtype) # Shape: (PatchLen, HiddenSize)
            x = x.unsqueeze(0) # Add batch dimension: (1, PatchLen, HiddenSize)

            # 2. Add N-gram features (if enabled)
            if self.n_gram_embeddings and self.n_gram_sizes:
                # Initialize features tensor
                n_gram_features = torch.zeros_like(x)
                for n in self.n_gram_sizes:
                    if patch_len >= n: # Only compute if patch is long enough
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_long, n)
                        if n_gram_hashes.numel() > 0:
                            # Get embeddings for the hashes
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes).to(model_dtype) # Shape: (num_windows, HiddenSize)
                            ngram_embeds = ngram_embeds.unsqueeze(0) # Shape: (1, num_windows, HiddenSize)

                            # Use scatter_add_ to add n-gram embeddings to the corresponding byte positions
                            # The n-gram ending at index `k` corresponds to the `k`-th byte's embedding (using n-1 offset)
                            num_windows = ngram_embeds.size(1)
                            # Indices where the n-gram embeddings should be added (corresponding to the *last* byte of each n-gram)
                            indices = torch.arange(n - 1, n - 1 + num_windows, device=device, dtype=torch.long)
                            # Ensure indices are within the patch length bounds
                            valid_mask = indices < patch_len
                            valid_indices = indices[valid_mask]
                            valid_embeds = ngram_embeds[:, valid_mask, :]

                            if valid_indices.numel() > 0:
                                # Reshape indices for scatter_add_
                                index_reshaped = valid_indices.view(1, -1, 1)
                                index_expanded = index_reshaped.expand(1, valid_indices.size(0), self.hidden_size)
                                # Add the valid embeddings at the correct positions
                                n_gram_features.scatter_add_(1, index_expanded, valid_embeds)

                # Add combined n-gram features to byte embeddings
                x = x + n_gram_features

            # Stability check before Transformer
            if not torch.isfinite(x).all():
                logger.warning(f"NaN/Inf in LocalEncoder input pre-Transformer (PatchLen={patch_len}). Replacing.")
                x = torch.nan_to_num(x, nan=0.0)

            # 3. Apply dropout and Transformer
            x = self.dropout(x)
            # The Transformer encodes the relationships between bytes *within* the patch
            processed_bytes = self.transformer(x) # Shape: (1, PatchLen, HiddenSize)

            # Stability check after Transformer
            if not torch.isfinite(processed_bytes).all():
                logger.warning(f"NaN/Inf in LocalEncoder output post-Transformer (PatchLen={patch_len}). Replacing.")
                processed_bytes = torch.nan_to_num(processed_bytes, nan=0.0)

            # 4. Pool byte representations into a single patch representation
            # Expand the learnable query to match the batch size (which is 1 here)
            batch_query = self.patch_query.expand(1, -1, -1).to(dtype=model_dtype)
            # Use cross-attention: query attends to the processed byte sequence
            patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes)
            # Shape: (1, 1, HiddenSize)

            # Stability check after pooling
            if not torch.isfinite(patch_repr).all():
                logger.warning(f"NaN/Inf in LocalEncoder output post-pooling (PatchLen={patch_len}). Replacing.")
                patch_repr = torch.nan_to_num(patch_repr, nan=0.0)

            # Squeeze the query dimension and append
            patch_representations.append(patch_repr.squeeze(1)) # Shape: (1, HiddenSize)

        if not patch_representations:
            # Return empty tensor if no valid patches were processed
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        # Concatenate representations of all patches for this sequence item
        patches_combined = torch.cat(patch_representations, dim=0) # Shape: (NumPatches, HiddenSize)
        # Add the batch dimension back
        patches_combined = patches_combined.unsqueeze(0) # Shape: (1, NumPatches, HiddenSize)

        # Apply final LayerNorm
        normed_output = self.norm(patches_combined)

        # Final stability check
        if not torch.isfinite(normed_output).all():
            logger.warning("NaN/Inf in LocalEncoder final output. Replacing.")
            normed_output = torch.nan_to_num(normed_output, nan=0.0)

        return normed_output


# =====================================================================
# HAKMEM-Enhanced Local Decoder (Full Implementation)
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
        self.vocab_size = 256 # Explicitly set vocab size for bytes

        # Adjust head count if necessary
        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads:
            logger.warning(f"HAKMEMLocalDecoder adjusted heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        # Embeddings for target bytes and positions
        self.byte_embeddings = nn.Embedding(self.vocab_size, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))
        # Positional embeddings for the target sequence length
        self.positional_encoding = nn.Embedding(max_decode_len, hidden_size)
        nn.init.normal_(self.positional_encoding.weight, std=0.02)

        # Projection layer to adapt memory dimension (from WuBu) to decoder dimension
        # Includes non-linearity and LayerNorm for better adaptation
        self.memory_projection = nn.Sequential(
            nn.Linear(global_hidden_size, hidden_size * 2, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=1e-6) # Normalize projected memory
        )
        # Initialize projection layers
        for layer in self.memory_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)

        # Standard Transformer Decoder
        # Using norm_first=True for potentially better stability
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True
        )
        # Final normalization layer applied after all decoder layers
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=self.decoder_norm)

        # Output prediction head
        if self.use_hierarchical:
            # Predict 16 coarse classes (e.g., 0-15, 16-31, ...)
            self.byte_class_pred = nn.Linear(hidden_size, 16)
            # Predict specific byte within each class (16 heads, each predicting 16 possibilities)
            self.byte_specific_pred = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(16)])
            # Initialize prediction heads
            nn.init.normal_(self.byte_class_pred.weight, std=0.02)
            if self.byte_class_pred.bias is not None: nn.init.zeros_(self.byte_class_pred.bias)
            # Smaller init for specific preds as they combine probabilities
            for layer in self.byte_specific_pred:
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(16)) # Scale down init
                if layer.bias is not None: nn.init.zeros_(layer.bias)
            logger.info("HAKMEMLocalDecoder using Hierarchical Prediction Head.")
        else:
            # Standard flat prediction head
            self.byte_pred = nn.Linear(hidden_size, self.vocab_size)
            nn.init.normal_(self.byte_pred.weight, std=0.02)
            if self.byte_pred.bias is not None: nn.init.zeros_(self.byte_pred.bias)
            logger.info("HAKMEMLocalDecoder using Flat Prediction Head.")

        # Dropout for embeddings + positional encoding
        self.dropout_embed = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask for self-attention. True values indicate positions to be masked."""
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs decoding step.

        Args:
            tgt_byte_seq (Tensor): Target sequence (input to the decoder). Shape: (Batch, TargetSeqLen).
            memory (Tensor): Memory tensor from the encoder/WuBu layers. Shape: (Batch, MemorySeqLen, GlobalHiddenSize).
            tgt_mask (Optional[Tensor]): Mask for the target sequence's self-attention (causal mask).
                                         Shape: (TargetSeqLen, TargetSeqLen). Boolean, True = Mask.
            memory_key_padding_mask (Optional[Tensor]): Mask indicating padded elements in the memory.
                                                        Shape: (Batch, MemorySeqLen). Boolean, True = Mask/Pad.

        Returns:
            Tensor: Output logits for the next byte prediction. Shape: (Batch, TargetSeqLen, VocabSize).
        """
        batch_size, tgt_len = tgt_byte_seq.size()
        device = tgt_byte_seq.device
        mem_batch_size, mem_len, mem_dim_in = memory.size()
        model_dtype = next(self.parameters()).dtype

        # Handle empty target sequence
        if tgt_len == 0:
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)

        # Project and normalize memory from encoder/complex layers
        projected_memory: torch.Tensor
        if mem_len == 0:
            # Handle case where memory is empty (e.g., no patches from encoder)
            logger.debug("HAKMEMLocalDecoder received empty memory.")
            # Create a zero tensor with the expected dimensions for the decoder
            projected_memory = torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=model_dtype)
            # If memory is empty, the padding mask should also be irrelevant or None
            memory_key_padding_mask = None
        else:
            # --- Stability Check (Memory Input) ---
            if not torch.isfinite(memory).all():
                logger.warning("NaN/Inf detected in memory input to LocalDecoder. Replacing.")
                memory = torch.nan_to_num(memory, nan=0.0)
            # Project memory to decoder's hidden size
            projected_memory = self.memory_projection(memory.to(model_dtype))
            # --- Stability Check (Projected Memory) ---
            if not torch.isfinite(projected_memory).all():
                logger.warning("NaN/Inf detected after memory projection in LocalDecoder. Replacing.")
                projected_memory = torch.nan_to_num(projected_memory, nan=0.0)

        # Prepare target sequence embeddings + positional encodings
        tgt_embed = self.byte_embeddings(tgt_byte_seq.long()).to(model_dtype) # Shape: (B, TgtLen, Hidden)
        # Create position indices [0, 1, ..., TgtLen-1], clamp to max embedding length
        positions = torch.arange(0, tgt_len, device=device).unsqueeze(0) # Shape: (1, TgtLen)
        positions = torch.clamp(positions, max=self.positional_encoding.num_embeddings - 1)
        pos_embed = self.positional_encoding(positions).to(model_dtype) # Shape: (1, TgtLen, Hidden) -> broadcast to (B, TgtLen, Hidden)
        # Combine embeddings and apply dropout
        tgt_prepared = self.dropout_embed(tgt_embed + pos_embed)

        # --- Stability Check (Target Input) ---
        if not torch.isfinite(tgt_prepared).all():
            logger.warning("NaN/Inf detected in prepared target sequence input to Decoder Transformer. Replacing.")
            tgt_prepared = torch.nan_to_num(tgt_prepared, nan=0.0)

        # Generate causal mask if not provided (standard for autoregressive decoding)
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt_len, device) # Shape: (TgtLen, TgtLen)

        # Ensure masks are boolean on the correct device
        if tgt_mask is not None: tgt_mask = tgt_mask.to(device=device, dtype=torch.bool)
        # `memory_key_padding_mask` should have True where memory elements are PADDED (invalid).
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(device=device, dtype=torch.bool)
            # Check shape compatibility with memory
            if memory_key_padding_mask.shape != (batch_size, mem_len):
                logger.warning(f"memory_key_padding_mask shape mismatch ({memory_key_padding_mask.shape}) with memory ({batch_size, mem_len}). Ignoring mask.")
                memory_key_padding_mask = None

        # Pass through Transformer Decoder
        # Note: memory_mask (tgt pos -> mem pos) is usually None for standard cross-attention.
        #       tgt_key_padding_mask is for masking padded elements in the target sequence itself (if any).
        output = self.transformer(
            tgt=tgt_prepared,              # Target sequence embeddings + pos encoding
            memory=projected_memory,       # Projected memory from WuBu/encoder
            tgt_mask=tgt_mask,             # Causal mask for self-attention on target seq
            memory_mask=None,              # Not typically used here
            tgt_key_padding_mask=None,     # Assuming target sequence is not padded
            memory_key_padding_mask=memory_key_padding_mask # Mask for padded elements in memory
        ) # Shape: (B, TgtLen, Hidden)

        # --- Stability Check (Decoder Output) ---
        if not torch.isfinite(output).all():
            logger.warning("NaN/Inf detected in output of Decoder Transformer. Replacing.")
            output = torch.nan_to_num(output, nan=0.0)

        # Generate final byte logits using the prediction head
        byte_logits: torch.Tensor
        if self.use_hierarchical:
            byte_class_logits = self.byte_class_pred(output) # Shape (B, S, 16)
            # --- Stability Check ---
            if not torch.isfinite(byte_class_logits).all():
                logger.warning("NaN/Inf in hierarchical class logits. Replacing.")
                byte_class_logits = torch.nan_to_num(byte_class_logits, nan=0.0)

            # Use log_softmax for numerical stability when combining probabilities later
            log_class_probs = F.log_softmax(byte_class_logits, dim=-1) # Shape (B, S, 16)

            log_specific_probs_list = []
            for i in range(16): # Iterate through the 16 specific prediction heads
                specific_logits = self.byte_specific_pred[i](output) # Shape (B, S, 16)
                # --- Stability Check ---
                if not torch.isfinite(specific_logits).all():
                    logger.warning(f"NaN/Inf in hierarchical specific logits head {i}. Replacing.")
                    specific_logits = torch.nan_to_num(specific_logits, nan=0.0)
                # Calculate log probabilities for specific bytes within the class
                log_specific_probs_list.append(F.log_softmax(specific_logits, dim=-1)) # Shape (B, S, 16)

            # Stack the specific log probabilities along a new dimension
            log_specific_probs_stacked = torch.stack(log_specific_probs_list, dim=2) # Shape (B, S, 16, 16)

            # Combine log probabilities: log P(byte) = log P(class) + log P(specific | class)
            # Unsqueeze log_class_probs to broadcast: (B, S, 16, 1) + (B, S, 16, 16) -> (B, S, 16, 16)
            combined_log_probs = log_class_probs.unsqueeze(-1) + log_specific_probs_stacked

            # Reshape to final logits: (B, S, 256)
            byte_logits = combined_log_probs.view(batch_size, tgt_len, self.vocab_size)
        else:
            # Flat prediction: Simple linear layer
            byte_logits = self.byte_pred(output) # Shape: (B, S, VocabSize)

        # Ensure final output is float32 for loss calculation and handle any remaining NaNs
        byte_logits = byte_logits.float()
        if not torch.isfinite(byte_logits).all():
            logger.warning("NaN/Inf detected in final decoder logits. Replacing with zeros.")
            byte_logits = torch.nan_to_num(byte_logits, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0

        return byte_logits

# =====================================================================
# HAKMEM-Enhanced Q-Learning Controller & Optimizer (Full Implementations)
# =====================================================================
class HAKMEMQController:
    """A Q-learning agent to dynamically adjust optimizer hyperparameters."""
    def __init__(self, learning_rate: float=0.01, discount: float=0.95, epsilon: float=0.2, epsilon_decay: float=0.9998, min_epsilon: float=0.01, lr_scale_options: Optional[List[float]]=None, momentum_scale_options: Optional[List[float]]=None, max_q_table_size: int=10000):
        self.q_table: Dict[Tuple, Dict[str, np.ndarray]] = {} # State -> {ParamName: QValuesArray}
        self.alpha = learning_rate # Q-learning learning rate
        self.gamma = discount      # Discount factor for future rewards
        self.epsilon = epsilon     # Initial exploration rate
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay # Rate at which exploration decreases
        self.prev_loss: Optional[float] = None
        self.prev_state: Optional[Tuple] = None
        self.prev_action: Optional[Dict[str, float]] = None # Action taken in the previous step

        # Define the discrete action space (scaling factors for LR and Momentum)
        if lr_scale_options is None: lr_scale_options = [0.9, 0.95, 1.0, 1.05, 1.1]
        if momentum_scale_options is None: momentum_scale_options = [0.95, 0.98, 1.0, 1.01, 1.02]
        self.action_ranges = {
            'lr_scale': np.array(lr_scale_options, dtype=np.float32),
            'momentum_scale': np.array(momentum_scale_options, dtype=np.float32)
        }
        self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}

        # State tracking windows (use deques for efficient fixed-size history)
        self.loss_window = deque(maxlen=10)
        self.grad_norm_window = deque(maxlen=10)
        self.lr_window = deque(maxlen=5)
        self.momentum_window = deque(maxlen=5)
        self.performance_window = deque(maxlen=20) # Tracks recent rewards for stability/oscillation checks

        # State variables for reward shaping / advanced state features
        self.stable_steps = 0       # Counter for consecutive steps with positive reward
        self.oscillation_counter = 0 # Counter for detecting oscillating rewards
        self.prev_actions_log = deque(maxlen=5) # Track recent actions (optional, for more complex state)

        # Q-table management
        self.max_q_table_size = max_q_table_size
        self.q_table_access_count: Dict[Tuple, int] = defaultdict(int)
        self.q_table_creation_time: Dict[Tuple, float] = {}

        # Reward shaping parameters
        self.flow_coefficient = 0.05     # Modulates alpha based on TD error magnitude (adaptive learning rate for Q-table)
        self.oscillation_penalty = 0.15 # Penalty applied if reward oscillates significantly
        self.stability_reward_bonus = 0.05 # Bonus for maintaining stable positive rewards

        logger.info(f"QController initialized: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}, decay={self.epsilon_decay}, min_eps={self.min_epsilon}")
        logger.info(f"QController action ranges: LR={self.action_ranges['lr_scale']}, Momentum={self.action_ranges['momentum_scale']}")
        logger.info(f"QController reward params: OscPenalty={self.oscillation_penalty}, StabBonus={self.stability_reward_bonus}, FlowCoef={self.flow_coefficient}")

    def get_state(self, lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float]) -> Optional[Tuple]:
        """Discretizes the current training status into a state tuple."""
        # Handle invalid inputs gracefully - return None to indicate invalid state
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm):
            logger.debug(f"Q-state calculation skipped: Invalid input (Loss: {loss}, GradNorm: {grad_norm})")
            return None # Return None if state is invalid

        # Update state tracking windows with current finite values
        self.loss_window.append(loss)
        self.grad_norm_window.append(grad_norm)
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)

        # Need sufficient history to calculate trends/levels reliably
        if len(self.loss_window) < 3 or len(self.grad_norm_window) < 3:
            logger.debug("Q-state calculation skipped: Insufficient history.")
            return None # Not enough data for meaningful state yet

        # Default state bins (used if calculation fails)
        loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0

        try:
            # --- Loss Trend ---
            # Use recent history (last 5 steps)
            y = np.array(list(self.loss_window)[-5:], dtype=np.float32)
            x = np.arange(len(y))
            # Check for sufficient points and variance for polyfit
            if len(y) >= 2 and len(np.unique(y)) > 1:
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
            else:
                slope = 0.0 # No trend if constant loss or too few points
            # Normalize slope by average loss magnitude for scale invariance
            avg_loss = np.mean(y)
            normalized_slope = slope / (abs(avg_loss) + EPS) # Avoid division by zero
            # Bin the normalized slope into categories (e.g., decreasing sharply, decreasing, stable, increasing, increasing sharply)
            loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05]).item() # .item() converts numpy int

            # --- Gradient Norm Level ---
            # Use median for robustness to outliers
            avg_grad_norm = np.median(list(self.grad_norm_window))
            # Bin gradient norm magnitude (e.g., very low, low, medium, high, very high)
            grad_norm_level_bin = np.digitize(avg_grad_norm, bins=[0.1, 0.5, 1.5, 5.0]).item()

            # --- Learning Rate Level ---
            # Bin LR magnitude (adjust bins based on expected LR range)
            lr_level_bin = np.digitize(lr / 1e-4, bins=[0.5, 2.0, 10.0, 50.0]).item() # Example bins relative to 1e-4

            # --- Momentum Level ---
            # Bin momentum value
            momentum_level_bin = np.digitize(momentum, bins=[0.85, 0.92, 0.97]).item() # Example bins

            # --- Oscillation Detection ---
            # Check if recent rewards are flipping sign aggressively
            if len(self.performance_window) >= 2:
                # Check for sign flips with significant magnitude
                if (self.performance_window[-1] > 1e-2 and self.performance_window[-2] < -1e-2) or \
                   (self.performance_window[-1] < -1e-2 and self.performance_window[-2] > 1e-2):
                    self.oscillation_counter = min(self.oscillation_counter + 1, 5) # Increase counter, capped
                else:
                    self.oscillation_counter = max(0, self.oscillation_counter - 1) # Decrease counter if stable
            # Binary flag: 1 if oscillating significantly, 0 otherwise
            oscillation_bin = 1 if self.oscillation_counter >= 3 else 0

        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
            logger.warning(f"Q-state calculation numerical error: {e}. Using default bins.")
            # Fallback to default bins on error
            loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0
        except Exception as e_state:
            logger.error(f"Unexpected error during Q-state calculation: {e_state}", exc_info=True)
            return None # Return None on unexpected errors

        # Final state is a tuple of the binned features
        state = (loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin)

        # Track state usage for Q-table pruning
        self.q_table_access_count[state] += 1
        return state

    def compute_reward(self, current_loss: Optional[float], prev_loss: Optional[float], grad_norm: Optional[float]) -> float:
        """Calculates the reward based on loss change and stability metrics."""
        # Check for invalid inputs
        if current_loss is None or prev_loss is None or grad_norm is None or \
           not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm):
            logger.debug(f"Reward calculation skipped: Invalid input (CurrL:{current_loss}, PrevL:{prev_loss}, GradN:{grad_norm})")
            return 0.0 # Return neutral reward if inputs are invalid

        # Primary reward component: normalized loss change
        loss_change = prev_loss - current_loss
        # Normalize by previous loss magnitude (add EPS to avoid division by zero)
        loss_change_ratio = loss_change / (abs(prev_loss) + EPS)
        # Use tanh to squash the reward between -1 and 1, amplifying smaller changes
        reward = np.tanh(loss_change_ratio * 10.0) # Scaling factor 10 amplifies sensitivity

        # Penalty for very high gradient norms (potential instability)
        if grad_norm > 5.0:
            reward -= 0.1 * min(1.0, max(0.0, (grad_norm - 5.0) / 10.0)) # Linear penalty capped at -0.1
        # Small bonus for very low gradient norms (potential convergence)
        elif grad_norm < 0.05:
            reward += 0.02

        # Penalty for detected reward oscillation
        if self.oscillation_counter >= 3:
            reward -= self.oscillation_penalty

        # Track reward history for oscillation detection and stability bonus
        self.performance_window.append(reward)

        # Bonus for stable improvement (consecutive positive rewards)
        if reward > 0.0:
            self.stable_steps += 1
            # Add a small bonus that increases slightly with stability duration
            reward += min(0.1, self.stability_reward_bonus * (self.stable_steps // 5)) # Bonus capped at 0.1
        else:
            self.stable_steps = 0 # Reset counter if reward is not positive

        # Clip final reward to [-1, 1] range
        return float(np.clip(reward, -1.0, 1.0))

    def choose_action(self, state: Optional[Tuple]) -> Dict[str, float]:
        """Selects actions (LR scale, Momentum scale) based on state using epsilon-greedy."""
        # Default action (no change) if state is invalid
        if state is None:
            return {'lr_scale': 1.0, 'momentum_scale': 1.0}

        # Initialize Q-values for new state if encountered for the first time
        if state not in self.q_table:
            self.q_table[state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[state] = time.time() # Record creation time
            self.q_table_access_count[state] = 1 # Initialize access count
            self._manage_q_table_size() # Check if pruning is needed

        # Update epsilon (decay exploration rate)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        chosen_actions = {}
        for param, q_values in self.q_table[state].items():
            action_space = self.action_ranges[param]
            if random.random() < self.epsilon: # Explore: Choose a random action
                chosen_idx = random.randrange(len(action_space))
            else: # Exploit: Choose the action with the highest Q-value
                # Filter out non-finite Q-values before finding max
                finite_q_mask = np.isfinite(q_values)
                if not np.any(finite_q_mask): # If all Q-values are non-finite, choose randomly
                    chosen_idx = random.randrange(len(action_space))
                else:
                    finite_q_values = q_values[finite_q_mask]
                    max_q = np.max(finite_q_values)
                    # Find all indices corresponding to the max Q-value among finite ones
                    best_indices = np.where(np.isclose(q_values, max_q) & finite_q_mask)[0]
                    if len(best_indices) > 0:
                        chosen_idx = np.random.choice(best_indices) # Randomly choose among best actions
                    else: # Should not happen if finite_q_mask is not all False, but safety fallback
                        chosen_idx = random.randrange(len(action_space))

            # Get the scaling factor corresponding to the chosen index
            chosen_actions[param] = float(action_space[chosen_idx])

        # Log the chosen action for potential use in state or analysis
        self.prev_actions_log.append(chosen_actions.copy())
        return chosen_actions

    def update(self, state: Optional[Tuple], action: Optional[Dict[str, float]], reward: float, next_state: Optional[Tuple]):
        """Updates the Q-table using the Bellman equation (Q-learning update rule)."""
        # Do not update if current state, next state, or action is invalid/None
        if state is None or next_state is None or action is None:
            logger.debug("Q-update skipped: Invalid state/action.")
            return

        # Ensure current state exists in Q-table (should have been added in choose_action)
        if state not in self.q_table:
            logger.warning(f"QController: State {state} not found in Q-table during update. Skipping update for this state.")
            return

        # Ensure next state exists (or initialize it if newly encountered)
        if next_state not in self.q_table:
            self.q_table[next_state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[next_state] = time.time()
            self.q_table_access_count[next_state] = 0 # Initialize access count
            self._manage_q_table_size() # Check table size after adding new state

        # Update Q-value for each parameter's action (LR scale, Momentum scale)
        for param, chosen_value in action.items():
            action_space = self.action_ranges[param]
            # Find the index of the action taken
            action_indices = np.where(np.isclose(action_space, chosen_value))[0]
            if len(action_indices) == 0:
                logger.warning(f"Could not find action index for param {param}, value {chosen_value}. Skipping update.")
                continue
            action_idx = action_indices[0]

            # Get current Q-value estimate for the (state, action) pair
            current_q = self.q_table[state][param][action_idx]

            # Get Q-values for the next state
            next_q_values = self.q_table[next_state][param]
            # Find the maximum Q-value for the next state (considering only finite values)
            finite_next_q = next_q_values[np.isfinite(next_q_values)]
            max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0
            # Handle case where max_future_q might somehow still be non-finite (e.g., all next Qs are inf)
            if not np.isfinite(max_future_q): max_future_q = 0.0

            # Q-learning update rule: Q(s,a) = Q(s,a) + alpha * [R + gamma * max_Q(s') - Q(s,a)]
            td_target = reward + self.gamma * max_future_q
            td_error = td_target - current_q

            # Adaptive learning rate (alpha) based on TD error magnitude (optional)
            adaptive_alpha = min(0.5, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error))))

            # Calculate the new Q-value
            new_q = current_q + adaptive_alpha * td_error

            # Update Q-table, clipping to prevent extreme values and handling NaNs
            if np.isfinite(new_q):
                self.q_table[state][param][action_idx] = np.clip(new_q, -1e5, 1e5)
            else:
                # If new_q is NaN/Inf, reset to 0 to avoid corrupting the table
                logger.warning(f"Non-finite new Q-value calculated for state {state}, param {param}, action {action_idx}. Resetting to 0.")
                self.q_table[state][param][action_idx] = 0.0

    def _manage_q_table_size(self):
        """Prunes the Q-table if it exceeds the maximum size by removing least accessed/oldest states."""
        if len(self.q_table) > self.max_q_table_size:
            num_to_remove = len(self.q_table) - self.max_q_table_size
            logger.info(f"Q-table size ({len(self.q_table)}) exceeds max ({self.max_q_table_size}). Pruning {num_to_remove} states.")
            try:
                # Check if metadata exists (it should if states were added properly)
                if not self.q_table_access_count or not self.q_table_creation_time:
                    # Fallback: Remove random states if metadata is missing
                    logger.warning("Q-table metadata incomplete during pruning. Removing random states.")
                    states_to_remove = random.sample(list(self.q_table.keys()), min(num_to_remove, len(self.q_table)))
                else:
                    # Sort states primarily by access count (ascending), secondarily by creation time (ascending)
                    # This removes the least frequently accessed states first, and among those, the oldest ones.
                    sorted_states = sorted(self.q_table.keys(), key=lambda s: (
                        self.q_table_access_count.get(s, 0), # Access count (default 0 if somehow missing)
                        self.q_table_creation_time.get(s, float('inf'))) # Creation time (default inf if missing)
                    )
                    states_to_remove = sorted_states[:num_to_remove]

                # Remove the selected states from the Q-table and metadata dictionaries
                for state_to_remove in states_to_remove:
                    self.q_table.pop(state_to_remove, None)
                    self.q_table_access_count.pop(state_to_remove, None)
                    self.q_table_creation_time.pop(state_to_remove, None)
                logger.info(f"Pruned {len(states_to_remove)} states. New Q-table size: {len(self.q_table)}")
            except Exception as e:
                logger.warning(f"Error during Q-table pruning: {e}. Attempting random removal as fallback.", exc_info=False)
                # Fallback random removal on any error during sorted removal
                current_keys = list(self.q_table.keys())
                num_to_remove = max(0, len(current_keys) - self.max_q_table_size)
                if num_to_remove > 0:
                    states_to_remove = random.sample(current_keys, min(num_to_remove, len(current_keys)))
                    for state_to_remove in states_to_remove:
                        self.q_table.pop(state_to_remove, None)
                        self.q_table_access_count.pop(state_to_remove, None)
                        self.q_table_creation_time.pop(state_to_remove, None)
                    logger.info(f"Fallback pruned {len(states_to_remove)} random states. New Q-table size: {len(self.q_table)}")


    def get_info(self) -> Dict:
        """Returns current status information about the Q-controller."""
        last_action = self.prev_actions_log[-1] if self.prev_actions_log else None
        avg_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.0
        return {
            "epsilon": self.epsilon,
            "stable_steps": self.stable_steps,
            "oscillation_counter": self.oscillation_counter,
            "q_table_size": len(self.q_table),
            "last_action": last_action, # The action chosen in the *last* call to choose_action
            "avg_reward_last_20": avg_reward
        }


class HAKMEMEnhancedSGD(torch.optim.Optimizer):
    """SGD optimizer enhanced with momentum, weight decay, gradient clipping (via Trainer), and Q-learning control."""
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm=1.0, q_learning_config: Optional[Dict]=None, enable_flow=False, flow_coefficient=0.05, flow_momentum=0.95):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize Q-controller if config is provided and valid
        self.q_controller: Optional[HAKMEMQController] = None
        if isinstance(q_learning_config, dict):
            try:
                self.q_controller = HAKMEMQController(**q_learning_config)
                logger.info("HAKMEMEnhancedSGD: Q-Controller enabled.")
            except Exception as e:
                logger.error(f"Failed to initialize HAKMEMQController: {e}. Disabling Q-control.", exc_info=True)
        else:
            logger.info("HAKMEMEnhancedSGD: Q-Controller disabled (no config provided).")

        # Note: max_grad_norm stored here but actual clipping happens in Trainer
        self.max_grad_norm = max_grad_norm
        self._step_count = 0
        self.current_loss: Optional[float] = None # Track loss for Q-controller state

        # Gradient statistics tracking
        self.gradient_stats = GradientStats()

        # Flow parameters (currently unused in step, potentially for future Riemannian adaptations)
        self.flow_enabled = enable_flow
        self.flow_coefficient = flow_coefficient
        self.flow_momentum = flow_momentum

        # Initialize momentum buffer state for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Initialize momentum buffer lazily or here
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def zero_grad(self, set_to_none=True):
        """Zeros gradients. Pass set_to_none=True for potential memory savings."""
        super().zero_grad(set_to_none=set_to_none)

    def set_current_loss(self, loss: Optional[float]):
        """Sets the loss for the current step, used by the Q-controller."""
        if loss is not None and np.isfinite(loss):
            self.current_loss = loss
        else:
            # If loss is invalid, keep the previous valid loss for Q-state calculation
            # self.current_loss remains unchanged, or stays None if never valid
            logger.debug(f"Optimizer received invalid loss: {loss}. Q-controller will use previous loss: {self.current_loss}")
            pass

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step. Closure is ignored."""
        if closure is not None:
            # Standard PyTorch optimizers can optionally use closures
            # This implementation doesn't use it, so we log a warning if provided.
            logger.warning("HAKMEMEnhancedSGD.step received a closure, but it is not used.")

        # --- Q-Controller Action Application ---
        # Apply the action chosen in the *previous* Q-controller step
        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action # Get action decided after the *last* optimizer step
            for group in self.param_groups:
                base_lr = group['base_lr']
                base_momentum = group['base_momentum']
                # Apply scaling factors from the Q-action
                new_lr = base_lr * q_action.get('lr_scale', 1.0)
                new_momentum = base_momentum * q_action.get('momentum_scale', 1.0)
                # Apply clamps to keep hyperparams in reasonable ranges
                group['lr'] = float(np.clip(new_lr, 1e-8, 0.1)) # Clamp LR
                group['momentum'] = float(np.clip(new_momentum, 0.5, 0.999)) # Clamp momentum

        # --- Parameter Update Loop ---
        # Gradients are assumed to be unscaled (by AMP scaler) and clipped (by Trainer) before this point.
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None: # Skip parameters without gradients
                    continue
                if not p.requires_grad: # Skip parameters that are frozen
                    logger.debug(f"Skipping update for parameter not requiring grad: shape {p.shape}")
                    continue

                grad = p.grad # Get the (unscaled, potentially clipped) gradient

                # --- Stability Check: Gradient ---
                if not torch.isfinite(grad).all():
                    # This should ideally be caught earlier (e.g., during clipping), but double-check.
                    num_nan = torch.isnan(grad).sum().item()
                    num_inf = torch.isinf(grad).sum().item()
                    logger.error(f"Optimizer Error: Non-finite gradient detected for param shape {p.shape} during update step (NaNs: {num_nan}, Infs: {num_inf}). Skipping update for this parameter.")
                    continue # Skip update for this specific parameter

                param_data = p.data # Get parameter data
                param_state = self.state[p] # Get optimizer state for this parameter

                # Apply weight decay (L2 penalty): grad = grad + param * wd
                if weight_decay != 0:
                    # Perform calculation in float32 for stability, then cast back if needed
                    grad = grad.add(param_data.float(), alpha=weight_decay).to(grad.dtype)

                # Momentum calculation: buf = momentum * buf + grad
                if 'momentum_buffer' not in param_state:
                    # Initialize buffer if it doesn't exist (should have been done in __init__)
                    param_state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    logger.warning(f"Momentum buffer re-initialized for param shape {p.shape} during step.")

                buf = param_state['momentum_buffer']
                # Update momentum buffer in-place
                buf.mul_(momentum).add_(grad) # buf = momentum * buf + grad
                # No need to reassign buf to param_state['momentum_buffer'] as it's updated in-place

                # Update parameter: p = p - lr * buf
                update_step = buf * lr
                param_data.add_(-update_step) # Update parameter data in-place

        self._step_count += 1
        # This optimizer modifies parameters in-place, so it returns None.
        return None

    def get_q_info(self) -> Dict:
        """Retrieves information from the Q-controller, if enabled."""
        if hasattr(self, 'q_controller') and self.q_controller:
            return self.q_controller.get_info()
        return {"Q-Controller": "Disabled"}


# =====================================================================
# WuBu Nesting Components (Self-Contained Geometry & Design Doc)
# =====================================================================

# --- Quaternion Utils ---
def check_quat_dim(dim: int, layer_name: str = "Layer"):
    """Checks if dimension is divisible by 4 for quaternion operations."""
    if dim % 4 != 0:
        raise ValueError(f"{layer_name} dimension must be divisible by 4 for quaternion operations, but got {dim}")

def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the Hamilton product of two quaternions (or batches of quaternions)."""
    # Ensure inputs are broadcastable and extract components
    q1_shape = list(q1.shape); q2_shape = list(q2.shape)
    while len(q1_shape) < len(q2_shape): q1_shape.insert(0, 1)
    while len(q2_shape) < len(q1_shape): q2_shape.insert(0, 1)
    q1 = q1.view(q1_shape); q2 = q2.view(q2_shape) # Reshape for broadcasting if needed

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Hamilton product formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack components back into a tensor
    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion (negate vector part)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def quat_rotate_via_pvq(v: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotates vector v (represented as a quaternion) using p * v * q."""
    # Ensure dimensions are 4 for quaternion operations
    if v.shape[-1] != 4 or p.shape[-1] != 4 or q.shape[-1] != 4:
        raise ValueError(f"Inputs must be 4D for quat_rotate_via_pvq, shapes: v={v.shape}, p={p.shape}, q={q.shape}")

    # Ensure p and q are broadcastable to v's shape
    p = p.expand_as(v)
    q = q.expand_as(v)

    # Perform rotation: p * v * q
    pv = hamilton_product(p, v)
    pvq = hamilton_product(pv, q)
    return pvq

# --- Quaternion Linear Layer ---
class QuaternionLinear(nn.Module):
    """Linear layer for quaternion-valued inputs."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Validate dimensions are divisible by 4
        check_quat_dim(in_features, "QuaternionLinear Input")
        check_quat_dim(out_features, "QuaternionLinear Output")

        self.in_features_quat = in_features // 4
        self.out_features_quat = out_features // 4

        # Parameters for the quaternion weights (real and 3 imaginary parts)
        self.r_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.i_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.j_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.k_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))

        # Optional bias term (standard real-valued bias applied to the output quaternion)
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the quaternion weights."""
        # Standard Kaiming/Xavier-like initialization scaled for quaternions
        stdv = 1. / math.sqrt(self.in_features_quat) # Scale by number of input *quaternions*
        gain = 1.0
        scale = gain * stdv
        nn.init.uniform_(self.r_weight, -scale, scale)
        nn.init.uniform_(self.i_weight, -scale, scale)
        nn.init.uniform_(self.j_weight, -scale, scale)
        nn.init.uniform_(self.k_weight, -scale, scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the quaternion linear transformation."""
        # Input shape: (..., in_features) where in_features is divisible by 4
        if x.shape[-1] != self.in_features_quat * 4:
            raise ValueError(f"Input feature dim {x.shape[-1]} != expected {self.in_features_quat * 4}")

        batch_dims = x.shape[:-1]
        # Reshape input to expose the 4 quaternion components: (..., in_features_quat, 4)
        x_reshaped = x.view(*batch_dims, self.in_features_quat, 4)
        r_x, i_x, j_x, k_x = x_reshaped[..., 0], x_reshaped[..., 1], x_reshaped[..., 2], x_reshaped[..., 3]

        # Apply quaternion multiplication using linear layers for each component interaction
        # Based on (a + bi + cj + dk) * (x + yi + zj + wk) expansion
        out_r = F.linear(r_x, self.r_weight) - F.linear(i_x, self.i_weight) - F.linear(j_x, self.j_weight) - F.linear(k_x, self.k_weight)
        out_i = F.linear(r_x, self.i_weight) + F.linear(i_x, self.r_weight) + F.linear(j_x, self.k_weight) - F.linear(k_x, self.j_weight)
        out_j = F.linear(r_x, self.j_weight) - F.linear(i_x, self.k_weight) + F.linear(j_x, self.r_weight) + F.linear(k_x, self.i_weight)
        out_k = F.linear(r_x, self.k_weight) + F.linear(i_x, self.j_weight) - F.linear(j_x, self.i_weight) + F.linear(k_x, self.r_weight)

        # Stack the output components: (..., out_features_quat, 4)
        output = torch.stack([out_r, out_i, out_j, out_k], dim=-1)
        # Reshape back to flat feature dimension: (..., out_features)
        output = output.view(*batch_dims, self.out_features_quat * 4)

        # Add bias if enabled
        if self.bias is not None:
            output = output + self.bias

        return output


# --- Parameter Init Helper ---
def init_weights(m):
    """Applies standard weight initialization to common layer types."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # Initialize gamma (weight) to 1 and beta (bias) to 0
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # Normal initialization often used for embeddings
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, QuaternionLinear):
        # Use the layer's specific reset_parameters method
        m.reset_parameters()

# --- Parameter Constraint Helper ---
def get_constrained_params(param: torch.Tensor, min_val: float = EPS) -> torch.Tensor:
    """Ensures parameter stays positive (>= min_val) using softplus."""
    # Softplus(x) = log(1 + exp(x)) -> always positive
    # Add min_val to ensure it's strictly > 0 and >= min_val
    # Parameterization using log(exp(param)) = param directly, then add min_val
    # Or use softplus for better gradient behavior near zero.
    # Let's stick to softplus + min_val as before.
    return F.softplus(param) + min_val

# --- WuBu Components ---
class TangentFlow(nn.Module):
    """Applies a learned flow (displacement) in the tangent space."""
    def __init__(self, dim: int, flow_type: str = 'mlp', hidden_dim_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.flow_type = flow_type

        if flow_type == 'linear':
            self.flow_map = nn.Linear(dim, dim)
        elif flow_type == 'mlp':
            # Simple MLP: Linear -> GELU -> Dropout -> Linear
            hidden_dim = max(16, int(dim * hidden_dim_ratio)) # Ensure hidden dim is reasonable
            self.flow_map = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            )
        elif flow_type == 'none':
            self.flow_map = nn.Identity() # No flow applied
        else:
            raise ValueError(f"Unsupported tangent_flow_type: {flow_type}")

        # Initialize weights of the flow map
        self.flow_map.apply(init_weights)

    def forward(self, v_tangent: torch.Tensor) -> torch.Tensor:
        """Calculates the flow displacement vector."""
        if self.flow_type == 'none':
            # Return zero displacement if flow is disabled
            return torch.zeros_like(v_tangent)

        flow_displacement = self.flow_map(v_tangent)

        # Stability check
        if not torch.isfinite(flow_displacement).all():
            logger.warning("NaN/Inf detected in TangentFlow output. Replacing with zeros.")
            flow_displacement = torch.nan_to_num(flow_displacement, nan=0.0)

        return flow_displacement

class BoundaryManifold(nn.Module):
    """Represents the learnable boundary points for a WuBu level (in tangent space at origin)."""
    def __init__(self, level_idx: int, num_points: int, point_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim

        # Initialize tangent points as learnable parameters if num_points > 0
        if num_points > 0 and point_dim > 0:
            tangent_points = torch.randn(num_points, point_dim) * init_scale
            self.tangent_points = nn.Parameter(tangent_points)
        else:
            # If no points or zero dimension, register as None
            logger.debug(f"BoundaryManifold L{level_idx} initialized with zero points/dim ({num_points}/{point_dim}). No parameters created.")
            self.register_parameter('tangent_points', None) # Explicitly register as None

    def get_tangent_vectors_at_origin(self) -> Optional[torch.Tensor]:
        """Returns the current boundary points (tangent vectors at origin)."""
        if self.tangent_points is None:
            return None

        # Stability check before returning
        if not torch.isfinite(self.tangent_points).all():
            logger.warning(f"NaN/Inf detected in BoundaryManifold tangent_points (Level {self.level_idx}). Re-initializing gently.")
            # Re-initialize with small random values if NaN/Inf detected
            init_scale = 0.01
            self.tangent_points.data.normal_(0, init_scale) # Re-initialize in-place

        return self.tangent_points

class TangentSpaceRotation(nn.Module):
    """Applies a learnable rotation to tangent vectors (SO(n) or Quaternions)."""
    def __init__(self, dim: int, rotation_type: str = 'so_n'):
        super().__init__()
        self.dim = dim
        self.rotation_type = rotation_type

        if rotation_type == 'so_n':
            # Parameterize SO(n) via the exponential map of a skew-symmetric matrix
            # Initialize near zero for near-identity rotation initially
            self.skew_symmetric_params = nn.Parameter(torch.randn(dim, dim) * 0.01)
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using SO(n) via matrix exponential.")
        elif rotation_type == 'quat':
            if dim != 4:
                raise ValueError("Quaternion rotation requires dim=4.")
            # Parameterize SO(4) rotation using p*v*q formulation (two quaternions p, q)
            # Initialize near (1,0,0,0) for near-identity rotation
            # Note: We parameterize p and q directly and normalize them in forward pass.
            # Adding small noise to break symmetry.
            init_p = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            init_q = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            self.quat_params_p = nn.Parameter(init_p)
            self.quat_params_q = nn.Parameter(init_q)
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using SO(4) via Quaternions (p*v*q).")
        elif rotation_type == 'identity':
            # No learnable parameters needed for identity rotation
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using Identity (no rotation).")
        else:
            raise ValueError(f"Unsupported rotation type: {rotation_type}")

    def _get_rotation_operator(self, device: torch.device):
        """Constructs the rotation operator (matrix or quaternions) on the correct device."""
        if self.rotation_type == 'so_n':
            # Ensure parameters are on the correct device
            params = self.skew_symmetric_params.to(device)
            # Create skew-symmetric matrix: A = P - P^T
            skew_matrix = params - params.T
            # Compute rotation matrix using matrix exponential: R = exp(A)
            R = torch.matrix_exp(skew_matrix)
            return R
        elif self.rotation_type == 'quat':
            # Ensure parameters are on the correct device
            p = self.quat_params_p.to(device)
            q = self.quat_params_q.to(device)
            # Normalize p and q to be unit quaternions for rotation
            p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            unit_p = p / p_norm
            q_norm = torch.norm(q, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            unit_q = q / q_norm
            # Check for NaNs after normalization (shouldn't happen with clamp)
            if not torch.isfinite(unit_p).all() or not torch.isfinite(unit_q).all():
                logger.warning("NaN/Inf detected during quaternion normalization. Resetting p/q.")
                # Reset parameters if normalization failed
                self.quat_params_p.data.normal_(0, 0.01).add_(torch.tensor([1.,0,0,0], device=p.device)) # Ensure reset tensor is on correct device
                self.quat_params_q.data.normal_(0, 0.01).add_(torch.tensor([1.,0,0,0], device=q.device))
                # Re-calculate unit quaternions after reset
                p = self.quat_params_p.to(device); q = self.quat_params_q.to(device)
                p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp(min=EPS); unit_p = p / p_norm
                q_norm = torch.norm(q, p=2, dim=-1, keepdim=True).clamp(min=EPS); unit_q = q / q_norm
            return unit_p, unit_q # Return normalized quaternions
        elif self.rotation_type == 'identity':
            return None # No operator needed
        else:
            raise RuntimeError("Invalid internal rotation type state.")

    def forward(self, v_main: torch.Tensor, v_boundaries_tangent: Optional[torch.Tensor], v_descriptor: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Applies the rotation to main, boundary (if present), and descriptor vectors."""
        has_boundaries = v_boundaries_tangent is not None and v_boundaries_tangent.numel() > 0
        batch_size, seq_len, dim = v_main.shape

        # --- Input Validation ---
        if dim != self.dim:
            raise ValueError(f"Rotation input dim mismatch: expected {self.dim}, got {dim} for v_main")
        if has_boundaries and v_boundaries_tangent.shape[-1] != self.dim:
            raise ValueError(f"Rotation input dim mismatch: expected {self.dim}, got {v_boundaries_tangent.shape[-1]} for v_boundaries")
        if v_descriptor.shape[-1] != self.dim:
            # Descriptor might be shape [Dim] or [1, 1, Dim] after expansion
            if v_descriptor.ndim == 1 and v_descriptor.shape[0] == self.dim:
                # Reshape to (1, 1, Dim) for consistent processing
                v_descriptor = v_descriptor.view(1, 1, self.dim)
            elif v_descriptor.shape != (1, 1, self.dim):
                raise ValueError(f"Rotation input dim mismatch: expected {self.dim}, got {v_descriptor.shape[-1]} for v_descriptor with shape {v_descriptor.shape}")

        device = v_main.device
        # Ensure descriptor is on the correct device
        v_descriptor = v_descriptor.to(device)

        # Initialize outputs
        v_main_rotated, v_boundaries_rotated, v_descriptor_rotated = v_main, v_boundaries_tangent, v_descriptor

        # --- Apply Rotation ---
        operator = self._get_rotation_operator(device)

        if self.rotation_type == 'identity' or operator is None:
            # No rotation applied
            pass
        elif self.rotation_type == 'so_n':
            R = operator # Rotation matrix
            # Apply matrix multiplication: y = x @ R^T (Standard convention is R applied to column vectors)
            # Or y^T = x^T @ R if x represents row vectors. Let's assume row vectors (Batch, Seq, Dim).
            # So we need v_main @ R
            v_main_rotated = torch.matmul(v_main, R)
            if has_boundaries:
                # Boundaries are likely [NumPoints, Dim], treat as row vectors
                v_boundaries_rotated = torch.matmul(v_boundaries_tangent, R)
            # Descriptor is [1, 1, Dim] or [Dim], treat as row vector
            v_descriptor_rotated = torch.matmul(v_descriptor, R) # Shape [1, 1, Dim]
        elif self.rotation_type == 'quat':
            unit_p, unit_q = operator # Normalized quaternions
            # Reshape p and q for broadcasting if needed
            p_b = unit_p.view(1, 1, 4) # For main and descriptor
            q_b = unit_q.view(1, 1, 4)
            v_main_rotated = quat_rotate_via_pvq(v_main, p_b, q_b)
            v_descriptor_rotated = quat_rotate_via_pvq(v_descriptor, p_b, q_b) # Shape [1, 1, 4]
            if has_boundaries:
                # For boundary points (NumPoints, Dim=4)
                p_nb = unit_p.expand(v_boundaries_tangent.shape[:-1] + (4,))
                q_nb = unit_q.expand(v_boundaries_tangent.shape[:-1] + (4,))
                v_boundaries_rotated = quat_rotate_via_pvq(v_boundaries_tangent, p_nb, q_nb)


        # --- Stability Checks & Return ---
        outputs = [v_main_rotated, v_boundaries_rotated, v_descriptor_rotated]
        names = ["main", "boundaries", "descriptor"]
        final_outputs = []
        for i, out in enumerate(outputs):
            # Handle boundary case correctly
            if i == 1 and not has_boundaries:
                final_outputs.append(None) # Pass None if no boundaries input
                continue
            # Should not be None otherwise
            if out is None:
                raise ValueError(f"Rotation output '{names[i]}' is None unexpectedly.")

            # Check for NaN/Inf in the output tensor
            if not torch.isfinite(out).all():
                logger.warning(f"NaN/Inf detected in Rotation output ({names[i]}). Replacing with zeros.")
                out = torch.nan_to_num(out, nan=0.0)
            final_outputs.append(out)

        # Return tuple matching input structure (main, boundaries|None, descriptor)
        return tuple(final_outputs)

class InterLevelTransform(nn.Module):
    """Applies a learnable non-rotational transformation between tangent spaces of different levels."""
    def __init__(self, in_dim: int, out_dim: int, transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.transform_type = transform_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        if transform_type == 'mlp':
            # Determine hidden dimension for MLP
            if hidden_dim is None or hidden_dim <= 0:
                # Default hidden dim: average of in/out, clipped to minimum 16
                h_dim = max(16, (in_dim + out_dim) // 2)
            else:
                h_dim = hidden_dim
            # MLP: Linear -> LayerNorm -> GELU -> Dropout -> Linear
            self.transform = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim), # Add LayerNorm for stability
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, out_dim)
            )
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using MLP (Hidden Dim: {h_dim})")
        elif transform_type == 'linear':
            self.transform = nn.Linear(in_dim, out_dim)
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using Linear")
        elif transform_type == 'quat':
            # Use QuaternionLinear layer if dims are compatible
            check_quat_dim(in_dim, "InterLevelTransform Quat Input")
            check_quat_dim(out_dim, "InterLevelTransform Quat Output")
            self.transform = QuaternionLinear(in_dim, out_dim, bias=True)
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using QuaternionLinear")
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")

        # Initialize weights
        self.transform.apply(init_weights)

    def forward(self, v_rotated: torch.Tensor, v_boundaries_rotated: Optional[torch.Tensor], v_descriptor_rotated: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Applies the transformation to the rotated vectors."""
        has_boundaries = v_boundaries_rotated is not None and v_boundaries_rotated.numel() > 0

        # Apply transformation
        v_transformed = self.transform(v_rotated)
        v_boundaries_transformed = None # Initialize
        if has_boundaries:
            v_boundaries_transformed = self.transform(v_boundaries_rotated)
        # Descriptor should always be present
        v_descriptor_transformed = self.transform(v_descriptor_rotated) # Shape [1, 1, out_dim]

        # --- Stability Checks & Return ---
        outputs = [v_transformed, v_boundaries_transformed, v_descriptor_transformed]
        names = ["main", "boundaries", "descriptor"]
        final_outputs = []
        for i, out in enumerate(outputs):
            # Handle boundary case
            if i == 1 and not has_boundaries:
                final_outputs.append(None)
                continue
            # Check for None
            if out is None:
                raise ValueError(f"Transform output '{names[i]}' is None unexpectedly.")
            # Check for NaN/Inf
            if not torch.isfinite(out).all():
                logger.warning(f"NaN/Inf detected in Transform output ({names[i]}). Replacing with zeros.")
                out = torch.nan_to_num(out, nan=0.0)
            final_outputs.append(out)

        return tuple(final_outputs)


class WuBuNestingLevel(nn.Module):
    """Implements a single level of the WuBu Nesting architecture."""
    def __init__(self, level_idx: int, config: Dict):
        super().__init__()
        self.level_idx = level_idx
        self.dim = config["hyperbolic_dims"][level_idx]
        self.relative_vector_aggregation = config["relative_vector_aggregation"]
        self.dropout = config["dropout"]
        self.use_ld = config["use_level_descriptors"]
        self.use_spread = config["use_level_spread"]
        self.use_flow = config["use_tangent_flow"]

        # Minimum values for constrained parameters
        self.curvature_min = config.get("curvature_min_value", EPS)
        self.scale_min = config.get("scale_min_value", EPS)
        self.spread_min = config.get("spread_min_value", EPS)
        self.ld_init_scale = config.get("level_descriptor_init_scale", 0.01)

        # --- Curvature (c_i) ---
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

        # --- Scale (s_i) ---
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

        # --- Level Descriptor (ld_i) ---
        if self.use_ld:
            # Learnable vector in the tangent space
            self.level_descriptor_param = nn.Parameter(torch.randn(self.dim) * self.ld_init_scale)
        else:
            # Fixed zero vector if LD is not used
            self.register_buffer('level_descriptor_param', torch.zeros(self.dim))

        # --- Spread (sigma_i) ---
        if self.use_spread:
            # Default to initial_scales if initial_spread_values is None or too short
            initial_spread_values_list = config.get("initial_spread_values")
            if initial_spread_values_list is None or len(initial_spread_values_list) <= level_idx:
                logger.debug(f"L{level_idx}: Using initial_scale as initial_spread_value.")
                initial_spread_values_list = config["initial_scales"]
            # Check length again after defaulting
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
            # Fixed small value if spread is not used (log(EPS) ensures get_constrained_params returns approx spread_min)
            self.register_buffer('unconstrained_spread', torch.tensor(math.log(EPS)))

        # --- Tangent Combiner MLP ---
        # Calculates the input dimension based on enabled features
        combiner_input_dim = self.dim # Base dimension from v_tangent_in
        if config["relative_vector_aggregation"] != 'none':
            combiner_input_dim += self.dim # Add dim for aggregated relative vectors
        if self.use_ld:
            combiner_input_dim += self.dim # Add dim for level descriptor input
        if self.use_spread:
            combiner_input_dim += 1 # Add 1 for scalar spread input

        # Define the MLP structure
        combiner_hidden_dims = config.get("tangent_input_combination_dims", [max(16, combiner_input_dim // 2)])
        layers = []; in_d = combiner_input_dim
        for h_dim in combiner_hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_d = h_dim
        layers.append(nn.Linear(in_d, self.dim)) # Final layer maps back to level dimension
        self.tangent_combiner = nn.Sequential(*layers)
        self.tangent_combiner.apply(init_weights)

        # --- Optional Tangent Flow ---
        if self.use_flow:
            self.tangent_flow = TangentFlow(
                dim=self.dim,
                flow_type=config.get("tangent_flow_type", "mlp"),
                hidden_dim_ratio=config.get("tangent_flow_hidden_dim_ratio", 0.5),
                dropout=self.dropout
            )
            self.tangent_flow_scale = config.get("tangent_flow_scale", 1.0)
        else:
            self.tangent_flow = None
            self.tangent_flow_scale = 0.0

        # --- Intra-ball Processor (Placeholder) ---
        # This could be a hyperbolic MLP or attention layer in the future
        self.intra_ball_processor = nn.Identity()

        logger.info(f"WuBuNestingLevel {level_idx} (Dim {self.dim}) Init. Learn(c/s/sprd): {config['learnable_curvature']}/{config['learnable_scales']}/{config['learnable_spread']}, Use(LD/Sprd/Flow): {self.use_ld}/{self.use_spread}/{self.use_flow}, CombinerInDim: {combiner_input_dim}")

    def get_current_geometry(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the current curvature and scale parameters on the specified device."""
        # Use constrained parameter helper to ensure positivity
        current_c = get_constrained_params(self.unconstrained_curvature, self.curvature_min).to(device)
        current_s = get_constrained_params(self.unconstrained_scale, self.scale_min).to(device)
        return current_c, current_s

    def get_current_spread(self, device: torch.device) -> torch.Tensor:
        """Returns the current spread parameter on the specified device."""
        if not self.use_spread:
            # Return min value if spread is disabled for this level
            return torch.tensor(self.spread_min, device=device, dtype=torch.float) # Ensure correct dtype
        elif hasattr(self, 'unconstrained_spread'):
            # Calculate constrained spread if enabled and parameter exists
            return get_constrained_params(self.unconstrained_spread, self.spread_min).to(device)
        else:
            # Should not happen if init logic is correct
            logger.error(f"L{self.level_idx}: Spread requested but 'unconstrained_spread' not found!")
            return torch.tensor(self.spread_min, device=device, dtype=torch.float)

    def forward(self, v_tangent_in: torch.Tensor, relative_vectors_in: Optional[torch.Tensor], ld_tangent_in: Optional[torch.Tensor], sigma_in: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes input through the WuBu level.

        Args:
            v_tangent_in (Tensor): Main input tangent vector. Shape (B, S, Dim).
            relative_vectors_in (Optional[Tensor]): Aggregated relative boundary vectors from previous level. Shape (B, S, Dim) or None.
            ld_tangent_in (Optional[Tensor]): Level descriptor vector from previous level. Shape (B, S, Dim) or None.
            sigma_in (Optional[Tensor]): Spread parameter from previous level. Shape (B, S, 1) or (1,) or None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                - x_hyp_processed (Tensor): Hyperbolic point after processing within the ball. Shape (B, S, Dim).
                - v_tangent_out (Tensor): Output tangent vector (log map of x_hyp_processed). Shape (B, S, Dim).
                - ld_param_out (Tensor): This level's descriptor parameter (tangent vector). Shape (Dim).
                - sigma_param_out (Tensor): This level's spread parameter (scalar). Shape (1).
        """
        batch_size, seq_len, d_in = v_tangent_in.shape
        device = v_tangent_in.device
        model_dtype = next(self.parameters()).dtype # Get dtype from a parameter

        # --- Input Validation ---
        if d_in != self.dim:
            raise ValueError(f"L{self.level_idx} input dim mismatch: expected {self.dim}, got {d_in} for v_tangent_in")

        # --- Get Current Level Parameters ---
        current_c, current_scale = self.get_current_geometry(device)
        current_spread = self.get_current_spread(device) # Scalar tensor

        # --- Prepare Inputs for Tangent Combiner ---
        inputs_to_combine = [v_tangent_in]

        # Add aggregated relative vectors if enabled and provided
        if self.relative_vector_aggregation != 'none':
            if relative_vectors_in is not None and relative_vectors_in.numel() > 0:
                if relative_vectors_in.shape == (batch_size, seq_len, self.dim):
                    inputs_to_combine.append(relative_vectors_in.to(model_dtype))
                else:
                    logger.warning(f"L{self.level_idx}: Unexpected relative_vectors_in shape {relative_vectors_in.shape}. Expected {(batch_size, seq_len, self.dim)}. Using zeros.")
                    inputs_to_combine.append(torch.zeros_like(v_tangent_in))
            else: # If None or empty, append zeros
                if self.level_idx > 0: # Only warn if not the first level
                     logger.debug(f"L{self.level_idx}: Relative vectors input is None or empty. Using zeros.")
                inputs_to_combine.append(torch.zeros_like(v_tangent_in))

        # Add level descriptor input if enabled and provided
        if self.use_ld:
            if ld_tangent_in is None:
                if self.level_idx == 0:
                    logger.debug(f"L0: ld_tangent_in is None, as expected. Using zeros.")
                else:
                    logger.warning(f"L{self.level_idx}: Expected ld_tangent_in from previous level but received None. Using zeros.")
                inputs_to_combine.append(torch.zeros_like(v_tangent_in))
            elif ld_tangent_in.shape == (batch_size, seq_len, self.dim):
                inputs_to_combine.append(ld_tangent_in.to(model_dtype))
            else:
                logger.error(f"L{self.level_idx}: ld_tangent_in shape mismatch {ld_tangent_in.shape} vs expected {(batch_size, seq_len, self.dim)}. Using zeros.")
                inputs_to_combine.append(torch.zeros_like(v_tangent_in))

        # Add spread input if enabled and provided
        if self.use_spread:
            if sigma_in is None:
                if self.level_idx == 0:
                    logger.debug(f"L0: sigma_in is None, as expected. Using zeros.")
                else:
                    logger.warning(f"L{self.level_idx}: Expected sigma_in from previous level but received None. Using zeros.")
                sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device, dtype=model_dtype)
            # Handle different possible shapes for sigma_in (scalar, per-batch, per-item)
            elif sigma_in.numel() == 1: # Scalar spread from previous level
                sigma_in_tensor = sigma_in.expand(batch_size, seq_len, 1).to(model_dtype)
            elif sigma_in.shape == (batch_size, seq_len): # Spread per sequence item
                sigma_in_tensor = sigma_in.unsqueeze(-1).to(model_dtype)
            elif sigma_in.shape == (batch_size, seq_len, 1): # Already correct shape
                sigma_in_tensor = sigma_in.to(model_dtype)
            else: # Unexpected shape
                logger.error(f"L{self.level_idx}: sigma_in shape mismatch {sigma_in.shape}. Using zeros.")
                sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device, dtype=model_dtype)
            inputs_to_combine.append(sigma_in_tensor)

        # Concatenate all inputs for the combiner MLP
        combined_tangent_inputs = torch.cat(inputs_to_combine, dim=-1)

        # --- Stability Check ---
        if not torch.isfinite(combined_tangent_inputs).all():
            logger.warning(f"NaN/Inf detected in L{self.level_idx} combined tangent inputs before combiner MLP. Replacing.")
            combined_tangent_inputs = torch.nan_to_num(combined_tangent_inputs, nan=0.0)

        # Check if the concatenated dimension matches the MLP's expected input dimension
        expected_combiner_dim = self.tangent_combiner[0].in_features
        if combined_tangent_inputs.shape[-1] != expected_combiner_dim:
            raise ValueError(f"L{self.level_idx} TangentCombiner input dimension mismatch: expected {expected_combiner_dim}, got {combined_tangent_inputs.shape[-1]}")

        # --- Process Combined Tangent Vector ---
        # 1. Pass through combiner MLP
        v_tangent_combined = self.tangent_combiner(combined_tangent_inputs)

        # 2. Apply optional tangent flow
        flow_displacement = torch.zeros_like(v_tangent_combined)
        if self.use_flow and self.tangent_flow is not None:
            flow_displacement = self.tangent_flow(v_tangent_combined)
        # Add scaled flow displacement
        v_tangent_flowed = v_tangent_combined + flow_displacement * self.tangent_flow_scale

        # --- Stability Check ---
        if not torch.isfinite(v_tangent_flowed).all():
            logger.warning(f"NaN/Inf detected in L{self.level_idx} tangent vector post-flow. Replacing.")
            v_tangent_flowed = torch.nan_to_num(v_tangent_flowed, nan=0.0)

        # --- Project to Hyperbolic Space & Process ---
        v_for_exp_map = v_tangent_flowed
        x_hyp = HyperbolicUtils.exponential_map(v_for_exp_map, current_c.item()) # Pass scalar curvature
        x_hyp_processed = self.intra_ball_processor(x_hyp)
        x_hyp_processed = HyperbolicUtils.poincare_clip(x_hyp_processed, current_c.item()) # Pass scalar curvature

        # --- Map Back to Tangent Space ---
        v_tangent_out = HyperbolicUtils.logarithmic_map(x_hyp_processed, current_c.item()) # Pass scalar curvature

        # --- Stability Check ---
        if not torch.isfinite(v_tangent_out).all():
            logger.warning(f"NaN/Inf detected in L{self.level_idx} output tangent vector (v_tangent_out). Replacing.")
            v_tangent_out = torch.nan_to_num(v_tangent_out, nan=0.0)

        # --- Prepare Outputs for Next Level / Aggregation ---
        ld_param_out = self.level_descriptor_param # Shape (Dim)
        sigma_param_out = current_spread # Shape (1)

        # Ensure outputs have correct dtype
        v_tangent_out = v_tangent_out.to(model_dtype)
        ld_param_out = ld_param_out.to(model_dtype)
        sigma_param_out = sigma_param_out.to(model_dtype)

        return x_hyp_processed, v_tangent_out, ld_param_out, sigma_param_out

# =====================================================================
# WuBuNestingSequenceModel
# =====================================================================
class WuBuNestingSequenceModel(nn.Module):
    """WuBu Nesting model adapted for sequence modeling (Design Doc 3)."""
    def __init__(self, wubu_config: Dict, sequence_config: Dict):
        super().__init__()
        self.wubu_config = wubu_config
        self.sequence_config = sequence_config

        # --- Extract and Validate Config Parameters ---
        num_levels = wubu_config["num_levels"]
        hyperbolic_dims = wubu_config["hyperbolic_dims"]
        boundary_points = wubu_config["boundary_points_per_level"]
        rotation_types = wubu_config["rotation_types"]
        transform_types = wubu_config["transform_types"]
        transform_hdims = wubu_config["transform_hidden_dims"] # Can be None or list

        if len(hyperbolic_dims) != num_levels: raise ValueError("Length of hyperbolic_dims must match num_levels")
        if len(boundary_points) != num_levels: raise ValueError("Length of boundary_points_per_level must match num_levels")

        num_transitions = max(0, num_levels - 1)
        if len(rotation_types) != num_transitions: raise ValueError(f"Length of rotation_types ({len(rotation_types)}) must be num_levels - 1 ({num_transitions})")
        if len(transform_types) != num_transitions: raise ValueError(f"Length of transform_types ({len(transform_types)}) must be num_levels - 1 ({num_transitions})")
        # Ensure transform_hdims is a list of correct length, filling with None if needed
        if transform_hdims is None:
            transform_hdims = [None] * num_transitions
        elif len(transform_hdims) != num_transitions:
            raise ValueError(f"Length of transform_hidden_dims ({len(transform_hdims)}) must be num_levels - 1 ({num_transitions})")
        # Ensure None is used consistently if hidden dim is not specified (e.g., for linear transform)
        self.transform_hdims = [d if d is not None and d > 0 else None for d in transform_hdims]

        # --- Sequence Processing Components ---
        self.local_hidden_size = sequence_config["local_hidden_size"]
        self.decoder_memory_dim = sequence_config["decoder_memory_dim"]
        self.context_window = sequence_config["context_window"]
        self.vocab_size = sequence_config.get("vocab_size", 256)
        if self.vocab_size != 256:
            logger.warning(f"Sequence vocab_size set to {self.vocab_size}, but model typically uses 256 for bytes.")

        self.patcher = HAKMEMBabylonIndex()
        self.local_encoder = HAKMEMLocalEncoder(
            hidden_size=self.local_hidden_size,
            num_layers=sequence_config.get("num_local_encoder_layers", 2),
            num_heads=sequence_config.get("num_local_encoder_heads", 8),
            dropout=wubu_config.get("dropout", 0.1),
            n_gram_sizes=sequence_config.get("n_gram_sizes", [3, 4]),
            n_gram_vocab_size=sequence_config.get("n_gram_vocab_size", 30000)
        )

        # --- WuBu Nesting Components ---
        # Linear layer to project initial patch embeddings to the first tangent space
        self.to_first_tangent = nn.Linear(self.local_hidden_size, hyperbolic_dims[0])
        self.to_first_tangent.apply(init_weights)

        # WuBu Levels
        self.levels = nn.ModuleList(
            [WuBuNestingLevel(i, wubu_config) for i in range(num_levels)]
        )
        # Boundary Manifolds for each level
        self.boundaries = nn.ModuleList(
            [BoundaryManifold(i, boundary_points[i], hyperbolic_dims[i],
                              init_scale=wubu_config.get('level_descriptor_init_scale', 0.01)) # Use same scale for boundary init
             for i in range(num_levels)]
        )
        # Inter-level Rotations and Transformations
        self.rotations = nn.ModuleList()
        self.transforms = nn.ModuleList()
        if num_levels > 1:
            self.rotations = nn.ModuleList(
                [TangentSpaceRotation(hyperbolic_dims[i], rotation_types[i])
                 for i in range(num_transitions)]
            )
            self.transforms = nn.ModuleList(
                [InterLevelTransform(hyperbolic_dims[i], hyperbolic_dims[i+1], transform_types[i],
                                     self.transform_hdims[i], wubu_config.get("dropout", 0.1))
                 for i in range(num_transitions)]
            )

        # --- Aggregation and Final Projection ---
        self.aggregation_method = wubu_config["aggregation_method"]
        if self.aggregation_method == "concat_tangent":
            # Concatenate the output tangent vectors from all levels
            total_tangent_dim = sum(hyperbolic_dims)
            self.projection_to_decoder_memory = nn.Linear(total_tangent_dim, self.decoder_memory_dim)
            self.projection_to_decoder_memory.apply(init_weights)
            logger.info(f"Aggregation: Concatenating tangent outputs (Total Dim: {total_tangent_dim}) -> Decoder Memory Dim: {self.decoder_memory_dim}")
        else:
            # Add other aggregation methods here if needed (e.g., weighted sum, attention)
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not yet supported.")

        # --- Local Decoder (Using FULL Implementation) ---
        self.local_decoder = HAKMEMLocalDecoder(
            hidden_size=self.local_hidden_size, # Decoder internal dim matches encoder
            global_hidden_size=self.decoder_memory_dim, # Input memory dim from WuBu aggregation
            num_layers=sequence_config.get("num_local_decoder_layers", 4),
            num_heads=sequence_config.get("num_local_decoder_heads", 8),
            dropout=wubu_config.get("dropout", 0.1),
            use_hierarchical_pred=sequence_config.get("use_hierarchical_decoder", False),
            # Allow decoder context potentially longer than input context for generation
            max_decode_len=max(self.context_window * 2, 2048) # Example: twice input or 2048
        )
        logger.info("WuBuNestingSequenceModel Initialized (Full Decoder/Optimizer, v0.02 with RelVecs).")

    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the WuBu Nesting Sequence Model.

        Args:
            byte_seq (Tensor): Input byte sequence. Shape (Batch, SeqLen).
            target_byte_seq (Optional[Tensor]): Target sequence for teacher forcing during training or
                                                 the sequence generated so far during inference.
                                                 Shape (Batch, TargetSeqLen). If None or empty, returns zeros.

        Returns:
            Tensor: Output logits. Shape (Batch, TargetSeqLen, VocabSize).
        """
        batch_size, seq_len = byte_seq.shape
        device = byte_seq.device
        model_dtype = next(self.parameters()).dtype # Get model's dtype

        # --- 1. Patching and Local Encoding ---
        batch_patch_repr_list = [] # Stores encoded patches for each batch item
        num_patches_per_item = [] # Stores number of patches for each batch item
        valid_batch_indices = [] # Indices of batch items that yielded patches
        max_num_patches = 0 # Max patches found in any batch item

        for i in range(batch_size):
            seq = byte_seq[i] # Get sequence for this batch item
            # Create patches (list of (tensor, entropy))
            patches_with_entropy = self.patcher.create_patches(seq)

            if patches_with_entropy:
                # Ensure patches are on the correct device before encoding
                patches_on_device = [(p.to(device), e) for p, e in patches_with_entropy]
                # Encode the patches for this sequence item
                patch_repr_single = self.local_encoder(patches_on_device) # Shape [1, NumPatches, LocalHidden]

                # --- Stability Check ---
                if not torch.isfinite(patch_repr_single).all():
                    logger.warning(f"NaN/Inf detected in local encoder output for batch item {i}. Replacing.")
                    patch_repr_single = torch.nan_to_num(patch_repr_single, nan=0.0)

                num_p = patch_repr_single.size(1) # Number of patches encoded
                if num_p > 0:
                    # Store the representation (remove batch dim) and count
                    batch_patch_repr_list.append(patch_repr_single.squeeze(0)) # Shape [NumPatches, LocalHidden]
                    num_patches_per_item.append(num_p)
                    valid_batch_indices.append(i) # Mark this index as valid
                    max_num_patches = max(max_num_patches, num_p)
                else:
                    # If encoding resulted in 0 patches (e.g., all empty after filtering)
                    num_patches_per_item.append(0)
            else:
                # If no patches were created by the patcher
                num_patches_per_item.append(0)

        # Determine target length (needed for output shape even if no valid inputs)
        target_len = 0
        if target_byte_seq is not None: target_len = target_byte_seq.size(1)
        # Handle case where no valid patches were found in the entire batch
        if not valid_batch_indices:
            logger.warning(f"No valid patches found for any item in the batch (Batch Size: {batch_size}). Returning zero logits.")
            return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)

        num_valid = len(valid_batch_indices)
        # Handle case where patches were found, but maybe max_num_patches is still 0 (edge case?)
        if max_num_patches == 0:
            logger.warning(f"Valid batch items exist, but max_num_patches is zero. Returning zero logits.")
            return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)

        # Pad patch representations to form a batch tensor for WuBu processing
        # Only includes valid batch items. Shape: [NumValid, MaxPatches, LocalHidden]
        padded_patch_repr = torch.zeros(num_valid, max_num_patches, self.local_hidden_size, device=device, dtype=model_dtype)
        # Create padding mask (True where padded, False where valid data)
        # Shape: [NumValid, MaxPatches]
        memory_padding_mask = torch.ones(num_valid, max_num_patches, dtype=torch.bool, device=device)

        valid_item_counter = 0
        for i in range(batch_size): # Iterate original batch indices
            if i in valid_batch_indices:
                # Get the representation and num patches for this valid item
                patch_repr_tensor = batch_patch_repr_list[valid_item_counter]
                num_p = num_patches_per_item[i]
                if num_p > 0:
                    # Copy data into padded tensor
                    padded_patch_repr[valid_item_counter, :num_p, :] = patch_repr_tensor
                    # Mark valid positions in mask as False
                    memory_padding_mask[valid_item_counter, :num_p] = False
                valid_item_counter += 1

        # --- 2. Project to First Tangent Space ---
        # Project the initial patch representations to the tangent space of the first level
        current_tangent_main = self.to_first_tangent(padded_patch_repr) # Shape: [NumValid, MaxPatches, Dim0]

        # --- 3. Iterate Through WuBu Levels ---
        level_tangent_outputs = [] # Store tangent output from each level for final aggregation
        aggregated_relative_vectors_for_next_level = None # Start with no relative vectors for Level 0
        current_ld_tangent = None # No descriptor input for Level 0
        current_sigma = None      # No spread input for Level 0

        for i in range(self.wubu_config["num_levels"]):
            level_module = self.levels[i]
            boundary_module = self.boundaries[i]

            # --- Forward pass through the WuBu level ---
            _, v_tangent_i_out, ld_i_param, sigma_i_param = level_module(
                v_tangent_in=current_tangent_main,
                relative_vectors_in=aggregated_relative_vectors_for_next_level, # Pass aggregated relatives from previous step
                ld_tangent_in=current_ld_tangent,                               # Pass transformed LD from previous step
                sigma_in=current_sigma                                           # Pass spread from previous level
            )
            # v_tangent_i_out: Tangent vector output of this level (B_valid, S_patch, Dim_i)
            # ld_i_param: Learnable descriptor for this level (Dim_i)
            # sigma_i_param: Learnable spread for this level (scalar)

            # Store the tangent output for final aggregation
            level_tangent_outputs.append(v_tangent_i_out)

            # --- Inter-Level Transformation (if not the last level) ---
            if i < self.wubu_config["num_levels"] - 1:
                rotation_module = self.rotations[i]
                transform_module = self.transforms[i]

                # Get tangent vectors for boundary points *at the origin* for this level
                v_boundaries_tangent_origin = boundary_module.get_tangent_vectors_at_origin()
                num_boundary_points_i = v_boundaries_tangent_origin.shape[0] if v_boundaries_tangent_origin is not None else 0

                if v_boundaries_tangent_origin is not None:
                    v_boundaries_tangent_origin = v_boundaries_tangent_origin.to(device=device, dtype=model_dtype)
                    # Expected shape: [NumBoundaryPoints_i, Dim_i]

                # Ensure descriptor param is on correct device/dtype before rotation/transform
                ld_i_param_dev = ld_i_param.to(device=device, dtype=model_dtype).view(1, 1, -1) # Reshape to (1,1,Dim)

                # Rotate the main output tangent, the boundary points (if any), and the level descriptor
                v_main_rotated, v_boundaries_rotated, ld_rotated = rotation_module(
                    v_main=v_tangent_i_out,
                    v_boundaries_tangent=v_boundaries_tangent_origin, # Pass the tangent boundary points from this level
                    v_descriptor=ld_i_param_dev                     # Pass this level's descriptor
                )
                # v_boundaries_rotated will be None if v_boundaries_tangent_origin was None or num_boundary_points_i = 0

                # Apply non-rotational transformation to map to the *next* level's tangent space
                v_next_tangent_main, v_boundaries_transformed, ld_next_tangent = transform_module(
                    v_rotated=v_main_rotated,
                    v_boundaries_rotated=v_boundaries_rotated, # Pass rotated boundaries
                    v_descriptor_rotated=ld_rotated            # Pass rotated descriptor
                )
                # v_next_tangent_main shape: (B_valid, S_patch, Dim_{i+1})
                # v_boundaries_transformed shape: (NumBoundaryPoints_i, Dim_{i+1}) or None
                # ld_next_tangent shape: (1, 1, Dim_{i+1})

                # --- ** Calculate and Aggregate Relative Vectors for Next Level ** ---
                if v_boundaries_transformed is not None and num_boundary_points_i > 0:
                    # Expand dimensions for broadcasting subtraction
                    # main: (B_valid, S_patch, 1, Dim_{i+1})
                    main_expanded = v_next_tangent_main.unsqueeze(2)
                    # boundaries: (1, 1, NumBoundaryPoints_i, Dim_{i+1})
                    boundaries_expanded = v_boundaries_transformed.unsqueeze(0).unsqueeze(0)

                    # Calculate relative vectors: main - boundary
                    relative_vectors_calc = main_expanded - boundaries_expanded # Shape: (B_valid, S_patch, NumBoundaryPoints_i, Dim_{i+1})

                    # Aggregate relative vectors based on configuration
                    agg_method = self.wubu_config.get("relative_vector_aggregation", "none")
                    if agg_method == 'mean':
                        aggregated_relative_vectors_for_next_level = torch.mean(relative_vectors_calc, dim=2) # Shape: (B_valid, S_patch, Dim_{i+1})
                    elif agg_method == 'sum':
                        aggregated_relative_vectors_for_next_level = torch.sum(relative_vectors_calc, dim=2) # Shape: (B_valid, S_patch, Dim_{i+1})
                    elif agg_method == 'none':
                        aggregated_relative_vectors_for_next_level = None # Don't pass relative vectors
                    else:
                        logger.warning(f"Unsupported relative_vector_aggregation '{agg_method}'. Defaulting to 'none'.")
                        aggregated_relative_vectors_for_next_level = None

                    # Stability check
                    if aggregated_relative_vectors_for_next_level is not None and not torch.isfinite(aggregated_relative_vectors_for_next_level).all():
                         logger.warning(f"NaN/Inf detected in aggregated relative vectors for L{i+1}. Replacing with zeros.")
                         aggregated_relative_vectors_for_next_level = torch.zeros_like(v_next_tangent_main)

                else:
                    # No boundary points or transformation resulted in None
                    logger.debug(f"No valid boundaries to calculate relative vectors for L{i+1}.")
                    aggregated_relative_vectors_for_next_level = None

                # --- Prepare inputs for the *next* level's forward call ---
                current_tangent_main = v_next_tangent_main
                # Expand the transformed level descriptor for the next level's input combiner
                current_ld_tangent = ld_next_tangent.expand(num_valid, max_num_patches, -1) # Shape (B_valid, S_patch, Dim_{i+1})
                # Pass this level's spread parameter to the next level
                current_sigma = sigma_i_param # Scalar tensor

        # --- 4. Aggregate Level Outputs ---
        if self.aggregation_method == "concat_tangent":
            # Concatenate tangent outputs from all levels along the feature dimension
            # Ensure all tensors in level_tangent_outputs have the same Batch x Seq dimensions
            # Check shapes before concat: should be [(B_valid, S_patch, D0), (B_valid, S_patch, D1), ...]
            try:
                aggregated_repr = torch.cat(level_tangent_outputs, dim=-1) # Shape: (B_valid, S_patch, Sum(Dims))
            except RuntimeError as e:
                logger.error(f"Error concatenating level outputs: {e}")
                logger.error(f"Shapes: {[t.shape for t in level_tangent_outputs]}")
                # Handle error: return zeros? Or re-raise?
                return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)
        else:
            # Implement other aggregation methods if needed
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not implemented.")

        # --- 5. Project to Decoder Memory ---
        # Project the aggregated representation to the dimension expected by the decoder's memory input
        decoder_memory = self.projection_to_decoder_memory(aggregated_repr) # Shape: [NumValid, MaxPatches, DecMemDim]

        # --- Stability Check ---
        if not torch.isfinite(decoder_memory).all():
            logger.warning("NaN/Inf detected in projected decoder memory. Replacing.")
            decoder_memory = torch.nan_to_num(decoder_memory, nan=0.0)

        # --- 6. Decode ---
        # Check if target sequence is provided (required for training/teacher-forcing generation)
        if target_byte_seq is None:
            logger.debug("Forward called without target_byte_seq for decoding. Returning zeros.")
            # Cannot decode without target input for the decoder transformer
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)
        if target_byte_seq.size(1) == 0:
            logger.debug("Forward called with empty target_byte_seq. Returning zeros.")
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)

        # Select only the target sequences corresponding to the valid inputs processed by WuBu
        valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
        if num_valid > 0:
            # Select the relevant rows from the target sequence batch
            valid_target_byte_seq = torch.index_select(target_byte_seq, 0, valid_indices_tensor).to(device).long()
        else: # Should not happen if we checked valid_batch_indices earlier, but safety check
            valid_target_byte_seq = torch.empty(0, target_len, dtype=torch.long, device=device)


        if num_valid > 0:
            # Pass the valid targets, the computed memory, and the padding mask to the decoder
            # memory_padding_mask (True indicates padding) is used for cross-attention
            byte_logits_valid = self.local_decoder(
                tgt_byte_seq=valid_target_byte_seq,   # Shape (NumValid, TargetSeqLen)
                memory=decoder_memory,                # Shape (NumValid, MaxPatches, DecMemDim)
                memory_key_padding_mask=memory_padding_mask # Shape (NumValid, MaxPatches)
            ) # Output Shape: (NumValid, TargetSeqLen, VocabSize)

            # --- Stability Check ---
            if not torch.isfinite(byte_logits_valid).all():
                logger.warning("NaN/Inf detected in decoder logits output. Replacing.")
                byte_logits_valid = torch.nan_to_num(byte_logits_valid, nan=0.0)
        else:
            # If no valid inputs, create empty logits tensor with correct dimensions
            byte_logits_valid = torch.empty((0, target_len, self.vocab_size), device=device, dtype=torch.float32)

        # --- 7. Reconstruct Full Batch Output ---
        # Create a zero tensor for the full batch output
        final_byte_logits = torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)

        if num_valid > 0 and byte_logits_valid.numel() > 0:
            try:
                # Use index_copy_ to place the logits for valid items back into the full batch tensor
                # dim 0: batch dimension
                # index: tensor containing the original batch indices that were valid
                # source: the logits calculated for the valid items
                if byte_logits_valid.shape[0] == valid_indices_tensor.shape[0]:
                    final_byte_logits.index_copy_(0, valid_indices_tensor, byte_logits_valid)
                else:
                    # This indicates a logic error somewhere above
                    logger.error(f"Shape mismatch during output reconstruction. Target Batch={batch_size}, Valid Items={num_valid}, Logits Shape={byte_logits_valid.shape}, Indices Shape={valid_indices_tensor.shape}. Cannot perform index_copy.")
                    # Output remains zeros in case of error
            except IndexError as e:
                logger.error(f"Index error during logit reconstruction: {e}. NumValid: {num_valid}, IdxShape: {valid_indices_tensor.shape}, LogitsShape: {byte_logits_valid.shape}")
                # Output remains zeros
            except Exception as e_scatter:
                logger.error(f"Unexpected error scattering logits: {e_scatter}")
                # Output remains zeros

        # --- Final Stability Check ---
        if not torch.isfinite(final_byte_logits).all():
            logger.error("NaN/Inf detected in final scattered logits! Replacing with zeros.")
            final_byte_logits = torch.nan_to_num(final_byte_logits, nan=0.0)

        return final_byte_logits


    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        """
        Computes cross-entropy loss for sequence prediction.
        Handles sequence shifting for next-token prediction.

        Args:
            logits (Tensor): Raw output logits from the model. Shape (Batch, SeqLen, VocabSize).
            targets (Tensor): Target byte sequence. Shape (Batch, SeqLen). Long dtype.
            mask (Optional[Tensor]): Padding mask for the target sequence. Shape (Batch, SeqLen).
                                      True indicates a padded position (should be ignored in loss).
            smoothing (float): Label smoothing factor (0.0 to disable).

        Returns:
            Tensor: Scalar loss value (average over non-masked elements).
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Loss is calculated based on predicting the *next* token.
        # If sequence length is 1 or less, we cannot form (input, target) pairs.
        if seq_len <= 1:
            # Return zero loss with gradient enabled if no prediction possible
            return torch.tensor(0.0, device=logits.device, requires_grad=True, dtype=logits.dtype)

        # Shift logits and targets for next-token prediction:
        # Logits for predicting target[t] are based on input up to time t-1, found at logits[:, t-1, :]
        # Target for logits[:, t-1, :] is targets[:, t]
        logits_shifted = logits[:, :-1, :].contiguous() # Shape: (Batch, SeqLen-1, VocabSize)
        targets_shifted = targets[:, 1:].contiguous()  # Shape: (Batch, SeqLen-1)

        # Flatten logits and targets for cross_entropy input
        logits_flat = logits_shifted.view(-1, vocab_size) # Shape: (Batch * (SeqLen-1), VocabSize)
        targets_flat = targets_shifted.view(-1)          # Shape: (Batch * (SeqLen-1))

        # Ensure target values are valid indices (0 to vocab_size-1)
        # This should ideally not be necessary if data is clean, but acts as a safeguard.
        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)

        # --- Stability Check: Logits before loss calculation ---
        if not torch.isfinite(logits_flat).all():
            num_nan = torch.isnan(logits_flat).sum().item()
            num_inf = torch.isinf(logits_flat).sum().item()
            logger.error(f"NaN/Inf detected in logits passed to compute_loss (NaNs: {num_nan}, Infs: {num_inf}). Returning high loss.")
            # Return a high loss value that requires grad to signal error upstream
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        # Calculate loss (either standard CE or with label smoothing)
        loss: torch.Tensor
        if smoothing > 0.0 and 0.0 < smoothing < 1.0:
            # Label smoothing: Create target distribution
            with torch.no_grad():
                smooth_val_on = 1.0 - smoothing
                # Distribute smoothing mass among other classes
                smooth_val_off = smoothing / max(1, vocab_size - 1) # Avoid division by zero if vocab_size=1
                # Initialize distribution with off-value
                true_dist = torch.full_like(logits_flat, smooth_val_off)
                # Set the target index to the on-value
                true_dist.scatter_(1, targets_flat.unsqueeze(1), smooth_val_on)

            # Calculate KL divergence (equivalent to CE with smoothed labels)
            log_probs = F.log_softmax(logits_flat, dim=-1) # Calculate log probabilities
            # Loss is the negative sum of target_dist * log_probs
            loss = -(true_dist * log_probs).sum(dim=-1) # Shape: (Batch * (SeqLen-1))
        else:
            # Standard cross-entropy loss
            # Use float32 logits for F.cross_entropy stability, targets must be long
            loss = F.cross_entropy(logits_flat.float(), targets_flat.long(), reduction='none') # Shape: (Batch * (SeqLen-1))

        # --- Stability Check: Loss after calculation ---
        if not torch.isfinite(loss).all():
            logger.error(f"NaN/Inf loss calculated by cross_entropy/smoothing. Returning high loss.")
            # Replace non-finite values in the loss tensor before potential masking
            loss = torch.nan_to_num(loss, nan=100.0, posinf=100.0, neginf=-100.0) # Use large finite values
            # Return a high loss value to signal error? Or rely on masking?
            # Let's try masking first. If mean is still bad, handle below.
            # return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        # Apply mask if provided
        mean_loss: torch.Tensor
        if mask is not None:
            # Align mask with shifted targets/logits
            mask_shifted = mask[:, 1:].contiguous() # Shape: (Batch, SeqLen-1)
            if mask_shifted.shape == targets_shifted.shape:
                mask_flat = mask_shifted.view(-1) # Shape: (Batch * (SeqLen-1))
                # Ensure mask is boolean (True means ignore/mask out)
                mask_flat_bool = mask_flat.bool() if mask_flat.dtype == torch.bool else mask_flat > 0
                # Apply mask: set loss to 0 for masked positions
                loss = loss.masked_fill(mask_flat_bool, 0.0)
                # Calculate mean loss only over non-masked elements
                num_active_elements = (~mask_flat_bool).sum()
                if num_active_elements.item() > 0:
                    mean_loss = loss.sum() / num_active_elements
                else: # Handle case where all elements are masked
                    logger.warning("All target elements masked in loss calculation. Returning zero loss.")
                    mean_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            else:
                # Mask shape mismatch - ignore the mask and average over all elements
                logger.warning(f"Mask shape mismatch ({mask.shape[-1]}) vs target sequence length ({targets.shape[-1]}) for loss calculation. Ignoring mask.")
                mean_loss = loss.mean()
        else:
            # No mask provided, calculate mean over all elements
            mean_loss = loss.mean()

        # --- Final Stability Check: Mean Loss ---
        if not torch.isfinite(mean_loss):
            logger.error(f"NaN/Inf final mean loss detected. Returning high loss.")
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        return mean_loss


    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None, repetition_penalty: float = 1.1, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generates sequences autoregressively starting from seed_bytes.

        Args:
            seed_bytes (Tensor): Initial sequence tensor. Shape (Batch, SeedLen). Long dtype.
            max_length (int): Maximum number of *new* bytes to generate.
            temperature (float): Softmax temperature for sampling. Lower values make it greedier.
            sampling_config (Optional[SamplerConfig]): Configuration for adaptive temperature.
            repetition_penalty (float): Penalty > 1.0 discourages repeating tokens. 1.0 disables.
            top_k (Optional[int]): Keep only top k tokens by probability. 0 disables.
            top_p (Optional[float]): Keep smallest set of tokens whose cumulative probability >= p. 0.0 disables.

        Returns:
            Tensor: Generated sequences including the seed. Shape (Batch, SeedLen + GeneratedLen). Long dtype.
        """
        self.eval() # Set model to evaluation mode
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype # Not strictly needed for generation, but consistent

        # Ensure seed is on the correct device and is long type
        if seed_bytes.device != device: seed_bytes = seed_bytes.to(device)
        seed_bytes = seed_bytes.long()
        batch_size, seed_len = seed_bytes.size()

        if seed_len == 0:
            logger.warning("Empty seed provided for generation. Returning empty tensor.")
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)

        # Initialize the generated sequence with the seed
        generated_sequence = seed_bytes.clone()

        # Default sampling config if not provided
        if sampling_config is None: sampling_config = SamplerConfig()

        # Temperature settings
        base_temperature = max(temperature, EPS) # Ensure temperature is positive
        min_temp = max(0.1, base_temperature * 0.5) # Min adaptive temp
        max_temp = min(2.0, base_temperature * 1.5) # Max adaptive temp

        # Disable tqdm progress bar for non-main processes or short generation/large batches
        # Note: Inference usually runs on a single process, so is_main_process() might not be needed here.
        # Let's keep it simple: disable if batch > 8 or length < 20
        disable_tqdm = batch_size > 8 or max_length < 20
        gen_iterator = tqdm(range(max_length), desc="Generating", disable=disable_tqdm, total=max_length, unit="byte", leave=False)

        # Autoregressive generation loop
        for step in gen_iterator:
            current_context = generated_sequence.long() # Use the sequence generated so far
            context_len = current_context.size(1)

            # --- Model Forward Pass ---
            # Disable AMP during generation for stability, even if used in training
            amp_context = amp.autocast(device_type=device.type, enabled=False)
            with torch.no_grad(), amp_context:
                # Pass current context as both input and target. The causal mask in the
                # decoder ensures it only uses previous tokens to predict the next one.
                logits_all = self(byte_seq=current_context, target_byte_seq=current_context)
                # logits_all shape: (Batch, context_len, VocabSize)

            # Check if logits generation failed or produced NaNs
            if logits_all is None or logits_all.numel() == 0 or logits_all.shape[1] == 0:
                logger.warning(f"Logits generation failed at step {step}. Stopping generation.")
                break # Stop generation if model output is invalid
            if not torch.isfinite(logits_all).all():
                logger.warning(f"NaN/Inf detected in generated logits at step {step}. Using uniform distribution for this step.")
                # Replace non-finite logits with zeros, leading to uniform probabilities after softmax
                logits_all = torch.zeros_like(logits_all)

            # Get logits for the *next* byte prediction.
            # This corresponds to the output logits at the position of the *last* input token.
            next_byte_logits = logits_all[:, -1, :].float() # Shape: (BatchSize, VocabSize). Use float32 for sampling stability.

            # Prepare tensor to store the chosen next byte index for each batch item
            next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

            # --- Sampling Logic (per batch item) ---
            for i in range(batch_size):
                current_logits = next_byte_logits[i].clone() # Work with a copy
                current_seq = generated_sequence[i] # Sequence for this batch item so far
                current_seq_len = current_seq.size(0)

                # 1. Apply repetition penalty
                if repetition_penalty > 1.0 and current_seq_len > 0:
                    # Find unique bytes seen in the sequence so far
                    seen_bytes = torch.unique(current_seq)
                    # Penalize the logits corresponding to these seen bytes
                    for byte_val_tensor in seen_bytes:
                        byte_val = byte_val_tensor.item()
                        if 0 <= byte_val < self.vocab_size:
                            # Divide positive logits, multiply negative logits
                            if current_logits[byte_val] > 0:
                                current_logits[byte_val] /= repetition_penalty
                            else:
                                current_logits[byte_val] *= repetition_penalty # Make negative logits more negative

                # 2. Adaptive temperature based on prediction entropy
                adaptive_temp = base_temperature
                try:
                    # Calculate probabilities and entropy safely
                    probs_orig = F.softmax(current_logits, dim=-1)
                    if torch.isnan(probs_orig).any():
                        # If softmax results in NaN (e.g., all logits are -inf), entropy is max possible
                        entropy = math.log2(self.vocab_size)
                        logger.debug(f"Item {i} step {step}: NaN in probs_orig, using max entropy.")
                    else:
                        # Calculate entropy: -sum(p * log2(p))
                        entropy = -torch.sum(probs_orig * torch.log2(probs_orig + EPS)).item()

                    # Adjust temperature based on entropy thresholds
                    if entropy < sampling_config.low_entropy_threshold: # Low entropy -> more confident -> reduce temp (be greedier)
                        adaptive_temp *= 0.8
                    elif entropy > sampling_config.medium_entropy_threshold: # High entropy -> less confident -> increase temp (more exploration)
                        adaptive_temp *= 1.1
                    # Clamp adaptive temperature within bounds
                    adaptive_temp = max(min_temp, min(adaptive_temp, max_temp))
                except Exception as e_entropy:
                    logger.warning(f"Error calculating entropy/adaptive temp for item {i} step {step}: {e_entropy}. Using base temperature.")
                    adaptive_temp = base_temperature

                # 3. Apply temperature scaling
                scaled_logits = current_logits / adaptive_temp
                filtered_logits = scaled_logits # Start filtering from scaled logits

                # 4. Apply Top-K filtering
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1)) # Ensure k is valid
                    if k > 0:
                        # Find the k-th largest logit value
                        top_k_threshold = torch.topk(filtered_logits, k)[0][..., -1, None] # Shape (1,)
                        # Create mask for logits below the threshold
                        indices_to_remove = filtered_logits < top_k_threshold
                        # Set removed logits to -inf
                        filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))
                    else: # Should not happen with min(top_k, size) if size > 0
                        filtered_logits.fill_(-float('Inf')) # Mask all if k=0

                # 5. Apply Top-P (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    # Sort logits descending to find nucleus
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Create mask for tokens outside the nucleus (cumulative prob > p)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the mask: always keep at least the token with the highest probability
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0 # Ensure first element is never removed
                    # Scatter the mask back to the original logit order
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    # Set removed logits to -inf
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))

                # 6. Sample from the final filtered distribution
                probs_final = F.softmax(filtered_logits, dim=-1)

                # Handle potential NaN/Inf/Zero sum in final probabilities
                if torch.isnan(probs_final).any() or torch.isinf(probs_final).any() or probs_final.sum() < EPS:
                    logger.warning(f"Invalid final probability distribution for item {i} step {step}. Sampling uniformly.")
                    # Fallback to uniform distribution if probs are invalid
                    probs_final = torch.ones_like(current_logits) / current_logits.size(-1)

                # Choose sampling method: greedy or multinomial
                if temperature <= EPS: # Use greedy sampling for zero temperature
                    next_byte_idx = torch.argmax(probs_final)
                else: # Use multinomial sampling otherwise
                    next_byte_idx = torch.multinomial(probs_final, num_samples=1).squeeze(-1)

                # Store the chosen index
                next_byte_indices[i] = next_byte_idx.item()

            # Append the newly generated bytes for all items in the batch
            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)

            # Update progress bar description
            if not disable_tqdm:
                gen_iterator.set_description(f"Generating (Len {generated_sequence.size(1)})")

        # Clean up progress bar
        if not disable_tqdm: gen_iterator.close()

        return generated_sequence


# =====================================================================
# ByteTokenizer, ByteIterableDataset, Trainer
# =====================================================================
class ByteTokenizer:
    """Simple stateless tokenizer for converting between text and utf-8 byte sequences."""
    def encode(self, text: str) -> List[int]:
        """Encodes text to a list of UTF-8 byte values."""
        return list(text.encode('utf-8', errors='replace')) # Replace invalid chars

    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        """Decodes a sequence of bytes (int or tensor) back to text."""
        valid_bytes = []
        for b in byte_sequence:
            try:
                # Handle both int and tensor inputs
                val = b.item() if isinstance(b, torch.Tensor) else int(b)
            except Exception:
                # Skip invalid elements gracefully
                continue
            # Ensure byte value is valid (0-255)
            if 0 <= val <= 255:
                valid_bytes.append(val)
        # Decode the collected valid bytes using UTF-8, replacing errors
        return bytes(valid_bytes).decode('utf-8', errors='replace')

class ByteIterableDataset(IterableDataset):
    """
    An IterableDataset for efficiently reading byte sequences from a NumPy (.npy) file.
    Handles distributed training by splitting data based on worker ID.
    Provides (context, target) pairs where target is context shifted by one byte.
    """
    def __init__(self, npy_file_path: str, context_size: int = 256, data_fraction: float = 1.0):
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"Dataset file not found: {npy_file_path}")
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if not (0.0 < data_fraction <= 1.0):
            raise ValueError("data_fraction must be between 0 (exclusive) and 1 (inclusive)")

        self.npy_file_path = npy_file_path
        self.context_size = context_size
        self.data_fraction = data_fraction
        self.full_data_size = 0
        self.data_size = 0 # Effective size after applying fraction
        self.num_possible_samples = 0 # Number of valid starting indices
        self.data_dtype = np.uint8 # Assuming byte data
        self.seed = None # Base seed for shuffling (set by Trainer)
        self.epoch = 0   # Current epoch (set by Trainer)

        try:
            # Read only header to get shape and dtype quickly without loading data
            with open(self.npy_file_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                # _read_array_header is internal, but efficient here
                shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)

            if len(shape) != 1:
                raise ValueError(f"Dataset file expected to be 1D array, but found shape {shape}")
            self.full_data_size = shape[0]
            self.data_dtype = dtype
            # Ensure dtype is suitable (e.g., uint8 for bytes)
            if self.data_dtype != np.uint8:
                logger.warning(f"Dataset dtype is {self.data_dtype}, expected uint8. Ensure this is intended.")

            if self.full_data_size == 0:
                raise ValueError("Dataset file appears empty (size 0).")

            # Calculate effective data size based on fraction
            self.data_size = int(self.full_data_size * self.data_fraction)

            # Ensure effective size is large enough for at least one context+target window
            # Need context_size + 1 bytes for one sample (context[0..N-1], target[1..N])
            if self.data_size <= self.context_size:
                raise ValueError(f"Effective data size ({self.data_size:,}) after fraction ({self.data_fraction:.1%}) is not larger than context size ({self.context_size:,}). No samples possible.")

            # Calculate the number of possible starting indices for a sample
            # Max start index is data_size - (context_size + 1)
            self.num_possible_samples = max(0, self.data_size - self.context_size)

            if self.num_possible_samples == 0:
                raise ValueError(f"No samples possible with ContextSize={self.context_size:,} and EffectiveDataSize={self.data_size:,}. Check context_size and data_fraction.")

            logger.info(f"Dataset '{os.path.basename(npy_file_path)}': Effective Size={self.data_size:,}/{self.full_data_size:,} bytes ({self.data_fraction:.1%}), Num Samples={self.num_possible_samples:,} (Ctx: {self.context_size:,}), DType: {self.data_dtype}")

        except ImportError:
            logger.error("NumPy is required for ByteIterableDataset.")
            raise
        except Exception as e:
            logger.error(f"Error reading dataset metadata from {npy_file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        """Provides an estimated length based on possible samples and workers (for progress bars)."""
        # Note: Actual number yielded might differ slightly due to worker rounding/drop_last.
        if self.num_possible_samples == 0: return 0

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        world_size = get_world_size() if is_initialized() else 1
        total_effective_workers = num_workers * world_size

        # Estimate samples per effective worker
        return self.num_possible_samples // total_effective_workers

    def __iter__(self):
        """Iterator yielding (context, target) tensors for the current worker."""
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rank = get_rank() if is_initialized() else 0
        world_size = get_world_size() if is_initialized() else 1

        # Check if any samples are possible at all
        if self.num_possible_samples == 0:
            return iter([]) # Return empty iterator

        # --- Determine sample indices for this worker ---
        # Total number of effective workers across all nodes
        total_effective_workers = num_workers * world_size
        # Global worker ID across all nodes/workers
        global_worker_id = rank * num_workers + worker_id

        # Calculate number of samples per global worker
        num_samples_per_global_worker = self.num_possible_samples // total_effective_workers
        remainder = self.num_possible_samples % total_effective_workers

        # Determine the start and end sample index for this global worker
        start_sample_idx = global_worker_id * num_samples_per_global_worker + min(global_worker_id, remainder)
        end_sample_idx = start_sample_idx + num_samples_per_global_worker + (1 if global_worker_id < remainder else 0)
        # Ensure end index doesn't exceed total possible samples
        end_sample_idx = min(end_sample_idx, self.num_possible_samples)

        # Initialize data access variables
        bytes_data = None
        mmap_handle = None

        try:
            # Use mmap_mode for efficient memory usage, especially with large files
            bytes_data = np.load(self.npy_file_path, mmap_mode='r')
            # Store mmap handle for explicit closing later (important on some OS)
            if hasattr(bytes_data, '_mmap') and bytes_data._mmap is not None:
                mmap_handle = bytes_data._mmap
            # Check if the loaded array is valid
            if bytes_data is None or bytes_data.size == 0:
                logger.error(f"Worker {worker_id} Rank {rank}: Failed to load or empty data from {self.npy_file_path}")
                return iter([])

            # Check if this worker has any samples assigned
            if start_sample_idx >= end_sample_idx:
                logger.debug(f"Worker {worker_id} Rank {rank}: No samples assigned ({start_sample_idx} >= {end_sample_idx}).")
                return iter([]) # No samples for this worker

            # Generate the list of sample starting indices for this worker
            worker_indices = np.arange(start_sample_idx, end_sample_idx, dtype=np.int64)

            # Shuffle indices based on seed, epoch, and worker ID for reproducibility
            base_seed = self.seed if self.seed is not None else int(time.time())
            current_epoch = self.epoch
            # Combine seeds for unique shuffling per worker per epoch
            seed_for_worker = (base_seed + global_worker_id + current_epoch) % (2**32)
            rng = np.random.default_rng(seed=seed_for_worker)
            rng.shuffle(worker_indices)
            logger.debug(f"Worker {worker_id} Rank {rank}: Processing indices {start_sample_idx}-{end_sample_idx-1} (Shuffled with seed {seed_for_worker})")

            # Iterate through shuffled indices assigned to this worker
            for idx in worker_indices:
                # Calculate start/end indices for context and target
                start_ctx = idx                   # Start index of the context window
                end_ctx = idx + self.context_size # End index of context (exclusive)
                end_tgt = end_ctx + 1             # End index for target (exclusive)

                # Check bounds against the *effective* data size
                if end_tgt > self.data_size:
                    # This should ideally not happen if num_possible_samples is calculated correctly, but safety check.
                    logger.warning(f"Worker {worker_id} Rank {rank}: Calculated index {idx} leads to out-of-bounds access ({end_tgt} > {self.data_size}). Skipping.")
                    continue

                try:
                    # Slice data: context is bytes[idx : idx+ctx_size]
                    context_slice = bytes_data[start_ctx : end_ctx]
                    # Target is shifted: bytes[idx+1 : idx+ctx_size+1]
                    target_slice = bytes_data[start_ctx + 1 : end_tgt]

                    # --- Sanity Check Slice Lengths ---
                    if len(context_slice) != self.context_size or len(target_slice) != self.context_size:
                        logger.warning(f"Worker {worker_id} Rank {rank}: Slice length mismatch at index {idx}. Context: {len(context_slice)}, Target: {len(target_slice)}. Expected: {self.context_size}. Skipping.")
                        continue

                    # Convert numpy slices to PyTorch tensors (copy avoids mmap issues later)
                    # Use torch.long as expected by embedding layers
                    context_tensor = torch.tensor(context_slice.copy(), dtype=torch.long)
                    target_tensor = torch.tensor(target_slice.copy(), dtype=torch.long)

                    yield context_tensor, target_tensor

                except IndexError:
                    # Catch potential index errors during slicing (though bounds check should prevent this)
                    logger.warning(f"Worker {worker_id} Rank {rank}: Caught IndexError accessing index {idx}. Skipping.")
                    continue
                except Exception as e:
                    # Catch any other errors during processing
                    logger.error(f"Worker {worker_id} Rank {rank}: Error processing index {idx}: {e}")
                    continue

        except FileNotFoundError:
            logger.error(f"Worker {worker_id} Rank {rank}: Dataset file not found during iteration: {self.npy_file_path}")
        except Exception as e:
            # Catch errors during iterator setup (e.g., file loading)
            logger.error(f"Worker {worker_id} Rank {rank}: Iterator setup or main loop failed: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            # Ensure mmap handle is closed if it was opened
            if mmap_handle is not None:
                try:
                    mmap_handle.close()
                except Exception as close_err:
                    # Log warning but don't crash the worker
                    logger.warning(f"Worker {worker_id} Rank {rank}: Error closing mmap handle: {close_err}")
            # Explicitly delete reference to potentially large mmap array
            del bytes_data
            # Suggest garbage collection (might help release mmap resources)
            gc.collect()

    def set_seed(self, seed: int):
        """Sets the base seed for the dataset's shuffling."""
        self.seed = seed

    def set_epoch(self, epoch: int):
        """Sets the current epoch, affecting the shuffling sequence."""
        self.epoch = epoch

def seed_worker(worker_id: int, base_seed: int, rank_offset: int):
    """
    Sets random seeds for dataloader workers to ensure reproducibility.
    Called by DataLoader with worker_id.
    """
    # Create a unique seed for each worker based on base seed, rank, and worker ID
    worker_seed = base_seed + rank_offset + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # Note: Setting torch seed here might be redundant if all randomness
    # within __iter__ comes from numpy/random, but can be included for safety.
    # torch.manual_seed(worker_seed)
    # logger.debug(f"Worker {worker_id}: Seed set to {worker_seed}")


class Trainer:
    """Handles the training and validation loops, checkpointing, and logging."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10,
                 save_interval: int = 1000, checkpoint_dir: str = "checkpoints",
                 wandb_enabled: bool = False, max_grad_norm: float = 1.0,
                 rank: int = 0, world_size: int = 1, detect_anomaly: bool = False):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_accum_steps = max(1, grad_accum_steps)
        # Enable AMP only if CUDA is available and torch.amp exists
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch, "amp") and device.type == 'cuda'
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.log_interval = log_interval
        self.save_interval = max(1, save_interval) if save_interval > 0 else 0 # 0 disables intermediate saving
        self.checkpoint_dir = checkpoint_dir
        # Enable WandB only if available, requested, and on the main process
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE and is_main_process()
        self.max_grad_norm = max_grad_norm
        self.global_step = 0 # Total optimizer steps taken
        self.current_epoch = 0
        self.last_val_metrics: Optional[Dict[str, float]] = None # Store last validation results
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main_process()
        self.detect_anomaly = detect_anomaly # Enable/disable autograd anomaly detection

        # Create checkpoint directory if it doesn't exist (only on main process)
        if self.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Check if optimizer has gradient stats and Q-controller capabilities
        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(getattr(self.optimizer, 'gradient_stats', None), GradientStats)
        self.has_q_controller = hasattr(self.optimizer, 'q_controller') and isinstance(getattr(self.optimizer, 'q_controller', None), HAKMEMQController)

        # Store wandb run object if enabled
        self.wandb_run = wandb.run if self.wandb_enabled and wandb is not None else None

        logger.info(f"Trainer Rank {rank}: AMP={self.use_amp}, Accum={self.grad_accum_steps}, MaxNorm={self.max_grad_norm}, AnomalyDetect={self.detect_anomaly}, GradStats={self.has_grad_stats}, QCtrl={self.has_q_controller}")

    def _train_epoch(self):
        """Runs a single training epoch."""
        self.model.train() # Set model to training mode
        # Initialize epoch statistics
        total_loss_accum_cycle = 0.0 # Loss accumulated over grad_accum_steps
        optimizer_steps_in_epoch = 0 # Number of optimizer steps taken in this epoch
        micro_step_count_cycle = 0 # Steps within the current accumulation cycle
        total_micro_batches_processed_epoch = 0 # Total forward/backward passes processed
        approx_total_optim_steps = None # Estimate for progress bar
        total_micro_batches_estimate = None # Estimate for progress bar

        # --- Estimate total batches for tqdm progress bar ---
        try:
            dataset_len = 0
            # Use sampler length if available (more accurate with DDP/drop_last)
            if hasattr(self.train_loader.sampler, '__len__'):
                # DistributedSampler length already accounts for subset per rank
                dataset_len = len(self.train_loader.sampler)
            elif hasattr(self.train_loader.dataset, '__len__'):
                # Fallback for non-distributed or map-style datasets
                dset_total_len = len(self.train_loader.dataset)
                # Adjust for world size if DDP seems active but sampler has no len
                if self.world_size > 1:
                    dataset_len = dset_total_len // self.world_size
                else:
                    dataset_len = dset_total_len
            # Else: Iterable dataset without sampler length - cannot estimate total

            if dataset_len > 0:
                loader_batch_size = self.train_loader.batch_size or 1
                # Total micro-batches this rank will process (approximate if no drop_last)
                total_micro_batches_estimate = math.ceil(dataset_len / loader_batch_size)
                if total_micro_batches_estimate > 0 and self.grad_accum_steps > 0:
                    # Estimate total optimizer steps for this rank in the epoch
                    approx_total_optim_steps = max(1, total_micro_batches_estimate // self.grad_accum_steps)
                logger.debug(f"Rank {self.rank} Epoch Est: {total_micro_batches_estimate} micro-batches, {approx_total_optim_steps} optim steps.")
            else:
                logger.info("Cannot estimate epoch length for tqdm (IterableDataset without len or sampler len).")

        except Exception as e:
            logger.warning(f"Could not estimate epoch length: {e}")

        # Configure tqdm progress bar (only shown on main process)
        disable_tqdm = not self.is_main
        batch_iterator = tqdm(self.train_loader,
                                desc=f"Epoch {self.current_epoch + 1} | Opt Step 0/?",
                                disable=disable_tqdm, total=total_micro_batches_estimate,
                                unit="batch", dynamic_ncols=True, leave=False)

        # Zero gradients before starting the epoch loop
        self.optimizer.zero_grad(set_to_none=True)

        # --- Batch Loop ---
        for i, batch_data in enumerate(batch_iterator):
            total_micro_batches_processed_epoch += 1
            micro_step_count_cycle += 1
            # Determine if this is the last micro-step in the accumulation cycle
            should_optimizer_step = (micro_step_count_cycle % self.grad_accum_steps == 0)

            # Context manager for DDP gradient synchronization:
            # Disable sync during accumulation steps, enable only when optimizer should step.
            sync_context = contextlib.nullcontext()
            if self.world_size > 1 and isinstance(self.model, DistributedDataParallel):
                if not should_optimizer_step:
                    sync_context = self.model.no_sync()

            # Context manager for anomaly detection (only active if enabled)
            anomaly_context = torch.autograd.detect_anomaly(check_nan=True) if self.detect_anomaly else contextlib.nullcontext()

            # Forward and Backward Pass
            loss_value_scaled = None; loss = None; current_step_loss = 0.0
            try:
                with sync_context, anomaly_context:
                    # --- Data Handling ---
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        context, targets = batch_data
                    else:
                        logger.warning(f"Rank {self.rank}: Skip unexpected batch format {type(batch_data)} at index {i}. Expected (context, target).")
                        continue

                    context = context.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    batch_size, ctx_len = context.shape
                    if ctx_len == 0 or batch_size == 0:
                        logger.warning(f"Rank {self.rank}: Skip empty batch with shape {context.shape} at index {i}")
                        continue

                    # --- Forward Pass (with AMP if enabled) ---
                    with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                        # Model expects both input and target context for autoregressive prediction
                        logits = self.model(byte_seq=context, target_byte_seq=context)

                        if logits is None: # Should not happen if model implementation is correct
                            raise ValueError("Model forward pass returned None logits")
                        # Check logits stability *before* loss calculation
                        if not torch.isfinite(logits).all():
                            logger.warning(f"Rank {self.rank}: NaN/Inf logits detected pre-loss at step {self.global_step}, micro-step {micro_step_count_cycle}. Replacing.")
                            logits = torch.nan_to_num(logits, nan=0.0) # Replace with zeros

                        # Compute loss (use static method from model)
                        # Use label smoothing during training
                        loss = WuBuNestingSequenceModel.compute_loss(logits, targets, smoothing=0.1)
                        if loss is None or not torch.isfinite(loss):
                            # Loss function should handle internal NaNs and return finite value or high loss
                            raise ValueError(f"Loss computation returned non-finite/None value: {loss}")

                        # Scale loss for gradient accumulation
                        loss_value_scaled = loss / self.grad_accum_steps

                # --- Backward Pass ---
                # Use AMP scaler to scale the loss and perform backward pass
                # Handles gradient scaling automatically if AMP is enabled
                self.scaler.scale(loss_value_scaled).backward()

                # Store the *unscaled* loss for accumulation and logging
                current_step_loss = loss.item()
                if not np.isfinite(current_step_loss):
                    logger.warning(f"Rank {self.rank}: Non-finite loss ({current_step_loss}) computed in micro-step {micro_step_count_cycle}. Loss for this micro-batch will not be accumulated.")
                    current_step_loss = 0.0 # Treat as zero for accumulation to avoid NaN propagation

            except Exception as batch_ex:
                # Catch errors during forward/backward (e.g., OOM, anomaly detected)
                logger.error(f"Error during training micro-step {micro_step_count_cycle} (Batch {i}, Global {self.global_step}) Rank {self.rank}: {batch_ex}", exc_info=False)
                # Reset accumulation cycle state on error to prevent corrupted gradients
                total_loss_accum_cycle = 0.0
                micro_step_count_cycle = 0
                should_optimizer_step = False # Don't step optimizer if an error occurred in the cycle
                # Safely zero gradients that might have been computed partially
                try: self.optimizer.zero_grad(set_to_none=True)
                except Exception as zero_grad_err: logger.error(f"Error zeroing gradients after batch error: {zero_grad_err}")
                # Clear CUDA cache if it might help recovery (e.g., OOM)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                # Continue to the next micro-batch, skipping optimizer step for this cycle
                continue

            # Accumulate the unscaled loss for the cycle average
            total_loss_accum_cycle += current_step_loss

            # --- Optimizer Step (if accumulation cycle is complete) ---
            if should_optimizer_step:
                avg_loss_cycle = total_loss_accum_cycle / self.grad_accum_steps if self.grad_accum_steps > 0 else total_loss_accum_cycle
                optimizer_step_skipped = False
                unclipped_norm_val = 0.0
                has_nonfinite_grad = False
                is_clipped = False
                clip_ratio = None

                # --- 1. Unscale Gradients (Required before clipping/optimizer step with AMP) ---
                try:
                    # Modifies gradients in-place
                    self.scaler.unscale_(self.optimizer)
                except Exception as unscale_err:
                    logger.error(f"Error unscaling gradients at step {self.global_step} Rank {self.rank}: {unscale_err}. Skipping optimizer step.")
                    has_nonfinite_grad = True # Treat as non-finite grad scenario
                    unclipped_norm_val = float('inf')
                    optimizer_step_skipped = True

                # --- 2. Check Gradient Norm & Clip (if grads are finite) ---
                if not optimizer_step_skipped:
                    # Calculate total gradient norm of *unscaled* gradients
                    params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
                    total_norm_unclipped = torch.tensor(0.0, device=self.device)
                    if params_with_grad:
                        try:
                            # Calculate norm using float32 for stability
                            all_norms_sq = [torch.sum(p.grad.detach().float()**2) for p in params_with_grad]
                            finite_norms_sq = [n for n in all_norms_sq if torch.isfinite(n)]

                            if len(finite_norms_sq) < len(all_norms_sq):
                                # If any individual parameter's grad norm is non-finite
                                has_nonfinite_grad = True
                                total_norm_unclipped = torch.tensor(float('inf'), device=self.device)
                                num_non_finite = len(all_norms_sq) - len(finite_norms_sq)
                                logger.warning(f"Rank {self.rank}: Non-finite gradients detected in {num_non_finite} parameter(s) *before* clipping at step {self.global_step}.")
                            elif finite_norms_sq:
                                # Sum of squares of finite norms
                                total_norm_unclipped = torch.sqrt(torch.stack(finite_norms_sq).sum())
                            # Else: all grads might be zero, total_norm_unclipped remains 0.0
                        except Exception as norm_ex:
                            logger.error(f"Error calculating gradient norm at step {self.global_step} Rank {self.rank}: {norm_ex}", exc_info=False)
                            total_norm_unclipped = torch.tensor(float('inf'), device=self.device)
                            has_nonfinite_grad = True

                    unclipped_norm_val = total_norm_unclipped.item()

                    # Skip optimizer step if gradients are non-finite
                    if has_nonfinite_grad:
                        logger.warning(f"Rank {self.rank}: Skipping optimizer step {self.global_step} due to non-finite gradients (Norm: {unclipped_norm_val}).")
                        optimizer_step_skipped = True
                    else:
                        # Clip gradients if norm exceeds threshold
                        if self.max_grad_norm > 0 and unclipped_norm_val > self.max_grad_norm:
                            is_clipped = True
                            clip_ratio = self.max_grad_norm / (unclipped_norm_val + EPS) # Calculate ratio for logging
                            # Clip the gradients in-place using the norm calculated *before* potential clipping
                            torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm)

                        # Record gradient stats (uses unclipped norm) if optimizer supports it
                        if self.has_grad_stats:
                            # Pass the original (unclipped) norm and whether clipping was applied
                            self.optimizer.gradient_stats.record_gradient(unclipped_norm_val, is_clipped, clip_ratio)

                        # --- 3. Q-Controller Update & Action Selection (if enabled) ---
                        if self.has_q_controller:
                            try:
                                # Set current loss for Q-controller state calculation
                                self.optimizer.set_current_loss(avg_loss_cycle)
                                # Get current LR/Momentum (assuming one param group for simplicity)
                                group = self.optimizer.param_groups[0]
                                current_lr = group['lr']; current_mom = group.get('momentum', 0.0)

                                # Get the current state based on recent history
                                q_state = self.optimizer.q_controller.get_state(lr=current_lr, momentum=current_mom, grad_norm=unclipped_norm_val, loss=avg_loss_cycle)

                                # Update Q-table based on the *previous* step's state/action and the current reward/state
                                if self.optimizer.q_controller.prev_state is not None and self.optimizer.q_controller.prev_action is not None and q_state is not None:
                                    # Compute reward based on the transition from prev_loss to current loss
                                    reward = self.optimizer.q_controller.compute_reward(avg_loss_cycle, self.optimizer.q_controller.prev_loss, unclipped_norm_val)
                                    if np.isfinite(reward):
                                        self.optimizer.q_controller.update(self.optimizer.q_controller.prev_state, self.optimizer.q_controller.prev_action, reward, q_state)
                                    else:
                                        logger.warning(f"Q-Controller received non-finite reward ({reward}). Skipping Q-update.")

                                # Store current loss and state for the *next* Q-update calculation
                                self.optimizer.q_controller.prev_loss = avg_loss_cycle # Store the loss that led to the current state
                                if q_state is not None:
                                    self.optimizer.q_controller.prev_state = q_state # Store the state derived from this step
                                    # Choose the action for the *next* optimizer step based on the current state
                                    next_q_action = self.optimizer.q_controller.choose_action(q_state)
                                    # Store this action to be applied *before* the next optimizer.step() call
                                    self.optimizer.q_controller.prev_action = next_q_action
                                else: # If current state is invalid, reset previous state/action tracking
                                    self.optimizer.q_controller.prev_state = None
                                    self.optimizer.q_controller.prev_action = None
                            except Exception as q_err:
                                logger.error(f"Q-Controller update/action selection error at step {self.global_step} Rank {self.rank}: {q_err}", exc_info=False)
                                # Ensure Q-controller state doesn't prevent future steps
                                if self.has_q_controller and self.optimizer.q_controller: # Check existence again
                                     self.optimizer.q_controller.prev_state = None; self.optimizer.q_controller.prev_action = None

                        # --- 4. Optimizer Step ---
                        # Pass gradients (already unscaled, potentially clipped) to the optimizer
                        # scaler.step() also checks for inf/NaN gradients scaled by the scaler
                        # If NaNs/Infs were introduced *after* unscaling (e.g., by clipping?), step might be skipped by scaler.
                        self.scaler.step(self.optimizer)

                # --- 5. Update AMP Scaler ---
                # Update the scale factor for the next iteration based on whether inf/NaNs were encountered
                self.scaler.update()

                # --- 6. Zero Gradients ---
                # Crucial to zero gradients *after* optimizer step and scaler update
                self.optimizer.zero_grad(set_to_none=True)

                # --- 7. Logging and Checkpointing ---
                # Record step statistics if optimizer supports it
                grad_stats = {}
                if self.has_grad_stats:
                    grad_stats = self.optimizer.gradient_stats.record_step(self.global_step, skipped=optimizer_step_skipped)

                if not optimizer_step_skipped:
                    optimizer_steps_in_epoch += 1
                    self.global_step += 1 # Increment global optimizer step count

                    # Update progress bar description and postfix (only on main process)
                    if self.is_main:
                        optim_step_str = f"{optimizer_steps_in_epoch}/{(approx_total_optim_steps or '?')}"
                        batch_iterator.set_description(f"Epoch {self.current_epoch + 1} | Opt Step {optim_step_str}")
                        current_lr = self.optimizer.param_groups[0]['lr']
                        # Use the norm value that was actually used for update (clipped if needed)
                        logged_norm = min(unclipped_norm_val, self.max_grad_norm) if self.max_grad_norm > 0 and is_clipped else unclipped_norm_val
                        batch_iterator.set_postfix(Loss=f"{avg_loss_cycle:.3f}", LR=f"{current_lr:.3e}", Grad=f"{logged_norm:.2f}", Scale=f"{self.scaler.get_scale():.0f}", refresh=False)

                    # Log detailed metrics at specified interval (only on main process)
                    if self.is_main and self.global_step % self.log_interval == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        current_mom = self.optimizer.param_groups[0].get('momentum', 0.0)
                        q_info = self.optimizer.get_q_info() if self.has_q_controller else {}
                        # Log the norm that was effectively applied (clipped or unclipped)
                        logged_norm = min(unclipped_norm_val, self.max_grad_norm) if self.max_grad_norm > 0 and is_clipped else unclipped_norm_val

                        log_data = {
                            "Epoch": self.current_epoch + 1, "Step": self.global_step,
                            "Train Loss (Cycle Avg)": avg_loss_cycle,
                            "Learning Rate": current_lr, "Momentum": current_mom,
                            "Grad Norm (Applied)": logged_norm, # Norm after clipping, if any
                            "Grad Norm (Unclipped Max)": grad_stats.get('max_gradient', 0.0), # Max unclipped norm encountered in step
                            "Clip %": grad_stats.get('clip_percentage', 0.0),
                            "Avg Clip Ratio": grad_stats.get('clip_ratio_avg', 0.0),
                            "NonFinite Grads Encountered": grad_stats.get('non_finite_grads', 0), # Num non-finite grads *before* clip/step attempt
                            "AMP Scale": self.scaler.get_scale() if self.use_amp else 1.0
                        }
                        # Add Q-controller info, handling potential None action
                        log_data.update({f"Q_{k}": v for k, v in q_info.items() if k != 'last_action'})
                        last_q_action = q_info.get('last_action')
                        if last_q_action:
                            log_data["Q_LR_Scale"] = last_q_action.get('lr_scale', 1.0)
                            log_data["Q_Mom_Scale"] = last_q_action.get('momentum_scale', 1.0)

                        # Format log message
                        log_msg_parts = [
                            f"Step {self.global_step}", f"Ep {self.current_epoch + 1} Opt {optimizer_steps_in_epoch}",
                            f"Loss(avg): {log_data['Train Loss (Cycle Avg)']:.4f}", f"LR: {log_data['Learning Rate']:.3e}", f"Mom: {log_data['Momentum']:.3f}",
                            f"GradNorm(App): {log_data['Grad Norm (Applied)']:.2f}", f"MaxUnclip: {log_data['Grad Norm (Unclipped Max)']:.2f}",
                            f"Clip%: {log_data['Clip %']:.1f}%", f"NFGrads: {log_data['NonFinite Grads Encountered']}", f"Scale: {log_data['AMP Scale']:.0f}"
                        ]
                        if self.has_q_controller:
                            log_msg_parts.extend([
                                f"QScale(LR/M): {log_data.get('Q_LR_Scale', 1.0):.2f}/{log_data.get('Q_Mom_Scale', 1.0):.2f}",
                                f"QEps: {log_data.get('Q_epsilon', 0):.3f}"
                            ])
                        log_msg = " | ".join(log_msg_parts)
                        logger.info(log_msg)

                        # Log to WandB if enabled
                        if self.wandb_run:
                            try:
                                wandb.log({"train": log_data}, step=self.global_step)
                            except Exception as wb_err:
                                logger.error(f"WandB logging failed at step {self.global_step}: {wb_err}")

                    # Save intermediate checkpoint if interval is met (only on main process)
                    if self.is_main and self.save_interval > 0 and self.global_step % self.save_interval == 0:
                        self._save_checkpoint(is_intermediate=True, metrics={'train_loss_cycle': avg_loss_cycle})

                # --- End of Optimizer Step ---
                # Reset accumulation cycle state
                total_loss_accum_cycle = 0.0
                micro_step_count_cycle = 0

        # --- End of Epoch ---
        if self.is_main and hasattr(batch_iterator, 'close'):
            batch_iterator.close() # Close tqdm progress bar

        # Calculate approximate average epoch loss (based on cycle averages)
        # Note: This is an approximation as cycles might have different numbers of micro-batches
        avg_epoch_loss = 0.0 # Will be calculated more accurately via reduction if DDP active

        # Synchronize at the end of the epoch if using DDP
        if self.world_size > 1:
            logger.debug(f"Rank {self.rank} entering end-of-epoch barrier.")
            torch.distributed.barrier()
            logger.debug(f"Rank {self.rank} exited end-of-epoch barrier.")
            # Average metrics across ranks if needed (e.g., epoch loss, although _validate handles its own sync)
            # Example: Gather average cycle losses and compute overall mean if needed

        return avg_epoch_loss # Return value currently unused, validation provides main metrics

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Runs validation loop and returns metrics."""
        if self.val_loader is None:
            logger.info("Validation loader not provided. Skipping validation.")
            return {}

        self.model.eval() # Set model to evaluation mode
        approx_total_val_batches = None

        # Estimate total validation batches for tqdm
        try:
            val_ds_len = 0
            if hasattr(self.val_loader.sampler, '__len__'):
                val_ds_len = len(self.val_loader.sampler)
            elif hasattr(self.val_loader.dataset, '__len__'):
                dset_total_len = len(self.val_loader.dataset)
                if self.world_size > 1: val_ds_len = dset_total_len // self.world_size
                else: val_ds_len = dset_total_len

            if val_ds_len > 0:
                val_bs = self.val_loader.batch_size or 1
                # Use ceil because we don't drop last in validation
                approx_total_val_batches = math.ceil(val_ds_len / val_bs)
            else:
                logger.info("Cannot estimate validation length for tqdm.")
        except Exception as e:
            logger.warning(f"Could not estimate validation length: {e}")

        # Configure validation progress bar
        val_iterator = tqdm(self.val_loader,
                            desc=f"Val Ep {self.current_epoch + 1}",
                            disable=not self.is_main, # Only show on main process
                            total=approx_total_val_batches, unit="batch", leave=False)
        batch_losses = [] # Store losses from batches processed *by this rank*

        # --- Validation Batch Loop ---
        for batch_data in val_iterator:
            try:
                # --- Data Handling ---
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    context, targets = batch_data
                else:
                    logger.warning(f"Rank {self.rank}: Skip unexpected val batch format: {type(batch_data)}")
                    continue

                context = context.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_size, ctx_len = context.shape
                if batch_size == 0 or ctx_len == 0:
                    logger.debug(f"Rank {self.rank}: Skipping empty validation batch.")
                    continue

                # --- Forward Pass (No AMP needed typically, but use context manager for consistency if enabled) ---
                with torch.no_grad(), amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # Get the underlying model if wrapped in DDP
                    model_to_eval = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

                    logits = model_to_eval(byte_seq=context, target_byte_seq=context)
                    # Get vocab size from model if possible, default to 256
                    vocab_size = getattr(model_to_eval, 'vocab_size', 256)

                    # Basic sanity check on logits shape
                    if logits.shape[0] != batch_size or logits.shape[1] != ctx_len or logits.shape[2] != vocab_size:
                        logger.warning(f"Rank {self.rank}: Validation logits shape mismatch. Got {logits.shape}, expect B={batch_size}, L={ctx_len}, V={vocab_size}. Skipping batch loss.")
                        continue
                    if not torch.isfinite(logits).all():
                        logger.warning(f"Rank {self.rank}: NaN/Inf found in validation logits. Skipping batch loss.")
                        continue

                    # Compute loss without label smoothing for validation
                    loss = WuBuNestingSequenceModel.compute_loss(logits, targets, smoothing=0.0)

                loss_item = loss.item()
                if np.isfinite(loss_item):
                    batch_losses.append(loss_item)
                else:
                    # Log non-finite loss but don't append it
                    logger.warning(f"Rank {self.rank}: Non-finite validation loss calculated: {loss_item}")

            except Exception as val_ex:
                logger.error(f"Rank {self.rank} Error during validation batch: {val_ex}", exc_info=False)
                # Continue to next batch on error
                continue

        # --- Aggregate Losses Across Ranks ---
        all_losses = []
        if self.world_size > 1:
            # Use Float64 for potentially more precise aggregation
            losses_tensor = torch.tensor(batch_losses, dtype=torch.float64, device=self.device)
            # Create tensors to receive data from all ranks
            gathered_losses_list = [torch.zeros_like(losses_tensor) for _ in range(self.world_size)]
            try:
                # Gather tensors from all ranks into the list
                torch.distributed.all_gather(gathered_losses_list, losses_tensor)
                # Concatenate gathered tensors and convert to a flat list of floats
                all_losses = torch.cat(gathered_losses_list).cpu().tolist()
                logger.debug(f"Rank {self.rank}: Gathered {len(all_losses)} val losses from {self.world_size} ranks.")
            except Exception as gather_err:
                logger.error(f"Rank {self.rank}: Validation loss gather failed: {gather_err}. Using local losses only.")
                all_losses = batch_losses # Fallback to local losses on error
        else:
            # No DDP, just use the losses from this single process
            all_losses = batch_losses

        # --- Calculate and Log Metrics (Main Process Only) ---
        metrics = {}
        if self.is_main:
            if not all_losses:
                logger.warning("No valid validation losses collected across ranks. Cannot compute metrics.")
                avg_loss = float('inf')
                perplexity = float('inf')
            else:
                avg_loss = sum(all_losses) / len(all_losses)
                perplexity = float('inf')
                # Calculate perplexity safely
                if np.isfinite(avg_loss):
                    try:
                        # Clamp avg_loss to prevent overflow in math.exp
                        perplexity = math.exp(min(avg_loss, 700))
                    except OverflowError:
                        logger.warning(f"Perplexity calculation overflowed (Avg Loss: {avg_loss}). Setting PPL to Inf.")
                        perplexity = float('inf')
                    except ValueError: # Handle potential domain errors if avg_loss is negative?
                        logger.warning(f"Perplexity calculation ValueError (Avg Loss: {avg_loss}). Setting PPL to Inf.")
                        perplexity = float('inf')
                else: # If avg_loss itself is inf/nan
                    perplexity = float('inf')

            metrics = {'val_loss': avg_loss, 'val_perplexity': perplexity}
            self.last_val_metrics = metrics # Store metrics for potential checkpoint saving
            logger.info(f"Validation Epoch {self.current_epoch + 1} | Avg Loss: {metrics['val_loss']:.4f} | Perplexity: {metrics['val_perplexity']:.2f}")

            # Log validation metrics to WandB
            if self.wandb_enabled and self.wandb_run:
                try:
                    # Log validation metrics, associating with the current global step
                    wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": self.current_epoch + 1}, step=self.global_step)
                except Exception as wb_err:
                    logger.error(f"WandB validation logging failed: {wb_err}")

        if hasattr(val_iterator, 'close'):
            val_iterator.close() # Ensure tqdm iterator is closed

        return metrics

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict] = None):
        """Saves model checkpoint (only on main process)."""
        if not self.is_main: return # Only rank 0 saves checkpoints

        # --- Determine Filename ---
        state_indicator = f"step_{self.global_step}" if is_intermediate else f"epoch_{self.current_epoch+1}_final"
        current_metrics = metrics if metrics is not None else self.last_val_metrics
        metric_str = ""

        # Add primary metric (val_loss or train_loss) to filename for easier identification
        if current_metrics:
            val_loss = current_metrics.get('val_loss')
            train_loss = current_metrics.get('train_loss_cycle') # From intermediate save
            train_loss_epoch = current_metrics.get('loss') # From final save (if no val)

            metric_key = None; metric_val = None
            # Prioritize validation loss if available and finite
            if val_loss is not None and np.isfinite(val_loss):
                metric_key = 'vloss'
                metric_val = val_loss
            # Use intermediate train loss if available and finite
            elif train_loss is not None and np.isfinite(train_loss):
                metric_key = 'tloss'
                metric_val = train_loss
            # Use final epoch train loss if available and finite
            elif train_loss_epoch is not None and np.isfinite(train_loss_epoch):
                metric_key = 'tloss'
                metric_val = train_loss_epoch

            # Format metric value nicely for filename
            if metric_key and metric_val is not None:
                # Use scientific notation for very small losses, fixed precision otherwise
                mf = f"{metric_val:.2e}" if abs(metric_val) < 1e-3 and metric_val != 0.0 else f"{metric_val:.3f}"
                metric_str = f"_{metric_key}{mf}"

        state_indicator += metric_str
        filename = f"checkpoint_{state_indicator}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        # --- Prepare Checkpoint Data ---
        # Get state dict from the underlying model if DDP is used
        model_state = self.model.module.state_dict() if isinstance(self.model, DistributedDataParallel) else self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch, # Epoch just completed
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'metrics': current_metrics, # Save metrics associated with this checkpoint
            'amp_enabled': self.use_amp,
            'args': getattr(self, 'args', None) # Save args used for this run if available
        }

        # Save Q-controller state if available and enabled
        if self.has_q_controller and self.optimizer.q_controller:
            try:
                # Serialize Q-table keys (tuples) to strings for broader compatibility?
                # PyTorch save should handle tuples, but string conversion might be safer if loading elsewhere.
                # Let's try saving tuples directly first.
                q_state_data = {
                    'q_table': self.optimizer.q_controller.q_table, # {str(k): v for k, v in self.optimizer.q_controller.q_table.items()},
                    'epsilon': self.optimizer.q_controller.epsilon,
                    'access_count': self.optimizer.q_controller.q_table_access_count, # {str(k): v for k, v in ...}
                    'creation_time': self.optimizer.q_controller.q_table_creation_time, # {str(k): v for k, v in ...}
                    'loss_window': list(self.optimizer.q_controller.loss_window),
                    'grad_norm_window': list(self.optimizer.q_controller.grad_norm_window),
                    'performance_window': list(self.optimizer.q_controller.performance_window),
                    'stable_steps': self.optimizer.q_controller.stable_steps,
                    'oscillation_counter': self.optimizer.q_controller.oscillation_counter,
                    'prev_loss': self.optimizer.q_controller.prev_loss,
                    'prev_state': self.optimizer.q_controller.prev_state, # May be None
                    'prev_action': self.optimizer.q_controller.prev_action # May be None
                }
                checkpoint['q_controller_state'] = q_state_data
            except Exception as q_save_err:
                logger.error(f"Error preparing Q-Controller state for saving: {q_save_err}")

        # --- Save Atomically ---
        # Save to a temporary file first, then atomically replace/rename
        temp_filepath = filepath + ".tmp"
        try:
            torch.save(checkpoint, temp_filepath)
            os.replace(temp_filepath, filepath) # Atomic rename/replace (safer)
            logger.info(f"Checkpoint saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}", exc_info=True)
            # Clean up temp file if saving failed
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError as rm_err: logger.error(f"Failed to remove temporary checkpoint file {temp_filepath}: {rm_err}")

    def load_checkpoint(self, filepath: str) -> int:
        """Loads state from a checkpoint file. Returns the epoch to start from."""
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return 0 # Start from epoch 0 if checkpoint not found

        try:
            # Load checkpoint onto CPU first to avoid device mismatches
            checkpoint = torch.load(filepath, map_location='cpu')
            logger.info(f"Loading checkpoint from: {filepath}")

            # --- Load Model State ---
            # Get the underlying model, handling potential DDP wrapping
            model_to_load = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
            # Use strict=False to allow flexibility (e.g., architectural changes)
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # Report missing/unexpected keys
            if incompatible_keys.missing_keys:
                logger.warning(f"Missing keys when loading model state: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                logger.warning(f"Unexpected keys when loading model state: {incompatible_keys.unexpected_keys}")

            # --- Load Optimizer State ---
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Move optimizer state tensors (like momentum buffers) to the correct device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                try:
                                    state[k] = v.to(self.device)
                                except Exception as e_state:
                                    logger.warning(f"Failed moving optimizer state tensor '{k}' to {self.device}: {e_state}")
                    logger.info("Optimizer state loaded successfully.")
                except Exception as optim_ex:
                    logger.warning(f"Failed loading optimizer state: {optim_ex}. Optimizer state might be reset.", exc_info=True)
            else:
                logger.warning("Optimizer state dict not found in checkpoint. Optimizer starts from scratch.")

            # --- Load AMP Scaler State ---
            saved_amp_enabled = checkpoint.get('amp_enabled', False)
            if self.use_amp: # If AMP is currently enabled
                if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and saved_amp_enabled:
                    try:
                        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        logger.info("AMP scaler state loaded successfully.")
                    except Exception as scaler_ex:
                        logger.warning(f"Failed loading AMP scaler state: {scaler_ex}. Scaler state might be reset.")
                elif saved_amp_enabled: # Checkpoint used AMP, but state missing
                    logger.warning("Checkpoint indicates AMP was used, but scaler state is missing. Scaler state reset.")
                # else: # Checkpoint didn't use AMP, current run does -> Scaler starts fresh (default state)
            elif saved_amp_enabled: # AMP was used in saved checkpoint, but not enabled now
                logger.warning("Checkpoint was saved with AMP enabled, but AMP is currently disabled.")

            # --- Load Training Progress ---
            # Resume from the epoch *after* the one saved
            start_epoch = checkpoint.get('epoch', -1) + 1
            self.global_step = checkpoint.get('global_step', 0)
            # Set current epoch to the one that *completed* for logging/saving purposes
            self.current_epoch = start_epoch - 1 if start_epoch > 0 else 0
            # Restore last known metrics
            self.last_val_metrics = checkpoint.get('metrics')
            if self.last_val_metrics:
                logger.info(f"Restored validation metrics from checkpoint: {self.last_val_metrics}")

            # --- Load Q-Controller State ---
            if self.has_q_controller and self.optimizer.q_controller and 'q_controller_state' in checkpoint:
                q_state = checkpoint['q_controller_state']
                logger.info("Attempting to load Q-Controller state...")
                try:
                    # Restore Q-table (keys should be tuples)
                    # If keys were saved as strings, need conversion: eval(k) or similar safe eval
                    self.optimizer.q_controller.q_table = q_state.get('q_table', {})
                    self.optimizer.q_controller.epsilon = q_state.get('epsilon', self.optimizer.q_controller.epsilon)

                    # Restore access counts and creation times (keys should be tuples)
                    self.optimizer.q_controller.q_table_access_count = defaultdict(int, q_state.get('access_count', {}))
                    self.optimizer.q_controller.q_table_creation_time = q_state.get('creation_time', {})

                    # Restore deques (ensure maxlen is preserved)
                    maxlen_loss = self.optimizer.q_controller.loss_window.maxlen
                    self.optimizer.q_controller.loss_window = deque(q_state.get('loss_window', []), maxlen=maxlen_loss)
                    maxlen_grad = self.optimizer.q_controller.grad_norm_window.maxlen
                    self.optimizer.q_controller.grad_norm_window = deque(q_state.get('grad_norm_window', []), maxlen=maxlen_grad)
                    maxlen_perf = self.optimizer.q_controller.performance_window.maxlen
                    self.optimizer.q_controller.performance_window = deque(q_state.get('performance_window', []), maxlen=maxlen_perf)

                    # Restore counters and previous step info
                    self.optimizer.q_controller.stable_steps = q_state.get('stable_steps', 0)
                    self.optimizer.q_controller.oscillation_counter = q_state.get('oscillation_counter', 0)
                    self.optimizer.q_controller.prev_loss = q_state.get('prev_loss')
                    self.optimizer.q_controller.prev_state = q_state.get('prev_state')
                    self.optimizer.q_controller.prev_action = q_state.get('prev_action')
                    logger.info("Q-Controller state loaded successfully from checkpoint.")
                except Exception as q_load_err:
                    logger.warning(f"Failed loading Q-Controller state from checkpoint: {q_load_err}. Q-Controller state may be reset.", exc_info=True)
                    # Reset Q-controller state partially or fully if loading fails
                    # self.optimizer.q_controller = HAKMEMQController(...) # Or reset specific parts
            elif self.has_q_controller:
                logger.warning("Q-Controller is active, but its state was not found in the checkpoint.")

            # --- Argument Comparison ---
            if 'args' in checkpoint and checkpoint['args'] is not None and hasattr(self, 'args') and self.args is not None:
                loaded_args_dict = vars(checkpoint['args'])
                current_args_dict = vars(self.args)
                mismatched_args = {}
                all_keys = set(loaded_args_dict.keys()) | set(current_args_dict.keys())
                # Keys to ignore during comparison (often change or specific to run instance)
                ignore_keys = {'resume', 'local_rank', 'rank', 'world_size', 'device', # Runtime specifics
                               'wandb', 'wandb_project', 'wandb_entity', # Logging specifics
                               'data_path', 'val_data_path', 'checkpoint_dir'} # Path specifics
                for key in all_keys:
                    if key in ignore_keys: continue
                    loaded_val = loaded_args_dict.get(key, '<<Missing_Loaded>>')
                    current_val = current_args_dict.get(key, '<<Missing_Current>>')
                    # Simple string comparison for basic check (might miss type differences)
                    if str(loaded_val) != str(current_val):
                        mismatched_args[key] = {'loaded': loaded_val, 'current': current_val}
                if mismatched_args:
                    logger.warning(f"Argument mismatch detected between checkpoint and current run:\n{mismatched_args}")
                else:
                    logger.info("Checkpoint arguments match current arguments (excluding ignored keys).")
            elif hasattr(self, 'args') and self.args is not None:
                logger.warning("Checkpoint did not contain saved arguments for comparison.")

            # --- Final Steps ---
            # Move model to the correct device *after* loading state dict
            model_to_load.to(self.device)
            logger.info(f"Checkpoint '{os.path.basename(filepath)}' loaded. Resuming from Epoch {start_epoch} (Global Step {self.global_step})")
            return start_epoch # Return the epoch number to START training from

        except FileNotFoundError:
            logger.error(f"Load checkpoint failed: File not found at '{filepath}'")
            return 0
        except Exception as e:
            logger.error(f"Failed to load checkpoint from '{filepath}': {e}", exc_info=True)
            return 0 # Start from scratch on load failure

    def train(self, epochs: int, start_epoch: int = 0):
        """Main training loop over multiple epochs."""
        self.current_epoch = start_epoch
        if self.is_main:
            logger.info(f"Starting training from Epoch {start_epoch + 1}/{epochs} (Global Step: {self.global_step}).")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            if self.is_main:
                logger.info(f"--- Starting Epoch {epoch + 1}/{epochs} ---")

            # --- Set Epoch for Samplers and Datasets ---
            # Essential for DistributedSampler shuffling and IterableDataset state/shuffling
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if self.val_loader and isinstance(self.val_loader.sampler, DistributedSampler):
                self.val_loader.sampler.set_epoch(epoch)
            # Also set epoch for the dataset itself if it supports it (for internal shuffling seed)
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader.dataset, 'set_epoch'):
                self.val_loader.dataset.set_epoch(epoch)

            # --- Train one epoch ---
            self._train_epoch()
            # Note: _train_epoch now returns approximate loss, but validation is the main metric source.

            # --- Validate (if validation loader exists) ---
            val_metrics = self._validate() # Validation aggregates and logs metrics on main process

            # --- Save checkpoint at the end of the epoch (main process only) ---
            if self.is_main:
                # Use validation metrics if available, otherwise use placeholder/train loss if needed
                save_metrics = val_metrics if val_metrics else None
                # Always save at epoch end unless save_interval was set to a negative value (e.g. -1) explicitly
                # If save_interval == 0, intermediate saves are off, but epoch end save still happens.
                if self.save_interval >= 0:
                    self._save_checkpoint(is_intermediate=False, metrics=save_metrics)

            # Barrier to ensure all ranks finish the epoch (including validation and saving)
            # before starting the next epoch's data loading and training.
            if self.world_size > 1:
                logger.debug(f"Rank {self.rank} entering end-of-epoch {epoch+1} barrier.")
                torch.distributed.barrier()
                logger.debug(f"Rank {self.rank} exited end-of-epoch {epoch+1} barrier.")

        if self.is_main:
            logger.info(f"Training finished after {epochs} epochs.")


# =====================================================================
# Default Configuration
# =====================================================================
DEFAULT_CONFIG_WUBU = {
    # Structure
    "num_levels": 3,
    "hyperbolic_dims": [128, 64, 32],
    "boundary_points_per_level": [5, 5, 5],
    # Geometry Parameters (Initial Values & Learnability)
    "initial_curvatures": [1.0, 1.0, 1.0], # Start with moderate curvature
    "initial_scales": [1.0, 1.0, 1.0],
    "initial_spread_values": None, # Default: use initial_scales if None
    "learnable_curvature": True,
    "learnable_scales": True,
    "learnable_spread": True,
    "curvature_min_value": 1e-5, # Minimum allowed value for safety
    "scale_min_value": 1e-5,
    "spread_min_value": 1e-5,
    # Level Features
    "use_level_descriptors": True,
    "level_descriptor_init_scale": 0.01,
    "use_level_spread": True,
    # Inter-Level Transforms
    "rotation_types": ["so_n", "so_n"], # Default: SO(n) rotation
    "transform_types": ["mlp", "mlp"],   # Default: MLP transformation
    "transform_hidden_dims": [None, None], # Default hidden dims calculated in InterLevelTransform
    # Intra-Level Processing
    "use_tangent_flow": True,
    "tangent_flow_type": "mlp",
    "tangent_flow_hidden_dim_ratio": 0.5,
    "tangent_flow_scale": 1.0, # How much flow displacement to add
    "relative_vector_aggregation": "mean", # How boundary info influences next level (mean, sum, none)
    "tangent_input_combination_dims": [64], # Hidden layer sizes for the MLP combining level inputs
    # Final Aggregation
    "aggregation_method": "concat_tangent", # How outputs from all levels are combined
    # General
    "dropout": 0.1,
}


# =====================================================================
# Argument Parsing
# =====================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu Nesting Sequence Model Trainer (v0.02 - RelVecs Implemented)")

    # --- Group: Data and Checkpointing ---
    grp_data = parser.add_argument_group('Data and Checkpointing')
    grp_data.add_argument('--data_path', type=str, required=True, help='Path to training data (.npy file)')
    grp_data.add_argument('--val_data_path', type=str, default=None, help='Path to validation data (.npy file, optional)')
    grp_data.add_argument('--checkpoint_dir', type=str, default='wubu_checkpoints_v0.02', help='Directory to save checkpoints')
    grp_data.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from')
    grp_data.add_argument('--save_interval', type=int, default=1000, help='Save checkpoint every N optimizer steps. 0 saves only at epoch end. <0 disables epoch end save too.')
    grp_data.add_argument('--log_interval', type=int, default=50, help='Log metrics every N optimizer steps')
    grp_data.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of training data to use (0.0 < f <= 1.0)')
    grp_data.add_argument("--context_window", type=int, default=256, help="Sequence length for training context.")

    # --- Group: WuBu Nesting Architecture ---
    grp_wubu = parser.add_argument_group('WuBu Nesting Architecture')
    grp_wubu.add_argument('--num_levels', type=int, default=DEFAULT_CONFIG_WUBU['num_levels'], help='Number of nesting levels')
    grp_wubu.add_argument('--hyperbolic_dims', nargs='+', type=int, default=DEFAULT_CONFIG_WUBU['hyperbolic_dims'], help='Dimension of hyperbolic space at each level (list of ints)')
    grp_wubu.add_argument('--initial_curvatures', nargs='+', type=float, default=DEFAULT_CONFIG_WUBU['initial_curvatures'], help='Initial curvature for each level (list of floats)')
    grp_wubu.add_argument('--initial_scales', nargs='+', type=float, default=DEFAULT_CONFIG_WUBU['initial_scales'], help='Initial scale param for each level (list of floats)')
    grp_wubu.add_argument('--initial_spread_values', nargs='+', type=float, default=None, help='Initial spread param for each level (list of floats, defaults to initial_scales if None)') # Allow defaulting
    grp_wubu.add_argument('--boundary_points_per_level', nargs='+', type=int, default=DEFAULT_CONFIG_WUBU['boundary_points_per_level'], help='Number of boundary points per level (list of ints)')
    grp_wubu.add_argument('--rotation_types', nargs='*', type=str, default=DEFAULT_CONFIG_WUBU['rotation_types'], help='Rotation types for inter-level transform (so_n/quat/identity). Need num_levels-1 values.')
    grp_wubu.add_argument('--transform_types', nargs='*', type=str, default=DEFAULT_CONFIG_WUBU['transform_types'], help='Non-rotational map types for inter-level transform (mlp/linear/quat). Need num_levels-1 values.')
    grp_wubu.add_argument('--transform_hidden_dims', nargs='*', type=int, default=None, help='Hidden dims for MLP transforms (0 or None=default). Need num_levels-1 values.') # Allow defaulting
    grp_wubu.add_argument('--aggregation_method', type=str, default=DEFAULT_CONFIG_WUBU['aggregation_method'], choices=['concat_tangent'], help='Method to aggregate outputs from all levels')
    grp_wubu.add_argument('--relative_vector_aggregation', type=str, default=DEFAULT_CONFIG_WUBU['relative_vector_aggregation'], choices=['mean', 'sum', 'none'], help='How to aggregate relative boundary vectors for level input')
    grp_wubu.add_argument('--tangent_flow_type', type=str, default=DEFAULT_CONFIG_WUBU['tangent_flow_type'], choices=['mlp', 'linear', 'none'], help='Type of optional flow map in tangent space')
    grp_wubu.add_argument('--tangent_flow_scale', type=float, default=DEFAULT_CONFIG_WUBU['tangent_flow_scale'], help='Scaling factor for tangent flow displacement')
    # Boolean flags for enabling/disabling features (use store_true for disabling)
    grp_wubu.add_argument('--no_learnable_curvature', action='store_true', help='Fix curvature values (use initial values)')
    grp_wubu.add_argument('--no_learnable_scales', action='store_true', help='Fix scale parameters (use initial values)')
    grp_wubu.add_argument('--no_learnable_spread', action='store_true', help='Fix spread parameters (use initial values)')
    grp_wubu.add_argument('--no_level_descriptors', action='store_true', help='Disable learnable level descriptors')
    grp_wubu.add_argument('--no_level_spread', action='store_true', help='Disable learnable level spread parameters')
    grp_wubu.add_argument('--no_tangent_flow', action='store_true', help='Disable tangent space flow')

    # --- Group: Sequence Model Specifics (Encoder/Decoder) ---
    grp_seq = parser.add_argument_group('Sequence Model Specifics (Encoder/Decoder)')
    grp_seq.add_argument('--local_hidden_size', type=int, default=256, help='Hidden size for Local Encoder and Decoder Transformer layers')
    grp_seq.add_argument('--decoder_memory_dim', type=int, default=512, help='Dimension of the memory vector fed to the Local Decoder (output of WuBu aggregation projection)')
    grp_seq.add_argument('--n_gram_sizes', nargs='+', type=int, default=[3, 4], help='N-gram sizes for features in Local Encoder (list of ints)')
    grp_seq.add_argument('--n_gram_vocab_size', type=int, default=30000, help='Vocabulary size for N-gram hashing')
    grp_seq.add_argument('--no_hierarchical_decoder', action='store_true', help='Use a standard flat prediction head in Local Decoder instead of hierarchical')
    # Add args for encoder/decoder layers/heads if needed
    grp_seq.add_argument('--num_encoder_layers', type=int, default=2, help='Number of layers in Local Encoder Transformer')
    grp_seq.add_argument('--num_decoder_layers', type=int, default=4, help='Number of layers in Local Decoder Transformer')
    grp_seq.add_argument('--num_encoder_heads', type=int, default=8, help='Number of heads in Local Encoder Transformer')
    grp_seq.add_argument('--num_decoder_heads', type=int, default=8, help='Number of heads in Local Decoder Transformer')


    # --- Group: Training Parameters ---
    grp_train = parser.add_argument_group('Training Parameters')
    grp_train.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    grp_train.add_argument('--batch_size', type=int, default=32, help='Global batch size (divided across all GPUs)')
    grp_train.add_argument('--grad_accum_steps', type=int, default=2, help='Number of steps to accumulate gradients before optimizer step')
    grp_train.add_argument('--learning_rate', type=float, default=5e-4, help='Base learning rate for the optimizer')
    grp_train.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay coefficient (L2 penalty)')
    grp_train.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping (0 to disable)')
    grp_train.add_argument('--dropout', type=float, default=DEFAULT_CONFIG_WUBU['dropout'], help='General dropout rate (used in WuBu levels, Encoder, Decoder)')
    grp_train.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    grp_train.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers (0 uses main process)')
    grp_train.add_argument('--no_amp', action='store_true', help='Disable Automatic Mixed Precision (AMP)')
    grp_train.add_argument('--detect_anomaly', action='store_true', help='Enable PyTorch autograd anomaly detection (for debugging, slows down training)')

    # --- Group: Distributed Training ---
    grp_dist = parser.add_argument_group('Distributed Training')
    # LOCAL_RANK is typically set by the launch utility (torchrun, torch.distributed.launch)
    # Defaulting to -1 signifies non-distributed mode if the variable isn't set.
    grp_dist.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), help="Local rank for DDP (set by launch utility, do not set manually)")

    # --- Group: Q-Learning Optimizer Control ---
    grp_q = parser.add_argument_group('Q-Learning Optimizer Control')
    grp_q.add_argument('--q_learning_rate', type=float, default=0.1, help='Q-learning learning rate (alpha) for Q-table updates')
    grp_q.add_argument('--q_discount', type=float, default=0.9, help='Q-learning discount factor (gamma) for future rewards')
    grp_q.add_argument('--q_epsilon', type=float, default=0.2, help='Q-learning initial exploration rate (epsilon)')
    grp_q.add_argument('--q_epsilon_decay', type=float, default=0.995, help='Q-learning epsilon decay rate (multiplicative)')
    grp_q.add_argument('--q_min_epsilon', type=float, default=0.05, help='Q-learning minimum exploration rate (epsilon)')
    # Add option to disable Q-Controller easily?
    grp_q.add_argument('--disable_q_controller', action='store_true', help='Disable the Q-learning controller for optimizer hyperparameters')


    # --- Group: Logging ---
    grp_log = parser.add_argument_group('Logging')
    grp_log.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging (requires wandb installed and login)')
    grp_log.add_argument('--wandb_project', type=str, default='wubu_nesting_v0.02', help='WandB project name') # Updated default
    grp_log.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team, optional)')

    args = parser.parse_args()

    # --- Argument Validation and Post-processing ---
    if args.num_levels <= 0: parser.error("--num_levels must be positive")
    num_transitions = max(0, args.num_levels - 1)

    def validate_list_len(arg_name, arg_list, expected_len, allow_empty_if_zero=False):
        """Helper function to validate list argument lengths."""
        is_list = isinstance(arg_list, list)
        if allow_empty_if_zero and expected_len == 0:
            if is_list and not arg_list: return # Empty list is okay if 0 expected
            if not is_list and arg_list is None: return # None is also okay
        # If not the allowed empty case, check length
        if not is_list or len(arg_list) != expected_len:
            parser.error(f"Argument --{arg_name} requires {expected_len} values, but got {arg_list}")

    validate_list_len("hyperbolic_dims", args.hyperbolic_dims, args.num_levels)
    validate_list_len("initial_curvatures", args.initial_curvatures, args.num_levels)
    validate_list_len("initial_scales", args.initial_scales, args.num_levels)
    validate_list_len("boundary_points_per_level", args.boundary_points_per_level, args.num_levels)

    # Handle optional spread values, default to scales if not provided
    if args.initial_spread_values is None:
        args.initial_spread_values = args.initial_scales
    else:
        validate_list_len("initial_spread_values", args.initial_spread_values, args.num_levels)

    # Validate transition list lengths only if num_levels > 1
    if args.num_levels > 1:
        validate_list_len("rotation_types", args.rotation_types, num_transitions)
        validate_list_len("transform_types", args.transform_types, num_transitions)
        # Handle transform_hidden_dims default and validation
        if args.transform_hidden_dims is None:
            args.transform_hidden_dims = [None] * num_transitions # Default to None for all transitions
        validate_list_len("transform_hidden_dims", args.transform_hidden_dims, num_transitions)
        # Convert 0 to None for hidden dims (0 often used as cli placeholder for None)
        args.transform_hidden_dims = [None if d == 0 else d for d in args.transform_hidden_dims]
    elif args.num_levels == 1:
        # Ensure transition args are empty if num_levels is 1, warn if provided
        if args.rotation_types: logger.warning("Ignoring --rotation_types as num_levels=1."); args.rotation_types = []
        if args.transform_types: logger.warning("Ignoring --transform_types as num_levels=1."); args.transform_types = []
        if args.transform_hidden_dims: logger.warning("Ignoring --transform_hidden_dims as num_levels=1."); args.transform_hidden_dims = []

    # Validate data fraction
    if not (0.0 < args.data_fraction <= 1.0):
        parser.error("--data_fraction must be between 0 (exclusive) and 1 (inclusive)")

    # Validate gradient accumulation steps
    if args.grad_accum_steps < 1:
        logger.warning("--grad_accum_steps must be >= 1. Setting to 1.")
        args.grad_accum_steps = 1

    # Validate num workers
    if args.num_workers < 0:
        logger.warning("--num_workers cannot be negative. Setting to 0.")
        args.num_workers = 0

    return args

# =====================================================================
# Main Execution Logic
# =====================================================================
def setup_distributed(local_rank):
    """Initializes DDP environment based on local_rank."""
    if local_rank == -1: # Not using DDP
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        is_distributed = False
        # Log device selection
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device)
            logger.info(f"Single Process: Using CUDA device {torch.cuda.current_device()} ({gpu_name})")
        else:
            logger.info("Single Process: Using CPU.")
    else: # Using DDP
        if not torch.cuda.is_available():
            logger.error("Distributed training requested (local_rank != -1) but CUDA is not available.")
            sys.exit(1)
        if torch.cuda.device_count() <= local_rank:
            logger.error(f"Invalid local_rank ({local_rank}). Only {torch.cuda.device_count()} GPU(s) available.")
            sys.exit(1)

        # Set the device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # Check required environment variables for 'env://' init method
        required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
        missing_env = [env for env in required_env if env not in os.environ]
        if missing_env:
            # If launched directly without torchrun/torch.distributed.launch, these might be missing
            logger.error(f"DDP environment variables missing: {missing_env}. Use 'torchrun' or 'torch.distributed.launch'.")
            # Attempting init anyway might work in some Slurm environments, but likely fails.
            # sys.exit(1) # Exit here for stricter check
            logger.warning("Attempting DDP init despite missing env vars. This might fail.")

        try:
            # Increase timeout for potentially large model/data transfers during setup
            timeout = timedelta(minutes=30)
            # Initialize the process group using NCCL backend (recommended for NVIDIA GPUs)
            init_process_group(backend="nccl", init_method="env://", timeout=timeout)

            world_size = get_world_size()
            rank = get_rank()
            is_distributed = True
            gpu_name = torch.cuda.get_device_name(device)
            logger.info(f"DDP Initialized: Rank {rank}/{world_size} | Device: cuda:{local_rank} ({gpu_name})")

            # Barrier to ensure all processes are initialized before proceeding
            torch.distributed.barrier()
            logger.debug(f"Rank {rank}/{world_size} passed initial DDP barrier.")

        except Exception as e:
            logger.error(f"DDP Initialization Failed on Rank {local_rank}: {e}", exc_info=True)
            if is_initialized(): # Attempt cleanup if partially initialized
                try: destroy_process_group()
                except Exception as cleanup_e: logger.error(f"Error during DDP cleanup on Rank {local_rank}: {cleanup_e}")
            sys.exit(1) # Exit if initialization fails

    return is_distributed, device, rank, world_size

def is_main_process():
    """Checks if the current process is the main process (rank 0 in DDP or single process)."""
    return not is_initialized() or get_rank() == 0

def run():
    """Main function to parse args, set up, and run training."""
    args = parse_arguments()

    # --- Setup Logging (Rank-aware) ---
    # Determine initial main process status before full DDP setup for early logging control
    # This helps suppress excessive logs from non-main ranks during DDP init itself
    temp_is_main = args.local_rank == -1 or args.local_rank == 0
    initial_log_level = logging.INFO if temp_is_main else logging.WARNING

    # Force reconfig of root logger to apply new level and format
    # Remove existing handlers to prevent duplicate messages if script is re-run in same process
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]: root_logger.removeHandler(h)
    logging.basicConfig(level=initial_log_level,
                        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                        force=True) # force=True ensures reconfiguration

    # --- Setup DDP Environment ---
    ddp_active, device, rank, world_size = setup_distributed(args.local_rank)
    am_main_process = is_main_process() # Use definitive check after DDP setup

    # Adjust log level again based on the final rank after DDP setup
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.getLogger().setLevel(log_level)

    # Add file handler only on the main process *after* DDP setup is complete
    if am_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log_filename = os.path.join(args.checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log")
        try:
            # Create a file handler
            file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO) # Log INFO level and above to file
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
            # Add the handler to the root logger
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Main process (Rank {rank}) logging detailed output to file: {log_filename}")
        except Exception as e:
            logger.error(f"Failed to set up file logging for Rank {rank}: {e}")

    # --- Log Initial Setup Info ---
    logger.info("="*60 + f"\n--- WuBu Nesting Trainer (v0.02 - Rank {rank}/{world_size}) ---") # Version updated
    logger.info(f"Status | DDP: {'Active' if ddp_active else 'Inactive'} | Device: {device} | Main Process: {am_main_process}")
    # Log all arguments only on the main process for clarity
    if am_main_process:
        logger.info("--- Command Line Arguments ---")
        for arg, value in sorted(vars(args).items()):
            logger.info(f"  --{arg}: {value}")
        logger.info("-----------------------------")
    # Log system info on all ranks for debugging potential environment differences
    logger.info(f"System | OS={platform.system()}/{platform.release()}, Python={sys.version.split()[0]}, Torch={torch.__version__}, CUDA={'Available ('+torch.version.cuda+')' if torch.cuda.is_available() else 'Not Available'}")
    logger.info("="*60)


    # --- Set Random Seeds ---
    # Ensure different seed per rank for independent initialization where needed (e.g., dropout masks)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: CUDNN benchmark and determinism settings
        # torch.backends.cudnn.benchmark = False # Might hurt performance, but increases reproducibility
        # torch.backends.cudnn.deterministic = True # Ensure deterministic conv algorithms
    logger.info(f"Random seed set to {seed} for Rank {rank}")

    # --- Build Configuration Dictionaries from Args ---
    # WuBu Config
    wubu_config = DEFAULT_CONFIG_WUBU.copy()
    wubu_arg_keys = [
        "num_levels", "hyperbolic_dims", "initial_curvatures", "initial_scales",
        "initial_spread_values", "boundary_points_per_level", "rotation_types",
        "transform_types", "transform_hidden_dims", "tangent_flow_type",
        "tangent_flow_scale", "aggregation_method", "relative_vector_aggregation",
        "dropout", "level_descriptor_init_scale", "curvature_min_value",
        "scale_min_value", "spread_min_value", "tangent_input_combination_dims"
    ]
    for key in wubu_arg_keys:
        if hasattr(args, key) and getattr(args, key) is not None: # Check if arg exists and is not None
            wubu_config[key] = getattr(args, key)
    # Handle boolean flags (args store True if flag is *present*)
    wubu_config["learnable_curvature"] = not args.no_learnable_curvature
    wubu_config["learnable_scales"] = not args.no_learnable_scales
    wubu_config["learnable_spread"] = not args.no_learnable_spread
    wubu_config["use_level_descriptors"] = not args.no_level_descriptors
    wubu_config["use_level_spread"] = not args.no_level_spread
    wubu_config["use_tangent_flow"] = not args.no_tangent_flow

    # Sequence Model Config
    sequence_config = { k: getattr(args, k) for k in [
        "local_hidden_size", "decoder_memory_dim", "context_window",
        "n_gram_sizes", "n_gram_vocab_size",
        "num_encoder_layers", "num_decoder_layers", # Added layers
        "num_encoder_heads", "num_decoder_heads"     # Added heads
        ] if hasattr(args, k)} # Only include args that exist
    sequence_config["use_hierarchical_decoder"] = not args.no_hierarchical_decoder
    sequence_config["vocab_size"] = 256 # Fixed for byte-level modeling

    # Log the final computed configurations on the main process
    if am_main_process:
        logger.info("--- Final WuBu Config Used ---")
        for k, v in sorted(wubu_config.items()): logger.info(f"  {k}: {v}")
        logger.info("--- Final Sequence Config Used ---")
        for k, v in sorted(sequence_config.items()): logger.info(f"  {k}: {v}")
        logger.info("---------------------------------")

    # --- Initialize WandB ---
    use_wandb = args.wandb and am_main_process and WANDB_AVAILABLE
    if use_wandb:
        try:
            # Combine all configs for logging (sanitize lists/complex objects if needed)
            full_config = {**vars(args), "wubu_config": wubu_config, "sequence_config": sequence_config}
            # Basic sanitization: convert lists to tuples, handle None
            sanitized_config = {}
            for k, v in full_config.items():
                if isinstance(v, list): sanitized_config[k] = tuple(v)
                elif v is None: sanitized_config[k] = 'None'
                else: sanitized_config[k] = v
                # Could add more checks for dicts, etc. if needed

            # Create a descriptive run name
            run_name = f"WuBu_L{args.num_levels}_D{'x'.join(map(str, args.hyperbolic_dims))}_B{args.batch_size}_LR{args.learning_rate:.1e}_{datetime.now().strftime('%H%M')}_v0.02" # Updated version
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity, # Optional entity (user/team)
                config=sanitized_config, # Log hyperparameters
                name=run_name,           # Set run name
                resume="allow",          # Allow resuming previous runs if ID matches
                id=wandb.util.generate_id() if args.resume is None else None # Generate new ID unless resuming
            )
            logger.info(f"WandB initialized: project='{args.wandb_project}', run='{wandb.run.name}' (ID: {wandb.run.id})")
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}. Disabling WandB for this run.", exc_info=True)
            use_wandb = False # Disable if initialization fails

    # --- Load Datasets ---
    train_dataset = None; val_dataset = None
    try:
        logger.info(f"Loading train dataset: {args.data_path}")
        train_dataset = ByteIterableDataset(args.data_path, args.context_window, args.data_fraction)
        train_dataset.set_seed(args.seed) # Set initial seed for shuffling

        if args.val_data_path:
            if os.path.exists(args.val_data_path):
                logger.info(f"Loading validation dataset: {args.val_data_path}")
                # Use full validation dataset (data_fraction=1.0)
                val_dataset = ByteIterableDataset(args.val_data_path, args.context_window, 1.0)
                val_dataset.set_seed(args.seed) # Use same base seed for consistency if desired
            else:
                logger.warning(f"Validation data path specified but not found: {args.val_data_path}. Validation will be skipped.")
    except Exception as e:
        logger.error(f"Dataset initialization failed: {e}", exc_info=True)
        if ddp_active: destroy_process_group() # Clean up DDP if data fails
        sys.exit(1)

    # --- Create DataLoaders ---
    # Calculate per-GPU batch size
    if args.batch_size < world_size:
        # Ensure batch size is at least 1 per GPU
        logger.warning(f"Global batch size ({args.batch_size}) is less than world size ({world_size}). Setting batch_size_per_gpu to 1.")
        batch_size_per_gpu = 1
    else:
        # Divide global batch size by number of GPUs
        batch_size_per_gpu = max(1, args.batch_size // world_size)

    # Calculate effective batch size (considering accumulation)
    effective_bs = batch_size_per_gpu * world_size * args.grad_accum_steps
    logger.info(f"Batch Config | Global BS: {args.batch_size} | Per-GPU BS: {batch_size_per_gpu} | Accum Steps: {args.grad_accum_steps} | Effective BS: {effective_bs}")

    # Create samplers for DDP if active
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed, drop_last=True) if ddp_active else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if ddp_active and val_dataset else None # No shuffle/drop_last for val

    # Worker init function for reproducible dataloading across workers/epochs
    base_seed_offset = args.seed * world_size # Base offset unique to this run
    # Use functools.partial to pass fixed arguments (base_seed, rank_offset) to seed_worker
    # Rank offset ensures workers on different ranks get different initial seeds
    train_worker_init_fn = functools.partial(seed_worker, base_seed=base_seed_offset, rank_offset=rank * args.num_workers) if args.num_workers > 0 else None
    # Use slightly different base seed for validation workers for more independent randomness
    val_worker_init_fn = functools.partial(seed_worker, base_seed=base_seed_offset + 1, rank_offset=rank * args.num_workers) if args.num_workers > 0 and val_dataset else None

    # Persistent workers can speed up epoch starts if workers > 0 and not on Windows (due to fork limitations)
    use_persistent_workers = (args.num_workers > 0) and (platform.system() != 'Windows')
    if use_persistent_workers:
        logger.info(f"Using persistent workers for DataLoaders (Num Workers: {args.num_workers})")
    else:
        logger.info(f"Not using persistent workers (Num Workers: {args.num_workers}, OS: {platform.system()})")


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True, # Pin memory if using GPU
        drop_last=True, # Drop last batch if incomplete (recommended for DDP consistency)
        worker_init_fn=train_worker_init_fn,
        persistent_workers=use_persistent_workers,
        shuffle=False # Never shuffle even if using DistributedSampler (IterableDataset handles its own internal shuffle)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False, # Do not drop last batch for validation
        worker_init_fn=val_worker_init_fn,
        persistent_workers=use_persistent_workers,
        shuffle=False # No need to shuffle validation data
    ) if val_dataset else None

    # --- Initialize Model ---
    try:
        model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config).to(device)
        if am_main_process:
            # Calculate and log parameter counts
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model Initialized | Total Parameters: {total_params:,} | Trainable Parameters: {trainable_params:,}")
            # Optional: Print model summary (can be very verbose)
            # print(model)
    except Exception as model_ex:
        logger.error(f"Model initialization failed: {model_ex}", exc_info=True)
        if ddp_active: destroy_process_group() # Cleanup DDP if model fails
        sys.exit(1)

    # --- Wrap Model for DDP ---
    if ddp_active:
        # find_unused_parameters=True can help debug DDP hangs if some params aren't used in forward, but adds overhead.
        # Set to False unless needed
        find_unused = False # Set to True only if debugging DDP issues.
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],       # List of GPU IDs on this node for this process
            output_device=args.local_rank,      # Device for the output of the module
            find_unused_parameters=find_unused,
            gradient_as_bucket_view=True # Optimization for DDP communication
        )
        logger.info(f"Model wrapped with DistributedDataParallel on Rank {rank} (find_unused={find_unused}).")
        # Barrier after wrapping ensures all ranks have the DDP model before proceeding (e.g., to optimizer init)
        torch.distributed.barrier()

    # --- Initialize Optimizer ---
    q_cfg = None # Default to disabled
    if not args.disable_q_controller:
        q_cfg = {
            "learning_rate": args.q_learning_rate, "discount": args.q_discount,
            "epsilon": args.q_epsilon, "epsilon_decay": args.q_epsilon_decay,
            "min_epsilon": args.q_min_epsilon
            # lr_scale_options / momentum_scale_options use defaults in controller if None
        }
    try:
        optimizer = HAKMEMEnhancedSGD(
            model.parameters(), # Pass model parameters
            lr=args.learning_rate,
            momentum=0.9, # Default momentum, Q-controller might adjust scale
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm, # Max norm passed for reference, clipping done by Trainer
            q_learning_config=q_cfg # Pass the Q-learning config dict or None
        )
        logger.info(f"Optimizer '{type(optimizer).__name__}' initialized on Rank {rank} with BaseLR={args.learning_rate}, WD={args.weight_decay}.")
    except Exception as optim_ex:
        logger.error(f"Optimizer initialization failed: {optim_ex}", exc_info=True)
        if ddp_active: destroy_process_group()
        sys.exit(1)

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model, optimizer=optimizer, device=device,
        train_loader=train_loader, val_loader=val_loader,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=(not args.no_amp), # Use AMP unless explicitly disabled
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        wandb_enabled=use_wandb,
        max_grad_norm=args.max_grad_norm, # Pass max_grad_norm to trainer for clipping
        rank=rank, world_size=world_size,
        detect_anomaly=args.detect_anomaly
    )
    trainer.args = args # Store args in trainer for saving in checkpoint

    # --- Load Checkpoint if Resuming ---
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Rank {rank}: Attempting to resume training from checkpoint: {args.resume}")
            # load_checkpoint returns the epoch to START from (epoch after the saved one)
            start_epoch = trainer.load_checkpoint(args.resume)
        else:
            logger.warning(f"Rank {rank}: Resume checkpoint specified but not found: {args.resume}. Starting training from scratch.")
        # Barrier ensures all ranks have loaded checkpoint (or decided not to) before starting training
        if ddp_active:
            logger.debug(f"Rank {rank}: Entering barrier after checkpoint load attempt.")
            torch.distributed.barrier()
            logger.debug(f"Rank {rank}: Exited barrier after checkpoint load attempt.")
    else:
        logger.info(f"Rank {rank}: Starting training from scratch.")

    # --- Start Training ---
    if ddp_active: torch.distributed.barrier() # Final barrier ensure setup is complete everywhere
    training_successful = False
    try:
        trainer.train(args.epochs, start_epoch=start_epoch)
        training_successful = True # Mark training as completed successfully
    except KeyboardInterrupt:
        logger.info(f"Training interrupted by user (Rank {rank}).")
        # Consider training partially successful for saving purposes
        training_successful = True
    except Exception as train_ex:
        logger.error(f"Critical error during training on Rank {rank}: {train_ex}", exc_info=True)
        # Training failed, but still try to save if needed
        training_successful = False
    finally:
        # --- Cleanup and Final Save ---
        # Save final checkpoint on the main process after training finishes or is interrupted/fails
        if am_main_process:
            logger.info("Attempting to save final checkpoint...")
            # Use last validation metrics if available, otherwise pass None
            metrics = getattr(trainer, 'last_val_metrics', None)
            # Pass training success status? Currently just saves regardless.
            trainer._save_checkpoint(is_intermediate=False, metrics=metrics)

        # Destroy DDP process group
        if ddp_active:
            logger.debug(f"Rank {rank}: Entering final DDP cleanup barrier.")
            torch.distributed.barrier() # Ensure all ranks are ready to cleanup
            logger.debug(f"Rank {rank}: Attempting to destroy process group.")
            try:
                destroy_process_group()
                logger.info(f"DDP process group destroyed successfully (Rank {rank}).")
            except Exception as ddp_err:
                logger.error(f"Error destroying DDP process group on Rank {rank}: {ddp_err}")

        # Finish WandB run if it was initialized
        if use_wandb and wandb.run:
            logger.info("Finishing WandB run...")
            try:
                # Mark run as finished (or crashed if training failed?)
                # wandb.finish(exit_code=0 if training_successful else 1)
                wandb.finish()
                logger.info("WandB run finished.")
            except Exception as wb_err:
                logger.error(f"Error finishing WandB run: {wb_err}")

    logger.info(f"Training script finished execution (Rank {rank}).")


# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    # Recommended DDP launch method:
    # torchrun --nproc_per_node=NUM_GPUS WuBuNest_Trainer.py [ARGS]
    # or legacy:
    # python -m torch.distributed.launch --nproc_per_node=NUM_GPUS WuBuNest_Trainer...py [ARGS]
    run()