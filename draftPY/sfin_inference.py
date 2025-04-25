#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFIN Inference Script (Adapted from BSFIN v2 and Bytropix Inference)

Runs inference using the BSFINModel architecture.
Supports standard single-prompt and interactive modes.
(Corrected PositionalEncoding max_len for checkpoint loading)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# Kept for potential future use, not directly needed for inference script core logic
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
from torch import amp  # For automatic mixed precision
from dataclasses import dataclass
import itertools
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SFIN_Inference")

# =====================================================================
# Data Structures and Configuration Classes (from bsfin_main.py)
# =====================================================================


@dataclass
class SamplerConfig:
    """Configuration for entropy-based sampling (used in generate)."""
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# =====================================================================
# Quantum Noise Function (from bsfin_main.py)
# =====================================================================


def add_quantum_noise(tensor, noise_prob=0.05, noise_scale=0.1, noise_type="phase_and_amplitude"):
    """Inject quantum-inspired noise. Not typically used during inference."""
    # During inference (model.eval()), requires_grad is often False, so this won't add noise.
    # If noise is needed during eval, ensure requires_grad is True or call model.train().
    if noise_scale <= 0 or not tensor.requires_grad or not tensor.is_floating_point() or not nn.Module.training:
        return tensor

    with torch.no_grad():
        if noise_type == "phase_only":
            mask = torch.rand_like(tensor) < noise_prob
            flip = torch.where(mask, -1.0, 1.0)
            noisy_tensor = tensor * flip
        elif noise_type == "amplitude_only":
            mask = torch.rand_like(tensor) < noise_prob
            noise = torch.randn_like(tensor) * noise_scale * mask
            noisy_tensor = tensor + noise
        else:  # phase_and_amplitude (default)
            phase_mask = torch.rand_like(tensor) < noise_prob
            amp_mask = torch.rand_like(tensor) < noise_prob
            flip = torch.where(phase_mask, -1.0, 1.0)
            noise = torch.randn_like(tensor) * noise_scale * amp_mask
            noisy_tensor = tensor * flip + noise

    # Re-attach gradient history if needed (only relevant if tensor initially requires grad)
    return noisy_tensor.clone().requires_grad_(tensor.requires_grad)


# =====================================================================
# ByteTokenizer (from bsfin_main.py - handles tensors)
# =====================================================================

class ByteTokenizer:
    """Simple tokenizer for byte-level processing."""

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of byte values."""
        return list(text.encode('utf-8'))

    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        """Decodes a sequence of byte values back into a string."""
        valid_bytes = []
        for b in byte_sequence:
            # Ensure bytes are valid integers in the 0-255 range
            if isinstance(b, (int, np.integer)) and 0 <= b <= 255:
                valid_bytes.append(int(b))
            elif isinstance(b, torch.Tensor):  # Handle tensor elements
                b_item = b.item()
                if isinstance(b_item, int) and 0 <= b_item <= 255:
                    valid_bytes.append(b_item)
                # Handle potential float tensors
                elif isinstance(b_item, float) and 0 <= int(b_item) <= 255:
                    valid_bytes.append(int(b_item))
        # Replace invalid UTF-8 sequences
        return bytes(valid_bytes).decode('utf-8', errors='replace')

# =====================================================================
# Babylon Index (from bsfin_main.py - more refined)
# =====================================================================


class BabylonIndex:
    """Entropy-based index for byte sequence analysis and patching."""

    def __init__(self, scales: List[int] = [3, 5, 7], max_cache_size: int = 50000, min_entropy_threshold: float = 0.5):
        self.scales = sorted(list(set(scales)))
        self.entropy_cache = {}
        self.max_cache_size = max_cache_size
        self.min_entropy_threshold = min_entropy_threshold
        # logger.info(f"BabylonIndex initialized with scales: {self.scales}, cache size: {max_cache_size}, entropy threshold: {min_entropy_threshold}")

    def _clean_cache(self):
        """Removes oldest items if cache exceeds max size."""
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - \
                (self.max_cache_size * 4 // 5)
            keys_to_remove = list(itertools.islice(
                self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache:
                    del self.entropy_cache[k]

    def _is_valid_utf8_boundary(self, byte_seq: Union[List[int], np.ndarray], boundary: int) -> bool:
        """Checks if a potential boundary is valid (not mid-UTF8 char)."""
        if boundary <= 0 or boundary >= len(byte_seq):
            return True
        byte_at_boundary = byte_seq[boundary]
        # Continuation bytes are in the range 0x80 (10000000) to 0xBF (10111111)
        return not (0x80 <= byte_at_boundary <= 0xBF)

    # Use Tuple[int, ...] for type hint
    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...]]) -> float:
        """Computes Shannon entropy for a window of bytes, using caching."""
        cache_key = None
        if isinstance(byte_window, tuple):
            cache_key = byte_window
            if cache_key in self.entropy_cache:
                return self.entropy_cache[cache_key]
            if not byte_window:
                return 0.0
            byte_window_np = np.array(byte_window, dtype=np.uint8)
        elif isinstance(byte_window, np.ndarray):
            if byte_window.size == 0:
                return 0.0
            byte_window_np = byte_window
            # Caching numpy arrays directly is tricky, convert to tuple if needed
            # cache_key = tuple(byte_window_np.tolist())
            # if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        else:
            logger.warning(
                f"Unsupported type for compute_entropy: {type(byte_window)}")
            return 0.0

        try:
            if not np.issubdtype(byte_window_np.dtype, np.integer):
                logger.warning(
                    f"Non-integer dtype {byte_window_np.dtype} passed to bincount. Attempting cast.")
                byte_window_np = byte_window_np.astype(np.uint8)
            byte_counts = np.bincount(byte_window_np, minlength=256)
        except TypeError as e:
            logger.error(
                f"TypeError in np.bincount. Input type: {type(byte_window_np)}, dtype: {byte_window_np.dtype}. Error: {e}")
            return 0.0  # Or handle differently

        total_bytes = byte_counts.sum()
        if total_bytes == 0:
            return 0.0
        probs = byte_counts[byte_counts > 0] / total_bytes
        entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))

        if cache_key is not None:  # Cache only if input was tuple or converted
            self.entropy_cache[cache_key] = entropy
            self._clean_cache()

        return entropy

    def find_patch_boundaries(self, byte_seq_tensor: torch.Tensor) -> List[int]:
        """Identifies potential patch boundaries based on entropy."""
        if byte_seq_tensor.numel() == 0:
            return []

        if byte_seq_tensor.dim() > 1:
            if byte_seq_tensor.size(0) == 1:  # Handle batch size 1
                # Use tolist() for list of ints
                byte_seq_list = byte_seq_tensor[0].cpu().tolist()
            else:
                logger.warning(
                    "find_patch_boundaries expects 1D tensor or batch size 1. Using first element.")
                byte_seq_list = byte_seq_tensor[0].cpu().tolist()
        else:
            byte_seq_list = byte_seq_tensor.cpu().tolist()

        seq_len = len(byte_seq_list)
        min_scale = min(self.scales, default=1)
        if seq_len < min_scale:
            return []

        potential_boundaries = set()
        # Adaptive window based on scales/length, capped
        window_size = min(max(self.scales, default=16), seq_len // 2, 64)
        window_size = max(window_size, min_scale)  # Ensure window is at least min scale

        entropies = []
        for i in range(seq_len - window_size + 1):
            window_tuple = tuple(byte_seq_list[i: i + window_size])
            entropy = self.compute_entropy(window_tuple)
            entropies.append((i, entropy))  # Store window start index and entropy

        entropies.sort(key=lambda x: x[1], reverse=True)
        num_boundaries_target = max(1, seq_len // 128)
        selected_count = 0

        for start_pos, entropy_val in entropies:
            boundary_candidate = start_pos
            if entropy_val > self.min_entropy_threshold and selected_count < num_boundaries_target * 2:
                if self._is_valid_utf8_boundary(byte_seq_list, boundary_candidate):
                    potential_boundaries.add(boundary_candidate)
                    selected_count += 1

        final_boundaries = sorted(
            [b for b in list(potential_boundaries) if 0 < b < seq_len])
        min_patch_size = 16  # Minimum allowed patch size
        merged_boundaries = []
        if final_boundaries:
            last_boundary = 0  # Implicit start boundary
            for b in final_boundaries:
                if b - last_boundary >= min_patch_size:
                    merged_boundaries.append(b)
                    last_boundary = b
            final_boundaries = merged_boundaries

        # logger.debug(f"Found boundaries: {final_boundaries} for seq_len {seq_len}")
        return final_boundaries

    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Splits a 1D byte tensor into patches based on found boundaries."""
        if byte_seq_tensor.numel() == 0:
            return []
        if byte_seq_tensor.dim() != 1:
            if byte_seq_tensor.dim() == 2 and byte_seq_tensor.size(0) == 1:
                byte_seq_tensor = byte_seq_tensor.squeeze(0)
            else:
                raise ValueError(
                    f"create_patches expects a 1D tensor, got shape {byte_seq_tensor.shape}")

        boundaries = self.find_patch_boundaries(byte_seq_tensor)
        patches = []
        start_idx = 0
        seq_len = byte_seq_tensor.size(0)

        for end_idx in boundaries:
            if start_idx < end_idx <= seq_len:
                patch = byte_seq_tensor[start_idx:end_idx]
                if patch.numel() > 0:
                    patches.append(patch)
                start_idx = end_idx
            elif end_idx <= start_idx:
                logger.warning(
                    f"Skipping invalid or out-of-order boundary {end_idx} <= {start_idx}")
            elif end_idx > seq_len:
                logger.warning(
                    f"Boundary {end_idx} exceeds sequence length {seq_len}. Ignoring.")

        if start_idx < seq_len:
            final_patch = byte_seq_tensor[start_idx:]
            if final_patch.numel() > 0:
                patches.append(final_patch)

        # logger.debug(f"Created {len(patches)} patches.")
        return patches

    @torch.no_grad()
    def reset_context(self):
        """Resets internal caches."""
        self.entropy_cache = {}
        # logger.debug("BabylonIndex context reset.")

# =====================================================================
# Core Model Components (from bsfin_main.py)
# =====================================================================


class CrossAttentionBlock(nn.Module):
    """Standard Cross-attention block."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        original_num_heads = num_heads
        # Auto-adjust num_heads if hidden_size is not divisible
        if hidden_size % num_heads != 0:
            possible_heads = [
                h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
            if not possible_heads:
                raise ValueError(
                    f"hidden_size {hidden_size} not divisible by any number of heads <= {original_num_heads}")
            num_heads = possible_heads[0]
            logger.warning(
                f"Adjusted num_heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.zeros_(self.v_proj.bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: Tensor [B, Nq, H]
            keys_values: Tensor [B, Nkv, H]
            attention_mask: Optional Tensor [B, Nkv] or [B, Nq, Nkv]. True indicates MASKED position.
        """
        batch_size, num_queries, _ = queries.size()
        seq_len_kv = keys_values.size(1)  # Length of key/value sequence
        device = queries.device

        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)

        q = self.q_proj(queries_norm).view(
            batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, Nq, d]
        k = self.k_proj(keys_values_norm).view(
            batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, Nkv, d]
        v = self.v_proj(keys_values_norm).view(
            batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, Nkv, d]

        # --- Attention Mask Handling ---
        attn_mask_bool = None
        if attention_mask is not None:
            # Handle different mask dimensions and ensure boolean type
            if attention_mask.dim() == 2:  # Shape [B, Nkv]
                attn_mask_bool = attention_mask.unsqueeze(1).unsqueeze(
                    2).expand(-1, self.num_heads, num_queries, -1)  # [B, h, Nq, Nkv]
            elif attention_mask.dim() == 3:  # Shape [B, Nq, Nkv]
                attn_mask_bool = attention_mask.unsqueeze(
                    1).expand(-1, self.num_heads, -1, -1)  # [B, h, Nq, Nkv]
            elif attention_mask.dim() == 4:  # Shape [B, h, Nq, Nkv]
                attn_mask_bool = attention_mask
            else:
                logger.warning(
                    f"Unsupported attention mask shape: {attention_mask.shape}. Ignoring mask.")
            if attn_mask_bool is not None:
                attn_mask_bool = attn_mask_bool.bool()

        # --- Attention Calculation (Prioritize Flash Attention) ---
        use_flash = hasattr(F, 'scaled_dot_product_attention') and (
            attn_mask_bool is None or attn_mask_bool.dtype == torch.bool)

        if use_flash:
            # Flash attention mask: True means KEEP (inverse of our bool mask)
            flash_mask = ~attn_mask_bool if attn_mask_bool is not None else None
            try:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=flash_mask,  # Boolean mask where True=KEEP
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False  # Cross-attention is not causal
                )
            except Exception as e:
                logger.warning(
                    f"Flash attention failed: {e}. Falling back to manual.")
                use_flash = False  # Fallback needed

        if not use_flash:  # Manual calculation or fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * \
                scale  # [B, h, Nq, Nkv]
            if attn_mask_bool is not None:
                # Apply boolean mask (True means MASK)
                scores = scores.masked_fill(attn_mask_bool, float('-inf'))
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v)  # [B, h, Nq, d]

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(
            batch_size, num_queries, self.hidden_size)  # [B, Nq, H]
        output = self.out_proj(output)
        # NOTE: BSFIN main script doesn't have residual connection here, keeping consistent
        # output = queries + output # Optional residual connection
        return output


class LocalEncoder(nn.Module):
    """Encodes byte patches into real-valued representations using a Transformer, incorporating N-grams."""

    def __init__(self, hidden_size: int = 256, num_layers: int = 1, num_heads: int = 8, dropout: float = 0.1, n_gram_sizes: List[int] = [3, 4], n_gram_vocab_size: int = 30000):
        super().__init__()
        self.hidden_size = hidden_size
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight,
                        mean=0.0, std=1.0/math.sqrt(hidden_size))

        # N-gram Embeddings Setup
        self.n_gram_sizes = sorted(
            list(set(n_gram_sizes))) if n_gram_sizes else []
        self.n_gram_embeddings = None
        self.n_gram_vocab_size = n_gram_vocab_size
        if self.n_gram_sizes:
            self.n_gram_embeddings = nn.ModuleDict({
                f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes
            })
            for emb in self.n_gram_embeddings.values():
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)  # Smaller std
            logger.info(
                f"LocalEncoder using N-grams: {self.n_gram_sizes} with vocab size {n_gram_vocab_size}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size *
                                                   4, dropout=dropout, batch_first=True, activation=F.gelu)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.patch_pooling_attention = CrossAttentionBlock(
            hidden_size, num_heads, dropout)
        # Learnable query
        self.patch_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _get_n_gram_hashes(self, byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        """Calculates rolling hashes for n-grams (simple modulo hashing)."""
        if byte_sequence.size(1) < n:
            return torch.empty(byte_sequence.size(0), 0, dtype=torch.long, device=byte_sequence.device)
        # Shape [B, NumWindows, n]
        windows = byte_sequence.unfold(1, n, 1)
        hashes = windows.long().sum(dim=-1)  # Simple sum hashing
        return hashes % self.n_gram_vocab_size  # Shape [B, NumWindows]

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates a causal mask (True = ignore)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, patches: List[torch.Tensor]) -> torch.Tensor:
        """
        Encodes a list of byte patches (assumed from a single batch item).
        Input: List[Tensor[patch_len]] -> Output: Tensor[1, num_patches, hidden_size]
        """
        if not patches:
            device = next(self.parameters()).device
            return torch.empty((1, 0, self.hidden_size), device=device)

        batch_size = 1  # Process one sequence at a time
        device = patches[0].device
        patch_representations = []

        for patch_bytes in patches:  # patch_bytes is [patch_len]
            patch_len = patch_bytes.size(0)
            if patch_len == 0:
                continue
            # [1, patch_len]
            patch_bytes_batched = patch_bytes.unsqueeze(0)

            x = self.byte_embeddings(patch_bytes_batched)  # [1, patch_len, H]
            if self.n_gram_embeddings:
                n_gram_features = torch.zeros_like(x)  # [1, patch_len, H]
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        n_gram_hashes = self._get_n_gram_hashes(
                            patch_bytes_batched, n)  # [1, NumWindows]
                        ngram_embeds = self.n_gram_embeddings[f'n{n}'](
                            n_gram_hashes)  # [1, NumWindows, H]
                        num_windows = ngram_embeds.size(1)
                        if num_windows < patch_len:
                            padding_size = patch_len - num_windows
                            padding = torch.zeros(
                                1, padding_size, self.hidden_size, device=device)
                            ngram_embeds_padded = torch.cat(
                                [ngram_embeds, padding], dim=1)
                        else:
                            ngram_embeds_padded = ngram_embeds[:, :patch_len, :]
                        n_gram_features += ngram_embeds_padded  # Accumulate
                x = x + n_gram_features
            x = self.dropout(x)

            # [patch_len, patch_len]
            causal_mask = self.create_causal_mask(patch_len, device)
            processed_bytes = self.transformer(
                x, mask=causal_mask)  # [1, patch_len, H]

            batch_query = self.patch_query  # [1, 1, H]
            patch_repr = self.patch_pooling_attention(
                queries=batch_query, keys_values=processed_bytes)  # [1, 1, H]
            patch_representations.append(patch_repr)

        if not patch_representations:
            return torch.empty((1, 0, self.hidden_size), device=device)

        # [1, num_patches, H]
        patches_combined = torch.cat(patch_representations, dim=1)
        return self.norm(patches_combined)


class RealToComplexProjection(nn.Module):
    """Projects real vectors to complex representations (real, imag parts)."""

    def __init__(self, input_dim: int, complex_dim: int):
        super().__init__()
        self.proj_real = nn.Linear(input_dim, complex_dim)
        self.proj_imag = nn.Linear(input_dim, complex_dim)
        nn.init.xavier_uniform_(self.proj_real.weight)
        nn.init.zeros_(self.proj_real.bias)
        nn.init.xavier_uniform_(self.proj_imag.weight)
        nn.init.zeros_(self.proj_imag.bias)

    def forward(self, x_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.proj_real(x_real), self.proj_imag(x_real)


class ComplexToRealProjection(nn.Module):
    """Projects complex representations (real, imag) back to real vectors."""

    def __init__(self, complex_dim: int, output_dim: int, method: str = "concat"):
        super().__init__()
        self.method = method
        if method == "concat":
            self.proj = nn.Linear(complex_dim * 2, output_dim)
        elif method == "magnitude":
            self.proj = nn.Linear(complex_dim, output_dim)  # Loses phase
        else:
            logger.warning(
                f"Unknown ComplexToRealProjection method '{method}'. Defaulting to 'concat'.")
            self.method = "concat"
            self.proj = nn.Linear(complex_dim * 2, output_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        real, imag = x
        if self.method == "concat":
            combined = torch.cat([real, imag], dim=-1)
            return self.proj(combined)
        elif self.method == "magnitude":
            magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
            return self.proj(magnitude)
        else:  # Default concat
            combined = torch.cat([real, imag], dim=-1)
            return self.proj(combined)


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex inputs (real, imag)."""

    def __init__(self, dim, eps=1e-5, coupled=True):
        super().__init__()
        self.real_norm = nn.LayerNorm(dim, eps=eps)
        self.imag_norm = nn.LayerNorm(dim, eps=eps)
        self.coupled = coupled
        if coupled:
            self.coupling = nn.Parameter(torch.tensor(0.1))
            self.cross_gain_ri = nn.Parameter(torch.zeros(dim))
            self.cross_gain_ir = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x
        if self.coupled:
            real_normed = self.real_norm(real)
            imag_normed = self.imag_norm(imag)
            coupling_strength = torch.sigmoid(self.coupling)
            real_out = real_normed + coupling_strength * self.cross_gain_ri * imag_normed
            imag_out = imag_normed + coupling_strength * self.cross_gain_ir * real_normed
            return real_out, imag_out
        else:
            return self.real_norm(real), self.imag_norm(imag)


class PositionalEncoding(nn.Module):
    """Complex positional encoding with optional learnable frequency scaling."""

    def __init__(self, dim, max_len=1024, phase_shift=True, learnable=True):
        super().__init__()
        self.dim = dim
        self.learnable = learnable
        self.max_len = max_len # Store max_len used for buffer creation

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term_base = torch.exp(torch.arange(
            0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe_real = torch.zeros(max_len, dim)
        pe_real[:, 0::2] = torch.sin(position * div_term_base)
        pe_real[:, 1::2] = torch.cos(position * div_term_base)
        pe_imag = torch.zeros(max_len, dim)
        if phase_shift:
            pe_imag[:, 0::2] = torch.cos(position * div_term_base)
            pe_imag[:, 1::2] = -torch.sin(position * div_term_base)
        else:
            pe_imag[:, 0::2] = torch.sin(position * div_term_base + math.pi / 4)
            pe_imag[:, 1::2] = torch.cos(position * div_term_base + math.pi / 4)
        self.register_buffer('pe_real_base', pe_real)
        self.register_buffer('pe_imag_base', pe_imag)
        # Store base frequencies
        self.register_buffer('div_term_base', div_term_base)

        if learnable:
            self.real_scale = nn.Parameter(torch.ones(1, 1, dim))
            self.imag_scale = nn.Parameter(torch.ones(1, 1, dim))
            self.real_shift = nn.Parameter(torch.zeros(1, 1, dim))
            self.imag_shift = nn.Parameter(torch.zeros(1, 1, dim))
            # Shape [dim/2]
            self.frequency_scale_factors = nn.Parameter(torch.ones(dim // 2))
            logger.info("Using learnable Positional Encoding.")

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adds positional encoding to complex input."""
        real, imag = x
        seq_len = real.size(1)
        device = real.device

        if self.learnable:
            scaled_div_term = self.div_term_base.to(
                device) * torch.clamp(self.frequency_scale_factors, min=1e-2)  # Shape [dim/2]
            # [Seq, 1]
            position = torch.arange(seq_len, device=device).unsqueeze(1).float()
            pe_real_learn = torch.zeros(seq_len, self.dim, device=device)
            pe_imag_learn = torch.zeros(seq_len, self.dim, device=device)
            angles = position * scaled_div_term  # [Seq, dim/2]
            pe_real_learn[:, 0::2] = torch.sin(angles)
            pe_real_learn[:, 1::2] = torch.cos(angles)
            pe_imag_learn[:, 0::2] = torch.cos(angles)
            pe_imag_learn[:, 1::2] = -torch.sin(angles)
            pe_real = pe_real_learn * self.real_scale + self.real_shift
            pe_imag = pe_imag_learn * self.imag_scale + self.imag_shift
            return real + pe_real, imag + pe_imag
        else:  # Use fixed positional encoding
            # Slice the base PE to match sequence length and add
            # Ensure slicing does not go out of bounds for registered buffers
            slice_len = min(seq_len, self.max_len)
            if seq_len > self.max_len:
                logger.warning(
                    f"Input sequence length ({seq_len}) > PE max_len ({self.max_len}). PE will be reused/truncated.")
                # Simple tiling or other strategy could be implemented if needed
            return real + self.pe_real_base[:slice_len, :].to(device), imag + self.pe_imag_base[:slice_len, :].to(device)


class EntangledInterferenceLayer(nn.Module):
    """Quantum-inspired interference layer with mask handling."""

    def __init__(self, dim, heads=8, dropout=0.1, interference_type="quantum", use_entanglement=True, noise_scale=0.1, use_rotary=True, adaptive_attention=True):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.interference_type = interference_type
        self.use_entanglement = use_entanglement
        self.noise_scale = noise_scale  # Note: Noise usually off during inference
        self.use_rotary = use_rotary
        self.adaptive_attention = adaptive_attention
        self.phase_shifts = nn.Parameter(
            torch.randn(heads, self.head_dim) * 0.02)
        if use_entanglement:
            self.entanglement_matrix = nn.Parameter(
                torch.eye(heads) + torch.randn(heads, heads) * 0.01)
        else:
            self.register_buffer('entanglement_matrix',
                                 torch.eye(heads), persistent=False)
        self.q_real = nn.Linear(dim, dim)
        self.k_real = nn.Linear(dim, dim)
        self.v_real = nn.Linear(dim, dim)
        self.q_imag = nn.Linear(dim, dim)
        self.k_imag = nn.Linear(dim, dim)
        self.v_imag = nn.Linear(dim, dim)
        self.out_real = nn.Linear(dim, dim)
        self.out_imag = nn.Linear(dim, dim)
        if use_rotary:
            self.rotary_dim = min(self.head_dim, 32)  # Rotary dimension cap
            base_freqs = 10000.0**(-torch.arange(0,
                                    self.rotary_dim, 2).float() / self.rotary_dim)
            if adaptive_attention:  # Adaptive RoPE frequencies
                self.rotary_freqs = nn.Parameter(base_freqs)
            else:  # Fixed RoPE frequencies
                self.register_buffer(
                    'rotary_freqs', base_freqs, persistent=False)
        self.interference_strength = nn.Parameter(torch.ones(1))
        if adaptive_attention:  # Adaptive temperature
            self.attention_temperature = nn.Parameter(torch.ones(1))
        else:  # Fixed temperature
            self.register_buffer('attention_temperature',
                                 torch.ones(1), persistent=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.saved_attn_weights = None  # For potential analysis

    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Positional Embedding."""
        if not self.use_rotary or self.rotary_dim <= 0:
            return q, k
        device = q.device
        dim_rotary = self.rotary_dim
        q_rot, q_pass = q[..., :dim_rotary], q[..., dim_rotary:]
        k_rot, k_pass = k[..., :dim_rotary], k[..., dim_rotary:]
        position = torch.arange(seq_len, device=device).float()
        freqs = self.rotary_freqs.to(device)  # Use learned or fixed freqs
        emb = torch.outer(position, freqs)  # [S, rotary_dim/2]
        # [1, 1, S, rotary_dim/2]
        cos_emb = torch.cos(emb).unsqueeze(0).unsqueeze(1)
        # [1, 1, S, rotary_dim/2]
        sin_emb = torch.sin(emb).unsqueeze(0).unsqueeze(1)
        # Reshape features for rotation
        q_rot = q_rot.reshape(*q_rot.shape[:-1], -1, 2)
        k_rot = k_rot.reshape(*k_rot.shape[:-1], -1, 2)
        # [..., Dr/2, 1]
        cos, sin = cos_emb.unsqueeze(-1), sin_emb.unsqueeze(-1)
        # Apply rotation (complex multiplication logic)
        q_rot_out = torch.zeros_like(q_rot)
        k_rot_out = torch.zeros_like(k_rot)
        q_rot_out[..., 0] = q_rot[..., 0] * cos[..., 0] - \
            q_rot[..., 1] * sin[..., 0]
        q_rot_out[..., 1] = q_rot[..., 1] * cos[..., 0] + \
            q_rot[..., 0] * sin[..., 0]
        k_rot_out[..., 0] = k_rot[..., 0] * cos[..., 0] - \
            k_rot[..., 1] * sin[..., 0]
        k_rot_out[..., 1] = k_rot[..., 1] * cos[..., 0] + \
            k_rot[..., 0] * sin[..., 0]
        # Reshape back and concatenate
        q_rot_out, k_rot_out = q_rot_out.flatten(
            start_dim=-2), k_rot_out.flatten(start_dim=-2)
        q_out = torch.cat([q_rot_out, q_pass], dim=-1)
        k_out = torch.cat([k_rot_out, k_pass], dim=-1)
        return q_out, k_out

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tuple of (real, imag) tensors, shape [B, S, D]
            attention_mask: Optional padding mask, shape [B, S] or broadcastable. True indicates position should be MASKED (ignored).
        """
        real, imag = x
        batch_size, seq_len, _ = real.shape
        device = real.device
        # 1. Projections
        q_r = self.q_real(real).view(
            batch_size, seq_len, self.heads, self.head_dim)
        k_r = self.k_real(real).view(
            batch_size, seq_len, self.heads, self.head_dim)
        v_r = self.v_real(real).view(
            batch_size, seq_len, self.heads, self.head_dim)
        q_i = self.q_imag(imag).view(
            batch_size, seq_len, self.heads, self.head_dim)
        k_i = self.k_imag(imag).view(
            batch_size, seq_len, self.heads, self.head_dim)
        v_i = self.v_imag(imag).view(
            batch_size, seq_len, self.heads, self.head_dim)
        # 2. Noise (skipped during eval)
        # 3. RoPE (apply after projection, transpose for RoPE format)
        q_r, k_r = self._apply_rotary_pos_emb(
            q_r.transpose(1, 2), k_r.transpose(1, 2), seq_len)
        q_i, k_i = self._apply_rotary_pos_emb(
            q_i.transpose(1, 2), k_i.transpose(1, 2), seq_len)
        v_r, v_i = v_r.transpose(1, 2), v_i.transpose(
            1, 2)  # Transpose V too [B, h, S, d]
        # 4. Entanglement
        entanglement_matrix_eff = self.entanglement_matrix.to(device)
        if self.use_entanglement:
            # einsum for head mixing
            q_r = torch.einsum("bhsd,hx->bxsd", q_r, entanglement_matrix_eff)
            q_i = torch.einsum("bhsd,hx->bxsd", q_i, entanglement_matrix_eff)
            k_r = torch.einsum("bhsd,hx->bxsd", k_r, entanglement_matrix_eff)
            k_i = torch.einsum("bhsd,hx->bxsd", k_i, entanglement_matrix_eff)
        # 5. Phase Shifts
        # [1, h, 1, d]
        phase_cos = torch.cos(
            self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device)
        # [1, h, 1, d]
        phase_sin = torch.sin(
            self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device)
        q_r_shifted = q_r * phase_cos - q_i * phase_sin
        q_i_shifted = q_r * phase_sin + q_i * phase_cos
        k_r_shifted = k_r * phase_cos - k_i * phase_sin
        k_i_shifted = k_r * phase_sin + k_i * phase_cos
        q_r, q_i = q_r_shifted, q_i_shifted
        k_r, k_i = k_r_shifted, k_i_shifted
        # 6. Attention Scores (Quantum or Classical)
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.interference_type == "quantum":
            attn_r = torch.matmul(q_r, k_r.transpose(-2, -1)) + \
                torch.matmul(q_i, k_i.transpose(-2, -1))
            attn_i = torch.matmul(q_i, k_r.transpose(-2, -1)) - \
                torch.matmul(q_r, k_i.transpose(-2, -1))
            attn_r *= scale
            attn_i *= scale
            # Use magnitude for softmax
            attn_mag = torch.sqrt(attn_r**2 + attn_i**2 + 1e-6)
        else:  # classical dot-product
            attn_mag = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale

        # 7. Apply Masking (Causal + Padding)
        # Causal mask (True=masked)
        # [1, 1, S, S]
        causal_mask = torch.triu(torch.ones(
            seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1)
        final_mask = causal_mask  # Start with causal

        # Combine with padding mask (input `attention_mask`: True means MASKED position)
        if attention_mask is not None:
             # Input padding mask [B, S] -> expand to [B, 1, 1, S] for broadcasting
            if attention_mask.dim() == 2:
                 # [B, 1, 1, S]
                padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
                # Combine: True if causal OR padding
                final_mask = final_mask | padding_mask
             # Add handling for pre-expanded masks if necessary, e.g. shape [B, 1, S_q, S_k]
            elif attention_mask.dim() == 4 and attention_mask.shape[1] == 1:
                 # Assuming broadcastable, e.g. [B, 1, 1, S] or [B, 1, S, S]
                 final_mask = final_mask | attention_mask.bool()
            else:
                 logger.warning(
                     f"Unsupported padding mask shape {attention_mask.shape} in EntangledInterferenceLayer. Ignoring padding mask.")

        # Apply final mask to scores (set masked positions to -inf)
        # Ensure final_mask broadcasts correctly to attn_mag shape [B, h, S, S]
        attn_mag = attn_mag.masked_fill(final_mask, float('-inf'))

        # 8. Softmax (with temperature and strength)
        # Ensure temp > 0
        temp = torch.clamp(self.attention_temperature.to(device), min=1e-2)
        # Sigmoid for strength [0, 1]
        strength = torch.sigmoid(self.interference_strength.to(device))
        attn_weights = F.softmax((attn_mag * strength) / temp, dim=-1)
        # Apply dropout (usually off during eval)
        attn_weights = self.attn_dropout(attn_weights)
        # self.saved_attn_weights = attn_weights.detach().cpu() # Optional: save weights

        # 9. Weighted Sum (Value)
        out_r = torch.matmul(attn_weights, v_r)  # [B, h, S, d]
        out_i = torch.matmul(attn_weights, v_i)  # [B, h, S, d]
        # 10. Reshape and Project Output
        # [B, S, D]
        out_r = out_r.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        # [B, S, D]
        out_i = out_i.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out_r = self.out_real(out_r)
        out_i = self.out_imag(out_i)
        # Apply residual dropout (usually off during eval)
        out_r = self.resid_dropout(out_r)
        out_i = self.resid_dropout(out_i)
        return (out_r, out_i)


class LocalDecoder(nn.Module):
    """Decodes processed REAL patch representations back to bytes using a TransformerDecoder."""

    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight,
                        mean=0.0, std=1.0 / math.sqrt(hidden_size))
        self.memory_projection = nn.Linear(global_hidden_size, hidden_size)
        nn.init.normal_(self.memory_projection.weight, std=0.02)
        nn.init.zeros_(self.memory_projection.bias)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size *
                                                   4, dropout=dropout, batch_first=True, activation=F.gelu)
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)
        self.byte_pred = nn.Linear(hidden_size, 256)
        nn.init.normal_(self.byte_pred.weight, std=0.02)
        nn.init.zeros_(self.byte_pred.bias)
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates a causal mask (True = ignore)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt_byte_seq: Target byte sequence, shape [B, T].
            memory: Encoded context (REAL), shape [B, M, global_hidden_size].
            tgt_mask: Causal mask for target self-attention, shape [T, T]. True = ignore.
            memory_key_padding_mask: Mask for memory attention, shape [B, M]. True = ignore padding.
        """
        batch_size, tgt_len = tgt_byte_seq.size()
        device = tgt_byte_seq.device
        if tgt_len == 0:
            return torch.zeros((batch_size, 0, 256), device=device)

        tgt_embed = self.byte_embeddings(tgt_byte_seq)  # [B, T, H]
        tgt_embed = self.dropout(tgt_embed)
        projected_memory = self.memory_projection(memory)  # [B, M, H]

        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt_len, device)  # [T, T]

        # TransformerDecoder expects masks where True means ignore.
        # memory_key_padding_mask (if provided) should also have True for ignored positions.
        output = self.transformer(
            tgt=tgt_embed,
            memory=projected_memory,
            tgt_mask=tgt_mask,
            # Pass the padding mask for memory
            memory_key_padding_mask=memory_key_padding_mask
        )  # [B, T, H]
        byte_logits = self.byte_pred(output)  # [B, T, 256]
        return byte_logits

# =====================================================================
# BSFIN Model Definition (from bsfin_main.py)
# =====================================================================


class BSFINModel(nn.Module):
    """BabylonIndex Semantic Field Interference Network (BSFIN v2)."""

    def __init__(
        self,
        local_hidden_size: int = 256, complex_dim: int = 512, num_complex_layers: int = 8,
        num_complex_heads: int = 8, decoder_memory_dim: int = 1024, dropout: float = 0.15,
        context_window: int = 256, n_gram_sizes: List[int] = [3, 4], n_gram_vocab_size: int = 30000,
        sfin_noise_scale: float = 0.05, sfin_use_entanglement: bool = True, sfin_use_rotary: bool = True,
        projection_method: str = "concat"
    ):
        super().__init__()
        self.local_hidden_size = local_hidden_size
        self.complex_dim = complex_dim
        self.decoder_memory_dim = decoder_memory_dim
        # Needed for generate context slicing (though generate doesn't use it directly here)
        self.context_window = context_window
        if complex_dim % num_complex_heads != 0:
            raise ValueError(
                f"complex_dim ({complex_dim}) must be divisible by num_complex_heads ({num_complex_heads})")

        self.patcher = BabylonIndex(scales=n_gram_sizes)
        self.local_encoder = LocalEncoder(local_hidden_size, num_layers=1, num_heads=8, dropout=dropout,
                                          n_gram_sizes=n_gram_sizes, n_gram_vocab_size=n_gram_vocab_size)
        self.real_to_complex = RealToComplexProjection(
            local_hidden_size, complex_dim)
        # --- FIX: Use max_len consistent with training checkpoint ---
        # Checkpoint indicates max_len=1024 was used during training.
        # Using 2048 here caused the size mismatch error during loading.
        self.complex_pos_encoding = PositionalEncoding(
            complex_dim, max_len=1024, learnable=True)
        # --- End Fix ---
        self.complex_norm_in = ComplexLayerNorm(complex_dim, coupled=True)
        self.complex_interference_layers = nn.ModuleList([
            EntangledInterferenceLayer(complex_dim, num_complex_heads, dropout, noise_scale=sfin_noise_scale,
                                       use_entanglement=sfin_use_entanglement, use_rotary=sfin_use_rotary, adaptive_attention=True)
            for _ in range(num_complex_layers)])
        self.complex_norms_mid = nn.ModuleList(
            [ComplexLayerNorm(complex_dim, coupled=True) for _ in range(num_complex_layers)])
        self.complex_dropout = nn.Dropout(dropout)
        self.complex_to_real = ComplexToRealProjection(
            complex_dim, decoder_memory_dim, method=projection_method)
        self.local_decoder = LocalDecoder(
            local_hidden_size, decoder_memory_dim, num_layers=4, num_heads=8, dropout=dropout)

        logger.info(f"BSFIN Initialized for Inference: LocalDim={local_hidden_size}, ComplexDim={complex_dim}, ComplexLayers={num_complex_layers}, DecoderMemDim={decoder_memory_dim}, Dropout={dropout}, SFIN Noise={sfin_noise_scale}, Entangle={sfin_use_entanglement}, RoPE={sfin_use_rotary}, N-grams={n_gram_sizes}, PE MaxLen={self.complex_pos_encoding.max_len}") # Log PE max len

    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training or inference.
        During inference (via generate method), target_byte_seq is usually the same as byte_seq (autoregressive).
        """
        batch_size = byte_seq.size(0)
        device = byte_seq.device

        # --- Patching and Encoding per Sequence ---
        batch_patch_repr_list = []
        num_patches_per_item = []
        valid_batch_indices = []  # Indices of batch items that produced patches

        for i in range(batch_size):
            seq = byte_seq[i]  # [S_in]
            patches = self.patcher.create_patches(seq)  # List[Tensor[patch_len]]
            if patches:
                # Encode patches for this sequence (returns [1, num_p, local_hidden])
                real_patch_repr_single = self.local_encoder(patches)
                if real_patch_repr_single.numel() > 0 and real_patch_repr_single.size(1) > 0:  # Check num_patches > 0
                    batch_patch_repr_list.append(real_patch_repr_single)
                    num_patches_per_item.append(real_patch_repr_single.size(1))
                    valid_batch_indices.append(i)
                else: logger.debug(f"Batch item {i}: Encoding yielded empty tensor or zero patches.")
            else: logger.debug(f"Batch item {i}: Patching yielded no patches.")

        # If no sequences produced valid patches
        if not batch_patch_repr_list:
            # Use target len if available, else input len for target shape
            target_len = target_byte_seq.size(
                1) if target_byte_seq is not None else byte_seq.size(1)
            logger.warning(
                "No valid patches produced for any item in the batch. Returning zeros.")
            # Return zeros matching potential target shape
            return torch.zeros((batch_size, target_len, 256), device=device)

        # --- Pad and Stack Patch Representations ---
        max_num_patches = max(num_patches_per_item) if num_patches_per_item else 0
        if max_num_patches == 0: # Should be caught above, but safeguard
            target_len = target_byte_seq.size(
                1) if target_byte_seq is not None else byte_seq.size(1)
            logger.warning(
                "Max number of patches is zero after filtering. Returning zeros.")
            return torch.zeros((batch_size, target_len, 256), device=device)

        padded_repr_list = []
        for repr_tensor in batch_patch_repr_list:
            num_patches = repr_tensor.size(1)
            padding_size = max_num_patches - num_patches
            if padding_size > 0:
                padding = torch.zeros(
                    (1, padding_size, self.local_hidden_size), device=device)
                padded_repr = torch.cat([repr_tensor, padding], dim=1)
            else:
                padded_repr = repr_tensor
            padded_repr_list.append(padded_repr)

        # Stack into a batch tensor containing only valid items
        # [B_valid, max_num_p, local_hidden]
        real_patch_repr_batched = torch.cat(padded_repr_list, dim=0)

        # Create padding mask for the complex layers (True = MASKED)
        # [B_valid]
        num_valid_patches_tensor = torch.tensor(num_patches_per_item, device=device)
        # Mask where index >= num_valid_patches
        # [B_valid, max_num_p]
        memory_padding_mask = torch.arange(
            max_num_patches, device=device)[None, :] >= num_valid_patches_tensor[:, None]

        # --- Complex Processing ---
        complex_patch_repr = self.real_to_complex(real_patch_repr_batched)
        complex_patch_repr = self.complex_pos_encoding(complex_patch_repr)
        complex_patch_repr = self.complex_norm_in(complex_patch_repr)
        real, imag = complex_patch_repr
        for i, layer in enumerate(self.complex_interference_layers):
            real_res, imag_res = real, imag
            normed_real, normed_imag = self.complex_norms_mid[i]((real, imag))
            # Pass the padding mask (True=MASKED) to the interference layer
            # Mask shape [B_valid, max_num_p] needs to be broadcastable or handled internally by layer
            # Pass padding mask here
            out_real, out_imag = layer(
                (normed_real, normed_imag), attention_mask=memory_padding_mask)
            # Residual connection + dropout (dropout off during eval)
            real = real_res + self.complex_dropout(out_real)
            imag = imag_res + self.complex_dropout(out_imag)
        processed_complex_repr = (real, imag)

        # --- Complex -> Real Projection ---
        # [B_valid, max_num_p, decoder_mem_dim]
        processed_real_repr = self.complex_to_real(processed_complex_repr)

        # --- Decoding ---
        # Determine the target sequence for the decoder
        # For inference/generation, usually the same as the input byte_seq
        decoder_tgt_seq = target_byte_seq if target_byte_seq is not None else byte_seq

        # Select target sequences for valid batch items
        if len(valid_batch_indices) < batch_size:
            # [B_valid, S_tgt]
            valid_decoder_tgt_seq = decoder_tgt_seq[valid_batch_indices]
        else:
            valid_decoder_tgt_seq = decoder_tgt_seq  # All items were valid

        # Decoder needs memory and memory padding mask (True=MASKED)
        byte_logits_valid = self.local_decoder(
            tgt_byte_seq=valid_decoder_tgt_seq,
            memory=processed_real_repr,  # [B_valid, M, H_mem] where M=max_num_p
            # [B_valid, M] -> True where memory is padding
            memory_key_padding_mask=memory_padding_mask
            # Causal mask for tgt is handled internally by LocalDecoder
        )  # Output: [B_valid, S_tgt, 256]

        # --- Reconstruct full batch output ---
        # Initialize with zeros, matching the expected full output shape
        final_target_len = decoder_tgt_seq.size(1)
        # Ensure correct dtype
        final_byte_logits = torch.zeros(
            (batch_size, final_target_len, 256), dtype=byte_logits_valid.dtype, device=device)

        # Place valid logits back into the full tensor using advanced indexing
        if valid_batch_indices:  # Ensure there are valid indices to work with
            valid_indices_tensor = torch.tensor(
                valid_batch_indices, device=device, dtype=torch.long)
            # Robustness check: Ensure dimensions match before assignment
            if byte_logits_valid.size(0) == len(valid_indices_tensor):
                final_byte_logits[valid_indices_tensor] = byte_logits_valid
            else:
                # This case should ideally not happen if logic is correct, but good to log
                logger.error(
                    f"CRITICAL: Mismatch between valid indices count ({len(valid_indices_tensor)}) and valid logits batch size ({byte_logits_valid.size(0)}). Output may be incorrect.")
                # Handle error appropriately, e.g., return the zero tensor or raise exception
                # Returning zeros might mask the error, consider raising. For now, return zeros.
                return torch.zeros((batch_size, final_target_len, 256), dtype=byte_logits_valid.dtype, device=device)

        return final_byte_logits  # [B, S_tgt, 256]

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        """Computes cross-entropy loss with label smoothing."""
        # This function is primarily for training, but included for completeness
        # It's not directly used by the generate method.
        batch_size, seq_len, vocab_size = logits.size()
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        if torch.any(targets_flat >= vocab_size) or torch.any(targets_flat < 0):
            invalid_indices = torch.where(
                (targets_flat < 0) | (targets_flat >= vocab_size))[0]
            logger.error(
                f"Target indices out of bounds (0 <= index < {vocab_size}). Found: {targets_flat[invalid_indices[:10]]}")
            raise ValueError(
                f"Target indices exceed vocabulary size ({vocab_size})!")

        with torch.no_grad():
            true_dist = torch.zeros_like(logits_flat)
            smooth_val = smoothing / \
                (vocab_size - 1) if vocab_size > 1 else 0.0
            true_dist.fill_(smooth_val)
            true_dist.scatter_(1, targets_flat.unsqueeze(1), 1.0 - smoothing)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        # Sum over vocab dim -> [B*S]
        loss = F.kl_div(log_probs, true_dist, reduction='none').sum(dim=-1)

        if mask is not None:
            mask_flat = mask.reshape(-1)  # [B*S]
            loss = loss * mask_flat
            num_active_elements = mask_flat.sum()
            mean_loss = loss.sum(
            ) / num_active_elements if num_active_elements > 0 else torch.tensor(0.0, device=logits.device)
        else:
            mean_loss = loss.mean()

        return mean_loss

    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None) -> torch.Tensor:
        """Generates byte sequences autoregressively."""
        self.eval()
        # Get device from model parameters
        device = next(self.parameters()).device
        # Ensure seed_bytes is on the correct device
        seed_bytes = seed_bytes.to(device)

        batch_size, seed_len = seed_bytes.size()
        generated_sequence = seed_bytes.clone()
        if sampling_config is None:
            sampling_config = SamplerConfig()  # Default config
        self.patcher.reset_context()  # Reset patcher state for new generation

        for _ in tqdm(range(max_length), desc="Generating", disable=batch_size > 1):
            # Prepare inputs for the model's forward pass
            # Use the currently generated sequence as both input and target context
            current_context = generated_sequence
            # Optional: Limit context length fed to model if it exceeds positional encoding limits?
            # context_to_feed = current_context[:, -self.complex_pos_encoding.max_len:]
            # Using full context as PositionalEncoding handles longer sequences by reuse/truncation.

            # Get logits for the next token prediction using the full model forward pass
            # Pass the current sequence as both byte_seq and target_byte_seq
            logits_all = self(
                byte_seq=current_context, target_byte_seq=current_context)  # [B, current_len, 256]

            if logits_all.shape[1] == 0:  # Check if forward pass failed (e.g., no patches)
                logger.warning(
                    "Generation stopped: Model returned empty logits (sequence length 0).")
                break
            if logits_all.shape[1] != current_context.shape[1]:
                 logger.warning(
                    f"Generation warning: Logits length ({logits_all.shape[1]}) != context length ({current_context.shape[1]}). Using last available logit.")
                 if logits_all.shape[1] == 0: break # Cannot proceed if truly empty

            # Get the logits for the very last token prediction
            # Use index -1 safely, handles potential length mismatch warning above
            next_byte_logits = logits_all[:, -1, :]  # [B, 256]

            # --- Sampling Strategy ---
            if temperature <= 0:  # Greedy decoding
                next_byte_indices = torch.argmax(
                    next_byte_logits, dim=-1)  # [B]
            else:  # Sampling with temperature and entropy-based strategy
                scaled_logits = next_byte_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)  # [B, 256]
                # [B]
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)

                next_byte_indices = torch.zeros(
                    batch_size, dtype=torch.long, device=device)
                for i in range(batch_size):  # Apply sampling per batch item
                    current_entropy = entropy[i].item()
                    current_probs = probs[i]  # [256]

                    if current_entropy < sampling_config.low_entropy_threshold:  # Low entropy -> Greedy
                        next_byte_idx = torch.argmax(current_probs)
                    elif current_entropy < sampling_config.medium_entropy_threshold:  # Medium entropy -> Top-k
                        top_k = 10
                        actual_k = min(top_k, 256)
                        top_k_probs, top_k_indices = torch.topk(
                            current_probs, k=actual_k)
                        # Renormalize
                        top_k_probs = top_k_probs / (top_k_probs.sum() + 1e-9)
                        sampled_relative_idx = torch.multinomial(
                            top_k_probs, num_samples=1).squeeze(-1)
                        next_byte_idx = top_k_indices[sampled_relative_idx]
                    else:  # High entropy -> Full sampling
                        next_byte_idx = torch.multinomial(
                            current_probs, num_samples=1).squeeze(-1)

                    next_byte_indices[i] = next_byte_idx

            # Append the chosen next byte to the sequence
            generated_sequence = torch.cat(
                [generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)

        return generated_sequence

# =====================================================================
# Inference Functions (Adapted from inference.py)
# =====================================================================


def run_inference(args):
    """Loads the BSFIN model and runs standard inference."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")

    # IMPORTANT: Ensure model architecture arguments match the loaded checkpoint!
    logger.warning("Ensure that the provided model architecture arguments (--local_hidden_size, --complex_dim, etc.) "
                   "match the parameters used to train the loaded checkpoint for correct results.")

    # Initialize BSFINModel using arguments
    logger.info("Initializing BSFINModel for inference...")
    model_config = {
        "local_hidden_size": args.local_hidden_size, "complex_dim": args.complex_dim,
        "num_complex_layers": args.num_complex_layers, "num_complex_heads": args.num_complex_heads,
        "decoder_memory_dim": args.decoder_memory_dim, "dropout": args.dropout,  # Use dropout arg (default 0 for inference)
        "context_window": args.context_window, "n_gram_sizes": args.n_gram_sizes,
        "n_gram_vocab_size": args.n_gram_vocab_size, "sfin_noise_scale": args.sfin_noise_scale, # Default 0 for inference
        "sfin_use_entanglement": not args.no_entanglement, "sfin_use_rotary": not args.no_rope,
        "projection_method": args.projection_method
    }
    logger.info(f"Attempting to load model with config: {model_config}")
    try:
        model = BSFINModel(**model_config)
    except ValueError as e:
        logger.error(
            f"Model initialization error: {e}. Check hyperparameters (e.g., complex_dim divisibility by heads).")
        return

    # Load checkpoint
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return

    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    try:
        # Added weights_only=True for security, adjust if complex objects are saved
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        # Handle potential DataParallel prefix 'module.' and check if checkpoint is just state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
             state_dict = checkpoint['model_state_dict']
             logger.info(f"Loading model state dict from epoch {checkpoint.get('epoch', 'N/A')}, global step {checkpoint.get('global_step', 'N/A')}.")
        elif isinstance(checkpoint, dict): # Assume the dict IS the state_dict
             state_dict = checkpoint
             logger.info("Loading model state dict directly from checkpoint file.")
        else:
             logger.error(f"Unrecognized checkpoint format in {args.checkpoint_path}. Expected a dict.")
             return

        new_state_dict = {}
        model_keys_prefix = 'module.'
        for k, v in state_dict.items():
            name = k[len(model_keys_prefix):] if k.startswith(model_keys_prefix) else k  # remove `module.` prefix
            new_state_dict[name] = v

        # Load the state dict (allow missing/unexpected keys for flexibility)
        incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
        if incompatible_keys.missing_keys:
            logger.warning(
                f"Missing keys when loading state_dict: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading state_dict: {incompatible_keys.unexpected_keys}")

        logger.info(f"Model state dict loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)
        return

    model.eval()  # Set model to evaluation mode
    model.to(device)
    logger.info("Model loaded and set to evaluation mode.")

    tokenizer = ByteTokenizer()  # Use the updated tokenizer

    # Prepare input text
    input_text = args.input_text
    logger.info(f"Input text: '{input_text}'")
    encoded_bytes = tokenizer.encode(input_text)
    # Add batch dimension
    input_bytes = torch.tensor(
        encoded_bytes, dtype=torch.long).unsqueeze(0).to(device)
    logger.info(
        f"Input bytes (shape {input_bytes.shape}): {input_bytes.tolist()}")

    # Configure sampling
    sampling_config = SamplerConfig(
        low_entropy_threshold=args.low_entropy,
        medium_entropy_threshold=args.medium_entropy,
        high_entropy_threshold=args.high_entropy
    )
    logger.info(
        f"Sampler Config: Low={sampling_config.low_entropy_threshold}, Medium={sampling_config.medium_entropy_threshold}, High={sampling_config.high_entropy_threshold}")
    logger.info(
        f"Generation settings: Max Length={args.max_length}, Temperature={args.temperature}")

    # Generate text
    logger.info("Starting generation...")
    try:
        # Use AMP if enabled
        with torch.no_grad(), amp.autocast(device_type=device.type, enabled=not args.no_amp):
            generated_bytes_tensor = model.generate(
                seed_bytes=input_bytes,
                max_length=args.max_length,
                temperature=args.temperature,
                sampling_config=sampling_config
            )
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        return

    # Convert bytes back to text
    generated_bytes_list = generated_bytes_tensor[0].cpu().numpy().tolist()
    logger.info(
        f"Generated bytes ({len(generated_bytes_list)}): {generated_bytes_list}")

    # Decode the full sequence (seed + generated)
    generated_text = tokenizer.decode(generated_bytes_list)

    # Print only the newly generated part
    generated_part = generated_text[len(input_text):]

    print("\n--- Input ---")
    print(input_text)
    print("\n--- Generated ---")
    print(generated_part)
    print("\n--- Full Output ---")
    print(generated_text)
    logging.info("Inference complete.")


def interactive_inference(args):
    """Loads the BSFIN model and runs interactive inference."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")

    # IMPORTANT: Ensure model architecture arguments match the loaded checkpoint!
    logger.warning("Ensure that the provided model architecture arguments (--local_hidden_size, --complex_dim, etc.) "
                   "match the parameters used to train the loaded checkpoint for correct results.")


    # Initialize BSFINModel using arguments
    logger.info("Initializing BSFINModel for interactive inference...")
    model_config = {
        "local_hidden_size": args.local_hidden_size, "complex_dim": args.complex_dim,
        "num_complex_layers": args.num_complex_layers, "num_complex_heads": args.num_complex_heads,
        "decoder_memory_dim": args.decoder_memory_dim, "dropout": args.dropout,
        "context_window": args.context_window, "n_gram_sizes": args.n_gram_sizes,
        "n_gram_vocab_size": args.n_gram_vocab_size, "sfin_noise_scale": args.sfin_noise_scale,
        "sfin_use_entanglement": not args.no_entanglement, "sfin_use_rotary": not args.no_rope,
        "projection_method": args.projection_method
    }
    logger.info(f"Attempting to load model with config: {model_config}")
    try:
        model = BSFINModel(**model_config)
    except ValueError as e:
        logger.error(f"Model initialization error: {e}. Check hyperparameters.")
        return

    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return

    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    try:
        # Added weights_only=True for security, adjust if complex objects are saved
        # Check if file exists before loading
        if not os.path.isfile(args.checkpoint_path):
             logger.error(f"Checkpoint path is not a file: {args.checkpoint_path}")
             return
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True) # Use weights_only=True

        # Handle different checkpoint structures (dict with 'model_state_dict' vs just state_dict)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
             state_dict = checkpoint['model_state_dict']
             logger.info(f"Loading model state dict from epoch {checkpoint.get('epoch', 'N/A')}, global step {checkpoint.get('global_step', 'N/A')}.")
        elif isinstance(checkpoint, dict): # Assume the dict IS the state_dict
             state_dict = checkpoint
             logger.info("Loading model state dict directly from checkpoint file (assumed format).")
        else:
             # If weights_only=True returns only tensors, handle that case if necessary
             # For now, assume it's a dict or error out
             logger.error(f"Unrecognized checkpoint format in {args.checkpoint_path}. Expected a dict.")
             return


        new_state_dict = {}
        model_keys_prefix = 'module.'
        for k, v in state_dict.items():
            name = k[len(model_keys_prefix):] if k.startswith(model_keys_prefix) else k
            new_state_dict[name] = v

        # Load the state dict (allow missing/unexpected keys for flexibility, but strict=False might hide real issues)
        # Keep strict=False as shape mismatch was the issue, other mismatches might be tolerable
        incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
        if incompatible_keys.missing_keys:
            logger.warning(
                f"Missing keys when loading state_dict: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading state_dict: {incompatible_keys.unexpected_keys}")
        logger.info(f"Model state dict loaded successfully.")

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}", exc_info=True)
        return

    model.eval()
    model.to(device)
    logger.info("Model ready for interaction.")

    tokenizer = ByteTokenizer()

    # Configure sampling
    sampling_config = SamplerConfig(
        low_entropy_threshold=args.low_entropy,
        medium_entropy_threshold=args.medium_entropy,
        high_entropy_threshold=args.high_entropy
    )
    logger.info(
        f"Sampler Config: Low={sampling_config.low_entropy_threshold}, Medium={sampling_config.medium_entropy_threshold}, High={sampling_config.high_entropy_threshold}")
    logger.info(
        f"Generation settings: Max New Bytes={args.max_length}, Temperature={args.temperature}")
    print("\nSFIN Interactive Mode (BSFINModel)")
    print("Enter your text prompt below. Type 'quit' or 'exit' to end.")
    print("---")

    while True:
        try:
            input_text = input("> ")
            if input_text.lower() in ["quit", "exit"]:
                break
            if not input_text:
                continue

            encoded_bytes = tokenizer.encode(input_text)
            input_bytes = torch.tensor(
                encoded_bytes, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad(), amp.autocast(device_type=device.type, enabled=not args.no_amp):
                generated_bytes_tensor = model.generate(
                    seed_bytes=input_bytes,
                    max_length=args.max_length,  # Max *new* bytes
                    temperature=args.temperature,
                    sampling_config=sampling_config
                )

            generated_bytes_list = generated_bytes_tensor[0].cpu().numpy().tolist()
            full_text = tokenizer.decode(generated_bytes_list)
            generated_part = full_text[len(input_text):]

            print(f"\nModel: {generated_part.strip()}")
            print("---")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(
                f"An error occurred during generation: {e}", exc_info=True)
            print("An error occurred. Please try again.")
            print("---")

    logging.info("Interactive inference finished.")
    print("Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFIN Inference Script (BSFINModel)")
    subparsers = parser.add_subparsers(
        dest='command', help='Choose inference mode', required=True)

    # --- Common Arguments ---
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
    common_parser.add_argument(
        "--cpu", action="store_true", help="Force inference on CPU")
    common_parser.add_argument(
        "--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    # Sampling Args
    common_parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum number of NEW bytes to generate")
    common_parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature (<=0 for greedy)")
    common_parser.add_argument(
        "--low_entropy", type=float, default=0.3, help="Threshold for low entropy (greedy sampling)")
    common_parser.add_argument(
        "--medium_entropy", type=float, default=1.2, help="Threshold for medium entropy (top-k sampling)")
    common_parser.add_argument(
        "--high_entropy", type=float, default=2.5, help="Threshold for high entropy (full sampling)")
    # Model Architecture Args (match bsfin_main.py defaults or allow override)
    # Defaults should ideally match the training script's defaults.
    common_parser.add_argument("--local_hidden_size", type=int, default=256,
                               help="Hidden size for LocalEncoder/Decoder (MUST match checkpoint)")
    common_parser.add_argument("--complex_dim", type=int, default=512,
                               help="Dimension for complex layers (MUST match checkpoint)")
    common_parser.add_argument("--num_complex_layers", type=int,
                               default=6, help="Number of SFIN-like layers (MUST match checkpoint)")
    common_parser.add_argument("--num_complex_heads", type=int, default=8,
                               help="Number of heads in complex layers (MUST match checkpoint)")
    common_parser.add_argument("--decoder_memory_dim", type=int, default=768,
                               help="Real dim projected from complex stack for decoder memory (MUST match checkpoint)")
    common_parser.add_argument("--context_window", type=int, default=256,
                               help="Context window size (informational, model's internal limit may differ)")
    common_parser.add_argument("--n_gram_sizes", type=int, nargs='+',
                               default=[3, 4], help="N-gram sizes for patcher/encoder (MUST match checkpoint)")
    common_parser.add_argument("--n_gram_vocab_size", type=int, default=30000,
                               help="Vocab size for N-gram hashing (MUST match checkpoint)")
    common_parser.add_argument("--sfin_noise_scale", type=float, default=0.0,
                               help="Noise scale in SFIN layers (default 0 for inference)")
    common_parser.add_argument("--no_entanglement", action="store_true",
                               help="Disable head entanglement in SFIN layers (MUST match checkpoint)")
    common_parser.add_argument("--no_rope", action="store_true",
                               help="Disable Rotary Positional Embeddings in SFIN layers (MUST match checkpoint)")
    common_parser.add_argument("--projection_method", type=str, default="concat", choices=[
                               "concat", "magnitude"], help="Method for Complex->Real projection (MUST match checkpoint)")
    common_parser.add_argument("--dropout", type=float, default=0.0,
                               help="Dropout rate (default 0 for inference)")

    # Standard inference parser
    standard_parser = subparsers.add_parser(
        'standard', help='Standard single-prompt inference', parents=[common_parser])
    standard_parser.add_argument(
        "--input_text", type=str, required=True, help="Seed text for generation")
    standard_parser.set_defaults(func=run_inference)

    # Interactive inference parser
    interactive_parser = subparsers.add_parser(
        'interactive', help='Interactive chat-like inference', parents=[common_parser])
    # Override max_length default for interactive mode
    interactive_parser.set_defaults(func=interactive_inference, max_length=150)

    args = parser.parse_args()

    # Add a check to ensure architecture args match checkpoint (manual for now)
    logger.warning("Reminder: Ensure model architecture arguments match the training parameters of the loaded checkpoint.")

    if hasattr(args, 'func'):
        args.func(args)
    else:
        # Should not happen due to `required=True` in subparsers
        parser.print_help()