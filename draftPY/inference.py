#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bytropix - Inference Script (Aligned with main.py)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from dataclasses import dataclass
import itertools
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =====================================================================
# Data Structures and Configuration Classes (from main.py)
# =====================================================================

@dataclass
class SamplerConfig:
    """Configuration for entropy-based sampling."""
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# =====================================================================
# ByteTokenizer (from main.py)
# =====================================================================

class ByteTokenizer:
    """Simple tokenizer for byte-level processing."""

    def encode(self, text: str) -> List[int]:
        """Convert text to bytes."""
        return [b for b in text.encode('utf-8')]

    def decode(self, byte_sequence: Iterable[int]) -> str:
        """Convert bytes back to text."""
        # Ensure byte_sequence contains integers
        valid_bytes = [b for b in byte_sequence if isinstance(b, int) and 0 <= b <= 255]
        return bytes(valid_bytes).decode('utf-8', errors='replace')

# =====================================================================
# Babylon Index (from main.py)
# =====================================================================

class BabylonIndex:
    """Entropy-based index for byte sequence analysis with efficient window management."""

    def __init__(
        self,
        scales: List[int] = [3, 5, 7],
        max_cache_size: int = 100000,
        min_entropy_threshold: float = 0.3
    ):
        self.scales = scales
        self.hash_index = {}  # Maps hash values to window positions
        self.entropy_cache = {}  # Cache for entropy values
        self.max_cache_size = max_cache_size
        self.min_entropy_threshold = min_entropy_threshold

        # Pre-compute base powers for efficient hashing
        self.base = 256
        self.mod = 2**32
        self.base_powers = {
            scale: [pow(self.base, i, self.mod) for i in range(scale)]
            for scale in scales
        }

    def _clean_cache(self):
        """Clean entropy cache if it exceeds max size."""
        if len(self.entropy_cache) > self.max_cache_size:
            # Remove 20% of oldest entries
            remove_count = self.max_cache_size // 5
            old_keys = list(self.entropy_cache.keys())[:remove_count]
            for k in old_keys:
                if k in self.entropy_cache: # Check if key still exists
                    del self.entropy_cache[k]

    def _is_valid_utf8_boundary(self, byte_seq: List[int], boundary: int) -> bool:
        """Check if a boundary in the byte sequence does not split a multi-byte UTF-8 character."""
        if boundary == 0 or boundary >= len(byte_seq):
            return True

        # Check the byte at the boundary to see if it's a continuation byte
        if 0x80 <= byte_seq[boundary] <= 0xBF:
            num_continuation_bytes = 0
            pos = boundary - 1
            while pos >= 0 and 0x80 <= byte_seq[pos] <= 0xBF:
                num_continuation_bytes += 1
                pos -= 1

            if pos < 0: return False # Invalid UTF-8

            first_byte = byte_seq[pos]
            if first_byte >> 5 == 0b110: expected_continuations = 1
            elif first_byte >> 4 == 0b1110: expected_continuations = 2
            elif first_byte >> 3 == 0b11110: expected_continuations = 3
            else: return True # ASCII or invalid start byte

            return num_continuation_bytes >= expected_continuations
        return True

    def rolling_hash(self, byte_sequence: List[int], scale: int) -> List[Tuple[int, List[int], int]]:
        """Compute rolling hashes with corresponding windows and start index."""
        results = []
        byte_len = len(byte_sequence)
        if byte_len < scale: return results

        window = byte_sequence[:scale]
        hash_val = 0
        for i in range(scale):
            hash_val = (hash_val + window[i] * self.base_powers[scale][scale - 1 - i]) % self.mod
        results.append((hash_val, window.copy(), 0))
        self.hash_index[hash_val] = (0, scale)

        power_scale_minus_1 = self.base_powers[scale][-1]
        for i in range(1, byte_len - scale + 1):
            left_byte = byte_sequence[i - 1]
            new_byte = byte_sequence[i + scale - 1]
            hash_val = (hash_val - left_byte * power_scale_minus_1) % self.mod
            hash_val = (hash_val * self.base + new_byte) % self.mod
            hash_val = (hash_val + self.mod) % self.mod
            window = byte_sequence[i:i + scale]
            results.append((hash_val, window.copy(), i))
            self.hash_index[hash_val] = (i, i + scale)
        return results

    def compute_entropy(self, byte_window: np.ndarray) -> float:
        """Compute Shannon entropy of byte window."""
        if byte_window.size == 0: return 0.0
        byte_counts = np.bincount(byte_window, minlength=256)
        total_bytes = byte_counts.sum()
        if total_bytes == 0: return 0.0
        probs = byte_counts[byte_counts > 0] / total_bytes
        return float(-np.sum(probs * np.log2(probs + 1e-9)))

    def get_window_features(self, window: List[int]) -> Dict[str, float]:
        """Extract additional features from byte window."""
        if not window: return {'mean': 0.0, 'std': 0.0, 'unique_ratio': 0.0, 'max_run': 0}
        arr = np.array(window)
        unique_count = len(np.unique(arr))
        max_run = max((len(list(g)) for _, g in itertools.groupby(window)), default=0) if window else 0
        return {
            'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
            'unique_ratio': unique_count / len(arr) if len(arr) > 0 else 0.0, 'max_run': max_run
        }

    def prioritize(self, byte_sequence: List[int]) -> List[Tuple[int, int, Dict[str, float]]]:
        """Prioritize sequence regions based on entropy and features. Returns (start_index, hash_val, metrics)."""
        self._clean_cache()
        all_scores = []
        for scale in self.scales:
            hash_windows = self.rolling_hash(byte_sequence, scale)
            for hash_val, window, start_index in hash_windows:
                cache_key = (hash_val, scale)
                if cache_key in self.entropy_cache:
                    metrics = self.entropy_cache[cache_key]
                else:
                    window_arr = np.array(window)
                    entropy = self.compute_entropy(window_arr)
                    if entropy > self.min_entropy_threshold:
                        features = self.get_window_features(window)
                        metrics = {'entropy': entropy, 'scale': float(scale), **features}
                        self.entropy_cache[cache_key] = metrics
                        all_scores.append((start_index, hash_val, metrics))

        def score_window(metrics):
            scale_score = (1.0 / metrics['scale']) * 0.2 if metrics.get('scale', 0) > 0 else 0
            run_penalty = (1.0 / (1.0 + metrics.get('max_run', 0))) * 0.1
            return (metrics.get('entropy', 0) * 0.4 + metrics.get('unique_ratio', 0) * 0.3 +
                    scale_score + run_penalty)

        all_scores.sort(key=lambda item: score_window(item[2]), reverse=True)
        return all_scores

    def get_window_position(self, hash_val: int) -> Optional[Tuple[int, int]]:
        """Get the position (start, end) of a window from its hash value."""
        return self.hash_index.get(hash_val)

    def find_patch_boundaries(self, byte_seq: torch.Tensor) -> List[int]:
        """Find patch boundaries ensuring no multi-byte UTF-8 characters are split."""
        if byte_seq.numel() == 0: return []
        if byte_seq.dim() > 1: byte_seq_np = byte_seq[0].cpu().numpy().tolist()
        else: byte_seq_np = byte_seq.cpu().numpy().tolist()
        seq_len = len(byte_seq_np)
        if seq_len == 0: return []

        prioritized_windows = self.prioritize(byte_seq_np)
        selected_boundaries = set()
        for start_index, hash_val, metrics in prioritized_windows:
             pos = self.get_window_position(hash_val)
             if pos:
                 boundary_candidate = pos[0]
                 if self._is_valid_utf8_boundary(byte_seq_np, boundary_candidate):
                     selected_boundaries.add(boundary_candidate)

        final_boundaries = sorted([b for b in selected_boundaries if 0 < b < seq_len])
        final_boundaries = [0] + final_boundaries
        if seq_len not in final_boundaries: final_boundaries.append(seq_len)
        final_boundaries = sorted(list(set(final_boundaries)))

        min_patch_size = 5
        filtered_boundaries = [final_boundaries[0]]
        for i in range(1, len(final_boundaries)):
            if final_boundaries[i] - filtered_boundaries[-1] >= min_patch_size:
                filtered_boundaries.append(final_boundaries[i])
        if seq_len not in filtered_boundaries and seq_len in final_boundaries:
             if filtered_boundaries[-1] != seq_len: filtered_boundaries.append(seq_len)
        final_boundaries = sorted(list(set(filtered_boundaries)))
        return final_boundaries

    def create_patches(self, byte_seq: torch.Tensor) -> List[torch.Tensor]:
        """Convert byte sequence into patches based on entropy boundaries."""
        if byte_seq.numel() == 0: return []
        boundaries = self.find_patch_boundaries(byte_seq)
        patches = []
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i+1]
            if end_idx > start_idx:
                patch = byte_seq[:, start_idx:end_idx]
                if patch.size(1) > 0: patches.append(patch)
        return patches

    @torch.no_grad()
    def reset_context(self):
        """Reset context when starting new document/segment."""
        self.hash_index = {}
        self.entropy_cache = {}

# =====================================================================
# Model Architecture (Copied from main.py)
# =====================================================================

class CrossAttentionBlock(nn.Module):
    """Cross-attention block that properly mixes byte and patch information."""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None: nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.out_proj.bias is not None: nn.init.zeros_(self.out_proj.bias)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_queries, _ = queries.size()
        seq_len = keys_values.size(1)
        queries_norm = self.norm(queries)
        keys_values_norm = self.norm(keys_values)
        q = self.qkv(queries_norm)[:, :, :self.hidden_size]
        kv = self.qkv(keys_values_norm)
        k = kv[:, :, self.hidden_size:2*self.hidden_size]
        v = kv[:, :, 2*self.hidden_size:]
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        output = self.dropout(output)
        output = queries + output
        return output

class GlobalLatentTransformer(nn.Module):
    """Large global transformer that processes patch representations."""
    def __init__(self, hidden_size: int = 1024, num_layers: int = 16, num_heads: int = 32, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_size, eps=1e-6),
                'attention': nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True),
                'norm2': nn.LayerNorm(hidden_size, eps=1e-6),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout))})
            for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        causal_mask = self.create_causal_mask(seq_len, device)
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        for layer_module in self.layers:
            normed_x = layer_module['norm1'](x)
            attn_output, _ = layer_module['attention'](
                query=normed_x, key=normed_x, value=normed_x,
                attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + attn_output
            normed_x = layer_module['norm2'](x)
            mlp_output = layer_module['mlp'](normed_x)
            x = x + mlp_output
        return self.final_norm(x)

class LocalEncoder(nn.Module):
    """Local encoder that efficiently maps bytes to patches."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 1, num_heads: int = 8,
                 window_size: int = 512, dropout: float = 0.1, n_gram_sizes: List[int] = [3, 4, 5],
                 n_gram_vocab_size: int = 30000):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0: raise ValueError(f"LE hidden_size must be positive int, got {hidden_size}")
        self.hidden_size = hidden_size
        self.byte_embeddings = nn.Embedding(256, self.hidden_size)
        self.n_gram_embeddings = nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, self.hidden_size) for n in n_gram_sizes})
        self.n_gram_sizes = n_gram_sizes
        self.window_size = window_size
        self.n_gram_vocab_size = n_gram_vocab_size
        self.transformer = nn.ModuleList([
             nn.TransformerEncoderLayer(self.hidden_size, num_heads, self.hidden_size * 4, dropout, batch_first=True)
             for _ in range(num_layers)])
        self.cross_attention = CrossAttentionBlock(self.hidden_size, num_heads, dropout)
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
         nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
         for embed in self.n_gram_embeddings.values(): nn.init.normal_(embed.weight, mean=0.0, std=0.02)

    def compute_n_gram_hashes(self, byte_seq: torch.Tensor) -> Dict[int, torch.Tensor]:
        batch_size, seq_len = byte_seq.size(); device = byte_seq.device; hashes = {}
        base = 31; mod = self.n_gram_vocab_size
        for n in self.n_gram_sizes:
            if seq_len < n: continue
            powers = torch.tensor([pow(base, n - 1 - i, mod) for i in range(n)], dtype=torch.long, device=device)
            first_window = byte_seq[:, :n]; current_hash = (first_window * powers).sum(dim=1) % mod
            n_gram_hashes = [current_hash]
            power_n_minus_1 = pow(base, n - 1, mod)
            for i in range(1, seq_len - n + 1):
                 hash_update = current_hash - (byte_seq[:, i - 1] * power_n_minus_1)
                 hash_update = (hash_update * base + byte_seq[:, i + n - 1]) % mod
                 current_hash = (hash_update + mod) % mod
                 n_gram_hashes.append(current_hash)
            hashes[n] = torch.stack(n_gram_hashes, dim=1)
        return hashes

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, byte_seq: torch.Tensor, patch_boundaries: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = byte_seq.size(); device = byte_seq.device
        if seq_len == 0:
            empty_bytes = torch.zeros((batch_size, 0, self.hidden_size), dtype=torch.float, device=device)
            empty_patches = torch.zeros((batch_size, 0, self.hidden_size), dtype=torch.float, device=device)
            return empty_bytes, empty_patches
        x = self.byte_embeddings(byte_seq)
        n_gram_hashes = self.compute_n_gram_hashes(byte_seq)
        n_gram_scale = 1.0 / (len(self.n_gram_sizes) + 1)
        for n, hash_vals in n_gram_hashes.items():
            valid_length = hash_vals.size(1)
            if valid_length > 0:
                n_gram_embeds = self.n_gram_embeddings[f'n{n}'](hash_vals) * n_gram_scale
                start_pos = n - 1; end_pos = start_pos + valid_length
                if end_pos <= seq_len: x[:, start_pos:end_pos] += n_gram_embeds
                else:
                     valid_embed_len = seq_len - start_pos
                     if valid_embed_len > 0: x[:, start_pos:] += n_gram_embeds[:, :valid_embed_len]
        x = self.dropout(x)
        causal_mask = self.create_causal_mask(seq_len, device)
        for layer in self.transformer: x = layer(src=x, src_mask=causal_mask, src_key_padding_mask=None)
        patches = []
        for i in range(len(patch_boundaries) - 1):
            start_idx, end_idx = patch_boundaries[i], patch_boundaries[i+1]
            if end_idx > start_idx:
                patch_bytes_processed = x[:, start_idx:end_idx]
                if patch_bytes_processed.size(1) > 0:
                    query = torch.mean(patch_bytes_processed, dim=1, keepdim=True)
                    patch_repr = self.cross_attention(query, patch_bytes_processed)
                    patches.append(patch_repr)
        if patches: patches_combined = torch.cat(patches, dim=1)
        else: patches_combined = torch.mean(x, dim=1, keepdim=True) if x.size(1) > 0 else \
                               torch.zeros((batch_size, 1, self.hidden_size), dtype=x.dtype, device=device)
        return self.norm(x), self.norm(patches_combined)

class LocalDecoder(nn.Module):
    """Local decoder that maps patches back to bytes (matching main.py)."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4, num_heads: int = 8,
                 dropout: float = 0.1, global_hidden_size: int = 1024):
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0: raise ValueError(f"LD hidden_size must be positive int, got {hidden_size}")
        if not isinstance(global_hidden_size, int) or global_hidden_size <= 0: raise ValueError(f"LD global_hidden_size must be positive int, got {global_hidden_size}")
        self.hidden_size = hidden_size
        self.byte_embeddings = nn.Embedding(256, self.hidden_size)
        self.initial_cross_attention = CrossAttentionBlock(self.hidden_size, num_heads, dropout)
        self.transformer = nn.ModuleList([
            nn.TransformerDecoderLayer(self.hidden_size, num_heads, self.hidden_size * 4, dropout, batch_first=True)
            for _ in range(num_layers)])
        self.memory_projection = nn.Linear(global_hidden_size, self.hidden_size)
        self.byte_pred = nn.Linear(self.hidden_size, 256)
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.memory_projection.weight, std=0.02)
        if self.memory_projection.bias is not None: nn.init.zeros_(self.memory_projection.bias)
        nn.init.normal_(self.byte_pred.weight, std=0.02)
        if self.byte_pred.bias is not None: nn.init.zeros_(self.byte_pred.bias)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, tgt_len = tgt_byte_seq.size(); device = tgt_byte_seq.device
        # 1. Embed target
        tgt_embed = self.byte_embeddings(tgt_byte_seq)
        tgt_embed = self.dropout(tgt_embed)
        # 2. Project memory (received from BLTModel, has global_hidden_size)
        projected_memory = self.memory_projection(memory) # Now expects global_hidden, outputs local_hidden
        # 3. Initial cross attention (Query: target, Key/Value: projected_memory)
        x = self.initial_cross_attention(tgt_embed, projected_memory, attention_mask=memory_key_padding_mask)
        # 4. Causal mask for self-attention
        if tgt_mask is None: tgt_mask = self.create_causal_mask(tgt_len, device)
        # 5. Decoder layers
        for layer in self.transformer:
            x = layer(tgt=x, memory=projected_memory, tgt_mask=tgt_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        # 6. Final projection
        output = self.norm(x)
        byte_logits = self.byte_pred(output)
        return byte_logits

class BLTModel(nn.Module):
    """Complete Byte Latent Transformer model implementation (matching main.py)."""
    def __init__(self, local_hidden_size: int = 256, global_hidden_size: int = 1024,
                 num_local_encoder_layers: int = 1, num_global_layers: int = 16,
                 num_local_decoder_layers: int = 4, dropout: float = 0.1,
                 window_size: int = 256, n_gram_sizes: List[int] = [3, 4],
                 n_gram_vocab_size: int = 30000):
        super().__init__()
        if not isinstance(local_hidden_size, int) or local_hidden_size <= 0: raise ValueError(f"BLT local_hidden must be positive int, got {local_hidden_size}")
        if not isinstance(global_hidden_size, int) or global_hidden_size <= 0: raise ValueError(f"BLT global_hidden must be positive int, got {global_hidden_size}")
        self.local_hidden_size = local_hidden_size
        self.global_hidden_size = global_hidden_size
        self.context_size = window_size
        # Configure attention backend
        try:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch.cuda.is_available(): logging.info("Using PyTorch 2.0 scaled_dot_product_attention backend.")
            elif hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'): torch.backends.cuda.enable_mem_efficient_sdp(True); logging.info("Enabled memory-efficient SDP backend.")
            else: logging.info("Memory-efficient attention backend not available.")
        except Exception as e: logging.warning(f"Could not configure memory-efficient attention: {e}")

        self.patcher = BabylonIndex(n_gram_sizes, 100000, 0.3)
        self.local_encoder = LocalEncoder(self.local_hidden_size, num_local_encoder_layers, 8, window_size, dropout, n_gram_sizes, n_gram_vocab_size)
        self.patch_projection = nn.Sequential(
            nn.Linear(self.local_hidden_size, self.global_hidden_size),
            nn.LayerNorm(self.global_hidden_size, eps=1e-6), nn.Dropout(dropout))
        self.global_transformer = GlobalLatentTransformer(self.global_hidden_size, num_global_layers, 16, dropout) # Assuming 16 heads for global
        self.patch_deprojection = nn.Sequential( # Kept for checkpoint compatibility
            nn.Linear(self.global_hidden_size, self.local_hidden_size),
            nn.LayerNorm(self.local_hidden_size, eps=1e-6), nn.Dropout(dropout))
        self.local_decoder = LocalDecoder(self.local_hidden_size, num_local_decoder_layers, 8, dropout, self.global_hidden_size)

    def forward(self, byte_seq: torch.Tensor, return_patch_boundaries: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        batch_size, seq_len = byte_seq.size(); device = byte_seq.device
        # 1. Patching
        patch_boundaries = self.patcher.find_patch_boundaries(byte_seq[0].unsqueeze(0))
        # 2. Local Encoding
        processed_byte_repr, local_patches = self.local_encoder(byte_seq, patch_boundaries)
        patch_padding_mask = None # Assuming fixed num patches or handled elsewhere
        # 3. Project to Global
        global_patches = self.patch_projection(local_patches)
        # 4. Global Transformer
        global_context = self.global_transformer(global_patches, attention_mask=patch_padding_mask)
        # 5. Deprojection (kept for compatibility, but output not directly used by decoder now)
        deprojected_context = self.patch_deprojection(global_context)
        # 6. Local Decoding (Pass GLOBAL context as memory)
        tgt_causal_mask = self.local_decoder.create_causal_mask(seq_len, device)
        byte_logits = self.local_decoder(
            tgt_byte_seq=byte_seq,
            memory=global_context, # <<< Pass global_context (shape B, N, global_hidden)
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=patch_padding_mask
        )
        if return_patch_boundaries: return byte_logits, patch_boundaries
        return byte_logits

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.size()
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        if targets_flat.max() >= vocab_size or targets_flat.min() < 0:
            logging.error(f"Target OOB: min={targets_flat.min()}, max={targets_flat.max()}")
            targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits_flat)
            smooth_val = smoothing / max(1, vocab_size - 1)
            true_dist.fill_(smooth_val)
            true_dist.scatter_(1, targets_flat.unsqueeze(1), 1.0 - smoothing)
        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = F.kl_div(log_probs, true_dist, reduction='batchmean')
        return loss

    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None) -> torch.Tensor:
        self.eval(); device = next(self.parameters()).device
        seed_bytes = seed_bytes.to(device)
        batch_size, seed_len = seed_bytes.size()
        generated_sequence = seed_bytes.clone()
        if sampling_config is None: sampling_config = SamplerConfig()
        self.patcher.reset_context()

        for _ in range(max_length):
            current_context = generated_sequence
            logits_all = self(current_context)
            next_byte_logits = logits_all[:, -1, :]
            scaled_logits = next_byte_logits / temperature if temperature > 0 else next_byte_logits
            probs = F.softmax(scaled_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
            next_bytes = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                current_entropy = entropy[i].item(); current_probs = probs[i]
                if current_entropy < sampling_config.low_entropy_threshold:
                    next_byte_idx = torch.argmax(current_probs)
                elif current_entropy < sampling_config.medium_entropy_threshold:
                    top_k = 10; actual_k = min(top_k, current_probs.numel())
                    if actual_k <= 0: next_byte_idx = torch.argmax(current_probs)
                    else:
                        top_k_probs, top_k_indices = torch.topk(current_probs, k=actual_k)
                        top_k_probs = top_k_probs / top_k_probs.sum()
                        sampled_relative_idx = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
                        next_byte_idx = top_k_indices[sampled_relative_idx]
                else:
                    current_probs = current_probs / current_probs.sum()
                    next_byte_idx = torch.multinomial(current_probs, num_samples=1).squeeze(-1)
                next_bytes[i] = next_byte_idx
            generated_sequence = torch.cat([generated_sequence, next_bytes.unsqueeze(1)], dim=1)
        return generated_sequence

# =====================================================================
# Inference Functions (Updated Argument Handling)
# =====================================================================

def get_model_config_from_args(args):
    """Extracts model configuration from args, using defaults if args are None."""
    defaults = {
        'local_hidden_size': 256, 'global_hidden_size': 512, 'num_local_encoder_layers': 1,
        'num_global_layers': 8, 'num_local_decoder_layers': 4, 'window_size': 256,
        'n_gram_sizes': [3, 4], 'n_gram_vocab_size': 30000, 'dropout': 0.1 }
    config = {}
    for key, default_val in defaults.items():
        arg_val = getattr(args, key, None)
        config[key] = arg_val if arg_val is not None else default_val
    return config

def run_inference(args):
    """Loads the model and runs inference based on arguments."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")
    logging.info("Initializing BLTModel for inference...")
    model_config = get_model_config_from_args(args)
    try:
        model = BLTModel(**model_config)
        logging.info(f"Model initialized with: {model_config}")
    except ValueError as e:
        logging.error(f"Error initializing BLTModel: {e}", exc_info=True); return

    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint file not found: {args.checkpoint_path}"); return
    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        # Use strict=False because we know keys are missing from previous logs
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys: logging.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys: logging.warning(f"Unexpected keys: {unexpected_keys}")
        logging.info(f"Model state dict loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")
    except RuntimeError as e:
         logging.error(f"RuntimeError loading checkpoint: {e}")
         if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e): logging.error("Key mismatch details:\n" + str(e))
         return
    except Exception as e:
        logging.error(f"Unexpected error loading checkpoint: {e}", exc_info=True); return

    model.eval(); model.to(device)
    logging.info("Model loaded and set to evaluation mode.")
    tokenizer = ByteTokenizer()
    input_text = args.input_text
    logging.info(f"Input text: '{input_text}'")
    encoded_bytes = tokenizer.encode(input_text)
    input_bytes = torch.tensor(encoded_bytes, dtype=torch.long).unsqueeze(0).to(device)
    logging.info(f"Input bytes (shape {input_bytes.shape}): {input_bytes.tolist()}")
    sampling_config = SamplerConfig(args.low_entropy, args.medium_entropy)
    logging.info(f"Sampler Config: Low={sampling_config.low_entropy_threshold}, Medium={sampling_config.medium_entropy_threshold}")
    logging.info(f"Generation settings: Max New Bytes={args.max_length}, Temperature={args.temperature}")
    logging.info("Starting generation...")
    try:
        with torch.no_grad():
            generated_bytes_tensor = model.generate(
                seed_bytes=input_bytes, max_length=args.max_length,
                temperature=args.temperature, sampling_config=sampling_config)
    except Exception as gen_e:
        logging.error(f"Error during model generation: {gen_e}", exc_info=True); return

    generated_bytes_list = generated_bytes_tensor[0].cpu().numpy().tolist()
    logging.info(f"Generated bytes ({len(generated_bytes_list)} total): {generated_bytes_list}")
    generated_text = tokenizer.decode(generated_bytes_list)
    input_byte_len = len(encoded_bytes)
    generated_part = tokenizer.decode(generated_bytes_list[input_byte_len:])
    print("\n--- Input ---"); print(input_text)
    print("\n--- Generated ---"); print(generated_part)
    print("\n--- Full Output ---"); print(generated_text)
    logging.info("Inference complete.")

def interactive_inference(args):
    """Loads the model and runs interactive inference."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")
    logging.info("Initializing BLTModel for inference...")
    model_config = get_model_config_from_args(args)
    try:
        model = BLTModel(**model_config)
        logging.info(f"Model initialized with: {model_config}")
    except ValueError as e:
        logging.error(f"Error initializing BLTModel: {e}", exc_info=True); return

    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint file not found: {args.checkpoint_path}"); return
    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        # Use strict=False because we know keys are missing
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys: logging.warning(f"Missing keys: {missing_keys}") # Log missing keys
        if unexpected_keys: logging.warning(f"Unexpected keys: {unexpected_keys}")
        logging.info(f"Model state dict loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")
    except RuntimeError as e:
         logging.error(f"RuntimeError loading checkpoint: {e}")
         if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e): logging.error("Key mismatch details:\n" + str(e))
         return
    except Exception as e:
        logging.error(f"Unexpected error loading checkpoint: {e}", exc_info=True); return

    model.eval(); model.to(device)
    logging.info("Model loaded and ready for interaction.")
    tokenizer = ByteTokenizer()
    sampling_config = SamplerConfig(args.low_entropy, args.medium_entropy)
    logging.info(f"Sampler Config: Low={sampling_config.low_entropy_threshold}, Medium={sampling_config.medium_entropy_threshold}")
    logging.info(f"Generation settings: Max New Bytes={args.max_length}, Temperature={args.temperature}")
    print("\n*******************************")
    print("* Bytropix Inference Tool   *")
    print("*******************************")
    print("\nEnter prompt (or 'quit'/'exit'):"); print("---")

    while True:
        try:
            input_text = input("> ")
            if input_text.lower() in ["quit", "exit"]: break
            if not input_text: continue
            encoded_bytes = tokenizer.encode(input_text)
            input_bytes = torch.tensor(encoded_bytes, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                generated_bytes_tensor = model.generate(
                    seed_bytes=input_bytes, max_length=args.max_length,
                    temperature=args.temperature, sampling_config=sampling_config)
            generated_bytes_list = generated_bytes_tensor[0].cpu().numpy().tolist()
            input_byte_len = len(encoded_bytes)
            generated_part = tokenizer.decode(generated_bytes_list[input_byte_len:])
            print(f"\nModel: {generated_part.strip()}"); print("---")
        except KeyboardInterrupt: print("\nExiting..."); break
        except Exception as e:
            logging.error(f"An error occurred during generation: {e}", exc_info=True)
            print("An error occurred. Please try again."); print("---")

    logging.info("Interactive inference finished."); print("Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bytropix Inference Script")
    subparsers = parser.add_subparsers(dest='command', help='Choose inference mode', required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parent_parser.add_argument("--max_length", type=int, default=150, help="Max new bytes")
    parent_parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature")
    parent_parser.add_argument("--low_entropy", type=float, default=0.3, help="Low entropy threshold")
    parent_parser.add_argument("--medium_entropy", type=float, default=1.2, help="Medium entropy threshold")
    parent_parser.add_argument("--high_entropy", type=float, default=2.5, help="High entropy threshold")
    parent_parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parent_parser.add_argument("--local_hidden_size", type=int, help="Override Local hidden size")
    parent_parser.add_argument("--global_hidden_size", type=int, help="Override Global hidden size")
    parent_parser.add_argument("--num_local_encoder_layers", type=int, help="Override Local Encoder layers")
    parent_parser.add_argument("--num_global_layers", type=int, help="Override Global layers")
    parent_parser.add_argument("--num_local_decoder_layers", type=int, help="Override Local Decoder layers")
    parent_parser.add_argument("--window_size", type=int, help="Override Local Encoder window size")
    standard_parser = subparsers.add_parser('standard', help='Standard single-prompt inference', parents=[parent_parser])
    standard_parser.add_argument("--input_text", type=str, required=True, help="Seed text")
    standard_parser.set_defaults(func=run_inference)
    interactive_parser = subparsers.add_parser('interactive', help='Interactive chat-like inference', parents=[parent_parser])
    interactive_parser.set_defaults(func=interactive_inference)
    args = parser.parse_args()
    args.func(args)

