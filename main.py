# main.py

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from typing import Optional, Tuple, List, Dict, Any, Iterable, Union
import math
from dataclasses import dataclass
from torch import amp
from torch.amp import autocast  # Corrected import
import gc
import wandb
import argparse
import platform
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized
import socket
from collections import deque
import logging
import numpy as np
import random
import itertools

from EnhancedSGD import EnhancedSGD  # Ensure this is correctly implemented and accessible

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO for standard logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------------------
# Sampler Configuration
# ----------------------------

@dataclass
class SamplerConfig:
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# ----------------------------
# ByteTokenizer
# ----------------------------

class ByteTokenizer:
    def encode(self, text: str) -> List[int]:
        return [b for b in text.encode('utf-8')]
    
    def decode(self, byte_sequence: Iterable[int]) -> str:
        return bytes(byte_sequence).decode('utf-8', errors='replace')

# ----------------------------
# BabylonIndex
# ----------------------------

class BabylonIndex:
    """Enhanced index for byte sequence analysis with efficient window management."""
    
    def __init__(
        self, 
        scales: List[int] = [3, 5, 7],
        max_cache_size: int = 100000,
        min_entropy_threshold: float = 0.3
    ):
        self.scales = scales
        self.hash_index = {}  # Maps hash values to window positions
        self.entropy_cache = {}  # LRU cache for entropy values
        self.max_cache_size = max_cache_size
        self.min_entropy_threshold = min_entropy_threshold
        
        # Pre-compute base powers for efficiency
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
                del self.entropy_cache[k]

    def _is_valid_utf8_boundary(self, byte_seq: List[int], boundary: int) -> bool:
        """Check if a boundary in the byte sequence does not split a multi-byte UTF-8 character."""
        if boundary == 0 or boundary >= len(byte_seq):
            return True
        # Check the byte before the boundary to see if it's a continuation byte
        # Continuation bytes have the form 10xxxxxx
        prev_byte = byte_seq[boundary - 1]
        if 0x80 <= prev_byte <= 0xBF:
            # Find the start of the multi-byte character
            start = boundary - 1
            while start > 0 and 0x80 <= byte_seq[start - 1] <= 0xBF:
                start -= 1
            # Number of continuation bytes
            num_continuations = boundary - start
            first_byte = byte_seq[start]
            # Determine the number of bytes in this character based on first byte
            if first_byte >> 5 == 0b110:
                expected = 2
            elif first_byte >> 4 == 0b1110:
                expected = 3
            elif first_byte >> 3 == 0b11110:
                expected = 4
            else:
                return True  # Single-byte character or invalid, allow boundary
            return num_continuations < (expected - 1)
        return True  # Not a continuation byte, allow boundary

    def rolling_hash(self, byte_sequence: List[int], scale: int) -> List[Tuple[int, List[int]]]:
        """Compute rolling hashes with corresponding windows.
        
        Args:
            byte_sequence: Input bytes as list of integers
            scale: Window size
            
        Returns:
            List of tuples (hash_value, window_bytes)
        """
        results = []
        byte_len = len(byte_sequence)
        if byte_len < scale:
            return results
            
        # Initialize hash value for the first window
        window = byte_sequence[:scale]
        hash_val = 0
        for b in window:
            hash_val = (hash_val * self.base + b) % self.mod
        results.append((hash_val, window.copy()))
        
        # Rolling hash
        for i in range(1, byte_len - scale + 1):
            # Remove the leftmost byte and add the new rightmost byte
            left_byte = byte_sequence[i - 1]
            new_byte = byte_sequence[i + scale - 1]
            hash_val = (hash_val * self.base - left_byte * self.base_powers[scale][-1] + new_byte) % self.mod
            window = byte_sequence[i:i + scale]
            results.append((hash_val, window.copy()))
            
            # Update hash index for future lookups
            self.hash_index[hash_val] = (i, i + scale)
            
        return results

    def compute_entropy(self, byte_window: np.ndarray) -> float:
        """Compute Shannon entropy of byte window.
        
        Args:
            byte_window: Numpy array of bytes
            
        Returns:
            Entropy value
        """
        # Use numpy for efficient counting
        byte_counts = np.bincount(byte_window, minlength=256)
        total_bytes = byte_counts.sum()
        
        # Handle zero counts
        probs = byte_counts[byte_counts > 0] / total_bytes
        return float(-np.sum(probs * np.log2(probs)))

    def get_window_features(self, window: List[int]) -> Dict[str, float]:
        """Extract additional features from byte window.
        
        Args:
            window: List of bytes
            
        Returns:
            Dictionary of feature values
        """
        arr = np.array(window)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'unique_ratio': len(np.unique(arr)) / len(arr),
            'max_run': max(len(list(g)) for _, g in itertools.groupby(window))
        }

    def prioritize(self, byte_sequence: List[int]) -> List[Tuple[int, Dict[str, float]]]:
        """Prioritize sequence regions based on entropy and features.
        
        Args:
            byte_sequence: Input byte sequence as list of integers
            
        Returns:
            List of (hash_value, metrics) tuples sorted by priority
        """
        self._clean_cache()  # Maintain cache size
        
        # Process each scale
        all_scores = []
        for scale in self.scales:
            hash_windows = self.rolling_hash(byte_sequence, scale)
            
            for hash_val, window in hash_windows:
                if hash_val in self.entropy_cache:
                    metrics = self.entropy_cache[hash_val]
                else:
                    # Compute entropy and features
                    window_arr = np.array(window)
                    entropy = self.compute_entropy(window_arr)
                    
                    # Only process high-entropy regions
                    if entropy > self.min_entropy_threshold:
                        features = self.get_window_features(window)
                        metrics = {
                            'entropy': entropy,
                            'scale': scale,
                            **features
                        }
                        self.entropy_cache[hash_val] = metrics
                        all_scores.append((hash_val, metrics))
        
        # Sort by composite score
        def score_window(metrics):
            return (
                metrics['entropy'] * 0.4 +
                metrics['unique_ratio'] * 0.3 +
                (1.0 / metrics['scale']) * 0.2 +  # Prefer smaller windows
                (1.0 / (1.0 + metrics['max_run'])) * 0.1  # Penalize long runs
            )
        
        all_scores.sort(key=lambda x: score_window(x[1]), reverse=True)
        return all_scores

    def get_window_position(self, hash_val: int) -> Optional[Tuple[int, int]]:
        """Get the position of a window from its hash value."""
        return self.hash_index.get(hash_val)

    def analyze_region(self, byte_sequence: List[int], start: int, end: int) -> Dict[str, float]:
        """Detailed analysis of a specific sequence region."""
        if start >= end or end > len(byte_sequence):
            return {}
            
        region = byte_sequence[start:end]
        return {
            'entropy': self.compute_entropy(np.array(region)),
            **self.get_window_features(region)
        }

    def find_patch_boundaries(self, byte_seq: torch.Tensor) -> List[int]:
        """Find patch boundaries ensuring no multi-byte UTF-8 characters are split."""
        byte_seq_np = byte_seq.cpu().numpy().tolist()[0]  # Assuming batch size 1
        # Decode to string with replacement for invalid sequences
        try:
            decoded_str = bytes(byte_seq_np).decode('utf-8', errors='replace')
            # Convert string back to byte indices for valid characters
            byte_indices = []
            current_byte = 0
            for char in decoded_str:
                char_bytes = char.encode('utf-8')
                current_byte += len(char_bytes)
                byte_indices.append(current_byte)
        except Exception as e:
            logging.error(f"Error decoding byte sequence: {e}")
            byte_indices = []
        
        # Now, use the byte_indices as possible patch boundaries
        boundaries = []
        for boundary in byte_indices:
            # Ensure boundary is valid in the original byte sequence
            if 0 < boundary < len(byte_seq_np):
                boundaries.append(boundary)
        
        # Now prioritize the boundaries using entropy and features
        prioritized_boundaries = self.prioritize(byte_seq_np)
        selected_boundaries = []
        for hash_val, metrics in prioritized_boundaries:
            pos = self.get_window_position(hash_val)
            if pos and self._is_valid_utf8_boundary(byte_seq_np, pos[0]):
                selected_boundaries.append(pos[0])
        
        # Ensure boundaries are sorted and unique
        selected_boundaries = sorted(list(set(selected_boundaries)))
        
        # Final boundaries aligned with valid UTF-8 character boundaries
        final_boundaries = [b for b in selected_boundaries if b in byte_indices]
        
        return final_boundaries

    def create_patches(self, byte_seq: torch.Tensor) -> List[torch.Tensor]:
        """Convert byte sequence into patches based on entropy boundaries."""
        boundaries = self.find_patch_boundaries(byte_seq)
        patches = []
        
        start_idx = 0
        for end_idx in boundaries:
            if end_idx > start_idx:
                patch = byte_seq[:, start_idx:end_idx]  
                patches.append(patch)
                start_idx = end_idx
                
        # Add final patch
        if start_idx < byte_seq.size(1):
            patches.append(byte_seq[:, start_idx:])
            
        return patches

    @torch.no_grad()
    def reset_context(self):
        """Reset context when starting new document/segment."""
        # Clear any stateful buffers in entropy model
        pass

# ----------------------------
# CrossAttentionBlock
# ----------------------------

class CrossAttentionBlock(nn.Module):
    """Cross-attention block that properly mixes byte and patch information."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Single projection for Q, K, V
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,  # [batch_size, num_queries, hidden_size]
        keys_values: torch.Tensor,  # [batch_size, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with proper masking and scaling."""
        batch_size, num_queries, _ = queries.size()
        seq_len = keys_values.size(1)
        
        # Layer norm first
        queries = self.norm(queries)
        keys_values = self.norm(keys_values)
        
        # Project all Q, K, V at once
        qkv = self.qkv(queries)  # [batch_size, num_queries, 3 * hidden_size]
        kv = self.qkv(keys_values)  # [batch_size, seq_len, 3 * hidden_size]
        
        # Split into Q, K, V
        q = qkv[:, :, :self.hidden_size]
        k = kv[:, :, self.hidden_size:2*self.hidden_size]
        v = kv[:, :, 2*self.hidden_size:]
        
        # Split heads
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_queries, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # [batch_size, num_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch_size, num_heads, num_queries, seq_len]
        
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(~attention_mask, float('-inf'))
            
        # Convert scores to probabilities
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, num_queries, head_dim]
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output

# ----------------------------
# GlobalLatentTransformer
# ----------------------------

class GlobalLatentTransformer(nn.Module):
    """Large global transformer that processes patch representations."""
    def __init__(
        self,
        hidden_size: int = 1024,  # Reduced from 4096
        num_layers: int = 16,      # Reduced from 32
        num_heads: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer initialization with pre-LayerNorm
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_size, eps=1e-6),
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm2': nn.LayerNorm(hidden_size, eps=1e-6),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process patch representations with causal attention."""
        # Create causal mask
        mask = self.create_causal_mask(x.size(1), x.device)
        
        # Process through transformer layers
        for layer in self.layers:
            # Pre-LayerNorm for attention
            normed = layer['norm1'](x)
            
            # Self-attention with causal masking
            attn_out, _ = layer['attention'](
                query=normed,
                key=normed,
                value=normed,
                attn_mask=mask,
                need_weights=False
            )
            x = x + attn_out
            
            # Pre-LayerNorm for MLP
            normed = layer['norm2'](x)
            
            # MLP block
            x = x + layer['mlp'](normed)
            
        return self.final_norm(x)

# ----------------------------
# EntropyGuidedAttention Placeholder
# ----------------------------

class EntropyGuidedAttention(nn.Module):
    """Placeholder for EntropyGuidedAttention. Replace with actual implementation."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attention_mask)
        attn_output = self.dropout(attn_output)
        x = self.norm(x + attn_output)
        return x

# ----------------------------
# BLTModel
# ----------------------------

class BLTModel(nn.Module):
    """Complete Byte Latent Transformer model implementation."""
    def __init__(
        self,
        local_hidden_size: int = 256,  # Adjusted to 256
        global_hidden_size: int = 1024,  # Reduced from 4096
        num_local_encoder_layers: int = 1,
        num_global_layers: int = 16,      # Reduced from 32
        num_local_decoder_layers: int = 4,  # Reduced from 9
        dropout: float = 0.1,
        window_size: int = 256,        # Smaller window for better memory handling
        n_gram_sizes: List[int] = [3, 4],  # Reduced n-gram sizes
        n_gram_vocab_size: int = 30000      # Smaller n-gram vocab
    ):
        super().__init__()
        
        # Enable memory-efficient attention
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Entropy-based patching
        self.patcher = BabylonIndex(
            scales=n_gram_sizes,
            max_cache_size=100000,
            min_entropy_threshold=0.3
        )
        
        # Local encoder for bytes to patches
        self.local_encoder = LocalEncoder(
            hidden_size=local_hidden_size,
            num_layers=num_local_encoder_layers,
            num_heads=8,
            window_size=window_size,
            n_gram_sizes=n_gram_sizes,
            n_gram_vocab_size=n_gram_vocab_size
        )
        
        # Project patches to global hidden size
        self.patch_projection = nn.Sequential(
            nn.Linear(local_hidden_size, global_hidden_size),
            nn.LayerNorm(global_hidden_size, eps=1e-6)
        )
        
        # Global transformer for patches
        self.global_transformer = GlobalLatentTransformer(
            hidden_size=global_hidden_size,
            num_layers=num_global_layers,
            num_heads=32,
            dropout=dropout
        )
        
        # Project back to local hidden size
        self.patch_deprojection = nn.Sequential(
            nn.Linear(global_hidden_size, local_hidden_size),
            nn.LayerNorm(local_hidden_size, eps=1e-6)
        )
        
        # Local decoder for patches to bytes
        self.local_decoder = LocalDecoder(
            hidden_size=local_hidden_size,
            num_layers=num_local_decoder_layers,
            num_heads=8,
            dropout=dropout
        )
        
        self.context_size = window_size  # For sampling purposes

    def forward(
        self, 
        byte_seq: torch.Tensor,  # [batch_size, seq_len]
        return_patch_boundaries: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        """Complete forward pass of BLT model."""
        batch_size, seq_len = byte_seq.size()
        device = byte_seq.device
        
        # Find patch boundaries using entropy model
        patch_boundaries = self.patcher.find_patch_boundaries(byte_seq)
        
        # Encode bytes into patches
        patch_repr = self.local_encoder(byte_seq, patch_boundaries)  # [batch_size, num_patches, local_hidden]
        
        # Project to global hidden size
        patch_repr = self.patch_projection(patch_repr)  # [batch_size, num_patches, global_hidden]
        
        # Process with global transformer
        patch_repr = self.global_transformer(patch_repr)  # [batch_size, num_patches, global_hidden]
        
        # Project back to local hidden size
        patch_repr = self.patch_deprojection(patch_repr)  # [batch_size, num_patches, local_hidden]
        
        # Create causal mask for decoder
        history_len = 1  # Changed from min(seq_len, 256) to 1 for single-byte prediction
        causal_mask = torch.ones(history_len, history_len, device=device).bool()  # No masking needed for single byte
        
        # Get byte history embeddings
        history = byte_seq[:, -history_len:]
        history_embeds = self.local_encoder.byte_embeddings(history)
        
        # Decode to byte predictions
        byte_logits = self.local_decoder(
            patches=patch_repr,
            byte_history=history_embeds,
            causal_mask=causal_mask
        )  # [batch_size, 1, 256]
        
        if return_patch_boundaries:
            return byte_logits, patch_boundaries
        return byte_logits

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,  # [batch_size, 1, 256] 
        targets: torch.Tensor,  # [batch_size]
        smoothing: float = 0.1
    ) -> torch.Tensor:
        """Compute cross entropy loss with label smoothing."""
        vocab_size = logits.size(-1)
        
        # Reshape logits and targets for loss computation
        logits = logits.view(-1, vocab_size)  # [(batch_size * 1), 256]
        targets = targets.view(-1)             # [(batch_size * 1)]
        
        # Check target indices are within [0, vocab_size-1]
        if targets.max() >= vocab_size or targets.min() < 0:
            logging.error(f"Target indices out of bounds: min={targets.min()}, max={targets.max()}")
            raise ValueError("Target indices exceed vocabulary size!")
        
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (vocab_size - 1))
            true_dist.scatter_(-1, targets.unsqueeze(-1), 1.0 - smoothing)
            
        return -torch.sum(true_dist * F.log_softmax(logits, dim=-1), dim=-1).mean()

def entropy_based_sampling(logits: torch.Tensor, config: SamplerConfig) -> torch.Tensor:
    """Sample next tokens based on entropy thresholds."""
    probs = F.softmax(logits, dim=-1)
    entropy = calculate_entropy(probs)

    # Initialize output tensor
    sampled = torch.zeros_like(entropy, dtype=torch.long)

    # Low entropy: greedy sampling
    low_mask = entropy < config.low_entropy_threshold
    if low_mask.any():
        sampled[low_mask] = torch.argmax(probs[low_mask], dim=-1)

    # Medium entropy: top-k sampling
    med_mask = (entropy >= config.low_entropy_threshold) & (entropy < config.medium_entropy_threshold)
    if med_mask.any():
        top_k = 10
        top_k_probs, top_k_indices = torch.topk(probs[med_mask], k=top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
        sampled[med_mask] = top_k_indices.gather(1, sampled_indices.unsqueeze(-1)).squeeze(-1)

    # High entropy: random sampling
    high_mask = entropy >= config.medium_entropy_threshold
    if high_mask.any():
        sampled[high_mask] = torch.multinomial(probs[high_mask], num_samples=1).squeeze(-1)

    return sampled

def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Calculate entropy for each sample in the batch."""
    return -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)

# ----------------------------
# LocalEncoder
# ----------------------------

class LocalEncoder(nn.Module):
    """Local encoder that efficiently maps bytes to patches."""
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_heads: int = 8,
        window_size: int = 512,
        dropout: float = 0.1,
        n_gram_sizes: List[int] = [3, 4, 5],
        n_gram_vocab_size: int = 30000
    ):
        super().__init__()
        
        # Byte embeddings
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        
        # N-gram hash embeddings
        self.n_gram_embeddings = nn.ModuleDict({
            f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size)
            for n in n_gram_sizes
        })
        for embed in self.n_gram_embeddings.values():
            nn.init.normal_(embed.weight, mean=0.0, std=0.02)
            
        self.n_gram_sizes = n_gram_sizes
        self.window_size = window_size
        self.n_gram_vocab_size = n_gram_vocab_size  # Store for use in hashing
        
        # Local transformer layers with fixed window attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        
        # Cross attention for patch creation
        self.cross_attention = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def compute_n_gram_hashes(self, byte_seq: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Compute rolling hash values for n-grams efficiently with bounds checking."""
        batch_size, seq_len = byte_seq.size()
        device = byte_seq.device
        hashes = {}
        
        for n in self.n_gram_sizes:
            if seq_len < n:
                continue
                
            # Use safe indexing with proper bounds checking
            n_gram = byte_seq.unfold(1, n, 1)  # [batch_size, seq_len - n + 1, n]
            powers = torch.tensor([pow(256, i, 2**32) for i in range(n)], 
                                 dtype=torch.long, device=device)
            
            # Compute hashes with bounds checking
            valid_length = n_gram.size(1)
            hash_vals = torch.zeros((batch_size, valid_length), 
                                  dtype=torch.long, device=device)
            
            if valid_length > 0:  # Only compute if we have valid n-grams
                # Compute hash: sum(byte * (256^i)) mod 2^32
                hash_vals = (n_gram * powers).sum(dim=-1) % 2**32
                # **FIX**: Map hash_vals to [0, n_gram_vocab_size - 1]
                hash_vals = hash_vals % self.n_gram_vocab_size
            
            hashes[n] = hash_vals

        return hashes

    def create_local_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local window attention mask with proper bounds checking."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        window_size = min(self.window_size, seq_len)  # Ensure window size doesn't exceed sequence length
        
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + 1)  # +1 for causal masking
            mask[i, start:end] = False
            
        return mask

    def forward(
        self, 
        byte_seq: torch.Tensor,
        patch_boundaries: List[int]
    ) -> torch.Tensor:
        """Forward pass with safe indexing and bounds checking."""
        batch_size, seq_len = byte_seq.size()
        
        # Input validation
        if seq_len == 0:
            return torch.zeros((batch_size, 1, self.hidden_size), 
                             device=byte_seq.device)
        
        # Get byte embeddings
        x = self.byte_embeddings(byte_seq)  # [batch_size, seq_len, hidden_size]
        
        # Add n-gram embeddings with safe indexing
        n_gram_hashes = self.compute_n_gram_hashes(byte_seq)
        n_gram_scale = 1.0 / (len(self.n_gram_sizes) + 1)
        
        for n, hash_vals in n_gram_hashes.items():
            valid_length = hash_vals.size(1)
            if valid_length > 0:
                n_gram_embeds = self.n_gram_embeddings[f'n{n}'](hash_vals) * n_gram_scale
                # Safely add embeddings only where we have valid n-grams
                x[:, :valid_length] += n_gram_embeds
                
        x = self.dropout(x)
        
        # Process through local transformer layers with safe masking
        attention_mask = self.create_local_attention_mask(seq_len, x.device)
        
        for layer in self.transformer:
            x = layer(x, src_mask=attention_mask)
            
        # Create patch representations through cross attention with bounds checking
        patches = []
        start_idx = 0
        
        # Ensure patch boundaries are valid
        valid_boundaries = [b for b in patch_boundaries if 0 < b <= seq_len]
        if not valid_boundaries:
            valid_boundaries = [seq_len]
        
        for end_idx in valid_boundaries:
            if end_idx > start_idx:
                # Get bytes for this patch
                patch_bytes = x[:, start_idx:end_idx]
                
                # Safe mean pooling for query
                if patch_bytes.size(1) > 0:
                    query = torch.mean(patch_bytes, dim=1, keepdim=True)
                    
                    # Cross attend to create patch representation
                    patch_repr = self.cross_attention(
                        queries=query,
                        keys_values=patch_bytes
                    )
                    
                    patches.append(patch_repr)
                
                start_idx = end_idx
                
        # Handle final patch if needed
        if start_idx < seq_len:
            patch_bytes = x[:, start_idx:]
            query = torch.mean(patch_bytes, dim=1, keepdim=True)
            patch_repr = self.cross_attention(
                queries=query,
                keys_values=patch_bytes
            )
            patches.append(patch_repr)
            
        # Combine patches with safe handling
        if patches:
            patches = torch.cat(patches, dim=1)
        else:
            # Fallback if no valid patches were created
            patches = x.mean(dim=1, keepdim=True)
        
        return self.norm(patches)
        
# ----------------------------
# LocalDecoder
# ----------------------------

class LocalDecoder(nn.Module):
    """Local decoder that maps patches back to bytes."""
    def __init__(
        self,
        hidden_size: int = 256,  # Adjusted to 256
        num_layers: int = 4,      # Reduced from 9
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Cross attention to get initial byte representations from patches
        self.initial_cross_attention = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
        # Output projection to bytes with proper initialization
        self.byte_pred = nn.Linear(hidden_size, 256)
        nn.init.normal_(self.byte_pred.weight, std=0.02)
        nn.init.zeros_(self.byte_pred.bias)
        
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for decoder."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()

    def forward(
        self,
        patches: torch.Tensor,  # [batch_size, num_patches, hidden_size]
        byte_history: torch.Tensor,  # [batch_size, history_len, hidden_size]
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode patches to byte predictions.
        
        Args:
            patches: Patch representations from global transformer
            byte_history: Embeddings of previous bytes for causal decoding
            causal_mask: Optional pre-computed causal mask
            
        Returns:
            Logits for next byte prediction [batch_size, history_len, 256]
        """
        batch_size = patches.size(0)
        history_len = byte_history.size(1)
        
        # Create causal mask if not provided
        if causal_mask is None:
            causal_mask = self.create_causal_mask(history_len, patches.device)
            
        # Initial cross attention from history to patches
        byte_repr = self.initial_cross_attention(
            queries=byte_history,
            keys_values=patches
        )
        byte_repr = self.dropout(byte_repr)
        
        # Process through transformer decoder layers
        for layer in self.transformer:
            byte_repr = layer(
                tgt=byte_repr,
                memory=patches,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=None
            )
                
        # Project to byte predictions
        byte_repr = self.norm(byte_repr)
        byte_logits = self.byte_pred(byte_repr)  # [batch_size, history_len, 256]
        
        return byte_logits

    def generate_step(
        self,
        patches: torch.Tensor,
        byte_history: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generate single next byte prediction for autoregressive generation.
        
        Args:
            patches: Current patch representations
            byte_history: Previous byte history
            temperature: Sampling temperature
            top_k: Number of top logits to sample from
            
        Returns:
            Predicted next byte index [batch_size, 1]
        """
        logits = self(patches, byte_history)[:, -1]  # Get predictions for last position
        logits = logits / temperature
        
        # Apply top-k sampling
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        
        # Sample next byte
        next_byte_idx = torch.multinomial(probs, num_samples=1)
        next_byte = torch.gather(top_k_indices, -1, next_byte_idx)
        
        return next_byte

# ----------------------------
# RLHFTrainer
# ----------------------------

class RLHFTrainer:
    """Trainer class that integrates RLHF mechanisms with the training loop."""
    def __init__(self, model: nn.Module, optimizer: Optimizer, babylon_index: BabylonIndex, tokenizer: ByteTokenizer, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.babylon_index = babylon_index
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = amp.GradScaler(enabled=True)
    
    def train_step(self, context: torch.Tensor, target: torch.Tensor) -> float:
        """
        Perform a single training step.

        Args:
            context (torch.Tensor): Input byte sequences [batch_size, context_size].
            target (torch.Tensor): Target bytes [batch_size].

        Returns:
            float: Loss value.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Add assertions to ensure target indices are within [0, 255]
        if target.max() >= 256 or target.min() < 0:
            logging.error(f"Target indices out of bounds: min={target.min()}, max={target.max()}")
            raise ValueError("Target indices exceed vocabulary size!")
        
        with autocast(device_type='cuda', enabled=self.scaler.is_enabled()):   # Corrected autocast usage
            # Forward pass with proper handling of cross-attention masking
            logits = self.model(context)  # [batch_size, 1, 256]
            loss = self.model.compute_loss(logits, target)
        
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
        return loss.item()

    def save_model(self, epoch: int, checkpoint_dir: str = "checkpoints"):
        """Save model checkpoint along with optimizer state."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        model_state_dict = self.model.module.state_dict() if isinstance(
            self.model, DDP
        ) else self.model.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }, checkpoint_path)
        logging.info(f'Checkpoint saved at {checkpoint_path}')

    def load_model(self, checkpoint_path: str) -> int:
        """Load model checkpoint along with optimizer state."""
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file {checkpoint_path} does not exist.")
            return 0  # Starting from epoch 0

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        scaler_state_dict = checkpoint['scaler_state_dict']
        epoch = checkpoint.get('epoch', 0)

        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.scaler.load_state_dict(scaler_state_dict)

        logging.info(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}")
        return epoch

    def evaluate(self, validation_loader: DataLoader, device: torch.device, use_amp: bool, scaler: amp.GradScaler) -> float:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):  # Corrected autocast usage
                    outputs = self.model(inputs)
                    loss = self.model.compute_loss(outputs, targets)
                    total_loss += loss.item()
        avg_loss = total_loss / len(validation_loader)
        logging.info(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss

# ----------------------------
# ByteIterableDataset
# ----------------------------

class ByteIterableDataset(IterableDataset):
    def __init__(self, csv_file: str, context_size: int = 64):
        """
        Args:
            csv_file (str): Path to the CSV file containing text data.
            context_size (int): Size of the context window.
        """
        self.csv_file = csv_file
        self.context_size = context_size

    def parse_csv(self):
        """Generator that yields texts line by line from the CSV."""
        try:
            for chunk in pd.read_csv(self.csv_file, chunksize=1000):
                texts = chunk['text'].dropna().astype(str).tolist()
                for text in texts:
                    yield text
        except Exception as e:
            logging.error(f"Error reading CSV file {self.csv_file}: {e}")
            return

    def __iter__(self):
        return self.generator()

    def generator(self):
        """Generator function to yield (context, target) pairs."""
        for text in self.parse_csv():
            if len(text.strip()) == 0:
                continue  # Skip empty texts

            try:
                bytes_data = text.encode('utf-8', errors='replace')
            except Exception as e:
                logging.error(f"Encoding error for text: {e}")
                continue  # Skip texts that cause encoding errors

            if len(bytes_data) < self.context_size + 1:
                continue  # Skip texts that are too short

            byte_seq = [b for b in bytes_data]
            for i in range(len(byte_seq) - self.context_size):
                context = byte_seq[i:i + self.context_size]
                target = byte_seq[i + self.context_size]

                last_byte = context[-1]
                if not (0x80 <= last_byte <= 0xBF):
                    yield (torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long))
                else:
                    # Ensure that we're not in the middle of a multi-byte character
                    # Skip this target to prevent partial character prediction
                    continue

# ----------------------------
# Utility Functions
# ----------------------------

def decode_bytes(byte_tensor: torch.Tensor) -> str:
    """Convert a tensor of byte values back to text."""
    try:
        return bytes(byte_tensor.cpu().numpy().tolist()).decode('utf-8', errors='replace')
    except Exception as e:
        logging.error(f"Error decoding bytes: {e}")
        return ""

def byte_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Custom cross entropy with label smoothing for byte-level training.

    Args:
        logits: Shape [batch_size, vocab_size]
        targets: Shape [batch_size]
        smoothing: Label smoothing factor
    """
    if logits.size(0) != targets.size(0):
        raise ValueError(f"Batch size mismatch: logits {logits.size(0)}, targets {targets.size(0)}")
        
    confidence = 1.0 - smoothing
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()

def generate_text(model: nn.Module, seed_text: str, length: int, config: SamplerConfig, device: torch.device) -> str:
    """
    Generate text using the trained model based on the seed_text.

    Args:
        model (nn.Module): Trained model.
        seed_text (str): Seed text to start generation.
        length (int): Number of bytes to generate.
        config (SamplerConfig): Configuration for entropy-based sampling.
        device (torch.device): Device to run the model on.

    Returns:
        str: Generated text.
    """
    model.eval()
    generated = seed_text.encode('utf-8')
    context_size = model.context_size
    context_bytes = generated[-context_size:] if len(generated) >= context_size else generated
    context = torch.tensor([byte for byte in context_bytes], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(length):
            logits = model(context)  # [batch_size, 1, 256]
            sampled_byte = entropy_based_sampling(logits[:, -1, :], config)  # [batch_size]
            sampled_byte = sampled_byte.to(device)
            generated += sampled_byte.cpu().numpy().tolist()
            # Update context
            if len(generated) >= context_size + 1:
                new_context = generated[-context_size:]
            else:
                new_context = generated
            context = torch.tensor([byte for byte in new_context], dtype=torch.long).unsqueeze(0).to(device)

    return decode_bytes(torch.tensor(generated))


# ----------------------------
# Checkpointing Functions
# ----------------------------

def save_checkpoint(
    epoch: int,
    trainer: RLHFTrainer,
    checkpoint_dir: str = "checkpoints"
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    model_state_dict = trainer.model.module.state_dict() if isinstance(
        trainer.model, DDP
    ) else trainer.model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scaler_state_dict': trainer.scaler.state_dict(),
    }, checkpoint_path)
    logging.info(f'Checkpoint saved at {checkpoint_path}')

def load_checkpoint(
    trainer: RLHFTrainer,
    checkpoint_path: str
) -> int:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file {checkpoint_path} does not exist.")
        return 0  # Starting from epoch 0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    scaler_state_dict = checkpoint['scaler_state_dict']
    epoch = checkpoint.get('epoch', 0)

    if isinstance(trainer.model, DDP):
        trainer.model.module.load_state_dict(model_state_dict)
    else:
        trainer.model.load_state_dict(model_state_dict)
    trainer.optimizer.load_state_dict(optimizer_state_dict)
    trainer.scaler.load_state_dict(scaler_state_dict)

    logging.info(f"Loaded checkpoint from {checkpoint_path}, epoch {epoch}")
    return epoch

# ----------------------------
# Validation Function
# ----------------------------

def validate(model, validation_loader, device, use_amp, scaler: amp.GradScaler) -> float:
    """Validation loop with mixed precision."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast('cuda', enabled=use_amp):  # Corrected autocast usage
                outputs = model(inputs)
                loss = model.compute_loss(outputs, targets)
                total_loss += loss.item()
    model.train()  # Reset to training mode
    return total_loss / len(validation_loader)

# ----------------------------
# Training Function
# ----------------------------

def setup_training(
    trainer: RLHFTrainer,
    train_loader: DataLoader,
    validation_loader: Optional[DataLoader],
    config: Dict[str, Any]
):
    """Optimized training setup for better GPU utilization and memory efficiency."""
    logging.info("Starting training setup...")
    # Enable mixed precision for better GPU memory efficiency
    use_amp = config.get("use_amp", True)
    # Set cudnn settings for improved performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Gradient checkpointing for memory-constrained systems
    if config.get("use_gradient_checkpointing", False):
        model = trainer.model
        model.apply(torch.utils.checkpoint.checkpoint_sequential)
    # Initialize statistics tracking
    running_loss = deque(maxlen=100)
    global_step = 0
    # Training loop
    start_epoch = config.get("start_epoch", 0)
    for epoch in range(start_epoch, config["epochs"]):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (context, target) in enumerate(train_loader):
            try:
                # Move data to device
                inputs = context.to(trainer.device, non_blocking=True)
                targets = target.to(trainer.device, non_blocking=True)
                # Perform training step
                loss = trainer.train_step(inputs, targets)
                
                # Accumulate loss
                running_loss.append(loss)
                epoch_loss += loss
                num_batches += 1
                global_step += 1
                # Log progress
                if config.get("rank", 0) == 0 and batch_idx % config.get("log_interval", 100) == 0:
                    avg_loss = sum(running_loss) / len(running_loss) if running_loss else loss
                    
                    # Use get_statistics to retrieve the current learning rate
                    optimizer_stats = trainer.optimizer.get_statistics()
                    current_lr = optimizer_stats.get('avg_learning_rates', config["learning_rate"])
                    
                    wandb.log({
                        "batch_loss": loss,
                        "running_loss": avg_loss,
                        "learning_rate": current_lr,
                        "global_step": global_step
                    })
                    
                    logging.info(
                        f'Epoch {epoch}, Batch {batch_idx}, '
                        f'Loss: {loss:.4f}, '
                        f'Running Loss: {avg_loss:.4f}, '
                        f'LR: {current_lr:.6f}, '
                        f'Step: {global_step}'
                    )
                    
            except RuntimeError as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                trainer.optimizer.zero_grad(set_to_none=True)  # Reset gradients
                continue
                
        # End of epoch processing
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            if config.get("rank", 0) == 0:
                wandb.log({
                    "epoch": epoch,
                    "epoch_loss": avg_epoch_loss,
                    "global_step": global_step
                })
                logging.info(f'Epoch {epoch} completed, Average Loss: {avg_epoch_loss:.4f}')
                
        # Validation
        if validation_loader and (epoch + 1) % config.get("val_interval", 2) == 0:
            val_loss = trainer.evaluate(validation_loader, trainer.device, use_amp, trainer.scaler)
            if config.get("rank", 0) == 0:
                wandb.log({"val_loss": val_loss})
                logging.info(f'Validation Loss: {val_loss:.4f}')
                
        # Save checkpoint
        if (epoch + 1) % config.get("save_interval", 5) == 0 and config.get("rank", 0) == 0:
            trainer.save_model(epoch, config.get("checkpoint_dir", "checkpoints"))
    logging.info("Training completed.")

# ----------------------------
# DataLoader Preparation
# ----------------------------

def prepare_dataloaders(
    train_csv: str,
    test_csv: str,
    batch_size: int,
    context_size: int,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and test DataLoaders from CSV files.

    Args:
        train_csv (str): Path to training CSV file.
        test_csv (str): Path to test CSV file.
        batch_size (int): Batch size.
        context_size (int): Context window size.
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to pin memory for GPU transfer.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create IterableDatasets
    train_dataset = ByteIterableDataset(
        csv_file=train_csv,
        context_size=context_size
    )
    test_dataset = ByteIterableDataset(
        csv_file=test_csv,
        context_size=context_size
    )

    # Create DataLoaders with optimized memory settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset doesn't support shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=1,        # Reduced from 2
        persistent_workers=True,  # Keeps workers alive to reduce memory overhead
        drop_last=True             # Avoid irregular batch sizes
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=1,
        persistent_workers=True,
        drop_last=True
    )

    logging.info(f"Train dataset is an IterableDataset.")
    logging.info(f"Test dataset is an IterableDataset.")

    return train_loader, test_loader

# ----------------------------
# Additional Memory Optimization Functions
# ----------------------------

def log_memory_stats(step: int):
    """Log current CUDA memory usage."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_reserved = torch.cuda.memory_reserved() / 1024**2
        logging.info(f"Step {step} - Memory allocated: {memory_allocated:.2f}MB, "
                    f"reserved: {memory_reserved:.2f}MB")

def adjust_batch_size(current_batch_size: int, memory_threshold: float = 0.9) -> int:
    """
    Adjust the batch size based on current memory usage.

    Args:
        current_batch_size (int): Current batch size.
        memory_threshold (float): Memory usage threshold (0.0 to 1.0).

    Returns:
        int: Adjusted batch size.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        if allocated > memory_threshold:
            return max(1, current_batch_size // 2)
    return current_batch_size

# ----------------------------
# Main Function and Entry Point
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced Byte-Level Transformer Training Script")
    parser.add_argument('--distributed', action='store_true', help='Enable Distributed Data Parallel training')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')  # Updated to 25
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')  # Set to 128 to prevent memory issues
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='Initial learning rate')  # Increased to 5e-3
    parser.add_argument('--context_size', type=int, default=64, help='Context size for the model')  # Updated to 64
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers')  # Increased to 8
    parser.add_argument('--validation', action='store_true', help='Enable validation during training')
    parser.add_argument('--validation_batch_size', type=int, default=128, help='Batch size for validation')  # Adjusted as needed
    parser.add_argument('--no_anomaly', action='store_true', help='Disable anomaly detection for faster training')
    parser.add_argument('--gpu_index', type=int, default=None, help='Specify GPU index to use (if multiple GPUs are present)')
    parser.add_argument('--project', type=str, default="blt-training", help='WandB project name')
    return parser.parse_args()

def setup_distributed(args: argparse.Namespace) -> Tuple[torch.device, int]:
    """Initialize distributed training if multiple GPUs are available and enabled.

    Ensures that the NVIDIA GeForce RTX 4050 GPU is selected for training.

    Args:
        args: Parsed command-line arguments.

    Returns:
        tuple: (device, world_size)
    """
    if args.distributed and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        logging.info(f"Number of CUDA devices available: {available_gpus}")

        # Attempt to find the NVIDIA GeForce RTX 4050
        target_gpu_name = "NVIDIA GeForce RTX 4050"
        target_gpu_index = None
        for i in range(available_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            logging.info(f"CUDA Device {i}: {gpu_name}")
            if target_gpu_name.lower() in gpu_name.lower():
                target_gpu_index = i
                logging.info(f"Selected GPU {i}: {gpu_name}")
                break

        if target_gpu_index is None:
            logging.warning(f"{target_gpu_name} not found. Using default CUDA device.")
            target_gpu_index = 0  # Default to first CUDA device

        # Set CUDA device
        torch.cuda.set_device(target_gpu_index)
        device = torch.device(f'cuda:{target_gpu_index}')

        # Initialize Distributed Data Parallel if multiple GPUs are present
        if available_gpus > 1:
            if platform.system() == "Windows":
                backend = "gloo"
            else:
                backend = "nccl"

            # Automatically find a free port for initialization
            port = find_free_port()
            init_method = f'tcp://127.0.0.1:{port}'

            world_size = available_gpus
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(port)

            try:
                init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=args.rank)
                logging.info(f"Distributed training initialized on device {device}")
                return device, world_size
            except Exception as e:
                logging.error(f"Distributed initialization failed: {e}")
                logging.info("Falling back to single GPU training.")
                return device, 1
        else:
            logging.info(f"Training on single GPU: {device}")
            return device, 1
    else:
        if torch.cuda.is_available():
            # Attempt to find the NVIDIA GeForce RTX 4050
            available_gpus = torch.cuda.device_count()
            logging.info(f"Number of CUDA devices available: {available_gpus}")

            target_gpu_name = "NVIDIA GeForce RTX 4050"
            target_gpu_index = None
            for i in range(available_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                logging.info(f"CUDA Device {i}: {gpu_name}")
                if target_gpu_name.lower() in gpu_name.lower():
                    target_gpu_index = i
                    logging.info(f"Selected GPU {i}: {gpu_name}")
                    break

            if target_gpu_index is None:
                logging.warning(f"{target_gpu_name} not found. Using default CUDA device.")
                target_gpu_index = 0  # Default to first CUDA device

            # Set CUDA device
            torch.cuda.set_device(target_gpu_index)
            device = torch.device(f'cuda:{target_gpu_index}')
            logging.info(f"Training on GPU: {device}")
            return device, 1
        else:
            logging.info("CUDA not available. Training on CPU.")
            device = torch.device("cpu")
            return device, 1

def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# ----------------------------
# Inspect Dataset Function
# ----------------------------

def inspect_dataset(dataset: IterableDataset, num_samples: int = 5):
    """Inspect a few samples from the dataset."""
    logging.info(f"Inspecting {num_samples} samples from the dataset:")
    for i, (context, target) in enumerate(dataset):
        if i >= num_samples:
            break
        logging.info(f"Sample {i+1}: Context={context.tolist()}, Target={target.item()}")

# ----------------------------
# Main Function and Entry Point
# ----------------------------

def main():
    args = parse_args()
    device, world_size = setup_distributed(args)

    # Initialize wandb only on the first process
    if args.rank == 0:
        wandb.init(project=args.project, config={
            "epochs": args.epochs,
            "batch_size": args.batch_size * world_size,
            "learning_rate": args.learning_rate,
            "context_size": args.context_size,
            "num_workers": args.num_workers,
            "world_size": world_size
        })

    # Optionally enable anomaly detection for debugging
    if not args.no_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Updated configuration
    config = {
        "context_size": args.context_size,         # Set to 64
        "batch_size": args.batch_size,             # Set to 128
        "epochs": args.epochs,                     # Set to 25
        "learning_rate": args.learning_rate,       # Set to 5e-3
        "num_workers": args.num_workers,           # Set to 8
        "gradient_clip_norm": 1.0,                 # Increased from 0.5
        "weight_decay": 0.02,                      # Increased from 0.01
        "checkpoint_dir": "checkpoints",           # Default checkpoint directory
        "log_interval": 100,
        "val_interval": 2,
        "save_interval": 5,
        "use_amp": True,
        "use_gradient_checkpointing": False,        # Set to True if needed
        "rank": args.rank
    }

    # Prepare DataLoaders
    train_loader, test_loader = prepare_dataloaders(
        train_csv="data/wikitext_train.csv",  # Update dataset file path as needed
        test_csv="data/wikitext_test.csv",    # Update dataset file path as needed
        batch_size=config["batch_size"],
        context_size=config["context_size"],
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Initialize model with memory-optimized configuration
    model = BLTModel(
        local_hidden_size=256,                   # Adjusted to 256
        global_hidden_size=1024,                 # Reduced from 4096
        num_local_encoder_layers=1,
        num_global_layers=16,                     # Reduced from 32
        num_local_decoder_layers=4,               # Reduced from 9
        dropout=0.05,                             # Reduced to save VRAM
        window_size=256,                          # Smaller window for better memory handling
        n_gram_sizes=[3, 4],                      # Reduced n-gram sizes
        n_gram_vocab_size=30000                   # Smaller n-gram vocab
    ).to(device)

    # If using Distributed Data Parallel
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)

    # Configure optimizer
    optimizer = EnhancedSGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=0.9,
        weight_decay=0.01
    )
    
    # Initialize BabylonIndex and Tokenizer
    babylon_index = BabylonIndex(scales=[3, 4])
    tokenizer = ByteTokenizer()

    # Initialize RLHFTrainer
    rlhf_trainer = RLHFTrainer(
        model=model,
        optimizer=optimizer,
        babylon_index=babylon_index,
        tokenizer=tokenizer,
        device=device
    )

    # Optionally prepare validation DataLoader
    validation_dataloader = None

    if args.validation:
        validation_data_path = "data/wikitext_validation.csv"
        try:
            validation_dataset = ByteIterableDataset(
                csv_file=validation_data_path,
                context_size=config["context_size"]
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=args.validation_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=1,        # Reduced from 2
                persistent_workers=True,
                drop_last=True
            )
            logging.info(f"Validation dataset is an IterableDataset.")
            inspect_dataset(validation_dataloader.dataset, num_samples=5)

        except FileNotFoundError:
            logging.error(f"Validation dataset file '{validation_data_path}' not found. Skipping validation.")
        except pd.errors.EmptyDataError:
            logging.error(f"Validation dataset file '{validation_data_path}' is empty. Skipping validation.")
        except Exception as e:
            logging.error(f"Validation dataset file at '{validation_data_path}' has an incompatible format or is corrupted. Skipping validation.")
            logging.error(f"Error details: {e}")

    # Train with byte-optimized parameters
    setup_training(
        trainer=rlhf_trainer,
        train_loader=train_loader,
        validation_loader=validation_dataloader,
        config=config
    )

    # Sampling Configuration
    sampler_config = SamplerConfig(
        low_entropy_threshold=0.3,
        medium_entropy_threshold=1.2,
        high_entropy_threshold=2.5
    )

    # Generate Text (Ensure this runs only on one process to avoid multiple outputs)
    if args.rank == 0:
        seed_text = "The quick brown fox jumps over the lazy dog."
        logging.info("\nGenerating Text:")
        generated_text = generate_text(
            model=model,
            seed_text=seed_text,
            length=100,
            config=sampler_config,
            device=device
        )
        print(generated_text)

    # Clean up distributed training
    if is_initialized():
        destroy_process_group()

# ----------------------------
# Entry Point Check
# ----------------------------

if __name__ == "__main__":
    main()
