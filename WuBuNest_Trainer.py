# -*- coding: utf-8 -*-
"""
WuBuNest_Trainer.py
Fully Hyperbolic WuBu Nesting Model Trainer (v0.04 - Experimental Hyperbolic Core - Reordered Corrected v2)

WARNING: This version attempts a "fully hyperbolic" implementation, replacing many
         standard PyTorch components with custom hyperbolic geometry equivalents.
         This is highly experimental, increases complexity significantly, and may
         suffer from numerical instability or performance issues. Use with caution.
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
from torch.nn.parallel import DistributedDataParallel as DDP # Use alias for clarity
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
import inspect
import string
import hashlib
import functools

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Setup logger - Initial basic config, will be refined in main
logger = logging.getLogger("WuBuNestTrainerHyperbolic")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)

# Constants
EPS = 1e-7 # Small epsilon for numerical stability


# =====================================================================
# START: HAKMEM Components (Definitions moved BEFORE optimizer)
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

class HAKMEMEntropyHelper:
    """Calculates Shannon entropy for byte sequences with caching."""
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_cache = {}; self.max_cache_size = max_cache_size
    def _clean_cache(self):
        """Removes older entries if cache exceeds max size."""
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5)
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache: del self.entropy_cache[k]
    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...], List[int], bytes, torch.Tensor]) -> float:
        """Computes entropy, using cache if possible."""
        cache_key = None; byte_list = []
        # Convert various input types to a list of bytes and a hashable key
        if isinstance(byte_window, tuple): cache_key = byte_window; byte_list = list(byte_window)
        elif isinstance(byte_window, list): cache_key = tuple(byte_window); byte_list = byte_window
        elif isinstance(byte_window, bytes): cache_key = byte_window; byte_list = list(byte_window)
        elif isinstance(byte_window, np.ndarray): byte_list = byte_window.tolist(); cache_key = tuple(byte_list)
        elif isinstance(byte_window, torch.Tensor): byte_list = byte_window.cpu().byte().tolist(); cache_key = tuple(byte_list)
        else: return 0.0 # Skip unknown types
        if cache_key is not None and cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        if not byte_list: return 0.0
        try:
            byte_counts = np.bincount(np.array(byte_list, dtype=np.uint8), minlength=256)
            total_bytes = byte_counts.sum()
            if total_bytes == 0: return 0.0
            probs = byte_counts[byte_counts > 0] / total_bytes
            entropy = float(-np.sum(probs * np.log2(probs + EPS)))
            result = max(0.0, entropy)
            if cache_key is not None:
                self.entropy_cache[cache_key] = result
                self._clean_cache()
            return result
        except Exception as e: logger.warning(f"Entropy calc failed: {e}"); return 0.0

class HAKMEMBabylonIndex:
    """Splits byte sequences into 'word' and 'delimiter' patches based on text decoding."""
    def __init__(self, max_cache_size: int = 50000):
        self.entropy_helper = HAKMEMEntropyHelper(max_cache_size)
        self.whitespace_chars = set(string.whitespace)
        self.punctuation_chars = set(string.punctuation)
        logger.info("HAKMEMBabylonIndex initialized (Word/Punctuation Patching).")
    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Creates patches from a byte tensor, attempting UTF-8 decoding."""
        if byte_seq_tensor.numel() == 0: return []
        if byte_seq_tensor.dim() != 1: byte_seq_tensor = byte_seq_tensor.flatten()
        if byte_seq_tensor.numel() == 0: return []
        device = byte_seq_tensor.device
        try: text = byte_seq_tensor.cpu().numpy().tobytes().decode('utf-8', errors='replace')
        except Exception as e: logger.warning(f"UTF-8 Decode error: {e}. Patching failed."); return [] # Cannot decode, return no patches
        patches_with_entropy = []; current_patch_start = 0; in_word = False
        for i, char in enumerate(text):
            is_delimiter = char in self.whitespace_chars or char in self.punctuation_chars
            if is_delimiter:
                if in_word: # End of a word
                    word_str = text[current_patch_start:i]
                    try:
                        word_bytes = torch.tensor(list(word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                        if word_bytes.numel() > 0: patches_with_entropy.append((word_bytes, self.entropy_helper.compute_entropy(word_bytes)))
                    except Exception as enc_e: logger.warning(f"Word encoding error: {enc_e}") # Ignore encoding errors for this word
                    in_word = False
                try: # Process the delimiter itself
                    delim_bytes = torch.tensor(list(char.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                    if delim_bytes.numel() > 0: patches_with_entropy.append((delim_bytes, self.entropy_helper.compute_entropy(delim_bytes)))
                except Exception as enc_e: logger.warning(f"Delimiter encoding error: {enc_e}") # Ignore encoding errors for this delimiter
                current_patch_start = i + 1
            else: # Not a delimiter
                if not in_word: # Start of a new word
                    in_word = True; current_patch_start = i
        # Handle trailing word if sequence doesn't end with delimiter
        if in_word and current_patch_start < len(text):
            trailing_word_str = text[current_patch_start:]
            try:
                trailing_word_bytes = torch.tensor(list(trailing_word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                if trailing_word_bytes.numel() > 0: patches_with_entropy.append((trailing_word_bytes, self.entropy_helper.compute_entropy(trailing_word_bytes)))
            except Exception as enc_e: logger.warning(f"Trailing word encoding error: {enc_e}")
        # Filter out any empty patches that might have snuck through
        return [(p, e) for p, e in patches_with_entropy if p.numel() > 0]
    @torch.no_grad()
    def reset_context(self): self.entropy_helper.entropy_cache = {}

class HAKMEMCrossAttentionBlock(nn.Module):
    """A standard cross-attention block with LayerNorm and optional Flash Attention."""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size <= 0: raise ValueError("hidden_size must be positive")
        # Ensure num_heads is valid
        valid_heads = [h for h in range(max(1, num_heads), 0, -1) if hidden_size % h == 0] # Start check from num_heads
        if not valid_heads: original_num_heads=num_heads; num_heads = 1; logger.warning(f"HAKMEMCrossAttn: No valid head count for size {hidden_size}. Using 1 head (Requested: {original_num_heads}).")
        elif hidden_size % num_heads != 0: original_num_heads=num_heads; num_heads = valid_heads[0]; logger.warning(f"HAKMEMCrossAttn: Adjusted head count {original_num_heads} -> {num_heads} for hidden_size {hidden_size}.")
        self.hidden_size = hidden_size; self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(max(1, self.head_dim))
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Moved init_weights call outside __init__ to be applied by parent model

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Performs cross-attention.
            queries: [B, N_queries, D]
            keys_values: [B, N_kv, D]
            attention_mask: [B, N_kv] or [B, N_queries, N_kv] or [B, N_heads, N_queries, N_kv] (True means MASK)
        """
        batch_size, num_queries, _ = queries.size(); _, seq_len_kv, kv_hidden_size = keys_values.size()
        if seq_len_kv == 0: return torch.zeros_like(queries) # Handle empty key/value sequence
        if kv_hidden_size != self.hidden_size: raise ValueError(f"KV size mismatch: {kv_hidden_size} vs {self.hidden_size}")
        queries_norm = self.norm_q(queries); keys_values_norm = self.norm_kv(keys_values)
        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # Prepare mask for scaled_dot_product_attention (expects True where *not* masked)
        attn_mask_sdpa = None # Mask for SDPA (True = NOT MASKED)
        attn_mask_manual = None # Mask for manual (True = MASKED)
        if attention_mask is not None:
            mask_dtype = torch.bool; # Use bool for masking
            if attention_mask.dtype != torch.bool: attention_mask = attention_mask > 0 # Convert if needed
            # Reshape mask to [B, N_Heads, N_Queries, N_KV]
            if attention_mask.dim() == 2: # Shape [B, N_KV] -> assume applies to all queries
                attn_mask_manual = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, num_queries, -1)
            elif attention_mask.dim() == 3: # Shape [B, N_Queries, N_KV] -> assume applies to all heads
                attn_mask_manual = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif attention_mask.dim() == 4: attn_mask_manual = attention_mask # Shape [B, N_Heads, N_Queries, N_KV]
            else: logger.warning(f"Ignoring CrossAttention mask shape {attention_mask.shape}.")

            if attn_mask_manual is not None:
                 attn_mask_manual = attn_mask_manual.to(device=queries.device, dtype=mask_dtype)
                 expected_shape = (batch_size, self.num_heads, num_queries, seq_len_kv)
                 if attn_mask_manual.shape != expected_shape: attn_mask_manual = None; logger.warning("CrossAttention mask shape mismatch ignored.")
                 else: attn_mask_sdpa = ~attn_mask_manual # Invert for SDPA

        # Use Flash Attention if available
        use_flash = hasattr(F, 'scaled_dot_product_attention')
        output = None
        if use_flash:
            try:
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_sdpa, dropout_p=self.dropout.p if self.training else 0.0)
                if not torch.isfinite(output).all(): raise ValueError("Flash Attention produced NaN/Inf.")
            except Exception as flash_ex:
                logger.debug(f"Flash Attention failed: {flash_ex}. Falling back.", exc_info=False); use_flash = False; output = None
        if output is None: # Fallback to manual implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale; scores = torch.clamp(scores, min=-30.0, max=30.0)
            if attn_mask_manual is not None: # Manual masking expects True=MASK
                scores = scores.masked_fill(attn_mask_manual, float('-inf'))
            attn_probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype); attn_probs = torch.nan_to_num(attn_probs); attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        if not torch.isfinite(output).all(): logger.warning("NaN/Inf in CrossAttention final output. Replacing."); output = torch.nan_to_num(output, nan=0.0)
        return output

class HAKMEMQController:
    """A Q-learning agent to dynamically adjust optimizer hyperparameters."""
    def __init__(self, learning_rate: float=0.01, discount: float=0.95, epsilon: float=0.2, epsilon_decay: float=0.9998, min_epsilon: float=0.01, lr_scale_options: Optional[List[float]]=None, momentum_scale_options: Optional[List[float]]=None, max_q_table_size: int=10000):
        self.q_table: Dict[Tuple, Dict[str, np.ndarray]] = {}
        self.alpha = learning_rate; self.gamma = discount; self.epsilon = epsilon
        self.min_epsilon = min_epsilon; self.epsilon_decay = epsilon_decay
        self.prev_loss: Optional[float] = None; self.prev_state: Optional[Tuple] = None; self.prev_action: Optional[Dict[str, float]] = None
        if lr_scale_options is None: lr_scale_options = [0.9, 0.95, 1.0, 1.05, 1.1]
        if momentum_scale_options is None: momentum_scale_options = [0.95, 0.98, 1.0, 1.01, 1.02]
        self.action_ranges = {'lr_scale': np.array(lr_scale_options, dtype=np.float32), 'momentum_scale': np.array(momentum_scale_options, dtype=np.float32)}
        self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}
        self.loss_window = deque(maxlen=10); self.grad_norm_window = deque(maxlen=10)
        self.lr_window = deque(maxlen=5); self.momentum_window = deque(maxlen=5)
        self.performance_window = deque(maxlen=20); self.stable_steps = 0; self.oscillation_counter = 0
        self.prev_actions_log = deque(maxlen=5); self.max_q_table_size = max_q_table_size
        self.q_table_access_count: Dict[Tuple, int] = defaultdict(int)
        self.q_table_creation_time: Dict[Tuple, float] = {}
        self.flow_coefficient = 0.05; self.oscillation_penalty = 0.15; self.stability_reward_bonus = 0.05
        logger.info(f"QController initialized: alpha={self.alpha:.3f}, gamma={self.gamma:.3f}, epsilon={self.epsilon:.3f}, decay={self.epsilon_decay:.5f}, min_eps={self.min_epsilon:.3f}")
        logger.info(f"QController action ranges: LR={self.action_ranges['lr_scale']}, Momentum={self.action_ranges['momentum_scale']}")
        logger.info(f"QController reward params: OscPenalty={self.oscillation_penalty:.3f}, StabBonus={self.stability_reward_bonus:.3f}, FlowCoef={self.flow_coefficient:.3f}")

    def get_state(self, lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float]) -> Optional[Tuple]:
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm): logger.debug(f"Q-state calc skipped: Invalid input (L:{loss}, G:{grad_norm})"); return None
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm); self.lr_window.append(lr); self.momentum_window.append(momentum)
        if len(self.loss_window) < 3 or len(self.grad_norm_window) < 3: logger.debug("Q-state calc skipped: Insufficient history."); return None
        loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0
        try:
            y = np.array(list(self.loss_window)[-5:], dtype=np.float32); x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0] if len(y) >= 2 and len(np.unique(y)) > 1 else 0.0
            avg_loss = np.mean(y); normalized_slope = slope / (abs(avg_loss) + EPS)
            loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05]).item()
            avg_grad_norm = np.median(list(self.grad_norm_window))
            grad_norm_level_bin = np.digitize(avg_grad_norm, bins=[0.1, 0.5, 1.5, 5.0]).item()
            lr_level_bin = np.digitize(lr / 1e-4, bins=[0.5, 2.0, 10.0, 50.0]).item()
            momentum_level_bin = np.digitize(momentum, bins=[0.85, 0.92, 0.97]).item()
            if len(self.performance_window) >= 2:
                if (self.performance_window[-1] > 1e-2 and self.performance_window[-2] < -1e-2) or (self.performance_window[-1] < -1e-2 and self.performance_window[-2] > 1e-2): self.oscillation_counter = min(self.oscillation_counter + 1, 5)
                else: self.oscillation_counter = max(0, self.oscillation_counter - 1)
            oscillation_bin = 1 if self.oscillation_counter >= 3 else 0
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e: logger.warning(f"Q-state calc numerical error: {e}. Using default bins."); loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0
        except Exception as e_state: logger.error(f"Unexpected error during Q-state calc: {e_state}", exc_info=True); return None
        state = (loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin)
        self.q_table_access_count[state] += 1
        return state

    def compute_reward(self, current_loss: Optional[float], prev_loss: Optional[float], grad_norm: Optional[float]) -> float:
        if current_loss is None or prev_loss is None or grad_norm is None or not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm): logger.debug(f"Reward calc skipped: Invalid input (CurrL:{current_loss}, PrevL:{prev_loss}, GradN:{grad_norm})"); return 0.0
        loss_change = prev_loss - current_loss; loss_change_ratio = loss_change / (abs(prev_loss) + EPS)
        reward = np.tanh(loss_change_ratio * 10.0) # Reward based on relative loss change
        # Penalty for very large gradients
        if grad_norm > 5.0: reward -= 0.1 * min(1.0, max(0.0, (grad_norm - 5.0) / 10.0))
        # Small reward for very small gradients (encourages convergence?)
        elif grad_norm < 0.05: reward += 0.02
        # Penalty for oscillation
        if self.oscillation_counter >= 3: reward -= self.oscillation_penalty
        self.performance_window.append(reward)
        # Bonus for stability (consistent positive reward)
        if reward > 0.0: self.stable_steps += 1; reward += min(0.1, self.stability_reward_bonus * (self.stable_steps // 5))
        else: self.stable_steps = 0
        return float(np.clip(reward, -1.0, 1.0)) # Clip reward to [-1, 1]

    def choose_action(self, state: Optional[Tuple]) -> Dict[str, float]:
        if state is None: return {'lr_scale': 1.0, 'momentum_scale': 1.0} # Default action if state is invalid
        if state not in self.q_table:
            self.q_table[state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[state] = time.time(); self.q_table_access_count[state] = 1; self._manage_q_table_size()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay) # Decay epsilon
        chosen_actions = {}
        for param, q_values in self.q_table[state].items():
            action_space = self.action_ranges[param]
            if random.random() < self.epsilon: chosen_idx = random.randrange(len(action_space)) # Explore
            else: # Exploit
                finite_q_mask = np.isfinite(q_values)
                if not np.any(finite_q_mask): chosen_idx = random.randrange(len(action_space)) # Random if all Q are NaN/Inf
                else:
                    finite_q_values = q_values[finite_q_mask]; max_q = np.max(finite_q_values)
                    # Choose randomly among best actions if multiple have the same max Q-value
                    best_indices = np.where(np.isclose(q_values, max_q) & finite_q_mask)[0]
                    chosen_idx = np.random.choice(best_indices) if len(best_indices) > 0 else random.randrange(len(action_space)) # Fallback random
            chosen_actions[param] = float(action_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy())
        return chosen_actions

    def update(self, state: Optional[Tuple], action: Optional[Dict[str, float]], reward: float, next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None: logger.debug("Q-update skipped: Invalid state/action."); return
        if state not in self.q_table: logger.warning(f"QController: State {state} not found in Q-table during update. Skipping."); return
        if next_state not in self.q_table: # Initialize Q-values for new state if needed
            self.q_table[next_state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[next_state] = time.time(); self.q_table_access_count[next_state] = 0; self._manage_q_table_size()
        for param, chosen_value in action.items():
            action_space = self.action_ranges[param]
            # Find index of the chosen action
            action_indices = np.where(np.isclose(action_space, chosen_value))[0]
            if len(action_indices) == 0: logger.warning(f"Could not find action index for param {param}, value {chosen_value}. Skipping update."); continue
            action_idx = action_indices[0]
            # Q-learning update rule
            current_q = self.q_table[state][param][action_idx]
            next_q_values = self.q_table[next_state][param]
            finite_next_q = next_q_values[np.isfinite(next_q_values)]
            max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0 # Max Q-value for the next state
            if not np.isfinite(max_future_q): max_future_q = 0.0 # Handle NaN/Inf future Q
            td_target = reward + self.gamma * max_future_q # TD target
            td_error = td_target - current_q # TD error
            # Adaptive learning rate based on TD error magnitude (optional)
            adaptive_alpha = min(0.5, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error))))
            new_q = current_q + adaptive_alpha * td_error
            # Update Q-table, ensuring value is finite
            if np.isfinite(new_q): self.q_table[state][param][action_idx] = np.clip(new_q, -1e5, 1e5) # Clip Q-values
            else: logger.warning(f"Non-finite new Q-value calculated for state {state}, param {param}, action {action_idx}. Resetting to 0."); self.q_table[state][param][action_idx] = 0.0

    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            num_to_remove = len(self.q_table) - self.max_q_table_size
            logger.info(f"Q-table size ({len(self.q_table)}) exceeds max ({self.max_q_table_size}). Pruning {num_to_remove} states.")
            try:
                if not self.q_table_access_count or not self.q_table_creation_time: # Check if metadata is available
                    logger.warning("Q-table metadata incomplete. Removing random states.")
                    states_to_remove = random.sample(list(self.q_table.keys()), min(num_to_remove, len(self.q_table)))
                else:
                    # Prune based on least accessed and oldest states
                    sorted_states = sorted(self.q_table.keys(), key=lambda s: (self.q_table_access_count.get(s, 0), self.q_table_creation_time.get(s, float('inf'))))
                    states_to_remove = sorted_states[:num_to_remove]
                for state_to_remove in states_to_remove:
                    self.q_table.pop(state_to_remove, None)
                    self.q_table_access_count.pop(state_to_remove, None)
                    self.q_table_creation_time.pop(state_to_remove, None)
                logger.info(f"Pruned {len(states_to_remove)} states. New Q-table size: {len(self.q_table)}")
            except Exception as e:
                logger.warning(f"Error during Q-table pruning: {e}. Fallback random removal.", exc_info=False)
                current_keys = list(self.q_table.keys()); num_to_remove = max(0, len(current_keys) - self.max_q_table_size)
                if num_to_remove > 0:
                    states_to_remove = random.sample(current_keys, min(num_to_remove, len(current_keys)))
                    for state_to_remove in states_to_remove:
                        self.q_table.pop(state_to_remove, None); self.q_table_access_count.pop(state_to_remove, None); self.q_table_creation_time.pop(state_to_remove, None)
                    logger.info(f"Fallback pruned {len(states_to_remove)} random states. New Q-table size: {len(self.q_table)}")

    def get_info(self) -> Dict:
        last_action = self.prev_actions_log[-1] if self.prev_actions_log else None
        avg_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.0
        return {"epsilon": self.epsilon, "stable_steps": self.stable_steps, "oscillation_counter": self.oscillation_counter, "q_table_size": len(self.q_table), "last_action": last_action, "avg_reward_last_20": avg_reward}

# =====================================================================
# END: HAKMEM Components
# =====================================================================


# =====================================================================
# START: Data Handling Classes (Re-integrated)
# =====================================================================
class ByteTokenizer:
    """Simple stateless tokenizer for converting between text and utf-8 byte sequences."""
    def encode(self, text: str) -> List[int]: return list(text.encode('utf-8', errors='replace'))
    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        valid_bytes = []
        for b in byte_sequence:
            try:
                val = b.item() if isinstance(b, torch.Tensor) else int(b)
            except Exception:
                continue
            if 0 <= val <= 255:
                valid_bytes.append(val)
        return bytes(valid_bytes).decode('utf-8', errors='replace')

class ByteIterableDataset(IterableDataset):
    """ IterableDataset for reading byte sequences from a NumPy (.npy) file. """
    def __init__(self, npy_file_path: str, context_size: int = 256, data_fraction: float = 1.0):
        if not os.path.exists(npy_file_path): raise FileNotFoundError(f"Dataset file not found: {npy_file_path}")
        if context_size <= 0: raise ValueError("context_size must be positive")
        if not (0.0 < data_fraction <= 1.0): raise ValueError("data_fraction must be between 0 and 1")
        self.npy_file_path = npy_file_path; self.context_size = context_size; self.data_fraction = data_fraction
        self.full_data_size = 0; self.data_size = 0; self.num_possible_samples = 0; self.data_dtype = np.uint8
        self.seed = None; self.epoch = 0
        try:
            # Determine shape and size without loading full file
            with open(self.npy_file_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format._read_array_header(f, version)
            if len(shape) != 1: raise ValueError(f"Dataset must be 1D, found {shape}")
            self.full_data_size = shape[0]; self.data_dtype = dtype
            if self.data_dtype != np.uint8: logger.warning(f"Dataset dtype is {self.data_dtype}, expected uint8.")
            if self.full_data_size == 0: raise ValueError("Dataset file empty.")
            self.data_size = int(self.full_data_size * self.data_fraction)
            if self.data_size <= self.context_size: raise ValueError(f"Effective data size ({self.data_size:,}) <= context size ({self.context_size:,}). No samples.")
            # Need context_size + 1 bytes for one sample (context + target)
            self.num_possible_samples = max(0, self.data_size - self.context_size)
            if self.num_possible_samples == 0: raise ValueError(f"No samples possible with Ctx={self.context_size:,} and EffSize={self.data_size:,}.")
            logger.info(f"Dataset '{os.path.basename(npy_file_path)}': EffSize={self.data_size:,}/{self.full_data_size:,} ({self.data_fraction:.1%}), Samples={self.num_possible_samples:,}, DType={self.data_dtype}")
        except ImportError: logger.error("NumPy required for ByteIterableDataset."); raise
        except Exception as e: logger.error(f"Error reading dataset metadata from {npy_file_path}: {e}", exc_info=True); raise

    def __len__(self):
        if self.num_possible_samples == 0: return 0
        worker_info = torch.utils.data.get_worker_info(); num_workers = worker_info.num_workers if worker_info else 1
        world_size = get_world_size() if is_initialized() else 1; total_effective_workers = num_workers * world_size
        # Calculate length per worker, ensuring it's at least 0
        return max(0, self.num_possible_samples // total_effective_workers)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info(); num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0; rank = get_rank() if is_initialized() else 0
        world_size = get_world_size() if is_initialized() else 1
        if self.num_possible_samples == 0: return iter([])
        total_effective_workers = num_workers * world_size; global_worker_id = rank * num_workers + worker_id
        num_samples_per_global_worker = self.num_possible_samples // total_effective_workers
        remainder = self.num_possible_samples % total_effective_workers
        start_sample_idx = global_worker_id * num_samples_per_global_worker + min(global_worker_id, remainder)
        end_sample_idx = start_sample_idx + num_samples_per_global_worker + (1 if global_worker_id < remainder else 0)
        end_sample_idx = min(end_sample_idx, self.num_possible_samples)
        bytes_data = None; mmap_handle = None
        try:
            bytes_data = np.load(self.npy_file_path, mmap_mode='r')
            if hasattr(bytes_data, '_mmap') and bytes_data._mmap is not None: mmap_handle = bytes_data._mmap
            if bytes_data is None or bytes_data.size == 0: logger.error(f"Worker {worker_id} Rank {rank}: Failed load/empty data."); return iter([])
            if start_sample_idx >= end_sample_idx: logger.debug(f"Worker {worker_id} Rank {rank}: No samples assigned."); return iter([])
            worker_indices = np.arange(start_sample_idx, end_sample_idx, dtype=np.int64)
            base_seed = self.seed if self.seed is not None else int(time.time()); current_epoch = self.epoch
            seed_for_worker = (base_seed + global_worker_id + current_epoch) % (2**32)
            rng = np.random.default_rng(seed=seed_for_worker); rng.shuffle(worker_indices)
            logger.debug(f"Worker {worker_id} Rank {rank}: Processing indices {start_sample_idx}-{end_sample_idx-1} (Seed {seed_for_worker})")
            for idx in worker_indices:
                start_ctx = idx; end_ctx = idx + self.context_size; end_tgt = end_ctx + 1 # Target is shifted by 1
                if end_tgt > self.data_size: logger.warning(f"W:{worker_id} R:{rank}: Index {idx} out-of-bounds (Need up to {end_tgt}, have {self.data_size}). Skipping."); continue
                try:
                    context_slice = bytes_data[start_ctx : end_ctx];
                    target_slice = bytes_data[start_ctx + 1 : end_tgt] # Target is the next byte for each context byte
                    if len(context_slice) != self.context_size or len(target_slice) != self.context_size: logger.warning(f"W:{worker_id} R:{rank}: Slice length mismatch idx {idx}. Ctx={len(context_slice)}, Tgt={len(target_slice)}. Skip."); continue
                    # Use copy() to avoid issues with mmap/readahead/multiprocessing
                    yield torch.tensor(context_slice.copy(), dtype=torch.long), torch.tensor(target_slice.copy(), dtype=torch.long)
                except IndexError: logger.warning(f"W:{worker_id} R:{rank}: IndexError idx {idx}. Skip."); continue
                except Exception as e: logger.error(f"W:{worker_id} R:{rank}: Error processing idx {idx}: {e}"); continue
        except FileNotFoundError: logger.error(f"W:{worker_id} R:{rank}: Dataset file not found: {self.npy_file_path}")
        except Exception as e: logger.error(f"W:{worker_id} R:{rank}: Iterator setup/loop failed: {e}", exc_info=True)
        finally:
            if mmap_handle is not None:
                try: mmap_handle.close()
                except Exception as close_err: logger.warning(f"W:{worker_id} R:{rank}: Error closing mmap: {close_err}")
            del bytes_data; gc.collect() # Explicit cleanup attempt

    def set_seed(self, seed: int): self.seed = seed
    def set_epoch(self, epoch: int): self.epoch = epoch

def seed_worker(worker_id: int, base_seed: int, rank_offset: int):
    """Sets the seed for a dataloader worker."""
    worker_seed = base_seed + rank_offset + worker_id
    np.random.seed(worker_seed); random.seed(worker_seed)
    # torch.manual_seed(worker_seed) # Typically not needed/recommended here

# =====================================================================
# END: Data Handling Classes
# =====================================================================

# =====================================================================
# START: Custom Hyperbolic Geometry Implementation
# =====================================================================
# ... (Manifold, PoincareBall, get_manifold definitions - unchanged from previous correct version) ...
# --- Manifold Base Class ---
class Manifold:
    """Abstract base class for manifolds."""
    def __init__(self): pass
    def dist(self, x, y, keepdim=False): raise NotImplementedError
    def sqdist(self, x, y, keepdim=False): raise NotImplementedError
    def egrad2rgrad(self, p, dp): raise NotImplementedError
    def proj(self, p, dp): raise NotImplementedError # Project vector dp onto tangent space at p
    def proju(self, p): raise NotImplementedError # Project point p onto the manifold
    def expmap(self, p, dp): raise NotImplementedError # Exponential map
    def logmap(self, p, y): raise NotImplementedError # Logarithmic map
    def expmap0(self, dp): raise NotImplementedError # Exp map from origin
    def logmap0(self, p): raise NotImplementedError # Log map to origin
    def mobius_add(self, x, y): raise NotImplementedError
    def mobius_matvec(self, m, x): raise NotImplementedError
    def init_weights(self, w, irange=1e-5): raise NotImplementedError
    def zero_grad(self, p): p.grad.data.zero_()
    def normalize(self, p): return self.proju(p) # Default normalize is projection
    def check_point_on_manifold(self, p, atol=1e-5): raise NotImplementedError
    def check_vector_on_tangent(self, p, dp, atol=1e-5): raise NotImplementedError

# --- Poincaré Ball Manifold ---
class PoincareBall(Manifold):
    """Poincaré Ball manifold class."""
    def __init__(self, c=1.0):
        super().__init__()
        if isinstance(c, torch.Tensor):
            self.c = c.item() # Store scalar value
        elif not isinstance(c, (float, int)):
             raise TypeError(f"Curvature c must be a float or int, got {type(c)}")
        else:
            self.c = float(c)

        if self.c <= 0:
            logger.warning(f"PoincareBall initialized with non-positive curvature c={self.c}. Operations may behave like Euclidean.")
            self.k = 0.0 # Treat as Euclidean for safety? Or allow negative curvature interpretation?
            self.sqrt_c = 0.0
            self.max_norm = float('inf')
        else:
            self.k = -self.c # Curvature is -c for Poincaré ball
            self.sqrt_c = math.sqrt(self.c)
            # Calculate max norm slightly inside the boundary
            self.max_norm = (1.0 / self.sqrt_c) * (1.0 - EPS)

        self.min_norm = EPS # Minimum norm for safety
        self.name = f'PoincareBall(c={self.c:.3g})' # Use general format for c

    def _check_c(self, require_positive=True):
        if require_positive and self.c <= 0:
            raise ValueError(f"{self.name}: This operation requires positive curvature c > 0.")

    def lambda_x(self, x, keepdim=False):
        self._check_c(require_positive=True)
        # Conformality factor: λ_x = 2 / (1 - c ||x||^2)
        x_norm_sq = torch.sum(x.data.pow(2), dim=-1, keepdim=keepdim)
        # Clamp denominator: ensure 1 - c*norm_sq > EPS
        denominator = torch.clamp(1. - self.c * x_norm_sq, min=EPS)
        return 2. / denominator

    def sqdist(self, x, y, keepdim=False):
        if self.c <= 0: # Euclidean distance for c <= 0
            diff_norm_sq = torch.sum((x - y).pow(2), dim=-1, keepdim=keepdim)
            return torch.clamp(diff_norm_sq, min=0.0) # Ensure non-negative

        self._check_c(require_positive=True)
        # Clamp inputs to be strictly inside the ball for distance calculation
        x_proj = self.proju(x)
        y_proj = self.proju(y)

        diff_norm_sq = torch.sum((x_proj - y_proj).pow(2), dim=-1, keepdim=keepdim)
        x_norm_sq = torch.sum(x_proj.pow(2), dim=-1, keepdim=keepdim)
        y_norm_sq = torch.sum(y_proj.pow(2), dim=-1, keepdim=keepdim)

        # Clamp individual denominators for stability before multiplying
        denom_x = torch.clamp(1. - self.c * x_norm_sq, min=EPS)
        denom_y = torch.clamp(1. - self.c * y_norm_sq, min=EPS)
        denominator_product = denom_x * denom_y

        # Ensure arcosh argument is >= 1.0
        arcosh_arg = 1. + 2. * self.c * diff_norm_sq / (denominator_product + EPS) # Add eps to product
        arcosh_arg = torch.clamp(arcosh_arg, min=1.0 + EPS) # Clamp slightly above 1

        # dist = (1/sqrt(c)) * acosh(...)
        # sqdist = (1/c) * acosh(...)^2
        # Use float32 for acosh calculation stability
        acosh_val = torch.acosh(arcosh_arg.float())
        sq_dist_val = (1.0 / self.c) * acosh_val.pow(2)

        return sq_dist_val.to(x.dtype) # Cast back to original type

    def dist(self, x, y, keepdim=False):
        if self.c <= 0: # Euclidean distance for c <= 0
            diff_norm = torch.norm(x - y, p=2, dim=-1, keepdim=keepdim)
            return torch.clamp(diff_norm, min=0.0)

        self._check_c(require_positive=True)
        # Avoid sqrt(neg) by calculating sqdist first
        sq_dist = self.sqdist(x, y, keepdim=keepdim)
        # Add epsilon before sqrt for stability
        return torch.sqrt(torch.clamp(sq_dist, min=0) + EPS)

    def proju(self, x):
        """Project point x onto the Poincaré Ball."""
        if self.c <= 0: return x # No projection needed if not hyperbolic
        if not torch.is_tensor(x): # Handle non-tensor input if necessary
            logger.warning(f"proju received non-tensor input type {type(x)}. Returning.")
            return x

        # Project potentially non-finite values first? Or assume input is finite?
        if not torch.isfinite(x).all():
             logger.warning(f"Non-finite values detected in proju input. Clamping.")
             # Use maximum allowed norm for infinity replacement if c > 0
             inf_replace = self.max_norm if self.c > 0 else 1e6 # Large finite number if Euclidean-like
             x = torch.nan_to_num(x, nan=0.0, posinf=inf_replace, neginf=-inf_replace)

        d = x.size(-1)
        x_norm = torch.norm(x.data, p=2, dim=-1, keepdim=True)

        # Calculate scaling factor based on max_norm
        scale = torch.where(x_norm >= self.max_norm, self.max_norm / (x_norm + EPS), torch.ones_like(x_norm))
        # Apply scaling
        projected_x = x * scale
        return projected_x

    def proj(self, p, dp):
        """Project vector dp onto the tangent space T_p H^n_c."""
        # For Poincaré ball, the tangent space at any point p is R^n,
        # so the projection is the identity map.
        return dp

    def expmap(self, p, dp):
        """Exponential map from tangent vector dp at point p."""
        if self.c <= 0: return p + dp # Euclidean exponential map

        self._check_c(require_positive=True)
        p = self.proju(p) # Ensure p is on manifold
        # Use float32 for norm calculation for stability
        dp_norm = torch.norm(dp.float(), p=2, dim=-1, keepdim=True).clamp(min=EPS)
        lambda_p = self.lambda_x(p, keepdim=True) # 2 / (1 - c ||p||^2)

        # tanh argument: sqrt(c) * lambda_p * ||dp|| / 2
        tanh_arg = self.sqrt_c * lambda_p * dp_norm / 2.
        # Clamp tanh input to avoid overflow
        tanh_arg_clamped = torch.clamp(tanh_arg, min=-30.0, max=30.0)

        # Result: mobius_add(p, tanh(arg) * dp / (sqrt(c) * ||dp||))
        # Calculate factor in float32, cast back if needed
        factor = torch.tanh(tanh_arg_clamped) / (self.sqrt_c * dp_norm + EPS)

        # Ensure dp and factor have compatible types for multiplication
        exp_res = self.mobius_add(p, factor.to(p.dtype) * dp)
        return self.proju(exp_res) # Project back just in case

    def logmap(self, p, y):
        """Logarithmic map from point y to tangent space at point p."""
        if self.c <= 0: return y - p # Euclidean log map

        self._check_c(require_positive=True)
        p = self.proju(p)
        y = self.proju(y)
        # mobius_add(-p, y)
        sub = self.mobius_add(-p, y)
        # Use float32 for norm calculation
        sub_norm = torch.norm(sub.float(), p=2, dim=-1, keepdim=True).clamp(min=EPS)
        lambda_p = self.lambda_x(p, keepdim=True)

        # atanh argument: sqrt(c) * ||sub||
        atanh_arg = self.sqrt_c * sub_norm
        # Clamp atanh input: (-1 + eps, 1 - eps)
        atanh_arg_clamped = torch.clamp(atanh_arg, min=-1.0 + EPS, max=1.0 - EPS)

        # Result: (2 / (sqrt(c) * lambda_p)) * atanh(arg) * sub / ||sub||
        # Calculate factor in float32, cast back if needed
        factor = (2. / (self.sqrt_c * lambda_p + EPS)) * torch.atanh(atanh_arg_clamped.float()) / (sub_norm + EPS)
        return factor.to(p.dtype) * sub

    def expmap0(self, dp):
        """Exponential map from tangent vector dp at the origin."""
        if self.c <= 0: return dp # Euclidean exp map from origin

        self._check_c(require_positive=True)
        # Use float32 for norm calculation
        dp_norm = torch.norm(dp.float(), p=2, dim=-1, keepdim=True).clamp(min=EPS)
        tanh_arg = self.sqrt_c * dp_norm
        # Clamp tanh input
        tanh_arg_clamped = torch.clamp(tanh_arg, min=-30.0, max=30.0)
        # Result: tanh(sqrt(c)||dp||) * dp / (sqrt(c)||dp||)
        # Calculate factor in float32, cast back if needed
        factor = torch.tanh(tanh_arg_clamped.float()) / (self.sqrt_c * dp_norm + EPS)
        exp0_res = factor.to(dp.dtype) * dp
        return self.proju(exp0_res) # Project to ensure it's in the ball

    def logmap0(self, p):
        """Logarithmic map from point p to the tangent space at the origin."""
        if self.c <= 0: return p # Euclidean log map to origin

        self._check_c(require_positive=True)
        p = self.proju(p)
        # Use float32 for norm calculation
        p_norm = torch.norm(p.float(), p=2, dim=-1, keepdim=True).clamp(min=EPS)
        atanh_arg = self.sqrt_c * p_norm
        # Clamp atanh input: (-1 + eps, 1 - eps)
        atanh_arg_clamped = torch.clamp(atanh_arg, min=-1.0 + EPS, max=1.0 - EPS)
        # Result: atanh(sqrt(c)||p||) * p / (sqrt(c)||p||)
        # Calculate factor in float32, cast back if needed
        factor = torch.atanh(atanh_arg_clamped.float()) / (self.sqrt_c * p_norm + EPS)
        return factor.to(p.dtype) * p

    def mobius_add(self, x, y):
        """Möbius addition: x ⊕ y."""
        if self.c <= 0: return x + y # Euclidean addition

        self._check_c(require_positive=True)
        x_norm_sq = torch.sum(x.pow(2), dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y.pow(2), dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

        # Numerator: (1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y
        num_factor_x = 1. + 2. * self.c * xy_dot + self.c * y_norm_sq
        num_factor_y = 1. - self.c * x_norm_sq
        numerator = num_factor_x * x + num_factor_y * y

        # Denominator: 1 + 2c<x,y> + c^2||x||^2||y||^2
        denominator = 1. + 2. * self.c * xy_dot + self.c**2 * x_norm_sq * y_norm_sq
        # Clamp denominator to avoid division by zero
        denominator = torch.clamp(denominator, min=EPS)

        # Result needs projection for stability
        res = numerator / denominator
        return self.proju(res)

    def mobius_scalar_mul(self, r, x):
        """Möbius scalar multiplication: r ⊗ x."""
        if self.c <= 0: return r * x # Euclidean scalar mult

        self._check_c(require_positive=True)
        x = self.proju(x)
        # Use float32 for norm calculation
        x_norm = torch.norm(x.float(), p=2, dim=-1, keepdim=True).clamp(min=EPS)
        # Clamp atanh input
        atanh_arg = torch.clamp(self.sqrt_c * x_norm, min=-1.0 + EPS, max=1.0 - EPS)
        # Calculate tanh in float32
        tanh_term = torch.tanh(r * torch.atanh(atanh_arg.float()))
        # Calculate result norm in float32
        res_norm = tanh_term / (self.sqrt_c + EPS)
        res = res_norm.to(x.dtype) * (x / (x_norm.to(x.dtype) + EPS)) # Add EPS to norm divisor
        return self.proju(res)

    def mobius_matvec(self, M, x):
        """Möbius matrix-vector multiplication (approximated via tangent space)."""
        if self.c <= 0: return F.linear(x, M) # Euclidean matvec

        self._check_c(require_positive=True)
        x_log = self.logmap0(x) # Map to tangent space T_0
        # Apply linear transformation in tangent space
        Mx_log = F.linear(x_log, M) # Assumes M is a standard nn.Linear weight matrix
        Mx_hyp = self.expmap0(Mx_log) # Map back to hyperbolic space H
        return self.proju(Mx_hyp)

    def egrad2rgrad(self, p, dp):
        """Convert Euclidean gradient dp to Riemannian gradient at p."""
        if self.c <= 0: return dp # Euclidean gradient is Riemannian gradient

        self._check_c(require_positive=True)
        lambda_p_sq = self.lambda_x(p, keepdim=True).pow(2)
        # Riemannian gradient = (1 / lambda_p^2) * dp = ((1 - c||p||^2)/2)^2 * dp
        factor = ((1. - self.c * torch.sum(p.pow(2), dim=-1, keepdim=True)) / 2.).pow(2)
        # Clamp factor to prevent explosion? Or rely on gradient clipping?
        factor = torch.clamp(factor, max=1e4) # Add clamping for stability
        return factor * dp

    def init_weights(self, w, irange=1e-5):
        """Initialize weights for tangent space operations (e.g., in GyroLinear)."""
        # Uses standard PyTorch initialization, suitable for tangent space linear layers
        w.data.uniform_(-irange, irange)

    def check_point_on_manifold(self, p, atol=1e-5):
        if self.c <= 0: return True # All points are on Euclidean manifold
        norm_sq = torch.sum(p.pow(2), dim=-1)
        # Check against squared max_norm for consistency
        return torch.all(norm_sq <= self.max_norm**2 + atol)

    def check_vector_on_tangent(self, p, dp, atol=1e-5):
        # Tangent space is R^n for Poincare ball, so any vector is valid
        # (Assuming dp has the correct dimension)
        return True

# --- Helper to get manifold object (can be extended for other manifolds) ---
def get_manifold(name="poincare", curvature=1.0) -> Manifold:
    if name.lower() == "poincare":
        return PoincareBall(c=curvature)
    # elif name.lower() == "euclidean": # Example extension
    #     return EuclideanManifold()
    else:
        raise ValueError(f"Unknown manifold: {name}")

# =====================================================================
# END: Custom Hyperbolic Geometry Implementation
# =====================================================================


# =====================================================================
# Helper Function for Weight Initialization (Apply to Euclidean modules)
# =====================================================================
def init_weights(module):
    """Initialize weights for standard Linear and Embedding layers."""
    if isinstance(module, nn.Linear):
        # Avoid initializing GyroLinear bias parameter here if it's hyperbolic
        is_gyro_bias = hasattr(module, 'b') and module.b is module.bias and hasattr(module.bias, 'manifold') and module.bias.manifold is not None
        if not is_gyro_bias:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
         # Avoid initializing HyperbolicEmbedding weights here
         # Check if it's a HyperbolicEmbedding instance OR has a manifold attribute
         is_hyperbolic_embedding = isinstance(module, HyperbolicEmbedding) or (hasattr(module.weight, 'manifold') and module.weight.manifold is not None)
         if not is_hyperbolic_embedding:
             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        # Avoid initializing RiemannianLayerNorm params here
        is_riemannian_ln = isinstance(module, RiemannianLayerNorm) or (hasattr(module, 'manifold') and module.manifold is not None)
        if not is_riemannian_ln:
             if module.elementwise_affine:
                 torch.nn.init.ones_(module.weight)
                 torch.nn.init.zeros_(module.bias)


# =====================================================================
# START: Hyperbolic Layers Implementation
# =====================================================================

# --- Hyperbolic Embedding ---
class HyperbolicEmbedding(nn.Module):
    """Embedding layer that maps indices to points in a Poincaré Ball."""
    def __init__(self, num_embeddings, embedding_dim, manifold: PoincareBall, sparse=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if not isinstance(manifold, PoincareBall):
             raise TypeError("HyperbolicEmbedding requires a PoincareBall manifold instance.")
        self.manifold = manifold
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        # Add manifold attribute for optimizer
        self.weight.manifold = manifold
        self.sparse = sparse
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize near origin
        with torch.no_grad():
            irange = 1e-5 # Initialize very close to origin
            self.weight.uniform_(-irange, irange)
            # Project to ensure they are within the ball initially
            self.weight.data = self.manifold.proju(self.weight.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Ensure indices are within bounds
        input_clamped = torch.clamp(input, 0, self.num_embeddings - 1)
        # Perform embedding lookup
        embeddings = F.embedding(input_clamped, self.weight, sparse=self.sparse)
        # Project result onto the manifold (safety check)
        return self.manifold.proju(embeddings)


# --- GyroLinear Layer (using tangent space bridge) ---
class GyroLinear(nn.Module):
    """Hyperbolic Linear layer using tangent space bridge."""
    def __init__(self, in_features, out_features, manifold: PoincareBall, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if not isinstance(manifold, PoincareBall):
             raise TypeError("GyroLinear requires a PoincareBall manifold instance.")
        self.manifold = manifold
        self.bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Initialize weights for tangent space linear transform (Euclidean)
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if bias:
            # Bias is a point on the manifold, initialize near origin
            self.b = nn.Parameter(torch.Tensor(out_features))
            self.b.manifold = manifold # Mark for optimizer
            with torch.no_grad():
                irange = 1e-5
                self.b.uniform_(-irange, irange)
                self.b.data = self.manifold.proju(self.b.data) # Project initial bias
        else:
             self.register_parameter('b', None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input point x from manifold H_in to tangent space T_0 H_in
        x_tangent = self.manifold.logmap0(x)
        # Apply linear transformation in tangent space
        output_tangent = F.linear(x_tangent, self.weight)
        # Map result from tangent space T_0 H_out to manifold H_out
        output_hyperbolic = self.manifold.expmap0(output_tangent)
        # Apply Möbius bias addition if enabled
        if self.bias and self.b is not None:
            # Ensure bias is projected (important during training)
            bias_proj = self.manifold.proju(self.b)
            output_hyperbolic = self.manifold.mobius_add(output_hyperbolic, bias_proj)

        # Final projection for numerical stability
        return self.manifold.proju(output_hyperbolic)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, manifold={}'.format(
            self.in_features, self.out_features, self.bias, self.manifold.name
        )

# --- Riemannian Layer Normalization ---
class RiemannianLayerNorm(nn.Module):
    """Layer Normalization adapted for Riemannian manifolds via tangent space."""
    def __init__(self, normalized_shape, manifold: PoincareBall, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        if not isinstance(manifold, PoincareBall):
             raise TypeError("RiemannianLayerNorm requires a PoincareBall manifold instance.")
        self.manifold = manifold
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.Tensor(*normalized_shape)) # Euclidean scale
            self.beta = nn.Parameter(torch.Tensor(*normalized_shape))  # Euclidean shift (in tangent space)
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map input point x from manifold to tangent space T_0 H
        x_tangent = self.manifold.logmap0(x)

        # Compute mean and variance in tangent space along the normalization axis
        axis = -1 # Assume normalization over the feature dimension
        mean = torch.mean(x_tangent, dim=axis, keepdim=True)
        # Center the tangent vectors
        x_centered = x_tangent - mean
        # Compute variance of centered vectors
        variance = torch.mean(x_centered.pow(2), dim=axis, keepdim=True)

        # Normalize tangent vectors
        x_tangent_normalized = x_centered / torch.sqrt(variance + self.eps)

        # Apply affine transformation (if enabled) in tangent space
        if self.elementwise_affine:
            x_tangent_normalized = self.gamma * x_tangent_normalized + self.beta

        # Map normalized tangent vector back to the manifold H
        output_hyperbolic = self.manifold.expmap0(x_tangent_normalized)

        # Final projection for stability
        return self.manifold.proju(output_hyperbolic)

    def extra_repr(self):
         return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, manifold={manifold}'.format(
               normalized_shape=self.normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine, manifold=self.manifold.name)


# --- Hyperbolic Distance Attention (Hybrid Tangent Aggregation) ---
# NOTE: This remains highly experimental. Aggregating values represented as points
#       is non-trivial. The tangent space aggregation is a common simplification.
class HyperbolicDistanceAttention(nn.Module):
    """Attention mechanism using hyperbolic distance for similarity."""
    def __init__(self, embed_dim, num_heads, manifold: PoincareBall, dropout=0.1, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not isinstance(manifold, PoincareBall):
             raise TypeError("HyperbolicDistanceAttention requires a PoincareBall manifold instance.")
        self.manifold = manifold

        # Use GyroLinear for projections
        self.q_proj = GyroLinear(embed_dim, embed_dim, manifold, bias=bias)
        self.k_proj = GyroLinear(embed_dim, embed_dim, manifold, bias=bias)
        self.v_proj = GyroLinear(embed_dim, embed_dim, manifold, bias=bias) # Values are also points

        # Output projection (maps aggregated tangent vector back)
        self.out_proj = GyroLinear(embed_dim, embed_dim, manifold, bias=bias)

        self.dropout = nn.Dropout(dropout)
        # Learnable temperature parameter for scaling distance
        self.neg_dist_scale = nn.Parameter(torch.tensor(1.0)) # Initialize scale to 1

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            query, key, value: Points in the Poincaré ball [Batch, SeqLen, Dim]
            key_padding_mask: Mask for keys [Batch, SeqLen_K] (True means pad/mask)
            attn_mask: Additive mask for attention scores [Batch, NumHeads, SeqLen_Q, SeqLen_K] or [SeqLen_Q, SeqLen_K] (True means mask)
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # 1. Project Q, K, V to points using GyroLinear
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value) # Values are also points now

        # 2. Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, NumHeads, TgtLen, HeadDim]
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, NumHeads, SrcLen, HeadDim]
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # [B, NumHeads, SrcLen, HeadDim]

        # 3. Calculate pairwise hyperbolic distances
        # Expand dims for broadcasting distance calculation
        q_expanded = q.unsqueeze(3) # [B, NumHeads, TgtLen, 1, HeadDim]
        k_expanded = k.unsqueeze(2) # [B, NumHeads, 1, SrcLen, HeadDim]

        # Calculate squared distance for stability, then sqrt
        sq_dist = self.manifold.sqdist(q_expanded, k_expanded, keepdim=True) # [B, NumHeads, TgtLen, SrcLen, 1]
        dist = torch.sqrt(sq_dist.clamp(min=0) + EPS).squeeze(-1) # [B, NumHeads, TgtLen, SrcLen]

        # 4. Convert distance to similarity score
        # Use negative scaled distance: sim = -scale * dist
        attn_scores = -torch.abs(self.neg_dist_scale) * dist # Ensure scale is positive during use

        # 5. Apply masks (True means MASK)
        if attn_mask is not None:
            # Ensure mask is boolean and has correct shape [B, H, Q, K]
            attn_mask = attn_mask.to(dtype=torch.bool)
            if attn_mask.dim() == 2: # [TgtLen, SrcLen] -> [1, 1, TgtLen, SrcLen]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3: # [B, TgtLen, SrcLen] -> [B, 1, TgtLen, SrcLen]
                 attn_mask = attn_mask.unsqueeze(1)
            # Expand mask if needed
            expected_attn_mask_shape = (batch_size, self.num_heads, tgt_len, src_len)
            if attn_mask.shape != expected_attn_mask_shape:
                try: attn_mask = attn_mask.expand(expected_attn_mask_shape)
                except RuntimeError: attn_mask=None; logger.warning("Attention mask broadcast failed.")
            if attn_mask is not None: attn_scores = attn_scores.masked_fill(attn_mask, -1e9)

        if key_padding_mask is not None:
            # [B, SrcLen] -> [B, 1, 1, SrcLen] -> Expand for heads/queries
            mask = key_padding_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            expected_pad_mask_shape = (batch_size, 1, 1, src_len)
            if mask.shape != expected_pad_mask_shape: mask=None; logger.warning(f"Key padding mask shape incorrect. Got {mask.shape}, expected {expected_pad_mask_shape}")
            if mask is not None: attn_scores = attn_scores.masked_fill(mask, -1e9)

        # 6. Normalize scores using softmax
        attn_probs = F.softmax(attn_scores, dim=-1) # [B, NumHeads, TgtLen, SrcLen]
        # Handle potential NaNs from softmax if all inputs were masked
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.dropout(attn_probs)

        # 7. Aggregate Values (Hybrid Tangent Space Approach)
        # Map hyperbolic value points v to tangent space at origin
        v_tangent = self.manifold.logmap0(v) # [B, NumHeads, SrcLen, HeadDim]

        # Weighted sum in tangent space
        attn_output_tangent = torch.matmul(attn_probs, v_tangent) # [B, NumHeads, TgtLen, HeadDim]

        # 8. Reshape and project back
        # Reshape: [B, TgtLen, NumHeads * HeadDim] = [B, TgtLen, EmbedDim]
        attn_output_tangent_reshaped = attn_output_tangent.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)

        # Map aggregated tangent vector back to hyperbolic space (expmap0)
        # AND apply the final output projection using GyroLinear
        output_hyperbolic = self.out_proj(self.manifold.expmap0(attn_output_tangent_reshaped))

        return self.manifold.proju(output_hyperbolic) # Final projection

# =====================================================================
# END: Hyperbolic Layers Implementation
# =====================================================================


# =====================================================================
# START: Modified WuBu Nesting Core (Hyperbolic Attempt)
# =====================================================================

# --- Boundary Manifold (Storing Hyperbolic Points) ---
class BoundaryManifoldHyperbolic(nn.Module):
    """ Boundary points represented directly in the Poincaré Ball. """
    def __init__(self, level_idx: int, num_points: int, point_dim: int, manifold: PoincareBall): # Requires manifold at init!
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim
        if not isinstance(manifold, PoincareBall):
             raise ValueError("BoundaryManifoldHyperbolic requires a valid PoincareBall manifold instance during init.")
        self.manifold = manifold # Store the provided (initial) manifold

        if num_points > 0 and point_dim > 0:
            self.hyperbolic_points = nn.Parameter(torch.Tensor(num_points, point_dim))
            # Add manifold attribute for optimizer - using the initial manifold reference
            self.hyperbolic_points.manifold = self.manifold
            self.reset_parameters() # Now self.manifold is valid
            logger.info(f"BoundaryManifoldHyp L{level_idx}: {num_points} points in {point_dim}D {manifold.name}.")
        else:
            self.register_parameter('hyperbolic_points', None)
            logger.info(f"BoundaryManifoldHyp L{level_idx}: No boundary points (num={num_points}, dim={point_dim}).")

    def reset_parameters(self):
        if self.hyperbolic_points is not None:
            with torch.no_grad():
                # Initialize near origin
                irange = 1e-5 # Very close to origin
                self.hyperbolic_points.uniform_(-irange, irange)
                # self.manifold is now guaranteed to be a PoincareBall instance
                self.hyperbolic_points.data = self.manifold.proju(self.hyperbolic_points.data)

    def get_points(self) -> Optional[torch.Tensor]:
        """Returns the current boundary points (points in the Ball), ensuring projection."""
        if self.hyperbolic_points is None:
            return None
        # Ensure points stay on the manifold during training by projecting them
        # Critical: Use the *current* manifold reference stored in self.manifold
        # which should be updated by the HyperbolicWuBuNestingLevel forward pass.
        return self.manifold.proju(self.hyperbolic_points)


# --- Hyperbolic Inter-Level Transform (Tangent Space Bridge) ---
class HyperbolicInterLevelTransform(nn.Module):
    """ Transforms points between hyperbolic spaces via tangent space. """
    def __init__(self, in_dim: int, out_dim: int, manifold_in: PoincareBall, manifold_out: PoincareBall,
                 transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if not isinstance(manifold_in, PoincareBall) or not isinstance(manifold_out, PoincareBall):
             raise TypeError("HyperbolicInterLevelTransform requires PoincareBall manifold instances.")
        # Store the initial manifolds used for defining the transform layer
        self.manifold_in_init = manifold_in
        self.manifold_out_init = manifold_out
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transform_type = transform_type

        # Euclidean transform operating in tangent space
        if transform_type == 'mlp':
            mlp_hidden_dim = hidden_dim if hidden_dim is not None and hidden_dim > 0 else max(16, (in_dim + out_dim) // 2)
            self.tangent_transform = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim), # Standard LayerNorm in tangent space
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, out_dim)
            )
            logger.info(f"HypInterLevelTransform ({in_dim}->{out_dim}): MLP (Hidden: {mlp_hidden_dim}) in tangent")
        elif transform_type == 'linear':
            self.tangent_transform = nn.Linear(in_dim, out_dim)
            logger.info(f"HypInterLevelTransform ({in_dim}->{out_dim}): Linear in tangent")
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")

        self.apply(init_weights) # Initialize Euclidean weights of the transform

    def forward(self, point_in: torch.Tensor,
                boundaries_in: Optional[torch.Tensor],
                descriptor_in: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Transforms points from manifold_in to manifold_out.
        Args:
            point_in: Points on the input manifold (H_i) [..., in_dim]
            boundaries_in: Boundary points on H_i [num_points, in_dim] or None
            descriptor_in: Descriptor point on H_i [1, 1, in_dim] or None
        Returns:
            Tuple of transformed points, theoretically on H_{i+1}, BUT need projection
            onto the *actual* manifold_next in the main model forward pass.
        """
        # 1. Map inputs from H_i to tangent space T_0 H_i using the *initial* input manifold
        # This is a simplification; assumes the tangent space structure defined initially is sufficient
        current_manifold_in = self.manifold_in_init # Use initial for consistency within transform

        tangent_main = current_manifold_in.logmap0(point_in)
        tangent_boundaries = current_manifold_in.logmap0(boundaries_in) if boundaries_in is not None else None
        tangent_descriptor = current_manifold_in.logmap0(descriptor_in) if descriptor_in is not None else None

        # 2. Apply Euclidean transformation in tangent space
        tangent_main_out = self.tangent_transform(tangent_main)
        tangent_boundaries_out = self.tangent_transform(tangent_boundaries) if tangent_boundaries is not None else None
        tangent_descriptor_out = self.tangent_transform(tangent_descriptor) if tangent_descriptor is not None else None

        # 3. Map results from tangent space to the *initial* output manifold H_{i+1}_init
        # The result will then be projected onto the *actual* H_{i+1} by the caller.
        current_manifold_out = self.manifold_out_init

        point_out = current_manifold_out.expmap0(tangent_main_out)
        boundaries_out = current_manifold_out.expmap0(tangent_boundaries_out) if tangent_boundaries_out is not None else None
        descriptor_out = current_manifold_out.expmap0(tangent_descriptor_out) if tangent_descriptor_out is not None else None

        # DO NOT project here. Projection happens in the main model using the *actual* next manifold.
        return point_out, boundaries_out, descriptor_out


# --- Hyperbolic WuBu Nesting Level ---
class HyperbolicWuBuNestingLevel(nn.Module):
    """ Single level of the WuBu Nesting architecture operating in hyperbolic space. """
    def __init__(self, level_idx: int, dim: int, config: Dict, initial_curvature: float): # Added initial_curvature
        super().__init__()
        self.level_idx = level_idx
        self.dim = dim
        self.config = config
        self.initial_curvature = initial_curvature # Store initial curvature

        # Extract relevant config values
        self.use_ld = config.get("use_level_descriptors", True)
        self.use_spread = config.get("use_level_spread", True)
        self.dropout = config.get("dropout", 0.1)
        self.ld_init_scale = config.get("level_descriptor_init_scale", 0.01)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean")

        self.min_curvature = config.get("curvature_min_value", EPS)
        self.min_scale = config.get("scale_min_value", EPS)
        self.min_spread = config.get("spread_min_value", EPS)

        # --- Constrained Parameters (Curvature, Scale, Spread - Euclidean) ---
        def _init_constrained_param(value, min_val):
             y = max(float(value), min_val + EPS) - min_val
             try:
                 unconstrained_val = math.log(math.expm1(y)) if y >= 1e-6 else math.log(y + EPS)
             except (ValueError, OverflowError):
                 logger.error(f"Error init constrained param: val={value}, min={min_val}, y={y}. Fallback.")
                 unconstrained_val = math.log(EPS)
             return torch.tensor(unconstrained_val, dtype=torch.float)

        # Use initial curvature passed to constructor
        init_c = self.initial_curvature
        self.log_curvature = nn.Parameter(_init_constrained_param(init_c, self.min_curvature))
        init_s = config["initial_scales"][level_idx]
        self.log_scale = nn.Parameter(_init_constrained_param(init_s, self.min_scale))
        # Spread parameter (scalar)
        init_spread = config["initial_spread_values"][level_idx]
        # Register buffer if not learnable (prevents it from being optimized)
        spread_param_value = _init_constrained_param(init_spread, self.min_spread) if self.use_spread else _init_constrained_param(self.min_spread + EPS, self.min_spread)
        learnable_spread_flag = config.get("learnable_spread", True) and self.use_spread
        if learnable_spread_flag:
             self.log_spread = nn.Parameter(spread_param_value)
        else:
             self.register_buffer('log_spread', spread_param_value)


        # --- Manifold (created dynamically based on current curvature) ---
        # We need an initial manifold for boundary setup
        initial_manifold = PoincareBall(c=self.initial_curvature)

        # --- Level Descriptor (Point on the Manifold) ---
        self.level_descriptor_param = nn.Parameter(torch.Tensor(dim))
        self.level_descriptor_param.manifold = None # Set in forward pass to current manifold
        with torch.no_grad():
            self.level_descriptor_param.uniform_(-self.ld_init_scale, self.ld_init_scale)
            # No need to project here, let forward handle with current curvature

        # --- Boundary Manifold (Pass initial manifold) ---
        self.num_boundaries = config["boundary_points_per_level"][level_idx]
        self.boundary_manifold = BoundaryManifoldHyperbolic(
            level_idx, self.num_boundaries, dim, manifold=initial_manifold # <<< Pass initial manifold here
        )

        # --- Hyperbolic Combiner (Tangent Space Aggregation) ---
        combiner_input_dim = self.dim # Main tangent
        if self.relative_vector_aggregation != 'none': combiner_input_dim += self.dim # Aggregated relative tangent
        if self.use_ld: combiner_input_dim += self.dim # Descriptor tangent
        if self.use_spread: combiner_input_dim += 1 # Spread value
        combiner_hidden_dims = config.get("tangent_input_combination_dims", [max(16, combiner_input_dim // 2)])
        layers = []; in_d = combiner_input_dim
        for h_dim in combiner_hidden_dims:
            layers.extend([ nn.Linear(in_d, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(self.dropout) ]); in_d = h_dim
        layers.append(nn.Linear(in_d, self.dim))
        self.tangent_combiner = nn.Sequential(*layers) # Operates in tangent space

        # --- Optional Tangent Flow (Remains Euclidean) ---
        self.use_flow = config.get("use_tangent_flow", True)
        self.tangent_flow = None; self.flow_scale = 0.0
        if self.use_flow:
             flow_hidden_dim = max(16, int(dim * config.get("tangent_flow_hidden_dim_ratio", 0.5)))
             flow_type = config.get("tangent_flow_type", "mlp")
             if flow_type == 'mlp': self.tangent_flow = nn.Sequential( nn.Linear(dim, flow_hidden_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(flow_hidden_dim, dim) )
             elif flow_type == 'linear': self.tangent_flow = nn.Linear(dim, dim)
             elif flow_type != 'none': logger.warning(f"Unsupported tangent_flow_type '{flow_type}', disabling flow."); self.use_flow=False
             if self.use_flow: self.flow_scale = config.get("tangent_flow_scale", 1.0)

        # Initialize Euclidean weights (combiner, flow)
        self.tangent_combiner.apply(init_weights)
        if self.tangent_flow: self.tangent_flow.apply(init_weights)

    def get_curvature(self) -> torch.Tensor: return F.softplus(self.log_curvature) + self.min_curvature
    def get_scale(self) -> torch.Tensor: return F.softplus(self.log_scale) + self.min_scale
    def get_spread(self) -> torch.Tensor:
        if not self.use_spread:
            # Ensure device/dtype match other params if returning fixed value
            param_device = self.log_curvature.device; param_dtype = self.log_curvature.dtype
            return torch.tensor(self.min_spread, device=param_device, dtype=param_dtype)
        # Use softplus + min for learnable or fixed spread parameter
        return F.softplus(self.log_spread) + self.min_spread

    def forward(self, point_in: torch.Tensor,
                relative_vectors_tangent_in: Optional[torch.Tensor], # Relative vectors from prev level (Tangent T_0)
                descriptor_point_in: Optional[torch.Tensor], # Descriptor from prev level (Point H_i)
                sigma_in: Optional[torch.Tensor] # Spread from prev level (Scalar)
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes points through the hyperbolic level.
        Args:
            point_in: Input points on *this* level's manifold H_i [B, S, Dim]
            relative_vectors_tangent_in: Aggregated tangent vectors (T_0 H_i) [B, S, Dim] or None
            descriptor_point_in: Descriptor point on H_i [B, S, Dim] or None
            sigma_in: Scalar spread from previous level [B, S, 1] or scalar tensor
        Returns:
            Tuple: (point_out, tangent_out, level_descriptor_point, boundaries_points, sigma_out)
                   point_out, level_descriptor_point, boundaries_points are points on H_i.
                   tangent_out is a vector in T_0 H_i.
                   sigma_out is a scalar.
        """
        batch_size, seq_len, d_in = point_in.shape
        device = point_in.device
        # Get dtype from a learnable parameter robustly
        model_dtype = next(iter(self.parameters())).dtype

        if d_in != self.dim: raise ValueError(f"HypL{self.level_idx} dim mismatch: {d_in} != {self.dim}")

        # --- Get Level Geometry (Dynamically) ---
        curvature = self.get_curvature().to(device)
        scale = self.get_scale().to(device)
        spread = self.get_spread().to(device)
        # Create manifold instance for this forward pass based on current curvature
        manifold = PoincareBall(c=curvature)
        # Assign current manifold to parameters for Riemannian optimizer step
        self.level_descriptor_param.manifold = manifold
        # Boundary manifold points also use the *current* manifold geometry for updates
        if self.boundary_manifold.hyperbolic_points is not None:
            self.boundary_manifold.hyperbolic_points.manifold = manifold
            # Also update the manifold reference within the BoundaryManifold object itself
            # so get_points() uses the current manifold for projection.
            self.boundary_manifold.manifold = manifold

        # --- Prepare Inputs for Tangent Space Combination ---
        # 1. Main input point -> Tangent vector at origin T_0 of current manifold H_i
        tangent_main = manifold.logmap0(point_in)

        # 2. Relative tangent vectors (already T_0 vectors from previous level's transform)
        tangent_relative = relative_vectors_tangent_in if relative_vectors_tangent_in is not None else torch.zeros_like(tangent_main)

        # 3. Level Descriptor -> Tangent vector T_0 of current manifold H_i
        # Project descriptor param onto current manifold and map to tangent
        ld_point_self = manifold.proju(self.level_descriptor_param) # Project the parameter itself
        tangent_ld_self = manifold.logmap0(ld_point_self).expand(batch_size, seq_len, -1) # Expand dims

        # Combine with descriptor point from previous level (if provided)
        tangent_ld_combined = tangent_ld_self # Start with self descriptor tangent
        if self.use_ld and descriptor_point_in is not None:
             # descriptor_point_in is a point in H_i (output of previous transform, projected)
             tangent_ld_in = manifold.logmap0(descriptor_point_in) # Map prev LD point to *current* tangent T_0
             # Simple addition of tangent vectors
             tangent_ld_combined = tangent_ld_combined + tangent_ld_in

        # 4. Spread scalar
        sigma_tensor = torch.full((batch_size, seq_len, 1), spread.item(), device=device, dtype=model_dtype) # Use current level's spread
        # Option: Combine sigma_in with current spread? For now, just use current spread.
        # if self.use_spread and sigma_in is not None:
        #     sigma_tensor = (sigma_tensor + sigma_in.to(dtype=model_dtype)) / 2.0 # Example: average

        # --- Combine Tangent Vectors ---
        inputs_to_combine = [tangent_main.to(model_dtype)]
        if self.relative_vector_aggregation != 'none': inputs_to_combine.append(tangent_relative.to(model_dtype))
        if self.use_ld: inputs_to_combine.append(tangent_ld_combined.to(model_dtype))
        if self.use_spread: inputs_to_combine.append(sigma_tensor)

        # Handle potential dtype mismatches before cat
        processed_inputs = []
        ref_dtype = tangent_main.dtype
        for idx, inp in enumerate(inputs_to_combine):
            if inp is not None and inp.dtype != ref_dtype: # Check for None too
                logger.warning(f"L{self.level_idx} dtype mismatch pre-combiner input {idx}. Casting.")
                processed_inputs.append(inp.to(ref_dtype))
            elif inp is not None:
                processed_inputs.append(inp)
            # Skip None inputs if they occur (e.g., optional components disabled)

        if not processed_inputs: raise ValueError(f"L{self.level_idx}: No valid inputs to tangent combiner.")

        combined_tangents = torch.cat(processed_inputs, dim=-1)
        # Check input dims before combiner
        expected_combiner_dim = self.tangent_combiner[0].in_features
        if combined_tangents.shape[-1] != expected_combiner_dim:
            raise ValueError(f"L{self.level_idx} Combiner dim mismatch: expect {expected_combiner_dim}, got {combined_tangents.shape[-1]}")

        v_combined_tangent = self.tangent_combiner(combined_tangents)

        # Apply optional tangent flow
        if self.use_flow and self.tangent_flow is not None:
             flow_displacement = self.tangent_flow(v_combined_tangent)
             v_combined_tangent = v_combined_tangent + flow_displacement * self.flow_scale

        # --- Map combined tangent vector back to manifold with scale ---
        # Simplification: Apply scale to the tangent vector before expmap0
        point_out = manifold.expmap0(v_combined_tangent * scale)

        tangent_out = manifold.logmap0(point_out) # Map back to tangent T_0 for next level aggregation

        # Get boundary points for this level (projected onto current manifold)
        boundaries_points = self.boundary_manifold.get_points() # Points in H_i

        # --- Prepare Outputs ---
        # Ensure outputs are projected and have correct dtype
        point_out = manifold.proju(point_out).to(model_dtype)
        tangent_out = tangent_out.to(model_dtype) # T_0 vector, no projection needed
        # Output this level's projected descriptor point, expanded
        level_descriptor_out = manifold.proju(ld_point_self).expand(batch_size, seq_len, -1).to(model_dtype)
        boundaries_points_out = boundaries_points.to(model_dtype) if boundaries_points is not None else None
        sigma_out = spread.to(model_dtype) # This level's spread scalar

        # Check for NaN/Inf in outputs
        outputs = [point_out, tangent_out, level_descriptor_out, boundaries_points_out, sigma_out]
        names = ["point_out", "tangent_out", "ld_point", "boundaries", "sigma"]
        final_outputs = []
        for name, out_tensor in zip(names, outputs):
            if out_tensor is not None and not torch.isfinite(out_tensor).all():
                logger.warning(f"NaN/Inf detected in L{self.level_idx} output '{name}'. Replacing with zeros.")
                out_tensor = torch.nan_to_num(out_tensor, nan=0.0)
            final_outputs.append(out_tensor)

        return tuple(final_outputs)


# --- Fully Hyperbolic WuBu Nesting Model (Orchestrator) ---
class FullyHyperbolicWuBuNestingModel(nn.Module):
    """ WuBu Nesting model using hyperbolic components. """
    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        self.num_levels = config.get("num_levels", 3)
        self.hyperbolic_dims = config.get("hyperbolic_dims", [128, 64, 32])
        self.initial_curvatures = config.get("initial_curvatures", [1.0] * self.num_levels)
        self.dropout = config.get("dropout", 0.1)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean")
        self.aggregation_method = config.get("aggregation_method", "concat_tangent")

        # --- Euclidean Input to Hyperbolic Bridge ---
        self.input_to_tangent = nn.Linear(input_dim, self.hyperbolic_dims[0])

        # --- Levels ---
        self.levels = nn.ModuleList([
            HyperbolicWuBuNestingLevel(
                i,
                self.hyperbolic_dims[i],
                self.config,
                initial_curvature=self.initial_curvatures[i] # Pass initial curvature
            )
            for i in range(self.num_levels)
        ])

        # --- Transitions ---
        self.transforms = nn.ModuleList()
        num_transitions = max(0, self.num_levels - 1)
        transform_types = config.get("transform_types", ["linear"] * num_transitions)
        transform_hdims = config.get("transform_hidden_dims", [None] * num_transitions)

        for i in range(num_transitions):
            manifold_in = PoincareBall(c=self.initial_curvatures[i])
            manifold_out = PoincareBall(c=self.initial_curvatures[i+1])
            self.transforms.append(
                HyperbolicInterLevelTransform(
                    self.hyperbolic_dims[i], self.hyperbolic_dims[i+1],
                    manifold_in, manifold_out, # Use initial manifolds for setup
                    transform_types[i], transform_hdims[i], self.dropout
                )
            )

        # --- Output Aggregation (Tangent Space Concat) ---
        if self.aggregation_method == "concat_tangent":
            combined_dim = sum(self.hyperbolic_dims)
            self.tangent_to_output = nn.Linear(combined_dim, output_dim)
        else:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not supported.")

        # Apply standard weight init to Euclidean modules
        self.input_to_tangent.apply(init_weights)
        self.tangent_to_output.apply(init_weights)
        # Sub-modules (levels, transforms) apply their own init internally

        total_params = sum(p.numel() for p in self.parameters()); trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"FullyHyperbolicWuBuNestingModel initialized: {self.num_levels} levels.")
        logger.info(f"InputDim->Hyp{self.hyperbolic_dims[0]} | Levels: {self.hyperbolic_dims} | Agg: {self.aggregation_method} | AggDim->{output_dim}")
        logger.info(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")


    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass through the Hyperbolic WuBu Nesting model. """
        batch_size, seq_len, _ = x.shape
        device = x.device
        model_dtype = next(iter(self.parameters())).dtype

        # 1. Project Euclidean input to tangent space T_0 H_0
        initial_tangent = self.input_to_tangent(x)
        # Map to first level's manifold using its *current* curvature
        manifold_0 = PoincareBall(c=self.levels[0].get_curvature().to(device))
        current_point = manifold_0.expmap0(initial_tangent) # Point in H_0

        level_tangent_outputs = [] # Store T_0 tangent vectors from each level
        aggregated_relative_vectors_tangent = None # T_0 vectors for next level
        current_descriptor_point = None # Point in H_i for next level
        current_sigma = None # Scalar spread for next level

        for i in range(self.num_levels):
            level_module = self.levels[i]
            # Get the manifold for this level based on its current curvature
            current_manifold = PoincareBall(c=level_module.get_curvature().to(device)) # H_i

            # Ensure input point is projected onto the current manifold H_i
            current_point = current_manifold.proju(current_point)
            # Ensure descriptor point (if from prev level) is projected onto H_i
            if current_descriptor_point is not None:
                 current_descriptor_point = current_manifold.proju(current_descriptor_point)

            # Process through the current level
            point_out, tangent_out, ld_point, boundaries_points, sigma_out = level_module(
                point_in=current_point,                         # Point in H_i
                relative_vectors_tangent_in=aggregated_relative_vectors_tangent, # Vector in T_0 H_i
                descriptor_point_in=current_descriptor_point,   # Point in H_i
                sigma_in=current_sigma                          # Scalar
            )
            # Outputs: point_out (H_i), tangent_out (T_0 H_i), ld_point (H_i), boundaries_points (H_i)

            # Store tangent output (T_0 vector of this level H_i) for final aggregation
            level_tangent_outputs.append(tangent_out)

            # Prepare inputs for the next level (if not the last)
            if i < self.num_levels - 1:
                transform_module = self.transforms[i]
                # Get the actual manifold for the *next* level
                manifold_next = PoincareBall(c=self.levels[i+1].get_curvature().to(device)) # H_{i+1}

                # No explicit rotation here. Assume transform handles implicitly.
                point_rotated = point_out # Point in H_i
                boundaries_rotated = boundaries_points # Points in H_i
                ld_rotated = ld_point # Point in H_i

                # --- Transform Points using the fixed transform module ---
                point_next_unproj, boundaries_transformed_unproj, ld_next_unproj = transform_module(
                    point_in=point_rotated,
                    boundaries_in=boundaries_rotated,
                    descriptor_in=ld_rotated
                )

                # --- Critical Step: Project outputs onto the *actual* next level manifold H_{i+1} ---
                point_next = manifold_next.proju(point_next_unproj)
                boundaries_transformed = manifold_next.proju(boundaries_transformed_unproj) if boundaries_transformed_unproj is not None else None
                ld_next = manifold_next.proju(ld_next_unproj) if ld_next_unproj is not None else None

                # --- Calculate Relative Vectors (in Tangent Space T_0 of *next* level H_{i+1}) ---
                aggregated_relative_vectors_tangent = None # Reset
                has_boundaries = boundaries_transformed is not None and boundaries_transformed.numel() > 0
                if has_boundaries and self.relative_vector_aggregation != 'none':
                    num_boundaries = boundaries_transformed.size(0)
                    # Map points to tangent space T_0 of the *next* manifold H_{i+1}
                    tangent_main_next = manifold_next.logmap0(point_next) # [B, S, D_next]
                    tangent_boundaries_next = manifold_next.logmap0(boundaries_transformed) # [N, D_next]

                    # Expand for subtraction: main [B, S, 1, D], boundaries [1, 1, N, D]
                    tangent_main_exp = tangent_main_next.unsqueeze(2)
                    tangent_bound_exp = tangent_boundaries_next.unsqueeze(0).unsqueeze(0)
                    relative_tangents_at_origin = tangent_main_exp - tangent_bound_exp # [B, S, N, D_next]

                    # Aggregate these T_0 relative vectors
                    agg_method = self.relative_vector_aggregation
                    if agg_method == "mean": aggregated_relative_vectors_tangent = torch.mean(relative_tangents_at_origin, dim=2)
                    elif agg_method == "sum": aggregated_relative_vectors_tangent = torch.sum(relative_tangents_at_origin, dim=2)
                    else: logger.warning(f"Unsupported rel vec agg '{agg_method}'. Setting to None."); aggregated_relative_vectors_tangent = None

                    if aggregated_relative_vectors_tangent is not None and not torch.isfinite(aggregated_relative_vectors_tangent).all():
                        logger.warning(f"NaN/Inf detected in L{i+1} relative vectors. Replacing with zeros.")
                        aggregated_relative_vectors_tangent = torch.zeros_like(tangent_main_next)

                # --- Update inputs for the next iteration ---
                current_point = point_next # Point in H_{i+1}
                current_descriptor_point = ld_next # Point in H_{i+1}
                current_sigma = sigma_out # Scalar spread from level i

        # Aggregate tangent outputs from all levels
        if self.aggregation_method == "concat_tangent":
             try:
                 # Ensure all tangent outputs have compatible dtypes and are finite
                 compatible_tangents = []
                 for t_idx, t in enumerate(level_tangent_outputs):
                     if not torch.isfinite(t).all():
                         logger.warning(f"NaN/Inf in tangent output from level {t_idx} before concat. Replacing with zeros.")
                         t = torch.nan_to_num(t, nan=0.0)
                     compatible_tangents.append(t.to(model_dtype))
                 aggregated_tangent = torch.cat(compatible_tangents, dim=-1)
             except Exception as concat_err:
                 logger.error(f"Error concatenating level tangents: {concat_err}. Shapes: {[t.shape for t in level_tangent_outputs]}", exc_info=True)
                 # Return zeros as fallback
                 return torch.zeros((batch_size, seq_len, self.output_dim), device=device, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not supported.")

        # Final projection from aggregated tangent space to Euclidean output dimension
        final_output = self.tangent_to_output(aggregated_tangent)

        # Apply padding mask if provided (to final Euclidean output)
        if padding_mask is not None:
             # Ensure mask is boolean and expanded correctly
             padding_mask_expanded = padding_mask.unsqueeze(-1).bool()
             final_output = final_output.masked_fill(padding_mask_expanded, 0.0)

        # Final stability check
        if not torch.isfinite(final_output).all():
            logger.error("NaN/Inf in final WuBu output tensor! Replacing with zeros.")
            final_output = torch.nan_to_num(final_output, nan=0.0, posinf=0.0, neginf=0.0)

        return final_output

# =====================================================================
# END: Modified WuBu Nesting Core (Hyperbolic Attempt)
# =====================================================================


# =====================================================================
# START: Modified Sequence Model Components (Hyperbolic Attempt)
# =====================================================================

# --- Hyperbolic Local Encoder (Compromise: Standard Transformer on Tangent Vectors) ---
class HyperbolicLocalEncoder(nn.Module):
    """Encodes patches using Hyperbolic Embeddings and a standard Transformer on tangent vectors."""
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, manifold: PoincareBall, dropout: float = 0.1, n_gram_sizes: List[int] = [], n_gram_vocab_size: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        if not isinstance(manifold, PoincareBall):
             raise TypeError("HyperbolicLocalEncoder requires a PoincareBall manifold instance.")
        self.manifold = manifold # Manifold for embeddings

        # Hyperbolic Byte Embeddings
        self.byte_embeddings = HyperbolicEmbedding(256, hidden_size, manifold)

        # N-grams (using standard embeddings, added in tangent space)
        self.n_gram_sizes = sorted(list(set(s for s in n_gram_sizes if s > 0)))
        self.n_gram_vocab_size = n_gram_vocab_size
        self.n_gram_embeddings = None
        self.hash_multipliers = {}
        if self.n_gram_sizes and self.n_gram_vocab_size > 0:
             self.n_gram_embeddings = nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes})
             for emb in self.n_gram_embeddings.values(): nn.init.normal_(emb.weight, std=0.02) # Standard init
             self.hash_multipliers = {n: torch.tensor([self._get_prime(n * 10 + i + 1) for i in range(n)], dtype=torch.long) for n in self.n_gram_sizes}
             logger.info(f"HypLocalEncoder N-grams enabled: Sizes={self.n_gram_sizes}, Vocab={self.n_gram_vocab_size} (Euclidean)")
        else: logger.info("HypLocalEncoder: N-gram features disabled.")

        # Standard Transformer Encoder Layer (Compromise: Operates on tangent vectors)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling (Standard Cross-Attention on Tangent Vectors)
        self.patch_pooling_attention = HAKMEMCrossAttentionBlock(hidden_size, num_heads, dropout)
        self.patch_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01) # Euclidean query
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6) # Standard Norm on output tangent vectors
        self.dropout = nn.Dropout(dropout)
        # Apply standard initialization to Transformer, pooling layers, Norm
        self.transformer.apply(init_weights)
        self.patch_pooling_attention.apply(init_weights)
        self.norm.apply(init_weights)

    def _get_prime(self, n): # Helper for N-grams
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
        patch_len = patch_byte_sequence.size(0); device = patch_byte_sequence.device
        if patch_len < n: return torch.empty(0, dtype=torch.long, device=device)
        # Ensure sequence is long for unfold
        patch_byte_sequence_long = patch_byte_sequence.long()
        windows = patch_byte_sequence_long.unsqueeze(0).unfold(dimension=1, size=n, step=1) # [1, NumWindows, n]
        multipliers = self.hash_multipliers.get(n)
        if multipliers is None: multipliers = torch.tensor([31**i for i in range(n)], device=device, dtype=torch.long)
        else: multipliers = multipliers.to(device=device, dtype=torch.long)
        multipliers = multipliers.view(1, 1, n)
        hashes = (windows * multipliers).sum(dim=-1) # [1, NumWindows]
        return (hashes % self.n_gram_vocab_size).squeeze(0) # [NumWindows]


    def forward(self, patches_with_entropy: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """ Output: Tangent vectors at origin [1, NumPatches, Dim] """
        if not patches_with_entropy:
            # Ensure consistent parameter access for device/dtype
            param = next(iter(self.parameters()), None)
            device = param.device if param is not None else torch.device('cpu')
            dtype = param.dtype if param is not None else torch.float32
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        device = patches_with_entropy[0][0].device
        model_dtype = next(iter(self.parameters())).dtype
        patch_tangent_representations = []

        for patch_bytes, patch_entropy in patches_with_entropy:
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue
            patch_bytes_long = patch_bytes.long()

            # Hyperbolic embedding -> points H_in
            x_hyp = self.byte_embeddings(patch_bytes_long).to(model_dtype).unsqueeze(0) # [1, PatchLen, Dim]

            # Map to tangent space T_0 H_in for Transformer processing (Compromise)
            x_tangent = self.manifold.logmap0(x_hyp) # [1, PatchLen, Dim]

            # Add N-gram features (Euclidean) in tangent space
            if self.n_gram_embeddings and self.n_gram_sizes:
                n_gram_features_tangent = torch.zeros_like(x_tangent)
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_long, n)
                        if n_gram_hashes.numel() > 0:
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes).to(model_dtype).unsqueeze(0) # [1, NWindows, D]
                            num_windows = ngram_embeds.size(1)
                            # Indices where the *end* of the n-gram window falls
                            indices = torch.arange(n - 1, n - 1 + num_windows, device=device, dtype=torch.long)
                            valid_mask = indices < patch_len # Ensure indices are within the patch length
                            valid_indices = indices[valid_mask]
                            valid_embeds = ngram_embeds[:, valid_mask, :] # Select embeddings for valid windows
                            if valid_indices.numel() > 0:
                                # Index needs reshaping for scatter_add_
                                index_reshaped = valid_indices.view(1, -1, 1).expand(1, valid_indices.size(0), self.hidden_size)
                                if index_reshaped.shape == valid_embeds.shape:
                                    n_gram_features_tangent.scatter_add_(1, index_reshaped, valid_embeds)
                                else: logger.warning(f"Ngram scatter shape mismatch: Idx {index_reshaped.shape}, Val {valid_embeds.shape}")


                x_tangent = x_tangent + n_gram_features_tangent # Add in tangent space

            # Process with standard Transformer in tangent space
            x_tangent = self.dropout(x_tangent)
            processed_tangent = self.transformer(x_tangent) # Output is tangent [1, PatchLen, Dim]

            # Pool tangent vectors using standard cross-attention
            batch_query = self.patch_query.expand(1, -1, -1).to(dtype=model_dtype) # Euclidean query
            patch_repr_tangent = self.patch_pooling_attention(queries=batch_query, keys_values=processed_tangent) # Output is tangent [1, 1, Dim]
            patch_tangent_representations.append(patch_repr_tangent.squeeze(1)) # [1, Dim]

        if not patch_tangent_representations:
            param = next(iter(self.parameters()), None)
            device = param.device if param is not None else torch.device('cpu')
            dtype = param.dtype if param is not None else torch.float32
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)

        # Combine patch tangent vectors and normalize
        patches_combined_tangent = torch.cat(patch_tangent_representations, dim=0).unsqueeze(0) # [1, NumPatches, Dim]
        normed_output_tangent = self.norm(patches_combined_tangent) # Normalize in tangent space

        # Final stability check
        if not torch.isfinite(normed_output_tangent).all():
             logger.warning("NaN/Inf in final local encoder output. Replacing.")
             normed_output_tangent = torch.nan_to_num(normed_output_tangent, nan=0.0)

        return normed_output_tangent # Return T_0 vectors for WuBu model input


# --- Hyperbolic Local Decoder (Compromise: Standard Transformer on Tangent Vectors) ---
class HyperbolicLocalDecoder(nn.Module):
    """Decodes using Hyperbolic Embeddings, standard Transformer, and tangent memory."""
    def __init__(self, hidden_size: int, global_tangent_dim: int, num_layers: int, num_heads: int,
                 manifold: PoincareBall, dropout: float = 0.1, use_hierarchical_pred: bool = True, max_decode_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        if not isinstance(manifold, PoincareBall):
             raise TypeError("HyperbolicLocalDecoder requires a PoincareBall manifold instance.")
        self.manifold = manifold # Manifold for embeddings
        self.max_decode_len = max_decode_len; self.vocab_size = 256

        # Hyperbolic Byte Embeddings
        self.byte_embeddings = HyperbolicEmbedding(self.vocab_size, hidden_size, manifold)
        # Standard Positional Encoding (added in tangent space)
        self.positional_encoding = nn.Embedding(max_decode_len, hidden_size)
        nn.init.normal_(self.positional_encoding.weight, std=0.02) # Standard init

        # Project Euclidean Tangent Memory to Decoder Tangent Dim (Standard MLP)
        self.memory_projection = nn.Sequential(
             nn.Linear(global_tangent_dim, hidden_size * 2), nn.GELU(),
             nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size))
        self.memory_projection.apply(init_weights) # Standard init for projection MLP

        # Standard Transformer Decoder Layer (Compromise)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=self.decoder_norm)
        # Transformer layers init weights internally

        # Output Prediction (Standard Linear from final tangent state)
        self.use_hierarchical = use_hierarchical_pred
        if self.use_hierarchical:
             self.byte_class_pred = nn.Linear(hidden_size, 16)
             self.byte_specific_pred = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(16)])
             self.byte_class_pred.apply(init_weights)
             for layer in self.byte_specific_pred: layer.apply(init_weights)
             logger.info("HypLocalDecoder using Hierarchical Prediction Head (Euclidean).")
        else:
             self.byte_pred = nn.Linear(hidden_size, self.vocab_size)
             self.byte_pred.apply(init_weights)
             logger.info("HypLocalDecoder using Flat Prediction Head (Euclidean).")

        self.dropout_embed = nn.Dropout(dropout)


    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
         return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tgt_byte_seq: torch.Tensor,
                memory_tangent: torch.Tensor, # Expect T_0 vectors from WuBu Model
                tgt_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, tgt_len = tgt_byte_seq.size(); device = tgt_byte_seq.device
        mem_batch_size, mem_len, mem_dim_in = memory_tangent.size()
        model_dtype = next(iter(self.parameters())).dtype

        # 1. Prepare Target Sequence
        # Hyperbolic embedding -> points H_in
        tgt_hyp = self.byte_embeddings(tgt_byte_seq.long()).to(model_dtype)
        # Map target points to tangent space T_0 H_in
        tgt_tangent = self.manifold.logmap0(tgt_hyp)
        # Add standard positional encoding in tangent space
        positions = torch.arange(0, tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        # Clamp positions to max length of positional encoding table
        positions_clamped = torch.clamp(positions, max=self.positional_encoding.num_embeddings - 1)
        pos_embed_tangent = self.positional_encoding(positions_clamped).to(model_dtype)
        tgt_prepared_tangent = self.dropout_embed(tgt_tangent + pos_embed_tangent)

        # 2. Prepare Memory (already tangent, project to hidden_size)
        if mem_len == 0:
             # Handle empty memory sequence
             projected_memory_tangent = torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=model_dtype)
             memory_key_padding_mask = None # No padding needed for empty memory
        else:
             projected_memory_tangent = self.memory_projection(memory_tangent.to(model_dtype))
             # Check mask shape consistency
             if memory_key_padding_mask is not None and memory_key_padding_mask.shape != (batch_size, mem_len):
                 logger.warning(f"Decoder memory key padding mask shape mismatch ({memory_key_padding_mask.shape}) vs memory ({batch_size, mem_len}). Ignoring mask.")
                 memory_key_padding_mask = None

        # 3. Process with Standard Transformer Decoder (operates entirely in tangent space)
        if tgt_mask is None: tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)

        # Ensure masks are boolean
        tgt_mask_bool = tgt_mask.to(torch.bool)
        mem_key_pad_mask_bool = memory_key_padding_mask.to(torch.bool) if memory_key_padding_mask is not None else None

        output_tangent = self.transformer(
             tgt=tgt_prepared_tangent,        # Tangent vectors [B, TgtLen, D]
             memory=projected_memory_tangent, # Tangent vectors [B, MemLen, D]
             tgt_mask=tgt_mask_bool,               # Boolean [TgtLen, TgtLen], True=mask
             memory_key_padding_mask=mem_key_pad_mask_bool # Boolean [B, MemLen], True=mask
        ) # Output is tangent vectors [B, TgtLen, D]

        # 4. Predict logits from final tangent states (Euclidean)
        byte_logits: torch.Tensor
        if self.use_hierarchical:
            byte_class_logits = self.byte_class_pred(output_tangent)
            if not torch.isfinite(byte_class_logits).all(): logger.warning("NaN/Inf hierarchical class logits. Replacing."); byte_class_logits=torch.nan_to_num(byte_class_logits, nan=0.0)
            log_class_probs = F.log_softmax(byte_class_logits, dim=-1)
            log_specific_probs_list = []
            for i in range(16):
                specific_logits = self.byte_specific_pred[i](output_tangent)
                if not torch.isfinite(specific_logits).all(): logger.warning(f"NaN/Inf hierarchical specific {i} logits. Replacing."); specific_logits=torch.nan_to_num(specific_logits, nan=0.0)
                log_specific_probs_list.append(F.log_softmax(specific_logits, dim=-1))
            log_specific_probs_stacked = torch.stack(log_specific_probs_list, dim=2) # [B, TgtLen, 16, 16]
            # Broadcast: [B, TgtLen, 16, 1] + [B, TgtLen, 16, 16] -> [B, TgtLen, 16, 16]
            combined_log_probs = log_class_probs.unsqueeze(-1) + log_specific_probs_stacked
            byte_logits = combined_log_probs.view(batch_size, tgt_len, self.vocab_size) # Reshape
        else:
             byte_logits = self.byte_pred(output_tangent)

        # Ensure final logits are float32 and finite
        byte_logits = byte_logits.float()
        if not torch.isfinite(byte_logits).all():
             logger.warning("NaN/Inf in final decoder logits. Replacing with zeros.")
             byte_logits = torch.nan_to_num(byte_logits, nan=0.0, posinf=0.0, neginf=0.0)

        return byte_logits

# =====================================================================
# END: Modified Sequence Model Components (Hyperbolic Attempt)
# =====================================================================


# =====================================================================
# START: Riemannian Optimizer (Requires GradientStats, HAKMEMQController)
# =====================================================================
class RiemannianEnhancedSGD(torch.optim.Optimizer):
    """ SGD optimizer enhanced with momentum, weight decay, Q-learning control,
        and support for Riemannian parameters defined by '.manifold' attribute. """
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01,
                 max_grad_norm=1.0, q_learning_config: Optional[Dict]=None):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # --- Q-Controller Setup ---
        self.q_controller: Optional[HAKMEMQController] = None
        if isinstance(q_learning_config, dict):
             try: self.q_controller = HAKMEMQController(**q_learning_config); logger.info("RiemannianEnhancedSGD: Q-Controller enabled.")
             except Exception as e: logger.error(f"Failed init QController: {e}. Disable.", exc_info=True)
        else: logger.info("RiemannianEnhancedSGD: Q-Controller disabled.")

        self.max_grad_norm = max_grad_norm # Note: Currently used by Trainer, not internally by step
        self._step_count = 0
        self.current_loss: Optional[float] = None
        self.gradient_stats = GradientStats() # Use GradientStats class defined earlier

        # Initialize momentum buffers lazily in step
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # State is initialized implicitly by super().__init__
                    # Buffers are created in the first step for each param
                    pass

    def zero_grad(self, set_to_none=True): super().zero_grad(set_to_none=set_to_none)
    def set_current_loss(self, loss: Optional[float]):
        if loss is not None and np.isfinite(loss): self.current_loss = loss
        else: logger.debug(f"Optimizer invalid loss: {loss}. Q uses prev: {self.current_loss}")


    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: logger.warning("RiemannianEnhancedSGD received unused closure.")

        # --- Q-Controller Update (applies scales to LR/Momentum) ---
        if self.q_controller and self.q_controller.prev_action:
             q_action = self.q_controller.prev_action
             for group in self.param_groups:
                 base_lr = group['base_lr']; base_momentum = group['base_momentum']
                 new_lr = base_lr * q_action.get('lr_scale', 1.0)
                 new_momentum = base_momentum * q_action.get('momentum_scale', 1.0)
                 group['lr'] = float(np.clip(new_lr, 1e-8, 0.1))
                 group['momentum'] = float(np.clip(new_momentum, 0.5, 0.999))

        # --- Parameter Update Loop ---
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']; weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad # Euclidean gradient dL/dp
                param_state = self.state[p]

                # Check for non-finite gradients early
                if not torch.isfinite(grad).all():
                     num_nan = torch.isnan(grad).sum().item(); num_inf = torch.isinf(grad).sum().item()
                     logger.error(f"Optim Error: Non-finite grad param {p.shape} (NaN:{num_nan}, Inf:{num_inf}). Skip update.")
                     # Clear momentum buffer if grad is bad
                     if 'momentum_buffer' in param_state: del param_state['momentum_buffer']
                     continue

                # --- Check if parameter is Riemannian ---
                is_riemannian = hasattr(p, 'manifold') and isinstance(p.manifold, Manifold)

                if is_riemannian:
                    # --- Riemannian Parameter Update ---
                    manifold = p.manifold
                    param_data = p.data # Current point on manifold

                    # 1. Convert Euclidean gradient to Riemannian gradient (vector in T_p M)
                    try:
                        riemannian_grad = manifold.egrad2rgrad(param_data, grad)
                        if not torch.isfinite(riemannian_grad).all():
                             raise ValueError("Non-finite Riemannian gradient calculated.")
                    except Exception as rgrad_err:
                        logger.error(f"Error converting to Riemannian grad param {p.shape} manifold {manifold.name}: {rgrad_err}. Skip update.", exc_info=False)
                        if 'momentum_buffer' in param_state: del param_state['momentum_buffer']
                        continue

                    # 2. Apply Riemannian Weight Decay (Decay towards origin - Tangent space approximation)
                    # Apply WD to the *update vector* later, not the raw gradient.

                    # 3. Apply Momentum (in Tangent Space T_p M)
                    if 'momentum_buffer' not in param_state:
                         # Initialize buffer in T_p M (zero vector)
                         param_state['momentum_buffer'] = torch.zeros_like(riemannian_grad)
                    buf = param_state['momentum_buffer'] # Lives in T_p M

                    # Momentum update: buf = momentum * buf + riemannian_grad
                    # Simplification: Ignore parallel transport. Assumes T_p spaces are locally similar.
                    buf.mul_(momentum).add_(riemannian_grad)
                    # Ensure buffer remains finite
                    if not torch.isfinite(buf).all():
                        logger.warning(f"Non-finite momentum buffer for param {p.shape}. Resetting buffer.")
                        buf.zero_() # Reset buffer if it becomes non-finite

                    # 4. Compute update step in tangent space T_p M
                    update_tangent = buf * lr # Note: step is -lr * buf

                    # Apply weight decay to the update vector (approximate decay towards origin)
                    if weight_decay != 0:
                         # Get vector pointing towards origin in T_0 (approximate direction)
                         try: vector_to_origin_t0 = manifold.logmap0(param_data)
                         except NotImplementedError: vector_to_origin_t0 = param_data # Fallback? Maybe zero? Use param_data.
                         # Project this vector onto T_p M? Identity for Poincare.
                         # Add decay term: lr * wd * vector_to_origin
                         decay_term = lr * weight_decay * vector_to_origin_t0
                         update_tangent = update_tangent + decay_term # Add decay term to tangent update

                    # 5. Retract: Move along manifold using exponential map
                    try:
                        # Perform retraction: p_next = Exp_p(-update_tangent)
                        # Note the negative sign for the update direction
                        new_param_data = manifold.expmap(param_data, -update_tangent)
                        # Ensure the new point is on the manifold (projection)
                        p.data = manifold.proju(new_param_data)
                        if not torch.isfinite(p.data).all():
                            raise ValueError("Retraction resulted in non-finite parameter values.")
                    except Exception as retract_err:
                        logger.error(f"Error retracting param {p.shape} manifold {manifold.name}: {retract_err}. Skip update.", exc_info=False)
                        if 'momentum_buffer' in param_state: del param_state['momentum_buffer'] # Reset momentum

                else:
                    # --- Euclidean Parameter Update (Standard SGD with Momentum/WD) ---
                    param_data = p.data
                    # Apply weight decay (adds wd * param to gradient)
                    if weight_decay != 0: grad = grad.add(param_data.float(), alpha=weight_decay).to(grad.dtype)
                    # Momentum buffer update
                    if 'momentum_buffer' not in param_state: param_state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    buf = param_state['momentum_buffer']; buf.mul_(momentum).add_(grad)
                    # Update step
                    update_step = buf * lr
                    param_data.add_(-update_step) # In-place update

        self._step_count += 1
        return None # step() should return None

    def get_q_info(self) -> Dict:
        if hasattr(self, 'q_controller') and self.q_controller: return self.q_controller.get_info()
        return {"Q-Controller": "Disabled"}

# =====================================================================
# END: Riemannian Optimizer
# =====================================================================


# =====================================================================
# START: Modified WuBuNestingSequenceModel (Using Hyperbolic Components)
# =====================================================================
class WuBuNestingSequenceModel(nn.Module):
    """ WuBu Nesting Sequence Model using Hyperbolic components (Experimental v0.04 - Corrected). """
    def __init__(self, wubu_config: Dict, sequence_config: Dict):
        super().__init__()
        self.wubu_config = wubu_config
        self.sequence_config = sequence_config

        # --- Sequence Config ---
        self.local_hidden_size = sequence_config["local_hidden_size"]
        self.decoder_memory_dim = sequence_config["decoder_memory_dim"] # WuBu output dim (Tangent)
        self.context_window = sequence_config["context_window"]
        self.vocab_size = 256

        # --- Manifold (Using initial curvature of first level for Encoder/Decoder Embeddings) ---
        self.shared_manifold = PoincareBall(c=wubu_config["initial_curvatures"][0])
        logger.info(f"SequenceModel Shared Manifold for Embeddings: {self.shared_manifold.name}")

        # --- Patching ---
        self.patcher = HAKMEMBabylonIndex()

        # --- Hyperbolic Local Encoder ---
        self.local_encoder = HyperbolicLocalEncoder(
             hidden_size=self.local_hidden_size,
             num_layers=sequence_config.get("num_encoder_layers", 2),
             num_heads=sequence_config.get("num_encoder_heads", 8),
             manifold=self.shared_manifold, # Use shared manifold for byte embeddings
             dropout=wubu_config.get("dropout", 0.1),
             n_gram_sizes=sequence_config.get("n_gram_sizes", []),
             n_gram_vocab_size=sequence_config.get("n_gram_vocab_size", 0)
        )

        # --- WuBu Nesting Core (Hyperbolic Version) ---
        self.wubu_model = FullyHyperbolicWuBuNestingModel(
             input_dim=self.local_hidden_size, # Encoder outputs tangent vectors
             output_dim=self.decoder_memory_dim, # WuBu outputs aggregated tangent vectors
             config=self.wubu_config
        )

        # --- Hyperbolic Local Decoder ---
        self.local_decoder = HyperbolicLocalDecoder(
             hidden_size=self.local_hidden_size,
             global_tangent_dim=self.decoder_memory_dim, # Expects tangent memory
             num_layers=sequence_config.get("num_decoder_layers", 4),
             num_heads=sequence_config.get("num_decoder_heads", 8),
             manifold=self.shared_manifold, # Use shared manifold for target byte embeddings
             dropout=wubu_config.get("dropout", 0.1),
             use_hierarchical_pred=sequence_config.get("use_hierarchical_decoder", True),
             max_decode_len=max(self.context_window * 2, 2048)
        )
        logger.info("Hyperbolic WuBuNestingSequenceModel Initialized (v0.04 - Corrected).")


    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = byte_seq.shape
        device = byte_seq.device
        model_dtype = next(iter(self.parameters())).dtype

        # 1. Patching and Local Encoding (Outputs Tangent Vectors)
        batch_patch_repr_list = []; num_patches_per_item = []; valid_batch_indices = []; max_num_patches = 0
        for i in range(batch_size):
            seq = byte_seq[i]; patches = self.patcher.create_patches(seq)
            if patches:
                patches_on_device = [(p.to(device), e) for p, e in patches]
                # Encoder outputs TANGENT vectors [1, NumP, LocalHidden]
                try:
                    patch_repr_tangent = self.local_encoder(patches_on_device)
                    if not torch.isfinite(patch_repr_tangent).all():
                        logger.warning(f"NaN/Inf in LocalEncoder output batch item {i}. Replacing."); patch_repr_tangent = torch.nan_to_num(patch_repr_tangent, nan=0.0)
                except Exception as enc_err:
                     logger.error(f"Error in local encoder batch item {i}: {enc_err}", exc_info=False)
                     patch_repr_tangent = torch.empty(1, 0, self.local_hidden_size, device=device, dtype=model_dtype) # Empty if error

                num_p = patch_repr_tangent.size(1)
                if num_p > 0:
                    batch_patch_repr_list.append(patch_repr_tangent.squeeze(0)) # [NumP, LocalHidden]
                    num_patches_per_item.append(num_p); valid_batch_indices.append(i); max_num_patches = max(max_num_patches, num_p)
                else: num_patches_per_item.append(0) # Append 0 count if encoder returns empty
            else: num_patches_per_item.append(0) # Append 0 count if no patches created

        target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
        # Handle case where no items in the batch produced valid patches
        if not valid_batch_indices or max_num_patches == 0:
            logger.warning(f"No valid patches found/encoded for batch (Size {batch_size}). Returning zero logits.")
            return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)

        num_valid = len(valid_batch_indices)
        # Pad tangent representations for valid items: [NumValid, MaxPatches, LocalHidden]
        padded_patch_tangent_repr = torch.zeros(num_valid, max_num_patches, self.local_hidden_size, device=device, dtype=model_dtype)
        # Padding mask (True where padded): [NumValid, MaxPatches]
        memory_padding_mask = torch.ones(num_valid, max_num_patches, dtype=torch.bool, device=device)

        # --- Corrected Padding Logic ---
        valid_item_counter = 0
        for i in range(batch_size):
            if i in valid_batch_indices:
                if valid_item_counter < len(batch_patch_repr_list): # Check if we have a tensor for this valid index
                    patch_repr_tensor = batch_patch_repr_list[valid_item_counter]
                    num_p = patch_repr_tensor.shape[0]
                    if num_p > 0 and num_p == num_patches_per_item[i]: # Check if patch count matches
                        padded_patch_tangent_repr[valid_item_counter, :num_p, :] = patch_repr_tensor
                        memory_padding_mask[valid_item_counter, :num_p] = False # Unmask the valid patches
                    else:
                        # Mismatch or zero patches for a supposedly valid item: keep masked
                         logger.warning(f"Item {i}: Patch count mismatch {num_p} vs {num_patches_per_item[i]} or 0 patches. Keeping masked.")
                         memory_padding_mask[valid_item_counter, :] = True # Ensure it stays masked
                else:
                     logger.error(f"Item {i}: Index mismatch during padding. Counter {valid_item_counter}, List len {len(batch_patch_repr_list)}. Keeping masked.")
                     if valid_item_counter < num_valid: # Avoid index out of bounds
                        memory_padding_mask[valid_item_counter, :] = True
                valid_item_counter += 1 # Increment only when processing a valid index 'i'

        # 2. WuBu Nesting Model Processing (Input: Tangent, Output: Tangent)
        try:
            decoder_memory_tangent = self.wubu_model(padded_patch_tangent_repr, padding_mask=memory_padding_mask)
            if not torch.isfinite(decoder_memory_tangent).all():
                logger.warning("NaN/Inf from wubu_model output. Replacing."); decoder_memory_tangent = torch.nan_to_num(decoder_memory_tangent, nan=0.0)
        except Exception as wubu_err:
            logger.error(f"Error in WuBu core model forward: {wubu_err}", exc_info=True)
            # Return zeros if WuBu core fails
            return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)

        # 3. Decode (Input: Tangent Memory, Output: Euclidean Logits)
        if target_byte_seq is None or target_len == 0:
            # If no target, maybe return empty logits or handle differently depending on use case
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)

        # Select target sequences only for the valid batch items that went through WuBu
        valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
        if num_valid > 0:
            valid_target_byte_seq = torch.index_select(target_byte_seq, 0, valid_indices_tensor).to(device).long()
            try:
                byte_logits_valid = self.local_decoder(
                     tgt_byte_seq=valid_target_byte_seq,
                     memory_tangent=decoder_memory_tangent, # Pass tangent memory
                     memory_key_padding_mask=memory_padding_mask # Pass mask for cross-attn
                ) # Output: Euclidean Logits [NumValid, TargetLen, VocabSize]
                if not torch.isfinite(byte_logits_valid).all():
                    logger.warning("NaN/Inf from local_decoder output. Replacing."); byte_logits_valid = torch.nan_to_num(byte_logits_valid, nan=0.0)
            except Exception as dec_err:
                 logger.error(f"Error in local decoder forward: {dec_err}", exc_info=True)
                 byte_logits_valid = torch.zeros((num_valid, target_len, self.vocab_size), device=device, dtype=torch.float32) # Fallback
        else:
            # Should not happen if we checked valid_batch_indices earlier, but handle defensively
            byte_logits_valid = torch.empty((0, target_len, self.vocab_size), device=device, dtype=torch.float32)

        # 4. Reconstruct Full Batch Output (Euclidean Logits)
        final_byte_logits = torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)
        if num_valid > 0 and byte_logits_valid.numel() > 0:
            # Ensure shapes are compatible before index_copy_
            if byte_logits_valid.shape[0] == num_valid and valid_indices_tensor.shape[0] == num_valid:
                 try:
                     final_byte_logits.index_copy_(0, valid_indices_tensor, byte_logits_valid)
                 except Exception as e_scatter: logger.error(f"Error scattering logits: {e_scatter}. Logits shape {byte_logits_valid.shape}, Idx shape {valid_indices_tensor.shape}")
            else: logger.error(f"Shape mismatch for index_copy: Logits {byte_logits_valid.shape[0]}, ValidIdx {num_valid}. Skipping scatter.")

        return final_byte_logits

    # --- Static Loss Computation (Operates on Euclidean Logits) ---
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        if seq_len <= 1: return torch.tensor(0.0, device=logits.device, requires_grad=True, dtype=logits.dtype)
        # Shift logits and targets for next token prediction
        logits_shifted = logits[:, :-1, :].contiguous() # [B, S-1, V]
        targets_shifted = targets[:, 1:].contiguous() # [B, S-1]
        # Flatten for cross-entropy
        logits_flat = logits_shifted.view(-1, vocab_size) # [B*(S-1), V]
        targets_flat = targets_shifted.view(-1) # [B*(S-1)]
        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1) # Ensure targets are valid indices

        # Check for non-finite logits before loss calculation
        if not torch.isfinite(logits_flat).all():
             num_nan = torch.isnan(logits_flat).sum().item(); num_inf = torch.isinf(logits_flat).sum().item()
             logger.error(f"NaN/Inf logits passed to compute_loss (NaN:{num_nan}, Inf:{num_inf}). Returning high loss.")
             return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        loss: torch.Tensor
        # Calculate loss (use float32 for stability)
        if smoothing > 0.0 and 0.0 < smoothing < 1.0:
            with torch.no_grad():
                smooth_val_on = 1.0 - smoothing; smooth_val_off = smoothing / max(1, vocab_size - 1)
                true_dist = torch.full_like(logits_flat, smooth_val_off)
                true_dist.scatter_(1, targets_flat.unsqueeze(1), smooth_val_on)
            log_probs = F.log_softmax(logits_flat.float(), dim=-1) # Use float32
            loss = -(true_dist * log_probs).sum(dim=-1) # Shape: [B*(S-1)]
        else:
            loss = F.cross_entropy(logits_flat.float(), targets_flat.long(), reduction='none') # Shape: [B*(S-1)]

        # Check for non-finite loss values after calculation
        if not torch.isfinite(loss).all():
            logger.error(f"NaN/Inf loss calculated. Replacing before masking."); loss = torch.nan_to_num(loss, nan=100.0, posinf=100.0, neginf=-100.0)

        # Apply mask if provided
        mean_loss: torch.Tensor
        if mask is not None:
            mask_shifted = mask[:, 1:].contiguous() # Shift mask like targets [B, S-1]
            if mask_shifted.shape == targets_shifted.shape:
                mask_flat = mask_shifted.view(-1) # [B*(S-1)]
                # Ensure mask is boolean (True = MASKED)
                mask_flat_bool = mask_flat.bool() if mask_flat.dtype == torch.bool else mask_flat > 0
                # Zero out loss for masked elements
                loss = loss.masked_fill(mask_flat_bool, 0.0)
                # Compute mean over non-masked elements
                num_active_elements = (~mask_flat_bool).sum()
                if num_active_elements.item() > 0:
                    mean_loss = loss.sum() / num_active_elements
                else:
                    logger.warning("All target elements masked in loss calculation. Returning zero loss.")
                    mean_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            else:
                 logger.warning(f"Mask shape mismatch ({mask.shape}) vs target ({targets.shape}). Ignoring mask.")
                 mean_loss = loss.mean()
        else:
            # No mask, compute mean over all elements
            mean_loss = loss.mean()

        # Final check for mean loss
        if not torch.isfinite(mean_loss):
            logger.error(f"NaN/Inf final mean loss. Returning high loss.");
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

        return mean_loss

    # --- Generation Function (Operates on Euclidean Logits) ---
    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None, repetition_penalty: float = 1.1, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        self.eval(); device = next(iter(self.parameters())).device
        if seed_bytes.device != device: seed_bytes = seed_bytes.to(device)
        seed_bytes = seed_bytes.long()
        batch_size, seed_len = seed_bytes.size()
        if seed_len == 0: return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        generated_sequence = seed_bytes.clone()
        if sampling_config is None: sampling_config = SamplerConfig()
        base_temperature = max(temperature, EPS); min_temp = max(0.1, base_temperature * 0.5); max_temp = min(2.0, base_temperature * 1.5)
        disable_tqdm = batch_size > 1 # Reduce tqdm noise for multi-batch generation
        gen_iterator = tqdm(range(max_length), desc="Generating", disable=disable_tqdm, total=max_length, unit="byte", leave=False)

        for step in gen_iterator:
            current_context = generated_sequence.long()
            amp_context = amp.autocast(device_type=device.type, enabled=False) # Disable AMP for generation
            with torch.no_grad(), amp_context:
                # Pass current context as both input and target (decoder needs a target sequence)
                logits_all = self(byte_seq=current_context, target_byte_seq=current_context)

            if logits_all is None or logits_all.numel() == 0 or logits_all.shape[1] == 0:
                logger.warning(f"Logits generation failed at step {step}. Stopping."); break
            if not torch.isfinite(logits_all).all():
                logger.warning(f"NaN/Inf logits at step {step}. Using uniform distribution.");
                logits_all = torch.zeros_like(logits_all) # Create uniform logits as fallback

            # Get logits for the *next* byte prediction (last position in sequence dim)
            next_byte_logits = logits_all[:, -1, :].float() # [B, VocabSize]
            next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

            # Sample independently for each item in the batch
            for i in range(batch_size):
                current_logits = next_byte_logits[i].clone() # [VocabSize]
                current_seq = generated_sequence[i] # [CurrentSeqLen]

                # Apply repetition penalty
                if repetition_penalty > 1.0 and current_seq.numel() > 0:
                    seen_bytes = torch.unique(current_seq);
                    for byte_val_tensor in seen_bytes:
                        byte_val = byte_val_tensor.item()
                        if 0 <= byte_val < self.vocab_size:
                            # Penalize based on sign
                            if current_logits[byte_val] > 0: current_logits[byte_val] /= repetition_penalty
                            else: current_logits[byte_val] *= repetition_penalty # Make negative logits more negative

                # Temperature scaling
                adaptive_temp = base_temperature
                scaled_logits = current_logits / adaptive_temp
                filtered_logits = scaled_logits

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    if k > 0: # Ensure k is positive
                         top_k_vals, _ = torch.topk(filtered_logits, k)
                         top_k_threshold = top_k_vals[..., -1, None] # Get the k-th value
                         indices_to_remove = filtered_logits < top_k_threshold
                         filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))
                    else: filtered_logits.fill_(-float('Inf')) # Mask all if k=0

                # Top-P filtering (Nucleus sampling)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))

                # Compute probabilities and sample
                probs_final = F.softmax(filtered_logits, dim=-1)
                if not torch.isfinite(probs_final).all() or probs_final.sum() < EPS:
                     logger.warning(f"Invalid final probs item {i} step {step}. Sampling uniformly.")
                     probs_final = torch.ones_like(current_logits) / current_logits.size(-1)

                # Sample: Argmax if temp is zero/negative, multinomial otherwise
                if temperature <= EPS: next_byte_idx = torch.argmax(probs_final)
                else: next_byte_idx = torch.multinomial(probs_final, num_samples=1).squeeze(-1)
                next_byte_indices[i] = next_byte_idx.item()

            # Append sampled bytes to the generated sequence
            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)

        if not disable_tqdm: gen_iterator.close()
        return generated_sequence

# =====================================================================
# END: Modified WuBuNestingSequenceModel
# =====================================================================


# =====================================================================
# Trainer Class (Corrected Indentation, Logging, Checkpointing)
# =====================================================================
class Trainer:
    """Handles training/validation loops, checkpointing, logging."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10,
                 save_interval: int = 1000, checkpoint_dir: str = "checkpoints",
                 wandb_enabled: bool = False, max_grad_norm: float = 1.0,
                 rank: int = 0, world_size: int = 1, detect_anomaly: bool = False):
        self.model = model; self.optimizer = optimizer; self.device = device
        self.train_loader = train_loader; self.val_loader = val_loader
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch, "amp") and device.type == 'cuda'
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.log_interval = log_interval; self.save_interval = max(1, save_interval) if save_interval > 0 else 0
        self.checkpoint_dir = checkpoint_dir
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE and is_main_process() # Corrected check
        self.max_grad_norm = max_grad_norm; self.global_step = 0; self.current_epoch = 0
        self.last_val_metrics: Optional[Dict[str, float]] = None
        self.rank = rank; self.world_size = world_size; self.is_main = is_main_process()
        self.detect_anomaly = detect_anomaly
        if self.is_main: os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Check for optimizer attributes (works for RiemannianEnhancedSGD too)
        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(getattr(self.optimizer, 'gradient_stats', None), GradientStats)
        self.has_q_controller = hasattr(self.optimizer, 'q_controller') and isinstance(getattr(self.optimizer, 'q_controller', None), HAKMEMQController)
        self.wandb_run = wandb.run if self.wandb_enabled and wandb is not None else None
        logger.info(f"Trainer(Hyp) Rank {rank}: AMP={self.use_amp}, Accum={self.grad_accum_steps}, MaxNorm={self.max_grad_norm}, Anomaly={self.detect_anomaly}, QCtrl={self.has_q_controller}")

    def _train_epoch(self):
        self.model.train()
        total_loss_accum_cycle = 0.0; optimizer_steps_in_epoch = 0; micro_step_count_cycle = 0; total_micro_batches_processed_epoch = 0
        approx_total_optim_steps = None; total_micro_batches_estimate = None
        # --- Estimate total batches for tqdm ---
        try:
            dataset_len = 0
            if hasattr(self.train_loader.sampler, '__len__'): dataset_len = len(self.train_loader.sampler)
            elif hasattr(self.train_loader.dataset, '__len__'):
                 dset_len = len(self.train_loader.dataset)
                 dataset_len = max(1, dset_len // self.world_size) if self.world_size > 1 else dset_len

            if dataset_len > 0:
                loader_bs = self.train_loader.batch_size or 1; total_micro_batches_estimate = math.ceil(dataset_len / loader_bs)
                if total_micro_batches_estimate > 0 and self.grad_accum_steps > 0: approx_total_optim_steps = max(1, total_micro_batches_estimate // self.grad_accum_steps)
                logger.debug(f"Rank {self.rank} Epoch Est: {total_micro_batches_estimate} micro-batches, {approx_total_optim_steps} optim steps.")
            else:
                logger.info("Cannot estimate epoch length for tqdm (dataset or sampler lacks length).")

        except Exception as e: logger.warning(f"Could not estimate epoch length: {e}")

        disable_tqdm = not self.is_main
        batch_iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} | Opt Step 0/?", disable=disable_tqdm, total=total_micro_batches_estimate, unit="batch", dynamic_ncols=True, leave=False)
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch_data in enumerate(batch_iterator):
            total_micro_batches_processed_epoch += 1; micro_step_count_cycle += 1
            is_last_batch_in_epoch = (i == (total_micro_batches_estimate - 1)) if total_micro_batches_estimate is not None else False
            should_optimizer_step = (micro_step_count_cycle % self.grad_accum_steps == 0) or is_last_batch_in_epoch
            sync_context = contextlib.nullcontext()
            if self.world_size > 1 and isinstance(self.model, DDP):
                if not should_optimizer_step: sync_context = self.model.no_sync()
            anomaly_context = torch.autograd.detect_anomaly(check_nan=True) if self.detect_anomaly else contextlib.nullcontext()
            loss_value_scaled = None; loss = None; current_step_loss = 0.0

            try: # Forward/Backward Pass
                with sync_context, anomaly_context:
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2: context, targets = batch_data
                    else: logger.warning(f"Rank {self.rank}: Skip unexpected batch type {type(batch_data)}"); continue
                    context = context.to(self.device, non_blocking=True); targets = targets.to(self.device, non_blocking=True)
                    if context.numel() == 0: logger.warning(f"Rank {self.rank}: Skip empty batch"); continue
                    with amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=self.scaler.is_enabled()):
                        logits = self.model(byte_seq=context, target_byte_seq=context)
                        if logits is None: raise ValueError("Model forward returned None logits")
                        loss = WuBuNestingSequenceModel.compute_loss(logits.float(), targets, smoothing=0.1) # Use static method
                        if loss is None or not torch.isfinite(loss): raise ValueError(f"Non-finite/None loss: {loss}")
                    loss_value_scaled = loss / self.grad_accum_steps
                self.scaler.scale(loss_value_scaled).backward()
                current_step_loss = loss.item()
                if not np.isfinite(current_step_loss): logger.warning(f"Rank {self.rank}: Non-finite loss ({current_step_loss}) step {self.global_step}. Not accumulated."); current_step_loss = 0.0

            except Exception as batch_ex:
                logger.error(f"Micro-step Error G{self.global_step} M{micro_step_count_cycle} R{self.rank}: {batch_ex}", exc_info=False)
                total_loss_accum_cycle = 0.0; micro_step_count_cycle = 0; should_optimizer_step = False
                try: self.optimizer.zero_grad(set_to_none=True)
                except Exception: pass
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue # Skip to next micro-batch

            total_loss_accum_cycle += current_step_loss

            if should_optimizer_step:
                avg_loss_cycle = total_loss_accum_cycle / micro_step_count_cycle if micro_step_count_cycle > 0 else 0.0
                optimizer_step_skipped = False; unclipped_norm_val = 0.0; is_clipped = False; clip_ratio = None; has_nonfinite_grad = False

                # 1. Unscale Gradients
                try: self.scaler.unscale_(self.optimizer)
                except Exception as unscale_err: logger.error(f"Unscale Error G{self.global_step} R{self.rank}: {unscale_err}. Skip optim."); has_nonfinite_grad = True; optimizer_step_skipped = True

                # 2. Check Gradient Norm & Clip (Standard Euclidean Norm)
                if not optimizer_step_skipped:
                    params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
                    if params_with_grad:
                        try: # Calculate Euclidean norm
                            all_norms_sq = [torch.sum(p.grad.detach().float()**2) for p in params_with_grad]
                            finite_norms_sq = [n for n in all_norms_sq if torch.isfinite(n)]
                            if len(finite_norms_sq) < len(all_norms_sq):
                                has_nonfinite_grad = True; total_norm_unclipped = torch.tensor(float('inf'), device=self.device)
                                num_non_finite = len(all_norms_sq) - len(finite_norms_sq); logger.warning(f"Rank {self.rank}: Non-finite grads in {num_non_finite} param(s) step {self.global_step}.")
                            elif finite_norms_sq: total_norm_unclipped = torch.sqrt(torch.stack(finite_norms_sq).sum())
                            else: total_norm_unclipped = torch.tensor(0.0, device=self.device) # No finite grads?
                        except Exception as norm_ex: logger.error(f"Grad norm Error G{self.global_step} R{self.rank}: {norm_ex}"); total_norm_unclipped = torch.tensor(float('inf'), device=self.device); has_nonfinite_grad = True
                        unclipped_norm_val = total_norm_unclipped.item()
                    else: unclipped_norm_val = 0.0 # No grads

                    if has_nonfinite_grad:
                        logger.warning(f"Rank {self.rank}: Skipping optim step {self.global_step} due to non-finite grads/norm (Norm: {unclipped_norm_val}).")
                        optimizer_step_skipped = True
                    else:
                        # Clip gradients if norm exceeds max_grad_norm
                        if self.max_grad_norm > 0 and unclipped_norm_val > self.max_grad_norm:
                            is_clipped = True; clip_ratio = self.max_grad_norm / (unclipped_norm_val + EPS)
                            torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm) # Standard clip

                        # Record grad stats
                        if self.has_grad_stats: self.optimizer.gradient_stats.record_gradient(unclipped_norm_val, is_clipped, clip_ratio)

                        # 3. Q-Controller Update
                        if self.has_q_controller:
                            try:
                                self.optimizer.set_current_loss(avg_loss_cycle); group = self.optimizer.param_groups[0]; current_lr = group['lr']; current_mom = group.get('momentum', 0.0)
                                q_state = self.optimizer.q_controller.get_state(lr=current_lr, momentum=current_mom, grad_norm=unclipped_norm_val, loss=avg_loss_cycle)
                                if self.optimizer.q_controller.prev_state is not None and self.optimizer.q_controller.prev_action is not None and q_state is not None:
                                    reward = self.optimizer.q_controller.compute_reward(avg_loss_cycle, self.optimizer.q_controller.prev_loss, unclipped_norm_val)
                                    if np.isfinite(reward): self.optimizer.q_controller.update(self.optimizer.q_controller.prev_state, self.optimizer.q_controller.prev_action, reward, q_state)
                                    else: logger.warning(f"Q-Controller non-finite reward ({reward}). Skip Q-update.")
                                self.optimizer.q_controller.prev_loss = avg_loss_cycle; self.optimizer.q_controller.prev_state = q_state
                                if q_state is not None: next_q_action = self.optimizer.q_controller.choose_action(q_state); self.optimizer.q_controller.prev_action = next_q_action
                                else: self.optimizer.q_controller.prev_action = None
                            except Exception as q_err: logger.error(f"Q-Controller Error G{self.global_step} R{self.rank}: {q_err}", exc_info=False); self.optimizer.q_controller.prev_state = None; self.optimizer.q_controller.prev_action = None

                        # 4. Optimizer Step (using scaled gradients)
                        self.scaler.step(self.optimizer) # Calls RiemannianEnhancedSGD.step

                # 5. Update AMP Scaler
                self.scaler.update()
                # 6. Zero Gradients
                self.optimizer.zero_grad(set_to_none=True)

                # 7. Logging and Checkpointing
                grad_stats = {}
                if self.has_grad_stats: grad_stats = self.optimizer.gradient_stats.record_step(self.global_step, skipped=optimizer_step_skipped)

                if not optimizer_step_skipped:
                    optimizer_steps_in_epoch += 1; self.global_step += 1
                    if self.is_main:
                        # Logging logic
                        optim_step_str = f"{optimizer_steps_in_epoch}/{(approx_total_optim_steps or '?')}"
                        batch_iterator.set_description(f"Epoch {self.current_epoch + 1} | Opt Step {optim_step_str}")
                        current_lr = self.optimizer.param_groups[0]['lr']
                        logged_norm = min(unclipped_norm_val, self.max_grad_norm) if self.max_grad_norm > 0 and is_clipped else unclipped_norm_val
                        batch_iterator.set_postfix(Loss=f"{avg_loss_cycle:.3f}", LR=f"{current_lr:.3e}", Grad=f"{logged_norm:.2f}", Scale=f"{self.scaler.get_scale():.0f}", refresh=False)
                        if self.global_step % self.log_interval == 0:
                             current_mom = self.optimizer.param_groups[0].get('momentum', 0.0); q_info = self.optimizer.get_q_info() if self.has_q_controller else {}
                             log_data = { "Epoch": self.current_epoch + 1, "Step": self.global_step, "Train Loss (Cycle Avg)": avg_loss_cycle, "Learning Rate": current_lr, "Momentum": current_mom, "Grad Norm (Applied)": logged_norm, "Grad Norm (Unclipped Max)": grad_stats.get('max_gradient', 0.0), "Clip %": grad_stats.get('clip_percentage', 0.0), "NonFinite Grads Encountered": grad_stats.get('non_finite_grads', 0), "AMP Scale": self.scaler.get_scale()}
                             log_data.update({f"Q_{k}": v for k, v in q_info.items() if k != 'last_action'}); last_q_action = q_info.get('last_action')
                             if last_q_action: log_data["Q_LR_Scale"] = last_q_action.get('lr_scale', 1.0); log_data["Q_Mom_Scale"] = last_q_action.get('momentum_scale', 1.0)
                             # Add hyperbolic stats
                             hyp_stats = {}
                             model_to_log = self.model.module if isinstance(self.model, DDP) else self.model
                             if hasattr(model_to_log, 'wubu_model') and hasattr(model_to_log.wubu_model, 'levels'):
                                 hyp_levels = model_to_log.wubu_model.levels
                                 for i_lvl, lvl in enumerate(hyp_levels):
                                     try:
                                         if hasattr(lvl, 'get_curvature'): hyp_stats[f"L{i_lvl}_Curvature"] = lvl.get_curvature().item()
                                         if hasattr(lvl, 'get_scale'): hyp_stats[f"L{i_lvl}_Scale"] = lvl.get_scale().item()
                                         if hasattr(lvl, 'get_spread'): hyp_stats[f"L{i_lvl}_Spread"] = lvl.get_spread().item()
                                     except Exception as stat_err: logger.warning(f"Error getting hyp stats L{i_lvl}: {stat_err}")
                             log_data.update(hyp_stats)
                             log_msg_parts = [f"Step {self.global_step}", f"Ep {self.current_epoch+1} Opt {optimizer_steps_in_epoch}", f"Loss {log_data['Train Loss (Cycle Avg)']:.4f}", f"LR {log_data['Learning Rate']:.3e}", f"Grad {log_data['Grad Norm (Applied)']:.2f}", f"Scale {log_data['AMP Scale']:.0f}"]
                             if hyp_stats: log_msg_parts.append(f"C[0] {hyp_stats.get('L0_Curvature',0):.2f}")
                             if self.has_q_controller: log_msg_parts.append(f"QScale(LR/M) {log_data.get('Q_LR_Scale', 1.0):.2f}/{log_data.get('Q_Mom_Scale', 1.0):.2f}")
                             logger.info(" | ".join(log_msg_parts))
                             if self.wandb_run:
                                 try: wandb.log({"train": log_data}, step=self.global_step)
                                 except Exception as wb_err: logger.error(f"WandB log failed: {wb_err}")

                        if self.is_main and self.save_interval > 0 and self.global_step % self.save_interval == 0:
                            self._save_checkpoint(is_intermediate=True, metrics={'train_loss_cycle': avg_loss_cycle})

                # Reset accumulation cycle
                total_loss_accum_cycle = 0.0; micro_step_count_cycle = 0

        if self.is_main and hasattr(batch_iterator, 'close'): batch_iterator.close()
        if self.world_size > 1: logger.debug(f"Rank {self.rank} end-of-epoch barrier."); torch.distributed.barrier(); logger.debug(f"Rank {self.rank} exited barrier.")


    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_loader is None: return {}
        self.model.eval(); approx_total_val_batches = None
        try: # Estimate val length
             val_ds_len = 0
             if hasattr(self.val_loader.sampler, '__len__'): val_ds_len = len(self.val_loader.sampler)
             elif hasattr(self.val_loader.dataset, '__len__'): val_ds_len = max(1, len(self.val_loader.dataset) // self.world_size) if self.world_size > 1 else len(self.val_loader.dataset)
             if val_ds_len > 0: approx_total_val_batches = math.ceil(val_ds_len / (self.val_loader.batch_size or 1))
        except Exception: pass

        val_iterator = tqdm(self.val_loader, desc=f"Val Ep {self.current_epoch + 1}", disable=not self.is_main, total=approx_total_val_batches, unit="batch", leave=False)
        batch_losses = []
        for batch_data in val_iterator:
            try:
                 if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2: context, targets = batch_data
                 else: continue
                 context = context.to(self.device, non_blocking=True); targets = targets.to(self.device, non_blocking=True)
                 if context.numel() == 0: continue
                 batch_size, ctx_len = context.shape

                 with torch.no_grad(), amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=self.use_amp):
                     model_to_eval = self.model.module if isinstance(self.model, DDP) else self.model
                     logits = model_to_eval(byte_seq=context, target_byte_seq=context)
                     loss = WuBuNestingSequenceModel.compute_loss(logits.float(), targets, smoothing=0.0) # No smoothing for val

                 if loss is not None and torch.isfinite(loss): batch_losses.append(loss.item())
            except Exception as val_ex: logger.error(f"Rank {self.rank} Error val batch: {val_ex}", exc_info=False); continue

        # Gather losses from all ranks
        all_losses = []
        if self.world_size > 1:
             losses_tensor = torch.tensor(batch_losses, dtype=torch.float64, device=self.device)
             gathered_losses_list = [torch.zeros_like(losses_tensor) for _ in range(self.world_size)]
             try: torch.distributed.all_gather(gathered_losses_list, losses_tensor); all_losses = torch.cat(gathered_losses_list).cpu().tolist()
             except Exception as gather_err: logger.error(f"Rank {self.rank}: Val loss gather failed: {gather_err}. Using local."); all_losses = batch_losses
        else: all_losses = batch_losses

        # Calculate metrics on main process
        metrics = {}
        if self.is_main:
             if not all_losses: avg_loss = float('inf'); logger.warning("No valid validation losses collected.")
             else: avg_loss = sum(all_losses) / len(all_losses)
             perplexity = float('inf')
             if np.isfinite(avg_loss):
                 try: perplexity = math.exp(min(avg_loss, 700)) # Cap loss
                 except Exception: perplexity = float('inf')
             metrics = {'val_loss': avg_loss, 'val_perplexity': perplexity}; self.last_val_metrics = metrics
             logger.info(f"Validation Epoch {self.current_epoch + 1} | Avg Loss: {metrics['val_loss']:.4f} | PPL: {metrics['val_perplexity']:.2f}")
             if self.wandb_enabled and self.wandb_run:
                 try: wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": self.current_epoch + 1}, step=self.global_step)
                 except Exception as wb_err: logger.error(f"WandB validation log failed: {wb_err}")
        if hasattr(val_iterator, 'close'): val_iterator.close()
        return metrics

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict] = None):
        if not self.is_main: return
        state_indicator = f"step_{self.global_step}" if is_intermediate else f"epoch_{self.current_epoch+1}_final"
        current_metrics = metrics if metrics is not None else self.last_val_metrics; metric_str = ""
        if current_metrics:
            val_loss = current_metrics.get('val_loss'); train_loss = current_metrics.get('train_loss_cycle')
            mkey, mval = ('vloss', val_loss) if val_loss is not None and np.isfinite(val_loss) else \
                         ('tloss', train_loss) if train_loss is not None and np.isfinite(train_loss) else (None, None)
            if mkey: mf = f"{mval:.2e}" if abs(mval) < 1e-3 and mval != 0 else f"{mval:.3f}"; metric_str = f"_{mkey}{mf}"
        state_indicator += metric_str; filename = f"checkpoint_{state_indicator}.pt"; filepath = os.path.join(self.checkpoint_dir, filename)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        # Clean state dict: Remove '.manifold' refs if they accidentally get saved
        cleaned_model_state = {k: v for k, v in model_to_save.state_dict().items() if not k.endswith('.manifold')}
        checkpoint = {'epoch': self.current_epoch, 'global_step': self.global_step, 'model_state_dict': cleaned_model_state, 'optimizer_state_dict': self.optimizer.state_dict(), 'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None, 'metrics': current_metrics, 'amp_enabled': self.use_amp, 'args': getattr(self, 'args', None), 'wubu_config': getattr(model_to_save, 'wubu_config', {}), 'sequence_config': getattr(model_to_save, 'sequence_config', {})}

        # Save Q-controller state
        if self.has_q_controller and self.optimizer.q_controller:
             try: # Serialize Q-controller state safely
                 # Ensure state keys (tuples) are converted to lists for JSON compatibility if needed, but torch.save handles tuples
                 q_state_data = {'q_table': {k: {p: v.tolist() for p, v in qs.items()} for k, qs in self.optimizer.q_controller.q_table.items()},
                                  'epsilon': self.optimizer.q_controller.epsilon, 'access_count': dict(self.optimizer.q_controller.q_table_access_count),
                                  'creation_time': self.optimizer.q_controller.q_table_creation_time, 'loss_window': list(self.optimizer.q_controller.loss_window),
                                  'grad_norm_window': list(self.optimizer.q_controller.grad_norm_window), 'performance_window': list(self.optimizer.q_controller.performance_window),
                                  'stable_steps': self.optimizer.q_controller.stable_steps, 'oscillation_counter': self.optimizer.q_controller.oscillation_counter,
                                  'prev_loss': self.optimizer.q_controller.prev_loss, 'prev_state': self.optimizer.q_controller.prev_state, 'prev_action': self.optimizer.q_controller.prev_action}
                 checkpoint['q_controller_state'] = q_state_data
             except Exception as q_save_err: logger.error(f"Error preparing Q-Controller state save: {q_save_err}")
        temp_filepath = filepath + ".tmp." + str(random.randint(1000,9999)) # Add random suffix
        try:
            torch.save(checkpoint, temp_filepath)
            os.replace(temp_filepath, filepath) # Atomic replace if possible
            logger.info(f"Checkpoint saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed save checkpoint {filepath}: {e}", exc_info=True)
        finally: # Ensure temp file removal even if replace fails
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError as rm_err: logger.error(f"Failed remove temp ckpt {temp_filepath}: {rm_err}")


    def load_checkpoint(self, filepath: str) -> int:
        if not os.path.exists(filepath): logger.error(f"Checkpoint not found: {filepath}"); return 0
        try:
            checkpoint = torch.load(filepath, map_location='cpu'); logger.info(f"Loading checkpoint: {filepath}")
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            # Load state dict non-strictly
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible_keys.missing_keys: logger.warning(f"Missing model keys: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys: logger.warning(f"Unexpected model keys: {incompatible_keys.unexpected_keys}")

            # Load Optimizer state
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Move optimizer state to device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                try: state[k] = v.to(self.device)
                                except Exception as e_state: logger.warning(f"Failed moving optim state '{k}' to {self.device}: {e_state}")
                    logger.info("Optimizer state loaded.")
                except Exception as optim_ex: logger.warning(f"Failed loading optimizer state: {optim_ex}. State reset.", exc_info=True)
            else: logger.warning("Optimizer state missing. Starts from scratch.")

            # Load AMP Scaler state
            saved_amp_enabled = checkpoint.get('amp_enabled', False)
            if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and saved_amp_enabled:
                 try: self.scaler.load_state_dict(checkpoint['scaler_state_dict']); logger.info("AMP scaler state loaded.")
                 except Exception as scaler_ex: logger.warning(f"Failed loading AMP scaler state: {scaler_ex}. State reset.")
            elif saved_amp_enabled and not self.use_amp: logger.warning("Checkpoint saved with AMP, but AMP is disabled now.")
            elif self.use_amp and not saved_amp_enabled: logger.warning("Checkpoint saved without AMP, but AMP is enabled now.")

            # Load Epoch and Step
            start_epoch = checkpoint.get('epoch', -1) + 1; self.global_step = checkpoint.get('global_step', 0)
            self.current_epoch = start_epoch - 1 if start_epoch > 0 else 0; self.last_val_metrics = checkpoint.get('metrics')
            if self.last_val_metrics: logger.info(f"Restored metrics: {self.last_val_metrics}")

            # Load Q-controller state
            if self.has_q_controller and self.optimizer.q_controller and 'q_controller_state' in checkpoint:
                 q_state = checkpoint['q_controller_state']; logger.info("Attempting load Q-Controller state...")
                 try:
                     # Convert keys back to tuples when loading from checkpoint
                     q_table_loaded = {tuple(k) if isinstance(k, list) else k: {p: np.array(v, dtype=np.float32) for p, v in qs.items()} for k, qs in q_state.get('q_table', {}).items()}
                     self.optimizer.q_controller.q_table = q_table_loaded
                     self.optimizer.q_controller.epsilon = q_state.get('epsilon', self.optimizer.q_controller.epsilon)
                     access_count_loaded = {tuple(k) if isinstance(k, list) else k: v for k,v in q_state.get('access_count', {}).items()}
                     self.optimizer.q_controller.q_table_access_count = defaultdict(int, access_count_loaded)
                     creation_time_loaded = {tuple(k) if isinstance(k, list) else k: v for k,v in q_state.get('creation_time', {}).items()}
                     self.optimizer.q_controller.q_table_creation_time = creation_time_loaded
                     maxlen_loss = self.optimizer.q_controller.loss_window.maxlen; self.optimizer.q_controller.loss_window = deque(q_state.get('loss_window', []), maxlen=maxlen_loss)
                     maxlen_grad = self.optimizer.q_controller.grad_norm_window.maxlen; self.optimizer.q_controller.grad_norm_window = deque(q_state.get('grad_norm_window', []), maxlen=maxlen_grad)
                     maxlen_perf = self.optimizer.q_controller.performance_window.maxlen; self.optimizer.q_controller.performance_window = deque(q_state.get('performance_window', []), maxlen=maxlen_perf)
                     self.optimizer.q_controller.stable_steps = q_state.get('stable_steps', 0); self.optimizer.q_controller.oscillation_counter = q_state.get('oscillation_counter', 0)
                     # Ensure prev_state is loaded as tuple if not None
                     prev_state_loaded = q_state.get('prev_state')
                     self.optimizer.q_controller.prev_loss = q_state.get('prev_loss'); self.optimizer.q_controller.prev_state = tuple(prev_state_loaded) if prev_state_loaded else None; self.optimizer.q_controller.prev_action = q_state.get('prev_action')
                     logger.info("Q-Controller state loaded.")
                 except Exception as q_load_err: logger.warning(f"Failed loading Q-Controller state: {q_load_err}. State may be reset.", exc_info=True)
            elif self.has_q_controller: logger.warning("Q-Controller enabled but state not found in checkpoint.")

            # Check Args / Config Consistency
            loaded_args = checkpoint.get('args')
            # Perform comparison logic if needed (omitted for brevity)

            model_to_load.to(self.device); logger.info(f"Checkpoint '{os.path.basename(filepath)}' loaded. Resume Epoch {start_epoch} (Glob Step {self.global_step})")
            return start_epoch
        except FileNotFoundError: logger.error(f"Load ckpt failed: File not found '{filepath}'")
        except Exception as e: logger.error(f"Failed load ckpt '{filepath}': {e}", exc_info=True)
        return 0 # Return 0 if loading failed

    def train(self, epochs: int, start_epoch: int = 0):
        self.current_epoch = start_epoch
        if self.is_main: logger.info(f"Starting training from Epoch {start_epoch + 1}/{epochs} (Global Step: {self.global_step}).")
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            if self.is_main: logger.info(f"--- Starting Epoch {epoch + 1}/{epochs} ---")
            # Set epoch for samplers/datasets
            if isinstance(self.train_loader.sampler, DistributedSampler): self.train_loader.sampler.set_epoch(epoch)
            if self.val_loader and isinstance(self.val_loader.sampler, DistributedSampler): self.val_loader.sampler.set_epoch(epoch)
            if hasattr(self.train_loader.dataset, 'set_epoch'): self.train_loader.dataset.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader.dataset, 'set_epoch'): self.val_loader.dataset.set_epoch(epoch)
            # Train -> Validate -> Save
            self._train_epoch()
            val_metrics = self._validate()
            # Save checkpoint at end of epoch (main process only)
            if self.is_main and self.save_interval >= 0: # Check if saving is enabled
                self._save_checkpoint(is_intermediate=False, metrics=val_metrics if val_metrics else None)
            if self.world_size > 1: logger.debug(f"Rank {self.rank} entering end-of-epoch barrier."); torch.distributed.barrier(); logger.debug(f"Rank {self.rank} exited barrier.")
        if self.is_main: logger.info(f"Training finished after {epochs} epochs.")

# =====================================================================
# Default Configuration (Copied from v0.03, matches WuBu core defaults)
# =====================================================================
DEFAULT_CONFIG_WUBU = {
    "num_levels": 3, "hyperbolic_dims": [128, 64, 32],
    "initial_curvatures": [1.0, 0.5, 0.25], "initial_scales": [1.0, 1.0, 1.0],
    "initial_spread_values": [1.0, 1.0, 1.0], "boundary_points_per_level": [5, 4, 3],
    "learnable_curvature": True, "learnable_scales": True, "learnable_spread": True,
    "curvature_min_value": 1e-6, "scale_min_value": 1e-6, "spread_min_value": 1e-6,
    "use_level_descriptors": True, "use_level_spread": True, "level_descriptor_init_scale": 0.01,
    "relative_vector_aggregation": "mean", "tangent_input_combination_dims": [64],
    "use_tangent_flow": True, "tangent_flow_type": "mlp", "tangent_flow_hidden_dim_ratio": 0.5,
    "tangent_flow_scale": 1.0,
    # Transition params (lengths checked/adjusted in run())
    "rotation_types": ["so_n", "so_n"], # Not used in this version
    "transform_types": ["linear", "linear"],
    "transform_hidden_dims": [None, None],
    "aggregation_method": "concat_tangent", "dropout": 0.1,
}


# =====================================================================
# Argument Parsing (Validated)
# =====================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu Nesting Model Trainer (v0.04 - Hyperbolic Core - Corrected)")
    # --- DDP ---
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP.')
    # --- Data ---
    parser.add_argument('--data_path', type=str, required=True, help='Path training data (.npy)')
    parser.add_argument('--val_data_path', type=str, default=None, help='Path validation data (.npy)')
    parser.add_argument('--context_window', type=int, default=512, help='Sequence length')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of training data')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers per GPU')
    # --- Training ---
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Global batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--no_amp', action='store_true', help='Disable Automatic Mixed Precision')
    parser.add_argument('--detect_anomaly', action='store_true', help='Enable autograd anomaly detection (slow)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # --- Checkpointing & Logging ---
    parser.add_argument('--checkpoint_dir', type=str, default='wubu_nest_checkpoints_v04_hyp', help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N global steps')
    parser.add_argument('--save_interval', type=int, default=2000, help='Save intermediate checkpoint every N global steps (0 to disable)')
    # --- WandB ---
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='WuBuNestingHyperbolic', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team)')
    # --- Sequence Model Config ---
    parser.add_argument('--local_hidden_size', type=int, default=256, help='Hidden size for local encoder/decoder (Tangent space)')
    parser.add_argument('--decoder_memory_dim', type=int, default=512, help='Output dimension of WuBu model (Tangent space)')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of layers in Local Encoder (Transformer)')
    parser.add_argument('--num_encoder_heads', type=int, default=8, help='Number of heads in Local Encoder (Transformer)')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of layers in Local Decoder (Transformer)')
    parser.add_argument('--num_decoder_heads', type=int, default=8, help='Number of heads in Local Decoder (Transformer)')
    parser.add_argument('--n_gram_sizes', type=int, nargs='+', default=[], help='N-gram sizes for local encoder (Euclidean)')
    parser.add_argument('--n_gram_vocab_size', type=int, default=30000, help='Vocab size for N-gram embeddings (Euclidean)')
    parser.add_argument('--use-hierarchical-decoder', action=argparse.BooleanOptionalAction, default=True, help='Use hierarchical prediction head (Euclidean)')
    # --- WuBu Nesting Config (Overrides Defaults) ---
    parser.add_argument('--num_levels', type=int, default=DEFAULT_CONFIG_WUBU['num_levels'], help='Number of hyperbolic levels')
    parser.add_argument('--hyperbolic_dims', type=int, nargs='+', default=None, help='Dimensions for each hyperbolic level (overrides default)')
    parser.add_argument('--initial_curvatures', type=float, nargs='+', default=None, help='Initial curvatures for each level (overrides default)')
    parser.add_argument('--initial_scales', type=float, nargs='+', default=None, help='Initial scales for each level (overrides default)')
    parser.add_argument('--initial_spread_values', type=float, nargs='+', default=None, help='Initial spread values (defaults to scales if None)')
    parser.add_argument('--boundary_points_per_level', type=int, nargs='+', default=None, help='Number of boundary points per level (overrides default)')
    parser.add_argument('--learnable-curvature', dest='learnable_curvature', action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG_WUBU['learnable_curvature'])
    parser.add_argument('--learnable-scales', dest='learnable_scales', action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG_WUBU['learnable_scales'])
    parser.add_argument('--learnable-spread', dest='learnable_spread', action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG_WUBU['learnable_spread'])
    parser.add_argument('--use-level-descriptors', dest='use_level_descriptors', action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG_WUBU['use_level_descriptors'])
    parser.add_argument('--use-level-spread', dest='use_level_spread', action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG_WUBU['use_level_spread'])
    parser.add_argument('--use-tangent-flow', dest='use_tangent_flow', action=argparse.BooleanOptionalAction, default=DEFAULT_CONFIG_WUBU['use_tangent_flow'])
    parser.add_argument('--curvature_min_value', type=float, default=DEFAULT_CONFIG_WUBU['curvature_min_value'], help='Minimum curvature value')
    parser.add_argument('--scale_min_value', type=float, default=DEFAULT_CONFIG_WUBU['scale_min_value'], help='Minimum scale value')
    parser.add_argument('--spread_min_value', type=float, default=DEFAULT_CONFIG_WUBU['spread_min_value'], help='Minimum spread value')
    parser.add_argument('--level_descriptor_init_scale', type=float, default=DEFAULT_CONFIG_WUBU['level_descriptor_init_scale'], help='Initialization scale for level descriptors and boundaries')
    parser.add_argument('--relative_vector_aggregation', type=str, default=DEFAULT_CONFIG_WUBU['relative_vector_aggregation'], choices=['mean', 'sum', 'none'], help='Aggregation method for relative vectors (Tangent)')
    parser.add_argument('--tangent_input_combination_dims', type=int, nargs='+', default=DEFAULT_CONFIG_WUBU['tangent_input_combination_dims'], help='Hidden dims for tangent input combiner MLP')
    parser.add_argument('--tangent_flow_type', type=str, default=DEFAULT_CONFIG_WUBU['tangent_flow_type'], choices=['mlp', 'linear', 'none'], help='Type of tangent flow module')
    parser.add_argument('--tangent_flow_hidden_dim_ratio', type=float, default=DEFAULT_CONFIG_WUBU['tangent_flow_hidden_dim_ratio'], help='Hidden dim ratio for MLP tangent flow')
    parser.add_argument('--tangent_flow_scale', type=float, default=DEFAULT_CONFIG_WUBU['tangent_flow_scale'], help='Scaling factor for tangent flow displacement')
    parser.add_argument('--aggregation_method', type=str, default=DEFAULT_CONFIG_WUBU['aggregation_method'], choices=['concat_tangent'], help='Method to aggregate level tangent outputs')
    parser.add_argument('--dropout', type=float, default=DEFAULT_CONFIG_WUBU['dropout'], help='Dropout rate')
    # --- Q-Controller Config ---
    parser.add_argument('--enable-q-controller', dest='q_controller_enabled', action=argparse.BooleanOptionalAction, default=True, help='Enable the Q-learning optimizer controller')
    parser.add_argument('--q_learning_rate', type=float, default=0.02, help='Learning rate for Q-controller')
    parser.add_argument('--q_discount', type=float, default=0.95, help='Discount factor for Q-controller')
    parser.add_argument('--q_epsilon', type=float, default=0.25, help='Initial epsilon for Q-controller exploration')
    parser.add_argument('--q_epsilon_decay', type=float, default=0.9999, help='Epsilon decay rate for Q-controller')
    parser.add_argument('--q_min_epsilon', type=float, default=0.02, help='Minimum epsilon for Q-controller')

    args = parser.parse_args()

    # --- Post-processing and Validation of Arguments ---
    if args.hyperbolic_dims is None: args.hyperbolic_dims = DEFAULT_CONFIG_WUBU['hyperbolic_dims']
    if args.initial_curvatures is None: args.initial_curvatures = DEFAULT_CONFIG_WUBU['initial_curvatures']
    if args.initial_scales is None: args.initial_scales = DEFAULT_CONFIG_WUBU['initial_scales']
    if args.boundary_points_per_level is None: args.boundary_points_per_level = DEFAULT_CONFIG_WUBU['boundary_points_per_level']
    if args.initial_spread_values is None: args.initial_spread_values = args.initial_scales # Default to scales

    num_levels = args.num_levels
    list_args_to_check = {'hyperbolic_dims': num_levels, 'initial_curvatures': num_levels, 'initial_scales': num_levels, 'boundary_points_per_level': num_levels, 'initial_spread_values': num_levels}
    for arg_name, expected_len in list_args_to_check.items():
        current_list = getattr(args, arg_name)
        if len(current_list) != expected_len:
            parser.error(f"--{arg_name} requires {expected_len} values for --num_levels={num_levels}, but got {len(current_list)}")

    # Transition params are handled in run() when building final config
    args.rotation_types = DEFAULT_CONFIG_WUBU['rotation_types']
    args.transform_types = DEFAULT_CONFIG_WUBU['transform_types']
    args.transform_hidden_dims = DEFAULT_CONFIG_WUBU['transform_hidden_dims']

    return args

# =====================================================================
# Distributed Setup Utilities (Validated)
# =====================================================================
def setup_distributed(local_rank):
    ddp_active = local_rank != -1; device = torch.device("cpu"); rank = 0; world_size = 1
    if ddp_active:
        if not torch.cuda.is_available(): raise RuntimeError("DDP requested but CUDA not available.")
        if not is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'; master_addr = os.getenv('MASTER_ADDR', 'localhost'); master_port = os.getenv('MASTER_PORT', '12355')
            if os.getenv('WORLD_SIZE') is not None and int(os.getenv('WORLD_SIZE')) > 1 and (os.getenv('MASTER_ADDR') is None or os.getenv('MASTER_PORT') is None): logger.warning("MASTER_ADDR/PORT not set for multi-node DDP. Using defaults.")
            try: init_process_group(backend=backend, timeout=timedelta(seconds=1800)); rank = get_rank(); world_size = get_world_size(); device = torch.device(f"cuda:{local_rank}"); torch.cuda.set_device(device); logger.info(f"DDP Initialized: Rank {rank}/{world_size} on {device} ({backend}). Master: {master_addr}:{master_port}")
            except Exception as e: logger.error(f"DDP Init failed: {e}", exc_info=True); raise
        else: rank = get_rank(); world_size = get_world_size(); device = torch.device(f"cuda:{local_rank}"); torch.cuda.set_device(device); logger.warning(f"DDP already initialized: Rank {rank}/{world_size} on {device}.")
    else:
        if torch.cuda.is_available(): device = torch.device("cuda:0"); torch.cuda.set_device(device); logger.info("DDP disabled. Using single GPU: cuda:0")
        else: device = torch.device("cpu"); logger.info("DDP disabled. Using CPU.")
    return ddp_active, device, rank, world_size

def is_main_process():
    if is_initialized(): return get_rank() == 0
    return True

# =====================================================================
# Main Execution Logic (Corrected)
# =====================================================================
def run():
    args = parse_arguments()
    temp_is_main = args.local_rank == -1 or args.local_rank == 0
    initial_log_level = logging.INFO if temp_is_main else logging.WARNING
    root_logger = logging.getLogger(); [root_logger.removeHandler(h) for h in root_logger.handlers[:]]
    logging.basicConfig(level=initial_log_level, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)
    ddp_active, device, rank, world_size = setup_distributed(args.local_rank)
    am_main_process = is_main_process()
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.getLogger().setLevel(log_level)
    if am_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True); log_filename = os.path.join(args.checkpoint_dir, f"train_hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log")
        try:
            file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8'); file_handler.setLevel(logging.INFO); file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler); logger.info(f"Main process (Rank {rank}) logging to: {log_filename}")
        except Exception as e: logger.error(f"Failed file logging Rank {rank}: {e}")

    logger.info("="*60 + f"\n--- WuBu Nesting Trainer (v0.04 - Hyperbolic Core - Corrected v2 - Rank {rank}/{world_size}) ---") # Updated version string
    logger.info(f"Status | DDP: {'Active' if ddp_active else 'Inactive'} | Device: {device} | Main: {am_main_process}")
    if am_main_process: logger.info("--- Args ---\n" + "\n".join([f"  --{k}: {v}" for k,v in sorted(vars(args).items())]) + "\n" + "-"*20)
    logger.info(f"System | OS={platform.system()}/{platform.release()}, Py={sys.version.split()[0]}, Torch={torch.__version__}, CUDA={'Yes ('+torch.version.cuda+')' if torch.cuda.is_available() else 'No'}")
    logger.info("="*60)

    seed = args.seed + rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed} for Rank {rank}")

    # --- Build Configuration Dicts ---
    wubu_config = DEFAULT_CONFIG_WUBU.copy()
    for key in wubu_config.keys(): # Update from args
        if hasattr(args, key):
            arg_val = getattr(args, key)
            if isinstance(wubu_config[key], bool) and isinstance(arg_val, bool): wubu_config[key] = arg_val
            elif arg_val is not None: wubu_config[key] = arg_val
    # Correct transition list lengths
    num_levels = wubu_config['num_levels']; num_transitions = max(0, num_levels - 1)
    for key in ['transform_types', 'transform_hidden_dims', 'rotation_types']:
        if key in wubu_config and isinstance(wubu_config[key], list) and len(wubu_config[key]) != num_transitions:
             logger.warning(f"Correcting length of WuBu config '{key}'. Expected {num_transitions}. Repeating first.")
             first_val = wubu_config[key][0] if wubu_config[key] else DEFAULT_CONFIG_WUBU[key][0]
             wubu_config[key] = [first_val] * num_transitions

    sequence_config = { k: getattr(args, k) for k in [
        "local_hidden_size", "decoder_memory_dim", "context_window", "n_gram_sizes",
        "n_gram_vocab_size", "num_encoder_layers", "num_decoder_layers",
        "num_encoder_heads", "num_decoder_heads", "use_hierarchical_decoder" ] if hasattr(args, k)}
    sequence_config["vocab_size"] = 256

    if am_main_process:
        logger.info("--- Final WuBu Config (Hyperbolic v0.04) ---"); [logger.info(f"  {k}: {v}") for k,v in sorted(wubu_config.items())]
        logger.info("--- Final Sequence Config ---"); [logger.info(f"  {k}: {v}") for k,v in sorted(sequence_config.items())]
        logger.info("--------------------------")

    # --- Init WandB ---
    use_wandb = args.wandb and am_main_process and WANDB_AVAILABLE
    if use_wandb:
        try:
            full_config = {**vars(args), "wubu_config": wubu_config, "sequence_config": sequence_config}; sanitized_config = {}
            for k, v in full_config.items(): sanitized_config[k] = tuple(v) if isinstance(v, list) else 'None' if v is None else vars(v) if isinstance(v, argparse.Namespace) else v
            run_name = f"HypWuBuV04_L{wubu_config['num_levels']}_D{'x'.join(map(str, wubu_config['hyperbolic_dims']))}_B{args.batch_size}_LR{args.learning_rate:.1e}_{datetime.now().strftime('%H%M')}"
            run_id = hashlib.sha1(f"{run_name}_{args.seed}".encode()).hexdigest()[:10] if args.resume is None else None
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=sanitized_config, name=run_name, resume="allow", id=run_id)
            logger.info(f"WandB initialized: project='{args.wandb_project}', run='{wandb.run.name}' (ID: {wandb.run.id})")
        except Exception as e: logger.warning(f"WandB init failed: {e}. Disabling.", exc_info=True); use_wandb = False

    # --- Load Datasets ---
    try:
        train_dataset = ByteIterableDataset(args.data_path, args.context_window, args.data_fraction); train_dataset.set_seed(args.seed)
        val_dataset = None
        if args.val_data_path and os.path.exists(args.val_data_path): val_dataset = ByteIterableDataset(args.val_data_path, args.context_window, 1.0); val_dataset.set_seed(args.seed + 1)
        elif args.val_data_path: logger.warning(f"Val path {args.val_data_path} not found.")
    except Exception as e: logger.error(f"Dataset init failed: {e}", exc_info=True); sys.exit(1)

    # --- Create DataLoaders ---
    batch_size_per_gpu = max(1, args.batch_size // world_size) if args.batch_size >= world_size else 1
    effective_bs = batch_size_per_gpu * world_size * args.grad_accum_steps
    logger.info(f"Batch Cfg | Global:{args.batch_size} PerGPU:{batch_size_per_gpu} Accum:{args.grad_accum_steps} Eff:{effective_bs}")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed, drop_last=True) if ddp_active else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if ddp_active and val_dataset else None
    base_seed_offset = args.seed * world_size
    train_worker_init_fn = functools.partial(seed_worker, base_seed=base_seed_offset, rank_offset=rank * args.num_workers) if args.num_workers > 0 else None
    val_worker_init_fn = functools.partial(seed_worker, base_seed=base_seed_offset + 1, rank_offset=rank * args.num_workers) if args.num_workers > 0 and val_dataset else None
    use_persistent_workers = (args.num_workers > 0) and (platform.system() != 'Windows')
    logger.info(f"Persistent workers: {use_persistent_workers} (Num Workers: {args.num_workers})")
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=train_worker_init_fn, persistent_workers=use_persistent_workers, shuffle=False) # Must Always be set to false we never shuffle data due to dataloader bug ValueError: DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=True
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=val_worker_init_fn, persistent_workers=use_persistent_workers, shuffle=False) if val_dataset else None

    # --- Initialize Hyperbolic Model ---
    try:
        model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config).to(device)
        if am_main_process:
             total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
             logger.info(f"Hyperbolic Model Init | Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    except Exception as model_ex: logger.error(f"Model init failed: {model_ex}", exc_info=True); sys.exit(1)

    # --- Wrap Model for DDP ---
    if ddp_active:
         find_unused = False # Usually False unless debugging DDP issues
         model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=find_unused, gradient_as_bucket_view=True)
         logger.info(f"Model DDP wrapped Rank {rank} (find_unused={find_unused})."); torch.distributed.barrier()

    # --- Initialize Riemannian Optimizer ---
    q_cfg = None if not args.q_controller_enabled else {"learning_rate": args.q_learning_rate, "discount": args.q_discount, "epsilon": args.q_epsilon, "epsilon_decay": args.q_epsilon_decay, "min_epsilon": args.q_min_epsilon}
    try:
        optimizer = RiemannianEnhancedSGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, q_learning_config=q_cfg)
        logger.info(f"Optimizer '{type(optimizer).__name__}' Rank {rank} | BaseLR={args.learning_rate}, WD={args.weight_decay}.")
    except Exception as optim_ex: logger.error(f"Optimizer init failed: {optim_ex}", exc_info=True); sys.exit(1)

    # --- Initialize Trainer ---
    trainer = Trainer(model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader, grad_accum_steps=args.grad_accum_steps, use_amp=(not args.no_amp), log_interval=args.log_interval, save_interval=args.save_interval, checkpoint_dir=args.checkpoint_dir, wandb_enabled=use_wandb, max_grad_norm=args.max_grad_norm, rank=rank, world_size=world_size, detect_anomaly=args.detect_anomaly)
    trainer.args = args

    # --- Load Checkpoint ---
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume): start_epoch = trainer.load_checkpoint(args.resume)
        else: logger.warning(f"Resume ckpt not found: {args.resume}. Starting fresh.")
        if ddp_active: torch.distributed.barrier()
    else: logger.info(f"Rank {rank}: Starting fresh training.")

    # --- Start Training ---
    if ddp_active: torch.distributed.barrier()
    training_successful = False
    try:
        trainer.train(args.epochs, start_epoch=start_epoch)
        training_successful = True
    except KeyboardInterrupt: logger.info(f"Training interrupted by user (Rank {rank})."); training_successful = True # Allow saving on interrupt
    except Exception as train_ex: logger.error(f"Critical training error Rank {rank}: {train_ex}", exc_info=True); training_successful = False
    finally: # Cleanup
        if am_main_process: logger.info("Saving final checkpoint..."); trainer._save_checkpoint(is_intermediate=False, metrics=getattr(trainer, 'last_val_metrics', None))
        if ddp_active:
            try: destroy_process_group(); logger.info(f"DDP group destroyed (Rank {rank}).")
            except Exception as ddp_err: logger.error(f"DDP destroy error Rank {rank}: {ddp_err}")
        if use_wandb and wandb.run:
            try: wandb.finish(); logger.info("WandB finished.")
            except Exception as wb_err: logger.error(f"WandB finish error: {wb_err}")

    logger.info(f"Script finished (Rank {rank}). Status: {'Success' if training_successful else 'Failed'}")

# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    # Launch command:
    # torchrun --standalone --nproc_per_node=NUM_GPUS WuBuNest_Trainer.py [ARGS]
    run()
