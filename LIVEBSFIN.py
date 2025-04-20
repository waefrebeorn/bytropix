#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIVEBSFIN.py - Hash-Indexed Integrated Continual Learning Framework (v1.0 - Wubu Concept)

Implements the continual learning loop for the BSFIN model using the
hash-indexed memory concept discussed. Integrates actual BSFIN components
copied from bsfin_main.py (v2 - Syntax Fixed v5).

Core Idea:
1. Use BabylonIndex hashing on input sequences.
2. Maintain memory mapping hashes to metadata (last access, importance).
3. Modulate gradients based on metadata before optimizer step.
4. Implement time-based importance decay (forgetting).
5. Ground in a base model checkpoint.
"""

# --- Imports ---
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset # Dataset needed for wrapper
import numpy as np
import math
import random
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque, defaultdict # Use defaultdict for memory
import gc
import socket
import platform
from torch import amp  # For automatic mixed precision
from dataclasses import dataclass, field
import itertools
from tqdm import tqdm
import hashlib # For hashing sequences
# Imports potentially needed by copied code (ensure these cover all needs)
from torch.nn.parallel import DistributedDataParallel # Not used in live script, but maybe in loaded components?
from torch.distributed import init_process_group, destroy_process_group, is_initialized # Not used directly here

# Try importing wandb, but make it optional (copied from bsfin_main.py)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# --- Logging Setup ---
logger = logging.getLogger("LIVEBSFIN")
# Basic configuration will be set in main()

# =====================================================================
# COPIED COMPONENTS from bsfin_main.py
# =====================================================================
# ** These are copied directly from the uploaded bsfin_main.py **

# --- Configs ---
@dataclass
class SamplerConfig:
    """Configuration for entropy-based sampling (used in generate)."""
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# --- Quantum Noise ---
def add_quantum_noise(tensor, noise_prob=0.05, noise_scale=0.1, noise_type="phase_and_amplitude"):
    """Inject quantum-inspired noise."""
    if noise_scale <= 0 or not tensor.requires_grad or not tensor.is_floating_point():
        return tensor
    with torch.no_grad():
        if noise_type == "phase_only":
            mask = torch.rand_like(tensor) < noise_prob; flip = torch.where(mask, -1.0, 1.0); noisy_tensor = tensor * flip
        elif noise_type == "amplitude_only":
            mask = torch.rand_like(tensor) < noise_prob; noise = torch.randn_like(tensor) * noise_scale * mask; noisy_tensor = tensor + noise
        else:
            phase_mask = torch.rand_like(tensor) < noise_prob; amp_mask = torch.rand_like(tensor) < noise_prob
            flip = torch.where(phase_mask, -1.0, 1.0); noise = torch.randn_like(tensor) * noise_scale * amp_mask
            noisy_tensor = tensor * flip + noise
    return noisy_tensor.clone().requires_grad_(tensor.requires_grad)

# --- Tokenizer ---
class ByteTokenizer:
    """Simple tokenizer for byte-level processing."""
    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of byte values."""
        return list(text.encode('utf-8'))
    def decode(self, byte_sequence: Iterable[Union[int, torch.Tensor]]) -> str:
        """Decodes a sequence of byte values back into a string."""
        valid_bytes = []
        for b in byte_sequence:
            val = None
            if isinstance(b, (int, np.integer)): val = int(b)
            elif isinstance(b, torch.Tensor):
                b_item = b.item();
                if isinstance(b_item, int): val = b_item
                elif isinstance(b_item, float): val = int(b_item)
            if val is not None and 0 <= val <= 255: valid_bytes.append(val)
        return bytes(valid_bytes).decode('utf-8', errors='replace')

# --- Dataset ---
class ByteIterableDataset(IterableDataset):
    """Byte-level dataset for efficient streaming from large numpy files."""
    def __init__(self, npy_file_path: str, context_size: int = 128, data_fraction: float = 1.0):
        self.npy_file_path = npy_file_path
        self.context_size = context_size
        self.data_fraction = max(0.0, min(1.0, data_fraction))
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"File not found: {npy_file_path}")
        try:
            mmap_data = np.load(self.npy_file_path, mmap_mode='r')
            self.full_data_size = mmap_data.shape[0]
            del mmap_data
            gc.collect()
            self.data_size = int(self.full_data_size * self.data_fraction)
            if self.data_size <= self.context_size:
                raise ValueError(f"Dataset size after fraction ({self.data_size}) is too small for context size ({self.context_size}). Needs at least {self.context_size + 1} bytes.")
            logger.info(f"Using {self.data_size}/{self.full_data_size} bytes ({self.data_fraction:.1%}) from {npy_file_path}")
        except Exception as e:
            logger.error(f"Error accessing or checking size of {npy_file_path}: {e}", exc_info=True)
            raise
    def __len__(self):
        return max(0, self.data_size - self.context_size)
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        bytes_data = None
        try:
            bytes_data = np.load(self.npy_file_path, mmap_mode='r')
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to load {self.npy_file_path}: {e}", exc_info=True)
            return
        try:
            effective_data_len = self.data_size
            num_indices = max(0, effective_data_len - self.context_size)
            indices_per_worker = num_indices // num_workers
            start_idx = worker_id * indices_per_worker
            remainder = num_indices % num_workers
            if worker_id < remainder: start_idx += worker_id; indices_per_worker += 1
            else: start_idx += remainder
            end_idx = start_idx + indices_per_worker
            if start_idx >= end_idx: return
            seed = (worker_id + int(time.time() * 1000) + os.getpid()) % (2**32)
            rng = np.random.default_rng(seed=seed)
            worker_indices = rng.permutation(np.arange(start_idx, end_idx))
            for idx in worker_indices:
                if idx + self.context_size + 1 <= effective_data_len:
                    try:
                        context = np.copy(bytes_data[idx : idx + self.context_size])
                        target = np.copy(bytes_data[idx + self.context_size])
                        context_tensor = torch.tensor(context, dtype=torch.long)
                        target_tensor = torch.tensor(target, dtype=torch.long)
                        yield context_tensor, target_tensor
                    except IndexError: logger.warning(f"Worker {worker_id}: IndexError accessing data at index {idx}. Skipping.")
        finally:
            if bytes_data is not None:
                 if hasattr(bytes_data, '_mmap') and bytes_data._mmap is not None:
                      try: bytes_data._mmap.close()
                      except Exception: pass # Ignore close errors
                 del bytes_data; gc.collect()

# --- Babylon Index ---
class BabylonIndex:
    """Entropy-based index for byte sequence analysis and patching."""
    def __init__(self, scales: List[int] = [3, 5, 7], max_cache_size: int = 50000, min_entropy_threshold: float = 0.5):
        self.scales = sorted(list(set(scales)))
        self.entropy_cache = {}
        self.max_cache_size = max_cache_size
        self.min_entropy_threshold = min_entropy_threshold
    def _clean_cache(self):
        if len(self.entropy_cache) > self.max_cache_size:
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5)
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                if k in self.entropy_cache: del self.entropy_cache[k]
    def _is_valid_utf8_boundary(self, byte_seq: Union[List[int], np.ndarray], boundary: int) -> bool:
        if boundary <= 0 or boundary >= len(byte_seq): return True
        byte_at_boundary = byte_seq[boundary]; return not (0x80 <= byte_at_boundary <= 0xBF)
    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...]]) -> float:
        cache_key = None; byte_window_np = None
        if isinstance(byte_window, tuple):
            cache_key = byte_window;
            if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
            if not byte_window: return 0.0
            byte_window_np = np.array(byte_window, dtype=np.uint8)
        elif isinstance(byte_window, np.ndarray):
            if byte_window.size == 0: return 0.0
            byte_window_np = byte_window
        else: logger.warning(f"Unsupported type for compute_entropy: {type(byte_window)}"); return 0.0
        try:
            if not np.issubdtype(byte_window_np.dtype, np.integer): byte_window_np = byte_window_np.astype(np.uint8)
            if np.any(byte_window_np < 0) or np.any(byte_window_np > 255): byte_window_np = np.clip(byte_window_np, 0, 255)
            byte_counts = np.bincount(byte_window_np, minlength=256)
        except TypeError as e: logger.error(f"TypeError in np.bincount: {e}"); return 0.0
        total_bytes = byte_counts.sum();
        if total_bytes == 0: return 0.0
        probs = byte_counts[byte_counts > 0] / total_bytes
        entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
        if cache_key is not None: self.entropy_cache[cache_key] = entropy; self._clean_cache()
        return entropy
    def find_patch_boundaries(self, byte_seq_tensor: torch.Tensor) -> List[int]:
        if byte_seq_tensor.numel() == 0: return []
        if byte_seq_tensor.dim() > 1:
            if byte_seq_tensor.size(0) == 1: byte_seq_list = byte_seq_tensor[0].cpu().tolist()
            else: logger.warning("find_patch_boundaries expects 1D or [1, S]. Using first element."); byte_seq_list = byte_seq_tensor[0].cpu().tolist()
        else: byte_seq_list = byte_seq_tensor.cpu().tolist()
        seq_len = len(byte_seq_list); min_scale = min(self.scales, default=1)
        if seq_len < min_scale: return []
        potential_boundaries = set(); window_size = max(min(max(self.scales, default=16), seq_len // 2, 64), min_scale)
        entropies = []
        for i in range(seq_len - window_size + 1): entropies.append((i, self.compute_entropy(tuple(byte_seq_list[i : i + window_size]))))
        entropies.sort(key=lambda x: x[1], reverse=True)
        num_boundaries_target = max(1, seq_len // 128); selected_count = 0
        for start_pos, entropy_val in entropies:
            boundary_candidate = start_pos
            if entropy_val > self.min_entropy_threshold and selected_count < num_boundaries_target * 2:
                if self._is_valid_utf8_boundary(byte_seq_list, boundary_candidate): potential_boundaries.add(boundary_candidate); selected_count += 1
        final_boundaries = sorted([b for b in list(potential_boundaries) if 0 < b < seq_len])
        min_patch_size = 16; merged_boundaries = []
        if final_boundaries:
            last_boundary = 0
            for b in final_boundaries:
                if b - last_boundary >= min_patch_size: merged_boundaries.append(b); last_boundary = b
            if seq_len > 0 and last_boundary > 0 and seq_len - last_boundary < min_patch_size and merged_boundaries: merged_boundaries.pop()
            final_boundaries = merged_boundaries
        return final_boundaries
    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[torch.Tensor]:
        if byte_seq_tensor.dim() != 1:
             if byte_seq_tensor.dim() == 2 and byte_seq_tensor.size(0) == 1: byte_seq_tensor = byte_seq_tensor.squeeze(0)
             else: raise ValueError(f"create_patches expects a 1D tensor, got shape {byte_seq_tensor.shape}")
        boundaries = self.find_patch_boundaries(byte_seq_tensor); patches = []; start_idx = 0; seq_len = byte_seq_tensor.size(0)
        for end_idx in boundaries:
            if start_idx < end_idx <= seq_len:
                patch = byte_seq_tensor[start_idx:end_idx];
                if patch.numel() > 0: patches.append(patch)
                start_idx = end_idx
        if start_idx < seq_len:
            final_patch = byte_seq_tensor[start_idx:];
            if final_patch.numel() > 0: patches.append(final_patch)
        return patches
    def get_sequence_hash(self, byte_seq_tensor: torch.Tensor) -> str:
        """Generates a hash for the input byte sequence. (Added for live learning)"""
        if byte_seq_tensor.dim() > 1: byte_seq_tensor = byte_seq_tensor.squeeze()
        if byte_seq_tensor.dim() != 1: logger.error(f"Cannot hash tensor with shape {byte_seq_tensor.shape}"); return "error_hash"
        try:
            byte_data = bytes(byte_seq_tensor.cpu().numpy().astype(np.uint8).tolist())
            return hashlib.sha256(byte_data).hexdigest()[:16] # Truncate hash
        except Exception as e: logger.error(f"Error hashing sequence: {e}"); return "error_hash"
    @torch.no_grad()
    def reset_context(self): self.entropy_cache = {}

# --- Optimizer Components ---
class GradientStats:
    def __init__(self): self.reset()
    def reset(self): self.total_gradients = 0; self.clipped_gradients = 0; self.max_gradient_norm = 0.0; self.sum_clip_ratios = 0.0; self.step_stats = {}
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        self.total_gradients += 1; self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
        if clipped: self.clipped_gradients += 1; self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)
    def get_step_stats(self) -> dict:
        if self.total_gradients == 0: return {"gradients_clipped": 0, "total_gradients": 0, "clip_ratio_avg": 0.0, "max_gradient": 0.0, "clip_percentage": 0.0}
        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100 if self.total_gradients > 0 else 0.0
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0
        return {"gradients_clipped": self.clipped_gradients, "total_gradients": self.total_gradients, "clip_ratio_avg": avg_clip_ratio, "max_gradient": self.max_gradient_norm, "clip_percentage": clip_percentage}
    def record_step(self, step: int): stats = self.get_step_stats(); self.step_stats[step] = stats; self.reset(); return stats

class QController:
    def __init__(self, learning_rate: float=0.02, discount: float=0.97, epsilon: float=0.15, epsilon_decay: float=0.9995, min_epsilon: float = 0.02, lr_scale_bounds: tuple=(0.85, 1.15), momentum_scale_bounds: tuple=(0.9, 1.1), max_q_table_size: int=15000):
        self.q_table = {}; self.alpha = learning_rate; self.gamma = discount; self.epsilon = epsilon; self.min_epsilon = min_epsilon; self.epsilon_decay = epsilon_decay
        self.prev_loss = None; self.prev_state = None; self.prev_action = None; self.lr_scale_bounds = lr_scale_bounds; self.momentum_scale_bounds = momentum_scale_bounds
        self.loss_window = deque(maxlen=10); self.grad_var_window = deque(maxlen=15); self.lr_window = deque(maxlen=8); self.momentum_window = deque(maxlen=8)
        lr_actions = np.array([0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.15]); mom_actions = np.array([0.9, 0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05, 1.1])
        self.action_ranges = {'lr_scale': lr_actions, 'momentum_scale': mom_actions}
        self.performance_window = deque(maxlen=50); self.stable_steps = 0; self.max_q_table_size = max_q_table_size; self.q_table_access_count = {}
    def get_state(self, lr: float, momentum: float, grad_var: float, loss: float) -> tuple:
        self.loss_window.append(loss); self.grad_var_window.append(grad_var); self.lr_window.append(lr); self.momentum_window.append(momentum)
        loss_trend_bin = 2; grad_var_bin = 2; lr_bin = 2; momentum_bin = 1
        if len(self.loss_window) >= 5:
            y = np.array(list(self.loss_window)[-5:]); x = np.arange(len(y))
            try:
                if np.allclose(y, y[0]): slope = 0.0
                else: slope = np.polyfit(x, y, 1)[0]
                if np.isfinite(slope): avg_loss = np.mean(y); normalized_slope = slope / (abs(avg_loss) + 1e-6); loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05])
            except (np.linalg.LinAlgError, ValueError): loss_trend_bin = 2
        if self.grad_var_window:
            median_grad_var = np.median(list(self.grad_var_window))
            if np.isfinite(median_grad_var): grad_var_bin = np.digitize(median_grad_var, bins=[1e-5, 1e-3, 0.1, 1.0])
        lr_bin = np.digitize(lr, bins=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]); momentum_bin = np.digitize(momentum, bins=[0.85, 0.9, 0.95, 0.99])
        state = (loss_trend_bin, grad_var_bin, lr_bin, momentum_bin); self.q_table_access_count[state] = self.q_table_access_count.get(state, 0) + 1; return state
    def compute_reward(self, loss_improvement: float, grad_health: float, consistent_improvement: bool) -> float:
        base_reward = 2.0 * loss_improvement if loss_improvement > 0 else 1.0 * loss_improvement; health_penalty = 0.0
        if grad_health < 0.5: health_penalty = -0.3 * (1.0 - grad_health / 0.5)
        consistency_bonus = 0.0
        if consistent_improvement: self.stable_steps += 1; consistency_bonus = min(0.3, 0.05 * math.log1p(self.stable_steps))
        else: self.stable_steps = 0
        total_reward = base_reward + health_penalty + consistency_bonus; final_reward = float(np.clip(total_reward, -1.0, 1.0)); self.performance_window.append(final_reward); return final_reward
    def choose_action(self, state: tuple) -> Dict[str, float]:
        if state not in self.q_table: self.q_table[state] = {p: np.zeros(len(s)) for p, s in self.action_ranges.items()}; self._manage_q_table_size()
        action = {}; self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        for param, space in self.action_ranges.items():
            if random.random() < self.epsilon: chosen_idx = random.randrange(len(space))
            else:
                q_values = self.q_table[state][param]
                if len(q_values) > 0 and np.any(np.isfinite(q_values)):
                     max_q = np.nanmax(q_values); best_indices = np.where(np.abs(q_values - max_q) < 1e-6)[0]
                     if len(best_indices) > 0: chosen_idx = np.random.choice(best_indices)
                     else: chosen_idx = random.randrange(len(space))
                else: chosen_idx = random.randrange(len(space))
            action[param] = float(space[chosen_idx])
        return action
    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: Optional[tuple], should_log: bool = False):
        if next_state is not None and next_state not in self.q_table: self.q_table[next_state] = {p: np.zeros(len(s)) for p, s in self.action_ranges.items()}; self._manage_q_table_size()
        for param, value in action.items():
            space = self.action_ranges[param]
            try:
                action_idx = np.abs(space - value).argmin()
                if not np.isclose(space[action_idx], value): raise ValueError("Action value not found precisely.")
            except ValueError: logger.warning(f"Q-update: Action value {value} not found for {param}. Skipping."); continue
            max_future_q = 0.0
            if next_state is not None and next_state in self.q_table:
                next_q_values = self.q_table[next_state][param]
                if len(next_q_values) > 0 and np.any(np.isfinite(next_q_values)): max_future_q = np.nanmax(next_q_values)
            current_q = self.q_table[state][param][action_idx]; td_target = reward + self.gamma * max_future_q; td_error = td_target - current_q
            self.q_table[state][param][action_idx] += self.alpha * td_error
    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            try:
                if self.q_table_access_count:
                     sorted_states = sorted(self.q_table_access_count.items(), key=lambda item: item[1])
                     num_to_remove = len(self.q_table) - int(self.max_q_table_size * 0.9); num_removed = 0
                     for state, count in sorted_states:
                         if num_removed >= num_to_remove: break
                         if state in self.q_table: del self.q_table[state]
                         if state in self.q_table_access_count: del self.q_table_access_count[state]
                         num_removed += 1
            except (ValueError, KeyError) as e: logger.warning(f"Could not prune Q-table: {e}")

class EnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.003, momentum: float = 0.9, weight_decay: float = 0.005, max_grad_norm: float = 1.0, lr_scale_bounds: tuple = (0.85, 1.15), momentum_scale_bounds: tuple = (0.9, 1.1), q_learning_config: Dict[str, Any] = {}, **kwargs):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict): param_groups = params
        else: params = list(params); param_groups = [{'params': params}]
        for group in param_groups: group.setdefault('lr', lr); group.setdefault('momentum', momentum); group.setdefault('weight_decay', weight_decay); group.setdefault('base_lr', lr); group.setdefault('q_scale', 1.0)
        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        self._init_optimization_state(max_grad_norm=max_grad_norm, lr_scale_bounds=lr_scale_bounds, momentum_scale_bounds=momentum_scale_bounds, q_learning_config=q_learning_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                     self.state[p] = {}
                     self.state[p]['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    def _init_optimization_state(self, **kwargs):
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0); lr_scale_bounds = kwargs.get('lr_scale_bounds', (0.85, 1.15)); momentum_scale_bounds = kwargs.get('momentum_scale_bounds', (0.9, 1.1)); q_learning_config = kwargs.get('q_learning_config', {})
        self.q_controller = QController(learning_rate=q_learning_config.get('learning_rate', 0.02), discount=q_learning_config.get('discount', 0.97), epsilon=q_learning_config.get('epsilon', 0.15), epsilon_decay=q_learning_config.get('epsilon_decay', 0.9995), min_epsilon=q_learning_config.get('min_epsilon', 0.02), lr_scale_bounds=lr_scale_bounds, momentum_scale_bounds=momentum_scale_bounds, max_q_table_size=q_learning_config.get('max_q_table_size', 15000))
        self._step_count = 0; self.gradient_stats = GradientStats(); self.current_loss = None
    def _get_gradient_stats(self) -> Dict[str, Any]:
        grad_norms = []; grad_vars = []; num_finite_params = 0; num_non_finite_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.detach()
                if torch.isfinite(grad).all():
                    grad_float = grad.float(); grad_norms.append(torch.norm(grad_float).item())
                    if grad_float.numel() > 1: grad_vars.append(torch.var(grad_float).item())
                    num_finite_params += 1
                else: num_non_finite_params += 1
        saw_grads = num_finite_params + num_non_finite_params > 0; saw_finite_grads = num_finite_params > 0
        if saw_finite_grads: mean_grad_norm = np.mean(grad_norms); mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
        else: mean_grad_norm = 0.0; mean_grad_var = 0.0
        is_norm_finite = np.isfinite(mean_grad_norm); is_var_finite = np.isfinite(mean_grad_var)
        return {'saw_grads': saw_grads, 'saw_finite_grads': saw_finite_grads, 'mean_grad_norm': mean_grad_norm if is_norm_finite else 0.0, 'mean_grad_var': mean_grad_var if is_var_finite else float('inf'), 'num_non_finite_params': num_non_finite_params, 'is_norm_finite': is_norm_finite, 'is_var_finite': is_var_finite}
    def zero_grad(self, set_to_none: bool=True): super().zero_grad(set_to_none=set_to_none)
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        grad_stats = self._get_gradient_stats()
        if grad_stats['num_non_finite_params'] > 0: logger.warning(f"Step {self._step_count}: Non-finite grads in {grad_stats['num_non_finite_params']} params.")
        current_loss_value = self.current_loss
        if grad_stats['saw_finite_grads'] and current_loss_value is not None and np.isfinite(current_loss_value):
            safe_grad_var = grad_stats['mean_grad_var'] if grad_stats['is_var_finite'] else 0.0
            q_state = self.q_controller.get_state(lr=self.param_groups[0]['lr'], momentum=self.param_groups[0]['momentum'], grad_var=safe_grad_var, loss=current_loss_value)
            if self.q_controller.prev_loss is not None and self.q_controller.prev_state is not None and self.q_controller.prev_action is not None:
                if np.isfinite(self.q_controller.prev_loss) and abs(self.q_controller.prev_loss) > 1e-9:
                    loss_improvement = (self.q_controller.prev_loss - current_loss_value) / (abs(self.q_controller.prev_loss) + 1e-9)
                    grad_health = 1.0 / (1.0 + max(0, safe_grad_var))
                    consistent_improvement = all([r > -0.01 for r in list(self.q_controller.performance_window)[-10:]])
                    reward = self.q_controller.compute_reward(loss_improvement=loss_improvement, grad_health=grad_health, consistent_improvement=consistent_improvement)
                    if np.isfinite(reward): self.q_controller.update(state=self.q_controller.prev_state, action=self.q_controller.prev_action, reward=reward, next_state=q_state, should_log=(self._step_count % 10 == 0))
                    else: logger.warning(f"Step {self._step_count}: Skipping Q-update due to non-finite reward.")
                else: logger.warning(f"Step {self._step_count}: Skipping Q-update due to non-finite/zero prev loss.")
            q_action = self.q_controller.choose_action(q_state)
            for group in self.param_groups:
                group['q_scale'] *= float(np.clip(q_action['lr_scale'], self.q_controller.lr_scale_bounds[0], self.q_controller.lr_scale_bounds[1]))
                min_lr = 1e-7; max_lr = 0.01
                group['lr'] = float(np.clip(group['base_lr'] * group['q_scale'], min_lr, max_lr))
                group['momentum'] = float(np.clip(group['momentum'] * q_action['momentum_scale'], self.q_controller.momentum_scale_bounds[0], self.q_controller.momentum_scale_bounds[1]))
            self.q_controller.prev_state = q_state; self.q_controller.prev_action = q_action; self.q_controller.prev_loss = current_loss_value
        num_params_updated = 0
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']; weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                state = self.state[p]
                updated = self._apply_update(p=p, grad_in=p.grad, momentum=momentum, lr=lr, weight_decay=weight_decay, state=state, current_loss=self.q_controller.prev_loss if self.q_controller.prev_loss is not None else 0.0)
                if updated: num_params_updated += 1
        if num_params_updated > 0:
             self._step_count += 1
             stats = self.gradient_stats.record_step(self._step_count)
             # Log less frequently
             # if self._step_count % 50 == 0: logger.info(f"Step {self._step_count} grad stats: Clipped {stats['gradients_clipped']}/{stats['total_gradients']}...")
        elif grad_stats['saw_grads']: logger.warning(f"Step {self._step_count}: Grads present but no params updated.")
        self.current_loss = None
        return loss
    def _apply_update(self, p: torch.Tensor, grad_in: torch.Tensor, momentum: float, lr: float, weight_decay: float, state: dict, current_loss: float) -> bool:
        if not torch.isfinite(grad_in).all(): grad = torch.zeros_like(grad_in)
        else: grad = grad_in.clone()
        if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        buf = state['momentum_buffer']
        if weight_decay != 0 and torch.isfinite(grad).all(): grad = grad.add(p, alpha=weight_decay)
        grad_norm_before_clip = torch.norm(grad).item(); was_clipped = False; clip_ratio = 1.0
        if grad_norm_before_clip > self.max_grad_norm: clip_ratio = self.max_grad_norm / (grad_norm_before_clip + 1e-6); grad.mul_(clip_ratio); was_clipped = True
        self.gradient_stats.record_gradient(original_norm=grad_norm_before_clip, clipped=was_clipped, clip_ratio=clip_ratio if was_clipped else 1.0)
        buf.mul_(momentum).add_(grad); update = buf; p.add_(update, alpha=-lr); return True
    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        try:
            state_dict['q_table'] = self.q_controller.q_table; state_dict['q_controller_epsilon'] = float(self.q_controller.epsilon); state_dict['q_controller_prev_loss'] = self.q_controller.prev_loss
            state_dict['q_controller_prev_state'] = self.q_controller.prev_state; state_dict['q_controller_prev_action'] = self.q_controller.prev_action; state_dict['q_controller_access_count'] = self.q_controller.q_table_access_count
            state_dict['_step_count'] = self._step_count
        except Exception as e: logger.error(f"Error creating EnhancedSGD state dict: {e}"); state_dict['q_table'] = {}; state_dict['q_controller_epsilon'] = 0.15
        return state_dict
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        q_table = state_dict.pop('q_table', None); epsilon = state_dict.pop('q_controller_epsilon', None); prev_loss = state_dict.pop('q_controller_prev_loss', None)
        prev_state = state_dict.pop('q_controller_prev_state', None); prev_action = state_dict.pop('q_controller_prev_action', None); access_count = state_dict.pop('q_controller_access_count', None)
        step_count = state_dict.pop('_step_count', None)
        try:
            super().load_state_dict(state_dict)
            if q_table is not None: self.q_controller.q_table = q_table
            if epsilon is not None: self.q_controller.epsilon = float(epsilon)
            if prev_loss is not None: self.q_controller.prev_loss = float(prev_loss) if prev_loss is not None else None
            if prev_state is not None: self.q_controller.prev_state = prev_state
            if prev_action is not None: self.q_controller.prev_action = prev_action
            if access_count is not None: self.q_controller.q_table_access_count = access_count
            if step_count is not None: self._step_count = int(step_count)
            for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
                 if 'q_scale' in saved_group: group['q_scale'] = saved_group['q_scale']
                 elif 'base_lr' in group and group['base_lr'] > 1e-9: group['q_scale'] = group['lr'] / group['base_lr']
                 else: group['q_scale'] = 1.0
        except Exception as e:
            logger.error(f"Error loading EnhancedSGD state dict: {e}", exc_info=True)
            self.q_controller.q_table = {}; self.q_controller.epsilon = 0.15; self.q_controller.prev_loss = None; self.q_controller.prev_state = None; self.q_controller.prev_action = None; self.q_controller.q_table_access_count = {}; self._step_count = 0
            for group in self.param_groups: group['q_scale'] = 1.0; group['lr'] = group.get('base_lr', group['lr'])

# --- Model Components ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__(); original_num_heads = num_heads
        if hidden_size == 0: raise ValueError("hidden_size cannot be zero")
        if num_heads <= 0: num_heads = 1
        while hidden_size % num_heads != 0 and num_heads > 1: num_heads -= 1
        if hidden_size % num_heads != 0: num_heads = 1
        if num_heads != original_num_heads: logger.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")
        self.hidden_size = hidden_size; self.num_heads = num_heads; self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size); self.k_proj = nn.Linear(hidden_size, hidden_size); self.v_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.q_proj.weight, std=0.02); nn.init.zeros_(self.q_proj.bias); nn.init.normal_(self.k_proj.weight, std=0.02); nn.init.zeros_(self.k_proj.bias); nn.init.normal_(self.v_proj.weight, std=0.02); nn.init.zeros_(self.v_proj.bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size); nn.init.normal_(self.out_proj.weight, std=0.02); nn.init.zeros_(self.out_proj.bias)
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6); self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6); self.dropout = nn.Dropout(dropout)
    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_queries, _ = queries.size(); seq_len_kv = keys_values.size(1); device = queries.device
        queries_norm = self.norm_q(queries); keys_values_norm = self.norm_kv(keys_values)
        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2: attn_mask_bool = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3: attn_mask_bool = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4: attn_mask_bool = attention_mask
            else: logger.warning(f"Unsupported attention mask shape: {attention_mask.shape}. Ignoring mask.")
            if attn_mask_bool is not None: attn_mask_bool = attn_mask_bool.bool()
        use_flash = hasattr(F, 'scaled_dot_product_attention')
        if use_flash:
            try: output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_bool, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
            except Exception: use_flash = False
        if not use_flash:
            scale = 1.0 / math.sqrt(self.head_dim) if self.head_dim > 0 else 1.0
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask_bool is not None: scores = scores.masked_fill(attn_mask_bool.to(scores.device), float('-inf'))
            attn_probs = torch.softmax(scores, dim=-1); attn_probs = self.dropout(attn_probs); output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size); output = self.out_proj(output); return output

class LocalEncoder(nn.Module):
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1, n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__(); self.hidden_size=hidden_size; self.byte_embeddings=nn.Embedding(256, hidden_size); nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0/math.sqrt(hidden_size))
        self.n_gram_sizes=sorted(list(set(n_gram_sizes))) if n_gram_sizes else []; self.n_gram_embeddings=None; self.n_gram_vocab_size=n_gram_vocab_size
        if self.n_gram_sizes:
            self.n_gram_embeddings=nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes})
            for emb in self.n_gram_embeddings.values(): nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=num_heads,dim_feedforward=hidden_size*4,dropout=dropout,batch_first=True, activation=F.gelu)
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=num_layers); self.patch_pooling_attention=CrossAttentionBlock(hidden_size, num_heads, dropout); self.patch_query=nn.Parameter(torch.randn(1, 1, hidden_size))
        self.norm=nn.LayerNorm(hidden_size,eps=1e-6); self.dropout=nn.Dropout(dropout)
    def _get_n_gram_hashes(self, byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        batch_size, seq_len = byte_sequence.shape; device=byte_sequence.device
        if seq_len < n: return torch.empty(batch_size, 0, dtype=torch.long, device=device)
        windows = byte_sequence.unfold(1, n, 1); hashes = windows.long().sum(dim=-1); return hashes % self.n_gram_vocab_size
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor: return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    def forward(self, patches: List[torch.Tensor]) -> torch.Tensor: # Expects List[1D Tensor]
        if not patches: device = next(self.parameters()).device; return torch.empty((1, 0, self.hidden_size), device=device)
        device = patches[0].device; patch_representations = []
        for patch_bytes in patches:
            patch_len = patch_bytes.size(0);
            if patch_len == 0: continue
            patch_bytes_batched = patch_bytes.unsqueeze(0); x = self.byte_embeddings(patch_bytes_batched)
            if self.n_gram_embeddings:
                n_gram_features = torch.zeros_like(x)
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_batched, n);
                        if n_gram_hashes.numel() > 0:
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes)
                            num_windows = ngram_embeds.size(1)
                            n_gram_features[:, :num_windows, :] += ngram_embeds
                x = x + n_gram_features
            x = self.dropout(x); causal_mask = self.create_causal_mask(patch_len, device);
            processed_bytes = self.transformer(x, mask=causal_mask)
            batch_query = self.patch_query; patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes); patch_representations.append(patch_repr)
        if not patch_representations: return torch.empty((1, 0, self.hidden_size), device=device)
        patches_combined = torch.cat(patch_representations, dim=1); return self.norm(patches_combined)

class RealToComplexProjection(nn.Module):
    def __init__(self, input_dim: int, complex_dim: int):
        super().__init__(); self.proj_real = nn.Linear(input_dim, complex_dim); self.proj_imag = nn.Linear(input_dim, complex_dim)
        nn.init.xavier_uniform_(self.proj_real.weight); nn.init.zeros_(self.proj_real.bias); nn.init.xavier_uniform_(self.proj_imag.weight); nn.init.zeros_(self.proj_imag.bias)
    def forward(self, x_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: return self.proj_real(x_real), self.proj_imag(x_real)

class ComplexToRealProjection(nn.Module):
    def __init__(self, complex_dim: int, output_dim: int, method: str = "concat"):
        super().__init__(); self.method = method
        if method == "concat": self.proj = nn.Linear(complex_dim * 2, output_dim)
        elif method == "magnitude": self.proj = nn.Linear(complex_dim, output_dim)
        else: logger.warning(f"Unknown ComplexToRealProjection method '{method}'. Defaulting to 'concat'."); self.method = "concat"; self.proj = nn.Linear(complex_dim * 2, output_dim)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        real, imag = x
        if self.method == "concat": combined = torch.cat([real, imag], dim=-1); return self.proj(combined)
        elif self.method == "magnitude": magnitude = torch.sqrt(real**2 + imag**2 + 1e-8); return self.proj(magnitude)
        else: combined = torch.cat([real, imag], dim=-1); return self.proj(combined)

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, coupled=True):
        super().__init__(); self.real_norm = nn.LayerNorm(dim, eps=eps); self.imag_norm = nn.LayerNorm(dim, eps=eps); self.coupled = coupled
        if coupled: self.coupling = nn.Parameter(torch.tensor(0.1)); self.cross_gain_ri = nn.Parameter(torch.zeros(dim)); self.cross_gain_ir = nn.Parameter(torch.zeros(dim))
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x
        if self.coupled:
            real_normed = self.real_norm(real); imag_normed = self.imag_norm(imag); coupling_strength = torch.sigmoid(self.coupling)
            real_out = real_normed + coupling_strength * self.cross_gain_ri * imag_normed; imag_out = imag_normed + coupling_strength * self.cross_gain_ir * real_normed
            return real_out, imag_out
        else: return self.real_norm(real), self.imag_norm(imag)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024, phase_shift=True, learnable=True):
        super().__init__(); self.dim = dim; self.learnable = learnable
        position = torch.arange(max_len).unsqueeze(1).float(); div_term_base = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe_real = torch.zeros(max_len, dim); pe_real[:, 0::2] = torch.sin(position * div_term_base); pe_real[:, 1::2] = torch.cos(position * div_term_base)
        pe_imag = torch.zeros(max_len, dim)
        if phase_shift: pe_imag[:, 0::2] = torch.cos(position * div_term_base); pe_imag[:, 1::2] = -torch.sin(position * div_term_base)
        else: pe_imag[:, 0::2] = torch.sin(position * div_term_base + math.pi / 4); pe_imag[:, 1::2] = torch.cos(position * div_term_base + math.pi / 4)
        self.register_buffer('pe_real_base', pe_real); self.register_buffer('pe_imag_base', pe_imag); self.register_buffer('div_term_base', div_term_base)
        if learnable:
             self.real_scale = nn.Parameter(torch.ones(1, 1, dim)); self.imag_scale = nn.Parameter(torch.ones(1, 1, dim)); self.real_shift = nn.Parameter(torch.zeros(1, 1, dim)); self.imag_shift = nn.Parameter(torch.zeros(1, 1, dim))
             self.frequency_scale_factors = nn.Parameter(torch.ones(dim // 2));
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x; seq_len = real.size(1); device = real.device
        if self.learnable:
            scaled_div_term = self.div_term_base.to(device) * torch.clamp(self.frequency_scale_factors, min=1e-2)
            position = torch.arange(seq_len, device=device).unsqueeze(1).float(); pe_real_learn = torch.zeros(seq_len, self.dim, device=device); pe_imag_learn = torch.zeros(seq_len, self.dim, device=device)
            angles = position * scaled_div_term; pe_real_learn[:, 0::2] = torch.sin(angles); pe_real_learn[:, 1::2] = torch.cos(angles)
            pe_imag_learn[:, 0::2] = torch.cos(angles); pe_imag_learn[:, 1::2] = -torch.sin(angles)
            pe_real = pe_real_learn * self.real_scale + self.real_shift; pe_imag = pe_imag_learn * self.imag_scale + self.imag_shift
            return real + pe_real, imag + pe_imag
        else:
             max_len_buffer = self.pe_real_base.size(0); slice_len = min(seq_len, max_len_buffer)
             return real + self.pe_real_base[:slice_len, :].to(device), imag + self.pe_imag_base[:slice_len, :].to(device)

class EntangledInterferenceLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, interference_type="quantum", use_entanglement=True, noise_scale=0.1, use_rotary=True, adaptive_attention=True):
        super().__init__();
        if dim == 0: raise ValueError("Dimension cannot be zero.")
        if heads <= 0: heads = 1
        while dim % heads != 0 and heads > 1: heads -= 1
        if dim % heads != 0: heads = 1
        self.dim = dim; self.heads = heads; self.head_dim = dim // heads; self.dropout = dropout; self.interference_type = interference_type; self.use_entanglement = use_entanglement; self.noise_scale = noise_scale; self.use_rotary = use_rotary; self.adaptive_attention = adaptive_attention
        self.phase_shifts = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        if use_entanglement: self.entanglement_matrix = nn.Parameter(torch.eye(heads) + torch.randn(heads, heads) * 0.01)
        else: self.register_buffer('entanglement_matrix', torch.eye(heads), persistent=False)
        self.q_real = nn.Linear(dim, dim); self.k_real = nn.Linear(dim, dim); self.v_real = nn.Linear(dim, dim); self.q_imag = nn.Linear(dim, dim); self.k_imag = nn.Linear(dim, dim); self.v_imag = nn.Linear(dim, dim); self.out_real = nn.Linear(dim, dim); self.out_imag = nn.Linear(dim, dim)
        if use_rotary:
            self.rotary_dim = min(self.head_dim, 32) if self.head_dim > 0 else 0
            if self.rotary_dim > 0:
                 base_freqs = 10000.0**(-torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
                 if adaptive_attention: self.rotary_freqs = nn.Parameter(base_freqs)
                 else: self.register_buffer('rotary_freqs', base_freqs, persistent=False)
            else: self.use_rotary = False
        self.interference_strength = nn.Parameter(torch.ones(1))
        if adaptive_attention: self.attention_temperature = nn.Parameter(torch.ones(1))
        else: self.register_buffer('attention_temperature', torch.ones(1), persistent=False)
        self.attn_dropout = nn.Dropout(dropout); self.resid_dropout = nn.Dropout(dropout); self.saved_attn_weights = None
    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rotary or self.rotary_dim <= 0: return q, k
        device = q.device; dim_rotary = self.rotary_dim; q_rot = q[..., :dim_rotary]; q_pass = q[..., dim_rotary:]; k_rot = k[..., :dim_rotary]; k_pass = k[..., dim_rotary:]
        position = torch.arange(seq_len, device=device).float(); freqs = self.rotary_freqs.to(device); emb = torch.outer(position, freqs)
        cos_emb = torch.cos(emb).unsqueeze(0).unsqueeze(1); sin_emb = torch.sin(emb).unsqueeze(0).unsqueeze(1)
        q_rot = q_rot.reshape(*q_rot.shape[:-1], -1, 2); k_rot = k_rot.reshape(*k_rot.shape[:-1], -1, 2)
        cos = cos_emb.unsqueeze(-1); sin = sin_emb.unsqueeze(-1)
        q_rot_out = torch.zeros_like(q_rot); k_rot_out = torch.zeros_like(k_rot)
        q_rot_out[..., 0] = q_rot[..., 0] * cos[..., 0] - q_rot[..., 1] * sin[..., 0]; q_rot_out[..., 1] = q_rot[..., 1] * cos[..., 0] + q_rot[..., 0] * sin[..., 0]
        k_rot_out[..., 0] = k_rot[..., 0] * cos[..., 0] - k_rot[..., 1] * sin[..., 0]; k_rot_out[..., 1] = k_rot[..., 1] * cos[..., 0] + k_rot[..., 0] * sin[..., 0]
        q_rot_out = q_rot_out.flatten(start_dim=-2); k_rot_out = k_rot_out.flatten(start_dim=-2)
        q_out = torch.cat([q_rot_out, q_pass], dim=-1); k_out = torch.cat([k_rot_out, k_pass], dim=-1); return q_out, k_out
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = x; batch_size, seq_len, _ = real.shape; device = real.device
        q_r = self.q_real(real).view(batch_size, seq_len, self.heads, self.head_dim); k_r = self.k_real(real).view(batch_size, seq_len, self.heads, self.head_dim); v_r = self.v_real(real).view(batch_size, seq_len, self.heads, self.head_dim)
        q_i = self.q_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim); k_i = self.k_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim); v_i = self.v_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim)
        if self.training and self.noise_scale > 0: q_r = add_quantum_noise(q_r, noise_scale=self.noise_scale); q_i = add_quantum_noise(q_i, noise_scale=self.noise_scale); k_r = add_quantum_noise(k_r, noise_scale=self.noise_scale); k_i = add_quantum_noise(k_i, noise_scale=self.noise_scale)
        q_r, k_r = self._apply_rotary_pos_emb(q_r.transpose(1, 2), k_r.transpose(1, 2), seq_len); q_i, k_i = self._apply_rotary_pos_emb(q_i.transpose(1, 2), k_i.transpose(1, 2), seq_len)
        v_r = v_r.transpose(1, 2); v_i = v_i.transpose(1, 2)
        entanglement_matrix_eff = self.entanglement_matrix.to(device)
        if self.use_entanglement: q_r = torch.einsum("bhsd,hx->bxsd", q_r, entanglement_matrix_eff); q_i = torch.einsum("bhsd,hx->bxsd", q_i, entanglement_matrix_eff); k_r = torch.einsum("bhsd,hx->bxsd", k_r, entanglement_matrix_eff); k_i = torch.einsum("bhsd,hx->bxsd", k_i, entanglement_matrix_eff)
        phase_cos = torch.cos(self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device); phase_sin = torch.sin(self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device)
        q_r_shifted = q_r * phase_cos - q_i * phase_sin; q_i_shifted = q_r * phase_sin + q_i * phase_cos; k_r_shifted = k_r * phase_cos - k_i * phase_sin; k_i_shifted = k_r * phase_sin + k_i * phase_cos
        q_r, q_i = q_r_shifted, q_i_shifted; k_r, k_i = k_r_shifted, k_i_shifted
        scale = 1.0 / math.sqrt(self.head_dim) if self.head_dim > 0 else 1.0
        if self.interference_type == "quantum":
            attn_r = torch.matmul(q_r, k_r.transpose(-2, -1)) + torch.matmul(q_i, k_i.transpose(-2, -1)); attn_i = torch.matmul(q_i, k_r.transpose(-2, -1)) - torch.matmul(q_r, k_i.transpose(-2, -1))
            attn_r *= scale; attn_i *= scale; attn_mag = torch.sqrt(attn_r**2 + attn_i**2 + 1e-6)
        else: attn_mag = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1); final_mask = causal_mask
        if attention_mask is not None: # Mask is True where padded
             if attention_mask.dim() == 2: padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool(); final_mask = final_mask | padding_mask
             elif attention_mask.dim() == 4: final_mask = final_mask | attention_mask.bool()
             else: logger.warning(f"Unsupported mask shape {attention_mask.shape}. Ignoring padding mask.")
        attn_mag = attn_mag.masked_fill(final_mask.to(attn_mag.device), float('-inf'))
        temp = torch.clamp(self.attention_temperature.to(device), min=1e-2); strength = torch.sigmoid(self.interference_strength.to(device))
        attn_weights = F.softmax((attn_mag * strength) / temp, dim=-1); attn_weights = self.attn_dropout(attn_weights); # self.saved_attn_weights = attn_weights.detach().cpu()
        out_r = torch.matmul(attn_weights, v_r); out_i = torch.matmul(attn_weights, v_i)
        out_r = out_r.transpose(1, 2).reshape(batch_size, seq_len, self.dim); out_i = out_i.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out_r = self.out_real(out_r); out_i = self.out_imag(out_i); out_r = self.resid_dropout(out_r); out_i = self.resid_dropout(out_i); return (out_r, out_i)

class LocalDecoder(nn.Module):
    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__(); self.hidden_size = hidden_size; self.byte_embeddings = nn.Embedding(256, hidden_size); nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        self.memory_projection = nn.Linear(global_hidden_size, hidden_size); nn.init.normal_(self.memory_projection.weight, std=0.02); nn.init.zeros_(self.memory_projection.bias)
        if hidden_size == 0: raise ValueError("LocalDecoder hidden_size cannot be zero")
        if num_heads <= 0: num_heads = 1
        while hidden_size % num_heads != 0 and num_heads > 1: num_heads -= 1
        if hidden_size % num_heads != 0: num_heads = 1
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, activation=F.gelu)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers); self.byte_pred = nn.Linear(hidden_size, 256); nn.init.normal_(self.byte_pred.weight, std=0.02); nn.init.zeros_(self.byte_pred.bias)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6); self.dropout = nn.Dropout(dropout)
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor: return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, tgt_len = tgt_byte_seq.size(); device = tgt_byte_seq.device
        if tgt_len == 0: return torch.zeros((batch_size, 0, 256), device=device)
        tgt_embed = self.byte_embeddings(tgt_byte_seq); tgt_embed = self.dropout(tgt_embed); projected_memory = self.memory_projection(memory)
        tgt_mask = self.create_causal_mask(tgt_len, device)
        output = self.transformer_decoder(tgt=tgt_embed, memory=projected_memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.norm(output); byte_logits = self.byte_pred(output); return byte_logits

# --- BSFIN Model Definition ---
class BSFINModel(nn.Module):
    def __init__(
        self, local_hidden_size: int = 256, complex_dim: int = 512, num_complex_layers: int = 8, num_complex_heads: int = 8,
        decoder_memory_dim: int = 1024, dropout: float = 0.15, context_window: int = 256, n_gram_sizes: List[int] = [3, 4],
        n_gram_vocab_size: int = 30000, sfin_noise_scale: float = 0.05, sfin_use_entanglement: bool = True,
        sfin_use_rotary: bool = True, projection_method: str = "concat"
    ):
        super().__init__(); self.local_hidden_size = local_hidden_size; self.complex_dim = complex_dim; self.decoder_memory_dim = decoder_memory_dim; self.context_window = context_window; self.vocab_size = 256
        if complex_dim % num_complex_heads != 0: raise ValueError(f"complex_dim must be divisible by heads")
        self.patcher = BabylonIndex(scales=n_gram_sizes)
        self.local_encoder = LocalEncoder(local_hidden_size, num_layers=1, num_heads=8, dropout=dropout, n_gram_sizes=n_gram_sizes, n_gram_vocab_size=n_gram_vocab_size)
        self.real_to_complex = RealToComplexProjection(local_hidden_size, complex_dim)
        self.complex_pos_encoding = PositionalEncoding(complex_dim, max_len=max(2048, context_window*4), learnable=True)
        self.complex_norm_in = ComplexLayerNorm(complex_dim, coupled=True)
        self.complex_interference_layers = nn.ModuleList([EntangledInterferenceLayer(complex_dim, num_complex_heads, dropout, noise_scale=sfin_noise_scale, use_entanglement=sfin_use_entanglement, use_rotary=sfin_use_rotary, adaptive_attention=True) for _ in range(num_complex_layers)])
        self.complex_norms_mid = nn.ModuleList([ComplexLayerNorm(complex_dim, coupled=True) for _ in range(num_complex_layers)])
        self.complex_dropout = nn.Dropout(dropout)
        self.complex_to_real = ComplexToRealProjection(complex_dim, decoder_memory_dim, method=projection_method)
        self.local_decoder = LocalDecoder(local_hidden_size, decoder_memory_dim, num_layers=4, num_heads=8, dropout=dropout)
        logger.info(f"BSFIN Initialized: Local={local_hidden_size}, Complex={complex_dim}x{num_complex_layers}, DecoderMem={decoder_memory_dim}")
    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = byte_seq.size(0); device = byte_seq.device; target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
        batch_patch_repr_list = []; num_patches_per_item = []; valid_batch_indices = []
        for i in range(batch_size):
            seq = byte_seq[i]; patches = self.patcher.create_patches(seq)
            if patches:
                real_patch_repr_single = self.local_encoder(patches)
                if real_patch_repr_single.numel() > 0 and real_patch_repr_single.size(1) > 0:
                    batch_patch_repr_list.append(real_patch_repr_single); num_patches_per_item.append(real_patch_repr_single.size(1)); valid_batch_indices.append(i)
        if not batch_patch_repr_list: logger.warning("No valid patches for batch."); return torch.zeros((batch_size, target_len, self.vocab_size), device=device)
        max_num_patches = max(num_patches_per_item) if num_patches_per_item else 0
        if max_num_patches == 0: logger.warning("Max patches is zero."); return torch.zeros((batch_size, target_len, self.vocab_size), device=device)
        padded_repr_list = []
        for repr_tensor in batch_patch_repr_list:
             num_patches = repr_tensor.size(1); padding_size = max_num_patches - num_patches
             if padding_size > 0: padding = torch.zeros((1, padding_size, self.local_hidden_size), device=device); padded_repr = torch.cat([repr_tensor, padding], dim=1)
             else: padded_repr = repr_tensor
             padded_repr_list.append(padded_repr)
        real_patch_repr_batched = torch.cat(padded_repr_list, dim=0)
        num_valid_patches_tensor = torch.tensor(num_patches_per_item, device=device)
        memory_padding_mask = torch.arange(max_num_patches, device=device)[None, :] >= num_valid_patches_tensor[:, None]
        complex_patch_repr = self.real_to_complex(real_patch_repr_batched); complex_patch_repr = self.complex_pos_encoding(complex_patch_repr); complex_patch_repr = self.complex_norm_in(complex_patch_repr)
        real, imag = complex_patch_repr
        for i, layer in enumerate(self.complex_interference_layers):
            real_res, imag_res = real, imag; normed_real, normed_imag = self.complex_norms_mid[i]((real, imag))
            out_real, out_imag = layer((normed_real, normed_imag), attention_mask=memory_padding_mask)
            real = real_res + self.complex_dropout(out_real); imag = imag_res + self.complex_dropout(out_imag)
        processed_complex_repr = (real, imag); processed_real_repr = self.complex_to_real(processed_complex_repr)
        if target_byte_seq is None: # Handle generation case used by LiveTrainer
             if byte_seq.size(1) > 0:
                  last_input_bytes = byte_seq[:, -1:]
                  valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
                  if valid_indices_tensor.numel() == 0: return torch.zeros((batch_size, 1, self.vocab_size), device=device)
                  valid_last_input = last_input_bytes.index_select(0, valid_indices_tensor)
                  # Ensure memory corresponds to valid items
                  valid_memory = processed_real_repr # Already filtered
                  valid_mask = memory_padding_mask # Already filtered
                  byte_logits_valid = self.local_decoder(tgt_byte_seq=valid_last_input, memory=valid_memory, memory_key_padding_mask=valid_mask)
                  final_byte_logits = torch.zeros((batch_size, 1, self.vocab_size), device=device, dtype=torch.float32)
                  final_byte_logits.index_put_((valid_indices_tensor,), byte_logits_valid.to(final_byte_logits.dtype))
                  return final_byte_logits
             else: return torch.zeros((batch_size, 0, self.vocab_size), device=device)
        else: # Training case (standard)
             valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
             if valid_indices_tensor.numel() == 0: return torch.zeros((batch_size, target_len, self.vocab_size), device=device)
             valid_target_byte_seq = target_byte_seq.index_select(0, valid_indices_tensor)
             # Ensure memory corresponds to valid items
             valid_memory = processed_real_repr # Already filtered
             valid_mask = memory_padding_mask # Already filtered
             byte_logits_valid = self.local_decoder(tgt_byte_seq=valid_target_byte_seq, memory=valid_memory, memory_key_padding_mask=valid_mask)
             final_byte_logits = torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)
             final_byte_logits.index_put_((valid_indices_tensor,), byte_logits_valid.to(final_byte_logits.dtype))
             return final_byte_logits
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        """Computes cross-entropy loss, adaptable for sequence or next-byte."""
        batch_size, seq_len, vocab_size = logits.size()
        if seq_len == 1: # Next-byte prediction case
             logits_flat = logits.squeeze(1); targets_flat = targets.squeeze()
             if targets_flat.dim() == 0: targets_flat = targets_flat.unsqueeze(0)
             if logits_flat.dim() == 1: logits_flat = logits_flat.unsqueeze(0)
        else: # Sequence prediction case
             logits_shifted = logits[:, :-1, :].contiguous(); targets_shifted = targets[:, 1:].contiguous()
             logits_flat = logits_shifted.view(-1, vocab_size); targets_flat = targets_shifted.view(-1)
        current_vocab_size = logits_flat.size(-1)
        if targets_flat.numel() > 0:
            # Ensure targets are within the valid range [0, vocab_size-1]
            targets_flat = targets_flat.long() # Ensure long type
            if torch.any(targets_flat >= current_vocab_size) or torch.any(targets_flat < 0):
                invalid_indices = torch.where((targets_flat < 0) | (targets_flat >= current_vocab_size))[0]
                logger.error(f"Target idx OOB ({current_vocab_size}): {targets_flat[invalid_indices[:10]]}")
                targets_flat = torch.clamp(targets_flat, 0, current_vocab_size - 1)
        if targets_flat.numel() == 0 or logits_flat.numel() == 0:
             return torch.tensor(0.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=smoothing, reduction='mean')
        # Masking logic removed for simplicity
        return loss

# =====================================================================
# Hash-Indexed Continual Learning Components (v1.0)
# =====================================================================

@dataclass
class LearningEventMetadata:
    """Stores metadata for a learning event associated with a hash."""
    last_accessed_time: float = field(default_factory=time.time)
    frequency_count: int = 1
    importance_score: float = 1.0 # Higher means more important

class HashIndexedMemory:
    """Stores metadata about learning events keyed by input hashes."""
    def __init__(self, capacity: int = 100000, decay_rate: float = 0.01, importance_threshold: float = 0.1):
        self.memory: Dict[str, LearningEventMetadata] = {}
        self.capacity = capacity; self.decay_rate = decay_rate; self.importance_threshold = importance_threshold
        self.access_order = deque()
        logger.info(f"Initialized HashIndexedMemory. Capacity: {capacity}, Decay Rate: {decay_rate}")
    def update_event(self, input_hash: str, current_time: float, initial_importance: float = 1.0) -> LearningEventMetadata:
        if input_hash in self.memory:
            metadata = self.memory[input_hash]; metadata.last_accessed_time = current_time; metadata.frequency_count += 1
            self._update_access_order(input_hash)
        else:
            metadata = LearningEventMetadata(last_accessed_time=current_time, importance_score=initial_importance)
            self.memory[input_hash] = metadata; self.access_order.append(input_hash); self._prune()
        return metadata
    def get_metadata(self, input_hash: str) -> Optional[LearningEventMetadata]: return self.memory.get(input_hash)
    def _update_access_order(self, input_hash: str):
        try: self.access_order.remove(input_hash)
        except ValueError: pass
        self.access_order.append(input_hash)
    def _prune(self):
        removed_count = 0
        while len(self.access_order) > self.capacity:
            lru_hash = self.access_order.popleft()
            if lru_hash in self.memory: del self.memory[lru_hash]; removed_count += 1
        # if removed_count > 0: logger.debug(f"Pruned {removed_count} LRU items.")
    def decay_importance(self, current_time: float, time_threshold_seconds: float = 3600 * 24):
        hashes_to_remove = []
        for hash_key, metadata in self.memory.items():
            time_since_access = current_time - metadata.last_accessed_time
            if time_since_access > time_threshold_seconds:
                decay_factor = 1.0 - min(1.0, self.decay_rate * (time_since_access / time_threshold_seconds))
                metadata.importance_score *= decay_factor
                if metadata.importance_score < self.importance_threshold: hashes_to_remove.append(hash_key)
        for hash_key in hashes_to_remove:
            if hash_key in self.memory:
                del self.memory[hash_key]
                try: self.access_order.remove(hash_key)
                except ValueError: pass
        if hashes_to_remove: logger.info(f"Pruned {len(hashes_to_remove)} entries via importance decay.")
    def __len__(self) -> int: return len(self.memory)

class LiveTrainerHashIndexed:
    """Trainer using hash-indexed memory to modulate learning."""
    def __init__(self, model: BSFINModel, optimizer: torch.optim.Optimizer, hash_memory: HashIndexedMemory,
                 babylon_indexer: BabylonIndex, device: torch.device, context_size: int, batch_size: int = 8,
                 grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10, save_interval: int = 100,
                 decay_interval: int = 500, checkpoint_dir: str = "live_hash_checkpoints", max_grad_norm: float = 1.0,
                 importance_influence_factor: float = 0.5):
        self.model = model; self.optimizer = optimizer; self.memory = hash_memory; self.indexer = babylon_indexer; self.device = device
        self.context_size = context_size; self.batch_size = batch_size; self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch, "amp"); self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.log_interval = log_interval; self.save_interval = save_interval; self.decay_interval = decay_interval; self.checkpoint_dir = checkpoint_dir
        self.max_grad_norm = max_grad_norm; self.importance_influence_factor = np.clip(importance_influence_factor, 0.0, 1.0)
        self.global_step = 0; self.processed_data_count = 0; self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(getattr(self.optimizer, 'gradient_stats', None), GradientStats) # Safer check
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"LiveTrainerHashIndexed initialized. Batch Size: {batch_size}, Importance Factor: {self.importance_influence_factor}, AMP: {self.use_amp}")

    def process_data_stream(self, data_iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]], max_steps: Optional[int] = None):
        self.model.train(); self.optimizer.zero_grad(); steps_taken = 0; accumulation_counter = 0
        pbar = tqdm(total=max_steps, desc="Live Steps (HashIdx)") if max_steps else None; current_batch_list = []
        while True:
            try:
                context, target = next(data_iterator)
                if context.shape[0] != self.context_size: continue
                target = target.squeeze();
                if target.dim() != 0: continue
            except StopIteration: logger.info("Data stream finished."); break
            except Exception as e: logger.error(f"Error getting data: {e}"); break
            if max_steps is not None and steps_taken >= max_steps: logger.info(f"Reached max_steps ({max_steps}). Stopping."); break
            self.processed_data_count += 1; current_batch_list.append((context, target))
            if len(current_batch_list) >= self.batch_size:
                accumulation_counter += 1
                contexts_list = [item[0] for item in current_batch_list]; targets_list = [item[1] for item in current_batch_list]
                try: batch_contexts = torch.stack(contexts_list).to(self.device); batch_targets = torch.stack(targets_list).to(self.device)
                except Exception as e: logger.error(f"Error collating batch: {e}. Skipping."); current_batch_list = []; continue
                # --- Get Hashes and Importance ---
                batch_hashes = [self.indexer.get_sequence_hash(ctx) for ctx in contexts_list]
                current_time = time.time(); batch_importance = []
                for h in batch_hashes: meta = self.memory.update_event(h, current_time); batch_importance.append(meta.importance_score)
                avg_importance = np.mean(batch_importance) if batch_importance else 1.0
                gradient_scale_factor = (1.0 - self.importance_influence_factor) + (self.importance_influence_factor * avg_importance)
                gradient_scale_factor = np.clip(gradient_scale_factor, 0.1, 1.0)
                current_batch_list = []
                # --- Training Step ---
                try:
                    with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                        # Pass None for target_byte_seq to get next-byte prediction logits
                        logits = self.model(byte_seq=batch_contexts, target_byte_seq=None) # [B, 1, V]
                        # Loss expects [B, 1, V] and [B, 1]
                        loss = BSFINModel.compute_loss(logits, batch_targets.long().unsqueeze(1), smoothing=0.1)
                        loss = loss / self.grad_accum_steps
                    self.scaler.scale(loss).backward()
                    current_step_loss = loss.item() * self.grad_accum_steps
                except Exception as e:
                    logger.error(f"Error during training step {self.global_step + 1}: {e}", exc_info=True)
                    if accumulation_counter % self.grad_accum_steps != 0: self.optimizer.zero_grad(set_to_none=True)
                    accumulation_counter = self.grad_accum_steps; continue
                # --- Optimizer Step ---
                if accumulation_counter % self.grad_accum_steps == 0:
                    steps_taken += 1; self.global_step += 1;
                    if pbar: pbar.update(1)
                    self.scaler.unscale_(self.optimizer)
                    # --- Modulate Gradients ---
                    with torch.no_grad():
                        for group in self.optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None: p.grad.mul_(gradient_scale_factor)
                    # --- Clip and Step ---
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # Pass loss to optimizer *before* step if needed (like EnhancedSGD)
                    if hasattr(self.optimizer, 'current_loss'):
                         self.optimizer.current_loss = current_step_loss # Pass unscaled loss for Q-learning state

                    self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True)
                    # --- Logging & Saving ---
                    if self.global_step % self.log_interval == 0:
                        lr_val = self.optimizer.param_groups[0]['lr']
                        log_msg = (f"Step: {self.global_step} | Data: {self.processed_data_count} | Mem: {len(self.memory)} | "
                                   f"Loss: {current_step_loss:.4f} | GradNorm: {grad_norm:.2f} | LR: {lr_val:.6e} | "
                                   f"Importance (Avg): {avg_importance:.3f} | GradScale: {gradient_scale_factor:.3f}")
                        logger.info(log_msg)
                        if WANDB_AVAILABLE and wandb.run:
                            wandb.log({ "live/loss": current_step_loss, "live/grad_norm": grad_norm, "live/lr": lr_val,
                                        "live/avg_importance": avg_importance, "live/grad_scale": gradient_scale_factor,
                                        "live/memory_size": len(self.memory)}, step=self.global_step)
                    if self.global_step % self.save_interval == 0: self.save_checkpoint()
                    if self.global_step % self.decay_interval == 0: logger.info(f"Decaying importance..."); self.memory.decay_importance(time.time())
        if pbar: pbar.close(); logger.info("Finished processing data stream.")

    def save_checkpoint(self):
        filepath = os.path.join(self.checkpoint_dir, f"live_hashidx_step_{self.global_step}.pt")
        model_state = self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict()
        memory_state = dict(self.memory.memory)
        checkpoint = {'global_step': self.global_step, 'processed_data_count': self.processed_data_count,'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict(), 'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None, 'hash_memory_state': memory_state }
        try: torch.save(checkpoint, filepath); logger.info(f"Live checkpoint saved to {filepath}")
        except Exception as e: logger.error(f"Failed to save live checkpoint {filepath}: {e}", exc_info=True)
    def load_checkpoint(self, filepath: str):
        if not os.path.exists(filepath): logger.error(f"Checkpoint file not found: {filepath}"); return
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            model_to_load = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            state_dict = checkpoint['model_state_dict'];
            if all(k.startswith('module.') for k in state_dict.keys()): state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            incompatible_keys = model_to_load.load_state_dict(state_dict, strict=False)
            if incompatible_keys.missing_keys: logger.warning(f"Missing keys loading model: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys: logger.warning(f"Unexpected keys loading model: {incompatible_keys.unexpected_keys}")
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else: logger.warning("Optimizer state not found in checkpoint.")
            if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.global_step = checkpoint.get('global_step', 0); self.processed_data_count = checkpoint.get('processed_data_count', 0)
            if 'hash_memory_state' in checkpoint:
                self.memory.memory = checkpoint['hash_memory_state'] # Load memory dict
                sorted_items = sorted(self.memory.memory.items(), key=lambda item: item[1].last_accessed_time) # Rebuild access order
                self.memory.access_order = deque([k for k, v in sorted_items], maxlen=self.memory.capacity)
                logger.info(f"Loaded HashIndexedMemory state with {len(self.memory.memory)} entries.")
            logger.info(f"Loaded live checkpoint from {filepath}. Resuming from Global Step {self.global_step}")
        except Exception as e: logger.error(f"Failed loading live checkpoint '{filepath}': {e}", exc_info=True)

# =====================================================================
# Example Data Stream Simulation (Identical)
# =====================================================================
def simulate_byte_stream(text_data: str, context_size: int, chunk_size: int = 1):
    logger.info("Simulating byte stream...")
    byte_array = np.array(list(text_data.encode('utf-8')), dtype=np.uint8)
    data_len = len(byte_array)
    if data_len <= context_size: logger.warning("Data too short for stream."); return
    current_pos = context_size
    while current_pos < data_len:
        end_pos = min(current_pos + chunk_size, data_len)
        for i in range(current_pos, end_pos):
            if i - context_size < 0: continue
            context = torch.tensor(byte_array[i - context_size : i], dtype=torch.long)
            target = torch.tensor(byte_array[i], dtype=torch.long)
            yield context, target
        current_pos = end_pos
    logger.info("Finished simulating byte stream.")

# =====================================================================
# Main Execution Logic
# =====================================================================
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="LIVEBSFIN - Hash-Indexed Continual Learning v1.0")
    parser.add_argument("--base_checkpoint", type=str, required=True, help="Path to initial trained BSFIN model checkpoint.")
    parser.add_argument("--data_stream_file", type=str, required=True, help="Path to a text file to simulate the data stream.")
    parser.add_argument("--context_size", type=int, default=256, help="Context size used by the model.")
    parser.add_argument("--memory_capacity", type=int, default=100000, help="Capacity of the hash-indexed memory.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for live training updates.")
    # parser.add_argument("--replay_ratio", type=float, default=0.5) # Not used in this version
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum number of optimizer steps for this run.")
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N steps.")
    parser.add_argument("--log_interval", type=int, default=20, help="Log every N steps.")
    parser.add_argument("--decay_interval", type=int, default=500, help="Steps between importance decay runs.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Initial learning rate for continuous updates.")
    parser.add_argument("--importance_factor", type=float, default=0.5, help="Influence of importance on grad scale (0-1).")
    parser.add_argument("--checkpoint_dir", type=str, default="./bsfin_live_checkpoints_v1", help="Directory for live checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP.")
    # --- BSFIN Model Args (Copied from bsfin_main.py parser) ---
    parser.add_argument("--local_hidden_size", type=int, default=256)
    parser.add_argument("--complex_dim", type=int, default=512)
    parser.add_argument("--num_complex_layers", type=int, default=6)
    parser.add_argument("--num_complex_heads", type=int, default=8)
    parser.add_argument("--decoder_memory_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--n_gram_sizes", type=int, nargs='+', default=[3, 4])
    parser.add_argument("--n_gram_vocab_size", type=int, default=30000)
    parser.add_argument("--sfin_noise_scale", type=float, default=0.05)
    parser.add_argument("--no_entanglement", action="store_true")
    parser.add_argument("--no_rope", action="store_true")
    parser.add_argument("--projection_method", type=str, default="concat")
    args = parser.parse_args()

    # --- Logging Setup ---
    log_level = logging.INFO
    log_handlers = [logging.StreamHandler()]
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_filename = os.path.join(args.checkpoint_dir, f"LIVEBSFIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    try: log_handlers.append(logging.FileHandler(log_filename))
    except Exception as log_ex: print(f"Warning: Could not create file logger: {log_ex}")
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', handlers=log_handlers, force=True)
    logger.info(f"Run Arguments: {args}")

    # --- Setup ---
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Using device: {device}")

    # --- Initialize Model ---
    model_config = { k: getattr(args, k) for k in [
        "local_hidden_size", "complex_dim", "num_complex_layers", "num_complex_heads",
        "decoder_memory_dim", "dropout", "n_gram_sizes", "n_gram_vocab_size",
        "sfin_noise_scale", "projection_method"]}
    model_config["sfin_use_entanglement"] = not args.no_entanglement
    model_config["sfin_use_rotary"] = not args.no_rope
    model_config["context_window"] = args.context_size
    model = BSFINModel(**model_config).to(device)

    # --- Load Base Checkpoint ---
    if not os.path.exists(args.base_checkpoint): logger.error(f"Base checkpoint missing: {args.base_checkpoint}"); sys.exit(1)
    try:
        logger.info(f"Loading base model state from: {args.base_checkpoint}")
        checkpoint = torch.load(args.base_checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict']
        if all(k.startswith('module.') for k in state_dict.keys()): state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys: logger.warning(f"Missing keys loading base model: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys: logger.warning(f"Unexpected keys loading base model: {incompatible_keys.unexpected_keys}")
        logger.info("Base model state loaded.")
    except Exception as e: logger.error(f"Failed loading base checkpoint: {e}", exc_info=True); sys.exit(1)

    # --- Initialize Optimizer ---
    optimizer = EnhancedSGD(model.parameters(), lr=args.learning_rate, weight_decay=0.01) # Example WD

    # --- Initialize Hash-Indexed Memory & Indexer ---
    hash_memory = HashIndexedMemory(capacity=args.memory_capacity, decay_rate=0.01, importance_threshold=0.1)
    babylon_indexer = BabylonIndex(scales=args.n_gram_sizes)

    # --- Initialize LiveTrainer ---
    live_trainer = LiveTrainerHashIndexed(
        model=model, optimizer=optimizer, hash_memory=hash_memory, babylon_indexer=babylon_indexer,
        device=device, context_size=args.context_size, batch_size=args.batch_size,
        grad_accum_steps=1, use_amp=not args.no_amp, log_interval=args.log_interval,
        save_interval=args.save_interval, decay_interval=args.decay_interval,
        checkpoint_dir=args.checkpoint_dir, max_grad_norm=1.0,
        importance_influence_factor=args.importance_factor
    )

    # --- Load Live Checkpoint (Optional) ---
    # Add logic here to find and load the latest checkpoint from args.checkpoint_dir

    # --- Prepare Data Stream ---
    try:
        with open(args.data_stream_file, 'r', encoding='utf-8') as f: text_data = f.read()
        data_iterator = simulate_byte_stream(text_data, args.context_size)
    except Exception as e: logger.error(f"Failed reading data stream file {args.data_stream_file}: {e}"); sys.exit(1)

    # --- Start Live Learning ---
    try:
        logger.info("Starting hash-indexed live learning process...")
        live_trainer.process_data_stream(data_iterator, max_steps=args.max_steps)
    except KeyboardInterrupt: logger.info("Live learning interrupted.")
    except Exception as e: logger.error(f"Error during live learning: {e}", exc_info=True)
    finally:
        logger.info("Saving final live model state...")
        live_trainer.save_checkpoint()
        logger.info("Live learning script finished.")

if __name__ == "__main__":
    main()
