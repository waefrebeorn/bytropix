#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BSFIN Main - BabylonIndex Semantic Field Interference Network (v2 - Syntax Fixed v5)

Hybrid architecture combining:
- Byte-level input and BabylonIndex patching
- Complex-valued representations and quantum-inspired interference layers
- Q-Learning enhanced optimizer and Trainer infrastructure

Version 2 incorporates implementations for previously placeholder sections:
- N-gram features in LocalEncoder.
- Learnable frequency scaling in PositionalEncoding.
- Attention mask handling in EntangledInterferenceLayer.
- Refined batch processing in BSFINModel forward pass.

Syntax fixes applied.
Removed conflicting external LR scheduler when EnhancedSGD is used.
Fixed wandb.log dictionary syntax error.
Fixed np.load with statement error in ByteIterableDataset.__init__.
Fixed AMP dtype mismatch in BSFINModel.forward output reconstruction.
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
# Define logger at the top level
logger = logging.getLogger("BSFIN")
# Basic configuration will be set in main() to allow setting level and file handler properly

# =====================================================================
# Data Structures and Configuration Classes
# =====================================================================

@dataclass
class SamplerConfig:
    """Configuration for entropy-based sampling (used in generate)."""
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# =====================================================================
# Quantum Noise Function
# =====================================================================

def add_quantum_noise(tensor, noise_prob=0.05, noise_scale=0.1, noise_type="phase_and_amplitude"):
    """Inject quantum-inspired noise."""
    if noise_scale <= 0 or not tensor.requires_grad or not tensor.is_floating_point():
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

    # Re-attach gradient history if needed
    return noisy_tensor.clone().requires_grad_(tensor.requires_grad)


# =====================================================================
# ByteTokenizer and ByteIterableDataset
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
            elif isinstance(b, torch.Tensor): # Handle tensor elements
                b_item = b.item()
                if isinstance(b_item, int) and 0 <= b_item <= 255:
                     valid_bytes.append(b_item)
                elif isinstance(b_item, float) and 0 <= int(b_item) <= 255: # Handle potential float tensors
                     valid_bytes.append(int(b_item))
        return bytes(valid_bytes).decode('utf-8', errors='replace') # Replace invalid UTF-8 sequences


class ByteIterableDataset(IterableDataset):
    """Byte-level dataset for efficient streaming from large numpy files."""
    def __init__(self, npy_file_path: str, context_size: int = 128, data_fraction: float = 1.0):
        self.npy_file_path = npy_file_path
        self.context_size = context_size
        self.data_fraction = max(0.0, min(1.0, data_fraction))
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"File not found: {npy_file_path}")
        try:
            # Load memmap object to get shape, then delete it (no 'with' statement)
            mmap_data = np.load(self.npy_file_path, mmap_mode='r')
            self.full_data_size = mmap_data.shape[0]
            del mmap_data # Explicitly delete to close the memmap handle
            gc.collect() # Suggest garbage collection

            self.data_size = int(self.full_data_size * self.data_fraction)
            if self.data_size <= self.context_size: # Need at least context_size + 1 bytes
                raise ValueError(f"Dataset size after fraction ({self.data_size}) is too small for context size ({self.context_size}). Needs at least {self.context_size + 1} bytes.")
            logger.info(f"Using {self.data_size}/{self.full_data_size} bytes ({self.data_fraction:.1%}) from {npy_file_path}")
        except Exception as e:
            logger.error(f"Error accessing or checking size of {npy_file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        # The number of possible starting indices for a context+target sequence
        return max(0, self.data_size - self.context_size)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Load data within the worker process using memmap
        bytes_data = None # Initialize to prevent UnboundLocalError in finally
        try:
            # Keep the memmap object open for the duration of iteration
            bytes_data = np.load(self.npy_file_path, mmap_mode='r')
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to load {self.npy_file_path}: {e}", exc_info=True)
            return # Stop iteration for this worker

        try:
            effective_data_len = self.data_size
            # Total number of possible starting indices within the allowed data fraction
            num_indices = max(0, effective_data_len - self.context_size)

            # Determine indices for this worker
            indices_per_worker = num_indices // num_workers
            start_idx = worker_id * indices_per_worker
            # Distribute remainder indices to the first few workers
            remainder = num_indices % num_workers
            if worker_id < remainder:
                start_idx += worker_id
                indices_per_worker += 1
            else:
                start_idx += remainder

            end_idx = start_idx + indices_per_worker

            # Ensure worker has indices to process
            if start_idx >= end_idx:
                 return

            # Use a separate random generator for each worker, seeded uniquely
            # Combine time, worker_id, and pid for better uniqueness
            seed = (worker_id + int(time.time() * 1000) + os.getpid()) % (2**32)
            rng = np.random.default_rng(seed=seed)
            # Generate indices within the worker's range and shuffle them
            worker_indices = rng.permutation(np.arange(start_idx, end_idx))

            # Iterate through the assigned and shuffled indices
            for idx in worker_indices:
                # Double check boundary condition within the loop
                if idx + self.context_size + 1 <= effective_data_len:
                    try:
                        # Slice context and target directly from memmap object
                        # Ensure slices are copied to avoid issues if memmap closes early
                        context = np.copy(bytes_data[idx : idx + self.context_size])
                        target = np.copy(bytes_data[idx + self.context_size]) # Predict the single next byte
                        # Convert numpy slices to tensors
                        context_tensor = torch.tensor(context, dtype=torch.long)
                        target_tensor = torch.tensor(target, dtype=torch.long)
                        yield context_tensor, target_tensor
                    except IndexError:
                        logger.warning(f"Worker {worker_id}: IndexError accessing data at index {idx}. Effective len: {effective_data_len}. Skipping.")
                        continue
                # else: # This condition should ideally not be hit due to range calculation
                #    logger.debug(f"Worker {worker_id}: Index {idx} out of bounds. Skipping.")
        finally:
            # Clean up the mmap object when the iterator is exhausted or an error occurs
            if bytes_data is not None:
                 # Check if it's a memmap object before deleting attributes
                 if hasattr(bytes_data, '_mmap'):
                      # Ensure the mmap object exists before trying to close
                      if bytes_data._mmap is not None:
                           bytes_data._mmap.close()
                 del bytes_data
            gc.collect() # Explicitly request garbage collection


# =====================================================================
# Babylon Index (Entropy-based Patching)
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
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5)
            # Use iterator to avoid creating a potentially large list of keys
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                # Check if key still exists before deleting (might be removed by other ops)
                if k in self.entropy_cache:
                     del self.entropy_cache[k]

    def _is_valid_utf8_boundary(self, byte_seq: Union[List[int], np.ndarray], boundary: int) -> bool:
        """Checks if a potential boundary is valid (not mid-UTF8 char)."""
        if boundary <= 0 or boundary >= len(byte_seq):
            return True
        byte_at_boundary = byte_seq[boundary]
        # Continuation bytes are in the range 0x80 (10000000) to 0xBF (10111111)
        return not (0x80 <= byte_at_boundary <= 0xBF)

    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...]]) -> float: # Use Tuple[int, ...] for type hint
        """Computes Shannon entropy for a window of bytes, using caching."""
        cache_key = None
        if isinstance(byte_window, tuple):
            cache_key = byte_window
            if cache_key in self.entropy_cache:
                return self.entropy_cache[cache_key]
            if not byte_window: return 0.0
            # Convert tuple to numpy array for bincount
            byte_window_np = np.array(byte_window, dtype=np.uint8)
        elif isinstance(byte_window, np.ndarray):
             if byte_window.size == 0: return 0.0
             byte_window_np = byte_window
             # Caching numpy arrays directly is tricky, convert to tuple if needed
             # cache_key = tuple(byte_window_np.tolist())
             # if cache_key in self.entropy_cache: return self.entropy_cache[cache_key]
        else:
             logger.warning(f"Unsupported type for compute_entropy: {type(byte_window)}")
             return 0.0

        try:
            # Ensure input is integer type for bincount
            if not np.issubdtype(byte_window_np.dtype, np.integer):
                 logger.warning(f"Non-integer dtype {byte_window_np.dtype} passed to bincount. Attempting cast.")
                 byte_window_np = byte_window_np.astype(np.uint8)
            byte_counts = np.bincount(byte_window_np, minlength=256)
        except TypeError as e:
             logger.error(f"TypeError in np.bincount. Input type: {type(byte_window_np)}, dtype: {byte_window_np.dtype}. Error: {e}")
             # Provide a default entropy value or re-raise
             return 0.0 # Or handle differently

        total_bytes = byte_counts.sum()
        if total_bytes == 0: return 0.0
        # Calculate probabilities only for non-zero counts
        probs = byte_counts[byte_counts > 0] / total_bytes
        # Calculate entropy, add epsilon for numerical stability inside log
        entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))

        if cache_key is not None: # Cache only if input was tuple or converted
            self.entropy_cache[cache_key] = entropy
            self._clean_cache()

        return entropy

    def find_patch_boundaries(self, byte_seq_tensor: torch.Tensor) -> List[int]:
        """Identifies potential patch boundaries based on entropy."""
        if byte_seq_tensor.numel() == 0: return []

        # Ensure input is 1D list of ints for processing
        if byte_seq_tensor.dim() > 1:
            if byte_seq_tensor.size(0) == 1: # Handle batch size 1
                 byte_seq_list = byte_seq_tensor[0].cpu().tolist() # Use tolist() for list of ints
            else:
                 logger.warning("find_patch_boundaries expects 1D tensor or batch size 1. Using first element.")
                 byte_seq_list = byte_seq_tensor[0].cpu().tolist()
        else:
            byte_seq_list = byte_seq_tensor.cpu().tolist()

        seq_len = len(byte_seq_list)
        min_scale = min(self.scales, default=1)
        if seq_len < min_scale: return []

        potential_boundaries = set()
        # Determine window size dynamically, ensure it's reasonable
        window_size = min(max(self.scales, default=16), seq_len // 2, 64) # Adaptive window based on scales/length, capped
        window_size = max(window_size, min_scale) # Ensure window is at least min scale

        entropies = []
        # Calculate entropy for sliding windows
        for i in range(seq_len - window_size + 1):
            # Convert window slice to tuple for caching in compute_entropy
            window_tuple = tuple(byte_seq_list[i : i + window_size])
            entropy = self.compute_entropy(window_tuple)
            entropies.append((i, entropy)) # Store window start index and entropy

        # Sort potential boundaries by entropy (descending)
        entropies.sort(key=lambda x: x[1], reverse=True)

        # Target number of boundaries (heuristic)
        num_boundaries_target = max(1, seq_len // 128)
        selected_count = 0

        # Select high-entropy, valid boundaries
        for start_pos, entropy_val in entropies:
            # Use window start position as the potential boundary point
            boundary_candidate = start_pos
            # Allow slightly more candidates initially than the target
            if entropy_val > self.min_entropy_threshold and selected_count < num_boundaries_target * 2:
                if self._is_valid_utf8_boundary(byte_seq_list, boundary_candidate):
                    potential_boundaries.add(boundary_candidate)
                    selected_count += 1
            # Stop early if enough candidates found (optional optimization)
            # if selected_count >= num_boundaries_target * 2.5: break

        # Filter and merge boundaries
        # Ensure boundaries are within valid range (0 < b < seq_len) and sorted
        final_boundaries = sorted([b for b in list(potential_boundaries) if 0 < b < seq_len])

        # Merge boundaries that are too close together
        min_patch_size = 16 # Minimum allowed patch size
        merged_boundaries = []
        if final_boundaries:
            last_boundary = 0 # Implicit start boundary
            for b in final_boundaries:
                if b - last_boundary >= min_patch_size:
                    merged_boundaries.append(b)
                    last_boundary = b
            final_boundaries = merged_boundaries

        # logger.debug(f"Found boundaries: {final_boundaries} for seq_len {seq_len}")
        return final_boundaries

    def create_patches(self, byte_seq_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Splits a 1D byte tensor into patches based on found boundaries."""
        if byte_seq_tensor.numel() == 0: return []
        if byte_seq_tensor.dim() != 1:
             # Allow batch dim of 1
             if byte_seq_tensor.dim() == 2 and byte_seq_tensor.size(0) == 1:
                  byte_seq_tensor = byte_seq_tensor.squeeze(0)
             else:
                  raise ValueError(f"create_patches expects a 1D tensor, got shape {byte_seq_tensor.shape}")

        boundaries = self.find_patch_boundaries(byte_seq_tensor)
        patches = []
        start_idx = 0
        seq_len = byte_seq_tensor.size(0)

        for end_idx in boundaries:
            # Ensure boundary is valid and creates a non-empty patch
            if start_idx < end_idx <= seq_len:
                patch = byte_seq_tensor[start_idx:end_idx]
                if patch.numel() > 0: patches.append(patch)
                start_idx = end_idx
            elif end_idx <= start_idx:
                logger.warning(f"Skipping invalid or out-of-order boundary {end_idx} <= {start_idx}")
            elif end_idx > seq_len:
                 logger.warning(f"Boundary {end_idx} exceeds sequence length {seq_len}. Ignoring.")

        # Add the final patch from the last boundary to the end
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
# Q-Learning Controller and Enhanced SGD
# =====================================================================
class GradientStats:
    """Tracks gradient statistics."""
    def __init__(self): self.reset()
    def reset(self):
        self.total_gradients = 0
        self.clipped_gradients = 0
        self.max_gradient_norm = 0.0
        self.sum_clip_ratios = 0.0
        self.step_stats = {}
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        self.total_gradients += 1
        self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
        if clipped:
            self.clipped_gradients += 1
            self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)
    def get_step_stats(self) -> dict:
        if self.total_gradients == 0:
            return {"gradients_clipped": 0, "total_gradients": 0, "clip_ratio_avg": 0.0, "max_gradient": 0.0, "clip_percentage": 0.0}
        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0
        return {"gradients_clipped": self.clipped_gradients, "total_gradients": self.total_gradients, "clip_ratio_avg": avg_clip_ratio, "max_gradient": self.max_gradient_norm, "clip_percentage": clip_percentage}
    def record_step(self, step: int):
        stats = self.get_step_stats()
        self.step_stats[step] = stats
        self.reset()
        return stats

class QController:
    """Q-Learning Controller for hyperparameter tuning."""
    def __init__(self, learning_rate: float=0.02, discount: float=0.97, epsilon: float=0.15, epsilon_decay: float=0.9995, lr_scale_bounds: tuple=(0.9, 1.1), momentum_scale_bounds: tuple=(0.95, 1.05), max_q_table_size: int=15000):
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.min_epsilon = 0.02
        self.epsilon_decay = epsilon_decay
        self.prev_loss = None
        self.prev_state = None
        self.prev_action = None
        self.lr_scale_bounds = lr_scale_bounds # Note: These bounds are informational now if action_ranges are hardcoded
        self.momentum_scale_bounds = momentum_scale_bounds # Note: These bounds are informational now if action_ranges are hardcoded
        self.loss_window = deque(maxlen=10)
        self.grad_norm_window = deque(maxlen=10)
        self.lr_window = deque(maxlen=5)
        self.momentum_window = deque(maxlen=5)

        # --- Suggestion 3 Example: Modify Action Ranges ---
        # Original action ranges (commented out for clarity):
        # self.action_ranges = {'lr_scale': np.array([0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10]),
        #                       'momentum_scale': np.array([0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05])}

        # Example: Remove the most aggressive scaling options (1.10 for LR, 1.05 for momentum)
        lr_actions = np.array([0.90, 0.95, 0.98, 1.0, 1.02, 1.05]) # Removed 1.10
        mom_actions = np.array([0.95, 0.98, 0.99, 1.0, 1.01, 1.02]) # Removed 1.05
        self.action_ranges = {'lr_scale': lr_actions, 'momentum_scale': mom_actions}
        # --- End Suggestion 3 Example ---

        self.performance_window = deque(maxlen=30)
        self.stable_steps = 0
        self.max_q_table_size = max_q_table_size
        self.q_table_access_count = {}
    def get_state(self, lr: float, momentum: float, grad_norm: float, loss: float) -> tuple:
        self.loss_window.append(loss)
        self.grad_norm_window.append(grad_norm)
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)
        loss_trend_bin = 2
        grad_norm_bin = 2
        lr_bin = 2
        momentum_bin = 1
        if len(self.loss_window) >= 5:
            y = np.array(list(self.loss_window)[-5:])
            x = np.arange(len(y))
            # Handle potential NaN/Inf in polyfit if y values are constant or problematic
            try:
                # Check for constant y values
                if np.all(y == y[0]):
                     slope = 0.0
                else:
                     slope = np.polyfit(x, y, 1)[0]
                normalized_slope = slope / (np.mean(y) + 1e-6)
                loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05])
            except (np.linalg.LinAlgError, ValueError):
                loss_trend_bin = 2 # Default to stable if polyfit fails
        if self.grad_norm_window:
            avg_grad_norm = np.mean(list(self.grad_norm_window))
            grad_norm_bin = np.digitize(avg_grad_norm, bins=[0.1, 0.5, 1.5, 5.0])
        lr_bin = np.digitize(lr / 1e-3, bins=[0.1, 0.5, 2.0, 5.0])
        momentum_bin = np.digitize(momentum, bins=[0.85, 0.92, 0.97])
        state = (loss_trend_bin, grad_norm_bin, lr_bin, momentum_bin)
        self.q_table_access_count[state] = self.q_table_access_count.get(state, 0) + 1
        return state
    def compute_reward(self, current_loss: float, prev_loss: Optional[float], grad_norm: float) -> float:
        reward = 0.0
        if prev_loss is not None:
            loss_reduction = prev_loss - current_loss
            reward += np.tanh((loss_reduction / (prev_loss + 1e-6)) * 10)
        if grad_norm > 5.0:
            reward -= 0.2 * min(1.0, (grad_norm - 5.0) / 5.0)
        elif grad_norm < 1e-4:
            reward -= 0.1
        self.performance_window.append(reward)
        self.stable_steps = self.stable_steps + 1 if reward > 0 else 0
        if reward > 0:
            reward += min(0.2, 0.05 * math.log1p(self.stable_steps))
        return float(np.clip(reward, -1.0, 1.0))
    def choose_action(self, state: tuple) -> Dict[str, float]:
        if state not in self.q_table:
            self.q_table[state] = {p: np.zeros(len(s)) for p, s in self.action_ranges.items()}
            self._manage_q_table_size()
        action = {}
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        for param, space in self.action_ranges.items():
            if random.random() < self.epsilon:
                chosen_idx = random.randrange(len(space))
            else:
                q_values = self.q_table[state][param]
                # Handle cases where q_values might be empty or all NaN/Inf (though unlikely with init)
                if len(q_values) > 0 and np.any(np.isfinite(q_values)):
                     max_q = np.nanmax(q_values) # Use nanmax to handle potential NaNs
                     best_indices = np.where(np.abs(q_values - max_q) < 1e-6)[0]
                     if len(best_indices) > 0:
                          chosen_idx = np.random.choice(best_indices)
                     else: # Fallback if all are NaN or some other issue
                          chosen_idx = random.randrange(len(space))
                else: # Fallback if q_values is empty or all non-finite
                     chosen_idx = random.randrange(len(space))
            action[param] = float(space[chosen_idx])
        return action
    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: tuple):
        if next_state not in self.q_table:
            self.q_table[next_state] = {p: np.zeros(len(s)) for p, s in self.action_ranges.items()}
            self._manage_q_table_size()
        for param, value in action.items():
            space = self.action_ranges[param]
            try:
                # Find the index of the action value in the discrete space
                action_idx = np.abs(space - value).argmin()
                # Verify that the found index actually corresponds to the value (within tolerance)
                if not np.isclose(space[action_idx], value):
                     raise ValueError("Action value not found precisely in space.")
            except ValueError:
                logger.warning(f"Q-update: Action value {value} not found for {param}. Skipping update for this param.")
                continue

            current_q = self.q_table[state][param][action_idx]
            # Ensure next_state Q-values are valid before taking max
            next_q_values = self.q_table[next_state][param]
            if len(next_q_values) > 0 and np.any(np.isfinite(next_q_values)):
                 max_future_q = np.nanmax(next_q_values)
            else:
                 max_future_q = 0.0 # Default if next state Q-values are problematic

            td_error = reward + self.gamma * max_future_q - current_q
            self.q_table[state][param][action_idx] += self.alpha * td_error
    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            try:
                # Ensure access count dict is not empty before finding min
                if self.q_table_access_count:
                     least_accessed_state = min(self.q_table_access_count, key=self.q_table_access_count.get)
                     # Check if state exists before deleting (could be removed concurrently?)
                     if least_accessed_state in self.q_table:
                          del self.q_table[least_accessed_state]
                     if least_accessed_state in self.q_table_access_count:
                          del self.q_table_access_count[least_accessed_state]
                # else: logger.debug("Q-table access count is empty, cannot prune.")
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not prune Q-table: {e}")

class EnhancedSGD(torch.optim.Optimizer):
    """
    Enhanced SGD Optimizer with Q-Learning tuning.
    (Updated with modified clamping limits as per Suggestion 2)
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float=0.003, momentum: float=0.9, weight_decay: float=0.005, max_grad_norm: float=1.0, q_learning_config: Dict[str,Any]={}):
        if lr<0.0 or momentum<0.0 or weight_decay<0.0: raise ValueError("Invalid hyperparameter value")
        defaults=dict(lr=lr,momentum=momentum,weight_decay=weight_decay)
        super().__init__(params,defaults)
        # Assuming QController class is defined and imported correctly
        self.q_controller=QController(**q_learning_config)
        self.max_grad_norm=max_grad_norm # Note: Actual clipping is usually handled by the Trainer
        self._step_count=0
        self.current_loss: Optional[float] = None # Explicitly type hint
        # Assuming GradientStats class is defined and imported correctly
        self.gradient_stats=GradientStats()
        # Initialize momentum buffer
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad: # Only create buffer for trainable params
                     # Ensure state is initialized for each parameter
                     if p not in self.state:
                         self.state[p] = {}
                     self.state[p]['momentum_buffer']=torch.zeros_like(p,memory_format=torch.preserve_format)

    def zero_grad(self, set_to_none: bool=True):
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Optional[callable]=None) -> Optional[torch.Tensor]:
        """Performs a single optimization step with updated clamping."""
        loss: Optional[torch.Tensor] = None
        # self.current_loss should be set by the Trainer *before* calling step

        grad_norm_avg=self._get_average_grad_norm()
        q_action=None
        if self.current_loss is not None and grad_norm_avg is not None:
            # Use LR/Momentum from the first parameter group for Q-state
            current_lr = self.param_groups[0]['lr']
            current_momentum = self.param_groups[0]['momentum']
            q_state=self.q_controller.get_state(lr=current_lr, momentum=current_momentum, grad_norm=grad_norm_avg, loss=self.current_loss)
            if self.q_controller.prev_state is not None and self.q_controller.prev_action is not None:
                reward=self.q_controller.compute_reward(current_loss=self.current_loss, prev_loss=self.q_controller.prev_loss, grad_norm=grad_norm_avg)
                self.q_controller.update(state=self.q_controller.prev_state, action=self.q_controller.prev_action, reward=reward, next_state=q_state)
            q_action=self.q_controller.choose_action(q_state)

            # Apply Q-action to all parameter groups and clamp results
            for group in self.param_groups:
                 # --- Modified Absolute Clamping Limits ---
                 # Example: Lower the maximum allowed learning rate to 0.01
                 group['lr'] = float(np.clip(group['lr'] * q_action['lr_scale'], 1e-7, 0.01)) # Max LR is now 0.01
                 # Example: Slightly raise the minimum momentum
                 group['momentum'] = float(np.clip(group['momentum'] * q_action['momentum_scale'], 0.7, 0.999)) # Min momentum is now 0.7
                 # --- End Modified Clamping ---

            self.q_controller.prev_state=q_state
            self.q_controller.prev_action=q_action
            self.q_controller.prev_loss=self.current_loss
        else:
             # Log if Q-learning step was skipped
             if self.current_loss is None: logger.debug("Skipping Q-learning update: current_loss is None.")
             if grad_norm_avg is None: logger.debug("Skipping Q-learning update: grad_norm_avg is None.")

        # --- Parameter Update ---
        for group in self.param_groups:
            lr = group['lr'] # Use the potentially modified and clamped LR
            momentum = group['momentum'] # Use the potentially modified and clamped momentum
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None: continue
                if not p.requires_grad: continue # Skip params that don't require grads

                grad = p.grad
                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Momentum update
                param_state = self.state[p]
                # Check if momentum buffer exists (might not if param added dynamically)
                if 'momentum_buffer' not in param_state:
                     param_state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(grad) # m = beta*m + grad

                # Parameter update: p = p - lr * m
                p.add_(buf, alpha=-lr)

        self._step_count += 1
        # Return the loss value if it was computed via closure (currently not used this way)
        return loss

    def _get_average_grad_norm(self) -> Optional[float]:
        """Calculates the average L2 norm of gradients across all parameters."""
        total_norm_sq = 0.0
        num_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if not p.requires_grad: continue
                # Ensure gradient is finite before calculating norm
                if not torch.isfinite(p.grad).all():
                     # logger.warning(f"Non-finite gradient found for parameter. Skipping norm calculation for this param.")
                     continue # Skip non-finite gradients

                param_norm_sq = torch.norm(p.grad.detach())**2
                # Double-check norm is finite (though detach() shouldn't change finiteness)
                if torch.isfinite(param_norm_sq):
                    total_norm_sq += param_norm_sq
                    num_params += 1
                # else: logger.warning(f"Norm calculation resulted in non-finite value.")


        if num_params == 0:
             # logger.debug("No valid gradients found to calculate average norm.")
             return None # Avoid division by zero or sqrt of zero/negative

        avg_norm_sq = total_norm_sq / num_params
        # Check for negative values before sqrt (unlikely with squared norms)
        # and handle potential inf/nan from the division if total_norm_sq was inf
        if not np.isfinite(avg_norm_sq.item()):
            # logger.warning(f"Average squared norm is non-finite ({avg_norm_sq.item()}). Returning None.")
            return None

        return math.sqrt(max(0.0, avg_norm_sq.item()))

# =====================================================================
# Core Model Components
# =====================================================================

class CrossAttentionBlock(nn.Module):
    """Standard Cross-attention block."""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        original_num_heads = num_heads
        if hidden_size % num_heads != 0:
            possible_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
            if not possible_heads:
                raise ValueError(f"hidden_size {hidden_size} not divisible by any number of heads <= {original_num_heads}")
            num_heads = possible_heads[0]
            logger.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")
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
        seq_len_kv = keys_values.size(1) # Length of key/value sequence
        device = queries.device

        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)

        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nq, d]
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nkv, d]
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nkv, d]

        # --- Attention Mask Handling ---
        # Goal: Create a mask of shape [B, h, Nq, Nkv] where True means MASK
        attn_mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2: # Shape [B, Nkv] -> Applies to all queries
                attn_mask_bool = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, num_queries, -1) # [B, h, Nq, Nkv]
            elif attention_mask.dim() == 3: # Shape [B, Nq, Nkv]
                attn_mask_bool = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # [B, h, Nq, Nkv]
            elif attention_mask.dim() == 4: # Shape [B, h, Nq, Nkv]
                attn_mask_bool = attention_mask
            else:
                logger.warning(f"Unsupported attention mask shape: {attention_mask.shape}. Ignoring mask.")
            # Ensure mask is boolean
            if attn_mask_bool is not None:
                 attn_mask_bool = attn_mask_bool.bool()


        # --- Attention Calculation ---
        # Use flash attention if available and mask is suitable (or None)
        use_flash = hasattr(F, 'scaled_dot_product_attention') and (attn_mask_bool is None or attn_mask_bool.dtype == torch.bool)

        if use_flash:
            # Flash attention expects mask where True means KEEP (inverse of our mask)
            flash_mask = ~attn_mask_bool if attn_mask_bool is not None else None
            try:
                 output = F.scaled_dot_product_attention(
                     q, k, v,
                     attn_mask=flash_mask, # Needs boolean mask where True=KEEP
                     dropout_p=self.dropout.p if self.training else 0.0,
                     is_causal=False
                 )
            except Exception as e:
                 logger.warning(f"Flash attention failed: {e}. Falling back to manual attention.")
                 use_flash = False # Fallback to manual calculation

        if not use_flash: # Manual calculation or fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale # [B, h, Nq, Nkv]
            if attn_mask_bool is not None:
                scores = scores.masked_fill(attn_mask_bool, float('-inf'))
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v) # [B, h, Nq, d]

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size) # [B, Nq, H]
        output = self.out_proj(output)
        return output


class LocalEncoder(nn.Module):
    """Encodes byte patches into real-valued representations using a Transformer, incorporating N-grams."""
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1, n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__()
        self.hidden_size=hidden_size
        self.byte_embeddings=nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0/math.sqrt(hidden_size))

        # N-gram Embeddings Setup
        self.n_gram_sizes=sorted(list(set(n_gram_sizes))) if n_gram_sizes else []
        self.n_gram_embeddings=None
        self.n_gram_vocab_size=n_gram_vocab_size
        if self.n_gram_sizes:
            self.n_gram_embeddings=nn.ModuleDict({
                f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes
            })
            # Initialize n-gram embeddings
            for emb in self.n_gram_embeddings.values():
                 nn.init.normal_(emb.weight, mean=0.0, std=0.02) # Smaller std for n-grams
            logger.info(f"LocalEncoder using N-grams: {self.n_gram_sizes} with vocab size {n_gram_vocab_size}")

        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=num_heads,dim_feedforward=hidden_size*4,dropout=dropout,batch_first=True, activation=F.gelu)
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_pooling_attention=CrossAttentionBlock(hidden_size, num_heads, dropout)
        self.patch_query=nn.Parameter(torch.randn(1, 1, hidden_size)) # Learnable query for pooling
        self.norm=nn.LayerNorm(hidden_size,eps=1e-6)
        self.dropout=nn.Dropout(dropout)

    def _get_n_gram_hashes(self, byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        """Calculates rolling hashes for n-grams (simple modulo hashing)."""
        # byte_sequence shape: [B, SeqLen]
        if byte_sequence.size(1) < n:
            # Return empty tensor of appropriate shape if sequence is too short
            return torch.empty(byte_sequence.size(0), 0, dtype=torch.long, device=byte_sequence.device)

        # Use unfold to get sliding windows
        windows = byte_sequence.unfold(1, n, 1) # Shape [B, NumWindows, n]

        # Simple sum hashing (less unique, but faster)
        # Ensure dtype is appropriate for sum (long or float) before sum
        hashes = windows.long().sum(dim=-1) # Sum bytes in window

        # Modulo to fit vocab size
        return hashes % self.n_gram_vocab_size # Shape [B, NumWindows]

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates a causal mask (True = ignore)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, patches: List[torch.Tensor]) -> torch.Tensor:
        """
        Encodes a list of byte patches (assumed from a single batch item).
        Input: List[Tensor[patch_len]]
        Output: Tensor[1, num_patches, hidden_size]
        """
        if not patches:
            # Determine device from parameters if patches list is empty
            device = next(self.parameters()).device
            return torch.empty((1, 0, self.hidden_size), device=device)

        # Assuming input is List[Tensor[patch_len]] for one sequence
        batch_size = 1 # Process one sequence at a time
        device = patches[0].device
        patch_representations = []

        for patch_bytes in patches: # patch_bytes is [patch_len]
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue

            # Add batch dimension for processing: [1, patch_len]
            patch_bytes_batched = patch_bytes.unsqueeze(0)

            # 1. Get byte embeddings
            x = self.byte_embeddings(patch_bytes_batched) # [1, patch_len, H]

            # 2. Add N-gram features
            if self.n_gram_embeddings:
                n_gram_features = torch.zeros_like(x) # [1, patch_len, H]
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        # Get hashes for this n-gram size
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_batched, n) # [1, NumWindows]
                        # Get embeddings for these hashes
                        # Shape [1, NumWindows, H]
                        ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes)
                        # Add embeddings to the corresponding positions
                        # Simplest: Add n-gram embedding starting at pos 'i' to byte embedding at pos 'i'.
                        # Pad ngram_embeds to match patch_len
                        num_windows = ngram_embeds.size(1)
                        if num_windows < patch_len:
                             padding_size = patch_len - num_windows
                             padding = torch.zeros(1, padding_size, self.hidden_size, device=device)
                             # Add padding at the end
                             ngram_embeds_padded = torch.cat([ngram_embeds, padding], dim=1)
                        else:
                             # Truncate if more windows than length (shouldn't happen with unfold step=1)
                             ngram_embeds_padded = ngram_embeds[:, :patch_len, :]

                        n_gram_features += ngram_embeds_padded # Accumulate features from different Ns
                # Add combined n-gram features to byte embeddings
                x = x + n_gram_features

            x = self.dropout(x)

            # 3. Process with Transformer Encoder
            # Using causal mask for self-attention within the patch.
            causal_mask = self.create_causal_mask(patch_len, device) # [patch_len, patch_len]
            processed_bytes = self.transformer(x, mask=causal_mask) # [1, patch_len, H]

            # 4. Pool patch representation using Cross-Attention
            # Query attends to the processed byte sequence of the patch
            # Expand query to match batch dim 1
            batch_query = self.patch_query # Already [1, 1, H]
            patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes) # [1, 1, H]
            patch_representations.append(patch_repr)

        if not patch_representations:
             return torch.empty((1, 0, self.hidden_size), device=device)

        patches_combined = torch.cat(patch_representations, dim=1) # [1, num_patches, H]
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
            self.proj = nn.Linear(complex_dim, output_dim) # Loses phase
        else:
            logger.warning(f"Unknown ComplexToRealProjection method '{method}'. Defaulting to 'concat'.")
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
        else: # Default concat
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

        # --- Fixed Positional Encoding Base ---
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term_base = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
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
        self.register_buffer('div_term_base', div_term_base) # Store base frequencies

        # --- Learnable Parameters (if enabled) ---
        if learnable:
             # Add Batch and Seq dims for broadcast: [1, 1, Dim]
             self.real_scale = nn.Parameter(torch.ones(1, 1, dim))
             self.imag_scale = nn.Parameter(torch.ones(1, 1, dim))
             self.real_shift = nn.Parameter(torch.zeros(1, 1, dim))
             self.imag_shift = nn.Parameter(torch.zeros(1, 1, dim))
             # Learnable scaling factors for base frequencies (one per frequency/pair)
             self.frequency_scale_factors = nn.Parameter(torch.ones(dim // 2)) # Shape [dim/2]
             logger.info("Using learnable Positional Encoding.")

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adds positional encoding to complex input."""
        real, imag = x
        # Input shape: [B, Seq, Dim]
        seq_len = real.size(1)
        device = real.device

        if self.learnable:
            # Apply learnable frequency scaling factors to base frequencies
            # Ensure factors are positive (e.g., using softplus or clamp > 0)
            # Using clamp(min=1e-2) to avoid zero or negative frequencies
            scaled_div_term = self.div_term_base.to(device) * torch.clamp(self.frequency_scale_factors, min=1e-2) # Shape [dim/2]

            # Recalculate PE with scaled frequencies for the required sequence length
            position = torch.arange(seq_len, device=device).unsqueeze(1).float() # [Seq, 1]
            pe_real_learn = torch.zeros(seq_len, self.dim, device=device)
            pe_imag_learn = torch.zeros(seq_len, self.dim, device=device)

            # Calculate angles using scaled frequencies
            angles = position * scaled_div_term # [Seq, dim/2]

            # Fill PE matrices
            pe_real_learn[:, 0::2] = torch.sin(angles)
            pe_real_learn[:, 1::2] = torch.cos(angles)
            # Assuming phase_shift=True logic for learnable version
            pe_imag_learn[:, 0::2] = torch.cos(angles)
            pe_imag_learn[:, 1::2] = -torch.sin(angles)

            # Apply learnable scale and shift (broadcasts across batch and sequence dims)
            pe_real = pe_real_learn * self.real_scale + self.real_shift
            pe_imag = pe_imag_learn * self.imag_scale + self.imag_shift
            return real + pe_real, imag + pe_imag
        else: # Use fixed positional encoding
             # Slice the base PE to match sequence length and add
             return real + self.pe_real_base[:seq_len, :].to(device), imag + self.pe_imag_base[:seq_len, :].to(device)


class EntangledInterferenceLayer(nn.Module):
    """Quantum-inspired interference layer with mask handling."""
    def __init__(self, dim, heads=8, dropout=0.1, interference_type="quantum", use_entanglement=True, noise_scale=0.1, use_rotary=True, adaptive_attention=True):
        super().__init__()
        if dim % heads != 0: raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.interference_type = interference_type
        self.use_entanglement = use_entanglement
        self.noise_scale = noise_scale
        self.use_rotary = use_rotary
        self.adaptive_attention = adaptive_attention
        self.phase_shifts = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        if use_entanglement:
            self.entanglement_matrix = nn.Parameter(torch.eye(heads) + torch.randn(heads, heads) * 0.01)
        else:
            self.register_buffer('entanglement_matrix', torch.eye(heads), persistent=False)
        self.q_real = nn.Linear(dim, dim)
        self.k_real = nn.Linear(dim, dim)
        self.v_real = nn.Linear(dim, dim)
        self.q_imag = nn.Linear(dim, dim)
        self.k_imag = nn.Linear(dim, dim)
        self.v_imag = nn.Linear(dim, dim)
        self.out_real = nn.Linear(dim, dim)
        self.out_imag = nn.Linear(dim, dim)
        if use_rotary:
            self.rotary_dim = min(self.head_dim, 32)
            base_freqs = 10000.0**(-torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
            if adaptive_attention:
                self.rotary_freqs = nn.Parameter(base_freqs)
            else:
                self.register_buffer('rotary_freqs', base_freqs, persistent=False)
        self.interference_strength = nn.Parameter(torch.ones(1))
        if adaptive_attention:
            self.attention_temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('attention_temperature', torch.ones(1), persistent=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.saved_attn_weights = None

    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies Rotary Positional Embedding."""
        # q, k shape: [B, h, S, d]
        if not self.use_rotary or self.rotary_dim <= 0: return q, k
        device = q.device
        dim_rotary = self.rotary_dim
        q_rot = q[..., :dim_rotary]
        q_pass = q[..., dim_rotary:]
        k_rot = k[..., :dim_rotary]
        k_pass = k[..., dim_rotary:]
        position = torch.arange(seq_len, device=device).float()
        freqs = self.rotary_freqs.to(device)
        emb = torch.outer(position, freqs) # [S, rotary_dim/2]
        cos_emb = torch.cos(emb).unsqueeze(0).unsqueeze(1) # [1, 1, S, rotary_dim/2]
        sin_emb = torch.sin(emb).unsqueeze(0).unsqueeze(1) # [1, 1, S, rotary_dim/2]
        # Reshape features for rotation: [B, h, S, Dr] -> [B, h, S, Dr/2, 2]
        q_rot = q_rot.reshape(*q_rot.shape[:-1], -1, 2)
        k_rot = k_rot.reshape(*k_rot.shape[:-1], -1, 2)
        # Reshape cos/sin for broadcast: [1, 1, S, Dr/2] -> [1, 1, S, Dr/2, 1]
        cos = cos_emb.unsqueeze(-1)
        sin = sin_emb.unsqueeze(-1)
        # Apply rotation
        q_rot_out = torch.zeros_like(q_rot)
        k_rot_out = torch.zeros_like(k_rot)
        q_rot_out[..., 0] = q_rot[..., 0] * cos[..., 0] - q_rot[..., 1] * sin[..., 0]
        q_rot_out[..., 1] = q_rot[..., 1] * cos[..., 0] + q_rot[..., 0] * sin[..., 0]
        k_rot_out[..., 0] = k_rot[..., 0] * cos[..., 0] - k_rot[..., 1] * sin[..., 0]
        k_rot_out[..., 1] = k_rot[..., 1] * cos[..., 0] + k_rot[..., 0] * sin[..., 0]
        # Reshape back and concatenate
        q_rot_out = q_rot_out.flatten(start_dim=-2) # [B, h, S, Dr]
        k_rot_out = k_rot_out.flatten(start_dim=-2) # [B, h, S, Dr]
        q_out = torch.cat([q_rot_out, q_pass], dim=-1) # [B, h, S, d]
        k_out = torch.cat([k_rot_out, k_pass], dim=-1) # [B, h, S, d]
        return q_out, k_out

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tuple of (real, imag) tensors, shape [B, S, D]
            attention_mask: Optional mask, shape [B, S]. True indicates position should be MASKED (ignored).
        """
        real, imag = x
        batch_size, seq_len, _ = real.shape
        device = real.device
        # 1. Project inputs
        q_r = self.q_real(real).view(batch_size, seq_len, self.heads, self.head_dim)
        k_r = self.k_real(real).view(batch_size, seq_len, self.heads, self.head_dim)
        v_r = self.v_real(real).view(batch_size, seq_len, self.heads, self.head_dim)
        q_i = self.q_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim)
        k_i = self.k_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim)
        v_i = self.v_imag(imag).view(batch_size, seq_len, self.heads, self.head_dim)
        # 2. Noise
        if self.training and self.noise_scale > 0:
            q_r = add_quantum_noise(q_r, noise_scale=self.noise_scale)
            q_i = add_quantum_noise(q_i, noise_scale=self.noise_scale)
            k_r = add_quantum_noise(k_r, noise_scale=self.noise_scale)
            k_i = add_quantum_noise(k_i, noise_scale=self.noise_scale)
        # 3. RoPE (transpose before, result is [B, h, S, d])
        q_r, k_r = self._apply_rotary_pos_emb(q_r.transpose(1, 2), k_r.transpose(1, 2), seq_len)
        q_i, k_i = self._apply_rotary_pos_emb(q_i.transpose(1, 2), k_i.transpose(1, 2), seq_len)
        v_r = v_r.transpose(1, 2)
        v_i = v_i.transpose(1, 2) # Transpose V as well
        # 4. Entanglement
        entanglement_matrix_eff = self.entanglement_matrix.to(device)
        if self.use_entanglement:
            q_r = torch.einsum("bhsd,hx->bxsd", q_r, entanglement_matrix_eff)
            q_i = torch.einsum("bhsd,hx->bxsd", q_i, entanglement_matrix_eff)
            k_r = torch.einsum("bhsd,hx->bxsd", k_r, entanglement_matrix_eff)
            k_i = torch.einsum("bhsd,hx->bxsd", k_i, entanglement_matrix_eff)
        # 5. Phase Shifts
        phase_cos = torch.cos(self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device) # [1, h, 1, d]
        phase_sin = torch.sin(self.phase_shifts).unsqueeze(0).unsqueeze(2).to(device) # [1, h, 1, d]
        q_r_shifted = q_r * phase_cos - q_i * phase_sin
        q_i_shifted = q_r * phase_sin + q_i * phase_cos
        k_r_shifted = k_r * phase_cos - k_i * phase_sin
        k_i_shifted = k_r * phase_sin + k_i * phase_cos
        q_r, q_i = q_r_shifted, q_i_shifted
        k_r, k_i = k_r_shifted, k_i_shifted
        # 6. Attention Scores
        scale = 1.0 / math.sqrt(self.head_dim)
        if self.interference_type == "quantum":
            attn_r = torch.matmul(q_r, k_r.transpose(-2, -1)) + torch.matmul(q_i, k_i.transpose(-2, -1))
            attn_i = torch.matmul(q_i, k_r.transpose(-2, -1)) - torch.matmul(q_r, k_i.transpose(-2, -1))
            attn_r *= scale
            attn_i *= scale
            attn_mag = torch.sqrt(attn_r**2 + attn_i**2 + 1e-6)
        else: # classical
            attn_mag = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale

        # 7. Apply Masking (Causal + Padding)
        # Causal mask: True for upper triangle (masked)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(1) # [1, 1, S, S]
        final_mask = causal_mask # Start with causal mask

        # Padding mask (input `attention_mask`): True means MASKED position
        if attention_mask is not None:
             # Expected shape [B, S] -> expand to [B, 1, 1, S] for broadcasting with scores [B, h, S, S]
             if attention_mask.dim() == 2:
                  padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool() # [B, 1, 1, S]
                  # Ensure padding_mask is broadcastable to final_mask shape [1, 1, S, S] or attn_mag shape [B, h, S, S]
                  # It needs to expand along head and query dim (dim 1 and 2)
                  final_mask = final_mask | padding_mask # Combine: True if causal OR padding
             elif attention_mask.dim() == 4: # Assume [B, 1, S_q, S_k] or similar
                  # Ensure it broadcasts correctly with [B, h, S, S]
                  # If S_q=1, it should broadcast. If S_q=S, it should match.
                  final_mask = final_mask | attention_mask.bool()
             else:
                  logger.warning(f"Unsupported attention mask shape {attention_mask.shape} in EntangledInterferenceLayer. Ignoring padding mask.")


        # Apply final mask to scores (set masked positions to -inf)
        # Ensure final_mask is broadcastable to attn_mag shape [B, h, S, S]
        attn_mag = attn_mag.masked_fill(final_mask, float('-inf'))

        # 8. Softmax
        temp = torch.clamp(self.attention_temperature.to(device), min=1e-2) # Ensure temp > 0
        strength = torch.sigmoid(self.interference_strength.to(device))
        # Apply strength and temperature
        attn_weights = F.softmax((attn_mag * strength) / temp, dim=-1)
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)
        self.saved_attn_weights = attn_weights.detach().cpu() # Save weights for analysis

        # 9. Weighted Sum (Value)
        out_r = torch.matmul(attn_weights, v_r) # [B, h, S, d]
        out_i = torch.matmul(attn_weights, v_i) # [B, h, S, d]
        # 10. Reshape and Project Output
        out_r = out_r.transpose(1, 2).reshape(batch_size, seq_len, self.dim) # [B, S, D]
        out_i = out_i.transpose(1, 2).reshape(batch_size, seq_len, self.dim) # [B, S, D]
        out_r = self.out_real(out_r)
        out_i = self.out_imag(out_i)
        # Apply residual dropout
        out_r = self.resid_dropout(out_r)
        out_i = self.resid_dropout(out_i)
        return (out_r, out_i)


class LocalDecoder(nn.Module):
    """Decodes processed REAL patch representations back to bytes using a TransformerDecoder."""
    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        self.memory_projection = nn.Linear(global_hidden_size, hidden_size)
        nn.init.normal_(self.memory_projection.weight, std=0.02)
        nn.init.zeros_(self.memory_projection.bias)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, activation=F.gelu)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
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

        tgt_embed = self.byte_embeddings(tgt_byte_seq) # [B, T, H]
        tgt_embed = self.dropout(tgt_embed)
        projected_memory = self.memory_projection(memory) # [B, M, H]

        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(tgt_len, device) # [T, T]

        # TransformerDecoder expects masks where True means ignore.
        # memory_key_padding_mask (if provided) should also have True for ignored positions.
        output = self.transformer(
            tgt=tgt_embed,
            memory=projected_memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask # Pass the padding mask for memory
        ) # [B, T, H]
        byte_logits = self.byte_pred(output) # [B, T, 256]
        return byte_logits


# =====================================================================
# BSFIN Model Definition
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
        self.context_window = context_window
        if complex_dim % num_complex_heads != 0:
            raise ValueError(f"complex_dim ({complex_dim}) must be divisible by num_complex_heads ({num_complex_heads})")

        # 1. Patching
        self.patcher = BabylonIndex(scales=n_gram_sizes)
        # 2. Local Encoding (with N-grams)
        self.local_encoder = LocalEncoder(local_hidden_size, num_layers=1, num_heads=8, dropout=dropout, n_gram_sizes=n_gram_sizes, n_gram_vocab_size=n_gram_vocab_size)
        # 3. Real -> Complex Projection
        self.real_to_complex = RealToComplexProjection(local_hidden_size, complex_dim)
        # 4. Complex Positional Encoding (Learnable)
        self.complex_pos_encoding = PositionalEncoding(complex_dim, max_len=1024, learnable=True) # Increased max_len
        # 5. Complex Interference Stack
        self.complex_norm_in = ComplexLayerNorm(complex_dim, coupled=True)
        self.complex_interference_layers = nn.ModuleList([
            EntangledInterferenceLayer(complex_dim, num_complex_heads, dropout, noise_scale=sfin_noise_scale, use_entanglement=sfin_use_entanglement, use_rotary=sfin_use_rotary, adaptive_attention=True)
            for _ in range(num_complex_layers)])
        self.complex_norms_mid = nn.ModuleList([ComplexLayerNorm(complex_dim, coupled=True) for _ in range(num_complex_layers)])
        self.complex_dropout = nn.Dropout(dropout)
        # 6. Complex -> Real Projection
        self.complex_to_real = ComplexToRealProjection(complex_dim, decoder_memory_dim, method=projection_method)
        # 7. Local Decoder
        self.local_decoder = LocalDecoder(local_hidden_size, decoder_memory_dim, num_layers=4, num_heads=8, dropout=dropout)

        logger.info(f"BSFIN Initialized: LocalDim={local_hidden_size}, ComplexDim={complex_dim}, ComplexLayers={num_complex_layers}, DecoderMemDim={decoder_memory_dim}, Dropout={dropout}, SFIN Noise={sfin_noise_scale}, Entangle={sfin_use_entanglement}, RoPE={sfin_use_rotary}, N-grams={n_gram_sizes}")

    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            byte_seq: Input context tensor, shape [B, S_in].
            target_byte_seq: Target sequence for decoder, shape [B, S_tgt].
        """
        batch_size = byte_seq.size(0)
        device = byte_seq.device

        # --- Patching and Encoding per Sequence ---
        batch_patch_repr_list = []
        num_patches_per_item = []
        valid_batch_indices = [] # Indices of batch items that produced patches

        for i in range(batch_size):
            seq = byte_seq[i] # [S_in]
            patches = self.patcher.create_patches(seq) # List[Tensor[patch_len]]
            if patches:
                # Encode patches for this sequence (returns [1, num_p, local_hidden])
                real_patch_repr_single = self.local_encoder(patches)
                if real_patch_repr_single.numel() > 0 and real_patch_repr_single.size(1) > 0: # Check num_patches > 0
                    batch_patch_repr_list.append(real_patch_repr_single)
                    num_patches_per_item.append(real_patch_repr_single.size(1))
                    valid_batch_indices.append(i)
                # else: logger.debug(f"Batch item {i}: Encoding yielded empty tensor or zero patches.")
            # else: logger.debug(f"Batch item {i}: Patching yielded no patches.")

        # If no sequences produced valid patches
        if not batch_patch_repr_list:
             target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
             logger.warning("No valid patches produced for any item in the batch.")
             # Return zeros matching expected output shape
             return torch.zeros((batch_size, target_len, 256), device=device)

        # --- Pad and Stack Patch Representations ---
        max_num_patches = max(num_patches_per_item) if num_patches_per_item else 0
        if max_num_patches == 0: # Should be caught by above check, but as safeguard
             target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
             logger.warning("Max number of patches is zero after filtering.")
             return torch.zeros((batch_size, target_len, 256), device=device)

        padded_repr_list = []
        for repr_tensor in batch_patch_repr_list:
             num_patches = repr_tensor.size(1)
             padding_size = max_num_patches - num_patches
             if padding_size > 0:
                 # Pad with zeros on the right (sequence dimension for patches)
                 padding = torch.zeros((1, padding_size, self.local_hidden_size), device=device)
                 padded_repr = torch.cat([repr_tensor, padding], dim=1)
             else:
                 padded_repr = repr_tensor
             padded_repr_list.append(padded_repr)

        # Stack into a batch tensor containing only valid items
        real_patch_repr_batched = torch.cat(padded_repr_list, dim=0) # [B_valid, max_num_p, local_hidden]

        # Create padding mask for the complex layers (True = MASKED)
        num_valid_patches_tensor = torch.tensor(num_patches_per_item, device=device) # [B_valid]
        # Mask where index >= num_valid_patches
        memory_padding_mask = torch.arange(max_num_patches, device=device)[None, :] >= num_valid_patches_tensor[:, None] # [B_valid, max_num_p]

        # --- Complex Processing ---
        complex_patch_repr = self.real_to_complex(real_patch_repr_batched)
        complex_patch_repr = self.complex_pos_encoding(complex_patch_repr)
        complex_patch_repr = self.complex_norm_in(complex_patch_repr)
        real, imag = complex_patch_repr
        for i, layer in enumerate(self.complex_interference_layers):
            real_res, imag_res = real, imag
            normed_real, normed_imag = self.complex_norms_mid[i]((real, imag))
            # Pass the padding mask (True=MASKED) to the interference layer
            out_real, out_imag = layer((normed_real, normed_imag), attention_mask=memory_padding_mask)
            real = real_res + self.complex_dropout(out_real)
            imag = imag_res + self.complex_dropout(out_imag)
        processed_complex_repr = (real, imag)

        # --- Complex -> Real Projection ---
        processed_real_repr = self.complex_to_real(processed_complex_repr) # [B_valid, max_num_p, decoder_mem_dim]

        # --- Decoding ---
        if target_byte_seq is None: # Generation mode (handled by generate method)
             # Return empty tensor matching batch size
             return torch.zeros((batch_size, 0, 256), device=device)
        else: # Training mode
             # Select target sequences for valid batch items
             valid_target_byte_seq = target_byte_seq[valid_batch_indices] # [B_valid, S_tgt]

             # Decoder needs memory and memory padding mask (True=MASKED)
             byte_logits_valid = self.local_decoder(
                 tgt_byte_seq=valid_target_byte_seq,
                 memory=processed_real_repr, # [B_valid, M, H_mem] where M=max_num_p
                 memory_key_padding_mask=memory_padding_mask # [B_valid, M] -> True where memory is padding
             ) # Output: [B_valid, S_tgt, 256]

             # Reconstruct full batch output (fill invalid items with zeros)
             final_byte_logits = torch.zeros((batch_size, target_byte_seq.size(1), 256), device=device) # Default dtype is float32
             # Use tensor indexing to place valid logits back into the full tensor
             # Cast byte_logits_valid to the dtype of final_byte_logits before assignment to handle AMP mismatch
             final_byte_logits[torch.tensor(valid_batch_indices, device=device)] = byte_logits_valid.to(final_byte_logits.dtype)

             return final_byte_logits # [B, S_tgt, 256]

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        """Computes cross-entropy loss with label smoothing."""
        batch_size, seq_len, vocab_size = logits.size()
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Validate target indices (ensure they are within [0, vocab_size-1])
        if torch.any(targets_flat >= vocab_size) or torch.any(targets_flat < 0):
            invalid_indices = torch.where((targets_flat < 0) | (targets_flat >= vocab_size))[0]
            logger.error(f"Target indices out of bounds (0 <= index < {vocab_size}). Found values like: {targets_flat[invalid_indices[:10]]}")
            # Clamp targets as a temporary fix? Or raise error? Raising is safer.
            raise ValueError(f"Target indices exceed vocabulary size ({vocab_size})!")

        with torch.no_grad():
            true_dist = torch.zeros_like(logits_flat)
            smooth_val = smoothing / (vocab_size - 1) if vocab_size > 1 else 0.0
            true_dist.fill_(smooth_val)
            # Use index_fill_ or scatter_ for potentially better performance? Scatter is fine.
            true_dist.scatter_(1, targets_flat.unsqueeze(1), 1.0 - smoothing)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        # Calculate KL divergence loss
        loss = F.kl_div(log_probs, true_dist, reduction='none').sum(dim=-1) # Sum over vocab dim -> [B*S]

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1) # [B*S]
            loss = loss * mask_flat # Zero out loss for masked positions
            num_active_elements = mask_flat.sum()
            # Calculate mean loss only over non-masked elements
            mean_loss = loss.sum() / num_active_elements if num_active_elements > 0 else torch.tensor(0.0, device=logits.device)
        else:
            # Calculate mean loss over all elements if no mask
            mean_loss = loss.mean()

        return mean_loss

    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None) -> torch.Tensor:
        """Generates byte sequences autoregressively."""
        self.eval()
        device = seed_bytes.device
        batch_size, seed_len = seed_bytes.size()
        generated_sequence = seed_bytes.clone()
        if sampling_config is None:
            sampling_config = SamplerConfig()
        # self.patcher.reset_context() # Optional: Reset patcher state if needed

        for _ in tqdm(range(max_length), desc="Generating", disable=batch_size > 1):
            # Prepare inputs for the model's forward pass
            current_context = generated_sequence
            # Context for patching can be the whole sequence or a window
            context_for_patches = current_context #[:, -self.context_window:] # Use full context for now
            # Decoder input is the sequence generated so far
            decoder_input = current_context

            # Get logits for the next token prediction
            logits_all = self(byte_seq=context_for_patches, target_byte_seq=decoder_input) # [B, current_len, 256]

            # Check if logits are valid (e.g., not all zeros if patching failed)
            if logits_all.shape[1] == 0:
                 logger.warning("Generation stopped: Model returned empty logits (possibly due to patching/encoding failure).")
                 break

            next_byte_logits = logits_all[:, -1, :] # [B, 256]

            # Sampling strategy
            if temperature <= 0: # Greedy decoding
                next_byte_indices = torch.argmax(next_byte_logits, dim=-1) # [B]
            else: # Sampling with temperature and entropy-based strategy
                scaled_logits = next_byte_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1) # [B, 256]
                # Calculate entropy for each sequence in the batch
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1) # [B]

                next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                # Apply sampling strategy per batch item
                for i in range(batch_size):
                    current_entropy = entropy[i].item()
                    current_probs = probs[i] # [256]

                    if current_entropy < sampling_config.low_entropy_threshold:
                        # Low entropy: Use argmax (greedy)
                        next_byte_idx = torch.argmax(current_probs)
                    elif current_entropy < sampling_config.medium_entropy_threshold:
                        # Medium entropy: Top-k sampling
                        top_k = 10
                        actual_k = min(top_k, 256) # Ensure k is not larger than vocab size
                        top_k_probs, top_k_indices = torch.topk(current_probs, k=actual_k)
                        # Renormalize top-k probabilities
                        top_k_probs = top_k_probs / (top_k_probs.sum() + 1e-9)
                        # Sample from the renormalized top-k distribution
                        sampled_relative_idx = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
                        next_byte_idx = top_k_indices[sampled_relative_idx]
                    else: # High entropy: Sample from the full distribution
                        next_byte_idx = torch.multinomial(current_probs, num_samples=1).squeeze(-1)

                    next_byte_indices[i] = next_byte_idx

            # Append the chosen next byte to the sequence
            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)

        return generated_sequence


# =====================================================================
# Trainer Class
# =====================================================================
class Trainer:
    """Trainer class for BSFIN model."""
    # Removed lr_scheduler from init and usage as EnhancedSGD handles LR internally
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], grad_accum_steps: int = 1, use_amp: bool = True, log_interval: int = 10, save_interval: int = 1000, checkpoint_dir: str = "checkpoints", wandb_enabled: bool = False, max_grad_norm: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.lr_scheduler = None # No external scheduler needed with EnhancedSGD
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch, "amp")
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE # Check if wandb was imported
        self.max_grad_norm = max_grad_norm
        self.global_step = 0
        self.current_epoch = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(self.optimizer.gradient_stats, GradientStats)
        logger.info(f"Trainer initialized. AMP: {self.use_amp}, Grad Accum: {self.grad_accum_steps}, Optimizer: {type(self.optimizer).__name__}")

    def _train_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        epoch_loss = 0.0
        steps_in_epoch = 0 # Tracks accumulation steps
        optimizer_steps = 0 # Tracks optimizer steps
        approx_total_batches = -1
        try:
            approx_total_batches = len(self.train_loader)
        except TypeError:
            logger.warning("DataLoader has no __len__. Progress bar may be inaccurate.")

        batch_iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}", disable=(approx_total_batches <= 0), total=approx_total_batches)
        self.optimizer.zero_grad() # Zero gradients at the start

        for i, batch_data in enumerate(batch_iterator):
            current_batch_step = i + 1
            is_accumulating = current_batch_step % self.grad_accum_steps != 0
            is_final_batch = current_batch_step == approx_total_batches if approx_total_batches > 0 else False

            try:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    context, target = batch_data # Expect [B, S_ctx], [B]
                else:
                    logger.warning(f"Skipping unexpected batch format at step {i}.")
                    continue

                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # Prepare inputs for next-byte prediction task
                decoder_input = context # Decoder sees the context
                loss_target = target # Loss compares against the single next byte

                # Forward pass with AMP
                with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    logits = self.model(byte_seq=context, target_byte_seq=decoder_input) # [B, S_ctx, 256]
                    # Extract the logit for the prediction *after* the context sequence
                    last_logit = logits[:, -1, :] # [B, 256]
                    # Compute loss using the single target byte
                    loss = self.model.compute_loss(
                        last_logit.unsqueeze(1), # Reshape to [B, 1, 256]
                        loss_target.unsqueeze(1), # Reshape to [B, 1]
                        smoothing=0.1 # Use label smoothing
                    )
                    # Normalize loss for gradient accumulation
                    loss = loss / self.grad_accum_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                # Detach loss for logging to avoid holding computation graph
                current_step_loss = loss.item() * self.grad_accum_steps # Log un-normalized loss
                epoch_loss += current_step_loss

            except Exception as batch_ex:
                logger.error(f"Error in training step {current_batch_step} (Global {self.global_step}): {batch_ex}", exc_info=True)
                # Skip optimizer step if error occurs during accumulation? Maybe not needed.
                continue # Continue to next batch

            steps_in_epoch += 1 # Count accumulation steps processed

            # --- Optimizer Step ---
            if not is_accumulating or is_final_batch:
                optimizer_steps += 1

                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)

                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Pass loss to optimizer *before* step if needed (like EnhancedSGD)
                if hasattr(self.optimizer, 'current_loss'):
                    self.optimizer.current_loss = current_step_loss

                # Optimizer step
                self.scaler.step(self.optimizer)
                # Update the scaler for next iteration
                self.scaler.update()
                # Zero gradients *after* optimizer step
                self.optimizer.zero_grad(set_to_none=True)

                # Scheduler step AFTER optimizer step - REMOVED as EnhancedSGD handles LR
                # if self.lr_scheduler:
                #     self.lr_scheduler.step()

                self.global_step += 1 # Increment global step counter

                # --- Logging ---
                if self.global_step % self.log_interval == 0:
                    lr_val = self.optimizer.param_groups[0]['lr']
                    log_data = {
                        "train/loss_step": current_step_loss,
                        "train/learning_rate": lr_val,
                        "train/epoch": self.current_epoch + 1,
                        "train/global_step": self.global_step,
                        "train/grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else -1.0
                    }
                    if self.has_grad_stats:
                        opt_stats = self.optimizer.gradient_stats.get_step_stats()
                        log_data.update({f"train/opt/{k}": v for k, v in opt_stats.items()})
                    if hasattr(self.optimizer, 'q_controller'):
                        log_data["train/q_epsilon"] = self.optimizer.q_controller.epsilon
                        # Optionally log chosen Q-actions if needed
                        if self.optimizer.q_controller.prev_action:
                           log_data["train/q_lr_scale"] = self.optimizer.q_controller.prev_action['lr_scale']
                           log_data["train/q_mom_scale"] = self.optimizer.q_controller.prev_action['momentum_scale']

                    log_str = f"Step {self.global_step} | Loss: {log_data['train/loss_step']:.4f} | LR: {lr_val:.6e} | GradNorm: {log_data['train/grad_norm']:.2f}"
                    logger.info(log_str)
                    if self.wandb_enabled:
                        wandb.log(log_data, step=self.global_step)
                    batch_iterator.set_description(f"Epoch {self.current_epoch + 1} (Step {self.global_step}) Loss: {log_data['train/loss_step']:.3f}")

                # --- Checkpointing ---
                if self.global_step % self.save_interval == 0:
                    is_main = not is_initialized() or torch.distributed.get_rank() == 0
                    if is_main:
                        self._save_checkpoint(is_intermediate=True)

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / optimizer_steps if optimizer_steps > 0 else 0.0 # Average loss per optimizer step
        logger.info(f"Epoch {self.current_epoch + 1} Average Training Loss (per step): {avg_epoch_loss:.4f}")
        if self.wandb_enabled:
            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": self.current_epoch + 1}, step=self.global_step)
        if self.has_grad_stats:
            # Record stats for the last optimizer step of the epoch
            final_opt_stats = self.optimizer.gradient_stats.record_step(self.global_step)
            # logger.info(f"Epoch {self.current_epoch + 1} Final Optimizer Step Stats: {final_opt_stats}")

        return avg_epoch_loss


    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Runs validation loop."""
        if self.val_loader is None: return {}
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        val_iterator = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch + 1}")

        for batch_data in val_iterator:
            try:
                context, target = batch_data # Expect [B, S_ctx], [B]
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                batch_size = context.size(0)
                num_samples += batch_size

                # Use AMP context manager for consistency
                with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # Prepare inputs for next-byte prediction
                    decoder_input = context
                    logits = self.model(byte_seq=context, target_byte_seq=decoder_input) # [B, S_ctx, 256]
                    # Extract last logit and compute loss against the single target byte
                    last_logit = logits[:, -1, :] # [B, 256]
                    loss = self.model.compute_loss(
                        last_logit.unsqueeze(1), # [B, 1, 256]
                        target.unsqueeze(1),     # [B, 1]
                        smoothing=0.0 # No label smoothing for validation
                    )

                # Accumulate loss, weighted by batch size
                total_loss += loss.item() * batch_size

            except Exception as val_ex:
                logger.error(f"Validation Error: {val_ex}", exc_info=True)
                continue # Skip batch on error

        # Calculate average validation loss
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        # Calculate perplexity = exp(average_loss)
        perplexity = math.exp(min(avg_loss, 20)) if avg_loss >= 0 else float('inf') # Cap loss for exp, handle potential negative loss?
        metrics = {'val_loss': avg_loss, 'val_perplexity': perplexity}

        logger.info(f"Validation Epoch {self.current_epoch + 1} | Loss: {metrics['val_loss']:.4f} | Perplexity: {metrics['val_perplexity']:.2f}")
        if self.wandb_enabled:
            # Corrected wandb log call using dictionary unpacking
            wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": self.current_epoch + 1}, step=self.global_step)

        return metrics

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict] = None):
        """Saves model checkpoint (only called from main process)."""
        filename = f"checkpoint_step_{self.global_step}.pt" if is_intermediate else f"checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt" # Save epoch completed
        filepath = os.path.join(self.checkpoint_dir, filename)

        model_state = self.model.module.state_dict() if isinstance(self.model, DistributedDataParallel) else self.model.state_dict()
        checkpoint = {
            'epoch': self.current_epoch, # Epoch number completed
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            # 'lr_scheduler_state_dict': None, # Removed scheduler
            'metrics': metrics,
            'amp_enabled': self.use_amp
            # 'q_controller_state': self.optimizer.q_controller.q_table if hasattr(self.optimizer, 'q_controller') else None # Optional: Save Q-table
        }
        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}", exc_info=True)

    def load_checkpoint(self, filepath: str):
        """Loads model checkpoint."""
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return 0 # Return starting epoch 0

        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # Load model state
            model_to_load = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
            # Handle missing/unexpected keys during loading
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible_keys.missing_keys: logger.warning(f"Missing keys when loading model state: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys: logger.warning(f"Unexpected keys when loading model state: {incompatible_keys.unexpected_keys}")

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load GradScaler state
            # Check if AMP state exists and current setting matches checkpoint
            saved_amp_enabled = checkpoint.get('amp_enabled', False)
            if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                if saved_amp_enabled:
                     self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                else:
                     logger.warning("Loading checkpoint without AMP state, but AMP is enabled now. Scaler state not loaded.")
            elif not self.use_amp and saved_amp_enabled:
                 logger.warning("Loading checkpoint with AMP state, but AMP is disabled now.")

            # Load LR scheduler state - REMOVED
            # if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint and checkpoint['lr_scheduler_state_dict']:
            #     self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            # Load training progress (resume from *next* epoch)
            start_epoch = checkpoint.get('epoch', -1) + 1 # Start from epoch after the saved one
            self.global_step = checkpoint.get('global_step', 0)

            # Optionally load Q-table state
            # if hasattr(self.optimizer, 'q_controller') and 'q_controller_state' in checkpoint:
            #     self.optimizer.q_controller.q_table = checkpoint['q_controller_state']
            #     logger.info("Loaded Q-controller state.")

            logger.info(f"Loaded checkpoint from {filepath}. Resuming from Epoch {start_epoch}, Global Step {self.global_step}")
            return start_epoch

        except Exception as e:
            logger.error(f"Failed loading checkpoint '{filepath}': {e}", exc_info=True)
            return 0 # Start from scratch on failure

    def train(self, epochs: int, start_epoch: int = 0):
        """Main training loop."""
        self.current_epoch = start_epoch
        logger.info(f"Starting training from epoch {start_epoch + 1} for {epochs} total epochs.")
        is_main = not is_initialized() or torch.distributed.get_rank() == 0

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            logger.info(f"--- Starting Epoch {epoch + 1}/{epochs} (Global Step: {self.global_step}) ---")

            # Set epoch for distributed sampler
            if isinstance(self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            # Run training epoch
            avg_train_loss = self._train_epoch()

            # Run validation epoch
            val_metrics = self._validate()

            # Save checkpoint at the end of the epoch (only on main process)
            if is_main:
                self._save_checkpoint(is_intermediate=False, metrics=val_metrics)

            # Optional: Add early stopping logic based on val_metrics here

        logger.info("Training finished.")


# =====================================================================
# Argument Parsing and Main Execution
# =====================================================================
# --- Need get_linear_schedule_with_warmup if used ---
# Example implementation (if not importing from transformers):
# def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
#     """ Linear warmup and decay scheduler. """
#     def lr_lambda(current_step: int):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         # Calculate decay factor
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         return max(0.0, 1.0 - progress) # Linear decay from 1 to 0
#
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
# --- End Scheduler ---


def parse_args():
    parser = argparse.ArgumentParser(description="BSFIN Hybrid Model Training (v2)")
    # Data Args
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data numpy file (.npy)")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation data numpy file (.npy)")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use (0.0 to 1.0)")
    # Model Architecture Args
    parser.add_argument("--local_hidden_size", type=int, default=256, help="Hidden size for LocalEncoder/Decoder")
    parser.add_argument("--complex_dim", type=int, default=512, help="Dimension for complex layers")
    parser.add_argument("--num_complex_layers", type=int, default=6, help="Number of SFIN-like layers")
    parser.add_argument("--num_complex_heads", type=int, default=8, help="Number of heads in complex layers")
    parser.add_argument("--decoder_memory_dim", type=int, default=768, help="Real dim projected from complex stack for decoder memory")
    parser.add_argument("--context_window", type=int, default=256, help="Context window size for dataset/generation")
    parser.add_argument("--n_gram_sizes", type=int, nargs='+', default=[3, 4], help="N-gram sizes for patcher/encoder")
    parser.add_argument("--n_gram_vocab_size", type=int, default=30000, help="Vocab size for N-gram hashing")
    parser.add_argument("--sfin_noise_scale", type=float, default=0.05, help="Noise scale in SFIN layers")
    parser.add_argument("--no_entanglement", action="store_true", help="Disable head entanglement in SFIN layers")
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings in SFIN layers")
    parser.add_argument("--projection_method", type=str, default="concat", choices=["concat", "magnitude"], help="Method for Complex->Real projection")
    # Training Args
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="*Initial* learning rate (EnhancedSGD will adapt it)") # Clarified help text
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--checkpoint_dir", type=str, default="./bsfin_checkpoints_v2", help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=50, help="Log training status every N global steps")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save intermediate checkpoint every N global steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    # parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "none"], help="Learning rate scheduler type") # Removed scheduler args
    # parser.add_argument("--warmup_steps", type=int, default=200, help="Number of linear warmup steps for LR scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization)")
    # Misc Args
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (if installed)")
    parser.add_argument("--wandb_project", type=str, default="bsfin-hybrid-v2", help="Wandb project name")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader worker processes")
    # DDP Args
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), help="Local rank for Distributed Data Parallel")

    args = parser.parse_args()
    # Create checkpoint dir if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args

def setup_distributed(local_rank):
    """Initializes distributed training environment."""
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running on single device: {device}")
        return False, device

    # DDP Setup
    if not torch.cuda.is_available() or torch.cuda.device_count() <= local_rank:
        logger.error("Distributed training requested but CUDA is not available or local_rank is invalid.")
        return False, torch.device("cpu") # Fallback to CPU

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Check if environment variables are set (usually by torchrun/launch)
    if "WORLD_SIZE" not in os.environ:
        logger.warning("DDP environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) not set. Assuming single-node setup.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(10000, 20000)) # Random port
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

    try:
        # Initialize the process group
        init_process_group(backend="nccl", init_method="env://")
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        logger.info(f"Distributed Training Initialized. Rank: {rank}/{world_size}, Device: {device}")
        # Synchronize processes before starting training
        if torch.cuda.is_available():
             torch.distributed.barrier()
        return True, device
    except Exception as e:
        logger.error(f"Distributed process group initialization failed: {e}", exc_info=True)
        return False, device # Fallback if init fails


def main():
    args = parse_args()
    # --- Logging Setup ---
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)
    # --- Distributed Setup ---
    is_distributed, device = setup_distributed(args.local_rank)
    is_main_process = not is_distributed or torch.distributed.get_rank() == 0
    # Add FileHandler only on the main process
    if is_main_process:
        log_filename = os.path.join(args.checkpoint_dir, f"bsfin_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        try:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler) # Add handler to the root logger
            logger.info(f"File logging enabled: {log_filename}")
        except Exception as e:
            logger.error(f"Failed to set up file logging: {e}")

    logger.info(f"Effective Device: {device}, Distributed: {is_distributed}, Main Process: {is_main_process}")
    logger.info(f"Run Arguments: {args}")
    # --- Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Optional: torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    # --- Wandb Setup ---
    use_wandb = args.wandb and is_main_process and WANDB_AVAILABLE
    if use_wandb:
        try:
            wandb.init(project=args.wandb_project, config=args)
            logger.info("Wandb Initialized")
        except Exception as e:
            logger.warning(f"Wandb init failed: {e}. Disabling.")
            use_wandb = False
    # --- Dataset and DataLoader ---
    try:
        logger.info(f"Loading training data from: {args.data_path}")
        train_dataset = ByteIterableDataset(args.data_path, context_size=args.context_window, data_fraction=args.data_fraction)
        val_dataset = None
        if args.val_data_path:
            logger.info(f"Loading validation data from: {args.val_data_path}")
            val_dataset = ByteIterableDataset(args.val_data_path, context_size=args.context_window, data_fraction=1.0) # Use full val set
        else:
            logger.info("No validation data path provided.")
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}", exc_info=True)
        sys.exit(1)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, seed=args.seed, drop_last=True) if is_distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if is_distributed and val_dataset else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False) if val_dataset else None
    # --- Model Initialization ---
    model_config = {
        "local_hidden_size": args.local_hidden_size, "complex_dim": args.complex_dim,
        "num_complex_layers": args.num_complex_layers, "num_complex_heads": args.num_complex_heads,
        "decoder_memory_dim": args.decoder_memory_dim, "dropout": 0.15, # Using fixed dropout for now
        "context_window": args.context_window, "n_gram_sizes": args.n_gram_sizes,
        "n_gram_vocab_size": args.n_gram_vocab_size, "sfin_noise_scale": args.sfin_noise_scale,
        "sfin_use_entanglement": not args.no_entanglement, "sfin_use_rotary": not args.no_rope,
        "projection_method": args.projection_method
    }
    model = BSFINModel(**model_config).to(device)

    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model Params: Total={total_params:,}, Trainable={trainable_params:,}")

    # Wrap model with DDP if distributed
    if is_distributed:
        # find_unused_parameters can help debug DDP issues but adds overhead
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        logger.info("Model wrapped with DistributedDataParallel.")

    # --- Optimizer ---
    q_config = {
        "learning_rate": 0.01, # Q-agent's own learning rate
        "discount": 0.95,
        "epsilon": 0.2,
        "lr_scale_bounds": (0.95, 1.05), # Tighter bounds for LR scaling factor
        "momentum_scale_bounds": (0.98, 1.02) # Slightly tighter momentum bounds
    }
    optimizer = EnhancedSGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, # Initial LR from args
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        q_learning_config=q_config # Pass the modified config
    )
    logger.info(f"Using Optimizer: {type(optimizer).__name__}")

    # --- LR Scheduler (REMOVED - Handled by EnhancedSGD) ---
    lr_scheduler = None
    logger.info("No external LR scheduler used (EnhancedSGD handles LR adaptation).")

    # --- Trainer Initialization ---
    trainer = Trainer(
        model=model, optimizer=optimizer, device=device, train_loader=train_loader,
        val_loader=val_loader, 
        grad_accum_steps=args.grad_accum_steps,
        use_amp=not args.no_amp, log_interval=args.log_interval, save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir, wandb_enabled=use_wandb, max_grad_norm=args.max_grad_norm
    )

    # --- Resume Checkpoint ---
    start_epoch = 0
    if args.resume:
        logger.info(f"Attempting to resume from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        # TODO: Potentially advance scheduler state based on loaded global_step if NOT using EnhancedSGD

    # --- Start Training ---
    save_final = False # Flag to save checkpoint in finally block
    try:
        trainer.train(args.epochs, start_epoch=start_epoch)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user (KeyboardInterrupt).")
        save_final = True
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        save_final = True
    finally:
        # Save final checkpoint on main process if interrupted or error occurred
        if save_final and is_main_process:
            logger.info("Saving final checkpoint...")
            trainer._save_checkpoint(is_intermediate=False)
        # Clean up DDP
        if is_distributed:
            destroy_process_group()
            logger.info("Distributed process group destroyed.")
        # Finish Wandb run
        if use_wandb and wandb.run:
            wandb.finish()
            logger.info("Wandb run finished.")

    logger.info("BSFIN training script finished.")

if __name__ == "__main__":
    main()
