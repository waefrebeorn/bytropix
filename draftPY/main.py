#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bytropix - Byte Latent Transformer with Babylon Index and Q-Learning Optimization
(EnhancedSGD Stabilized)
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
from tqdm import tqdm
# Try importing wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # Set wandb to None if not available
    WANDB_AVAILABLE = False
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque
import gc
import socket
import platform
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group, is_initialized
from torch import amp
from dataclasses import dataclass
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =====================================================================
# Data Structures and Configuration Classes
# =====================================================================

@dataclass
class SamplerConfig:
    """Configuration for entropy-based sampling."""
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# =====================================================================
# ByteTokenizer and ByteIterableDataset
# =====================================================================

class ByteTokenizer:
    """Simple tokenizer for byte-level processing."""

    def encode(self, text: str) -> List[int]:
        """Convert text to bytes."""
        return [b for b in text.encode('utf-8')]

    def decode(self, byte_sequence: Iterable[int]) -> str:
        """Convert bytes back to text."""
        # Ensure bytes are valid integers in the 0-255 range before decoding
        valid_bytes = []
        for b in byte_sequence:
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
        """
        Initialize the dataset.

        Args:
            npy_file_path: Path to numpy array file containing byte data
            context_size: Size of context window
            data_fraction: Fraction of data to use (0.0 to 1.0)
        """
        self.npy_file_path = npy_file_path
        self.context_size = context_size
        self.data_fraction = max(0.0, min(1.0, data_fraction))

        # Validate file exists
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"File not found: {npy_file_path}")

        try:
            # Load memmap object to get shape, then delete it
            mmap_data = np.load(self.npy_file_path, mmap_mode='r')
            self.full_data_size = mmap_data.shape[0]
            del mmap_data # Explicitly delete to close the memmap handle
            gc.collect() # Suggest garbage collection

            self.data_size = int(self.full_data_size * self.data_fraction)
            if self.data_size <= self.context_size: # Need at least context_size + 1 bytes
                raise ValueError(f"Dataset size after fraction ({self.data_size}) is too small for context size ({self.context_size}). Needs at least {self.context_size + 1} bytes.")
            logging.info(f"Using {self.data_size}/{self.full_data_size} bytes ({self.data_fraction:.1%}) from {npy_file_path}")
        except Exception as e:
            logging.error(f"Error accessing or checking size of {npy_file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        """Return the number of possible starting indices for a context+target sequence."""
        return max(0, self.data_size - self.context_size)

    def __iter__(self):
        """Generate (context, target) pairs from byte data."""
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        # Load data within the worker process using memmap
        bytes_data = None # Initialize to prevent UnboundLocalError in finally
        try:
            # Keep the memmap object open for the duration of iteration
            bytes_data = np.load(self.npy_file_path, mmap_mode='r')
        except Exception as e:
            logging.error(f"Worker {worker_id}: Failed to load {self.npy_file_path}: {e}", exc_info=True)
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
            seed = (worker_id + int(datetime.now().timestamp() * 1000) + os.getpid()) % (2**32)
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
                        logging.warning(f"Worker {worker_id}: IndexError accessing data at index {idx}. Effective len: {effective_data_len}. Skipping.")
                        continue
                # else: # This condition should ideally not be hit due to range calculation
                #    logging.debug(f"Worker {worker_id}: Index {idx} out of bounds. Skipping.")
        finally:
            # Clean up the mmap object when the iterator is exhausted or an error occurs
            if bytes_data is not None:
                 # Check if it's a memmap object before deleting attributes
                 if hasattr(bytes_data, '_mmap'):
                      # Ensure the mmap object exists before trying to close
                      if bytes_data._mmap is not None:
                           try:
                               bytes_data._mmap.close()
                           except Exception as close_ex:
                               logging.warning(f"Worker {worker_id}: Error closing memmap: {close_ex}")
                 del bytes_data
            gc.collect() # Explicitly request garbage collection


# =====================================================================
# Babylon Index for Byte Sequence Analysis
# =====================================================================

class BabylonIndex:
    """Entropy-based index for byte sequence analysis with efficient window management."""

    def __init__(
        self,
        scales: List[int] = [3, 5, 7],
        max_cache_size: int = 100000,
        min_entropy_threshold: float = 0.3
    ):
        self.scales = sorted(list(set(scales))) # Ensure unique and sorted scales
        self.hash_index = {}  # Maps hash values to window positions (Not used in current find_patch_boundaries)
        self.entropy_cache = {}  # Cache for entropy values
        self.max_cache_size = max_cache_size
        self.min_entropy_threshold = min_entropy_threshold

        # --- Removed unused base_powers and rolling_hash ---

    def _clean_cache(self):
        """Clean entropy cache if it exceeds max size."""
        if len(self.entropy_cache) > self.max_cache_size:
            # Remove 20% of oldest entries
            remove_count = len(self.entropy_cache) - (self.max_cache_size * 4 // 5) # Keep 80%
            # Use iterator to avoid creating a potentially large list of keys
            keys_to_remove = list(itertools.islice(self.entropy_cache.keys(), remove_count))
            for k in keys_to_remove:
                 # Check if key still exists before deleting
                 if k in self.entropy_cache:
                     del self.entropy_cache[k]

    def _is_valid_utf8_boundary(self, byte_seq: Union[List[int], np.ndarray], boundary: int) -> bool:
        """Checks if a potential boundary is valid (not mid-UTF8 char)."""
        if boundary <= 0 or boundary >= len(byte_seq):
            return True
        byte_at_boundary = byte_seq[boundary]
        # Continuation bytes are in the range 0x80 (10000000) to 0xBF (10111111)
        return not (0x80 <= byte_at_boundary <= 0xBF)

    def compute_entropy(self, byte_window: Union[np.ndarray, Tuple[int, ...]]) -> float:
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
             logging.warning(f"Unsupported type for compute_entropy: {type(byte_window)}")
             return 0.0

        try:
            # Ensure input is integer type for bincount
            if not np.issubdtype(byte_window_np.dtype, np.integer):
                 logging.warning(f"Non-integer dtype {byte_window_np.dtype} passed to bincount. Attempting cast.")
                 byte_window_np = byte_window_np.astype(np.uint8)
            # Ensure values are within the valid byte range [0, 255] before bincount
            if np.any(byte_window_np < 0) or np.any(byte_window_np > 255):
                 logging.warning(f"Invalid byte values detected (outside 0-255). Clamping.")
                 byte_window_np = np.clip(byte_window_np, 0, 255)

            byte_counts = np.bincount(byte_window_np, minlength=256)
        except TypeError as e:
             logging.error(f"TypeError in np.bincount. Input type: {type(byte_window_np)}, dtype: {byte_window_np.dtype}. Error: {e}")
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

    # --- Removed unused get_window_features and prioritize methods ---

    def find_patch_boundaries(self, byte_seq_tensor: torch.Tensor) -> List[int]:
        """Find patch boundaries based on entropy minima/maxima and UTF-8 validity."""
        if byte_seq_tensor.numel() == 0: return []

        # Ensure input is 1D list of ints for processing
        if byte_seq_tensor.dim() > 1:
            if byte_seq_tensor.size(0) == 1: # Handle batch size 1
                 byte_seq_list = byte_seq_tensor[0].cpu().tolist()
            else:
                 logging.warning("find_patch_boundaries expects 1D tensor or batch size 1. Using first element.")
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
        num_boundaries_target = max(1, seq_len // 128) # Adjust divisor as needed
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
            # Ensure the last segment isn't too small either
            if seq_len - last_boundary < min_patch_size and merged_boundaries:
                 # Remove the last boundary to merge the last two segments
                 merged_boundaries.pop()

            final_boundaries = merged_boundaries

        # logging.debug(f"Found boundaries: {final_boundaries} for seq_len {seq_len}")
        return final_boundaries

    def create_patches(self, byte_seq: torch.Tensor) -> List[torch.Tensor]:
        """Convert byte sequence into patches based on entropy boundaries."""
        if byte_seq.dim() != 2 or byte_seq.size(0) != 1:
             raise ValueError(f"create_patches expects a tensor of shape [1, seq_len], got {byte_seq.shape}")

        boundaries = self.find_patch_boundaries(byte_seq)
        patches = []
        start_idx = 0
        seq_len = byte_seq.size(1)

        for end_idx in boundaries:
            # Ensure boundary is valid and creates a non-empty patch
            if start_idx < end_idx <= seq_len:
                patch = byte_seq[:, start_idx:end_idx] # Keep batch dim
                if patch.numel() > 0: patches.append(patch)
                start_idx = end_idx
            elif end_idx <= start_idx:
                logging.warning(f"Skipping invalid or out-of-order boundary {end_idx} <= {start_idx}")
            elif end_idx > seq_len:
                 logging.warning(f"Boundary {end_idx} exceeds sequence length {seq_len}. Ignoring.")

        # Add the final patch from the last boundary to the end
        if start_idx < seq_len:
            final_patch = byte_seq[:, start_idx:] # Keep batch dim
            if final_patch.numel() > 0:
                patches.append(final_patch)

        # logging.debug(f"Created {len(patches)} patches.")
        return patches

    @torch.no_grad()
    def reset_context(self):
        """Reset context when starting new document/segment."""
        # Clear any stateful buffers like entropy cache
        self.entropy_cache = {}
        self.hash_index = {} # Reset hash index too if used
        # logging.debug("BabylonIndex context reset.")


# =====================================================================
# Q-Learning Controller for Adaptive Optimization
# =====================================================================

class GradientStats:
    """Tracks gradient statistics for reporting."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_gradients = 0
        self.clipped_gradients = 0
        self.max_gradient_norm = 0.0 # Use float
        self.sum_clip_ratios = 0.0 # Use float
        self.step_stats = {}

    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        self.total_gradients += 1
        self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
        if clipped:
            self.clipped_gradients += 1
            self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)

    def get_step_stats(self) -> dict:
        if self.total_gradients == 0:
            return {
                "gradients_clipped": 0,
                "total_gradients": 0,
                "clip_ratio_avg": 0.0, # Renamed for clarity
                "max_gradient": 0.0,
                "clip_percentage": 0.0 # Added percentage
            }

        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0

        return {
            "gradients_clipped": self.clipped_gradients,
            "total_gradients": self.total_gradients,
            "clip_ratio_avg": avg_clip_ratio, # Average ratio when clipping occurred
            "max_gradient": self.max_gradient_norm,
            "clip_percentage": clip_percentage
        }

    def record_step(self, step: int):
        stats = self.get_step_stats()
        self.step_stats[step] = stats
        self.reset() # Reset stats after recording for the step
        return stats


class QController:
    """Q-Learning Controller with adaptive exploration and state management."""

    def __init__(
        self,
        learning_rate: float = 0.02,
        discount: float = 0.97,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.9995, # Slower decay
        min_epsilon: float = 0.02, # Minimum epsilon
        initial_mix_prob: float = 0.9, # Not used in current choose_action
        lr_scale_bounds: tuple = (0.85, 1.15),
        momentum_scale_bounds: tuple = (0.9, 1.1),
        min_weight_decay: float = 1e-4, # Not used in current logic
        state_memory_size: int = 300,
        max_q_table_size: int = 15000
    ):
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon # Store min epsilon
        self.epsilon_decay = epsilon_decay
        # self.mix_prob = initial_mix_prob # Removed as unused
        self.prev_loss = None
        self.prev_state = None
        self.prev_action = None
        self.lr_scale_bounds = lr_scale_bounds # Used for clipping Q-action application
        self.momentum_scale_bounds = momentum_scale_bounds # Used for clipping Q-action application
        # self.min_weight_decay = min_weight_decay # Removed as unused

        # Enhanced state tracking
        self.state_memory = deque(maxlen=state_memory_size) # Stores (state, action, reward) tuples
        self.loss_window = deque(maxlen=15)
        self.grad_var_window = deque(maxlen=15) # Renamed from grad_window for clarity
        self.lr_window = deque(maxlen=8)
        self.momentum_window = deque(maxlen=8)

        # Action space
        self.action_ranges = {
            'lr_scale': np.array([0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.15]), # Refined steps
            'momentum_scale': np.array([0.9, 0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05, 1.1]) # Refined steps
        }

        # Success tracking with decay (Removed as choose_action logic simplified)
        # self.action_success = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        # self.action_counts = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        # self.success_decay = 0.99

        # Performance tracking
        self.performance_window = deque(maxlen=50) # Tracks rewards
        self.stable_steps = 0

        # Q-Table memory management
        self.max_q_table_size = max_q_table_size
        self.q_table_access_count = {} # Track access for pruning

    def get_state(self, lr: float, momentum: float, grad_var: float, loss: float) -> tuple:
        """Simplified state representation focusing on key trends."""
        self.loss_window.append(loss)
        self.grad_var_window.append(grad_var) # Use grad_var
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)

        # Loss trend binning (simplified)
        loss_trend_bin = 2 # Default: stable
        if len(self.loss_window) >= 5:
            y = np.array(list(self.loss_window)[-5:])
            x = np.arange(len(y))
            try:
                # Check for constant y values to avoid LinAlgError
                if np.all(y == y[0]):
                     slope = 0.0
                else:
                     slope = np.polyfit(x, y, 1)[0]

                # Check if slope is finite
                if np.isfinite(slope):
                     # Normalize slope by recent average loss (add epsilon for stability)
                     avg_loss = np.mean(y) + 1e-6
                     normalized_slope = slope / avg_loss
                     loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05])
                # else: stay with default bin
            except (np.linalg.LinAlgError, ValueError):
                # Handle potential errors during polyfit (e.g., if loss becomes NaN/Inf)
                loss_trend_bin = 2 # Default to stable if polyfit fails

        # Gradient variance binning
        grad_var_bin = 2 # Default: medium variance
        if self.grad_var_window:
            # Use median to be more robust to outliers
            median_grad_var = np.median(list(self.grad_var_window))
            if np.isfinite(median_grad_var): # Check if median is finite
                 grad_var_bin = np.digitize(median_grad_var, bins=[1e-5, 1e-3, 0.1, 1.0]) # Adjusted bins
            # else: stay with default bin

        # Learning rate binning (log scale might be better)
        lr_bin = np.digitize(lr, bins=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

        # Momentum binning
        momentum_bin = np.digitize(momentum, bins=[0.85, 0.9, 0.95, 0.99])

        state = (loss_trend_bin, grad_var_bin, lr_bin, momentum_bin)
        # Track state access for pruning
        self.q_table_access_count[state] = self.q_table_access_count.get(state, 0) + 1
        return state

    def compute_reward(
        self,
        loss_improvement: float, # Relative improvement: (prev_loss - current_loss) / prev_loss
        grad_health: float, # Inverse relationship with variance: 1 / (1 + grad_var)
        consistent_improvement: bool # Flag indicating recent positive trend
    ) -> float:
        """Simplified reward computation."""
        # Base reward: positive for improvement, negative for worsening
        # Scale improvement reward more aggressively
        base_reward = 2.0 * loss_improvement if loss_improvement > 0 else 1.0 * loss_improvement

        # Penalty for poor gradient health (low grad_health means high variance)
        # Penalize more if health is very low
        health_penalty = 0.0
        if grad_health < 0.5: # Threshold for penalty (adjust as needed)
             health_penalty = -0.3 * (1.0 - grad_health / 0.5) # Penalty increases as health drops below 0.5

        # Bonus for consistent improvement streak
        consistency_bonus = 0.0
        if consistent_improvement:
            self.stable_steps += 1
            consistency_bonus = min(0.3, 0.05 * math.log1p(self.stable_steps)) # Capped log bonus
        else:
            self.stable_steps = 0 # Reset streak

        # Combine rewards
        total_reward = base_reward + health_penalty + consistency_bonus

        # Clip reward to a reasonable range
        final_reward = float(np.clip(total_reward, -1.0, 1.0))

        # Track performance
        self.performance_window.append(final_reward)

        return final_reward

    def choose_action(self, state: tuple) -> Dict[str, float]:
        """Simplified epsilon-greedy action selection."""
        if state not in self.q_table:
            self.q_table[state] = {
                param: np.zeros(len(space))
                for param, space in self.action_ranges.items()
            }
            self._manage_q_table_size() # Prune if needed when adding new state

        action = {}

        # Decay epsilon based on overall performance trend (optional refinement)
        # Could check self.performance_window average here if desired

        # Simple epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        for param, space in self.action_ranges.items():
            if random.random() < self.epsilon:
                # Exploration: Choose a random action
                chosen_idx = random.randrange(len(space))
            else:
                # Exploitation: Choose the action with the highest Q-value
                q_values = self.q_table[state][param]
                # Handle cases where q_values might be empty or all NaN/Inf (though unlikely with init)
                if len(q_values) > 0 and np.any(np.isfinite(q_values)):
                     max_q = np.nanmax(q_values) # Use nanmax to handle potential NaNs
                     # Find all indices with Q-value close to maximum
                     best_indices = np.where(np.abs(q_values - max_q) < 1e-6)[0]
                     if len(best_indices) > 0:
                          # Break ties by choosing randomly among the best
                          chosen_idx = np.random.choice(best_indices)
                     else: # Fallback if all are NaN or some other issue
                          chosen_idx = random.randrange(len(space))
                else: # Fallback if q_values is empty or all non-finite
                     chosen_idx = random.randrange(len(space))

            action[param] = float(space[chosen_idx])

        return action

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: Optional[tuple], should_log: bool = False):
        """Standard Q-learning update."""
        # Record experience (optional, mainly for debugging or more complex replay)
        # self.state_memory.append((state, action, reward))
        # if should_log:
        #     logging.info(f"State memory updated with state={state}, action={action}, reward={reward}")

        # Initialize Q-values for next_state if it's new
        if next_state is not None and next_state not in self.q_table:
            self.q_table[next_state] = {
                param: np.zeros(len(space))
                for param, space in self.action_ranges.items()
            }
            self._manage_q_table_size() # Prune if needed

        # Update Q-values for the taken action in the current state
        for param, value in action.items():
            space = self.action_ranges[param]
            try:
                # Find the index of the action value in the discrete space
                action_idx = np.abs(space - value).argmin()
                # Verify that the found index actually corresponds to the value (within tolerance)
                if not np.isclose(space[action_idx], value):
                     raise ValueError("Action value not found precisely in space.")
            except ValueError:
                logging.warning(f"Q-update: Action value {value} not found for {param}. Skipping update for this param.")
                continue

            # Get max future Q-value for the next state
            max_future_q = 0.0
            if next_state is not None and next_state in self.q_table:
                next_q_values = self.q_table[next_state][param]
                # Ensure next_q_values is not empty and contains finite values before taking max
                if len(next_q_values) > 0 and np.any(np.isfinite(next_q_values)):
                     max_future_q = np.nanmax(next_q_values)

            # Current Q-value
            current_q = self.q_table[state][param][action_idx]

            # Q-learning update rule: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_q(s') - Q(s,a))
            td_target = reward + self.gamma * max_future_q
            td_error = td_target - current_q

            # Update Q-value
            self.q_table[state][param][action_idx] += self.alpha * td_error

            # Logging (optional)
            # if should_log:
            #     logging.info(f"Updated Q-table for state={state}, param={param}, action_idx={action_idx}: "
            #                  f"current_q={current_q:.4f}, max_future_q={max_future_q:.4f}, td_error={td_error:.4f}, "
            #                  f"reward={reward:.4f}")

    def _manage_q_table_size(self):
        """Prunes the Q-table if it exceeds the maximum size by removing least accessed states."""
        if len(self.q_table) > self.max_q_table_size:
            try:
                # Ensure access count dict is not empty before finding min
                if self.q_table_access_count:
                     # Sort states by access count (ascending)
                     sorted_states = sorted(self.q_table_access_count.items(), key=lambda item: item[1])
                     # Determine number of states to remove
                     num_to_remove = len(self.q_table) - int(self.max_q_table_size * 0.9) # Remove down to 90%
                     num_removed = 0
                     for state, count in sorted_states:
                         if num_removed >= num_to_remove:
                             break
                         # Check if state exists before deleting (could be removed concurrently?)
                         if state in self.q_table:
                              del self.q_table[state]
                         if state in self.q_table_access_count:
                              del self.q_table_access_count[state]
                         num_removed += 1
                     logging.info(f"Pruned {num_removed} states from Q-table (new size: {len(self.q_table)}).")

                # else: logging.debug("Q-table access count is empty, cannot prune.")
            except (ValueError, KeyError) as e:
                logging.warning(f"Could not prune Q-table: {e}")


class EnhancedSGD(torch.optim.Optimizer):
    """Enhanced SGD Optimizer with Q-Learning based Adaptive Hyperparameter Tuning and Stability Fixes."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.003,
        momentum: float = 0.9,
        weight_decay: float = 0.005,
        # smoothing_factor: float = 0.05, # Removed unused args
        # entropy_threshold: float = 0.3,
        max_grad_norm: float = 1.0,
        # noise_scale: float = 0.001,
        lr_scale_bounds: tuple = (0.85, 1.15), # Bounds for Q-action clipping
        momentum_scale_bounds: tuple = (0.9, 1.1), # Bounds for Q-action clipping
        q_learning_config: Dict[str, Any] = {},
        **kwargs # Catch any other potential args
    ):
        """
        Initializes the EnhancedSGD optimizer.
        """
        # Ensure params is properly formatted as parameter groups
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups = params
        else:
            # Ensure params is iterable
            if not isinstance(params, Iterable):
                 params = list(params)
            param_groups = [{'params': params}]

        # Enhanced parameter group initialization
        for group in param_groups:
            group.setdefault('lr', lr)
            group.setdefault('momentum', momentum)
            group.setdefault('weight_decay', weight_decay)
            group.setdefault('base_lr', lr) # Store initial LR
            group.setdefault('q_scale', 1.0) # Initialize cumulative scale factor
            # Add minimum weight decay (not currently used but could be added)
            # group.setdefault('min_weight_decay', weight_decay * 0.2)

        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

        # Initialize optimization state
        self._init_optimization_state(
            # smoothing_factor=smoothing_factor, # Removed
            # entropy_threshold=entropy_threshold, # Removed
            max_grad_norm=max_grad_norm,
            # noise_scale=noise_scale, # Removed
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            q_learning_config=q_learning_config
            # **kwargs # Pass any extra args if needed
        )

        # Pre-allocate buffers with proper device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Removed unused self.stats deque dictionary

    def _init_optimization_state(self, **kwargs):
        """Initialize optimization state with safe handling."""
        # smoothing_factor = kwargs.get('smoothing_factor', 0.05) # Removed
        # entropy_threshold = kwargs.get('entropy_threshold', 0.3) # Removed
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0) # Store max_grad_norm
        # noise_scale = kwargs.get('noise_scale', 0.001) # Removed
        lr_scale_bounds = kwargs.get('lr_scale_bounds', (0.85, 1.15))
        momentum_scale_bounds = kwargs.get('momentum_scale_bounds', (0.9, 1.1))
        q_learning_config = kwargs.get('q_learning_config', {})

        self.q_controller = QController(
            learning_rate=q_learning_config.get('learning_rate', 0.02),
            discount=q_learning_config.get('discount', 0.97),
            epsilon=q_learning_config.get('epsilon', 0.15),
            epsilon_decay=q_learning_config.get('epsilon_decay', 0.9995), # Use updated value
            min_epsilon=q_learning_config.get('min_epsilon', 0.02), # Use updated value
            # initial_mix_prob=q_learning_config.get('initial_mix_prob', 0.9), # Removed
            lr_scale_bounds=lr_scale_bounds, # Pass bounds to QController if needed, though mainly used here
            momentum_scale_bounds=momentum_scale_bounds, # Pass bounds to QController if needed
            # min_weight_decay=q_learning_config.get('min_weight_decay', 1e-4) # Removed
            max_q_table_size=q_learning_config.get('max_q_table_size', 15000) # Pass max size
        )

        self._step_count = 0
        # Removed self.prev_state, self.prev_action, self.prev_loss (managed in QController)
        # Removed self.grad_memory

        # Initialize GradientStats
        self.gradient_stats = GradientStats()

    # Removed _track_gradient_memory, _compute_entropy, get_statistics

    # --- STABILITY FIX 1: Robust Gradient Stats Calculation ---
    def _get_gradient_stats(self) -> Dict[str, Any]:
        """Gather gradient statistics for the current step, checking for finite values."""
        grad_norms = []
        grad_vars = []
        num_finite_params = 0
        num_non_finite_params = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach() # Detach to avoid modifying original grad

                # Check if gradient is finite BEFORE calculations
                if torch.isfinite(grad).all():
                    grad_float = grad.float() # Ensure float for variance calculation
                    grad_norms.append(torch.norm(grad_float).item())
                    if grad_float.numel() > 1:
                        grad_vars.append(torch.var(grad_float).item())
                    num_finite_params += 1
                else:
                    # Log non-finite gradients (optional, can be noisy)
                    # logging.warning(f"Non-finite gradient detected for parameter (shape: {p.shape}). Skipping stats.")
                    num_non_finite_params += 1

        saw_grads = num_finite_params + num_non_finite_params > 0
        saw_finite_grads = num_finite_params > 0

        if saw_finite_grads:
            # Use np.nanmean/np.nanvar if you expect NaNs within finite stats, otherwise mean is fine
            mean_grad_norm = np.mean(grad_norms)
            # Ensure grad_vars is not empty before calculating mean
            mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
        else:
            mean_grad_norm = 0.0
            mean_grad_var = 0.0 # Or potentially float('nan') or float('inf') if needed

        # Check if the calculated means are finite
        is_norm_finite = np.isfinite(mean_grad_norm)
        is_var_finite = np.isfinite(mean_grad_var)

        grad_stats = {
            'saw_grads': saw_grads, # Indicates if *any* grads were seen
            'saw_finite_grads': saw_finite_grads, # Indicates if *finite* grads were seen
            'mean_grad_norm': mean_grad_norm if is_norm_finite else 0.0, # Return 0 if non-finite
            'mean_grad_var': mean_grad_var if is_var_finite else float('inf'), # Return Inf if non-finite? Or 0? Needs careful consideration based on usage. Let's use Inf for now to signal issues clearly downstream.
            'num_non_finite_params': num_non_finite_params, # Track how many params had bad grads
            'is_norm_finite': is_norm_finite,
            'is_var_finite': is_var_finite,
        }
        return grad_stats

    # --- STABILITY FIX 2: Updated Step Method with LR Clamping and Safe Q-Update ---
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """Optimizes the parameters based on the current gradient with stability fixes."""
        loss = None
        if closure is not None:
            try:
                loss = closure()
            except Exception as e:
                logging.error(f"Error in closure execution: {e}", exc_info=True)
                return None # Stop step if closure fails

        # Get gradient statistics (now with finiteness checks)
        grad_stats = self._get_gradient_stats()

        # Log if non-finite gradients were encountered
        if grad_stats['num_non_finite_params'] > 0:
             logging.warning(f"Step {self._step_count}: Encountered non-finite gradients in {grad_stats['num_non_finite_params']} parameters.")

        # Apply Q-learning adjustments only if we have *finite* gradients and loss
        if grad_stats['saw_finite_grads'] and loss is not None and torch.isfinite(loss):
            current_loss = loss.item()
            # Ensure stats passed to Q-controller are reasonable, even if some grads were non-finite
            # Use the mean_grad_var calculated from *finite* gradients.
            # If mean_grad_var itself became non-finite (e.g., due to overflow despite finite inputs), handle it.
            safe_grad_var = grad_stats['mean_grad_var'] if grad_stats['is_var_finite'] else 0.0 # Use 0 if variance calculation failed

            # Get current state using safe grad var
            q_state = self.q_controller.get_state(
                lr=self.param_groups[0]['lr'],
                momentum=self.param_groups[0]['momentum'],
                grad_var=safe_grad_var, # Pass the safe variance
                loss=current_loss
            )

            # Update Q-table with previous experience if available
            if self.q_controller.prev_loss is not None and \
               self.q_controller.prev_state is not None and \
               self.q_controller.prev_action is not None:

                # Ensure previous loss was finite for reward calculation
                if np.isfinite(self.q_controller.prev_loss) and abs(self.q_controller.prev_loss) > 1e-9: # Avoid division by zero
                    # Calculate relative loss improvement safely
                    loss_improvement = (self.q_controller.prev_loss - current_loss) / abs(self.q_controller.prev_loss + 1e-9)
                    # Calculate grad health safely (ensure safe_grad_var >= 0)
                    grad_health = 1.0 / (1.0 + max(0, safe_grad_var))

                    # Check for consistent improvement
                    self.q_controller.performance_window.append(loss_improvement)
                    # Check if most recent rewards are non-negative (or small negative)
                    consistent_improvement = all([r > -0.01 for r in list(self.q_controller.performance_window)[-10:]])

                    # Compute reward for the previous action
                    reward = self.q_controller.compute_reward(
                        loss_improvement=loss_improvement, # Pass relative improvement
                        grad_health=grad_health,
                        consistent_improvement=consistent_improvement
                    )

                    # Update Q-table only if reward is finite
                    if np.isfinite(reward):
                        self.q_controller.update(
                            state=self.q_controller.prev_state,
                            action=self.q_controller.prev_action,
                            reward=reward,
                            next_state=q_state,
                            should_log=(self._step_count % 10 == 0) # Log Q-update periodically
                        )
                    else:
                        logging.warning(f"Step {self._step_count}: Skipping Q-update due to non-finite reward.")

                else:
                     logging.warning(f"Step {self._step_count}: Skipping Q-update due to non-finite or zero previous loss.")


            # Choose new action based on the current state
            q_action = self.q_controller.choose_action(q_state)

            # Apply learning rate and momentum adjustments
            for group in self.param_groups:
                # Update cumulative scale factor
                group['q_scale'] *= float(np.clip(
                    q_action['lr_scale'],
                    self.q_controller.lr_scale_bounds[0],
                    self.q_controller.lr_scale_bounds[1]
                ))
                # --- Absolute LR Clamping ---
                # Clamp the final LR, not just the scale factor
                min_lr = 1e-7 # Define minimum LR
                max_lr = 0.01 # Define maximum LR (adjust as needed)
                group['lr'] = float(np.clip(group['base_lr'] * group['q_scale'], min_lr, max_lr))
                # --- End Absolute LR Clamping ---

                # Scale momentum (already clipped in bounds)
                group['momentum'] = float(np.clip(
                    group['momentum'] * q_action['momentum_scale'], # Apply scale to current momentum
                    self.q_controller.momentum_scale_bounds[0],
                    self.q_controller.momentum_scale_bounds[1]
                ))

            # Update Q-learning state for the next step
            self.q_controller.prev_state = q_state
            self.q_controller.prev_action = q_action
            self.q_controller.prev_loss = current_loss
        # --- End Q-Learning Section ---

        # Apply updates using the potentially modified learning rates
        num_params_updated = 0
        for group in self.param_groups:
            lr = group['lr'] # Use the final clamped LR
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Get state for this parameter
                state = self.state[p]

                # Apply update (pass necessary args, _apply_update now handles grad checks)
                updated = self._apply_update(
                    p=p,
                    grad_in=p.grad, # Pass original grad
                    momentum=momentum,
                    lr=lr,
                    weight_decay=weight_decay,
                    state=state,
                    current_loss=self.q_controller.prev_loss if self.q_controller.prev_loss is not None else 0.0
                )
                if updated:
                    num_params_updated += 1

        if num_params_updated > 0: # Only increment step if updates were applied
             self._step_count += 1
             # Record and log gradient clipping statistics for the step
             stats = self.gradient_stats.record_step(self._step_count) # Record stats *after* applying updates
             if self._step_count % 10 == 0: # Log less frequently maybe
                 logging.info(
                     f"Step {self._step_count} grad stats: "
                     f"Clipped {stats['gradients_clipped']}/{stats['total_gradients']} "
                     f"({stats['clip_percentage']:.1%}). " # Use percentage
                     f"Max norm: {stats['max_gradient']:.3f}, "
                     f"Avg clip ratio: {stats['clip_ratio_avg']:.3f}" # Use avg ratio
                 )
        elif grad_stats['saw_grads']:
             logging.warning(f"Step {self._step_count}: Gradients present but no parameters updated (likely due to non-finite grads).")


        return loss

    # --- STABILITY FIX 3: Safer Parameter Update ---
    def _apply_update(
        self,
        p: torch.Tensor,
        grad_in: torch.Tensor, # Renamed to avoid clash with local 'grad'
        momentum: float,
        lr: float,
        weight_decay: float,
        state: dict,
        current_loss: float # Kept arg for potential future use, though not used now
    ) -> bool: # Return True if update was applied, False otherwise
        """Enhanced parameter update with robust gradient handling."""

        # --- Check for non-finite gradient FIRST ---
        if not torch.isfinite(grad_in).all():
            # Option 1: Zero the gradient (safer than skipping momentum update)
            grad = torch.zeros_like(grad_in)
            # Log only once per step maybe to avoid spam
            # if self._step_count % 10 == 0:
            #      logging.warning(f"Step {self._step_count}: Zeroing non-finite gradient for parameter (shape: {p.shape}).")
            # Option 2: Skip update entirely for this parameter (uncomment if preferred)
            # logging.warning(f"Step {self._step_count}: Skipping update for parameter (shape: {p.shape}) due to non-finite gradient.")
            # return False # Indicate no update applied
        else:
            # Make a clone only if the gradient is finite and needs processing
            grad = grad_in.clone() # Clone to avoid modifying p.grad if it's used elsewhere
        # --- End Non-Finite Check ---

        # Initialize momentum buffer if needed
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        buf = state['momentum_buffer']

        # Apply weight decay (only if grad is finite and WD > 0)
        if weight_decay != 0 and torch.isfinite(grad).all(): # Check again just in case
            # Apply WD directly to the parameter before momentum update (common practice)
            # p.add_(p, alpha=-weight_decay * lr) # Decoupled weight decay (alternative)
            # Or add WD to gradient (original method)
            grad = grad.add(p, alpha=weight_decay)

        # Clip the (potentially zeroed or WD-modified) gradient before momentum update
        grad_norm_before_clip = torch.norm(grad).item() # Use .item()
        was_clipped = False
        clip_ratio = 1.0 # Default to 1.0

        # Use stored max_grad_norm
        if grad_norm_before_clip > self.max_grad_norm:
            clip_ratio = self.max_grad_norm / (grad_norm_before_clip + 1e-6) # Add epsilon
            grad.mul_(clip_ratio)
            was_clipped = True

        # Record gradient clipping statistics (using potentially modified grad norm)
        # Note: original_norm here is AFTER potential zeroing/WD but BEFORE clipping
        self.gradient_stats.record_gradient(
            original_norm=grad_norm_before_clip, # Log norm before clipping
            clipped=was_clipped,
            clip_ratio=clip_ratio if was_clipped else 1.0
        )

        # Update momentum buffer using the (potentially zeroed/clipped) gradient
        # Standard momentum: buf = momentum * buf + grad
        buf.mul_(momentum).add_(grad) # Nesterov momentum would calculate grad after applying momentum part of update

        # Compute update step: update = momentum_buffer (if using standard momentum)
        update = buf

        # Apply final update to parameter: p = p - lr * update
        p.add_(update, alpha=-lr)

        return True # Indicate update was applied

    def state_dict(self) -> Dict[str, Any]:
        """Returns the optimizer's state dict with safe serialization."""
        state_dict = super().state_dict()
        try:
            # state_dict['statistics'] = self.get_statistics() # Removed statistics tracking
            state_dict['q_table'] = self.q_controller.q_table
            state_dict['q_controller_epsilon'] = float(self.q_controller.epsilon)
            state_dict['q_controller_prev_loss'] = self.q_controller.prev_loss
            state_dict['q_controller_prev_state'] = self.q_controller.prev_state
            state_dict['q_controller_prev_action'] = self.q_controller.prev_action
            state_dict['q_controller_access_count'] = self.q_controller.q_table_access_count
            state_dict['_step_count'] = self._step_count
        except Exception as e:
            logging.error(f"Error creating EnhancedSGD state dict: {e}")
            # Avoid saving potentially problematic state
            state_dict['q_table'] = {}
            state_dict['q_controller_epsilon'] = 0.15 # Default
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads optimizer state with safe type handling."""
        # Pop custom state elements before calling super
        q_table = state_dict.pop('q_table', None)
        epsilon = state_dict.pop('q_controller_epsilon', None)
        prev_loss = state_dict.pop('q_controller_prev_loss', None)
        prev_state = state_dict.pop('q_controller_prev_state', None)
        prev_action = state_dict.pop('q_controller_prev_action', None)
        access_count = state_dict.pop('q_controller_access_count', None)
        step_count = state_dict.pop('_step_count', None)
        # statistics = state_dict.pop('statistics', None) # Removed

        try:
            super().load_state_dict(state_dict)

            # Load Q-controller state if available
            if q_table is not None:
                self.q_controller.q_table = q_table
            if epsilon is not None:
                self.q_controller.epsilon = float(epsilon)
            if prev_loss is not None:
                 # Ensure loaded loss is float or None
                 self.q_controller.prev_loss = float(prev_loss) if prev_loss is not None else None
            if prev_state is not None:
                self.q_controller.prev_state = prev_state # Assumes state is serializable (tuple)
            if prev_action is not None:
                self.q_controller.prev_action = prev_action # Assumes action is serializable (dict)
            if access_count is not None:
                 self.q_controller.q_table_access_count = access_count
            if step_count is not None:
                 self._step_count = int(step_count)

            # Restore parameter group specific states like q_scale if needed
            # This requires iterating through param_groups and saved_groups in state_dict['param_groups']
            # For simplicity, we might rely on base_lr and recalculate q_scale based on loaded lr
            for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
                 if 'q_scale' in saved_group:
                      group['q_scale'] = saved_group['q_scale']
                 elif 'base_lr' in group and group['base_lr'] > 1e-9: # Recalculate if base_lr exists
                      group['q_scale'] = group['lr'] / group['base_lr']
                 else:
                      group['q_scale'] = 1.0 # Default if cannot recalculate


            # Removed statistics loading

        except Exception as e:
            logging.error(f"Error loading EnhancedSGD state dict: {e}", exc_info=True)
            # Re-initialize Q-controller state if loading fails?
            self.q_controller.q_table = {}
            self.q_controller.epsilon = 0.15
            self.q_controller.prev_loss = None
            self.q_controller.prev_state = None
            self.q_controller.prev_action = None
            self.q_controller.q_table_access_count = {}
            self._step_count = 0
            # Reset q_scale in groups
            for group in self.param_groups:
                 group['q_scale'] = 1.0
                 group['lr'] = group['base_lr'] # Reset LR to base


# =====================================================================
# Model Architecture
# =====================================================================

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
        # Ensure num_heads divides hidden_size
        if hidden_size % num_heads != 0:
             # Find the nearest smaller number of heads that divides hidden_size
             valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
             if not valid_heads:
                  raise ValueError(f"hidden_size {hidden_size} is not divisible by any number of heads <= {num_heads}")
             original_num_heads = num_heads
             num_heads = valid_heads[0]
             logging.warning(f"Adjusted num_heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads" # Already checked

        # Use separate projections for Q, K, V for potentially better representation
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

        # Use separate norms for query and key/value inputs
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,  # [batch_size, num_queries, hidden_size]
        keys_values: torch.Tensor,  # [batch_size, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None # Expects mask where True means MASKED
    ) -> torch.Tensor:
        """Forward pass with proper masking and scaling."""
        batch_size, num_queries, _ = queries.size()
        seq_len_kv = keys_values.size(1) # Use different name for key/value sequence length

        # Layer norm first
        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)

        # Project Q, K, V separately
        q = self.q_proj(queries_norm)
        k = self.k_proj(keys_values_norm)
        v = self.v_proj(keys_values_norm)

        # Split heads
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nq, d]
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nkv, d]
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, h, Nkv, d]

        # --- Use F.scaled_dot_product_attention for efficiency ---
        # Prepare mask: needs shape [B, h, Nq, Nkv] or broadcastable.
        # Input mask is [B, Nkv] (True=MASKED)
        attn_mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2: # Shape [B, Nkv]
                # Expand to [B, 1, 1, Nkv] for broadcasting
                attn_mask_bool = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, Nkv]
            elif attention_mask.dim() == 4: # Shape [B, h, Nq, Nkv]
                 attn_mask_bool = attention_mask
            else:
                 logging.warning(f"Unsupported attention mask shape: {attention_mask.shape}. Ignoring mask.")
            # Ensure boolean type
            if attn_mask_bool is not None:
                 attn_mask_bool = attn_mask_bool.bool()

        # F.scaled_dot_product_attention expects mask where True means KEEP
        # Our mask has True means MASK, so we invert if needed (or pass directly if using attn_mask argument)
        # Note: The `attn_mask` argument in F.scaled_dot_product_attention expects True=MASKED (additive mask)
        # or a boolean mask where True=MASKED.

        try:
             # Use flash attention if available and mask is suitable
             output = F.scaled_dot_product_attention(
                 q, k, v,
                 attn_mask=attn_mask_bool, # Pass boolean mask directly (True=MASKED)
                 dropout_p=self.dropout.p if self.training else 0.0,
                 is_causal=False # Not causal for cross-attention
             )
        except Exception as e:
             # Fallback to manual calculation if flash attention fails
             logging.warning(f"Flash attention failed: {e}. Falling back to manual calculation.")
             scale = 1.0 / math.sqrt(self.head_dim)
             scores = torch.matmul(q, k.transpose(-2, -1)) * scale # [B, h, Nq, Nkv]
             if attn_mask_bool is not None:
                 # Mask scores where mask is True
                 scores = scores.masked_fill(attn_mask_bool, float('-inf'))
             attn_probs = torch.softmax(scores, dim=-1)
             attn_probs = self.dropout(attn_probs)
             output = torch.matmul(attn_probs, v) # [B, h, Nq, d]


        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size) # [B, Nq, H]
        output = self.out_proj(output)
        # Removed dropout here, typically applied after residual connection

        return output


class GlobalLatentTransformer(nn.Module):
    """Large global transformer that processes patch representations."""
    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 16,
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
                'dropout1': nn.Dropout(dropout), # Dropout after attention
                'norm2': nn.LayerNorm(hidden_size, eps=1e-6),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(dropout), # Dropout within MLP
                    nn.Linear(hidden_size * 4, hidden_size)
                ),
                'dropout2': nn.Dropout(dropout) # Dropout after MLP
            }) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (returns boolean mask where True means MASK)."""
        # MultiheadAttention expects mask where True indicates masking
        return torch.triu(
            torch.full((seq_len, seq_len), True, device=device), # Use True for masked positions
            diagonal=1
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process patch representations with causal attention and padding mask.
        Args:
            x: Input tensor [B, S, H]
            padding_mask: Optional padding mask [B, S], True indicates padding (MASKED)
        """
        seq_len = x.size(1)
        device = x.device

        # Create causal mask (True = MASK)
        causal_mask = self.create_causal_mask(seq_len, device) # [S, S]

        # Prepare key_padding_mask for MultiheadAttention (True = MASK)
        # Input padding_mask is [B, S] where True means MASK
        mha_key_padding_mask = padding_mask # Use directly

        # Process through transformer layers
        for layer in self.layers:
            # Residual connection for attention
            residual = x
            # Pre-LayerNorm for attention
            normed = layer['norm1'](x)

            # Self-attention with causal and padding masking
            # attn_mask is for causal masking (prevents attending to future positions)
            # key_padding_mask prevents attending to padding tokens
            attn_output, _ = layer['attention'](
                query=normed,
                key=normed,
                value=normed,
                attn_mask=causal_mask, # Causal mask [S, S]
                key_padding_mask=mha_key_padding_mask, # Padding mask [B, S]
                need_weights=False
            )
            x = residual + layer['dropout1'](attn_output) # Apply dropout after attention

            # Residual connection for MLP
            residual = x
            # Pre-LayerNorm for MLP
            normed = layer['norm2'](x)

            # MLP block
            x = residual + layer['dropout2'](layer['mlp'](normed)) # Apply dropout after MLP

        return self.final_norm(x)


class LocalEncoder(nn.Module):
    """Local encoder that efficiently maps bytes to patches."""
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_heads: int = 8,
        window_size: int = 512, # Max window for local attention
        dropout: float = 0.1,
        n_gram_sizes: List[int] = [3, 4, 5],
        n_gram_vocab_size: int = 30000
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # Byte embeddings
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))

        # N-gram hash embeddings
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
            logging.info(f"LocalEncoder using N-grams: {self.n_gram_sizes} with vocab size {n_gram_vocab_size}")


        self.window_size = window_size

        # Local transformer layers with fixed window attention
        # Use standard TransformerEncoderLayer which includes norm+dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4, # Standard expansion
            dropout=dropout,
            batch_first=True,
            activation=F.gelu # Use GELU activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        # Cross attention for patch creation (pooling)
        self.patch_pooling_attention = CrossAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads, # Use same heads as encoder
            dropout=dropout
        )
        # Learnable query vector for pooling each patch
        self.patch_query = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.norm = nn.LayerNorm(hidden_size, eps=1e-6) # Final norm for output patches
        self.dropout = nn.Dropout(dropout) # General dropout

    def _get_n_gram_hashes(self, byte_sequence: torch.Tensor, n: int) -> torch.Tensor:
        """Calculates rolling hashes for n-grams (simple modulo hashing)."""
        # byte_sequence shape: [B, SeqLen]
        batch_size, seq_len = byte_sequence.shape
        device = byte_sequence.device
        if seq_len < n:
            # Return empty tensor of appropriate shape if sequence is too short
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)

        # Use unfold to get sliding windows
        windows = byte_sequence.unfold(1, n, 1) # Shape [B, NumWindows, n]

        # Simple sum hashing (less unique, but faster and avoids large powers)
        # Ensure dtype is appropriate for sum (long or float) before sum
        hashes = windows.long().sum(dim=-1) # Sum bytes in window [B, NumWindows]

        # Modulo to fit vocab size
        return hashes % self.n_gram_vocab_size # Shape [B, NumWindows]


    def create_local_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local window attention mask (True = MASK)."""
        # Create a full causal mask first (True=MASK upper triangle)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

        # Create a band mask (True=MASK outside window)
        # Indices tensor
        idx = torch.arange(seq_len, device=device)
        # Absolute difference between indices
        abs_diff = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
        # Mask where difference is greater than window size
        band_mask = abs_diff >= self.window_size # True if outside window

        # Combine masks: Mask if causal OR outside window
        final_mask = causal_mask | band_mask
        return final_mask

    def forward(
        self,
        byte_seq: torch.Tensor, # [B, S]
        patch_boundaries: List[List[int]] # List (batch) of lists of boundary indices
    ) -> Tuple[torch.Tensor, torch.Tensor]: # Returns (Padded Patches [B, max_num_p, H], Padding Mask [B, max_num_p])
        """Forward pass with safe indexing and bounds checking."""
        batch_size, seq_len = byte_seq.size()
        device = byte_seq.device

        # Input validation
        if seq_len == 0:
            # Return empty patches and an empty mask
            return torch.zeros((batch_size, 0, self.hidden_size), device=device), \
                   torch.zeros((batch_size, 0), dtype=torch.bool, device=device)

        # 1. Get byte embeddings
        x = self.byte_embeddings(byte_seq)  # [B, S, H]

        # 2. Add N-gram features
        if self.n_gram_embeddings:
            n_gram_features = torch.zeros_like(x) # [B, S, H]
            for n in self.n_gram_sizes:
                if seq_len >= n:
                    # Get hashes for this n-gram size
                    n_gram_hashes = self._get_n_gram_hashes(byte_seq, n) # [B, NumWindows]
                    # Get embeddings for these hashes
                    # Shape [B, NumWindows, H]
                    ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes)
                    # Add embeddings to the corresponding positions
                    # Simplest: Add n-gram embedding starting at pos 'i' to byte embedding at pos 'i'.
                    num_windows = ngram_embeds.size(1)
                    # Add to the first 'num_windows' positions
                    n_gram_features[:, :num_windows, :] += ngram_embeds

            # Add combined n-gram features to byte embeddings (consider scaling?)
            x = x + n_gram_features

        x = self.dropout(x) # Apply dropout after embeddings

        # 3. Process with Local Transformer Encoder
        # Create attention mask (True = MASK) for local window + causality
        local_attn_mask = self.create_local_attention_mask(seq_len, device) # [S, S]
        # TransformerEncoder expects mask where True means MASK
        processed_bytes = self.transformer(x, mask=local_attn_mask) # [B, S, H]

        # 4. Create patch representations using Cross-Attention Pooling
        all_patches = []
        num_patches_per_item = []
        max_num_patches = 0

        for i in range(batch_size):
            item_boundaries = patch_boundaries[i] # Boundaries for this item
            item_processed_bytes = processed_bytes[i] # [S, H]
            item_patches = []
            start_idx = 0

            # Ensure boundaries are valid for this item's sequence length
            valid_boundaries = [b for b in item_boundaries if 0 < b <= seq_len]
            # Add end of sequence as implicit boundary if needed
            if not valid_boundaries or valid_boundaries[-1] < seq_len:
                 # Check if last segment would be too small if seq_len is added
                 last_b = valid_boundaries[-1] if valid_boundaries else 0
                 min_patch_size = 1 # Define minimum patch size
                 if seq_len - last_b >= min_patch_size:
                      valid_boundaries.append(seq_len)


            for end_idx in valid_boundaries:
                if end_idx > start_idx:
                    # Get bytes for this patch [patch_len, H]
                    patch_byte_repr = item_processed_bytes[start_idx:end_idx, :]

                    # Safe mean pooling for query (or use learnable query)
                    if patch_byte_repr.size(0) > 0:
                        # Expand learnable query for batch dim
                        query = self.patch_query # [1, 1, H]

                        # Cross attend: Query attends to patch bytes
                        # Input shapes: query=[1, 1, H], keys_values=[1, patch_len, H]
                        patch_repr = self.patch_pooling_attention(
                            queries=query,
                            keys_values=patch_byte_repr.unsqueeze(0) # Add batch dim
                        ) # Output: [1, 1, H]
                        item_patches.append(patch_repr.squeeze(0)) # Remove batch dim -> [1, H]

                    start_idx = end_idx

            if item_patches:
                # Stack patches for this item: [num_p, H]
                item_patches_stacked = torch.cat(item_patches, dim=0)
                all_patches.append(item_patches_stacked)
                num_p = item_patches_stacked.size(0)
                num_patches_per_item.append(num_p)
                max_num_patches = max(max_num_patches, num_p)
            else:
                # Handle case with no valid patches for an item (e.g., short sequence)
                # Create a single placeholder patch (e.g., mean of all processed bytes)
                placeholder_patch = item_processed_bytes.mean(dim=0, keepdim=True) # [1, H]
                all_patches.append(placeholder_patch)
                num_patches_per_item.append(1)
                max_num_patches = max(max_num_patches, 1)


        # 5. Pad patch sequences and create padding mask
        padded_patches_list = []
        for patches_tensor in all_patches:
            num_p = patches_tensor.size(0)
            padding_size = max_num_patches - num_p
            if padding_size > 0:
                padding = torch.zeros(padding_size, self.hidden_size, device=device)
                padded_patches = torch.cat([patches_tensor, padding], dim=0)
            else:
                padded_patches = patches_tensor
            padded_patches_list.append(padded_patches.unsqueeze(0)) # Add batch dim back

        # Stack padded patches into a batch tensor
        final_patches = torch.cat(padded_patches_list, dim=0) # [B, max_num_p, H]
        final_patches = self.norm(final_patches) # Apply final norm

        # Create padding mask (True = MASKED)
        num_patches_tensor = torch.tensor(num_patches_per_item, device=device) # [B]
        # Mask where index >= num_valid_patches
        padding_mask = torch.arange(max_num_patches, device=device)[None, :] >= num_patches_tensor[:, None] # [B, max_num_p]

        return final_patches, padding_mask


class LocalDecoder(nn.Module):
    """Local decoder that maps patches back to bytes."""
    def __init__(
        self,
        hidden_size: int = 256,
        global_hidden_size: int = 512, # Size of incoming patch representations
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size # Internal hidden size for decoder

        # Byte embeddings for target sequence
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))

        # Projection from global patch representation size to decoder hidden size
        self.memory_projection = nn.Linear(global_hidden_size, hidden_size)
        nn.init.normal_(self.memory_projection.weight, std=0.02)
        nn.init.zeros_(self.memory_projection.bias)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4, # Standard expansion
            dropout=dropout,
            batch_first=True,
            activation=F.gelu # Use GELU
        )
        # Use TransformerDecoder which handles the stack of layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to bytes with proper initialization
        self.byte_pred = nn.Linear(hidden_size, 256)
        nn.init.normal_(self.byte_pred.weight, std=0.02)
        nn.init.zeros_(self.byte_pred.bias)

        self.norm = nn.LayerNorm(hidden_size, eps=1e-6) # Final norm before output projection
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for decoder (True = MASK)."""
        # TransformerDecoderLayer expects mask where True means ignore
        return torch.triu(
            torch.full((seq_len, seq_len), True, device=device), # Use True for masked positions
            diagonal=1
        )

    def forward(
        self,
        tgt_byte_seq: torch.Tensor, # Target byte sequence [B, T]
        memory: torch.Tensor,  # Encoded patches [B, M, global_hidden_size]
        memory_key_padding_mask: Optional[torch.Tensor] = None # Padding mask for memory [B, M] (True=MASK)
    ) -> torch.Tensor:
        """
        Decode patches to byte predictions.

        Args:
            tgt_byte_seq: Target byte sequence (e.g., for teacher forcing) [B, T]
            memory: Patch representations from global transformer [B, M, global_hidden_size]
            memory_key_padding_mask: Mask for memory padding [B, M] (True=MASK)

        Returns:
            Logits for next byte prediction [B, T, 256]
        """
        batch_size, tgt_len = tgt_byte_seq.size()
        device = tgt_byte_seq.device

        # 1. Embed target sequence
        tgt_embed = self.byte_embeddings(tgt_byte_seq) # [B, T, H]
        tgt_embed = self.dropout(tgt_embed)

        # 2. Project memory to decoder hidden size
        projected_memory = self.memory_projection(memory) # [B, M, H]

        # 3. Create causal mask for target self-attention (True = MASK)
        tgt_mask = self.create_causal_mask(tgt_len, device) # [T, T]

        # 4. Process through transformer decoder layers
        # TransformerDecoder expects masks where True means ignore.
        output = self.transformer_decoder(
            tgt=tgt_embed, # Target sequence embeddings
            memory=projected_memory, # Projected patch representations
            tgt_mask=tgt_mask, # Causal mask for self-attention [T, T]
            memory_mask=None, # Optional mask for cross-attention (usually not needed)
            tgt_key_padding_mask=None, # Optional padding mask for target (if target has padding)
            memory_key_padding_mask=memory_key_padding_mask # Padding mask for memory [B, M]
        ) # Output: [B, T, H]

        # 5. Final normalization and projection to byte logits
        output = self.norm(output)
        byte_logits = self.byte_pred(output)  # [B, T, 256]

        return byte_logits

    # Removed generate_step as generation logic is usually handled outside the core model forward pass


# =====================================================================
# BLTModel: Complete Model Architecture
# =====================================================================

class BLTModel(nn.Module):
    """Complete Byte Latent Transformer model implementation."""
    def __init__(
        self,
        local_hidden_size: int = 256,
        global_hidden_size: int = 512, # Keep consistent with args default
        num_local_encoder_layers: int = 1,
        num_global_layers: int = 8, # Keep consistent with args default
        num_local_decoder_layers: int = 4,
        dropout: float = 0.1,
        window_size: int = 256, # Local attention window
        n_gram_sizes: List[int] = [3, 4], # Default n-grams
        n_gram_vocab_size: int = 30000
    ):
        super().__init__()

        # Enable memory-efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp') and torch.cuda.is_available():
            try:
                 torch.backends.cuda.enable_mem_efficient_sdp(True)
                 logging.info("Memory-efficient SDP enabled.")
            except Exception as e:
                 logging.warning(f"Could not enable memory-efficient SDP: {e}")
        else:
             logging.info("Memory-efficient SDP not available or not enabled.")


        # Entropy-based patching
        self.patcher = BabylonIndex(
            scales=n_gram_sizes,
            max_cache_size=100000, # Cache size for entropy calculations
            min_entropy_threshold=0.3 # Threshold for boundary detection
        )

        # Local encoder for bytes to patches
        self.local_encoder = LocalEncoder(
            hidden_size=local_hidden_size,
            num_layers=num_local_encoder_layers,
            num_heads=8, # Standard number of heads for local encoder
            window_size=window_size, # Local attention window size
            n_gram_sizes=n_gram_sizes,
            n_gram_vocab_size=n_gram_vocab_size,
            dropout=dropout
        )

        # Project patches to global hidden size
        self.patch_projection = nn.Sequential(
            nn.Linear(local_hidden_size, global_hidden_size),
            nn.LayerNorm(global_hidden_size, eps=1e-6),
            nn.Dropout(dropout) # Add dropout after projection
        )

        # Global transformer for patches
        self.global_transformer = GlobalLatentTransformer(
            hidden_size=global_hidden_size,
            num_layers=num_global_layers,
            num_heads=16, # Adjust heads based on global_hidden_size (e.g., 512 // 16 = 32)
            dropout=dropout
        )

        # Project back to local hidden size for decoder
        self.patch_deprojection = nn.Sequential(
            nn.Linear(global_hidden_size, local_hidden_size),
            nn.LayerNorm(local_hidden_size, eps=1e-6),
            nn.Dropout(dropout) # Add dropout after deprojection
        )

        # Local decoder for patches to bytes
        self.local_decoder = LocalDecoder(
            hidden_size=local_hidden_size,
            global_hidden_size=local_hidden_size, # Decoder expects memory projected back to its size
            num_layers=num_local_decoder_layers,
            num_heads=8, # Standard number of heads for local decoder
            dropout=dropout
        )

        self.context_size = window_size  # Store for reference/generation logic if needed

    def forward(
        self,
        byte_seq: torch.Tensor,  # Input byte sequence [B, S]
        target_byte_seq: Optional[torch.Tensor] = None # Target sequence for training [B, T]
        # return_patch_boundaries: bool = False # Removed, simplify forward pass
    ) -> torch.Tensor: # Returns byte logits [B, T, 256] or similar based on task
        """Complete forward pass of BLT model for training or inference."""
        batch_size, seq_len = byte_seq.size()
        device = byte_seq.device

        # 1. Find patch boundaries for each item in the batch
        #    Requires iterating or modifying patcher for batch processing
        batch_boundaries = []
        for i in range(batch_size):
             # Need to handle potential errors in find_patch_boundaries per item
             try:
                  boundaries = self.patcher.find_patch_boundaries(byte_seq[i].unsqueeze(0)) # Pass [1, S] tensor
                  batch_boundaries.append(boundaries)
             except Exception as patch_ex:
                  logging.error(f"Error finding boundaries for batch item {i}: {patch_ex}", exc_info=True)
                  batch_boundaries.append([seq_len]) # Fallback: treat whole sequence as one patch


        # 2. Encode bytes into patches, get padding mask
        #    LocalEncoder now handles batching and padding internally
        patch_repr, patch_padding_mask = self.local_encoder(byte_seq, batch_boundaries)
        # patch_repr: [B, max_num_p, local_hidden]
        # patch_padding_mask: [B, max_num_p] (True = MASKED)

        # 3. Project patches to global hidden size
        projected_patches = self.patch_projection(patch_repr)  # [B, max_num_p, global_hidden]

        # 4. Process with global transformer (pass padding mask)
        global_patch_repr = self.global_transformer(projected_patches, padding_mask=patch_padding_mask)
        # global_patch_repr: [B, max_num_p, global_hidden]

        # 5. Project back to local hidden size for decoder memory
        decoder_memory = self.patch_deprojection(global_patch_repr)  # [B, max_num_p, local_hidden]

        # 6. Decode using the target sequence (if provided for training)
        if target_byte_seq is not None:
            # Standard teacher forcing: Decode using the target sequence
            byte_logits = self.local_decoder(
                tgt_byte_seq=target_byte_seq, # [B, T]
                memory=decoder_memory, # [B, M, H_local]
                memory_key_padding_mask=patch_padding_mask # [B, M] (True=MASK)
            ) # Output: [B, T, 256]
            return byte_logits
        else:
            # --- Inference/Generation Mode ---
            # This part depends heavily on the generation task.
            # For simple next-byte prediction based on input context `byte_seq`:
            # We need a way to query the decoder based on the final state of the context.
            # Option A: Use the last byte of the input as the initial target for the decoder.
            last_input_byte = byte_seq[:, -1:] # [B, 1]
            byte_logits = self.local_decoder(
                tgt_byte_seq=last_input_byte,
                memory=decoder_memory,
                memory_key_padding_mask=patch_padding_mask
            ) # Output: [B, 1, 256]
            return byte_logits
            # Option B: Add a special <PREDICT> token/query for the decoder. (More complex)
            # Option C: Generation loop handled outside `forward`. (Most common)
            # For now, returning logits based on the last input byte.


    @staticmethod
    
    def compute_loss(
        logits: torch.Tensor,  # [B, S, 256] - Raw logits from the model's forward pass
        targets: torch.Tensor, # [B, S] - The *input* sequence (like `context`), used for shifting
        mask: Optional[torch.Tensor] = None, # Optional mask [B, S] (True=IGNORE) - Should align with targets
        smoothing: float = 0.1
    ) -> torch.Tensor:
        """Compute cross entropy loss for next-token prediction with label smoothing."""
        batch_size, seq_len, vocab_size = logits.size()

        # --- Shift logits and targets for next-token prediction ---
        logits_shifted = logits[:, :-1, :].contiguous()  # Shape: [B, S-1, V] (Predictions for token 1 to S)
        targets_shifted = targets[:, 1:].contiguous()    # Shape: [B, S-1] (Actual tokens 1 to S)

        # Reshape logits and targets for loss computation
        logits_flat = logits_shifted.view(-1, vocab_size)      # Shape: [B * (S-1), V]
        targets_flat = targets_shifted.view(-1)                # Shape: [B * (S-1)]

        # --- Validate target indices ---
        current_vocab_size = logits_flat.size(-1)
        if torch.any(targets_flat >= current_vocab_size) or torch.any(targets_flat < 0):
            invalid_indices = torch.where((targets_flat < 0) | (targets_flat >= current_vocab_size))[0]
            logging.error(f"Target indices out of bounds (0 <= index < {current_vocab_size}). Found values like: {targets_flat[invalid_indices[:10]]}")
            targets_flat = torch.clamp(targets_flat, 0, current_vocab_size - 1)

        # Calculate loss using cross_entropy with label smoothing
        loss_per_element = F.cross_entropy(
            logits_flat,
            targets_flat,
            label_smoothing=smoothing,
            reduction='none'
        ) # Output shape: [B * (S-1)]

        # Apply mask if provided
        mean_loss = None # Initialize mean_loss
        if mask is not None:
             if mask.size(1) != seq_len:
                  logging.warning(f"Mask shape {mask.shape} does not match target sequence length {seq_len}. Masking might be incorrect.")
                  mask_shifted = mask[:, 1:seq_len].contiguous()
             else:
                   mask_shifted = mask[:, 1:].contiguous()

             if mask_shifted.shape == targets_shifted.shape:
                 mask_flat = mask_shifted.view(-1)
                 loss_per_element = loss_per_element * (~mask_flat.bool())
                 num_active_elements = (~mask_flat.bool()).sum()
                 mean_loss = loss_per_element.sum() / num_active_elements if num_active_elements > 0 else torch.tensor(0.0, device=logits.device)
             else:
                  logging.error(f"Mask shape {mask_shifted.shape} after shifting does not match target shape {targets_shifted.shape}. Skipping mask.")
                  mean_loss = loss_per_element.mean()
        else:
             mean_loss = loss_per_element.mean()

        
        return mean_loss
        


    @torch.no_grad()
    def generate(
        self,
        seed_bytes: torch.Tensor, # [B, seed_len]
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None, # Add top_k sampling
        top_p: Optional[float] = None, # Add top_p (nucleus) sampling
        sampling_config: Optional[SamplerConfig] = None # Keep entropy sampling option
    ) -> torch.Tensor:
        """
        Generate new bytes starting from seed bytes using autoregressive sampling.

        Args:
            seed_bytes: Starting byte sequence [B, seed_len]
            max_length: Maximum number of *new* bytes to generate
            temperature: Sampling temperature (0 for greedy)
            top_k: Keep only top_k most likely tokens (if not None)
            top_p: Keep smallest set of tokens whose cumulative probability exceeds top_p (if not None)
            sampling_config: Config for entropy-based adaptive sampling (overrides top_k/top_p if used)

        Returns:
            Generated byte sequence [B, seed_len + max_length]
        """
        self.eval() # Set model to evaluation mode
        device = seed_bytes.device
        batch_size, seed_len = seed_bytes.size()
        generated_sequence = seed_bytes.clone()

        # Use default sampling config if needed (only if top_k/top_p are None)
        use_entropy_sampling = sampling_config is not None and top_k is None and top_p is None
        if use_entropy_sampling and sampling_config is None:
            sampling_config = SamplerConfig()

        # Reset patcher context if stateful (optional)
        # self.patcher.reset_context()

        for _ in range(max_length):
            # Prepare context for the model's forward pass
            # Use the entire generated sequence as input context for patching/encoding
            current_context = generated_sequence
            # Decoder input is the sequence generated so far (needed for teacher forcing setup)
            # For generation, we only need the *last* predicted token's logits.
            # The forward pass needs a target sequence input. We can pass the current generated sequence.
            target_for_forward = current_context # Pass the whole sequence

            # Get logits for the *next* token prediction after the current sequence
            logits_all = self(byte_seq=current_context, target_byte_seq=target_for_forward) # [B, current_len, 256]

            # Check if logits are valid
            if logits_all.shape[1] == 0:
                 logging.warning("Generation stopped: Model returned empty logits.")
                 break

            next_byte_logits = logits_all[:, -1, :] # Get logits for the last position [B, 256]

            # --- Apply Sampling Strategy ---
            if temperature <= 0: # Greedy decoding
                next_byte_indices = torch.argmax(next_byte_logits, dim=-1) # [B]
            else:
                # Apply temperature scaling
                scaled_logits = next_byte_logits / temperature

                # --- Entropy-based Adaptive Sampling ---
                if use_entropy_sampling:
                    probs = F.softmax(scaled_logits, dim=-1) # [B, 256]
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1) # [B]
                    next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                    for i in range(batch_size):
                        current_entropy = entropy[i].item()
                        current_probs = probs[i] # [256]
                        if current_entropy < sampling_config.low_entropy_threshold:
                            next_byte_idx = torch.argmax(current_probs)
                        elif current_entropy < sampling_config.medium_entropy_threshold:
                            k = 10 # Example top-k for medium entropy
                            actual_k = min(k, 256)
                            top_k_probs, top_k_indices = torch.topk(current_probs, k=actual_k)
                            top_k_probs = top_k_probs / (top_k_probs.sum() + 1e-9)
                            sampled_relative_idx = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
                            next_byte_idx = top_k_indices[sampled_relative_idx]
                        else: # High entropy
                            next_byte_idx = torch.multinomial(current_probs, num_samples=1).squeeze(-1)
                        next_byte_indices[i] = next_byte_idx
                # --- Top-k / Top-p Sampling ---
                else:
                    # Apply Top-K filtering
                    if top_k is not None and top_k > 0:
                        k = min(top_k, scaled_logits.size(-1)) # Ensure k is not larger than vocab size
                        # Remove logits below the top-k threshold
                        indices_to_remove = scaled_logits < torch.topk(scaled_logits, k, dim=-1)[0][..., -1, None]
                        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))

                    # Apply Top-P (nucleus) filtering
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        # Scatter sorted tensors to original indexing
                        indices_to_remove = torch.zeros_like(scaled_logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))

                    # Sample from the filtered distribution
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_byte_indices = torch.multinomial(probs, num_samples=1).squeeze(-1) # [B]


            # Append the chosen next byte to the sequence
            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)

        self.train() # Set model back to training mode
        return generated_sequence


# =====================================================================
# Trainer Class
# =====================================================================

class RLHFTrainer: # Renaming might be confusing if no RLHF is actually happening
    """Trainer class that integrates Q-Learning EnhancedSGD with the training loop."""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer, # Expects EnhancedSGD or similar
        device: torch.device,
        scaler = None # Optional GradScaler
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        # Initialize GradScaler if not provided and AMP is desired
        self.scaler = scaler if scaler is not None else amp.GradScaler(enabled=torch.cuda.is_available())
        self._step_count = 0 # Tracks optimizer steps

    def train_step(
        self,
        context: torch.Tensor,  # Input byte sequences [B, S]
        target: torch.Tensor,  # Target bytes for next step prediction [B, S] (shifted input)
        accumulation_steps: int = 1
    ) -> Optional[float]: # Return loss value or None if accumulating
        """
        Perform a single training step with gradient accumulation.

        Args:
            context: Input byte sequences [B, S]
            target: Target bytes (usually context shifted by 1) [B, S]
            accumulation_steps: Number of steps to accumulate gradients

        Returns:
            loss: Loss value for this step (before accumulation scaling), or None if accumulating.
        """
        self.model.train()
        is_optimizing_step = (self._step_count + 1) % accumulation_steps == 0

        # Forward pass with AMP
        with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            # Model expects byte_seq and target_byte_seq for teacher forcing
            logits = self.model(byte_seq=context, target_byte_seq=context) # Use context as both input and target for teacher forcing
            # Loss calculation needs logits [B, S, V] and targets [B, S]
            # NEW LINE: Pass 'context' as the second argument for shifting inside compute_loss
            loss = self.model.compute_loss(logits, context)
            
            # Scale loss for accumulation
            loss_scaled = loss / accumulation_steps

        # Backward pass with gradient scaling
        # Detach loss before scaling/backward to avoid graph issues? Usually not needed.
        self.scaler.scale(loss_scaled).backward()

        step_loss = loss.item() # Store unscaled loss for reporting

        # Optimizer step only when accumulation is complete
        if is_optimizing_step:
            # Unscale gradients before clipping (important!)
            self.scaler.unscale_(self.optimizer)
            # Optional: Clip gradients (EnhancedSGD might do this internally, check its implementation)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.optimizer.max_grad_norm)

            # Pass loss to optimizer *before* step if needed (like EnhancedSGD)
            if hasattr(self.optimizer, 'q_controller') and hasattr(self.optimizer.q_controller, 'prev_loss'):
                 # Pass the unscaled loss for the current step/batch
                 self.optimizer.q_controller.current_loss_for_step = step_loss # Use a temporary attribute if needed

            # Optimizer step
            self.scaler.step(self.optimizer)
            # Update the scaler for next iteration
            self.scaler.update()
            # Zero gradients *after* optimizer step
            self.optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential memory savings

            self._step_count += 1 # Increment optimizer step counter
            return step_loss # Return loss for the step where optimization happened
        else:
             self._step_count += 1 # Increment accumulation step counter
             return None # Indicate accumulation step


    @torch.no_grad() # Ensure no gradients are computed during validation
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model on the validation dataset.

        Args:
            dataloader: Validation dataloader

        Returns:
            metrics: Dictionary of validation metrics (e.g., loss, perplexity)
        """
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        num_samples = 0
        # entropy_vals = [] # Entropy calculation removed for simplicity

        val_iterator = iter(dataloader) # Use iterator for clarity
        num_batches = 0

        while True: # Loop until iterator is exhausted
            try:
                 batch_data = next(val_iterator)
                 num_batches += 1
                 # --- Process Batch ---
                 if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                     context, target = batch_data
                 else:
                     logging.warning(f"Unexpected validation batch format at batch {num_batches}. Skipping.")
                     continue

                 context = context.to(self.device, non_blocking=True)
                 target = target.to(self.device, non_blocking=True)
                 batch_size = context.size(0)

                 # Use AMP context manager for consistency, though grads aren't needed
                 with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                     # Forward pass - use context as both inputs for teacher-forcing style eval
                     logits = self.model(byte_seq=context, target_byte_seq=context)
                     loss = self.model.compute_loss(logits, target, smoothing=0.0) # No label smoothing for validation

                 # Accumulate loss, weighted by batch size if batches are uneven
                 total_loss += loss.item() * batch_size
                 num_samples += batch_size
                 # --- End Batch Processing ---

            except StopIteration:
                 break # Exit loop when dataloader is exhausted
            except Exception as val_ex:
                 logging.error(f"Validation Error during batch {num_batches}: {val_ex}", exc_info=True)
                 continue # Skip batch on error


        # Calculate average validation loss
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        # Calculate perplexity = exp(average_loss)
        perplexity = math.exp(min(avg_loss, 20)) if avg_loss > 0 else float('inf') # Cap loss for exp, handle non-positive loss

        metrics = {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
            # 'val_entropy': avg_entropy # Removed
        }

        self.model.train() # Set model back to training mode
        return metrics

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int, # Epoch number completed
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Save model checkpoint with optimizer state.
        """
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Get model state dict, handling DDP if necessary
        model_to_save = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        model_state = model_to_save.state_dict()

        # Get optimizer state dict
        optimizer_state = self.optimizer.state_dict()

        # Get scaler state dict
        scaler_state = self.scaler.state_dict()

        checkpoint = {
            'epoch': epoch, # Save epoch number completed
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scaler_state_dict': scaler_state,
            'val_metrics': val_metrics,
            'amp_enabled': self.scaler.is_enabled() # Save AMP status
        }

        try:
            torch.save(checkpoint, filepath)
            logging.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint {filepath}: {e}", exc_info=True)


    def load_checkpoint(self, filepath: str) -> int:
        """
        Load model checkpoint with optimizer state.

        Args:
            filepath: Path to checkpoint

        Returns:
            start_epoch: Epoch number to start training from (epoch saved + 1)
        """
        if not os.path.exists(filepath):
            logging.error(f"Checkpoint {filepath} does not exist")
            return 0 # Start from epoch 0 if checkpoint not found

        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # Load model state, handling DDP if necessary
            model_to_load = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
            # Use strict=False to handle potential architecture changes gracefully
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible_keys.missing_keys:
                 logging.warning(f"Missing keys when loading model state: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                 logging.warning(f"Unexpected keys when loading model state: {incompatible_keys.unexpected_keys}")


            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load GradScaler state
            # Check if AMP state exists and current setting matches checkpoint
            saved_amp_enabled = checkpoint.get('amp_enabled', False)
            if self.scaler.is_enabled() and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                if saved_amp_enabled:
                     self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                else:
                     logging.warning("Loading checkpoint without AMP state, but AMP is enabled now. Scaler state not loaded.")
            elif not self.scaler.is_enabled() and saved_amp_enabled:
                 logging.warning("Loading checkpoint with AMP state, but AMP is disabled now.")


            # Load epoch number (epoch completed)
            epoch_completed = checkpoint.get('epoch', 0)
            start_epoch = epoch_completed # Start from the epoch *after* the saved one

            logging.info(f"Loaded checkpoint from {filepath}. Resuming from Epoch {start_epoch}")
            return start_epoch

        except Exception as e:
            logging.error(f"Failed loading checkpoint '{filepath}': {e}", exc_info=True)
            return 0 # Start from scratch on failure

def parse_args():
    """Parse command-line arguments for Bytropix training."""
    parser = argparse.ArgumentParser(description="Bytropix - Byte Latent Transformer Training (Stabilized)")

    # Data parameters
    parser.add_argument("--data_path", type=str, default="C:/projects/bytropix/data/wikitext_train.npy",
                        help="Path to training data numpy file")
    parser.add_argument("--val_data_path", type=str, default="C:/projects/bytropix/data/wikitext_val.npy",
                        help="Path to validation data numpy file")
    parser.add_argument("--context_size", type=int, default=128,
                        help="Context size for training and generation")
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of data to use (0.0 to 1.0)")


    # Model parameters
    parser.add_argument("--local_hidden_size", type=int, default=256,
                        help="Hidden size for local encoder/decoder")
    parser.add_argument("--global_hidden_size", type=int, default=512,  # Reduced from 1024 for lower memory usage
                        help="Hidden size for global transformer")
    parser.add_argument("--num_local_encoder_layers", type=int, default=1,
                        help="Number of layers in local encoder")
    parser.add_argument("--num_global_layers", type=int, default=8,  # Reduced from 16 for lower memory usage
                        help="Number of layers in global transformer")
    parser.add_argument("--num_local_decoder_layers", type=int, default=4,
                        help="Number of layers in local decoder")
    parser.add_argument("--window_size", type=int, default=256,
                        help="Window size for local encoding attention")
    parser.add_argument("--n_gram_sizes", type=int, nargs='+', default=[3, 4], help="N-gram sizes for patcher/encoder")
    parser.add_argument("--n_gram_vocab_size", type=int, default=30000, help="Vocab size for N-gram hashing")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for model layers")


    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,  # Smaller batch size for memory efficiency
                        help="Batch size per GPU for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, # Adjusted initial LR
                        help="Initial learning rate for EnhancedSGD")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (used by EnhancedSGD)")
    parser.add_argument("--checkpoint_dir", type=str, default="./bytropix_checkpoints_stabilized", # Changed default dir
                        help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=50, # Log less frequently
                        help="Logging interval in optimizer steps")
    # parser.add_argument("--save_interval", type=int, default=1000, # Removed, saving per epoch/half-epoch now
    #                     help="Checkpoint saving interval in steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for EnhancedSGD")


    # Distributed training parameters
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Local rank for distributed training (set by torchrun/launch)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging (requires wandb installed)")
    parser.add_argument("--wandb_project", type=str, default="bytropix-stabilized",
                        help="Weights & Biases project name")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader worker processes")


    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist (only needed on rank 0, but safe to call on all)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args

def setup_distributed(local_rank):
    """Initializes distributed training environment."""
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Running on single device: {device}")
        return False, device, 1 # is_distributed, device, world_size

    # DDP Setup
    if not torch.cuda.is_available() or torch.cuda.device_count() <= local_rank:
        logging.error("Distributed training requested but CUDA is not available or local_rank is invalid.")
        # Fallback to CPU? Or raise error? Let's fallback for now.
        return False, torch.device("cpu"), 1

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Check if environment variables are set (usually by torchrun/launch)
    if "WORLD_SIZE" not in os.environ:
        # This case should ideally not happen if using torchrun
        logging.warning("DDP environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) not set. Assuming single-node setup.")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(random.randint(10000, 20000)) # Random port
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

    try:
        # Initialize the process group
        init_process_group(backend="nccl", init_method="env://") # Use env:// which reads from os.environ
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        logging.info(f"Distributed Training Initialized. Rank: {rank}/{world_size}, Device: {device}")
        # Synchronize processes before starting training
        if torch.cuda.is_available():
             torch.distributed.barrier()
        return True, device, world_size
    except Exception as e:
        logging.error(f"Distributed process group initialization failed: {e}", exc_info=True)
        return False, device, 1 # Fallback if init fails


def train(args):
    """Main training function."""
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Optional: Configure cuDNN behavior
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Set up device and distributed training
    is_distributed, device, world_size = setup_distributed(args.local_rank)
    is_main_process = not is_distributed or torch.distributed.get_rank() == 0

    # Initialize wandb if enabled (only on main process)
    if args.wandb and is_main_process and WANDB_AVAILABLE:
        try:
            wandb.init(project=args.wandb_project, config=args)
            logging.info("Initialized wandb logging")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            args.wandb = False # Disable wandb if init fails
    elif args.wandb and not WANDB_AVAILABLE:
         logging.warning("Wandb requested but not installed. Disabling.")
         args.wandb = False


    # Create datasets
    try:
        logging.info(f"Loading training data from {args.data_path}")
        train_dataset = ByteIterableDataset(args.data_path, context_size=args.context_size, data_fraction=args.data_fraction)

        logging.info(f"Loading validation data from {args.val_data_path}")
        val_dataset = ByteIterableDataset(args.val_data_path, context_size=args.context_size, data_fraction=1.0) # Use full val set
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}", exc_info=True)
        # Clean up DDP if initialized before exiting
        if is_distributed and is_initialized():
            destroy_process_group()
        sys.exit(1) # Exit if data loading fails


    # Create data samplers and loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=args.local_rank if is_distributed else 0, shuffle=False, seed=args.seed, drop_last=True) if is_distributed else None
    # Note: IterableDatasets often handle shuffling internally. If using DistributedSampler, shuffle=False is typical.
    # Ensure each worker gets unique data if num_workers > 0.

    # Calculate approximate batches per epoch for IterableDataset
    try:
        # This relies on __len__ being implemented correctly and meaningfully
        dataset_len = len(train_dataset)
        # Adjust length for DDP drop_last=True behavior
        if is_distributed:
             # Sampler effectively shortens dataset per rank
             effective_len_per_rank = dataset_len // world_size
        else:
             effective_len_per_rank = dataset_len

        # Calculate steps based on effective length and batch size
        approx_total_steps_per_epoch = effective_len_per_rank // args.batch_size
        logging.info(f"Estimated steps per epoch (per rank): {approx_total_steps_per_epoch}")
    except TypeError:
        logging.warning("Could not determine dataset length from __len__. Progress bar and mid-epoch saving might be inaccurate.")
        # Fallback or fixed number if length is unknown
        approx_total_steps_per_epoch = getattr(args, 'steps_per_epoch', 10000) # Use arg or a large default
        logging.warning(f"Using fallback steps per epoch: {approx_total_steps_per_epoch}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True # Important for DDP to ensure consistent batch counts across ranks
        # shuffle=False # Shuffle is handled by sampler or dataset iterator
    )

    # Validation loader doesn't need distributed sampling usually, evaluate on full set on rank 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, # Use larger batch size for validation if memory allows
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if is_main_process else None # Only create val_loader on main process


    # Log dataset sizes (if available)
    logging.info(f"Training data size: {getattr(train_dataset, 'data_size', 'N/A')} bytes")
    if is_main_process:
        logging.info(f"Validation data size: {getattr(val_dataset, 'data_size', 'N/A')} bytes")

    # Create model
    logging.info("Initializing BLTModel")
    model_config = {
        "local_hidden_size": args.local_hidden_size,
        "global_hidden_size": args.global_hidden_size,
        "num_local_encoder_layers": args.num_local_encoder_layers,
        "num_global_layers": args.num_global_layers,
        "num_local_decoder_layers": args.num_local_decoder_layers,
        "dropout": args.dropout,
        "window_size": args.window_size,
        "n_gram_sizes": args.n_gram_sizes,
        "n_gram_vocab_size": args.n_gram_vocab_size
    }
    model = BLTModel(**model_config).to(device)


    # Log model architecture on main process
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
        # Optional: Log model structure
        # logging.info(model)


    # Wrap model with DDP if using distributed training
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False # Set to True if you encounter issues, but potentially slower
        )
        logging.info("Model wrapped with DistributedDataParallel")

    # Create optimizer with Q-Learning enhanced SGD
    logging.info("Creating EnhancedSGD optimizer")
    optimizer = EnhancedSGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9, # Default momentum
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scale_bounds=(0.85, 1.15), # Example bounds
        momentum_scale_bounds=(0.9, 1.1), # Example bounds
        q_learning_config={ # Pass Q-learning specific hyperparams here
            "learning_rate": 0.02, # Q-agent's LR
            "discount": 0.97,
            "epsilon": 0.15,
            "epsilon_decay": 0.9995,
            "min_epsilon": 0.02,
            "max_q_table_size": 20000 # Increased size
        }
    )

    # Create GradScaler for AMP
    # Enable based on args and CUDA availability
    amp_enabled = not args.no_amp and torch.cuda.is_available()
    scaler = amp.GradScaler(enabled=amp_enabled)
    logging.info(f"Created GradScaler (AMP {'enabled' if amp_enabled else 'disabled'})")

    # Create trainer instance
    trainer = RLHFTrainer(model, optimizer, device, scaler)
    # Reset optimizer step count for the trainer instance
    trainer._step_count = 0

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
             try:
                 logging.info(f"Attempting to resume from checkpoint: {args.resume}")
                 start_epoch = trainer.load_checkpoint(args.resume) # load_checkpoint returns epoch to start FROM
                 logging.info(f"Resumed from checkpoint. Starting training from epoch {start_epoch}")
             except Exception as e:
                 logging.error(f"Failed to load checkpoint '{args.resume}': {e}", exc_info=True)
                 logging.warning("Starting training from scratch.")
                 start_epoch = 0
        else:
             logging.warning(f"Resume checkpoint not found: {args.resume}. Starting training from scratch.")
             start_epoch = 0


    # Create checkpoint directory if it doesn't exist (main process only)
    if args.checkpoint_dir and is_main_process:
         os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Training Loop ---
    logging.info(f"Starting training from epoch {start_epoch} for {args.epochs} total epochs")
    global_step_counter = start_epoch * approx_total_steps_per_epoch # Estimate starting global step

    try:
        for epoch in range(start_epoch, args.epochs):
            if is_distributed:
                # Set epoch for sampler to ensure proper shuffling/data distribution per epoch
                train_sampler.set_epoch(epoch)

            model.train() # Ensure model is in training mode
            epoch_loss_sum = 0.0
            batches_processed_in_epoch = 0
            optimizer_steps_in_epoch = 0

            # Use tqdm for progress bar if possible (requires knowing total steps)
            if approx_total_steps_per_epoch > 0:
                 progress_bar = tqdm(enumerate(train_loader),
                                     total=approx_total_steps_per_epoch,
                                     desc=f"Epoch {epoch+1}/{args.epochs}",
                                     disable=not is_main_process) # Disable bar on non-main processes
            else:
                 progress_bar = enumerate(train_loader) # No total steps known


            for i, batch_data in progress_bar:
                 current_batch_step_in_epoch = i + 1
                 # --- Batch Processing ---
                 try:
                      # Adjust depending on your dataset's output format
                      if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                          context, target = batch_data # Assumes [B,S], [B,S] for next token prediction
                      else:
                          logging.warning(f"Unexpected batch format at step {i}. Skipping.")
                          continue

                      context = context.to(device, non_blocking=True)
                      target = target.to(device, non_blocking=True)

                      # Perform training step (handles accumulation)
                      loss = trainer.train_step(context, target, args.grad_accum_steps)

                      # Check if an optimizer step was performed in train_step
                      is_optimizing_step = (trainer._step_count % args.grad_accum_steps == 0)

                      if loss is not None: # Loss is returned only on optimizing steps
                          epoch_loss_sum += loss
                          optimizer_steps_in_epoch += 1
                          global_step_counter += 1 # Increment global step only on optimizer step

                          # Log progress at intervals based on optimizer steps
                          if optimizer_steps_in_epoch % args.log_interval == 0 and is_main_process:
                              current_lr = optimizer.param_groups[0]['lr']
                              avg_loss_so_far = epoch_loss_sum / optimizer_steps_in_epoch
                              log_msg = (f"Epoch {epoch+1} | Step {optimizer_steps_in_epoch}/{approx_total_steps_per_epoch // args.grad_accum_steps} | "
                                         f"Loss: {loss:.4f} | Avg Loss: {avg_loss_so_far:.4f} | LR: {current_lr:.6e}")
                              logging.info(log_msg)
                              if isinstance(progress_bar, tqdm):
                                   progress_bar.set_postfix({"Loss": f"{loss:.3f}", "Avg Loss": f"{avg_loss_so_far:.3f}", "LR": f"{current_lr:.2e}"})

                              if args.wandb and WANDB_AVAILABLE:
                                  wandb.log({
                                      "train/loss_step": loss,
                                      "train/avg_epoch_loss_so_far": avg_loss_so_far,
                                      "train/learning_rate": current_lr,
                                      "epoch": epoch + 1,
                                      "optimizer_step_in_epoch": optimizer_steps_in_epoch,
                                      "global_step": global_step_counter
                                  })

                 except Exception as batch_ex:
                      logging.error(f"Error processing batch {i} in epoch {epoch}: {batch_ex}", exc_info=True)
                      # Option: skip batch or re-raise
                      continue # Skip to next batch

                 batches_processed_in_epoch += 1
                 # Optional: Break loop if exceeding estimated steps (safety net)
                 # if approx_total_steps_per_epoch > 0 and i >= approx_total_steps_per_epoch -1 :
                 #      break

            # --- End of Inner Loop (Epoch Training Steps) ---
            if isinstance(progress_bar, tqdm):
                 progress_bar.close()

            # Calculate average epoch loss (ensure division by non-zero count)
            if optimizer_steps_in_epoch > 0:
                 avg_epoch_loss = epoch_loss_sum / optimizer_steps_in_epoch
                 if is_main_process:
                      logging.info(f"Epoch {epoch + 1} training completed. Average Loss: {avg_epoch_loss:.4f}")
                      if args.wandb and WANDB_AVAILABLE:
                           wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch + 1})
            else:
                 logging.warning(f"Epoch {epoch + 1} completed with 0 optimizer steps.")
                 avg_epoch_loss = 0.0

            # --- Validation and Checkpointing (on main process) ---
            if is_main_process:
                val_metrics = {}
                if val_loader: # Check if val_loader exists
                    try:
                        logging.info(f"Starting validation for epoch {epoch + 1}...")
                        val_metrics = trainer.validate(val_loader)
                        logging.info(f"Epoch {epoch + 1} validation | Metrics: {val_metrics}")

                        if args.wandb and WANDB_AVAILABLE:
                            # Prefix validation metrics for clarity in wandb
                            wandb_val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                            wandb_val_metrics["epoch"] = epoch + 1
                            wandb.log(wandb_val_metrics)

                    except Exception as e:
                        logging.error(f"Error during validation for epoch {epoch + 1}: {e}", exc_info=True)
                else:
                     logging.info("Skipping validation as no validation loader was provided.")


                # --- Save Checkpoint ---
                if args.checkpoint_dir:
                     checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt") # Save completed epoch
                     try:
                         trainer.save_checkpoint(checkpoint_path, epoch + 1, val_metrics) # Save completed epoch number
                     except Exception as save_ex:
                         logging.error(f"Failed to save end-of-epoch checkpoint {checkpoint_path}: {save_ex}", exc_info=True)


    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Unhandled error during training loop: {e}", exc_info=True)
    finally:
        # --- Final Checkpoint Save ---
        if args.checkpoint_dir and is_main_process:
            final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
            try:
                 # Determine the last completed epoch index correctly
                 current_final_epoch = epoch + 1 if 'epoch' in locals() else start_epoch
                 logging.info(f"Saving final checkpoint for completed epoch {current_final_epoch}...")
                 trainer.save_checkpoint(final_checkpoint_path, current_final_epoch)
            except Exception as save_ex:
                 logging.error(f"Failed to save final checkpoint {final_checkpoint_path}: {save_ex}", exc_info=True)


        # Clean up distributed training
        if is_distributed and is_initialized():
            destroy_process_group()
            logging.info("Destroyed process group.")

        # Finish wandb run
        if args.wandb and WANDB_AVAILABLE and wandb.run:
            wandb.finish()

    logging.info("Training script finished.")

# =====================================================================
# Main Function
# =====================================================================

def main():
    """Main function."""
    args = parse_args()

    # Set up logging (configure once)
    log_handlers = [logging.StreamHandler()]
    # Add file handler only on main process if distributed setup allows checking rank early,
    # otherwise, log to file on all ranks but maybe add rank info to filename/log format.
    # For simplicity here, adding file handler always, but be mindful of multiple logs in DDP.
    try:
        log_filename = f"bytropix_stabilized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        # Optional: Add rank info if distributed
        if int(os.environ.get("RANK", -1)) != -1:
             log_filename = f"bytropix_stabilized_rank{os.environ['RANK']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_handlers.append(logging.FileHandler(log_filename))
    except Exception as log_ex:
         print(f"Warning: Could not create file handler for logging: {log_ex}")


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # Added file/line info
        handlers=log_handlers,
        force=True # Override any root logger configs set by libraries
    )

    # Log args
    logging.info(f"Script started with Arguments: {args}")
    logging.info(f"Torch version: {torch.__version__}")
    logging.info(f"Platform: {platform.system()} - {platform.release()}")
    logging.info(f"Hostname: {socket.gethostname()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA available. Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
             logging.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.info("CUDA not available.")


    # Start training
    train(args)


if __name__ == "__main__":
    main()
