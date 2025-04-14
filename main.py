#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bytropix - Byte Latent Transformer with Babylon Index and Q-Learning Optimization
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
import wandb
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
        return bytes(byte_sequence).decode('utf-8', errors='replace')


class ByteIterableDataset(IterableDataset):
    """Byte-level dataset for efficient streaming from large text files."""
    def __len__(self):
        """Return the size of the dataset."""
        # Determine how many complete context windows we can form
        return max(0, (self.data_size - self.context_size - 1))
        
    def __init__(self, npy_file_path: str, context_size: int = 128):
        """
        Initialize the dataset.
        
        Args:
            npy_file_path: Path to numpy array file containing byte data
            context_size: Size of context window
        """
        self.npy_file_path = npy_file_path
        self.context_size = context_size
        
        # Validate file exists
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"File not found: {npy_file_path}")
        
        # Get size for efficient streaming
        self.data_size = np.load(npy_file_path, mmap_mode='r').shape[0]
    
    def __iter__(self):
        """Generate (context, target) pairs from byte data."""
        # Memory-map the file for efficient access
        bytes_data = np.load(self.npy_file_path, mmap_mode='r')
        data_len = len(bytes_data)
        
        # Determine valid starting positions (account for context size and target)
        valid_starts = max(0, data_len - self.context_size - 1)
        
        # Generate random indices for starting positions
        indices = torch.randperm(valid_starts).tolist()
        
        for idx in indices:
            # Ensure we don't go out of bounds
            if idx + self.context_size + 1 <= data_len:
                context = bytes_data[idx:idx + self.context_size]
                target = bytes_data[idx + self.context_size]
                
                # Convert to tensors
                context_tensor = torch.tensor(context, dtype=torch.long)
                target_tensor = torch.tensor(target, dtype=torch.long)
                
                yield context_tensor, target_tensor


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
        """Compute rolling hashes with corresponding windows."""
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
        """Compute Shannon entropy of byte window."""
        # Use numpy for efficient counting
        byte_counts = np.bincount(byte_window, minlength=256)
        total_bytes = byte_counts.sum()
        
        # Handle zero counts
        probs = byte_counts[byte_counts > 0] / total_bytes
        return float(-np.sum(probs * np.log2(probs)))
    
    def get_window_features(self, window: List[int]) -> Dict[str, float]:
        """Extract additional features from byte window."""
        arr = np.array(window)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'unique_ratio': len(np.unique(arr)) / len(arr),
            'max_run': max(len(list(g)) for _, g in itertools.groupby(window))
        }
    
    def prioritize(self, byte_sequence: List[int]) -> List[Tuple[int, Dict[str, float]]]:
        """Prioritize sequence regions based on entropy and features."""
        self._clean_cache()
        
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
        
        # Use the BabylonIndex's prioritize method to sort boundaries
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
        self.max_gradient_norm = 0
        self.sum_clip_ratios = 0
        self.step_stats = {}
    
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: float = None):
        self.total_gradients += 1
        if clipped:
            self.clipped_gradients += 1
            if clip_ratio:
                self.sum_clip_ratios += clip_ratio
        self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
    
    def get_step_stats(self) -> dict:
        if self.total_gradients == 0:
            return {
                "gradients_clipped": 0,
                "total_gradients": 0,
                "clip_ratio": 0,
                "max_gradient": 0,
                "avg_clip_amount": 0
            }
            
        return {
            "gradients_clipped": self.clipped_gradients,
            "total_gradients": self.total_gradients,
            "clip_ratio": self.clipped_gradients / self.total_gradients,
            "max_gradient": self.max_gradient_norm,
            "avg_clip_amount": self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0
        }
    
    def record_step(self, step: int):
        stats = self.get_step_stats()
        self.step_stats[step] = stats
        self.reset()
        return stats


class QController:
    """Q-Learning Controller with adaptive exploration and state management."""
    
    def __init__(
        self,
        learning_rate: float = 0.02,
        discount: float = 0.97,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.999,
        initial_mix_prob: float = 0.9,
        lr_scale_bounds: tuple = (0.85, 1.15),
        momentum_scale_bounds: tuple = (0.9, 1.1),
        min_weight_decay: float = 1e-4,
        state_memory_size: int = 300,
        max_q_table_size: int = 15000
    ):
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.mix_prob = initial_mix_prob
        self.prev_loss = None
        self.prev_state = None
        self.prev_action = None
        self.lr_scale_bounds = lr_scale_bounds
        self.momentum_scale_bounds = momentum_scale_bounds
        self.min_weight_decay = min_weight_decay

        # Enhanced state tracking
        self.state_memory = deque(maxlen=state_memory_size)
        self.loss_window = deque(maxlen=15)
        self.grad_window = deque(maxlen=15)
        self.lr_window = deque(maxlen=8)
        self.momentum_window = deque(maxlen=8)
        
        # Action space
        self.action_ranges = {
            'lr_scale': np.array([0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15]),
            'momentum_scale': np.array([0.9, 0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05, 1.075, 1.1])
        }

        # Success tracking with decay
        self.action_success = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        self.action_counts = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        self.success_decay = 0.99

        # Performance tracking
        self.performance_window = deque(maxlen=50)
        self.stable_steps = 0

        # Q-Table memory management
        self.max_q_table_size = max_q_table_size
    
    def get_state(self, lr: float, momentum: float, grad_var: float, loss: float) -> tuple:
        """Enhanced state representation with more nuanced binning."""
        self.loss_window.append(loss)
        self.grad_window.append(grad_var)
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)
        
        # Improved loss trend calculation with exponential moving average
        if len(self.loss_window) >= 3:
            weights = np.exp([-0.1 * i for i in range(3)])
            weights = weights / weights.sum()
            recent_losses = list(self.loss_window)[-3:]
            weighted_loss_trend = np.sum(weights * [(recent_losses[0] - x) / recent_losses[0] for x in recent_losses[1:]])
            loss_trend_bin = np.digitize(weighted_loss_trend, 
                                            bins=[-0.1, -0.03, -0.01, 0.01, 0.03, 0.1])
        else:
            loss_trend_bin = 3  # Middle bin
        
        # Improved gradient stability metric using exponential moving average
        if len(self.grad_window) >= 3:
            recent_grads = list(self.grad_window)[-3:]
            weights = np.exp([-0.1 * i for i in range(3)])
            weights = weights / weights.sum()
            weighted_grad_mean = np.sum(weights * recent_grads)
            grad_stability = np.std(recent_grads) / (weighted_grad_mean + 1e-8)
            grad_stability_bin = np.digitize(grad_stability, 
                                                bins=[0.05, 0.1, 0.2, 0.3, 0.5])
        else:
            grad_stability_bin = 3
        
        # Learning rate bins with relative scaling
        lr_base = 0.001
        lr_relative = lr / lr_base
        lr_bin = np.digitize(lr_relative, 
                                 bins=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        
        # Momentum bins
        momentum_bin = np.digitize(momentum, 
                                    bins=[0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0])
        
        return (lr_bin, momentum_bin, loss_trend_bin, grad_stability_bin)
    
    def compute_reward(
        self,
        loss_trend: float,
        grad_health: float,
        consistent_improvement: bool
    ) -> float:
        """Enhanced reward computation with stability bonus."""
        # Base reward from loss improvement
        base_reward = 2.0 * loss_trend if loss_trend > 0 else 1.5 * loss_trend
        
        # Enhanced stability bonus
        stability_threshold = 0.9
        stability_bonus = 0.0
        if grad_health > stability_threshold:
            bonus_scale = (grad_health - stability_threshold) / (1 - stability_threshold)
            stability_bonus = 0.5 * bonus_scale
        
        # Progressive consistency reward
        if consistent_improvement:
            self.stable_steps += 1
            consistency_reward = min(1.0, 0.2 * np.log1p(self.stable_steps))
        else:
            self.stable_steps = max(0, self.stable_steps - 1)
            consistency_reward = 0.0
        
        # Combine rewards with dynamic weighting
        combined_reward = (
            base_reward +
            stability_bonus +
            consistency_reward
        )
        
        # Smooth reward scaling
        scaled_reward = np.tanh(combined_reward)
        
        return float(scaled_reward)
    
    def choose_action(self, state: tuple) -> Dict[str, float]:
        """Enhanced action selection with adaptive exploration."""
        if state not in self.q_table:
            self.q_table[state] = {
                param: np.zeros(len(space))
                for param, space in self.action_ranges.items()
            }

        action = {}
        
        # Compute recent performance
        recent_rewards = [r for _, _, r in list(self.state_memory)[-20:]]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        # Adaptive exploration rates
        explore_lr = np.random.random() < self.epsilon * (1.0 - max(0, avg_reward))
        explore_momentum = np.random.random() < self.epsilon * 0.7 * (1.0 - max(0, avg_reward))

        for param, space in self.action_ranges.items():
            should_explore = explore_lr if param == 'lr_scale' else explore_momentum
            
            if should_explore:
                # Smart exploration based on action success history
                success_rates = self.action_success[param] / (self.action_counts[param] + 1)
                
                if len(recent_rewards) > 0 and np.mean(recent_rewards) < -0.2:
                    # If performing poorly, explore more widely but favor previously successful actions
                    p = self._softmax(success_rates + 0.1)
                    chosen_idx = np.random.choice(len(space), p=p)
                else:
                    # If performing well, explore near current value
                    current_idx = np.argmin(np.abs(space - 1.0))
                    exploration_range = 2
                    valid_indices = np.arange(
                        max(0, current_idx - exploration_range),
                        min(len(space), current_idx + exploration_range + 1)
                    )
                    chosen_idx = np.random.choice(valid_indices)
                
                chosen_action = float(space[chosen_idx])
            else:
                # Enhanced greedy selection with success history influence
                q_values = self.q_table[state][param]
                success_rates = self.action_success[param] / (self.action_counts[param] + 1)
                
                # Combine Q-values with success history
                combined_values = q_values + 0.2 * success_rates
                max_val = np.max(combined_values)
                best_actions = np.where(np.abs(combined_values - max_val) < 1e-6)[0]
                
                # Choose action closest to 1.0 when tied
                chosen_idx = min(best_actions, key=lambda i: abs(space[i] - 1.0))
                chosen_action = float(space[chosen_idx])
            
            action[param] = chosen_action

        # Adaptive epsilon decay based on performance
        if len(self.performance_window) > 20:
            avg_performance = np.mean(self.performance_window)
            if avg_performance > 0.3:
                self.epsilon = max(0.05, self.epsilon * 0.997)
            else:
                self.epsilon = max(0.05, self.epsilon * 0.995)

        return action
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for scores."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: Optional[tuple], should_log: bool = False):
        """Enhanced Q-learning update with experience tracking."""
        # Record reward and update state memory
        self.state_memory.append((state, action, reward))
        if should_log:
            logging.info(f"State memory updated with state={state}, action={action}, reward={reward}")

        # Initialize Q-values if needed
        for s in [state, next_state] if next_state is not None else [state]:
            if s not in self.q_table:
                self.q_table[s] = {
                    param: np.zeros(len(space))
                    for param, space in self.action_ranges.items()
                }

        # Q-Table memory management
        if len(self.q_table) > self.max_q_table_size:
            oldest_state = self.state_memory.popleft()[0]
            if oldest_state in self.q_table:
                del self.q_table[oldest_state]
                if should_log:
                    logging.info(f"Evicted oldest state from Q-table: {oldest_state}")

        # Update Q-values with adaptive learning rate
        for param, value in action.items():
            space = self.action_ranges[param]
            action_idx = np.abs(space - value).argmin()

            # Get max future Q-value
            if next_state is not None and next_state in self.q_table:
                next_q_values = self.q_table[next_state][param]
                max_future_q = np.max(next_q_values)
            else:
                max_future_q = 0.0

            # Current Q-value
            current_q = self.q_table[state][param][action_idx]

            # Compute TD error with less aggressive clipping
            td_error = reward + self.gamma * max_future_q - current_q
            td_error = np.clip(td_error, -1.0, 1.0)

            # Adaptive learning rate based on visit count
            self.action_counts[param][action_idx] += 1
            visit_count = self.action_counts[param][action_idx]
            effective_lr = self.alpha / (1 + np.log1p(visit_count) * 0.1)

            # Update Q-value
            self.q_table[state][param][action_idx] += effective_lr * td_error

            if should_log:
                logging.info(f"Updated Q-table for state={state}, param={param}, action_idx={action_idx}: "
                             f"current_q={current_q:.4f}, max_future_q={max_future_q:.4f}, td_error={td_error:.4f}, "
                             f"effective_lr={effective_lr:.6f}")

            # Update success tracking with decay
            if reward > 0:
                self.action_success[param][action_idx] = (self.action_success[param][action_idx] * self.success_decay) + 1
            else:
                self.action_success[param][action_idx] *= self.success_decay


class EnhancedSGD(torch.optim.Optimizer):
    """Enhanced SGD Optimizer with Q-Learning based Adaptive Hyperparameter Tuning."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.003,
        momentum: float = 0.9,
        weight_decay: float = 0.005,
        smoothing_factor: float = 0.05,
        entropy_threshold: float = 0.3,
        max_grad_norm: float = 1.0,
        noise_scale: float = 0.001,
        lr_scale_bounds: tuple = (0.7, 1.3),
        momentum_scale_bounds: tuple = (0.85, 1.1),
        q_learning_config: Dict[str, Any] = {},
        **kwargs
    ):
        """
        Initializes the EnhancedSGD optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Initial learning rate.
            momentum: Initial momentum.
            weight_decay: Initial weight decay.
            smoothing_factor: Smoothing factor for updates.
            entropy_threshold: Threshold for entropy-based adjustments.
            max_grad_norm: Maximum gradient norm for clipping.
            noise_scale: Scale of noise to inject.
            lr_scale_bounds: Bounds for learning rate scaling.
            momentum_scale_bounds: Bounds for momentum scaling.
            q_learning_config: Configuration for Q-Learning controller.
            **kwargs: Additional keyword arguments.
        """
        # Ensure params is properly formatted as parameter groups
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups = params
        else:
            param_groups = [{'params': params}]
        
        # Enhanced parameter group initialization
        for group in param_groups:
            group.setdefault('lr', lr)
            group.setdefault('momentum', momentum)
            group.setdefault('weight_decay', weight_decay)
            group.setdefault('base_lr', lr)
            group.setdefault('q_scale', 1.0)
            # Add minimum weight decay
            group.setdefault('min_weight_decay', weight_decay * 0.2)
        
        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        
        # Initialize optimization state
        self._init_optimization_state(
            smoothing_factor=smoothing_factor,
            entropy_threshold=entropy_threshold,
            max_grad_norm=max_grad_norm,
            noise_scale=noise_scale,
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            q_learning_config=q_learning_config,
            **kwargs
        )
        
        # Pre-allocate buffers with proper device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = {
            'grad_norms': deque(maxlen=100),
            'learning_rates': deque(maxlen=100),
            'momentum_values': deque(maxlen=100),
            'entropy_values': deque(maxlen=100),
            'update_norms': deque(maxlen=100)
        }

    def _init_optimization_state(self, **kwargs):
        """Initialize optimization state with safe handling."""
        smoothing_factor = kwargs.get('smoothing_factor', 0.05)
        entropy_threshold = kwargs.get('entropy_threshold', 0.3)
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        noise_scale = kwargs.get('noise_scale', 0.001)
        lr_scale_bounds = kwargs.get('lr_scale_bounds', (0.7, 1.3))
        momentum_scale_bounds = kwargs.get('momentum_scale_bounds', (0.85, 1.1))
        q_learning_config = kwargs.get('q_learning_config', {})

        self.max_grad_norm = max_grad_norm

        self.q_controller = QController(
            learning_rate=q_learning_config.get('learning_rate', 0.02),
            discount=q_learning_config.get('discount', 0.97),
            epsilon=q_learning_config.get('epsilon', 0.15),
            epsilon_decay=q_learning_config.get('epsilon_decay', 0.999),
            initial_mix_prob=q_learning_config.get('initial_mix_prob', 0.9),
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            min_weight_decay=q_learning_config.get('min_weight_decay', 1e-4)
        )
        
        self._step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_loss = None
        
        # Initialize gradient memory
        self.grad_memory = deque(maxlen=100)
        
        # Initialize GradientStats
        self.gradient_stats = GradientStats()

    def _track_gradient_memory(self, grad_norm: float):
        """Track gradient history for better adaptation."""
        self.grad_memory.append(grad_norm)

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Efficient entropy computation using torch operations with safe calculations."""
        if tensor.numel() <= 1:
            return 0.0
        
        values = tensor.flatten()
        # Safe histogram calculation
        hist = torch.histc(values, bins=min(100, values.numel()))
        # Add small epsilon to prevent log(0)
        eps = 1e-7
        hist = hist / (hist.sum() + eps) + eps
        return float(-torch.sum(hist * torch.log(hist)).item())

    def get_statistics(self) -> Dict[str, float]:
        """Compute and return statistics using pre-allocated deques."""
        stats = {}

        # Add current learning rate from first param group
        if len(self.param_groups) > 0:
            stats['current_lr'] = float(self.param_groups[0]['lr'])

        # Add other statistics
        for key, values in self.stats.items():
            if values:
                try:
                    tensor_values = torch.tensor(list(values), dtype=torch.float32)
                    stats[f'avg_{key}'] = float(torch.mean(tensor_values).item())
                    stats[f'std_{key}'] = float(torch.std(tensor_values).item())
                except (RuntimeError, ValueError) as e:
                    logging.warning(f"Error computing statistics for {key}: {e}")
                    stats[f'avg_{key}'] = 0.0
                    stats[f'std_{key}'] = 0.0
        return stats

    def _get_gradient_stats(self) -> Dict[str, Any]:
        """Gather gradient statistics for the current step."""
        grad_norms = []
        grad_vars = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_norms.append(torch.norm(grad).item())
                if p.grad.numel() > 1:
                    grad_vars.append(torch.var(grad.float()).item())
        
        saw_grads = len(grad_norms) > 0
        
        if saw_grads:
            mean_grad_norm = np.mean(grad_norms)
            mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
        else:
            mean_grad_norm = 0.0
            mean_grad_var = 0.0
        
        grad_stats = {
            'saw_grads': saw_grads,
            'mean_grad_norm': mean_grad_norm,
            'mean_grad_var': mean_grad_var
        }
        return grad_stats
        
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """Optimizes the parameters based on the current gradient."""
        loss = None
        if closure is not None:
            loss = closure()

        # Get gradient statistics
        grad_stats = self._get_gradient_stats()

        # Apply Q-learning adjustments if we have gradients and loss
        if grad_stats['saw_grads'] and loss is not None:
            current_loss = loss.item()

            # Get current state
            q_state = self.q_controller.get_state(
                lr=self.param_groups[0]['lr'],
                momentum=self.param_groups[0]['momentum'],
                grad_var=grad_stats['mean_grad_var'],
                loss=current_loss
            )

            # Update Q-table with previous experience if available
            if self.q_controller.prev_loss is not None and \
               self.q_controller.prev_state is not None and \
               self.q_controller.prev_action is not None:
                # Calculate relative loss improvement
                loss_improvement = (self.q_controller.prev_loss - current_loss) / self.q_controller.prev_loss
                grad_health = 1.0 / (1.0 + grad_stats['mean_grad_var'])
                
                # Check for consistent improvement
                self.q_controller.performance_window.append(loss_improvement)
                consistent_improvement = all([r > 0 for r in list(self.q_controller.performance_window)[-10:]])

                # Compute reward for the previous action
                reward = self.q_controller.compute_reward(
                    loss_trend=loss_improvement,
                    grad_health=grad_health,
                    consistent_improvement=consistent_improvement
                )
                
                # Update Q-table with the previous state, action, and received reward
                self.q_controller.update(
                    state=self.q_controller.prev_state,
                    action=self.q_controller.prev_action,
                    reward=reward,
                    next_state=q_state,
                    should_log=(self._step_count % 10 == 0)
                )

            # Choose new action based on the current state
            q_action = self.q_controller.choose_action(q_state)

            # Apply learning rate and momentum adjustments
            for group in self.param_groups:
                # Scale learning rate
                group['q_scale'] *= float(np.clip(
                    q_action['lr_scale'],
                    self.q_controller.lr_scale_bounds[0],
                    self.q_controller.lr_scale_bounds[1]
                ))
                group['lr'] = group['base_lr'] * group['q_scale']

                # Scale momentum
                group['momentum'] = float(np.clip(
                    group['momentum'] * q_action['momentum_scale'],
                    self.q_controller.momentum_scale_bounds[0],
                    self.q_controller.momentum_scale_bounds[1]
                ))

            # Update Q-learning state for the next step
            self.q_controller.prev_state = q_state
            self.q_controller.prev_action = q_action
            self.q_controller.prev_loss = current_loss

        # Apply updates with the adjusted learning rates
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad_stats['mean_grad_norm'] > self.max_grad_norm * 5 or \
                   np.isnan(grad_stats['mean_grad_norm']) or \
                   np.isnan(grad_stats['mean_grad_var']):
                    grad = torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm)

                state = self.state[p]

                # Initialize momentum buffer if needed
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                # Apply update with current learning rate
                self._apply_update(
                    p=p,
                    grad=grad,
                    momentum=group['momentum'],
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    state=state,
                    current_loss=self.q_controller.prev_loss if self.q_controller.prev_loss is not None else 0.0
                )

        if grad_stats['saw_grads']:
            self._step_count += 1
            # Record and log gradient clipping statistics
            stats = self.gradient_stats.record_step(self._step_count)
            if self._step_count % 10 == 0:
                logging.info(
                    f"Step {self._step_count} gradient stats: "
                    f"Clipped {stats['gradients_clipped']}/{stats['total_gradients']} "
                    f"({stats['clip_ratio']:.1%}) gradients. "
                    f"Max norm: {stats['max_gradient']:.3f}, "
                    f"Avg clip amount: {stats['avg_clip_amount']:.3f}"
                )

        return loss
        
    def _apply_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        momentum: float,
        lr: float,
        weight_decay: float,
        state: dict,
        current_loss: float
    ):
        """Enhanced parameter update with improved gradient handling."""
        buf = state['momentum_buffer']

        # Apply weight decay with smoother sigmoid scaling
        if weight_decay != 0:
            param_norm = torch.norm(p)
            # Use sigmoid for smoother transitions
            adaptive_wd = weight_decay * (1.0 / (1.0 + torch.exp(-param_norm * 0.05)))
            grad = grad.add(p, alpha=adaptive_wd)

        # Apply gradient clipping before updating the momentum buffer
        grad_norm = torch.norm(grad)
        was_clipped = False
        clip_ratio = 0

        if grad_norm > self.max_grad_norm:
            clip_ratio = float(self.max_grad_norm / grad_norm)
            grad.mul_(clip_ratio)
            was_clipped = True

        # Record gradient clipping statistics
        self.gradient_stats.record_gradient(
            original_norm=float(grad_norm),
            clipped=was_clipped,
            clip_ratio=clip_ratio
        )

        # Update momentum buffer
        buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

        # Compute update
        update = -lr * buf

        # Apply gradient clipping on parameter update
        update_norm = torch.norm(update).item()
        max_update_norm = 1.0  # Prevent too large updates
        if update_norm > max_update_norm:
            update.mul_(max_update_norm / (update_norm + 1e-6))

        # Update parameters
        p.data.add_(update)
        
    def state_dict(self) -> Dict[str, Any]:
        """Returns the optimizer's state dict with safe serialization."""
        state_dict = super().state_dict()
        try:
            state_dict['statistics'] = self.get_statistics()
            state_dict['q_table'] = self.q_controller.q_table
            state_dict['epsilon'] = float(self.q_controller.epsilon)
        except Exception as e:
            logging.error(f"Error creating state dict: {e}")
            state_dict['statistics'] = {}
            state_dict['q_table'] = {}
            state_dict['epsilon'] = 0.15
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads optimizer state with safe type handling."""
        try:
            statistics = state_dict.pop('statistics', None)
            q_table = state_dict.pop('q_table', None)
            epsilon = state_dict.pop('epsilon', None)

            super().load_state_dict(state_dict)

            if statistics is not None:
                for key in self.stats:
                    avg_key = f'avg_{key}'
                    if avg_key in statistics:
                        # Fill the deque with the average value for simplicity
                        self.stats[key].extend([float(statistics[avg_key])] * self.stats[key].maxlen)

            if q_table is not None:
                self.q_controller.q_table = q_table

            if epsilon is not None:
                self.q_controller.epsilon = float(epsilon)

        except Exception as e:
            logging.error(f"Error loading state dict: {e}")
            # Initialize fresh statistics and Q-table if loading fails
            self.stats = {
                'grad_norms': deque(maxlen=100),
                'learning_rates': deque(maxlen=100),
                'momentum_values': deque(maxlen=100),
                'entropy_values': deque(maxlen=100),
                'update_norms': deque(maxlen=100)
            }
            self.q_controller.q_table = {}
            self.q_controller.epsilon = 0.15


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
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            # Expand mask to match attention scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~attention_mask, float('-inf'))

        # Convert scores to probabilities
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        output = torch.matmul(attn_probs, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        output = self.dropout(output)

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
            attn_output, _ = layer['attention'](
                query=normed,
                key=normed,
                value=normed,
                attn_mask=mask,
                need_weights=False
            )
            x = x + attn_output

            # Pre-LayerNorm for MLP
            normed = layer['norm2'](x)

            # MLP block
            x = x + layer['mlp'](normed)

        return self.final_norm(x)


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
        self.n_gram_vocab_size = n_gram_vocab_size

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

            # Compute hash: sum(byte * (256^i)) mod 2^32
            hash_vals = (n_gram * powers).sum(dim=-1) % 2**32
            # Map hash_vals to [0, n_gram_vocab_size - 1]
            hash_vals = hash_vals % self.n_gram_vocab_size

            hashes[n] = hash_vals

        return hashes

    def create_local_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local window attention mask with proper bounds checking."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        window_size = min(self.window_size, seq_len)

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
            return torch.zeros((batch_size, 1, self.byte_embeddings.embedding_dim), 
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


class LocalDecoder(nn.Module):
    """Local decoder that maps patches back to bytes."""
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 4,
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
        next_byte = torch.gather(top_k_indices, -1, next_byte_idx).squeeze(-1)

        return next_byte


# =====================================================================
# BLTModel: Complete Model Architecture
# =====================================================================

class BLTModel(nn.Module):
    """Complete Byte Latent Transformer model implementation."""
    def __init__(
        self,
        local_hidden_size: int = 256,
        global_hidden_size: int = 1024,
        num_local_encoder_layers: int = 1,
        num_global_layers: int = 16,
        num_local_decoder_layers: int = 4,
        dropout: float = 0.1,
        window_size: int = 256,
        n_gram_sizes: List[int] = [3, 4],
        n_gram_vocab_size: int = 30000
    ):
        super().__init__()

        # Enable memory-efficient attention if available
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
        history_len = 1  # For single-byte prediction
        causal_mask = torch.ones(history_len, history_len, device=device).bool()

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
        targets = targets.view(-1)          # [(batch_size * 1)]

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
        
    def generate(
        self,
        seed_bytes: torch.Tensor,  
        max_length: int = 100,
        temperature: float = 1.0,
        sampling_config: SamplerConfig = None
    ) -> torch.Tensor:
        """
        Generate new bytes starting from seed bytes.
        
        Args:
            seed_bytes: Starting byte sequence [batch_size, seed_len]
            max_length: Maximum number of bytes to generate
            temperature: Sampling temperature
            sampling_config: Config for entropy-based sampling
            
        Returns:
            Generated byte sequence [batch_size, seed_len + max_length]
        """
        device = seed_bytes.device
        batch_size, seed_len = seed_bytes.size()
        generated = seed_bytes.clone()
        
        # Use default sampling config if not provided
        if sampling_config is None:
            sampling_config = SamplerConfig()
            
        # Reset context for generation
        self.patcher.reset_context()
        
        # Generate bytes one by one
        for _ in range(max_length):
            # Get context window
            if generated.size(1) <= self.context_size:
                context = generated
            else:
                context = generated[:, -self.context_size:]
                
            # Get next byte predictions
            with torch.no_grad():
                logits = self(context)  # [batch_size, 1, 256]
                next_byte_logits = logits[:, -1]  # [batch_size, 256]
                
                # Apply temperature
                scaled_logits = next_byte_logits / temperature
                
                # Sample based on entropy
                probs = F.softmax(scaled_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
                
                # Low entropy: greedy sampling
                low_mask = entropy < sampling_config.low_entropy_threshold
                med_mask = (entropy >= sampling_config.low_entropy_threshold) & (entropy < sampling_config.medium_entropy_threshold)
                high_mask = entropy >= sampling_config.medium_entropy_threshold
                
                next_bytes = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                # Apply different sampling strategies based on entropy
                if low_mask.any():
                    next_bytes[low_mask] = torch.argmax(probs[low_mask], dim=-1)
                    
                if med_mask.any():
                    top_k = 10
                    top_k_probs, top_k_indices = torch.topk(probs[med_mask], k=top_k, dim=-1)
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                    sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
                    next_bytes[med_mask] = top_k_indices.gather(1, sampled_indices.unsqueeze(-1)).squeeze(-1)
                    
                if high_mask.any():
                    next_bytes[high_mask] = torch.multinomial(probs[high_mask], num_samples=1).squeeze(-1)
                
            # Append to generated sequence
            next_bytes = next_bytes.unsqueeze(1)  # [batch_size, 1]
            generated = torch.cat([generated, next_bytes], dim=1)
            
        return generated
        
        
# =====================================================================
# Trainer Class
# =====================================================================

class RLHFTrainer:
    """Trainer class that integrates Q-Learning with the training loop."""
    def __init__(
        self,  
        model: nn.Module,  
        optimizer: torch.optim.Optimizer,  
        device: torch.device,
        scaler = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler if scaler is not None else amp.GradScaler(enabled=True)
        self._step_count = 0
        
    def train_step(
        self,  
        context: torch.Tensor,  
        target: torch.Tensor,  
        accumulation_steps: int = 1
    ) -> float:
        """
        Perform a single training step with gradient accumulation.
        
        Args:
            context: Input byte sequences [batch_size, context_size]
            target: Target bytes [batch_size]
            accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            loss: Loss value
        """
        self.model.train()
        
        with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            # Forward pass
            logits = self.model(context)  # [batch_size, 1, 256]
            loss = self.model.compute_loss(logits, target) / accumulation_steps
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step
        if (self._step_count + 1) % accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
        self._step_count += 1
        
        return loss.item()
    
    def validate(
        self,  
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model on the validation dataset.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        entropy_vals = []
        
        with torch.no_grad():
            for context, target in dataloader:
                context = context.to(self.device)
                target = target.to(self.device)
                
                with amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    logits = self.model(context)
                    loss = self.model.compute_loss(logits, target)
                    
                    # Calculate entropy
                    probs = F.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
                    
                total_loss += loss.item() * context.size(0)
                entropy_vals.append(entropy.item())
            
        avg_loss = total_loss / len(dataloader.dataset)
        avg_entropy = np.mean(entropy_vals)
        
        metrics = {
            'val_loss': avg_loss,
            'val_entropy': avg_entropy
        }
        
        self.model.train()
        return metrics
    
    def save_checkpoint(
        self,  
        filepath: str,  
        epoch: int,  
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Save model checkpoint with optimizer state.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            val_metrics: Validation metrics
        """
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get model state dict, handling DDP if necessary
        if isinstance(self.model, DistributedDataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_metrics': val_metrics
        }
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
        
    def load_checkpoint(self, filepath: str) -> int:
        """
        Load model checkpoint with optimizer state.
        
        Args:
            filepath: Path to checkpoint
            
        Returns:
            epoch: Epoch of checkpoint
        """
        if not os.path.exists(filepath):
            logging.error(f"Checkpoint {filepath} does not exist")
            return 0
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state, handling DDP if necessary
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        logging.info(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch

def parse_args():
    """Parse command-line arguments for Bytropix training."""
    parser = argparse.ArgumentParser(description="Bytropix - Byte Latent Transformer Training")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="C:/projects/bytropix/data/wikitext_train.npy",
                        help="Path to training data numpy file")
    parser.add_argument("--val_data_path", type=str, default="C:/projects/bytropix/data/wikitext_val.npy",
                        help="Path to validation data numpy file")
    parser.add_argument("--context_size", type=int, default=128,
                        help="Context size for training")
    
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
                        help="Window size for local encoding")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,  # Smaller batch size for memory efficiency
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.003,
                        help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--checkpoint_dir", type=str, default="C:/projects/bytropix/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval in steps")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Checkpoint saving interval in steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Distributed training parameters
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="bytropix",
                        help="Weights & Biases project name")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable automatic mixed precision")
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    return args

def train(args):
    """Main training function."""
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize distributed training if needed
    distributed = False
    if args.local_rank != -1:
        if not torch.distributed.is_initialized():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            try:
                init_process_group(backend="nccl")
                distributed = True
                logging.info(f"Initialized distributed training with rank {args.local_rank}")
            except Exception as e:
                logging.warning(f"Failed to initialize distributed training: {e}")
    
    # Initialize wandb if enabled
    if args.wandb and (not distributed or args.local_rank == 0):
        try:
            wandb.init(project=args.wandb_project)
            wandb.config.update(args)
            logging.info("Initialized wandb logging")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            args.wandb = False
    
    # Create datasets
    try:
        logging.info(f"Loading training data from {args.data_path}")
        train_dataset = ByteIterableDataset(args.data_path, context_size=args.context_size)
        
        logging.info(f"Loading validation data from {args.val_data_path}")
        val_dataset = ByteIterableDataset(args.val_data_path, context_size=args.context_size)
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        return
    
    # Create data samplers and loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Log dataset sizes
    logging.info(f"Training data size: {train_dataset.data_size} bytes")
    logging.info(f"Validation data size: {val_dataset.data_size} bytes")
    
    # Create model
    logging.info("Initializing BLTModel")
    model = BLTModel(
        local_hidden_size=args.local_hidden_size,
        global_hidden_size=args.global_hidden_size,
        num_local_encoder_layers=args.num_local_encoder_layers,
        num_global_layers=args.num_global_layers,
        num_local_decoder_layers=args.num_local_decoder_layers,
        window_size=args.window_size
    )
    model.to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Wrap model with DDP if using distributed training
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        logging.info("Model wrapped with DistributedDataParallel")
    
    # Create optimizer with Q-Learning enhanced SGD
    logging.info("Creating EnhancedSGD optimizer")
    optimizer = EnhancedSGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.005,
        q_learning_config={
            "learning_rate": 0.02,
            "discount": 0.97,
            "epsilon": 0.15
        }
    )
    
    # Create GradScaler for AMP
    scaler = amp.GradScaler(enabled=not args.no_amp)
    logging.info(f"Created GradScaler (AMP {'enabled' if not args.no_amp else 'disabled'})")
    
    # Create trainer
    trainer = RLHFTrainer(model, optimizer, device, scaler)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        try:
            start_epoch = trainer.load_checkpoint(args.resume)
            logging.info(f"Resumed from checkpoint {args.resume} at epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            if not os.path.exists(args.resume):
                logging.error(f"Checkpoint file {args.resume} does not exist")
    
    # Training loop
    logging.info(f"Starting training for {args.epochs} epochs")
    try:
        for epoch in range(start_epoch, args.epochs):
            if distributed:
                train_sampler.set_epoch(epoch)
            
            # Train for one epoch
            epoch_loss = 0.0
            for i, (context, target) in enumerate(train_loader):
                context = context.to(device)
                target = target.to(device)
                
                loss = trainer.train_step(context, target, args.grad_accum_steps)
                epoch_loss += loss
                
                global_step = epoch * len(train_loader) + i
                
                # Log progress
                if i % args.log_interval == 0:
                    logging.info(f"Epoch {epoch}, Step {i}/{len(train_loader)}, Loss: {loss:.4f}")
                    if args.wandb and (not distributed or args.local_rank == 0):
                        wandb.log({
                            "loss": loss, 
                            "epoch": epoch, 
                            "step": i,
                            "global_step": global_step,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })
                
                # Save checkpoint
                if (i + 1) % args.save_interval == 0 and (not distributed or args.local_rank == 0):
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                    trainer.save_checkpoint(checkpoint_path, epoch)
                    logging.info(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
            
            # Average epoch loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch} completed with average loss: {avg_epoch_loss:.4f}")
            
            # Validate at the end of epoch
            if not distributed or args.local_rank == 0:
                try:
                    val_metrics = trainer.validate(val_loader)
                    logging.info(f"Epoch {epoch} validation: {val_metrics}")
                    
                    if args.wandb:
                        wandb.log({**val_metrics, "epoch": epoch})
                    
                    # Save epoch checkpoint
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                    trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
                    logging.info(f"Saved checkpoint for epoch {epoch} to {checkpoint_path}")
                except Exception as e:
                    logging.error(f"Error during validation: {e}")
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Error during training: {e}")
    finally:
        # Clean up distributed training
        if distributed and is_initialized():
            destroy_process_group()
        
        # Final checkpoint
        if not distributed or args.local_rank == 0:
            final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
            trainer.save_checkpoint(final_checkpoint_path, epoch)
            logging.info(f"Saved final checkpoint to {final_checkpoint_path}")
    
    logging.info("Training completed")

# =====================================================================
# Main Function
# =====================================================================

def main():
    """Main function."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"bytropix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    # Log args
    logging.info(f"Arguments: {args}")
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()
