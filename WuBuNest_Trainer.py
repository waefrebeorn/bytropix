# -*- coding: utf-8 -*-
"""
WuBu Nesting Model Trainer (v0.03 - True Nesting Integration)

Integrates the WuBu Nesting architecture from wubu_nesting_impl.py,
which uses scale-aware maps, explicit rotations, and transformations
for a more geometrically grounded nesting concept.

Version 0.03 Changes:
- Replaced previous WuBu logic with the full WuBuNestingModel from wubu_nesting_impl.py.
- Incorporated scale-aware hyperbolic maps (log/exp).
- Integrated TangentSpaceRotation and InterLevelTransform modules.
- Utilized BoundaryManifold for learnable boundary points.
- Configuration now directly drives the WuBuNestingModel parameters.
- Data flow: HAKMEM Encoder -> WuBuNestingModel -> HAKMEM Decoder.
- Removed old/placeholder WuBu components.
- Fixed indentation error in Trainer class definition.
- Fixed try-except syntax errors within if blocks and missing except blocks.

Model Flow (v0.03):
1. Input bytes -> HAKMEMBabylonIndex -> Patches
2. Patches -> HAKMEMLocalEncoder -> Initial Euclidean Patch Embeddings [Batch, Patches, EncDim]
3. Euclidean Embeddings -> WuBuNestingModel:
    - Input Projection: EncDim -> Dim_0 (tangent space)
    - Iterates through levels:
        - WuBuNestingLevel processing (combiner, flow, hyperbolic maps) using current geometry (c_i, s_i) and inputs (main vec, relative vecs, descriptor, spread).
        - Stores tangent output v_tangent_i_out.
        - If not last level:
            - Get boundary points from BoundaryManifold.
            - Rotate v_tangent_i_out, boundaries, ld_i_param using TangentSpaceRotation.
            - Transform rotated vectors to next level's tangent space using InterLevelTransform -> v_next_main, v_next_boundaries, ld_next.
            - Calculate relative vectors: d_{i+1} = v_next_main - v_next_boundaries.
            - Aggregate d_{i+1} -> aggregated_relative_vectors_for_next_level.
            - Prepare inputs for next level (current_tangent_main, current_relative_vectors, current_ld_tangent, current_sigma).
    - Aggregate Tangent Space outputs (v_tangent_out from all levels) based on config.
    - Output Projection: AggregatedTangentDim -> DecoderMemoryDim. Output: [Batch, Patches, DecMemDim] (Memory)
4. HAKMEMLocalDecoder(Target Sequence, Memory, MemoryPaddingMask) -> Output Byte Logits
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


# Helper Function for Weight Initialization (Apply to modules)
def init_weights(module):
    """Initialize weights for Linear and Embedding layers."""
    if isinstance(module, nn.Linear):
        # Use Xavier initialization for better variance stability
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm parameters to default values (gamma=1, beta=0)
        if module.elementwise_affine: # Check if learnable params exist
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)


# =====================================================================
# START: Code copied from wubu_nesting_impl.py
# =====================================================================

# --- Hyperbolic Geometry Utilities with Proper Scale Support ---
class HyperbolicUtils:
    """
    Enhanced utility functions for Poincare ball model of hyperbolic geometry.
    Implements scale-aware exponential and logarithmic maps for proper nesting.
    """
    @staticmethod
    def poincare_clip(x: torch.Tensor, c: float, radius: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """Clips points to stay strictly inside the Poincare ball boundary."""
        if c <= 0: return x  # Not hyperbolic if curvature is non-positive
        # Use float32 for curvature calculation stability
        c_float = float(c)
        sqrt_c = math.sqrt(max(c_float, float(eps))) # Ensure c is positive for sqrt

        # Clamp radius to avoid issues near 1.0
        radius = min(radius, 1.0 - float(eps))
        max_norm = (radius / sqrt_c) * (1.0 - float(eps))  # Max Euclidean norm allowed

        # Use float32 for norm calculation for stability, then cast back
        original_dtype = x.dtype
        x_norm_sq = torch.sum(x.float().pow(2), dim=-1, keepdim=True)
        norm = torch.sqrt(torch.clamp(x_norm_sq, min=0) + float(eps)) # Add eps inside sqrt

        cond = norm > max_norm
        # Ensure scale_factor is calculated and applied using the same dtype as x
        scale_factor = torch.where(cond, max_norm / (norm + float(eps)), torch.ones_like(norm)).to(original_dtype)
        clipped_x = x * scale_factor

        # Final sanity check
        if not torch.isfinite(clipped_x).all():
            # Use logger instead of print
            logger.warning("NaN/Inf detected *after* poincare_clip. Replacing with zeros.")
            clipped_x = torch.nan_to_num(clipped_x, nan=0.0)  # Replace with 0
        return clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c: float, scale: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        """
        Maps a tangent vector v at the origin to the Poincare ball with scale awareness.
        Implements the scale-aware version:
        exp_0^c,s(v) = tanh(s * sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||) -> Correction: Original paper formula has /2 inside tanh argument? Let's use tanh(arg) * v / (norm * coeff) structure.
        exp_0^c,s(v) = tanh(s * sqrt(c) * ||v||) * v / (sqrt(c) * ||v||) -- Let's use this consistent form.
        """
        if c <= 0: return v  # No mapping needed for Euclidean/Spherical space
        original_dtype = v.dtype
        v_float = v.float() # Convert to float32 for calculations

        # Compute norm in float32 for stability
        v_norm_sq = torch.sum(v_float.pow(2), dim=-1, keepdim=True)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=0) + float(eps)) # Add eps inside sqrt

        c_float = float(c)
        scale_float = float(scale)
        sqrt_c = math.sqrt(max(c_float, float(eps)))

        # Apply scale to the hyperbolic radius calculation inside the tanh argument
        scaled_hyperbolic_radius_arg = scale_float * sqrt_c * v_norm

        # Use tanh for the scale-aware map
        # Clamp the input to tanh to prevent potential overflow with very large inputs
        tanh_input = torch.clamp(scaled_hyperbolic_radius_arg, min=-30.0, max=30.0)
        tanh_term = torch.tanh(tanh_input).to(original_dtype) # Calculate tanh, cast back

        # Ensure lambda calculation uses consistent dtype
        denominator = (sqrt_c * v_norm + float(eps)).to(original_dtype)
        lambda_v = torch.where(
            v_norm > eps,
            tanh_term / denominator,
            torch.ones_like(v_norm).to(original_dtype) # Avoid 0/0 for zero vectors
        )

        mapped_v = lambda_v * v
        # Clip result to ensure it stays in the ball (using the original curvature c)
        return HyperbolicUtils.poincare_clip(mapped_v, c, eps=float(eps)) # Pass eps for clipping

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c: float, scale: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
        """
        Maps a point y in the Poincare ball back to the tangent space at the origin with scale awareness.
        Implements the scale-aware version:
        log_0^c,s(y) = (1/s) * atanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
        """
        if c <= 0: return y  # No mapping needed for Euclidean/Spherical space
        original_dtype = y.dtype

        # Clip input first to ensure it's strictly inside the ball
        y_clipped = HyperbolicUtils.poincare_clip(y, c, eps=float(eps))
        y_float = y_clipped.float() # Convert to float32 for calculations

        # Compute norm in float32 for stability
        y_norm_sq = torch.sum(y_float.pow(2), dim=-1, keepdim=True)
        y_norm = torch.sqrt(torch.clamp(y_norm_sq, min=0) + float(eps)) # Add eps inside sqrt

        c_float = float(c)
        scale_float = float(scale)
        sqrt_c = math.sqrt(max(c_float, float(eps)))

        # Clamp input to atanh carefully to stay within (-1 + eps, 1 - eps)
        atanh_input_raw = sqrt_c * y_norm
        atanh_input = torch.clamp(atanh_input_raw, min=-1.0 + float(eps), max=1.0 - float(eps))

        # Calculate atanh in float32, cast back
        atanh_term = torch.atanh(atanh_input).to(original_dtype)

        # Apply inverse scale to the hyperbolic radius calculation
        # Division by scale implements the inverse mapping
        scaled_atanh = atanh_term / max(scale_float, float(eps)) # Avoid division by zero

        # Ensure lambda calculation uses consistent dtype
        denominator = (sqrt_c * y_norm + float(eps)).to(original_dtype)
        lambda_y = torch.where(
            y_norm > eps,
            scaled_atanh / denominator,
            torch.ones_like(y_norm).to(original_dtype) # Avoid 0/0 for zero vectors
        )

        mapped_y = lambda_y * y_clipped

        # Handle numerical instabilities
        if not torch.isfinite(mapped_y).all():
            logger.warning("NaN/Inf detected in scale_aware_logarithmic_map output. Replacing with zeros.")
            mapped_y = torch.nan_to_num(mapped_y, nan=0.0)

        return mapped_y

    @staticmethod # Add the standard maps for potential use if needed (e.g. projection)
    def exponential_map(v: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
        """Maps a tangent vector v at the origin to the Poincare ball (exp_0^c(v)). STANDARD (scale=1)"""
        return HyperbolicUtils.scale_aware_exponential_map(v, c, scale=1.0, eps=eps)

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
        """Maps a point y in the Poincare ball back to the tangent space at the origin (log_0^c(y)). STANDARD (scale=1)"""
        return HyperbolicUtils.scale_aware_logarithmic_map(y, c, scale=1.0, eps=eps)

    @staticmethod
    def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-7) -> torch.Tensor:
        """Computes the hyperbolic distance between points x and y in the Poincare ball."""
        if c <= 0: # Using Euclidean distance for c<=0
            return torch.norm(x - y, dim=-1)

        c_float = float(c)
        sqrt_c = math.sqrt(max(c_float, float(eps)))
        radius_clip = 0.999 # Use a slightly tighter radius for distance calculation stability
        # Clip points before calculation
        x_clipped = HyperbolicUtils.poincare_clip(x, c, radius=radius_clip, eps=float(eps))
        y_clipped = HyperbolicUtils.poincare_clip(y, c, radius=radius_clip, eps=float(eps))

        original_dtype = x.dtype
        x_float = x_clipped.float(); y_float = y_clipped.float()

        # Calculate intermediate terms in float32 for stability
        x_norm_sq = torch.sum(x_float.pow(2), dim=-1)
        y_norm_sq = torch.sum(y_float.pow(2), dim=-1)
        diff_norm_sq = torch.sum((x_float - y_float).pow(2), dim=-1)

        denom_x = torch.clamp(1.0 - c_float * x_norm_sq, min=float(eps))
        denom_y = torch.clamp(1.0 - c_float * y_norm_sq, min=float(eps))

        # Add eps to denominator inside arcosh_arg calculation for stability
        arcosh_arg = 1.0 + (2.0 * c_float * diff_norm_sq / (denom_x * denom_y + float(eps)))
        # Ensure arcosh argument is >= 1.0
        arcosh_arg_clamped = torch.clamp(arcosh_arg, min=1.0 + float(eps))

        # Calculate acosh in float32 and cast back
        distance = (1.0 / sqrt_c) * torch.acosh(arcosh_arg_clamped)
        distance = distance.to(original_dtype)

        if not torch.isfinite(distance).all():
            logger.warning(f"NaN/Inf detected in poincare_distance. Input c={c}. Replacing with 100.")
            # Replace non-finite distances with a large but finite number
            distance = torch.nan_to_num(distance, nan=100.0, posinf=100.0, neginf=-100.0) # Return positive large value
        return distance


# --- Quaternion Operations for Tangent Space Rotations ---
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
    """
    Rotates vector v (represented as a quaternion) using p * v * q.
    Handles batching correctly.
    """
    # Ensure dimensions are 4 for quaternion operations
    if v.shape[-1] != 4 or p.shape[-1] != 4 or q.shape[-1] != 4:
        raise ValueError(f"Inputs must be 4D for quat_rotate_via_pvq, shapes: v={v.shape}, p={p.shape}, q={q.shape}")

    # Expand p and q for broadcasting if they have fewer dimensions than v
    # This assumes p and q are single quaternions [4] or broadcastable like [1,1,4]
    if p.dim() < v.dim():
        # Add leading dims to p until its ndim matches v's ndim
        p_view_shape = [1] * (v.dim() - p.dim()) + list(p.shape)
        p = p.view(p_view_shape) # e.g., [4] -> [1, 1, 4] for v=[B, S, 4]
    if q.dim() < v.dim():
        q_view_shape = [1] * (v.dim() - q.dim()) + list(q.shape)
        q = q.view(q_view_shape)

    # Perform rotation: p * v * q (Hamilton product handles broadcasting)
    pv = hamilton_product(p, v)
    pvq = hamilton_product(pv, q)
    return pvq

# --- SO(n) Rotation Implementation ---
class SO_n_Rotation(nn.Module):
    """
    Implements SO(n) rotation matrix using exponential map from skew-symmetric matrices.
    This provides a differentiable parameterization of rotation matrices.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Initialize skew-symmetric parameters close to zero (near-identity rotation)
        self.skew_params = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def _get_rotation_matrix(self) -> torch.Tensor:
        """Constructs a rotation matrix from skew-symmetric parameters."""
        # Create skew-symmetric matrix: A = P - P^T
        skew_matrix = self.skew_params - self.skew_params.T
        # Compute rotation matrix using matrix exponential: R = exp(A)
        # Use float32 for matrix_exp calculation for stability
        R = torch.matrix_exp(skew_matrix.float()).to(self.skew_params.dtype)
        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to inputs. Assumes x contains row vectors.
        Input shape: [..., dim]
        Output shape: [..., dim]
        """
        R = self._get_rotation_matrix()  # [dim, dim]

        # Apply rotation: y = x @ R
        # Handles multi-dimensional inputs by applying to the last dimension
        original_shape = x.shape
        if x.dim() == 1: # Single vector [dim]
            x_rotated = torch.matmul(x.unsqueeze(0), R).squeeze(0)
        elif x.dim() > 1: # Batch of vectors [..., dim]
            x_rotated = torch.matmul(x, R)
        else: # Handle 0-dim tensor? Return as is.
            x_rotated = x

        if x_rotated.shape != original_shape:
             # Fallback reshape just in case matmul changes shape unexpectedly
             logger.warning(f"Shape mismatch after SO(n) rotation matmul: {original_shape} -> {x_rotated.shape}. Reshaping.")
             x_rotated = x_rotated.reshape(original_shape)

        return x_rotated


# --- Tangent Space Rotation with Proper Broadcasting ---
class TangentSpaceRotation(nn.Module):
    """
    Applies rotation to vectors in tangent space, properly handling
    broadcasting for main, boundary, and descriptor vectors.
    Uses implementations from wubu_nesting_impl.
    """
    def __init__(self, dim: int, rotation_type: str = 'so_n'):
        super().__init__()
        self.dim = dim
        self.rotation_type = rotation_type

        if rotation_type == 'so_n':
            self.rotation_impl = SO_n_Rotation(dim)
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using SO(n)")
        elif rotation_type == 'quat':
            if dim != 4:
                raise ValueError("Quaternion rotation requires dim=4")
            # Parameterize with two quaternions p, q for p*v*q rotation
            init_p = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            init_q = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            self.quat_p = nn.Parameter(init_p)
            self.quat_q = nn.Parameter(init_q)
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using Quaternion")
            # We don't need a separate rotation_impl module here
        elif rotation_type == 'identity':
            self.rotation_impl = nn.Identity() # Use Identity module for clarity
            logger.info(f"TangentSpaceRotation (Dim {dim}): Using Identity")
        else:
            raise ValueError(f"Unsupported rotation type: {rotation_type}")

    def forward(self, v_main: torch.Tensor, v_boundaries_tangent: Optional[torch.Tensor],
                v_descriptor: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Applies rotation to main, boundary (if present), and descriptor vectors.

        Args:
            v_main: Tensor of shape [batch_size, seq_len, dim] or [..., dim]
            v_boundaries_tangent: Tensor of shape [num_points, dim] or None
            v_descriptor: Tensor of shape [dim] or [1, 1, dim] or None

        Returns:
            Tuple of rotated tensors (v_main_rotated, v_boundaries_rotated, v_descriptor_rotated)
        """
        if v_main.shape[-1] != self.dim:
            raise ValueError(f"TangentSpaceRotation: Main vector dim mismatch {v_main.shape[-1]} != {self.dim}")
        # Check boundary dims if present
        has_boundaries = v_boundaries_tangent is not None and v_boundaries_tangent.numel() > 0
        if has_boundaries and v_boundaries_tangent.shape[-1] != self.dim:
             raise ValueError(f"TangentSpaceRotation: Boundary vector dim mismatch {v_boundaries_tangent.shape[-1]} != {self.dim}")
        # Check descriptor dims if present
        has_descriptor = v_descriptor is not None and v_descriptor.numel() > 0
        if has_descriptor:
            if v_descriptor.dim() == 1 and v_descriptor.shape[0] == self.dim:
                # Reshape [dim] -> [1, 1, dim] for consistency
                v_descriptor = v_descriptor.view(1, 1, self.dim)
            elif v_descriptor.shape[-1] != self.dim:
                 raise ValueError(f"TangentSpaceRotation: Descriptor vector dim mismatch {v_descriptor.shape[-1]} != {self.dim}. Shape: {v_descriptor.shape}")

        # Identity rotation: just pass through
        if self.rotation_type == 'identity':
            return v_main, v_boundaries_tangent, v_descriptor

        device = v_main.device
        model_dtype = v_main.dtype

        v_main_rotated = v_main
        v_boundaries_rotated = v_boundaries_tangent
        v_descriptor_rotated = v_descriptor

        # --- Apply Rotation ---
        if self.rotation_type == 'so_n':
            # rotation_impl is SO_n_Rotation module
            v_main_rotated = self.rotation_impl(v_main)
            if has_boundaries:
                v_boundaries_rotated = self.rotation_impl(v_boundaries_tangent)
            if has_descriptor:
                v_descriptor_rotated = self.rotation_impl(v_descriptor)

        elif self.rotation_type == 'quat':
            # Normalize p and q from parameters
            p = self.quat_p.to(device=device, dtype=model_dtype)
            q = self.quat_q.to(device=device, dtype=model_dtype)
            p_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            q_norm = torch.norm(q, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            unit_p = p / p_norm
            unit_q = q / q_norm

            # Check for NaNs after normalization
            if not torch.isfinite(unit_p).all() or not torch.isfinite(unit_q).all():
                logger.warning("NaN/Inf in quaternion normalization during rotation. Using identity for this step.")
                # Return original inputs if normalization fails
                return v_main, v_boundaries_tangent, v_descriptor

            # Rotate main using p*v*q (broadcasting handled by quat_rotate_via_pvq)
            v_main_rotated = quat_rotate_via_pvq(v_main, unit_p, unit_q)

            # Rotate boundaries if present
            if has_boundaries:
                v_boundaries_rotated = quat_rotate_via_pvq(v_boundaries_tangent, unit_p, unit_q)

            # Rotate descriptor if present
            if has_descriptor:
                v_descriptor_rotated = quat_rotate_via_pvq(v_descriptor, unit_p, unit_q)

        # --- Final Check for NaN/Inf ---
        outputs = [v_main_rotated, v_boundaries_rotated, v_descriptor_rotated]
        cleaned_outputs = []
        names = ["main", "boundaries", "descriptor"]
        for i, output in enumerate(outputs):
            # Handle None cases correctly
            is_boundary_case = (i == 1)
            is_descriptor_case = (i == 2)
            input_was_present = (is_boundary_case and has_boundaries) or \
                                (is_descriptor_case and has_descriptor) or \
                                (i == 0) # Main is always present

            if output is None:
                if input_was_present:
                    logger.error(f"TangentSpaceRotation Error: Output '{names[i]}' is None unexpectedly.")
                    # Fallback: return zeros of expected shape? Or original input? Let's use zeros.
                    if i == 0: output = torch.zeros_like(v_main)
                    elif is_boundary_case: output = torch.zeros_like(v_boundaries_tangent) if has_boundaries else None
                    elif is_descriptor_case: output = torch.zeros_like(v_descriptor) if has_descriptor else None
                else:
                    # Output is None because input was None (correct behavior)
                    pass
            elif not torch.isfinite(output).all():
                logger.warning(f"NaN/Inf detected in TangentSpaceRotation output ({names[i]}). Replacing with zeros.")
                output = torch.nan_to_num(output, nan=0.0)

            cleaned_outputs.append(output)

        return tuple(cleaned_outputs)


# --- Inter-Level Transform with Proper Broadcasting ---
class InterLevelTransform(nn.Module):
    """
    Handles transformation between tangent spaces of different hyperbolic levels.
    Properly broadcasts operations between main, boundary and descriptor vectors.
    Uses implementation from wubu_nesting_impl.
    """
    def __init__(self, in_dim: int, out_dim: int, transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.transform_type = transform_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Determine actual hidden dimension for MLP
        mlp_hidden_dim = hidden_dim # Keep track if None was passed

        if transform_type == 'mlp':
            if mlp_hidden_dim is None or mlp_hidden_dim <= 0:
                # Default logic if hidden_dim not specified or invalid
                mlp_hidden_dim = max(16, (in_dim + out_dim) // 2)
            # MLP: Linear -> LayerNorm -> GELU -> Dropout -> Linear
            self.transform = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim), # Add LayerNorm for stability
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, out_dim)
            )
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using MLP (Hidden Dim: {mlp_hidden_dim})")
        elif transform_type == 'linear':
            self.transform = nn.Linear(in_dim, out_dim)
            logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using Linear")
        # elif transform_type == 'quat': # Removed QuatLinear from here - keep transform non-rotational?
        #     check_quat_dim(in_dim, "InterLevelTransform Quat Input")
        #     check_quat_dim(out_dim, "InterLevelTransform Quat Output")
        #     self.transform = QuaternionLinear(in_dim, out_dim, bias=True)
        #     logger.info(f"InterLevelTransform ({in_dim}->{out_dim}): Using QuaternionLinear")
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")

        # Initialize weights (using helper function)
        self.apply(init_weights)

    def forward(self, v_rotated: torch.Tensor, v_boundaries_rotated: Optional[torch.Tensor],
                v_descriptor_rotated: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Applies transformation to rotated main, boundary, and descriptor vectors.

        Args:
            v_rotated: Rotated main vectors [..., in_dim]
            v_boundaries_rotated: Rotated boundary vectors [num_points, in_dim] or None
            v_descriptor_rotated: Rotated descriptor vector [1, 1, in_dim] or None

        Returns:
            Tuple of transformed tensors (v_main_transformed, v_boundaries_transformed, v_descriptor_transformed)
            Shapes: [..., out_dim], [num_points, out_dim] or None, [1, 1, out_dim] or None
        """
        has_boundaries = v_boundaries_rotated is not None and v_boundaries_rotated.numel() > 0
        has_descriptor = v_descriptor_rotated is not None and v_descriptor_rotated.numel() > 0

        # --- Apply Transformation ---
        v_main_transformed = self.transform(v_rotated)
        v_boundaries_transformed = None
        if has_boundaries:
            v_boundaries_transformed = self.transform(v_boundaries_rotated)
        v_descriptor_transformed = None
        if has_descriptor:
            v_descriptor_transformed = self.transform(v_descriptor_rotated) # Shape should be [1, 1, out_dim]

        # --- Final Check for NaN/Inf ---
        outputs = [v_main_transformed, v_boundaries_transformed, v_descriptor_transformed]
        cleaned_outputs = []
        names = ["main", "boundaries", "descriptor"]
        for i, output in enumerate(outputs):
            is_boundary_case = (i == 1)
            is_descriptor_case = (i == 2)
            input_was_present = (is_boundary_case and has_boundaries) or \
                                (is_descriptor_case and has_descriptor) or \
                                (i == 0) # Main is always present

            if output is None:
                if input_was_present:
                    logger.error(f"InterLevelTransform Error: Output '{names[i]}' is None unexpectedly.")
                    # Fallback: return zeros of expected shape? Or original input? Let's use zeros.
                    out_shape_main = list(v_rotated.shape[:-1]) + [self.out_dim]
                    out_shape_bound = list(v_boundaries_rotated.shape[:-1]) + [self.out_dim] if has_boundaries else None
                    out_shape_desc = list(v_descriptor_rotated.shape[:-1]) + [self.out_dim] if has_descriptor else None

                    if i == 0: output = torch.zeros(out_shape_main, device=v_rotated.device, dtype=v_rotated.dtype)
                    elif is_boundary_case: output = torch.zeros(out_shape_bound, device=v_boundaries_rotated.device, dtype=v_boundaries_rotated.dtype) if has_boundaries else None
                    elif is_descriptor_case: output = torch.zeros(out_shape_desc, device=v_descriptor_rotated.device, dtype=v_descriptor_rotated.dtype) if has_descriptor else None
                else:
                    pass # Output is None because input was None (correct)
            elif not torch.isfinite(output).all():
                logger.warning(f"NaN/Inf detected in InterLevelTransform output ({names[i]}). Replacing with zeros.")
                output = torch.nan_to_num(output, nan=0.0)

            cleaned_outputs.append(output)

        return tuple(cleaned_outputs)

# --- Boundary Manifold with Learnable Points ---
class BoundaryManifold(nn.Module):
    """
    Represents the learnable boundary points for a WuBu level.
    These points define sub-manifolds in the tangent space.
    Uses implementation from wubu_nesting_impl.
    """
    def __init__(self, level_idx: int, num_points: int, point_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim

        # Initialize tangent points as learnable parameters
        if num_points > 0 and point_dim > 0:
            # Initialize with small random values
            tangent_points = torch.randn(num_points, point_dim) * init_scale
            self.tangent_points = nn.Parameter(tangent_points)
            logger.info(f"BoundaryManifold L{level_idx}: {num_points} points in {point_dim}D tangent space.")
        else:
            # Register None if no points or zero dimension
            self.register_parameter('tangent_points', None)
            logger.info(f"BoundaryManifold L{level_idx}: No boundary points (num_points={num_points}, dim={point_dim}).")

    def get_tangent_vectors_at_origin(self) -> Optional[torch.Tensor]:
        """Returns the current boundary points (tangent vectors at origin), checking stability."""
        if self.tangent_points is None:
            return None

        # Stability check before returning
        if not torch.isfinite(self.tangent_points).all():
            logger.warning(f"NaN/Inf detected in BoundaryManifold L{self.level_idx} tangent_points. Re-initializing gently.")
            # Re-initialize with small random values if NaN/Inf detected
            init_scale = 0.01
            self.tangent_points.data.normal_(0, init_scale) # Re-initialize in-place

        return self.tangent_points


# --- WuBu Nesting Level Implementation ---
class WuBuNestingLevel(nn.Module):
    """
    Implements a single level of the WuBu Nesting architecture.
    Handles processing within a level including tangent space operations.
    Uses implementation from wubu_nesting_impl.
    """
    def __init__(self, level_idx: int, dim: int, config: Dict): # Pass full config dict
        super().__init__()
        self.level_idx = level_idx
        self.dim = dim

        # Extract relevant config values with defaults
        self.use_ld = config.get("use_level_descriptors", True)
        self.use_spread = config.get("use_level_spread", True)
        self.use_flow = config.get("use_tangent_flow", True)
        self.dropout = config.get("dropout", 0.1)
        self.ld_init_scale = config.get("level_descriptor_init_scale", 0.01)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean") # Added this

        # Minimum values for constrained parameters (get from config or use default)
        self.min_curvature = config.get("curvature_min_value", EPS)
        self.min_scale = config.get("scale_min_value", EPS)
        self.min_spread = config.get("spread_min_value", EPS)

        # Helper for constrained parameterization
        def _init_constrained_param(value, min_val):
            clamped_value = max(float(value), min_val + EPS) # Ensure value > min_val
            # Simplified inverse softplus for stability: log(exp(y) - 1) approx log(y) for small y
            # Avoid potential issues with expm1(y) for extremely small y
            y = clamped_value - min_val
            if y < 1e-6:
                 unconstrained_val = math.log(y + EPS) # Add EPS to avoid log(0)
            else:
                 try: unconstrained_val = math.log(math.expm1(y))
                 except (ValueError, OverflowError) as e:
                     logger.error(f"Error inv softplus val={value}, min={min_val}, y={y}: {e}")
                     unconstrained_val = math.log(EPS) # Fallback to log of small number
            return torch.tensor(unconstrained_val, dtype=torch.float)

        # --- Curvature (c_i) ---
        learnable_c = config.get("learnable_curvature", True)
        init_c = config["initial_curvatures"][level_idx]
        log_init_c = _init_constrained_param(init_c, self.min_curvature)
        if learnable_c: self.log_curvature = nn.Parameter(log_init_c)
        else: self.register_buffer('log_curvature', log_init_c)

        # --- Scale (s_i) ---
        learnable_s = config.get("learnable_scales", True)
        init_s = config["initial_scales"][level_idx]
        log_init_s = _init_constrained_param(init_s, self.min_scale)
        if learnable_s: self.log_scale = nn.Parameter(log_init_s)
        else: self.register_buffer('log_scale', log_init_s)

        # --- Level Descriptor (ld_i) ---
        if self.use_ld: self.level_descriptor = nn.Parameter(torch.randn(dim) * self.ld_init_scale)
        else: self.register_buffer('level_descriptor', torch.zeros(dim))

        # --- Spread (sigma_i) ---
        learnable_spread = config.get("learnable_spread", True)
        initial_spread_values_list = config.get("initial_spread_values")
        if initial_spread_values_list is None or len(initial_spread_values_list) <= level_idx: init_spread = config["initial_scales"][level_idx] # Fallback to scale
        else: init_spread = initial_spread_values_list[level_idx]
        if self.use_spread:
            log_init_spread = _init_constrained_param(init_spread, self.min_spread)
            if learnable_spread: self.log_spread = nn.Parameter(log_init_spread)
            else: self.register_buffer('log_spread', log_init_spread)
        else: self.register_buffer('log_spread', _init_constrained_param(self.min_spread + EPS, self.min_spread))

        # --- Tangent Space Combiner MLP ---
        combiner_input_dim = self.dim
        if self.relative_vector_aggregation != 'none': combiner_input_dim += self.dim
        if self.use_ld: combiner_input_dim += self.dim
        if self.use_spread: combiner_input_dim += 1
        combiner_hidden_dims = config.get("tangent_input_combination_dims", [max(16, combiner_input_dim // 2)])
        layers = []; in_d = combiner_input_dim
        for h_dim in combiner_hidden_dims:
            layers.extend([ nn.Linear(in_d, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(self.dropout) ]); in_d = h_dim
        layers.append(nn.Linear(in_d, self.dim))
        self.tangent_combiner = nn.Sequential(*layers)

        # --- Optional Tangent Flow ---
        self.tangent_flow = None; self.flow_scale = 0.0
        if self.use_flow:
            flow_hidden_dim = max(16, int(dim * config.get("tangent_flow_hidden_dim_ratio", 0.5)))
            flow_type = config.get("tangent_flow_type", "mlp")
            if flow_type == 'mlp':
                 self.tangent_flow = nn.Sequential( nn.Linear(dim, flow_hidden_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(flow_hidden_dim, dim) )
            elif flow_type == 'linear':
                 self.tangent_flow = nn.Linear(dim, dim)
            elif flow_type != 'none':
                 logger.warning(f"L{level_idx}: Unsupported tangent_flow_type '{flow_type}', disabling flow.")
                 self.use_flow = False # Disable if type is invalid

            if self.use_flow: # Check again in case it was disabled
                self.flow_scale = config.get("tangent_flow_scale", 1.0)

        # --- Apply Initialization ---
        self.apply(init_weights)

        # Log level initialization details
        c_val = self.get_curvature().item(); s_val = self.get_scale().item()
        spr_val = self.get_spread().item() if self.use_spread else 'N/A'
        logger.info(f"WuBuLevel {level_idx} (Dim {dim}) Init: c={c_val:.3f}, s={s_val:.3f}, sprd={spr_val}, CombinerInDim={combiner_input_dim}, Learn(c/s/sprd): {learnable_c}/{learnable_s}/{learnable_spread}, Use(LD/Sprd/Flow): {self.use_ld}/{self.use_spread}/{self.use_flow}")


    def get_curvature(self) -> torch.Tensor:
        """Returns the constrained positive curvature parameter."""
        return F.softplus(self.log_curvature) + self.min_curvature

    def get_scale(self) -> torch.Tensor:
        """Returns the constrained positive scale parameter."""
        return F.softplus(self.log_scale) + self.min_scale

    def get_spread(self) -> torch.Tensor:
        """Returns the constrained positive spread parameter."""
        if not self.use_spread:
            # Return the minimum value (as scalar tensor) if spread is disabled
            param_device = self.log_curvature.device # Get device from another param/buffer
            return torch.tensor(self.min_spread, device=param_device, dtype=torch.float)
        # Use softplus + min for learnable or fixed spread parameter
        return F.softplus(self.log_spread) + self.min_spread

    def forward(self, v_tangent_in: torch.Tensor,
                relative_vectors_in: Optional[torch.Tensor],
                ld_tangent_in: Optional[torch.Tensor],
                sigma_in: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process tangent vectors through the WuBu level.
        """
        batch_size, seq_len, d_in = v_tangent_in.shape
        device = v_tangent_in.device
        # Get dtype from a parameter (safer if model is mixed precision)
        model_dtype = next(self.parameters()).dtype

        if d_in != self.dim: raise ValueError(f"L{self.level_idx} input dim mismatch: {d_in} != {self.dim}")

        curvature = self.get_curvature().to(device); scale = self.get_scale().to(device); spread = self.get_spread().to(device)

        inputs_to_combine = [v_tangent_in]
        # Relative Vectors
        if self.relative_vector_aggregation != 'none':
            if relative_vectors_in is not None:
                if relative_vectors_in.shape == (batch_size, seq_len, self.dim): inputs_to_combine.append(relative_vectors_in.to(model_dtype))
                else: logger.warning(f"L{self.level_idx}: Rel vec shape mismatch {relative_vectors_in.shape}. Use zeros."); inputs_to_combine.append(torch.zeros_like(v_tangent_in))
            else: inputs_to_combine.append(torch.zeros_like(v_tangent_in)) # Add zeros if aggregation is enabled but no input provided
        # Level Descriptor
        if self.use_ld:
            if ld_tangent_in is not None:
                if ld_tangent_in.shape == (batch_size, seq_len, self.dim): inputs_to_combine.append(ld_tangent_in.to(model_dtype))
                else: logger.warning(f"L{self.level_idx}: LD shape mismatch {ld_tangent_in.shape}. Use zeros."); inputs_to_combine.append(torch.zeros_like(v_tangent_in))
            else: inputs_to_combine.append(torch.zeros_like(v_tangent_in)) # Add zeros if LD is enabled but no input provided
        # Spread
        if self.use_spread:
            if sigma_in is None: sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device, dtype=model_dtype)
            elif sigma_in.numel() == 1: sigma_in_tensor = sigma_in.expand(batch_size, seq_len, 1).to(model_dtype)
            elif sigma_in.shape == (batch_size, seq_len): sigma_in_tensor = sigma_in.unsqueeze(-1).to(model_dtype)
            elif sigma_in.shape == (batch_size, seq_len, 1): sigma_in_tensor = sigma_in.to(model_dtype)
            else: logger.warning(f"L{self.level_idx}: Sigma shape {sigma_in.shape}. Use zeros."); sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device, dtype=model_dtype)
            inputs_to_combine.append(sigma_in_tensor)

        try:
            combined_inputs = torch.cat(inputs_to_combine, dim=-1)
        except RuntimeError as e:
            logger.error(f"L{self.level_idx}: Concat error: {e}\nShapes: {[inp.shape for inp in inputs_to_combine]}")
            raise e
        # Check combiner input dimension dynamically
        expected_combiner_dim = self.tangent_combiner[0].in_features
        if combined_inputs.shape[-1] != expected_combiner_dim: raise ValueError(f"L{self.level_idx} Combiner dim mismatch: expect {expected_combiner_dim}, got {combined_inputs.shape[-1]}")

        if not torch.isfinite(combined_inputs).all(): logger.warning(f"NaN/Inf L{self.level_idx} combined inputs pre-combiner. Replace."); combined_inputs = torch.nan_to_num(combined_inputs, nan=0.0)
        v_combined = self.tangent_combiner(combined_inputs)
        if self.use_flow and self.tangent_flow is not None:
            flow_displacement = self.tangent_flow(v_combined)
            if not torch.isfinite(flow_displacement).all(): logger.warning(f"NaN/Inf L{self.level_idx} flow displacement. Replace."); flow_displacement = torch.nan_to_num(flow_displacement, nan=0.0)
            v_combined = v_combined + flow_displacement * self.flow_scale

        # Perform hyperbolic operations
        x_hyperbolic = HyperbolicUtils.scale_aware_exponential_map(v_combined, curvature.item(), scale.item())
        v_tangent_out = HyperbolicUtils.scale_aware_logarithmic_map(x_hyperbolic, curvature.item(), scale.item())

        # Ensure outputs match the expected model dtype
        x_hyperbolic = x_hyperbolic.to(model_dtype); v_tangent_out = v_tangent_out.to(model_dtype)
        # Ensure parameter outputs also match dtype for consistency
        ld_param_out = self.level_descriptor.to(model_dtype); sigma_param_out = spread.to(model_dtype)

        return x_hyperbolic, v_tangent_out, ld_param_out, sigma_param_out


# --- Complete WuBu Nesting Model ---
class WuBuNestingModel(nn.Module):
    """
    Full implementation of the WuBu Nesting architecture with proper nested hyperbolic
    spaces and tangent space transitions, including boundary manifolds, rotation, and
    relative vector computation. Uses implementation from wubu_nesting_impl.
    """
    def __init__(self, input_dim: int, output_dim: int, config: Dict): # Pass full config dict
        super().__init__()
        self.input_dim = input_dim; self.output_dim = output_dim; self.config = config

        self.num_levels = config.get("num_levels", 3)
        self.hyperbolic_dims = config.get("hyperbolic_dims", [128, 64, 32])
        self.boundary_points = config.get("boundary_points_per_level", [5, 4, 3])
        num_transitions = max(0, self.num_levels - 1)
        self.rotation_types = config.get("rotation_types", ["so_n"] * num_transitions)
        self.transform_types = config.get("transform_types", ["linear"] * num_transitions)
        self.transform_hdims = config.get("transform_hidden_dims", [None] * num_transitions)
        self.dropout = config.get("dropout", 0.1)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean")
        self.aggregation_method = config.get("aggregation_method", "concat_tangent")

        if len(self.hyperbolic_dims) != self.num_levels: raise ValueError(f"Len(hyperbolic_dims) != num_levels")
        if len(self.boundary_points) != self.num_levels: raise ValueError(f"Len(boundary_points) != num_levels")
        if len(self.rotation_types) != num_transitions: raise ValueError(f"Len(rotation_types) != num_levels-1")
        if len(self.transform_types) != num_transitions: raise ValueError(f"Len(transform_types) != num_levels-1")
        if len(self.transform_hdims) != num_transitions: raise ValueError(f"Len(transform_hidden_dims) != num_levels-1")

        # Input Projection
        self.input_to_tangent = nn.Linear(input_dim, self.hyperbolic_dims[0])

        # Levels and Boundaries
        self.levels = nn.ModuleList([WuBuNestingLevel(i, self.hyperbolic_dims[i], self.config) for i in range(self.num_levels)])
        self.boundaries = nn.ModuleList([BoundaryManifold(i, self.boundary_points[i], self.hyperbolic_dims[i], init_scale=config.get('level_descriptor_init_scale', 0.01)) for i in range(self.num_levels)])

        # Transitions
        self.rotations = nn.ModuleList()
        if num_transitions > 0: self.rotations = nn.ModuleList([TangentSpaceRotation(self.hyperbolic_dims[i], self.rotation_types[i]) for i in range(num_transitions)])
        self.transforms = nn.ModuleList()
        if num_transitions > 0: self.transforms = nn.ModuleList([InterLevelTransform(self.hyperbolic_dims[i], self.hyperbolic_dims[i+1], self.transform_types[i], self.transform_hdims[i], self.dropout) for i in range(num_transitions)])

        # Output Aggregation
        if self.aggregation_method == "concat_tangent":
            combined_dim = sum(self.hyperbolic_dims)
            self.tangent_to_output = nn.Linear(combined_dim, output_dim)
        else: raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not supported.")

        # Apply weight initialization
        self.apply(init_weights)

        total_params = sum(p.numel() for p in self.parameters()); trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"WuBuNestingModel initialized: {self.num_levels} levels.")
        logger.info(f"InputDim->{self.hyperbolic_dims[0]} | Levels: {self.hyperbolic_dims} | Agg: {self.aggregation_method} | AggDim->{output_dim}")
        logger.info(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass through the WuBu Nesting model. """
        batch_size, seq_len, _ = x.shape
        device = x.device
        model_dtype = next(self.parameters()).dtype # Get dtype from a parameter

        # Project input to the tangent space of the first level
        current_tangent_main = self.input_to_tangent(x)

        level_tangent_outputs = []
        aggregated_relative_vectors_for_next_level = None # Initialized to None for level 0
        current_ld_tangent = None # Initialized to None for level 0
        current_sigma = None      # Initialized to None for level 0

        for i in range(self.num_levels):
            level_module = self.levels[i]; boundary_module = self.boundaries[i]

            # Process through the current level
            # Input: current_tangent_main, relative_vectors (from prev), ld (from prev), sigma (from prev)
            # Output: x_hyperbolic_i, v_tangent_i_out, ld_i_param, sigma_i_param
            _, v_tangent_i_out, ld_i_param, sigma_i_param = level_module(
                v_tangent_in=current_tangent_main,
                relative_vectors_in=aggregated_relative_vectors_for_next_level,
                ld_tangent_in=current_ld_tangent,
                sigma_in=current_sigma
            )

            # Store the tangent output for final aggregation
            level_tangent_outputs.append(v_tangent_i_out)

            # Prepare inputs for the next level (if not the last level)
            if i < self.num_levels - 1:
                # Check if modules exist
                if not self.rotations or i >= len(self.rotations): raise RuntimeError(f"Missing rotation module for level transition {i}")
                if not self.transforms or i >= len(self.transforms): raise RuntimeError(f"Missing transform module for level transition {i}")
                rotation_module = self.rotations[i]; transform_module = self.transforms[i]

                # Get boundary points from the current level's manifold
                v_boundaries_tangent_origin = boundary_module.get_tangent_vectors_at_origin()
                if v_boundaries_tangent_origin is not None: v_boundaries_tangent_origin = v_boundaries_tangent_origin.to(device=device, dtype=model_dtype)

                # Ensure LD parameter is on the correct device/dtype for rotation
                ld_i_param_ready = ld_i_param.to(device=device, dtype=model_dtype)

                # Rotate: Output tangent vector, boundary vectors, level descriptor
                v_main_rotated, v_boundaries_rotated, ld_rotated = rotation_module(
                    v_main=v_tangent_i_out,
                    v_boundaries_tangent=v_boundaries_tangent_origin,
                    v_descriptor=ld_i_param_ready # Pass LD parameter here
                )

                # Transform: Rotated main vector, boundary vectors, level descriptor to next level's tangent space
                v_next_tangent_main, v_boundaries_transformed, ld_next_tangent = transform_module(
                    v_rotated=v_main_rotated,
                    v_boundaries_rotated=v_boundaries_rotated,
                    v_descriptor_rotated=ld_rotated
                )

                # Calculate relative vectors for the next level
                aggregated_relative_vectors_for_next_level = None # Reset for calculation
                has_next_boundaries = v_boundaries_transformed is not None and v_boundaries_transformed.numel() > 0
                if has_next_boundaries and self.relative_vector_aggregation != 'none':
                    # Expand dims for broadcasting: main [B, S, D], boundaries [1, 1, N, D] -> [B, S, N, D]
                    main_expanded = v_next_tangent_main.unsqueeze(2)        # [B, S, 1, D_next]
                    boundaries_expanded = v_boundaries_transformed.unsqueeze(0).unsqueeze(0) # [1, 1, N_points, D_next]
                    # Calculate differences: [B, S, N_points, D_next]
                    relative_vectors_calc = main_expanded - boundaries_expanded

                    # Aggregate relative vectors
                    agg_method = self.relative_vector_aggregation
                    if agg_method == "mean": aggregated_relative_vectors_for_next_level = torch.mean(relative_vectors_calc, dim=2)
                    elif agg_method == "sum": aggregated_relative_vectors_for_next_level = torch.sum(relative_vectors_calc, dim=2)
                    # Add other aggregation methods (max, etc.) here if needed
                    else: aggregated_relative_vectors_for_next_level = None # Handle 'none' or invalid

                    # Stability check
                    if aggregated_relative_vectors_for_next_level is not None and not torch.isfinite(aggregated_relative_vectors_for_next_level).all():
                        logger.warning(f"NaN/Inf L{i+1} rel vecs. Replace."); aggregated_relative_vectors_for_next_level = torch.zeros_like(v_next_tangent_main)
                else:
                    # If no boundaries or aggregation is none, pass None (or zeros)
                    aggregated_relative_vectors_for_next_level = None

                # --- Update inputs for the next iteration ---
                current_tangent_main = v_next_tangent_main
                # Expand LD if present to match batch/seq dims
                if ld_next_tangent is not None:
                    current_ld_tangent = ld_next_tangent.expand(batch_size, seq_len, -1)
                else: current_ld_tangent = None
                # Pass the spread parameter from the current level to the next
                current_sigma = sigma_i_param

        # Aggregate outputs from all levels (tangent space)
        if self.aggregation_method == "concat_tangent":
            try:
                aggregated_tangent = torch.cat(level_tangent_outputs, dim=-1)
            except RuntimeError as e:
                logger.error(f"Concat error: {e}\nShapes: {[t.shape for t in level_tangent_outputs]}");
                return torch.zeros((batch_size, seq_len, self.output_dim), device=device, dtype=model_dtype) # Fallback
        else: raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not supported.")

        # Final projection to output dimension
        final_output = self.tangent_to_output(aggregated_tangent)

        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask_expanded = padding_mask.unsqueeze(-1).bool() # Ensure boolean and correct shape [B, S, 1]
            final_output = final_output.masked_fill(padding_mask_expanded, 0.0)

        # Final stability check
        if not torch.isfinite(final_output).all(): logger.warning("NaN/Inf final WuBuModel output. Replace."); final_output = torch.nan_to_num(final_output, nan=0.0)

        return final_output


# =====================================================================
# END: Code copied from wubu_nesting_impl.py
# =====================================================================


# =====================================================================
# HAKMEM-Inspired Components (Unchanged from v0.02)
# =====================================================================

# =====================================================================
# Data Structures and Configuration Classes (Unchanged)
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
# HAKMEM-Inspired Entropy Calculation Helper (Unchanged)
# =====================================================================
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
        if isinstance(byte_window, tuple): cache_key = byte_window; byte_list = list(byte_window)
        elif isinstance(byte_window, list): cache_key = tuple(byte_window); byte_list = byte_window
        elif isinstance(byte_window, bytes): cache_key = byte_window; byte_list = list(byte_window)
        elif isinstance(byte_window, np.ndarray): byte_list = byte_window.tolist(); cache_key = tuple(byte_list)
        elif isinstance(byte_window, torch.Tensor): byte_list = byte_window.cpu().byte().tolist(); cache_key = tuple(byte_list)
        else:
            logger.warning(f"Unsupported type for entropy calculation: {type(byte_window)}")
            return 0.0
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
        except Exception as e:
            logger.warning(f"Error during entropy calculation: {e}")
            return 0.0

# =====================================================================
# HAKMEM Babylon Index (Word/Punctuation Based) (Unchanged)
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
        if byte_seq_tensor.dim() != 1: byte_seq_tensor = byte_seq_tensor.flatten()
        if byte_seq_tensor.numel() == 0: return []
        device = byte_seq_tensor.device
        try:
            text = byte_seq_tensor.cpu().numpy().tobytes().decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error decoding byte tensor (len {byte_seq_tensor.numel()}): {e}. Returning no patches.")
            return []
        patches_with_entropy = []; current_patch_start = 0; in_word = False
        for i, char in enumerate(text):
            is_delimiter = char in self.whitespace_chars or char in self.punctuation_chars
            if is_delimiter:
                if in_word:
                    word_str = text[current_patch_start:i]
                    try:
                        word_bytes = torch.tensor(list(word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                        if word_bytes.numel() > 0:
                            entropy = self.entropy_helper.compute_entropy(word_bytes)
                            patches_with_entropy.append((word_bytes, entropy))
                    except Exception as enc_e:
                        logger.warning(f"Error encoding word patch '{word_str[:20]}...': {enc_e}")
                    in_word = False
                try:
                    delim_bytes = torch.tensor(list(char.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                    if delim_bytes.numel() > 0:
                        entropy = self.entropy_helper.compute_entropy(delim_bytes)
                        patches_with_entropy.append((delim_bytes, entropy))
                except Exception as enc_e:
                    logger.warning(f"Error encoding delimiter patch '{char}': {enc_e}")
                current_patch_start = i + 1
            else:
                if not in_word:
                    in_word = True; current_patch_start = i
        if in_word and current_patch_start < len(text):
            trailing_word_str = text[current_patch_start:]
            try:
                trailing_word_bytes = torch.tensor(list(trailing_word_str.encode('utf-8', errors='replace')), dtype=torch.uint8, device=device)
                if trailing_word_bytes.numel() > 0:
                    entropy = self.entropy_helper.compute_entropy(trailing_word_bytes)
                    patches_with_entropy.append((trailing_word_bytes, entropy))
            except Exception as enc_e:
                logger.warning(f"Error encoding trailing word patch '{trailing_word_str[:20]}...': {enc_e}")
        patches_with_entropy = [(p, e) for p, e in patches_with_entropy if p.numel() > 0]
        return patches_with_entropy
    @torch.no_grad()
    def reset_context(self): self.entropy_helper.entropy_cache = {}

# =====================================================================
# HAKMEM-Enhanced Cross Attention Block (Unchanged)
# =====================================================================
class HAKMEMCrossAttentionBlock(nn.Module):
    """A standard cross-attention block with LayerNorm and optional Flash Attention."""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size <= 0: raise ValueError("hidden_size must be positive")
        if num_heads <= 0 : num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1; logger.warning(f"Using 1 head for CrossAttention size {hidden_size}.")
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]; logger.warning(f"Adjusted CrossAttention heads: {original_num_heads} -> {num_heads} for hidden_size {hidden_size}.")
        self.hidden_size = hidden_size; self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(max(1, self.head_dim))
        self.norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]: nn.init.xavier_uniform_(layer.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_queries, _ = queries.size()
        _, seq_len_kv, kv_hidden_size = keys_values.size()
        if seq_len_kv == 0: return torch.zeros_like(queries)
        if kv_hidden_size != self.hidden_size: raise ValueError(f"Key/Value hidden size mismatch: K/V has {kv_hidden_size}, Q expects {self.hidden_size}")
        queries_norm = self.norm_q(queries)
        keys_values_norm = self.norm_kv(keys_values)
        q = self.q_proj(queries_norm).view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask_sdpa = None
        if attention_mask is not None:
            mask_dtype = torch.bool
            if attention_mask.dtype != torch.bool: attention_mask = attention_mask > 0
            # Reshape mask for SDPA: [B, N_Heads, N_Queries, N_KV]
            if attention_mask.dim() == 2: # Shape [B, N_KV]
                attn_mask_sdpa = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, num_queries, -1).to(dtype=mask_dtype)
            elif attention_mask.dim() == 3: # Shape [B, N_Queries, N_KV]
                attn_mask_sdpa = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).to(dtype=mask_dtype)
            elif attention_mask.dim() == 4: # Shape [B, N_Heads, N_Queries, N_KV]
                attn_mask_sdpa = attention_mask.to(dtype=mask_dtype)
            else: logger.warning(f"Ignoring unsupported attention mask shape {attention_mask.shape}.")

            if attn_mask_sdpa is not None:
                attn_mask_sdpa = attn_mask_sdpa.to(device=queries.device)
                expected_shape = (batch_size, self.num_heads, num_queries, seq_len_kv)
                if attn_mask_sdpa.shape != expected_shape:
                    logger.warning(f"Attention mask shape {attn_mask_sdpa.shape} not match target shape {expected_shape}. Ignoring mask."); attn_mask_sdpa = None

        use_flash = hasattr(F, 'scaled_dot_product_attention')
        output = None
        if use_flash:
            try:
                # SDPA expects True where attention *is allowed*.
                # Assuming input mask is True where PAD/masked (standard convention).
                sdpa_mask_input = ~attn_mask_sdpa if attn_mask_sdpa is not None else None

                output = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask_input, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
                if not torch.isfinite(output).all(): raise ValueError("Flash Attention produced NaN/Inf values.")
            except Exception as e: logger.warning(f"Flash Attention failed: {e}. Falling back.", exc_info=False); use_flash = False; output = None
        if output is None: # Fallback to manual attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = torch.clamp(scores, min=-30.0, max=30.0)
            # Manual attention masking: True means MASK OUT
            if attn_mask_sdpa is not None: # Use the original mask (True=PAD)
                scores = scores.masked_fill(attn_mask_sdpa, float('-inf'))
            attn_probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn_probs = torch.nan_to_num(attn_probs)
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.hidden_size)
        output = self.out_proj(output)
        if not torch.isfinite(output).all(): logger.warning("NaN/Inf in CrossAttention final output. Replacing."); output = torch.nan_to_num(output, nan=0.0)
        return output


# =====================================================================
# HAKMEM-Enhanced Local Encoder (Unchanged)
# =====================================================================
class HAKMEMLocalEncoder(nn.Module):
    """Encodes individual patches using byte embeddings, optional N-grams, and a Transformer."""
    def __init__(self, hidden_size: int=256, num_layers: int=1, num_heads: int=8, dropout: float=0.1, n_gram_sizes: List[int]=[3,4], n_gram_vocab_size: int=30000):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Local Encoder hidden_size must be positive.")
        self.hidden_size = hidden_size
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))
        self.n_gram_sizes = sorted(list(set(s for s in n_gram_sizes if isinstance(s, int) and s > 0)))
        self.n_gram_vocab_size = n_gram_vocab_size
        self.n_gram_embeddings = None
        self.hash_multipliers = {}
        if self.n_gram_sizes:
            if n_gram_vocab_size <= 0: logger.warning("Disabling N-gram features: n_gram_vocab_size <= 0."); self.n_gram_sizes = []
            else:
                self.n_gram_embeddings = nn.ModuleDict({f'n{n}': nn.Embedding(n_gram_vocab_size, hidden_size) for n in self.n_gram_sizes})
                for emb in self.n_gram_embeddings.values(): nn.init.normal_(emb.weight, std=0.02)
                self.hash_multipliers = {n: torch.tensor([self._get_prime(n * 10 + i + 1) for i in range(n)], dtype=torch.long) for n in self.n_gram_sizes}
                logger.info(f"HAKMEMLocalEncoder N-grams enabled: Sizes={self.n_gram_sizes}, Vocab={self.n_gram_vocab_size}")
        else: logger.info("HAKMEMLocalEncoder: N-gram features disabled.")
        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads: logger.warning(f"Local Encoder Transformer adjusted heads: {original_num_heads} -> {num_heads} for hidden_size {hidden_size}.")
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.patch_pooling_attention = HAKMEMCrossAttentionBlock(hidden_size, num_heads, dropout)
        self.patch_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights) # Apply initialization

    def _get_prime(self, n):
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
        windows = patch_byte_sequence.long().unsqueeze(0).unfold(dimension=1, size=n, step=1)
        multipliers = self.hash_multipliers.get(n)
        if multipliers is None: logger.warning(f"Hash multipliers not found for n={n}. Using defaults."); multipliers = torch.tensor([31**i for i in range(n)], device=device, dtype=torch.long)
        else: multipliers = multipliers.to(device=device, dtype=torch.long)
        multipliers = multipliers.view(1, 1, n)
        hashes = (windows * multipliers).sum(dim=-1)
        return (hashes % self.n_gram_vocab_size).squeeze(0)
    def forward(self, patches_with_entropy: List[Tuple[torch.Tensor, float]]) -> torch.Tensor:
        if not patches_with_entropy:
            device = next(self.parameters()).device; dtype = next(self.parameters()).dtype
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)
        device = patches_with_entropy[0][0].device; model_dtype = next(self.parameters()).dtype
        patch_representations = []
        for patch_bytes, patch_entropy in patches_with_entropy:
            patch_len = patch_bytes.size(0)
            if patch_len == 0: continue
            patch_bytes_long = patch_bytes.long()
            x = self.byte_embeddings(patch_bytes_long).to(model_dtype).unsqueeze(0)
            if self.n_gram_embeddings and self.n_gram_sizes:
                n_gram_features = torch.zeros_like(x)
                for n in self.n_gram_sizes:
                    if patch_len >= n:
                        n_gram_hashes = self._get_n_gram_hashes(patch_bytes_long, n)
                        if n_gram_hashes.numel() > 0:
                            ngram_embeds = self.n_gram_embeddings[f'n{n}'](n_gram_hashes).to(model_dtype).unsqueeze(0)
                            num_windows = ngram_embeds.size(1)
                            # Correct indexing for scatter_add_: indices from 0 to patch_len-1
                            indices = torch.arange(n - 1, n - 1 + num_windows, device=device, dtype=torch.long)
                            valid_mask = indices < patch_len # Ensure indices are within bounds of x
                            valid_indices = indices[valid_mask]
                            valid_embeds = ngram_embeds[:, valid_mask, :] # Select only embeddings corresponding to valid indices
                            if valid_indices.numel() > 0:
                                # Index needs to match the dimension being scattered into
                                index_reshaped = valid_indices.view(1, -1, 1)
                                index_expanded = index_reshaped.expand(1, valid_indices.size(0), self.hidden_size)
                                # Ensure shapes match for scatter_add_
                                if index_expanded.shape == valid_embeds.shape:
                                    n_gram_features.scatter_add_(1, index_expanded, valid_embeds)
                                else:
                                    logger.warning(f"N-gram scatter shape mismatch: index {index_expanded.shape}, embeds {valid_embeds.shape}. Skipping for n={n}")
                x = x + n_gram_features
            if not torch.isfinite(x).all(): logger.warning(f"NaN/Inf in LocalEncoder input pre-Transformer (PatchLen={patch_len}). Replacing."); x = torch.nan_to_num(x, nan=0.0)
            x = self.dropout(x)
            processed_bytes = self.transformer(x)
            if not torch.isfinite(processed_bytes).all(): logger.warning(f"NaN/Inf in LocalEncoder output post-Transformer (PatchLen={patch_len}). Replacing."); processed_bytes = torch.nan_to_num(processed_bytes, nan=0.0)
            batch_query = self.patch_query.expand(1, -1, -1).to(dtype=model_dtype)
            patch_repr = self.patch_pooling_attention(queries=batch_query, keys_values=processed_bytes)
            if not torch.isfinite(patch_repr).all(): logger.warning(f"NaN/Inf in LocalEncoder output post-pooling (PatchLen={patch_len}). Replacing."); patch_repr = torch.nan_to_num(patch_repr, nan=0.0)
            patch_representations.append(patch_repr.squeeze(1))
        if not patch_representations:
            device = next(self.parameters()).device; dtype = next(self.parameters()).dtype
            return torch.empty((1, 0, self.hidden_size), device=device, dtype=dtype)
        patches_combined = torch.cat(patch_representations, dim=0).unsqueeze(0)
        normed_output = self.norm(patches_combined)
        if not torch.isfinite(normed_output).all(): logger.warning("NaN/Inf in LocalEncoder final output. Replacing."); normed_output = torch.nan_to_num(normed_output, nan=0.0)
        return normed_output


# =====================================================================
# HAKMEM-Enhanced Local Decoder (Full Implementation) (Unchanged)
# =====================================================================
class HAKMEMLocalDecoder(nn.Module):
    """Decodes byte sequences using embeddings, positional encodings, and cross-attention to memory."""
    def __init__(self, hidden_size: int = 256, global_hidden_size: int = 1024, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1, use_hierarchical_pred: bool = True, max_decode_len: int = 2048):
        super().__init__()
        if hidden_size <= 0: raise ValueError("Decoder hidden size must be positive.")
        if global_hidden_size <= 0: raise ValueError("Decoder global_hidden_size must be positive.")
        self.hidden_size = hidden_size; self.use_hierarchical = use_hierarchical_pred
        self.max_decode_len = max_decode_len; self.vocab_size = 256
        if num_heads <= 0: num_heads = max(1, hidden_size // 64)
        original_num_heads = num_heads
        valid_heads = [h for h in range(num_heads, 0, -1) if hidden_size % h == 0]
        if not valid_heads: num_heads = 1
        elif hidden_size % num_heads != 0: num_heads = valid_heads[0]
        if num_heads != original_num_heads: logger.warning(f"HAKMEMLocalDecoder adjusted heads from {original_num_heads} to {num_heads} for hidden_size {hidden_size}.")
        self.byte_embeddings = nn.Embedding(self.vocab_size, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, std=1.0 / math.sqrt(hidden_size))
        self.positional_encoding = nn.Embedding(max_decode_len, hidden_size)
        nn.init.normal_(self.positional_encoding.weight, std=0.02)
        self.memory_projection = nn.Sequential(
            nn.Linear(global_hidden_size, hidden_size * 2, bias=True), nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=True), nn.LayerNorm(hidden_size, eps=1e-6))
        for layer in self.memory_projection:
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight);
            if hasattr(layer, 'bias') and layer.bias is not None: nn.init.zeros_(layer.bias)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True, activation=F.gelu, norm_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=self.decoder_norm)
        if self.use_hierarchical:
            self.byte_class_pred = nn.Linear(hidden_size, 16)
            self.byte_specific_pred = nn.ModuleList([nn.Linear(hidden_size, 16) for _ in range(16)])
            nn.init.normal_(self.byte_class_pred.weight, std=0.02)
            if self.byte_class_pred.bias is not None: nn.init.zeros_(self.byte_class_pred.bias)
            for layer in self.byte_specific_pred:
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(16))
                if layer.bias is not None: nn.init.zeros_(layer.bias)
            logger.info("HAKMEMLocalDecoder using Hierarchical Prediction Head.")
        else:
            self.byte_pred = nn.Linear(hidden_size, self.vocab_size)
            nn.init.normal_(self.byte_pred.weight, std=0.02)
            if self.byte_pred.bias is not None: nn.init.zeros_(self.byte_pred.bias)
            logger.info("HAKMEMLocalDecoder using Flat Prediction Head.")
        self.dropout_embed = nn.Dropout(dropout)
        self.apply(init_weights) # Apply initialization

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
    def forward(self, tgt_byte_seq: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, tgt_len = tgt_byte_seq.size(); device = tgt_byte_seq.device
        mem_batch_size, mem_len, mem_dim_in = memory.size(); model_dtype = next(self.parameters()).dtype
        if tgt_len == 0: return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)
        projected_memory: torch.Tensor
        if mem_len == 0:
            logger.debug("HAKMEMLocalDecoder received empty memory.")
            projected_memory = torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=model_dtype)
            memory_key_padding_mask = None # No memory means no padding mask needed
        else:
            if not torch.isfinite(memory).all(): logger.warning("NaN/Inf in memory input to LocalDecoder. Replacing."); memory = torch.nan_to_num(memory, nan=0.0)
            projected_memory = self.memory_projection(memory.to(model_dtype))
            if not torch.isfinite(projected_memory).all(): logger.warning("NaN/Inf after memory projection in LocalDecoder. Replacing."); projected_memory = torch.nan_to_num(projected_memory, nan=0.0)
            # Validate memory_key_padding_mask if memory is not empty
            if memory_key_padding_mask is not None:
                if memory_key_padding_mask.shape != (batch_size, mem_len):
                    logger.warning(f"memory_key_padding_mask shape mismatch ({memory_key_padding_mask.shape}) with memory ({batch_size, mem_len}). Ignoring mask.")
                    memory_key_padding_mask = None
                else:
                     memory_key_padding_mask = memory_key_padding_mask.to(device=device, dtype=torch.bool)

        tgt_embed = self.byte_embeddings(tgt_byte_seq.long()).to(model_dtype)
        positions = torch.arange(0, tgt_len, device=device).unsqueeze(0)
        positions = torch.clamp(positions, max=self.positional_encoding.num_embeddings - 1)
        pos_embed = self.positional_encoding(positions).to(model_dtype)
        tgt_prepared = self.dropout_embed(tgt_embed + pos_embed)
        if not torch.isfinite(tgt_prepared).all(): logger.warning("NaN/Inf in prepared target sequence input to Decoder Transformer. Replacing."); tgt_prepared = torch.nan_to_num(tgt_prepared, nan=0.0)
        if tgt_mask is None: tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
        if tgt_mask is not None: tgt_mask = tgt_mask.to(device=device, dtype=torch.bool)

        # Call the transformer decoder
        output = self.transformer(
            tgt=tgt_prepared,             # Shape: [B, TgtLen, D_model]
            memory=projected_memory,      # Shape: [B, MemLen, D_model]
            tgt_mask=tgt_mask,            # Shape: [TgtLen, TgtLen] (causal mask)
            memory_mask=None,             # Standard Transformer doesn't use memory mask here
            tgt_key_padding_mask=None,    # No padding assumed for target sequence
            memory_key_padding_mask=memory_key_padding_mask # Shape: [B, MemLen] (True=ignore)
        )

        if not torch.isfinite(output).all(): logger.warning("NaN/Inf detected in output of Decoder Transformer. Replacing."); output = torch.nan_to_num(output, nan=0.0)
        byte_logits: torch.Tensor
        if self.use_hierarchical:
            byte_class_logits = self.byte_class_pred(output)
            if not torch.isfinite(byte_class_logits).all(): logger.warning("NaN/Inf in hierarchical class logits. Replacing."); byte_class_logits = torch.nan_to_num(byte_class_logits, nan=0.0)
            log_class_probs = F.log_softmax(byte_class_logits, dim=-1)
            log_specific_probs_list = []
            for i in range(16):
                specific_logits = self.byte_specific_pred[i](output)
                if not torch.isfinite(specific_logits).all(): logger.warning(f"NaN/Inf in hierarchical specific logits head {i}. Replacing."); specific_logits = torch.nan_to_num(specific_logits, nan=0.0)
                log_specific_probs_list.append(F.log_softmax(specific_logits, dim=-1))
            log_specific_probs_stacked = torch.stack(log_specific_probs_list, dim=2) # [B, TgtLen, 16, 16]
            combined_log_probs = log_class_probs.unsqueeze(-1) + log_specific_probs_stacked # Broadcast: [B, TgtLen, 16, 1] + [B, TgtLen, 16, 16] -> [B, TgtLen, 16, 16]
            byte_logits = combined_log_probs.view(batch_size, tgt_len, self.vocab_size) # Reshape to [B, TgtLen, 256]
        else: byte_logits = self.byte_pred(output)
        byte_logits = byte_logits.float()
        if not torch.isfinite(byte_logits).all(): logger.warning("NaN/Inf detected in final decoder logits. Replacing with zeros."); byte_logits = torch.nan_to_num(byte_logits, nan=0.0, posinf=0.0, neginf=0.0)
        return byte_logits


# =====================================================================
# HAKMEM-Enhanced Q-Learning Controller & Optimizer (Full Implementations) (Unchanged)
# =====================================================================
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
        logger.info(f"QController initialized: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}, decay={self.epsilon_decay}, min_eps={self.min_epsilon}")
        logger.info(f"QController action ranges: LR={self.action_ranges['lr_scale']}, Momentum={self.action_ranges['momentum_scale']}")
        logger.info(f"QController reward params: OscPenalty={self.oscillation_penalty}, StabBonus={self.stability_reward_bonus}, FlowCoef={self.flow_coefficient}")

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
        reward = np.tanh(loss_change_ratio * 10.0)
        if grad_norm > 5.0: reward -= 0.1 * min(1.0, max(0.0, (grad_norm - 5.0) / 10.0))
        elif grad_norm < 0.05: reward += 0.02
        if self.oscillation_counter >= 3: reward -= self.oscillation_penalty
        self.performance_window.append(reward)
        if reward > 0.0: self.stable_steps += 1; reward += min(0.1, self.stability_reward_bonus * (self.stable_steps // 5))
        else: self.stable_steps = 0
        return float(np.clip(reward, -1.0, 1.0))

    def choose_action(self, state: Optional[Tuple]) -> Dict[str, float]:
        if state is None: return {'lr_scale': 1.0, 'momentum_scale': 1.0}
        if state not in self.q_table:
            self.q_table[state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[state] = time.time(); self.q_table_access_count[state] = 1; self._manage_q_table_size()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        chosen_actions = {}
        for param, q_values in self.q_table[state].items():
            action_space = self.action_ranges[param]
            if random.random() < self.epsilon: chosen_idx = random.randrange(len(action_space))
            else:
                finite_q_mask = np.isfinite(q_values)
                if not np.any(finite_q_mask): chosen_idx = random.randrange(len(action_space))
                else:
                    finite_q_values = q_values[finite_q_mask]; max_q = np.max(finite_q_values)
                    best_indices = np.where(np.isclose(q_values, max_q) & finite_q_mask)[0]
                    chosen_idx = np.random.choice(best_indices) if len(best_indices) > 0 else random.randrange(len(action_space))
            chosen_actions[param] = float(action_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy())
        return chosen_actions

    def update(self, state: Optional[Tuple], action: Optional[Dict[str, float]], reward: float, next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None: logger.debug("Q-update skipped: Invalid state/action."); return
        if state not in self.q_table: logger.warning(f"QController: State {state} not found in Q-table during update. Skipping."); return
        if next_state not in self.q_table:
            self.q_table[next_state] = {p: np.zeros(self.num_actions[p], dtype=np.float32) for p in self.action_ranges.keys()}
            self.q_table_creation_time[next_state] = time.time(); self.q_table_access_count[next_state] = 0; self._manage_q_table_size()
        for param, chosen_value in action.items():
            action_space = self.action_ranges[param]
            action_indices = np.where(np.isclose(action_space, chosen_value))[0]
            if len(action_indices) == 0: logger.warning(f"Could not find action index for param {param}, value {chosen_value}. Skipping update."); continue
            action_idx = action_indices[0]
            current_q = self.q_table[state][param][action_idx]
            next_q_values = self.q_table[next_state][param]
            finite_next_q = next_q_values[np.isfinite(next_q_values)]
            max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0
            if not np.isfinite(max_future_q): max_future_q = 0.0
            td_target = reward + self.gamma * max_future_q; td_error = td_target - current_q
            adaptive_alpha = min(0.5, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error))))
            new_q = current_q + adaptive_alpha * td_error
            if np.isfinite(new_q): self.q_table[state][param][action_idx] = np.clip(new_q, -1e5, 1e5)
            else: logger.warning(f"Non-finite new Q-value calculated for state {state}, param {param}, action {action_idx}. Resetting to 0."); self.q_table[state][param][action_idx] = 0.0

    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            num_to_remove = len(self.q_table) - self.max_q_table_size
            logger.info(f"Q-table size ({len(self.q_table)}) exceeds max ({self.max_q_table_size}). Pruning {num_to_remove} states.")
            try:
                if not self.q_table_access_count or not self.q_table_creation_time:
                    logger.warning("Q-table metadata incomplete. Removing random states.")
                    states_to_remove = random.sample(list(self.q_table.keys()), min(num_to_remove, len(self.q_table)))
                else:
                    sorted_states = sorted(self.q_table.keys(), key=lambda s: (self.q_table_access_count.get(s, 0), self.q_table_creation_time.get(s, float('inf'))))
                    states_to_remove = sorted_states[:num_to_remove]
                for state_to_remove in states_to_remove:
                    self.q_table.pop(state_to_remove, None)
                    self.q_table_access_count.pop(state_to_remove, None)
                    self.q_table_creation_time.pop(state_to_remove, None)
                logger.info(f"Pruned {len(states_to_remove)} states. New Q-table size: {len(self.q_table)}")
            except Exception as e:
                logger.warning(f"Error during Q-table pruning: {e}. Fallback random removal.", exc_info=False)
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
        last_action = self.prev_actions_log[-1] if self.prev_actions_log else None
        avg_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.0
        return {"epsilon": self.epsilon, "stable_steps": self.stable_steps, "oscillation_counter": self.oscillation_counter, "q_table_size": len(self.q_table), "last_action": last_action, "avg_reward_last_20": avg_reward}

class HAKMEMEnhancedSGD(torch.optim.Optimizer):
    """SGD optimizer enhanced with momentum, weight decay, gradient clipping (via Trainer), and Q-learning control."""
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm=1.0, q_learning_config: Optional[Dict]=None, enable_flow=False, flow_coefficient=0.05, flow_momentum=0.95):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.q_controller: Optional[HAKMEMQController] = None
        if isinstance(q_learning_config, dict):
            try:
                self.q_controller = HAKMEMQController(**q_learning_config)
                logger.info("HAKMEMEnhancedSGD: Q-Controller enabled.")
            except Exception as e:
                logger.error(f"Failed to initialize HAKMEMQController: {e}. Disabling Q-control.", exc_info=True)
        else: logger.info("HAKMEMEnhancedSGD: Q-Controller disabled.")
        self.max_grad_norm = max_grad_norm; self._step_count = 0
        self.current_loss: Optional[float] = None; self.gradient_stats = GradientStats()
        self.flow_enabled = enable_flow; self.flow_coefficient = flow_coefficient; self.flow_momentum = flow_momentum
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad: state = self.state[p]; state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def zero_grad(self, set_to_none=True): super().zero_grad(set_to_none=set_to_none)
    def set_current_loss(self, loss: Optional[float]):
        if loss is not None and np.isfinite(loss): self.current_loss = loss
        else: logger.debug(f"Optimizer received invalid loss: {loss}. Q-controller uses previous loss: {self.current_loss}")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None: logger.warning("HAKMEMEnhancedSGD.step received unused closure.")
        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action
            for group in self.param_groups:
                base_lr = group['base_lr']; base_momentum = group['base_momentum']
                new_lr = base_lr * q_action.get('lr_scale', 1.0)
                new_momentum = base_momentum * q_action.get('momentum_scale', 1.0)
                group['lr'] = float(np.clip(new_lr, 1e-8, 0.1))
                group['momentum'] = float(np.clip(new_momentum, 0.5, 0.999))
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']; weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad
                if not torch.isfinite(grad).all():
                    num_nan = torch.isnan(grad).sum().item(); num_inf = torch.isinf(grad).sum().item()
                    # Error vs Warning: Error might be better as it prevents update
                    logger.error(f"Optimizer Error: Non-finite gradient for param {p.shape} (NaNs: {num_nan}, Infs: {num_inf}). Skipping update.")
                    continue
                param_data = p.data; param_state = self.state[p]
                if weight_decay != 0: grad = grad.add(param_data.float(), alpha=weight_decay).to(grad.dtype)
                if 'momentum_buffer' not in param_state: param_state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format); logger.warning(f"Momentum buffer re-initialized for param {p.shape}.")
                buf = param_state['momentum_buffer']; buf.mul_(momentum).add_(grad)
                update_step = buf * lr
                # Perform update in higher precision if needed? For now, use param_data.dtype
                param_data.add_(-update_step) # In-place update
        self._step_count += 1
        return None # step() usually returns None (or loss if closure used, which it isn't here)

    def get_q_info(self) -> Dict:
        if hasattr(self, 'q_controller') and self.q_controller: return self.q_controller.get_info()
        return {"Q-Controller": "Disabled"}


# =====================================================================
# Updated WuBuNestingSequenceModel - Using the new WuBuNestingModel
# =====================================================================
class WuBuNestingSequenceModel(nn.Module):
    """
    WuBu Nesting model adapted for sequence modeling (v0.03).
    Uses the detailed WuBuNestingModel for hyperbolic processing.
    """
    def __init__(self, wubu_config: Dict, sequence_config: Dict):
        super().__init__()
        self.wubu_config = wubu_config
        self.sequence_config = sequence_config

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
            num_layers=sequence_config.get("num_encoder_layers", 2), # Renamed from local_encoder
            num_heads=sequence_config.get("num_encoder_heads", 8),   # Renamed from local_encoder
            dropout=wubu_config.get("dropout", 0.1),
            n_gram_sizes=sequence_config.get("n_gram_sizes", [3, 4]),
            n_gram_vocab_size=sequence_config.get("n_gram_vocab_size", 30000)
        )

        # --- WuBu Nesting Component ---
        # Initialize the actual WuBuNestingModel
        # Input dim = local_hidden_size, Output dim = decoder_memory_dim
        self.wubu_model = WuBuNestingModel(
            input_dim=self.local_hidden_size,
            output_dim=self.decoder_memory_dim,
            config=self.wubu_config
        )

        # --- Local Decoder ---
        self.local_decoder = HAKMEMLocalDecoder(
            hidden_size=self.local_hidden_size,         # Decoder internal dim
            global_hidden_size=self.decoder_memory_dim, # Input memory dim from WuBu model
            num_layers=sequence_config.get("num_decoder_layers", 4), # Renamed from local_decoder
            num_heads=sequence_config.get("num_decoder_heads", 8),   # Renamed from local_decoder
            dropout=wubu_config.get("dropout", 0.1),
            use_hierarchical_pred=sequence_config.get("use_hierarchical_decoder", False),
            max_decode_len=max(self.context_window * 2, 2048) # Allow longer decoding context
        )
        logger.info("WuBuNestingSequenceModel Initialized (v0.03 - Using Integrated WuBuNestingModel).")
        self.apply(init_weights) # Apply initialization to the whole sequence model

    def forward(self, byte_seq: torch.Tensor, target_byte_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the WuBu Nesting Sequence Model.

        Args:
            byte_seq (Tensor): Input byte sequence. Shape (Batch, SeqLen).
            target_byte_seq (Optional[Tensor]): Target sequence for teacher forcing/generation. Shape (Batch, TargetSeqLen).

        Returns:
            Tensor: Output logits. Shape (Batch, TargetSeqLen, VocabSize).
        """
        batch_size, seq_len = byte_seq.shape
        device = byte_seq.device
        model_dtype = next(self.parameters()).dtype

        # --- 1. Patching and Local Encoding ---
        batch_patch_repr_list = [] # Encoded patches per item
        num_patches_per_item = [] # Num patches per item
        valid_batch_indices = [] # Indices of items with > 0 patches
        max_num_patches = 0

        for i in range(batch_size):
            seq = byte_seq[i]
            patches_with_entropy = self.patcher.create_patches(seq)
            if patches_with_entropy:
                patches_on_device = [(p.to(device), e) for p, e in patches_with_entropy]
                patch_repr_single = self.local_encoder(patches_on_device) # Shape [1, NumP, LocalHidden]
                if not torch.isfinite(patch_repr_single).all():
                    logger.warning(f"NaN/Inf in local encoder output batch item {i}. Replacing."); patch_repr_single = torch.nan_to_num(patch_repr_single, nan=0.0)
                num_p = patch_repr_single.size(1)
                if num_p > 0:
                    batch_patch_repr_list.append(patch_repr_single.squeeze(0)) # Shape [NumP, LocalHidden]
                    num_patches_per_item.append(num_p)
                    valid_batch_indices.append(i)
                    max_num_patches = max(max_num_patches, num_p)
                else: num_patches_per_item.append(0) # Still append 0 if encoder returns empty
            else: num_patches_per_item.append(0)

        target_len = target_byte_seq.size(1) if target_byte_seq is not None else 0
        if not valid_batch_indices or max_num_patches == 0:
            logger.warning(f"No valid patches for batch (Size {batch_size}). Returning zero logits.")
            return torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)

        num_valid = len(valid_batch_indices)
        # Pad patch representations for valid items: [NumValid, MaxPatches, LocalHidden]
        padded_patch_repr = torch.zeros(num_valid, max_num_patches, self.local_hidden_size, device=device, dtype=model_dtype)
        # Padding mask (True where padded): [NumValid, MaxPatches]
        memory_padding_mask = torch.ones(num_valid, max_num_patches, dtype=torch.bool, device=device)

        valid_item_counter = 0
        for i in range(batch_size):
            if i in valid_batch_indices: # Only process items that had patches
                # Ensure the index exists in batch_patch_repr_list
                if valid_item_counter < len(batch_patch_repr_list):
                    patch_repr_tensor = batch_patch_repr_list[valid_item_counter]
                    num_p = patch_repr_tensor.shape[0] # Use shape from stored tensor
                    if num_p > 0 and num_p == num_patches_per_item[i]: # Sanity check
                        padded_patch_repr[valid_item_counter, :num_p, :] = patch_repr_tensor
                        memory_padding_mask[valid_item_counter, :num_p] = False
                    elif num_p != num_patches_per_item[i]:
                        logger.error(f"Batch item {i}: Patch count mismatch during padding. Stored={num_p}, Counted={num_patches_per_item[i]}. Skipping.")
                        # Invalidate this item - mark all as padded
                        memory_padding_mask[valid_item_counter, :] = True
                    else:
                         # Item was valid but had 0 patches after encoding? Mark all as padded
                         memory_padding_mask[valid_item_counter, :] = True

                    valid_item_counter += 1
                else:
                    logger.error(f"Batch item {i}: Index mismatch in valid items. Should be {valid_item_counter} but len is {len(batch_patch_repr_list)}. Skipping.")
                    # Mark as padded? This shouldn't happen if logic is correct.
                    if valid_item_counter < num_valid: # Check bounds before assigning
                        memory_padding_mask[valid_item_counter, :] = True
                    valid_item_counter +=1 # Increment anyway to maintain alignment


        # --- 2. WuBu Nesting Model Processing ---
        # The WuBuNestingModel handles projection, level iteration, aggregation, and final projection internally.
        # Input: Padded patch representations [NumValid, MaxPatches, LocalHidden]
        # Output: Decoder memory [NumValid, MaxPatches, DecMemDim]
        decoder_memory = self.wubu_model(padded_patch_repr, padding_mask=memory_padding_mask)

        # --- Stability Check ---
        if not torch.isfinite(decoder_memory).all():
            logger.warning("NaN/Inf detected in WuBuNestingModel output (decoder memory). Replacing.")
            decoder_memory = torch.nan_to_num(decoder_memory, nan=0.0)

        # --- 3. Decode ---
        if target_byte_seq is None or target_len == 0:
            logger.debug("No target_byte_seq for decoding. Returning zeros.")
            return torch.zeros((batch_size, 0, self.vocab_size), device=device, dtype=torch.float32)

        # Select target sequences for valid inputs
        valid_indices_tensor = torch.tensor(valid_batch_indices, device=device, dtype=torch.long)
        if num_valid > 0:
            valid_target_byte_seq = torch.index_select(target_byte_seq, 0, valid_indices_tensor).to(device).long()
        else:
             valid_target_byte_seq = torch.empty(0, target_len, dtype=torch.long, device=device)

        # Decode using the valid targets and the memory from WuBu model
        if num_valid > 0:
            # Pass memory_padding_mask to decoder for cross-attention masking
            byte_logits_valid = self.local_decoder(
                tgt_byte_seq=valid_target_byte_seq,         # [NumValid, TargetLen]
                memory=decoder_memory,                      # [NumValid, MaxPatches, DecMemDim]
                memory_key_padding_mask=memory_padding_mask # [NumValid, MaxPatches] (True=Pad)
            ) # Output Shape: [NumValid, TargetLen, VocabSize]
            if not torch.isfinite(byte_logits_valid).all():
                logger.warning("NaN/Inf in decoder logits output. Replacing."); byte_logits_valid = torch.nan_to_num(byte_logits_valid, nan=0.0)
        else:
            byte_logits_valid = torch.empty((0, target_len, self.vocab_size), device=device, dtype=torch.float32)

        # --- 4. Reconstruct Full Batch Output ---
        final_byte_logits = torch.zeros((batch_size, target_len, self.vocab_size), device=device, dtype=torch.float32)
        if num_valid > 0 and byte_logits_valid.numel() > 0:
            try:
                # Ensure shapes match for index_copy_
                if byte_logits_valid.shape[0] == valid_indices_tensor.shape[0]:
                    final_byte_logits.index_copy_(0, valid_indices_tensor, byte_logits_valid)
                else: logger.error(f"Shape mismatch reconstructing output. B={batch_size}, V={num_valid}, Logits={byte_logits_valid.shape}, Idx={valid_indices_tensor.shape}")
            except Exception as e_scatter: logger.error(f"Error scattering logits: {e_scatter}")

        if not torch.isfinite(final_byte_logits).all():
            logger.error("NaN/Inf in final scattered logits! Replacing."); final_byte_logits = torch.nan_to_num(final_byte_logits, nan=0.0)

        return final_byte_logits

    # --- Static Loss Computation (Unchanged) ---
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None, smoothing: float = 0.1) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        if seq_len <= 1: return torch.tensor(0.0, device=logits.device, requires_grad=True, dtype=logits.dtype)
        logits_shifted = logits[:, :-1, :].contiguous()
        targets_shifted = targets[:, 1:].contiguous()
        logits_flat = logits_shifted.view(-1, vocab_size)
        targets_flat = targets_shifted.view(-1)
        targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)
        if not torch.isfinite(logits_flat).all():
            num_nan = torch.isnan(logits_flat).sum().item(); num_inf = torch.isinf(logits_flat).sum().item()
            logger.error(f"NaN/Inf logits passed to compute_loss (NaNs:{num_nan}, Infs:{num_inf}). Returning high loss.")
            return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        loss: torch.Tensor
        if smoothing > 0.0 and 0.0 < smoothing < 1.0:
            with torch.no_grad():
                smooth_val_on = 1.0 - smoothing; smooth_val_off = smoothing / max(1, vocab_size - 1)
                true_dist = torch.full_like(logits_flat, smooth_val_off)
                true_dist.scatter_(1, targets_flat.unsqueeze(1), smooth_val_on)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            loss = -(true_dist * log_probs).sum(dim=-1)
        else: loss = F.cross_entropy(logits_flat.float(), targets_flat.long(), reduction='none')
        if not torch.isfinite(loss).all():
            logger.error(f"NaN/Inf loss calculated. Replacing before masking."); loss = torch.nan_to_num(loss, nan=100.0, posinf=100.0, neginf=-100.0)
        mean_loss: torch.Tensor
        if mask is not None:
            mask_shifted = mask[:, 1:].contiguous()
            if mask_shifted.shape == targets_shifted.shape:
                mask_flat = mask_shifted.view(-1); mask_flat_bool = mask_flat.bool() if mask_flat.dtype == torch.bool else mask_flat > 0
                loss = loss.masked_fill(mask_flat_bool, 0.0)
                num_active_elements = (~mask_flat_bool).sum()
                if num_active_elements.item() > 0: mean_loss = loss.sum() / num_active_elements
                else: logger.warning("All target elements masked. Returning zero loss."); mean_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            else: logger.warning(f"Mask shape mismatch ({mask.shape[-1]}) vs target ({targets.shape[-1]}). Ignoring mask."); mean_loss = loss.mean()
        else: mean_loss = loss.mean()
        if not torch.isfinite(mean_loss): logger.error(f"NaN/Inf final mean loss. Returning high loss."); return torch.tensor(100.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
        return mean_loss

    # --- Generation Function (Unchanged) ---
    @torch.no_grad()
    def generate(self, seed_bytes: torch.Tensor, max_length: int = 100, temperature: float = 1.0, sampling_config: Optional[SamplerConfig] = None, repetition_penalty: float = 1.1, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        self.eval(); device = next(self.parameters()).device
        if seed_bytes.device != device: seed_bytes = seed_bytes.to(device)
        seed_bytes = seed_bytes.long()
        batch_size, seed_len = seed_bytes.size()
        if seed_len == 0: logger.warning("Empty seed for generation."); return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        generated_sequence = seed_bytes.clone()
        if sampling_config is None: sampling_config = SamplerConfig()
        base_temperature = max(temperature, EPS); min_temp = max(0.1, base_temperature * 0.5); max_temp = min(2.0, base_temperature * 1.5)
        disable_tqdm = batch_size > 8 or max_length < 20
        gen_iterator = tqdm(range(max_length), desc="Generating", disable=disable_tqdm, total=max_length, unit="byte", leave=False)
        for step in gen_iterator:
            current_context = generated_sequence.long(); context_len = current_context.size(1)
            # Disable AMP for generation for stability/simplicity unless explicitly needed
            amp_context = amp.autocast(device_type=device.type, enabled=False)
            with torch.no_grad(), amp_context:
                # Pass current generated sequence as both input and target context
                logits_all = self(byte_seq=current_context, target_byte_seq=current_context)
            if logits_all is None or logits_all.numel() == 0 or logits_all.shape[1] == 0: logger.warning(f"Logits generation failed step {step}. Stopping."); break
            if not torch.isfinite(logits_all).all(): logger.warning(f"NaN/Inf logits step {step}. Using uniform."); logits_all = torch.zeros_like(logits_all)
            next_byte_logits = logits_all[:, -1, :].float() # Use logits for the *next* byte
            next_byte_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                current_logits = next_byte_logits[i].clone(); current_seq = generated_sequence[i]; current_seq_len = current_seq.size(0)
                if repetition_penalty > 1.0 and current_seq_len > 0:
                    seen_bytes = torch.unique(current_seq)
                    for byte_val_tensor in seen_bytes:
                        byte_val = byte_val_tensor.item()
                        if 0 <= byte_val < self.vocab_size:
                            if current_logits[byte_val] > 0: current_logits[byte_val] /= repetition_penalty
                            else: current_logits[byte_val] *= repetition_penalty
                adaptive_temp = base_temperature
                try:
                    probs_orig = F.softmax(current_logits, dim=-1)
                    if torch.isnan(probs_orig).any(): entropy = math.log2(self.vocab_size); logger.debug(f"Item {i} step {step}: NaN probs, max entropy.")
                    else: entropy = -torch.sum(probs_orig * torch.log2(probs_orig + EPS)).item()
                    if entropy < sampling_config.low_entropy_threshold: adaptive_temp *= 0.8
                    elif entropy > sampling_config.medium_entropy_threshold: adaptive_temp *= 1.1
                    adaptive_temp = max(min_temp, min(adaptive_temp, max_temp))
                except Exception as e_entropy: logger.warning(f"Error calculating entropy/temp item {i} step {step}: {e_entropy}. Using base."); adaptive_temp = base_temperature
                scaled_logits = current_logits / adaptive_temp; filtered_logits = scaled_logits
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    if k > 0: top_k_threshold = torch.topk(filtered_logits, k)[0][..., -1, None]; indices_to_remove = filtered_logits < top_k_threshold; filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))
                    else: filtered_logits.fill_(-float('Inf'))
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p; sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone(); sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, -float('Inf'))
                probs_final = F.softmax(filtered_logits, dim=-1)
                if torch.isnan(probs_final).any() or torch.isinf(probs_final).any() or probs_final.sum() < EPS: logger.warning(f"Invalid final probs item {i} step {step}. Sampling uniformly."); probs_final = torch.ones_like(current_logits) / current_logits.size(-1)
                if temperature <= EPS: next_byte_idx = torch.argmax(probs_final)
                else: next_byte_idx = torch.multinomial(probs_final, num_samples=1).squeeze(-1)
                next_byte_indices[i] = next_byte_idx.item()
            generated_sequence = torch.cat([generated_sequence, next_byte_indices.unsqueeze(1)], dim=1)
            if not disable_tqdm: gen_iterator.set_description(f"Generating (Len {generated_sequence.size(1)})")
        if not disable_tqdm: gen_iterator.close()
        return generated_sequence


# =====================================================================
# ByteTokenizer, ByteIterableDataset, Trainer (Fixed Syntax Errors)
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
        except ImportError:
            logger.error("NumPy required for ByteIterableDataset.")
            raise
        except Exception as e:
            logger.error(f"Error reading dataset metadata from {npy_file_path}: {e}", exc_info=True)
            raise

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
                    context_tensor = torch.tensor(context_slice.copy(), dtype=torch.long)
                    target_tensor = torch.tensor(target_slice.copy(), dtype=torch.long)
                    yield context_tensor, target_tensor
                except IndexError: logger.warning(f"W:{worker_id} R:{rank}: IndexError idx {idx}. Skip."); continue
                except Exception as e: logger.error(f"W:{worker_id} R:{rank}: Error processing idx {idx}: {e}"); continue
        except FileNotFoundError: logger.error(f"W:{worker_id} R:{rank}: Dataset file not found: {self.npy_file_path}")
        except Exception as e: logger.error(f"W:{worker_id} R:{rank}: Iterator setup/loop failed: {e}", exc_info=True)
        finally:
            if mmap_handle is not None:
                try:
                    mmap_handle.close()
                except Exception as close_err:
                    logger.warning(f"W:{worker_id} R:{rank}: Error closing mmap: {close_err}")
            del bytes_data; gc.collect() # Explicit cleanup attempt

    def set_seed(self, seed: int): self.seed = seed
    def set_epoch(self, epoch: int): self.epoch = epoch

def seed_worker(worker_id: int, base_seed: int, rank_offset: int):
    worker_seed = base_seed + rank_offset + worker_id
    np.random.seed(worker_seed); random.seed(worker_seed)
    # Note: torch seeding in worker is often unnecessary or handled by DataLoader

class Trainer:
    """Handles the training and validation loops, checkpointing, and logging."""
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
        # Initialize GradScaler correctly
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        self.log_interval = log_interval; self.save_interval = max(1, save_interval) if save_interval > 0 else 0
        self.checkpoint_dir = checkpoint_dir
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE and is_main_process() # Corrected logic
        self.max_grad_norm = max_grad_norm; self.global_step = 0; self.current_epoch = 0
        self.last_val_metrics: Optional[Dict[str, float]] = None
        self.rank = rank; self.world_size = world_size; self.is_main = is_main_process() # Use helper
        self.detect_anomaly = detect_anomaly
        if self.is_main: os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.has_grad_stats = hasattr(self.optimizer, 'gradient_stats') and isinstance(getattr(self.optimizer, 'gradient_stats', None), GradientStats)
        self.has_q_controller = hasattr(self.optimizer, 'q_controller') and isinstance(getattr(self.optimizer, 'q_controller', None), HAKMEMQController)
        self.wandb_run = wandb.run if self.wandb_enabled and wandb is not None else None
        logger.info(f"Trainer Rank {rank}: AMP={self.use_amp}, Accum={self.grad_accum_steps}, MaxNorm={self.max_grad_norm}, Anomaly={self.detect_anomaly}, GradStats={self.has_grad_stats}, QCtrl={self.has_q_controller}")

    # --- Trainer Methods Correctly Indented ---
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
                if self.world_size > 1 and not hasattr(self.train_loader.sampler, '__len__'): # Be more specific
                    dataset_len = dset_total_len // self.world_size
                else:
                    dataset_len = dset_total_len
            # Case where dataset_len is still 0 (e.g., iterable dataset without len) is handled below

            # --- Corrected Indentation START ---
            if dataset_len > 0:
                loader_batch_size = self.train_loader.batch_size or 1
                # Total micro-batches this rank will process (approximate if no drop_last)
                total_micro_batches_estimate = math.ceil(dataset_len / loader_batch_size)
                if total_micro_batches_estimate > 0 and self.grad_accum_steps > 0:
                    # Estimate total optimizer steps for this rank in the epoch
                    approx_total_optim_steps = max(1, total_micro_batches_estimate // self.grad_accum_steps)
                logger.debug(f"Rank {self.rank} Epoch Est: {total_micro_batches_estimate} micro-batches, {approx_total_optim_steps} optim steps.")
            else:
                # This else corresponds to the if dataset_len > 0:
                logger.info("Cannot estimate epoch length for tqdm (dataset or sampler lacks length).")
            # --- Corrected Indentation END ---

        except Exception as e:
            logger.warning(f"Could not estimate epoch length: {e}")

        # Configure tqdm progress bar (only shown on main process)
        disable_tqdm = not self.is_main
        batch_iterator = tqdm(self.train_loader,
                                desc=f"Epoch {self.current_epoch + 1} | Opt Step 0/?",
                                disable=disable_tqdm, total=total_micro_batches_estimate, # Pass estimate here
                                unit="batch", dynamic_ncols=True, leave=False)

        # Zero gradients before starting the epoch loop
        self.optimizer.zero_grad(set_to_none=True)

        # --- Batch Loop ---
        for i, batch_data in enumerate(batch_iterator):
            total_micro_batches_processed_epoch += 1
            micro_step_count_cycle += 1
            # Determine if this is the last micro-step in the accumulation cycle
            # Or if it's the very last batch of the epoch (to avoid losing last gradients)
            is_last_batch_in_epoch = (i == (total_micro_batches_estimate - 1)) if total_micro_batches_estimate is not None else False # Check if it's the last estimated batch
            should_optimizer_step = (micro_step_count_cycle % self.grad_accum_steps == 0) or is_last_batch_in_epoch

            # Context manager for DDP gradient synchronization:
            # Sync grads ONLY when optimizer_step happens OR if it's the last batch
            sync_context = contextlib.nullcontext()
            if self.world_size > 1 and isinstance(self.model, DistributedDataParallel):
                if not should_optimizer_step:
                     sync_context = self.model.no_sync()
                # No else needed, default context allows sync

            # Context manager for anomaly detection (only active if enabled)
            anomaly_context = torch.autograd.detect_anomaly(check_nan=True) if self.detect_anomaly else contextlib.nullcontext()

            # Forward and Backward Pass
            loss_value_scaled = None; loss = None; current_step_loss = 0.0
            try:
                with sync_context, anomaly_context:
                    # Data Handling
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2: context, targets = batch_data
                    else: logger.warning(f"Rank {self.rank}: Skip unexpected batch format {type(batch_data)} idx {i}."); continue
                    context = context.to(self.device, non_blocking=True); targets = targets.to(self.device, non_blocking=True)
                    batch_size, ctx_len = context.shape
                    if ctx_len == 0 or batch_size == 0: logger.warning(f"Rank {self.rank}: Skip empty batch {context.shape} idx {i}"); continue

                    # Forward Pass (with AMP if enabled)
                    with amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=self.scaler.is_enabled()):
                        logits = self.model(byte_seq=context, target_byte_seq=context) # Use context for both input and target
                        if logits is None: raise ValueError("Model forward returned None logits")
                        if not torch.isfinite(logits).all(): logger.warning(f"Rank {self.rank}: NaN/Inf logits pre-loss step {self.global_step} micro {micro_step_count_cycle}. Replacing."); logits = torch.nan_to_num(logits, nan=0.0)
                        # Compute loss in float32 for stability, regardless of autocast
                        loss = WuBuNestingSequenceModel.compute_loss(logits.float(), targets, smoothing=0.1)
                        if loss is None or not torch.isfinite(loss): raise ValueError(f"Non-finite/None loss: {loss}")

                    # Scale loss for accumulation
                    loss_value_scaled = loss / self.grad_accum_steps

                # Backward Pass (with scaler)
                self.scaler.scale(loss_value_scaled).backward()
                current_step_loss = loss.item() # Use original unscaled loss for logging
                if not np.isfinite(current_step_loss): logger.warning(f"Rank {self.rank}: Non-finite loss ({current_step_loss}) micro {micro_step_count_cycle}. Not accumulated."); current_step_loss = 0.0

            except Exception as batch_ex:
                logger.error(f"Error micro-step {micro_step_count_cycle} (Glob {self.global_step}) Rank {self.rank}: {batch_ex}", exc_info=False)
                # Reset accumulation state for this cycle
                total_loss_accum_cycle = 0.0; micro_step_count_cycle = 0; should_optimizer_step = False
                try:
                    self.optimizer.zero_grad(set_to_none=True) # Clear potentially corrupted grads
                except Exception as zero_grad_err:
                    logger.error(f"Error zeroing grads after batch error: {zero_grad_err}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # Attempt to free memory
                continue # Skip to next micro-batch

            # Accumulate the unscaled loss for the cycle average
            total_loss_accum_cycle += current_step_loss

            # --- Optimizer Step ---
            if should_optimizer_step:
                avg_loss_cycle = total_loss_accum_cycle / micro_step_count_cycle if micro_step_count_cycle > 0 else 0.0 # Use actual steps in cycle
                optimizer_step_skipped = False; unclipped_norm_val = 0.0; has_nonfinite_grad = False; is_clipped = False; clip_ratio = None

                # 1. Unscale Gradients
                try:
                    self.scaler.unscale_(self.optimizer)
                except Exception as unscale_err:
                    logger.error(f"Error unscaling grads step {self.global_step} Rank {self.rank}: {unscale_err}. Skipping optim.")
                    has_nonfinite_grad = True
                    unclipped_norm_val = float('inf')
                    optimizer_step_skipped = True

                # 2. Check Gradient Norm & Clip
                if not optimizer_step_skipped:
                    params_with_grad = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
                    total_norm_unclipped = torch.tensor(0.0, device=self.device)
                    if params_with_grad:
                        try:
                            # Calculate norm in float32 for stability
                            all_norms_sq = [torch.sum(p.grad.detach().float()**2) for p in params_with_grad]
                            finite_norms_sq = [n for n in all_norms_sq if torch.isfinite(n)]
                            if len(finite_norms_sq) < len(all_norms_sq):
                                has_nonfinite_grad = True; total_norm_unclipped = torch.tensor(float('inf'), device=self.device)
                                num_non_finite = len(all_norms_sq) - len(finite_norms_sq)
                                logger.warning(f"Rank {self.rank}: Non-finite grads in {num_non_finite} param(s) before clip step {self.global_step}.")
                            elif finite_norms_sq:
                                total_norm_unclipped = torch.sqrt(torch.stack(finite_norms_sq).sum())
                        except Exception as norm_ex:
                            logger.error(f"Error calculating grad norm step {self.global_step} Rank {self.rank}: {norm_ex}", exc_info=False)
                            total_norm_unclipped = torch.tensor(float('inf'), device=self.device); has_nonfinite_grad = True
                    unclipped_norm_val = total_norm_unclipped.item()

                    if has_nonfinite_grad:
                        logger.warning(f"Rank {self.rank}: Skipping optim step {self.global_step} due to non-finite grads (Norm: {unclipped_norm_val}).")
                        optimizer_step_skipped = True
                    else:
                        # Clip gradients if norm is finite and exceeds max_grad_norm
                        if self.max_grad_norm > 0 and unclipped_norm_val > self.max_grad_norm:
                            is_clipped = True
                            clip_ratio = self.max_grad_norm / (unclipped_norm_val + EPS)
                            torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm)
                        # Record grad stats (even if not clipped, for max norm tracking)
                        if self.has_grad_stats:
                            self.optimizer.gradient_stats.record_gradient(unclipped_norm_val, is_clipped, clip_ratio)

                        # 3. Q-Controller Update (before optimizer step)
                        if self.has_q_controller:
                            try:
                                self.optimizer.set_current_loss(avg_loss_cycle); group = self.optimizer.param_groups[0]; current_lr = group['lr']; current_mom = group.get('momentum', 0.0)
                                q_state = self.optimizer.q_controller.get_state(lr=current_lr, momentum=current_mom, grad_norm=unclipped_norm_val, loss=avg_loss_cycle)
                                # Update Q-table based on previous state/action and current reward/state
                                if self.optimizer.q_controller.prev_state is not None and self.optimizer.q_controller.prev_action is not None and q_state is not None:
                                    reward = self.optimizer.q_controller.compute_reward(avg_loss_cycle, self.optimizer.q_controller.prev_loss, unclipped_norm_val)
                                    if np.isfinite(reward): self.optimizer.q_controller.update(self.optimizer.q_controller.prev_state, self.optimizer.q_controller.prev_action, reward, q_state)
                                    else: logger.warning(f"Q-Controller non-finite reward ({reward}). Skip Q-update.")
                                # Store current state/loss for next update
                                self.optimizer.q_controller.prev_loss = avg_loss_cycle
                                self.optimizer.q_controller.prev_state = q_state
                                # Choose next action (applied in the next iteration's step)
                                if q_state is not None: next_q_action = self.optimizer.q_controller.choose_action(q_state); self.optimizer.q_controller.prev_action = next_q_action
                                else: self.optimizer.q_controller.prev_action = None # Reset if state is invalid
                            except Exception as q_err: logger.error(f"Q-Controller error step {self.global_step} Rank {self.rank}: {q_err}", exc_info=False); self.optimizer.q_controller.prev_state = None; self.optimizer.q_controller.prev_action = None # Reset on error

                        # 4. Optimizer Step (using scaled gradients)
                        self.scaler.step(self.optimizer)

                # 5. Update AMP Scaler (after potential step)
                self.scaler.update()
                # 6. Zero Gradients (ready for next accumulation cycle)
                self.optimizer.zero_grad(set_to_none=True)

                # 7. Logging and Checkpointing (after optimizer step attempt)
                grad_stats = {}
                if self.has_grad_stats: grad_stats = self.optimizer.gradient_stats.record_step(self.global_step, skipped=optimizer_step_skipped)

                if not optimizer_step_skipped:
                    optimizer_steps_in_epoch += 1; self.global_step += 1
                    # Update progress bar and log (only on main process)
                    if self.is_main:
                        optim_step_str = f"{optimizer_steps_in_epoch}/{(approx_total_optim_steps or '?')}"; batch_iterator.set_description(f"Epoch {self.current_epoch + 1} | Opt Step {optim_step_str}")
                        current_lr = self.optimizer.param_groups[0]['lr']; logged_norm = min(unclipped_norm_val, self.max_grad_norm) if self.max_grad_norm > 0 and is_clipped else unclipped_norm_val
                        batch_iterator.set_postfix(Loss=f"{avg_loss_cycle:.3f}", LR=f"{current_lr:.3e}", Grad=f"{logged_norm:.2f}", Scale=f"{self.scaler.get_scale():.0f}", refresh=False)
                        if self.global_step % self.log_interval == 0:
                             current_mom = self.optimizer.param_groups[0].get('momentum', 0.0)
                             q_info = self.optimizer.get_q_info() if self.has_q_controller else {}
                             log_data = { "Epoch": self.current_epoch + 1, "Step": self.global_step, "Train Loss (Cycle Avg)": avg_loss_cycle, "Learning Rate": current_lr, "Momentum": current_mom, "Grad Norm (Applied)": logged_norm, "Grad Norm (Unclipped Max)": grad_stats.get('max_gradient', 0.0), "Clip %": grad_stats.get('clip_percentage', 0.0), "Avg Clip Ratio": grad_stats.get('clip_ratio_avg', 0.0), "NonFinite Grads Encountered": grad_stats.get('non_finite_grads', 0), "AMP Scale": self.scaler.get_scale() if self.use_amp else 1.0 }
                             log_data.update({f"Q_{k}": v for k, v in q_info.items() if k != 'last_action'}); last_q_action = q_info.get('last_action')
                             if last_q_action: log_data["Q_LR_Scale"] = last_q_action.get('lr_scale', 1.0); log_data["Q_Mom_Scale"] = last_q_action.get('momentum_scale', 1.0)
                             log_msg_parts = [ f"Step {self.global_step}", f"Ep {self.current_epoch + 1} Opt {optimizer_steps_in_epoch}", f"Loss(avg): {log_data['Train Loss (Cycle Avg)']:.4f}", f"LR: {log_data['Learning Rate']:.3e}", f"Mom: {log_data['Momentum']:.3f}", f"GradNorm(App): {log_data['Grad Norm (Applied)']:.2f}", f"MaxUnclip: {log_data['Grad Norm (Unclipped Max)']:.2f}", f"Clip%: {log_data['Clip %']:.1f}%", f"NFGrads: {log_data['NonFinite Grads Encountered']}", f"Scale: {log_data['AMP Scale']:.0f}" ]
                             if self.has_q_controller: log_msg_parts.extend([ f"QScale(LR/M): {log_data.get('Q_LR_Scale', 1.0):.2f}/{log_data.get('Q_Mom_Scale', 1.0):.2f}", f"QEps: {log_data.get('Q_epsilon', 0):.3f}" ])
                             logger.info(" | ".join(log_msg_parts))
                             # --- WandB Logging (Corrected Syntax) ---
                             if self.wandb_run:
                                 try:
                                     wandb.log({"train": log_data}, step=self.global_step)
                                 except Exception as wb_err:
                                     logger.error(f"WandB log failed step {self.global_step}: {wb_err}")
                             # --- End WandB Logging ---
                    # Save checkpoint
                    if self.is_main and self.save_interval > 0 and self.global_step % self.save_interval == 0:
                        self._save_checkpoint(is_intermediate=True, metrics={'train_loss_cycle': avg_loss_cycle})

                # --- End of Optimizer Step Logic ---
                # Reset accumulation cycle counters regardless of step success
                total_loss_accum_cycle = 0.0
                micro_step_count_cycle = 0

        # --- End of Epoch ---
        if self.is_main and hasattr(batch_iterator, 'close'):
            batch_iterator.close()
        avg_epoch_loss = 0.0 # Placeholder, could compute if needed but not used elsewhere
        if self.world_size > 1: logger.debug(f"Rank {self.rank} end-of-epoch barrier."); torch.distributed.barrier(); logger.debug(f"Rank {self.rank} exited barrier.")
        return avg_epoch_loss

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_loader is None: logger.info("No validation loader. Skipping."); return {}
        self.model.eval(); approx_total_val_batches = None
        try: # Estimate val length
            val_ds_len = 0
            if hasattr(self.val_loader.sampler, '__len__'): val_ds_len = len(self.val_loader.sampler)
            elif hasattr(self.val_loader.dataset, '__len__'):
                dset_total_len = len(self.val_loader.dataset)
                # Ensure division doesn't result in zero if world_size > dset_total_len
                val_ds_len = max(1, dset_total_len // self.world_size) if self.world_size > 1 else dset_total_len
            if val_ds_len > 0:
                 val_bs = self.val_loader.batch_size or 1
                 approx_total_val_batches = math.ceil(val_ds_len / val_bs)
            else: logger.info("Cannot estimate validation length.")
        except Exception as e: logger.warning(f"Could not estimate validation length: {e}")

        val_iterator = tqdm(self.val_loader, desc=f"Val Ep {self.current_epoch + 1}", disable=not self.is_main, total=approx_total_val_batches, unit="batch", leave=False)
        batch_losses = []
        for batch_data in val_iterator:
            try:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2: context, targets = batch_data
                else: logger.warning(f"Rank {self.rank}: Skip unexpected val batch format: {type(batch_data)}"); continue
                context = context.to(self.device, non_blocking=True); targets = targets.to(self.device, non_blocking=True)
                batch_size, ctx_len = context.shape
                if batch_size == 0 or ctx_len == 0: logger.debug(f"Rank {self.rank}: Skipping empty val batch."); continue

                # Validation uses model.eval() context, AMP can still be used if enabled
                with torch.no_grad(), amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=self.use_amp):
                    model_to_eval = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
                    logits = model_to_eval(byte_seq=context, target_byte_seq=context)
                    vocab_size = getattr(model_to_eval, 'vocab_size', 256)
                    if logits.shape[0]!=batch_size or logits.shape[1]!=ctx_len or logits.shape[2]!=vocab_size: logger.warning(f"Rank {self.rank}: Val logits shape mismatch. Got {logits.shape}, expect B={batch_size},L={ctx_len},V={vocab_size}. Skip batch."); continue
                    if not torch.isfinite(logits).all(): logger.warning(f"Rank {self.rank}: NaN/Inf val logits. Skip batch."); continue
                    # Loss computed in float32 for stability
                    loss = WuBuNestingSequenceModel.compute_loss(logits.float(), targets, smoothing=0.0)

                loss_item = loss.item()
                if np.isfinite(loss_item): batch_losses.append(loss_item)
                else: logger.warning(f"Rank {self.rank}: Non-finite val loss: {loss_item}")
            except Exception as val_ex: logger.error(f"Rank {self.rank} Error val batch: {val_ex}", exc_info=False); continue

        # Gather losses from all ranks if using DDP
        all_losses = []
        if self.world_size > 1:
            losses_tensor = torch.tensor(batch_losses, dtype=torch.float64, device=self.device)
            gathered_losses_list = [torch.zeros_like(losses_tensor) for _ in range(self.world_size)]
            try:
                torch.distributed.all_gather(gathered_losses_list, losses_tensor)
                all_losses = torch.cat(gathered_losses_list).cpu().tolist()
                logger.debug(f"Rank {self.rank}: Gathered {len(all_losses)} val losses.")
            except Exception as gather_err: logger.error(f"Rank {self.rank}: Val loss gather failed: {gather_err}. Using local."); all_losses = batch_losses # Fallback
        else: all_losses = batch_losses

        metrics = {}
        if self.is_main:
            if not all_losses: logger.warning("No valid validation losses collected."); avg_loss = float('inf'); perplexity = float('inf')
            else:
                avg_loss = sum(all_losses) / len(all_losses); perplexity = float('inf')
                if np.isfinite(avg_loss):
                    try: perplexity = math.exp(min(avg_loss, 700)) # Cap loss to avoid overflow
                    except (OverflowError, ValueError): logger.warning(f"PPL calc error (Avg Loss: {avg_loss}). PPL=Inf."); perplexity = float('inf')
                else: perplexity = float('inf')
            metrics = {'val_loss': avg_loss, 'val_perplexity': perplexity}; self.last_val_metrics = metrics
            logger.info(f"Validation Epoch {self.current_epoch + 1} | Avg Loss: {metrics['val_loss']:.4f} | Perplexity: {metrics['val_perplexity']:.2f}")
            # --- WandB Logging (Corrected Syntax) ---
            if self.wandb_enabled and self.wandb_run:
                try:
                    wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "epoch": self.current_epoch + 1}, step=self.global_step)
                except Exception as wb_err:
                    logger.error(f"WandB validation log failed: {wb_err}")
            # --- End WandB Logging ---
        if hasattr(val_iterator, 'close'): val_iterator.close()
        return metrics

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict] = None):
        if not self.is_main: return
        state_indicator = f"step_{self.global_step}" if is_intermediate else f"epoch_{self.current_epoch+1}_final"
        current_metrics = metrics if metrics is not None else self.last_val_metrics; metric_str = ""
        if current_metrics:
            val_loss = current_metrics.get('val_loss'); train_loss = current_metrics.get('train_loss_cycle') or current_metrics.get('loss')
            metric_key = None; metric_val = None
            if val_loss is not None and np.isfinite(val_loss): metric_key = 'vloss'; metric_val = val_loss
            elif train_loss is not None and np.isfinite(train_loss): metric_key = 'tloss'; metric_val = train_loss
            if metric_key and metric_val is not None: mf = f"{metric_val:.2e}" if abs(metric_val) < 1e-3 and metric_val != 0.0 else f"{metric_val:.3f}"; metric_str = f"_{metric_key}{mf}"
        state_indicator += metric_str; filename = f"checkpoint_{state_indicator}.pt"; filepath = os.path.join(self.checkpoint_dir, filename)
        model_state = self.model.module.state_dict() if isinstance(self.model, DistributedDataParallel) else self.model.state_dict()
        checkpoint = {'epoch': self.current_epoch, 'global_step': self.global_step, 'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict(), 'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None, 'metrics': current_metrics, 'amp_enabled': self.use_amp, 'args': getattr(self, 'args', None)}
        if self.has_q_controller and self.optimizer.q_controller:
            try:
                # Serialize Q-controller state safely
                q_state_data = {
                    'q_table': {k: {p: v.tolist() for p, v in param_qs.items()} for k, param_qs in self.optimizer.q_controller.q_table.items()}, # Convert numpy arrays
                    'epsilon': self.optimizer.q_controller.epsilon,
                    'access_count': dict(self.optimizer.q_controller.q_table_access_count), # Convert defaultdict
                    'creation_time': self.optimizer.q_controller.q_table_creation_time,
                    'loss_window': list(self.optimizer.q_controller.loss_window), # Convert deque
                    'grad_norm_window': list(self.optimizer.q_controller.grad_norm_window), # Convert deque
                    'performance_window': list(self.optimizer.q_controller.performance_window), # Convert deque
                    'stable_steps': self.optimizer.q_controller.stable_steps,
                    'oscillation_counter': self.optimizer.q_controller.oscillation_counter,
                    'prev_loss': self.optimizer.q_controller.prev_loss,
                    'prev_state': self.optimizer.q_controller.prev_state,
                    'prev_action': self.optimizer.q_controller.prev_action
                 }
                checkpoint['q_controller_state'] = q_state_data
            except Exception as q_save_err: logger.error(f"Error preparing Q-Controller state for saving: {q_save_err}")
        temp_filepath = filepath + ".tmp"
        try:
            torch.save(checkpoint, temp_filepath)
            os.replace(temp_filepath, filepath)
            logger.info(f"Checkpoint saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}", exc_info=True)
        # --- Corrected Temp File Removal (Syntax Fixed) ---
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError as rm_err:
                logger.error(f"Failed remove temp ckpt {temp_filepath}: {rm_err}")
        # --- End Corrected Temp File Removal ---

    def load_checkpoint(self, filepath: str) -> int:
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint file not found: {filepath}")
            return 0
        try:
            checkpoint = torch.load(filepath, map_location='cpu'); logger.info(f"Loading checkpoint: {filepath}")
            model_to_load = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible_keys.missing_keys: logger.warning(f"Missing model keys: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys: logger.warning(f"Unexpected model keys: {incompatible_keys.unexpected_keys}")

            # --- Corrected Optimizer State Loading (Syntax Fixed) ---
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Move optimizer state tensors to the correct device
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                try:
                                    state[k] = v.to(self.device)
                                except Exception as e_state:
                                    logger.warning(f"Failed moving optim state '{k}' to {self.device}: {e_state}")
                    logger.info("Optimizer state loaded.")
                except Exception as optim_ex:
                    logger.warning(f"Failed loading optimizer state: {optim_ex}. State reset.", exc_info=True)
            else:
                logger.warning("Optimizer state missing. Starts from scratch.")
            # --- End Corrected Optimizer State Loading ---

            saved_amp_enabled = checkpoint.get('amp_enabled', False)
            if self.use_amp:
                if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None and saved_amp_enabled:
                    try:
                        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        logger.info("AMP scaler state loaded.")
                    except Exception as scaler_ex:
                        logger.warning(f"Failed loading AMP scaler state: {scaler_ex}. State reset.")
                elif saved_amp_enabled: logger.warning("AMP used in ckpt, but state missing. Scaler reset.")
            elif saved_amp_enabled: logger.warning("Ckpt saved with AMP, but AMP disabled now.")

            start_epoch = checkpoint.get('epoch', -1) + 1; self.global_step = checkpoint.get('global_step', 0)
            self.current_epoch = start_epoch - 1 if start_epoch > 0 else 0; self.last_val_metrics = checkpoint.get('metrics')
            if self.last_val_metrics: logger.info(f"Restored metrics: {self.last_val_metrics}")

            if self.has_q_controller and self.optimizer.q_controller and 'q_controller_state' in checkpoint:
                q_state = checkpoint['q_controller_state']; logger.info("Attempting load Q-Controller state...")
                try:
                    # Deserialize Q-controller state safely
                    self.optimizer.q_controller.q_table = {k: {p: np.array(v, dtype=np.float32) for p, v in param_qs.items()} for k, param_qs in q_state.get('q_table', {}).items()} # Convert lists back to numpy
                    self.optimizer.q_controller.epsilon = q_state.get('epsilon', self.optimizer.q_controller.epsilon)
                    self.optimizer.q_controller.q_table_access_count = defaultdict(int, q_state.get('access_count', {}))
                    self.optimizer.q_controller.q_table_creation_time = q_state.get('creation_time', {})
                    maxlen_loss = self.optimizer.q_controller.loss_window.maxlen; self.optimizer.q_controller.loss_window = deque(q_state.get('loss_window', []), maxlen=maxlen_loss)
                    maxlen_grad = self.optimizer.q_controller.grad_norm_window.maxlen; self.optimizer.q_controller.grad_norm_window = deque(q_state.get('grad_norm_window', []), maxlen=maxlen_grad)
                    maxlen_perf = self.optimizer.q_controller.performance_window.maxlen; self.optimizer.q_controller.performance_window = deque(q_state.get('performance_window', []), maxlen=maxlen_perf)
                    self.optimizer.q_controller.stable_steps = q_state.get('stable_steps', 0); self.optimizer.q_controller.oscillation_counter = q_state.get('oscillation_counter', 0)
                    self.optimizer.q_controller.prev_loss = q_state.get('prev_loss'); self.optimizer.q_controller.prev_state = q_state.get('prev_state'); self.optimizer.q_controller.prev_action = q_state.get('prev_action')
                    logger.info("Q-Controller state loaded.")
                except Exception as q_load_err: logger.warning(f"Failed loading Q-Controller state: {q_load_err}. State may be reset.", exc_info=True)
            elif self.has_q_controller: logger.warning("Q-Controller active, but state not found in ckpt.")

            if 'args' in checkpoint and checkpoint['args'] is not None and hasattr(self, 'args') and self.args is not None:
                # Safely compare arguments, handle potential missing keys or type changes
                try:
                    loaded_args_dict = vars(checkpoint['args']); current_args_dict = vars(self.args); mismatched_args = {}
                    all_keys = set(loaded_args_dict.keys()) | set(current_args_dict.keys())
                    ignore_keys = {'resume', 'local_rank', 'rank', 'world_size', 'device', 'wandb', 'wandb_project', 'wandb_entity', 'data_path', 'val_data_path', 'checkpoint_dir'}
                    for key in all_keys:
                        if key in ignore_keys: continue
                        loaded_val = loaded_args_dict.get(key, '<<MissLoad>>'); current_val = current_args_dict.get(key, '<<MissCurr>>')
                        # Basic string comparison to handle potential type differences gracefully
                        if str(loaded_val) != str(current_val): mismatched_args[key] = {'loaded': loaded_val, 'current': current_val}
                    if mismatched_args: logger.warning(f"Argument mismatch:\n{mismatched_args}")
                    else: logger.info("Checkpoint arguments match current (excluding ignored).")
                except Exception as arg_ex: logger.warning(f"Error comparing checkpoint args: {arg_ex}")
            elif hasattr(self, 'args') and self.args is not None: logger.warning("Ckpt has no saved arguments.")

            model_to_load.to(self.device); logger.info(f"Checkpoint '{os.path.basename(filepath)}' loaded. Resume Epoch {start_epoch} (Glob Step {self.global_step})")
            return start_epoch
        except FileNotFoundError:
            logger.error(f"Load ckpt failed: File not found '{filepath}'")
            return 0
        except Exception as e:
            logger.error(f"Failed load ckpt '{filepath}': {e}", exc_info=True)
            return 0

    def train(self, epochs: int, start_epoch: int = 0):
        """Main training loop over multiple epochs."""
        self.current_epoch = start_epoch
        if self.is_main:
            logger.info(f"Starting training from Epoch {start_epoch + 1}/{epochs} (Global Step: {self.global_step}).")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            if self.is_main:
                logger.info(f"--- Starting Epoch {epoch + 1}/{epochs} ---")

            # Set Epoch for Samplers and Datasets (important for DDP and shuffling in IterableDataset)
            if isinstance(self.train_loader.sampler, DistributedSampler): self.train_loader.sampler.set_epoch(epoch)
            if self.val_loader and isinstance(self.val_loader.sampler, DistributedSampler): self.val_loader.sampler.set_epoch(epoch)
            # Also set epoch for the dataset itself if it supports it (for random seed coordination)
            if hasattr(self.train_loader.dataset, 'set_epoch'): self.train_loader.dataset.set_epoch(epoch)
            if self.val_loader and hasattr(self.val_loader.dataset, 'set_epoch'): self.val_loader.dataset.set_epoch(epoch)

            # Train one epoch
            self._train_epoch()

            # Validate (if validation loader exists)
            val_metrics = self._validate()

            # Save checkpoint at the end of the epoch (main process only)
            if self.is_main:
                save_metrics = val_metrics if val_metrics else None # Use val metrics if available
                if self.save_interval >= 0: # Check if saving is enabled
                    self._save_checkpoint(is_intermediate=False, metrics=save_metrics)

            # Barrier to sync ranks before next epoch
            if self.world_size > 1:
                logger.debug(f"Rank {self.rank} entering end-of-epoch {epoch+1} barrier.")
                torch.distributed.barrier()
                logger.debug(f"Rank {self.rank} exited end-of-epoch {epoch+1} barrier.")

        if self.is_main:
            logger.info(f"Training finished after {epochs} epochs.")


# =====================================================================
# Default Configuration (Copied from impl, ensure consistency)
# =====================================================================
DEFAULT_CONFIG_WUBU = {
    "num_levels": 3,
    "hyperbolic_dims": [128, 64, 32],
    "initial_curvatures": [1.0, 0.5, 0.25],
    "initial_scales": [1.0, 1.0, 1.0],
    "initial_spread_values": [1.0, 1.0, 1.0], # Optional: Defaults to initial_scales if not provided
    "boundary_points_per_level": [5, 4, 3],
    "learnable_curvature": True,
    "learnable_scales": True,
    "learnable_spread": True,
    "curvature_min_value": 1e-6,
    "scale_min_value": 1e-6,
    "spread_min_value": 1e-6,
    "use_level_descriptors": True,
    "use_level_spread": True,
    "level_descriptor_init_scale": 0.01,
    "relative_vector_aggregation": "mean",  # Options: 'mean', 'sum', 'none'
    "tangent_input_combination_dims": [64], # Hidden dims for MLP combining inputs in tangent space
    "use_tangent_flow": True,
    "tangent_flow_type": "mlp",            # Options: 'mlp', 'linear', 'none'
    "tangent_flow_hidden_dim_ratio": 0.5,  # Ratio of dim for hidden layer in MLP flow
    "tangent_flow_scale": 1.0,             # Scaling factor for flow displacement
    # num_transitions should not be in config, it's derived
    "rotation_types": ["so_n", "so_n"],    # Options: 'so_n', 'quat', 'identity' (length num_levels-1)
    "transform_types": ["linear", "linear"], # Options: 'mlp', 'linear' (length num_levels-1)
    "transform_hidden_dims": [None, None], # Hidden dims for MLP transforms (length num_levels-1, None uses default)
    "aggregation_method": "concat_tangent", # Options: 'concat_tangent' (more to come?)
    "dropout": 0.1,
}


# =====================================================================
# Argument Parsing (Adjusted for new WuBu config keys)
# =====================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu Nesting Model Trainer (v0.03)")

    # --- DDP ---
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP. -1 means not using DDP.')

    # --- Data ---
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data (.npy file)')
    parser.add_argument('--val_data_path', type=str, default=None, help='Path to the validation data (.npy file)')
    parser.add_argument('--context_window', type=int, default=512, help='Sequence length for training')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of training data to use')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers per GPU')

    # --- Training ---
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Global batch size (across all GPUs)')
    parser.add_argument('--grad_accum_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--no_amp', action='store_true', help='Disable Automatic Mixed Precision')
    parser.add_argument('--detect_anomaly', action='store_true', help='Enable autograd anomaly detection (slow)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # --- Checkpointing & Logging ---
    parser.add_argument('--checkpoint_dir', type=str, default='wubu_nest_checkpoints_v03', help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N global steps')
    parser.add_argument('--save_interval', type=int, default=2000, help='Save intermediate checkpoint every N global steps (0 to disable)')

    # --- WandB ---
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='WuBuNesting', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team)')

    # --- Sequence Model Config ---
    parser.add_argument('--local_hidden_size', type=int, default=256, help='Hidden size for local encoder/decoder')
    parser.add_argument('--decoder_memory_dim', type=int, default=512, help='Output dimension of WuBu model (input to decoder memory)')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of layers in HAKMEM Local Encoder')
    parser.add_argument('--num_encoder_heads', type=int, default=8, help='Number of heads in HAKMEM Local Encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of layers in HAKMEM Local Decoder')
    parser.add_argument('--num_decoder_heads', type=int, default=8, help='Number of heads in HAKMEM Local Decoder')
    parser.add_argument('--n_gram_sizes', type=int, nargs='+', default=[3, 4], help='N-gram sizes for local encoder')
    parser.add_argument('--n_gram_vocab_size', type=int, default=30000, help='Vocab size for N-gram embeddings')
    # MODIFIED: Use BooleanOptionalAction for consistency
    parser.add_argument('--use-hierarchical-decoder', action=argparse.BooleanOptionalAction, default=True, help='Use hierarchical prediction head (default: True)')

    # --- WuBu Nesting Config (Overrides Defaults) ---
    # Note: Use default values from DEFAULT_CONFIG_WUBU where appropriate
    parser.add_argument('--num_levels', type=int, default=DEFAULT_CONFIG_WUBU['num_levels'], help='Number of hyperbolic levels')
    parser.add_argument('--hyperbolic_dims', type=int, nargs='+', default=None, help='Dimensions for each hyperbolic level (overrides default)')
    parser.add_argument('--initial_curvatures', type=float, nargs='+', default=None, help='Initial curvatures for each level (overrides default)')
    parser.add_argument('--initial_scales', type=float, nargs='+', default=None, help='Initial scales for each level (overrides default)')
    parser.add_argument('--initial_spread_values', type=float, nargs='+', default=None, help='Initial spread values (defaults to scales if None)') # None default
    parser.add_argument('--boundary_points_per_level', type=int, nargs='+', default=None, help='Number of boundary points per level (overrides default)')

    # Boolean flags with defaults matching the config dict
    # Using 'dest' ensures the args attribute has the original name used in the config dict
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
    parser.add_argument('--relative_vector_aggregation', type=str, default=DEFAULT_CONFIG_WUBU['relative_vector_aggregation'], choices=['mean', 'sum', 'none'], help='Aggregation method for relative vectors')
    parser.add_argument('--tangent_input_combination_dims', type=int, nargs='+', default=DEFAULT_CONFIG_WUBU['tangent_input_combination_dims'], help='Hidden dims for tangent input combiner MLP')
    parser.add_argument('--tangent_flow_type', type=str, default=DEFAULT_CONFIG_WUBU['tangent_flow_type'], choices=['mlp', 'linear', 'none'], help='Type of tangent flow module')
    parser.add_argument('--tangent_flow_hidden_dim_ratio', type=float, default=DEFAULT_CONFIG_WUBU['tangent_flow_hidden_dim_ratio'], help='Hidden dim ratio for MLP tangent flow')
    parser.add_argument('--tangent_flow_scale', type=float, default=DEFAULT_CONFIG_WUBU['tangent_flow_scale'], help='Scaling factor for tangent flow displacement')
    # Rotation/Transform types and hidden dims are kept using defaults unless user modifies the DEFAULT_CONFIG_WUBU dict or this script
    parser.add_argument('--aggregation_method', type=str, default=DEFAULT_CONFIG_WUBU['aggregation_method'], choices=['concat_tangent'], help='Method to aggregate level outputs')
    parser.add_argument('--dropout', type=float, default=DEFAULT_CONFIG_WUBU['dropout'], help='Dropout rate for various components')

    # --- Q-Controller Config ---
    # MODIFIED: Use BooleanOptionalAction and dest
    parser.add_argument('--enable-q-controller', dest='q_controller_enabled', action=argparse.BooleanOptionalAction, default=True, help='Enable the Q-learning optimizer controller (default: True)')
    parser.add_argument('--q_learning_rate', type=float, default=0.02, help='Learning rate for Q-controller')
    parser.add_argument('--q_discount', type=float, default=0.95, help='Discount factor for Q-controller')
    parser.add_argument('--q_epsilon', type=float, default=0.25, help='Initial epsilon for Q-controller exploration')
    parser.add_argument('--q_epsilon_decay', type=float, default=0.9999, help='Epsilon decay rate for Q-controller')
    parser.add_argument('--q_min_epsilon', type=float, default=0.02, help='Minimum epsilon for Q-controller')

    args = parser.parse_args()

    # --- Post-processing and Validation of Arguments ---
    # Fill in None list arguments with defaults
    if args.hyperbolic_dims is None: args.hyperbolic_dims = DEFAULT_CONFIG_WUBU['hyperbolic_dims']
    if args.initial_curvatures is None: args.initial_curvatures = DEFAULT_CONFIG_WUBU['initial_curvatures']
    if args.initial_scales is None: args.initial_scales = DEFAULT_CONFIG_WUBU['initial_scales']
    if args.boundary_points_per_level is None: args.boundary_points_per_level = DEFAULT_CONFIG_WUBU['boundary_points_per_level']

    # Handle defaults that depend on other args (like spread values)
    if args.initial_spread_values is None:
        args.initial_spread_values = args.initial_scales # Default to scales if not provided
    # Now validate lengths against num_levels
    num_levels = args.num_levels
    list_args_to_check = {
        'hyperbolic_dims': num_levels,
        'initial_curvatures': num_levels,
        'initial_scales': num_levels,
        'boundary_points_per_level': num_levels,
        'initial_spread_values': num_levels
    }
    for arg_name, expected_len in list_args_to_check.items():
        current_list = getattr(args, arg_name)
        if len(current_list) != expected_len:
            parser.error(f"--{arg_name} requires {expected_len} values for --num_levels={num_levels}, but got {len(current_list)}")

    # Set defaults for transition params based on num_levels if needed
    # These are harder to provide via CLI, so use defaults from config dict later
    # This ensures they are available in the args namespace if needed, but the final config
    # construction in run() will handle the correct lengths.
    args.rotation_types = DEFAULT_CONFIG_WUBU['rotation_types']
    args.transform_types = DEFAULT_CONFIG_WUBU['transform_types']
    args.transform_hidden_dims = DEFAULT_CONFIG_WUBU['transform_hidden_dims']

    return args

# =====================================================================
# Distributed Setup Utilities
# =====================================================================
def setup_distributed(local_rank):
    """Initializes DDP if local_rank is non-negative."""
    ddp_active = local_rank != -1
    device = torch.device("cpu")
    rank = 0
    world_size = 1

    if ddp_active:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested (local_rank >= 0) but CUDA is not available.")
        if not is_initialized():
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            master_addr = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '12355') # Default port

            # Ensure MASTER_ADDR and MASTER_PORT are set if WORLD_SIZE > 1
            # torchrun usually handles this, but good practice to check
            if os.getenv('WORLD_SIZE') is not None and int(os.getenv('WORLD_SIZE')) > 1:
                 if os.getenv('MASTER_ADDR') is None or os.getenv('MASTER_PORT') is None:
                     logger.warning("MASTER_ADDR or MASTER_PORT not set for multi-node DDP. Using defaults.")

            try:
                # torchrun sets rank and world_size env variables
                init_process_group(
                    backend=backend,
                    timeout=timedelta(seconds=1800) # 30 min timeout
                    # rank and world_size are inferred from env vars by torchrun
                )
                rank = get_rank()
                world_size = get_world_size()
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
                logger.info(f"DDP Initialized: Rank {rank}/{world_size} on {device} using {backend}. Master: {master_addr}:{master_port}")
            except Exception as e:
                logger.error(f"DDP Initialization failed: {e}", exc_info=True)
                raise
        else:
            # Already initialized (e.g., nested script call?)
            rank = get_rank()
            world_size = get_world_size()
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            logger.warning(f"DDP already initialized: Rank {rank}/{world_size} on {device}.")
    else:
        # DDP not used, check for single GPU or use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0") # Default to first GPU
            torch.cuda.set_device(device)
            logger.info("DDP disabled. Using single GPU: cuda:0")
        else:
            device = torch.device("cpu")
            logger.info("DDP disabled. Using CPU.")

    return ddp_active, device, rank, world_size

def is_main_process():
    """Checks if the current process is the main process (rank 0)."""
    if is_initialized():
        return get_rank() == 0
    return True # Assume main process if DDP is not initialized


# =====================================================================
# Main Execution Logic (Adjusted for new WuBu config building)
# =====================================================================
def run():
    args = parse_arguments()
    # Determine if main process early for initial logging setup
    temp_is_main = args.local_rank == -1 or args.local_rank == 0
    initial_log_level = logging.INFO if temp_is_main else logging.WARNING

    # Configure root logger based on whether it's the main process
    # Remove default handlers to avoid duplicate messages if re-configuring
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]: root_logger.removeHandler(h) # Clear existing handlers
    logging.basicConfig(level=initial_log_level, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True) # Force applies config

    # Setup DDP and get final rank/device
    ddp_active, device, rank, world_size = setup_distributed(args.local_rank)
    am_main_process = is_main_process() # Use the helper function now DDP is setup

    # Re-configure log level based on actual rank AFTER DDP setup
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.getLogger().setLevel(log_level)

    # Setup file logging ONLY on the main process
    if am_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log_filename = os.path.join(args.checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log")
        try:
            file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO) # Log INFO level and above to file
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Main process (Rank {rank}) logging to: {log_filename}")
        except Exception as e:
            logger.error(f"Failed file logging Rank {rank}: {e}")

    logger.info("="*60 + f"\n--- WuBu Nesting Trainer (v0.03 - Rank {rank}/{world_size}) ---")
    logger.info(f"Status | DDP: {'Active' if ddp_active else 'Inactive'} | Device: {device} | Main: {am_main_process}")
    if am_main_process: logger.info("--- Args ---\n" + "\n".join([f"  --{k}: {v}" for k,v in sorted(vars(args).items())]) + "\n" + "-"*20)
    logger.info(f"System | OS={platform.system()}/{platform.release()}, Py={sys.version.split()[0]}, Torch={torch.__version__}, CUDA={'Yes ('+torch.version.cuda+')' if torch.cuda.is_available() else 'No'}")
    logger.info("="*60)

    # Set seed for reproducibility
    seed = args.seed + rank; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed} for Rank {rank}")

    # --- Build Configuration Dicts (Using DEFAULT_CONFIG_WUBU as base) ---
    wubu_config = DEFAULT_CONFIG_WUBU.copy()
    # Update wubu_config from args namespace
    # This assumes args have been validated/corrected by parse_arguments
    for key in wubu_config.keys():
        if hasattr(args, key):
            arg_val = getattr(args, key)
            # Handle BooleanOptionalAction correctly - it's never None if default is set
            if isinstance(wubu_config[key], bool) and isinstance(arg_val, bool):
                 wubu_config[key] = arg_val
            elif arg_val is not None: # For non-boolean or lists/ints/floats where None means override
                wubu_config[key] = arg_val
            # else: keep default if arg wasn't provided (arg_val might be None for list args, handled before)

    # Ensure transition list lengths match num_levels - 1 AFTER updating num_levels
    num_levels = wubu_config['num_levels']
    num_transitions = max(0, num_levels - 1)
    transition_keys = ['rotation_types', 'transform_types', 'transform_hidden_dims']
    for key in transition_keys:
         # Check if key exists and if its length needs correction
         if key in wubu_config and isinstance(wubu_config[key], list) and len(wubu_config[key]) != num_transitions:
             logger.warning(f"Correcting length of WuBu config '{key}'. Expected {num_transitions}, got {len(wubu_config[key])}. Repeating first element.")
             # Repeat the first element or use default if list is empty
             first_val = wubu_config[key][0] if wubu_config[key] else DEFAULT_CONFIG_WUBU[key][0] # Handle empty list case
             wubu_config[key] = [first_val] * num_transitions

    sequence_config = { k: getattr(args, k) for k in [
        "local_hidden_size", "decoder_memory_dim", "context_window", "n_gram_sizes",
        "n_gram_vocab_size", "num_encoder_layers", "num_decoder_layers",
        "num_encoder_heads", "num_decoder_heads" ] if hasattr(args, k)}
    # MODIFIED: Use the correctly parsed arg name
    sequence_config["use_hierarchical_decoder"] = args.use_hierarchical_decoder
    sequence_config["vocab_size"] = 256 # Hardcoded for byte-level

    if am_main_process:
        logger.info("--- Final WuBu Config ---"); [logger.info(f"  {k}: {v}") for k,v in sorted(wubu_config.items())]
        logger.info("--- Final Sequence Config ---"); [logger.info(f"  {k}: {v}") for k,v in sorted(sequence_config.items())]
        logger.info("--------------------------")

    # --- Init WandB ---
    use_wandb = args.wandb and am_main_process and WANDB_AVAILABLE
    if use_wandb:
        try:
            # Combine all configs for logging
            full_config = {**vars(args), "wubu_config": wubu_config, "sequence_config": sequence_config}
            # Sanitize config for wandb (handle lists, None, etc.)
            sanitized_config = {}
            for k, v in full_config.items():
                 if isinstance(v, list): sanitized_config[k] = tuple(v)
                 elif v is None: sanitized_config[k] = 'None'
                 elif isinstance(v, argparse.Namespace): sanitized_config[k] = vars(v) # Handle nested Namespace if any
                 else: sanitized_config[k] = v

            run_name = f"WuBuV03_L{wubu_config['num_levels']}_D{'x'.join(map(str, wubu_config['hyperbolic_dims']))}_B{args.batch_size}_LR{args.learning_rate:.1e}_{datetime.now().strftime('%H%M')}"
            # Attempt to generate a potentially reusable ID, or let wandb handle it
            try:
                run_id = hashlib.sha1(f"{run_name}_{args.seed}".encode()).hexdigest()[:10]
            except Exception:
                run_id = wandb.util.generate_id()

            wandb.init(
                 project=args.wandb_project,
                 entity=args.wandb_entity,
                 config=sanitized_config,
                 name=run_name,
                 resume="allow", # Allow resuming if run_id exists
                 id=run_id if args.resume is None else None # Use generated ID for new runs, allow wandb to handle resume ID
            )
            logger.info(f"WandB initialized: project='{args.wandb_project}', run='{wandb.run.name}' (ID: {wandb.run.id})")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}. Disabling.", exc_info=True)
            use_wandb = False

    # --- Load Datasets ---
    train_dataset = None; val_dataset = None
    try:
        logger.info(f"Loading train dataset: {args.data_path}")
        train_dataset = ByteIterableDataset(args.data_path, args.context_window, args.data_fraction); train_dataset.set_seed(args.seed)
        if args.val_data_path and os.path.exists(args.val_data_path):
            logger.info(f"Loading validation dataset: {args.val_data_path}")
            val_dataset = ByteIterableDataset(args.val_data_path, args.context_window, 1.0); val_dataset.set_seed(args.seed + 1) # Use different seed offset for val
        elif args.val_data_path: logger.warning(f"Val path specified but not found: {args.val_data_path}. Skipping val.")
    except Exception as e:
        logger.error(f"Dataset init failed: {e}", exc_info=True)
        if ddp_active: destroy_process_group() # Cleanup DDP if data fails
        sys.exit(1) # Exit if dataset loading fails

    # --- Create DataLoaders ---
    batch_size_per_gpu = max(1, args.batch_size // world_size) if args.batch_size >= world_size else 1
    effective_bs = batch_size_per_gpu * world_size * args.grad_accum_steps
    logger.info(f"Batch Cfg | Global:{args.batch_size} PerGPU:{batch_size_per_gpu} Accum:{args.grad_accum_steps} Eff:{effective_bs}")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed, drop_last=True) if ddp_active else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if ddp_active and val_dataset else None
    base_seed_offset = args.seed * world_size
    # Use functools.partial to pass arguments correctly to worker_init_fn
    train_worker_init_fn = functools.partial(seed_worker, base_seed=base_seed_offset, rank_offset=rank * args.num_workers) if args.num_workers > 0 else None
    val_worker_init_fn = functools.partial(seed_worker, base_seed=base_seed_offset + 1, rank_offset=rank * args.num_workers) if args.num_workers > 0 and val_dataset else None
    # persistent_workers can cause issues, use carefully or disable if problems arise
    use_persistent_workers = (args.num_workers > 0) and (platform.system() != 'Windows')
    logger.info(f"Persistent workers: {use_persistent_workers} (Workers: {args.num_workers}, OS: {platform.system()})")
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=train_worker_init_fn, persistent_workers=use_persistent_workers, shuffle=False) # Shuffle must always be false due to dataloader error
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False, worker_init_fn=val_worker_init_fn, persistent_workers=use_persistent_workers, shuffle=False) if val_dataset else None

    # --- Initialize Model ---
    try:
        model = WuBuNestingSequenceModel(wubu_config=wubu_config, sequence_config=sequence_config).to(device)
        if am_main_process:
            total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model Init | Total Params: {total_params:,} | Trainable: {trainable_params:,}")
    except Exception as model_ex:
        logger.error(f"Model init failed: {model_ex}", exc_info=True)
        if ddp_active: destroy_process_group()
        sys.exit(1) # Exit if model init fails

    # --- Wrap Model for DDP ---
    if ddp_active:
        find_unused = False # Set True only for DDP debugging if encountering errors
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=find_unused, gradient_as_bucket_view=True)
        logger.info(f"Model DDP wrapped Rank {rank} (find_unused={find_unused}).");
        # Ensure all ranks have wrapped the model before proceeding
        logger.debug(f"Rank {rank} entering barrier after DDP wrap.")
        torch.distributed.barrier()
        logger.debug(f"Rank {rank} exited barrier after DDP wrap.")

    # --- Initialize Optimizer ---
    # MODIFIED: Use the correct boolean flag name from args
    q_cfg = None if not args.q_controller_enabled else {"learning_rate": args.q_learning_rate, "discount": args.q_discount, "epsilon": args.q_epsilon, "epsilon_decay": args.q_epsilon_decay, "min_epsilon": args.q_min_epsilon}
    try:
        optimizer = HAKMEMEnhancedSGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, q_learning_config=q_cfg)
        logger.info(f"Optimizer '{type(optimizer).__name__}' Rank {rank} | BaseLR={args.learning_rate}, WD={args.weight_decay}.")
    except Exception as optim_ex:
        logger.error(f"Optimizer init failed: {optim_ex}", exc_info=True)
        if ddp_active: destroy_process_group()
        sys.exit(1) # Exit if optimizer init fails

    # --- Initialize Trainer ---
    trainer = Trainer(model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader, grad_accum_steps=args.grad_accum_steps, use_amp=(not args.no_amp), log_interval=args.log_interval, save_interval=args.save_interval, checkpoint_dir=args.checkpoint_dir, wandb_enabled=use_wandb, max_grad_norm=args.max_grad_norm, rank=rank, world_size=world_size, detect_anomaly=args.detect_anomaly)
    trainer.args = args # Store args in trainer for checkpointing

    # --- Load Checkpoint ---
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
             logger.info(f"Rank {rank}: Attempting resume from: {args.resume}")
             start_epoch = trainer.load_checkpoint(args.resume)
        else: logger.warning(f"Rank {rank}: Resume ckpt not found: {args.resume}. Starting fresh.")
        # Barrier after loading checkpoint to ensure all ranks loaded before training
        if ddp_active: logger.debug(f"Rank {rank} barrier after ckpt load attempt."); torch.distributed.barrier(); logger.debug(f"Rank {rank} exited barrier.")
    else: logger.info(f"Rank {rank}: Starting fresh.")

    # --- Start Training ---
    if ddp_active: torch.distributed.barrier() # Final barrier before training loop
    training_successful = False
    try:
        trainer.train(args.epochs, start_epoch=start_epoch)
        training_successful = True
    except KeyboardInterrupt:
        logger.info(f"Training interrupted by user (Rank {rank}).")
        training_successful = True # Consider interrupted training potentially successful for saving
    except Exception as train_ex:
        logger.error(f"Critical training error Rank {rank}: {train_ex}", exc_info=True)
        training_successful = False
    finally:
        if am_main_process:
            logger.info("Saving final checkpoint...")
            metrics = getattr(trainer, 'last_val_metrics', None)
            trainer._save_checkpoint(is_intermediate=False, metrics=metrics) # Save final state

        # DDP Cleanup
        if ddp_active:
            logger.debug(f"Rank {rank} final cleanup barrier.")
            torch.distributed.barrier()
            logger.debug(f"Rank {rank} destroying process group.")
            try:
                destroy_process_group()
                logger.info(f"DDP group destroyed (Rank {rank}).")
            except Exception as ddp_err:
                logger.error(f"DDP destroy error Rank {rank}: {ddp_err}")

        # WandB Cleanup
        if use_wandb and wandb.run:
            logger.info("Finishing WandB...")
            try:
                wandb.finish()
                logger.info("WandB finished.")
            except Exception as wb_err:
                logger.error(f"WandB finish error: {wb_err}")

    logger.info(f"Script finished (Rank {rank}). Status: {'Success' if training_successful else 'Failed'}")
# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    # Recommended launch command:
    # torchrun --standalone --nproc_per_node=NUM_GPUS WuBuNest_Trainer.py [ARGS]
    # Replace NUM_GPUS with the number of GPUs you want to use.
    run()
