#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WuBuHypCD.py - WuBu Nesting Model Implementation (v2.0 - Rotation Enhanced)

Implements the WuBu Nesting conceptual framework with:
- Nested hyperbolic spaces (Poincaré Ball via geoopt).
- Learnable scales and curvatures per level.
- Learnable tangent space rotations (SO(n) via geoopt) applied between levels.
- Boundary Manifolds represented by learnable points.
- Simultaneous rotation of primary data and boundary points in tangent space.
- Generation of relative tangent vectors (for potential downstream use).
- Aggregation of tangent space representations from all levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds.poincare import PoincareBall
from geoopt.linalg import OrthogonalMatrix # For SO(n) rotations
import math
import logging
from typing import List, Tuple, Dict, Optional

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
EPS = 1e-7 # Small epsilon for numerical stability

# --- Configuration (Example based on previous discussions) ---
DEFAULT_CONFIG = {
    "input_dim": 784,           # Example: MNIST
    "initial_embedding_dim": 128,
    "num_levels": 3,            # Number of nested levels
    "hyperbolic_dims": [64, 48, 32], # Dimension for Outer, Middle, Inner
    "curvatures": [1.0, 0.8, 1.2],   # Curvature for each level (learnable recommended)
    "learnable_curvature": False,     # Option to make curvatures learnable
    "learnable_scales": True,
    "initial_scales": [1.0, 0.8, 0.6], # Initial relative scale/density
    "scale_min_values": [1e-4, 1e-5, 1e-6], # Minimum allowed scale
    "boundary_points_per_level": [5, 4, 3], # Number of "circle" points per level
    "rotation_types": ['so_n', 'so_n'], # Rotation type for T(1->2), T(2->3) ('so_n' or potentially 'quat' if dim=4)
    "transform_types": ['mlp', 'mlp'],   # Non-rotational transform type T(1->2), T(2->3) ('mlp', 'linear')
    "transform_hidden_dims": [56, 40],  # Hidden dim for MLP transforms
    "output_dim": 10,                  # Example: MNIST classes
    "aggregation_method": "concat_tangent", # How to combine level outputs
    "dropout": 0.1
}
# Note: If rotation_types included 'quat', corresponding hyperbolic_dims must be divisible by 4.

# --- Helper Functions ---
def check_manifold_params(c, scale, min_scale):
    """Ensure curvature is positive and scale meets minimum."""
    c_clamped = torch.clamp(c, min=EPS)
    scale_clamped = torch.clamp(scale, min=min_scale)
    return c_clamped, scale_clamped

# --- WuBu Nesting Components ---

class WuBuNestingLevel(nn.Module):
    """Represents a single level in the WuBu Nesting hierarchy."""
    def __init__(self, level_idx, dim, initial_curvature, initial_scale,
                 learnable_curvature=False, learnable_scale=True, scale_min_value=1e-6):
        super().__init__()
        self.level_idx = level_idx
        self.dim = dim
        self.scale_min_value = scale_min_value

        # Initialize curvature (ensure positive)
        init_c = torch.tensor([max(initial_curvature, EPS)], dtype=torch.float)
        if learnable_curvature:
            self.curvature = nn.Parameter(init_c)
        else:
            self.register_buffer('curvature', init_c)

        # Initialize scale (ensure >= min_value)
        init_s = torch.tensor([max(initial_scale, self.scale_min_value)], dtype=torch.float)
        if learnable_scale:
            self.scale = nn.Parameter(init_s)
        else:
            self.register_buffer('scale', init_s)

        # Manifold instance (curvature will be updated dynamically)
        self.manifold = PoincareBall(c=self.curvature.item()) # Initial c

        # Placeholder for future intra-ball processing (e.g., hyperbolic attention/GCN)
        self.intra_ball_processor = nn.Identity()

    def forward(self, v_tangent_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input tangent vector through this hyperbolic level.
        Args:
            v_tangent_in: Input vector in the tangent space at the origin.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_hyp_processed: Point representation in the hyperbolic ball after processing.
                - v_tangent_out: Output vector in the tangent space at the origin.
        """
        device = v_tangent_in.device
        current_c, current_scale = check_manifold_params(
            self.curvature, self.scale, self.scale_min_value
        )
        current_c = current_c.to(device)
        current_scale = current_scale.to(device)
        self.manifold.c = current_c.item() # Update manifold curvature

        # 1. Scale tangent vector
        v_scaled = v_tangent_in * current_scale

        # 2. Map to hyperbolic ball
        x_hyp = self.manifold.expmap0(v_scaled)
        x_hyp = self.manifold.projx(x_hyp) # Project for stability

        # 3. (Optional) Intra-ball processing
        x_hyp_processed = self.intra_ball_processor(x_hyp)
        x_hyp_processed = self.manifold.projx(x_hyp_processed) # Project again if processed

        # 4. Map back to tangent space
        v_out_scaled = self.manifold.logmap0(x_hyp_processed)

        # 5. Unscale
        # Add eps to prevent division by zero if scale somehow becomes zero
        v_tangent_out = v_out_scaled / (current_scale + EPS)

        # Stability check
        if not torch.isfinite(v_tangent_out).all():
             logger.warning(f"Level {self.level_idx}: NaN/Inf detected in output tangent vector. Replacing.")
             v_tangent_out = torch.nan_to_num(v_tangent_out)

        return x_hyp_processed, v_tangent_out


class BoundaryManifold(nn.Module):
    """Represents boundary sub-manifolds via learnable points in a hyperbolic ball."""
    def __init__(self, level_idx, num_points, point_dim, initial_curvature, init_scale=0.01):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim
        # Initialize points in tangent space near origin, then project
        # Requires manifold object from the level
        tangent_points = torch.randn(num_points, point_dim) * init_scale
        # Store the tangent vectors; projection happens dynamically using the level's manifold
        self.tangent_points = nn.Parameter(tangent_points)
        logger.info(f"Initialized BoundaryManifold for Level {level_idx} with {num_points} points of dim {point_dim}.")

    def get_points(self, manifold: PoincareBall) -> torch.Tensor:
        """Get current boundary points in the hyperbolic ball."""
        # Project current tangent points into the ball using the provided manifold settings
        points_hyp = manifold.expmap0(self.tangent_points)
        points_hyp = manifold.projx(points_hyp) # Ensure stability
        return points_hyp

    def get_tangent_vectors(self, manifold: PoincareBall) -> torch.Tensor:
        """Get the tangent vectors corresponding to the current boundary points."""
        # Map current points back to tangent space
        # Option 1: Use the stored tangent points directly (might drift if points updated differently)
        # return self.tangent_points
        # Option 2: Map the current *ball* positions back (more consistent if points were updated in ball)
        points_hyp = self.get_points(manifold)
        tangent_vectors = manifold.logmap0(points_hyp)
        return tangent_vectors

    def get_rotated_tangent_vectors(self, manifold: PoincareBall, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Get tangent vectors and apply rotation."""
        tangent_vectors = self.get_tangent_vectors(manifold) # Shape: [num_points, dim]
        # Apply rotation: R @ v^T -> transpose result back
        # Rotation matrix shape: [dim, dim]
        # Ensure batch dimension isn't needed if applying same rotation to all points' vectors
        rotated_tangent_vectors = torch.matmul(tangent_vectors, rotation_matrix.T) # Correct matmul order
        return rotated_tangent_vectors


class TangentSpaceRotation(nn.Module):
    """Applies a learnable rotation in the tangent space."""
    def __init__(self, dim, rotation_type='so_n'):
        super().__init__()
        self.dim = dim
        self.rotation_type = rotation_type

        if rotation_type == 'so_n':
            # Use geoopt OrthogonalMatrix with Stiefel manifold constraint for SO(n)
            # Parameterize using exponential map from skew-symmetric matrices (Lie Algebra)
            self.rotation = OrthogonalMatrix(dim, dim, triv=geoopt.linalg.expm)
            logger.info(f"Initialized SO({dim}) rotation using expm.")
        elif rotation_type == 'quat':
            if dim != 4:
                raise ValueError("Quaternion rotation only applicable for dim=4.")
            # Parameterize unit quaternion on S^3 manifold
            # TODO: Implement Quaternion rotation logic
            self.quat_manifold = geoopt.manifolds.Sphere()
            # Store 4 params, project to unit sphere for unit quaternion
            self.quat_params = nn.Parameter(torch.randn(4))
            logger.info("Initialized Quaternion rotation (S^3 parameterization).")
        else:
            raise ValueError(f"Unsupported rotation type: {rotation_type}")

    def _get_so_n_matrix(self) -> torch.Tensor:
        """Returns the current SO(n) rotation matrix."""
        # The OrthogonalMatrix layer directly represents the matrix
        # We might need to ensure det=1 for SO(n), geoopt might handle this via parameterization
        # or we might need a check/projection. Check geoopt docs.
        # For now, assume self.rotation IS the SO(n) matrix.
        mat = self.rotation() # Call the layer to get the matrix
        # Optional: Ensure determinant is +1 (projection to SO(n))
        # U, S, V = torch.linalg.svd(mat)
        # det = torch.linalg.det(torch.matmul(U, V.T))
        # V_corrected = V.clone()
        # V_corrected[-1, :] *= det.sign() # Ensure det=1
        # return torch.matmul(U, V_corrected.T)
        return mat # Trust geoopt parameterization for now

    def _apply_quat_rotation(self, v: torch.Tensor) -> torch.Tensor:
        """Applies quaternion rotation to a batch of 4D vectors."""
        # v shape: [..., 4]
        if self.dim != 4: raise RuntimeError("Quaternion rotation called on non-4D space.")
        # 1. Get unit quaternion q = w + xi + yj + zk from parameters
        unit_quat = self.quat_manifold.projx(self.quat_params) # Project to unit sphere
        w, x, y, z = unit_quat[0], unit_quat[1], unit_quat[2], unit_quat[3]
        # 2. Represent input vector v as pure quaternion p = (0, vx, vy, vz)
        p_shape = v.shape
        p = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1) # Prepend 0 part
        # 3. Compute conjugate q_conj = (w, -x, -y, -z)
        q_conj = torch.tensor([w, -x, -y, -z], device=v.device, dtype=v.dtype)
        # 4. Compute rotated vector: p' = q * p * q_conj
        # Need quaternion multiplication function (Hamilton product)
        # qp = hamilton_product(unit_quat.unsqueeze(0).expand_as(p), p) # Batch quat mul
        # p_prime = hamilton_product(qp, q_conj.unsqueeze(0).expand_as(p))
        # For now, let's *placeholder* this with matrix equivalent for simplicity
        # TODO: Replace with actual batched quaternion multiplication
        # Build rotation matrix from quaternion
        # R = ... build 4x4 matrix R from w,x,y,z ...
        # rotated_v = torch.matmul(v.view(-1, 4), R.T).view(p_shape)
        logger.warning("Quaternion rotation not fully implemented, returning identity.")
        rotated_v = v # Placeholder
        # 5. Extract vector part from p'
        return rotated_v # Return vector part [..., 4]

    def forward(self, v_main: torch.Tensor, v_boundaries: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Applies the *same* learned rotation to the main vector and all boundary vectors."""
        if self.rotation_type == 'so_n':
            R = self._get_so_n_matrix() # Get the SO(n) matrix [dim, dim]
            # Apply rotation: R @ v^T -> transpose result back
            v_main_rotated = torch.matmul(v_main, R.T)
            v_boundaries_rotated = [torch.matmul(v_b, R.T) for v_b in v_boundaries]
        elif self.rotation_type == 'quat':
            # Apply quaternion rotation (needs proper batch implementation)
            v_main_rotated = self._apply_quat_rotation(v_main)
            v_boundaries_rotated = [self._apply_quat_rotation(v_b) for v_b in v_boundaries]
        else:
            v_main_rotated, v_boundaries_rotated = v_main, v_boundaries # Should not happen

        return v_main_rotated, v_boundaries_rotated


class InterLevelTransform(nn.Module):
    """Non-rotational transformation between tangent spaces."""
    def __init__(self, in_dim, out_dim, transform_type, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.transform_type = transform_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        if transform_type == 'mlp':
            h_dim = hidden_dim if hidden_dim is not None else (in_dim + out_dim) // 2
            h_dim = max(h_dim, 1) # Ensure hidden dim is positive
            self.transform = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, out_dim)
            )
            self._init_weights(self.transform)
            logger.info(f"Initialized MLP Transform ({in_dim} -> {h_dim} -> {out_dim})")
        elif transform_type == 'linear':
            self.transform = nn.Linear(in_dim, out_dim)
            self._init_weights(self.transform)
            logger.info(f"Initialized Linear Transform ({in_dim} -> {out_dim})")
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def forward(self, v_rotated: torch.Tensor, v_boundaries_rotated: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Applies the transformation to rotated tangent vectors."""
        v_transformed = self.transform(v_rotated)
        v_boundaries_transformed = [self.transform(v_b) for v_b in v_boundaries_rotated]
        return v_transformed, v_boundaries_transformed


# --- Main WuBu Nesting Model ---

class WuBuNestingModel(nn.Module):
    """The WuBu Nesting model with rotations and boundary manifolds."""
    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        num_levels = config["num_levels"]
        if len(config["hyperbolic_dims"]) != num_levels \
           or len(config["curvatures"]) != num_levels \
           or len(config["initial_scales"]) != num_levels \
           or len(config["scale_min_values"]) != num_levels \
           or len(config["boundary_points_per_level"]) != num_levels:
           raise ValueError("Config list lengths for levels do not match num_levels.")
        if len(config["rotation_types"]) != num_levels - 1 \
           or len(config["transform_types"]) != num_levels - 1 \
           or len(config["transform_hidden_dims"]) != num_levels - 1:
           raise ValueError("Config list lengths for transitions do not match num_levels - 1.")

        # 1. Initial Euclidean Encoding
        self.initial_encoder = nn.Sequential(
            nn.Linear(config["input_dim"], config["initial_embedding_dim"] * 2), # Expand more initially
            nn.LayerNorm(config["initial_embedding_dim"] * 2), # Add LayerNorm
            nn.GELU(), # Use GELU
            nn.Dropout(config["dropout"]),
            nn.Linear(config["initial_embedding_dim"] * 2, config["initial_embedding_dim"]),
        )
        self.to_first_tangent = nn.Linear(
            config["initial_embedding_dim"],
            config["hyperbolic_dims"][0] # Dimension of first level
        )
        self._init_weights(self.initial_encoder)
        self._init_weights(self.to_first_tangent)

        # 2. Nested Hyperbolic Levels
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            level = WuBuNestingLevel(
                level_idx=i,
                dim=config["hyperbolic_dims"][i],
                initial_curvature=config["curvatures"][i],
                initial_scale=config["initial_scales"][i],
                learnable_curvature=config["learnable_curvature"],
                learnable_scale=config["learnable_scales"],
                scale_min_value=config["scale_min_values"][i],
            )
            self.levels.append(level)

        # 3. Boundary Manifolds (Learnable Points)
        self.boundaries = nn.ModuleList()
        for i in range(num_levels):
            boundary = BoundaryManifold(
                level_idx=i,
                num_points=config["boundary_points_per_level"][i],
                point_dim=config["hyperbolic_dims"][i],
                initial_curvature=config["curvatures"][i],
                init_scale=0.01 # Initialize points near origin
            )
            self.boundaries.append(boundary)

        # 4. Inter-Level Rotations and Transformations
        self.rotations = nn.ModuleList()
        self.transforms = nn.ModuleList()
        for i in range(num_levels - 1):
            in_dim = config["hyperbolic_dims"][i]
            out_dim = config["hyperbolic_dims"][i+1]
            # Rotation Module
            rot_type = config["rotation_types"][i]
            self.rotations.append(TangentSpaceRotation(dim=in_dim, rotation_type=rot_type))
            # Non-Rotational Transform Module
            trans_type = config["transform_types"][i]
            trans_hidden = config["transform_hidden_dims"][i]
            self.transforms.append(InterLevelTransform(in_dim, out_dim, trans_type, trans_hidden, config["dropout"]))

        # 5. Aggregation and Final Output
        self.aggregation_method = config["aggregation_method"]
        if self.aggregation_method == "concat_tangent":
            total_tangent_dim = sum(config["hyperbolic_dims"])
            final_hidden_dim = max(config["output_dim"] * 4, total_tangent_dim // 2) # Heuristic hidden dim
            self.final_processor = nn.Sequential(
                nn.Linear(total_tangent_dim, final_hidden_dim),
                nn.LayerNorm(final_hidden_dim), # Add LayerNorm
                nn.GELU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(final_hidden_dim, config["output_dim"])
            )
            self._init_weights(self.final_processor)
        else:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not implemented.")

    def _init_weights(self, module):
        """Initialize weights for linear layers."""
        if isinstance(module, nn.Linear):
            # Consider Kaiming He init for ReLU/GELU or Xavier for others
            # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the WuBu Nesting structure."""
        batch_size = x.shape[0]
        device = x.device

        # 1. Initial Encoding -> First Tangent Space
        encoded = self.initial_encoder(x)
        current_tangent_main = self.to_first_tangent(encoded) # v_i+1_in

        # Store tangent outputs from each level for aggregation
        level_tangent_outputs = []

        # --- Process through Levels ---
        for i in range(self.config["num_levels"]):
            level_module = self.levels[i]
            boundary_module = self.boundaries[i]

            # --- Pass through Hyperbolic Level ---
            # Input: current_tangent_main
            # Output: x_hyp_i, v_tangent_i_out
            _, v_tangent_i_out = level_module(current_tangent_main)
            level_tangent_outputs.append(v_tangent_i_out)

            # --- Prepare for next level (if not the last level) ---
            if i < self.config["num_levels"] - 1:
                rotation_module = self.rotations[i]
                transform_module = self.transforms[i]

                # Get boundary points in current level's tangent space
                # Note: This uses the *current* state of the manifold (updated c)
                # Boundary points are parameters, logmap maps them using current curvature
                v_boundaries = boundary_module.get_tangent_vectors(level_module.manifold) # [num_points, dim_i]
                # Expand boundary vectors for batch compatibility (repeat for each batch item)
                v_boundaries_batch = v_boundaries.unsqueeze(0).expand(batch_size, -1, -1) # [B, num_points, dim_i]

                # --- Apply Rotation ---
                # Apply same rotation to main vector and all boundary vectors
                v_main_rotated, v_boundaries_rotated_list = rotation_module(
                    v_tangent_i_out, # Shape [B, dim_i]
                    # Pass boundary vectors reshaped for rotation function if needed,
                    # assuming rotation function handles batching or operates elementwise.
                    # Let's assume rotation handles [B, ..., dim] -> [B, ..., dim]
                    # We need to process each boundary point vector.
                    # Input to rotation needs adjustment. Let's reshape boundary vectors.
                    [v_b.unsqueeze(0).expand(batch_size, -1) for v_b in v_boundaries] # List of [B, dim_i]
                )
                # v_boundaries_rotated list contains rotated versions of each point's tangent vector for the batch

                # --- Apply Non-Rotational Transform ---
                # Transform the rotated main vector and boundary vectors
                v_next_tangent_main, v_boundaries_transformed_list = transform_module(
                    v_main_rotated,
                    v_boundaries_rotated_list
                )

                # --- Compute Relative Vectors (Optional for downstream use) ---
                # Store them or use them immediately? For now, just compute.
                relative_vectors_list = [] # List of tensors, one per boundary point: [B, dim_{i+1}]
                for v_b_transformed in v_boundaries_transformed_list:
                    relative_vectors_list.append(v_next_tangent_main - v_b_transformed)

                # --- Prepare Input for Next Level ---
                current_tangent_main = v_next_tangent_main # This becomes input for level i+1
                # How relative vectors are used is TBD (e.g., passed to next level's intra-ball proc)

        # --- Aggregation ---
        if self.aggregation_method == "concat_tangent":
            # Concatenate the tangent space outputs from all levels
            aggregated_repr = torch.cat(level_tangent_outputs, dim=-1)
        else:
            # Implement other methods if needed
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not implemented.")

        # --- Final Processing ---
        output = self.final_processor(aggregated_repr)

        return output

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Initializing WuBuNestingModel (Rotation Enhanced) ---")
    # Use default config for demonstration
    config = DEFAULT_CONFIG
    model = WuBuNestingModel(config)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Dummy Data ---
    batch_size = 8
    dummy_input = torch.randn(batch_size, config["input_dim"])
    dummy_labels = torch.randint(0, config["output_dim"], (batch_size,))
    print(f"\nInput shape: {dummy_input.shape}")

    # --- Forward Pass ---
    print("\n--- Performing Forward Pass ---")
    try:
        output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")

        # --- Loss Calculation ---
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, dummy_labels)
        print(f"Loss calculation successful: {loss.item():.4f}")

        # --- Backward Pass (Check gradients) ---
        # loss.backward()
        # print("Backward pass successful (gradients computed).")
        # # Optional: Check gradient norms
        # total_norm = 0
        # for p in model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f"Total gradient norm: {total_norm:.4f}")

    except Exception as e:
        print(f"\nAn error occurred during forward/backward pass: {e}")
        logger.exception("Error details:")

    print("\n--- WuBu Nesting Example Finished ---")