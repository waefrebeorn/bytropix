#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WuBuHypCD_Quat.py - WuBu Nesting Model Implementation (v2.2 - SO(4) Rotation, Relative Vec Usage, Vectorized)

Implements the WuBu Nesting conceptual framework with:
- Nested hyperbolic spaces (Poincaré Ball via geoopt).
- Learnable scales and curvatures per level.
- Learnable tangent space rotations:
    - General SO(4) via pairs of unit quaternions (pvq) for 4D spaces.
    - General SO(n) via OrthogonalMatrix for other dimensions.
- Boundary Manifolds represented by learnable points.
- Simultaneous rotation of primary data and boundary points in tangent space.
- Generation and *Utilization* of relative tangent vectors.
- Inter-level transformations using MLPs or QuaternionLinear layers.
- Aggregation of tangent space representations from all levels.
- Vectorized operations on boundary points for efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds.poincare import PoincareBall
from geoopt.linalg import OrthogonalMatrix # For SO(n) rotations
import math
import logging
from typing import List, Tuple, Dict, Optional, Union

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
EPS = 1e-7 # Small epsilon for numerical stability

# --- Configuration (Example emphasizing Quaternions/SO(4)) ---
# Let's make Level 2 4D to demonstrate SO(4)/quat rotation
DEFAULT_CONFIG_QUAT_SO4 = {
    "input_dim": 784,           # Example: MNIST
    "initial_embedding_dim": 128,
    "num_levels": 3,            # Number of nested levels
    "hyperbolic_dims": [64, 4, 32], # Outer(SO(n)), Middle(SO(4)/Quat), Inner(SO(n))
                                    # Inner dim 32 also divisible by 4 for potential QuatLinear
    "curvatures": [1.0, 0.8, 1.2],   # Curvature for each level
    "learnable_curvature": False,
    "learnable_scales": True,
    "initial_scales": [1.0, 0.8, 0.6],
    "scale_min_values": [1e-4, 1e-5, 1e-6],
    "boundary_points_per_level": [5, 3, 4], # Number of "circle" points per level
    # Rotation applied to the *source* tangent space dimension
    "rotation_types": ['so_n', 'quat'], # Rotate T(1->2) in R^64, Rotate T(2->3) in R^4 (SO(4))
    # Non-rotational transform mapping source tangent -> target tangent
    "transform_types": ['mlp', 'mlp'], # T(1->2)=MLP(64->4), T(2->3)=MLP(4->32)
    "transform_hidden_dims": [32, 16],  # Hidden dim for MLP transforms
    "output_dim": 10,
    "aggregation_method": "concat_tangent",
    "relative_vector_aggregation": "mean", # How to aggregate relative vectors ('mean', 'max', 'sum')
    "dropout": 0.1
}

# ==================================
# === Quaternion Utility Functions ===
# ==================================

def check_quat_dim(dim: int, layer_name: str = "Layer"):
    """Check if dimension is divisible by 4 for quaternion operations."""
    if dim % 4 != 0:
        raise ValueError(f"{layer_name} dimension must be divisible by 4 for quaternion operations, but got {dim}")

def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hamilton product of two quaternions (or batches of quaternions).
    Assumes input tensors have shape [..., 4] where the last dim is (w, x, y, z).
    Handles broadcasting.
    """
    # Ensure q1 and q2 are broadcastable
    q1_shape = list(q1.shape); q2_shape = list(q2.shape)
    while len(q1_shape) < len(q2_shape): q1_shape.insert(0, 1)
    while len(q2_shape) < len(q1_shape): q2_shape.insert(0, 1)
    q1 = q1.view(q1_shape)
    q2 = q2.view(q2_shape)

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion q = [w, x, y, z] -> q* = [w, -x, -y, -z]."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def quat_rotate_via_pvq(v: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotates a 4D vector v using the general SO(4) representation pvq.
    v: Tensor shape [..., 4]
    p: Tensor shape broadcastable to v (unit quaternion for left mult)
    q: Tensor shape broadcastable to v (unit quaternion for right mult)
    Returns rotated vector shape [..., 4].
    """
    if v.shape[-1] != 4 or p.shape[-1] != 4 or q.shape[-1] != 4:
        raise ValueError("Inputs must be 4D for quat_rotate_via_pvq")

    # Ensure broadcasting
    p = p.expand_as(v); q = q.expand_as(v)

    pv = hamilton_product(p, v)
    pvq = hamilton_product(pv, q)
    return pvq

# ==============================================
# === Quaternion Linear Layer (As Before) ===
# ==============================================
class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True): # (Implementation unchanged)
        super().__init__(); check_quat_dim(in_features, "QuaternionLinear Input"); check_quat_dim(out_features, "QuaternionLinear Output")
        self.in_features_quat = in_features // 4; self.out_features_quat = out_features // 4
        self.r_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat)); self.i_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat)); self.j_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat)); self.k_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None; self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features_quat);
        for weight in [self.r_weight, self.i_weight, self.j_weight, self.k_weight]: weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: self.bias.data.zero_()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_features_quat * 4; batch_dims = x.shape[:-1]; x_reshaped = x.view(*batch_dims, self.in_features_quat, 4)
        r_x, i_x, j_x, k_x = x_reshaped[..., 0], x_reshaped[..., 1], x_reshaped[..., 2], x_reshaped[..., 3]
        out_r = F.linear(r_x, self.r_weight) - F.linear(i_x, self.i_weight) - F.linear(j_x, self.j_weight) - F.linear(k_x, self.k_weight); out_i = F.linear(r_x, self.i_weight) + F.linear(i_x, self.r_weight) + F.linear(j_x, self.k_weight) - F.linear(k_x, self.j_weight); out_j = F.linear(r_x, self.j_weight) - F.linear(i_x, self.k_weight) + F.linear(j_x, self.r_weight) + F.linear(k_x, self.i_weight); out_k = F.linear(r_x, self.k_weight) + F.linear(i_x, self.j_weight) - F.linear(j_x, self.i_weight) + F.linear(k_x, self.r_weight)
        output = torch.stack([out_r, out_i, out_j, out_k], dim=-1); output = output.view(*batch_dims, self.out_features_quat * 4)
        if self.bias is not None: output = output + self.bias
        return output

# =====================================================
# === WuBu Nesting Components (with Relative Vec Input) ===
# =====================================================

def check_manifold_params(c, scale, min_scale):
    """Ensure curvature is positive and scale meets minimum."""
    c_clamped = torch.clamp(c, min=EPS)
    scale_clamped = torch.clamp(scale, min=min_scale)
    return c_clamped, scale_clamped

class WuBuNestingLevel(nn.Module):
    """Represents a single level, now processing relative vectors."""
    def __init__(self, level_idx, dim, initial_curvature, initial_scale,
                 learnable_curvature=False, learnable_scale=True, scale_min_value=1e-6,
                 relative_vector_aggregation="mean", dropout=0.1):
        super().__init__()
        self.level_idx = level_idx; self.dim = dim; self.scale_min_value = scale_min_value
        self.relative_vector_aggregation = relative_vector_aggregation
        init_c = torch.tensor([max(initial_curvature, EPS)], dtype=torch.float); init_s = torch.tensor([max(initial_scale, self.scale_min_value)], dtype=torch.float)
        self.curvature = nn.Parameter(init_c) if learnable_curvature else self.register_buffer('curvature', init_c)
        self.scale = nn.Parameter(init_s) if learnable_scale else self.register_buffer('scale', init_s)
        self.manifold = PoincareBall(c=self.curvature.item())
        self.intra_ball_processor = nn.Identity() # Placeholder

        # Layer to combine input tangent vector with aggregated relative vectors
        # Input dim = level_dim (main) + level_dim (aggregated relative)
        self.tangent_combiner = nn.Sequential(
            nn.Linear(dim * 2, dim), # Project concatenated vector back to level dim
            nn.GELU(),              # Activation
            nn.Dropout(dropout)     # Dropout
        )
        self._init_weights(self.tangent_combiner)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): nn.init.xavier_uniform_(module.weight);
        if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for layer in module: self._init_weights(layer)

    def forward(self, v_tangent_in: torch.Tensor,
                relative_vectors_stacked: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input tangent vector, incorporating aggregated relative vectors if provided.
        Args:
            v_tangent_in: Input vector in the tangent space at the origin [B, dim].
            relative_vectors_stacked: Stacked relative vectors from previous transition
                                       [B, num_boundary_points, dim]. Optional.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_hyp_processed: Point representation in the hyperbolic ball.
                - v_tangent_out: Output vector in the tangent space.
        """
        device = v_tangent_in.device
        current_c, current_scale = check_manifold_params(self.curvature, self.scale, self.scale_min_value)
        current_c, current_scale = current_c.to(device), current_scale.to(device)
        self.manifold.c = current_c.item()

        # --- Incorporate Relative Vectors ---
        if relative_vectors_stacked is not None and relative_vectors_stacked.numel() > 0:
            # Aggregate relative vectors [B, num_pts, dim] -> [B, dim]
            if self.relative_vector_aggregation == "mean":
                rel_vec_summary = relative_vectors_stacked.mean(dim=1)
            elif self.relative_vector_aggregation == "max":
                rel_vec_summary = relative_vectors_stacked.max(dim=1)[0]
            elif self.relative_vector_aggregation == "sum":
                rel_vec_summary = relative_vectors_stacked.sum(dim=1)
            else: # Default to mean
                rel_vec_summary = relative_vectors_stacked.mean(dim=1)

            # Concatenate and project
            combined_tangent = torch.cat([v_tangent_in, rel_vec_summary], dim=-1)
            v_tangent_processed_in = self.tangent_combiner(combined_tangent)
        else:
            # No relative vectors provided or empty
            v_tangent_processed_in = v_tangent_in # Use input directly

        # --- Hyperbolic Mapping ---
        v_scaled = v_tangent_processed_in * current_scale
        x_hyp = self.manifold.expmap0(v_scaled); x_hyp = self.manifold.projx(x_hyp)
        x_hyp_processed = self.intra_ball_processor(x_hyp); x_hyp_processed = self.manifold.projx(x_hyp_processed)
        v_out_scaled = self.manifold.logmap0(x_hyp_processed)
        v_tangent_out = v_out_scaled / (current_scale + EPS)

        if not torch.isfinite(v_tangent_out).all():
            logger.warning(f"Level {self.level_idx} NaN/Inf detected in output tangent vector. Replacing."); v_tangent_out = torch.nan_to_num(v_tangent_out)
        return x_hyp_processed, v_tangent_out

class BoundaryManifold(nn.Module): # (Unchanged)
    def __init__(self, level_idx, num_points, point_dim, initial_curvature, init_scale=0.01):
        super().__init__(); self.level_idx = level_idx; self.num_points = num_points; self.point_dim = point_dim
        tangent_points = torch.randn(num_points, point_dim) * init_scale
        self.tangent_points = nn.Parameter(tangent_points)
    def get_points(self, manifold: PoincareBall) -> torch.Tensor:
        points_hyp = manifold.expmap0(self.tangent_points); points_hyp = manifold.projx(points_hyp)
        return points_hyp
    def get_tangent_vectors(self, manifold: PoincareBall) -> torch.Tensor:
        points_hyp = self.get_points(manifold); tangent_vectors = manifold.logmap0(points_hyp)
        return tangent_vectors


# =========================================================
# === Rotation and Transformation Modules (Vectorized) ===
# =========================================================

class TangentSpaceRotation(nn.Module):
    """Applies a learnable rotation (SO(n) or SO(4) via Quat) to batched tangent vectors."""
    def __init__(self, dim, rotation_type='so_n'):
        super().__init__(); self.dim = dim; self.rotation_type = rotation_type
        if rotation_type == 'so_n':
            self.rotation = OrthogonalMatrix(dim, dim, triv=geoopt.linalg.expm)
            logger.info(f"Initialized SO({dim}) rotation.")
        elif rotation_type == 'quat':
            if dim != 4: raise ValueError("Quaternion rotation requires dim=4.")
            self.quat_manifold = geoopt.manifolds.Sphere()
            # Learn TWO quaternions for p v q rotation
            self.quat_params_p = nn.Parameter(torch.randn(4)); nn.init.normal_(self.quat_params_p, std=0.1)
            self.quat_params_q = nn.Parameter(torch.randn(4)); nn.init.normal_(self.quat_params_q, std=0.1)
            logger.info("Initialized SO(4) rotation via pvq quaternions.")
        else: raise ValueError(f"Unsupported rotation type: {rotation_type}")

    def _get_rotation_operator(self, device: torch.device) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Gets the operator (SO(n) matrix or pair of unit quaternions p,q)."""
        if self.rotation_type == 'so_n':
            return self.rotation().to(device) # Ensure matrix is on correct device
        elif self.rotation_type == 'quat':
            # Project to unit sphere S^3 and normalize
            unit_p = self.quat_manifold.projx(self.quat_params)
            unit_p = unit_p / (torch.norm(unit_p) + EPS)
            unit_q = self.quat_manifold.projx(self.quat_params_q)
            unit_q = unit_q / (torch.norm(unit_q) + EPS)
            return unit_p.to(device), unit_q.to(device) # Return p, q on correct device

    def forward(self, v_main: torch.Tensor, v_boundaries_stacked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the *same* learned rotation to the main vector and stacked boundary vectors."""
        # v_main shape: [B, dim]
        # v_boundaries_stacked shape: [B, num_points, dim]
        batch_size, num_points, dim = v_boundaries_stacked.shape
        device = v_main.device

        if self.rotation_type == 'so_n':
            R = self._get_rotation_operator(device) # SO(n) matrix [dim, dim]
            # Apply rotation: v @ R.T
            v_main_rotated = torch.matmul(v_main, R.T)
            # Reshape boundaries for batched matrix multiplication
            v_boundaries_flat = v_boundaries_stacked.view(batch_size * num_points, dim)
            v_boundaries_rotated_flat = torch.matmul(v_boundaries_flat, R.T)
            v_boundaries_rotated = v_boundaries_rotated_flat.view(batch_size, num_points, dim)
        elif self.rotation_type == 'quat':
            p, q = self._get_rotation_operator(device) # Unit quaternions p, q [4]
            # Unsqueeze for broadcasting: [1, 4]
            p_b = p.unsqueeze(0); q_b = q.unsqueeze(0)
            # Rotate main vector [B, 4]
            v_main_rotated = quat_rotate_via_pvq(v_main, p_b, q_b)
            # Rotate boundary vectors [B, num_points, 4]
            # Expand p, q to [B, num_points, 4] for element-wise quat mul
            p_bb = p_b.unsqueeze(1).expand(-1, num_points, -1)
            q_bb = q_b.unsqueeze(1).expand(-1, num_points, -1)
            v_boundaries_rotated = quat_rotate_via_pvq(v_boundaries_stacked, p_bb, q_bb)
        else: # Should not happen
            v_main_rotated, v_boundaries_rotated = v_main, v_boundaries_stacked

        # --- Stability Check ---
        if not torch.isfinite(v_main_rotated).all():
            logger.warning("NaN/Inf in TangentSpaceRotation output (main). Replacing."); v_main_rotated = torch.nan_to_num(v_main_rotated)
        if not torch.isfinite(v_boundaries_rotated).all():
            logger.warning(f"NaN/Inf in TangentSpaceRotation output (boundaries). Replacing."); v_boundaries_rotated = torch.nan_to_num(v_boundaries_rotated)
        return v_main_rotated, v_boundaries_rotated


class InterLevelTransform(nn.Module):
    """Non-rotational transformation between tangent spaces (MLP or QuaternionLinear), vectorized."""
    def __init__(self, in_dim, out_dim, transform_type, hidden_dim=None, dropout=0.1):
        super().__init__(); self.transform_type = transform_type; self.in_dim = in_dim; self.out_dim = out_dim
        if transform_type == 'mlp':
            h_dim = hidden_dim if hidden_dim is not None else max(1, (in_dim + out_dim) // 2)
            self.transform = nn.Sequential(nn.Linear(in_dim, h_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(h_dim, out_dim))
            self._init_weights(self.transform); logger.info(f"Initialized MLP Transform ({in_dim} -> {h_dim} -> {out_dim})")
        elif transform_type == 'linear':
            self.transform = nn.Linear(in_dim, out_dim)
            self._init_weights(self.transform); logger.info(f"Initialized Linear Transform ({in_dim} -> {out_dim})")
        elif transform_type == 'quat':
            check_quat_dim(in_dim, "Transform Quat Input"); check_quat_dim(out_dim, "Transform Quat Output")
            self.transform = QuaternionLinear(in_dim, out_dim, bias=True); logger.info(f"Initialized QuaternionLinear Transform ({in_dim} -> {out_dim})")
        else: raise ValueError(f"Unsupported transform_type: {transform_type}")
    def _init_weights(self, module): # (Implementation unchanged)
        if isinstance(module, nn.Linear): nn.init.xavier_uniform_(module.weight);
        if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for layer in module: self._init_weights(layer)
    def forward(self, v_rotated: torch.Tensor, v_boundaries_rotated_stacked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transformation to rotated stacked tangent vectors."""
        # v_rotated shape: [B, in_dim]
        # v_boundaries_rotated_stacked shape: [B, num_points, in_dim]
        v_transformed = self.transform(v_rotated)
        # Linear layers and QuatLinear apply correctly to the last dimension
        v_boundaries_transformed_stacked = self.transform(v_boundaries_rotated_stacked)
        # --- Stability Check ---
        if not torch.isfinite(v_transformed).all(): logger.warning("NaN/Inf in InterLevelTransform output (main). Replacing."); v_transformed = torch.nan_to_num(v_transformed)
        if not torch.isfinite(v_boundaries_transformed_stacked).all(): logger.warning(f"NaN/Inf in InterLevelTransform output (boundaries). Replacing."); v_boundaries_transformed_stacked = torch.nan_to_num(v_boundaries_transformed_stacked)
        return v_transformed, v_boundaries_transformed_stacked


# ==================================
# === Main WuBu Nesting Model ===
# ==================================

class WuBuNestingModel(nn.Module):
    """WuBu Nesting model with SO(4)/SO(n) rotations and relative vector usage."""
    def __init__(self, config=DEFAULT_CONFIG_QUAT_SO4): # Use updated Quat/SO4 config
        super().__init__(); self.config = config; num_levels = config["num_levels"]
        # Config validation (as before)
        if len(config["hyperbolic_dims"])!=num_levels or len(config["rotation_types"])!=num_levels-1 or len(config["transform_types"])!=num_levels-1:
             raise ValueError("Config list lengths mismatch num_levels.")
        # Add validation for rotation/transform types vs dimensions
        for i in range(num_levels - 1):
            if config["rotation_types"][i] == 'quat' and config["hyperbolic_dims"][i] != 4:
                raise ValueError(f"Rotation 'quat' for T({i}->{i+1}) needs dim=4, got {config['hyperbolic_dims'][i]}")
            if config["transform_types"][i] == 'quat':
                check_quat_dim(config["hyperbolic_dims"][i], f"Transform Quat Input (T{i}->{i+1})")
                check_quat_dim(config["hyperbolic_dims"][i+1], f"Transform Quat Output (T{i}->{i+1})")

        # 1. Initial Euclidean Encoding (Unchanged)
        self.initial_encoder = nn.Sequential(
            nn.Linear(config["input_dim"], config["initial_embedding_dim"]*2), nn.LayerNorm(config["initial_embedding_dim"]*2), nn.GELU(),
            nn.Dropout(config["dropout"]), nn.Linear(config["initial_embedding_dim"]*2, config["initial_embedding_dim"]),)
        self.to_first_tangent = nn.Linear(config["initial_embedding_dim"], config["hyperbolic_dims"][0])
        self._init_weights(self.initial_encoder); self._init_weights(self.to_first_tangent)

        # 2. Nested Hyperbolic Levels (Now uses relative vectors)
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            self.levels.append(WuBuNestingLevel(
                level_idx=i, dim=config["hyperbolic_dims"][i], initial_curvature=config["curvatures"][i],
                initial_scale=config["initial_scales"][i], learnable_curvature=config["learnable_curvature"],
                learnable_scale=config["learnable_scales"], scale_min_value=config["scale_min_values"][i],
                relative_vector_aggregation=config["relative_vector_aggregation"], dropout=config["dropout"] )) # Pass aggregation method

        # 3. Boundary Manifolds (Unchanged)
        self.boundaries = nn.ModuleList()
        for i in range(num_levels):
            self.boundaries.append(BoundaryManifold(
                level_idx=i, num_points=config["boundary_points_per_level"][i],
                point_dim=config["hyperbolic_dims"][i], initial_curvature=config["curvatures"][i]))

        # 4. Inter-Level Rotations and Transformations (Unchanged init)
        self.rotations = nn.ModuleList()
        self.transforms = nn.ModuleList()
        for i in range(num_levels - 1):
            in_dim = config["hyperbolic_dims"][i]; out_dim = config["hyperbolic_dims"][i+1]
            rot_type = config["rotation_types"][i]
            self.rotations.append(TangentSpaceRotation(dim=in_dim, rotation_type=rot_type))
            trans_type = config["transform_types"][i]
            trans_hidden = config["transform_hidden_dims"][i]
            self.transforms.append(InterLevelTransform(in_dim, out_dim, trans_type, trans_hidden, config["dropout"]))

        # 5. Aggregation and Final Output (Unchanged init)
        self.aggregation_method = config["aggregation_method"]
        if self.aggregation_method == "concat_tangent":
            total_tangent_dim = sum(config["hyperbolic_dims"])
            final_hidden_dim = max(config["output_dim"] * 4, total_tangent_dim // 2)
            self.final_processor = nn.Sequential(
                nn.Linear(total_tangent_dim, final_hidden_dim), nn.LayerNorm(final_hidden_dim), nn.GELU(),
                nn.Dropout(config["dropout"]), nn.Linear(final_hidden_dim, config["output_dim"]))
            self._init_weights(self.final_processor)
        else: raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not implemented.")

    def _init_weights(self, module): # (Unchanged)
        if isinstance(module, nn.Linear): nn.init.xavier_uniform_(module.weight);
        if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for layer in module: self._init_weights(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]; device = x.device
        encoded = self.initial_encoder(x); current_tangent_main = self.to_first_tangent(encoded)
        level_tangent_outputs = []
        relative_vectors_for_next_level = None # Initialize for first level

        for i in range(self.config["num_levels"]):
            level_module = self.levels[i]; boundary_module = self.boundaries[i]

            # Pass through Hyperbolic Level, passing relative vectors from *previous* transition
            _, v_tangent_i_out = level_module(current_tangent_main, relative_vectors_for_next_level)
            level_tangent_outputs.append(v_tangent_i_out)

            # Perform inter-level transition if not the last level
            if i < self.config["num_levels"] - 1:
                rotation_module = self.rotations[i]; transform_module = self.transforms[i]
                # Get stacked boundary tangent vectors [B, num_points, dim_i]
                v_boundaries = boundary_module.get_tangent_vectors(level_module.manifold) # [num_points, dim_i]
                v_boundaries_stacked = v_boundaries.unsqueeze(0).expand(batch_size, -1, -1) # [B, num_points, dim_i]

                # Apply Rotation (Vectorized)
                v_main_rotated, v_boundaries_rotated_stacked = rotation_module(v_tangent_i_out, v_boundaries_stacked)

                # Apply Transform (Vectorized)
                v_next_tangent_main, v_boundaries_transformed_stacked = transform_module(v_main_rotated, v_boundaries_rotated_stacked)

                # Compute Relative Vectors (Vectorized) [B, num_points, dim_{i+1}]
                # Unsqueeze main vector for broadcasting: [B, 1, dim_{i+1}]
                relative_vectors_stacked = v_next_tangent_main.unsqueeze(1) - v_boundaries_transformed_stacked
                relative_vectors_for_next_level = relative_vectors_stacked # Store for next loop iteration

                # Update main tangent vector for next level's input
                current_tangent_main = v_next_tangent_main

            else: # Last level, no more transitions
                relative_vectors_for_next_level = None # Clear for safety

        # Aggregate and Final Output
        if self.aggregation_method == "concat_tangent":
            aggregated_repr = torch.cat(level_tangent_outputs, dim=-1)
        else: pass
        output = self.final_processor(aggregated_repr)
        return output

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Initializing WuBuNestingModel (SO4 Rotation, Relative Vec Usage, Vectorized) ---")
    config = DEFAULT_CONFIG_QUAT_SO4 # Use the SO(4) config
    # --- Configuration Validation ---
    valid_config = True # (Validation logic as before)
    for i in range(config["num_levels"] - 1):
        if config["rotation_types"][i] == 'quat' and config["hyperbolic_dims"][i] != 4: valid_config=False; print(f"Config Error: rot 'quat' L{i} needs dim 4, got {config['hyperbolic_dims'][i]}")
        if config["transform_types"][i] == 'quat':
            if config["hyperbolic_dims"][i]%4!=0: valid_config=False; print(f"Config Error: trans 'quat' L{i} needs input dim {config['hyperbolic_dims'][i]} div by 4.")
            if config["hyperbolic_dims"][i+1]%4!=0: valid_config=False; print(f"Config Error: trans 'quat' L{i} needs output dim {config['hyperbolic_dims'][i+1]} div by 4.")
    if not valid_config: print("Configuration errors found. Exiting."); exit()
    # --- End Validation ---

    model = WuBuNestingModel(config)
    print(f"Model created with config: {config['hyperbolic_dims']}, Rotations: {config['rotation_types']}, Transforms: {config['transform_types']}")
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
        # Check for NaNs in output
        if not torch.isfinite(output).all():
             print("\nERROR: NaN/Inf detected in final model output!")
        else:
             print("Output seems finite.")

        # --- Loss Calculation ---
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, dummy_labels)
        print(f"Loss calculation successful: {loss.item():.4f}")

    except Exception as e:
        print(f"\nAn error occurred during forward pass: {e}")
        logger.exception("Error details:")

    print("\n--- WuBu Nesting Refined Example Finished ---")