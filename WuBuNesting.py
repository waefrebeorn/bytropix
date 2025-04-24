#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WuBuNesting_Full.py - WuBu Nesting Model Implementation (v3.0 - Full Framework)

Implements the comprehensive WuBu Nesting conceptual framework with:
- Nested hyperbolic spaces (Poincaré Ball via geoopt).
- Learnable scales (s_i) and curvatures (c_i) per level.
- Boundary Manifolds represented by learnable points {b_ijk}.
- Learnable tangent space rotations (R_i: SO(n) or SO(4)/Quat).
- Learnable non-rotational mappings (~T_i: MLP, Linear, QuatLinear).
- Learnable Level Descriptor vectors (ld_i), rotated and mapped.
- Learnable Level Spread parameters (sigma_i), passed as context.
- Learnable Intra-Level Tangent Flows (F_i: Linear or MLP), applied in tangent space.
- Simultaneous transformation (Rotation R_i + Map ~T_i) of primary data, boundary points,
  and level descriptors in tangent space.
- Generation and utilization of relative tangent vectors {d_ijk}.
- Rich intra-level processing utilizing all incoming information.
- Aggregation of tangent space representations from all levels.
- Vectorized operations for efficiency.
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

# --- Default Configuration for Full WuBu Nesting ---
DEFAULT_CONFIG_FULL = {
    "input_dim": 784,           # Example: MNIST
    "initial_embedding_dim": 128,
    "num_levels": 3,
    "hyperbolic_dims": [64, 32, 16], # Example dimensions for each level
    "curvatures": [1.0, 1.0, 1.0],   # Initial curvatures
    "learnable_curvature": True,     # Learn curvatures c_i
    "curvature_min_value": 1e-3,
    "initial_scales": [1.0, 1.0, 1.0], # Initial scales s_i
    "learnable_scales": True,        # Learn scales s_i
    "scale_min_value": 1e-4,
    "boundary_points_per_level": [5, 4, 3], # Num boundary points {b_ijk} per level

    # --- Inter-Level Transition Config ---
    # Rotation R_i applied to *source* tangent space dim H_i
    "rotation_types": ['so_n', 'so_n'], # e.g., T(1->2) use SO(64), T(2->3) use SO(32)
    # Non-rotational map ~T_i : T(H_i) -> T(H_{i+1})
    "transform_types": ['mlp', 'mlp'], # e.g., MLP: R^64 -> R^32, MLP: R^32 -> R^16
    "transform_hidden_dims": [48, 24], # Hidden dims for MLP transforms

    # --- Level Descriptor Config ---
    "use_level_descriptors": True,
    "level_descriptor_init_scale": 0.01, # Initial norm for random ld_i

    # --- Level Spread Config ---
    "use_level_spread": True,
    "initial_spread_values": [0.5, 0.5, 0.5], # Initial sigma_i values
    "learnable_spread": True,
    "spread_min_value": 1e-5,

    # --- Intra-Level Flow Config ---
    "use_tangent_flow": True,
    "tangent_flow_type": 'mlp', # 'none', 'linear', 'mlp'
    "tangent_flow_hidden_dim_ratio": 0.5, # Hidden dim = ratio * level_dim
    "tangent_flow_scale": 0.1, # Multiplier for flow output (controls step size)

    # --- Intra-Level Processing Config ---
    # How to combine inputs (v_in, rel_vecs, ld_in, sigma_in) in tangent space
    "tangent_input_combination_dims": [96, 48], # Hidden dims for MLP combiner

    # --- Aggregation & Output ---
    "aggregation_method": "concat_tangent", # 'concat_tangent'
    "relative_vector_aggregation": "mean",  # How to aggregate {d_ijk} for input ('mean', 'max', 'sum')
    "output_dim": 10,                       # Final classification dim
    "dropout": 0.1
}


# ==================================
# === Quaternion Utility Functions ===
# ==================================
# (Unchanged from previous version)
def check_quat_dim(dim: int, layer_name: str = "Layer"):
    if dim % 4 != 0: raise ValueError(f"{layer_name} dimension must be divisible by 4 for quaternion operations, but got {dim}")
def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1_shape = list(q1.shape); q2_shape = list(q2.shape)
    while len(q1_shape) < len(q2_shape): q1_shape.insert(0, 1)
    while len(q2_shape) < len(q1_shape): q2_shape.insert(0, 1)
    q1 = q1.view(q1_shape); q2 = q2.view(q2_shape)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)
def quat_rotate_via_pvq(v: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if v.shape[-1] != 4 or p.shape[-1] != 4 or q.shape[-1] != 4: raise ValueError("Inputs must be 4D for quat_rotate_via_pvq")
    p = p.expand_as(v); q = q.expand_as(v); pv = hamilton_product(p, v); pvq = hamilton_product(pv, q); return pvq

# ==============================================
# === Quaternion Linear Layer ===
# ==============================================
# (Unchanged from previous version)
class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
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


# ============================================
# === Utility for Parameter Initialization ===
# ============================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# ============================================
# === Parameter Constraint Handling        ===
# ============================================
def get_constrained_params(param, min_val=EPS):
    """Helper to get positive value from potentially unconstrained param."""
    # Simple softplus approach to ensure positivity
    return F.softplus(param) + min_val

# =====================================================
# === WuBu Nesting Components (Full Framework)      ===
# =====================================================

class TangentFlow(nn.Module):
    """Implements the Intra-Level Tangent Flow F_i."""
    def __init__(self, dim, flow_type='mlp', hidden_dim_ratio=0.5, dropout=0.1):
        super().__init__()
        self.flow_type = flow_type
        if flow_type == 'linear':
            self.flow_map = nn.Linear(dim, dim)
            logger.info(f"Initialized Linear Tangent Flow (dim={dim})")
        elif flow_type == 'mlp':
            hidden_dim = max(16, int(dim * hidden_dim_ratio)) # Ensure minimum hidden dim
            self.flow_map = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            )
            logger.info(f"Initialized MLP Tangent Flow (dim={dim}, hidden={hidden_dim})")
        elif flow_type == 'none':
            self.flow_map = nn.Identity()
            logger.info("Tangent Flow disabled for this level.")
        else:
            raise ValueError(f"Unsupported tangent_flow_type: {flow_type}")
        self.flow_map.apply(init_weights)

    def forward(self, v_tangent: torch.Tensor) -> torch.Tensor:
        """Applies the flow. Assumes additive flow: v_out = v_in + flow(v_in)."""
        if self.flow_type == 'none':
            return torch.zeros_like(v_tangent) # No flow displacement

        flow_displacement = self.flow_map(v_tangent)

        # Stability check
        if not torch.isfinite(flow_displacement).all():
            logger.warning("NaN/Inf detected in TangentFlow output. Replacing with zeros.")
            flow_displacement = torch.nan_to_num(flow_displacement, nan=0.0, posinf=0.0, neginf=0.0)

        return flow_displacement


class WuBuNestingLevel(nn.Module):
    """
    Represents a single level in the WuBu Nesting framework.
    Handles intra-level processing including tangent input combination, flow,
    and hyperbolic mapping. Owns its level-specific learnable parameters.
    """
    def __init__(self, level_idx, config: Dict):
        super().__init__()
        self.level_idx = level_idx
        self.dim = config["hyperbolic_dims"][level_idx]
        self.relative_vector_aggregation = config["relative_vector_aggregation"]
        self.dropout = config["dropout"]
        self.use_ld = config["use_level_descriptors"]
        self.use_spread = config["use_level_spread"]
        self.use_flow = config["use_tangent_flow"]

        # --- Learnable Geometric Parameters ---
        # Curvature (learn log(c - min) + min via softplus)
        self.curvature_min = config["curvature_min_value"]
        init_c_unconstrained = torch.tensor(math.log(max(EPS, config["curvatures"][level_idx] - self.curvature_min)), dtype=torch.float)
        if config["learnable_curvature"]:
            self.unconstrained_curvature = nn.Parameter(init_c_unconstrained)
        else:
            self.register_buffer('unconstrained_curvature', init_c_unconstrained)

        # Scale (learn log(s - min) + min via softplus)
        self.scale_min = config["scale_min_value"]
        init_s_unconstrained = torch.tensor(math.log(max(EPS, config["initial_scales"][level_idx] - self.scale_min)), dtype=torch.float)
        if config["learnable_scales"]:
            self.unconstrained_scale = nn.Parameter(init_s_unconstrained)
        else:
            self.register_buffer('unconstrained_scale', init_s_unconstrained)

        # Level Descriptor Vector (ld_i)
        if self.use_ld:
            self.level_descriptor = nn.Parameter(torch.randn(self.dim) * config["level_descriptor_init_scale"])
        else:
            self.register_buffer('level_descriptor', torch.zeros(self.dim))

        # Level Spread (sigma_i, learn log(sigma - min) + min via softplus)
        self.spread_min = config["spread_min_value"]
        init_spread_unconstrained = torch.tensor(math.log(max(EPS, config["initial_spread_values"][level_idx] - self.spread_min)), dtype=torch.float)
        if self.use_spread and config["learnable_spread"]:
            self.unconstrained_spread = nn.Parameter(init_spread_unconstrained)
        else:
            self.register_buffer('unconstrained_spread', init_spread_unconstrained)

        # --- Manifold ---
        self.manifold = PoincareBall(c=1.0) # Initial c, will be updated dynamically

        # --- Intra-Level Processing Modules ---
        # 1. Tangent Input Combiner (MLP)
        combiner_input_dim = self.dim # Main input tangent v_in
        if config["relative_vector_aggregation"] != 'none':
             combiner_input_dim += self.dim # Aggregated relative vectors d_agg
        if self.use_ld:
             combiner_input_dim += self.dim # Level descriptor ld_in
        if self.use_spread:
             combiner_input_dim += 1 # Spread sigma_in (scalar)

        combiner_hidden_dims = config.get("tangent_input_combination_dims", [combiner_input_dim // 2]) # Default hidden layer
        layers = []
        in_d = combiner_input_dim
        for h_dim in combiner_hidden_dims:
            layers.extend([nn.Linear(in_d, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(self.dropout)])
            in_d = h_dim
        layers.append(nn.Linear(in_d, self.dim)) # Project back to level dimension
        self.tangent_combiner = nn.Sequential(*layers)
        self.tangent_combiner.apply(init_weights)
        logger.info(f"Level {level_idx} Tangent Combiner: Input Dim={combiner_input_dim}, Output Dim={self.dim}")

        # 2. Intra-Level Tangent Flow (F_i)
        if self.use_flow:
            self.tangent_flow = TangentFlow(
                dim=self.dim,
                flow_type=config["tangent_flow_type"],
                hidden_dim_ratio=config["tangent_flow_hidden_dim_ratio"],
                dropout=self.dropout
            )
            self.tangent_flow_scale = config["tangent_flow_scale"]
        else:
            self.tangent_flow = None
            self.tangent_flow_scale = 0.0

        # 3. Placeholder for further intra-ball processing (e.g., hyperbolic attention)
        self.intra_ball_processor = nn.Identity()

    def get_current_geometry(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns current positive curvature and scale on the correct device."""
        current_c = get_constrained_params(self.unconstrained_curvature, self.curvature_min).to(device)
        current_s = get_constrained_params(self.unconstrained_scale, self.scale_min).to(device)
        return current_c, current_s

    def get_current_spread(self, device) -> torch.Tensor:
        """Returns current positive spread value on the correct device."""
        if not self.use_spread:
            return torch.tensor(0.0, device=device) # Return 0 if spread not used
        return get_constrained_params(self.unconstrained_spread, self.spread_min).to(device)


    def forward(self,
                v_tangent_in: torch.Tensor, # From previous level's transform [B, dim]
                relative_vectors_stacked: Optional[torch.Tensor] = None, # {d_i} from prev transform [B, num_bnd_pts, dim]
                ld_tangent_in: Optional[torch.Tensor] = None, # Transformed ld_{i-1} [B, dim]
                sigma_in: Optional[torch.Tensor] = None       # sigma_{i-1} [B, 1]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes inputs for the current level.
        Returns:
            - x_hyp_processed: Final hyperbolic state for this level [B, dim].
            - v_tangent_out: Tangent vector output for next transition [B, dim].
            - ld_param_out: This level's descriptor parameter [dim] (for next rotation).
            - sigma_param_out: This level's spread parameter [1] (for next context).
        """
        batch_size = v_tangent_in.shape[0]
        device = v_tangent_in.device

        # --- Get Current Geometric Params ---
        current_c, current_scale = self.get_current_geometry(device)
        self.manifold.c = current_c.item() # Update manifold curvature
        current_spread = self.get_current_spread(device) # This level's own spread

        # --- 1. Aggregate & Combine Tangent Inputs ---
        inputs_to_combine = [v_tangent_in]

        # Aggregate relative vectors
        if relative_vectors_stacked is not None and self.relative_vector_aggregation != 'none':
            if relative_vectors_stacked.numel() > 0:
                agg_fn = getattr(torch, self.relative_vector_aggregation, torch.mean)
                # Handle max aggregation returning tuple
                if self.relative_vector_aggregation == 'max':
                    rel_vec_summary = agg_fn(relative_vectors_stacked, dim=1)[0]
                else:
                    rel_vec_summary = agg_fn(relative_vectors_stacked, dim=1)
                inputs_to_combine.append(rel_vec_summary)
            else: # Handle empty relative vectors if aggregation needed
                 inputs_to_combine.append(torch.zeros_like(v_tangent_in))
        elif self.relative_vector_aggregation != 'none': # If aggregation is enabled but no vectors provided
            inputs_to_combine.append(torch.zeros_like(v_tangent_in))


        # Add level descriptor input (ld_{i-1})
        if self.use_ld:
            if ld_tangent_in is None: # Handle first level case
                ld_tangent_in = torch.zeros_like(v_tangent_in)
            inputs_to_combine.append(ld_tangent_in)

        # Add spread context input (sigma_{i-1})
        if self.use_spread:
            if sigma_in is None: # Handle first level case
                sigma_in = torch.zeros(batch_size, 1, device=device)
            # Ensure sigma_in is [B, 1] for concatenation
            inputs_to_combine.append(sigma_in.view(batch_size, 1))

        # Concatenate all inputs
        combined_tangent_inputs = torch.cat(inputs_to_combine, dim=-1)

        # Project combined inputs back to level dimension
        v_tangent_combined = self.tangent_combiner(combined_tangent_inputs)

        # --- 2. Apply Intra-Level Tangent Flow (F_i) ---
        flow_displacement = torch.zeros_like(v_tangent_combined)
        if self.use_flow and self.tangent_flow is not None:
            flow_displacement = self.tangent_flow(v_tangent_combined)

        # Apply additive flow with scaling
        v_tangent_flowed = v_tangent_combined + flow_displacement * self.tangent_flow_scale

        # Stability check after flow
        if not torch.isfinite(v_tangent_flowed).all():
            logger.warning(f"Level {self.level_idx} NaN/Inf detected after tangent flow. Replacing.");
            v_tangent_flowed = torch.nan_to_num(v_tangent_flowed)

        # --- 3. Map to Hyperbolic Space & Process ---
        # Apply scale s_i before ExpMap
        v_scaled = v_tangent_flowed * current_scale
        # Numerical stability: Clip norm before expmap
        # max_norm = (1.0 / math.sqrt(current_c.item())) - EPS # Max norm slightly inside boundary
        # v_norm = torch.norm(v_scaled, dim=-1, keepdim=True)
        # v_clipped = v_scaled * torch.clamp(max_norm / (v_norm + EPS), max=1.0)

        x_hyp = self.manifold.expmap0(v_scaled)
        x_hyp = self.manifold.projx(x_hyp) # Ensure point is strictly inside the ball

        # Apply intra-ball processing (currently identity)
        x_hyp_processed = self.intra_ball_processor(x_hyp)
        x_hyp_processed = self.manifold.projx(x_hyp_processed) # Project again after processing

        # --- 4. Map Back to Tangent Space for Output ---
        v_out_scaled = self.manifold.logmap0(x_hyp_processed)
        v_tangent_out = v_out_scaled / (current_scale + EPS) # Apply inverse scale 1/s_i

        # --- Final Stability Checks ---
        if not torch.isfinite(v_tangent_out).all():
            logger.warning(f"Level {self.level_idx} NaN/Inf detected in output tangent vector. Replacing.");
            v_tangent_out = torch.nan_to_num(v_tangent_out)

        # --- Return values ---
        # Detach parameters before returning if they are used directly in next stage's graph
        ld_param_out = self.level_descriptor if self.use_ld else torch.zeros(self.dim, device=device)
        sigma_param_out = current_spread.unsqueeze(0) # Ensure [1] shape

        return x_hyp_processed, v_tangent_out, ld_param_out, sigma_param_out


class BoundaryManifold(nn.Module):
    """Represents learnable boundary points {b_ijk}."""
    def __init__(self, level_idx, num_points, point_dim, init_scale=0.01):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim
        # Initialize tangent points near origin
        tangent_points = torch.randn(num_points, point_dim) * init_scale
        self.tangent_points = nn.Parameter(tangent_points)

    def get_points(self, manifold: PoincareBall) -> torch.Tensor:
        """Get boundary points in the hyperbolic ball."""
        # Project points onto the manifold using current geometry
        points_hyp = manifold.expmap0(self.tangent_points)
        points_hyp = manifold.projx(points_hyp)
        return points_hyp

    def get_tangent_vectors(self, manifold: PoincareBall) -> torch.Tensor:
        """Get boundary points as vectors in the tangent space at origin."""
        points_hyp = self.get_points(manifold)
        # Ensure points are valid before logmap
        if not torch.isfinite(points_hyp).all():
             logger.warning(f"Boundary points for level {self.level_idx} have NaN/Inf. Resetting tangent points slightly?")
             # Simple reset strategy (could be improved)
             self.tangent_points.data.normal_(0, 0.01)
             points_hyp = manifold.expmap0(self.tangent_points); points_hyp = manifold.projx(points_hyp)

        tangent_vectors = manifold.logmap0(points_hyp)
        if not torch.isfinite(tangent_vectors).all():
            logger.warning(f"Level {self.level_idx} NaN/Inf in boundary tangent vectors after logmap. Replacing.");
            tangent_vectors = torch.nan_to_num(tangent_vectors)
        return tangent_vectors


# =========================================================
# === Rotation and Transformation Modules (Updated Inputs) ===
# =========================================================

class TangentSpaceRotation(nn.Module):
    """Applies learnable rotation (SO(n) or SO(4)) to main, boundary, and descriptor vectors."""
    def __init__(self, dim, rotation_type='so_n'):
        super().__init__(); self.dim = dim; self.rotation_type = rotation_type
        if rotation_type == 'so_n':
            # Use expm parameterization for stability
            self.rotation = OrthogonalMatrix(dim, dim, triv=geoopt.linalg.expm)
            logger.info(f"Initialized SO({dim}) rotation (via expm).")
        elif rotation_type == 'quat':
            if dim != 4: raise ValueError("Quaternion rotation requires dim=4.")
            # Use geoopt sphere manifold for projection
            self.quat_manifold = geoopt.manifolds.Sphere()
            self.quat_params_p = nn.Parameter(torch.randn(4)); nn.init.normal_(self.quat_params_p, std=0.01) # Smaller init std
            self.quat_params_q = nn.Parameter(torch.randn(4)); nn.init.normal_(self.quat_params_q, std=0.01)
            logger.info("Initialized SO(4) rotation via pvq quaternions.")
        else: raise ValueError(f"Unsupported rotation type: {rotation_type}")

    def _get_rotation_operator(self, device: torch.device) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Gets the operator (SO(n) matrix or pair of unit quaternions p,q)."""
        if self.rotation_type == 'so_n':
            return self.rotation().to(device) # Get current rotation matrix
        elif self.rotation_type == 'quat':
            # Project parameters onto S^3 and normalize for stability
            unit_p = self.quat_manifold.projx(self.quat_params_p)
            unit_p = unit_p / (torch.norm(unit_p, p=2, dim=-1, keepdim=True) + EPS)
            unit_q = self.quat_manifold.projx(self.quat_params_q)
            unit_q = unit_q / (torch.norm(unit_q, p=2, dim=-1, keepdim=True) + EPS)
            return unit_p.to(device), unit_q.to(device)

    def forward(self,
                v_main: torch.Tensor,              # [B, dim]
                v_boundaries_stacked: torch.Tensor,# [B, num_points, dim]
                v_descriptor: torch.Tensor         # [dim] or [1, dim]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies the *same* learned rotation to all input vectors."""
        batch_size, num_points, dim = v_boundaries_stacked.shape
        device = v_main.device
        v_descriptor = v_descriptor.to(device).view(1, dim) # Ensure [1, dim]

        v_main_rotated, v_boundaries_rotated, v_descriptor_rotated = None, None, None

        if self.rotation_type == 'so_n':
            R = self._get_rotation_operator(device) # SO(n) matrix [dim, dim]
            # Apply rotation: v @ R.T (or v @ R if R is inverse convention)
            # geoopt OrthogonalMatrix returns R such that y = x @ R
            v_main_rotated = torch.matmul(v_main, R)
            # Batched rotation for boundaries
            v_boundaries_flat = v_boundaries_stacked.reshape(batch_size * num_points, dim)
            v_boundaries_rotated_flat = torch.matmul(v_boundaries_flat, R)
            v_boundaries_rotated = v_boundaries_rotated_flat.reshape(batch_size, num_points, dim)
            # Rotate descriptor
            v_descriptor_rotated = torch.matmul(v_descriptor, R) # [1, dim]

        elif self.rotation_type == 'quat':
            p, q = self._get_rotation_operator(device) # Unit quaternions p, q [4]
            # Add batch dim for broadcasting: [1, 4]
            p_b = p.unsqueeze(0); q_b = q.unsqueeze(0)
            # Rotate main vector [B, 4]
            v_main_rotated = quat_rotate_via_pvq(v_main, p_b, q_b)
            # Rotate boundary vectors [B, num_points, 4]
            p_bb = p_b.unsqueeze(1).expand(-1, num_points, -1)
            q_bb = q_b.unsqueeze(1).expand(-1, num_points, -1)
            v_boundaries_rotated = quat_rotate_via_pvq(v_boundaries_stacked, p_bb, q_bb)
            # Rotate descriptor [1, 4]
            v_descriptor_rotated = quat_rotate_via_pvq(v_descriptor, p_b, q_b)

        # --- Stability Checks ---
        outputs = [v_main_rotated, v_boundaries_rotated, v_descriptor_rotated]
        names = ["main", "boundaries", "descriptor"]
        final_outputs = []
        for i, out in enumerate(outputs):
            if out is None: continue # Should not happen if logic is correct
            if not torch.isfinite(out).all():
                logger.warning(f"NaN/Inf in TangentSpaceRotation output ({names[i]}). Replacing.");
                out = torch.nan_to_num(out)
            final_outputs.append(out)

        return tuple(final_outputs)


class InterLevelTransform(nn.Module):
    """Non-rotational transformation ~T_i (MLP, Linear, QuatLinear), vectorized."""
    def __init__(self, in_dim, out_dim, transform_type, hidden_dim=None, dropout=0.1):
        super().__init__(); self.transform_type = transform_type; self.in_dim = in_dim; self.out_dim = out_dim
        if transform_type == 'mlp':
            h_dim = hidden_dim if hidden_dim is not None else max(16, (in_dim + out_dim) // 2)
            self.transform = nn.Sequential(
                nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(h_dim, out_dim)
            )
            logger.info(f"Initialized MLP Transform ({in_dim} -> {h_dim} -> {out_dim})")
        elif transform_type == 'linear':
            self.transform = nn.Linear(in_dim, out_dim)
            logger.info(f"Initialized Linear Transform ({in_dim} -> {out_dim})")
        elif transform_type == 'quat':
            check_quat_dim(in_dim, "Transform Quat Input"); check_quat_dim(out_dim, "Transform Quat Output")
            self.transform = QuaternionLinear(in_dim, out_dim, bias=True);
            logger.info(f"Initialized QuaternionLinear Transform ({in_dim} -> {out_dim})")
        else: raise ValueError(f"Unsupported transform_type: {transform_type}")
        self.transform.apply(init_weights)

    def forward(self,
                v_rotated: torch.Tensor,                # [B, in_dim]
                v_boundaries_rotated_stacked: torch.Tensor, # [B, num_points, in_dim]
                v_descriptor_rotated: torch.Tensor      # [1, in_dim]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies the transformation to all input vectors."""
        # Apply transform (works correctly on last dimension)
        v_transformed = self.transform(v_rotated)
        v_boundaries_transformed_stacked = self.transform(v_boundaries_rotated_stacked)
        v_descriptor_transformed = self.transform(v_descriptor_rotated) # [1, out_dim]

        # --- Stability Checks ---
        outputs = [v_transformed, v_boundaries_transformed_stacked, v_descriptor_transformed]
        names = ["main", "boundaries", "descriptor"]
        final_outputs = []
        for i, out in enumerate(outputs):
             if not torch.isfinite(out).all():
                logger.warning(f"NaN/Inf in InterLevelTransform output ({names[i]}). Replacing.");
                out = torch.nan_to_num(out)
             final_outputs.append(out)

        return tuple(final_outputs)


# ==================================
# === Main WuBu Nesting Model    ===
# ==================================

class WuBuNestingModel(nn.Module):
    """
    Full WuBu Nesting model implementing the comprehensive framework.
    """
    def __init__(self, config=DEFAULT_CONFIG_FULL):
        super().__init__()
        self.config = config
        num_levels = config["num_levels"]

        # --- Configuration Validation ---
        assert len(config["hyperbolic_dims"]) == num_levels
        assert len(config["curvatures"]) == num_levels
        assert len(config["initial_scales"]) == num_levels
        assert len(config["boundary_points_per_level"]) == num_levels
        assert len(config["initial_spread_values"]) == num_levels
        assert len(config["rotation_types"]) == num_levels - 1
        assert len(config["transform_types"]) == num_levels - 1
        assert len(config["transform_hidden_dims"]) == num_levels - 1
        # Add more validation checks as needed (e.g., quat dims)
        for i in range(num_levels - 1):
            if config["rotation_types"][i] == 'quat' and config["hyperbolic_dims"][i] != 4:
                raise ValueError(f"Rotation 'quat' for T({i}->{i+1}) needs dim=4, got {config['hyperbolic_dims'][i]}")
            if config["transform_types"][i] == 'quat':
                check_quat_dim(config["hyperbolic_dims"][i], f"Transform Quat Input (T{i}->{i+1})")
                check_quat_dim(config["hyperbolic_dims"][i+1], f"Transform Quat Output (T{i}->{i+1})")
        logger.info("WuBu Nesting Configuration Validated.")

        # 1. Initial Euclidean Encoding
        self.initial_encoder = nn.Sequential(
            nn.Linear(config["input_dim"], config["initial_embedding_dim"]),
            nn.LayerNorm(config["initial_embedding_dim"]), nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["initial_embedding_dim"], config["initial_embedding_dim"]),
        )
        self.initial_encoder.apply(init_weights)
        # Project to the tangent space of the first level
        self.to_first_tangent = nn.Linear(config["initial_embedding_dim"], config["hyperbolic_dims"][0])
        self.to_first_tangent.apply(init_weights)
        logger.info("Initialized Initial Encoder.")

        # 2. Nested Hyperbolic Levels (with internal parameters)
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            self.levels.append(WuBuNestingLevel(level_idx=i, config=config))
        logger.info(f"Initialized {num_levels} WuBu Nesting Levels.")

        # 3. Boundary Manifolds (parameterized by points)
        self.boundaries = nn.ModuleList()
        for i in range(num_levels):
            self.boundaries.append(BoundaryManifold(
                level_idx=i,
                num_points=config["boundary_points_per_level"][i],
                point_dim=config["hyperbolic_dims"][i],
                init_scale=0.01 # Small initial tangent norms
            ))
        logger.info("Initialized Boundary Manifolds.")

        # 4. Inter-Level Rotations (R_i)
        self.rotations = nn.ModuleList()
        for i in range(num_levels - 1):
            in_dim = config["hyperbolic_dims"][i]
            rot_type = config["rotation_types"][i]
            self.rotations.append(TangentSpaceRotation(dim=in_dim, rotation_type=rot_type))
        logger.info("Initialized Inter-Level Rotations.")

        # 5. Inter-Level Transformations (~T_i)
        self.transforms = nn.ModuleList()
        for i in range(num_levels - 1):
            in_dim = config["hyperbolic_dims"][i]
            out_dim = config["hyperbolic_dims"][i+1]
            trans_type = config["transform_types"][i]
            trans_hidden = config["transform_hidden_dims"][i] if trans_type == 'mlp' else None
            self.transforms.append(InterLevelTransform(
                in_dim, out_dim, trans_type, trans_hidden, config["dropout"]
            ))
        logger.info("Initialized Inter-Level Transforms.")

        # 6. Aggregation and Final Output
        self.aggregation_method = config["aggregation_method"]
        if self.aggregation_method == "concat_tangent":
            # Aggregate the output tangent vectors v_i^out from each level
            total_tangent_dim = sum(config["hyperbolic_dims"])
            final_hidden_dim = max(config["output_dim"] * 2, total_tangent_dim // 2) # Example heuristic
            self.final_processor = nn.Sequential(
                nn.Linear(total_tangent_dim, final_hidden_dim),
                nn.LayerNorm(final_hidden_dim), nn.GELU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(final_hidden_dim, config["output_dim"])
            )
            self.final_processor.apply(init_weights)
            logger.info(f"Initialized Final Processor (Concat Tangent: {total_tangent_dim} -> {config['output_dim']}).")
        else:
            raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full WuBu Nesting architecture."""
        batch_size = x.shape[0]
        device = x.device

        # --- Initial Encoding ---
        encoded = self.initial_encoder(x)
        current_tangent_main = self.to_first_tangent(encoded) # v_1^in

        # --- Initialize Loop Variables ---
        level_tangent_outputs = []  # To store v_i^out for final aggregation
        relative_vectors_for_next_level = None # d_{i+1}
        current_ld_tangent = None              # Transformed ld_i passed to level i+1
        current_sigma = None                   # sigma_i passed to level i+1

        # --- Recursive Nesting Loop ---
        for i in range(self.config["num_levels"]):
            level_module = self.levels[i]
            boundary_module = self.boundaries[i]

            # --- Intra-Level Processing ---
            # Pass all available info into the level
            x_hyp_processed, v_tangent_i_out, ld_i_param, sigma_i_param = level_module(
                v_tangent_in=current_tangent_main,
                relative_vectors_stacked=relative_vectors_for_next_level,
                ld_tangent_in=current_ld_tangent,
                sigma_in=current_sigma
            )

            # Store the output tangent vector for aggregation
            level_tangent_outputs.append(v_tangent_i_out) # Store v_i^out

            # --- Inter-Level Transition (if not the last level) ---
            if i < self.config["num_levels"] - 1:
                rotation_module = self.rotations[i]
                transform_module = self.transforms[i]

                # Get tangent vectors for boundary points {b_ijk}
                v_boundaries = boundary_module.get_tangent_vectors(level_module.manifold) # [num_bnd_pts, dim_i]
                # Stack for batch: [B, num_bnd_pts, dim_i]
                v_boundaries_stacked = v_boundaries.unsqueeze(0).expand(batch_size, -1, -1)

                # Apply Rotation R_i (to v_i^out, {v_bijk}, ld_i^param)
                v_main_rotated, v_boundaries_rotated_stacked, ld_rotated = rotation_module(
                    v_main=v_tangent_i_out,
                    v_boundaries_stacked=v_boundaries_stacked,
                    v_descriptor=ld_i_param # Use this level's own descriptor parameter
                )

                # Apply Non-Rotational Map ~T_i
                v_next_tangent_main, v_boundaries_transformed_stacked, ld_next_tangent = transform_module(
                    v_rotated=v_main_rotated,
                    v_boundaries_rotated_stacked=v_boundaries_rotated_stacked,
                    v_descriptor_rotated=ld_rotated
                )
                # Note: ld_next_tangent is the transformed ld_i, becomes ld_{i+1}^in

                # Compute Relative Vectors d_{i+1}
                # Unsqueeze main vector [B, 1, dim_{i+1}] for broadcasting
                relative_vectors_stacked = v_next_tangent_main.unsqueeze(1) - v_boundaries_transformed_stacked #[B, num_pts, dim_{i+1}]

                # --- Update variables for the *next* loop iteration ---
                current_tangent_main = v_next_tangent_main              # Input for level i+1
                relative_vectors_for_next_level = relative_vectors_stacked # Input for level i+1
                current_ld_tangent = ld_next_tangent.expand(batch_size, -1) # Input for level i+1 (expand batch dim)
                current_sigma = sigma_i_param.expand(batch_size, -1)       # Context for level i+1 (expand batch dim)

            else:
                # Last level, clear context for safety (though loop terminates)
                relative_vectors_for_next_level = None
                current_ld_tangent = None
                current_sigma = None

        # --- Final Aggregation and Output ---
        if self.aggregation_method == "concat_tangent":
            # Concatenate the output tangent vectors from all levels
            aggregated_repr = torch.cat(level_tangent_outputs, dim=-1)
        else:
            # Implement other aggregation methods if needed
            raise NotImplementedError

        # Final classification head
        output = self.final_processor(aggregated_repr)

        return output

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Initializing Full WuBuNestingModel (v3.0) ---")
    config = DEFAULT_CONFIG_FULL
    try:
        model = WuBuNestingModel(config)
        print(f"\nModel created successfully with config:")
        for key, val in config.items(): print(f"  {key}: {val}")

        print(f"\nTotal trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # --- Dummy Data ---
        batch_size = 16
        dummy_input = torch.randn(batch_size, config["input_dim"])
        dummy_labels = torch.randint(0, config["output_dim"], (batch_size,))
        print(f"\nInput shape: {dummy_input.shape}")

        # --- Forward Pass ---
        print("\n--- Performing Forward Pass ---")
        model.train() # Set model to training mode (affects dropout etc.)
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

        # --- Backward Pass (Optional Check) ---
        # print("\n--- Performing Backward Pass (Checking Gradients) ---")
        # model.zero_grad()
        # loss.backward()
        # grad_check = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad]
        # if all(grad_check):
        #     print("Backward pass successful, gradients seem finite.")
        # else:
        #     print("ERROR: Found None or NaN/Inf gradients during backward pass!")
        #     # Further inspection might be needed here

    except ValueError as ve:
        print(f"\nCONFIG ERROR: {ve}")
        logger.exception("Configuration error:")
    except NotImplementedError as nie:
        print(f"\nNOT IMPLEMENTED ERROR: {nie}")
        logger.exception("Feature not implemented:")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logger.exception("Error details:")

    print("\n--- WuBu Nesting Full Example Finished ---")