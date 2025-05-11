
"""
WuBuNestDiffusion_v0.04_GAAD_Phi_Infused.py
Diffusion Model with Golden Aspect Adaptive Decomposition (GAAD)
and Phi-Influenced WuBu Spatio-Temporal Nesting.
"""

# =====================================================================
# Python Imports and Setup (includes torchvision.ops.roi_align)
# =====================================================================
import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
# Custom np.load ... (same as before)
orig_np_load = np.load
def custom_np_load(*args, **kwargs):
    allow_pickle_set = 'allow_pickle' in kwargs
    allow_pickle = kwargs.pop('allow_pickle', True)
    mmap_mode_set = 'mmap_mode' in kwargs
    mmap_mode = kwargs.pop('mmap_mode', None)
    if mmap_mode is not None:
        kwargs['mode'] = mmap_mode
        return np.lib.format.open_memmap(*args, allow_pickle=allow_pickle, **kwargs) if allow_pickle_set else np.lib.format.open_memmap(*args, **kwargs)
    return orig_np_load(*args, allow_pickle=allow_pickle, **kwargs)
np.load = custom_np_load

import math, random, argparse, logging, time, contextlib, os, platform, gc, functools
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Iterable
from collections import deque, defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
try:
    import torchvision.io as video_io
    VIDEO_IO_AVAILABLE = True
except ImportError:
    video_io = None
    VIDEO_IO_AVAILABLE = False
    print("Warn: torchvision.io unavailable.")
from torchvision.ops import roi_align
from torchvision.utils import save_image # For demo saving

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logger = logging.getLogger("WuBuGAADPhiDiffusionV04")
# Basic config will be overridden in main if DDP is active or for rank-specific logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')


# Constants and Default Configs
EPS = 1e-5 # This is where SqrtBackward0 NaN comes from if set too high when gradient inf/NaN
PHI = (1 + math.sqrt(5)) / 2 # Modifiable PHI

DEFAULT_CONFIG_WUBU = { # Base template, will be φ-influenced
    "num_levels": 1, "hyperbolic_dims": [32], "initial_curvatures": [1.0],
    "dropout": 0.1, "relative_vector_aggregation": "mean", "aggregation_method": "concat_tangent",
    "use_level_descriptors": True, "use_level_spread": True, "level_descriptor_init_scale": 1e-5,
    "curvature_min_value": EPS, "scale_min_value": EPS, "spread_min_value": EPS,
    "learnable_curvature": True, "initial_scales": [1.0], "learnable_scales": True,
    "learnable_spread": True, "initial_spread_values": [0.1],
    "boundary_points_per_level": [4], "tangent_input_combination_dims": [32],
    "use_tangent_flow": False, "tangent_flow_hidden_dim_ratio": 0.5, # Keep flow simpler initially
    "tangent_flow_type": "mlp", "tangent_flow_scale": 1.0,
    "transform_types": [], "transform_hidden_dims": [], # For single level
    "use_rotation_in_transform": False, # Rotation is handled differently or with phi-bias
    "phi_influence_curvature": False, # New flag
    "phi_influence_rotation_init": False, # New flag
}
DEFAULT_CONFIG_QLEARN_DIFFUSION = { "learning_rate": 0.01, "discount": 0.95, "epsilon": 0.2, "epsilon_decay": 0.9998, "min_epsilon": 0.01, "lr_scale_options": [0.9,0.95,1.,1.05,1.1], "momentum_scale_options": [0.95,0.98,1.,1.01,1.02], "max_q_table_size": 10000}

# =====================================================================
# Geometric, Optimizer, WuBu Core Components
# =====================================================================
class HyperbolicUtils:
    @staticmethod
    def poincare_clip(x: torch.Tensor, c_scalar: float, radius: float = 1.0, eps: float = EPS) -> torch.Tensor:
        if c_scalar <= 0:
            return x
        sqrt_c = math.sqrt(max(c_scalar, eps))
        effective_radius_factor = min(radius, 1.0 - eps)
        max_norm = effective_radius_factor / sqrt_c
        original_dtype = x.dtype
        x_float = x.float()
        x_norm_sq = torch.sum(x_float.pow(2), dim=-1, keepdim=True)
        norm = torch.sqrt(torch.clamp(x_norm_sq, min=0.0) + eps)
        cond = norm > max_norm
        scale_factor = torch.where(cond, max_norm / (norm + eps), torch.ones_like(norm))
        clipped_x = x * scale_factor.to(original_dtype)
        if not torch.isfinite(clipped_x).all():
            return torch.nan_to_num(clipped_x, nan=0.0)
        return clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        if c_scalar <= 0:
            return v
        original_dtype = v.dtype
        v_float = v.float()
        v_norm_sq = torch.sum(v_float.pow(2), dim=-1, keepdim=True)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=0.0) + eps)
        sqrt_c = math.sqrt(max(c_scalar, eps))
        scaled_hyperbolic_radius_arg = scale_scalar * sqrt_c * v_norm
        tanh_input = torch.clamp(scaled_hyperbolic_radius_arg, min=-30.0, max=30.0)
        tanh_term = torch.tanh(tanh_input)
        denominator_lambda = (sqrt_c * v_norm + eps)
        lambda_v = torch.where(v_norm > eps, tanh_term / denominator_lambda, torch.full_like(v_norm, scale_scalar))
        mapped_v = lambda_v.to(original_dtype) * v
        return HyperbolicUtils.poincare_clip(mapped_v, c_scalar, eps=eps)

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        if c_scalar <= 0:
            return y
        original_dtype = y.dtype
        y_clipped = HyperbolicUtils.poincare_clip(y, c_scalar, eps=eps)
        y_float = y_clipped.float()
        y_norm_sq = torch.sum(y_float.pow(2), dim=-1, keepdim=True)
        y_norm = torch.sqrt(torch.clamp(y_norm_sq, min=0.0) + eps)
        sqrt_c = math.sqrt(max(c_scalar, eps))
        arctanh_input_raw = sqrt_c * y_norm
        arctanh_input = torch.clamp(arctanh_input_raw, min=-1.0 + eps * 10, max=1.0 - eps * 10)
        atanh_term = torch.atanh(arctanh_input)
        denominator_lambda = (scale_scalar * sqrt_c * y_norm + eps)
        lambda_y = torch.where(y_norm > eps, atanh_term / denominator_lambda, torch.full_like(y_norm, 1.0 / max(scale_scalar, eps)))
        mapped_y = lambda_y.to(original_dtype) * y_clipped
        if not torch.isfinite(mapped_y).all():
            return torch.nan_to_num(mapped_y, nan=0.0)
        return mapped_y

    @staticmethod
    def exponential_map(v: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_exponential_map(v, c_scalar, scale_scalar=1.0, eps=eps)

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_logarithmic_map(y, c_scalar, scale_scalar=1.0, eps=eps)

class Manifold:
    def __init__(self, c_scalar=0.0):
        self.c = float(c_scalar)
    def proju(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def logmap0(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        raise NotImplementedError
    def expmap(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        return self.proju(p + dp) if self.c > 0 else p + dp
    @property
    def name(self) -> str:
        return self.__class__.__name__

class PoincareBall(Manifold):
    def __init__(self, c_scalar: float = 1.0):
        super().__init__(c_scalar) # This sets self.c
        if self.c <= 0:
            self.k = 0.
            self.sqrt_c = 0. # Explicitly define it
            self.radius = float('inf')
        else:
            self.k = -self.c
            self.sqrt_c = math.sqrt(self.c) # Calculate and assign sqrt_c first
            self.radius = 1. / self.sqrt_c  # Then use it
        
        self.max_norm = self.radius * (1. - EPS * 10) if self.c > 0 and self.radius != float('inf') else float('inf') # Added check for inf radius
        self._name = f'PoincareBall(c={self.c:.3g})'

    @property
    def name(self) -> str:
        return self._name

    def proju(self, x: torch.Tensor) -> torch.Tensor:
        return HyperbolicUtils.poincare_clip(x, self.c, radius=1., eps=EPS * 10)

    def expmap0(self, dp: torch.Tensor) -> torch.Tensor:
        return HyperbolicUtils.exponential_map(dp, self.c, eps=EPS)

    def logmap0(self, p: torch.Tensor) -> torch.Tensor:
        return HyperbolicUtils.logarithmic_map(p, self.c, eps=EPS)

    def expmap0_scaled(self, dp: torch.Tensor, scale_scalar: float) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_exponential_map(dp, self.c, scale_scalar=scale_scalar, eps=EPS)

    def logmap0_scaled(self, p: torch.Tensor, scale_scalar: float) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_logarithmic_map(p, self.c, scale_scalar=scale_scalar, eps=EPS)

    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        if self.c <= 0:
            return dp
        p_proj = self.proju(p)
        lambda_p_sq_val = (1. - self.c * torch.sum(p_proj.pow(2), dim=-1, keepdim=True).clamp_max(1. / (self.c + EPS) - EPS * 100)) # Added EPS to self.c in clamp_max
        factor = (lambda_p_sq_val / 2.).pow(2)
        return torch.clamp(factor, min=EPS) * dp

    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad():
            w.data.uniform_(-irange, irange)
            w.data = self.expmap0(w.data)
            w.data = self.proju(w.data)

def init_weights_general(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm): # Handle LayerNorm specifically
        # ---- Optional: Keep debug prints for LayerNorm if you want ----
        # print(f"DEBUG: Checking module m: {type(m)}")
        # print(f"DEBUG: Is m instance of nn.LayerNorm? {isinstance(m, nn.LayerNorm)}")
        # print(f"DEBUG: Attributes of m (LayerNorm): {dir(m)}")
        # ---- End Optional Debug ----
        if m.elementwise_affine: # Correct for LayerNorm
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm): # Handle GroupNorm specifically
        # ---- Optional: Keep debug prints for GroupNorm if you want ----
        # print(f"DEBUG: Checking module m: {type(m)}")
        # print(f"DEBUG: Is m instance of nn.GroupNorm? {isinstance(m, nn.GroupNorm)}")
        # print(f"DEBUG: Attributes of m (GroupNorm): {dir(m)}")
        # ---- End Optional Debug ----
        if m.affine:  # Correct for GroupNorm (uses 'affine' not 'elementwise_affine')
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor:
    return F.softplus(param_unconstrained) + min_val

class BoundaryManifoldHyperbolic(nn.Module):
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold_c: float):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim
        self.current_manifold_c = initial_manifold_c # Store initial C
        if num_points > 0 and point_dim > 0:
            self.hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim))
            PoincareBall(initial_manifold_c).init_weights(self.hyperbolic_points_params, irange=1e-3)
            self.hyperbolic_points_params.manifold = PoincareBall(initial_manifold_c) # type: ignore
        else:
            self.register_parameter('hyperbolic_points_params', None)

    def set_current_manifold_c(self, c_scalar: float):
        self.current_manifold_c = c_scalar
        if self.hyperbolic_points_params is not None:
            self.hyperbolic_points_params.manifold = PoincareBall(c_scalar) # type: ignore

    def get_points(self) -> Optional[torch.Tensor]:
        if self.hyperbolic_points_params is not None:
            return PoincareBall(self.current_manifold_c).proju(self.hyperbolic_points_params)
        return None

def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis, p=2, dim=-1)
    angle_half = angle_rad / 2.0
    q_w = torch.cos(angle_half)
    q_xyz = axis * torch.sin(angle_half)
    return torch.cat([q_w, q_xyz], dim=-1)

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_apply_to_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_quat = F.pad(v, (1, 0), value=0)
    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    rotated_v_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return rotated_v_quat[..., 1:]

class HyperbolicInterLevelTransform(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, initial_c_in: float, initial_c_out: float, transform_type: str,
                 hidden_dim: Optional[int] = None, dropout: float = 0.1, use_rotation: bool = False,
                 phi_influence_rotation_init: bool = False, level_idx_for_phi: int = 0):
        super().__init__()
        self.in_dim, self.out_dim, self.transform_type = in_dim, out_dim, transform_type.lower()
        self.use_rotation = use_rotation
        self.rotation_module = None
        self.phi_influence_rotation_init = phi_influence_rotation_init

        if self.use_rotation and self.in_dim > 0:
            if self.in_dim == 4 and self.phi_influence_rotation_init:
                self.rot_axis_param = nn.Parameter(torch.randn(3))
                self.rot_angle_unconstrained = nn.Parameter(torch.tensor(0.0))
                self.phi_angle_scale = PHI**(level_idx_for_phi % 5 - 2) * (math.pi / 4)
                logger.info(f"InterLevelTransform L{level_idx_for_phi} (4D): Using learnable Quaternion phi-biased rotation. Angle scale: {self.phi_angle_scale:.3f}")
            elif self.in_dim == 2 and self.phi_influence_rotation_init:
                self.rot_angle_unconstrained_2d = nn.Parameter(torch.tensor(0.0))
                self.phi_angle_scale_2d = PHI**(level_idx_for_phi % 3) * (math.pi / 3)
                logger.info(f"InterLevelTransform L{level_idx_for_phi} (2D): Using learnable SO(2) phi-biased rotation. Angle scale: {self.phi_angle_scale_2d:.3f}")
            else:
                self.rotation_module = nn.Linear(self.in_dim, self.in_dim, bias=False)
                if self.in_dim > 0:
                    nn.init.eye_(self.rotation_module.weight)

        mlp_hidden_dim = hidden_dim if hidden_dim and hidden_dim > 0 else max(16, (in_dim + out_dim) // 2)
        if self.transform_type == 'mlp' and all(d > 0 for d in [in_dim, out_dim, mlp_hidden_dim]):
            self.non_rotational_map = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, out_dim)
            )
        elif self.transform_type == 'linear' and in_dim > 0 and out_dim > 0:
            self.non_rotational_map = nn.Linear(in_dim, out_dim)
        else:
            self.non_rotational_map = nn.Identity()
        self.apply(init_weights_general)

    def _apply_rotation(self, x_tan: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_tan is None or not self.use_rotation:
            return x_tan
        B = x_tan.shape[0]

        if self.in_dim == 4 and self.phi_influence_rotation_init and hasattr(self, 'rot_axis_param'):
            angle = F.softplus(self.rot_angle_unconstrained) * self.phi_angle_scale
            current_axis = self.rot_axis_param.to(x_tan.device).unsqueeze(0).expand(B, -1)
            angle_b = angle.unsqueeze(0).expand(B, 1)
            q_rot = quaternion_from_axis_angle(angle_b, current_axis)

            # Fallback to linear rotation module for general 4D tangent vectors
            # as quaternion_apply_to_vector is for 3D vectors.
            if self.rotation_module:
                return self.rotation_module(x_tan)
            return x_tan # No rotation if module not defined (should be if use_rotation and not special case)

        elif self.in_dim == 2 and self.phi_influence_rotation_init and hasattr(self, 'rot_angle_unconstrained_2d'):
            angle_2d = F.softplus(self.rot_angle_unconstrained_2d) * self.phi_angle_scale_2d
            cos_a = torch.cos(angle_2d)
            sin_a = torch.sin(angle_2d)
            x_comp = x_tan[..., 0]
            y_comp = x_tan[..., 1]
            x_rot = x_comp * cos_a - y_comp * sin_a
            y_rot = x_comp * sin_a + y_comp * cos_a
            return torch.stack([x_rot, y_rot], dim=-1)

        if self.rotation_module:
            return self.rotation_module(x_tan)
        return x_tan

    def forward(self, point_in: torch.Tensor, boundaries_in: Optional[torch.Tensor], descriptor_in: Optional[torch.Tensor], current_c_in: float, current_c_out: float, current_s_in: Optional[float]=None, current_s_out: Optional[float]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        m_in, m_out = PoincareBall(current_c_in), PoincareBall(current_c_out)
        tan_main = m_in.logmap0(point_in)
        tan_bound = m_in.logmap0(boundaries_in) if boundaries_in is not None else None
        tan_desc = m_in.logmap0(descriptor_in) if descriptor_in is not None else None

        tan_main_rot = self._apply_rotation(tan_main)
        tan_bound_rot = self._apply_rotation(tan_bound)
        tan_desc_rot = self._apply_rotation(tan_desc)

        tan_main_out = self.non_rotational_map(tan_main_rot)
        tan_bound_out = self.non_rotational_map(tan_bound_rot) if tan_bound_rot is not None else None
        tan_desc_out = self.non_rotational_map(tan_desc_rot) if tan_desc_rot is not None else None

        return (m_out.expmap0(tan_main_out),
                m_out.expmap0(tan_bound_out) if tan_bound_out is not None else None,
                m_out.expmap0(tan_desc_out) if tan_desc_out is not None else None)

class HyperbolicWuBuNestingLevel(nn.Module):
    def __init__(self, level_idx: int, dim: int, config: Dict, initial_curvature_val_base: float):
        super().__init__()
        self.level_idx, self.dim, self.config = level_idx, dim, config
        self.phi_influence_curvature = config.get("phi_influence_curvature", False)
        if self.phi_influence_curvature:
            phi_scale_c = PHI**(level_idx % 4 - 1.5)
            self.initial_curvature_val = initial_curvature_val_base * phi_scale_c
            logger.info(f"WuBuLevel {level_idx}: Phi-influenced curvature. Base: {initial_curvature_val_base:.2f}, Scale: {phi_scale_c:.2f}, ActualInitial: {self.initial_curvature_val:.2f}")
        else:
            self.initial_curvature_val = initial_curvature_val_base

        self.use_ld = config.get("use_level_descriptors", True)
        self.use_spread = config.get("use_level_spread", True)
        self.dropout_rate = config.get("dropout", 0.1)
        self.ld_init_scale = config.get("level_descriptor_init_scale", 1e-5)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean")
        self.min_curvature = max(EPS, config.get("curvature_min_value", EPS))
        self.min_scale = max(EPS, config.get("scale_min_value", EPS))
        self.min_spread = max(EPS, config.get("spread_min_value", EPS))

        def _init_unconstrained_param(target_val, min_val):
            val_for_softplus = max(float(target_val), min_val + EPS) - min_val
            return torch.tensor(math.log(val_for_softplus + EPS) if val_for_softplus < 1e-6 else math.log(math.expm1(val_for_softplus)), dtype=torch.float)

        param_init_args = {
            'learn_c': ("learnable_curvature", self.initial_curvature_val, self.min_curvature, 'log_curvature_unconstrained'),
            'learn_s': ("learnable_scales", "initial_scales", self.min_scale, 'log_scale_unconstrained'),
            'learn_spread': ("learnable_spread", "initial_spread_values", self.min_spread, 'log_spread_unconstrained')
        }
        for key, (learn_flag_name, init_val_name_or_direct, min_val_local, param_name) in param_init_args.items():
            if key == 'learn_spread' and not self.use_spread:
                self.register_parameter(param_name, None)
                continue
            learn_flag = config.get(learn_flag_name, True)
            if isinstance(init_val_name_or_direct, str):
                default_list = [1.0 if key == 'learn_s' else 0.1]
                init_list = config.get(init_val_name_or_direct, default_list)
                if level_idx < len(init_list):
                    init_val = init_list[level_idx]
                elif init_list:
                    init_val = init_list[-1]
                else: # Should not happen if default_list is used
                    init_val = default_list[0]
            else:
                init_val = init_val_name_or_direct

            unconstrained_val = _init_unconstrained_param(init_val, min_val_local)
            if learn_flag:
                setattr(self, param_name, nn.Parameter(unconstrained_val))
            else:
                self.register_buffer(param_name, unconstrained_val)

        if self.use_ld and self.dim > 0:
            self.level_descriptor_param = nn.Parameter(torch.Tensor(dim))
            PoincareBall(c_scalar=self.initial_curvature_val).init_weights(self.level_descriptor_param, irange=self.ld_init_scale)
            self.level_descriptor_param.manifold = PoincareBall(c_scalar=self.initial_curvature_val) # type: ignore
        else:
            self.register_parameter('level_descriptor_param', None)

        num_bounds_list = config.get("boundary_points_per_level", [8])
        num_boundaries_val = num_bounds_list[level_idx] if level_idx < len(num_bounds_list) else (num_bounds_list[-1] if num_bounds_list else 8)
        self.boundary_manifold_module = BoundaryManifoldHyperbolic(level_idx, num_boundaries_val, dim, initial_manifold_c=self.initial_curvature_val) if self.dim > 0 else None

        comb_in_dim = self.dim
        if self.relative_vector_aggregation not in ['none', None]:
            comb_in_dim += self.dim
        if self.use_ld:
            comb_in_dim += self.dim
        if self.use_spread:
            comb_in_dim += 1

        comb_h_dims_cfg = config.get("tangent_input_combination_dims", [max(16, comb_in_dim // 2)])
        # Ensure comb_h_dims_cfg is a list for iteration
        comb_h_dims = comb_h_dims_cfg if isinstance(comb_h_dims_cfg, list) else [comb_h_dims_cfg]

        layers = []
        in_d = comb_in_dim
        if self.dim > 0 and comb_in_dim > 0:
            for h_d in comb_h_dims:
                if in_d > 0 and h_d > 0:
                    layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(self.dropout_rate)])
                    in_d = h_d
            if in_d > 0:
                layers.append(nn.Linear(in_d, self.dim))
        self.tangent_combiner = nn.Sequential(*layers) if layers else nn.Identity()

        self.use_flow = config.get("use_tangent_flow", True)
        self.tangent_flow_module = None
        self.flow_scale_val = 0.0
        if self.use_flow and self.dim > 0:
            flow_h_dim = max(16, int(dim * config.get("tangent_flow_hidden_dim_ratio", 0.5)))
            flow_type = config.get("tangent_flow_type", "mlp").lower()
            if flow_type == 'mlp' and flow_h_dim > 0:
                self.tangent_flow_module = nn.Sequential(nn.Linear(dim, flow_h_dim), nn.GELU(), nn.Dropout(self.dropout_rate), nn.Linear(flow_h_dim, dim))
            elif flow_type == 'linear':
                self.tangent_flow_module = nn.Linear(dim, dim)

            if self.tangent_flow_module is not None:
                self.flow_scale_val = config.get("tangent_flow_scale", 1.0)
                self.tangent_flow_module.apply(init_weights_general)
        self.tangent_combiner.apply(init_weights_general)

    def get_current_curvature_scalar(self) -> float:
        return get_constrained_param_val(self.log_curvature_unconstrained, self.min_curvature).item()

    def get_current_scale_scalar(self) -> float:
        return get_constrained_param_val(self.log_scale_unconstrained, self.min_scale).item()

    def get_current_spread_scalar_tensor(self) -> torch.Tensor:
        if self.use_spread and hasattr(self, 'log_spread_unconstrained') and self.log_spread_unconstrained is not None:
            return get_constrained_param_val(self.log_spread_unconstrained, self.min_spread)
        # Ensure log_curvature_unconstrained exists before accessing its device/dtype
        ref_device = self.log_curvature_unconstrained.device if hasattr(self, 'log_curvature_unconstrained') else torch.device('cpu')
        ref_dtype = self.log_curvature_unconstrained.dtype if hasattr(self, 'log_curvature_unconstrained') else torch.float
        return torch.tensor(self.min_spread, device=ref_device, dtype=ref_dtype)


    def forward(self, point_in_hyperbolic: torch.Tensor, relative_vectors_tangent_in: Optional[torch.Tensor], descriptor_point_in_hyperbolic: Optional[torch.Tensor], sigma_in_scalar_tensor: Optional[torch.Tensor] ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        B, S, D_in = point_in_hyperbolic.shape
        if self.dim == 0:
            dummy_out_shape = (B, S, 0)
            dummy_dtype_dev = {'device':point_in_hyperbolic.device, 'dtype':point_in_hyperbolic.dtype}
            current_spread_tensor = self.get_current_spread_scalar_tensor().to(point_in_hyperbolic.dtype)
            return (torch.zeros(dummy_out_shape, **dummy_dtype_dev),
                    torch.zeros(dummy_out_shape, **dummy_dtype_dev),
                    None, None, current_spread_tensor)

        dev = point_in_hyperbolic.device
        ref_param_for_dtype = next(iter(self.parameters()), None)
        dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else point_in_hyperbolic.dtype

        current_c_val = self.get_current_curvature_scalar()
        current_s_val = self.get_current_scale_scalar()
        current_sigma_out_tensor = self.get_current_spread_scalar_tensor()
        current_manifold_obj = PoincareBall(c_scalar=current_c_val)

        if self.level_descriptor_param is not None and hasattr(self.level_descriptor_param, 'manifold'): # Check if manifold attr exists
            self.level_descriptor_param.manifold = PoincareBall(c_scalar=current_c_val) # type: ignore
        if self.boundary_manifold_module is not None:
            self.boundary_manifold_module.set_current_manifold_c(current_c_val)

        point_in_proj = current_manifold_obj.proju(point_in_hyperbolic.to(dtype_to_use))
        tan_main_component = current_manifold_obj.logmap0(point_in_proj)

        tan_rel_component = torch.zeros_like(tan_main_component)
        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None]:
            tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)

        ld_point_self_hyperbolic = None
        if self.use_ld and self.level_descriptor_param is not None:
            ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use)) # B, S, D or D
            # ld_point_self_expanded = ld_point_self_hyperbolic.unsqueeze(0).unsqueeze(0).expand(B,S,-1) # if D
            # tan_ld_self_component = current_manifold_obj.logmap0(ld_point_self_expanded) # This line was unused

        tan_desc_prev_level_component = torch.zeros_like(tan_main_component) # Placeholder
        if descriptor_point_in_hyperbolic is not None and self.use_ld : # only use if use_ld is true
            desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use))
            tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)


        inputs_for_combiner = [tan_main_component]
        if self.relative_vector_aggregation not in ['none', None]:
            inputs_for_combiner.append(tan_rel_component)
        if self.use_ld: # This now correctly refers to descriptor from PREVIOUS level or self if no prev
            inputs_for_combiner.append(tan_desc_prev_level_component) # This should be the transformed descriptor

        if self.use_spread and sigma_in_scalar_tensor is not None:
            sigma_in_expanded = sigma_in_scalar_tensor.view(-1,1,1).expand(B,S,1).to(dtype_to_use)
            inputs_for_combiner.append(sigma_in_expanded)

        combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1)
        v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features)

        if self.use_flow and self.tangent_flow_module is not None:
            flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val
            v_combined_tangent_processed = v_combined_tangent_processed + flow_effect

        scaled_output_tangent_for_expmap = v_combined_tangent_processed * current_s_val
        point_this_level_out_hyperbolic = current_manifold_obj.expmap0(scaled_output_tangent_for_expmap)
        tangent_out_for_aggregation = v_combined_tangent_processed.to(dtype_to_use) # Should be dtype_to_use

        boundary_points_this_level_hyperbolic = None
        if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None:
            boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype_to_use)

        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None: # This is the descriptor *of this level*
            if ld_point_self_hyperbolic.dim() == 1: # D
                 descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B, S, -1).to(dtype_to_use) # B, S, D
            elif ld_point_self_hyperbolic.dim() == 2 and ld_point_self_hyperbolic.shape[0] == B and ld_point_self_hyperbolic.shape[1] == S: # B, S, D
                 descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.to(dtype_to_use)
            else: # Fallback, e.g. (1,D) or other shapes
                 descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B,S,-1) if ld_point_self_hyperbolic.dim() < 3 else ld_point_self_hyperbolic
                 descriptor_point_out_for_transform_hyperbolic = descriptor_point_out_for_transform_hyperbolic.to(dtype_to_use)


        return (point_this_level_out_hyperbolic.to(dtype_to_use),
                tangent_out_for_aggregation,
                descriptor_point_out_for_transform_hyperbolic,
                boundary_points_this_level_hyperbolic,
                current_sigma_out_tensor.to(dtype_to_use))


class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super().__init__()
        self.input_dim, self.output_dim, self.config = input_dim, output_dim, config
        self.num_levels = config.get("num_levels", 3)
        assert self.num_levels >= 0
        self.hyperbolic_dims_list = config.get("hyperbolic_dims", [])
        self.initial_curvatures_list = config.get("initial_curvatures", [])
        self.dropout_val = config.get("dropout", 0.1)
        self.relative_vector_aggregation_mode = config.get("relative_vector_aggregation", "mean")
        self.aggregation_method_mode = config.get("aggregation_method", "concat_tangent")
        assert self.aggregation_method_mode == "concat_tangent"
        self.use_rotation_in_transform_flag = config.get("use_rotation_in_transform", False)
        self.phi_influence_rotation_init = config.get("phi_influence_rotation_init", False)

        first_level_dim = self.hyperbolic_dims_list[0] if self.num_levels > 0 and self.hyperbolic_dims_list else 0
        self.input_tangent_projection = nn.Linear(input_dim, first_level_dim) if input_dim > 0 and first_level_dim > 0 else nn.Identity()

        self.levels_modulelist = nn.ModuleList()
        if self.num_levels > 0:
            for i in range(self.num_levels):
                self.levels_modulelist.append(HyperbolicWuBuNestingLevel(i, self.hyperbolic_dims_list[i], self.config, self.initial_curvatures_list[i]))

        self.transforms_modulelist = nn.ModuleList()
        num_transforms = max(0, self.num_levels - 1)
        if num_transforms > 0:
            transform_types_list = config.get("transform_types", ["linear"] * num_transforms)
            transform_hidden_dims_list = config.get("transform_hidden_dims", [None] * num_transforms)
            for i in range(num_transforms):
                # Check if enough dims for next level exist
                if i+1 < len(self.hyperbolic_dims_list) and i+1 < len(self.initial_curvatures_list):
                    self.transforms_modulelist.append(HyperbolicInterLevelTransform(
                        self.hyperbolic_dims_list[i], self.hyperbolic_dims_list[i+1],
                        self.initial_curvatures_list[i], self.initial_curvatures_list[i+1],
                        transform_types_list[i] if i < len(transform_types_list) else "linear",
                        transform_hidden_dims_list[i] if i < len(transform_hidden_dims_list) else None,
                        self.dropout_val, self.use_rotation_in_transform_flag,
                        self.phi_influence_rotation_init, level_idx_for_phi=i
                    ))
                else:
                    logger.warning(f"Skipping transform {i} due to insufficient config for next level dims/curvatures.")


        actual_output_dims_from_levels = [d for d_idx, d in enumerate(self.hyperbolic_dims_list[:self.num_levels]) if d > 0 and d_idx < len(self.levels_modulelist)]
        aggregated_tangent_dim_val = sum(actual_output_dims_from_levels) if actual_output_dims_from_levels else input_dim

        self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, output_dim) if aggregated_tangent_dim_val > 0 and output_dim > 0 else nn.Identity()
        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"FullyHypWuBuModel: {self.num_levels} levels. {param_count:,} params. Rotation: {self.use_rotation_in_transform_flag}, PhiRotInit: {self.phi_influence_rotation_init}. InDim {input_dim}, AggTangentDim {aggregated_tangent_dim_val}, OutDim {output_dim}")

    def forward(self, x_initial_tangent_in: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.num_levels == 0:
            return self.output_tangent_projection(x_initial_tangent_in)
        if x_initial_tangent_in.dim() == 2:
            x_initial_tangent_in = x_initial_tangent_in.unsqueeze(1) # B, S, D

        B, S, _ = x_initial_tangent_in.shape
        dev = x_initial_tangent_in.device
        ref_param_for_dtype = next(iter(self.parameters()), None)
        dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else x_initial_tangent_in.dtype
        x_initial_tangent_in = x_initial_tangent_in.to(dtype_to_use)

        current_tangent_for_level0 = self.input_tangent_projection(x_initial_tangent_in)
        if not self.levels_modulelist: # Should not happen if num_levels > 0
             default_out_dim = self.output_dim if self.output_dim > 0 else (self.input_dim if self.input_dim > 0 else 1)
             return torch.zeros((B, S, default_out_dim), device=dev, dtype=dtype_to_use)

        level0_module = self.levels_modulelist[0]
        c0_val = level0_module.get_current_curvature_scalar()
        m0_obj = PoincareBall(c_scalar=c0_val)

        if self.hyperbolic_dims_list[0] > 0:
            current_point_repr_hyperbolic = m0_obj.expmap0(current_tangent_for_level0)
        else:
            current_point_repr_hyperbolic = torch.empty(B, S, 0, device=dev, dtype=dtype_to_use)

        level_tangent_outputs_for_aggregation = []
        aggregated_relative_vectors_from_prev_transform = None
        descriptor_from_prev_transform_hyperbolic = None # This will carry descriptor from L_i to L_{i+1}
        sigma_from_prev_level_tensor = torch.full((B,), 0.0, device=dev, dtype=dtype_to_use)


        for i in range(self.num_levels):
            level_module = self.levels_modulelist[i]
            (point_out_of_level_hyperbolic,
             tangent_out_of_level_for_aggregation,
             descriptor_generated_by_level_hyperbolic, # Descriptor OF this level L_i
             boundary_points_of_level_hyperbolic,
             sigma_out_of_level_tensor) = level_module(current_point_repr_hyperbolic,
                                                       aggregated_relative_vectors_from_prev_transform,
                                                       descriptor_from_prev_transform_hyperbolic, # Descriptor FROM L_{i-1} (or None for L0)
                                                       sigma_from_prev_level_tensor)

            if self.hyperbolic_dims_list[i] > 0:
                level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)

            if i < self.num_levels - 1:
                if i >= len(self.transforms_modulelist): # Check if transform exists
                    logger.warning(f"Missing transform for level {i} to {i+1}. Stopping propagation.")
                    break
                transform_module = self.transforms_modulelist[i]
                next_level_module = self.levels_modulelist[i+1]

                c_in_for_transform = level_module.get_current_curvature_scalar()
                c_out_for_transform = next_level_module.get_current_curvature_scalar()

                # The descriptor passed to transform is the one *generated by* the current level L_i
                (point_transformed_to_next_level_hyperbolic,
                 boundaries_transformed_to_next_level_hyperbolic,
                 descriptor_transformed_to_next_level_hyperbolic # This is L_i's descriptor, transformed for L_{i+1}
                ) = transform_module(point_out_of_level_hyperbolic,
                                     boundary_points_of_level_hyperbolic,
                                     descriptor_generated_by_level_hyperbolic, # Pass L_i's own descriptor
                                     c_in_for_transform, c_out_for_transform)

                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic
                descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic # For next iteration
                sigma_from_prev_level_tensor = sigma_out_of_level_tensor.expand(B) if sigma_out_of_level_tensor.numel() == 1 else sigma_out_of_level_tensor


                aggregated_relative_vectors_from_prev_transform = None # Reset for current transform output
                if boundaries_transformed_to_next_level_hyperbolic is not None and \
                   self.relative_vector_aggregation_mode not in ['none', None] and \
                   self.hyperbolic_dims_list[i+1] > 0:

                    manifold_next_level_obj = PoincareBall(c_scalar=c_out_for_transform)
                    tan_main_next_level = manifold_next_level_obj.logmap0(current_point_repr_hyperbolic) # B,S,D_next
                    tan_bounds_next_level = manifold_next_level_obj.logmap0(boundaries_transformed_to_next_level_hyperbolic) # NumBounds_i, D_next or B,S,NumBounds_i,D_next

                    # Ensure tan_bounds_next_level is B,S,N,D
                    if tan_bounds_next_level.dim() == 2: # NumBounds, Dim -> B,S,NumBounds,Dim
                        tan_bounds_next_level = tan_bounds_next_level.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
                    elif tan_bounds_next_level.dim() == 3 and tan_bounds_next_level.shape[0] != B : # Should be B, NumBounds, Dim -> B,S,NumBounds,Dim
                        tan_bounds_next_level = tan_bounds_next_level.unsqueeze(1).expand(-1, S, -1, -1)


                    relative_tangent_vectors = tan_main_next_level.unsqueeze(2) - tan_bounds_next_level # B,S,1,D - B,S,N,D -> B,S,N,D
                    agg_mode = self.relative_vector_aggregation_mode
                    if agg_mode == "mean":
                        agg_rel_vec = torch.mean(relative_tangent_vectors, dim=2)
                    elif agg_mode == "sum":
                        agg_rel_vec = torch.sum(relative_tangent_vectors, dim=2)
                    else:
                        agg_rel_vec = None

                    if agg_rel_vec is not None and not torch.isfinite(agg_rel_vec).all():
                        agg_rel_vec = torch.zeros_like(tan_main_next_level)
                    aggregated_relative_vectors_from_prev_transform = agg_rel_vec


        if not level_tangent_outputs_for_aggregation:
            default_out_dim = self.output_dim if self.output_dim > 0 else (self.input_dim if self.input_dim > 0 else 1)
            return torch.zeros((B, S, default_out_dim), device=dev, dtype=dtype_to_use)

        compatible_tangent_outputs = []
        for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation):
             if t_val is not None and t_idx < len(self.hyperbolic_dims_list) and self.hyperbolic_dims_list[t_idx] > 0 and torch.isfinite(t_val).all():
                 compatible_tangent_outputs.append(t_val.to(dtype_to_use))


        if not compatible_tangent_outputs:
            default_out_dim = self.output_dim if self.output_dim > 0 else (self.input_dim if self.input_dim > 0 else 1)
            return torch.zeros((B, S, default_out_dim), device=dev, dtype=dtype_to_use)

        aggregated_tangent_final = torch.cat(compatible_tangent_outputs, dim=-1)
        final_output = self.output_tangent_projection(aggregated_tangent_final)

        if padding_mask is not None:
            final_output = final_output.masked_fill(padding_mask.unsqueeze(-1).bool(), 0.0)
        if not torch.isfinite(final_output).all():
            final_output = torch.nan_to_num(final_output, nan=0.0)
        return final_output

class GradientStats:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total_params_updated = 0
        self.total_finite_grads_processed = 0
        self.total_non_finite_grads_encountered = 0
        self.params_skipped_due_non_finite_grad = 0
        self.max_grad_norm_observed = 0.
        self.step_summary = {}
    def record_param_grad(self, grad_is_finite: bool, original_norm_if_finite: Optional[float] = None):
        if grad_is_finite:
            self.total_finite_grads_processed += 1
            if original_norm_if_finite is not None:
                self.max_grad_norm_observed = max(self.max_grad_norm_observed, original_norm_if_finite)
        else:
            self.total_non_finite_grads_encountered += 1
            self.params_skipped_due_non_finite_grad += 1
    def finalize_step_stats(self, num_params_in_optimizer_step: int):
        self.total_params_updated = num_params_in_optimizer_step - self.params_skipped_due_non_finite_grad
        self.step_summary = {
            "params_in_step": num_params_in_optimizer_step,
            "params_updated": self.total_params_updated,
            "params_skipped_non_finite_grad": self.params_skipped_due_non_finite_grad,
            "initial_finite_grads": self.total_finite_grads_processed,
            "initial_non_finite_grads": self.total_non_finite_grads_encountered,
            "max_finite_grad_norm_observed": self.max_grad_norm_observed
        }
    def get_step_summary_for_logging(self) -> dict:
        return self.step_summary.copy()

class HAKMEMQController:
    def __init__(self, learning_rate:float=0.01, discount:float=0.95, epsilon:float=0.2, epsilon_decay:float=0.9998, min_epsilon:float=0.01, lr_scale_options:Optional[List[float]]=None, momentum_scale_options:Optional[List[float]]=None, max_q_table_size:int=10000):
        self.q_table: Dict[Tuple, Dict[str, np.ndarray]] = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.prev_loss: Optional[float] = None
        self.prev_state: Optional[Tuple] = None
        self.prev_action: Optional[Dict[str, float]] = None
        self.action_ranges = {
            'lr_scale': np.array(lr_scale_options if lr_scale_options else [0.9, 0.95, 1., 1.05, 1.1]),
            'momentum_scale': np.array(momentum_scale_options if momentum_scale_options else [0.95, 0.98, 1., 1.01, 1.02])
        }
        self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}
        self.loss_window = deque(maxlen=20)
        self.grad_norm_window = deque(maxlen=20)
        self.lr_window = deque(maxlen=10)
        self.momentum_window = deque(maxlen=10)
        self.performance_window = deque(maxlen=50)
        self.stable_steps = 0
        self.oscillation_counter = 0
        self.prev_actions_log = deque(maxlen=10)
        self.max_q_table_size = max(100, max_q_table_size)
        self.q_table_access_count: Dict[Tuple, int] = defaultdict(int)
        self.q_table_creation_time: Dict[Tuple, float] = {}
        self.flow_coefficient = 0.05
        self.oscillation_penalty = 0.15
        self.stability_reward_bonus = 0.05

    def get_state(self, lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float]) -> Optional[Tuple]:
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm):
            return None
        self.loss_window.append(loss)
        self.grad_norm_window.append(grad_norm)
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)
        loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0
        if len(self.loss_window) < 5 or len(self.grad_norm_window) < 5:
            return None
        try:
            loss_arr = np.array(list(self.loss_window)[-10:])
            slope_loss = np.polyfit(np.arange(len(loss_arr)), loss_arr, 1)[0] if len(loss_arr) >= 3 and len(np.unique(loss_arr)) > 1 else 0.0
            loss_trend_bin = np.digitize(slope_loss / (abs(np.median(loss_arr)) + EPS), bins=[-0.05, -0.005, 0.005, 0.05]).item()
            grad_norm_level_bin = np.digitize(np.median(list(self.grad_norm_window)), bins=[0.1, 0.5, 1.5, 5.0]).item()
            lr_level_bin = np.digitize(lr / 1e-4, bins=[0.5, 2.0, 10.0, 50.0]).item()
            momentum_level_bin = np.digitize(momentum, bins=[0.85, 0.92, 0.97]).item()
            if len(self.performance_window) >= 5:
                recent_rewards = np.sign([r for r in list(self.performance_window)[-5:] if r != 0])
                self.oscillation_counter = min(self.oscillation_counter + 1, 5) if len(recent_rewards) >= 2 and np.sum(np.abs(np.diff(recent_rewards))) / 2.0 >= 2 else max(0, self.oscillation_counter - 1)
            oscillation_bin = 1 if self.oscillation_counter >= 3 else 0
        except Exception:
            return None
        state_tuple = (loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin)
        self.q_table_access_count[state_tuple] += 1
        return state_tuple

    def compute_reward(self, current_loss: Optional[float], prev_loss: Optional[float], grad_norm: Optional[float]) -> float:
        if current_loss is None or prev_loss is None or grad_norm is None or \
           not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm):
            return 0.
        median_loss = np.median(list(self.loss_window)[:-1]) if len(self.loss_window) > 1 else prev_loss
        base_reward = np.tanh((prev_loss - current_loss) / (abs(median_loss) + EPS) * 10.)
        grad_penalty = -0.1 * min(1., max(0., (grad_norm - 5.) / 10.)) if grad_norm > 5. else 0.
        osc_penalty = -self.oscillation_penalty if self.oscillation_counter >= 3 else 0.
        current_perf_reward = base_reward + grad_penalty + osc_penalty
        self.performance_window.append(current_perf_reward)
        self.stable_steps = self.stable_steps + 1 if current_perf_reward > 0.01 else 0
        stab_bonus = min(0.15, self.stability_reward_bonus * math.log1p(self.stable_steps / 5.)) if current_perf_reward > 0.01 else 0.
        return float(np.clip(base_reward + grad_penalty + osc_penalty + stab_bonus, -1., 1.))

    def choose_action(self, state: Optional[Tuple]) -> Dict[str, float]:
        if state is None:
            return {'lr_scale': 1., 'momentum_scale': 1.}
        if state not in self.q_table:
            self.q_table[state] = {p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}
            self.q_table_creation_time[state] = time.time()
            self.q_table_access_count[state] = 1
            self._manage_q_table_size()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        chosen_actions = {}
        for p_type, q_vals in self.q_table[state].items():
            act_space = self.action_ranges[p_type]
            if random.random() < self.epsilon:
                chosen_idx = random.randrange(len(act_space))
            else:
                finite_q_mask = np.isfinite(q_vals)
                if not np.any(finite_q_mask):
                    chosen_idx = random.randrange(len(act_space))
                else:
                    finite_q_values = q_vals[finite_q_mask]
                    max_q = np.max(finite_q_values)
                    best_indices_original = np.where(finite_q_mask)[0][np.isclose(finite_q_values, max_q)]
                    chosen_idx = np.random.choice(best_indices_original) if len(best_indices_original) > 0 else random.randrange(len(act_space))
            chosen_actions[p_type] = float(act_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy())
        return chosen_actions

    def update(self, state: Optional[Tuple], action: Optional[Dict[str, float]], reward: float, next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None or state not in self.q_table:
            return
        if next_state not in self.q_table:
            self.q_table[next_state] = {p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}
            self.q_table_creation_time[next_state] = time.time()
            self.q_table_access_count[next_state] = 0
            self._manage_q_table_size()

        for p_type, chosen_val in action.items():
            if p_type not in self.q_table[state]:
                continue
            act_idx_arr = np.where(np.isclose(self.action_ranges[p_type], chosen_val))[0]
            act_idx = act_idx_arr[0] if len(act_idx_arr) > 0 else -1
            if act_idx == -1:
                continue
            current_q = self.q_table[state][p_type][act_idx]
            next_q_vals = self.q_table[next_state][p_type]
            finite_next_q = next_q_vals[np.isfinite(next_q_vals)]
            max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0
            max_future_q = 0.0 if not np.isfinite(max_future_q) else max_future_q
            td_target = reward + self.gamma * max_future_q
            td_error = td_target - current_q
            adaptive_alpha = min(0.5, max(0.001, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error) * 0.5))))
            new_q = current_q + adaptive_alpha * td_error
            self.q_table[state][p_type][act_idx] = np.clip(new_q, -1e4, 1e4) if np.isfinite(new_q) else 0.0

    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            can_smart_prune = all([
                self.q_table_access_count,
                self.q_table_creation_time,
                len(self.q_table_access_count) > len(self.q_table) // 2,
                len(self.q_table_creation_time) > len(self.q_table) // 2
            ])
            to_remove = sorted(
                self.q_table.keys(),
                key=lambda s: (self.q_table_access_count.get(s, 0), self.q_table_creation_time.get(s, float('inf')))
            )[:len(self.q_table) - self.max_q_table_size] if can_smart_prune else random.sample(
                list(self.q_table.keys()), len(self.q_table) - self.max_q_table_size
            )
            for s_rm in to_remove:
                self.q_table.pop(s_rm, None)
                self.q_table_access_count.pop(s_rm, None)
                self.q_table_creation_time.pop(s_rm, None)

    def get_info(self) -> Dict:
        q_mem_mb = 0.0
        if self.q_table:
             q_mem_mb = sum(sys.getsizeof(s) + sum(a.nbytes + sys.getsizeof(k) for k,a in v.items()) for s,v in self.q_table.items())/(1024**2)
        avg_perf_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.
        return {
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "q_table_mem_mb_approx": round(q_mem_mb, 2),
            "last_action": self.prev_actions_log[-1] if self.prev_actions_log else None,
            f"avg_reward_last_{self.performance_window.maxlen}": avg_perf_reward,
            "stable_steps": self.stable_steps,
            "oscillation_counter": self.oscillation_counter
        }

class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm_risgd=1.0, q_learning_config:Optional[Dict]=None):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_learning_config) if isinstance(q_learning_config, dict) else None
        logger.info(f"RiSGD: Q-Controller {'en' if self.q_controller else 'dis'}abled.")
        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count = 0
        self.current_loss_for_q: Optional[float] = None
        self.grad_stats = GradientStats()
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state.setdefault(p, {})

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        self.grad_stats.reset()

    def set_current_loss_for_q_controller(self, loss: Optional[float]):
        if loss is not None and np.isfinite(loss):
            self.current_loss_for_q = loss
        else:
            self.current_loss_for_q = None
            if self.q_controller:
                self.q_controller.prev_loss = None


    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action
            for group in self.param_groups:
                group['base_lr'] = group.setdefault('base_lr', group['lr'])
                group['base_momentum'] = group.setdefault('base_momentum', group['momentum'])
                group['lr'] = float(np.clip(group['base_lr'] * q_action.get('lr_scale', 1.0), 1e-8, 1.0))
                group['momentum'] = float(np.clip(group['base_momentum'] * q_action.get('momentum_scale', 1.0), 0.1, 0.999))

        num_params_total = 0
        for group in self.param_groups:
            lr, mom, wd = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                num_params_total += 1
                grad = p.grad.data
                finite_grad = torch.isfinite(grad).all()
                norm_val = grad.norm().item() if finite_grad else None
                self.grad_stats.record_param_grad(finite_grad, norm_val)

                if not finite_grad:
                    self.state[p].pop('momentum_buffer', None)
                    continue

                if norm_val is not None and norm_val > self.max_grad_norm_risgd:
                    grad.mul_(self.max_grad_norm_risgd / (norm_val + EPS))

                state = self.state[p]
                manifold: Optional[Manifold] = getattr(p, 'manifold', None)

                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    p.data = manifold.proju(p.data)
                    try:
                        r_grad = manifold.egrad2rgrad(p.data, grad)
                    except Exception:
                        self.grad_stats.params_skipped_due_non_finite_grad += 1
                        continue
                    if not torch.isfinite(r_grad).all():
                        self.grad_stats.params_skipped_due_non_finite_grad += 1
                        continue

                    update_vec = r_grad
                    if wd != 0:
                        try:
                            log_p = manifold.logmap0(p.data)
                            if torch.isfinite(log_p).all():
                                update_vec = update_vec.add(log_p, alpha=wd)
                        except Exception:
                            pass # Ignore errors in logmap for wd
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(update_vec))
                    buf.mul_(mom).add_(update_vec)

                    if not torch.isfinite(buf).all(): buf.zero_()
                    try:
                        p.data = manifold.proju(manifold.expmap(p.data, buf.mul(-lr)))
                    except Exception:
                        state['momentum_buffer'].zero_() # type: ignore
                    if not torch.isfinite(p.data).all():
                        p.data = manifold.proju(torch.nan_to_num(p.data, nan=0.0))
                        state['momentum_buffer'].zero_() # type: ignore
                else: # Euclidean / no manifold
                    d_p = grad.clone()
                    if wd != 0:
                        d_p.add_(p.data, alpha=wd)
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(p.data))
                    buf.mul_(mom).add_(d_p)
                    if not torch.isfinite(buf).all(): buf.zero_()
                    p.data.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all():
                        p.data = torch.nan_to_num(p.data, nan=0.0)
                        state['momentum_buffer'].zero_() # type: ignore

        self.grad_stats.finalize_step_stats(num_params_total)
        self._step_count += 1
        return loss

    def get_q_controller_info(self) -> Dict:
        return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}

    def get_gradient_stats_summary(self) -> Dict:
        return self.grad_stats.get_step_summary_for_logging()

# =====================================================================
# GAAD Components
# =====================================================================
def golden_subdivide_rect_fixed_n(frame_dims:Tuple[int,int], num_regions_target:int, device='cpu', dtype=torch.float, min_size_px=5) -> torch.Tensor:
    W, H = frame_dims
    all_rects = []
    all_rects.append([0, 0, W, H]) # Full frame
    rect_queue = deque([(0, 0, W, H, 0)]) # x, y, w, h, depth

    while rect_queue and len(all_rects) < num_regions_target * 3: # Generate more to pick from
        x_off, y_off, w_curr, h_curr, depth = rect_queue.popleft()
        if min(w_curr, h_curr) < min_size_px or depth > 6 : # Stop if too small or too deep
            continue

        is_landscape = w_curr > h_curr + EPS # Add EPS for float comparison
        is_portrait = h_curr > w_curr + EPS

        if is_landscape: # Wider than tall
            cut_w = w_curr / PHI
            r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px and h_curr >= min_size_px:
                 all_rects.append([x_off, y_off, x_off + r1_w, y_off + h_curr])
                 rect_queue.append((x_off, y_off, r1_w, h_curr, depth + 1))
            if r2_w >= min_size_px and h_curr >= min_size_px:
                 all_rects.append([x_off + r1_w, y_off, x_off + r1_w + r2_w, y_off + h_curr])
                 rect_queue.append((x_off + r1_w, y_off, r2_w, h_curr, depth + 1))

        elif is_portrait: # Taller than wide
            cut_h = h_curr / PHI
            r1_h, r2_h = cut_h, h_curr - cut_h
            if w_curr >= min_size_px and r1_h >= min_size_px:
                 all_rects.append([x_off, y_off, x_off + w_curr, y_off + r1_h])
                 rect_queue.append((x_off, y_off, w_curr, r1_h, depth + 1))
            if w_curr >= min_size_px and r2_h >= min_size_px:
                 all_rects.append([x_off, y_off + r1_h, x_off + w_curr, y_off + r1_h + r2_h])
                 rect_queue.append((x_off, y_off + r1_h, w_curr, r2_h, depth + 1))
        # else: square, do not subdivide further by this rule.

    unique_valid_rects_tensors = []
    seen_hashes = set()
    for r_coords in all_rects:
        # Ensure x1 < x2 and y1 < y2 and positive area
        if r_coords[0] >= r_coords[2] - EPS or r_coords[1] >= r_coords[3] - EPS:
            continue
        r_tensor = torch.tensor(r_coords, dtype=dtype, device=device)
        # Use a hashable representation for uniqueness check
        r_hashable = tuple(round(c, 3) for c in r_coords) # Round to avoid float precision issues
        if r_hashable not in seen_hashes:
            unique_valid_rects_tensors.append(r_tensor)
            seen_hashes.add(r_hashable)

    # Sort by area (descending)
    unique_valid_rects_tensors.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)

    selected_rects = unique_valid_rects_tensors[:num_regions_target]

    if len(selected_rects) < num_regions_target:
        padding_box = selected_rects[-1] if selected_rects else torch.tensor([0,0,W,H],dtype=dtype,device=device)
        selected_rects.extend([padding_box.clone() for _ in range(num_regions_target - len(selected_rects))])

    return torch.stack(selected_rects)

def phi_spiral_patch_centers_fixed_n(frame_dims:Tuple[int,int], num_centers:int, device='cpu', dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]:
    W, H = frame_dims
    centers_xy = []
    scale_factors = []
    cx, cy = W / 2.0, H / 2.0

    if num_centers <= 0:
        return torch.empty(0,2,device=device,dtype=dtype), torch.empty(0,1,device=device,dtype=dtype)

    # Always include center point if num_centers > 0
    centers_xy.append([cx, cy])
    scale_factors.append(0.25) # Scale factor for patch size relative to min(W,H)

    num_spiral_points_to_generate = num_centers - 1 # one is already center

    if num_spiral_points_to_generate <= 0:
        return torch.tensor(centers_xy, dtype=dtype, device=device), torch.tensor(scale_factors, dtype=dtype, device=device).unsqueeze(-1)

    # Spiral generation parameters
    a = 0.05 * min(W, H)  # Initial distance from center
    b = math.log(PHI) / (math.pi / 2) # Growth rate: r grows by PHI every 90 degrees

    angle_step = PHI * 2 * math.pi / num_spiral_points_to_generate if num_spiral_points_to_generate > 0 else 0
    current_angle = 0.0

    for i in range(num_spiral_points_to_generate):
        r = a * math.exp(b * current_angle)
        # Ensure points are somewhat within image boundaries, cap r
        max_r = max(W,H) * 0.6 # Cap spiral extent
        if r > max_r : r = max_r

        x = cx + r * math.cos(current_angle)
        y = cy + r * math.sin(current_angle)

        # Clamp to be within frame boundaries
        x_clamped = max(0.0, min(x, float(W)))
        y_clamped = max(0.0, min(y, float(H)))

        centers_xy.append([x_clamped, y_clamped])
        # Scale factor can decrease with distance or be constant
        patch_scale = max(0.05, 0.20 * math.exp(-0.5 * r / (min(W,H)*0.1)))
        scale_factors.append(patch_scale)
        current_angle += angle_step


    # Pad if not enough unique points generated (shouldn't happen with this method)
    if len(centers_xy) < num_centers:
        num_to_pad = num_centers - len(centers_xy)
        last_xy = centers_xy[-1] if centers_xy else [cx,cy]
        last_scale = scale_factors[-1] if scale_factors else 0.1
        centers_xy.extend([last_xy] * num_to_pad)
        scale_factors.extend([last_scale] * num_to_pad)

    return torch.tensor(centers_xy[:num_centers], dtype=dtype, device=device), \
           torch.tensor(scale_factors[:num_centers], dtype=dtype, device=device).unsqueeze(-1)


class GAADFrameProcessor(nn.Module):
    def __init__(self, num_total_regions: int, region_roi_output_size: Tuple[int,int],
                 base_cnn_encoder_convs: nn.Module, base_cnn_out_channels: int,
                 gaad_region_feature_dim: int, decomposition_type: str = "hybrid"):
        super().__init__()
        self.num_total_regions=num_total_regions
        self.region_roi_output_size=region_roi_output_size
        self.base_cnn_encoder_convs = base_cnn_encoder_convs
        self.decomposition_type=decomposition_type
        self.base_cnn_out_channels = base_cnn_out_channels
        roi_flat_dim = base_cnn_out_channels * region_roi_output_size[0] * region_roi_output_size[1]
        self.region_projector = nn.Sequential(
            nn.Linear(roi_flat_dim,gaad_region_feature_dim*2),
            nn.GELU(),
            nn.Linear(gaad_region_feature_dim*2,gaad_region_feature_dim)
        )
        self.apply(init_weights_general)
        logger.info(f"GAADFrameProcessor: NumTotalRegions {num_total_regions}, Decomp '{decomposition_type}', RegionFeatDim {gaad_region_feature_dim}")

    def forward(self, frame_pixels: torch.Tensor) -> torch.Tensor:
        B, _, frame_h, frame_w = frame_pixels.shape
        dev = frame_pixels.device
        dtype = frame_pixels.dtype
        feature_maps = self.base_cnn_encoder_convs(frame_pixels) # (B, C_base_feat, H_map, W_map)
        map_h, map_w = feature_maps.shape[2], feature_maps.shape[3]
        scale_h, scale_w = map_h / float(frame_h), map_w / float(frame_w)

        batch_region_features_list = []
        for b_idx in range(B):
            all_bboxes_for_frame_list = [] # Use list to append
            if self.decomposition_type == "hybrid":
                num_subdivide = self.num_total_regions // 2
                num_spiral = self.num_total_regions - num_subdivide
                if num_subdivide > 0:
                    subdiv_bboxes = golden_subdivide_rect_fixed_n((frame_w,frame_h),num_subdivide,dev,dtype)
                    all_bboxes_for_frame_list.append(subdiv_bboxes)
                if num_spiral > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n((frame_w,frame_h),num_spiral,dev,dtype)
                    patch_base_size = min(frame_w, frame_h)
                    spiral_bboxes_current = torch.zeros(num_spiral, 4, device=dev, dtype=dtype)
                    patch_hs = patch_base_size * spiral_scales[:,0] / 2.0
                    patch_ws = patch_hs # Square patches
                    
                    # Convert frame dimensions to tensors of the same dtype and device as spiral_centers for clamp
                    # This ensures compatibility when min/max are tensors.
                    frame_w_tensor = torch.tensor(frame_w, dtype=spiral_centers.dtype, device=spiral_centers.device)
                    frame_h_tensor = torch.tensor(frame_h, dtype=spiral_centers.dtype, device=spiral_centers.device)
                    zero_tensor = torch.tensor(0.0, dtype=spiral_centers.dtype, device=spiral_centers.device) # For min=0.0

                    # For these clamps, input, min, and max can all be Numbers (scalars)
                    # or input is Tensor, min/max are Numbers.
                    spiral_bboxes_current[:,0]=torch.clamp(spiral_centers[:,0]-patch_ws, 
                                                           min=0.0,  # Number
                                                           max=(frame_w_tensor-EPS).item()) # Number (scalar from tensor)
                    spiral_bboxes_current[:,1]=torch.clamp(spiral_centers[:,1]-patch_hs, 
                                                           min=0.0,  # Number
                                                           max=(frame_h_tensor-EPS).item()) # Number

                    # For these clamps, min is a Tensor, so max must also be a Tensor.
                    # The input is also a Tensor.
                    min_x_tensor = spiral_bboxes_current[:,0]+EPS
                    min_y_tensor = spiral_bboxes_current[:,1]+EPS

                    spiral_bboxes_current[:,2]=torch.clamp(spiral_centers[:,0]+patch_ws, 
                                                           min=min_x_tensor,      # Tensor
                                                           max=frame_w_tensor)    # Tensor
                    spiral_bboxes_current[:,3]=torch.clamp(spiral_centers[:,1]+patch_hs, 
                                                           min=min_y_tensor,      # Tensor
                                                           max=frame_h_tensor)    # Tensor
                    all_bboxes_for_frame_list.append(spiral_bboxes_current)

            elif self.decomposition_type == "spiral":
                if self.num_total_regions > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n((frame_w,frame_h),self.num_total_regions,dev,dtype)
                    patch_base_size = min(frame_w, frame_h)
                    spiral_bboxes_current = torch.zeros(self.num_total_regions, 4, device=dev, dtype=dtype)
                    patch_hs = patch_base_size * spiral_scales[:,0] / 2.0
                    patch_ws = patch_hs
                    spiral_bboxes_current[:,0]=torch.clamp(spiral_centers[:,0]-patch_ws,0,frame_w-EPS)
                    spiral_bboxes_current[:,1]=torch.clamp(spiral_centers[:,1]-patch_hs,0,frame_h-EPS)
                    spiral_bboxes_current[:,2]=torch.clamp(spiral_centers[:,0]+patch_ws,spiral_bboxes_current[:,0]+EPS,frame_w)
                    spiral_bboxes_current[:,3]=torch.clamp(spiral_centers[:,1]+patch_hs,spiral_bboxes_current[:,1]+EPS,frame_h)
                    all_bboxes_for_frame_list.append(spiral_bboxes_current)
            else: # subdivide
                if self.num_total_regions > 0:
                    subdiv_bboxes = golden_subdivide_rect_fixed_n((frame_w,frame_h),self.num_total_regions,dev,dtype)
                    all_bboxes_for_frame_list.append(subdiv_bboxes)

            if not all_bboxes_for_frame_list: # Fallback if no bboxes generated
                final_bboxes_for_frame = torch.tensor([[0,0,frame_w,frame_h]] * self.num_total_regions, dtype=dtype, device=dev)
            else:
                final_bboxes_for_frame = torch.cat(all_bboxes_for_frame_list, dim=0)


            if final_bboxes_for_frame.shape[0] < self.num_total_regions:
                padding_count = self.num_total_regions - final_bboxes_for_frame.shape[0]
                if final_bboxes_for_frame.shape[0] > 0:
                    padding = final_bboxes_for_frame[-1:].repeat(padding_count, 1)
                else:
                    padding = torch.tensor([[0,0,frame_w,frame_h]] * padding_count, dtype=dtype, device=dev)
                final_bboxes_for_frame = torch.cat([final_bboxes_for_frame, padding], dim=0)
            elif final_bboxes_for_frame.shape[0] > self.num_total_regions:
                final_bboxes_for_frame = final_bboxes_for_frame[:self.num_total_regions]

            scaled_bboxes_for_roi = torch.zeros_like(final_bboxes_for_frame) # final_bboxes_for_frame is already on dev, dtype

            # Convert map dimensions to tensors of the same dtype and device as final_bboxes_for_frame
            map_w_tensor = torch.tensor(map_w, dtype=final_bboxes_for_frame.dtype, device=final_bboxes_for_frame.device)
            map_h_tensor = torch.tensor(map_h, dtype=final_bboxes_for_frame.dtype, device=final_bboxes_for_frame.device)
            # zero_tensor = torch.tensor(0.0, dtype=final_bboxes_for_frame.dtype, device=final_bboxes_for_frame.device) # If using tensor for min 0

            # For these clamps, input is Tensor, min and max are Numbers.
            scaled_bboxes_for_roi[:,0] = torch.clamp(final_bboxes_for_frame[:,0] * scale_w, 
                                                     min=0.0, 
                                                     max=(map_w_tensor - EPS).item()) # Use .item() to get Python scalar
            scaled_bboxes_for_roi[:,1] = torch.clamp(final_bboxes_for_frame[:,1] * scale_h, 
                                                     min=0.0, 
                                                     max=(map_h_tensor - EPS).item())

            # For these clamps, min is a Tensor, so max must also be a Tensor.
            min_x2_tensor = scaled_bboxes_for_roi[:,0] + EPS
            min_y2_tensor = scaled_bboxes_for_roi[:,1] + EPS
            
            scaled_bboxes_for_roi[:,2] = torch.clamp(final_bboxes_for_frame[:,2] * scale_w, 
                                                     min=min_x2_tensor,  # Tensor
                                                     max=map_w_tensor)   # Tensor
            scaled_bboxes_for_roi[:,3] = torch.clamp(final_bboxes_for_frame[:,3] * scale_h, 
                                                     min=min_y2_tensor,  # Tensor
                                                     max=map_h_tensor)   # Tensor
                                                     
            # Add batch index for roi_align (expects list of [batch_idx, x1, y1, x2, y2])
            # Since we process one batch item feature_maps[b_idx] at a time, batch_idx for roi_align is 0
            rois_with_batch_idx = torch.cat([torch.full((scaled_bboxes_for_roi.shape[0],1), 0, device=dev, dtype=dtype), scaled_bboxes_for_roi], dim=1)

            aligned_regions = roi_align(
                feature_maps[b_idx].unsqueeze(0), # Input feature map for this batch item
                rois_with_batch_idx,             # ROIs for this batch item
                self.region_roi_output_size,
                spatial_scale=1.0, # Already scaled bboxes
                aligned=True
            )
            aligned_regions_flat = aligned_regions.view(self.num_total_regions, -1)
            batch_region_features_list.append(self.region_projector(aligned_regions_flat))
        return torch.stack(batch_region_features_list)

# =====================================================================
# Diffusion Model Specific Components
# =====================================================================
class InitialFrameAutoencoderCNN(nn.Module):
    def __init__(self, image_channels: int, feature_dim: int, image_size: Tuple[int, int]):
        super().__init__()
        self.image_channels=image_channels
        self.feature_dim=feature_dim
        self.image_h,self.image_w=image_size
        self.encoder_convs = nn.Sequential(
            nn.Conv2d(image_channels,32,4,2,1),nn.GroupNorm(8,32),nn.GELU(),
            nn.Conv2d(32,64,4,2,1),nn.GroupNorm(16,64),nn.GELU(),
            nn.Conv2d(64,128,4,2,1),nn.GroupNorm(32,128),nn.GELU(),
            nn.Conv2d(128,256,4,2,1),nn.GroupNorm(64,256),nn.GELU()
        )
        with torch.no_grad():
            self.conv_out_shape=self.encoder_convs(torch.randn(1,image_channels,self.image_h,self.image_w)).shape
        self.flattened_dim=self.conv_out_shape[1]*self.conv_out_shape[2]*self.conv_out_shape[3]
        self.encoder_fc = nn.Linear(self.flattened_dim,feature_dim)
        self.decoder_fc = nn.Linear(feature_dim,self.flattened_dim)
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),nn.GroupNorm(32,128),nn.GELU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(16,64),nn.GELU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(8,32),nn.GELU(),
            nn.ConvTranspose2d(32,image_channels,4,2,1),nn.Tanh()
        )
        self.apply(init_weights_general)

    def encode(self, x_frames: torch.Tensor) -> torch.Tensor:
        orig_dim=x_frames.dim()
        is_batched_seq = orig_dim==5
        if is_batched_seq:
            B,N,C,H,W = x_frames.shape
            x_frames=x_frames.reshape(B*N,C,H,W)
        feats=self.encoder_convs(x_frames)
        feats_flat=feats.view(feats.size(0),-1)
        vec=self.encoder_fc(feats_flat)
        return vec.view(B,N,-1) if is_batched_seq else vec

    def encode_conv_features(self, x_frames: torch.Tensor) -> torch.Tensor:
        return self.encoder_convs(x_frames)

    def decode(self, features_vec: torch.Tensor) -> torch.Tensor:
        orig_dim=features_vec.dim()
        is_batched_seq = orig_dim==3
        if is_batched_seq:
            B,N,D = features_vec.shape
            features_vec=features_vec.reshape(B*N,D)
        x=self.decoder_fc(features_vec)
        x=x.view(-1,self.conv_out_shape[1],self.conv_out_shape[2],self.conv_out_shape[3])
        pixels=self.decoder_convs(x)
        return pixels.view(B,N,self.image_channels,self.image_h,self.image_w) if is_batched_seq else pixels

class SinusoidalPhiEmbedding(nn.Module):
    def __init__(self, dim: int, base_freq_phi_scaled: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base_freq = base_freq_phi_scaled
        half_dim = dim // 2
        denominators = torch.exp(torch.arange(half_dim) * -(math.log(self.base_freq) / (half_dim-1 if half_dim > 1 else 1.0))) # Ensure float division
        self.register_buffer("denominators", denominators)
        logger.info(f"SinusoidalPhiEmbedding: Dim {dim}, BaseFreq {base_freq_phi_scaled}.")

    def forward(self, t: torch.Tensor, phi_time_scale: float = 1.0) -> torch.Tensor:
        t_scaled = t.float() * phi_time_scale
        args = t_scaled[:, None] * self.denominators[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0,1), value=0.0)
        return embedding

class WuBuSTDiffusionNet(nn.Module):
    def __init__(self, wubu_s_config: Dict, wubu_t_config: Dict, video_config: Dict, gaad_config: Dict, time_embedding_dim: int):
        super().__init__()
        self.video_config = video_config
        self.gaad_config = gaad_config
        self.initial_global_feature_dim = video_config["initial_cnn_feature_dim"]
        self.gaad_region_feature_dim = gaad_config["gaad_region_feature_dim"]
        self.wubu_s_output_dim = video_config["wubu_s_output_dim"]
        self.wubu_t_output_dim = video_config["wubu_t_output_dim"]

        self.frame_autoencoder = InitialFrameAutoencoderCNN(video_config["num_channels"], self.initial_global_feature_dim, video_config["image_size"])
        self.gaad_processor = GAADFrameProcessor(gaad_config["num_regions"], gaad_config["region_roi_output_size"], self.frame_autoencoder.encoder_convs, self.frame_autoencoder.conv_out_shape[1], self.gaad_region_feature_dim, gaad_config["decomposition_type"])

        self.wubu_s = FullyHyperbolicWuBuNestingModel(self.gaad_region_feature_dim, self.wubu_s_output_dim, wubu_s_config)
        self.wubu_t = FullyHyperbolicWuBuNestingModel(self.wubu_s_output_dim, self.wubu_t_output_dim, wubu_t_config)

        # MODIFIED PART: Separate SinusoidalPhiEmbedding from the rest of the MLP
        self.time_sin_embedding = SinusoidalPhiEmbedding(time_embedding_dim, base_freq_phi_scaled=gaad_config.get("phi_time_base_freq", 10000.0))
        self.time_fc_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2), nn.GELU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim)
        )
        # END MODIFIED PART

        head_input_dim = self.initial_global_feature_dim + self.wubu_t_output_dim + time_embedding_dim
        self.noise_pred_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim * 2), nn.GELU(),
            nn.Linear(head_input_dim * 2, self.initial_global_feature_dim)
        )
        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"WuBuSTDiffNet+GAAD ({param_count:,} params): GlobalFeat {self.initial_global_feature_dim}, GAADRegionFeat {self.gaad_region_feature_dim}, WuBuS_Out(s_t) {self.wubu_s_output_dim}, WuBuT_Out(ctx) {self.wubu_t_output_dim}")

    def forward(self, xt_target_global_features: torch.Tensor, conditioning_frames_pixels: torch.Tensor, time_t_integers: torch.Tensor) -> torch.Tensor:
        B = xt_target_global_features.shape[0]
        num_cond_frames = conditioning_frames_pixels.shape[1]
        dev = xt_target_global_features.device
        # --- THIS LINE WAS MISSING IN THE PREVIOUS FULL CLASS OUTPUT I GAVE YOU ---
        dtype_to_use = self.frame_autoencoder.encoder_fc.weight.dtype # Use consistent dtype 
        # --- END MISSING LINE ---

        s_t_list = []
        for i in range(num_cond_frames):
            cond_frame_pixels_i = conditioning_frames_pixels[:, i, ...].to(dtype=dtype_to_use) # Corrected use of dtype_to_use
            gaad_region_features_i = self.gaad_processor(cond_frame_pixels_i) # (B, NumRegions, gaad_region_feature_dim)

            s_t_i_from_wubu_s = self.wubu_s(gaad_region_features_i) # (B, NumRegions, wubu_s_output_dim)

            # Aggregate over regions
            if s_t_i_from_wubu_s.shape[1] == self.gaad_config["num_regions"] and self.gaad_config["num_regions"] > 0:
                s_t_i_aggregated = torch.mean(s_t_i_from_wubu_s, dim=1)
            elif s_t_i_from_wubu_s.shape[1] == 1: # WuBu-S output already aggregated
                s_t_i_aggregated = s_t_i_from_wubu_s.squeeze(1)
            elif s_t_i_from_wubu_s.numel() > 0 and s_t_i_from_wubu_s.shape[1] > 0:
                 logger.warning(f"Unexpected WuBu-S output shape for regions: {s_t_i_from_wubu_s.shape}. Attempting mean pool on dim 1.")
                 s_t_i_aggregated = torch.mean(s_t_i_from_wubu_s, dim=1)
            else: # Fallback for empty or zero-region output
                 s_t_i_aggregated = torch.zeros(B, self.wubu_s_output_dim, device=dev, dtype=dtype_to_use) # Corrected use of dtype_to_use
            s_t_list.append(s_t_i_aggregated)

        if s_t_list:
            s_sequence_for_wubu_t = torch.stack(s_t_list, dim=1) # (B, NumCondFrames, wubu_s_output_dim)
        else: # No conditioning frames processed
            s_sequence_for_wubu_t = torch.empty(B, 0, self.wubu_s_output_dim, device=dev, dtype=dtype_to_use) # Corrected use of dtype_to_use


        temporal_context_ctx = torch.zeros(B, self.wubu_t_output_dim, device=dev, dtype=dtype_to_use) # Corrected use of dtype_to_use
        if s_sequence_for_wubu_t.shape[1] > 0:
            temporal_context_ctx_full = self.wubu_t(s_sequence_for_wubu_t) # (B, NumCondFrames, wubu_t_output_dim)
            if temporal_context_ctx_full.shape[1] == num_cond_frames and num_cond_frames > 0:
                temporal_context_ctx = torch.mean(temporal_context_ctx_full, dim=1)
            elif temporal_context_ctx_full.shape[1] == 1:
                temporal_context_ctx = temporal_context_ctx_full.squeeze(1)
            # else: could be error or unexpected aggregation in WuBu-T

        time_sin_emb_output = self.time_sin_embedding(time_t_integers, phi_time_scale=self.gaad_config.get("phi_time_diffusion_scale", 1.0))
        time_emb = self.time_fc_mlp(time_sin_emb_output).to(dtype=dtype_to_use) # dtype_to_use is needed here
        
        temporal_context_ctx_3d = temporal_context_ctx.unsqueeze(1).to(dtype=dtype_to_use) # Added .to(dtype_to_use) here too for consistency
        time_emb_3d = time_emb.unsqueeze(1) # time_emb is already .to(dtype_to_use)
        
        combined_features_for_head = torch.cat([
            xt_target_global_features.to(dtype=dtype_to_use),
            temporal_context_ctx_3d, # Already converted
            time_emb_3d                # Already converted
        ], dim=-1)
        
        predicted_noise_3d = self.noise_pred_head(combined_features_for_head)

        if not torch.isfinite(predicted_noise_3d).all():
            return torch.nan_to_num(predicted_noise_3d, nan=0.0)
        return predicted_noise_3d
        
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def q_sample(x_start_features: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x_start_features)
    reshape_dims = [-1] + [1]*(x_start_features.dim()-1)
    sqrt_alpha_t = sqrt_alphas_cumprod.gather(0,t).view(*reshape_dims)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod.gather(0,t).view(*reshape_dims)
    return sqrt_alpha_t*x_start_features + sqrt_one_minus_alpha_t*noise

class VideoFrameDataset(Dataset):
    def __init__(self,
                 video_path: str,                 # Path to the single video file (e.g., dummy_video.mp4)
                 num_frames_total: int,           # Total frames per sample (e.g., num_input + num_predict)
                 image_size: Tuple[int, int],     # Target (Height, Width) for T.Resize, e.g., (180, 320)
                 frame_skip: int = 1,
                 data_fraction: float = 1.0):
        super().__init__()
        self.video_path = video_path
        self.num_frames_total = num_frames_total
        self.image_size = image_size  # For T.Resize, should match video's actual H, W if no further resize needed
        self.frame_skip = frame_skip

        if not os.path.isfile(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        logger.info(f"Attempting to load entire video into RAM: {self.video_path}")
        try:
            # output_format="THWC" gives (NumFrames, Height, Width, Channels) as torch.uint8
            # This format is convenient for slicing individual (H,W,C) frames.
            video_data = video_io.read_video(self.video_path, output_format="THWC", pts_unit="sec")
            self.video_frames_in_ram = video_data[0] # read_video returns (video_frames, audio_frames, info)
            self.source_video_fps = video_data[2].get('video_fps', 30.0) # Get FPS from video info, default 30
            
            # video_frames_in_ram is a Tensor on CPU.
            # Ensure it's contiguous for potentially faster slicing, though often not strictly necessary.
            self.video_frames_in_ram = self.video_frames_in_ram.contiguous()
            
            ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3)
            logger.info(
                f"Successfully loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, "
                f"Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. "
                f"Estimated RAM usage for frames: {ram_usage_gb:.2f} GB."
            )
            if ram_usage_gb > 16: # Arbitrary threshold for a strong warning
                 logger.warning(f"Video RAM usage ({ram_usage_gb:.2f} GB) is very high. Ensure you have enough system RAM.")

        except Exception as e:
            logger.error(f"Failed to load video '{self.video_path}' into RAM: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load video '{self.video_path}' into RAM.") from e

        # Transformation pipeline
        # Input to transform will be a single frame tensor (H, W, C) of dtype uint8 from self.video_frames_in_ram
        # T.ToPILImage() can handle this (H, W, C) tensor if it's on CPU.
        # T.Resize expects (H, W) and works on PIL images.
        # T.ToTensor converts a PIL Image (HWC) to a FloatTensor (CHW, 0.0-1.0)
        # T.Normalize then normalizes this tensor.
        self.transform = T.Compose([
            T.ToPILImage(),             # Converts (H,W,C) uint8 tensor to PIL Image
            T.Resize(self.image_size),  # Resizes PIL Image to target (H,W)
            T.ToTensor(),               # Converts PIL Image to (C,H,W) float tensor [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.num_disk_frames = self.video_frames_in_ram.shape[0]
        self.samples = []  # List of starting indices for sequences

        required_span_len = (self.num_frames_total - 1) * self.frame_skip + 1

        if self.num_disk_frames >= required_span_len:
            for i in range(self.num_disk_frames - required_span_len + 1):
                self.samples.append(i)
        else:
            logger.warning(
                f"Not enough frames ({self.num_disk_frames}) in loaded video '{self.video_path}' "
                f"to form even one sample requiring a span of {required_span_len} frames "
                f"(num_frames_total={self.num_frames_total}, frame_skip={self.frame_skip})."
            )

        if data_fraction < 1.0 and len(self.samples) > 1:
            num_to_keep = max(1, int(len(self.samples) * data_fraction))
            self.samples = random.sample(self.samples, num_to_keep)
            logger.info(f"Using {data_fraction*100:.2f}% of samples: {len(self.samples)} samples.")

        if not self.samples:
            logger.error(
                f"VideoFrameDataset: No valid samples could be created. "
                f"Check frame count ({self.num_disk_frames}), num_frames_total ({self.num_frames_total}), "
                f"and frame_skip ({self.frame_skip})."
            )
        else:
            logger.info(
                f"VideoFrameDataset initialized. Using video loaded into RAM: {self.video_path}. "
                f"Total frames in RAM: {self.num_disk_frames}. Created {len(self.samples)} samples. "
                f"Each sample has {self.num_frames_total} frames (with skip {self.frame_skip})."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_frame_idx_in_ram = self.samples[idx]

        frames_for_sample = []
        for i in range(self.num_frames_total):
            current_frame_offset = i * self.frame_skip
            actual_frame_idx_in_ram = start_frame_idx_in_ram + current_frame_offset

            if actual_frame_idx_in_ram < self.num_disk_frames:
                try:
                    frame_tensor_hwc_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram] # Shape: (H, W, C)
                    
                    # ===> CORRECTED PART: Permute to (C, H, W) before ToPILImage <===
                    frame_tensor_chw_uint8 = frame_tensor_hwc_uint8.permute(2, 0, 1)
                    
                    # T.ToPILImage() expects (C,H,W) for a tensor input
                    transformed_frame = self.transform(frame_tensor_chw_uint8)
                    frames_for_sample.append(transformed_frame)
                except Exception as e:
                    logger.error(f"Error transforming frame at index {actual_frame_idx_in_ram} for sample idx {idx}: {e}", exc_info=True)
                    raise e
            else:
                logger.error(f"Frame index {actual_frame_idx_in_ram} out of bounds for in-RAM frames (total: {self.num_disk_frames}). Sample index: {idx}")
                raise IndexError(f"Frame index {actual_frame_idx_in_ram} out of bounds for in-RAM frames.")

        if len(frames_for_sample) != self.num_frames_total:
            logger.error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total} for sample idx {idx}")
            raise ValueError(f"Incorrect number of frames loaded for sample idx {idx}")
        
        return torch.stack(frames_for_sample)
        
class DiffusionTrainer:
    def __init__(self, model: WuBuSTDiffusionNet, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], args: argparse.Namespace, rank: int, world_size: int, ddp_active: bool, video_config: Dict, gaad_config: Dict):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.ddp_active = ddp_active
        self.video_config = video_config
        self.gaad_config = gaad_config
        self.am_main_process = (rank == 0)

        self.timesteps = args.timesteps
        if args.beta_schedule=='linear':
            self.betas = linear_beta_schedule(args.timesteps,args.beta_start,args.beta_end).to(device)
        else: # cosine
            self.betas = cosine_beta_schedule(args.timesteps,args.cosine_s).to(device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0/(self.alphas+EPS))
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0)
        self.posterior_variance = torch.clamp(self.betas*(1.-self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS),min=EPS*10)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS)
        self.posterior_mean_coef2 = (1.-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.-self.alphas_cumprod+EPS)

        self.scaler = amp.GradScaler(enabled=args.use_amp and device.type=='cuda')
        self.global_step=0
        self.current_epoch=0
        self.best_val_loss=float('inf')
        self.last_val_metrics:Dict[str,Any]={}
        if self.am_main_process:
            os.makedirs(args.checkpoint_dir,exist_ok=True)
        m = self.model.module if ddp_active else self.model
        self.frame_autoencoder = m.frame_autoencoder

    def _get_x0_target_global_features(self, target_pixels: torch.Tensor) -> torch.Tensor:
        return self.frame_autoencoder.encode(target_pixels)

    def train_step(self, batch_video_frames: torch.Tensor):
        num_cond = self.video_config["num_input_frames"]
        cond_pixels = batch_video_frames[:,:num_cond,...].to(self.device)
        target_pixels = batch_video_frames[:,num_cond:,...].to(self.device) # Corrected slicing for target
        B = target_pixels.shape[0]

        x0_target_global_features = self._get_x0_target_global_features(target_pixels)
        t = torch.randint(0,self.timesteps,(B,),device=self.device,dtype=torch.long)
        noise = torch.randn_like(x0_target_global_features)
        xt_target_global_features = q_sample(x0_target_global_features,t,self.sqrt_alphas_cumprod,self.sqrt_one_minus_alphas_cumprod,noise)

        with amp.autocast(device_type=self.device.type,enabled=self.args.use_amp and self.device.type=='cuda'):
            pred_noise = self.model(xt_target_global_features, cond_pixels, t)
            loss = F.mse_loss(pred_noise,noise)
        return loss, pred_noise.detach(), noise.detach()

    def train(self, start_epoch:int=0, initial_global_step:int=0):
            self.global_step = initial_global_step
            self.current_epoch = start_epoch
            if self.am_main_process:
                logger.info(f"Starting training from epoch {start_epoch}, step {initial_global_step}...")
            
            # For overall interval logging (as before)
            total_loss_interval = 0.0
            items_interval = 0
            
            # NEW: Accumulators for the current gradient accumulation cycle for Q-controller
            current_cycle_loss_sum = 0.0
            micro_batches_in_current_cycle = 0
    
            for epoch in range(start_epoch, self.args.epochs):
                self.current_epoch = epoch
                if self.am_main_process:
                    logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting...")
                if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)
                
                self.model.train()
                
                # Correctly estimate total micro-batches for tqdm if possible
                # This uses the logic from your WuBuNest_TrainerV1.py for better estimation
                total_micro_batches_estimate = None
                try:
                    dataset_len = 0
                    if hasattr(self.train_loader.sampler, '__len__'):
                        dataset_len = len(self.train_loader.sampler)
                    elif hasattr(self.train_loader.dataset, '__len__'):
                        dset_total_len = len(self.train_loader.dataset)
                        dataset_len = dset_total_len // self.world_size if self.world_size > 1 else dset_total_len
                    
                    if dataset_len > 0:
                        loader_batch_size = self.train_loader.batch_size or 1
                        total_micro_batches_estimate = math.ceil(dataset_len / loader_batch_size)
                except Exception:
                    logger.warning("Could not accurately estimate epoch length for tqdm.", exc_info=False)
    
                prog_bar = tqdm(self.train_loader, 
                                desc=f"Epoch {epoch+1}", 
                                disable=not self.am_main_process or os.getenv('CI')=='true', 
                                dynamic_ncols=True,
                                total=total_micro_batches_estimate)
                
                # micro_batch_idx was already defined in the original script's outer scope.
                # We need to ensure it's reset per epoch, or rather, manage accumulation cycle correctly.
                # Let's use micro_batches_in_current_cycle which is reset properly.
    
                for batch_idx, batch_frames_raw in enumerate(prog_bar):
                    batch_frames = batch_frames_raw.to(self.device)
                    
                    # Determine if an optimizer step should occur after this micro-batch
                    # is_last_batch_in_loader = (batch_idx + 1) == len(self.train_loader) # Problematic with IterableDataset
                    is_last_batch_in_loader = (batch_idx + 1) == total_micro_batches_estimate if total_micro_batches_estimate is not None else False
    
                    # Loss for the current micro-batch
                    loss_this_micro_batch = torch.tensor(0.0, device=self.device)
    
                    # DDP sync context (no_sync if accumulating and not ready for step)
                    # An optimizer step occurs when micro_batches_in_current_cycle + 1 == grad_accum_steps OR it's the last batch
                    is_optimizer_step_time = ((micro_batches_in_current_cycle + 1) % self.args.grad_accum_steps == 0) or \
                                            (is_last_batch_in_loader and (micro_batches_in_current_cycle + 1) > 0)
    
                    sync_context = contextlib.nullcontext()
                    if self.ddp_active and isinstance(self.model, DDP) and not is_optimizer_step_time:
                        sync_context = self.model.no_sync()
                    
                    with sync_context:
                        with (torch.autograd.detect_anomaly() if self.args.detect_anomaly else contextlib.nullcontext()):
                            try:
                                loss, _, _ = self.train_step(batch_frames) # train_step handles AMP internally for the model call
                                if torch.isnan(loss) or torch.isinf(loss):
                                    logger.warning(f"Rank {self.rank}: NaN/Inf loss from train_step. Skipping accumulation for this micro-batch.")
                                    # Optionally, if a micro-batch fails, you might want to skip the whole optimizer step
                                    # or try to recover, but for now, just don't accumulate its loss.
                                    if is_optimizer_step_time: # If it was time for an optim step, reset cycle vars
                                        current_cycle_loss_sum = 0.0
                                        micro_batches_in_current_cycle = 0
                                    continue # Skip to next micro-batch
    
                                loss_this_micro_batch = loss
                                loss_scaled_for_backward = loss / self.args.grad_accum_steps
                                
                                # Backward pass (scaler handles AMP)
                                self.scaler.scale(loss_scaled_for_backward).backward()
    
                            except Exception as e_train_step:
                                logger.error(f"Rank {self.rank}: Error in train_step or backward pass: {e_train_step}", exc_info=True)
                                if is_optimizer_step_time: # Reset if it was time for an optim step
                                    current_cycle_loss_sum = 0.0
                                    micro_batches_in_current_cycle = 0
                                # Ensure grads are cleared on optimizer if error happens mid-accumulation before an optimizer step
                                if not is_optimizer_step_time and self.args.grad_accum_steps > 1:
                                    self.optimizer.zero_grad(set_to_none=True) # Clear potentially partial/corrupted grads
                                continue # Skip to next micro-batch
    
                    # Accumulate for logging and Q-controller (using unscaled loss)
                    total_loss_interval += loss_this_micro_batch.item() * batch_frames.size(0)
                    items_interval += batch_frames.size(0)
                    current_cycle_loss_sum += loss_this_micro_batch.item()
                    micro_batches_in_current_cycle += 1
    
                    if is_optimizer_step_time:
                        # 1. Unscale gradients
                        self.scaler.unscale_(self.optimizer)
    
                        # 2. Calculate current UNCLIPPED gradient norm for Q-Controller
                        current_unclipped_grad_norm = 0.0
                        params_for_norm_calc = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None and p.requires_grad]
                        if params_for_norm_calc:
                            try:
                                all_norms_sq = [torch.sum(p.grad.detach().float()**2) for p in params_for_norm_calc]
                                finite_norms_sq = [n_sq for n_sq in all_norms_sq if torch.isfinite(n_sq)]
                                if len(finite_norms_sq) == len(all_norms_sq) and finite_norms_sq: # All grads for norm are finite
                                    current_unclipped_grad_norm = torch.sqrt(torch.stack(finite_norms_sq).sum()).item()
                                elif finite_norms_sq: # Some grads were non-finite, norm of finite ones
                                    logger.warning(f"Rank {self.rank}, Step {self.global_step}: Some non-finite grads encountered calculating norm for Q-controller. Norm based on finite ones.")
                                    current_unclipped_grad_norm = torch.sqrt(torch.stack(finite_norms_sq).sum()).item()
                                else: # All grads for norm were non-finite or no grads
                                    logger.warning(f"Rank {self.rank}, Step {self.global_step}: All grads non-finite or no grads for Q-controller norm. Norm set to inf.")
                                    current_unclipped_grad_norm = float('inf')
                            except Exception as e_norm:
                                logger.error(f"Rank {self.rank}, Step {self.global_step}: Error calculating grad norm for Q: {e_norm}", exc_info=True)
                                current_unclipped_grad_norm = float('inf') # Treat as problematic
    
                        # 3. Global Gradient Clipping (if enabled by user)
                        if self.args.global_max_grad_norm > 0 and \
                        np.isfinite(current_unclipped_grad_norm) and \
                        current_unclipped_grad_norm > self.args.global_max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.global_max_grad_norm)
                            # For Q-controller, you might want it to see the norm *before* this global clip,
                            # or the *clipped* norm. Current_unclipped_grad_norm is pre-global-clip.
    
                        # 4. Q-Controller Logic
                        if hasattr(self.optimizer, 'q_controller') and self.optimizer.q_controller:
                            q_ctrl = self.optimizer.q_controller
                            avg_loss_for_q_cycle = current_cycle_loss_sum / micro_batches_in_current_cycle if micro_batches_in_current_cycle > 0 else 0.0
    
                            # Get current LR/Momentum (these are from *previous* Q-action or initial)
                            current_lr_for_q_state = self.optimizer.param_groups[0]['lr']
                            current_mom_for_q_state = self.optimizer.param_groups[0]['momentum']
                            
                            # Get state using this cycle's info
                            q_state = q_ctrl.get_state(
                                current_lr_for_q_state,
                                current_mom_for_q_state,
                                current_unclipped_grad_norm, # Now using the freshly calculated unclipped norm
                                avg_loss_for_q_cycle
                            )
    
                            # Update Q-table using previous state/action and current state/reward
                            if q_ctrl.prev_state is not None and \
                            q_ctrl.prev_action is not None and \
                            q_ctrl.prev_loss is not None and \
                            q_state is not None: # Ensure current q_state is valid
                                reward = q_ctrl.compute_reward(
                                    avg_loss_for_q_cycle, # Current cycle's average loss
                                    q_ctrl.prev_loss,     # Previous cycle's average loss
                                    current_unclipped_grad_norm
                                )
                                if np.isfinite(reward):
                                    q_ctrl.update(q_ctrl.prev_state, q_ctrl.prev_action, reward, q_state)
                                else:
                                    logger.warning(f"Rank {self.rank}, Step {self.global_step}: Q-Controller encountered non-finite reward ({reward}). Skipping Q-update.")
                            
                            # Store current info for the next Q-update cycle
                            q_ctrl.prev_state = q_state
                            if q_state is not None: # Only choose new action if current state is valid
                                q_ctrl.prev_action = q_ctrl.choose_action(q_state) # Action for *NEXT* optimizer step
                            else: # If current state is bad, perhaps revert to a default action or no-op
                                logger.warning(f"Rank {self.rank}, Step {self.global_step}: Q-state was None. Q-action for next step might be default/stale.")
                                # q_ctrl.prev_action = {'lr_scale': 1.0, 'momentum_scale': 1.0} # Optionally reset action
                            q_ctrl.prev_loss = avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else q_ctrl.prev_loss # Avoid storing NaN prev_loss
    
                        # 5. Optimizer Step (applies gradients)
                        # The optimizer will internally use the LR/Momentum set by q_ctrl.prev_action
                        # from the *previous* Q-cycle (applied at the start of its step method).
                        self.scaler.step(self.optimizer) 
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True) # Resets optimizer.grad_stats too
    
                        self.global_step += 1
                        
                        # Reset for next accumulation cycle
                        current_cycle_loss_sum = 0.0
                        micro_batches_in_current_cycle = 0
    
                        # Logging (as before, but now Q-controller gets better info)
                        if self.global_step % self.args.log_interval == 0 and self.am_main_process:
                            # Use avg_loss_for_q_cycle for "Train Loss (Cycle Avg)"
                            log_lr = self.optimizer.param_groups[0]['lr'] # This is LR *after* Q-action for this step was applied
                            log_metrics = {
                                "train/loss_cycle_avg": avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else -1.0,
                                "train/lr_effective": log_lr, # LR used for this step
                                "train/grad_norm_unclipped_for_q": current_unclipped_grad_norm if np.isfinite(current_unclipped_grad_norm) else -1.0,
                                "epoch_frac": epoch + ((batch_idx + 1) / total_micro_batches_estimate if total_micro_batches_estimate else 0),
                                "global_step": self.global_step
                            }
                            logger.info(f"E {epoch+1}, S {self.global_step}, Loss(avg_cycle) {log_metrics['train/loss_cycle_avg']:.4f}, LR(eff) {log_metrics['train/lr_effective']:.2e}, GradNorm(Q) {log_metrics['train/grad_norm_unclipped_for_q']:.2f}")
                            if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                                wandb.log(log_metrics, step=self.global_step)
                            # Reset overall interval logging sums
                            total_loss_interval = 0.0
                            items_interval = 0
                        
                        # Save checkpoint (as before)
                        if self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0 and self.am_main_process:
                            self._save_checkpoint(is_intermediate=True, metrics={"train_loss_cycle_avg": avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else -1.0})
                
                # End of epoch (prog_bar closes automatically)
                if self.am_main_process:
                    # This overall average might be slightly different if some micro-batches were skipped
                    avg_epoch_loss_val = total_loss_interval / items_interval if items_interval > 0 else float('nan')
                    logger.info(f"Epoch {epoch+1} finished. Approx avg epoch loss (logged intervals): {avg_epoch_loss_val:.4f}")
    
                if self.val_loader and self.am_main_process: # Validation (as before)
                    val_metrics = self.validate() # Call your existing validate method
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run and val_metrics:
                        wandb.log({f"val/{k}": v for k,v in val_metrics.items()}, step=self.global_step)
    
    
                if self.am_main_process: # Save end-of-epoch checkpoint (as before)
                    save_metrics = self.last_val_metrics.copy() if self.last_val_metrics else {}
                    save_metrics["epoch_end_train_loss_logged_intervals_avg"] = avg_epoch_loss_val if np.isfinite(avg_epoch_loss_val) else -1.0
                    self._save_checkpoint(metrics=save_metrics)
    
    def validate(self) -> float:
        if not self.val_loader or not self.am_main_process:
            return float('inf')
        self.model.eval()
        total_val_loss=0.0
        total_val_items=0
        with torch.no_grad():
            for batch_frames_raw in tqdm(self.val_loader, desc="Validating", dynamic_ncols=True, disable=os.getenv('CI')=='true'):
                batch_frames=batch_frames_raw.to(self.device)
                num_cond=self.video_config["num_input_frames"]
                cond_pixels=batch_frames[:,:num_cond,...]
                target_pixels=batch_frames[:,num_cond:,...] # Corrected slicing
                B_val=target_pixels.shape[0]
                x0_target_global_features=self._get_x0_target_global_features(target_pixels)
                t_val=torch.randint(0,self.timesteps,(B_val,),device=self.device,dtype=torch.long)
                noise_val=torch.randn_like(x0_target_global_features)
                xt_target_global_features_val=q_sample(x0_target_global_features,t_val,self.sqrt_alphas_cumprod,self.sqrt_one_minus_alphas_cumprod,noise_val)
                with amp.autocast(device_type=self.device.type,enabled=self.args.use_amp and self.device.type=='cuda'):
                    pred_noise_val=self.model(xt_target_global_features_val,cond_pixels,t_val)
                    loss_val=F.mse_loss(pred_noise_val,noise_val)
                total_val_loss+=loss_val.item()*B_val
                total_val_items+=B_val
        avg_val_loss=total_val_loss/total_val_items if total_val_items>0 else float('inf')
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict]=None, is_best:bool=False):
        if not self.am_main_process:
            return
        m_save = self.model.module if self.ddp_active else self.model
        data = {
            'global_step':self.global_step,
            'epoch':self.current_epoch,
            'model_state_dict':m_save.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scaler_state_dict':self.scaler.state_dict() if self.args.use_amp and self.device.type=='cuda' else None,
            'args':vars(self.args),
            'metrics':metrics if metrics else self.last_val_metrics,
            'wubu_s_config':m_save.wubu_s.config if hasattr(m_save,'wubu_s') else {},
            'wubu_t_config':m_save.wubu_t.config if hasattr(m_save,'wubu_t') else {},
            'video_config':self.video_config,
            'gaad_config':self.gaad_config
        }
        fname_prefix="wubugaad_ckpt"
        if is_best:
            fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_best.pt")
        elif is_intermediate:
            fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_step{self.global_step}.pt")
        else:
            fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_ep{self.current_epoch+1}_step{self.global_step}.pt")
        try:
            torch.save(data,fpath)
            logger.info(f"Ckpt saved: {fpath}")
        except Exception as e:
            logger.error(f"Save ckpt error {fpath}: {e}",exc_info=True)

    def load_checkpoint(self, checkpoint_path:str) -> Tuple[int,int]:
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Ckpt {checkpoint_path} not found.")
            return 0,0
        try:
            ckpt=torch.load(checkpoint_path,map_location=self.device)
            m_load = self.model.module if self.ddp_active else self.model
            try:
                m_load.load_state_dict(ckpt['model_state_dict'])
            except RuntimeError as e:
                logger.warning(f"Strict load failed: {e}. Trying non-strict.")
                m_load.load_state_dict(ckpt['model_state_dict'],strict=False)

            if 'optimizer_state_dict' in ckpt and self.optimizer:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt and self.scaler and ckpt['scaler_state_dict'] is not None:
                 self.scaler.load_state_dict(ckpt['scaler_state_dict'])

            logger.info(f"Loaded ckpt from {checkpoint_path} (Step {ckpt.get('global_step',0)}, Epoch {ckpt.get('epoch',0)}).")
            return ckpt.get('global_step',0), ckpt.get('epoch',0)
        except Exception as e:
            logger.error(f"Load ckpt error {checkpoint_path}: {e}",exc_info=True)
            return 0,0

    @torch.no_grad()
    def p_sample(self, xt_target_global_features: torch.Tensor, conditioning_frames_pixels: torch.Tensor, t_tensor: torch.Tensor, t_int_val: int) -> torch.Tensor:
        m_ref = self.model.module if self.ddp_active else self.model
        m_ref.eval()
        s_o_m_a_c_t = self.sqrt_one_minus_alphas_cumprod.gather(0,t_tensor).view(-1,1)
        s_a_c_t = self.sqrt_alphas_cumprod.gather(0,t_tensor).view(-1,1)
        p_m_c1_t = self.posterior_mean_coef1.gather(0,t_tensor).view(-1,1)
        p_m_c2_t = self.posterior_mean_coef2.gather(0,t_tensor).view(-1,1)

        pred_noise = m_ref(xt_target_global_features,conditioning_frames_pixels,t_tensor)
        x0_hat_global_features = (xt_target_global_features - s_o_m_a_c_t * pred_noise) / (s_a_c_t + EPS)
        model_mean_global_features = p_m_c1_t * x0_hat_global_features + p_m_c2_t * xt_target_global_features

        if t_int_val==0:
            return model_mean_global_features
        else:
            p_l_v_t = self.posterior_log_variance_clipped.gather(0,t_tensor).view(-1,1)
            noise_s = torch.randn_like(xt_target_global_features)
            return model_mean_global_features + (0.5 * p_l_v_t).exp() * noise_s

    @torch.no_grad()
    def sample(self, conditioning_frames_pixels: torch.Tensor, num_inference_steps: Optional[int]=None) -> torch.Tensor:
        if not self.am_main_process:
            logger.warning(f"Rank {self.rank}: Sample on non-main. Skip.")
            return torch.empty(0,device=self.device)
        self.model.eval()
        B = conditioning_frames_pixels.shape[0]
        dev = conditioning_frames_pixels.device # Corrected: use conditioning_frames_pixels.device
        eff_steps = min(num_inference_steps if num_inference_steps is not None else self.timesteps, self.timesteps)
        target_global_feat_dim = self.video_config["initial_cnn_feature_dim"]
        
        # Initialize xt_target_global_features as 3D: (B, 1, Dim)
        # The '1' represents the single target frame sequence length we are working with.
        xt_target_global_features = torch.randn((B, 1, target_global_feat_dim), device=dev) # MODIFIED HERE
        
        logger.info(f"Rank {self.rank}: Sampling. BS={B}, CondFrames={conditioning_frames_pixels.shape[1]}, Steps={eff_steps}")
        time_sched = torch.linspace(self.timesteps-1,0,eff_steps,dtype=torch.long,device=dev)
        
        # Ensure model is on the correct device for sampling if not using DDP for sampling
        m_ref = self.model.module if self.ddp_active else self.model 
        m_ref.to(dev) # Ensure model is on the sampling device

        for t_val_tensor in tqdm(time_sched,desc="Sampling",leave=False,dynamic_ncols=True,disable=not self.am_main_process or os.getenv('CI')=='true'):
            t_int = t_val_tensor.item()
            t_batch = torch.full((B,),t_int,device=dev,dtype=torch.long)
            # p_sample expects and returns 3D tensor, so xt_target_global_features remains 3D
            xt_target_global_features = self.p_sample(xt_target_global_features,conditioning_frames_pixels.to(dev),t_batch,t_int) 
        
        # frame_autoencoder.decode expects (B, N, D) or (B*N, D).
        # Our xt_target_global_features is (B, 1, D), which fits the (B, N, D) pattern.
        predicted_pixels = self.frame_autoencoder.decode(xt_target_global_features)
        logger.info(f"Rank {self.rank}: Sampling finished. Returning predicted pixels.")
        return predicted_pixels

def seed_everything(seed:int,rank:int=0,world_size:int=1):
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu-GAAD-Phi Diffusion Model (v0.04)")
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_dir")
    parser.add_argument('--single_video_roll', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1) # For DDP
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_h', type=int, default=32)
    parser.add_argument('--image_w', type=int, default=32)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_input_frames', type=int, default=2)
    parser.add_argument('--num_predict_frames', type=int, default=1) # Note: current model predicts 1 frame features
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--initial_cnn_feature_dim', type=int, default=32)
    parser.add_argument('--wubu_s_output_dim', type=int, default=16)
    parser.add_argument('--wubu_t_output_dim', type=int, default=32)
    parser.add_argument('--gaad_num_regions', type=int, default=5)
    parser.add_argument('--gaad_region_roi_output_h', type=int, default=5)
    parser.add_argument('--gaad_region_roi_output_w', type=int, default=5)
    parser.add_argument('--gaad_region_feature_dim', type=int, default=16)
    parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])
    parser.add_argument('--phi_time_diffusion_scale', type=float, default=1.0)
    parser.add_argument('--phi_time_base_freq', type=float, default=10000.0)
    parser.add_argument('--diffusion_time_embedding_dim', type=int, default=32)
    parser.add_argument('--timesteps', type=int, default=50)
    parser.add_argument('--beta_schedule',type=str,default='cosine', choices=['linear','cosine'])
    parser.add_argument('--beta_start',type=float,default=1e-4)
    parser.add_argument('--beta_end',type=float,default=0.02)
    parser.add_argument('--cosine_s',type=float,default=0.008)
    parser.add_argument('--learning_rate',type=float,default=1e-4)
    parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0)
    parser.add_argument('--global_max_grad_norm',type=float,default=1.0) # Set to 0 to disable global clip
    parser.add_argument('--q_controller_enabled',action='store_true')
    parser.add_argument('--wubu_s_num_levels', type=int, default=1)
    parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[16])
    parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0])
    parser.add_argument('--wubu_s_use_rotation', action='store_true', default=False)
    parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_t_num_levels', type=int, default=1)
    parser.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[16])
    parser.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[1.0])
    parser.add_argument('--wubu_t_use_rotation', action='store_true', default=False)
    parser.add_argument('--wubu_t_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_dropout', type=float, default=0.1)
    parser.add_argument('--checkpoint_dir',type=str, default='wubugaadphi_diffusion_checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--grad_accum_steps',type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_interval',type=int, default=10) # Increased default
    parser.add_argument('--save_interval',type=int, default=100) # Increased default
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--wandb_project',type=str,default='WuBuGAADPhiDiffV04')
    parser.add_argument('--wandb_run_name',type=str,default=None)
    parser.add_argument('--detect_anomaly',action='store_true')
    parsed_args = parser.parse_args()

    # Validations for wubu_s
    if len(parsed_args.wubu_s_hyperbolic_dims) != parsed_args.wubu_s_num_levels:
        parser.error(f"Wubu-S: Length of wubu_s_hyperbolic_dims ({len(parsed_args.wubu_s_hyperbolic_dims)}) must match wubu_s_num_levels ({parsed_args.wubu_s_num_levels}).")
    s_curvatures = parsed_args.wubu_s_initial_curvatures
    s_num_levels = parsed_args.wubu_s_num_levels
    if len(s_curvatures) < s_num_levels:
        fill_val = s_curvatures[-1] if s_curvatures else 1.0
        s_curvatures.extend([fill_val] * (s_num_levels - len(s_curvatures)))
    parsed_args.wubu_s_initial_curvatures = s_curvatures[:s_num_levels]

    # Validations for wubu_t
    if len(parsed_args.wubu_t_hyperbolic_dims) != parsed_args.wubu_t_num_levels:
        parser.error(f"Wubu-T: Length of wubu_t_hyperbolic_dims ({len(parsed_args.wubu_t_hyperbolic_dims)}) must match wubu_t_num_levels ({parsed_args.wubu_t_num_levels}).")
    t_curvatures = parsed_args.wubu_t_initial_curvatures
    t_num_levels = parsed_args.wubu_t_num_levels
    if len(t_curvatures) < t_num_levels:
        fill_val = t_curvatures[-1] if t_curvatures else 1.0
        t_curvatures.extend([fill_val] * (t_num_levels - len(t_curvatures)))
    parsed_args.wubu_t_initial_curvatures = t_curvatures[:t_num_levels]

    return parsed_args

def _configure_wubu_stack(args: argparse.Namespace, prefix: str) -> Dict:
    config = DEFAULT_CONFIG_WUBU.copy()
    config["num_levels"] = getattr(args, f"{prefix}_num_levels")
    config["hyperbolic_dims"] = getattr(args, f"{prefix}_hyperbolic_dims")
    config["initial_curvatures"] = getattr(args, f"{prefix}_initial_curvatures") # Base curvatures
    config["use_rotation_in_transform"] = getattr(args, f"{prefix}_use_rotation", False)
    config["phi_influence_curvature"] = getattr(args, f"{prefix}_phi_influence_curvature", False)
    config["phi_influence_rotation_init"] = getattr(args, f"{prefix}_phi_influence_rotation_init", False)
    config["dropout"] = args.wubu_dropout

    num_levels_val = config['num_levels']
    num_transitions_val = max(0, num_levels_val-1)

    def _ensure_list_config_len(cfg_dict, key, target_len, default_fill_list_from_base):
        current_list_val = cfg_dict.get(key)
        if not isinstance(current_list_val, list) and current_list_val is not None :
            current_list_val = [current_list_val] * target_len # Repeat if single val given
        elif current_list_val is None:
            current_list_val = []

        # Determine default value for filling
        if default_fill_list_from_base:
            base_default_val = default_fill_list_from_base[0]
        elif "scales" in key or "curvatures" in key :
            base_default_val = 1.0
        elif "spread" in key:
            base_default_val = 0.1
        elif "types" in key:
            base_default_val = "linear"
        else: # Includes tangent_input_combination_dims, transform_hidden_dims
            base_default_val = None # For hidden_dims, or handle specifically if needed

        fill_val = current_list_val[-1] if current_list_val else base_default_val

        # Adjust list length
        if len(current_list_val) < target_len:
            cfg_dict[key] = (current_list_val + [fill_val]*(target_len-len(current_list_val)))[:target_len]
        elif len(current_list_val) > target_len:
            cfg_dict[key] = current_list_val[:target_len]

    # Ensure main lists related to levels are correctly sized
    for key_to_check, default_key_in_base_config in [
        ("hyperbolic_dims", "hyperbolic_dims"),
        ("initial_curvatures", "initial_curvatures"),
        ("initial_scales", "initial_scales"),
        ("initial_spread_values", "initial_spread_values"),
        ("boundary_points_per_level", "boundary_points_per_level"),
        # tangent_input_combination_dims might not be per-level, but per-model or a fixed list.
        # If it's intended to be per level, it should be handled like others.
        # For now, let's assume it's a list not necessarily matching num_levels.
        # If it *should* be per level, this logic needs to be reviewed for it.
        # ("tangent_input_combination_dims", "tangent_input_combination_dims") # Example if per-level
    ]:
         _ensure_list_config_len(config, key_to_check, num_levels_val, DEFAULT_CONFIG_WUBU[default_key_in_base_config])

    # Handle tangent_input_combination_dims specifically if it's not strictly per-level
    # (e.g. if it's a fixed list for the combiner MLP regardless of levels)
    # The current default is [32], a list. If it's meant to be a single list for the whole stack:
    if not isinstance(config.get("tangent_input_combination_dims"), list):
        config["tangent_input_combination_dims"] = [config.get("tangent_input_combination_dims", DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0])]


    # Ensure transition-related lists are correctly sized
    if num_transitions_val > 0:
        _ensure_list_config_len(config,"transform_types",num_transitions_val,DEFAULT_CONFIG_WUBU["transform_types"])
        _ensure_list_config_len(config,"transform_hidden_dims",num_transitions_val,DEFAULT_CONFIG_WUBU["transform_hidden_dims"])
    else: # No transitions if 0 or 1 level
        config["transform_types"]=[]
        config["transform_hidden_dims"]=[]
    return config

def main():
    args = parse_arguments()
    ddp_active = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    if ddp_active:
        rank=int(os.environ["RANK"])
        local_rank=int(os.environ["LOCAL_RANK"])
        world_size=int(os.environ["WORLD_SIZE"])
        init_process_group(backend="nccl")
        device=torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank=0
        local_rank=0
        world_size=1
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type=='cuda':
            _ = torch.cuda.set_device(device) # Ensures correct CUDA context for single GPU
    am_main_process=(rank==0)

    # Setup logging after DDP init to get correct rank
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO if am_main_process else logging.WARNING,
        format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s',
        force=True # Override any existing StreamHandler from basicConfig
    )

    logger.info(f"--- WuBuGAADPhiDiffV04 (R{rank}/{world_size},Dev {device},DDP:{ddp_active}) ---")
    seed_everything(args.seed,rank,world_size)
    if am_main_process:
        logger.info(f"Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name if args.wandb_run_name else f"wubugaadphi_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                   config=vars(args),
                   resume="allow",
                   id=wandb.util.generate_id() if wandb.run is None else wandb.run.id)


    video_config = {
        "image_size":(args.image_h,args.image_w),
        "num_channels":args.num_channels,
        "num_input_frames":args.num_input_frames,
        "num_predict_frames":args.num_predict_frames, # Note: model predicts features for 1 target frame
        "initial_cnn_feature_dim":args.initial_cnn_feature_dim,
        "wubu_s_output_dim":args.wubu_s_output_dim,
        "wubu_t_output_dim":args.wubu_t_output_dim
    }
    gaad_config = {
        "num_regions":args.gaad_num_regions,
        "region_roi_output_size":(args.gaad_region_roi_output_h, args.gaad_region_roi_output_w),
        "gaad_region_feature_dim":args.gaad_region_feature_dim,
        "decomposition_type":args.gaad_decomposition_type,
        "phi_time_diffusion_scale": args.phi_time_diffusion_scale,
        "phi_time_base_freq": args.phi_time_base_freq
    }
    wubu_s_config = _configure_wubu_stack(args,"wubu_s")
    wubu_t_config = _configure_wubu_stack(args,"wubu_t")
    if am_main_process:
        logger.info(f"VideoCfg:{video_config}\nGAADCfg:{gaad_config}\nWuBuS-Cfg:{wubu_s_config}\nWuBuT-Cfg:{wubu_t_config}")

    model=WuBuSTDiffusionNet(wubu_s_config,wubu_t_config,video_config,gaad_config,args.diffusion_time_embedding_dim).to(device)
    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model,log="all",log_freq=max(100,args.log_interval*5),log_graph=False) # log_graph can be slow
    if ddp_active:
        model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False) # Set find_unused_parameters based on actual need

    q_cfg = DEFAULT_CONFIG_QLEARN_DIFFUSION.copy() if args.q_controller_enabled else None
    optimizer = RiemannianEnhancedSGD(model.parameters(),lr=args.learning_rate,q_learning_config=q_cfg,max_grad_norm_risgd=args.risgd_max_grad_norm)

    # Create dummy video data if specified path is "demo_video_data_dir" and it's empty/doesn't exist
    if "demo_video_data" in args.video_data_path and not (os.path.exists(args.video_data_path) and (os.listdir(args.video_data_path) if os.path.isdir(args.video_data_path) else True)):
        if am_main_process:
            logger.info(f"Demo video data at {args.video_data_path} not found/empty. Creating dummy video...")
            if not os.path.isdir(args.video_data_path):
                os.makedirs(args.video_data_path, exist_ok=True)
            dummy_video_path = os.path.join(args.video_data_path, "dummy_video.mp4")
            if not os.path.exists(dummy_video_path) and VIDEO_IO_AVAILABLE:
                # Ensure enough frames for at least one sample
                min_raw_frames_needed = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                num_dummy_frames = max(25, min_raw_frames_needed + 5) # A bit more for safety
                # Create TCHW first as it's natural for PyTorch
                dummy_frames_tchw = torch.randint(0, 256, (num_dummy_frames, args.num_channels, args.image_h, args.image_w), dtype=torch.uint8)
                
                # Permute to THWC for video_io.write_video, which often expects this for its av backend
                # TCHW -> THWC  (0, 1, 2, 3) -> (0, 2, 3, 1)
                dummy_frames_tchw = torch.randint(0, 256, (num_dummy_frames, args.num_channels, args.image_h, args.image_w), dtype=torch.uint8)
                
                # For PyAV, we'll iterate and convert each frame
                try:
                    import av # Make sure PyAV is imported

                    container = av.open(dummy_video_path, mode='w')
                    stream = container.add_stream('mpeg4', rate=10) # Use a common codec like mpeg4
                    stream.width = args.image_w
                    stream.height = args.image_h
                    stream.pix_fmt = 'yuv420p' # A common pixel format for video

                    for i in range(num_dummy_frames):
                        # Get one frame: CHW, uint8 tensor
                        frame_chw_tensor = dummy_frames_tchw[i]
                        # Convert to HWC NumPy array for PyAV
                        frame_hwc_numpy = frame_chw_tensor.permute(1, 2, 0).numpy()
                        
                        av_frame = av.VideoFrame.from_ndarray(frame_hwc_numpy, format='rgb24')
                        
                        # PyAV handles pict_type internally when encoding packets
                        for packet in stream.encode(av_frame):
                            container.mux(packet)

                    # Flush stream
                    for packet in stream.encode():
                        container.mux(packet)
                    
                    # Close the file
                    container.close()
                    logger.info(f"Created dummy video with PyAV: {dummy_video_path} with {num_dummy_frames} frames.")

                except ImportError:
                    logger.error("PyAV not installed. Cannot create dummy video with direct PyAV method.")
                except Exception as e_av:
                    logger.error(f"Error creating dummy video with PyAV: {e_av}", exc_info=True)

            elif not VIDEO_IO_AVAILABLE:
                logger.warning("Cannot create dummy video: torchvision.io not available.")
        if ddp_active:
            torch.distributed.barrier() # Ensure main process creates data before others proceed

    if not os.path.exists(args.video_data_path):
        logger.error(f"Video data path {args.video_data_path} not found. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)


    total_frames_sample = args.num_input_frames + args.num_predict_frames
    
    # Construct the full path to your dummy_video.mp4
    # VIDEO_DATA_PATH from your batch file is "C:\projects\bytropix\draftPY\..\data\demo_video_data_dir"
    # So, dummy_video.mp4 should be inside it.
    path_to_dummy_video = os.path.join(args.video_data_path, "dummy_video.mp4") # Or whatever you named it

    try:
        train_dataset = VideoFrameDataset(
            video_path=path_to_dummy_video,  # Pass the direct path to the video file
            num_frames_total=total_frames_sample,
            image_size=(args.image_h, args.image_w), # This is (H, W), e.g., (180, 320)
            frame_skip=args.frame_skip
            # data_fraction=args.data_fraction # If you add such an arg
        )
    except Exception as e:
        logger.error(f"Failed to initialize VideoFrameDataset with in-RAM loading: {e}", exc_info=True)
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    if not train_dataset or len(train_dataset) == 0 :
        logger.error("Dataset is empty or failed to load. Check video path and content. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)


    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True,seed=args.seed) if ddp_active else None
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=(train_sampler is None),num_workers=args.num_workers,sampler=train_sampler,pin_memory=(device.type=='cuda'),worker_init_fn=lambda wid: seed_everything(args.seed+wid+rank*100,rank,world_size) if args.num_workers>0 else None,drop_last=True)

    trainer = DiffusionTrainer(model,optimizer,device,train_loader,None,args,rank,world_size,ddp_active,video_config,gaad_config)
    start_global_step,start_epoch=0,0
    if args.load_checkpoint:
        start_global_step,start_epoch=trainer.load_checkpoint(args.load_checkpoint)

    try:
        trainer.train(start_epoch=start_epoch,initial_global_step=start_global_step)
    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: Training interrupted.")
    except Exception as train_exc:
        logger.error(f"Rank {rank}: Training loop crashed: {train_exc}",exc_info=True)
    finally:
        if am_main_process:
            logger.info("Finalizing run...")
            trainer._save_checkpoint() # Save final checkpoint
            if args.epochs>0 and hasattr(trainer,'sample') and trainer.global_step > 0 and len(train_loader)>0:
                logger.info("DEMO SAMPLING...")
                try:
                    demo_batch_for_cond = next(iter(train_loader)) # Get a batch
                    demo_cond_pixels = demo_batch_for_cond[:, :args.num_input_frames, ...].to(device)
                    demo_cond_pixels = demo_cond_pixels[0:1] # Take first sample of the batch for demo

                    pred_pixels = trainer.sample(demo_cond_pixels, num_inference_steps=max(10, args.timesteps//5))
                    logger.info(f"Demo predicted pixels shape: {pred_pixels.shape}")
                    if pred_pixels.numel() > 0:
                        save_path_dir = os.path.join(args.checkpoint_dir, "demo_samples")
                        os.makedirs(save_path_dir, exist_ok=True)
                        for i in range(min(args.num_input_frames, demo_cond_pixels.shape[1])): # Handle if less cond frames available
                            save_image(demo_cond_pixels[0, i].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_path_dir, f"demo_cond_frame_{i}.png"))
                        save_image(pred_pixels[0].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_path_dir, "demo_pred_frame.png"))
                        logger.info(f"Saved demo sample frames to {save_path_dir}")
                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            wandb_images = [wandb.Image(demo_cond_pixels[0, i].cpu().clamp(-1,1)*0.5+0.5, caption=f"Cond Frame {i}") for i in range(min(args.num_input_frames, demo_cond_pixels.shape[1]))]
                            wandb_images.append(wandb.Image(pred_pixels[0].cpu().clamp(-1,1)*0.5+0.5, caption="Pred Frame"))
                            wandb.log({"demo_sequence": wandb_images}, step=trainer.global_step)
                except StopIteration:
                    logger.warning("Demo sampling skipped: DataLoader was empty (likely due to very small dataset/batch size).")
                except Exception as e:
                    logger.error(f"Demo sampling error: {e}",exc_info=True)

            if args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish()
        if ddp_active and is_initialized():
            destroy_process_group()
        logger.info(f"Rank {rank}: WuBuGAADPhiDiffusionNet (v0.04) script finished.")

if __name__ == "__main__":
    main()
