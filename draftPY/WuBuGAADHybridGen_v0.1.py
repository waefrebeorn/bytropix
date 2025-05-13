# WuBuGAADHybridGen_v0.1.py
# VAE-GAN Hybrid Model with GAAD-WuBu Regional Hyperbolic Latent Space
# Incorporates Optical Flow for Motion Encoding Branch.
# Operating directly on GAAD-defined regions with WuBu nesting.
# LAST UPDATE: Refactored from Diffusion to VAE-GAN (v0.1 internal rev from Diff v0.10.1)

# =====================================================================
# Python Imports and Setup
# =====================================================================
import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SubsetRandomSampler
import numpy as np

# Custom np.load to handle memmap and allow_pickle gracefully
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
import torchvision.transforms.functional as TF # Added for potential flow preprocessing
from PIL import Image
try:
    import torchvision.io as video_io
    VIDEO_IO_AVAILABLE = True
except ImportError:
    video_io = None
    VIDEO_IO_AVAILABLE = False
    print("Warn: torchvision.io unavailable.")
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None
    IMAGEIO_AVAILABLE = False
    print("Warn: imageio unavailable.")

from torchvision.ops import roi_align
from torchvision.utils import save_image

# --- Optical Flow Import ---
try:
    import torchvision.models.optical_flow as tv_flow
    OPTICAL_FLOW_AVAILABLE = True
    # Map string names to weights and models (add more as needed)
    FLOW_MODELS = {
        'raft_large': (tv_flow.Raft_Large_Weights.DEFAULT, tv_flow.raft_large),
        'raft_small': (tv_flow.Raft_Small_Weights.DEFAULT, tv_flow.raft_small),
        # Add other flow models if torchvision supports them with pre-trained weights
    }
except ImportError:
    tv_flow = None
    OPTICAL_FLOW_AVAILABLE = False
    FLOW_MODELS = {}
    print("Warning: torchvision.models.optical_flow not available. Motion branch will be disabled if selected.")


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    TORCHMETRICS_SSIM_AVAILABLE = True
except ImportError:
    StructuralSimilarityIndexMeasure = None
    TORCHMETRICS_SSIM_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False

# Setup logging
logger = logging.getLogger("WuBuGAADHybridGenV01") # Renamed logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')


# Constants and Default Configs
EPS = 1e-5
PHI = (1 + math.sqrt(5)) / 2
TAN_VEC_CLAMP_VAL = 1e4
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e8
MIN_WUBU_LEVEL_SCALE = EPS
MAX_WUBU_LEVEL_SCALE = 10.0


DEFAULT_CONFIG_WUBU = {
    "num_levels": 1, "hyperbolic_dims": [32], "initial_curvatures": [1.0],
    "dropout": 0.1,
    "relative_vector_aggregation": "sum",
    "aggregation_method": "concat_tangent",
    "use_level_descriptors": True, "use_level_spread": True,
    "level_descriptor_init_scale": 0.01,
    "curvature_min_value": EPS, "scale_min_value": EPS, "spread_min_value": EPS,
    "learnable_curvature": True, "initial_scales": [1.0], "learnable_scales": True,
    "learnable_spread": True, "initial_spread_values": [0.1],
    "boundary_points_per_level": [4], "tangent_input_combination_dims": [32],
    "use_tangent_flow": False, "tangent_flow_hidden_dim_ratio": 0.5,
    "tangent_flow_type": "mlp", "tangent_flow_scale": 1.0,
    "transform_types": [], "transform_hidden_dims": [],
    "use_rotation_in_transform": False,
    "phi_influence_curvature": False,
    "phi_influence_rotation_init": False,
    "use_transformer_block": False,
    "transformer_num_heads": 4,
    "transformer_feedforward_dim_ratio": 2.0,
}
# --- REMOVED Diffusion specific Q-learn config ---
# DEFAULT_CONFIG_QLEARN_DIFFUSION = { "learning_rate": 0.01, "discount": 0.95, "epsilon": 0.2, "epsilon_decay": 0.9998, "min_epsilon": 0.01, "lr_scale_options": [0.9,0.95,1.,1.05,1.1], "momentum_scale_options": [0.95,0.98,1.,1.01,1.02], "max_q_table_size": 10000}
DEFAULT_CONFIG_QLEARN_HYBRID = { "learning_rate": 0.01, "discount": 0.95, "epsilon": 0.2, "epsilon_decay": 0.9998, "min_epsilon": 0.01, "lr_scale_options": [0.9,0.95,1.,1.05,1.1], "momentum_scale_options": [0.95,0.98,1.,1.01,1.02], "max_q_table_size": 10000} # For VAE-GAN trainer

# --- REMOVED Transformer Noise Predictor Config ---
# DEFAULT_CONFIG_TRANSFORMER_NOISE_PREDICTOR = {
#     "num_layers": 4,
#     "num_heads": 8,
#     "d_model": 256,
#     "d_ff_ratio": 4.0,
#     "dropout": 0.1,
#     "activation": "gelu",
# }

# =====================================================================
# Geometric, Optimizer, WuBu Core Components (Largely Unchanged)
# =====================================================================
class HyperbolicUtils:
    @staticmethod
    def poincare_clip(x: torch.Tensor, c_scalar: float, radius: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_input_dtype = x.dtype; c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            x_compute = x.float(); x_compute = torch.nan_to_num(x_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(x_compute).all() else x_compute
            if original_input_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; x_compute = torch.clamp(x_compute, min=-f16_max, max=f16_max)
            return x_compute.to(original_input_dtype)
        sqrt_c = math.sqrt(c_scalar + eps); effective_radius_factor = min(radius, 1.0 - eps); max_norm_val_f32 = effective_radius_factor / sqrt_c
        x_compute = x.float(); x_compute = torch.nan_to_num(x_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(x_compute).all() else x_compute
        x_norm_sq = torch.sum(x_compute.pow(2), dim=-1, keepdim=True); sqrt_input_val = torch.clamp(x_norm_sq, min=0.0) + eps; sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=1.0, neginf=eps); sqrt_input_val.clamp_min_(eps); norm = torch.sqrt(sqrt_input_val); cond = norm > max_norm_val_f32; norm_plus_eps_for_div = norm + eps; norm_plus_eps_for_div.clamp_min_(eps); scale_factor = torch.where(cond, max_norm_val_f32 / norm_plus_eps_for_div, torch.ones_like(norm)); clipped_x_f32 = x_compute * scale_factor
        if original_input_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; clipped_x_f32 = torch.clamp(clipped_x_f32, min=-f16_max, max=f16_max)
        final_clipped_x = clipped_x_f32.to(original_input_dtype)
        return torch.nan_to_num(final_clipped_x,nan=0.0,posinf=float(max_norm_val_f32),neginf=-float(max_norm_val_f32)) if not torch.isfinite(final_clipped_x).all() else final_clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = v.dtype; c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            v_compute = v.float(); v_compute = torch.nan_to_num(v_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(v_compute).all() else v_compute
            if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; v_compute = torch.clamp(v_compute, min=-f16_max, max=f16_max)
            return v_compute.to(original_dtype)
        v_compute = v.float(); v_compute = torch.nan_to_num(v_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(v_compute).all() else v_compute
        v_norm_sq_unclamped = torch.sum(v_compute.pow(2), dim=-1, keepdim=True); v_norm_sq_clamped = torch.clamp(v_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL); sqrt_input_val = v_norm_sq_clamped + eps; sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps); sqrt_input_val.clamp_min_(eps); v_norm = torch.sqrt(sqrt_input_val)
        if not torch.isfinite(v_norm).all(): return HyperbolicUtils.poincare_clip(torch.zeros_like(v_compute), c_scalar, eps=eps).to(original_dtype)
        sqrt_c_val = math.sqrt(c_scalar + eps); scaled_radius_arg = float(scale_scalar) * sqrt_c_val * v_norm; tanh_input_val = torch.clamp(scaled_radius_arg, min=-30.0, max=30.0); tanh_term_val = torch.tanh(tanh_input_val); denominator_lambda_candidate = sqrt_c_val * v_norm + eps; denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps); lambda_v_val = torch.where(v_norm > eps, tanh_term_val / denominator_lambda_val, torch.full_like(v_norm, float(scale_scalar), dtype=torch.float32)); mapped_v_f32 = lambda_v_val * v_compute
        if not torch.isfinite(mapped_v_f32).all(): mapped_v_f32 = torch.zeros_like(v_compute)
        clipped_mapped_v_f32 = HyperbolicUtils.poincare_clip(mapped_v_f32, c_scalar, eps=eps); final_result = clipped_mapped_v_f32
        if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; final_result = torch.clamp(clipped_mapped_v_f32, min=-f16_max, max=f16_max)
        return final_result.to(original_dtype)

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = y.dtype; c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            y_compute = y.float(); y_compute = torch.nan_to_num(y_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(y_compute).all() else y_compute
            if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; y_compute = torch.clamp(y_compute, min=-f16_max, max=f16_max)
            return y_compute.to(original_dtype)
        y_clipped_original_dtype = HyperbolicUtils.poincare_clip(y, c_scalar, eps=eps); y_compute = y_clipped_original_dtype.float(); y_compute = torch.nan_to_num(y_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(y_compute).all() else y_compute
        y_norm_sq_unclamped = torch.sum(y_compute.pow(2), dim=-1, keepdim=True); y_norm_sq_clamped = torch.clamp(y_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL); sqrt_input_val = y_norm_sq_clamped + eps; sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps); sqrt_input_val.clamp_min_(eps); y_norm = torch.sqrt(sqrt_input_val)
        if not torch.isfinite(y_norm).all(): return torch.zeros_like(y, dtype=original_dtype)
        sqrt_c_val = math.sqrt(c_scalar + eps); arctanh_arg_raw = sqrt_c_val * y_norm; arctanh_arg_clamped = torch.clamp(arctanh_arg_raw, min=-1.0 + eps*10, max=1.0 - eps*10); atanh_term_val = torch.atanh(arctanh_arg_clamped); denominator_lambda_candidate = float(scale_scalar) * sqrt_c_val * y_norm + eps; denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps); default_lambda_y_val = 1.0 / max(float(scale_scalar), eps); lambda_y_val = torch.where(y_norm > eps, atanh_term_val / denominator_lambda_val, torch.full_like(y_norm, default_lambda_y_val, dtype=torch.float32)); mapped_y_f32 = lambda_y_val * y_compute
        if not torch.isfinite(mapped_y_f32).all(): mapped_y_f32 = torch.zeros_like(y_compute)
        final_result = mapped_y_f32
        if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; final_result = torch.clamp(mapped_y_f32, min=-f16_max, max=f16_max)
        return final_result.to(original_dtype)

    @staticmethod
    def exponential_map(v: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor: return HyperbolicUtils.scale_aware_exponential_map(v, c_scalar, scale_scalar=1.0, eps=eps)
    @staticmethod
    def logarithmic_map(y: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor: return HyperbolicUtils.scale_aware_logarithmic_map(y, c_scalar, scale_scalar=1.0, eps=eps)

class Manifold:
    def __init__(self, c_scalar=0.0): self.c = float(c_scalar)
    def proju(self, p: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def logmap0(self, p: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5): raise NotImplementedError
    def expmap(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor: return self.proju(p + dp) if self.c > 0 else p + dp
    @property
    def name(self) -> str: return self.__class__.__name__

class PoincareBall(Manifold):
    def __init__(self, c_scalar: float = 1.0):
        super().__init__(c_scalar)
        c_scalar = float(max(c_scalar, 0.0)) # Ensure c is non-negative
        if c_scalar <= 0:
            self.c = 0.0
            self.k = 0.
            self.sqrt_c = 0.
            self.radius = float('inf') # Explicitly set c to 0 if input is invalid
        else:
            self.c = c_scalar
            self.k = -self.c
            self.sqrt_c = math.sqrt(self.c) # Ensure self.c is used here
            self.radius = 1. / self.sqrt_c # Ensure self.sqrt_c is used here
        
        # self.max_norm should use self.radius which depends on self.sqrt_c
        self.max_norm = self.radius * (1. - EPS * 10) if self.c > 0 and self.radius != float('inf') else float('inf')
        self._name = f'PoincareBall(c={self.c:.3g})'

    @property
    def name(self) -> str:
        return self._name

    def proju(self, x: torch.Tensor) -> torch.Tensor:
        # Pass self.c from the instance, not a new calculation
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
        # p is the point on the manifold (parameter value)
        # dp is the Euclidean gradient at p
        
        if self.c <= 0: # Euclidean case or c is invalidly negative (should be caught by __init__)
            return dp

        # For Poincare ball model, c > 0
        # Project p to ensure it's strictly inside the ball for safety,
        # though parameters with 'manifold' attribute should already be on it.
        p_projected = self.proju(p) # Uses instance self.c

        # Calculate squared norm of the projected point
        p_norm_sq = torch.sum(p_projected.pow(2), dim=-1, keepdim=True)
        
        # Clamp p_norm_sq to be numerically stable and < 1/c (squared radius)
        # max_sq_norm_val = (1. / (self.c + EPS)) - (EPS * 10) # Slightly less than 1/c
        # Using self.radius for clarity:
        # self.radius = 1 / sqrt(c) -> self.radius^2 = 1/c
        # We need ||p|| < self.radius, so ||p||^2 < self.radius^2
        # self.max_norm is slightly less than self.radius. So self.max_norm^2 is target.
        if self.radius == float('inf'): # Should only happen if c became 0 somehow after init
            max_sq_norm_val = float('inf')
        else:
            # max_sq_norm_val = self.radius**2 * (1. - EPS*20) # Even safer margin
            # The max_norm in __init__ is (1/sqrt(c)) * (1 - EPS*10).
            # So max_norm_sq = (1/c) * (1 - EPS*10)^2 approx (1/c) * (1 - 2*EPS*10)
            max_sq_norm_val = self.max_norm**2

        # Ensure p_norm_sq does not exceed this strict maximum before multiplication with c
        p_norm_sq_clamped = torch.clamp(p_norm_sq, min=0.0, max=max_sq_norm_val)
        
        # The factor is ((1 - c * ||p||^2) / 2)^2
        # ||p||^2 is p_norm_sq_clamped
        # c is self.c
        
        term_inside_paren = 1. - self.c * p_norm_sq_clamped
        lambda_p_factor = term_inside_paren / 2.0
        
        # Square the factor
        riemannian_scaling_factor = lambda_p_factor.pow(2)
        
        # Clamp the final scaling factor to a minimum positive value
        # This prevents scaling by zero if lambda_p_factor was zero,
        # and also avoids issues if it somehow became negative (though .pow(2) handles that).
        final_factor = torch.clamp(riemannian_scaling_factor, min=EPS)
        
        # Apply the Riemannian scaling factor to the Euclidean gradient
        r_grad = final_factor * dp
        
        if not torch.isfinite(r_grad).all():
            self.logger.warning(f"Non-finite Riemannian gradient computed in egrad2rgrad for param shape {p.shape}, c={self.c}. Factor: {final_factor.mean().item() if final_factor.numel()>0 else 'N/A'}, dp_norm: {dp.norm().item() if torch.isfinite(dp).all() else 'NaN'}. Input p_norm_sq: {p_norm_sq.mean().item() if p_norm_sq.numel()>0 and torch.isfinite(p_norm_sq).all() else 'NaN'}. Projected p norm: {p_projected.norm().item() if torch.isfinite(p_projected).all() else 'NaN'}")
            # Fallback to Euclidean gradient or zero gradient if r_grad is bad
            # For safety, could return dp or torch.zeros_like(dp)
            return dp # Fallback to Euclidean gradient if rgrad calculation fails numerically
            
        return r_grad

    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad():
            w.data.uniform_(-irange, irange)
            # If c is 0, expmap0 is identity, proju is identity.
            if self.c > 0 : # Only map to hyperbolic if curvature is positive
                w.data = self.expmap0(w.data)
                w.data = self.proju(w.data) # Ensure it's on the manifold after init



def init_weights_general(m):
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)): # Consolidate Norm layers
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', False)): # Check for affine params
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.zeros_(m.bias) if m.bias is not None else None

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor: return F.softplus(param_unconstrained) + min_val

class BoundaryManifoldHyperbolic(nn.Module):
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold_c: float):
        super().__init__(); self.level_idx = level_idx; self.num_points = num_points; self.point_dim = point_dim; self.current_manifold_c = initial_manifold_c
        if num_points > 0 and point_dim > 0: self.hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim)); PoincareBall(initial_manifold_c).init_weights(self.hyperbolic_points_params, irange=1e-3); setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(initial_manifold_c))
        else: self.register_parameter('hyperbolic_points_params', None)
    def set_current_manifold_c(self, c_scalar: float): self.current_manifold_c = c_scalar; setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(c_scalar)) if self.hyperbolic_points_params is not None else None
    def get_points(self) -> Optional[torch.Tensor]: return PoincareBall(self.current_manifold_c).proju(self.hyperbolic_points_params) if self.hyperbolic_points_params is not None else None

def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis = F.normalize(axis, p=2, dim=-1); angle_half = angle_rad / 2.0; q_w = torch.cos(angle_half); q_xyz = axis * torch.sin(angle_half); return torch.cat([q_w, q_xyz], dim=-1)
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1); w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2; return torch.stack([w, x, y, z], dim=-1)
def quaternion_apply_to_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_quat = F.pad(v, (1, 0), value=0); q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1); rotated_v_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj); return rotated_v_quat[..., 1:]

class HyperbolicInterLevelTransform(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, initial_c_in: float, initial_c_out: float, transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1, use_rotation: bool = False, phi_influence_rotation_init: bool = False, level_idx_for_phi: int = 0):
        super().__init__(); self.in_dim, self.out_dim, self.transform_type = in_dim, out_dim, transform_type.lower(); self.use_rotation = use_rotation; self.rotation_module = None; self.phi_influence_rotation_init = phi_influence_rotation_init; current_logger=logging.getLogger("WuBuGAADHybridGenV01.HILT") # Updated logger
        if self.use_rotation and self.in_dim > 0:
            if self.in_dim == 4 and self.phi_influence_rotation_init: self.rot_axis_param = nn.Parameter(torch.randn(3)); self.rot_angle_unconstrained = nn.Parameter(torch.tensor(0.0)); self.phi_angle_scale = PHI**(level_idx_for_phi % 5 - 2) * (math.pi / 4); current_logger.info(f"L{level_idx_for_phi} (4D): Quat rot. Scale: {self.phi_angle_scale:.3f}")
            elif self.in_dim == 2 and self.phi_influence_rotation_init: self.rot_angle_unconstrained_2d = nn.Parameter(torch.tensor(0.0)); self.phi_angle_scale_2d = PHI**(level_idx_for_phi % 3) * (math.pi / 3); current_logger.info(f"L{level_idx_for_phi} (2D): SO(2) rot. Scale: {self.phi_angle_scale_2d:.3f}")
            else: self.rotation_module = nn.Linear(self.in_dim, self.in_dim, bias=False); nn.init.eye_(self.rotation_module.weight) if self.in_dim > 0 else None
        mlp_hidden_dim = hidden_dim if hidden_dim and hidden_dim > 0 else max(16, (in_dim + out_dim) // 2)
        if self.transform_type == 'mlp' and all(d > 0 for d in [in_dim, out_dim, mlp_hidden_dim]): self.non_rotational_map = nn.Sequential(nn.Linear(in_dim, mlp_hidden_dim), nn.LayerNorm(mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim, out_dim))
        elif self.transform_type == 'linear' and in_dim > 0 and out_dim > 0: self.non_rotational_map = nn.Linear(in_dim, out_dim)
        else: self.non_rotational_map = nn.Identity(); current_logger.info(f"L{level_idx_for_phi}: Using Identity transform as in_dim={in_dim} or out_dim={out_dim} or hidden_dim={mlp_hidden_dim} is non-positive.")
        self.apply(init_weights_general)
    def _apply_rotation(self, x_tan: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_tan is None or not self.use_rotation: return x_tan; B_maybe = x_tan.shape[0] if x_tan.dim() > 1 else 1
        if self.in_dim == 4 and self.phi_influence_rotation_init and hasattr(self, 'rot_axis_param'):
            angle = F.softplus(self.rot_angle_unconstrained) * self.phi_angle_scale; current_axis = self.rot_axis_param.to(x_tan.device).unsqueeze(0).expand(B_maybe, -1); angle_b = angle.unsqueeze(0).expand(B_maybe, 1); q_rot = quaternion_from_axis_angle(angle_b, current_axis); return quaternion_apply_to_vector(q_rot, x_tan) # Fixed: Apply actual rotation
        elif self.in_dim == 2 and self.phi_influence_rotation_init and hasattr(self, 'rot_angle_unconstrained_2d'): angle_2d = F.softplus(self.rot_angle_unconstrained_2d) * self.phi_angle_scale_2d; cos_a = torch.cos(angle_2d); sin_a = torch.sin(angle_2d); x_comp = x_tan[..., 0]; y_comp = x_tan[..., 1]; x_rot = x_comp * cos_a - y_comp * sin_a; y_rot = x_comp * sin_a + y_comp * cos_a; return torch.stack([x_rot, y_rot], dim=-1)
        return self.rotation_module(x_tan) if self.rotation_module else x_tan
    def forward(self, point_in: torch.Tensor, boundaries_in: Optional[torch.Tensor], descriptor_in: Optional[torch.Tensor], current_c_in: float, current_c_out: float, current_s_in: Optional[float]=None, current_s_out: Optional[float]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        m_in, m_out = PoincareBall(current_c_in), PoincareBall(current_c_out)
        tan_main = m_in.logmap0(point_in); tan_bound = m_in.logmap0(boundaries_in) if boundaries_in is not None else None; tan_desc = m_in.logmap0(descriptor_in) if descriptor_in is not None else None
        tan_main_rot = self._apply_rotation(tan_main); tan_bound_rot = self._apply_rotation(tan_bound); tan_desc_rot = self._apply_rotation(tan_desc)
        def apply_map_and_clamp(tan_vec): return torch.clamp(self.non_rotational_map(tan_vec), -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL) if tan_vec is not None else None
        tan_main_out_clamped = apply_map_and_clamp(tan_main_rot); tan_bound_out_clamped = apply_map_and_clamp(tan_bound_rot); tan_desc_out_clamped = apply_map_and_clamp(tan_desc_rot)
        default_out_shape = (point_in.shape[0], self.out_dim) if point_in.dim() > 1 else (self.out_dim,) # Handle batch or single vector
        expmap_main_out = m_out.expmap0(tan_main_out_clamped) if tan_main_out_clamped is not None else m_out.expmap0(torch.zeros(default_out_shape, device=point_in.device, dtype=point_in.dtype)) # Default to origin if input is None
        expmap_bound_out = m_out.expmap0(tan_bound_out_clamped) if tan_bound_out_clamped is not None else None
        expmap_desc_out = m_out.expmap0(tan_desc_out_clamped) if tan_desc_out_clamped is not None else None
        return (expmap_main_out, expmap_bound_out, expmap_desc_out)

class HyperbolicWuBuNestingLevel(nn.Module):
    def __init__(self, level_idx: int, dim: int, config: Dict, initial_curvature_val_base: float):
        super().__init__()
        self.level_idx, self.dim, self.config = level_idx, dim, config
        # Make logger an instance variable and specific to the level
        self.logger = logging.getLogger(f"WuBuGAADHybridGenV01.Level{self.level_idx}")
        current_logger = self.logger # Local alias for convenience if preferred

        self.phi_influence_curvature = config.get("phi_influence_curvature", False)
        self.initial_curvature_val = initial_curvature_val_base * (PHI**(level_idx % 4 - 1.5) if self.phi_influence_curvature else 1.0)
        current_logger.info(f"InitialC={self.initial_curvature_val:.2f}"+(f" (PhiBase {initial_curvature_val_base:.2f})" if self.phi_influence_curvature else ""))
        self.use_ld = config.get("use_level_descriptors", True)
        self.use_spread = config.get("use_level_spread", True) # This still controls if spread params are learned/used by level
        self.dropout_rate = config.get("dropout", 0.1)
        self.ld_init_scale = config.get("level_descriptor_init_scale", 1e-5)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean")
        self.min_curvature = max(EPS, config.get("curvature_min_value", EPS))
        self.min_scale = max(EPS, config.get("scale_min_value", EPS))
        self.min_spread = max(EPS, config.get("spread_min_value", EPS))

        def _init_unconstrained_param_sigmoid_scaled(target_val, min_val_range, max_val_range):
            if not (min_val_range < max_val_range):
                current_logger.warning(f"SigmoidInit: Invalid range [{min_val_range}, {max_val_range}]. Init unconstrained to 0.")
                return torch.tensor(0.0)
            clamped_target_val = torch.clamp(torch.as_tensor(target_val, dtype=torch.float), min_val_range + EPS, max_val_range - EPS).item()
            initial_sigmoid_target = (clamped_target_val - min_val_range) / (max_val_range - min_val_range)
            initial_sigmoid_target_clamped = max(EPS, min(initial_sigmoid_target, 1.0 - EPS))
            unconstrained_val = math.log(initial_sigmoid_target_clamped / (1.0 - initial_sigmoid_target_clamped))
            return torch.tensor(unconstrained_val)

        def _init_unconstrained_param_softplus(target_val, min_val):
            val_for_softplus = max(float(target_val), min_val + EPS) - min_val
            return torch.tensor(math.log(math.expm1(val_for_softplus)) if val_for_softplus > 1e-6 else math.log(val_for_softplus + EPS))

        param_init_args = {
            'learn_c': ("learnable_curvature", self.initial_curvature_val, self.min_curvature, 'log_curvature_unconstrained', 'softplus'),
            'learn_s': ("learnable_scales", "initial_scales", (MIN_WUBU_LEVEL_SCALE, MAX_WUBU_LEVEL_SCALE), 'log_scale_unconstrained', 'sigmoid_scaled'),
            'learn_spread': ("learnable_spread", "initial_spread_values", self.min_spread, 'log_spread_unconstrained', 'softplus')
        }

        for key, (learn_flag_name, init_val_name_or_direct, min_or_range_val_local, param_name, init_type) in param_init_args.items():
            if key == 'learn_spread' and not self.use_spread:
                self.register_parameter(param_name, None)
                continue
            learn_flag = config.get(learn_flag_name, True)
            default_list_val = [1.0] if key == 'learn_s' else [0.1] if key == 'learn_spread' else [self.initial_curvature_val]
            if isinstance(init_val_name_or_direct, str):
                init_list = config.get(init_val_name_or_direct, default_list_val)
                init_val = init_list[level_idx] if level_idx < len(init_list) else (init_list[-1] if init_list else default_list_val[0])
            else:
                init_val = init_val_name_or_direct

            if init_type == 'softplus':
                unconstrained_val = _init_unconstrained_param_softplus(init_val, min_or_range_val_local)
            elif init_type == 'sigmoid_scaled':
                min_r, max_r = min_or_range_val_local
                unconstrained_val = _init_unconstrained_param_sigmoid_scaled(init_val, min_r, max_r)
            else:
                raise ValueError(f"Unknown init_type: {init_type}")

            if learn_flag:
                setattr(self, param_name, nn.Parameter(unconstrained_val))
            else:
                self.register_buffer(param_name, unconstrained_val)

        if self.use_ld and self.dim > 0:
            self.level_descriptor_param = nn.Parameter(torch.Tensor(dim))
            PoincareBall(c_scalar=self.initial_curvature_val).init_weights(self.level_descriptor_param, irange=self.ld_init_scale)
            setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=self.initial_curvature_val))
        else:
            self.register_parameter('level_descriptor_param', None)

        num_bounds_list = config.get("boundary_points_per_level", [8])
        num_boundaries_val = num_bounds_list[level_idx] if level_idx < len(num_bounds_list) else (num_bounds_list[-1] if num_bounds_list else 8)
        self.boundary_manifold_module = BoundaryManifoldHyperbolic(level_idx, num_boundaries_val, dim, initial_manifold_c=self.initial_curvature_val) if self.dim > 0 else None

        # CRITICAL CHANGE HERE: self.comb_in_dim calculation adjusted
        # It no longer adds '+1' for self.use_spread, because the spread value from
        # sigma_in_scalar_tensor is not being concatenated into inputs_for_combiner.
        self.comb_in_dim = self.dim + \
                           (self.dim if self.relative_vector_aggregation not in ['none', None] else 0) + \
                           (self.dim if self.use_ld else 0)
                           # The "+ (1 if self.use_spread else 0)" was removed from here.

        comb_h_dims_cfg = config.get("tangent_input_combination_dims", [max(16, self.comb_in_dim // 2)]) if self.comb_in_dim > 0 else []
        comb_h_dims = comb_h_dims_cfg if isinstance(comb_h_dims_cfg, list) else [comb_h_dims_cfg]

        layers = []
        in_d = self.comb_in_dim # Use the corrected self.comb_in_dim
        if self.dim > 0 and self.comb_in_dim > 0: # Check if combiner is actually needed
            for h_d in comb_h_dims:
                if in_d > 0 and h_d > 0:
                    layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(self.dropout_rate)])
                    in_d = h_d
            if in_d > 0 and self.dim > 0: # Ensure final projection to self.dim
                layers.append(nn.Linear(in_d, self.dim))
                layers.append(nn.LayerNorm(self.dim))
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
        if hasattr(self, 'log_scale_unconstrained') and self.log_scale_unconstrained is not None:
            scaled_sigmoid = torch.sigmoid(self.log_scale_unconstrained)
            val_tensor = MIN_WUBU_LEVEL_SCALE + (MAX_WUBU_LEVEL_SCALE - MIN_WUBU_LEVEL_SCALE) * scaled_sigmoid
            return val_tensor.item()
        return MIN_WUBU_LEVEL_SCALE # Default if not learnable or not present

    def get_current_spread_scalar_tensor(self) -> torch.Tensor:
        if self.use_spread and hasattr(self, 'log_spread_unconstrained') and self.log_spread_unconstrained is not None:
            return get_constrained_param_val(self.log_spread_unconstrained, self.min_spread)
        # Fallback to get device/dtype if spread is not used/learnable
        ref_param = next(iter(self.parameters()), None)
        if ref_param is None and isinstance(self.tangent_combiner, nn.Sequential) and list(self.tangent_combiner.parameters()):
             ref_param = next(iter(self.tangent_combiner.parameters()), None)

        ref_device = ref_param.device if ref_param is not None else torch.device('cpu')
        ref_dtype = ref_param.dtype if ref_param is not None else torch.float
        return torch.tensor(self.min_spread, device=ref_device, dtype=ref_dtype)

    def forward(self, point_in_hyperbolic: torch.Tensor,
                relative_vectors_tangent_in: Optional[torch.Tensor],
                descriptor_point_in_hyperbolic: Optional[torch.Tensor],
                sigma_in_scalar_tensor: Optional[torch.Tensor] = None # This input is present but not used by tangent_combiner
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:

        if point_in_hyperbolic.dim() != 2:
            raise ValueError(f"WuBuLevel forward expects 2D input (B', D), got {point_in_hyperbolic.dim()}D shape {point_in_hyperbolic.shape}")

        B_prime, D_in = point_in_hyperbolic.shape
        dev = point_in_hyperbolic.device
        ref_param_for_dtype = next(iter(self.parameters()), None)
        dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else point_in_hyperbolic.dtype

        if self.dim == 0: # Handle zero-dimension case explicitly
            dummy_out_shape = (B_prime, 0)
            dummy_dtype_dev = {'device': dev, 'dtype': dtype_to_use}
            current_spread_tensor = self.get_current_spread_scalar_tensor().to(dtype_to_use)
            return (torch.zeros(dummy_out_shape, **dummy_dtype_dev),
                    torch.zeros(dummy_out_shape, **dummy_dtype_dev),
                    None, None, current_spread_tensor)

        current_c_val = self.get_current_curvature_scalar()
        current_s_val = self.get_current_scale_scalar()
        # This is the spread value *produced* by this level
        current_sigma_out_tensor = self.get_current_spread_scalar_tensor()
        current_manifold_obj = PoincareBall(c_scalar=current_c_val)

        if self.level_descriptor_param is not None and hasattr(self.level_descriptor_param, 'manifold'):
            setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=current_c_val))
        if self.boundary_manifold_module is not None:
            self.boundary_manifold_module.set_current_manifold_c(current_c_val)

        point_in_proj = current_manifold_obj.proju(point_in_hyperbolic.to(dtype_to_use))
        tan_main_component = current_manifold_obj.logmap0(point_in_proj)
        tan_rel_component = torch.zeros_like(tan_main_component) # Default
        ld_point_self_hyperbolic = None

        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None]:
            if relative_vectors_tangent_in.shape[0] != B_prime:
                raise ValueError(f"RelVec shape mismatch: {relative_vectors_tangent_in.shape[0]} != B' {B_prime}")
            tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)

        if self.use_ld and self.level_descriptor_param is not None:
            ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use))

        tan_desc_prev_level_component = torch.zeros_like(tan_main_component) # Default
        if descriptor_point_in_hyperbolic is not None and self.use_ld:
            if descriptor_point_in_hyperbolic.shape[0] != B_prime:
                raise ValueError(f"DescIn shape mismatch: {descriptor_point_in_hyperbolic.shape[0]} != B' {B_prime}")
            desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use))
            tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)

        inputs_for_combiner = [tan_main_component]
        if self.relative_vector_aggregation not in ['none', None]:
            inputs_for_combiner.append(tan_rel_component)
        if self.use_ld:
            inputs_for_combiner.append(tan_desc_prev_level_component)

        # The sigma_in_scalar_tensor is NOT added to inputs_for_combiner here,
        # matching the change in self.comb_in_dim calculation.

        if not inputs_for_combiner:
            # This case should ideally not happen if self.dim > 0, as tan_main_component is always added.
            # If self.dim == 0, it's handled at the start of forward.
            # If it somehow happens with self.dim > 0, create zeros of self.dim.
             combined_tangent_features = torch.zeros(B_prime, self.dim, device=dev, dtype=dtype_to_use)
        elif len(inputs_for_combiner) > 1:
             combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1)
        else: # len(inputs_for_combiner) == 1
             combined_tangent_features = inputs_for_combiner[0]

        # Universal padding/truncation logic using self.comb_in_dim
        if self.comb_in_dim > 0: # Only if combiner expects non-zero input
            if combined_tangent_features.shape[-1] < self.comb_in_dim:
                padding_size = self.comb_in_dim - combined_tangent_features.shape[-1]
                if padding_size > 0: # Ensure padding_size is positive
                    combined_tangent_features = F.pad(combined_tangent_features, (0, padding_size))
            elif combined_tangent_features.shape[-1] > self.comb_in_dim:
                self.logger.warning(
                    f"Tangent Combiner input dim {combined_tangent_features.shape[-1]} > expected {self.comb_in_dim}. Truncating."
                )
                combined_tangent_features = combined_tangent_features[..., :self.comb_in_dim]
        elif combined_tangent_features.shape[-1] > 0 and self.comb_in_dim == 0:
            # This case means tangent_combiner is nn.Identity expecting 0-dim input,
            # but combined_tangent_features has features. This is an anomaly.
            B_prime_local = combined_tangent_features.shape[0] # Use local B_prime from current tensor
            self.logger.warning(
                f"Tangent Combiner expects 0-dim input (self.comb_in_dim=0), but got {combined_tangent_features.shape[-1]} features. Forcing to (Batch={B_prime_local}, 0)."
            )
            combined_tangent_features = torch.empty(B_prime_local, 0, device=combined_tangent_features.device, dtype=combined_tangent_features.dtype)
        # If self.comb_in_dim is 0 and combined_tangent_features.shape[-1] is also 0, it's fine.

        v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features)
        v_final_for_expmap_unclamped = v_combined_tangent_processed * current_s_val

        if self.use_flow and self.tangent_flow_module is not None:
            flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val
            v_final_for_expmap_unclamped = v_final_for_expmap_unclamped + flow_effect

        scaled_output_tangent_for_expmap = torch.clamp(v_final_for_expmap_unclamped, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        point_this_level_out_hyperbolic = current_manifold_obj.expmap0(scaled_output_tangent_for_expmap)
        tangent_out_for_aggregation = v_combined_tangent_processed.to(dtype_to_use) # Use the processed tangent vector

        boundary_points_this_level_hyperbolic = None
        if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None:
            boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype=dtype_to_use, device=dev)

        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None: # This is the level's own descriptor
            if ld_point_self_hyperbolic.dim() == 1: # Expand if it's a single vector
                descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B_prime, -1).to(dtype=dtype_to_use)
            else: # Already batched (should not happen for self.level_descriptor_param)
                descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.to(dtype=dtype_to_use)


        # Match output dtype to input hyperbolic point dtype for consistency
        output_dtype = point_in_hyperbolic.dtype
        return (point_this_level_out_hyperbolic.to(dtype=output_dtype),
                tangent_out_for_aggregation.to(dtype=output_dtype),
                descriptor_point_out_for_transform_hyperbolic.to(dtype=output_dtype) if descriptor_point_out_for_transform_hyperbolic is not None else None,
                boundary_points_this_level_hyperbolic.to(dtype=output_dtype) if boundary_points_this_level_hyperbolic is not None else None,
                current_sigma_out_tensor.to(dtype=output_dtype)) # This is the spread this level *produces*



class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_tangent_dim: int, output_tangent_dim: int, config: Dict):
        super().__init__(); current_logger=logging.getLogger("WuBuGAADHybridGenV01.WuBuModel") # Updated logger
        self.input_tangent_dim, self.output_tangent_dim, self.config = input_tangent_dim, output_tangent_dim, config; self.num_levels = config.get("num_levels", 3); assert self.num_levels >= 0; self.hyperbolic_dims_list = config.get("hyperbolic_dims", []); self.initial_curvatures_list = config.get("initial_curvatures", []); self.dropout_val = config.get("dropout", 0.1); self.relative_vector_aggregation_mode = config.get("relative_vector_aggregation", "mean"); self.aggregation_method_mode = config.get("aggregation_method", "concat_tangent"); assert self.aggregation_method_mode == "concat_tangent"; self.use_rotation_in_transform_flag = config.get("use_rotation_in_transform", False); self.phi_influence_rotation_init = config.get("phi_influence_rotation_init", False)
        first_level_dim = self.hyperbolic_dims_list[0] if self.num_levels > 0 and self.hyperbolic_dims_list else 0
        self.input_tangent_projection = nn.Linear(input_tangent_dim, first_level_dim) if input_tangent_dim > 0 and first_level_dim > 0 and input_tangent_dim != first_level_dim else nn.Identity()
        self.input_tangent_layernorm = nn.LayerNorm(first_level_dim) if first_level_dim > 0 else nn.Identity()
        self.levels_modulelist = nn.ModuleList(); self.transforms_modulelist = nn.ModuleList()
        if self.num_levels > 0:
            for i in range(self.num_levels):
                if i < len(self.hyperbolic_dims_list) and i < len(self.initial_curvatures_list): # Check lengths
                    self.levels_modulelist.append(HyperbolicWuBuNestingLevel(i, self.hyperbolic_dims_list[i], self.config, self.initial_curvatures_list[i]))
                else: current_logger.error(f"Level {i} skipped: Config lists too short (dims:{len(self.hyperbolic_dims_list)}, curves:{len(self.initial_curvatures_list)})"); break # Stop adding levels if config is short
            num_transforms_needed = max(0, len(self.levels_modulelist) - 1) # Based on actual levels added
            if num_transforms_needed > 0:
                transform_types_list = config.get("transform_types", ["linear"] * num_transforms_needed); transform_hidden_dims_list = config.get("transform_hidden_dims", [None] * num_transforms_needed)
                for i in range(num_transforms_needed):
                    if i + 1 < len(self.levels_modulelist) and \
                       i + 1 < len(self.hyperbolic_dims_list) and \
                       i + 1 < len(self.initial_curvatures_list): # Ensure next level configs exist
                        self.transforms_modulelist.append(HyperbolicInterLevelTransform(self.hyperbolic_dims_list[i], self.hyperbolic_dims_list[i+1], self.initial_curvatures_list[i], self.initial_curvatures_list[i+1], transform_types_list[i] if i < len(transform_types_list) else "linear", transform_hidden_dims_list[i] if i < len(transform_hidden_dims_list) else None, self.dropout_val, self.use_rotation_in_transform_flag, self.phi_influence_rotation_init, level_idx_for_phi=i))
                    else: current_logger.warning(f"Skipping transform {i} to {i+1} due to insufficient config/levels for next level.")
        actual_output_dims_from_levels = [d for d_idx, d in enumerate(self.hyperbolic_dims_list[:len(self.levels_modulelist)]) if d > 0]; aggregated_tangent_dim_val = sum(actual_output_dims_from_levels) if actual_output_dims_from_levels else input_tangent_dim
        self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, output_tangent_dim) if aggregated_tangent_dim_val > 0 and output_tangent_dim > 0 and aggregated_tangent_dim_val != output_tangent_dim else nn.Identity()
        self.apply(init_weights_general); param_count = sum(p.numel() for p in self.parameters() if p.requires_grad); current_logger.info(f"Levels: {len(self.levels_modulelist)}. Params: {param_count:,}. InDim {input_tangent_dim}, AggDim {aggregated_tangent_dim_val}, OutDim {output_tangent_dim}")
    def forward(self, x_initial_tangent_in: torch.Tensor) -> torch.Tensor:
        input_dim = x_initial_tangent_in.dim(); B_orig, S_orig, D_orig = -1, -1, -1; B_prime = -1
        if input_dim == 3: B_orig, S_orig, D_orig = x_initial_tangent_in.shape; x_proc = x_initial_tangent_in.reshape(B_orig * S_orig, D_orig); B_prime_for_levels = B_orig * S_orig
        elif input_dim == 2: B_prime, D_orig = x_initial_tangent_in.shape; x_proc = x_initial_tangent_in; B_prime_for_levels = B_prime
        else: raise ValueError(f"WuBuModel expects 2D/3D input, got {input_dim}D")
        if D_orig != self.input_tangent_dim: raise ValueError(f"Input feature dim {D_orig} != model input_tangent_dim {self.input_tangent_dim}")
        if self.num_levels == 0 or not self.levels_modulelist: return self.output_tangent_projection(x_proc).reshape(B_orig, S_orig, -1) if input_dim==3 else self.output_tangent_projection(x_proc)
        dev = x_proc.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else x_proc.dtype; x_proc = x_proc.to(dtype_to_use)
        current_tangent_projected = self.input_tangent_projection(x_proc); current_tangent_for_level0 = self.input_tangent_layernorm(current_tangent_projected)
        level0_module = self.levels_modulelist[0]; c0_val = level0_module.get_current_curvature_scalar(); m0_obj = PoincareBall(c_scalar=c0_val)
        current_point_repr_hyperbolic = m0_obj.expmap0(current_tangent_for_level0) if self.hyperbolic_dims_list[0] > 0 else torch.empty(B_prime_for_levels, 0, device=dev, dtype=dtype_to_use)
        level_tangent_outputs_for_aggregation = []; aggregated_relative_vectors_from_prev_transform = None; descriptor_from_prev_transform_hyperbolic = None; sigma_from_prev_level_tensor = torch.tensor(0.0, device=dev, dtype=dtype_to_use)
        for i, level_module in enumerate(self.levels_modulelist):
            # --- Pass sigma_from_prev_level_tensor instead of None for sigma_in ---
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation, descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic, sigma_out_of_level_tensor) = level_module(current_point_repr_hyperbolic, aggregated_relative_vectors_from_prev_transform, descriptor_from_prev_transform_hyperbolic, sigma_from_prev_level_tensor)
            if self.hyperbolic_dims_list[i] > 0: level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)
            if i < len(self.levels_modulelist) - 1: # Check if there's a next level
                if i >= len(self.transforms_modulelist): logging.getLogger("WuBuGAADHybridGenV01.WuBuModel").warning(f"Missing transform L{i}->L{i+1}. Stop."); break
                transform_module = self.transforms_modulelist[i]; next_level_module = self.levels_modulelist[i+1]
                c_in_for_transform = level_module.get_current_curvature_scalar(); c_out_for_transform = next_level_module.get_current_curvature_scalar()
                (point_transformed_to_next_level_hyperbolic, boundaries_transformed_to_next_level_hyperbolic, descriptor_transformed_to_next_level_hyperbolic) = transform_module(point_out_of_level_hyperbolic, boundary_points_of_level_hyperbolic, descriptor_generated_by_level_hyperbolic, c_in_for_transform, c_out_for_transform)
                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic; descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic; sigma_from_prev_level_tensor = sigma_out_of_level_tensor; aggregated_relative_vectors_from_prev_transform = None
                if boundaries_transformed_to_next_level_hyperbolic is not None and self.relative_vector_aggregation_mode not in ['none', None] and self.hyperbolic_dims_list[i+1] > 0 and current_point_repr_hyperbolic.shape[-1] > 0 :
                    manifold_next_level_obj = PoincareBall(c_scalar=c_out_for_transform); tan_main_next_level = manifold_next_level_obj.logmap0(current_point_repr_hyperbolic); tan_bounds_next_level = manifold_next_level_obj.logmap0(boundaries_transformed_to_next_level_hyperbolic); tan_bounds_next_level_expanded = tan_bounds_next_level.unsqueeze(0).expand(B_prime_for_levels, -1, -1); relative_tangent_vectors = tan_main_next_level.unsqueeze(1) - tan_bounds_next_level_expanded; agg_mode = self.relative_vector_aggregation_mode
                    if agg_mode == "mean": agg_rel_vec = torch.mean(relative_tangent_vectors, dim=1)
                    elif agg_mode == "sum": agg_rel_vec = torch.sum(relative_tangent_vectors, dim=1)
                    elif agg_mode == "max_norm": norms = torch.norm(relative_tangent_vectors, p=2, dim=-1); best_idx = torch.argmax(norms, dim=1, keepdim=True); best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, relative_tangent_vectors.shape[-1]); agg_rel_vec = torch.gather(relative_tangent_vectors, 1, best_idx_expanded).squeeze(1)
                    else: agg_rel_vec = None
                    aggregated_relative_vectors_from_prev_transform = torch.zeros_like(tan_main_next_level) if agg_rel_vec is not None and not torch.isfinite(agg_rel_vec).all() else agg_rel_vec
        compatible_tangent_outputs = [t_val.to(dtype_to_use) for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation) if t_val is not None and t_idx < len(self.hyperbolic_dims_list) and self.hyperbolic_dims_list[t_idx] > 0 and torch.isfinite(t_val).all()]
        if not compatible_tangent_outputs:
            out_zeros = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use)
            if input_dim == 3: return out_zeros.reshape(B_orig, S_orig, self.output_tangent_dim)
            return out_zeros
        aggregated_tangent_final = torch.cat(compatible_tangent_outputs, dim=-1); final_output_flat = self.output_tangent_projection(aggregated_tangent_final); final_output_flat = torch.nan_to_num(final_output_flat, nan=0.0) if not torch.isfinite(final_output_flat).all() else final_output_flat
        return final_output_flat.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim == 3 else final_output_flat

class GradientStats:
    def __init__(self): self.reset()
    def reset(self): self.total_params_updated=0; self.total_finite_grads_processed=0; self.total_non_finite_grads_encountered=0; self.params_skipped_due_non_finite_grad=0; self.max_grad_norm_observed=0.; self.step_summary={}
    def record_param_grad(self, grad_is_finite: bool, original_norm_if_finite: Optional[float] = None):
        if grad_is_finite: self.total_finite_grads_processed += 1; self.max_grad_norm_observed = max(self.max_grad_norm_observed, original_norm_if_finite if original_norm_if_finite is not None else 0.0)
        else: self.total_non_finite_grads_encountered += 1; self.params_skipped_due_non_finite_grad += 1
    def finalize_step_stats(self, num_params_in_optimizer_step: int): self.total_params_updated=num_params_in_optimizer_step-self.params_skipped_due_non_finite_grad; self.step_summary={"params_in_step":num_params_in_optimizer_step, "params_updated":self.total_params_updated, "params_skipped_non_finite_grad":self.params_skipped_due_non_finite_grad, "initial_finite_grads":self.total_finite_grads_processed, "initial_non_finite_grads":self.total_non_finite_grads_encountered, "max_finite_grad_norm_observed":self.max_grad_norm_observed}
    def get_step_summary_for_logging(self) -> dict: return self.step_summary.copy()

class HAKMEMQController:
    """
    Implements a Q-learning controller inspired by HAKMEM item 175 concepts
    for dynamically adjusting optimizer hyperparameters (learning rate scale, momentum scale).
    Includes enhanced heuristic pruning and GAN balance term in reward calculation.
    """
    def __init__(self,
                 learning_rate: float = 0.01,
                 discount: float = 0.95,
                 epsilon: float = 0.2,
                 epsilon_decay: float = 0.9998,
                 min_epsilon: float = 0.01,
                 lr_scale_options: Optional[List[float]] = None,
                 momentum_scale_options: Optional[List[float]] = None,
                 max_q_table_size: int = 10000):
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
            'lr_scale': np.array(lr_scale_options if lr_scale_options else [0.9, 0.95, 1.0, 1.05, 1.1]),
            'momentum_scale': np.array(momentum_scale_options if momentum_scale_options else [0.95, 0.98, 1.0, 1.01, 1.02])
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
        self.q_table_last_access_time: Dict[Tuple, float] = {}

        self.flow_coefficient = 0.05
        self.oscillation_penalty = 0.15
        self.stability_reward_bonus = 0.05
        self.gan_balance_reward_scale = 0.25

        self.logger = logging.getLogger("WuBuGAADHybridGenV01.QCtrl")

    def get_state(self, lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float]) -> Optional[Tuple]:
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm):
            # self.logger.debug("Invalid input for get_state (NaN/None loss or grad_norm). Returning None.")
            return None
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm); self.lr_window.append(lr); self.momentum_window.append(momentum)
        if len(self.loss_window) < 5 or len(self.grad_norm_window) < 5: return None
        loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2, 2, 2, 1, 0
        try:
            loss_arr = np.array(list(self.loss_window)[-10:]); median_loss = np.median(loss_arr)
            if len(loss_arr) >= 3 and len(np.unique(loss_arr)) > 1: slope_loss = np.polyfit(np.arange(len(loss_arr)), loss_arr, 1)[0]; relative_slope = slope_loss / (abs(median_loss) + EPS); loss_trend_bin = np.digitize(relative_slope, bins=[-0.05, -0.005, 0.005, 0.05]).item()
            else: loss_trend_bin = 2
            grad_norm_level_bin = np.digitize(np.median(list(self.grad_norm_window)), bins=[0.1, 0.5, 1.5, 5.0]).item()
            lr_level_bin = np.digitize(lr / 1e-4, bins=[0.5, 2.0, 10.0, 50.0]).item()
            momentum_level_bin = np.digitize(momentum, bins=[0.85, 0.92, 0.97]).item()
            if len(self.performance_window) >= 5:
                recent_rewards = np.sign([r for r in list(self.performance_window)[-5:] if r != 0])
                if len(recent_rewards) >= 2: num_sign_changes = np.sum(np.abs(np.diff(recent_rewards))) / 2.0; self.oscillation_counter = min(self.oscillation_counter + 1, 5) if num_sign_changes >= 2 else max(0, self.oscillation_counter - 1)
                else: self.oscillation_counter = max(0, self.oscillation_counter - 1)
            oscillation_bin = 1 if self.oscillation_counter >= 3 else 0
        except Exception as e: self.logger.warning(f"Q State calculation error: {e}. Returning None.", exc_info=False); return None
        state_tuple = (loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin)
        if state_tuple is not None:
             current_time = time.time(); self.q_table_access_count[state_tuple] += 1; self.q_table_last_access_time[state_tuple] = current_time
             if state_tuple not in self.q_table: self.q_table[state_tuple] = {p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}; self.q_table_creation_time[state_tuple] = current_time; self._manage_q_table_size()
        return state_tuple

    def choose_action(self, state: Optional[Tuple]) -> Dict[str, float]:
        default_action = {'lr_scale': 1.0, 'momentum_scale': 1.0}
        if state is None: return default_action
        if state not in self.q_table: # Should be initialized by get_state, but as a safeguard
            current_time = time.time(); self.q_table[state] = {p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}; self.q_table_creation_time[state] = current_time; self.q_table_access_count[state] = 1; self.q_table_last_access_time[state] = current_time; self._manage_q_table_size(); self.logger.debug(f"Init new state in Q-table (choose_action): {state}")
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay); chosen_actions = {}
        for p_type, q_vals in self.q_table[state].items():
            action_space = self.action_ranges[p_type]
            if random.random() < self.epsilon: chosen_idx = random.randrange(len(action_space))
            else:
                finite_q_mask = np.isfinite(q_vals)
                if np.any(finite_q_mask): best_q_val = np.max(q_vals[finite_q_mask]); best_indices = np.where(finite_q_mask & np.isclose(q_vals, best_q_val))[0]; chosen_idx = random.choice(best_indices) if len(best_indices) > 0 else random.randrange(len(action_space))
                else: chosen_idx = random.randrange(len(action_space)); self.logger.warning(f"All Q-vals non-finite for {state}, type {p_type}. Random action.")
            chosen_actions[p_type] = float(action_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy()); return chosen_actions

    def update(self, state: Optional[Tuple], action: Optional[Dict[str, float]], reward: float, next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None or state not in self.q_table: return
        if next_state not in self.q_table: # Should be initialized by get_state, but as a safeguard
            current_time = time.time(); self.q_table[next_state] = {p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}; self.q_table_creation_time[next_state] = current_time; self.q_table_access_count[next_state] = 0; self.q_table_last_access_time[next_state] = current_time; self._manage_q_table_size(); self.logger.debug(f"Init new next_state in Q-table (update): {next_state}")
        for p_type, chosen_val in action.items():
            if p_type not in self.q_table[state]: continue
            action_idx_arr = np.where(np.isclose(self.action_ranges[p_type], chosen_val))[0]
            if len(action_idx_arr) == 0: continue
            action_idx = action_idx_arr[0]; current_q = self.q_table[state][p_type][action_idx]; next_q_vals = self.q_table[next_state][p_type]; finite_next_q = next_q_vals[np.isfinite(next_q_vals)]; max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0; max_future_q = 0.0 if not np.isfinite(max_future_q) else max_future_q
            td_target = reward + self.gamma * max_future_q; td_error = td_target - current_q; adaptive_alpha = min(0.5, max(0.001, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error) * 0.5)))); new_q = current_q + adaptive_alpha * td_error
            if np.isfinite(new_q): self.q_table[state][p_type][action_idx] = np.clip(new_q, -1e4, 1e4)
            else: self.logger.warning(f"Non-finite Q-val @ update for {state}, type {p_type}, idx {action_idx}. Reset Q=0."); self.q_table[state][p_type][action_idx] = 0.0

    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            num_to_prune = len(self.q_table) - self.max_q_table_size;
            can_smart_prune = all([self.q_table_access_count, self.q_table_last_access_time, len(self.q_table_access_count) >= len(self.q_table) // 2, len(self.q_table_last_access_time) >= len(self.q_table) // 2])
            if can_smart_prune:
                try: sorted_states = sorted(self.q_table.keys(), key=lambda s: (self.q_table_access_count.get(s, 0), self.q_table_last_access_time.get(s, 0.0))); to_remove = sorted_states[:num_to_prune]
                except Exception as e_sort: self.logger.warning(f"Smart prune sort fail: {e_sort}. Random prune."); to_remove = random.sample(list(self.q_table.keys()), num_to_prune)
            else: to_remove = random.sample(list(self.q_table.keys()), num_to_prune)
            pruned_count = 0
            for s_rm in to_remove:
                if s_rm in self.q_table: self.q_table.pop(s_rm, None); self.q_table_access_count.pop(s_rm, None); self.q_table_creation_time.pop(s_rm, None); self.q_table_last_access_time.pop(s_rm, None); pruned_count += 1
            if pruned_count > 0: self.logger.info(f"Pruned {pruned_count} Q-table entries. New size: {len(self.q_table)}")

    def get_info(self) -> Dict:
        q_mem_mb = 0.0
        try: q_mem_mb = sum(sys.getsizeof(s) + sum(a.nbytes + sys.getsizeof(k) for k, a in v.items()) for s, v in self.q_table.items()) / (1024**2) if self.q_table else 0.0
        except Exception: q_mem_mb = -1.0
        avg_perf_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.0
        return {"epsilon": round(self.epsilon, 4), "q_table_size": len(self.q_table), "q_table_mem_mb_approx": round(q_mem_mb, 2), "last_action": self.prev_actions_log[-1] if self.prev_actions_log else None, f"avg_reward_last_{self.performance_window.maxlen}": round(avg_perf_reward, 3), "stable_steps": self.stable_steps, "oscillation_counter": self.oscillation_counter}

    def compute_reward(self, current_loss_self: Optional[float], prev_loss_self: Optional[float], current_loss_opponent: Optional[float], grad_norm_self: Optional[float], is_generator_q: bool) -> float:
        if current_loss_self is None or prev_loss_self is None or current_loss_opponent is None or grad_norm_self is None or not np.isfinite(current_loss_self) or not np.isfinite(prev_loss_self) or not np.isfinite(current_loss_opponent) or not np.isfinite(grad_norm_self):
            return 0.0
        median_loss_hist = np.median(list(self.loss_window)[:-1]) if len(self.loss_window) > 1 else prev_loss_self
        loss_improvement = prev_loss_self - current_loss_self
        base_reward = np.tanh(loss_improvement / (abs(median_loss_hist) + EPS) * 5.0)
        gan_balance_reward = 0.0; target_disc_loss = 0.693
        disc_loss = current_loss_opponent if is_generator_q else current_loss_self
        disc_loss_deviation = abs(disc_loss - target_disc_loss); balance_scale = self.gan_balance_reward_scale

        if is_generator_q:
            gan_balance_reward = balance_scale * math.exp(-5.0 * (disc_loss_deviation**2))
            if disc_loss < 0.3: gan_balance_reward -= balance_scale * ((0.3 - disc_loss) / 0.3) * 1.0
            # No penalty for high D_loss for G in balance term; G's adv loss handles pushing D loss high.
        else: # Discriminator's reward
            if disc_loss < 0.2: gan_balance_reward = -balance_scale * ((0.2 - disc_loss) / 0.2)
            elif disc_loss > 1.5: gan_balance_reward = -balance_scale * min(1.0, (disc_loss - 1.5) / 1.0)
            else: gan_balance_reward = balance_scale * 0.25 * math.exp(-3.0 * (disc_loss_deviation**2))

        grad_penalty = -0.1 * min(1.0, max(0.0, (grad_norm_self - 5.0) / 10.0)) if grad_norm_self > 5.0 else 0.0
        osc_penalty = -self.oscillation_penalty if self.oscillation_counter >= 3 else 0.0
        current_perf_reward = base_reward + gan_balance_reward + grad_penalty + osc_penalty
        self.performance_window.append(current_perf_reward)
        self.stable_steps = self.stable_steps + 1 if current_perf_reward > 0.01 else 0
        stab_bonus = min(0.15, self.stability_reward_bonus * math.log1p(self.stable_steps / 5.0)) if current_perf_reward > 0.01 else 0.0
        final_reward = current_perf_reward + stab_bonus

        if self.logger.isEnabledFor(logging.DEBUG):
            role = 'G' if is_generator_q else 'D'
            self.logger.debug(
                f"QCtrl Rew ({role}): Base={base_reward:.2f}, Bal={gan_balance_reward:.2f} (D_Loss={disc_loss:.3f}), "
                f"GradP={grad_penalty:.2f}, OscP={osc_penalty:.2f}, StabB={stab_bonus:.2f} --> TOTAL={final_reward:.3f}"
            )
        return float(np.clip(final_reward, -1.0, 1.0))



class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm_risgd=1.0, q_learning_config:Optional[Dict]=None):
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # Initialize Q-controller if config provided
        self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_learning_config) if isinstance(q_learning_config, dict) else None
        self.logger=logging.getLogger("WuBuGAADHybridGenV01.RiSGD")
        if not self.logger.hasHandlers(): # Basic config if no handlers exist
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
        self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled.")
        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count = 0
        self.grad_stats = GradientStats()
        # Initialize state for all parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state.setdefault(p, {})

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        self.grad_stats.reset() # Reset gradient stats on zero_grad

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None # Note: closure loss is not used for Q-update here

        # --- Q-Controller Action Application ---
        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action
            for group in self.param_groups:
                group.setdefault('base_lr', group['lr'])
                group.setdefault('base_momentum', group['momentum'])
                group['lr'] = float(np.clip(group['base_lr'] * q_action.get('lr_scale', 1.0), 1e-8, 1.0))
                group['momentum'] = float(np.clip(group['base_momentum'] * q_action.get('momentum_scale', 1.0), 0.1, 0.999))
        # --- End Q-Controller Action Application ---

        num_params_total = 0
        for group in self.param_groups:
            lr, mom, wd = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                num_params_total += 1
                grad = p.grad.data
                finite_grad = torch.isfinite(grad).all()
                norm_val = grad.norm().item() if finite_grad else None
                self.grad_stats.record_param_grad(finite_grad, norm_val)

                if not finite_grad:
                    self.state[p].pop('momentum_buffer', None); continue

                if norm_val is not None and norm_val > self.max_grad_norm_risgd and self.max_grad_norm_risgd > 0:
                    grad.mul_(self.max_grad_norm_risgd / (norm_val + EPS))

                state = self.state[p]
                manifold: Optional[Manifold] = getattr(p, 'manifold', None)

                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    p_proj = manifold.proju(p.data)
                    try:
                        r_grad = manifold.egrad2rgrad(p_proj, grad)
                    except Exception as e_egrad:
                        self.logger.error(f"RiGrad Err P:{p.shape}: {e_egrad}. Skip."); self.grad_stats.params_skipped_due_non_finite_grad +=1; state.pop('momentum_buffer', None); continue
                    if not torch.isfinite(r_grad).all():
                        self.logger.warning(f"NonFinite RiGrad P:{p.shape}. Skip."); self.grad_stats.params_skipped_due_non_finite_grad += 1; state.pop('momentum_buffer', None); continue

                    update_vec = r_grad
                    if wd != 0:
                        try: # Added try for weight decay section
                            log_p = manifold.logmap0(p_proj)
                            if torch.isfinite(log_p).all():
                                update_vec = update_vec.add(log_p, alpha=wd)
                            else:
                                self.logger.warning(f"NonFinite Logmap WD P:{p.shape}. Skip WD.")
                        except Exception as e_wd: # Added except for weight decay
                            self.logger.warning(f"Logmap WD Err P:{p.shape}: {e_wd}. Skip WD.")

                    buf = state.setdefault('momentum_buffer', torch.zeros_like(update_vec))
                    buf.mul_(mom).add_(update_vec)
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"NonFinite Momentum P:{p.shape}. Reset buf."); buf.zero_()

                    try:
                        expmap_arg = buf.mul(-lr)
                        if not torch.isfinite(expmap_arg).all():
                            self.logger.warning(f"NonFinite Expmap Arg P:{p.shape}. Reset mom."); state.get('momentum_buffer',torch.zeros(0)).zero_(); continue
                        new_p_candidate = manifold.expmap(p_proj, expmap_arg)
                        if not torch.isfinite(new_p_candidate).all():
                            self.logger.warning(f"NonFinite Expmap Result P:{p.shape}. Fallback."); new_p_candidate = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0)); state.get('momentum_buffer',torch.zeros(0)).zero_()
                        p.data = manifold.proju(new_p_candidate)
                        if not torch.isfinite(p.data).all():
                            self.logger.error(f"Param NonFinite PostUpdate P:{p.shape}. Reset origin."); p.data = manifold.expmap0(torch.zeros_like(p.data)); state.get('momentum_buffer',torch.zeros(0)).zero_()
                    except Exception as e_hyp_update:
                        self.logger.error(f"Hyp Update Err P:{p.shape}: {e_hyp_update}. Reset mom."); state.get('momentum_buffer',torch.zeros(0)).zero_()
                else:  # Euclidean parameter update
                    d_p = grad.clone();
                    if wd != 0: d_p.add_(p.data, alpha=wd)
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(p.data))
                    buf.mul_(mom).add_(d_p)
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"NonFinite Euc Mom P:{p.shape}. Reset."); buf.zero_()
                    p.data.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all():
                        self.logger.warning(f"Euc Param NonFinite P:{p.shape}. NaN & Reset mom."); p.data = torch.nan_to_num(p.data, nan=0.0); state.get('momentum_buffer',torch.zeros(0)).zero_()

        self.grad_stats.finalize_step_stats(num_params_total)
        self._step_count += 1
        return loss

    def get_q_controller_info(self) -> Dict:
        return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}

    def get_gradient_stats_summary(self) -> Dict:
        return self.grad_stats.get_step_summary_for_logging()

# =====================================================================
# GAAD Components (Unchanged)
# =====================================================================
def golden_subdivide_rect_fixed_n(frame_dims:Tuple[int,int], num_regions_target:int, device='cpu', dtype=torch.float, min_size_px=5) -> torch.Tensor:
    W, H = frame_dims; all_rects = [[0,0,W,H]]; rect_queue = deque([(0,0,W,H,0)])
    while rect_queue and len(all_rects) < num_regions_target * 3:
        x_off, y_off, w_curr, h_curr, depth = rect_queue.popleft()
        if min(w_curr, h_curr) < min_size_px or depth > 6 : continue
        is_landscape = w_curr > h_curr + EPS; is_portrait = h_curr > w_curr + EPS
        if is_landscape:
            cut_w = w_curr / PHI; r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px and h_curr >= min_size_px: all_rects.append([x_off, y_off, x_off + r1_w, y_off + h_curr]); rect_queue.append((x_off, y_off, r1_w, h_curr, depth + 1))
            if r2_w >= min_size_px and h_curr >= min_size_px: all_rects.append([x_off + r1_w, y_off, x_off + r1_w + r2_w, y_off + h_curr]); rect_queue.append((x_off + r1_w, y_off, r2_w, h_curr, depth + 1))
        elif is_portrait:
            cut_h = h_curr / PHI; r1_h, r2_h = cut_h, h_curr - cut_h
            if w_curr >= min_size_px and r1_h >= min_size_px: all_rects.append([x_off, y_off, x_off + w_curr, y_off + r1_h]); rect_queue.append((x_off, y_off, w_curr, r1_h, depth + 1))
            if w_curr >= min_size_px and r2_h >= min_size_px: all_rects.append([x_off, y_off + r1_h, x_off + w_curr, y_off + r1_h + r2_h]); rect_queue.append((x_off, y_off + r1_h, w_curr, r2_h, depth + 1))
        elif abs(w_curr - h_curr) < EPS and w_curr > min_size_px * PHI :
            cut_w = w_curr / PHI; r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px and h_curr >= min_size_px: all_rects.append([x_off, y_off, x_off + r1_w, y_off + h_curr]); rect_queue.append((x_off, y_off, r1_w, h_curr, depth + 1))
            if r2_w >= min_size_px and h_curr >= min_size_px: all_rects.append([x_off + r1_w, y_off, x_off + r1_w + r2_w, y_off + h_curr]); rect_queue.append((x_off + r1_w, y_off, r2_w, h_curr, depth + 1))
    unique_valid_rects_tensors = []; seen_hashes = set()
    for r_coords in all_rects:
        if r_coords[0] >= r_coords[2] - EPS or r_coords[1] >= r_coords[3] - EPS: continue
        r_tensor = torch.tensor(r_coords, dtype=dtype, device=device); r_hashable = tuple(round(c, 3) for c in r_coords)
        if r_hashable not in seen_hashes: unique_valid_rects_tensors.append(r_tensor); seen_hashes.add(r_hashable)
    unique_valid_rects_tensors.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True); selected_rects = unique_valid_rects_tensors[:num_regions_target]
    if len(selected_rects) < num_regions_target: padding_box = selected_rects[-1] if selected_rects else torch.tensor([0,0,float(W),float(H)],dtype=dtype,device=device); selected_rects.extend([padding_box.clone() for _ in range(num_regions_target - len(selected_rects))])
    return torch.stack(selected_rects)

def phi_spiral_patch_centers_fixed_n(frame_dims:Tuple[int,int], num_centers:int, device='cpu', dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]:
    W, H = frame_dims; centers_xy = []; scale_factors = []; cx, cy = W / 2.0, H / 2.0
    if num_centers <= 0: return torch.empty(0,2,device=device,dtype=dtype), torch.empty(0,1,device=device,dtype=dtype)
    centers_xy.append([cx, cy]); scale_factors.append(0.25)
    num_spiral_points_to_generate = num_centers - 1
    if num_spiral_points_to_generate <= 0:
        if num_centers == 1: return torch.tensor(centers_xy, dtype=dtype, device=device), torch.tensor(scale_factors, dtype=dtype, device=device).unsqueeze(-1)
        else: return torch.empty(0,2,device=device,dtype=dtype), torch.empty(0,1,device=device,dtype=dtype)
    a = 0.05 * min(W, H); b = math.log(PHI) / (math.pi / 2); angle_step = PHI * 2 * math.pi / num_spiral_points_to_generate if num_spiral_points_to_generate > 0 else 0; current_angle = 0.0
    for i in range(num_spiral_points_to_generate):
        r = a * math.exp(b * current_angle); max_r = max(W,H) * 0.6; r = min(r,max_r)
        x = cx + r * math.cos(current_angle); y = cy + r * math.sin(current_angle)
        x_clamped = max(0.0, min(x, float(W))); y_clamped = max(0.0, min(y, float(H)))
        centers_xy.append([x_clamped, y_clamped]); patch_scale = max(0.05, 0.20 * math.exp(-0.5 * r / (min(W,H)*0.1))); scale_factors.append(patch_scale); current_angle += angle_step
    if len(centers_xy) < num_centers: num_to_pad = num_centers - len(centers_xy); last_xy = centers_xy[-1] if centers_xy else [cx,cy]; last_scale = scale_factors[-1] if scale_factors else 0.1; centers_xy.extend([last_xy] * num_to_pad); scale_factors.extend([last_scale] * num_to_pad)
    return torch.tensor(centers_xy[:num_centers], dtype=dtype, device=device), torch.tensor(scale_factors[:num_centers], dtype=dtype, device=device).unsqueeze(-1)

# =====================================================================
# Architectural Components (v0.1 - VAE-GAN Refactor)
# =====================================================================

class RegionalPatchExtractor(nn.Module):
    def __init__(self, patch_output_size: Optional[Tuple[int, int]] = None, feature_extractor: Optional[nn.Module] = None, feature_map_spatial_scale: float = 1.0, roi_align_output_size: Optional[Tuple[int, int]] = None, use_roi_align: bool = False):
        super().__init__(); self.patch_output_size = patch_output_size; self.feature_extractor = feature_extractor; self.feature_map_spatial_scale = feature_map_spatial_scale; self.roi_align_output_size = roi_align_output_size; self.use_roi_align = use_roi_align; current_logger=logging.getLogger("WuBuGAADHybridGenV01.PatchExtract"); self.resize_transform=None # Updated logger
        if self.use_roi_align:
            if self.feature_extractor is None or self.roi_align_output_size is None: raise ValueError("feature_extractor and roi_align_output_size needed for use_roi_align=True")
            current_logger.info(f"Using RoIAlign. Output: {roi_align_output_size}, FeatMapScale: {feature_map_spatial_scale:.2f}")
        else:
            if self.patch_output_size is None: raise ValueError("patch_output_size needed for use_roi_align=False")
            current_logger.info(f"Using Pixel Patches. Resizing to: {patch_output_size}")
            self.resize_transform = T.Resize(patch_output_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    def forward(self, images: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
        B, NumRegions, _ = bboxes_batch.shape; device = images.device; original_images_dtype = images.dtype; compute_dtype = torch.float32 if images.dtype == torch.uint8 else images.dtype; images_for_processing = images.to(compute_dtype)
        if self.use_roi_align and self.feature_extractor is not None and self.roi_align_output_size is not None:
            feature_maps = self.feature_extractor(images_for_processing); h_feat, w_feat = feature_maps.shape[2:]; max_w_feat_scalar=float(w_feat); max_h_feat_scalar=float(h_feat); scaled_bboxes_for_roialign_list = []
            for b in range(B):
                current_bboxes_scaled=bboxes_batch[b].to(torch.float32)*self.feature_map_spatial_scale; current_bboxes_scaled[:,0]=torch.clamp(current_bboxes_scaled[:,0],min=0.0,max=max_w_feat_scalar-EPS); current_bboxes_scaled[:,1]=torch.clamp(current_bboxes_scaled[:,1],min=0.0,max=max_h_feat_scalar-EPS); min_for_x2=current_bboxes_scaled[:,0]; current_bboxes_scaled[:,2]=torch.clamp(current_bboxes_scaled[:,2],max=max_w_feat_scalar); current_bboxes_scaled[:,2]=torch.maximum(current_bboxes_scaled[:,2],min_for_x2); min_for_y2=current_bboxes_scaled[:,1]; current_bboxes_scaled[:,3]=torch.clamp(current_bboxes_scaled[:,3],max=max_h_feat_scalar); current_bboxes_scaled[:,3]=torch.maximum(current_bboxes_scaled[:,3],min_for_y2);
                batch_indices=torch.full((NumRegions,1),float(b),device=device,dtype=current_bboxes_scaled.dtype); scaled_bboxes_for_roialign_list.append(torch.cat([batch_indices,current_bboxes_scaled],dim=1))
            all_rois = torch.cat(scaled_bboxes_for_roialign_list, dim=0)
            try: aligned_features = roi_align(feature_maps, all_rois, output_size=self.roi_align_output_size, spatial_scale=1.0, aligned=True)
            except Exception as e_roi: logging.getLogger("WuBuGAADHybridGenV01.PatchExtract").error(f"RoIAlign failed: {e_roi}. FeatMap:{feature_maps.shape}, RoIs:{all_rois.shape}, Output:{self.roi_align_output_size}"); raise e_roi
            C_feat=feature_maps.shape[1]; H_roi, W_roi = self.roi_align_output_size; aligned_features=aligned_features.view(B,NumRegions,C_feat,H_roi,W_roi); return aligned_features.to(original_images_dtype)
        else:
            all_patches = []; H_img, W_img = images.shape[2:]
            for b in range(B):
                batch_patches = []
                for r in range(NumRegions):
                    x1,y1,x2,y2 = bboxes_batch[b,r].round().int().tolist(); x1_c,y1_c=max(0,x1),max(0,y1); x2_c,y2_c=min(W_img,x2),min(H_img,y2)
                    if x1_c >= x2_c or y1_c >= y2_c: patch = torch.zeros((images.shape[1],)+self.patch_output_size, device=device, dtype=original_images_dtype)
                    else: patch = images_for_processing[b, :, y1_c:y2_c, x1_c:x2_c]; patch = self.resize_transform(patch) if self.resize_transform else patch
                    batch_patches.append(patch)
                all_patches.append(torch.stack(batch_patches))
            return torch.stack(all_patches).to(original_images_dtype)

class PatchEmbed(nn.Module):
    def __init__(self, patch_feature_dim: int, embed_dim: int):
        super().__init__(); self.proj = nn.Linear(patch_feature_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, *patch_dims = x.shape; x = x.view(B, N, -1); return self.proj(x)

# Renamed from RegionalHyperbolicEncoder
class RegionalVAEEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_config: Dict, latent_dim: int):
        super().__init__(); self.args = args; self.video_config = video_config; self.gaad_config = gaad_config; self.wubu_s_config = wubu_s_config; self.latent_dim = latent_dim; self.image_size = (args.image_h, args.image_w); self.num_appearance_regions = gaad_config['num_regions']; self.decomposition_type = gaad_config['decomposition_type']; self.gaad_min_size_px = gaad_config.get('min_size_px', 5); current_logger=logging.getLogger("WuBuGAADHybridGenV01.EncoderVAE"); self.feature_extractor: Optional[nn.Module] = None; feature_map_scale = 1.0; patch_input_channels = self.video_config['num_channels']; roi_align_output_size = None; use_roi_align = False # Updated logger
        if args.encoder_use_roi_align:
            self.feature_extractor = nn.Sequential(nn.Conv2d(self.video_config['num_channels'], args.encoder_shallow_cnn_channels, kernel_size=3, stride=1, padding=1), nn.GroupNorm(8, args.encoder_shallow_cnn_channels), nn.GELU()); patch_input_channels = args.encoder_shallow_cnn_channels; roi_align_output_size = (args.encoder_roi_align_output_h, args.encoder_roi_align_output_w); use_roi_align = True; current_logger.info(f"Using RoIAlign (OutCh: {patch_input_channels}, RoISize: {roi_align_output_size})")
        else: current_logger.info(f"Using Pixel Patches (Resize: {args.encoder_pixel_patch_size}x{args.encoder_pixel_patch_size})")
        self.patch_extractor = RegionalPatchExtractor(patch_output_size=(args.encoder_pixel_patch_size, args.encoder_pixel_patch_size) if not use_roi_align else None, feature_extractor=self.feature_extractor, feature_map_spatial_scale=feature_map_scale, roi_align_output_size=roi_align_output_size, use_roi_align=use_roi_align)
        patch_output_h = roi_align_output_size[0] if use_roi_align else args.encoder_pixel_patch_size; patch_output_w = roi_align_output_size[1] if use_roi_align else args.encoder_pixel_patch_size; patch_feature_dim = patch_input_channels * patch_output_h * patch_output_w; self.patch_embed = PatchEmbed(patch_feature_dim, args.encoder_initial_tangent_dim)
        # WuBu-S maps regional features to an intermediate representation
        self.wubu_s = FullyHyperbolicWuBuNestingModel(input_tangent_dim=args.encoder_initial_tangent_dim, output_tangent_dim=video_config['wubu_s_output_dim'], config=wubu_s_config);
        # Store final curvature if WuBu-S uses hyperbolic layers
        self.wubu_s_final_hyp_dim = wubu_s_config['hyperbolic_dims'][-1] if wubu_s_config['num_levels'] > 0 and wubu_s_config['hyperbolic_dims'] else 0; self.wubu_s_final_curvature = 1.0
        if wubu_s_config['num_levels'] > 0 and self.wubu_s_final_hyp_dim > 0:
            last_level_idx = wubu_s_config['num_levels'] - 1
            try: temp_level = HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_s_final_hyp_dim, wubu_s_config, wubu_s_config['initial_curvatures'][last_level_idx]); self.wubu_s_final_curvature = temp_level.get_current_curvature_scalar(); del temp_level; current_logger.info(f"WuBu-S final level curvature estimated as {self.wubu_s_final_curvature:.3f}")
            except IndexError: current_logger.error(f"Index error accessing init curvatures WuBu-S L{last_level_idx}. Default C=1.0."); self.wubu_s_final_curvature = 1.0

        # WuBu-T processes temporal sequence and outputs latent distribution params
        self.wubu_t_input_dim = video_config['wubu_s_output_dim']
        self.wubu_m_output_dim = video_config.get('wubu_m_output_dim', 0)
        if args.use_wubu_motion_branch and self.wubu_m_output_dim > 0:
            self.wubu_t_input_dim += self.wubu_m_output_dim
            current_logger.info(f"VAE Encoder: Including motion features (dim {self.wubu_m_output_dim}) for WuBu-T input.")
        elif args.use_wubu_motion_branch:
            current_logger.warning("VAE Encoder: Motion branch enabled but dim is 0. Not included in WuBu-T.")

        self.wubu_t_config = _configure_wubu_stack(args, "wubu_t") # Get WuBu-T config
        self.wubu_t: Optional[FullyHyperbolicWuBuNestingModel] = None
        if self.wubu_t_config and self.wubu_t_config['num_levels'] > 0 and self.wubu_t_input_dim > 0:
             # WuBu-T outputs an intermediate temporal feature vector
             self.wubu_t_output_dim = self.wubu_t_config['hyperbolic_dims'][-1] if self.wubu_t_config['hyperbolic_dims'] else 0
             self.wubu_t = FullyHyperbolicWuBuNestingModel(input_tangent_dim=self.wubu_t_input_dim, output_tangent_dim=self.wubu_t_output_dim, config=self.wubu_t_config)
             current_logger.info(f"VAE Encoder WuBu-T Enabled: InputDim {self.wubu_t_input_dim}, OutputDim {self.wubu_t_output_dim}")
             # Projection layers to get mu and logvar from WuBu-T's output
             self.fc_mu = nn.Linear(self.wubu_t_output_dim, self.latent_dim)
             self.fc_logvar = nn.Linear(self.wubu_t_output_dim, self.latent_dim)
        else:
             current_logger.warning("VAE Encoder WuBu-T disabled (no levels, input dim 0, or config missing). Latent space will be direct projection.")
             # If no WuBu-T, project directly from aggregated S (and maybe M) features
             self.fc_mu = nn.Linear(self.wubu_t_input_dim, self.latent_dim) # Use wubu_t_input_dim here
             self.fc_logvar = nn.Linear(self.wubu_t_input_dim, self.latent_dim)

        self.apply(init_weights_general)

    # Forward pass for the VAE Encoder
    def forward(self, frames_pixels: torch.Tensor, motion_features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, N_frames, C, H, W = frames_pixels.shape; device = frames_pixels.device; dtype = next(self.parameters()).dtype;
        frames_pixels_flat = frames_pixels.reshape(B * N_frames, C, H, W); gaad_bboxes_list = []
        for b_idx in range(B):
            current_frame_h, current_frame_w = H, W; frame_dims = (current_frame_w, current_frame_h); max_w_scalar=float(frame_dims[0]); max_h_scalar=float(frame_dims[1]);
            # GAAD logic (same as before)
            if self.decomposition_type == "hybrid":
                num_subdivide=self.num_appearance_regions//2; num_spiral=self.num_appearance_regions-num_subdivide; bboxes_for_item=[]
                if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims,num_subdivide,device,dtype,self.gaad_min_size_px))
                if num_spiral > 0:
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); bboxes_for_item.append(spiral_bboxes_current)
                frame_bboxes = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_appearance_regions, dtype=dtype, device=device)
            elif self.decomposition_type == "spiral":
                 spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_appearance_regions, device, dtype); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(self.num_appearance_regions, 4, device=device, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); frame_bboxes = spiral_bboxes_current
            else: frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims,self.num_appearance_regions,device,dtype,self.gaad_min_size_px)
            if frame_bboxes.shape[0] < self.num_appearance_regions: num_to_pad=self.num_appearance_regions-frame_bboxes.shape[0]; padding_box=frame_bboxes[-1:].clone() if frame_bboxes.shape[0]>0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]],dtype=dtype,device=device); padding=padding_box.repeat(num_to_pad,1); frame_bboxes=torch.cat([frame_bboxes, padding], dim=0)
            elif frame_bboxes.shape[0] > self.num_appearance_regions: frame_bboxes=frame_bboxes[:self.num_appearance_regions]
            gaad_bboxes_list.append(frame_bboxes)

        gaad_bboxes_batch = torch.stack(gaad_bboxes_list); gaad_bboxes_full = gaad_bboxes_batch.unsqueeze(1).repeat(1, N_frames, 1, 1); gaad_bboxes_flat = gaad_bboxes_full.reshape(B * N_frames, self.num_appearance_regions, 4)
        extracted_patches = self.patch_extractor(frames_pixels_flat, gaad_bboxes_flat); B_flat_post, NumReg_post, C_patch, H_patch, W_patch = extracted_patches.shape; patches_for_embed = extracted_patches.reshape(B_flat_post, NumReg_post, -1); initial_tangent_vectors = self.patch_embed(patches_for_embed)

        wubu_s_input = initial_tangent_vectors.reshape(B_flat_post * NumReg_post, -1);
        wubu_s_output_tangent_flat = self.wubu_s(wubu_s_input) # Output is tangent space feature
        D_out_s = wubu_s_output_tangent_flat.shape[-1]
        regional_app_features_tangent = wubu_s_output_tangent_flat.reshape(B, N_frames, NumReg_post, D_out_s)

        # Aggregate regional features per frame (e.g., max pooling or mean pooling)
        agg_app_features = torch.mean(regional_app_features_tangent, dim=2) # (B, N_frames, D_out_s)

        # Combine with motion features if available
        wubu_t_input_features = agg_app_features
        if motion_features is not None and self.args.use_wubu_motion_branch:
             # Ensure motion features are tangent if they came from a hyperbolic WuBu-M
             if hasattr(motion_features, 'manifold') and isinstance(motion_features.manifold, PoincareBall):
                 # This assumes motion_features are the hyperbolic output of WuBu-M
                 final_manifold_m = PoincareBall(self.wubu_m_final_curvature if hasattr(self, 'wubu_m_final_curvature') else 1.0)
                 motion_features_tangent = final_manifold_m.logmap0(motion_features.to(dtype))
             else: # Assume motion features are already tangent-like (e.g. direct stats or Euclidean WuBu-M)
                 motion_features_tangent = motion_features.to(dtype)

             # Aggregate motion features per frame (assuming motion_features has shape B, N_pairs, NumReg_motion, D_out_m)
             # Match temporal dimension N_frames. Use last motion feature for last frame? Or pad? Let's pad with zeros.
             N_pairs = motion_features_tangent.shape[1]
             agg_motion_features = torch.mean(motion_features_tangent, dim=2) # (B, N_pairs, D_out_m)
             if N_pairs < N_frames:
                 padding = torch.zeros(B, N_frames - N_pairs, agg_motion_features.shape[-1], device=device, dtype=dtype)
                 agg_motion_features = torch.cat([agg_motion_features, padding], dim=1)
             elif N_pairs > N_frames:
                 agg_motion_features = agg_motion_features[:, :N_frames, :]

             wubu_t_input_features = torch.cat([agg_app_features, agg_motion_features], dim=-1) # (B, N_frames, D_out_s + D_out_m)

        # Process through WuBu-T if enabled
        if self.wubu_t:
            temporal_features = self.wubu_t(wubu_t_input_features) # (B, N_frames, wubu_t_output_dim)
            # Aggregate temporal features (e.g., take the last time step)
            final_temporal_feature = temporal_features[:, -1, :] # (B, wubu_t_output_dim)
        else:
            # If no WuBu-T, aggregate directly from input features
            final_temporal_feature = torch.mean(wubu_t_input_features, dim=1) # (B, wubu_t_input_dim)

        # Project to latent distribution parameters
        mu = self.fc_mu(final_temporal_feature)
        logvar = self.fc_logvar(final_temporal_feature)

        # Return latent params and the original GAAD boxes for potential use in decoder/generator
        return mu, logvar, gaad_bboxes_full, regional_app_features_tangent # Return tangent features for potential use

# --- UPDATED: Motion Encoder using Optical Flow (Largely Unchanged from Diff version) ---
class RegionalHyperbolicMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_motion_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_m_config = wubu_m_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV01.EncoderM")
        self.enabled = args.use_wubu_motion_branch and wubu_m_config is not None and gaad_motion_config is not None and OPTICAL_FLOW_AVAILABLE

        # --- Optical Flow Network Setup ---
        self.flow_net = None
        if self.enabled:
            if args.optical_flow_net_type not in FLOW_MODELS:
                self.logger.error(f"Optical flow model type '{args.optical_flow_net_type}' not found or torchvision.models.optical_flow unavailable. Disabling motion branch.")
                self.enabled = False
            else:
                weights, model_builder = FLOW_MODELS[args.optical_flow_net_type]
                try:
                    self.flow_net = model_builder(weights=weights)
                    if args.freeze_flow_net:
                        for param in self.flow_net.parameters(): param.requires_grad = False
                        self.flow_net.eval()
                        self.logger.info(f"Loaded and FROZE pre-trained Optical Flow Net: {args.optical_flow_net_type}")
                    else:
                        self.logger.info(f"Loaded pre-trained Optical Flow Net (TRAINABLE): {args.optical_flow_net_type}")
                except Exception as e:
                    self.logger.error(f"Failed to load optical flow model '{args.optical_flow_net_type}': {e}. Disabling motion branch.", exc_info=True)
                    self.flow_net = None; self.enabled = False
        else:
             if not args.use_wubu_motion_branch: self.logger.info("Motion Encoder branch DISABLED by config.")
             elif not OPTICAL_FLOW_AVAILABLE: self.logger.warning("Motion Encoder branch DISABLED (torchvision.models.optical_flow unavailable).")
             elif not wubu_m_config: self.logger.warning("Motion Encoder branch DISABLED (WuBu-M config missing).")
             elif not gaad_motion_config: self.logger.warning("Motion Encoder branch DISABLED (GAAD-Motion config missing).")

        if not self.enabled: self.wubu_m_final_curvature = 1.0; return

        self.image_size = (args.image_h, args.image_w)
        self.num_motion_regions = gaad_motion_config['num_regions']
        self.motion_decomposition_type = gaad_motion_config['decomposition_type']
        self.gaad_min_size_px = gaad_motion_config.get('min_size_px', 5)

        self.flow_stats_components = args.flow_stats_components
        self.flow_stats_dim = 0
        if 'mag_mean' in self.flow_stats_components: self.flow_stats_dim += 1
        if 'angle_mean' in self.flow_stats_components: self.flow_stats_dim += 2 # cos/sin
        if 'mag_std' in self.flow_stats_components: self.flow_stats_dim += 1
        if 'angle_std' in self.flow_stats_components: self.flow_stats_dim += 1 # std of angle (or var of cos/sin)
        if self.flow_stats_dim == 0: self.logger.warning("No flow statistics components selected for motion encoder.")

        self.motion_feature_embed = nn.Linear(self.flow_stats_dim, args.encoder_initial_tangent_dim) if self.flow_stats_dim > 0 else nn.Identity()

        if self.wubu_m_config is not None and self.flow_stats_dim > 0:
            # WuBu-M output dim needs careful consideration - maybe it should output tangent features?
            # Let's keep the original output dim for now, assuming it might map to hyperbolic space.
            self.wubu_m = FullyHyperbolicWuBuNestingModel(input_tangent_dim=args.encoder_initial_tangent_dim, output_tangent_dim=video_config['wubu_m_output_dim'], config=wubu_m_config)
            self.wubu_m_final_hyp_dim = wubu_m_config['hyperbolic_dims'][-1] if wubu_m_config['num_levels'] > 0 and wubu_m_config['hyperbolic_dims'] else 0
            self.wubu_m_final_curvature = 1.0
            if wubu_m_config['num_levels'] > 0 and self.wubu_m_final_hyp_dim > 0:
                last_level_idx = wubu_m_config['num_levels'] - 1
                try: temp_level_m=HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_m_final_hyp_dim, wubu_m_config, wubu_m_config['initial_curvatures'][last_level_idx]); self.wubu_m_final_curvature = temp_level_m.get_current_curvature_scalar(); del temp_level_m; self.logger.info(f"WuBu-M final level curvature estimated as {self.wubu_m_final_curvature:.3f}")
                except IndexError: self.logger.error(f"Index error accessing init curvatures WuBu-M L{last_level_idx}. Default C=1.0."); self.wubu_m_final_curvature = 1.0
            # Add manifold attribute to WuBu-M output if hyperbolic
            if self.wubu_m_final_hyp_dim > 0:
                 for p in self.wubu_m.output_tangent_projection.parameters(): # Hacky way to attach manifold info
                    setattr(p, 'manifold', PoincareBall(self.wubu_m_final_curvature))

        elif self.wubu_m_config is not None and self.flow_stats_dim == 0:
             self.logger.warning("WuBu-M configured, but flow_stats_dim is 0. WuBu-M will effectively not run or produce zero features."); self.wubu_m = None; self.wubu_m_final_hyp_dim=0; self.wubu_m_final_curvature=1.0
        else: self.logger.error("MotionEncoder: wubu_m_config is None, cannot initialize WuBu-M model."); self.wubu_m = None; self.wubu_m_final_hyp_dim=0; self.wubu_m_final_curvature=1.0; self.enabled = False
        self.apply(init_weights_general)

    def _get_motion_gaad_bboxes(self, analysis_map: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        B_eff, _, H, W = analysis_map.shape; all_batch_bboxes = []
        for i in range(B_eff):
            frame_dims = (W, H); max_w_scalar=float(W); max_h_scalar=float(H)
            if self.motion_decomposition_type == "hybrid": # content_aware treated as hybrid for bbox generation
                num_subdivide=self.num_motion_regions//2; num_spiral=self.num_motion_regions-num_subdivide; bboxes_for_item=[]
                if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims,num_subdivide,device,dtype,self.gaad_min_size_px))
                if num_spiral > 0:
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); bboxes_for_item.append(spiral_bboxes_current)
                frame_bboxes = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_motion_regions, dtype=dtype, device=device)
            elif self.motion_decomposition_type == "spiral":
                 spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_motion_regions, device, dtype); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(self.num_motion_regions, 4, device=device, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); frame_bboxes = spiral_bboxes_current
            else: frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims, self.num_motion_regions, device, dtype, self.gaad_min_size_px)
            if frame_bboxes.shape[0] < self.num_motion_regions: num_to_pad = self.num_motion_regions - frame_bboxes.shape[0]; padding_box = frame_bboxes[-1:].clone() if frame_bboxes.shape[0] > 0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]], dtype=dtype, device=device); padding = padding_box.repeat(num_to_pad, 1); frame_bboxes = torch.cat([frame_bboxes, padding], dim=0)
            elif frame_bboxes.shape[0] > self.num_motion_regions: frame_bboxes = frame_bboxes[:self.num_motion_regions]
            all_batch_bboxes.append(frame_bboxes)
        return torch.stack(all_batch_bboxes)

    def _extract_flow_statistics(self, flow_field: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        B, _, H, W = flow_field.shape; N_reg = bboxes.shape[1]; device = flow_field.device; dtype = flow_field.dtype
        all_stats = torch.zeros(B, N_reg, self.flow_stats_dim, device=device, dtype=dtype)
        for b in range(B):
            for r in range(N_reg):
                x1, y1, x2, y2 = bboxes[b, r].round().int().tolist(); x1_c, y1_c = max(0, x1), max(0, y1); x2_c, y2_c = min(W, x2), min(H, y2)
                if x1_c >= x2_c or y1_c >= y2_c: continue
                region_flow = flow_field[b, :, y1_c:y2_c, x1_c:x2_c]; flow_dx = region_flow[0, ...].flatten(); flow_dy = region_flow[1, ...].flatten()
                if flow_dx.numel() == 0: continue
                stat_idx = 0; magnitudes = torch.sqrt(flow_dx**2 + flow_dy**2)
                if 'mag_mean' in self.flow_stats_components: all_stats[b, r, stat_idx] = torch.mean(magnitudes); stat_idx += 1
                if 'mag_std' in self.flow_stats_components: all_stats[b, r, stat_idx] = torch.std(magnitudes) if magnitudes.numel() > 1 else 0.0; stat_idx += 1
                angles = torch.atan2(flow_dy, flow_dx) # Compute angles once if needed
                if 'angle_mean' in self.flow_stats_components: all_stats[b,r,stat_idx]=torch.mean(torch.cos(angles)); stat_idx+=1; all_stats[b,r,stat_idx]=torch.mean(torch.sin(angles)); stat_idx+=1
                if 'angle_std' in self.flow_stats_components: angle_std = torch.std(angles) if angles.numel() > 1 else 0.0; all_stats[b,r,stat_idx]=angle_std if torch.isfinite(angle_std) else 0.0; stat_idx+=1
        return all_stats

    def forward(self, frames_pixels: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.enabled or self.flow_net is None : return None
        B, N_frames, C, H, W = frames_pixels.shape; device = frames_pixels.device; original_dtype = frames_pixels.dtype; compute_dtype = next(self.parameters()).dtype
        if N_frames < 2: self.logger.debug(f"Not enough frames ({N_frames}) for flow. Skip motion."); return None
        num_pairs = N_frames - 1; all_motion_features_list = []; all_motion_bboxes_list = [] # Store features (hyperbolic or tangent)

        flow_context = torch.no_grad() if self.args.freeze_flow_net else contextlib.nullcontext()
        with flow_context:
            for i in range(num_pairs):
                frame_t = frames_pixels[:, i+1, ...]
                frame_t_minus_1 = frames_pixels[:, i, ...]
                frame_t_float = frame_t.to(torch.float32)
                frame_t_minus_1_float = frame_t_minus_1.to(torch.float32)
                try:
                    flow_predictions = self.flow_net(frame_t_minus_1_float, frame_t_float)
                    flow_field = flow_predictions[-1].to(compute_dtype)
                except ValueError as ve:
                    if "should be divisible by 8" in str(ve):
                         self.logger.error(f"Optical flow input shape error for pair {i}: {ve}. Frame shape: H={H}, W={W}. Ensure image_h and image_w are divisible by 8.", exc_info=False)
                    else:
                        self.logger.error(f"Optical flow failed pair {i} (Input Shapes: {frame_t_minus_1_float.shape}, {frame_t_float.shape}): {ve}", exc_info=True)
                    return None # Stop processing if flow fails
                except Exception as e_flow:
                    self.logger.error(f"Optical flow failed pair {i} (Input Shapes: {frame_t_minus_1_float.shape}, {frame_t_float.shape}): {e_flow}", exc_info=True)
                    return None # Stop processing if flow fails

                flow_magnitude = torch.sqrt(flow_field[:, 0:1, :, :]**2 + flow_field[:, 1:2, :, :]**2)
                motion_gaad_bboxes_batch = self._get_motion_gaad_bboxes(flow_magnitude, device, compute_dtype)

                if self.flow_stats_dim > 0:
                    flow_stats = self._extract_flow_statistics(flow_field, motion_gaad_bboxes_batch)
                    flow_stats_flat = flow_stats.view(B * self.num_motion_regions, self.flow_stats_dim)
                    initial_motion_tangent_vectors_flat = self.motion_feature_embed(flow_stats_flat)
                else:
                    initial_motion_tangent_vectors_flat = torch.zeros(B * self.num_motion_regions, self.args.encoder_initial_tangent_dim, device=device, dtype=compute_dtype)

                # --- WuBu-M processing ---
                if self.wubu_m is None:
                    motion_features_pair_flat = initial_motion_tangent_vectors_flat # Use tangent if no WuBu-M
                else:
                    wubu_m_output_tangent_flat = self.wubu_m(initial_motion_tangent_vectors_flat)
                    if self.wubu_m_final_hyp_dim > 0:
                        # Map to hyperbolic space if WuBu-M is hyperbolic
                        final_manifold_m = PoincareBall(self.wubu_m_final_curvature)
                        motion_features_pair_flat = final_manifold_m.expmap0(wubu_m_output_tangent_flat)
                    else: # Output remains tangent if WuBu-M is Euclidean
                        motion_features_pair_flat = wubu_m_output_tangent_flat

                # Reshape and store results for the current pair
                motion_features_pair = motion_features_pair_flat.reshape(B, self.num_motion_regions, -1)
                all_motion_features_list.append(motion_features_pair)
                all_motion_bboxes_list.append(motion_gaad_bboxes_batch)

        if not all_motion_features_list:
            self.logger.warning("No motion features were generated (likely due to flow errors or N_frames < 2).")
            return None

        # Stack features and bboxes across pairs, converting back to original dtype
        final_motion_features = torch.stack(all_motion_features_list, dim=1).to(original_dtype)
        final_motion_bboxes = torch.stack(all_motion_bboxes_list, dim=1).to(original_dtype)

        return final_motion_features, final_motion_bboxes

# Renamed from RegionalPixelSynthesisDecoder
class RegionalGeneratorDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_output_dim: int, latent_dim: int):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.image_size = (args.image_h, args.image_w)
        self.num_regions = gaad_config['num_regions']
        self.decoder_type = args.decoder_type
        self.num_channels = video_config['num_channels']
        self.latent_dim = latent_dim
        self.regional_feature_dim = wubu_s_output_dim 
        self.num_predict_frames = video_config["num_predict_frames"]
        self.logger = logging.getLogger("WuBuGAADHybridGenV01.Generator") # Instance logger

        gen_temporal_hidden_dim = self.latent_dim * 2 
        self.temporal_expander = nn.Sequential(
            nn.Linear(self.latent_dim, gen_temporal_hidden_dim),
            nn.GELU(),
            nn.Linear(gen_temporal_hidden_dim, self.num_predict_frames * self.num_regions * self.regional_feature_dim)
        )

        if self.decoder_type == "patch_gen":
            self.patch_size = args.decoder_patch_gen_size
            patch_pixels = self.num_channels * self.patch_size * self.patch_size
            self.patch_generator = nn.Sequential(
                nn.Linear(self.regional_feature_dim, self.regional_feature_dim * 2),
                nn.GELU(),
                nn.Linear(self.regional_feature_dim * 2, patch_pixels),
                nn.Tanh() 
            )
            self.patch_resize_mode = args.decoder_patch_resize_mode
            self.logger.info(f"Using Patch Generator (Input RegionalDim: {self.regional_feature_dim}, OutPatchSize: {self.patch_size}x{self.patch_size}, Resize: {self.patch_resize_mode})")
        elif self.decoder_type == "transformer":
            self.logger.warning("Decoder Transformer type NIY.")
            raise NotImplementedError("Transformer decoder NIY")
        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

        self.apply(init_weights_general)

    def forward(self, latent_code: torch.Tensor, gaad_bboxes: torch.Tensor) -> torch.Tensor:
        # gaad_bboxes is expected to have shape (B, N_pred, NumReg, 4)
        # where N_pred is self.num_predict_frames
        B = latent_code.shape[0]
        N_pred_from_bboxes = gaad_bboxes.shape[1] # Number of frames of bboxes we received
        NumReg = self.num_regions # From config, should match gaad_bboxes.shape[2]
        
        C, H, W = self.num_channels, self.image_size[0], self.image_size[1]
        device = latent_code.device
        dtype = latent_code.dtype

        # Temporal expander always produces features for self.num_predict_frames
        regional_tangent_features_flat = self.temporal_expander(latent_code)
        # Shape: (B, self.num_predict_frames * NumReg * regional_feature_dim)
        regional_tangent_features = regional_tangent_features_flat.view(B, self.num_predict_frames, NumReg, self.regional_feature_dim)

        # The number of frames we actually generate pixels for will be min(self.num_predict_frames, N_pred_from_bboxes)
        # if the generator's patch assembly loop depends on the number of bbox sets.
        # However, patch_generator input uses self.num_predict_frames.
        # The critical part is matching `gaad_bboxes_flat` to the frames loop.
        # Let's ensure operations use `self.num_predict_frames` consistently for generation,
        # and that `gaad_bboxes` also has `self.num_predict_frames` in its time dimension.

        if N_pred_from_bboxes != self.num_predict_frames:
            self.logger.warning(f"Generator received {N_pred_from_bboxes} bbox sets, but configured for {self.num_predict_frames} predict frames. This might lead to issues if not handled upstream or if decoder is not robust to it. Upstream should ensure correct bbox slicing.")
            # This warning indicates an issue in how WuBuGAADHybridGenNet.forward calls this.
            # For robustness here, we might be forced to use min(N_pred_from_bboxes, self.num_predict_frames)
            # or error out if the mismatch is problematic for subsequent logic.
            # Given the temporal_expander generates for self.num_predict_frames, we should ideally have bboxes for all of them.
            # If upstream logic in WuBuGAADHybridGenNet.forward correctly pads/truncates bboxes, this check might not be needed,
            # or it would catch an error from upstream.
            # For now, assume gaad_bboxes.shape[1] == self.num_predict_frames due to upstream handling.
            if not (gaad_bboxes.shape[0] == B and gaad_bboxes.shape[1] == self.num_predict_frames and gaad_bboxes.shape[2] == NumReg):
                 self.logger.error(f"Corrected bboxes in MainNet.forward still lead to mismatch in Generator. Bboxes shape: {gaad_bboxes.shape}. Expected N_pred={self.num_predict_frames}")
                 # Fallback to default if bboxes are critically mis-shaped despite upstream efforts
                 frame_dims=(W, H); default_bboxes_gen=golden_subdivide_rect_fixed_n(frame_dims, NumReg, device=device, dtype=dtype, min_size_px=self.args.gaad_min_size_px);
                 gaad_bboxes = default_bboxes_gen.unsqueeze(0).unsqueeze(0).repeat(B, self.num_predict_frames, 1, 1)


        regional_features_gen_input = regional_tangent_features.reshape(B * self.num_predict_frames * NumReg, self.regional_feature_dim)
        gaad_bboxes_flat = gaad_bboxes.reshape(B * self.num_predict_frames, NumReg, 4)

        if self.decoder_type == "patch_gen":
            generated_patch_pixels_flat = self.patch_generator(regional_features_gen_input)
            generated_patches = generated_patch_pixels_flat.view(B * self.num_predict_frames, NumReg, C, self.patch_size, self.patch_size)

            canvas = torch.zeros(B * self.num_predict_frames, C, H, W, device=device, dtype=dtype)
            counts = torch.zeros(B * self.num_predict_frames, 1, H, W, device=device, dtype=dtype)

            # Loop iterates B * self.num_predict_frames times.
            # gaad_bboxes_flat must also have B * self.num_predict_frames in its first dimension.
            for i in range(B * self.num_predict_frames): # Iterate over each frame to be generated for each batch item
                for r in range(NumReg):
                    patch = generated_patches[i, r]
                    x1, y1, x2, y2 = gaad_bboxes_flat[i, r].tolist()
                    target_h = int(round(y2 - y1))
                    target_w = int(round(x2 - x1))
                    place_y1 = int(round(y1))
                    place_x1 = int(round(x1))

                    if target_h <= 0 or target_w <= 0: continue

                    resize_kwargs = {'size': (target_h, target_w), 
                                     'mode': self.patch_resize_mode, 
                                     'antialias': True if self.patch_resize_mode != 'nearest' else None}
                    if self.patch_resize_mode != 'nearest': 
                        resize_kwargs['align_corners'] = False # Common practice for non-nearest

                    resized_patch = F.interpolate(patch.unsqueeze(0), **resize_kwargs).squeeze(0)

                    place_y2 = min(H, place_y1 + target_h)
                    place_x2 = min(W, place_x1 + target_w)
                    place_y1 = max(0, place_y1) # Clamp to canvas bounds
                    place_x1 = max(0, place_x1)
                    
                    slice_h = place_y2 - place_y1
                    slice_w = place_x2 - place_x1

                    if slice_h <= 0 or slice_w <= 0: continue
                    
                    canvas[i, :, place_y1:place_y2, place_x1:place_x2] += resized_patch[:, :slice_h, :slice_w]
                    counts[i, :, place_y1:place_y2, place_x1:place_x2] += 1
            
            output_canvas_flat = torch.where(counts > 0, canvas / counts.clamp(min=1.0), canvas)
            # Reshape to (B, self.num_predict_frames, C, H, W)
            output_frames = output_canvas_flat.view(B, self.num_predict_frames, C, H, W)
        else:
            raise NotImplementedError(f"Decoder type {self.decoder_type} forward pass NIY.")

        return output_frames

# --- NEW: Discriminator ---
class RegionalDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_config = gaad_config
        self.disc_config = disc_config # e.g., {"type": "regional_cnn", "cnn_channels": [64, 128], "use_wubu": False}
        self.logger = logging.getLogger("WuBuGAADHybridGenV01.Discriminator")

        self.image_size = (args.image_h, args.image_w)
        self.num_channels = video_config['num_channels']
        self.num_regions = gaad_config['num_regions']
        self.gaad_min_size_px = gaad_config['min_size_px']
        self.decomposition_type = gaad_config['decomposition_type'] # Use appearance GAAD for discriminator

        disc_type = self.disc_config.get("type", "regional_cnn")
        self.logger.info(f"Initializing Discriminator Type: {disc_type}")

        if disc_type == "regional_cnn":
            # Feature extractor for regions (similar to encoder's but simpler)
            self.patch_size = disc_config.get("patch_size", 16)
            self.resize_transform = T.Resize((self.patch_size, self.patch_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
            cnn_channels = disc_config.get("cnn_channels", [32, 64, 128])
            patch_feature_dim = cnn_channels[-1] # Output dim of CNN feature extractor per patch
            layers = []
            in_c = self.num_channels
            for out_c in cnn_channels:
                 layers.extend([
                     nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                     nn.InstanceNorm2d(out_c, affine=True), # Use InstanceNorm maybe?
                     nn.LeakyReLU(0.2, inplace=True)
                 ])
                 in_c = out_c
            # Reduce feature map size - depends on patch_size and strides
            # Calculate final feature map size (H_f, W_f) after convs
            # This needs adjustment based on actual conv parameters
            test_input = torch.randn(1, self.num_channels, self.patch_size, self.patch_size)
            h_f, w_f = nn.Sequential(*layers)(test_input).shape[-2:]
            self.regional_feature_extractor = nn.Sequential(*layers)
            # Flatten and project regional features
            final_feature_dim_per_region = cnn_channels[-1] * h_f * w_f
            # Aggregate features across regions (simple mean) and project to single logit
            self.final_layer = nn.Sequential(
                nn.Linear(final_feature_dim_per_region, 1)
                # Sigmoid is usually applied implicitly by BCEWithLogitsLoss
            )
            self.logger.info(f"RegionalCNN Disc: PatchSize {self.patch_size}, CNN Channels {cnn_channels}, Final Feat/Region {final_feature_dim_per_region}")

        elif disc_type == "wubu_regional":
             # Placeholder: This would involve GAAD, Patch Extraction, WuBu-S/M/T stacks
             # similar to the encoder, finally projecting aggregated features to a logit.
             self.logger.warning("WuBu Discriminator NIY, using simplified CNN structure instead.")
             # Fallback to CNN temporarily
             self.disc_config["type"] = "regional_cnn"
             self.__init__(args, video_config, gaad_config, self.disc_config) # Re-initialize with CNN

        else:
            raise ValueError(f"Unsupported discriminator type: {disc_type}")

        self.apply(init_weights_general)


    def forward(self, frames_pixels: torch.Tensor) -> torch.Tensor:
        # Input: (B, N_frames, C, H, W) - Discriminator typically acts per frame or on sequence features
        # Let's assume it acts per frame for simplicity now.
        B, N, C, H, W = frames_pixels.shape
        device = frames_pixels.device
        dtype = frames_pixels.dtype # Use input dtype

        # Process first frame of the sequence for simplicity
        frame_to_disc = frames_pixels[:, 0, ...] # (B, C, H, W)

        if self.disc_config.get("type", "regional_cnn") == "regional_cnn":
            # 1. Get GAAD regions for the frame
            gaad_bboxes_list = []
            for b in range(B):
                frame_dims=(W,H); max_w_scalar=float(W); max_h_scalar=float(H)
                # Use the same GAAD logic as encoder
                if self.decomposition_type == "hybrid":
                    num_subdivide=self.num_regions//2; num_spiral=self.num_regions-num_subdivide; bboxes_for_item=[]
                    if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims,num_subdivide,device,dtype,self.gaad_min_size_px))
                    if num_spiral > 0:
                         spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); bboxes_for_item.append(spiral_bboxes_current)
                    frame_bboxes = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_regions, dtype=dtype, device=device)
                elif self.decomposition_type == "spiral":
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_regions, device, dtype); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(self.num_regions, 4, device=device, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); frame_bboxes = spiral_bboxes_current
                else: frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims,self.num_regions,device,dtype,self.gaad_min_size_px)
                if frame_bboxes.shape[0] < self.num_regions: num_to_pad=self.num_regions-frame_bboxes.shape[0]; padding_box=frame_bboxes[-1:].clone() if frame_bboxes.shape[0]>0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]],dtype=dtype,device=device); padding=padding_box.repeat(num_to_pad,1); frame_bboxes=torch.cat([frame_bboxes, padding], dim=0)
                elif frame_bboxes.shape[0] > self.num_regions: frame_bboxes=frame_bboxes[:self.num_regions]
                gaad_bboxes_list.append(frame_bboxes)
            gaad_bboxes_batch = torch.stack(gaad_bboxes_list) # (B, NumReg, 4)

            # 2. Extract patches and features per region
            all_regional_features = []
            for b in range(B):
                batch_region_features = []
                for r in range(self.num_regions):
                    x1,y1,x2,y2 = gaad_bboxes_batch[b,r].round().int().tolist(); x1_c,y1_c=max(0,x1),max(0,y1); x2_c,y2_c=min(W,x2),min(H,y2)
                    if x1_c >= x2_c or y1_c >= y2_c:
                         # Use zero features for empty regions
                         patch_features = torch.zeros(1, self.regional_feature_extractor[0].out_channels, 1, 1, device=device, dtype=dtype) # Shape after pooling
                         # Calculate expected output dimension from final_layer's input
                         feat_dim = self.final_layer[0].in_features
                         patch_features = torch.zeros(1, feat_dim, device=device, dtype=dtype)

                    else:
                         patch = frame_to_disc[b, :, y1_c:y2_c, x1_c:x2_c]
                         resized_patch = self.resize_transform(patch).unsqueeze(0) # (1, C, patch_size, patch_size)
                         # Extract features using the CNN
                         patch_features = self.regional_feature_extractor(resized_patch) # (1, C_final, H_f, W_f)
                         patch_features = patch_features.view(1, -1) # Flatten features (1, final_feature_dim_per_region)
                    batch_region_features.append(patch_features)
                all_regional_features.append(torch.cat(batch_region_features, dim=0)) # (NumReg, final_feature_dim_per_region)

            regional_features_tensor = torch.stack(all_regional_features) # (B, NumReg, final_feature_dim_per_region)

            # 3. Aggregate regional features (e.g., mean)
            aggregated_features = torch.mean(regional_features_tensor, dim=1) # (B, final_feature_dim_per_region)

            # 4. Final layer for real/fake prediction
            logits = self.final_layer(aggregated_features) # (B, 1)
            return logits

        else:
             raise NotImplementedError(f"Discriminator forward not implemented for type {self.disc_config.get('type')}")


# =====================================================================
# VAE-GAN Model Components
# =====================================================================

class WuBuGAADHybridGenNet(nn.Module):
    """Combines Encoder and Generator for VAE-GAN."""
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_appearance_config: Dict, gaad_motion_config: Optional[Dict], wubu_s_config: Dict, wubu_t_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_appearance_config = gaad_appearance_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_s_config = wubu_s_config
        self.wubu_t_config = wubu_t_config 
        self.wubu_m_config = wubu_m_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV01.MainNet") # Use instance logger

        self.latent_dim = args.latent_dim

        self.encoder = RegionalVAEEncoder(args, video_config, gaad_appearance_config, wubu_s_config, self.latent_dim)
        self.motion_encoder: Optional[RegionalHyperbolicMotionEncoder] = None
        if args.use_wubu_motion_branch:
             temp_motion_encoder = RegionalHyperbolicMotionEncoder(args, video_config, gaad_motion_config, wubu_m_config)
             if temp_motion_encoder.enabled: self.motion_encoder = temp_motion_encoder; self.logger.info("Motion Encoder Branch Activated (Optical Flow based).")
             else: self.logger.warning("Motion branch requested but disabled (check optical flow availability/config)."); args.use_wubu_motion_branch = False; self.wubu_m_config = None; self.gaad_motion_config = None; video_config['wubu_m_output_dim'] = 0

        self.generator = RegionalGeneratorDecoder(args, video_config, gaad_appearance_config, video_config['wubu_s_output_dim'], self.latent_dim)

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"WuBuGAADHybridGenNet Initialized: {param_count:,} params.")

    def encode(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        motion_features = None
        if self.motion_encoder is not None and self.motion_encoder.enabled:
            motion_output_tuple = self.motion_encoder(frames_pixels)
            if motion_output_tuple is not None:
                motion_features, _ = motion_output_tuple 

        mu, logvar, gaad_bboxes_all_frames, regional_app_features_tangent = self.encoder(frames_pixels, motion_features)
        return mu, logvar, gaad_bboxes_all_frames, regional_app_features_tangent # gaad_bboxes_all_frames has (B, total_sample_frames, NumReg, 4)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, gaad_bboxes_for_decode: torch.Tensor) -> torch.Tensor:
        # gaad_bboxes_for_decode should have shape (B, num_predict_frames, NumReg, 4)
        return self.generator(z, gaad_bboxes_for_decode)

    def forward(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, gaad_bboxes_all_input, _ = self.encode(frames_pixels)
        z = self.reparameterize(mu, logvar)
        
        num_input_f = self.video_config["num_input_frames"]
        num_predict_f = self.video_config["num_predict_frames"]
        
        # Ensure gaad_bboxes_all_input has enough frames to slice from
        expected_total_bbox_frames = frames_pixels.shape[1] # Should match total_frames_sample from dataset
        if gaad_bboxes_all_input.shape[1] != expected_total_bbox_frames :
             self.logger.warning(f"MainNet Forward: Encoder returned {gaad_bboxes_all_input.shape[1]} bbox sets, but input video had {expected_total_bbox_frames} frames. This might indicate an issue in encoder's bbox handling if it doesn't match input frame count.")
             # Attempt to proceed if possible, otherwise this is a critical error
        
        if gaad_bboxes_all_input.shape[1] < num_input_f + num_predict_f:
            self.logger.error(f"MainNet Forward: Not enough bbox sets from encoder ({gaad_bboxes_all_input.shape[1]}) for desired input ({num_input_f}) + pred ({num_predict_f}). Total expected from input: {expected_total_bbox_frames}.")
            # Fallback strategy:
            if gaad_bboxes_all_input.shape[1] >= num_predict_f:
                 self.logger.warning(f"Using last {num_predict_f} available bbox sets for decoder due to insufficient total.")
                 decoder_bboxes_selected = gaad_bboxes_all_input[:, -num_predict_f:, ...]
            elif gaad_bboxes_all_input.shape[1] > 0:
                 self.logger.warning(f"Using all {gaad_bboxes_all_input.shape[1]} available bbox sets for decoder as fewer than {num_predict_f} were available.")
                 decoder_bboxes_selected = gaad_bboxes_all_input 
                 # Note: Generator might produce fewer frames if it strictly follows bbox count.
                 # Or it might pad/error. The generator currently expects num_predict_frames bboxes.
                 # If this path is taken, RegionalGeneratorDecoder.forward needs to be robust to this.
            else:
                raise ValueError("MainNet Forward: Encoder returned no GAAD bboxes, cannot proceed with decoding.")
        else:
            decoder_bboxes_selected = gaad_bboxes_all_input[:, num_input_f : num_input_f + num_predict_f, ...]
        
        # Ensure decoder_bboxes_selected has num_predict_f in its time dimension for the generator
        if decoder_bboxes_selected.shape[1] != num_predict_f:
            self.logger.warning(f"MainNet Forward: After slicing, decoder_bboxes_selected has {decoder_bboxes_selected.shape[1]} frames, but generator expects {num_predict_f}. Padding/truncating bboxes for decoder.")
            if decoder_bboxes_selected.shape[1] < num_predict_f:
                if decoder_bboxes_selected.shape[1] == 0: # Should have been caught by earlier checks
                    raise ValueError("MainNet Forward: No bboxes to provide to decoder after slicing attempt.")
                num_to_pad = num_predict_f - decoder_bboxes_selected.shape[1]
                padding_slice = decoder_bboxes_selected[:, -1:, ...].repeat(1, num_to_pad, 1, 1) # Repeat last available bbox set
                decoder_bboxes_selected = torch.cat([decoder_bboxes_selected, padding_slice], dim=1)
            else: # decoder_bboxes_selected.shape[1] > num_predict_f (should not happen with correct slicing)
                decoder_bboxes_selected = decoder_bboxes_selected[:, :num_predict_f, ...]


        recon_frames = self.decode(z, decoder_bboxes_selected)
        return recon_frames, mu, logvar, decoder_bboxes_selected





# =====================================================================
# Dataset (Unchanged from Diffusion version)
# =====================================================================
class VideoFrameDataset(Dataset):
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int], frame_skip: int = 1, data_fraction: float = 1.0):
        super().__init__(); self.video_path = video_path; self.num_frames_total = num_frames_total; self.image_size = image_size; self.frame_skip = frame_skip; current_logger=logging.getLogger("WuBuGAADHybridGenV01.Dataset") # Updated logger
        if not os.path.isfile(self.video_path): current_logger.error(f"Video file not found: {self.video_path}"); raise FileNotFoundError(f"Video file not found: {self.video_path}")
        current_logger.info(f"Attempting to load entire video into RAM: {self.video_path}")
        self.video_frames_in_ram = None
        self.source_video_fps = 30.0 # Default

        read_success = False
        if VIDEO_IO_AVAILABLE and video_io is not None:
            try:
                video_data = video_io.read_video(self.video_path, output_format="TCHW", pts_unit="sec")
                self.video_frames_in_ram = video_data[0].contiguous()
                self.source_video_fps = video_data[2].get('video_fps', 30.0)
                read_success = True
            except Exception as e_tv:
                current_logger.warning(f"torchvision.io failed to read {self.video_path}: {e_tv}. Trying imageio...")
        else:
            current_logger.warning("torchvision.io not available. Trying imageio...")

        if not read_success and IMAGEIO_AVAILABLE and imageio is not None:
            try:
                reader = imageio.get_reader(self.video_path)
                meta = reader.get_meta_data()
                self.source_video_fps = meta.get('fps', 30.0)
                frames = []
                for frame_np in reader:
                    # Convert HWC (imageio default) to CHW
                    frame_th = torch.from_numpy(frame_np).permute(2, 0, 1)
                    frames.append(frame_th)
                self.video_frames_in_ram = torch.stack(frames).contiguous()
                reader.close()
                read_success = True
            except Exception as e_ii:
                current_logger.error(f"imageio also failed to read {self.video_path}: {e_ii}", exc_info=True)
                raise RuntimeError(f"Failed to load video '{self.video_path}' using both torchvision and imageio.") from e_ii
        elif not read_success:
             raise RuntimeError(f"Failed to load video '{self.video_path}'. Neither torchvision.io nor imageio could read it or are available.")

        ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3); current_logger.info(f"Loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. Est RAM: {ram_usage_gb:.2f} GB.")

        self.resize_transform = T.Resize(self.image_size, antialias=True); self.normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.num_disk_frames = self.video_frames_in_ram.shape[0]; self.samples = []; required_span_len = (self.num_frames_total - 1) * self.frame_skip + 1
        if self.num_disk_frames >= required_span_len:
            for i in range(self.num_disk_frames - required_span_len + 1): self.samples.append(i)
        else: current_logger.warning(f"Not enough frames ({self.num_disk_frames}) in video '{self.video_path}' for span {required_span_len} (total {self.num_frames_total}, skip {self.frame_skip}).")
        if data_fraction < 1.0 and len(self.samples) > 1: num_to_keep = max(1, int(len(self.samples) * data_fraction)); self.samples = random.sample(self.samples, num_to_keep); current_logger.info(f"Using {data_fraction*100:.2f}% of samples: {len(self.samples)} samples.")
        if not self.samples: current_logger.error(f"VideoFrameDataset: No valid samples. Frames: {self.num_disk_frames}, Total required: {self.num_frames_total}, Skip: {self.frame_skip}.")
        else: current_logger.info(f"VideoFrameDataset initialized (RAM). Frames: {self.num_disk_frames}. Samples: {len(self.samples)}. Sample len: {self.num_frames_total} (skip {self.frame_skip}).")
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> torch.Tensor:
        start_frame_idx_in_ram = self.samples[idx]; frames_for_sample = []
        for i in range(self.num_frames_total):
            actual_frame_idx_in_ram = start_frame_idx_in_ram + i * self.frame_skip
            if actual_frame_idx_in_ram < self.num_disk_frames:
                try: frame_tensor_chw_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram]; resized_frame_tensor = self.resize_transform(frame_tensor_chw_uint8); frame_float_01 = resized_frame_tensor.float() / 255.0; transformed_frame = self.normalize_transform(frame_float_01); frames_for_sample.append(transformed_frame)
                except Exception as e: logging.getLogger("WuBuGAADHybridGenV01.Dataset").error(f"Error transforming frame {actual_frame_idx_in_ram} for sample {idx}: {e}", exc_info=True); raise e
            else: logging.getLogger("WuBuGAADHybridGenV01.Dataset").error(f"Frame index {actual_frame_idx_in_ram} out of bounds (total: {self.num_disk_frames}). Sample: {idx}"); raise IndexError("Frame index out of bounds.")
        if len(frames_for_sample) != self.num_frames_total: logging.getLogger("WuBuGAADHybridGenV01.Dataset").error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total} for sample {idx}"); raise ValueError("Incorrect number of frames loaded for sample.")
        return torch.stack(frames_for_sample)

# =====================================================================
# VAE-GAN Trainer
# =====================================================================
class HybridTrainer:
    def __init__(self,
                 model: "WuBuGAADHybridGenNet", # Use string for forward reference
                 discriminator: "RegionalDiscriminator", # Use string
                 optimizer_enc_gen: torch.optim.Optimizer,
                 optimizer_disc: torch.optim.Optimizer,
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 args: argparse.Namespace,
                 rank: int,
                 world_size: int,
                 ddp_active: bool):

        self.model = model; self.discriminator = discriminator; self.optimizer_enc_gen = optimizer_enc_gen; self.optimizer_disc = optimizer_disc; self.device = device; self.train_loader = train_loader; self.val_loader = val_loader; self.args = args; self.rank = rank; self.world_size = world_size; self.ddp_active = ddp_active; self.am_main_process = (rank == 0); self.logger = logging.getLogger("WuBuGAADHybridGenV01.Trainer")
        self.video_config = model.video_config; self.gaad_appearance_config = model.gaad_appearance_config
        self.lambda_recon = args.lambda_recon; self.lambda_kl = args.lambda_kl; self.lambda_gan = args.lambda_gan
        self.scaler_enc_gen = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda'); self.scaler_disc = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.global_step = 0; self.current_epoch = 0; self.best_val_loss = float('inf'); self.last_val_metrics: Dict[str, Any] = {}
        if self.am_main_process: os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.lpips_loss_fn = None; self.ssim_metric = None
        if self.am_main_process and self.args.use_lpips_for_verification:
             if LPIPS_AVAILABLE and lpips is not None: self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device); self.logger.info("LPIPS metric enabled.")
             else: self.logger.warning("LPIPS lib unavailable. Skip LPIPS.")
        if self.am_main_process and TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
             try: self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device); self.logger.info("SSIM metric enabled.")
             except Exception as e: self.logger.warning(f"SSIM init failed: {e}. Skip SSIM."); self.ssim_metric = None
        elif self.am_main_process: self.logger.warning("torchmetrics SSIM unavailable. Skip SSIM.")
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        if self.grad_accum_steps > 1 and self.am_main_process: self.logger.info(f"Gradient accumulation enabled: {self.grad_accum_steps} steps.")

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1); return kl_div.mean()
    def _compute_recon_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_x, x)

    def _train_discriminator_step(self, real_frames_full: torch.Tensor, m_ref: "WuBuGAADHybridGenNet", d_ref: "RegionalDiscriminator") -> Dict[str, torch.Tensor]:
        B = real_frames_full.shape[0]; device = real_frames_full.device; dtype_model = next(m_ref.parameters()).dtype
        real_frames_disc = real_frames_full[:, 0:1, ...].to(device, dtype_model) # Discriminate first frame
        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model); fake_labels = torch.zeros(B, 1, device=device, dtype=dtype_model)
        losses_d = {};
        for p in d_ref.parameters(): p.requires_grad = True;
        for p in m_ref.parameters(): p.requires_grad = False
        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'):
            real_logits = d_ref(real_frames_disc); loss_d_real = self.adversarial_loss(real_logits, real_labels)
            with torch.no_grad():
                # Use m_ref.forward to get generated frames consistent with how G is trained
                # This will internally handle bbox selection for decoder
                fake_frames_full_sequence, _, _, _ = m_ref(real_frames_full)
                fake_frames_disc = fake_frames_full_sequence[:, 0:1, ...].to(device, dtype_model) # Discriminate first generated frame
            fake_logits = d_ref(fake_frames_disc.detach()); loss_d_fake = self.adversarial_loss(fake_logits, fake_labels)
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5; loss_d_total_scaled_micro = loss_d_total_micro / self.grad_accum_steps
        self.scaler_disc.scale(loss_d_total_scaled_micro).backward()
        losses_d['loss_d_real_micro'] = loss_d_real.detach(); losses_d['loss_d_fake_micro'] = loss_d_fake.detach(); losses_d['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d

    def _train_generator_step(self, real_frames_full: torch.Tensor, m_ref: "WuBuGAADHybridGenNet", d_ref: "RegionalDiscriminator") -> Dict[str, torch.Tensor]:
        B = real_frames_full.shape[0]; device = real_frames_full.device; dtype_model = next(m_ref.parameters()).dtype
        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model); losses_g = {}
        for p in d_ref.parameters(): p.requires_grad = False;
        for p in m_ref.parameters(): p.requires_grad = True
        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'):
            recon_frames_pred, mu, logvar, _ = m_ref(real_frames_full.to(device, dtype_model)) # m_ref.forward gives num_predict_frames
            start_idx_target = self.video_config["num_input_frames"]; end_idx_target = start_idx_target + self.video_config["num_predict_frames"]
            # Ensure target frames align with what recon_frames_pred represents
            if end_idx_target > real_frames_full.shape[1] or recon_frames_pred.shape[1] != self.video_config["num_predict_frames"]:
                self.logger.warning(f"_train_G: Frame mismatch. Real:{real_frames_full.shape[1]}, Recon:{recon_frames_pred.shape[1]}, ExpEnd:{end_idx_target}. Adjusting."); actual_pred_len = min(recon_frames_pred.shape[1], real_frames_full.shape[1] - start_idx_target, self.video_config["num_predict_frames"]);
                if actual_pred_len <= 0: self.logger.error(f"Cannot compute recon loss with {actual_pred_len} overlapping frames. Real: {real_frames_full.shape}, Recon: {recon_frames_pred.shape}, StartIdx: {start_idx_target}"); raise ValueError("Recon loss frame error.")
                target_frames_for_recon = real_frames_full[:, start_idx_target : start_idx_target + actual_pred_len, ...].to(device, dtype_model); recon_frames_for_loss = recon_frames_pred[:, :actual_pred_len, ...]
            else: target_frames_for_recon = real_frames_full[:, start_idx_target:end_idx_target, ...].to(device, dtype_model); recon_frames_for_loss = recon_frames_pred
            loss_recon = self._compute_recon_loss(recon_frames_for_loss, target_frames_for_recon); loss_kl = self._compute_kl_loss(mu, logvar)
            fake_frames_disc_for_gen = recon_frames_pred[:, 0:1, ...].to(device, dtype_model) # Discriminate first generated frame
            fake_logits_gen = d_ref(fake_frames_disc_for_gen); loss_g_adv = self.adversarial_loss(fake_logits_gen, real_labels)
            loss_g_total_micro = (self.lambda_recon * loss_recon + self.lambda_kl * loss_kl + self.lambda_gan * loss_g_adv); loss_g_total_scaled_micro = loss_g_total_micro / self.grad_accum_steps
        self.scaler_enc_gen.scale(loss_g_total_scaled_micro).backward()
        losses_g['loss_recon_micro'] = loss_recon.detach(); losses_g['loss_kl_micro'] = loss_kl.detach(); losses_g['loss_g_adv_micro'] = loss_g_adv.detach(); losses_g['loss_g_total_micro'] = loss_g_total_micro.detach()
        return losses_g

    def train(self, start_epoch:int=0, initial_global_step:int=0):
        self.global_step = initial_global_step; self.current_epoch = start_epoch
        if self.am_main_process: self.logger.info(f"Training Hybrid VAE-GAN from epoch {start_epoch}, step {initial_global_step}..."); self.logger.info(f"Grad Accum Steps: {self.grad_accum_steps}")
        accum_losses_log = defaultdict(float); items_interval_log = 0; accum_loss_g_q = 0.0; accum_loss_d_q = 0.0
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        d_ref = self.discriminator.module if self.ddp_active and isinstance(self.discriminator, DDP) else self.discriminator
        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch;
            if self.am_main_process: self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting...")
            if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler): self.train_loader.sampler.set_epoch(epoch)
            m_ref.train(); d_ref.train()
            dataset_len = 0;
            try: dataset_len = len(self.train_loader.sampler) if hasattr(self.train_loader.sampler,'__len__') else (len(self.train_loader.dataset)//self.world_size if hasattr(self.train_loader.dataset,'__len__') and self.world_size>0 else 0)
            except Exception: dataset_len = None
            total_micro_batches_estimate = math.ceil(dataset_len / (self.train_loader.batch_size or 1)) if dataset_len and dataset_len > 0 else None
            prog_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=not self.am_main_process or os.getenv('CI')=='true', dynamic_ncols=True, total=total_micro_batches_estimate)
            self.optimizer_disc.zero_grad(set_to_none=True); self.optimizer_enc_gen.zero_grad(set_to_none=True)

            for batch_idx, batch_frames_raw in enumerate(prog_bar):
                batch_frames = batch_frames_raw.to(self.device); batch_size_micro = batch_frames.size(0)
                losses_d_micro = self._train_discriminator_step(batch_frames, m_ref, d_ref)
                if torch.isfinite(losses_d_micro['loss_d_total_micro']): accum_loss_d_q += losses_d_micro['loss_d_total_micro'].item()
                for k,v_d in losses_d_micro.items():
                    if torch.isfinite(v_d): accum_losses_log[k.replace('_micro','')] += v_d.item() * batch_size_micro
                losses_g_micro = self._train_generator_step(batch_frames, m_ref, d_ref)
                if torch.isfinite(losses_g_micro['loss_g_total_micro']): accum_loss_g_q += losses_g_micro['loss_g_total_micro'].item()
                for k,v_g in losses_g_micro.items():
                    if torch.isfinite(v_g): accum_losses_log[k.replace('_micro','')] += v_g.item() * batch_size_micro
                items_interval_log += batch_size_micro

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.args.global_max_grad_norm > 0: self.scaler_disc.unscale_(self.optimizer_disc); torch.nn.utils.clip_grad_norm_(d_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_disc.step(self.optimizer_disc); self.scaler_disc.update(); self.optimizer_disc.zero_grad(set_to_none=True)
                    if self.args.global_max_grad_norm > 0: self.scaler_enc_gen.unscale_(self.optimizer_enc_gen); torch.nn.utils.clip_grad_norm_(m_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_enc_gen.step(self.optimizer_enc_gen); self.scaler_enc_gen.update(); self.optimizer_enc_gen.zero_grad(set_to_none=True)
                    self.global_step += 1

                    avg_loss_g_macro = accum_loss_g_q / self.grad_accum_steps if self.grad_accum_steps > 0 and np.isfinite(accum_loss_g_q) else None
                    avg_loss_d_macro = accum_loss_d_q / self.grad_accum_steps if self.grad_accum_steps > 0 and np.isfinite(accum_loss_d_q) else None
                    grad_norm_g_val, grad_norm_d_val = None, None
                    if hasattr(self.optimizer_enc_gen, 'grad_stats'): grad_norm_g_val = self.optimizer_enc_gen.grad_stats.max_grad_norm_observed
                    if hasattr(self.optimizer_disc, 'grad_stats'): grad_norm_d_val = self.optimizer_disc.grad_stats.max_grad_norm_observed
                    q_ctrl_gen = getattr(self.optimizer_enc_gen, 'q_controller', None)
                    if q_ctrl_gen:
                        q_lr_g = self.optimizer_enc_gen.param_groups[0]['lr']; q_mom_g = self.optimizer_enc_gen.param_groups[0]['momentum']; q_state_g = q_ctrl_gen.get_state(q_lr_g, q_mom_g, grad_norm_g_val, avg_loss_g_macro)
                        if q_ctrl_gen.prev_state is not None and q_ctrl_gen.prev_action is not None and q_ctrl_gen.prev_loss is not None and q_state_g is not None and avg_loss_g_macro is not None and avg_loss_d_macro is not None:
                            reward_g = q_ctrl_gen.compute_reward(avg_loss_g_macro, q_ctrl_gen.prev_loss, avg_loss_d_macro, grad_norm_g_val, True)
                            if np.isfinite(reward_g): q_ctrl_gen.update(q_ctrl_gen.prev_state, q_ctrl_gen.prev_action, reward_g, q_state_g)
                        q_ctrl_gen.prev_state = q_state_g; q_ctrl_gen.prev_action = q_ctrl_gen.choose_action(q_state_g) if q_state_g is not None else q_ctrl_gen.prev_action; q_ctrl_gen.prev_loss = avg_loss_g_macro
                    q_ctrl_disc = getattr(self.optimizer_disc, 'q_controller', None)
                    if q_ctrl_disc:
                        q_lr_d = self.optimizer_disc.param_groups[0]['lr']; q_mom_d = self.optimizer_disc.param_groups[0]['momentum']; q_state_d = q_ctrl_disc.get_state(q_lr_d, q_mom_d, grad_norm_d_val, avg_loss_d_macro)
                        if q_ctrl_disc.prev_state is not None and q_ctrl_disc.prev_action is not None and q_ctrl_disc.prev_loss is not None and q_state_d is not None and avg_loss_d_macro is not None and avg_loss_g_macro is not None:
                            reward_d = q_ctrl_disc.compute_reward(avg_loss_d_macro, q_ctrl_disc.prev_loss, avg_loss_g_macro, grad_norm_d_val, False)
                            if np.isfinite(reward_d): q_ctrl_disc.update(q_ctrl_disc.prev_state, q_ctrl_disc.prev_action, reward_d, q_state_d)
                        q_ctrl_disc.prev_state = q_state_d; q_ctrl_disc.prev_action = q_ctrl_disc.choose_action(q_state_d) if q_state_d is not None else q_ctrl_disc.prev_action; q_ctrl_disc.prev_loss = avg_loss_d_macro
                    accum_loss_g_q = 0.0; accum_loss_d_q = 0.0

                    if self.global_step % self.args.log_interval == 0 and items_interval_log > 0:
                        log_metrics = {f"train/{k.replace('_micro','')}": v / items_interval_log for k, v in accum_losses_log.items()}
                        if self.args.log_grad_norm:
                            if grad_norm_g_val is not None: log_metrics["train/grad_norm_gen_max_obs"] = grad_norm_g_val
                            if grad_norm_d_val is not None: log_metrics["train/grad_norm_disc_max_obs"] = grad_norm_d_val
                        lr_g = self.optimizer_enc_gen.param_groups[0]['lr']; lr_d = self.optimizer_disc.param_groups[0]['lr']
                        log_metrics["train/lr_gen"] = lr_g; log_metrics["train/lr_disc"] = lr_d; log_metrics["epoch_frac"] = epoch + ((batch_idx + 1) / total_micro_batches_estimate if total_micro_batches_estimate and total_micro_batches_estimate > 0 else 0); log_metrics["global_step"] = self.global_step
                        if q_ctrl_gen: log_metrics.update({f"q_ctrl_gen/{k.replace('_', '')}":v for k,v in q_ctrl_gen.get_info().items()})
                        if q_ctrl_disc: log_metrics.update({f"q_ctrl_disc/{k.replace('_', '')}":v for k,v in q_ctrl_disc.get_info().items()})
                        if self.am_main_process:
                            g_total = log_metrics.get('train/loss_g_total', -1); d_total = log_metrics.get('train/loss_d_total', -1)
                            g_recon = log_metrics.get('train/loss_recon', -1); g_kl = log_metrics.get('train/loss_kl',-1); g_adv = log_metrics.get('train/loss_g_adv',-1)
                            d_real = log_metrics.get('train/loss_d_real',-1); d_fake = log_metrics.get('train/loss_d_fake',-1)
                            q_eps_g = log_metrics.get('q_ctrl_gen/epsilon',-1.0); q_eps_d = log_metrics.get('q_ctrl_disc/epsilon',-1.0)
                            q_last_act_g = log_metrics.get('q_ctrl_gen/lastaction', {}); q_last_act_d = log_metrics.get('q_ctrl_disc/lastaction', {})
                            q_g_lr_s = q_last_act_g.get('lr_scale', 1.0) if q_last_act_g else 1.0
                            q_d_lr_s = q_last_act_d.get('lr_scale', 1.0) if q_last_act_d else 1.0


                            log_str = (f"E {epoch+1} S{self.global_step} | G_tot:{g_total:.3f} (Rec:{g_recon:.3f} KL:{g_kl:.4f} Adv:{g_adv:.3f}) | "
                                       f"D_tot:{d_total:.3f} (Real:{d_real:.3f} Fake:{d_fake:.3f}) | "
                                       f"LR(G/D):{lr_g:.1e}/{lr_d:.1e} | Q_Eps(G/D):{q_eps_g:.2f}/{q_eps_d:.2f} | Q_Scale(G/D):{q_g_lr_s:.2f}/{q_d_lr_s:.2f}")
                            prog_bar.set_postfix_str(f"G:{g_total:.2f} D:{d_total:.2f} LR(G/D):{lr_g:.1e}/{lr_d:.1e}", refresh=True)
                            self.logger.info(log_str)
                            if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(log_metrics, step=self.global_step)
                        accum_losses_log = defaultdict(float); items_interval_log = 0

                if self.args.save_interval > 0 and self.global_step > 0 and (self.global_step % self.args.save_interval == 0) and ((batch_idx + 1) % self.grad_accum_steps == 0) and self.am_main_process:
                    inter_metrics = {"train_loss_g_total": losses_g_micro.get('loss_g_total_micro', torch.tensor(0.0)).item(), "train_loss_d_total": losses_d_micro.get('loss_d_total_micro', torch.tensor(0.0)).item()}; self._save_checkpoint(is_intermediate=True, metrics=inter_metrics)

            # End of Epoch
            if (batch_idx + 1) % self.grad_accum_steps != 0: # Handle final incomplete cycle
                self.logger.info(f"Performing final optimizer step for epoch {epoch+1} (incomplete cycle).");
                if self.args.global_max_grad_norm > 0: self.scaler_disc.unscale_(self.optimizer_disc); torch.nn.utils.clip_grad_norm_(d_ref.parameters(), self.args.global_max_grad_norm)
                self.scaler_disc.step(self.optimizer_disc); self.scaler_disc.update();
                if self.args.global_max_grad_norm > 0: self.scaler_enc_gen.unscale_(self.optimizer_enc_gen); torch.nn.utils.clip_grad_norm_(m_ref.parameters(), self.args.global_max_grad_norm)
                self.scaler_enc_gen.step(self.optimizer_enc_gen); self.scaler_enc_gen.update();
                self.global_step += 1;

            if self.am_main_process:
                 final_items = items_interval_log if items_interval_log > 0 else (batch_idx+1) % self.grad_accum_steps if (batch_idx+1) % self.grad_accum_steps != 0 else self.grad_accum_steps
                 avg_g = (accum_losses_log['loss_g_total']/final_items if items_interval_log > 0 else accum_loss_g_q/final_items) if final_items > 0 else float('nan')
                 avg_d = (accum_losses_log['loss_d_total']/final_items if items_interval_log > 0 else accum_loss_d_q/final_items) if final_items > 0 else float('nan')
                 self.logger.info(f"Epoch {epoch+1} finished. Final Avg Loss G:{avg_g:.4f}, D:{avg_d:.4f}")
                 if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log({"epoch": epoch+1, "epoch_avg_train_loss_g": avg_g, "epoch_avg_train_loss_d": avg_d}, step=self.global_step)

            if self.val_loader and self.am_main_process:
                val_metrics = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if val_metrics:
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log({f"val/{k}":v for k,v in val_metrics.items()}, step=self.global_step)
                    current_val_metric = val_metrics.get(self.args.val_primary_metric, float('inf'))
                    if current_val_metric < self.best_val_loss: self.logger.info(f"New best val metric ({self.args.val_primary_metric}): {current_val_metric:.4f}. Save best."); self.best_val_loss = current_val_metric; self._save_checkpoint(is_best=True, metrics=val_metrics)
            if self.am_main_process:
                save_metrics = self.last_val_metrics.copy() if self.last_val_metrics else {}; save_metrics["epoch_end_train_loss_g_avg"] = avg_g if np.isfinite(avg_g) else -1.0; save_metrics["epoch_end_train_loss_d_avg"] = avg_d if np.isfinite(avg_d) else -1.0; self._save_checkpoint(metrics=save_metrics)

    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Optional[Dict[str, float]]:
        if not self.val_loader or not self.am_main_process: return None
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; d_ref = self.discriminator.module if self.ddp_active and isinstance(self.discriminator, DDP) else self.discriminator
        m_ref.eval(); d_ref.eval()
        total_recon_loss_sum = 0.0; total_psnr_sum = 0.0; total_ssim_sum = 0.0; total_lpips_sum = 0.0; total_compared_frames = 0
        logged_samples_count = 0; wandb_val_samples = []; dtype_model = next(m_ref.parameters()).dtype
        for batch_idx, batch_frames_raw in enumerate(tqdm(self.val_loader, desc="Validating", dynamic_ncols=True, disable=os.getenv('CI')=='true' or not self.am_main_process)):
            batch_frames = batch_frames_raw.to(self.device); real_frames_full = batch_frames.to(self.device, dtype_model); B_val = real_frames_full.shape[0]
            recon_frames_full, mu, logvar, _ = m_ref(real_frames_full)
            num_cond = self.video_config["num_input_frames"]; num_pred = self.video_config["num_predict_frames"]
            available_pred_len = recon_frames_full.shape[1]; available_gt_len = real_frames_full.shape[1] - num_cond
            compare_len = min(available_pred_len, available_gt_len, num_pred)
            if compare_len <=0: self.logger.warning(f"Val: Skipping batch, cannot compare frames. AvailPred:{available_pred_len}, AvailGT:{available_gt_len}, NumPred:{num_pred}"); continue
            pred_for_metrics = recon_frames_full[:, :compare_len, ...]; gt_for_metrics = real_frames_full[:, num_cond : num_cond + compare_len, ...]
            pred_norm = (pred_for_metrics.clamp(-1, 1) + 1) / 2.0; gt_norm = (gt_for_metrics.clamp(-1, 1) + 1) / 2.0
            pred_norm_flat = pred_norm.reshape(-1, pred_norm.shape[-3], pred_norm.shape[-2], pred_norm.shape[-1]); gt_norm_flat = gt_norm.reshape(-1, gt_norm.shape[-3], gt_norm.shape[-2], gt_norm.shape[-1])
            current_batch_compared_frames = pred_norm_flat.shape[0] # Number of (C,H,W) frames in this flat batch
            
            recon_loss_val_sum_pixels = F.mse_loss(pred_norm_flat, gt_norm_flat, reduction='sum')
            if torch.isfinite(recon_loss_val_sum_pixels):
                total_recon_loss_sum += recon_loss_val_sum_pixels.item()
                avg_mse_per_pixel_this_batch = recon_loss_val_sum_pixels.item() / (pred_norm_flat.numel() + EPS)
                psnr_val = 10 * torch.log10(1.0 / (avg_mse_per_pixel_this_batch + EPS))
                total_psnr_sum += psnr_val * current_batch_compared_frames # Sum of PSNRs (weighted by num frames)
            else: self.logger.warning("Non-finite val recon loss.")

            if self.ssim_metric:
                try: ssim_val_batch_frames = self.ssim_metric(pred_norm_flat, gt_norm_flat); # Returns per-frame SSIM
                except Exception as e: self.logger.warning(f"SSIM failed: {e}"); ssim_val_batch_frames=torch.zeros(current_batch_compared_frames, device=self.device);
                if torch.isfinite(ssim_val_batch_frames).all(): total_ssim_sum += ssim_val_batch_frames.sum().item()

            if self.lpips_loss_fn:
                try: lpips_val_batch_frames = self.lpips_loss_fn(pred_norm_flat*2-1, gt_norm_flat*2-1) # LPIPS expects [-1,1]
                except Exception as e: self.logger.warning(f"LPIPS failed: {e}"); lpips_val_batch_frames=torch.zeros(current_batch_compared_frames,1,1,1, device=self.device);
                if torch.isfinite(lpips_val_batch_frames).all(): total_lpips_sum += lpips_val_batch_frames.sum().item()

            total_compared_frames += current_batch_compared_frames

            if self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run and logged_samples_count < num_val_samples_to_log:
                # ... (WandB logging for samples, unchanged) ...
                num_log_batch = min(B_val, num_val_samples_to_log - logged_samples_count)
                for k_idx in range(num_log_batch):
                    sample_imgs_wandb = []
                    for frame_c_idx in range(min(num_cond, real_frames_full.shape[1])): sample_imgs_wandb.append(wandb.Image(real_frames_full[k_idx, frame_c_idx].cpu().float().clamp(-1, 1) * 0.5 + 0.5, caption=f"ValCond {frame_c_idx} S{k_idx}"))
                    for frame_p_idx in range(min(num_pred, gt_for_metrics.shape[1])): sample_imgs_wandb.append(wandb.Image(gt_for_metrics[k_idx, frame_p_idx].cpu().float().clamp(-1, 1) * 0.5 + 0.5, caption=f"ValGT {frame_p_idx} S{k_idx}"))
                    for frame_p_idx in range(min(num_pred, pred_for_metrics.shape[1])): sample_imgs_wandb.append(wandb.Image(pred_for_metrics[k_idx, frame_p_idx].cpu().float().clamp(-1, 1) * 0.5 + 0.5, caption=f"ValRecon {frame_p_idx} S{k_idx}"))
                    wandb_val_samples.extend(sample_imgs_wandb)
                logged_samples_count += num_log_batch

        m_ref.train(); d_ref.train()
        avg_recon_loss_per_pixel = total_recon_loss_sum / (total_compared_frames * pred_norm_flat.shape[-3:].numel() + EPS) if total_compared_frames > 0 else float('inf')
        avg_psnr = total_psnr_sum / total_compared_frames if total_compared_frames > 0 else 0.0
        avg_ssim = total_ssim_sum / total_compared_frames if total_compared_frames > 0 and self.ssim_metric else 0.0
        avg_lpips = total_lpips_sum / total_compared_frames if total_compared_frames > 0 and self.lpips_loss_fn else 0.0
        metrics = {"avg_val_recon_mse": avg_recon_loss_per_pixel, "avg_val_psnr": avg_psnr, "avg_val_ssim": avg_ssim, "avg_val_lpips": avg_lpips}; self.last_val_metrics = metrics; self.logger.info(f"Validation Metrics: ReconMSE:{avg_recon_loss_per_pixel:.4f}, PSNR:{avg_psnr:.2f}, SSIM:{avg_ssim:.4f}, LPIPS:{avg_lpips:.4f}")
        if wandb_val_samples and self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log({"validation_samples_sequence": wandb_val_samples}, step=self.global_step)
        return metrics

    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict]=None, is_best:bool=False):
        if not self.am_main_process: return
        m_save = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; d_save = self.discriminator.module if self.ddp_active and isinstance(self.discriminator, DDP) else self.discriminator
        data = {'global_step': self.global_step, 'epoch': self.current_epoch, 'model_state_dict': m_save.state_dict(), 'discriminator_state_dict': d_save.state_dict(), 'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict(), 'optimizer_disc_state_dict': self.optimizer_disc.state_dict(), 'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict() if self.args.use_amp and self.device.type == 'cuda' else None, 'scaler_disc_state_dict': self.scaler_disc.state_dict() if self.args.use_amp and self.device.type == 'cuda' else None, 'args': vars(self.args), 'metrics': metrics if metrics else self.last_val_metrics, 'video_config': self.video_config, 'best_val_loss': self.best_val_loss,}
        q_ctrl_gen = getattr(self.optimizer_enc_gen, 'q_controller', None); q_ctrl_disc = getattr(self.optimizer_disc, 'q_controller', None)
        if q_ctrl_gen: data['q_controller_enc_gen_state'] = q_ctrl_gen.__dict__ # Save full controller state
        if q_ctrl_disc: data['q_controller_disc_state'] = q_ctrl_disc.__dict__
        fname_prefix="wubugaad_hybridgen_ckpt_v01"; fpath=""
        if is_best: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_best.pt")
        elif is_intermediate: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_step{self.global_step}.pt")
        else: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_ep{self.current_epoch+1}_step{self.global_step}.pt")
        try: torch.save(data, fpath); self.logger.info(f"Checkpoint saved: {os.path.basename(fpath)}")
        except Exception as e: self.logger.error(f"Save checkpoint error {fpath}: {e}", exc_info=True)

    def load_checkpoint(self, checkpoint_path:str) -> Tuple[int,int]:
        if not os.path.exists(checkpoint_path): self.logger.warning(f"Checkpoint {checkpoint_path} not found."); return 0,0
        try: ckpt = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e_load: self.logger.error(f"Failed load ckpt {checkpoint_path}: {e_load}"); return 0,0
        m_load = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; d_load = self.discriminator.module if self.ddp_active and isinstance(self.discriminator, DDP) else self.discriminator
        try: m_load.load_state_dict(ckpt['model_state_dict'], strict=False); self.logger.info("Loaded model state_dict.")
        except Exception as e: self.logger.error(f"Error loading model state_dict: {e}")
        try: d_load.load_state_dict(ckpt['discriminator_state_dict'], strict=False); self.logger.info("Loaded discriminator state_dict.")
        except Exception as e: self.logger.error(f"Error loading discriminator state_dict: {e}")
        if 'optimizer_enc_gen_state_dict' in ckpt and self.optimizer_enc_gen:
            try: self.optimizer_enc_gen.load_state_dict(ckpt['optimizer_enc_gen_state_dict']); self.logger.info("Loaded enc/gen optimizer state.")
            except Exception as e: self.logger.warning(f"Could not load Enc/Gen optimizer state: {e}")
        if 'optimizer_disc_state_dict' in ckpt and self.optimizer_disc:
             try: self.optimizer_disc.load_state_dict(ckpt['optimizer_disc_state_dict']); self.logger.info("Loaded disc optimizer state.")
             except Exception as e: self.logger.warning(f"Could not load Disc optimizer state: {e}")
        if 'scaler_enc_gen_state_dict' in ckpt and self.scaler_enc_gen and ckpt['scaler_enc_gen_state_dict'] is not None: self.scaler_enc_gen.load_state_dict(ckpt['scaler_enc_gen_state_dict'])
        if 'scaler_disc_state_dict' in ckpt and self.scaler_disc and ckpt['scaler_disc_state_dict'] is not None: self.scaler_disc.load_state_dict(ckpt['scaler_disc_state_dict'])
        q_ctrl_gen = getattr(self.optimizer_enc_gen, 'q_controller', None); q_ctrl_disc = getattr(self.optimizer_disc, 'q_controller', None)
        if q_ctrl_gen and 'q_controller_enc_gen_state' in ckpt and isinstance(ckpt['q_controller_enc_gen_state'], dict) :
            try: q_ctrl_gen.__dict__.update(ckpt['q_controller_enc_gen_state']); self.logger.info("Loaded Q-Ctrl Gen state.")
            except Exception as e: self.logger.warning(f"Could not load Q-controller state for Enc/Gen: {e}")
        if q_ctrl_disc and 'q_controller_disc_state' in ckpt and isinstance(ckpt['q_controller_disc_state'], dict):
            try: q_ctrl_disc.__dict__.update(ckpt['q_controller_disc_state']); self.logger.info("Loaded Q-Ctrl Disc state.")
            except Exception as e: self.logger.warning(f"Could not load Q-controller state for Disc: {e}")
        loaded_global_step = ckpt.get('global_step', 0); loaded_epoch = ckpt.get('epoch', 0); self.best_val_loss = ckpt.get('best_val_loss', ckpt.get('metrics', {}).get(self.args.val_primary_metric, float('inf')))
        self.logger.info(f"Loaded checkpoint {checkpoint_path} (Step {loaded_global_step}, Ep {loaded_epoch}). BestValLoss ({self.args.val_primary_metric}): {self.best_val_loss:.4f}")
        return loaded_global_step, loaded_epoch

    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; m_ref.eval(); device = self.device; dtype = next(m_ref.parameters()).dtype; latent_dim = self.args.latent_dim
        if noise is None: z = torch.randn(num_samples, latent_dim, device=device, dtype=dtype)
        else: z = noise.to(device, dtype);
        if z.shape[0] != num_samples: self.logger.warning(f"Noise BS {z.shape[0]} != num_samples {num_samples}. Use noise BS."); num_samples = z.shape[0]
        num_pred = self.video_config["num_predict_frames"]; num_regions = self.gaad_appearance_config["num_regions"]; frame_dims = (self.args.image_w, self.args.image_h); decoder_bboxes_list = []
        for _ in range(num_samples):
            bboxes_single_sample_all_frames = []; current_frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims, num_regions, device=device, dtype=dtype, min_size_px=self.args.gaad_min_size_px)
            for _ in range(num_pred): bboxes_single_sample_all_frames.append(current_frame_bboxes)
            decoder_bboxes_list.append(torch.stack(bboxes_single_sample_all_frames))
        decoder_bboxes_batch = torch.stack(decoder_bboxes_list)
        self.logger.info(f"Sampling {num_samples} sequences from prior noise..."); generated_frames = m_ref.decode(z, decoder_bboxes_batch); self.logger.info("Sampling finished.")
        return generated_frames





# =====================================================================
# Arg Parsing and Main Execution Logic
# =====================================================================
def seed_worker_init_fn(worker_id, base_seed, rank, world_size):
     worker_seed = base_seed + worker_id + rank * world_size
     random.seed(worker_seed); np.random.seed(worker_seed); torch.manual_seed(worker_seed)

def seed_everything(seed:int,rank:int=0,world_size:int=1):
    actual_seed = seed + rank; random.seed(actual_seed); np.random.seed(actual_seed); torch.manual_seed(actual_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(actual_seed)


# Function remains the same as it configures the WuBu stack based on args
def _configure_wubu_stack(args: argparse.Namespace, prefix: str) -> Optional[Dict]:
    if prefix == "wubu_m" and not (args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE): return None # Check flow avail too
    config = DEFAULT_CONFIG_WUBU.copy(); num_levels_val = getattr(args, f"{prefix}_num_levels", 0); config["num_levels"] = num_levels_val
    if num_levels_val == 0:
        for key in ["hyperbolic_dims", "initial_curvatures", "initial_scales", "initial_spread_values", "boundary_points_per_level", "transform_types", "transform_hidden_dims", "tangent_input_combination_dims"]: config[key] = [] if key not in ["tangent_input_combination_dims"] else [DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0]]
        return config
    config["hyperbolic_dims"] = getattr(args, f"{prefix}_hyperbolic_dims", DEFAULT_CONFIG_WUBU["hyperbolic_dims"]); config["initial_curvatures"] = getattr(args, f"{prefix}_initial_curvatures", DEFAULT_CONFIG_WUBU["initial_curvatures"]); config["use_rotation_in_transform"] = getattr(args, f"{prefix}_use_rotation", DEFAULT_CONFIG_WUBU["use_rotation_in_transform"]); config["phi_influence_curvature"] = getattr(args, f"{prefix}_phi_influence_curvature", DEFAULT_CONFIG_WUBU["phi_influence_curvature"]); config["phi_influence_rotation_init"] = getattr(args, f"{prefix}_phi_influence_rotation_init", DEFAULT_CONFIG_WUBU["phi_influence_rotation_init"]); config["dropout"] = args.wubu_dropout
    def _ensure_list_len(cfg_dict, key, target_len, default_fill_list):
        current_val = cfg_dict.get(key, []); is_list_orig = isinstance(current_val, list); current_list_val = current_val if is_list_orig else [current_val]
        base_default = default_fill_list[0] if default_fill_list else (1.0 if "scales" in key or "curvatures" in key else (0.1 if "spread" in key else ("linear" if "types" in key else 32))); fill_val = current_list_val[-1] if current_list_val else base_default
        if len(current_list_val) < target_len: cfg_dict[key] = (current_list_val + [fill_val]*(target_len-len(current_list_val)))[:target_len]
        elif len(current_list_val) > target_len: cfg_dict[key] = current_list_val[:target_len]
        if not is_list_orig and target_len == 1 and isinstance(cfg_dict[key], list): cfg_dict[key] = cfg_dict[key][0]
    for key_chk, default_key in [("hyperbolic_dims", "hyperbolic_dims"), ("initial_curvatures", "initial_curvatures"), ("initial_scales", "initial_scales"), ("initial_spread_values", "initial_spread_values"), ("boundary_points_per_level", "boundary_points_per_level")]: _ensure_list_len(config, key_chk, num_levels_val, DEFAULT_CONFIG_WUBU[default_key])
    if not isinstance(config.get("tangent_input_combination_dims"), list): config["tangent_input_combination_dims"] = [config.get("tangent_input_combination_dims", DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0])]
    num_transitions = max(0, num_levels_val-1)
    if num_transitions > 0: _ensure_list_len(config,"transform_types",num_transitions,DEFAULT_CONFIG_WUBU["transform_types"]); _ensure_list_len(config,"transform_hidden_dims",num_transitions,DEFAULT_CONFIG_WUBU["transform_hidden_dims"])
    else: config["transform_types"]=[]; config["transform_hidden_dims"]=[]
    return config


def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu-GAAD Regional VAE-GAN w/ Optical Flow (v0.1)")
    # --- Data and DDP ---
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_dir")
    parser.add_argument('--local_rank', type=int, default=-1); parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2); parser.add_argument('--image_h', type=int, default=64)
    parser.add_argument('--image_w', type=int, default=64); parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_input_frames', type=int, default=3); parser.add_argument('--num_predict_frames', type=int, default=1) # Num frames Generator outputs
    parser.add_argument('--frame_skip', type=int, default=1); parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--num_workers',type=int, default=0); parser.add_argument('--checkpoint_dir',type=str, default='wubugaad_hybridgen_checkpoints_v01') # Updated dir
    parser.add_argument('--load_checkpoint', type=str, default=None); parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--wandb_project',type=str,default='WuBuGAADHybridGenV01') # Updated project
    parser.add_argument('--wandb_run_name',type=str,default=None); parser.add_argument('--log_interval',type=int, default=20)
    parser.add_argument('--save_interval',type=int, default=500)

    # --- GAAD ---
    parser.add_argument('--gaad_num_regions', type=int, default=16); parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])
    parser.add_argument('--gaad_min_size_px', type=int, default=4)

    # --- Motion Branch ---
    parser.add_argument('--use_wubu_motion_branch', action='store_true', help="Enable GAAD+WuBu-M+OpticalFlow motion branch.")
    parser.add_argument('--gaad_motion_num_regions', type=int, default=12)
    parser.add_argument('--gaad_motion_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="Spatial layout for motion regions.")
    parser.add_argument('--optical_flow_net_type', type=str, default='raft_small', choices=list(FLOW_MODELS.keys()) if OPTICAL_FLOW_AVAILABLE else [], help="Type of optical flow network from torchvision.")
    parser.add_argument('--freeze_flow_net', action='store_true', help="Freeze weights of the pre-trained optical flow network.")
    parser.add_argument('--flow_stats_components', nargs='+', type=str, default=['mag_mean', 'angle_mean'], help="Flow statistics to embed (mag_mean, angle_mean, mag_std, angle_std). 'angle_mean' uses cos/sin.")

    # --- Encoder/Generator Architecture ---
    parser.add_argument('--latent_dim', type=int, default=128, help="Dimensionality of the VAE latent space.")
    parser.add_argument('--encoder_use_roi_align', action='store_true'); parser.add_argument('--encoder_shallow_cnn_channels', type=int, default=32)
    parser.add_argument('--encoder_roi_align_output_h', type=int, default=4); parser.add_argument('--encoder_roi_align_output_w', type=int, default=4)
    parser.add_argument('--encoder_pixel_patch_size', type=int, default=16); parser.add_argument('--encoder_initial_tangent_dim', type=int, default=128) # Input dim to WuBu-S
    parser.add_argument('--decoder_type', type=str, default="patch_gen", choices=["patch_gen", "transformer"]); parser.add_argument('--decoder_patch_gen_size', type=int, default=16)
    parser.add_argument('--decoder_patch_resize_mode', type=str, default="bilinear", choices=["bilinear", "nearest"])

    # --- Discriminator Architecture ---
    parser.add_argument('--discriminator_type', type=str, default="regional_cnn", choices=["regional_cnn", "wubu_regional"], help="Type of discriminator architecture.")
    # Add args for specific discriminator types if needed (e.g., --disc_cnn_channels)

    # --- WuBu Stacks ---
    parser.add_argument('--wubu_dropout', type=float, default=0.1)
    # WuBu-S (Encoder Appearance)
    parser.add_argument('--wubu_s_num_levels', type=int, default=2); parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32]); parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8]); parser.add_argument('--wubu_s_use_rotation', action='store_true'); parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true')
    # WuBu-M (Encoder Motion)
    parser.add_argument('--wubu_m_num_levels', type=int, default=2); parser.add_argument('--wubu_m_hyperbolic_dims', nargs='+', type=int, default=[64,32]); parser.add_argument('--wubu_m_initial_curvatures', nargs='+', type=float, default=[1.0, 0.7]); parser.add_argument('--wubu_m_use_rotation', action='store_true'); parser.add_argument('--wubu_m_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_m_phi_influence_rotation_init', action='store_true')
    # WuBu-T (Encoder Temporal -> Latent)
    parser.add_argument('--wubu_t_num_levels', type=int, default=2); parser.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[128,64]); parser.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[1.0,0.5]); parser.add_argument('--wubu_t_use_rotation', action='store_true'); parser.add_argument('--wubu_t_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true')
    # Output dims (determined by WuBu stacks or projection layers)
    parser.add_argument('--wubu_s_output_dim', type=int, default=32, help="Output dim of WuBu-S stack (input to WuBu-T or latent projection).");
    parser.add_argument('--wubu_m_output_dim', type=int, default=32, help="Output dim of WuBu-M stack (input to WuBu-T or latent projection).");
    # wubu_t_output_dim is intermediate before latent projection

    # --- Training ---
    parser.add_argument('--lambda_recon', type=float, default=1.0, help="Weight for VAE reconstruction loss.")
    parser.add_argument('--lambda_kl', type=float, default=0.01, help="Weight for VAE KL divergence loss.")
    parser.add_argument('--lambda_gan', type=float, default=0.1, help="Weight for GAN adversarial loss (Generator part).")
    parser.add_argument('--learning_rate_gen',type=float,default=2e-4); # Separate LRs common in GANs
    parser.add_argument('--learning_rate_disc',type=float,default=2e-4);
    parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0); parser.add_argument('--global_max_grad_norm',type=float,default=1.0);
    parser.add_argument('--q_controller_enabled',action='store_true'); # Apply to both optimizers? Needs thought.
    parser.add_argument('--grad_accum_steps',type=int, default=1, help="Number of steps to accumulate gradients before optimizer step.")
    parser.add_argument('--use_amp', action='store_true'); parser.add_argument('--detect_anomaly',action='store_true')
    parser.add_argument('--log_grad_norm', action='store_true', help="Log gradient norms (can slow down training).")

    # --- Validation & Sampling ---
    parser.add_argument('--use_lpips_for_verification', action='store_true'); parser.add_argument('--validation_video_path', type=str, default=None);
    parser.add_argument('--validation_split_fraction', type=float, default=0.1);
    parser.add_argument('--val_block_size', type=int, default=20, help="Number of consecutive samples to include in each validation block.") # Added argument
    parser.add_argument('--val_primary_metric', type=str, default="avg_val_recon_mse", choices=["avg_val_recon_mse", "avg_val_psnr", "avg_val_ssim", "avg_val_lpips"]);
    parser.add_argument('--num_val_samples_to_log', type=int, default=2);
    parser.add_argument('--demo_num_samples', type=int, default=4, help="Number of samples to generate in demo.")

    parsed_args = parser.parse_args()

    # --- Argument Validation & Post-processing ---
    if parsed_args.use_wubu_motion_branch and not OPTICAL_FLOW_AVAILABLE: parser.error("Motion branch (--use_wubu_motion_branch) needs optical flow, but torchvision.models.optical_flow unavailable.")
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.optical_flow_net_type not in FLOW_MODELS: parser.error(f"Optical flow net type '{parsed_args.optical_flow_net_type}' not in available: {list(FLOW_MODELS.keys())}")
    def validate_wubu_config(args_obj, prefix_str, parser_ref, is_motion_branch_active):
        num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0); is_motion = prefix_str == "wubu_m"
        if num_levels > 0 and (not is_motion or is_motion_branch_active):
            for suffix, attr_name in [("hyperbolic_dims", f"{prefix_str}_hyperbolic_dims"), ("initial_curvatures", f"{prefix_str}_initial_curvatures")]:
                val_list = getattr(args_obj, attr_name); is_list=isinstance(val_list, list); val_list = val_list if is_list else [val_list]
                if len(val_list) != num_levels:
                    if len(val_list) == 1 and num_levels > 1: setattr(args_obj, attr_name, [val_list[0]] * num_levels)
                    elif not val_list and suffix=="initial_curvatures": setattr(args_obj, attr_name, [1.0] * num_levels)
                    else: parser_ref.error(f"{prefix_str}: Length mismatch {attr_name} ({len(val_list)}) vs num_levels ({num_levels})")
    validate_wubu_config(parsed_args, "wubu_s", parser, True); validate_wubu_config(parsed_args, "wubu_t", parser, True); validate_wubu_config(parsed_args, "wubu_m", parser, parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE)

    # Update output dims based on actual WuBu configs used
    # If WuBu-S has levels, its output dim is the last level dim. Otherwise, it's the input embed dim.
    if parsed_args.wubu_s_num_levels > 0 and parsed_args.wubu_s_hyperbolic_dims:
         parsed_args.wubu_s_output_dim = parsed_args.wubu_s_hyperbolic_dims[-1]
    else:
         parsed_args.wubu_s_output_dim = parsed_args.encoder_initial_tangent_dim
         if parsed_args.wubu_s_num_levels > 0: parsed_args.wubu_s_num_levels = 0 # Correct config if dims missing

    # Same logic for WuBu-M
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.wubu_m_num_levels > 0 and parsed_args.wubu_m_hyperbolic_dims:
         parsed_args.wubu_m_output_dim = parsed_args.wubu_m_hyperbolic_dims[-1]
    else:
         parsed_args.wubu_m_output_dim = 0 # No motion features if branch disabled or no levels/dims
         if parsed_args.wubu_m_num_levels > 0: parsed_args.wubu_m_num_levels = 0

    # WuBu-T output dim is intermediate, not a direct arg for VAE output (latent_dim is)

    valid_stats = {'mag_mean', 'angle_mean', 'mag_std', 'angle_std'};
    if any(s not in valid_stats for s in parsed_args.flow_stats_components): parser.error(f"Invalid flow_stats_components. Allowed: {valid_stats}. Got: {parsed_args.flow_stats_components}")

    return parsed_args

def main():
    args = parse_arguments()
    ddp_active = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    if ddp_active: rank=int(os.environ["RANK"]); local_rank=int(os.environ["LOCAL_RANK"]); world_size=int(os.environ["WORLD_SIZE"]); init_process_group(backend="nccl"); device=torch.device(f"cuda:{local_rank}"); torch.cuda.set_device(device)
    else: rank=0; local_rank=0; world_size=1; device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); _ = torch.cuda.set_device(device) if device.type=='cuda' else None

    am_main_process=(rank==0)
    # Ensure logger is configured per process
    current_logger_main = logging.getLogger("WuBuGAADHybridGenV01")
    if current_logger_main.hasHandlers(): # Clear existing handlers if any (e.g., from re-runs)
        for handler in current_logger_main.handlers[:]: current_logger_main.removeHandler(handler)
        for handler in current_logger_main.root.handlers[:]: current_logger_main.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO if am_main_process else logging.WARNING, format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s', force=True)
    current_logger_main.info(f"--- WuBuGAADHybridGenV01 (R{rank}/{world_size},Dev {device},DDP:{ddp_active}) ---")
    seed_everything(args.seed,rank,world_size)
    if am_main_process: current_logger_main.info(f"Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_id=wandb.util.generate_id() if wandb.run is None else wandb.run.id
        wandb.init(project=args.wandb_project, name=args.wandb_run_name if args.wandb_run_name else f"wubuhybrid_v01_{datetime.now().strftime('%y%m%d%H%M')}", config=vars(args), resume="allow", id=run_id)

    # --- Configure Model Components ---
    video_config = {"image_size":(args.image_h,args.image_w),"num_channels":args.num_channels,"num_input_frames":args.num_input_frames, "num_predict_frames":args.num_predict_frames,"wubu_s_output_dim":args.wubu_s_output_dim,"wubu_m_output_dim": args.wubu_m_output_dim,} # wubu_t_output_dim is internal
    gaad_appearance_config = {"num_regions":args.gaad_num_regions,"decomposition_type":args.gaad_decomposition_type,"min_size_px": args.gaad_min_size_px}
    gaad_motion_config = {"num_regions": args.gaad_motion_num_regions,"decomposition_type":args.gaad_motion_decomposition_type,"min_size_px": args.gaad_min_size_px,} if args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE else None

    wubu_s_config = _configure_wubu_stack(args, "wubu_s")
    wubu_t_config = _configure_wubu_stack(args, "wubu_t") # Used by Encoder
    wubu_m_config = _configure_wubu_stack(args, "wubu_m")

    # Discriminator config (example)
    discriminator_config = {"type": args.discriminator_type}
    if args.discriminator_type == "regional_cnn":
        discriminator_config["patch_size"] = 16 # Example
        discriminator_config["cnn_channels"] = [32, 64, 128] # Example

    if am_main_process: current_logger_main.info(f"VideoCfg:{video_config}\nGAADAppCfg:{gaad_appearance_config}\nGAADMotCfg:{gaad_motion_config}\nWuBuS:{wubu_s_config}\nWuBuT:{wubu_t_config}\nWuBuM:{wubu_m_config}\nDiscCfg:{discriminator_config}")

    # --- Instantiate Models ---
    model = WuBuGAADHybridGenNet(args, video_config, gaad_appearance_config, gaad_motion_config, wubu_s_config, wubu_t_config, wubu_m_config).to(device)
    discriminator = RegionalDiscriminator(args, video_config, gaad_appearance_config, discriminator_config).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model, log="all", log_freq=max(100, args.log_interval*5), log_graph=False)
        wandb.watch(discriminator, log="all", log_freq=max(100, args.log_interval*5), log_graph=False)

    if ddp_active:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # Disc less likely unused

    # --- Optimizers ---
    q_cfg = DEFAULT_CONFIG_QLEARN_HYBRID.copy() if args.q_controller_enabled else None
    optimizer_enc_gen = RiemannianEnhancedSGD(model.parameters(), lr=args.learning_rate_gen, q_learning_config=q_cfg, max_grad_norm_risgd=args.risgd_max_grad_norm)
    optimizer_disc = RiemannianEnhancedSGD(discriminator.parameters(), lr=args.learning_rate_disc, q_learning_config=q_cfg, max_grad_norm_risgd=args.risgd_max_grad_norm) # Can use same Q-config or separate

    # --- Prepare Data ---
    actual_video_path = args.video_data_path; demo_file_name = "dummy_video_hybridgen_v01.mp4"
    if "demo_video_data" in args.video_data_path: actual_video_path = os.path.join(args.video_data_path, demo_file_name)

    # Create dummy video if needed (using imageio if available)
    if "demo_video_data" in args.video_data_path and am_main_process:
        os.makedirs(args.video_data_path, exist_ok=True)
        if not os.path.exists(actual_video_path):
            if IMAGEIO_AVAILABLE and imageio is not None:
                current_logger_main.info(f"Creating dummy video using imageio: {actual_video_path}...")
                min_raw_frames = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                num_dummy = max(50, min_raw_frames + 20)
                dummy_h = int(args.image_h); dummy_w = int(args.image_w)
                frames_np_list = [np.random.randint(0, 256, (dummy_h, dummy_w, args.num_channels), dtype=np.uint8) for _ in range(num_dummy)]
                current_fps = int(10)
                try:
                    imageio.mimwrite(actual_video_path, frames_np_list, fps=current_fps, quality=8) # Add quality
                    current_logger_main.info(f"Successfully created dummy video using imageio: {actual_video_path} with {len(frames_np_list)} frames.")
                except Exception as e_imageio_write:
                    current_logger_main.error(f"Error creating dummy video using imageio: {e_imageio_write}", exc_info=True)
            else:
                current_logger_main.error("imageio library not found or failed previously. Cannot create dummy video.")
        elif os.path.exists(actual_video_path):
             current_logger_main.info(f"Dummy video {actual_video_path} already exists.")

    if ddp_active: torch.distributed.barrier() # Wait for main process to potentially create data

    if not os.path.isfile(actual_video_path):
        current_logger_main.error(f"Video path '{actual_video_path}' is not a file or does not exist. Check path or dummy video creation. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    # --- Datasets and Loaders ---
    total_frames_sample = args.num_input_frames + args.num_predict_frames # VAE needs full sequence
    full_dataset = None
    try: full_dataset = VideoFrameDataset(video_path=actual_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
    except Exception as e: current_logger_main.error(f"Failed init main Dataset from {actual_video_path}: {e}", exc_info=True); sys.exit(1)
    if not full_dataset or len(full_dataset) == 0 : current_logger_main.error("Main dataset empty/failed load. Check path/content. Exit."); sys.exit(1)

    train_dataset, val_dataset = full_dataset, None
    total_possible_samples = len(full_dataset)

    # Validation split logic (modified for block sampling)
    if args.validation_video_path and os.path.exists(args.validation_video_path) and os.path.isfile(args.validation_video_path):
        try:
            val_dataset = VideoFrameDataset(video_path=args.validation_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
            if len(val_dataset) > 0: current_logger_main.info(f"Using separate val video: {args.validation_video_path}, {len(val_dataset)} samples.")
            else: current_logger_main.warning(f"Validation video {args.validation_video_path} loaded 0 samples, using split instead if applicable."); val_dataset = None
        except Exception as e: current_logger_main.warning(f"Could not load val dataset {args.validation_video_path}: {e}. Using split if applicable."); val_dataset = None

    # If no separate val video, perform split with block sampling
    if val_dataset is None and args.validation_split_fraction > 0 and total_possible_samples > 10 :
        val_block_size = args.val_block_size
        if val_block_size <= 0:
            current_logger_main.warning(f"val_block_size must be positive, got {val_block_size}. Falling back to simple random split.")
            # Fallback to original random_split
            num_val = int(total_possible_samples * args.validation_split_fraction)
            num_train = total_possible_samples - num_val
            if num_train > 0 and num_val > 0:
                 train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(args.seed + rank))
                 current_logger_main.info(f"Split main dataset (simple random): {len(train_dataset)} train, {len(val_dataset)} val.")
            else:
                 current_logger_main.warning(f"Not enough samples ({total_possible_samples}) for val split."); val_dataset = None
        elif total_possible_samples < val_block_size:
             current_logger_main.warning(f"Total samples ({total_possible_samples}) less than val_block_size ({val_block_size}). Cannot create blocks. Falling back to simple random split.")
             # Fallback to original random_split
             num_val = int(total_possible_samples * args.validation_split_fraction)
             num_train = total_possible_samples - num_val
             if num_train > 0 and num_val > 0:
                 train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(args.seed + rank))
                 current_logger_main.info(f"Split main dataset (simple random): {len(train_dataset)} train, {len(val_dataset)} val.")
             else:
                 current_logger_main.warning(f"Not enough samples ({total_possible_samples}) for val split."); val_dataset = None
        else:
            # Calculate target number of validation samples and blocks
            target_val_samples = int(total_possible_samples * args.validation_split_fraction)
            num_val_blocks = max(1, target_val_samples // val_block_size)

            # Calculate the maximum possible starting index for a block
            max_block_start = total_possible_samples - val_block_size
            if max_block_start < 0:
                 current_logger_main.warning(f"max_block_start is negative ({max_block_start}). Falling back to simple random split.")
                 # Fallback to original random_split
                 num_val = int(total_possible_samples * args.validation_split_fraction)
                 num_train = total_possible_samples - num_val
                 if num_train > 0 and num_val > 0:
                      train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(args.seed + rank))
                      current_logger_main.info(f"Split main dataset (simple random): {len(train_dataset)} train, {len(val_dataset)} val.")
                 else:
                      current_logger_main.warning(f"Not enough samples ({total_possible_samples}) for val split."); val_dataset = None

            else:
                # Use seeded random generator to select block starting indices
                rng = random.Random(args.seed + rank) # Use standard library random with seed
                # Sample unique starting indices for blocks
                if num_val_blocks > max_block_start + 1:
                     current_logger_main.warning(f"Number of requested validation blocks ({num_val_blocks}) exceeds possible starting positions ({max_block_start + 1}). Using all possible blocks.")
                     block_starts = sorted(range(max_block_start + 1))
                else:
                    block_starts = sorted(rng.sample(range(max_block_start + 1), num_val_blocks)) # Ensure sampling within bounds and sort for easier index management

                # Construct validation indices from the blocks
                val_indices_set = set()
                for start in block_starts:
                    for i in range(val_block_size):
                        # Add indices for the current block
                        if start + i < total_possible_samples: # Ensure index is within dataset bounds
                             val_indices_set.add(start + i)
                        else:
                             current_logger_main.warning(f"Skipping index {start+i} out of bounds (total samples: {total_possible_samples}) during block construction.")

                # Construct all possible indices
                all_indices_set = set(range(total_possible_samples))

                # Training indices are all indices NOT in validation indices
                train_indices_set = all_indices_set - val_indices_set

                # Convert sets back to sorted lists
                train_indices = sorted(list(train_indices_set))
                val_indices = sorted(list(val_indices_set))

                # Create Subset datasets
                if len(train_indices) > 0 and len(val_indices) > 0:
                     train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
                     val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
                     current_logger_main.info(f"Split main dataset (block sampling): {len(train_dataset)} train, {len(val_dataset)} val ({num_val_blocks} blocks of size {val_block_size}).")
                else:
                     current_logger_main.warning(f"Block sampling resulted in empty train ({len(train_indices)}) or val ({len(val_indices)}) sets. Falling back to no split.");
                     # Fallback to no split if block sampling fails to create valid sets
                     train_dataset = full_dataset
                     val_dataset = None


    # --- Sampler and DataLoader creation remains the same, using Subset if applicable ---
    partial_seed_worker = functools.partial(seed_worker_init_fn,base_seed=args.seed,rank=rank,world_size=world_size)
    # Samplers should be applied to the Subset datasets if they were created
    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True,seed=args.seed) if ddp_active else None
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=(train_sampler is None),num_workers=args.num_workers,sampler=train_sampler,pin_memory=(device.type=='cuda'),worker_init_fn=partial_seed_worker if args.num_workers>0 else None,drop_last=True)

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        # Use SequentialSampler or DistributedSampler without shuffle for validation Subset
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None # No shuffle for val
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=(device.type=='cuda'), drop_last=False, worker_init_fn=partial_seed_worker if args.num_workers > 0 else None)
    elif am_main_process: current_logger_main.info("No validation dataset/loader configured or empty.")


    # --- Trainer ---
    trainer = HybridTrainer(model, discriminator, optimizer_enc_gen, optimizer_disc, device, train_loader, val_loader, args, rank, world_size, ddp_active)

    start_global_step,start_epoch=0,0
    if args.load_checkpoint: start_global_step,start_epoch=trainer.load_checkpoint(args.load_checkpoint)

    # --- Training Loop ---
    try:
        trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt:
        current_logger_main.info(f"Rank {rank}: Training interrupted by user.")
    except Exception as e:
        current_logger_main.error(f"Rank {rank}: Training loop crashed: {e}", exc_info=True)
    finally:
        if am_main_process:
            current_logger_main.info("Finalizing run...")
            trainer._save_checkpoint(metrics=trainer.last_val_metrics if trainer.last_val_metrics else {}) # Save final checkpoint

            # --- Final Demo Sampling ---
            if args.epochs > 0 and hasattr(trainer, 'sample') and trainer.global_step > 0:
                current_logger_main.info("DEMO SAMPLING...")
                try:
                    # Sample from prior noise
                    pred_pixels = trainer.sample(num_samples=args.demo_num_samples)
                    current_logger_main.info(f"Demo predicted pixels shape: {pred_pixels.shape}") #(B, N_pred, C, H, W)

                    if pred_pixels.numel() > 0 and pred_pixels.shape[0] > 0:
                        save_dir = os.path.join(args.checkpoint_dir, "demo_samples_hybrid_v01"); os.makedirs(save_dir, exist_ok=True)
                        # Save the first predicted frame of each sample
                        for b in range(args.demo_num_samples):
                            save_image(pred_pixels[b, 0].cpu().clamp(-1, 1) * 0.5 + 0.5, os.path.join(save_dir, f"demo_sample_{b}_frame_0.png"))
                        current_logger_main.info(f"Saved demo sample frames to {save_dir}")

                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            wb_imgs=[wandb.Image(pred_pixels[b, 0].cpu().float().clamp(-1, 1) * 0.5 + 0.5, caption=f"Sample {b} Frame 0") for b in range(args.demo_num_samples)]
                            wandb.log({"demo_samples_final": wb_imgs}, step=trainer.global_step)

                except Exception as e_demo:
                    current_logger_main.error(f"Demo sampling error: {e_demo}", exc_info=True)

            if args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish()

        if ddp_active and is_initialized():
            destroy_process_group()

        current_logger_main.info(f"Rank {rank}: WuBuGAADHybridGen (v0.1) script finished.")


if __name__ == "__main__":
    # Check for incompatible args before full parsing if possible
    if not OPTICAL_FLOW_AVAILABLE:
        motion_branch_requested = False
        for i in range(1, len(sys.argv)): # More robust check
            if sys.argv[i] == '--use_wubu_motion_branch': motion_branch_requested = True; break
            if sys.argv[i].startswith('--use_wubu_motion_branch='):
                val = sys.argv[i].split('=', 1)[1]
                if val.lower() in ['true', '1', 'yes']: motion_branch_requested = True
                break
        if motion_branch_requested:
            print("FATAL ERROR: Motion branch (--use_wubu_motion_branch) requested, but torchvision.models.optical_flow unavailable. Install compatible torchvision or omit the flag.")
            sys.exit(1)
    main()