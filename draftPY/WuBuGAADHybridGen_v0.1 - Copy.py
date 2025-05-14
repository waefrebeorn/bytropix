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
from torch.nn.utils import spectral_norm
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
DEFAULT_CONFIG_QLEARN_HYBRID = {
    "q_learning_rate": 0.01,        # Was "learning_rate"
    "discount_factor": 0.90,        # Was "discount", value slightly changed as per V2 design
    "epsilon_start": 0.5,           # Was "epsilon", value changed as per V2 design
    "epsilon_min": 0.05,            # Was "min_epsilon", value changed
    "epsilon_decay": 0.9995,        # Value changed as per V2 design
    "lr_scale_options": [0.8, 0.9, 1.0, 1.1, 1.2], # Kept V2 example, adjust if needed
    "momentum_scale_options": [0.95, 0.98, 1.0, 1.01, 1.02], # Kept V2 example
    "max_q_table_size": 20000,      # Increased as per V2 design
    "state_history_len": 5,         # Changed as per V2 design
    "reward_clipping": (-2.0, 2.0), # Added as per V2 design
    "q_value_clipping": (-30.0, 30.0) # Added as per V2 design
    # 'initial_lambda_kl' is NOT set here because it's specific to the trainer's current state
    # and will be passed dynamically via `set_current_lambda_kl` if needed.
    # The HAKMEMQController.__init__ has a default for it anyway.
}

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

class HAKMEMQController: # Renamed from HAKMEMQController_VAEGan_v2
    def __init__(self,
                 q_learning_rate: float = 0.01, # Alpha
                 discount_factor: float = 0.90, # Gamma
                 epsilon_start: float = 0.5,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995,
                 lr_scale_options: list[float] | None = None,
                 momentum_scale_options: list[float] | None = None,
                 max_q_table_size: int = 20000,
                 state_history_len: int = 5,
                 reward_clipping: tuple[float, float] | None = (-2.0, 2.0),
                 q_value_clipping: tuple[float, float] | None = (-30.0, 30.0),
                 # initial_lambda_kl is passed to set_current_lambda_kl by trainer
                 # if used by the state definition or reward.
                 ):

        self.q_table: dict[tuple, dict[str, np.ndarray]] = {}
        self.alpha = q_learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_clipping = reward_clipping
        self.q_value_clipping = q_value_clipping

        self.current_lambda_kl: float = 0.0001 # Default, will be updated by trainer

        self.action_ranges = {
            'lr_scale': np.array(lr_scale_options if lr_scale_options else [0.8, 0.9, 1.0, 1.1, 1.2]),
            'momentum_scale': np.array(momentum_scale_options if momentum_scale_options else [0.95, 0.98, 1.0, 1.01, 1.02])
        }
        self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}

        self.state_history_len = max(3, state_history_len)
        self.loss_g_total_hist = deque(maxlen=self.state_history_len)
        self.loss_g_recon_hist = deque(maxlen=self.state_history_len)
        self.loss_g_kl_hist = deque(maxlen=self.state_history_len)
        self.loss_g_adv_hist = deque(maxlen=self.state_history_len)
        self.loss_d_total_hist = deque(maxlen=self.state_history_len)
        self.loss_d_real_hist = deque(maxlen=self.state_history_len) # New for D state/reward
        self.loss_d_fake_hist = deque(maxlen=self.state_history_len) # New for D state/reward

        self.prev_state: tuple | None = None
        self.prev_action: dict[str, float] | None = None
        self.reward_hist = deque(maxlen=50)

        self.max_q_table_size = max_q_table_size
        self.q_table_access_count: dict[tuple, int] = defaultdict(int)
        self.q_table_creation_time: dict[tuple, float] = {}
        self.q_table_last_access_time: dict[tuple, float] = {}

        self.reward_weights = {
            "g_recon_improvement": 2.5, # Slightly higher emphasis
            "g_adv_improvement": 1.2,
            "g_kl_control": 0.3,         # Penalize if KL high, recon bad, lambda_kl not tiny
            "g_loss_stability": 0.1,
            "d_balance": 1.5,            # Target D_total around 0.5-0.6
            "d_real_low": 0.7,           # Reward D for low D_real_loss
            "d_fake_low_meaningful": 0.7,# Reward D for low D_fake_loss (if G isn't collapsed)
            "d_loss_stability": 0.1,
            "gan_balance_for_g": 0.3,    # G reward based on D's performance
            "gan_balance_for_d": 0.3,    # D penalty if G is too strong
            "oscillation_penalty": 0.25,
            "extreme_loss_penalty": 0.75
        }

        self.logger = logging.getLogger(f"WuBuGAADHybridGenV01.QController") # Simplified name
        self.logger.info(f"HAKMEMQController (VAEGanV2 logic) initialized. Eps: {self.epsilon:.2f}->{self.epsilon_min:.2f}")
        self._internal_step_counter = 0 # For less frequent debug logging

    def _get_trend_bin(self, history: deque, current_val: float, relative_to_median: bool = True, value_scale_for_diff:float = 1.0) -> int:
        # Ensure history is not empty and has previous values for comparison
        if len(history) < 1 : # Need at least one past value in history to compare with current_val
             if len(history) == 0 and current_val is not None: # First value ever
                return 2 # Neutral trend for the very first data point
             # If history is empty but current_val is None or NaN, also neutral
             return 2


        # Use the last element of history as previous_value for trend calculation
        # This makes it more reactive to the immediate past step.
        # For median-based trend, we'd need more history.
        
        # temp_hist_for_slope will contain [history[-1], current_val] or more if history is longer
        # We want to compare current_val to the median of the history *before* current_val was added.
        
        valid_history = [h for h in history if np.isfinite(h)]
        if not valid_history: # No finite past values
            return 2 # Neutral

        prev_median = np.median(valid_history)
        
        # Calculate difference for trend (current vs. previous median)
        # Polyfit is more robust for longer histories if desired, but simple diff to median is fine.
        diff = current_val - prev_median

        if relative_to_median:
            # Denominator based on the scale of the values being compared
            # Use median of history or current value if history median is too small
            denominator_scale = abs(prev_median)
            if denominator_scale < value_scale_for_diff * 0.1: # If median is very small
                denominator_scale = max(abs(current_val), value_scale_for_diff * 0.1) # Use current value or a minimum scale
            denominator = denominator_scale + EPS
            relative_diff = diff / denominator
        else:
            relative_diff = diff / (value_scale_for_diff + EPS) # Normalize by a general scale if not relative to median

        # Bins: Strong Decrease, Decrease, Stable, Increase, Strong Increase
        # Tuned thresholds for more sensitivity
        if relative_diff < -0.15: return 0  # Strong Decrease
        if relative_diff < -0.02: return 1  # Decrease
        if relative_diff <= 0.02: return 2  # Stable
        if relative_diff <= 0.15: return 3  # Increase
        return 4                             # Strong Increase

    def get_state(self, current_losses: dict[str, float], 
                  current_lr: float, current_momentum: float, # Kept for signature compatibility
                  is_generator_q: bool) -> tuple | None:
        self._internal_step_counter +=1

        # Update history deques
        loss_map = {
            'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist
        }
        for name, deq in loss_map.items():
            if name in current_losses and np.isfinite(current_losses[name]):
                deq.append(current_losses[name])
            # else:
                # If a crucial loss is missing, we might not be able to form a state
                # self.logger.debug(f"State: Missing or non-finite crucial loss '{name}'")
                # return None # This is strict, handled by required_keys check below

        # Check for required losses
        if is_generator_q:
            required_keys = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        else: # Discriminator
            required_keys = ['loss_d_total', 'loss_g_total', 'loss_d_real', 'loss_d_fake', 'loss_g_adv']
        
        if not all(key in current_losses and np.isfinite(current_losses[key]) for key in required_keys):
            # self.logger.debug(f"QState ({'G' if is_generator_q else 'D'}): Insufficient/non-finite. Need: {required_keys}, Got: {list(current_losses.keys())}")
            return None

        # Ensure enough history for trends (at least one past point in relevant deques)
        if is_generator_q:
            if not (self.loss_g_total_hist and self.loss_g_recon_hist and self.loss_d_total_hist): return None
        else: # Discriminator
            if not (self.loss_d_total_hist and self.loss_g_total_hist and self.loss_d_real_hist and self.loss_d_fake_hist): return None

        # State components
        if is_generator_q:
            s_g_total_trend = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total'], value_scale_for_diff=1.0)
            s_d_total_trend_opp = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total'], value_scale_for_diff=0.5)
            s_g_recon_trend = self._get_trend_bin(self.loss_g_recon_hist, current_losses['loss_g_recon'], value_scale_for_diff=0.1)
            
            kl_val = current_losses['loss_g_kl']
            recon_val = current_losses['loss_g_recon']
            s_kl_problem = 0 # No problem
            # If KL (weighted) is much larger than Recon (weighted), and Recon is bad, and lambda_kl isn't tiny
            if (self.current_lambda_kl * kl_val > self.reward_weights.get("g_recon_improvement", 2.0) * recon_val * 2.0 and 
                recon_val > 0.10 and self.current_lambda_kl > 0.0005):
                s_kl_problem = 1 # KL likely too dominant
            elif kl_val > 150.0 and recon_val > 0.15 and self.current_lambda_kl > 0.005: # General high KL, bad recon
                s_kl_problem = 2
            
            g_adv_val = current_losses['loss_g_adv']
            s_g_adv_level = np.digitize(g_adv_val, [0.3, 0.6, 1.0]).item() # Low (good for G), Med, High (bad for G)

            state_tuple = (
                1, # ID for G_state
                s_g_total_trend,
                s_d_total_trend_opp,
                s_g_recon_trend,
                s_kl_problem,
                s_g_adv_level,
                np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item() # Epsilon phase
            )
        else: # Discriminator state
            s_d_total_trend = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total'], value_scale_for_diff=0.5)
            s_g_total_trend_opp = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total'], value_scale_for_diff=1.0)
            
            d_total_abs = current_losses['loss_d_total']
            s_d_balance_bin = np.digitize(d_total_abs, [0.35, 0.65, 0.85]).item() # 0:Low, 1:Mid-Low, 2:Mid-High, 3:High
            
            # Ratio of D_fake to D_real: Higher means D struggles more with fakes than reals
            d_real_val = current_losses['loss_d_real']
            d_fake_val = current_losses['loss_d_fake']
            s_d_fake_vs_real_ratio_bin = np.digitize(d_fake_val / (d_real_val + EPS), [0.8, 1.2, 2.0]).item() 
            # 0: D_fake easier than D_real; 1: About same; 2: D_fake harder; 3: D_fake much harder

            state_tuple = (
                0, # ID for D_state
                s_d_total_trend,
                s_g_total_trend_opp,
                s_d_balance_bin,
                s_d_fake_vs_real_ratio_bin, 
                np.digitize(current_losses.get('loss_g_adv', 0.7), [0.2, 0.5]).item(), # G's power level (low G_adv means G is strong)
                np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item()
            )

        current_time = time.time()
        self.q_table_access_count[state_tuple] += 1
        self.q_table_last_access_time[state_tuple] = current_time
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {p_type: np.zeros(n_actions) for p_type, n_actions in self.num_actions.items()}
            self.q_table_creation_time[state_tuple] = current_time
            self._manage_q_table_size()
        return state_tuple

    def choose_action(self, state: tuple | None) -> dict[str, float]:
        # ... (choose_action method remains identical to your V2 logic)
        default_action = {'lr_scale': 1.0, 'momentum_scale': 1.0}
        if state is None or state not in self.q_table:
            return default_action
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        chosen_actions = {}
        for param_type, q_values_for_param in self.q_table[state].items():
            action_space = self.action_ranges[param_type]
            if random.random() < self.epsilon:
                chosen_idx = random.randrange(len(action_space))
            else:
                finite_q_mask = np.isfinite(q_values_for_param)
                if np.any(finite_q_mask):
                    q_values_finite = q_values_for_param[finite_q_mask]
                    action_indices_finite = np.arange(len(q_values_for_param))[finite_q_mask]
                    best_q_val_finite = np.max(q_values_finite)
                    best_indices_options = action_indices_finite[np.isclose(q_values_finite, best_q_val_finite)]
                    chosen_idx = random.choice(best_indices_options)
                else:
                    self.logger.warning(f"State {state}, PType {param_type}: All Q-vals non-finite. Random action.")
                    chosen_idx = random.randrange(len(action_space))
            chosen_actions[param_type] = float(action_space[chosen_idx])
        self.prev_action = chosen_actions.copy()
        return chosen_actions


    def update(self, state: tuple, action: dict[str, float], reward: float, next_state: tuple | None):
        # ... (update method remains identical to your V2 logic)
        if state not in self.q_table: return
        if self.reward_clipping: reward = np.clip(reward, self.reward_clipping[0], self.reward_clipping[1])
        for param_type, chosen_scale_value in action.items():
            action_idx_arr = np.where(np.isclose(self.action_ranges[param_type], chosen_scale_value))[0]
            if not action_idx_arr.size: continue
            action_idx = action_idx_arr[0]
            current_q_value = self.q_table[state][param_type][action_idx]
            max_future_q = 0.0
            if next_state is not None and next_state in self.q_table:
                next_q_vals = self.q_table[next_state][param_type]
                if np.any(np.isfinite(next_q_vals)): max_future_q = np.max(next_q_vals[np.isfinite(next_q_vals)])
            td_target = reward + self.gamma * max_future_q
            td_error = td_target - current_q_value
            new_q_value = current_q_value + self.alpha * td_error
            if np.isfinite(new_q_value):
                if self.q_value_clipping: new_q_value = np.clip(new_q_value, self.q_value_clipping[0], self.q_value_clipping[1])
                self.q_table[state][param_type][action_idx] = new_q_value
        self.reward_hist.append(reward)


    def _manage_q_table_size(self):
        # ... (manage_q_table_size method remains identical to your V2 logic)
        if len(self.q_table) <= self.max_q_table_size: return
        num_to_prune = len(self.q_table) - self.max_q_table_size
        current_time = time.time()
        state_scores = {s_tuple: (self.q_table_access_count.get(s_tuple, 1) *
                                (1.0 + np.log1p((current_time - self.q_table_creation_time.get(s_tuple, current_time)) / 86400.0)) *
                                (1.0 / (1.0 + np.log1p((current_time - self.q_table_last_access_time.get(s_tuple, current_time)) / 3600.0))))
                        for s_tuple in self.q_table.keys()}
        sorted_states_for_pruning = sorted(state_scores.keys(), key=lambda s: state_scores[s])
        for i in range(num_to_prune):
            s_rm = sorted_states_for_pruning[i]
            self.q_table.pop(s_rm, None); self.q_table_access_count.pop(s_rm, None)
            self.q_table_creation_time.pop(s_rm, None); self.q_table_last_access_time.pop(s_rm, None)
        if num_to_prune > 0: self.logger.info(f"Pruned {num_to_prune} Q-table entries. New size: {len(self.q_table)}.")


    def compute_reward(self, current_losses: dict[str, float], is_generator_q: bool) -> float:
        total_reward = 0.0
        w = self.reward_weights # Shorthand

        # --- Universal Penalty for Bad Losses ---
        for loss_name, loss_val in current_losses.items():
            if not np.isfinite(loss_val):
                total_reward -= w["extreme_loss_penalty"] * 5
                self.logger.warning(f"Non-finite loss '{loss_name}' = {loss_val} in reward calc.")
                current_losses[loss_name] = 100.0 # Assign high placeholder if non-finite
            elif abs(loss_val) > 500: # Very large loss
                total_reward -= w["extreme_loss_penalty"] * (abs(loss_val) / 500.0)
                current_losses[loss_name] = np.sign(loss_val) * 500 # Clip for stable calculations

        # Get previous median losses from history for trend calculation
        def get_prev_median(hist_deque, current_val_fallback):
            valid_hist = [v for v in hist_deque if np.isfinite(v)]
            return np.median(valid_hist[:-1]) if len(valid_hist) > 1 else (valid_hist[0] if len(valid_hist) == 1 else current_val_fallback)

        # --- Generator Rewards ---
        if is_generator_q:
            loss_g_recon = current_losses.get('loss_g_recon', 1.0)
            prev_g_recon = get_prev_median(self.loss_g_recon_hist, loss_g_recon)
            recon_improvement = prev_g_recon - loss_g_recon
            recon_scale = 1.0 + math.log1p(max(0, loss_g_recon - 0.02) * 20) # Reward more if recon_loss is further from ~0.02
            total_reward += np.tanh(recon_improvement / (abs(prev_g_recon) + 0.01 + EPS) * recon_scale) * w["g_recon_improvement"]

            loss_g_adv = current_losses.get('loss_g_adv', 0.7)
            prev_g_adv = get_prev_median(self.loss_g_adv_hist, loss_g_adv)
            adv_improvement = prev_g_adv - loss_g_adv # G wants G_adv to decrease
            total_reward += np.tanh(adv_improvement / (abs(prev_g_adv) + EPS)) * w["g_adv_improvement"]

            loss_g_kl = current_losses.get('loss_g_kl', 0.0)
            # Penalize high KL if recon is bad and lambda_kl isn't already tiny
            if loss_g_kl > 100.0 and self.current_lambda_kl >= 0.0005 and loss_g_recon > 0.1:
                total_reward -= w["g_kl_control"] * min(1.0, (loss_g_kl - 100.0) / 200.0)
            
            loss_d_total = current_losses.get('loss_d_total', 0.7) # D's performance
            # Reward G if D is in a "balanced" state (not too strong, not too weak)
            if 0.4 < loss_d_total < 0.75:
                total_reward += w["gan_balance_for_g"]
            elif loss_d_total <= 0.3: # D is too strong
                total_reward -= w["gan_balance_for_g"] * 1.5
            
            loss_g_total = current_losses.get('loss_g_total', 1.0)
            prev_g_total = get_prev_median(self.loss_g_total_hist, loss_g_total)
            g_total_improvement = prev_g_total - loss_g_total
            total_reward += np.tanh(g_total_improvement / (abs(prev_g_total) + EPS)) * w["g_loss_stability"]

        # --- Discriminator Rewards ---
        else:
            loss_d_total = current_losses.get('loss_d_total', 0.7)
            # Reward D for being in a balanced state (loss around 0.4-0.65)
            if 0.4 < loss_d_total < 0.65:
                total_reward += w["d_balance"]
            elif loss_d_total < 0.3: # D too strong or G collapsed
                total_reward -= w["d_balance"] * 0.5 
            elif loss_d_total > 0.8: # D struggling
                total_reward -= w["d_balance"] * 0.75

            loss_d_real = current_losses.get('loss_d_real', 0.7)
            if loss_d_real < 0.3: # Good at reals
                total_reward += w["d_real_low"] * (0.3 - loss_d_real) / 0.3
            
            loss_d_fake = current_losses.get('loss_d_fake', 0.7)
            loss_g_adv_opp = current_losses.get('loss_g_adv', 0.7) # G's adv loss from D's perspective
            if loss_d_fake < 0.3 and loss_g_adv_opp > 0.4: # Good at fakes, and G isn't totally collapsed
                total_reward += w["d_fake_low_meaningful"] * (0.3 - loss_d_fake) / 0.3
            
            # Penalize D if G is fooling it too easily
            if loss_g_adv_opp < 0.25: 
                total_reward -= w["gan_balance_for_d"]

            prev_d_total = get_prev_median(self.loss_d_total_hist, loss_d_total)
            d_total_improvement = prev_d_total - loss_d_total
            total_reward += np.tanh(d_total_improvement / (abs(prev_d_total) + EPS)) * w["d_loss_stability"]

        # --- Oscillation Penalty (Applied to Both) ---
        if len(self.reward_hist) >= self.state_history_len: # Use state_history_len for consistency
            recent_rewards = list(self.reward_hist)[-self.state_history_len:]
            # Count sign changes in recent rewards
            sign_flips = 0
            for i in range(len(recent_rewards) - 1):
                if np.sign(recent_rewards[i]) != np.sign(recent_rewards[i+1]) and \
                   abs(recent_rewards[i]) > 0.05 and abs(recent_rewards[i+1]) > 0.05: # Only count significant flips
                    sign_flips += 1
            if sign_flips >= (self.state_history_len // 2) : # If half or more of recent steps were flips
                total_reward -= w["oscillation_penalty"] * (sign_flips / self.state_history_len)

        role_char = 'G' if is_generator_q else 'D'
        if self.logger.isEnabledFor(logging.DEBUG) and self._internal_step_counter % 50 == 1: # Log less frequently
             self.logger.debug(
                 f"QCRew({role_char} S{self._internal_step_counter}): RawRew={total_reward:.3f}. Losses={ {k:f'{v:.3f}' for k,v in current_losses.items()} }"
             )
        
        if self.reward_clipping:
            total_reward = np.clip(total_reward, self.reward_clipping[0], self.reward_clipping[1])
        return float(total_reward)

    def set_current_lambda_kl(self, lambda_kl_val: float):
        if np.isfinite(lambda_kl_val):
            self.current_lambda_kl = float(lambda_kl_val)
        else:
            self.logger.warning(f"Attempted to set non-finite lambda_kl: {lambda_kl_val}")


    def get_info(self) -> dict:
        # ... (get_info method remains identical to your V2 logic)
        q_mem_mb = 0.0
        try:
            if self.q_table: q_mem_mb = sum(sys.getsizeof(s_tuple) + sum(q_vals.nbytes + sys.getsizeof(p_type) for p_type, q_vals in q_actions.items()) for s_tuple, q_actions in self.q_table.items()) / (1024**2)
        except Exception: q_mem_mb = -1.0 # Error calculating
        avg_reward_recent = np.mean(list(self.reward_hist)) if self.reward_hist else 0.0
        return {
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "q_table_mem_mb_approx": round(q_mem_mb, 3),
            "last_chosen_action": self.prev_action if self.prev_action else "None",
            f"avg_reward_last_{self.reward_hist.maxlen}": round(avg_reward_recent, 3),
        }


    def set_initial_losses(self, losses: dict[str, float], is_generator_q: bool):
        # ... (set_initial_losses method remains identical to your V2 logic)
        # This is important for the first few Q-state calculations to have some history.
        loss_map_init = {
            'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist
        }
        relevant_loss_keys = []
        if is_generator_q:
            relevant_loss_keys.extend(['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv'])
            relevant_loss_keys.append('loss_d_total') # Opponent
        else: # Discriminator
            relevant_loss_keys.extend(['loss_d_total', 'loss_d_real', 'loss_d_fake'])
            relevant_loss_keys.extend(['loss_g_total', 'loss_g_adv']) # Opponent info

        for name in relevant_loss_keys:
            if name in losses and np.isfinite(losses[name]):
                loss_map_init[name].append(losses[name])
        
        # Fill history deques if they are shorter than state_history_len
        for deq_name, deq_obj in loss_map_init.items():
            if deq_obj: # If deque is not empty after appending
                 while len(deq_obj) < self.state_history_len:
                    deq_obj.appendleft(deq_obj[0]) # Pad with the earliest available value


class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[nn.Parameter], # More specific type hint
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 0.01,
                 max_grad_norm_risgd: float = 1.0,
                 q_learning_config: Optional[Dict] = None,
                 optimizer_type: str = "generator"  # "generator" or "discriminator"
                ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Store base_lr and base_momentum in defaults, they will be used by Q-controller logic
        # The actual 'lr' and 'momentum' in defaults will be dynamically set by Q-controller if active
        defaults = dict(
            lr=lr,
            initial_lr=lr, # Store the initially configured LR
            momentum=momentum,
            initial_momentum=momentum, # Store the initially configured momentum
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        self.optimizer_type = optimizer_type.lower()
        if self.optimizer_type not in ["generator", "discriminator"]:
            raise ValueError("optimizer_type must be 'generator' or 'discriminator'")

        if isinstance(q_learning_config, dict):
            q_params = q_learning_config.copy()
            self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_params)
        else:
            self.q_controller = None

        self.logger = logging.getLogger(f"WuBuGAADHybridGenV01.RiSGD.{self.optimizer_type.capitalize()}")
        if not self.logger.hasHandlers() and not logging.getLogger("WuBuGAADHybridGenV01").hasHandlers(): # Check root too
            # Basic config if no handlers exist anywhere up the chain
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
        self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled for {self.optimizer_type} optimizer.")

        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count_internal = 0 # Renamed to avoid conflict if trainer has its own global_step
        self.grad_stats = GradientStats()

        # Initialize state for all parameters (momentum_buffer)
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state.setdefault(p, {}) # state[p] will store 'momentum_buffer'

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        # grad_stats should be reset by the trainer before new backward pass
        # or if it's per-step, then here is fine too.
        # Let's assume trainer handles its reset timing relative to accumulation.
        # self.grad_stats.reset() # Keep here if optimizer manages its own grad_stats lifecycle fully

    def q_controller_update_and_set_hyperparams(self,
                                                # Average losses over the last macro-batch
                                                avg_losses_dict: Dict[str, Optional[float]],
                                                current_lambda_kl_value: Optional[float] = None
                                               ):
        if not self.q_controller:
            return

        # Filter out None values from losses_dict for Q-controller
        finite_losses_for_q_state: Dict[str, float] = {
            k: v for k, v in avg_losses_dict.items() if v is not None and np.isfinite(v)
        }

        is_gen_q = (self.optimizer_type == "generator")
        # Define required keys based on the Q-controller's needs (from HAKMEMQController_VAEGan_v2)
        if is_gen_q:
            required_keys = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        else: # Discriminator
            required_keys = ['loss_d_total', 'loss_g_total', 'loss_g_adv']

        if not all(key in finite_losses_for_q_state for key in required_keys):
            self.logger.debug(f"QCtrl ({self.optimizer_type}): Insufficient finite losses. Skipping Q-update. "
                              f"Need: {required_keys}, Got: {list(finite_losses_for_q_state.keys())}")
            return # Cannot form a valid state

        if hasattr(self.q_controller, 'set_current_lambda_kl') and current_lambda_kl_value is not None:
            self.q_controller.set_current_lambda_kl(current_lambda_kl_value)

        # Use 'initial_lr' and 'initial_momentum' from group as the true base for Q-scaling
        # This ensures that if an external scheduler changes 'lr', Q-controller still scales from the original.
        # However, most Q-controllers assume they *are* the scheduler.
        # So, let's use the current group['lr'] and group['momentum'] as the base for *this iteration's calculation*
        # but the scaling applies to group['initial_lr'] / group['initial_momentum'].

        current_lr_for_q_state = self.param_groups[0]['lr']
        current_mom_for_q_state = self.param_groups[0]['momentum']
        
        q_state_current = self.q_controller.get_state(
            finite_losses_for_q_state, current_lr_for_q_state, current_mom_for_q_state,
            is_generator_q=is_gen_q
        )

        if self.q_controller.prev_state is not None and \
           self.q_controller.prev_action is not None and \
           q_state_current is not None:
            reward = self.q_controller.compute_reward(finite_losses_for_q_state, is_generator_q=is_gen_q)
            if np.isfinite(reward):
                self.q_controller.update(self.q_controller.prev_state, self.q_controller.prev_action, reward, q_state_current)
        elif q_state_current is not None and hasattr(self.q_controller, 'set_initial_losses'):
            self.q_controller.set_initial_losses(finite_losses_for_q_state, is_generator_q=is_gen_q)

        self.q_controller.prev_state = q_state_current
        action_for_upcoming_step = self.q_controller.choose_action(q_state_current)
        self.q_controller.prev_action = action_for_upcoming_step # This action will be used for the step

        if action_for_upcoming_step:
            for group in self.param_groups:
                base_lr = group['initial_lr'] # The LR set at optimizer construction
                base_mom = group['initial_momentum'] # The momentum set at optimizer construction

                group['lr'] = float(np.clip(base_lr * action_for_upcoming_step.get('lr_scale', 1.0), 1e-8, 1.0))
                group['momentum'] = float(np.clip(base_mom * action_for_upcoming_step.get('momentum_scale', 1.0), 0.0, 0.999)) # Allow 0 momentum

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        # self.grad_stats should have been populated by the trainer after backward() calls for the macro-batch
        # and finalized by calling self.grad_stats.finalize_step_stats()

        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            initial_lr = group['initial_lr'] # For logging or reference

            for p in group['params']:
                if p.grad is None: # Skip if no gradient
                    continue
                if not p.requires_grad: # Skip if not requiring grad (e.g., frozen)
                    self.logger.warning(f"Parameter {p.shape} has grad but requires_grad is False. Skipping.")
                    continue

                grad = p.grad # No .data needed with @torch.no_grad()

                # Check for non-finite gradients (e.g. NaN/Inf) that might appear AFTER unscaling by GradScaler
                # but BEFORE any clipping.
                if not torch.isfinite(grad).all():
                    self.logger.warning(f"Optimizer step: Non-finite gradient for param shape {p.shape} "
                                        f"({self.optimizer_type}). Skipping update for this parameter.")
                    # Optionally clear momentum buffer if grad is bad
                    self.state[p].pop('momentum_buffer', None)
                    # self.grad_stats was recorded on potentially finite grad before unscale blew it up.
                    # Or, GradScaler itself might have skipped the optimizer step if inf/nan were found *during* unscale.
                    # This check here is a safeguard if this optimizer step is still called.
                    continue

                # 1. Apply per-parameter gradient clipping (RiSGD specific)
                if self.max_grad_norm_risgd > 0 and self.max_grad_norm_risgd != float('inf') :
                    param_grad_norm = grad.norm().item()
                    if param_grad_norm > self.max_grad_norm_risgd:
                        clip_coef = self.max_grad_norm_risgd / (param_grad_norm + EPS)
                        grad.mul_(clip_coef)

                # Get manifold associated with parameter, if any
                manifold: Optional[Manifold] = getattr(p, 'manifold', None)

                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    # --- Hyperbolic Parameter Update ---
                    p_projected_on_manifold = manifold.proju(p) # Ensure p is on manifold

                    # Consistent Weight Decay for Hyperbolic:
                    # Apply decay in Euclidean space *before* converting to Riemannian gradient.
                    # This is generally more stable and interpretable than logmap0-based decay.
                    grad_eff = grad.clone() # Effective gradient
                    if weight_decay != 0:
                        grad_eff.add_(p, alpha=weight_decay) # Add L2 penalty in Euclidean space to grad

                    try:
                        riemannian_grad = manifold.egrad2rgrad(p_projected_on_manifold, grad_eff)
                    except Exception as e_egrad:
                        self.logger.error(f"egrad2rgrad failed for P:{p.shape} (c={manifold.c:.2e}): {e_egrad}. Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue
                    
                    if not torch.isfinite(riemannian_grad).all():
                        self.logger.warning(f"Non-finite Riemannian grad for P:{p.shape} (c={manifold.c:.2e}). Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue

                    # Momentum update in the tangent space T_{p_projected_on_manifold}M
                    # Note: For true RSGD, momentum vectors should be parallel transported if base point changes.
                    # This is often simplified by assuming momentum is in T_0M or by re-calculating/approximating.
                    # Here, we'll use the common simplification: momentum buffer stores vectors in T_{p_current}M.
                    # This means when p changes, the interpretation of the momentum buffer also implicitly changes.
                    # A more advanced RSGD would transport the buffer.
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None:
                            buf = torch.clone(riemannian_grad).detach()
                        else:
                            # Ensure buffer has same shape as riemannian_grad if manifold dims change (rare)
                            if buf.shape != riemannian_grad.shape:
                                self.logger.warning(f"Momentum buffer shape {buf.shape} != grad shape {riemannian_grad.shape}. Resetting.")
                                buf = torch.clone(riemannian_grad).detach()
                            else:
                                buf.mul_(momentum).add_(riemannian_grad) # Standard momentum update
                        self.state[p]['momentum_buffer'] = buf
                    else: # No momentum
                        buf = riemannian_grad # Use current Riemannian gradient directly

                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite momentum buffer for P:{p.shape} (c={manifold.c:.2e}). Resetting buffer.")
                        buf.zero_() # Reset if it became non-finite
                        self.state[p]['momentum_buffer'] = buf


                    # Retraction step: p_new = Retract_{p_old}(-lr * momentum_vector_at_p_old)
                    # For Poincare, Retraction is often the Exponential Map.
                    expmap_tangent_vector = buf.mul(-lr)
                    if not torch.isfinite(expmap_tangent_vector).all():
                        self.logger.warning(f"Non-finite tangent vector for expmap P:{p.shape} (c={manifold.c:.2e}). Skipping param update.")
                        continue # Don't update if the direction is bad

                    try:
                        # Use manifold.expmap for retraction from p_projected_on_manifold
                        new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
                        if not torch.isfinite(new_p_candidate).all():
                            self.logger.warning(f"Expmap resulted in non-finite P:{p.shape} (c={manifold.c:.2e}). Projecting and zeroing momentum.")
                            p.data = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0))
                            if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()
                        else:
                            p.data = manifold.proju(new_p_candidate) # Final projection for safety
                    except Exception as e_expmap:
                        self.logger.error(f"Expmap failed for P:{p.shape} (c={manifold.c:.2e}): {e_expmap}. Zeroing momentum.")
                        if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()
                        continue
                    
                    if not torch.isfinite(p.data).all(): # Should be caught by proju, but final safety
                        self.logger.error(f"Parameter P:{p.shape} (c={manifold.c:.2e}) became non-finite after update. Resetting to origin.")
                        p.data = manifold.expmap0(torch.zeros_like(p.data, device=p.device)) # Reset to origin of its manifold
                        if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()

                else:
                    # --- Euclidean Parameter Update ---
                    grad_eff_euc = grad.clone()
                    if weight_decay != 0:
                        grad_eff_euc.add_(p, alpha=weight_decay) # Standard L2 weight decay

                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None:
                            buf = torch.clone(grad_eff_euc).detach()
                        else:
                            if buf.shape != grad_eff_euc.shape: # Safety check
                                buf = torch.clone(grad_eff_euc).detach()
                            else:
                                buf.mul_(momentum).add_(grad_eff_euc)
                        self.state[p]['momentum_buffer'] = buf
                    else: # No momentum
                        buf = grad_eff_euc
                    
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite Euclidean momentum buffer for P:{p.shape}. Resetting buffer.")
                        buf.zero_()
                        self.state[p]['momentum_buffer'] = buf

                    p.add_(buf, alpha=-lr) # Update: p = p - lr * buf

                    if not torch.isfinite(p.data).all():
                        self.logger.warning(f"Euclidean P:{p.shape} became non-finite. Clamping and zeroing momentum.")
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e5, neginf=-1e5) # Clamp extreme values
                        if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()

        self._step_count_internal += 1
        return loss

    def get_q_controller_info(self) -> Dict:
        return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}

    # grad_stats is now primarily managed by the trainer due to gradient accumulation.
    # This optimizer's self.grad_stats will reflect only the last micro-batch's pre-clip stats
    # if record_param_grad was called before step().
    # The trainer should aggregate GradientStats across micro-batches.
    def get_gradient_stats_summary_optimizer_view(self) -> Dict:
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


# --- FiLM Layer (Helper for GAAD modulation) ---
class FiLMLayer(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.channels = channels
        self.condition_dim = condition_dim
        # MLP to produce gamma and beta from condition
        self.to_gamma_beta = nn.Linear(condition_dim, channels * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) or (B, C, H, W) feature map
        # condition: (B, condition_dim)
        gamma_beta = self.to_gamma_beta(condition)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # Reshape gamma and beta to be broadcastable to x
        if x.dim() == 5: # Spatio-temporal features (B, C, D, H, W)
            gamma = gamma.view(-1, self.channels, 1, 1, 1)
            beta = beta.view(-1, self.channels, 1, 1, 1)
        elif x.dim() == 4: # Spatial features (B, C, H, W)
            gamma = gamma.view(-1, self.channels, 1, 1)
            beta = beta.view(-1, self.channels, 1, 1)
        else:
            raise ValueError(f"FiLMLayer input x has unsupported dimension: {x.dim()}")

        return (1 + gamma) * x + beta


class RegionalGeneratorDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_output_dim: int, latent_dim: int):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.image_size = (args.image_h, args.image_w)
        self.num_regions = gaad_config['num_regions']
        self.num_channels = video_config['num_channels']
        self.latent_dim = latent_dim
        self.num_predict_frames = video_config["num_predict_frames"]
        self.logger = logging.getLogger("WuBuGAADHybridGenV01.Generator")

        # --- Derived Generator Structural Parameters ---
        # Initial spatial resolution: Aim for something like H/16 or W/16
        # Ensure it's a power of 2 for clean upsampling, or the smallest possible (e.g., 4x4)
        # We want self.gen_init_spatial_res * (2^self.gen_num_upsampling_layers) = self.image_size
        min_target_dim = min(self.image_size[0], self.image_size[1])
        if min_target_dim <= 8: # Very small images
            self.gen_init_spatial_res = 1
            self.gen_num_upsampling_layers = int(math.log2(min_target_dim)) if min_target_dim > 0 and math.log2(min_target_dim).is_integer() else max(1, int(math.ceil(math.log2(min_target_dim))))
        elif min_target_dim <= 32:
            self.gen_init_spatial_res = 2
            self.gen_num_upsampling_layers = int(math.log2(min_target_dim / 2)) if (min_target_dim/2)>0 and math.log2(min_target_dim/2).is_integer() else max(1, int(math.ceil(math.log2(min_target_dim/2))))
        else: # min_target_dim > 32
            self.gen_init_spatial_res = 4 # Default starting point for larger images
            self.gen_num_upsampling_layers = int(math.log2(min_target_dim / 4)) if (min_target_dim/4)>0 and math.log2(min_target_dim/4).is_integer() else max(1, int(math.ceil(math.log2(min_target_dim/4))))

        # Check if calculated upsampling perfectly reaches target, warn if not
        calculated_final_res = self.gen_init_spatial_res * (2**self.gen_num_upsampling_layers)
        if calculated_final_res != min_target_dim:
             self.logger.warning(
                f"Generator calculated final res {calculated_final_res} (from init_res {self.gen_init_spatial_res} "
                f"and {self.gen_num_upsampling_layers} upsample layers) does not exactly match target min_dim {min_target_dim}. "
                f"Final adaptive pooling will be crucial."
            )

        # Initial channels: Derived from latent_dim or a fixed large value scaled down
        # Let's make it a multiple of latent_dim, e.g., 2x or 4x latent_dim, capped for stability
        self.gen_init_channels = min(512, max(128, self.latent_dim * 2))

        # Temporal kernel size can remain a fixed arg or default
        self.gen_temporal_kernel_size = getattr(args, 'gen_temporal_kernel_size', 3)


        # 1. Latent to Spatio-Temporal Base
        self.fc_expand_latent = nn.Linear(
            self.latent_dim,
            self.gen_init_channels * self.num_predict_frames * self.gen_init_spatial_res * self.gen_init_spatial_res
        )

        # 2. GAAD BBox Processing for Modulation
        # Condition dimension derived from latent_dim (e.g., latent_dim / 4 or a fixed reasonable value)
        self.gaad_condition_dim = max(32, self.latent_dim // 4)
        if self.num_regions > 0:
            self.bbox_feature_dim = 4 # cx, cy, w, h (normalized)
            # MLP to embed aggregated bbox features for a frame
            hidden_bbox_embed_dim = max(self.gaad_condition_dim, self.num_regions * self.bbox_feature_dim // 2) # Intermediate dim
            self.frame_gaad_embedder = nn.Sequential(
                nn.Linear(self.num_regions * self.bbox_feature_dim, hidden_bbox_embed_dim),
                nn.GELU(),
                nn.Linear(hidden_bbox_embed_dim, self.gaad_condition_dim)
            )
        else:
            self.frame_gaad_embedder = None


        # 3. Spatio-Temporal Upsampling and Refinement with FiLM
        self.upsample_blocks = nn.ModuleList()
        current_channels = self.gen_init_channels
        padding_temp = self.gen_temporal_kernel_size // 2

        # Channel reduction strategy: Halve channels per upsample, but don't go below a minimum (e.g., 32 or 64)
        min_gen_channels = max(32, self.num_channels * 4)

        for i in range(self.gen_num_upsampling_layers):
            # Progressively reduce channels, but not too aggressively
            if i < self.gen_num_upsampling_layers -1 : # Not the last upsampling layer
                 out_channels = max(min_gen_channels, current_channels // 2)
            else: # Last upsampling layer, aim closer to final channel count, e.g., 2x num_channels
                 out_channels = max(min_gen_channels, self.num_channels * 2 if self.num_channels > 1 else min_gen_channels)

            block = nn.ModuleDict()
            block['conv_transpose'] = nn.ConvTranspose3d(
                current_channels, out_channels,
                kernel_size=(self.gen_temporal_kernel_size, 4, 4), # (D, H, W) kernels
                stride=(1, 2, 2), # Stride 1 for temporal, 2 for spatial
                padding=(padding_temp, 1, 1), # (D_pad, H_pad, W_pad)
                bias=False
            )
            block['norm'] = nn.InstanceNorm3d(out_channels, affine=True) # Affine=False for FiLM True when disabled
            if self.frame_gaad_embedder is not None:
                block['film'] = FiLMLayer(out_channels, self.gaad_condition_dim)
            block['activation'] = nn.GELU()
            self.upsample_blocks.append(block)
            current_channels = out_channels

        # Final convolution to map to image channels
        # Padding for final_conv should ensure spatial dimensions are maintained if kernel is >1
        final_conv_padding_spatial = 1 if getattr(args, 'gen_final_conv_kernel_spatial', 3) > 1 else 0
        final_conv_padding_temporal = padding_temp # Keep consistent with other temporal convs
        
        self.final_conv = nn.Conv3d(
            current_channels, self.num_channels,
            kernel_size=(self.gen_temporal_kernel_size, getattr(args, 'gen_final_conv_kernel_spatial', 3), getattr(args, 'gen_final_conv_kernel_spatial', 3)),
            padding=(final_conv_padding_temporal, final_conv_padding_spatial, final_conv_padding_spatial)
        )
        self.final_activation = nn.Tanh()

        self.apply(init_weights_general) # Assuming init_weights_general is defined elsewhere
        self.logger.info(
            f"Derived Generator Config: InitCh={self.gen_init_channels}, "
            f"InitSpatialRes={self.gen_init_spatial_res}, NumUpsampleLayers={self.gen_num_upsampling_layers}, "
            f"TargetImageSize={self.image_size}, FinalCalcResTargetDim={self.gen_init_spatial_res * (2**self.gen_num_upsampling_layers)}, "
            f"GAADCondDim={self.gaad_condition_dim if self.frame_gaad_embedder else 'N/A'}"
        )

    def _normalize_bboxes(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # bboxes: (B, N_frames, NumRegions, 4) [x1, y1, x2, y2]
        # Output: (B, N_frames, NumRegions, 4) [norm_cx, norm_cy, norm_w, norm_h]
        x1, y1, x2, y2 = bboxes.unbind(-1)
        # Ensure W and H are not zero to prevent division by zero
        img_W = float(W) if W > 0 else 1.0
        img_H = float(H) if H > 0 else 1.0

        norm_cx = ((x1 + x2) / 2.0) / img_W
        norm_cy = ((y1 + y2) / 2.0) / img_H
        norm_w = (x2 - x1) / img_W
        norm_h = (y1 - y2) / img_H # Note: often y2 > y1, so h would be positive. If y is from top, (y2-y1) is correct.
        return torch.stack([norm_cx, norm_cy, norm_w, norm_h], dim=-1)


    def forward(self, latent_code: torch.Tensor, gaad_bboxes: Optional[torch.Tensor]) -> torch.Tensor:
        B = latent_code.shape[0]
        device = latent_code.device
        dtype_in = latent_code.dtype # Match output dtype to latent_code dtype

        # 1. Expand latent code
        x = self.fc_expand_latent(latent_code)
        x = x.view(
            B,
            self.gen_init_channels,
            self.num_predict_frames,
            self.gen_init_spatial_res,
            self.gen_init_spatial_res
        ).to(dtype_in) # Ensure dtype matches after view

        # 2. Prepare GAAD frame-wise conditions if applicable
        sequence_condition = None # Initialize to None
        if self.frame_gaad_embedder is not None:
            if gaad_bboxes is not None and \
               (gaad_bboxes.shape[0] != B or \
                gaad_bboxes.shape[1] != self.num_predict_frames or \
                gaad_bboxes.shape[2] != self.num_regions):
                self.logger.warning(
                    f"Generator GAAD bbox shape mismatch. Expected (B={B}, N_pred={self.num_predict_frames}, NumReg={self.num_regions}, 4), "
                    f"got {gaad_bboxes.shape}. Will use zero condition."
                )
                # Fallback to zero condition if shapes don't match critical dimensions
                frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)
            elif gaad_bboxes is not None: # Correct shape
                norm_bboxes = self._normalize_bboxes(gaad_bboxes.to(dtype_in), self.image_size[0], self.image_size[1])
                norm_bboxes_flat = norm_bboxes.view(B * self.num_predict_frames, -1)
                frame_conditions_flat = self.frame_gaad_embedder(norm_bboxes_flat)
            else: # No bboxes provided, but embedder exists
                self.logger.debug("GAAD Embedder present but no bboxes provided to Generator. Using zero condition.")
                frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)

            # Create sequence_condition by averaging frame conditions
            if frame_conditions_flat is not None:
                frame_conditions_reshaped = frame_conditions_flat.view(B, self.num_predict_frames, self.gaad_condition_dim)
                sequence_condition = torch.mean(frame_conditions_reshaped, dim=1).to(dtype_in) # (B, gaad_condition_dim)


        # 3. Spatio-temporal upsampling with FiLM
        for block_idx, block in enumerate(self.upsample_blocks):
            x = block['conv_transpose'](x)
            x = block['norm'](x)
            if 'film' in block and sequence_condition is not None:
                x = block['film'](x, sequence_condition)
            x = block['activation'](x)

        x = self.final_conv(x)
        generated_frames_sequence = self.final_activation(x)
        # Current shape: (B, num_img_channels, N_pred, H_intermediate, W_intermediate)

        # Permute to (B, N_pred, num_img_channels, H_intermediate, W_intermediate)
        generated_frames_sequence = generated_frames_sequence.permute(0, 2, 1, 3, 4)

        # Final check and adaptive pooling if necessary to match exact image_size
        # This is important because derived parameters might not lead to exact target dimensions.
        final_h_actual, final_w_actual = generated_frames_sequence.shape[-2:]
        if final_h_actual != self.image_size[0] or final_w_actual != self.image_size[1]:
            self.logger.debug(
                f"Generator output spatial size {final_h_actual}x{final_w_actual} before final pool, target {self.image_size}. "
                f"Performing adaptive pool for exact match."
            )
            # Permute for adaptive_avg_pool3d: (B, C, D, H, W)
            temp_permuted_for_pool = generated_frames_sequence.permute(0, 2, 1, 3, 4)
            pooled = F.adaptive_avg_pool3d(
                temp_permuted_for_pool,
                (self.num_predict_frames, self.image_size[0], self.image_size[1]) # (D_out, H_out, W_out)
            )
            generated_frames_sequence = pooled.permute(0, 2, 1, 3, 4) # Back to (B, N_pred, C, H, W)

        return generated_frames_sequence.to(dtype_in) # Ensure output dtype matches input

class RegionalDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_config = gaad_config
        self.disc_config = disc_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV01.Discriminator")

        self.image_size = (args.image_h, args.image_w)
        self.num_channels = video_config['num_channels']
        self.num_frames_to_discriminate = video_config.get("num_predict_frames", 1)
        if self.num_frames_to_discriminate == 0: self.num_frames_to_discriminate = 1

        self.num_regions = self.gaad_config.get('num_regions', 0)
        self.use_gaad_film_condition = disc_config.get("use_gaad_film_condition", getattr(args, 'disc_use_gaad_film_condition', False)) and self.num_regions > 0
        # Get spectral norm setting from disc_config (passed from args) or directly from args
        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", getattr(args, 'disc_apply_spectral_norm', True))


        if self.use_gaad_film_condition:
            self.gaad_condition_dim = disc_config.get("gaad_condition_dim_disc", getattr(args, 'disc_gaad_condition_dim_disc', 64))
            self.bbox_feature_dim = 4
            hidden_bbox_embed_dim = max(self.gaad_condition_dim, self.num_regions * self.bbox_feature_dim // 2)
            self.frame_gaad_embedder_disc = nn.Sequential(
                nn.Linear(self.num_regions * self.bbox_feature_dim, hidden_bbox_embed_dim),
                nn.GELU(),
                nn.Linear(hidden_bbox_embed_dim, self.gaad_condition_dim)
            )
            self.logger.info(f"Discriminator GAAD-FiLM conditioning ENABLED. Condition Dim: {self.gaad_condition_dim}")
        else:
            self.frame_gaad_embedder_disc = None
            self.gaad_condition_dim = 0
            self.logger.info("Discriminator GAAD-FiLM conditioning DISABLED.")


        self.disc_type = self.disc_config.get("type", getattr(args, 'discriminator_type', "spatio_temporal_cnn"))
        self.logger.info(f"Initializing Discriminator Type: {self.disc_type}")

        if self.disc_type == "spatio_temporal_cnn":
            min_input_dim = min(self.image_size[0], self.image_size[1])
            num_spatial_downsamples_target = int(math.log2(min_input_dim / 4)) if min_input_dim >=8 else 1
            max_possible_downsamples = int(math.log2(min_input_dim)) if min_input_dim > 0 else 0
            num_spatial_downsamples = max(1, min(num_spatial_downsamples_target, max_possible_downsamples))

            base_disc_channels = disc_config.get("base_disc_channels", getattr(args, 'disc_base_disc_channels', 64))
            cnn3d_channels_list = [base_disc_channels * (2**i) for i in range(num_spatial_downsamples)]
            max_disc_channels = disc_config.get("max_disc_channels", getattr(args, 'disc_max_disc_channels', 512))
            cnn3d_channels_list = [min(c, max_disc_channels) for c in cnn3d_channels_list]
            if not cnn3d_channels_list: cnn3d_channels_list = [base_disc_channels]

            temporal_kernel_size = disc_config.get("temporal_kernel_size", getattr(args, 'disc_temporal_kernel_size', 3))
            default_temporal_stride = 1

            layers = []
            in_c = self.num_channels
            current_d_dim = self.num_frames_to_discriminate
            current_h_dim = self.image_size[0]
            current_w_dim = self.image_size[1]

            for i, out_c in enumerate(cnn3d_channels_list):
                can_halve_spatial = current_h_dim >= 8 and current_w_dim >= 8
                spatial_stride = 2 if can_halve_spatial and i < num_spatial_downsamples else 1
                
                apply_temporal_stride_val = 2
                # MODIFICATION: Stride temporally only in the first layer if possible
                can_stride_temporally = current_d_dim > temporal_kernel_size and current_d_dim >= apply_temporal_stride_val # Check if stride makes sense
                actual_temporal_stride = apply_temporal_stride_val if can_stride_temporally and i == 0 else default_temporal_stride # Only in first block

                current_t_kernel = min(temporal_kernel_size, current_d_dim) if current_d_dim > 1 else 1
                current_t_padding = current_t_kernel // 2 if current_t_kernel > 1 else 0

                block = nn.ModuleDict()
                conv_layer = nn.Conv3d(
                    in_c, out_c,
                    kernel_size=(current_t_kernel, 4, 4),
                    stride=(actual_temporal_stride, spatial_stride, spatial_stride),
                    padding=(current_t_padding, 1, 1),
                    bias=False # Bias is False as InstanceNorm will handle it or FiLM implicitly does
                )
                if self.apply_spectral_norm:
                    block['conv'] = spectral_norm(conv_layer)
                    if i == 0: self.logger.info("Applying Spectral Norm to Discriminator Conv3D layers.")
                else:
                    block['conv'] = conv_layer

                block['norm'] = nn.InstanceNorm3d(out_c, affine=not self.use_gaad_film_condition) # Correct: affine=False if FiLM
                if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None:
                    block['film'] = FiLMLayer(out_c, self.gaad_condition_dim)
                block['activation'] = nn.LeakyReLU(0.2, inplace=True)
                layers.append(block)

                in_c = out_c
                # Calculate output dimensions after conv
                # D_out = floor((D_in + 2*P_d - K_d)/S_d) + 1
                if current_d_dim > 0:
                    current_d_dim = (current_d_dim + 2 * current_t_padding - (current_t_kernel -1) -1 ) // actual_temporal_stride + 1
                if current_h_dim > 0:
                    current_h_dim = (current_h_dim + 2 * 1 - (4-1) -1 ) // spatial_stride + 1
                if current_w_dim > 0:
                    current_w_dim = (current_w_dim + 2 * 1 - (4-1) -1 ) // spatial_stride + 1
                
                current_d_dim = max(1, current_d_dim)
                current_h_dim = max(1, current_h_dim)
                current_w_dim = max(1, current_w_dim)

            self.feature_extractor_blocks = nn.ModuleList(layers)
            
            # --- Shape Calculation with device handling ---
            _device_for_shape_calc = torch.device('cpu')
            try:
                if len(list(self.parameters())) > 0: _device_for_shape_calc = next(self.parameters()).device
            except StopIteration: pass # No parameters yet, keep CPU
            
            if hasattr(self.args, 'device') and self.args.device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("Requested CUDA device for shape calculation but CUDA not available. Using CPU.")
                _device_for_shape_calc = torch.device('cpu')

            test_input_shape = (1, self.num_channels, self.num_frames_to_discriminate, self.image_size[0], self.image_size[1])
            test_input = torch.randn(test_input_shape).to(_device_for_shape_calc)

            dummy_sequence_condition_disc = None
            original_embedder_device = None
            if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None:
                if len(list(self.frame_gaad_embedder_disc.parameters())) > 0:
                    original_embedder_device = next(self.frame_gaad_embedder_disc.parameters()).device
                    if original_embedder_device != _device_for_shape_calc: self.frame_gaad_embedder_disc.to(_device_for_shape_calc)
                
                dummy_bboxes_norm = torch.rand(1, self.num_frames_to_discriminate, self.num_regions, self.bbox_feature_dim).to(_device_for_shape_calc)
                norm_bboxes_flat_dummy = dummy_bboxes_norm.view(1 * self.num_frames_to_discriminate, -1)
                frame_cond_flat_dummy = self.frame_gaad_embedder_disc(norm_bboxes_flat_dummy)
                frame_cond_reshaped_dummy = frame_cond_flat_dummy.view(1, self.num_frames_to_discriminate, self.gaad_condition_dim)
                dummy_sequence_condition_disc = torch.mean(frame_cond_reshaped_dummy, dim=1)

                if original_embedder_device and original_embedder_device != _device_for_shape_calc: self.frame_gaad_embedder_disc.to(original_embedder_device)

            temp_features = test_input
            original_fe_devices = {} # Store original device for each block
            
            # Move blocks to calc device
            for i_fe, block_module_item_fe in enumerate(self.feature_extractor_blocks):
                if len(list(block_module_item_fe.parameters())) > 0:
                    original_fe_devices[i_fe] = next(block_module_item_fe.parameters()).device
                    if original_fe_devices[i_fe] != _device_for_shape_calc:
                        block_module_item_fe.to(_device_for_shape_calc)
            
            with torch.no_grad(): 
                for block_module_item_fe in self.feature_extractor_blocks:
                    temp_features = block_module_item_fe['conv'](temp_features)
                    temp_features = block_module_item_fe['norm'](temp_features)
                    if 'film' in block_module_item_fe and dummy_sequence_condition_disc is not None:
                        temp_features = block_module_item_fe['film'](temp_features, dummy_sequence_condition_disc)
                    temp_features = block_module_item_fe['activation'](temp_features)
            
            final_feature_map_shape_pre_pool = temp_features.shape

            self.adaptive_pool = nn.AdaptiveAvgPool3d((max(1, final_feature_map_shape_pre_pool[2]), 1, 1))
            self.adaptive_pool.to(_device_for_shape_calc) 
            
            with torch.no_grad():
                 pooled_features_test = self.adaptive_pool(temp_features.to(_device_for_shape_calc))
            
            # Restore original devices for FE blocks
            for i_fe, block_module_item_fe in enumerate(self.feature_extractor_blocks):
                if i_fe in original_fe_devices and original_fe_devices[i_fe] != _device_for_shape_calc:
                    block_module_item_fe.to(original_fe_devices[i_fe])
            # --- End Shape Calculation with device handling ---

            final_flattened_dim = pooled_features_test.numel() // pooled_features_test.shape[0]
            final_flattened_dim = max(1, final_flattened_dim)

            min_hidden_fc_dim = getattr(args, 'disc_min_hidden_fc_dim', 128)
            max_hidden_fc_dim = getattr(args, 'disc_max_hidden_fc_dim', 512)

            if final_flattened_dim <= min_hidden_fc_dim * 1.5 and final_flattened_dim > 0 :
                self.logger.info(f"Discriminator final FC: Using direct projection from {final_flattened_dim} to 1.")
                fc_layer = nn.Linear(final_flattened_dim, 1)
                if self.apply_spectral_norm: self.final_fc_layers = spectral_norm(fc_layer)
                else: self.final_fc_layers = fc_layer
            elif final_flattened_dim > 0 :
                hidden_fc_dim = max(min_hidden_fc_dim, final_flattened_dim // 2)
                hidden_fc_dim = min(hidden_fc_dim, max_hidden_fc_dim)
                self.logger.info(f"Discriminator final FC: Input {final_flattened_dim}, Hidden {hidden_fc_dim}, Output 1.")
                
                fc1 = nn.Linear(final_flattened_dim, hidden_fc_dim)
                fc2 = nn.Linear(hidden_fc_dim, 1)
                if self.apply_spectral_norm:
                    self.final_fc_layers = nn.Sequential(
                        spectral_norm(fc1),
                        nn.LeakyReLU(0.2, inplace=True),
                        spectral_norm(fc2)
                    )
                else:
                     self.final_fc_layers = nn.Sequential(
                        fc1,
                        nn.LeakyReLU(0.2, inplace=True),
                        fc2
                    )
            else:
                self.logger.error(f"Discriminator final_flattened_dim is {final_flattened_dim}, which is invalid. Defaulting to small FC.")
                self.final_fc_layers = nn.Linear(1,1) 

            self.logger.info(
                f"SpatioTemporalCNN Disc: Frames={self.num_frames_to_discriminate}, "
                f"Derived CNN3D_Channels={cnn3d_channels_list}, "
                f"FinalFeatMapShape (pre-pool)={final_feature_map_shape_pre_pool}, PooledFeatMapShape={pooled_features_test.shape}, FlattenedDimForFC={final_flattened_dim}, "
                f"FiLM Active={self.use_gaad_film_condition}, SpectralNorm Active={self.apply_spectral_norm}"
            )

        elif self.disc_type == "regional_cnn":
            self.logger.warning("Using 'regional_cnn' discriminator (first-frame only, no FiLM). Spectral norm can be applied if configured.")
            self.patch_size = disc_config.get("patch_size", getattr(args, 'disc_patch_size', 16))
            try:
                self.resize_transform_regional = T.Resize((self.patch_size, self.patch_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)
            except ImportError: 
                self.logger.error("torchvision.transforms (T) not available for RegionalDiscriminator 'regional_cnn' type. This will fail.")
                self.resize_transform_regional = None

            cnn_channels_2d = disc_config.get("cnn_channels_2d", getattr(args, 'disc_cnn_channels_2d', [64, 128, 256]))
            layers_2d = []
            in_c_2d = self.num_channels
            for i_2d, out_c_2d in enumerate(cnn_channels_2d):
                conv2d_layer = nn.Conv2d(in_c_2d, out_c_2d, kernel_size=4, stride=2, padding=1, bias=False)
                if self.apply_spectral_norm: 
                    layers_2d.append(spectral_norm(conv2d_layer))
                    if i_2d == 0: self.logger.info("Applying Spectral Norm to RegionalDiscriminator (regional_cnn) Conv2D layers.")
                else:
                    layers_2d.append(conv2d_layer)
                layers_2d.extend([
                     nn.InstanceNorm2d(out_c_2d, affine=True), 
                     nn.LeakyReLU(0.2, inplace=True)
                 ])
                in_c_2d = out_c_2d
            
            _temp_regional_feature_extractor_2d = nn.Sequential(*layers_2d)
            
            _device_for_shape_calc_reg = torch.device('cpu')
            try:
                if len(list(self.parameters())) > 0: _device_for_shape_calc_reg = next(self.parameters()).device
            except StopIteration: pass
            if hasattr(self.args, 'device') and self.args.device == 'cuda' and not torch.cuda.is_available(): _device_for_shape_calc_reg = torch.device('cpu')

            test_input_2d = torch.randn(1, self.num_channels, self.patch_size, self.patch_size).to(_device_for_shape_calc_reg)
            
            original_reg_fe_device = None
            if len(list(_temp_regional_feature_extractor_2d.parameters())) > 0:
                original_reg_fe_device = next(_temp_regional_feature_extractor_2d.parameters()).device
                if original_reg_fe_device != _device_for_shape_calc_reg: _temp_regional_feature_extractor_2d.to(_device_for_shape_calc_reg)

            with torch.no_grad():
                dummy_out_reg_fe = _temp_regional_feature_extractor_2d(test_input_2d)
                h_f, w_f = dummy_out_reg_fe.shape[-2:]

            if original_reg_fe_device and original_reg_fe_device != _device_for_shape_calc_reg:
                 _temp_regional_feature_extractor_2d.to(original_reg_fe_device)
            
            self.regional_feature_extractor_2d = _temp_regional_feature_extractor_2d
            final_feature_dim_per_region = cnn_channels_2d[-1] * h_f * w_f if cnn_channels_2d else 0
            final_feature_dim_per_region = max(1, final_feature_dim_per_region)
            
            fc_regional = nn.Linear(final_feature_dim_per_region, 1)
            if self.apply_spectral_norm: self.final_fc_layers_regional = spectral_norm(fc_regional)
            else: self.final_fc_layers_regional = fc_regional


            self.gaad_min_size_px_regional = self.gaad_config.get('min_size_px', 5)
            self.decomposition_type_regional = self.gaad_config.get('decomposition_type', "hybrid")
        else:
            raise ValueError(f"Unsupported discriminator type in __init__: '{self.disc_type}'")

        if 'init_weights_general' in globals() and callable(init_weights_general):
            self.apply(init_weights_general)
        else:
            self.logger.warning("init_weights_general not found. Skipping custom weight initialization for Discriminator.")


    def _normalize_bboxes_disc(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # bboxes: (B, N_frames_disc, NumRegions, 4) [x1, y1, x2, y2]
        # Output: (B, N_frames_disc, NumRegions, 4) [norm_cx, norm_cy, norm_w, norm_h]
        x1, y1, x2, y2 = bboxes.unbind(-1)
        img_W = float(W) if W > 0 else 1.0
        img_H = float(H) if H > 0 else 1.0

        norm_cx = ((x1 + x2) / 2.0) / img_W
        norm_cy = ((y1 + y2) / 2.0) / img_H
        norm_w = (x2 - x1).abs() / img_W 
        norm_h = (y2 - y1).abs() / img_H 
        return torch.stack([norm_cx, norm_cy, norm_w, norm_h], dim=-1)

    def forward(self, frames_pixels: torch.Tensor, gaad_bboxes_for_disc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_seq, C, H, W = frames_pixels.shape
        device = frames_pixels.device
        dtype_in = frames_pixels.dtype

        if N_seq < self.num_frames_to_discriminate:
            padding_needed = self.num_frames_to_discriminate - N_seq
            last_frame_repeated = frames_pixels[:, -1:, ...].repeat(1, padding_needed, 1, 1, 1)
            frames_to_process = torch.cat([frames_pixels, last_frame_repeated], dim=1)
        elif N_seq > self.num_frames_to_discriminate:
            frames_to_process = frames_pixels[:, :self.num_frames_to_discriminate, ...]
        else:
            frames_to_process = frames_pixels
        
        sequence_condition_disc = None
        if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None:
            if gaad_bboxes_for_disc is not None:
                N_bboxes_seq = gaad_bboxes_for_disc.shape[1]
                bboxes_to_process_for_film = gaad_bboxes_for_disc
                if N_bboxes_seq != self.num_frames_to_discriminate:
                     if N_bboxes_seq < self.num_frames_to_discriminate:
                         bbox_padding = self.num_frames_to_discriminate - N_bboxes_seq
                         last_bbox_set_repeated = gaad_bboxes_for_disc[:, -1:, ...].repeat(1, bbox_padding, 1, 1) 
                         bboxes_to_process_for_film = torch.cat([gaad_bboxes_for_disc, last_bbox_set_repeated], dim=1)
                     else: 
                         bboxes_to_process_for_film = gaad_bboxes_for_disc[:, :self.num_frames_to_discriminate, ...]

                if bboxes_to_process_for_film.shape[0] != B or \
                   bboxes_to_process_for_film.shape[1] != self.num_frames_to_discriminate or \
                   (self.num_regions > 0 and bboxes_to_process_for_film.shape[2] != self.num_regions):
                    self.logger.warning(
                        f"Discriminator GAAD bbox shape mismatch for FiLM. Expected (B={B}, N_disc_frames={self.num_frames_to_discriminate}, NumReg={self.num_regions if self.num_regions > 0 else 'Any'}), "
                        f"got {bboxes_to_process_for_film.shape if bboxes_to_process_for_film is not None else 'None'}. Using zero condition."
                    )
                    frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
                elif self.num_regions > 0 : # Only proceed if num_regions > 0 for normalization step
                    norm_bboxes_disc = self._normalize_bboxes_disc(bboxes_to_process_for_film.to(dtype_in), H, W)
                    norm_bboxes_flat_disc = norm_bboxes_disc.view(B * self.num_frames_to_discriminate, -1)
                    frame_conditions_flat_disc = self.frame_gaad_embedder_disc(norm_bboxes_flat_disc)
                else: # num_regions is 0, but FiLM is on somehow - this is an odd state.
                    self.logger.warning("FiLM condition active but num_regions is 0. Using zero condition.")
                    frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
            else: 
                self.logger.debug("Discriminator GAAD Embedder active but no bboxes provided. Using zero condition for FiLM.")
                frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)

            if frame_conditions_flat_disc is not None:
                frame_conditions_reshaped_disc = frame_conditions_flat_disc.view(B, self.num_frames_to_discriminate, self.gaad_condition_dim)
                sequence_condition_disc = torch.mean(frame_conditions_reshaped_disc, dim=1).to(dtype_in)

        if self.disc_type == "spatio_temporal_cnn":
            features = frames_to_process.permute(0, 2, 1, 3, 4).to(dtype_in) 
            for block_module in self.feature_extractor_blocks:
                features = block_module['conv'](features)
                features = block_module['norm'](features)
                if 'film' in block_module and sequence_condition_disc is not None:
                    features = block_module['film'](features, sequence_condition_disc)
                features = block_module['activation'](features)
            
            features = self.adaptive_pool(features)
            features_flat = features.view(B, -1)
            logits = self.final_fc_layers(features_flat)
            return logits.to(dtype_in)

        elif self.disc_type == "regional_cnn":
            if self.resize_transform_regional is None:
                 raise RuntimeError("resize_transform_regional not initialized in RegionalDiscriminator 'regional_cnn' type.")
            frame_to_disc_regional = frames_to_process[:, 0, ...].to(dtype_in) 
            gaad_bboxes_list_regional = []
            for b_idx in range(B):
                frame_dims_regional=(W,H); max_w_scalar=float(W); max_h_scalar=float(H)
                if 'golden_subdivide_rect_fixed_n' not in globals() or 'phi_spiral_patch_centers_fixed_n' not in globals():
                    self.logger.error("GAAD utility functions not found for 'regional_cnn' discriminator.")
                    frame_bboxes_reg = torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_regions, dtype=dtype_in, device=device) if self.num_regions > 0 else torch.empty(0,4,dtype=dtype_in, device=device)
                elif self.num_regions == 0: # No regions to generate for regional_cnn
                    frame_bboxes_reg = torch.empty(0,4,dtype=dtype_in, device=device)
                elif self.decomposition_type_regional == "hybrid":
                    num_subdivide=self.num_regions//2; num_spiral=self.num_regions-num_subdivide; bboxes_for_item=[]
                    if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims_regional,num_subdivide,device,dtype_in,self.gaad_min_size_px_regional))
                    if num_spiral > 0:
                         spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims_regional, num_spiral, device, dtype_in); patch_base_size = min(frame_dims_regional); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype_in); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; EPS_local=1e-5; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS_local); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS_local); min_for_x2=spiral_bboxes_current[:,0]+EPS_local; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS_local; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); bboxes_for_item.append(spiral_bboxes_current)
                    frame_bboxes_reg = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else (torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_regions, dtype=dtype_in, device=device) if self.num_regions > 0 else torch.empty(0,4,dtype=dtype_in, device=device))
                elif self.decomposition_type_regional == "spiral":
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims_regional, self.num_regions, device, dtype_in); patch_base_size = min(frame_dims_regional); spiral_bboxes_current = torch.zeros(self.num_regions, 4, device=device, dtype=dtype_in); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; EPS_local=1e-5; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS_local); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS_local); min_for_x2=spiral_bboxes_current[:,0]+EPS_local; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS_local; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); frame_bboxes_reg = spiral_bboxes_current
                else: # subdivide
                    frame_bboxes_reg = golden_subdivide_rect_fixed_n(frame_dims_regional,self.num_regions,device,dtype_in,self.gaad_min_size_px_regional)

                if self.num_regions > 0 and frame_bboxes_reg.shape[0] < self.num_regions:
                    num_to_pad=self.num_regions-frame_bboxes_reg.shape[0]; padding_box=frame_bboxes_reg[-1:].clone() if frame_bboxes_reg.shape[0]>0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]],dtype=dtype_in,device=device); padding=padding_box.repeat(num_to_pad,1); frame_bboxes_reg=torch.cat([frame_bboxes_reg, padding], dim=0)
                elif self.num_regions > 0 and frame_bboxes_reg.shape[0] > self.num_regions:
                    frame_bboxes_reg=frame_bboxes_reg[:self.num_regions]
                gaad_bboxes_list_regional.append(frame_bboxes_reg)
            
            # If no regions were generated (e.g. self.num_regions == 0), we can't proceed with regional features.
            # This case might imply a global discriminator logic for regional_cnn if num_regions is 0.
            # For now, assume num_regions > 0 for this path.
            if self.num_regions == 0:
                self.logger.warning("RegionalDiscriminator 'regional_cnn' called with num_regions=0. This is not typical. Producing zero logits.")
                return torch.zeros(B, 1, device=device, dtype=dtype_in)

            gaad_bboxes_batch_regional = torch.stack(gaad_bboxes_list_regional)
            all_regional_features = []
            for b_idx in range(B):
                batch_region_features = []
                for r_idx in range(self.num_regions):
                    x1,y1,x2,y2 = gaad_bboxes_batch_regional[b_idx,r_idx].round().int().tolist(); x1_c,y1_c=max(0,x1),max(0,y1); x2_c,y2_c=min(W,x2),min(H,y2)
                    if x1_c >= x2_c or y1_c >= y2_c:
                         feat_dim_input_to_fc = self.final_fc_layers_regional.in_features
                         patch_features_reg = torch.zeros(1, feat_dim_input_to_fc, device=device, dtype=dtype_in)
                    else:
                         patch = frame_to_disc_regional[b_idx, :, y1_c:y2_c, x1_c:x2_c] 
                         resized_patch = self.resize_transform_regional(patch).unsqueeze(0) 
                         patch_features_extracted = self.regional_feature_extractor_2d(resized_patch) 
                         patch_features_reg = patch_features_extracted.view(1, -1) 
                    batch_region_features.append(patch_features_reg)
                all_regional_features.append(torch.cat(batch_region_features, dim=0)) 
            
            regional_features_tensor = torch.stack(all_regional_features) 
            aggregated_features = torch.mean(regional_features_tensor, dim=1) 
            logits = self.final_fc_layers_regional(aggregated_features) 
            return logits.to(dtype_in)
        else:
            raise NotImplementedError(f"Discriminator forward not implemented for type {self.disc_type}")




           
            
            
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
                 model: "WuBuGAADHybridGenNet",
                 discriminator: "RegionalDiscriminator",
                 optimizer_enc_gen: torch.optim.Optimizer,
                 optimizer_disc: torch.optim.Optimizer,
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 args: argparse.Namespace,
                 rank: int,
                 world_size: int,
                 ddp_active: bool):

        self.model = model
        self.discriminator = discriminator
        self.optimizer_enc_gen = optimizer_enc_gen
        self.optimizer_disc = optimizer_disc
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.ddp_active = ddp_active
        self.am_main_process = (rank == 0)
        self.logger = logging.getLogger("WuBuGAADHybridGenV01.Trainer")

        self.video_config = model.video_config # type: ignore
        self.gaad_appearance_config = model.gaad_appearance_config # type: ignore

        self.lambda_recon = args.lambda_recon
        self.lambda_kl = args.lambda_kl # Current effective lambda_kl for the VAE loss
        self.lambda_gan = args.lambda_gan

        self.scaler_enc_gen = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.scaler_disc = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric_val = -float('inf') if args.val_primary_metric in ["avg_val_psnr", "avg_val_ssim"] else float('inf')
        self.last_val_metrics: Dict[str, Any] = {}

        if self.am_main_process:
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        self.lpips_loss_fn: Optional[lpips.LPIPS] = None # type: ignore
        self.ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None # type: ignore

        if self.am_main_process and self.args.use_lpips_for_verification:
             if LPIPS_AVAILABLE and lpips is not None:
                 try:
                     self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device) # type: ignore
                     self.logger.info("LPIPS metric enabled.")
                 except Exception as e_lpips:
                     self.logger.warning(f"LPIPS init failed: {e_lpips}. Disabling LPIPS.")
                     self.lpips_loss_fn = None
             else:
                 self.logger.warning("LPIPS lib unavailable. Skip LPIPS.")

        if self.am_main_process and TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
             try:
                 self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device) # type: ignore
                 self.logger.info("SSIM metric enabled.")
             except Exception as e:
                 self.logger.warning(f"SSIM init failed: {e}. Skip SSIM.")
                 self.ssim_metric = None
        elif self.am_main_process:
             self.logger.warning("torchmetrics SSIM unavailable. Skip SSIM.")

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        if self.grad_accum_steps > 1 and self.am_main_process:
            self.logger.info(f"Gradient accumulation enabled: {self.grad_accum_steps} steps.")

        self.fixed_noise_for_sampling: Optional[torch.Tensor] = None

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()

    def _compute_recon_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_x, x)

    @torch.no_grad()
    def _log_samples_to_wandb(self,
                              tag_prefix: str,
                              frames_to_log: Optional[torch.Tensor], # Made Optional
                              num_frames_per_sequence_to_log: int = 1,
                              num_sequences_to_log_max: int = 2):
        if not (self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run): # type: ignore
            return
        if frames_to_log is None or frames_to_log.numel() == 0:
            self.logger.debug(f"Skipping WandB image log for {tag_prefix} due to None or empty frames_to_log.")
            return

        B_log, N_seq_log, C_log, H_log, W_log = frames_to_log.shape
        num_to_actually_log_sequences = min(B_log, num_sequences_to_log_max)
        num_frames_to_log_this_seq = min(N_seq_log, num_frames_per_sequence_to_log)
        wandb_images_for_log = []
        for b_idx in range(num_to_actually_log_sequences):
            for frame_idx_in_seq in range(num_frames_to_log_this_seq):
                frame_tensor = frames_to_log[b_idx, frame_idx_in_seq, ...].cpu().float()
                img_0_1 = (frame_tensor.clamp(-1,1) + 1) / 2.0
                caption = f"{tag_prefix} Sample {b_idx} Frame {frame_idx_in_seq} Ep{self.current_epoch+1} GStep{self.global_step}"
                wandb_images_for_log.append(wandb.Image(img_0_1, caption=caption)) # type: ignore
        if wandb_images_for_log:
            wandb.log({f"samples/{tag_prefix}": wandb_images_for_log}, step=self.global_step) # type: ignore
            self.logger.debug(f"Logged {len(wandb_images_for_log)} image frames to WandB with prefix samples/{tag_prefix}")

    def _train_discriminator_step(self, real_frames_full: torch.Tensor, m_ref: "WuBuGAADHybridGenNet", d_ref: "RegionalDiscriminator") -> Dict[str, torch.Tensor]:
        B = real_frames_full.shape[0]
        device = real_frames_full.device
        dtype_model = next(m_ref.parameters()).dtype # Assuming m_ref has params

        frames_for_d_processing = real_frames_full[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)

        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model)
        fake_labels = torch.zeros(B, 1, device=device, dtype=dtype_model)
        losses_d_micro: Dict[str, torch.Tensor] = {}

        # Set requires_grad appropriately for D training phase
        for p in d_ref.parameters(): p.requires_grad = True
        for p in m_ref.parameters(): p.requires_grad = False

        gaad_bboxes_for_d_real_cond = None
        if d_ref.use_gaad_film_condition: # Check D's config if it uses FiLM
            with torch.no_grad():
                # Encoder part of m_ref is used to get bboxes for real frames
                _, _, gaad_bboxes_from_encoder_full, _ = m_ref.encode(real_frames_full.to(device, dtype_model))
            if gaad_bboxes_from_encoder_full is not None:
                gaad_bboxes_for_d_real_cond = gaad_bboxes_from_encoder_full[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'):
            real_logits = d_ref(frames_for_d_processing, gaad_bboxes_for_d_real_cond)
            loss_d_real = self.adversarial_loss(real_logits, real_labels)

            with torch.no_grad(): # Generator pass for fake samples
                fake_frames_full_sequence, _, _, bboxes_used_for_fake_gen = m_ref(real_frames_full) # G generates based on real_frames input
                fake_frames_for_d_processing = fake_frames_full_sequence[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
                
                gaad_bboxes_for_d_fake_cond = None
                if d_ref.use_gaad_film_condition and bboxes_used_for_fake_gen is not None:
                    gaad_bboxes_for_d_fake_cond = bboxes_used_for_fake_gen[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)

            fake_logits = d_ref(fake_frames_for_d_processing.detach(), gaad_bboxes_for_d_fake_cond) # Detach fake frames
            loss_d_fake = self.adversarial_loss(fake_logits, fake_labels)
            
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total_scaled_for_accum_micro = loss_d_total_micro / self.grad_accum_steps

        self.scaler_disc.scale(loss_d_total_scaled_for_accum_micro).backward()
        
        losses_d_micro['loss_d_real_micro'] = loss_d_real.detach()
        losses_d_micro['loss_d_fake_micro'] = loss_d_fake.detach()
        losses_d_micro['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d_micro

    def _train_generator_step(self, real_frames_full: torch.Tensor, m_ref: "WuBuGAADHybridGenNet", d_ref: "RegionalDiscriminator") \
                              -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        B = real_frames_full.shape[0]
        device = real_frames_full.device
        dtype_model = next(m_ref.parameters()).dtype
        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model)
        losses_g_micro: Dict[str, torch.Tensor] = {}
        recon_frames_for_log: Optional[torch.Tensor] = None

        # Set requires_grad appropriately for G training phase
        for p in d_ref.parameters(): p.requires_grad = False
        for p in m_ref.parameters(): p.requires_grad = True

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'):
            recon_frames_pred_sequence, mu, logvar, bboxes_used_by_decoder = m_ref(real_frames_full.to(device, dtype_model))

            # Store for logging if interval matches (global_step increments AFTER this full macro-batch)
            if self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and \
               ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0 or self.global_step == 0) :
                recon_frames_for_log = recon_frames_pred_sequence.detach().clone()

            start_idx_target = self.video_config["num_input_frames"]
            actual_pred_len_for_loss = min(
                recon_frames_pred_sequence.shape[1],
                real_frames_full.shape[1] - start_idx_target,
                self.video_config["num_predict_frames"]
            )
            if actual_pred_len_for_loss <= 0:
                self.logger.error(f"G_Step: Cannot compute recon loss (actual_pred_len_for_loss={actual_pred_len_for_loss}). PredShape1:{recon_frames_pred_sequence.shape[1]}, RealAvailForTarget:{real_frames_full.shape[1] - start_idx_target}, ConfigPredFrames:{self.video_config['num_predict_frames']}")
                loss_recon = torch.tensor(1000.0, device=device, dtype=dtype_model) # High placeholder if error
            else:
                target_frames_for_recon = real_frames_full[:, start_idx_target : start_idx_target + actual_pred_len_for_loss, ...].to(device, dtype_model)
                recon_frames_for_loss_calc = recon_frames_pred_sequence[:, :actual_pred_len_for_loss, ...]
                loss_recon = self._compute_recon_loss(recon_frames_for_loss_calc, target_frames_for_recon)
            
            loss_kl = self._compute_kl_loss(mu, logvar)

            fake_frames_for_g_adv = recon_frames_pred_sequence[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
            gaad_bboxes_for_g_adv_cond = None
            if d_ref.use_gaad_film_condition and bboxes_used_by_decoder is not None: # Check D's config
                gaad_bboxes_for_g_adv_cond = bboxes_used_by_decoder[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
            
            fake_logits_gen = d_ref(fake_frames_for_g_adv, gaad_bboxes_for_g_adv_cond)
            loss_g_adv = self.adversarial_loss(fake_logits_gen, real_labels)
            
            loss_g_total_micro = (self.lambda_recon * loss_recon + 
                                  self.lambda_kl * loss_kl + 
                                  self.lambda_gan * loss_g_adv)
            loss_g_total_scaled_for_accum_micro = loss_g_total_micro / self.grad_accum_steps

        self.scaler_enc_gen.scale(loss_g_total_scaled_for_accum_micro).backward()
        
        losses_g_micro['loss_recon_micro'] = loss_recon.detach()
        losses_g_micro['loss_kl_micro'] = loss_kl.detach()
        losses_g_micro['loss_g_adv_micro'] = loss_g_adv.detach()
        losses_g_micro['loss_g_total_micro'] = loss_g_total_micro.detach()
        return losses_g_micro, recon_frames_for_log

    def train(self, start_epoch:int=0, initial_global_step:int=0):
        self.global_step = initial_global_step
        self.current_epoch = start_epoch

        if self.am_main_process:
            self.logger.info(f"Starting training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch}, InitialGStep: {initial_global_step}")
            self.logger.info(f"Grad Accum: {self.grad_accum_steps}, Lambda_KL: {self.lambda_kl:.2e}, Lambda_Recon: {self.lambda_recon}, Lambda_GAN: {self.lambda_gan}")
            if self.args.wandb_log_fixed_noise_samples_interval > 0 and self.args.num_val_samples_to_log > 0:
                m_ref_temp = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
                self.fixed_noise_for_sampling = torch.randn(
                    self.args.num_val_samples_to_log, self.args.latent_dim,
                    device=self.device, dtype=next(m_ref_temp.parameters()).dtype
                )
                self.logger.info(f"Created fixed noise tensor for sampling: {self.fixed_noise_for_sampling.shape}")

        # Accumulators for Q-Controller (average over grad_accum_steps)
        accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
        accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0
        
        # Accumulators for logging interval (sum over log_interval * grad_accum_steps micro-batches)
        log_interval_accum_losses = defaultdict(float)
        log_interval_items_processed = 0 # Counts individual samples

        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        d_ref = self.discriminator.module if self.ddp_active and isinstance(self.discriminator, DDP) else self.discriminator

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process: self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting...")
            if self.ddp_active and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch) # type: ignore

            m_ref.train(); d_ref.train()
            
            # Zero gradients at the beginning of each new epoch / accumulation cycle start
            self.optimizer_disc.zero_grad(set_to_none=True)
            self.optimizer_enc_gen.zero_grad(set_to_none=True)
            
            dataset_len_approx = len(self.train_loader.dataset) // self.world_size if hasattr(self.train_loader.dataset, '__len__') else None # type: ignore
            num_micro_batches_epoch = dataset_len_approx // self.train_loader.batch_size if dataset_len_approx and self.train_loader.batch_size else None # type: ignore

            prog_bar = tqdm(self.train_loader, desc=f"E{epoch+1}", disable=not self.am_main_process or os.getenv('CI')=='true', dynamic_ncols=True, total=num_micro_batches_epoch)

            for batch_idx, batch_frames_raw in enumerate(prog_bar):
                batch_frames = batch_frames_raw.to(self.device)
                batch_size_micro = batch_frames.size(0)

                # --- D Step ---
                # (requires_grad set inside the function)
                losses_d_micro = self._train_discriminator_step(batch_frames, m_ref, d_ref)
                if torch.isfinite(losses_d_micro['loss_d_total_micro']): accum_d_total_q += losses_d_micro['loss_d_total_micro'].item()
                if torch.isfinite(losses_d_micro['loss_d_real_micro']): accum_d_real_q += losses_d_micro['loss_d_real_micro'].item()
                if torch.isfinite(losses_d_micro['loss_d_fake_micro']): accum_d_fake_q += losses_d_micro['loss_d_fake_micro'].item()
                for k, v_d in losses_d_micro.items():
                    if torch.isfinite(v_d): log_interval_accum_losses[k.replace('_micro','_agg')] += v_d.item() * batch_size_micro

                # --- G Step ---
                # (requires_grad set inside the function)
                losses_g_micro, recon_frames_for_logging = self._train_generator_step(batch_frames, m_ref, d_ref)
                if torch.isfinite(losses_g_micro['loss_g_total_micro']): accum_g_total_q += losses_g_micro['loss_g_total_micro'].item()
                if torch.isfinite(losses_g_micro['loss_recon_micro']): accum_g_recon_q += losses_g_micro['loss_recon_micro'].item()
                if torch.isfinite(losses_g_micro['loss_kl_micro']): accum_g_kl_q += losses_g_micro['loss_kl_micro'].item()
                if torch.isfinite(losses_g_micro['loss_g_adv_micro']): accum_g_adv_q += losses_g_micro['loss_g_adv_micro'].item()
                for k, v_g in losses_g_micro.items():
                    if torch.isfinite(v_g): log_interval_accum_losses[k.replace('_micro','_agg')] += v_g.item() * batch_size_micro
                
                log_interval_items_processed += batch_size_micro

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Finalize grad stats from accumulated micro-batches
                    if hasattr(self.optimizer_disc, 'grad_stats'):
                        num_params_d_total = sum(p.numel() for group in self.optimizer_disc.param_groups for p in group['params'] if p.requires_grad) # Check current requires_grad
                        self.optimizer_disc.grad_stats.finalize_step_stats(num_params_d_total) # type: ignore
                    if hasattr(self.optimizer_enc_gen, 'grad_stats'):
                        num_params_g_total = sum(p.numel() for group in self.optimizer_enc_gen.param_groups for p in group['params'] if p.requires_grad)
                        self.optimizer_enc_gen.grad_stats.finalize_step_stats(num_params_g_total) # type: ignore

                    # Prepare MACRO-batch average losses for Q-Controllers
                    avg_g_total_macro = accum_g_total_q / self.grad_accum_steps
                    avg_g_recon_macro = accum_g_recon_q / self.grad_accum_steps
                    avg_g_kl_macro = accum_g_kl_q / self.grad_accum_steps
                    avg_g_adv_macro = accum_g_adv_q / self.grad_accum_steps
                    avg_d_total_macro = accum_d_total_q / self.grad_accum_steps
                    avg_d_real_macro = accum_d_real_q / self.grad_accum_steps
                    avg_d_fake_macro = accum_d_fake_q / self.grad_accum_steps
                    
                    # Q-Controller for Discriminator
                    # Set D grads true, G grads false FOR THIS OPTIMIZER'S Q-update and step
                    for p in d_ref.parameters(): p.requires_grad = True
                    for p in m_ref.parameters(): p.requires_grad = False
                    if hasattr(self.optimizer_disc, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_disc.q_controller_update_and_set_hyperparams( # type: ignore
                            avg_losses_dict={'loss_g_total': avg_g_total_macro, 'loss_g_adv': avg_g_adv_macro,
                                             'loss_d_total': avg_d_total_macro, 'loss_d_real': avg_d_real_macro, 'loss_d_fake': avg_d_fake_macro},
                            current_lambda_kl_value=self.lambda_kl)
                    if self.args.global_max_grad_norm > 0: self.scaler_disc.unscale_(self.optimizer_disc); torch.nn.utils.clip_grad_norm_(d_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_disc.step(self.optimizer_disc); self.scaler_disc.update()
                    
                    # Q-Controller for Generator
                    # Set G grads true, D grads false FOR THIS OPTIMIZER'S Q-update and step
                    for p in d_ref.parameters(): p.requires_grad = False
                    for p in m_ref.parameters(): p.requires_grad = True
                    if hasattr(self.optimizer_enc_gen, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_enc_gen.q_controller_update_and_set_hyperparams( # type: ignore
                            avg_losses_dict={'loss_g_total': avg_g_total_macro, 'loss_g_recon': avg_g_recon_macro, 'loss_g_kl': avg_g_kl_macro,
                                             'loss_g_adv': avg_g_adv_macro, 'loss_d_total': avg_d_total_macro},
                            current_lambda_kl_value=self.lambda_kl)
                    if self.args.global_max_grad_norm > 0: self.scaler_enc_gen.unscale_(self.optimizer_enc_gen); torch.nn.utils.clip_grad_norm_(m_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_enc_gen.step(self.optimizer_enc_gen); self.scaler_enc_gen.update()
                    
                    # Zero grads for the next accumulation cycle
                    self.optimizer_disc.zero_grad(set_to_none=True)
                    self.optimizer_enc_gen.zero_grad(set_to_none=True)

                    self.global_step += 1

                    # Reset Q-accumulators
                    accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
                    accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0
                    
                    # Logging logic (same as before, using log_interval_accum_losses)
                    if self.global_step % self.args.log_interval == 0 and log_interval_items_processed > 0 and self.am_main_process:
                        log_metrics = {f"train/{k.replace('_agg','')}": v / log_interval_items_processed for k, v in log_interval_accum_losses.items()}
                        lr_g = self.optimizer_enc_gen.param_groups[0]['lr']; lr_d = self.optimizer_disc.param_groups[0]['lr']
                        log_metrics.update({"train/lr_gen": lr_g, "train/lr_disc": lr_d, "epoch_frac": epoch + ((batch_idx + 1) / (num_micro_batches_epoch or 1)), "global_step": self.global_step, "train/lambda_kl_eff": self.lambda_kl})
                        q_ctrl_gen_info = getattr(self.optimizer_enc_gen, 'get_q_controller_info', lambda: None)(); q_ctrl_disc_info = getattr(self.optimizer_disc, 'get_q_controller_info', lambda: None)()
                        if q_ctrl_gen_info: log_metrics.update({f"q_ctrl_gen/{k.replace('_', '')}":v for k,v in q_ctrl_gen_info.items()})
                        if q_ctrl_disc_info: log_metrics.update({f"q_ctrl_disc/{k.replace('_', '')}":v for k,v in q_ctrl_disc_info.items()})
                        
                        gt, dt = log_metrics.get('train/loss_g_total', -1), log_metrics.get('train/loss_d_total', -1)
                        gr, gk, ga = log_metrics.get('train/loss_recon', -1), log_metrics.get('train/loss_kl',-1), log_metrics.get('train/loss_g_adv',-1)
                        dr, df = log_metrics.get('train/loss_d_real',-1), log_metrics.get('train/loss_d_fake',-1)
                        qeg, qed = log_metrics.get('q_ctrl_gen/epsilon',-1), log_metrics.get('q_ctrl_disc/epsilon',-1)
                        qag, qad = log_metrics.get('q_ctrl_gen/lastchosenaction',{}), log_metrics.get('q_ctrl_disc/lastchosenaction',{})
                        qslg, qsld = (qag.get('lr_scale',1.0) if isinstance(qag,dict) else 1.0), (qad.get('lr_scale',1.0) if isinstance(qad,dict) else 1.0)

                        log_str = (f"E{epoch+1} S{self.global_step} | G_tot:{gt:.3f}(Rec:{gr:.3f} KL:{gk:.3f} Adv:{ga:.3f}) | "
                                   f"D_tot:{dt:.3f}(Real:{dr:.3f} Fake:{df:.3f}) | LR(G/D):{lr_g:.1e}/{lr_d:.1e} | "
                                   f"Q_Eps(G/D):{qeg:.2f}/{qed:.2f} Q_Scl(G/D):{qslg:.2f}/{qsld:.2f}")
                        prog_bar.set_postfix_str(f"G:{gt:.2f} D:{dt:.2f} Rec:{gr:.3f}", refresh=True)
                        self.logger.info(log_str)
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(log_metrics, step=self.global_step) # type: ignore
                        log_interval_accum_losses = defaultdict(float); log_interval_items_processed = 0

                    # Log training reconstruction samples (same logic)
                    if self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and \
                       self.global_step % self.args.wandb_log_train_recon_interval == 0 and recon_frames_for_logging is not None:
                        self._log_samples_to_wandb("train_recon", recon_frames_for_logging, recon_frames_for_logging.shape[1], self.args.num_val_samples_to_log)
                        self._log_samples_to_wandb("train_context", batch_frames[:, :self.video_config["num_input_frames"], ...], self.video_config["num_input_frames"], self.args.num_val_samples_to_log)
                        s_idx_target_log = self.video_config["num_input_frames"]; gt_len_log = min(self.video_config["num_predict_frames"], batch_frames.shape[1] - s_idx_target_log)
                        if gt_len_log > 0: self._log_samples_to_wandb("train_ground_truth", batch_frames[:, s_idx_target_log : s_idx_target_log + gt_len_log, ...], gt_len_log, self.args.num_val_samples_to_log)
                    
                    # Log fixed noise samples (same logic)
                    if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and \
                       self.global_step % self.args.wandb_log_fixed_noise_samples_interval == 0 and self.fixed_noise_for_sampling is not None:
                        m_ref.eval()
                        with torch.no_grad():
                            # ... (fixed noise sampling logic remains the same) ...
                            num_fs = self.fixed_noise_for_sampling.shape[0]; npf_cfg = self.video_config["num_predict_frames"]
                            nra_cfg = self.gaad_appearance_config["num_regions"]; fd_cfg = (self.args.image_w, self.args.image_h)
                            dbb_list_fixed = []
                            if nra_cfg > 0:
                                for _ in range(num_fs):
                                    cs_bbox_1f = golden_subdivide_rect_fixed_n(fd_cfg, nra_cfg, device=self.device, dtype=self.fixed_noise_for_sampling.dtype, min_size_px=self.gaad_appearance_config['min_size_px']) # type: ignore
                                    bss_all_f_rep = cs_bbox_1f.unsqueeze(0).repeat(npf_cfg, 1, 1)
                                    dbb_list_fixed.append(bss_all_f_rep)
                                dbb_batch_fixed = torch.stack(dbb_list_fixed) if dbb_list_fixed else None
                            else: dbb_batch_fixed = None
                            
                            if dbb_batch_fixed is not None or nra_cfg == 0 : # Proceed if bboxes generated or not needed
                                fixed_noise_samples_gen = m_ref.decode(self.fixed_noise_for_sampling, dbb_batch_fixed)
                                self._log_samples_to_wandb("fixed_noise_generated", fixed_noise_samples_gen, fixed_noise_samples_gen.shape[1], num_fs)
                        m_ref.train()

                    if self.args.save_interval > 0 and self.global_step > 0 and (self.global_step % self.args.save_interval == 0) and self.am_main_process:
                        # Use the macro-batch averages for intermediate checkpoint metrics
                        chkpt_metrics = {'train_loss_g_total_macro': avg_g_total_macro, 'train_loss_d_total_macro': avg_d_total_macro}
                        self._save_checkpoint(is_intermediate=True, metrics=chkpt_metrics)

            # End of micro-batch loop for epoch
            # Handle any remaining gradients if epoch ends mid-accumulation cycle (grad_accum_steps > 1)
            # This part is mostly for completeness if an epoch doesn't divide evenly by grad_accum_steps.
            # However, with drop_last=True in DataLoader, this might be less common unless dataset size itself is not a multiple.
            # For simplicity, we assume the main loop handles accumulation correctly. If an epoch ends, the partial grads
            # would be stepped or zeroed at the start of the next epoch's accumulation.
            # The provided structure already zeroes grads after the accumulation step, which is fine.

            # --- End of Epoch Actions ---
            if self.am_main_process:
                 final_avg_g_loss = log_metrics.get('train/loss_g_total', float('nan')) if 'log_metrics' in locals() and log_interval_items_processed == 0 else \
                                  (log_interval_accum_losses['loss_g_total_agg'] / log_interval_items_processed if log_interval_items_processed > 0 else float('nan'))
                 final_avg_d_loss = log_metrics.get('train/loss_d_total', float('nan')) if 'log_metrics' in locals() and log_interval_items_processed == 0 else \
                                  (log_interval_accum_losses['loss_d_total_agg'] / log_interval_items_processed if log_interval_items_processed > 0 else float('nan'))

                 self.logger.info(f"Epoch {epoch+1} finished. Approx Avg Loss (last interval or partial): G:{final_avg_g_loss:.4f}, D:{final_avg_d_loss:.4f}")
                 if self.args.wandb and WANDB_AVAILABLE and wandb.run: # type: ignore
                     wandb.log({"epoch": epoch+1, 
                                "epoch_avg_train_loss_g_approx": final_avg_g_loss if np.isfinite(final_avg_g_loss) else -1.0, 
                                "epoch_avg_train_loss_d_approx": final_avg_d_loss if np.isfinite(final_avg_d_loss) else -1.0}, 
                               step=self.global_step)

            if self.val_loader and self.am_main_process: # Validation only on main process
                val_metrics = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if val_metrics: # val_metrics can be None if val_loader is None
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run: # type: ignore
                        wandb.log({f"val/{k}":v for k,v in val_metrics.items()}, step=self.global_step)
                    
                    # Check for best model based on primary metric
                    metric_to_check = self.args.val_primary_metric
                    current_val_for_best = val_metrics.get(metric_to_check, 
                                                           float('inf') if metric_to_check not in ["avg_val_psnr", "avg_val_ssim"] else -float('inf'))
                    is_better = False
                    if metric_to_check in ["avg_val_psnr", "avg_val_ssim"]: # Higher is better
                        is_better = current_val_for_best > self.best_val_metric_val
                    else: # Lower is better (MSE, LPIPS)
                        is_better = current_val_for_best < self.best_val_metric_val
                    
                    if is_better:
                        self.logger.info(f"New best val metric ({metric_to_check}): {current_val_for_best:.4f} (prev: {self.best_val_metric_val:.4f}). Saving best model.")
                        self.best_val_metric_val = current_val_for_best
                        self._save_checkpoint(is_best=True, metrics=val_metrics)
            
            if self.am_main_process: # Save regular end-of-epoch checkpoint
                epoch_end_metrics = self.last_val_metrics.copy() if self.last_val_metrics else {}
                # Add training losses if available and finite
                if 'final_avg_g_loss' in locals() and np.isfinite(final_avg_g_loss): epoch_end_metrics["epoch_end_train_loss_g_avg_approx"] = final_avg_g_loss
                if 'final_avg_d_loss' in locals() and np.isfinite(final_avg_d_loss): epoch_end_metrics["epoch_end_train_loss_d_avg_approx"] = final_avg_d_loss
                self._save_checkpoint(metrics=epoch_end_metrics) # Saves as epX_stepY.pt

    # --- Validation, Checkpointing, Sampling methods remain largely the same ---
    # (Copied from previous correct version for completeness)
    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Optional[Dict[str, float]]:
        if not self.val_loader or not self.am_main_process: return None
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        d_ref = self.discriminator.module if self.ddp_active and isinstance(self.discriminator, DDP) else self.discriminator # Not used in val, but good practice
        m_ref.eval(); # d_ref.eval() # Not strictly needed for VAE validation

        total_recon_loss_mse_sum = 0.0; total_psnr_sum = 0.0; total_ssim_sum = 0.0; total_lpips_sum = 0.0
        total_compared_frames_flat_count = 0
        wandb_val_samples_log_payload: Dict[str, List] = defaultdict(list) # type: ignore
        num_sequences_actually_logged_wandb = 0
        dtype_model = next(m_ref.parameters()).dtype

        for batch_idx, batch_frames_raw in enumerate(tqdm(self.val_loader, desc="Validating", dynamic_ncols=True, disable=os.getenv('CI')=='true' or not self.am_main_process)):
            batch_frames = batch_frames_raw.to(self.device)
            real_frames_full = batch_frames.to(self.device, dtype=dtype_model)
            B_val, N_total_val, C_val, H_val, W_val = real_frames_full.shape

            recon_frames_pred_sequence, _, _, _ = m_ref(real_frames_full)

            num_cond = self.video_config["num_input_frames"]; num_pred_config = self.video_config["num_predict_frames"]
            available_pred_len_from_g = recon_frames_pred_sequence.shape[1]
            available_gt_len = N_total_val - num_cond
            compare_len = min(available_pred_len_from_g, available_gt_len, num_pred_config)
            if compare_len <=0: continue

            pred_for_metrics = recon_frames_pred_sequence[:, :compare_len, ...]
            gt_for_metrics = real_frames_full[:, num_cond : num_cond + compare_len, ...]
            pred_norm_01 = (pred_for_metrics.clamp(-1, 1) + 1) / 2.0
            gt_norm_01 = (gt_for_metrics.clamp(-1, 1) + 1) / 2.0
            pred_norm_flat = pred_norm_01.reshape(-1, C_val, H_val, W_val)
            gt_norm_flat = gt_norm_01.reshape(-1, C_val, H_val, W_val)
            current_batch_num_flat_frames_processed = pred_norm_flat.shape[0]

            mse_loss_val_batch = F.mse_loss(pred_norm_flat, gt_norm_flat, reduction='mean')
            if torch.isfinite(mse_loss_val_batch):
                total_recon_loss_mse_sum += mse_loss_val_batch.item() * current_batch_num_flat_frames_processed
                psnr_val_batch_avg = 10 * math.log10(1.0 / (mse_loss_val_batch.item() + EPS)) if mse_loss_val_batch.item() > EPS else 100.0
                total_psnr_sum += psnr_val_batch_avg * current_batch_num_flat_frames_processed
            
            if self.ssim_metric:
                try: ssim_val_batch_avg = self.ssim_metric(pred_norm_flat, gt_norm_flat); total_ssim_sum += ssim_val_batch_avg.item() * current_batch_num_flat_frames_processed
                except Exception: pass # Ignore SSIM errors for a batch
            
            if self.lpips_loss_fn:
                try: lpips_val_batch_frames = self.lpips_loss_fn(pred_norm_flat*2.0-1.0, gt_norm_flat*2.0-1.0); total_lpips_sum += lpips_val_batch_frames.sum().item()
                except Exception: pass # Ignore LPIPS errors

            total_compared_frames_flat_count += current_batch_num_flat_frames_processed

            if self.args.wandb and WANDB_AVAILABLE and wandb.run and num_sequences_actually_logged_wandb < num_val_samples_to_log: # type: ignore
                num_to_log_this_val_batch = min(B_val, num_val_samples_to_log - num_sequences_actually_logged_wandb)
                for k_idx in range(num_to_log_this_val_batch):
                    # ... (WandB image logging logic for val - same as before) ...
                    csio = num_sequences_actually_logged_wandb + k_idx # current_sample_idx_overall
                    for fci in range(min(num_cond, N_total_val)): wandb_val_samples_log_payload["val_context_samples"].append(wandb.Image(((real_frames_full[k_idx, fci].cpu().float().clamp(-1,1)+1)/2.0), caption=f"ValCond_S{csio}_F{fci}_Ep{self.current_epoch+1}"))
                    for fpi in range(compare_len):
                        wandb_val_samples_log_payload["val_ground_truth_samples"].append(wandb.Image(gt_norm_01[k_idx, fpi].cpu().float(), caption=f"ValGT_S{csio}_F{fpi}_Ep{self.current_epoch+1}"))
                        wandb_val_samples_log_payload["val_reconstruction_samples"].append(wandb.Image(pred_norm_01[k_idx, fpi].cpu().float(), caption=f"ValRecon_S{csio}_F{fpi}_Ep{self.current_epoch+1}"))
                num_sequences_actually_logged_wandb += num_to_log_this_val_batch
        
        m_ref.train() # Set model back to train mode

        avg_val_recon_mse = total_recon_loss_mse_sum / total_compared_frames_flat_count if total_compared_frames_flat_count > 0 else float('inf')
        avg_val_psnr = total_psnr_sum / total_compared_frames_flat_count if total_compared_frames_flat_count > 0 else 0.0
        avg_val_ssim = total_ssim_sum / total_compared_frames_flat_count if total_compared_frames_flat_count > 0 and self.ssim_metric else 0.0
        avg_val_lpips = total_lpips_sum / total_compared_frames_flat_count if total_compared_frames_flat_count > 0 and self.lpips_loss_fn else float('inf')

        metrics = {"avg_val_recon_mse": avg_val_recon_mse, "avg_val_psnr": avg_val_psnr, "avg_val_ssim": avg_val_ssim, "avg_val_lpips": avg_val_lpips}
        self.last_val_metrics = metrics
        self.logger.info(f"Validation Metrics (Ep {self.current_epoch+1}, GStep {self.global_step}): ReconMSE:{avg_val_recon_mse:.4f}, PSNR:{avg_val_psnr:.2f}, SSIM:{avg_val_ssim:.4f}, LPIPS:{avg_val_lpips:.4f}")
        if wandb_val_samples_log_payload and self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(dict(wandb_val_samples_log_payload), step=self.global_step) # type: ignore
        return metrics

    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict]=None, is_best:bool=False):
        # ... (Save checkpoint method remains identical to your V2 logic) ...
        if not self.am_main_process: return
        m_save = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        d_save = self.discriminator.module if self.ddp_active and isinstance(self.discriminator, DDP) else self.discriminator
        data = {'global_step': self.global_step, 'epoch': self.current_epoch, 'model_state_dict': m_save.state_dict(),'discriminator_state_dict': d_save.state_dict(),'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict(),'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict() if self.args.use_amp else None,'scaler_disc_state_dict': self.scaler_disc.state_dict() if self.args.use_amp else None,'args': vars(self.args),'metrics': metrics if metrics else self.last_val_metrics,'video_config': self.video_config,'best_val_metric_val': self.best_val_metric_val,'current_lambda_kl': self.lambda_kl}
        q_ctrl_gen_obj = getattr(self.optimizer_enc_gen, 'q_controller', None); q_ctrl_disc_obj = getattr(self.optimizer_disc, 'q_controller', None)
        if q_ctrl_gen_obj and hasattr(q_ctrl_gen_obj, '__dict__'): data['q_controller_enc_gen_state'] = q_ctrl_gen_obj.__dict__
        if q_ctrl_disc_obj and hasattr(q_ctrl_disc_obj, '__dict__'): data['q_controller_disc_state'] = q_ctrl_disc_obj.__dict__
        fname_prefix="wubugaad_hybridgen_ckpt_v01"; fpath=""
        if is_best: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_best.pt")
        elif is_intermediate: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_step{self.global_step}.pt")
        else: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_ep{self.current_epoch+1}_step{self.global_step}.pt")
        try: torch.save(data, fpath); self.logger.info(f"Checkpoint saved: {os.path.basename(fpath)}")
        except Exception as e: self.logger.error(f"Save checkpoint error {fpath}: {e}", exc_info=True)


    def load_checkpoint(self, checkpoint_path:str) -> Tuple[int,int]:
        # ... (Load checkpoint method remains identical to your V2 logic) ...
        if not os.path.exists(checkpoint_path): self.logger.warning(f"CKPT {checkpoint_path} not found."); return 0,0
        try: ckpt = torch.load(checkpoint_path, map_location=self.device); self.logger.info(f"Loaded CKPT: {checkpoint_path}")
        except Exception as e: self.logger.error(f"Failed load CKPT {checkpoint_path}: {e}"); return 0,0
        m_load = self.model.module if self.ddp_active else self.model; d_load = self.discriminator.module if self.ddp_active else self.discriminator
        try: m_load.load_state_dict(ckpt['model_state_dict'], strict=self.args.load_strict); self.logger.info("Model state_dict loaded.")
        except Exception as e: self.logger.error(f"Err loading model state_dict: {e}", exc_info=False)
        try: d_load.load_state_dict(ckpt['discriminator_state_dict'], strict=self.args.load_strict); self.logger.info("Disc state_dict loaded.")
        except Exception as e: self.logger.error(f"Err loading disc state_dict: {e}", exc_info=False)
        if 'optimizer_enc_gen_state_dict' in ckpt and self.optimizer_enc_gen:
            try: self.optimizer_enc_gen.load_state_dict(ckpt['optimizer_enc_gen_state_dict']); self.logger.info("Opt G state loaded.")
            except Exception as e: self.logger.warning(f"Could not load Opt G state: {e}")
        if 'optimizer_disc_state_dict' in ckpt and self.optimizer_disc:
            try: self.optimizer_disc.load_state_dict(ckpt['optimizer_disc_state_dict']); self.logger.info("Opt D state loaded.")
            except Exception as e: self.logger.warning(f"Could not load Opt D state: {e}")
        if self.args.use_amp:
            if 'scaler_enc_gen_state_dict' in ckpt and self.scaler_enc_gen and ckpt['scaler_enc_gen_state_dict']: self.scaler_enc_gen.load_state_dict(ckpt['scaler_enc_gen_state_dict'])
            if 'scaler_disc_state_dict' in ckpt and self.scaler_disc and ckpt['scaler_disc_state_dict']: self.scaler_disc.load_state_dict(ckpt['scaler_disc_state_dict'])
        q_g = getattr(self.optimizer_enc_gen, 'q_controller', None); q_d = getattr(self.optimizer_disc, 'q_controller', None)
        if q_g and 'q_controller_enc_gen_state' in ckpt and ckpt['q_controller_enc_gen_state']:
            try: q_g.__dict__.update(ckpt['q_controller_enc_gen_state']); self.logger.info("Q-Ctrl G state loaded.")
            except Exception as e: self.logger.warning(f"Could not load Q-Ctrl G state: {e}")
        if q_d and 'q_controller_disc_state' in ckpt and ckpt['q_controller_disc_state']:
            try: q_d.__dict__.update(ckpt['q_controller_disc_state']); self.logger.info("Q-Ctrl D state loaded.")
            except Exception as e: self.logger.warning(f"Could not load Q-Ctrl D state: {e}")
        loaded_gs = ckpt.get('global_step', 0); loaded_ep = ckpt.get('epoch', 0)
        next_epoch_to_start = loaded_ep + 1 if loaded_gs > 0 and not self.args.load_strict else loaded_ep # If strict, assume it's start of loaded_ep
        default_best_val = -float('inf') if self.args.val_primary_metric in ["avg_val_psnr", "avg_val_ssim"] else float('inf')
        self.best_val_metric_val = ckpt.get('best_val_metric_val', default_best_val)
        self.lambda_kl = float(ckpt.get('current_lambda_kl', self.args.lambda_kl)) # Ensure float
        self.logger.info(f"Resuming. GlobalStep:{loaded_gs}, NextEpoch:{next_epoch_to_start}. BestVal({self.args.val_primary_metric}):{self.best_val_metric_val:.4f}. LambdaKL:{self.lambda_kl:.2e}")
        return loaded_gs, next_epoch_to_start

    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]: # Return Optional
        # ... (Sample method remains identical to your V2 logic) ...
        m_ref = self.model.module if self.ddp_active else self.model; m_ref.eval()
        device = self.device; dtype_model = next(m_ref.parameters()).dtype; latent_dim = self.args.latent_dim
        if noise is None: z = torch.randn(num_samples, latent_dim, device=device, dtype=dtype_model)
        else: z = noise.to(device=device, dtype=dtype_model)
        if z.shape[0] != num_samples: num_samples = z.shape[0]
        npf_cfg = self.video_config["num_predict_frames"]; nra_cfg = self.gaad_appearance_config["num_regions"]
        fd_cfg = (self.args.image_w, self.args.image_h); dbb_list = []
        if nra_cfg > 0:
            for _ in range(num_samples):
                cs_bbox_1f = golden_subdivide_rect_fixed_n(fd_cfg, nra_cfg, device=device, dtype=dtype_model, min_size_px=self.gaad_appearance_config.get('min_size_px', 5)) # type: ignore
                bss_all_f_rep = cs_bbox_1f.unsqueeze(0).repeat(npf_cfg, 1, 1)
                dbb_list.append(bss_all_f_rep)
            dbb_batch = torch.stack(dbb_list) if dbb_list else None
        else: dbb_batch = None
        self.logger.info(f"Sampling {num_samples} sequences...")
        if dbb_batch is not None or nra_cfg == 0: # Proceed if bboxes generated or not needed
            generated_frames = m_ref.decode(z, dbb_batch)
            self.logger.info("Sampling finished."); return generated_frames
        else:
            self.logger.warning("Sampling skipped: Bounding boxes required but not generated."); return None




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
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_dir", help="Path to video file or directory containing it.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DDP local rank.")
    parser.add_argument('--epochs', type=int, default=100, help="Total training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per GPU.")
    parser.add_argument('--image_h', type=int, default=64, help="Target image height.")
    parser.add_argument('--image_w', type=int, default=64, help="Target image width.")
    parser.add_argument('--num_channels', type=int, default=3, help="Number of image channels.")
    parser.add_argument('--num_input_frames', type=int, default=3, help="Number of context frames for VAE encoder.")
    parser.add_argument('--num_predict_frames', type=int, default=1, help="Number of frames the Generator predicts.")
    parser.add_argument('--frame_skip', type=int, default=1, help="Frame skip in dataset.")
    parser.add_argument('--seed',type=int, default=42, help="Random seed.")
    parser.add_argument('--num_workers',type=int, default=2, help="DataLoader workers.")
    parser.add_argument('--checkpoint_dir',type=str, default='wubugaad_hybridgen_checkpoints_v01', help="Directory for checkpoints.")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to checkpoint to load.")
    parser.add_argument('--load_strict', action='store_true', help="Use strict=True when loading state_dict.")
    parser.add_argument('--wandb',action='store_true', help="Enable WandB logging.")
    parser.add_argument('--wandb_project',type=str,default='WuBuGAADHybridGenV01', help="WandB project name.")
    parser.add_argument('--wandb_run_name',type=str,default=None, help="WandB run name (auto-generated if None).")
    parser.add_argument('--log_interval',type=int, default=50, help="Log interval (in global steps).")
    parser.add_argument('--save_interval',type=int, default=1000, help="Checkpoint save interval (in global steps).")

    # --- GAAD ---
    parser.add_argument('--gaad_num_regions', type=int, default=12, help="Number of regions for GAAD (appearance).")
    parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="GAAD type for appearance.")
    parser.add_argument('--gaad_min_size_px', type=int, default=4, help="Min region size for GAAD.")

    # --- Motion Branch ---
    parser.add_argument('--use_wubu_motion_branch', action='store_true', help="Enable GAAD+WuBu-M+OpticalFlow motion branch.")
    parser.add_argument('--gaad_motion_num_regions', type=int, default=8, help="GAAD regions for motion.")
    parser.add_argument('--gaad_motion_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="GAAD type for motion.")
    parser.add_argument('--optical_flow_net_type', type=str, default='raft_small', choices=list(FLOW_MODELS.keys()) if OPTICAL_FLOW_AVAILABLE else [], help="Optical flow network.")
    parser.add_argument('--freeze_flow_net', action='store_true', help="Freeze optical flow network weights.")
    parser.add_argument('--flow_stats_components', nargs='+', type=str, default=['mag_mean', 'angle_mean'], help="Flow stats (mag_mean, angle_mean, mag_std, angle_std).")

    # --- Encoder Architecture ---
    parser.add_argument('--latent_dim', type=int, default=256, help="VAE latent space dimensionality.")
    parser.add_argument('--encoder_use_roi_align', action='store_true', help="Use RoIAlign in encoder patch extractor.")
    parser.add_argument('--encoder_shallow_cnn_channels', type=int, default=32, help="Channels for shallow CNN if RoIAlign used.")
    parser.add_argument('--encoder_roi_align_output_h', type=int, default=4, help="RoIAlign output height.")
    parser.add_argument('--encoder_roi_align_output_w', type=int, default=4, help="RoIAlign output width.")
    parser.add_argument('--encoder_pixel_patch_size', type=int, default=16, help="Pixel patch size if not RoIAlign.")
    parser.add_argument('--encoder_initial_tangent_dim', type=int, default=128, help="Input tangent dim to WuBu-S.")

    # --- Generator Architecture ---
    parser.add_argument('--gen_temporal_kernel_size', type=int, default=3, help="Temporal kernel size in Generator 3D convs.")
    parser.add_argument('--gen_final_conv_kernel_spatial', type=int, default=3, help="Spatial kernel for final Generator conv layer.")

    # --- Discriminator Architecture ---
    parser.add_argument('--discriminator_type', type=str, default="spatio_temporal_cnn", choices=["spatio_temporal_cnn", "regional_cnn"], help="Discriminator architecture type.")
    parser.add_argument('--disc_apply_spectral_norm', action='store_true', help="Apply spectral normalization to D's conv/linear layers.")
    parser.add_argument('--disc_base_disc_channels', type=int, default=64, help="Base channels for D's 3D CNN.")
    parser.add_argument('--disc_max_disc_channels', type=int, default=512, help="Max channels for D's 3D CNN.")
    parser.add_argument('--disc_temporal_kernel_size', type=int, default=3, help="Temporal kernel for D's 3D CNN.")
    parser.add_argument('--disc_min_hidden_fc_dim', type=int, default=128, help="Min hidden dim for D's final FC layer if using one.")
    parser.add_argument('--disc_max_hidden_fc_dim', type=int, default=512, help="Max hidden dim for D's final FC layer if using one.")
    parser.add_argument('--disc_use_gaad_film_condition', action='store_true', help="Enable GAAD-based FiLM conditioning in Discriminator.")
    parser.add_argument('--disc_gaad_condition_dim_disc', type=int, default=64, help="Condition dim for D's FiLM if enabled.")
    parser.add_argument('--disc_patch_size', type=int, default=16, help="Patch size for 'regional_cnn' D type.")
    parser.add_argument('--disc_cnn_channels_2d', nargs='+', type=int, default=[64, 128, 256], help="2D CNN channels for 'regional_cnn' D type.")

    # --- WuBu Stacks ---
    parser.add_argument('--wubu_dropout', type=float, default=0.1, help="Dropout for WuBu layers.")
    parser.add_argument('--wubu_s_num_levels', type=int, default=2); parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32]); parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8]); parser.add_argument('--wubu_s_use_rotation', action='store_true'); parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_m_num_levels', type=int, default=1); parser.add_argument('--wubu_m_hyperbolic_dims', nargs='+', type=int, default=[32]); parser.add_argument('--wubu_m_initial_curvatures', nargs='+', type=float, default=[0.7]); parser.add_argument('--wubu_m_use_rotation', action='store_true'); parser.add_argument('--wubu_m_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_m_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_t_num_levels', type=int, default=1); parser.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[128]); parser.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[0.5]); parser.add_argument('--wubu_t_use_rotation', action='store_true'); parser.add_argument('--wubu_t_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_s_output_dim', type=int, default=32, help="Output dim of WuBu-S (if no levels, then encoder_initial_tangent_dim).");
    parser.add_argument('--wubu_m_output_dim', type=int, default=32, help="Output dim of WuBu-M.");

    # --- Training ---
    parser.add_argument('--lambda_recon', type=float, default=10.0, help="Weight for VAE reconstruction loss (MSE).")
    parser.add_argument('--lambda_kl', type=float, default=0.1, help="Weight for VAE KL divergence loss.")
    parser.add_argument('--lambda_gan', type=float, default=1.0, help="Weight for GAN adversarial loss (Generator part).")
    parser.add_argument('--learning_rate_gen',type=float,default=1e-4, help="Learning rate for Generator/Encoder.");
    parser.add_argument('--learning_rate_disc',type=float,default=1e-4, help="Learning rate for Discriminator.");
    parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0, help="Max grad norm for Riemannian SGD parameter updates.");
    parser.add_argument('--global_max_grad_norm',type=float,default=5.0, help="Global gradient clipping norm for optimizers.");
    parser.add_argument('--q_controller_enabled',action='store_true', help="Enable HAKMEM Q-Controller for optimizers.");
    parser.add_argument('--grad_accum_steps',type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument('--use_amp', action='store_true', help="Use Automatic Mixed Precision (AMP).");
    parser.add_argument('--detect_anomaly',action='store_true', help="Enable autograd.detect_anomaly (for debugging).")
    parser.add_argument('--log_grad_norm', action='store_true', help="Log optimizer gradient norms (can slow training).")

    # --- Validation & Sampling ---
    parser.add_argument('--wandb_log_train_recon_interval', type=int, default=0, help="Log training batch reconstructions to WandB every N global steps (0 to disable).")
    parser.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=0, help="Log samples from fixed noise to WandB every N global steps (0 to disable).")
    parser.add_argument('--use_lpips_for_verification', action='store_true', help="Use LPIPS metric for validation output.");
    parser.add_argument('--validation_video_path', type=str, default=None, help="Optional separate video for validation.");
    parser.add_argument('--validation_split_fraction', type=float, default=0.1, help="Fraction of main data for validation if no separate val video.");
    parser.add_argument('--val_block_size', type=int, default=20, help="Block size for validation split if using block sampling.");
    parser.add_argument('--val_primary_metric', type=str, default="avg_val_psnr", choices=["avg_val_recon_mse", "avg_val_psnr", "avg_val_ssim", "avg_val_lpips"], help="Primary metric for saving best checkpoint.");
    parser.add_argument('--num_val_samples_to_log', type=int, default=2, help="Number of validation video samples to log to WandB.");
    parser.add_argument('--demo_num_samples', type=int, default=4, help="Number of samples for final demo generation.")

    parsed_args = parser.parse_args()

    if parsed_args.use_wubu_motion_branch and not OPTICAL_FLOW_AVAILABLE: parser.error("Motion branch needs optical flow, but torchvision.models.optical_flow unavailable.")
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.optical_flow_net_type not in FLOW_MODELS: parser.error(f"Optical flow net type '{parsed_args.optical_flow_net_type}' not in available: {list(FLOW_MODELS.keys())}")
    
    def validate_wubu_config_for_argparse(args_obj, prefix_str, parser_ref, is_motion_branch_active):
        num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0); is_motion = prefix_str == "wubu_m"
        if num_levels > 0 and (not is_motion or is_motion_branch_active):
            for suffix, attr_name in [("hyperbolic_dims", f"{prefix_str}_hyperbolic_dims"), ("initial_curvatures", f"{prefix_str}_initial_curvatures")]:
                val_list = getattr(args_obj, attr_name); is_list=isinstance(val_list, list); val_list = val_list if is_list else [val_list]
                if len(val_list) != num_levels:
                    if len(val_list) == 1 and num_levels > 1: setattr(args_obj, attr_name, [val_list[0]] * num_levels)
                    elif not val_list and suffix=="initial_curvatures": setattr(args_obj, attr_name, [1.0] * num_levels)
                    elif not val_list and suffix=="hyperbolic_dims": setattr(args_obj, attr_name, [getattr(args_obj, 'latent_dim', 32)//num_levels if num_levels>0 else []]*num_levels)
                    else: parser_ref.error(f"{prefix_str}: Length mismatch {attr_name} ({len(val_list)}) vs num_levels ({num_levels})")
    validate_wubu_config_for_argparse(parsed_args, "wubu_s", parser, True); validate_wubu_config_for_argparse(parsed_args, "wubu_t", parser, True); validate_wubu_config_for_argparse(parsed_args, "wubu_m", parser, parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE)

    if parsed_args.wubu_s_num_levels > 0 and parsed_args.wubu_s_hyperbolic_dims: parsed_args.wubu_s_output_dim = parsed_args.wubu_s_hyperbolic_dims[-1]
    else: parsed_args.wubu_s_output_dim = parsed_args.encoder_initial_tangent_dim; parsed_args.wubu_s_num_levels = 0
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.wubu_m_num_levels > 0 and parsed_args.wubu_m_hyperbolic_dims: parsed_args.wubu_m_output_dim = parsed_args.wubu_m_hyperbolic_dims[-1]
    else: parsed_args.wubu_m_output_dim = 0; parsed_args.wubu_m_num_levels = 0

    valid_stats = {'mag_mean', 'angle_mean', 'mag_std', 'angle_std'};
    if any(s not in valid_stats for s in parsed_args.flow_stats_components): parser.error(f"Invalid flow_stats_components. Allowed: {valid_stats}. Got: {parsed_args.flow_stats_components}")

    return parsed_args

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
        if device.type == 'cuda':
            torch.cuda.set_device(device)

    am_main_process = (rank == 0)
    
    # Setup root logger AND specific logger
    # This ensures messages from deeper modules also get formatted if they use getLogger
    base_logger_name = "WuBuGAADHybridGenV01"
    root_logger = logging.getLogger() # Get root logger
    # Remove all existing handlers from root to avoid conflicts or duplicate messages
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Remove handlers from our specific logger if any were attached before basicConfig
    specific_logger = logging.getLogger(base_logger_name)
    for handler in specific_logger.handlers[:]:
        specific_logger.removeHandler(handler)

    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s',
        force=True # Override any existing root logger configuration
    )
    
    current_logger_main = logging.getLogger(f"{base_logger_name}.Main") # More specific logger for main
    current_logger_main.info(f"--- {base_logger_name} (R{rank}/{world_size}, Dev {device}, DDP:{ddp_active}, AMP:{args.use_amp}) ---")
    
    seed_everything(args.seed, rank, world_size)
    
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        current_logger_main.warning("Autograd anomaly detection ENABLED.")
    
    if am_main_process:
        current_logger_main.info(f"Effective Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubuhybrid_v0.1_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                resume="allow", # Allow resuming if run_id matches
                id=wandb.util.generate_id() if wandb.run is None else wandb.run.id # Handles new vs resume
            )
            current_logger_main.info(f"WandB initialized for run: {run_name}, Project: {args.wandb_project}")
        except Exception as e_wandb:
            current_logger_main.error(f"WandB initialization failed: {e_wandb}", exc_info=True)
            args.wandb = False # Disable wandb if init fails


    video_config = {
        "image_size": (args.image_h, args.image_w),
        "num_channels": args.num_channels,
        "num_input_frames": args.num_input_frames,
        "num_predict_frames": args.num_predict_frames,
        "wubu_s_output_dim": args.wubu_s_output_dim,
        "wubu_m_output_dim": args.wubu_m_output_dim,
    }
    gaad_appearance_config = {
        "num_regions": args.gaad_num_regions,
        "decomposition_type": args.gaad_decomposition_type,
        "min_size_px": args.gaad_min_size_px
    }
    gaad_motion_config = {
        "num_regions": args.gaad_motion_num_regions,
        "decomposition_type": args.gaad_motion_decomposition_type,
        "min_size_px": args.gaad_min_size_px,
    } if args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE else None

    wubu_s_config = _configure_wubu_stack(args, "wubu_s")
    wubu_t_config = _configure_wubu_stack(args, "wubu_t")
    wubu_m_config = _configure_wubu_stack(args, "wubu_m")

    discriminator_config = {
        "type": args.discriminator_type,
        "apply_spectral_norm": args.disc_apply_spectral_norm,
        "use_gaad_film_condition": args.disc_use_gaad_film_condition,
        "gaad_condition_dim_disc": args.disc_gaad_condition_dim_disc,
        "base_disc_channels": args.disc_base_disc_channels,
        "max_disc_channels": args.disc_max_disc_channels,
        "temporal_kernel_size": args.disc_temporal_kernel_size,
        "patch_size": args.disc_patch_size,
        "cnn_channels_2d": args.disc_cnn_channels_2d,
    }

    if am_main_process:
        current_logger_main.info(f"VideoCfg:{video_config}\nGAADAppCfg:{gaad_appearance_config}\n"
                                 f"GAADMotCfg:{gaad_motion_config}\nWuBuS:{wubu_s_config}\n"
                                 f"WuBuT:{wubu_t_config}\nWuBuM:{wubu_m_config}\nDiscCfg:{discriminator_config}")

    model = WuBuGAADHybridGenNet(args, video_config, gaad_appearance_config, gaad_motion_config, wubu_s_config, wubu_t_config, wubu_m_config).to(device)
    discriminator = RegionalDiscriminator(args, video_config, gaad_appearance_config, discriminator_config).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)
        wandb.watch(discriminator, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)

    if ddp_active:
        # find_unused_parameters=True for model because WuBu paths might be complex
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # find_unused_parameters=False for discriminator if confident all paths are used
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Prepare Q-Controller configs
    q_cfg_gen = None
    q_cfg_disc = None
    if args.q_controller_enabled:
        # Assuming DEFAULT_CONFIG_QLEARN_HYBRID is updated with correct keys
        # like "q_learning_rate", "discount_factor", etc.
        q_cfg_gen = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
        q_cfg_disc = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
        # Potentially customize q_cfg_disc if needed, e.g., different exploration for D
        # q_cfg_disc["epsilon_start"] = 0.6 # Example
        current_logger_main.info(f"Q-Controller config for G: {q_cfg_gen}")
        current_logger_main.info(f"Q-Controller config for D: {q_cfg_disc}")


    optimizer_enc_gen = RiemannianEnhancedSGD(
        model.parameters(),
        lr=args.learning_rate_gen,
        q_learning_config=q_cfg_gen, # Pass specific config
        max_grad_norm_risgd=args.risgd_max_grad_norm,
        optimizer_type="generator"  # CRITICAL: Specify optimizer type
    )
    optimizer_disc = RiemannianEnhancedSGD(
        discriminator.parameters(),
        lr=args.learning_rate_disc,
        q_learning_config=q_cfg_disc, # Pass specific config
        max_grad_norm_risgd=args.risgd_max_grad_norm,
        optimizer_type="discriminator" # CRITICAL: Specify optimizer type
    )

    # --- Dataset and DataLoader Setup ---
    actual_video_path = args.video_data_path
    demo_file_name = "dummy_video_hybridgen_v01.mp4" # Centralize filename
    if "demo_video_data" in args.video_data_path: # Check if using demo path
        actual_video_path = os.path.join(args.video_data_path, demo_file_name)
        if am_main_process: # Only main process creates dummy video
            os.makedirs(args.video_data_path, exist_ok=True)
            if not os.path.exists(actual_video_path):
                if IMAGEIO_AVAILABLE and imageio is not None:
                    current_logger_main.info(f"Creating dummy video using imageio: {actual_video_path}...")
                    min_raw_frames_needed = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                    num_dummy_frames = max(100, min_raw_frames_needed + 50) # Ensure enough frames
                    dummy_h, dummy_w = int(args.image_h), int(args.image_w)
                    try:
                        # Create frames with some variation to aid learning if it were real data
                        video_writer = imageio.get_writer(actual_video_path, fps=15, quality=8, macro_block_size=16)
                        for i in range(num_dummy_frames):
                            frame = np.zeros((dummy_h, dummy_w, args.num_channels), dtype=np.uint8)
                            # Add a moving square for some basic temporal structure
                            sq_size = dummy_h // 4
                            x_pos = (i * 5) % (dummy_w - sq_size)
                            y_pos = (i * 3) % (dummy_h - sq_size)
                            frame[y_pos:y_pos+sq_size, x_pos:x_pos+sq_size, :] = [ (i*10)%255, (i*5)%255, (i*2)%255 ]
                            video_writer.append_data(frame)
                        video_writer.close()
                        current_logger_main.info(f"Dummy video created with {num_dummy_frames} frames.")
                    except Exception as e_imageio_write:
                        current_logger_main.error(f"Error creating dummy video: {e_imageio_write}", exc_info=True)
                else:
                    current_logger_main.error("imageio not available. Cannot create dummy video.")
    
    if ddp_active:
        torch.distributed.barrier() # Ensure dummy video is created before other ranks proceed

    if not os.path.isfile(actual_video_path):
        current_logger_main.error(f"Video path '{actual_video_path}' not found or is not a file. Exiting.")
        sys.exit(1)

    total_frames_per_sample = args.num_input_frames + args.num_predict_frames
    try:
        full_dataset = VideoFrameDataset(
            video_path=actual_video_path,
            num_frames_total=total_frames_per_sample,
            image_size=(args.image_h, args.image_w),
            frame_skip=args.frame_skip
        )
    except Exception as e:
        current_logger_main.error(f"Failed to initialize main Dataset from '{actual_video_path}': {e}", exc_info=True)
        sys.exit(1)

    if not full_dataset or len(full_dataset) == 0:
        current_logger_main.error("Main dataset is empty. Exiting.")
        sys.exit(1)

    train_dataset, val_dataset = full_dataset, None
    num_total_samples = len(full_dataset)

    if args.validation_video_path and os.path.isfile(args.validation_video_path):
        try:
            val_dataset_candidate = VideoFrameDataset(video_path=args.validation_video_path, num_frames_total=total_frames_per_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
            if len(val_dataset_candidate) > 0:
                val_dataset = val_dataset_candidate
                current_logger_main.info(f"Using separate validation video: {args.validation_video_path}, Samples: {len(val_dataset)}")
            else:
                current_logger_main.warning(f"Validation video {args.validation_video_path} loaded but is empty. Using split from main data.")
        except Exception as e:
            current_logger_main.warning(f"Could not load validation dataset from '{args.validation_video_path}': {e}. Attempting split from main data.")

    if val_dataset is None and args.validation_split_fraction > 0.0 and num_total_samples > 10: # Min samples to split
        val_block_size = args.val_block_size
        if val_block_size <= 0 or num_total_samples < val_block_size * 2: # Ensure enough for train and val blocks
            # Simple random split if block sampling isn't feasible
            num_val = int(num_total_samples * args.validation_split_fraction)
            num_train = num_total_samples - num_val
            if num_train > 0 and num_val > 0:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    full_dataset, [num_train, num_val],
                    generator=torch.Generator().manual_seed(args.seed + rank) # Ensure consistent split across ranks if DDP
                )
                current_logger_main.info(f"Split main dataset (random): Train={len(train_dataset)}, Val={len(val_dataset)}")
            else:
                current_logger_main.warning("Random split resulted in 0 samples for train or val. No validation set used.")
                val_dataset = None # Ensure it's None
        else: # Block sampling
            target_val_samples = int(num_total_samples * args.validation_split_fraction)
            num_val_blocks = max(1, target_val_samples // val_block_size)
            max_possible_block_starts = num_total_samples - val_block_size + 1
            
            if num_val_blocks >= max_possible_block_starts: # Not enough distinct blocks, take first ones
                 block_start_indices = list(range(max_possible_block_starts))[:num_val_blocks]
            else:
                 # Ensure same blocks chosen across ranks for DDP consistency by seeding Random
                 rng_val_split = random.Random(args.seed + 7) # Different seed for this specific choice
                 block_start_indices = sorted(rng_val_split.sample(range(max_possible_block_starts), num_val_blocks))

            val_indices_set = set()
            for start_idx in block_start_indices:
                val_indices_set.update(range(start_idx, min(start_idx + val_block_size, num_total_samples)))
            
            all_indices = list(range(num_total_samples))
            train_indices = sorted(list(set(all_indices) - val_indices_set))
            val_indices = sorted(list(val_indices_set))

            if len(train_indices) > 0 and len(val_indices) > 0:
                train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
                val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
                current_logger_main.info(f"Split main dataset (block): Train={len(train_dataset)}, Val={len(val_indices)}")
            else:
                current_logger_main.warning("Block sampling resulted in 0 samples for train or val. No validation set used.")
                val_dataset = None

    if am_main_process:
        current_logger_main.info(f"Final dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}")

    # Common worker init function for reproducibility
    worker_init_fn_seeded = functools.partial(seed_worker_init_fn, base_seed=args.seed, rank=rank, world_size=world_size) if args.num_workers > 0 else None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_active else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, sampler=train_sampler,
        pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded,
        drop_last=True # Important for consistent batch sizes, DDP
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, sampler=val_sampler,
            pin_memory=(device.type == 'cuda'), drop_last=False, # Don't drop last for validation
            worker_init_fn=worker_init_fn_seeded
        )

    trainer = HybridTrainer(model, discriminator, optimizer_enc_gen, optimizer_disc, device, train_loader, val_loader, args, rank, world_size, ddp_active)
    
    start_global_step, start_epoch = 0, 0
    if args.load_checkpoint:
        start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint)
        # load_checkpoint now returns next epoch to start from

    try:
        trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt:
        current_logger_main.info(f"Rank {rank}: Training interrupted by user.")
    except Exception as e:
        current_logger_main.error(f"Rank {rank}: Training loop crashed: {e}", exc_info=True)
    finally:
        if am_main_process:
            current_logger_main.info("Finalizing run and saving final checkpoint...")
            final_metrics_to_save = trainer.last_val_metrics.copy() if trainer.last_val_metrics else {}
            final_metrics_to_save['best_val_metric_val_at_end'] = trainer.best_val_metric_val
            trainer._save_checkpoint(metrics=final_metrics_to_save) # Save last state
            
            if args.epochs > 0 and hasattr(trainer, 'sample') and trainer.global_step > 0:
                current_logger_main.info("Generating final demo samples...")
                try:
                    # Ensure model is on correct device for sampling if DDP was used
                    sampling_model = model.module if ddp_active else model
                    trainer_for_sampling = HybridTrainer(sampling_model, discriminator.module if ddp_active else discriminator, optimizer_enc_gen, optimizer_disc, device, train_loader, val_loader, args, rank, world_size, ddp_active=False) # Temp trainer for sampling on main model
                    trainer_for_sampling.global_step = trainer.global_step # copy necessary attributes
                    trainer_for_sampling.current_epoch = trainer.current_epoch
                    trainer_for_sampling.video_config = trainer.video_config
                    trainer_for_sampling.gaad_appearance_config = trainer.gaad_appearance_config


                    pred_pixels = trainer_for_sampling.sample(num_samples=args.demo_num_samples)
                    if pred_pixels is not None and pred_pixels.numel() > 0 and pred_pixels.shape[0] > 0:
                        save_dir = os.path.join(args.checkpoint_dir, "demo_samples_hybrid_v01")
                        os.makedirs(save_dir, exist_ok=True)
                        num_frames_to_save_per_sample = min(pred_pixels.shape[1], 3) # Save up to 3 frames per sample

                        for b_idx in range(min(args.demo_num_samples, pred_pixels.shape[0])):
                            for frame_s_idx in range(num_frames_to_save_per_sample):
                                img_tensor = (pred_pixels[b_idx, frame_s_idx].cpu().clamp(-1, 1) + 1) / 2.0
                                save_image(img_tensor, os.path.join(save_dir, f"demo_sample_{b_idx}_frame_{frame_s_idx}_ep{trainer.current_epoch+1}.png"))
                        current_logger_main.info(f"Saved demo sample frames to {save_dir}")
                        
                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            wb_imgs_final_demo = []
                            for b_idx in range(min(args.demo_num_samples, pred_pixels.shape[0])):
                                for frame_s_idx in range(num_frames_to_save_per_sample):
                                     wb_imgs_final_demo.append(wandb.Image((pred_pixels[b_idx, frame_s_idx].cpu().float().clamp(-1,1)+1)/2.0, caption=f"FinalSample {b_idx} Frame {frame_s_idx} Ep{trainer.current_epoch+1}"))
                            if wb_imgs_final_demo:
                                wandb.log({"demo_samples_final": wb_imgs_final_demo}, step=trainer.global_step)
                except Exception as e_demo:
                    current_logger_main.error(f"Demo sampling error: {e_demo}", exc_info=True)
            
            if args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish()
        
        if ddp_active and is_initialized():
            destroy_process_group()
        current_logger_main.info(f"Rank {rank}: {base_logger_name} (v0.1) script finished.")



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