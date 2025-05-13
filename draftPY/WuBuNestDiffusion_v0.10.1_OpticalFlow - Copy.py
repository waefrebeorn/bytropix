# WuBuNestDiffusion_v0.10.1_OpticalFlow.py
# Diffusion Model with GAAD-WuBu Regional Hyperbolic Latent Space Autoencoder
# Incorporates Optical Flow for Motion Encoding Branch.
# Operating directly on GAAD-defined regions with WuBu nesting.
# LAST UPDATE: Integrated Optical Flow for Motion Encoding (v0.10.1 internal rev from v0.10.0)

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
logger = logging.getLogger("WuBuGAADOpticalFlowDiffV0101") # Renamed logger
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
DEFAULT_CONFIG_QLEARN_DIFFUSION = { "learning_rate": 0.01, "discount": 0.95, "epsilon": 0.2, "epsilon_decay": 0.9998, "min_epsilon": 0.01, "lr_scale_options": [0.9,0.95,1.,1.05,1.1], "momentum_scale_options": [0.95,0.98,1.,1.01,1.02], "max_q_table_size": 10000}

DEFAULT_CONFIG_TRANSFORMER_NOISE_PREDICTOR = {
    "num_layers": 4,
    "num_heads": 8,
    "d_model": 256,
    "d_ff_ratio": 4.0,
    "dropout": 0.1,
    "activation": "gelu",
}

# =====================================================================
# Geometric, Optimizer, WuBu Core Components
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
        super().__init__(c_scalar); c_scalar=float(max(c_scalar,0.0)) # Ensure c is non-negative
        if c_scalar <= 0: self.c=0.0; self.k = 0.; self.sqrt_c = 0.; self.radius = float('inf') # Explicitly set c to 0 if input is invalid
        else: self.c=c_scalar; self.k = -self.c; self.sqrt_c = math.sqrt(self.c); self.radius = 1. / self.sqrt_c
        self.max_norm = self.radius * (1. - EPS * 10) if self.c > 0 and self.radius != float('inf') else float('inf'); self._name = f'PoincareBall(c={self.c:.3g})'
    @property
    def name(self) -> str: return self._name
    def proju(self, x: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.poincare_clip(x, self.c, radius=1., eps=EPS * 10)
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.exponential_map(dp, self.c, eps=EPS)
    def logmap0(self, p: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.logarithmic_map(p, self.c, eps=EPS)
    def expmap0_scaled(self, dp: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_exponential_map(dp, self.c, scale_scalar=scale_scalar, eps=EPS)
    def logmap0_scaled(self, p: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_logarithmic_map(p, self.c, scale_scalar=scale_scalar, eps=EPS)
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        if self.c <= 0: return dp; p_proj = self.proju(p); p_norm_sq = torch.sum(p_proj.pow(2), dim=-1, keepdim=True)
        max_sq_norm = (1. / (self.c + EPS) - EPS * 100) if self.c > EPS else float('inf') # Max norm based on radius for stability
        p_norm_sq = torch.clamp(p_norm_sq, max=max_sq_norm)
        lambda_p_sq_val = (1. - self.c * p_norm_sq); factor = (lambda_p_sq_val / 2.).pow(2); return torch.clamp(factor, min=EPS) * dp
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad(): w.data.uniform_(-irange, irange); w.data = self.expmap0(w.data); w.data = self.proju(w.data)

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
        super().__init__(); self.in_dim, self.out_dim, self.transform_type = in_dim, out_dim, transform_type.lower(); self.use_rotation = use_rotation; self.rotation_module = None; self.phi_influence_rotation_init = phi_influence_rotation_init; current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.HILT") # Updated logger
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
        super().__init__(); self.level_idx, self.dim, self.config = level_idx, dim, config; current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Level") # Updated logger
        self.phi_influence_curvature = config.get("phi_influence_curvature", False); self.initial_curvature_val = initial_curvature_val_base * (PHI**(level_idx % 4 - 1.5) if self.phi_influence_curvature else 1.0); current_logger.info(f"L{level_idx}: InitialC={self.initial_curvature_val:.2f}"+(f" (PhiBase {initial_curvature_val_base:.2f})" if self.phi_influence_curvature else "")); self.use_ld = config.get("use_level_descriptors", True); self.use_spread = config.get("use_level_spread", True); self.dropout_rate = config.get("dropout", 0.1); self.ld_init_scale = config.get("level_descriptor_init_scale", 1e-5); self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean"); self.min_curvature = max(EPS, config.get("curvature_min_value", EPS)); self.min_scale = max(EPS, config.get("scale_min_value", EPS)); self.min_spread = max(EPS, config.get("spread_min_value", EPS))
        def _init_unconstrained_param_sigmoid_scaled(target_val, min_val_range, max_val_range):
            if not (min_val_range < max_val_range): current_logger.warning(f"L{level_idx} SigmoidInit: Invalid range [{min_val_range}, {max_val_range}]. Init unconstrained to 0."); return torch.tensor(0.0)
            clamped_target_val=torch.clamp(torch.as_tensor(target_val,dtype=torch.float),min_val_range+EPS,max_val_range-EPS).item();initial_sigmoid_target=(clamped_target_val-min_val_range)/(max_val_range-min_val_range);initial_sigmoid_target_clamped=max(EPS,min(initial_sigmoid_target,1.0-EPS));unconstrained_val=math.log(initial_sigmoid_target_clamped/(1.0-initial_sigmoid_target_clamped)); return torch.tensor(unconstrained_val)
        def _init_unconstrained_param_softplus(target_val, min_val): val_for_softplus=max(float(target_val),min_val+EPS)-min_val; return torch.tensor(math.log(math.expm1(val_for_softplus)) if val_for_softplus > 1e-6 else math.log(val_for_softplus + EPS))
        param_init_args = {'learn_c': ("learnable_curvature", self.initial_curvature_val, self.min_curvature, 'log_curvature_unconstrained', 'softplus'), 'learn_s': ("learnable_scales", "initial_scales", (MIN_WUBU_LEVEL_SCALE, MAX_WUBU_LEVEL_SCALE), 'log_scale_unconstrained', 'sigmoid_scaled'), 'learn_spread': ("learnable_spread", "initial_spread_values", self.min_spread, 'log_spread_unconstrained', 'softplus')}
        for key, (learn_flag_name, init_val_name_or_direct, min_or_range_val_local, param_name, init_type) in param_init_args.items():
            if key == 'learn_spread' and not self.use_spread: self.register_parameter(param_name, None); continue
            learn_flag = config.get(learn_flag_name, True); default_list_val = [1.0] if key == 'learn_s' else [0.1] if key == 'learn_spread' else [self.initial_curvature_val] # Use self.initial_curvature_val for 'c'
            if isinstance(init_val_name_or_direct, str): init_list = config.get(init_val_name_or_direct, default_list_val); init_val = init_list[level_idx] if level_idx < len(init_list) else (init_list[-1] if init_list else default_list_val[0])
            else: init_val = init_val_name_or_direct
            if init_type == 'softplus': unconstrained_val = _init_unconstrained_param_softplus(init_val, min_or_range_val_local)
            elif init_type == 'sigmoid_scaled': min_r, max_r = min_or_range_val_local; unconstrained_val = _init_unconstrained_param_sigmoid_scaled(init_val, min_r, max_r)
            else: raise ValueError(f"Unknown init_type: {init_type}")
            if learn_flag: setattr(self, param_name, nn.Parameter(unconstrained_val))
            else: self.register_buffer(param_name, unconstrained_val)
        if self.use_ld and self.dim > 0: self.level_descriptor_param = nn.Parameter(torch.Tensor(dim)); PoincareBall(c_scalar=self.initial_curvature_val).init_weights(self.level_descriptor_param, irange=self.ld_init_scale); setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=self.initial_curvature_val))
        else: self.register_parameter('level_descriptor_param', None)
        num_bounds_list = config.get("boundary_points_per_level", [8]); num_boundaries_val = num_bounds_list[level_idx] if level_idx < len(num_bounds_list) else (num_bounds_list[-1] if num_bounds_list else 8); self.boundary_manifold_module = BoundaryManifoldHyperbolic(level_idx, num_boundaries_val, dim, initial_manifold_c=self.initial_curvature_val) if self.dim > 0 else None
        comb_in_dim = self.dim + (self.dim if self.relative_vector_aggregation not in ['none', None] else 0) + (self.dim if self.use_ld else 0) + (1 if self.use_spread else 0); comb_h_dims_cfg = config.get("tangent_input_combination_dims", [max(16, comb_in_dim // 2)]) if comb_in_dim > 0 else []; comb_h_dims = comb_h_dims_cfg if isinstance(comb_h_dims_cfg, list) else [comb_h_dims_cfg]
        layers = []; in_d = comb_in_dim
        if self.dim > 0 and comb_in_dim > 0:
            for h_d in comb_h_dims:
                if in_d > 0 and h_d > 0: layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(self.dropout_rate)]); in_d = h_d
            if in_d > 0 and self.dim > 0: layers.append(nn.Linear(in_d, self.dim)); layers.append(nn.LayerNorm(self.dim))
        self.tangent_combiner = nn.Sequential(*layers) if layers else nn.Identity()
        self.use_flow = config.get("use_tangent_flow", True); self.tangent_flow_module = None; self.flow_scale_val = 0.0
        if self.use_flow and self.dim > 0:
            flow_h_dim = max(16, int(dim * config.get("tangent_flow_hidden_dim_ratio", 0.5))); flow_type = config.get("tangent_flow_type", "mlp").lower()
            if flow_type == 'mlp' and flow_h_dim > 0: self.tangent_flow_module = nn.Sequential(nn.Linear(dim, flow_h_dim), nn.GELU(), nn.Dropout(self.dropout_rate), nn.Linear(flow_h_dim, dim))
            elif flow_type == 'linear': self.tangent_flow_module = nn.Linear(dim, dim)
            if self.tangent_flow_module is not None: self.flow_scale_val = config.get("tangent_flow_scale", 1.0); self.tangent_flow_module.apply(init_weights_general)
        self.tangent_combiner.apply(init_weights_general)
    def get_current_curvature_scalar(self) -> float: return get_constrained_param_val(self.log_curvature_unconstrained, self.min_curvature).item()
    def get_current_scale_scalar(self) -> float:
        if hasattr(self,'log_scale_unconstrained') and self.log_scale_unconstrained is not None: scaled_sigmoid=torch.sigmoid(self.log_scale_unconstrained); val_tensor=MIN_WUBU_LEVEL_SCALE+(MAX_WUBU_LEVEL_SCALE-MIN_WUBU_LEVEL_SCALE)*scaled_sigmoid; return val_tensor.item()
        return MIN_WUBU_LEVEL_SCALE
    def get_current_spread_scalar_tensor(self) -> torch.Tensor:
        if self.use_spread and hasattr(self, 'log_spread_unconstrained') and self.log_spread_unconstrained is not None: return get_constrained_param_val(self.log_spread_unconstrained, self.min_spread)
        ref_param = next(iter(self.parameters()), self.tangent_combiner.parameters().__next__() if isinstance(self.tangent_combiner, nn.Sequential) and list(self.tangent_combiner.parameters()) else None) # Robust way to get a ref param
        ref_device = ref_param.device if ref_param is not None else torch.device('cpu'); ref_dtype = ref_param.dtype if ref_param is not None else torch.float; return torch.tensor(self.min_spread, device=ref_device, dtype=ref_dtype)
    def forward(self, point_in_hyperbolic: torch.Tensor, relative_vectors_tangent_in: Optional[torch.Tensor], descriptor_point_in_hyperbolic: Optional[torch.Tensor], sigma_in_scalar_tensor: Optional[torch.Tensor] ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if point_in_hyperbolic.dim() != 2: raise ValueError(f"WuBuLevel forward expects 2D input (B', D), got {point_in_hyperbolic.dim()}D shape {point_in_hyperbolic.shape}")
        B_prime, D_in = point_in_hyperbolic.shape; dev = point_in_hyperbolic.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else point_in_hyperbolic.dtype;
        if self.dim == 0: dummy_out_shape = (B_prime, 0); dummy_dtype_dev = {'device':dev, 'dtype':dtype_to_use}; current_spread_tensor = self.get_current_spread_scalar_tensor().to(dtype_to_use); return (torch.zeros(dummy_out_shape, **dummy_dtype_dev), torch.zeros(dummy_out_shape, **dummy_dtype_dev), None, None, current_spread_tensor)
        current_c_val = self.get_current_curvature_scalar(); current_s_val = self.get_current_scale_scalar(); current_sigma_out_tensor = self.get_current_spread_scalar_tensor(); current_manifold_obj = PoincareBall(c_scalar=current_c_val)
        if self.level_descriptor_param is not None and hasattr(self.level_descriptor_param, 'manifold'): setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=current_c_val))
        if self.boundary_manifold_module is not None: self.boundary_manifold_module.set_current_manifold_c(current_c_val)
        point_in_proj = current_manifold_obj.proju(point_in_hyperbolic.to(dtype_to_use)); tan_main_component = current_manifold_obj.logmap0(point_in_proj); tan_rel_component = torch.zeros_like(tan_main_component); ld_point_self_hyperbolic = None
        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None]:
             if relative_vectors_tangent_in.shape[0] != B_prime: raise ValueError(f"RelVec shape mismatch: {relative_vectors_tangent_in.shape[0]} != B' {B_prime}"); tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)
        if self.use_ld and self.level_descriptor_param is not None: ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use))
        tan_desc_prev_level_component = torch.zeros_like(tan_main_component)
        if descriptor_point_in_hyperbolic is not None and self.use_ld :
             if descriptor_point_in_hyperbolic.shape[0] != B_prime: raise ValueError(f"DescIn shape mismatch: {descriptor_point_in_hyperbolic.shape[0]} != B' {B_prime}"); desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use)); tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)
        inputs_for_combiner = [tan_main_component]
        if self.relative_vector_aggregation not in ['none', None]: inputs_for_combiner.append(tan_rel_component)
        if self.use_ld: inputs_for_combiner.append(tan_desc_prev_level_component)
        if self.use_spread and sigma_in_scalar_tensor is not None:
            sigma_in_expanded = sigma_in_scalar_tensor.view(-1, 1).expand(B_prime, 1).to(device=dev, dtype=dtype_to_use) # Expand sigma correctly for B'
            inputs_for_combiner.append(sigma_in_expanded)
        combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1) if len(inputs_for_combiner)>1 else inputs_for_combiner[0]; v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features); v_final_for_expmap_unclamped = v_combined_tangent_processed * current_s_val
        if self.use_flow and self.tangent_flow_module is not None: flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val; v_final_for_expmap_unclamped = v_final_for_expmap_unclamped + flow_effect
        scaled_output_tangent_for_expmap = torch.clamp(v_final_for_expmap_unclamped, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL); point_this_level_out_hyperbolic = current_manifold_obj.expmap0(scaled_output_tangent_for_expmap); tangent_out_for_aggregation = v_combined_tangent_processed.to(dtype_to_use)
        boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype=dtype_to_use, device=dev) if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None else None
        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None:
            descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B_prime, -1).to(dtype=dtype_to_use) if ld_point_self_hyperbolic.dim() == 1 else ld_point_self_hyperbolic.to(dtype=dtype_to_use)
        return (point_this_level_out_hyperbolic.to(dtype=point_in_hyperbolic.dtype), # Match input type
                tangent_out_for_aggregation.to(dtype=point_in_hyperbolic.dtype),
                descriptor_point_out_for_transform_hyperbolic.to(dtype=point_in_hyperbolic.dtype) if descriptor_point_out_for_transform_hyperbolic is not None else None,
                boundary_points_this_level_hyperbolic.to(dtype=point_in_hyperbolic.dtype) if boundary_points_this_level_hyperbolic is not None else None,
                current_sigma_out_tensor.to(dtype=point_in_hyperbolic.dtype))

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_tangent_dim: int, output_tangent_dim: int, config: Dict):
        super().__init__(); current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.WuBuModel") # Updated logger
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
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation, descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic, sigma_out_of_level_tensor) = level_module(current_point_repr_hyperbolic, aggregated_relative_vectors_from_prev_transform, descriptor_from_prev_transform_hyperbolic, sigma_from_prev_level_tensor)
            if self.hyperbolic_dims_list[i] > 0: level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)
            if i < len(self.levels_modulelist) - 1: # Check if there's a next level
                if i >= len(self.transforms_modulelist): logging.getLogger("WuBuGAADOpticalFlowDiffV0101.WuBuModel").warning(f"Missing transform L{i}->L{i+1}. Stop."); break
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
    def __init__(self, learning_rate:float=0.01, discount:float=0.95, epsilon:float=0.2, epsilon_decay:float=0.9998, min_epsilon:float=0.01, lr_scale_options:Optional[List[float]]=None, momentum_scale_options:Optional[List[float]]=None, max_q_table_size:int=10000):
        self.q_table: Dict[Tuple, Dict[str, np.ndarray]] = {}; self.alpha=learning_rate; self.gamma=discount; self.epsilon=epsilon; self.min_epsilon=min_epsilon; self.epsilon_decay=epsilon_decay; self.prev_loss:Optional[float]=None; self.prev_state:Optional[Tuple]=None; self.prev_action:Optional[Dict[str, float]]=None; self.action_ranges = {'lr_scale': np.array(lr_scale_options if lr_scale_options else [0.9,0.95,1.,1.05,1.1]), 'momentum_scale': np.array(momentum_scale_options if momentum_scale_options else [0.95,0.98,1.,1.01,1.02])}; self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}; self.loss_window=deque(maxlen=20); self.grad_norm_window=deque(maxlen=20); self.lr_window=deque(maxlen=10); self.momentum_window=deque(maxlen=10); self.performance_window=deque(maxlen=50); self.stable_steps=0; self.oscillation_counter=0; self.prev_actions_log=deque(maxlen=10); self.max_q_table_size=max(100, max_q_table_size); self.q_table_access_count:Dict[Tuple,int]=defaultdict(int); self.q_table_creation_time:Dict[Tuple,float]={}; self.flow_coefficient=0.05; self.oscillation_penalty=0.15; self.stability_reward_bonus=0.05; self.logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.QCtrl") # Updated logger
    def get_state(self, lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float]) -> Optional[Tuple]:
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm): return None
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm); self.lr_window.append(lr); self.momentum_window.append(momentum); loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2,2,2,1,0
        if len(self.loss_window) < 5 or len(self.grad_norm_window) < 5: return None
        try:
            loss_arr=np.array(list(self.loss_window)[-10:]); slope_loss=np.polyfit(np.arange(len(loss_arr)),loss_arr,1)[0] if len(loss_arr)>=3 and len(np.unique(loss_arr))>1 else 0.0; loss_trend_bin=np.digitize(slope_loss/(abs(np.median(loss_arr))+EPS),bins=[-0.05,-0.005,0.005,0.05]).item()
            grad_norm_level_bin=np.digitize(np.median(list(self.grad_norm_window)),bins=[0.1,0.5,1.5,5.0]).item(); lr_level_bin=np.digitize(lr/1e-4,bins=[0.5,2.0,10.0,50.0]).item(); momentum_level_bin=np.digitize(momentum,bins=[0.85,0.92,0.97]).item()
            if len(self.performance_window)>=5: recent_rewards=np.sign([r for r in list(self.performance_window)[-5:] if r != 0]); self.oscillation_counter=min(self.oscillation_counter+1,5) if len(recent_rewards)>=2 and np.sum(np.abs(np.diff(recent_rewards)))/2.0>=2 else max(0,self.oscillation_counter-1)
            oscillation_bin=1 if self.oscillation_counter>=3 else 0
        except Exception as e: self.logger.warning(f"Q State calculation error: {e}"); return None
        state_tuple=(loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin); self.q_table_access_count[state_tuple]+=1; return state_tuple
    def compute_reward(self, current_loss: Optional[float], prev_loss: Optional[float], grad_norm: Optional[float]) -> float:
        if current_loss is None or prev_loss is None or grad_norm is None or not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm): return 0.
        median_loss=np.median(list(self.loss_window)[:-1]) if len(self.loss_window)>1 else prev_loss; base_reward=np.tanh((prev_loss-current_loss)/(abs(median_loss)+EPS)*10.); grad_penalty=-0.1*min(1.,max(0.,(grad_norm-5.)/10.)) if grad_norm > 5. else 0.; osc_penalty=-self.oscillation_penalty if self.oscillation_counter>=3 else 0.
        current_perf_reward=base_reward+grad_penalty+osc_penalty; self.performance_window.append(current_perf_reward); self.stable_steps=self.stable_steps+1 if current_perf_reward > 0.01 else 0; stab_bonus=min(0.15,self.stability_reward_bonus*math.log1p(self.stable_steps/5.)) if current_perf_reward > 0.01 else 0.; return float(np.clip(base_reward+grad_penalty+osc_penalty+stab_bonus,-1.,1.))
    def choose_action(self, state: Optional[Tuple]) -> Dict[str, float]:
        if state is None: return {'lr_scale': 1., 'momentum_scale': 1.}
        if state not in self.q_table: self.q_table[state]={p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}; self.q_table_creation_time[state]=time.time(); self.q_table_access_count[state]=1; self._manage_q_table_size()
        self.epsilon=max(self.min_epsilon,self.epsilon*self.epsilon_decay); chosen_actions={}
        for p_type, q_vals in self.q_table[state].items():
            act_space=self.action_ranges[p_type]
            if random.random()<self.epsilon: chosen_idx=random.randrange(len(act_space))
            else: finite_q_mask=np.isfinite(q_vals); best_q_val=np.max(q_vals[finite_q_mask]) if np.any(finite_q_mask) else -np.inf; best_indices=np.where(finite_q_mask & np.isclose(q_vals,best_q_val))[0] if np.any(finite_q_mask) else []; chosen_idx=random.choice(best_indices) if len(best_indices)>0 else random.randrange(len(act_space))
            chosen_actions[p_type]=float(act_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy()); return chosen_actions
    def update(self, state: Optional[Tuple], action: Optional[Dict[str, float]], reward: float, next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None or state not in self.q_table: return
        if next_state not in self.q_table: self.q_table[next_state]={p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}; self.q_table_creation_time[next_state]=time.time(); self.q_table_access_count[next_state]=0; self._manage_q_table_size()
        for p_type, chosen_val in action.items():
            if p_type not in self.q_table[state]: continue
            act_idx_arr=np.where(np.isclose(self.action_ranges[p_type],chosen_val))[0]; act_idx=act_idx_arr[0] if len(act_idx_arr)>0 else -1
            if act_idx == -1: continue
            current_q=self.q_table[state][p_type][act_idx]; next_q_vals=self.q_table[next_state][p_type]; finite_next_q=next_q_vals[np.isfinite(next_q_vals)]; max_future_q=np.max(finite_next_q) if len(finite_next_q)>0 else 0.0; max_future_q=0.0 if not np.isfinite(max_future_q) else max_future_q; td_target=reward+self.gamma*max_future_q; td_error=td_target-current_q; adaptive_alpha=min(0.5,max(0.001,self.alpha*(1.0+self.flow_coefficient*np.tanh(abs(td_error)*0.5)))); new_q=current_q+adaptive_alpha*td_error; self.q_table[state][p_type][act_idx]=np.clip(new_q,-1e4,1e4) if np.isfinite(new_q) else 0.0
    def _manage_q_table_size(self):
        if len(self.q_table)>self.max_q_table_size:
            can_smart_prune = all([self.q_table_access_count, self.q_table_creation_time, len(self.q_table_access_count) > len(self.q_table)//2, len(self.q_table_creation_time) > len(self.q_table)//2])
            to_remove = sorted(self.q_table.keys(), key=lambda s: (self.q_table_access_count.get(s,0), self.q_table_creation_time.get(s, float('inf'))))[:len(self.q_table)-self.max_q_table_size] if can_smart_prune else random.sample(list(self.q_table.keys()), len(self.q_table) - self.max_q_table_size)
            for s_rm in to_remove: self.q_table.pop(s_rm, None); self.q_table_access_count.pop(s_rm, None); self.q_table_creation_time.pop(s_rm, None); self.logger.debug(f"Pruned Q table. New size: {len(self.q_table)}")
    def get_info(self) -> Dict: q_mem_mb=sum(sys.getsizeof(s)+sum(a.nbytes+sys.getsizeof(k) for k,a in v.items()) for s,v in self.q_table.items())/(1024**2) if self.q_table else 0.0; avg_perf_reward=np.mean(list(self.performance_window)) if self.performance_window else 0.; return {"epsilon": self.epsilon, "q_table_size": len(self.q_table), "q_table_mem_mb_approx": round(q_mem_mb, 2), "last_action": self.prev_actions_log[-1] if self.prev_actions_log else None, f"avg_reward_last_{self.performance_window.maxlen}": avg_perf_reward, "stable_steps": self.stable_steps, "oscillation_counter": self.oscillation_counter}

class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm_risgd=1.0, q_learning_config:Optional[Dict]=None):
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        # --- MOVED super().__init__ CALL TO BE BEFORE ACCESSING self.param_groups ---
        super().__init__(params, defaults)
        # --- END OF MOVE ---
        self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_learning_config) if isinstance(q_learning_config, dict) else None
        self.logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.RiSGD")
        self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled.")
        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count = 0
        self.current_loss_for_q: Optional[float] = None
        self.grad_stats = GradientStats()
        # self.param_groups is now initialized by super().__init__, so this loop is fine
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state.setdefault(p, {}) # Initialize state for each parameter

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        self.grad_stats.reset()

    def set_current_loss_for_q_controller(self, loss: Optional[float]):
        self.current_loss_for_q = loss if loss is not None and np.isfinite(loss) else None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action
            for group in self.param_groups:
                group.setdefault('base_lr', group['lr'])
                group.setdefault('base_momentum', group['momentum'])
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
                    self.logger.debug(f"Non-finite grad for param shape {p.shape}. Skipping update for this param.")
                    continue

                if norm_val is not None and norm_val > self.max_grad_norm_risgd and self.max_grad_norm_risgd > 0:
                    grad.mul_(self.max_grad_norm_risgd / (norm_val + EPS))

                state = self.state[p]
                manifold: Optional[Manifold] = getattr(p, 'manifold', None)

                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    p_proj = manifold.proju(p.data)
                    try:
                        r_grad = manifold.egrad2rgrad(p_proj, grad)
                    except Exception as e_egrad:
                        self.logger.error(f"Error calculating Riemannian gradient for param {p.shape}: {e_egrad}. Skipping param update.")
                        self.grad_stats.params_skipped_due_non_finite_grad +=1
                        state.pop('momentum_buffer', None)
                        continue

                    if not torch.isfinite(r_grad).all():
                        self.logger.warning(f"Non-finite Riemannian gradient for param shape {p.shape}. Skipping param update.")
                        self.grad_stats.params_skipped_due_non_finite_grad += 1
                        state.pop('momentum_buffer', None)
                        continue

                    update_vec = r_grad
                    if wd != 0:
                        try:
                            log_p = manifold.logmap0(p_proj)
                            if torch.isfinite(log_p).all():
                                update_vec = update_vec.add(log_p, alpha=wd)
                            else:
                                self.logger.warning(f"Non-finite logmap for weight decay on param {p.shape}. Skipping WD.")
                        except Exception as e_wd:
                            self.logger.warning(f"Error in weight decay logmap for param {p.shape}: {e_wd}. Skipping WD.")

                    buf = state.setdefault('momentum_buffer', torch.zeros_like(update_vec))
                    buf.mul_(mom).add_(update_vec)

                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite momentum buffer for param {p.shape}. Resetting buffer.")
                        buf.zero_()

                    try:
                        expmap_arg = buf.mul(-lr)
                        if not torch.isfinite(expmap_arg).all():
                            self.logger.warning(f"Non-finite argument to expmap for param {p.shape}. Resetting update.")
                            state.get('momentum_buffer',torch.zeros(0)).zero_() # Reset momentum
                            continue

                        new_p_candidate = manifold.expmap(p_proj, expmap_arg)

                        if not torch.isfinite(new_p_candidate).all():
                            self.logger.warning(f"Non-finite result from expmap for param {p.shape}. Trying fallback.")
                            new_p_candidate = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0))
                            state.get('momentum_buffer',torch.zeros(0)).zero_()

                        p.data = manifold.proju(new_p_candidate)

                        if not torch.isfinite(p.data).all():
                            self.logger.error(f"Parameter became non-finite after update and projection for param {p.shape}. Resetting to origin.")
                            p.data = manifold.expmap0(torch.zeros_like(p.data))
                            state.get('momentum_buffer',torch.zeros(0)).zero_()
                    except Exception as e_hyp_update:
                        self.logger.error(f"Error during hyperbolic update for param {p.shape}: {e_hyp_update}. Resetting momentum.")
                        state.get('momentum_buffer',torch.zeros(0)).zero_()
                else:  # Euclidean parameter update
                    d_p = grad.clone()
                    if wd != 0:
                        d_p.add_(p.data, alpha=wd)
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(p.data))
                    buf.mul_(mom).add_(d_p)
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite Euclidean momentum buffer for param {p.shape}. Resetting.")
                        buf.zero_()
                    p.data.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all():
                        self.logger.warning(f"Euclidean param became non-finite after update {p.shape}. Resetting momentum & NaNing.")
                        p.data = torch.nan_to_num(p.data, nan=0.0)
                        state.get('momentum_buffer',torch.zeros(0)).zero_()

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
# Architectural Components (v0.10.1 - Motion Encoder Updated)
# =====================================================================

class RegionalPatchExtractor(nn.Module):
    def __init__(self, patch_output_size: Optional[Tuple[int, int]] = None, feature_extractor: Optional[nn.Module] = None, feature_map_spatial_scale: float = 1.0, roi_align_output_size: Optional[Tuple[int, int]] = None, use_roi_align: bool = False):
        super().__init__(); self.patch_output_size = patch_output_size; self.feature_extractor = feature_extractor; self.feature_map_spatial_scale = feature_map_spatial_scale; self.roi_align_output_size = roi_align_output_size; self.use_roi_align = use_roi_align; current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.PatchExtract"); self.resize_transform=None # Updated logger
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
            except Exception as e_roi: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.PatchExtract").error(f"RoIAlign failed: {e_roi}. FeatMap:{feature_maps.shape}, RoIs:{all_rois.shape}, Output:{self.roi_align_output_size}"); raise e_roi
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

class RegionalHyperbolicEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_config: Dict):
        super().__init__(); self.args = args; self.video_config = video_config; self.gaad_config = gaad_config; self.wubu_s_config = wubu_s_config; self.image_size = (args.image_h, args.image_w); self.num_appearance_regions = gaad_config['num_regions']; self.decomposition_type = gaad_config['decomposition_type']; self.gaad_min_size_px = gaad_config.get('min_size_px', 5); current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.EncoderS"); self.feature_extractor: Optional[nn.Module] = None; feature_map_scale = 1.0; patch_input_channels = self.video_config['num_channels']; roi_align_output_size = None; use_roi_align = False # Updated logger
        if args.encoder_use_roi_align:
            self.feature_extractor = nn.Sequential(nn.Conv2d(self.video_config['num_channels'], args.encoder_shallow_cnn_channels, kernel_size=3, stride=1, padding=1), nn.GroupNorm(8, args.encoder_shallow_cnn_channels), nn.GELU()); patch_input_channels = args.encoder_shallow_cnn_channels; roi_align_output_size = (args.encoder_roi_align_output_h, args.encoder_roi_align_output_w); use_roi_align = True; current_logger.info(f"Using RoIAlign (OutCh: {patch_input_channels}, RoISize: {roi_align_output_size})")
        else: current_logger.info(f"Using Pixel Patches (Resize: {args.encoder_pixel_patch_size}x{args.encoder_pixel_patch_size})")
        self.patch_extractor = RegionalPatchExtractor(patch_output_size=(args.encoder_pixel_patch_size, args.encoder_pixel_patch_size) if not use_roi_align else None, feature_extractor=self.feature_extractor, feature_map_spatial_scale=feature_map_scale, roi_align_output_size=roi_align_output_size, use_roi_align=use_roi_align)
        patch_output_h = roi_align_output_size[0] if use_roi_align else args.encoder_pixel_patch_size; patch_output_w = roi_align_output_size[1] if use_roi_align else args.encoder_pixel_patch_size; patch_feature_dim = patch_input_channels * patch_output_h * patch_output_w; self.patch_embed = PatchEmbed(patch_feature_dim, args.encoder_initial_tangent_dim)
        self.wubu_s = FullyHyperbolicWuBuNestingModel(input_tangent_dim=args.encoder_initial_tangent_dim, output_tangent_dim=video_config['wubu_s_output_dim'], config=wubu_s_config); self.wubu_s_final_hyp_dim = wubu_s_config['hyperbolic_dims'][-1] if wubu_s_config['num_levels'] > 0 and wubu_s_config['hyperbolic_dims'] else 0; self.wubu_s_final_curvature = 1.0
        if wubu_s_config['num_levels'] > 0 and self.wubu_s_final_hyp_dim > 0:
            last_level_idx = wubu_s_config['num_levels'] - 1
            try: temp_level = HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_s_final_hyp_dim, wubu_s_config, wubu_s_config['initial_curvatures'][last_level_idx]); self.wubu_s_final_curvature = temp_level.get_current_curvature_scalar(); del temp_level; current_logger.info(f"WuBu-S final level curvature estimated as {self.wubu_s_final_curvature:.3f}")
            except IndexError: current_logger.error(f"Index error accessing init curvatures WuBu-S L{last_level_idx}. Default C=1.0."); self.wubu_s_final_curvature = 1.0
        self.apply(init_weights_general)
    def forward(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_frames, C, H, W = frames_pixels.shape; device = frames_pixels.device; dtype = frames_pixels.dtype; frames_pixels_flat = frames_pixels.reshape(B * N_frames, C, H, W); gaad_bboxes_list = []
        for b_idx in range(B):
            current_frame_h, current_frame_w = frames_pixels.shape[3], frames_pixels.shape[4]; frame_dims = (current_frame_w, current_frame_h); max_w_scalar=float(frame_dims[0]); max_h_scalar=float(frame_dims[1]);
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
        wubu_input = initial_tangent_vectors.reshape(B_flat_post * NumReg_post, -1); wubu_output_tangent = self.wubu_s(wubu_input);
        if self.wubu_s_config['num_levels'] > 0 and self.wubu_s_final_hyp_dim > 0: final_manifold=PoincareBall(self.wubu_s_final_curvature); regional_hyperbolic_features_flat=final_manifold.expmap0(wubu_output_tangent)
        else: regional_hyperbolic_features_flat=wubu_output_tangent; logging.getLogger("WuBuGAADOpticalFlowDiffV0101.EncoderS").debug("WuBu-S has no hyp levels/output; feats remain tangent.")
        D_out = regional_hyperbolic_features_flat.shape[-1]; regional_hyperbolic_features = regional_hyperbolic_features_flat.reshape(B, N_frames, NumReg_post, D_out)
        return regional_hyperbolic_features, gaad_bboxes_full

# --- UPDATED: Motion Encoder using Optical Flow ---
# --- UPDATED: Motion Encoder using Optical Flow ---
class RegionalHyperbolicMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_motion_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_m_config = wubu_m_config
        self.logger = logging.getLogger("WuBuGAADOpticalFlowDiffV0101.EncoderM")
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
            self.wubu_m = FullyHyperbolicWuBuNestingModel(input_tangent_dim=args.encoder_initial_tangent_dim, output_tangent_dim=video_config['wubu_m_output_dim'], config=wubu_m_config)
            self.wubu_m_final_hyp_dim = wubu_m_config['hyperbolic_dims'][-1] if wubu_m_config['num_levels'] > 0 and wubu_m_config['hyperbolic_dims'] else 0
            self.wubu_m_final_curvature = 1.0
            if wubu_m_config['num_levels'] > 0 and self.wubu_m_final_hyp_dim > 0:
                last_level_idx = wubu_m_config['num_levels'] - 1
                try: temp_level_m=HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_m_final_hyp_dim, wubu_m_config, wubu_m_config['initial_curvatures'][last_level_idx]); self.wubu_m_final_curvature = temp_level_m.get_current_curvature_scalar(); del temp_level_m; self.logger.info(f"WuBu-M final level curvature estimated as {self.wubu_m_final_curvature:.3f}")
                except IndexError: self.logger.error(f"Index error accessing init curvatures WuBu-M L{last_level_idx}. Default C=1.0."); self.wubu_m_final_curvature = 1.0
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
        if not self.enabled or self.flow_net is None : return None # WuBu-M check is implicit if self.enabled and flow_stats_dim > 0
        B, N_frames, C, H, W = frames_pixels.shape; device = frames_pixels.device; original_dtype = frames_pixels.dtype; compute_dtype = next(self.parameters()).dtype
        if N_frames < 2: self.logger.debug(f"Not enough frames ({N_frames}) for flow. Skip motion."); return None
        num_pairs = N_frames - 1; all_motion_features_hyperbolic_list = []; all_motion_bboxes_list = []

        # --- FIX: Removed uint8 conversion. Use frames_pixels directly ---
        # frames_uint8 = ((frames_pixels.clamp(-1, 1) * 0.5 + 0.5) * 255.0).to(torch.uint8) # REMOVED
        # Assuming frames_pixels is float and normalized [-1, 1] or [0, 1]

        flow_context = torch.no_grad() if self.args.freeze_flow_net else contextlib.nullcontext()
        with flow_context:
            for i in range(num_pairs):
                # Use slices from the original float tensor
                frame_t = frames_pixels[:, i+1, ...]
                frame_t_minus_1 = frames_pixels[:, i, ...]

                # --- Ensure frames are float32 for the network ---
                # RAFT models in torchvision often expect float32 input
                frame_t_float = frame_t.to(torch.float32)
                frame_t_minus_1_float = frame_t_minus_1.to(torch.float32)

                try:
                    # Pass the float32 tensors to the flow network
                    flow_predictions = self.flow_net(frame_t_minus_1_float, frame_t_float)
                    # Extract the final flow field and convert to the model's compute dtype (e.g., float16 if using AMP)
                    flow_field = flow_predictions[-1].to(compute_dtype)
                except ValueError as ve:
                    # Specific check for the divisibility error
                    if "should be divisible by 8" in str(ve):
                         self.logger.error(f"Optical flow input shape error for pair {i}: {ve}. Frame shape: H={H}, W={W}. Ensure image_h and image_w are divisible by 8.", exc_info=False)
                    else:
                        self.logger.error(f"Optical flow failed pair {i} (Input Shapes: {frame_t_minus_1_float.shape}, {frame_t_float.shape}): {ve}", exc_info=True)
                    return None # Stop processing if flow fails
                except Exception as e_flow:
                    self.logger.error(f"Optical flow failed pair {i} (Input Shapes: {frame_t_minus_1_float.shape}, {frame_t_float.shape}): {e_flow}", exc_info=True)
                    return None # Stop processing if flow fails

                # --- Remainder of the flow processing ---
                # Calculate magnitude using the computed flow_field (already compute_dtype)
                flow_magnitude = torch.sqrt(flow_field[:, 0:1, :, :]**2 + flow_field[:, 1:2, :, :]**2)

                # Get bounding boxes based on flow magnitude map dimensions
                motion_gaad_bboxes_batch = self._get_motion_gaad_bboxes(flow_magnitude, device, compute_dtype)

                if self.flow_stats_dim > 0:
                    # Extract stats using the computed flow_field
                    flow_stats = self._extract_flow_statistics(flow_field, motion_gaad_bboxes_batch)
                    flow_stats_flat = flow_stats.view(B * self.num_motion_regions, self.flow_stats_dim)
                    initial_motion_tangent_vectors_flat = self.motion_feature_embed(flow_stats_flat)
                else:
                    initial_motion_tangent_vectors_flat = torch.zeros(B * self.num_motion_regions, self.args.encoder_initial_tangent_dim, device=device, dtype=compute_dtype)

                # --- WuBu-M processing ---
                if self.wubu_m is None: # If WuBu-M wasn't initialized (e.g. flow_stats_dim=0)
                    motion_features_hyperbolic_flat = initial_motion_tangent_vectors_flat # Pass through tangent
                else:
                    wubu_m_output_tangent_flat = self.wubu_m(initial_motion_tangent_vectors_flat)
                    if self.wubu_m_config is not None and self.wubu_m_config['num_levels'] > 0 and self.wubu_m_final_hyp_dim > 0:
                        # Use the stored curvature for the final manifold
                        final_manifold_m = PoincareBall(self.wubu_m_final_curvature)
                        motion_features_hyperbolic_flat = final_manifold_m.expmap0(wubu_m_output_tangent_flat)
                    else: # If no hyperbolic levels in WuBu-M, output remains tangent
                        motion_features_hyperbolic_flat = wubu_m_output_tangent_flat

                # Reshape and store results for the current pair
                motion_features_hyperbolic_pair = motion_features_hyperbolic_flat.reshape(B, self.num_motion_regions, -1)
                all_motion_features_hyperbolic_list.append(motion_features_hyperbolic_pair)
                all_motion_bboxes_list.append(motion_gaad_bboxes_batch)
        # --- End of loop ---

        if not all_motion_features_hyperbolic_list:
            self.logger.warning("No motion features were generated (likely due to flow errors or N_frames < 2).")
            return None # Return None if no features were produced

        # Stack features and bboxes across pairs, converting back to original dtype
        final_motion_features = torch.stack(all_motion_features_hyperbolic_list, dim=1).to(original_dtype)
        final_motion_bboxes = torch.stack(all_motion_bboxes_list, dim=1).to(original_dtype)

        return final_motion_features, final_motion_bboxes
        
class RegionalPixelSynthesisDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_config: Dict):
        super().__init__(); self.args = args; self.video_config = video_config; self.image_size = (args.image_h, args.image_w); self.num_regions = gaad_config['num_regions']; self.decoder_type = args.decoder_type; self.num_channels = video_config['num_channels']; self.input_tangent_dim = video_config['wubu_s_output_dim']; current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Decoder") # Updated logger
        if self.decoder_type == "patch_gen":
            self.patch_size = args.decoder_patch_gen_size; patch_pixels = self.num_channels * self.patch_size * self.patch_size
            self.patch_generator = nn.Sequential(nn.Linear(self.input_tangent_dim, self.input_tangent_dim * 2), nn.GELU(), nn.Linear(self.input_tangent_dim * 2, patch_pixels), nn.Tanh()); self.patch_resize_mode = args.decoder_patch_resize_mode; current_logger.info(f"Using Patch Generator (OutSize: {self.patch_size}x{self.patch_size}, Resize: {self.patch_resize_mode})")
        elif self.decoder_type == "transformer": current_logger.warning("Decoder Transformer type NIY."); raise NotImplementedError("Transformer decoder NIY")
        else: raise ValueError(f"Unknown decoder_type: {self.decoder_type}")
        self.apply(init_weights_general)
    def forward(self, regional_tangent_features: torch.Tensor, gaad_bboxes: torch.Tensor) -> torch.Tensor:
        B, N_pred, NumReg, D_tan = regional_tangent_features.shape; C, H, W = self.num_channels, self.image_size[0], self.image_size[1]; device = regional_tangent_features.device; dtype = regional_tangent_features.dtype; regional_tangent_flat = regional_tangent_features.view(B * N_pred * NumReg, D_tan); gaad_bboxes_flat = gaad_bboxes.view(B * N_pred, NumReg, 4); output_frames = torch.zeros(B, N_pred, C, H, W, device=device, dtype=dtype)
        if self.decoder_type == "patch_gen":
            generated_patch_pixels_flat = self.patch_generator(regional_tangent_flat); generated_patches = generated_patch_pixels_flat.view(B * N_pred, NumReg, C, self.patch_size, self.patch_size); canvas = torch.zeros(B * N_pred, C, H, W, device=device, dtype=dtype); counts = torch.zeros(B * N_pred, 1, H, W, device=device, dtype=dtype)
            for i in range(B * N_pred):
                for r in range(NumReg):
                    patch = generated_patches[i, r]; x1, y1, x2, y2 = gaad_bboxes_flat[i, r].tolist(); target_h = int(round(y2 - y1)); target_w = int(round(x2 - x1)); place_y1 = int(round(y1)); place_x1 = int(round(x1));
                    if target_h <= 0 or target_w <= 0: continue
                    resize_kwargs = {'size': (target_h, target_w), 'mode': self.patch_resize_mode}
                    if self.patch_resize_mode != 'nearest': resize_kwargs['align_corners'] = False
                    if self.patch_resize_mode == 'nearest': resize_kwargs.pop('align_corners', None)
                    resized_patch = F.interpolate(patch.unsqueeze(0), **resize_kwargs).squeeze(0)
                    place_y2 = min(H, place_y1 + target_h); place_x2 = min(W, place_x1 + target_w); place_y1 = max(0, place_y1); place_x1 = max(0, place_x1); slice_h = place_y2 - place_y1; slice_w = place_x2 - place_x1;
                    if slice_h <= 0 or slice_w <= 0: continue
                    canvas[i, :, place_y1:place_y2, place_x1:place_x2] += resized_patch[:, :slice_h, :slice_w]; counts[i, :, place_y1:place_y2, place_x1:place_x2] += 1
            output_canvas_flat = torch.where(counts > 0, canvas / counts.clamp(min=1.0), canvas); output_frames = output_canvas_flat.view(B, N_pred, C, H, W)
        else: raise NotImplementedError(f"Decoder type {self.decoder_type} forward pass NIY.")
        return output_frames

class TransformerNoisePredictor(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, wubu_s_config: Dict, wubu_t_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__(); self.args = args; self.video_config = video_config; self.transformer_config = args.transformer_noise_predictor_config; self.num_regions = args.gaad_num_regions; self.d_model = self.transformer_config['d_model']; self.input_feat_dim = video_config['wubu_s_output_dim']; self.time_embed_dim = args.diffusion_time_embedding_dim; self.wubu_t_output_dim = video_config.get('wubu_t_output_dim', 0) if wubu_t_config and wubu_t_config.get('num_levels', 0) > 0 else 0; self.input_proj = nn.Linear(self.input_feat_dim, self.d_model); self.time_proj = nn.Linear(self.time_embed_dim, self.d_model); self.pos_embed = nn.Parameter(torch.zeros(1, self.num_regions, self.d_model)); nn.init.normal_(self.pos_embed, std=0.02); self.context_proj = nn.Linear(self.wubu_t_output_dim, self.d_model) if self.wubu_t_output_dim > 0 else None; encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.transformer_config['num_heads'], dim_feedforward=int(self.d_model * self.transformer_config['d_ff_ratio']), dropout=self.transformer_config['dropout'], activation=self.transformer_config['activation'], batch_first=True); self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_config['num_layers']); self.output_proj = nn.Linear(self.d_model, self.input_feat_dim); self.apply(init_weights_general); logging.getLogger("WuBuGAADOpticalFlowDiffV0101.TNP").info(f"Layers={self.transformer_config['num_layers']}, Heads={self.transformer_config['num_heads']}, Dim={self.d_model}") # Updated logger
    def forward(self, noisy_regional_tangent_features: torch.Tensor, time_embedding: torch.Tensor, temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_pred, NumReg, D_in = noisy_regional_tangent_features.shape; noisy_features_flat = noisy_regional_tangent_features.reshape(B * N_pred, NumReg, D_in); time_embedding_flat = time_embedding.reshape(B * N_pred, -1); x = self.input_proj(noisy_features_flat); t_emb = self.time_proj(time_embedding_flat); x = x + t_emb.unsqueeze(1) + self.pos_embed
        if temporal_context is not None and self.context_proj is not None: context_flat = temporal_context.reshape(B * N_pred, -1); ctx_emb = self.context_proj(context_flat); x = x + ctx_emb.unsqueeze(1)
        transformer_output = self.transformer_encoder(x); predicted_noise_flat = self.output_proj(transformer_output); predicted_noise = predicted_noise_flat.reshape(B, N_pred, NumReg, D_in); return predicted_noise

# =====================================================================
# Diffusion Model Specific Components
# =====================================================================

class SinusoidalPhiEmbedding(nn.Module):
    def __init__(self, dim: int, base_freq_phi_scaled: float = 10000.0, use_phi_paper_scaling_arg: bool = False, phi_constant: float = PHI):
        super().__init__(); self.dim = dim; self.base_freq_for_paper_scaling = base_freq_phi_scaled; self.base_period_for_ddpm_scaling = base_freq_phi_scaled; self.use_phi_paper_scaling = use_phi_paper_scaling_arg; self.phi_constant = phi_constant; half_dim = dim // 2; denominators = torch.empty(0, dtype=torch.float); current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.SinEmb") # Updated logger
        if half_dim > 0:
            if self.use_phi_paper_scaling: exponent_val=torch.arange(half_dim).float()/float(half_dim); phi_scaling_factor=self.phi_constant**exponent_val; denominators=self.base_freq_for_paper_scaling/(phi_scaling_factor+EPS); current_logger.info(f"PHI paper scaling. Dim {dim}, BaseFreq ω: {self.base_freq_for_paper_scaling:.1f}, PHI: {self.phi_constant:.3f}")
            else: denominators=torch.exp(torch.arange(half_dim).float()*-(math.log(self.base_period_for_ddpm_scaling)/(half_dim-1.0))) if half_dim>1 else torch.tensor([1.0/self.base_period_for_ddpm_scaling]); current_logger.info(f"DDPM-style scaling. Dim {dim}, BasePeriod: {self.base_period_for_ddpm_scaling:.1f}")
        else: current_logger.info(f"Dim {dim} too small, empty/zero embedding.")
        self.register_buffer("denominators_actual", denominators)
    def forward(self, t: torch.Tensor, phi_time_scale: float = 1.0) -> torch.Tensor:
        if self.dim==0: return torch.empty(t.shape[0],0,device=t.device,dtype=torch.float)
        if self.denominators_actual.numel()==0: return torch.zeros(t.shape[0],self.dim,device=t.device,dtype=torch.float) if self.dim !=1 else (t.float()*phi_time_scale).unsqueeze(-1)
        t_scaled=t.float()*phi_time_scale; args=t_scaled[:,None]*self.denominators_actual[None,:]; embedding=torch.cat([torch.cos(args),torch.sin(args)],dim=-1);
        if self.dim % 2 != 0 and self.dim > 0: embedding=F.pad(embedding,(0,1),value=0.0)
        return embedding

class GAADWuBuRegionalDiffNet(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_appearance_config: Dict, gaad_motion_config: Optional[Dict], wubu_s_config: Dict, wubu_t_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args; self.video_config = video_config; self.gaad_appearance_config = gaad_appearance_config; self.gaad_motion_config = gaad_motion_config; self.wubu_s_config = wubu_s_config; self.wubu_t_config = wubu_t_config; self.wubu_m_config = wubu_m_config; current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.MainNet") # Updated logger
        self.encoder = RegionalHyperbolicEncoder(args, video_config, gaad_appearance_config, wubu_s_config)
        self.motion_encoder: Optional[RegionalHyperbolicMotionEncoder] = None
        if args.use_wubu_motion_branch:
             temp_motion_encoder = RegionalHyperbolicMotionEncoder(args, video_config, gaad_motion_config, wubu_m_config)
             if temp_motion_encoder.enabled: self.motion_encoder = temp_motion_encoder; current_logger.info("Motion Encoder Branch Activated (Optical Flow based).")
             else: current_logger.warning("Motion branch requested but disabled (check optical flow availability/config)."); args.use_wubu_motion_branch = False; self.wubu_m_config = None; self.gaad_motion_config = None; video_config['wubu_m_output_dim'] = 0
        self.wubu_t: Optional[FullyHyperbolicWuBuNestingModel] = None; self.wubu_t_input_dim = 0
        if self.wubu_t_config and self.wubu_t_config['num_levels'] > 0:
            self.wubu_t_input_dim = video_config['wubu_s_output_dim']
            if args.use_wubu_motion_branch and self.motion_encoder and self.motion_encoder.enabled and video_config.get('wubu_m_output_dim', 0) > 0:
                 self.wubu_t_input_dim += video_config['wubu_m_output_dim']; current_logger.info(f"WuBu-T: Including motion features (dim {video_config.get('wubu_m_output_dim', 0)}).")
            elif args.use_wubu_motion_branch: current_logger.info("WuBu-T: Motion branch features not included (disabled or dim 0).")
            if self.wubu_t_input_dim > 0:
                 self.wubu_t = FullyHyperbolicWuBuNestingModel(input_tangent_dim=self.wubu_t_input_dim, output_tangent_dim=video_config['wubu_t_output_dim'], config=wubu_t_config)
                 current_logger.info(f"WuBu-T Enabled: InputDim {self.wubu_t_input_dim}, OutputDim {video_config['wubu_t_output_dim']}")
            else: current_logger.warning("WuBu-T configured but effective input dim is 0. Disabling WuBu-T.");
            if self.wubu_t_config and self.wubu_t_input_dim == 0: self.wubu_t_config['num_levels'] = 0; video_config['wubu_t_output_dim'] = 0
        self.time_sin_embedding = SinusoidalPhiEmbedding(args.diffusion_time_embedding_dim, base_freq_phi_scaled=gaad_appearance_config.get("phi_time_base_freq", 10000.0), use_phi_paper_scaling_arg=args.use_phi_frequency_scaling_for_time_emb, phi_constant=PHI)
        self.time_fc_mlp = nn.Sequential(nn.Linear(args.diffusion_time_embedding_dim, args.diffusion_time_embedding_dim * 2), nn.GELU(), nn.Linear(args.diffusion_time_embedding_dim * 2, args.diffusion_time_embedding_dim))
        self.noise_predictor = TransformerNoisePredictor(args, video_config, wubu_s_config, self.wubu_t_config, self.wubu_m_config)
        args.transformer_noise_predictor_config.setdefault('d_model', DEFAULT_CONFIG_TRANSFORMER_NOISE_PREDICTOR['d_model'])
        self.decoder = RegionalPixelSynthesisDecoder(args, video_config, gaad_appearance_config, wubu_s_config)
        self.apply(init_weights_general); param_count = sum(p.numel() for p in self.parameters() if p.requires_grad); current_logger.info(f"GAADWuBuRegionalDiffNet (v0.10.1 OpticalFlow) Initialized: {param_count:,} params.")
    def encode_frames(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        regional_app_features, gaad_bboxes_app = self.encoder(frames_pixels); regional_motion_features = None
        if self.motion_encoder is not None and self.motion_encoder.enabled:
            motion_output_tuple = self.motion_encoder(frames_pixels)
            if motion_output_tuple is not None: regional_motion_features, _ = motion_output_tuple; regional_motion_features = regional_motion_features.to(regional_app_features.dtype) if regional_motion_features.dtype != regional_app_features.dtype else regional_motion_features
        return regional_app_features, gaad_bboxes_app, regional_motion_features
    def decode_features(self, regional_tangent_features: torch.Tensor, gaad_bboxes: torch.Tensor) -> torch.Tensor:
        return self.decoder(regional_tangent_features, gaad_bboxes)
    def forward(self, noisy_regional_tangent_features: torch.Tensor, time_t_integers: torch.Tensor, conditioning_regional_app_features: Optional[torch.Tensor] = None, conditioning_regional_motion_features: Optional[torch.Tensor] = None, cfg_unconditional_flag: bool = False ) -> torch.Tensor:
        B, N_pred, NumReg, D_tan_s = noisy_regional_tangent_features.shape; device = noisy_regional_tangent_features.device; dtype = noisy_regional_tangent_features.dtype; time_sin_emb = self.time_sin_embedding(time_t_integers, phi_time_scale=self.gaad_appearance_config.get("phi_time_diffusion_scale", 1.0)); time_emb = self.time_fc_mlp(time_sin_emb).to(dtype); time_emb_expanded = time_emb.unsqueeze(1).expand(-1, N_pred, -1); temporal_context: Optional[torch.Tensor] = None
        if self.wubu_t is not None and conditioning_regional_app_features is not None and not cfg_unconditional_flag:
            N_cond_app = conditioning_regional_app_features.shape[1]; final_manifold_s = PoincareBall(self.encoder.wubu_s_final_curvature); cond_app_tangent = final_manifold_s.logmap0(conditioning_regional_app_features.to(dtype)); aggregated_app_context = torch.max(cond_app_tangent, dim=2)[0]; wubu_t_sequence_input_list = [aggregated_app_context]
            if conditioning_regional_motion_features is not None and self.motion_encoder and self.motion_encoder.enabled and hasattr(self.motion_encoder, 'wubu_m_final_curvature'):
                N_cond_mot = conditioning_regional_motion_features.shape[1]; motion_final_c = getattr(self.motion_encoder, 'wubu_m_final_curvature', 1.0); motion_final_c = float(motion_final_c) if isinstance(motion_final_c, (float, int)) else 1.0; final_manifold_m = PoincareBall(motion_final_c); cond_mot_tangent = final_manifold_m.logmap0(conditioning_regional_motion_features.to(dtype)); aggregated_mot_context = torch.max(cond_mot_tangent, dim=2)[0]
                if N_cond_mot < N_cond_app: padding=torch.zeros(B,N_cond_app-N_cond_mot,aggregated_mot_context.shape[-1],device=device,dtype=dtype); aggregated_mot_context_padded=torch.cat([padding, aggregated_mot_context], dim=1)
                elif N_cond_mot > N_cond_app: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.MainNet").warning(f"Motion ctx ({N_cond_mot}) > App ctx ({N_cond_app}). Truncating."); aggregated_mot_context_padded=aggregated_mot_context[:,:N_cond_app,:]
                else: aggregated_mot_context_padded=aggregated_mot_context
                if aggregated_mot_context_padded.shape[-1] == self.video_config.get('wubu_m_output_dim', 0): wubu_t_sequence_input_list.append(aggregated_mot_context_padded)
                else: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.MainNet").warning(f"Motion feature dim mismatch ({aggregated_mot_context_padded.shape[-1]}) vs expected ({self.video_config.get('wubu_m_output_dim', 0)}). Skip motion for WuBu-T.")
            elif conditioning_regional_motion_features is not None and self.args.use_wubu_motion_branch: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.MainNet").warning("Motion feats provided/expected, but motion encoder/curvature invalid. Skip motion for WuBu-T.")
            wubu_t_sequence_input = torch.cat(wubu_t_sequence_input_list, dim=-1)
            if wubu_t_sequence_input.shape[1] > 0 and self.wubu_t_input_dim > 0 and wubu_t_sequence_input.shape[2] == self.wubu_t_input_dim:
                 wubu_t_output = self.wubu_t(wubu_t_sequence_input); temporal_context_per_batch = wubu_t_output[:, -1, :]; temporal_context = temporal_context_per_batch.unsqueeze(1).expand(-1, N_pred, -1)
            elif self.wubu_t_input_dim > 0: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.MainNet").warning(f"WuBu-T input shape mismatch/empty seq. Got:{wubu_t_sequence_input.shape}, ExpectFeat:{self.wubu_t_input_dim}. Skip temporal ctx.")
        elif self.wubu_t is not None and not cfg_unconditional_flag and self.wubu_t_input_dim > 0: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.MainNet").debug("WuBu-T active but no cond app feats or CFG is unconditional. Skip temporal ctx.")
        predicted_noise = self.noise_predictor(noisy_regional_tangent_features, time_emb_expanded, temporal_context.to(dtype) if temporal_context is not None else None); return predicted_noise

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02): return torch.linspace(beta_start, beta_end, timesteps)
def cosine_beta_schedule(timesteps, s=0.008): steps = timesteps + 1; x = torch.linspace(0, timesteps, steps); alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2; alphas_cumprod = alphas_cumprod / alphas_cumprod[0]; betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]); return torch.clip(betas, 0.0001, 0.9999)

def q_sample_regional(x0_regional_hyperbolic: torch.Tensor, t: torch.Tensor, manifold: Manifold, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N_pred, NumReg, D_hyp = x0_regional_hyperbolic.shape; device = x0_regional_hyperbolic.device; dtype = x0_regional_hyperbolic.dtype; x0_flat = x0_regional_hyperbolic.view(B * N_pred * NumReg, D_hyp); current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.QSample") # Updated logger
    try: tangent_x0_flat = manifold.logmap0(x0_flat); tangent_x0_flat = torch.nan_to_num(tangent_x0_flat,nan=0.0,posinf=TAN_VEC_CLAMP_VAL,neginf=-TAN_VEC_CLAMP_VAL) if not torch.isfinite(tangent_x0_flat).all() else tangent_x0_flat
    except Exception as e_logmap: current_logger.error(f"Logmap0 failed: {e_logmap}. Use zeros.", exc_info=False); tangent_x0_flat = torch.zeros_like(x0_flat)
    noise_tangent = torch.randn_like(tangent_x0_flat) if noise is None else noise.reshape(B*N_pred*NumReg,D_hyp).to(device,dtype)
    t_expanded = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, N_pred, NumReg, 1).reshape(B * N_pred * NumReg)
    sqrt_alpha_t = sqrt_alphas_cumprod.gather(0,t_expanded).unsqueeze(-1).to(device,dtype); sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod.gather(0,t_expanded).unsqueeze(-1).to(device,dtype); noisy_tangent_flat = sqrt_alpha_t * tangent_x0_flat + sqrt_one_minus_alpha_t * noise_tangent
    try: xt_regional_hyperbolic_flat = manifold.expmap0(noisy_tangent_flat); xt_regional_hyperbolic_flat = manifold.proju(torch.nan_to_num(xt_regional_hyperbolic_flat,nan=0.0)) if not torch.isfinite(xt_regional_hyperbolic_flat).all() else xt_regional_hyperbolic_flat
    except Exception as e_expmap: current_logger.error(f"Expmap0 failed: {e_expmap}. Use origin proj.", exc_info=False); xt_regional_hyperbolic_flat=manifold.proju(torch.zeros_like(noisy_tangent_flat))
    xt_regional_hyperbolic = xt_regional_hyperbolic_flat.view(B, N_pred, NumReg, D_hyp); noise_tangent_reshaped = noise_tangent.view(B, N_pred, NumReg, D_hyp); return xt_regional_hyperbolic, noise_tangent_reshaped

class VideoFrameDataset(Dataset):
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int], frame_skip: int = 1, data_fraction: float = 1.0):
        super().__init__(); self.video_path = video_path; self.num_frames_total = num_frames_total; self.image_size = image_size; self.frame_skip = frame_skip; current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Dataset") # Updated logger
        if not os.path.isfile(self.video_path): current_logger.error(f"Video file not found: {self.video_path}"); raise FileNotFoundError(f"Video file not found: {self.video_path}")
        current_logger.info(f"Attempting to load entire video into RAM: {self.video_path}")
        if not VIDEO_IO_AVAILABLE: current_logger.error("torchvision.io.read_video is not available."); raise RuntimeError("torchvision.io.read_video is not available.")
        try: video_data = video_io.read_video(self.video_path, output_format="TCHW", pts_unit="sec"); self.video_frames_in_ram = video_data[0].contiguous(); self.source_video_fps = video_data[2].get('video_fps', 30.0); ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3); current_logger.info(f"Loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. Est RAM: {ram_usage_gb:.2f} GB.")
        except Exception as e: current_logger.error(f"Failed to load video '{self.video_path}' into RAM: {e}", exc_info=True); raise RuntimeError(f"Failed to load video '{self.video_path}' into RAM.") from e
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
                except Exception as e: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Dataset").error(f"Error transforming frame {actual_frame_idx_in_ram} for sample {idx}: {e}", exc_info=True); raise e
            else: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Dataset").error(f"Frame index {actual_frame_idx_in_ram} out of bounds (total: {self.num_disk_frames}). Sample: {idx}"); raise IndexError("Frame index out of bounds.")
        if len(frames_for_sample) != self.num_frames_total: logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Dataset").error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total} for sample {idx}"); raise ValueError("Incorrect number of frames loaded for sample.")
        return torch.stack(frames_for_sample)

class DiffusionTrainer:
    def __init__(self, model: GAADWuBuRegionalDiffNet, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], args: argparse.Namespace, rank: int, world_size: int, ddp_active: bool):
        self.model = model; self.optimizer = optimizer; self.device = device; self.train_loader = train_loader; self.val_loader = val_loader; self.args = args; self.rank = rank; self.world_size = world_size; self.ddp_active = ddp_active; self.am_main_process = (rank == 0); current_logger=logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Trainer") # Updated logger
        self.video_config=model.video_config; self.gaad_appearance_config=model.gaad_appearance_config; self.gaad_motion_config=model.gaad_motion_config; self.wubu_s_config=model.wubu_s_config; self.wubu_t_config=model.wubu_t_config; self.wubu_m_config=model.wubu_m_config
        self.timesteps=args.timesteps; self.betas=(linear_beta_schedule(args.timesteps,args.beta_start,args.beta_end) if args.beta_schedule=='linear' else cosine_beta_schedule(args.timesteps,args.cosine_s)).to(device); self.alphas=1.-self.betas; self.alphas_cumprod=torch.cumprod(self.alphas,axis=0); self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.-self.alphas_cumprod); self.sqrt_recip_alphas=torch.sqrt(1.0/(self.alphas+EPS)); self.alphas_cumprod_prev=F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0); self.posterior_variance=torch.clamp(self.betas*(1.-self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS),min=EPS*10); self.posterior_log_variance_clipped=torch.log(self.posterior_variance); self.posterior_mean_coef1=self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS); self.posterior_mean_coef2=(1.-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.-self.alphas_cumprod+EPS)
        self.scaler = amp.GradScaler(enabled=args.use_amp and device.type=='cuda'); self.global_step=0; self.current_epoch=0; self.best_val_loss=float('inf'); self.last_val_metrics:Dict[str,Any]={};
        if self.am_main_process: os.makedirs(args.checkpoint_dir,exist_ok=True)
        m_ref = self.model.module if ddp_active and isinstance(self.model, DDP) else self.model; self.encoder_final_manifold = PoincareBall(m_ref.encoder.wubu_s_final_curvature)
        self.lpips_loss_fn = None; self.ssim_metric = None
        if self.am_main_process and self.args.use_lpips_for_verification:
            if LPIPS_AVAILABLE and lpips is not None: self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device); current_logger.info("LPIPS metric enabled.")
            else: current_logger.warning("LPIPS lib not found/init failed. Skip LPIPS.")
        if self.am_main_process and TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
            try: self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device); current_logger.info("SSIM metric enabled.")
            except Exception as e: current_logger.warning(f"SSIM init failed: {e}. Skip SSIM."); self.ssim_metric = None
        elif self.am_main_process: current_logger.warning("torchmetrics SSIM not found/init failed. Skip SSIM.")

    def train_step(self, batch_video_frames: torch.Tensor):
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; num_cond = self.video_config["num_input_frames"]; num_pred = self.video_config["num_predict_frames"]; B = batch_video_frames.shape[0]; device = batch_video_frames.device; dtype = next(m_ref.parameters()).dtype
        cond_pixels = batch_video_frames[:, :num_cond, ...].to(device); target_pixels = batch_video_frames[:, num_cond : num_cond + num_pred, ...].to(device)
        with torch.no_grad(): cond_app_features, cond_gaad_bboxes, cond_motion_features = m_ref.encode_frames(cond_pixels); x0_regional_hyperbolic, target_gaad_bboxes, _ = m_ref.encode_frames(target_pixels); x0_regional_hyperbolic = x0_regional_hyperbolic.to(dtype)
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long); tangent_noise = torch.randn_like(x0_regional_hyperbolic, dtype=dtype)
        xt_regional_hyperbolic, actual_noise_tangent = q_sample_regional(x0_regional_hyperbolic, t, self.encoder_final_manifold, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, tangent_noise); xt_regional_tangent = self.encoder_final_manifold.logmap0(xt_regional_hyperbolic)
        is_unconditional = torch.rand(1).item() < self.args.cfg_unconditional_dropout_prob if self.args.cfg_unconditional_dropout_prob > 0 else False
        cond_app_for_model = None if is_unconditional else (cond_app_features.to(dtype) if cond_app_features is not None else None); cond_motion_for_model = None if is_unconditional else (cond_motion_features.to(dtype) if cond_motion_features is not None else None)
        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'):
            predicted_noise_tangent = self.model(xt_regional_tangent, t, cond_app_for_model, cond_motion_for_model, cfg_unconditional_flag=is_unconditional)
            if self.args.loss_type == 'noise_tangent': loss = F.mse_loss(predicted_noise_tangent, actual_noise_tangent)
            elif self.args.loss_type == 'pixel_reconstruction':
                sqrt_alpha_t=self.sqrt_alphas_cumprod.gather(0,t).view(B,1,1,1).to(device,dtype); sqrt_one_minus_alpha_t=self.sqrt_one_minus_alphas_cumprod.gather(0,t).view(B,1,1,1).to(device,dtype); x0_pred_tangent=(xt_regional_tangent-sqrt_one_minus_alpha_t*predicted_noise_tangent)/(sqrt_alpha_t+EPS); x0_pred_pixels=m_ref.decode_features(x0_pred_tangent, target_gaad_bboxes.to(dtype)); loss = F.mse_loss(x0_pred_pixels, target_pixels.to(dtype))
            else: raise ValueError(f"Unknown loss_type: {self.args.loss_type}")
        return loss, predicted_noise_tangent.detach(), actual_noise_tangent.detach()

    def train(self, start_epoch:int=0, initial_global_step:int=0):
        self.global_step = initial_global_step; self.current_epoch = start_epoch
        trainer_logger = logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Trainer") # Use a local logger instance
        if self.am_main_process:
            trainer_logger.info(f"Training from epoch {start_epoch}, step {initial_global_step}...")
        total_loss_interval = 0.0; items_interval = 0; current_cycle_loss_sum = 0.0; micro_batches_in_current_cycle = 0; avg_loss_for_q_cycle = 0.0; current_unclipped_grad_norm = 0.0
        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process:
                trainer_logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting...")
            if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler): self.train_loader.sampler.set_epoch(epoch)
            self.model.train(); total_micro_batches_estimate = None; dataset_len = 0
            try: dataset_len = len(self.train_loader.sampler) if hasattr(self.train_loader.sampler,'__len__') else (len(self.train_loader.dataset)//self.world_size if hasattr(self.train_loader.dataset,'__len__') and self.world_size>0 else 0) # type: ignore
            except Exception: trainer_logger.warning("Could not estimate epoch length for tqdm.", exc_info=False)
            if dataset_len > 0: total_micro_batches_estimate = math.ceil(dataset_len / (self.train_loader.batch_size or 1))
            prog_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=not self.am_main_process or os.getenv('CI')=='true', dynamic_ncols=True, total=total_micro_batches_estimate)
            for batch_idx, batch_frames_raw in enumerate(prog_bar):
                batch_frames = batch_frames_raw.to(self.device); is_last_batch = (batch_idx + 1) == total_micro_batches_estimate if total_micro_batches_estimate is not None else False; is_optimizer_step_time = ((micro_batches_in_current_cycle + 1) % self.args.grad_accum_steps == 0) or (is_last_batch and (micro_batches_in_current_cycle + 1) > 0); sync_context = self.model.no_sync() if self.ddp_active and isinstance(self.model, DDP) and not is_optimizer_step_time else contextlib.nullcontext() # type: ignore
                with sync_context:
                    with (torch.autograd.detect_anomaly() if self.args.detect_anomaly else contextlib.nullcontext()):
                        try: loss, _, _ = self.train_step(batch_frames)
                        except Exception as e_train: trainer_logger.error(f"R{self.rank} Train Step Error: {e_train}", exc_info=True); loss = None
                        if loss is None or torch.isnan(loss) or torch.isinf(loss): trainer_logger.warning(f"R{self.rank} S{self.global_step}: NaN/Inf/Error loss. Skip micro-batch."); loss_this_micro=torch.tensor(0.0, device=self.device); loss_for_backward=torch.tensor(0.0, device=self.device); skip_backward = True
                        else: loss_this_micro=loss; loss_for_backward=loss/self.args.grad_accum_steps; skip_backward=False
                        if not skip_backward:
                            try: self.scaler.scale(loss_for_backward).backward()
                            except Exception as e_bw: trainer_logger.error(f"R{self.rank} Backward Error: {e_bw}", exc_info=True)
                total_loss_interval += loss_this_micro.item() * batch_frames.size(0); items_interval += batch_frames.size(0); current_cycle_loss_sum += loss_this_micro.item(); micro_batches_in_current_cycle += 1
                if is_optimizer_step_time and micro_batches_in_current_cycle > 0:
                    self.scaler.unscale_(self.optimizer); current_unclipped_grad_norm=0.0; params_for_norm=[p for grp in self.optimizer.param_groups for p in grp['params'] if p.grad is not None and p.requires_grad]
                    if params_for_norm:
                        try: all_norms_sq=[torch.sum(p.grad.detach().float()**2) for p in params_for_norm]; finite_norms_sq=[n for n in all_norms_sq if torch.isfinite(n)]; current_unclipped_grad_norm=torch.sqrt(torch.stack(finite_norms_sq).sum()).item() if finite_norms_sq else float('inf')
                        except Exception as e_norm: trainer_logger.error(f"R{self.rank} S{self.global_step}: GradNorm calc error: {e_norm}"); current_unclipped_grad_norm=float('inf')
                    if self.args.global_max_grad_norm>0 and np.isfinite(current_unclipped_grad_norm) and current_unclipped_grad_norm > self.args.global_max_grad_norm: torch.nn.utils.clip_grad_norm_(params_for_norm, self.args.global_max_grad_norm)
                    if hasattr(self.optimizer, 'q_controller') and self.optimizer.q_controller:
                        q_ctrl=self.optimizer.q_controller; avg_loss_for_q_cycle=current_cycle_loss_sum/micro_batches_in_current_cycle; q_lr=self.optimizer.param_groups[0]['lr']; q_mom=self.optimizer.param_groups[0]['momentum']; q_state=q_ctrl.get_state(q_lr,q_mom,current_unclipped_grad_norm,avg_loss_for_q_cycle)
                        if q_ctrl.prev_state is not None and q_ctrl.prev_action is not None and q_ctrl.prev_loss is not None and q_state is not None: reward=q_ctrl.compute_reward(avg_loss_for_q_cycle,q_ctrl.prev_loss,current_unclipped_grad_norm); q_ctrl.update(q_ctrl.prev_state,q_ctrl.prev_action,reward,q_state) if np.isfinite(reward) else trainer_logger.warning(f"R{self.rank} S{self.global_step}: Q-Ctrl non-finite reward.")
                        q_ctrl.prev_state = q_state; q_ctrl.prev_action = q_ctrl.choose_action(q_state) if q_state is not None else q_ctrl.prev_action; q_ctrl.prev_loss = avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else q_ctrl.prev_loss
                    self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True); self.global_step += 1
                    if self.global_step % self.args.log_interval == 0 and self.am_main_process:
                        log_lr=self.optimizer.param_groups[0]['lr']; log_metrics={"train/loss_cycle_avg":avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle)else -1.0,"train/lr_effective":log_lr,"train/grad_norm_unclipped_for_q":current_unclipped_grad_norm if np.isfinite(current_unclipped_grad_norm)else -1.0,"epoch_frac":epoch+((batch_idx+1)/total_micro_batches_estimate if total_micro_batches_estimate and total_micro_batches_estimate > 0 else 0),"global_step":self.global_step}
                        if hasattr(self.optimizer,'q_controller') and self.optimizer.q_controller: log_metrics.update({f"q_ctrl/{k}":v for k,v in self.optimizer.get_q_controller_info().items()})
                        if hasattr(self.optimizer,'grad_stats'): log_metrics.update({f"grad_stats/{k}":v for k,v in self.optimizer.get_gradient_stats_summary().items()}); self.optimizer.grad_stats.reset()
                        log_metrics_flat={};[log_metrics_flat.update({f"{k}/{sk}":sv}) if isinstance(v,dict) else log_metrics_flat.update({k:v}) for k,v in log_metrics.items() for sk,sv in (v.items() if isinstance(v,dict) else [(None,v)]) if sk is not None or not isinstance(v,dict)]
                        trainer_logger.info(f"E {epoch+1} S{self.global_step} L(cyc){log_metrics_flat.get('train/loss_cycle_avg', -1.0):.4f} LR {log_metrics_flat.get('train/lr_effective', 0.0):.2e} GradN(Q){log_metrics_flat.get('train/grad_norm_unclipped_for_q', -1.0):.2f}")
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(log_metrics_flat, step=self.global_step)
                        total_loss_interval=0.0; items_interval=0
                    if self.args.save_interval>0 and self.global_step % self.args.save_interval == 0 and self.am_main_process: self._save_checkpoint(is_intermediate=True, metrics={"train_loss_cycle_avg":avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle)else -1.0})
                    current_cycle_loss_sum=0.0; micro_batches_in_current_cycle=0
            if self.am_main_process: avg_epoch_loss_val=total_loss_interval/items_interval if items_interval>0 else (avg_loss_for_q_cycle if micro_batches_in_current_cycle==0 and is_optimizer_step_time else float('nan')); trainer_logger.info(f"Epoch {epoch+1} finished. Approx avg loss: {avg_epoch_loss_val:.4f}")
            if self.val_loader and self.am_main_process:
                val_metrics_dict=self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if self.args.wandb and WANDB_AVAILABLE and wandb.run and val_metrics_dict: wandb.log({f"val/{k}":v for k,v in val_metrics_dict.items()}, step=self.global_step)
                current_val_primary_metric=val_metrics_dict.get(self.args.val_primary_metric, float('inf'))
                if current_val_primary_metric < self.best_val_loss: self.best_val_loss=current_val_primary_metric; self._save_checkpoint(is_best=True, metrics=val_metrics_dict)
            if self.am_main_process: save_metrics=self.last_val_metrics.copy() if self.last_val_metrics else {}; save_metrics["epoch_end_train_loss_logged_intervals_avg"]=avg_epoch_loss_val if np.isfinite(avg_epoch_loss_val) else -1.0; self._save_checkpoint(metrics=save_metrics)

    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Dict[str, float]:
        trainer_logger = logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Trainer")
        if not self.val_loader or not self.am_main_process: return {"avg_val_pixel_mse": float('inf')}
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; m_ref.eval(); total_mse_pixel = 0.0; total_psnr = 0.0; total_ssim = 0.0; total_lpips_val = 0.0; total_val_items = 0; logged_samples_count = 0; wandb_val_samples = []
        dtype_model = next(m_ref.parameters()).dtype
        for batch_idx, batch_frames_raw in enumerate(tqdm(self.val_loader, desc="Validating", dynamic_ncols=True, disable=os.getenv('CI')=='true' or not self.am_main_process)):
            batch_frames = batch_frames_raw.to(self.device); num_cond = self.video_config["num_input_frames"]; num_pred = self.video_config["num_predict_frames"]; cond_pixels = batch_frames[:, :num_cond, ...]; target_pixels_gt = batch_frames[:, num_cond : num_cond + num_pred, ...]; B_val = target_pixels_gt.shape[0]
            predicted_target_pixels = self.sample(conditioning_frames_pixels=cond_pixels, num_inference_steps=self.args.val_sampling_steps, sampler_type=self.args.val_sampler_type, ddim_eta=0.0, cfg_guidance_scale=self.args.val_cfg_scale, force_on_main_process=True, batch_size_if_uncond=B_val).to(dtype_model)
            pred_for_metrics = predicted_target_pixels[:,0,...]; gt_for_metrics = target_pixels_gt[:,0,...].to(dtype_model);
            pred_norm=(pred_for_metrics.clamp(-1,1)+1)/2.0; gt_norm=(gt_for_metrics.clamp(-1,1)+1)/2.0
            mse_pixel_val = F.mse_loss(pred_norm, gt_norm); total_mse_pixel += mse_pixel_val.item()*B_val; psnr_val = 10*torch.log10(1.0/(mse_pixel_val+EPS)) if mse_pixel_val > 0 else torch.tensor(100.0,device=self.device); total_psnr += psnr_val.item()*B_val
            if self.ssim_metric:
                try: ssim_val = self.ssim_metric(pred_norm, gt_norm); total_ssim += ssim_val.item()*B_val
                except Exception as e: trainer_logger.warning(f"SSIM failed: {e}")
            if self.lpips_loss_fn:
                try: lpips_val = self.lpips_loss_fn(pred_for_metrics, gt_for_metrics).mean(); total_lpips_val += lpips_val.item()*B_val
                except Exception as e: trainer_logger.warning(f"LPIPS failed: {e}")
            total_val_items += B_val
            if self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run and logged_samples_count < num_val_samples_to_log:
                num_log_batch = min(B_val, num_val_samples_to_log - logged_samples_count)
                for k in range(num_log_batch): wandb_val_samples.extend([wandb.Image(cond_pixels[k,i].cpu().float().clamp(-1,1)*0.5+0.5,caption=f"ValCond {i}") for i in range(cond_pixels.shape[1])]+[wandb.Image(pred_for_metrics[k].cpu().float().clamp(-1,1)*0.5+0.5,caption="ValPred"),wandb.Image(gt_for_metrics[k].cpu().float().clamp(-1,1)*0.5+0.5,caption="ValGT")])
                logged_samples_count += num_log_batch
        avg_mse=total_mse_pixel/total_val_items if total_val_items>0 else float('inf'); avg_psnr=total_psnr/total_val_items if total_val_items>0 else 0.0; avg_ssim=total_ssim/total_val_items if total_val_items>0 and self.ssim_metric else 0.0; avg_lpips=total_lpips_val/total_val_items if total_val_items>0 and self.lpips_loss_fn else 0.0
        metrics={"avg_val_pixel_mse":avg_mse,"avg_val_psnr":avg_psnr,"avg_val_ssim":avg_ssim,"avg_val_lpips":avg_lpips}; self.last_val_metrics=metrics; trainer_logger.info(f"Validation Metrics: MSE:{avg_mse:.4f}, PSNR:{avg_psnr:.2f}, SSIM:{avg_ssim:.4f}, LPIPS:{avg_lpips:.4f}")
        if wandb_val_samples and self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log({"validation_samples_sequence": wandb_val_samples}, step=self.global_step)
        return metrics

    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict]=None, is_best:bool=False):
        trainer_logger = logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Trainer")
        if not self.am_main_process: return
        m_save = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        data = {'global_step':self.global_step,'epoch':self.current_epoch,'model_state_dict':m_save.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),'scaler_state_dict':self.scaler.state_dict() if self.args.use_amp and self.device.type=='cuda' else None,'args':vars(self.args),'metrics':metrics if metrics else self.last_val_metrics,'video_config':self.video_config,'gaad_appearance_config':self.gaad_appearance_config,'gaad_motion_config':self.gaad_motion_config,'wubu_s_config':self.wubu_s_config,'wubu_t_config':self.wubu_t_config,'wubu_m_config':self.wubu_m_config,'transformer_noise_predictor_config':getattr(self.args,'transformer_noise_predictor_config',{})}
        fname_prefix="wuburegional_diff_ckpt_v0101_flow"; fpath=""
        if is_best: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_best.pt")
        elif is_intermediate: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_step{self.global_step}.pt")
        else: fpath=os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_ep{self.current_epoch+1}_step{self.global_step}.pt")
        try: torch.save(data,fpath); trainer_logger.info(f"Ckpt saved: {os.path.basename(fpath)}")
        except Exception as e: trainer_logger.error(f"Save ckpt error {fpath}: {e}",exc_info=True)

    def load_checkpoint(self, checkpoint_path:str) -> Tuple[int,int]:
        trainer_logger = logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Trainer")
        if not os.path.exists(checkpoint_path): trainer_logger.warning(f"Ckpt {checkpoint_path} not found."); return 0,0
        try: ckpt=torch.load(checkpoint_path,map_location=self.device); m_load=self.model.module if self.ddp_active and isinstance(self.model,DDP) else self.model; loaded_args=ckpt.get('args',{}); current_args=vars(self.args)
        except Exception as e_load: trainer_logger.error(f"Failed to load ckpt file {checkpoint_path}: {e_load}"); return 0,0
        try: m_load.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e_strict: trainer_logger.warning(f"Strict load failed: {e_strict}. Try non-strict."); m_load.load_state_dict(ckpt['model_state_dict'],strict=False)
        except Exception as e_load_state: trainer_logger.error(f"Error loading model state dict: {e_load_state}"); return 0,0
        if 'optimizer_state_dict' in ckpt and self.optimizer: self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt and self.scaler and ckpt['scaler_state_dict'] is not None: self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        loaded_global_step=ckpt.get('global_step',0); loaded_epoch=ckpt.get('epoch',0); self.best_val_loss=ckpt.get('metrics',{}).get(self.args.val_primary_metric,float('inf')) if 'metrics' in ckpt else float('inf'); trainer_logger.info(f"Loaded ckpt {checkpoint_path} (Step {loaded_global_step}, Ep {loaded_epoch}). BestValLoss: {self.best_val_loss:.4f}"); return loaded_global_step, loaded_epoch

    @torch.no_grad()
    def p_sample_ddpm(self, xt_regional_tangent: torch.Tensor, conditioning_app_features: Optional[torch.Tensor], conditioning_motion_features: Optional[torch.Tensor], t_tensor: torch.Tensor, t_int_val: int, cfg_guidance_scale: float = 1.0) -> torch.Tensor:
        m_ref=self.model.module if self.ddp_active and isinstance(self.model,DDP) else self.model; m_ref.eval(); B, N_pred, NumReg, D_tan=xt_regional_tangent.shape; shape_bc=(B,1,1,1); dev=xt_regional_tangent.device; dtype=xt_regional_tangent.dtype
        betas_t=self.betas.gather(0,t_tensor).view(shape_bc).to(dev,dtype); sqrt_one_minus_alphas_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(0,t_tensor).view(shape_bc).to(dev,dtype); sqrt_recip_alphas_t=self.sqrt_recip_alphas.gather(0,t_tensor).view(shape_bc).to(dev,dtype); post_logvar_t=self.posterior_log_variance_clipped.gather(0,t_tensor).view(shape_bc).to(dev,dtype)
        pred_noise_cond = self.model(xt_regional_tangent,t_tensor,conditioning_app_features,conditioning_motion_features,cfg_unconditional_flag=False)
        if cfg_guidance_scale > 1.0: pred_noise_uncond = self.model(xt_regional_tangent,t_tensor,None,None,cfg_unconditional_flag=True); pred_noise_tangent = pred_noise_uncond + cfg_guidance_scale*(pred_noise_cond-pred_noise_uncond)
        else: pred_noise_tangent = pred_noise_cond
        term_in_paren = xt_regional_tangent-(betas_t/(sqrt_one_minus_alphas_cumprod_t+EPS))*pred_noise_tangent; model_mean_tangent = sqrt_recip_alphas_t*term_in_paren
        if t_int_val == 0: return model_mean_tangent
        else: noise_sample=torch.randn_like(xt_regional_tangent); return model_mean_tangent+(0.5*post_logvar_t).exp()*noise_sample

    @torch.no_grad()
    def p_sample_ddim(self, xt_regional_tangent: torch.Tensor, conditioning_app_features: Optional[torch.Tensor], conditioning_motion_features: Optional[torch.Tensor], t_tensor: torch.Tensor, t_prev_tensor: torch.Tensor, eta: float = 0.0, cfg_guidance_scale: float = 1.0) -> torch.Tensor:
        m_ref=self.model.module if self.ddp_active and isinstance(self.model,DDP) else self.model; m_ref.eval(); B, N_pred, NumReg, D_tan=xt_regional_tangent.shape; shape_bc=(B,1,1,1); dev=xt_regional_tangent.device; dtype=xt_regional_tangent.dtype
        alphas_cumprod_t=self.alphas_cumprod.gather(0,t_tensor).view(shape_bc).to(dev,dtype); safe_t_prev=torch.clamp(t_prev_tensor,min=0); alphas_cumprod_t_prev=self.alphas_cumprod.gather(0,safe_t_prev).view(shape_bc).to(dev,dtype); alphas_cumprod_t_prev=torch.where(t_prev_tensor.view(shape_bc)<0,torch.ones_like(alphas_cumprod_t_prev),alphas_cumprod_t_prev)
        pred_noise_cond=self.model(xt_regional_tangent,t_tensor,conditioning_app_features,conditioning_motion_features,cfg_unconditional_flag=False)
        if cfg_guidance_scale > 1.0: pred_noise_uncond=self.model(xt_regional_tangent,t_tensor,None,None,cfg_unconditional_flag=True); pred_noise_tangent=pred_noise_uncond+cfg_guidance_scale*(pred_noise_cond-pred_noise_uncond)
        else: pred_noise_tangent=pred_noise_cond
        sqrt_one_minus_alphas_cumprod_t=torch.sqrt(torch.clamp(1.-alphas_cumprod_t,min=EPS)); sqrt_alphas_cumprod_t=torch.sqrt(alphas_cumprod_t); x0_pred_tangent=(xt_regional_tangent-sqrt_one_minus_alphas_cumprod_t*pred_noise_tangent)/(sqrt_alphas_cumprod_t+EPS)
        if self.args.ddim_x0_clip_val > 0: x0_pred_tangent=torch.clamp(x0_pred_tangent,-self.args.ddim_x0_clip_val,self.args.ddim_x0_clip_val)
        sigma_t_num=torch.clamp(1.-alphas_cumprod_t_prev,min=0.0); sigma_t_den=torch.clamp(1.-alphas_cumprod_t,min=EPS); sigma_t_ratio_alphacomp=torch.clamp(1.-alphas_cumprod_t/(alphas_cumprod_t_prev+EPS),min=0.0); sigma_t=eta*torch.sqrt((sigma_t_num/sigma_t_den)*sigma_t_ratio_alphacomp); sigma_t=torch.where(t_prev_tensor.view(shape_bc)<0,torch.zeros_like(sigma_t),sigma_t); pred_dir_xt=torch.sqrt(torch.clamp(1.-alphas_cumprod_t_prev-sigma_t**2,min=0.0))*pred_noise_tangent; xt_prev_tangent=torch.sqrt(alphas_cumprod_t_prev)*x0_pred_tangent+pred_dir_xt
        if eta > 0 and t_prev_tensor.min()>=0: xt_prev_tangent=xt_prev_tangent+sigma_t*torch.randn_like(xt_regional_tangent)
        return xt_prev_tangent

    @torch.no_grad()
    def sample(self, conditioning_frames_pixels: Optional[torch.Tensor], num_inference_steps: Optional[int] = None, sampler_type: str = "ddpm", ddim_eta: float = 0.0, cfg_guidance_scale: float = 1.0, force_on_main_process: bool = False, batch_size_if_uncond: int = 1) -> torch.Tensor:
        trainer_logger = logging.getLogger("WuBuGAADOpticalFlowDiffV0101.Trainer")
        if not (self.am_main_process or force_on_main_process): trainer_logger.warning(f"R{self.rank}: Sample on non-main, not forced. Skip."); return torch.empty(0, device=self.device)
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; m_ref.eval(); m_ref.to(self.device); B = conditioning_frames_pixels.shape[0] if conditioning_frames_pixels is not None else batch_size_if_uncond; dev = self.device; dtype = next(m_ref.parameters()).dtype; eff_steps = min(num_inference_steps if num_inference_steps is not None else self.timesteps, self.timesteps); num_pred=self.video_config["num_predict_frames"]; num_regions=self.gaad_appearance_config["num_regions"]; tangent_feat_dim=self.video_config['wubu_s_output_dim']
        cond_app, cond_motion, decoder_bboxes = None, None, None
        if conditioning_frames_pixels is not None:
            cond_app, cond_gaad_bboxes_app, cond_motion = m_ref.encode_frames(conditioning_frames_pixels.to(dev, dtype))
            decoder_bboxes = cond_gaad_bboxes_app[:, -1:, ...].repeat(1, num_pred, 1, 1) if cond_gaad_bboxes_app is not None else None
        if decoder_bboxes is None: frame_dims=(self.args.image_w,self.args.image_h); default_bboxes=golden_subdivide_rect_fixed_n(frame_dims, num_regions, device=dev, dtype=dtype, min_size_px=self.args.gaad_min_size_px); decoder_bboxes=default_bboxes.unsqueeze(0).unsqueeze(0).repeat(B,num_pred,1,1)
        xt_regional_tangent=torch.randn((B,num_pred,num_regions,tangent_feat_dim),device=dev,dtype=dtype)
        time_schedule=torch.linspace(self.timesteps-1,0,eff_steps,dtype=torch.long,device=dev)
        cond_str=f"CondFrames={conditioning_frames_pixels.shape[1]}" if conditioning_frames_pixels is not None else "UNCOND"; proc_id_str=f"R{self.rank}" if self.ddp_active else "Main"; trainer_logger.info(f"{proc_id_str}(Forced:{force_on_main_process}): Sampling {sampler_type.upper()}. BS={B}, {cond_str}, Steps={eff_steps}, Eta={ddim_eta if sampler_type=='ddim' else 'N/A'}, CFG={cfg_guidance_scale}")
        for i in tqdm(range(eff_steps), desc="Sampling", leave=False, dynamic_ncols=True, disable=not (self.am_main_process or force_on_main_process) or os.getenv('CI') == 'true'):
            t_idx=time_schedule[i]; t_batch=torch.full((B,), t_idx.item(), device=dev, dtype=torch.long)
            if sampler_type.lower()=="ddim": t_prev_idx=time_schedule[i+1] if i<eff_steps-1 else torch.tensor(-1,device=dev,dtype=torch.long); t_prev_batch=torch.full((B,), t_prev_idx.item(), device=dev, dtype=torch.long); xt_regional_tangent=self.p_sample_ddim(xt_regional_tangent, cond_app, cond_motion, t_batch, t_prev_batch, eta=ddim_eta, cfg_guidance_scale=cfg_guidance_scale)
            elif sampler_type.lower()=="ddpm": xt_regional_tangent=self.p_sample_ddpm(xt_regional_tangent, cond_app, cond_motion, t_batch, t_idx.item(), cfg_guidance_scale=cfg_guidance_scale)
            else: raise ValueError(f"Unknown sampler: {sampler_type}")
        predicted_pixels=m_ref.decode_features(xt_regional_tangent, decoder_bboxes.to(dtype)); trainer_logger.info(f"{proc_id_str}(Forced:{force_on_main_process}): Sampling finished."); return predicted_pixels
# =====================================================================
# Arg Parsing and Main Execution Logic
# =====================================================================
def seed_worker_init_fn(worker_id, base_seed, rank, world_size):
     worker_seed = base_seed + worker_id + rank * world_size
     random.seed(worker_seed); np.random.seed(worker_seed); torch.manual_seed(worker_seed)

def seed_everything(seed:int,rank:int=0,world_size:int=1):
    actual_seed = seed + rank; random.seed(actual_seed); np.random.seed(actual_seed); torch.manual_seed(actual_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(actual_seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu-GAAD Regional Latent Diffusion Model w/ Optical Flow (v0.10.1)")
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_dir")
    parser.add_argument('--local_rank', type=int, default=-1); parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2); parser.add_argument('--image_h', type=int, default=64)
    parser.add_argument('--image_w', type=int, default=64); parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_input_frames', type=int, default=3); parser.add_argument('--num_predict_frames', type=int, default=1)
    parser.add_argument('--frame_skip', type=int, default=1); parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--num_workers',type=int, default=0); parser.add_argument('--checkpoint_dir',type=str, default='wuburegional_diff_checkpoints_v0101_flow') # Updated dir
    parser.add_argument('--load_checkpoint', type=str, default=None); parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--wandb_project',type=str,default='WuBuGAADOpticalFlowDiffV0101') # Updated project
    parser.add_argument('--wandb_run_name',type=str,default=None); parser.add_argument('--log_interval',type=int, default=20)
    parser.add_argument('--save_interval',type=int, default=500)
    parser.add_argument('--gaad_num_regions', type=int, default=16); parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])
    parser.add_argument('--gaad_min_size_px', type=int, default=4)
    parser.add_argument('--use_wubu_motion_branch', action='store_true', help="Enable GAAD+WuBu-M+OpticalFlow motion branch.")
    parser.add_argument('--gaad_motion_num_regions', type=int, default=12)
    parser.add_argument('--gaad_motion_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="Spatial layout for motion regions.") # Content-aware was removed, flow magnitude used instead
    parser.add_argument('--encoder_use_roi_align', action='store_true'); parser.add_argument('--encoder_shallow_cnn_channels', type=int, default=32)
    parser.add_argument('--encoder_roi_align_output_h', type=int, default=4); parser.add_argument('--encoder_roi_align_output_w', type=int, default=4)
    parser.add_argument('--encoder_pixel_patch_size', type=int, default=16); parser.add_argument('--encoder_initial_tangent_dim', type=int, default=128)
    # --- Motion Encoder Optical Flow Parameters (NEW) ---
    parser.add_argument('--optical_flow_net_type', type=str, default='raft_small', choices=list(FLOW_MODELS.keys()) if OPTICAL_FLOW_AVAILABLE else [], help="Type of optical flow network from torchvision.")
    parser.add_argument('--freeze_flow_net', action='store_true', help="Freeze weights of the pre-trained optical flow network.")
    parser.add_argument('--flow_stats_components', nargs='+', type=str, default=['mag_mean', 'angle_mean'], help="Flow statistics to embed (mag_mean, angle_mean, mag_std, angle_std). 'angle_mean' uses cos/sin.")
    parser.add_argument('--decoder_type', type=str, default="patch_gen", choices=["patch_gen", "transformer"]); parser.add_argument('--decoder_patch_gen_size', type=int, default=16)
    parser.add_argument('--decoder_patch_resize_mode', type=str, default="bilinear", choices=["bilinear", "nearest"])
    parser.add_argument('--wubu_dropout', type=float, default=0.1)
    parser.add_argument('--wubu_s_num_levels', type=int, default=2); parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32]); parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8]); parser.add_argument('--wubu_s_use_rotation', action='store_true'); parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_m_num_levels', type=int, default=2); parser.add_argument('--wubu_m_hyperbolic_dims', nargs='+', type=int, default=[64,32]); parser.add_argument('--wubu_m_initial_curvatures', nargs='+', type=float, default=[1.0, 0.7]); parser.add_argument('--wubu_m_use_rotation', action='store_true'); parser.add_argument('--wubu_m_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_m_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_t_num_levels', type=int, default=2); parser.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[128,64]); parser.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[1.0,0.5]); parser.add_argument('--wubu_t_use_rotation', action='store_true'); parser.add_argument('--wubu_t_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_s_output_dim', type=int, default=32); parser.add_argument('--wubu_m_output_dim', type=int, default=32); parser.add_argument('--wubu_t_output_dim', type=int, default=128)
    parser.add_argument('--tnp_num_layers', type=int, default=4); parser.add_argument('--tnp_num_heads', type=int, default=8); parser.add_argument('--tnp_d_model', type=int, default=256); parser.add_argument('--tnp_d_ff_ratio', type=float, default=4.0); parser.add_argument('--tnp_dropout', type=float, default=0.1)
    parser.add_argument('--timesteps', type=int, default=100); parser.add_argument('--beta_schedule',type=str,default='cosine', choices=['linear','cosine']); parser.add_argument('--beta_start',type=float,default=1e-4); parser.add_argument('--beta_end',type=float,default=0.02); parser.add_argument('--cosine_s',type=float,default=0.008); parser.add_argument('--phi_time_diffusion_scale', type=float, default=1.0); parser.add_argument('--phi_time_base_freq', type=float, default=10000.0); parser.add_argument('--use_phi_frequency_scaling_for_time_emb', action='store_true'); parser.add_argument('--diffusion_time_embedding_dim', type=int, default=128); parser.add_argument('--loss_type', type=str, default='pixel_reconstruction', choices=['noise_tangent', 'pixel_reconstruction'])
    parser.add_argument('--learning_rate',type=float,default=5e-5); parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0); parser.add_argument('--global_max_grad_norm',type=float,default=1.0); parser.add_argument('--q_controller_enabled',action='store_true'); parser.add_argument('--grad_accum_steps',type=int, default=1); parser.add_argument('--use_amp', action='store_true'); parser.add_argument('--detect_anomaly',action='store_true')
    parser.add_argument('--cfg_unconditional_dropout_prob', type=float, default=0.1); parser.add_argument('--val_cfg_scale', type=float, default=1.5); parser.add_argument('--val_sampler_type', type=str, default="ddim", choices=["ddpm", "ddim"]); parser.add_argument('--val_sampling_steps', type=int, default=20); parser.add_argument('--ddim_x0_clip_val', type=float, default=1.0); parser.add_argument('--use_lpips_for_verification', action='store_true'); parser.add_argument('--validation_video_path', type=str, default=None); parser.add_argument('--validation_split_fraction', type=float, default=0.1); parser.add_argument('--val_primary_metric', type=str, default="avg_val_pixel_mse", choices=["avg_val_pixel_mse", "avg_val_psnr", "avg_val_ssim", "avg_val_lpips"]); parser.add_argument('--num_val_samples_to_log', type=int, default=2); parser.add_argument('--decoder_use_target_gt_bboxes_for_sampling', action='store_true')
    parser.add_argument('--demo_sampler_type', type=str, default="ddim", choices=["ddpm", "ddim"]); parser.add_argument('--demo_ddim_eta', type=float, default=0.0); parser.add_argument('--demo_cfg_scale', type=float, default=3.0); parser.add_argument('--demo_sampling_steps', type=int, default=50)
    parsed_args = parser.parse_args()
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
    if parsed_args.wubu_s_num_levels > 0 and parsed_args.wubu_s_hyperbolic_dims: parsed_args.wubu_s_output_dim = parsed_args.wubu_s_hyperbolic_dims[-1]
    else: parsed_args.wubu_s_output_dim = parsed_args.encoder_initial_tangent_dim
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.wubu_m_num_levels > 0 and parsed_args.wubu_m_hyperbolic_dims: parsed_args.wubu_m_output_dim = parsed_args.wubu_m_hyperbolic_dims[-1]
    else: parsed_args.wubu_m_output_dim = 0
    if parsed_args.wubu_t_num_levels > 0 and parsed_args.wubu_t_hyperbolic_dims: parsed_args.wubu_t_output_dim = parsed_args.wubu_t_hyperbolic_dims[-1]
    else: parsed_args.wubu_t_output_dim = 0
    parsed_args.transformer_noise_predictor_config = {"num_layers": parsed_args.tnp_num_layers, "num_heads": parsed_args.tnp_num_heads, "d_model": parsed_args.tnp_d_model, "d_ff_ratio": parsed_args.tnp_d_ff_ratio, "dropout": parsed_args.tnp_dropout, "activation": "gelu"}
    valid_stats = {'mag_mean', 'angle_mean', 'mag_std', 'angle_std'};
    if any(s not in valid_stats for s in parsed_args.flow_stats_components): parser.error(f"Invalid flow_stats_components. Allowed: {valid_stats}. Got: {parsed_args.flow_stats_components}")
    return parsed_args

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

def main():
    # --- Add imageio import ---
    try:
        import imageio
        IMAGEIO_AVAILABLE = True
    except ImportError:
        imageio = None
        IMAGEIO_AVAILABLE = False
        # Log this potential issue later if needed
    # --- End add imageio import ---

    if not OPTICAL_FLOW_AVAILABLE: # Early check if user intends to use motion branch
        temp_args_check, _ = argparse.ArgumentParser(add_help=False).parse_known_args() # Quick peek
        motion_branch_requested = False
        for arg_idx, arg_val in enumerate(sys.argv): # Robust check for the flag
            if arg_val == '--use_wubu_motion_branch': motion_branch_requested = True; break
            if arg_val.startswith('--use_wubu_motion_branch='):
                try: motion_branch_requested = str(sys.argv[arg_idx].split('=')[1]).lower() in ['true', '1', 'yes']
                except: pass; break
        if motion_branch_requested:
            print("FATAL ERROR: Motion branch (--use_wubu_motion_branch) requested, but torchvision.models.optical_flow unavailable. Install compatible torchvision or omit the flag.")
            sys.exit(1)

    args = parse_arguments()
    ddp_active = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    if ddp_active: rank=int(os.environ["RANK"]); local_rank=int(os.environ["LOCAL_RANK"]); world_size=int(os.environ["WORLD_SIZE"]); init_process_group(backend="nccl"); device=torch.device(f"cuda:{local_rank}"); torch.cuda.set_device(device)
    else: rank=0; local_rank=0; world_size=1; device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); _ = torch.cuda.set_device(device) if device.type=='cuda' else None

    am_main_process=(rank==0)
    # Ensure logger is configured per process to avoid duplicate handlers from previous runs if script is re-imported/re-run in same session
    current_logger_main = logging.getLogger("WuBuGAADOpticalFlowDiffV0101")
    if current_logger_main.hasHandlers():
        for handler in current_logger_main.handlers[:]: current_logger_main.removeHandler(handler)
        for handler in current_logger_main.root.handlers[:]: current_logger_main.root.removeHandler(handler) # Also clear root

    logging.basicConfig(level=logging.INFO if am_main_process else logging.WARNING, format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s', force=True)
    current_logger_main.info(f"--- WuBuGAADOpticalFlowDiffV0101 (R{rank}/{world_size},Dev {device},DDP:{ddp_active}) ---")
    seed_everything(args.seed,rank,world_size)
    if am_main_process: current_logger_main.info(f"Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_id=wandb.util.generate_id() if wandb.run is None else wandb.run.id
        wandb.init(project=args.wandb_project, name=args.wandb_run_name if args.wandb_run_name else f"wubuflow_v0101_{datetime.now().strftime('%y%m%d%H%M')}", config=vars(args), resume="allow", id=run_id)

    video_config = {"image_size":(args.image_h,args.image_w),"num_channels":args.num_channels,"num_input_frames":args.num_input_frames, "num_predict_frames":args.num_predict_frames,"wubu_s_output_dim":args.wubu_s_output_dim,"wubu_m_output_dim": args.wubu_m_output_dim,"wubu_t_output_dim":args.wubu_t_output_dim,}
    gaad_appearance_config = {"num_regions":args.gaad_num_regions,"decomposition_type":args.gaad_decomposition_type,"min_size_px": args.gaad_min_size_px,"phi_time_diffusion_scale": args.phi_time_diffusion_scale,"phi_time_base_freq": args.phi_time_base_freq}
    gaad_motion_config = {"num_regions": args.gaad_motion_num_regions,"decomposition_type":args.gaad_motion_decomposition_type,"min_size_px": args.gaad_min_size_px,} if args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE else None

    wubu_s_config = _configure_wubu_stack(args, "wubu_s")
    wubu_t_config = _configure_wubu_stack(args, "wubu_t")
    wubu_m_config = _configure_wubu_stack(args, "wubu_m")

    if am_main_process: current_logger_main.info(f"VideoCfg:{video_config}\nGAADAppCfg:{gaad_appearance_config}\nGAADMotCfg:{gaad_motion_config}\nWuBuS:{wubu_s_config}\nWuBuT:{wubu_t_config}\nWuBuM:{wubu_m_config}\nTNP:{args.transformer_noise_predictor_config}")

    model = GAADWuBuRegionalDiffNet(args, video_config, gaad_appearance_config, gaad_motion_config, wubu_s_config, wubu_t_config, wubu_m_config).to(device)
    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run: wandb.watch(model,log="all",log_freq=max(100,args.log_interval*5),log_graph=False)
    if ddp_active: model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    q_cfg = DEFAULT_CONFIG_QLEARN_DIFFUSION.copy() if args.q_controller_enabled else None
    optimizer = RiemannianEnhancedSGD(model.parameters(),lr=args.learning_rate,q_learning_config=q_cfg,max_grad_norm_risgd=args.risgd_max_grad_norm)

    actual_video_path = args.video_data_path; demo_file_name = "dummy_video_v0101_flow.mp4"
    if "demo_video_data" in args.video_data_path: actual_video_path = os.path.join(args.video_data_path, demo_file_name)

    # <<<--- START REPLACEMENT: Dummy Video Creation --- >>>
    if "demo_video_data" in args.video_data_path and am_main_process:
        os.makedirs(args.video_data_path, exist_ok=True)
        if not os.path.exists(actual_video_path):
            if IMAGEIO_AVAILABLE and imageio is not None:
                current_logger_main.info(f"Creating dummy video using imageio: {actual_video_path}...")
                min_raw_frames = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                num_dummy = max(50, min_raw_frames + 20)
                dummy_h = int(args.image_h)
                dummy_w = int(args.image_w)
                # Create random frames directly as HWC uint8 numpy arrays
                frames_np_list = [np.random.randint(0, 256, (dummy_h, dummy_w, args.num_channels), dtype=np.uint8) for _ in range(num_dummy)]
                current_fps = int(10)

                try:
                    imageio.mimwrite(actual_video_path, frames_np_list, fps=current_fps) # Use imageio.mimwrite
                    current_logger_main.info(f"Successfully created dummy video using imageio: {actual_video_path} with {len(frames_np_list)} frames at {current_fps} FPS.")
                except Exception as e_imageio_write:
                    current_logger_main.error(f"Error creating dummy video using imageio: {e_imageio_write}", exc_info=True)

            # Fallback message if imageio isn't available (torchvision already failed)
            elif not IMAGEIO_AVAILABLE:
                 current_logger_main.error("imageio library not found. Cannot create dummy video. Please install imageio and imageio-ffmpeg.")
            elif not VIDEO_IO_AVAILABLE:
                 current_logger_main.error("Neither torchvision.io nor imageio are available. Cannot create dummy video.")
            # Keep the original error message if imageio is available but torchvision wasn't
            elif VIDEO_IO_AVAILABLE and not imageio:
                 current_logger_main.error("torchvision.io failed and imageio is not installed. Cannot create dummy video.")

        elif os.path.exists(actual_video_path) and am_main_process:
             current_logger_main.info(f"Dummy video {actual_video_path} already exists.")
    # <<<--- END REPLACEMENT --- >>>

    if ddp_active: torch.distributed.barrier()

    if not os.path.isfile(actual_video_path):
        current_logger_main.error(f"Video path '{actual_video_path}' is not a file or does not exist. This could be due to a failure in creating the dummy video. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    total_frames_sample = args.num_input_frames + args.num_predict_frames; full_dataset = None
    try: full_dataset = VideoFrameDataset(video_path=actual_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
    except Exception as e: current_logger_main.error(f"Failed init main Dataset from {actual_video_path}: {e}", exc_info=True); sys.exit(1)
    if not full_dataset or len(full_dataset) == 0 : current_logger_main.error("Main dataset empty/failed load. Check path/content. Exit."); sys.exit(1)

    train_dataset, val_dataset = full_dataset, None
    if args.validation_video_path and os.path.exists(args.validation_video_path) and os.path.isfile(args.validation_video_path):
        try:
            val_dataset = VideoFrameDataset(video_path=args.validation_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
            if len(val_dataset) > 0: current_logger_main.info(f"Using separate val video: {args.validation_video_path}, {len(val_dataset)} samples.")
            else: current_logger_main.warning(f"Validation video {args.validation_video_path} loaded 0 samples, using split instead if applicable."); val_dataset = None
        except Exception as e: current_logger_main.warning(f"Could not load val dataset {args.validation_video_path}: {e}. Using split if applicable."); val_dataset = None

    if val_dataset is None and args.validation_split_fraction > 0 and len(full_dataset) > 10 :
        num_val = int(len(full_dataset) * args.validation_split_fraction); num_train = len(full_dataset) - num_val
        if num_train > 0 and num_val > 0: train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(args.seed + rank)); current_logger_main.info(f"Split main dataset: {len(train_dataset)} train, {len(val_dataset)} val.")
        else: current_logger_main.warning(f"Not enough samples ({len(full_dataset)}) for val split."); val_dataset = None

    partial_seed_worker = functools.partial(seed_worker_init_fn,base_seed=args.seed,rank=rank,world_size=world_size)
    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True,seed=args.seed) if ddp_active else None
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=(train_sampler is None),num_workers=args.num_workers,sampler=train_sampler,pin_memory=(device.type=='cuda'),worker_init_fn=partial_seed_worker if args.num_workers>0 else None,drop_last=True)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=(device.type=='cuda'), drop_last=False, worker_init_fn=partial_seed_worker if args.num_workers > 0 else None)
    elif am_main_process: current_logger_main.info("No validation dataset/loader configured or empty.")

    trainer = DiffusionTrainer(model,optimizer,device,train_loader,val_loader,args,rank,world_size,ddp_active)
    start_global_step,start_epoch=0,0
    if args.load_checkpoint: start_global_step,start_epoch=trainer.load_checkpoint(args.load_checkpoint)
    try: trainer.train(start_epoch=start_epoch,initial_global_step=start_global_step)
    except KeyboardInterrupt: current_logger_main.info(f"Rank {rank}: Training interrupted.")
    except Exception as e: current_logger_main.error(f"Rank {rank}: Training loop crashed: {e}",exc_info=True)
    finally:
        if am_main_process:
            current_logger_main.info("Finalizing run...")
            trainer._save_checkpoint(metrics=trainer.last_val_metrics if trainer.last_val_metrics else {})
            if args.epochs>0 and hasattr(trainer,'sample') and trainer.global_step > 0 and hasattr(train_loader, '__len__') and len(train_loader)>0: # Check len of train_loader
                current_logger_main.info("DEMO SAMPLING...")
                try: demo_batch = next(iter(train_loader)); demo_cond_pixels = demo_batch[:, :args.num_input_frames, ...].to(device)[0:1]
                except StopIteration: current_logger_main.warning("Demo sampling skipped: DataLoader empty."); demo_cond_pixels=None
                except Exception as e_demo_batch: current_logger_main.error(f"Demo batch error: {e_demo_batch}"); demo_cond_pixels=None
                if demo_cond_pixels is not None:
                    try:
                        pred_pixels = trainer.sample(demo_cond_pixels,num_inference_steps=args.demo_sampling_steps,sampler_type=args.demo_sampler_type,ddim_eta=args.demo_ddim_eta,cfg_guidance_scale=args.demo_cfg_scale,force_on_main_process=True)
                        current_logger_main.info(f"Demo predicted pixels shape: {pred_pixels.shape}")
                        if pred_pixels.numel() > 0 and pred_pixels.shape[0] > 0:
                            save_dir = os.path.join(args.checkpoint_dir, "demo_samples_v0101_flow"); os.makedirs(save_dir, exist_ok=True)
                            for i in range(min(args.num_input_frames, demo_cond_pixels.shape[1])): save_image(demo_cond_pixels[0, i].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_dir, f"demo_cond_{i}.png"))
                            for i in range(pred_pixels.shape[1]): save_image(pred_pixels[0,i].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_dir, f"demo_pred_{i}.png"))
                            current_logger_main.info(f"Saved demo sample frames to {save_dir}")
                            if args.wandb and WANDB_AVAILABLE and wandb.run:
                                wb_imgs=[wandb.Image(demo_cond_pixels[0,i].cpu().float().clamp(-1,1)*0.5+0.5,caption=f"Cond {i}") for i in range(min(args.num_input_frames, demo_cond_pixels.shape[1]))] + [wandb.Image(pred_pixels[0,i].cpu().float().clamp(-1,1)*0.5+0.5,caption=f"Pred {i}") for i in range(pred_pixels.shape[1])]
                                wandb.log({"demo_sequence_final": wb_imgs}, step=trainer.global_step)
                    except Exception as e_demo: current_logger_main.error(f"Demo sampling error: {e_demo}",exc_info=True)
            if args.wandb and WANDB_AVAILABLE and wandb.run: wandb.finish()
        if ddp_active and is_initialized(): destroy_process_group()
        current_logger_main.info(f"Rank {rank}: WuBuGAADOpticalFlowDiffNet (v0.10.1) script finished.")



if __name__ == "__main__":
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