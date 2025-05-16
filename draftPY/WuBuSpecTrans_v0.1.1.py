# WuBuSpecTrans_v0.1.1.py
# VAE-GAN Hybrid Model for 1-Second Audio Segment Synthesis
# Operates on GAAD-defined regional DCT coefficients of Mel Spectrograms with WuBu nesting.
# LAST UPDATE: Refactored from WuBuGAADHybridGen_v0.1 (video) to WuBuSpecTrans_v0.1.1 (audio, 1s segments)

# =====================================================================
# Python Imports and Setup
# =====================================================================
import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SubsetRandomSampler
import numpy as np
import soundfile as sf
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
from pathlib import Path # Added Path import
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import spectral_norm
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from tqdm import tqdm

import torchvision.transforms as T # Still useful for image-like Mel spectrograms
import torchvision.transforms.functional as TF
from PIL import Image # For saving Mel spectrograms as images if needed

import librosa # For audio processing and Mel spectrograms
import librosa.display # For visualizing spectrograms
try:
    from torch_dct import dct_2d, idct_2d # For 2D DCT
    TORCH_DCT_AVAILABLE = True
except ImportError:
    dct_2d, idct_2d = None, None
    TORCH_DCT_AVAILABLE = False
    print("CRITICAL WARNING: torch-dct library not found. DCT/IDCT operations will be placeholders or fail. Install with 'pip install torch-dct'.")


try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None
    IMAGEIO_AVAILABLE = False
    print("Warn: imageio unavailable (used for dummy audio generation).")


from torchvision.utils import save_image # Can save Mel spectrograms as images

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure # Could be used on Mel spectrograms
    TORCHMETRICS_SSIM_AVAILABLE = True
except ImportError:
    StructuralSimilarityIndexMeasure = None
    TORCHMETRICS_SSIM_AVAILABLE = False

try:
    import lpips # Could be used on Mel spectrograms
    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False

# Setup logging
logger = logging.getLogger("WuBuSpecTransV01") # Renamed logger
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
    "boundary_points_per_level": [0], # Default to 0 for audio DCT features
    "tangent_input_combination_dims": [32],
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
    "q_learning_rate": 0.01,
    "discount_factor": 0.90,
    "epsilon_start": 0.5,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9995,
    "lr_scale_options": [0.8, 0.9, 1.0, 1.1, 1.2],
    "momentum_scale_options": [0.95, 0.98, 1.0, 1.01, 1.02],
    "max_q_table_size": 20000,
    "state_history_len": 5,
    "reward_clipping": (-2.0, 2.0),
    "q_value_clipping": (-30.0, 30.0)
}

# =====================================================================
# Geometric, Optimizer, WuBu Core Components
# =====================================================================
class HyperbolicUtils:
    @staticmethod
    def poincare_clip(x: torch.Tensor, c_scalar: float, radius: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_input_dtype = x.dtype
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            x_compute = x.float()
            x_compute = torch.nan_to_num(x_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(x_compute).all() else x_compute
            if original_input_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                x_compute = torch.clamp(x_compute, min=-f16_max, max=f16_max)
            return x_compute.to(original_input_dtype)
        sqrt_c = math.sqrt(c_scalar + eps)
        effective_radius_factor = min(radius, 1.0 - eps)
        max_norm_val_f32 = effective_radius_factor / sqrt_c
        x_compute = x.float()
        x_compute = torch.nan_to_num(x_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(x_compute).all() else x_compute
        x_norm_sq = torch.sum(x_compute.pow(2), dim=-1, keepdim=True)
        sqrt_input_val = torch.clamp(x_norm_sq, min=0.0) + eps
        sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=1.0, neginf=eps)
        sqrt_input_val.clamp_min_(eps)
        norm = torch.sqrt(sqrt_input_val)
        cond = norm > max_norm_val_f32
        norm_plus_eps_for_div = norm + eps
        norm_plus_eps_for_div.clamp_min_(eps)
        scale_factor = torch.where(cond, max_norm_val_f32 / norm_plus_eps_for_div, torch.ones_like(norm))
        clipped_x_f32 = x_compute * scale_factor
        if original_input_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            clipped_x_f32 = torch.clamp(clipped_x_f32, min=-f16_max, max=f16_max)
        final_clipped_x = clipped_x_f32.to(original_input_dtype)
        return torch.nan_to_num(final_clipped_x,nan=0.0,posinf=float(max_norm_val_f32),neginf=-float(max_norm_val_f32)) if not torch.isfinite(final_clipped_x).all() else final_clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = v.dtype
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            v_compute = v.float()
            v_compute = torch.nan_to_num(v_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(v_compute).all() else v_compute
            if original_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                v_compute = torch.clamp(v_compute, min=-f16_max, max=f16_max)
            return v_compute.to(original_dtype)
        v_compute = v.float()
        v_compute = torch.nan_to_num(v_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(v_compute).all() else v_compute
        v_norm_sq_unclamped = torch.sum(v_compute.pow(2), dim=-1, keepdim=True)
        v_norm_sq_clamped = torch.clamp(v_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)
        sqrt_input_val = v_norm_sq_clamped + eps
        sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps)
        sqrt_input_val.clamp_min_(eps)
        v_norm = torch.sqrt(sqrt_input_val)
        if not torch.isfinite(v_norm).all():
            return HyperbolicUtils.poincare_clip(torch.zeros_like(v_compute), c_scalar, eps=eps).to(original_dtype)
        sqrt_c_val = math.sqrt(c_scalar + eps)
        scaled_radius_arg = float(scale_scalar) * sqrt_c_val * v_norm
        tanh_input_val = torch.clamp(scaled_radius_arg, min=-30.0, max=30.0)
        tanh_term_val = torch.tanh(tanh_input_val)
        denominator_lambda_candidate = sqrt_c_val * v_norm + eps
        denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps)
        lambda_v_val = torch.where(v_norm > eps, tanh_term_val / denominator_lambda_val, torch.full_like(v_norm, float(scale_scalar), dtype=torch.float32))
        mapped_v_f32 = lambda_v_val * v_compute
        if not torch.isfinite(mapped_v_f32).all():
            mapped_v_f32 = torch.zeros_like(v_compute)
        clipped_mapped_v_f32 = HyperbolicUtils.poincare_clip(mapped_v_f32, c_scalar, eps=eps)
        final_result = clipped_mapped_v_f32
        if original_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            final_result = torch.clamp(clipped_mapped_v_f32, min=-f16_max, max=f16_max)
        return final_result.to(original_dtype)

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = y.dtype
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            y_compute = y.float()
            y_compute = torch.nan_to_num(y_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(y_compute).all() else y_compute
            if original_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                y_compute = torch.clamp(y_compute, min=-f16_max, max=f16_max)
            return y_compute.to(original_dtype)
        y_clipped_original_dtype = HyperbolicUtils.poincare_clip(y, c_scalar, eps=eps)
        y_compute = y_clipped_original_dtype.float()
        y_compute = torch.nan_to_num(y_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(y_compute).all() else y_compute
        y_norm_sq_unclamped = torch.sum(y_compute.pow(2), dim=-1, keepdim=True)
        y_norm_sq_clamped = torch.clamp(y_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)
        sqrt_input_val = y_norm_sq_clamped + eps
        sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps)
        sqrt_input_val.clamp_min_(eps)
        y_norm = torch.sqrt(sqrt_input_val)
        if not torch.isfinite(y_norm).all():
            return torch.zeros_like(y, dtype=original_dtype)
        sqrt_c_val = math.sqrt(c_scalar + eps)
        arctanh_arg_raw = sqrt_c_val * y_norm
        arctanh_arg_clamped = torch.clamp(arctanh_arg_raw, min=-1.0 + eps*10, max=1.0 - eps*10)
        atanh_term_val = torch.atanh(arctanh_arg_clamped)
        denominator_lambda_candidate = float(scale_scalar) * sqrt_c_val * y_norm + eps
        denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps)
        default_lambda_y_val = 1.0 / max(float(scale_scalar), eps)
        lambda_y_val = torch.where(y_norm > eps, atanh_term_val / denominator_lambda_val, torch.full_like(y_norm, default_lambda_y_val, dtype=torch.float32))
        mapped_y_f32 = lambda_y_val * y_compute
        if not torch.isfinite(mapped_y_f32).all():
            mapped_y_f32 = torch.zeros_like(y_compute)
        final_result = mapped_y_f32
        if original_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            final_result = torch.clamp(mapped_y_f32, min=-f16_max, max=f16_max)
        return final_result.to(original_dtype)

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
        super().__init__(c_scalar)
        self.logger = logging.getLogger("WuBuSpecTransV01.PoincareBall")
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            self.c = 0.0; self.k = 0.; self.sqrt_c = 0.; self.radius = float('inf')
        else:
            self.c = c_scalar; self.k = -self.c; self.sqrt_c = math.sqrt(self.c); self.radius = 1. / self.sqrt_c
        self.max_norm = self.radius * (1. - EPS * 10) if self.c > 0 and self.radius != float('inf') else float('inf')
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
        if self.c <= 0: return dp
        p_projected = self.proju(p)
        p_norm_sq = torch.sum(p_projected.pow(2), dim=-1, keepdim=True)
        if self.radius == float('inf'):
            max_sq_norm_val = float('inf')
        else:
            max_sq_norm_val = self.max_norm**2
        p_norm_sq_clamped = torch.clamp(p_norm_sq, min=0.0, max=max_sq_norm_val)
        term_inside_paren = 1. - self.c * p_norm_sq_clamped
        lambda_p_factor = term_inside_paren / 2.0
        riemannian_scaling_factor = lambda_p_factor.pow(2)
        final_factor = torch.clamp(riemannian_scaling_factor, min=EPS)
        r_grad = final_factor * dp
        if not torch.isfinite(r_grad).all():
            dp_norm_str = dp.norm().item() if torch.isfinite(dp).all() else 'NaN'
            p_norm_sq_str = p_norm_sq.mean().item() if p_norm_sq.numel()>0 and torch.isfinite(p_norm_sq).all() else 'NaN'
            p_proj_norm_str = p_projected.norm().item() if torch.isfinite(p_projected).all() else 'NaN'
            factor_str = final_factor.mean().item() if final_factor.numel()>0 else 'N/A'
            self.logger.warning(f"Non-finite Riemannian gradient computed in egrad2rgrad for param shape {p.shape}, c={self.c}. Factor: {factor_str}, dp_norm: {dp_norm_str}. Input p_norm_sq: {p_norm_sq_str}. Projected p norm: {p_proj_norm_str}")
            return dp
        return r_grad
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad():
            w.data.uniform_(-irange, irange)
            if self.c > 0 :
                w.data = self.expmap0(w.data)
                w.data = self.proju(w.data)

def init_weights_general(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', False)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d, nn.Conv3d, nn.ConvTranspose3d)):
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
        self.current_manifold_c = initial_manifold_c
        if num_points > 0 and point_dim > 0:
            self.hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim))
            PoincareBall(initial_manifold_c).init_weights(self.hyperbolic_points_params, irange=1e-3)
            setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(initial_manifold_c))
        else:
            self.register_parameter('hyperbolic_points_params', None)
    def set_current_manifold_c(self, c_scalar: float):
        self.current_manifold_c = c_scalar
        if self.hyperbolic_points_params is not None:
            setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(c_scalar))
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
        current_logger=logging.getLogger("WuBuSpecTransV01.HILT")
        if self.use_rotation and self.in_dim > 0:
            if self.in_dim == 4 and self.phi_influence_rotation_init:
                self.rot_axis_param = nn.Parameter(torch.randn(3))
                self.rot_angle_unconstrained = nn.Parameter(torch.tensor(0.0))
                self.phi_angle_scale = PHI**(level_idx_for_phi % 5 - 2) * (math.pi / 4)
                current_logger.info(f"L{level_idx_for_phi} (4D): Quat rot. Scale: {self.phi_angle_scale:.3f}")
            elif self.in_dim == 2 and self.phi_influence_rotation_init:
                self.rot_angle_unconstrained_2d = nn.Parameter(torch.tensor(0.0))
                self.phi_angle_scale_2d = PHI**(level_idx_for_phi % 3) * (math.pi / 3)
                current_logger.info(f"L{level_idx_for_phi} (2D): SO(2) rot. Scale: {self.phi_angle_scale_2d:.3f}")
            else:
                self.rotation_module = nn.Linear(self.in_dim, self.in_dim, bias=False)
                if self.in_dim > 0:
                    nn.init.eye_(self.rotation_module.weight)
        mlp_hidden_dim = hidden_dim if hidden_dim and hidden_dim > 0 else max(16, (in_dim + out_dim) // 2)
        if self.transform_type == 'mlp' and all(d > 0 for d in [in_dim, out_dim, mlp_hidden_dim]):
            self.non_rotational_map = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden_dim), nn.LayerNorm(mlp_hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim, out_dim)
            )
        elif self.transform_type == 'linear' and in_dim > 0 and out_dim > 0:
            self.non_rotational_map = nn.Linear(in_dim, out_dim)
        else:
            self.non_rotational_map = nn.Identity()
            current_logger.info(f"L{level_idx_for_phi}: Using Identity transform as in_dim={in_dim} or out_dim={out_dim} or hidden_dim={mlp_hidden_dim} is non-positive.")
        self.apply(init_weights_general)
    def _apply_rotation(self, x_tan: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_tan is None or not self.use_rotation: return x_tan
        B_maybe = x_tan.shape[0] if x_tan.dim() > 1 else 1
        if self.in_dim == 4 and self.phi_influence_rotation_init and hasattr(self, 'rot_axis_param'):
            angle = F.softplus(self.rot_angle_unconstrained) * self.phi_angle_scale
            current_axis = self.rot_axis_param.to(x_tan.device).unsqueeze(0).expand(B_maybe, -1)
            angle_b = angle.unsqueeze(0).expand(B_maybe, 1)
            q_rot = quaternion_from_axis_angle(angle_b, current_axis)
            return quaternion_apply_to_vector(q_rot, x_tan)
        elif self.in_dim == 2 and self.phi_influence_rotation_init and hasattr(self, 'rot_angle_unconstrained_2d'):
            angle_2d = F.softplus(self.rot_angle_unconstrained_2d) * self.phi_angle_scale_2d
            cos_a = torch.cos(angle_2d); sin_a = torch.sin(angle_2d)
            x_comp = x_tan[..., 0]; y_comp = x_tan[..., 1]
            x_rot = x_comp * cos_a - y_comp * sin_a
            y_rot = x_comp * sin_a + y_comp * cos_a
            return torch.stack([x_rot, y_rot], dim=-1)
        return self.rotation_module(x_tan) if self.rotation_module else x_tan
    def forward(self, point_in: torch.Tensor, boundaries_in: Optional[torch.Tensor],
                descriptor_in: Optional[torch.Tensor], current_c_in: float, current_c_out: float,
                current_s_in: Optional[float]=None, current_s_out: Optional[float]=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        m_in, m_out = PoincareBall(current_c_in), PoincareBall(current_c_out)
        tan_main = m_in.logmap0(point_in)
        tan_bound = m_in.logmap0(boundaries_in) if boundaries_in is not None else None
        tan_desc = m_in.logmap0(descriptor_in) if descriptor_in is not None else None
        tan_main_rot = self._apply_rotation(tan_main)
        tan_bound_rot = self._apply_rotation(tan_bound)
        tan_desc_rot = self._apply_rotation(tan_desc)
        def apply_map_and_clamp(tan_vec):
            if tan_vec is not None:
                return torch.clamp(self.non_rotational_map(tan_vec), -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
            return None
        tan_main_out_clamped = apply_map_and_clamp(tan_main_rot)
        tan_bound_out_clamped = apply_map_and_clamp(tan_bound_rot)
        tan_desc_out_clamped = apply_map_and_clamp(tan_desc_rot)
        default_out_shape = (point_in.shape[0], self.out_dim) if point_in.dim() > 1 else (self.out_dim,)
        if tan_main_out_clamped is not None:
            expmap_main_out = m_out.expmap0(tan_main_out_clamped)
        else:
            expmap_main_out = m_out.expmap0(torch.zeros(default_out_shape, device=point_in.device, dtype=point_in.dtype))
        expmap_bound_out = m_out.expmap0(tan_bound_out_clamped) if tan_bound_out_clamped is not None else None
        expmap_desc_out = m_out.expmap0(tan_desc_out_clamped) if tan_desc_out_clamped is not None else None
        return (expmap_main_out, expmap_bound_out, expmap_desc_out)

class HyperbolicWuBuNestingLevel(nn.Module):
    def __init__(self, level_idx: int, dim: int, config: Dict, initial_curvature_val_base: float):
        super().__init__()
        self.level_idx, self.dim, self.config = level_idx, dim, config
        self.logger = logging.getLogger(f"WuBuSpecTransV01.Level{self.level_idx}")
        current_logger = self.logger
        self.phi_influence_curvature = config.get("phi_influence_curvature", False)
        self.initial_curvature_val = initial_curvature_val_base * (PHI**(level_idx % 4 - 1.5) if self.phi_influence_curvature else 1.0)
        phi_base_str = f" (PhiBase {initial_curvature_val_base:.2f})" if self.phi_influence_curvature else ""
        current_logger.info(f"InitialC={self.initial_curvature_val:.2f}{phi_base_str}")
        self.use_ld = config.get("use_level_descriptors", True)
        self.use_spread = config.get("use_level_spread", True)
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
            else: raise ValueError(f"Unknown init_type: {init_type}")
            if learn_flag: setattr(self, param_name, nn.Parameter(unconstrained_val))
            else: self.register_buffer(param_name, unconstrained_val)
        if self.use_ld and self.dim > 0:
            self.level_descriptor_param = nn.Parameter(torch.Tensor(dim))
            PoincareBall(c_scalar=self.initial_curvature_val).init_weights(self.level_descriptor_param, irange=self.ld_init_scale)
            setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=self.initial_curvature_val))
        else:
            self.register_parameter('level_descriptor_param', None)
        num_bounds_list = config.get("boundary_points_per_level", [0])
        num_boundaries_val = num_bounds_list[level_idx] if level_idx < len(num_bounds_list) else (num_bounds_list[-1] if num_bounds_list else 0)
        if self.dim > 0 and num_boundaries_val > 0:
            self.boundary_manifold_module = BoundaryManifoldHyperbolic(level_idx, num_boundaries_val, dim, initial_manifold_c=self.initial_curvature_val)
        else:
            self.boundary_manifold_module = None
        self.comb_in_dim = self.dim
        if self.relative_vector_aggregation not in ['none', None] and num_boundaries_val > 0:
            self.comb_in_dim += self.dim
        if self.use_ld:
            self.comb_in_dim += self.dim
        comb_h_dims_cfg = config.get("tangent_input_combination_dims", [max(16, self.comb_in_dim // 2)]) if self.comb_in_dim > 0 else []
        comb_h_dims = comb_h_dims_cfg if isinstance(comb_h_dims_cfg, list) else [comb_h_dims_cfg]
        layers = []; in_d = self.comb_in_dim
        if self.dim > 0 and self.comb_in_dim > 0:
            for h_d in comb_h_dims:
                if in_d > 0 and h_d > 0:
                    layers.extend([nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(self.dropout_rate)])
                    in_d = h_d
            if in_d > 0 and self.dim > 0:
                layers.append(nn.Linear(in_d, self.dim))
                layers.append(nn.LayerNorm(self.dim))
        self.tangent_combiner = nn.Sequential(*layers) if layers else nn.Identity()
        self.use_flow = config.get("use_tangent_flow", True); self.tangent_flow_module = None; self.flow_scale_val = 0.0
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
        return MIN_WUBU_LEVEL_SCALE
    def get_current_spread_scalar_tensor(self) -> torch.Tensor:
        if self.use_spread and hasattr(self, 'log_spread_unconstrained') and self.log_spread_unconstrained is not None:
            return get_constrained_param_val(self.log_spread_unconstrained, self.min_spread)
        ref_param = next(iter(self.parameters()), None)
        if ref_param is None and isinstance(self.tangent_combiner, nn.Sequential) and list(self.tangent_combiner.parameters()):
            ref_param = next(iter(self.tangent_combiner.parameters()), None)
        ref_device = ref_param.device if ref_param is not None else torch.device('cpu')
        ref_dtype = ref_param.dtype if ref_param is not None else torch.float
        return torch.tensor(self.min_spread, device=ref_device, dtype=ref_dtype)
    def forward(self, point_in_hyperbolic: torch.Tensor, relative_vectors_tangent_in: Optional[torch.Tensor],
                descriptor_point_in_hyperbolic: Optional[torch.Tensor], sigma_in_scalar_tensor: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if point_in_hyperbolic.dim() != 2:
            raise ValueError(f"WuBuLevel forward expects 2D input (B', D), got {point_in_hyperbolic.dim()}D shape {point_in_hyperbolic.shape}")
        B_prime, D_in = point_in_hyperbolic.shape
        dev = point_in_hyperbolic.device
        ref_param_for_dtype = next(iter(self.parameters()), None)
        dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else point_in_hyperbolic.dtype
        if self.dim == 0:
            dummy_out_shape = (B_prime, 0)
            dummy_dtype_dev = {'device': dev, 'dtype': dtype_to_use}
            current_spread_tensor = self.get_current_spread_scalar_tensor().to(dtype_to_use)
            return (torch.zeros(dummy_out_shape, **dummy_dtype_dev), torch.zeros(dummy_out_shape, **dummy_dtype_dev),
                    None, None, current_spread_tensor)
        current_c_val = self.get_current_curvature_scalar()
        current_s_val = self.get_current_scale_scalar()
        current_sigma_out_tensor = self.get_current_spread_scalar_tensor()
        current_manifold_obj = PoincareBall(c_scalar=current_c_val)
        if self.level_descriptor_param is not None and hasattr(self.level_descriptor_param, 'manifold'):
            setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=current_c_val))
        if self.boundary_manifold_module is not None:
            self.boundary_manifold_module.set_current_manifold_c(current_c_val)
        point_in_proj = current_manifold_obj.proju(point_in_hyperbolic.to(dtype_to_use))
        tan_main_component = current_manifold_obj.logmap0(point_in_proj)
        tan_rel_component = torch.zeros_like(tan_main_component)
        ld_point_self_hyperbolic = None
        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None] and self.boundary_manifold_module is not None and self.boundary_manifold_module.num_points > 0:
            if relative_vectors_tangent_in.shape[0] != B_prime:
                raise ValueError(f"RelVec shape mismatch: {relative_vectors_tangent_in.shape[0]} != B' {B_prime}")
            tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)
        if self.use_ld and self.level_descriptor_param is not None:
            ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use))
        tan_desc_prev_level_component = torch.zeros_like(tan_main_component)
        if descriptor_point_in_hyperbolic is not None and self.use_ld:
            if descriptor_point_in_hyperbolic.shape[0] != B_prime:
                raise ValueError(f"DescIn shape mismatch: {descriptor_point_in_hyperbolic.shape[0]} != B' {B_prime}")
            desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use))
            tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)
        inputs_for_combiner = [tan_main_component]
        if self.relative_vector_aggregation not in ['none', None] and self.boundary_manifold_module is not None and self.boundary_manifold_module.num_points > 0:
            inputs_for_combiner.append(tan_rel_component)
        if self.use_ld:
            inputs_for_combiner.append(tan_desc_prev_level_component)
        if not inputs_for_combiner:
            combined_tangent_features = torch.zeros(B_prime, self.dim, device=dev, dtype=dtype_to_use)
        elif len(inputs_for_combiner) > 1:
            combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1)
        else:
            combined_tangent_features = inputs_for_combiner[0]
        if self.comb_in_dim > 0:
            if combined_tangent_features.shape[-1] < self.comb_in_dim:
                padding_size = self.comb_in_dim - combined_tangent_features.shape[-1]
                if padding_size > 0:
                    combined_tangent_features = F.pad(combined_tangent_features, (0, padding_size))
            elif combined_tangent_features.shape[-1] > self.comb_in_dim:
                self.logger.warning(f"Tangent Combiner input dim {combined_tangent_features.shape[-1]} > expected {self.comb_in_dim}. Truncating.")
                combined_tangent_features = combined_tangent_features[..., :self.comb_in_dim]
        elif combined_tangent_features.shape[-1] > 0 and self.comb_in_dim == 0:
            B_prime_local = combined_tangent_features.shape[0]
            self.logger.warning(f"Tangent Combiner expects 0-dim input (self.comb_in_dim=0), but got {combined_tangent_features.shape[-1]} features. Forcing to (Batch={B_prime_local}, 0).")
            combined_tangent_features = torch.empty(B_prime_local, 0, device=combined_tangent_features.device, dtype=combined_tangent_features.dtype)
        v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features)
        v_final_for_expmap_unclamped = v_combined_tangent_processed * current_s_val
        if self.use_flow and self.tangent_flow_module is not None:
            flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val
            v_final_for_expmap_unclamped = v_final_for_expmap_unclamped + flow_effect
        scaled_output_tangent_for_expmap = torch.clamp(v_final_for_expmap_unclamped, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        point_this_level_out_hyperbolic = current_manifold_obj.expmap0(scaled_output_tangent_for_expmap)
        tangent_out_for_aggregation = v_combined_tangent_processed.to(dtype_to_use)
        boundary_points_this_level_hyperbolic = None
        if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None:
            boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype=dtype_to_use, device=dev)
        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None:
            if ld_point_self_hyperbolic.dim() == 1:
                descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B_prime, -1).to(dtype=dtype_to_use)
            else:
                descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.to(dtype=dtype_to_use)
        output_dtype = point_in_hyperbolic.dtype
        point_out = point_this_level_out_hyperbolic.to(dtype=output_dtype)
        tangent_out = tangent_out_for_aggregation.to(dtype=output_dtype)
        desc_out = descriptor_point_out_for_transform_hyperbolic.to(dtype=output_dtype) if descriptor_point_out_for_transform_hyperbolic is not None else None
        bound_out = boundary_points_this_level_hyperbolic.to(dtype=output_dtype) if boundary_points_this_level_hyperbolic is not None else None
        sigma_out = current_sigma_out_tensor.to(dtype=output_dtype)
        return (point_out, tangent_out, desc_out, bound_out, sigma_out)

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_tangent_dim: int, output_tangent_dim: int, config: Dict):
        super().__init__()
        current_logger=logging.getLogger("WuBuSpecTransV01.WuBuModel")
        self.input_tangent_dim, self.output_tangent_dim, self.config = input_tangent_dim, output_tangent_dim, config
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
        if input_tangent_dim > 0 and first_level_dim > 0 and input_tangent_dim != first_level_dim:
            self.input_tangent_projection = nn.Linear(input_tangent_dim, first_level_dim)
        else:
            self.input_tangent_projection = nn.Identity()
        self.input_tangent_layernorm = nn.LayerNorm(first_level_dim) if first_level_dim > 0 else nn.Identity()
        self.levels_modulelist = nn.ModuleList()
        self.transforms_modulelist = nn.ModuleList()
        if self.num_levels > 0:
            for i in range(self.num_levels):
                if i < len(self.hyperbolic_dims_list) and i < len(self.initial_curvatures_list):
                    self.levels_modulelist.append(
                        HyperbolicWuBuNestingLevel(i, self.hyperbolic_dims_list[i], self.config, self.initial_curvatures_list[i])
                    )
                else:
                    current_logger.error(f"Level {i} skipped: Config lists too short (dims:{len(self.hyperbolic_dims_list)}, curves:{len(self.initial_curvatures_list)})")
                    break
            num_transforms_needed = max(0, len(self.levels_modulelist) - 1)
            if num_transforms_needed > 0:
                transform_types_list = config.get("transform_types", ["linear"] * num_transforms_needed)
                transform_hidden_dims_list = config.get("transform_hidden_dims", [None] * num_transforms_needed)
                for i in range(num_transforms_needed):
                    if i + 1 < len(self.levels_modulelist) and \
                       i + 1 < len(self.hyperbolic_dims_list) and \
                       i + 1 < len(self.initial_curvatures_list):
                        t_type = transform_types_list[i] if i < len(transform_types_list) else "linear"
                        t_hidden = transform_hidden_dims_list[i] if i < len(transform_hidden_dims_list) else None
                        self.transforms_modulelist.append(
                            HyperbolicInterLevelTransform(
                                self.hyperbolic_dims_list[i], self.hyperbolic_dims_list[i+1],
                                self.initial_curvatures_list[i], self.initial_curvatures_list[i+1],
                                t_type, t_hidden, self.dropout_val,
                                self.use_rotation_in_transform_flag, self.phi_influence_rotation_init, level_idx_for_phi=i
                            )
                        )
                    else:
                        current_logger.warning(f"Skipping transform {i} to {i+1} due to insufficient config/levels for next level.")
        actual_output_dims_from_levels = [
            d for d_idx, d in enumerate(self.hyperbolic_dims_list[:len(self.levels_modulelist)]) if d > 0
        ]
        aggregated_tangent_dim_val = sum(actual_output_dims_from_levels) if actual_output_dims_from_levels else input_tangent_dim
        if aggregated_tangent_dim_val > 0 and output_tangent_dim > 0 and aggregated_tangent_dim_val != output_tangent_dim:
            self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, output_tangent_dim)
        else:
            self.output_tangent_projection = nn.Identity()
        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        current_logger.info(f"Levels: {len(self.levels_modulelist)}. Params: {param_count:,}. InDim {input_tangent_dim}, AggDim {aggregated_tangent_dim_val}, OutDim {output_tangent_dim}")
    def forward(self, x_initial_tangent_in: torch.Tensor) -> torch.Tensor:
        input_dim = x_initial_tangent_in.dim()
        B_orig, S_orig, D_orig = -1, -1, -1
        if input_dim == 3:
            B_orig, S_orig, D_orig = x_initial_tangent_in.shape
            x_proc = x_initial_tangent_in.reshape(B_orig * S_orig, D_orig)
            B_prime_for_levels = B_orig * S_orig
        elif input_dim == 2:
            B_prime, D_orig = x_initial_tangent_in.shape
            x_proc = x_initial_tangent_in
            B_prime_for_levels = B_prime
        else:
            raise ValueError(f"WuBuModel expects 2D/3D input, got {input_dim}D")
        if D_orig != self.input_tangent_dim:
            raise ValueError(f"Input feature dim {D_orig} != model input_tangent_dim {self.input_tangent_dim}")
        if self.num_levels == 0 or not self.levels_modulelist:
            out_proj = self.output_tangent_projection(x_proc)
            return out_proj.reshape(B_orig, S_orig, -1) if input_dim==3 else out_proj
        dev = x_proc.device
        ref_param_for_dtype = next(iter(self.parameters()), None)
        dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else x_proc.dtype
        x_proc = x_proc.to(dtype_to_use)
        current_tangent_projected = self.input_tangent_projection(x_proc)
        current_tangent_for_level0 = self.input_tangent_layernorm(current_tangent_projected)
        level0_module = self.levels_modulelist[0]
        c0_val = level0_module.get_current_curvature_scalar()
        m0_obj = PoincareBall(c_scalar=c0_val)
        if self.hyperbolic_dims_list[0] > 0:
            current_point_repr_hyperbolic = m0_obj.expmap0(current_tangent_for_level0)
        else:
            current_point_repr_hyperbolic = torch.empty(B_prime_for_levels, 0, device=dev, dtype=dtype_to_use)
        level_tangent_outputs_for_aggregation = []
        aggregated_relative_vectors_from_prev_transform = None
        descriptor_from_prev_transform_hyperbolic = None
        sigma_from_prev_level_tensor = torch.tensor(0.0, device=dev, dtype=dtype_to_use)
        for i, level_module in enumerate(self.levels_modulelist):
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation,
             descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic,
             sigma_out_of_level_tensor) = level_module(
                current_point_repr_hyperbolic, aggregated_relative_vectors_from_prev_transform,
                descriptor_from_prev_transform_hyperbolic, sigma_from_prev_level_tensor
            )
            if self.hyperbolic_dims_list[i] > 0:
                level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)
            if i < len(self.levels_modulelist) - 1:
                if i >= len(self.transforms_modulelist):
                    logging.getLogger("WuBuSpecTransV01.WuBuModel").warning(f"Missing transform L{i}->L{i+1}. Stop.")
                    break
                transform_module = self.transforms_modulelist[i]
                next_level_module = self.levels_modulelist[i+1]
                c_in_for_transform = level_module.get_current_curvature_scalar()
                c_out_for_transform = next_level_module.get_current_curvature_scalar()
                (point_transformed_to_next_level_hyperbolic,
                 boundaries_transformed_to_next_level_hyperbolic,
                 descriptor_transformed_to_next_level_hyperbolic
                ) = transform_module(
                    point_out_of_level_hyperbolic, boundary_points_of_level_hyperbolic,
                    descriptor_generated_by_level_hyperbolic, c_in_for_transform, c_out_for_transform
                )
                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic
                descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic
                sigma_from_prev_level_tensor = sigma_out_of_level_tensor
                aggregated_relative_vectors_from_prev_transform = None
                valid_boundary_conditions = (
                    boundaries_transformed_to_next_level_hyperbolic is not None and
                    self.relative_vector_aggregation_mode not in ['none', None] and
                    self.hyperbolic_dims_list[i+1] > 0 and
                    current_point_repr_hyperbolic.shape[-1] > 0 and
                    self.levels_modulelist[i].boundary_manifold_module is not None and
                    self.levels_modulelist[i].boundary_manifold_module.num_points > 0
                )
                if valid_boundary_conditions:
                    manifold_next_level_obj = PoincareBall(c_scalar=c_out_for_transform)
                    tan_main_next_level = manifold_next_level_obj.logmap0(current_point_repr_hyperbolic)
                    tan_bounds_next_level = manifold_next_level_obj.logmap0(boundaries_transformed_to_next_level_hyperbolic)
                    tan_bounds_next_level_expanded = tan_bounds_next_level.unsqueeze(0).expand(B_prime_for_levels, -1, -1)
                    relative_tangent_vectors = tan_main_next_level.unsqueeze(1) - tan_bounds_next_level_expanded
                    agg_mode = self.relative_vector_aggregation_mode
                    if agg_mode == "mean": agg_rel_vec = torch.mean(relative_tangent_vectors, dim=1)
                    elif agg_mode == "sum": agg_rel_vec = torch.sum(relative_tangent_vectors, dim=1)
                    elif agg_mode == "max_norm":
                        norms = torch.norm(relative_tangent_vectors, p=2, dim=-1)
                        best_idx = torch.argmax(norms, dim=1, keepdim=True)
                        best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, relative_tangent_vectors.shape[-1])
                        agg_rel_vec = torch.gather(relative_tangent_vectors, 1, best_idx_expanded).squeeze(1)
                    else: agg_rel_vec = None
                    if agg_rel_vec is not None and not torch.isfinite(agg_rel_vec).all():
                        agg_rel_vec = torch.zeros_like(tan_main_next_level)
                    aggregated_relative_vectors_from_prev_transform = agg_rel_vec
        compatible_tangent_outputs = [
            t_val.to(dtype_to_use) for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation)
            if t_val is not None and t_idx < len(self.hyperbolic_dims_list) and
               self.hyperbolic_dims_list[t_idx] > 0 and torch.isfinite(t_val).all()
        ]
        if not compatible_tangent_outputs:
            out_zeros = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use)
            return out_zeros.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim == 3 else out_zeros
        aggregated_tangent_final = torch.cat(compatible_tangent_outputs, dim=-1)
        final_output_flat = self.output_tangent_projection(aggregated_tangent_final)
        if not torch.isfinite(final_output_flat).all():
            final_output_flat = torch.nan_to_num(final_output_flat, nan=0.0)
        return final_output_flat.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim == 3 else final_output_flat

class GradientStats:
    def __init__(self): self.reset()
    def reset(self):
        self.total_params_updated=0
        self.total_finite_grads_processed=0
        self.total_non_finite_grads_encountered=0
        self.params_skipped_due_non_finite_grad=0
        self.max_grad_norm_observed=0.
        self.step_summary={}
    def record_param_grad(self, grad_is_finite: bool, original_norm_if_finite: Optional[float] = None):
        if grad_is_finite:
            self.total_finite_grads_processed += 1
            self.max_grad_norm_observed = max(self.max_grad_norm_observed, original_norm_if_finite if original_norm_if_finite is not None else 0.0)
        else:
            self.total_non_finite_grads_encountered += 1
            self.params_skipped_due_non_finite_grad += 1
    def finalize_step_stats(self, num_params_in_optimizer_step: int):
        self.total_params_updated=num_params_in_optimizer_step-self.params_skipped_due_non_finite_grad
        self.step_summary={
            "params_in_step":num_params_in_optimizer_step,
            "params_updated":self.total_params_updated,
            "params_skipped_non_finite_grad":self.params_skipped_due_non_finite_grad,
            "initial_finite_grads":self.total_finite_grads_processed,
            "initial_non_finite_grads":self.total_non_finite_grads_encountered,
            "max_finite_grad_norm_observed":self.max_grad_norm_observed
        }
    def get_step_summary_for_logging(self) -> dict:
        return self.step_summary.copy()

class HAKMEMQController:
    def __init__(self, q_learning_rate: float = 0.01, discount_factor: float = 0.90,
                 epsilon_start: float = 0.5, epsilon_min: float = 0.05, epsilon_decay: float = 0.9995,
                 lr_scale_options: Optional[List[float]] = None,
                 momentum_scale_options: Optional[List[float]] = None,
                 lambda_kl_scale_options: Optional[List[float]] = None,
                 max_q_table_size: int = 25000, state_history_len: int = 5,
                 lambda_kl_state_history_len: int = 5,
                 reward_clipping: Optional[Tuple[float, float]] = (-2.0, 2.0),
                 q_value_clipping: Optional[Tuple[float, float]] = (-30.0, 30.0)):
        self.q_table: Dict[tuple, Dict[str, np.ndarray]] = {}
        self.alpha = q_learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_clipping = reward_clipping
        self.q_value_clipping = q_value_clipping
        self.current_lambda_kl: float = 0.0001
        _lr_options = lr_scale_options if lr_scale_options is not None else [0.8, 0.9, 1.0, 1.1, 1.2]
        _mom_options = momentum_scale_options if momentum_scale_options is not None else [0.95, 0.98, 1.0, 1.01, 1.02]
        _lkl_options = lambda_kl_scale_options if lambda_kl_scale_options is not None else [0.94, 0.97, 1.0, 1.03, 1.06]
        self.action_ranges = {
            'lr_scale': np.array(_lr_options, dtype=np.float32),
            'momentum_scale': np.array(_mom_options, dtype=np.float32),
            'lambda_kl_scale': np.array(_lkl_options, dtype=np.float32)
        }
        self.num_actions = {p_type: len(actions) for p_type, actions in self.action_ranges.items()}
        self.state_history_len = max(3, state_history_len)
        self.loss_g_total_hist = deque(maxlen=self.state_history_len)
        self.loss_g_recon_hist = deque(maxlen=self.state_history_len)
        self.loss_g_kl_hist = deque(maxlen=self.state_history_len)
        self.loss_g_adv_hist = deque(maxlen=self.state_history_len)
        self.loss_d_total_hist = deque(maxlen=self.state_history_len)
        self.loss_d_real_hist = deque(maxlen=self.state_history_len)
        self.loss_d_fake_hist = deque(maxlen=self.state_history_len)
        self.lambda_kl_state_history_len = max(2, lambda_kl_state_history_len)
        self.interval_avg_recon_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_avg_kl_div_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_avg_d_total_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_val_metric_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.prev_lr_mom_state: Optional[tuple] = None
        self.prev_lr_mom_action: Optional[Dict[str, float]] = None
        self.prev_lambda_kl_state: Optional[tuple] = None
        self.prev_lambda_kl_action: Optional[Dict[str, float]] = None
        self.reward_hist = deque(maxlen=100)
        self.max_q_table_size = max_q_table_size
        self.q_table_access_count: Dict[tuple, int] = defaultdict(int)
        self.q_table_creation_time: Dict[tuple, float] = {}
        self.q_table_last_access_time: Dict[tuple, float] = {}
        self.reward_weights = {
            "g_recon_improvement": 2.5, "g_adv_improvement": 1.2, "g_kl_control_penalty": 0.3,
            "g_loss_stability": 0.1, "d_balance_target": 1.5, "d_real_low_bonus": 0.7,
            "d_fake_low_meaningful_bonus": 0.7, "d_loss_stability": 0.1,
            "gan_balance_g_bonus": 0.3, "gan_balance_d_penalty": 0.3,
            "oscillation_penalty": 0.25, "extreme_loss_penalty": 0.75,
            "lambda_kl_recon_focus": 1.5, "lambda_kl_kl_target_range": 1.0,
            "lambda_kl_val_metric_improvement": 2.0, "lambda_kl_stability_penalty": 0.5
        }
        self.logger = logging.getLogger(f"WuBuSpecTransV01.QController")
        self.logger.info(f"HAKMEMQController (LR/Mom + Scheduled LambdaKL) initialized. Eps: {self.epsilon_start:.2f}->{self.epsilon_min:.2f}")
        self._internal_step_counter = 0
    def _get_trend_bin(self, history: deque, current_val: Optional[float],
                       relative_to_median: bool = True, value_scale_for_diff:float = 1.0) -> int:
        if current_val is None or not np.isfinite(current_val): return 2 # Neutral/Unknown
        valid_history = [h for h in history if np.isfinite(h)]
        if not valid_history: return 2 # Neutral if no history
        prev_median = np.median(valid_history)
        diff = current_val - prev_median
        if relative_to_median:
            denominator_scale = abs(prev_median)
            if denominator_scale < value_scale_for_diff * 0.01 + EPS:
                denominator_scale = max(abs(current_val), value_scale_for_diff * 0.01 + EPS)
            denominator = denominator_scale + EPS
            relative_diff = diff / denominator
        else:
            denominator = value_scale_for_diff + EPS
            relative_diff = diff / denominator
        if relative_diff < -0.15: return 0    # Strong decrease
        if relative_diff < -0.02: return 1    # Mild decrease
        if relative_diff <= 0.02: return 2    # Stable
        if relative_diff <= 0.15: return 3    # Mild increase
        return 4                           # Strong increase
    def _update_loss_histories(self, current_losses: Dict[str, float]):
        loss_map = {
            'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist
        }
        for name, deq in loss_map.items():
            loss_val = current_losses.get(name)
            if loss_val is not None and np.isfinite(loss_val):
                deq.append(loss_val)
    def get_lr_mom_state(self, current_losses: Dict[str, float], current_lr: float,
                         current_momentum: float, is_generator_q: bool) -> Optional[tuple]:
        self._internal_step_counter +=1
        self._update_loss_histories(current_losses)
        required_keys_g = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        required_keys_d = ['loss_d_total', 'loss_g_total', 'loss_d_real', 'loss_d_fake', 'loss_g_adv']
        required_keys = required_keys_g if is_generator_q else required_keys_d
        if not all(key in current_losses and np.isfinite(current_losses[key]) for key in required_keys):
            return None
        if is_generator_q:
            s_g_total_trend = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total'])
            s_d_total_trend_opp = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total'])
            s_g_recon_trend = self._get_trend_bin(self.loss_g_recon_hist, current_losses['loss_g_recon'])
            kl_val, recon_val = current_losses['loss_g_kl'], current_losses['loss_g_recon']
            s_kl_problem = 0
            kl_weight = self.reward_weights.get("g_recon_improvement", 2.0)
            if (self.current_lambda_kl * kl_val > kl_weight * recon_val * 2.0 and
                recon_val > 0.10 and self.current_lambda_kl > 0.0005):
                s_kl_problem = 1
            elif kl_val > 150.0 and recon_val > 0.15 and self.current_lambda_kl > 0.005:
                s_kl_problem = 2
            s_g_adv_level = np.digitize(current_losses['loss_g_adv'], [0.3, 0.6, 1.0]).item()
            s_lr_bin = np.digitize(current_lr, [1e-5, 5e-5, 2e-4]).item()
            s_mom_bin = np.digitize(current_momentum, [0.85, 0.95]).item()
            eps_bin = np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item()
            state_tuple = ("LRM_G", s_g_total_trend, s_d_total_trend_opp, s_g_recon_trend,
                           s_kl_problem, s_g_adv_level, s_lr_bin, s_mom_bin, eps_bin)
        else:
            s_d_total_trend = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total'])
            s_g_total_trend_opp = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total'])
            s_d_balance_bin = np.digitize(current_losses['loss_d_total'], [0.35, 0.65, 0.85]).item()
            ratio_fake_real = current_losses['loss_d_fake'] / (current_losses['loss_d_real'] + EPS)
            s_d_fake_vs_real_ratio_bin = np.digitize(ratio_fake_real, [0.8, 1.2, 2.0]).item()
            s_lr_bin = np.digitize(current_lr, [1e-5, 5e-5, 2e-4]).item()
            s_mom_bin = np.digitize(current_momentum, [0.85, 0.95]).item()
            eps_bin = np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item()
            s_g_adv_opp_level = np.digitize(current_losses.get('loss_g_adv', 0.7), [0.2, 0.5]).item()
            state_tuple = ("LRM_D", s_d_total_trend, s_g_total_trend_opp, s_d_balance_bin,
                           s_d_fake_vs_real_ratio_bin, s_g_adv_opp_level, s_lr_bin, s_mom_bin, eps_bin)
        self._ensure_q_state_exists(state_tuple)
        return state_tuple
    def get_lambda_kl_state(self, interval_metrics: Dict[str, Optional[float]]) -> Optional[tuple]:
        required_keys = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric', 'current_lambda_kl_val']
        valid_metrics = True
        for key in required_keys:
            val = interval_metrics.get(key)
            if val is None or not np.isfinite(val):
                valid_metrics = False
                break
        if not valid_metrics:
            self.logger.debug(f"LambdaKL QState: Insufficient/non-finite interval metrics. Need: {required_keys}, Got: {interval_metrics}")
            return None
        # Type ignore justifications: Handled by the check above.
        self.interval_avg_recon_hist.append(float(interval_metrics['avg_recon'])) # type: ignore
        self.interval_avg_kl_div_hist.append(float(interval_metrics['avg_kl_div'])) # type: ignore
        self.interval_avg_d_total_hist.append(float(interval_metrics['avg_d_total'])) # type: ignore
        self.interval_val_metric_hist.append(float(interval_metrics['val_metric'])) # type: ignore
        s_interval_recon_trend = self._get_trend_bin(self.interval_avg_recon_hist, float(interval_metrics['avg_recon'])) # type: ignore
        s_interval_kl_trend = self._get_trend_bin(self.interval_avg_kl_div_hist, float(interval_metrics['avg_kl_div'])) # type: ignore
        s_interval_val_metric_trend = self._get_trend_bin(self.interval_val_metric_hist, float(interval_metrics['val_metric'])) # type: ignore
        s_current_lambda_kl_bin = np.digitize(float(interval_metrics['current_lambda_kl_val']), [0.0005, 0.005, 0.05]).item() # type: ignore
        s_interval_d_balance_bin = np.digitize(float(interval_metrics['avg_d_total']), [0.35, 0.65, 0.85]).item() # type: ignore
        eps_bin = np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item()
        state_tuple = ("LKL", s_interval_recon_trend, s_interval_kl_trend, s_interval_val_metric_trend,
                       s_current_lambda_kl_bin, s_interval_d_balance_bin, eps_bin)
        self._ensure_q_state_exists(state_tuple)
        return state_tuple
    def _ensure_q_state_exists(self, state_tuple: tuple):
        current_time = time.time()
        self.q_table_access_count[state_tuple] += 1
        self.q_table_last_access_time[state_tuple] = current_time
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {
                p_type: np.zeros(n_actions, dtype=np.float32)
                for p_type, n_actions in self.num_actions.items()
            }
            self.q_table_creation_time[state_tuple] = current_time
            self._manage_q_table_size()
    def choose_action(self, state: Optional[tuple], mode: str = 'lr_mom') -> Dict[str, float]:
        default_actions = {'lr_scale': 1.0, 'momentum_scale': 1.0, 'lambda_kl_scale': 1.0}
        action_types_to_choose = []
        if mode == 'lr_mom': action_types_to_choose = ['lr_scale', 'momentum_scale']
        elif mode == 'lambda_kl': action_types_to_choose = ['lambda_kl_scale']
        else: raise ValueError(f"Invalid mode for choose_action: {mode}")
        if state is None or state not in self.q_table:
            return {k: default_actions[k] for k in action_types_to_choose}
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        chosen_actions = {}
        for param_type in action_types_to_choose:
            q_values = self.q_table[state].get(param_type)
            action_space = self.action_ranges[param_type]
            if q_values is None:
                self.logger.error(f"Q-values for {param_type} missing in state {state}. Choosing default.")
                chosen_actions[param_type] = default_actions[param_type]
                continue
            if random.random() < self.epsilon:
                chosen_idx = random.randrange(len(action_space))
            else:
                finite_q = q_values[np.isfinite(q_values)]
                if finite_q.size > 0:
                    best_q_val = np.max(finite_q)
                    best_indices_options = np.where(np.isclose(q_values, best_q_val) & np.isfinite(q_values))[0]
                    if best_indices_options.size > 0:
                        chosen_idx = random.choice(best_indices_options)
                    else: # Should not happen if finite_q.size > 0
                        chosen_idx = random.randrange(len(action_space))
                else:
                    chosen_idx = random.randrange(len(action_space))
                    self.logger.warning(f"State {state}, PType {param_type}: All Q-vals non-finite. Random action.")
            chosen_actions[param_type] = float(action_space[chosen_idx])
        if mode == 'lr_mom': self.prev_lr_mom_action = chosen_actions.copy()
        elif mode == 'lambda_kl': self.prev_lambda_kl_action = chosen_actions.copy()
        return chosen_actions
    def update_q_values(self, state: tuple, action: Dict[str, float], reward: float,
                        next_state: Optional[tuple], mode: str = 'lr_mom'):
        if state not in self.q_table: return
        if self.reward_clipping:
            reward = np.clip(reward, self.reward_clipping[0], self.reward_clipping[1])
        self.reward_hist.append(reward)
        action_types_to_update = list(action.keys())
        for param_type in action_types_to_update:
            chosen_scale_value = action[param_type]
            action_idx_arr = np.where(np.isclose(self.action_ranges[param_type], chosen_scale_value))[0]
            if not action_idx_arr.size: continue
            action_idx = action_idx_arr[0]
            current_q_value = self.q_table[state][param_type][action_idx]
            max_future_q = 0.0
            if next_state is not None and next_state in self.q_table and param_type in self.q_table[next_state]:
                next_q_vals = self.q_table[next_state][param_type]
                if np.any(np.isfinite(next_q_vals)):
                    max_future_q = np.max(next_q_vals[np.isfinite(next_q_vals)])
            td_target = reward + self.gamma * max_future_q
            td_error = td_target - current_q_value
            new_q_value = current_q_value + self.alpha * td_error
            if np.isfinite(new_q_value):
                if self.q_value_clipping:
                    new_q_value = np.clip(new_q_value, self.q_value_clipping[0], self.q_value_clipping[1])
                self.q_table[state][param_type][action_idx] = new_q_value
    def _manage_q_table_size(self):
        if len(self.q_table) <= self.max_q_table_size: return
        num_to_prune = len(self.q_table) - self.max_q_table_size
        current_time = time.time()
        state_scores = {
            s_tuple: (
                self.q_table_access_count.get(s_tuple, 1) *
                (1.0 + np.log1p((current_time - self.q_table_creation_time.get(s_tuple, current_time)) / 86400.0)) *
                (1.0 / (1.0 + np.log1p((current_time - self.q_table_last_access_time.get(s_tuple, current_time)) / 3600.0)))
            ) for s_tuple in self.q_table.keys()
        }
        sorted_states_for_pruning = sorted(state_scores.keys(), key=lambda s: state_scores[s])
        pruned_count = 0
        for i in range(num_to_prune):
            if i < len(sorted_states_for_pruning):
                s_rm = sorted_states_for_pruning[i]
                self.q_table.pop(s_rm, None)
                self.q_table_access_count.pop(s_rm, None)
                self.q_table_creation_time.pop(s_rm, None)
                self.q_table_last_access_time.pop(s_rm, None)
                pruned_count +=1
        if pruned_count > 0:
            self.logger.info(f"Pruned {pruned_count} Q-table entries. New size: {len(self.q_table)}.")
    def compute_lr_mom_reward(self, current_losses: Dict[str, float], is_generator_q: bool) -> float:
        total_reward = 0.0
        w = self.reward_weights
        for loss_name, loss_val in current_losses.items():
            if not np.isfinite(loss_val):
                total_reward -= w["extreme_loss_penalty"] * 5
                current_losses[loss_name] = 100.0 # Cap for calculations
            elif abs(loss_val) > 500:
                total_reward -= w["extreme_loss_penalty"] * (abs(loss_val) / 500.0)
                current_losses[loss_name] = np.sign(loss_val) * 500
        def get_prev_median(hist_deque, current_val_fallback):
            valid_hist = [v for v in hist_deque if np.isfinite(v)]
            if len(valid_hist) > 1: return np.median(valid_hist[:-1])
            if len(valid_hist) == 1: return valid_hist[0]
            return current_val_fallback
        if is_generator_q:
            loss_g_recon = current_losses.get('loss_g_recon', 1.0)
            prev_g_recon = get_prev_median(self.loss_g_recon_hist, loss_g_recon)
            recon_improvement = prev_g_recon - loss_g_recon
            recon_scale = 1.0 + math.log1p(max(0, loss_g_recon - 0.02) * 20)
            total_reward += np.tanh(recon_improvement / (abs(prev_g_recon) + 0.01 + EPS) * recon_scale) * w["g_recon_improvement"]
            loss_g_adv = current_losses.get('loss_g_adv', 0.7)
            prev_g_adv = get_prev_median(self.loss_g_adv_hist, loss_g_adv)
            adv_improvement = prev_g_adv - loss_g_adv
            total_reward += np.tanh(adv_improvement / (abs(prev_g_adv) + EPS)) * w["g_adv_improvement"]
            loss_g_kl = current_losses.get('loss_g_kl', 0.0)
            if loss_g_kl > 100.0 and self.current_lambda_kl >= 0.0005 and loss_g_recon > 0.1:
                total_reward -= w["g_kl_control_penalty"] * min(1.0, (loss_g_kl - 100.0) / 200.0)
            loss_d_total = current_losses.get('loss_d_total', 0.7)
            if 0.4 < loss_d_total < 0.75: total_reward += w["gan_balance_g_bonus"]
            elif loss_d_total <= 0.3: total_reward -= w["gan_balance_g_bonus"] * 1.5
            loss_g_total = current_losses.get('loss_g_total', 1.0)
            prev_g_total = get_prev_median(self.loss_g_total_hist, loss_g_total)
            g_total_improvement = prev_g_total - loss_g_total
            total_reward += np.tanh(g_total_improvement / (abs(prev_g_total) + EPS)) * w["g_loss_stability"]
        else: # Discriminator Q
            loss_d_total = current_losses.get('loss_d_total', 0.7)
            if 0.4 < loss_d_total < 0.65: total_reward += w["d_balance_target"]
            elif loss_d_total < 0.3: total_reward -= w["d_balance_target"] * 0.5
            elif loss_d_total > 0.8: total_reward -= w["d_balance_target"] * 0.75
            loss_d_real = current_losses.get('loss_d_real', 0.7)
            if loss_d_real < 0.3: total_reward += w["d_real_low_bonus"] * (0.3 - loss_d_real) / 0.3
            loss_d_fake = current_losses.get('loss_d_fake', 0.7)
            loss_g_adv_opp = current_losses.get('loss_g_adv', 0.7)
            if loss_d_fake < 0.3 and loss_g_adv_opp > 0.4:
                total_reward += w["d_fake_low_meaningful_bonus"] * (0.3 - loss_d_fake) / 0.3
            if loss_g_adv_opp < 0.25: total_reward -= w["gan_balance_d_penalty"]
            prev_d_total = get_prev_median(self.loss_d_total_hist, loss_d_total)
            d_total_improvement = prev_d_total - loss_d_total
            total_reward += np.tanh(d_total_improvement / (abs(prev_d_total) + EPS)) * w["d_loss_stability"]
        if len(self.reward_hist) >= self.state_history_len:
            recent_rewards = list(self.reward_hist)[-self.state_history_len:]
            sign_flips = 0
            for i in range(len(recent_rewards) - 1):
                if (np.sign(recent_rewards[i]) != np.sign(recent_rewards[i+1]) and
                    abs(recent_rewards[i]) > 0.05 and abs(recent_rewards[i+1]) > 0.05):
                    sign_flips += 1
            if sign_flips >= (self.state_history_len // 2) :
                total_reward -= w["oscillation_penalty"] * (sign_flips / self.state_history_len)
        if self.reward_clipping:
            total_reward = np.clip(total_reward, self.reward_clipping[0], self.reward_clipping[1])
        return float(total_reward)
    def compute_lambda_kl_reward(self, interval_metrics: Dict[str, Optional[float]],
                                 prev_interval_metrics: Optional[Dict[str, Optional[float]]]) -> float:
        total_reward = 0.0
        w = self.reward_weights
        _prev_metrics = prev_interval_metrics if prev_interval_metrics is not None else {}
        current_val_metric = interval_metrics.get('val_metric')
        prev_val_metric = _prev_metrics.get('val_metric', current_val_metric)
        if current_val_metric is not None and prev_val_metric is not None and \
           np.isfinite(current_val_metric) and np.isfinite(prev_val_metric):
            val_metric_change = float(current_val_metric) - float(prev_val_metric)
            total_reward += np.tanh(val_metric_change * 5.0) * w["lambda_kl_val_metric_improvement"]
        current_avg_recon = interval_metrics.get('avg_recon')
        prev_avg_recon = _prev_metrics.get('avg_recon', current_avg_recon)
        if current_avg_recon is not None and prev_avg_recon is not None and \
           np.isfinite(current_avg_recon) and np.isfinite(prev_avg_recon):
            recon_change = float(prev_avg_recon) - float(current_avg_recon)
            recon_penalty_factor = 1.0 if recon_change >= -0.05 else (1.0 + abs(recon_change * 10))
            total_reward += np.tanh(recon_change * 10.0 / recon_penalty_factor) * w["lambda_kl_recon_focus"]
        current_kl_div = interval_metrics.get('avg_kl_div')
        prev_kl_div = _prev_metrics.get('avg_kl_div', current_kl_div)
        if current_kl_div is not None and prev_kl_div is not None and \
           np.isfinite(current_kl_div) and np.isfinite(prev_kl_div):
            if float(current_kl_div) > 100:
                kl_div_decrease = float(prev_kl_div) - float(current_kl_div)
                total_reward += np.tanh(kl_div_decrease / 50.0) * w["lambda_kl_kl_target_range"]
            elif float(current_kl_div) < 20 and current_avg_recon is not None and float(current_avg_recon) > 0.05:
                total_reward -= w["lambda_kl_kl_target_range"] * 0.5
        current_avg_d_total = interval_metrics.get('avg_d_total')
        prev_avg_d_total = _prev_metrics.get('avg_d_total', current_avg_d_total)
        if current_avg_d_total is not None and prev_avg_d_total is not None and \
           np.isfinite(current_avg_d_total) and np.isfinite(prev_avg_d_total):
            d_total_stability_change = abs(float(current_avg_d_total) - float(prev_avg_d_total))
            if d_total_stability_change > 0.2:
                total_reward -= w["lambda_kl_stability_penalty"] * (d_total_stability_change / 0.2)
        current_lambda_kl_val = interval_metrics.get('current_lambda_kl_val')
        if current_lambda_kl_val is not None and float(current_lambda_kl_val) > 0.5 and \
           current_avg_recon is not None and float(current_avg_recon) > 0.15:
            total_reward -= 0.5
        if self.logger.isEnabledFor(logging.DEBUG):
            log_mets = {k: f'{v:.3f}' if isinstance(v, (float, np.float32, np.float64)) and np.isfinite(v) else str(v)
                        for k,v in interval_metrics.items()}
            self.logger.debug(f"LambdaKL_Rew: Raw={total_reward:.3f}. IntervalMet: {log_mets}")
        if self.reward_clipping:
            total_reward = np.clip(total_reward, self.reward_clipping[0], self.reward_clipping[1])
        return float(total_reward)
    def set_current_lambda_kl(self, lambda_kl_val: float):
        if np.isfinite(lambda_kl_val):
            self.current_lambda_kl = float(lambda_kl_val)
        else:
            self.logger.warning(f"Attempted to set non-finite lambda_kl: {lambda_kl_val}")
    def get_info(self) -> Dict:
        q_mem_mb = 0.0
        try:
            if self.q_table:
                q_mem_mb = sum(
                    sys.getsizeof(s_tuple) +
                    sum(q_vals.nbytes + sys.getsizeof(p_type) for p_type, q_vals in q_actions.items())
                    for s_tuple, q_actions in self.q_table.items()
                ) / (1024**2)
        except Exception: q_mem_mb = -1.0 # Defensive
        avg_reward_recent = np.mean(list(self.reward_hist)) if self.reward_hist else 0.0
        return {
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "q_table_mem_mb_approx": round(q_mem_mb, 3),
            "last_lr_mom_action": self.prev_lr_mom_action if self.prev_lr_mom_action else "None",
            "last_lambda_kl_action": self.prev_lambda_kl_action if self.prev_lambda_kl_action else "None",
            f"avg_reward_last_{len(self.reward_hist) if self.reward_hist.maxlen is None else self.reward_hist.maxlen}": round(avg_reward_recent, 3)
        }
    def set_initial_losses(self, losses: Dict[str, float], is_generator_q: bool):
        loss_map_init = {
            'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist
        }
        relevant_keys_g = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        relevant_keys_d = ['loss_d_total', 'loss_d_real', 'loss_d_fake', 'loss_g_total', 'loss_g_adv']
        relevant_keys = relevant_keys_g if is_generator_q else relevant_keys_d
        for name in relevant_keys:
            val = losses.get(name)
            if val is not None and np.isfinite(val):
                loss_map_init[name].append(val)
        for deq_obj in loss_map_init.values():
            if deq_obj: # If it has at least one element
                 while len(deq_obj) < self.state_history_len:
                    deq_obj.appendleft(deq_obj[0])
    def set_initial_lambda_kl_metrics(self, interval_metrics: Dict[str, Optional[float]]):
        metric_map = {
            'avg_recon': self.interval_avg_recon_hist, 'avg_kl_div': self.interval_avg_kl_div_hist,
            'avg_d_total': self.interval_avg_d_total_hist, 'val_metric': self.interval_val_metric_hist
        }
        for name, deq in metric_map.items():
            val = interval_metrics.get(name)
            if val is not None and np.isfinite(val):
                deq.append(float(val))
        for deq_obj in metric_map.values():
            if deq_obj:
                while len(deq_obj) < self.lambda_kl_state_history_len:
                    deq_obj.appendleft(deq_obj[0])

class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, momentum: float = 0.9,
                 weight_decay: float = 0.01, max_grad_norm_risgd: float = 1.0,
                 q_learning_config: Optional[Dict] = None, optimizer_type: str = "generator"):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, initial_lr=lr, momentum=momentum, initial_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.optimizer_type = optimizer_type.lower()
        if self.optimizer_type not in ["generator", "discriminator"]:
            raise ValueError("optimizer_type must be 'generator' or 'discriminator'")
        if isinstance(q_learning_config, dict):
            q_params = q_learning_config.copy()
            self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_params)
        else:
            self.q_controller = None
        self.logger = logging.getLogger(f"WuBuSpecTransV01.RiSGD.{self.optimizer_type.capitalize()}")
        if not self.logger.hasHandlers() and not logging.getLogger("WuBuSpecTransV01").hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
        self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled for {self.optimizer_type} optimizer.")
        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count_internal = 0
        self.grad_stats = GradientStats()
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state.setdefault(p, {})
    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
    def q_controller_update_and_set_hyperparams(self, avg_losses_dict: Dict[str, Optional[float]],
                                                current_lambda_kl_value: Optional[float] = None):
        if not self.q_controller: return
        finite_losses_for_q_state: Dict[str, float] = {
            k: v for k, v in avg_losses_dict.items() if v is not None and np.isfinite(v)
        }
        is_gen_q = (self.optimizer_type == "generator")
        req_keys_g = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        req_keys_d = ['loss_d_total', 'loss_g_total', 'loss_g_adv', 'loss_d_real', 'loss_d_fake']
        required_keys = req_keys_g if is_gen_q else req_keys_d
        if not all(key in finite_losses_for_q_state for key in required_keys):
            self.logger.debug(f"QCtrl ({self.optimizer_type}): Insufficient finite losses for LR/Mom state. Skipping Q-update. Need: {required_keys}, Got: {list(finite_losses_for_q_state.keys())}")
            return
        if hasattr(self.q_controller, 'set_current_lambda_kl') and current_lambda_kl_value is not None:
            self.q_controller.set_current_lambda_kl(current_lambda_kl_value)
        current_lr_for_q_state = self.param_groups[0]['lr']
        current_mom_for_q_state = self.param_groups[0]['momentum']
        q_state_current = self.q_controller.get_lr_mom_state(
            finite_losses_for_q_state, current_lr_for_q_state, current_mom_for_q_state, is_generator_q=is_gen_q
        )
        if self.q_controller.prev_lr_mom_state is not None and \
           self.q_controller.prev_lr_mom_action is not None and q_state_current is not None:
            reward = self.q_controller.compute_lr_mom_reward(finite_losses_for_q_state, is_generator_q=is_gen_q)
            if np.isfinite(reward):
                self.q_controller.update_q_values(
                    self.q_controller.prev_lr_mom_state, self.q_controller.prev_lr_mom_action,
                    reward, q_state_current, mode='lr_mom'
                )
        elif q_state_current is not None and hasattr(self.q_controller, 'set_initial_losses'):
             self.q_controller.set_initial_losses(finite_losses_for_q_state, is_generator_q=is_gen_q)
        self.q_controller.prev_lr_mom_state = q_state_current
        action_for_upcoming_step = self.q_controller.choose_action(q_state_current, mode='lr_mom')
        if action_for_upcoming_step:
            for group in self.param_groups:
                base_lr = group['initial_lr']
                base_mom = group['initial_momentum']
                group['lr'] = float(np.clip(base_lr * action_for_upcoming_step.get('lr_scale', 1.0), 1e-8, 1.0))
                group['momentum'] = float(np.clip(base_mom * action_for_upcoming_step.get('momentum_scale', 1.0), 0.0, 0.999))
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad
                if not torch.isfinite(grad).all():
                    self.logger.warning(f"Optimizer step: Non-finite gradient for param shape {p.shape} ({self.optimizer_type}). Skipping update.")
                    self.state[p].pop('momentum_buffer', None)
                    continue
                if self.max_grad_norm_risgd > 0 and self.max_grad_norm_risgd != float('inf'):
                    param_grad_norm = grad.norm().item()
                    if param_grad_norm > self.max_grad_norm_risgd:
                        grad.mul_(self.max_grad_norm_risgd / (param_grad_norm + EPS))
                manifold: Optional[Manifold] = getattr(p, 'manifold', None)
                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    p_projected_on_manifold = manifold.proju(p)
                    grad_eff = grad.clone()
                    if weight_decay != 0: grad_eff.add_(p, alpha=weight_decay)
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
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None:
                            buf = torch.clone(riemannian_grad).detach()
                        else:
                            if buf.shape == riemannian_grad.shape:
                                buf.mul_(momentum).add_(riemannian_grad)
                            else: # Shape mismatch, reinitialize
                                buf = torch.clone(riemannian_grad).detach()
                        self.state[p]['momentum_buffer'] = buf
                    else:
                        buf = riemannian_grad
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite momentum buffer for P:{p.shape} (c={manifold.c:.2e}). Resetting.")
                        buf.zero_()
                        self.state[p]['momentum_buffer'] = buf
                    expmap_tangent_vector = buf.mul(-lr)
                    if not torch.isfinite(expmap_tangent_vector).all():
                        self.logger.warning(f"Non-finite tangent vector for expmap P:{p.shape} (c={manifold.c:.2e}). Skipping.")
                        continue
                    try:
                        new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
                        if not torch.isfinite(new_p_candidate).all():
                            self.logger.warning(f"Expmap resulted in non-finite P:{p.shape} (c={manifold.c:.2e}). Projecting and zeroing momentum.")
                            p.data = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0))
                            if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                        else:
                            p.data = manifold.proju(new_p_candidate)
                    except Exception as e_expmap:
                        self.logger.error(f"Expmap failed for P:{p.shape} (c={manifold.c:.2e}): {e_expmap}. Zeroing momentum.")
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                        continue
                    if not torch.isfinite(p.data).all():
                        self.logger.error(f"Parameter P:{p.shape} (c={manifold.c:.2e}) became non-finite. Resetting to origin.")
                        p.data = manifold.expmap0(torch.zeros_like(p.data, device=p.device))
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                else: # Euclidean update
                    grad_eff_euc = grad.clone()
                    if weight_decay != 0: grad_eff_euc.add_(p, alpha=weight_decay)
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None:
                            buf = torch.clone(grad_eff_euc).detach()
                        else:
                            if buf.shape == grad_eff_euc.shape:
                                buf.mul_(momentum).add_(grad_eff_euc)
                            else: # Shape mismatch
                                buf = torch.clone(grad_eff_euc).detach()
                        self.state[p]['momentum_buffer'] = buf
                    else:
                        buf = grad_eff_euc
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite Euclidean momentum buffer for P:{p.shape}. Resetting.")
                        buf.zero_()
                        self.state[p]['momentum_buffer'] = buf
                    p.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all():
                        self.logger.warning(f"Euclidean P:{p.shape} became non-finite. Clamping and zeroing momentum.")
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e5, neginf=-1e5)
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
        self._step_count_internal += 1
        return loss
    def get_q_controller_info(self) -> Dict:
        return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}
    def get_gradient_stats_summary_optimizer_view(self) -> Dict:
        return self.grad_stats.get_step_summary_for_logging()

# =====================================================================
# GAAD Components
# =====================================================================
def golden_subdivide_rect_fixed_n(frame_dims:Tuple[int,int], num_regions_target:int,
                                  device='cpu', dtype=torch.float, min_size_px=5) -> torch.Tensor:
    W, H = frame_dims
    all_rects = [[0,0,W,H]]
    rect_queue = deque([(0,0,W,H,0)]) # x_off, y_off, w, h, depth
    while rect_queue and len(all_rects) < num_regions_target * 3: # Generate more than needed
        x_off, y_off, w_curr, h_curr, depth = rect_queue.popleft()
        if min(w_curr, h_curr) < min_size_px or depth > 6 : continue
        is_landscape = w_curr > h_curr + EPS
        is_portrait = h_curr > w_curr + EPS
        if is_landscape:
            cut_w = w_curr / PHI
            r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px and h_curr >= min_size_px:
                all_rects.append([x_off, y_off, x_off + r1_w, y_off + h_curr])
                rect_queue.append((x_off, y_off, r1_w, h_curr, depth + 1))
            if r2_w >= min_size_px and h_curr >= min_size_px:
                all_rects.append([x_off + r1_w, y_off, x_off + r1_w + r2_w, y_off + h_curr])
                rect_queue.append((x_off + r1_w, y_off, r2_w, h_curr, depth + 1))
        elif is_portrait:
            cut_h = h_curr / PHI
            r1_h, r2_h = cut_h, h_curr - cut_h
            if w_curr >= min_size_px and r1_h >= min_size_px:
                all_rects.append([x_off, y_off, x_off + w_curr, y_off + r1_h])
                rect_queue.append((x_off, y_off, w_curr, r1_h, depth + 1))
            if w_curr >= min_size_px and r2_h >= min_size_px:
                all_rects.append([x_off, y_off + r1_h, x_off + w_curr, y_off + r1_h + r2_h])
                rect_queue.append((x_off, y_off + r1_h, w_curr, r2_h, depth + 1))
        elif abs(w_curr - h_curr) < EPS and w_curr > min_size_px * PHI : # Square-ish, subdivide like landscape
            cut_w = w_curr / PHI
            r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px and h_curr >= min_size_px:
                all_rects.append([x_off, y_off, x_off + r1_w, y_off + h_curr])
                rect_queue.append((x_off, y_off, r1_w, h_curr, depth + 1))
            if r2_w >= min_size_px and h_curr >= min_size_px:
                all_rects.append([x_off + r1_w, y_off, x_off + r1_w + r2_w, y_off + h_curr])
                rect_queue.append((x_off + r1_w, y_off, r2_w, h_curr, depth + 1))
    unique_valid_rects_tensors = []
    seen_hashes = set()
    for r_coords in all_rects:
        if r_coords[0] >= r_coords[2] - EPS or r_coords[1] >= r_coords[3] - EPS: continue # Skip zero-area
        r_tensor = torch.tensor(r_coords, dtype=dtype, device=device)
        r_hashable = tuple(round(c, 3) for c in r_coords) # Hash rounded coords
        if r_hashable not in seen_hashes:
            unique_valid_rects_tensors.append(r_tensor)
            seen_hashes.add(r_hashable)
    unique_valid_rects_tensors.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True) # Sort by area desc
    selected_rects = unique_valid_rects_tensors[:num_regions_target]
    if len(selected_rects) < num_regions_target: # Pad if not enough unique regions
        padding_box = selected_rects[-1] if selected_rects else torch.tensor([0,0,float(W),float(H)],dtype=dtype,device=device)
        selected_rects.extend([padding_box.clone() for _ in range(num_regions_target - len(selected_rects))])
    return torch.stack(selected_rects)

def phi_spiral_patch_centers_fixed_n(frame_dims:Tuple[int,int], num_centers:int,
                                     device='cpu', dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]:
    W, H = frame_dims
    centers_xy = []
    scale_factors = []
    cx, cy = W / 2.0, H / 2.0
    if num_centers <= 0:
        return torch.empty(0,2,device=device,dtype=dtype), torch.empty(0,1,device=device,dtype=dtype)
    # Center point first
    centers_xy.append([cx, cy])
    scale_factors.append(0.25) # Scale for center patch
    num_spiral_points_to_generate = num_centers - 1
    if num_spiral_points_to_generate <= 0:
        if num_centers == 1:
            return torch.tensor(centers_xy, dtype=dtype, device=device), \
                   torch.tensor(scale_factors, dtype=dtype, device=device).unsqueeze(-1)
        else: # num_centers == 0
            return torch.empty(0,2,device=device,dtype=dtype), \
                   torch.empty(0,1,device=device,dtype=dtype)
    a = 0.05 * min(W, H) # Spiral scale factor
    b = math.log(PHI) / (math.pi / 2) # Spiral shape factor
    angle_step = PHI * 2 * math.pi / num_spiral_points_to_generate if num_spiral_points_to_generate > 0 else 0
    current_angle = 0.0
    for i in range(num_spiral_points_to_generate):
        r = a * math.exp(b * current_angle)
        max_r = max(W,H) * 0.6 # Don't let spiral go too far off screen too quickly
        r = min(r,max_r)
        x = cx + r * math.cos(current_angle)
        y = cy + r * math.sin(current_angle)
        x_clamped = max(0.0, min(x, float(W)))
        y_clamped = max(0.0, min(y, float(H)))
        centers_xy.append([x_clamped, y_clamped])
        patch_scale = max(0.05, 0.20 * math.exp(-0.5 * r / (min(W,H)*0.1))) # Smaller patches further out
        scale_factors.append(patch_scale)
        current_angle += angle_step
    if len(centers_xy) < num_centers: # Pad if needed (e.g. if num_centers was small)
        num_to_pad = num_centers - len(centers_xy)
        last_xy = centers_xy[-1] if centers_xy else [cx,cy]
        last_scale = scale_factors[-1] if scale_factors else 0.1
        centers_xy.extend([last_xy] * num_to_pad)
        scale_factors.extend([last_scale] * num_to_pad)
    return torch.tensor(centers_xy[:num_centers], dtype=dtype, device=device), \
           torch.tensor(scale_factors[:num_centers], dtype=dtype, device=device).unsqueeze(-1)

# =====================================================================
# Architectural Components (WuBuSpecTrans_v0.1.1)
# =====================================================================

class RegionalSpectrogramRegionExtractor(nn.Module):
    """
    Extracts GAAD-defined regions from Mel spectrograms and resizes them to a fixed processing size.
    This fixed-size region is then considered the "block" for subsequent DCT.
    """
    def __init__(self, region_proc_size: Tuple[int, int]): # (Time_dim, Freq_dim)
        super().__init__()
        self.region_proc_size = region_proc_size # (T_proc, F_proc)
        self.logger = logging.getLogger("WuBuSpecTransV01.RegionExtractor")
        # T.Resize takes (H, W) as size, so (F_proc, T_proc)
        self.resize_transform = T.Resize((region_proc_size[1], region_proc_size[0]),
                                         interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        self.logger.info(f"Initialized RegionalSpectrogramRegionExtractor to resize regions to (T,F): {self.region_proc_size}.")

    def forward(self, mel_spectrograms: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
        # mel_spectrograms: (B, 1, N_Mels_Total, N_Time_Total) i.e. (B, C, H, W) where H=Freq, W=Time
        # bboxes_batch: (B, Num_GAAD_Regions, 4) [x1, y1, x2, y2] in pixel/bin coordinates
        # where x maps to Time (W_spec) and y maps to Freq (H_spec)
        B, NumRegions, _ = bboxes_batch.shape
        _, C_spec, H_spec, W_spec = mel_spectrograms.shape
        device = mel_spectrograms.device
        original_dtype = mel_spectrograms.dtype

        all_processed_regions = []
        for b in range(B):
            batch_regions = []
            for r in range(NumRegions):
                t1, f1, t2, f2 = bboxes_batch[b, r].round().int().tolist()

                t1_c, f1_c = max(0, t1), max(0, f1)
                t2_c, f2_c = min(W_spec, t2), min(H_spec, f2)

                if t1_c >= t2_c or f1_c >= f2_c: # If region is invalid or zero-size
                    # Create a zero tensor of the target processed size (F_proc, T_proc)
                    region_patch = torch.zeros((C_spec, self.region_proc_size[1], self.region_proc_size[0]),
                                               device=device, dtype=original_dtype)
                else:
                    # Crop: mel_spectrograms[b, :, freq_start:freq_end, time_start:time_end]
                    # So, bboxes: H_spec (Freq) uses y (f1,f2), W_spec (Time) uses x (t1,t2)
                    region_patch_raw = mel_spectrograms[b, :, f1_c:f2_c, t1_c:t2_c]
                    # Resize expects (C, H, W) -> (C, F_proc, T_proc)
                    region_patch = self.resize_transform(region_patch_raw)

                batch_regions.append(region_patch)
            all_processed_regions.append(torch.stack(batch_regions))

        # Output shape: (B, NumRegions, C_spec, F_proc, T_proc)
        return torch.stack(all_processed_regions)


class DCTCoeffEmbed(nn.Module):
    """Embeds flattened DCT coefficient vectors."""
    def __init__(self, num_dct_coeffs_per_region: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(num_dct_coeffs_per_region, embed_dim)
        self.logger = logging.getLogger("WuBuSpecTransV01.DCTCoeffEmbed")
        self.logger.info(f"Initialized DCTCoeffEmbed: {num_dct_coeffs_per_region} DCT coeffs -> {embed_dim} embed_dim.")

    def forward(self, x_flat_dct_coeffs: torch.Tensor) -> torch.Tensor:
        # x_flat_dct_coeffs: (..., num_dct_coeffs_per_region)
        return self.proj(x_flat_dct_coeffs)


class AudioSpecEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict,
                 wubu_s_config: Dict, latent_dim: int):
        super().__init__()
        self.args = args
        self.audio_config = audio_config
        self.gaad_config = gaad_config
        self.wubu_s_config = wubu_s_config
        self.latent_dim = latent_dim
        self.logger = logging.getLogger("WuBuSpecTransV01.Encoder")

        self.num_gaad_regions = gaad_config['num_regions']
        self.gaad_decomposition_type = gaad_config['decomposition_type']
        self.gaad_min_size_px = gaad_config.get('min_size_px', 5)

        self.region_proc_size = (args.region_proc_size_t, args.region_proc_size_f) # (T_proc, F_proc)
        self.region_extractor = RegionalSpectrogramRegionExtractor(region_proc_size=self.region_proc_size)

        self.num_dct_coeffs_flat = self.region_proc_size[0] * self.region_proc_size[1] # T_proc * F_proc

        self.dct_coeff_embed = DCTCoeffEmbed(
            num_dct_coeffs_per_region=self.num_dct_coeffs_flat,
            embed_dim=args.encoder_initial_tangent_dim
        )

        self.wubu_s_encoder = FullyHyperbolicWuBuNestingModel(
            input_tangent_dim=args.encoder_initial_tangent_dim,
            output_tangent_dim=audio_config['wubu_s_output_dim_encoder'],
            config=wubu_s_config
        )

        self.fc_mu = nn.Linear(audio_config['wubu_s_output_dim_encoder'], self.latent_dim)
        self.fc_logvar = nn.Linear(audio_config['wubu_s_output_dim_encoder'], self.latent_dim)

        self.apply(init_weights_general)
        self.logger.info(f"AudioSpecEncoder initialized. Region Proc Size (T,F): {self.region_proc_size}, "
                         f"DCT Coeffs/Region: {self.num_dct_coeffs_flat}")

    def _apply_dct_and_normalize(self, region_patches: torch.Tensor) -> torch.Tensor:
        # region_patches: (B, NumRegions, C_spec, F_proc, T_proc)
        B, N_Reg, C, F_p, T_p = region_patches.shape

        # Reshape for dct_2d: (B * N_Reg * C, F_proc, T_proc)
        patches_for_dct = region_patches.reshape(-1, F_p, T_p)

        if dct_2d is None or not TORCH_DCT_AVAILABLE:
            self.logger.error("dct_2d function is not available from torch_dct. Cannot perform DCT. Returning zeros.")
            return torch.zeros_like(patches_for_dct)

        dct_coeffs = dct_2d(patches_for_dct) # Output: (B * N_Reg * C, F_proc, T_proc)

        if self.args.dct_norm_type == "none":
            norm_dct_coeffs = dct_coeffs
        elif self.args.dct_norm_type == "global_scale":
            norm_dct_coeffs = dct_coeffs / self.args.dct_norm_global_scale
        elif self.args.dct_norm_type == "tanh":
            # Scale to a range then apply tanh; e.g. divide by 50-100 then tanh
            norm_dct_coeffs = torch.tanh(dct_coeffs / self.args.dct_norm_tanh_scale)
        else: # Default to global scale for safety
            self.logger.warning(f"Unknown DCT norm type: {self.args.dct_norm_type}. Using global_scale.")
            norm_dct_coeffs = dct_coeffs / self.args.dct_norm_global_scale

        return norm_dct_coeffs.reshape(B, N_Reg, C, F_p, T_p)


    def forward(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # mel_spectrogram: (B, 1, Freq_dim, Time_dim) matching (B, C, H, W)
        B, C_spec, H_spec, W_spec = mel_spectrogram.shape
        device = mel_spectrogram.device
        dtype = mel_spectrogram.dtype

        # 1. Generate GAAD bboxes on the Mel spectrogram
        gaad_bboxes_list = []
        for b_idx in range(B):
            spec_dims = (W_spec, H_spec) # (Time, Freq) for GAAD
            bboxes_current_spec = golden_subdivide_rect_fixed_n(
                spec_dims, self.num_gaad_regions, device, dtype, self.gaad_min_size_px
            )
            gaad_bboxes_list.append(bboxes_current_spec)
        gaad_bboxes_batch = torch.stack(gaad_bboxes_list) # (B, Num_GAAD_Regions, 4)

        # 2. Extract and resize GAAD regions from spectrogram
        # Output: (B, NumRegions, C_spec, F_proc, T_proc)
        processed_regions = self.region_extractor(mel_spectrogram, gaad_bboxes_batch)

        # 3. Apply 2D DCT and normalize
        # Output: (B, NumRegions, C_spec, F_proc, T_proc)
        norm_dct_coeffs_structured = self._apply_dct_and_normalize(processed_regions)

        # Assuming C_spec is 1 for Mel spectrograms
        # Flatten DCT coefficients for each region: (B, NumRegions, F_proc * T_proc)
        flat_norm_dct_coeffs = norm_dct_coeffs_structured.squeeze(2).reshape(B, self.num_gaad_regions, -1)

        # 4. Embed DCT coefficients
        # Output: (B, NumRegions, encoder_initial_tangent_dim)
        embedded_dct_coeffs = self.dct_coeff_embed(flat_norm_dct_coeffs)

        # 5. Process with WuBu-S Encoder
        wubu_s_input = embedded_dct_coeffs.reshape(B * self.num_gaad_regions, -1)
        wubu_s_features_flat = self.wubu_s_encoder(wubu_s_input) # (B*NumRegions, wubu_s_output_dim_encoder)

        # 6. Aggregate features (e.g., mean pooling over regions)
        wubu_s_features_structured = wubu_s_features_flat.reshape(B, self.num_gaad_regions, -1)
        aggregated_features = torch.mean(wubu_s_features_structured, dim=1) # (B, wubu_s_output_dim_encoder)

        # 7. Project to latent distribution parameters
        mu = self.fc_mu(aggregated_features)
        logvar = self.fc_logvar(aggregated_features)

        # Return normalized DCTs (for recon loss), bboxes, mu, logvar
        return mu, logvar, norm_dct_coeffs_structured.squeeze(2), gaad_bboxes_batch


class AudioSpecGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, latent_dim: int):
        super().__init__()
        self.args = args
        self.audio_config = audio_config
        self.gaad_config = gaad_config
        self.latent_dim = latent_dim
        self.logger = logging.getLogger("WuBuSpecTransV01.Generator")

        self.num_gaad_regions = gaad_config['num_regions']
        self.region_proc_size = (args.region_proc_size_t, args.region_proc_size_f) # (T_proc, F_proc)
        self.num_dct_coeffs_flat = self.region_proc_size[0] * self.region_proc_size[1]

        self.initial_gen_wubu_dim = args.encoder_initial_tangent_dim
        self.fc_expand_latent = nn.Linear(
            self.latent_dim,
            self.num_gaad_regions * self.initial_gen_wubu_dim
        )

        wubu_g_config = _configure_wubu_stack(args, "wubu_g")
        if wubu_g_config is None or wubu_g_config["num_levels"] == 0:
             self.logger.warning("WuBu-G config is None or num_levels is 0. Generator using MLP fallback.")
             self.wubu_generator = nn.Sequential(
                 nn.Linear(self.initial_gen_wubu_dim, self.initial_gen_wubu_dim * 2),
                 nn.GELU(),
                 nn.LayerNorm(self.initial_gen_wubu_dim * 2),
                 nn.Linear(self.initial_gen_wubu_dim * 2, self.num_dct_coeffs_flat)
             )
        else:
            self.wubu_generator = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=self.initial_gen_wubu_dim,
                output_tangent_dim=self.num_dct_coeffs_flat,
                config=wubu_g_config
            )

        if self.args.dct_norm_type == "tanh":
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

        self.apply(init_weights_general)
        self.logger.info(f"AudioSpecGenerator initialized. Outputting {self.num_gaad_regions} regions, "
                         f"each with {self.num_dct_coeffs_flat} DCT coeffs.")

    @staticmethod
    def _unnormalize_dct(norm_dct_coeffs: torch.Tensor, args_ref: argparse.Namespace) -> torch.Tensor:
        # norm_dct_coeffs: (..., F_proc, T_proc) or (..., num_dct_coeffs_flat)
        # This needs to be the inverse of the normalization in AudioSpecEncoder._apply_dct_and_normalize
        if args_ref.dct_norm_type == "none":
            return norm_dct_coeffs
        elif args_ref.dct_norm_type == "global_scale":
            return norm_dct_coeffs * args_ref.dct_norm_global_scale
        elif args_ref.dct_norm_type == "tanh":
            # If generator output Y = tanh(D_orig / S), then D_orig = atanh(Y) * S.
            # The generator's final_activation is Tanh, so norm_dct_coeffs is Y.
            # This means to get D_orig scale, we need atanh(norm_dct_coeffs) * scale.
            # This is a more complex inverse and depends on how loss is calculated.
            # If loss is on the tanh-scaled space, this unnormalization is for visualization/assembly.
            # For now, if generator output IS Tanh, assume it is already in the target "normalized" domain
            # for reconstruction loss comparisons. Actual unscaling to original DCT values would require atanh.
            # This function is for getting back to the *original domain of DCT coefficients*
            # before any normalization was applied by the encoder.
            # If the generator's self.final_activation is Tanh, then `norm_dct_coeffs` is already
            # in the Tanh-compressed space. To reverse *that* plus the scaling:
            if torch.is_tensor(norm_dct_coeffs): # Check if it is a tensor before atanh
                # Clamp to avoid atanh(+-1) = inf
                clamped_coeffs = torch.clamp(norm_dct_coeffs, -1.0 + EPS, 1.0 - EPS)
                unscaled_dct = torch.atanh(clamped_coeffs) * args_ref.dct_norm_tanh_scale
                return unscaled_dct
            else: # Should not happen if called correctly
                logging.warning("_unnormalize_dct (tanh): input not a tensor. Returning as is.")
                return norm_dct_coeffs
        else: # Default to global scale for safety
            return norm_dct_coeffs * args_ref.dct_norm_global_scale


    def forward(self, latent_code: torch.Tensor) -> torch.Tensor:
        B = latent_code.shape[0]

        expanded_z = self.fc_expand_latent(latent_code) # (B, NumRegions * initial_gen_wubu_dim)
        wubu_gen_input = expanded_z.view(B * self.num_gaad_regions, self.initial_gen_wubu_dim)
        generated_flat_dct_coeffs = self.wubu_generator(wubu_gen_input) # (B * NumRegions, num_dct_coeffs_flat)
        generated_flat_dct_coeffs_activated = self.final_activation(generated_flat_dct_coeffs)

        # Reshape to structured output: (B, NumRegions, F_proc, T_proc)
        F_proc, T_proc = self.region_proc_size[1], self.region_proc_size[0]
        generated_dct_structured = generated_flat_dct_coeffs_activated.view(
            B, self.num_gaad_regions, F_proc, T_proc
        )
        return generated_dct_structured


class AudioSpecDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.audio_config = audio_config
        self.gaad_config = gaad_config
        self.disc_config = disc_config
        self.logger = logging.getLogger("WuBuSpecTransV01.Discriminator")

        self.num_gaad_regions = gaad_config['num_regions']
        self.region_proc_size = (args.region_proc_size_t, args.region_proc_size_f)
        self.num_dct_coeffs_flat = self.region_proc_size[0] * self.region_proc_size[1]

        self.input_type = disc_config.get("input_type", "mel")
        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", True)
        self.logger.info(f"AudioSpecDiscriminator initialized. Input type: {self.input_type}")

        if self.input_type == "dct":
            wubu_d_input_dim = args.encoder_initial_tangent_dim
            self.dct_coeff_embed_disc = DCTCoeffEmbed(
                num_dct_coeffs_per_region=self.num_dct_coeffs_flat,
                embed_dim=wubu_d_input_dim
            )
            wubu_d_config = _configure_wubu_stack(args, "wubu_d")
            if wubu_d_config is None or wubu_d_config["num_levels"] == 0:
                self.logger.warning("WuBu-D config is None or num_levels is 0. Discriminator (DCT input) using MLP fallback.")
                self.feature_extractor = nn.Sequential(
                    nn.Linear(wubu_d_input_dim, wubu_d_input_dim * 2), nn.LeakyReLU(0.2, True), nn.LayerNorm(wubu_d_input_dim * 2),
                    nn.Linear(wubu_d_input_dim * 2, audio_config['wubu_d_output_dim'])
                )
            else:
                self.feature_extractor = FullyHyperbolicWuBuNestingModel(
                    input_tangent_dim=wubu_d_input_dim,
                    output_tangent_dim=audio_config['wubu_d_output_dim'],
                    config=wubu_d_config
                )
            self.final_fc = nn.Linear(audio_config['wubu_d_output_dim'], 1)

        elif self.input_type == "mel":
            n_mels_total = args.n_mels
            n_time_total = audio_config.get("num_time_frames_for_1s_segment", 86)
            min_input_dim = min(n_mels_total, n_time_total)
            num_spatial_downsamples_target = int(math.log2(min_input_dim / 4.0)) if min_input_dim >=8 else 1
            max_possible_downsamples = int(math.log2(min_input_dim)) if min_input_dim > 0 else 0
            num_downsamples = max(1, min(num_spatial_downsamples_target, max_possible_downsamples))
            base_ch = disc_config.get("base_disc_channels", 64)
            max_ch = disc_config.get("max_disc_channels", 512)
            cnn_layers = []
            in_c = 1 # Mel spectrograms are single channel
            curr_h, curr_w = n_mels_total, n_time_total

            for i in range(num_downsamples):
                out_c = min(base_ch * (2**i), max_ch)
                conv_l = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
                if self.apply_spectral_norm: cnn_layers.append(spectral_norm(conv_l))
                else: cnn_layers.append(conv_l)
                cnn_layers.append(nn.InstanceNorm2d(out_c, affine=True))
                cnn_layers.append(nn.LeakyReLU(0.2, inplace=True))
                in_c = out_c
                curr_h = (curr_h + 2*1 - 4) // 2 + 1
                curr_w = (curr_w + 2*1 - 4) // 2 + 1
            
            self.feature_extractor = nn.Sequential(*cnn_layers)
            # Ensure curr_h and curr_w are at least 1 for the kernel
            final_kernel_h = max(1, curr_h)
            final_kernel_w = max(1, curr_w)
            final_conv_layer = nn.Conv2d(in_c, 1, kernel_size=(final_kernel_h, final_kernel_w), stride=1, padding=0)
            if self.apply_spectral_norm: self.final_fc = spectral_norm(final_conv_layer)
            else: self.final_fc = final_conv_layer
        else:
            raise ValueError(f"Unsupported discriminator input_type: {self.input_type}")

        self.apply(init_weights_general)

    def _assemble_mel_from_dct_regions(self, dct_regions: torch.Tensor, gaad_bboxes: torch.Tensor,
                                       target_mel_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        # dct_regions: (B, NumRegions, F_proc, T_proc) - These should be UNNORMALIZED DCT coeffs
        B, N_Reg, F_p, T_p = dct_regions.shape
        _, C_target, H_target, W_target = target_mel_shape # H=N_Mels, W=N_Time
        device = dct_regions.device
        dtype = dct_regions.dtype

        if idct_2d is None or not TORCH_DCT_AVAILABLE:
            self.logger.error("idct_2d function is not available. Cannot assemble Mel. Returning zeros.")
            return torch.zeros(target_mel_shape, device=device, dtype=dtype)

        dct_regions_flat = dct_regions.reshape(-1, F_p, T_p)
        spatial_regions_flat = idct_2d(dct_regions_flat) # (B*N_Reg, F_p, T_p)
        spatial_regions = spatial_regions_flat.reshape(B, N_Reg, F_p, T_p)

        assembled_mel_canvas = torch.zeros(target_mel_shape, device=device, dtype=dtype)
        counts_canvas = torch.zeros(target_mel_shape, device=device, dtype=dtype)

        for b in range(B):
            for r in range(N_Reg):
                t1, f1, t2, f2 = gaad_bboxes[b, r].round().int().tolist()
                t1_c, f1_c = max(0, t1), max(0, f1)
                t2_c, f2_c = min(W_target, t2), min(H_target, f2)

                if t1_c >= t2_c or f1_c >= f2_c: continue

                current_spatial_region = spatial_regions[b, r, :, :].unsqueeze(0).unsqueeze(0) # (1,1,F_p,T_p)
                target_h_bbox = f2_c - f1_c
                target_w_bbox = t2_c - t1_c

                if target_h_bbox <=0 or target_w_bbox <=0: continue

                resized_region = TF.resize(current_spatial_region, (target_h_bbox, target_w_bbox),
                                           interpolation=T.InterpolationMode.BILINEAR, antialias=True)
                
                assembled_mel_canvas[b, 0, f1_c:f2_c, t1_c:t2_c] += resized_region.squeeze(0).squeeze(0)
                counts_canvas[b, 0, f1_c:f2_c, t1_c:t2_c] += 1
        
        assembled_mel_canvas = torch.where(counts_canvas > 0, assembled_mel_canvas / counts_canvas, assembled_mel_canvas)
        return assembled_mel_canvas

    def forward(self, input_data: torch.Tensor,
                      gaad_bboxes_for_assembly: Optional[torch.Tensor] = None,
                      target_mel_shape_for_assembly: Optional[Tuple[int,int,int,int]] = None
                     ) -> torch.Tensor:
        B = input_data.shape[0]
        
        if self.input_type == "dct":
            # input_data: (B, NumRegions, F_proc, T_proc) - assumes normalized DCTs from generator/encoder
            flat_dct_coeffs = input_data.reshape(B, self.num_gaad_regions, -1)
            embedded_coeffs = self.dct_coeff_embed_disc(flat_dct_coeffs)
            
            wubu_d_input = embedded_coeffs.reshape(B * self.num_gaad_regions, -1)
            wubu_d_features_flat = self.feature_extractor(wubu_d_input)
            
            aggregated_features = wubu_d_features_flat.reshape(B, self.num_gaad_regions, -1).mean(dim=1)
            logits = self.final_fc(aggregated_features)

        elif self.input_type == "mel":
            mel_input_for_d: torch.Tensor
            if input_data.ndim == 4 and input_data.shape[1] == self.num_gaad_regions: # Input is DCTs (B, N_Reg, F_p, T_p)
                if gaad_bboxes_for_assembly is None or target_mel_shape_for_assembly is None:
                    raise ValueError("GAAD bboxes and target_mel_shape needed for D when input_type='mel' but DCTs provided.")
                
                # input_data here are the *normalized* DCTs from the generator.
                # We need to unnormalize them to their original scale before IDCT.
                unnorm_dct_coeffs_for_assembly = AudioSpecGenerator._unnormalize_dct(
                    input_data.to(self.final_fc.weight.device), self.args
                )
                
                mel_input_for_d = self._assemble_mel_from_dct_regions(
                    unnorm_dct_coeffs_for_assembly, 
                    gaad_bboxes_for_assembly, 
                    target_mel_shape_for_assembly
                )
            elif input_data.ndim == 4 and input_data.shape[1] == 1: # Already a Mel spectrogram (B, 1, H, W)
                mel_input_for_d = input_data
            else:
                raise ValueError(f"Unsupported input_data shape for D (mel type): {input_data.shape}")

            features = self.feature_extractor(mel_input_for_d)
            logits = self.final_fc(features) # Output: (B, 1, 1, 1)
            logits = logits.view(B, -1) # Flatten to (B, num_patches_or_1)
            if logits.shape[1] > 1: # If multiple patches (PatchGAN style), average them
                logits = torch.mean(logits, dim=1, keepdim=True)
        else:
            raise NotImplementedError(f"Discriminator forward not implemented for type {self.input_type}")
        
        return logits

# =====================================================================
# VAE-GAN Model Components
# =====================================================================
class WuBuSpecTransNet(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict,
                 wubu_s_config_enc: Dict, wubu_g_config_gen: Optional[Dict]):
        super().__init__()
        self.args = args
        self.audio_config = audio_config
        self.gaad_config = gaad_config
        self.wubu_s_config_enc = wubu_s_config_enc
        self.logger = logging.getLogger("WuBuSpecTransV01.MainNet")
        self.latent_dim = args.latent_dim

        self.encoder = AudioSpecEncoder(args, audio_config, gaad_config, wubu_s_config_enc, self.latent_dim)
        self.generator = AudioSpecGenerator(args, audio_config, gaad_config, self.latent_dim)

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"WuBuSpecTransNet Initialized: {param_count:,} params.")

    def encode(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # mu, logvar, norm_dct_coeffs_target (from encoder), gaad_bboxes
        return self.encoder(mel_spectrogram)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Output: (B, NumRegions, F_proc, T_proc) - normalized DCTs
        return self.generator(z)

    def forward(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, norm_dct_coeffs_target, gaad_bboxes_from_enc = self.encode(mel_spectrogram)
        z = self.reparameterize(mu, logvar)
        recon_norm_dct_coeffs = self.decode(z) # G(z), which is G(E(X))
        
        # Returns:
        # 1. recon_norm_dct_coeffs (from G(E(X)))
        # 2. mu (from E(X))
        # 3. logvar (from E(X))
        # 4. gaad_bboxes_from_enc (from E(X), can be used for D if it's on Mels, or visualization)
        # 5. norm_dct_coeffs_target (from E(X), this is the target for reconstruction loss)
        # The trainer step will need to call encode separately to get the target DCTs.
        # Re-designing forward to return target DCTs too for simplicity in trainer.
        return recon_norm_dct_coeffs, mu, logvar, gaad_bboxes_from_enc, norm_dct_coeffs_target


# =====================================================================
# Dataset
# =====================================================================
class AudioSegmentDataset(Dataset):
    def __init__(self,
                 audio_file_paths: List[str],
                 args: argparse.Namespace,
                 segment_duration_sec: float = 1.0,
                 segment_overlap_sec: float = 0.0,
                 data_fraction: float = 1.0,
                 preload_to_ram: bool = False):
        super().__init__()
        self.audio_file_paths = audio_file_paths
        self.args = args
        self.segment_duration_sec = segment_duration_sec
        self.segment_overlap_sec = segment_overlap_sec
        self.sample_rate = args.sample_rate
        self.n_fft = args.n_fft
        self.hop_length = args.hop_length
        self.n_mels = args.n_mels
        self.logger = logging.getLogger("WuBuSpecTransV01.Dataset")
        
        self.segments_info: List[Tuple[int, int, int]] = []
        self.preloaded_mel_spectrograms: Optional[List[torch.Tensor]] = [] if preload_to_ram else None

        segment_samples = int(self.segment_duration_sec * self.sample_rate)
        overlap_samples = int(self.segment_overlap_sec * self.sample_rate)
        step_samples = segment_samples - overlap_samples

        for file_idx, audio_path_str in enumerate(tqdm(audio_file_paths, desc="Processing audio files for dataset")):
            audio_path = Path(audio_path_str)
            if not audio_path.is_file():
                self.logger.warning(f"Audio file not found: {audio_path}. Skipping.")
                continue
            try:
                waveform, sr_orig = librosa.load(audio_path, sr=None, mono=True)
                if sr_orig != self.sample_rate:
                    waveform = librosa.resample(waveform, orig_sr=sr_orig, target_sr=self.sample_rate)
                
                num_wav_samples = waveform.shape[0]
                current_pos = 0
                while current_pos + segment_samples <= num_wav_samples:
                    start_s = current_pos
                    end_s = current_pos + segment_samples
                    self.segments_info.append((file_idx, start_s, end_s))
                    
                    if preload_to_ram and self.preloaded_mel_spectrograms is not None:
                        segment_audio = waveform[start_s:end_s]
                        mel_spec = self._audio_to_mel_spectrogram(segment_audio)
                        self.preloaded_mel_spectrograms.append(mel_spec)
                    current_pos += step_samples
            except Exception as e:
                self.logger.error(f"Error processing audio file {audio_path}: {e}", exc_info=True)
        
        if data_fraction < 1.0 and len(self.segments_info) > 1:
            num_to_keep = max(1, int(len(self.segments_info) * data_fraction))
            sampled_indices = random.sample(range(len(self.segments_info)), num_to_keep)
            self.segments_info = [self.segments_info[i] for i in sampled_indices]
            if preload_to_ram and self.preloaded_mel_spectrograms is not None:
                self.preloaded_mel_spectrograms = [self.preloaded_mel_spectrograms[i] for i in sampled_indices]
            self.logger.info(f"Using {data_fraction*100:.1f}% of segments: {len(self.segments_info)} segments.")

        if not self.segments_info:
            self.logger.error("AudioSegmentDataset: No valid audio segments found or processed.")
            raise ValueError("No audio segments available for the dataset.")
        
        self.audio_waveforms_cache: Dict[str, np.ndarray] = {}
        if not preload_to_ram:
            self.logger.info("Caching full waveforms to RAM (not preloading Mel spectrograms)...")
            for audio_path_str_cache in tqdm(self.audio_file_paths, desc="Caching waveforms"):
                 audio_path_cache = Path(audio_path_str_cache)
                 if audio_path_cache.is_file():
                     try:
                        waveform, sr_orig = librosa.load(audio_path_cache, sr=None, mono=True)
                        if sr_orig != self.sample_rate:
                            waveform = librosa.resample(waveform, orig_sr=sr_orig, target_sr=self.sample_rate)
                        self.audio_waveforms_cache[str(audio_path_cache)] = waveform
                     except Exception as e:
                        self.logger.error(f"Error pre-caching waveform for {audio_path_cache}: {e}")

        self.logger.info(f"AudioSegmentDataset initialized. Total segments: {len(self.segments_info)}. "
                         f"Segment duration: {self.segment_duration_sec}s. Preloaded Mels: {preload_to_ram}")

    def _audio_to_mel_spectrogram(self, audio_segment: np.ndarray) -> torch.Tensor:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmin=self.args.fmin, fmax=self.args.fmax if self.args.fmax is not None else self.sample_rate/2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_db_normalized = (mel_spec_db - self.args.db_norm_min) / (self.args.db_norm_max - self.args.db_norm_min)
        mel_spec_db_normalized = np.clip(mel_spec_db_normalized, 0.0, 1.0)
        mel_spec_db_normalized = (mel_spec_db_normalized * 2.0) - 1.0 # to [-1,1]
        
        return torch.from_numpy(mel_spec_db_normalized).float().unsqueeze(0) # (1, n_mels, time_frames)

    def __len__(self) -> int:
        return len(self.segments_info)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.preloaded_mel_spectrograms is not None:
            return self.preloaded_mel_spectrograms[idx]
        
        file_idx, start_sample, end_sample = self.segments_info[idx]
        audio_path_str = self.audio_file_paths[file_idx]
        
        waveform = self.audio_waveforms_cache.get(audio_path_str)
        if waveform is None:
             self.logger.error(f"Waveform for {audio_path_str} not found in cache. This indicates an issue.")
             # Fallback: load on the fly, though this is slow and unexpected
             try:
                 waveform, sr_orig = librosa.load(Path(audio_path_str), sr=None, mono=True)
                 if sr_orig != self.sample_rate:
                     waveform = librosa.resample(waveform, orig_sr=sr_orig, target_sr=self.sample_rate)
                 # Optionally add to cache here if desired, but it signals a problem
             except Exception as e_load:
                self.logger.error(f"CRITICAL: Failed to load waveform {audio_path_str} on demand: {e_load}")
                # Return a zero tensor as a last resort to prevent crash
                num_time_frames = math.ceil(int(self.segment_duration_sec * self.sample_rate) / self.hop_length)
                return torch.zeros((1, self.n_mels, num_time_frames), dtype=torch.float)


        segment_audio = waveform[start_sample:end_sample]
        mel_spectrogram_tensor = self._audio_to_mel_spectrogram(segment_audio)
        return mel_spectrogram_tensor


# =====================================================================
# VAE-GAN Trainer
# =====================================================================
class HybridTrainer:
    def __init__(self,
                 model: "WuBuSpecTransNet",
                 discriminator: "AudioSpecDiscriminator",
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
        self.logger = logging.getLogger("WuBuSpecTransV01.Trainer")

        self.audio_config = getattr(model, 'audio_config', {})
        self.gaad_config = getattr(model, 'gaad_config', {})

        self.lambda_recon = args.lambda_recon
        self.lambda_kl = args.lambda_kl # This can be dynamically updated by Q-controller
        self.lambda_gan = args.lambda_gan

        self.scaler_enc_gen = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.scaler_disc = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')

        self.global_step = 0; self.current_epoch = 0
        self.best_val_metric_val = -float('inf') if args.val_primary_metric in ["avg_val_ssim_mel", "avg_val_psnr_mel"] else float('inf')
        self.last_val_metrics: Dict[str, Any] = {}

        if self.am_main_process: os.makedirs(args.checkpoint_dir, exist_ok=True)

        self.lpips_loss_fn: Optional[lpips.LPIPS] = None
        self.ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None

        if self.am_main_process and self.args.use_lpips_for_mel_verification:
             if LPIPS_AVAILABLE and lpips is not None:
                 try:
                     self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)
                     self.logger.info("LPIPS metric (for Mel) enabled.")
                 except Exception as e_lpips:
                     self.logger.warning(f"LPIPS init failed: {e_lpips}. Disabling LPIPS.")
                     self.lpips_loss_fn = None
             else: self.logger.warning("LPIPS lib unavailable. Skip LPIPS.")

        if self.am_main_process and TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
             try:
                 self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device) # data_range for [0,1] Mels
                 self.logger.info("SSIM metric (for Mel) enabled.")
             except Exception as e:
                 self.logger.warning(f"SSIM init failed: {e}. Skip SSIM.")
                 self.ssim_metric = None
        elif self.am_main_process: self.logger.warning("torchmetrics SSIM unavailable. Skip SSIM.")
        
        global MATPLOTLIB_AVAILABLE, plt
        try:
            import matplotlib.pyplot as plt
            MATPLOTLIB_AVAILABLE = True
        except ImportError:
            plt = None
            MATPLOTLIB_AVAILABLE = False
            if self.am_main_process:
                self.logger.warning("Matplotlib not available. Mel spectrogram WandB viz will be basic.")

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        if self.grad_accum_steps > 1 and self.am_main_process:
            self.logger.info(f"Gradient accumulation enabled: {self.grad_accum_steps} steps.")
        self.fixed_noise_for_sampling: Optional[torch.Tensor] = None

        self.lambda_kl_update_interval = getattr(args, 'lambda_kl_update_interval', 0)
        self.lambda_kl_q_controller: Optional[HAKMEMQController] = None
        if self.args.q_controller_enabled and self.lambda_kl_update_interval > 0:
            q_cfg_lambda_kl = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
            # Adjust lambda_kl_scale_options if needed, or use default
            q_cfg_lambda_kl["lambda_kl_scale_options"] = [0.90, 0.95, 1.0, 1.05, 1.10] # Example adjustment
            self.lambda_kl_q_controller = HAKMEMQController(**q_cfg_lambda_kl)
            if self.am_main_process:
                self.logger.info(f"Separate Lambda_KL Q-Control ENABLED. Update interval: {self.lambda_kl_update_interval} global steps.")
            if hasattr(self.lambda_kl_q_controller, 'set_current_lambda_kl'):
                self.lambda_kl_q_controller.set_current_lambda_kl(self.lambda_kl)
        elif self.am_main_process:
            if not self.args.q_controller_enabled: self.logger.info("Lambda_KL Q-Control DISABLED (Q-controller globally disabled).")
            elif self.lambda_kl_update_interval <= 0: self.logger.info("Lambda_KL Q-Control DISABLED (lambda_kl_update_interval <= 0).")
        
        self.interval_metrics_accum: Dict[str, float] = defaultdict(float)
        self.interval_steps_count = 0
        self.prev_interval_metrics_for_lambda_kl_reward: Optional[Dict[str, Union[float, None]]] = None
        self.min_lambda_kl_q_control = getattr(args, 'min_lambda_kl_q_control', 1e-6)
        self.max_lambda_kl_q_control = getattr(args, 'max_lambda_kl_q_control', 1.0)

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()

    def _compute_recon_loss(self, recon_norm_dcts: torch.Tensor, target_norm_dcts: torch.Tensor) -> torch.Tensor:
        # Both inputs are (B, NumRegions, F_proc, T_proc) - normalized DCT coefficients
        return F.mse_loss(recon_norm_dcts, target_norm_dcts)

    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, mel_spectrograms_to_log: Optional[torch.Tensor],
                              num_sequences_to_log_max: int = 2):
        if not (self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run): return
        if mel_spectrograms_to_log is None or mel_spectrograms_to_log.numel() == 0:
            self.logger.debug(f"Skipping WandB image log for {tag_prefix} due to None or empty data.")
            return

        B_log, _, H_log, W_log = mel_spectrograms_to_log.shape # H_log=Freq, W_log=Time
        num_to_actually_log = min(B_log, num_sequences_to_log_max)
        wandb_images_for_log = []
        
        fig_created = False
        if MATPLOTLIB_AVAILABLE and plt is not None and librosa is not None:
            # Adjust figsize dynamically based on aspect ratio, ensure reasonable min size
            aspect_ratio = W_log / H_log if H_log > 0 and W_log > 0 else 1.0
            fig_width = max(5, min(15, 5 * aspect_ratio))
            fig_height = max(4, min(10, fig_width / aspect_ratio if aspect_ratio > 0 else fig_width))
            try:
                fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
                fig_created = True
            except Exception as e_plt_create:
                self.logger.warning(f"Failed to create matplotlib figure for WandB: {e_plt_create}")
                fig, ax = None, None # Fallback

        for b_idx in range(num_to_actually_log):
            mel_tensor = mel_spectrograms_to_log[b_idx, 0, ...].cpu().float() # Remove channel dim
            
            if fig_created and fig is not None and ax is not None:
                ax.clear()
                try:
                    # The mel_tensor is already normalized to [-1,1]. Librosa display works with this.
                    # Colorbar should reflect that it's a normalized scale, not absolute dB.
                    img = librosa.display.specshow(mel_tensor.numpy(), ax=ax,
                                             sr=self.args.sample_rate, hop_length=self.args.hop_length,
                                             x_axis='time', y_axis='mel',
                                             fmin=self.args.fmin, fmax=self.args.fmax if self.args.fmax else self.args.sample_rate/2.0)
                    if hasattr(fig, 'colorbar') and not ax.images[-1].colorbar: # Check if colorbar exists for the image
                         fig.colorbar(img, ax=ax, format='%+.2f (norm)')
                    ax.set_title(f"{tag_prefix} S{b_idx} Ep{self.current_epoch+1} GStep{self.global_step}")
                    wandb_images_for_log.append(wandb.Image(fig))
                except Exception as e_disp:
                    self.logger.warning(f"Librosa display failed for {tag_prefix} S{b_idx}: {e_disp}. Falling back.")
                    img_0_1 = (mel_tensor.clamp(-1,1) + 1) / 2.0 # Fallback
                    caption = f"{tag_prefix} S{b_idx} Ep{self.current_epoch+1} GStep{self.global_step} (raw_fallback)"
                    wandb_images_for_log.append(wandb.Image(img_0_1, caption=caption))
            else: # Fallback to simple image
                img_0_1 = (mel_tensor.clamp(-1,1) + 1) / 2.0
                caption = f"{tag_prefix} S{b_idx} Ep{self.current_epoch+1} GStep{self.global_step} (raw)"
                wandb_images_for_log.append(wandb.Image(img_0_1, caption=caption))

        if fig_created and fig is not None and plt is not None: plt.close(fig)

        if wandb_images_for_log:
            try:
                wandb.log({f"samples_mel/{tag_prefix}": wandb_images_for_log}, step=self.global_step)
            except Exception as e_wandb_log:
                self.logger.error(f"Failed to log images to WandB for {tag_prefix}: {e_wandb_log}")


    def _train_discriminator_step(self, real_mel_spectrograms: torch.Tensor,
                                  m_ref: "WuBuSpecTransNet", d_ref: "AudioSpecDiscriminator"
                                  ) -> Dict[str, torch.Tensor]:
        B = real_mel_spectrograms.shape[0]; device = real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0, device=device)).dtype
        
        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model)
        fake_labels = torch.zeros(B, 1, device=device, dtype=dtype_model)
        losses_d_micro: Dict[str, torch.Tensor] = {}

        for p in d_ref.parameters(): p.requires_grad = True
        for p in m_ref.parameters(): p.requires_grad = False
        
        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            # --- REAL DISCRIMINATION ---
            if d_ref.input_type == "mel":
                real_input_for_d = real_mel_spectrograms.to(device, dtype=model)
                real_logits = d_ref(real_input_for_d)
            elif d_ref.input_type == "dct":
                with torch.no_grad(): # Encoder part is not part of D's training
                    _, _, real_norm_dcts_target, _ = m_ref.encode(real_mel_spectrograms.to(device, dtype_model))
                real_input_for_d = real_norm_dcts_target.to(device, dtype_model)
                real_logits = d_ref(real_input_for_d)
            else:
                raise ValueError(f"Unsupported D input type for training: {d_ref.input_type}")
            loss_d_real = self.adversarial_loss(real_logits, real_labels)

            # --- FAKE DISCRIMINATION ---
            with torch.no_grad(): # Generator part is not part of D's training
                # Get generated (normalized) DCTs and bboxes
                # m_ref.forward returns (recon_norm_dct_coeffs, mu, logvar, gaad_bboxes, target_norm_dcts)
                fake_norm_dct_coeffs, _, _, gaad_bboxes_for_assembly, _ = m_ref(real_mel_spectrograms.to(device, dtype_model))

            if d_ref.input_type == "mel":
                # D needs assembled Mel. fake_norm_dct_coeffs are from G(E(x))
                # These are already in the "normalized" domain if generator's Tanh is used.
                # _assemble_mel_from_dct_regions expects *unnormalized* DCTs if it's to match original signal stats.
                # So, we unnormalize the generator's output.
                unnorm_fake_dcts = AudioSpecGenerator._unnormalize_dct(
                    fake_norm_dct_coeffs.detach(), self.args
                )
                fake_mel_input_for_d = d_ref._assemble_mel_from_dct_regions(
                    unnorm_fake_dcts, gaad_bboxes_for_assembly.detach(),
                    real_mel_spectrograms.shape
                )
                fake_logits = d_ref(fake_mel_input_for_d.detach())
            elif d_ref.input_type == "dct":
                # D operates directly on normalized DCTs from the generator
                fake_input_for_d = fake_norm_dct_coeffs.to(device, dtype_model).detach()
                fake_logits = d_ref(fake_input_for_d)
            else:
                raise ValueError(f"Unsupported D input type for training (fake): {d_ref.input_type}")

            loss_d_fake = self.adversarial_loss(fake_logits, fake_labels)
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total_scaled_for_accum_micro = loss_d_total_micro / self.grad_accum_steps

        self.scaler_disc.scale(loss_d_total_scaled_for_accum_micro).backward()
        
        losses_d_micro['loss_d_real_micro'] = loss_d_real.detach()
        losses_d_micro['loss_d_fake_micro'] = loss_d_fake.detach()
        losses_d_micro['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d_micro

    def _train_generator_step(self, real_mel_spectrograms: torch.Tensor,
                              m_ref: "WuBuSpecTransNet", d_ref: "AudioSpecDiscriminator"
                              ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        B = real_mel_spectrograms.shape[0]; device = real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0, device=device)).dtype
        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model)
        losses_g_micro: Dict[str, torch.Tensor] = {}
        recon_mel_for_log: Optional[torch.Tensor] = None

        for p in d_ref.parameters(): p.requires_grad = False
        for p in m_ref.parameters(): p.requires_grad = True

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            # Full VAE-GAN forward pass for G
            # recon_norm_dct_coeffs is G(E(X))
            # target_norm_dct_coeffs is E(X)
            # gaad_bboxes_from_enc are bboxes from E(X)
            recon_norm_dct_coeffs, mu, logvar, gaad_bboxes_from_enc, target_norm_dct_coeffs = \
                m_ref(real_mel_spectrograms.to(device,dtype_model))

            loss_recon = self._compute_recon_loss(recon_norm_dct_coeffs, target_norm_dct_coeffs)
            loss_kl = self._compute_kl_loss(mu, logvar)
            
            # Adversarial loss for generator
            if d_ref.input_type == "mel":
                # Unnormalize the generator's output (normalized DCTs) for assembly
                unnorm_recon_dcts_for_adv = AudioSpecGenerator._unnormalize_dct(
                    recon_norm_dct_coeffs, self.args
                )
                recon_mel_for_adv = d_ref._assemble_mel_from_dct_regions(
                    unnorm_recon_dcts_for_adv, gaad_bboxes_from_enc,
                    real_mel_spectrograms.shape
                )
                log_condition_met = (
                    self.am_main_process and
                    self.args.wandb_log_train_recon_interval > 0 and
                    ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0 or self.global_step == 0)
                )
                if log_condition_met :
                    recon_mel_for_log = recon_mel_for_adv.detach().clone()
                fake_logits_gen = d_ref(recon_mel_for_adv) # Feed assembled Mel to D
            elif d_ref.input_type == "dct":
                # D operates directly on generator's output (normalized DCTs)
                fake_logits_gen = d_ref(recon_norm_dct_coeffs)
            else:
                raise ValueError(f"Unsupported D input type for G training: {d_ref.input_type}")
            
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
        return losses_g_micro, recon_mel_for_log

    def train(self, start_epoch:int=0, initial_global_step:int=0):
        self.global_step = initial_global_step
        self.current_epoch = start_epoch
        if self.am_main_process:
            self.logger.info(f"Starting training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch}, InitialGStep: {initial_global_step}")
            self.logger.info(f"Grad Accum: {self.grad_accum_steps}, Initial Lambda_KL: {self.lambda_kl:.3e}, Lambda_Recon: {self.lambda_recon}, Lambda_GAN: {self.lambda_gan}")
        
        if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and self.args.num_val_samples_to_log > 0:
            m_ref_temp = self.model.module if self.ddp_active else self.model
            default_dtype = next(iter(m_ref_temp.parameters()), torch.tensor(0.0, device=self.device)).dtype
            self.fixed_noise_for_sampling = torch.randn(
                self.args.num_val_samples_to_log, self.args.latent_dim, device=self.device, dtype=default_dtype
            )
            self.logger.info(f"Created fixed noise tensor for sampling: {self.fixed_noise_for_sampling.shape}")

        accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
        accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0
        log_interval_accum_losses: Dict[str, float] = defaultdict(float)
        log_interval_items_processed = 0
        
        m_ref = self.model.module if self.ddp_active else self.model
        d_ref = self.discriminator.module if self.ddp_active else self.discriminator
        
        opt_gen_q_controller = getattr(self.optimizer_enc_gen, 'q_controller', None)
        opt_disc_q_controller = getattr(self.optimizer_disc, 'q_controller', None)
        
        if opt_gen_q_controller and hasattr(opt_gen_q_controller, 'set_current_lambda_kl'):
            opt_gen_q_controller.set_current_lambda_kl(self.lambda_kl)
        if opt_disc_q_controller and hasattr(opt_disc_q_controller, 'set_current_lambda_kl'):
            opt_disc_q_controller.set_current_lambda_kl(self.lambda_kl)

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process:
                self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting (current lambda_kl: {self.lambda_kl:.3e})...")
            if self.ddp_active and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch) # type: ignore
            
            m_ref.train()
            d_ref.train()
            self.optimizer_disc.zero_grad(set_to_none=True)
            self.optimizer_enc_gen.zero_grad(set_to_none=True)
            
            num_batches_epoch = len(self.train_loader)
            prog_bar_disabled = not self.am_main_process or os.getenv('CI') == 'true'
            prog_bar = tqdm(self.train_loader, desc=f"E{epoch+1}", disable=prog_bar_disabled, dynamic_ncols=True, total=num_batches_epoch)

            for batch_idx, batch_mel_segments in enumerate(prog_bar):
                batch_mel_segments = batch_mel_segments.to(self.device)
                batch_size_micro = batch_mel_segments.size(0)

                # --- Train Discriminator ---
                losses_d_micro = self._train_discriminator_step(batch_mel_segments, m_ref, d_ref)
                if torch.isfinite(losses_d_micro['loss_d_total_micro']):
                    d_total_val = losses_d_micro['loss_d_total_micro'].item()
                    accum_d_total_q += d_total_val
                    self.interval_metrics_accum['d_total'] += d_total_val
                if torch.isfinite(losses_d_micro['loss_d_real_micro']):
                    accum_d_real_q += losses_d_micro['loss_d_real_micro'].item()
                if torch.isfinite(losses_d_micro['loss_d_fake_micro']):
                    accum_d_fake_q += losses_d_micro['loss_d_fake_micro'].item()
                for k,v_tensor in losses_d_micro.items():
                    if torch.isfinite(v_tensor):
                        log_interval_accum_losses[k.replace('_micro','_agg')] += v_tensor.item() * batch_size_micro

                # --- Train Generator (Encoder + Decoder) ---
                losses_g_micro, recon_mel_for_logging = self._train_generator_step(batch_mel_segments, m_ref, d_ref)
                if torch.isfinite(losses_g_micro['loss_g_total_micro']):
                    accum_g_total_q += losses_g_micro['loss_g_total_micro'].item()
                if torch.isfinite(losses_g_micro['loss_recon_micro']):
                    recon_val = losses_g_micro['loss_recon_micro'].item()
                    accum_g_recon_q += recon_val
                    self.interval_metrics_accum['recon_dct'] += recon_val
                if torch.isfinite(losses_g_micro['loss_kl_micro']):
                    kl_val = losses_g_micro['loss_kl_micro'].item()
                    accum_g_kl_q += kl_val
                    self.interval_metrics_accum['kl_div'] += kl_val
                if torch.isfinite(losses_g_micro['loss_g_adv_micro']):
                    accum_g_adv_q += losses_g_micro['loss_g_adv_micro'].item()
                for k,v_tensor in losses_g_micro.items():
                    if torch.isfinite(v_tensor):
                        log_interval_accum_losses[k.replace('_micro','_agg')] += v_tensor.item() * batch_size_micro
                
                log_interval_items_processed += batch_size_micro
                self.interval_steps_count += 1

                # --- Gradient Accumulation Step ---
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if hasattr(self.optimizer_disc, 'grad_stats') and hasattr(self.optimizer_disc.grad_stats, 'finalize_step_stats'):
                        self.optimizer_disc.grad_stats.finalize_step_stats(sum(p.numel() for grp in self.optimizer_disc.param_groups for p in grp['params'] if p.requires_grad)) # type: ignore
                    if hasattr(self.optimizer_enc_gen, 'grad_stats') and hasattr(self.optimizer_enc_gen.grad_stats, 'finalize_step_stats'):
                        self.optimizer_enc_gen.grad_stats.finalize_step_stats(sum(p.numel() for grp in self.optimizer_enc_gen.param_groups for p in grp['params'] if p.requires_grad)) # type: ignore
                    
                    avg_g_total_macro = accum_g_total_q / self.grad_accum_steps
                    avg_g_recon_macro = accum_g_recon_q / self.grad_accum_steps
                    avg_g_kl_macro = accum_g_kl_q / self.grad_accum_steps
                    avg_g_adv_macro = accum_g_adv_q / self.grad_accum_steps
                    avg_d_total_macro = accum_d_total_q / self.grad_accum_steps
                    avg_d_real_macro = accum_d_real_q / self.grad_accum_steps
                    avg_d_fake_macro = accum_d_fake_q / self.grad_accum_steps

                    # Discriminator optimizer step
                    for p in d_ref.parameters(): p.requires_grad = True # Ensure D grads are on
                    for p in m_ref.parameters(): p.requires_grad = False # G grads off for D step
                    if opt_disc_q_controller and hasattr(self.optimizer_disc, 'q_controller_update_and_set_hyperparams'):
                        losses_d_q_lr_mom = {'loss_g_total': avg_g_total_macro, 'loss_g_adv': avg_g_adv_macro,
                                             'loss_d_total': avg_d_total_macro, 'loss_d_real': avg_d_real_macro,
                                             'loss_d_fake': avg_d_fake_macro}
                        self.optimizer_disc.q_controller_update_and_set_hyperparams(losses_d_q_lr_mom, self.lambda_kl) # type: ignore
                    if self.args.global_max_grad_norm > 0:
                        self.scaler_disc.unscale_(self.optimizer_disc)
                        torch.nn.utils.clip_grad_norm_(d_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_disc.step(self.optimizer_disc)
                    self.scaler_disc.update()

                    # Generator optimizer step
                    for p in d_ref.parameters(): p.requires_grad = False # D grads off for G step
                    for p in m_ref.parameters(): p.requires_grad = True  # G grads on
                    if opt_gen_q_controller and hasattr(self.optimizer_enc_gen, 'q_controller_update_and_set_hyperparams'):
                        losses_g_q_lr_mom = {'loss_g_total': avg_g_total_macro, 'loss_g_recon': avg_g_recon_macro,
                                             'loss_g_kl': avg_g_kl_macro, 'loss_g_adv': avg_g_adv_macro,
                                             'loss_d_total': avg_d_total_macro}
                        self.optimizer_enc_gen.q_controller_update_and_set_hyperparams(losses_g_q_lr_mom, self.lambda_kl) # type: ignore
                    if self.args.global_max_grad_norm > 0:
                        self.scaler_enc_gen.unscale_(self.optimizer_enc_gen)
                        torch.nn.utils.clip_grad_norm_(m_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_enc_gen.step(self.optimizer_enc_gen)
                    self.scaler_enc_gen.update()

                    self.optimizer_disc.zero_grad(set_to_none=True)
                    self.optimizer_enc_gen.zero_grad(set_to_none=True)
                    self.global_step += 1
                    
                    accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
                    accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0

                    # --- Lambda_KL Q-Controller Update (if enabled) ---
                    if self.lambda_kl_q_controller is not None and self.lambda_kl_update_interval > 0 and \
                       self.global_step > 0 and self.global_step % self.lambda_kl_update_interval == 0 and \
                       self.interval_steps_count > 0:
                        
                        current_interval_metrics: Dict[str, Union[float, None]] = {
                            'avg_recon': self.interval_metrics_accum['recon_dct'] / self.interval_steps_count,
                            'avg_kl_div': self.interval_metrics_accum['kl_div'] / self.interval_steps_count,
                            'avg_d_total': self.interval_metrics_accum['d_total'] / self.interval_steps_count,
                            'val_metric': self.last_val_metrics.get(self.args.val_primary_metric),
                            'current_lambda_kl_val': self.lambda_kl
                        }
                        if self.am_main_process:
                            self.logger.info(f"GStep {self.global_step}: Lambda_KL Q-Ctrl block. Current lambda_kl: {self.lambda_kl:.4e}")
                            log_metrics_str = {k: (f'{v:.4f}' if isinstance(v, float) else str(v)) for k,v in current_interval_metrics.items()}
                            self.logger.info(f"  Interval Metrics for Q-State: {log_metrics_str}")
                            self.logger.info(f"  Previous Lambda_KL Q-State: {self.lambda_kl_q_controller.prev_lambda_kl_state}")
                            self.logger.info(f"  Previous Lambda_KL Action: {self.lambda_kl_q_controller.prev_lambda_kl_action}")

                        q_state_lambda_kl = self.lambda_kl_q_controller.get_lambda_kl_state(current_interval_metrics)
                        if self.am_main_process and q_state_lambda_kl is not None:
                            self.logger.info(f"  New Lambda_KL Q-State: {q_state_lambda_kl}")
                            q_vals_state = self.lambda_kl_q_controller.q_table.get(q_state_lambda_kl, {}).get('lambda_kl_scale')
                            self.logger.info(f"  Q-values for this state (LKL_scale): {q_vals_state}")

                        if self.lambda_kl_q_controller.prev_lambda_kl_state is not None and \
                           self.lambda_kl_q_controller.prev_lambda_kl_action is not None and \
                           q_state_lambda_kl is not None and self.prev_interval_metrics_for_lambda_kl_reward is not None:
                            reward_for_lambda_kl = self.lambda_kl_q_controller.compute_lambda_kl_reward(
                                current_interval_metrics, self.prev_interval_metrics_for_lambda_kl_reward
                            )
                            if self.am_main_process: self.logger.info(f"  Lambda_KL Q-Ctrl reward computed: {reward_for_lambda_kl:.3f}")
                            self.lambda_kl_q_controller.update_q_values(
                                self.lambda_kl_q_controller.prev_lambda_kl_state,
                                self.lambda_kl_q_controller.prev_lambda_kl_action,
                                reward_for_lambda_kl, q_state_lambda_kl, mode='lambda_kl'
                            )
                        elif q_state_lambda_kl is not None and hasattr(self.lambda_kl_q_controller, 'set_initial_lambda_kl_metrics'):
                            self.lambda_kl_q_controller.set_initial_lambda_kl_metrics(current_interval_metrics)
                        
                        if q_state_lambda_kl is not None:
                            lambda_kl_action_dict = self.lambda_kl_q_controller.choose_action(q_state_lambda_kl, mode='lambda_kl')
                            chosen_scale = lambda_kl_action_dict.get('lambda_kl_scale', 1.0)
                            if self.am_main_process:
                                self.logger.info(f"  Lambda_KL Q-Ctrl CHOSE scale: {chosen_scale:.2f} (Epsilon: {self.lambda_kl_q_controller.epsilon:.3f})")
                            self.prev_interval_metrics_for_lambda_kl_reward = current_interval_metrics.copy()
                            new_lambda_kl_val = self.lambda_kl * chosen_scale
                            self.lambda_kl = float(np.clip(new_lambda_kl_val, self.min_lambda_kl_q_control, self.max_lambda_kl_q_control))
                            if self.am_main_process:
                                self.logger.info(f"GStep {self.global_step}: Lambda_KL Q-Ctrl updated trainer's self.lambda_kl to {self.lambda_kl:.4e} (using scale: {chosen_scale:.2f})")
                            self.lambda_kl_q_controller.prev_lambda_kl_state = q_state_lambda_kl
                            self.lambda_kl_q_controller.prev_lambda_kl_action = lambda_kl_action_dict
                        
                        # Update lambda_kl in optimizer Q-controllers
                        for q_ctrl_opt in [opt_gen_q_controller, opt_disc_q_controller, self.lambda_kl_q_controller]:
                            if q_ctrl_opt and hasattr(q_ctrl_opt, 'set_current_lambda_kl'):
                                q_ctrl_opt.set_current_lambda_kl(self.lambda_kl)
                        
                        self.interval_metrics_accum = defaultdict(float)
                        self.interval_steps_count = 0

                    # --- Logging ---
                    if self.global_step > 0 and self.global_step % self.args.log_interval == 0 and \
                       log_interval_items_processed > 0 and self.am_main_process:
                        log_metrics = {f"train/{k.replace('_agg','')}": v / log_interval_items_processed
                                       for k, v in log_interval_accum_losses.items()}
                        lr_g = self.optimizer_enc_gen.param_groups[0]['lr']
                        lr_d = self.optimizer_disc.param_groups[0]['lr']
                        log_metrics.update({
                            "train/lr_gen": lr_g, "train/lr_disc": lr_d,
                            "epoch_frac": epoch + ((batch_idx + 1) / (num_batches_epoch or 1)),
                            "global_step": self.global_step, "train/lambda_kl_eff": self.lambda_kl
                        })
                        if opt_gen_q_controller and hasattr(opt_gen_q_controller, 'get_info'):
                            log_metrics.update({f"q_ctrl_gen/{k.replace('_','')}": v for k,v in opt_gen_q_controller.get_info().items()})
                        if opt_disc_q_controller and hasattr(opt_disc_q_controller, 'get_info'):
                            log_metrics.update({f"q_ctrl_disc/{k.replace('_','')}": v for k,v in opt_disc_q_controller.get_info().items()})
                        if self.lambda_kl_q_controller and hasattr(self.lambda_kl_q_controller, 'get_info'):
                             log_metrics.update({f"q_ctrl_lkl/{k.replace('_','')}":v for k,v in self.lambda_kl_q_controller.get_info().items()})

                        gt, dt = log_metrics.get('train/loss_g_total',-1.0), log_metrics.get('train/loss_d_total',-1.0)
                        gr, gk, ga = log_metrics.get('train/loss_recon',-1.0), log_metrics.get('train/loss_kl',-1.0), log_metrics.get('train/loss_g_adv',-1.0)
                        dr, df = log_metrics.get('train/loss_d_real',-1.0), log_metrics.get('train/loss_d_fake',-1.0)
                        qeg_eps, qed_eps = log_metrics.get('q_ctrl_gen/epsilon',-1.0), log_metrics.get('q_ctrl_disc/epsilon',-1.0)
                        qelkl_eps = log_metrics.get('q_ctrl_lkl/epsilon', -1.0)
                        qag_act, qad_act = log_metrics.get('q_ctrl_gen/lastlrmomaction',{}), log_metrics.get('q_ctrl_disc/lastlrmomaction',{})
                        qal_act = log_metrics.get('q_ctrl_lkl/lastlambdaklaction',{})
                        qslg = qag_act.get('lr_scale',1.0) if isinstance(qag_act,dict) else 1.0
                        qsld = qad_act.get('lr_scale',1.0) if isinstance(qad_act,dict) else 1.0
                        qslkl = qal_act.get('lambda_kl_scale',1.0) if isinstance(qal_act,dict) else 1.0

                        log_str=(f"E{epoch+1} S{self.global_step} | G_tot:{gt:.3f}(RecDCT:{gr:.3f} KL:{gk:.3f} Adv:{ga:.3f}) | "
                                 f"D_tot:{dt:.3f}(R:{dr:.3f} F:{df:.3f}) | LR(G/D):{lr_g:.1e}/{lr_d:.1e} | "
                                 f"Q_Eps(LRM G/D):{qeg_eps:.2f}/{qed_eps:.2f} Q_Scl(LRM G/D):{qslg:.2f}/{qsld:.2f} | "
                                 f"Q_Eps(LKL):{qelkl_eps:.2f} Q_Scl(LKL):{qslkl:.2f} LKL_eff:{self.lambda_kl:.2e}")
                        prog_bar.set_postfix_str(f"G:{gt:.2f} D:{dt:.2f} RecDCT:{gr:.3f} LKL:{self.lambda_kl:.1e}",refresh=True)
                        self.logger.info(log_str)
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                            wandb.log(log_metrics, step=self.global_step)
                        log_interval_accum_losses = defaultdict(float)
                        log_interval_items_processed = 0

                    # --- Log Training Samples to WandB ---
                    if recon_mel_for_logging is not None: # This is already guarded by am_main_process and interval
                        self._log_samples_to_wandb("train_recon_mel", recon_mel_for_logging, self.args.num_val_samples_to_log)
                        self._log_samples_to_wandb("train_target_mel", batch_mel_segments, self.args.num_val_samples_to_log)

                    # --- Log Fixed Noise Samples to WandB ---
                    log_fixed_noise = (
                        self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and
                        self.global_step > 0 and self.global_step % self.args.wandb_log_fixed_noise_samples_interval == 0 and
                        self.fixed_noise_for_sampling is not None
                    )
                    if log_fixed_noise:
                        m_ref.eval() # Set to eval for sampling
                        with torch.no_grad():
                            generated_norm_dcts_fixed = m_ref.decode(self.fixed_noise_for_sampling) # type: ignore
                            unnorm_dcts_fixed = AudioSpecGenerator._unnormalize_dct(generated_norm_dcts_fixed, self.args)
                            
                            spec_dims_canon = (self.audio_config.get("num_time_frames_for_1s_segment", 86), self.args.n_mels)
                            canonical_bboxes_list = []
                            for _ in range(self.fixed_noise_for_sampling.shape[0]): # type: ignore
                                bboxes_one = golden_subdivide_rect_fixed_n(
                                    spec_dims_canon, self.gaad_config['num_regions'], self.device,
                                    self.fixed_noise_for_sampling.dtype, self.gaad_config['min_size_px'] # type: ignore
                                )
                                canonical_bboxes_list.append(bboxes_one)
                            canonical_bboxes_batch = torch.stack(canonical_bboxes_list)
                            target_mel_shape_fixed = (
                                self.fixed_noise_for_sampling.shape[0], 1, self.args.n_mels, # type: ignore
                                self.audio_config.get("num_time_frames_for_1s_segment", 86)
                            )
                            fixed_noise_mels_gen = d_ref._assemble_mel_from_dct_regions(
                                unnorm_dcts_fixed, canonical_bboxes_batch, target_mel_shape_fixed
                            )
                            self._log_samples_to_wandb("fixed_noise_generated_mel", fixed_noise_mels_gen,
                                                       self.fixed_noise_for_sampling.shape[0]) # type: ignore
                        m_ref.train() # Set back to train

                    # --- Save Intermediate Checkpoint ---
                    if self.args.save_interval > 0 and self.global_step > 0 and \
                       self.global_step % self.args.save_interval == 0 and self.am_main_process:
                        chkpt_metrics={'train_loss_g_total_macro':avg_g_total_macro,
                                       'train_loss_d_total_macro':avg_d_total_macro}
                        self._save_checkpoint(is_intermediate=True,metrics=chkpt_metrics)
            
            # --- End of Epoch ---
            if self.am_main_process:
                final_avg_g = log_interval_accum_losses['loss_g_total_agg'] / log_interval_items_processed \
                              if log_interval_items_processed > 0 else (log_metrics.get('train/loss_g_total', float('nan')) if 'log_metrics' in locals() else float('nan'))
                final_avg_d = log_interval_accum_losses['loss_d_total_agg'] / log_interval_items_processed \
                              if log_interval_items_processed > 0 else (log_metrics.get('train/loss_d_total', float('nan')) if 'log_metrics' in locals() else float('nan'))
                self.logger.info(f"Epoch {epoch+1} finished. Approx Avg Loss (last interval or batch): "
                                 f"G:{final_avg_g:.4f}, D:{final_avg_d:.4f}, LambdaKL_eff:{self.lambda_kl:.3e}")
                if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                    wandb.log({"epoch":epoch+1,
                               "epoch_avg_train_loss_g_approx":final_avg_g if np.isfinite(final_avg_g) else -1.0,
                               "epoch_avg_train_loss_d_approx":final_avg_d if np.isfinite(final_avg_d) else -1.0,
                               "epoch_lambda_kl": self.lambda_kl}, step=self.global_step)
            
            # --- Validation ---
            if self.val_loader and self.am_main_process:
                val_metrics = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if val_metrics:
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                        wandb.log({f"val/{k}":v for k,v in val_metrics.items()}, step=self.global_step)
                    
                    metric_to_check = self.args.val_primary_metric
                    default_val = -float('inf') if metric_to_check in ["avg_val_ssim_mel", "avg_val_psnr_mel"] else float('inf')
                    current_val_for_best = val_metrics.get(metric_to_check, default_val)
                    
                    is_better = (current_val_for_best > self.best_val_metric_val) \
                                if metric_to_check in ["avg_val_ssim_mel", "avg_val_psnr_mel"] \
                                else (current_val_for_best < self.best_val_metric_val)
                    
                    if is_better and np.isfinite(current_val_for_best):
                        self.logger.info(f"New best val metric ({metric_to_check}): {current_val_for_best:.4f} "
                                         f"(prev: {self.best_val_metric_val:.4f}). Save best.")
                        self.best_val_metric_val = current_val_for_best
                        self._save_checkpoint(is_best=True, metrics=val_metrics)
            
            # --- Save Epoch End Checkpoint (always, if main process) ---
            if self.am_main_process:
                epoch_end_metrics = self.last_val_metrics.copy() if self.last_val_metrics else {}
                epoch_end_metrics["epoch_end_train_loss_g_avg_approx"] = final_avg_g if 'final_avg_g' in locals() and np.isfinite(final_avg_g) else -1.0
                epoch_end_metrics["epoch_end_train_loss_d_avg_approx"] = final_avg_d if 'final_avg_d' in locals() and np.isfinite(final_avg_d) else -1.0
                self._save_checkpoint(metrics=epoch_end_metrics)

    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Optional[Dict[str, float]]:
        if not self.val_loader or not self.am_main_process: return None
        m_ref = self.model.module if self.ddp_active else self.model
        d_ref_val = self.discriminator.module if self.ddp_active else self.discriminator
        m_ref.eval()

        total_recon_dct_mse_sum = 0.0; total_mel_mse_sum = 0.0; total_psnr_mel_sum = 0.0
        total_ssim_mel_sum = 0.0; total_lpips_mel_sum = 0.0; total_items_evaluated = 0
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0, device=self.device)).dtype
        prog_bar_disabled = not self.am_main_process or os.getenv('CI') == 'true'
        logged_samples_count_this_val = 0

        for batch_idx_val, batch_real_mel_segments in enumerate(
            tqdm(self.val_loader, desc="Validating", disable=prog_bar_disabled, dynamic_ncols=True)
        ):
            real_mel_segments = batch_real_mel_segments.to(self.device, dtype=dtype_m)
            B, _, H_mel, W_mel = real_mel_segments.shape

            # VAE Forward pass
            recon_norm_dcts, mu, logvar, gaad_bboxes_from_enc, target_norm_dcts_for_loss = m_ref(real_mel_segments)

            # 1. DCT Reconstruction Loss
            loss_recon_dct_batch = self._compute_recon_loss(recon_norm_dcts, target_norm_dcts_for_loss)
            if torch.isfinite(loss_recon_dct_batch):
                total_recon_dct_mse_sum += loss_recon_dct_batch.item() * B
            
            # 2. Mel Spectrogram Metrics (requires assembly)
            unnorm_recon_dcts = AudioSpecGenerator._unnormalize_dct(recon_norm_dcts, self.args)
            recon_mel_assembled = d_ref_val._assemble_mel_from_dct_regions(
                unnorm_recon_dcts, gaad_bboxes_from_enc, real_mel_segments.shape
            )

            if recon_mel_assembled.shape == real_mel_segments.shape:
                loss_mel_mse_batch = F.mse_loss(recon_mel_assembled, real_mel_segments, reduction='mean')
                if torch.isfinite(loss_mel_mse_batch):
                    total_mel_mse_sum += loss_mel_mse_batch.item() * B
                    mse_val = loss_mel_mse_batch.item()
                    psnr_mel_batch_avg = 10 * math.log10(1.0 / (mse_val + EPS)) if mse_val > EPS else 100.0
                    total_psnr_mel_sum += psnr_mel_batch_avg * B
                
                recon_mel_01 = (recon_mel_assembled.clamp(-1,1)+1)/2.0
                real_mel_01 = (real_mel_segments.clamp(-1,1)+1)/2.0

                if self.ssim_metric:
                    try:
                        ssim_val = self.ssim_metric(recon_mel_01, real_mel_01)
                        if torch.isfinite(ssim_val): total_ssim_mel_sum += ssim_val.item() * B
                    except Exception as e_ssim: self.logger.debug(f"SSIM (Mel) calculation failed: {e_ssim}")
                
                if self.lpips_loss_fn:
                    try:
                        recon_for_lpips = recon_mel_assembled.repeat(1,3,1,1) if recon_mel_assembled.shape[1]==1 else recon_mel_assembled
                        real_for_lpips = real_mel_segments.repeat(1,3,1,1) if real_mel_segments.shape[1]==1 else real_mel_segments
                        lpips_val = self.lpips_loss_fn(recon_for_lpips, real_for_lpips)
                        if torch.isfinite(lpips_val.sum()): total_lpips_mel_sum += lpips_val.sum().item()
                    except Exception as e_lpips: self.logger.debug(f"LPIPS (Mel) calculation failed: {e_lpips}")
            
            total_items_evaluated += B

            if logged_samples_count_this_val < num_val_samples_to_log and \
               self.args.wandb and WANDB_AVAILABLE and wandb.run:
                num_to_log_now = min(B, num_val_samples_to_log - logged_samples_count_this_val)
                if num_to_log_now > 0:
                    self._log_samples_to_wandb("val_recon_mel", recon_mel_assembled[:num_to_log_now], num_to_log_now)
                    self._log_samples_to_wandb("val_target_mel", real_mel_segments[:num_to_log_now], num_to_log_now)
                logged_samples_count_this_val += num_to_log_now
        
        m_ref.train()
        metrics = {
            "avg_val_recon_dct_mse": total_recon_dct_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_mel_mse": total_mel_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_psnr_mel": total_psnr_mel_sum / total_items_evaluated if total_items_evaluated > 0 else 0.0,
            "avg_val_ssim_mel": total_ssim_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.ssim_metric else 0.0,
            "avg_val_lpips_mel": total_lpips_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.lpips_loss_fn else float('inf')
        }
        self.last_val_metrics = metrics
        self.logger.info(f"Validation Metrics (Ep {self.current_epoch+1}, GStep {self.global_step}): " +
                         ", ".join([f"{k}:{v:.4f}" for k,v in metrics.items()]))
        return metrics

    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict[str, Any]]=None, is_best:bool=False):
        if not self.am_main_process: return
        m_s = self.model.module if self.ddp_active else self.model
        d_s = self.discriminator.module if self.ddp_active else self.discriminator
        data = {
            'global_step': self.global_step, 'epoch': self.current_epoch,
            'model_state_dict': m_s.state_dict(), 'discriminator_state_dict': d_s.state_dict(),
            'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict() if self.args.use_amp and self.device.type == 'cuda' else None,
            'scaler_disc_state_dict': self.scaler_disc.state_dict() if self.args.use_amp and self.device.type == 'cuda' else None,
            'args': vars(self.args), 'metrics': metrics if metrics else self.last_val_metrics,
            'audio_config': self.audio_config,
            'best_val_metric_val': self.best_val_metric_val, 'current_lambda_kl': self.lambda_kl
        }
        q_hist_names = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']
        q_lkl_hist_names = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']
        
        def get_q_state(q_obj):
            if not q_obj or not hasattr(q_obj, 'q_table'): return None
            state = {'q_table': q_obj.q_table, 'epsilon': q_obj.epsilon,
                     'prev_lr_mom_state': q_obj.prev_lr_mom_state,
                     'prev_lr_mom_action': q_obj.prev_lr_mom_action,
                     'prev_lambda_kl_state': q_obj.prev_lambda_kl_state,
                     'prev_lambda_kl_action': q_obj.prev_lambda_kl_action,
                     'reward_hist': list(q_obj.reward_hist),
                     'q_table_access_count': dict(q_obj.q_table_access_count),
                     'q_table_creation_time': q_obj.q_table_creation_time,
                     'q_table_last_access_time': q_obj.q_table_last_access_time}
            if hasattr(q_obj, 'loss_g_total_hist'): # LRM Q-Controller
                state['loss_histories'] = {hname: list(getattr(q_obj, f"loss_{hname}_hist")) for hname in q_hist_names}
            if hasattr(q_obj, 'interval_avg_recon_hist'): # LKL Q-Controller
                state['interval_histories'] = {hname: list(getattr(q_obj, f"interval_{hname}_hist")) for hname in q_lkl_hist_names}
            return state

        data['q_controller_enc_gen_state'] = get_q_state(getattr(self.optimizer_enc_gen, 'q_controller', None))
        data['q_controller_disc_state'] = get_q_state(getattr(self.optimizer_disc, 'q_controller', None))
        data['q_controller_lambda_kl_state'] = get_q_state(self.lambda_kl_q_controller)

        fprefix = "wubuspectrans_ckpt_v011"
        if is_best: fp_str = f"{fprefix}_best.pt"
        elif is_intermediate: fp_str = f"{fprefix}_step{self.global_step}.pt"
        else: fp_str = f"{fprefix}_ep{self.current_epoch+1}_step{self.global_step}.pt"
        fp = Path(self.args.checkpoint_dir) / fp_str
        try:
            torch.save(data, fp)
            self.logger.info(f"Checkpoint saved: {fp.name}")
        except Exception as e:
            self.logger.error(f"Save CKPT error {fp}: {e}", exc_info=True)

    def load_checkpoint(self, checkpoint_path_str:str) -> Tuple[int,int]:
        checkpoint_path = Path(checkpoint_path_str)
        if not checkpoint_path.exists():
            self.logger.warning(f"CKPT {checkpoint_path} not found. Starting from scratch.")
            return 0,0
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.logger.info(f"Loaded CKPT: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load CKPT {checkpoint_path}: {e}. Starting from scratch.")
            return 0,0

        m_load = self.model.module if self.ddp_active else self.model
        d_load = self.discriminator.module if self.ddp_active else self.discriminator
        try: m_load.load_state_dict(ckpt['model_state_dict'], strict=self.args.load_strict)
        except Exception as e: self.logger.error(f"Error loading model state_dict: {e}", exc_info=not self.args.load_strict)
        try: d_load.load_state_dict(ckpt['discriminator_state_dict'], strict=self.args.load_strict)
        except Exception as e: self.logger.error(f"Error loading discriminator state_dict: {e}", exc_info=not self.args.load_strict)

        if 'optimizer_enc_gen_state_dict' in ckpt and self.optimizer_enc_gen:
            try: self.optimizer_enc_gen.load_state_dict(ckpt['optimizer_enc_gen_state_dict'])
            except Exception as e: self.logger.warning(f"Could not load Optimizer Enc/Gen state: {e}")
        if 'optimizer_disc_state_dict' in ckpt and self.optimizer_disc:
            try: self.optimizer_disc.load_state_dict(ckpt['optimizer_disc_state_dict'])
            except Exception as e: self.logger.warning(f"Could not load Optimizer Disc state: {e}")

        if self.args.use_amp and self.device.type == 'cuda':
            if 'scaler_enc_gen_state_dict' in ckpt and self.scaler_enc_gen and ckpt['scaler_enc_gen_state_dict'] is not None:
                self.scaler_enc_gen.load_state_dict(ckpt['scaler_enc_gen_state_dict'])
            if 'scaler_disc_state_dict' in ckpt and self.scaler_disc and ckpt['scaler_disc_state_dict'] is not None:
                self.scaler_disc.load_state_dict(ckpt['scaler_disc_state_dict'])
        
        q_hist_names = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']
        q_lkl_hist_names = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']

        def _load_q_state(q_ctrl_obj, q_state_data):
            if not q_ctrl_obj or not q_state_data: return
            try:
                q_ctrl_obj.q_table = q_state_data.get('q_table', {})
                q_ctrl_obj.epsilon = q_state_data.get('epsilon', q_ctrl_obj.epsilon_start)
                q_ctrl_obj.prev_lr_mom_state = q_state_data.get('prev_lr_mom_state')
                q_ctrl_obj.prev_lr_mom_action = q_state_data.get('prev_lr_mom_action')
                q_ctrl_obj.prev_lambda_kl_state = q_state_data.get('prev_lambda_kl_state')
                q_ctrl_obj.prev_lambda_kl_action = q_state_data.get('prev_lambda_kl_action')
                q_ctrl_obj.reward_hist = deque(q_state_data.get('reward_hist',[]), maxlen=q_ctrl_obj.reward_hist.maxlen if q_ctrl_obj.reward_hist else 100)
                q_ctrl_obj.q_table_access_count = defaultdict(int, q_state_data.get('q_table_access_count', {}))
                q_ctrl_obj.q_table_creation_time = q_state_data.get('q_table_creation_time', {})
                q_ctrl_obj.q_table_last_access_time = q_state_data.get('q_table_last_access_time', {})

                if 'loss_histories' in q_state_data and hasattr(q_ctrl_obj, 'state_history_len'):
                    lh = q_state_data['loss_histories']
                    for hname in q_hist_names:
                        setattr(q_ctrl_obj, f"loss_{hname}_hist", deque(lh.get(hname, []), maxlen=q_ctrl_obj.state_history_len))
                if 'interval_histories' in q_state_data and hasattr(q_ctrl_obj, 'lambda_kl_state_history_len'):
                    ih = q_state_data['interval_histories']
                    for hname in q_lkl_hist_names:
                         setattr(q_ctrl_obj, f"interval_{hname}_hist", deque(ih.get(hname, []), maxlen=q_ctrl_obj.lambda_kl_state_history_len))
                self.logger.info(f"Q-Controller state for loaded from checkpoint.")
            except Exception as e_qc_load: self.logger.warning(f"Could not fully load Q-Controller state: {e_qc_load}", exc_info=True)

        _load_q_state(getattr(self.optimizer_enc_gen, 'q_controller', None), ckpt.get('q_controller_enc_gen_state'))
        _load_q_state(getattr(self.optimizer_disc, 'q_controller', None), ckpt.get('q_controller_disc_state'))
        _load_q_state(self.lambda_kl_q_controller, ckpt.get('q_controller_lambda_kl_state'))
        
        loaded_gs = ckpt.get('global_step', 0)
        loaded_ep = ckpt.get('epoch', 0)
        next_ep_start = loaded_ep + 1 if (loaded_gs > 0 and not self.args.load_strict and batch_idx == num_batches_epoch -1) else loaded_ep # If not strict and was mid-epoch, restart that epoch. If end of epoch, next.
        
        default_best_val = -float('inf') if self.args.val_primary_metric in ["avg_val_ssim_mel","avg_val_psnr_mel"] else float('inf')
        self.best_val_metric_val = ckpt.get('best_val_metric_val', default_best_val)
        
        loaded_lambda_kl_from_ckpt = float(ckpt.get('current_lambda_kl', self.args.lambda_kl))
        if self.args.load_checkpoint and 'current_lambda_kl' in ckpt:
            self.lambda_kl = loaded_lambda_kl_from_ckpt
            self.logger.info(f"Loaded self.lambda_kl from checkpoint: {self.lambda_kl:.4e}")
        else:
            self.lambda_kl = self.args.lambda_kl # Use arg if not in ckpt or not loading
            self.logger.info(f"Using initial self.lambda_kl from args: {self.lambda_kl:.4e}")

        for q_ctrl in [getattr(self.optimizer_enc_gen, 'q_controller', None),
                       getattr(self.optimizer_disc, 'q_controller', None),
                       self.lambda_kl_q_controller]:
            if q_ctrl and hasattr(q_ctrl, 'set_current_lambda_kl'):
                q_ctrl.set_current_lambda_kl(self.lambda_kl)

        self.logger.info(f"Resuming training. GlobalStep: {loaded_gs}, NextEpochStart: {next_ep_start}. "
                         f"BestVal({self.args.val_primary_metric}): {self.best_val_metric_val:.4f}. Effective LambdaKL: {self.lambda_kl:.4e}")
        return loaded_gs, next_ep_start

    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        m_ref = self.model.module if self.ddp_active else self.model
        d_ref_sample = self.discriminator.module if self.ddp_active else self.discriminator
        m_ref.eval()
        dev = self.device
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0, device=self.device)).dtype
        lat_dim = self.args.latent_dim

        if noise is None:
            z = torch.randn(num_samples, lat_dim, device=dev, dtype=dtype_m)
        else:
            z = noise.to(device=dev, dtype=dtype_m)
        
        if z.shape[0] != num_samples:
            num_samples = z.shape[0]
            self.logger.warning(f"Number of samples adjusted to noise shape: {num_samples}")

        self.logger.info(f"Sampling {num_samples} Mel spectrograms...")
        generated_norm_dcts = m_ref.decode(z)
        unnorm_dcts_for_assembly = AudioSpecGenerator._unnormalize_dct(generated_norm_dcts, self.args)
        
        spec_dims_canonical = (self.audio_config.get("num_time_frames_for_1s_segment", 86), self.args.n_mels)
        canonical_gaad_bboxes_list = []
        for _ in range(num_samples):
            bboxes_one_sample = golden_subdivide_rect_fixed_n(
                spec_dims_canonical, self.gaad_config['num_regions'], dev, dtype_m, self.gaad_config['min_size_px']
            )
            canonical_gaad_bboxes_list.append(bboxes_one_sample)
        canonical_gaad_bboxes_batch = torch.stack(canonical_gaad_bboxes_list)

        target_mel_shape_for_sample = (
            num_samples, 1, self.args.n_mels, self.audio_config.get("num_time_frames_for_1s_segment", 86)
        )
        generated_mel_spectrograms = d_ref_sample._assemble_mel_from_dct_regions(
            unnorm_dcts_for_assembly, canonical_gaad_bboxes_batch, target_mel_shape_for_sample
        )
        self.logger.info("Sampling finished. Returning Mel spectrograms.")
        return generated_mel_spectrograms

# =====================================================================
# Arg Parsing and Main Execution Logic
# =====================================================================
def seed_worker_init_fn(worker_id, base_seed, rank, world_size):
     worker_seed = base_seed + worker_id + rank * world_size
     random.seed(worker_seed); np.random.seed(worker_seed); torch.manual_seed(worker_seed)

def seed_everything(seed:int,rank:int=0,world_size:int=1):
    actual_seed = seed + rank; random.seed(actual_seed); np.random.seed(actual_seed); torch.manual_seed(actual_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(actual_seed)

def _configure_wubu_stack(args: argparse.Namespace, prefix: str) -> Dict:
    config = DEFAULT_CONFIG_WUBU.copy()
    num_levels_val = getattr(args, f"{prefix}_num_levels", 0)
    config["num_levels"] = num_levels_val
    if num_levels_val == 0:
        for key in ["hyperbolic_dims", "initial_curvatures", "initial_scales",
                    "initial_spread_values", "boundary_points_per_level",
                    "transform_types", "transform_hidden_dims"]:
            config[key] = []
        # tangent_input_combination_dims needs a default if layers are created
        config["tangent_input_combination_dims"] = [DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0]]
        return config

    config["hyperbolic_dims"] = getattr(args, f"{prefix}_hyperbolic_dims", DEFAULT_CONFIG_WUBU["hyperbolic_dims"])
    config["initial_curvatures"] = getattr(args, f"{prefix}_initial_curvatures", DEFAULT_CONFIG_WUBU["initial_curvatures"])
    config["use_rotation_in_transform"] = getattr(args, f"{prefix}_use_rotation", DEFAULT_CONFIG_WUBU["use_rotation_in_transform"])
    config["phi_influence_curvature"] = getattr(args, f"{prefix}_phi_influence_curvature", DEFAULT_CONFIG_WUBU["phi_influence_curvature"])
    config["phi_influence_rotation_init"] = getattr(args, f"{prefix}_phi_influence_rotation_init", DEFAULT_CONFIG_WUBU["phi_influence_rotation_init"])
    config["dropout"] = args.wubu_dropout

    def _ensure_list_len(cfg_dict, key, target_len, default_fill_list_from_defaults):
        current_val = cfg_dict.get(key, [])
        is_list_orig = isinstance(current_val, list)
        current_list_val = current_val if is_list_orig else [current_val]
        
        base_default = default_fill_list_from_defaults[0] if default_fill_list_from_defaults else \
                       (1.0 if "scales" in key or "curvatures" in key else \
                       (0.1 if "spread" in key else ("linear" if "types" in key else 32)))
        fill_val = current_list_val[-1] if current_list_val else base_default
        
        if len(current_list_val) < target_len:
            cfg_dict[key] = (current_list_val + [fill_val]*(target_len-len(current_list_val)))[:target_len]
        elif len(current_list_val) > target_len:
            cfg_dict[key] = current_list_val[:target_len]
        
        # If original was not list, but target_len is 1, ensure it's not a list of one item
        if not is_list_orig and target_len == 1 and isinstance(cfg_dict[key], list):
            cfg_dict[key] = cfg_dict[key][0]

    for key_chk, default_key_in_wubu_defaults in [
        ("hyperbolic_dims", "hyperbolic_dims"), ("initial_curvatures", "initial_curvatures"),
        ("initial_scales", "initial_scales"), ("initial_spread_values", "initial_spread_values"),
        ("boundary_points_per_level", "boundary_points_per_level")]:
        _ensure_list_len(config, key_chk, num_levels_val, DEFAULT_CONFIG_WUBU[default_key_in_wubu_defaults])

    # Explicitly set boundary_points_per_level to 0 for all levels if not specified for audio context
    if "boundary_points_per_level" in config and num_levels_val > 0:
        if not hasattr(args, f"{prefix}_boundary_points_per_level"): # If user did not provide this arg
            config["boundary_points_per_level"] = [0] * num_levels_val
            
    if not isinstance(config.get("tangent_input_combination_dims"), list):
        config["tangent_input_combination_dims"] = [config.get("tangent_input_combination_dims", DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0])]

    num_transitions = max(0, num_levels_val-1)
    if num_transitions > 0:
        _ensure_list_len(config,"transform_types",num_transitions,DEFAULT_CONFIG_WUBU["transform_types"])
        _ensure_list_len(config,"transform_hidden_dims",num_transitions,DEFAULT_CONFIG_WUBU["transform_hidden_dims"])
    else:
        config["transform_types"]=[]
        config["transform_hidden_dims"]=[]
    return config


def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBuSpecTrans_v0.1.1: 1-Second Audio Segment VAE-GAN")
    # --- Data, DDP, General ---
    parser.add_argument('--audio_dir_path', type=str, default="demo_audio_data_dir", help="Path to directory containing audio files or a single audio file.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DDP local rank.")
    parser.add_argument('--epochs', type=int, default=150, help="Total training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per GPU.")
    parser.add_argument('--seed',type=int, default=42, help="Random seed.")
    parser.add_argument('--num_workers',type=int, default=2, help="DataLoader workers.")
    parser.add_argument('--checkpoint_dir',type=str, default='wubuspectrans_checkpoints_v011', help="Directory for checkpoints.")
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to checkpoint to load.")
    parser.add_argument('--load_strict', action='store_true', help="Use strict=True when loading state_dict.")
    parser.add_argument('--wandb',action='store_true', help="Enable WandB logging.")
    parser.add_argument('--wandb_project',type=str,default='WuBuSpecTransV011', help="WandB project name.")
    parser.add_argument('--wandb_run_name',type=str,default=None, help="WandB run name (auto-generated if None).")
    parser.add_argument('--log_interval',type=int, default=50, help="Log interval (in global steps).")
    parser.add_argument('--save_interval',type=int, default=1000, help="Checkpoint save interval (in global steps).")
    
    # --- Audio Processing ---
    parser.add_argument('--sample_rate', type=int, default=22050, help="Target sample rate for audio.")
    parser.add_argument('--n_fft', type=int, default=1024, help="FFT window size.")
    parser.add_argument('--hop_length', type=int, default=256, help="Hop length for STFT.")
    parser.add_argument('--n_mels', type=int, default=128, help="Number of Mel bands.")
    parser.add_argument('--fmin', type=float, default=0.0, help="Minimum frequency for Mel bands.")
    parser.add_argument('--fmax', type=float, default=None, help="Maximum frequency for Mel bands (None for sr/2).")
    parser.add_argument('--segment_duration_sec', type=float, default=1.0, help="Duration of audio segments in seconds.")
    parser.add_argument('--segment_overlap_sec', type=float, default=0.0, help="Overlap between audio segments in seconds.")
    parser.add_argument('--db_norm_min', type=float, default=-80.0, help="Min dB for Mel spectrogram normalization.")
    parser.add_argument('--db_norm_max', type=float, default=0.0, help="Max dB for Mel spectrogram normalization.")
    parser.add_argument('--preload_audio_dataset_to_ram', action='store_true', help="Preload entire audio dataset (as Mel spectrograms) into RAM.")

    # --- GAAD (for Spectrograms) ---
    parser.add_argument('--gaad_num_regions', type=int, default=10, help="Number of regions for GAAD on Mel spectrograms.")
    parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="GAAD type for spectrograms.")
    parser.add_argument('--gaad_min_size_px', type=int, default=4, help="Min region size (in Mel bins/time frames) for GAAD.")

    # --- DCT Processing ---
    parser.add_argument('--region_proc_size_t', type=int, default=16, help="Time dimension of processed GAAD region (for DCT block).")
    parser.add_argument('--region_proc_size_f', type=int, default=16, help="Frequency dimension of processed GAAD region (for DCT block).")
    parser.add_argument('--dct_norm_type', type=str, default="global_scale", choices=["none", "global_scale", "tanh"], help="Normalization type for DCT coefficients.")
    parser.add_argument('--dct_norm_global_scale', type=float, default=100.0, help="Global scaling factor if dct_norm_type is global_scale.")
    parser.add_argument('--dct_norm_tanh_scale', type=float, default=50.0, help="Scaling factor before tanh if dct_norm_type is tanh.")
    
    # --- Encoder Architecture (AudioSpecEncoder) ---
    parser.add_argument('--latent_dim', type=int, default=256, help="VAE latent space dimensionality.")
    parser.add_argument('--encoder_initial_tangent_dim', type=int, default=128, help="Input tangent dim to WuBu-S in Encoder.")

    # --- Discriminator Architecture (AudioSpecDiscriminator) ---
    parser.add_argument('--disc_input_type', type=str, default="mel", choices=["mel", "dct"], help="Input type for Discriminator.")
    parser.add_argument('--disc_apply_spectral_norm', action='store_true', help="Apply spectral normalization to D's conv/linear layers.")
    parser.add_argument('--disc_base_disc_channels', type=int, default=64, help="Base channels for D's CNN (if Mel input).")
    parser.add_argument('--disc_max_disc_channels', type=int, default=512, help="Max channels for D's CNN (if Mel input).")

    # --- WuBu Stacks (S_enc, G_gen, D_disc) ---
    parser.add_argument('--wubu_dropout', type=float, default=0.1, help="Dropout for WuBu layers.")
    # WuBu-S (Encoder)
    parser.add_argument('--wubu_s_num_levels', type=int, default=2); parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32]); parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8]); parser.add_argument('--wubu_s_use_rotation', action='store_true'); parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_s_output_dim_encoder', type=int, default=32, help="Output dim of WuBu-S in Encoder.");
    # WuBu-G (Generator)
    parser.add_argument('--wubu_g_num_levels', type=int, default=2); parser.add_argument('--wubu_g_hyperbolic_dims', nargs='+', type=int, default=[64,128]); parser.add_argument('--wubu_g_initial_curvatures', nargs='+', type=float, default=[0.8,1.0]); parser.add_argument('--wubu_g_use_rotation', action='store_true'); parser.add_argument('--wubu_g_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_g_phi_influence_rotation_init', action='store_true')
    # WuBu-D (Discriminator, if DCT input)
    parser.add_argument('--wubu_d_num_levels', type=int, default=1); parser.add_argument('--wubu_d_hyperbolic_dims', nargs='+', type=int, default=[64]); parser.add_argument('--wubu_d_initial_curvatures', nargs='+', type=float, default=[0.7]); parser.add_argument('--wubu_d_use_rotation', action='store_true'); parser.add_argument('--wubu_d_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_d_phi_influence_rotation_init', action='store_true')
    parser.add_argument('--wubu_d_output_dim', type=int, default=64, help="Output dim of WuBu-D (if disc_input_type='dct').");

    # --- Training ---
    parser.add_argument('--lambda_recon', type=float, default=10.0, help="Weight for VAE reconstruction loss (MSE on DCTs).")
    parser.add_argument('--lambda_kl', type=float, default=0.1, help="Weight for VAE KL divergence loss.")
    parser.add_argument('--lambda_gan', type=float, default=1.0, help="Weight for GAN adversarial loss (Generator part).")
    parser.add_argument('--learning_rate_gen',type=float,default=1e-4)
    parser.add_argument('--learning_rate_disc',type=float,default=1e-4)
    parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0)
    parser.add_argument('--global_max_grad_norm',type=float,default=5.0)
    parser.add_argument('--q_controller_enabled',action='store_true')
    parser.add_argument('--grad_accum_steps',type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--detect_anomaly',action='store_true')
    parser.add_argument('--lambda_kl_update_interval', type=int, default=200, help="Global steps between lambda_kl Q-controller updates.")
    parser.add_argument('--min_lambda_kl_q_control', type=float, default=1e-6, help="Min value for lambda_kl when Q-controlled.")
    parser.add_argument('--max_lambda_kl_q_control', type=float, default=0.5, help="Max value for lambda_kl when Q-controlled.")
    
    # --- Validation & Sampling ---
    parser.add_argument('--wandb_log_train_recon_interval', type=int, default=0, help="Log train recon Mel to WandB every N global steps (0=disable).")
    parser.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=0, help="Log fixed noise generated Mel to WandB every N global steps (0=disable).")
    parser.add_argument('--use_lpips_for_mel_verification', action='store_true', help="Use LPIPS for Mel spectrogram quality during validation.")
    parser.add_argument('--validation_audio_dir_path', type=str, default=None, help="Path to separate validation audio files. If None, uses split from train dir.")
    parser.add_argument('--validation_split_fraction', type=float, default=0.1, help="Fraction of main dataset to use for validation if validation_audio_dir_path is not set.")
    parser.add_argument('--val_primary_metric', type=str, default="avg_val_recon_dct_mse",
                        choices=["avg_val_recon_dct_mse", "avg_val_mel_mse", "avg_val_psnr_mel", "avg_val_ssim_mel", "avg_val_lpips_mel"],
                        help="Primary metric for choosing best checkpoint during validation.")
    parser.add_argument('--num_val_samples_to_log', type=int, default=2, help="Number of validation samples (real & recon Mel) to log to WandB.")
    parser.add_argument('--demo_num_samples', type=int, default=4, help="Number of Mel spectrograms to generate as demo images at end of training.")

    parsed_args = parser.parse_args()

    if not TORCH_DCT_AVAILABLE:
        parser.error("torch-dct library is required but not found. Please install it: pip install torch-dct")

    def validate_wubu_config_for_argparse(args_obj, prefix_str, parser_ref):
        num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0)
        if num_levels > 0:
            for suffix, attr_name in [("hyperbolic_dims", f"{prefix_str}_hyperbolic_dims"),
                                      ("initial_curvatures", f"{prefix_str}_initial_curvatures")]:
                val_list = getattr(args_obj, attr_name)
                is_list_original = isinstance(val_list, list)
                val_list = val_list if is_list_original else [val_list] # Ensure it's a list for processing

                if len(val_list) != num_levels:
                    if len(val_list) == 1 and num_levels > 1: # Single value provided for multiple levels
                        setattr(args_obj, attr_name, [val_list[0]] * num_levels)
                    elif not val_list: # Empty list provided
                        if suffix == "initial_curvatures":
                            setattr(args_obj, attr_name, [1.0] * num_levels) # Default curvature
                        elif suffix == "hyperbolic_dims":
                            # Default dim: try to divide latent_dim, or use a fallback
                            default_dim_val = getattr(args_obj, 'latent_dim', 32*num_levels) // num_levels if num_levels > 0 else 32
                            setattr(args_obj, attr_name, [max(1,default_dim_val)] * num_levels)
                        else: # Other lists, if empty, might cause issues later if not handled by _configure_wubu_stack
                             pass # Let _configure_wubu_stack handle it
                    else: # Mismatch and not a single value or empty
                        parser_ref.error(f"{prefix_str}: Length mismatch {attr_name} ({len(val_list)}) vs num_levels ({num_levels})")
    
    validate_wubu_config_for_argparse(parsed_args, "wubu_s", parser)
    validate_wubu_config_for_argparse(parsed_args, "wubu_g", parser)
    if parsed_args.disc_input_type == "dct":
        validate_wubu_config_for_argparse(parsed_args, "wubu_d", parser)

    # --- Set derived output dimensions based on WuBu stack configurations ---
    # WuBu-S (Encoder)
    if parsed_args.wubu_s_num_levels > 0 and parsed_args.wubu_s_hyperbolic_dims:
        # wubu_s_output_dim_encoder is an explicit arg, but check consistency or if it should be derived
        # For now, we assume wubu_s_output_dim_encoder is the target from the arg,
        # and the WuBuModel's output_tangent_projection handles mapping to it.
        # The FullyHyperbolicWuBuNestingModel's output_tangent_dim is set to this arg value.
        pass # Already handled by FullyHyperbolicWuBuNestingModel init
    else: # If WuBu-S is effectively disabled (0 levels)
        # The output of the WuBu stack (if it were active) would be passed to fc_mu, fc_logvar.
        # If it's disabled, the `aggregated_features` in AudioSpecEncoder will be the output
        # of `dct_coeff_embed` aggregated. This path needs checking.
        # Current: `wubu_s_encoder` output is fed to fc_mu/fc_logvar. If num_levels=0,
        # `FullyHyperbolicWuBuNestingModel` outputs from `input_tangent_projection` if input_tangent_dim == output_tangent_dim,
        # or from `output_tangent_projection` if they differ.
        # So, `audio_config['wubu_s_output_dim_encoder']` should be used for fc_mu/fc_logvar,
        # and if wubu_s_num_levels=0, the WuBuModel will essentially pass through (potentially with a Linear layer).
        # This seems acceptable as `_configure_wubu_stack` sets num_levels=0 if args make it so.
        # If `wubu_s_num_levels` is set to 0 in args, _configure_wubu_stack ensures this.
        # The `AudioSpecEncoder` initializes `wubu_s_encoder` with `output_tangent_dim=audio_config['wubu_s_output_dim_encoder']`.
        # This `audio_config['wubu_s_output_dim_encoder']` comes from `args.wubu_s_output_dim_encoder`.
        if parsed_args.wubu_s_num_levels == 0 and parsed_args.wubu_s_output_dim_encoder != parsed_args.encoder_initial_tangent_dim:
            # If WuBu-S is disabled, its effective "output" before fc_mu/logvar would be the aggregated embedded DCTs.
            # So wubu_s_output_dim_encoder should match encoder_initial_tangent_dim for direct passthrough sense.
            # However, the current design uses args.wubu_s_output_dim_encoder for the fc_mu/logvar input.
            # The WuBuModel with 0 levels will project from encoder_initial_tangent_dim to args.wubu_s_output_dim_encoder if they differ.
            # This is fine.
            pass


    # WuBu-G (Generator) output is fixed to num_dct_coeffs_flat.
    # Its internal structure is configured by wubu_g_* args.

    # WuBu-D (Discriminator, if DCT input)
    if parsed_args.disc_input_type == "dct":
        if parsed_args.wubu_d_num_levels > 0 and parsed_args.wubu_d_hyperbolic_dims:
            # `audio_config['wubu_d_output_dim']` (from `args.wubu_d_output_dim`) is used
            # as the output_tangent_dim for the WuBuModel in Discriminator. This is correct.
            pass
        else: # If WuBu-D is disabled for DCT input
            # The `feature_extractor` becomes an MLP. Its input is `wubu_d_input_dim` (which is `args.encoder_initial_tangent_dim`).
            # Its output is `audio_config['wubu_d_output_dim']` (from `args.wubu_d_output_dim`).
            # This is also fine.
            pass
    else: # For Mel input D, WuBu-D is not used (num_levels should be 0 from _configure_wubu_stack if prefix args not found)
        if parsed_args.wubu_d_num_levels != 0: # Should have been set to 0 if Mel input D
             # This case should be handled if _configure_wubu_stack doesn't find wubu_d_ args.
             # Or we can force it here. For now, assume _configure_wubu_stack handles it.
             pass

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
        rank=0; local_rank=0; world_size=1
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda' and torch.cuda.is_available(): # Check cuda availability again for safety
        torch.cuda.set_device(device) # Redundant if DDP already set, but safe.
    
    am_main_process = (rank == 0)
    
    base_logger_name = "WuBuSpecTransV01"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    specific_logger = logging.getLogger(base_logger_name)
    for handler in specific_logger.handlers[:]: specific_logger.removeHandler(handler)
    
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.basicConfig(level=log_level,
                        format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s',
                        force=True) # force=True to override any existing config
    
    current_logger_main = logging.getLogger(f"{base_logger_name}.Main")
    current_logger_main.info(f"--- {base_logger_name} (R{rank}/{world_size}, Dev {device}, DDP:{ddp_active}, AMP:{args.use_amp}) ---")
    seed_everything(args.seed, rank, world_size)
    
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        current_logger_main.warning("Autograd anomaly detection ENABLED.")
    
    if am_main_process:
        current_logger_main.info(f"Effective Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubuspectrans_v011_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        try:
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
                       resume="allow", id=wandb.util.generate_id() if wandb.run is None else wandb.run.id) # type: ignore
            current_logger_main.info(f"WandB initialized for run: {run_name}, Project: {args.wandb_project}")
        except Exception as e_wandb:
            current_logger_main.error(f"WandB initialization failed: {e_wandb}", exc_info=True)
            args.wandb = False # Disable WandB if init fails

    segment_samples = int(args.segment_duration_sec * args.sample_rate)
    num_time_frames_for_1s_segment = math.ceil(segment_samples / args.hop_length)

    audio_config = {
        "sample_rate": args.sample_rate, "n_fft": args.n_fft, "hop_length": args.hop_length,
        "n_mels": args.n_mels, "fmin": args.fmin, "fmax": args.fmax,
        "segment_duration_sec": args.segment_duration_sec,
        "region_proc_size_t": args.region_proc_size_t, "region_proc_size_f": args.region_proc_size_f,
        "wubu_s_output_dim_encoder": args.wubu_s_output_dim_encoder,
        "wubu_d_output_dim": args.wubu_d_output_dim,
        "num_time_frames_for_1s_segment": num_time_frames_for_1s_segment,
    }
    gaad_config = {
        "num_regions": args.gaad_num_regions, "decomposition_type": args.gaad_decomposition_type,
        "min_size_px": args.gaad_min_size_px
    }
    wubu_s_config_enc = _configure_wubu_stack(args, "wubu_s")
    wubu_g_config_gen = _configure_wubu_stack(args, "wubu_g")
    wubu_d_config_disc = _configure_wubu_stack(args, "wubu_d") # This will return a 0-level config if wubu_d args not used

    discriminator_config = {
        "input_type": args.disc_input_type, "apply_spectral_norm": args.disc_apply_spectral_norm,
        "base_disc_channels": args.disc_base_disc_channels, "max_disc_channels": args.disc_max_disc_channels,
    }

    if am_main_process:
        current_logger_main.info(f"AudioCfg:{audio_config}\nGAADCfg:{gaad_config}\n"
                                 f"WuBuS_Enc:{wubu_s_config_enc}\nWuBuG_Gen:{wubu_g_config_gen}\n"
                                 f"WuBuD_Disc:{wubu_d_config_disc}\nDiscCfg:{discriminator_config}")

    model = WuBuSpecTransNet(args, audio_config, gaad_config, wubu_s_config_enc, wubu_g_config_gen).to(device)
    discriminator = AudioSpecDiscriminator(args, audio_config, gaad_config, discriminator_config).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False) # type: ignore
        wandb.watch(discriminator, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False) # type: ignore
    
    if ddp_active:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # True for dev
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    q_cfg_gen, q_cfg_disc = None, None
    if args.q_controller_enabled:
        q_cfg_gen = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
        q_cfg_disc = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
        current_logger_main.info(f"Q-Controller config for G: {q_cfg_gen}")
        current_logger_main.info(f"Q-Controller config for D: {q_cfg_disc}")
    
    optimizer_enc_gen = RiemannianEnhancedSGD(model.parameters(), lr=args.learning_rate_gen,
                                              q_learning_config=q_cfg_gen,
                                              max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type="generator")
    optimizer_disc = RiemannianEnhancedSGD(discriminator.parameters(), lr=args.learning_rate_disc,
                                           q_learning_config=q_cfg_disc,
                                           max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type="discriminator")

    audio_files_list = []
    audio_dir_path_obj = Path(args.audio_dir_path)
    if audio_dir_path_obj.is_dir():
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
            audio_files_list.extend([str(p) for p in audio_dir_path_obj.rglob(ext)])
    elif audio_dir_path_obj.is_file():
        audio_files_list.append(str(audio_dir_path_obj))

    if not audio_files_list and "demo_audio_data" in args.audio_dir_path:
        if am_main_process:
            demo_dir = Path(args.audio_dir_path)
            demo_dir.mkdir(parents=True, exist_ok=True)
            dummy_audio_path = demo_dir / "dummy_sine_wubuspectrans.wav"
            if not dummy_audio_path.exists():
                current_logger_main.info(f"Attempting to create dummy audio: {dummy_audio_path}...")
                try:
                    # Use librosa for reliable audio writing, or soundfile if preferred
                    import soundfile as sf # Add this import if not already present, or use librosa.output.write_wav
                    sr_dummy = args.sample_rate
                    duration_dummy = 5.0
                    t_dummy = np.linspace(0, duration_dummy, int(sr_dummy * duration_dummy), endpoint=False)
                    wav_dummy = 0.3 * np.sin(2 * np.pi * 220.0 * t_dummy) + \
                                0.2 * np.sin(2 * np.pi * 440.0 * t_dummy) + \
                                0.1 * np.sin(2 * np.pi * 880.0 * t_dummy)
                    wav_dummy_norm = (wav_dummy / (np.max(np.abs(wav_dummy)) + EPS) * 0.9).astype(np.float32)
                    
                    # Replace imageio.imwrite with soundfile.write or librosa.output.write_wav
                    sf.write(str(dummy_audio_path), wav_dummy_norm, sr_dummy)
                    # Alternatively, using librosa (might need to ensure librosa.output is available)
                    # librosa.output.write_wav(str(dummy_audio_path), wav_dummy_norm, sr_dummy)
                    
                    current_logger_main.info(f"Dummy audio created: {dummy_audio_path}")
                    audio_files_list.append(str(dummy_audio_path))
                except ImportError:
                    current_logger_main.error("soundfile library not found. Cannot create dummy audio. Please install it: pip install soundfile")
                except Exception as e_dummy_audio:
                    current_logger_main.error(f"Error creating dummy audio: {e_dummy_audio}", exc_info=True)
        if ddp_active: torch.distributed.barrier()

    if not audio_files_list:
        current_logger_main.error(f"No audio files found in '{args.audio_dir_path}'. Exiting."); sys.exit(1)
    current_logger_main.info(f"Found {len(audio_files_list)} audio files for main dataset pool.")

    try:
        full_dataset = AudioSegmentDataset(audio_file_paths=audio_files_list, args=args,
                                           segment_duration_sec=args.segment_duration_sec,
                                           segment_overlap_sec=args.segment_overlap_sec,
                                           preload_to_ram=args.preload_audio_dataset_to_ram)
    except Exception as e:
        current_logger_main.error(f"Failed to initialize main Dataset: {e}", exc_info=True); sys.exit(1)
    if not full_dataset or len(full_dataset) == 0:
        current_logger_main.error("Main dataset is empty. Exiting."); sys.exit(1)

    train_dataset: Union[AudioSegmentDataset, SubsetRandomSampler] = full_dataset
    val_dataset: Optional[Union[AudioSegmentDataset, SubsetRandomSampler]] = None
    num_total_samples = len(full_dataset)
    
    val_audio_files_list = []
    if args.validation_audio_dir_path:
        val_dir_path_obj = Path(args.validation_audio_dir_path)
        if val_dir_path_obj.is_dir():
            for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
                val_audio_files_list.extend([str(p) for p in val_dir_path_obj.rglob(ext)])
        elif val_dir_path_obj.is_file():
            val_audio_files_list.append(str(val_dir_path_obj))
    
    if val_audio_files_list:
        try:
            val_dataset_candidate = AudioSegmentDataset(audio_file_paths=val_audio_files_list, args=args,
                                                        segment_duration_sec=args.segment_duration_sec,
                                                        preload_to_ram=args.preload_audio_dataset_to_ram)
            if len(val_dataset_candidate) > 0:
                val_dataset = val_dataset_candidate
                current_logger_main.info(f"Using separate validation audio dir: {args.validation_audio_dir_path}, Segments: {len(val_dataset)}")
            else:
                current_logger_main.warning(f"Validation audio dir {args.validation_audio_dir_path} loaded but resulted in 0 segments.")
        except Exception as e:
            current_logger_main.warning(f"Could not load validation dataset from '{args.validation_audio_dir_path}': {e}.")
    
    if val_dataset is None and args.validation_split_fraction > 0.0 and num_total_samples > 10 : # Only split if enough samples
        num_val = int(num_total_samples * args.validation_split_fraction)
        num_train = num_total_samples - num_val
        if num_train > 0 and num_val > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val],
                                                                       generator=torch.Generator().manual_seed(args.seed + rank)) # Ensure consistent split across ranks for DDP
            current_logger_main.info(f"Split main dataset (random): Train={len(train_dataset)}, Val={len(val_dataset)}")
        else:
            current_logger_main.warning("Random split for validation resulted in 0 samples for train or val. No validation set used from split."); val_dataset = None
            train_dataset = full_dataset # Use full dataset for training
    
    if am_main_process:
        current_logger_main.info(f"Final dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}")

    worker_init_fn_seeded = functools.partial(seed_worker_init_fn, base_seed=args.seed, rank=rank, world_size=world_size) if args.num_workers > 0 else None
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_active else None # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler,
                              pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded, drop_last=True)
    val_loader = None
    if val_dataset and len(val_dataset) > 0: # type: ignore
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None # type: ignore
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, # type: ignore
                                num_workers=args.num_workers, sampler=val_sampler,
                                pin_memory=(device.type == 'cuda'), drop_last=False, worker_init_fn=worker_init_fn_seeded)

    trainer = HybridTrainer(model, discriminator, optimizer_enc_gen, optimizer_disc, device,
                            train_loader, val_loader, args, rank, world_size, ddp_active)
    start_global_step, start_epoch = 0, 0
    if args.load_checkpoint:
        start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint)
    
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
            trainer._save_checkpoint(metrics=final_metrics_to_save)
            
            if args.epochs > 0 and hasattr(trainer, 'sample') and trainer.global_step > 0 and args.demo_num_samples > 0:
                current_logger_main.info("Generating final demo samples (Mel Spectrograms)...")
                try:
                    # Ensure model is on correct device for non-DDP sampling if DDP was active
                    sampling_model = model.module if ddp_active else model
                    d_ref_final_sample = discriminator.module if ddp_active else discriminator
                    
                    # Create a temporary trainer for sampling if needed, or use existing one
                    # If using existing trainer, ensure its model refs are correct for non-DDP context
                    trainer_for_sampling = HybridTrainer(
                        sampling_model, d_ref_final_sample, optimizer_enc_gen, optimizer_disc,
                        device, train_loader, val_loader, args, 0, 1, ddp_active=False # Simulate non-DDP for sampling
                    )
                    trainer_for_sampling.global_step = trainer.global_step
                    trainer_for_sampling.current_epoch = trainer.current_epoch
                    trainer_for_sampling.audio_config = trainer.audio_config
                    trainer_for_sampling.gaad_config = trainer.gaad_config
                    
                    generated_mels = trainer_for_sampling.sample(num_samples=args.demo_num_samples)
                    if generated_mels is not None and generated_mels.numel() > 0:
                        save_dir = Path(args.checkpoint_dir) / "demo_samples_mel_spectrograms_v011"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        for b_idx in range(min(args.demo_num_samples, generated_mels.shape[0])):
                            mel_to_save = (generated_mels[b_idx, 0].cpu().clamp(-1,1) + 1) / 2.0
                            save_image_path = save_dir / f"demo_mel_sample_{b_idx}_ep{trainer.current_epoch+1}.png"
                            save_image(mel_to_save, str(save_image_path))
                        current_logger_main.info(f"Saved demo Mel spectrogram images to {save_dir}")
                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            trainer_for_sampling._log_samples_to_wandb("final_demo_mel", generated_mels, args.demo_num_samples)
                except Exception as e_demo:
                    current_logger_main.error(f"Demo Mel sampling/saving error: {e_demo}", exc_info=True)
            
            if args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish() # type: ignore
        
        if ddp_active and is_initialized():
            destroy_process_group()
        current_logger_main.info(f"Rank {rank}: {base_logger_name} (v0.1.1) script finished.")

if __name__ == "__main__":
    main()