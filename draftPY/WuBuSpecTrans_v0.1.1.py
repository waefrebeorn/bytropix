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

import json
from torchvision.utils import save_image # Can save Mel spectrograms as images
MATPLOTLIB_AVAILABLE = True
if MATPLOTLIB_AVAILABLE:
    try:
        import matplotlib.pyplot as plt
        import librosa 
        import librosa.display
    except ImportError:
        plt = None # type: ignore
        librosa = None # type: ignore
        MATPLOTLIB_AVAILABLE = False
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
EPS = 1e-4
PHI = (1 + math.sqrt(5)) / 2
TAN_VEC_CLAMP_VAL = 1e3
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e7
MIN_WUBU_LEVEL_SCALE = 1e-3
MAX_WUBU_LEVEL_SCALE = 5.0


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
    MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False

    def __init__(self, q_learning_rate: float = 0.01, discount_factor: float = 0.90,
                 epsilon_start: float = 0.6, epsilon_min: float = 0.05, epsilon_decay: float = 0.9995,
                 lr_scale_options: Optional[List[float]] = None,
                 momentum_scale_options: Optional[List[float]] = None,
                 lambda_kl_scale_options: Optional[List[float]] = None,
                 max_q_table_size: int = 15000, 
                 state_history_len: int = 7,    
                 lambda_kl_state_history_len: int = 7, 
                 reward_clipping: Optional[Tuple[float, float]] = (-2.5, 2.5), 
                 q_value_clipping: Optional[Tuple[float, float]] = (-35.0, 35.0), 
                 num_probation_steps: Optional[int] = None,
                 lkl_num_probation_steps: Optional[int] = None):

        self.q_table: Dict[tuple, Dict[str, np.ndarray]] = {}
        self.alpha = q_learning_rate
        self.gamma = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_clipping = reward_clipping
        self.q_value_clipping = q_value_clipping
        self.current_lambda_kl: float = 0.0001 # Default; will be set by trainer

        _lr_options = lr_scale_options if lr_scale_options is not None else [0.7, 0.85, 1.0, 1.15, 1.3] 
        _mom_options = momentum_scale_options if momentum_scale_options is not None else [0.9, 0.95, 0.99, 1.0, 1.01] 
        _lkl_options = lambda_kl_scale_options if lambda_kl_scale_options is not None else [0.80, 0.90, 1.0, 1.10, 1.20] 

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

        self.lambda_kl_state_history_len = max(3, lambda_kl_state_history_len)
        self.interval_avg_recon_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_avg_kl_div_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_avg_d_total_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_val_metric_hist = deque(maxlen=self.lambda_kl_state_history_len)

        self.prev_lr_mom_state: Optional[tuple] = None
        self.prev_lr_mom_action: Optional[Dict[str, float]] = None
        self.prev_lambda_kl_state: Optional[tuple] = None
        self.prev_lambda_kl_action: Optional[Dict[str, float]] = None

        self.reward_hist = deque(maxlen=150) 
        self.max_q_table_size = max_q_table_size
        self.q_table_access_count: Dict[tuple, int] = defaultdict(int)
        self.q_table_creation_time: Dict[tuple, float] = {}
        self.q_table_last_access_time: Dict[tuple, float] = {}

        # Updated and New Reward Weights
        self.reward_weights = {
            "g_recon_improvement": 3.0, 
            "g_adv_improvement": 1.5,  
            "g_kl_control_penalty_ratio_trigger": 0.75, 
            "g_kl_control_penalty_abs_trigger_low": 30.0, 
            "g_kl_control_penalty_abs_trigger_high": 100.0,
            "g_kl_control_penalty": 0.4, 
            "g_loss_stability": 0.15,
            "g_easy_win_adv_thresh": 0.1, 
            "g_easy_win_recon_thresh": 0.15, 
            "g_easy_win_recon_bad_penalty": 0.75, 

            "d_balance_target": 1.8, 
            "d_real_low_bonus": 0.8,
            "d_fake_low_meaningful_bonus": 0.8,
            "d_misclassifies_fake_penalty": 1.2, 
            "d_loss_stability": 0.15,
            "d_very_weak_penalty": 1.0, 

            "gan_balance_g_bonus": 0.4,
            "gan_balance_d_penalty": 0.4,
            "extreme_gan_imbalance_penalty_g": 1.2,
            "g_stagnation_penalty_for_d": 0.3,
            
            "oscillation_penalty": 0.3, 
            "extreme_loss_penalty": 1.0, 
            "q_learner_stagnation_penalty_trigger": -0.3, 
            "q_learner_stagnation_penalty": 0.25, 

            "lambda_kl_recon_focus": 1.8,
            "lambda_kl_kl_target_range_low": 15.0, 
            "lambda_kl_kl_target_range_high": 80.0, 
            "lambda_kl_kl_target_range_bonus": 1.2,
            "lambda_kl_val_metric_improvement": 2.2,
            "lambda_kl_stability_penalty": 0.6,
            "lambda_kl_too_high_recon_bad_penalty": 0.7, 
        }

        self.num_probation_steps = num_probation_steps if num_probation_steps is not None else self.state_history_len + 3
        self.current_probation_step = 0
        self.on_probation = False

        self.lkl_num_probation_steps = lkl_num_probation_steps if lkl_num_probation_steps is not None else \
                                       max(3, self.lambda_kl_state_history_len + 2)
        self.lkl_current_probation_step = 0
        self.lkl_on_probation = False

        self.logger = logging.getLogger(f"WuBuSpecTransV01.QController.{id(self)}")
        if HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD:
             self.logger.warning(f"HAKMEMQController Inst ({self.logger.name}): Global FLUSH_Q_TABLES ON. Q-table will be cleared.")
        self.logger.info(f"HAKMEMQController ({self.logger.name}) initialized. Eps: {self.epsilon_start:.2f}->{self.epsilon_min:.2f}. "
                         f"LR/Mom Probation: {self.num_probation_steps} steps. LKL Probation: {self.lkl_num_probation_steps} steps.")
        self._internal_step_counter = 0
        self.original_epsilon_before_boost: Optional[float] = None
        self.epsilon_boost_active_steps: int = 0
        self.boosted_epsilon_value: Optional[float] = None

    def start_probation(self):
        if not self.on_probation:
            self.logger.info(f"Q-Ctrl ({self.logger.name}, LR/Mom) entering probation ({self.num_probation_steps} steps).")
            self.on_probation = True
            self.current_probation_step = 0
        if not self.lkl_on_probation:
            self.logger.info(f"Q-Ctrl ({self.logger.name}, LambdaKL) entering probation ({self.lkl_num_probation_steps} steps).")
            self.lkl_on_probation = True
            self.lkl_current_probation_step = 0

    def _tick_probation_lr_mom(self):
        if self.on_probation:
            self.current_probation_step += 1
            if self.current_probation_step >= self.num_probation_steps:
                self.logger.info(f"Q-Ctrl ({self.logger.name}, LR/Mom) probation ended after {self.current_probation_step} steps.")
                self.on_probation = False; self.current_probation_step = 0

    def _tick_probation_lkl(self):
        if self.lkl_on_probation:
            self.lkl_current_probation_step += 1
            if self.lkl_current_probation_step >= self.lkl_num_probation_steps:
                self.logger.info(f"Q-Ctrl ({self.logger.name}, LambdaKL) probation ended after {self.lkl_current_probation_step} steps.")
                self.lkl_on_probation = False; self.lkl_current_probation_step = 0
    
    def force_exploration_boost(self, duration_steps: int = 5, boost_epsilon_to: float = 0.6):
        if self.on_probation or self.lkl_on_probation :
            self.logger.debug(f"Q-Ctrl ({self.logger.name}): Exploration boost requested but controller is on probation. Ignoring.")
            return

        if self.epsilon_boost_active_steps > 0: 
             self.epsilon_boost_active_steps = max(self.epsilon_boost_active_steps, duration_steps) 
             self.logger.info(f"Q-Ctrl ({self.logger.name}): Exploration boost extended to {self.epsilon_boost_active_steps} total steps.")
        else: 
            self.original_epsilon_before_boost = self.epsilon
            self.boosted_epsilon_value = max(self.epsilon, boost_epsilon_to) 
            self.epsilon = self.boosted_epsilon_value
            self.epsilon_boost_active_steps = duration_steps
            self.logger.info(f"Q-Ctrl ({self.logger.name}): Exploration boost ACTIVATED. Epsilon: {self.epsilon:.3f} for {duration_steps} steps.")

    def _tick_exploration_boost(self):
        if hasattr(self, 'epsilon_boost_active_steps') and self.epsilon_boost_active_steps > 0:
            self.epsilon_boost_active_steps -= 1
            if self.epsilon_boost_active_steps == 0:
                if self.original_epsilon_before_boost is not None:
                    self.epsilon = self.original_epsilon_before_boost
                    self.logger.info(f"Q-Ctrl ({self.logger.name}): Exploration boost ENDED. Epsilon restored to {self.epsilon:.3f}.")
                else: 
                    self.logger.error(f"Q-Ctrl ({self.logger.name}): Exploration boost ended, but original_epsilon was None!")
                    self.epsilon = self.epsilon_start 
                self.original_epsilon_before_boost = None
                self.boosted_epsilon_value = None

    def reset_q_learning_state(self, reset_q_table: bool = True, reset_epsilon: bool = True,
                               context_msg: str = "Q-Ctrl Reset", start_probation: bool = False):
        self.logger.info(f"{context_msg} ({self.logger.name}): Resetting Q-Controller state. Reset Q-table: {reset_q_table}, Reset Epsilon: {reset_epsilon}, Start Probation: {start_probation}")
        history_deques = [
            self.loss_g_total_hist, self.loss_g_recon_hist, self.loss_g_kl_hist,
            self.loss_g_adv_hist, self.loss_d_total_hist, self.loss_d_real_hist,
            self.loss_d_fake_hist, self.interval_avg_recon_hist, self.interval_avg_kl_div_hist,
            self.interval_avg_d_total_hist, self.interval_val_metric_hist, self.reward_hist
        ]
        for deq in history_deques: deq.clear()
        self.prev_lr_mom_state = None; self.prev_lr_mom_action = None
        self.prev_lambda_kl_state = None; self.prev_lambda_kl_action = None
        if reset_epsilon: self.epsilon = self.epsilon_start; self.logger.info(f"{context_msg} ({self.logger.name}): Epsilon reset to {self.epsilon_start:.2f}")
        if reset_q_table:
            self.logger.info(f"{context_msg} ({self.logger.name}): Clearing Q-table and related stats.")
            self.q_table.clear(); self.q_table_access_count.clear()
            self.q_table_creation_time.clear(); self.q_table_last_access_time.clear()
        if start_probation: self.start_probation()
        else:
            self.on_probation = False; self.current_probation_step = 0
            self.lkl_on_probation = False; self.lkl_current_probation_step = 0
        self._internal_step_counter = 0
        self.epsilon_boost_active_steps = 0 

    def _get_trend_bin(self, history: deque, current_val: Optional[float], 
                       relative_to_median: bool = True, value_scale_for_diff:float = 1.0,
                       thresholds: List[float] = [-0.15, -0.02, 0.02, 0.15]) -> int:
        if current_val is None or not np.isfinite(current_val): return (len(thresholds) + 1) // 2
        valid_history = [h for h in history if h is not None and np.isfinite(h)] # Added h is not None
        if not valid_history: return (len(thresholds) + 1) // 2

        prev_ref = np.median(valid_history) if len(valid_history) > 1 else valid_history[0]
        
        diff = current_val - prev_ref
        if relative_to_median:
            denominator_val = abs(prev_ref) + (value_scale_for_diff * 0.001) 
            if abs(prev_ref) < denominator_val * 0.1 : 
                denominator_val = max(abs(current_val), denominator_val) + EPS
            relative_diff = diff / (denominator_val + EPS)
        else:
            relative_diff = diff / (value_scale_for_diff + EPS)

        for i, th in enumerate(thresholds):
            if relative_diff < th: return i
        return len(thresholds)

    def _update_loss_histories(self, current_losses: Dict[str, float]):
        loss_map = { 'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist }
        for name, deq in loss_map.items():
            loss_val = current_losses.get(name)
            if loss_val is not None and np.isfinite(loss_val): deq.append(loss_val)

    def get_lr_mom_state(self, current_losses: Dict[str, float], current_lr: float,
                         current_momentum: float, is_generator_q: bool) -> Optional[tuple]:
        self._internal_step_counter +=1
        self._update_loss_histories(current_losses)

        req_keys_g = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        req_keys_d = ['loss_d_total', 'loss_g_total', 'loss_d_real', 'loss_d_fake', 'loss_g_adv']
        required_keys = req_keys_g if is_generator_q else req_keys_d

        if not all(key in current_losses and np.isfinite(current_losses[key]) for key in required_keys):
            self.logger.debug(f"LR/Mom QState ({self.logger.name}): Insufficient/non-finite. Need: {required_keys}.")
            return None
        if not (np.isfinite(current_lr) and np.isfinite(current_momentum)):
            self.logger.debug(f"LR/Mom QState ({self.logger.name}): Non-finite LR/Mom.")
            return None

        if is_generator_q:
            s_g_total_trend = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total'])
            s_d_total_level_opp = np.digitize(current_losses['loss_d_total'], [0.15, 0.4, 0.7, 1.2]).item()
            s_g_recon_trend = self._get_trend_bin(self.loss_g_recon_hist, current_losses['loss_g_recon'])
            s_g_recon_level = np.digitize(current_losses['loss_g_recon'], [0.02, 0.08, 0.2, 0.5]).item()

            kl_val, recon_val = current_losses['loss_g_kl'], current_losses['loss_g_recon']
            s_kl_problem = 0 
            if (self.current_lambda_kl * kl_val > self.reward_weights.get("g_kl_control_penalty_ratio_trigger", 0.75) * recon_val and
                recon_val > 0.03 and self.current_lambda_kl > 1e-5):
                s_kl_problem = 1
            elif kl_val > self.reward_weights.get("g_kl_control_penalty_abs_trigger_high", 100.0):
                 s_kl_problem = 2
            
            s_g_adv_level = np.digitize(current_losses['loss_g_adv'], [0.05, 0.2, 0.6, 1.5]).item()
            s_lr_bin = np.digitize(current_lr, [5e-6, 2e-5, 1e-4, 5e-4]).item()
            s_mom_bin = np.digitize(current_momentum, [0.8, 0.9, 0.97]).item()
            eps_bin = np.digitize(self.epsilon, [self.epsilon_min * 1.2, self.epsilon_start * 0.3, self.epsilon_start * 0.7]).item()

            state_tuple = ("LRM_G", s_g_total_trend, s_d_total_level_opp, s_g_recon_trend, s_g_recon_level,
                           s_kl_problem, s_g_adv_level, s_lr_bin, s_mom_bin, eps_bin)
        else: 
            s_d_total_trend = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total'])
            s_g_adv_level_opp = np.digitize(current_losses.get('loss_g_adv', 0.7), [0.05, 0.2, 0.6, 1.5]).item()
            s_d_total_level = np.digitize(current_losses['loss_d_total'], [0.1, 0.3, 0.7, 1.2, 2.0]).item()
            
            d_fake_val = current_losses['loss_d_fake']
            d_real_val = current_losses['loss_d_real']
            s_d_fake_level = np.digitize(d_fake_val, [0.1, 0.5, 1.0, 2.0]).item()
            s_d_real_level = np.digitize(d_real_val, [0.05, 0.2, 0.5, 0.8]).item()
            
            s_lr_bin = np.digitize(current_lr, [5e-6, 2e-5, 1e-4, 5e-4]).item()
            s_mom_bin = np.digitize(current_momentum, [0.8, 0.9, 0.97]).item()
            eps_bin = np.digitize(self.epsilon, [self.epsilon_min * 1.2, self.epsilon_start * 0.3, self.epsilon_start * 0.7]).item()

            state_tuple = ("LRM_D", s_d_total_trend, s_g_adv_level_opp, s_d_total_level,
                           s_d_fake_level, s_d_real_level, s_lr_bin, s_mom_bin, eps_bin)

        self._ensure_q_state_exists(state_tuple)
        return state_tuple

    def get_lambda_kl_state(self, interval_metrics: Dict[str, Optional[float]]) -> Optional[tuple]:
        required_keys = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric', 'current_lambda_kl_val']
        valid_metrics = True; current_metrics_for_hist: Dict[str, float] = {}
        for key in required_keys:
            val = interval_metrics.get(key)
            if val is None or not np.isfinite(val): valid_metrics = False; break
            current_metrics_for_hist[key] = float(val)
        if not valid_metrics: return None

        self.interval_avg_recon_hist.append(current_metrics_for_hist['avg_recon'])
        self.interval_avg_kl_div_hist.append(current_metrics_for_hist['avg_kl_div'])
        self.interval_avg_d_total_hist.append(current_metrics_for_hist['avg_d_total'])
        # Only append val_metric if it's valid, otherwise, the deque might get non-finite values
        # which can break median calculations or trend binning.
        if current_metrics_for_hist['val_metric'] is not None and np.isfinite(current_metrics_for_hist['val_metric']):
             self.interval_val_metric_hist.append(current_metrics_for_hist['val_metric'])
        # If val_metric is None/NaN for the current interval, _get_trend_bin will handle it by returning a neutral bin.

        s_interval_recon_trend = self._get_trend_bin(self.interval_avg_recon_hist, current_metrics_for_hist['avg_recon'], value_scale_for_diff=0.1)
        s_interval_kl_trend = self._get_trend_bin(self.interval_avg_kl_div_hist, current_metrics_for_hist['avg_kl_div'], value_scale_for_diff=5.0) 
        s_interval_val_metric_trend = self._get_trend_bin(self.interval_val_metric_hist, current_metrics_for_hist['val_metric'], value_scale_for_diff=0.05) 
        
        s_current_lambda_kl_bin = np.digitize(current_metrics_for_hist['current_lambda_kl_val'], [1e-5, 1e-4, 0.001, 0.01, 0.1]).item() 
        s_interval_d_balance_level = np.digitize(current_metrics_for_hist['avg_d_total'], [0.15, 0.4, 0.7, 1.2]).item() 
        eps_bin = np.digitize(self.epsilon, [self.epsilon_min * 1.2, self.epsilon_start * 0.3, self.epsilon_start * 0.7]).item()

        state_tuple = ("LKL", s_interval_recon_trend, s_interval_kl_trend, s_interval_val_metric_trend,
                       s_current_lambda_kl_bin, s_interval_d_balance_level, eps_bin)
        self._ensure_q_state_exists(state_tuple)
        return state_tuple

    def _ensure_q_state_exists(self, state_tuple: tuple):
        # ... (same as before) ...
        current_time = time.time()
        self.q_table_access_count[state_tuple] += 1
        self.q_table_last_access_time[state_tuple] = current_time
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = { p_type: np.zeros(n_actions, dtype=np.float32)
                for p_type, n_actions in self.num_actions.items() }
            self.q_table_creation_time[state_tuple] = current_time
            self._manage_q_table_size() # Check size on new entry

    def choose_action(self, state: Optional[tuple], mode: str = 'lr_mom') -> Dict[str, float]:
        self._tick_exploration_boost() # Tick any active exploration boost
        default_actions = {'lr_scale': 1.0, 'momentum_scale': 1.0, 'lambda_kl_scale': 1.0}
        action_types_to_choose = []
        chosen_actions: Dict[str, float] = {}

        if mode == 'lr_mom':
            self._tick_probation_lr_mom()
            action_types_to_choose = ['lr_scale', 'momentum_scale']
            if self.on_probation:
                self.logger.debug(f"Q-Ctrl ({self.logger.name}, LR/Mom) on probation (step {self.current_probation_step}/{self.num_probation_steps}): Using neutral scales.")
                chosen_actions = {'lr_scale': 1.0, 'momentum_scale': 1.0}
                # self.prev_lr_mom_action is set by the caller (HybridTrainer)
                return chosen_actions
        elif mode == 'lambda_kl':
            self._tick_probation_lkl()
            action_types_to_choose = ['lambda_kl_scale']
            if self.lkl_on_probation:
                self.logger.debug(f"Q-Ctrl ({self.logger.name}, LambdaKL) on probation (step {self.lkl_current_probation_step}/{self.lkl_num_probation_steps}): Using neutral lambda_kl_scale.")
                chosen_actions = {'lambda_kl_scale': 1.0}
                # self.prev_lambda_kl_action is set by the caller (HybridTrainer)
                return chosen_actions
        else:
            raise ValueError(f"Invalid mode for choose_action: {mode}")

        if state is None or state not in self.q_table:
            self.logger.warning(f"Q-Ctrl ({self.logger.name}, Mode {mode}): State is None or not in Q-table. Using default actions.")
            chosen_actions = {k: default_actions[k] for k in action_types_to_choose}
            # Storing these defaults as prev_action is handled by HybridTrainer
            return chosen_actions

        # Epsilon decay happens here, only if not boosted currently
        if not (hasattr(self, 'epsilon_boost_active_steps') and self.epsilon_boost_active_steps > 0):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        for param_type in action_types_to_choose:
            q_values_arr = self.q_table[state].get(param_type) 
            action_space_arr = self.action_ranges[param_type]  

            if q_values_arr is None or not isinstance(q_values_arr, np.ndarray):
                self.logger.error(f"Q-values for {param_type} missing or not ndarray in state {state} for ({self.logger.name}). Choosing default.")
                chosen_actions[param_type] = default_actions[param_type]
                continue

            if random.random() < self.epsilon: # Use current epsilon (possibly boosted)
                chosen_idx = random.randrange(len(action_space_arr))
            else:
                finite_q_indices = np.where(np.isfinite(q_values_arr))[0]
                if finite_q_indices.size > 0:
                    best_q_val_among_finite = np.max(q_values_arr[finite_q_indices])
                    best_indices_options = np.where(
                        np.isclose(q_values_arr, best_q_val_among_finite) & np.isfinite(q_values_arr)
                    )[0]
                    chosen_idx = random.choice(best_indices_options) if best_indices_options.size > 0 else random.randrange(len(action_space_arr))
                else: 
                    chosen_idx = random.randrange(len(action_space_arr))
                    self.logger.warning(f"State {state}, PType {param_type} ({self.logger.name}): All Q-vals non-finite. Random action.")
            chosen_actions[param_type] = float(action_space_arr[chosen_idx])
        
        # Storing prev_action is done by HybridTrainer
        return chosen_actions

    def update_q_values(self, state: tuple, action: Dict[str, float], reward: float,
                        next_state: Optional[tuple], mode: str = 'lr_mom'):
        # ... (same as before) ...
        if state not in self.q_table:
            self.logger.warning(f"Updating Q for non-existent state ({self.logger.name}): {state}. Ensuring state exists.")
            self._ensure_q_state_exists(state)
            if state not in self.q_table: self.logger.error(f"Failed to ensure state {state} for Q-update ({self.logger.name})."); return
        if self.reward_clipping: reward = np.clip(reward, self.reward_clipping[0], self.reward_clipping[1])
        self.reward_hist.append(reward)
        for param_type, chosen_scale_value in action.items():
            if param_type not in self.action_ranges: continue
            action_idx_arr = np.where(np.isclose(self.action_ranges[param_type], chosen_scale_value))[0]
            if not action_idx_arr.size: continue
            action_idx = action_idx_arr[0]
            current_q = self.q_table[state][param_type][action_idx]
            max_future_q = 0.0
            if next_state and next_state in self.q_table and param_type in self.q_table[next_state]:
                next_q_vals = self.q_table[next_state][param_type]
                if np.any(np.isfinite(next_q_vals)): max_future_q = np.nanmax(next_q_vals[np.isfinite(next_q_vals)]) # Use nanmax for safety
            
            td_target = reward + self.gamma * max_future_q
            new_q = current_q + self.alpha * (td_target - current_q)
            if np.isfinite(new_q):
                if self.q_value_clipping: new_q = np.clip(new_q, self.q_value_clipping[0], self.q_value_clipping[1])
                self.q_table[state][param_type][action_idx] = new_q

    def _manage_q_table_size(self):
        # ... (same as before, LRU with recency/frequency weighting) ...
        if len(self.q_table) <= self.max_q_table_size: return
        num_to_prune = len(self.q_table) - self.max_q_table_size
        current_time = time.time()
        # Score: access_count * log(age) / log(time_since_last_access + 1)
        # Higher score is better to keep. Prune lowest scores.
        state_scores = { s_tuple: ( self.q_table_access_count.get(s_tuple, 1) *
                (1.0 + np.log1p((current_time - self.q_table_creation_time.get(s_tuple, current_time)) / 3600.0)) / # Favor older states slightly more
                (1.0 + np.log1p((current_time - self.q_table_last_access_time.get(s_tuple, current_time)) / 600.0)) # Penalize inactivity more
            ) for s_tuple in self.q_table.keys() }
        sorted_states_for_pruning = sorted(state_scores.keys(), key=lambda s: state_scores[s])
        pruned_count = 0
        for i in range(num_to_prune):
            if i < len(sorted_states_for_pruning):
                s_rm = sorted_states_for_pruning[i]
                self.q_table.pop(s_rm, None); self.q_table_access_count.pop(s_rm, None)
                self.q_table_creation_time.pop(s_rm, None); self.q_table_last_access_time.pop(s_rm, None)
                pruned_count +=1
        if pruned_count > 0: self.logger.info(f"Pruned {pruned_count} Q-table entries. New size: {len(self.q_table)}.")


    def compute_lr_mom_reward(self, current_losses: Dict[str, float], is_generator_q: bool) -> float:
        total_reward = 0.0
        w = self.reward_weights
        losses_to_use = {k: (100.0 if not np.isfinite(v) else np.clip(v, -500, 500)) 
                         for k,v in current_losses.items()}
        if any(not np.isfinite(v) for v in current_losses.values()):
            total_reward -= w["extreme_loss_penalty"] * 2.5

        def get_prev_median(hist_deque: deque, current_val_fallback: float) -> float:
            valid_hist = [v for v in hist_deque if v is not None and np.isfinite(v)]
            if not valid_hist: return current_val_fallback
            if len(valid_hist) > self.state_history_len // 2 + 1 and len(valid_hist) > 1 :
                 return np.median(valid_hist[:-1]) if len(valid_hist) > 1 else valid_hist[0]
            return np.median(valid_hist) if valid_hist else current_val_fallback

        if is_generator_q:
            loss_g_recon = losses_to_use.get('loss_g_recon', 1.0)
            prev_g_recon = get_prev_median(self.loss_g_recon_hist, loss_g_recon)
            recon_improvement = prev_g_recon - loss_g_recon
            recon_scale = 1.0 / (loss_g_recon + 0.01) 
            total_reward += np.tanh(recon_improvement * recon_scale * 15.0) * w["g_recon_improvement"]

            loss_g_adv = losses_to_use.get('loss_g_adv', 0.7)
            prev_g_adv = get_prev_median(self.loss_g_adv_hist, loss_g_adv)
            adv_improvement = prev_g_adv - loss_g_adv
            total_reward += np.tanh(adv_improvement / (abs(prev_g_adv) + 0.05 + EPS)) * w["g_adv_improvement"]

            loss_g_kl = losses_to_use.get('loss_g_kl', 0.0)
            if (self.current_lambda_kl * loss_g_kl > w.get("g_kl_control_penalty_ratio_trigger", 0.75) * loss_g_recon and
                loss_g_recon > 0.03 and loss_g_kl > w.get("g_kl_control_penalty_abs_trigger_low", 30.0)):
                total_reward -= w["g_kl_control_penalty"] * (1 + min(1.0, (loss_g_kl - w.get("g_kl_control_penalty_abs_trigger_low", 30.0)) / 100.0))

            loss_d_total_opp = losses_to_use.get('loss_d_total', 0.7)
            if 0.2 < loss_d_total_opp < 0.8: total_reward += w["gan_balance_g_bonus"]
            elif loss_d_total_opp <= 0.1: total_reward -= w["extreme_gan_imbalance_penalty_g"] * 2.0

            if loss_g_adv < w.get("g_easy_win_adv_thresh", 0.1) and loss_g_recon > w.get("g_easy_win_recon_thresh", 0.15):
                total_reward -= w.get("g_easy_win_recon_bad_penalty", 0.75) * (loss_g_recon / (w.get("g_easy_win_recon_thresh", 0.15)+EPS))

        else: # Discriminator Q
            loss_d_total = losses_to_use.get('loss_d_total', 0.7)
            if 0.2 < loss_d_total < 0.8: total_reward += w["d_balance_target"]
            elif loss_d_total < 0.1: total_reward -= w["d_balance_target"] * 1.5 
            elif loss_d_total > 1.5: total_reward -= w.get("d_very_weak_penalty", 1.0) * (1 + min(1.0, (loss_d_total - 1.5)/1.0))

            loss_d_real = losses_to_use.get('loss_d_real', 0.7)
            if loss_d_real < 0.15: total_reward += w["d_real_low_bonus"] * 1.5

            loss_d_fake = losses_to_use.get('loss_d_fake', 0.7)
            loss_g_adv_opp = losses_to_use.get('loss_g_adv', 0.7) 
            if loss_d_fake < 0.15 and loss_g_adv_opp > 0.7: 
                 total_reward += w["d_fake_low_meaningful_bonus"] * 1.5
            elif loss_d_fake > 2.0 and loss_g_adv_opp < 0.1: 
                total_reward -= w.get("d_misclassifies_fake_penalty", 1.2) * 2.0

            if loss_g_adv_opp < 0.05: total_reward -= w["gan_balance_d_penalty"] * 2.0
            
            # G stagnation penalty for D
            if len(self.loss_g_adv_hist) >= max(3, self.state_history_len-1): # Ensure enough history
                # Check if G_adv (from G's perspective, stored in self.loss_g_adv_hist) has been high
                g_adv_hist_for_check = list(self.loss_g_adv_hist)[-max(3, self.state_history_len//2):]
                if g_adv_hist_for_check and np.median(g_adv_hist_for_check) > w.get("g_stagnation_adv_high_thresh", 1.8): # New weight
                    if loss_d_total < w.get("g_stagnation_d_strong_thresh", 0.2): # And D is currently strong
                        total_reward -= w.get("g_stagnation_penalty_for_d", 0.3)


        if len(self.reward_hist) >= self.state_history_len:
            recent_q_rewards = list(self.reward_hist)[-max(5, self.state_history_len//2):] 
            if len(recent_q_rewards) > 2 : 
                sign_flips = 0
                for i in range(len(recent_q_rewards) - 1):
                    if (np.sign(recent_q_rewards[i]) != np.sign(recent_q_rewards[i+1]) and
                        abs(recent_q_rewards[i]) > 0.15 and abs(recent_q_rewards[i+1]) > 0.15):
                        sign_flips += 1
                if sign_flips >= (len(recent_q_rewards) // 2) :
                    total_reward -= w["oscillation_penalty"] * (sign_flips / (len(recent_q_rewards) -1))

        if len(self.reward_hist) >= 15: 
            if np.median(list(self.reward_hist)[-15:]) < w.get("q_learner_stagnation_penalty_trigger", -0.3):
                total_reward -= w.get("q_learner_stagnation_penalty", 0.25)

        if self.reward_clipping:
            total_reward = np.clip(total_reward, self.reward_clipping[0], self.reward_clipping[1])
        return float(total_reward)

    def compute_lambda_kl_reward(self, interval_metrics: Dict[str, Optional[float]],
                                 prev_interval_metrics: Optional[Dict[str, Optional[float]]]) -> float:
        total_reward = 0.0; w = self.reward_weights; _prev_metrics = prev_interval_metrics or {}
        
        required = ['val_metric', 'avg_recon', 'avg_kl_div', 'avg_d_total', 'current_lambda_kl_val']
        current_finite_metrics: Dict[str, float] = {}
        for key in required:
            val = interval_metrics.get(key)
            if val is None or not np.isfinite(val):
                self.logger.warning(f"LKL_Rew: Metric '{key}' missing/non-finite. Reward may be impacted."); return -0.2 # Return small penalty
            current_finite_metrics[key] = float(val)

        val_metric_imp = float(_prev_metrics.get('val_metric', current_finite_metrics['val_metric'])) - current_finite_metrics['val_metric']
        total_reward += np.tanh(val_metric_imp * 8.0) * w["lambda_kl_val_metric_improvement"]

        recon_imp = float(_prev_metrics.get('avg_recon', current_finite_metrics['avg_recon'])) - current_finite_metrics['avg_recon']
        recon_penalty_factor = 1.0 if recon_imp >= -0.02 else (1.0 + abs(recon_imp * 20))
        total_reward += np.tanh(recon_imp * 15.0 / recon_penalty_factor) * w["lambda_kl_recon_focus"]

        kl_low = w.get("lambda_kl_kl_target_range_low", 15.0); kl_high = w.get("lambda_kl_kl_target_range_high", 80.0)
        kl_div = current_finite_metrics['avg_kl_div']
        if kl_div < kl_low and current_finite_metrics['avg_recon'] > 0.04 : 
            total_reward -= w["lambda_kl_kl_target_range_bonus"] * (1.0 - kl_div/kl_low) * 0.75
        elif kl_div > kl_high: 
            kl_decrease = float(_prev_metrics.get('avg_kl_div', kl_div)) - kl_div
            total_reward += np.tanh(kl_decrease / (kl_high * 0.5)) * w["lambda_kl_kl_target_range_bonus"] 
        else: total_reward += w["lambda_kl_kl_target_range_bonus"] * 0.25 

        d_total_change = abs(current_finite_metrics['avg_d_total'] - float(_prev_metrics.get('avg_d_total', current_finite_metrics['avg_d_total'])))
        if d_total_change > 0.25: total_reward -= w["lambda_kl_stability_penalty"] * (d_total_change / 0.25) * 1.5

        current_lkl_val = current_finite_metrics['current_lambda_kl_val']
        if current_lkl_val > 0.1 and current_finite_metrics['avg_recon'] > 0.1:
             total_reward -= w.get("lambda_kl_too_high_recon_bad_penalty", 0.7)

        if self.logger.isEnabledFor(logging.DEBUG):
            log_mets_debug = {k: f'{v:.3f}' if isinstance(v, (float, np.float32)) and np.isfinite(v) else str(v) for k,v in interval_metrics.items()}
            self.logger.debug(f"LKL_Rew ({self.logger.name}): Raw={total_reward:.3f}. IntervalMet: {log_mets_debug}")

        return float(np.clip(total_reward, self.reward_clipping[0], self.reward_clipping[1])) if self.reward_clipping else float(total_reward)

    def set_current_lambda_kl(self, lambda_kl_val: float):
        if np.isfinite(lambda_kl_val): self.current_lambda_kl = float(lambda_kl_val)
        else: self.logger.warning(f"Attempt to set non-finite lambda_kl ({self.logger.name}): {lambda_kl_val}")

    def get_info(self) -> Dict:
        q_mem_mb = 0.0
        try:
            if self.q_table:
                q_mem_mb = sum( sys.getsizeof(s_tuple) + sum(q_vals.nbytes + sys.getsizeof(p_type) for p_type, q_vals in q_actions.items())
                    for s_tuple, q_actions in self.q_table.items() ) / (1024**2)
        except Exception as e_mem: self.logger.error(f"Error Q-table mem ({self.logger.name}): {e_mem}"); q_mem_mb = -1.0

        avg_reward_recent = np.mean(list(self.reward_hist)) if self.reward_hist else 0.0
        info_dict = {
            "epsilon": round(self.epsilon, 4), "q_table_size": len(self.q_table),
            "q_table_mem_mb_approx": round(q_mem_mb, 3),
            "last_lr_mom_action": self.prev_lr_mom_action or "None",
            "last_lambda_kl_action": self.prev_lambda_kl_action or "None",
            f"avg_reward_last_{self.reward_hist.maxlen}": round(avg_reward_recent, 3),
            "on_probation_lr_mom": self.on_probation, "probation_step_lr_mom": self.current_probation_step if self.on_probation else -1,
            "on_probation_lkl": self.lkl_on_probation, "probation_step_lkl": self.lkl_current_probation_step if self.lkl_on_probation else -1
        }
        if hasattr(self, 'epsilon_boost_active_steps') and self.epsilon_boost_active_steps > 0:
            info_dict["epsilon_boost_active_for_steps"] = self.epsilon_boost_active_steps
        return info_dict

    def set_initial_losses(self, losses: Dict[str, float], is_generator_q: bool):
        loss_map_init = { 'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist }
        relevant_keys = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total'] if is_generator_q \
                        else ['loss_d_total', 'loss_d_real', 'loss_d_fake', 'loss_g_total', 'loss_g_adv']
        
        needs_init = any(not loss_map_init[name] for name in relevant_keys if name in loss_map_init)
        if needs_init:
            self.logger.info(f"Initializing Q-Ctrl loss histories for {'G' if is_generator_q else 'D'} ({self.logger.name}).")
            for name in relevant_keys:
                deq = loss_map_init.get(name)
                if deq is not None and not deq: 
                    val = losses.get(name)
                    fill_val = val if val is not None and np.isfinite(val) else 1.0
                    if val is None or not np.isfinite(val): self.logger.warning(f"Missing/non-finite '{name}' for Q-hist init. Using {fill_val}.")
                    for _ in range(self.state_history_len): deq.append(fill_val)

    def set_initial_lambda_kl_metrics(self, interval_metrics: Dict[str, Optional[float]]):
        metric_map = { 'avg_recon': self.interval_avg_recon_hist, 'avg_kl_div': self.interval_avg_kl_div_hist,
            'avg_d_total': self.interval_avg_d_total_hist, 'val_metric': self.interval_val_metric_hist }
        
        needs_init_any = any(not deq for deq in metric_map.values())
        if needs_init_any:
            self.logger.info(f"Initializing Q-Ctrl Lambda_KL interval metrics histories ({self.logger.name}).")
            for name, deq in metric_map.items():
                if not deq: 
                    val = interval_metrics.get(name)
                    default_val = 1.0 
                    fill_val = float(val) if val is not None and np.isfinite(val) else default_val
                    if val is None or not np.isfinite(val): self.logger.warning(f"Missing/non-finite '{name}' for LKL Q-hist init. Using {fill_val}.")
                    for _ in range(self.lambda_kl_state_history_len): deq.append(fill_val)







class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, momentum: float = 0.9,
                 weight_decay: float = 0.01, max_grad_norm_risgd: float = 1.0,
                 q_learning_config: Optional[Dict] = None, optimizer_type: str = "generator"):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, initial_lr=lr, momentum=momentum, initial_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.full_optimizer_type_str = optimizer_type.lower() # Store the full descriptive string
        
        # Determine the core role for Q-learning and internal logic
        if "generator" in self.full_optimizer_type_str:
            self.core_optimizer_role = "generator"
        elif "discriminator" in self.full_optimizer_type_str:
            self.core_optimizer_role = "discriminator"
        else:
            # Fallback or error if the type string is completely unexpected
            # For safety, default to 'generator' or raise a more specific error if preferred
            self.logger_init_temp = logging.getLogger(f"WuBuSpecTransV01.RiSGD.InitCheck") # Temp logger
            self.logger_init_temp.warning(f"Unclear core role from optimizer_type '{optimizer_type}'. Defaulting to 'generator' for Q-role. Check type string.")
            self.core_optimizer_role = "generator" # Default, or raise ValueError("optimizer_type must clearly indicate 'generator' or 'discriminator'")

        if isinstance(q_learning_config, dict):
            q_params = q_learning_config.copy()
            if 'lkl_num_probation_steps' not in q_params:
                 q_params['lkl_num_probation_steps'] = max(3, q_params.get('lambda_kl_state_history_len', 5) + 1)
            self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_params)
        else:
            self.q_controller = None
            
        # Use the full descriptive string for the logger name for better identification
        self.logger = logging.getLogger(f"WuBuSpecTransV01.RiSGD.{self.full_optimizer_type_str.replace('_', ' ').title().replace(' ', '')}")
        if not self.logger.hasHandlers() and not logging.getLogger("WuBuSpecTransV01").hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
        self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled for {self.full_optimizer_type_str} optimizer (Core Role: {self.core_optimizer_role}).")
        
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
        
        # Use self.core_optimizer_role for Q-learning logic
        is_gen_q = (self.core_optimizer_role == "generator")
        
        req_keys_g = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        req_keys_d = ['loss_d_total', 'loss_g_total', 'loss_g_adv', 'loss_d_real', 'loss_d_fake']
        required_keys = req_keys_g if is_gen_q else req_keys_d
        
        if not all(key in finite_losses_for_q_state for key in required_keys):
            self.logger.debug(f"QCtrl ({self.full_optimizer_type_str}): Insufficient finite losses for LR/Mom state. Skipping Q-update. Need: {required_keys}, Got: {list(finite_losses_for_q_state.keys())}")
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
                    self.logger.warning(f"Optimizer step: Non-finite gradient for param shape {p.shape} ({self.full_optimizer_type_str}). Skipping update.")
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
                        self.logger.error(f"egrad2rgrad failed for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}): {e_egrad}. Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue
                    if not torch.isfinite(riemannian_grad).all():
                        self.logger.warning(f"Non-finite Riemannian grad for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None:
                            buf = torch.clone(riemannian_grad).detach()
                        else:
                            if buf.shape == riemannian_grad.shape:
                                buf.mul_(momentum).add_(riemannian_grad)
                            else: 
                                buf = torch.clone(riemannian_grad).detach()
                        self.state[p]['momentum_buffer'] = buf
                    else:
                        buf = riemannian_grad
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite momentum buffer for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Resetting.")
                        buf.zero_()
                        self.state[p]['momentum_buffer'] = buf
                    expmap_tangent_vector = buf.mul(-lr)
                    if not torch.isfinite(expmap_tangent_vector).all():
                        self.logger.warning(f"Non-finite tangent vector for expmap P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Skipping.")
                        continue
                    try:
                        new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
                        if not torch.isfinite(new_p_candidate).all():
                            self.logger.warning(f"Expmap resulted in non-finite P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Projecting and zeroing momentum.")
                            p.data = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0))
                            if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                        else:
                            p.data = manifold.proju(new_p_candidate)
                    except Exception as e_expmap:
                        self.logger.error(f"Expmap failed for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}): {e_expmap}. Zeroing momentum.")
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                        continue
                    if not torch.isfinite(p.data).all():
                        self.logger.error(f"Parameter P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}) became non-finite. Resetting to origin.")
                        p.data = manifold.expmap0(torch.zeros_like(p.data, device=p.device))
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                else: 
                    grad_eff_euc = grad.clone()
                    if weight_decay != 0: grad_eff_euc.add_(p, alpha=weight_decay)
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None:
                            buf = torch.clone(grad_eff_euc).detach()
                        else:
                            if buf.shape == grad_eff_euc.shape:
                                buf.mul_(momentum).add_(grad_eff_euc)
                            else: 
                                buf = torch.clone(grad_eff_euc).detach()
                        self.state[p]['momentum_buffer'] = buf
                    else:
                        buf = grad_eff_euc
                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite Euclidean momentum buffer for P:{p.shape} (opt: {self.full_optimizer_type_str}). Resetting.")
                        buf.zero_()
                        self.state[p]['momentum_buffer'] = buf
                    p.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all():
                        self.logger.warning(f"Euclidean P:{p.shape} (opt: {self.full_optimizer_type_str}) became non-finite. Clamping and zeroing momentum.")
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

        self.initial_gen_wubu_dim = args.encoder_initial_tangent_dim # Match encoder's embed dim for symmetry
        self.fc_expand_latent = nn.Linear(
            self.latent_dim,
            self.num_gaad_regions * self.initial_gen_wubu_dim
        )

        wubu_g_config = _configure_wubu_stack(args, "wubu_g")
        if wubu_g_config is None or wubu_g_config["num_levels"] == 0:
             self.logger.warning("WuBu-G config is None or num_levels is 0. Generator using MLP fallback.")
             # Fallback MLP needs to map from initial_gen_wubu_dim to num_dct_coeffs_flat per region
             self.wubu_generator = nn.Sequential(
                 nn.Linear(self.initial_gen_wubu_dim, self.initial_gen_wubu_dim * 2),
                 nn.GELU(),
                 nn.LayerNorm(self.initial_gen_wubu_dim * 2),
                 nn.Linear(self.initial_gen_wubu_dim * 2, self.num_dct_coeffs_flat)
             )
        else:
            self.wubu_generator = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=self.initial_gen_wubu_dim,
                output_tangent_dim=self.num_dct_coeffs_flat, # WuBu outputs DCTs for one region at a time
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
        if args_ref.dct_norm_type == "none":
            return norm_dct_coeffs
        elif args_ref.dct_norm_type == "global_scale":
            if not torch.isfinite(norm_dct_coeffs).all():
                norm_dct_coeffs = torch.nan_to_num(norm_dct_coeffs, nan=0.0, posinf=1.0, neginf=-1.0)
            scaled_output = norm_dct_coeffs * args_ref.dct_norm_global_scale
            if not torch.isfinite(scaled_output).all():
                finfo = torch.finfo(scaled_output.dtype)
                safe_max_val = min(finfo.max / 2 if finfo.max < float('inf') else TAN_VEC_CLAMP_VAL * 100, TAN_VEC_CLAMP_VAL * 10) 
                scaled_output = torch.clamp(
                    torch.nan_to_num(scaled_output, nan=0.0, posinf=safe_max_val, neginf=-safe_max_val),
                    min=-safe_max_val, max=safe_max_val
                )
            return scaled_output

        elif args_ref.dct_norm_type == "tanh":
            if not torch.is_tensor(norm_dct_coeffs):
                logging.error("_unnormalize_dct (tanh): CRITICAL! Input norm_dct_coeffs is not a tensor. Returning input.")
                return norm_dct_coeffs

            if not torch.isfinite(norm_dct_coeffs).all():
                norm_dct_coeffs = torch.nan_to_num(norm_dct_coeffs, nan=0.0, posinf=0.999, neginf=-0.999)
            
            input_dtype = norm_dct_coeffs.dtype
            device = norm_dct_coeffs.device
            one_tensor = torch.tensor(1.0, dtype=input_dtype, device=device)
            
            # Determine clamping bounds for atanh
            # Using a fixed epsilon for float16/bfloat16 as nextafter can be problematic.
            # For float32, nextafter is preferred.
            if input_dtype in [torch.float16, torch.bfloat16]:
                eps_clamp = torch.finfo(input_dtype).eps * 4 
                # Ensure the epsilon makes a difference; if 1.0 - eps is still 1.0, use a hardcoded value.
                upper_b = one_tensor - eps_clamp
                lower_b = -one_tensor + eps_clamp
                if upper_b >= one_tensor: upper_b = torch.tensor(0.999 if input_dtype == torch.float16 else 0.99, dtype=input_dtype, device=device)
                if lower_b <= -one_tensor: lower_b = torch.tensor(-0.999 if input_dtype == torch.float16 else -0.99, dtype=input_dtype, device=device)
            else: # float32 or other
                try:
                    upper_b = torch.nextafter(one_tensor, torch.tensor(0.0, dtype=input_dtype, device=device))
                    lower_b = torch.nextafter(-one_tensor, torch.tensor(0.0, dtype=input_dtype, device=device))
                except RuntimeError: # Fallback if nextafter fails for some other combo
                    eps_fallback = torch.finfo(input_dtype).eps * 10
                    upper_b = one_tensor - eps_fallback
                    lower_b = -one_tensor + eps_fallback

            clamped_for_atanh = torch.clamp(norm_dct_coeffs, min=lower_b, max=upper_b)
            
            compute_dtype_for_atanh = torch.float32 if input_dtype in [torch.float16, torch.bfloat16] else input_dtype
            atanh_output = torch.atanh(clamped_for_atanh.to(compute_dtype_for_atanh))
            unscaled_dct_intermediate = atanh_output * args_ref.dct_norm_tanh_scale

            if not torch.isfinite(unscaled_dct_intermediate).all():
                final_output_clamp_val = TAN_VEC_CLAMP_VAL 
                unscaled_dct_intermediate = torch.nan_to_num(
                    unscaled_dct_intermediate, nan=0.0, 
                    posinf=final_output_clamp_val, neginf=-final_output_clamp_val
                )
                unscaled_dct_intermediate = torch.clamp(unscaled_dct_intermediate, -final_output_clamp_val, final_output_clamp_val)

            final_unscaled_dct = unscaled_dct_intermediate.to(input_dtype)
            return final_unscaled_dct
        
        else:
            logging.error(f"_unnormalize_dct: CRITICAL! Unknown DCT norm type '{args_ref.dct_norm_type}'. Returning input scaled by global_scale.")
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


# Helper Self-Attention Module (simplified from SAGAN / ViT for 2D features)
# For a production model, you might use a more established implementation or nn.MultiheadAttention carefully adapted.
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, k_reduction_factor=8, use_spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // k_reduction_factor) # At least 1 channel

        conv_fn = functools.partial(nn.Conv2d, kernel_size=1, padding=0, bias=False)
        
        self.query_conv = conv_fn(self.in_channels, self.inter_channels)
        self.key_conv = conv_fn(self.in_channels, self.inter_channels)
        self.value_conv = conv_fn(self.in_channels, self.in_channels) # No reduction for value often
        self.out_conv = conv_fn(self.in_channels, self.in_channels)

        if use_spectral_norm:
            self.query_conv = spectral_norm(self.query_conv)
            self.key_conv = spectral_norm(self.key_conv)
            self.value_conv = spectral_norm(self.value_conv)
            self.out_conv = spectral_norm(self.out_conv) # SN on output conv can also be beneficial

        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling factor for attention output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(B, self.inter_channels, -1).permute(0, 2, 1)  # B x (H*W) x C_inter
        proj_key = self.key_conv(x).view(B, self.inter_channels, -1)  # B x C_inter x (H*W)
        energy = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(B, C, -1)  # B x C_in x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C_in x (H*W)
        out = out.view(B, C, H, W)

        out = self.out_conv(out)
        return self.gamma * out + x # Residual connection

# --- Now the updated Discriminator class ---

class _SingleScaleMelDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, disc_config: Dict, scale_index: int = 0):
        super().__init__()
        self.args = args
        self.scale_index = scale_index
        self.apply_spectral_norm = disc_config.get(f"mel_d_scale{scale_index}_apply_sn", disc_config.get("apply_spectral_norm", False))
        self.logger = logging.getLogger(f"WuBuSpecTransV01.SingleMelD.Scale{scale_index}.{id(self)}")

        n_mels_effective = args.n_mels // (2**scale_index) 
        n_time_effective = audio_config.get("num_time_frames_for_1s_segment", 87) // (2**scale_index)

        base_ch = disc_config.get(f"mel_d_scale{scale_index}_base_ch", disc_config.get("base_disc_channels", 64))
        max_ch = disc_config.get(f"mel_d_scale{scale_index}_max_ch", disc_config.get("max_disc_channels", 512))
        
        target_final_dim_config = disc_config.get(f"mel_d_scale{scale_index}_target_final_dim", disc_config.get("target_mel_disc_final_feature_dim", [4,4]))
        # Ensure target_final_dim_config is a list/tuple of 2 elements
        if isinstance(target_final_dim_config, int): target_final_dim_h = target_final_dim_w = target_final_dim_config
        elif isinstance(target_final_dim_config, (list, tuple)) and len(target_final_dim_config) == 2: target_final_dim_h, target_final_dim_w = target_final_dim_config
        elif isinstance(target_final_dim_config, (list, tuple)) and len(target_final_dim_config) == 1: target_final_dim_h = target_final_dim_w = target_final_dim_config[0]
        else: target_final_dim_h = target_final_dim_w = 4 # Default fallback

        max_downs_limit = disc_config.get(f"mel_d_scale{scale_index}_max_downs", disc_config.get("max_mel_disc_downsample_layers", 5))
        self.use_attention_in_mel_scale = getattr(args, f"mel_d_scale{scale_index}_use_attention", getattr(args, 'use_mel_d_attention', False)) # New arg
        self.attention_after_layer_idx = getattr(args, f"mel_d_scale{scale_index}_attention_idx", 2) # New arg: after which conv block

        cnn_layers_list = []
        in_c = 1 
        curr_h, curr_w = n_mels_effective, n_time_effective
        
        if curr_h <= target_final_dim_h and curr_w <= target_final_dim_w and curr_h > 0 and curr_w > 0: # check > 0
            num_downsamples = 0
            self.logger.info(f"  SingleMelD Scale {scale_index}: Input ({curr_h}x{curr_w}) small. No CNN downsampling. Using direct conv.")
        else:
            num_downsamples = 0
            temp_h, temp_w = curr_h, curr_w
            while (temp_h > target_final_dim_h or temp_w > target_final_dim_w) and num_downsamples < max_downs_limit and temp_h > 1 and temp_w > 1:
                next_h = (temp_h - 4 + 2*1) // 2 + 1 
                next_w = (temp_w - 4 + 2*1) // 2 + 1
                if (next_h < target_final_dim_h and next_w < target_final_dim_w and num_downsamples > 0) or next_h < 1 or next_w < 1:
                    if (temp_h <= target_final_dim_h or temp_w <= target_final_dim_w) and num_downsamples > 0: break
                if next_h < 1 or next_w < 1 : break
                temp_h, temp_w = next_h, next_w
                num_downsamples +=1
            num_downsamples = max(0, num_downsamples)

        self.logger.info(f"  SingleMelD Scale {scale_index}: Input Mel (H,W): ({n_mels_effective},{n_time_effective}). Target ~({target_final_dim_h}x{target_final_dim_w}). CNN Downs: {num_downsamples}.")

        for i in range(num_downsamples):
            out_c = min(base_ch * (2**i), max_ch)
            conv_l = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False) 
            if self.apply_spectral_norm: cnn_layers_list.append(spectral_norm(conv_l))
            else: cnn_layers_list.append(conv_l)
            cnn_layers_list.append(nn.InstanceNorm2d(out_c, affine=True)) 
            cnn_layers_list.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = out_c 
            curr_h = (curr_h - 4 + 2*1) // 2 + 1 if curr_h > 1 else 1
            curr_w = (curr_w - 4 + 2*1) // 2 + 1 if curr_w > 1 else 1
            if self.use_attention_in_mel_scale and i == self.attention_after_layer_idx and in_c > 0:
                self.logger.debug(f"  SingleMelD Scale {scale_index} Layer {i+1}: Adding SelfAttention2D with {in_c} channels.")
                cnn_layers_list.append(SelfAttention2D(in_c, use_spectral_norm=self.apply_spectral_norm))
        
        self.feature_extractor = nn.Sequential(*cnn_layers_list) if cnn_layers_list else nn.Identity()
        self.final_conv_in_channels = in_c # Channels going into the final conv
        
        # Final decision layer for this scale
        final_kernel_h = curr_h if curr_h > 0 else 1 # Ensure kernel dim > 0
        final_kernel_w = curr_w if curr_w > 0 else 1
        final_padding_h = 0 # No padding if kernel spans full feature map
        final_padding_w = 0
        
        # If feature map is still larger than 1x1, use a patch-style conv
        if curr_h > 1 or curr_w > 1:
             final_kernel_h = min(3, curr_h if curr_h > 0 else 1)
             final_kernel_w = min(3, curr_w if curr_w > 0 else 1)
             final_padding_h = final_kernel_h // 2
             final_padding_w = final_kernel_w // 2

        self.final_conv = nn.Conv2d(self.final_conv_in_channels, 1, 
                                    kernel_size=(final_kernel_h, final_kernel_w), 
                                    stride=1, 
                                    padding=(final_padding_h, final_padding_w), bias=True)
        if self.apply_spectral_norm:
            self.final_conv = spectral_norm(self.final_conv)
        
        self.logger.debug(f"  SingleMelD Scale {scale_index}: Feature Extractor Output C={self.final_conv_in_channels}, H={curr_h}, W={curr_w}. Final Conv Kernel=({final_kernel_h},{final_kernel_w})")

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.feature_extractor(x) # (B, C_final, H_feat, W_feat)
        patch_logits_map = self.final_conv(features) # (B, 1, H_patch_out, W_patch_out)
        
        if patch_logits_map.shape[2] > 1 or patch_logits_map.shape[3] > 1:
            logits = torch.mean(patch_logits_map, dim=[2,3], keepdim=False) # (B, 1) -> squeeze later
        else:
            logits = patch_logits_map # (B, 1, 1, 1) -> squeeze later

        if return_features:
            # For feature matching, typically use features before the final logit projection
            return logits, features 
        return logits


class AudioSpecDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.audio_config = audio_config
        self.gaad_config = gaad_config
        self.disc_config = disc_config 
        self.logger = logging.getLogger(f"WuBuSpecTransV01.Discriminator.{id(self)}") 

        self.num_gaad_regions = gaad_config['num_regions']
        self.region_proc_size_t = args.region_proc_size_t
        self.region_proc_size_f = args.region_proc_size_f
        self.num_dct_coeffs_flat = self.region_proc_size_t * self.region_proc_size_f

        self.input_type = self.disc_config.get("input_type", "mel")
        self.apply_spectral_norm = self.disc_config.get("apply_spectral_norm", False)
        self.use_global_stats_aux_input = getattr(args, 'disc_use_global_stats_aux', False)
        self.logger.info(f"Initializing Discriminator ({self.input_type} type). SpectralNorm: {self.apply_spectral_norm}, GlobalStatsAux: {self.use_global_stats_aux_input}")

        self.feature_extractor_module: nn.Module 
        self.final_decision_layer: Optional[nn.Module] = None # Can be None if MSD handles it

        if self.use_global_stats_aux_input:
            self.num_global_stats = 2 
            self.global_stats_mlp_hidden_dim = getattr(args, 'disc_global_stats_mlp_hidden_dim', 32) 
            self.global_stats_mlp = nn.Sequential(
                nn.Linear(self.num_global_stats, self.global_stats_mlp_hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.global_stats_mlp_hidden_dim, self.global_stats_mlp_hidden_dim)
            )
            if self.apply_spectral_norm:
                self.global_stats_mlp[0] = spectral_norm(self.global_stats_mlp[0])
                self.global_stats_mlp[2] = spectral_norm(self.global_stats_mlp[2])

        if self.input_type == "dct":
            self.dct_coeff_embed_dim = getattr(args, 'disc_dct_embed_dim', args.encoder_initial_tangent_dim) 
            self.dct_coeff_embed_disc = DCTCoeffEmbed(
                num_dct_coeffs_per_region=self.num_dct_coeffs_flat,
                embed_dim=self.dct_coeff_embed_dim
            )
            self.wubu_d_region_num_levels = getattr(args, 'wubu_d_region_num_levels', 1) 
            self.wubu_d_region_feature_dim = getattr(args, 'wubu_d_region_feature_dim', 128) 
            
            wubu_d_region_config = None
            if self.wubu_d_region_num_levels > 0:
                wubu_d_region_config = _configure_wubu_stack(args, "wubu_d_region") 
                if wubu_d_region_config.get("num_levels", 0) == 0: 
                    wubu_d_region_config = DEFAULT_CONFIG_WUBU.copy() 
                    wubu_d_region_config["num_levels"] = self.wubu_d_region_num_levels
                    wubu_d_region_config["hyperbolic_dims"] = [self.wubu_d_region_feature_dim] * self.wubu_d_region_num_levels
                    wubu_d_region_config["initial_curvatures"] = [0.5] * self.wubu_d_region_num_levels
                    for key_to_size in ["initial_scales", "initial_spread_values", "boundary_points_per_level"]:
                        if isinstance(wubu_d_region_config[key_to_size], list) and wubu_d_region_config[key_to_size]:
                            wubu_d_region_config[key_to_size] = [wubu_d_region_config[key_to_size][0]] * self.wubu_d_region_num_levels
                    num_transforms = max(0, self.wubu_d_region_num_levels - 1)
                    wubu_d_region_config["transform_types"] = [DEFAULT_CONFIG_WUBU["transform_types"][0]] * num_transforms if num_transforms > 0 and DEFAULT_CONFIG_WUBU["transform_types"] else []
                    wubu_d_region_config["transform_hidden_dims"] = [DEFAULT_CONFIG_WUBU["transform_hidden_dims"][0]] * num_transforms if num_transforms > 0 and DEFAULT_CONFIG_WUBU["transform_hidden_dims"] else []
                    self.logger.info(f"D-DCT: Using simplified default WuBu config for regional processor ({self.wubu_d_region_num_levels} levels).")

            if self.wubu_d_region_num_levels > 0 and wubu_d_region_config is not None and wubu_d_region_config.get("num_levels") > 0:
                self.logger.info(f"D-DCT: Regional Processor is WuBuNestingModel (In: {self.dct_coeff_embed_dim}, Out: {self.wubu_d_region_feature_dim}, Levels: {wubu_d_region_config.get('num_levels')}).")
                self.feature_extractor_module = FullyHyperbolicWuBuNestingModel(
                    input_tangent_dim=self.dct_coeff_embed_dim,
                    output_tangent_dim=self.wubu_d_region_feature_dim,
                    config=wubu_d_region_config
                )
            else: 
                self.logger.info(f"D-DCT: Regional Processor is MLP (In: {self.dct_coeff_embed_dim}, Out: {self.wubu_d_region_feature_dim}).")
                self.feature_extractor_module = nn.Sequential( 
                    nn.Linear(self.dct_coeff_embed_dim, self.wubu_d_region_feature_dim * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(self.wubu_d_region_feature_dim * 2, self.wubu_d_region_feature_dim),
                    nn.LayerNorm(self.wubu_d_region_feature_dim) 
                )

            self.use_region_pos_embed = getattr(args, 'disc_dct_use_pos_embed', True) 
            if self.use_region_pos_embed:
                self.region_pos_embed = nn.Parameter(torch.randn(1, self.num_gaad_regions, self.wubu_d_region_feature_dim))
            
            # [CLS] token for Transformer
            self.use_cls_token_dct_d = getattr(args, 'disc_dct_use_cls_token', True) # New arg
            if self.use_cls_token_dct_d:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.wubu_d_region_feature_dim))
                if self.use_region_pos_embed: # Adjust pos embedding if CLS token is used
                    self.region_pos_embed_eff = nn.Parameter(torch.randn(1, self.num_gaad_regions + 1, self.wubu_d_region_feature_dim))
            elif self.use_region_pos_embed:
                self.region_pos_embed_eff = self.region_pos_embed # Use original if no CLS

            self.disc_transformer_nhead = getattr(args, 'disc_transformer_nhead', 4) 
            self.disc_transformer_dim_feedforward = getattr(args, 'disc_transformer_dim_feedforward', self.wubu_d_region_feature_dim * 4) 
            self.disc_transformer_dropout = getattr(args, 'disc_transformer_dropout', 0.1) 
            self.disc_transformer_num_layers = getattr(args, 'disc_transformer_num_layers', 2) 

            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.wubu_d_region_feature_dim,
                nhead=self.disc_transformer_nhead,
                dim_feedforward=self.disc_transformer_dim_feedforward,
                dropout=self.disc_transformer_dropout,
                batch_first=True, 
                norm_first=getattr(args, 'disc_transformer_norm_first', True) 
            )
            self.context_transformer = nn.TransformerEncoder(
                transformer_encoder_layer, 
                num_layers=self.disc_transformer_num_layers
            )
            
            final_decision_input_dim = self.wubu_d_region_feature_dim
            if self.use_global_stats_aux_input:
                final_decision_input_dim += self.global_stats_mlp_hidden_dim

            self.final_decision_layer = nn.Linear(final_decision_input_dim, 1)
            if self.apply_spectral_norm:
                self.final_decision_layer = spectral_norm(self.final_decision_layer)
            
            self.logger.info(f"D-DCT: Using Transformer for aggregation (CLS token: {self.use_cls_token_dct_d}). Region feat: {self.wubu_d_region_feature_dim}, "
                             f"Transformer Layers: {self.disc_transformer_num_layers}, Heads: {self.disc_transformer_nhead}. "
                             f"Global stats aux input: {self.use_global_stats_aux_input}.")

        elif self.input_type == "mel":
            self.msd_num_scales = getattr(args, 'mel_d_msd_num_scales', 1) 
            self.msd_share_weights = getattr(args, 'mel_d_msd_share_weights', False) # New arg

            if self.msd_num_scales > 1:
                self.logger.info(f"D-Mel: Initializing Multi-Scale Discriminator with {self.msd_num_scales} scales. Share weights: {self.msd_share_weights}")
                if self.msd_share_weights:
                    # Create one _SingleScaleMelDiscriminator and reuse it
                    shared_single_scale_d = _SingleScaleMelDiscriminator(args, audio_config, self.disc_config, scale_index=0)
                    self.feature_extractor_module = nn.ModuleList([shared_single_scale_d] * self.msd_num_scales)
                else:
                    self.feature_extractor_module = nn.ModuleList()
                    for i in range(self.msd_num_scales):
                        self.feature_extractor_module.append(
                            _SingleScaleMelDiscriminator(args, audio_config, self.disc_config, scale_index=i)
                        )
                
                # For MSD, final decision typically involves combining outputs from scales.
                # If global_stats_aux is used, this combination needs a final linear layer.
                if self.use_global_stats_aux_input:
                    # We average the logits from each scale (each is 1 dim from sub-D) -> 1 scalar
                    # Then concatenate with projected_global_stats
                    self.final_decision_layer = nn.Linear(1 + self.global_stats_mlp_hidden_dim, 1)
                    if self.apply_spectral_norm:
                        self.final_decision_layer = spectral_norm(self.final_decision_layer)
                    self.logger.info(f"D-Mel (MSD, Aux): Outputs from {self.msd_num_scales} scales will be averaged. "
                                     f"Resulting scalar + GlobalStats({self.global_stats_mlp_hidden_dim}) -> Linear({1 + self.global_stats_mlp_hidden_dim}, 1).")
                else:
                    self.final_decision_layer = None # Logits are averaged directly from sub-discriminators
                    self.logger.info(f"D-Mel (MSD): Outputs from {self.msd_num_scales} scales will be averaged directly.")

            else: # Single-scale Mel Discriminator
                self.logger.info("D-Mel: Initializing Single-Scale Discriminator.")
                # Build the CNN backbone (self.feature_extractor_module) and final_decision_layer
                # (Copied and adapted from your previous version)
                n_mels = args.n_mels
                n_time = audio_config.get("num_time_frames_for_1s_segment", 87) 
                base_ch = self.disc_config.get("base_disc_channels", 64)
                max_ch = self.disc_config.get("max_disc_channels", 512)
                target_final_dim_config = self.disc_config.get("target_mel_disc_final_feature_dim", [4,4]) 
                target_final_dim_h, target_final_dim_w = target_final_dim_config[0], target_final_dim_config[1]
                max_downs_limit = self.disc_config.get("max_mel_disc_downsample_layers", 6) 
                use_attention_in_mel_single = getattr(args, 'use_mel_d_attention', False)
                attention_after_layer_idx_single = getattr(args, 'mel_d_attention_idx', 2)


                cnn_layers_list_single = []
                in_c_cnn = 1 
                curr_h, curr_w = n_mels, n_time
                num_downsamples = 0
                temp_h, temp_w = curr_h, curr_w
                while (temp_h > target_final_dim_h or temp_w > target_final_dim_w) and num_downsamples < max_downs_limit and temp_h > 1 and temp_w > 1 :
                    next_h = (temp_h - 4 + 2*1) // 2 + 1 
                    next_w = (temp_w - 4 + 2*1) // 2 + 1 
                    if (next_h < target_final_dim_h and next_w < target_final_dim_w and num_downsamples > 0) or next_h < 1 or next_w < 1:
                        if (temp_h <= target_final_dim_h or temp_w <= target_final_dim_w) and num_downsamples > 0: break
                    if next_h < 1 or next_w < 1 : break
                    temp_h, temp_w = next_h, next_w
                    num_downsamples +=1
                num_downsamples = max(1, num_downsamples) if (curr_h > target_final_dim_h or curr_w > target_final_dim_w) else 0


                for i in range(num_downsamples):
                    out_c = min(base_ch * (2**i), max_ch)
                    conv_l = nn.Conv2d(in_c_cnn, out_c, kernel_size=4, stride=2, padding=1, bias=False) 
                    if self.apply_spectral_norm: cnn_layers_list_single.append(spectral_norm(conv_l))
                    else: cnn_layers_list_single.append(conv_l)
                    cnn_layers_list_single.append(nn.InstanceNorm2d(out_c, affine=True)) 
                    cnn_layers_list_single.append(nn.LeakyReLU(0.2, inplace=True))
                    in_c_cnn = out_c 
                    curr_h = (curr_h - 4 + 2*1) // 2 + 1 if curr_h > 1 else 1
                    curr_w = (curr_w - 4 + 2*1) // 2 + 1 if curr_w > 1 else 1
                    if use_attention_in_mel_single and i == attention_after_layer_idx_single and in_c_cnn >0:
                        self.logger.debug(f"D-Mel (Single-Scale) Layer {i+1}: Adding SelfAttention2D with {in_c_cnn} channels.")
                        cnn_layers_list_single.append(SelfAttention2D(in_c_cnn, use_spectral_norm=self.apply_spectral_norm))
                
                self.feature_extractor_module = nn.Sequential(*cnn_layers_list_single) if cnn_layers_list_single else nn.Identity()
                
                # Final decision layer logic for single-scale Mel D
                final_decision_input_channels_single = in_c_cnn
                if self.use_global_stats_aux_input:
                    # CNN features are pooled, then concatenated with global stats for a Linear layer
                    self.global_pool_for_mel_d = nn.AdaptiveAvgPool2d(1) # Pools cnn_feature_map to (B, C_final, 1, 1)
                    self.final_decision_layer = nn.Linear(final_decision_input_channels_single + self.global_stats_mlp_hidden_dim, 1)
                    if self.apply_spectral_norm: self.final_decision_layer = spectral_norm(self.final_decision_layer)
                    self.logger.info(f"D-Mel (Single-Scale, Aux): CNN backbone C_out={final_decision_input_channels_single}. Pooled features + GlobalStats({self.global_stats_mlp_hidden_dim}) -> Linear.")
                else:
                    # Standard PatchGAN final conv
                    final_kernel_h_s = curr_h if curr_h > 0 else 1
                    final_kernel_w_s = curr_w if curr_w > 0 else 1
                    final_padding_h_s = 0; final_padding_w_s = 0
                    if curr_h > 1 or curr_w > 1: # If feature map not 1x1, use 3x3 patch conv
                         final_kernel_h_s = min(3, curr_h if curr_h > 0 else 1)
                         final_kernel_w_s = min(3, curr_w if curr_w > 0 else 1)
                         final_padding_h_s = final_kernel_h_s // 2
                         final_padding_w_s = final_kernel_w_s // 2
                    self.final_decision_layer = nn.Conv2d(final_decision_input_channels_single, 1, 
                                                          kernel_size=(final_kernel_h_s, final_kernel_w_s), 
                                                          stride=1, 
                                                          padding=(final_padding_h_s, final_padding_w_s), bias=True) 
                    if self.apply_spectral_norm: 
                        self.final_decision_layer = spectral_norm(self.final_decision_layer)
                    self.logger.info(f"D-Mel (Single-Scale): CNN C_out={final_decision_input_channels_single}. Final Conv HxW ({curr_h}x{curr_w}) -> 1ch patch map.")
        else:
            raise ValueError(f"Unsupported discriminator input_type: {self.input_type}")

        self.apply(init_weights_general)
        self.logger.info(f"AudioSpecDiscriminator ({id(self)}) fully initialized. Total Params: {sum(p.numel() for p in self.parameters()):,}")

    def _assemble_mel_from_dct_regions(self, dct_regions: torch.Tensor, gaad_bboxes: torch.Tensor,
                                       target_mel_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        B, N_Reg, F_p, T_p = dct_regions.shape
        _, C_target, H_target, W_target = target_mel_shape
        device = dct_regions.device; dtype = dct_regions.dtype

        if idct_2d is None or not TORCH_DCT_AVAILABLE:
            self.logger.error("idct_2d function is not available. Cannot assemble Mel. Returning zeros.")
            return torch.zeros(target_mel_shape, device=device, dtype=dtype)

        dct_regions_flat = dct_regions.reshape(-1, F_p, T_p)
        compute_dtype_for_idct = torch.float32 if dct_regions_flat.dtype in [torch.float16, torch.bfloat16] else dct_regions_flat.dtype
        spatial_regions_flat = idct_2d(dct_regions_flat.to(compute_dtype_for_idct)).to(dtype)
        spatial_regions = spatial_regions_flat.reshape(B, N_Reg, F_p, T_p)

        assembled_mel_canvas = torch.zeros(target_mel_shape, device=device, dtype=dtype)
        counts_canvas = torch.zeros(target_mel_shape, device=device, dtype=dtype) + EPS 

        for b in range(B):
            for r in range(N_Reg):
                t1_abs, f1_abs, t2_abs, f2_abs = gaad_bboxes[b, r].round().int().tolist()
                f1_clip = max(0, f1_abs); f2_clip = min(H_target, f2_abs)
                t1_clip = max(0, t1_abs); t2_clip = min(W_target, t2_abs)
                if t1_clip >= t2_clip or f1_clip >= f2_clip: continue

                current_spatial_region_patch = spatial_regions[b, r, :, :].unsqueeze(0).unsqueeze(0) 
                target_h_bbox_on_canvas = f2_clip - f1_clip
                target_w_bbox_on_canvas = t2_clip - t1_clip
                if target_h_bbox_on_canvas <= 0 or target_w_bbox_on_canvas <= 0: continue
                
                resized_region_patch = TF.resize(current_spatial_region_patch, 
                                                 (target_h_bbox_on_canvas, target_w_bbox_on_canvas),
                                                 interpolation=T.InterpolationMode.BILINEAR, antialias=True)
                
                assembled_mel_canvas[b, 0, f1_clip:f2_clip, t1_clip:t2_clip] += resized_region_patch.squeeze(0).squeeze(0)
                counts_canvas[b, 0, f1_clip:f2_clip, t1_clip:t2_clip] += 1.0
        
        assembled_mel_canvas = assembled_mel_canvas / counts_canvas
        return assembled_mel_canvas

    def _calculate_global_stats(self, data_for_stats: torch.Tensor) -> torch.Tensor:
        dims_to_reduce = tuple(range(1, data_for_stats.ndim))
        mean_stat = torch.mean(data_for_stats, dim=dims_to_reduce, keepdim=False)
        std_stat = torch.std(data_for_stats, dim=dims_to_reduce, keepdim=False)
        std_stat = torch.max(std_stat, torch.tensor(EPS, device=std_stat.device, dtype=std_stat.dtype))
        return torch.stack([mean_stat, std_stat], dim=-1) 

    def forward(self, input_data: torch.Tensor,
                gaad_bboxes_for_assembly: Optional[torch.Tensor] = None,
                target_mel_shape_for_assembly: Optional[Tuple[int,int,int,int]] = None,
                return_features: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = input_data.shape[0]
        device = input_data.device
        dtype = input_data.dtype
        main_features: Optional[torch.Tensor] = None # Features from the main path of D
        logits: torch.Tensor
        projected_global_stats: Optional[torch.Tensor] = None
        
        input_data_for_stats_calc: Optional[torch.Tensor] = None

        if self.input_type == "dct":
            if input_data.ndim == 4: flat_dct_coeffs = input_data.reshape(B, self.num_gaad_regions, -1)
            elif input_data.ndim == 3 and input_data.shape[-1] == self.num_dct_coeffs_flat: flat_dct_coeffs = input_data
            else: raise ValueError(f"D-DCT: Unsupported input_data shape {input_data.shape}.")
            input_data_for_stats_calc = flat_dct_coeffs 

            embedded_coeffs = self.dct_coeff_embed_disc(flat_dct_coeffs) 
            regional_input_flat = embedded_coeffs.reshape(B * self.num_gaad_regions, self.dct_coeff_embed_dim)
            regional_features_flat = self.feature_extractor_module(regional_input_flat) 
            regional_features_seq = regional_features_flat.reshape(B, self.num_gaad_regions, self.wubu_d_region_feature_dim)

            transformer_input_seq = regional_features_seq
            if self.use_region_pos_embed and hasattr(self, 'region_pos_embed_eff'):
                 if self.use_cls_token_dct_d and hasattr(self, 'cls_token'):
                    cls_tokens = self.cls_token.expand(B, -1, -1)
                    transformer_input_seq = torch.cat((cls_tokens, regional_features_seq), dim=1)
                    transformer_input_seq = transformer_input_seq + self.region_pos_embed_eff[:, :(self.num_gaad_regions + 1), :]
                 else: # No CLS token, but pos_embed is used for regions
                    transformer_input_seq = regional_features_seq + self.region_pos_embed_eff[:, :self.num_gaad_regions, :]
            
            transformer_output = self.context_transformer(transformer_input_seq) 
            
            if self.use_cls_token_dct_d and hasattr(self, 'cls_token'):
                main_features = transformer_output[:, 0] # Take the CLS token output
            else:
                main_features = transformer_output.mean(dim=1) 
            
            if self.use_global_stats_aux_input and input_data_for_stats_calc is not None:
                global_stats_raw = self._calculate_global_stats(input_data_for_stats_calc)
                projected_global_stats = self.global_stats_mlp(global_stats_raw)
                combined_features_for_decision = torch.cat([main_features, projected_global_stats], dim=-1)
                logits = self.final_decision_layer(combined_features_for_decision)
            else:
                logits = self.final_decision_layer(main_features) 

        elif self.input_type == "mel":
            mel_input_for_d: torch.Tensor
            if input_data.ndim == 4 and input_data.shape[1] == self.num_gaad_regions : # DCTs provided
                if gaad_bboxes_for_assembly is None or target_mel_shape_for_assembly is None:
                    raise ValueError("GAAD bboxes and target_mel_shape needed for D (mel type) with DCT region input.")
                unnorm_dct_coeffs = AudioSpecGenerator._unnormalize_dct(input_data, self.args)
                mel_input_for_d = self._assemble_mel_from_dct_regions(
                    unnorm_dct_coeffs, gaad_bboxes_for_assembly, target_mel_shape_for_assembly
                )
                input_data_for_stats_calc = mel_input_for_d 
            elif input_data.ndim == 4 and input_data.shape[1] == 1: 
                mel_input_for_d = input_data
                input_data_for_stats_calc = mel_input_for_d 
            else:
                raise ValueError(f"D-Mel: Unsupported input_data shape {input_data.shape}.")

            if self.msd_num_scales > 1 and isinstance(self.feature_extractor_module, nn.ModuleList): 
                all_scale_logits = []
                all_scale_features_for_matching = [] 
                current_mel_scale = mel_input_for_d
                for i, sub_d_module in enumerate(self.feature_extractor_module):
                    # Pass return_features=True to get both logits and backbone features from sub-discriminator
                    sub_logits_raw, sub_backbone_features = sub_d_module(current_mel_scale, return_features=True)
                    all_scale_logits.append(sub_logits_raw.squeeze()) # Ensure (B)
                    if i == 0: # Use features from the highest-resolution scale for feature matching
                        main_features = sub_backbone_features 
                    if i < self.msd_num_scales - 1:
                        current_mel_scale = F.avg_pool2d(current_mel_scale, kernel_size=3, stride=2, padding=1, count_include_pad=False)
                
                averaged_msd_logits = torch.stack(all_scale_logits, dim=0).mean(dim=0) # (B)
                
                if self.use_global_stats_aux_input and input_data_for_stats_calc is not None:
                    global_stats_raw = self._calculate_global_stats(input_data_for_stats_calc)
                    projected_global_stats = self.global_stats_mlp(global_stats_raw) # (B, global_stats_mlp_hidden_dim)
                    # Concatenate the averaged MSD logit (scalar per batch item) with global stats
                    combined_input_for_decision = torch.cat([averaged_msd_logits.unsqueeze(1), projected_global_stats], dim=-1) # (B, 1 + global_stats_mlp_hidden_dim)
                    logits = self.final_decision_layer(combined_input_for_decision) # final_decision_layer is Linear
                else:
                    logits = averaged_msd_logits # (B)
            
            else: # Single-Scale Mel D
                cnn_feature_map = self.feature_extractor_module(mel_input_for_d) 
                main_features = cnn_feature_map 
                
                if self.use_global_stats_aux_input and input_data_for_stats_calc is not None:
                    # Here, final_decision_layer is Linear, patch_head_conv gets patch logits first
                    patch_logits_map = self.patch_head_conv(cnn_feature_map) 
                    averaged_patch_logit = torch.mean(patch_logits_map, dim=[2,3], keepdim=False) 
                    
                    global_stats_raw = self._calculate_global_stats(input_data_for_stats_calc)
                    projected_global_stats = self.global_stats_mlp(global_stats_raw) 
                    combined_input_for_decision = torch.cat([averaged_patch_logit, projected_global_stats], dim=-1) 
                    logits = self.final_decision_layer(combined_input_for_decision)
                else: # Standard Single-Scale PatchGAN
                    patch_logits_map = self.final_decision_layer(cnn_feature_map) 
                    if patch_logits_map.shape[2] > 1 or patch_logits_map.shape[3] > 1:
                        logits = torch.mean(patch_logits_map, dim=[2,3], keepdim=False) 
                    else:
                        logits = patch_logits_map # Will be squeezed later
        else:
            raise NotImplementedError(f"Discriminator forward not implemented for type {self.input_type}")
        
        # --- Final Logit Shaping ---
        if logits.ndim > 1 and logits.shape[-1] == 1: 
            logits = logits.squeeze(-1) 
        elif logits.ndim == 0 and B == 1: 
            logits = logits.unsqueeze(0) 
        
        if logits.ndim > 1:
            self.logger.warning(f"Discriminator final output logits have unexpected shape {logits.shape} after squeeze attempts. Averaging over the last dimension.")
            logits = torch.mean(logits, dim=-1)
        
        if return_features:
            if main_features is None: 
                self.logger.warning("return_features=True but main_features (intermediate features) are None. Using logits as fallback.")
                fallback_features = logits.detach().clone().unsqueeze(-1) if logits.ndim ==1 else logits.detach().clone()
                return logits, fallback_features
            return logits, main_features
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
                 model: "WuBuSpecTransNet", # Model is already on device and DDP wrapped
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 args: argparse.Namespace,
                 rank: int,
                 world_size: int,
                 ddp_active: bool):

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args # args from parse_arguments is stored here
        self.rank = rank
        self.world_size = world_size
        self.ddp_active = ddp_active
        self.am_main_process = (rank == 0)
        self.logger = logging.getLogger("WuBuSpecTransV01.Trainer")

        # --- 1. Initialize Optimizers (including optimizer_enc_gen) ---
        q_cfg_gen = None
        if args.q_controller_enabled:
            q_cfg_gen = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
            if 'lkl_num_probation_steps' not in q_cfg_gen:
                 q_cfg_gen['lkl_num_probation_steps'] = max(3, q_cfg_gen.get('lambda_kl_state_history_len', 5) + 1)
        
        self.optimizer_enc_gen: RiemannianEnhancedSGD = RiemannianEnhancedSGD(
            self.model.parameters(),
            lr=self.args.learning_rate_gen,
            q_learning_config=q_cfg_gen,
            max_grad_norm_risgd=self.args.risgd_max_grad_norm,
            optimizer_type="generator"
        )
        if self.am_main_process: self.logger.info("Optimizer_Enc_Gen initialized.")
        self.q_controller_gen = getattr(self.optimizer_enc_gen, 'q_controller', None) 

        # --- 2. Discriminator Setup ---
        if self.am_main_process: self.logger.info("Initializing Discriminators and their Optimizers...")
        self.initial_disc_type_arg = args.initial_disc_type if args.initial_disc_type is not None else args.disc_input_type
        self.alternative_disc_type_arg = 'mel' if self.initial_disc_type_arg == 'dct' else 'dct'
        if self.am_main_process:
            self.logger.info(f"Intended Initial Active D type (from args): '{self.initial_disc_type_arg}'")
            self.logger.info(f"Intended Alternative D type: '{self.alternative_disc_type_arg}'")

        # _get_discriminator_configs will now use self.args which contains all parsed arguments
        primary_disc_config_dict, primary_wubu_d_config = self._get_discriminator_configs(self.args, self.initial_disc_type_arg, is_primary=True)
        alt_disc_config_dict, alt_wubu_d_config = self._get_discriminator_configs(self.args, self.alternative_disc_type_arg, is_primary=False)
        
        primary_disc_config_dict["wubu_stack_config"] = primary_wubu_d_config
        alt_disc_config_dict["wubu_stack_config"] = alt_wubu_d_config

        # Pass self.args to AudioSpecDiscriminator so it can access all arguments
        self.discriminator_primary_obj = AudioSpecDiscriminator(self.args, self._get_audio_config_ref(), self._get_gaad_config_ref(), primary_disc_config_dict).to(device)
        self.discriminator_alternative_obj = AudioSpecDiscriminator(self.args, self._get_audio_config_ref(), self._get_gaad_config_ref(), alt_disc_config_dict).to(device)

        self.primary_disc_actual_type = primary_disc_config_dict.get("input_type", "unknown_primary_type")
        self.alternative_disc_actual_type = alt_disc_config_dict.get("input_type", "unknown_alt_type")
        if self.am_main_process:
            self.logger.info(f"Primary D (intended as '{self.initial_disc_type_arg}', actual type: '{self.primary_disc_actual_type}') initialized. Params: {sum(p.numel() for p in self.discriminator_primary_obj.parameters()):,}")
            self.logger.info(f"Alternative D (intended as '{self.alternative_disc_type_arg}', actual type: '{self.alternative_disc_actual_type}') initialized. Params: {sum(p.numel() for p in self.discriminator_alternative_obj.parameters()):,}")

        if self.ddp_active:
            local_rank_ddp = self.args.local_rank if hasattr(self.args, 'local_rank') and self.args.local_rank != -1 else rank
            ddp_find_unused_d = getattr(self.args, 'ddp_find_unused_params_d', False)
            self.discriminator_primary_obj = DDP(self.discriminator_primary_obj, device_ids=[local_rank_ddp], output_device=local_rank_ddp, find_unused_parameters=ddp_find_unused_d)
            self.discriminator_alternative_obj = DDP(self.discriminator_alternative_obj, device_ids=[local_rank_ddp], output_device=local_rank_ddp, find_unused_parameters=ddp_find_unused_d)
            if self.am_main_process: self.logger.info(f"Discriminators DDP wrapped (find_unused_parameters={ddp_find_unused_d}).")

        q_cfg_disc_shared = None
        if args.q_controller_enabled:
            q_cfg_disc_shared = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
            if 'lkl_num_probation_steps' not in q_cfg_disc_shared:
                 q_cfg_disc_shared['lkl_num_probation_steps'] = max(3, q_cfg_disc_shared.get('lambda_kl_state_history_len', 5) + 1)
        
        lr_disc_alt = getattr(args, 'learning_rate_disc_alt', args.learning_rate_disc)
        self.optimizer_disc_primary = RiemannianEnhancedSGD(
            self.discriminator_primary_obj.parameters(), lr=args.learning_rate_disc,
            q_learning_config=q_cfg_disc_shared.copy() if q_cfg_disc_shared else None,
            max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type=f"discriminator_primary_{self.primary_disc_actual_type}"
        )
        self.optimizer_disc_alternative = RiemannianEnhancedSGD(
            self.discriminator_alternative_obj.parameters(), lr=lr_disc_alt,
            q_learning_config=q_cfg_disc_shared.copy() if q_cfg_disc_shared else None,
            max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type=f"discriminator_alt_{self.alternative_disc_actual_type}"
        )
        if self.am_main_process: self.logger.info("Discriminator optimizers initialized.")

        self.q_controller_d_primary = getattr(self.optimizer_disc_primary, 'q_controller', None)
        self.q_controller_d_alt = getattr(self.optimizer_disc_alternative, 'q_controller', None)
        
        self.active_discriminator_key = 'primary' 
        if self.args.enable_heuristic_disc_switching:
            if self.initial_disc_type_arg == self.primary_disc_actual_type:
                self.active_discriminator_key = 'primary'
            elif self.initial_disc_type_arg == self.alternative_disc_actual_type:
                self.active_discriminator_key = 'alternative'
            else:
                 if self.am_main_process: self.logger.warning(
                     f"Mismatch or ambiguity in initial active D. args.initial_disc_type ('{self.args.initial_disc_type}') led to "
                     f"initial_disc_type_arg ('{self.initial_disc_type_arg}'). Defaulting to 'primary'."
                 )
        
        self.q_controller_d_active = None 
        self._update_active_discriminator_pointers() 
        
        self.lambda_recon = args.lambda_recon
        self.lambda_kl = args.lambda_kl 
        self.lambda_gan = args.lambda_gan

        self.scaler_enc_gen = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.scaler_disc = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')

        self.global_step = 0; self.current_epoch = 0
        self.is_val_metric_higher_better = self.args.val_primary_metric in ["avg_val_ssim_mel", "avg_val_psnr_mel"]
        self.best_val_metric_val = -float('inf') if self.is_val_metric_higher_better else float('inf')
        self.last_val_metrics: Dict[str, Any] = {}
        self.prev_interval_metrics_for_lambda_kl_reward: Optional[Dict[str, Union[float, None]]] = None
        if self.am_main_process: os.makedirs(args.checkpoint_dir, exist_ok=True)

        self.lpips_loss_fn = None; self.ssim_metric = None
        if self.am_main_process:
            if self.args.use_lpips_for_mel_verification and LPIPS_AVAILABLE and lpips is not None:
                try: self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device) 
                except Exception as e: self.logger.warning(f"LPIPS init failed: {e}")
            if TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
                try: self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                except Exception as e: self.logger.warning(f"SSIM init failed: {e}")
        
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))
        self.fixed_noise_for_sampling: Optional[torch.Tensor] = None

        self.lambda_kl_update_interval = getattr(args, 'lambda_kl_update_interval', 0)
        self.lambda_kl_q_controller: Optional[HAKMEMQController] = None
        if self.args.q_controller_enabled and self.lambda_kl_update_interval > 0:
            q_cfg_lkl = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
            q_cfg_lkl["lambda_kl_scale_options"] = getattr(args, 'q_lkl_scale_options', [0.85, 0.95, 1.0, 1.05, 1.15])
            q_cfg_lkl["num_probation_steps"] = getattr(args, 'q_lkl_lr_mom_probation_steps', q_cfg_lkl.get('state_history_len', 5) + 3)
            q_cfg_lkl['lkl_num_probation_steps'] = getattr(args, 'q_lkl_action_probation_steps', max(3, q_cfg_lkl.get('lambda_kl_state_history_len', 5) + 2))
            self.lambda_kl_q_controller = HAKMEMQController(**q_cfg_lkl)
            if self.am_main_process: self.logger.info(f"Separate Lambda_KL Q-Control ENABLED. Update interval: {self.lambda_kl_update_interval} global steps.")
            if hasattr(self.lambda_kl_q_controller, 'set_current_lambda_kl'): self.lambda_kl_q_controller.set_current_lambda_kl(self.lambda_kl)

        self.interval_metrics_accum: Dict[str, float] = defaultdict(float)
        self.interval_steps_count = 0
        self.min_lambda_kl_q_control = getattr(args, 'min_lambda_kl_q_control', 1e-7)
        self.max_lambda_kl_q_control = getattr(args, 'max_lambda_kl_q_control', 0.5)

        # --- Heuristic Setup ---
        self.enable_heuristic_interventions = getattr(args, 'enable_heuristic_interventions', False) 
        self.enable_heuristic_disc_switching = getattr(args, 'enable_heuristic_disc_switching', False)

        if self.am_main_process:
            self.logger.info(f"HybridTrainer Init: args.enable_heuristic_interventions = {args.enable_heuristic_interventions}")
            self.logger.info(f"HybridTrainer Init: self.enable_heuristic_interventions (instance var) = {self.enable_heuristic_interventions}")
            self.logger.info(f"HybridTrainer Init: args.enable_heuristic_disc_switching = {args.enable_heuristic_disc_switching}")
            self.logger.info(f"HybridTrainer Init: self.enable_heuristic_disc_switching (instance var) = {self.enable_heuristic_disc_switching}")
            
        self.heuristic_check_interval = args.heuristic_check_interval if args.heuristic_check_interval is not None else \
                                        (args.disc_switch_check_interval if self.enable_heuristic_disc_switching else args.log_interval)
        
        self.disc_switch_min_steps_between = args.disc_switch_min_steps_between
        self.disc_switch_problem_state_count_thresh = args.disc_switch_problem_state_count_thresh
        self.steps_since_last_d_switch = 0
        self.consecutive_trigger_primary_to_alt_count = 0 
        self.consecutive_trigger_alt_to_primary_count = 0
        self.consecutive_heuristic_trigger_counts: Dict[str, int] = defaultdict(int)

        self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS = getattr(args, 'heuristic_short_term_history_len', 7)
        self.avg_g_recon_hist_for_stagnation = deque(maxlen=self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS)
        self.q_data_derived_g_recon_hist = deque(maxlen=self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS)
        self.rec_dct_stagnant = False

        self.D_STRONG_THRESH = getattr(args, 'heuristic_d_strong_thresh', 0.25) 
        self.D_WEAK_THRESH = getattr(args, 'heuristic_d_weak_thresh', 1.0)    
        self.D_VERY_WEAK_THRESH = getattr(args, 'heuristic_d_very_weak_thresh', 1.8) 
        self.G_STALLED_THRESH = getattr(args, 'heuristic_g_stalled_thresh', 1.5) 
        self.G_WINNING_THRESH = getattr(args, 'heuristic_g_winning_thresh', 0.2) 
        self.G_VERY_MUCH_WINNING_THRESH = getattr(args, 'heuristic_g_very_much_winning_thresh', 0.05)
        self.KL_HIGH_THRESH = getattr(args, 'heuristic_kl_high_thresh', 25.0)
        self.RECON_STAGNATION_IMPROVEMENT_THRESH_REL = getattr(args, 'heuristic_recon_stagnation_improvement_thresh_rel', 0.001)
        self.TARGET_GOOD_RECON_THRESH_HEURISTIC = getattr(args, 'target_good_recon_thresh_heuristic', 0.03)
        self.Q_REWARD_STAGNATION_THRESH = getattr(args, 'heuristic_q_reward_stagnation_thresh', -0.25)
        self.HEURISTIC_TRIGGER_COUNT_THRESH = getattr(args, 'heuristic_trigger_count_thresh', 2) 

        self.heuristic_vae_feature_match_active = False
        self.heuristic_penalize_g_easy_win_active = False
        self.heuristic_boost_active_d_lr_active = False
        self.heuristic_force_d_q_explore_active = False

        self.heuristic_override_lambda_recon_factor = 1.0
        self.heuristic_override_lambda_kl_factor = 1.0 
        self.heuristic_override_lambda_gan_factor = 1.0 
        self.lambda_feat_match_heuristic = getattr(args, 'lambda_feat_match_heuristic', 0.75)
        self.lambda_g_easy_win_penalty_heuristic = getattr(args, 'lambda_g_easy_win_penalty_heuristic', 1.5)
        self.heuristic_active_d_lr_boost_factor = getattr(args, 'heuristic_active_d_lr_boost_factor', 1.8)
        self.heuristic_d_q_explore_boost_epsilon = getattr(args, 'heuristic_d_q_explore_boost_epsilon', 0.7)
        self.heuristic_d_q_explore_duration = getattr(args, 'heuristic_d_q_explore_duration', 10)
        self.target_d_features_for_vae_boost: Optional[torch.Tensor] = None

        if self.am_main_process:
             self.logger.info(f"HybridTrainer initialized. Initial Active D: '{self.active_discriminator_key}' (Type: '{self.active_disc_actual_type}'). Heuristics {'ENABLED' if self.enable_heuristic_interventions else 'DISABLED'}.")

    def _get_audio_config_ref(self) -> Dict:
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        return getattr(m_ref, 'audio_config', {})

    def _get_gaad_config_ref(self) -> Dict:
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        return getattr(m_ref, 'gaad_config', {})

    def _get_discriminator_configs(self, current_args: argparse.Namespace, disc_type_to_config: str, is_primary: bool) -> Tuple[Dict, Optional[Dict]]:
        # This method now correctly uses `current_args` which is `self.args` passed from `__init__`
        disc_config = {
            "input_type": disc_type_to_config,
            "apply_spectral_norm": current_args.disc_apply_spectral_norm, 
            "base_disc_channels": current_args.disc_base_disc_channels,
            "max_disc_channels": current_args.disc_max_disc_channels,
            # Use getattr for disc_target_final_feature_dim as it's a list in current_args
            "target_mel_disc_final_feature_dim": getattr(current_args, 'disc_target_final_feature_dim', [4,4]), # Default to [4,4] if not found
            "max_mel_disc_downsample_layers": getattr(current_args, 'max_mel_disc_downsample_layers', 6), # Default from args
            "disc_use_global_stats_aux": getattr(current_args, 'disc_use_global_stats_aux', False), # For both D types
            "disc_global_stats_mlp_hidden_dim": getattr(current_args, 'disc_global_stats_mlp_hidden_dim', 32) # For both D types
        }
        
        wubu_d_config_for_this_d = None 
        if disc_type_to_config == 'dct':
            # Determine which set of wubu_d_* args to use based on whether it's primary or alternative
            # And if specific "alt" args are even defined by the user.
            wubu_prefix_to_try: Optional[str] = None
            if is_primary:
                wubu_prefix_to_try = "wubu_d" # For primary D, always try "wubu_d" first
            else: # For alternative D
                if hasattr(current_args, "wubu_d_alt_num_levels") and current_args.wubu_d_alt_num_levels is not None and current_args.wubu_d_alt_num_levels > 0 :
                    wubu_prefix_to_try = "wubu_d_alt"
                    self.logger.info(f"Configuring Alternative DCT D using specific '{wubu_prefix_to_try}_*' args.")
                else: # No specific alt config, or it's 0 levels. Try primary D's WuBu config for alt D.
                    wubu_prefix_to_try = "wubu_d" 
                    self.logger.info(f"Alternative DCT D: No specific/valid 'wubu_d_alt_*' args. Trying primary 'wubu_d_*' args for its WuBu stack.")

            if wubu_prefix_to_try:
                 wubu_d_config_for_this_d = _configure_wubu_stack(current_args, wubu_prefix_to_try)

            # If, after trying, the config is still None or 0 levels (e.g., primary wubu_d had 0 levels)
            # and this is for the *alternative* D, then create a very simple default for it.
            if not is_primary and (wubu_d_config_for_this_d is None or wubu_d_config_for_this_d.get("num_levels", 0) == 0):
                self.logger.warning(f"Alternative DCT D: Config from '{wubu_prefix_to_try}_*' resulted in 0 levels or was None. Using simplified default for alt DCT D.")
                wubu_d_config_for_this_d = DEFAULT_CONFIG_WUBU.copy()
                wubu_d_config_for_this_d["num_levels"] = 1 # Minimal
                default_hyp_dim = max(16, getattr(current_args, 'disc_dct_embed_dim', 128) // 4)
                wubu_d_config_for_this_d["hyperbolic_dims"] = [default_hyp_dim]
                wubu_d_config_for_this_d["initial_curvatures"] = [0.25]
                wubu_d_config_for_this_d["initial_scales"] = [0.5]
                wubu_d_config_for_this_d["boundary_points_per_level"] = [0]
                wubu_d_config_for_this_d["tangent_input_combination_dims"] = [max(16, default_hyp_dim // 2)]
                wubu_d_config_for_this_d["transform_types"] = []
                wubu_d_config_for_this_d["transform_hidden_dims"] = []
                wubu_d_config_for_this_d["use_tangent_flow"] = False
                wubu_d_config_for_this_d["use_rotation_in_transform"] = False
                wubu_d_config_for_this_d["dropout"] = 0.0
        
        return disc_config, wubu_d_config_for_this_d


    def _update_active_discriminator_pointers(self):
        if self.active_discriminator_key == 'primary':
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_actual_type = self.primary_disc_actual_type
            self.q_controller_d_active = self.q_controller_d_primary
        elif self.active_discriminator_key == 'alternative':
            self.active_discriminator = self.discriminator_alternative_obj
            self.optimizer_disc_active = self.optimizer_disc_alternative
            self.active_disc_actual_type = self.alternative_disc_actual_type
            self.q_controller_d_active = self.q_controller_d_alt
        else:
            self.logger.error(f"Invalid active_discriminator_key: {self.active_discriminator_key}. Defaulting to primary.")
            self.active_discriminator_key = 'primary' 
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_actual_type = self.primary_disc_actual_type
            self.q_controller_d_active = self.q_controller_d_primary

        if self.am_main_process:
            self.logger.info(f"Active Discriminator successfully set to: '{self.active_discriminator_key}' (Actual Type: '{self.active_disc_actual_type}')")

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()

    def _compute_recon_loss(self, recon_norm_dcts: torch.Tensor, target_norm_dcts: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_norm_dcts, target_norm_dcts)

    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, mel_spectrograms_to_log: Optional[torch.Tensor],
                              num_sequences_to_log_max: int = 2):
        if not (self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb is not None and wandb.run):
            return
        if mel_spectrograms_to_log is None or mel_spectrograms_to_log.numel() == 0:
            self.logger.debug(f"Skipping WandB image log for {tag_prefix} due to None or empty data.")
            return

        B_log, C_log, H_log, W_log = mel_spectrograms_to_log.shape
        if C_log != 1: 
            self.logger.warning(f"Mel spectrograms for logging have {C_log} channels, expected 1. Taking first channel.")
            mel_spectrograms_to_log = mel_spectrograms_to_log[:,0:1,:,:]

        num_to_actually_log = min(B_log, num_sequences_to_log_max)
        wandb_images_for_log = []
        
        for b_idx in range(num_to_actually_log):
            mel_tensor = mel_spectrograms_to_log[b_idx, 0, ...].cpu().float()
            fig_iter, ax_iter = None, None
            try:
                if MATPLOTLIB_AVAILABLE and plt is not None and librosa is not None and librosa.display is not None:
                    aspect_ratio = W_log / H_log if H_log > 0 and W_log > 0 else 1.0
                    fig_width = max(5, min(15, int(5 * aspect_ratio)))
                    fig_height = max(4, min(10, int(fig_width / aspect_ratio if aspect_ratio > 0 else fig_width)))
                    fig_iter, ax_iter = plt.subplots(1, 1, figsize=(fig_width, fig_height))
                    
                    fmax_val = self.args.fmax if self.args.fmax is not None and self.args.fmax > self.args.fmin else self.args.sample_rate / 2.0
                    img = librosa.display.specshow(mel_tensor.numpy(), ax=ax_iter,
                                             sr=self.args.sample_rate, hop_length=self.args.hop_length,
                                             x_axis='time', y_axis='mel', fmin=self.args.fmin, fmax=fmax_val, cmap='magma')
                    fig_iter.colorbar(img, ax=ax_iter, format='%+.2f (norm val)')
                    ax_iter.set_title(f"{tag_prefix} S{b_idx} Ep{self.current_epoch+1} GStep{self.global_step}")
                    wandb_images_for_log.append(wandb.Image(fig_iter))
                else: 
                    raise RuntimeError("Matplotlib/Librosa display unavailable for logging.") 
            except Exception as e_disp:
                self.logger.debug(f"Librosa display failed for {tag_prefix} S{b_idx}: {e_disp}. Falling back to raw image.")
                img_0_1 = (mel_tensor.clamp(-1,1) + 1) / 2.0 
                caption = f"{tag_prefix} S{b_idx} Ep{self.current_epoch+1} GStep{self.global_step} (raw_fallback)"
                wandb_images_for_log.append(wandb.Image(img_0_1, caption=caption))
            finally:
                if fig_iter is not None and plt is not None: plt.close(fig_iter)

        if wandb_images_for_log:
            try:
                wandb.log({f"samples_mel/{tag_prefix}": wandb_images_for_log}, step=self.global_step)
            except Exception as e_wandb_log:
                self.logger.error(f"Failed to log images to WandB for {tag_prefix}: {e_wandb_log}", exc_info=True)

    def _train_discriminator_step(self, real_mel_spectrograms: torch.Tensor,
                                  m_ref: "WuBuSpecTransNet") -> Dict[str, torch.Tensor]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        B = real_mel_spectrograms.shape[0]; device = real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters())).dtype

        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model).squeeze(-1) 
        fake_labels = torch.zeros(B, 1, device=device, dtype=dtype_model).squeeze(-1)
        losses_d_micro: Dict[str, torch.Tensor] = {}

        for p in d_ref_active.parameters(): p.requires_grad = True
        for p in m_ref.parameters(): p.requires_grad = False
        if self.optimizer_disc_active: self.optimizer_disc_active.zero_grad(set_to_none=True)

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            if d_ref_active.input_type == "mel":
                real_input_for_d = real_mel_spectrograms.to(device, dtype=dtype_model)
                real_logits = d_ref_active(real_input_for_d)
            elif d_ref_active.input_type == "dct":
                with torch.no_grad():
                    _, _, real_norm_dcts_target, _ = m_ref.encode(real_mel_spectrograms.to(device, dtype=dtype_model))
                real_input_for_d = real_norm_dcts_target.to(device, dtype=dtype_model)
                real_logits = d_ref_active(real_input_for_d)
            else: raise ValueError(f"Unsupported D input type: {d_ref_active.input_type}")
            loss_d_real = self.adversarial_loss(real_logits.squeeze(), real_labels)

            with torch.no_grad():
                fake_norm_dct_coeffs, _, _, gaad_bboxes_for_assembly, _ = m_ref(real_mel_spectrograms.to(device, dtype=dtype_model))

            if d_ref_active.input_type == "mel":
                unnorm_fake_dcts = AudioSpecGenerator._unnormalize_dct(fake_norm_dct_coeffs.detach(), self.args)
                fake_mel_input_for_d = d_ref_active._assemble_mel_from_dct_regions(unnorm_fake_dcts, gaad_bboxes_for_assembly.detach(), real_mel_spectrograms.shape)
                fake_logits = d_ref_active(fake_mel_input_for_d.detach())
            elif d_ref_active.input_type == "dct":
                fake_input_for_d = fake_norm_dct_coeffs.to(device, dtype=dtype_model).detach()
                fake_logits = d_ref_active(fake_input_for_d)
            else: raise ValueError(f"Unsupported D input type (fake): {d_ref_active.input_type}")
            loss_d_fake = self.adversarial_loss(fake_logits.squeeze(), fake_labels)
            
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total_scaled_for_accum_micro = loss_d_total_micro / self.grad_accum_steps

        self.scaler_disc.scale(loss_d_total_scaled_for_accum_micro).backward()

        losses_d_micro['loss_d_real_micro'] = loss_d_real.detach()
        losses_d_micro['loss_d_fake_micro'] = loss_d_fake.detach()
        losses_d_micro['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d_micro

    def _train_generator_step(self, real_mel_spectrograms: torch.Tensor,
                              m_ref: "WuBuSpecTransNet") -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        B = real_mel_spectrograms.shape[0]; device = real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters())).dtype
        real_labels_for_g = torch.ones(B, 1, device=device, dtype=dtype_model).squeeze(-1)
        losses_g_micro: Dict[str, torch.Tensor] = {}
        recon_mel_for_log: Optional[torch.Tensor] = None

        for p in d_ref_active.parameters(): p.requires_grad = False
        for p in m_ref.parameters(): p.requires_grad = True
        if self.optimizer_enc_gen: self.optimizer_enc_gen.zero_grad(set_to_none=True)

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            recon_norm_dct_coeffs, mu, logvar, gaad_bboxes_from_enc, target_norm_dct_coeffs = \
                m_ref(real_mel_spectrograms.to(device,dtype=dtype_model))

            loss_recon_raw = self._compute_recon_loss(recon_norm_dct_coeffs, target_norm_dct_coeffs)
            loss_kl_raw = self._compute_kl_loss(mu, logvar)
            
            loss_recon_eff = self.lambda_recon * self.heuristic_override_lambda_recon_factor * loss_recon_raw
            loss_kl_eff = self.lambda_kl * self.heuristic_override_lambda_kl_factor * loss_kl_raw
            
            fake_logits_for_g: torch.Tensor
            features_for_g_feat_match: Optional[torch.Tensor] = None

            if d_ref_active.input_type == "mel":
                unnorm_recon_dcts_for_adv = AudioSpecGenerator._unnormalize_dct(recon_norm_dct_coeffs, self.args)
                recon_mel_for_adv = d_ref_active._assemble_mel_from_dct_regions(unnorm_recon_dcts_for_adv, gaad_bboxes_from_enc, real_mel_spectrograms.shape)
                if self.heuristic_vae_feature_match_active:
                    output_d = d_ref_active(recon_mel_for_adv, return_features=True)
                    fake_logits_for_g, features_for_g_feat_match = output_d if isinstance(output_d, tuple) else (output_d, None)
                else:
                    fake_logits_for_g = d_ref_active(recon_mel_for_adv, return_features=False)
                
                if self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and self.global_step > 0 and \
                   ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0): 
                    recon_mel_for_log = recon_mel_for_adv.detach().clone()

            elif d_ref_active.input_type == "dct":
                if self.heuristic_vae_feature_match_active:
                    output_d = d_ref_active(recon_norm_dct_coeffs, return_features=True)
                    fake_logits_for_g, features_for_g_feat_match = output_d if isinstance(output_d, tuple) else (output_d, None)
                else:
                    fake_logits_for_g = d_ref_active(recon_norm_dct_coeffs, return_features=False)
            else: raise ValueError(f"Unsupported D input type for G: {d_ref_active.input_type}")

            loss_g_adv_raw = self.adversarial_loss(fake_logits_for_g.squeeze(), real_labels_for_g)
            loss_g_adv_eff = self.lambda_gan * self.heuristic_override_lambda_gan_factor * loss_g_adv_raw
            
            loss_g_total_micro = loss_recon_eff + loss_kl_eff + loss_g_adv_eff

            if self.heuristic_vae_feature_match_active and features_for_g_feat_match is not None:
                with torch.no_grad():
                    target_d_output_for_feat_match: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                    if d_ref_active.input_type == "mel":
                        target_d_output_for_feat_match = d_ref_active(real_mel_spectrograms.to(device, dtype=dtype_model), return_features=True)
                    else: 
                        target_d_output_for_feat_match = d_ref_active(target_norm_dct_coeffs.to(device, dtype=dtype_model).detach(), return_features=True)
                    
                    target_features_d = target_d_output_for_feat_match[1] if isinstance(target_d_output_for_feat_match, tuple) else None
                
                if target_features_d is not None:
                    if features_for_g_feat_match.ndim > 2 and target_features_d.ndim > 2 : 
                        features_for_g_feat_match_flat = torch.mean(features_for_g_feat_match, dim=list(range(2, features_for_g_feat_match.ndim)))
                        target_features_d_flat = torch.mean(target_features_d, dim=list(range(2, target_features_d.ndim)))
                    else: 
                        features_for_g_feat_match_flat = features_for_g_feat_match
                        target_features_d_flat = target_features_d
                    
                    if features_for_g_feat_match_flat.shape == target_features_d_flat.shape:
                        loss_g_feat_match = F.mse_loss(features_for_g_feat_match_flat, target_features_d_flat.detach())
                        loss_g_total_micro += self.lambda_feat_match_heuristic * loss_g_feat_match
                        losses_g_micro['loss_g_feat_match_micro'] = loss_g_feat_match.detach()
                    else: self.logger.warning(f"Feat Match shapes mismatch: G_feat_flat {features_for_g_feat_match_flat.shape}, D_feat_flat {target_features_d_flat.shape}")

            if self.heuristic_penalize_g_easy_win_active:
                if loss_g_adv_raw.item() < self.G_WINNING_THRESH and loss_recon_raw.item() > self.TARGET_GOOD_RECON_THRESH_HEURISTIC:
                    denominator_penalty = loss_g_adv_raw.item() + getattr(self.args, 'g_easy_win_penalty_eps_denom', 1e-4)
                    penalty_g_easy_win_val: float
                    if denominator_penalty < EPS : 
                        penalty_g_easy_win_val = self.lambda_g_easy_win_penalty_heuristic * 10.0 
                        if self.am_main_process and self.global_step > 0 and self.global_step % (self.args.log_interval * 20) == 0: # Log less often for this specific warning
                            self.logger.warning(f"G_Easy_Win_Penalty: Denom for penalty factor was < EPS. Capping penalty value to {penalty_g_easy_win_val:.2f}")
                    else:
                        penalty_factor = (loss_recon_raw.item() - self.TARGET_GOOD_RECON_THRESH_HEURISTIC) / denominator_penalty
                        penalty_g_easy_win_val = penalty_factor * self.lambda_g_easy_win_penalty_heuristic
                    
                    max_penalty_val_abs = getattr(self.args, 'max_g_easy_win_penalty_abs', 20.0)
                    penalty_g_easy_win_clamped = torch.clamp(torch.tensor(penalty_g_easy_win_val, device=device, dtype=dtype_model), 
                                                             0, max_penalty_val_abs) 

                    loss_g_total_micro += penalty_g_easy_win_clamped
                    losses_g_micro['loss_g_easy_win_penalty_micro'] = penalty_g_easy_win_clamped.detach()
                    if self.am_main_process and self.global_step > 0 and self.global_step % (self.args.log_interval * 5) == 0: 
                         self.logger.info(f"HEURISTIC APPLIED: G_Easy_Win_Penalty={penalty_g_easy_win_clamped.item():.2f} (RawAdv={loss_g_adv_raw.item():.3f}, RawRec={loss_recon_raw.item():.3f})")

            loss_g_total_scaled_for_accum_micro = loss_g_total_micro / self.grad_accum_steps

        self.scaler_enc_gen.scale(loss_g_total_scaled_for_accum_micro).backward()

        losses_g_micro['loss_recon_micro'] = loss_recon_raw.detach()
        losses_g_micro['loss_kl_micro'] = loss_kl_raw.detach()
        losses_g_micro['loss_g_adv_micro'] = loss_g_adv_raw.detach()
        losses_g_micro['loss_g_total_micro'] = loss_g_total_micro.detach() 
        return losses_g_micro, recon_mel_for_log

    def _get_q_controller_data_for_heuristics(self) -> Dict[str, Any]:
        q_data: Dict[str, Any] = {'gen': {'is_valid': False}, 'active_d': {'is_valid': False}, 'lkl': {'is_valid': False}}
        controllers_map = {
            'gen': self.q_controller_gen, 'active_d': self.q_controller_d_active, 'lkl': self.lambda_kl_q_controller
        }
        hist_names_g_d = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']
        hist_names_lkl = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']

        for key, controller in controllers_map.items():
            if controller:
                q_data[key]['is_valid'] = True
                q_data[key]['epsilon'] = controller.epsilon
                q_data[key]['on_probation'] = getattr(controller, 'on_probation', False) or getattr(controller, 'lkl_on_probation', False)
                q_data[key]['reward_median_short_term'] = np.median(list(controller.reward_hist)[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:]) \
                                                          if len(controller.reward_hist) >= self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS else \
                                                          (np.median(list(controller.reward_hist)) if controller.reward_hist else 0.0)
                
                current_hist_names = hist_names_g_d if key in ['gen', 'active_d'] else hist_names_lkl
                hist_prefix = "loss_" if key in ['gen', 'active_d'] else "interval_"

                for lname in current_hist_names:
                    hist_attr = f"{hist_prefix}{lname}_hist"
                    if hasattr(controller, hist_attr):
                        hist_deque = getattr(controller, hist_attr)
                        val_for_trend = list(hist_deque)[-1] if hist_deque else None
                        median_val = None
                        if len(hist_deque) >= self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:
                            median_val = np.median(list(hist_deque)[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:])
                        elif hist_deque: median_val = np.median(list(hist_deque))
                        
                        q_data[key][f"{lname}_median_short_term"] = median_val
                        q_data[key][f"{lname}_trend_short_term"] = controller._get_trend_bin(hist_deque, val_for_trend) if val_for_trend is not None else 2
        
        if q_data['gen']['is_valid'] and q_data['gen'].get('g_recon_median_short_term') is not None:
            self.q_data_derived_g_recon_hist.append(q_data['gen']['g_recon_median_short_term'])
            self.avg_g_recon_hist_for_stagnation.append(q_data['gen']['g_recon_median_short_term']) 

            if len(self.avg_g_recon_hist_for_stagnation) >= max(2, self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS // 2): 
                current_recon_median = self.avg_g_recon_hist_for_stagnation[-1]
                past_relevant_history = list(self.avg_g_recon_hist_for_stagnation)[:-1]
                if len(past_relevant_history) > 1:
                     past_recon_avg = np.mean(past_relevant_history)
                elif past_relevant_history: 
                     past_recon_avg = past_relevant_history[0]
                else: 
                     past_recon_avg = current_recon_median 
                
                self.rec_dct_stagnant = (past_recon_avg - current_recon_median) < (past_recon_avg * self.RECON_STAGNATION_IMPROVEMENT_THRESH_REL)
            else: self.rec_dct_stagnant = False
        return q_data

    def _evaluate_training_state_and_apply_heuristics(self):
        if not self.am_main_process:
            return

        if not self.enable_heuristic_interventions:
            if hasattr(self, 'global_step') and hasattr(self, 'heuristic_check_interval') and \
               self.heuristic_check_interval > 0 and self.global_step > 0 and \
               self.global_step % (self.heuristic_check_interval * 10) == 0:
                 self.logger.info(f"GStep {self.global_step}: Heuristic interventions are globally DISABLED (self.enable_heuristic_interventions is False). Skipping evaluation.")
            
            self.heuristic_vae_feature_match_active = False
            self.heuristic_penalize_g_easy_win_active = False
            self.heuristic_boost_active_d_lr_active = False
            self.heuristic_force_d_q_explore_active = False
            self.heuristic_override_lambda_recon_factor = 1.0
            self.heuristic_override_lambda_kl_factor = 1.0
            self.heuristic_override_lambda_gan_factor = 1.0
            return
        
        gstep_log_val = self.global_step if hasattr(self, 'global_step') else "N/A"
        if self.global_step == 0 or (self.heuristic_check_interval > 0 and self.global_step % (self.heuristic_check_interval * 5) == 0):
            self.logger.info(f"GStep {gstep_log_val}: Evaluating training state for heuristics (self.enable_heuristic_interventions is True).")
            # Add extended debug logging less frequently
            q_data_dbg = self._get_q_controller_data_for_heuristics()
            gen_q_dbg, active_d_q_dbg = q_data_dbg.get('gen', {}), q_data_dbg.get('active_d', {})
            g_adv_median_dbg = gen_q_dbg.get('g_adv_median_short_term', float('nan'))
            d_total_median_dbg = active_d_q_dbg.get('d_total_median_short_term', float('nan'))
            # ... (add other dbg logs as in previous suggestion)
            self.logger.info(f"  DBG Heuristic Inputs - G_Adv_Med: {g_adv_median_dbg:.3f}, D_Total_Med: {d_total_median_dbg:.3f}, ReconStagnant: {self.rec_dct_stagnant}")


        q_data = self._get_q_controller_data_for_heuristics() # Call again to ensure freshest data for actual logic
        gen_q, active_d_q = q_data.get('gen', {}), q_data.get('active_d', {})
        log_msgs = []

        current_lambda_recon_factor = 1.0 
        current_lambda_kl_factor = 1.0
        current_lambda_gan_factor = self.heuristic_override_lambda_gan_factor 

        current_boost_active_d_lr = False
        current_force_d_q_explore = False
        current_penalize_g_easy_win = False 
        current_vae_feature_match = False 

        g_adv_median = gen_q.get('g_adv_median_short_term', 0.7)
        d_total_median = active_d_q.get('d_total_median_short_term', 0.7)
        d_q_reward_median = active_d_q.get('reward_median_short_term', 0.0)
        
        is_g_dominating_very_much = g_adv_median < self.G_VERY_MUCH_WINNING_THRESH
        is_d_very_weak = d_total_median > self.D_VERY_WEAK_THRESH
        is_d_q_learner_stagnant = d_q_reward_median < self.Q_REWARD_STAGNATION_THRESH
        is_d_strong = d_total_median < self.D_STRONG_THRESH
        is_g_stalled_adv = g_adv_median > self.G_STALLED_THRESH

        switched_d_this_cycle = False
        if self.enable_heuristic_disc_switching:
            switched_d_this_cycle = self._check_and_perform_disc_switch(
                is_g_dominating_adv=is_g_dominating_very_much, is_d_weak_overall=is_d_very_weak,
                is_d_struggling_q=is_d_q_learner_stagnant, is_d_strong_overall=is_d_strong,
                is_g_stalled_adv=is_g_stalled_adv,
                current_g_kl_median = gen_q.get('g_kl_median_short_term', 0.0),
                log_msgs_ref = log_msgs
            )
        
        if switched_d_this_cycle:
            self.consecutive_heuristic_trigger_counts = defaultdict(int) 
            current_lambda_gan_factor = 1.0 
        else: 
            condition_gan_rebalance = is_g_dominating_very_much and (is_d_very_weak or is_d_q_learner_stagnant) and self.rec_dct_stagnant
            if condition_gan_rebalance:
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] += 1
                if self.consecutive_heuristic_trigger_counts['gan_rebalance'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_penalize_g_easy_win = True
                    current_lambda_recon_factor = self.args.heuristic_recon_boost_factor
                    current_lambda_gan_factor = min(current_lambda_gan_factor * 1.05, getattr(self.args, 'heuristic_max_lambda_gan_factor', 1.3))
                    if is_d_q_learner_stagnant:
                        current_boost_active_d_lr = True
                        current_force_d_q_explore = True
                    log_msgs.append(f"HEURISTIC: GAN REBALANCE ACTIVE. PenalizeG:{current_penalize_g_easy_win}, LRecF:{current_lambda_recon_factor:.2f}, LGanF:{current_lambda_gan_factor:.2f}, D_LR_Boost:{current_boost_active_d_lr}, D_Q_Explore:{current_force_d_q_explore}")
            else: 
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] = 0
                if current_lambda_gan_factor > 1.0: # If it was boosted, decay it
                     current_lambda_gan_factor = max(current_lambda_gan_factor * 0.98, 1.0) 

            condition_vae_feat_match = (not is_g_dominating_very_much and not is_d_very_weak and 
                                        (is_d_strong or not is_d_q_learner_stagnant) and self.rec_dct_stagnant)
            if condition_vae_feat_match:
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] += 1
                if self.consecutive_heuristic_trigger_counts['vae_feat_match'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_vae_feature_match = True
                    if self.lambda_kl * self.heuristic_override_lambda_kl_factor < 1e-4 : current_lambda_kl_factor = 1.5 # Using old factor for this check
                    current_lambda_gan_factor = max(current_lambda_gan_factor * 0.95, getattr(self.args, 'heuristic_min_lambda_gan_factor', 0.7))
                    log_msgs.append(f"HEURISTIC: VAE FEATURE MATCH ACTIVE. LKLF:{current_lambda_kl_factor:.2f}, LGanF:{current_lambda_gan_factor:.2f}")
            else: 
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] = 0
                if current_lambda_gan_factor < 1.0: # If it was reduced, decay it back towards 1.0
                     current_lambda_gan_factor = min(current_lambda_gan_factor * 1.02, 1.0) 

        # Update persistent heuristic states based on this evaluation
        self.heuristic_penalize_g_easy_win_active = current_penalize_g_easy_win
        self.heuristic_override_lambda_recon_factor = current_lambda_recon_factor
        self.heuristic_boost_active_d_lr_active = current_boost_active_d_lr
        self.heuristic_vae_feature_match_active = current_vae_feature_match
        self.heuristic_override_lambda_kl_factor = current_lambda_kl_factor
        self.heuristic_override_lambda_gan_factor = current_lambda_gan_factor 
        
        if current_force_d_q_explore and self.q_controller_d_active: 
            self.q_controller_d_active.force_exploration_boost(
                duration_steps=self.heuristic_d_q_explore_duration,
                boost_epsilon_to=self.heuristic_d_q_explore_boost_epsilon
            )
            log_msgs.append(f"HEURISTIC: Active D Q-Controller exploration boosted for {self.heuristic_d_q_explore_duration} steps.")

        if log_msgs and self.am_main_process: 
            for msg in log_msgs: self.logger.info(msg)

    def _check_and_perform_disc_switch(self, 
                                         is_g_dominating_adv: bool, is_d_weak_overall: bool, is_d_struggling_q: bool,
                                         is_d_strong_overall: bool, is_g_stalled_adv: bool,
                                         current_g_kl_median: float,
                                         log_msgs_ref: List[str]) -> bool:
        if not self.enable_heuristic_disc_switching or self.steps_since_last_d_switch < self.disc_switch_min_steps_between:
            return False

        switched_this_check = False
        
        effective_kl_val = current_g_kl_median * self.lambda_kl * self.heuristic_override_lambda_kl_factor 
        condition_A = (is_d_strong_overall and is_g_stalled_adv and 
                       (effective_kl_val > self.KL_HIGH_THRESH * 0.1 or self.rec_dct_stagnant))

        if condition_A:
            self.consecutive_trigger_primary_to_alt_count += 1
            if self.consecutive_trigger_primary_to_alt_count >= self.disc_switch_problem_state_count_thresh:
                if self.active_discriminator_key == 'primary':
                    target_key = 'alternative'
                    log_msgs_ref.append(f"DISC_SWITCH: Trigger A! D_strong & G_stalled & (HighEffKL or ReconStagnant). Switching Primary -> Alternative.")
                    self.active_discriminator_key = target_key
                    self._update_active_discriminator_pointers()
                    if self.q_controller_d_active: self.q_controller_d_active.reset_q_learning_state(True, True, f"DSwitch P->A by Heuristic A", True)
                    self.steps_since_last_d_switch = 0; self.consecutive_trigger_primary_to_alt_count = 0; self.consecutive_trigger_alt_to_primary_count = 0
                    switched_this_check = True
                else: # Already alternative, reset count for this condition but don't switch back
                    self.consecutive_trigger_primary_to_alt_count = 0 
        else: self.consecutive_trigger_primary_to_alt_count = 0

        condition_B = (is_g_dominating_adv and is_d_weak_overall and
                       (not self.rec_dct_stagnant or is_d_struggling_q) ) 

        if not switched_this_check and condition_B: # Only consider if not already switched by A
            self.consecutive_trigger_alt_to_primary_count += 1
            if self.consecutive_trigger_alt_to_primary_count >= self.disc_switch_problem_state_count_thresh:
                if self.active_discriminator_key == 'alternative':
                    target_key = 'primary'
                    log_msgs_ref.append(f"DISC_SWITCH: Trigger B! G_dominating & D_weak & (NotReconStagnant or D_Q_Struggling). Switching Alternative -> Primary.")
                    self.active_discriminator_key = target_key
                    self._update_active_discriminator_pointers()
                    if self.q_controller_d_active: self.q_controller_d_active.reset_q_learning_state(True, True, f"DSwitch A->P by Heuristic B", True)
                    self.steps_since_last_d_switch = 0; self.consecutive_trigger_alt_to_primary_count = 0; self.consecutive_trigger_primary_to_alt_count = 0
                    switched_this_check = True
                else: # Already primary, reset count for this condition
                    self.consecutive_trigger_alt_to_primary_count = 0 
        elif not switched_this_check: # If not switched by A, and B not met, reset B's counter
             self.consecutive_trigger_alt_to_primary_count = 0
        
        if switched_this_check:
            self.rec_dct_stagnant = False 
            self.avg_g_recon_hist_for_stagnation.clear() 
            self.q_data_derived_g_recon_hist.clear() # Clear this too
            log_msgs_ref.append(f"  --> Post D-Switch: Heuristic flags reset. New active D: '{self.active_discriminator_key}' (type: {self.active_disc_actual_type}).")
        return switched_this_check

    def train(self, start_epoch: int = 0, initial_global_step: int = 0):
        self.global_step = initial_global_step
        self.current_epoch = start_epoch
        
        if self.am_main_process:
            self.logger.info(f"Starting training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch}, InitialGStep: {initial_global_step}")
            self.logger.info(f"Initial Active D: {self.active_discriminator_key} (Type: {self.active_disc_actual_type})")

        if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and \
           self.args.num_val_samples_to_log > 0 and self.fixed_noise_for_sampling is None:
            m_ref_temp = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
            default_dtype = torch.float32
            try: default_dtype = next(iter(m_ref_temp.parameters())).dtype
            except StopIteration: self.logger.warning("Model has no parameters; using float32 for fixed noise.")
            if self.args.latent_dim > 0:
                self.fixed_noise_for_sampling = torch.randn(self.args.num_val_samples_to_log, self.args.latent_dim, device=self.device, dtype=default_dtype)
                self.logger.info(f"Created fixed noise tensor: {self.fixed_noise_for_sampling.shape} on {self.device}")
            else: self.logger.warning("latent_dim <= 0, cannot create fixed_noise_for_sampling.")

        log_interval_accum_losses: Dict[str, float] = defaultdict(float)
        log_interval_items_processed = 0
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        
        all_q_controllers_to_sync_lkl = [self.q_controller_gen, self.q_controller_d_primary, self.q_controller_d_alt, self.lambda_kl_q_controller]
        for q_ctrl in all_q_controllers_to_sync_lkl:
            if q_ctrl and hasattr(q_ctrl, 'set_current_lambda_kl'):
                q_ctrl.set_current_lambda_kl(self.lambda_kl)

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process:
                self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting (L_KL: {self.lambda_kl:.3e}*KLF:{self.heuristic_override_lambda_kl_factor:.2f}, LRecF: {self.heuristic_override_lambda_recon_factor:.2f}, LGanF: {self.heuristic_override_lambda_gan_factor:.2f}, ActD: {self.active_discriminator_key} [{self.active_disc_actual_type}]).")
            if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch) 

            m_ref.train(); self.active_discriminator.train()
            
            num_batches_epoch = len(self.train_loader)
            prog_bar = tqdm(self.train_loader, desc=f"E{epoch+1}", disable=not self.am_main_process, dynamic_ncols=True, total=num_batches_epoch)
            
            accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
            accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0

            for batch_idx, batch_mel_segments in enumerate(prog_bar):
                batch_mel_segments = batch_mel_segments.to(self.device)
                batch_size_micro = batch_mel_segments.size(0)
                self.steps_since_last_d_switch +=1 

                losses_d_micro = self._train_discriminator_step(batch_mel_segments, m_ref)
                for k, v_tensor in losses_d_micro.items():
                    if torch.isfinite(v_tensor): 
                        val = v_tensor.item()
                        if np.isfinite(val): 
                            accum_key = k.replace('_micro', '_agg') 
                            log_interval_accum_losses[accum_key] += val * batch_size_micro
                            if k == 'loss_d_total_micro': accum_d_total_q += val; self.interval_metrics_accum['d_total'] += val
                            elif k == 'loss_d_real_micro': accum_d_real_q += val
                            elif k == 'loss_d_fake_micro': accum_d_fake_q += val
                
                losses_g_micro, recon_mel_for_logging = self._train_generator_step(batch_mel_segments, m_ref)
                for k, v_tensor in losses_g_micro.items():
                    if torch.isfinite(v_tensor): 
                        val = v_tensor.item()
                        if np.isfinite(val): 
                            accum_key = k.replace('_micro', '_agg')
                            log_interval_accum_losses[accum_key] += val * batch_size_micro
                            if k == 'loss_g_total_micro': accum_g_total_q += val
                            elif k == 'loss_recon_micro': accum_g_recon_q += val; self.interval_metrics_accum['recon_dct'] += val
                            elif k == 'loss_kl_micro': accum_g_kl_q += val; self.interval_metrics_accum['kl_div'] += val
                            elif k == 'loss_g_adv_micro': accum_g_adv_q += val
                            # Log heuristic penalty contributions if they exist
                            elif k == 'loss_g_feat_match_micro': log_interval_accum_losses['loss_g_feat_match_eff_contrib_agg'] += (self.lambda_feat_match_heuristic * val * batch_size_micro)
                            elif k == 'loss_g_easy_win_penalty_micro': log_interval_accum_losses['loss_g_easy_win_penalty_eff_contrib_agg'] += (val * batch_size_micro) # Value is already the scaled penalty

                log_interval_items_processed += batch_size_micro
                self.interval_steps_count += 1

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.optimizer_disc_active:
                        if hasattr(self.optimizer_disc_active, 'grad_stats'):
                            num_disc_params = sum(p.numel() for grp in self.optimizer_disc_active.param_groups for p in grp['params'] if p.requires_grad)
                            self.optimizer_disc_active.grad_stats.finalize_step_stats(num_disc_params)
                        if self.args.global_max_grad_norm > 0:
                            self.scaler_disc.unscale_(self.optimizer_disc_active)
                            d_to_clip = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                            torch.nn.utils.clip_grad_norm_(d_to_clip.parameters(), self.args.global_max_grad_norm)
                        self.scaler_disc.step(self.optimizer_disc_active)
                        self.scaler_disc.update()
                        self.optimizer_disc_active.zero_grad(set_to_none=True)

                    if self.optimizer_enc_gen:
                        if hasattr(self.optimizer_enc_gen, 'grad_stats'):
                            num_gen_params = sum(p.numel() for grp in self.optimizer_enc_gen.param_groups for p in grp['params'] if p.requires_grad)
                            self.optimizer_enc_gen.grad_stats.finalize_step_stats(num_gen_params)
                        if self.args.global_max_grad_norm > 0:
                            self.scaler_enc_gen.unscale_(self.optimizer_enc_gen)
                            torch.nn.utils.clip_grad_norm_(m_ref.parameters(), self.args.global_max_grad_norm)
                        self.scaler_enc_gen.step(self.optimizer_enc_gen)
                        self.scaler_enc_gen.update()
                        self.optimizer_enc_gen.zero_grad(set_to_none=True)
                    
                    self.global_step += 1

                    avg_losses_for_q = {
                        'loss_g_total': accum_g_total_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_g_recon': accum_g_recon_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_g_kl': accum_g_kl_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_g_adv': accum_g_adv_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_d_total': accum_d_total_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_d_real': accum_d_real_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_d_fake': accum_d_fake_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0
                    }
                    for k_q_check, v_q_check in avg_losses_for_q.items(): 
                        if not np.isfinite(v_q_check): 
                            avg_losses_for_q[k_q_check] = 1.0 
                            if self.am_main_process: self.logger.warning(f"Q-Ctrl Input: Non-finite avg loss for {k_q_check} ({v_q_check}), replaced with 1.0.")


                    if self.q_controller_d_active and hasattr(self.optimizer_disc_active, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_disc_active.q_controller_update_and_set_hyperparams(avg_losses_for_q, self.lambda_kl * self.heuristic_override_lambda_kl_factor)
                        if self.heuristic_boost_active_d_lr_active and self.optimizer_disc_active:
                            for group in self.optimizer_disc_active.param_groups:
                                boosted_lr = group['lr'] * self.heuristic_active_d_lr_boost_factor
                                group['lr'] = float(np.clip(boosted_lr, 1e-8, 1.0))
                            if self.am_main_process: self.logger.info(f"HEURISTIC: D LR boosted to {self.optimizer_disc_active.param_groups[0]['lr']:.2e}")

                    if self.q_controller_gen and hasattr(self.optimizer_enc_gen, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_enc_gen.q_controller_update_and_set_hyperparams(avg_losses_for_q, self.lambda_kl * self.heuristic_override_lambda_kl_factor)
                    
                    accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
                    accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0

                    if self.global_step > 0 and self.global_step % self.heuristic_check_interval == 0:
                        self._evaluate_training_state_and_apply_heuristics() 

                    if self.lambda_kl_q_controller is not None and self.lambda_kl_update_interval > 0 and \
                       self.global_step > 0 and self.global_step % self.lambda_kl_update_interval == 0 and \
                       self.interval_steps_count > 0:
                        current_interval_metrics: Dict[str, Union[float, None]] = {
                            'avg_recon': self.interval_metrics_accum['recon_dct'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_kl_div': self.interval_metrics_accum['kl_div'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_d_total': self.interval_metrics_accum['d_total'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'val_metric': self.last_val_metrics.get(self.args.val_primary_metric),
                            'current_lambda_kl_val': self.lambda_kl 
                        }
                        if self.am_main_process: 
                            log_metrics_str = {k_interval: (f'{v_interval:.4f}' if isinstance(v_interval, float) and v_interval is not None and np.isfinite(v_interval) else str(v_interval)) for k_interval,v_interval in current_interval_metrics.items()}
                            lkl_q_log_str = (f"GStep {self.global_step}: Lambda_KL Q-Ctrl block. Current Base LKL: {self.lambda_kl:.4e}. "
                                             f"Interval Metrics: {log_metrics_str}. Prev LKL_QState: {self.lambda_kl_q_controller.prev_lambda_kl_state}. "
                                             f"Prev LKL_Action: {self.lambda_kl_q_controller.prev_lambda_kl_action}")
                            prog_bar.write(lkl_q_log_str)

                        q_state_lambda_kl = self.lambda_kl_q_controller.get_lambda_kl_state(current_interval_metrics)
                        if self.am_main_process and q_state_lambda_kl is not None:
                            q_vals_state_lkl = self.lambda_kl_q_controller.q_table.get(q_state_lambda_kl, {}).get('lambda_kl_scale')
                            prog_bar.write(f"  New LKL_QState: {q_state_lambda_kl}. Q-vals(LKL_scale): {q_vals_state_lkl}")

                        if self.lambda_kl_q_controller.prev_lambda_kl_state is not None and \
                           self.lambda_kl_q_controller.prev_lambda_kl_action is not None and \
                           q_state_lambda_kl is not None and self.prev_interval_metrics_for_lambda_kl_reward is not None:
                            reward_for_lambda_kl = self.lambda_kl_q_controller.compute_lambda_kl_reward(current_interval_metrics, self.prev_interval_metrics_for_lambda_kl_reward)
                            if self.am_main_process: prog_bar.write(f"  Lambda_KL Q-Ctrl reward computed: {reward_for_lambda_kl:.3f}")
                            self.lambda_kl_q_controller.update_q_values(self.lambda_kl_q_controller.prev_lambda_kl_state, self.lambda_kl_q_controller.prev_lambda_kl_action, reward_for_lambda_kl, q_state_lambda_kl, mode='lambda_kl')
                        elif q_state_lambda_kl is not None and hasattr(self.lambda_kl_q_controller, 'set_initial_lambda_kl_metrics'):
                            self.lambda_kl_q_controller.set_initial_lambda_kl_metrics(current_interval_metrics)
                        
                        if q_state_lambda_kl is not None: 
                            lambda_kl_action_dict = self.lambda_kl_q_controller.choose_action(q_state_lambda_kl, mode='lambda_kl')
                            chosen_scale = lambda_kl_action_dict.get('lambda_kl_scale', 1.0)
                            if self.am_main_process:
                                prog_bar.write(f"  LKL_Q-Ctrl CHOSE scale: {chosen_scale:.3f} (Eps: {self.lambda_kl_q_controller.epsilon:.3f}, Probation: {self.lambda_kl_q_controller.lkl_on_probation})")
                            
                            new_lambda_kl_val = self.lambda_kl * chosen_scale 
                            self.lambda_kl = float(np.clip(new_lambda_kl_val, self.min_lambda_kl_q_control, self.max_lambda_kl_q_control))
                            
                            if self.am_main_process:
                                prog_bar.write(f"  GStep {self.global_step}: LKL_Q-Ctrl updated trainer's self.lambda_kl to {self.lambda_kl:.4e}")
                            
                            self.lambda_kl_q_controller.prev_lambda_kl_state = q_state_lambda_kl
                            self.lambda_kl_q_controller.prev_lambda_kl_action = lambda_kl_action_dict 
                        
                        self.prev_interval_metrics_for_lambda_kl_reward = current_interval_metrics.copy()
                        for q_ctrl_opt in all_q_controllers_to_sync_lkl: 
                            if q_ctrl_opt and hasattr(q_ctrl_opt, 'set_current_lambda_kl'): q_ctrl_opt.set_current_lambda_kl(self.lambda_kl)
                        self.interval_metrics_accum = defaultdict(float); self.interval_steps_count = 0

                    if self.global_step > 0 and self.args.log_interval > 0 and \
                       (self.global_step % self.args.log_interval == 0) and \
                       log_interval_items_processed > 0 and self.am_main_process:
                        
                        current_log_metrics_wandb: Dict[str, Any] = {}
                        for k, v_sum in log_interval_accum_losses.items():
                            if "_eff_contrib_agg" in k: 
                                current_log_metrics_wandb[f"train/{k.replace('_eff_contrib_agg', '_eff_contrib')}"] = v_sum / log_interval_items_processed
                            elif "_agg" in k: 
                                current_log_metrics_wandb[f"train/{k.replace('_agg', '')}"] = v_sum / log_interval_items_processed
                        
                        avg_raw_recon = current_log_metrics_wandb.get('train/loss_recon', 0.0)
                        avg_raw_kl = current_log_metrics_wandb.get('train/loss_kl', 0.0)
                        avg_raw_g_adv = current_log_metrics_wandb.get('train/loss_g_adv', 0.0)

                        effective_lambda_recon_component = avg_raw_recon * self.lambda_recon * self.heuristic_override_lambda_recon_factor
                        effective_lambda_kl_component = avg_raw_kl * self.lambda_kl * self.heuristic_override_lambda_kl_factor
                        effective_lambda_gan_component = avg_raw_g_adv * self.lambda_gan * self.heuristic_override_lambda_gan_factor
                        
                        current_log_metrics_wandb["train/lambda_recon_eff_contrib"] = effective_lambda_recon_component
                        current_log_metrics_wandb["train/lambda_kl_eff_contrib"] = effective_lambda_kl_component
                        current_log_metrics_wandb["train/lambda_gan_eff_contrib"] = effective_lambda_gan_component
                        current_log_metrics_wandb["train/lambda_kl_base_from_q_lkl"] = self.lambda_kl

                        loss_g_feat_match_contrib = current_log_metrics_wandb.get('train/loss_g_feat_match_eff_contrib', 0.0)
                        loss_g_easy_win_penalty_contrib = current_log_metrics_wandb.get('train/loss_g_easy_win_penalty_eff_contrib', 0.0)
                        
                        calculated_g_total_for_log = effective_lambda_recon_component + effective_lambda_kl_component + \
                                                     effective_lambda_gan_component + loss_g_feat_match_contrib + \
                                                     loss_g_easy_win_penalty_contrib
                        current_log_metrics_wandb["train/loss_g_total_calculated_from_components"] = calculated_g_total_for_log


                        lr_g = self.optimizer_enc_gen.param_groups[0]['lr'] if self.optimizer_enc_gen else -1.0
                        lr_d_active = self.optimizer_disc_active.param_groups[0]['lr'] if self.optimizer_disc_active else -1.0
                        
                        current_log_metrics_wandb.update({
                            "train/lr_gen": lr_g, 
                            f"train/lr_disc_{self.active_discriminator_key}_{self.active_disc_actual_type}": lr_d_active, 
                            "epoch_frac": epoch + ((batch_idx + 1) / (num_batches_epoch if num_batches_epoch > 0 else 1)),
                            "global_step": self.global_step,
                            f"active_disc_is_primary_val": 1 if self.active_discriminator_key == 'primary' else 0, 
                            f"active_disc_is_mel_val": 1 if self.active_disc_actual_type == 'mel' else 0
                        })
                        
                        q_controller_info_map_log = {
                            "q_gen": self.q_controller_gen, 
                            f"q_d_{self.active_discriminator_key}": self.q_controller_d_active, 
                            "q_lkl": self.lambda_kl_q_controller
                        }
                        for prefix_log, controller_obj_log in q_controller_info_map_log.items():
                            if controller_obj_log and hasattr(controller_obj_log, 'get_info'):
                                info_dict_log = controller_obj_log.get_info()
                                for k_info_log, v_info_log in info_dict_log.items():
                                    clean_k_info_log = ''.join(c if c.isalnum() or c in ['_', '/'] else '_' for c in str(k_info_log)).lower()
                                    clean_k_info_log = clean_k_info_log.replace('lrmom','').replace('lambdakl','') 
                                    wandb_log_key = f"q_info/{prefix_log}/{clean_k_info_log}"
                                    current_log_metrics_wandb[wandb_log_key] = v_info_log
                        
                        current_log_metrics_wandb["heuristic/vae_fm_active_val"] = 1 if self.heuristic_vae_feature_match_active else 0
                        current_log_metrics_wandb["heuristic/pen_g_ez_win_val"] = 1 if self.heuristic_penalize_g_easy_win_active else 0
                        current_log_metrics_wandb["heuristic/d_lr_boost_active_val"] = 1 if self.heuristic_boost_active_d_lr_active else 0
                        current_log_metrics_wandb["heuristic/lrec_factor_val"] = self.heuristic_override_lambda_recon_factor
                        current_log_metrics_wandb["heuristic/lkl_factor_val"] = self.heuristic_override_lambda_kl_factor
                        current_log_metrics_wandb["heuristic/lgan_factor_val"] = self.heuristic_override_lambda_gan_factor 
                        
                        dt_log_val = current_log_metrics_wandb.get('train/loss_d_total',-1.0) # Renamed to avoid conflict
                        dr_log_val = current_log_metrics_wandb.get('train/loss_d_real',-1.0)
                        df_log_val = current_log_metrics_wandb.get('train/loss_d_fake',-1.0)
                        
                        qeg_eps_log = current_log_metrics_wandb.get(f'q_info/q_gen/epsilon',-1.0)
                        qad_eps_log = current_log_metrics_wandb.get(f'q_info/q_d_{self.active_discriminator_key}/epsilon',-1.0)
                        qelkl_eps_log = current_log_metrics_wandb.get(f'q_info/q_lkl/epsilon', -1.0)

                        qslg_log = HybridTrainer.get_scale_from_action_value(current_log_metrics_wandb.get(f'q_info/q_gen/last_lr_mom_action'), 'lr_scale') 
                        qsld_log = HybridTrainer.get_scale_from_action_value(current_log_metrics_wandb.get(f'q_info/q_d_{self.active_discriminator_key}/last_lr_mom_action'), 'lr_scale') 
                        qslkl_log = HybridTrainer.get_scale_from_action_value(current_log_metrics_wandb.get(f'q_info/q_lkl/last_lambda_kl_action'), 'lambda_kl_scale') 

                        active_d_short_name_log = f"{self.active_discriminator_key[0].upper()}:{self.active_disc_actual_type}"
                        
                        log_str_console_final = (
                            f"E{epoch+1} S{self.global_step} ActD:{active_d_short_name_log} | "
                            f"G_tot:{calculated_g_total_for_log:.2f}(Rec:{effective_lambda_recon_component:.2f} " 
                            f"KL:{effective_lambda_kl_component:.2f} "
                            f"Adv:{effective_lambda_gan_component:.2f}"
                            + (f" FeatM:{loss_g_feat_match_contrib:.2f}" if loss_g_feat_match_contrib != 0 else "") 
                            + (f" GPen:{loss_g_easy_win_penalty_contrib:.2f}" if loss_g_easy_win_penalty_contrib != 0 else "") 
                            + f") | D_tot:{dt_log_val:.2f}(R:{dr_log_val:.3f} F:{df_log_val:.3f}) | " # Used renamed vars
                            f"LR(G/D):{lr_g:.1e}/{lr_d_active:.1e} | "
                            f"Q_Eps(G/D/LKL):{qeg_eps_log:.2f}/{qad_eps_log:.2f}/{qelkl_eps_log:.2f} | "
                            f"Q_Scl(G/D/LKL):{qslg_log:.2f}/{qsld_log:.2f}/{qslkl_log:.2f} | "
                            f"BaseLKL:{self.lambda_kl:.2e}(x{self.heuristic_override_lambda_kl_factor:.1f})" 
                        )
                        
                        prog_bar.set_postfix_str(f"ActD:{active_d_short_name_log} G:{calculated_g_total_for_log:.2f} D:{dt_log_val:.2f} RecRaw:{avg_raw_recon:.3f} EffKL_Cont:{effective_lambda_kl_component:.1e}", refresh=True)
                        prog_bar.write(log_str_console_final)
                        
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                            wandb.log(current_log_metrics_wandb, step=self.global_step)
                        
                        log_interval_accum_losses = defaultdict(float) 
                        log_interval_items_processed = 0

                    if recon_mel_for_logging is not None and self.am_main_process and \
                       self.args.wandb_log_train_recon_interval > 0 and self.global_step > 0 and \
                       (self.global_step % self.args.wandb_log_train_recon_interval == 0): 
                        self._log_samples_to_wandb("train_recon_mel", recon_mel_for_logging, self.args.num_val_samples_to_log)
                        if self.global_step % (self.args.wandb_log_train_recon_interval * getattr(self.args, 'train_target_log_freq_multiplier', 5)) == 0 :
                           self._log_samples_to_wandb("train_target_mel", batch_mel_segments, self.args.num_val_samples_to_log)

                    if self.fixed_noise_for_sampling is not None and self.am_main_process and \
                       self.args.wandb_log_fixed_noise_samples_interval > 0 and self.global_step > 0 and \
                       (self.global_step % self.args.wandb_log_fixed_noise_samples_interval == 0):
                        m_ref.eval()
                        self.active_discriminator.eval() 
                        with torch.no_grad():
                            d_ref_sample_unwrapped = self.active_discriminator.module \
                                if self.ddp_active and hasattr(self.active_discriminator, 'module') \
                                else self.active_discriminator

                            generated_norm_dcts_fixed = m_ref.decode(self.fixed_noise_for_sampling)
                            unnorm_dcts_fixed = AudioSpecGenerator._unnormalize_dct(generated_norm_dcts_fixed, self.args)
                            
                            current_audio_config_ref = self._get_audio_config_ref()
                            current_gaad_config_ref = self._get_gaad_config_ref()
                            spec_dims_canon = (current_audio_config_ref.get("num_time_frames_for_1s_segment", 86), self.args.n_mels) 
                            
                            num_fixed_samples = self.fixed_noise_for_sampling.shape[0]
                            canonical_bboxes_list_fixed = [] 
                            for _ in range(num_fixed_samples):
                                bboxes_one_fixed = golden_subdivide_rect_fixed_n(
                                    spec_dims_canon, 
                                    current_gaad_config_ref['num_regions'], 
                                    self.device, 
                                    self.fixed_noise_for_sampling.dtype,
                                    current_gaad_config_ref.get('min_size_px', 5)
                                )
                                canonical_bboxes_list_fixed.append(bboxes_one_fixed)
                            canonical_bboxes_batch_fixed_final = torch.stack(canonical_bboxes_list_fixed) 
                            
                            target_mel_shape_fixed_final = ( 
                                num_fixed_samples, 1, self.args.n_mels,
                                current_audio_config_ref.get("num_time_frames_for_1s_segment", 86)
                            )
                            fixed_noise_mels_gen_final = d_ref_sample_unwrapped._assemble_mel_from_dct_regions( 
                                unnorm_dcts_fixed, canonical_bboxes_batch_fixed_final, target_mel_shape_fixed_final
                            )
                            self._log_samples_to_wandb("fixed_noise_generated_mel", fixed_noise_mels_gen_final, num_fixed_samples)
                        m_ref.train()
                        self.active_discriminator.train() 

                    if self.args.save_interval > 0 and self.global_step > 0 and \
                       (self.global_step % self.args.save_interval == 0) and self.am_main_process:
                        avg_g_total_current = avg_losses_for_q.get('loss_g_total', -1.0) 
                        avg_d_total_current = avg_losses_for_q.get('loss_d_total', -1.0) 
                        chkpt_metrics_current = {
                            'train_loss_g_total_macro': avg_g_total_current if np.isfinite(avg_g_total_current) else -1.0,
                            'train_loss_d_total_macro': avg_d_total_current if np.isfinite(avg_d_total_current) else -1.0
                        }
                        self._save_checkpoint(is_intermediate=True, metrics=chkpt_metrics_current)

            validation_interval_epochs = getattr(self.args, 'validation_interval_epochs', 1) 
            if self.val_loader and self.am_main_process and validation_interval_epochs > 0 and \
               (epoch + 1) % validation_interval_epochs == 0:
                
                val_metrics_eoe = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log) 
                if val_metrics_eoe: 
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run: 
                        wandb.log({f"val/{k_val}": v_val for k_val, v_val in val_metrics_eoe.items()}, step=self.global_step)
                    
                    metric_to_check_eoe = self.args.val_primary_metric
                    current_val_for_best_eoe: float = val_metrics_eoe.get(metric_to_check_eoe, self.best_val_metric_val) 
                    
                    is_better_eoe = (current_val_for_best_eoe > self.best_val_metric_val) if self.is_val_metric_higher_better \
                                else (current_val_for_best_eoe < self.best_val_metric_val)
                    
                    if is_better_eoe and np.isfinite(current_val_for_best_eoe):
                        prog_bar.write(
                            f"New best val metric ({metric_to_check_eoe}): {current_val_for_best_eoe:.4f} "
                            f"(prev: {self.best_val_metric_val:.4f}). Saving best checkpoint."
                        )
                        self.best_val_metric_val = current_val_for_best_eoe
                        self._save_checkpoint(is_best=True, metrics=val_metrics_eoe)
            
            save_epoch_interval_epochs = getattr(self.args, 'save_epoch_interval', 1)
            if self.am_main_process and save_epoch_interval_epochs > 0 and (epoch + 1) % save_epoch_interval_epochs == 0:
                already_saved_as_best_this_epoch = 'is_better_eoe' in locals() and locals().get('is_better_eoe', False) and \
                                        np.isfinite(locals().get('current_val_for_best_eoe', float('inf')))
                
                # Check if an intermediate save happened at the very end of the epoch
                is_last_grad_accum_step_of_epoch = (batch_idx +1) == num_batches_epoch and \
                                                    (batch_idx +1) % self.grad_accum_steps == 0 
                
                already_saved_as_intermediate_this_step = self.args.save_interval > 0 and \
                                                self.global_step > 0 and \
                                                self.global_step % self.args.save_interval == 0 and \
                                                is_last_grad_accum_step_of_epoch


                if not (already_saved_as_best_this_epoch or already_saved_as_intermediate_this_step):
                    eoe_metrics_for_save = self.last_val_metrics.copy() if self.last_val_metrics else {}
                    # Use the most recent avg_losses_for_q if available for train loss approximation
                    if 'avg_losses_for_q' in locals() and isinstance(locals()['avg_losses_for_q'], dict):
                        eoe_metrics_for_save["epoch_end_train_g_total_approx"] = locals()['avg_losses_for_q'].get('loss_g_total', -1.0)
                        eoe_metrics_for_save["epoch_end_train_d_total_approx"] = locals()['avg_losses_for_q'].get('loss_d_total', -1.0)
                    self._save_checkpoint(metrics=eoe_metrics_for_save)

    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Optional[Dict[str, float]]:
        if not self.val_loader or not self.am_main_process: return None
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_ref_val = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        
        original_training_mode_m = m_ref.training
        original_training_mode_d = d_ref_val.training
        m_ref.eval(); d_ref_val.eval()

        total_recon_dct_mse_sum = 0.0; total_mel_mse_sum = 0.0; total_psnr_mel_sum = 0.0
        total_ssim_mel_sum = 0.0; total_lpips_mel_sum = 0.0; total_items_evaluated = 0
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0, device=self.device)).dtype 
        
        prog_bar_disabled = not self.am_main_process or os.getenv('CI') == 'true' or getattr(self.args, 'disable_val_tqdm', False)
        logged_samples_count_this_val = 0

        for batch_idx_val, batch_real_mel_segments in enumerate(
            tqdm(self.val_loader, desc="Validating", disable=prog_bar_disabled, dynamic_ncols=True)
        ):
            real_mel_segments = batch_real_mel_segments.to(self.device, dtype=dtype_m)
            B, _, H_mel, W_mel = real_mel_segments.shape

            recon_norm_dcts, _, _, gaad_bboxes_from_enc, target_norm_dcts_for_loss = m_ref(real_mel_segments)
            loss_recon_dct_batch = self._compute_recon_loss(recon_norm_dcts, target_norm_dcts_for_loss)
            if torch.isfinite(loss_recon_dct_batch): total_recon_dct_mse_sum += loss_recon_dct_batch.item() * B
            
            unnorm_recon_dcts = AudioSpecGenerator._unnormalize_dct(recon_norm_dcts, self.args)
            recon_mel_assembled = d_ref_val._assemble_mel_from_dct_regions(unnorm_recon_dcts, gaad_bboxes_from_enc, real_mel_segments.shape)
            
            if recon_mel_assembled.shape == real_mel_segments.shape:
                loss_mel_mse_batch = F.mse_loss(recon_mel_assembled, real_mel_segments, reduction='mean')
                if torch.isfinite(loss_mel_mse_batch):
                    total_mel_mse_sum += loss_mel_mse_batch.item() * B
                    mse_val = loss_mel_mse_batch.item()
                    psnr_val = 10 * math.log10(1.0 / (mse_val + EPS)) if mse_val > EPS else 100.0 
                    total_psnr_mel_sum += psnr_val * B
                
                recon_mel_01 = (recon_mel_assembled.clamp(-1,1)+1)/2.0
                real_mel_01 = (real_mel_segments.clamp(-1,1)+1)/2.0

                if self.ssim_metric:
                    try: ssim_val = self.ssim_metric(recon_mel_01, real_mel_01); total_ssim_mel_sum += ssim_val.item() * B
                    except Exception as e_ssim: self.logger.debug(f"Val SSIM failed: {e_ssim}")
                if self.lpips_loss_fn:
                    try:
                        rec_lpips_in = recon_mel_assembled.repeat(1,3,1,1) if recon_mel_assembled.shape[1]==1 else recon_mel_assembled
                        real_lpips_in = real_mel_segments.repeat(1,3,1,1) if real_mel_segments.shape[1]==1 else real_mel_segments
                        lpips_val = self.lpips_loss_fn(rec_lpips_in, real_lpips_in); total_lpips_mel_sum += lpips_val.sum().item() 
                    except Exception as e_lpips: self.logger.debug(f"Val LPIPS failed: {e_lpips}")
            
            total_items_evaluated += B
            if logged_samples_count_this_val < num_val_samples_to_log and \
               self.args.wandb and WANDB_AVAILABLE and wandb.run: 
                num_to_log_now = min(B, num_val_samples_to_log - logged_samples_count_this_val)
                if num_to_log_now > 0:
                    self._log_samples_to_wandb("val_recon_mel", recon_mel_assembled[:num_to_log_now], num_to_log_now)
                    self._log_samples_to_wandb("val_target_mel", real_mel_segments[:num_to_log_now], num_to_log_now)
                logged_samples_count_this_val += num_to_log_now
        
        m_ref.train(original_training_mode_m)
        d_ref_val.train(original_training_mode_d)

        if total_items_evaluated == 0: return None
        metrics = {
            "avg_val_recon_dct_mse": total_recon_dct_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_mel_mse": total_mel_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_psnr_mel": total_psnr_mel_sum / total_items_evaluated if total_items_evaluated > 0 else 0.0,
            "avg_val_ssim_mel": total_ssim_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.ssim_metric else 0.0,
            "avg_val_lpips_mel": total_lpips_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.lpips_loss_fn else float('inf')
        }
        self.last_val_metrics = metrics
        self.logger.info(f"Validation Metrics (Ep {self.current_epoch+1}, GStep {self.global_step}, ActiveD: {self.active_discriminator_key}): " + 
                         ", ".join([f"{k}:{v:.4f}" for k,v in metrics.items()]))
        return metrics

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict[str, Any]] = None, is_best: bool = False):
        if not self.am_main_process: return
        m_s = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_primary_s = self.discriminator_primary_obj.module if self.ddp_active and hasattr(self.discriminator_primary_obj, 'module') else self.discriminator_primary_obj
        d_alt_s = self.discriminator_alternative_obj.module if self.ddp_active and hasattr(self.discriminator_alternative_obj, 'module') else self.discriminator_alternative_obj

        def get_q_state_from_controller_or_optimizer(obj_with_q_controller):
            if obj_with_q_controller is None: return None
            q_ctrl_to_save: Optional[HAKMEMQController] = None
            if isinstance(obj_with_q_controller, HAKMEMQController): q_ctrl_to_save = obj_with_q_controller
            elif hasattr(obj_with_q_controller, 'q_controller'): q_ctrl_to_save = getattr(obj_with_q_controller, 'q_controller', None)
            if not q_ctrl_to_save or not hasattr(q_ctrl_to_save, 'q_table'): return None
            state = { 'q_table': q_ctrl_to_save.q_table, 'epsilon': q_ctrl_to_save.epsilon,
                     'prev_lr_mom_state': q_ctrl_to_save.prev_lr_mom_state, 'prev_lr_mom_action': q_ctrl_to_save.prev_lr_mom_action,
                     'prev_lambda_kl_state': q_ctrl_to_save.prev_lambda_kl_state, 'prev_lambda_kl_action': q_ctrl_to_save.prev_lambda_kl_action,
                     'reward_hist': list(q_ctrl_to_save.reward_hist),
                     'q_table_access_count': dict(q_ctrl_to_save.q_table_access_count),
                     'q_table_creation_time': q_ctrl_to_save.q_table_creation_time, 'q_table_last_access_time': q_ctrl_to_save.q_table_last_access_time,
                     'on_probation': getattr(q_ctrl_to_save, 'on_probation', False), 'current_probation_step': getattr(q_ctrl_to_save, 'current_probation_step', 0),
                     'lkl_on_probation': getattr(q_ctrl_to_save, 'lkl_on_probation', False), 'lkl_current_probation_step': getattr(q_ctrl_to_save, 'lkl_current_probation_step', 0)}
            if hasattr(q_ctrl_to_save, 'loss_g_total_hist'):
                q_hist_names = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']
                state['loss_histories'] = {hname: list(getattr(q_ctrl_to_save, f"loss_{hname}_hist")) for hname in q_hist_names if hasattr(q_ctrl_to_save, f"loss_{hname}_hist")}
            if hasattr(q_ctrl_to_save, 'interval_avg_recon_hist'):
                q_lkl_hist_names = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']
                state['interval_histories'] = {hname: list(getattr(q_ctrl_to_save, f"interval_{hname}_hist")) for hname in q_lkl_hist_names if hasattr(q_ctrl_to_save, f"interval_{hname}_hist")}
            return state


        data = {
            'global_step': self.global_step, 'epoch': self.current_epoch,
            'model_state_dict': m_s.state_dict(),
            'discriminator_primary_state_dict': d_primary_s.state_dict(),
            'discriminator_alternative_state_dict': d_alt_s.state_dict(),
            'active_discriminator_key': self.active_discriminator_key,
            'active_disc_actual_type': self.active_disc_actual_type, 
            'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict() if self.optimizer_enc_gen else None,
            'optimizer_disc_primary_state_dict': self.optimizer_disc_primary.state_dict(),
            'optimizer_disc_alternative_state_dict': self.optimizer_disc_alternative.state_dict(),
            'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict(),
            'scaler_disc_state_dict': self.scaler_disc.state_dict(),
            'args': vars(self.args), 'metrics': metrics if metrics is not None else self.last_val_metrics.copy(),
            'best_val_metric_val': self.best_val_metric_val, 'current_lambda_kl': self.lambda_kl,
            'prev_interval_metrics_for_lambda_kl_reward': self.prev_interval_metrics_for_lambda_kl_reward,
            
            'steps_since_last_d_switch': self.steps_since_last_d_switch,
            'consecutive_trigger_primary_to_alt_count': self.consecutive_trigger_primary_to_alt_count,
            'consecutive_trigger_alt_to_primary_count': self.consecutive_trigger_alt_to_primary_count,
            'consecutive_heuristic_trigger_counts': dict(self.consecutive_heuristic_trigger_counts),
            'q_data_derived_g_recon_hist': list(self.q_data_derived_g_recon_hist),
            'avg_g_recon_hist_for_stagnation': list(self.avg_g_recon_hist_for_stagnation),
            
            'heuristic_vae_feature_match_active': self.heuristic_vae_feature_match_active,
            'heuristic_penalize_g_easy_win_active': self.heuristic_penalize_g_easy_win_active,
            'heuristic_boost_active_d_lr_active': self.heuristic_boost_active_d_lr_active,
            'heuristic_force_d_q_explore_active': self.heuristic_force_d_q_explore_active, 
            'heuristic_override_lambda_recon_factor': self.heuristic_override_lambda_recon_factor,
            'heuristic_override_lambda_kl_factor': self.heuristic_override_lambda_kl_factor,
            'heuristic_override_lambda_gan_factor': self.heuristic_override_lambda_gan_factor,
        }
        
        data['q_controller_enc_gen_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_gen)
        data['q_controller_disc_primary_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_d_primary)
        data['q_controller_disc_alternative_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_d_alt)
        data['q_controller_lambda_kl_state'] = get_q_state_from_controller_or_optimizer(self.lambda_kl_q_controller)

        fprefix = "wubuspectrans_ckpt_v011"
        if is_best: fp_str = f"{fprefix}_best_ep{self.current_epoch + 1}_step{self.global_step}.pt" 
        elif is_intermediate: fp_str = f"{fprefix}_step{self.global_step}.pt"
        else: fp_str = f"{fprefix}_ep{self.current_epoch + 1}_step{self.global_step}.pt"
        fp = Path(self.args.checkpoint_dir) / fp_str
        try: torch.save(data, fp); self.logger.info(f"Checkpoint saved: {fp.name}")
        except Exception as e: self.logger.error(f"Save CKPT error {fp}: {e}", exc_info=True)

    def _load_q_state_helper_inner(self, q_controller_instance: Optional[HAKMEMQController],
                                   q_state_from_ckpt: Optional[Dict],
                                   perform_manual_flush_for_this_controller: bool,
                                   is_associated_optimizer_state_loaded: bool):
        if q_controller_instance is None:
            return

        reset_this_q_controller = perform_manual_flush_for_this_controller or \
                                  (not is_associated_optimizer_state_loaded and not perform_manual_flush_for_this_controller) or \
                                  (q_state_from_ckpt is None)

        context_msg = f"Q-Ctrl Load Helper for {q_controller_instance.logger.name if hasattr(q_controller_instance, 'logger') else 'UnknownQCtrl'}"

        if reset_this_q_controller:
            reset_reason = "global flush request" if perform_manual_flush_for_this_controller else \
                           ("associated optimizer not loaded" if not is_associated_optimizer_state_loaded else "no Q-state in checkpoint")
            if self.am_main_process:
                self.logger.info(f"{context_msg}: Resetting Q-controller state (Reason: {reset_reason}). It will start fresh, possibly on probation.")
            q_controller_instance.reset_q_learning_state(
                reset_q_table=True,
                reset_epsilon=True,
                context_msg=f"{context_msg} (Full Reset due to {reset_reason})",
                start_probation=True 
            )
            return

        try:
            q_controller_instance.q_table = q_state_from_ckpt.get('q_table', {})
            q_controller_instance.epsilon = q_state_from_ckpt.get('epsilon', q_controller_instance.epsilon_start)
            q_controller_instance.prev_lr_mom_state = q_state_from_ckpt.get('prev_lr_mom_state')
            q_controller_instance.prev_lr_mom_action = q_state_from_ckpt.get('prev_lr_mom_action')
            q_controller_instance.prev_lambda_kl_state = q_state_from_ckpt.get('prev_lambda_kl_state')
            q_controller_instance.prev_lambda_kl_action = q_state_from_ckpt.get('prev_lambda_kl_action')

            reward_hist_list = q_state_from_ckpt.get('reward_hist', [])
            q_controller_instance.reward_hist.clear()
            q_controller_instance.reward_hist.extend(reward_hist_list)

            if 'loss_histories' in q_state_from_ckpt:
                for hname, hlist in q_state_from_ckpt['loss_histories'].items():
                    hist_attr_deque = getattr(q_controller_instance, f"loss_{hname}_hist", None)
                    if hist_attr_deque is not None and isinstance(hist_attr_deque, deque):
                        hist_attr_deque.clear()
                        hist_attr_deque.extend(hlist)

            if 'interval_histories' in q_state_from_ckpt:
                 for hname, hlist in q_state_from_ckpt['interval_histories'].items():
                    hist_attr_deque = getattr(q_controller_instance, f"interval_{hname}_hist", None)
                    if hist_attr_deque is not None and isinstance(hist_attr_deque, deque):
                        hist_attr_deque.clear()
                        hist_attr_deque.extend(hlist)


            q_controller_instance.q_table_access_count = defaultdict(int, q_state_from_ckpt.get('q_table_access_count', {}))
            q_controller_instance.q_table_creation_time = q_state_from_ckpt.get('q_table_creation_time', {})
            q_controller_instance.q_table_last_access_time = q_state_from_ckpt.get('q_table_last_access_time', {})

            q_controller_instance.on_probation = q_state_from_ckpt.get('on_probation', False)
            q_controller_instance.current_probation_step = q_state_from_ckpt.get('current_probation_step', 0)
            q_controller_instance.lkl_on_probation = q_state_from_ckpt.get('lkl_on_probation', False)
            q_controller_instance.lkl_current_probation_step = q_state_from_ckpt.get('lkl_current_probation_step', 0)

            if self.am_main_process:
                self.logger.info(f"{context_msg}: Successfully loaded Q-state from checkpoint.")
        except Exception as e_load_q:
            if self.am_main_process:
                self.logger.warning(f"{context_msg}: Error loading Q-state from checkpoint: {e_load_q}. Resetting this Q-controller.", exc_info=True)
            q_controller_instance.reset_q_learning_state(
                reset_q_table=True, reset_epsilon=True,
                context_msg=f"{context_msg} (Reset due to load error)",
                start_probation=True
            )

    def load_checkpoint(self, checkpoint_path_str: str) -> Tuple[int, int]:
        checkpoint_path = Path(checkpoint_path_str)

        q_ctrl_gen = getattr(self.optimizer_enc_gen, 'q_controller', None)
        q_ctrl_d_pri = self.q_controller_d_primary 
        q_ctrl_d_alt = self.q_controller_d_alt   
        q_ctrl_lkl = self.lambda_kl_q_controller 

        all_q_controllers = [
            qc for qc in [q_ctrl_gen, q_ctrl_d_pri, q_ctrl_d_alt, q_ctrl_lkl] if qc is not None
        ]

        global_manual_flush_requested = getattr(HAKMEMQController, 'MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD', False)
        effective_reset_request_for_q = global_manual_flush_requested or self.args.reset_q_controllers_on_load

        if not checkpoint_path.exists():
            self.logger.warning(f"CKPT {checkpoint_path} not found. Starting fresh.")
            if self.args.reset_lkl_q_controller_on_load:
                self.lambda_kl = float(self.args.lambda_kl) # Reset to arg value
                self.logger.info(f"CKPT not found, but --reset_lkl_q_controller_on_load is True. Setting self.lambda_kl to args.lambda_kl: {self.lambda_kl:.2e}")

            for qc_obj in all_q_controllers:
                is_lkl_and_reset_lkl_arg = (qc_obj == q_ctrl_lkl and self.args.reset_lkl_q_controller_on_load)
                perform_reset_for_this_specific_controller = effective_reset_request_for_q or is_lkl_and_reset_lkl_arg

                self._load_q_state_helper_inner(qc_obj, None,
                                                perform_manual_flush_for_this_controller=perform_reset_for_this_specific_controller,
                                                is_associated_optimizer_state_loaded=False)
                if is_lkl_and_reset_lkl_arg and qc_obj is not None: 
                    qc_obj.set_current_lambda_kl(self.lambda_kl) # Ensure LKL Q-Ctrl has the reset lambda_kl

            if global_manual_flush_requested and not self.args.reset_q_controllers_on_load: # If only manual flush was on
                HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False # Reset the global flag
            return 0, 0

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.logger.info(f"Loaded CKPT: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load CKPT {checkpoint_path}: {e}. Starting fresh.", exc_info=True)
            if self.args.reset_lkl_q_controller_on_load:
                self.lambda_kl = float(self.args.lambda_kl) # Reset to arg value
                self.logger.info(f"CKPT load failed, but --reset_lkl_q_controller_on_load is True. Setting self.lambda_kl to args.lambda_kl: {self.lambda_kl:.2e}")

            for qc_obj in all_q_controllers:
                is_lkl_and_reset_lkl_arg = (qc_obj == q_ctrl_lkl and self.args.reset_lkl_q_controller_on_load)
                perform_reset_for_this_specific_controller = effective_reset_request_for_q or is_lkl_and_reset_lkl_arg
                self._load_q_state_helper_inner(qc_obj, None,
                                                perform_manual_flush_for_this_controller=perform_reset_for_this_specific_controller,
                                                is_associated_optimizer_state_loaded=False)
                if is_lkl_and_reset_lkl_arg and qc_obj is not None:
                     qc_obj.set_current_lambda_kl(self.lambda_kl) # Ensure LKL Q-Ctrl has the reset lambda_kl

            if global_manual_flush_requested and not self.args.reset_q_controllers_on_load: # If only manual flush was on
                HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False # Reset the global flag
            return 0, 0

        # --- State Dict Loading for Models ---
        ckpt_args_dict = ckpt.get('args', vars(self.args))
        
        # Determine architecture flags from the checkpoint for more robust loading
        ckpt_disc_apply_spectral_norm = ckpt_args_dict.get('disc_apply_spectral_norm', self.args.disc_apply_spectral_norm)
        
        # We need to know if the *current args* for this run specify SN for the D that will be built
        # The D objects are built using current_args.disc_apply_spectral_norm in _get_discriminator_configs
        current_run_disc_apply_spectral_norm = self.args.disc_apply_spectral_norm

        m_load = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_primary_load = self.discriminator_primary_obj.module if self.ddp_active and hasattr(self.discriminator_primary_obj, 'module') else self.discriminator_primary_obj
        d_alt_load = self.discriminator_alternative_obj.module if self.ddp_active and hasattr(self.discriminator_alternative_obj, 'module') else self.discriminator_alternative_obj
        
        model_loaded_ok, disc_primary_model_loaded_ok, disc_alt_model_loaded_ok = False, False, False

        try:
            if 'model_state_dict' in ckpt:
                 # For VAE, strict=False is generally safer due to potential WuBu config changes
                m_load.load_state_dict(ckpt['model_state_dict'], strict=False) 
                model_loaded_ok = True
                self.logger.info("Main model state_dict loaded (strict=False).")
            else: self.logger.warning("Main model_state_dict not found in checkpoint.")
        except Exception as e: self.logger.error(f"Error loading main model state_dict: {e}", exc_info=True) # Show full error

        # Handle primary discriminator
        if 'discriminator_primary_state_dict' in ckpt and d_primary_load:
            try:
                if ckpt_disc_apply_spectral_norm != current_run_disc_apply_spectral_norm:
                    self.logger.warning(f"Primary D spectral_norm mismatch! Ckpt SN: {ckpt_disc_apply_spectral_norm}, Current Run SN: {current_run_disc_apply_spectral_norm}. "
                                        "Loading with strict=False. THIS CAN LEAD TO UNEXPECTED BEHAVIOR OR ERRORS if parameter names changed.")
                d_primary_load.load_state_dict(ckpt['discriminator_primary_state_dict'], strict=False) 
                disc_primary_model_loaded_ok = True
                self.logger.info(f"Primary D ({self.primary_disc_actual_type}) state_dict loaded (strict=False).")
            except Exception as e: self.logger.error(f"Error loading D_primary state_dict: {e}", exc_info=True)
        elif not d_primary_load: self.logger.warning("discriminator_primary_obj is None, cannot load its state.")
        else: self.logger.warning("discriminator_primary_state_dict not found in checkpoint.")
        
        # Handle alternative discriminator
        if 'discriminator_alternative_state_dict' in ckpt and d_alt_load:
            try:
                if ckpt_disc_apply_spectral_norm != current_run_disc_apply_spectral_norm: # Assuming alt D uses same SN flag from args
                    self.logger.warning(f"Alternative D spectral_norm mismatch! Ckpt SN: {ckpt_disc_apply_spectral_norm}, Current Run SN: {current_run_disc_apply_spectral_norm}. "
                                        "Loading with strict=False.")
                d_alt_load.load_state_dict(ckpt['discriminator_alternative_state_dict'], strict=False)
                disc_alt_model_loaded_ok = True
                self.logger.info(f"Alternative D ({self.alternative_disc_actual_type}) state_dict loaded (strict=False).")
            except Exception as e: self.logger.error(f"Error loading D_alternative state_dict: {e}", exc_info=True)
        elif not d_alt_load: self.logger.warning("discriminator_alternative_obj is None, cannot load its state.")
        else: self.logger.warning("discriminator_alternative_state_dict not found in checkpoint.")


        # --- Resume other trainer states ---
        self.is_val_metric_higher_better = self.args.val_primary_metric in ["avg_val_ssim_mel", "avg_val_psnr_mel"]
        default_best_val = -float('inf') if self.is_val_metric_higher_better else float('inf')
        self.best_val_metric_val = ckpt.get('best_val_metric_val', default_best_val)
        self.last_val_metrics = ckpt.get('metrics', {}).copy() if ckpt.get('metrics') is not None else {}
        
        if not self.args.reset_lkl_q_controller_on_load:
            self.lambda_kl = float(ckpt.get('current_lambda_kl', self.args.lambda_kl))
        # If reset_lkl_q_controller_on_load is True, self.lambda_kl should be set to args.lambda_kl
        # This happens before LKL Q-Ctrl reset.
        elif self.args.reset_lkl_q_controller_on_load:
             self.lambda_kl = float(self.args.lambda_kl) # Ensure it's from current args if LKL Q is reset
             self.logger.info(f"Trainer's self.lambda_kl set to args.lambda_kl: {self.lambda_kl:.2e} due to --reset_lkl_q_controller_on_load.")


        self.prev_interval_metrics_for_lambda_kl_reward = ckpt.get('prev_interval_metrics_for_lambda_kl_reward')


        loaded_gs = ckpt.get('global_step', 0)
        loaded_ep = ckpt.get('epoch', 0)
        
        next_ep_start = loaded_ep + 1 if model_loaded_ok and loaded_gs > 0 and loaded_ep < self.args.epochs else loaded_ep 
        if getattr(self.args, 'force_start_epoch_on_load', None) is not None:
            next_ep_start = self.args.force_start_epoch_on_load
            loaded_gs = getattr(self.args, 'force_start_gstep_on_load', 0 if self.args.force_start_epoch_on_load is not None else loaded_gs)
            if self.am_main_process: self.logger.info(f"CKPT Load: Overriding start epoch to {next_ep_start} and GStep to {loaded_gs} due to force_start args.")

        # --- Discriminator switching state ---
        saved_active_disc_key = ckpt.get('active_discriminator_key', 'primary')
        saved_active_disc_actual_type = ckpt.get('active_disc_actual_type', 'unknown_type_in_ckpt')
        target_active_key_for_this_resume = saved_active_disc_key 
        forced_switch_on_resume = False
        if self.args.enable_heuristic_disc_switching and self.initial_disc_type_arg is not None:
            current_args_implied_active_key = None
            if self.initial_disc_type_arg == self.primary_disc_actual_type: current_args_implied_active_key = 'primary'
            elif self.initial_disc_type_arg == self.alternative_disc_actual_type: current_args_implied_active_key = 'alternative'
            
            if current_args_implied_active_key is not None and current_args_implied_active_key != saved_active_disc_key:
                if self.am_main_process: self.logger.warning(f"LOAD_CKPT_OVERRIDE: Checkpoint active D was '{saved_active_disc_key}' (type: '{saved_active_disc_actual_type}'). Current args.initial_disc_type ('{self.initial_disc_type_arg}') implies '{current_args_implied_active_key}'. FORCING active D to '{current_args_implied_active_key}' for this resume.")
                target_active_key_for_this_resume = current_args_implied_active_key
                forced_switch_on_resume = True
            elif current_args_implied_active_key is None and self.am_main_process: 
                self.logger.warning(f"LOAD_CKPT_WARNING: args.initial_disc_type ('{self.initial_disc_type_arg}') did not match actual types of primary ('{self.primary_disc_actual_type}') or alternative ('{self.alternative_disc_actual_type}'). Using active D key from checkpoint: '{saved_active_disc_key}'.")
        
        self.active_discriminator_key = target_active_key_for_this_resume
        self._update_active_discriminator_pointers()

        # --- Optimizer and Q-Controller states ---
        opt_g_loaded_ok, opt_d_primary_loaded_ok, opt_d_alt_loaded_ok = False, False, False
        if self.optimizer_enc_gen and 'optimizer_enc_gen_state_dict' in ckpt and ckpt['optimizer_enc_gen_state_dict'] is not None:
            if model_loaded_ok: 
                try: self.optimizer_enc_gen.load_state_dict(ckpt['optimizer_enc_gen_state_dict']); opt_g_loaded_ok = True
                except Exception as e: self.logger.warning(f"Could not load Opt_Gen state: {e}. It will start fresh.")
            else: self.logger.warning("Main model failed to load, Opt_Gen will start fresh.")
        if self.optimizer_enc_gen: 
            for group in self.optimizer_enc_gen.param_groups: 
                group['initial_lr'] = self.args.learning_rate_gen 
                group['initial_momentum'] = self.optimizer_enc_gen.defaults.get('momentum', 0.9)

        if self.optimizer_disc_primary and 'optimizer_disc_primary_state_dict' in ckpt and ckpt['optimizer_disc_primary_state_dict'] is not None:
            if disc_primary_model_loaded_ok:
                try: self.optimizer_disc_primary.load_state_dict(ckpt['optimizer_disc_primary_state_dict']); opt_d_primary_loaded_ok = True
                except Exception as e: self.logger.warning(f"Could not load Opt_D_Primary state: {e}. It will start fresh.")
            else: self.logger.warning("D_Primary model failed to load, Opt_D_Primary will start fresh.")
        if self.optimizer_disc_primary:
            for group in self.optimizer_disc_primary.param_groups:
                group['initial_lr'] = self.args.learning_rate_disc
                group['initial_momentum'] = self.optimizer_disc_primary.defaults.get('momentum', 0.9)

        lr_disc_alt_load = getattr(self.args, 'learning_rate_disc_alt', self.args.learning_rate_disc)
        if self.optimizer_disc_alternative and 'optimizer_disc_alternative_state_dict' in ckpt and ckpt['optimizer_disc_alternative_state_dict'] is not None:
            if disc_alt_model_loaded_ok:
                try: self.optimizer_disc_alternative.load_state_dict(ckpt['optimizer_disc_alternative_state_dict']); opt_d_alt_loaded_ok = True
                except Exception as e: self.logger.warning(f"Could not load Opt_D_Alt state: {e}. It will start fresh.")
            else: self.logger.warning("D_Alternative model failed to load, Opt_D_Alt will start fresh.")
        if self.optimizer_disc_alternative:
            for group in self.optimizer_disc_alternative.param_groups:
                group['initial_lr'] = lr_disc_alt_load
                group['initial_momentum'] = self.optimizer_disc_alternative.defaults.get('momentum', 0.9)

        self._load_q_state_helper_inner(q_ctrl_gen, ckpt.get('q_controller_enc_gen_state'), effective_reset_request_for_q, opt_g_loaded_ok)
        self._load_q_state_helper_inner(q_ctrl_d_pri, ckpt.get('q_controller_disc_primary_state'), effective_reset_request_for_q, opt_d_primary_loaded_ok)
        self._load_q_state_helper_inner(q_ctrl_d_alt, ckpt.get('q_controller_disc_alternative_state'), effective_reset_request_for_q, opt_d_alt_loaded_ok)

        if self.args.reset_lkl_q_controller_on_load and q_ctrl_lkl is not None:
            self.logger.info(f"FORCE RESETTING Lambda_KL Q-Controller due to --reset_lkl_q_controller_on_load.")
            # self.lambda_kl is already set from args if this flag is true and ckpt didn't exist,
            # or from ckpt then potentially overridden by args if force_start_epoch.
            # Here, explicitly ensure it's from current args if this flag is active.
            self.lambda_kl = float(self.args.lambda_kl)
            q_ctrl_lkl.reset_q_learning_state(reset_q_table=True, reset_epsilon=True, context_msg="LKL Q-Ctrl Force Reset on Load by Arg", start_probation=True)
            self.logger.info(f"Trainer's self.lambda_kl (set to args value): {self.lambda_kl:.2e} after LKL Q-Ctrl reset.")
            q_ctrl_lkl.set_current_lambda_kl(self.lambda_kl) 
            self.prev_interval_metrics_for_lambda_kl_reward = None 
            self.logger.info("prev_interval_metrics_for_lambda_kl_reward also reset due to LKL Q-Ctrl reset.")
        else: # Not resetting LKL Q-Ctrl due to the specific arg, so load its state
             # effective_reset_request_for_q still applies if --reset_q_controllers_on_load (general) was true
            self._load_q_state_helper_inner(q_ctrl_lkl, ckpt.get('q_controller_lambda_kl_state'), effective_reset_request_for_q, True) 

        if forced_switch_on_resume:
            active_d_q_to_reset = getattr(self.optimizer_disc_active, 'q_controller', None)
            if active_d_q_to_reset:
                if self.am_main_process: self.logger.warning(f"Due to resume override, resetting Q-controller for newly FORCED active D: '{self.active_discriminator_key}' (Type: {self.active_disc_actual_type}).")
                active_d_q_to_reset.reset_q_learning_state(True, True, f"Forced D switch to {self.active_discriminator_key} on Resume Override", True)
            
            self.steps_since_last_d_switch = 0
            self.consecutive_trigger_primary_to_alt_count = 0; self.consecutive_trigger_alt_to_primary_count = 0
            self.consecutive_heuristic_trigger_counts = defaultdict(int)
            self.q_data_derived_g_recon_hist.clear(); self.rec_dct_stagnant = False
            self.avg_g_recon_hist_for_stagnation.clear() 
            if self.am_main_process: self.logger.info("Heuristic switching counters and short-term recon history reset due to forced D switch on resume.")
        else: # Load heuristic counters if no forced switch
            self.steps_since_last_d_switch = ckpt.get('steps_since_last_d_switch', 0)
            self.consecutive_trigger_primary_to_alt_count = ckpt.get('consecutive_trigger_primary_to_alt_count', 0)
            self.consecutive_trigger_alt_to_primary_count = ckpt.get('consecutive_trigger_alt_to_primary_count', 0)
            self.consecutive_heuristic_trigger_counts = defaultdict(int, ckpt.get('consecutive_heuristic_trigger_counts', {}))
            if 'q_data_derived_g_recon_hist' in ckpt and ckpt['q_data_derived_g_recon_hist'] is not None:
                try:
                    self.q_data_derived_g_recon_hist.clear()
                    self.q_data_derived_g_recon_hist.extend(list(ckpt['q_data_derived_g_recon_hist']))
                except TypeError: self.logger.warning(f"Could not extend deque q_data_derived_g_recon_hist from checkpoint.")
            if 'avg_g_recon_hist_for_stagnation' in ckpt and ckpt['avg_g_recon_hist_for_stagnation'] is not None:
                try:
                    self.avg_g_recon_hist_for_stagnation.clear()
                    self.avg_g_recon_hist_for_stagnation.extend(list(ckpt['avg_g_recon_hist_for_stagnation']))
                except TypeError: self.logger.warning(f"Could not extend deque avg_g_recon_hist_for_stagnation from checkpoint.")

        # Load heuristic active states and factors
        self.heuristic_vae_feature_match_active = ckpt.get('heuristic_vae_feature_match_active', False)
        self.heuristic_penalize_g_easy_win_active = ckpt.get('heuristic_penalize_g_easy_win_active', False)
        self.heuristic_boost_active_d_lr_active = ckpt.get('heuristic_boost_active_d_lr_active', False)
        self.heuristic_force_d_q_explore_active = ckpt.get('heuristic_force_d_q_explore_active', False) 
        self.heuristic_override_lambda_recon_factor = ckpt.get('heuristic_override_lambda_recon_factor', 1.0)
        self.heuristic_override_lambda_kl_factor = ckpt.get('heuristic_override_lambda_kl_factor', 1.0)
        self.heuristic_override_lambda_gan_factor = ckpt.get('heuristic_override_lambda_gan_factor', 1.0) 

        if global_manual_flush_requested and not self.args.reset_q_controllers_on_load:
            HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
            if self.am_main_process: self.logger.info("Global MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD applied and reset.")
        elif self.args.reset_q_controllers_on_load and self.am_main_process:
             self.logger.info("Global Q-controller reset triggered by --reset_q_controllers_on_load argument for all applicable Q-controllers.")

        if self.args.use_amp and self.device.type == 'cuda':
            if 'scaler_enc_gen_state_dict' in ckpt and self.scaler_enc_gen and ckpt['scaler_enc_gen_state_dict'] is not None:
                try: self.scaler_enc_gen.load_state_dict(ckpt['scaler_enc_gen_state_dict'])
                except Exception as e_sc_g: self.logger.warning(f"Could not load scaler_enc_gen state: {e_sc_g}")
            if 'scaler_disc_state_dict' in ckpt and self.scaler_disc and ckpt['scaler_disc_state_dict'] is not None:
                try: self.scaler_disc.load_state_dict(ckpt['scaler_disc_state_dict'])
                except Exception as e_sc_d: self.logger.warning(f"Could not load scaler_disc state: {e_sc_d}")

        for q_ctrl_sync in all_q_controllers:
            if q_ctrl_sync and hasattr(q_ctrl_sync, 'set_current_lambda_kl'):
                q_ctrl_sync.set_current_lambda_kl(self.lambda_kl)

        self.logger.info(
            f"Resuming training. GlobalStep: {loaded_gs}, NextEpochStart: {next_ep_start}. "
            f"ActiveD upon resume: '{self.active_discriminator_key}' (Type: '{self.active_disc_actual_type}'). "
            f"Effective Lambda_KL (base): {self.lambda_kl:.4e}" 
        )
        return loaded_gs, next_ep_start
        
    @staticmethod
    def get_scale_from_action_value(action_val: Union[Dict, str, None], scale_key: str, default: float = 1.0) -> float:
        if isinstance(action_val, dict): 
            val = action_val.get(scale_key)
            if val is not None:
                try: return float(val)
                except (ValueError, TypeError): return default
            return default
        return default

    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_ref_sample = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        
        original_mode_m = m_ref.training
        original_mode_d = d_ref_sample.training 
        m_ref.eval(); d_ref_sample.eval()

        dev = self.device
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0, device=self.device)).dtype
        lat_dim = self.args.latent_dim

        if noise is None: z = torch.randn(num_samples, lat_dim, device=dev, dtype=dtype_m)
        else: z = noise.to(device=dev, dtype=dtype_m); num_samples = z.shape[0]

        generated_norm_dcts = m_ref.decode(z) 
        unnorm_dcts_for_assembly = AudioSpecGenerator._unnormalize_dct(generated_norm_dcts, self.args)
        
        current_audio_config = self._get_audio_config_ref()
        current_gaad_config = self._get_gaad_config_ref()
        spec_time_frames = current_audio_config.get("num_time_frames_for_1s_segment", 86)
        spec_mels = self.args.n_mels
        spec_dims_canonical_for_gaad = (spec_time_frames, spec_mels) 
        
        canonical_bboxes_list = [] 
        for _ in range(num_samples):
            bboxes_one_sample = golden_subdivide_rect_fixed_n(
                spec_dims_canonical_for_gaad, 
                current_gaad_config['num_regions'], 
                dev, dtype_m, 
                current_gaad_config.get('min_size_px', 5)
            )
            canonical_bboxes_list.append(bboxes_one_sample)
        canonical_gaad_bboxes_batch = torch.stack(canonical_bboxes_list)
        
        target_mel_shape_for_sample = (num_samples, 1, spec_mels, spec_time_frames)
        
        generated_mel_spectrograms = d_ref_sample._assemble_mel_from_dct_regions(
            unnorm_dcts_for_assembly, canonical_gaad_bboxes_batch, target_mel_shape_for_sample
        )
        
        m_ref.train(original_mode_m)
        d_ref_sample.train(original_mode_d) 
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
        config["tangent_input_combination_dims"] = [DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0]]
        return config

    config["hyperbolic_dims"] = getattr(args, f"{prefix}_hyperbolic_dims", DEFAULT_CONFIG_WUBU["hyperbolic_dims"])
    config["initial_curvatures"] = getattr(args, f"{prefix}_initial_curvatures", DEFAULT_CONFIG_WUBU["initial_curvatures"])
    
    # Use the full argument name now since 'dest' was removed in parse_arguments
    config["use_rotation_in_transform"] = getattr(args, f"{prefix}_use_rotation", DEFAULT_CONFIG_WUBU["use_rotation_in_transform"])
    config["phi_influence_curvature"] = getattr(args, f"{prefix}_phi_influence_curvature", DEFAULT_CONFIG_WUBU["phi_influence_curvature"]) 
    config["phi_influence_rotation_init"] = getattr(args, f"{prefix}_phi_influence_rotation_init", DEFAULT_CONFIG_WUBU["phi_influence_rotation_init"]) 
    config["dropout"] = args.wubu_dropout

    def _ensure_list_len(cfg_dict, key, target_len, default_fill_list_from_defaults_ref):
        default_config_value_for_key = DEFAULT_CONFIG_WUBU.get(key, [])
        
        # Construct the full argument name based on prefix and key
        arg_name_for_key = f"{prefix}_{key}"
        current_val_from_args = getattr(args, arg_name_for_key, None)

        if current_val_from_args is None: 
             current_val = default_config_value_for_key
        else: 
             current_val = current_val_from_args
        
        is_list_orig = isinstance(current_val, list)
        current_list_val = current_val if is_list_orig else [current_val]

        if isinstance(default_config_value_for_key, list) and default_config_value_for_key:
            base_default_for_fill = default_config_value_for_key[0]
        elif not isinstance(default_config_value_for_key, list): 
             base_default_for_fill = default_config_value_for_key
        else: 
            base_default_for_fill = 1.0 if "scales" in key or "curvatures" in key else \
                                    0.1 if "spread" in key else ("linear" if "types" in key else 32)
        
        fill_val = current_list_val[-1] if current_list_val else base_default_for_fill
        
        if len(current_list_val) < target_len:
            cfg_dict[key] = (current_list_val + [fill_val]*(target_len-len(current_list_val)))[:target_len]
        elif len(current_list_val) > target_len:
            cfg_dict[key] = current_list_val[:target_len]
        else: 
            cfg_dict[key] = current_list_val

    for key_chk in ["hyperbolic_dims", "initial_curvatures",
                    "initial_scales", "initial_spread_values",
                    "boundary_points_per_level"]:
        _ensure_list_len(config, key_chk, num_levels_val, DEFAULT_CONFIG_WUBU.get(key_chk, []))
        
    if "boundary_points_per_level" in config and num_levels_val > 0:
        if not hasattr(args, f"{prefix}_boundary_points_per_level"): 
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
def validate_wubu_config_for_argparse(args_obj, prefix_str, parser_ref):
    num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0)
    if num_levels > 0:
        for suffix, attr_name in [("hyperbolic_dims", f"{prefix_str}_hyperbolic_dims"),
                                  ("initial_curvatures", f"{prefix_str}_initial_curvatures")]:
            val_list = getattr(args_obj, attr_name)
            is_list_original = isinstance(val_list, list)
            # Ensure it's a list for processing, handling single int/float if provided
            if not is_list_original and (isinstance(val_list, (int, float))):
                val_list = [val_list]
            elif not isinstance(val_list, list): # If not list and not single int/float, problem or unhandled type
                 parser_ref.error(f"Argument {attr_name} for {prefix_str} must be a list or a single number.")
                 return # Should not be reached due to parser.error

            if len(val_list) != num_levels:
                if len(val_list) == 1 and num_levels > 1: # Single value provided for multiple levels
                    setattr(args_obj, attr_name, [val_list[0]] * num_levels)
                elif not val_list: # Empty list provided
                    if suffix == "initial_curvatures":
                        setattr(args_obj, attr_name, [1.0] * num_levels) # Default curvature
                    elif suffix == "hyperbolic_dims":
                        default_dim_val = getattr(args_obj, 'latent_dim', 32 * num_levels) // num_levels if num_levels > 0 else 32
                        setattr(args_obj, attr_name, [max(1, default_dim_val)] * num_levels)
                    # No else needed for other lists, _configure_wubu_stack will use defaults
                else: # Mismatch and not a single value or empty
                    parser_ref.error(f"{prefix_str}: Length mismatch for {attr_name} (length {len(val_list)}) vs num_levels ({num_levels}). Provide one value or a list of {num_levels} values.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBuSpecTrans_v0.1.1: 1-Second Audio Segment VAE-GAN")

    # --- Group: Core Paths and DDP/General Setup ---
    core_group = parser.add_argument_group('Core Paths and DDP/General Setup')
    core_group.add_argument('--audio_dir_path', type=str, default="demo_audio_data_dir", help="Path to directory containing audio files or a single audio file.")
    core_group.add_argument('--checkpoint_dir',type=str, default='wubuspectrans_checkpoints_v011', help="Directory for checkpoints.")
    core_group.add_argument('--load_checkpoint', type=str, default=None, help="Path to checkpoint to load.")
    core_group.add_argument('--load_strict', action='store_true', help="Use strict=True when loading model state_dict.")
    core_group.add_argument('--local_rank', type=int, default=-1, help="DDP local rank (set by launch utility).")
    core_group.add_argument('--seed',type=int, default=42, help="Random seed for reproducibility.")
    core_group.add_argument('--num_workers',type=int, default=2, help="Number of DataLoader workers.")
    core_group.add_argument('--use_amp', action='store_true', help="Enable Automatic Mixed Precision training.")
    core_group.add_argument('--detect_anomaly',action='store_true', help="Enable PyTorch autograd anomaly detection (for debugging).")
    core_group.add_argument('--ddp_find_unused_params_d', action='store_true', help="Set find_unused_parameters=True for DDP wrapped Discriminators.")

    # --- Group: Training Hyperparameters ---
    train_hp_group = parser.add_argument_group('Training Hyperparameters')
    train_hp_group.add_argument('--epochs', type=int, default=1500, help="Total training epochs.")
    train_hp_group.add_argument('--batch_size', type=int, default=16, help="Batch size per GPU.")
    train_hp_group.add_argument('--grad_accum_steps',type=int, default=1, help="Number of steps to accumulate gradients.")
    train_hp_group.add_argument('--learning_rate_gen',type=float,default=1e-4, help="Learning rate for Generator/VAE.")
    train_hp_group.add_argument('--learning_rate_disc',type=float,default=1e-4, help="Learning rate for the primary Discriminator.")
    train_hp_group.add_argument('--learning_rate_disc_alt',type=float,default=None, help="Specific LR for alt Discriminator (defaults to learning_rate_disc).")
    train_hp_group.add_argument('--risgd_max_grad_norm',type=float,default=1.0, help="Max grad norm for Riemannian SGD per-parameter clipping.")
    train_hp_group.add_argument('--global_max_grad_norm',type=float,default=5.0, help="Global gradient clipping norm for optimizers (0 to disable).")

    # --- Group: Loss Weights ---
    loss_group = parser.add_argument_group('Loss Weights')
    loss_group.add_argument('--lambda_recon', type=float, default=10.0, help="Weight for VAE reconstruction loss.")
    loss_group.add_argument('--lambda_kl', type=float, default=0.01, help="Initial base weight for VAE KL divergence loss.")
    loss_group.add_argument('--lambda_gan', type=float, default=1.0, help="Weight for GAN adversarial loss (Generator part).")

    # --- Group: Audio Processing & Dataset ---
    audio_group = parser.add_argument_group('Audio Processing & Dataset')
    audio_group.add_argument('--sample_rate', type=int, default=22050)
    audio_group.add_argument('--n_fft', type=int, default=1024)
    audio_group.add_argument('--hop_length', type=int, default=256)
    audio_group.add_argument('--n_mels', type=int, default=128)
    audio_group.add_argument('--fmin', type=float, default=30.0)
    audio_group.add_argument('--fmax', type=float, default=None)
    audio_group.add_argument('--segment_duration_sec', type=float, default=1.0)
    audio_group.add_argument('--segment_overlap_sec', type=float, default=0.0)
    audio_group.add_argument('--db_norm_min', type=float, default=-80.0)
    audio_group.add_argument('--db_norm_max', type=float, default=0.0)
    audio_group.add_argument('--preload_audio_dataset_to_ram', action='store_true')
    audio_group.add_argument('--validation_audio_dir_path', type=str, default=None)
    audio_group.add_argument('--validation_split_fraction', type=float, default=0.1)

    # --- Group: GAAD (Spectrogram Regions) & DCT Processing ---
    gaad_dct_group = parser.add_argument_group('GAAD & DCT Processing')
    gaad_dct_group.add_argument('--gaad_num_regions', type=int, default=10)
    gaad_dct_group.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])
    gaad_dct_group.add_argument('--gaad_min_size_px', type=int, default=4)
    gaad_dct_group.add_argument('--region_proc_size_t', type=int, default=16)
    gaad_dct_group.add_argument('--region_proc_size_f', type=int, default=16)
    gaad_dct_group.add_argument('--dct_norm_type', type=str, default="tanh", choices=["none", "global_scale", "tanh"])
    gaad_dct_group.add_argument('--dct_norm_global_scale', type=float, default=100.0)
    gaad_dct_group.add_argument('--dct_norm_tanh_scale', type=float, default=30.0)

    # --- Group: Model Architecture (VAE & Discriminator Base) ---
    model_arch_group = parser.add_argument_group('Model Architecture (VAE & Discriminator Base)')
    model_arch_group.add_argument('--latent_dim', type=int, default=256)
    model_arch_group.add_argument('--encoder_initial_tangent_dim', type=int, default=128)
    model_arch_group.add_argument('--disc_input_type', type=str, default="mel", choices=["mel", "dct"])
    model_arch_group.add_argument('--disc_apply_spectral_norm', action='store_true')
    # Mel D CNN specific
    model_arch_group.add_argument('--disc_base_disc_channels', type=int, default=64)
    model_arch_group.add_argument('--disc_max_disc_channels', type=int, default=512)
    model_arch_group.add_argument('--disc_target_final_feature_dim', nargs='+', type=int, default=[4,4], help="Target HxW dim(s) of Mel D CNN feature map. Single int for square, two for H W.")
    model_arch_group.add_argument('--max_mel_disc_downsample_layers', type=int, default=6, help="Max downsampling layers in Mel D CNN.")
    model_arch_group.add_argument('--use_mel_d_attention', action='store_true', help="Use Self-Attention layer in Mel D CNN backbone(s).")
    model_arch_group.add_argument('--mel_d_attention_idx', type=int, default=2, help="Insert Self-Attention after this conv block index in Mel D (0-indexed).")
    model_arch_group.add_argument('--mel_d_msd_num_scales', type=int, default=1, help="Number of scales for Multi-Scale Mel Discriminator (1 means single D).")
    model_arch_group.add_argument('--mel_d_msd_share_weights', action='store_true', help="Share weights across scales in Mel MSD.")
    # DCT D specific (Transformer path)
    model_arch_group.add_argument('--disc_dct_embed_dim', type=int, default=None, help="Embedding dim for DCT coefficients in DCT D (default: encoder_initial_tangent_dim).")
    model_arch_group.add_argument('--disc_dct_use_pos_embed', action='store_true', help="Use positional embedding for regional features in DCT Transformer D.")
    model_arch_group.add_argument('--disc_dct_use_cls_token', action='store_true', help="Use a [CLS] token for aggregation in Transformer DCT D instead of mean pooling.")
    model_arch_group.add_argument('--disc_transformer_nhead', type=int, default=4)
    model_arch_group.add_argument('--disc_transformer_dim_feedforward', type=int, default=None)
    model_arch_group.add_argument('--disc_transformer_dropout', type=float, default=0.1)
    model_arch_group.add_argument('--disc_transformer_num_layers', type=int, default=2)
    model_arch_group.add_argument('--disc_transformer_norm_first', action='store_true', help="Use Pre-LN in TransformerEncoderLayer for DCT D.")
    # Auxiliary global stats input for Discriminator
    model_arch_group.add_argument('--disc_use_global_stats_aux', action='store_true', help="Add global mean/std of input as auxiliary features to D's final decision.")
    model_arch_group.add_argument('--disc_global_stats_mlp_hidden_dim', type=int, default=32)


    # --- Group: WuBu Stack Configurations ---
    parser.add_argument('--wubu_dropout', type=float, default=0.1, help="General dropout for WuBu layers.")
    
    wubu_s_group = parser.add_argument_group('WuBu-S (Encoder)')
    wubu_s_group.add_argument('--wubu_s_num_levels', type=int, default=2)
    wubu_s_group.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32])
    wubu_s_group.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8])
    wubu_s_group.add_argument('--wubu_s_use_rotation', action='store_true')
    wubu_s_group.add_argument('--wubu_s_phi_influence_curvature', action='store_true') 
    wubu_s_group.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true') 
    wubu_s_group.add_argument('--wubu_s_output_dim_encoder', type=int, default=128)

    wubu_g_group = parser.add_argument_group('WuBu-G (Generator)')
    wubu_g_group.add_argument('--wubu_g_num_levels', type=int, default=2)
    wubu_g_group.add_argument('--wubu_g_hyperbolic_dims', nargs='+', type=int, default=[128,256])
    wubu_g_group.add_argument('--wubu_g_initial_curvatures', nargs='+', type=float, default=[0.8,1.0])
    wubu_g_group.add_argument('--wubu_g_use_rotation', action='store_true')
    wubu_g_group.add_argument('--wubu_g_phi_influence_curvature', action='store_true') 
    wubu_g_group.add_argument('--wubu_g_phi_influence_rotation_init', action='store_true') 
    
    wubu_d_group = parser.add_argument_group('WuBu-D (Primary DCT D - if not using Transformer / fallback)')
    wubu_d_group.add_argument('--wubu_d_num_levels', type=int, default=1)
    wubu_d_group.add_argument('--wubu_d_hyperbolic_dims', nargs='+', type=int, default=[64])
    wubu_d_group.add_argument('--wubu_d_initial_curvatures', nargs='+', type=float, default=[0.7])
    wubu_d_group.add_argument('--wubu_d_use_rotation', action='store_true')
    wubu_d_group.add_argument('--wubu_d_phi_influence_curvature', action='store_true') 
    wubu_d_group.add_argument('--wubu_d_phi_influence_rotation_init', action='store_true') 
    wubu_d_group.add_argument('--wubu_d_output_dim', type=int, default=64)
    
    wubu_d_region_group = parser.add_argument_group('WuBu-D-Region (DCT D Regional Processor for Transformer)')
    wubu_d_region_group.add_argument('--wubu_d_region_num_levels', type=int, default=1)
    wubu_d_region_group.add_argument('--wubu_d_region_feature_dim', type=int, default=128)
    wubu_d_region_group.add_argument('--wubu_d_region_hyperbolic_dims', nargs='+', type=int, default=None)
    wubu_d_region_group.add_argument('--wubu_d_region_initial_curvatures', nargs='+', type=float, default=None)
    wubu_d_region_group.add_argument('--wubu_d_region_use_rotation', action='store_true')
    wubu_d_region_group.add_argument('--wubu_d_region_phi_influence_curvature', action='store_true')
    wubu_d_region_group.add_argument('--wubu_d_region_phi_influence_rotation_init', action='store_true')

    wubu_d_alt_group = parser.add_argument_group('WuBu-D-Alt (Alternative DCT Discriminator Components)')
    wubu_d_alt_group.add_argument('--wubu_d_alt_num_levels', type=int, default=None)
    wubu_d_alt_group.add_argument('--wubu_d_alt_hyperbolic_dims', nargs='+', type=int, default=None)
    wubu_d_alt_group.add_argument('--wubu_d_alt_initial_curvatures', nargs='+', type=float, default=None)
    wubu_d_alt_group.add_argument('--wubu_d_alt_use_rotation', action='store_true')
    wubu_d_alt_group.add_argument('--wubu_d_alt_phi_influence_curvature', action='store_true')
    wubu_d_alt_group.add_argument('--wubu_d_alt_phi_influence_rotation_init', action='store_true')
    wubu_d_alt_group.add_argument('--wubu_d_alt_output_dim', type=int, default=None)


    # --- Group: Q-Learning Controller (General & Lambda_KL) ---
    q_learn_group = parser.add_argument_group('Q-Learning Controller')
    q_learn_group.add_argument('--q_controller_enabled',action='store_true')
    q_learn_group.add_argument('--reset_q_controllers_on_load', action='store_true')
    q_learn_group.add_argument('--lambda_kl_update_interval', type=int, default=100)
    q_learn_group.add_argument('--min_lambda_kl_q_control', type=float, default=1e-7)
    q_learn_group.add_argument('--max_lambda_kl_q_control', type=float, default=0.2)
    q_learn_group.add_argument('--q_lkl_scale_options', nargs='+', type=float, default=[0.80, 0.90, 1.0, 1.10, 1.20])
    q_learn_group.add_argument('--q_lkl_lr_mom_probation_steps', type=int, default=None)
    q_learn_group.add_argument('--q_lkl_action_probation_steps', type=int, default=None)
    q_learn_group.add_argument('--reset_lkl_q_controller_on_load', action='store_true')
    
    # --- Group: Heuristic Interventions & Discriminator Switching ---
    heuristic_group = parser.add_argument_group('Heuristic Interventions')
    heuristic_group.add_argument('--enable_heuristic_interventions', action='store_true')
    heuristic_group.add_argument('--enable_heuristic_disc_switching', action='store_true')
    heuristic_group.add_argument('--initial_disc_type', type=str, default=None, choices=['mel', 'dct'])
    heuristic_group.add_argument('--heuristic_check_interval', type=int, default=None)
    heuristic_group.add_argument('--heuristic_short_term_history_len', type=int, default=7)
    heuristic_group.add_argument('--heuristic_trigger_count_thresh', type=int, default=2)
    
    heuristic_group.add_argument('--disc_switch_check_interval', type=int, default=50) 
    heuristic_group.add_argument('--disc_switch_min_steps_between', type=int, default=250)
    heuristic_group.add_argument('--disc_switch_problem_state_count_thresh', type=int, default=2)
    
    heuristic_group.add_argument('--heuristic_d_strong_thresh', type=float, default=0.25)
    heuristic_group.add_argument('--heuristic_d_weak_thresh', type=float, default=1.0)
    heuristic_group.add_argument('--heuristic_d_very_weak_thresh', type=float, default=1.8)
    heuristic_group.add_argument('--heuristic_g_stalled_thresh', type=float, default=1.5)
    heuristic_group.add_argument('--heuristic_g_winning_thresh', type=float, default=0.2)
    heuristic_group.add_argument('--heuristic_g_very_much_winning_thresh', type=float, default=0.05)
    heuristic_group.add_argument('--heuristic_kl_high_thresh', type=float, default=25.0)
    heuristic_group.add_argument('--heuristic_recon_stagnation_improvement_thresh_rel', type=float, default=0.001)
    heuristic_group.add_argument('--target_good_recon_thresh_heuristic', type=float, default=0.03)
    heuristic_group.add_argument('--heuristic_q_reward_stagnation_thresh', type=float, default=-0.25)
    
    heuristic_group.add_argument('--heuristic_recon_boost_factor', type=float, default=1.8)
    heuristic_group.add_argument('--lambda_feat_match_heuristic', type=float, default=0.75)
    heuristic_group.add_argument('--lambda_g_easy_win_penalty_heuristic', type=float, default=1.5)
    heuristic_group.add_argument('--g_easy_win_penalty_eps_denom', type=float, default=1e-4) 
    heuristic_group.add_argument('--max_g_easy_win_penalty_abs', type=float, default=20.0) 
    heuristic_group.add_argument('--heuristic_active_d_lr_boost_factor', type=float, default=1.8)
    heuristic_group.add_argument('--heuristic_d_q_explore_boost_epsilon', type=float, default=0.7)
    heuristic_group.add_argument('--heuristic_d_q_explore_duration', type=int, default=10)
    heuristic_group.add_argument('--heuristic_min_lambda_gan_factor', type=float, default=0.7) 
    heuristic_group.add_argument('--heuristic_max_lambda_gan_factor', type=float, default=1.3) 
    parser.add_argument('--force_start_epoch_on_load', type=int, default=None, help="Force start epoch on load.")
    parser.add_argument('--force_start_gstep_on_load', type=int, default=None, help="Force start GStep on load (use with force_start_epoch).")

    # --- Group: Logging, Sampling, Validation & Checkpointing ---
    log_group = parser.add_argument_group('Logging and Saving')
    log_group.add_argument('--log_interval',type=int, default=20)
    log_group.add_argument('--save_interval',type=int, default=500)
    log_group.add_argument('--save_epoch_interval', type=int, default=1)
    log_group.add_argument('--validation_interval_epochs', type=int, default=1) 
    log_group.add_argument('--disable_val_tqdm', action='store_true')
    log_group.add_argument('--wandb',action='store_true')
    log_group.add_argument('--wandb_project',type=str,default='WuBuSpecTransV011_Robust') 
    log_group.add_argument('--wandb_run_name',type=str,default=None)
    log_group.add_argument('--wandb_log_train_recon_interval', type=int, default=100)
    log_group.add_argument('--train_target_log_freq_multiplier', type=int, default=5)
    log_group.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=250)
    log_group.add_argument('--use_lpips_for_mel_verification', action='store_true')
    log_group.add_argument('--val_primary_metric', type=str, default="avg_val_lpips_mel",
                        choices=["avg_val_recon_dct_mse", "avg_val_mel_mse", "avg_val_psnr_mel", "avg_val_ssim_mel", "avg_val_lpips_mel"])
    log_group.add_argument('--num_val_samples_to_log', type=int, default=3)
    log_group.add_argument('--demo_num_samples', type=int, default=5)
    
    parsed_args = parser.parse_args()

    if not TORCH_DCT_AVAILABLE:
        parser.error("torch-dct library is required but not found. Please install it: 'pip install torch-dct'")

    # --- Post-parsing argument validation and defaults ---
    if isinstance(parsed_args.disc_target_final_feature_dim, list):
        if len(parsed_args.disc_target_final_feature_dim) == 1:
            # If one int provided, use it for both H and W target
            parsed_args.disc_target_final_feature_dim = [parsed_args.disc_target_final_feature_dim[0], parsed_args.disc_target_final_feature_dim[0]]
        elif len(parsed_args.disc_target_final_feature_dim) > 2: 
            parser.error("--disc_target_final_feature_dim must be 1 or 2 integers.")
    elif isinstance(parsed_args.disc_target_final_feature_dim, int):
         # This case should ideally be handled by nargs='+' making it a list, but as a fallback
         parsed_args.disc_target_final_feature_dim = [parsed_args.disc_target_final_feature_dim, parsed_args.disc_target_final_feature_dim]
    # Ensure it's a list of two, even if default [4,4] was used (nargs='+' makes it a list)
    if not (isinstance(parsed_args.disc_target_final_feature_dim, list) and len(parsed_args.disc_target_final_feature_dim) == 2):
        # This case should be rare if nargs='+' and default=[4,4] are set correctly
        # If it was a single default int not caught by above, make it a list of two
        if isinstance(parsed_args.disc_target_final_feature_dim, int):
             parsed_args.disc_target_final_feature_dim = [parsed_args.disc_target_final_feature_dim, parsed_args.disc_target_final_feature_dim]
        else: # Fallback if something else went wrong
            print(f"Warning: disc_target_final_feature_dim had unexpected value {parsed_args.disc_target_final_feature_dim}. Defaulting to [4,4].")
            parsed_args.disc_target_final_feature_dim = [4,4]


    if parsed_args.disc_dct_embed_dim is None:
        parsed_args.disc_dct_embed_dim = parsed_args.encoder_initial_tangent_dim
    
    if parsed_args.disc_transformer_dim_feedforward is None : 
        # Default for disc_transformer_dim_feedforward depends on wubu_d_region_feature_dim
        # It's best to set this after wubu_d_region_feature_dim is confirmed.
        # For now, we can use a placeholder or the default wubu_d_region_feature_dim.
        # This will be correctly resolved in _get_discriminator_configs or D init.
        # No action here, as wubu_d_region_feature_dim is defined.
        pass


    if parsed_args.heuristic_check_interval is None:
        if parsed_args.enable_heuristic_disc_switching and parsed_args.disc_switch_check_interval is not None:
            parsed_args.heuristic_check_interval = parsed_args.disc_switch_check_interval
        else:
            parsed_args.heuristic_check_interval = parsed_args.log_interval
    
    if parsed_args.enable_heuristic_disc_switching and parsed_args.initial_disc_type is None:
        parsed_args.initial_disc_type = parsed_args.disc_input_type

    # Validate WuBu stack configurations
    # These prefixes will be used by _configure_wubu_stack
    wubu_prefixes_to_validate = ["wubu_s", "wubu_g", "wubu_d", "wubu_d_region", "wubu_d_alt"]
    for prefix in wubu_prefixes_to_validate:
        num_levels_attr = f"{prefix}_num_levels"
        # Check if num_levels attribute exists and is > 0 before validating its list args
        if hasattr(parsed_args, num_levels_attr):
            num_levels_val = getattr(parsed_args, num_levels_attr)
            # For alt, num_levels can be None, in which case it defaults later or isn't built.
            if num_levels_val is not None and num_levels_val > 0:
                validate_wubu_config_for_argparse(parsed_args, prefix, parser)
            elif num_levels_val is None and "alt" in prefix: # Specifically for wubu_d_alt_num_levels if None
                # It will default later if not specified. No validation needed if explicitly None.
                pass
    
    return parsed_args



# Helper function (assumed to be defined elsewhere, content unchanged)
def validate_wubu_config_for_argparse(args_obj, prefix_str, parser_ref):
    num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0)
    if num_levels > 0:
        for suffix, attr_name_fmt in [("hyperbolic_dims", "{prefix}_hyperbolic_dims"),
                                  ("initial_curvatures", "{prefix}_initial_curvatures")]:
            attr_name = attr_name_fmt.format(prefix=prefix_str)
            val_list = getattr(args_obj, attr_name)
            
            if not isinstance(val_list, list): val_list = [val_list]
            setattr(args_obj, attr_name, val_list) # Store as list for consistent processing below

            if len(val_list) != num_levels:
                if len(val_list) == 1 and num_levels > 1:
                    setattr(args_obj, attr_name, [val_list[0]] * num_levels)
                elif not val_list: # Empty list
                    default_val = 1.0 if "curvatures" in suffix else \
                                  (max(1, getattr(args_obj, 'latent_dim', 32*num_levels)//num_levels if num_levels > 0 else 32) if "dims" in suffix else None)
                    if default_val is not None: setattr(args_obj, attr_name, [default_val] * num_levels)
                    else: parser_ref.error(f"{attr_name} empty and no clear default for {num_levels} levels.")
                else: parser_ref.error(f"{attr_name} length {len(val_list)} != num_levels {num_levels}")



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

    # These configs are primarily for WuBuSpecTransNet and AudioSpecDiscriminator if directly instantiated outside trainer
    # HybridTrainer will internally use args to configure its components.
    audio_config = {
        "sample_rate": args.sample_rate, "n_fft": args.n_fft, "hop_length": args.hop_length,
        "n_mels": args.n_mels, "fmin": args.fmin, "fmax": args.fmax,
        "segment_duration_sec": args.segment_duration_sec,
        "region_proc_size_t": args.region_proc_size_t, "region_proc_size_f": args.region_proc_size_f,
        "wubu_s_output_dim_encoder": args.wubu_s_output_dim_encoder,
        "wubu_d_output_dim": args.wubu_d_output_dim, # Used by WuBu-D if configured
        "num_time_frames_for_1s_segment": num_time_frames_for_1s_segment,
    }
    gaad_config = {
        "num_regions": args.gaad_num_regions, "decomposition_type": args.gaad_decomposition_type,
        "min_size_px": args.gaad_min_size_px
    }
    wubu_s_config_enc = _configure_wubu_stack(args, "wubu_s")
    wubu_g_config_gen = _configure_wubu_stack(args, "wubu_g")
    # wubu_d_config_disc will be determined inside HybridTrainer for primary/alt discriminators

    if am_main_process:
        current_logger_main.info(f"AudioCfg (for model):{audio_config}\nGAADCfg (for model):{gaad_config}\n"
                                 f"WuBuS_Enc (for model):{wubu_s_config_enc}\nWuBuG_Gen (for model):{wubu_g_config_gen}")

    # Model (VAE components)
    model = WuBuSpecTransNet(args, audio_config, gaad_config, wubu_s_config_enc, wubu_g_config_gen).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False) # type: ignore
        # Discriminators are watched inside HybridTrainer if heuristic switching is on, or handled by the main D watch if not.
        # For simplicity, we can remove the direct D watch here if HybridTrainer handles its internal Ds.
        # If HybridTrainer doesn't handle switching, then the old D watch is fine.
        # Given the change, let HybridTrainer's __init__ handle D watching if needed.

    if ddp_active:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # Discriminators will be DDP wrapped inside HybridTrainer if ddp_active

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
                    import soundfile as sf
                    sr_dummy = args.sample_rate; duration_dummy = 5.0
                    t_dummy = np.linspace(0, duration_dummy, int(sr_dummy * duration_dummy), endpoint=False)
                    wav_dummy = (0.3 * np.sin(2 * np.pi * 220.0 * t_dummy) +
                                 0.2 * np.sin(2 * np.pi * 440.0 * t_dummy) +
                                 0.1 * np.sin(2 * np.pi * 880.0 * t_dummy))
                    wav_dummy_norm = (wav_dummy / (np.max(np.abs(wav_dummy)) + EPS) * 0.9).astype(np.float32)
                    sf.write(str(dummy_audio_path), wav_dummy_norm, sr_dummy)
                    current_logger_main.info(f"Dummy audio created: {dummy_audio_path}")
                    audio_files_list.append(str(dummy_audio_path))
                except ImportError: current_logger_main.error("soundfile library not found. Cannot create dummy audio.")
                except Exception as e_dummy_audio: current_logger_main.error(f"Error creating dummy audio: {e_dummy_audio}", exc_info=True)
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

    train_dataset: Union[AudioSegmentDataset, SubsetRandomSampler] = full_dataset # type: ignore
    val_dataset: Optional[Union[AudioSegmentDataset, SubsetRandomSampler]] = None # type: ignore
    num_total_samples = len(full_dataset)

    val_audio_files_list = []
    if args.validation_audio_dir_path:
        val_dir_path_obj = Path(args.validation_audio_dir_path)
        if val_dir_path_obj.is_dir():
            for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]: val_audio_files_list.extend([str(p) for p in val_dir_path_obj.rglob(ext)])
        elif val_dir_path_obj.is_file(): val_audio_files_list.append(str(val_dir_path_obj))

    if val_audio_files_list:
        try:
            val_dataset_candidate = AudioSegmentDataset(audio_file_paths=val_audio_files_list, args=args, segment_duration_sec=args.segment_duration_sec, preload_to_ram=args.preload_audio_dataset_to_ram)
            if len(val_dataset_candidate) > 0: val_dataset = val_dataset_candidate; current_logger_main.info(f"Using separate validation audio dir: {args.validation_audio_dir_path}, Segments: {len(val_dataset)}")
            else: current_logger_main.warning(f"Validation audio dir {args.validation_audio_dir_path} loaded but resulted in 0 segments.")
        except Exception as e: current_logger_main.warning(f"Could not load validation dataset from '{args.validation_audio_dir_path}': {e}.")

    if val_dataset is None and args.validation_split_fraction > 0.0 and num_total_samples > 10 :
        num_val = int(num_total_samples * args.validation_split_fraction)
        num_train = num_total_samples - num_val
        if num_train > 0 and num_val > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(args.seed + rank))
            current_logger_main.info(f"Split main dataset (random): Train={len(train_dataset)}, Val={len(val_dataset)}") # type: ignore
        else:
            current_logger_main.warning("Random split for validation resulted in 0 samples for train or val. No validation set used from split."); val_dataset = None
            train_dataset = full_dataset

    if am_main_process:
        current_logger_main.info(f"Final dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}") # type: ignore

    worker_init_fn_seeded = functools.partial(seed_worker_init_fn, base_seed=args.seed, rank=rank, world_size=world_size) if args.num_workers > 0 else None
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_active else None # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler,
                              pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded,
                              drop_last=False)
    val_loader = None
    if val_dataset and len(val_dataset) > 0: # type: ignore
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None # type: ignore
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, # type: ignore
                                num_workers=args.num_workers, sampler=val_sampler,
                                pin_memory=(device.type == 'cuda'),
                                drop_last=False,
                                worker_init_fn=worker_init_fn_seeded)

    # Instantiate HybridTrainer with the corrected signature
    trainer = HybridTrainer(model=model,
                            device=device,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            args=args,
                            rank=rank,
                            world_size=world_size,
                            ddp_active=ddp_active)

    start_global_step, start_epoch = 0, 0
    if args.load_checkpoint:
        start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint)
    else: # New run without checkpoint
        if am_main_process:
            trainer.logger.info("Starting a new run from scratch (no checkpoint specified). Q-controllers will initialize fresh.")
        # Ensure Q-controllers get their initial states set up (e.g., probation if enabled by default in HAKMEMQController)
        # This is mostly handled by HAKMEMQController's own __init__ and _load_q_state_helper_inner with None data.
        # For example, ensuring their prev_loss histories are populated for the first state calculation.
        # The trainer's Q-controllers are initialized in its __init__
        # We might need an explicit call to `set_initial_losses` for them if not loading a checkpoint
        # Get references to the Q-controllers from the trainer's optimizers
        opt_gen_q_ctrl = getattr(trainer.optimizer_enc_gen, 'q_controller', None)
        opt_d_primary_q_ctrl = getattr(trainer.optimizer_disc_primary, 'q_controller', None)
        opt_d_alt_q_ctrl = getattr(trainer.optimizer_disc_alternative, 'q_controller', None)
        
        # A dummy loss dict to initialize histories for Q-controllers if starting fresh
        # This helps them form a valid first state.
        initial_dummy_losses = {
            'loss_g_total': 1.0, 'loss_g_recon': 1.0, 'loss_g_kl': 0.1, 'loss_g_adv': 0.7,
            'loss_d_total': 0.7, 'loss_d_real': 0.7, 'loss_d_fake': 0.7
        }
        if opt_gen_q_ctrl: opt_gen_q_ctrl.set_initial_losses(initial_dummy_losses, is_generator_q=True)
        if opt_d_primary_q_ctrl: opt_d_primary_q_ctrl.set_initial_losses(initial_dummy_losses, is_generator_q=False)
        if opt_d_alt_q_ctrl: opt_d_alt_q_ctrl.set_initial_losses(initial_dummy_losses, is_generator_q=False)
        if trainer.lambda_kl_q_controller:
            initial_lkl_metrics = {
                'avg_recon': 1.0, 'avg_kl_div': 0.1, 'avg_d_total': 0.7,
                'val_metric': 1.0, # Assuming lower is better for default val_primary_metric
                'current_lambda_kl_val': trainer.lambda_kl
            }
            trainer.lambda_kl_q_controller.set_initial_lambda_kl_metrics(initial_lkl_metrics)
            # Start LKL probation if starting fresh
            trainer.lambda_kl_q_controller.start_probation()


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
                    # For sampling, we need a non-DDP wrapped model if ddp was active
                    # The HybridTrainer's sample method handles getting the correct model reference.
                    generated_mels = trainer.sample(num_samples=args.demo_num_samples)
                    if generated_mels is not None and generated_mels.numel() > 0:
                        save_dir = Path(args.checkpoint_dir) / "demo_samples_mel_spectrograms_v011"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        for b_idx in range(min(args.demo_num_samples, generated_mels.shape[0])):
                            mel_to_save = (generated_mels[b_idx, 0].cpu().clamp(-1,1) + 1) / 2.0
                            save_image_path = save_dir / f"demo_mel_sample_{b_idx}_ep{trainer.current_epoch+1}.png"
                            save_image(mel_to_save, str(save_image_path))
                        current_logger_main.info(f"Saved demo Mel spectrogram images to {save_dir}")
                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            # Use the trainer's internal logging for consistency
                            trainer._log_samples_to_wandb("final_demo_mel", generated_mels, args.demo_num_samples)
                except Exception as e_demo:
                    current_logger_main.error(f"Demo Mel sampling/saving error: {e_demo}", exc_info=True)

            if args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish() # type: ignore

        if ddp_active and is_initialized():
            destroy_process_group()
        current_logger_main.info(f"Rank {rank}: {base_logger_name} (v0.1.1) script finished.")





if __name__ == "__main__":
    main()