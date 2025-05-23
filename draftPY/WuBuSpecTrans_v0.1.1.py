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
    def __init__(self, region_proc_size: Tuple[int, int]): # (Time_dim, Freq_dim)
        super().__init__()
        self.region_proc_size = region_proc_size # (T_proc, F_proc)
        self.logger = logging.getLogger("WuBuSpecTransV01.RegionExtractor")
        self.resize_transform = T.Resize((region_proc_size[1], region_proc_size[0]),
                                         interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        self.logger.info(f"Initialized RegionalSpectrogramRegionExtractor to resize regions to (T,F): {self.region_proc_size}.")

    def forward(self, mel_spectrograms: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
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
                if t1_c >= t2_c or f1_c >= f2_c:
                    region_patch = torch.zeros((C_spec, self.region_proc_size[1], self.region_proc_size[0]),
                                               device=device, dtype=original_dtype)
                else:
                    region_patch_raw = mel_spectrograms[b, :, f1_c:f2_c, t1_c:t2_c]
                    region_patch = self.resize_transform(region_patch_raw)
                batch_regions.append(region_patch)
            all_processed_regions.append(torch.stack(batch_regions))
        return torch.stack(all_processed_regions)

class RegionalFeatureEmbedder(nn.Module):
    def __init__(self, num_features_per_region: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(num_features_per_region, embed_dim)
        self.logger = logging.getLogger("WuBuSpecTransV01.RegionalFeatureEmbedder")
        self.logger.info(f"Initialized RegionalFeatureEmbedder: {num_features_per_region} features/region -> {embed_dim} embed_dim.")

    def forward(self, x_flat_features: torch.Tensor) -> torch.Tensor:
        return self.proj(x_flat_features)


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
        # self.wubu_s_config = wubu_s_config # Not stored if only used for sub-module
        self.latent_dim = latent_dim
        self.logger = logging.getLogger("WuBuSpecTransV01.Encoder")

        self.num_gaad_regions = gaad_config['num_regions']
        self.region_proc_size = (args.region_proc_size_t, args.region_proc_size_f)
        self.region_extractor = RegionalSpectrogramRegionExtractor(region_proc_size=self.region_proc_size)

        self.vae_transform_type = getattr(args, 'vae_transform_type', 'complex_dft_ri')
        
        if self.vae_transform_type == 'dct':
            if not TORCH_DCT_AVAILABLE:
                self.logger.critical("Encoder configured for DCT, but torch-dct is not available. This will fail.")
            self.num_features_per_region_transform = self.region_proc_size[0] * self.region_proc_size[1]
        elif self.vae_transform_type == 'complex_dft_ri':
            self.num_features_per_region_transform = 2 * self.region_proc_size[0] * self.region_proc_size[1]
        else:
            raise ValueError(f"Encoder: Unsupported vae_transform_type: {self.vae_transform_type}")
        self.logger.info(f"Encoder using transform: {self.vae_transform_type}. Features/region from transform: {self.num_features_per_region_transform}")

        self.num_local_perceptual_stats = 0 # getattr(args, 'encoder_num_local_perceptual_stats', 0)
        self.num_features_per_region_total = self.num_features_per_region_transform + self.num_local_perceptual_stats

        self.feature_embedder = RegionalFeatureEmbedder(
            num_features_per_region=self.num_features_per_region_total,
            embed_dim=args.encoder_initial_tangent_dim
        )

        self.wubu_s_encoder = FullyHyperbolicWuBuNestingModel(
            input_tangent_dim=args.encoder_initial_tangent_dim,
            output_tangent_dim=audio_config['wubu_s_output_dim_encoder'],
            config=wubu_s_config # Pass the pre-configured dict
        )

        self.fc_mu = nn.Linear(audio_config['wubu_s_output_dim_encoder'], self.latent_dim)
        self.fc_logvar = nn.Linear(audio_config['wubu_s_output_dim_encoder'], self.latent_dim)

        self.apply(init_weights_general)
        self.logger.info(f"AudioSpecEncoder initialized. Total features/region for embedding: {self.num_features_per_region_total}.")

    def _apply_regional_transform_and_normalize(self, region_patches: torch.Tensor) -> torch.Tensor:
        B, N_Reg, C, F_p, T_p = region_patches.shape
        patches_for_transform = region_patches.reshape(-1, F_p, T_p) 
        
        if self.vae_transform_type == 'dct':
            if dct_2d is None: 
                self.logger.error("dct_2d function is None. Cannot perform DCT. Returning zeros.")
                return torch.zeros(B * N_Reg, F_p * T_p, device=patches_for_transform.device, dtype=patches_for_transform.dtype).reshape(B, N_Reg, -1)
            
            transformed_coeffs = dct_2d(patches_for_transform.float()) 
            
            if self.args.dct_norm_type == "none": norm_coeffs = transformed_coeffs
            elif self.args.dct_norm_type == "global_scale": norm_coeffs = transformed_coeffs / self.args.dct_norm_global_scale
            elif self.args.dct_norm_type == "tanh": norm_coeffs = torch.tanh(transformed_coeffs / self.args.dct_norm_tanh_scale)
            else: norm_coeffs = transformed_coeffs / self.args.dct_norm_global_scale
            
            return norm_coeffs.reshape(B * N_Reg, -1).reshape(B, N_Reg, -1)

        elif self.vae_transform_type == 'complex_dft_ri':
            dft_complex_coeffs = torch.fft.fft2(patches_for_transform.float(), norm=getattr(self.args, 'dft_fft_norm', "backward"))
            real_parts = dft_complex_coeffs.real
            imag_parts = dft_complex_coeffs.imag
            norm_scale = getattr(self.args, 'dft_complex_norm_scale', 75.0)
            norm_real = torch.tanh(real_parts / norm_scale)
            norm_imag = torch.tanh(imag_parts / norm_scale)
            norm_dft_flat = torch.cat((norm_real.reshape(B * N_Reg, -1), norm_imag.reshape(B * N_Reg, -1)), dim=1)
            return norm_dft_flat.reshape(B, N_Reg, -1)
        else:
            raise NotImplementedError(f"Transform type {self.vae_transform_type} not implemented in encoder normalize.")

    def forward(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H_spec, W_spec = mel_spectrogram.shape
        device = mel_spectrogram.device
        dtype = mel_spectrogram.dtype
        gaad_bboxes_list = []
        for b_idx in range(B):
            spec_dims = (W_spec, H_spec)
            bboxes_current_spec = golden_subdivide_rect_fixed_n(
                spec_dims, self.num_gaad_regions, device, dtype, self.gaad_config.get('min_size_px', 5)
            )
            gaad_bboxes_list.append(bboxes_current_spec)
        gaad_bboxes_batch = torch.stack(gaad_bboxes_list)
        processed_regions = self.region_extractor(mel_spectrogram, gaad_bboxes_batch)
        regional_features_from_transform = self._apply_regional_transform_and_normalize(processed_regions)
        target_features_for_recon = regional_features_from_transform
        embedded_features = self.feature_embedder(target_features_for_recon)
        wubu_s_input = embedded_features.reshape(B * self.num_gaad_regions, -1)
        wubu_s_features_flat = self.wubu_s_encoder(wubu_s_input)
        aggregated_features = torch.mean(wubu_s_features_flat.reshape(B, self.num_gaad_regions, -1), dim=1)
        mu = self.fc_mu(aggregated_features)
        logvar = self.fc_logvar(aggregated_features)
        return mu, logvar, target_features_for_recon, gaad_bboxes_batch


class AudioSpecGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, latent_dim: int):
        super().__init__()
        self.args = args
        # self.audio_config = audio_config # Not strictly needed if all info comes from args/gaad_config
        # self.gaad_config = gaad_config
        self.latent_dim = latent_dim
        self.logger = logging.getLogger("WuBuSpecTransV01.Generator")

        self.num_gaad_regions = gaad_config['num_regions']
        self.region_proc_size = (args.region_proc_size_t, args.region_proc_size_f)
        self.vae_transform_type = getattr(args, 'vae_transform_type', 'complex_dft_ri')

        if self.vae_transform_type == 'dct':
            self.num_output_features_per_region_transform = self.region_proc_size[0] * self.region_proc_size[1]
        elif self.vae_transform_type == 'complex_dft_ri':
            self.num_output_features_per_region_transform = 2 * self.region_proc_size[0] * self.region_proc_size[1]
        else:
            raise ValueError(f"Generator: Unsupported vae_transform_type: {self.vae_transform_type}")

        self.num_local_perceptual_stats = 0 # getattr(args, 'encoder_num_local_perceptual_stats', 0)
        self.num_output_features_per_region_total = self.num_output_features_per_region_transform + self.num_local_perceptual_stats

        self.initial_gen_wubu_dim = args.encoder_initial_tangent_dim
        self.fc_expand_latent = nn.Linear(
            self.latent_dim,
            self.num_gaad_regions * self.initial_gen_wubu_dim
        )

        wubu_g_config = _configure_wubu_stack(args, "wubu_g") 
        wubu_g_output_dim_target = self.num_output_features_per_region_total

        if wubu_g_config is None or wubu_g_config["num_levels"] == 0:
            self.logger.warning("WuBu-G config is None or num_levels is 0. Generator using MLP fallback.")
            self.wubu_generator = nn.Sequential(
                nn.Linear(self.initial_gen_wubu_dim, self.initial_gen_wubu_dim * 2), nn.GELU(),
                nn.LayerNorm(self.initial_gen_wubu_dim * 2),
                nn.Linear(self.initial_gen_wubu_dim * 2, wubu_g_output_dim_target)
            )
        else:
            self.wubu_generator = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=self.initial_gen_wubu_dim,
                output_tangent_dim=wubu_g_output_dim_target,
                config=wubu_g_config
            )

        if self.vae_transform_type == 'dct' and self.args.dct_norm_type == "tanh":
            self.final_activation = nn.Tanh()
        elif self.vae_transform_type == 'complex_dft_ri':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

        self.apply(init_weights_general)
        self.logger.info(f"AudioSpecGenerator initialized for VAE transform '{self.vae_transform_type}'. Outputting {self.num_output_features_per_region_total} features/region.")

    @staticmethod
    def _unnormalize_and_reconstruct_coeffs_to_complex_dft(
            generated_regional_features: torch.Tensor, # (B, N_Reg, TotalFeatsPerReg_Flat) or (B*N_Reg, TotalFeatsPerReg_Flat)
            args_ref: argparse.Namespace,
            region_proc_size: Tuple[int, int] # (T_proc, F_proc)
        ) -> torch.Tensor: # Output: (B, N_Reg, F_p, T_p) complex or (B*N_Reg, F_p, T_p) complex
        _logger_static = logging.getLogger("WuBuSpecTransV01.Generator.StaticUtilDFT")
        
        F_p, T_p = region_proc_size[1], region_proc_size[0] # region_proc_size is (T,F)
        num_complex_elements_flat = F_p * T_p
        expected_dft_features_flat = 2 * num_complex_elements_flat
        
        original_shape = generated_regional_features.shape
        if generated_regional_features.ndim == 3: # (B, N_Reg, Feats)
            B_orig, N_Reg_orig = original_shape[0], original_shape[1]
            norm_dft_real_imag_parts_flat = generated_regional_features.reshape(-1, original_shape[-1])
            is_input_pre_batched_regions = False
        elif generated_regional_features.ndim == 2: # (B*N_Reg, Feats)
            norm_dft_real_imag_parts_flat = generated_regional_features
            is_input_pre_batched_regions = True
        else:
            _logger_static.error(f"UnnormalizeDFT: Unsupported input shape {original_shape}. Returning zeros.")
            dummy_out_shape = (original_shape[0], original_shape[1], F_p, T_p) if generated_regional_features.ndim ==3 else (original_shape[0], F_p, T_p)
            return torch.zeros(dummy_out_shape, dtype=torch.complex64, device=generated_regional_features.device)

        if norm_dft_real_imag_parts_flat.shape[-1] < expected_dft_features_flat:
            _logger_static.error(f"UnnormalizeDFT: Not enough features ({norm_dft_real_imag_parts_flat.shape[-1]}) for complex_dft_ri ({expected_dft_features_flat} needed). Returning zeros.")
            dummy_out_shape = (B_orig, N_Reg_orig, F_p, T_p) if not is_input_pre_batched_regions else (norm_dft_real_imag_parts_flat.shape[0], F_p, T_p)
            return torch.zeros(dummy_out_shape, dtype=torch.complex64, device=generated_regional_features.device)

        # Take only the expected number of features if more are provided
        norm_dft_real_imag_parts_flat = norm_dft_real_imag_parts_flat[:, :expected_dft_features_flat]
        
        norm_scale = getattr(args_ref, 'dft_complex_norm_scale', 75.0)
        input_dtype_norm = norm_dft_real_imag_parts_flat.dtype
        one_tensor = torch.tensor(1.0, dtype=input_dtype_norm, device=norm_dft_real_imag_parts_flat.device)
        
        # Clamping for atanh stability
        if input_dtype_norm in [torch.float16, torch.bfloat16]:
            eps_clamp_atanh = torch.finfo(input_dtype_norm).eps * 8 # type: ignore
            upper_b_atanh = min(one_tensor - eps_clamp_atanh, torch.tensor(1.0 - EPS, dtype=input_dtype_norm, device=norm_dft_real_imag_parts_flat.device)) # type: ignore
            lower_b_atanh = max(-one_tensor + eps_clamp_atanh, torch.tensor(-1.0 + EPS, dtype=input_dtype_norm, device=norm_dft_real_imag_parts_flat.device)) # type: ignore
        else:
            eps_for_clamp = torch.finfo(torch.float32).eps 
            upper_b_atanh = torch.tensor(1.0 - eps_for_clamp, dtype=input_dtype_norm, device=norm_dft_real_imag_parts_flat.device)
            lower_b_atanh = torch.tensor(-1.0 + eps_for_clamp, dtype=input_dtype_norm, device=norm_dft_real_imag_parts_flat.device)
        
        clamped_norm_dft = torch.clamp(norm_dft_real_imag_parts_flat, min=lower_b_atanh, max=upper_b_atanh)
        compute_dtype_atanh = torch.float32 if input_dtype_norm in [torch.float16, torch.bfloat16] else input_dtype_norm
        unscaled_dft_parts = torch.atanh(clamped_norm_dft.to(compute_dtype_atanh)) * norm_scale
        unscaled_dft_parts = unscaled_dft_parts.to(input_dtype_norm) # Convert back to original or model's precision
        
        denorm_real_flat = unscaled_dft_parts[:, :num_complex_elements_flat]
        denorm_imag_flat = unscaled_dft_parts[:, num_complex_elements_flat : expected_dft_features_flat]
        denorm_real = denorm_real_flat.reshape(-1, F_p, T_p)
        denorm_imag = denorm_imag_flat.reshape(-1, F_p, T_p)
        
        complex_dft_coeffs = torch.complex(denorm_real, denorm_imag)
        
        if is_input_pre_batched_regions:
            return complex_dft_coeffs # Shape (B*N_Reg, F_p, T_p)
        else:
            # B_orig, N_Reg_orig were defined if ndim == 3
            return complex_dft_coeffs.reshape(B_orig, N_Reg_orig, F_p, T_p) # Shape (B, N_Reg, F_p, T_p)

    @staticmethod
    def _unnormalize_dct_coeffs(
            generated_norm_dct_coeffs: torch.Tensor, # (B, N_Reg, FeaturesPerRegion_Flat) or (B*N_Reg, FeaturesPerRegion_Flat)
            args_ref: argparse.Namespace,
            region_proc_size: Tuple[int, int] # (T_proc, F_proc)
        ) -> torch.Tensor: # Output: (B, N_Reg, F_p, T_p) real or (B*N_Reg, F_p, T_p) real
        _logger_static = logging.getLogger("WuBuSpecTransV01.Generator.StaticUnnormDCT")

        F_p_actual, T_p_actual = region_proc_size[1], region_proc_size[0] # region_proc_size is (T,F)
        expected_features_flat = F_p_actual * T_p_actual
        original_shape = generated_norm_dct_coeffs.shape
        
        if generated_norm_dct_coeffs.ndim == 3: # (B, N_Reg, Feats)
            B_orig, N_Reg_orig = original_shape[0], original_shape[1]
            flat_coeffs = generated_norm_dct_coeffs.reshape(-1, original_shape[-1])
            is_input_pre_batched_regions = False
        elif generated_norm_dct_coeffs.ndim == 2: # (B*N_Reg, Feats)
            flat_coeffs = generated_norm_dct_coeffs
            is_input_pre_batched_regions = True
        else:
            _logger_static.error(f"UnnormalizeDCT: Unsupported input shape {original_shape}. Returning input.")
            return generated_norm_dct_coeffs 

        if flat_coeffs.shape[-1] < expected_features_flat:
            _logger_static.error(f"UnnormalizeDCT: Not enough features ({flat_coeffs.shape[-1]}) for DCT ({expected_features_flat} needed). Returning zeros.")
            dummy_out_shape = (B_orig, N_Reg_orig, F_p_actual, T_p_actual) if not is_input_pre_batched_regions else (flat_coeffs.shape[0], F_p_actual, T_p_actual)
            return torch.zeros(dummy_out_shape, dtype=generated_norm_dct_coeffs.dtype, device=generated_norm_dct_coeffs.device)
        
        # Take only the expected number of features
        flat_coeffs = flat_coeffs[:, :expected_features_flat]

        unnorm_coeffs_flat: torch.Tensor
        if args_ref.dct_norm_type == "none":
            unnorm_coeffs_flat = flat_coeffs
        elif args_ref.dct_norm_type == "global_scale":
            unnorm_coeffs_flat = flat_coeffs * args_ref.dct_norm_global_scale
        elif args_ref.dct_norm_type == "tanh":
            input_dtype_norm = flat_coeffs.dtype
            one_tensor = torch.tensor(1.0, dtype=input_dtype_norm, device=flat_coeffs.device)
            if input_dtype_norm in [torch.float16, torch.bfloat16]:
                eps_clamp_atanh = torch.finfo(input_dtype_norm).eps * 8 # type: ignore
                upper_b_atanh = min(one_tensor - eps_clamp_atanh, torch.tensor(1.0 - EPS, dtype=input_dtype_norm, device=flat_coeffs.device)) # type: ignore
                lower_b_atanh = max(-one_tensor + eps_clamp_atanh, torch.tensor(-1.0 + EPS, dtype=input_dtype_norm, device=flat_coeffs.device)) # type: ignore
            else:
                eps_for_clamp = torch.finfo(torch.float32).eps 
                upper_b_atanh = torch.tensor(1.0 - eps_for_clamp, dtype=input_dtype_norm, device=flat_coeffs.device)
                lower_b_atanh = torch.tensor(-1.0 + eps_for_clamp, dtype=input_dtype_norm, device=flat_coeffs.device)
            
            clamped_coeffs = torch.clamp(flat_coeffs, min=lower_b_atanh, max=upper_b_atanh)
            compute_dtype_atanh = torch.float32 if input_dtype_norm in [torch.float16, torch.bfloat16] else input_dtype_norm
            unnorm_coeffs_flat = torch.atanh(clamped_coeffs.to(compute_dtype_atanh)) * args_ref.dct_norm_tanh_scale
            unnorm_coeffs_flat = unnorm_coeffs_flat.to(input_dtype_norm)
        else: 
            _logger_static.warning(f"UnnormalizeDCT: Unknown dct_norm_type '{args_ref.dct_norm_type}'. Using features as is.")
            unnorm_coeffs_flat = flat_coeffs 
        
        if is_input_pre_batched_regions:
            return unnorm_coeffs_flat.reshape(-1, F_p_actual, T_p_actual)
        else:
            return unnorm_coeffs_flat.reshape(B_orig, N_Reg_orig, F_p_actual, T_p_actual)

    def forward(self, latent_code: torch.Tensor) -> torch.Tensor:
        B = latent_code.shape[0]
        expanded_z = self.fc_expand_latent(latent_code)
        wubu_gen_input = expanded_z.view(B * self.num_gaad_regions, self.initial_gen_wubu_dim)
        generated_flat_features = self.wubu_generator(wubu_gen_input) # Output (B*NumRegions, TotalOutputFeaturesPerRegion)
        generated_flat_features_activated = self.final_activation(generated_flat_features)
        
        # Reshape to (B, NumRegions, TotalOutputFeaturesPerRegion) for consistency
        return generated_flat_features_activated.view(B, self.num_gaad_regions, -1)


# Helper Self-Attention Module
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, k_reduction_factor=8, use_spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // k_reduction_factor)

        conv_fn = functools.partial(nn.Conv2d, kernel_size=1, padding=0, bias=False)
        self.query_conv = conv_fn(self.in_channels, self.inter_channels)
        self.key_conv = conv_fn(self.in_channels, self.inter_channels)
        self.value_conv = conv_fn(self.in_channels, self.in_channels)
        self.out_conv = conv_fn(self.in_channels, self.in_channels)

        if use_spectral_norm:
            self.query_conv = spectral_norm(self.query_conv)
            self.key_conv = spectral_norm(self.key_conv)
            self.value_conv = spectral_norm(self.value_conv)
            self.out_conv = spectral_norm(self.out_conv)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, self.inter_channels, -1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, C, -1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        return self.gamma * out + x

class _SingleScaleMelDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, disc_config: Dict, scale_index: int = 0):
        super().__init__()
        self.args = args
        self.scale_index = scale_index
        # Get SN setting from disc_config first, then general args.disc_apply_spectral_norm
        self.apply_spectral_norm = disc_config.get(f"mel_d_scale{scale_index}_apply_sn", 
                                                   disc_config.get("apply_spectral_norm", 
                                                                   getattr(args, 'disc_apply_spectral_norm', False)))
        self.logger = logging.getLogger(f"WuBuSpecTransV01.SingleMelD.Scale{scale_index}.{id(self)}")

        n_mels_effective = args.n_mels // (2**scale_index)
        n_time_effective = audio_config.get("num_time_frames_for_1s_segment", 87) // (2**scale_index)

        # Use disc_config for base_ch, max_ch etc. if they are specific to this D's overall config
        base_ch = disc_config.get("base_disc_channels", getattr(args, 'disc_base_disc_channels', 64))
        max_ch = disc_config.get("max_disc_channels", getattr(args, 'disc_max_disc_channels', 512))
        target_final_dim_config = disc_config.get("target_mel_disc_final_feature_dim", getattr(args, 'disc_target_final_feature_dim', [4,4]))

        if isinstance(target_final_dim_config, int): target_final_dim_h = target_final_dim_w = target_final_dim_config
        elif isinstance(target_final_dim_config, (list, tuple)) and len(target_final_dim_config) == 2: target_final_dim_h, target_final_dim_w = target_final_dim_config
        elif isinstance(target_final_dim_config, (list, tuple)) and len(target_final_dim_config) == 1: target_final_dim_h = target_final_dim_w = target_final_dim_config[0]
        else: target_final_dim_h = target_final_dim_w = 4

        max_downs_limit = disc_config.get("max_mel_disc_downsample_layers", getattr(args, 'max_mel_disc_downsample_layers', 5))
        
        # Check for scale-specific attention args, then general mel_d attention args
        # This makes _SingleScaleMelDiscriminator usable by different parent Ds (default MelD, WudioMelD)
        parent_prefix = disc_config.get("parent_discriminator_arg_prefix", "mel_d") # e.g. "wudio_d_mel" or "mel_d"
        
        self.use_attention_in_mel_scale = getattr(args, f"{parent_prefix}_scale{scale_index}_use_attention",
                                            getattr(args, f"{parent_prefix}_use_attention", # general for this D type
                                            getattr(args, 'use_mel_d_attention', False))) # global fallback
        self.attention_after_layer_idx = getattr(args, f"{parent_prefix}_scale{scale_index}_attention_idx",
                                            getattr(args, f"{parent_prefix}_attention_idx",
                                            getattr(args, 'mel_d_attention_idx', 2)))
        cnn_layers_list = []
        in_c = 1
        curr_h, curr_w = n_mels_effective, n_time_effective

        if curr_h <= 0 or curr_w <= 0:
            self.logger.warning(f"  SingleMelD Scale {scale_index}: Effective input dims ({curr_h}x{curr_w}) non-positive. Using Identity.")
            self.feature_extractor = nn.Identity()
            self.final_conv_in_channels = 1
            self.final_conv = nn.Identity()
            return

        if curr_h <= target_final_dim_h and curr_w <= target_final_dim_w: num_downsamples = 0
        else:
            num_downsamples = 0; temp_h, temp_w = curr_h, curr_w
            while (temp_h > target_final_dim_h or temp_w > target_final_dim_w) and num_downsamples < max_downs_limit and temp_h > 1 and temp_w > 1:
                next_h = (temp_h - 4 + 2*1)//2 + 1; next_w = (temp_w - 4 + 2*1)//2 + 1
                if (next_h<target_final_dim_h and next_w<target_final_dim_w and num_downsamples>0) or next_h<1 or next_w<1:
                    if (temp_h<=target_final_dim_h or temp_w<=target_final_dim_w and num_downsamples>0): break
                if next_h<1 or next_w<1: break
                temp_h, temp_w = next_h, next_w; num_downsamples +=1
            num_downsamples = max(0, num_downsamples)

        for i in range(num_downsamples):
            out_c = min(base_ch * (2**i), max_ch)
            conv_l = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)
            if self.apply_spectral_norm: cnn_layers_list.append(spectral_norm(conv_l))
            else: cnn_layers_list.append(conv_l)
            cnn_layers_list.append(nn.InstanceNorm2d(out_c, affine=True)); cnn_layers_list.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = out_c
            prev_curr_h, prev_curr_w = curr_h, curr_w
            curr_h = (curr_h - 4 + 2*1)//2 + 1 if curr_h > 1 else 1
            curr_w = (curr_w - 4 + 2*1)//2 + 1 if curr_w > 1 else 1
            if curr_h <= 0 or curr_w <=0 :
                curr_h, curr_w = prev_curr_h, prev_curr_w
                self.logger.warning(f"  SingleMelD Scale {scale_index} Layer {i}: Dims non-positive ({curr_h}x{curr_w}). Stopping downsampling."); break
            if self.use_attention_in_mel_scale and i == self.attention_after_layer_idx and in_c > 0:
                cnn_layers_list.append(SelfAttention2D(in_c, use_spectral_norm=self.apply_spectral_norm))

        self.feature_extractor = nn.Sequential(*cnn_layers_list) if cnn_layers_list else nn.Identity()
        self.final_conv_in_channels = in_c
        final_kernel_h = curr_h if curr_h > 0 else 1; final_kernel_w = curr_w if curr_w > 0 else 1
        final_padding_h = 0; final_padding_w = 0
        if curr_h > 1 or curr_w > 1:
             final_kernel_h=min(3,curr_h if curr_h>0 else 1); final_kernel_w=min(3,curr_w if curr_w>0 else 1)
             final_padding_h=final_kernel_h//2; final_padding_w=final_kernel_w//2
        self.final_conv = nn.Conv2d(self.final_conv_in_channels, 1, (final_kernel_h, final_kernel_w), 1, (final_padding_h, final_padding_w), bias=True)
        if self.apply_spectral_norm: self.final_conv = spectral_norm(self.final_conv)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(self.feature_extractor, nn.Identity) and isinstance(self.final_conv, nn.Identity):
            dummy_logits = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            dummy_features = torch.zeros(x.size(0),1,1,1,device=x.device,dtype=x.dtype) if x.ndim==4 else torch.zeros(x.size(0),1,device=x.device,dtype=x.dtype)
            return (dummy_logits, dummy_features) if return_features else dummy_logits
        features = self.feature_extractor(x)
        patch_logits_map = self.final_conv(features)
        logits = torch.mean(patch_logits_map, dim=[2,3], keepdim=False) if (patch_logits_map.shape[2]>1 or patch_logits_map.shape[3]>1) else patch_logits_map
        return (logits, features) if return_features else logits

# In GlobalWuBuDCTDiscriminator (now effectively GlobalWuBuFeatureDiscriminator)
class GlobalWuBuDCTDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        # self.gaad_config = gaad_config # Not directly used if features are already regional
        self.logger = logging.getLogger(f"WuBuSpecTransV01.GlobalWuBuFeatureD.{id(self)}")
        
        self.effective_input_type_for_trainer = "dft_features_regional" # Explicitly set

        self.num_gaad_regions = gaad_config['num_regions']
        self.region_proc_size_t = args.region_proc_size_t # Needed to calc num_features_per_region_input
        self.region_proc_size_f = args.region_proc_size_f

        self.vae_transform_type_g_output = getattr(args, 'vae_transform_type', 'complex_dft_ri')
        if self.vae_transform_type_g_output == 'dct':
            features_per_region_from_g_transform = self.region_proc_size_t * self.region_proc_size_f
        elif self.vae_transform_type_g_output == 'complex_dft_ri':
            features_per_region_from_g_transform = 2 * self.region_proc_size_t * self.region_proc_size_f
        else:
            raise ValueError(f"GlobalWuBuFeatureD: Unknown vae_transform_type '{self.vae_transform_type_g_output}' for G's output.")
        
        num_local_stats_from_g = 0 # getattr(args, 'encoder_num_local_perceptual_stats', 0) # Assuming G outputs these if E did
        self.num_features_per_region_input = features_per_region_from_g_transform + num_local_stats_from_g
        self.total_input_feature_dim = self.num_gaad_regions * self.num_features_per_region_input

        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", getattr(args, 'disc_apply_spectral_norm', False))
        self.use_global_stats_aux = disc_config.get("disc_use_global_stats_aux_dct_global_wubu", 
                                                    getattr(args, 'disc_use_global_stats_aux_dct_global_wubu', False))

        self.global_wubu_input_tangent_dim = getattr(args, 'dct_global_wubu_d_input_tangent_dim', 512)
        
        current_projection_input_dim = self.total_input_feature_dim
        if self.use_global_stats_aux:
            self.num_global_stats_outputs = 2 # Mean, Std of (mean regional features)
            stats_mlp_hidden_dim_key = "disc_global_stats_mlp_hidden_dim_dct_global_wubu"
            self.global_stats_mlp_hidden_dim = disc_config.get(stats_mlp_hidden_dim_key, getattr(args, stats_mlp_hidden_dim_key, 64))
            if self.num_global_stats_outputs > 0 and self.global_stats_mlp_hidden_dim > 0:
                self.global_stats_mlp = nn.Sequential(
                    nn.Linear(self.num_global_stats_outputs, self.global_stats_mlp_hidden_dim), nn.LeakyReLU(0.2, True),
                    nn.Linear(self.global_stats_mlp_hidden_dim, self.global_stats_mlp_hidden_dim)
                )
                if self.apply_spectral_norm:
                    if isinstance(self.global_stats_mlp[0], nn.Linear): self.global_stats_mlp[0] = spectral_norm(self.global_stats_mlp[0])
                    if isinstance(self.global_stats_mlp[2], nn.Linear): self.global_stats_mlp[2] = spectral_norm(self.global_stats_mlp[2])
                current_projection_input_dim += self.global_stats_mlp_hidden_dim
            else: self.use_global_stats_aux = False; self.global_stats_mlp = None
        else:
            self.global_stats_mlp = None

        if current_projection_input_dim > 0 and self.global_wubu_input_tangent_dim > 0:
            self.initial_projection = nn.Linear(current_projection_input_dim, self.global_wubu_input_tangent_dim)
            self.initial_layernorm = nn.LayerNorm(self.global_wubu_input_tangent_dim)
        else:
            self.logger.warning("GlobalWuBuFeatureD: Initial projection has zero input/output dim. Using Identity.")
            self.initial_projection = nn.Identity(); self.initial_layernorm = nn.Identity()

        wubu_config = _configure_wubu_stack(args, "wubu_d_global")
        self.wubu_output_dim = getattr(args, 'dct_global_wubu_d_output_feature_dim', 256) # Used if MLP fallback

        if wubu_config and wubu_config.get("num_levels", 0) > 0 and self.global_wubu_input_tangent_dim > 0 :
            if wubu_config.get('hyperbolic_dims') and wubu_config['hyperbolic_dims'][-1] > 0:
                self.wubu_output_dim = wubu_config['hyperbolic_dims'][-1] # Actual output from WuBu
            self.wubu_stack = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=self.global_wubu_input_tangent_dim,
                output_tangent_dim=self.wubu_output_dim,
                config=wubu_config
            )
            self.logger.info(f"GlobalWuBuFeatureD: WuBu stack active ({wubu_config.get('num_levels')} levels). Output dim: {self.wubu_output_dim}")
        else:
            self.logger.warning(f"GlobalWuBuFeatureD: WuBu stack not configured or input_tangent_dim is 0. Using MLP.")
            if self.global_wubu_input_tangent_dim > 0 and self.wubu_output_dim > 0:
                self.wubu_stack = nn.Sequential(
                    nn.Linear(self.global_wubu_input_tangent_dim, self.global_wubu_input_tangent_dim * 2),
                    nn.LeakyReLU(0.2, True), nn.LayerNorm(self.global_wubu_input_tangent_dim * 2),
                    nn.Linear(self.global_wubu_input_tangent_dim * 2, self.wubu_output_dim))
            else: self.wubu_stack = nn.Identity()

        if self.wubu_output_dim > 0:
            self.final_decision_layer = nn.Linear(self.wubu_output_dim, 1)
            if self.apply_spectral_norm: self.final_decision_layer = spectral_norm(self.final_decision_layer)
        else:
            self.logger.error("GlobalWuBuFeatureD: final_decision_layer input dim is 0. Using Identity.")
            self.final_decision_layer = nn.Identity()
        
        self.apply(init_weights_general)
        self.logger.info(f"GlobalWuBuFeatureD initialized. Expects regional features. Total input features (before global stats projection): {self.total_input_feature_dim}")

    def _calculate_global_feature_stats(self, regional_features: torch.Tensor) -> torch.Tensor:
        # regional_features: (B, NumRegions, FeaturesPerRegion)
        if regional_features.numel() == 0: return torch.zeros(regional_features.shape[0], 2, device=regional_features.device, dtype=regional_features.dtype)
        
        # Calculate mean across regions first, then mean/std of those feature vectors
        mean_regional_features = torch.mean(regional_features, dim=1) # (B, FeaturesPerRegion)
        
        mean_stat = torch.mean(mean_regional_features, dim=1) # (B,)
        std_stat = torch.std(mean_regional_features, dim=1)   # (B,)
            
        std_stat = torch.max(std_stat, torch.tensor(EPS, device=std_stat.device, dtype=std_stat.dtype))
        return torch.stack([mean_stat, std_stat], dim=-1) # (B, 2)

    def forward(self, regional_features_input: torch.Tensor, # (B, NumRegions, FeaturesPerRegion)
                raw_audio_stats_input: Optional[torch.Tensor] = None, # Not used by this D, but kept for consistent signature with wrapper
                return_features: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = regional_features_input.shape[0]
        if regional_features_input.numel() == 0:
            dummy_logits = torch.zeros(B, device=regional_features_input.device, dtype=regional_features_input.dtype)
            dummy_features_ret = torch.zeros(B, self.wubu_output_dim if self.wubu_output_dim > 0 else 1, device=regional_features_input.device,dtype=regional_features_input.dtype)
            return (dummy_logits, dummy_features_ret) if return_features else dummy_logits

        flat_all_batch_features = regional_features_input.reshape(B, -1)
        
        current_input_to_projection = flat_all_batch_features
        if self.use_global_stats_aux and self.global_stats_mlp is not None:
            stats = self._calculate_global_feature_stats(regional_features_input.detach())
            projected_stats = self.global_stats_mlp(stats)
            current_input_to_projection = torch.cat([flat_all_batch_features, projected_stats], dim=-1)

        projected_tangent = self.initial_projection(current_input_to_projection)
        projected_tangent_norm = self.initial_layernorm(projected_tangent)
        
        wubu_out_features = self.wubu_stack(projected_tangent_norm)
        
        if isinstance(self.final_decision_layer, nn.Identity):
            logits = torch.mean(wubu_out_features, dim=-1) if wubu_out_features.numel() > 0 else torch.zeros(B, device=wubu_out_features.device, dtype=wubu_out_features.dtype)
        else:
            logits = self.final_decision_layer(wubu_out_features).squeeze(-1)

        return (logits, wubu_out_features) if return_features else logits



class MultiModalWudioMelDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.audio_config = audio_config
        self.logger = logging.getLogger(f"WuBuSpecTransV01.MultiModalWudioMel_D.{id(self)}")
        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", getattr(args, 'disc_apply_spectral_norm', False))
        self.effective_input_type_for_trainer = "assembled_mel_and_raw_audio_stats"
        self.architecture_variant = "multi_modal_wudio_mel" # Store for clarity

        self.fusion_method = getattr(args, 'wudio_d_fusion_method', 'concat_linear')
        self.final_decision_layer: Optional[nn.Module] = None
        total_fused_feature_dim = 0
        self.num_active_heads_for_avg_logits = 0

        # Head 1: Mel CNN
        self.use_mel_cnn_head = getattr(args, 'wudio_d_use_mel_cnn_head', True)
        self.mel_cnn_backbone: Optional[Union[nn.ModuleList, _SingleScaleMelDiscriminator]] = None
        self.mel_cnn_feature_projector: Optional[nn.Module] = None
        self.mel_cnn_feat_dim_for_fusion = 0
        self.mel_cnn_logit_layer_for_avg: Optional[nn.Module] = None

        if self.use_mel_cnn_head:
            msd_num_scales = getattr(args, 'wudio_d_mel_msd_num_scales', args.mel_d_msd_num_scales)
            msd_share_weights = getattr(args, 'wudio_d_mel_msd_share_weights', args.mel_d_msd_share_weights)
            
            mel_cnn_sub_disc_config = {
                "apply_spectral_norm": self.apply_spectral_norm,
                "base_disc_channels": disc_config.get("base_disc_channels", args.disc_base_disc_channels), # Use from passed disc_config first
                "max_disc_channels": disc_config.get("max_disc_channels", args.disc_max_disc_channels),
                "target_mel_disc_final_feature_dim": disc_config.get("target_mel_disc_final_feature_dim", args.disc_target_final_feature_dim),
                "max_mel_disc_downsample_layers": disc_config.get("max_mel_disc_downsample_layers", args.max_mel_disc_downsample_layers),
                "parent_discriminator_arg_prefix": "wudio_d_mel"
            }

            if msd_num_scales > 1:
                self.mel_cnn_backbone = nn.ModuleList()
                first_scale_module_for_ref = _SingleScaleMelDiscriminator(args, audio_config, mel_cnn_sub_disc_config.copy(), scale_index=0)
                if msd_share_weights: self.mel_cnn_backbone = nn.ModuleList([first_scale_module_for_ref] * msd_num_scales)
                else:
                    self.mel_cnn_backbone.append(first_scale_module_for_ref)
                    for i in range(1, msd_num_scales): self.mel_cnn_backbone.append(_SingleScaleMelDiscriminator(args, audio_config, mel_cnn_sub_disc_config.copy(), scale_index=i))
            else:
                self.mel_cnn_backbone = _SingleScaleMelDiscriminator(args, audio_config, mel_cnn_sub_disc_config.copy(), scale_index=0)
            
            example_first_scale_module = self.mel_cnn_backbone[0] if isinstance(self.mel_cnn_backbone, nn.ModuleList) else self.mel_cnn_backbone
            internal_feat_dim = example_first_scale_module.final_conv_in_channels if hasattr(example_first_scale_module, 'final_conv_in_channels') and example_first_scale_module.final_conv_in_channels > 0 else 1
            
            self.mel_cnn_feat_dim_for_fusion = getattr(args, 'wudio_d_mel_cnn_output_feat_dim_for_fusion', 256)
            if internal_feat_dim > 0 and self.mel_cnn_feat_dim_for_fusion > 0:
                self.mel_cnn_feature_projector = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
                    nn.Linear(internal_feat_dim, self.mel_cnn_feat_dim_for_fusion), nn.LeakyReLU(0.2, True))
                if self.apply_spectral_norm and isinstance(self.mel_cnn_feature_projector[2], nn.Linear):
                     self.mel_cnn_feature_projector[2] = spectral_norm(self.mel_cnn_feature_projector[2])
                total_fused_feature_dim += self.mel_cnn_feat_dim_for_fusion
            elif internal_feat_dim > 0 :
                self.mel_cnn_feature_projector = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
                total_fused_feature_dim += internal_feat_dim
                self.mel_cnn_feat_dim_for_fusion = internal_feat_dim
            
            if self.fusion_method == 'average_logits':
                current_feat_dim_for_logit = self.mel_cnn_feat_dim_for_fusion if self.mel_cnn_feat_dim_for_fusion > 0 else internal_feat_dim
                if current_feat_dim_for_logit > 0:
                    self.mel_cnn_logit_layer_for_avg = nn.Linear(current_feat_dim_for_logit, 1)
                    if self.apply_spectral_norm: self.mel_cnn_logit_layer_for_avg = spectral_norm(self.mel_cnn_logit_layer_for_avg)
                    self.num_active_heads_for_avg_logits += 1
                else: self.logger.warning("WudioMelD: Mel CNN head has no features for avg_logits fusion.")
        
        # Head 2: Global Audio Stats MLP
        self.use_global_audio_stats_head = getattr(args, 'wudio_d_use_global_audio_stats_head', False)
        self.global_audio_stats_processor: Optional[nn.Module] = None
        self.global_audio_stats_feat_dim_for_fusion = 0
        self.global_audio_stats_logit_layer: Optional[nn.Module] = None

        if self.use_global_audio_stats_head:
            num_stats_features = getattr(args, 'wudio_d_num_global_audio_stats', 2) 
            mlp_hidden = getattr(args, 'wudio_d_global_audio_stats_mlp_hidden', 64)
            self.global_audio_stats_feat_dim_for_fusion = getattr(args, 'wudio_d_global_audio_stats_feat_dim_for_fusion', 32)
            if num_stats_features > 0 and self.global_audio_stats_feat_dim_for_fusion > 0 and mlp_hidden > 0:
                self.global_audio_stats_processor = nn.Sequential(
                    nn.Linear(num_stats_features, mlp_hidden), nn.LeakyReLU(0.2, True),
                    nn.Linear(mlp_hidden, self.global_audio_stats_feat_dim_for_fusion), nn.LeakyReLU(0.2, True))
                if self.apply_spectral_norm:
                    if isinstance(self.global_audio_stats_processor[0], nn.Linear): self.global_audio_stats_processor[0] = spectral_norm(self.global_audio_stats_processor[0])
                    if isinstance(self.global_audio_stats_processor[2], nn.Linear): self.global_audio_stats_processor[2] = spectral_norm(self.global_audio_stats_processor[2])
                total_fused_feature_dim += self.global_audio_stats_feat_dim_for_fusion
                if self.fusion_method == 'average_logits':
                    self.global_audio_stats_logit_layer = nn.Linear(self.global_audio_stats_feat_dim_for_fusion, 1)
                    if self.apply_spectral_norm: self.global_audio_stats_logit_layer = spectral_norm(self.global_audio_stats_logit_layer)
                    self.num_active_heads_for_avg_logits += 1
            else: self.use_global_audio_stats_head = False

        # Fusion Layer
        if self.fusion_method == 'concat_linear':
            if total_fused_feature_dim > 0:
                fusion_hidden_dim = getattr(args, 'wudio_d_fusion_mlp_hidden_dim', total_fused_feature_dim // 2 if total_fused_feature_dim > 1 else 0) # Ensure hidden is smaller or 0
                if fusion_hidden_dim > 0 and total_fused_feature_dim > fusion_hidden_dim :
                     self.final_decision_layer = nn.Sequential(
                         nn.Linear(total_fused_feature_dim, fusion_hidden_dim), nn.LeakyReLU(0.2, True),
                         nn.Linear(fusion_hidden_dim, 1))
                     if self.apply_spectral_norm:
                         if isinstance(self.final_decision_layer[0], nn.Linear): self.final_decision_layer[0] = spectral_norm(self.final_decision_layer[0])
                         if isinstance(self.final_decision_layer[2], nn.Linear): self.final_decision_layer[2] = spectral_norm(self.final_decision_layer[2])
                else:
                    self.final_decision_layer = nn.Linear(total_fused_feature_dim, 1)
                    if self.apply_spectral_norm: self.final_decision_layer = spectral_norm(self.final_decision_layer)
            else:
                self.logger.error("WudioMelD: concat_linear, but total_fused_feature_dim=0. D is broken.")
                self.final_decision_layer = nn.Identity()
        elif self.fusion_method == 'average_logits':
            if self.num_active_heads_for_avg_logits == 0: self.logger.error("WudioMelD: average_logits, but no active heads. D is broken.")
            self.final_decision_layer = None
        
        self.apply(init_weights_general)
        self.logger.info(f"MultiModalWudioMelD initialized. Fusion: {self.fusion_method}. MelCNN: {self.use_mel_cnn_head}, GlobalStats: {self.use_global_audio_stats_head}.")

    def forward(self, assembled_mel: torch.Tensor,
                raw_audio_stats_input: Optional[torch.Tensor] = None,
                return_features: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = assembled_mel.size(0)
        head_logits_list = []
        head_features_for_concat_list = []
        main_features_for_matching: Optional[torch.Tensor] = None 

        if self.use_mel_cnn_head and self.mel_cnn_backbone is not None:
            scale_cnn_backbone_features_list = []
            current_mel_scale = assembled_mel
            backbone_modules = self.mel_cnn_backbone if isinstance(self.mel_cnn_backbone, nn.ModuleList) else [self.mel_cnn_backbone]
            
            for i, mel_cnn_module in enumerate(backbone_modules):
                _, cnn_backbone_feats_this_scale = mel_cnn_module(current_mel_scale, return_features=True)
                scale_cnn_backbone_features_list.append(cnn_backbone_feats_this_scale)
                if i < len(backbone_modules) - 1: current_mel_scale = F.avg_pool2d(current_mel_scale, 3, 2, 1, count_include_pad=False)
            
            if scale_cnn_backbone_features_list:
                main_features_for_matching = scale_cnn_backbone_features_list[0] # Highest-res backbone features
                avg_cnn_backbone_features = torch.stack(scale_cnn_backbone_features_list, dim=0).mean(dim=0)
                
                processed_mel_head_feats: Optional[torch.Tensor] = None
                if self.mel_cnn_feature_projector is not None: # Projector handles pooling and flattening
                    processed_mel_head_feats = self.mel_cnn_feature_projector(avg_cnn_backbone_features)
                
                if self.fusion_method == 'concat_linear' and processed_mel_head_feats is not None:
                    head_features_for_concat_list.append(processed_mel_head_feats)
                elif self.fusion_method == 'average_logits' and self.mel_cnn_logit_layer_for_avg is not None:
                    if processed_mel_head_feats is not None: # Logits from projected features
                        mel_cnn_logits = self.mel_cnn_logit_layer_for_avg(processed_mel_head_feats)
                        head_logits_list.append(mel_cnn_logits.squeeze(-1))
        
        if self.use_global_audio_stats_head and self.global_audio_stats_processor is not None and raw_audio_stats_input is not None:
            expected_stats_shape = getattr(self.args, 'wudio_d_num_global_audio_stats', 2)
            if raw_audio_stats_input.shape[0] == B and raw_audio_stats_input.shape[1] == expected_stats_shape :
                stats_feats_processed = self.global_audio_stats_processor(raw_audio_stats_input)
                if self.fusion_method == 'concat_linear': head_features_for_concat_list.append(stats_feats_processed)
                elif self.fusion_method == 'average_logits' and self.global_audio_stats_logit_layer is not None:
                    stats_logits = self.global_audio_stats_logit_layer(stats_feats_processed).squeeze(-1)
                    head_logits_list.append(stats_logits)
            elif self.fusion_method == 'average_logits': # Add zeros if stats head was expected but input failed
                 head_logits_list.append(torch.zeros(B, device=assembled_mel.device, dtype=assembled_mel.dtype))

        final_logits: torch.Tensor
        if self.fusion_method == 'concat_linear':
            if head_features_for_concat_list and self.final_decision_layer and not isinstance(self.final_decision_layer, nn.Identity):
                fused_features = torch.cat(head_features_for_concat_list, dim=-1) if len(head_features_for_concat_list) > 1 else head_features_for_concat_list[0]
                final_logits = self.final_decision_layer(fused_features).squeeze(-1)
            else: final_logits = torch.zeros(B, device=assembled_mel.device, dtype=assembled_mel.dtype)
        elif self.fusion_method == 'average_logits':
            if head_logits_list and self.num_active_heads_for_avg_logits > 0:
                final_logits = torch.stack(head_logits_list, dim=0).mean(dim=0)
            else: final_logits = torch.zeros(B, device=assembled_mel.device, dtype=assembled_mel.dtype)
        else: final_logits = torch.zeros(B, device=assembled_mel.device, dtype=assembled_mel.dtype)

        if return_features:
            if main_features_for_matching is None:
                placeholder_features = final_logits.detach().clone()
                main_features_for_matching = placeholder_features.unsqueeze(-1).unsqueeze(-1) if placeholder_features.ndim == 1 else placeholder_features
            return final_logits, main_features_for_matching
        return final_logits

# --- Refactored AudioSpecDiscriminator (Wrapper/Hook) ---
class AudioSpecDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.disc_config_orig = disc_config.copy()

        self.input_type_configured = disc_config.get("input_type", "mel")
        self.architecture_variant = disc_config.get("architecture_variant", "default")

        self.logger = logging.getLogger(f"WuBuSpecTransV01.DiscriminatorWrapper.{self.input_type_configured}_{self.architecture_variant}.{id(self)}")
        self.actual_discriminator_module: Optional[nn.Module] = None
        self.effective_input_type_for_trainer: str = "unknown"

        if self.architecture_variant == "global_wubu_dct": # No input_type_configured check needed, variant implies input.
            self.logger.info(f"Wrapper: Instantiating GlobalWuBuDCTDiscriminator.")
            self.actual_discriminator_module = GlobalWuBuDCTDiscriminator(args, audio_config, gaad_config, disc_config)
        elif self.architecture_variant == "multi_modal_wudio_mel":
            self.logger.info(f"Wrapper: Instantiating MultiModalWudioMelDiscriminator.")
            self.actual_discriminator_module = MultiModalWudioMelDiscriminator(args, audio_config, gaad_config, disc_config)
        elif self.architecture_variant == "default":
            self.logger.info(f"Wrapper: Using 'default' D logic for input_type '{self.input_type_configured}'.")
            self._init_default_discriminator(args, audio_config, gaad_config, disc_config)
            # When default, 'self' is the discriminator. actual_discriminator_module remains None.
        else:
            # This case should ideally not be reached if architecture_variant is validated by argparser choices.
            # But as a fallback:
            self.logger.error(f"Wrapper: Unknown architecture_variant '{self.architecture_variant}'. Falling back to default D logic for input_type '{self.input_type_configured}'.")
            self._init_default_discriminator(args, audio_config, gaad_config, disc_config)


        if self.actual_discriminator_module is not None and hasattr(self.actual_discriminator_module, 'effective_input_type_for_trainer'):
            self.effective_input_type_for_trainer = self.actual_discriminator_module.effective_input_type_for_trainer
        elif self.architecture_variant == "default":
            # For the "default" D, the effective input type depends on its self.input_type
            # It doesn't use "assembled_mel_and_raw_audio_stats". If it's "mel", it's "assembled_mel".
            # If it's "dct", it's for regional features (e.g., "dct_features_regional" or "dft_features_regional"
            # depending on VAE output). Let's use a more generic term if it's regional features.
            if self.input_type_configured == "mel":
                self.effective_input_type_for_trainer = "assembled_mel"
            elif self.input_type_configured == "dct": # This "dct" means it processes feature vectors.
                # The features could be DCT or DFT based on VAE.
                # For simplicity, let's align with how GlobalWuBuDCTDiscriminator signals its input.
                self.effective_input_type_for_trainer = f"{getattr(args, 'vae_transform_type', 'complex_dft_ri').split('_')[0]}_features_regional"
            else:
                self.effective_input_type_for_trainer = "unknown_default_type"
        else: # Fallback if specialized D doesn't set it (should not happen)
            self.logger.warning(f"Effective input type not set by specialized D '{self.architecture_variant}', defaulting based on input_type_configured '{self.input_type_configured}'.")
            self.effective_input_type_for_trainer = f"{self.input_type_configured}_features_regional" if self.input_type_configured == "dct" else "assembled_mel"


        active_module_name = self.actual_discriminator_module.__class__.__name__ if self.actual_discriminator_module else self.__class__.__name__ + " (Default Logic)"
        self.logger.info(f"Discriminator Wrapper: Active D Module: {active_module_name}, Effective input for trainer: '{self.effective_input_type_for_trainer}'")

    def _init_default_discriminator(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict, disc_config: Dict):
        self.input_type = disc_config.get("input_type", "mel") # This refers to the input *type* for the default D architecture
        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", getattr(args, 'disc_apply_spectral_norm', False))
        
        # Note: The 'default' D's global_stats_aux uses 'disc_use_global_stats_aux'
        # This is different from 'disc_use_global_stats_aux_dct_global_wubu' used by GlobalWuBuDCTDiscriminator
        self.use_global_stats_aux_input = getattr(args, 'disc_use_global_stats_aux', False)

        self.num_gaad_regions = gaad_config['num_regions']
        self.region_proc_size_t = args.region_proc_size_t
        self.region_proc_size_f = args.region_proc_size_f
        
        # num_features_per_region_input for default D's "dct" mode
        # This depends on the VAE's output type.
        if getattr(args, 'vae_transform_type', 'complex_dft_ri') == 'dct':
            self.num_features_per_region_for_default_dct_d = self.region_proc_size_t * self.region_proc_size_f
        else: # complex_dft_ri
            self.num_features_per_region_for_default_dct_d = 2 * self.region_proc_size_t * self.region_proc_size_f
        
        self.feature_extractor_module: nn.Module
        self.final_decision_layer: Optional[nn.Module] = None
        self.global_stats_mlp: Optional[nn.Module] = None
        self.global_pool_for_mel_d: Optional[nn.Module] = None

        if self.use_global_stats_aux_input:
            self.num_global_stats = 2 # Assuming mean, std for now for the default D's aux input
            self.global_stats_mlp_hidden_dim = getattr(args, 'disc_global_stats_mlp_hidden_dim', 32)
            if self.num_global_stats > 0 and self.global_stats_mlp_hidden_dim > 0:
                self.global_stats_mlp = nn.Sequential(
                    nn.Linear(self.num_global_stats, self.global_stats_mlp_hidden_dim), nn.LeakyReLU(0.2, True),
                    nn.Linear(self.global_stats_mlp_hidden_dim, self.global_stats_mlp_hidden_dim)
                )
                if self.apply_spectral_norm:
                    if isinstance(self.global_stats_mlp[0], nn.Linear): self.global_stats_mlp[0] = spectral_norm(self.global_stats_mlp[0])
                    if isinstance(self.global_stats_mlp[2], nn.Linear): self.global_stats_mlp[2] = spectral_norm(self.global_stats_mlp[2])
            else: self.use_global_stats_aux_input = False 

        if self.input_type == "dct": # "dct" here means it processes regional features from VAE
            self.dct_coeff_embed_dim = getattr(args, 'disc_dct_embed_dim', args.encoder_initial_tangent_dim)
            # The input to DCTCoeffEmbed should match the actual number of features from VAE (DCT or DFT)
            self.dct_coeff_embed_disc = DCTCoeffEmbed(self.num_features_per_region_for_default_dct_d, self.dct_coeff_embed_dim) 
            
            wubu_d_region_config = _configure_wubu_stack(args, "wubu_d_region")
            self.wubu_d_region_feature_dim = getattr(args, 'wubu_d_region_feature_dim', 128)
            if wubu_d_region_config and wubu_d_region_config.get("num_levels", 0) > 0:
                if wubu_d_region_config.get('hyperbolic_dims') and wubu_d_region_config['hyperbolic_dims'][-1] > 0:
                     self.wubu_d_region_feature_dim = wubu_d_region_config['hyperbolic_dims'][-1]
                self.feature_extractor_module = FullyHyperbolicWuBuNestingModel(
                    self.dct_coeff_embed_dim, self.wubu_d_region_feature_dim, wubu_d_region_config)
            else:
                self.feature_extractor_module = nn.Sequential(
                    nn.Linear(self.dct_coeff_embed_dim, self.wubu_d_region_feature_dim*2), nn.LeakyReLU(0.2,True),
                    nn.Linear(self.wubu_d_region_feature_dim*2, self.wubu_d_region_feature_dim), nn.LayerNorm(self.wubu_d_region_feature_dim))
            self.use_region_pos_embed = getattr(args, 'disc_dct_use_pos_embed', True)
            if self.use_region_pos_embed: self.region_pos_embed = nn.Parameter(torch.randn(1, self.num_gaad_regions, self.wubu_d_region_feature_dim))
            self.use_cls_token_dct_d = getattr(args, 'disc_dct_use_cls_token', True)
            if self.use_cls_token_dct_d:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.wubu_d_region_feature_dim))
                if self.use_region_pos_embed: self.region_pos_embed_eff = nn.Parameter(torch.randn(1, self.num_gaad_regions+1, self.wubu_d_region_feature_dim))
                else: self.region_pos_embed_eff = None
            elif self.use_region_pos_embed: self.region_pos_embed_eff = self.region_pos_embed
            else: self.region_pos_embed_eff = None
            tf_d_model=self.wubu_d_region_feature_dim
            tf_enc_layer = nn.TransformerEncoderLayer(tf_d_model, getattr(args,'disc_transformer_nhead',4), getattr(args,'disc_transformer_dim_feedforward',tf_d_model*4), getattr(args,'disc_transformer_dropout',0.1), batch_first=True, norm_first=getattr(args,'disc_transformer_norm_first',True))
            self.context_transformer = nn.TransformerEncoder(tf_enc_layer, getattr(args,'disc_transformer_num_layers',2))
            final_dec_in_dim = tf_d_model
            if self.use_global_stats_aux_input and self.global_stats_mlp: final_dec_in_dim += self.global_stats_mlp_hidden_dim 
            self.final_decision_layer = nn.Linear(final_dec_in_dim,1)
        elif self.input_type == "mel":
            self.msd_num_scales = getattr(args, 'mel_d_msd_num_scales', 1)
            sub_mel_disc_config = disc_config.copy()
            sub_mel_disc_config["parent_discriminator_arg_prefix"] = "mel_d" 

            if self.msd_num_scales > 1:
                share_weights = getattr(args, 'mel_d_msd_share_weights', False)
                if share_weights:
                    shared_d_instance = _SingleScaleMelDiscriminator(args, audio_config, sub_mel_disc_config, 0)
                    self.feature_extractor_module = nn.ModuleList([shared_d_instance] * self.msd_num_scales)
                else:
                    self.feature_extractor_module = nn.ModuleList([_SingleScaleMelDiscriminator(args, audio_config, sub_mel_disc_config, i) for i in range(self.msd_num_scales)])
                if self.use_global_stats_aux_input and self.global_stats_mlp: self.final_decision_layer = nn.Linear(1 + self.global_stats_mlp_hidden_dim, 1) 
                else: self.final_decision_layer = None # Logits are averaged from scales
            else: # Single Scale Mel Discriminator
                self.feature_extractor_module = _SingleScaleMelDiscriminator(args, audio_config, sub_mel_disc_config, 0)
                if self.use_global_stats_aux_input and self.global_stats_mlp:
                    self.global_pool_for_mel_d = nn.AdaptiveAvgPool2d(1)
                    # Get final_conv_in_channels from the _SingleScaleMelDiscriminator instance
                    ssmd_instance = self.feature_extractor_module
                    cnn_out_ch = ssmd_instance.final_conv_in_channels if hasattr(ssmd_instance, 'final_conv_in_channels') else 1 # Fallback
                    self.final_decision_layer = nn.Linear(cnn_out_ch + self.global_stats_mlp_hidden_dim, 1) 
                else: self.final_decision_layer = None # Logit comes directly from _SingleScaleMelDiscriminator's final_conv
        else: raise ValueError(f"Default D init: Unsupported input_type {self.input_type}")
        
        if self.final_decision_layer and self.apply_spectral_norm and isinstance(self.final_decision_layer, nn.Linear):
            self.final_decision_layer = spectral_norm(self.final_decision_layer)
        self.apply(init_weights_general)

    def forward(self, input_data: torch.Tensor,
                # gaad_bboxes_for_assembly: Optional[torch.Tensor] = None, # Not used by wrapper directly
                # target_mel_shape_for_assembly: Optional[Tuple[int,int,int,int]] = None, # Not used by wrapper
                raw_audio_stats_input: Optional[torch.Tensor] = None, # NEW: Accept this
                return_features: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self.actual_discriminator_module is not None:
            if self.architecture_variant == "global_wubu_dct":
                # GlobalWuBuDCTDiscriminator expects regional features as input_data,
                # and can optionally take raw_audio_stats (though it might not use them if its internal aux is off)
                return self.actual_discriminator_module(input_data, raw_audio_stats_input=raw_audio_stats_input, return_features=return_features)
            elif self.architecture_variant == "multi_modal_wudio_mel":
                # MultiModalWudioMelDiscriminator expects assembled_mel as input_data and raw_audio_stats_input
                return self.actual_discriminator_module(input_data, raw_audio_stats_input=raw_audio_stats_input, return_features=return_features)
            else: 
                 raise RuntimeError(f"actual_discriminator_module is set, but variant '{self.architecture_variant}' not handled for delegation in forward.")
        else: # "default" variant, 'self' is the discriminator
            B = input_data.shape[0]; device = input_data.device; dtype = input_data.dtype
            main_features: Optional[torch.Tensor] = None; logits: torch.Tensor
            input_data_for_stats_calc: Optional[torch.Tensor] = None

            if self.input_type == "dct": # Default D processing regional features
                if input_data.ndim == 3 and input_data.shape[1] == self.num_gaad_regions and input_data.shape[2] == self.num_features_per_region_for_default_dct_d:
                    flat_regional_features = input_data # (B, NumRegions, FeaturesPerRegion)
                elif input_data.ndim == 2 and input_data.shape[0] % self.num_gaad_regions == 0 and input_data.shape[1] == self.num_features_per_region_for_default_dct_d:
                    flat_regional_features = input_data.reshape(B, self.num_gaad_regions, self.num_features_per_region_for_default_dct_d)
                else: raise ValueError(f"Default D-DCT: Unsupported input_data shape {input_data.shape}. Expected (B, NumRegions, FeaturesPerRegion) or (B*NumRegions, FeaturesPerRegion).")
                
                input_data_for_stats_calc = flat_regional_features # Use the regional features for stats
                embedded_coeffs = self.dct_coeff_embed_disc(flat_regional_features) 
                regional_input_flat = embedded_coeffs.reshape(B*self.num_gaad_regions, self.dct_coeff_embed_dim) 
                regional_features_flat = self.feature_extractor_module(regional_input_flat) 
                regional_features_seq = regional_features_flat.reshape(B, self.num_gaad_regions, self.wubu_d_region_feature_dim) 
                transformer_input = regional_features_seq
                if self.use_region_pos_embed and self.region_pos_embed_eff is not None: 
                    if self.use_cls_token_dct_d and hasattr(self, 'cls_token'): 
                        transformer_input = torch.cat((self.cls_token.expand(B,-1,-1), regional_features_seq),dim=1) + self.region_pos_embed_eff[:,:(self.num_gaad_regions+1),:] 
                    else: transformer_input = regional_features_seq + self.region_pos_embed_eff[:,:self.num_gaad_regions,:] 
                tf_out = self.context_transformer(transformer_input) 
                main_feat_agg = tf_out[:,0] if self.use_cls_token_dct_d and hasattr(self, 'cls_token') else tf_out.mean(dim=1) 
                main_features = main_feat_agg
                current_decision_input = main_feat_agg
                
                # Default D's 'dct' mode can use global stats from raw_audio_stats_input if provided
                if self.use_global_stats_aux_input and self.global_stats_mlp:
                    if raw_audio_stats_input is not None: # Use pre-calculated stats if available
                        if raw_audio_stats_input.shape[0] == B and raw_audio_stats_input.shape[1] == self.num_global_stats:
                            proj_gs = self.global_stats_mlp(raw_audio_stats_input)
                            current_decision_input = torch.cat([main_feat_agg, proj_gs], dim=-1)
                        else:
                            self.logger.warning_once(f"Default D-DCT: raw_audio_stats_input shape {raw_audio_stats_input.shape} mismatch with expected ({B}, {self.num_global_stats}). Not using for aux.")
                    elif input_data_for_stats_calc is not None: # Fallback to calculating from regional features
                        gs_raw = self._calculate_global_stats(input_data_for_stats_calc); proj_gs = self.global_stats_mlp(gs_raw)
                        current_decision_input = torch.cat([main_feat_agg, proj_gs], dim=-1)

                logits = self.final_decision_layer(current_decision_input) if self.final_decision_layer else torch.mean(current_decision_input, -1) 
            
            elif self.input_type == "mel": # Default D processing assembled Mel
                mel_in = input_data
                # This path for "default" D with "mel" input does not assemble from DCT.
                # It expects assembled_mel directly.
                # gaad_bboxes_for_assembly and target_mel_shape_for_assembly are not used here.
                input_data_for_stats_calc = mel_in # Use the Mel spectrogram for stats if needed
                
                if self.msd_num_scales > 1 and isinstance(self.feature_extractor_module, nn.ModuleList): 
                    all_scale_logits=[]; current_mel=mel_in
                    for i, sub_d in enumerate(self.feature_extractor_module):
                        sub_l, sub_f = sub_d(current_mel, return_features=True); all_scale_logits.append(sub_l)
                        if i==0: main_features=sub_f
                        if i < self.msd_num_scales-1: current_mel=F.avg_pool2d(current_mel,3,2,1,count_include_pad=False)
                    avg_logits = torch.stack(all_scale_logits).mean(0)
                    if self.use_global_stats_aux_input and self.global_stats_mlp and self.final_decision_layer:
                        if raw_audio_stats_input is not None: # Use pre-calculated stats if available
                             gs_from_raw = raw_audio_stats_input
                        elif input_data_for_stats_calc is not None:
                             gs_from_raw = self._calculate_global_stats(input_data_for_stats_calc)
                        else: gs_from_raw = None
                        
                        if gs_from_raw is not None and gs_from_raw.shape[0]==B and gs_from_raw.shape[1]==self.num_global_stats:
                            proj_gs=self.global_stats_mlp(gs_from_raw)
                            logits=self.final_decision_layer(torch.cat([avg_logits.unsqueeze(1),proj_gs],dim=-1))
                        else:
                            self.logger.warning_once(f"Default D-Mel (MSD): Global stats not usable (shape mismatch or missing). Using avg_logits directly.")
                            logits = avg_logits
                    else: logits=avg_logits
                else: # Default Single Scale Mel
                    if self.use_global_stats_aux_input and self.global_stats_mlp and self.global_pool_for_mel_d and self.final_decision_layer:
                        cnn_map = self.feature_extractor_module.feature_extractor(mel_in) 
                        main_features=cnn_map; pooled=self.global_pool_for_mel_d(cnn_map); squeezed=pooled.view(B,-1)
                        
                        if raw_audio_stats_input is not None: # Use pre-calculated stats if available
                             gs_from_raw = raw_audio_stats_input
                        elif input_data_for_stats_calc is not None:
                             gs_from_raw = self._calculate_global_stats(input_data_for_stats_calc)
                        else: gs_from_raw = None

                        if gs_from_raw is not None and gs_from_raw.shape[0]==B and gs_from_raw.shape[1]==self.num_global_stats:
                            proj_gs = self.global_stats_mlp(gs_from_raw)
                            logits = self.final_decision_layer(torch.cat([squeezed,proj_gs],dim=-1))
                        else:
                             self.logger.warning_once(f"Default D-Mel (SSD): Global stats not usable for aux. Falling back to direct CNN output for decision (if no final_decision_layer for main path).")
                             # Fallback: if final_decision_layer was *only* for aux, then we need to get logits from feature_extractor_module
                             # This part of the logic for single-scale + aux needs to ensure logits are always produced.
                             # The original _SingleScaleMelDiscriminator's forward produces logits.
                             output_ssd = self.feature_extractor_module(mel_in, return_features=return_features)
                             if return_features: logits, main_features = output_ssd
                             else: logits = output_ssd; main_features = None
                    else: # No aux, direct path
                        output_ssd = self.feature_extractor_module(mel_in, return_features=return_features) 
                        if return_features: logits, main_features = output_ssd
                        else: logits = output_ssd; main_features = None
            else: raise ValueError(f"Default D forward: Unsupported input_type {self.input_type}")

            if logits.ndim > 1 and logits.shape[-1]==1: logits=logits.squeeze(-1)
            elif logits.ndim > 2 and all(s==1 for s in logits.shape[1:]): logits=logits.squeeze() # type: ignore
            if logits.ndim == 0 and B==1: logits=logits.unsqueeze(0)
            if logits.ndim > 1 and logits.shape[0]==B and logits.numel() != B : # If it's (B, K) where K > 1, average
                 logits=torch.mean(logits,dim=tuple(range(1,logits.ndim)))
            elif logits.ndim > 1 and logits.shape[0]!=B: # Should not happen if logic is correct
                 self.logger.error(f"Logits shape {logits.shape} mismatch with batch size {B}. Returning zeros.")
                 logits=torch.zeros(B,device=device,dtype=dtype)

            if return_features:
                if main_features is None:
                    if self.input_type == "mel" and self.msd_num_scales <= 1 and not self.use_global_stats_aux_input and hasattr(self.feature_extractor_module, 'feature_extractor'): 
                        main_features = self.feature_extractor_module.feature_extractor(mel_in if 'mel_in' in locals() else input_data) 
                    else: main_features = logits.detach().clone().unsqueeze(-1) if logits.ndim==1 else logits.detach().clone()
                return logits, main_features
            return logits

    def _assemble_mel_from_dct_regions(self, dct_regions: torch.Tensor, gaad_bboxes: torch.Tensor,
                                   target_mel_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        # This method is specific to the "default" D when its input_type is "mel"
        # AND it's being fed regional DCT features that need assembly.
        # It's effectively a utility for that specific path within the default D.
        # It should NOT be called if self.actual_discriminator_module is active.
        # The HybridTrainer._assemble_mel_from_transformed_coeffs_regions is the more general one.
        
        _asm_logger = logging.getLogger("WuBuSpecTransV01.DefaultD.MelAssembly")

        B, N_Reg, F_p, T_p = dct_regions.shape # Expects unnormalized DCT (B, N_Reg, Fp, Tp)
        _, C_target, H_target, W_target = target_mel_shape
        device=dct_regions.device; dtype=dct_regions.dtype
        
        if idct_2d is None or not TORCH_DCT_AVAILABLE: 
            _asm_logger.error("IDCT function not available for Mel assembly in Default D. Returning zeros.")
            return torch.zeros(target_mel_shape, device=device, dtype=dtype)

        dct_flat = dct_regions.reshape(-1, F_p, T_p)
        # Ensure float32 for idct_2d if current dtype is not (e.g. float16 from AMP)
        input_dtype_for_idct = dct_flat.dtype
        compute_dtype_for_idct = torch.float32 if input_dtype_for_idct != torch.float32 else input_dtype_for_idct
        
        spatial_flat = idct_2d(dct_flat.to(compute_dtype_for_idct))
        spatial_regions = spatial_flat.reshape(B, N_Reg, F_p, T_p).to(input_dtype_for_idct) # Convert back

        canvas = torch.zeros(target_mel_shape, device=device, dtype=dtype)
        counts = torch.zeros(target_mel_shape, device=device, dtype=dtype) + EPS
        for b_idx in range(B):
            for r_idx in range(N_Reg):
                t1,f1,t2,f2 = gaad_bboxes[b_idx,r_idx].round().int().tolist()
                fc,tc=max(0,f1),max(0,t1); fce,tce=min(H_target,f2),min(W_target,t2)
                if tc>=tce or fc>=fce: continue
                patch=spatial_regions[b_idx,r_idx].unsqueeze(0).unsqueeze(0)
                th,tw = fce-fc, tce-tc
                if th<=0 or tw<=0: continue
                try: r_patch = TF.resize(patch,(th,tw),interpolation=TF.InterpolationMode.BILINEAR,antialias=True)
                except RuntimeError as e_resize: 
                    _asm_logger.debug(f"TF.resize failed for region {r_idx} in batch {b_idx}: {e_resize}. Skipping patch.") 
                    continue
                canvas[b_idx,0,fc:fce,tc:tce]+=r_patch.squeeze(0).squeeze(0)
                counts[b_idx,0,fc:fce,tc:tce]+=1.0
        return canvas/counts

    def _calculate_global_stats(self, data_for_stats: torch.Tensor) -> torch.Tensor:
        # This method is specific to the "default" D logic.
        if data_for_stats.numel()==0: return torch.zeros(data_for_stats.shape[0],2,device=data_for_stats.device,dtype=data_for_stats.dtype)
        dims_to_reduce = tuple(range(1,data_for_stats.ndim))
        if not dims_to_reduce: # Handle scalar per batch item (e.g., if data_for_stats is (B,))
            mean_stat=data_for_stats.clone()
            std_stat=torch.zeros_like(data_for_stats) 
        else: 
            mean_stat=torch.mean(data_for_stats,dim=dims_to_reduce,keepdim=False)
            std_stat=torch.std(data_for_stats,dim=dims_to_reduce,keepdim=False)
        std_stat = torch.max(std_stat, torch.tensor(EPS,device=std_stat.device,dtype=std_stat.dtype)) # Avoid division by zero if std is 0
        return torch.stack([mean_stat,std_stat],dim=-1)



# =====================================================================
# VAE-GAN Model Components
# =====================================================================
class WuBuSpecTransNet(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config: Dict, gaad_config: Dict,
                 wubu_s_config_enc: Dict, wubu_g_config_gen: Dict): # wubu_g_config_gen no longer Optional
        super().__init__()
        self.args = args
        # Storing configs can be useful for debugging or if sub-modules need them directly
        self.audio_config = audio_config 
        self.gaad_config = gaad_config
        self.logger = logging.getLogger("WuBuSpecTransV01.MainNet")
        self.latent_dim = args.latent_dim

        self.encoder = AudioSpecEncoder(args, audio_config, gaad_config, wubu_s_config_enc, self.latent_dim)
        self.generator = AudioSpecGenerator(args, audio_config, gaad_config, self.latent_dim) # wubu_g_config_gen is used internally by G

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"WuBuSpecTransNet Initialized with VAE transform '{getattr(args, 'vae_transform_type', 'complex_dft_ri')}'. Params: {param_count:,}")

    def encode(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder(mel_spectrogram)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def forward(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, target_features_for_recon, gaad_bboxes_from_enc = self.encode(mel_spectrogram)
        z = self.reparameterize(mu, logvar)
        recon_features = self.decode(z)
        return recon_features, mu, logvar, gaad_bboxes_from_enc, target_features_for_recon


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
                 preload_to_ram: bool = False,
                 yield_raw_audio: bool = False):
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
        self.yield_raw_audio = yield_raw_audio

        self.segments_info: List[Tuple[int, int, int]] = [] # file_idx, start_sample, end_sample
        # Type hint for preloaded_data items now reflects that 'raw_audio' key might be absent
        self.preloaded_data: Optional[List[Dict[str, Union[torch.Tensor, np.ndarray]]]] = [] if preload_to_ram else None

        segment_samples = int(self.segment_duration_sec * self.sample_rate)
        overlap_samples = int(self.segment_overlap_sec * self.sample_rate)
        step_samples = segment_samples - overlap_samples

        for file_idx, audio_path_str in enumerate(tqdm(audio_file_paths, desc="Processing audio files for dataset")):
            audio_path = Path(audio_path_str)
            if not audio_path.is_file():
                self.logger.warning(f"Audio file not found: {audio_path}. Skipping.")
                continue
            try:
                waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                waveform = waveform.astype(np.float32)
                
                num_wav_samples = waveform.shape[0]
                current_pos = 0
                while current_pos + segment_samples <= num_wav_samples:
                    start_s = current_pos
                    end_s = current_pos + segment_samples
                    self.segments_info.append((file_idx, start_s, end_s))
                    
                    if preload_to_ram and self.preloaded_data is not None:
                        segment_audio_np = waveform[start_s:end_s]
                        mel_spec_tensor = self._audio_to_mel_spectrogram(segment_audio_np)
                        # Initialize item_data with 'mel'
                        item_data_preload: Dict[str, Union[torch.Tensor, np.ndarray]] = {'mel': mel_spec_tensor}
                        if self.yield_raw_audio:
                            item_data_preload['raw_audio'] = segment_audio_np
                        # 'raw_audio' key is only added if self.yield_raw_audio is True
                        self.preloaded_data.append(item_data_preload)
                    current_pos += step_samples
            except Exception as e:
                self.logger.error(f"Error processing audio file {audio_path}: {e}", exc_info=True)
        
        if data_fraction < 1.0 and len(self.segments_info) > 1:
            num_to_keep = max(1, int(len(self.segments_info) * data_fraction))
            if num_to_keep < len(self.segments_info):
                sampled_indices = random.sample(range(len(self.segments_info)), num_to_keep)
                self.segments_info = [self.segments_info[i] for i in sampled_indices]
                if preload_to_ram and self.preloaded_data is not None:
                    self.preloaded_data = [self.preloaded_data[i] for i in sampled_indices]
            self.logger.info(f"Using {data_fraction*100:.1f}% of segments: {len(self.segments_info)} segments.")

        if not self.segments_info:
            self.logger.error("AudioSegmentDataset: No valid audio segments found or processed.")
            raise ValueError("No audio segments available for the dataset.")
        
        self.audio_waveforms_cache: Dict[str, np.ndarray] = {}
        if not preload_to_ram:
            self.logger.info("Caching full waveforms to RAM for on-demand segment extraction...")
            unique_file_indices = sorted(list(set(info[0] for info in self.segments_info)))
            for file_idx in tqdm(unique_file_indices, desc="Caching full waveforms"):
                audio_path_str_cache = self.audio_file_paths[file_idx]
                if str(audio_path_str_cache) not in self.audio_waveforms_cache:
                    audio_path_cache = Path(audio_path_str_cache)
                    if audio_path_cache.is_file():
                        try:
                            wf_cache, _ = librosa.load(audio_path_cache, sr=self.sample_rate, mono=True)
                            self.audio_waveforms_cache[str(audio_path_cache)] = wf_cache.astype(np.float32)
                        except Exception as e_cache_wf:
                            self.logger.error(f"Error pre-caching waveform for {audio_path_cache}: {e_cache_wf}")

        self.logger.info(f"AudioSegmentDataset initialized. Total segments: {len(self.segments_info)}. "
                         f"Yield raw audio: {self.yield_raw_audio}. Preloaded segment data: {preload_to_ram}.")

    def _audio_to_mel_spectrogram(self, audio_segment: np.ndarray) -> torch.Tensor:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment, sr=self.sample_rate, n_fft=self.args.n_fft,
            hop_length=self.args.hop_length, n_mels=self.args.n_mels,
            fmin=self.args.fmin, fmax=self.args.fmax if self.args.fmax is not None else self.sample_rate/2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_db_normalized = (mel_spec_db - self.args.db_norm_min) / (self.args.db_norm_max - self.args.db_norm_min)
        mel_spec_db_normalized = np.clip(mel_spec_db_normalized, 0.0, 1.0)
        mel_spec_db_normalized = (mel_spec_db_normalized * 2.0) - 1.0
        
        return torch.from_numpy(mel_spec_db_normalized).float().unsqueeze(0)

    def __len__(self) -> int:
        return len(self.segments_info)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]: # type: ignore
        if self.preloaded_data is not None:
            item = self.preloaded_data[idx].copy() # Make a copy to avoid modifying cached item
            # Ensure 'raw_audio' key is only present if self.yield_raw_audio is True for preloaded
            if not self.yield_raw_audio and 'raw_audio' in item:
                del item['raw_audio']
            elif self.yield_raw_audio and 'raw_audio' not in item:
                # This case ideally shouldn't happen if preloading logic is correct.
                # If it does, we should add a placeholder or log a warning.
                # For now, let's assume preloading correctly includes/omits based on yield_raw_audio
                self.logger.warning_once(f"Preloaded item for idx {idx} missing 'raw_audio' key when yield_raw_audio=True. This might cause issues.")
            return item
        
        file_idx, start_sample, end_sample = self.segments_info[idx]
        audio_path_str = self.audio_file_paths[file_idx]
        
        waveform = self.audio_waveforms_cache.get(audio_path_str)

        if waveform is None: 
            self.logger.debug(f"Waveform for {audio_path_str} not in cache (idx {idx}). Loading on demand.")
            try:
                wf_fly, _ = librosa.load(Path(audio_path_str), sr=self.sample_rate, mono=True)
                waveform = wf_fly.astype(np.float32)
            except Exception as e_load_fly:
                self.logger.error(f"CRITICAL: Failed to load waveform {audio_path_str} on demand for idx {idx}: {e_load_fly}")
                num_time_frames = math.ceil(int(self.segment_duration_sec * self.sample_rate) / self.hop_length)
                dummy_mel = torch.zeros((1, self.n_mels, num_time_frames), dtype=torch.float)
                item_data: Dict[str, Union[torch.Tensor, Optional[np.ndarray]]] = {'mel': dummy_mel}
                if self.yield_raw_audio: 
                     # If we must return a dict with 'raw_audio', provide a dummy, but it indicates an issue
                    item_data['raw_audio'] = np.zeros(int(self.segment_duration_sec * self.sample_rate), dtype=np.float32)
                return item_data # type: ignore[return-value]

        segment_audio_np_slice = waveform[start_sample:end_sample]
        mel_spectrogram_tensor = self._audio_to_mel_spectrogram(segment_audio_np_slice)
        
        item_data_runtime: Dict[str, Union[torch.Tensor, np.ndarray]] = {'mel': mel_spectrogram_tensor}
        if self.yield_raw_audio:
            item_data_runtime['raw_audio'] = segment_audio_np_slice
            
        return item_data_runtime # type: ignore[return-value]

# =====================================================================
# VAE-GAN Trainer
# =====================================================================
class HybridTrainer:
    def __init__(self,
                 model: "WuBuSpecTransNet", 
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
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.ddp_active = ddp_active
        self.am_main_process = (rank == 0)
        self.logger = logging.getLogger("WuBuSpecTransV01.Trainer")
        
        self.vae_transform_type = getattr(args, 'vae_transform_type', 'complex_dft_ri')
        self.region_proc_size_tuple = (args.region_proc_size_t, args.region_proc_size_f)

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

        if self.am_main_process: self.logger.info("Initializing Discriminators and their Optimizers...")
        
        def _determine_disc_input_type_from_variant(variant_name: str, args_ref: argparse.Namespace) -> str:
            if variant_name == "multi_modal_wudio_mel": return "mel"
            elif variant_name == "global_wubu_dct": return "dct" 
            elif variant_name == "default": return args_ref.disc_input_type
            self.logger.warning(f"Unknown discriminator variant '{variant_name}', defaulting D input type to 'mel'.")
            return "mel"

        primary_disc_intended_input_type = _determine_disc_input_type_from_variant(args.primary_disc_architecture_variant, args)
        alt_disc_intended_input_type = _determine_disc_input_type_from_variant(args.alt_disc_architecture_variant, args)

        primary_disc_config_dict, _ = self._get_discriminator_configs(self.args, primary_disc_intended_input_type, is_primary=True, variant_hint=args.primary_disc_architecture_variant)
        alt_disc_config_dict, _ = self._get_discriminator_configs(self.args, alt_disc_intended_input_type, is_primary=False, variant_hint=args.alt_disc_architecture_variant)
        
        primary_disc_config_dict["architecture_variant"] = args.primary_disc_architecture_variant
        alt_disc_config_dict["architecture_variant"] = args.alt_disc_architecture_variant

        self.discriminator_primary_obj = AudioSpecDiscriminator(self.args, self._get_audio_config_ref(), self._get_gaad_config_ref(), primary_disc_config_dict).to(device)
        self.discriminator_alternative_obj = AudioSpecDiscriminator(self.args, self._get_audio_config_ref(), self._get_gaad_config_ref(), alt_disc_config_dict).to(device)
        
        self.primary_disc_effective_trainer_input_type = self.discriminator_primary_obj.effective_input_type_for_trainer
        self.alternative_disc_effective_trainer_input_type = self.discriminator_alternative_obj.effective_input_type_for_trainer
        
        if self.am_main_process:
            self.logger.info(f"Primary D (Variant: '{args.primary_disc_architecture_variant}', Effective Trainer Input: '{self.primary_disc_effective_trainer_input_type}') initialized. Params: {sum(p.numel() for p in self.discriminator_primary_obj.parameters()):,}")
            self.logger.info(f"Alternative D (Variant: '{args.alt_disc_architecture_variant}', Effective Trainer Input: '{self.alternative_disc_effective_trainer_input_type}') initialized. Params: {sum(p.numel() for p in self.discriminator_alternative_obj.parameters()):,}")

        if self.ddp_active:
            local_rank_ddp = self.args.local_rank if hasattr(self.args, 'local_rank') and self.args.local_rank != -1 else rank
            ddp_find_unused_d = getattr(self.args, 'ddp_find_unused_params_d', False)
            self.discriminator_primary_obj = DDP(self.discriminator_primary_obj, device_ids=[local_rank_ddp], output_device=local_rank_ddp, find_unused_parameters=ddp_find_unused_d)
            self.discriminator_alternative_obj = DDP(self.discriminator_alternative_obj, device_ids=[local_rank_ddp], output_device=local_rank_ddp, find_unused_parameters=ddp_find_unused_d)

        q_cfg_disc_shared = None
        if args.q_controller_enabled:
            q_cfg_disc_shared = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
            if 'lkl_num_probation_steps' not in q_cfg_disc_shared:
                 q_cfg_disc_shared['lkl_num_probation_steps'] = max(3, q_cfg_disc_shared.get('lambda_kl_state_history_len', 5) + 1)
        
        lr_disc_alt = getattr(args, 'learning_rate_disc_alt', args.learning_rate_disc)
        self.optimizer_disc_primary = RiemannianEnhancedSGD(
            self.discriminator_primary_obj.parameters(), lr=args.learning_rate_disc,
            q_learning_config=q_cfg_disc_shared.copy() if q_cfg_disc_shared else None,
            max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type=f"discriminator_primary_{args.primary_disc_architecture_variant}"
        )
        self.optimizer_disc_alternative = RiemannianEnhancedSGD(
            self.discriminator_alternative_obj.parameters(), lr=lr_disc_alt,
            q_learning_config=q_cfg_disc_shared.copy() if q_cfg_disc_shared else None,
            max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type=f"discriminator_alt_{args.alt_disc_architecture_variant}"
        )
        self.q_controller_d_primary = getattr(self.optimizer_disc_primary, 'q_controller', None)
        self.q_controller_d_alt = getattr(self.optimizer_disc_alternative, 'q_controller', None)
        
        self.active_discriminator_key = 'primary' 
        if self.args.enable_heuristic_disc_switching and self.args.initial_disc_type:
            user_prefers_mel_like = (self.args.initial_disc_type == 'mel')
            user_prefers_feature_like = (self.args.initial_disc_type == 'dct') 

            primary_is_mel_like = "mel" in self.primary_disc_effective_trainer_input_type
            primary_is_feature_like = "features_regional" in self.primary_disc_effective_trainer_input_type
            alt_is_mel_like = "mel" in self.alternative_disc_effective_trainer_input_type
            alt_is_feature_like = "features_regional" in self.alternative_disc_effective_trainer_input_type

            if user_prefers_mel_like:
                if primary_is_mel_like: self.active_discriminator_key = 'primary'
                elif alt_is_mel_like: self.active_discriminator_key = 'alternative'
            elif user_prefers_feature_like:
                if primary_is_feature_like: self.active_discriminator_key = 'primary'
                elif alt_is_feature_like: self.active_discriminator_key = 'alternative'
        
        self.active_discriminator: nn.Module # Will be set by _update_active_discriminator_pointers
        self.optimizer_disc_active: RiemannianEnhancedSGD # Will be set
        self.q_controller_d_active: Optional[HAKMEMQController] = None 
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
                try: self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device) # Mel is [-1,1], this might need data_range=2.0 or normalize to [0,1]
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

        self.enable_heuristic_interventions = getattr(args, 'enable_heuristic_interventions', False) 
        self.enable_heuristic_disc_switching = getattr(args, 'enable_heuristic_disc_switching', False)
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
        self.rec_features_stagnant = False 

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
        self.lambda_feat_match_heuristic = getattr(args, 'lambda_feat_match_heuristic', 0.0) 
        self.lambda_g_easy_win_penalty_heuristic = getattr(args, 'lambda_g_easy_win_penalty_heuristic', 1.5)
        self.heuristic_active_d_lr_boost_factor = getattr(args, 'heuristic_active_d_lr_boost_factor', 1.8)
        self.heuristic_d_q_explore_boost_epsilon = getattr(args, 'heuristic_d_q_explore_boost_epsilon', 0.7)
        self.heuristic_d_q_explore_duration = getattr(args, 'heuristic_d_q_explore_duration', 10)
        
        if self.am_main_process:
             self.logger.info(f"HybridTrainer initialized. VAE Transform: {self.vae_transform_type}. Initial Active D: '{self.active_discriminator_key}' (Effective Input: '{self.active_disc_effective_trainer_input_type}'). Heuristics {'ENABLED' if self.enable_heuristic_interventions else 'DISABLED'}.")

    def _get_audio_config_ref(self) -> Dict:
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        return getattr(m_ref, 'audio_config', {})

    def _get_gaad_config_ref(self) -> Dict:
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        return getattr(m_ref, 'gaad_config', {})

    def _get_discriminator_configs(self, current_args: argparse.Namespace, disc_intended_input_type: str, is_primary: bool, variant_hint: str) -> Tuple[Dict, Optional[Dict]]:
        disc_config = {
            "input_type": disc_intended_input_type,
            "architecture_variant": variant_hint,
            "apply_spectral_norm": current_args.disc_apply_spectral_norm,
        }

        if variant_hint == "multi_modal_wudio_mel":
            disc_config["base_disc_channels"] = getattr(current_args, 'wudio_d_mel_cnn_base_ch', current_args.disc_base_disc_channels)
            disc_config["max_disc_channels"] = getattr(current_args, 'wudio_d_mel_cnn_max_ch', current_args.disc_max_disc_channels)
            target_dim_wudio = getattr(current_args, 'wudio_d_mel_cnn_target_dim', current_args.disc_target_final_feature_dim)
            if isinstance(target_dim_wudio, list) and len(target_dim_wudio) == 1: target_dim_wudio = [target_dim_wudio[0]]*2
            elif not (isinstance(target_dim_wudio, list) and len(target_dim_wudio) == 2): target_dim_wudio = [4,4] 
            disc_config["target_mel_disc_final_feature_dim"] = target_dim_wudio
            disc_config["max_mel_disc_downsample_layers"] = getattr(current_args, 'wudio_d_mel_cnn_max_downs', current_args.max_mel_disc_downsample_layers)
        elif variant_hint == "default": 
            disc_config["base_disc_channels"] = current_args.disc_base_disc_channels 
            disc_config["max_disc_channels"] = current_args.disc_max_disc_channels
            disc_config["target_mel_disc_final_feature_dim"] = current_args.disc_target_final_feature_dim
            disc_config["max_mel_disc_downsample_layers"] = current_args.max_mel_disc_downsample_layers
            disc_config["disc_use_global_stats_aux"] = getattr(current_args, 'disc_use_global_stats_aux', False)
            disc_config["disc_global_stats_mlp_hidden_dim"] = getattr(current_args, 'disc_global_stats_mlp_hidden_dim', 32)
        elif variant_hint == "global_wubu_dct": 
            disc_config["disc_use_global_stats_aux_dct_global_wubu"] = getattr(current_args, 'disc_use_global_stats_aux_dct_global_wubu', False)
            disc_config["disc_global_stats_mlp_hidden_dim_dct_global_wubu"] = getattr(current_args, 'disc_global_stats_mlp_hidden_dim_dct_global_wubu', 64)
        
        wubu_d_config_for_this_d = None
        wubu_prefix_for_d: Optional[str] = None
        if variant_hint == "global_wubu_dct":
            wubu_prefix_for_d = "wubu_d_global"
        elif variant_hint == "default" and disc_intended_input_type == "dct": 
            wubu_prefix_for_d = "wubu_d_region" 
        
        if wubu_prefix_for_d:
            if self.am_main_process: self.logger.info(f"Configuring WuBu stack with prefix '{wubu_prefix_for_d}' for D (is_primary: {is_primary}, variant: {variant_hint}, intended_input: {disc_intended_input_type})")
            wubu_d_config_for_this_d = _configure_wubu_stack(current_args, wubu_prefix_for_d)
        
        return disc_config, wubu_d_config_for_this_d

    def _update_active_discriminator_pointers(self):
        if self.active_discriminator_key == 'primary':
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_effective_trainer_input_type = self.primary_disc_effective_trainer_input_type
            self.q_controller_d_active = self.q_controller_d_primary
        elif self.active_discriminator_key == 'alternative':
            self.active_discriminator = self.discriminator_alternative_obj
            self.optimizer_disc_active = self.optimizer_disc_alternative
            self.active_disc_effective_trainer_input_type = self.alternative_disc_effective_trainer_input_type
            self.q_controller_d_active = self.q_controller_d_alt
        else:
            self.logger.error(f"Invalid active_discriminator_key: {self.active_discriminator_key}. Defaulting to primary.")
            self.active_discriminator_key = 'primary' 
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_effective_trainer_input_type = self.primary_disc_effective_trainer_input_type
            self.q_controller_d_active = self.q_controller_d_primary

        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_arch_variant = getattr(d_ref_active, 'architecture_variant', 'unknown_variant')
        
        if self.am_main_process:
            self.logger.info(f"Active Discriminator is now: '{self.active_discriminator_key}' (Arch Variant: '{active_d_arch_variant}', Effective Trainer Input: '{self.active_disc_effective_trainer_input_type}')")

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()

    def _compute_recon_loss(self, recon_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_features, target_features)

    def _prepare_raw_audio_stats(self,
                                 mel_spectrograms_norm: Optional[torch.Tensor],
                                 raw_audio_batch_list: Optional[List[Optional[Union[torch.Tensor, np.ndarray]]]],
                                 is_real_sample: bool
                                ) -> Optional[torch.Tensor]:
        if self.active_disc_effective_trainer_input_type != "assembled_mel_and_raw_audio_stats":
            return None

        B = 0
        if mel_spectrograms_norm is not None: B = mel_spectrograms_norm.shape[0]
        elif raw_audio_batch_list is not None: B = len(raw_audio_batch_list)

        if B == 0: self.logger.debug_once("_prepare_raw_audio_stats: No input (Mel or Raw)."); return None

        device = self.device
        output_dtype = next(iter(self.model.parameters()), torch.tensor(0.0)).dtype if hasattr(self.model, 'parameters') and next(self.model.parameters(), None) is not None else torch.float32

        num_stats_expected = getattr(self.args, 'wudio_d_num_global_audio_stats', 0)
        if num_stats_expected == 0: return None

        if not (MATPLOTLIB_AVAILABLE and librosa is not None):
            self.logger.warning_once("_prepare_raw_audio_stats: Librosa unavailable. Returning zeros.")
            return torch.zeros(B, num_stats_expected, device=device, dtype=output_dtype)

        all_batch_stats_np = []
        n_mels = self.args.n_mels
        # Define a reasonable dB range for librosa feature extraction from Mel
        # This helps prevent overflow/underflow when converting back to power.
        # These values can be tuned.
        SAFE_DB_MIN = -80.0 
        SAFE_DB_MAX = 0.0 # Assuming max dB for normalization was 0

        for b_idx in range(B):
            sample_stats_list = [] # Use a list to append, then convert to array
            current_raw_audio_item: Optional[Union[torch.Tensor, np.ndarray]] = None
            if is_real_sample and raw_audio_batch_list is not None and b_idx < len(raw_audio_batch_list):
                current_raw_audio_item = raw_audio_batch_list[b_idx]

            audio_for_librosa: Optional[np.ndarray] = None
            is_valid_audio_item_for_stats = False

            if current_raw_audio_item is not None:
                if isinstance(current_raw_audio_item, torch.Tensor):
                    if current_raw_audio_item.ndim > 0 and current_raw_audio_item.numel() > 0:
                        is_valid_audio_item_for_stats = True
                        audio_for_librosa = current_raw_audio_item.cpu().numpy()
                elif isinstance(current_raw_audio_item, np.ndarray):
                    if current_raw_audio_item.ndim > 0 and current_raw_audio_item.size > 0:
                        is_valid_audio_item_for_stats = True
                        audio_for_librosa = current_raw_audio_item

            if is_valid_audio_item_for_stats and audio_for_librosa is not None:
                try:
                    rms_frames = librosa.feature.rms(y=audio_for_librosa, frame_length=self.args.n_fft, hop_length=self.args.hop_length)[0]
                    sample_stats_list.extend([np.mean(rms_frames), np.std(rms_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                try:
                    cent_frames = librosa.feature.spectral_centroid(y=audio_for_librosa, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_length)[0]
                    sample_stats_list.extend([np.mean(cent_frames), np.std(cent_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                try:
                    rolloff_frames = librosa.feature.spectral_rolloff(y=audio_for_librosa, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_length)[0]
                    sample_stats_list.extend([np.mean(rolloff_frames), np.std(rolloff_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                try:
                    S_power_raw_audio = np.abs(librosa.stft(audio_for_librosa, n_fft=self.args.n_fft, hop_length=self.args.hop_length))**2
                    flatness_frames = librosa.feature.spectral_flatness(S=S_power_raw_audio, n_fft=self.args.n_fft, hop_length=self.args.hop_length)[0]
                    sample_stats_list.extend([np.mean(flatness_frames), np.std(flatness_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                if num_stats_expected > 8:
                    try:
                        zcr_frames = librosa.feature.zero_crossing_rate(audio_for_librosa, frame_length=self.args.n_fft, hop_length=self.args.hop_length)[0]
                        sample_stats_list.append(np.mean(zcr_frames))
                    except Exception: sample_stats_list.append(0.0)
            else:
                if mel_spectrograms_norm is None or b_idx >= mel_spectrograms_norm.shape[0]:
                    all_batch_stats_np.append(np.array([0.0] * num_stats_expected, dtype=np.float32)); continue # Ensure appending ndarray

                current_mel_tensor_for_stats = mel_spectrograms_norm[b_idx]
                if not is_real_sample:
                    current_mel_tensor_for_stats = current_mel_tensor_for_stats.detach()

                min_db_norm, max_db_norm = self.args.db_norm_min, self.args.db_norm_max
                mel_01 = (current_mel_tensor_for_stats.squeeze(0) + 1.0) / 2.0
                mel_db_unnormalized_raw = mel_01 * (max_db_norm - min_db_norm) + min_db_norm
                
                # Clip to a safer range before converting to power
                mel_db_unnormalized_clipped_np = np.clip(
                    mel_db_unnormalized_raw.cpu().float().numpy(), 
                    SAFE_DB_MIN, 
                    SAFE_DB_MAX 
                )
                
                # Convert to power, ensuring it's non-negative
                S_power_est_np = np.maximum(librosa.db_to_power(mel_db_unnormalized_clipped_np, ref=1.0), EPS)
                S_magnitude_est_np = np.sqrt(S_power_est_np) # Magnitude also non-negative

                try:
                    rms_frames = librosa.feature.rms(S=S_magnitude_est_np)[0]
                    sample_stats_list.extend([np.mean(rms_frames), np.std(rms_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                try:
                    cent_frames = librosa.feature.spectral_centroid(S=S_magnitude_est_np, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_length)[0]
                    sample_stats_list.extend([np.mean(cent_frames), np.std(cent_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                try:
                    rolloff_frames = librosa.feature.spectral_rolloff(S=S_magnitude_est_np, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_length)[0]
                    sample_stats_list.extend([np.mean(rolloff_frames), np.std(rolloff_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                try:
                    # Ensure S_power_est_np is used for flatness, and it's positive
                    flatness_frames = librosa.feature.spectral_flatness(S=S_power_est_np, n_fft=self.args.n_fft, hop_length=self.args.hop_length, amin=EPS)[0]
                    sample_stats_list.extend([np.mean(flatness_frames), np.std(flatness_frames)])
                except Exception: sample_stats_list.extend([0.0, 0.0])
                if num_stats_expected > 8:
                    try:
                        energy_total = np.sum(S_magnitude_est_np)
                        zcr_proxy = np.sum(S_magnitude_est_np[int(n_mels * 0.75):, :]) / max(EPS, energy_total) if energy_total > EPS else 0.0
                        sample_stats_list.append(zcr_proxy)
                    except Exception: sample_stats_list.append(0.0)
            
            # Clean NaNs/Infs from the collected stats for this sample
            sample_stats_cleaned_np = np.array([0.0 if not np.isfinite(s) else s for s in sample_stats_list], dtype=np.float32)
            all_batch_stats_np.append(sample_stats_cleaned_np)

        try:
            # Check if all_batch_stats_np contains actual numpy arrays
            if not all(isinstance(item, np.ndarray) for item in all_batch_stats_np):
                 self.logger.error(f"Not all items in all_batch_stats_np are numpy arrays. Type example: {type(all_batch_stats_np[0]) if all_batch_stats_np else 'Empty'}")
                 # Attempt to convert to ndarray, but this might fail if shapes are inconsistent
                 stats_tensor_np_final = np.array([np.asarray(item, dtype=np.float32) for item in all_batch_stats_np], dtype=np.float32)
            else:
                stats_tensor_np_final = np.array(all_batch_stats_np, dtype=np.float32)

            stats_tensor = torch.from_numpy(stats_tensor_np_final).to(device=device, dtype=output_dtype)
        except Exception as e:
            self.logger.error(f"Error converting stats to tensor (final): {e}. Stats list (first 5 items): {all_batch_stats_np[:5]}")
            return torch.zeros(B, num_stats_expected, device=device, dtype=output_dtype)

        num_gen_stats = stats_tensor.shape[1]
        if num_gen_stats < num_stats_expected: stats_tensor = torch.cat([stats_tensor, torch.zeros(B, num_stats_expected - num_gen_stats, device=device, dtype=output_dtype)], dim=1)
        elif num_gen_stats > num_stats_expected: stats_tensor = stats_tensor[:, :num_stats_expected]
        return stats_tensor




    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, mel_spectrograms_to_log: Optional[torch.Tensor],
                              num_sequences_to_log_max: int = 2):
        if not (self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb is not None and wandb.run): return
        if mel_spectrograms_to_log is None or mel_spectrograms_to_log.numel() == 0: return
        B_log, C_log, H_log, W_log = mel_spectrograms_to_log.shape
        if C_log != 1: mel_spectrograms_to_log = mel_spectrograms_to_log[:,0:1,:,:] # Use first channel if multi-channel
        num_to_actually_log = min(B_log, num_sequences_to_log_max)
        wandb_images_for_log = []
        for b_idx in range(num_to_actually_log):
            mel_tensor = mel_spectrograms_to_log[b_idx, 0, ...].cpu().float()
            fig_iter, ax_iter = None, None
            try:
                if MATPLOTLIB_AVAILABLE and plt is not None and librosa is not None and librosa.display is not None:
                    fig_iter, ax_iter = plt.subplots(1, 1, figsize=(max(5, W_log/H_log * 5 if H_log > 0 else 5), 5))
                    # De-normalize for display: [-1,1] -> [db_norm_min, db_norm_max]
                    mel_db_for_disp = mel_tensor * (self.args.db_norm_max - self.args.db_norm_min) / 2.0 + \
                                      (self.args.db_norm_max + self.args.db_norm_min) / 2.0
                    fmax_val = self.args.fmax if self.args.fmax is not None and self.args.fmax > self.args.fmin else self.args.sample_rate / 2.0
                    img = librosa.display.specshow(mel_db_for_disp.numpy(), ax=ax_iter, sr=self.args.sample_rate, 
                                                   hop_length=self.args.hop_length, x_axis='time', y_axis='mel', 
                                                   fmin=self.args.fmin, fmax=fmax_val, cmap='magma')
                    fig_iter.colorbar(img, ax=ax_iter, format='%+.2f dB')
                    ax_iter.set_title(f"{tag_prefix} S{b_idx} Ep{self.current_epoch+1} GStep{self.global_step}")
                    wandb_images_for_log.append(wandb.Image(fig_iter))
                else: raise RuntimeError("Matplotlib/Librosa display unavailable for detailed Mel plot.") 
            except Exception as e_disp:
                self.logger.debug_once(f"Librosa display failed for {tag_prefix} S{b_idx}: {e_disp}. Falling back to raw image.")
                img_0_1 = (mel_tensor.clamp(-1,1) + 1) / 2.0 # Normalize to [0,1] for basic image display
                wandb_images_for_log.append(wandb.Image(img_0_1, caption=f"{tag_prefix} S{b_idx} Ep{self.current_epoch+1} GStep{self.global_step} (raw)"))
            finally:
                if fig_iter is not None and plt is not None: plt.close(fig_iter) # Close the figure to free memory
        if wandb_images_for_log:
            try: wandb.log({f"samples_mel/{tag_prefix}": wandb_images_for_log}, step=self.global_step)
            except Exception as e_wandb_log: self.logger.error(f"Failed to log images to WandB for {tag_prefix}: {e_wandb_log}", exc_info=True)

    @staticmethod
    def _assemble_mel_from_transformed_coeffs_regions(
        transformed_coeffs_regions: torch.Tensor, 
        gaad_bboxes: torch.Tensor,
        target_mel_shape: Tuple[int, int, int, int],
        args_ref: argparse.Namespace, 
        region_proc_size: Tuple[int, int] 
    ) -> torch.Tensor:
        B, N_Reg = transformed_coeffs_regions.shape[0], transformed_coeffs_regions.shape[1]
        F_p_proc, T_p_proc = region_proc_size[1], region_proc_size[0] 
        
        _, C_target, H_target, W_target = target_mel_shape
        device = transformed_coeffs_regions.device
        output_dtype = torch.float32 
        _static_asm_logger = logging.getLogger("WuBuSpecTransV01.Trainer.StaticAssembly")

        spatial_regions_real: torch.Tensor
        vae_transform_type = getattr(args_ref, 'vae_transform_type', 'complex_dft_ri')

        if vae_transform_type == 'complex_dft_ri':
            if not torch.is_complex(transformed_coeffs_regions):
                 _static_asm_logger.error(f"Assembly: Expected complex DFT coeffs for complex_dft_ri, got {transformed_coeffs_regions.dtype}. Shape: {transformed_coeffs_regions.shape}")
                 return torch.zeros(target_mel_shape, device=device, dtype=output_dtype)
            if transformed_coeffs_regions.shape[2:] != (F_p_proc, T_p_proc):
                 _static_asm_logger.error(f"Assembly: Complex DFT coeffs shape {transformed_coeffs_regions.shape[2:]} != proc_size ({F_p_proc},{T_p_proc})")
                 return torch.zeros(target_mel_shape, device=device, dtype=output_dtype)

            spatial_flat = torch.fft.ifft2(transformed_coeffs_regions.reshape(-1, F_p_proc, T_p_proc), 
                                           norm=getattr(args_ref, 'dft_fft_norm', "backward"))
            spatial_flat_real = spatial_flat.real.float()
            spatial_regions_real = spatial_flat_real.reshape(B, N_Reg, F_p_proc, T_p_proc)
        
        elif vae_transform_type == 'dct':
            if torch.is_complex(transformed_coeffs_regions):
                 _static_asm_logger.error(f"Assembly: Expected real DCT coeffs for dct, got {transformed_coeffs_regions.dtype}.")
                 return torch.zeros(target_mel_shape, device=device, dtype=output_dtype)
            if transformed_coeffs_regions.shape[2:] != (F_p_proc, T_p_proc):
                 _static_asm_logger.error(f"Assembly: DCT coeffs shape {transformed_coeffs_regions.shape[2:]} != proc_size ({F_p_proc},{T_p_proc})")
                 return torch.zeros(target_mel_shape, device=device, dtype=output_dtype)
            
            if idct_2d is None or not TORCH_DCT_AVAILABLE:
                _static_asm_logger.error("Assembly: idct_2d function is None or torch-dct not available for 'dct' transform.")
                return torch.zeros(target_mel_shape, device=device, dtype=output_dtype)
            
            dct_coeffs_for_idct = transformed_coeffs_regions.reshape(-1, F_p_proc, T_p_proc)
            input_dtype_for_idct = dct_coeffs_for_idct.dtype
            compute_dtype_for_idct = torch.float32 if input_dtype_for_idct != torch.float32 else input_dtype_for_idct
            
            spatial_flat = idct_2d(dct_coeffs_for_idct.to(compute_dtype_for_idct))
            spatial_regions_real = spatial_flat.reshape(B, N_Reg, F_p_proc, T_p_proc).to(input_dtype_for_idct)
        else:
            _static_asm_logger.error(f"Assembly: Unsupported vae_transform_type '{vae_transform_type}'.")
            return torch.zeros(target_mel_shape, device=device, dtype=output_dtype)

        canvas = torch.zeros(target_mel_shape, device=device, dtype=output_dtype)
        counts = torch.zeros(target_mel_shape, device=device, dtype=output_dtype) + EPS

        for b_idx in range(B):
            for r_idx in range(N_Reg):
                t1, f1, t2, f2 = gaad_bboxes[b_idx, r_idx].round().int().tolist()
                f1_c, t1_c = max(0, f1), max(0, t1)
                f2_c, t2_c = min(H_target, f2), min(W_target, t2)
                if t1_c >= t2_c or f1_c >= f2_c: continue
                patch_to_place = spatial_regions_real[b_idx, r_idx].unsqueeze(0).unsqueeze(0)
                th_canvas, tw_canvas = f2_c - f1_c, t2_c - t1_c
                if th_canvas <= 0 or tw_canvas <= 0: continue
                try:
                    resized_patch = TF.resize(patch_to_place, (th_canvas, tw_canvas), 
                                              interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                except RuntimeError as e_resize:
                    _static_asm_logger.debug(f"TF.resize failed for region {r_idx} in batch {b_idx}: {e_resize}. Patch shape {patch_to_place.shape}, target ({th_canvas},{tw_canvas}). Skipping.")
                    continue
                canvas[b_idx, 0, f1_c:f2_c, t1_c:t2_c] += resized_patch.squeeze(0).squeeze(0)
                counts[b_idx, 0, f1_c:f2_c, t1_c:t2_c] += 1.0
        return canvas / counts

    def _train_discriminator_step(self,
                                  batch_real_mel_spectrograms: torch.Tensor,
                                  batch_real_raw_audio: Optional[List[Optional[Union[torch.Tensor, np.ndarray]]]],
                                  m_ref: "WuBuSpecTransNet") -> Dict[str, torch.Tensor]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_trainer_input_type = self.active_disc_effective_trainer_input_type
        
        B = batch_real_mel_spectrograms.shape[0]; device = batch_real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0)).dtype 

        real_labels = torch.ones(B, device=device, dtype=dtype_model) 
        fake_labels = torch.zeros(B, device=device, dtype=dtype_model)
        losses_d_micro: Dict[str, torch.Tensor] = {}

        for p in d_ref_active.parameters(): p.requires_grad = True
        for p in m_ref.parameters(): p.requires_grad = False
        if self.optimizer_disc_active: self.optimizer_disc_active.zero_grad(set_to_none=True)

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            real_input_for_d_main: torch.Tensor
            real_stats_for_d_aux: Optional[torch.Tensor] = None

            if active_d_trainer_input_type == "assembled_mel":
                real_input_for_d_main = batch_real_mel_spectrograms.to(device, dtype=dtype_model)
            elif active_d_trainer_input_type == "assembled_mel_and_raw_audio_stats":
                real_input_for_d_main = batch_real_mel_spectrograms.to(device, dtype=dtype_model)
                real_stats_for_d_aux = self._prepare_raw_audio_stats(None, batch_real_raw_audio, is_real_sample=True)
            elif active_d_trainer_input_type == "dft_features_regional" or active_d_trainer_input_type == "dct_features_regional": # Generalized
                with torch.no_grad():
                    _, _, real_features_target, _ = m_ref.encode(batch_real_mel_spectrograms.to(device, dtype=dtype_model))
                real_input_for_d_main = real_features_target.to(device, dtype=dtype_model)
            else:
                raise ValueError(f"D training (real): Unsupported type {active_d_trainer_input_type}")

            real_logits = d_ref_active(real_input_for_d_main, raw_audio_stats_input=real_stats_for_d_aux)
            loss_d_real = self.adversarial_loss(real_logits.squeeze(-1) if real_logits.ndim > 1 else real_logits, real_labels)

            fake_input_for_d_main: torch.Tensor
            fake_stats_for_d_aux: Optional[torch.Tensor] = None
            
            with torch.no_grad():
                fake_regional_features, _, _, gaad_bboxes_for_assembly, _ = m_ref(batch_real_mel_spectrograms.to(device, dtype=dtype_model))

            if active_d_trainer_input_type == "assembled_mel" or \
               active_d_trainer_input_type == "assembled_mel_and_raw_audio_stats":
                
                unnormalized_coeffs_for_assembly: torch.Tensor
                if self.vae_transform_type == 'complex_dft_ri':
                    unnormalized_coeffs_for_assembly = AudioSpecGenerator._unnormalize_and_reconstruct_coeffs_to_complex_dft(
                        fake_regional_features.detach(), self.args, self.region_proc_size_tuple
                    )
                elif self.vae_transform_type == 'dct':
                     unnormalized_coeffs_for_assembly = AudioSpecGenerator._unnormalize_dct_coeffs(
                        fake_regional_features.detach(), self.args, self.region_proc_size_tuple
                    )
                else: raise ValueError(f"D fake assembly: Unknown VAE transform type {self.vae_transform_type}")

                fake_input_for_d_main = HybridTrainer._assemble_mel_from_transformed_coeffs_regions(
                    unnormalized_coeffs_for_assembly, gaad_bboxes_for_assembly.detach(), 
                    batch_real_mel_spectrograms.shape, self.args, self.region_proc_size_tuple
                )
                if active_d_trainer_input_type == "assembled_mel_and_raw_audio_stats":
                    fake_stats_for_d_aux = self._prepare_raw_audio_stats(fake_input_for_d_main.detach(), None, is_real_sample=False)
            
            elif active_d_trainer_input_type == "dft_features_regional" or active_d_trainer_input_type == "dct_features_regional":
                fake_input_for_d_main = fake_regional_features.to(device, dtype=dtype_model).detach()
            else:
                raise ValueError(f"D training (fake): Unsupported type {active_d_trainer_input_type}")
            
            fake_logits = d_ref_active(fake_input_for_d_main, raw_audio_stats_input=fake_stats_for_d_aux)
            loss_d_fake = self.adversarial_loss(fake_logits.squeeze(-1) if fake_logits.ndim > 1 else fake_logits, fake_labels)
            
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total_scaled_for_accum_micro = loss_d_total_micro / self.grad_accum_steps

        self.scaler_disc.scale(loss_d_total_scaled_for_accum_micro).backward()
        losses_d_micro['loss_d_real_micro'] = loss_d_real.detach()
        losses_d_micro['loss_d_fake_micro'] = loss_d_fake.detach()
        losses_d_micro['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d_micro

    def _train_generator_step(self,
                              batch_real_mel_spectrograms: torch.Tensor,
                              batch_real_raw_audio: Optional[List[Optional[Union[torch.Tensor, np.ndarray]]]],
                              m_ref: "WuBuSpecTransNet") -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_trainer_input_type = self.active_disc_effective_trainer_input_type

        B = batch_real_mel_spectrograms.shape[0]; device = batch_real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0)).dtype
        real_labels_for_g = torch.ones(B, device=device, dtype=dtype_model)
        losses_g_micro: Dict[str, torch.Tensor] = {}
        recon_mel_for_log: Optional[torch.Tensor] = None # This will be the range-normalized version for display

        for p in d_ref_active.parameters(): p.requires_grad = False
        for p in m_ref.parameters(): p.requires_grad = True
        if self.optimizer_enc_gen: self.optimizer_enc_gen.zero_grad(set_to_none=True)

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            recon_regional_features, mu, logvar, gaad_bboxes_from_enc, target_regional_features = \
                m_ref(batch_real_mel_spectrograms.to(device,dtype=dtype_model))

            loss_recon_raw = self._compute_recon_loss(recon_regional_features, target_regional_features)
            loss_kl_raw = self._compute_kl_loss(mu, logvar)
            loss_recon_eff = self.lambda_recon * self.heuristic_override_lambda_recon_factor * loss_recon_raw
            loss_kl_eff = self.lambda_kl * self.heuristic_override_lambda_kl_factor * loss_kl_raw

            adv_input_for_d_main: torch.Tensor
            adv_stats_for_d_aux: Optional[torch.Tensor] = None
            d_output_for_adv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            assembled_mel_spatial_domain: Optional[torch.Tensor] = None # Raw output from assembly

            if active_d_trainer_input_type == "assembled_mel" or \
               active_d_trainer_input_type == "assembled_mel_and_raw_audio_stats":

                unnormalized_coeffs_for_assembly: torch.Tensor
                if self.vae_transform_type == 'complex_dft_ri':
                    unnormalized_coeffs_for_assembly = AudioSpecGenerator._unnormalize_and_reconstruct_coeffs_to_complex_dft(
                        recon_regional_features, self.args, self.region_proc_size_tuple
                    )
                elif self.vae_transform_type == 'dct':
                     unnormalized_coeffs_for_assembly = AudioSpecGenerator._unnormalize_dct_coeffs(
                        recon_regional_features, self.args, self.region_proc_size_tuple
                    )
                else: raise ValueError(f"G adv assembly: Unknown VAE transform type {self.vae_transform_type}")

                adv_input_for_d_main = HybridTrainer._assemble_mel_from_transformed_coeffs_regions(
                    unnormalized_coeffs_for_assembly, gaad_bboxes_from_enc,
                    batch_real_mel_spectrograms.shape, self.args, self.region_proc_size_tuple
                )
                assembled_mel_spatial_domain = adv_input_for_d_main # Store the raw assembled output
                if active_d_trainer_input_type == "assembled_mel_and_raw_audio_stats":
                    adv_stats_for_d_aux = self._prepare_raw_audio_stats(adv_input_for_d_main, None, is_real_sample=False)

            elif active_d_trainer_input_type == "dft_features_regional" or active_d_trainer_input_type == "dct_features_regional":
                adv_input_for_d_main = recon_regional_features
            else:
                raise ValueError(f"G training (adv): Unsupported D input type {active_d_trainer_input_type}")

            d_output_for_adv = d_ref_active(adv_input_for_d_main, raw_audio_stats_input=adv_stats_for_d_aux, return_features=self.heuristic_vae_feature_match_active)

            fake_logits_for_g: torch.Tensor
            features_from_d_for_g_feat_match: Optional[torch.Tensor] = None
            if isinstance(d_output_for_adv, tuple):
                fake_logits_for_g, features_from_d_for_g_feat_match = d_output_for_adv
            else: fake_logits_for_g = d_output_for_adv

            loss_g_adv_raw = self.adversarial_loss(fake_logits_for_g.squeeze(-1) if fake_logits_for_g.ndim > 1 else fake_logits_for_g, real_labels_for_g)
            loss_g_adv_eff = self.lambda_gan * self.heuristic_override_lambda_gan_factor * loss_g_adv_raw
            loss_g_total_micro = loss_recon_eff + loss_kl_eff + loss_g_adv_eff

            # --- NORMALIZATION FOR VISUAL LOGGING ---
            if assembled_mel_spatial_domain is not None and \
               self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and self.global_step > 0 and \
               ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0):
                
                temp_recon_for_norm = assembled_mel_spatial_domain.detach().clone()
                # Normalize each image in the batch independently to [-1, 1] for consistent visual scaling
                b_vis, c_vis, h_vis, w_vis = temp_recon_for_norm.shape
                if b_vis > 0 and temp_recon_for_norm.numel() > 0 : # Check if there's actual data
                    temp_recon_flat_per_img = temp_recon_for_norm.view(b_vis, -1)
                    min_vals = temp_recon_flat_per_img.min(dim=1, keepdim=True)[0]
                    max_vals = temp_recon_flat_per_img.max(dim=1, keepdim=True)[0]
                    range_vals = max_vals - min_vals
                    # Prevent division by zero if an image is flat (all same values)
                    range_vals[range_vals < EPS] = EPS 
                    
                    normalized_to_01_vis = (temp_recon_flat_per_img - min_vals) / range_vals
                    normalized_to_pm1_vis = (normalized_to_01_vis * 2.0) - 1.0
                    recon_mel_for_log = normalized_to_pm1_vis.view(b_vis, c_vis, h_vis, w_vis)
                else:
                    recon_mel_for_log = temp_recon_for_norm # Pass as is if empty or problematic
            # --- END NORMALIZATION FOR VISUAL LOGGING ---


            if self.heuristic_vae_feature_match_active and features_from_d_for_g_feat_match is not None and self.lambda_feat_match_heuristic > 0:
                with torch.no_grad():
                    real_input_for_d_fm_main: torch.Tensor
                    real_stats_for_d_fm_aux: Optional[torch.Tensor] = None
                    if active_d_trainer_input_type == "assembled_mel" or \
                       active_d_trainer_input_type == "assembled_mel_and_raw_audio_stats":
                        real_input_for_d_fm_main = batch_real_mel_spectrograms.to(device, dtype=dtype_model)
                        if active_d_trainer_input_type == "assembled_mel_and_raw_audio_stats":
                            real_stats_for_d_fm_aux = self._prepare_raw_audio_stats(None, batch_real_raw_audio, is_real_sample=True)
                    elif active_d_trainer_input_type == "dft_features_regional" or active_d_trainer_input_type == "dct_features_regional":
                        real_input_for_d_fm_main = target_regional_features.to(device, dtype=dtype_model).detach()
                    else: real_input_for_d_fm_main = torch.empty(0, device=device, dtype=dtype_model)

                    target_features_d: Optional[torch.Tensor] = None
                    if real_input_for_d_fm_main.numel() > 0:
                        target_d_output_fm = d_ref_active(real_input_for_d_fm_main, raw_audio_stats_input=real_stats_for_d_fm_aux, return_features=True)
                        target_features_d = target_d_output_fm[1] if isinstance(target_d_output_fm, tuple) else None

                if target_features_d is not None and features_from_d_for_g_feat_match.shape == target_features_d.shape:
                    loss_g_feat_match = F.mse_loss(features_from_d_for_g_feat_match, target_features_d.detach())
                    loss_g_total_micro += self.lambda_feat_match_heuristic * loss_g_feat_match
                    losses_g_micro['loss_g_feat_match_micro'] = loss_g_feat_match.detach()
                elif target_features_d is not None:
                    self.logger.warning_once(f"FM shapes mismatch: G_feat {features_from_d_for_g_feat_match.shape}, D_feat_real {target_features_d.shape}")

            if self.heuristic_penalize_g_easy_win_active:
                if loss_g_adv_raw.item() < self.G_WINNING_THRESH and loss_recon_raw.item() > self.TARGET_GOOD_RECON_THRESH_HEURISTIC:
                    denominator_penalty = loss_g_adv_raw.item() + getattr(self.args, 'g_easy_win_penalty_eps_denom', 1e-4)
                    penalty_val = (loss_recon_raw.item() - self.TARGET_GOOD_RECON_THRESH_HEURISTIC) / max(EPS, denominator_penalty) * self.lambda_g_easy_win_penalty_heuristic
                    penalty_clamped = torch.clamp(torch.tensor(penalty_val, device=device, dtype=dtype_model), 0, getattr(self.args, 'max_g_easy_win_penalty_abs', 20.0))
                    loss_g_total_micro += penalty_clamped
                    losses_g_micro['loss_g_easy_win_penalty_micro'] = penalty_clamped.detach()

            loss_g_total_scaled_for_accum_micro = loss_g_total_micro / self.grad_accum_steps
        self.scaler_enc_gen.scale(loss_g_total_scaled_for_accum_micro).backward()
        losses_g_micro['loss_recon_micro'] = loss_recon_raw.detach()
        losses_g_micro['loss_kl_micro'] = loss_kl_raw.detach()
        losses_g_micro['loss_g_adv_micro'] = loss_g_adv_raw.detach()
        losses_g_micro['loss_g_total_micro'] = loss_g_total_micro.detach()
        return losses_g_micro, recon_mel_for_log

    def _get_q_controller_data_for_heuristics(self) -> Dict[str, Any]:
        q_data: Dict[str, Any] = {'gen': {'is_valid': False}, 'active_d': {'is_valid': False}, 'lkl': {'is_valid': False}}
        controllers_map = { 'gen': self.q_controller_gen, 'active_d': self.q_controller_d_active, 'lkl': self.lambda_kl_q_controller }
        hist_names_g_d = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']
        hist_names_lkl = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']
        for key, controller in controllers_map.items():
            if controller:
                q_data[key]['is_valid'] = True
                q_data[key]['epsilon'] = controller.epsilon
                q_data[key]['on_probation'] = getattr(controller, 'on_probation', False) or getattr(controller, 'lkl_on_probation', False)
                q_data[key]['reward_median_short_term'] = np.median(list(controller.reward_hist)[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:]) if len(controller.reward_hist) >= self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS else (np.median(list(controller.reward_hist)) if controller.reward_hist else 0.0)
                current_hist_names = hist_names_g_d if key in ['gen', 'active_d'] else hist_names_lkl
                hist_prefix = "loss_" if key in ['gen', 'active_d'] else "interval_"
                for lname in current_hist_names:
                    hist_attr_name = f"{hist_prefix}{lname}_hist"
                    if hasattr(controller, hist_attr_name):
                        hist_deque = getattr(controller, hist_attr_name)
                        finite_hist_values = [v for v in hist_deque if np.isfinite(v)] if hist_deque else []
                        val_for_trend = finite_hist_values[-1] if finite_hist_values else None
                        median_val = np.median(finite_hist_values[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:]) if len(finite_hist_values) >= self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS else (np.median(finite_hist_values) if finite_hist_values else None)
                        q_data[key][f"{lname}_median_short_term"] = median_val
                        q_data[key][f"{lname}_trend_short_term"] = controller._get_trend_bin(hist_deque, val_for_trend) if val_for_trend is not None else 2 
        if q_data['gen']['is_valid'] and q_data['gen'].get('g_recon_median_short_term') is not None:
            current_feature_recon_median = q_data['gen']['g_recon_median_short_term']
            self.q_data_derived_g_recon_hist.append(current_feature_recon_median)
            self.avg_g_recon_hist_for_stagnation.append(current_feature_recon_median)
            if len(self.avg_g_recon_hist_for_stagnation) >= max(2, self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS // 2): 
                past_relevant_history = list(self.avg_g_recon_hist_for_stagnation)[:-1]
                past_recon_avg = np.mean(past_relevant_history) if len(past_relevant_history) > 1 else (past_relevant_history[0] if past_relevant_history else current_feature_recon_median)
                self.rec_features_stagnant = (past_recon_avg - current_feature_recon_median) < (abs(past_recon_avg) * self.RECON_STAGNATION_IMPROVEMENT_THRESH_REL + EPS)
            else: self.rec_features_stagnant = False
        return q_data

    def _evaluate_training_state_and_apply_heuristics(self):
        if not self.am_main_process: return
        if not self.enable_heuristic_interventions:
            if hasattr(self, 'global_step') and self.global_step > 0 and self.heuristic_check_interval > 0 and self.global_step % (self.heuristic_check_interval * 10) == 0:
                 self.logger.info(f"GStep {self.global_step}: Heuristic interventions globally DISABLED.")
            self.heuristic_vae_feature_match_active = False; self.heuristic_penalize_g_easy_win_active = False
            self.heuristic_boost_active_d_lr_active = False; self.heuristic_force_d_q_explore_active = False
            self.heuristic_override_lambda_recon_factor = 1.0; self.heuristic_override_lambda_kl_factor = 1.0
            self.heuristic_override_lambda_gan_factor = 1.0
            return
        if self.global_step == 0 or (self.heuristic_check_interval > 0 and self.global_step % (self.heuristic_check_interval * 5) == 0): # Log less frequently
            self.logger.info(f"GStep {self.global_step}: Evaluating training state for heuristics.")
        q_data = self._get_q_controller_data_for_heuristics()
        gen_q, active_d_q = q_data.get('gen', {}), q_data.get('active_d', {})
        log_msgs = []
        current_lambda_recon_factor = 1.0; current_lambda_kl_factor = 1.0; current_lambda_gan_factor = self.heuristic_override_lambda_gan_factor 
        current_boost_active_d_lr = False; current_force_d_q_explore = False; current_penalize_g_easy_win = False; current_vae_feature_match = False
        g_adv_median = gen_q.get('g_adv_median_short_term', 0.7) if gen_q.get('is_valid') else 0.7
        d_total_median = active_d_q.get('d_total_median_short_term', 0.7) if active_d_q.get('is_valid') else 0.7
        d_q_reward_median = active_d_q.get('reward_median_short_term', 0.0) if active_d_q.get('is_valid') else 0.0
        is_g_dominating_very_much = g_adv_median < self.G_VERY_MUCH_WINNING_THRESH
        is_d_very_weak = d_total_median > self.D_VERY_WEAK_THRESH
        is_d_q_learner_stagnant = d_q_reward_median < self.Q_REWARD_STAGNATION_THRESH
        is_d_strong = d_total_median < self.D_STRONG_THRESH
        is_g_stalled_adv = g_adv_median > self.G_STALLED_THRESH
        active_d_is_feature_based = "features_regional" in self.active_disc_effective_trainer_input_type
        switched_d_this_cycle = False
        if self.enable_heuristic_disc_switching:
            switched_d_this_cycle = self._check_and_perform_disc_switch(is_g_dominating_very_much, is_d_very_weak, is_d_q_learner_stagnant, is_d_strong, is_g_stalled_adv, gen_q.get('g_kl_median_short_term', 0.0) if gen_q.get('is_valid') else 0.0, log_msgs)
        if switched_d_this_cycle:
            self.consecutive_heuristic_trigger_counts = defaultdict(int); current_lambda_gan_factor = 1.0; current_lambda_recon_factor = 1.0; current_lambda_kl_factor = 1.0
        else: 
            condition_gan_rebalance = is_g_dominating_very_much and (is_d_very_weak or is_d_q_learner_stagnant) and self.rec_features_stagnant
            if condition_gan_rebalance:
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] += 1
                if self.consecutive_heuristic_trigger_counts['gan_rebalance'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_penalize_g_easy_win = True; current_lambda_recon_factor = self.args.heuristic_recon_boost_factor
                    current_lambda_gan_factor = min(current_lambda_gan_factor * 1.05, getattr(self.args, 'heuristic_max_lambda_gan_factor', 1.3))
                    if is_d_q_learner_stagnant: current_boost_active_d_lr = True; current_force_d_q_explore = True
                    log_msgs.append(f"HEURISTIC: GAN REBALANCE. PenalizeG:{current_penalize_g_easy_win}, LRecF:{current_lambda_recon_factor:.2f}, LGanF:{current_lambda_gan_factor:.2f}, D_LRB:{current_boost_active_d_lr}, D_QE:{current_force_d_q_explore}")
            else: 
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] = 0
                if current_lambda_gan_factor > 1.0: current_lambda_gan_factor = max(current_lambda_gan_factor * 0.98, 1.0) 
            condition_vae_feat_match = (active_d_is_feature_based and self.lambda_feat_match_heuristic > 0 and not is_g_dominating_very_much and not is_d_very_weak and (is_d_strong or not is_d_q_learner_stagnant) and self.rec_features_stagnant)
            if condition_vae_feat_match:
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] += 1
                if self.consecutive_heuristic_trigger_counts['vae_feat_match'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_vae_feature_match = True
                    if self.lambda_kl * self.heuristic_override_lambda_kl_factor < 1e-4 : current_lambda_kl_factor = 1.5 
                    current_lambda_gan_factor = max(current_lambda_gan_factor * 0.95, getattr(self.args, 'heuristic_min_lambda_gan_factor', 0.7))
                    log_msgs.append(f"HEURISTIC: VAE FEATURE MATCH. LKLF:{current_lambda_kl_factor:.2f}, LGanF:{current_lambda_gan_factor:.2f}")
            else: 
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] = 0
                if not active_d_is_feature_based and self.heuristic_vae_feature_match_active: log_msgs.append(f"HEURISTIC: Disabling VAE FM as Active D ('{self.active_disc_effective_trainer_input_type}') is not feature-based.")
                if current_lambda_gan_factor < 1.0: current_lambda_gan_factor = min(current_lambda_gan_factor * 1.02, 1.0) 
        self.heuristic_penalize_g_easy_win_active = current_penalize_g_easy_win; self.heuristic_override_lambda_recon_factor = current_lambda_recon_factor
        self.heuristic_boost_active_d_lr_active = current_boost_active_d_lr; self.heuristic_vae_feature_match_active = current_vae_feature_match
        self.heuristic_override_lambda_kl_factor = current_lambda_kl_factor; self.heuristic_override_lambda_gan_factor = current_lambda_gan_factor 
        if current_force_d_q_explore and self.q_controller_d_active: 
            self.q_controller_d_active.force_exploration_boost(self.heuristic_d_q_explore_duration, self.heuristic_d_q_explore_boost_epsilon)
            log_msgs.append(f"HEURISTIC: Active D Q-Ctrl exploration boosted.")
        if log_msgs and self.am_main_process: 
            for msg in log_msgs: self.logger.info(msg)

    def _check_and_perform_disc_switch(self,
                                         is_g_dominating_adv: bool, is_d_weak_overall: bool, is_d_struggling_q: bool,
                                         is_d_strong_overall: bool, is_g_stalled_adv: bool,
                                         current_g_kl_median: float,
                                         log_msgs_ref: List[str]) -> bool:
        if not self.enable_heuristic_disc_switching or self.steps_since_last_d_switch < self.disc_switch_min_steps_between: return False
        switched_this_check = False
        effective_kl_val = current_g_kl_median * self.lambda_kl * self.heuristic_override_lambda_kl_factor 
        condition_A = (is_d_strong_overall and is_g_stalled_adv and (effective_kl_val > self.KL_HIGH_THRESH * 0.1 or self.rec_features_stagnant))
        if condition_A:
            self.consecutive_trigger_primary_to_alt_count += 1
            if self.consecutive_trigger_primary_to_alt_count >= self.disc_switch_problem_state_count_thresh:
                if self.active_discriminator_key == 'primary':
                    log_msgs_ref.append(f"DISC_SWITCH: Trigger A! Switching Primary -> Alternative."); self.active_discriminator_key = 'alternative'; self._update_active_discriminator_pointers()
                    if self.q_controller_d_active: self.q_controller_d_active.reset_q_learning_state(True, True, "DSwitch P->A Heuristic A", True)
                    self.steps_since_last_d_switch = 0; self.consecutive_trigger_primary_to_alt_count = 0; self.consecutive_trigger_alt_to_primary_count = 0; switched_this_check = True
                else: self.consecutive_trigger_primary_to_alt_count = 0 
        else: self.consecutive_trigger_primary_to_alt_count = 0
        condition_B = (is_g_dominating_adv and is_d_weak_overall and (not self.rec_features_stagnant or is_d_struggling_q) ) 
        if not switched_this_check and condition_B:
            self.consecutive_trigger_alt_to_primary_count += 1
            if self.consecutive_trigger_alt_to_primary_count >= self.disc_switch_problem_state_count_thresh:
                if self.active_discriminator_key == 'alternative':
                    log_msgs_ref.append(f"DISC_SWITCH: Trigger B! Switching Alternative -> Primary."); self.active_discriminator_key = 'primary'; self._update_active_discriminator_pointers()
                    if self.q_controller_d_active: self.q_controller_d_active.reset_q_learning_state(True, True, "DSwitch A->P Heuristic B", True)
                    self.steps_since_last_d_switch = 0; self.consecutive_trigger_alt_to_primary_count = 0; self.consecutive_trigger_primary_to_alt_count = 0; switched_this_check = True
                else: self.consecutive_trigger_alt_to_primary_count = 0 
        elif not switched_this_check: self.consecutive_trigger_alt_to_primary_count = 0
        if switched_this_check:
            self.rec_features_stagnant = False; self.avg_g_recon_hist_for_stagnation.clear(); self.q_data_derived_g_recon_hist.clear()
            d_ref_switched = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
            active_d_arch_variant_switched = getattr(d_ref_switched, 'architecture_variant', 'unknown_variant')
            log_msgs_ref.append(f"  --> Post D-Switch: Heuristics reset. New active D: '{self.active_discriminator_key}' (Arch:{active_d_arch_variant_switched}, EffIn:{self.active_disc_effective_trainer_input_type}).")
        return switched_this_check

    def _load_q_state_helper_inner(self, q_controller_instance: Optional[HAKMEMQController],
                                   q_state_from_ckpt: Optional[Dict],
                                   perform_manual_flush_for_this_controller: bool,
                                   is_associated_optimizer_state_loaded: bool):
        if q_controller_instance is None: return

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
                self.lambda_kl = float(self.args.lambda_kl) 
                self.logger.info(f"CKPT not found, --reset_lkl_q_controller_on_load is True. Setting self.lambda_kl to args.lambda_kl: {self.lambda_kl:.2e}")

            for qc_obj in all_q_controllers:
                is_lkl_and_reset_lkl_arg = (qc_obj == q_ctrl_lkl and self.args.reset_lkl_q_controller_on_load)
                perform_reset_for_this_specific_controller = effective_reset_request_for_q or is_lkl_and_reset_lkl_arg

                self._load_q_state_helper_inner(qc_obj, None,
                                                perform_manual_flush_for_this_controller=perform_reset_for_this_specific_controller,
                                                is_associated_optimizer_state_loaded=False)
                if is_lkl_and_reset_lkl_arg and qc_obj is not None: 
                    qc_obj.set_current_lambda_kl(self.lambda_kl)

            if global_manual_flush_requested and not self.args.reset_q_controllers_on_load: 
                HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
            return 0, 0

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.logger.info(f"Loaded CKPT: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load CKPT {checkpoint_path}: {e}. Starting fresh.", exc_info=True)
            if self.args.reset_lkl_q_controller_on_load:
                self.lambda_kl = float(self.args.lambda_kl) 
                self.logger.info(f"CKPT load failed, --reset_lkl_q_controller_on_load is True. Setting self.lambda_kl to args.lambda_kl: {self.lambda_kl:.2e}")

            for qc_obj in all_q_controllers:
                is_lkl_and_reset_lkl_arg = (qc_obj == q_ctrl_lkl and self.args.reset_lkl_q_controller_on_load)
                perform_reset_for_this_specific_controller = effective_reset_request_for_q or is_lkl_and_reset_lkl_arg
                self._load_q_state_helper_inner(qc_obj, None,
                                                perform_manual_flush_for_this_controller=perform_reset_for_this_specific_controller,
                                                is_associated_optimizer_state_loaded=False)
                if is_lkl_and_reset_lkl_arg and qc_obj is not None:
                     qc_obj.set_current_lambda_kl(self.lambda_kl) 

            if global_manual_flush_requested and not self.args.reset_q_controllers_on_load: 
                HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False 
            return 0, 0
        
        m_load = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_primary_load = self.discriminator_primary_obj.module if self.ddp_active and hasattr(self.discriminator_primary_obj, 'module') else self.discriminator_primary_obj
        d_alt_load = self.discriminator_alternative_obj.module if self.ddp_active and hasattr(self.discriminator_alternative_obj, 'module') else self.discriminator_alternative_obj
        
        model_loaded_ok, disc_primary_model_loaded_ok, disc_alt_model_loaded_ok = False, False, False

        try:
            if 'model_state_dict' in ckpt:
                m_load.load_state_dict(ckpt['model_state_dict'], strict=False) 
                model_loaded_ok = True
                self.logger.info("Main model state_dict loaded (strict=False).")
            else: self.logger.warning("Main model_state_dict not found in checkpoint.")
        except Exception as e: self.logger.error(f"Error loading main model state_dict: {e}", exc_info=True)

        if 'discriminator_primary_state_dict' in ckpt and d_primary_load:
            try:
                d_primary_load.load_state_dict(ckpt['discriminator_primary_state_dict'], strict=False) 
                disc_primary_model_loaded_ok = True
                self.logger.info(f"Primary D state_dict loaded (strict=False).")
            except Exception as e: self.logger.error(f"Error loading D_primary state_dict: {e}", exc_info=True)
        elif not d_primary_load: self.logger.warning("discriminator_primary_obj is None, cannot load its state.")
        else: self.logger.warning("discriminator_primary_state_dict not found in checkpoint.")
        
        if 'discriminator_alternative_state_dict' in ckpt and d_alt_load:
            try:
                d_alt_load.load_state_dict(ckpt['discriminator_alternative_state_dict'], strict=False)
                disc_alt_model_loaded_ok = True
                self.logger.info(f"Alternative D state_dict loaded (strict=False).")
            except Exception as e: self.logger.error(f"Error loading D_alternative state_dict: {e}", exc_info=True)
        elif not d_alt_load: self.logger.warning("discriminator_alternative_obj is None, cannot load its state.")
        else: self.logger.warning("discriminator_alternative_state_dict not found in checkpoint.")

        self.is_val_metric_higher_better = self.args.val_primary_metric in ["avg_val_ssim_mel", "avg_val_psnr_mel"]
        default_best_val = -float('inf') if self.is_val_metric_higher_better else float('inf')
        self.best_val_metric_val = ckpt.get('best_val_metric_val', default_best_val)
        self.last_val_metrics = ckpt.get('metrics', {}).copy() if ckpt.get('metrics') is not None else {}
        
        if not self.args.reset_lkl_q_controller_on_load:
            self.lambda_kl = float(ckpt.get('current_lambda_kl', self.args.lambda_kl))
        else: 
             self.lambda_kl = float(self.args.lambda_kl) 
             self.logger.info(f"Trainer's self.lambda_kl set to args.lambda_kl: {self.lambda_kl:.2e} due to --reset_lkl_q_controller_on_load.")

        self.prev_interval_metrics_for_lambda_kl_reward = ckpt.get('prev_interval_metrics_for_lambda_kl_reward')

        loaded_gs = ckpt.get('global_step', 0)
        loaded_ep = ckpt.get('epoch', 0)
        
        next_ep_start = loaded_ep + 1 if model_loaded_ok and loaded_gs > 0 and loaded_ep < self.args.epochs else loaded_ep 
        if getattr(self.args, 'force_start_epoch_on_load', None) is not None:
            next_ep_start = self.args.force_start_epoch_on_load
            loaded_gs = getattr(self.args, 'force_start_gstep_on_load', 0 if self.args.force_start_epoch_on_load is not None else loaded_gs)
            if self.am_main_process: self.logger.info(f"CKPT Load: Overriding start epoch to {next_ep_start} and GStep to {loaded_gs} due to force_start args.")

        saved_active_disc_key = ckpt.get('active_discriminator_key', 'primary')
        saved_active_disc_effective_type = ckpt.get('active_disc_effective_trainer_input_type', 'unknown_in_ckpt')
        target_active_key_for_this_resume = saved_active_disc_key 
        forced_switch_on_resume = False

        if self.args.enable_heuristic_disc_switching and self.args.initial_disc_type:
            user_prefers_mel_like = (self.args.initial_disc_type == 'mel')
            user_prefers_feature_like = (self.args.initial_disc_type == 'dct')

            current_args_implied_active_key: Optional[str] = None
            if user_prefers_mel_like:
                if "mel" in self.primary_disc_effective_trainer_input_type: current_args_implied_active_key = 'primary'
                elif "mel" in self.alternative_disc_effective_trainer_input_type: current_args_implied_active_key = 'alternative'
            elif user_prefers_feature_like:
                if "features_regional" in self.primary_disc_effective_trainer_input_type: current_args_implied_active_key = 'primary'
                elif "features_regional" in self.alternative_disc_effective_trainer_input_type: current_args_implied_active_key = 'alternative'
            
            if current_args_implied_active_key is not None and current_args_implied_active_key != saved_active_disc_key:
                if self.am_main_process: self.logger.warning(f"LOAD_CKPT_OVERRIDE: Ckpt active D was '{saved_active_disc_key}' (Effective Type: '{saved_active_disc_effective_type}'). Current args.initial_disc_type ('{self.args.initial_disc_type}') implies '{current_args_implied_active_key}'. FORCING active D to '{current_args_implied_active_key}' for this resume.")
                target_active_key_for_this_resume = current_args_implied_active_key
                forced_switch_on_resume = True
            elif current_args_implied_active_key is None and self.am_main_process: 
                self.logger.warning(f"LOAD_CKPT_WARNING: args.initial_disc_type ('{self.args.initial_disc_type}') did not match effective types of primary ('{self.primary_disc_effective_trainer_input_type}') or alternative ('{self.alternative_disc_effective_trainer_input_type}'). Using active D key from checkpoint: '{saved_active_disc_key}'.")
        
        self.active_discriminator_key = target_active_key_for_this_resume
        self._update_active_discriminator_pointers() 

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
            self.lambda_kl = float(self.args.lambda_kl)
            q_ctrl_lkl.reset_q_learning_state(reset_q_table=True, reset_epsilon=True, context_msg="LKL Q-Ctrl Force Reset on Load by Arg", start_probation=True)
            self.logger.info(f"Trainer's self.lambda_kl (set to args value): {self.lambda_kl:.2e} after LKL Q-Ctrl reset.")
            q_ctrl_lkl.set_current_lambda_kl(self.lambda_kl) 
            self.prev_interval_metrics_for_lambda_kl_reward = None 
        else: 
            self._load_q_state_helper_inner(q_ctrl_lkl, ckpt.get('q_controller_lambda_kl_state'), effective_reset_request_for_q, True) 

        if forced_switch_on_resume:
            active_d_q_to_reset = getattr(self.optimizer_disc_active, 'q_controller', None)
            if active_d_q_to_reset:
                if self.am_main_process: self.logger.warning(f"Due to resume override, resetting Q-controller for newly FORCED active D: '{self.active_discriminator_key}' (Effective Input Type: {self.active_disc_effective_trainer_input_type}).")
                active_d_q_to_reset.reset_q_learning_state(True, True, f"Forced D switch to {self.active_discriminator_key} on Resume Override", True)
            
            self.steps_since_last_d_switch = 0
            self.consecutive_trigger_primary_to_alt_count = 0; self.consecutive_trigger_alt_to_primary_count = 0
            self.consecutive_heuristic_trigger_counts = defaultdict(int)
            self.q_data_derived_g_recon_hist.clear(); self.rec_features_stagnant = False 
            self.avg_g_recon_hist_for_stagnation.clear() 
            if self.am_main_process: self.logger.info("Heuristic switching counters and short-term recon history reset due to forced D switch on resume.")
        else: 
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

        d_ref_resume_log = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_arch_variant_resume_log = getattr(d_ref_resume_log, 'architecture_variant', 'unknown_variant')
        self.logger.info(
            f"Resuming training. GlobalStep: {loaded_gs}, NextEpochStart: {next_ep_start}. "
            f"ActiveD upon resume: '{self.active_discriminator_key}' (Arch:{active_d_arch_variant_resume_log}, EffIn:{self.active_disc_effective_trainer_input_type}). "
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
            
            state = {
                'q_table': q_ctrl_to_save.q_table, 'epsilon': q_ctrl_to_save.epsilon,
                'prev_lr_mom_state': q_ctrl_to_save.prev_lr_mom_state,
                'prev_lr_mom_action': q_ctrl_to_save.prev_lr_mom_action,
                'prev_lambda_kl_state': q_ctrl_to_save.prev_lambda_kl_state,
                'prev_lambda_kl_action': q_ctrl_to_save.prev_lambda_kl_action,
                'reward_hist': list(q_ctrl_to_save.reward_hist),
                'q_table_access_count': dict(q_ctrl_to_save.q_table_access_count),
                'q_table_creation_time': q_ctrl_to_save.q_table_creation_time,
                'q_table_last_access_time': q_ctrl_to_save.q_table_last_access_time,
                'on_probation': getattr(q_ctrl_to_save, 'on_probation', False),
                'current_probation_step': getattr(q_ctrl_to_save, 'current_probation_step', 0),
                'lkl_on_probation': getattr(q_ctrl_to_save, 'lkl_on_probation', False),
                'lkl_current_probation_step': getattr(q_ctrl_to_save, 'lkl_current_probation_step', 0)
            }
            if hasattr(q_ctrl_to_save, 'loss_g_total_hist'):
                q_hist_names = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']
                state['loss_histories'] = {
                    hname: list(getattr(q_ctrl_to_save, f"loss_{hname}_hist")) 
                    for hname in q_hist_names if hasattr(q_ctrl_to_save, f"loss_{hname}_hist")
                }
            if hasattr(q_ctrl_to_save, 'interval_avg_recon_hist'): 
                q_lkl_hist_names = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']
                state['interval_histories'] = {
                    hname: list(getattr(q_ctrl_to_save, f"interval_{hname}_hist")) 
                    for hname in q_lkl_hist_names if hasattr(q_ctrl_to_save, f"interval_{hname}_hist")
                }
            return state

        data_to_save = {
            'global_step': self.global_step, 
            'epoch': self.current_epoch,
            'model_state_dict': m_s.state_dict(),
            'discriminator_primary_state_dict': d_primary_s.state_dict(),
            'discriminator_alternative_state_dict': d_alt_s.state_dict(),
            'active_discriminator_key': self.active_discriminator_key,
            'active_disc_effective_trainer_input_type': self.active_disc_effective_trainer_input_type, 
            'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict() if self.optimizer_enc_gen else None,
            'optimizer_disc_primary_state_dict': self.optimizer_disc_primary.state_dict(),
            'optimizer_disc_alternative_state_dict': self.optimizer_disc_alternative.state_dict(),
            'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict(),
            'scaler_disc_state_dict': self.scaler_disc.state_dict(),
            'args': vars(self.args), 
            'metrics': metrics if metrics is not None else self.last_val_metrics.copy(),
            'best_val_metric_val': self.best_val_metric_val, 
            'current_lambda_kl': self.lambda_kl,
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
        
        data_to_save['q_controller_enc_gen_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_gen)
        data_to_save['q_controller_disc_primary_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_d_primary)
        data_to_save['q_controller_disc_alternative_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_d_alt)
        data_to_save['q_controller_lambda_kl_state'] = get_q_state_from_controller_or_optimizer(self.lambda_kl_q_controller)

        fprefix = "wubuspectrans_ckpt_v011"
        if is_best: 
            fp_str = f"{fprefix}_best_ep{self.current_epoch + 1}_step{self.global_step}.pt" 
        elif is_intermediate: 
            fp_str = f"{fprefix}_step{self.global_step}.pt"
        else: 
            fp_str = f"{fprefix}_ep{self.current_epoch + 1}_step{self.global_step}.pt"
            
        fp = Path(self.args.checkpoint_dir) / fp_str
        try:
            torch.save(data_to_save, fp)
            self.logger.info(f"Checkpoint saved: {fp.name}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint {fp}: {e}", exc_info=True)

    def train(self, start_epoch: int = 0, initial_global_step: int = 0):
        self.global_step = initial_global_step
        self.current_epoch = start_epoch
        if self.am_main_process:
            d_ref_active_log = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
            active_d_arch_variant_log = getattr(d_ref_active_log, 'architecture_variant', 'unknown_variant')
            self.logger.info(f"Starting training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch}, InitialGStep: {initial_global_step}. Initial Active D: {self.active_discriminator_key} (Arch: {active_d_arch_variant_log}, EffIn: {self.active_disc_effective_trainer_input_type})")
        if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and self.args.num_val_samples_to_log > 0 and self.fixed_noise_for_sampling is None:
            m_ref_temp = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
            default_dtype = next(iter(m_ref_temp.parameters()), torch.tensor(0.0)).dtype if hasattr(m_ref_temp, 'parameters') and next(m_ref_temp.parameters(), None) is not None else torch.float32
            if self.args.latent_dim > 0: self.fixed_noise_for_sampling = torch.randn(self.args.num_val_samples_to_log, self.args.latent_dim, device=self.device, dtype=default_dtype)

        log_interval_accum_losses: Dict[str, float] = defaultdict(float)
        log_interval_items_processed = 0
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        all_q_controllers_to_sync_lkl = [self.q_controller_gen, self.q_controller_d_primary, self.q_controller_d_alt, self.lambda_kl_q_controller]
        for q_ctrl in all_q_controllers_to_sync_lkl:
            if q_ctrl and hasattr(q_ctrl, 'set_current_lambda_kl'): q_ctrl.set_current_lambda_kl(self.lambda_kl)

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process:
                 d_ref_active_ep_log = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                 active_d_arch_variant_ep_log = getattr(d_ref_active_ep_log, 'architecture_variant', 'unknown_variant')
                 self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting (L_KL: {self.lambda_kl:.3e}*KLF:{self.heuristic_override_lambda_kl_factor:.2f}, LRecF: {self.heuristic_override_lambda_recon_factor:.2f}, LGanF: {self.heuristic_override_lambda_gan_factor:.2f}, ActD: {self.active_discriminator_key} [Arch:{active_d_arch_variant_ep_log}, EffIn:{self.active_disc_effective_trainer_input_type}]).")
            if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler): self.train_loader.sampler.set_epoch(epoch)
            m_ref.train(); self.active_discriminator.train()
            num_batches_epoch = len(self.train_loader)
            prog_bar = tqdm(self.train_loader, desc=f"E{epoch+1}", disable=not self.am_main_process, dynamic_ncols=True, total=num_batches_epoch)

            for batch_idx, batch_data_dict in enumerate(prog_bar):
                batch_real_mel_spectrograms = batch_data_dict['mel'].to(self.device)
                batch_real_raw_audio: Optional[List[Optional[Union[torch.Tensor, np.ndarray]]]] = None
                if 'raw_audio' in batch_data_dict:
                    raw_audio_from_loader = batch_data_dict['raw_audio']
                    if isinstance(raw_audio_from_loader, list): # Expected: list of Tensors or np.arrays
                        batch_real_raw_audio = raw_audio_from_loader
                    elif isinstance(raw_audio_from_loader, torch.Tensor) and raw_audio_from_loader.ndim >=2 : # Collated Tensor
                        batch_real_raw_audio = [raw_audio_from_loader[i] for i in range(raw_audio_from_loader.shape[0])]
                    elif isinstance(raw_audio_from_loader, np.ndarray) and raw_audio_from_loader.ndim >=2: # Collated ndarray
                        batch_real_raw_audio = [raw_audio_from_loader[i] for i in range(raw_audio_from_loader.shape[0])]
                    else: # Single item not in a list (should not happen with DataLoader typically)
                        batch_real_raw_audio = [raw_audio_from_loader]

                batch_size_micro = batch_real_mel_spectrograms.size(0)
                self.steps_since_last_d_switch +=1

                losses_d_micro = self._train_discriminator_step(batch_real_mel_spectrograms, batch_real_raw_audio, m_ref)
                for k, v_tensor in losses_d_micro.items():
                    if torch.isfinite(v_tensor):
                        val = v_tensor.item(); accum_key = k.replace('_micro', '_agg')
                        log_interval_accum_losses[accum_key] += val * batch_size_micro
                        if k == 'loss_d_total_micro': self.interval_metrics_accum['d_total'] += val

                losses_g_micro, recon_mel_for_logging = self._train_generator_step(batch_real_mel_spectrograms, batch_real_raw_audio, m_ref)
                for k, v_tensor in losses_g_micro.items():
                    if torch.isfinite(v_tensor):
                        val = v_tensor.item(); accum_key = k.replace('_micro', '_agg')
                        log_interval_accum_losses[accum_key] += val * batch_size_micro
                        if k == 'loss_recon_micro': self.interval_metrics_accum['recon_features'] += val
                        elif k == 'loss_kl_micro':  self.interval_metrics_accum['kl_div'] += val
                        elif k == 'loss_g_feat_match_micro': log_interval_accum_losses['loss_g_feat_match_eff_contrib_agg'] += (self.lambda_feat_match_heuristic * val * batch_size_micro)
                        elif k == 'loss_g_easy_win_penalty_micro': log_interval_accum_losses['loss_g_easy_win_penalty_eff_contrib_agg'] += (val * batch_size_micro)

                log_interval_items_processed += batch_size_micro
                self.interval_steps_count += 1

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.optimizer_disc_active:
                        if hasattr(self.optimizer_disc_active, 'grad_stats'): self.optimizer_disc_active.grad_stats.finalize_step_stats(sum(p.numel() for grp in self.optimizer_disc_active.param_groups for p in grp['params'] if p.requires_grad))
                        if self.args.global_max_grad_norm > 0:
                            self.scaler_disc.unscale_(self.optimizer_disc_active)
                            d_to_clip = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                            torch.nn.utils.clip_grad_norm_(d_to_clip.parameters(), self.args.global_max_grad_norm)
                        self.scaler_disc.step(self.optimizer_disc_active); self.scaler_disc.update(); self.optimizer_disc_active.zero_grad(set_to_none=True)
                    if self.optimizer_enc_gen:
                        if hasattr(self.optimizer_enc_gen, 'grad_stats'): self.optimizer_enc_gen.grad_stats.finalize_step_stats(sum(p.numel() for grp in self.optimizer_enc_gen.param_groups for p in grp['params'] if p.requires_grad))
                        if self.args.global_max_grad_norm > 0:
                            self.scaler_enc_gen.unscale_(self.optimizer_enc_gen)
                            torch.nn.utils.clip_grad_norm_(m_ref.parameters(), self.args.global_max_grad_norm)
                        self.scaler_enc_gen.step(self.optimizer_enc_gen); self.scaler_enc_gen.update(); self.optimizer_enc_gen.zero_grad(set_to_none=True)
                    self.global_step += 1

                    avg_losses_for_q = { k.replace('_agg',''): v_sum / (log_interval_items_processed if log_interval_items_processed > 0 else 1)
                                         for k,v_sum in log_interval_accum_losses.items() if '_agg' in k and 'eff_contrib' not in k}
                    for base_loss_key in ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total', 'loss_d_real', 'loss_d_fake']:
                        if base_loss_key not in avg_losses_for_q: avg_losses_for_q[base_loss_key] = 0.0

                    for k_q_check, v_q_check in avg_losses_for_q.items():
                        if not np.isfinite(v_q_check): avg_losses_for_q[k_q_check] = 1.0

                    if self.q_controller_d_active and hasattr(self.optimizer_disc_active, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_disc_active.q_controller_update_and_set_hyperparams(avg_losses_for_q, self.lambda_kl * self.heuristic_override_lambda_kl_factor)
                        if self.heuristic_boost_active_d_lr_active and self.optimizer_disc_active:
                            for group in self.optimizer_disc_active.param_groups: group['lr'] = float(np.clip(group['lr'] * self.heuristic_active_d_lr_boost_factor, 1e-8, 1.0))
                    if self.q_controller_gen and hasattr(self.optimizer_enc_gen, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_enc_gen.q_controller_update_and_set_hyperparams(avg_losses_for_q, self.lambda_kl * self.heuristic_override_lambda_kl_factor)

                    if self.global_step > 0 and self.heuristic_check_interval > 0 and self.global_step % self.heuristic_check_interval == 0:
                        self._evaluate_training_state_and_apply_heuristics()
                    if self.lambda_kl_q_controller is not None and self.lambda_kl_update_interval > 0 and self.global_step > 0 and self.global_step % self.lambda_kl_update_interval == 0 and self.interval_steps_count > 0:
                        current_interval_metrics: Dict[str, Union[float, None]] = {
                            'avg_recon': self.interval_metrics_accum['recon_features'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_kl_div': self.interval_metrics_accum['kl_div'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_d_total': self.interval_metrics_accum['d_total'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'val_metric': self.last_val_metrics.get(self.args.val_primary_metric), 'current_lambda_kl_val': self.lambda_kl }
                        if self.am_main_process: prog_bar.write(f"GStep {self.global_step}: LKL Q-Ctrl. LKL_Base: {self.lambda_kl:.4e}. Metrics: { {k: f'{v:.3f}' if isinstance(v,float) else v for k,v in current_interval_metrics.items()} }")
                        q_state_lambda_kl = self.lambda_kl_q_controller.get_lambda_kl_state(current_interval_metrics)
                        if self.lambda_kl_q_controller.prev_lambda_kl_state is not None and self.lambda_kl_q_controller.prev_lambda_kl_action is not None and q_state_lambda_kl is not None and self.prev_interval_metrics_for_lambda_kl_reward is not None:
                            reward_for_lambda_kl = self.lambda_kl_q_controller.compute_lambda_kl_reward(current_interval_metrics, self.prev_interval_metrics_for_lambda_kl_reward)
                            if self.am_main_process: prog_bar.write(f"  LKL Q-Ctrl reward: {reward_for_lambda_kl:.3f}")
                            self.lambda_kl_q_controller.update_q_values(self.lambda_kl_q_controller.prev_lambda_kl_state, self.lambda_kl_q_controller.prev_lambda_kl_action, reward_for_lambda_kl, q_state_lambda_kl, mode='lambda_kl')
                        elif q_state_lambda_kl is not None and hasattr(self.lambda_kl_q_controller, 'set_initial_lambda_kl_metrics'): self.lambda_kl_q_controller.set_initial_lambda_kl_metrics(current_interval_metrics)
                        if q_state_lambda_kl is not None:
                            lambda_kl_action_dict = self.lambda_kl_q_controller.choose_action(q_state_lambda_kl, mode='lambda_kl')
                            chosen_scale = HybridTrainer.get_scale_from_action_value(lambda_kl_action_dict, 'lambda_kl_scale', 1.0)
                            self.lambda_kl = float(np.clip(self.lambda_kl * chosen_scale, self.min_lambda_kl_q_control, self.max_lambda_kl_q_control))
                            if self.am_main_process: prog_bar.write(f"  LKL_Q-Ctrl CHOSE scale: {chosen_scale:.3f}. New LKL: {self.lambda_kl:.4e}")
                            self.lambda_kl_q_controller.prev_lambda_kl_state = q_state_lambda_kl; self.lambda_kl_q_controller.prev_lambda_kl_action = lambda_kl_action_dict
                        self.prev_interval_metrics_for_lambda_kl_reward = current_interval_metrics.copy()
                        for q_ctrl_opt in all_q_controllers_to_sync_lkl:
                            if q_ctrl_opt and hasattr(q_ctrl_opt, 'set_current_lambda_kl'): q_ctrl_opt.set_current_lambda_kl(self.lambda_kl)
                        self.interval_metrics_accum = defaultdict(float); self.interval_steps_count = 0

                    if self.global_step > 0 and self.args.log_interval > 0 and (self.global_step % self.args.log_interval == 0) and log_interval_items_processed > 0 and self.am_main_process:
                        current_log_metrics_wandb: Dict[str, Any] = {}
                        for k, v_sum in log_interval_accum_losses.items():
                            key_suffix = "_eff_contrib" if "_eff_contrib_agg" in k else ""
                            wandb_key = f"train/{k.replace('_eff_contrib_agg', key_suffix).replace('_agg', '')}"
                            current_log_metrics_wandb[wandb_key] = v_sum / log_interval_items_processed

                        avg_raw_recon_feat = current_log_metrics_wandb.get('train/loss_recon', 0.0)
                        avg_raw_kl = current_log_metrics_wandb.get('train/loss_kl', 0.0)
                        avg_raw_g_adv = current_log_metrics_wandb.get('train/loss_g_adv', 0.0)
                        avg_raw_d_real = current_log_metrics_wandb.get('train/loss_d_real', 0.0)
                        avg_raw_d_fake = current_log_metrics_wandb.get('train/loss_d_fake', 0.0)
                        avg_raw_d_total = current_log_metrics_wandb.get('train/loss_d_total', 0.0)

                        eff_recon = avg_raw_recon_feat * self.lambda_recon * self.heuristic_override_lambda_recon_factor
                        eff_kl = avg_raw_kl * self.lambda_kl * self.heuristic_override_lambda_kl_factor
                        eff_gan = avg_raw_g_adv * self.lambda_gan * self.heuristic_override_lambda_gan_factor

                        current_log_metrics_wandb.update({
                            "train/lambda_recon_feat_eff_contrib": eff_recon,
                            "train/lambda_kl_eff_contrib": eff_kl,
                            "train/lambda_gan_eff_contrib": eff_gan,
                            "train/lambda_kl_base_from_q_lkl": self.lambda_kl,
                            "train/loss_d_real_raw": avg_raw_d_real, # Already in current_log_metrics_wandb from loop if present
                            "train/loss_d_fake_raw": avg_raw_d_fake, # Already in current_log_metrics_wandb from loop if present
                        })

                        loss_g_feat_match_contrib = current_log_metrics_wandb.get('train/loss_g_feat_match_eff_contrib', 0.0)
                        loss_g_easy_win_penalty_contrib = current_log_metrics_wandb.get('train/loss_g_easy_win_penalty_eff_contrib', 0.0)
                        calculated_g_total_for_log = eff_recon + eff_kl + eff_gan + loss_g_feat_match_contrib + loss_g_easy_win_penalty_contrib
                        current_log_metrics_wandb["train/loss_g_total_calculated_from_components"] = calculated_g_total_for_log

                        lr_g = self.optimizer_enc_gen.param_groups[0]['lr'] if self.optimizer_enc_gen else -1.0
                        lr_d_active = self.optimizer_disc_active.param_groups[0]['lr'] if self.optimizer_disc_active else -1.0
                        d_ref_log_console = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                        active_d_arch_variant_console_log = getattr(d_ref_log_console, 'architecture_variant', 'unk')[:3]
                        active_d_eff_in_console_log = self.active_disc_effective_trainer_input_type.split('_')[0][:3]

                        current_log_metrics_wandb.update({
                            "train/lr_gen": lr_g,
                            f"train/lr_disc_{self.active_discriminator_key}_{active_d_arch_variant_console_log}_{active_d_eff_in_console_log}": lr_d_active,
                            "epoch_frac": epoch + ((batch_idx + 1) / max(1, num_batches_epoch)),
                            "global_step": self.global_step,
                            f"active_disc_is_primary_val": 1 if self.active_discriminator_key == 'primary' else 0,
                            f"active_disc_is_mel_val": 1 if "mel" in self.active_disc_effective_trainer_input_type else 0
                        })

                        q_controller_info_map_log = {
                            'gen':self.q_controller_gen,
                            'd_pri':self.q_controller_d_primary,
                            'd_alt':self.q_controller_d_alt,
                            'lkl':self.lambda_kl_q_controller
                        }
                        for prefix_log, controller_obj_log in q_controller_info_map_log.items():
                            if controller_obj_log and hasattr(controller_obj_log, 'get_info'):
                                for k_info_log, v_info_log in controller_obj_log.get_info().items():
                                    clean_k_info_log = ''.join(c if c.isalnum() or c in ['_', '/'] else '_' for c in str(k_info_log)).lower().replace('lrmom','').replace('lambdakl','')
                                    current_log_metrics_wandb[f"q_info/{prefix_log}/{clean_k_info_log}"] = v_info_log
                                if hasattr(controller_obj_log, 'prev_lr_mom_action') and controller_obj_log.prev_lr_mom_action:
                                    current_log_metrics_wandb[f"q_actions/{prefix_log}/lr_scale"] = HybridTrainer.get_scale_from_action_value(controller_obj_log.prev_lr_mom_action, 'lr_scale')
                                    current_log_metrics_wandb[f"q_actions/{prefix_log}/mom_scale"] = HybridTrainer.get_scale_from_action_value(controller_obj_log.prev_lr_mom_action, 'momentum_scale')
                                if hasattr(controller_obj_log, 'prev_lambda_kl_action') and controller_obj_log.prev_lambda_kl_action:
                                    current_log_metrics_wandb[f"q_actions/{prefix_log}/lkl_scale"] = HybridTrainer.get_scale_from_action_value(controller_obj_log.prev_lambda_kl_action, 'lambda_kl_scale')

                        current_log_metrics_wandb.update({
                            "heuristic/vae_fm_active_val": 1 if self.heuristic_vae_feature_match_active else 0,
                            "heuristic/pen_g_ez_win_val": 1 if self.heuristic_penalize_g_easy_win_active else 0,
                            "heuristic/d_lr_boost_active_val": 1 if self.heuristic_boost_active_d_lr_active else 0,
                            "heuristic/lrec_factor_val": self.heuristic_override_lambda_recon_factor,
                            "heuristic/lkl_factor_val": self.heuristic_override_lambda_kl_factor,
                            "heuristic/lgan_factor_val": self.heuristic_override_lambda_gan_factor,
                            "heuristic/lambda_feat_match_heuristic_val": self.lambda_feat_match_heuristic if self.heuristic_vae_feature_match_active else 0.0,
                            "heuristic/lambda_g_easy_win_penalty_heuristic_val": self.lambda_g_easy_win_penalty_heuristic if self.heuristic_penalize_g_easy_win_active else 0.0,
                            "heuristic/active_d_lr_boost_factor_applied": self.heuristic_active_d_lr_boost_factor if self.heuristic_boost_active_d_lr_active else 1.0,
                            "heuristic/d_q_explore_active_val": 1 if (self.q_controller_d_active and hasattr(self.q_controller_d_active, 'epsilon_boost_active_steps') and self.q_controller_d_active.epsilon_boost_active_steps > 0) else 0,
                            "heuristic/rec_features_stagnant_val": 1 if self.rec_features_stagnant else 0,
                        })

                        current_log_metrics_wandb["disc_switch/steps_since_last_switch"] = self.steps_since_last_d_switch
                        current_log_metrics_wandb["disc_switch/p_to_a_trigger_count"] = self.consecutive_trigger_primary_to_alt_count
                        current_log_metrics_wandb["disc_switch/a_to_p_trigger_count"] = self.consecutive_trigger_alt_to_primary_count
                        for k_heur_trig, v_heur_trig in self.consecutive_heuristic_trigger_counts.items():
                            current_log_metrics_wandb[f"heuristic/trigger_count/{k_heur_trig}"] = v_heur_trig

                        if self.global_step % (self.args.log_interval * 5) == 0:
                            if self.optimizer_enc_gen and hasattr(self.optimizer_enc_gen, 'get_gradient_stats_summary_optimizer_view'):
                                current_log_metrics_wandb.update({f"grad_stats/gen/{k}": v for k, v in self.optimizer_enc_gen.get_gradient_stats_summary_optimizer_view().items()})
                            if self.optimizer_disc_active and hasattr(self.optimizer_disc_active, 'get_gradient_stats_summary_optimizer_view'):
                                current_log_metrics_wandb.update({f"grad_stats/d_active_{self.active_discriminator_key}/{k}": v for k, v in self.optimizer_disc_active.get_gradient_stats_summary_optimizer_view().items()})

                        gen_q_eps_str = f"{self.q_controller_gen.epsilon:.2f}" if self.q_controller_gen else "N/A"
                        d_act_q_eps_str = f"{self.q_controller_d_active.epsilon:.2f}" if self.q_controller_d_active else "N/A"
                        lkl_q_eps_str = f"{self.lambda_kl_q_controller.epsilon:.2f}" if self.lambda_kl_q_controller else "N/A"

                        gen_q_last_lr_scale = HybridTrainer.get_scale_from_action_value(getattr(self.q_controller_gen, 'prev_lr_mom_action', None), 'lr_scale', -1.0)
                        d_q_last_lr_scale = HybridTrainer.get_scale_from_action_value(getattr(self.q_controller_d_active, 'prev_lr_mom_action', None), 'lr_scale', -1.0)
                        lkl_q_last_lkl_scale = HybridTrainer.get_scale_from_action_value(getattr(self.lambda_kl_q_controller, 'prev_lambda_kl_action', None), 'lambda_kl_scale', -1.0)

                        q_scales_str = f"QSc(G:{gen_q_last_lr_scale:.1f}|D:{d_q_last_lr_scale:.1f}|LKL:{lkl_q_last_lkl_scale:.1f})"

                        heuristic_flags_short = []
                        if self.heuristic_vae_feature_match_active: heuristic_flags_short.append(f"FM(x{self.lambda_feat_match_heuristic:.1f})")
                        if self.heuristic_penalize_g_easy_win_active: heuristic_flags_short.append(f"GPEW(x{self.lambda_g_easy_win_penalty_heuristic:.1f})")
                        if self.heuristic_boost_active_d_lr_active: heuristic_flags_short.append(f"DLRB(x{self.heuristic_active_d_lr_boost_factor:.1f})")
                        if self.q_controller_d_active and hasattr(self.q_controller_d_active, 'epsilon_boost_active_steps') and self.q_controller_d_active.epsilon_boost_active_steps > 0:
                            heuristic_flags_short.append(f"DQE({self.q_controller_d_active.epsilon_boost_active_steps})")
                        if self.rec_features_stagnant: heuristic_flags_short.append("RecStag")
                        heur_flags_console_str = f"H:[{','.join(heuristic_flags_short)}]" if heuristic_flags_short else ""

                        lambda_factors_str = f"LF(Rec:{self.heuristic_override_lambda_recon_factor:.1f}|KL:{self.heuristic_override_lambda_kl_factor:.1f}|GAN:{self.heuristic_override_lambda_gan_factor:.1f})"

                        log_str_console = (
                            f"E{epoch+1} S{self.global_step} ActD:{self.active_discriminator_key[0]}:{active_d_arch_variant_console_log}:{active_d_eff_in_console_log} "
                            f"| G_Tot:{calculated_g_total_for_log:.2f}="
                            f"(R:{eff_recon:.2f}[{avg_raw_recon_feat:.2f}] "
                            f"K:{eff_kl:.2f}[{avg_raw_kl:.2f}] "
                            f"A:{eff_gan:.2f}[{avg_raw_g_adv:.2f}]"
                            + (f" FM:{loss_g_feat_match_contrib:.2f}" if self.heuristic_vae_feature_match_active and loss_g_feat_match_contrib!=0 else "")
                            + (f" GPen:{loss_g_easy_win_penalty_contrib:.2f}" if self.heuristic_penalize_g_easy_win_active and loss_g_easy_win_penalty_contrib!=0 else "")
                            + f") | D_Tot:{avg_raw_d_total:.2f}="
                            f"(Rl:{avg_raw_d_real:.2f} Fk:{avg_raw_d_fake:.2f})"
                            f" | LR(G/D):{lr_g:.1e}/{lr_d_active:.1e} | LKL:{self.lambda_kl:.2e} "
                            f"Qε(G:{gen_q_eps_str} D:{d_act_q_eps_str} L:{lkl_q_eps_str}) {q_scales_str} "
                            f"DSw(Trig P>A:{self.consecutive_trigger_primary_to_alt_count},A>P:{self.consecutive_trigger_alt_to_primary_count}|Steps:{self.steps_since_last_d_switch}) "
                            f"{lambda_factors_str} {heur_flags_console_str}"
                        )

                        prog_bar.set_postfix_str(f"ActD:{self.active_discriminator_key[0]} G:{calculated_g_total_for_log:.2f} D:{avg_raw_d_total:.2f} RecFRaw:{avg_raw_recon_feat:.3f}", refresh=True)
                        prog_bar.write(log_str_console)
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(current_log_metrics_wandb, step=self.global_step)

                        log_interval_accum_losses = defaultdict(float); log_interval_items_processed = 0

                    if recon_mel_for_logging is not None and self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and self.global_step > 0 and ((self.global_step) % self.args.wandb_log_train_recon_interval == 0):
                        self._log_samples_to_wandb("train_recon_mel", recon_mel_for_logging, self.args.num_val_samples_to_log)
                        if self.global_step % (self.args.wandb_log_train_recon_interval * getattr(self.args, 'train_target_log_freq_multiplier', 5)) == 0 :
                           self._log_samples_to_wandb("train_target_mel", batch_real_mel_spectrograms, self.args.num_val_samples_to_log)
                    if self.fixed_noise_for_sampling is not None and self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and self.global_step > 0 and (self.global_step % self.args.wandb_log_fixed_noise_samples_interval == 0):
                        fixed_noise_mels = self.sample(self.args.num_val_samples_to_log, noise=self.fixed_noise_for_sampling)
                        if fixed_noise_mels is not None: self._log_samples_to_wandb("fixed_noise_generated_mel", fixed_noise_mels, self.args.num_val_samples_to_log)
                    if self.args.save_interval > 0 and self.global_step > 0 and (self.global_step % self.args.save_interval == 0) and self.am_main_process:
                        self._save_checkpoint(is_intermediate=True, metrics=avg_losses_for_q if 'avg_losses_for_q' in locals() else None)

            validation_interval_epochs = getattr(self.args, 'validation_interval_epochs', 1)
            if self.val_loader and self.am_main_process and validation_interval_epochs > 0 and (epoch + 1) % validation_interval_epochs == 0:
                val_metrics_eoe = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if val_metrics_eoe:
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log({f"val/{k_val}": v_val for k_val, v_val in val_metrics_eoe.items()}, step=self.global_step)
                    metric_to_check_eoe = self.args.val_primary_metric
                    current_val_for_best_eoe: float = val_metrics_eoe.get(metric_to_check_eoe, self.best_val_metric_val)
                    is_better_eoe = (current_val_for_best_eoe > self.best_val_metric_val) if self.is_val_metric_higher_better else (current_val_for_best_eoe < self.best_val_metric_val)
                    if is_better_eoe and np.isfinite(current_val_for_best_eoe):
                        prog_bar.write(f"New best val metric ({metric_to_check_eoe}): {current_val_for_best_eoe:.4f} (prev: {self.best_val_metric_val:.4f}). Saving.")
                        self.best_val_metric_val = current_val_for_best_eoe
                        self._save_checkpoint(is_best=True, metrics=val_metrics_eoe)
            save_epoch_interval_epochs = getattr(self.args, 'save_epoch_interval', 1)
            if self.am_main_process and save_epoch_interval_epochs > 0 and (epoch + 1) % save_epoch_interval_epochs == 0:
                already_saved_as_best_this_epoch = 'is_better_eoe' in locals() and locals().get('is_better_eoe', False) and np.isfinite(locals().get('current_val_for_best_eoe', float('inf')))
                is_last_grad_accum_step_of_epoch = (batch_idx +1) == num_batches_epoch and (batch_idx +1) % self.grad_accum_steps == 0
                already_saved_as_intermediate_this_step = self.args.save_interval > 0 and self.global_step > 0 and self.global_step % self.args.save_interval == 0 and is_last_grad_accum_step_of_epoch
                if not (already_saved_as_best_this_epoch or already_saved_as_intermediate_this_step):
                    eoe_metrics_for_save = self.last_val_metrics.copy() if self.last_val_metrics else {}
                    if 'avg_losses_for_q' in locals() and isinstance(locals()['avg_losses_for_q'], dict):
                        eoe_metrics_for_save["epoch_end_train_g_total_approx"] = locals()['avg_losses_for_q'].get('loss_g_total', -1.0)
                        eoe_metrics_for_save["epoch_end_train_d_total_approx"] = locals()['avg_losses_for_q'].get('loss_d_total', -1.0)
                    self._save_checkpoint(metrics=eoe_metrics_for_save)


    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Optional[Dict[str, float]]:
        if not self.val_loader or not self.am_main_process: return None
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        
        original_training_mode_m = m_ref.training
        m_ref.eval() 

        total_recon_features_mse_sum = 0.0; total_mel_mse_sum = 0.0; total_psnr_mel_sum = 0.0
        total_ssim_mel_sum = 0.0; total_lpips_mel_sum = 0.0; total_items_evaluated = 0
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0)).dtype if hasattr(m_ref, 'parameters') and next(m_ref.parameters(), None) is not None else torch.float32
        logged_samples_count_this_val = 0

        for batch_idx_val, batch_data_dict_val in enumerate(
            tqdm(self.val_loader, desc="Validating", disable=(not self.am_main_process or os.getenv('CI') == 'true' or getattr(self.args, 'disable_val_tqdm', False)), dynamic_ncols=True)
        ):
            real_mel_segments = batch_data_dict_val['mel'].to(self.device, dtype=dtype_m)
            B, _, H_mel, W_mel = real_mel_segments.shape

            recon_regional_features, _, _, gaad_bboxes_from_enc, target_regional_features_for_loss = m_ref(real_mel_segments)
            loss_recon_features_batch = self._compute_recon_loss(recon_regional_features, target_regional_features_for_loss)
            if torch.isfinite(loss_recon_features_batch): total_recon_features_mse_sum += loss_recon_features_batch.item() * B
            
            unnormalized_coeffs_for_assembly_val: torch.Tensor
            if self.vae_transform_type == 'complex_dft_ri':
                unnormalized_coeffs_for_assembly_val = AudioSpecGenerator._unnormalize_and_reconstruct_coeffs_to_complex_dft(
                    recon_regional_features, self.args, self.region_proc_size_tuple
                )
            elif self.vae_transform_type == 'dct':
                 unnormalized_coeffs_for_assembly_val = AudioSpecGenerator._unnormalize_dct_coeffs(
                    recon_regional_features, self.args, self.region_proc_size_tuple
                )
            else:
                self.logger.error(f"Validate: Unknown VAE transform {self.vae_transform_type} for assembly.")
                continue # Skip this batch if transform type is unknown

            recon_mel_assembled = HybridTrainer._assemble_mel_from_transformed_coeffs_regions(
                unnormalized_coeffs_for_assembly_val, gaad_bboxes_from_enc, 
                real_mel_segments.shape, self.args, self.region_proc_size_tuple
            )
            
            if recon_mel_assembled.shape == real_mel_segments.shape:
                loss_mel_mse_batch = F.mse_loss(recon_mel_assembled, real_mel_segments, reduction='mean')
                if torch.isfinite(loss_mel_mse_batch):
                    total_mel_mse_sum += loss_mel_mse_batch.item() * B
                    mse_val = loss_mel_mse_batch.item(); psnr_val = 10 * math.log10(1.0 / (mse_val + EPS)) if mse_val > EPS else 100.0 
                    total_psnr_mel_sum += psnr_val * B
                
                recon_mel_01 = (recon_mel_assembled.clamp(-1,1)+1)/2.0
                real_mel_01 = (real_mel_segments.clamp(-1,1)+1)/2.0
                if self.ssim_metric:
                    try: total_ssim_mel_sum += self.ssim_metric(recon_mel_01, real_mel_01).item() * B
                    except Exception as e_ssim: self.logger.debug(f"Val SSIM failed: {e_ssim}")
                if self.lpips_loss_fn:
                    try:
                        rec_lpips_in = recon_mel_assembled.repeat(1,3,1,1) if recon_mel_assembled.shape[1]==1 else recon_mel_assembled
                        real_lpips_in = real_mel_segments.repeat(1,3,1,1) if real_mel_segments.shape[1]==1 else real_mel_segments
                        total_lpips_mel_sum += self.lpips_loss_fn(rec_lpips_in, real_lpips_in).sum().item() 
                    except Exception as e_lpips: self.logger.debug(f"Val LPIPS failed: {e_lpips}")
            
            total_items_evaluated += B
            if logged_samples_count_this_val < num_val_samples_to_log and self.args.wandb and WANDB_AVAILABLE and wandb.run: 
                num_to_log_now = min(B, num_val_samples_to_log - logged_samples_count_this_val)
                if num_to_log_now > 0:
                    self._log_samples_to_wandb("val_recon_mel", recon_mel_assembled[:num_to_log_now], num_to_log_now)
                    self._log_samples_to_wandb("val_target_mel", real_mel_segments[:num_to_log_now], num_to_log_now)
                logged_samples_count_this_val += num_to_log_now
        
        m_ref.train(original_training_mode_m) 
        if total_items_evaluated == 0: return None
        metrics = {
            "avg_val_recon_features_mse": total_recon_features_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_mel_mse": total_mel_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_psnr_mel": total_psnr_mel_sum / total_items_evaluated if total_items_evaluated > 0 else 0.0,
            "avg_val_ssim_mel": total_ssim_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.ssim_metric else 0.0,
            "avg_val_lpips_mel": total_lpips_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.lpips_loss_fn else float('inf') }
        self.last_val_metrics = metrics
        self.logger.info(f"Validation Metrics (Ep {self.current_epoch+1}, GStep {self.global_step}, ActiveD: {self.active_discriminator_key}): " + ", ".join([f"{k}:{v:.4f}" for k,v in metrics.items()]))
        return metrics

    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        
        original_mode_m = m_ref.training; m_ref.eval()
        dev = self.device
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0)).dtype if hasattr(m_ref, 'parameters') and next(m_ref.parameters(), None) is not None else torch.float32
        
        if noise is None: z = torch.randn(num_samples, self.args.latent_dim, device=dev, dtype=dtype_m)
        else: z = noise.to(device=dev, dtype=dtype_m); num_samples = z.shape[0]

        generated_regional_features = m_ref.decode(z) 
        
        unnormalized_coeffs_for_assembly_sample: torch.Tensor
        if self.vae_transform_type == 'complex_dft_ri':
            unnormalized_coeffs_for_assembly_sample = AudioSpecGenerator._unnormalize_and_reconstruct_coeffs_to_complex_dft(
                generated_regional_features, self.args, self.region_proc_size_tuple
            )
        elif self.vae_transform_type == 'dct':
            unnormalized_coeffs_for_assembly_sample = AudioSpecGenerator._unnormalize_dct_coeffs(
                generated_regional_features, self.args, self.region_proc_size_tuple
            )
        else:
            self.logger.error(f"Sample: Unknown VAE transform {self.vae_transform_type} for assembly.")
            m_ref.train(original_mode_m)
            return None
        
        current_audio_config = self._get_audio_config_ref()
        current_gaad_config = self._get_gaad_config_ref()
        spec_time_frames = current_audio_config.get("num_time_frames_for_1s_segment", 86) # Default for 1s @ 22050, hop 256
        spec_mels = self.args.n_mels
        spec_dims_canonical_for_gaad = (spec_time_frames, spec_mels) 
        
        canonical_bboxes_list = [golden_subdivide_rect_fixed_n(spec_dims_canonical_for_gaad, current_gaad_config['num_regions'], dev, dtype_m, current_gaad_config.get('min_size_px', 5)) for _ in range(num_samples)]
        canonical_gaad_bboxes_batch = torch.stack(canonical_bboxes_list)
        target_mel_shape_for_sample = (num_samples, 1, spec_mels, spec_time_frames)
        
        generated_mel_spectrograms = HybridTrainer._assemble_mel_from_transformed_coeffs_regions(
            unnormalized_coeffs_for_assembly_sample, canonical_gaad_bboxes_batch, 
            target_mel_shape_for_sample, self.args, self.region_proc_size_tuple
        )
        m_ref.train(original_mode_m)
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
    core_group.add_argument('--audio_dir_path', type=str, default="demo_audio_data_dir")
    core_group.add_argument('--checkpoint_dir',type=str, default='wubuspectrans_checkpoints_v011')
    core_group.add_argument('--load_checkpoint', type=str, default=None)
    core_group.add_argument('--local_rank', type=int, default=-1)
    core_group.add_argument('--seed',type=int, default=1337) # From batch
    core_group.add_argument('--num_workers',type=int, default=6) # From batch
    core_group.add_argument('--use_amp', action='store_true', default=True) # From batch
    core_group.add_argument('--detect_anomaly',action='store_true', default=False) # From batch
    core_group.add_argument('--ddp_find_unused_params_d', action='store_true', default=True) # From batch
    core_group.add_argument('--ddp_find_unused_params_g', action='store_true', default=True) # From batch

    # --- Group: Training Hyperparameters ---
    train_hp_group = parser.add_argument_group('Training Hyperparameters')
    train_hp_group.add_argument('--epochs', type=int, default=3000)
    train_hp_group.add_argument('--batch_size', type=int, default=32) # This is per-GPU
    train_hp_group.add_argument('--grad_accum_steps',type=int, default=1)
    train_hp_group.add_argument('--learning_rate_gen',type=float,default=2.5e-5)
    train_hp_group.add_argument('--learning_rate_disc',type=float,default=7.5e-5)
    train_hp_group.add_argument('--learning_rate_disc_alt',type=float,default=7e-5)
    train_hp_group.add_argument('--risgd_max_grad_norm',type=float,default=2.5)
    train_hp_group.add_argument('--global_max_grad_norm',type=float,default=4.0)

    # --- Group: Loss Weights ---
    loss_group = parser.add_argument_group('Loss Weights')
    loss_group.add_argument('--lambda_recon', type=float, default=12.0)
    loss_group.add_argument('--lambda_kl', type=float, default=1.5e-3)
    loss_group.add_argument('--lambda_gan', type=float, default=1.2)

    # --- Group: Audio Processing & Dataset ---
    audio_group = parser.add_argument_group('Audio Processing & Dataset')
    audio_group.add_argument('--sample_rate', type=int, default=44100)
    audio_group.add_argument('--n_fft', type=int, default=2048)
    audio_group.add_argument('--hop_length', type=int, default=256)
    audio_group.add_argument('--n_mels', type=int, default=256)
    audio_group.add_argument('--fmin', type=float, default=20.0)
    audio_group.add_argument('--fmax', type=float, default=20000.0)
    audio_group.add_argument('--segment_duration_sec', type=float, default=1.0)
    audio_group.add_argument('--segment_overlap_sec', type=float, default=0.0)
    audio_group.add_argument('--db_norm_min', type=float, default=-100.0)
    audio_group.add_argument('--db_norm_max', type=float, default=0.0)
    audio_group.add_argument('--preload_audio_dataset_to_ram', action='store_true', default=True)
    audio_group.add_argument('--dataset_yield_raw_audio', action='store_true', default=True)
    audio_group.add_argument('--validation_audio_dir_path', type=str, default=None)
    audio_group.add_argument('--validation_split_fraction', type=float, default=0.01)
    audio_group.add_argument('--data_fraction', type=float, default=1.0)

    # --- Group: GAAD (Spectrogram Regions) & VAE Feature Transform ---
    gaad_feat_group = parser.add_argument_group('GAAD & VAE Feature Transform')
    gaad_feat_group.add_argument('--gaad_num_regions', type=int, default=256)
    gaad_feat_group.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])
    gaad_feat_group.add_argument('--gaad_min_size_px', type=int, default=2)
    gaad_feat_group.add_argument('--region_proc_size_t', type=int, default=32)
    gaad_feat_group.add_argument('--region_proc_size_f', type=int, default=32)
    
    gaad_feat_group.add_argument('--vae_transform_type', type=str, default="complex_dft_ri", choices=["dct", "complex_dft_ri"])
    # DCT specific (only relevant if vae_transform_type is 'dct')
    # Provide defaults matching the batch script, but these will only be used if vae_transform_type=dct
    gaad_feat_group.add_argument('--dct_norm_type', type=str, default="tanh", choices=["none", "global_scale", "tanh"])
    gaad_feat_group.add_argument('--dct_norm_global_scale', type=float, default=200.0)
    gaad_feat_group.add_argument('--dct_norm_tanh_scale', type=float, default=60.0)
    # DFT specific (only relevant if vae_transform_type is 'complex_dft_ri')
    gaad_feat_group.add_argument('--dft_fft_norm', type=str, default="ortho", choices=["backward", "ortho", "forward"])
    gaad_feat_group.add_argument('--dft_complex_norm_scale', type=float, default=80.0)

    # --- Group: Model Architecture (VAE & General Discriminator Base) ---
    model_arch_group = parser.add_argument_group('Model Architecture (VAE & General Discriminator Base)')
    model_arch_group.add_argument('--latent_dim', type=int, default=1024)
    model_arch_group.add_argument('--encoder_initial_tangent_dim', type=int, default=384)
    model_arch_group.add_argument('--disc_input_type', type=str, default="mel", choices=["mel", "dct"])
    model_arch_group.add_argument('--disc_apply_spectral_norm', action='store_true', default=True)
    model_arch_group.add_argument('--disc_base_disc_channels', type=int, default=96)
    model_arch_group.add_argument('--disc_max_disc_channels', type=int, default=768)
    model_arch_group.add_argument('--disc_target_final_feature_dim', nargs='+', type=int, default=[6,6])
    model_arch_group.add_argument('--max_mel_disc_downsample_layers', type=int, default=5)
    model_arch_group.add_argument('--use_mel_d_attention', action='store_true', default=True)
    model_arch_group.add_argument('--mel_d_attention_idx', type=int, default=2)
    model_arch_group.add_argument('--mel_d_msd_num_scales', type=int, default=3)
    model_arch_group.add_argument('--mel_d_msd_share_weights', action='store_false', default=False)
    model_arch_group.add_argument('--disc_dct_embed_dim', type=int, default=192)
    model_arch_group.add_argument('--disc_dct_use_pos_embed', action='store_true', default=True)
    model_arch_group.add_argument('--disc_dct_use_cls_token', action='store_true', default=True)
    model_arch_group.add_argument('--disc_transformer_nhead', type=int, default=8)
    model_arch_group.add_argument('--disc_transformer_dim_feedforward', type=int, default=1536)
    model_arch_group.add_argument('--disc_transformer_dropout', type=float, default=0.05)
    model_arch_group.add_argument('--disc_transformer_num_layers', type=int, default=4)
    model_arch_group.add_argument('--disc_transformer_norm_first', action='store_true', default=True)
    model_arch_group.add_argument('--disc_use_global_stats_aux', action='store_true', default=True)
    model_arch_group.add_argument('--disc_global_stats_mlp_hidden_dim', type=int, default=64)

    # --- Group: Discriminator Architecture Variants ---
    disc_variant_group = parser.add_argument_group('Discriminator Architecture Variants')
    disc_variant_group.add_argument('--primary_disc_architecture_variant', type=str, default="multi_modal_wudio_mel", choices=["default", "global_wubu_dct", "multi_modal_wudio_mel"])
    disc_variant_group.add_argument('--alt_disc_architecture_variant', type=str, default="global_wubu_dct", choices=["default", "global_wubu_dct", "multi_modal_wudio_mel"])

    # --- Group: WuBu Stack Configurations (General for VAE) ---
    parser.add_argument('--wubu_dropout', type=float, default=0.05)
    wubu_s_group = parser.add_argument_group('WuBu-S (Encoder)')
    wubu_s_group.add_argument('--wubu_s_num_levels', type=int, default=5)
    wubu_s_group.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[384, 256, 192, 128, 96])
    wubu_s_group.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[0.95, 0.8, 0.7, 0.6, 0.5])
    wubu_s_group.add_argument('--wubu_s_initial_scales', nargs='+', type=float, default=[1.2, 1.1, 1.0, 0.9, 0.8]) # Default from batch
    wubu_s_group.add_argument('--wubu_s_initial_spread_values', nargs='+', type=float, default=[0.05, 0.07, 0.1, 0.12, 0.15]) # Default from batch
    wubu_s_group.add_argument('--wubu_s_boundary_points_per_level', nargs='+', type=int, default=[8, 6, 4, 2, 0]) # Default from batch
    wubu_s_group.add_argument('--wubu_s_use_rotation', action='store_true', default=True)
    wubu_s_group.add_argument('--wubu_s_phi_influence_curvature', action='store_true', default=True)
    wubu_s_group.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true', default=True)
    wubu_s_group.add_argument('--wubu_s_output_dim_encoder', type=int, default=768)

    wubu_g_group = parser.add_argument_group('WuBu-G (Generator)')
    wubu_g_group.add_argument('--wubu_g_num_levels', type=int, default=5)
    wubu_g_group.add_argument('--wubu_g_hyperbolic_dims', nargs='+', type=int, default=[256, 384, 512, 768, 0]) # Use 0 for None from batch
    wubu_g_group.add_argument('--wubu_g_initial_curvatures', nargs='+', type=float, default=[0.5, 0.6, 0.7, 0.8, 0.95])
    wubu_g_group.add_argument('--wubu_g_initial_scales', nargs='+', type=float, default=[0.8, 0.9, 1.0, 1.1, 1.2])
    wubu_g_group.add_argument('--wubu_g_initial_spread_values', nargs='+', type=float, default=[0.15, 0.12, 0.1, 0.07, 0.05])
    wubu_g_group.add_argument('--wubu_g_boundary_points_per_level', nargs='+', type=int, default=[0, 2, 4, 6, 8])
    wubu_g_group.add_argument('--wubu_g_use_rotation', action='store_true', default=True)
    wubu_g_group.add_argument('--wubu_g_phi_influence_curvature', action='store_true', default=True)
    wubu_g_group.add_argument('--wubu_g_phi_influence_rotation_init', action='store_true', default=True)

    wubu_d_region_group = parser.add_argument_group('WuBu-D-Region (Default Feature D Regional Processor)')
    wubu_d_region_group.add_argument('--wubu_d_region_num_levels', type=int, default=3)
    wubu_d_region_group.add_argument('--wubu_d_region_feature_dim', type=int, default=192)
    wubu_d_region_group.add_argument('--wubu_d_region_hyperbolic_dims', nargs='+', type=int, default=[192, 160, 128])
    wubu_d_region_group.add_argument('--wubu_d_region_initial_curvatures', nargs='+', type=float, default=[0.7, 0.5, 0.3])
    wubu_d_region_group.add_argument('--wubu_d_region_initial_scales', nargs='+', type=float, default=[1.0, 0.9, 0.8])
    wubu_d_region_group.add_argument('--wubu_d_region_initial_spread_values', nargs='+', type=float, default=[0.1, 0.12, 0.15])
    wubu_d_region_group.add_argument('--wubu_d_region_boundary_points_per_level', nargs='+', type=int, default=[4, 2, 0])
    wubu_d_region_group.add_argument('--wubu_d_region_use_rotation', action='store_true', default=True)
    wubu_d_region_group.add_argument('--wubu_d_region_phi_influence_curvature', action='store_true', default=True)
    wubu_d_region_group.add_argument('--wubu_d_region_phi_influence_rotation_init', action='store_true', default=True)

    wubu_d_global_group = parser.add_argument_group('WuBu-D-Global (Global WuBu Feature D Specific)')
    wubu_d_global_group.add_argument('--wubu_d_global_num_levels', type=int, default=4)
    wubu_d_global_group.add_argument('--wubu_d_global_hyperbolic_dims', nargs='+', type=int, default=[512, 384, 256, 192])
    wubu_d_global_group.add_argument('--wubu_d_global_initial_curvatures', nargs='+', type=float, default=[0.9, 0.75, 0.6, 0.45])
    wubu_d_global_group.add_argument('--wubu_d_global_initial_scales', nargs='+', type=float, default=[1.1, 1.0, 0.9, 0.8])
    wubu_d_global_group.add_argument('--wubu_d_global_initial_spread_values', nargs='+', type=float, default=[0.05, 0.08, 0.11, 0.14])
    wubu_d_global_group.add_argument('--wubu_d_global_boundary_points_per_level', nargs='+', type=int, default=[6, 4, 2, 0])
    wubu_d_global_group.add_argument('--wubu_d_global_use_rotation', action='store_true', default=True)
    wubu_d_global_group.add_argument('--wubu_d_global_phi_influence_curvature', action='store_true', default=True)
    wubu_d_global_group.add_argument('--wubu_d_global_phi_influence_rotation_init', action='store_true', default=True)
    wubu_d_global_group.add_argument('--dct_global_wubu_d_input_tangent_dim', type=int, default=768) # 'dct' prefix is legacy
    wubu_d_global_group.add_argument('--dct_global_wubu_d_output_feature_dim', type=int, default=256) # 'dct' prefix is legacy
    wubu_d_global_group.add_argument('--disc_use_global_stats_aux_dct_global_wubu', action='store_true', default=True)
    wubu_d_global_group.add_argument('--disc_global_stats_mlp_hidden_dim_dct_global_wubu', type=int, default=96)

    # --- Group: MultiModalWudioMelDiscriminator Specifics ---
    wudio_mel_d_group = parser.add_argument_group('Multi-Modal Wudio Mel Discriminator Specifics')
    wudio_mel_d_group.add_argument('--wudio_d_use_mel_cnn_head', action='store_true', default=True)
    wudio_mel_d_group.add_argument('--wudio_d_mel_msd_num_scales', type=int, default=3)
    wudio_mel_d_group.add_argument('--wudio_d_mel_msd_share_weights', action='store_false', default=False)
    wudio_mel_d_group.add_argument('--wudio_d_mel_cnn_base_ch', type=int, default=128)
    wudio_mel_d_group.add_argument('--wudio_d_mel_cnn_max_ch', type=int, default=1024)
    wudio_mel_d_group.add_argument('--wudio_d_mel_cnn_target_dim', nargs='+', type=int, default=[4,4])
    wudio_mel_d_group.add_argument('--wudio_d_mel_cnn_max_downs', type=int, default=6)
    wudio_mel_d_group.add_argument('--wudio_d_mel_cnn_output_feat_dim_for_fusion', type=int, default=384)
    wudio_mel_d_group.add_argument('--wudio_d_mel_use_attention', action='store_true', default=True)
    wudio_mel_d_group.add_argument('--wudio_d_mel_attention_idx', type=int, default=2)
    wudio_mel_d_group.add_argument('--wudio_d_mel_scale0_use_attention', action='store_true', default=True)
    wudio_mel_d_group.add_argument('--wudio_d_mel_scale0_attention_idx', type=int, default=2)
    wudio_mel_d_group.add_argument('--wudio_d_mel_scale1_use_attention', action='store_true', default=True)
    wudio_mel_d_group.add_argument('--wudio_d_mel_scale1_attention_idx', type=int, default=1)
    wudio_mel_d_group.add_argument('--wudio_d_mel_scale2_use_attention', action='store_false', default=False)
    # wudio_mel_d_group.add_argument('--wudio_d_use_stft_head', action='store_false', default=False) # STFT head not implemented yet
    # wudio_mel_d_group.add_argument('--wudio_d_stft_output_feat_dim_for_fusion', type=int, default=192)
    wudio_mel_d_group.add_argument('--wudio_d_use_global_audio_stats_head', action='store_true', default=True)
    wudio_mel_d_group.add_argument('--wudio_d_num_global_audio_stats', type=int, default=8)
    wudio_mel_d_group.add_argument('--wudio_d_global_audio_stats_mlp_hidden', type=int, default=96)
    wudio_mel_d_group.add_argument('--wudio_d_global_audio_stats_feat_dim_for_fusion', type=int, default=64)
    wudio_mel_d_group.add_argument('--wudio_d_fusion_method', type=str, default='concat_linear', choices=['concat_linear', 'average_logits'])
    wudio_mel_d_group.add_argument('--wudio_d_fusion_mlp_hidden_dim', type=int, default=192)

    # --- Group: Q-Learning Controller ---
    q_learn_group = parser.add_argument_group('Q-Learning Controller')
    q_learn_group.add_argument('--q_controller_enabled',action='store_true', default=True)
    q_learn_group.add_argument('--reset_q_controllers_on_load', action='store_false', default=False)
    q_learn_group.add_argument('--reset_lkl_q_controller_on_load', action='store_false', default=False)
    q_learn_group.add_argument('--lambda_kl_update_interval', type=int, default=10)
    q_learn_group.add_argument('--min_lambda_kl_q_control', type=float, default=5e-4)
    q_learn_group.add_argument('--max_lambda_kl_q_control', type=float, default=0.05)
    q_learn_group.add_argument('--q_lkl_scale_options', nargs='+', type=float, default=[0.6, 0.8, 1.0, 1.2, 1.4])
    q_learn_group.add_argument('--q_lkl_lr_mom_probation_steps', type=int, default=15)
    q_learn_group.add_argument('--q_lkl_action_probation_steps', type=int, default=15)

    # --- Group: Heuristic Interventions & Discriminator Switching ---
    heuristic_group = parser.add_argument_group('Heuristic Interventions')
    heuristic_group.add_argument('--enable_heuristic_interventions', action='store_true', default=True)
    heuristic_group.add_argument('--enable_heuristic_disc_switching', action='store_true', default=True)
    heuristic_group.add_argument('--initial_disc_type', type=str, default='mel', choices=['mel', 'dct'])
    heuristic_group.add_argument('--heuristic_check_interval', type=int, default=10)
    heuristic_group.add_argument('--heuristic_short_term_history_len', type=int, default=7)
    heuristic_group.add_argument('--heuristic_trigger_count_thresh', type=int, default=2)
    heuristic_group.add_argument('--disc_switch_check_interval', type=int, default=30)
    heuristic_group.add_argument('--disc_switch_min_steps_between', type=int, default=30)
    heuristic_group.add_argument('--disc_switch_problem_state_count_thresh', type=int, default=2)
    heuristic_group.add_argument('--heuristic_d_strong_thresh', type=float, default=0.20)
    heuristic_group.add_argument('--heuristic_d_weak_thresh', type=float, default=1.2)
    heuristic_group.add_argument('--heuristic_d_very_weak_thresh', type=float, default=2.0)
    heuristic_group.add_argument('--heuristic_g_stalled_thresh', type=float, default=1.8)
    heuristic_group.add_argument('--heuristic_g_winning_thresh', type=float, default=0.15)
    heuristic_group.add_argument('--heuristic_g_very_much_winning_thresh', type=float, default=0.03)
    heuristic_group.add_argument('--heuristic_kl_high_thresh', type=float, default=15.0)
    heuristic_group.add_argument('--heuristic_recon_stagnation_improvement_thresh_rel', type=float, default=0.001)
    heuristic_group.add_argument('--target_good_recon_thresh_heuristic', type=float, default=0.015)
    heuristic_group.add_argument('--heuristic_q_reward_stagnation_thresh', type=float, default=-0.3)
    heuristic_group.add_argument('--heuristic_recon_boost_factor', type=float, default=1.5)
    heuristic_group.add_argument('--lambda_feat_match_heuristic', type=float, default=1.0)
    heuristic_group.add_argument('--lambda_g_easy_win_penalty_heuristic', type=float, default=2.0)
    heuristic_group.add_argument('--g_easy_win_penalty_eps_denom', type=float, default=1e-5)
    heuristic_group.add_argument('--max_g_easy_win_penalty_abs', type=float, default=10.0)
    heuristic_group.add_argument('--heuristic_active_d_lr_boost_factor', type=float, default=2.0)
    heuristic_group.add_argument('--heuristic_d_q_explore_boost_epsilon', type=float, default=0.75)
    heuristic_group.add_argument('--heuristic_d_q_explore_duration', type=int, default=15)
    heuristic_group.add_argument('--heuristic_min_lambda_gan_factor', type=float, default=0.5)
    heuristic_group.add_argument('--heuristic_max_lambda_gan_factor', type=float, default=1.5)
    parser.add_argument('--force_start_epoch_on_load', type=int, default=None)
    parser.add_argument('--force_start_gstep_on_load', type=int, default=None)

    # --- Group: Logging, Sampling, Validation & Checkpointing ---
    log_group = parser.add_argument_group('Logging and Saving')
    log_group.add_argument('--log_interval',type=int, default=5)
    log_group.add_argument('--save_interval',type=int, default=2000)
    log_group.add_argument('--save_epoch_interval', type=int, default=1)
    log_group.add_argument('--validation_interval_epochs', type=int, default=1)
    log_group.add_argument('--disable_val_tqdm', action='store_false', default=False) # Default False (so tqdm is ON)
    log_group.add_argument('--wandb',action='store_true', default=True)
    log_group.add_argument('--wandb_project',type=str,default='WuBuSpecTransV011_AdvancedDs')
    log_group.add_argument('--wandb_run_name',type=str,default=None)
    log_group.add_argument('--wandb_log_train_recon_interval', type=int, default=20)
    log_group.add_argument('--train_target_log_freq_multiplier', type=int, default=3)
    log_group.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=50)
    log_group.add_argument('--use_lpips_for_mel_verification', action='store_true', default=True)
    log_group.add_argument('--val_primary_metric', type=str, default="avg_val_lpips_mel",
                        choices=["avg_val_recon_features_mse", "avg_val_mel_mse", "avg_val_psnr_mel", "avg_val_ssim_mel", "avg_val_lpips_mel"])
    log_group.add_argument('--num_val_samples_to_log', type=int, default=6)
    log_group.add_argument('--demo_num_samples', type=int, default=8)

    parsed_args = parser.parse_args()

    if parsed_args.vae_transform_type == 'dct' and not TORCH_DCT_AVAILABLE:
        parser.error("VAE transform type is 'dct', but torch-dct library is not found. Please install it: 'pip install torch-dct'")

    # --- Post-parsing argument validation and defaults ---
    if isinstance(parsed_args.disc_target_final_feature_dim, list):
        if len(parsed_args.disc_target_final_feature_dim) == 1: parsed_args.disc_target_final_feature_dim = [parsed_args.disc_target_final_feature_dim[0]] * 2
        elif len(parsed_args.disc_target_final_feature_dim) > 2: parser.error("--disc_target_final_feature_dim must be 1 or 2 integers.")
    elif isinstance(parsed_args.disc_target_final_feature_dim, int): parsed_args.disc_target_final_feature_dim = [parsed_args.disc_target_final_feature_dim] * 2
    if not (isinstance(parsed_args.disc_target_final_feature_dim, list) and len(parsed_args.disc_target_final_feature_dim) == 2):
        parsed_args.disc_target_final_feature_dim = [4,4]

    if parsed_args.wudio_d_use_mel_cnn_head:
        if parsed_args.wudio_d_mel_msd_num_scales is None: parsed_args.wudio_d_mel_msd_num_scales = parsed_args.mel_d_msd_num_scales
        if parsed_args.wudio_d_mel_cnn_base_ch is None: parsed_args.wudio_d_mel_cnn_base_ch = parsed_args.disc_base_disc_channels
        if parsed_args.wudio_d_mel_cnn_max_ch is None: parsed_args.wudio_d_mel_cnn_max_ch = parsed_args.disc_max_disc_channels
        wudio_target_dim = parsed_args.wudio_d_mel_cnn_target_dim
        if wudio_target_dim is None: wudio_target_dim = parsed_args.disc_target_final_feature_dim
        if isinstance(wudio_target_dim, list) and len(wudio_target_dim) == 1: wudio_target_dim = [wudio_target_dim[0]] * 2
        elif not (isinstance(wudio_target_dim, list) and len(wudio_target_dim) == 2): parser.error("--wudio_d_mel_cnn_target_dim must be 1 or 2 integers if provided.")
        parsed_args.wudio_d_mel_cnn_target_dim = wudio_target_dim
        if parsed_args.wudio_d_mel_cnn_max_downs is None: parsed_args.wudio_d_mel_cnn_max_downs = parsed_args.max_mel_disc_downsample_layers
        if parsed_args.wudio_d_fusion_mlp_hidden_dim is None: # Will be set by script if not provided
            feat_sum = parsed_args.wudio_d_mel_cnn_output_feat_dim_for_fusion
            if parsed_args.wudio_d_use_global_audio_stats_head: feat_sum += parsed_args.wudio_d_global_audio_stats_feat_dim_for_fusion
            parsed_args.wudio_d_fusion_mlp_hidden_dim = max(32, feat_sum // 2)

    if parsed_args.disc_dct_embed_dim is None:
        parsed_args.disc_dct_embed_dim = parsed_args.encoder_initial_tangent_dim
    if parsed_args.disc_transformer_dim_feedforward is None: # Will be set by script
        pass 

    if parsed_args.heuristic_check_interval is None: # Will be set by script
        parsed_args.heuristic_check_interval = parsed_args.disc_switch_check_interval if parsed_args.enable_heuristic_disc_switching else parsed_args.log_interval
    
    if parsed_args.enable_heuristic_disc_switching and parsed_args.initial_disc_type is None: # Will be set by script
        if parsed_args.primary_disc_architecture_variant == "multi_modal_wudio_mel": parsed_args.initial_disc_type = "mel"
        elif parsed_args.primary_disc_architecture_variant == "global_wubu_dct": parsed_args.initial_disc_type = "dct"
        else: parsed_args.initial_disc_type = parsed_args.disc_input_type

    wubu_prefixes_to_validate = ["wubu_s", "wubu_g", "wubu_d_global"]
    if (parsed_args.primary_disc_architecture_variant == "default" and parsed_args.disc_input_type == "dct") or \
       (parsed_args.alt_disc_architecture_variant == "default" and parsed_args.initial_disc_type == "dct"): # Use inferred initial_disc_type
        wubu_prefixes_to_validate.append("wubu_d_region")

    for prefix in wubu_prefixes_to_validate:
        num_levels_attr = f"{prefix}_num_levels"
        if hasattr(parsed_args, num_levels_attr):
            num_levels_val = getattr(parsed_args, num_levels_attr)
            if num_levels_val is not None and num_levels_val > 0:
                for list_arg_suffix in ["hyperbolic_dims", "initial_curvatures", "initial_scales", "initial_spread_values", "boundary_points_per_level"]:
                    attr_name = f"{prefix}_{list_arg_suffix}"
                    if not hasattr(parsed_args, attr_name) or getattr(parsed_args, attr_name) is None:
                        # Batch script should provide these lists; if not, _configure_wubu_stack uses DEFAULTS
                        # Forcing a default here might override intended None from batch script.
                        # Setattr to None if not present, so _configure_wubu_stack takes over.
                        setattr(parsed_args, attr_name, None)
                validate_wubu_config_for_argparse(parsed_args, prefix, parser)

    if parsed_args.wubu_d_region_num_levels > 0 and parsed_args.wubu_d_region_hyperbolic_dims is None: # Script default is list
        parsed_args.wubu_d_region_hyperbolic_dims = [parsed_args.wubu_d_region_feature_dim] * parsed_args.wubu_d_region_num_levels
    
    if parsed_args.wubu_g_num_levels > 0 and parsed_args.wubu_g_hyperbolic_dims is not None:
        vae_transform = getattr(parsed_args, 'vae_transform_type', 'complex_dft_ri')
        if vae_transform == 'dct': feats_per_reg = parsed_args.region_proc_size_t * parsed_args.region_proc_size_f
        elif vae_transform == 'complex_dft_ri': feats_per_reg = 2 * parsed_args.region_proc_size_t * parsed_args.region_proc_size_f
        else: feats_per_reg = 256 
        
        temp_dims = []
        for i, dim_val_str in enumerate(parsed_args.wubu_g_hyperbolic_dims): # Process list from batch (elements are strings)
            if isinstance(dim_val_str, str) and dim_val_str.lower() == 'none':
                dim_val = 0 # Treat 'None' string as 0 or placeholder
            else:
                try: dim_val = int(dim_val_str)
                except ValueError: dim_val = 0 # If conversion fails
            
            if dim_val <=0 : # Handle 0 or 'None'
                if i == len(parsed_args.wubu_g_hyperbolic_dims) - 1: temp_dims.append(feats_per_reg)
                else: temp_dims.append(max(32, feats_per_reg // 2))
            else:
                temp_dims.append(dim_val)
        parsed_args.wubu_g_hyperbolic_dims = temp_dims
        
    # Ensure all list-like WuBu args that might be None from batch are set to empty lists if num_levels is 0
    # or if they remain None after batch parsing and are required by _configure_wubu_stack
    for prefix in wubu_prefixes_to_validate:
        num_levels_val = getattr(parsed_args, f"{prefix}_num_levels", 0)
        for list_arg_suffix in ["initial_scales", "initial_spread_values", "boundary_points_per_level"]:
            attr_name = f"{prefix}_{list_arg_suffix}"
            if getattr(parsed_args, attr_name) is None:
                if num_levels_val > 0:
                    # _configure_wubu_stack will use its internal defaults if these are still None
                    pass
                else: # num_levels is 0, ensure these are empty lists
                    setattr(parsed_args, attr_name, [])
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
    args = parse_arguments() # This should call your provided parse_arguments function
    ddp_active = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    if ddp_active:
        rank=int(os.environ["RANK"])
        local_rank=int(os.environ["LOCAL_RANK"])
        world_size=int(os.environ["WORLD_SIZE"])
        init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            device=torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device=torch.device("cpu") # DDP on CPU
    else:
        rank=0; local_rank=0; world_size=1
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(device)

    am_main_process = (rank == 0)

    base_logger_name = "WuBuSpecTransV01"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    specific_logger = logging.getLogger(base_logger_name)
    for handler in specific_logger.handlers[:]: specific_logger.removeHandler(handler)

    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.basicConfig(level=log_level,
                        format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s',
                        force=True)

    current_logger_main = logging.getLogger(f"{base_logger_name}.Main")
    current_logger_main.info(f"--- {base_logger_name} (R{rank}/{world_size}, Dev {device}, DDP:{ddp_active}, AMP:{args.use_amp}) ---")
    seed_everything(args.seed, rank, world_size)

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        current_logger_main.warning("Autograd anomaly detection ENABLED.")

    if am_main_process:
        current_logger_main.info(f"Effective Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubuspec_{args.vae_transform_type}_{datetime.now().strftime('%y%m%d_%H%M')}"
        try:
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
                       resume="allow", id=wandb.util.generate_id() if wandb.run is None else wandb.run.id)
            current_logger_main.info(f"WandB initialized for run: {run_name}, Project: {args.wandb_project}")
        except Exception as e_wandb:
            current_logger_main.error(f"WandB initialization failed: {e_wandb}", exc_info=True)
            args.wandb = False

    segment_samples = int(args.segment_duration_sec * args.sample_rate)
    num_time_frames_for_1s_segment = math.ceil(segment_samples / args.hop_length)

    audio_config = {
        "sample_rate": args.sample_rate, "n_fft": args.n_fft, "hop_length": args.hop_length,
        "n_mels": args.n_mels, "fmin": args.fmin, "fmax": args.fmax,
        "segment_duration_sec": args.segment_duration_sec,
        "region_proc_size_t": args.region_proc_size_t, "region_proc_size_f": args.region_proc_size_f,
        "wubu_s_output_dim_encoder": args.wubu_s_output_dim_encoder,
        "num_time_frames_for_1s_segment": num_time_frames_for_1s_segment,
    }
    gaad_config = {
        "num_regions": args.gaad_num_regions, "decomposition_type": args.gaad_decomposition_type,
        "min_size_px": args.gaad_min_size_px
    }
    wubu_s_config_enc = _configure_wubu_stack(args, "wubu_s")
    wubu_g_config_gen = _configure_wubu_stack(args, "wubu_g")

    if am_main_process:
        current_logger_main.info(f"AudioCfg:{audio_config}\nGAADCfg:{gaad_config}\nWuBuS_Enc:{wubu_s_config_enc}\nWuBuG_Gen:{wubu_g_config_gen}")

    model = WuBuSpecTransNet(args, audio_config, gaad_config, wubu_s_config_enc, wubu_g_config_gen).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)

    if ddp_active:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                    output_device=local_rank if device.type == 'cuda' else None,
                    find_unused_parameters=args.ddp_find_unused_params_g)

    audio_files_list = []
    audio_dir_path_obj = Path(args.audio_dir_path)
    if audio_dir_path_obj.is_dir():
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
            audio_files_list.extend([str(p) for p in audio_dir_path_obj.rglob(ext)])
    elif audio_dir_path_obj.is_file():
        audio_files_list.append(str(audio_dir_path_obj))

    if not audio_files_list and "demo_audio_data" in str(args.audio_dir_path):
        if am_main_process:
            demo_dir = Path(args.audio_dir_path)
            demo_dir.mkdir(parents=True, exist_ok=True)
            dummy_audio_path = demo_dir / "dummy_sine_wubuspectrans.wav"
            if not dummy_audio_path.exists():
                current_logger_main.info(f"Attempting to create dummy audio: {dummy_audio_path}...")
                try:
                    import soundfile as sf
                    sr_dummy, duration_dummy = args.sample_rate, 5.0
                    t_dummy = np.linspace(0, duration_dummy, int(sr_dummy * duration_dummy), endpoint=False)
                    wav_dummy = (0.3*np.sin(2*np.pi*220*t_dummy) + 0.2*np.sin(2*np.pi*440*t_dummy) + 0.1*np.sin(2*np.pi*880*t_dummy))
                    wav_dummy = (wav_dummy / (np.max(np.abs(wav_dummy)) + EPS) * 0.9).astype(np.float32)
                    sf.write(str(dummy_audio_path), wav_dummy, sr_dummy)
                    current_logger_main.info(f"Dummy audio created: {dummy_audio_path}")
                    audio_files_list.append(str(dummy_audio_path))
                except ImportError: current_logger_main.error("soundfile missing for dummy audio.")
                except Exception as e: current_logger_main.error(f"Error dummy audio: {e}", exc_info=True)
        if ddp_active: torch.distributed.barrier()

    if not audio_files_list:
        current_logger_main.error(f"No audio files in '{args.audio_dir_path}'. Exiting."); sys.exit(1)
    current_logger_main.info(f"Found {len(audio_files_list)} audio files for main dataset pool.")
    
    needs_raw_audio_for_stats = False
    if (args.primary_disc_architecture_variant == "multi_modal_wudio_mel" and getattr(args, 'wudio_d_use_global_audio_stats_head', False)) or \
       (args.alt_disc_architecture_variant == "multi_modal_wudio_mel" and getattr(args, 'wudio_d_use_global_audio_stats_head', False)):
        needs_raw_audio_for_stats = True
    
    effective_yield_raw_audio = False
    if needs_raw_audio_for_stats:
        if args.dataset_yield_raw_audio:
            effective_yield_raw_audio = True
            if am_main_process: current_logger_main.info("Dataset WILL yield raw audio segments for WudioMelD stats as requested.")
        else:
            if am_main_process: current_logger_main.warning("WudioMelD expects raw audio stats, but --dataset_yield_raw_audio is NOT set. Stats for REAL samples will be ESTIMATED from Mel.")
    elif args.dataset_yield_raw_audio: 
            if am_main_process: current_logger_main.info("Dataset yielding raw audio (--dataset_yield_raw_audio set), but no active D config appears to require it. This is okay.")
            effective_yield_raw_audio = True


    try:
        full_dataset = AudioSegmentDataset(audio_file_paths=audio_files_list, args=args,
                                           segment_duration_sec=args.segment_duration_sec,
                                           segment_overlap_sec=args.segment_overlap_sec,
                                           data_fraction=args.data_fraction,
                                           preload_to_ram=args.preload_audio_dataset_to_ram,
                                           yield_raw_audio=effective_yield_raw_audio)
    except Exception as e:
        current_logger_main.error(f"Failed to initialize main Dataset: {e}", exc_info=True); sys.exit(1)
    if not full_dataset or len(full_dataset) == 0:
        current_logger_main.error("Main dataset is empty. Exiting."); sys.exit(1)

    train_dataset: Union[AudioSegmentDataset, torch.utils.data.Subset] = full_dataset
    val_dataset: Optional[Union[AudioSegmentDataset, torch.utils.data.Subset]] = None
    num_total_samples = len(full_dataset)

    val_audio_files_list = []
    if args.validation_audio_dir_path:
        val_dir_path_obj = Path(args.validation_audio_dir_path)
        if val_dir_path_obj.is_dir():
            for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]: val_audio_files_list.extend([str(p) for p in val_dir_path_obj.rglob(ext)])
        elif val_dir_path_obj.is_file(): val_audio_files_list.append(str(val_dir_path_obj))

    if val_audio_files_list:
        try:
            val_dataset_candidate = AudioSegmentDataset(
                audio_file_paths=val_audio_files_list, args=args, 
                segment_duration_sec=args.segment_duration_sec, 
                data_fraction=1.0, 
                preload_to_ram=args.preload_audio_dataset_to_ram,
                yield_raw_audio=effective_yield_raw_audio 
            )
            if len(val_dataset_candidate) > 0:
                val_dataset = val_dataset_candidate
                current_logger_main.info(f"Using separate validation audio dir: {args.validation_audio_dir_path}, Segments: {len(val_dataset)}")
            else: current_logger_main.warning(f"Validation audio dir {args.validation_audio_dir_path} empty.")
        except Exception as e: current_logger_main.warning(f"Could not load validation dataset from '{args.validation_audio_dir_path}': {e}.")

    if val_dataset is None and args.validation_split_fraction > 0.0 and num_total_samples > 10 :
        num_val = int(num_total_samples * args.validation_split_fraction)
        num_train = num_total_samples - num_val
        if num_train > 0 and num_val > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(args.seed + rank))
            current_logger_main.info(f"Split main dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
        else:
            current_logger_main.warning("Random split for validation resulted in 0 samples. No validation set from split."); val_dataset = None
            train_dataset = full_dataset

    if am_main_process:
        current_logger_main.info(f"Final dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}")

    worker_init_fn_seeded = functools.partial(seed_worker_init_fn, base_seed=args.seed, rank=rank, world_size=world_size) if args.num_workers > 0 else None
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_active else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler,
                              pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded,
                              drop_last=True if ddp_active else False)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, sampler=val_sampler,
                                pin_memory=(device.type == 'cuda'),
                                drop_last=False, worker_init_fn=worker_init_fn_seeded)

    trainer = HybridTrainer(model=model, device=device, train_loader=train_loader, val_loader=val_loader,
                            args=args, rank=rank, world_size=world_size, ddp_active=ddp_active)
    
    if am_main_process: # DEBUG CHECK for _save_checkpoint
        if hasattr(trainer, '_save_checkpoint') and callable(getattr(trainer, '_save_checkpoint')):
            current_logger_main.info("DEBUG (main): HybridTrainer instance HAS a callable _save_checkpoint method.")
        else:
            current_logger_main.error("DEBUG (main): HybridTrainer instance DOES NOT HAVE a callable _save_checkpoint method.")

    start_global_step, start_epoch = 0, 0
    if args.load_checkpoint:
        start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint)
    else: 
        if am_main_process: trainer.logger.info("New run. Initializing Q-controllers fresh.")
        controllers_to_init = [getattr(trainer.optimizer_enc_gen, 'q_controller', None),
                               getattr(trainer.optimizer_disc_primary, 'q_controller', None),
                               getattr(trainer.optimizer_disc_alternative, 'q_controller', None),
                               trainer.lambda_kl_q_controller]
        initial_dummy_losses = {'loss_g_total':1.0,'loss_g_recon':1.0,'loss_g_kl':0.1,'loss_g_adv':0.7,
                                'loss_d_total':0.7,'loss_d_real':0.7,'loss_d_fake':0.7}
        for i_qc, qc_obj in enumerate(controllers_to_init):
            if qc_obj:
                is_gen_q = (i_qc == 0)
                if hasattr(qc_obj, 'set_initial_losses'): qc_obj.set_initial_losses(initial_dummy_losses, is_generator_q=is_gen_q if i_qc < 3 else False)
                if qc_obj == trainer.lambda_kl_q_controller and hasattr(qc_obj, 'set_initial_lambda_kl_metrics'):
                    initial_lkl_metrics = {'avg_recon':1.0,'avg_kl_div':0.1,'avg_d_total':0.7,'val_metric':1.0,'current_lambda_kl_val':trainer.lambda_kl}
                    qc_obj.set_initial_lambda_kl_metrics(initial_lkl_metrics)
                if hasattr(qc_obj, 'start_probation'): qc_obj.start_probation()

    try:
        trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt:
        current_logger_main.info(f"Rank {rank}: Training interrupted by user.")
    except Exception as e:
        current_logger_main.error(f"Rank {rank}: Training loop crashed: {e}", exc_info=True)
    finally:
        if am_main_process:
            current_logger_main.info("Finalizing run and saving final checkpoint...")
            final_metrics_to_save = trainer.last_val_metrics.copy() if hasattr(trainer, 'last_val_metrics') and trainer.last_val_metrics else {}
            if hasattr(trainer, 'best_val_metric_val'): 
                final_metrics_to_save['best_val_metric_val_at_end'] = trainer.best_val_metric_val
            
            if hasattr(trainer, '_save_checkpoint') and callable(getattr(trainer, '_save_checkpoint')) :
                trainer._save_checkpoint(metrics=final_metrics_to_save) 
            else:
                current_logger_main.error("CRITICAL: trainer object does not have a callable _save_checkpoint method in finally block!")

            if args.epochs > 0 and hasattr(trainer, 'sample') and hasattr(trainer, 'global_step') and trainer.global_step > 0 and args.demo_num_samples > 0:
                current_logger_main.info("Generating final demo samples (Mel Spectrograms)...")
                try:
                    generated_mels = trainer.sample(num_samples=args.demo_num_samples)
                    if generated_mels is not None and generated_mels.numel() > 0 and save_image is not None:
                        save_dir = Path(args.checkpoint_dir) / f"demo_samples_mel_{args.vae_transform_type}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        for b_idx in range(min(args.demo_num_samples, generated_mels.shape[0])):
                            mel_to_save = (generated_mels[b_idx, 0].cpu().clamp(-1,1) + 1) / 2.0
                            save_image_path = save_dir / f"demo_mel_sample_{b_idx}_ep{trainer.current_epoch+1}_gs{trainer.global_step}.png"
                            save_image(mel_to_save, str(save_image_path))
                        current_logger_main.info(f"Saved demo Mel spectrogram images to {save_dir}")
                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            trainer._log_samples_to_wandb("final_demo_mel", generated_mels, args.demo_num_samples)
                    elif save_image is None:
                        current_logger_main.warning("torchvision.utils.save_image not available. Cannot save demo images.")
                except Exception as e_demo:
                    current_logger_main.error(f"Demo Mel sampling/saving error: {e_demo}", exc_info=True)

            if args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish()

        if ddp_active and is_initialized():
            destroy_process_group()
        current_logger_main.info(f"Rank {rank}: {base_logger_name} (v0.1.1) script finished.")

if __name__ == "__main__":
    main()