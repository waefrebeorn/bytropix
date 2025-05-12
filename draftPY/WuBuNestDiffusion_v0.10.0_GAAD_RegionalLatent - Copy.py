# WuBuNestDiffusion_v0.10.0_GAAD_RegionalLatent.py
# Diffusion Model with GAAD-WuBu Regional Hyperbolic Latent Space Autoencoder
# Operating directly on GAAD-defined regions with WuBu nesting.
# LAST UPDATE: Major Architectural Shift to GAAD-WuBu AE (v0.10.0 internal rev)

# =====================================================================
# Python Imports and Setup (includes torchvision.ops.roi_align)
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
logger = logging.getLogger("WuBuGAADRegionalLatentDiffV010")
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

# Default Transformer Config (NEW for v0.10.0 Noise Predictor)
DEFAULT_CONFIG_TRANSFORMER_NOISE_PREDICTOR = {
    "num_layers": 4,
    "num_heads": 8,
    "d_model": 256, # Should ideally match or be compatible with WuBu output + time embedding
    "d_ff_ratio": 4.0,
    "dropout": 0.1,
    "activation": "gelu",
}

# =====================================================================
# Geometric, Optimizer, WuBu Core Components (Largely Unchanged from v0.05.2)
# =====================================================================
class HyperbolicUtils:
    @staticmethod
    def poincare_clip(x: torch.Tensor, c_scalar: float, radius: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_input_dtype = x.dtype
        if c_scalar <= 0:
            x_compute = x.float()
            if not torch.isfinite(x_compute).all():
                try: logger_pc_euc = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.poincare_clip"); logger_pc_euc.warning(f"Euclidean/Invalid C: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
                except NameError: print(f"Warning: WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.poincare_clip: Euclidean/Invalid C: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
                x_compute = torch.nan_to_num(x_compute, nan=0.0, posinf=0.0, neginf=0.0)
            if original_input_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; x_compute = torch.clamp(x_compute, min=-f16_max, max=f16_max)
            return x_compute.to(original_input_dtype)
        sqrt_c = math.sqrt(max(c_scalar, eps)); effective_radius_factor = min(radius, 1.0 - eps); max_norm_val_f32 = effective_radius_factor / sqrt_c
        x_compute = x.float()
        if not torch.isfinite(x_compute).all():
            try: logger_pc_hyp = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.poincare_clip"); logger_pc_hyp.warning(f"Hyperbolic: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
            except NameError: print(f"Warning: WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.poincare_clip: Hyperbolic: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
            x_compute = torch.nan_to_num(x_compute, nan=0.0, posinf=0.0, neginf=0.0)
        x_norm_sq = torch.sum(x_compute.pow(2), dim=-1, keepdim=True); sqrt_input_val = torch.clamp(x_norm_sq, min=0.0) + eps; sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=1.0, neginf=eps); sqrt_input_val.clamp_min_(eps); norm = torch.sqrt(sqrt_input_val); cond = norm > max_norm_val_f32; norm_plus_eps_for_div = norm + eps; norm_plus_eps_for_div.clamp_min_(eps); scale_factor = torch.where(cond, max_norm_val_f32 / norm_plus_eps_for_div, torch.ones_like(norm)); clipped_x_f32 = x_compute * scale_factor
        if original_input_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; clipped_x_f32 = torch.clamp(clipped_x_f32, min=-f16_max, max=f16_max)
        final_clipped_x = clipped_x_f32.to(original_input_dtype)
        if not torch.isfinite(final_clipped_x).all(): current_max_norm_for_nan_fill = float(max_norm_val_f32); return torch.nan_to_num(final_clipped_x, nan=0.0, posinf=current_max_norm_for_nan_fill, neginf=-current_max_norm_for_nan_fill)
        return final_clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = v.dtype
        if c_scalar <= 0:
            v_compute = v.float()
            if not torch.isfinite(v_compute).all(): current_logger_exp_euc = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_exponential_map"); current_logger_exp_euc.warning(f"Euclidean: Non-finite input v (shape {v_compute.shape}). Sanitizing."); v_compute = torch.nan_to_num(v_compute, nan=0.0, posinf=0.0, neginf=0.0)
            if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; v_compute = torch.clamp(v_compute, min=-f16_max, max=f16_max)
            return v_compute.to(original_dtype)
        v_compute = v.float()
        if not torch.isfinite(v_compute).all(): current_logger_exp_hyp = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_exponential_map"); current_logger_exp_hyp.warning(f"Hyperbolic: Non-finite input v (shape {v_compute.shape}). Sanitizing."); v_compute = torch.nan_to_num(v_compute, nan=0.0, posinf=0.0, neginf=0.0)
        v_norm_sq_unclamped = torch.sum(v_compute.pow(2), dim=-1, keepdim=True); v_norm_sq_clamped = torch.clamp(v_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL); sqrt_input_val = v_norm_sq_clamped + eps; sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps); sqrt_input_val.clamp_min_(eps); v_norm = torch.sqrt(sqrt_input_val)
        if not torch.isfinite(v_norm).all(): current_logger_exp_hyp_vn_err = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_exponential_map"); current_logger_exp_hyp_vn_err.error(f"v_norm non-finite despite sanitization! Fallback to zero vector."); return HyperbolicUtils.poincare_clip(torch.zeros_like(v_compute), c_scalar, eps=eps).to(original_dtype)
        sqrt_c_val = math.sqrt(max(c_scalar, eps)); scaled_radius_arg = float(scale_scalar) * sqrt_c_val * v_norm; tanh_input_val = torch.clamp(scaled_radius_arg, min=-30.0, max=30.0); tanh_term_val = torch.tanh(tanh_input_val); denominator_lambda_candidate = sqrt_c_val * v_norm + eps; denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps); lambda_v_val = torch.where(v_norm > eps, tanh_term_val / denominator_lambda_val, torch.full_like(v_norm, float(scale_scalar), dtype=torch.float32)); mapped_v_f32 = lambda_v_val * v_compute
        if not torch.isfinite(mapped_v_f32).all(): current_logger_exp_hyp_mv_err = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_exponential_map"); current_logger_exp_hyp_mv_err.warning(f"mapped_v_f32 non-finite. Zeroing before Poincare clip."); mapped_v_f32 = torch.zeros_like(v_compute)
        clipped_mapped_v_f32 = HyperbolicUtils.poincare_clip(mapped_v_f32, c_scalar, eps=eps); final_result = clipped_mapped_v_f32
        if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; final_result = torch.clamp(clipped_mapped_v_f32, min=-f16_max, max=f16_max)
        return final_result.to(original_dtype)

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = y.dtype
        if c_scalar <= 0:
            y_compute = y.float()
            if not torch.isfinite(y_compute).all(): current_logger_log_euc = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_logarithmic_map"); current_logger_log_euc.warning(f"Euclidean: Non-finite input y (shape {y_compute.shape}). Sanitizing."); y_compute = torch.nan_to_num(y_compute, nan=0.0, posinf=0.0, neginf=0.0)
            if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; y_compute = torch.clamp(y_compute, min=-f16_max, max=f16_max)
            return y_compute.to(original_dtype)
        y_clipped_original_dtype = HyperbolicUtils.poincare_clip(y, c_scalar, eps=eps); y_compute = y_clipped_original_dtype.float()
        if not torch.isfinite(y_compute).all(): current_logger_log_hyp_yc_err = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_logarithmic_map"); current_logger_log_hyp_yc_err.warning(f"y_compute (from y_clipped.float()) non-finite. Sanitizing."); y_compute = torch.nan_to_num(y_compute, nan=0.0, posinf=0.0, neginf=0.0)
        y_norm_sq_unclamped = torch.sum(y_compute.pow(2), dim=-1, keepdim=True); y_norm_sq_clamped = torch.clamp(y_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL); sqrt_input_val = y_norm_sq_clamped + eps; sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps); sqrt_input_val.clamp_min_(eps); y_norm = torch.sqrt(sqrt_input_val)
        if not torch.isfinite(y_norm).all(): current_logger_log_hyp_yn_err = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_logarithmic_map"); current_logger_log_hyp_yn_err.error(f"y_norm non-finite despite sanitization! Fallback to zero vector."); return torch.zeros_like(y, dtype=original_dtype)
        sqrt_c_val = math.sqrt(max(c_scalar, eps)); arctanh_arg_raw = sqrt_c_val * y_norm; arctanh_arg_clamped = torch.clamp(arctanh_arg_raw, min=-1.0 + eps*10, max=1.0 - eps*10); atanh_term_val = torch.atanh(arctanh_arg_clamped); denominator_lambda_candidate = float(scale_scalar) * sqrt_c_val * y_norm + eps; denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps); default_lambda_y_val = 1.0 / max(float(scale_scalar), eps); lambda_y_val = torch.where(y_norm > eps, atanh_term_val / denominator_lambda_val, torch.full_like(y_norm, default_lambda_y_val, dtype=torch.float32)); mapped_y_f32 = lambda_y_val * y_compute
        if not torch.isfinite(mapped_y_f32).all(): current_logger_log_hyp_my_err = logging.getLogger("WuBuGAADRegionalLatentDiffV010.HyperbolicUtils.scale_aware_logarithmic_map"); current_logger_log_hyp_my_err.warning(f"mapped_y_f32 non-finite. Returning zeros."); mapped_y_f32 = torch.zeros_like(y_compute)
        final_result = mapped_y_f32
        if original_dtype == torch.float16: f16_max = torch.finfo(torch.float16).max; final_result = torch.clamp(mapped_y_f32, min=-f16_max, max=f16_max)
        return final_result.to(original_dtype)

    @staticmethod
    def exponential_map(v: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_exponential_map(v, c_scalar, scale_scalar=1.0, eps=eps)

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_logarithmic_map(y, c_scalar, scale_scalar=1.0, eps=eps)

# Manifold classes (PoincareBall, etc.) - unchanged from v0.05.2
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
        if self.c <= 0: self.k = 0.; self.sqrt_c = 0.; self.radius = float('inf')
        else: self.k = -self.c; self.sqrt_c = math.sqrt(self.c); self.radius = 1. / self.sqrt_c
        self.max_norm = self.radius * (1. - EPS * 10) if self.c > 0 and self.radius != float('inf') else float('inf')
        self._name = f'PoincareBall(c={self.c:.3g})'
    @property
    def name(self) -> str: return self._name
    def proju(self, x: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.poincare_clip(x, self.c, radius=1., eps=EPS * 10)
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.exponential_map(dp, self.c, eps=EPS)
    def logmap0(self, p: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.logarithmic_map(p, self.c, eps=EPS)
    def expmap0_scaled(self, dp: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_exponential_map(dp, self.c, scale_scalar=scale_scalar, eps=EPS)
    def logmap0_scaled(self, p: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_logarithmic_map(p, self.c, scale_scalar=scale_scalar, eps=EPS)
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        if self.c <= 0: return dp
        p_proj = self.proju(p)
        lambda_p_sq_val = (1. - self.c * torch.sum(p_proj.pow(2), dim=-1, keepdim=True).clamp_max(1. / (self.c + EPS) - EPS * 100))
        factor = (lambda_p_sq_val / 2.).pow(2)
        return torch.clamp(factor, min=EPS) * dp
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad(): w.data.uniform_(-irange, irange); w.data = self.expmap0(w.data); w.data = self.proju(w.data)

def init_weights_general(m): # Unchanged
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine: nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm):
        if m.affine: nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.zeros_(m.bias) if m.bias is not None else None

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor: return F.softplus(param_unconstrained) + min_val

# BoundaryManifoldHyperbolic, HyperbolicInterLevelTransform, HyperbolicWuBuNestingLevel, FullyHyperbolicWuBuNestingModel
# These WuBu core components remain largely the same structurally but will be *used* differently (applied per-region).
class BoundaryManifoldHyperbolic(nn.Module): # Unchanged
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold_c: float):
        super().__init__(); self.level_idx = level_idx; self.num_points = num_points; self.point_dim = point_dim; self.current_manifold_c = initial_manifold_c
        if num_points > 0 and point_dim > 0: self.hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim)); PoincareBall(initial_manifold_c).init_weights(self.hyperbolic_points_params, irange=1e-3); setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(initial_manifold_c))
        else: self.register_parameter('hyperbolic_points_params', None)
    def set_current_manifold_c(self, c_scalar: float): self.current_manifold_c = c_scalar; setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(c_scalar)) if self.hyperbolic_points_params is not None else None
    def get_points(self) -> Optional[torch.Tensor]: return PoincareBall(self.current_manifold_c).proju(self.hyperbolic_points_params) if self.hyperbolic_points_params is not None else None

def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor: # Unchanged
    axis = F.normalize(axis, p=2, dim=-1); angle_half = angle_rad / 2.0; q_w = torch.cos(angle_half); q_xyz = axis * torch.sin(angle_half); return torch.cat([q_w, q_xyz], dim=-1)
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor: # Unchanged
    w1, x1, y1, z1 = q1.unbind(-1); w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2; return torch.stack([w, x, y, z], dim=-1)
def quaternion_apply_to_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: # Unchanged
    v_quat = F.pad(v, (1, 0), value=0); q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1); rotated_v_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj); return rotated_v_quat[..., 1:]

class HyperbolicInterLevelTransform(nn.Module): # Unchanged
    def __init__(self, in_dim: int, out_dim: int, initial_c_in: float, initial_c_out: float, transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1, use_rotation: bool = False, phi_influence_rotation_init: bool = False, level_idx_for_phi: int = 0):
        super().__init__(); self.in_dim, self.out_dim, self.transform_type = in_dim, out_dim, transform_type.lower(); self.use_rotation = use_rotation; self.rotation_module = None; self.phi_influence_rotation_init = phi_influence_rotation_init
        if self.use_rotation and self.in_dim > 0:
            if self.in_dim == 4 and self.phi_influence_rotation_init: self.rot_axis_param = nn.Parameter(torch.randn(3)); self.rot_angle_unconstrained = nn.Parameter(torch.tensor(0.0)); self.phi_angle_scale = PHI**(level_idx_for_phi % 5 - 2) * (math.pi / 4); logger.info(f"InterLevelTransform L{level_idx_for_phi} (4D): Learnable Quat phi-biased rot. Angle scale: {self.phi_angle_scale:.3f}")
            elif self.in_dim == 2 and self.phi_influence_rotation_init: self.rot_angle_unconstrained_2d = nn.Parameter(torch.tensor(0.0)); self.phi_angle_scale_2d = PHI**(level_idx_for_phi % 3) * (math.pi / 3); logger.info(f"InterLevelTransform L{level_idx_for_phi} (2D): Learnable SO(2) phi-biased rot. Angle scale: {self.phi_angle_scale_2d:.3f}")
            else: self.rotation_module = nn.Linear(self.in_dim, self.in_dim, bias=False); nn.init.eye_(self.rotation_module.weight) if self.in_dim > 0 else None
        mlp_hidden_dim = hidden_dim if hidden_dim and hidden_dim > 0 else max(16, (in_dim + out_dim) // 2)
        if self.transform_type == 'mlp' and all(d > 0 for d in [in_dim, out_dim, mlp_hidden_dim]): self.non_rotational_map = nn.Sequential(nn.Linear(in_dim, mlp_hidden_dim), nn.LayerNorm(mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim, out_dim))
        elif self.transform_type == 'linear' and in_dim > 0 and out_dim > 0: self.non_rotational_map = nn.Linear(in_dim, out_dim)
        else: self.non_rotational_map = nn.Identity()
        self.apply(init_weights_general)
    def _apply_rotation(self, x_tan: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_tan is None or not self.use_rotation: return x_tan; B_maybe = x_tan.shape[0] if x_tan.dim() > 1 else 1 # Handle cases where B might be missing if input is single vector
        if self.in_dim == 4 and self.phi_influence_rotation_init and hasattr(self, 'rot_axis_param'): angle = F.softplus(self.rot_angle_unconstrained) * self.phi_angle_scale; current_axis = self.rot_axis_param.to(x_tan.device).unsqueeze(0).expand(B_maybe, -1); angle_b = angle.unsqueeze(0).expand(B_maybe, 1); q_rot = quaternion_from_axis_angle(angle_b, current_axis); return self.rotation_module(x_tan) if self.rotation_module else x_tan # MISSING: actual rotation call with quaternion_apply_to_vector
        elif self.in_dim == 2 and self.phi_influence_rotation_init and hasattr(self, 'rot_angle_unconstrained_2d'): angle_2d = F.softplus(self.rot_angle_unconstrained_2d) * self.phi_angle_scale_2d; cos_a = torch.cos(angle_2d); sin_a = torch.sin(angle_2d); x_comp = x_tan[..., 0]; y_comp = x_tan[..., 1]; x_rot = x_comp * cos_a - y_comp * sin_a; y_rot = x_comp * sin_a + y_comp * cos_a; return torch.stack([x_rot, y_rot], dim=-1)
        return self.rotation_module(x_tan) if self.rotation_module else x_tan
    def forward(self, point_in: torch.Tensor, boundaries_in: Optional[torch.Tensor], descriptor_in: Optional[torch.Tensor], current_c_in: float, current_c_out: float, current_s_in: Optional[float]=None, current_s_out: Optional[float]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        m_in, m_out = PoincareBall(current_c_in), PoincareBall(current_c_out)
        tan_main = m_in.logmap0(point_in); tan_bound = m_in.logmap0(boundaries_in) if boundaries_in is not None else None; tan_desc = m_in.logmap0(descriptor_in) if descriptor_in is not None else None
        tan_main_rot = self._apply_rotation(tan_main); tan_bound_rot = self._apply_rotation(tan_bound); tan_desc_rot = self._apply_rotation(tan_desc)
        def apply_map_and_clamp(tan_vec):
            if tan_vec is None: return None
            mapped_tan = self.non_rotational_map(tan_vec)
            return torch.clamp(mapped_tan, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        tan_main_out_clamped = apply_map_and_clamp(tan_main_rot)
        tan_bound_out_clamped = apply_map_and_clamp(tan_bound_rot)
        tan_desc_out_clamped = apply_map_and_clamp(tan_desc_rot)
        expmap_main_out = m_out.expmap0(tan_main_out_clamped) if tan_main_out_clamped is not None else torch.zeros_like(point_in) # Requires input shape knowledge
        expmap_bound_out = m_out.expmap0(tan_bound_out_clamped) if tan_bound_out_clamped is not None else None
        expmap_desc_out = m_out.expmap0(tan_desc_out_clamped) if tan_desc_out_clamped is not None else None
        return (expmap_main_out, expmap_bound_out, expmap_desc_out)

class HyperbolicWuBuNestingLevel(nn.Module): # Unchanged
    def __init__(self, level_idx: int, dim: int, config: Dict, initial_curvature_val_base: float):
        super().__init__(); self.level_idx, self.dim, self.config = level_idx, dim, config
        self.phi_influence_curvature = config.get("phi_influence_curvature", False)
        self.initial_curvature_val = initial_curvature_val_base * (PHI**(level_idx % 4 - 1.5) if self.phi_influence_curvature else 1.0)
        if self.phi_influence_curvature: logger.info(f"WuBuLevel {level_idx}: Phi-influenced curvature. Base: {initial_curvature_val_base:.2f}, ActualInitial: {self.initial_curvature_val:.2f}")
        self.use_ld = config.get("use_level_descriptors", True); self.use_spread = config.get("use_level_spread", True); self.dropout_rate = config.get("dropout", 0.1); self.ld_init_scale = config.get("level_descriptor_init_scale", 1e-5); self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean"); self.min_curvature = max(EPS, config.get("curvature_min_value", EPS)); self.min_scale = max(EPS, config.get("scale_min_value", EPS)); self.min_spread = max(EPS, config.get("spread_min_value", EPS))
        def _init_unconstrained_param_sigmoid_scaled(target_val, min_val_range, max_val_range):
            if not (min_val_range < max_val_range): logger.warning(f"WuBuLevel {level_idx} Scale Init: Invalid range [{min_val_range}, {max_val_range}]. Defaulting unconstrained to 0."); return torch.tensor(0.0, dtype=torch.float)
            clamped_target_val = torch.clamp(torch.as_tensor(target_val, dtype=torch.float), min_val_range + EPS, max_val_range - EPS).item()
            initial_sigmoid_target = (clamped_target_val - min_val_range) / (max_val_range - min_val_range); initial_sigmoid_target_clamped = max(EPS, min(initial_sigmoid_target, 1.0 - EPS))
            unconstrained_val = math.log(initial_sigmoid_target_clamped / (1.0 - initial_sigmoid_target_clamped)); return torch.tensor(unconstrained_val, dtype=torch.float)
        def _init_unconstrained_param_softplus(target_val, min_val):
            val_for_softplus = max(float(target_val), min_val + EPS) - min_val
            return torch.tensor(math.log(math.expm1(val_for_softplus)) if val_for_softplus > 1e-6 else math.log(val_for_softplus + EPS), dtype=torch.float)
        param_init_args = {'learn_c': ("learnable_curvature", self.initial_curvature_val, self.min_curvature, 'log_curvature_unconstrained', 'softplus'), 'learn_s': ("learnable_scales", "initial_scales", (MIN_WUBU_LEVEL_SCALE, MAX_WUBU_LEVEL_SCALE), 'log_scale_unconstrained', 'sigmoid_scaled'), 'learn_spread': ("learnable_spread", "initial_spread_values", self.min_spread, 'log_spread_unconstrained', 'softplus')}
        for key, (learn_flag_name, init_val_name_or_direct, min_or_range_val_local, param_name, init_type) in param_init_args.items():
            if key == 'learn_spread' and not self.use_spread: self.register_parameter(param_name, None); continue
            learn_flag = config.get(learn_flag_name, True)
            if isinstance(init_val_name_or_direct, str): default_list = [1.0 if key == 'learn_s' else 0.1]; init_list = config.get(init_val_name_or_direct, default_list); init_val = init_list[level_idx] if level_idx < len(init_list) else (init_list[-1] if init_list else default_list[0])
            else: init_val = init_val_name_or_direct
            if init_type == 'softplus': unconstrained_val = _init_unconstrained_param_softplus(init_val, min_or_range_val_local)
            elif init_type == 'sigmoid_scaled': min_r, max_r = min_or_range_val_local; unconstrained_val = _init_unconstrained_param_sigmoid_scaled(init_val, min_r, max_r)
            else: raise ValueError(f"Unknown init_type: {init_type}")
            if learn_flag: setattr(self, param_name, nn.Parameter(unconstrained_val))
            else: self.register_buffer(param_name, unconstrained_val)
        if self.use_ld and self.dim > 0: self.level_descriptor_param = nn.Parameter(torch.Tensor(dim)); PoincareBall(c_scalar=self.initial_curvature_val).init_weights(self.level_descriptor_param, irange=self.ld_init_scale); setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=self.initial_curvature_val))
        else: self.register_parameter('level_descriptor_param', None)
        num_bounds_list = config.get("boundary_points_per_level", [8]); num_boundaries_val = num_bounds_list[level_idx] if level_idx < len(num_bounds_list) else (num_bounds_list[-1] if num_bounds_list else 8)
        self.boundary_manifold_module = BoundaryManifoldHyperbolic(level_idx, num_boundaries_val, dim, initial_manifold_c=self.initial_curvature_val) if self.dim > 0 else None
        comb_in_dim = self.dim; comb_in_dim += self.dim if self.relative_vector_aggregation not in ['none', None] else 0; comb_in_dim += self.dim if self.use_ld else 0; comb_in_dim += 1 if self.use_spread else 0
        comb_h_dims_cfg = config.get("tangent_input_combination_dims", [max(16, comb_in_dim // 2)]); comb_h_dims = comb_h_dims_cfg if isinstance(comb_h_dims_cfg, list) else [comb_h_dims_cfg]
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
        if hasattr(self, 'log_scale_unconstrained') and self.log_scale_unconstrained is not None: scaled_sigmoid = torch.sigmoid(self.log_scale_unconstrained); val_tensor = MIN_WUBU_LEVEL_SCALE + (MAX_WUBU_LEVEL_SCALE - MIN_WUBU_LEVEL_SCALE) * scaled_sigmoid; return val_tensor.item()
        return MIN_WUBU_LEVEL_SCALE
    def get_current_spread_scalar_tensor(self) -> torch.Tensor:
        if self.use_spread and hasattr(self, 'log_spread_unconstrained') and self.log_spread_unconstrained is not None: return get_constrained_param_val(self.log_spread_unconstrained, self.min_spread)
        ref_device = self.log_curvature_unconstrained.device if hasattr(self, 'log_curvature_unconstrained') else torch.device('cpu'); ref_dtype = self.log_curvature_unconstrained.dtype if hasattr(self, 'log_curvature_unconstrained') else torch.float; return torch.tensor(self.min_spread, device=ref_device, dtype=ref_dtype)
    def forward(self, point_in_hyperbolic: torch.Tensor, relative_vectors_tangent_in: Optional[torch.Tensor], descriptor_point_in_hyperbolic: Optional[torch.Tensor], sigma_in_scalar_tensor: Optional[torch.Tensor] ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        # NOTE: The shape assumption (B, S, D) might need adjustment if this is used per-region.
        # If called with (B * N_frames * N_regions, D), the B dim needs careful handling.
        input_shape = point_in_hyperbolic.shape
        # Assume input is now potentially flattened (B*N_frames*N_regions, D) or (B*N_frames, N_regions, D)
        # Let's assume (B', D) where B' = B * N_frames * N_regions for simplicity in this unchanged version.
        # If the input shape changes significantly, logic needs rework.
        # B, S, D_in = point_in_hyperbolic.shape # Original assumption
        if point_in_hyperbolic.dim() == 2: # Assume flattened B'= B*N_frames*N_regions
            B_prime, D_in = point_in_hyperbolic.shape
        else: # Maybe (B, N_frames*N_regions, D) or (B*N_frames, N_regions, D)? Needs clarification on usage.
             raise ValueError(f"WuBuLevel forward expects 2D input (flattened regions), got {point_in_hyperbolic.dim()}D")

        if self.dim == 0: dummy_out_shape = (B_prime, 0); dummy_dtype_dev = {'device':point_in_hyperbolic.device, 'dtype':point_in_hyperbolic.dtype}; current_spread_tensor = self.get_current_spread_scalar_tensor().to(point_in_hyperbolic.dtype); return (torch.zeros(dummy_out_shape, **dummy_dtype_dev), torch.zeros(dummy_out_shape, **dummy_dtype_dev), None, None, current_spread_tensor)

        dev = point_in_hyperbolic.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else point_in_hyperbolic.dtype
        current_c_val = self.get_current_curvature_scalar(); current_s_val = self.get_current_scale_scalar(); current_sigma_out_tensor = self.get_current_spread_scalar_tensor(); current_manifold_obj = PoincareBall(c_scalar=current_c_val)
        if self.level_descriptor_param is not None and hasattr(self.level_descriptor_param, 'manifold'): setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=current_c_val))
        if self.boundary_manifold_module is not None: self.boundary_manifold_module.set_current_manifold_c(current_c_val)

        point_in_proj = current_manifold_obj.proju(point_in_hyperbolic.to(dtype_to_use)); tan_main_component = current_manifold_obj.logmap0(point_in_proj)
        tan_rel_component = torch.zeros_like(tan_main_component); ld_point_self_hyperbolic = None
        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None]:
             # Ensure rel vectors have the same B' dim
             if relative_vectors_tangent_in.shape[0] != B_prime: raise ValueError("Relative vectors shape mismatch")
             tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)

        if self.use_ld and self.level_descriptor_param is not None: ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use))

        tan_desc_prev_level_component = torch.zeros_like(tan_main_component)
        if descriptor_point_in_hyperbolic is not None and self.use_ld :
             if descriptor_point_in_hyperbolic.shape[0] != B_prime: raise ValueError("Descriptor input shape mismatch")
             desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use)); tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)

        inputs_for_combiner = [tan_main_component]; inputs_for_combiner.append(tan_rel_component) if self.relative_vector_aggregation not in ['none', None] else None; inputs_for_combiner.append(tan_desc_prev_level_component) if self.use_ld else None
        if self.use_spread and sigma_in_scalar_tensor is not None:
            # Expand sigma correctly for B'
            sigma_in_expanded = sigma_in_scalar_tensor.view(-1,1).expand(B_prime, 1).to(dtype_to_use) # Assume sigma_in is per B or single value
            inputs_for_combiner.append(sigma_in_expanded)

        combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1); v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features)
        v_final_for_expmap_unclamped = v_combined_tangent_processed * current_s_val
        if self.use_flow and self.tangent_flow_module is not None: flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val; v_final_for_expmap_unclamped = v_final_for_expmap_unclamped + flow_effect
        scaled_output_tangent_for_expmap = torch.clamp(v_final_for_expmap_unclamped, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        point_this_level_out_hyperbolic = current_manifold_obj.expmap0(scaled_output_tangent_for_expmap); tangent_out_for_aggregation = v_combined_tangent_processed.to(dtype_to_use)
        boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype_to_use) if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None else None
        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None:
            # Handle descriptor expansion for B'
            if ld_point_self_hyperbolic.dim() == 1: descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B_prime, -1).to(dtype_to_use)
            else: # If descriptor is already per-item, just use it
                 descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.to(dtype_to_use)

        # Ensure outputs have B_prime as the first dimension
        return (point_this_level_out_hyperbolic.to(dtype_to_use),
                tangent_out_for_aggregation,
                descriptor_point_out_for_transform_hyperbolic,
                boundary_points_this_level_hyperbolic, # This is global per level, not per item B'
                current_sigma_out_tensor.to(dtype_to_use))

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_tangent_dim: int, output_tangent_dim: int, config: Dict):
        super().__init__(); self.input_tangent_dim, self.output_tangent_dim, self.config = input_tangent_dim, output_tangent_dim, config; self.num_levels = config.get("num_levels", 3); assert self.num_levels >= 0; self.hyperbolic_dims_list = config.get("hyperbolic_dims", []); self.initial_curvatures_list = config.get("initial_curvatures", []); self.dropout_val = config.get("dropout", 0.1); self.relative_vector_aggregation_mode = config.get("relative_vector_aggregation", "mean"); self.aggregation_method_mode = config.get("aggregation_method", "concat_tangent"); assert self.aggregation_method_mode == "concat_tangent"; self.use_rotation_in_transform_flag = config.get("use_rotation_in_transform", False); self.phi_influence_rotation_init = config.get("phi_influence_rotation_init", False)

        first_level_dim = self.hyperbolic_dims_list[0] if self.num_levels > 0 and self.hyperbolic_dims_list else 0
        self.input_tangent_projection = nn.Linear(input_tangent_dim, first_level_dim) if input_tangent_dim > 0 and first_level_dim > 0 and input_tangent_dim != first_level_dim else nn.Identity()
        self.input_tangent_layernorm = nn.LayerNorm(first_level_dim) if first_level_dim > 0 else nn.Identity()

        self.levels_modulelist = nn.ModuleList(); self.transforms_modulelist = nn.ModuleList()
        if self.num_levels > 0:
            for i in range(self.num_levels): self.levels_modulelist.append(HyperbolicWuBuNestingLevel(i, self.hyperbolic_dims_list[i], self.config, self.initial_curvatures_list[i]))
            num_transforms = max(0, self.num_levels - 1)
            if num_transforms > 0:
                transform_types_list = config.get("transform_types", ["linear"] * num_transforms); transform_hidden_dims_list = config.get("transform_hidden_dims", [None] * num_transforms)
                for i in range(num_transforms):
                    if i+1 < len(self.hyperbolic_dims_list) and i+1 < len(self.initial_curvatures_list): self.transforms_modulelist.append(HyperbolicInterLevelTransform(self.hyperbolic_dims_list[i], self.hyperbolic_dims_list[i+1], self.initial_curvatures_list[i], self.initial_curvatures_list[i+1], transform_types_list[i] if i < len(transform_types_list) else "linear", transform_hidden_dims_list[i] if i < len(transform_hidden_dims_list) else None, self.dropout_val, self.use_rotation_in_transform_flag, self.phi_influence_rotation_init, level_idx_for_phi=i))
                    else: logger.warning(f"Skipping transform {i} due to insufficient config for next level.")

        actual_output_dims_from_levels = [d for d_idx, d in enumerate(self.hyperbolic_dims_list[:self.num_levels]) if d > 0 and d_idx < len(self.levels_modulelist)];
        aggregated_tangent_dim_val = sum(actual_output_dims_from_levels) if actual_output_dims_from_levels else input_tangent_dim
        self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, output_tangent_dim) if aggregated_tangent_dim_val > 0 and output_tangent_dim > 0 and aggregated_tangent_dim_val != output_tangent_dim else nn.Identity()

        self.apply(init_weights_general); param_count = sum(p.numel() for p in self.parameters() if p.requires_grad); logger.info(f"FullyHypWuBuModel: {self.num_levels} levels. {param_count:,} params. InTangentDim {input_tangent_dim}, AggTangentDim {aggregated_tangent_dim_val}, OutTangentDim {output_tangent_dim}")

    def forward(self, x_initial_tangent_in: torch.Tensor) -> torch.Tensor:
        # Input can be 2D (B_prime, input_tangent_dim) for regional processing (S, M)
        # OR 3D (Batch, SeqLen_Time, input_tangent_dim) for temporal processing (T)
        if self.num_levels == 0:
            return self.output_tangent_projection(x_initial_tangent_in)

        input_dim = x_initial_tangent_in.dim()
        B_orig, S_orig, D_orig = -1, -1, -1 # For 3D input
        B_prime = -1 # For 2D input

        if input_dim == 3: # e.g., (Batch, SeqLen_Time, Dim) for WuBu-T
            B_orig, S_orig, D_orig = x_initial_tangent_in.shape
            if D_orig != self.input_tangent_dim:
                raise ValueError(f"Input feature dim {D_orig} does not match model's input_tangent_dim {self.input_tangent_dim}")
            # Reshape to (Batch * SeqLen_Time, Dim) for level processing
            x_proc = x_initial_tangent_in.reshape(B_orig * S_orig, D_orig)
            B_prime_for_levels = B_orig * S_orig
        elif input_dim == 2: # e.g., (Batch*NumRegions, Dim) for WuBu-S/M
            B_prime, D_orig = x_initial_tangent_in.shape
            if D_orig != self.input_tangent_dim:
                raise ValueError(f"Input feature dim {D_orig} does not match model's input_tangent_dim {self.input_tangent_dim}")
            x_proc = x_initial_tangent_in
            B_prime_for_levels = B_prime
        else:
            raise ValueError(f"FullyHyperbolicWuBuNestingModel expects 2D or 3D input, got {input_dim}D")

        dev = x_proc.device
        ref_param_for_dtype = next(iter(self.parameters()), None)
        dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else x_proc.dtype
        x_proc = x_proc.to(dtype_to_use)

        current_tangent_projected = self.input_tangent_projection(x_proc)
        current_tangent_for_level0 = self.input_tangent_layernorm(current_tangent_projected)

        if not self.levels_modulelist:
            # If input was 3D, reshape output back to 3D
            out_zeros = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use)
            if input_dim == 3:
                return out_zeros.reshape(B_orig, S_orig, self.output_tangent_dim)
            return out_zeros


        level0_module = self.levels_modulelist[0]; c0_val = level0_module.get_current_curvature_scalar(); m0_obj = PoincareBall(c_scalar=c0_val)
        # Ensure current_tangent_for_level0 has features if hyperbolic_dims_list[0] is > 0
        if self.hyperbolic_dims_list[0] > 0 :
            current_point_repr_hyperbolic = m0_obj.expmap0(current_tangent_for_level0)
        else: # Handle case where first level might be 0-dim (shouldn't happen if first_level_dim check passed)
            current_point_repr_hyperbolic = torch.empty(B_prime_for_levels, 0, device=dev, dtype=dtype_to_use)


        level_tangent_outputs_for_aggregation = []; aggregated_relative_vectors_from_prev_transform = None; descriptor_from_prev_transform_hyperbolic = None;
        sigma_from_prev_level_tensor = torch.tensor(0.0, device=dev, dtype=dtype_to_use)

        for i in range(self.num_levels):
            level_module = self.levels_modulelist[i]
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation,
             descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic,
             sigma_out_of_level_tensor) = level_module(current_point_repr_hyperbolic,
                                                       aggregated_relative_vectors_from_prev_transform,
                                                       descriptor_from_prev_transform_hyperbolic,
                                                       sigma_from_prev_level_tensor)

            if self.hyperbolic_dims_list[i] > 0: level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)

            if i < self.num_levels - 1:
                if i >= len(self.transforms_modulelist): logger.warning(f"Missing transform for level {i} to {i+1}. Stop."); break
                transform_module = self.transforms_modulelist[i]; next_level_module = self.levels_modulelist[i+1]
                c_in_for_transform = level_module.get_current_curvature_scalar(); c_out_for_transform = next_level_module.get_current_curvature_scalar()

                (point_transformed_to_next_level_hyperbolic,
                 boundaries_transformed_to_next_level_hyperbolic,
                 descriptor_transformed_to_next_level_hyperbolic
                 ) = transform_module(point_out_of_level_hyperbolic,
                                      boundary_points_of_level_hyperbolic,
                                      descriptor_generated_by_level_hyperbolic,
                                      c_in_for_transform, c_out_for_transform)

                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic
                descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic
                sigma_from_prev_level_tensor = sigma_out_of_level_tensor

                aggregated_relative_vectors_from_prev_transform = None
                if boundaries_transformed_to_next_level_hyperbolic is not None and \
                   self.relative_vector_aggregation_mode not in ['none', None] and \
                   self.hyperbolic_dims_list[i+1] > 0 and \
                   current_point_repr_hyperbolic.shape[-1] > 0 : # Ensure main point has features

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
                    aggregated_relative_vectors_from_prev_transform = torch.zeros_like(tan_main_next_level) if agg_rel_vec is not None and not torch.isfinite(agg_rel_vec).all() else agg_rel_vec


        if not level_tangent_outputs_for_aggregation:
            out_zeros = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use)
            if input_dim == 3: return out_zeros.reshape(B_orig, S_orig, self.output_tangent_dim)
            return out_zeros

        compatible_tangent_outputs = [t_val.to(dtype_to_use) for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation) if t_val is not None and t_idx < len(self.hyperbolic_dims_list) and self.hyperbolic_dims_list[t_idx] > 0 and torch.isfinite(t_val).all()]

        if not compatible_tangent_outputs:
             out_zeros = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use)
             if input_dim == 3: return out_zeros.reshape(B_orig, S_orig, self.output_tangent_dim)
             return out_zeros

        aggregated_tangent_final = torch.cat(compatible_tangent_outputs, dim=-1)
        final_output_flat = self.output_tangent_projection(aggregated_tangent_final) # Shape (B_prime_for_levels, output_tangent_dim)

        final_output_flat = torch.nan_to_num(final_output_flat, nan=0.0) if not torch.isfinite(final_output_flat).all() else final_output_flat

        # Reshape back to original input's batch/sequence structure if it was 3D
        if input_dim == 3:
            return final_output_flat.reshape(B_orig, S_orig, self.output_tangent_dim)
        else: # Input was 2D
            return final_output_flat

# GradientStats, HAKMEMQController, RiemannianEnhancedSGD - Unchanged from v0.05.2
class GradientStats:
    def __init__(self): self.reset()
    def reset(self): self.total_params_updated = 0; self.total_finite_grads_processed = 0; self.total_non_finite_grads_encountered = 0; self.params_skipped_due_non_finite_grad = 0; self.max_grad_norm_observed = 0.; self.step_summary = {}
    def record_param_grad(self, grad_is_finite: bool, original_norm_if_finite: Optional[float] = None):
        if grad_is_finite: self.total_finite_grads_processed += 1; self.max_grad_norm_observed = max(self.max_grad_norm_observed, original_norm_if_finite) if original_norm_if_finite is not None else self.max_grad_norm_observed
        else: self.total_non_finite_grads_encountered += 1; self.params_skipped_due_non_finite_grad += 1
    def finalize_step_stats(self, num_params_in_optimizer_step: int): self.total_params_updated = num_params_in_optimizer_step - self.params_skipped_due_non_finite_grad; self.step_summary = {"params_in_step": num_params_in_optimizer_step, "params_updated": self.total_params_updated, "params_skipped_non_finite_grad": self.params_skipped_due_non_finite_grad, "initial_finite_grads": self.total_finite_grads_processed, "initial_non_finite_grads": self.total_non_finite_grads_encountered, "max_finite_grad_norm_observed": self.max_grad_norm_observed}
    def get_step_summary_for_logging(self) -> dict: return self.step_summary.copy()

class HAKMEMQController:
    def __init__(self, learning_rate:float=0.01, discount:float=0.95, epsilon:float=0.2, epsilon_decay:float=0.9998, min_epsilon:float=0.01, lr_scale_options:Optional[List[float]]=None, momentum_scale_options:Optional[List[float]]=None, max_q_table_size:int=10000):
        self.q_table: Dict[Tuple, Dict[str, np.ndarray]] = {}; self.alpha = learning_rate; self.gamma = discount; self.epsilon = epsilon; self.min_epsilon = min_epsilon; self.epsilon_decay = epsilon_decay; self.prev_loss: Optional[float] = None; self.prev_state: Optional[Tuple] = None; self.prev_action: Optional[Dict[str, float]] = None
        self.action_ranges = {'lr_scale': np.array(lr_scale_options if lr_scale_options else [0.9,0.95,1.,1.05,1.1]), 'momentum_scale': np.array(momentum_scale_options if momentum_scale_options else [0.95,0.98,1.,1.01,1.02])}; self.num_actions = {p: len(s) for p, s in self.action_ranges.items()}
        self.loss_window = deque(maxlen=20); self.grad_norm_window = deque(maxlen=20); self.lr_window = deque(maxlen=10); self.momentum_window = deque(maxlen=10); self.performance_window = deque(maxlen=50); self.stable_steps = 0; self.oscillation_counter = 0; self.prev_actions_log = deque(maxlen=10); self.max_q_table_size = max(100, max_q_table_size); self.q_table_access_count: Dict[Tuple, int] = defaultdict(int); self.q_table_creation_time: Dict[Tuple, float] = {}; self.flow_coefficient = 0.05; self.oscillation_penalty = 0.15; self.stability_reward_bonus = 0.05
    def get_state(self, lr: float, momentum: float, grad_norm: Optional[float], loss: Optional[float]) -> Optional[Tuple]:
        if loss is None or grad_norm is None or not np.isfinite(loss) or not np.isfinite(grad_norm): return None
        self.loss_window.append(loss); self.grad_norm_window.append(grad_norm); self.lr_window.append(lr); self.momentum_window.append(momentum); loss_trend_bin, grad_norm_level_bin, lr_level_bin, momentum_level_bin, oscillation_bin = 2,2,2,1,0
        if len(self.loss_window) < 5 or len(self.grad_norm_window) < 5: return None
        try:
            loss_arr = np.array(list(self.loss_window)[-10:]); slope_loss = np.polyfit(np.arange(len(loss_arr)), loss_arr, 1)[0] if len(loss_arr) >= 3 and len(np.unique(loss_arr)) > 1 else 0.0; loss_trend_bin = np.digitize(slope_loss / (abs(np.median(loss_arr)) + EPS), bins=[-0.05,-0.005,0.005,0.05]).item()
            grad_norm_level_bin = np.digitize(np.median(list(self.grad_norm_window)), bins=[0.1,0.5,1.5,5.0]).item(); lr_level_bin = np.digitize(lr / 1e-4, bins=[0.5,2.0,10.0,50.0]).item(); momentum_level_bin = np.digitize(momentum, bins=[0.85,0.92,0.97]).item()
            if len(self.performance_window) >= 5: recent_rewards = np.sign([r for r in list(self.performance_window)[-5:] if r != 0]); self.oscillation_counter = min(self.oscillation_counter + 1, 5) if len(recent_rewards) >= 2 and np.sum(np.abs(np.diff(recent_rewards))) / 2.0 >= 2 else max(0, self.oscillation_counter - 1)
            oscillation_bin = 1 if self.oscillation_counter >= 3 else 0
        except Exception: return None
        state_tuple = (loss_trend_bin, grad_norm_level_bin, oscillation_bin, lr_level_bin, momentum_level_bin); self.q_table_access_count[state_tuple] += 1; return state_tuple
    def compute_reward(self, current_loss: Optional[float], prev_loss: Optional[float], grad_norm: Optional[float]) -> float:
        if current_loss is None or prev_loss is None or grad_norm is None or not np.isfinite(current_loss) or not np.isfinite(prev_loss) or not np.isfinite(grad_norm): return 0.
        median_loss = np.median(list(self.loss_window)[:-1]) if len(self.loss_window) > 1 else prev_loss; base_reward = np.tanh((prev_loss - current_loss) / (abs(median_loss) + EPS) * 10.); grad_penalty = -0.1 * min(1., max(0., (grad_norm - 5.) / 10.)) if grad_norm > 5. else 0.; osc_penalty = -self.oscillation_penalty if self.oscillation_counter >= 3 else 0.
        current_perf_reward = base_reward + grad_penalty + osc_penalty; self.performance_window.append(current_perf_reward); self.stable_steps = self.stable_steps + 1 if current_perf_reward > 0.01 else 0; stab_bonus = min(0.15, self.stability_reward_bonus * math.log1p(self.stable_steps / 5.)) if current_perf_reward > 0.01 else 0.; return float(np.clip(base_reward + grad_penalty + osc_penalty + stab_bonus, -1., 1.))
    def choose_action(self, state: Optional[Tuple]) -> Dict[str, float]:
        if state is None: return {'lr_scale': 1., 'momentum_scale': 1.}
        if state not in self.q_table: self.q_table[state] = {p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}; self.q_table_creation_time[state] = time.time(); self.q_table_access_count[state] = 1; self._manage_q_table_size()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay); chosen_actions = {}
        for p_type, q_vals in self.q_table[state].items():
            act_space = self.action_ranges[p_type]
            if random.random() < self.epsilon: chosen_idx = random.randrange(len(act_space))
            else:
                finite_q_mask = np.isfinite(q_vals)
                if not np.any(finite_q_mask): chosen_idx = random.randrange(len(act_space))
                else:
                    best_q_val = np.max(q_vals[finite_q_mask])
                    best_indices = np.where(finite_q_mask)[0][np.isclose(q_vals[finite_q_mask], best_q_val)]
                    chosen_idx = random.choice(best_indices) if len(best_indices) > 0 else random.randrange(len(act_space))
            chosen_actions[p_type] = float(act_space[chosen_idx])
        self.prev_actions_log.append(chosen_actions.copy()); return chosen_actions
    def update(self, state: Optional[Tuple], action: Optional[Dict[str, float]], reward: float, next_state: Optional[Tuple]):
        if state is None or next_state is None or action is None or state not in self.q_table: return
        if next_state not in self.q_table: self.q_table[next_state] = {p: np.zeros(self.num_actions[p]) for p in self.action_ranges.keys()}; self.q_table_creation_time[next_state] = time.time(); self.q_table_access_count[next_state] = 0; self._manage_q_table_size()
        for p_type, chosen_val in action.items():
            if p_type not in self.q_table[state]: continue
            act_idx_arr = np.where(np.isclose(self.action_ranges[p_type], chosen_val))[0]; act_idx = act_idx_arr[0] if len(act_idx_arr) > 0 else -1
            if act_idx == -1: continue
            current_q = self.q_table[state][p_type][act_idx]; next_q_vals = self.q_table[next_state][p_type]; finite_next_q = next_q_vals[np.isfinite(next_q_vals)]; max_future_q = np.max(finite_next_q) if len(finite_next_q) > 0 else 0.0; max_future_q = 0.0 if not np.isfinite(max_future_q) else max_future_q
            td_target = reward + self.gamma * max_future_q; td_error = td_target - current_q; adaptive_alpha = min(0.5, max(0.001, self.alpha * (1.0 + self.flow_coefficient * np.tanh(abs(td_error) * 0.5)))); new_q = current_q + adaptive_alpha * td_error; self.q_table[state][p_type][act_idx] = np.clip(new_q, -1e4, 1e4) if np.isfinite(new_q) else 0.0
    def _manage_q_table_size(self):
        if len(self.q_table) > self.max_q_table_size:
            can_smart_prune = all([self.q_table_access_count, self.q_table_creation_time, len(self.q_table_access_count) > len(self.q_table) // 2, len(self.q_table_creation_time) > len(self.q_table) // 2])
            to_remove = sorted(self.q_table.keys(), key=lambda s: (self.q_table_access_count.get(s,0), self.q_table_creation_time.get(s, float('inf'))))[:len(self.q_table)-self.max_q_table_size] if can_smart_prune else random.sample(list(self.q_table.keys()), len(self.q_table) - self.max_q_table_size)
            for s_rm in to_remove: self.q_table.pop(s_rm, None); self.q_table_access_count.pop(s_rm, None); self.q_table_creation_time.pop(s_rm, None)
    def get_info(self) -> Dict: q_mem_mb = sum(sys.getsizeof(s) + sum(a.nbytes + sys.getsizeof(k) for k,a in v.items()) for s,v in self.q_table.items())/(1024**2) if self.q_table else 0.0; avg_perf_reward = np.mean(list(self.performance_window)) if self.performance_window else 0.; return {"epsilon": self.epsilon, "q_table_size": len(self.q_table), "q_table_mem_mb_approx": round(q_mem_mb, 2), "last_action": self.prev_actions_log[-1] if self.prev_actions_log else None, f"avg_reward_last_{self.performance_window.maxlen}": avg_perf_reward, "stable_steps": self.stable_steps, "oscillation_counter": self.oscillation_counter}

class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.01, max_grad_norm_risgd=1.0, q_learning_config:Optional[Dict]=None):
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum, base_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_learning_config) if isinstance(q_learning_config, dict) else None
        logger.info(f"RiSGD: Q-Controller {'en' if self.q_controller else 'dis'}abled.")
        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count = 0; self.current_loss_for_q: Optional[float] = None; self.grad_stats = GradientStats()
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad: self.state.setdefault(p, {})
    def zero_grad(self, set_to_none: bool = True): super().zero_grad(set_to_none=set_to_none); self.grad_stats.reset()
    def set_current_loss_for_q_controller(self, loss: Optional[float]): self.current_loss_for_q = loss if loss is not None and np.isfinite(loss) else None;
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.q_controller and self.q_controller.prev_action:
            q_action = self.q_controller.prev_action
            for group in self.param_groups: group.setdefault('base_lr', group['lr']); group.setdefault('base_momentum', group['momentum']); group['lr'] = float(np.clip(group['base_lr'] * q_action.get('lr_scale', 1.0), 1e-8, 1.0)); group['momentum'] = float(np.clip(group['base_momentum'] * q_action.get('momentum_scale', 1.0), 0.1, 0.999))
        num_params_total = 0
        for group in self.param_groups:
            lr, mom, wd = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                num_params_total += 1; grad = p.grad.data; finite_grad = torch.isfinite(grad).all(); norm_val = grad.norm().item() if finite_grad else None; self.grad_stats.record_param_grad(finite_grad, norm_val)
                if not finite_grad: self.state[p].pop('momentum_buffer', None); continue
                if norm_val is not None and norm_val > self.max_grad_norm_risgd and self.max_grad_norm_risgd > 0: grad.mul_(self.max_grad_norm_risgd / (norm_val + EPS))
                state = self.state[p]; manifold: Optional[Manifold] = getattr(p, 'manifold', None)
                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    # Ensure p.data is projected before use
                    p_proj = manifold.proju(p.data)
                    try:
                        r_grad = manifold.egrad2rgrad(p_proj, grad)
                        if not torch.isfinite(r_grad).all():
                            logger.warning(f"RiSGD Step: Non-finite Riemannian gradient for param shape {p.shape}. Skipping param update.")
                            self.grad_stats.params_skipped_due_non_finite_grad += 1; state.pop('momentum_buffer', None); continue

                        update_vec = r_grad
                        if wd != 0:
                           try:
                               log_p = manifold.logmap0(p_proj)
                               if torch.isfinite(log_p).all(): update_vec = update_vec.add(log_p, alpha=wd)
                               else: logger.warning(f"RiSGD Step: Non-finite logmap for weight decay on param {p.shape}. Skipping WD.")
                           except Exception as e_wd: logger.warning(f"RiSGD Step: Error in weight decay logmap for param {p.shape}: {e_wd}. Skipping WD.")

                        buf = state.setdefault('momentum_buffer', torch.zeros_like(update_vec)); buf.mul_(mom).add_(update_vec)
                        if not torch.isfinite(buf).all(): logger.warning(f"RiSGD Step: Non-finite momentum buffer for param {p.shape}. Resetting buffer."); buf.zero_()

                        try:
                            # Use p_proj as the point for expmap, apply update
                            expmap_arg = buf.mul(-lr)
                            # Check argument to expmap
                            if not torch.isfinite(expmap_arg).all():
                                logger.warning(f"RiSGD Step: Non-finite argument to expmap for param {p.shape}. Resetting update.")
                                state.get('momentum_buffer', torch.zeros(0)).zero_() # Reset momentum
                                continue # Skip update for this parameter

                            # Compute expmap from the *projected* point
                            new_p_candidate = manifold.expmap(p_proj, expmap_arg)

                            # Check result of expmap
                            if not torch.isfinite(new_p_candidate).all():
                                logger.warning(f"RiSGD Step: Non-finite result from expmap for param {p.shape}. Trying fallback.")
                                new_p_candidate = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0)) # Project sanitized value
                                state.get('momentum_buffer', torch.zeros(0)).zero_() # Reset momentum as state is suspect

                            # Final projection and assignment
                            p.data = manifold.proju(new_p_candidate)

                            # Final check
                            if not torch.isfinite(p.data).all():
                                logger.error(f"RiSGD Step: Parameter became non-finite after update and projection for param {p.shape}. Resetting to origin.")
                                p.data = manifold.expmap0(torch.zeros_like(p.data)) # Reset to origin
                                state.get('momentum_buffer', torch.zeros(0)).zero_() # Reset momentum

                        except Exception as e_hyp_update:
                            logger.error(f"RiSGD Step: Error during hyperbolic update for param {p.shape}: {e_hyp_update}. Resetting momentum.")
                            state.get('momentum_buffer', torch.zeros(0)).zero_()
                            # Optionally reset param data here too, e.g., p.data = manifold.proju(p_proj)

                    except Exception as e_egrad:
                        logger.error(f"RiSGD Step: Error calculating Riemannian gradient for param {p.shape}: {e_egrad}. Skipping param update.")
                        self.grad_stats.params_skipped_due_non_finite_grad += 1; state.pop('momentum_buffer', None); continue

                else: # Euclidean parameter update
                    d_p = grad.clone();
                    if wd != 0: d_p.add_(p.data, alpha=wd)
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(p.data)); buf.mul_(mom).add_(d_p)
                    if not torch.isfinite(buf).all(): logger.warning(f"RiSGD Step (Euc): Non-finite momentum buffer for param {p.shape}. Resetting."); buf.zero_()
                    p.data.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all(): logger.warning(f"RiSGD Step (Euc): Param became non-finite after update {p.shape}. Resetting momentum & NaN."); p.data = torch.nan_to_num(p.data, nan=0.0); state.get('momentum_buffer', torch.zeros(0)).zero_()

        self.grad_stats.finalize_step_stats(num_params_total); self._step_count += 1; return loss
    def get_q_controller_info(self) -> Dict: return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}
    def get_gradient_stats_summary(self) -> Dict: return self.grad_stats.get_step_summary_for_logging()


# =====================================================================
# GAAD Components (Largely Unchanged, but usage might adapt)
# =====================================================================
def golden_subdivide_rect_fixed_n(frame_dims:Tuple[int,int], num_regions_target:int, device='cpu', dtype=torch.float, min_size_px=5) -> torch.Tensor: # Unchanged
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

def phi_spiral_patch_centers_fixed_n(frame_dims:Tuple[int,int], num_centers:int, device='cpu', dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]: # Unchanged
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

# Removed GAADFrameProcessor (v0.05.2), as encoding is now integrated differently.
# Removed GAADMotionRegionProposal (v0.05.2), motion region proposal integrated into encoder.

# =====================================================================
# NEW Architectural Components for v0.10.0
# =====================================================================

class RegionalPatchExtractor(nn.Module):
    """
    Extracts patches (pixels or features) based on GAAD bounding boxes.
    Optionally uses RoIAlign if a feature extractor is provided.
    """
    def __init__(self,
                 patch_output_size: Optional[Tuple[int, int]] = None, # e.g., (16, 16) for pixel patches, or None if using RoIAlign
                 feature_extractor: Optional[nn.Module] = None, # e.g., a shallow CNN
                 feature_map_spatial_scale: float = 1.0, # Scale factor from input pixels to feature map
                 roi_align_output_size: Optional[Tuple[int, int]] = None, # e.g., (4, 4) if using RoIAlign
                 use_roi_align: bool = False):
        super().__init__()
        self.patch_output_size = patch_output_size
        self.feature_extractor = feature_extractor
        self.feature_map_spatial_scale = feature_map_spatial_scale
        self.roi_align_output_size = roi_align_output_size
        self.use_roi_align = use_roi_align

        if self.use_roi_align:
            if self.feature_extractor is None or self.roi_align_output_size is None:
                raise ValueError("feature_extractor and roi_align_output_size must be provided if use_roi_align=True")
            logger.info(f"RegionalPatchExtractor: Using RoIAlign with output {roi_align_output_size}")
        else:
            if self.patch_output_size is None:
                 raise ValueError("patch_output_size must be provided if use_roi_align=False (pixel patches)")
            logger.info(f"RegionalPatchExtractor: Extracting Pixel Patches, resizing to {patch_output_size}")
            self.resize_transform = T.Resize(patch_output_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)

    def forward(self, images: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Input images (B, C, H, W)
            bboxes_batch (torch.Tensor): Bounding boxes (B, NumRegions, 4) in [x1, y1, x2, y2] format (pixel coords)

        Returns:
            torch.Tensor: Extracted regional patches/features (B, NumRegions, C_out, H_patch, W_patch)
                         or (B, NumRegions, FeatureDim) if flattened later.
        """
        B, NumRegions, _ = bboxes_batch.shape
        device = images.device
        original_images_dtype = images.dtype # Store original dtype

        # Determine compute dtype, prefer float32 for conv/roi_align if input is uint8
        compute_dtype = torch.float32 if images.dtype == torch.uint8 else images.dtype
        images_for_processing = images.to(compute_dtype)


        if self.use_roi_align and self.feature_extractor is not None and self.roi_align_output_size is not None:
            feature_maps = self.feature_extractor(images_for_processing) # (B, C_feat, H_feat, W_feat)
            
            scaled_bboxes_for_roialign_list = []
            h_feat, w_feat = feature_maps.shape[2:] 
            max_w_feat_scalar = float(w_feat)
            max_h_feat_scalar = float(h_feat)

            for b in range(B):
                current_bboxes_scaled = bboxes_batch[b].to(torch.float32) * self.feature_map_spatial_scale

                # Clamp x1, y1 (top-left)
                current_bboxes_scaled[:, 0] = torch.clamp(current_bboxes_scaled[:, 0], min=0.0, max=max_w_feat_scalar - EPS) 
                current_bboxes_scaled[:, 1] = torch.clamp(current_bboxes_scaled[:, 1], min=0.0, max=max_h_feat_scalar - EPS) 

                # Clamp x2, y2 (bottom-right), ensuring x2 >= x1 and y2 >= y1 (corrected from x2 > x1)
                min_for_x2 = current_bboxes_scaled[:, 0] # No +EPS needed if we allow zero-width/height before RoIAlign handles it
                current_bboxes_scaled[:, 2] = torch.clamp(current_bboxes_scaled[:, 2], max=max_w_feat_scalar) 
                current_bboxes_scaled[:, 2] = torch.maximum(current_bboxes_scaled[:, 2], min_for_x2)       

                min_for_y2 = current_bboxes_scaled[:, 1] # No +EPS needed
                current_bboxes_scaled[:, 3] = torch.clamp(current_bboxes_scaled[:, 3], max=max_h_feat_scalar) 
                current_bboxes_scaled[:, 3] = torch.maximum(current_bboxes_scaled[:, 3], min_for_y2)       

                batch_indices = torch.full((NumRegions, 1), float(b), device=device, dtype=current_bboxes_scaled.dtype)
                scaled_bboxes_for_roialign_list.append(torch.cat([batch_indices, current_bboxes_scaled], dim=1))

            all_rois = torch.cat(scaled_bboxes_for_roialign_list, dim=0) 

            aligned_features = roi_align(feature_maps, all_rois, # feature_maps is already float
                                         output_size=self.roi_align_output_size,
                                         spatial_scale=1.0, 
                                         aligned=True)
            C_feat = feature_maps.shape[1]
            H_roi, W_roi = self.roi_align_output_size
            aligned_features = aligned_features.view(B, NumRegions, C_feat, H_roi, W_roi)
            return aligned_features.to(original_images_dtype) # Cast back to original input dtype

        else: # Extract pixel patches
            all_patches = []
            for b in range(B):
                batch_patches = []
                for r in range(NumRegions):
                    x1, y1, x2, y2 = bboxes_batch[b, r].round().int().tolist()
                    x1_c, y1_c = max(0, x1), max(0, y1) 
                    x2_c, y2_c = min(images.shape[3], x2), min(images.shape[2], y2)

                    if x1_c >= x2_c or y1_c >= y2_c: 
                         patch = torch.zeros((images.shape[1],) + self.patch_output_size, device=device, dtype=original_images_dtype)
                    else:
                        patch = images_for_processing[b, :, y1_c:y2_c, x1_c:x2_c] # Use images_for_processing
                        patch = self.resize_transform(patch) 
                    batch_patches.append(patch)
                all_patches.append(torch.stack(batch_patches)) 
            return torch.stack(all_patches).to(original_images_dtype) # Ensure output matches original input type



class PatchEmbed(nn.Module):
    """ Simple patch embedding: Flatten + Linear """
    def __init__(self, patch_feature_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(patch_feature_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, NumRegions, C_feat, H_patch, W_patch) or similar
        B, N, *patch_dims = x.shape
        x = x.view(B, N, -1) # Flatten patch dimensions -> (B, N, patch_feature_dim)
        x = self.proj(x)     # Project to embed_dim -> (B, N, embed_dim)
        return x


class RegionalHyperbolicEncoder(nn.Module):
    """ Encodes frames into sets of regional hyperbolic features using GAAD + WuBu. """
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_config: Dict):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_config = gaad_config
        self.wubu_s_config = wubu_s_config
        self.image_size = (args.image_h, args.image_w)
        self.num_appearance_regions = gaad_config['num_regions']
        self.decomposition_type = gaad_config['decomposition_type']
        self.gaad_min_size_px = gaad_config.get('min_size_px', 5)

        # 1. Feature Extractor (Optional, for RoIAlign) - Start with None
        self.feature_extractor: Optional[nn.Module] = None
        feature_map_scale = 1.0
        patch_input_channels = self.video_config['num_channels']
        roi_align_output_size = None
        use_roi_align = False # Default to pixel patches

        if args.encoder_use_roi_align:
             # Example: Define a shallow CNN
             self.feature_extractor = nn.Sequential(
                 nn.Conv2d(self.video_config['num_channels'], args.encoder_shallow_cnn_channels, kernel_size=3, stride=1, padding=1),
                 nn.GroupNorm(8, args.encoder_shallow_cnn_channels),
                 nn.GELU()
                 # Add more layers if needed, update feature_map_scale accordingly
             )
             # Assuming stride 1 for now
             patch_input_channels = args.encoder_shallow_cnn_channels
             roi_align_output_size = (args.encoder_roi_align_output_h, args.encoder_roi_align_output_w)
             use_roi_align = True
             logger.info(f"Encoder: Using RoIAlign with shallow CNN (out_ch: {patch_input_channels})")
        else:
             logger.info(f"Encoder: Using Pixel Patch Extraction (resize: {args.encoder_pixel_patch_size})")

        # 2. Patch Extractor
        self.patch_extractor = RegionalPatchExtractor(
            patch_output_size=(args.encoder_pixel_patch_size, args.encoder_pixel_patch_size) if not use_roi_align else None,
            feature_extractor=self.feature_extractor,
            feature_map_spatial_scale=feature_map_scale,
            roi_align_output_size=roi_align_output_size,
            use_roi_align=use_roi_align
        )

        # 3. Patch Embedding
        patch_output_h = roi_align_output_size[0] if use_roi_align else args.encoder_pixel_patch_size
        patch_output_w = roi_align_output_size[1] if use_roi_align else args.encoder_pixel_patch_size
        patch_feature_dim = patch_input_channels * patch_output_h * patch_output_w
        self.patch_embed = PatchEmbed(patch_feature_dim, args.encoder_initial_tangent_dim)

        # 4. WuBu-S (Appearance)
        self.wubu_s = FullyHyperbolicWuBuNestingModel(
            input_tangent_dim=args.encoder_initial_tangent_dim,
            output_tangent_dim=video_config['wubu_s_output_dim'], # This is now the final tangent dim after aggregation
            config=wubu_s_config
        )
        self.wubu_s_final_hyp_dim = wubu_s_config['hyperbolic_dims'][-1] if wubu_s_config['num_levels'] > 0 else 0
        # Find the curvature of the last WuBu-S level for expmap
        self.wubu_s_final_curvature = 1.0 # Default
        if wubu_s_config['num_levels'] > 0 and self.wubu_s_final_hyp_dim > 0: # Check if hyp dim is > 0
             last_level_idx = wubu_s_config['num_levels'] - 1
             # Need to instantiate a temporary level to get initial C (or store it)
             try:
                temp_level = HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_s_final_hyp_dim, wubu_s_config, wubu_s_config['initial_curvatures'][last_level_idx])
                self.wubu_s_final_curvature = temp_level.get_current_curvature_scalar() # Assumes learnable C starts at initial
                del temp_level
                logger.info(f"Encoder: WuBu-S final level curvature estimated as {self.wubu_s_final_curvature:.3f}")
             except IndexError:
                logger.error(f"Encoder: Index error accessing initial_curvatures for WuBu-S level {last_level_idx}. Defaulting curvature to 1.0.")
                self.wubu_s_final_curvature = 1.0

        self.apply(init_weights_general)

    def forward(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frames_pixels (torch.Tensor): Input frames (B, N_frames, C, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - regional_hyperbolic_features (torch.Tensor): (B, N_frames, NumRegions, D_hyp_final) - Output of WuBu-S
                - gaad_bboxes (torch.Tensor): (B, N_frames, NumRegions, 4) - BBoxes used for extraction
        """
        B, N_frames, C, H, W = frames_pixels.shape
        device = frames_pixels.device
        dtype = frames_pixels.dtype

        frames_pixels_flat = frames_pixels.reshape(B * N_frames, C, H, W)

        gaad_bboxes_list = []
        for b_idx in range(B):
            current_frame_h, current_frame_w = frames_pixels.shape[3], frames_pixels.shape[4]
            frame_dims = (current_frame_w, current_frame_h) # W, H for GAAD functions
            max_w_scalar = float(frame_dims[0])
            max_h_scalar = float(frame_dims[1])

            if self.decomposition_type == "hybrid":
                num_subdivide = self.num_appearance_regions // 2
                num_spiral = self.num_appearance_regions - num_subdivide
                bboxes_for_item = []
                if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims,num_subdivide,device,dtype, self.gaad_min_size_px))
                if num_spiral > 0:
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype)
                     patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype);
                     patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs

                     # Calculate initial coordinates
                     val_x1 = spiral_centers[:,0]-patch_ws
                     val_y1 = spiral_centers[:,1]-patch_hs
                     val_x2 = spiral_centers[:,0]+patch_ws
                     val_y2 = spiral_centers[:,1]+patch_hs

                     # Clamp x1, y1
                     spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar-EPS)
                     spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar-EPS)

                     # Clamp x2, y2 ensuring x2 > x1 and y2 > y1
                     min_for_x2 = spiral_bboxes_current[:,0] + EPS
                     spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar)
                     spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)

                     min_for_y2 = spiral_bboxes_current[:,1] + EPS
                     spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar)
                     spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2)
                     bboxes_for_item.append(spiral_bboxes_current)
                frame_bboxes = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_appearance_regions, dtype=dtype, device=device)

            elif self.decomposition_type == "spiral":
                 spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_appearance_regions, device, dtype)
                 patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(self.num_appearance_regions, 4, device=device, dtype=dtype);
                 patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs

                 val_x1 = spiral_centers[:,0]-patch_ws
                 val_y1 = spiral_centers[:,1]-patch_hs
                 val_x2 = spiral_centers[:,0]+patch_ws
                 val_y2 = spiral_centers[:,1]+patch_hs

                 spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar-EPS)
                 spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar-EPS)

                 min_for_x2 = spiral_bboxes_current[:,0] + EPS
                 spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar)
                 spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)

                 min_for_y2 = spiral_bboxes_current[:,1] + EPS
                 spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar)
                 spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2)
                 frame_bboxes = spiral_bboxes_current
            else: # subdivide
                frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims,self.num_appearance_regions,device,dtype, self.gaad_min_size_px)

            if frame_bboxes.shape[0] < self.num_appearance_regions:
                 num_to_pad = self.num_appearance_regions - frame_bboxes.shape[0]
                 padding_box = frame_bboxes[-1:].clone() if frame_bboxes.shape[0] > 0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]], dtype=dtype, device=device)
                 padding = padding_box.repeat(num_to_pad, 1)
                 frame_bboxes = torch.cat([frame_bboxes, padding], dim=0)
            elif frame_bboxes.shape[0] > self.num_appearance_regions:
                 frame_bboxes = frame_bboxes[:self.num_appearance_regions]
            gaad_bboxes_list.append(frame_bboxes)

        gaad_bboxes_batch = torch.stack(gaad_bboxes_list)
        gaad_bboxes_full = gaad_bboxes_batch.unsqueeze(1).repeat(1, N_frames, 1, 1)
        gaad_bboxes_flat = gaad_bboxes_full.reshape(B * N_frames, self.num_appearance_regions, 4)

        extracted_patches = self.patch_extractor(frames_pixels_flat, gaad_bboxes_flat)

        B_flat_post_patch_extract, NumReg_post_patch_extract, C_patch, H_patch, W_patch = extracted_patches.shape
        patches_for_embed = extracted_patches.reshape(B_flat_post_patch_extract, NumReg_post_patch_extract, -1)
        initial_tangent_vectors = self.patch_embed(patches_for_embed)

        wubu_input = initial_tangent_vectors.reshape(B_flat_post_patch_extract * NumReg_post_patch_extract, -1)
        wubu_output_tangent = self.wubu_s(wubu_input)

        if self.wubu_s_config['num_levels'] > 0 and self.wubu_s_final_hyp_dim > 0:
            final_manifold = PoincareBall(self.wubu_s_final_curvature)
            regional_hyperbolic_features_flat = final_manifold.expmap0(wubu_output_tangent)
        else:
            regional_hyperbolic_features_flat = wubu_output_tangent
            logger.debug("WuBu-S has no hyperbolic levels or output; features remain in tangent space.")

        D_out = regional_hyperbolic_features_flat.shape[-1]
        regional_hyperbolic_features = regional_hyperbolic_features_flat.reshape(B, N_frames, NumReg_post_patch_extract, D_out)

        return regional_hyperbolic_features, gaad_bboxes_full


# --- Placeholder for Motion Encoder ---
class RegionalHyperbolicMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_motion_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_m_config = wubu_m_config
        self.enabled = args.use_wubu_motion_branch and wubu_m_config is not None and gaad_motion_config is not None

        if not self.enabled:
            logger.info("Motion Encoder branch DISABLED.")
            self.wubu_m_final_curvature = 1.0 # Default, not used
            return

        logger.info("Motion Encoder branch ENABLED.")
        self.image_size = (args.image_h, args.image_w)
        self.num_motion_regions = gaad_motion_config['num_regions']
        self.motion_decomposition_type = gaad_motion_config['decomposition_type'] # Could be 'content_aware'
        self.gaad_min_size_px = gaad_motion_config.get('min_size_px', 5)
        self.diff_map_channels = 1 # Assuming grayscale diff map for now

        # Feature Extractor (Optional, for RoIAlign on frame diffs or individual frames)
        # For simplicity, let's assume pixel patch extraction for motion features as well.
        # If RoIAlign is desired for motion, a similar setup as RegionalHyperbolicEncoder would be needed.
        self.use_roi_align_motion = getattr(args, 'motion_encoder_use_roi_align', False) # Add this arg if needed
        self.motion_patch_size = getattr(args, 'motion_encoder_pixel_patch_size', args.encoder_pixel_patch_size) # Reuse or new arg

        # Using the same patch extractor logic, but will apply it to frame_t and frame_t-1
        self.patch_extractor = RegionalPatchExtractor(
            patch_output_size=(self.motion_patch_size, self.motion_patch_size) if not self.use_roi_align_motion else None,
            # feature_extractor=None, # Or a dedicated one for motion if RoIAlign
            # roi_align_output_size=None,
            use_roi_align=self.use_roi_align_motion
        )
        logger.info(f"MotionEncoder: Using Pixel Patch Extraction (resize: {self.motion_patch_size})")


        # Patch Embedding for motion features (from concatenated frame_t, frame_t-1 patches)
        # If each frame patch is C*pH*pW, concatenated is 2*C*pH*pW
        patch_feature_dim_single_frame = self.video_config['num_channels'] * self.motion_patch_size * self.motion_patch_size
        concatenated_patch_feature_dim = 2 * patch_feature_dim_single_frame
        self.motion_patch_embed = PatchEmbed(concatenated_patch_feature_dim, args.encoder_initial_tangent_dim) # Reuse initial_tangent_dim or have a specific one for motion

        # WuBu-M
        if self.wubu_m_config is not None : # Ensure wubu_m_config exists before trying to use it
            self.wubu_m = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=args.encoder_initial_tangent_dim, # Input after embedding
                output_tangent_dim=video_config['wubu_m_output_dim'],
                config=wubu_m_config
            )

            self.wubu_m_final_hyp_dim = wubu_m_config['hyperbolic_dims'][-1] if wubu_m_config['num_levels'] > 0 else 0
            self.wubu_m_final_curvature = 1.0 # Default
            if wubu_m_config['num_levels'] > 0 and self.wubu_m_final_hyp_dim > 0:
                last_level_idx = wubu_m_config['num_levels'] - 1
                try:
                    temp_level_m = HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_m_final_hyp_dim, wubu_m_config, wubu_m_config['initial_curvatures'][last_level_idx])
                    self.wubu_m_final_curvature = temp_level_m.get_current_curvature_scalar()
                    del temp_level_m
                    logger.info(f"MotionEncoder: WuBu-M final level curvature estimated as {self.wubu_m_final_curvature:.3f}")
                except IndexError:
                    logger.error(f"MotionEncoder: Index error accessing initial_curvatures for WuBu-M level {last_level_idx}. Defaulting curvature to 1.0.")
                    self.wubu_m_final_curvature = 1.0
        else: # This case should ideally not be hit if self.enabled is true due to __init__ checks
            logger.error("MotionEncoder: wubu_m_config is None, cannot initialize WuBu-M model.")
            self.wubu_m = None
            self.wubu_m_final_hyp_dim = 0
            self.wubu_m_final_curvature = 1.0
            self.enabled = False # Mark as disabled if WuBu-M cannot be initialized

        self.apply(init_weights_general)


    def _get_motion_gaad_bboxes(self, diff_map_for_bboxes: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """ Generates GAAD bboxes, potentially content-aware from diff_map. """
        B_eff, _, H, W = diff_map_for_bboxes.shape
        all_batch_bboxes = []

        for i in range(B_eff):
            # current_diff_map_slice = diff_map_for_bboxes[i] # (1 or C_diff, H, W) # Unused
            frame_dims = (W, H) # W, H
            max_w_scalar = float(W)
            max_h_scalar = float(H)

            if self.motion_decomposition_type == "hybrid" or self.motion_decomposition_type == "content_aware":
                num_subdivide = self.num_motion_regions // 2
                num_spiral = self.num_motion_regions - num_subdivide
                bboxes_for_item = []
                if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims,num_subdivide,device,dtype, self.gaad_min_size_px))
                if num_spiral > 0:
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype)
                     patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype);
                     patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs
                     val_x1 = spiral_centers[:,0]-patch_ws; val_y1 = spiral_centers[:,1]-patch_hs
                     val_x2 = spiral_centers[:,0]+patch_ws; val_y2 = spiral_centers[:,1]+patch_hs
                     spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar-EPS)
                     spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar-EPS)
                     min_for_x2 = spiral_bboxes_current[:,0] + EPS; spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar); spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)
                     min_for_y2 = spiral_bboxes_current[:,1] + EPS; spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar); spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2)
                     bboxes_for_item.append(spiral_bboxes_current)
                frame_bboxes = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_motion_regions, dtype=dtype, device=device)
            elif self.motion_decomposition_type == "spiral":
                 spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_motion_regions, device, dtype)
                 patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(self.num_motion_regions, 4, device=device, dtype=dtype);
                 patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs
                 val_x1 = spiral_centers[:,0]-patch_ws; val_y1 = spiral_centers[:,1]-patch_hs
                 val_x2 = spiral_centers[:,0]+patch_ws; val_y2 = spiral_centers[:,1]+patch_hs
                 spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar-EPS)
                 spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar-EPS)
                 min_for_x2 = spiral_bboxes_current[:,0] + EPS; spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar); spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)
                 min_for_y2 = spiral_bboxes_current[:,1] + EPS; spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar); spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2)
                 frame_bboxes = spiral_bboxes_current
            else: # subdivide (or other default)
                frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims, self.num_motion_regions, device, dtype, self.gaad_min_size_px)


            if frame_bboxes.shape[0] < self.num_motion_regions:
                 num_to_pad = self.num_motion_regions - frame_bboxes.shape[0]
                 padding_box = frame_bboxes[-1:].clone() if frame_bboxes.shape[0] > 0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]], dtype=dtype, device=device)
                 padding = padding_box.repeat(num_to_pad, 1)
                 frame_bboxes = torch.cat([frame_bboxes, padding], dim=0)
            elif frame_bboxes.shape[0] > self.num_motion_regions:
                 frame_bboxes = frame_bboxes[:self.num_motion_regions]
            all_batch_bboxes.append(frame_bboxes)
        return torch.stack(all_batch_bboxes)


    def forward(self, frames_pixels: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.enabled or self.wubu_m is None:
            return None

        B, N_frames, C, H, W = frames_pixels.shape
        device = frames_pixels.device
        dtype = frames_pixels.dtype # Use input dtype, cast later if needed by modules
        
        if N_frames < 2:
            if N_frames == 1:
                logger.debug(f"MotionEncoder: Received single frame (N_frames={N_frames}), cannot compute motion. This is expected for target frame encoding.")
            else:
                logger.warning(f"MotionEncoder: Not enough frames (N_frames={N_frames}) to compute motion (need at least 2). Skipping.")
            return None

        num_pairs = N_frames - 1
        all_motion_features_hyperbolic_list = []
        all_motion_bboxes_list = []

        # Ensure modules have their compute dtype from a parameter
        module_compute_dtype = next(self.parameters()).dtype

        for i in range(num_pairs):
            frame_t = frames_pixels[:, i+1, ...].to(module_compute_dtype)
            frame_t_minus_1 = frames_pixels[:, i, ...].to(module_compute_dtype)

            diff_map_raw = torch.abs(frame_t - frame_t_minus_1)
            if self.diff_map_channels == 1 and diff_map_raw.shape[1] > 1:
                diff_map_for_bboxes = torch.mean(diff_map_raw, dim=1, keepdim=True)
            else:
                diff_map_for_bboxes = diff_map_raw

            motion_gaad_bboxes_batch = self._get_motion_gaad_bboxes(diff_map_for_bboxes, device, module_compute_dtype)

            patches_t = self.patch_extractor(frame_t, motion_gaad_bboxes_batch)
            patches_t_minus_1 = self.patch_extractor(frame_t_minus_1, motion_gaad_bboxes_batch)

            patches_t_flat = patches_t.reshape(B * self.num_motion_regions, C, self.motion_patch_size, self.motion_patch_size)
            patches_t_minus_1_flat = patches_t_minus_1.reshape(B * self.num_motion_regions, C, self.motion_patch_size, self.motion_patch_size)
            
            patches_t_vec = patches_t_flat.reshape(B * self.num_motion_regions, -1)
            patches_t_minus_1_vec = patches_t_minus_1_flat.reshape(B * self.num_motion_regions, -1)
            concatenated_motion_patch_vecs = torch.cat([patches_t_vec, patches_t_minus_1_vec], dim=1)

            initial_motion_tangent_vectors_flat = self.motion_patch_embed.proj(concatenated_motion_patch_vecs)
            
            wubu_m_output_tangent_flat = self.wubu_m(initial_motion_tangent_vectors_flat)

            if self.wubu_m_config is not None and self.wubu_m_config['num_levels'] > 0 and self.wubu_m_final_hyp_dim > 0:
                final_manifold_m = PoincareBall(self.wubu_m_final_curvature)
                motion_features_hyperbolic_flat = final_manifold_m.expmap0(wubu_m_output_tangent_flat)
            else:
                motion_features_hyperbolic_flat = wubu_m_output_tangent_flat
            
            motion_features_hyperbolic_pair = motion_features_hyperbolic_flat.reshape(B, self.num_motion_regions, -1)
            
            all_motion_features_hyperbolic_list.append(motion_features_hyperbolic_pair)
            all_motion_bboxes_list.append(motion_gaad_bboxes_batch)

        if not all_motion_features_hyperbolic_list:
            return None

        final_motion_features = torch.stack(all_motion_features_hyperbolic_list, dim=1).to(dtype) # Cast back to original input dtype
        final_motion_bboxes = torch.stack(all_motion_bboxes_list, dim=1).to(dtype) # Cast back to original input dtype

        return final_motion_features, final_motion_bboxes



class RegionalPixelSynthesisDecoder(nn.Module):
    """ Decodes a set of regional tangent features back into pixels using GAAD coords. """
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_config: Dict):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.image_size = (args.image_h, args.image_w)
        self.num_regions = gaad_config['num_regions']
        self.decoder_type = args.decoder_type
        self.num_channels = video_config['num_channels']

        # Need the dimension of the tangent features being input
        self.input_tangent_dim = video_config['wubu_s_output_dim'] # Assumes input is tangent vector after WuBu-S aggregation

        if self.decoder_type == "patch_gen":
            self.patch_size = args.decoder_patch_gen_size
            patch_pixels = self.num_channels * self.patch_size * self.patch_size
            # Simple MLP or TransposeCNN to generate patch pixels from tangent vector
            self.patch_generator = nn.Sequential(
                nn.Linear(self.input_tangent_dim, self.input_tangent_dim * 2),
                nn.GELU(),
                nn.Linear(self.input_tangent_dim * 2, patch_pixels),
                nn.Tanh() # Output pixel values in [-1, 1]
            )
            self.patch_resize_mode = args.decoder_patch_resize_mode
            logger.info(f"Decoder: Using Patch Generator (Output Size: {self.patch_size}x{self.patch_size})")

        elif self.decoder_type == "transformer":
            # TODO: Implement Transformer-based spatial decoder
            logger.warning("Decoder: Transformer type not implemented yet.")
            raise NotImplementedError("Transformer decoder not implemented")
        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")

        self.apply(init_weights_general)

    def forward(self, regional_tangent_features: torch.Tensor, gaad_bboxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regional_tangent_features (torch.Tensor): (B, N_pred_frames, NumRegions, D_tangent)
            gaad_bboxes (torch.Tensor): (B, N_pred_frames, NumRegions, 4) - Pixel coords for placement

        Returns:
            torch.Tensor: Reconstructed pixel frames (B, N_pred_frames, C, H, W)
        """
        B, N_pred, NumReg, D_tan = regional_tangent_features.shape
        C, H, W = self.num_channels, self.image_size[0], self.image_size[1]
        device = regional_tangent_features.device
        dtype = regional_tangent_features.dtype

        # Flatten inputs
        regional_tangent_flat = regional_tangent_features.view(B * N_pred * NumReg, D_tan)
        gaad_bboxes_flat = gaad_bboxes.view(B * N_pred, NumReg, 4) # Keep N_pred dim separate for canvas creation

        output_frames = torch.zeros(B, N_pred, C, H, W, device=device, dtype=dtype)

        if self.decoder_type == "patch_gen":
            # 1. Generate patches
            generated_patch_pixels_flat = self.patch_generator(regional_tangent_flat) # (B*N_pred*NumReg, C*pH*pW)
            generated_patches = generated_patch_pixels_flat.view(B * N_pred, NumReg, C, self.patch_size, self.patch_size)

            # 2. Place/Blend patches onto canvas
            canvas = torch.zeros(B * N_pred, C, H, W, device=device, dtype=dtype)
            counts = torch.zeros(B * N_pred, 1, H, W, device=device, dtype=dtype) # For averaging overlaps

            for i in range(B * N_pred): # Iterate through batch * frames
                for r in range(NumReg):
                    patch = generated_patches[i, r] # (C, pH, pW)
                    x1, y1, x2, y2 = gaad_bboxes_flat[i, r].tolist()

                    # Calculate target placement coordinates (integer for slicing)
                    target_h = int(round(y2 - y1))
                    target_w = int(round(x2 - x1))
                    place_y1 = int(round(y1))
                    place_x1 = int(round(x1))

                    if target_h <= 0 or target_w <= 0: continue # Skip zero-size regions

                    # Resize the generated patch to fit the GAAD region size
                    resized_patch = F.interpolate(patch.unsqueeze(0),
                                                  size=(target_h, target_w),
                                                  mode=self.patch_resize_mode,
                                                  align_corners=False if self.patch_resize_mode != 'nearest' else None).squeeze(0)

                    # Ensure placement is within bounds
                    place_y2 = min(H, place_y1 + target_h)
                    place_x2 = min(W, place_x1 + target_w)
                    place_y1 = max(0, place_y1)
                    place_x1 = max(0, place_x1)

                    # Adjust patch slicing if placement was clipped
                    slice_h = place_y2 - place_y1
                    slice_w = place_x2 - place_x1
                    if slice_h <= 0 or slice_w <= 0: continue

                    # Add patch to canvas and increment count
                    canvas[i, :, place_y1:place_y2, place_x1:place_x2] += resized_patch[:, :slice_h, :slice_w]
                    counts[i, :, place_y1:place_y2, place_x1:place_x2] += 1

            # Average overlapping regions
            output_canvas_flat = torch.where(counts > 0, canvas / counts.clamp(min=1.0), canvas) # Avoid div by zero

            # Reshape back to (B, N_pred, C, H, W)
            output_frames = output_canvas_flat.view(B, N_pred, C, H, W)

        else: # Transformer or other types
             raise NotImplementedError(f"Decoder type {self.decoder_type} forward pass not implemented.")

        return output_frames

# --- Noise Predictor (Transformer based) ---
class TransformerNoisePredictor(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, wubu_s_config: Dict, wubu_t_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.transformer_config = args.transformer_noise_predictor_config
        self.num_regions = args.gaad_num_regions # Appearance regions define the sequence length
        self.d_model = self.transformer_config['d_model']

        # Input dimension per region feature (tangent space) + time embedding
        self.input_feat_dim = video_config['wubu_s_output_dim'] # Denoising target features
        self.time_embed_dim = args.diffusion_time_embedding_dim

        # Temporal context dimension (output of WuBu-T)
        self.wubu_t_output_dim = video_config['wubu_t_output_dim'] if wubu_t_config and wubu_t_config['num_levels'] > 0 else 0

        # Project input features and time embedding to d_model
        self.input_proj = nn.Linear(self.input_feat_dim, self.d_model)
        self.time_proj = nn.Linear(self.time_embed_dim, self.d_model)

        # Positional Encoding for regions (learned or sinusoidal)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_regions, self.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Temporal Context Projection (if WuBu-T is used)
        self.context_proj = nn.Linear(self.wubu_t_output_dim, self.d_model) if self.wubu_t_output_dim > 0 else None

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.transformer_config['num_heads'],
            dim_feedforward=int(self.d_model * self.transformer_config['d_ff_ratio']),
            dropout=self.transformer_config['dropout'],
            activation=self.transformer_config['activation'],
            batch_first=True # Important: Input is (B, SeqLen=NumRegions, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_config['num_layers'])

        # Output head to predict noise in the original tangent dimension
        self.output_proj = nn.Linear(self.d_model, self.input_feat_dim)

        self.apply(init_weights_general)
        logger.info(f"TransformerNoisePredictor: Layers={self.transformer_config['num_layers']}, Heads={self.transformer_config['num_heads']}, Dim={self.d_model}")

    def forward(self, noisy_regional_tangent_features: torch.Tensor,
                time_embedding: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            noisy_regional_tangent_features (torch.Tensor): (B, N_pred, NumRegions, D_tangent_in)
            time_embedding (torch.Tensor): (B, N_pred, D_time)
            temporal_context (Optional[torch.Tensor]): Aggregated temporal context (B, N_pred, D_wubu_t)

        Returns:
            torch.Tensor: Predicted noise in tangent space (B, N_pred, NumRegions, D_tangent_in)
        """
        B, N_pred, NumReg, D_in = noisy_regional_tangent_features.shape
        # device = noisy_regional_tangent_features.device # Not used directly
        # dtype = noisy_regional_tangent_features.dtype # Not used directly

        # Flatten N_pred dimension into Batch for processing
        # noisy_features_flat = noisy_regional_tangent_features.view(B * N_pred, NumReg, D_in) # OLD
        noisy_features_flat = noisy_regional_tangent_features.reshape(B * N_pred, NumReg, D_in) # NEW

        # time_embedding_flat = time_embedding.view(B * N_pred, -1) # OLD
        time_embedding_flat = time_embedding.reshape(B * N_pred, -1) # NEW

        # Project inputs
        x = self.input_proj(noisy_features_flat) # (B*N_pred, NumReg, d_model)
        t_emb = self.time_proj(time_embedding_flat) # (B*N_pred, d_model)

        # Add time embedding to each region token
        x = x + t_emb.unsqueeze(1) # Broadcast time across regions

        # Add positional embedding for regions
        x = x + self.pos_embed # Broadcast batch dim

        # Add temporal context (if available)
        if temporal_context is not None and self.context_proj is not None:
            # context_flat = temporal_context.view(B * N_pred, -1) # OLD
            context_flat = temporal_context.reshape(B * N_pred, -1) # NEW
            ctx_emb = self.context_proj(context_flat) # (B*N_pred, d_model)
            x = x + ctx_emb.unsqueeze(1) # Broadcast context across regions

        # Apply Transformer
        transformer_output = self.transformer_encoder(x) # (B*N_pred, NumReg, d_model)

        # Project to output noise dimension
        predicted_noise_flat = self.output_proj(transformer_output) # (B*N_pred, NumReg, D_tangent_in)

        # Reshape back to original N_pred dimension
        # predicted_noise = predicted_noise_flat.view(B, N_pred, NumReg, D_in) # OLD
        predicted_noise = predicted_noise_flat.reshape(B, N_pred, NumReg, D_in) # NEW

        return predicted_noise



# =====================================================================
# Diffusion Model Specific Components (Adapted for Regional Latents)
# =====================================================================

# Removed InitialFrameAutoencoderCNN (v0.05.2)

# SinusoidalPhiEmbedding remains the same
class SinusoidalPhiEmbedding(nn.Module): # Unchanged
    def __init__(self, dim: int, base_freq_phi_scaled: float = 10000.0, use_phi_paper_scaling_arg: bool = False, phi_constant: float = PHI):
        super().__init__(); self.dim = dim; self.base_freq_for_paper_scaling = base_freq_phi_scaled; self.base_period_for_ddpm_scaling = base_freq_phi_scaled; self.use_phi_paper_scaling = use_phi_paper_scaling_arg; self.phi_constant = phi_constant; half_dim = dim // 2; denominators = torch.empty(0, dtype=torch.float)
        if half_dim > 0:
            if self.use_phi_paper_scaling: exponent_val = torch.arange(half_dim).float() / float(half_dim); phi_scaling_factor = self.phi_constant ** exponent_val; denominators = self.base_freq_for_paper_scaling / (phi_scaling_factor + EPS); logger.info(f"SinusoidalPhiEmbedding: PHI paper scaling. Dim {dim}, BaseFreq ω: {self.base_freq_for_paper_scaling:.1f}, PHI: {self.phi_constant:.3f}")
            else:
                if half_dim == 1: denominators = torch.tensor([1.0 / self.base_period_for_ddpm_scaling])
                else: denominators = torch.exp(torch.arange(half_dim).float() * -(math.log(self.base_period_for_ddpm_scaling) / (half_dim - 1.0)))
                logger.info(f"SinusoidalPhiEmbedding: DDPM-style scaling. Dim {dim}, BasePeriod: {self.base_period_for_ddpm_scaling:.1f}")
        else: logger.info(f"SinusoidalPhiEmbedding: Dim {dim} too small, empty/zero embedding.")
        self.register_buffer("denominators_actual", denominators)
    def forward(self, t: torch.Tensor, phi_time_scale: float = 1.0) -> torch.Tensor:
        if self.dim == 0: return torch.empty(t.shape[0], 0, device=t.device, dtype=torch.float)
        if self.denominators_actual.numel() == 0 and self.dim > 0: return torch.zeros(t.shape[0], self.dim, device=t.device, dtype=torch.float) if self.dim !=1 else (t.float()*phi_time_scale).unsqueeze(-1)
        t_scaled = t.float() * phi_time_scale; args = t_scaled[:, None] * self.denominators_actual[None, :]; embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 != 0 and self.dim > 0: embedding = F.pad(embedding, (0, 1), value=0.0)
        return embedding


# --- Main Model: GAAD-WuBu Regional Diffusion Network ---
class GAADWuBuRegionalDiffNet(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_appearance_config: Dict, gaad_motion_config: Optional[Dict], wubu_s_config: Dict, wubu_t_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_appearance_config = gaad_appearance_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_s_config = wubu_s_config
        self.wubu_t_config = wubu_t_config
        self.wubu_m_config = wubu_m_config

        # --- Encoder ---
        self.encoder = RegionalHyperbolicEncoder(args, video_config, gaad_appearance_config, wubu_s_config)

        # --- Motion Encoder (Optional) ---
        self.motion_encoder: Optional[RegionalHyperbolicMotionEncoder] = None
        if args.use_wubu_motion_branch and self.wubu_m_config is not None and self.gaad_motion_config is not None: # Ensure configs exist
             self.motion_encoder = RegionalHyperbolicMotionEncoder(args, video_config, gaad_motion_config, wubu_m_config)
        else: # Explicitly log if motion branch is requested but configs are missing
            if args.use_wubu_motion_branch:
                logger.warning("Motion branch requested (--use_wubu_motion_branch) but wubu_m_config or gaad_motion_config is None. Disabling motion branch.")
            self.args.use_wubu_motion_branch = False # Ensure it's marked as disabled internally

        # --- Temporal Aggregation (WuBu-T) ---
        self.wubu_t: Optional[FullyHyperbolicWuBuNestingModel] = None
        self.wubu_t_input_dim = 0
        if self.wubu_t_config and self.wubu_t_config['num_levels'] > 0:
            self.wubu_t_input_dim = video_config['wubu_s_output_dim']
            # Only add motion dim if motion encoder is truly enabled and configured
            if self.args.use_wubu_motion_branch and self.motion_encoder and self.motion_encoder.enabled and video_config.get('wubu_m_output_dim', 0) > 0:
                 self.wubu_t_input_dim += video_config['wubu_m_output_dim']
            else: # If motion is not contributing, log it.
                 if self.args.use_wubu_motion_branch: # If it *was* requested
                    logger.info("WuBu-T: Motion branch not contributing features (either disabled or output_dim is 0). WuBu-T input dim will be S-only.")


            if self.wubu_t_input_dim > 0:
                 self.wubu_t = FullyHyperbolicWuBuNestingModel(
                     input_tangent_dim=self.wubu_t_input_dim,
                     output_tangent_dim=video_config['wubu_t_output_dim'],
                     config=wubu_t_config
                 )
                 logger.info(f"WuBu-T Enabled: Input Dim (Aggregated S+M): {self.wubu_t_input_dim}, Output Dim: {video_config['wubu_t_output_dim']}")
            else:
                 logger.warning("WuBu-T configured but effective input dim is 0. Disabling WuBu-T.")
                 if self.wubu_t_config: self.wubu_t_config['num_levels'] = 0


        # --- Time Embedding ---
        self.time_sin_embedding = SinusoidalPhiEmbedding(args.diffusion_time_embedding_dim, base_freq_phi_scaled=gaad_appearance_config.get("phi_time_base_freq", 10000.0), use_phi_paper_scaling_arg=args.use_phi_frequency_scaling_for_time_emb, phi_constant=PHI)
        self.time_fc_mlp = nn.Sequential(nn.Linear(args.diffusion_time_embedding_dim, args.diffusion_time_embedding_dim * 2), nn.GELU(), nn.Linear(args.diffusion_time_embedding_dim * 2, args.diffusion_time_embedding_dim))

        # --- Noise Predictor ---
        self.noise_predictor = TransformerNoisePredictor(args, video_config, wubu_s_config, self.wubu_t_config, self.wubu_m_config)
        args.transformer_noise_predictor_config.setdefault('d_model', DEFAULT_CONFIG_TRANSFORMER_NOISE_PREDICTOR['d_model'])


        # --- Decoder ---
        self.decoder = RegionalPixelSynthesisDecoder(args, video_config, gaad_appearance_config, wubu_s_config)

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"GAADWuBuRegionalDiffNet (v0.10.0) Initialized: {param_count:,} params.")

    def encode_frames(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """ Encodes frames to regional features (appearance and motion). """
        regional_app_features, gaad_bboxes_app = self.encoder(frames_pixels)
        regional_motion_features = None
        motion_output_tuple = None

        if self.motion_encoder is not None and self.motion_encoder.enabled:
            motion_output_tuple = self.motion_encoder(frames_pixels)
            if motion_output_tuple is not None:
                regional_motion_features, _ = motion_output_tuple
                # Ensure motion features have the correct dtype if different from appearance
                if regional_motion_features.dtype != regional_app_features.dtype:
                    regional_motion_features = regional_motion_features.to(regional_app_features.dtype)


        return regional_app_features, gaad_bboxes_app, regional_motion_features

    def decode_features(self, regional_tangent_features: torch.Tensor, gaad_bboxes: torch.Tensor) -> torch.Tensor:
        """ Decodes regional tangent features to pixels. """
        return self.decoder(regional_tangent_features, gaad_bboxes)

    def forward(self,
                noisy_regional_tangent_features: torch.Tensor,
                time_t_integers: torch.Tensor,
                conditioning_regional_app_features: Optional[torch.Tensor] = None, # (B, N_cond, NumReg, D_hyp_s)
                conditioning_regional_motion_features: Optional[torch.Tensor] = None, # (B, N_cond-1, NumReg_m, D_hyp_m)
                cfg_unconditional_flag: bool = False
                ) -> torch.Tensor:
        """
        Predicts noise in the TANGENT space of the regional features.
        """
        B, N_pred, NumReg, D_tan_s = noisy_regional_tangent_features.shape
        device = noisy_regional_tangent_features.device
        dtype = noisy_regional_tangent_features.dtype # Use the dtype of the input noisy features

        time_sin_emb = self.time_sin_embedding(time_t_integers, phi_time_scale=self.gaad_appearance_config.get("phi_time_diffusion_scale", 1.0))
        time_emb = self.time_fc_mlp(time_sin_emb).to(dtype) # Ensure time_emb has correct dtype
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, N_pred, -1)

        temporal_context: Optional[torch.Tensor] = None
        if self.wubu_t is not None and conditioning_regional_app_features is not None and not cfg_unconditional_flag:
            N_cond_app = conditioning_regional_app_features.shape[1]

            final_manifold_s = PoincareBall(self.encoder.wubu_s_final_curvature)
            cond_app_tangent = final_manifold_s.logmap0(conditioning_regional_app_features.to(dtype)) # Ensure dtype
            aggregated_app_context = torch.max(cond_app_tangent, dim=2)[0]

            wubu_t_sequence_input_list = [aggregated_app_context]

            if conditioning_regional_motion_features is not None and \
               self.motion_encoder and self.motion_encoder.enabled and \
               hasattr(self.motion_encoder, 'wubu_m_final_curvature'):

                N_cond_mot = conditioning_regional_motion_features.shape[1]
                # Ensure motion encoder has wubu_m_final_curvature and it's a valid float
                motion_final_c = getattr(self.motion_encoder, 'wubu_m_final_curvature', 1.0)
                if not isinstance(motion_final_c, float): motion_final_c = 1.0 # Fallback

                final_manifold_m = PoincareBall(motion_final_c)
                cond_mot_tangent = final_manifold_m.logmap0(conditioning_regional_motion_features.to(dtype)) # Ensure dtype
                aggregated_mot_context = torch.max(cond_mot_tangent, dim=2)[0]

                if N_cond_mot < N_cond_app:
                    padding_needed = N_cond_app - N_cond_mot
                    padding = torch.zeros(B, padding_needed, aggregated_mot_context.shape[-1], device=device, dtype=dtype)
                    aggregated_mot_context_padded = torch.cat([padding, aggregated_mot_context], dim=1)
                elif N_cond_mot > N_cond_app:
                    logger.warning(f"Motion context seq len {N_cond_mot} > App context seq len {N_cond_app}. Truncating motion.")
                    aggregated_mot_context_padded = aggregated_mot_context[:, :N_cond_app, :]
                else:
                    aggregated_mot_context_padded = aggregated_mot_context
                
                wubu_t_sequence_input_list.append(aggregated_mot_context_padded)
            
            elif conditioning_regional_motion_features is not None and self.args.use_wubu_motion_branch: # Log if motion was expected
                 logger.warning("Motion features provided/expected, but motion encoder or its 'wubu_m_final_curvature' is invalid/missing. Skipping motion for WuBu-T.")


            wubu_t_sequence_input = torch.cat(wubu_t_sequence_input_list, dim=-1)

            if wubu_t_sequence_input.shape[1] > 0 and self.wubu_t_input_dim > 0 and wubu_t_sequence_input.shape[2] == self.wubu_t_input_dim:
                 wubu_t_output = self.wubu_t(wubu_t_sequence_input)
                 temporal_context_per_batch = wubu_t_output[:, -1, :]
                 temporal_context = temporal_context_per_batch.unsqueeze(1).expand(-1, N_pred, -1)
            else:
                 if self.wubu_t_input_dim > 0 : # Only warn if WuBu-T was supposed to run
                    logger.warning(f"WuBu-T input shape mismatch or empty sequence. Got: {wubu_t_sequence_input.shape}, Expected feature dim: {self.wubu_t_input_dim}. Skipping temporal context.")
                 temporal_context = None
        else:
            if self.wubu_t is not None and not cfg_unconditional_flag and self.wubu_t_input_dim > 0:
                 logger.debug("WuBu-T active but no conditioning app features or CFG is unconditional. Skipping temporal context.")
            temporal_context = None

        # Ensure temporal_context has the correct dtype for the noise predictor if it's not None
        if temporal_context is not None:
            temporal_context = temporal_context.to(dtype)

        predicted_noise = self.noise_predictor(
            noisy_regional_tangent_features, # Already correct dtype
            time_emb_expanded,               # Already correct dtype
            temporal_context                 # Now correct dtype or None
        )
        return predicted_noise


# Diffusion schedules (linear, cosine) - Unchanged from v0.05.2
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02): return torch.linspace(beta_start, beta_end, timesteps)
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1; x = torch.linspace(0, timesteps, steps); alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]; betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]); return torch.clip(betas, 0.0001, 0.9999)

# Modified q_sample to operate on regional hyperbolic features via tangent space
def q_sample_regional(x0_regional_hyperbolic: torch.Tensor,
                      t: torch.Tensor, # Shape (B,)
                      manifold: Manifold, # Manifold object for logmap/expmap
                      sqrt_alphas_cumprod: torch.Tensor, # Shape (T,)
                      sqrt_one_minus_alphas_cumprod: torch.Tensor, # Shape (T,)
                      noise: Optional[torch.Tensor] = None # Provide tangent space noise
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adds noise to regional hyperbolic features in the tangent space.

    Args:
        x0_regional_hyperbolic: Clean features (B, N_pred, NumReg, D_hyp)
        t: Timesteps for batch items (B,)
        manifold: Manifold object (e.g., PoincareBall instance)
        sqrt_alphas_cumprod: Diffusion schedule component
        sqrt_one_minus_alphas_cumprod: Diffusion schedule component
        noise: Optional tangent space noise (B, N_pred, NumReg, D_hyp)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - xt_regional_hyperbolic: Noisy hyperbolic features (B, N_pred, NumReg, D_hyp)
            - noise_tangent: The tangent space noise used (B, N_pred, NumReg, D_hyp)
    """
    B, N_pred, NumReg, D_hyp = x0_regional_hyperbolic.shape
    device = x0_regional_hyperbolic.device
    dtype = x0_regional_hyperbolic.dtype

    # Flatten features for map operations
    x0_flat = x0_regional_hyperbolic.view(B * N_pred * NumReg, D_hyp)

    # Map to tangent space
    try:
        tangent_x0_flat = manifold.logmap0(x0_flat)
        if not torch.isfinite(tangent_x0_flat).all():
             logger.warning(f"q_sample: Non-finite values after logmap0. Clamping/NaNing.")
             tangent_x0_flat = torch.nan_to_num(tangent_x0_flat, nan=0.0, posinf=TAN_VEC_CLAMP_VAL, neginf=-TAN_VEC_CLAMP_VAL)
    except Exception as e_logmap:
        logger.error(f"q_sample: Error during logmap0: {e_logmap}. Returning zeros.", exc_info=True)
        tangent_x0_flat = torch.zeros_like(x0_flat)


    if noise is None:
        noise_tangent = torch.randn_like(tangent_x0_flat) # Noise in tangent space
    else:
        # Assume provided noise is already in tangent space and correctly shaped
        noise_tangent = noise.view(B * N_pred * NumReg, D_hyp).to(device, dtype)

    # Gather schedule values for the batch
    # Reshape t to match the first dim of flattened features (B * N_pred * NumReg)
    t_expanded = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, N_pred, NumReg, 1).reshape(B * N_pred * NumReg)

    sqrt_alpha_t = sqrt_alphas_cumprod.gather(0,t_expanded).unsqueeze(-1).to(device, dtype) # (B', 1)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod.gather(0,t_expanded).unsqueeze(-1).to(device, dtype) # (B', 1)

    # Calculate noisy tangent vector: sqrt(alpha_t)*tangent_x0 + sqrt(1-alpha_t)*noise
    noisy_tangent_flat = sqrt_alpha_t * tangent_x0_flat + sqrt_one_minus_alpha_t * noise_tangent

    # Map back to hyperbolic space
    try:
        xt_regional_hyperbolic_flat = manifold.expmap0(noisy_tangent_flat)
        if not torch.isfinite(xt_regional_hyperbolic_flat).all():
             logger.warning(f"q_sample: Non-finite values after expmap0. Projecting/NaNing.")
             xt_regional_hyperbolic_flat = manifold.proju(torch.nan_to_num(xt_regional_hyperbolic_flat, nan=0.0))
    except Exception as e_expmap:
        logger.error(f"q_sample: Error during expmap0: {e_expmap}. Returning zeros.", exc_info=True)
        xt_regional_hyperbolic_flat = torch.zeros_like(noisy_tangent_flat) # Should return projected origin


    # Reshape back
    xt_regional_hyperbolic = xt_regional_hyperbolic_flat.view(B, N_pred, NumReg, D_hyp)
    noise_tangent_reshaped = noise_tangent.view(B, N_pred, NumReg, D_hyp)

    return xt_regional_hyperbolic, noise_tangent_reshaped


# VideoFrameDataset (Unchanged from v0.05.2)
class VideoFrameDataset(Dataset):
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int], frame_skip: int = 1, data_fraction: float = 1.0):
        super().__init__()
        self.video_path = video_path
        self.num_frames_total = num_frames_total
        self.image_size = image_size
        self.frame_skip = frame_skip

        if not os.path.isfile(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        logger.info(f"Attempting to load entire video into RAM: {self.video_path}")
        if not VIDEO_IO_AVAILABLE:
            logger.error("torchvision.io.read_video is not available.")
            raise RuntimeError("torchvision.io.read_video is not available.")

        try:
            # pts_unit="sec" is generally recommended
            video_data = video_io.read_video(self.video_path, output_format="TCHW", pts_unit="sec")
            self.video_frames_in_ram = video_data[0].contiguous()
            self.source_video_fps = video_data[2].get('video_fps', 30.0) # Use .get for robustness
            ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3)
            logger.info(f"Loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. Est RAM: {ram_usage_gb:.2f} GB.")
        except Exception as e:
            logger.error(f"Failed to load video '{self.video_path}' into RAM: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load video '{self.video_path}' into RAM.") from e

        # Separate transformations for clarity and pickling compatibility
        self.resize_transform = T.Resize(self.image_size, antialias=True)
        # Conversion to float and normalization will be done in __getitem__
        self.normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.num_disk_frames = self.video_frames_in_ram.shape[0]
        self.samples = []
        required_span_len = (self.num_frames_total - 1) * self.frame_skip + 1

        if self.num_disk_frames >= required_span_len:
            # Generate list of valid starting indices
            for i in range(self.num_disk_frames - required_span_len + 1):
                self.samples.append(i)
        else:
            logger.warning(f"Not enough frames ({self.num_disk_frames}) in video '{self.video_path}' for span {required_span_len} (total {self.num_frames_total}, skip {self.frame_skip}).")

        if data_fraction < 1.0 and len(self.samples) > 1:
            num_to_keep = max(1, int(len(self.samples) * data_fraction))
            self.samples = random.sample(self.samples, num_to_keep)
            logger.info(f"Using {data_fraction*100:.2f}% of samples: {len(self.samples)} samples.")

        if not self.samples:
            logger.error(f"VideoFrameDataset: No valid samples. Frames: {self.num_disk_frames}, Total required: {self.num_frames_total}, Skip: {self.frame_skip}.")
            # Optionally raise an error here if an empty dataset is problematic
            # raise ValueError("VideoFrameDataset resulted in zero valid samples.")
        else:
            logger.info(f"VideoFrameDataset initialized (RAM). Frames: {self.num_disk_frames}. Samples: {len(self.samples)}. Sample len: {self.num_frames_total} (skip {self.frame_skip}).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_frame_idx_in_ram = self.samples[idx]
        frames_for_sample = []

        for i in range(self.num_frames_total):
            actual_frame_idx_in_ram = start_frame_idx_in_ram + i * self.frame_skip
            if actual_frame_idx_in_ram < self.num_disk_frames:
                try:
                    # 1. Get frame (uint8, CHW)
                    frame_tensor_chw_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram]

                    # 2. Resize (on uint8 or float - Resize handles both)
                    resized_frame_tensor = self.resize_transform(frame_tensor_chw_uint8)

                    # 3. Convert to float and scale to [0, 1]
                    frame_float_01 = resized_frame_tensor.float() / 255.0

                    # 4. Normalize to [-1, 1]
                    transformed_frame = self.normalize_transform(frame_float_01)

                    frames_for_sample.append(transformed_frame)
                except Exception as e:
                    logger.error(f"Error transforming frame {actual_frame_idx_in_ram} for sample {idx}: {e}", exc_info=True)
                    # Decide how to handle errors: re-raise, return None, return dummy data
                    raise e # Re-raising is often safest
            else:
                logger.error(f"Frame index {actual_frame_idx_in_ram} out of bounds (total: {self.num_disk_frames}). Sample: {idx}")
                raise IndexError("Frame index out of bounds.")

        if len(frames_for_sample) != self.num_frames_total:
            logger.error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total} for sample {idx}")
            raise ValueError("Incorrect number of frames loaded for sample.")

        return torch.stack(frames_for_sample)



# --- Diffusion Trainer (Adapted for Regional Latents) ---
class DiffusionTrainer:
    def __init__(self, model: GAADWuBuRegionalDiffNet, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], args: argparse.Namespace, rank: int, world_size: int, ddp_active: bool):
        self.model = model; self.optimizer = optimizer; self.device = device; self.train_loader = train_loader; self.val_loader = val_loader; self.args = args; self.rank = rank; self.world_size = world_size; self.ddp_active = ddp_active; self.am_main_process = (rank == 0)
        self.video_config = model.video_config # Get configs from model instance
        self.gaad_appearance_config = model.gaad_appearance_config
        self.gaad_motion_config = model.gaad_motion_config
        self.wubu_s_config = model.wubu_s_config
        self.wubu_t_config = model.wubu_t_config
        self.wubu_m_config = model.wubu_m_config

        # Diffusion Schedules
        self.timesteps = args.timesteps;
        self.betas = (linear_beta_schedule(args.timesteps,args.beta_start,args.beta_end) if args.beta_schedule=='linear' else cosine_beta_schedule(args.timesteps,args.cosine_s)).to(device);
        self.alphas = 1. - self.betas; self.alphas_cumprod = torch.cumprod(self.alphas,axis=0);
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod);
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod);
        self.sqrt_recip_alphas = torch.sqrt(1.0/(self.alphas+EPS));
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0);
        self.posterior_variance = torch.clamp(self.betas*(1.-self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS),min=EPS*10);
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance);
        self.posterior_mean_coef1 = self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS);
        self.posterior_mean_coef2 = (1.-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.-self.alphas_cumprod+EPS)

        self.scaler = amp.GradScaler(enabled=args.use_amp and device.type=='cuda');
        self.global_step=0; self.current_epoch=0; self.best_val_loss=float('inf');
        self.last_val_metrics:Dict[str,Any]={};

        if self.am_main_process: os.makedirs(args.checkpoint_dir,exist_ok=True)

        # Manifold for the final hyperbolic space of the encoder (WuBu-S output)
        m_ref = self.model.module if ddp_active and isinstance(self.model, DDP) else self.model
        self.encoder_final_manifold = PoincareBall(m_ref.encoder.wubu_s_final_curvature)

        # Loss functions (for validation mainly)
        self.lpips_loss_fn = None
        if self.am_main_process and self.args.use_lpips_for_verification:
            if LPIPS_AVAILABLE and lpips is not None:
                try: self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device); logger.info("LPIPS metric enabled.")
                except Exception as e: logger.warning(f"LPIPS init failed: {e}. Skip LPIPS."); self.lpips_loss_fn = None
            else: logger.warning("LPIPS lib not found. Skip LPIPS.")
        self.ssim_metric = None
        if self.am_main_process and TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
            try: self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device); logger.info("SSIM metric enabled.")
            except Exception as e: logger.warning(f"SSIM init failed: {e}. Skip SSIM."); self.ssim_metric = None
        elif self.am_main_process and not TORCHMETRICS_SSIM_AVAILABLE: logger.warning("torchmetrics SSIM not found. Skip SSIM.")

    def train_step(self, batch_video_frames: torch.Tensor):
        """ Performs a single training step including forward, loss, and backward """
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        num_cond = self.video_config["num_input_frames"]
        num_pred = self.video_config["num_predict_frames"]
        B = batch_video_frames.shape[0]
        device = batch_video_frames.device
        dtype = m_ref.encoder.wubu_s.output_tangent_projection.weight.dtype # Use model's compute dtype

        # 1. Split into conditioning and target frames (pixels)
        cond_pixels = batch_video_frames[:, :num_cond, ...].to(device)
        target_pixels = batch_video_frames[:, num_cond : num_cond + num_pred, ...].to(device)

        # 2. Encode conditioning and target frames
        with torch.no_grad(): # Encoder should not accumulate grads during target encoding
             # Encode conditioning frames
             cond_app_features, cond_gaad_bboxes, cond_motion_features = m_ref.encode_frames(cond_pixels)
             # Encode target frames to get the "clean" regional features (x0)
             x0_regional_hyperbolic, target_gaad_bboxes, _ = m_ref.encode_frames(target_pixels)
             x0_regional_hyperbolic = x0_regional_hyperbolic.to(dtype) # Ensure correct dtype

        # 3. Sample timestep t and noise
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)
        tangent_noise = torch.randn_like(x0_regional_hyperbolic, dtype=dtype) # Noise in tangent space

        # 4. Create noisy features xt using q_sample_regional
        xt_regional_hyperbolic, actual_noise_tangent = q_sample_regional(
            x0_regional_hyperbolic, t, self.encoder_final_manifold,
            self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, tangent_noise
        )
        # Map xt (hyperbolic) to tangent space for noise predictor input
        xt_regional_tangent = self.encoder_final_manifold.logmap0(xt_regional_hyperbolic)


        # 5. Prepare for CFG (Classifier-Free Guidance)
        is_this_batch_unconditional = False
        if self.args.cfg_unconditional_dropout_prob > 0 and torch.rand(1).item() < self.args.cfg_unconditional_dropout_prob:
            is_this_batch_unconditional = True
            # For unconditional, set conditioning features to None (or zeros?)
            cond_app_features_for_model = None
            cond_motion_features_for_model = None
        else:
            cond_app_features_for_model = cond_app_features.to(dtype) if cond_app_features is not None else None
            cond_motion_features_for_model = cond_motion_features.to(dtype) if cond_motion_features is not None else None


        # 6. Predict noise using the model
        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'):
            predicted_noise_tangent = self.model(
                xt_regional_tangent, # Input is tangent space
                t,
                cond_app_features_for_model,
                cond_motion_features_for_model,
                cfg_unconditional_flag=is_this_batch_unconditional
            ) # Output is tangent space noise

            # --- LOSS CALCULATION ---
            if self.args.loss_type == 'noise_tangent':
                # Option 1: Predict noise in tangent space
                loss = F.mse_loss(predicted_noise_tangent, actual_noise_tangent)
            elif self.args.loss_type == 'pixel_reconstruction':
                # Option 2: Predict x0_hat and decode to pixels
                # Estimate x0 in tangent space from xt and predicted noise
                sqrt_alpha_t = self.sqrt_alphas_cumprod.gather(0, t).view(B,1,1,1).to(device, dtype)
                sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(B,1,1,1).to(device, dtype)

                x0_pred_tangent = (xt_regional_tangent - sqrt_one_minus_alpha_t * predicted_noise_tangent) / (sqrt_alpha_t + EPS)
                # Decode the predicted x0_pred_tangent to pixels
                # Need the GAAD bboxes corresponding to the *target* frames
                x0_pred_pixels = m_ref.decode_features(x0_pred_tangent, target_gaad_bboxes.to(dtype)) # Requires target bboxes
                # Pixel loss
                loss = F.mse_loss(x0_pred_pixels, target_pixels)
            else:
                raise ValueError(f"Unknown loss_type: {self.args.loss_type}")

        return loss, predicted_noise_tangent.detach(), actual_noise_tangent.detach()

    def train(self, start_epoch:int=0, initial_global_step:int=0):
        self.global_step = initial_global_step; self.current_epoch = start_epoch
        if self.am_main_process: logger.info(f"Training from epoch {start_epoch}, step {initial_global_step}...")
        total_loss_interval = 0.0; items_interval = 0; current_cycle_loss_sum = 0.0; micro_batches_in_current_cycle = 0; avg_loss_for_q_cycle = 0.0; current_unclipped_grad_norm = 0.0
        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process: logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting...")
            if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler): self.train_loader.sampler.set_epoch(epoch)
            self.model.train(); total_micro_batches_estimate = None; dataset_len = 0
            try:
                if hasattr(self.train_loader.sampler, '__len__'): dataset_len = len(self.train_loader.sampler) # type: ignore
                elif hasattr(self.train_loader.dataset, '__len__'): dset_total_len = len(self.train_loader.dataset); dataset_len = dset_total_len // self.world_size if self.world_size > 0 else dset_total_len
                if dataset_len > 0: loader_batch_size = self.train_loader.batch_size or 1; total_micro_batches_estimate = math.ceil(dataset_len / loader_batch_size)
            except Exception: logger.warning("Could not estimate epoch length for tqdm.", exc_info=False)
            prog_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", disable=not self.am_main_process or os.getenv('CI')=='true', dynamic_ncols=True, total=total_micro_batches_estimate)
            for batch_idx, batch_frames_raw in enumerate(prog_bar):
                batch_frames = batch_frames_raw.to(self.device); is_last_batch_in_loader = (batch_idx + 1) == total_micro_batches_estimate if total_micro_batches_estimate is not None else False; loss_this_micro_batch = torch.tensor(0.0, device=self.device); is_optimizer_step_time = ((micro_batches_in_current_cycle + 1) % self.args.grad_accum_steps == 0) or (is_last_batch_in_loader and (micro_batches_in_current_cycle + 1) > 0)
                sync_context = self.model.no_sync() if self.ddp_active and isinstance(self.model, DDP) and not is_optimizer_step_time else contextlib.nullcontext()
                with sync_context: # type: ignore
                    with (torch.autograd.detect_anomaly() if self.args.detect_anomaly else contextlib.nullcontext()):
                        try:
                            loss, _, _ = self.train_step(batch_frames)
                            if torch.isnan(loss) or torch.isinf(loss):
                                logger.warning(f"R{self.rank}: NaN/Inf loss ({loss.item()}). Skip micro-batch.")
                                if is_optimizer_step_time:
                                    current_cycle_loss_sum = 0.0
                                    micro_batches_in_current_cycle = 0
                                    self.optimizer.zero_grad(set_to_none=True) # Reset grads even if skipping step
                                continue
                            loss_this_micro_batch = loss
                            loss_scaled_for_backward = loss / self.args.grad_accum_steps
                            self.scaler.scale(loss_scaled_for_backward).backward()
                        except Exception as e_train_step:
                            logger.error(f"Rank {self.rank}: Error in train_step/backward: {e_train_step}", exc_info=True)
                            if is_optimizer_step_time:
                                current_cycle_loss_sum = 0.0
                                micro_batches_in_current_cycle = 0
                                self.optimizer.zero_grad(set_to_none=True)
                            continue
                total_loss_interval += loss_this_micro_batch.item() * batch_frames.size(0); items_interval += batch_frames.size(0); current_cycle_loss_sum += loss_this_micro_batch.item(); micro_batches_in_current_cycle += 1
                if is_optimizer_step_time and micro_batches_in_current_cycle > 0:
                    self.scaler.unscale_(self.optimizer); current_unclipped_grad_norm = 0.0
                    params_for_norm_calc = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None and p.requires_grad]
                    if params_for_norm_calc:
                        try: all_norms_sq = [torch.sum(p.grad.detach().float()**2) for p in params_for_norm_calc]; finite_norms_sq = [n_sq for n_sq in all_norms_sq if torch.isfinite(n_sq)]; current_unclipped_grad_norm = torch.sqrt(torch.stack(finite_norms_sq).sum()).item() if finite_norms_sq else float('inf')
                        except Exception as e_norm_calc: logger.error(f"R{self.rank} S{self.global_step}: Error calc grad norm for Q: {e_norm_calc}", exc_info=True); current_unclipped_grad_norm = float('inf')
                    if self.args.global_max_grad_norm > 0 and np.isfinite(current_unclipped_grad_norm) and current_unclipped_grad_norm > self.args.global_max_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.global_max_grad_norm)

                    # Update Q-Controller state *before* optimizer step
                    if hasattr(self.optimizer, 'q_controller') and self.optimizer.q_controller:
                        q_ctrl = self.optimizer.q_controller; avg_loss_for_q_cycle = current_cycle_loss_sum / micro_batches_in_current_cycle; current_lr_for_q_state = self.optimizer.param_groups[0]['lr']; current_mom_for_q_state = self.optimizer.param_groups[0]['momentum']; q_state = q_ctrl.get_state(current_lr_for_q_state, current_mom_for_q_state, current_unclipped_grad_norm, avg_loss_for_q_cycle)
                        if q_ctrl.prev_state is not None and q_ctrl.prev_action is not None and q_ctrl.prev_loss is not None and q_state is not None:
                            reward = q_ctrl.compute_reward(avg_loss_for_q_cycle, q_ctrl.prev_loss, current_unclipped_grad_norm)
                            if np.isfinite(reward): q_ctrl.update(q_ctrl.prev_state, q_ctrl.prev_action, reward, q_state)
                            else: logger.warning(f"R{self.rank} S{self.global_step}: Q-Ctrl non-finite reward.")
                        q_ctrl.prev_state = q_state
                        if q_state is not None: q_ctrl.prev_action = q_ctrl.choose_action(q_state)
                        else: logger.warning(f"R{self.rank} S{self.global_step}: Q-state None. Q-action may be stale/default."); # q_ctrl.prev_action = q_ctrl.prev_action # Keep previous action or it defaults
                        q_ctrl.prev_loss = avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else q_ctrl.prev_loss

                    # Optimizer Step
                    self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True); self.global_step += 1

                    # Logging
                    if self.global_step % self.args.log_interval == 0 and self.am_main_process:
                        log_lr = self.optimizer.param_groups[0]['lr']
                        log_metrics = {"train/loss_cycle_avg": avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else -1.0, "train/lr_effective": log_lr, "train/grad_norm_unclipped_for_q": current_unclipped_grad_norm if np.isfinite(current_unclipped_grad_norm) else -1.0, "epoch_frac": epoch + ((batch_idx + 1) / total_micro_batches_estimate if total_micro_batches_estimate and total_micro_batches_estimate > 0 else 0), "global_step": self.global_step}
                        if hasattr(self.optimizer, 'q_controller') and self.optimizer.q_controller: log_metrics.update({f"q_ctrl/{k}": v for k,v in self.optimizer.get_q_controller_info().items()})
                        if hasattr(self.optimizer, 'grad_stats'): log_metrics.update({f"grad_stats/{k}": v for k,v in self.optimizer.get_gradient_stats_summary().items()})

                        # Flatten nested dicts for logging
                        log_metrics_flat = {}
                        for k, v in log_metrics.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    log_metrics_flat[f"{k}/{sub_k}"] = sub_v
                            else:
                                log_metrics_flat[k] = v

                        logger.info(f"E {epoch+1} S{self.global_step} L(cyc){log_metrics_flat.get('train/loss_cycle_avg', -1.0):.4f} LR {log_metrics_flat.get('train/lr_effective', 0.0):.2e} GradN(Q){log_metrics_flat.get('train/grad_norm_unclipped_for_q', -1.0):.2f}")
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(log_metrics_flat, step=self.global_step)
                        total_loss_interval = 0.0; items_interval = 0

                    # Checkpointing
                    if self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0 and self.am_main_process: self._save_checkpoint(is_intermediate=True, metrics={"train_loss_cycle_avg": avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else -1.0})

                    # Reset cycle accumulators
                    current_cycle_loss_sum = 0.0; micro_batches_in_current_cycle = 0

            # End of Epoch
            if self.am_main_process:
                avg_epoch_loss_val = total_loss_interval / items_interval if items_interval > 0 else (avg_loss_for_q_cycle if micro_batches_in_current_cycle == 0 and is_optimizer_step_time else float('nan'))
                logger.info(f"Epoch {epoch+1} finished. Approx avg epoch loss: {avg_epoch_loss_val:.4f}")

            # Validation
            if self.val_loader and self.am_main_process:
                val_metrics_dict = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if self.args.wandb and WANDB_AVAILABLE and wandb.run and val_metrics_dict: wandb.log({f"val/{k}": v for k,v in val_metrics_dict.items()}, step=self.global_step)
                current_val_primary_metric = val_metrics_dict.get(self.args.val_primary_metric, float('inf'))
                if current_val_primary_metric < self.best_val_loss : self.best_val_loss = current_val_primary_metric; self._save_checkpoint(is_best=True, metrics=val_metrics_dict)

            # End-of-epoch checkpoint
            if self.am_main_process:
                save_metrics = self.last_val_metrics.copy() if self.last_val_metrics else {}
                save_metrics["epoch_end_train_loss_logged_intervals_avg"] = avg_epoch_loss_val if np.isfinite(avg_epoch_loss_val) else -1.0
                self._save_checkpoint(metrics=save_metrics)


    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Dict[str, float]:
        if not self.val_loader or not self.am_main_process: return {"avg_val_pixel_mse": float('inf')}
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        m_ref.eval(); total_mse_pixel = 0.0; total_psnr = 0.0; total_ssim = 0.0; total_lpips_val = 0.0; total_val_items = 0; logged_samples_count = 0; wandb_val_samples = []
        for batch_idx, batch_frames_raw in enumerate(tqdm(self.val_loader, desc="Validating", dynamic_ncols=True, disable=os.getenv('CI')=='true' or not self.am_main_process)):
            batch_frames = batch_frames_raw.to(self.device);
            num_cond = self.video_config["num_input_frames"]
            num_pred = self.video_config["num_predict_frames"]
            cond_pixels = batch_frames[:, :num_cond, ...]
            target_pixels_ground_truth = batch_frames[:, num_cond : num_cond + num_pred, ...]
            B_val = target_pixels_ground_truth.shape[0]
            dtype = m_ref.encoder.wubu_s.output_tangent_projection.weight.dtype # Use model's compute dtype

            # Sample pixels
            predicted_target_pixels = self.sample(
                 conditioning_frames_pixels=cond_pixels,
                 num_inference_steps=self.args.val_sampling_steps,
                 sampler_type=self.args.val_sampler_type,
                 ddim_eta=0.0, # Typically 0 for validation DDIM
                 cfg_guidance_scale=self.args.val_cfg_scale,
                 force_on_main_process=True, # Validation runs on main process
                 batch_size_if_uncond=B_val
                 ).to(dtype) # Ensure dtype matches for metrics

            # Compare first predicted frame with first target frame for metrics
            pred_for_metrics = predicted_target_pixels[:,0,...];
            gt_for_metrics = target_pixels_ground_truth[:,0,...].to(dtype);

            # Normalize to [0, 1] for metrics
            pred_norm = (pred_for_metrics.clamp(-1, 1) + 1) / 2.0;
            gt_norm = (gt_for_metrics.clamp(-1, 1) + 1) / 2.0

            mse_pixel_val = F.mse_loss(pred_norm, gt_norm);
            total_mse_pixel += mse_pixel_val.item() * B_val;
            psnr_val = 10 * torch.log10(1.0 / (mse_pixel_val + EPS)) if mse_pixel_val > 0 else torch.tensor(100.0, device=self.device);
            total_psnr += psnr_val.item() * B_val

            if self.ssim_metric:
                try: ssim_val_current = self.ssim_metric(pred_norm, gt_norm); total_ssim += ssim_val_current.item() * B_val
                except Exception as e_ssim: logger.warning(f"SSIM calculation failed: {e_ssim}")
            if self.lpips_loss_fn:
                try: lpips_val_current = self.lpips_loss_fn(pred_for_metrics, gt_for_metrics).mean(); total_lpips_val += lpips_val_current.item() * B_val
                except Exception as e_lpips: logger.warning(f"LPIPS calculation failed: {e_lpips}")


            total_val_items += B_val

            # Log samples to WandB
            if self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run and logged_samples_count < num_val_samples_to_log:
                num_to_log_this_batch = min(B_val, num_val_samples_to_log - logged_samples_count)
                for k_sample in range(num_to_log_this_batch):
                    cond_imgs_wandb = [wandb.Image(cond_pixels[k_sample, i].cpu().clamp(-1,1)*0.5+0.5, caption=f"Val Cond {i}") for i in range(cond_pixels.shape[1])];
                    pred_img_wandb = wandb.Image(pred_for_metrics[k_sample].cpu().clamp(-1,1)*0.5+0.5, caption="Val Pred");
                    gt_img_wandb = wandb.Image(gt_for_metrics[k_sample].cpu().clamp(-1,1)*0.5+0.5, caption="Val GT");
                    wandb_val_samples.extend(cond_imgs_wandb + [pred_img_wandb, gt_img_wandb])
                logged_samples_count += num_to_log_this_batch

        avg_mse_pixel = total_mse_pixel / total_val_items if total_val_items > 0 else float('inf');
        avg_psnr = total_psnr / total_val_items if total_val_items > 0 else 0.0;
        avg_ssim = total_ssim / total_val_items if total_val_items > 0 and self.ssim_metric else 0.0;
        avg_lpips_metric = total_lpips_val / total_val_items if total_val_items > 0 and self.lpips_loss_fn else 0.0

        metrics = {"avg_val_pixel_mse": avg_mse_pixel, "avg_val_psnr": avg_psnr, "avg_val_ssim": avg_ssim, "avg_val_lpips": avg_lpips_metric};
        self.last_val_metrics = metrics;
        logger.info(f"Validation Metrics: Pixel MSE: {avg_mse_pixel:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips_metric:.4f}")
        if wandb_val_samples and self.args.wandb and WANDB_AVAILABLE and wandb.run:
            wandb.log({"validation_samples_sequence": wandb_val_samples}, step=self.global_step)
        return metrics


    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict]=None, is_best:bool=False):
        if not self.am_main_process: return
        m_save = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        data = {
            'global_step':self.global_step,
            'epoch':self.current_epoch,
            'model_state_dict':m_save.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scaler_state_dict':self.scaler.state_dict() if self.args.use_amp and self.device.type=='cuda' else None,
            'args':vars(self.args), # Save args used for this run
            'metrics':metrics if metrics else self.last_val_metrics,
            # Save relevant sub-configs (can be derived from args, but good practice)
            'video_config': self.video_config,
            'gaad_appearance_config': self.gaad_appearance_config,
            'gaad_motion_config': self.gaad_motion_config,
            'wubu_s_config': self.wubu_s_config,
            'wubu_t_config': self.wubu_t_config,
            'wubu_m_config': self.wubu_m_config,
            'transformer_noise_predictor_config': getattr(self.args, 'transformer_noise_predictor_config', {})
        }
        fname_prefix="wuburegional_diff_ckpt_v010"; fpath = ""
        if is_best: fpath = os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_best.pt")
        elif is_intermediate: fpath = os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_step{self.global_step}.pt")
        else: fpath = os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_ep{self.current_epoch+1}_step{self.global_step}.pt")
        try: torch.save(data,fpath); logger.info(f"Ckpt saved: {fpath}")
        except Exception as e: logger.error(f"Save ckpt error {fpath}: {e}",exc_info=True)

    def load_checkpoint(self, checkpoint_path:str) -> Tuple[int,int]:
        if not os.path.exists(checkpoint_path): logger.warning(f"Ckpt {checkpoint_path} not found."); return 0,0
        try:
            ckpt=torch.load(checkpoint_path,map_location=self.device);
            m_load = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model

            # TODO: Add compatibility check for loaded args vs current args
            loaded_args = ckpt.get('args', {})
            if loaded_args:
                 logger.info("Loaded args from checkpoint.")
                 # Compare key args, e.g., dimensions, num_regions?

            try: m_load.load_state_dict(ckpt['model_state_dict'])
            except RuntimeError as e: logger.warning(f"Strict load failed: {e}. Trying non-strict."); m_load.load_state_dict(ckpt['model_state_dict'],strict=False)

            if 'optimizer_state_dict' in ckpt and self.optimizer: self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt and self.scaler and ckpt['scaler_state_dict'] is not None: self.scaler.load_state_dict(ckpt['scaler_state_dict'])

            loaded_global_step = ckpt.get('global_step',0); loaded_epoch = ckpt.get('epoch',0);
            self.best_val_loss = ckpt.get('metrics',{}).get(self.args.val_primary_metric, float('inf')) if 'metrics' in ckpt else float('inf')

            logger.info(f"Loaded ckpt {checkpoint_path} (Step {loaded_global_step}, Ep {loaded_epoch}). BestValLoss: {self.best_val_loss:.4f}");
            return loaded_global_step, loaded_epoch
        except Exception as e: logger.error(f"Load ckpt error {checkpoint_path}: {e}",exc_info=True); return 0,0

    # --- Sampling Functions (Adapted for Regional Latents) ---
    @torch.no_grad()
    def p_sample_ddpm(self, xt_regional_tangent: torch.Tensor,
                      conditioning_app_features: Optional[torch.Tensor], # Hyperbolic
                      conditioning_motion_features: Optional[torch.Tensor], # Hyperbolic
                      t_tensor: torch.Tensor, t_int_val: int, cfg_guidance_scale: float = 1.0) -> torch.Tensor:
        """ Performs one DDPM sampling step in the regional tangent space. """
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        m_ref.eval()
        B, N_pred, NumReg, D_tan = xt_regional_tangent.shape
        shape_for_broadcast = (B, 1, 1, 1); # Broadcast across N_pred, NumReg, D_tan
        dev = xt_regional_tangent.device; dtype = xt_regional_tangent.dtype

        # Get schedule constants for time t
        betas_t = self.betas.gather(0, t_tensor).view(shape_for_broadcast).to(dev,dtype)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t_tensor).view(shape_for_broadcast).to(dev,dtype)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(0, t_tensor).view(shape_for_broadcast).to(dev,dtype)
        posterior_log_variance_clipped_t = self.posterior_log_variance_clipped.gather(0, t_tensor).view(shape_for_broadcast).to(dev,dtype)

        # Predict noise (handles CFG internally)
        pred_noise_tangent = m_ref(
            xt_regional_tangent, t_tensor,
            conditioning_app_features, conditioning_motion_features,
            cfg_unconditional_flag = (conditioning_app_features is None and cfg_guidance_scale > 1.0) # Uncond if no cond AND cfg>1
        )

        # Calculate mean using the DDPM formula (in tangent space)
        term_in_paren = xt_regional_tangent - (betas_t / (sqrt_one_minus_alphas_cumprod_t + EPS)) * pred_noise_tangent
        model_mean_tangent = sqrt_recip_alphas_t * term_in_paren

        if t_int_val == 0:
            return model_mean_tangent # Final step, return mean
        else:
            noise_sample = torch.randn_like(xt_regional_tangent)
            return model_mean_tangent + (0.5 * posterior_log_variance_clipped_t).exp() * noise_sample

    @torch.no_grad()
    def p_sample_ddim(self, xt_regional_tangent: torch.Tensor,
                      conditioning_app_features: Optional[torch.Tensor], # Hyperbolic
                      conditioning_motion_features: Optional[torch.Tensor], # Hyperbolic
                      t_tensor: torch.Tensor, t_prev_tensor: torch.Tensor, eta: float = 0.0, cfg_guidance_scale: float = 1.0) -> torch.Tensor:
        """ Performs one DDIM sampling step in the regional tangent space. """
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        m_ref.eval()
        B, N_pred, NumReg, D_tan = xt_regional_tangent.shape
        shape_for_broadcast = (B, 1, 1, 1);
        dev = xt_regional_tangent.device; dtype = xt_regional_tangent.dtype

        # Get schedule constants
        alphas_cumprod_t = self.alphas_cumprod.gather(0, t_tensor).view(shape_for_broadcast).to(dev,dtype)
        safe_t_prev = torch.clamp(t_prev_tensor, min=0)
        alphas_cumprod_t_prev = self.alphas_cumprod.gather(0, safe_t_prev).view(shape_for_broadcast).to(dev,dtype)
        # Ensure alpha_prev = 1.0 when t_prev = -1 (i.e., t = 0)
        alphas_cumprod_t_prev = torch.where(t_prev_tensor.view(shape_for_broadcast) < 0, torch.ones_like(alphas_cumprod_t_prev), alphas_cumprod_t_prev)

        # Predict noise (handles CFG internally)
        pred_noise_tangent = m_ref(
            xt_regional_tangent, t_tensor,
            conditioning_app_features, conditioning_motion_features,
            cfg_unconditional_flag = (conditioning_app_features is None and cfg_guidance_scale > 1.0) # Uncond if no cond AND cfg>1
        )

        # Predict x0_hat (in tangent space)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(torch.clamp(1. - alphas_cumprod_t, min=EPS))
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
        x0_pred_tangent = (xt_regional_tangent - sqrt_one_minus_alphas_cumprod_t * pred_noise_tangent) / (sqrt_alphas_cumprod_t + EPS)

        # Clip x0_pred if needed (in tangent space)
        if self.args.ddim_x0_clip_val > 0:
             x0_pred_tangent = torch.clamp(x0_pred_tangent, -self.args.ddim_x0_clip_val, self.args.ddim_x0_clip_val)

        # Calculate DDIM step components
        sigma_t_num = torch.clamp(1. - alphas_cumprod_t_prev, min=0.0)
        sigma_t_den = torch.clamp(1. - alphas_cumprod_t, min=EPS)
        sigma_t_ratio_alphacomp = torch.clamp(1. - alphas_cumprod_t / (alphas_cumprod_t_prev + EPS), min=0.0)
        sigma_t = eta * torch.sqrt( (sigma_t_num / sigma_t_den) * sigma_t_ratio_alphacomp )

        # Don't add noise variance when t_prev = -1
        sigma_t = torch.where(t_prev_tensor.view(shape_for_broadcast) < 0, torch.zeros_like(sigma_t), sigma_t)

        # Direction pointing to xt
        pred_dir_xt = torch.sqrt(torch.clamp(1. - alphas_cumprod_t_prev - sigma_t**2, min=0.0)) * pred_noise_tangent

        # Calculate x_{t-1}
        xt_prev_tangent = torch.sqrt(alphas_cumprod_t_prev) * x0_pred_tangent + pred_dir_xt

        # Add noise perturbation if eta > 0
        if eta > 0 and t_prev_tensor.min() >= 0 :
            xt_prev_tangent = xt_prev_tangent + sigma_t * torch.randn_like(xt_regional_tangent)

        return xt_prev_tangent

    @torch.no_grad()
    def sample(self, conditioning_frames_pixels: Optional[torch.Tensor], # (B, N_cond, C, H, W)
                num_inference_steps: Optional[int] = None, sampler_type: str = "ddpm",
                ddim_eta: float = 0.0, cfg_guidance_scale: float = 1.0,
                force_on_main_process: bool = False, batch_size_if_uncond: int = 1
                ) -> torch.Tensor:
        """ Samples pixel frames using the specified sampler. """
        if not (self.am_main_process or force_on_main_process):
            logger.warning(f"R{self.rank}: Sample on non-main, not forced. Skip."); return torch.empty(0, device=self.device)

        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        m_ref.eval(); m_ref.to(self.device)
        B = conditioning_frames_pixels.shape[0] if conditioning_frames_pixels is not None else batch_size_if_uncond
        dev = self.device
        dtype = m_ref.encoder.wubu_s.output_tangent_projection.weight.dtype # Use model's compute dtype

        eff_steps = min(num_inference_steps if num_inference_steps is not None else self.timesteps, self.timesteps)
        num_pred_frames_target = self.video_config["num_predict_frames"]
        num_regions = self.gaad_appearance_config["num_regions"]
        tangent_feat_dim = self.video_config['wubu_s_output_dim'] # Dimension of the tangent features

        # 1. Encode conditioning frames (if provided)
        cond_app_features: Optional[torch.Tensor] = None
        cond_motion_features: Optional[torch.Tensor] = None
        cond_gaad_bboxes: Optional[torch.Tensor] = None # Needed if decoder needs bboxes consistent with cond frames? No, decoder uses target bboxes.
        if conditioning_frames_pixels is not None:
            cond_app_features, cond_gaad_bboxes_app, cond_motion_features = m_ref.encode_frames(conditioning_frames_pixels.to(dev, dtype))
            # Store the GAAD bboxes from the *last* conditioning frame to use for the decoder
            # Assuming the layout is reasonably consistent frame-to-frame for sampling
            decoder_bboxes = cond_gaad_bboxes_app[:, -1:, ...].repeat(1, num_pred_frames_target, 1, 1) # (B, N_pred, NumReg, 4)
            if self.args.decoder_use_target_gt_bboxes_for_sampling and self.val_loader:
                # Hacky: try to get corresponding target GT bboxes if possible (e.g., from val run)
                # This is generally not feasible in pure inference. Sticking to last cond frame bboxes.
                pass
        else: # Unconditional - need placeholder bboxes for decoder
            # Generate default bboxes for the target image size
            frame_dims = (self.args.image_w, self.args.image_h)
            default_bboxes = golden_subdivide_rect_fixed_n(frame_dims, num_regions, device=dev, dtype=dtype, min_size_px=self.args.gaad_min_size_px)
            decoder_bboxes = default_bboxes.unsqueeze(0).unsqueeze(0).repeat(B, num_pred_frames_target, 1, 1)

        # 2. Initialize noise in TANGENT space
        xt_regional_tangent = torch.randn((B, num_pred_frames_target, num_regions, tangent_feat_dim), device=dev, dtype=dtype)

        # 3. Sampling loop
        time_schedule_indices = torch.linspace(self.timesteps - 1, 0, eff_steps, dtype=torch.long, device=dev)
        cond_str = f"CondFrames={conditioning_frames_pixels.shape[1]}" if conditioning_frames_pixels is not None else "UNCOND";
        proc_id_str = f"R{self.rank}" if self.ddp_active else "Main";
        logger.info(f"{proc_id_str}(Forced:{force_on_main_process}): Sampling {sampler_type.upper()}. BS={B}, {cond_str}, Steps={eff_steps}, Eta={ddim_eta if sampler_type=='ddim' else 'N/A'}, CFG={cfg_guidance_scale}")

        for i in tqdm(range(eff_steps), desc="Sampling", leave=False, dynamic_ncols=True, disable=not (self.am_main_process or force_on_main_process) or os.getenv('CI') == 'true'):
            t_idx = time_schedule_indices[i];
            t_batch = torch.full((B,), t_idx.item(), device=dev, dtype=torch.long);

            # Select conditioning based on CFG scale
            current_cond_app = cond_app_features if cfg_guidance_scale > 1.0 else None
            current_cond_motion = cond_motion_features if cfg_guidance_scale > 1.0 else None

            if sampler_type.lower() == "ddim":
                t_prev_idx = time_schedule_indices[i + 1] if i < eff_steps - 1 else torch.tensor(-1, device=dev, dtype=torch.long);
                t_prev_batch = torch.full((B,), t_prev_idx.item(), device=dev, dtype=torch.long);
                xt_regional_tangent = self.p_sample_ddim(xt_regional_tangent, current_cond_app, current_cond_motion, t_batch, t_prev_batch, eta=ddim_eta, cfg_guidance_scale=cfg_guidance_scale)
            elif sampler_type.lower() == "ddpm":
                xt_regional_tangent = self.p_sample_ddpm(xt_regional_tangent, current_cond_app, current_cond_motion, t_batch, t_idx.item(), cfg_guidance_scale=cfg_guidance_scale)
            else: raise ValueError(f"Unknown sampler: {sampler_type}")

        # 4. Decode final tangent features to pixels
        # xt_regional_tangent now represents the predicted x0_hat in tangent space
        predicted_pixels = m_ref.decode_features(xt_regional_tangent, decoder_bboxes.to(dtype))

        logger.info(f"{proc_id_str}(Forced:{force_on_main_process}): Sampling finished.")
        return predicted_pixels


# =====================================================================
# Arg Parsing and Main Execution Logic (Adapted for v0.10.0)
# =====================================================================

# --- Helper function for worker seeding (Pickle-friendly) ---
def seed_worker_init_fn(worker_id, base_seed, rank, world_size):
     """ Seeds a worker process based on its ID and rank. """
     # Use a unique seed calculation for each worker across all ranks
     worker_seed = base_seed + worker_id + rank * world_size # Offset by rank * world_size to ensure uniqueness
     random.seed(worker_seed)
     np.random.seed(worker_seed)
     torch.manual_seed(worker_seed)
     # No need to seed CUDA here usually, main process seeding should suffice


def seed_everything(seed:int,rank:int=0,world_size:int=1):
    """ Seeds main process libraries for reproducibility. """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    # Log seeding for main process if desired
    # logger.info(f"Seeded Main Process R{rank} with {actual_seed}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu-GAAD Regional Latent Diffusion Model (v0.10.0)")

    # --- Data and General ---
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_dir")
    parser.add_argument('--local_rank', type=int, default=-1) # For DDP
    parser.add_argument('--epochs', type=int, default=20) # Increased default
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_h', type=int, default=64)
    parser.add_argument('--image_w', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_input_frames', type=int, default=3, help="Number of conditioning frames.")
    parser.add_argument('--num_predict_frames', type=int, default=1, help="Number of frames to predict latents for (usually 1).")
    parser.add_argument('--frame_skip', type=int, default=1, help="Skip frames when creating sequences.")
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--checkpoint_dir',type=str, default='wuburegional_diff_checkpoints_v010')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--wandb_project',type=str,default='WuBuGAADRegionalDiffV010')
    parser.add_argument('--wandb_run_name',type=str,default=None)
    parser.add_argument('--log_interval',type=int, default=20)
    parser.add_argument('--save_interval',type=int, default=500) # Increased default

    # --- GAAD Parameters ---
    parser.add_argument('--gaad_num_regions', type=int, default=16, help="Number of regions for GAAD (Appearance).")
    parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])
    parser.add_argument('--gaad_min_size_px', type=int, default=4, help="Min size for subdivided regions.")
    # Motion GAAD
    parser.add_argument('--use_wubu_motion_branch', action='store_true', help="Enable the GAAD+WuBu-M motion processing branch.")
    parser.add_argument('--gaad_motion_num_regions', type=int, default=12, help="Number of regions for Motion GAAD.")
    parser.add_argument('--gaad_motion_decomposition_type', type=str, default="content_aware", choices=["spiral", "subdivide", "hybrid", "content_aware"], help="Content aware uses diff map.") # Added content_aware

    # --- Encoder Parameters ---
    parser.add_argument('--encoder_use_roi_align', action='store_true', help="Use shallow CNN + RoIAlign instead of pixel patches.")
    parser.add_argument('--encoder_shallow_cnn_channels', type=int, default=32, help="Output channels of shallow CNN if RoIAlign is used.")
    parser.add_argument('--encoder_roi_align_output_h', type=int, default=4, help="H of RoIAlign output if used.")
    parser.add_argument('--encoder_roi_align_output_w', type=int, default=4, help="W of RoIAlign output if used.")
    parser.add_argument('--encoder_pixel_patch_size', type=int, default=16, help="Size to resize pixel patches to if RoIAlign is NOT used.")
    parser.add_argument('--encoder_initial_tangent_dim', type=int, default=128, help="Dimension after patch embedding, input to WuBu-S/M.")

    # --- Decoder Parameters ---
    parser.add_argument('--decoder_type', type=str, default="patch_gen", choices=["patch_gen", "transformer"], help="Type of pixel synthesis decoder.")
    parser.add_argument('--decoder_patch_gen_size', type=int, default=16, help="Native size of patches generated by patch_gen decoder.")
    parser.add_argument('--decoder_patch_resize_mode', type=str, default="bilinear", choices=["bilinear", "nearest"], help="Interpolation for resizing generated patches.")
    # Add args for Transformer decoder if implemented

    # --- WuBu Parameters (Shared and Specific) ---
    parser.add_argument('--wubu_dropout', type=float, default=0.1)
    # WuBu-S (Appearance) - Applied per region
    parser.add_argument('--wubu_s_num_levels', type=int, default=2)
    parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32]) # Output dim defined by last value
    parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8])
    parser.add_argument('--wubu_s_use_rotation', action='store_true')
    parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true')
    # WuBu-M (Motion) - Applied per motion region
    parser.add_argument('--wubu_m_num_levels', type=int, default=2)
    parser.add_argument('--wubu_m_hyperbolic_dims', nargs='+', type=int, default=[64,32])
    parser.add_argument('--wubu_m_initial_curvatures', nargs='+', type=float, default=[1.0, 0.7])
    parser.add_argument('--wubu_m_use_rotation', action='store_true')
    parser.add_argument('--wubu_m_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_m_phi_influence_rotation_init', action='store_true')
    # WuBu-T (Temporal) - Applied to aggregated frame representations
    parser.add_argument('--wubu_t_num_levels', type=int, default=2)
    parser.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[128,64]) # Input dim derived from aggregated S (+M), output dim is last value
    parser.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[1.0,0.5])
    parser.add_argument('--wubu_t_use_rotation', action='store_true')
    parser.add_argument('--wubu_t_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true')

    # --- Latent Space & Noise Predictor Dimensions ---
    # WuBu-S output dim (hyperbolic feature dim per region)
    parser.add_argument('--wubu_s_output_dim', type=int, default=32, help="Final dimension of WuBu-S output per region (used as tangent dim for noise predictor).")
    # WuBu-M output dim (hyperbolic feature dim per motion region)
    parser.add_argument('--wubu_m_output_dim', type=int, default=32, help="Final dimension of WuBu-M output per motion region.")
     # WuBu-T output dim (temporal context vector dim)
    parser.add_argument('--wubu_t_output_dim', type=int, default=128, help="Final dimension of WuBu-T temporal context vector.")
    # Transformer Noise Predictor Args
    parser.add_argument('--tnp_num_layers', type=int, default=4)
    parser.add_argument('--tnp_num_heads', type=int, default=8)
    parser.add_argument('--tnp_d_model', type=int, default=256)
    parser.add_argument('--tnp_d_ff_ratio', type=float, default=4.0)
    parser.add_argument('--tnp_dropout', type=float, default=0.1)

    # --- Diffusion Parameters ---
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--beta_schedule',type=str,default='cosine', choices=['linear','cosine'])
    parser.add_argument('--beta_start',type=float,default=1e-4)
    parser.add_argument('--beta_end',type=float,default=0.02)
    parser.add_argument('--cosine_s',type=float,default=0.008)
    parser.add_argument('--phi_time_diffusion_scale', type=float, default=1.0, help="Global scale factor for time input to SinusoidalPhiEmbedding.")
    parser.add_argument('--phi_time_base_freq', type=float, default=10000.0, help="Base frequency/period parameter for SinusoidalPhiEmbedding.")
    parser.add_argument('--use_phi_frequency_scaling_for_time_emb', action='store_true', help="Use phi-based scaling for frequency spacing in SinusoidalPhiEmbedding.")
    parser.add_argument('--diffusion_time_embedding_dim', type=int, default=128)
    parser.add_argument('--loss_type', type=str, default='pixel_reconstruction', choices=['noise_tangent', 'pixel_reconstruction'], help="Loss function target.")

    # --- Training Parameters ---
    parser.add_argument('--learning_rate',type=float,default=5e-5)
    parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0, help="Per-parameter Riemannian grad norm clip in RiSGD. 0 to disable.")
    parser.add_argument('--global_max_grad_norm',type=float,default=1.0, help="Global gradient norm clip for all model params. 0 to disable.")
    parser.add_argument('--q_controller_enabled',action='store_true')
    parser.add_argument('--grad_accum_steps',type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--detect_anomaly',action='store_true')

    # --- CFG, Sampler, and Validation Parameters ---
    parser.add_argument('--cfg_unconditional_dropout_prob', type=float, default=0.1, help="Prob of dropping cond during training for CFG.")
    parser.add_argument('--val_cfg_scale', type=float, default=1.5, help="CFG scale for validation sampling.")
    parser.add_argument('--val_sampler_type', type=str, default="ddim", choices=["ddpm", "ddim"], help="Sampler for validation.")
    parser.add_argument('--val_sampling_steps', type=int, default=20, help="Sampling steps for validation.")
    parser.add_argument('--ddim_x0_clip_val', type=float, default=1.0, help="Value for clipping x0_pred_tangent in DDIM, 0 to disable.")
    parser.add_argument('--use_lpips_for_verification', action='store_true', help="Enable LPIPS metric for validation.")
    parser.add_argument('--validation_video_path', type=str, default=None, help="Path to a separate video file for validation.")
    parser.add_argument('--validation_split_fraction', type=float, default=0.1, help="Fraction of training data for validation if no separate val video.")
    parser.add_argument('--val_primary_metric', type=str, default="avg_val_pixel_mse", choices=["avg_val_pixel_mse", "avg_val_psnr", "avg_val_ssim", "avg_val_lpips"], help="Metric for best checkpoint.")
    parser.add_argument('--num_val_samples_to_log', type=int, default=2, help="Num validation samples to log to WandB.")
    parser.add_argument('--decoder_use_target_gt_bboxes_for_sampling', action='store_true', help="Attempt to use GT target bboxes during sampling (mainly for debug/oracle).")


    # --- Demo Sampling Params ---
    parser.add_argument('--demo_sampler_type', type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument('--demo_ddim_eta', type=float, default=0.0)
    parser.add_argument('--demo_cfg_scale', type=float, default=3.0)
    parser.add_argument('--demo_sampling_steps', type=int, default=50)

    parsed_args = parser.parse_args()

    # --- Post-parsing Validation and Configuration ---

    # Validate WuBu config list lengths
    def validate_wubu_config(args_obj, prefix_str, parser_ref, is_motion_branch_active_for_this_call):
        num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0)
        # Only validate if the branch is active OR if it's not the motion branch (S and T always validated if num_levels > 0)
        is_motion = prefix_str == "wubu_m"
        if num_levels > 0 and (not is_motion or is_motion_branch_active_for_this_call):
            dims_attr_name = f"{prefix_str}_hyperbolic_dims"
            curv_attr_name = f"{prefix_str}_initial_curvatures"
            dims = getattr(args_obj, dims_attr_name)
            curvatures = getattr(args_obj, curv_attr_name)
            if not isinstance(dims, list): dims = [dims]
            if not isinstance(curvatures, list): curvatures = [curvatures]

            if len(dims) != num_levels:
                if len(dims) == 1 and num_levels > 1: dims = [dims[0]] * num_levels; setattr(args_obj, dims_attr_name, dims)
                else: parser_ref.error(f"{prefix_str}: Length mismatch: hyperbolic_dims ({len(dims)}) vs num_levels ({num_levels})")
            if len(curvatures) != num_levels:
                if len(curvatures) == 1 and num_levels > 1: curvatures = [curvatures[0]] * num_levels
                elif not curvatures: curvatures = [1.0] * num_levels # Default if empty
                else: parser_ref.error(f"{prefix_str}: Length mismatch: initial_curvatures ({len(curvatures)}) vs num_levels ({num_levels})")
                setattr(args_obj, curv_attr_name, curvatures)

    validate_wubu_config(parsed_args, "wubu_s", parser, True)
    validate_wubu_config(parsed_args, "wubu_t", parser, True)
    validate_wubu_config(parsed_args, "wubu_m", parser, parsed_args.use_wubu_motion_branch)

    # Set derived output dims based on last list element
    if parsed_args.wubu_s_num_levels > 0: parsed_args.wubu_s_output_dim = parsed_args.wubu_s_hyperbolic_dims[-1]
    else: parsed_args.wubu_s_output_dim = parsed_args.encoder_initial_tangent_dim # If no WuBu-S, output is initial tangent
    if parsed_args.use_wubu_motion_branch and parsed_args.wubu_m_num_levels > 0: parsed_args.wubu_m_output_dim = parsed_args.wubu_m_hyperbolic_dims[-1]
    else: parsed_args.wubu_m_output_dim = 0
    if parsed_args.wubu_t_num_levels > 0: parsed_args.wubu_t_output_dim = parsed_args.wubu_t_hyperbolic_dims[-1]
    else: parsed_args.wubu_t_output_dim = 0 # No temporal context if no WuBu-T

    # Package Transformer config
    parsed_args.transformer_noise_predictor_config = {
        "num_layers": parsed_args.tnp_num_layers,
        "num_heads": parsed_args.tnp_num_heads,
        "d_model": parsed_args.tnp_d_model,
        "d_ff_ratio": parsed_args.tnp_d_ff_ratio,
        "dropout": parsed_args.tnp_dropout,
        "activation": "gelu", # Hardcoded for now
    }


    return parsed_args


def _configure_wubu_stack(args: argparse.Namespace, prefix: str) -> Optional[Dict]:
    # This helper remains mostly the same, ensures WuBu configs are populated correctly
    if prefix == "wubu_m" and not args.use_wubu_motion_branch: return None
    config = DEFAULT_CONFIG_WUBU.copy(); num_levels_val = getattr(args, f"{prefix}_num_levels", 0); config["num_levels"] = num_levels_val
    if num_levels_val == 0:
        for key in ["hyperbolic_dims", "initial_curvatures", "initial_scales", "initial_spread_values", "boundary_points_per_level", "transform_types", "transform_hidden_dims", "tangent_input_combination_dims"]: config[key] = [] if key not in ["tangent_input_combination_dims"] else [DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0]]
        return config
    config["hyperbolic_dims"] = getattr(args, f"{prefix}_hyperbolic_dims", DEFAULT_CONFIG_WUBU["hyperbolic_dims"]); config["initial_curvatures"] = getattr(args, f"{prefix}_initial_curvatures", DEFAULT_CONFIG_WUBU["initial_curvatures"]); config["use_rotation_in_transform"] = getattr(args, f"{prefix}_use_rotation", DEFAULT_CONFIG_WUBU["use_rotation_in_transform"]); config["phi_influence_curvature"] = getattr(args, f"{prefix}_phi_influence_curvature", DEFAULT_CONFIG_WUBU["phi_influence_curvature"]); config["phi_influence_rotation_init"] = getattr(args, f"{prefix}_phi_influence_rotation_init", DEFAULT_CONFIG_WUBU["phi_influence_rotation_init"]); config["dropout"] = args.wubu_dropout
    def _ensure_list_config_len(cfg_dict, key, target_len, default_fill_list_from_base):
        current_list_val = cfg_dict.get(key, []);
        if not isinstance(current_list_val, list): current_list_val = [current_list_val]
        base_default_val = default_fill_list_from_base[0] if default_fill_list_from_base else (1.0 if any(s in key for s in ["scales", "curvatures"]) else (0.1 if "spread" in key else ("linear" if "types" in key else (32 if key == "tangent_input_combination_dims" else None))))
        fill_val = current_list_val[-1] if current_list_val else base_default_val
        if len(current_list_val) < target_len: cfg_dict[key] = (current_list_val + [fill_val]*(target_len-len(current_list_val)))[:target_len]
        elif len(current_list_val) > target_len: cfg_dict[key] = current_list_val[:target_len]
    list_configs_to_ensure_len = [("hyperbolic_dims", "hyperbolic_dims"), ("initial_curvatures", "initial_curvatures"), ("initial_scales", "initial_scales"), ("initial_spread_values", "initial_spread_values"), ("boundary_points_per_level", "boundary_points_per_level"),]
    for key_to_check, default_key_in_base_config in list_configs_to_ensure_len: _ensure_list_config_len(config, key_to_check, num_levels_val, DEFAULT_CONFIG_WUBU[default_key_in_base_config])
    if not isinstance(config.get("tangent_input_combination_dims"), list): config["tangent_input_combination_dims"] = [config.get("tangent_input_combination_dims", DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0])]
    num_transitions_val = max(0, num_levels_val-1)
    if num_transitions_val > 0: _ensure_list_config_len(config,"transform_types",num_transitions_val,DEFAULT_CONFIG_WUBU["transform_types"]); _ensure_list_config_len(config,"transform_hidden_dims",num_transitions_val,DEFAULT_CONFIG_WUBU["transform_hidden_dims"])
    else: config["transform_types"]=[]; config["transform_hidden_dims"]=[]
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
        rank=0; local_rank=0; world_size=1
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type=='cuda': _ = torch.cuda.set_device(device)

    am_main_process=(rank==0)

    # Configure logging per process
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.basicConfig(level=log_level, format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s', force=True)

    logger.info(f"--- WuBuGAADRegionalLatentDiffV010 (R{rank}/{world_size},Dev {device},DDP:{ddp_active}) ---")
    seed_everything(args.seed,rank,world_size) # Seed the main process here

    if am_main_process: logger.info(f"Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_id = wandb.util.generate_id() if wandb.run is None else wandb.run.id
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name if args.wandb_run_name else f"wuburegional_v010_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                   config=vars(args), resume="allow", id=run_id)

    # --- Configure Model Components ---
    video_config = {
        "image_size":(args.image_h,args.image_w),"num_channels":args.num_channels,
        "num_input_frames":args.num_input_frames, "num_predict_frames":args.num_predict_frames,
        # Latent dimensions are now derived from WuBu configs in args
        "wubu_s_output_dim":args.wubu_s_output_dim,
        "wubu_m_output_dim": args.wubu_m_output_dim,
        "wubu_t_output_dim":args.wubu_t_output_dim,
    }
    gaad_appearance_config = {
        "num_regions":args.gaad_num_regions,
        "decomposition_type":args.gaad_decomposition_type,
        "min_size_px": args.gaad_min_size_px,
        "phi_time_diffusion_scale": args.phi_time_diffusion_scale,
        "phi_time_base_freq": args.phi_time_base_freq
    }
    gaad_motion_config = None
    if args.use_wubu_motion_branch:
        gaad_motion_config = {
            "num_regions": args.gaad_motion_num_regions,
            "decomposition_type": args.gaad_motion_decomposition_type,
            "min_size_px": args.gaad_min_size_px, # Use same min size for now
        }

    wubu_s_config = _configure_wubu_stack(args, "wubu_s")
    wubu_t_config = _configure_wubu_stack(args, "wubu_t")
    wubu_m_config = _configure_wubu_stack(args, "wubu_m")

    if am_main_process:
        logger.info(f"VideoCfg:{video_config}\nGAADAppearCfg:{gaad_appearance_config}\nGAADMotionCfg:{gaad_motion_config}\nWuBuS-Cfg:{wubu_s_config}\nWuBuT-Cfg:{wubu_t_config}\nWuBuM-Cfg:{wubu_m_config}\nTNP-Cfg:{args.transformer_noise_predictor_config}")

    # --- Initialize Model ---
    model = GAADWuBuRegionalDiffNet(args, video_config, gaad_appearance_config, gaad_motion_config,
                                    wubu_s_config, wubu_t_config, wubu_m_config).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model,log="all",log_freq=max(100,args.log_interval*5),log_graph=False)

    if ddp_active:
        # find_unused_parameters might be needed due to optional motion/temporal branches
        model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    q_cfg = DEFAULT_CONFIG_QLEARN_DIFFUSION.copy() if args.q_controller_enabled else None
    optimizer = RiemannianEnhancedSGD(model.parameters(),lr=args.learning_rate,q_learning_config=q_cfg,max_grad_norm_risgd=args.risgd_max_grad_norm)

    # --- Dataset and DataLoader Setup ---
    actual_video_path = args.video_data_path
    if "demo_video_data" in args.video_data_path: # Special handling for demo data
        actual_video_path = os.path.join(args.video_data_path, "dummy_video_v10.mp4") # New name for v10
        if am_main_process:
            if not os.path.isdir(args.video_data_path):
                os.makedirs(args.video_data_path, exist_ok=True)
            if not os.path.exists(actual_video_path) :
                logger.info(f"Demo video data at {actual_video_path} not found. Creating dummy video...")
                min_raw_frames_needed = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                num_dummy_frames = max(50, min_raw_frames_needed + 20)
                dummy_frames_tchw = torch.randint(0, 256, (num_dummy_frames, args.num_channels, args.image_h, args.image_w), dtype=torch.uint8)
                if VIDEO_IO_AVAILABLE and video_io is not None:
                    try: video_io.write_video(actual_video_path, dummy_frames_tchw.permute(0, 2, 3, 1), fps=10); logger.info(f"Created dummy video (torchvision.io): {actual_video_path}")
                    except Exception as e_tv_write: logger.error(f"Error creating dummy video with torchvision.io: {e_tv_write}", exc_info=True)
                else: logger.error("torchvision.io unavailable, cannot create dummy video.")
        if ddp_active: torch.distributed.barrier() # Wait for main process

    is_file_path = os.path.isfile(actual_video_path) and not os.path.isdir(actual_video_path)
    if not is_file_path:
        logger.error(f"Video path {actual_video_path} is not a file. VideoFrameDataset requires a file path. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)
    if not os.path.exists(actual_video_path):
         logger.error(f"Video file {actual_video_path} not found. Exiting.")
         if ddp_active and is_initialized(): destroy_process_group()
         sys.exit(1)

    total_frames_sample = args.num_input_frames + args.num_predict_frames

    full_dataset = None
    try:
        full_dataset = VideoFrameDataset(video_path=actual_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
    except Exception as e_dataset_init:
        logger.error(f"Failed to initialize main VideoFrameDataset from {actual_video_path}: {e_dataset_init}", exc_info=True)
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    if not full_dataset or len(full_dataset) == 0 :
        logger.error("Main dataset empty or failed to load. Check video path and content. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    train_dataset, val_dataset = full_dataset, None
    if args.validation_video_path and os.path.exists(args.validation_video_path) and os.path.isfile(args.validation_video_path):
        try:
            val_dataset = VideoFrameDataset(video_path=args.validation_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
            if len(val_dataset) > 0: logger.info(f"Using separate validation video: {args.validation_video_path}, {len(val_dataset)} samples.")
            else: logger.warning(f"Validation video {args.validation_video_path} loaded 0 samples."); val_dataset = None
        except Exception as e_val_dataset: logger.warning(f"Could not load validation dataset from {args.validation_video_path}: {e_val_dataset}."); val_dataset = None
    elif args.validation_split_fraction > 0 and len(full_dataset) > 10 :
        num_val_samples = int(len(full_dataset) * args.validation_split_fraction)
        num_train_samples = len(full_dataset) - num_val_samples
        if num_train_samples > 0 and num_val_samples > 0:
            # Use a generator with rank offset for consistent splits in DDP
            split_generator = torch.Generator().manual_seed(args.seed + rank)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [num_train_samples, num_val_samples], generator=split_generator
            )
            logger.info(f"Split main dataset: {len(train_dataset)} train, {len(val_dataset)} val samples.")
        else: logger.warning(f"Not enough samples ({len(full_dataset)}) for validation split."); val_dataset = None

    # --- Define worker init function using functools.partial ---
    import functools
    partial_seed_worker = functools.partial(
        seed_worker_init_fn,
        base_seed=args.seed,
        rank=rank,
        world_size=world_size
    )

    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True,seed=args.seed) if ddp_active else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=(device.type=='cuda'),
        worker_init_fn=partial_seed_worker if args.num_workers > 0 else None, # Use partial function
        drop_last=True
        # Removed lambda from worker_init_fn
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=val_sampler,
            pin_memory=(device.type=='cuda'),
            worker_init_fn=partial_seed_worker if args.num_workers > 0 else None, # Use partial function here too
            drop_last=False
        )
    elif am_main_process: logger.info("No validation dataset/loader configured or validation dataset is empty.")

    # --- Initialize Trainer ---
    trainer = DiffusionTrainer(model,optimizer,device,train_loader,val_loader,args,rank,world_size,ddp_active)
    start_global_step,start_epoch=0,0
    if args.load_checkpoint:
        start_global_step,start_epoch=trainer.load_checkpoint(args.load_checkpoint)

    # --- Training Loop ---
    try:
        trainer.train(start_epoch=start_epoch,initial_global_step=start_global_step)
    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: Training interrupted by user.")
    except Exception as train_exc:
        logger.error(f"Rank {rank}: Training loop crashed: {train_exc}",exc_info=True)
    finally:
        if am_main_process:
            logger.info("Finalizing run...")
            trainer._save_checkpoint(metrics=trainer.last_val_metrics if trainer.last_val_metrics else {}) # Save final checkpoint

            # --- Demo Sampling at End ---
            if args.epochs>0 and hasattr(trainer,'sample') and trainer.global_step > 0 and len(train_loader)>0:
                logger.info("DEMO SAMPLING...")
                try:
                    # Re-create iterator or reset dataloader state if necessary for demo sample
                    # A simple way is to get a new iterator
                    demo_batch_for_cond_iter = iter(train_loader)
                    demo_batch_for_cond = next(demo_batch_for_cond_iter)
                    demo_cond_pixels = demo_batch_for_cond[:, :args.num_input_frames, ...].to(device)
                    demo_cond_pixels = demo_cond_pixels[0:1] # Use only the first sample for demo

                    pred_pixels = trainer.sample(
                        demo_cond_pixels,
                        num_inference_steps=args.demo_sampling_steps,
                        sampler_type=args.demo_sampler_type,
                        ddim_eta=args.demo_ddim_eta,
                        cfg_guidance_scale=args.demo_cfg_scale,
                        force_on_main_process=True
                    )

                    logger.info(f"Demo predicted pixels shape: {pred_pixels.shape}")
                    if pred_pixels.numel() > 0 and pred_pixels.shape[0] > 0:
                        save_path_dir = os.path.join(args.checkpoint_dir, "demo_samples_v010")
                        os.makedirs(save_path_dir, exist_ok=True)
                        # Save conditioning frames
                        for i_demo in range(min(args.num_input_frames, demo_cond_pixels.shape[1])):
                            save_image(demo_cond_pixels[0, i_demo].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_path_dir, f"demo_cond_frame_{i_demo}.png"))
                        # Save predicted frames
                        for i_pred_demo in range(pred_pixels.shape[1]):
                             save_image(pred_pixels[0,i_pred_demo].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_path_dir, f"demo_pred_frame_{i_pred_demo}.png"))
                        logger.info(f"Saved demo sample frames to {save_path_dir}")
                        # Log to WandB
                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            wandb_images = [wandb.Image(demo_cond_pixels[0, i_demo].cpu().clamp(-1,1)*0.5+0.5, caption=f"Cond {i_demo}") for i_demo in range(min(args.num_input_frames, demo_cond_pixels.shape[1]))]
                            for i_pred_demo in range(pred_pixels.shape[1]):
                                wandb_images.append(wandb.Image(pred_pixels[0,i_pred_demo].cpu().clamp(-1,1)*0.5+0.5, caption=f"Pred {i_pred_demo}"))
                            wandb.log({"demo_sequence_final": wandb_images}, step=trainer.global_step)
                except StopIteration:
                    logger.warning("Demo sampling skipped: DataLoader was empty or could not get a batch.")
                except Exception as e_demo_sample:
                    logger.error(f"Demo sampling error: {e_demo_sample}",exc_info=True)

            if args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish()

        if ddp_active and is_initialized():
            destroy_process_group()
        logger.info(f"Rank {rank}: WuBuGAADRegionalLatentDiffNet (v0.10.0) script finished.")

if __name__ == "__main__":
    main()