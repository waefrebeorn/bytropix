# WuBuNestDiffusion_v0.05.2_GAAD_MotionWuBu_Live.py
# Diffusion Model with Golden Aspect Adaptive Decomposition (GAAD)
# for Appearance (F_t) and Motion (Diff_Map),
# Phi-Influenced WuBu Spatio-Temporal Nesting (S, T, and new M for Motion),
# designed for live video ingest and resume model paradigm.
# LAST UPDATE: Incorporating "Math Prover" findings (v0.05.2 internal rev) Numerical Stability Enhancements

# =====================================================================
# Python Imports and Setup (includes torchvision.ops.roi_align)
# =====================================================================
import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SubsetRandomSampler
import numpy as np

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

logger = logging.getLogger("WuBuGAADPhiMotionDiffV052")
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

# =====================================================================
# Geometric, Optimizer, WuBu Core Components
# =====================================================================
class HyperbolicUtils:
    @staticmethod
    def poincare_clip(x: torch.Tensor, c_scalar: float, radius: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_input_dtype = x.dtype # Preserve the original dtype of the input tensor x

        if c_scalar <= 0:
            # Euclidean case or invalid curvature: sanitize and return in original dtype
            x_compute = x.float()
            if not torch.isfinite(x_compute).all():
                try:
                    logger_pc_euc = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.poincare_clip")
                    logger_pc_euc.warning(f"Euclidean/Invalid C: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
                except NameError: # logger might not be defined if this class is used standalone
                     print(f"Warning: WuBuGAADPhiMotionDiffV052.HyperbolicUtils.poincare_clip: Euclidean/Invalid C: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
                x_compute = torch.nan_to_num(x_compute, nan=0.0, posinf=0.0, neginf=0.0)
            
            # If original was float16, clamp to float16 range before casting back
            if original_input_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                x_compute = torch.clamp(x_compute, min=-f16_max, max=f16_max)
            return x_compute.to(original_input_dtype)

        # Proceed with hyperbolic clipping, computations in float32
        sqrt_c = math.sqrt(max(c_scalar, eps))
        effective_radius_factor = min(radius, 1.0 - eps)
        max_norm_val_f32 = effective_radius_factor / sqrt_c # This is a float32 Python float

        x_compute = x.float() # Work with float32

        if not torch.isfinite(x_compute).all():
            try:
                logger_pc_hyp = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.poincare_clip")
                logger_pc_hyp.warning(f"Hyperbolic: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
            except NameError:
                 print(f"Warning: WuBuGAADPhiMotionDiffV052.HyperbolicUtils.poincare_clip: Hyperbolic: Non-finite input x (shape {x_compute.shape}). Sanitizing.")
            x_compute = torch.nan_to_num(x_compute, nan=0.0, posinf=0.0, neginf=0.0)

        x_norm_sq = torch.sum(x_compute.pow(2), dim=-1, keepdim=True)
        
        # Ensure sqrt_input is robustly positive
        sqrt_input_val = torch.clamp(x_norm_sq, min=0.0) + eps # Add eps after clamp
        sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=1.0, neginf=eps) # Handle rare NaN/Inf from x_norm_sq if it bypassed clamp logic
        sqrt_input_val.clamp_min_(eps) # Final safety: ensure >= eps

        norm = torch.sqrt(sqrt_input_val)
        
        cond = norm > max_norm_val_f32 # Compare f32 norm with f32 max_norm_val_f32
        
        # Ensure (norm + eps) is also robust for division
        norm_plus_eps_for_div = norm + eps
        norm_plus_eps_for_div.clamp_min_(eps) # Prevent division by zero or very small number

        scale_factor = torch.where(cond, max_norm_val_f32 / norm_plus_eps_for_div, torch.ones_like(norm))
        
        clipped_x_f32 = x_compute * scale_factor # Result is float32

        # Before casting back to original_input_dtype, clamp if original was float16
        if original_input_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            clipped_x_f32 = torch.clamp(clipped_x_f32, min=-f16_max, max=f16_max)
        
        final_clipped_x = clipped_x_f32.to(original_input_dtype)

        # Final safety net for the output tensor in its original dtype
        if not torch.isfinite(final_clipped_x).all():
            # If still non-finite, replace with a safe value (e.g., 0 or a scaled max_norm)
            # For NaN replacement, max_norm_val_f32 is a scalar. We need to fill.
            # Use a small multiple of max_norm_val_f32 if it's for posinf/neginf.
            # But for NaNs, 0.0 is often safest.
            current_max_norm_for_nan_fill = float(max_norm_val_f32) # Ensure Python float
            return torch.nan_to_num(final_clipped_x, nan=0.0, posinf=current_max_norm_for_nan_fill, neginf=-current_max_norm_for_nan_fill)
        
        return final_clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = v.dtype

        if c_scalar <= 0: # Euclidean case
            v_compute = v.float()
            if not torch.isfinite(v_compute).all():
                current_logger_exp_euc = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_exponential_map")
                current_logger_exp_euc.warning(f"Euclidean: Non-finite input v (shape {v_compute.shape}). Sanitizing.")
                v_compute = torch.nan_to_num(v_compute, nan=0.0, posinf=0.0, neginf=0.0)
            if original_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                v_compute = torch.clamp(v_compute, min=-f16_max, max=f16_max)
            return v_compute.to(original_dtype)

        # Hyperbolic case: all computations in float32
        v_compute = v.float()

        if not torch.isfinite(v_compute).all():
            current_logger_exp_hyp = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_exponential_map")
            current_logger_exp_hyp.warning(f"Hyperbolic: Non-finite input v (shape {v_compute.shape}). Sanitizing.")
            v_compute = torch.nan_to_num(v_compute, nan=0.0, posinf=0.0, neginf=0.0)

        v_norm_sq_unclamped = torch.sum(v_compute.pow(2), dim=-1, keepdim=True)
        v_norm_sq_clamped = torch.clamp(v_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)
        
        sqrt_input_val = v_norm_sq_clamped + eps
        sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps)
        sqrt_input_val.clamp_min_(eps)
        v_norm = torch.sqrt(sqrt_input_val)

        if not torch.isfinite(v_norm).all(): # Should not happen with the above measures
            current_logger_exp_hyp_vn_err = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_exponential_map")
            current_logger_exp_hyp_vn_err.error(f"v_norm non-finite despite sanitization! Fallback to zero vector. v_norm: {v_norm.min() if v_norm.numel()>0 else 'empty'}, {v_norm.max() if v_norm.numel()>0 else 'empty'}")
            return HyperbolicUtils.poincare_clip(torch.zeros_like(v_compute), c_scalar, eps=eps).to(original_dtype)


        sqrt_c_val = math.sqrt(max(c_scalar, eps)) # Python float
        scaled_radius_arg = float(scale_scalar) * sqrt_c_val * v_norm # v_norm is f32 tensor
        
        tanh_input_val = torch.clamp(scaled_radius_arg, min=-30.0, max=30.0)
        tanh_term_val = torch.tanh(tanh_input_val)
        
        denominator_lambda_candidate = sqrt_c_val * v_norm + eps
        denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps)
        
        lambda_v_val = torch.where(v_norm > eps, tanh_term_val / denominator_lambda_val, torch.full_like(v_norm, float(scale_scalar), dtype=torch.float32))
        
        mapped_v_f32 = lambda_v_val * v_compute # All float32

        if not torch.isfinite(mapped_v_f32).all():
            current_logger_exp_hyp_mv_err = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_exponential_map")
            current_logger_exp_hyp_mv_err.warning(f"mapped_v_f32 non-finite. Zeroing before Poincare clip. Shape: {v.shape}")
            mapped_v_f32 = torch.zeros_like(v_compute)
        
        # Poincare clip expects its input 'x' and will use x.dtype for its final output.
        # Since we want computations in f32 and final cast according to original_dtype,
        # we pass the f32 mapped_v_f32. poincare_clip will use its internal x_compute=x.float()
        # and then its final line will be .to(mapped_v_f32.dtype), which is f32.
        clipped_mapped_v_f32 = HyperbolicUtils.poincare_clip(mapped_v_f32, c_scalar, eps=eps)

        # Now, safely cast this f32 result to original_dtype
        final_result = clipped_mapped_v_f32
        if original_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            final_result = torch.clamp(clipped_mapped_v_f32, min=-f16_max, max=f16_max)
            
        return final_result.to(original_dtype)

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = y.dtype

        if c_scalar <= 0: # Euclidean case
            y_compute = y.float()
            if not torch.isfinite(y_compute).all():
                current_logger_log_euc = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_logarithmic_map")
                current_logger_log_euc.warning(f"Euclidean: Non-finite input y (shape {y_compute.shape}). Sanitizing.")
                y_compute = torch.nan_to_num(y_compute, nan=0.0, posinf=0.0, neginf=0.0)
            if original_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                y_compute = torch.clamp(y_compute, min=-f16_max, max=f16_max)
            return y_compute.to(original_dtype)

        # Hyperbolic case
        # Poincare clip first. It will return in original_dtype after internal f32 computation & f16 clamping.
        y_clipped_original_dtype = HyperbolicUtils.poincare_clip(y, c_scalar, eps=eps)
        
        y_compute = y_clipped_original_dtype.float() # Convert to f32 for logmap math

        # This check is mostly for extreme paranoia, as poincare_clip should be robust.
        if not torch.isfinite(y_compute).all():
            current_logger_log_hyp_yc_err = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_logarithmic_map")
            current_logger_log_hyp_yc_err.warning(f"y_compute (from y_clipped.float()) non-finite. Sanitizing. Shape: {y_compute.shape}")
            y_compute = torch.nan_to_num(y_compute, nan=0.0, posinf=0.0, neginf=0.0)

        y_norm_sq_unclamped = torch.sum(y_compute.pow(2), dim=-1, keepdim=True)
        y_norm_sq_clamped = torch.clamp(y_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)

        sqrt_input_val = y_norm_sq_clamped + eps
        sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps)
        sqrt_input_val.clamp_min_(eps)
        y_norm = torch.sqrt(sqrt_input_val)
        
        if not torch.isfinite(y_norm).all(): # Should not happen
            current_logger_log_hyp_yn_err = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_logarithmic_map")
            current_logger_log_hyp_yn_err.error(f"y_norm non-finite despite sanitization! Fallback to zero vector. y_norm: {y_norm.min() if y_norm.numel()>0 else 'empty'}, {y_norm.max() if y_norm.numel()>0 else 'empty'}")
            return torch.zeros_like(y, dtype=original_dtype)


        sqrt_c_val = math.sqrt(max(c_scalar, eps)) # Python float
        arctanh_arg_raw = sqrt_c_val * y_norm
        arctanh_arg_clamped = torch.clamp(arctanh_arg_raw, min=-1.0 + eps*10, max=1.0 - eps*10) # Ensure input to atanh is in (-1, 1)
        atanh_term_val = torch.atanh(arctanh_arg_clamped)
        
        denominator_lambda_candidate = float(scale_scalar) * sqrt_c_val * y_norm + eps
        denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps)
        
        default_lambda_y_val = 1.0 / max(float(scale_scalar), eps)
        lambda_y_val = torch.where(y_norm > eps, atanh_term_val / denominator_lambda_val, torch.full_like(y_norm, default_lambda_y_val, dtype=torch.float32))
        
        mapped_y_f32 = lambda_y_val * y_compute # All float32

        if not torch.isfinite(mapped_y_f32).all():
            current_logger_log_hyp_my_err = logging.getLogger("WuBuGAADPhiMotionDiffV052.HyperbolicUtils.scale_aware_logarithmic_map")
            current_logger_log_hyp_my_err.warning(f"mapped_y_f32 non-finite. Returning zeros. Shape: {y.shape}")
            mapped_y_f32 = torch.zeros_like(y_compute)

        # Safely cast this f32 result to original_dtype
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

def init_weights_general(m):
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine: nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm):
        if m.affine: nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.zeros_(m.bias) if m.bias is not None else None

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor: return F.softplus(param_unconstrained) + min_val
class BoundaryManifoldHyperbolic(nn.Module):
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold_c: float):
        super().__init__(); self.level_idx = level_idx; self.num_points = num_points; self.point_dim = point_dim; self.current_manifold_c = initial_manifold_c
        if num_points > 0 and point_dim > 0: self.hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim)); PoincareBall(initial_manifold_c).init_weights(self.hyperbolic_points_params, irange=1e-3); self.hyperbolic_points_params.manifold = PoincareBall(initial_manifold_c) # type: ignore
        else: self.register_parameter('hyperbolic_points_params', None)
    def set_current_manifold_c(self, c_scalar: float): self.current_manifold_c = c_scalar; self.hyperbolic_points_params.manifold = PoincareBall(c_scalar) if self.hyperbolic_points_params is not None else None # type: ignore
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
        if self.in_dim == 4 and self.phi_influence_rotation_init and hasattr(self, 'rot_axis_param'): angle = F.softplus(self.rot_angle_unconstrained) * self.phi_angle_scale; current_axis = self.rot_axis_param.to(x_tan.device).unsqueeze(0).expand(B_maybe, -1); angle_b = angle.unsqueeze(0).expand(B_maybe, 1); q_rot = quaternion_from_axis_angle(angle_b, current_axis); return self.rotation_module(x_tan) if self.rotation_module else x_tan
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
        expmap_main_out = m_out.expmap0(tan_main_out_clamped) if tan_main_out_clamped is not None else torch.zeros_like(point_in)
        expmap_bound_out = m_out.expmap0(tan_bound_out_clamped) if tan_bound_out_clamped is not None else None
        expmap_desc_out = m_out.expmap0(tan_desc_out_clamped) if tan_desc_out_clamped is not None else None
        return (expmap_main_out, expmap_bound_out, expmap_desc_out)

class HyperbolicWuBuNestingLevel(nn.Module):
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
        if self.use_ld and self.dim > 0: self.level_descriptor_param = nn.Parameter(torch.Tensor(dim)); PoincareBall(c_scalar=self.initial_curvature_val).init_weights(self.level_descriptor_param, irange=self.ld_init_scale); self.level_descriptor_param.manifold = PoincareBall(c_scalar=self.initial_curvature_val) # type: ignore
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
        B, S, D_in = point_in_hyperbolic.shape
        if self.dim == 0: dummy_out_shape = (B, S, 0); dummy_dtype_dev = {'device':point_in_hyperbolic.device, 'dtype':point_in_hyperbolic.dtype}; current_spread_tensor = self.get_current_spread_scalar_tensor().to(point_in_hyperbolic.dtype); return (torch.zeros(dummy_out_shape, **dummy_dtype_dev), torch.zeros(dummy_out_shape, **dummy_dtype_dev), None, None, current_spread_tensor)
        dev = point_in_hyperbolic.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else point_in_hyperbolic.dtype
        current_c_val = self.get_current_curvature_scalar(); current_s_val = self.get_current_scale_scalar(); current_sigma_out_tensor = self.get_current_spread_scalar_tensor(); current_manifold_obj = PoincareBall(c_scalar=current_c_val)
        if self.level_descriptor_param is not None and hasattr(self.level_descriptor_param, 'manifold'): self.level_descriptor_param.manifold = PoincareBall(c_scalar=current_c_val) # type: ignore
        if self.boundary_manifold_module is not None: self.boundary_manifold_module.set_current_manifold_c(current_c_val)
        point_in_proj = current_manifold_obj.proju(point_in_hyperbolic.to(dtype_to_use)); tan_main_component = current_manifold_obj.logmap0(point_in_proj)
        tan_rel_component = torch.zeros_like(tan_main_component); ld_point_self_hyperbolic = None
        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None]: tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)
        if self.use_ld and self.level_descriptor_param is not None: ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use))
        tan_desc_prev_level_component = torch.zeros_like(tan_main_component)
        if descriptor_point_in_hyperbolic is not None and self.use_ld : desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use)); tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)
        inputs_for_combiner = [tan_main_component]; inputs_for_combiner.append(tan_rel_component) if self.relative_vector_aggregation not in ['none', None] else None; inputs_for_combiner.append(tan_desc_prev_level_component) if self.use_ld else None
        if self.use_spread and sigma_in_scalar_tensor is not None: sigma_in_expanded = sigma_in_scalar_tensor.view(-1,1,1).expand(B,S,1).to(dtype_to_use); inputs_for_combiner.append(sigma_in_expanded)
        combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1); v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features)
        v_final_for_expmap_unclamped = v_combined_tangent_processed * current_s_val
        if self.use_flow and self.tangent_flow_module is not None: flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val; v_final_for_expmap_unclamped = v_final_for_expmap_unclamped + flow_effect
        scaled_output_tangent_for_expmap = torch.clamp(v_final_for_expmap_unclamped, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        point_this_level_out_hyperbolic = current_manifold_obj.expmap0(scaled_output_tangent_for_expmap); tangent_out_for_aggregation = v_combined_tangent_processed.to(dtype_to_use)
        boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype_to_use) if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None else None
        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None:
            if ld_point_self_hyperbolic.dim() == 1: descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B, S, -1).to(dtype_to_use)
            elif ld_point_self_hyperbolic.dim() == 2 and ld_point_self_hyperbolic.shape[0] == B and ld_point_self_hyperbolic.shape[1] == S: descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.to(dtype_to_use)
            else: descriptor_point_out_for_transform_hyperbolic = (ld_point_self_hyperbolic.unsqueeze(0).expand(B,S,-1) if ld_point_self_hyperbolic.dim() < 3 else ld_point_self_hyperbolic).to(dtype_to_use)
        return (point_this_level_out_hyperbolic.to(dtype_to_use), tangent_out_for_aggregation, descriptor_point_out_for_transform_hyperbolic, boundary_points_this_level_hyperbolic, current_sigma_out_tensor.to(dtype_to_use))

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super().__init__(); self.input_dim, self.output_dim, self.config = input_dim, output_dim, config; self.num_levels = config.get("num_levels", 3); assert self.num_levels >= 0; self.hyperbolic_dims_list = config.get("hyperbolic_dims", []); self.initial_curvatures_list = config.get("initial_curvatures", []); self.dropout_val = config.get("dropout", 0.1); self.relative_vector_aggregation_mode = config.get("relative_vector_aggregation", "mean"); self.aggregation_method_mode = config.get("aggregation_method", "concat_tangent"); assert self.aggregation_method_mode == "concat_tangent"; self.use_rotation_in_transform_flag = config.get("use_rotation_in_transform", False); self.phi_influence_rotation_init = config.get("phi_influence_rotation_init", False)
        first_level_dim = self.hyperbolic_dims_list[0] if self.num_levels > 0 and self.hyperbolic_dims_list else 0
        if input_dim > 0 and first_level_dim > 0: self.input_tangent_projection = nn.Linear(input_dim, first_level_dim); self.input_tangent_layernorm = nn.LayerNorm(first_level_dim)
        else: self.input_tangent_projection = nn.Identity(); self.input_tangent_layernorm = nn.Identity()
        self.levels_modulelist = nn.ModuleList(); self.transforms_modulelist = nn.ModuleList()
        if self.num_levels > 0:
            for i in range(self.num_levels): self.levels_modulelist.append(HyperbolicWuBuNestingLevel(i, self.hyperbolic_dims_list[i], self.config, self.initial_curvatures_list[i]))
            num_transforms = max(0, self.num_levels - 1)
            if num_transforms > 0:
                transform_types_list = config.get("transform_types", ["linear"] * num_transforms); transform_hidden_dims_list = config.get("transform_hidden_dims", [None] * num_transforms)
                for i in range(num_transforms):
                    if i+1 < len(self.hyperbolic_dims_list) and i+1 < len(self.initial_curvatures_list): self.transforms_modulelist.append(HyperbolicInterLevelTransform(self.hyperbolic_dims_list[i], self.hyperbolic_dims_list[i+1], self.initial_curvatures_list[i], self.initial_curvatures_list[i+1], transform_types_list[i] if i < len(transform_types_list) else "linear", transform_hidden_dims_list[i] if i < len(transform_hidden_dims_list) else None, self.dropout_val, self.use_rotation_in_transform_flag, self.phi_influence_rotation_init, level_idx_for_phi=i))
                    else: logger.warning(f"Skipping transform {i} due to insufficient config for next level.")
        actual_output_dims_from_levels = [d for d_idx, d in enumerate(self.hyperbolic_dims_list[:self.num_levels]) if d > 0 and d_idx < len(self.levels_modulelist)]; aggregated_tangent_dim_val = sum(actual_output_dims_from_levels) if actual_output_dims_from_levels else input_dim
        self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, output_dim) if aggregated_tangent_dim_val > 0 and output_dim > 0 else nn.Identity()
        self.apply(init_weights_general); param_count = sum(p.numel() for p in self.parameters() if p.requires_grad); logger.info(f"FullyHypWuBuModel: {self.num_levels} levels. {param_count:,} params. Rot: {self.use_rotation_in_transform_flag}, PhiRotInit: {self.phi_influence_rotation_init}. InDim {input_dim}, AggTangentDim {aggregated_tangent_dim_val}, OutDim {output_dim}")
    def forward(self, x_initial_tangent_in: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.num_levels == 0: return self.output_tangent_projection(x_initial_tangent_in)
        if x_initial_tangent_in.dim() == 2: x_initial_tangent_in = x_initial_tangent_in.unsqueeze(1)
        B, S, _ = x_initial_tangent_in.shape; dev = x_initial_tangent_in.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else x_initial_tangent_in.dtype; x_initial_tangent_in = x_initial_tangent_in.to(dtype_to_use)
        current_tangent_from_projection = self.input_tangent_projection(x_initial_tangent_in); current_tangent_for_level0 = self.input_tangent_layernorm(current_tangent_from_projection)
        if not self.levels_modulelist: default_out_dim = self.output_dim if self.output_dim > 0 else (self.input_dim if self.input_dim > 0 else 1); return torch.zeros((B, S, default_out_dim), device=dev, dtype=dtype_to_use)
        level0_module = self.levels_modulelist[0]; c0_val = level0_module.get_current_curvature_scalar(); m0_obj = PoincareBall(c_scalar=c0_val)
        current_point_repr_hyperbolic = m0_obj.expmap0(current_tangent_for_level0) if self.hyperbolic_dims_list[0] > 0 else torch.empty(B, S, 0, device=dev, dtype=dtype_to_use)
        level_tangent_outputs_for_aggregation = []; aggregated_relative_vectors_from_prev_transform = None; descriptor_from_prev_transform_hyperbolic = None; sigma_from_prev_level_tensor = torch.full((B,), 0.0, device=dev, dtype=dtype_to_use)
        for i in range(self.num_levels):
            level_module = self.levels_modulelist[i]
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation, descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic, sigma_out_of_level_tensor) = level_module(current_point_repr_hyperbolic, aggregated_relative_vectors_from_prev_transform, descriptor_from_prev_transform_hyperbolic, sigma_from_prev_level_tensor)
            if self.hyperbolic_dims_list[i] > 0: level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)
            if i < self.num_levels - 1:
                if i >= len(self.transforms_modulelist): logger.warning(f"Missing transform for level {i} to {i+1}. Stop."); break
                transform_module = self.transforms_modulelist[i]; next_level_module = self.levels_modulelist[i+1]
                c_in_for_transform = level_module.get_current_curvature_scalar(); c_out_for_transform = next_level_module.get_current_curvature_scalar()
                (point_transformed_to_next_level_hyperbolic, boundaries_transformed_to_next_level_hyperbolic, descriptor_transformed_to_next_level_hyperbolic) = transform_module(point_out_of_level_hyperbolic, boundary_points_of_level_hyperbolic, descriptor_generated_by_level_hyperbolic, c_in_for_transform, c_out_for_transform)
                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic; descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic; sigma_from_prev_level_tensor = sigma_out_of_level_tensor.expand(B) if sigma_out_of_level_tensor.numel() == 1 else sigma_out_of_level_tensor
                aggregated_relative_vectors_from_prev_transform = None
                if boundaries_transformed_to_next_level_hyperbolic is not None and self.relative_vector_aggregation_mode not in ['none', None] and self.hyperbolic_dims_list[i+1] > 0:
                    manifold_next_level_obj = PoincareBall(c_scalar=c_out_for_transform); tan_main_next_level = manifold_next_level_obj.logmap0(current_point_repr_hyperbolic); tan_bounds_next_level = manifold_next_level_obj.logmap0(boundaries_transformed_to_next_level_hyperbolic)
                    if tan_bounds_next_level.dim() == 2: tan_bounds_next_level = tan_bounds_next_level.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
                    elif tan_bounds_next_level.dim() == 3 and tan_bounds_next_level.shape[0] != B : tan_bounds_next_level = tan_bounds_next_level.unsqueeze(1).expand(-1, S, -1, -1)
                    relative_tangent_vectors = tan_main_next_level.unsqueeze(2) - tan_bounds_next_level
                    agg_mode = self.relative_vector_aggregation_mode
                    if agg_mode == "mean": agg_rel_vec = torch.mean(relative_tangent_vectors, dim=2)
                    elif agg_mode == "sum": agg_rel_vec = torch.sum(relative_tangent_vectors, dim=2)
                    elif agg_mode == "max_norm": norms = torch.norm(relative_tangent_vectors, p=2, dim=-1); best_idx = torch.argmax(norms, dim=2, keepdim=True); best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, -1, relative_tangent_vectors.shape[-1]); agg_rel_vec = torch.gather(relative_tangent_vectors, 2, best_idx_expanded).squeeze(2)
                    else: agg_rel_vec = None
                    aggregated_relative_vectors_from_prev_transform = torch.zeros_like(tan_main_next_level) if agg_rel_vec is not None and not torch.isfinite(agg_rel_vec).all() else agg_rel_vec
        if not level_tangent_outputs_for_aggregation: default_out_dim = self.output_dim if self.output_dim > 0 else (self.input_dim if self.input_dim > 0 else 1); return torch.zeros((B, S, default_out_dim), device=dev, dtype=dtype_to_use)
        compatible_tangent_outputs = [t_val.to(dtype_to_use) for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation) if t_val is not None and t_idx < len(self.hyperbolic_dims_list) and self.hyperbolic_dims_list[t_idx] > 0 and torch.isfinite(t_val).all()]
        if not compatible_tangent_outputs: default_out_dim = self.output_dim if self.output_dim > 0 else (self.input_dim if self.input_dim > 0 else 1); return torch.zeros((B, S, default_out_dim), device=dev, dtype=dtype_to_use)
        aggregated_tangent_final = torch.cat(compatible_tangent_outputs, dim=-1); final_output = self.output_tangent_projection(aggregated_tangent_final)
        if padding_mask is not None: final_output = final_output.masked_fill(padding_mask.unsqueeze(-1).bool(), 0.0)
        return torch.nan_to_num(final_output, nan=0.0) if not torch.isfinite(final_output).all() else final_output

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
                    p.data = manifold.proju(p.data)
                    try: r_grad = manifold.egrad2rgrad(p.data, grad)
                    except Exception: self.grad_stats.params_skipped_due_non_finite_grad += 1; continue
                    if not torch.isfinite(r_grad).all(): self.grad_stats.params_skipped_due_non_finite_grad += 1; continue
                    update_vec = r_grad
                    if wd != 0:
                        try: log_p = manifold.logmap0(p.data); update_vec = update_vec.add(log_p, alpha=wd) if torch.isfinite(log_p).all() else update_vec
                        except Exception: pass
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(update_vec)); buf.mul_(mom).add_(update_vec)
                    if not torch.isfinite(buf).all(): buf.zero_()
                    try: p.data = manifold.proju(manifold.expmap(p.data, buf.mul(-lr)))
                    except Exception: state.get('momentum_buffer', torch.zeros(0)).zero_()
                    if not torch.isfinite(p.data).all(): p.data = manifold.proju(torch.nan_to_num(p.data, nan=0.0)); state.get('momentum_buffer', torch.zeros(0)).zero_()
                else:
                    d_p = grad.clone();
                    if wd != 0: d_p.add_(p.data, alpha=wd)
                    buf = state.setdefault('momentum_buffer', torch.zeros_like(p.data)); buf.mul_(mom).add_(d_p)
                    if not torch.isfinite(buf).all(): buf.zero_()
                    p.data.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all(): p.data = torch.nan_to_num(p.data, nan=0.0); state.get('momentum_buffer', torch.zeros(0)).zero_()
        self.grad_stats.finalize_step_stats(num_params_total); self._step_count += 1; return loss
    def get_q_controller_info(self) -> Dict: return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}
    def get_gradient_stats_summary(self) -> Dict: return self.grad_stats.get_step_summary_for_logging()

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

class GAADFrameProcessor(nn.Module):
    def __init__(self, num_total_regions: int, region_roi_output_size: Tuple[int,int], base_cnn_encoder_convs: nn.Module, base_cnn_out_channels: int, gaad_region_feature_dim: int, decomposition_type: str = "hybrid", min_size_px: int = 5):
        super().__init__(); self.num_total_regions=num_total_regions; self.region_roi_output_size=region_roi_output_size; self.base_cnn_encoder_convs = base_cnn_encoder_convs; self.decomposition_type=decomposition_type; self.base_cnn_out_channels = base_cnn_out_channels; self.min_size_px = min_size_px
        roi_flat_dim = base_cnn_out_channels * region_roi_output_size[0] * region_roi_output_size[1]; self.region_projector = nn.Sequential(nn.Linear(roi_flat_dim,gaad_region_feature_dim*2), nn.GELU(), nn.Linear(gaad_region_feature_dim*2,gaad_region_feature_dim)); self.apply(init_weights_general); logger.info(f"GAADFrameProcessor (Appearance): NumTotalRegions {num_total_regions}, Decomp '{decomposition_type}', RegionFeatDim {gaad_region_feature_dim}")
    def forward(self, frame_pixels_for_decomposition: torch.Tensor, features_to_roi_from: torch.Tensor) -> torch.Tensor:
        B, _, frame_h_int, frame_w_int = frame_pixels_for_decomposition.shape; dev = frame_pixels_for_decomposition.device; dtype = frame_pixels_for_decomposition.dtype
        map_h_feat_int, map_w_feat_int = features_to_roi_from.shape[2], features_to_roi_from.shape[3]; scale_h_map, scale_w_map = float(map_h_feat_int) / float(frame_h_int), float(map_w_feat_int) / float(frame_w_int)
        frame_w_tensor = torch.tensor(float(frame_w_int), dtype=dtype, device=dev); frame_h_tensor = torch.tensor(float(frame_h_int), dtype=dtype, device=dev); map_w_feat_tensor = torch.tensor(float(map_w_feat_int), dtype=dtype, device=dev); map_h_feat_tensor = torch.tensor(float(map_h_feat_int), dtype=dtype, device=dev)
        batch_region_features_list = []
        for b_idx in range(B):
            all_bboxes_for_frame_list = []
            if self.decomposition_type == "hybrid":
                num_subdivide = self.num_total_regions // 2; num_spiral = self.num_total_regions - num_subdivide
                if num_subdivide > 0: all_bboxes_for_frame_list.append(golden_subdivide_rect_fixed_n((frame_w_int,frame_h_int),num_subdivide,dev,dtype, self.min_size_px))
                if num_spiral > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n((frame_w_int,frame_h_int),num_spiral,dev,dtype); patch_base_size = min(frame_w_int, frame_h_int); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=dev, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs
                    spiral_bboxes_current[:,0]=torch.clamp(spiral_centers[:,0]-patch_ws, min=0.0, max=(frame_w_tensor-EPS).item()); spiral_bboxes_current[:,1]=torch.clamp(spiral_centers[:,1]-patch_hs, min=0.0, max=(frame_h_tensor-EPS).item()); min_x_tensor = torch.clamp(spiral_bboxes_current[:,0]+EPS, max=frame_w_tensor); min_y_tensor = torch.clamp(spiral_bboxes_current[:,1]+EPS, max=frame_h_tensor); spiral_bboxes_current[:,2]=torch.clamp(spiral_centers[:,0]+patch_ws, min=min_x_tensor, max=frame_w_tensor); spiral_bboxes_current[:,3]=torch.clamp(spiral_centers[:,1]+patch_hs, min=min_y_tensor, max=frame_h_tensor); all_bboxes_for_frame_list.append(spiral_bboxes_current)
            elif self.decomposition_type == "spiral":
                if self.num_total_regions > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n((frame_w_int,frame_h_int),self.num_total_regions,dev,dtype); patch_base_size = min(frame_w_int, frame_h_int); spiral_bboxes_current = torch.zeros(self.num_total_regions, 4, device=dev, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs
                    spiral_bboxes_current[:,0]=torch.clamp(spiral_centers[:,0]-patch_ws, min=0.0, max=(frame_w_tensor-EPS).item()); spiral_bboxes_current[:,1]=torch.clamp(spiral_centers[:,1]-patch_hs, min=0.0, max=(frame_h_tensor-EPS).item()); min_x_tensor = torch.clamp(spiral_bboxes_current[:,0]+EPS, max=frame_w_tensor); min_y_tensor = torch.clamp(spiral_bboxes_current[:,1]+EPS, max=frame_h_tensor); spiral_bboxes_current[:,2]=torch.clamp(spiral_centers[:,0]+patch_ws, min=min_x_tensor, max=frame_w_tensor); spiral_bboxes_current[:,3]=torch.clamp(spiral_centers[:,1]+patch_hs, min=min_y_tensor, max=frame_h_tensor); all_bboxes_for_frame_list.append(spiral_bboxes_current)
            else: # subdivide
                if self.num_total_regions > 0: all_bboxes_for_frame_list.append(golden_subdivide_rect_fixed_n((frame_w_int,frame_h_int),self.num_total_regions,dev,dtype, self.min_size_px))
            if not all_bboxes_for_frame_list: final_bboxes_for_frame = torch.tensor([[0,0,frame_w_int,frame_h_int]] * self.num_total_regions, dtype=dtype, device=dev)
            else: final_bboxes_for_frame = torch.cat(all_bboxes_for_frame_list, dim=0)
            if final_bboxes_for_frame.shape[0] < self.num_total_regions: padding_count = self.num_total_regions - final_bboxes_for_frame.shape[0]; padding_box_coords = [0,0,frame_w_int,frame_h_int]; padding = final_bboxes_for_frame[-1:].repeat(padding_count, 1) if final_bboxes_for_frame.shape[0] > 0 else torch.tensor([padding_box_coords] * padding_count, dtype=dtype, device=dev); final_bboxes_for_frame = torch.cat([final_bboxes_for_frame, padding], dim=0)
            elif final_bboxes_for_frame.shape[0] > self.num_total_regions: final_bboxes_for_frame = final_bboxes_for_frame[:self.num_total_regions]
            scaled_bboxes_for_roi = torch.zeros_like(final_bboxes_for_frame)
            scaled_bboxes_for_roi[:,0] = torch.clamp(final_bboxes_for_frame[:,0] * scale_w_map, min=0.0, max=(map_w_feat_tensor - EPS).item()); # CORRECTED
            scaled_bboxes_for_roi[:,1] = torch.clamp(final_bboxes_for_frame[:,1] * scale_h_map, min=0.0, max=(map_h_feat_tensor - EPS).item()); # CORRECTED
            min_x2_tensor_roi = torch.clamp(scaled_bboxes_for_roi[:,0] + EPS, max=map_w_feat_tensor);
            min_y2_tensor_roi = torch.clamp(scaled_bboxes_for_roi[:,1] + EPS, max=map_h_feat_tensor);
            scaled_bboxes_for_roi[:,2] = torch.clamp(final_bboxes_for_frame[:,2] * scale_w_map, min=min_x2_tensor_roi, max=map_w_feat_tensor);
            scaled_bboxes_for_roi[:,3] = torch.clamp(final_bboxes_for_frame[:,3] * scale_h_map, min=min_y2_tensor_roi, max=map_h_feat_tensor);
            rois_with_batch_idx = torch.cat([torch.full((scaled_bboxes_for_roi.shape[0],1), 0, device=dev, dtype=dtype), scaled_bboxes_for_roi], dim=1)
            aligned_regions = roi_align(features_to_roi_from[b_idx].unsqueeze(0), rois_with_batch_idx, self.region_roi_output_size, spatial_scale=1.0, aligned=True)
            aligned_regions_flat = aligned_regions.view(self.num_total_regions, -1); batch_region_features_list.append(self.region_projector(aligned_regions_flat))
        return torch.stack(batch_region_features_list)

class GAADMotionRegionProposal(nn.Module):
    def __init__(self, num_total_regions: int, decomposition_type: str = "hybrid", min_size_px: int = 5):
        super().__init__(); self.num_total_regions = num_total_regions; self.decomposition_type = decomposition_type; self.min_size_px = min_size_px; logger.info(f"GAADMotionRegionProposal: NumTotalRegions {num_total_regions}, Decomp '{decomposition_type}'")
    def forward(self, map_for_coords: torch.Tensor) -> torch.Tensor:
        B, _, frame_h_int, frame_w_int = map_for_coords.shape; dev = map_for_coords.device; dtype = map_for_coords.dtype; batch_bboxes_list = []
        frame_w_tensor = torch.tensor(float(frame_w_int), dtype=dtype, device=dev); frame_h_tensor = torch.tensor(float(frame_h_int), dtype=dtype, device=dev)
        for b_idx in range(B):
            all_bboxes_for_frame_list = []
            if self.decomposition_type == "hybrid":
                num_subdivide = self.num_total_regions // 2; num_spiral = self.num_total_regions - num_subdivide
                if num_subdivide > 0: subdiv_bboxes = golden_subdivide_rect_fixed_n((frame_w_int,frame_h_int),num_subdivide,dev,dtype, self.min_size_px); all_bboxes_for_frame_list.append(subdiv_bboxes)
                if num_spiral > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n((frame_w_int,frame_h_int),num_spiral,dev,dtype); patch_base_size = min(frame_w_int, frame_h_int); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=dev, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs
                    spiral_bboxes_current[:,0]=torch.clamp(spiral_centers[:,0]-patch_ws, min=0.0, max=(frame_w_tensor-EPS).item()); # CORRECTED
                    spiral_bboxes_current[:,1]=torch.clamp(spiral_centers[:,1]-patch_hs, min=0.0, max=(frame_h_tensor-EPS).item()); # CORRECTED
                    min_x_tensor = torch.clamp(spiral_bboxes_current[:,0]+EPS, max=frame_w_tensor);
                    min_y_tensor = torch.clamp(spiral_bboxes_current[:,1]+EPS, max=frame_h_tensor);
                    spiral_bboxes_current[:,2]=torch.clamp(spiral_centers[:,0]+patch_ws, min=min_x_tensor, max=frame_w_tensor);
                    spiral_bboxes_current[:,3]=torch.clamp(spiral_centers[:,1]+patch_hs, min=min_y_tensor, max=frame_h_tensor);
                    all_bboxes_for_frame_list.append(spiral_bboxes_current)
            elif self.decomposition_type == "spiral":
                 if self.num_total_regions > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n((frame_w_int,frame_h_int),self.num_total_regions,dev,dtype); patch_base_size = min(frame_w_int, frame_h_int); spiral_bboxes_current = torch.zeros(self.num_total_regions, 4, device=dev, dtype=dtype); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs
                    spiral_bboxes_current[:,0]=torch.clamp(spiral_centers[:,0]-patch_ws, min=0.0, max=(frame_w_tensor-EPS).item()); # CORRECTED
                    spiral_bboxes_current[:,1]=torch.clamp(spiral_centers[:,1]-patch_hs, min=0.0, max=(frame_h_tensor-EPS).item()); # CORRECTED
                    min_x_tensor = torch.clamp(spiral_bboxes_current[:,0]+EPS, max=frame_w_tensor);
                    min_y_tensor = torch.clamp(spiral_bboxes_current[:,1]+EPS, max=frame_h_tensor);
                    spiral_bboxes_current[:,2]=torch.clamp(spiral_centers[:,0]+patch_ws, min=min_x_tensor, max=frame_w_tensor);
                    spiral_bboxes_current[:,3]=torch.clamp(spiral_centers[:,1]+patch_hs, min=min_y_tensor, max=frame_h_tensor);
                    all_bboxes_for_frame_list.append(spiral_bboxes_current)
            else: # subdivide
                if self.num_total_regions > 0: subdiv_bboxes = golden_subdivide_rect_fixed_n((frame_w_int,frame_h_int),self.num_total_regions,dev,dtype,self.min_size_px); all_bboxes_for_frame_list.append(subdiv_bboxes)
            if not all_bboxes_for_frame_list: final_bboxes_for_frame = torch.tensor([[0,0,frame_w_int,frame_h_int]] * self.num_total_regions, dtype=dtype, device=dev)
            else: final_bboxes_for_frame = torch.cat(all_bboxes_for_frame_list, dim=0)
            if final_bboxes_for_frame.shape[0] < self.num_total_regions: padding_count = self.num_total_regions - final_bboxes_for_frame.shape[0]; padding_box_coords = [0,0,frame_w_int,frame_h_int]; padding = final_bboxes_for_frame[-1:].repeat(padding_count, 1) if final_bboxes_for_frame.shape[0] > 0 else torch.tensor([padding_box_coords] * padding_count, dtype=dtype, device=dev); final_bboxes_for_frame = torch.cat([final_bboxes_for_frame, padding], dim=0)
            elif final_bboxes_for_frame.shape[0] > self.num_total_regions: final_bboxes_for_frame = final_bboxes_for_frame[:self.num_total_regions]
            batch_bboxes_list.append(final_bboxes_for_frame)
        return torch.stack(batch_bboxes_list)

# =====================================================================
# Diffusion Model Specific Components
# =====================================================================
class InitialFrameAutoencoderCNN(nn.Module):
    def __init__(self, image_channels: int, feature_dim: int, image_size: Tuple[int, int]):
        super().__init__(); self.image_channels=image_channels; self.feature_dim=feature_dim; self.image_h,self.image_w=image_size
        self.encoder_convs = nn.Sequential(nn.Conv2d(image_channels,32,4,2,1),nn.GroupNorm(8,32),nn.GELU(), nn.Conv2d(32,64,4,2,1),nn.GroupNorm(16,64),nn.GELU(), nn.Conv2d(64,128,4,2,1),nn.GroupNorm(32,128),nn.GELU(), nn.Conv2d(128,256,4,2,1),nn.GroupNorm(64,256),nn.GELU())
        with torch.no_grad(): self.conv_out_shape=self.encoder_convs(torch.randn(1,image_channels,self.image_h,self.image_w)).shape
        self.flattened_dim=self.conv_out_shape[1]*self.conv_out_shape[2]*self.conv_out_shape[3]; self.encoder_fc = nn.Linear(self.flattened_dim,feature_dim); self.decoder_fc = nn.Linear(feature_dim,self.flattened_dim)
        self.decoder_convs = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1),nn.GroupNorm(32,128),nn.GELU(), nn.ConvTranspose2d(128,64,4,2,1),nn.GroupNorm(16,64),nn.GELU(), nn.ConvTranspose2d(64,32,4,2,1),nn.GroupNorm(8,32),nn.GELU(), nn.ConvTranspose2d(32,image_channels,4,2,1),nn.Tanh())
        self.apply(init_weights_general)
    def encode(self, x_frames: torch.Tensor) -> torch.Tensor:
        orig_dim=x_frames.dim(); is_batched_seq = orig_dim==5; B_orig, N_orig = x_frames.shape[0], x_frames.shape[1] if is_batched_seq else 1
        x_frames_flat = x_frames.reshape(B_orig*N_orig, self.image_channels, self.image_h, self.image_w) if is_batched_seq else x_frames
        feats_conv=self.encoder_convs(x_frames_flat); feats_flat_for_fc=feats_conv.view(feats_conv.size(0),-1); vec=self.encoder_fc(feats_flat_for_fc)
        return vec.view(B_orig,N_orig,-1) if is_batched_seq else vec
    def encode_conv_features(self, x_frames: torch.Tensor) -> torch.Tensor: return self.encoder_convs(x_frames)
    def decode(self, features_vec: torch.Tensor) -> torch.Tensor:
        orig_dim=features_vec.dim(); is_batched_seq = orig_dim==3; B_orig, N_orig = features_vec.shape[0], features_vec.shape[1] if is_batched_seq else 1
        features_vec_flat = features_vec.reshape(B_orig*N_orig, self.feature_dim) if is_batched_seq else features_vec
        x_fc_out =self.decoder_fc(features_vec_flat); x_unflat = x_fc_out.view(-1,self.conv_out_shape[1],self.conv_out_shape[2],self.conv_out_shape[3]);
        pixels_out=self.decoder_convs(x_unflat)

        # Add interpolation to resize to target image size before final reshape
        # Check if spatial dimensions mismatch before resizing
        # Access image size using stored self.image_h and self.image_w
        target_image_size = (self.image_h, self.image_w) # Corrected line: Use individual attributes to form the tuple
        if pixels_out.shape[2:] != target_image_size: # Compare tuple slices
             # Log a warning if resizing is needed
             logger.warning(f"Decoder output shape mismatch: {pixels_out.shape[2:]} vs target {target_image_size}. Resizing using interpolation.")
             pixels_out = F.interpolate(pixels_out, size=target_image_size, mode='bilinear', align_corners=False)


        return pixels_out.view(B_orig,N_orig,self.image_channels,self.image_h,self.image_w) if is_batched_seq else pixels_out
class SinusoidalPhiEmbedding(nn.Module):
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

class WuBuSTDiffusionNet(nn.Module):
    def __init__(self, wubu_s_config: Dict, wubu_t_config: Dict, wubu_m_config: Optional[Dict], video_config: Dict, gaad_appearance_config: Dict, gaad_motion_config: Optional[Dict], time_embedding_dim: int, args: argparse.Namespace):
        super().__init__(); self.video_config = video_config; self.gaad_appearance_config = gaad_appearance_config; self.gaad_motion_config = gaad_motion_config; self.wubu_m_config = wubu_m_config; self.args = args
        self.initial_global_feature_dim = video_config["initial_cnn_feature_dim"]; self.gaad_app_region_feature_dim = gaad_appearance_config["gaad_region_feature_dim"]; self.wubu_s_output_dim = video_config["wubu_s_output_dim"]; self.wubu_m_output_dim = video_config.get("wubu_m_output_dim", 0) if args.use_wubu_motion_branch and wubu_m_config else 0; self.wubu_t_output_dim = video_config["wubu_t_output_dim"]
        self.frame_autoencoder = InitialFrameAutoencoderCNN(video_config["num_channels"], self.initial_global_feature_dim, video_config["image_size"])
        self.gaad_processor_appearance = GAADFrameProcessor(gaad_appearance_config["num_regions"], gaad_appearance_config["region_roi_output_size"], self.frame_autoencoder.encoder_convs, self.frame_autoencoder.conv_out_shape[1], self.gaad_app_region_feature_dim, gaad_appearance_config["decomposition_type"])
        self.wubu_s = FullyHyperbolicWuBuNestingModel(self.gaad_app_region_feature_dim, self.wubu_s_output_dim, wubu_s_config)
        self.gaad_motion_region_proposal: Optional[GAADMotionRegionProposal] = None; self.motion_feature_projector: Optional[nn.Module] = None; self.wubu_m: Optional[FullyHyperbolicWuBuNestingModel] = None; self.wubu_m_input_dim_actual = 0
        if self.args.use_wubu_motion_branch and self.wubu_m_config and self.gaad_motion_config:
            self.gaad_motion_region_proposal = GAADMotionRegionProposal(self.gaad_motion_config["num_regions"], self.gaad_motion_config["decomposition_type"])
            self.motion_roi_output_size = (args.gaad_motion_region_roi_output_h, args.gaad_motion_region_roi_output_w); single_motion_roi_feat_dim_flat = self.frame_autoencoder.conv_out_shape[1] * self.motion_roi_output_size[0] * self.motion_roi_output_size[1]; concatenated_motion_roi_feat_dim_for_projector = 2 * single_motion_roi_feat_dim_flat; self.wubu_m_input_dim_actual = args.motion_feature_dim_for_wubum_input
            if concatenated_motion_roi_feat_dim_for_projector != self.wubu_m_input_dim_actual: logger.info(f"Motion branch: Raw concat RoI dim ({concatenated_motion_roi_feat_dim_for_projector}) != WuBu-M target input dim ({self.wubu_m_input_dim_actual}). Adding projector."); self.motion_feature_projector = nn.Linear(concatenated_motion_roi_feat_dim_for_projector, self.wubu_m_input_dim_actual)
            else: self.motion_feature_projector = nn.Identity(); logger.info(f"Motion branch: Raw concat RoI dim ({concatenated_motion_roi_feat_dim_for_projector}) matches WuBu-M target input dim. Using Identity projector.")
            self.wubu_m = FullyHyperbolicWuBuNestingModel(self.wubu_m_input_dim_actual, self.wubu_m_output_dim, self.wubu_m_config)
        else: self.gaad_motion_region_proposal = None; self.motion_feature_projector = None; self.wubu_m = None
        wubu_t_actual_input_dim = self.wubu_s_output_dim;
        if self.args.use_wubu_motion_branch and self.wubu_m: wubu_t_actual_input_dim += self.wubu_m_output_dim
        self.wubu_t = FullyHyperbolicWuBuNestingModel(wubu_t_actual_input_dim, self.wubu_t_output_dim, wubu_t_config)
        self.time_sin_embedding = SinusoidalPhiEmbedding(time_embedding_dim, base_freq_phi_scaled=gaad_appearance_config.get("phi_time_base_freq", 10000.0), use_phi_paper_scaling_arg=self.args.use_phi_frequency_scaling_for_time_emb, phi_constant=PHI)
        self.time_fc_mlp = nn.Sequential(nn.Linear(time_embedding_dim, time_embedding_dim * 2), nn.GELU(), nn.Linear(time_embedding_dim * 2, time_embedding_dim))
        head_input_dim = self.initial_global_feature_dim + self.wubu_t_output_dim + time_embedding_dim; self.noise_pred_head = nn.Sequential(nn.Linear(head_input_dim, head_input_dim * 2), nn.GELU(), nn.Linear(head_input_dim * 2, self.initial_global_feature_dim))
        self.apply(init_weights_general); param_count = sum(p.numel() for p in self.parameters() if p.requires_grad); motion_branch_active_str = "ACTIVE" if self.args.use_wubu_motion_branch and self.wubu_m else "INACTIVE"; logger.info(f"WuBuSTDiffNet+GAAD+Motion(WuBuM {motion_branch_active_str}) ({param_count:,} params): GlobalFeat {self.initial_global_feature_dim}, GAADApp.Feat {self.gaad_app_region_feature_dim}, WuBuS_Out {self.wubu_s_output_dim}, WuBuM_In {self.wubu_m_input_dim_actual if self.wubu_m else 0} WuBuM_Out {self.wubu_m_output_dim}, WuBuT_In {wubu_t_actual_input_dim}, WuBuT_Out {self.wubu_t_output_dim}")
    def forward(self, xt_target_global_features: torch.Tensor, conditioning_frames_pixels: Optional[torch.Tensor], time_t_integers: torch.Tensor, cfg_unconditional_flag: bool = False) -> torch.Tensor:
        B = xt_target_global_features.shape[0]; dev = xt_target_global_features.device; dtype_to_use = next(self.parameters()).dtype
        s_sequence_for_wubu_t = torch.empty(B, 0, self.wubu_s_output_dim, device=dev, dtype=dtype_to_use); m_sequence_for_wubu_t: Optional[torch.Tensor] = None
        if self.args.use_wubu_motion_branch and self.wubu_m: m_sequence_for_wubu_t = torch.empty(B, 0, self.wubu_m_output_dim, device=dev, dtype=dtype_to_use)
        s_t_list = []; m_t_list: Optional[List[torch.Tensor]] = [] if self.args.use_wubu_motion_branch and self.wubu_m else None
        if not cfg_unconditional_flag and conditioning_frames_pixels is not None and conditioning_frames_pixels.shape[1] > 0:
            num_cond_frames = conditioning_frames_pixels.shape[1]; _, _, c_img_int, h_img_int, w_img_int = conditioning_frames_pixels.shape; cond_pixels_flat = conditioning_frames_pixels.reshape(B * num_cond_frames, c_img_int, h_img_int, w_img_int).to(dtype=dtype_to_use)
            M_all_cond_frames_conv_features = self.frame_autoencoder.encode_conv_features(cond_pixels_flat); _, C_ae_feat_int, H_ae_feat_int, W_ae_feat_int = M_all_cond_frames_conv_features.shape; M_all_cond_frames_conv_features = M_all_cond_frames_conv_features.view(B, num_cond_frames, C_ae_feat_int, H_ae_feat_int, W_ae_feat_int)
            feat_W_tensor = torch.tensor(float(W_ae_feat_int), device=dev, dtype=torch.float); feat_H_tensor = torch.tensor(float(H_ae_feat_int), device=dev, dtype=torch.float)
            for i in range(num_cond_frames):
                current_frame_pixels_for_decomp = conditioning_frames_pixels[:, i, ...].to(dtype=dtype_to_use); current_M_conv_features = M_all_cond_frames_conv_features[:, i, ...]
                gaad_appearance_regions_features = self.gaad_processor_appearance(current_frame_pixels_for_decomp, current_M_conv_features); s_t_i_from_wubu_s = self.wubu_s(gaad_appearance_regions_features)
                s_t_i_aggregated = torch.max(s_t_i_from_wubu_s, dim=1)[0] if s_t_i_from_wubu_s.numel() > 0 and s_t_i_from_wubu_s.shape[1] > 0 else torch.zeros(B, self.wubu_s_output_dim, device=dev, dtype=dtype_to_use); s_t_list.append(s_t_i_aggregated)
                if self.args.use_wubu_motion_branch and self.wubu_m and self.gaad_motion_region_proposal and self.motion_feature_projector and self.gaad_motion_config and m_t_list is not None:
                    if i > 0:
                        prev_frame_pixels = conditioning_frames_pixels[:, i-1, ...].to(dtype=dtype_to_use); prev_M_conv_features = M_all_cond_frames_conv_features[:, i-1, ...]; diff_map = torch.abs(current_frame_pixels_for_decomp - prev_frame_pixels)
                        if diff_map.shape[1] > 1 and self.args.diff_map_channels == 1: diff_map = torch.mean(diff_map, dim=1, keepdim=True)
                        motion_region_bboxes_orig_coords = self.gaad_motion_region_proposal(diff_map); scale_h_roi = float(H_ae_feat_int) / float(h_img_int); scale_w_roi = float(W_ae_feat_int) / float(w_img_int); motion_feats_for_wubum_this_frame_batch = []
                        for b_single_idx in range(B):
                            current_bboxes_single_item = motion_region_bboxes_orig_coords[b_single_idx]; scaled_motion_bboxes = current_bboxes_single_item.clone().to(torch.float); scaled_motion_bboxes[:, 0::2] *= scale_w_roi; scaled_motion_bboxes[:, 1::2] *= scale_h_roi
                            scaled_motion_bboxes[:,0]=torch.clamp(scaled_motion_bboxes[:,0], min=0.0, max=(feat_W_tensor-EPS).item());
                            scaled_motion_bboxes[:,1]=torch.clamp(scaled_motion_bboxes[:,1], min=0.0, max=(feat_H_tensor-EPS).item());
                            min_x_roi_clamped = torch.clamp(scaled_motion_bboxes[:,0]+EPS, max=(feat_W_tensor).item());
                            min_y_roi_clamped = torch.clamp(scaled_motion_bboxes[:,1]+EPS, max=(feat_H_tensor).item());
                            scaled_motion_bboxes[:,2]=torch.clamp(scaled_motion_bboxes[:,2], min=min_x_roi_clamped, max=feat_W_tensor); # CORRECTED
                            scaled_motion_bboxes[:,3]=torch.clamp(scaled_motion_bboxes[:,3], min=min_y_roi_clamped, max=feat_H_tensor); # CORRECTED
                            rois_for_align = torch.cat([torch.full((scaled_motion_bboxes.shape[0],1), 0, device=dev, dtype=dtype_to_use), scaled_motion_bboxes], dim=1)
                            aligned_motion_current = roi_align(current_M_conv_features[b_single_idx].unsqueeze(0), rois_for_align, self.motion_roi_output_size, spatial_scale=1.0, aligned=True)
                            aligned_motion_prev = roi_align(prev_M_conv_features[b_single_idx].unsqueeze(0), rois_for_align, self.motion_roi_output_size, spatial_scale=1.0, aligned=True)
                            num_actual_motion_regions = aligned_motion_current.shape[0]; concatenated_motion_feats = torch.cat([aligned_motion_current.view(num_actual_motion_regions, -1), aligned_motion_prev.view(num_actual_motion_regions, -1)], dim=-1); projected_motion_feats = self.motion_feature_projector(concatenated_motion_feats); motion_feats_for_wubum_this_frame_batch.append(projected_motion_feats)
                        stacked_motion_feats_for_wubum = torch.stack(motion_feats_for_wubum_this_frame_batch, dim=0); m_t_i_from_wubu_m = self.wubu_m(stacked_motion_feats_for_wubum)
                        m_t_i_aggregated = torch.max(m_t_i_from_wubu_m, dim=1)[0] if m_t_i_from_wubu_m.numel() > 0 and m_t_i_from_wubu_m.shape[1] > 0 else torch.zeros(B, self.wubu_m_output_dim, device=dev, dtype=dtype_to_use); m_t_list.append(m_t_i_aggregated)
                    elif m_t_list is not None: m_t_list.append(torch.zeros(B, self.wubu_m_output_dim, device=dev, dtype=dtype_to_use))
            if s_t_list: s_sequence_for_wubu_t = torch.stack(s_t_list, dim=1)
            if m_t_list and self.args.use_wubu_motion_branch and self.wubu_m:
                if len(m_t_list) == num_cond_frames: m_sequence_for_wubu_t = torch.stack(m_t_list, dim=1)
                else: logger.warning(f"Motion sequence length {len(m_t_list)} unexpected. Re-init m_sequence to empty."); m_sequence_for_wubu_t = torch.empty(B, 0, self.wubu_m_output_dim, device=dev, dtype=dtype_to_use) if m_sequence_for_wubu_t is not None else None
        if self.args.use_wubu_motion_branch and self.wubu_m and m_sequence_for_wubu_t is not None and m_sequence_for_wubu_t.shape[1] > 0:
            if s_sequence_for_wubu_t.shape[1] == m_sequence_for_wubu_t.shape[1] and s_sequence_for_wubu_t.shape[1] > 0: combined_sequence_for_wubu_t = torch.cat([s_sequence_for_wubu_t, m_sequence_for_wubu_t], dim=-1)
            elif s_sequence_for_wubu_t.shape[1] > 0 : logger.warning("Using only S-seq for WuBu-T (M-seq issue)."); combined_sequence_for_wubu_t = s_sequence_for_wubu_t
            elif m_sequence_for_wubu_t.shape[1] > 0 : logger.warning("S-seq empty, M-seq has data. Using only M-seq for WuBu-T (with S-zeros)."); s_zeros = torch.zeros(m_sequence_for_wubu_t.shape[0], m_sequence_for_wubu_t.shape[1], self.wubu_s_output_dim, device=dev, dtype=dtype_to_use); combined_sequence_for_wubu_t = torch.cat([s_zeros, m_sequence_for_wubu_t], dim=-1)
            else: combined_sequence_for_wubu_t = s_sequence_for_wubu_t # Both empty
        else: combined_sequence_for_wubu_t = s_sequence_for_wubu_t
        temporal_context_ctx = torch.zeros(B, self.wubu_t_output_dim, device=dev, dtype=dtype_to_use)
        if combined_sequence_for_wubu_t.shape[1] > 0: temporal_context_ctx_full = self.wubu_t(combined_sequence_for_wubu_t); temporal_context_ctx = temporal_context_ctx_full[:, -1, :] if temporal_context_ctx_full.numel() > 0 and temporal_context_ctx_full.shape[1] > 0 else temporal_context_ctx
        time_sin_emb_output = self.time_sin_embedding(time_t_integers, phi_time_scale=self.gaad_appearance_config.get("phi_time_diffusion_scale", 1.0)); time_emb = self.time_fc_mlp(time_sin_emb_output).to(dtype=dtype_to_use)
        xt_target_global_features_expanded = xt_target_global_features.unsqueeze(1) if xt_target_global_features.dim() == 2 else xt_target_global_features
        num_pred_frames_actual = xt_target_global_features_expanded.shape[1]; temporal_context_ctx_expanded = temporal_context_ctx.unsqueeze(1).expand(-1, num_pred_frames_actual, -1); time_emb_expanded = time_emb.unsqueeze(1).expand(-1, num_pred_frames_actual, -1)
        combined_features_for_head = torch.cat([xt_target_global_features_expanded.to(dtype=dtype_to_use), temporal_context_ctx_expanded.to(dtype=dtype_to_use), time_emb_expanded.to(dtype=dtype_to_use)], dim=-1)
        predicted_noise_n_dim = self.noise_pred_head(combined_features_for_head)
        return torch.nan_to_num(predicted_noise_n_dim, nan=0.0) if not torch.isfinite(predicted_noise_n_dim).all() else predicted_noise_n_dim

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02): return torch.linspace(beta_start, beta_end, timesteps)
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1; x = torch.linspace(0, timesteps, steps); alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]; betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]); return torch.clip(betas, 0.0001, 0.9999)
def q_sample(x_start_features: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    if noise is None: noise = torch.randn_like(x_start_features)
    reshape_dims = [-1] + [1]*(x_start_features.dim()-1)
    sqrt_alpha_t = sqrt_alphas_cumprod.gather(0,t).view(*reshape_dims).to(x_start_features.device, x_start_features.dtype)
    sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod.gather(0,t).view(*reshape_dims).to(x_start_features.device, x_start_features.dtype)
    return sqrt_alpha_t*x_start_features + sqrt_one_minus_alpha_t*noise

class VideoFrameDataset(Dataset):
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int], frame_skip: int = 1, data_fraction: float = 1.0):
        super().__init__(); self.video_path = video_path; self.num_frames_total = num_frames_total; self.image_size = image_size; self.frame_skip = frame_skip
        if not os.path.isfile(self.video_path): logger.error(f"Video file not found: {self.video_path}"); raise FileNotFoundError(f"Video file not found: {self.video_path}")
        logger.info(f"Attempting to load entire video into RAM: {self.video_path}")
        if not VIDEO_IO_AVAILABLE: logger.error("torchvision.io.read_video is not available."); raise RuntimeError("torchvision.io.read_video is not available.")
        try: video_data = video_io.read_video(self.video_path, output_format="TCHW", pts_unit="sec"); self.video_frames_in_ram = video_data[0].contiguous(); self.source_video_fps = video_data[2].get('video_fps', 30.0); ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3); logger.info(f"Loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. Est RAM: {ram_usage_gb:.2f} GB.");
        except Exception as e: logger.error(f"Failed to load video '{self.video_path}' into RAM: {e}", exc_info=True); raise RuntimeError(f"Failed to load video '{self.video_path}' into RAM.") from e
        self.transform = T.Compose([T.Resize(self.image_size, antialias=True), T.Lambda(lambda x: x / 255.0), T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
        self.num_disk_frames = self.video_frames_in_ram.shape[0]; self.samples = []
        required_span_len = (self.num_frames_total - 1) * self.frame_skip + 1
        if self.num_disk_frames >= required_span_len:
            for i in range(self.num_disk_frames - required_span_len + 1): self.samples.append(i)
        else: logger.warning(f"Not enough frames ({self.num_disk_frames}) in video '{self.video_path}' for span {required_span_len}.")
        if data_fraction < 1.0 and len(self.samples) > 1: num_to_keep = max(1, int(len(self.samples) * data_fraction)); self.samples = random.sample(self.samples, num_to_keep); logger.info(f"Using {data_fraction*100:.2f}% of samples: {len(self.samples)} samples.")
        if not self.samples: logger.error(f"VideoFrameDataset: No valid samples. Frames: {self.num_disk_frames}, Total: {self.num_frames_total}, Skip: {self.frame_skip}.")
        else: logger.info(f"VideoFrameDataset initialized (RAM). Frames: {self.num_disk_frames}. Samples: {len(self.samples)}. Sample len: {self.num_frames_total} (skip {self.frame_skip}).")
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> torch.Tensor:
        start_frame_idx_in_ram = self.samples[idx]; frames_for_sample = []
        for i in range(self.num_frames_total):
            actual_frame_idx_in_ram = start_frame_idx_in_ram + i * self.frame_skip
            if actual_frame_idx_in_ram < self.num_disk_frames:
                try: frame_tensor_chw_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram]; transformed_frame = self.transform(frame_tensor_chw_uint8); frames_for_sample.append(transformed_frame)
                except Exception as e: logger.error(f"Error transforming frame {actual_frame_idx_in_ram} for sample {idx}: {e}", exc_info=True); raise e
            else: logger.error(f"Frame index {actual_frame_idx_in_ram} out of bounds (total: {self.num_disk_frames}). Sample: {idx}"); raise IndexError("Frame index out of bounds.")
        if len(frames_for_sample) != self.num_frames_total: logger.error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total} for sample {idx}"); raise ValueError("Incorrect number of frames loaded.")
        return torch.stack(frames_for_sample)

class DiffusionTrainer:
    def __init__(self, model: WuBuSTDiffusionNet, optimizer: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], args: argparse.Namespace, rank: int, world_size: int, ddp_active: bool, video_config: Dict, gaad_appearance_config: Dict, gaad_motion_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        self.model = model; self.optimizer = optimizer; self.device = device; self.train_loader = train_loader; self.val_loader = val_loader; self.args = args; self.rank = rank; self.world_size = world_size; self.ddp_active = ddp_active; self.am_main_process = (rank == 0); self.video_config = video_config; self.gaad_appearance_config = gaad_appearance_config; self.gaad_motion_config = gaad_motion_config; self.wubu_m_config = wubu_m_config
        self.timesteps = args.timesteps; self.betas = (linear_beta_schedule(args.timesteps,args.beta_start,args.beta_end) if args.beta_schedule=='linear' else cosine_beta_schedule(args.timesteps,args.cosine_s)).to(device); self.alphas = 1. - self.betas; self.alphas_cumprod = torch.cumprod(self.alphas,axis=0); self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod); self.sqrt_recip_alphas = torch.sqrt(1.0/(self.alphas+EPS)); self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0); self.posterior_variance = torch.clamp(self.betas*(1.-self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS),min=EPS*10); self.posterior_log_variance_clipped = torch.log(self.posterior_variance); self.posterior_mean_coef1 = self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.-self.alphas_cumprod+EPS); self.posterior_mean_coef2 = (1.-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.-self.alphas_cumprod+EPS)
        self.scaler = amp.GradScaler(enabled=args.use_amp and device.type=='cuda'); self.global_step=0; self.current_epoch=0; self.best_val_loss=float('inf'); self.last_val_metrics:Dict[str,Any]={};
        if self.am_main_process: os.makedirs(args.checkpoint_dir,exist_ok=True)
        m = self.model.module if ddp_active and isinstance(self.model, DDP) else self.model; self.frame_autoencoder = m.frame_autoencoder
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
    def _get_x0_target_global_features(self, target_pixels: torch.Tensor) -> torch.Tensor: return self.frame_autoencoder.encode(target_pixels)
    def train_step(self, batch_video_frames: torch.Tensor):
        num_cond = self.video_config["num_input_frames"]; B = batch_video_frames.shape[0]; is_this_batch_unconditional = False
        if self.args.cfg_unconditional_dropout_prob > 0 and torch.rand(1).item() < self.args.cfg_unconditional_dropout_prob: is_this_batch_unconditional = True
        actual_cond_pixels_for_model = batch_video_frames[:, :num_cond, ...].to(self.device) if not is_this_batch_unconditional else None
        target_pixels = batch_video_frames[:, num_cond : num_cond + self.video_config["num_predict_frames"], ...].to(self.device)
        x0_target_global_features = self._get_x0_target_global_features(target_pixels); t = torch.randint(0, self.timesteps, (B,), device=self.device, dtype=torch.long); noise = torch.randn_like(x0_target_global_features); xt_target_global_features = q_sample(x0_target_global_features, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, noise)
        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp and self.device.type == 'cuda'): pred_noise = self.model(xt_target_global_features, actual_cond_pixels_for_model, t, cfg_unconditional_flag=is_this_batch_unconditional); loss = F.mse_loss(pred_noise, noise)
        return loss, pred_noise.detach(), noise.detach()

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
                                logger.warning(f"R{self.rank}: NaN/Inf loss. Skip micro-batch.")
                                if is_optimizer_step_time:
                                    current_cycle_loss_sum = 0.0
                                    micro_batches_in_current_cycle = 0
                                    self.optimizer.zero_grad(set_to_none=True)
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
                    if hasattr(self.optimizer, 'q_controller') and self.optimizer.q_controller:
                        q_ctrl = self.optimizer.q_controller; avg_loss_for_q_cycle = current_cycle_loss_sum / micro_batches_in_current_cycle; current_lr_for_q_state = self.optimizer.param_groups[0]['lr']; current_mom_for_q_state = self.optimizer.param_groups[0]['momentum']; q_state = q_ctrl.get_state(current_lr_for_q_state, current_mom_for_q_state, current_unclipped_grad_norm, avg_loss_for_q_cycle)
                        if q_ctrl.prev_state is not None and q_ctrl.prev_action is not None and q_ctrl.prev_loss is not None and q_state is not None:
                            reward = q_ctrl.compute_reward(avg_loss_for_q_cycle, q_ctrl.prev_loss, current_unclipped_grad_norm)
                            if np.isfinite(reward): q_ctrl.update(q_ctrl.prev_state, q_ctrl.prev_action, reward, q_state)
                            else: logger.warning(f"R{self.rank} S{self.global_step}: Q-Ctrl non-finite reward.")
                        q_ctrl.prev_state = q_state
                        if q_state is not None: q_ctrl.prev_action = q_ctrl.choose_action(q_state)
                        else: # If q_state is None, prev_action might become stale or stay default
                            logger.warning(f"R{self.rank} S{self.global_step}: Q-state None. Q-action may be stale/default.")
                            q_ctrl.prev_action = q_ctrl.prev_action # Keep previous action or it defaults
                        q_ctrl.prev_loss = avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else q_ctrl.prev_loss
                    self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad(set_to_none=True); self.global_step += 1
                    if self.global_step % self.args.log_interval == 0 and self.am_main_process:
                        log_lr = self.optimizer.param_groups[0]['lr']
                        log_metrics = {"train/loss_cycle_avg": avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else -1.0, "train/lr_effective": log_lr, "train/grad_norm_unclipped_for_q": current_unclipped_grad_norm if np.isfinite(current_unclipped_grad_norm) else -1.0, "epoch_frac": epoch + ((batch_idx + 1) / total_micro_batches_estimate if total_micro_batches_estimate and total_micro_batches_estimate > 0 else 0), "global_step": self.global_step}
                        logger.info(f"E {epoch+1} S{self.global_step} L(cyc){log_metrics['train/loss_cycle_avg']:.4f} LR {log_metrics['train/lr_effective']:.2e} GradN(Q){log_metrics['train/grad_norm_unclipped_for_q']:.2f}")
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(log_metrics, step=self.global_step)
                        total_loss_interval = 0.0; items_interval = 0
                    if self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0 and self.am_main_process: self._save_checkpoint(is_intermediate=True, metrics={"train_loss_cycle_avg": avg_loss_for_q_cycle if np.isfinite(avg_loss_for_q_cycle) else -1.0})
                    current_cycle_loss_sum = 0.0; micro_batches_in_current_cycle = 0
            if self.am_main_process:
                avg_epoch_loss_val = total_loss_interval / items_interval if items_interval > 0 else (avg_loss_for_q_cycle if micro_batches_in_current_cycle == 0 and is_optimizer_step_time else float('nan'))
                logger.info(f"Epoch {epoch+1} finished. Approx avg epoch loss: {avg_epoch_loss_val:.4f}")
            if self.val_loader and self.am_main_process:
                val_metrics_dict = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if self.args.wandb and WANDB_AVAILABLE and wandb.run and val_metrics_dict: wandb.log({f"val/{k}": v for k,v in val_metrics_dict.items()}, step=self.global_step)
                current_val_primary_metric = val_metrics_dict.get(self.args.val_primary_metric, float('inf'))
                if current_val_primary_metric < self.best_val_loss : self.best_val_loss = current_val_primary_metric; self._save_checkpoint(is_best=True, metrics=val_metrics_dict)
            if self.am_main_process:
                save_metrics = self.last_val_metrics.copy() if self.last_val_metrics else {}
                save_metrics["epoch_end_train_loss_logged_intervals_avg"] = avg_epoch_loss_val if np.isfinite(avg_epoch_loss_val) else -1.0
                self._save_checkpoint(metrics=save_metrics)

    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Dict[str, float]:
        if not self.val_loader or not self.am_main_process: return {"avg_val_pixel_mse": float('inf')}
        self.model.eval(); total_mse_pixel = 0.0; total_psnr = 0.0; total_ssim = 0.0; total_lpips_val = 0.0; total_val_items = 0; logged_samples_count = 0; wandb_val_samples = []
        for batch_idx, batch_frames_raw in enumerate(tqdm(self.val_loader, desc="Validating", dynamic_ncols=True, disable=os.getenv('CI')=='true' or not self.am_main_process)):
            batch_frames = batch_frames_raw.to(self.device); num_cond = self.video_config["num_input_frames"]; num_pred = self.video_config["num_predict_frames"]; cond_pixels = batch_frames[:, :num_cond, ...]; target_pixels_ground_truth = batch_frames[:, num_cond : num_cond + num_pred, ...]; B_val = target_pixels_ground_truth.shape[0]
            predicted_target_pixels = self.sample(cond_pixels, num_inference_steps=self.args.val_sampling_steps, sampler_type=self.args.val_sampler_type, ddim_eta=0.0, cfg_guidance_scale=self.args.val_cfg_scale, force_on_main_process=True)
            pred_for_metrics = predicted_target_pixels[:,0,...]; gt_for_metrics = target_pixels_ground_truth[:,0,...]; pred_norm = (pred_for_metrics.clamp(-1, 1) + 1) / 2.0; gt_norm = (gt_for_metrics.clamp(-1, 1) + 1) / 2.0
            mse_pixel_val = F.mse_loss(pred_norm, gt_norm); total_mse_pixel += mse_pixel_val.item() * B_val; psnr_val = 10 * torch.log10(1.0 / (mse_pixel_val + EPS)) if mse_pixel_val > 0 else torch.tensor(100.0, device=self.device); total_psnr += psnr_val.item() * B_val
            if self.ssim_metric: ssim_val_current = self.ssim_metric(pred_norm, gt_norm); total_ssim += ssim_val_current.item() * B_val
            if self.lpips_loss_fn: lpips_val_current = self.lpips_loss_fn(pred_for_metrics, gt_for_metrics).mean(); total_lpips_val += lpips_val_current.item() * B_val
            total_val_items += B_val
            if self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run and logged_samples_count < num_val_samples_to_log:
                num_to_log_this_batch = min(B_val, num_val_samples_to_log - logged_samples_count)
                for k_sample in range(num_to_log_this_batch): cond_imgs_wandb = [wandb.Image(cond_pixels[k_sample, i].cpu().clamp(-1,1)*0.5+0.5, caption=f"Val Cond {i}") for i in range(cond_pixels.shape[1])]; pred_img_wandb = wandb.Image(pred_for_metrics[k_sample].cpu().clamp(-1,1)*0.5+0.5, caption="Val Pred"); gt_img_wandb = wandb.Image(gt_for_metrics[k_sample].cpu().clamp(-1,1)*0.5+0.5, caption="Val GT"); wandb_val_samples.extend(cond_imgs_wandb + [pred_img_wandb, gt_img_wandb])
                logged_samples_count += num_to_log_this_batch
        avg_mse_pixel = total_mse_pixel / total_val_items if total_val_items > 0 else float('inf'); avg_psnr = total_psnr / total_val_items if total_val_items > 0 else 0.0; avg_ssim = total_ssim / total_val_items if total_val_items > 0 and self.ssim_metric else 0.0; avg_lpips_metric = total_lpips_val / total_val_items if total_val_items > 0 and self.lpips_loss_fn else 0.0
        metrics = {"avg_val_pixel_mse": avg_mse_pixel, "avg_val_psnr": avg_psnr, "avg_val_ssim": avg_ssim, "avg_val_lpips": avg_lpips_metric}; self.last_val_metrics = metrics; logger.info(f"Validation Metrics: Pixel MSE: {avg_mse_pixel:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips_metric:.4f}")
        if wandb_val_samples and self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log({"validation_samples_sequence": wandb_val_samples}, step=self.global_step)
        return metrics
    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict]=None, is_best:bool=False):
        if not self.am_main_process: return
        m_save = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
        data = {'global_step':self.global_step, 'epoch':self.current_epoch, 'model_state_dict':m_save.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict(), 'scaler_state_dict':self.scaler.state_dict() if self.args.use_amp and self.device.type=='cuda' else None, 'args':vars(self.args), 'metrics':metrics if metrics else self.last_val_metrics, 'video_config':self.video_config, 'gaad_appearance_config':self.gaad_appearance_config, 'gaad_motion_config': self.gaad_motion_config, 'wubu_s_config':m_save.wubu_s.config if hasattr(m_save,'wubu_s') and m_save.wubu_s else {}, 'wubu_t_config':m_save.wubu_t.config if hasattr(m_save,'wubu_t') and m_save.wubu_t else {}, 'wubu_m_config':m_save.wubu_m.config if hasattr(m_save,'wubu_m') and m_save.wubu_m is not None else {}}
        fname_prefix="wubugaadphi_motion_ckpt_v052"; fpath = ""
        if is_best: fpath = os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_best.pt")
        elif is_intermediate: fpath = os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_step{self.global_step}.pt")
        else: fpath = os.path.join(self.args.checkpoint_dir, f"{fname_prefix}_ep{self.current_epoch+1}_step{self.global_step}.pt")
        try: torch.save(data,fpath); logger.info(f"Ckpt saved: {fpath}")
        except Exception as e: logger.error(f"Save ckpt error {fpath}: {e}",exc_info=True)
    def load_checkpoint(self, checkpoint_path:str) -> Tuple[int,int]:
        if not os.path.exists(checkpoint_path): logger.warning(f"Ckpt {checkpoint_path} not found."); return 0,0
        try:
            ckpt=torch.load(checkpoint_path,map_location=self.device); m_load = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model
            try: m_load.load_state_dict(ckpt['model_state_dict'])
            except RuntimeError as e: logger.warning(f"Strict load failed: {e}. Trying non-strict."); m_load.load_state_dict(ckpt['model_state_dict'],strict=False)
            if 'optimizer_state_dict' in ckpt and self.optimizer: self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt and self.scaler and ckpt['scaler_state_dict'] is not None: self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            loaded_global_step = ckpt.get('global_step',0); loaded_epoch = ckpt.get('epoch',0); self.best_val_loss = ckpt.get('metrics',{}).get(self.args.val_primary_metric, float('inf')) if 'metrics' in ckpt else float('inf')
            logger.info(f"Loaded ckpt {checkpoint_path} (Step {loaded_global_step}, Ep {loaded_epoch}). BestValLoss: {self.best_val_loss:.4f}"); return loaded_global_step, loaded_epoch
        except Exception as e: logger.error(f"Load ckpt error {checkpoint_path}: {e}",exc_info=True); return 0,0
    @torch.no_grad()
    def p_sample_ddpm(self, xt_target_global_features: torch.Tensor, conditioning_frames_pixels: Optional[torch.Tensor], t_tensor: torch.Tensor, t_int_val: int, cfg_guidance_scale: float = 1.0) -> torch.Tensor:
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; m_ref.eval()
        shape_for_broadcast = [-1] + [1] * (xt_target_global_features.dim() -1); dev = xt_target_global_features.device; dtype = xt_target_global_features.dtype
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(0, t_tensor).view(*shape_for_broadcast).to(dev,dtype); sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t_tensor).view(*shape_for_broadcast).to(dev,dtype); betas_t = self.betas.gather(0, t_tensor).view(*shape_for_broadcast).to(dev,dtype); posterior_log_variance_clipped_t = self.posterior_log_variance_clipped.gather(0, t_tensor).view(*shape_for_broadcast).to(dev,dtype)
        pred_noise = m_ref(xt_target_global_features, conditioning_frames_pixels, t_tensor, cfg_unconditional_flag= (conditioning_frames_pixels is None) ) if cfg_guidance_scale == 1.0 or conditioning_frames_pixels is None else (m_ref(xt_target_global_features, None, t_tensor, cfg_unconditional_flag=True) + cfg_guidance_scale * (m_ref(xt_target_global_features, conditioning_frames_pixels, t_tensor, cfg_unconditional_flag=False) - m_ref(xt_target_global_features, None, t_tensor, cfg_unconditional_flag=True)))
        term_in_paren = xt_target_global_features - (betas_t / (sqrt_one_minus_alphas_cumprod_t + EPS)) * pred_noise; model_mean_global_features = sqrt_recip_alphas_t * term_in_paren
        if t_int_val == 0: return model_mean_global_features
        else: noise_sample = torch.randn_like(xt_target_global_features); return model_mean_global_features + (0.5 * posterior_log_variance_clipped_t).exp() * noise_sample
    @torch.no_grad()
    def p_sample_ddim(self, xt_target_global_features: torch.Tensor, conditioning_frames_pixels: Optional[torch.Tensor], t_tensor: torch.Tensor, t_prev_tensor: torch.Tensor, eta: float = 0.0, cfg_guidance_scale: float = 1.0) -> torch.Tensor:
        m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; m_ref.eval()
        shape_for_broadcast = [-1] + [1] * (xt_target_global_features.dim() -1); dev = xt_target_global_features.device; dtype = xt_target_global_features.dtype
        alphas_cumprod_t = self.alphas_cumprod.gather(0, t_tensor).view(*shape_for_broadcast).to(dev,dtype); safe_t_prev_tensor = torch.clamp(t_prev_tensor, min=0); alphas_cumprod_t_prev = self.alphas_cumprod.gather(0, safe_t_prev_tensor).view(*shape_for_broadcast).to(dev,dtype); alphas_cumprod_t_prev = torch.where(t_prev_tensor.view(*shape_for_broadcast) < 0, torch.ones_like(alphas_cumprod_t_prev), alphas_cumprod_t_prev)
        pred_noise = m_ref(xt_target_global_features, conditioning_frames_pixels, t_tensor, cfg_unconditional_flag=(conditioning_frames_pixels is None)) if cfg_guidance_scale == 1.0 or conditioning_frames_pixels is None else (m_ref(xt_target_global_features, None, t_tensor, cfg_unconditional_flag=True) + cfg_guidance_scale * (m_ref(xt_target_global_features, conditioning_frames_pixels, t_tensor, cfg_unconditional_flag=False) - m_ref(xt_target_global_features, None, t_tensor, cfg_unconditional_flag=True)))
        x0_pred_global_features = (xt_target_global_features - torch.sqrt(torch.clamp(1. - alphas_cumprod_t, min=EPS)) * pred_noise) / (torch.sqrt(alphas_cumprod_t) + EPS)
        if self.args.ddim_x0_clip_val > 0: x0_pred_global_features = torch.clamp(x0_pred_global_features, -self.args.ddim_x0_clip_val, self.args.ddim_x0_clip_val)
        sigma_t_num = torch.clamp(1. - alphas_cumprod_t_prev, min=0.0); sigma_t_den = torch.clamp(1. - alphas_cumprod_t, min=EPS); sigma_t_ratio_alphacomp = torch.clamp(1. - alphas_cumprod_t / (alphas_cumprod_t_prev + EPS), min=0.0); sigma_t = eta * torch.sqrt( (sigma_t_num / sigma_t_den) * sigma_t_ratio_alphacomp )
        sigma_t = torch.where(t_prev_tensor.view(*shape_for_broadcast) < 0, torch.zeros_like(sigma_t), sigma_t); sigma_t = torch.where(t_tensor.view(*shape_for_broadcast) == 0, torch.zeros_like(sigma_t), sigma_t)
        pred_dir_xt = torch.sqrt(torch.clamp(1. - alphas_cumprod_t_prev - sigma_t**2, min=0.0)) * pred_noise; xt_prev = torch.sqrt(alphas_cumprod_t_prev) * x0_pred_global_features + pred_dir_xt
        if eta > 0 and t_prev_tensor.min() >= 0 : xt_prev = xt_prev + sigma_t * torch.randn_like(xt_target_global_features)
        return xt_prev
    @torch.no_grad()
    def sample(self, conditioning_frames_pixels: Optional[torch.Tensor], num_inference_steps: Optional[int] = None, sampler_type: str = "ddpm", ddim_eta: float = 0.0, cfg_guidance_scale: float = 1.0, force_on_main_process: bool = False, batch_size_if_uncond: int = 1) -> torch.Tensor:
        if not (self.am_main_process or force_on_main_process): logger.warning(f"R{self.rank}: Sample on non-main, not forced. Skip."); return torch.empty(0, device=self.device)
        self.model.eval(); m_ref = self.model.module if self.ddp_active and isinstance(self.model, DDP) else self.model; m_ref.to(self.device); B = conditioning_frames_pixels.shape[0] if conditioning_frames_pixels is not None else batch_size_if_uncond; dev = self.device
        eff_steps = min(num_inference_steps if num_inference_steps is not None else self.timesteps, self.timesteps); target_global_feat_dim = self.video_config["initial_cnn_feature_dim"]; num_pred_frames_target = self.video_config["num_predict_frames"]
        xt_target_global_features = torch.randn((B, num_pred_frames_target, target_global_feat_dim), device=dev, dtype=next(m_ref.parameters()).dtype)
        cond_str = f"CondFrames={conditioning_frames_pixels.shape[1]}" if conditioning_frames_pixels is not None else "UNCOND"; proc_id_str = f"R{self.rank}" if self.ddp_active else "Main"; logger.info(f"{proc_id_str}(Forced:{force_on_main_process}): Sampling {sampler_type.upper()}. BS={B}, {cond_str}, Steps={eff_steps}, Eta={ddim_eta if sampler_type=='ddim' else 'N/A'}, CFG={cfg_guidance_scale}")
        time_schedule_indices = torch.linspace(self.timesteps - 1, 0, eff_steps, dtype=torch.long, device=dev)
        for i in tqdm(range(eff_steps), desc="Sampling", leave=False, dynamic_ncols=True, disable=not (self.am_main_process or force_on_main_process) or os.getenv('CI') == 'true'):
            t_idx = time_schedule_indices[i]; t_batch = torch.full((B,), t_idx.item(), device=dev, dtype=torch.long); cond_pixels_for_step = conditioning_frames_pixels.to(dev, dtype=xt_target_global_features.dtype) if conditioning_frames_pixels is not None else None
            if sampler_type.lower() == "ddim": t_prev_idx = time_schedule_indices[i + 1] if i < eff_steps - 1 else torch.tensor(-1, device=dev, dtype=torch.long); t_prev_batch = torch.full((B,), t_prev_idx.item(), device=dev, dtype=torch.long); xt_target_global_features = self.p_sample_ddim(xt_target_global_features, cond_pixels_for_step, t_batch, t_prev_batch, eta=ddim_eta, cfg_guidance_scale=cfg_guidance_scale)
            elif sampler_type.lower() == "ddpm": xt_target_global_features = self.p_sample_ddpm(xt_target_global_features, cond_pixels_for_step, t_batch, t_idx.item(), cfg_guidance_scale=cfg_guidance_scale)
            else: raise ValueError(f"Unknown sampler: {sampler_type}")
        predicted_pixels = self.frame_autoencoder.decode(xt_target_global_features); logger.info(f"{proc_id_str}(Forced:{force_on_main_process}): Sampling finished.")
        return predicted_pixels

def seed_everything(seed:int,rank:int=0,world_size:int=1):
    actual_seed = seed + rank; random.seed(actual_seed); np.random.seed(actual_seed); torch.manual_seed(actual_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(actual_seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu-GAAD-Phi-Motion Diffusion Model (v0.05.2)")
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_dir")
    parser.add_argument('--single_video_roll', action='store_true', help="Legacy, VideoFrameDataset now handles single video path directly.")
    parser.add_argument('--local_rank', type=int, default=-1) # For DDP
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_h', type=int, default=64)
    parser.add_argument('--image_w', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_input_frames', type=int, default=3, help="Number of conditioning frames fed to the model.")
    parser.add_argument('--num_predict_frames', type=int, default=1, help="Model predicts features for this many target frames (currently 1)")
    parser.add_argument('--frame_skip', type=int, default=1, help="Skip frames when creating sequences from video.")

    # Feature Dimensions
    parser.add_argument('--initial_cnn_feature_dim', type=int, default=128)
    parser.add_argument('--wubu_s_output_dim', type=int, default=64)
    parser.add_argument('--wubu_t_output_dim', type=int, default=128)
    parser.add_argument('--wubu_m_output_dim', type=int, default=64, help="Output dimension for WuBu-M branch if active.")


    # GAAD Parameters (Appearance Branch)
    parser.add_argument('--gaad_num_regions', type=int, default=8)
    parser.add_argument('--gaad_region_roi_output_h', type=int, default=4)
    parser.add_argument('--gaad_region_roi_output_w', type=int, default=4)
    parser.add_argument('--gaad_region_feature_dim', type=int, default=64)
    parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])

    # Diffusion Parameters
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--beta_schedule',type=str,default='cosine', choices=['linear','cosine'])
    parser.add_argument('--beta_start',type=float,default=1e-4)
    parser.add_argument('--beta_end',type=float,default=0.02)
    parser.add_argument('--cosine_s',type=float,default=0.008)
    parser.add_argument('--phi_time_diffusion_scale', type=float, default=1.0, help="Global scale factor for time input to SinusoidalPhiEmbedding's forward pass.")
    parser.add_argument('--phi_time_base_freq', type=float, default=10000.0, help="Base frequency/period parameter for SinusoidalPhiEmbedding.")
    parser.add_argument('--use_phi_frequency_scaling_for_time_emb', action='store_true',
                        help="Use phi-based scaling for frequency spacing in SinusoidalPhiEmbedding, as per GAAD-WuBu-ST paper.")
    parser.add_argument('--diffusion_time_embedding_dim', type=int, default=128)


    # WuBu-S Parameters (Appearance)
    parser.add_argument('--wubu_s_num_levels', type=int, default=2)
    parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32])
    parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8])
    parser.add_argument('--wubu_s_use_rotation', action='store_true')
    parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true')

    # WuBu-T Parameters
    parser.add_argument('--wubu_t_num_levels', type=int, default=2)
    parser.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[128,64]) # Input dim will be sum of S_out and M_out (if M active)
    parser.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[1.0,0.5])
    parser.add_argument('--wubu_t_use_rotation', action='store_true')
    parser.add_argument('--wubu_t_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true')

    parser.add_argument('--wubu_dropout', type=float, default=0.1)

    # Motion Branch Parameters
    parser.add_argument('--use_wubu_motion_branch', action='store_true', help="Enable the GAAD+WuBu-M motion processing branch.")
    parser.add_argument('--diff_map_channels', type=int, default=1, help="Channels for diff map (1 for grayscale, 3 for color diff).")
    parser.add_argument('--gaad_motion_num_regions', type=int, default=8)
    parser.add_argument('--gaad_motion_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"])
    parser.add_argument('--gaad_motion_region_roi_output_h', type=int, default=3, help="H of RoI for M_t/M_{t-1} features for WuBu-M.")
    parser.add_argument('--gaad_motion_region_roi_output_w', type=int, default=3, help="W of RoI for M_t/M_{t-1} features for WuBu-M.")
    parser.add_argument('--motion_feature_dim_for_wubum_input', type=int, default=256, help="Target input dim for WuBu-M (after concat M_t, M_{t-1} RoIs and optional projection).")
    parser.add_argument('--wubu_m_num_levels', type=int, default=2)
    parser.add_argument('--wubu_m_hyperbolic_dims', nargs='+', type=int, default=[128,64])
    parser.add_argument('--wubu_m_initial_curvatures', nargs='+', type=float, default=[1.0, 0.7])
    parser.add_argument('--wubu_m_use_rotation', action='store_true')
    parser.add_argument('--wubu_m_phi_influence_curvature', action='store_true')
    parser.add_argument('--wubu_m_phi_influence_rotation_init', action='store_true')

    # Training Parameters
    parser.add_argument('--learning_rate',type=float,default=5e-5)
    parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0, help="Per-parameter Riemannian grad norm clip in RiSGD. 0 to disable.")
    parser.add_argument('--global_max_grad_norm',type=float,default=1.0, help="Global gradient norm clip for all model params. 0 to disable.")
    parser.add_argument('--q_controller_enabled',action='store_true')
    parser.add_argument('--checkpoint_dir',type=str, default='wubugaadphi_motion_checkpoints_v052') # Updated default
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--grad_accum_steps',type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_interval',type=int, default=20)
    parser.add_argument('--save_interval',type=int, default=200)
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--wandb_project',type=str,default='WuBuGAADPhiMotionDiffV052') # Updated default
    parser.add_argument('--wandb_run_name',type=str,default=None)
    parser.add_argument('--detect_anomaly',action='store_true')

    # CFG, Sampler, and Validation Parameters
    parser.add_argument('--cfg_unconditional_dropout_prob', type=float, default=0.1, help="Prob of dropping cond during training for CFG.")
    parser.add_argument('--cfg_guidance_scale', type=float, default=3.0, help="Guidance scale for CFG during demo sampling.")
    parser.add_argument('--val_cfg_scale', type=float, default=1.5, help="CFG scale for validation sampling.")
    parser.add_argument('--val_sampler_type', type=str, default="ddim", choices=["ddpm", "ddim"], help="Sampler for validation.")
    parser.add_argument('--val_sampling_steps', type=int, default=20, help="Sampling steps for validation.")
    parser.add_argument('--ddim_x0_clip_val', type=float, default=1.0, help="Value for clipping x0_pred in DDIM, 0 to disable.")
    parser.add_argument('--use_lpips_for_verification', action='store_true', help="Enable LPIPS metric for validation.")
    parser.add_argument('--validation_video_path', type=str, default=None, help="Path to a separate video file for validation.")
    parser.add_argument('--validation_split_fraction', type=float, default=0.1, help="Fraction of training data to use for validation if no separate val video.")
    parser.add_argument('--val_primary_metric', type=str, default="avg_val_pixel_mse", choices=["avg_val_pixel_mse", "avg_val_psnr", "avg_val_ssim", "avg_val_lpips"], help="Metric to track for best checkpoint.")
    parser.add_argument('--num_val_samples_to_log', type=int, default=2, help="Num validation samples to log to WandB.")

    # Demo Sampling Params
    parser.add_argument('--demo_sampler_type', type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument('--demo_ddim_eta', type=float, default=0.0)
    parser.add_argument('--demo_cfg_scale', type=float, default=3.0)
    parser.add_argument('--demo_sampling_steps', type=int, default=50)

    parsed_args = parser.parse_args()

    # Validations for wubu_s, wubu_t, wubu_m hyperbolic_dims and initial_curvatures
    def validate_wubu_config(args_obj, prefix_str, parser_ref, is_motion_branch_active_for_this_call):
        # Only validate if the branch is active OR if it's not the motion branch (S and T always validated if num_levels > 0)
        if not (prefix_str == "wubu_m" and not is_motion_branch_active_for_this_call):
            num_levels = getattr(args_obj, f"{prefix_str}_num_levels")
            if num_levels > 0: # Only proceed if levels are actually configured
                dims_attr_name = f"{prefix_str}_hyperbolic_dims"
                curv_attr_name = f"{prefix_str}_initial_curvatures"

                dims = getattr(args_obj, dims_attr_name)
                curvatures = getattr(args_obj, curv_attr_name)

                if not isinstance(dims, list): dims = [dims]
                if not isinstance(curvatures, list): curvatures = [curvatures]


                if len(dims) != num_levels:
                    if len(dims) == 1 and num_levels > 1:
                        logger.warning(f"WuBu-{prefix_str.split('_')[-1].upper()}: Replicating single hyperbolic_dim ({dims[0]}) for {num_levels} levels.")
                        dims = [dims[0]] * num_levels
                        setattr(args_obj, dims_attr_name, dims)
                    else:
                        parser_ref.error(f"WuBu-{prefix_str.split('_')[-1].upper()}: Length of hyperbolic_dims ({len(dims)}) must match num_levels ({num_levels}) or be 1 to replicate.")

                if len(curvatures) != num_levels:
                    if len(curvatures) == 1 and num_levels > 1:
                        logger.warning(f"WuBu-{prefix_str.split('_')[-1].upper()}: Replicating single initial_curvature ({curvatures[0]}) for {num_levels} levels.")
                        curvatures = [curvatures[0]] * num_levels
                    elif not curvatures: # Empty list
                        logger.warning(f"WuBu-{prefix_str.split('_')[-1].upper()}: initial_curvatures is empty, defaulting to [1.0] * {num_levels} levels.")
                        curvatures = [1.0] * num_levels
                    else: # Mismatch, e.g. 2 values for 3 levels
                         parser_ref.error(f"WuBu-{prefix_str.split('_')[-1].upper()}: Length of initial_curvatures ({len(curvatures)}) must match num_levels ({num_levels}), be 1 to replicate, or be empty to default.")
                    setattr(args_obj, curv_attr_name, curvatures[:num_levels]) # Ensure correct length

    # Validate S and T branches first
    validate_wubu_config(parsed_args, "wubu_s", parser, True) # True as it's not motion branch
    validate_wubu_config(parsed_args, "wubu_t", parser, True) # True as it's not motion branch

    # Then, handle WuBu-M based on whether the motion branch is active
    if parsed_args.use_wubu_motion_branch:
        validate_wubu_config(parsed_args, "wubu_m", parser, True) # Motion branch is active
    else:
        # Ensure m-configs are default/empty if motion branch is off
        if parsed_args.wubu_m_num_levels > 0:
            logger.info("Motion branch is disabled (--use_wubu_motion_branch not set) but wubu_m_num_levels > 0. Setting wubu_m_num_levels to 0 for consistency.")
            parsed_args.wubu_m_num_levels = 0
        # Ensure lists are empty for inactive motion branch for _configure_wubu_stack
        parsed_args.wubu_m_hyperbolic_dims = []
        parsed_args.wubu_m_initial_curvatures = []

    return parsed_args

def _configure_wubu_stack(args: argparse.Namespace, prefix: str) -> Optional[Dict]:
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
        if device.type=='cuda': # Ensure CUDA is initialized if selected
            _ = torch.cuda.set_device(device) # This also initializes CUDA context on this device

    am_main_process=(rank==0)

    # Clear existing handlers and reconfigure logging for each process
    # This avoids duplicated logging messages if the script is re-run in the same session.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_level = logging.INFO if am_main_process else logging.WARNING # Main process logs INFO, others WARNING
    logging.basicConfig(level=log_level, format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s', force=True)

    logger.info(f"--- WuBuGAADPhiMotionDiffV052 (R{rank}/{world_size},Dev {device},DDP:{ddp_active}) ---")
    seed_everything(args.seed,rank,world_size)

    if am_main_process:
        logger.info(f"Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_id = wandb.util.generate_id() if wandb.run is None else wandb.run.id
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name if args.wandb_run_name else f"wubumotion_v052_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                   config=vars(args), resume="allow", id=run_id)

    video_config = {
        "image_size":(args.image_h,args.image_w),"num_channels":args.num_channels, "num_input_frames":args.num_input_frames,
        "num_predict_frames":args.num_predict_frames, "initial_cnn_feature_dim":args.initial_cnn_feature_dim,
        "wubu_s_output_dim":args.wubu_s_output_dim, "wubu_t_output_dim":args.wubu_t_output_dim,
        "wubu_m_output_dim": args.wubu_m_output_dim if args.use_wubu_motion_branch else 0,
    }
    gaad_appearance_config = {
        "num_regions":args.gaad_num_regions, "region_roi_output_size":(args.gaad_region_roi_output_h, args.gaad_region_roi_output_w),
        "gaad_region_feature_dim":args.gaad_region_feature_dim, "decomposition_type":args.gaad_decomposition_type,
        "phi_time_diffusion_scale": args.phi_time_diffusion_scale, "phi_time_base_freq": args.phi_time_base_freq
    }
    gaad_motion_config = None
    if args.use_wubu_motion_branch:
        gaad_motion_config = {
            "num_regions": args.gaad_motion_num_regions, "decomposition_type": args.gaad_motion_decomposition_type,
        }

    wubu_s_config = _configure_wubu_stack(args, "wubu_s")
    wubu_t_config = _configure_wubu_stack(args, "wubu_t")
    wubu_m_config = _configure_wubu_stack(args, "wubu_m") # This will be None if motion branch is off

    if am_main_process:
        logger.info(f"VideoCfg:{video_config}\nGAADAppearCfg:{gaad_appearance_config}\nGAADMotionCfg:{gaad_motion_config}\nWuBuS-Cfg:{wubu_s_config}\nWuBuT-Cfg:{wubu_t_config}\nWuBuM-Cfg:{wubu_m_config}")

    model = WuBuSTDiffusionNet(wubu_s_config, wubu_t_config, wubu_m_config,
                               video_config, gaad_appearance_config, gaad_motion_config,
                               args.diffusion_time_embedding_dim, args).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model,log="all",log_freq=max(100,args.log_interval*5),log_graph=False) # log_graph can be True for small models

    if ddp_active:
        model=DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=args.detect_anomaly) # Set find_unused_parameters=True if parts of model might not be used based on config, or False if sure all params get grads. Anomaly detection often needs it True.

    q_cfg = DEFAULT_CONFIG_QLEARN_DIFFUSION.copy() if args.q_controller_enabled else None
    optimizer = RiemannianEnhancedSGD(model.parameters(),lr=args.learning_rate,q_learning_config=q_cfg,max_grad_norm_risgd=args.risgd_max_grad_norm)

    # --- Dataset and DataLoader Setup ---
    actual_video_path = args.video_data_path
    if "demo_video_data" in args.video_data_path: # Special handling for demo data
        actual_video_path = os.path.join(args.video_data_path, "dummy_video.mp4")
        if am_main_process: # Only main process creates the dummy video
            if not os.path.isdir(args.video_data_path):
                os.makedirs(args.video_data_path, exist_ok=True)
            if not os.path.exists(actual_video_path) :
                logger.info(f"Demo video data at {actual_video_path} not found. Creating dummy video...")
                min_raw_frames_needed = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                num_dummy_frames = max(50, min_raw_frames_needed + 20) # Ensure enough frames
                dummy_frames_tchw = torch.randint(0, 256, (num_dummy_frames, args.num_channels, args.image_h, args.image_w), dtype=torch.uint8)
                if VIDEO_IO_AVAILABLE and video_io is not None:
                    try:
                        # Permute dimensions to THWC before writing
                        video_io.write_video(actual_video_path, dummy_frames_tchw.permute(0, 2, 3, 1), fps=10) # Corrected line
                        logger.info(f"Created dummy video (torchvision.io): {actual_video_path} ({num_dummy_frames} frames).")
                    except Exception as e_tv_write:
                        logger.error(f"Error creating dummy video with torchvision.io: {e_tv_write}", exc_info=True)
                        # Fallback or error
                else:
                    try:
                        import av # PyAV as a fallback
                        container = av.open(actual_video_path, mode='w')
                        stream = container.add_stream('mpeg4', rate=10)
                        stream.width = args.image_w; stream.height = args.image_h; stream.pix_fmt = 'yuv420p'
                        for i_frame in range(num_dummy_frames):
                            # Permute to HWC for PyAV from_ndarray
                            frame_hwc_numpy = dummy_frames_tchw[i_frame].permute(1, 2, 0).numpy() # Needs permutation here too for PyAV fallback
                            av_frame = av.VideoFrame.from_ndarray(frame_hwc_numpy, format='rgb24')
                            for packet in stream.encode(av_frame):
                                container.mux(packet)
                        for packet in stream.encode(): # Flush stream
                            container.mux(packet)
                        container.close()
                        logger.info(f"Created dummy video (PyAV): {actual_video_path} ({num_dummy_frames} frames).")
                    except ImportError:
                        logger.error("PyAV not installed and torchvision.io failed. Cannot create dummy video.")
                    except Exception as e_av:
                        logger.error(f"Error creating dummy video with PyAV: {e_av}", exc_info=True)
        if ddp_active: # Ensure all processes wait for main process to create demo video if needed
            torch.distributed.barrier()

    is_file_path = args.single_video_roll or (os.path.isfile(actual_video_path) and not os.path.isdir(actual_video_path))
    if (is_file_path and not os.path.exists(actual_video_path)) or \
       (not is_file_path and not os.path.isdir(actual_video_path)) :
        logger.error(f"Video path {actual_video_path} (isdir: {os.path.isdir(actual_video_path)}, isfile: {os.path.isfile(actual_video_path)}) not found. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    total_frames_sample = args.num_input_frames + args.num_predict_frames

    full_dataset = None
    try:
        # VideoFrameDataset now expects a file path directly.
        # If actual_video_path is a directory, this needs adjustment or a different dataset class for directories.
        # For now, assuming actual_video_path will be a file.
        if os.path.isdir(actual_video_path) and not is_file_path:
            logger.error(f"Video data path {actual_video_path} is a directory, but VideoFrameDataset expects a file. Please specify a video file path.")
            if ddp_active and is_initialized(): destroy_process_group()
            sys.exit(1)
        full_dataset = VideoFrameDataset(video_path=actual_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
    except Exception as e_dataset_init:
        logger.error(f"Failed to initialize main VideoFrameDataset from {actual_video_path}: {e_dataset_init}", exc_info=True)
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    if not full_dataset or len(full_dataset) == 0 :
        logger.error("Main dataset empty or failed to load. Check video path and content. Exiting.")
        if ddp_active and is_initialized(): destroy_process_group()
        sys.exit(1)

    train_dataset, val_dataset = full_dataset, None # Default to using full dataset for training
    if args.validation_video_path and os.path.exists(args.validation_video_path) and os.path.isfile(args.validation_video_path):
        try:
            val_dataset = VideoFrameDataset(video_path=args.validation_video_path, num_frames_total=total_frames_sample, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip)
            if len(val_dataset) > 0:
                logger.info(f"Using separate validation video: {args.validation_video_path}, {len(val_dataset)} samples.")
            else:
                logger.warning(f"Validation video {args.validation_video_path} loaded 0 samples. No validation.")
                val_dataset = None
        except Exception as e_val_dataset:
            logger.warning(f"Could not load validation dataset from {args.validation_video_path}: {e_val_dataset}. No validation will be performed.")
            val_dataset = None
    elif args.validation_split_fraction > 0 and len(full_dataset) > 10 : # Split from training if no separate val video
        num_val_samples = int(len(full_dataset) * args.validation_split_fraction)
        num_train_samples = len(full_dataset) - num_val_samples
        if num_train_samples > 0 and num_val_samples > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train_samples, num_val_samples], generator=torch.Generator().manual_seed(args.seed + rank)) # Add rank to seed for split
            logger.info(f"Split main dataset: {len(train_dataset)} train, {len(val_dataset)} val samples.")
        else:
            logger.warning(f"Not enough samples in main dataset ({len(full_dataset)}) to perform validation split. Using all for training.")
            val_dataset = None # Ensure val_dataset is None

    train_sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True,seed=args.seed) if ddp_active else None
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=(train_sampler is None),num_workers=args.num_workers,sampler=train_sampler,pin_memory=(device.type=='cuda'),worker_init_fn=lambda wid: seed_everything(args.seed+wid+rank*100,rank,world_size) if args.num_workers>0 else None,drop_last=True)

    val_loader = None
    if val_dataset and len(val_dataset) > 0: # Check if val_dataset is not None and has items
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=(device.type=='cuda'), drop_last=False)
    elif am_main_process:
        logger.info("No validation dataset/loader configured or validation dataset is empty.")

    trainer = DiffusionTrainer(model,optimizer,device,train_loader,val_loader,args,rank,world_size,ddp_active,video_config,gaad_appearance_config,gaad_motion_config,wubu_m_config)
    start_global_step,start_epoch=0,0
    if args.load_checkpoint:
        start_global_step,start_epoch=trainer.load_checkpoint(args.load_checkpoint)

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
            if args.epochs>0 and hasattr(trainer,'sample') and trainer.global_step > 0 and len(train_loader)>0:
                logger.info("DEMO SAMPLING...")
                try:
                    demo_batch_for_cond_iter = iter(train_loader)
                    demo_batch_for_cond = next(demo_batch_for_cond_iter)
                    demo_cond_pixels = demo_batch_for_cond[:, :args.num_input_frames, ...].to(device)
                    demo_cond_pixels = demo_cond_pixels[0:1] # Use only the first sample for demo

                    pred_pixels = trainer.sample(demo_cond_pixels, num_inference_steps=args.demo_sampling_steps,
                                                 sampler_type=args.demo_sampler_type, ddim_eta=args.demo_ddim_eta,
                                                 cfg_guidance_scale=args.demo_cfg_scale, force_on_main_process=True)

                    logger.info(f"Demo predicted pixels shape: {pred_pixels.shape}")
                    if pred_pixels.numel() > 0 and pred_pixels.shape[0] > 0:
                        save_path_dir = os.path.join(args.checkpoint_dir, "demo_samples_v052")
                        os.makedirs(save_path_dir, exist_ok=True)
                        for i_demo in range(min(args.num_input_frames, demo_cond_pixels.shape[1])):
                            save_image(demo_cond_pixels[0, i_demo].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_path_dir, f"demo_cond_frame_{i_demo}.png"))
                        for i_pred_demo in range(pred_pixels.shape[1]):
                             save_image(pred_pixels[0,i_pred_demo].cpu().clamp(-1,1)*0.5+0.5, os.path.join(save_path_dir, f"demo_pred_frame_{i_pred_demo}.png"))
                        logger.info(f"Saved demo sample frames to {save_path_dir}")
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
        logger.info(f"Rank {rank}: WuBuGAADPhiMotionDiffNet (v0.05.2) script finished.")

if __name__ == "__main__":
    main()
