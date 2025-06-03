# wubu_diffusion.py
# Denoising Diffusion Probabilistic Model (DDPM) using ADVANCED FractalDepthQuaternionWuBuRungs.
# CPU-only, NO CNNs, NO explicit DFT/DCT for features (WuBu processes patch pixels).
# Based on WuBuCPUOnlyGen_v1.py

import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math, random, argparse, logging, time, os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque
from pathlib import Path
import torchvision.transforms as T
import torchvision.transforms.functional as TF

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio unavailable. Dummy video creation and some loading might fail.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# --- Constants ---
EPS = 1e-5 # Epsilon for numerical stability
PHI = (1 + math.sqrt(5)) / 2 # Golden ratio
TAN_VEC_CLAMP_VAL = 1e4
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e8

# --- Basic Logging Setup ---
logger_wubu_diffusion = logging.getLogger("WuBuDiffusion")
if not logger_wubu_diffusion.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_wubu_diffusion.addHandler(handler)
    logger_wubu_diffusion.setLevel(logging.INFO)

# --- Global TQDM setup ---
_TQDM_INITIAL_WRAPPER = None
_tqdm_imported_successfully = False
try:
    from tqdm import tqdm as _tqdm_imported_module
    _TQDM_INITIAL_WRAPPER = _tqdm_imported_module
    _tqdm_imported_successfully = True
except ImportError:
    print("Warning: tqdm library not found. Progress bars will be disabled or use a basic dummy.")
    class _TqdmDummyMissing:
        _last_desc_printed = None # Class variable to track last printed description
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable if iterable is not None else []
            desc = kwargs.get('desc')
            if desc and desc != _TqdmDummyMissing._last_desc_printed:
                print(f"{desc} (tqdm not available, using basic dummy progress)")
                _TqdmDummyMissing._last_desc_printed = desc
        def __iter__(self): return iter(self.iterable)
        def __enter__(self): return self
        def __exit__(self, *exc_info): pass
        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs_p): pass
        def set_postfix_str(self, s: str = "", refresh: bool = True): pass
        def update(self, n: int = 1): pass
        def close(self): pass
    _TQDM_INITIAL_WRAPPER = _TqdmDummyMissing

tqdm = _TQDM_INITIAL_WRAPPER


# --- Global flag for detailed micro-transform debugging ---
DEBUG_MICRO_TRANSFORM_INTERNALS = False # Set to True for verbose prints from ManifoldUncrumpleTransform

def print_tensor_stats_mut(tensor: Optional[torch.Tensor], name: str, micro_level_idx: Optional[int] = None, stage_idx: Optional[int] = None, enabled: bool = True, batch_idx_to_print: int = 0):
    """ A local print_tensor_stats for ManifoldUncrumpleTransform module if not globally available """
    if not enabled or tensor is None or not DEBUG_MICRO_TRANSFORM_INTERNALS: return
    if tensor.numel() == 0: return
    # Simplified print for brevity, only if logger is at DEBUG level
    if logger_wubu_diffusion.isEnabledFor(logging.DEBUG):
        prefix = f"MUT_L{micro_level_idx if micro_level_idx is not None else 'N'}/S{stage_idx if stage_idx is not None else 'N'}| "
        # Only show detailed stats for the first item in batch if batched, else full tensor
        item_to_show = tensor
        if tensor.dim() > 1 and tensor.shape[0] > batch_idx_to_print:
            item_to_show = tensor[batch_idx_to_print].detach()
            prefix += f"{name}(i{batch_idx_to_print}):"
        elif tensor.dim() == 1:
            item_to_show = tensor.detach()
            prefix += f"{name}:"
        else: # scalar or 0-dim
            item_to_show = tensor.detach()
            prefix += f"{name}(s):"

        logger_wubu_diffusion.debug(prefix +
            f"Sh:{tensor.shape}, Dt:{tensor.dtype}, "
            f"NaN:{torch.isnan(item_to_show).any().item()}, Inf:{torch.isinf(item_to_show).any().item()}, "
            f"Min:{item_to_show.min().item():.2e}, Max:{item_to_show.max().item():.2e}, "
            f"Mu:{item_to_show.mean().item():.2e}, Std:{item_to_show.std().item():.2e}, "
            f"Nrm:{torch.linalg.norm(item_to_show.float()).item():.2e}")


# --- Utility Functions (Copied from WuBuCPUOnlyGen_v1.py, mostly unchanged) ---
def init_weights_general(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        if m.data is not None:
            if m.dim() > 1: nn.init.xavier_uniform_(m.data)
            else: nn.init.normal_(m.data, std=0.02)
    elif isinstance(m, (nn.LayerNorm)):
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', True)):
            if hasattr(m, 'weight') and m.weight is not None: nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor:
    return F.softplus(param_unconstrained) + min_val

# Original print_tensor_stats, separate from MUT version
def print_tensor_stats_fdq(tensor: Optional[torch.Tensor], name: str, micro_level_idx: Optional[int] = None, stage_idx: Optional[int] = None, enabled: bool = True, batch_idx_to_print: int = 0):
    if not enabled or tensor is None or not DEBUG_MICRO_TRANSFORM_INTERNALS: return # Controlled by same global flag for now
    if tensor.numel() == 0: return
    item_tensor = tensor; prefix = f"FDQ_L{micro_level_idx if micro_level_idx is not None else 'N'}/S{stage_idx if stage_idx is not None else 'N'}| "
    if tensor.dim() > 1 and tensor.shape[0] > batch_idx_to_print : item_tensor = tensor[batch_idx_to_print].detach(); prefix += f"{name}(i{batch_idx_to_print}):"
    elif tensor.dim() == 1: item_tensor = tensor.detach(); prefix += f"{name}:"
    else: item_tensor = tensor.detach(); prefix += f"{name}(f):"
    if logger_wubu_diffusion.isEnabledFor(logging.DEBUG):
        logger_wubu_diffusion.debug(prefix +
            f"Sh:{tensor.shape},Dt:{tensor.dtype},"
            f"Min:{item_tensor.min().item():.2e},Max:{item_tensor.max().item():.2e},"
            f"Mu:{item_tensor.mean().item():.2e},Std:{item_tensor.std().item():.2e},"
            f"Nrm:{torch.linalg.norm(item_tensor.float()).item():.2e}")


def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    if angle_rad.dim() == 0: angle_rad = angle_rad.unsqueeze(0)
    if axis.shape[-1] == 0:
        effective_axis_shape = list(angle_rad.shape) + [3]
        axis_normalized = torch.zeros(effective_axis_shape, device=angle_rad.device, dtype=angle_rad.dtype)
        if axis_normalized.numel() > 0: axis_normalized[..., 0] = 1.0
    elif axis.dim() == 1 and angle_rad.shape[0] > 1 :
        axis = axis.unsqueeze(0).expand(angle_rad.shape[0], -1)
        axis_normalized = F.normalize(axis, p=2, dim=-1)
    elif axis.dim() == angle_rad.dim() and axis.shape[:-1] == angle_rad.shape:
        axis_normalized = F.normalize(axis, p=2, dim=-1)
    elif axis.dim() == 1 and angle_rad.shape[0] == 1:
        axis = axis.unsqueeze(0)
        axis_normalized = F.normalize(axis, p=2, dim=-1)
    else:
        try:
            target_axis_batch_shape = list(angle_rad.shape)
            current_axis_dim = axis.shape[-1]
            axis_expanded = axis.expand(target_axis_batch_shape + [current_axis_dim])
            axis_normalized = F.normalize(axis_expanded, p=2, dim=-1)
        except RuntimeError:
            logger_wubu_diffusion.warning(f"Quat axis-angle: Cannot broadcast axis (shape {axis.shape}) to align with angle_rad (shape {angle_rad.shape}). Using default.")
            effective_axis_shape = list(angle_rad.shape) + [3]
            axis_normalized = torch.zeros(effective_axis_shape, device=angle_rad.device, dtype=angle_rad.dtype)
            if axis_normalized.numel() > 0: axis_normalized[..., 0] = 1.0
    angle_half = angle_rad / 2.0
    q_w = torch.cos(angle_half).unsqueeze(-1)
    q_xyz = axis_normalized * torch.sin(angle_half).unsqueeze(-1)
    return torch.cat([q_w, q_xyz], dim=-1)

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    s1_orig, s2_orig = list(q1.shape), list(q2.shape)
    if s1_orig[-1] != 4 or s2_orig[-1] != 4: raise ValueError(f"Quaternions must have 4 components. Got q1: {s1_orig}, q2: {s2_orig}")
    q1_eff, q2_eff = q1, q2
    if len(s1_orig) < len(s2_orig): q1_eff = q1.view([1]*(len(s2_orig)-len(s1_orig)) + s1_orig)
    elif len(s2_orig) < len(s1_orig): q2_eff = q2.view([1]*(len(s1_orig)-len(s2_orig)) + s2_orig)
    try: w1, x1, y1, z1 = q1_eff[..., 0:1], q1_eff[..., 1:2], q1_eff[..., 2:3], q1_eff[..., 3:4]; w2, x2, y2, z2 = q2_eff[..., 0:1], q2_eff[..., 1:2], q2_eff[..., 2:3], q2_eff[..., 3:4]
    except IndexError as e: logger_wubu_diffusion.error(f"Quat component extraction failed. q1_eff: {q1_eff.shape}, q2_eff: {q2_eff.shape}. Error: {e}"); raise e
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.cat([w, x, y, z], dim=-1)

def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor: return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)
def normalize_quaternion(q: torch.Tensor, eps: float = EPS) -> torch.Tensor: norm = torch.linalg.norm(q, dim=-1, keepdim=True); return q / (norm + eps)

# --- Hyperbolic Geometry Utilities (Copied from WuBuCPUOnlyGen_v1.py) ---
# ... (HyperbolicUtils and PoincareBall classes remain unchanged, kept for brevity) ...
class HyperbolicUtils: # (Content largely unchanged)
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

class PoincareBall:
    def __init__(self, c_scalar: float = 1.0):
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0: self.c, self.sqrt_c, self.radius = 0.0, 0.0, float('inf')
        else: self.c, self.sqrt_c, self.radius = c_scalar, math.sqrt(self.c), 1.0 / (math.sqrt(self.c) + EPS)
        self._name = f'PoincareBall(c={self.c:.3g})'
    @property
    def name(self) -> str: return self._name
    def proju(self, x: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.poincare_clip(x, self.c, radius=1.0, eps=EPS * 10)
    def expmap0_scaled(self, dp: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_exponential_map(dp, self.c, scale_scalar=scale_scalar, eps=EPS)
    def logmap0_scaled(self, p: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_logarithmic_map(p, self.c, scale_scalar=scale_scalar, eps=EPS)


# --- New ManifoldUncrumpleTransform ---
class ManifoldUncrumpleTransform(nn.Module):
    def __init__(self,
                 dim: int,
                 num_scaffold_points: int,
                 scaffold_init_scale: float = 0.01,
                 affinity_temp: float = 0.1,
                 hypernet_hidden_dim_factor: float = 1.0,
                 learn_rotation: bool = True,
                 learn_scale: bool = True,
                 scale_activation: str = 'sigmoid', # 'sigmoid', 'tanh', 'exp', 'softplus'
                 scale_min: float = 0.5,
                 scale_max: float = 2.0,
                 learn_translation: bool = True,
                 stage_info: Optional[str] = None): # For debugging
        super().__init__()
        self.dim = dim
        self.stage_info_str = stage_info if stage_info else "" # For debug prints

        if self.dim <=0:
            logger_wubu_diffusion.warning(f"ManifoldUncrumpleTransform {self.stage_info_str} initialized with dim <= 0. It will act as an identity.")
            self.is_identity = True
            return
        self.is_identity = False

        self.num_scaffold_points = num_scaffold_points
        self.affinity_temp = max(EPS, affinity_temp)

        self.learn_rotation = learn_rotation and self.dim > 1
        self.learn_scale = learn_scale
        self.scale_activation = scale_activation
        self.scale_min = scale_min
        self.scale_max = scale_max
        if self.scale_min >= self.scale_max and self.learn_scale:
            logger_wubu_diffusion.warning(f"ManifoldUncrumpleTransform {self.stage_info_str}: scale_min >= scale_max. Adjusting to make range valid or disabling scale.")
            self.scale_max = self.scale_min + 0.1 # Ensure some range

        self.learn_translation = learn_translation

        self.scaffold_points = nn.Parameter(torch.randn(num_scaffold_points, dim) * scaffold_init_scale)

        hypernet_input_dim = dim
        hypernet_hidden_dim = max(dim, int(dim * hypernet_hidden_dim_factor))

        self.param_gen_parts = nn.ModuleDict()
        current_total_transform_params_indicator = 0 # Just to check if any transform component is active

        if self.learn_rotation:
            self.rot_param_dim = self.dim * (self.dim - 1) // 2
            if self.rot_param_dim > 0:
                self.param_gen_parts['rot_gen'] = nn.Sequential(
                    nn.Linear(hypernet_input_dim, hypernet_hidden_dim), nn.GELU(),
                    nn.Linear(hypernet_hidden_dim, self.rot_param_dim)
                )
                current_total_transform_params_indicator +=1
            else: self.learn_rotation = False # No rotation for dim=1

        if self.learn_scale:
            self.scale_param_dim = dim
            self.param_gen_parts['scale_gen'] = nn.Sequential(
                nn.Linear(hypernet_input_dim, hypernet_hidden_dim), nn.GELU(),
                nn.Linear(hypernet_hidden_dim, self.scale_param_dim)
            )
            current_total_transform_params_indicator +=1

        if self.learn_translation:
            self.trans_param_dim = dim
            self.param_gen_parts['trans_gen'] = nn.Sequential(
                nn.Linear(hypernet_input_dim, hypernet_hidden_dim), nn.GELU(),
                nn.Linear(hypernet_hidden_dim, self.trans_param_dim)
            )
            current_total_transform_params_indicator +=1
        
        if current_total_transform_params_indicator == 0:
            logger_wubu_diffusion.warning(f"ManifoldUncrumpleTransform {self.stage_info_str}: No transform components (R,S,T) enabled. Will act as identity.")
            self.is_identity = True

    def _generate_rotation_matrix_batch(self, rot_params_batch: torch.Tensor) -> torch.Tensor:
        # rot_params_batch: (N_scaffold, rot_param_dim)
        # Output: (N_scaffold, dim, dim)
        Ns, rp_dim = rot_params_batch.shape
        if not self.learn_rotation or self.rot_param_dim == 0 or rp_dim == 0:
            return torch.eye(self.dim, device=rot_params_batch.device, dtype=rot_params_batch.dtype).unsqueeze(0).expand(Ns, -1, -1)

        R_matrices = torch.empty(Ns, self.dim, self.dim, device=rot_params_batch.device, dtype=rot_params_batch.dtype)
        indices = torch.triu_indices(self.dim, self.dim, offset=1, device=rot_params_batch.device)
        
        for k in range(Ns):
            X_k = torch.zeros(self.dim, self.dim, device=rot_params_batch.device, dtype=rot_params_batch.dtype)
            if X_k.numel() > 0 and indices.numel() > 0 and self.rot_param_dim > 0 :
                 if indices.shape[1] == rot_params_batch[k].shape[0]: # rot_param_dim matches
                    X_k[indices[0], indices[1]] = rot_params_batch[k]
                 else: # Fallback if somehow dimensions mismatch, log warning.
                    logger_wubu_diffusion.warning(f"ManifoldUncrumpleTransform {self.stage_info_str}: Mismatch in rotation param dim for scaffold {k}. Expected {self.rot_param_dim}, got {rot_params_batch[k].shape[0]}. Filling partially.")
                    num_to_fill = min(indices.shape[1], rot_params_batch[k].shape[0])
                    if num_to_fill > 0 : X_k[indices[0,:num_to_fill], indices[1,:num_to_fill]] = rot_params_batch[k, :num_to_fill]
            
            X_k = X_k - X_k.T
            try:
                R_matrices[k] = torch.linalg.matrix_exp(torch.clamp(X_k, -5, 5))
            except Exception: # More specific error catching could be added if needed
                # logger_wubu_diffusion.error(f"Matrix_exp error for scaffold {k} in {self.stage_info_str}: {e}. Using identity.", exc_info=False)
                R_matrices[k] = torch.eye(self.dim, device=X_k.device, dtype=X_k.dtype)
        return R_matrices

    def forward(self, v_current: torch.Tensor, current_sigma_gauss_skin: float, micro_level_idx: Optional[int] = None, stage_idx_override: Optional[int] = None) -> torch.Tensor:
        if self.is_identity:
            return v_current

        B, D = v_current.shape
        stage_id_for_log = stage_idx_override if stage_idx_override is not None else self.stage_info_str
        print_tensor_stats_mut(v_current, f"In_v_curr S:{stage_id_for_log}", micro_level_idx=micro_level_idx)

        diffs = v_current.unsqueeze(1) - self.scaffold_points.unsqueeze(0)
        sq_dists = torch.sum(diffs.pow(2), dim=-1)
        
        affinity_numerator = -sq_dists / (2 * (current_sigma_gauss_skin**2) + EPS)
        affinity_numerator_clamped = torch.clamp(affinity_numerator, min=-30.0, max=30.0)
        
        affinity_weights = F.softmax(affinity_numerator_clamped / self.affinity_temp, dim=-1)
        print_tensor_stats_mut(affinity_weights, f"AffW S:{stage_id_for_log}", micro_level_idx=micro_level_idx)

        all_rot_gen_params = self.param_gen_parts['rot_gen'](self.scaffold_points) if self.learn_rotation and self.rot_param_dim > 0 else None
        all_scale_gen_params = self.param_gen_parts['scale_gen'](self.scaffold_points) if self.learn_scale else None
        all_trans_gen_params = self.param_gen_parts['trans_gen'](self.scaffold_points) if self.learn_translation else None

        R_all_k = self._generate_rotation_matrix_batch(all_rot_gen_params) if all_rot_gen_params is not None else \
                  torch.eye(self.dim, device=v_current.device, dtype=v_current.dtype).unsqueeze(0).expand(self.num_scaffold_points, -1, -1)
        
        if all_scale_gen_params is not None:
            if self.scale_activation == 'sigmoid':
                s_params = torch.sigmoid(all_scale_gen_params) * (self.scale_max - self.scale_min) + self.scale_min
            elif self.scale_activation == 'tanh':
                s_params = (torch.tanh(all_scale_gen_params) * 0.5 + 0.5) * (self.scale_max - self.scale_min) + self.scale_min
            elif self.scale_activation == 'exp':
                s_params = torch.exp(torch.clamp(all_scale_gen_params, -5, 5))
            else: # default 'softplus' like
                s_params = F.softplus(all_scale_gen_params) + EPS 
            S_all_k = torch.clamp(s_params, min=self.scale_min, max=self.scale_max) # Ensure scales are within desired bounds
        else:
            S_all_k = torch.ones(self.num_scaffold_points, self.dim, device=v_current.device, dtype=v_current.dtype)
        
        T_all_k = all_trans_gen_params if all_trans_gen_params is not None else \
                  torch.zeros(self.num_scaffold_points, self.dim, device=v_current.device, dtype=v_current.dtype)
        
        print_tensor_stats_mut(R_all_k, f"R_all_k S:{stage_id_for_log}", micro_level_idx=micro_level_idx)
        print_tensor_stats_mut(S_all_k, f"S_all_k S:{stage_id_for_log}", micro_level_idx=micro_level_idx)
        print_tensor_stats_mut(T_all_k, f"T_all_k S:{stage_id_for_log}", micro_level_idx=micro_level_idx)

        v_current_exp = v_current.unsqueeze(1) # (B, 1, D)
        
        v_scaled_per_scaffold = v_current_exp * S_all_k.unsqueeze(0) # (B, Ns, D)
        print_tensor_stats_mut(v_scaled_per_scaffold, f"v_scaled_scaff S:{stage_id_for_log}", micro_level_idx=micro_level_idx)
        
        # R_all_k: (Ns, D, D). v_scaled_per_scaffold: (B, Ns, D)
        # Output[b,k,d_out] = sum_{d_in} v_scaled_per_scaffold[b,k,d_in] * R_all_k[k,d_in,d_out] (if R is transposed)
        # or Output[b,k,d_out] = sum_{d_in} R_all_k[k,d_out,d_in] * v_scaled_per_scaffold[b,k,d_in]
        v_rotated_per_scaffold = torch.einsum('bki,kji->bkj', v_scaled_per_scaffold, R_all_k) # R_all_k[k,j,i] * v_scaled[b,k,i] summed over i -> out[b,k,j]
        print_tensor_stats_mut(v_rotated_per_scaffold, f"v_rotated_scaff S:{stage_id_for_log}", micro_level_idx=micro_level_idx)
        
        v_transformed_per_scaffold = v_rotated_per_scaffold + T_all_k.unsqueeze(0) # (B, Ns, D)
        print_tensor_stats_mut(v_transformed_per_scaffold, f"v_transformed_scaff S:{stage_id_for_log}", micro_level_idx=micro_level_idx)
        
        affinity_weights_exp = affinity_weights.unsqueeze(-1) # (B, Ns, 1)
        v_next = torch.sum(affinity_weights_exp * v_transformed_per_scaffold, dim=1) # (B, D)
        print_tensor_stats_mut(v_next, f"Out_v_next S:{stage_id_for_log}", micro_level_idx=micro_level_idx)

        return v_next

    def get_affinity_weights(self, v_current: torch.Tensor, current_sigma_gauss_skin: float) -> torch.Tensor:
        """Helper to expose affinity weights if needed externally, e.g., for BSP gating."""
        if self.is_identity: # Should not be called if identity, but defensive
            return torch.ones(v_current.shape[0], self.num_scaffold_points, device=v_current.device, dtype=v_current.dtype) / self.num_scaffold_points

        diffs = v_current.unsqueeze(1) - self.scaffold_points.unsqueeze(0)
        sq_dists = torch.sum(diffs.pow(2), dim=-1)
        affinity_numerator = -sq_dists / (2 * (current_sigma_gauss_skin**2) + EPS)
        affinity_numerator_clamped = torch.clamp(affinity_numerator, min=-30.0, max=30.0)
        affinity_scores = torch.exp(affinity_numerator_clamped)
        affinity_weights = F.softmax(affinity_scores / self.affinity_temp, dim=-1)
        return affinity_weights


# --- Original FractalMicroTransformRungs (for comparison or fallback) ---
class OriginalFractalMicroTransformRungs(nn.Module): # Renamed for clarity
    def __init__(self, dim: int = 4, transform_type: str = "mlp", hidden_dim_factor: float = 1.0, use_quaternion_so4: bool = True,
                 enable_internal_sub_processing: bool = True, enable_stateful_micro_transform: bool = True, enable_hypernetwork_modulation: bool = True,
                 internal_state_dim_factor: float = 1.0, hyper_mod_strength: float = 0.01):
        super().__init__(); self.dim = dim; self.use_quaternion_so4 = use_quaternion_so4 and (dim == 4)
        self.enable_internal_sub_processing = enable_internal_sub_processing and (dim == 4); self.enable_stateful_micro_transform = enable_stateful_micro_transform
        self.enable_hypernetwork_modulation = enable_hypernetwork_modulation; self.hyper_mod_strength = hyper_mod_strength; self.debug_counter = 0
        if self.use_quaternion_so4: self.rotation_param_generator = QuaternionSO4From6Params(); self.rotation_generator_matrix = None
        elif dim > 0: self.rotation_generator_matrix = SkewSymmetricMatrix(dim); self.rotation_param_generator = None
        else: self.rotation_generator_matrix = nn.Identity(); self.rotation_param_generator = None
        mlp_hidden_dim = max(dim, int(dim * hidden_dim_factor)) if dim > 0 else 0
        if transform_type == 'mlp' and dim > 0 and mlp_hidden_dim > 0: self.non_rotational_map_layers = nn.ModuleList([nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim)])
        elif transform_type == 'linear' and dim > 0: self.non_rotational_map_layers = nn.ModuleList([nn.Linear(dim, dim)])
        else: self.non_rotational_map_layers = nn.ModuleList([nn.Identity()])
        if self.enable_stateful_micro_transform and self.dim > 0:
            self.internal_state_dim = max(1, int(self.dim * internal_state_dim_factor)); state_mlp_hidden = max(self.internal_state_dim // 2, dim // 2, 1)
            self.state_update_gate_mlp = nn.Sequential(nn.Linear(dim + self.internal_state_dim, state_mlp_hidden), nn.GELU(), nn.Linear(state_mlp_hidden, self.internal_state_dim), nn.Tanh())
            self.state_influence_mlp = nn.Sequential(nn.Linear(self.internal_state_dim, state_mlp_hidden), nn.GELU(), nn.Linear(state_mlp_hidden, dim))
        else: self.internal_state_dim = 0; self.state_update_gate_mlp = None; self.state_influence_mlp = None
        if self.enable_hypernetwork_modulation and self.dim > 0:
            if len(self.non_rotational_map_layers) > 0 and isinstance(self.non_rotational_map_layers[0], nn.Linear):
                first_linear_out_features = self.non_rotational_map_layers[0].out_features; hyper_hidden_dim = max(dim // 2, first_linear_out_features // 2, 1)
                self.hyper_bias_generator = nn.Sequential(nn.Linear(dim, hyper_hidden_dim), nn.GELU(), nn.Linear(hyper_hidden_dim, first_linear_out_features))
            else: self.enable_hypernetwork_modulation = False; self.hyper_bias_generator = None
            if self.enable_hypernetwork_modulation and self.hyper_bias_generator is None :
                 logger_wubu_diffusion.warning("HyperNet modulation enabled but no suitable MLP found or hyper_bias_generator is None. Disabling for this micro-transform.")
                 self.enable_hypernetwork_modulation = False
        else: self.hyper_bias_generator = None
        if self.enable_internal_sub_processing and self.dim == 4: self.w_modulator = nn.Sequential(nn.Linear(1, 2), nn.GELU(), nn.Linear(2, 1), nn.Sigmoid()); self.v_scaler_param = nn.Parameter(torch.randn(1) * 0.01)
        else: self.w_modulator = None; self.v_scaler_param = None
        self.apply(init_weights_general)
    def _apply_internal_quaternion_sub_processing(self, x_q: torch.Tensor, micro_level_idx: Optional[int], stage_idx: Optional[int]) -> torch.Tensor:
        if not (self.enable_internal_sub_processing and x_q.shape[-1] == 4 and self.w_modulator and self.v_scaler_param is not None): return x_q
        print_tensor_stats_fdq(x_q, "ISP_In_x_q", micro_level_idx, stage_idx); w, v = x_q[..., 0:1], x_q[..., 1:4]; w_modulation_factor = self.w_modulator(w)
        v_scaled = v * w_modulation_factor * (torch.tanh(self.v_scaler_param) + 1.0); x_q_processed = torch.cat([w, v_scaled], dim=-1); x_q_norm = normalize_quaternion(x_q_processed)
        # if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 10 == 0: print_tensor_stats_fdq(w_modulation_factor, "ISP_w_mod", micro_level_idx, stage_idx); print_tensor_stats_fdq(x_q_norm, "ISP_Out_Norm", micro_level_idx, stage_idx)
        return x_q_norm
    def apply_rotation(self, x_tan: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        R_matrix_this_step: Optional[torch.Tensor] = None
        if self.dim <= 0: return x_tan, None
        if self.use_quaternion_so4 and self.rotation_param_generator is not None:
            p_quat, q_quat = self.rotation_param_generator(); p_quat, q_quat = p_quat.to(x_tan.device, x_tan.dtype), q_quat.to(x_tan.device, x_tan.dtype)
            if x_tan.shape[-1] != 4: logger_wubu_diffusion.error(f"Quat rot needs dim 4, got {x_tan.shape[-1]}. Skip."); return x_tan, None
            p_b, q_b = (p_quat.unsqueeze(0).expand_as(x_tan), q_quat.unsqueeze(0).expand_as(x_tan)) if x_tan.dim() > 1 and p_quat.dim() == 1 else (p_quat, q_quat)
            rotated_x_tan = quaternion_multiply(quaternion_multiply(p_b, x_tan), q_b)
        elif self.rotation_generator_matrix is not None and isinstance(self.rotation_generator_matrix, SkewSymmetricMatrix):
            R_matrix_this_step = self.rotation_generator_matrix().to(x_tan.device, x_tan.dtype)
            if x_tan.dim() == 2: rotated_x_tan = torch.einsum('ij,bj->bi', R_matrix_this_step, x_tan)
            elif x_tan.dim() == 1: rotated_x_tan = torch.matmul(R_matrix_this_step, x_tan)
            elif x_tan.dim() == 3: rotated_x_tan = torch.einsum('ij,bnj->bni', R_matrix_this_step, x_tan)
            else: rotated_x_tan = x_tan; logger_wubu_diffusion.warning(f"MicroTrans apply_rot: Unexpected x_tan dim {x_tan.dim()}. Skip.")
        elif isinstance(self.rotation_generator_matrix, nn.Identity): rotated_x_tan = x_tan
        else: rotated_x_tan = x_tan; logger_wubu_diffusion.error("Rot module misconfigured.")
        return rotated_x_tan, R_matrix_this_step
    def _apply_non_rotational_map(self, x_in: torch.Tensor, original_main_tan_for_hypernet: torch.Tensor, micro_level_idx: Optional[int], stage_idx: Optional[int]) -> torch.Tensor:
        if not self.non_rotational_map_layers or isinstance(self.non_rotational_map_layers[0], nn.Identity): return x_in
        x_processed = x_in
        first_linear_layer = self.non_rotational_map_layers[0]
        if self.enable_hypernetwork_modulation and self.hyper_bias_generator is not None and isinstance(first_linear_layer, nn.Linear):
            dynamic_bias_offset = self.hyper_bias_generator(original_main_tan_for_hypernet)
            # if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 10 == 0: print_tensor_stats_fdq(dynamic_bias_offset, "Hyper_BiasOffset", micro_level_idx, stage_idx)
            effective_bias = (first_linear_layer.bias + dynamic_bias_offset * self.hyper_mod_strength) if first_linear_layer.bias is not None else (dynamic_bias_offset * self.hyper_mod_strength)
            x_processed = F.linear(x_processed, first_linear_layer.weight, effective_bias)
            for i in range(1, len(self.non_rotational_map_layers)):
                x_processed = self.non_rotational_map_layers[i](x_processed)
        else:
            temp_x = x_processed 
            for layer in self.non_rotational_map_layers:
                temp_x = layer(temp_x)
            x_processed = temp_x
        return x_processed
    def forward(self, main_tan: torch.Tensor, current_internal_state_in: Optional[torch.Tensor]=None, scaffold_modulation: Optional[torch.Tensor]=None, micro_level_idx: Optional[int]=None, stage_idx: Optional[int]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.debug_counter +=1
        if self.dim <= 0: return main_tan, None, current_internal_state_in
        # print_tensor_stats_fdq(main_tan, "MF_In_main", micro_level_idx, stage_idx);
        x_for_hypernet_mod = main_tan
        x_intermediate = self._apply_internal_quaternion_sub_processing(main_tan, micro_level_idx, stage_idx) if self.enable_internal_sub_processing else main_tan
        # print_tensor_stats_fdq(x_intermediate, "MF_PostISP", micro_level_idx, stage_idx);
        rotated_x, R_matrix = self.apply_rotation(x_intermediate)
        # print_tensor_stats_fdq(rotated_x, "MF_PostRot", micro_level_idx, stage_idx)
        # if R_matrix is not None: print_tensor_stats_fdq(R_matrix, "MF_R_matrix", micro_level_idx, stage_idx)
        mapped_x = self._apply_non_rotational_map(rotated_x, x_for_hypernet_mod, micro_level_idx, stage_idx)
        # print_tensor_stats_fdq(mapped_x, "MF_PostMap", micro_level_idx, stage_idx)
        next_state, final_tan_before_scaffold = current_internal_state_in, mapped_x
        if self.enable_stateful_micro_transform and current_internal_state_in is not None and self.state_influence_mlp and self.state_update_gate_mlp:
            # print_tensor_stats_fdq(current_internal_state_in, "STF_In_State", micro_level_idx, stage_idx);
            influence = self.state_influence_mlp(current_internal_state_in)
            # print_tensor_stats_fdq(influence, "STF_Influence", micro_level_idx, stage_idx);
            final_tan_before_scaffold = mapped_x + influence
            state_update_in = torch.cat([final_tan_before_scaffold, current_internal_state_in], dim=-1); delta = self.state_update_gate_mlp(state_update_in)
            # print_tensor_stats_fdq(delta, "STF_Delta", micro_level_idx, stage_idx);
            next_state = torch.tanh(current_internal_state_in + delta)
            # print_tensor_stats_fdq(next_state, "STF_Out_State", micro_level_idx, stage_idx)
        final_tan = final_tan_before_scaffold
        if scaffold_modulation is not None:
            mod = scaffold_modulation
            if final_tan.dim()>1 and mod.dim()==1 and self.dim>0: mod=mod.unsqueeze(0).expand_as(final_tan)
            elif final_tan.dim()==1 and mod.dim()==2 and mod.shape[0]==1 and self.dim>0: mod=mod.squeeze(0)
            if final_tan.shape==mod.shape: final_tan += mod
            # else: logger_wubu_diffusion.debug(f"Scaffold mod shape mismatch. final_tan: {final_tan.shape}, mod: {mod.shape}")
        # print_tensor_stats_fdq(final_tan, "MF_PostScaff", micro_level_idx, stage_idx);
        clamped_tan = torch.clamp(final_tan, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        # print_tensor_stats_fdq(clamped_tan, "MF_Out_Clamp", micro_level_idx, stage_idx)
        # if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 20 == 0: print("-" * 20)
        return clamped_tan, R_matrix, next_state


class FractalDepthQuaternionWuBuRungs(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_virtual_micro_levels: int = 10000, base_micro_level_dim: int = 4, num_physical_transform_stages: int = 10,
                 initial_s: float = 1.0, s_decay_factor_per_micro_level: float = 0.9999, initial_c_base: float = 1.0, c_phi_influence: bool = True,
                 num_phi_scaffold_points_per_stage: int = 5, phi_scaffold_init_scale_factor: float = 0.05, use_gaussian_rungs: bool = True, # This controls original scaffold logic
                 base_gaussian_std_dev_factor_rung: float = 0.02, gaussian_std_dev_decay_factor_rung: float = 0.99999, rung_affinity_temperature: float = 0.1, # This controls original scaffold affinity
                 rung_modulation_strength: float = 0.05, t_tilde_activation_scale: float = 1.0,
                 # Params for OriginalFractalMicroTransformRungs
                 micro_transform_type: str = "mlp", micro_transform_hidden_factor: float = 1.0,
                 use_quaternion_so4_micro: bool = True, scaffold_co_rotation_mode: str = "none",
                 enable_internal_sub_processing: bool = True, enable_stateful_micro_transform: bool = True,
                 enable_hypernetwork_modulation: bool = True, internal_state_dim_factor: float = 1.0, hyper_mod_strength: float = 0.01,
                 # New Params for ManifoldUncrumpleTransform
                 use_manifold_uncrumple_transform: bool = False, # Main flag to switch
                 uncrumple_hypernet_hidden_factor: float = 1.0,
                 uncrumple_learn_rotation: bool = True,
                 uncrumple_learn_scale: bool = True,
                 uncrumple_scale_activation: str = 'sigmoid',
                 uncrumple_scale_min: float = 0.5,
                 uncrumple_scale_max: float = 2.0,
                 uncrumple_learn_translation: bool = True
                 ):
        super().__init__()
        self.logger_fractal_rungs = logger_wubu_diffusion
        self.num_virtual_micro_levels = num_virtual_micro_levels
        self.base_micro_level_dim = base_micro_level_dim
        self.enable_stateful_micro_transform_globally = enable_stateful_micro_transform # For OriginalFMT
        self.use_manifold_uncrumple_transform = use_manifold_uncrumple_transform

        if self.base_micro_level_dim < 0: self.base_micro_level_dim = 0
        if input_dim < 0: input_dim = 0
        if output_dim < 0: output_dim = 0

        if self.use_manifold_uncrumple_transform:
            self.logger_fractal_rungs.info("FDQRungs will use ManifoldUncrumpleTransform for its physical stages.")
            # Disable original scaffold logic if ManifoldUncrumpleTransform is used, as it has its own.
            self.use_gaussian_rungs = False 
        elif self.base_micro_level_dim != 4:
            if use_quaternion_so4_micro:
                self.logger_fractal_rungs.warning("use_quaternion_so4_micro=True but base_dim!=4. Forced off for OriginalFMT.")
                use_quaternion_so4_micro = False
            if enable_internal_sub_processing:
                self.logger_fractal_rungs.warning("enable_internal_sub_processing=True but base_dim!=4. Forced off for OriginalFMT.")
                enable_internal_sub_processing = False
        
        self.num_physical_transform_stages = max(1, num_physical_transform_stages)
        if self.num_virtual_micro_levels > 0 and self.num_physical_transform_stages > 0:
            self.micro_levels_per_stage = max(1, self.num_virtual_micro_levels // self.num_physical_transform_stages)
        else: self.micro_levels_per_stage = 1

        self.initial_s, self.s_decay_factor = initial_s, s_decay_factor_per_micro_level
        self.initial_c_base, self.c_phi_influence = initial_c_base, c_phi_influence
        self.min_curvature, self.t_tilde_activation_scale = EPS, t_tilde_activation_scale
        self.scaffold_co_rotation_mode = scaffold_co_rotation_mode # For original scaffold logic

        # Input Projection (common for both transform types)
        if input_dim > 0 and self.base_micro_level_dim > 0:
            self.input_projection = nn.Linear(input_dim, self.base_micro_level_dim) if input_dim != self.base_micro_level_dim else nn.Identity()
            self.input_layernorm = nn.LayerNorm(self.base_micro_level_dim)
        elif input_dim == 0 and self.base_micro_level_dim > 0: # Handle 0-dim input by projecting from a dummy 1D input
            self.input_projection = nn.Linear(1, self.base_micro_level_dim)
            self.input_layernorm = nn.LayerNorm(self.base_micro_level_dim)
        else: # base_micro_level_dim is 0 or input_dim is fine
            self.input_projection = nn.Identity()
            self.input_layernorm = nn.Identity()

        # Physical Transforms
        if self.num_physical_transform_stages > 0 and self.base_micro_level_dim > 0:
            if self.use_manifold_uncrumple_transform:
                self.physical_micro_transforms = nn.ModuleList()
                for i in range(self.num_physical_transform_stages):
                    self.physical_micro_transforms.append(
                        ManifoldUncrumpleTransform(
                            dim=self.base_micro_level_dim,
                            num_scaffold_points=num_phi_scaffold_points_per_stage,
                            scaffold_init_scale=phi_scaffold_init_scale_factor,
                            affinity_temp=rung_affinity_temperature, # Use same temp for new transform
                            hypernet_hidden_dim_factor=uncrumple_hypernet_hidden_factor,
                            learn_rotation=uncrumple_learn_rotation,
                            learn_scale=uncrumple_learn_scale,
                            scale_activation=uncrumple_scale_activation,
                            scale_min=uncrumple_scale_min,
                            scale_max=uncrumple_scale_max,
                            learn_translation=uncrumple_learn_translation,
                            stage_info=f"PStage_{i}" # Pass stage index for debug
                        )
                    )
                # Curvature is not explicitly used by ManifoldUncrumpleTransform in current design
                self.log_stage_curvatures_unconstrained = None 
                self.phi_scaffold_base_tangent_vectors = None # MUT handles its own scaffolds
            else: # Original FractalMicroTransformRungs
                self.physical_micro_transforms = nn.ModuleList([
                    OriginalFractalMicroTransformRungs(
                        dim=self.base_micro_level_dim, transform_type=micro_transform_type,
                        hidden_dim_factor=micro_transform_hidden_factor, use_quaternion_so4=use_quaternion_so4_micro,
                        enable_internal_sub_processing=enable_internal_sub_processing,
                        enable_stateful_micro_transform=self.enable_stateful_micro_transform_globally, # Use global flag
                        enable_hypernetwork_modulation=enable_hypernetwork_modulation,
                        internal_state_dim_factor=internal_state_dim_factor, hyper_mod_strength=hyper_mod_strength
                    ) for _ in range(self.num_physical_transform_stages)])
                safe_initial_c_base = max(EPS*10, initial_c_base)
                self.log_stage_curvatures_unconstrained = nn.ParameterList([nn.Parameter(torch.tensor(math.log(math.expm1(safe_initial_c_base)))) for _ in range(self.num_physical_transform_stages)])
                if num_phi_scaffold_points_per_stage > 0:
                     self.phi_scaffold_base_tangent_vectors = nn.ParameterList([nn.Parameter(torch.randn(num_phi_scaffold_points_per_stage, self.base_micro_level_dim) * phi_scaffold_init_scale_factor) for _ in range(self.num_physical_transform_stages)])
                else: self.phi_scaffold_base_tangent_vectors = None
        else:
            self.physical_micro_transforms = nn.ModuleList()
            self.log_stage_curvatures_unconstrained = nn.ParameterList()
            self.phi_scaffold_base_tangent_vectors = None
        
        # Parameters for original scaffold affinity logic (only if not using MUT)
        self.num_phi_scaffold_points_per_stage = num_phi_scaffold_points_per_stage # Store for original logic
        self.use_gaussian_rungs = use_gaussian_rungs # Store for original logic
        self.base_gaussian_std_dev_factor_rung = base_gaussian_std_dev_factor_rung
        self.gaussian_std_dev_decay_factor_rung = gaussian_std_dev_decay_factor_rung
        self.rung_affinity_temperature = max(EPS, rung_affinity_temperature)
        self.rung_modulation_strength = rung_modulation_strength
        
        # Output Projection (common)
        if self.base_micro_level_dim > 0 and output_dim > 0 :
            self.output_projection = nn.Linear(self.base_micro_level_dim, output_dim) if self.base_micro_level_dim != output_dim else nn.Identity()
        elif self.base_micro_level_dim == 0 and output_dim > 0: # Project from dummy 1D if base_dim is 0
            self.output_projection = nn.Linear(1, output_dim)
        else: # Output dim is 0 or matches base_dim
            self.output_projection = nn.Identity()

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        in_feat_proj = self.input_projection.in_features if isinstance(self.input_projection, nn.Linear) else input_dim
        out_feat_proj = self.output_projection.out_features if isinstance(self.output_projection, nn.Linear) else output_dim
        transform_mode_str = "ManifoldUncrumpleTransform" if self.use_manifold_uncrumple_transform else "OriginalFMT"
        self.logger_fractal_rungs.info(f"FDQRungs Init ({transform_mode_str}): In {in_feat_proj}D -> {self.num_virtual_micro_levels} virtLvl ({self.base_micro_level_dim}D) over {self.num_physical_transform_stages} physStgs -> Out {out_feat_proj}D. Params: {param_count:,}.")
        if not self.use_manifold_uncrumple_transform:
             self.logger_fractal_rungs.info(f"  OriginalFMT AdvMicro: SubP={enable_internal_sub_processing}, StateF={self.enable_stateful_micro_transform_globally}, HyperM={enable_hypernetwork_modulation}")


    def get_s_c_gsigma_at_micro_level(self, micro_level_idx: int, stage_idx: int) -> Tuple[float, float, float]:
        s_i = self.initial_s * (self.s_decay_factor ** micro_level_idx); s_i = max(EPS, s_i)
        c_i = self.initial_c_base
        
        if not self.use_manifold_uncrumple_transform and self.log_stage_curvatures_unconstrained and stage_idx < len(self.log_stage_curvatures_unconstrained):
            stage_c_unconstrained = self.log_stage_curvatures_unconstrained[stage_idx]
            c_i = get_constrained_param_val(stage_c_unconstrained, self.min_curvature).item()
            if self.c_phi_influence and self.micro_levels_per_stage > 0 :
                micro_idx_in_stage = micro_level_idx % self.micro_levels_per_stage
                phi_exp = (micro_idx_in_stage % 4) - 1.5 # Cycle through PHI^-1.5, PHI^-0.5, PHI^0.5, PHI^1.5
                c_i *= (PHI ** phi_exp)
        c_i = max(self.min_curvature, c_i)
        
        # Gaussian sigma for affinity (used by both MUT and original scaffold logic)
        sigma_gauss_i_skin = (self.base_gaussian_std_dev_factor_rung / max(s_i, EPS)) * (self.gaussian_std_dev_decay_factor_rung ** micro_level_idx)
        sigma_gauss_i_skin = max(EPS*100, sigma_gauss_i_skin) # Ensure non-zero std dev
        return s_i, c_i, sigma_gauss_i_skin

    def _propagate_scaffold_points(self, base_scaffold_tan_vectors_stage: torch.Tensor, accumulated_R_matrix_stage: Optional[torch.Tensor], current_s_i: float, initial_s_for_stage: float) -> torch.Tensor:
        # This is only for the original scaffold logic
        propagated_scaffold = base_scaffold_tan_vectors_stage
        if accumulated_R_matrix_stage is not None and self.scaffold_co_rotation_mode == "matrix_only":
            propagated_scaffold = torch.einsum('ij,kj->ki', accumulated_R_matrix_stage.to(propagated_scaffold.device, propagated_scaffold.dtype), propagated_scaffold)
        return propagated_scaffold * (initial_s_for_stage / max(current_s_i, EPS))

    def forward(self, x_input: torch.Tensor, show_progress: bool = False, progress_desc: Optional[str] = None) -> torch.Tensor:
        input_original_dim_fwd = x_input.dim(); B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = -1, -1, -1
        if input_original_dim_fwd == 3: B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = x_input.shape; current_v_tan_fwd = x_input.reshape(B_orig_fwd * S_orig_fwd, d_in_runtime_fwd); current_batch_size = B_orig_fwd * S_orig_fwd
        elif input_original_dim_fwd == 2: current_batch_size, d_in_runtime_fwd = x_input.shape; current_v_tan_fwd = x_input; B_orig_fwd, S_orig_fwd = current_batch_size, 1
        elif input_original_dim_fwd == 1 : current_v_tan_fwd = x_input.unsqueeze(0); d_in_runtime_fwd = x_input.shape[0]; B_orig_fwd, S_orig_fwd, current_batch_size = 1,1,1
        else: raise ValueError(f"FDQRungs forward expects 1D, 2D or 3D input, got {input_original_dim_fwd}D.")

        if d_in_runtime_fwd == 0 and self.base_micro_level_dim > 0 and isinstance(self.input_projection, nn.Linear) and self.input_projection.in_features == 1:
            current_v_tan_fwd = torch.ones(current_v_tan_fwd.shape[0], 1, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
        
        current_v_tan_fwd = self.input_projection(current_v_tan_fwd)
        if isinstance(self.input_layernorm, nn.LayerNorm) and self.base_micro_level_dim > 0: # Apply LN only if it's an actual LN layer
            current_v_tan_fwd = self.input_layernorm(current_v_tan_fwd)

        if self.num_virtual_micro_levels == 0 or not self.physical_micro_transforms:
            final_v_tan_fwd = current_v_tan_fwd
        else:
            accumulated_R_for_stage: Optional[torch.Tensor] = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype) if self.base_micro_level_dim > 0 and not self.use_manifold_uncrumple_transform else None
            s_at_stage_start_val: float = self.initial_s
            batched_internal_micro_state: Optional[torch.Tensor] = None # Only for OriginalFMT
            
            # Initialize stateful components for OriginalFMT if enabled
            if not self.use_manifold_uncrumple_transform and \
               self.enable_stateful_micro_transform_globally and \
               self.base_micro_level_dim > 0 and \
               len(self.physical_micro_transforms) > 0 and \
               hasattr(self.physical_micro_transforms[0], 'enable_stateful_micro_transform') and \
               self.physical_micro_transforms[0].enable_stateful_micro_transform and \
               self.physical_micro_transforms[0].internal_state_dim > 0:
                batched_internal_micro_state = torch.zeros(current_batch_size, self.physical_micro_transforms[0].internal_state_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)

            micro_level_iterator_range = range(self.num_virtual_micro_levels)
            in_feat_display = self.input_projection.in_features if hasattr(self.input_projection, 'in_features') and isinstance(self.input_projection, nn.Linear) else d_in_runtime_fwd
            out_feat_display = self.output_projection.out_features if hasattr(self.output_projection, 'out_features') and isinstance(self.output_projection, nn.Linear) else 'N/A' # Use output_dim
            effective_progress_desc = progress_desc if progress_desc else f"FDQRungs ({in_feat_display}->{out_feat_display})"
            
            tqdm_iterator_obj = None
            if show_progress:
                tqdm_iterator_obj = tqdm(micro_level_iterator_range, desc=effective_progress_desc, total=self.num_virtual_micro_levels, leave=False, dynamic_ncols=True, disable=not show_progress)
                micro_level_iterator = tqdm_iterator_obj
            else:
                micro_level_iterator = micro_level_iterator_range

            for micro_i_fwd in micro_level_iterator:
                current_stage_idx_fwd = micro_i_fwd // self.micro_levels_per_stage
                current_stage_idx_fwd = min(current_stage_idx_fwd, len(self.physical_micro_transforms)-1)
                micro_idx_in_stage_fwd = micro_i_fwd % self.micro_levels_per_stage
                physical_transform_module_fwd = self.physical_micro_transforms[current_stage_idx_fwd]

                if micro_idx_in_stage_fwd == 0: # Start of a new physical stage
                    s_at_stage_start_val, _, _ = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)
                    if not self.use_manifold_uncrumple_transform:
                        if self.base_micro_level_dim > 0:
                            accumulated_R_for_stage = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
                        if self.enable_stateful_micro_transform_globally and \
                           self.base_micro_level_dim > 0 and \
                           hasattr(physical_transform_module_fwd, 'enable_stateful_micro_transform') and \
                           physical_transform_module_fwd.enable_stateful_micro_transform and \
                           physical_transform_module_fwd.internal_state_dim > 0:
                            batched_internal_micro_state = torch.zeros(current_batch_size, physical_transform_module_fwd.internal_state_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
                
                s_i_fwd, c_i_fwd, sigma_gauss_skin_i_fwd = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)

                if self.use_manifold_uncrumple_transform:
                    # Pass current sigma for affinity calculation within MUT
                    # stage_idx_override is used for MUT's internal logging if it has a stage_info field
                    transformed_v_tan_micro = physical_transform_module_fwd(current_v_tan_fwd, sigma_gauss_skin_i_fwd, micro_level_idx=micro_i_fwd, stage_idx_override=current_stage_idx_fwd)
                    # ManifoldUncrumpleTransform does not return R_step or next_state in this version
                    R_step, next_state = None, None 
                else: # Original FractalMicroTransformRungs
                    scaffold_modulation_input: Optional[torch.Tensor] = None
                    if self.use_gaussian_rungs and self.phi_scaffold_base_tangent_vectors is not None and \
                       self.num_phi_scaffold_points_per_stage > 0 and self.base_micro_level_dim > 0 and \
                       current_stage_idx_fwd < len(self.phi_scaffold_base_tangent_vectors):
                        
                        base_scaffolds = self.phi_scaffold_base_tangent_vectors[current_stage_idx_fwd].to(current_v_tan_fwd.device, current_v_tan_fwd.dtype)
                        R_prop = accumulated_R_for_stage if self.scaffold_co_rotation_mode == "matrix_only" else None
                        propagated_scaffolds = self._propagate_scaffold_points(base_scaffolds, R_prop, s_i_fwd, s_at_stage_start_val)
                        
                        diffs = current_v_tan_fwd.unsqueeze(1) - propagated_scaffolds.unsqueeze(0)
                        sq_dists = torch.sum(diffs**2, dim=-1)
                        affinity_numerator = -sq_dists / (2 * (sigma_gauss_skin_i_fwd**2) + EPS*100)
                        affinity_numerator_clamped = torch.clamp(affinity_numerator, min=-30.0, max=30.0)
                        affinity_scores = torch.exp(affinity_numerator_clamped)
                        affinity_weights = F.softmax(affinity_scores / self.rung_affinity_temperature, dim=-1)
                        scaffold_modulation_input = self.rung_modulation_strength * torch.einsum('bn,nd->bd', affinity_weights, propagated_scaffolds)
                    
                    transformed_v_tan_micro, R_step, next_state = physical_transform_module_fwd(
                        current_v_tan_fwd, batched_internal_micro_state, scaffold_modulation_input, micro_i_fwd, current_stage_idx_fwd
                    )
                    if self.enable_stateful_micro_transform_globally:
                        batched_internal_micro_state = next_state
                    if R_step is not None and self.scaffold_co_rotation_mode == "matrix_only" and accumulated_R_for_stage is not None:
                        accumulated_R_for_stage = torch.matmul(R_step, accumulated_R_for_stage)

                current_v_tan_fwd = transformed_v_tan_micro * self.t_tilde_activation_scale if self.t_tilde_activation_scale != 1.0 else transformed_v_tan_micro
                
                if tqdm_iterator_obj and hasattr(tqdm_iterator_obj, 'set_postfix') and micro_i_fwd > 0 and (micro_i_fwd % (max(1,self.num_virtual_micro_levels//100))==0):
                    log_dict = {"s":f"{s_i_fwd:.2e}","c":f"{c_i_fwd:.2e}","σ":f"{sigma_gauss_skin_i_fwd:.2e}"}
                    if not self.use_manifold_uncrumple_transform and scaffold_modulation_input is not None and 'affinity_weights' in locals() and affinity_weights.numel()>0 :
                        log_dict["aff_w"] = f"{affinity_weights.max().item():.2e}"
                    if not self.use_manifold_uncrumple_transform and batched_internal_micro_state is not None and batched_internal_micro_state.numel()>0:
                        log_dict["st_n"] = f"{torch.linalg.norm(batched_internal_micro_state).item():.2e}"
                    tqdm_iterator_obj.set_postfix(log_dict)
            
            final_v_tan_fwd = current_v_tan_fwd
            if tqdm_iterator_obj and hasattr(tqdm_iterator_obj, 'close'): tqdm_iterator_obj.close()

        # Output Projection
        if self.base_micro_level_dim == 0 and output_dim > 0 and isinstance(self.output_projection, nn.Linear) and self.output_projection.in_features == 1:
             output_features_fwd = self.output_projection(torch.ones(final_v_tan_fwd.shape[0], 1, device=final_v_tan_fwd.device, dtype=final_v_tan_fwd.dtype))
        else:
            output_features_fwd = self.output_projection(final_v_tan_fwd)
        
        # Reshape back to original batch structure if necessary
        if input_original_dim_fwd == 3 and B_orig_fwd != -1 and S_orig_fwd != -1:
            final_output_dim_val = output_features_fwd.shape[-1] if output_features_fwd.numel() > 0 else 0
            return output_features_fwd.reshape(B_orig_fwd, S_orig_fwd, final_output_dim_val)
        elif input_original_dim_fwd == 2 : return output_features_fwd
        elif input_original_dim_fwd == 1: return output_features_fwd.squeeze(0) if output_features_fwd.shape[0] == 1 and output_features_fwd.dim() > 1 else output_features_fwd # Careful with squeeze
        return output_features_fwd

# --- Dataset (Copied from WuBuCPUOnlyGen_v1.py, can be used as is if num_predict_frames is handled) ---
class VideoFrameDatasetCPU(Dataset): # Largely unchanged
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int],
                 frame_skip: int = 1, data_fraction: float = 1.0,
                 val_fraction: float = 0.0,
                 mode: str = 'train',
                 seed: int = 42,
                 args: argparse.Namespace = None):
        super().__init__(); self.video_path = video_path; self.num_frames_total_sequence = num_frames_total
        self.image_size = image_size; self.frame_skip = frame_skip; self.mode = mode.lower()
        self.logger = logger_wubu_diffusion.getChild(f"Dataset_{self.mode.upper()}_{os.getpid()}"); self.args_ref = args
        self.pid_counters = {}; self.getitem_log_interval = getattr(self.args_ref, 'getitem_log_interval', 100) if self.args_ref else 100
        self.getitem_slow_threshold = getattr(self.args_ref, 'getitem_slow_threshold', 1.0) if self.args_ref else 1.0
        if not os.path.isfile(self.video_path): self.logger.error(f"Video file not found: {self.video_path}"); raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.video_frames_in_ram = None; self.source_video_fps = 30.0; read_success = False
        if IMAGEIO_AVAILABLE and imageio is not None:
            try:
                self.logger.info(f"Attempting to load video into RAM using imageio: {self.video_path}"); reader = imageio.get_reader(self.video_path, 'ffmpeg')
                meta = reader.get_meta_data(); self.source_video_fps = meta.get('fps', 30.0); frames_list = []
                for frame_np in reader:
                    if frame_np.ndim == 3 and frame_np.shape[-1] in [3,4]: frame_np = np.transpose(frame_np[...,:3], (2,0,1))
                    elif frame_np.ndim == 2: frame_np = np.expand_dims(frame_np, axis=0)
                    else: self.logger.warning(f"Unexpected frame shape from imageio: {frame_np.shape}"); continue
                    frames_list.append(torch.from_numpy(frame_np.copy()))
                self.video_frames_in_ram = torch.stack(frames_list).contiguous() if frames_list else None; reader.close()
                if self.video_frames_in_ram is not None and self.video_frames_in_ram.shape[0] > 0 : read_success = True
            except Exception as e_ii: self.logger.error(f"imageio failed to read {self.video_path}: {e_ii}", exc_info=True)
        if not read_success: raise RuntimeError(f"Failed to load video '{self.video_path}'. Check imageio and ffmpeg installation.")
        if self.video_frames_in_ram is None or self.video_frames_in_ram.shape[0] == 0: raise RuntimeError(f"Video '{self.video_path}' loaded 0 frames.")
        ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3); self.logger.info(f"Loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. Est RAM: {ram_usage_gb:.2f} GB.")
        self.resize_transform = T.Resize(self.image_size, antialias=True); self.normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.num_disk_frames = self.video_frames_in_ram.shape[0]; all_samples_start_indices = []
        # Ensure num_frames_total_sequence is at least 1
        effective_num_frames_total_sequence = max(1, self.num_frames_total_sequence)
        required_span_on_disk = (effective_num_frames_total_sequence - 1) * self.frame_skip + 1
        
        if self.num_disk_frames >= required_span_on_disk:
            for i in range(self.num_disk_frames - required_span_on_disk + 1): all_samples_start_indices.append(i)
        else: self.logger.warning(f"Not enough frames ({self.num_disk_frames}) in '{self.video_path}' for sequence (needs {required_span_on_disk} for seq_len {effective_num_frames_total_sequence}). Dataset for mode '{self.mode}' will be empty."); self.samples_start_indices = []
        
        if not all_samples_start_indices: 
            # If sequence length is 1, but not enough frames, this logic might still lead to empty all_samples_start_indices
            # However, if effective_num_frames_total_sequence is 1, required_span_on_disk is 1.
            # If num_disk_frames >= 1, loop range(self.num_disk_frames) should work.
            self.logger.error(f"No valid samples from '{self.video_path}' for any split (effective_seq_len={effective_num_frames_total_sequence}, disk_frames={self.num_disk_frames})."); self.samples_start_indices = []
        else:
            rng_dataset = random.Random(seed); rng_dataset.shuffle(all_samples_start_indices)
            if val_fraction > 0.0 and val_fraction < 1.0:
                num_total_samples = len(all_samples_start_indices); num_val_samples = int(num_total_samples * val_fraction)
                # Ensure at least one training sample if possible, and num_val_samples is not too large
                num_val_samples = max(0, min(num_val_samples, num_total_samples - (1 if num_total_samples > num_val_samples else 0) ))

                if self.mode == 'train':
                    train_indices = all_samples_start_indices[num_val_samples:]
                    if data_fraction < 1.0 and len(train_indices) > 1: num_train_to_keep = max(1, int(len(train_indices) * data_fraction)); self.samples_start_indices = train_indices[:num_train_to_keep]; self.logger.info(f"Using {data_fraction*100:.2f}% of available training samples: {len(self.samples_start_indices)} samples.")
                    else: self.samples_start_indices = train_indices
                elif self.mode == 'val': self.samples_start_indices = all_samples_start_indices[:num_val_samples]
                else: raise ValueError(f"Invalid mode '{self.mode}'. Must be 'train' or 'val'.")
            else: # No validation split from this dataset instance
                self.samples_start_indices = all_samples_start_indices
                if self.mode == 'train' and data_fraction < 1.0 and len(self.samples_start_indices) > 1: 
                    num_to_keep = max(1, int(len(self.samples_start_indices) * data_fraction))
                    self.samples_start_indices = self.samples_start_indices[:num_to_keep]
                    self.logger.info(f"Using {data_fraction*100:.2f}% of available samples (no val split): {len(self.samples_start_indices)} samples.")
                elif self.mode == 'val': # If val_fraction is 0 or >=1, val set is empty unless it's the only set
                     self.samples_start_indices = [] if val_fraction > 0 else all_samples_start_indices # If val_fraction is 0, it's all data, but mode='val' means it *is* the val set
                                                                                                     # If val_fraction >= 1, also implies use all data for val.
                                                                                                     # This logic might need refinement based on intent.
                                                                                                     # Usually, if val_fraction=0, val set is empty.
                                                                                                     # If val_fraction=1, train set is empty.
                     if val_fraction == 0.0 : self.samples_start_indices = []


        if not self.samples_start_indices and self.mode == 'train': self.logger.error(f"VideoFrameDataset (TRAIN): No valid samples from '{self.video_path}'. Training might fail.")
        elif not self.samples_start_indices and self.mode == 'val' and val_fraction > 0.0 : self.logger.warning(f"VideoFrameDataset (VAL): No valid samples from '{self.video_path}' for validation split. Validation will be skipped.")
        self.logger.info(f"VideoFrameDatasetCPU ({self.mode.upper()}) Init. DiskFrames:{self.num_disk_frames}. NumSamples:{len(self.samples_start_indices)}. SeqLen:{effective_num_frames_total_sequence} (skip {self.frame_skip}).")

    def __len__(self) -> int:
        return len(self.samples_start_indices)

    def __getitem__(self, idx: int) -> torch.Tensor: # Returns a sequence of frames
        t_start_getitem = time.time()
        effective_num_frames_total_sequence = max(1, self.num_frames_total_sequence) # Ensure at least 1

        if not self.samples_start_indices or idx >= len(self.samples_start_indices):
            self.logger.error(f"Index {idx} out of bounds for {self.mode} dataset with {len(self.samples_start_indices)} samples. Returning zeros.")
            # t_end_getitem = time.time(); duration = t_end_getitem - t_start_getitem; # Duration calc moved
            # self.logger.error(f"__getitem__ (idx {idx}, worker {os.getpid()}, ERROR_OOB) from '{Path(self.video_path).name}' took {duration:.4f}s.")
            return torch.zeros((effective_num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        start_frame_idx_in_ram = self.samples_start_indices[idx]
        frames_for_sample = []
        for i in range(effective_num_frames_total_sequence):
            actual_frame_idx_in_ram = start_frame_idx_in_ram + i * self.frame_skip
            if actual_frame_idx_in_ram < self.num_disk_frames:
                try:
                    frame_tensor_chw_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram]
                    if frame_tensor_chw_uint8.shape[0] == 1: # Grayscale
                        frame_tensor_chw_uint8 = frame_tensor_chw_uint8.repeat(3,1,1)
                    elif frame_tensor_chw_uint8.shape[0] != 3 : # Other channel numbers
                        self.logger.warning(f"Frame {actual_frame_idx_in_ram} has {frame_tensor_chw_uint8.shape[0]} channels. Using first 3 or repeating first if <3.")
                        if frame_tensor_chw_uint8.shape[0] > 3:
                            frame_tensor_chw_uint8 = frame_tensor_chw_uint8[:3,...]
                        else: # < 3 channels, e.g. 2. This case is tricky. For simplicity, repeat first.
                            frame_tensor_chw_uint8 = frame_tensor_chw_uint8[0:1,...].repeat(3,1,1)
                    
                    # Ensure frame is C, H, W before resize
                    if frame_tensor_chw_uint8.ndim != 3 or frame_tensor_chw_uint8.shape[0] !=3 :
                        self.logger.error(f"Frame {actual_frame_idx_in_ram} has unexpected shape {frame_tensor_chw_uint8.shape} after channel adjustment. Padding with zeros.")
                        frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
                        continue

                    transformed_frame = self.normalize_transform(self.resize_transform(frame_tensor_chw_uint8).float()/255.0)
                    frames_for_sample.append(transformed_frame)
                except Exception as e:
                    self.logger.error(f"Error transforming frame {actual_frame_idx_in_ram} (sample {idx}): {e}", exc_info=True)
                    frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
            else:
                self.logger.error(f"Frame index {actual_frame_idx_in_ram} out of bounds for disk_frames {self.num_disk_frames} (sample {idx}). Padding with zeros.")
                frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
        
        if not frames_for_sample: # Should not happen if samples_start_indices is valid and num_frames_total_sequence >= 1
            self.logger.error(f"No frames collected for sample {idx}. Returning zeros.")
            return torch.zeros((effective_num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        result_frames = torch.stack(frames_for_sample)
        
        if result_frames.shape[0] != effective_num_frames_total_sequence: # Should ideally not be needed if loop runs correctly
            self.logger.error(f"Loaded {result_frames.shape[0]} frames, expected {effective_num_frames_total_sequence} for sample {idx}. Final shape: {result_frames.shape}. This indicates an issue in frame collection loop.")
            # Fallback to return correctly shaped zero tensor if something went very wrong
            return torch.zeros((effective_num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        t_end_getitem = time.time(); duration = t_end_getitem - t_start_getitem
        if self.args_ref: # Log getitem duration and count
            pid = os.getpid();
            if pid not in self.pid_counters: self.pid_counters[pid] = 0
            self.pid_counters[pid] += 1; current_worker_call_count = self.pid_counters[pid]; log_this_call = False; log_reason = ""
            if self.getitem_slow_threshold > 0 and duration > self.getitem_slow_threshold: log_this_call = True; log_reason = f"SLOW ({duration:.4f}s > {self.getitem_slow_threshold:.2f}s)"
            elif self.getitem_log_interval > 0 and current_worker_call_count % self.getitem_log_interval == 0: log_this_call = True; log_reason = f"periodic (call #{current_worker_call_count})"
            
            if log_this_call: # Changed from self.logger.info to self.logger.debug for less verbosity by default
                self.logger.debug(f"__getitem__ (idx {idx}, worker {pid}, {log_reason}) from '{Path(self.video_path).name}' took {duration:.4f}s.")
        return result_frames
# --- GAAD and Image Processing Utilities (Copied from WuBuCPUOnlyGen_v1.py) ---
# ... (golden_subdivide_rect_fixed_n_cpu, RegionalPatchExtractorCPU, ImageAssemblyUtilsCPU remain unchanged, kept for brevity) ...
def golden_subdivide_rect_fixed_n_cpu(frame_dims:Tuple[int,int], num_regions_target:int, dtype=torch.float, min_size_px=5) -> torch.Tensor:
    W, H = frame_dims; device = torch.device('cpu'); all_rects = [[0.0,0.0,float(W),float(H)]]; rect_queue = deque([(0.0,0.0,float(W),float(H),0)])
    while rect_queue and len(all_rects) < num_regions_target * 3 : # Generate more rects than needed initially
        x_off, y_off, w_curr, h_curr, depth = rect_queue.popleft()
        if min(w_curr, h_curr) < min_size_px or depth > 7 : continue # Depth limit to prevent excessive recursion

        is_landscape = w_curr > h_curr + EPS
        is_portrait = h_curr > w_curr + EPS
        is_square_like = abs(w_curr - h_curr) < EPS
        children_coords = []
        r1_w, r2_w = 0.0, 0.0
        r1_h, r2_h = 0.0, 0.0

        if is_landscape:
            cut_w = w_curr / PHI
            r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px: children_coords.append({'x':x_off, 'y':y_off, 'w':r1_w, 'h':h_curr})
            if r2_w >= min_size_px: children_coords.append({'x':x_off + r1_w, 'y':y_off, 'w':r2_w, 'h':h_curr})
        elif is_portrait:
            cut_h = h_curr / PHI
            r1_h, r2_h = cut_h, h_curr - cut_h
            if r1_h >= min_size_px: children_coords.append({'x':x_off, 'y':y_off, 'w':w_curr, 'h':r1_h})
            if r2_h >= min_size_px: children_coords.append({'x':x_off, 'y':y_off + r1_h, 'w':w_curr, 'h':r2_h})
        elif is_square_like and w_curr > min_size_px * PHI : # Default to landscape-style cut for squares
            cut_w = w_curr / PHI
            r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px: children_coords.append({'x':x_off, 'y':y_off, 'w':r1_w, 'h':h_curr})
            if r2_w >= min_size_px: children_coords.append({'x':x_off + r1_w, 'y':y_off, 'w':r2_w, 'h':h_curr})
        # If none of the above conditions are met (e.g., too small square), children_coords remains empty

        for child_d in children_coords:
            all_rects.append([child_d['x'],child_d['y'],child_d['x']+child_d['w'],child_d['y']+child_d['h']])
            rect_queue.append((child_d['x'],child_d['y'],child_d['w'],child_d['h'],depth+1))

    unique_valid_rects_tensors = []; seen_hashes = set()
    for r_coords in all_rects:
        if r_coords[0] >= r_coords[2] - EPS or r_coords[1] >= r_coords[3] - EPS: continue # Skip zero-area rects
        r_tensor = torch.tensor(r_coords, dtype=dtype, device=device); r_hashable = tuple(round(c, 2) for c in r_coords) # Hash rounded coords
        if r_hashable not in seen_hashes: unique_valid_rects_tensors.append(r_tensor); seen_hashes.add(r_hashable)
    
    # Sort by area (largest first) and select target number
    unique_valid_rects_tensors.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
    selected_rects = unique_valid_rects_tensors[:num_regions_target]
    
    # Pad if not enough unique regions were generated
    if not selected_rects and num_regions_target > 0: # If no rects at all, use full frame
        selected_rects = [torch.tensor([0.0,0.0,float(W),float(H)],dtype=dtype,device=device)]
    if len(selected_rects) < num_regions_target:
        padding_box = selected_rects[-1].clone() if selected_rects else torch.tensor([0.0,0.0,float(W),float(H)],dtype=dtype,device=device)
        selected_rects.extend([padding_box.clone() for _ in range(num_regions_target - len(selected_rects))])
    
    return torch.stack(selected_rects)

class RegionalPatchExtractorCPU(nn.Module):
    def __init__(self, patch_output_size: Tuple[int, int]):
        super().__init__(); self.patch_output_size = patch_output_size
        if patch_output_size[0] > 0 and patch_output_size[1] > 0: 
            self.resize_transform = T.Resize(self.patch_output_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        else: 
            logger_wubu_diffusion.warning("RegionalPatchExtractorCPU: patch_output_size has a zero dimension. Resize transform disabled.")
            self.resize_transform = None

    def forward(self, images: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
        # images: B_eff, C, H_img, W_img (where B_eff is typically B_train * N_frames_in_sequence)
        # bboxes_batch: B_eff, N_regions, 4
        B_img, NumCh, H_img, W_img = images.shape
        device, dtype = images.device, images.dtype
        all_patches_for_batch = [] # List to hold patch tensors for each image in B_img

        patch_h_out, patch_w_out = self.patch_output_size
        # Create a default zero patch once if needed
        default_zero_patch = None
        if patch_h_out <= 0 or patch_w_out <= 0:
            default_zero_patch = torch.zeros((NumCh, max(1, patch_h_out), max(1, patch_w_out)), device=device, dtype=dtype)
        
        for i in range(B_img): # Iterate over each image in the effective batch
            img_item = images[i] # C, H_img, W_img
            bboxes_for_item = bboxes_batch[i] # N_regions, 4
            current_img_patches_list = []

            for r in range(bboxes_for_item.shape[0]): # Iterate over regions for that image
                x1,y1,x2,y2 = bboxes_for_item[r].round().int().tolist()
                x1c,y1c,x2c,y2c = max(0,x1),max(0,y1),min(W_img,x2),min(H_img,y2)

                if x1c>=x2c or y1c>=y2c or patch_h_out<=0 or patch_w_out<=0:
                    # Use pre-created zero patch if output size is invalid, or create one if not pre-created (should not happen if logic correct)
                    patch = default_zero_patch if default_zero_patch is not None else torch.zeros((NumCh,max(1,patch_h_out),max(1,patch_w_out)),device=device,dtype=dtype)
                else:
                    patch_candidate = img_item[:,y1c:y2c,x1c:x2c] # C, h_roi, w_roi
                    if self.resize_transform:
                        patch = self.resize_transform(patch_candidate)
                    # If no resize_transform but shapes don't match (e.g. output size is 0), use zero patch
                    elif patch_candidate.shape[1]!=patch_h_out or patch_candidate.shape[2]!=patch_w_out:
                        patch = default_zero_patch if default_zero_patch is not None else torch.zeros((NumCh,patch_h_out,patch_w_out),device=device,dtype=dtype)
                    else:
                        patch = patch_candidate # No resize needed, shapes match
                current_img_patches_list.append(patch)
            
            all_patches_for_batch.append(torch.stack(current_img_patches_list)) # Stack patches for current image: [N_regions, C, patch_h, patch_w]
        
        return torch.stack(all_patches_for_batch) # Stack for all images in batch: [B_eff, N_regions, C, patch_h, patch_w]

class ImageAssemblyUtilsCPU:
    @staticmethod
    def assemble_frames_from_patches(patches_batch:torch.Tensor, bboxes_batch:torch.Tensor, target_img_size:Tuple[int,int], out_range:Tuple[float,float]=(-1.0,1.0)) -> torch.Tensor:
        # patches_batch: B, N_frames_to_assemble, N_regions, C, H_patch, W_patch
        # bboxes_batch: B, N_frames_to_assemble, N_regions, 4
        B,N_f,N_r,C,H_patch,W_patch = patches_batch.shape; H_img,W_img = target_img_size; device,dtype = patches_batch.device, patches_batch.dtype
        all_frames = torch.zeros(B,N_f,C,H_img,W_img,device=device,dtype=dtype)
        for b_idx in range(B):
            for f_idx in range(N_f):
                canvas = torch.zeros(C,H_img,W_img,device=device,dtype=dtype)
                count_map = torch.zeros(1,H_img,W_img,device=device,dtype=dtype) # Single channel for count
                for r_idx in range(N_r):
                    patch = patches_batch[b_idx,f_idx,r_idx]; 
                    x1,y1,x2,y2 = bboxes_batch[b_idx,f_idx,r_idx].float().round().int().tolist(); 
                    x1c,y1c,x2c,y2c = max(0,x1),max(0,y1),min(W_img,x2),min(H_img,y2)
                    
                    if x1c>=x2c or y1c>=y2c: continue # Skip if region is invalid or outside bounds
                    
                    target_h_on_canvas, target_w_on_canvas = y2c-y1c, x2c-x1c
                    if target_h_on_canvas<=0 or target_w_on_canvas<=0: continue

                    # Resize patch if its original dimensions (H_patch, W_patch) differ from target bbox size on canvas
                    if (H_patch != target_h_on_canvas or W_patch != target_w_on_canvas) and H_patch > 0 and W_patch > 0 :
                        resized_patch = TF.resize(patch, [target_h_on_canvas, target_w_on_canvas], antialias=True)
                    elif H_patch == target_h_on_canvas and W_patch == target_w_on_canvas:
                        resized_patch = patch # No resize needed
                    else: # H_patch or W_patch is 0, or other invalid case - shouldn't happen if patches are valid
                        continue 

                    canvas[:,y1c:y2c,x1c:x2c] += resized_patch; 
                    count_map[:,y1c:y2c,x1c:x2c] += 1.0
                
                # Average overlapping regions
                # Ensure count_map has same C dim as canvas for where, or use broadcasting
                all_frames[b_idx,f_idx] = torch.where(count_map.expand_as(canvas)>0, canvas/(count_map.expand_as(canvas)+EPS), canvas)
        
        return torch.clamp(all_frames,min=out_range[0],max=out_range[1]) if out_range else all_frames


# --- Diffusion Model Components ---
# ... (DiffusionProcess, SinusoidalPosEmb remain unchanged, kept for brevity) ...
def _extract_into_tensor(arr_or_list, timesteps, broadcast_shape):
    """Extract values from a 1-D numpy array or list for specified timesteps."""
    if isinstance(arr_or_list, list): res = torch.tensor(arr_or_list, device=timesteps.device)[timesteps].float()
    else: res = arr_or_list.to(timesteps.device)[timesteps].float() # Assumes arr is already a tensor
    while len(res.shape) < len(broadcast_shape): res = res[..., None]
    return res.expand(broadcast_shape)

class DiffusionProcess:
    def __init__(self, timesteps: int, beta_schedule: str = 'linear', beta_start: float = 0.0001, beta_end: float = 0.02, device: torch.device = torch.device("cpu")):
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.device = device

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) 

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        logger_wubu_diffusion.info(f"DiffusionProcess: {timesteps} steps, schedule '{beta_schedule}', beta_start={beta_start:.2e}, beta_end={beta_end:.2e}")

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas_cumprod_t = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps

    def p_mean_variance(self, model_output_eps: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_xstart = self._predict_xstart_from_eps(x_t, t, eps=model_output_eps)
        if clip_denoised: pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)
        posterior_mean_coef1_t = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * pred_xstart + posterior_mean_coef2_t * x_t
        posterior_variance_t = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t

    def p_sample(self, model_output_eps: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        posterior_mean, _, posterior_log_variance_clipped_t = self.p_mean_variance(model_output_eps, x_t, t, clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return posterior_mean + nonzero_mask * (0.5 * posterior_log_variance_clipped_t).exp() * noise

    @torch.no_grad()
    def sample(self, model_callable: callable, batch_size: int, image_channels: int, image_size: Tuple[int,int],
               patch_h: int, patch_w: int, num_regions: int,
               num_sampling_steps: Optional[int] = None,
               show_progress: bool = True, progress_desc: str = "Diffusion Sampling") -> torch.Tensor:
        img_patches_shape_flat = num_regions * image_channels * patch_h * patch_w
        x_t_patches_flat = torch.randn(batch_size, img_patches_shape_flat, device=self.device)
        bboxes_per_sample_list = []
        for _ in range(batch_size):
             bboxes_per_sample_list.append(golden_subdivide_rect_fixed_n_cpu(image_size, num_regions, dtype=torch.float32, min_size_px=max(1,min(patch_h,patch_w)//2)))
        sampling_bboxes = torch.stack(bboxes_per_sample_list).to(self.device)
        actual_timesteps_to_iterate = self.timesteps
        if num_sampling_steps is not None and num_sampling_steps > 0 and num_sampling_steps < self.timesteps:
            time_seq = torch.linspace(self.timesteps - 1, 0, num_sampling_steps, device=self.device).long()
            time_seq = torch.unique(time_seq, sorted=True).flip(dims=[0])
            iterable_range = time_seq
            actual_timesteps_to_iterate = len(time_seq)
            # logger_wubu_diffusion.debug(f"Sampling with reduced steps: {actual_timesteps_to_iterate} (target: {num_sampling_steps}) over timesteps: {time_seq[:3]}...{time_seq[-3:]}")
        else:
            iterable_range = reversed(range(0, self.timesteps))
            actual_timesteps_to_iterate = self.timesteps
        iterable_progress = iterable_range
        if show_progress:
            iterable_progress = tqdm(iterable_range, desc=progress_desc, total=actual_timesteps_to_iterate, leave=False, dynamic_ncols=True, disable=not show_progress)
        for i_val_t in iterable_progress:
            current_timestep_val = i_val_t.item() if torch.is_tensor(i_val_t) else i_val_t
            t = torch.full((batch_size,), current_timestep_val, device=self.device, dtype=torch.long)
            predicted_noise_patches_flat = model_callable(x_t_patches_flat, t)
            x_t_prev_patches_flat = self.p_sample(predicted_noise_patches_flat, x_t_patches_flat, t, clip_denoised=True)
            x_t_patches_flat = x_t_prev_patches_flat
        x_0_patches = x_t_patches_flat.reshape(batch_size, num_regions, image_channels, patch_h, patch_w)
        x_0_image = ImageAssemblyUtilsCPU.assemble_frames_from_patches(
            x_0_patches.unsqueeze(1), sampling_bboxes.unsqueeze(1), image_size
        )
        return x_0_image

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        if half_dim <=0 : # Handle dim=0 or 1 case for embedding
             if self.dim == 1: return torch.zeros_like(time.unsqueeze(-1).float()) # Return 0 for dim 1
             return torch.empty(time.shape[0], 0, device=device) # Return empty for dim 0
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1 and self.dim > 1: # zero pad if dim is odd and > 1
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

class WuBuDiffusionModel(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict):
        super().__init__()
        self.args = args
        self.num_channels = video_config['num_channels']
        self.num_regions = gaad_config['num_regions']
        self.patch_h, self.patch_w = args.patch_size_h, args.patch_size_w

        patch_feat_flat_one_frame = self.num_regions * self.num_channels * self.patch_h * self.patch_w
        patch_feat_flat_one_frame = max(1, patch_feat_flat_one_frame)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(args.diffusion_time_embed_dim),
            nn.Linear(args.diffusion_time_embed_dim, args.diffusion_time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(args.diffusion_time_embed_dim * 4, args.diffusion_time_embed_dim),
        ) if args.diffusion_time_embed_dim > 0 else None

        wubu_input_dim_for_stack = patch_feat_flat_one_frame
        if self.time_mlp is not None and args.diffusion_time_embed_dim > 0: # Check dim > 0
            wubu_input_dim_for_stack += args.diffusion_time_embed_dim
        wubu_input_dim_for_stack = max(1, wubu_input_dim_for_stack)

        wubu_output_dim_from_stack = patch_feat_flat_one_frame

        self.wubu_noise_predictor_stack = FractalDepthQuaternionWuBuRungs(
            input_dim=wubu_input_dim_for_stack,
            output_dim=args.bsp_intermediate_dim if args.use_bsp_gate else wubu_output_dim_from_stack,
            num_virtual_micro_levels=args.model_wubu_num_virtual_levels,
            base_micro_level_dim=args.model_wubu_base_micro_dim,
            num_physical_transform_stages=args.model_wubu_num_physical_stages,
            initial_s=args.wubu_initial_s, s_decay_factor_per_micro_level=args.wubu_s_decay,
            initial_c_base=args.wubu_initial_c, c_phi_influence=args.wubu_c_phi_influence,
            num_phi_scaffold_points_per_stage=args.wubu_num_phi_scaffold_points,
            phi_scaffold_init_scale_factor=args.wubu_phi_scaffold_init_scale,
            use_gaussian_rungs= (not args.use_manifold_uncrumple_transform) and args.wubu_use_gaussian_rungs, # Conditional
            base_gaussian_std_dev_factor_rung=args.wubu_base_gaussian_std_dev,
            gaussian_std_dev_decay_factor_rung=args.wubu_gaussian_std_dev_decay,
            rung_affinity_temperature=args.wubu_rung_affinity_temp, # Used by MUT too
            rung_modulation_strength=args.wubu_rung_modulation_strength,
            t_tilde_activation_scale=args.wubu_t_tilde_scale,
            micro_transform_type=args.wubu_micro_transform_type,
            micro_transform_hidden_factor=args.wubu_micro_transform_hidden_factor,
            use_quaternion_so4_micro=args.wubu_use_quaternion_so4,
            scaffold_co_rotation_mode=args.wubu_scaffold_co_rotation_mode,
            enable_internal_sub_processing=args.wubu_enable_isp,
            enable_stateful_micro_transform=args.wubu_enable_stateful,
            enable_hypernetwork_modulation=args.wubu_enable_hypernet,
            internal_state_dim_factor=args.wubu_internal_state_factor,
            hyper_mod_strength=args.wubu_hypernet_strength,
            # New MUT params
            use_manifold_uncrumple_transform=args.use_manifold_uncrumple_transform,
            uncrumple_hypernet_hidden_factor=args.uncrumple_hypernet_hidden_factor,
            uncrumple_learn_rotation=args.uncrumple_learn_rotation,
            uncrumple_learn_scale=args.uncrumple_learn_scale,
            uncrumple_scale_activation=args.uncrumple_scale_activation,
            uncrumple_scale_min=args.uncrumple_scale_min,
            uncrumple_scale_max=args.uncrumple_scale_max,
            uncrumple_learn_translation=args.uncrumple_learn_translation
        )

        self.use_bsp_gate = args.use_bsp_gate
        if self.use_bsp_gate:
            logger_wubu_diffusion.info("BSP Gate is ENABLED.")
            bsp_gate_input_dim = patch_feat_flat_one_frame
            bsp_gate_hidden_dim = max(args.bsp_gate_hidden_dim, 1)
            if bsp_gate_input_dim <= 0:
                 logger_wubu_diffusion.warning(f"BSP Gate input dim is {bsp_gate_input_dim}. Setting to 1.")
                 bsp_gate_input_dim = 1
            self.bsp_gate_mlp = nn.Sequential(
                nn.Linear(bsp_gate_input_dim, bsp_gate_hidden_dim), nn.GELU(),
                nn.Linear(bsp_gate_hidden_dim, args.bsp_num_paths),
            )
            self.bsp_output_heads = nn.ModuleList()
            for _ in range(args.bsp_num_paths):
                self.bsp_output_heads.append(
                    nn.Linear(args.bsp_intermediate_dim, wubu_output_dim_from_stack)
                )
            logger_wubu_diffusion.info(f"BSP Gate: Input {bsp_gate_input_dim}D -> MLP -> {args.bsp_num_paths} path logits.")
            logger_wubu_diffusion.info(f"BSP Output Heads: {args.bsp_num_paths} heads, each {args.bsp_intermediate_dim}D -> {wubu_output_dim_from_stack}D.")
        else:
            logger_wubu_diffusion.info("BSP Gate is DISABLED.")
            # If BSP is not used, FDQWR's output_dim should directly be wubu_output_dim_from_stack
            # This is handled by the conditional output_dim in FDQWR init.
            # The warning below might be redundant if FDQWR init logic is correct.
            if hasattr(args, 'bsp_intermediate_dim') and args.bsp_intermediate_dim != wubu_output_dim_from_stack:
                 logger_wubu_diffusion.debug(f"BSP not used, bsp_intermediate_dim ({args.bsp_intermediate_dim}) differs from wubu_output_dim_from_stack ({wubu_output_dim_from_stack}). FDQWR output_dim should be {wubu_output_dim_from_stack}.")

        self.apply(init_weights_general)
        logger_wubu_diffusion.info(f"WuBuDiffusionModel: WuBu Main Stack Input Dim: {wubu_input_dim_for_stack}D.")
        if self.use_bsp_gate:
            logger_wubu_diffusion.info(f"  WuBu Main Stack Output (Intermediate for BSP): {args.bsp_intermediate_dim}D.")
        else:
            logger_wubu_diffusion.info(f"  WuBu Main Stack Output (Final Noise): {wubu_output_dim_from_stack}D.")

    def forward(self, x_t_flat_patches: torch.Tensor, time_t: torch.Tensor, current_global_step: Optional[int] = None) -> torch.Tensor:
        wubu_stack_input = x_t_flat_patches
        if self.time_mlp and self.args.diffusion_time_embed_dim > 0:
            time_emb = self.time_mlp(time_t)
            if x_t_flat_patches.shape[0] != time_emb.shape[0]: # Handle batch size mismatches (e.g. during sampling)
                if x_t_flat_patches.shape[0] == 1 and time_emb.shape[0] > 1: # Sampling single, time_emb from training batch
                    time_emb = time_emb[0:1, :].expand(x_t_flat_patches.shape[0], -1)
                elif time_emb.shape[0] == 1 and x_t_flat_patches.shape[0] > 1: # time_emb for single, input is batch
                    time_emb = time_emb.expand(x_t_flat_patches.shape[0], -1)
                else: # General mismatch, log warning and try to expand (might be risky)
                    logger_wubu_diffusion.warning(f"Time embedding batch size {time_emb.shape[0]} mismatch with input patches batch size {x_t_flat_patches.shape[0]}. Expanding time_emb.")
                    time_emb = time_emb.expand(x_t_flat_patches.shape[0], -1) # This might fail if not expandable from 1
            wubu_stack_input = torch.cat((x_t_flat_patches, time_emb), dim=-1)

        if wubu_stack_input.shape[1] == 0 and \
           isinstance(self.wubu_noise_predictor_stack.input_projection, nn.Linear) and \
           self.wubu_noise_predictor_stack.input_projection.in_features == 1:
             wubu_stack_input = torch.ones(x_t_flat_patches.shape[0], 1, device=x_t_flat_patches.device, dtype=x_t_flat_patches.dtype)

        intermediate_or_final_features = self.wubu_noise_predictor_stack(
            wubu_stack_input,
            show_progress=getattr(self.args, 'show_train_progress_bar', False)
        )

        if self.use_bsp_gate:
            gate_input_for_bsp = x_t_flat_patches
            if gate_input_for_bsp.shape[1] == 0 and isinstance(self.bsp_gate_mlp[0], nn.Linear) and self.bsp_gate_mlp[0].in_features == 1:
                gate_input_for_bsp = torch.ones(x_t_flat_patches.shape[0], 1, device=x_t_flat_patches.device, dtype=x_t_flat_patches.dtype)
            path_logits = self.bsp_gate_mlp(gate_input_for_bsp)
            if self.args.bsp_num_paths == 1: path_weights = torch.ones_like(path_logits)
            elif self.args.bsp_num_paths == 2 and self.args.bsp_gate_activation == 'sigmoid':
                path_weights_p1 = torch.sigmoid(path_logits[:,0:1]); path_weights_p0 = 1.0 - path_weights_p1
                path_weights = torch.cat([path_weights_p0, path_weights_p1], dim=-1)
            else: path_weights = F.softmax(path_logits, dim=-1)
            
            outputs_from_heads = torch.stack([head(intermediate_or_final_features) for head in self.bsp_output_heads], dim=0) # (Ns, B, NoiseDim)
            weights_reshaped = path_weights.permute(1,0).unsqueeze(-1) # (Ns, B, 1)
            predicted_noise_flat_patches = torch.sum(weights_reshaped * outputs_from_heads, dim=0) # (B, NoiseDim)

            if self.args.wandb and WANDB_AVAILABLE and wandb.run and \
               current_global_step is not None and \
               current_global_step % (self.args.log_interval * getattr(self.args, 'bsp_log_gate_interval_mult', 5)) == 0:
                 for p_idx in range(min(self.args.bsp_num_paths, 16)): # Log only first few paths if too many
                     wandb.log({f"debug/bsp_gate_avg_weight_path_{p_idx}": path_weights[:,p_idx].mean().item()}, step=current_global_step)
                 if self.args.bsp_num_paths > 16: wandb.log({"debug/bsp_gate_avg_weight_path_last": path_weights[:,-1].mean().item()}, step=current_global_step)

        else:
            predicted_noise_flat_patches = intermediate_or_final_features
        return predicted_noise_flat_patches

# --- Trainer (CPUDiffusionTrainer) ---
# ... (CPUDiffusionTrainer class remains unchanged from your last provided version, with the debug logs) ...
# Please ensure the debug log section in trainer.train method is present:
# if self.global_step < 5 :
#     self.logger.info(f"DEBUG GSTEP {self.global_step+1} (Batch {batch_idx}, FrameInSeq {frame_idx_in_selected_seq}):")
#     ... (all the print statements for x0, eps, x_t, target noise, pred noise) ...
class CPUDiffusionTrainer:
    def __init__(self, model: WuBuDiffusionModel, diffusion_process: DiffusionProcess, args: argparse.Namespace):
        self.model = model
        self.diffusion_process = diffusion_process
        self.args = args
        self.device = torch.device("cpu")
        self.logger = logger_wubu_diffusion.getChild("CPUDiffusionTrainer")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        self.global_step = 0
        self.current_epoch = 0

        self.patch_extractor = RegionalPatchExtractorCPU(patch_output_size=(args.patch_size_h, args.patch_size_w))

        if os.path.exists(args.video_data_path) and os.path.isdir(args.video_data_path):
             video_files = [f for f in os.listdir(args.video_data_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
             if not video_files:
                 self.logger.error(f"No video files found in directory: {args.video_data_path}")
                 if IMAGEIO_AVAILABLE and imageio is not None:
                     dummy_video_path = Path(args.video_data_path) / "dummy_cpu_video_diffusion.mp4"
                     self.logger.info(f"Creating dummy video at {dummy_video_path}")
                     dataset_sequence_len_for_dummy = args.num_input_frames + args.num_predict_frames
                     min_req_disk_frames = (dataset_sequence_len_for_dummy -1) * args.frame_skip + 1
                     num_dummy_frames = max(min_req_disk_frames + 50, int(min_req_disk_frames / max(0.01, (1.0 - args.val_fraction))) + 20 if args.val_fraction > 0 and args.val_fraction < 1 else min_req_disk_frames + 50)
                     try:
                         Path(args.video_data_path).mkdir(parents=True, exist_ok=True)
                         with imageio.get_writer(str(dummy_video_path), fps=15, codec='libx264', quality=8, ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
                             for _ in range(num_dummy_frames): writer.append_data(np.random.randint(0,255, (args.image_h, args.image_w, args.num_channels), dtype=np.uint8))
                         self.args.video_file_path = str(dummy_video_path)
                         self.logger.info(f"Dummy video created with {num_dummy_frames} frames: {self.args.video_file_path}")
                     except Exception as e_write: self.logger.error(f"Failed to write dummy video: {e_write}. Check ffmpeg/permissions."); raise FileNotFoundError("Failed to create dummy video, and no videos found.")
                 else: raise FileNotFoundError(f"No video files in {args.video_data_path} and imageio not available to create dummy.")
             else: self.args.video_file_path = os.path.join(args.video_data_path, video_files[0]); self.logger.info(f"Using first video found: {self.args.video_file_path}")
        elif os.path.exists(args.video_data_path) and os.path.isfile(args.video_data_path): self.args.video_file_path = args.video_data_path
        else: raise FileNotFoundError(f"Video data path not found or invalid: {args.video_data_path}")

        dataset_sequence_len = args.num_input_frames + args.num_predict_frames

        self.train_dataset = VideoFrameDatasetCPU(video_path=self.args.video_file_path, num_frames_total=dataset_sequence_len, image_size=(args.image_h, args.image_w),
            frame_skip=args.frame_skip, data_fraction=args.data_fraction, val_fraction=args.val_fraction, mode='train', seed=args.seed, args=self.args)
        self.train_loader = DataLoader( self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=False )

        self.val_loader = None
        if args.val_fraction > 0.0 and args.val_fraction < 1.0:
            self.val_dataset = VideoFrameDatasetCPU(video_path=self.args.video_file_path, num_frames_total=dataset_sequence_len, image_size=(args.image_h, args.image_w),
                frame_skip=args.frame_skip, data_fraction=1.0, val_fraction=args.val_fraction, mode='val', seed=args.seed, args=self.args)
            if len(self.val_dataset) > 0:
                self.val_loader = DataLoader(self.val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
                self.logger.info(f"Validation DataLoader created with {len(self.val_dataset)} samples. Effective batches: {len(self.val_loader)}")
            else: self.logger.warning("Validation dataset is empty after split, validation will be skipped."); self.val_loader = None
        else: self.logger.info("val_fraction is not in (0,1), validation will be skipped.")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.logger.info("CPUDiffusionTrainer initialized for CPU execution.")

    def _compute_loss(self, model_output_noise_patches: torch.Tensor, target_noise_patches: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(model_output_noise_patches, target_noise_patches)

    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, frames_to_log: Optional[torch.Tensor], num_frames_per_sequence_to_log: int = 1, num_sequences_to_log_max: int = 2):
        if not (self.args.wandb and WANDB_AVAILABLE and wandb.run and frames_to_log is not None and frames_to_log.numel() > 0): return
        current_frames_dim = frames_to_log.ndim
        if current_frames_dim == 4: frames_to_log = frames_to_log.unsqueeze(1)
        if frames_to_log.ndim != 5: self.logger.warning(f"WandB log samples: unexpected shape {frames_to_log.shape} (original: {current_frames_dim}D). Expected 5D (B, N_seq, C, H, W). Skip."); return
        B_log, N_seq_log, C_log, _, _ = frames_to_log.shape
        num_to_log_seqs = min(B_log, num_sequences_to_log_max)
        num_frames_log_this = min(N_seq_log, num_frames_per_sequence_to_log)
        wandb_imgs = []
        for b in range(num_to_log_seqs):
            for f_idx in range(num_frames_log_this):
                frame = frames_to_log[b,f_idx,...].cpu().float()
                if C_log == 1: frame = frame.repeat(3,1,1)
                img_0_1 = (frame.clamp(-1,1)+1)/2.0
                wandb_imgs.append(wandb.Image(img_0_1, caption=f"{tag_prefix} S{b} F{f_idx} Ep{self.current_epoch+1} GStep{self.global_step}"))
        if wandb_imgs:
            try: wandb.log({f"samples_video_cpu_diffusion/{tag_prefix}": wandb_imgs}, step=self.global_step)
            except Exception as e: self.logger.error(f"WandB video sample log fail for {tag_prefix}: {e}", exc_info=True)

    @torch.no_grad()
    def _validate_epoch(self):
        if not self.val_loader: self.logger.info("Skipping validation: No validation data loader."); return
        self.model.eval(); total_val_loss = 0.0; num_val_batches = 0
        val_prog_bar_desc = f"Validating Epoch {self.current_epoch+1}"
        if hasattr(self.val_loader.dataset, 'video_path') and self.val_loader.dataset.video_path: val_prog_bar_desc += f" ({Path(self.val_loader.dataset.video_path).name})"
        val_prog_bar = tqdm(range(len(self.val_loader)), desc=val_prog_bar_desc, disable=(os.getenv('CI')=='true' or not self.args.show_train_progress_bar), dynamic_ncols=True, leave=False)
        first_batch_val_sampled_frames = None
        first_batch_val_target_frames = None
        val_loader_iter_for_epoch = iter(self.val_loader)

        for batch_idx in val_prog_bar:
            try: val_frames_seq_raw = next(val_loader_iter_for_epoch)
            except StopIteration: self.logger.error("Validation DataLoader exhausted prematurely."); break
            val_frames_seq = val_frames_seq_raw.to(self.device)
            num_frames_to_process_diffusion = self.args.num_predict_frames if self.args.num_predict_frames > 0 else self.args.num_input_frames
            start_idx_for_x0 = self.args.num_input_frames if self.args.num_predict_frames > 0 else 0
            if val_frames_seq.shape[1] < start_idx_for_x0 + num_frames_to_process_diffusion:
                self.logger.warning(f"Val: Not enough frames in sequence ({val_frames_seq.shape[1]}) for diffusion. Skipping batch.")
                continue
            x0_target_frames_batch = val_frames_seq[:, start_idx_for_x0 : start_idx_for_x0 + num_frames_to_process_diffusion, ...]
            batch_loss_sum_for_item = 0.0
            num_frames_processed_this_batch_item = 0
            current_batch_item_bboxes = None
            if self.args.gaad_num_regions > 0:
                temp_bboxes_list_batch_item = []
                for f_item in range(x0_target_frames_batch.shape[1]):
                     temp_bboxes_list_batch_item.append(golden_subdivide_rect_fixed_n_cpu((self.args.image_w, self.args.image_h), self.args.gaad_num_regions, x0_target_frames_batch.dtype, self.args.gaad_min_size_px))
                current_batch_item_bboxes = torch.stack(temp_bboxes_list_batch_item).to(self.device)
                current_batch_item_bboxes = current_batch_item_bboxes.unsqueeze(0).expand(x0_target_frames_batch.shape[0], -1, -1, -1)

            for frame_idx_in_seq in range(x0_target_frames_batch.shape[1]):
                x0_single_frame = x0_target_frames_batch[:, frame_idx_in_seq, ...]
                t = torch.randint(0, self.diffusion_process.timesteps, (x0_single_frame.shape[0],), device=self.device).long()
                # image_level_epsilon_for_xt = torch.randn_like(x0_single_frame) # Used for creating x_t
                # x_t_single_frame = self.diffusion_process.q_sample(x0_single_frame, t, noise=image_level_epsilon_for_xt)
                # For validation, the target noise should also be fixed if we want to compare apples-to-apples with training
                # However, standard practice is to sample random t and random epsilon for validation loss too.
                # The important part is that `true_noise_flat_patches` is N(0,1) in patch space.
                true_noise_for_loss_calculation_img_space = torch.randn_like(x0_single_frame) # This is the epsilon we want to predict
                x_t_single_frame = self.diffusion_process.q_sample(x0_single_frame, t, noise=true_noise_for_loss_calculation_img_space)


                frame_bboxes = current_batch_item_bboxes[:, frame_idx_in_seq, ...] if current_batch_item_bboxes is not None else None
                if frame_bboxes is None and self.args.gaad_num_regions > 0:
                    self.logger.error("Validation bboxes are None when expected.")
                    frame_bboxes = golden_subdivide_rect_fixed_n_cpu((self.args.image_w,self.args.image_h),self.args.gaad_num_regions,x0_single_frame.dtype, self.args.gaad_min_size_px).unsqueeze(0).expand(x0_single_frame.shape[0],-1,-1).to(self.device)
                
                x_t_patches = self.patch_extractor(x_t_single_frame, frame_bboxes)
                x_t_flat_patches = x_t_patches.reshape(x_t_patches.shape[0], -1)
                
                # Target noise for loss: N(0,1) noise in patch space, corresponding to image_level_epsilon_for_xt
                # The simplest is to generate N(0,1) noise of the same shape as x_t_patches
                target_noise_N01_in_patch_space = torch.randn_like(x_t_patches) # This is the epsilon model should predict in patch form
                true_noise_flat_patches = target_noise_N01_in_patch_space.reshape(x_t_patches.shape[0], -1)
                
                predicted_noise_flat_patches = self.model(x_t_flat_patches, t, self.global_step)
                loss = self._compute_loss(predicted_noise_flat_patches, true_noise_flat_patches)
                batch_loss_sum_for_item += loss.item()
                num_frames_processed_this_batch_item += 1
            
            if num_frames_processed_this_batch_item > 0:
                avg_loss_for_batch_item = batch_loss_sum_for_item / num_frames_processed_this_batch_item
                total_val_loss += avg_loss_for_batch_item
                num_val_batches += 1
            
            if batch_idx == 0 and self.args.num_val_samples_to_log > 0 and hasattr(self.args, 'val_sampling_log_steps') and self.args.val_sampling_log_steps > 0:
                if x0_target_frames_batch.numel() > 0:
                    num_samples_to_gen = min(x0_target_frames_batch.shape[0], self.args.num_val_samples_to_log)
                    model_lambda_for_sampling = lambda xt, t_sample: self.model(xt, t_sample, self.global_step)
                    self.logger.info(f"Generating {num_samples_to_gen} validation samples with {self.args.val_sampling_log_steps} steps...")
                    t_val_sample_start = time.time()
                    sampled_val_frames = self.diffusion_process.sample(
                        model_lambda_for_sampling, batch_size=num_samples_to_gen,
                        image_channels=self.args.num_channels, image_size=(self.args.image_h, self.args.image_w),
                        patch_h=self.args.patch_size_h, patch_w=self.args.patch_size_w, num_regions=self.args.gaad_num_regions,
                        num_sampling_steps=self.args.val_sampling_log_steps,
                        show_progress=getattr(self.args, 'show_val_sample_progress', False)
                    )
                    t_val_sample_end = time.time()
                    self.logger.info(f"Validation sample generation took {t_val_sample_end - t_val_sample_start:.2f}s.")
                    first_batch_val_sampled_frames = sampled_val_frames.detach().cpu()
                    first_batch_val_target_frames = x0_target_frames_batch[:num_samples_to_gen, 0:1, ...].detach().cpu()

        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            self.logger.info(f"Validation Epoch {self.current_epoch+1}: Avg Loss: {avg_val_loss:.4f}")
            if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb_log_dict = {"val/loss_mse": avg_val_loss, "epoch": self.current_epoch}
                if self.args.wandb_log_val_recon_interval_epochs > 0 and \
                   (self.current_epoch + 1) % self.args.wandb_log_val_recon_interval_epochs == 0:
                    if first_batch_val_sampled_frames is not None:
                        self._log_samples_to_wandb("val_sampled_reduced_steps", first_batch_val_sampled_frames, 1, self.args.num_val_samples_to_log)
                    if first_batch_val_target_frames is not None:
                        self._log_samples_to_wandb("val_target_frame0", first_batch_val_target_frames, 1, self.args.num_val_samples_to_log)
                wandb.log(wandb_log_dict, step=self.global_step)
        else: self.logger.info(f"Validation Epoch {self.current_epoch+1}: No batches processed.")
        self.model.train()

    def train(self, start_epoch: int = 0, initial_global_step: int = 0):
        self.global_step, self.current_epoch = initial_global_step, start_epoch
        self.logger.info(f"Starting CPU Diffusion training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch+1}, GStep: {initial_global_step}")
        if not self.train_loader or len(self.train_loader.dataset) == 0:
            self.logger.error("Training dataset is empty. Cannot start training."); return

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch; self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting.")
            self.model.train()
            prog_bar_desc = f"Epoch {epoch+1}"
            if hasattr(self.train_loader.dataset, 'video_path') and self.train_loader.dataset.video_path: prog_bar_desc += f" ({Path(self.train_loader.dataset.video_path).name})"
            train_loader_iter_for_epoch = iter(self.train_loader)
            prog_bar = tqdm(range(len(self.train_loader)), desc=prog_bar_desc, disable=(os.getenv('CI')=='true' or not self.args.show_train_progress_bar), dynamic_ncols=True)

            for batch_idx in prog_bar:
                t_dataload_start = time.time()
                try: real_frames_seq_raw = next(train_loader_iter_for_epoch)
                except StopIteration: self.logger.error("Train DataLoader exhausted prematurely."); break
                t_dataload_end = time.time(); time_dataload = t_dataload_end - t_dataload_start
                t_todevice_start = time.time(); real_frames_seq = real_frames_seq_raw.to(self.device); t_todevice_end = time.time(); time_todevice = t_todevice_end - t_todevice_start
                t_train_cycle_start = time.time()
                self.optimizer.zero_grad()
                num_frames_to_process_diffusion = self.args.num_predict_frames if self.args.num_predict_frames > 0 else self.args.num_input_frames
                start_idx_for_x0 = self.args.num_input_frames if self.args.num_predict_frames > 0 else 0
                if real_frames_seq.shape[1] < start_idx_for_x0 + num_frames_to_process_diffusion:
                    self.logger.warning(f"Train: Not enough frames in sequence ({real_frames_seq.shape[1]}) for diffusion. Skipping batch.")
                    continue
                x0_frames_for_diffusion_batch = real_frames_seq[:, start_idx_for_x0 : start_idx_for_x0 + num_frames_to_process_diffusion, ...]
                batch_loss_sum_train = 0.0
                num_frames_processed_train_batch_item = 0
                current_batch_item_bboxes_train = None
                if self.args.gaad_num_regions > 0:
                    temp_bboxes_list_batch_item_train = []
                    for f_item in range(x0_frames_for_diffusion_batch.shape[1]):
                         temp_bboxes_list_batch_item_train.append(golden_subdivide_rect_fixed_n_cpu((self.args.image_w, self.args.image_h), self.args.gaad_num_regions, x0_frames_for_diffusion_batch.dtype, self.args.gaad_min_size_px))
                    current_batch_item_bboxes_train = torch.stack(temp_bboxes_list_batch_item_train).to(self.device)
                    current_batch_item_bboxes_train = current_batch_item_bboxes_train.unsqueeze(0).expand(x0_frames_for_diffusion_batch.shape[0], -1, -1, -1)

                for frame_idx_in_selected_seq in range(x0_frames_for_diffusion_batch.shape[1]):
                    x0_single_frame_train = x0_frames_for_diffusion_batch[:, frame_idx_in_selected_seq, ...]
                    t_train = torch.randint(0, self.diffusion_process.timesteps, (x0_single_frame_train.shape[0],), device=self.device).long()
                    
                    # This is the image-level N(0,1) noise used to create x_t
                    image_level_epsilon_for_xt = torch.randn_like(x0_single_frame_train) 
                    x_t_single_frame_train = self.diffusion_process.q_sample(x0_single_frame_train, t_train, noise=image_level_epsilon_for_xt)
                    
                    frame_bboxes_train = current_batch_item_bboxes_train[:, frame_idx_in_selected_seq, ...] if current_batch_item_bboxes_train is not None else None
                    if frame_bboxes_train is None and self.args.gaad_num_regions > 0:
                        self.logger.error("Train bboxes are None when expected.")
                        frame_bboxes_train = golden_subdivide_rect_fixed_n_cpu((self.args.image_w,self.args.image_h),self.args.gaad_num_regions,x0_single_frame_train.dtype, self.args.gaad_min_size_px).unsqueeze(0).expand(x0_single_frame_train.shape[0],-1,-1).to(self.device)
                    
                    x_t_patches_train = self.patch_extractor(x_t_single_frame_train, frame_bboxes_train)
                    x_t_flat_patches_train = x_t_patches_train.reshape(x_t_patches_train.shape[0], -1)
                    
                    # Generate N(0,1) target noise directly in the patch space shape
                    target_noise_N01_in_patch_space = torch.randn_like(x_t_patches_train)
                    true_noise_flat_patches_train = target_noise_N01_in_patch_space.reshape(x_t_patches_train.shape[0], -1)
                    
                    model_timings = {}
                    t_model_fwd_start = time.time()
                    predicted_noise_flat_patches_train = self.model(x_t_flat_patches_train, t_train, self.global_step)
                    t_model_fwd_end = time.time()
                    model_timings["time_model_fwd"] = t_model_fwd_end - t_model_fwd_start
                    
                    if self.global_step < 5:
                        self.logger.info(f"DEBUG GSTEP {self.global_step+1} (Batch {batch_idx}, FrameInSeq {frame_idx_in_selected_seq}):")
                        self.logger.info(f"  x0_single_frame_train   - Mean: {x0_single_frame_train.mean().item():.4f}, Std: {x0_single_frame_train.std().item():.4f}")
                        self.logger.info(f"  image_level_eps_for_xt  - Mean: {image_level_epsilon_for_xt.mean().item():.4f}, Std: {image_level_epsilon_for_xt.std().item():.4f}")
                        self.logger.info(f"  x_t_single_frame_train  - Mean: {x_t_single_frame_train.mean().item():.4f}, Std: {x_t_single_frame_train.std().item():.4f}")
                        self.logger.info(f"  x_t_patches_train (flat)- Mean: {x_t_flat_patches_train.mean().item():.4f}, Std: {x_t_flat_patches_train.std().item():.4f}")
                        self.logger.info(f"  TARGET Noise Patches (N01)- Mean: {true_noise_flat_patches_train.mean().item():.4f}, Std: {true_noise_flat_patches_train.std().item():.4f}, Norm: {torch.linalg.norm(true_noise_flat_patches_train).item():.4f}")
                        self.logger.info(f"  PRED Noise Patches      - Mean: {predicted_noise_flat_patches_train.mean().item():.4f}, Std: {predicted_noise_flat_patches_train.std().item():.4f}, Norm: {torch.linalg.norm(predicted_noise_flat_patches_train).item():.4f}")

                    loss_train_for_frame = self._compute_loss(predicted_noise_flat_patches_train, true_noise_flat_patches_train)
                    batch_loss_sum_train += loss_train_for_frame
                    num_frames_processed_train_batch_item +=1

                if num_frames_processed_train_batch_item > 0:
                    avg_loss_for_batch_item = batch_loss_sum_train / num_frames_processed_train_batch_item
                    avg_loss_for_batch_item.backward()
                    self.optimizer.step()
                else: avg_loss_for_batch_item = torch.tensor(0.0)
                
                t_train_cycle_end = time.time(); time_train_cycle = t_train_cycle_end - t_train_cycle_start
                self.global_step +=1

                if self.global_step % self.args.log_interval == 0:
                    log_items = {"Loss": avg_loss_for_batch_item.item(), "T_DataLoad": time_dataload, "T_ToDevice": time_todevice,
                                 "T_ModelFwd": model_timings.get("time_model_fwd",0), "T_TrainCycle": time_train_cycle }
                    formatted_log_items = {k_l: (f"{v_l:.3f}s" if k_l.startswith("T_") else f"{v_l:.3f}") for k_l,v_l in log_items.items()}
                    log_str_parts = [f"{k_l}:{v_l}" for k_l,v_l in formatted_log_items.items()]
                    postfix_str = f"Loss:{avg_loss_for_batch_item.item():.3f} Data:{time_dataload:.2f}s"
                    if hasattr(prog_bar, 'set_postfix_str'): prog_bar.set_postfix_str(postfix_str)
                    full_log_msg = f"E:{epoch+1} S:{self.global_step} (B:{batch_idx+1}/{len(self.train_loader)}) | " + " | ".join(log_str_parts); self.logger.info(full_log_msg)
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                        wandb_log_data = {"train/loss_mse_avg_batch_item": avg_loss_for_batch_item.item(),
                                          "time/data_load_batch_item": time_dataload, "time/to_device_batch_item": time_todevice,
                                          "time/model_fwd_last_frame": model_timings.get("time_model_fwd",0),
                                          "time/train_cycle_batch_item": time_train_cycle,
                                          "global_step":self.global_step, "epoch_frac":epoch+((batch_idx+1)/max(1,len(self.train_loader)))}
                        wandb.log(wandb_log_data, step=self.global_step)

                if self.args.wandb_log_train_samples_interval > 0 and self.global_step > 0 and self.global_step % self.args.wandb_log_train_samples_interval == 0:
                    if x0_frames_for_diffusion_batch.numel() > 0 :
                        num_samples_to_gen_train = min(x0_frames_for_diffusion_batch.shape[0], self.args.num_train_samples_to_log)
                        model_lambda_for_sampling_train = lambda xt, t_sample: self.model(xt, t_sample, self.global_step)
                        train_sampling_steps = getattr(self.args, 'train_sampling_log_steps', self.args.val_sampling_log_steps)
                        sampled_train_img = self.diffusion_process.sample(
                            model_lambda_for_sampling_train, batch_size=num_samples_to_gen_train,
                            image_channels=self.args.num_channels, image_size=(self.args.image_h, self.args.image_w),
                            patch_h=self.args.patch_size_h, patch_w=self.args.patch_size_w, num_regions=self.args.gaad_num_regions,
                            num_sampling_steps=train_sampling_steps,
                            show_progress=getattr(self.args, 'show_train_sample_progress', False)
                        )
                        if sampled_train_img is not None: self._log_samples_to_wandb("train_sampled", sampled_train_img, 1, num_samples_to_gen_train)
                        self._log_samples_to_wandb("train_target_x0", x0_frames_for_diffusion_batch[:num_samples_to_gen_train,0:1,...].detach(), 1, num_samples_to_gen_train)

            if (epoch + 1) % self.args.validate_every_n_epochs == 0: self._validate_epoch()
            if self.args.save_interval > 0 and (epoch+1) > 0 and (epoch+1)%self.args.save_interval==0: self._save_checkpoint(epoch)
            if self.args.wandb_log_fixed_noise_samples_interval > 0 and (epoch+1) > 0 and (epoch+1)%self.args.wandb_log_fixed_noise_samples_interval==0:
                num_fixed_samples = self.args.num_val_samples_to_log
                model_lambda_for_fixed_noise = lambda xt, t_sample: self.model(xt, t_sample, self.global_step)
                fixed_noise_sampling_steps = getattr(self.args, 'fixed_noise_sampling_log_steps', self.args.val_sampling_log_steps)
                fixed_noise_pixels = self.diffusion_process.sample(
                    model_lambda_for_fixed_noise, batch_size=num_fixed_samples,
                    image_channels=self.args.num_channels, image_size=(self.args.image_h, self.args.image_w),
                    patch_h=self.args.patch_size_h, patch_w=self.args.patch_size_w, num_regions=self.args.gaad_num_regions,
                    num_sampling_steps=fixed_noise_sampling_steps,
                    show_progress=getattr(self.args, 'show_fixed_noise_sample_progress', False)
                )
                if fixed_noise_pixels is not None: self._log_samples_to_wandb("fixed_noise_gen_diffusion",fixed_noise_pixels,1,num_fixed_samples)

        self.logger.info("CPU Diffusion Training finished."); self._save_checkpoint(self.current_epoch,is_final=True)

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()}
        data = { 'epoch': epoch,'global_step':self.global_step,'model_state_dict':self.model.state_dict(),
                'optimizer_state_dict':self.optimizer.state_dict(), 'args':serializable_args}
        fn = f"wubudiffusion_v1_{'final' if is_final else f'ep{epoch+1}_step{self.global_step}'}.pt"
        try: torch.save(data, Path(self.args.checkpoint_dir)/fn); self.logger.info(f"Checkpoint saved: {fn}")
        except Exception as e: self.logger.error(f"Error saving ckpt {fn}: {e}", exc_info=True)

    def load_checkpoint(self, ckpt_path_str: Optional[str]) -> Tuple[int,int]:
        if not ckpt_path_str or not os.path.exists(ckpt_path_str):
            self.logger.warning(f"Checkpoint '{ckpt_path_str}' not found. Starting fresh."); return 0,0
        try:
            ckpt = torch.load(ckpt_path_str, map_location=self.device); self.logger.info(f"Loaded checkpoint: {ckpt_path_str}")
        except Exception as e: self.logger.error(f"Failed to load ckpt {ckpt_path_str}: {e}. Fresh start.", exc_info=True); return 0,0
        if 'args' in ckpt and ckpt['args'] is not None:
            self.logger.info("Attempting to load args from checkpoint."); loaded_ckpt_args_dict = ckpt['args']
            current_args_dict = vars(self.args)
            critical_structure_args = [
                'image_h', 'image_w', 'num_channels', 'gaad_num_regions', 'patch_size_h', 'patch_size_w', 'diffusion_time_embed_dim',
                'model_wubu_num_virtual_levels', 'model_wubu_base_micro_dim', 'model_wubu_num_physical_stages',
                'use_bsp_gate', 'bsp_num_paths', 'bsp_intermediate_dim', 'use_manifold_uncrumple_transform' # Added MUT flag
            ]
            for k_ckpt, v_ckpt_loaded in loaded_ckpt_args_dict.items():
                v_ckpt = v_ckpt_loaded
                if k_ckpt in critical_structure_args and k_ckpt in current_args_dict:
                    # Special handling for bool_type args, as they might be saved as actual booleans
                    current_val_str = str(current_args_dict[k_ckpt])
                    v_ckpt_str = str(v_ckpt)
                    # Handle boolean string representations
                    if isinstance(current_args_dict[k_ckpt], bool): current_val_str = str(current_args_dict[k_ckpt]).lower()
                    if isinstance(v_ckpt, bool): v_ckpt_str = str(v_ckpt).lower()

                    if current_val_str != v_ckpt_str:
                        self.logger.warning(f"CRITICAL ARG MISMATCH for '{k_ckpt}': Ckpt='{v_ckpt_str}', Current CLI='{current_val_str}'. Model might not load correctly.")
                if k_ckpt not in current_args_dict: setattr(self.args, k_ckpt, v_ckpt); self.logger.info(f"  Arg '{k_ckpt}' loaded from ckpt: '{v_ckpt}' (added to current args)")
                elif str(current_args_dict[k_ckpt]) != str(v_ckpt) and k_ckpt not in critical_structure_args:
                     self.logger.info(f"  Arg '{k_ckpt}': Ckpt='{v_ckpt}', Current CLI='{current_args_dict[k_ckpt]}'. Using current CLI value.")
        model_state_dict = ckpt.get('model_state_dict')
        if model_state_dict:
            try: self.model.load_state_dict(model_state_dict, strict=True); self.logger.info("Successfully loaded model_state_dict (strict=True).")
            except RuntimeError as e_strict:
                self.logger.warning(f"Error loading model_state_dict with strict=True: {e_strict}. Trying strict=False.")
                try: self.model.load_state_dict(model_state_dict, strict=False); self.logger.info("Successfully loaded model_state_dict with strict=False.")
                except Exception as e_non_strict: self.logger.error(f"Failed to load model_state_dict even with strict=False: {e_non_strict}. Model weights random.")
        optimizer_state_dict = ckpt.get('optimizer_state_dict')
        if optimizer_state_dict:
            try: self.optimizer.load_state_dict(optimizer_state_dict); self.logger.info("Successfully loaded optimizer_state_dict.")
            except ValueError as e_val: self.logger.error(f"Error loading optimizer_state_dict (param mismatch?): {e_val}. Optimizer fresh.")
            except Exception as e_opt: self.logger.error(f"Other error loading optimizer_state_dict: {e_opt}. Optimizer fresh.")
        gs = ckpt.get('global_step',0); ep_saved = ckpt.get('epoch',0)
        start_ep = 0 if self.args.load_checkpoint_reset_epoch else ep_saved + 1
        start_gs = 0 if self.args.load_checkpoint_reset_epoch else gs
        self.logger.info(f"Resuming. Ckpt Epoch: {ep_saved}, Ckpt GlobalStep: {gs}. Effective Start Epoch: {start_ep}, Effective Start GStep: {start_gs}.")
        return start_gs, start_ep

def parse_diffusion_arguments():
    def bool_type(s): s_lower = str(s).lower(); return s_lower in ('yes', 'true', 't', 'y', '1')
    parser = argparse.ArgumentParser(description="WuBu CPU-Only Diffusion Model (v1 Advanced FDQWR)")
    # Paths & General Training
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_cpu_diffusion")
    parser.add_argument('--checkpoint_dir', type=str, default='wubudiffusion_v1_checkpoints_adv')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--seed',type=int, default=42); parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200); parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate',type=float,default=1e-4)
    # Data/Model Specs
    parser.add_argument('--image_h', type=int, default=64); parser.add_argument('--image_w', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_input_frames', type=int, default=1)
    parser.add_argument('--num_predict_frames', type=int, default=0)
    parser.add_argument('--frame_skip', type=int, default=1)
    # GAAD & Patching
    parser.add_argument('--gaad_num_regions', type=int, default=16); parser.add_argument('--gaad_min_size_px', type=int, default=4)
    parser.add_argument('--patch_size_h', type=int, default=8); parser.add_argument('--patch_size_w', type=int, default=8)
    # Diffusion Specific Params
    parser.add_argument('--diffusion_timesteps', type=int, default=1000)
    parser.add_argument('--diffusion_beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--diffusion_beta_start', type=float, default=0.0001); parser.add_argument('--diffusion_beta_end', type=float, default=0.02)
    parser.add_argument('--diffusion_time_embed_dim', type=int, default=128)
    # WuBu Core Common Params
    parser.add_argument('--wubu_initial_s', type=float, default=1.0); parser.add_argument('--wubu_s_decay', type=float, default=0.9999)
    parser.add_argument('--wubu_initial_c', type=float, default=0.1); parser.add_argument('--wubu_c_phi_influence', type=bool_type, default=True)
    parser.add_argument('--wubu_num_phi_scaffold_points', type=int, default=3); parser.add_argument('--wubu_phi_scaffold_init_scale', type=float, default=0.001)
    parser.add_argument('--wubu_use_gaussian_rungs', type=bool_type, default=True)
    parser.add_argument('--wubu_base_gaussian_std_dev', type=float, default=0.005); parser.add_argument('--wubu_gaussian_std_dev_decay', type=float, default=0.999995)
    parser.add_argument('--wubu_rung_affinity_temp', type=float, default=0.01); parser.add_argument('--wubu_rung_modulation_strength', type=float, default=0.001)
    parser.add_argument('--wubu_t_tilde_scale', type=float, default=1.00001)
    parser.add_argument('--wubu_micro_transform_type', type=str, default="mlp", choices=["mlp", "linear", "identity"]) # For OriginalFMT
    parser.add_argument('--wubu_micro_transform_hidden_factor', type=float, default=0.5) # For OriginalFMT
    parser.add_argument('--wubu_use_quaternion_so4', type=bool_type, default=True) # For OriginalFMT SO(4) rotation
    parser.add_argument('--wubu_scaffold_co_rotation_mode', type=str, default="none", choices=["none", "matrix_only"]) # For OriginalFMT
    parser.add_argument('--wubu_enable_isp', type=bool_type, default=True) # For OriginalFMT
    parser.add_argument('--wubu_enable_stateful', type=bool_type, default=True) # For OriginalFMT
    parser.add_argument('--wubu_enable_hypernet', type=bool_type, default=True); # For OriginalFMT
    parser.add_argument('--wubu_internal_state_factor', type=float, default=0.5) # For OriginalFMT
    parser.add_argument('--wubu_hypernet_strength', type=float, default=0.01) # For OriginalFMT
    # WuBu Stack Specifics (for the single noise predictor model)
    parser.add_argument('--model_wubu_num_virtual_levels', type=int, default=1500)
    parser.add_argument('--model_wubu_base_micro_dim', type=int, default=4)
    parser.add_argument('--model_wubu_num_physical_stages', type=int, default=30)
    # New ManifoldUncrumpleTransform (MUT) Specific Params for FDQWR
    parser.add_argument('--use_manifold_uncrumple_transform', type=bool_type, default=False, help="Use ManifoldUncrumpleTransform instead of OriginalFractalMicroTransformRungs in FDQWR stages.")
    parser.add_argument('--uncrumple_hypernet_hidden_factor', type=float, default=1.0, help="Hidden dim factor for MUT's internal hypernet.")
    parser.add_argument('--uncrumple_learn_rotation', type=bool_type, default=True)
    parser.add_argument('--uncrumple_learn_scale', type=bool_type, default=True)
    parser.add_argument('--uncrumple_scale_activation', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'exp', 'softplus'])
    parser.add_argument('--uncrumple_scale_min', type=float, default=0.5)
    parser.add_argument('--uncrumple_scale_max', type=float, default=2.0)
    parser.add_argument('--uncrumple_learn_translation', type=bool_type, default=True)
    # BSP Gate Parameters
    parser.add_argument('--use_bsp_gate', type=bool_type, default=False, help="Enable BSP-like input gating for the diffusion model.")
    parser.add_argument('--bsp_gate_hidden_dim', type=int, default=64, help="Hidden dimension for the BSP gating MLP.")
    parser.add_argument('--bsp_num_paths', type=int, default=2, help="Number of 'paths' or 'experts' for the BSP gate.")
    parser.add_argument('--bsp_intermediate_dim', type=int, default=256, help="Intermediate feature dimension from main WuBu stack before BSP output heads.")
    parser.add_argument('--bsp_gate_activation', type=str, default='softmax', choices=['softmax', 'sigmoid'], help="Activation for BSP gate path selection.")
    parser.add_argument('--bsp_log_gate_interval_mult', type=int, default=5, help="Multiplier for log_interval to log BSP gate weights.")
    # Logging & Saving
    parser.add_argument('--wandb', type=bool_type, default=True); parser.add_argument('--wandb_project',type=str,default='WuBuDiffusionV1Adv')
    parser.add_argument('--wandb_run_name',type=str,default=None); parser.add_argument('--log_interval',type=int, default=10)
    parser.add_argument('--save_interval',type=int, default=5);
    parser.add_argument('--wandb_log_train_samples_interval', type=int, default=50)
    parser.add_argument('--train_sampling_log_steps', type=int, default=50, help="Sampling steps for train sample logging.")
    parser.add_argument('--show_train_sample_progress', type=bool_type, default=False)
    parser.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=10)
    parser.add_argument('--fixed_noise_sampling_log_steps', type=int, default=50, help="Sampling steps for fixed noise sample logging.")
    parser.add_argument('--show_fixed_noise_sample_progress', type=bool_type, default=False)
    parser.add_argument('--num_val_samples_to_log', type=int, default=2)
    parser.add_argument('--num_train_samples_to_log', type=int, default=2)
    parser.add_argument('--data_fraction', type=float, default=0.1); parser.add_argument('--show_train_progress_bar', type=bool_type, default=True)
    parser.add_argument('--getitem_log_interval', type=int, default=100); parser.add_argument('--getitem_slow_threshold', type=float, default=1.0)
    parser.add_argument('--load_checkpoint_reset_epoch', type=bool_type, default=False)
    # Validation
    parser.add_argument('--val_fraction', type=float, default=0.1); parser.add_argument('--val_batch_size', type=int, default=None)
    parser.add_argument('--validate_every_n_epochs', type=int, default=2)
    parser.add_argument('--wandb_log_val_recon_interval_epochs', type=int, default=1)
    parser.add_argument('--val_sampling_log_steps', type=int, default=64, help="Sampling steps for validation sample logging.")
    parser.add_argument('--show_val_sample_progress', type=bool_type, default=True, help="Show tqdm for validation sampling.")

    parsed_args = parser.parse_args()
    if parsed_args.val_batch_size is None: parsed_args.val_batch_size = parsed_args.batch_size
    if parsed_args.use_bsp_gate and parsed_args.bsp_num_paths <= 1:
        logger_wubu_diffusion.warning("use_bsp_gate is True but bsp_num_paths <= 1. Disabling BSP gate as it's ineffective.")
        parsed_args.use_bsp_gate = False
    if parsed_args.use_bsp_gate and parsed_args.bsp_intermediate_dim <=0:
        logger_wubu_diffusion.error("use_bsp_gate is True but bsp_intermediate_dim <=0. This is invalid. Setting to a default of 256.")
        parsed_args.bsp_intermediate_dim = 256
    if parsed_args.use_manifold_uncrumple_transform and (parsed_args.model_wubu_base_micro_dim <=0 or parsed_args.wubu_num_phi_scaffold_points <=0 ):
        logger_wubu_diffusion.warning("use_manifold_uncrumple_transform is True but base_micro_dim or num_phi_scaffold_points is <=0. Disabling MUT.")
        parsed_args.use_manifold_uncrumple_transform = False
    return parsed_args

def main_cpu_diffusion():
    args = parse_diffusion_arguments()
    global tqdm # Allow modification of the global tqdm variable

    # Configure TQDM based on args and import success
    if not args.show_train_progress_bar:
        logger_wubu_diffusion.info("TQDM progress bars are disabled via args.show_train_progress_bar=False.")
        class _TqdmDummyArgDisabled: # Define dummy if disabled by arg
            _last_desc_printed = None
            def __init__(self, iterable=None, *args_dummy, **kwargs_dummy):
                self.iterable = iterable if iterable is not None else []
                desc = kwargs_dummy.get('desc')
                if desc and desc != _TqdmDummyArgDisabled._last_desc_printed:
                    # print(f"{desc} (Progress bars disabled by args)") # Can be too verbose
                    _TqdmDummyArgDisabled._last_desc_printed = desc
            def __iter__(self): return iter(self.iterable)
            def __enter__(self): return self
            def __exit__(self, *exc_info): pass
            def set_postfix(self, ordered_dict=None, refresh=True, **kwargs_p): pass
            def set_postfix_str(self, s: str = "", refresh: bool = True): pass
            def update(self, n: int = 1): pass
            def close(self): pass
        tqdm = _TqdmDummyArgDisabled # Reassign global tqdm to the arg-disabled dummy
    elif _tqdm_imported_successfully:
        logger_wubu_diffusion.info("TQDM progress bars are ENABLED (tqdm library imported).")
        # tqdm is already _TQDM_INITIAL_WRAPPER which is the real tqdm module
    else: # tqdm import failed initially
        logger_wubu_diffusion.info("TQDM progress bars are using a basic dummy (tqdm library not found).")
        # tqdm is already _TQDM_INITIAL_WRAPPER which is _TqdmDummyMissing

    global DEBUG_MICRO_TRANSFORM_INTERNALS # Make sure this global flag is accessible
    # Set logging level (e.g. to DEBUG if you want to see print_tensor_stats)
    # DEBUG_MICRO_TRANSFORM_INTERNALS = True # Example: Enable for debugging
    # logger_wubu_diffusion.setLevel(logging.DEBUG if DEBUG_MICRO_TRANSFORM_INTERNALS else logging.INFO)
    
    # Reconfigure basicConfig for the root logger to ensure format applies if script is main
    # Or, ensure only the script's logger is configured if it's part of a larger system.
    # For standalone script, force=True can be useful.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True)
    if DEBUG_MICRO_TRANSFORM_INTERNALS:
        logger_wubu_diffusion.setLevel(logging.DEBUG) # Set this script's logger to DEBUG if flag is True
        logger_wubu_diffusion.debug("DEBUG_MICRO_TRANSFORM_INTERNALS is True. Expect verbose tensor stats.")


    logger_wubu_diffusion.info(f"--- WuBuDiffusionV1 (Advanced FDQWR with optional BSP & MUT) ---"); logger_wubu_diffusion.info(f"Effective Args: {vars(args)}")
    random.seed(args.seed); os.environ['PYTHONHASHSEED'] = str(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubudiff_adv_bsp_mut_{datetime.now().strftime('%y%m%d_%H%M')}"
        try:
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            logger_wubu_diffusion.info(f"WandB initialized: Run '{run_name}', Project '{args.wandb_project}'")
        except Exception as e_wandb:
            logger_wubu_diffusion.error(f"WandB initialization failed: {e_wandb}", exc_info=True)
            args.wandb = False # Disable wandb if init fails

    device = torch.device("cpu")
    video_config = { "num_channels": args.num_channels, "num_input_frames": args.num_input_frames, "num_predict_frames": args.num_predict_frames }
    gaad_config = { "num_regions": args.gaad_num_regions, "min_size_px": args.gaad_min_size_px }

    model = WuBuDiffusionModel(args, video_config, gaad_config).to(device)
    diffusion_process = DiffusionProcess(timesteps=args.diffusion_timesteps, beta_schedule=args.diffusion_beta_schedule,
                                         beta_start=args.diffusion_beta_start, beta_end=args.diffusion_beta_end, device=device)

    if args.wandb and WANDB_AVAILABLE and wandb.run is not None: # Check if wandb.run exists
        try:
            wandb.watch(model, log="gradients", log_freq=args.log_interval * 10, log_graph=False)
        except Exception as e_watch:
            logger_wubu_diffusion.error(f"WandB watch failed: {e_watch}")


    trainer = CPUDiffusionTrainer(model, diffusion_process, args)
    start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint) if args.load_checkpoint else (0,0)
    
    try:
        trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt:
        logger_wubu_diffusion.info("Training interrupted by user.")
    except Exception as e:
        logger_wubu_diffusion.error(f"Training loop crashed: {e}", exc_info=True)
    finally:
        logger_wubu_diffusion.info("Finalizing run...")
        if hasattr(trainer, 'current_epoch') and hasattr(trainer, 'global_step'): # Ensure trainer fully init
            trainer._save_checkpoint(epoch=trainer.current_epoch, is_final=True)
        if args.wandb and WANDB_AVAILABLE and wandb.run is not None: # Check if wandb.run exists
            try:
                wandb.finish()
            except Exception as e_finish:
                 logger_wubu_diffusion.error(f"WandB finish failed: {e_finish}")
        logger_wubu_diffusion.info("WuBuDiffusionV1 (Advanced FDQWR with optional BSP & MUT) script finished.")

if __name__ == "__main__":
    main_cpu_diffusion()
