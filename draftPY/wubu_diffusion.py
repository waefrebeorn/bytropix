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
ROT_PARAM_TANH_SCALE = math.pi # Scale for tanh'd rotation parameters

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
        _last_desc_printed = None 
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
DEBUG_MICRO_TRANSFORM_INTERNALS = False 

def print_tensor_stats_mut(tensor: Optional[torch.Tensor], name: str, micro_level_idx: Optional[int] = None, stage_idx: Optional[int] = None, enabled: bool = True, batch_idx_to_print: int = 0):
    if not enabled or tensor is None or not DEBUG_MICRO_TRANSFORM_INTERNALS: return
    if tensor.numel() == 0: return
    if logger_wubu_diffusion.isEnabledFor(logging.DEBUG):
        prefix = f"MUT_L{micro_level_idx if micro_level_idx is not None else 'N'}/S{stage_idx if stage_idx is not None else 'N'}| "
        item_to_show = tensor
        if tensor.dim() > 1 and tensor.shape[0] > batch_idx_to_print:
            item_to_show = tensor[batch_idx_to_print].detach()
            prefix += f"{name}(i{batch_idx_to_print}):"
        elif tensor.dim() == 1:
            item_to_show = tensor.detach()
            prefix += f"{name}:"
        else: 
            item_to_show = tensor.detach()
            prefix += f"{name}(s):"
        logger_wubu_diffusion.debug(prefix +
            f"Sh:{tensor.shape}, Dt:{tensor.dtype}, "
            f"NaN:{torch.isnan(item_to_show).any().item()}, Inf:{torch.isinf(item_to_show).any().item()}, "
            f"Min:{item_to_show.min().item():.2e}, Max:{item_to_show.max().item():.2e}, "
            f"Mu:{item_to_show.mean().item():.2e}, Std:{item_to_show.std().item():.2e}, "
            f"Nrm:{torch.linalg.norm(item_to_show.float()).item():.2e}")


# --- Utility Functions ---
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
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', True)): # Check if affine is enabled
            if hasattr(m, 'weight') and m.weight is not None: nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor:
    return F.softplus(param_unconstrained) + min_val

def print_tensor_stats_fdq(tensor: Optional[torch.Tensor], name: str, micro_level_idx: Optional[int] = None, stage_idx: Optional[int] = None, enabled: bool = True, batch_idx_to_print: int = 0):
    if not enabled or tensor is None or not DEBUG_MICRO_TRANSFORM_INTERNALS: return
    if tensor.numel() == 0: return
    item_tensor = tensor; prefix = f"FDQ_L{micro_level_idx if micro_level_idx is not None else 'N'}/S{stage_idx if stage_idx is not None else 'N'}| "
    if tensor.dim() > 1 and tensor.shape[0] > batch_idx_to_print : item_tensor = tensor[batch_idx_to_print].detach(); prefix += f"{name}(i{batch_idx_to_print}):"
    elif tensor.dim() == 1: item_tensor = tensor.detach(); prefix += f"{name}:"
    else: item_tensor = tensor.detach(); prefix += f"{name}(f):"
    if logger_wubu_diffusion.isEnabledFor(logging.DEBUG):
        logger_wubu_diffusion.debug(prefix +
            f"Sh:{tensor.shape},Dt:{tensor.dtype},"
            f"NaN:{torch.isnan(item_tensor).any().item()},Inf:{torch.isinf(item_tensor).any().item()},"
            f"Min:{item_tensor.min().item():.2e},Max:{item_tensor.max().item():.2e},"
            f"Mu:{item_tensor.mean().item():.2e},Std:{item_tensor.std().item():.2e},"
            f"Nrm:{torch.linalg.norm(item_tensor.float()).item():.2e}")

# --- Quaternion, HyperbolicUtils, PoincareBall ---
def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    if angle_rad.dim() == 0: angle_rad = angle_rad.unsqueeze(0)
    if axis.shape[-1] == 0:
        effective_axis_shape = list(angle_rad.shape) + [3]
        axis_normalized = torch.zeros(effective_axis_shape, device=angle_rad.device, dtype=angle_rad.dtype)
        if axis_normalized.numel() > 0: axis_normalized[..., 0] = 1.0
    elif axis.dim() == 1 and angle_rad.shape[0] > 1 :
        axis_expanded = axis.unsqueeze(0).expand(angle_rad.shape[0], -1)
        axis_normalized = F.normalize(axis_expanded, p=2, dim=-1)
    elif axis.dim() == angle_rad.dim() and axis.shape[:-1] == angle_rad.shape:
        axis_normalized = F.normalize(axis, p=2, dim=-1)
    elif axis.dim() == 1 and angle_rad.shape[0] == 1:
        axis_expanded = axis.unsqueeze(0)
        axis_normalized = F.normalize(axis_expanded, p=2, dim=-1)
    else:
        try:
            target_axis_batch_shape = list(angle_rad.shape)
            current_axis_dim = axis.shape[-1]
            axis_expanded = axis.expand(target_axis_batch_shape + [current_axis_dim])
            axis_normalized = F.normalize(axis_expanded, p=2, dim=-1)
        except RuntimeError:
            logger_wubu_diffusion.warning(f"Quat axis-angle: Cannot broadcast axis (shape {axis.shape}) to align with angle_rad (shape {angle_rad.shape}). Using default axis [1,0,0].")
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

class HyperbolicUtils: # Unchanged
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

class PoincareBall: # Unchanged
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

# --- ManifoldUncrumpleTransform (Unchanged) ---
class ManifoldUncrumpleTransform(nn.Module):
    def __init__(self,
                 dim: int,
                 num_scaffold_points: int,
                 scaffold_init_scale: float = 0.01,
                 affinity_temp: float = 0.1,
                 hypernet_hidden_dim_factor: float = 1.0,
                 learn_rotation: bool = True,
                 learn_scale: bool = True,
                 scale_activation: str = 'sigmoid', 
                 scale_min: float = 0.5,
                 scale_max: float = 2.0,
                 learn_translation: bool = True,
                 stage_info: Optional[str] = None): 
        super().__init__()
        self.dim = dim
        self.stage_info_str = stage_info if stage_info else "" 

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
            self.scale_max = self.scale_min + 0.1 

        self.learn_translation = learn_translation

        self.scaffold_points = nn.Parameter(torch.randn(num_scaffold_points, dim) * scaffold_init_scale)

        hypernet_input_dim = dim
        hypernet_hidden_dim = max(dim, int(dim * hypernet_hidden_dim_factor))

        self.param_gen_parts = nn.ModuleDict()
        current_total_transform_params_indicator = 0 

        if self.learn_rotation:
            self.rot_param_dim = self.dim * (self.dim - 1) // 2
            if self.rot_param_dim > 0:
                self.param_gen_parts['rot_gen'] = nn.Sequential(
                    nn.Linear(hypernet_input_dim, hypernet_hidden_dim), nn.GELU(),
                    nn.Linear(hypernet_hidden_dim, self.rot_param_dim)
                )
                current_total_transform_params_indicator +=1
            else: self.learn_rotation = False 

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
        Ns, rp_dim = rot_params_batch.shape
        if not self.learn_rotation or self.rot_param_dim == 0 or rp_dim == 0:
            return torch.eye(self.dim, device=rot_params_batch.device, dtype=rot_params_batch.dtype).unsqueeze(0).expand(Ns, -1, -1)

        R_matrices = torch.empty(Ns, self.dim, self.dim, device=rot_params_batch.device, dtype=rot_params_batch.dtype)
        indices = torch.triu_indices(self.dim, self.dim, offset=1, device=rot_params_batch.device)
        
        for k in range(Ns):
            X_k = torch.zeros(self.dim, self.dim, device=rot_params_batch.device, dtype=rot_params_batch.dtype)
            if X_k.numel() > 0 and indices.numel() > 0 and self.rot_param_dim > 0 :
                 if indices.shape[1] == rot_params_batch[k].shape[0]: 
                    X_k[indices[0], indices[1]] = rot_params_batch[k]
                 else: 
                    logger_wubu_diffusion.warning(f"ManifoldUncrumpleTransform {self.stage_info_str}: Mismatch in rotation param dim for scaffold {k}. Expected {self.rot_param_dim}, got {rot_params_batch[k].shape[0]}. Filling partially.")
                    num_to_fill = min(indices.shape[1], rot_params_batch[k].shape[0])
                    if num_to_fill > 0 : X_k[indices[0,:num_to_fill], indices[1,:num_to_fill]] = rot_params_batch[k, :num_to_fill]
            
            X_k = X_k - X_k.T
            try:
                R_matrices[k] = torch.linalg.matrix_exp(torch.clamp(X_k, -3, 3)) 
            except Exception as e: 
                logger_wubu_diffusion.error(f"Matrix_exp error for scaffold {k} in {self.stage_info_str}: {e}. Using identity.", exc_info=False)
                R_matrices[k] = torch.eye(self.dim, device=X_k.device, dtype=X_k.dtype)
        return R_matrices

    def forward(self, v_current: torch.Tensor, current_sigma_gauss_skin: float, micro_level_idx: Optional[int] = None, stage_idx_override: Optional[int] = None) -> torch.Tensor:
        if self.is_identity:
            return v_current

        B, D = v_current.shape
        stage_id_for_log = stage_idx_override if stage_idx_override is not None else self.stage_info_str
        print_tensor_stats_mut(v_current, f"In_v_curr S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)

        diffs = v_current.unsqueeze(1) - self.scaffold_points.unsqueeze(0)
        sq_dists = torch.sum(diffs.pow(2), dim=-1)
        
        affinity_numerator = -sq_dists / (2 * (current_sigma_gauss_skin**2) + EPS)
        affinity_numerator_clamped = torch.clamp(affinity_numerator, min=-30.0, max=30.0)
        
        affinity_weights = F.softmax(affinity_numerator_clamped / self.affinity_temp, dim=-1)
        print_tensor_stats_mut(affinity_weights, f"AffW S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)

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
                s_params = torch.exp(torch.clamp(all_scale_gen_params, -3, 3)) 
            else: 
                s_params = F.softplus(all_scale_gen_params) + EPS 
            S_all_k = torch.clamp(s_params, min=self.scale_min, max=self.scale_max) 
        else:
            S_all_k = torch.ones(self.num_scaffold_points, self.dim, device=v_current.device, dtype=v_current.dtype)
        
        T_all_k = all_trans_gen_params if all_trans_gen_params is not None else \
                  torch.zeros(self.num_scaffold_points, self.dim, device=v_current.device, dtype=v_current.dtype)
        
        print_tensor_stats_mut(R_all_k, f"R_all_k S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        print_tensor_stats_mut(S_all_k, f"S_all_k S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        print_tensor_stats_mut(T_all_k, f"T_all_k S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)

        v_current_exp = v_current.unsqueeze(1) 
        
        v_scaled_per_scaffold = v_current_exp * S_all_k.unsqueeze(0) 
        print_tensor_stats_mut(v_scaled_per_scaffold, f"v_scaled_scaff S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        
        v_rotated_per_scaffold = torch.einsum('bki,kji->bkj', v_scaled_per_scaffold, R_all_k) 
        print_tensor_stats_mut(v_rotated_per_scaffold, f"v_rotated_scaff S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        
        v_transformed_per_scaffold = v_rotated_per_scaffold + T_all_k.unsqueeze(0) 
        print_tensor_stats_mut(v_transformed_per_scaffold, f"v_transformed_scaff S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        
        affinity_weights_exp = affinity_weights.unsqueeze(-1) 
        v_next = torch.sum(affinity_weights_exp * v_transformed_per_scaffold, dim=1) 
        print_tensor_stats_mut(v_next, f"Out_v_next S:{stage_id_for_log}", micro_level_idx=micro_level_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)

        return v_next

    def get_affinity_weights(self, v_current: torch.Tensor, current_sigma_gauss_skin: float) -> torch.Tensor:
        if self.is_identity: 
            return torch.ones(v_current.shape[0], self.num_scaffold_points, device=v_current.device, dtype=v_current.dtype) / self.num_scaffold_points

        diffs = v_current.unsqueeze(1) - self.scaffold_points.unsqueeze(0)
        sq_dists = torch.sum(diffs.pow(2), dim=-1)
        affinity_numerator = -sq_dists / (2 * (current_sigma_gauss_skin**2) + EPS)
        affinity_numerator_clamped = torch.clamp(affinity_numerator, min=-30.0, max=30.0)
        affinity_weights = F.softmax(affinity_numerator_clamped / self.affinity_temp, dim=-1)
        return affinity_weights

# --- OriginalFractalMicroTransformRungs (Unchanged) ---
class SkewSymmetricMatrix(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        if dim > 1:
            num_params = dim * (dim - 1) // 2
            self.skew_params = nn.Parameter(torch.randn(num_params) * 0.01)
            self.indices = torch.triu_indices(dim, dim, offset=1)
        elif dim == 1: self.skew_params = None 
        else: self.skew_params = None
    def forward(self) -> torch.Tensor:
        if self.dim <= 1 : return torch.eye(self.dim if self.dim > 0 else 1, device=self.skew_params.device if self.skew_params is not None else torch.device('cpu'), dtype=torch.float32)
        effective_skew_params = torch.tanh(self.skew_params) * ROT_PARAM_TANH_SCALE 
        X = torch.zeros(self.dim, self.dim, device=effective_skew_params.device, dtype=effective_skew_params.dtype)
        if X.numel() > 0 and self.indices.numel() > 0 and effective_skew_params.numel() > 0:
            if self.indices.shape[1] == effective_skew_params.shape[0]:
                 X[self.indices[0], self.indices[1]] = effective_skew_params
            else: 
                num_to_fill = min(self.indices.shape[1], effective_skew_params.shape[0])
                if num_to_fill > 0 : X[self.indices[0,:num_to_fill], self.indices[1,:num_to_fill]] = effective_skew_params[:num_to_fill]
        X = X - X.T 
        try: R_matrix = torch.linalg.matrix_exp(torch.clamp(X, -3.0, 3.0)) 
        except Exception as e: logger_wubu_diffusion.error(f"SkewSymmetricMatrix matrix_exp error: {e}. Using identity.", exc_info=False); R_matrix = torch.eye(self.dim, device=X.device, dtype=X.dtype)
        return R_matrix
        
class QuaternionSO4From6Params(nn.Module):
    def __init__(self):
        super().__init__()
        self.params_p = nn.Parameter(torch.randn(3) * 0.01) 
        self.params_q = nn.Parameter(torch.randn(3) * 0.01) 
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled_params_p = torch.tanh(self.params_p) * ROT_PARAM_TANH_SCALE; angle_p = torch.linalg.norm(scaled_params_p); axis_p = scaled_params_p / (angle_p + EPS) 
        p_quat = quaternion_from_axis_angle(angle_p, axis_p)
        scaled_params_q = torch.tanh(self.params_q) * ROT_PARAM_TANH_SCALE; angle_q = torch.linalg.norm(scaled_params_q); axis_q = scaled_params_q / (angle_q + EPS)
        q_quat = quaternion_from_axis_angle(angle_q, axis_q)
        return normalize_quaternion(p_quat), normalize_quaternion(q_quat)

class OriginalFractalMicroTransformRungs(nn.Module):
    def __init__(self, dim: int = 4, transform_type: str = "mlp", hidden_dim_factor: float = 1.0, use_quaternion_so4: bool = True,
                 enable_internal_sub_processing: bool = True, enable_stateful_micro_transform: bool = True, enable_hypernetwork_modulation: bool = True,
                 internal_state_dim_factor: float = 1.0, hyper_mod_strength: float = 0.01,
                 use_residual: bool = True, 
                 cond_embed_dim: int = 0, 
                 condition_on_s: bool = False, condition_on_c: bool = False, 
                 condition_on_micro_idx: bool = False, condition_on_stage_idx: bool = False,
                 max_micro_idx_embed: int = 1000, max_stage_idx_embed: int = 100
                 ):
        super().__init__()
        self.dim = dim
        self.use_quaternion_so4 = use_quaternion_so4 and (dim == 4)
        self.enable_internal_sub_processing = enable_internal_sub_processing and (dim == 4)
        self.enable_stateful_micro_transform = enable_stateful_micro_transform
        self.enable_hypernetwork_modulation = enable_hypernetwork_modulation
        self.hyper_mod_strength = hyper_mod_strength
        self.use_residual = use_residual
        self.debug_counter = 0

        self.condition_on_s = condition_on_s; self.condition_on_c = condition_on_c
        self.condition_on_micro_idx = condition_on_micro_idx; self.condition_on_stage_idx = condition_on_stage_idx
        self.cond_embed_dim = cond_embed_dim if cond_embed_dim > 0 and \
            (condition_on_s or condition_on_c or condition_on_micro_idx or condition_on_stage_idx) else 0
        self.s_cond_embed, self.c_cond_embed, self.micro_idx_cond_embed, self.stage_idx_cond_embed = None, None, None, None
        self.total_cond_dim_added = 0
        if self.cond_embed_dim > 0:
            if self.condition_on_s: self.s_cond_embed = nn.Sequential(nn.Linear(1, self.cond_embed_dim), nn.GELU()); self.total_cond_dim_added += self.cond_embed_dim
            if self.condition_on_c: self.c_cond_embed = nn.Sequential(nn.Linear(1, self.cond_embed_dim), nn.GELU()); self.total_cond_dim_added += self.cond_embed_dim
            if self.condition_on_micro_idx: self.micro_idx_cond_embed = nn.Embedding(max_micro_idx_embed, self.cond_embed_dim); self.total_cond_dim_added += self.cond_embed_dim
            if self.condition_on_stage_idx: self.stage_idx_cond_embed = nn.Embedding(max_stage_idx_embed, self.cond_embed_dim); self.total_cond_dim_added += self.cond_embed_dim
        
        if self.use_quaternion_so4: self.rotation_param_generator = QuaternionSO4From6Params(); self.rotation_generator_matrix = None
        elif dim > 0: self.rotation_generator_matrix = SkewSymmetricMatrix(dim); self.rotation_param_generator = None
        else: self.rotation_generator_matrix = nn.Identity(); self.rotation_param_generator = None

        mlp_hidden_dim = max(dim, int(dim * hidden_dim_factor)) if dim > 0 else 0
        effective_input_dim_for_mlp, effective_input_dim_for_hypernet = dim, dim
        if self.enable_hypernetwork_modulation: effective_input_dim_for_hypernet += self.total_cond_dim_added
        else: effective_input_dim_for_mlp += self.total_cond_dim_added
        effective_input_dim_for_mlp, effective_input_dim_for_hypernet = max(1, effective_input_dim_for_mlp), max(1, effective_input_dim_for_hypernet)

        if transform_type == 'mlp' and dim > 0 and mlp_hidden_dim > 0: self.non_rotational_map_layers = nn.ModuleList([nn.Linear(effective_input_dim_for_mlp, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim)])
        elif transform_type == 'linear' and dim > 0: self.non_rotational_map_layers = nn.ModuleList([nn.Linear(effective_input_dim_for_mlp, dim)])
        else: self.non_rotational_map_layers = nn.ModuleList([nn.Identity()])

        self.internal_state_dim, self.state_update_gate_mlp, self.state_influence_mlp = 0, None, None
        if self.enable_stateful_micro_transform and self.dim > 0:
            self.internal_state_dim = max(1, int(self.dim * internal_state_dim_factor)); state_mlp_hidden = max(self.internal_state_dim // 2, dim // 2, 1)
            self.state_update_gate_mlp = nn.Sequential(nn.Linear(dim + self.internal_state_dim, state_mlp_hidden), nn.GELU(), nn.Linear(state_mlp_hidden, self.internal_state_dim), nn.Tanh())
            self.state_influence_mlp = nn.Sequential(nn.Linear(self.internal_state_dim, state_mlp_hidden), nn.GELU(), nn.Linear(state_mlp_hidden, dim))

        self.hyper_bias_generator = None
        if self.enable_hypernetwork_modulation and self.dim > 0:
            if len(self.non_rotational_map_layers) > 0 and isinstance(self.non_rotational_map_layers[0], nn.Linear):
                first_linear_out_features = self.non_rotational_map_layers[0].out_features; hyper_hidden_dim = max(dim // 2, first_linear_out_features // 2, 1)
                self.hyper_bias_generator = nn.Sequential(nn.Linear(effective_input_dim_for_hypernet, hyper_hidden_dim), nn.GELU(), nn.Linear(hyper_hidden_dim, first_linear_out_features))
            else: self.enable_hypernetwork_modulation = False
            if self.enable_hypernetwork_modulation and self.hyper_bias_generator is None : logger_wubu_diffusion.warning("HyperNet enabled but no suitable MLP/hyper_bias_generator. Disabling."); self.enable_hypernetwork_modulation = False
        
        self.w_modulator, self.v_scaler_param = None, None
        if self.enable_internal_sub_processing and self.dim == 4: self.w_modulator = nn.Sequential(nn.Linear(1, 2), nn.GELU(), nn.Linear(2, 1), nn.Sigmoid()); self.v_scaler_param = nn.Parameter(torch.randn(1) * 0.01)
        
        self.output_norm_origfmt = nn.LayerNorm(dim) if dim > 0 else nn.Identity()
        self.apply(init_weights_general)

    def _get_conditional_features(self, B: int, device: torch.device, dtype: torch.dtype, s_val, c_val, micro_idx_in_stage_val, current_stage_idx_val) -> Optional[torch.Tensor]:
        if self.total_cond_dim_added == 0: return None
        cond_feats_list = []
        if self.condition_on_s and self.s_cond_embed and s_val is not None: cond_feats_list.append(self.s_cond_embed(torch.full((B, 1), s_val, device=device, dtype=dtype)))
        if self.condition_on_c and self.c_cond_embed and c_val is not None: cond_feats_list.append(self.c_cond_embed(torch.full((B, 1), c_val, device=device, dtype=dtype)))
        if self.condition_on_micro_idx and self.micro_idx_cond_embed and micro_idx_in_stage_val is not None: cond_feats_list.append(self.micro_idx_cond_embed(torch.full((B,), micro_idx_in_stage_val, device=device, dtype=torch.long)))
        if self.condition_on_stage_idx and self.stage_idx_cond_embed and current_stage_idx_val is not None: cond_feats_list.append(self.stage_idx_cond_embed(torch.full((B,), current_stage_idx_val, device=device, dtype=torch.long)))
        return torch.cat(cond_feats_list, dim=-1) if cond_feats_list else None

    def _apply_internal_quaternion_sub_processing(self, x_q: torch.Tensor, micro_level_idx: Optional[int], stage_idx: Optional[int]) -> torch.Tensor:
        if not (self.enable_internal_sub_processing and x_q.shape[-1] == 4 and self.w_modulator and self.v_scaler_param is not None): return x_q
        print_tensor_stats_fdq(x_q, "ISP_In_x_q", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS); w, v = x_q[..., 0:1], x_q[..., 1:4]; w_modulation_factor = self.w_modulator(w)
        v_scaled = v * w_modulation_factor * (torch.tanh(self.v_scaler_param) + 1.0); x_q_processed = torch.cat([w, v_scaled], dim=-1); x_q_norm = normalize_quaternion(x_q_processed)
        print_tensor_stats_fdq(x_q_norm, "ISP_Out_Norm", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        return x_q_norm

    def apply_rotation(self, x_tan: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        R_matrix_this_step: Optional[torch.Tensor] = None
        if self.dim <= 0: return x_tan, None
        if self.use_quaternion_so4 and self.rotation_param_generator is not None:
            p_quat, q_quat = self.rotation_param_generator(); p_quat, q_quat = p_quat.to(x_tan.device, x_tan.dtype), q_quat.to(x_tan.device, x_tan.dtype)
            if x_tan.shape[-1] != 4: logger_wubu_diffusion.error(f"Quat rot needs dim 4, got {x_tan.shape[-1]}. Skip."); return x_tan, None
            if x_tan.dim() > 1 and p_quat.dim() == 1: p_b, q_b = p_quat.unsqueeze(0).expand_as(x_tan), q_quat.unsqueeze(0).expand_as(x_tan)
            elif x_tan.dim() == p_quat.dim(): p_b, q_b = p_quat, q_quat
            else: logger_wubu_diffusion.error(f"Quat rot broadcast error. Skip."); return x_tan, None
            rotated_x_tan = quaternion_multiply(quaternion_multiply(p_b, x_tan), quaternion_conjugate(q_b)) # Standard SO(4) rotation p x q^-1
        elif self.rotation_generator_matrix is not None and isinstance(self.rotation_generator_matrix, SkewSymmetricMatrix):
            R_matrix_this_step = self.rotation_generator_matrix().to(x_tan.device, x_tan.dtype)
            if x_tan.dim() == 2: rotated_x_tan = torch.einsum('ij,bj->bi', R_matrix_this_step, x_tan)
            elif x_tan.dim() == 1: rotated_x_tan = torch.matmul(R_matrix_this_step, x_tan)
            elif x_tan.dim() == 3: rotated_x_tan = torch.einsum('ij,bnj->bni', R_matrix_this_step, x_tan)
            else: rotated_x_tan = x_tan; logger_wubu_diffusion.warning(f"MicroTrans apply_rot: Unexpected x_tan dim {x_tan.dim()}. Skip.")
        elif isinstance(self.rotation_generator_matrix, nn.Identity): rotated_x_tan = x_tan
        else: rotated_x_tan = x_tan; logger_wubu_diffusion.error("Rot module misconfigured.")
        return rotated_x_tan, R_matrix_this_step

    def _apply_non_rotational_map(self, x_in: torch.Tensor, original_main_tan_for_hypernet: torch.Tensor, conditional_features: Optional[torch.Tensor], micro_level_idx: Optional[int], stage_idx: Optional[int]) -> torch.Tensor:
        if not self.non_rotational_map_layers or isinstance(self.non_rotational_map_layers[0], nn.Identity): return x_in
        x_processed, first_linear_layer = x_in, self.non_rotational_map_layers[0]
        if self.enable_hypernetwork_modulation and self.hyper_bias_generator is not None and isinstance(first_linear_layer, nn.Linear):
            hypernet_base_input_eff = original_main_tan_for_hypernet
            if conditional_features is not None:
                if hypernet_base_input_eff.shape[0]!=conditional_features.shape[0]: 
                    if conditional_features.shape[0]==1: conditional_features=conditional_features.expand(hypernet_base_input_eff.shape[0],-1)
                    elif hypernet_base_input_eff.shape[0]==1: hypernet_base_input_eff=hypernet_base_input_eff.expand(conditional_features.shape[0],-1)
                    else: raise ValueError(f"Batch mismatch hypernet ({hypernet_base_input_eff.shape[0]}) vs cond ({conditional_features.shape[0]})")
                hypernet_base_input_eff = torch.cat([hypernet_base_input_eff, conditional_features], dim=-1)
            dynamic_bias_offset = self.hyper_bias_generator(hypernet_base_input_eff)
            print_tensor_stats_fdq(dynamic_bias_offset, "Hyper_BiasOffset", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
            effective_bias = (first_linear_layer.bias + dynamic_bias_offset * self.hyper_mod_strength) if first_linear_layer.bias is not None else (dynamic_bias_offset * self.hyper_mod_strength)
            x_processed = F.linear(x_processed, first_linear_layer.weight, effective_bias)
            for i in range(1, len(self.non_rotational_map_layers)): x_processed = self.non_rotational_map_layers[i](x_processed)
        else: 
            if conditional_features is not None:
                if x_processed.shape[0]!=conditional_features.shape[0]:
                    if conditional_features.shape[0]==1: conditional_features=conditional_features.expand(x_processed.shape[0],-1)
                    elif x_processed.shape[0]==1: x_processed=x_processed.expand(conditional_features.shape[0], -1)
                    else: raise ValueError(f"Batch mismatch MLP ({x_processed.shape[0]}) vs cond ({conditional_features.shape[0]})")
                x_processed = torch.cat([x_processed, conditional_features], dim=-1)
            temp_x = x_processed 
            for layer_idx, layer in enumerate(self.non_rotational_map_layers): print_tensor_stats_fdq(temp_x, f"Pre_NonRotL{layer_idx}", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS); temp_x = layer(temp_x)
            x_processed = temp_x
        return x_processed

    def forward(self, main_tan: torch.Tensor, current_internal_state_in: Optional[torch.Tensor]=None, scaffold_modulation: Optional[torch.Tensor]=None, micro_level_idx: Optional[int]=None, stage_idx: Optional[int]=None, s_val: Optional[float] = None, c_val: Optional[float] = None, micro_idx_in_stage_val: Optional[int] = None, current_stage_idx_val: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.debug_counter +=1
        if self.dim <= 0: return main_tan, None, current_internal_state_in
        print_tensor_stats_fdq(main_tan, "MF_In_main", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        original_main_tan_for_hypernet_or_cond = main_tan 
        cond_feats = self._get_conditional_features(main_tan.shape[0], main_tan.device, main_tan.dtype, s_val, c_val, micro_idx_in_stage_val, current_stage_idx_val)
        print_tensor_stats_fdq(cond_feats, "MF_CondFeats", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        x_intermediate = self._apply_internal_quaternion_sub_processing(main_tan, micro_level_idx, stage_idx) if self.enable_internal_sub_processing else main_tan
        print_tensor_stats_fdq(x_intermediate, "MF_PostISP", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        rotated_x, R_matrix = self.apply_rotation(x_intermediate)
        print_tensor_stats_fdq(rotated_x, "MF_PostRot", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        if R_matrix is not None: print_tensor_stats_fdq(R_matrix, "MF_R_matrix", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        mapped_x = self._apply_non_rotational_map(rotated_x, original_main_tan_for_hypernet_or_cond, cond_feats, micro_level_idx, stage_idx)
        print_tensor_stats_fdq(mapped_x, "MF_PostMap", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        final_tan_before_scaffold_and_residual, next_state = mapped_x, current_internal_state_in
        if self.enable_stateful_micro_transform and current_internal_state_in is not None and self.state_influence_mlp and self.state_update_gate_mlp:
            print_tensor_stats_fdq(current_internal_state_in, "STF_In_State", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
            influence = self.state_influence_mlp(current_internal_state_in); print_tensor_stats_fdq(influence, "STF_Influence", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
            processed_with_state = mapped_x + influence; state_update_in = torch.cat([processed_with_state, current_internal_state_in], dim=-1); delta = self.state_update_gate_mlp(state_update_in)
            print_tensor_stats_fdq(delta, "STF_Delta", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
            next_state = torch.tanh(current_internal_state_in + delta); final_tan_before_scaffold_and_residual = processed_with_state
            print_tensor_stats_fdq(next_state, "STF_Out_State", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        transformed_output = (main_tan + final_tan_before_scaffold_and_residual) if self.use_residual else final_tan_before_scaffold_and_residual
        if self.use_residual: print_tensor_stats_fdq(transformed_output, "MF_PostResidual", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        final_tan_after_modulation = transformed_output
        if scaffold_modulation is not None:
            mod=scaffold_modulation
            if final_tan_after_modulation.dim()>1 and mod.dim()==1 and self.dim>0: mod=mod.unsqueeze(0).expand_as(final_tan_after_modulation)
            elif final_tan_after_modulation.dim()==1 and mod.dim()==2 and mod.shape[0]==1 and self.dim>0: mod=mod.squeeze(0)
            if final_tan_after_modulation.shape==mod.shape: final_tan_after_modulation += mod
            print_tensor_stats_fdq(final_tan_after_modulation, "MF_PostScaffMod", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        if isinstance(self.output_norm_origfmt, nn.LayerNorm): final_tan_after_modulation = self.output_norm_origfmt(final_tan_after_modulation); print_tensor_stats_fdq(final_tan_after_modulation, "MF_PostOutputLN", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        clamped_tan = torch.clamp(final_tan_after_modulation, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        print_tensor_stats_fdq(clamped_tan, "MF_Out_Clamp", micro_level_idx, stage_idx, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
        if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 20 == 0: print("-" * 20)
        return clamped_tan, R_matrix, next_state

# --- FractalDepthQuaternionWuBuRungs (Unchanged) ---
class FractalDepthQuaternionWuBuRungs(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_virtual_micro_levels: int = 10000, base_micro_level_dim: int = 4, num_physical_transform_stages: int = 10,
                 initial_s: float = 1.0, s_decay_factor_per_micro_level: float = 0.9999, initial_c_base: float = 1.0, c_phi_influence: bool = True,
                 num_phi_scaffold_points_per_stage: int = 5, phi_scaffold_init_scale_factor: float = 0.05, use_gaussian_rungs: bool = True, 
                 base_gaussian_std_dev_factor_rung: float = 0.02, gaussian_std_dev_decay_factor_rung: float = 0.99999, rung_affinity_temperature: float = 0.1, 
                 rung_modulation_strength: float = 0.05, t_tilde_activation_scale: float = 1.0,
                 micro_transform_type: str = "mlp", micro_transform_hidden_factor: float = 1.0,
                 use_quaternion_so4_micro: bool = True, scaffold_co_rotation_mode: str = "none",
                 enable_internal_sub_processing: bool = True, enable_stateful_micro_transform: bool = True,
                 enable_hypernetwork_modulation: bool = True, internal_state_dim_factor: float = 1.0, hyper_mod_strength: float = 0.01,
                 use_residual_in_micro: bool = True, 
                 learnable_affinity_params_per_stage: bool = True, 
                 use_manifold_uncrumple_transform: bool = False, 
                 uncrumple_hypernet_hidden_factor: float = 1.0, uncrumple_learn_rotation: bool = True,
                 uncrumple_learn_scale: bool = True, uncrumple_scale_activation: str = 'sigmoid',
                 uncrumple_scale_min: float = 0.5, uncrumple_scale_max: float = 2.0,
                 uncrumple_learn_translation: bool = True,
                 wubu_cond_embed_dim: int = 0, wubu_condition_on_s: bool = False, 
                 wubu_condition_on_c: bool = False, wubu_condition_on_micro_idx: bool = False, 
                 wubu_condition_on_stage_idx: bool = False,
                 wubu_max_micro_idx_embed: int = 1000, wubu_max_stage_idx_embed: int = 100, 
                 wubu_inter_micro_norm_interval: int = 0 
                 ):
        super().__init__()
        self.logger_fractal_rungs = logger_wubu_diffusion
        self.num_virtual_micro_levels = num_virtual_micro_levels; self.base_micro_level_dim = max(0, base_micro_level_dim)
        self.enable_stateful_micro_transform_globally = enable_stateful_micro_transform
        self.use_manifold_uncrumple_transform = use_manifold_uncrumple_transform
        self.learnable_affinity_params_per_stage = learnable_affinity_params_per_stage
        input_dim = max(0,input_dim); output_dim = max(0,output_dim)

        if self.use_manifold_uncrumple_transform: self.logger_fractal_rungs.info("FDQRungs will use ManifoldUncrumpleTransform for its physical stages."); self.use_gaussian_rungs = False 
        elif self.base_micro_level_dim != 4:
            if use_quaternion_so4_micro: self.logger_fractal_rungs.warning("use_quaternion_so4_micro=True but base_dim!=4. Forced off for OriginalFMT."); use_quaternion_so4_micro = False
            if enable_internal_sub_processing: self.logger_fractal_rungs.warning("enable_internal_sub_processing=True but base_dim!=4. Forced off for OriginalFMT."); enable_internal_sub_processing = False
        
        self.num_physical_transform_stages = max(1, num_physical_transform_stages)
        self.micro_levels_per_stage = max(1, self.num_virtual_micro_levels // self.num_physical_transform_stages) if self.num_virtual_micro_levels > 0 and self.num_physical_transform_stages > 0 else 1
        self.wubu_max_micro_idx_embed = max(wubu_max_micro_idx_embed, self.micro_levels_per_stage + 10)
        self.wubu_max_stage_idx_embed = max(wubu_max_stage_idx_embed, self.num_physical_transform_stages + 10)

        self.initial_s, self.s_decay_factor = initial_s, s_decay_factor_per_micro_level
        self.initial_c_base, self.c_phi_influence = initial_c_base, c_phi_influence
        self.min_curvature, self.t_tilde_activation_scale = EPS, t_tilde_activation_scale
        self.scaffold_co_rotation_mode = scaffold_co_rotation_mode 

        if input_dim > 0 and self.base_micro_level_dim > 0: self.input_projection, self.input_layernorm = (nn.Linear(input_dim, self.base_micro_level_dim) if input_dim != self.base_micro_level_dim else nn.Identity()), nn.LayerNorm(self.base_micro_level_dim)
        elif input_dim == 0 and self.base_micro_level_dim > 0: self.input_projection, self.input_layernorm = nn.Linear(1, self.base_micro_level_dim), nn.LayerNorm(self.base_micro_level_dim)
        else: self.input_projection, self.input_layernorm = nn.Identity(), nn.Identity()

        self.wubu_inter_micro_norm_interval = wubu_inter_micro_norm_interval
        if self.wubu_inter_micro_norm_interval > 0 and self.base_micro_level_dim > 0: self.inter_micro_layernorm = nn.LayerNorm(self.base_micro_level_dim); self.logger_fractal_rungs.info(f"FDQRungs: Inter-micro-level LayerNorm ENABLED every {self.wubu_inter_micro_norm_interval} levels.")
        else: self.inter_micro_layernorm = None

        self.physical_micro_transforms, self.log_stage_curvatures_unconstrained = nn.ModuleList(), nn.ParameterList()
        self.phi_scaffold_base_tangent_vectors, self.log_sigma_gauss_skin_stages, self.log_rung_affinity_temp_stages = None, None, None
        if self.num_physical_transform_stages > 0 and self.base_micro_level_dim > 0:
            if self.use_manifold_uncrumple_transform:
                for i in range(self.num_physical_transform_stages): self.physical_micro_transforms.append(ManifoldUncrumpleTransform(dim=self.base_micro_level_dim, num_scaffold_points=num_phi_scaffold_points_per_stage, scaffold_init_scale=phi_scaffold_init_scale_factor, affinity_temp=rung_affinity_temperature, hypernet_hidden_dim_factor=uncrumple_hypernet_hidden_factor, learn_rotation=uncrumple_learn_rotation, learn_scale=uncrumple_learn_scale, scale_activation=uncrumple_scale_activation, scale_min=uncrumple_scale_min, scale_max=uncrumple_scale_max, learn_translation=uncrumple_learn_translation, stage_info=f"PStage_{i}"))
            else: 
                for _ in range(self.num_physical_transform_stages): self.physical_micro_transforms.append(OriginalFractalMicroTransformRungs(dim=self.base_micro_level_dim, transform_type=micro_transform_type, hidden_dim_factor=micro_transform_hidden_factor, use_quaternion_so4=use_quaternion_so4_micro, enable_internal_sub_processing=enable_internal_sub_processing, enable_stateful_micro_transform=self.enable_stateful_micro_transform_globally, enable_hypernetwork_modulation=enable_hypernetwork_modulation, internal_state_dim_factor=internal_state_dim_factor, hyper_mod_strength=hyper_mod_strength, use_residual=use_residual_in_micro, cond_embed_dim=wubu_cond_embed_dim, condition_on_s=wubu_condition_on_s, condition_on_c=wubu_condition_on_c, condition_on_micro_idx=wubu_condition_on_micro_idx, condition_on_stage_idx=wubu_condition_on_stage_idx, max_micro_idx_embed=self.wubu_max_micro_idx_embed, max_stage_idx_embed=self.wubu_max_stage_idx_embed))
                safe_initial_c_base = max(EPS*10, initial_c_base); self.log_stage_curvatures_unconstrained = nn.ParameterList([nn.Parameter(torch.tensor(math.log(math.expm1(safe_initial_c_base)))) for _ in range(self.num_physical_transform_stages)])
                if num_phi_scaffold_points_per_stage > 0: self.phi_scaffold_base_tangent_vectors = nn.ParameterList([nn.Parameter(torch.randn(num_phi_scaffold_points_per_stage, self.base_micro_level_dim) * phi_scaffold_init_scale_factor) for _ in range(self.num_physical_transform_stages)])
                if self.learnable_affinity_params_per_stage:
                    initial_sigma_guess, initial_temp_guess = max(EPS, base_gaussian_std_dev_factor_rung * 2.0), max(EPS, rung_affinity_temperature)
                    self.log_sigma_gauss_skin_stages = nn.ParameterList([nn.Parameter(torch.tensor(math.log(initial_sigma_guess))) for _ in range(self.num_physical_transform_stages)])
                    self.log_rung_affinity_temp_stages = nn.ParameterList([nn.Parameter(torch.tensor(math.log(initial_temp_guess))) for _ in range(self.num_physical_transform_stages)])
                    self.logger_fractal_rungs.info(f"FDQRungs (OriginalFMT): Learnable affinity sigma/temp per stage ENABLED. Initial sigma guess: {initial_sigma_guess:.2e}, temp guess: {initial_temp_guess:.2e}")
        
        self.num_phi_scaffold_points_per_stage = num_phi_scaffold_points_per_stage 
        self.use_gaussian_rungs, self.base_gaussian_std_dev_factor_rung = use_gaussian_rungs, base_gaussian_std_dev_factor_rung
        self.gaussian_std_dev_decay_factor_rung, self.rung_affinity_temperature_base = gaussian_std_dev_decay_factor_rung, max(EPS, rung_affinity_temperature) 
        self.rung_modulation_strength = rung_modulation_strength
        
        if self.base_micro_level_dim > 0 and output_dim > 0 : self.output_projection = nn.Linear(self.base_micro_level_dim, output_dim) if self.base_micro_level_dim != output_dim else nn.Identity()
        elif self.base_micro_level_dim == 0 and output_dim > 0: self.output_projection = nn.Linear(1, output_dim)
        else: self.output_projection = nn.Identity()

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        in_feat_proj = self.input_projection.in_features if isinstance(self.input_projection, nn.Linear) else input_dim
        out_feat_proj = self.output_projection.out_features if isinstance(self.output_projection, nn.Linear) else output_dim
        transform_mode_str = "ManifoldUncrumpleTransform" if self.use_manifold_uncrumple_transform else "OriginalFMT"
        self.logger_fractal_rungs.info(f"FDQRungs Init ({transform_mode_str}): In {in_feat_proj}D -> {self.num_virtual_micro_levels} virtLvl ({self.base_micro_level_dim}D) over {self.num_physical_transform_stages} physStgs -> Out {out_feat_proj}D. Params: {param_count:,}.")
        if not self.use_manifold_uncrumple_transform:
             cond_str_parts = [c for c,v in [("s",wubu_condition_on_s),("c",wubu_condition_on_c),("mIdx",wubu_condition_on_micro_idx),("sIdx",wubu_condition_on_stage_idx)] if v]
             cond_active_str = f"CondOn:[{','.join(cond_str_parts)}](Dim:{wubu_cond_embed_dim})" if cond_str_parts else "CondOn:[]"
             self.logger_fractal_rungs.info(f"  OriginalFMT AdvMicro: SubP={enable_internal_sub_processing}, StateF={self.enable_stateful_micro_transform_globally}, HyperM={enable_hypernetwork_modulation}, ResidM={use_residual_in_micro}, {cond_active_str}")

    def get_s_c_gsigma_at_micro_level(self, micro_level_idx: int, stage_idx: int) -> Tuple[float, float, float, float]:
        s_i = max(EPS, self.initial_s * (self.s_decay_factor ** micro_level_idx)); c_i = self.initial_c_base
        if not self.use_manifold_uncrumple_transform and self.log_stage_curvatures_unconstrained and stage_idx < len(self.log_stage_curvatures_unconstrained):
            c_i = get_constrained_param_val(self.log_stage_curvatures_unconstrained[stage_idx], self.min_curvature).item()
            if self.c_phi_influence and self.micro_levels_per_stage > 0: c_i *= (PHI ** ((micro_level_idx % self.micro_levels_per_stage) % 4 - 1.5))
        c_i = max(self.min_curvature, c_i)
        sigma_gauss_i_skin_val, affinity_temp_val = self.base_gaussian_std_dev_factor_rung * s_i * (self.gaussian_std_dev_decay_factor_rung ** micro_level_idx), self.rung_affinity_temperature_base
        if self.learnable_affinity_params_per_stage and not self.use_manifold_uncrumple_transform and self.log_sigma_gauss_skin_stages and stage_idx < len(self.log_sigma_gauss_skin_stages) and self.log_rung_affinity_temp_stages and stage_idx < len(self.log_rung_affinity_temp_stages):
            sigma_gauss_i_skin_val, affinity_temp_val = get_constrained_param_val(self.log_sigma_gauss_skin_stages[stage_idx], EPS*10).item(), get_constrained_param_val(self.log_rung_affinity_temp_stages[stage_idx], EPS).item()
        return s_i, c_i, max(EPS*100, sigma_gauss_i_skin_val), max(EPS, affinity_temp_val)

    def _propagate_scaffold_points(self, base_scaffold_tan_vectors_stage: torch.Tensor, accumulated_R_matrix_stage: Optional[torch.Tensor], current_s_i: float, initial_s_for_stage: float) -> torch.Tensor:
        propagated_scaffold = base_scaffold_tan_vectors_stage * (initial_s_for_stage / max(current_s_i, EPS)) 
        if accumulated_R_matrix_stage is not None and self.scaffold_co_rotation_mode == "matrix_only": propagated_scaffold = torch.einsum('ij,kj->ki', accumulated_R_matrix_stage.to(propagated_scaffold.device, propagated_scaffold.dtype), propagated_scaffold)
        return propagated_scaffold

    def forward(self, x_input: torch.Tensor, show_progress: bool = False, progress_desc: Optional[str] = None) -> torch.Tensor:
        input_original_dim_fwd, B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = x_input.dim(), -1, -1, -1
        if input_original_dim_fwd == 3: B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = x_input.shape; current_v_tan_fwd, current_batch_size = x_input.reshape(B_orig_fwd * S_orig_fwd, d_in_runtime_fwd), B_orig_fwd * S_orig_fwd
        elif input_original_dim_fwd == 2: current_batch_size, d_in_runtime_fwd = x_input.shape; current_v_tan_fwd, B_orig_fwd, S_orig_fwd = x_input, current_batch_size, 1
        elif input_original_dim_fwd == 1 : current_v_tan_fwd, d_in_runtime_fwd, B_orig_fwd, S_orig_fwd, current_batch_size = x_input.unsqueeze(0), x_input.shape[0], 1,1,1
        else: raise ValueError(f"FDQRungs forward expects 1D, 2D or 3D input, got {input_original_dim_fwd}D.")
        if d_in_runtime_fwd == 0 and self.base_micro_level_dim > 0 and isinstance(self.input_projection, nn.Linear) and self.input_projection.in_features == 1: current_v_tan_fwd = torch.ones(current_v_tan_fwd.shape[0], 1, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
        current_v_tan_fwd = self.input_projection(current_v_tan_fwd)
        if isinstance(self.input_layernorm, nn.LayerNorm) and self.base_micro_level_dim > 0: current_v_tan_fwd = self.input_layernorm(current_v_tan_fwd)
        if self.num_virtual_micro_levels == 0 or not self.physical_micro_transforms: final_v_tan_fwd = current_v_tan_fwd
        else:
            accumulated_R_for_stage = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype) if self.base_micro_level_dim > 0 and not self.use_manifold_uncrumple_transform else None
            s_at_stage_start_val, batched_internal_micro_state = self.initial_s, None
            if not self.use_manifold_uncrumple_transform and self.enable_stateful_micro_transform_globally and self.base_micro_level_dim > 0 and len(self.physical_micro_transforms) > 0 and hasattr(self.physical_micro_transforms[0], 'enable_stateful_micro_transform') and self.physical_micro_transforms[0].enable_stateful_micro_transform and self.physical_micro_transforms[0].internal_state_dim > 0: batched_internal_micro_state = torch.zeros(current_batch_size, self.physical_micro_transforms[0].internal_state_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
            micro_level_iterator_range = range(self.num_virtual_micro_levels)
            in_feat_display = self.input_projection.in_features if hasattr(self.input_projection, 'in_features') and isinstance(self.input_projection, nn.Linear) else d_in_runtime_fwd
            out_feat_display = self.output_projection.out_features if hasattr(self.output_projection, 'out_features') and isinstance(self.output_projection, nn.Linear) else 'N/A' 
            effective_progress_desc = progress_desc if progress_desc else f"FDQRungs ({in_feat_display}->{out_feat_display})"
            tqdm_iterator_obj = tqdm(micro_level_iterator_range, desc=effective_progress_desc, total=self.num_virtual_micro_levels, leave=False, dynamic_ncols=True, disable=not show_progress) if show_progress else None
            micro_level_iterator = tqdm_iterator_obj if tqdm_iterator_obj else micro_level_iterator_range
            for micro_i_fwd in micro_level_iterator:
                current_stage_idx_fwd = min(micro_i_fwd // self.micro_levels_per_stage, len(self.physical_micro_transforms)-1)
                micro_idx_in_stage_fwd = micro_i_fwd % self.micro_levels_per_stage; physical_transform_module_fwd = self.physical_micro_transforms[current_stage_idx_fwd]
                if micro_idx_in_stage_fwd == 0: 
                    s_at_stage_start_val, _, _, _ = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd) 
                    if not self.use_manifold_uncrumple_transform:
                        if self.base_micro_level_dim > 0: accumulated_R_for_stage = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
                        if self.enable_stateful_micro_transform_globally and self.base_micro_level_dim > 0 and hasattr(physical_transform_module_fwd, 'enable_stateful_micro_transform') and physical_transform_module_fwd.enable_stateful_micro_transform and physical_transform_module_fwd.internal_state_dim > 0: batched_internal_micro_state = torch.zeros(current_batch_size, physical_transform_module_fwd.internal_state_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
                s_i_fwd, c_i_fwd, sigma_gauss_skin_i_fwd, affinity_temp_i_fwd = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)
                if self.use_manifold_uncrumple_transform: transformed_v_tan_micro, R_step, next_state = physical_transform_module_fwd(current_v_tan_fwd, sigma_gauss_skin_i_fwd, micro_i_fwd, current_stage_idx_fwd), None, None
                else: 
                    scaffold_modulation_input = None
                    if self.use_gaussian_rungs and self.phi_scaffold_base_tangent_vectors and self.num_phi_scaffold_points_per_stage > 0 and self.base_micro_level_dim > 0 and current_stage_idx_fwd < len(self.phi_scaffold_base_tangent_vectors):
                        base_scaffolds, R_prop = self.phi_scaffold_base_tangent_vectors[current_stage_idx_fwd].to(current_v_tan_fwd.device,current_v_tan_fwd.dtype), accumulated_R_for_stage if self.scaffold_co_rotation_mode == "matrix_only" else None
                        propagated_scaffolds = self._propagate_scaffold_points(base_scaffolds, R_prop, s_i_fwd, s_at_stage_start_val)
                        sq_dists = torch.sum((current_v_tan_fwd.unsqueeze(1) - propagated_scaffolds.unsqueeze(0))**2, dim=-1)
                        affinity_weights = F.softmax(torch.clamp(-sq_dists / (2*(sigma_gauss_skin_i_fwd**2)+EPS*100),-30.0,30.0) / affinity_temp_i_fwd, dim=-1) 
                        scaffold_modulation_input = self.rung_modulation_strength * torch.einsum('bn,nd->bd', affinity_weights, propagated_scaffolds)
                    transformed_v_tan_micro, R_step, next_state = physical_transform_module_fwd(current_v_tan_fwd, batched_internal_micro_state, scaffold_modulation_input, micro_i_fwd, current_stage_idx_fwd, s_i_fwd, c_i_fwd, micro_idx_in_stage_fwd, current_stage_idx_fwd)
                    if self.enable_stateful_micro_transform_globally: batched_internal_micro_state = next_state
                    if R_step is not None and self.scaffold_co_rotation_mode == "matrix_only" and accumulated_R_for_stage is not None: accumulated_R_for_stage = torch.matmul(R_step, accumulated_R_for_stage)
                current_v_tan_fwd = transformed_v_tan_micro * self.t_tilde_activation_scale if self.t_tilde_activation_scale != 1.0 else transformed_v_tan_micro
                if self.inter_micro_layernorm is not None and (micro_i_fwd + 1) % self.wubu_inter_micro_norm_interval == 0: current_v_tan_fwd = self.inter_micro_layernorm(current_v_tan_fwd); print_tensor_stats_fdq(current_v_tan_fwd, "InterMicroLN", micro_i_fwd, current_stage_idx_fwd, enabled=DEBUG_MICRO_TRANSFORM_INTERNALS)
                if tqdm_iterator_obj and hasattr(tqdm_iterator_obj,'set_postfix') and micro_i_fwd > 0 and (micro_i_fwd % (max(1,self.num_virtual_micro_levels//100))==0):
                    log_dict = {"s":f"{s_i_fwd:.2e}","c":f"{c_i_fwd:.2e}","σ":f"{sigma_gauss_skin_i_fwd:.2e}", "affT": f"{affinity_temp_i_fwd:.2e}"}
                    if not self.use_manifold_uncrumple_transform and scaffold_modulation_input is not None and 'affinity_weights' in locals() and affinity_weights.numel()>0 : log_dict["aff_w_max"] = f"{affinity_weights.max().item():.2e}"
                    if not self.use_manifold_uncrumple_transform and batched_internal_micro_state is not None and batched_internal_micro_state.numel()>0: log_dict["st_n"] = f"{torch.linalg.norm(batched_internal_micro_state[0] if batched_internal_micro_state.dim()>1 else batched_internal_micro_state).item():.2e}"
                    tqdm_iterator_obj.set_postfix(log_dict)
            final_v_tan_fwd = current_v_tan_fwd
            if tqdm_iterator_obj and hasattr(tqdm_iterator_obj,'close'): tqdm_iterator_obj.close()
        if self.base_micro_level_dim == 0 and output_dim > 0 and isinstance(self.output_projection, nn.Linear) and self.output_projection.in_features == 1: output_features_fwd = self.output_projection(torch.ones(final_v_tan_fwd.shape[0], 1, device=final_v_tan_fwd.device, dtype=final_v_tan_fwd.dtype))
        else: output_features_fwd = self.output_projection(final_v_tan_fwd)
        if input_original_dim_fwd == 3 and B_orig_fwd != -1 and S_orig_fwd != -1: final_output_dim_val = output_features_fwd.shape[-1] if output_features_fwd.numel() > 0 else 0; return output_features_fwd.reshape(B_orig_fwd, S_orig_fwd, final_output_dim_val)
        elif input_original_dim_fwd == 2 : return output_features_fwd
        elif input_original_dim_fwd == 1: return output_features_fwd.squeeze(0) if output_features_fwd.shape[0] == 1 and output_features_fwd.dim() > 1 else output_features_fwd
        return output_features_fwd

# --- Dataset (Unchanged) ---
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
                    if frame_np.ndim == 3 and frame_np.shape[-1] in [3,4]: frame_np = np.transpose(frame_np[...,:3], (2,0,1)) # C,H,W
                    elif frame_np.ndim == 2: frame_np = np.expand_dims(frame_np, axis=0) # 1,H,W
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
        effective_num_frames_total_sequence = max(1, self.num_frames_total_sequence)
        required_span_on_disk = (effective_num_frames_total_sequence - 1) * self.frame_skip + 1
        
        if self.num_disk_frames >= required_span_on_disk:
            for i in range(self.num_disk_frames - required_span_on_disk + 1): all_samples_start_indices.append(i)
        else: self.logger.warning(f"Not enough frames ({self.num_disk_frames}) in '{self.video_path}' for sequence (needs {required_span_on_disk} for seq_len {effective_num_frames_total_sequence}). Dataset for mode '{self.mode}' will be empty."); self.samples_start_indices = []
        
        if not all_samples_start_indices: 
            self.logger.error(f"No valid samples from '{self.video_path}' for any split (effective_seq_len={effective_num_frames_total_sequence}, disk_frames={self.num_disk_frames})."); self.samples_start_indices = []
        else:
            rng_dataset = random.Random(seed); rng_dataset.shuffle(all_samples_start_indices)
            if val_fraction > 0.0 and val_fraction < 1.0:
                num_total_samples = len(all_samples_start_indices); num_val_samples = int(num_total_samples * val_fraction)
                num_val_samples = max(0, min(num_val_samples, num_total_samples - (1 if num_total_samples > num_val_samples else 0) ))
                if self.mode == 'train':
                    train_indices = all_samples_start_indices[num_val_samples:]
                    if data_fraction < 1.0 and len(train_indices) > 1: num_train_to_keep = max(1, int(len(train_indices) * data_fraction)); self.samples_start_indices = train_indices[:num_train_to_keep]; self.logger.info(f"Using {data_fraction*100:.2f}% of available training samples: {len(self.samples_start_indices)} samples.")
                    else: self.samples_start_indices = train_indices
                elif self.mode == 'val': self.samples_start_indices = all_samples_start_indices[:num_val_samples]
                else: raise ValueError(f"Invalid mode '{self.mode}'. Must be 'train' or 'val'.")
            else: 
                self.samples_start_indices = all_samples_start_indices
                if self.mode == 'train' and data_fraction < 1.0 and len(self.samples_start_indices) > 1: 
                    num_to_keep = max(1, int(len(self.samples_start_indices) * data_fraction))
                    self.samples_start_indices = self.samples_start_indices[:num_to_keep]
                    self.logger.info(f"Using {data_fraction*100:.2f}% of available samples (no val split): {len(self.samples_start_indices)} samples.")
                elif self.mode == 'val':  self.samples_start_indices = [] if val_fraction == 0.0 else all_samples_start_indices
        if not self.samples_start_indices and self.mode == 'train': self.logger.error(f"VideoFrameDataset (TRAIN): No valid samples from '{self.video_path}'. Training might fail.")
        elif not self.samples_start_indices and self.mode == 'val' and val_fraction > 0.0 : self.logger.warning(f"VideoFrameDataset (VAL): No valid samples from '{self.video_path}' for validation split. Validation will be skipped.")
        self.logger.info(f"VideoFrameDatasetCPU ({self.mode.upper()}) Init. DiskFrames:{self.num_disk_frames}. NumSamples:{len(self.samples_start_indices)}. SeqLen:{effective_num_frames_total_sequence} (skip {self.frame_skip}).")
    def __len__(self) -> int: return len(self.samples_start_indices)
    def __getitem__(self, idx: int) -> torch.Tensor: 
        t_start_getitem = time.time()
        effective_num_frames_total_sequence = max(1, self.num_frames_total_sequence) 
        if not self.samples_start_indices or idx >= len(self.samples_start_indices):
            self.logger.error(f"Index {idx} out of bounds for {self.mode} dataset with {len(self.samples_start_indices)} samples. Returning zeros.")
            return torch.zeros((effective_num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        start_frame_idx_in_ram = self.samples_start_indices[idx]; frames_for_sample = []
        for i in range(effective_num_frames_total_sequence):
            actual_frame_idx_in_ram = start_frame_idx_in_ram + i * self.frame_skip
            if actual_frame_idx_in_ram < self.num_disk_frames:
                try:
                    frame_tensor_chw_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram]
                    if frame_tensor_chw_uint8.shape[0] == 1: frame_tensor_chw_uint8 = frame_tensor_chw_uint8.repeat(3,1,1)
                    elif frame_tensor_chw_uint8.shape[0] != 3 : 
                        self.logger.warning(f"Frame {actual_frame_idx_in_ram} has {frame_tensor_chw_uint8.shape[0]} channels. Adjusting.")
                        if frame_tensor_chw_uint8.shape[0] > 3: frame_tensor_chw_uint8 = frame_tensor_chw_uint8[:3,...]
                        else: frame_tensor_chw_uint8 = frame_tensor_chw_uint8[0:1,...].repeat(3,1,1)
                    if frame_tensor_chw_uint8.ndim != 3 or frame_tensor_chw_uint8.shape[0] !=3 :
                        self.logger.error(f"Frame {actual_frame_idx_in_ram} has unexpected shape {frame_tensor_chw_uint8.shape} after channel adjustment. Padding zeros.")
                        frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)); continue
                    transformed_frame = self.normalize_transform(self.resize_transform(frame_tensor_chw_uint8).float()/255.0)
                    frames_for_sample.append(transformed_frame)
                except Exception as e:
                    self.logger.error(f"Error transforming frame {actual_frame_idx_in_ram} (sample {idx}): {e}", exc_info=True)
                    frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
            else:
                self.logger.error(f"Frame index {actual_frame_idx_in_ram} out of bounds for disk_frames {self.num_disk_frames} (sample {idx}). Padding zeros.")
                frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
        if not frames_for_sample: 
            self.logger.error(f"No frames collected for sample {idx}. Returning zeros.")
            return torch.zeros((effective_num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        result_frames = torch.stack(frames_for_sample)
        if result_frames.shape[0] != effective_num_frames_total_sequence: 
            self.logger.error(f"Loaded {result_frames.shape[0]} frames, expected {effective_num_frames_total_sequence} for sample {idx}. Final shape: {result_frames.shape}.")
            return torch.zeros((effective_num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        t_end_getitem = time.time(); duration = t_end_getitem - t_start_getitem
        if self.args_ref: 
            pid = os.getpid();
            if pid not in self.pid_counters: self.pid_counters[pid] = 0
            self.pid_counters[pid] += 1; current_worker_call_count = self.pid_counters[pid]; log_this_call = False; log_reason = ""
            if self.getitem_slow_threshold > 0 and duration > self.getitem_slow_threshold: log_this_call = True; log_reason = f"SLOW ({duration:.4f}s > {self.getitem_slow_threshold:.2f}s)"
            elif self.getitem_log_interval > 0 and current_worker_call_count % self.getitem_log_interval == 0: log_this_call = True; log_reason = f"periodic (call #{current_worker_call_count})"
            if log_this_call: self.logger.debug(f"__getitem__ (idx {idx}, worker {pid}, {log_reason}) from '{Path(self.video_path).name}' took {duration:.4f}s.")
        return result_frames

# --- New Grid BBox Utility ---
def create_grid_bboxes(image_wh: Tuple[int, int], 
                       region_wh_on_image: Tuple[int, int], 
                       dtype=torch.float, 
                       device: Optional[torch.device]=None) -> Tuple[torch.Tensor, int, int]:
    img_w, img_h = image_wh
    region_w, region_h = region_wh_on_image

    if region_w <= 0 or region_h <= 0:
        logger_wubu_diffusion.error(f"Region dimensions must be positive. Got w={region_w}, h={region_h}. Defaulting to full image.")
        return torch.tensor([[0.0, 0.0, float(img_w), float(img_h)]], dtype=dtype, device=device), 1, 1
    
    num_regions_w = img_w // region_w
    num_regions_h = img_h // region_h
    
    if num_regions_w == 0 or num_regions_h == 0:
        logger_wubu_diffusion.warning(
            f"Image dimensions ({img_w}x{img_h}) are smaller than region dimensions ({region_w}x{region_h}) "
            f"in at least one axis. Defaulting to a single region covering the image."
        )
        return torch.tensor([[0.0, 0.0, float(img_w), float(img_h)]], dtype=dtype, device=device), 1, 1

    if img_w % region_w != 0 or img_h % region_h != 0:
        logger_wubu_diffusion.warning(
            f"Image dimensions ({img_w}x{img_h}) are not perfectly divisible by region dimensions ({region_w}x{region_h}). "
            f"The grid will cover {num_regions_w*region_w}x{num_regions_h*region_h}. Some pixels at edges might be missed."
        )
            
    bboxes = []
    for r_idx in range(num_regions_h):
        for c_idx in range(num_regions_w):
            x1 = c_idx * region_w
            y1 = r_idx * region_h
            x2 = x1 + region_w
            y2 = y1 + region_h
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            
    if not bboxes:
        logger_wubu_diffusion.error("No bboxes generated by create_grid_bboxes. This should not happen if num_regions_w/h > 0.")
        return torch.tensor([[0.0, 0.0, float(img_w), float(img_h)]], dtype=dtype, device=device), 1, 1

    return torch.tensor(bboxes, dtype=dtype, device=device), num_regions_w, num_regions_h

# --- Diffusion Model Components (Unchanged DiffusionProcess, SinusoidalPosEmb) ---
def _extract_into_tensor(arr_or_list, timesteps, broadcast_shape):
    if isinstance(arr_or_list, list): res = torch.tensor(arr_or_list, device=timesteps.device)[timesteps].float()
    else: res = arr_or_list.to(timesteps.device)[timesteps].float() 
    while len(res.shape) < len(broadcast_shape): res = res[..., None]
    return res.expand(broadcast_shape)

class DiffusionProcess:
    def __init__(self, timesteps: int, beta_schedule: str = 'linear', beta_start: float = 0.0001, beta_end: float = 0.02, device: torch.device = torch.device("cpu")):
        self.timesteps = timesteps; self.beta_schedule = beta_schedule; self.device = device
        if beta_schedule == "linear": self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        elif beta_schedule == "cosine":
            s = 0.008; steps = timesteps + 1; x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]; betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else: raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        self.alphas = 1.0 - self.betas; self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) 
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod); self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
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
    def p_mean_variance(self, model_output_eps: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True, clip_range: Tuple[float,float] = (-1.0, 1.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Added clip_range for latent space if needed
        pred_xstart = self._predict_xstart_from_eps(x_t, t, eps=model_output_eps)
        if clip_denoised: pred_xstart = torch.clamp(pred_xstart, clip_range[0], clip_range[1]) # Generalized clip
        posterior_mean_coef1_t = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * pred_xstart + posterior_mean_coef2_t * x_t
        posterior_variance_t = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t
    def p_sample(self, model_output_eps: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True, clip_range: Tuple[float,float] = (-1.0, 1.0)) -> torch.Tensor:
        posterior_mean, _, posterior_log_variance_clipped_t = self.p_mean_variance(model_output_eps, x_t, t, clip_denoised, clip_range)
        noise = torch.randn_like(x_t); nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return posterior_mean + nonzero_mask * (0.5 * posterior_log_variance_clipped_t).exp() * noise
    @torch.no_grad()
    def sample(self, model_callable: callable, batch_size: int, latent_feature_dim: int,
               num_sampling_steps: Optional[int] = None,
               show_progress: bool = True, progress_desc: str = "Diffusion Sampling (Latent)") -> torch.Tensor:
        # Samples in the latent feature space
        x_t_latent_flat = torch.randn(batch_size, latent_feature_dim, device=self.device)
        
        actual_timesteps_to_iterate = self.timesteps
        if num_sampling_steps is not None and num_sampling_steps > 0 and num_sampling_steps < self.timesteps:
            time_seq = torch.linspace(self.timesteps - 1, 0, num_sampling_steps, device=self.device).long()
            time_seq = torch.unique(time_seq, sorted=True).flip(dims=[0]); iterable_range = time_seq
            actual_timesteps_to_iterate = len(time_seq)
        else: iterable_range = reversed(range(0, self.timesteps)); actual_timesteps_to_iterate = self.timesteps
        
        iterable_progress = iterable_range
        if show_progress: iterable_progress = tqdm(iterable_range, desc=progress_desc, total=actual_timesteps_to_iterate, leave=False, dynamic_ncols=True, disable=not show_progress)
        
        for i_val_t in iterable_progress:
            current_timestep_val = i_val_t.item() if torch.is_tensor(i_val_t) else i_val_t
            t = torch.full((batch_size,), current_timestep_val, device=self.device, dtype=torch.long)
            predicted_noise_latent_flat = model_callable(x_t_latent_flat, t) # model_callable now predicts noise in latent space
            # Clip range for latent features might be different or not needed if activations are unbounded (e.g. no Tanh)
            # For now, assume latent features don't need aggressive clipping like images.
            x_t_prev_latent_flat = self.p_sample(predicted_noise_latent_flat, x_t_latent_flat, t, clip_denoised=False) 
            x_t_latent_flat = x_t_prev_latent_flat
            
        return x_t_latent_flat # Returns denoised latent features

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device; half_dim = self.dim // 2
        if half_dim <=0 : return torch.zeros_like(time.unsqueeze(-1).float()) if self.dim == 1 else torch.empty(time.shape[0], 0, device=device)
        embeddings = math.log(10000) / (half_dim - 1); embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]; embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1 and self.dim > 1: embeddings = F.pad(embeddings, (0,1))
        return embeddings

# --- New Image Encoder/Decoder Modules ---
class ImageInputEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.img_channels = args.num_channels
        self.grid_region_w_on_image = args.img_encoder_grid_region_w
        self.grid_region_h_on_image = args.img_encoder_grid_region_h
        self.patch_embedding_dim = args.img_encoder_patch_embedding_dim
        
        self.patch_input_flat_dim = self.img_channels * self.grid_region_w_on_image * self.grid_region_h_on_image
        
        encoder_hidden_dim = max(self.patch_embedding_dim, int(self.patch_input_flat_dim * args.img_encoder_mlp_hidden_factor))

        self.patch_encoder_mlp = nn.Sequential(
            nn.Linear(self.patch_input_flat_dim, encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(encoder_hidden_dim, self.patch_embedding_dim)
        )
        self.apply(init_weights_general)
        logger_wubu_diffusion.info(f"ImageInputEncoder: Grid Region {self.grid_region_w_on_image}x{self.grid_region_h_on_image} (pixels from img) -> {self.patch_input_flat_dim}D flat -> MLP -> {self.patch_embedding_dim}D embed.")

    def forward(self, image_batch: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]: # Returns: flat_encoded_features, N_actual_regions, num_regions_w, num_regions_h
        B, C, H_img, W_img = image_batch.shape
        
        bboxes_on_image, num_r_w, num_r_h = create_grid_bboxes(
            (W_img, H_img), 
            (self.grid_region_w_on_image, self.grid_region_h_on_image),
            dtype=image_batch.dtype, device=image_batch.device
        )
        N_actual_regions = bboxes_on_image.shape[0]
        
        all_batch_patch_embeddings_list = []
        for b_idx in range(B):
            img_item = image_batch[b_idx] # C, H_img, W_img
            item_patch_embeddings_list = []
            for r_idx in range(N_actual_regions):
                x1, y1, x2, y2 = bboxes_on_image[r_idx].round().int().tolist()
                
                # Extract pixel region based on bbox coordinates.
                # create_grid_bboxes ensures x2-x1 = region_w and y2-y1 = region_h.
                # So, the extracted pixel_region will have the correct dimensions.
                pixel_region = img_item[:, y1:y2, x1:x2] 
                
                # Sanity check for extracted region size (should not fail if create_grid_bboxes is correct)
                if pixel_region.shape[1] != self.grid_region_h_on_image or \
                   pixel_region.shape[2] != self.grid_region_w_on_image:
                    logger_wubu_diffusion.error(
                        f"Encoder: FATAL - Region {r_idx} extracted with unexpected size. "
                        f"Expected C x {self.grid_region_h_on_image} x {self.grid_region_w_on_image}, "
                        f"got {pixel_region.shape}. Bbox was [{x1},{y1},{x2},{y2}]. "
                        f"This indicates a bug in create_grid_bboxes or coordinate handling."
                    )
                    # As a very basic fallback to prevent crashes, create a zero patch, but this is an error state.
                    pixel_region = torch.zeros((C, self.grid_region_h_on_image, self.grid_region_w_on_image), 
                                               dtype=img_item.dtype, device=img_item.device)


                flattened_region = pixel_region.reshape(1, -1) # 1, C*grid_h*grid_w
                if flattened_region.shape[1] != self.patch_input_flat_dim:
                     # This error implies self.patch_input_flat_dim was miscalculated relative to C, H_region, W_region
                     raise ValueError(
                         f"Encoder: Flattened region dim mismatch. Expected {self.patch_input_flat_dim}, "
                         f"got {flattened_region.shape[1]} (from C={C}, H_patch={self.grid_region_h_on_image}, W_patch={self.grid_region_w_on_image}). "
                         f"Check calculation of self.patch_input_flat_dim."
                     )

                embedded_patch = self.patch_encoder_mlp(flattened_region) # 1, patch_embedding_dim
                item_patch_embeddings_list.append(embedded_patch)
            
            if item_patch_embeddings_list:
                all_batch_patch_embeddings_list.append(torch.cat(item_patch_embeddings_list, dim=0)) # N_actual_regions, patch_embedding_dim
            elif N_actual_regions > 0 : # If N_actual_regions > 0 but list is empty, something went wrong
                logger_wubu_diffusion.error(f"Encoder: No patch embeddings collected for batch item {b_idx} despite N_actual_regions={N_actual_regions}. Filling with zeros.")
                dummy_embeddings = torch.zeros((N_actual_regions, self.patch_embedding_dim), device=image_batch.device, dtype=image_batch.dtype)
                all_batch_patch_embeddings_list.append(dummy_embeddings)
            # If N_actual_regions is 0, list will be empty, and stack will be on empty list (handled by PyTorch) or needs guard.
            # create_grid_bboxes should always return at least one bbox.

        if not all_batch_patch_embeddings_list and N_actual_regions > 0 : # Should not happen if N_actual_regions > 0
             logger_wubu_diffusion.error(f"Encoder: all_batch_patch_embeddings_list is empty but N_actual_regions={N_actual_regions}. This is unexpected.")
             # Fallback: create a batch of zero embeddings if something went severely wrong
             encoded_features_per_image = torch.zeros((B, N_actual_regions, self.patch_embedding_dim), device=image_batch.device, dtype=image_batch.dtype)
        elif not all_batch_patch_embeddings_list and N_actual_regions == 0:
            # This case implies create_grid_bboxes returned 0 regions, which it's designed not to do.
            # If it did, the latent_feature_dim would be 0, causing issues later.
            logger_wubu_diffusion.error(f"Encoder: N_actual_regions is 0, cannot create features.")
            # Return empty tensor or raise error, as subsequent layers expect non-zero features if N_actual_regions > 0
            # Given create_grid_bboxes always returns at least one region, this path should not be hit.
            # If it were possible for N_actual_regions to be 0, then latent_feature_dim_one_frame would also be 0 and error.
            # For safety, if N_actual_regions somehow became 0 and didn't error earlier:
            if N_actual_regions == 0: N_actual_regions = 1 # Ensure at least 1 for shape, though this is bad
            encoded_features_per_image = torch.zeros((B, N_actual_regions, self.patch_embedding_dim), device=image_batch.device, dtype=image_batch.dtype)
        else:
            encoded_features_per_image = torch.stack(all_batch_patch_embeddings_list) # B, N_actual_regions, patch_embedding_dim

        flat_encoded_features = encoded_features_per_image.reshape(B, -1) # B, N_actual_regions * patch_embedding_dim
        return flat_encoded_features, N_actual_regions, num_r_w, num_r_h

class ImageOutputDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace, actual_num_regions: int, num_regions_wh: Tuple[int,int]):
        super().__init__()
        self.args = args
        self.actual_num_regions = actual_num_regions
        if self.actual_num_regions <= 0:
            # This would be a critical issue from upstream (e.g. create_grid_bboxes or model init)
            logger_wubu_diffusion.error(f"ImageOutputDecoder initialized with actual_num_regions <= 0 ({self.actual_num_regions}). This is invalid.")
            # To prevent immediate crash, set to 1, but this indicates a deeper problem.
            self.actual_num_regions = 1 
            
        self.patch_embedding_dim = args.img_encoder_patch_embedding_dim
        self.img_channels = args.num_channels
        self.grid_region_w_on_image = args.img_encoder_grid_region_w
        self.grid_region_h_on_image = args.img_encoder_grid_region_h
        self.target_image_w, self.target_image_h = args.image_w, args.image_h
        self.num_regions_w, self.num_regions_h = num_regions_wh # From encoder

        patch_output_pixels_flat = self.img_channels * self.grid_region_w_on_image * self.grid_region_h_on_image
        decoder_hidden_dim = max(self.patch_embedding_dim, int(patch_output_pixels_flat * args.img_encoder_mlp_hidden_factor))

        self.patch_decoder_mlp = nn.Sequential(
            nn.Linear(self.patch_embedding_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, patch_output_pixels_flat),
            nn.Tanh() # To ensure output pixels are in [-1, 1]
        )
        self.apply(init_weights_general)
        logger_wubu_diffusion.info(f"ImageOutputDecoder: {self.patch_embedding_dim}D embed -> MLP -> {patch_output_pixels_flat}D flat ({self.grid_region_w_on_image}x{self.grid_region_h_on_image} pixels). Assembling {self.actual_num_regions} regions for {self.target_image_w}x{self.target_image_h} image.")

    def forward(self, flat_denoised_features: torch.Tensor) -> torch.Tensor:
        B = flat_denoised_features.shape[0]
        expected_dim = self.actual_num_regions * self.patch_embedding_dim
        if flat_denoised_features.shape[1] != expected_dim:
            # If actual_num_regions was forced to 1 due to init error, this might mismatch.
            raise ValueError(
                f"Decoder: Input feature dim mismatch. Expected {expected_dim} (actual_num_regions={self.actual_num_regions} * patch_emb_dim={self.patch_embedding_dim}), "
                f"got {flat_denoised_features.shape[1]}"
            )

        per_patch_embeddings = flat_denoised_features.reshape(B, self.actual_num_regions, self.patch_embedding_dim)
        reconstructed_pixel_patches_flat = self.patch_decoder_mlp(per_patch_embeddings) 
        reconstructed_pixel_patches = reconstructed_pixel_patches_flat.reshape(
            B, self.actual_num_regions, self.img_channels, self.grid_region_h_on_image, self.grid_region_w_on_image
        )
        
        # The reconstructed image dimensions are determined by num_regions_w/h and grid_region_w/h,
        # which should align with target_image_w/h if target_image_w/h are divisible by grid_region_w/h.
        # If not perfectly divisible, output_images will be sized to target_image_w/h, and patches
        # will fill up to num_regions_w * grid_region_w_on_image, potentially leaving trailing pixels blank.
        output_images = torch.zeros(B, self.img_channels, self.target_image_h, self.target_image_w, 
                                    device=flat_denoised_features.device, dtype=flat_denoised_features.dtype)
        
        patch_idx_counter = 0
        for r_grid in range(self.num_regions_h):
            for c_grid in range(self.num_regions_w):
                if patch_idx_counter < self.actual_num_regions:
                    patch_to_place = reconstructed_pixel_patches[:, patch_idx_counter, :, :, :]
                    y_start = r_grid * self.grid_region_h_on_image
                    x_start = c_grid * self.grid_region_w_on_image
                    y_end = y_start + self.grid_region_h_on_image
                    x_end = x_start + self.grid_region_w_on_image
                    
                    # Place the patch. Since create_grid_bboxes uses floor division for num_regions,
                    # y_end will be <= target_image_h and x_end <= target_image_w.
                    # No explicit clipping needed here for placement boundaries.
                    output_images[:, :, y_start:y_end, x_start:x_end] = patch_to_place
                    patch_idx_counter += 1
                else: # Should not be reached if actual_num_regions = num_regions_w * num_regions_h
                    logger_wubu_diffusion.warning(f"Decoder: patch_idx_counter ({patch_idx_counter}) exceeded actual_num_regions ({self.actual_num_regions}) during assembly. This is unexpected.")
                    break 
            if patch_idx_counter >= self.actual_num_regions and r_grid < self.num_regions_h -1 : # Optimization: if all patches placed, break outer loop too
                 if patch_idx_counter > self.actual_num_regions: # This is a stronger error
                     logger_wubu_diffusion.error(f"Decoder: patch_idx_counter ({patch_idx_counter}) GREATER THAN actual_num_regions ({self.actual_num_regions}). Logic error.")
                 break
        
        if patch_idx_counter != self.actual_num_regions:
             logger_wubu_diffusion.warning(
                 f"Decoder: Mismatch between patches processed ({patch_idx_counter}) and actual_num_regions ({self.actual_num_regions}). "
                 f"Grid was {self.num_regions_w}x{self.num_regions_h}. Some regions might not have been filled or logic error."
             )
        return output_images
        
class WuBuDiffusionModel(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict): # Removed gaad_config
        super().__init__()
        self.args = args
        self.num_channels = video_config['num_channels'] # Retained for decoder

        # Instantiate Encoder and Decoder
        self.input_encoder = ImageInputEncoder(args)
        
        # Determine actual_num_regions and latent_feature_dim for FDQWR
        # This requires a dummy forward pass or pre-calculation.
        # For simplicity, pre-calculate based on args.
        # Note: This assumes image_w and image_h from args are the ones used.
        
        # Corrected unpacking and calculation of actual_num_regions
        bboxes_from_grid_calc, self.num_r_w, self.num_r_h = create_grid_bboxes(
            (args.image_w, args.image_h),
            (args.img_encoder_grid_region_w, args.img_encoder_grid_region_h)
        )
        self.actual_num_regions = bboxes_from_grid_calc.shape[0]
        
        logger_wubu_diffusion.info(f"Model Init: Grid {self.num_r_w}x{self.num_r_h} -> {self.actual_num_regions} actual regions.")

        self.latent_feature_dim_one_frame = self.actual_num_regions * args.img_encoder_patch_embedding_dim
        if self.latent_feature_dim_one_frame <= 0:
            raise ValueError(f"Calculated latent_feature_dim_one_frame is {self.latent_feature_dim_one_frame}. Must be > 0.")

        self.output_decoder = ImageOutputDecoder(args, self.actual_num_regions, (self.num_r_w, self.num_r_h))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(args.diffusion_time_embed_dim),
            nn.Linear(args.diffusion_time_embed_dim, args.diffusion_time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(args.diffusion_time_embed_dim * 4, args.diffusion_time_embed_dim),
        ) if args.diffusion_time_embed_dim > 0 else None

        wubu_input_dim_for_stack = self.latent_feature_dim_one_frame
        if self.time_mlp is not None and args.diffusion_time_embed_dim > 0: 
            wubu_input_dim_for_stack += args.diffusion_time_embed_dim
        wubu_input_dim_for_stack = max(1, wubu_input_dim_for_stack)
        
        # FDQWR now predicts noise in the latent space or the latent features themselves
        wubu_output_dim_from_stack_latent = self.latent_feature_dim_one_frame

        self.wubu_noise_predictor_stack = FractalDepthQuaternionWuBuRungs(
            input_dim=wubu_input_dim_for_stack,
            output_dim=args.bsp_intermediate_dim if args.use_bsp_gate else wubu_output_dim_from_stack_latent,
            num_virtual_micro_levels=args.model_wubu_num_virtual_levels,
            base_micro_level_dim=args.model_wubu_base_micro_dim,
            num_physical_transform_stages=args.model_wubu_num_physical_stages,
            initial_s=args.wubu_initial_s, s_decay_factor_per_micro_level=args.wubu_s_decay,
            initial_c_base=args.wubu_initial_c, c_phi_influence=args.wubu_c_phi_influence,
            num_phi_scaffold_points_per_stage=args.wubu_num_phi_scaffold_points,
            phi_scaffold_init_scale_factor=args.wubu_phi_scaffold_init_scale,
            use_gaussian_rungs= (not args.use_manifold_uncrumple_transform) and args.wubu_use_gaussian_rungs, 
            base_gaussian_std_dev_factor_rung=args.wubu_base_gaussian_std_dev,
            gaussian_std_dev_decay_factor_rung=args.wubu_gaussian_std_dev_decay,
            rung_affinity_temperature=args.wubu_rung_affinity_temp, 
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
            use_residual_in_micro=args.wubu_use_residual_in_micro, 
            learnable_affinity_params_per_stage=args.wubu_learnable_affinity_params, 
            use_manifold_uncrumple_transform=args.use_manifold_uncrumple_transform,
            uncrumple_hypernet_hidden_factor=args.uncrumple_hypernet_hidden_factor,
            uncrumple_learn_rotation=args.uncrumple_learn_rotation,
            uncrumple_learn_scale=args.uncrumple_learn_scale,
            uncrumple_scale_activation=args.uncrumple_scale_activation,
            uncrumple_scale_min=args.uncrumple_scale_min,
            uncrumple_scale_max=args.uncrumple_scale_max,
            uncrumple_learn_translation=args.uncrumple_learn_translation,
            wubu_cond_embed_dim=args.wubu_cond_embed_dim,
            wubu_condition_on_s=args.wubu_condition_on_s,
            wubu_condition_on_c=args.wubu_condition_on_c,
            wubu_condition_on_micro_idx=args.wubu_condition_on_micro_idx,
            wubu_condition_on_stage_idx=args.wubu_condition_on_stage_idx,
            wubu_max_micro_idx_embed=args.wubu_max_micro_idx_embed,
            wubu_max_stage_idx_embed=args.wubu_max_stage_idx_embed,
            wubu_inter_micro_norm_interval=args.wubu_inter_micro_norm_interval
        )

        self.use_bsp_gate = args.use_bsp_gate
        if self.use_bsp_gate:
            logger_wubu_diffusion.info("BSP Gate is ENABLED.")
            bsp_gate_input_dim = self.latent_feature_dim_one_frame # BSP Gate operates on encoded features
            bsp_gate_hidden_dim = max(args.bsp_gate_hidden_dim, 1)
            if bsp_gate_input_dim <= 0: bsp_gate_input_dim = 1; logger_wubu_diffusion.warning(f"BSP Gate input dim is <=0. Setting to 1.")
            self.bsp_gate_mlp = nn.Sequential(
                nn.Linear(bsp_gate_input_dim, bsp_gate_hidden_dim), nn.GELU(),
                nn.Linear(bsp_gate_hidden_dim, args.bsp_num_paths),
            )
            self.bsp_output_heads = nn.ModuleList()
            for _ in range(args.bsp_num_paths): self.bsp_output_heads.append(nn.Linear(args.bsp_intermediate_dim, wubu_output_dim_from_stack_latent))
            logger_wubu_diffusion.info(f"BSP Gate: Input {bsp_gate_input_dim}D (encoded features) -> MLP -> {args.bsp_num_paths} path logits.")
            logger_wubu_diffusion.info(f"BSP Output Heads: {args.bsp_num_paths} heads, each {args.bsp_intermediate_dim}D -> {wubu_output_dim_from_stack_latent}D (latent noise/features).")
        else:
            logger_wubu_diffusion.info("BSP Gate is DISABLED.")
            if hasattr(args, 'bsp_intermediate_dim') and args.bsp_intermediate_dim != wubu_output_dim_from_stack_latent :
                 logger_wubu_diffusion.debug(f"BSP not used, bsp_intermediate_dim ({args.bsp_intermediate_dim}) differs from wubu_output_dim_from_stack_latent ({wubu_output_dim_from_stack_latent}). FDQWR output_dim should be {wubu_output_dim_from_stack_latent}.")

        self.apply(init_weights_general)
        logger_wubu_diffusion.info(f"WuBuDiffusionModel: Latent Feature Dim per Frame: {self.latent_feature_dim_one_frame}D.")
        logger_wubu_diffusion.info(f"  FDQWR Input Dim (latent + time_emb): {wubu_input_dim_for_stack}D.")
        if self.use_bsp_gate: logger_wubu_diffusion.info(f"  FDQWR Output (Intermediate for BSP): {args.bsp_intermediate_dim}D.")
        else: logger_wubu_diffusion.info(f"  FDQWR Output (Final Latent Noise/Features): {wubu_output_dim_from_stack_latent}D.")

    def predict_noise_from_latent(self, x_t_latent_features: torch.Tensor, time_t: torch.Tensor, current_global_step: Optional[int] = None) -> torch.Tensor:
        """ Predicts noise in the latent feature space given noisy latent features and time."""
        wubu_stack_input = x_t_latent_features
        if self.time_mlp and self.args.diffusion_time_embed_dim > 0:
            time_emb = self.time_mlp(time_t)
            if x_t_latent_features.shape[0] != time_emb.shape[0]: # Batch alignment
                if time_emb.shape[0] == 1 and x_t_latent_features.shape[0] > 1: time_emb = time_emb.expand(x_t_latent_features.shape[0], -1)
                elif x_t_latent_features.shape[0] == 1 and time_emb.shape[0] > 1: time_emb = time_emb[0:1, :].expand(x_t_latent_features.shape[0], -1)
                else: raise ValueError(f"Batch size mismatch x_t_latent ({x_t_latent_features.shape[0]}) vs time_emb ({time_emb.shape[0]})")
            wubu_stack_input = torch.cat((x_t_latent_features, time_emb), dim=-1)

        intermediate_or_final_latent_noise = self.wubu_noise_predictor_stack(
            wubu_stack_input,
            show_progress=getattr(self.args, 'show_train_progress_bar', False) and \
                          (current_global_step is None or \
                           current_global_step % getattr(self.args, 'train_progress_update_interval_steps', self.args.log_interval) == 0)
        )

        if self.use_bsp_gate:
            # Gate input is the original x_t_latent_features (before time_emb typically, or a separate feature if desired)
            # For simplicity here, using x_t_latent_features as gate input.
            gate_input_for_bsp = x_t_latent_features 
            if gate_input_for_bsp.shape[1] == 0 and isinstance(self.bsp_gate_mlp[0], nn.Linear) and self.bsp_gate_mlp[0].in_features == 1:
                gate_input_for_bsp = torch.ones(x_t_latent_features.shape[0], 1, device=x_t_latent_features.device, dtype=x_t_latent_features.dtype)
            
            path_logits = self.bsp_gate_mlp(gate_input_for_bsp)
            if self.args.bsp_num_paths == 1: path_weights = torch.ones_like(path_logits)
            elif self.args.bsp_num_paths == 2 and self.args.bsp_gate_activation == 'sigmoid': path_weights_p1 = torch.sigmoid(path_logits[:,0:1]); path_weights_p0 = 1.0 - path_weights_p1; path_weights = torch.cat([path_weights_p0, path_weights_p1], dim=-1)
            else: path_weights = F.softmax(path_logits, dim=-1)
            
            outputs_from_heads = torch.stack([head(intermediate_or_final_latent_noise) for head in self.bsp_output_heads], dim=0) 
            weights_reshaped = path_weights.permute(1,0).unsqueeze(-1) 
            predicted_noise_for_latent = torch.sum(weights_reshaped * outputs_from_heads, dim=0) 

            if self.args.wandb and WANDB_AVAILABLE and wandb.run and \
               current_global_step is not None and \
               current_global_step % (self.args.log_interval * getattr(self.args, 'bsp_log_gate_interval_mult', 5)) == 0:
                 for p_idx in range(min(self.args.bsp_num_paths, 16)): wandb.log({f"debug/bsp_gate_avg_weight_path_{p_idx}": path_weights[:,p_idx].mean().item()}, step=current_global_step)
                 if self.args.bsp_num_paths > 16: wandb.log({"debug/bsp_gate_avg_weight_path_last": path_weights[:,-1].mean().item()}, step=current_global_step)
        else:
            predicted_noise_for_latent = intermediate_or_final_latent_noise
        return predicted_noise_for_latent

    def forward(self, x0_image_batch: torch.Tensor, time_t: torch.Tensor, diffusion_process: DiffusionProcess, current_global_step: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x0_image_batch is (B, C, H_img, W_img)
        # Encodes image to latent, adds noise, predicts noise in latent space.
        
        encoded_features_x0, _, _, _ = self.input_encoder(x0_image_batch) # Returns B, latent_dim
        target_noise_for_latent = torch.randn_like(encoded_features_x0)
        x_t_latent_features = diffusion_process.q_sample(encoded_features_x0, time_t, noise=target_noise_for_latent)
        
        predicted_noise_in_latent = self.predict_noise_from_latent(x_t_latent_features, time_t, current_global_step)
        
        return predicted_noise_in_latent, target_noise_for_latent


# --- Trainer ---
class CPUDiffusionTrainer:
    def __init__(self, model: WuBuDiffusionModel, diffusion_process: DiffusionProcess, args: argparse.Namespace):
        self.model = model; self.diffusion_process = diffusion_process; self.args = args
        self.device = torch.device("cpu"); self.logger = logger_wubu_diffusion.getChild("CPUDiffusionTrainer")
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        self.global_step = 0; self.current_epoch = 0
        
        if os.path.exists(args.video_data_path) and os.path.isdir(args.video_data_path):
             video_files = [f for f in os.listdir(args.video_data_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
             if not video_files:
                 self.logger.error(f"No video files found in directory: {args.video_data_path}")
                 if IMAGEIO_AVAILABLE and imageio is not None:
                     dummy_video_path = Path(args.video_data_path) / "dummy_cpu_video_diffusion.mp4"
                     self.logger.info(f"Creating dummy video at {dummy_video_path}")
                     dataset_sequence_len_for_dummy = args.num_input_frames + args.num_predict_frames
                     min_req_disk_frames_for_sequence = (dataset_sequence_len_for_dummy -1) * max(1, args.frame_skip) + 1
                     min_samples_for_one_train_batch = args.batch_size if args.drop_last_train else 1
                     total_samples_before_split_needed = math.ceil(min_samples_for_one_train_batch / (1.0 - args.val_fraction)) if args.val_fraction > 0 and args.val_fraction < 1 else min_samples_for_one_train_batch
                     num_dummy_frames_on_disk = max(total_samples_before_split_needed + min_req_disk_frames_for_sequence -1 + 50, min_req_disk_frames_for_sequence + 50)
                     try:
                         Path(args.video_data_path).mkdir(parents=True, exist_ok=True)
                         with imageio.get_writer(str(dummy_video_path), fps=15, codec='libx264', quality=8, ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
                             for _ in range(int(num_dummy_frames_on_disk)): writer.append_data(np.random.randint(0,255, (args.image_h, args.image_w, args.num_channels), dtype=np.uint8))
                         self.args.video_file_path = str(dummy_video_path); self.logger.info(f"Dummy video created with {int(num_dummy_frames_on_disk)} frames: {self.args.video_file_path}")
                     except Exception as e_write: self.logger.error(f"Failed to write dummy video: {e_write}. Check ffmpeg/permissions."); raise FileNotFoundError("Failed to create dummy video, and no videos found.")
                 else: raise FileNotFoundError(f"No video files in {args.video_data_path} and imageio not available to create dummy.")
             else: self.args.video_file_path = os.path.join(args.video_data_path, video_files[0]); self.logger.info(f"Using first video found: {self.args.video_file_path}")
        elif os.path.exists(args.video_data_path) and os.path.isfile(args.video_data_path): self.args.video_file_path = args.video_data_path
        else: raise FileNotFoundError(f"Video data path not found or invalid: {args.video_data_path}")
        
        dataset_sequence_len = args.num_input_frames + args.num_predict_frames
        self.train_dataset = VideoFrameDatasetCPU(video_path=self.args.video_file_path, num_frames_total=dataset_sequence_len, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip, data_fraction=args.data_fraction, val_fraction=args.val_fraction, mode='train', seed=args.seed, args=self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=args.drop_last_train)
        
        self.val_loader = None
        if args.val_fraction > 0.0 and args.val_fraction < 1.0:
            self.val_dataset = VideoFrameDatasetCPU(video_path=self.args.video_file_path, num_frames_total=dataset_sequence_len, image_size=(args.image_h, args.image_w), frame_skip=args.frame_skip, data_fraction=1.0, val_fraction=args.val_fraction, mode='val', seed=args.seed, args=self.args)
            if len(self.val_dataset) > 0:
                self.val_loader = DataLoader(self.val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=args.drop_last_val)
                if len(self.val_loader) == 0: self.logger.warning(f"Validation DataLoader created but will yield 0 batches. Validation will be effectively skipped."); self.val_loader = None
                else: self.logger.info(f"Validation DataLoader created with {len(self.val_dataset)} samples. Effective batches: {len(self.val_loader)}")
            else: self.logger.warning("Validation dataset is empty after split, validation will be skipped."); self.val_loader = None
        else: self.logger.info("val_fraction is not in (0,1), validation will be skipped.")
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.logger.info("CPUDiffusionTrainer initialized for CPU execution (with new Encoder/Decoder).")

    def _compute_loss(self, predicted_noise_latent: torch.Tensor, target_noise_latent: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predicted_noise_latent, target_noise_latent)

    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, frames_to_log: Optional[torch.Tensor], num_frames_per_sequence_to_log: int = 1, num_sequences_to_log_max: int = 2):
        if not (self.args.wandb and WANDB_AVAILABLE and wandb.run and frames_to_log is not None and frames_to_log.numel() > 0): return
        current_frames_dim = frames_to_log.ndim
        if current_frames_dim == 4: frames_to_log_eff = frames_to_log.unsqueeze(1) # Add sequence dim if single frame per batch item
        elif current_frames_dim == 5: frames_to_log_eff = frames_to_log
        else: self.logger.warning(f"WandB log samples: unexpected shape {frames_to_log.shape} (original: {current_frames_dim}D). Skip."); return
        
        B_log, N_seq_log, C_log, _, _ = frames_to_log_eff.shape; num_to_log_seqs = min(B_log, num_sequences_to_log_max); num_frames_log_this = min(N_seq_log, num_frames_per_sequence_to_log)
        wandb_imgs = []
        for b in range(num_to_log_seqs):
            for f_idx in range(num_frames_log_this):
                frame = frames_to_log_eff[b,f_idx,...].cpu().float()
                if C_log == 1: frame = frame.repeat(3,1,1) # Ensure 3 channels for wandb.Image
                img_0_1 = (frame.clamp(-1,1)+1)/2.0
                wandb_imgs.append(wandb.Image(img_0_1, caption=f"{tag_prefix} S{b} F{f_idx} Ep{self.current_epoch+1} GStep{self.global_step}"))
        if wandb_imgs:
            try: wandb.log({f"samples_video_cpu_diffusion/{tag_prefix}": wandb_imgs}, step=self.global_step)
            except Exception as e: self.logger.error(f"WandB video sample log fail for {tag_prefix}: {e}", exc_info=False)
    
    @torch.no_grad()
    def _validate_epoch(self):
        if not self.val_loader: self.logger.info("Skipping validation: No effective validation data loader."); return
            
        self.model.eval(); total_val_loss = 0.0; num_val_batches_processed = 0
        val_prog_bar_desc = f"Validating Epoch {self.current_epoch+1}"
        if hasattr(self.val_loader.dataset, 'video_path') and self.val_loader.dataset.video_path: val_prog_bar_desc += f" ({Path(self.val_loader.dataset.video_path).name})"
        val_prog_bar = tqdm(self.val_loader, desc=val_prog_bar_desc, disable=(os.getenv('CI')=='true' or not self.args.show_train_progress_bar), dynamic_ncols=True, leave=False)
        
        first_batch_val_sampled_images = None; first_batch_val_target_images = None
        
        for batch_idx, val_frames_seq_raw in enumerate(val_prog_bar):
            val_frames_seq = val_frames_seq_raw.to(self.device); current_batch_size_val = val_frames_seq.shape[0]
            num_frames_to_process_diffusion = self.args.num_predict_frames if self.args.num_predict_frames > 0 else self.args.num_input_frames
            start_idx_for_x0 = self.args.num_input_frames if self.args.num_predict_frames > 0 else 0
            if val_frames_seq.shape[1] < start_idx_for_x0 + num_frames_to_process_diffusion: self.logger.warning(f"Val: Not enough frames in sequence ({val_frames_seq.shape[1]}) for diffusion. Skipping batch."); continue
            x0_target_frames_batch = val_frames_seq[:, start_idx_for_x0 : start_idx_for_x0 + num_frames_to_process_diffusion, ...]
            
            batch_loss_sum_for_item = 0.0; num_frames_processed_this_batch_item = 0
            for frame_idx_in_seq in range(x0_target_frames_batch.shape[1]):
                x0_single_frame_val = x0_target_frames_batch[:, frame_idx_in_seq, ...] # B, C, H, W
                t_val = torch.randint(0, self.diffusion_process.timesteps, (x0_single_frame_val.shape[0],), device=self.device).long()
                
                # Model's forward now handles encoding, noising, and predicting noise in latent
                predicted_noise_in_latent, target_noise_for_latent = self.model(x0_single_frame_val, t_val, self.diffusion_process, self.global_step)
                loss = self._compute_loss(predicted_noise_in_latent, target_noise_for_latent) 
                
                batch_loss_sum_for_item += loss.item(); num_frames_processed_this_batch_item += 1
            
            if num_frames_processed_this_batch_item > 0:
                avg_loss_for_batch_item = batch_loss_sum_for_item / num_frames_processed_this_batch_item
                total_val_loss += avg_loss_for_batch_item; num_val_batches_processed += 1
            
            if batch_idx == 0 and self.args.num_val_samples_to_log > 0 and hasattr(self.args, 'val_sampling_log_steps') and self.args.val_sampling_log_steps > 0:
                if x0_target_frames_batch.numel() > 0:
                    num_samples_to_gen = min(x0_target_frames_batch.shape[0], self.args.num_val_samples_to_log)
                    
                    # model_callable for diffusion_process.sample now needs to predict noise in latent space
                    model_lambda_for_sampling = lambda x_t_latent, t_samp: self.model.predict_noise_from_latent(x_t_latent, t_samp, self.global_step)
                    
                    self.logger.info(f"Generating {num_samples_to_gen} validation samples with {self.args.val_sampling_log_steps} steps (in latent space)...")
                    t_val_sample_start = time.time()
                    
                    denoised_latent_x0 = self.diffusion_process.sample(
                        model_callable=model_lambda_for_sampling, 
                        batch_size=num_samples_to_gen,
                        latent_feature_dim=self.model.latent_feature_dim_one_frame, # Pass latent dim
                        num_sampling_steps=self.args.val_sampling_log_steps, 
                        show_progress=getattr(self.args, 'show_val_sample_progress', False)
                    )
                    # Decode latent features to image
                    sampled_val_images = self.model.output_decoder(denoised_latent_x0)

                    t_val_sample_end = time.time(); self.logger.info(f"Validation sample generation (latent + decode) took {t_val_sample_end - t_val_sample_start:.2f}s.")
                    if sampled_val_images is not None and sampled_val_images.numel() > 0:
                         first_batch_val_sampled_images = sampled_val_images.detach().cpu() # Shape (num_samples_to_gen, C, H, W)
                    if x0_target_frames_batch.numel() > 0: # This is the target image for the *first frame of the sequence*
                        first_batch_val_target_images = x0_target_frames_batch[:num_samples_to_gen, 0:1, ...].detach().cpu() 
        
        if num_val_batches_processed > 0:
            avg_val_loss = total_val_loss / num_val_batches_processed
            self.logger.info(f"Validation Epoch {self.current_epoch+1}: Avg Loss: {avg_val_loss:.4f}")
            if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb_log_dict = {"val/loss_mse": avg_val_loss, "epoch": self.current_epoch}
                if self.args.wandb_log_val_recon_interval_epochs > 0 and (self.current_epoch + 1) % self.args.wandb_log_val_recon_interval_epochs == 0:
                    if first_batch_val_sampled_images is not None: self._log_samples_to_wandb("val_sampled_reduced_steps", first_batch_val_sampled_images, 1, self.args.num_val_samples_to_log) # Log first frame of sequence
                    if first_batch_val_target_images is not None: self._log_samples_to_wandb("val_target_frame0", first_batch_val_target_images, 1, self.args.num_val_samples_to_log)
                wandb.log(wandb_log_dict, step=self.global_step)
        else: self.logger.info(f"Validation Epoch {self.current_epoch+1}: No batches processed or val_loader was effectively empty.")
        self.model.train()

    def train(self, start_epoch: int = 0, initial_global_step: int = 0):
        self.global_step, self.current_epoch = initial_global_step, start_epoch
        self.logger.info(f"Starting CPU Diffusion training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch+1}, GStep: {initial_global_step}")
        if not self.train_loader or len(self.train_loader) == 0: self.logger.error(f"Training DataLoader is effectively empty. Cannot start training."); return
        
        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch; self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting.")
            self.model.train()
            prog_bar_desc = f"Epoch {epoch+1}"
            if hasattr(self.train_loader.dataset, 'video_path') and self.train_loader.dataset.video_path: prog_bar_desc += f" ({Path(self.train_loader.dataset.video_path).name})"
            prog_bar = tqdm(self.train_loader, desc=prog_bar_desc, disable=(os.getenv('CI')=='true' or not self.args.show_train_progress_bar), dynamic_ncols=True)

            for batch_idx, real_frames_seq_raw in enumerate(prog_bar):
                t_dataload_start = time.time(); t_dataload_end = time.time(); time_dataload = t_dataload_end - t_dataload_start
                t_todevice_start = time.time(); real_frames_seq = real_frames_seq_raw.to(self.device); t_todevice_end = time.time(); time_todevice = t_todevice_end - t_todevice_start
                t_train_cycle_start = time.time(); self.optimizer.zero_grad()
                
                num_frames_to_process_diffusion = self.args.num_predict_frames if self.args.num_predict_frames > 0 else self.args.num_input_frames
                start_idx_for_x0 = self.args.num_input_frames if self.args.num_predict_frames > 0 else 0
                if real_frames_seq.shape[1] < start_idx_for_x0 + num_frames_to_process_diffusion: self.logger.warning(f"Train: Not enough frames ({real_frames_seq.shape[1]}) for diffusion. Skip batch."); continue
                x0_frames_for_diffusion_batch = real_frames_seq[:, start_idx_for_x0 : start_idx_for_x0 + num_frames_to_process_diffusion, ...]
                
                batch_loss_sum_train = 0.0; num_frames_processed_train_batch_item = 0
                for frame_idx_in_selected_seq in range(x0_frames_for_diffusion_batch.shape[1]):
                    x0_single_frame_train = x0_frames_for_diffusion_batch[:, frame_idx_in_selected_seq, ...] # B, C, H, W
                    t_train = torch.randint(0, self.diffusion_process.timesteps, (x0_single_frame_train.shape[0],), device=self.device).long()
                    
                    model_timings = {}; t_model_fwd_start = time.time()
                    # Model's forward now handles encoding, noising, and predicting noise in latent
                    predicted_noise_in_latent, target_noise_for_latent = self.model(x0_single_frame_train, t_train, self.diffusion_process, self.global_step)
                    t_model_fwd_end = time.time(); model_timings["time_model_fwd"] = t_model_fwd_end - t_model_fwd_start
                    
                    if self.global_step < getattr(self.args, 'debug_initial_n_steps', 0) : 
                        # Logging for the new latent-space diffusion
                        self.logger.log(logging.DEBUG if logger_wubu_diffusion.level <= logging.DEBUG else logging.INFO, f"DEBUG GSTEP {self.global_step+1} (Batch {batch_idx}, FrameInSeq {frame_idx_in_selected_seq}):")
                        debug_log_func = self.logger.debug if logger_wubu_diffusion.level <= logging.DEBUG else self.logger.info
                        debug_log_func(f"  x0_single_frame_train   - Sh: {x0_single_frame_train.shape}")
                        # Cannot easily log encoded_features_x0 here as it's internal to model.forward for q_sample
                        debug_log_func(f"  target_noise_for_latent - Sh: {target_noise_for_latent.shape}, Mn: {target_noise_for_latent.mean().item():.2e}, Std: {target_noise_for_latent.std().item():.2e}")
                        debug_log_func(f"  PRED_noise_in_latent    - Sh: {predicted_noise_in_latent.shape}, Mn: {predicted_noise_in_latent.mean().item():.2e}, Std: {predicted_noise_in_latent.std().item():.2e}")

                    loss_train_for_frame = self._compute_loss(predicted_noise_in_latent, target_noise_for_latent)
                    batch_loss_sum_train += loss_train_for_frame; num_frames_processed_train_batch_item +=1
                
                if num_frames_processed_train_batch_item > 0: avg_loss_for_batch_item = batch_loss_sum_train / num_frames_processed_train_batch_item; avg_loss_for_batch_item.backward(); self.optimizer.step()
                else: avg_loss_for_batch_item = torch.tensor(0.0) 
                
                t_train_cycle_end = time.time(); time_train_cycle = t_train_cycle_end - t_train_cycle_start
                self.global_step +=1
                if self.global_step % self.args.log_interval == 0:
                    log_items = {"Loss": avg_loss_for_batch_item.item(), "T_DataLoad": time_dataload, "T_ToDevice": time_todevice, "T_ModelFwd": model_timings.get("time_model_fwd",0), "T_TrainCycle": time_train_cycle }
                    formatted_log_items = {k_l: (f"{v_l:.3f}s" if k_l.startswith("T_") else f"{v_l:.3f}") for k_l,v_l in log_items.items()}
                    log_str_parts = [f"{k_l}:{v_l}" for k_l,v_l in formatted_log_items.items()]; postfix_str = f"Loss:{avg_loss_for_batch_item.item():.3f} Data:{time_dataload:.2f}s"
                    if hasattr(prog_bar, 'set_postfix_str'): prog_bar.set_postfix_str(postfix_str)
                    full_log_msg = f"E:{epoch+1} S:{self.global_step} (B:{batch_idx+1}/{len(self.train_loader)}) | " + " | ".join(log_str_parts); self.logger.info(full_log_msg)
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log({"train/loss_mse_avg_batch_item": avg_loss_for_batch_item.item(), "time/data_load_batch_item": time_dataload, "time/to_device_batch_item": time_todevice, "time/model_fwd_last_frame": model_timings.get("time_model_fwd",0), "time/train_cycle_batch_item": time_train_cycle, "global_step":self.global_step, "epoch_frac":epoch+((batch_idx+1)/max(1,len(self.train_loader)))}, step=self.global_step)
                
                if self.args.wandb_log_train_samples_interval > 0 and self.global_step > 0 and self.global_step % self.args.wandb_log_train_samples_interval == 0:
                    if x0_frames_for_diffusion_batch.numel() > 0 :
                        num_samples_to_gen_train = min(x0_frames_for_diffusion_batch.shape[0], self.args.num_train_samples_to_log)
                        model_lambda_for_sampling_train = lambda x_t_latent, t_samp: self.model.predict_noise_from_latent(x_t_latent, t_samp, self.global_step) 
                        train_sampling_steps = getattr(self.args, 'train_sampling_log_steps', self.args.val_sampling_log_steps)
                        
                        denoised_latent_x0_train = self.diffusion_process.sample(model_lambda_for_sampling_train, batch_size=num_samples_to_gen_train, latent_feature_dim=self.model.latent_feature_dim_one_frame, num_sampling_steps=train_sampling_steps, show_progress=getattr(self.args, 'show_train_sample_progress', False))
                        sampled_train_img = self.model.output_decoder(denoised_latent_x0_train) # Decode here
                        
                        if sampled_train_img is not None: self._log_samples_to_wandb("train_sampled", sampled_train_img, 1, num_samples_to_gen_train)
                        self._log_samples_to_wandb("train_target_x0", x0_frames_for_diffusion_batch[:num_samples_to_gen_train,0:1,...].detach(), 1, num_samples_to_gen_train)
            
            if (epoch + 1) % self.args.validate_every_n_epochs == 0: self._validate_epoch()
            if self.args.save_interval > 0 and (epoch+1) > 0 and (epoch+1)%self.args.save_interval==0: self._save_checkpoint(epoch)
            if self.args.wandb_log_fixed_noise_samples_interval > 0 and (epoch+1) > 0 and (epoch+1)%self.args.wandb_log_fixed_noise_samples_interval==0:
                num_fixed_samples = self.args.num_val_samples_to_log
                model_lambda_for_fixed_noise = lambda x_t_latent, t_samp: self.model.predict_noise_from_latent(x_t_latent, t_samp, self.global_step) 
                fixed_noise_sampling_steps = getattr(self.args, 'fixed_noise_sampling_log_steps', self.args.val_sampling_log_steps)
                
                denoised_latent_x0_fixed = self.diffusion_process.sample( model_lambda_for_fixed_noise, batch_size=num_fixed_samples, latent_feature_dim=self.model.latent_feature_dim_one_frame, num_sampling_steps=fixed_noise_sampling_steps, show_progress=getattr(self.args, 'show_fixed_noise_sample_progress', False) )
                fixed_noise_pixels = self.model.output_decoder(denoised_latent_x0_fixed) # Decode here

                if fixed_noise_pixels is not None: self._log_samples_to_wandb("fixed_noise_gen_diffusion",fixed_noise_pixels,1,num_fixed_samples)
        
        self.logger.info("CPU Diffusion Training finished.")
        self._save_checkpoint(self.current_epoch,is_final=True)

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()}
        for k_arg, v_arg in serializable_args.items():
            if not isinstance(v_arg, (str, int, float, bool, list, dict, type(None))): serializable_args[k_arg] = str(v_arg)
        data = { 'epoch': epoch,'global_step':self.global_step,'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict(), 'args':serializable_args}
        fn = f"wubudiffusion_v1_{'final' if is_final else f'ep{epoch+1}_step{self.global_step}'}.pt"
        try: torch.save(data, Path(self.args.checkpoint_dir)/fn); self.logger.info(f"Checkpoint saved: {fn}")
        except Exception as e: self.logger.error(f"Error saving ckpt {fn}: {e}", exc_info=True)

    def load_checkpoint(self, ckpt_path_str: Optional[str]) -> Tuple[int,int]:
        if not ckpt_path_str or not os.path.exists(ckpt_path_str): self.logger.warning(f"Checkpoint '{ckpt_path_str}' not found. Starting fresh."); return 0,0
        try: ckpt = torch.load(ckpt_path_str, map_location=self.device); self.logger.info(f"Loaded checkpoint: {ckpt_path_str}")
        except Exception as e: self.logger.error(f"Failed to load ckpt {ckpt_path_str}: {e}. Fresh start.", exc_info=True); return 0,0
        
        if 'args' in ckpt and ckpt['args'] is not None:
            self.logger.info("Attempting to load args from checkpoint."); loaded_ckpt_args_dict = ckpt['args']; current_args_dict = vars(self.args)
            critical_structure_args = [ 'image_h', 'image_w', 'num_channels', 'diffusion_time_embed_dim', 'model_wubu_num_virtual_levels', 'model_wubu_base_micro_dim', 'model_wubu_num_physical_stages', 'use_bsp_gate', 'bsp_num_paths', 'bsp_intermediate_dim', 'use_manifold_uncrumple_transform', 'wubu_cond_embed_dim', 'wubu_condition_on_s', 'wubu_condition_on_c', 'wubu_condition_on_micro_idx', 'wubu_condition_on_stage_idx', 'img_encoder_grid_region_w', 'img_encoder_grid_region_h', 'img_encoder_patch_embedding_dim'] # Added new encoder args
            for k_ckpt, v_ckpt_loaded in loaded_ckpt_args_dict.items():
                v_ckpt = v_ckpt_loaded 
                if k_ckpt in current_args_dict and isinstance(v_ckpt, str): 
                    # Attempt type conversion for bool, int, float if current arg has that type
                    if isinstance(current_args_dict[k_ckpt], bool):
                        v_ckpt_lower = v_ckpt.lower()
                        if v_ckpt_lower == 'true': v_ckpt = True
                        elif v_ckpt_lower == 'false': v_ckpt = False
                        # else: keep v_ckpt as str if not 'true'/'false'
                    elif isinstance(current_args_dict[k_ckpt], int):
                        try:
                            v_ckpt = int(v_ckpt)
                        except ValueError:
                            pass # Keep v_ckpt as str if conversion fails
                    elif isinstance(current_args_dict[k_ckpt], float):
                        try:
                            v_ckpt = float(v_ckpt)
                        except ValueError:
                            pass # Keep v_ckpt as str if conversion fails
                            
                if k_ckpt in critical_structure_args and k_ckpt in current_args_dict:
                    current_val_for_compare, v_ckpt_for_compare = current_args_dict[k_ckpt], v_ckpt
                    current_val_str_for_compare = str(current_val_for_compare).lower() if isinstance(current_val_for_compare, bool) else str(current_val_for_compare)
                    v_ckpt_str_for_compare = str(v_ckpt_for_compare).lower() if isinstance(v_ckpt_for_compare, bool) else str(v_ckpt_for_compare)
                    if current_val_str_for_compare != v_ckpt_str_for_compare: self.logger.warning(f"CRITICAL ARG MISMATCH for '{k_ckpt}': Ckpt='{v_ckpt_str_for_compare}', Current CLI='{current_val_str_for_compare}'. Model might not load correctly.")
                if k_ckpt not in current_args_dict: setattr(self.args, k_ckpt, v_ckpt); self.logger.info(f"  Arg '{k_ckpt}' loaded from ckpt: '{v_ckpt}' (added to current args)")
                elif current_args_dict[k_ckpt] != v_ckpt and k_ckpt not in critical_structure_args: self.logger.info(f"  Arg '{k_ckpt}': Ckpt='{v_ckpt}', Current CLI='{current_args_dict[k_ckpt]}'. Using current CLI value.")
        
        model_state_dict = ckpt.get('model_state_dict')
        if model_state_dict:
            try: self.model.load_state_dict(model_state_dict, strict=True); self.logger.info("Successfully loaded model_state_dict (strict=True).")
            except RuntimeError as e_strict:
                self.logger.warning(f"Error loading model_state_dict with strict=True: {e_strict}. Trying strict=False.")
                try: self.model.load_state_dict(model_state_dict, strict=False); self.logger.info("Successfully loaded model_state_dict with strict=False.")
                except Exception as e_non_strict: self.logger.error(f"Failed to load model_state_dict even with strict=False: {e_non_strict}. Model weights random.")
        
        optimizer_state_dict = ckpt.get('optimizer_state_dict')
        if optimizer_state_dict and not self.args.load_checkpoint_reset_epoch:
            try: self.optimizer.load_state_dict(optimizer_state_dict); self.logger.info("Successfully loaded optimizer_state_dict.")
            except ValueError as e_val: self.logger.error(f"Error loading optimizer_state_dict (param mismatch?): {e_val}. Optimizer fresh.")
            except Exception as e_opt: self.logger.error(f"Other error loading optimizer_state_dict: {e_opt}. Optimizer fresh.")
        elif self.args.load_checkpoint_reset_epoch: self.logger.info("load_checkpoint_reset_epoch is True. Optimizer state not loaded, starting fresh.")
        
        gs, ep_saved = ckpt.get('global_step',0), ckpt.get('epoch',0)
        start_ep, start_gs = (0 if self.args.load_checkpoint_reset_epoch else ep_saved + 1), (0 if self.args.load_checkpoint_reset_epoch else gs)
        self.logger.info(f"Resuming. Ckpt Epoch: {ep_saved}, Ckpt GlobalStep: {gs}. Effective Start Epoch: {start_ep}, Effective Start GStep: {start_gs}.")
        return start_gs, start_ep
        
def parse_diffusion_arguments():
    def bool_type(s): s_lower = str(s).lower(); return s_lower in ('yes', 'true', 't', 'y', '1')
    parser = argparse.ArgumentParser(description="WuBu CPU-Only Diffusion Model (v1 Advanced FDQWR - Latent Diffusion with Encoder/Decoder)")
    # Paths & General Training
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_cpu_diffusion")
    parser.add_argument('--checkpoint_dir', type=str, default='wubudiffusion_v1_checkpoints_latent_adv_robust')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--seed',type=int, default=42); parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200); parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate',type=float,default=1e-4)
    # Data/Model Specs
    parser.add_argument('--image_h', type=int, default=64); parser.add_argument('--image_w', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_input_frames', type=int, default=1); parser.add_argument('--num_predict_frames', type=int, default=0)
    parser.add_argument('--frame_skip', type=int, default=1)
    # New Image Encoder/Decoder Args
    parser.add_argument('--img_encoder_grid_region_w', type=int, default=16, help="Width of a region on original image for encoder.")
    parser.add_argument('--img_encoder_grid_region_h', type=int, default=16, help="Height of a region on original image for encoder.")
    parser.add_argument('--img_encoder_patch_embedding_dim', type=int, default=64, help="Dimension of the learned embedding for each image region.")
    parser.add_argument('--img_encoder_mlp_hidden_factor', type=float, default=1.0, help="Hidden layer factor for encoder/decoder patch MLPs.")
    # Diffusion Specific Params
    parser.add_argument('--diffusion_timesteps', type=int, default=1000)
    parser.add_argument('--diffusion_beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--diffusion_beta_start', type=float, default=0.0001); parser.add_argument('--diffusion_beta_end', type=float, default=0.02)
    parser.add_argument('--diffusion_time_embed_dim', type=int, default=128)
    # WuBu Core Common Params (FDQWR operates on latent embeddings now)
    parser.add_argument('--wubu_initial_s', type=float, default=1.0); parser.add_argument('--wubu_s_decay', type=float, default=0.9999)
    parser.add_argument('--wubu_initial_c', type=float, default=0.1); parser.add_argument('--wubu_c_phi_influence', type=bool_type, default=True)
    parser.add_argument('--wubu_num_phi_scaffold_points', type=int, default=3); parser.add_argument('--wubu_phi_scaffold_init_scale', type=float, default=0.001)
    parser.add_argument('--wubu_use_gaussian_rungs', type=bool_type, default=True)
    parser.add_argument('--wubu_base_gaussian_std_dev', type=float, default=0.5)
    parser.add_argument('--wubu_gaussian_std_dev_decay', type=float, default=0.999995)
    parser.add_argument('--wubu_rung_affinity_temp', type=float, default=0.01); parser.add_argument('--wubu_rung_modulation_strength', type=float, default=0.001)
    parser.add_argument('--wubu_t_tilde_scale', type=float, default=1.0) 
    parser.add_argument('--wubu_micro_transform_type', type=str, default="mlp", choices=["mlp", "linear", "identity"]) 
    parser.add_argument('--wubu_micro_transform_hidden_factor', type=float, default=0.5) 
    parser.add_argument('--wubu_use_quaternion_so4', type=bool_type, default=True) 
    parser.add_argument('--wubu_scaffold_co_rotation_mode', type=str, default="none", choices=["none", "matrix_only"]) 
    parser.add_argument('--wubu_enable_isp', type=bool_type, default=True) 
    parser.add_argument('--wubu_enable_stateful', type=bool_type, default=True) 
    parser.add_argument('--wubu_enable_hypernet', type=bool_type, default=True); 
    parser.add_argument('--wubu_internal_state_factor', type=float, default=0.5) 
    parser.add_argument('--wubu_hypernet_strength', type=float, default=0.01) 
    parser.add_argument('--wubu_use_residual_in_micro', type=bool_type, default=True)
    parser.add_argument('--wubu_learnable_affinity_params', type=bool_type, default=True)
    # WuBu Stack Specifics
    parser.add_argument('--model_wubu_num_virtual_levels', type=int, default=1500)
    parser.add_argument('--model_wubu_base_micro_dim', type=int, default=128) # Dimensionality FDQWR works in (not related to image patch size directly anymore)
    parser.add_argument('--model_wubu_num_physical_stages', type=int, default=30)
    # ManifoldUncrumpleTransform (MUT) Specific Params
    parser.add_argument('--use_manifold_uncrumple_transform', type=bool_type, default=False)
    parser.add_argument('--uncrumple_hypernet_hidden_factor', type=float, default=1.0)
    parser.add_argument('--uncrumple_learn_rotation', type=bool_type, default=True); parser.add_argument('--uncrumple_learn_scale', type=bool_type, default=True)
    parser.add_argument('--uncrumple_scale_activation', type=str, default='sigmoid', choices=['sigmoid', 'tanh', 'exp', 'softplus'])
    parser.add_argument('--uncrumple_scale_min', type=float, default=0.5); parser.add_argument('--uncrumple_scale_max', type=float, default=2.0)
    parser.add_argument('--uncrumple_learn_translation', type=bool_type, default=True)
    # BSP Gate Parameters (operates on latent FDQWR outputs)
    parser.add_argument('--use_bsp_gate', type=bool_type, default=False); parser.add_argument('--bsp_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--bsp_num_paths', type=int, default=2); parser.add_argument('--bsp_intermediate_dim', type=int, default=256) # Dim from FDQWR before BSP heads
    parser.add_argument('--bsp_gate_activation', type=str, default='softmax', choices=['softmax', 'sigmoid'])
    parser.add_argument('--bsp_log_gate_interval_mult', type=int, default=5)
    # WuBu Contextual Conditioning and Stability
    parser.add_argument('--wubu_cond_embed_dim', type=int, default=16)
    parser.add_argument('--wubu_condition_on_s', type=bool_type, default=True); parser.add_argument('--wubu_condition_on_c', type=bool_type, default=False) 
    parser.add_argument('--wubu_condition_on_micro_idx', type=bool_type, default=True); parser.add_argument('--wubu_condition_on_stage_idx', type=bool_type, default=True)
    parser.add_argument('--wubu_max_micro_idx_embed', type=int, default=1000); parser.add_argument('--wubu_max_stage_idx_embed', type=int, default=100)
    parser.add_argument('--wubu_inter_micro_norm_interval', type=int, default=0)
    # Logging & Saving
    parser.add_argument('--wandb', type=bool_type, default=True); parser.add_argument('--wandb_project',type=str,default='WuBuDiffusionV1Latent')
    parser.add_argument('--wandb_run_name',type=str,default=None); parser.add_argument('--log_interval',type=int, default=10)
    parser.add_argument('--train_progress_update_interval_steps', type=int, default=None, help="How often to update FDQWR progress bar during training (model forward). Default: log_interval")
    parser.add_argument('--save_interval',type=int, default=5);
    parser.add_argument('--wandb_log_train_samples_interval', type=int, default=50)
    parser.add_argument('--train_sampling_log_steps', type=int, default=50); parser.add_argument('--show_train_sample_progress', type=bool_type, default=False)
    parser.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=10)
    parser.add_argument('--fixed_noise_sampling_log_steps', type=int, default=50); parser.add_argument('--show_fixed_noise_sample_progress', type=bool_type, default=False)
    parser.add_argument('--num_val_samples_to_log', type=int, default=2); parser.add_argument('--num_train_samples_to_log', type=int, default=2)
    parser.add_argument('--data_fraction', type=float, default=0.1); parser.add_argument('--show_train_progress_bar', type=bool_type, default=True)
    parser.add_argument('--getitem_log_interval', type=int, default=100); parser.add_argument('--getitem_slow_threshold', type=float, default=1.0)
    parser.add_argument('--load_checkpoint_reset_epoch', type=bool_type, default=False)
    parser.add_argument('--debug_initial_n_steps', type=int, default=0, help="Number of initial training steps for verbose tensor logging in trainer.") 
    parser.add_argument('--debug_micro_transform_internals_global', type=bool_type, default=False, help="Enable detailed FDQ/MUT internal logging globally.")
    # Validation
    parser.add_argument('--val_fraction', type=float, default=0.1); parser.add_argument('--val_batch_size', type=int, default=None)
    parser.add_argument('--validate_every_n_epochs', type=int, default=2)
    parser.add_argument('--wandb_log_val_recon_interval_epochs', type=int, default=1)
    parser.add_argument('--val_sampling_log_steps', type=int, default=64); parser.add_argument('--show_val_sample_progress', type=bool_type, default=True)
    # DataLoader drop_last arguments
    parser.add_argument('--drop_last_train', type=bool_type, default=True, help="Whether to drop the last incomplete batch during training.")
    parser.add_argument('--drop_last_val', type=bool_type, default=False, help="Whether to drop the last incomplete batch during validation.")

    # DEPRECATED / Replaced by new encoder args (comment out or remove if fully confident)
    # parser.add_argument('--gaad_num_regions', type=int, default=16) # Now derived from grid
    # parser.add_argument('--gaad_min_size_px', type=int, default=4) # Not used by simple grid
    # parser.add_argument('--patch_size_h', type=int, default=8) # Replaced by img_encoder_grid_region_h
    # parser.add_argument('--patch_size_w', type=int, default=8) # Replaced by img_encoder_grid_region_w

    parsed_args = parser.parse_args()
    if parsed_args.val_batch_size is None: parsed_args.val_batch_size = parsed_args.batch_size
    if parsed_args.train_progress_update_interval_steps is None: parsed_args.train_progress_update_interval_steps = parsed_args.log_interval

    if parsed_args.use_bsp_gate and parsed_args.bsp_num_paths <= 1: logger_wubu_diffusion.warning("use_bsp_gate is True but bsp_num_paths <= 1. Disabling BSP gate."); parsed_args.use_bsp_gate = False
    if parsed_args.use_bsp_gate and parsed_args.bsp_intermediate_dim <=0: logger_wubu_diffusion.error("use_bsp_gate is True but bsp_intermediate_dim <=0. Setting to 256."); parsed_args.bsp_intermediate_dim = 256
    if parsed_args.use_manifold_uncrumple_transform and (parsed_args.model_wubu_base_micro_dim <=0 or parsed_args.wubu_num_phi_scaffold_points <=0 ): logger_wubu_diffusion.warning("use_manifold_uncrumple_transform is True but base_micro_dim or num_phi_scaffold_points is <=0. Disabling MUT."); parsed_args.use_manifold_uncrumple_transform = False
    
    # Validation for new encoder args
    if parsed_args.img_encoder_grid_region_w <= 0 or parsed_args.img_encoder_grid_region_h <= 0:
        raise ValueError("img_encoder_grid_region_w and img_encoder_grid_region_h must be positive.")
    if parsed_args.img_encoder_patch_embedding_dim <= 0:
        raise ValueError("img_encoder_patch_embedding_dim must be positive.")

    return parsed_args
    
def main_cpu_diffusion():
    args = parse_diffusion_arguments()
    
    global tqdm 
    if not args.show_train_progress_bar:
        logger_wubu_diffusion.info("TQDM progress bars are disabled via args.show_train_progress_bar=False.")
        class _TqdmDummyArgDisabled:
            _last_desc_printed = None
            def __init__(self, iterable=None, *args_dummy, **kwargs_dummy):
                self.iterable = iterable if iterable is not None else []
                desc = kwargs_dummy.get('desc')
                # if desc and desc != _TqdmDummyArgDisabled._last_desc_printed: _TqdmDummyArgDisabled._last_desc_printed = desc # Avoid print spam
            def __iter__(self): return iter(self.iterable)
            def __enter__(self): return self
            def __exit__(self, *exc_info): pass
            def set_postfix(self, ordered_dict=None, refresh=True, **kwargs_p): pass
            def set_postfix_str(self, s: str = "", refresh: bool = True): pass
            def update(self, n: int = 1): pass
            def close(self): pass
        tqdm = _TqdmDummyArgDisabled 
    elif _tqdm_imported_successfully: logger_wubu_diffusion.info("TQDM progress bars are ENABLED (tqdm library imported).")
    else: logger_wubu_diffusion.info("TQDM progress bars are using a basic dummy (tqdm library not found).")

    global DEBUG_MICRO_TRANSFORM_INTERNALS; DEBUG_MICRO_TRANSFORM_INTERNALS = args.debug_micro_transform_internals_global
    if args.debug_initial_n_steps > 0 or DEBUG_MICRO_TRANSFORM_INTERNALS: logger_wubu_diffusion.setLevel(logging.DEBUG); logger_wubu_diffusion.debug("Logger set to DEBUG level.")
    else: logger_wubu_diffusion.setLevel(logging.INFO)
    for h in logger_wubu_diffusion.handlers: h.setLevel(logger_wubu_diffusion.level)
    logging.basicConfig(level=logger_wubu_diffusion.level, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s', force=True) 
    
    logger_wubu_diffusion.info(f"--- WuBuDiffusionV1 (Latent Diffusion with Encoder/Decoder & Adv FDQWR) ---")
    logger_wubu_diffusion.info(f"Effective Args: {vars(args)}")
    
    random.seed(args.seed); os.environ['PYTHONHASHSEED'] = str(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubudiff_latent_adv_robust_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        try: wandb.init(project=args.wandb_project, name=run_name, config=vars(args)); logger_wubu_diffusion.info(f"WandB initialized: Run '{run_name}', Project '{args.wandb_project}'")
        except Exception as e_wandb: logger_wubu_diffusion.error(f"WandB initialization failed: {e_wandb}", exc_info=True); args.wandb = False

    device = torch.device("cpu")
    video_config = { "num_channels": args.num_channels, "num_input_frames": args.num_input_frames, "num_predict_frames": args.num_predict_frames }
    
    model = WuBuDiffusionModel(args, video_config).to(device) # gaad_config removed
    diffusion_process = DiffusionProcess(timesteps=args.diffusion_timesteps, beta_schedule=args.diffusion_beta_schedule, beta_start=args.diffusion_beta_start, beta_end=args.diffusion_beta_end, device=device)

    if args.wandb and WANDB_AVAILABLE and wandb.run is not None: 
        try: wandb.watch(model, log="gradients", log_freq=args.log_interval * 10, log_graph=False)
        except Exception as e_watch: logger_wubu_diffusion.error(f"WandB watch failed: {e_watch}")

    trainer = CPUDiffusionTrainer(model, diffusion_process, args)
    start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint) if args.load_checkpoint else (0,0)
    
    try: trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt: logger_wubu_diffusion.info("Training interrupted by user.")
    except Exception as e: logger_wubu_diffusion.error(f"Training loop crashed: {e}", exc_info=True)
    finally:
        logger_wubu_diffusion.info("Finalizing run...")
        if hasattr(trainer, 'current_epoch') and hasattr(trainer, 'global_step') and hasattr(trainer, '_save_checkpoint'): trainer._save_checkpoint(epoch=trainer.current_epoch, is_final=True)
        if args.wandb and WANDB_AVAILABLE and wandb.run is not None: 
            try: wandb.finish()
            except Exception as e_finish: logger_wubu_diffusion.error(f"WandB finish failed: {e_finish}")
        logger_wubu_diffusion.info("WuBuDiffusionV1 (Latent Diffusion with Encoder/Decoder & Adv FDQWR) script finished.")
        
if __name__ == "__main__":
    main_cpu_diffusion()
