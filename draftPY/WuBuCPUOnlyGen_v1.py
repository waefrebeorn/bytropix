
# WuBuCPUOnlyGen_v1.py
# VAE-GAN like structure using ADVANCED FractalDepthQuaternionWuBuRungs.
# CPU-only, NO CNNs, NO explicit DFT/DCT for features (WuBu processes patch pixels).
# Based on concepts from WuBuGAADHybridGen_v0.3.py and WuBu_Integrated_Full.py (v...08)
# THIS VERSION: Integrates advanced FDQWuBuR features.

import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
# import heapq # Not used
import math, random, argparse, logging, time, os #, platform, gc, functools # platform, gc, functools not used
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque, defaultdict # defaultdict used in analyze_model_parameters
from pathlib import Path
# from torch import amp # AMP not used for CPU
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
# from PIL import Image

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
EPS = 1e-5
PHI = (1 + math.sqrt(5)) / 2
TAN_VEC_CLAMP_VAL = 1e4
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e8
MIN_WUBU_LEVEL_SCALE = EPS # Not directly used by FDQWR, but good constant
MAX_WUBU_LEVEL_SCALE = 10.0 # Not directly used by FDQWR

# --- Basic Logging Setup ---
logger_wubu_cpu_gen = logging.getLogger("WuBuCPUOnlyGenV1")
if not logger_wubu_cpu_gen.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_wubu_cpu_gen.addHandler(handler)
    logger_wubu_cpu_gen.setLevel(logging.INFO)

# --- Global flag for detailed micro-transform debugging (from WuBu_Integrated_Full.py) ---
DEBUG_MICRO_TRANSFORM_INTERNALS = False


# --- Utility Functions (some from WuBu_Integrated_Full.py) ---
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
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', True)): # Check for affine before accessing weight/bias
            if hasattr(m, 'weight') and m.weight is not None: nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor:
    return F.softplus(param_unconstrained) + min_val

def print_tensor_stats(tensor: Optional[torch.Tensor], name: str, micro_level_idx: Optional[int] = None, stage_idx: Optional[int] = None, enabled: bool = True, batch_idx_to_print: int = 0):
    if not enabled or tensor is None or not DEBUG_MICRO_TRANSFORM_INTERNALS:
        return
    if tensor.numel() == 0:
        return
    item_tensor = tensor
    prefix = f"L{micro_level_idx if micro_level_idx is not None else 'N'}/S{stage_idx if stage_idx is not None else 'N'}| "
    if tensor.dim() > 1 and tensor.shape[0] > batch_idx_to_print :
        item_tensor = tensor[batch_idx_to_print].detach()
        prefix += f"{name}(i{batch_idx_to_print}):"
    elif tensor.dim() == 1:
        item_tensor = tensor.detach()
        prefix += f"{name}:"
    else:
        item_tensor = tensor.detach()
        prefix += f"{name}(f):"
    # Using logger for debug prints if it's set to DEBUG level
    if logger_wubu_cpu_gen.isEnabledFor(logging.DEBUG):
        logger_wubu_cpu_gen.debug(prefix +
            f"Sh:{tensor.shape},Dt:{tensor.dtype},"
            f"Min:{item_tensor.min().item():.2e},Max:{item_tensor.max().item():.2e},"
            f"Mu:{item_tensor.mean().item():.2e},Std:{item_tensor.std().item():.2e},"
            f"Nrm:{torch.linalg.norm(item_tensor.float()).item():.2e}")

# --- Quaternion Math (from WuBu_Integrated_Full.py, slightly adapted logger) ---
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
            logger_wubu_cpu_gen.warning_once(f"Quat axis-angle: Cannot broadcast axis (shape {axis.shape}) to align with angle_rad (shape {angle_rad.shape}). Using default.") # logger changed
            effective_axis_shape = list(angle_rad.shape) + [3]
            axis_normalized = torch.zeros(effective_axis_shape, device=angle_rad.device, dtype=angle_rad.dtype)
            if axis_normalized.numel() > 0: axis_normalized[..., 0] = 1.0
    angle_half = angle_rad / 2.0
    q_w = torch.cos(angle_half).unsqueeze(-1)
    q_xyz = axis_normalized * torch.sin(angle_half).unsqueeze(-1)
    return torch.cat([q_w, q_xyz], dim=-1)

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    s1_orig, s2_orig = list(q1.shape), list(q2.shape)
    if s1_orig[-1] != 4 or s2_orig[-1] != 4:
        raise ValueError(f"Quaternions must have 4 components. Got q1: {s1_orig}, q2: {s2_orig}")
    q1_eff, q2_eff = q1, q2
    if len(s1_orig) < len(s2_orig): q1_eff = q1.view([1]*(len(s2_orig)-len(s1_orig)) + s1_orig)
    elif len(s2_orig) < len(s1_orig): q2_eff = q2.view([1]*(len(s1_orig)-len(s2_orig)) + s2_orig)
    try:
        w1, x1, y1, z1 = q1_eff[..., 0:1], q1_eff[..., 1:2], q1_eff[..., 2:3], q1_eff[..., 3:4]
        w2, x2, y2, z2 = q2_eff[..., 0:1], q2_eff[..., 1:2], q2_eff[..., 2:3], q2_eff[..., 3:4]
    except IndexError as e:
        logger_wubu_cpu_gen.error(f"Quat component extraction failed. q1_eff: {q1_eff.shape}, q2_eff: {q2_eff.shape}. Error: {e}") # logger changed
        raise e
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.cat([w, x, y, z], dim=-1)

def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)

def normalize_quaternion(q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    return q / (norm + eps)

# --- Hyperbolic Geometry Utilities (from WuBu_Integrated_Full.py) ---
class HyperbolicUtils: # (Content largely unchanged, assumed to be robust)
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

# --- PoincareBall Class (from WuBu_Integrated_Full.py) ---
class PoincareBall:
    def __init__(self, c_scalar: float = 1.0): # From WuBu_Integrated_Full
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0: self.c, self.sqrt_c, self.radius = 0.0, 0.0, float('inf')
        else: self.c, self.sqrt_c, self.radius = c_scalar, math.sqrt(self.c), 1.0 / (math.sqrt(self.c) + EPS)
        self._name = f'PoincareBall(c={self.c:.3g})'
    @property
    def name(self) -> str: return self._name
    def proju(self, x: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.poincare_clip(x, self.c, radius=1.0, eps=EPS * 10)
    def expmap0_scaled(self, dp: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_exponential_map(dp, self.c, scale_scalar=scale_scalar, eps=EPS)
    def logmap0_scaled(self, p: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_logarithmic_map(p, self.c, scale_scalar=scale_scalar, eps=EPS)


# --- Core WuBu Rotational and Transformational Components (Advanced Version from WuBu_Integrated_Full.py v...08) ---
class SkewSymmetricMatrix(nn.Module): # From WuBu_Integrated_Full
    def __init__(self, n_dim: int):
        super().__init__()
        self.n_dim = n_dim
        if n_dim <= 1: self.num_params = 0; self.register_parameter('params', None)
        else: self.num_params = n_dim * (n_dim - 1) // 2; self.params = nn.Parameter(torch.randn(self.num_params) * 0.001)
    def forward(self) -> torch.Tensor:
        dev = self.params.device if self.params is not None else torch.device('cpu')
        dtype = self.params.dtype if self.params is not None else torch.float32
        if self.n_dim <= 1 or self.params is None: return torch.eye(self.n_dim, device=dev, dtype=dtype)
        X = torch.zeros(self.n_dim, self.n_dim, device=dev, dtype=dtype)
        indices = torch.triu_indices(self.n_dim, self.n_dim, offset=1, device=dev)
        if X.numel() > 0 and indices.numel() > 0 and self.params is not None and self.params.numel() > 0:
             if indices.shape[1] == self.params.shape[0]: X[indices[0], indices[1]] = self.params
             else:
                logger_wubu_cpu_gen.warning_once(f"SkewSymMtx: Mismatch triu_idx ({indices.shape[1]}) vs params ({self.params.shape[0]}) for dim {self.n_dim}.")
                num_to_fill = min(indices.shape[1], self.params.shape[0])
                if num_to_fill > 0: X[indices[0,:num_to_fill], indices[1,:num_to_fill]] = self.params[:num_to_fill]
        X = X - X.T
        try: R = torch.linalg.matrix_exp(torch.clamp(X, -5, 5))
        except Exception as e:
            logger_wubu_cpu_gen.error(f"matrix_exp error: {e}. Using identity. X norm: {X.norm().item() if X.numel()>0 else 'N/A'}", exc_info=True)
            R = torch.eye(self.n_dim, device=X.device, dtype=X.dtype)
        return R

class QuaternionSO4From6Params(nn.Module): # From WuBu_Integrated_Full
    def __init__(self, small_init_std: float = 0.001):
        super().__init__()
        self.so3_params_for_p = nn.Parameter(torch.randn(3) * small_init_std)
        self.so3_params_for_q = nn.Parameter(torch.randn(3) * small_init_std)
    def _so3_params_to_unit_quaternion(self, so3_params: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.norm(so3_params, dim=-1)
        identity_q_val = torch.zeros(4, device=so3_params.device, dtype=so3_params.dtype)
        if identity_q_val.numel() > 0 : identity_q_val[0] = 1.0
        if angle < EPS : return identity_q_val
        axis = so3_params / (angle.unsqueeze(-1) + EPS) # Ensure axis has last dim for broadcasting with angle
        return quaternion_from_axis_angle(angle, axis) # angle is scalar or (B,), axis is (3,) or (B,3)
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._so3_params_to_unit_quaternion(self.so3_params_for_p), self._so3_params_to_unit_quaternion(self.so3_params_for_q)

class FractalMicroTransformRungs(nn.Module): # From WuBu_Integrated_Full (v...08)
    def __init__(self, dim: int = 4, transform_type: str = "mlp",
                 hidden_dim_factor: float = 1.0,
                 use_quaternion_so4: bool = True,
                 enable_internal_sub_processing: bool = True,
                 enable_stateful_micro_transform: bool = True,
                 enable_hypernetwork_modulation: bool = True,
                 internal_state_dim_factor: float = 1.0,
                 hyper_mod_strength: float = 0.01
                 ):
        super().__init__()
        self.dim = dim
        self.use_quaternion_so4 = use_quaternion_so4 and (dim == 4)

        self.enable_internal_sub_processing = enable_internal_sub_processing and (dim == 4)
        self.enable_stateful_micro_transform = enable_stateful_micro_transform
        self.enable_hypernetwork_modulation = enable_hypernetwork_modulation
        self.hyper_mod_strength = hyper_mod_strength
        self.debug_counter = 0

        if self.use_quaternion_so4:
            self.rotation_param_generator = QuaternionSO4From6Params()
            self.rotation_generator_matrix = None
        elif dim > 0:
            self.rotation_generator_matrix = SkewSymmetricMatrix(dim)
            self.rotation_param_generator = None
        else:
            self.rotation_generator_matrix = nn.Identity()
            self.rotation_param_generator = None

        mlp_hidden_dim = max(dim, int(dim * hidden_dim_factor)) if dim > 0 else 0
        if transform_type == 'mlp' and dim > 0 and mlp_hidden_dim > 0:
            self.non_rotational_map_layers = nn.ModuleList([
                nn.Linear(dim, mlp_hidden_dim), nn.GELU(),
                nn.Linear(mlp_hidden_dim, dim)
            ])
        elif transform_type == 'linear' and dim > 0:
             self.non_rotational_map_layers = nn.ModuleList([nn.Linear(dim, dim)])
        else:
            self.non_rotational_map_layers = nn.ModuleList([nn.Identity()])

        if self.enable_stateful_micro_transform and self.dim > 0:
            self.internal_state_dim = max(1, int(self.dim * internal_state_dim_factor))
            state_mlp_hidden = max(self.internal_state_dim // 2, dim // 2, 1)
            self.state_update_gate_mlp = nn.Sequential(
                nn.Linear(dim + self.internal_state_dim, state_mlp_hidden), nn.GELU(),
                nn.Linear(state_mlp_hidden, self.internal_state_dim), nn.Tanh()
            )
            self.state_influence_mlp = nn.Sequential(
                nn.Linear(self.internal_state_dim, state_mlp_hidden), nn.GELU(),
                nn.Linear(state_mlp_hidden, dim)
            )
        else:
            self.internal_state_dim = 0
            self.state_update_gate_mlp = None
            self.state_influence_mlp = None

        if self.enable_hypernetwork_modulation and self.dim > 0:
            if len(self.non_rotational_map_layers) > 0 and isinstance(self.non_rotational_map_layers[0], nn.Linear):
                first_linear_out_features = self.non_rotational_map_layers[0].out_features
                hyper_hidden_dim = max(dim // 2, first_linear_out_features // 2, 1)
                self.hyper_bias_generator = nn.Sequential(
                    nn.Linear(dim, hyper_hidden_dim), nn.GELU(),
                    nn.Linear(hyper_hidden_dim, first_linear_out_features)
                )
            else:
                self.enable_hypernetwork_modulation = False
                self.hyper_bias_generator = None
                if dim > 0 : logger_wubu_cpu_gen.warning_once("HyperNet modulation enabled but no suitable MLP found. Disabling for this micro-transform.")
        else:
            self.hyper_bias_generator = None

        if self.enable_internal_sub_processing and self.dim == 4:
            self.w_modulator = nn.Sequential(nn.Linear(1, 2), nn.GELU(), nn.Linear(2, 1), nn.Sigmoid())
            self.v_scaler_param = nn.Parameter(torch.randn(1) * 0.01)
        else:
            self.w_modulator = None
            self.v_scaler_param = None
        self.apply(init_weights_general)

    def _apply_internal_quaternion_sub_processing(self, x_q: torch.Tensor, micro_level_idx: Optional[int], stage_idx: Optional[int]) -> torch.Tensor:
        if not (self.enable_internal_sub_processing and x_q.shape[-1] == 4 and self.w_modulator and self.v_scaler_param is not None):
            return x_q
        print_tensor_stats(x_q, "ISP_In_x_q", micro_level_idx, stage_idx)
        w, v = x_q[..., 0:1], x_q[..., 1:4]
        w_modulation_factor = self.w_modulator(w)
        v_scaled = v * w_modulation_factor * (torch.tanh(self.v_scaler_param) + 1.0)
        x_q_processed = torch.cat([w, v_scaled], dim=-1)
        x_q_norm = normalize_quaternion(x_q_processed)
        if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 10 == 0:
            print_tensor_stats(w_modulation_factor, "ISP_w_mod", micro_level_idx, stage_idx)
            print_tensor_stats(x_q_norm, "ISP_Out_Norm", micro_level_idx, stage_idx)
        return x_q_norm

    def apply_rotation(self, x_tan: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        R_matrix_this_step: Optional[torch.Tensor] = None
        if self.dim <= 0: return x_tan, None
        if self.use_quaternion_so4 and self.rotation_param_generator is not None:
            p_quat, q_quat = self.rotation_param_generator()
            p_quat, q_quat = p_quat.to(x_tan.device, x_tan.dtype), q_quat.to(x_tan.device, x_tan.dtype)
            if x_tan.shape[-1] != 4: logger_wubu_cpu_gen.error_once(f"Quat rot needs dim 4, got {x_tan.shape[-1]}. Skip."); return x_tan, None
            p_b, q_b = (p_quat.unsqueeze(0).expand_as(x_tan), q_quat.unsqueeze(0).expand_as(x_tan)) if x_tan.dim() > 1 and p_quat.dim() == 1 else (p_quat, q_quat)
            rotated_x_tan = quaternion_multiply(quaternion_multiply(p_b, x_tan), q_b)
        elif self.rotation_generator_matrix is not None and isinstance(self.rotation_generator_matrix, SkewSymmetricMatrix):
            R_matrix_this_step = self.rotation_generator_matrix().to(x_tan.device, x_tan.dtype)
            if x_tan.dim() == 2: rotated_x_tan = torch.einsum('ij,bj->bi', R_matrix_this_step, x_tan)
            elif x_tan.dim() == 1: rotated_x_tan = torch.matmul(R_matrix_this_step, x_tan)
            elif x_tan.dim() == 3: rotated_x_tan = torch.einsum('ij,bnj->bni', R_matrix_this_step, x_tan)
            else: rotated_x_tan = x_tan; logger_wubu_cpu_gen.warning_once(f"MicroTrans apply_rot: Unexpected x_tan dim {x_tan.dim()}. Skip.")
        elif isinstance(self.rotation_generator_matrix, nn.Identity): rotated_x_tan = x_tan
        else: rotated_x_tan = x_tan; logger_wubu_cpu_gen.error_once("Rot module misconfigured.")
        return rotated_x_tan, R_matrix_this_step

    def _apply_non_rotational_map(self, x_in: torch.Tensor, original_main_tan_for_hypernet: torch.Tensor, micro_level_idx: Optional[int], stage_idx: Optional[int]) -> torch.Tensor:
        if not self.non_rotational_map_layers or isinstance(self.non_rotational_map_layers[0], nn.Identity):
            return x_in
        x_processed = x_in
        first_linear_layer = self.non_rotational_map_layers[0]
        if self.enable_hypernetwork_modulation and self.hyper_bias_generator is not None and isinstance(first_linear_layer, nn.Linear):
            dynamic_bias_offset = self.hyper_bias_generator(original_main_tan_for_hypernet)
            if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 10 == 0:
                print_tensor_stats(dynamic_bias_offset, "Hyper_BiasOffset", micro_level_idx, stage_idx)
            effective_bias = (first_linear_layer.bias + dynamic_bias_offset * self.hyper_mod_strength) if first_linear_layer.bias is not None else (dynamic_bias_offset * self.hyper_mod_strength)
            x_processed = F.linear(x_processed, first_linear_layer.weight, effective_bias)
            for i in range(1, len(self.non_rotational_map_layers)): x_processed = self.non_rotational_map_layers[i](x_processed)
        else:
            temp_x = x_in;
            for layer in self.non_rotational_map_layers: temp_x = layer(temp_x)
            x_processed = temp_x
        return x_processed

    def forward(self, main_tan: torch.Tensor, current_internal_state_in: Optional[torch.Tensor]=None, scaffold_modulation: Optional[torch.Tensor]=None, micro_level_idx: Optional[int]=None, stage_idx: Optional[int]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.debug_counter +=1
        if self.dim <= 0: return main_tan, None, current_internal_state_in
        print_tensor_stats(main_tan, "MF_In_main", micro_level_idx, stage_idx)
        
        x_for_hypernet_mod = main_tan # Original input to stage for hypernet modulation
        
        x_intermediate = self._apply_internal_quaternion_sub_processing(main_tan, micro_level_idx, stage_idx) if self.enable_internal_sub_processing else main_tan
        print_tensor_stats(x_intermediate, "MF_PostISP", micro_level_idx, stage_idx)
        
        rotated_x, R_matrix = self.apply_rotation(x_intermediate)
        print_tensor_stats(rotated_x, "MF_PostRot", micro_level_idx, stage_idx)
        if R_matrix is not None: print_tensor_stats(R_matrix, "MF_R_matrix", micro_level_idx, stage_idx)
        
        mapped_x = self._apply_non_rotational_map(rotated_x, x_for_hypernet_mod, micro_level_idx, stage_idx)
        print_tensor_stats(mapped_x, "MF_PostMap", micro_level_idx, stage_idx)
        
        next_state, final_tan_before_scaffold = current_internal_state_in, mapped_x
        if self.enable_stateful_micro_transform and current_internal_state_in is not None and self.state_influence_mlp and self.state_update_gate_mlp:
            print_tensor_stats(current_internal_state_in, "STF_In_State", micro_level_idx, stage_idx)
            influence = self.state_influence_mlp(current_internal_state_in)
            print_tensor_stats(influence, "STF_Influence", micro_level_idx, stage_idx)
            final_tan_before_scaffold = mapped_x + influence
            
            state_update_in = torch.cat([final_tan_before_scaffold, current_internal_state_in], dim=-1)
            delta = self.state_update_gate_mlp(state_update_in)
            print_tensor_stats(delta, "STF_Delta", micro_level_idx, stage_idx)
            next_state = torch.tanh(current_internal_state_in + delta)
            print_tensor_stats(next_state, "STF_Out_State", micro_level_idx, stage_idx)

        final_tan = final_tan_before_scaffold # Start with (potentially state-influenced) tangent vector
        if scaffold_modulation is not None:
            mod = scaffold_modulation
            if final_tan.dim()>1 and mod.dim()==1 and self.dim>0: mod=mod.unsqueeze(0).expand_as(final_tan)
            elif final_tan.dim()==1 and mod.dim()==2 and mod.shape[0]==1 and self.dim>0: mod=mod.squeeze(0)
            if final_tan.shape==mod.shape: final_tan += mod
            else: logger_wubu_cpu_gen.debug_once("Scaffold mod shape mismatch.") # logger changed
        print_tensor_stats(final_tan, "MF_PostScaff", micro_level_idx, stage_idx)
        
        clamped_tan = torch.clamp(final_tan, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        print_tensor_stats(clamped_tan, "MF_Out_Clamp", micro_level_idx, stage_idx)
        if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 20 == 0: print("-" * 20)
        return clamped_tan, R_matrix, next_state

class FractalDepthQuaternionWuBuRungs(nn.Module): # From WuBu_Integrated_Full (v...08)
    def __init__(self,
                 input_dim: int, output_dim: int,
                 num_virtual_micro_levels: int = 10000,
                 base_micro_level_dim: int = 4,
                 num_physical_transform_stages: int = 10,
                 initial_s: float = 1.0,
                 s_decay_factor_per_micro_level: float = 0.9999,
                 initial_c_base: float = 1.0, c_phi_influence: bool = True,
                 num_phi_scaffold_points_per_stage: int = 5,
                 phi_scaffold_init_scale_factor: float = 0.05,
                 use_gaussian_rungs: bool = True,
                 base_gaussian_std_dev_factor_rung: float = 0.02,
                 gaussian_std_dev_decay_factor_rung: float = 0.99999,
                 rung_affinity_temperature: float = 0.1,
                 rung_modulation_strength: float = 0.05,
                 t_tilde_activation_scale: float = 1.0,
                 micro_transform_type: str = "mlp",
                 micro_transform_hidden_factor: float = 1.0,
                 use_quaternion_so4_micro: bool = True,
                 scaffold_co_rotation_mode: str = "none", # Changed default from matrix_only
                 # New args for advanced features
                 enable_internal_sub_processing: bool = True,
                 enable_stateful_micro_transform: bool = True,
                 enable_hypernetwork_modulation: bool = True,
                 internal_state_dim_factor: float = 1.0,
                 hyper_mod_strength: float = 0.01
                 ):
        super().__init__()
        self.logger_fractal_rungs = logger_wubu_cpu_gen # Use local logger
        self.num_virtual_micro_levels = num_virtual_micro_levels
        self.base_micro_level_dim = base_micro_level_dim
        self.enable_stateful_micro_transform_globally = enable_stateful_micro_transform

        if self.base_micro_level_dim < 0: self.base_micro_level_dim = 0
        if input_dim < 0: input_dim = 0
        if output_dim < 0: output_dim = 0

        if self.base_micro_level_dim != 4 and use_quaternion_so4_micro:
            self.logger_fractal_rungs.warning("use_quaternion_so4_micro=True but base_dim!=4. Forced off.")
            use_quaternion_so4_micro = False
        if self.base_micro_level_dim != 4 and enable_internal_sub_processing:
            self.logger_fractal_rungs.warning("enable_internal_sub_processing=True but base_dim!=4. Forced off.")
            enable_internal_sub_processing = False

        self.num_physical_transform_stages = max(1, num_physical_transform_stages)
        if self.num_virtual_micro_levels > 0 and self.num_physical_transform_stages > 0:
             self.micro_levels_per_stage = max(1, self.num_virtual_micro_levels // self.num_physical_transform_stages)
        else: self.micro_levels_per_stage = 1

        self.initial_s, self.s_decay_factor = initial_s, s_decay_factor_per_micro_level
        self.initial_c_base, self.c_phi_influence = initial_c_base, c_phi_influence
        self.min_curvature, self.t_tilde_activation_scale = EPS, t_tilde_activation_scale
        self.scaffold_co_rotation_mode = scaffold_co_rotation_mode

        if use_quaternion_so4_micro and scaffold_co_rotation_mode == "matrix_only":
            self.logger_fractal_rungs.warning_once("Scaffold co-rotation 'matrix_only' may not work with quat micro-transforms if they don't output R_matrix.")

        if input_dim > 0 and self.base_micro_level_dim > 0:
            self.input_projection = nn.Linear(input_dim, self.base_micro_level_dim) if input_dim != self.base_micro_level_dim else nn.Identity()
            self.input_layernorm = nn.LayerNorm(self.base_micro_level_dim)
        elif input_dim == 0 and self.base_micro_level_dim > 0:
            self.input_projection = nn.Linear(1,self.base_micro_level_dim) # Process dummy 1 to get to base_dim
            self.input_layernorm = nn.LayerNorm(self.base_micro_level_dim)
        else: # input_dim=0, base_micro_level_dim=0 OR input_dim > 0, base_micro_level_dim = 0
            self.input_projection = nn.Identity()
            self.input_layernorm = nn.Identity()


        if self.num_physical_transform_stages > 0 and self.base_micro_level_dim > 0:
            self.physical_micro_transforms = nn.ModuleList(
                [FractalMicroTransformRungs(dim=self.base_micro_level_dim,
                                            transform_type=micro_transform_type,
                                            hidden_dim_factor=micro_transform_hidden_factor,
                                            use_quaternion_so4=use_quaternion_so4_micro,
                                            enable_internal_sub_processing=enable_internal_sub_processing,
                                            enable_stateful_micro_transform=enable_stateful_micro_transform,
                                            enable_hypernetwork_modulation=enable_hypernetwork_modulation,
                                            internal_state_dim_factor=internal_state_dim_factor,
                                            hyper_mod_strength=hyper_mod_strength)
                 for _ in range(self.num_physical_transform_stages)]
            )
            safe_initial_c_base = max(EPS*10, initial_c_base)
            self.log_stage_curvatures_unconstrained = nn.ParameterList(
                [nn.Parameter(torch.tensor(math.log(math.expm1(safe_initial_c_base))))
                 for _ in range(self.num_physical_transform_stages)]
            )
        else:
            self.physical_micro_transforms = nn.ModuleList()
            self.log_stage_curvatures_unconstrained = nn.ParameterList()


        self.num_phi_scaffold_points_per_stage = num_phi_scaffold_points_per_stage
        if self.num_phi_scaffold_points_per_stage > 0 and self.base_micro_level_dim > 0 and self.num_physical_transform_stages > 0:
            self.phi_scaffold_base_tangent_vectors = nn.ParameterList([
                nn.Parameter(torch.randn(num_phi_scaffold_points_per_stage, self.base_micro_level_dim) * phi_scaffold_init_scale_factor)
                for _ in range(self.num_physical_transform_stages)])
        else: self.phi_scaffold_base_tangent_vectors = None
        self.use_gaussian_rungs = use_gaussian_rungs
        self.base_gaussian_std_dev_factor_rung, self.gaussian_std_dev_decay_factor_rung = base_gaussian_std_dev_factor_rung, gaussian_std_dev_decay_factor_rung
        self.rung_affinity_temperature, self.rung_modulation_strength = max(EPS, rung_affinity_temperature), rung_modulation_strength

        if self.base_micro_level_dim > 0 and output_dim > 0 :
            self.output_projection = nn.Linear(self.base_micro_level_dim, output_dim) if self.base_micro_level_dim != output_dim else nn.Identity()
        elif self.base_micro_level_dim == 0 and output_dim > 0:
             self.output_projection = nn.Linear(1, output_dim) # If base_dim is 0, assume output is derived from a dummy 1
        else: # output_dim = 0 or base_micro_level_dim = 0 and output_dim = 0
            self.output_projection = nn.Identity()

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        in_feat_proj = self.input_projection.in_features if isinstance(self.input_projection, nn.Linear) else input_dim
        out_feat_proj = self.output_projection.out_features if isinstance(self.output_projection, nn.Linear) else output_dim

        self.logger_fractal_rungs.info(f"FDQRungs Init: In {in_feat_proj}D -> {self.num_virtual_micro_levels} virtLvl ({self.base_micro_level_dim}D) "
                                 f"over {self.num_physical_transform_stages} physStgs -> Out {out_feat_proj}D. Params: {param_count:,}. "
                                 f"AdvMicro: SubP={enable_internal_sub_processing}, StateF={self.enable_stateful_micro_transform_globally}, HyperM={enable_hypernetwork_modulation}")

    def get_s_c_gsigma_at_micro_level(self, micro_level_idx: int, stage_idx: int) -> Tuple[float, float, float]:
        s_i = self.initial_s * (self.s_decay_factor ** micro_level_idx); s_i = max(EPS, s_i)
        c_i = self.initial_c_base
        if self.log_stage_curvatures_unconstrained and stage_idx < len(self.log_stage_curvatures_unconstrained):
            stage_c_unconstrained = self.log_stage_curvatures_unconstrained[stage_idx]
            c_i = get_constrained_param_val(stage_c_unconstrained, self.min_curvature).item()
            if self.c_phi_influence and self.micro_levels_per_stage > 0 :
                micro_idx_in_stage = micro_level_idx % self.micro_levels_per_stage
                phi_exp = (micro_idx_in_stage % 4) - 1.5
                c_i *= (PHI ** phi_exp)
        c_i = max(self.min_curvature, c_i)
        sigma_gauss_i_skin = (self.base_gaussian_std_dev_factor_rung / max(s_i, EPS)) * (self.gaussian_std_dev_decay_factor_rung ** micro_level_idx)
        sigma_gauss_i_skin = max(EPS*100, sigma_gauss_i_skin)
        return s_i, c_i, sigma_gauss_i_skin

    def _propagate_scaffold_points(self, base_scaffold_tan_vectors_stage: torch.Tensor,
                                   accumulated_R_matrix_stage: Optional[torch.Tensor],
                                   current_s_i: float, initial_s_for_stage: float) -> torch.Tensor:
        propagated_scaffold = base_scaffold_tan_vectors_stage
        if accumulated_R_matrix_stage is not None and self.scaffold_co_rotation_mode == "matrix_only":
            propagated_scaffold = torch.einsum('ij,kj->ki', accumulated_R_matrix_stage.to(propagated_scaffold.device, propagated_scaffold.dtype), propagated_scaffold)
        return propagated_scaffold * (initial_s_for_stage / max(current_s_i, EPS))

    def forward(self, x_input: torch.Tensor, show_progress: bool = False, progress_desc: Optional[str] = None) -> torch.Tensor:
        input_original_dim_fwd = x_input.dim()
        B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = -1, -1, -1
        current_batch_size: int

        if input_original_dim_fwd == 3:
            B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = x_input.shape
            current_v_tan_fwd = x_input.reshape(B_orig_fwd * S_orig_fwd, d_in_runtime_fwd)
            current_batch_size = B_orig_fwd * S_orig_fwd
        elif input_original_dim_fwd == 2:
            current_batch_size, d_in_runtime_fwd = x_input.shape
            current_v_tan_fwd = x_input
            B_orig_fwd, S_orig_fwd = current_batch_size, 1
        elif input_original_dim_fwd == 1 :
            current_v_tan_fwd = x_input.unsqueeze(0)
            d_in_runtime_fwd = x_input.shape[0]
            B_orig_fwd, S_orig_fwd, current_batch_size = 1,1,1
        else: raise ValueError(f"FDQRungs forward expects 1D, 2D or 3D input, got {input_original_dim_fwd}D.")
        
        if d_in_runtime_fwd == 0 and self.base_micro_level_dim > 0 and isinstance(self.input_projection, nn.Linear) and self.input_projection.in_features == 1:
            # Handle 0-dim input to Linear(1, base_dim) by providing a dummy 1
            current_v_tan_fwd = torch.ones(current_v_tan_fwd.shape[0], 1, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)

        current_v_tan_fwd = self.input_projection(current_v_tan_fwd)
        current_v_tan_fwd = self.input_layernorm(current_v_tan_fwd)

        if self.num_virtual_micro_levels == 0 or not self.physical_micro_transforms: # No processing if no levels or no transforms
            final_v_tan_fwd = current_v_tan_fwd
        else:
            accumulated_R_for_stage: Optional[torch.Tensor] = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype) if self.base_micro_level_dim > 0 else None
            s_at_stage_start_val: float = self.initial_s
            batched_internal_micro_state: Optional[torch.Tensor] = None
            
            if self.enable_stateful_micro_transform_globally and self.base_micro_level_dim > 0 and len(self.physical_micro_transforms) > 0:
                first_micro_transform = self.physical_micro_transforms[0]
                if hasattr(first_micro_transform, 'enable_stateful_micro_transform') and \
                   first_micro_transform.enable_stateful_micro_transform and first_micro_transform.internal_state_dim > 0:
                    batched_internal_micro_state = torch.zeros(current_batch_size, first_micro_transform.internal_state_dim,
                                                               device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)

            micro_level_iterator = range(self.num_virtual_micro_levels)
            tqdm_iterator = None
            in_feat_display = self.input_projection.in_features if hasattr(self.input_projection, 'in_features') and isinstance(self.input_projection, nn.Linear) else d_in_runtime_fwd
            out_feat_display = self.output_projection.out_features if hasattr(self.output_projection, 'out_features') and isinstance(self.output_projection, nn.Linear) else 'N/A'
            effective_progress_desc = progress_desc if progress_desc else f"FDQRungs ({in_feat_display}->{out_feat_display})"
            
            if show_progress:
                try: from tqdm import tqdm; tqdm_iterator = tqdm(micro_level_iterator, desc=effective_progress_desc, total=self.num_virtual_micro_levels, leave=False, dynamic_ncols=True, disable=not show_progress)
                except ImportError: self.logger_fractal_rungs.warning_once("TQDM not found. Progress bar disabled.")
                if tqdm_iterator: micro_level_iterator = tqdm_iterator

            for micro_i_fwd in micro_level_iterator:
                current_stage_idx_fwd = micro_i_fwd // self.micro_levels_per_stage
                current_stage_idx_fwd = min(current_stage_idx_fwd, len(self.physical_micro_transforms)-1) # Ensure valid index
                micro_idx_in_stage_fwd = micro_i_fwd % self.micro_levels_per_stage
                physical_transform_module_fwd = self.physical_micro_transforms[current_stage_idx_fwd]

                if micro_idx_in_stage_fwd == 0:
                    if self.base_micro_level_dim > 0: accumulated_R_for_stage = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
                    s_at_stage_start_val, _, _ = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)
                    if self.enable_stateful_micro_transform_globally and self.base_micro_level_dim > 0 and \
                       hasattr(physical_transform_module_fwd, 'enable_stateful_micro_transform') and \
                       physical_transform_module_fwd.enable_stateful_micro_transform and physical_transform_module_fwd.internal_state_dim > 0:
                        batched_internal_micro_state = torch.zeros(current_batch_size, physical_transform_module_fwd.internal_state_dim,
                                                                   device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)

                s_i_fwd, _c_i_fwd, sigma_gauss_skin_i_fwd = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)
                scaffold_modulation_input: Optional[torch.Tensor] = None
                if self.use_gaussian_rungs and self.phi_scaffold_base_tangent_vectors is not None and \
                   self.num_phi_scaffold_points_per_stage > 0 and self.base_micro_level_dim > 0 and \
                   current_stage_idx_fwd < len(self.phi_scaffold_base_tangent_vectors):
                    base_scaffolds = self.phi_scaffold_base_tangent_vectors[current_stage_idx_fwd].to(current_v_tan_fwd.device, current_v_tan_fwd.dtype)
                    R_prop = accumulated_R_for_stage if self.scaffold_co_rotation_mode == "matrix_only" else None
                    propagated_scaffolds = self._propagate_scaffold_points(base_scaffolds, R_prop, s_i_fwd, s_at_stage_start_val)
                    diffs = current_v_tan_fwd.unsqueeze(1) - propagated_scaffolds.unsqueeze(0)
                    sq_dists = torch.sum(diffs**2, dim=-1)
                    affinity_scores = torch.exp(-sq_dists / (2 * (sigma_gauss_skin_i_fwd**2) + EPS*100))
                    affinity_weights = F.softmax(affinity_scores / self.rung_affinity_temperature, dim=-1)
                    scaffold_modulation_input = self.rung_modulation_strength * torch.einsum('bn,nd->bd', affinity_weights, propagated_scaffolds)

                transformed_v_tan_micro, R_step, next_state = physical_transform_module_fwd(
                    current_v_tan_fwd, batched_internal_micro_state, scaffold_modulation_input, micro_i_fwd, current_stage_idx_fwd
                )
                if self.enable_stateful_micro_transform_globally: batched_internal_micro_state = next_state
                current_v_tan_fwd = transformed_v_tan_micro * self.t_tilde_activation_scale if self.t_tilde_activation_scale != 1.0 else transformed_v_tan_micro
                if R_step is not None and self.scaffold_co_rotation_mode == "matrix_only" and accumulated_R_for_stage is not None:
                     accumulated_R_for_stage = torch.matmul(R_step, accumulated_R_for_stage)

                if tqdm_iterator and micro_i_fwd > 0 and (micro_i_fwd % (max(1,self.num_virtual_micro_levels//100))==0):
                    log_dict = {"s":f"{s_i_fwd:.2e}","c":f"{_c_i_fwd:.2e}","σ":f"{sigma_gauss_skin_i_fwd:.2e}"}
                    if scaffold_modulation_input is not None and 'affinity_weights' in locals() and affinity_weights.numel()>0 : log_dict["aff_w"] = f"{affinity_weights.max().item():.2e}"
                    if batched_internal_micro_state is not None and batched_internal_micro_state.numel()>0: log_dict["st_n"] = f"{torch.linalg.norm(batched_internal_micro_state).item():.2e}"
                    tqdm_iterator.set_postfix(log_dict)
            final_v_tan_fwd = current_v_tan_fwd
            if tqdm_iterator: tqdm_iterator.close()

        if self.base_micro_level_dim == 0 and output_dim > 0 and isinstance(self.output_projection, nn.Linear) and self.output_projection.in_features == 1:
             output_features_fwd = self.output_projection(torch.ones(final_v_tan_fwd.shape[0], 1, device=final_v_tan_fwd.device, dtype=final_v_tan_fwd.dtype))
        else:
            output_features_fwd = self.output_projection(final_v_tan_fwd)


        if input_original_dim_fwd == 3 and B_orig_fwd != -1 and S_orig_fwd != -1:
            final_output_dim_val = output_features_fwd.shape[-1] if output_features_fwd.numel() > 0 else 0
            return output_features_fwd.reshape(B_orig_fwd, S_orig_fwd, final_output_dim_val)
        elif input_original_dim_fwd == 2 :
            return output_features_fwd
        elif input_original_dim_fwd == 1:
            return output_features_fwd.squeeze(0) if output_features_fwd.shape[0] == 1 else output_features_fwd
        return output_features_fwd


# --- GAAD and Image Processing Utilities ---
def golden_subdivide_rect_fixed_n_cpu(frame_dims:Tuple[int,int], num_regions_target:int, dtype=torch.float, min_size_px=5) -> torch.Tensor:
    W, H = frame_dims; device = torch.device('cpu')
    all_rects = [[0.0,0.0,float(W),float(H)]]; rect_queue = deque([(0.0,0.0,float(W),float(H),0)])
    while rect_queue and len(all_rects) < num_regions_target * 3 :
        x_off, y_off, w_curr, h_curr, depth = rect_queue.popleft()
        if min(w_curr, h_curr) < min_size_px or depth > 7 : continue
        is_landscape = w_curr > h_curr + EPS; is_portrait = h_curr > w_curr + EPS
        children_coords = []
        if is_landscape:
            cut_w = w_curr / PHI; r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px: children_coords.append({'x':x_off, 'y':y_off, 'w':r1_w, 'h':h_curr})
            if r2_w >= min_size_px: children_coords.append({'x':x_off + r1_w, 'y':y_off, 'w':r2_w, 'h':h_curr})
        elif is_portrait:
            cut_h = h_curr / PHI; r1_h, r2_h = cut_h, h_curr - cut_h
            if r1_h >= min_size_px: children_coords.append({'x':x_off, 'y':y_off, 'w':w_curr, 'h':r1_h})
            if r2_h >= min_size_px: children_coords.append({'x':x_off, 'y':y_off + r1_h, 'w':w_curr, 'h':r2_h})
        elif abs(w_curr - h_curr) < EPS and w_curr > min_size_px * PHI :
            cut_w = w_curr / PHI; r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px: children_coords.append({'x':x_off, 'y':y_off, 'w':r1_w, 'h':h_curr})
            if r2_w >= min_size_px: children_coords.append({'x':x_off + r1_w, 'y':y_off, 'w':r2_w, 'h':h_curr})
        for child_d in children_coords:
            all_rects.append([child_d['x'],child_d['y'],child_d['x']+child_d['w'],child_d['y']+child_d['h']])
            rect_queue.append((child_d['x'],child_d['y'],child_d['w'],child_d['h'],depth+1))
    unique_valid_rects_tensors = []; seen_hashes = set()
    for r_coords in all_rects:
        if r_coords[0] >= r_coords[2] - EPS or r_coords[1] >= r_coords[3] - EPS: continue
        r_tensor = torch.tensor(r_coords, dtype=dtype, device=device);
        r_hashable = tuple(round(c, 2) for c in r_coords)
        if r_hashable not in seen_hashes: unique_valid_rects_tensors.append(r_tensor); seen_hashes.add(r_hashable)
    unique_valid_rects_tensors.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
    selected_rects = unique_valid_rects_tensors[:num_regions_target]
    if not selected_rects and num_regions_target > 0: selected_rects = [torch.tensor([0.0,0.0,float(W),float(H)],dtype=dtype,device=device)]
    if len(selected_rects) < num_regions_target:
        padding_box = selected_rects[-1].clone() if selected_rects else torch.tensor([0.0,0.0,float(W),float(H)],dtype=dtype,device=device)
        selected_rects.extend([padding_box.clone() for _ in range(num_regions_target - len(selected_rects))])
    return torch.stack(selected_rects)

class RegionalPatchExtractorCPU(nn.Module):
    def __init__(self, patch_output_size: Tuple[int, int]):
        super().__init__()
        self.patch_output_size = patch_output_size
        if patch_output_size[0] > 0 and patch_output_size[1] > 0:
            self.resize_transform = T.Resize(self.patch_output_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        else: self.resize_transform = None
    def forward(self, images: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
        B_img, NumCh, H_img, W_img = images.shape; device, dtype = images.device, images.dtype
        all_patches = []
        patch_h_out, patch_w_out = self.patch_output_size
        for i in range(B_img):
            img, bboxes = images[i], bboxes_batch[i]; current_img_patches = []
            for r in range(bboxes.shape[0]):
                x1,y1,x2,y2 = bboxes[r].round().int().tolist()
                x1c,y1c,x2c,y2c = max(0,x1),max(0,y1),min(W_img,x2),min(H_img,y2)
                if x1c>=x2c or y1c>=y2c or patch_h_out<=0 or patch_w_out<=0: patch = torch.zeros((NumCh,max(1,patch_h_out),max(1,patch_w_out)),device=device,dtype=dtype)
                else:
                    patch = img[:,y1c:y2c,x1c:x2c]
                    if self.resize_transform: patch=self.resize_transform(patch)
                    elif patch.shape[1]!=patch_h_out or patch.shape[2]!=patch_w_out: patch=torch.zeros((NumCh,patch_h_out,patch_w_out),device=device,dtype=dtype)
                current_img_patches.append(patch)
            all_patches.append(torch.stack(current_img_patches))
        return torch.stack(all_patches)

class ImageAssemblyUtilsCPU:
    @staticmethod
    def assemble_frames_from_patches(patches_batch:torch.Tensor, bboxes_batch:torch.Tensor, target_img_size:Tuple[int,int], out_range:Tuple[float,float]=(-1.0,1.0)) -> torch.Tensor:
        B,N_f,N_r,C,H_patch,W_patch = patches_batch.shape; H_img,W_img = target_img_size
        device,dtype = patches_batch.device, patches_batch.dtype
        all_frames = torch.zeros(B,N_f,C,H_img,W_img,device=device,dtype=dtype)
        for b in range(B):
            for f in range(N_f):
                canvas, count_map = torch.zeros(C,H_img,W_img,device=device,dtype=dtype), torch.zeros(1,H_img,W_img,device=device,dtype=dtype)
                for r in range(N_r):
                    patch = patches_batch[b,f,r]; x1,y1,x2,y2 = bboxes_batch[b,f,r].float().round().int().tolist()
                    x1c,y1c,x2c,y2c = max(0,x1),max(0,y1),min(W_img,x2),min(H_img,y2)
                    if x1c>=x2c or y1c>=y2c: continue
                    th,tw = y2c-y1c, x2c-x1c
                    if th<=0 or tw<=0: continue
                    resized_patch = TF.resize(patch,[th,tw],antialias=True) if (H_patch!=th or W_patch!=tw) else patch
                    canvas[:,y1c:y2c,x1c:x2c]+=resized_patch; count_map[:,y1c:y2c,x1c:x2c]+=1.0
                all_frames[b,f] = torch.where(count_map>0,canvas/(count_map+EPS),canvas)
        return torch.clamp(all_frames,min=out_range[0],max=out_range[1]) if out_range else all_frames


# --- VAE-GAN Components ---
class WuBuCPUEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict):
        super().__init__()
        self.args = args
        self.patch_h, self.patch_w = args.patch_size_h, args.patch_size_w
        self.num_channels, self.num_regions, self.latent_dim = video_config['num_channels'], gaad_config['num_regions'], args.latent_dim
        self.patch_extractor = RegionalPatchExtractorCPU(patch_output_size=(self.patch_h, self.patch_w))
        patch_feat_flat = self.num_regions * self.num_channels * self.patch_h * self.patch_w
        total_frames_sample = args.num_input_frames + args.num_predict_frames
        enc_in_dim = total_frames_sample * patch_feat_flat if total_frames_sample * patch_feat_flat > 0 else 1

        self.wubu_encoder_stack = FractalDepthQuaternionWuBuRungs(
            input_dim=enc_in_dim, output_dim=max(1, self.latent_dim * 2),
            num_virtual_micro_levels=args.enc_wubu_num_virtual_levels, base_micro_level_dim=args.enc_wubu_base_micro_dim,
            num_physical_transform_stages=args.enc_wubu_num_physical_stages,
            initial_s=args.wubu_initial_s, s_decay_factor_per_micro_level=args.wubu_s_decay,
            initial_c_base=args.wubu_initial_c, c_phi_influence=args.wubu_c_phi_influence,
            num_phi_scaffold_points_per_stage=args.wubu_num_phi_scaffold_points, phi_scaffold_init_scale_factor=args.wubu_phi_scaffold_init_scale,
            use_gaussian_rungs=args.wubu_use_gaussian_rungs, base_gaussian_std_dev_factor_rung=args.wubu_base_gaussian_std_dev,
            gaussian_std_dev_decay_factor_rung=args.wubu_gaussian_std_dev_decay, rung_affinity_temperature=args.wubu_rung_affinity_temp,
            rung_modulation_strength=args.wubu_rung_modulation_strength, t_tilde_activation_scale=args.wubu_t_tilde_scale,
            micro_transform_type=args.wubu_micro_transform_type, micro_transform_hidden_factor=args.wubu_micro_transform_hidden_factor,
            use_quaternion_so4_micro=args.wubu_use_quaternion_so4, scaffold_co_rotation_mode=args.wubu_scaffold_co_rotation_mode,
            enable_internal_sub_processing=args.wubu_enable_isp, enable_stateful_micro_transform=args.wubu_enable_stateful,
            enable_hypernetwork_modulation=args.wubu_enable_hypernet, internal_state_dim_factor=args.wubu_internal_state_factor,
            hyper_mod_strength=args.wubu_hypernet_strength
        )
        self.apply(init_weights_general)
        logger_wubu_cpu_gen.info(f"WuBuCPUEncoder: In {enc_in_dim}D -> WuBu -> {max(1, self.latent_dim*2)}D.")

    def forward(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        timings = {}
        B, N_total, C_img, H_img, W_img = frames_pixels.shape
        device, dtype = frames_pixels.device, frames_pixels.dtype

        t_bbox_start = time.time()
        all_bboxes_list = [[golden_subdivide_rect_fixed_n_cpu((W_img,H_img),self.num_regions,dtype,self.args.gaad_min_size_px) for _ in range(N_total)] for _ in range(B)]
        all_bboxes = torch.stack([torch.stack(seq_bboxes) for seq_bboxes in all_bboxes_list])
        t_bbox_end = time.time()
        timings["time_bbox_gen"] = t_bbox_end - t_bbox_start
        
        t_patch_ext_start = time.time()
        patches = self.patch_extractor(frames_pixels.reshape(B*N_total,C_img,H_img,W_img), all_bboxes.reshape(B*N_total,self.num_regions,4))
        t_patch_ext_end = time.time()
        timings["time_patch_extract"] = t_patch_ext_end - t_patch_ext_start

        wubu_in = patches.reshape(B, N_total * self.num_regions * self.num_channels * self.patch_h * self.patch_w)
        if wubu_in.shape[1] == 0 and isinstance(self.wubu_encoder_stack.input_projection, nn.Linear) and self.wubu_encoder_stack.input_projection.in_features == 1:
             wubu_in = torch.ones(B, 1, device=device, dtype=dtype)

        mu_logvar = self.wubu_encoder_stack(wubu_in, show_progress=getattr(self.args, 'show_train_progress_bar', False))
        eff_lat_dim = max(1,self.latent_dim)
        return mu_logvar[:,:eff_lat_dim], mu_logvar[:,eff_lat_dim:], all_bboxes, timings

class WuBuCPUGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict):
        super().__init__()
        self.args = args
        self.latent_dim, self.num_predict_frames = args.latent_dim, video_config['num_predict_frames']
        self.num_regions, self.num_channels = gaad_config['num_regions'], video_config['num_channels']
        self.patch_h, self.patch_w = args.patch_size_h, args.patch_size_w
        self.image_size = (args.image_h, args.image_w)
        gen_out_dim = self.num_predict_frames*self.num_regions*self.num_channels*self.patch_h*self.patch_w
        gen_out_dim = max(1, gen_out_dim) # Ensure not zero
        lat_in_dim = max(1,self.latent_dim)

        self.wubu_generator_stack = FractalDepthQuaternionWuBuRungs(
            input_dim=lat_in_dim, output_dim=gen_out_dim,
            num_virtual_micro_levels=args.gen_wubu_num_virtual_levels, base_micro_level_dim=args.gen_wubu_base_micro_dim,
            num_physical_transform_stages=args.gen_wubu_num_physical_stages,
            initial_s=args.wubu_initial_s, s_decay_factor_per_micro_level=args.wubu_s_decay,
            initial_c_base=args.wubu_initial_c, c_phi_influence=args.wubu_c_phi_influence,
            num_phi_scaffold_points_per_stage=args.wubu_num_phi_scaffold_points, phi_scaffold_init_scale_factor=args.wubu_phi_scaffold_init_scale,
            use_gaussian_rungs=args.wubu_use_gaussian_rungs, base_gaussian_std_dev_factor_rung=args.wubu_base_gaussian_std_dev,
            gaussian_std_dev_decay_factor_rung=args.wubu_gaussian_std_dev_decay, rung_affinity_temperature=args.wubu_rung_affinity_temp,
            rung_modulation_strength=args.wubu_rung_modulation_strength, t_tilde_activation_scale=args.wubu_t_tilde_scale,
            micro_transform_type=args.wubu_micro_transform_type, micro_transform_hidden_factor=args.wubu_micro_transform_hidden_factor,
            use_quaternion_so4_micro=args.wubu_use_quaternion_so4, scaffold_co_rotation_mode=args.wubu_scaffold_co_rotation_mode,
            enable_internal_sub_processing=args.wubu_enable_isp, enable_stateful_micro_transform=args.wubu_enable_stateful,
            enable_hypernetwork_modulation=args.wubu_enable_hypernet, internal_state_dim_factor=args.wubu_internal_state_factor,
            hyper_mod_strength=args.wubu_hypernet_strength
        )
        self.final_patch_activation = nn.Tanh()
        self.apply(init_weights_general)
        logger_wubu_cpu_gen.info(f"WuBuCPUGenerator: In z({lat_in_dim}D) -> WuBu -> {gen_out_dim}D (patches).")

    def forward(self, z: torch.Tensor, gaad_bboxes_for_decode: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        timings = {}
        if z.shape[1] == 0 and isinstance(self.wubu_generator_stack.input_projection, nn.Linear) and self.wubu_generator_stack.input_projection.in_features == 1:
             z = torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)
        
        flat_patches = self.wubu_generator_stack(z, show_progress=getattr(self.args, 'show_train_progress_bar', False))
        
        # Handle cases where any of these dimensions are zero, which could lead to zero expected elements
        if self.num_predict_frames == 0 or self.num_regions == 0 or self.num_channels == 0 or self.patch_h == 0 or self.patch_w == 0:
            timings["time_patch_assembly"] = 0.0 
            empty_frames = torch.empty((z.shape[0], self.num_predict_frames, self.num_channels, self.image_size[0], self.image_size[1]), device=z.device, dtype=z.dtype)
            if flat_patches.shape[1] == 0: # WuBu output dim was 0 or 1 and resulted in empty
                 return empty_frames, timings
            else: # Mismatch, output dim was >0 but some config dim was 0.
                logger_wubu_cpu_gen.error_once(f"Gen: Mismatch due to zero-dim in patch structure (F/R/C/H/W all must be >0 if output_dim>0). "
                                          f"F:{self.num_predict_frames},R:{self.num_regions},C:{self.num_channels},H:{self.patch_h},W:{self.patch_w}. Output dim: {flat_patches.shape[1]}")
                return empty_frames, timings

        target_shape = (z.shape[0], self.num_predict_frames, self.num_regions, self.num_channels, self.patch_h, self.patch_w)
        expected_elems = np.prod(target_shape[1:])
        if flat_patches.shape[1] != expected_elems :
            timings["time_patch_assembly"] = 0.0
            logger_wubu_cpu_gen.error_once(f"Gen: Output dim {flat_patches.shape[1]} != expected {expected_elems} for reshape. Target: {target_shape}")
            return torch.empty((z.shape[0], self.num_predict_frames, self.num_channels, self.image_size[0], self.image_size[1]), device=z.device, dtype=z.dtype), timings

        pred_patches = flat_patches.reshape(target_shape)
        pred_patches = self.final_patch_activation(pred_patches)
        
        t_patch_asm_start = time.time()
        assembled_frames = ImageAssemblyUtilsCPU.assemble_frames_from_patches(pred_patches, gaad_bboxes_for_decode, self.image_size)
        t_patch_asm_end = time.time()
        timings["time_patch_assembly"] = t_patch_asm_end - t_patch_asm_start
        
        return assembled_frames, timings

class WuBuCPUNet(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict):
        super().__init__()
        self.args = args
        self.encoder = WuBuCPUEncoder(args, video_config, gaad_config)
        self.generator = WuBuCPUGenerator(args, video_config, gaad_config)
        self.latent_dim = args.latent_dim

    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        # Ensure logvar is not too large to prevent exp from overflowing
        logvar = torch.clamp(logvar, max=80.0) # Clamping before exp
        std = torch.exp(0.5*logvar)
        eps_val = torch.randn_like(std)
        return mu + eps_val * std

    def forward(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        timings = {}

        t_enc_start = time.time()
        mu, logvar, all_bboxes, enc_timings = self.encoder(frames_pixels)
        t_enc_end = time.time()
        timings["time_encoder_fwd"] = t_enc_end - t_enc_start
        timings.update(enc_timings)
        
        z = self.reparameterize(mu, logvar)
        
        n_in, n_pred = self.args.num_input_frames, self.args.num_predict_frames
        
        if all_bboxes.shape[1] < n_in + n_pred:
            available_for_pred = max(0, all_bboxes.shape[1] - n_in)
            actual_pred_bboxes = min(available_for_pred, n_pred)
            
            if actual_pred_bboxes > 0:
                bboxes_for_gen_partial = all_bboxes[:, n_in : n_in + actual_pred_bboxes, ...]
            else: 
                bboxes_for_gen_partial = torch.empty((all_bboxes.shape[0], 0, all_bboxes.shape[2], all_bboxes.shape[3]), 
                                                     device=all_bboxes.device, dtype=all_bboxes.dtype)

            if actual_pred_bboxes < n_pred: 
                padding_needed = n_pred - actual_pred_bboxes
                if actual_pred_bboxes > 0 and bboxes_for_gen_partial.numel() > 0 : 
                    padding = bboxes_for_gen_partial[:, -1:, ...].repeat(1, padding_needed, 1, 1)
                elif all_bboxes.numel() > 0 and all_bboxes.shape[1] > 0: 
                    padding = all_bboxes[:, -1:, ...].repeat(1, padding_needed, 1, 1)
                else: 
                    B,_,R,_ = all_bboxes.shape if all_bboxes.numel() > 0 else (frames_pixels.shape[0], 0, self.args.gaad_num_regions, 0)
                    dev,dt=frames_pixels.device,frames_pixels.dtype 
                    W_img_eff = frames_pixels.shape[-1] if frames_pixels.numel() > 0 else self.args.image_w
                    H_img_eff = frames_pixels.shape[-2] if frames_pixels.numel() > 0 else self.args.image_h
                    dummy_ff = torch.tensor([0,0,W_img_eff,H_img_eff],device=dev,dtype=dt)
                    padding = dummy_ff.view(1,1,1,4).expand(B, padding_needed, R if R > 0 else self.args.gaad_num_regions, 4)
                
                if bboxes_for_gen_partial.numel() > 0:
                    bboxes_for_gen = torch.cat([bboxes_for_gen_partial, padding], dim=1)
                else:
                    bboxes_for_gen = padding
            else: 
                bboxes_for_gen = bboxes_for_gen_partial
        else: 
            bboxes_for_gen = all_bboxes[:, n_in : n_in + n_pred, ...]
            
        t_gen_start = time.time()
        reconstructed_frames, gen_timings = self.generator(z, bboxes_for_gen)
        t_gen_end = time.time()
        timings["time_generator_fwd"] = t_gen_end - t_gen_start
        timings.update(gen_timings)
        
        return reconstructed_frames, mu, logvar, bboxes_for_gen, timings


class WuBuCPUDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict):
        super().__init__()
        self.args = args
        self.num_disc_frames = video_config['num_predict_frames'] # Discriminate predicted frames
        self.num_channels, self.img_h, self.img_w = video_config['num_channels'], args.image_h, args.image_w
        disc_in_dim = self.num_disc_frames*self.num_channels*self.img_h*self.img_w
        disc_in_dim = max(1, disc_in_dim)

        self.wubu_discriminator_stack = FractalDepthQuaternionWuBuRungs(
            input_dim=disc_in_dim, output_dim=1,
            num_virtual_micro_levels=args.disc_wubu_num_virtual_levels, base_micro_level_dim=args.disc_wubu_base_micro_dim,
            num_physical_transform_stages=args.disc_wubu_num_physical_stages,
            initial_s=args.wubu_initial_s, s_decay_factor_per_micro_level=args.wubu_s_decay,
            initial_c_base=args.wubu_initial_c, c_phi_influence=args.wubu_c_phi_influence,
            num_phi_scaffold_points_per_stage=args.wubu_num_phi_scaffold_points, phi_scaffold_init_scale_factor=args.wubu_phi_scaffold_init_scale,
            use_gaussian_rungs=args.wubu_use_gaussian_rungs, base_gaussian_std_dev_factor_rung=args.wubu_base_gaussian_std_dev,
            gaussian_std_dev_decay_factor_rung=args.wubu_gaussian_std_dev_decay, rung_affinity_temperature=args.wubu_rung_affinity_temp,
            rung_modulation_strength=args.wubu_rung_modulation_strength, t_tilde_activation_scale=args.wubu_t_tilde_scale,
            micro_transform_type=args.wubu_micro_transform_type, micro_transform_hidden_factor=args.wubu_micro_transform_hidden_factor,
            use_quaternion_so4_micro=args.wubu_use_quaternion_so4, scaffold_co_rotation_mode=args.wubu_scaffold_co_rotation_mode,
            enable_internal_sub_processing=args.wubu_enable_isp, enable_stateful_micro_transform=args.wubu_enable_stateful,
            enable_hypernetwork_modulation=args.wubu_enable_hypernet, internal_state_dim_factor=args.wubu_internal_state_factor,
            hyper_mod_strength=args.wubu_hypernet_strength
        )
        self.apply(init_weights_general)
        logger_wubu_cpu_gen.info(f"WuBuCPUDiscriminator: In {disc_in_dim}D -> WuBu -> 1D logit.")

    def forward(self, frames_pixels: torch.Tensor, gaad_bboxes_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = frames_pixels.shape[0]
        # Ensure correct number of frames for discriminator input
        if frames_pixels.shape[1] < self.num_disc_frames:
            pad_needed = self.num_disc_frames - frames_pixels.shape[1]
            padding = frames_pixels[:,-1:,...].repeat(1,pad_needed,1,1,1) if frames_pixels.shape[1]>0 else \
                      torch.zeros(B,pad_needed,self.num_channels,self.img_h,self.img_w,device=frames_pixels.device,dtype=frames_pixels.dtype)
            frames_proc = torch.cat([frames_pixels, padding], dim=1)
        elif frames_pixels.shape[1] > self.num_disc_frames:
            frames_proc = frames_pixels[:, :self.num_disc_frames, ...]
        else: frames_proc = frames_pixels
        
        wubu_in = frames_proc.reshape(B, -1)
        if wubu_in.shape[1] == 0 and isinstance(self.wubu_discriminator_stack.input_projection, nn.Linear) and self.wubu_discriminator_stack.input_projection.in_features == 1:
             wubu_in = torch.ones(B, 1, device=wubu_in.device, dtype=wubu_in.dtype)
        return self.wubu_discriminator_stack(wubu_in, show_progress=getattr(self.args, 'show_train_progress_bar', False))


# --- Dataset and Trainer  ---
class VideoFrameDatasetCPU(Dataset):
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int],
                 frame_skip: int = 1, data_fraction: float = 1.0,
                 val_fraction: float = 0.0,
                 mode: str = 'train',
                 seed: int = 42,
                 args: argparse.Namespace = None): # Added args for logging control
        super().__init__()
        self.video_path = video_path
        self.num_frames_total_sequence = num_frames_total
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.mode = mode.lower()
        self.logger = logger_wubu_cpu_gen.getChild(f"Dataset_{self.mode.upper()}_{os.getpid()}") # Per-worker logger name
        self.args_ref = args # Store args reference for logging params

        # For __getitem__ logging
        self.pid_counters = {} # Key: pid, Value: call_count for that pid
        self.getitem_log_interval = getattr(self.args_ref, 'getitem_log_interval', 100) if self.args_ref else 100
        self.getitem_slow_threshold = getattr(self.args_ref, 'getitem_slow_threshold', 1.0) if self.args_ref else 1.0


        if not os.path.isfile(self.video_path):
            self.logger.error(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.video_frames_in_ram = None
        self.source_video_fps = 30.0; read_success = False
        if IMAGEIO_AVAILABLE and imageio is not None:
            try:
                self.logger.info(f"Attempting to load video into RAM using imageio: {self.video_path}")
                reader = imageio.get_reader(self.video_path, 'ffmpeg')
                meta = reader.get_meta_data(); self.source_video_fps = meta.get('fps', 30.0)
                frames_list = []
                for frame_np in reader:
                    if frame_np.ndim == 3 and frame_np.shape[-1] in [3,4]: frame_np = np.transpose(frame_np[...,:3], (2,0,1))
                    elif frame_np.ndim == 2: frame_np = np.expand_dims(frame_np, axis=0)
                    else: self.logger.warning(f"Unexpected frame shape from imageio: {frame_np.shape}"); continue
                    frames_list.append(torch.from_numpy(frame_np.copy()))
                self.video_frames_in_ram = torch.stack(frames_list).contiguous() if frames_list else None
                reader.close()
                if self.video_frames_in_ram is not None and self.video_frames_in_ram.shape[0] > 0 : read_success = True
            except Exception as e_ii: self.logger.error(f"imageio failed to read {self.video_path}: {e_ii}", exc_info=True)
        if not read_success: raise RuntimeError(f"Failed to load video '{self.video_path}'. Check imageio and ffmpeg installation.")
        if self.video_frames_in_ram is None or self.video_frames_in_ram.shape[0] == 0:
             raise RuntimeError(f"Video '{self.video_path}' loaded 0 frames.")
        ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3)
        self.logger.info(f"Loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. Est RAM: {ram_usage_gb:.2f} GB.")

        self.resize_transform = T.Resize(self.image_size, antialias=True)
        self.normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.num_disk_frames = self.video_frames_in_ram.shape[0]

        all_samples_start_indices = []
        required_span_on_disk = (self.num_frames_total_sequence - 1) * self.frame_skip + 1
        if self.num_disk_frames >= required_span_on_disk:
            for i in range(self.num_disk_frames - required_span_on_disk + 1):
                all_samples_start_indices.append(i)
        else:
            self.logger.warning(f"Not enough frames ({self.num_disk_frames}) in '{self.video_path}' for sequence (needs {required_span_on_disk}). Dataset for mode '{self.mode}' will be empty.")
            self.samples_start_indices = []

        if not all_samples_start_indices:
            self.logger.error(f"No valid samples from '{self.video_path}' for any split.")
            self.samples_start_indices = []
        else:
            rng_dataset = random.Random(seed)
            rng_dataset.shuffle(all_samples_start_indices)

            if val_fraction > 0.0 and val_fraction < 1.0:
                num_total_samples = len(all_samples_start_indices)
                num_val_samples = int(num_total_samples * val_fraction)
                num_val_samples = max(0, min(num_val_samples, num_total_samples - (1 if num_total_samples > 1 else 0) ))


                if self.mode == 'train':
                    train_indices = all_samples_start_indices[num_val_samples:]
                    if data_fraction < 1.0 and len(train_indices) > 1:
                        num_train_to_keep = max(1, int(len(train_indices) * data_fraction))
                        self.samples_start_indices = train_indices[:num_train_to_keep]
                        self.logger.info(f"Using {data_fraction*100:.2f}% of available training samples: {len(self.samples_start_indices)} samples.")
                    else:
                        self.samples_start_indices = train_indices
                elif self.mode == 'val':
                    self.samples_start_indices = all_samples_start_indices[:num_val_samples]
                else:
                    raise ValueError(f"Invalid mode '{self.mode}'. Must be 'train' or 'val'.")
            else: # No validation split or invalid val_fraction, use all for the current mode
                self.samples_start_indices = all_samples_start_indices
                if self.mode == 'train' and data_fraction < 1.0 and len(self.samples_start_indices) > 1:
                    num_to_keep = max(1, int(len(self.samples_start_indices) * data_fraction))
                    self.samples_start_indices = self.samples_start_indices[:num_to_keep]
                    self.logger.info(f"Using {data_fraction*100:.2f}% of available samples (no val split): {len(self.samples_start_indices)} samples.")
                elif self.mode == 'val': # val_fraction is 0 or >= 1, so no validation set from this split logic
                    self.samples_start_indices = [] if val_fraction > 0 else all_samples_start_indices # if val_fraction >=1, use all for val? Or empty? Let's make it empty if not 0 < vf < 1

        if not self.samples_start_indices and self.mode == 'train':
             self.logger.error(f"VideoFrameDataset (TRAIN): No valid samples from '{self.video_path}'. Training might fail.")
        elif not self.samples_start_indices and self.mode == 'val' and val_fraction > 0.0 and val_fraction < 1.0 :
             self.logger.warning(f"VideoFrameDataset (VAL): No valid samples from '{self.video_path}' for validation split. Validation will be skipped.")

        self.logger.info(f"VideoFrameDatasetCPU ({self.mode.upper()}) Init. DiskFrames:{self.num_disk_frames}. NumSamples:{len(self.samples_start_indices)}. SeqLen:{self.num_frames_total_sequence} (skip {self.frame_skip}).")

    def __len__(self) -> int: return len(self.samples_start_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        t_start_getitem = time.time()

        if not self.samples_start_indices or idx >= len(self.samples_start_indices):
            self.logger.error(f"Index {idx} out of bounds for {self.mode} dataset with {len(self.samples_start_indices)} samples. Returning zeros.")
            # Log timing even for error case
            t_end_getitem = time.time()
            duration = t_end_getitem - t_start_getitem
            self.logger.error(f"__getitem__ (idx {idx}, worker {os.getpid()}, ERROR_OOB) from '{Path(self.video_path).name}' took {duration:.4f}s.")
            return torch.zeros((self.num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        start_frame_idx_in_ram = self.samples_start_indices[idx]; frames_for_sample = []
        for i in range(self.num_frames_total_sequence):
            actual_frame_idx_in_ram = start_frame_idx_in_ram + i * self.frame_skip
            if actual_frame_idx_in_ram < self.num_disk_frames:
                try:
                    frame_tensor_chw_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram]
                    if frame_tensor_chw_uint8.shape[0] == 1: frame_tensor_chw_uint8 = frame_tensor_chw_uint8.repeat(3,1,1)
                    elif frame_tensor_chw_uint8.shape[0] != 3 :
                        self.logger.warning(f"Frame {actual_frame_idx_in_ram} has {frame_tensor_chw_uint8.shape[0]} channels. Using first 3 or repeating.")
                        if frame_tensor_chw_uint8.shape[0] > 3: frame_tensor_chw_uint8 = frame_tensor_chw_uint8[:3,...]
                        else: frame_tensor_chw_uint8 = frame_tensor_chw_uint8[0:1,...].repeat(3,1,1)
                    transformed_frame = self.normalize_transform(self.resize_transform(frame_tensor_chw_uint8).float()/255.0)
                    frames_for_sample.append(transformed_frame)
                except Exception as e:
                    self.logger.error(f"Error transforming frame {actual_frame_idx_in_ram} (sample {idx}): {e}", exc_info=True)
                    frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
            else:
                self.logger.error(f"Frame index {actual_frame_idx_in_ram} out of bounds for disk_frames {self.num_disk_frames} (sample {idx}). Padding with zeros.")
                frames_for_sample.append(torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32))
        
        result_frames = torch.stack(frames_for_sample)
        if len(frames_for_sample) != self.num_frames_total_sequence: # Should be caught by result_frames.shape check
            self.logger.error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total_sequence} for sample {idx}. Padded. Result shape: {result_frames.shape}")
            # Ensure result_frames has correct first dimension if padding happened incorrectly
            if result_frames.shape[0] != self.num_frames_total_sequence:
                 # Fallback: create zeros if dimensions are totally off
                 result_frames = torch.zeros((self.num_frames_total_sequence, 3, self.image_size[0], self.image_size[1]), dtype=torch.float32)


        t_end_getitem = time.time()
        duration = t_end_getitem - t_start_getitem
        
        if self.args_ref: # Only log if args_ref is available
            pid = os.getpid()
            if pid not in self.pid_counters:
                self.pid_counters[pid] = 0
            self.pid_counters[pid] += 1
            current_worker_call_count = self.pid_counters[pid]

            log_this_call = False
            log_reason = ""
            if self.getitem_slow_threshold > 0 and duration > self.getitem_slow_threshold:
                log_this_call = True
                log_reason = f"SLOW ({duration:.4f}s > {self.getitem_slow_threshold:.2f}s)"
            elif self.getitem_log_interval > 0 and current_worker_call_count % self.getitem_log_interval == 0:
                log_this_call = True
                log_reason = f"periodic (call #{current_worker_call_count})"

            if log_this_call:
                self.logger.info(f"__getitem__ (idx {idx}, worker {pid}, {log_reason}) from '{Path(self.video_path).name}' took {duration:.4f}s.")
        
        return result_frames


def load_state_dict_robust(module: nn.Module, state_dict: Optional[dict], module_name: str):
    if state_dict is None:
        logger_wubu_cpu_gen.warning(f"No state_dict provided for {module_name}. It will remain initialized randomly/default.")
        return
    try:
        module.load_state_dict(state_dict)
        logger_wubu_cpu_gen.info(f"Successfully loaded state_dict for {module_name}.")
    except RuntimeError as e:
        logger_wubu_cpu_gen.error(f"RuntimeError loading state_dict for {module_name}: {e}. Trying with strict=False.")
        try:
            module.load_state_dict(state_dict, strict=False)
            logger_wubu_cpu_gen.info(f"Successfully loaded state_dict for {module_name} with strict=False.")
        except Exception as e2:
            logger_wubu_cpu_gen.error(f"Failed to load state_dict for {module_name} even with strict=False: {e2}", exc_info=True)
    except Exception as e_other:
        logger_wubu_cpu_gen.error(f"General error loading state_dict for {module_name}: {e_other}", exc_info=True)


class CPUTrainer:
    def __init__(self, model: WuBuCPUNet, discriminator: WuBuCPUDiscriminator, args: argparse.Namespace):
        self.model = model
        self.discriminator = discriminator
        self.args = args
        self.device = torch.device("cpu")
        self.logger = logger_wubu_cpu_gen.getChild("CPUTrainer")
        self.optimizer_enc_gen = torch.optim.Adam(model.parameters(), lr=args.learning_rate_gen)
        self.optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_disc)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.global_step = 0; self.current_epoch = 0

        if os.path.exists(args.video_data_path) and os.path.isdir(args.video_data_path):
             video_files = [f for f in os.listdir(args.video_data_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
             if not video_files:
                 self.logger.error(f"No video files found in directory: {args.video_data_path}")
                 if IMAGEIO_AVAILABLE and imageio is not None:
                     dummy_video_path = Path(args.video_data_path) / "dummy_cpu_video.mp4"
                     self.logger.info(f"Creating dummy video at {dummy_video_path}")
                     min_req_disk_frames = (args.num_input_frames + args.num_predict_frames - 1) * args.frame_skip + 1
                     num_dummy_frames = max(min_req_disk_frames + 50, int(min_req_disk_frames / max(0.01, (1.0 - args.val_fraction))) + 20 if args.val_fraction > 0 and args.val_fraction < 1 else min_req_disk_frames + 50)

                     try:
                         Path(args.video_data_path).mkdir(parents=True, exist_ok=True)
                         with imageio.get_writer(str(dummy_video_path), fps=15, codec='libx264', quality=8, ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
                             for _ in range(num_dummy_frames):
                                 writer.append_data(np.random.randint(0,255, (args.image_h, args.image_w, args.num_channels), dtype=np.uint8))
                         self.args.video_file_path = str(dummy_video_path) # Update args with path
                     except Exception as e_write:
                          self.logger.error(f"Failed to write dummy video: {e_write}. Check ffmpeg/permissions.")
                          raise FileNotFoundError(f"Failed to create dummy video, and no videos found.")
                 else: raise FileNotFoundError(f"No video files in {args.video_data_path} and imageio not available to create dummy.")
             else: self.args.video_file_path = os.path.join(args.video_data_path, video_files[0]); self.logger.info(f"Using first video found: {self.args.video_file_path}")
        elif os.path.exists(args.video_data_path) and os.path.isfile(args.video_data_path):
            self.args.video_file_path = args.video_data_path # Ensure args is updated
        else: raise FileNotFoundError(f"Video data path not found or invalid: {args.video_data_path}")

        self.train_dataset = VideoFrameDatasetCPU(
            video_path=self.args.video_file_path,
            num_frames_total=args.num_input_frames + args.num_predict_frames,
            image_size=(args.image_h, args.image_w),
            frame_skip=args.frame_skip,
            data_fraction=args.data_fraction,
            val_fraction=args.val_fraction,
            mode='train',
            seed=args.seed,
            args=self.args # Pass args for logging control
        )
        self.train_loader = DataLoader( self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False )

        self.val_loader = None
        if args.val_fraction > 0.0 and args.val_fraction < 1.0:
            self.val_dataset = VideoFrameDatasetCPU(
                video_path=self.args.video_file_path,
                num_frames_total=args.num_input_frames + args.num_predict_frames,
                image_size=(args.image_h, args.image_w),
                frame_skip=args.frame_skip,
                data_fraction=1.0,  # Use all of the validation split
                val_fraction=args.val_fraction,
                mode='val',
                seed=args.seed,
                args=self.args # Pass args for logging control
            )
            if len(self.val_dataset) > 0:
                self.val_loader = DataLoader(
                    self.val_dataset, batch_size=args.val_batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False
                )
                self.logger.info(f"Validation DataLoader created with {len(self.val_dataset)} samples.")
            else:
                self.logger.warning("Validation dataset is empty after split, validation will be skipped.")
                self.val_loader = None
        else:
            self.logger.info("val_fraction is not in (0,1), validation will be skipped.")


        self.fixed_noise_for_sampling: Optional[torch.Tensor] = None
        if args.wandb_log_fixed_noise_samples_interval > 0 and args.num_val_samples_to_log > 0 and args.latent_dim > 0:
            self.fixed_noise_for_sampling = torch.randn( args.num_val_samples_to_log, args.latent_dim, device=self.device )
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        self.logger.info("CPUTrainer initialized for CPU execution.")

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def _compute_recon_loss(self, recon_frames: torch.Tensor, target_frames: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_frames, target_frames)

    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, frames_to_log: Optional[torch.Tensor], num_frames_per_sequence_to_log: int = 1, num_sequences_to_log_max: int = 2):
        if not (self.args.wandb and WANDB_AVAILABLE and wandb.run and frames_to_log is not None and frames_to_log.numel() > 0): return

        current_frames_dim = frames_to_log.ndim
        if current_frames_dim == 4: 
            frames_to_log = frames_to_log.unsqueeze(1) 

        if frames_to_log.ndim != 5:
            self.logger.warning(f"WandB log samples: unexpected shape {frames_to_log.shape} (original: {current_frames_dim}D). Expected 5D. Skip."); return

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
            try: wandb.log({f"samples_video_cpu/{tag_prefix}": wandb_imgs}, step=self.global_step)
            except Exception as e: self.logger.error(f"WandB video sample log fail for {tag_prefix}: {e}", exc_info=True)


    @torch.no_grad()
    def _validate_epoch(self):
        if not self.val_loader:
            self.logger.info("Skipping validation: No validation data loader.")
            return

        self.model.eval()
        self.discriminator.eval()

        total_val_loss_recon = 0.0
        total_val_loss_kl = 0.0
        total_val_loss_g_adv = 0.0
        total_val_loss_d = 0.0
        num_val_batches = 0

        val_prog_bar_desc = f"Validating Epoch {self.current_epoch+1}"
        if hasattr(self.val_loader.dataset, 'video_path') and self.val_loader.dataset.video_path:
            val_prog_bar_desc += f" ({Path(self.val_loader.dataset.video_path).name})"

        
        iter_val_loader = iter(self.val_loader)
        val_prog_bar = tqdm(range(len(self.val_loader)), desc=val_prog_bar_desc,
                            disable=(os.getenv('CI')=='true' or not self.args.show_train_progress_bar),
                            dynamic_ncols=True, leave=False)


        first_batch_recon_frames_val = None
        first_batch_target_frames_val = None

        for batch_idx in val_prog_bar:
            val_frames_seq_raw = next(iter_val_loader) # Not timing val loader for now
            val_frames_seq = val_frames_seq_raw.to(self.device)

            recon_frames_val, mu_val, logvar_val, bboxes_for_gen_val, _ = self.model(val_frames_seq)
            target_frames_val = val_frames_seq[:, self.args.num_input_frames : self.args.num_input_frames + self.args.num_predict_frames, ...]

            loss_recon_val = self._compute_recon_loss(recon_frames_val, target_frames_val)
            loss_kl_val = self._compute_kl_loss(mu_val, logvar_val)

            real_logits_val = self.discriminator(target_frames_val)
            loss_d_real_val = self.adversarial_loss(real_logits_val, torch.ones_like(real_logits_val))

            fake_logits_val = self.discriminator(recon_frames_val.detach()) 
            loss_d_fake_val = self.adversarial_loss(fake_logits_val, torch.zeros_like(fake_logits_val))
            loss_d_val = (loss_d_real_val + loss_d_fake_val) * 0.5

            fake_logits_g_val = self.discriminator(recon_frames_val) 
            loss_g_adv_val = self.adversarial_loss(fake_logits_g_val, torch.ones_like(fake_logits_g_val))

            total_val_loss_recon += loss_recon_val.item()
            total_val_loss_kl += loss_kl_val.item()
            total_val_loss_g_adv += loss_g_adv_val.item()
            total_val_loss_d += loss_d_val.item()
            num_val_batches += 1

            if batch_idx == 0 and self.args.num_val_samples_to_log > 0:
                first_batch_recon_frames_val = recon_frames_val.detach().cpu()
                first_batch_target_frames_val = target_frames_val.detach().cpu()

        if num_val_batches > 0:
            avg_val_loss_recon = total_val_loss_recon / num_val_batches
            avg_val_loss_kl = total_val_loss_kl / num_val_batches
            avg_val_loss_g_adv = total_val_loss_g_adv / num_val_batches
            avg_val_loss_d = total_val_loss_d / num_val_batches
            avg_val_loss_g_total = self.args.lambda_recon * avg_val_loss_recon + \
                                self.args.lambda_kl * avg_val_loss_kl + \
                                self.args.lambda_gan * avg_val_loss_g_adv

            self.logger.info(f"Validation Epoch {self.current_epoch+1}: "
                            f"Recon: {avg_val_loss_recon:.4f}, KL: {avg_val_loss_kl:.4f}, "
                            f"G_Adv: {avg_val_loss_g_adv:.4f}, D_Loss: {avg_val_loss_d:.4f}, "
                            f"G_Total: {avg_val_loss_g_total:.4f}")

            if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                wandb_val_logs = {
                    "val/loss_recon": avg_val_loss_recon, "val/loss_kl": avg_val_loss_kl,
                    "val/loss_g_adv": avg_val_loss_g_adv, "val/loss_d": avg_val_loss_d,
                    "val/loss_g_total": avg_val_loss_g_total, "epoch": self.current_epoch
                }
                wandb.log(wandb_val_logs, step=self.global_step)

                if self.args.wandb_log_val_recon_interval_epochs > 0 and \
                   (self.current_epoch + 1) % self.args.wandb_log_val_recon_interval_epochs == 0:
                    if first_batch_recon_frames_val is not None:
                        self._log_samples_to_wandb("val_recon", first_batch_recon_frames_val,
                                                   min(first_batch_recon_frames_val.shape[1],3), self.args.num_val_samples_to_log)
                    if first_batch_target_frames_val is not None:
                        self._log_samples_to_wandb("val_target", first_batch_target_frames_val,
                                                   min(first_batch_target_frames_val.shape[1],3), self.args.num_val_samples_to_log)
        else:
             self.logger.info(f"Validation Epoch {self.current_epoch+1}: No batches processed (val_loader might be empty).")

        self.model.train()
        self.discriminator.train()


    def train(self, start_epoch: int = 0, initial_global_step: int = 0):
        self.global_step, self.current_epoch = initial_global_step, start_epoch
        self.logger.info(f"Starting CPU training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch+1}, GStep: {initial_global_step}")

        if not self.train_loader or len(self.train_loader.dataset) == 0:
            self.logger.error("Training dataset is empty. Cannot start training. Check video path and data fractions.")
            if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.finish(exit_code=1)
            return

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting.")
            self.model.train()
            self.discriminator.train()

            prog_bar_desc = f"Epoch {epoch+1}"
            if hasattr(self.train_loader.dataset, 'video_path') and self.train_loader.dataset.video_path:
                prog_bar_desc += f" ({Path(self.train_loader.dataset.video_path).name})"
            
            iter_train_loader = iter(self.train_loader)
            prog_bar = tqdm(range(len(self.train_loader)), desc=prog_bar_desc,
                            disable=(os.getenv('CI')=='true' or not self.args.show_train_progress_bar),
                            dynamic_ncols=True)

            for batch_idx in prog_bar:
                t_dataload_start = time.time()
                real_frames_seq_raw = next(iter_train_loader)
                t_dataload_end = time.time()
                time_dataload = t_dataload_end - t_dataload_start

                t_todevice_start = time.time()
                real_frames_seq = real_frames_seq_raw.to(self.device)
                t_todevice_end = time.time()
                time_todevice = t_todevice_end - t_todevice_start


                # --- Discriminator Training ---
                t_disc_cycle_start = time.time()
                self.optimizer_disc.zero_grad()

                target_real_for_d = real_frames_seq[:, self.args.num_input_frames : self.args.num_input_frames + self.args.num_predict_frames, ...]

                t_disc_real_fwd_start = time.time()
                real_logits = self.discriminator(target_real_for_d) 
                t_disc_real_fwd_end = time.time()
                time_disc_real_fwd = t_disc_real_fwd_end - t_disc_real_fwd_start

                loss_d_real = self.adversarial_loss(real_logits, torch.ones_like(real_logits))

                with torch.no_grad():
                    recon_frames_no_grad, _, _, _, _ = self.model(real_frames_seq)

                t_disc_fake_fwd_start = time.time()
                fake_logits = self.discriminator(recon_frames_no_grad.detach()) 
                t_disc_fake_fwd_end = time.time()
                time_disc_fake_fwd = t_disc_fake_fwd_end - t_disc_fake_fwd_start

                loss_d_fake = self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
                loss_d = (loss_d_real + loss_d_fake) * 0.5

                loss_d.backward()
                self.optimizer_disc.step()
                t_disc_cycle_end = time.time()
                time_disc_cycle = t_disc_cycle_end - t_disc_cycle_start

                # --- Generator Training ---
                t_gen_cycle_start = time.time()
                self.optimizer_enc_gen.zero_grad()

                recon_frames_g, mu_g, logvar_g, _, model_timings = self.model(real_frames_seq) 

                target_for_recon = real_frames_seq[:, self.args.num_input_frames : self.args.num_input_frames + self.args.num_predict_frames, ...]
                loss_recon = self._compute_recon_loss(recon_frames_g, target_for_recon)
                loss_kl = self._compute_kl_loss(mu_g, logvar_g)

                t_gen_adv_fwd_start = time.time()
                fake_logits_g = self.discriminator(recon_frames_g) 
                t_gen_adv_fwd_end = time.time()
                time_gen_adv_fwd = t_gen_adv_fwd_end - t_gen_adv_fwd_start

                loss_g_adv = self.adversarial_loss(fake_logits_g, torch.ones_like(fake_logits_g))
                loss_g = self.args.lambda_recon*loss_recon + self.args.lambda_kl*loss_kl + self.args.lambda_gan*loss_g_adv

                loss_g.backward()
                self.optimizer_enc_gen.step()
                t_gen_cycle_end = time.time()
                time_gen_cycle = t_gen_cycle_end - t_gen_cycle_start

                self.global_step +=1

                if self.global_step % self.args.log_interval == 0:
                    log_items = {
                        "L_D": loss_d.item(), "L_G": loss_g.item(),
                        "Rec": loss_recon.item(), "KL": loss_kl.item(), "AdvG": loss_g_adv.item(),
                        "T_DataLoad": time_dataload, "T_ToDevice": time_todevice,
                        "T_Enc": model_timings.get("time_encoder_fwd",0),
                        "T_BBoxGen": model_timings.get("time_bbox_gen", 0),
                        "T_PatchExt": model_timings.get("time_patch_extract", 0),
                        "T_Gen": model_timings.get("time_generator_fwd",0),
                        "T_PatchAsm": model_timings.get("time_patch_assembly", 0),
                        "T_D_Real": time_disc_real_fwd, "T_D_Fake": time_disc_fake_fwd,
                        "T_D_Cycle": time_disc_cycle,
                        "T_G_Adv": time_gen_adv_fwd, "T_G_Cycle": time_gen_cycle
                    }
                    formatted_log_items = {}
                    for k_l,v_l in log_items.items():
                        if k_l.startswith("T_"): formatted_log_items[k_l] = f"{v_l:.3f}s"
                        else: formatted_log_items[k_l] = f"{v_l:.3f}"
                    log_str_parts = [f"{k_l}:{v_l}" for k_l,v_l in formatted_log_items.items()]
                    postfix_str = f"L_D:{loss_d.item():.2f} L_G:{loss_g.item():.2f} Data:{time_dataload:.2f}s"
                    full_log_msg = f"E:{epoch+1} S:{self.global_step} | " + " | ".join(log_str_parts)
                    prog_bar.set_postfix_str(postfix_str)
                    self.logger.info(full_log_msg)

                    if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                        wandb_log_data = {
                            "train/loss_d": loss_d.item(), "train/loss_g": loss_g.item(),
                            "train/loss_recon": loss_recon.item(), "train/loss_kl": loss_kl.item(),
                            "train/loss_g_adv": loss_g_adv.item(),
                            "time/data_load": time_dataload, "time/to_device": time_todevice,
                            "time/encoder_fwd": model_timings.get("time_encoder_fwd",0),
                            "time/bbox_gen": model_timings.get("time_bbox_gen", 0),
                            "time/patch_extract": model_timings.get("time_patch_extract", 0),
                            "time/generator_fwd": model_timings.get("time_generator_fwd",0),
                            "time/patch_assembly": model_timings.get("time_patch_assembly",0),
                            "time/disc_real_fwd": time_disc_real_fwd, "time/disc_fake_fwd": time_disc_fake_fwd,
                            "time/disc_cycle": time_disc_cycle,
                            "time/gen_adv_fwd": time_gen_adv_fwd, "time/gen_cycle": time_gen_cycle,
                            "global_step":self.global_step, "epoch_frac":epoch+(batch_idx/max(1,len(self.train_loader)))
                        }
                        wandb.log(wandb_log_data, step=self.global_step)

                if self.args.wandb_log_train_recon_interval > 0 and self.global_step % self.args.wandb_log_train_recon_interval == 0:
                    self._log_samples_to_wandb("train_recon", recon_frames_g.detach(), min(recon_frames_g.shape[1],3), self.args.num_val_samples_to_log)
                    self._log_samples_to_wandb("train_target", target_for_recon.detach(), min(target_for_recon.shape[1],3), self.args.num_val_samples_to_log)

            if (epoch + 1) % self.args.validate_every_n_epochs == 0:
                self._validate_epoch()

            if self.args.save_interval > 0 and (epoch+1)%self.args.save_interval==0: self._save_checkpoint(epoch)
            if self.fixed_noise_for_sampling is not None and self.args.wandb_log_fixed_noise_samples_interval > 0 and (epoch+1)%self.args.wandb_log_fixed_noise_samples_interval==0:
                fixed_noise_pixels = self.sample(self.args.num_val_samples_to_log, noise=self.fixed_noise_for_sampling)
                if fixed_noise_pixels is not None: self._log_samples_to_wandb("fixed_noise_gen",fixed_noise_pixels,min(fixed_noise_pixels.shape[1],3),self.args.num_val_samples_to_log)

        self.logger.info("CPU Training finished."); self._save_checkpoint(self.current_epoch,is_final=True)


    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        self.model.eval()
        if self.args.latent_dim <= 0: self.logger.error("Sample: Latent dim <= 0."); return None
        z = noise.to(self.device) if noise is not None else torch.randn(num_samples, self.args.latent_dim, device=self.device)
        num_samples_eff = z.shape[0]

        dummy_bboxes_list = []
        frame_dims_s = (self.args.image_w, self.args.image_h)
        gaad_num_regions_eff = self.args.gaad_num_regions if self.args.gaad_num_regions > 0 else 1

        effective_num_predict_frames = self.args.num_predict_frames if self.args.num_predict_frames > 0 else 0

        for _ in range(num_samples_eff):
            item_bboxes_single_sample = [golden_subdivide_rect_fixed_n_cpu(frame_dims_s, gaad_num_regions_eff, z.dtype, self.args.gaad_min_size_px)
                                         for _ in range(effective_num_predict_frames)]
            if not item_bboxes_single_sample and effective_num_predict_frames > 0 : 
                 self.logger.warning(f"Sample: item_bboxes_single_sample empty despite num_predict_frames={effective_num_predict_frames}.")
                 ff_bbox_single_region = torch.tensor([0,0,self.args.image_w,self.args.image_h],device=self.device,dtype=z.dtype)
                 item_bboxes_single_sample = [ff_bbox_single_region.unsqueeze(0).expand(gaad_num_regions_eff,4) for _ in range(effective_num_predict_frames)]


            if not item_bboxes_single_sample: 
                 dummy_bboxes_list.append(torch.empty(0, gaad_num_regions_eff, 4, device=self.device, dtype=z.dtype))
            else:
                 dummy_bboxes_list.append(torch.stack(item_bboxes_single_sample))

        if not dummy_bboxes_list and num_samples_eff > 0:
            dummy_bboxes_decode = torch.empty(num_samples_eff, 0, gaad_num_regions_eff, 4, device=self.device, dtype=z.dtype)
        elif not dummy_bboxes_list and num_samples_eff == 0:
            dummy_bboxes_decode = torch.empty(0, effective_num_predict_frames, gaad_num_regions_eff, 4, device=self.device, dtype=z.dtype)
        else:
            try:
                dummy_bboxes_decode = torch.stack(dummy_bboxes_list)
            except RuntimeError as e: 
                self.logger.error(f"Error stacking bboxes in sample: {e}. Shapes: {[b.shape for b in dummy_bboxes_list]}. Using fallback.")
                ff_bbox = torch.tensor([0,0,self.args.image_w,self.args.image_h],device=self.device,dtype=z.dtype)
                dummy_bboxes_decode = ff_bbox.view(1,1,1,4).expand(num_samples_eff, effective_num_predict_frames, gaad_num_regions_eff,4)


        if dummy_bboxes_decode.shape[0] != num_samples_eff and num_samples_eff > 0:
             self.logger.warning(f"Sample: Generated bboxes batch size ({dummy_bboxes_decode.shape[0]}) != num_samples ({num_samples_eff}). This might indicate an issue.")

        generated_frames, _ = self.model.generator(z, dummy_bboxes_decode) # Ignore timings from generator here
        self.model.train()
        return generated_frames

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        serializable_args = {}
        for k, v in vars(self.args).items():
            if isinstance(v, Path): serializable_args[k] = str(v)
            else: serializable_args[k] = v

        data = {'epoch':epoch,'global_step':self.global_step,'model_state_dict':self.model.state_dict(),
                'discriminator_state_dict':self.discriminator.state_dict(),
                'optimizer_enc_gen_state_dict':self.optimizer_enc_gen.state_dict(),
                'optimizer_disc_state_dict':self.optimizer_disc.state_dict(),'args':serializable_args}
        fn = f"wubucpu_gen_v1_{'final' if is_final else f'ep{epoch+1}_step{self.global_step}'}.pt"
        try: torch.save(data, Path(self.args.checkpoint_dir)/fn); self.logger.info(f"Checkpoint saved: {fn}")
        except Exception as e: self.logger.error(f"Error saving ckpt {fn}: {e}", exc_info=True)

    def load_checkpoint(self, ckpt_path_str: Optional[str]) -> Tuple[int,int]:
        if not ckpt_path_str or not os.path.exists(ckpt_path_str):
            self.logger.warning(f"Checkpoint '{ckpt_path_str}' not found. Starting fresh."); return 0,0
        try: ckpt = torch.load(ckpt_path_str, map_location=self.device); self.logger.info(f"Loaded checkpoint: {ckpt_path_str}")
        except Exception as e: self.logger.error(f"Failed to load ckpt {ckpt_path_str}: {e}. Fresh start.", exc_info=True); return 0,0

        if 'args' in ckpt and ckpt['args'] is not None:
            self.logger.info("Attempting to load args from checkpoint.")
            loaded_ckpt_args = ckpt['args']
            current_args_dict = vars(self.args)
            for k_ckpt, v_ckpt in loaded_ckpt_args.items():
                if k_ckpt not in current_args_dict:
                    setattr(self.args, k_ckpt, v_ckpt)
                    self.logger.info(f"  Arg '{k_ckpt}' loaded from ckpt: '{v_ckpt}' (added)")
                elif current_args_dict[k_ckpt] != v_ckpt and k_ckpt not in ['load_checkpoint', 'epochs', 'wandb_run_name', 'video_data_path', 'checkpoint_dir']: # Don't override critical paths/control args
                     self.logger.info(f"  Arg '{k_ckpt}': Ckpt='{v_ckpt}', Current='{current_args_dict[k_ckpt]}'. Using current value.")


        load_state_dict_robust(self.model, ckpt.get('model_state_dict'), "Model")
        load_state_dict_robust(self.discriminator, ckpt.get('discriminator_state_dict'), "Discriminator")
        if ckpt.get('optimizer_enc_gen_state_dict'): load_state_dict_robust(self.optimizer_enc_gen, ckpt['optimizer_enc_gen_state_dict'], "Optimizer G")
        if ckpt.get('optimizer_disc_state_dict'): load_state_dict_robust(self.optimizer_disc, ckpt['optimizer_disc_state_dict'], "Optimizer D")

        gs = ckpt.get('global_step',0)
        ep_saved = ckpt.get('epoch',0)

        if self.args.load_checkpoint_reset_epoch:
            start_ep = ep_saved 
        else:
            start_ep = ep_saved + 1 
        
        if gs == 0 and ep_saved == 0 and not self.args.load_checkpoint_reset_epoch:
            start_ep = 0
            
        self.logger.info(f"Resuming. Saved Epoch: {ep_saved}, GlobalStep: {gs}. Starting training from Epoch: {start_ep}.")
        return gs, start_ep


def parse_cpu_arguments():
    def bool_type(s):
        s_lower = str(s).lower()
        if s_lower in ('yes', 'true', 't', 'y', '1'): return True
        elif s_lower in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError(f"Boolean value expected, got '{s}'")

    parser = argparse.ArgumentParser(description="WuBu CPU-Only Generator (v1 Advanced FDQWR)")
    # Paths
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_cpu")
    parser.add_argument('--checkpoint_dir', type=str, default='wubucpu_gen_v1_checkpoints_adv')
    parser.add_argument('--load_checkpoint', type=str, default=None)
    # General Training
    parser.add_argument('--seed',type=int, default=42); parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100); parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate_gen',type=float,default=1e-4); parser.add_argument('--learning_rate_disc',type=float,default=1e-4)
    parser.add_argument('--lambda_recon', type=float, default=10.0); parser.add_argument('--lambda_kl', type=float, default=0.01); parser.add_argument('--lambda_gan', type=float, default=1.0)
    # Data/Model Specs
    parser.add_argument('--latent_dim', type=int, default=128); parser.add_argument('--image_h', type=int, default=64); parser.add_argument('--image_w', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=3); parser.add_argument('--num_input_frames', type=int, default=2); parser.add_argument('--num_predict_frames', type=int, default=1)
    parser.add_argument('--frame_skip', type=int, default=1)
    # GAAD & Patching
    parser.add_argument('--gaad_num_regions', type=int, default=16); parser.add_argument('--gaad_min_size_px', type=int, default=4)
    parser.add_argument('--patch_size_h', type=int, default=8); parser.add_argument('--patch_size_w', type=int, default=8)

    # WuBu Core Common Params
    parser.add_argument('--wubu_initial_s', type=float, default=1.0); parser.add_argument('--wubu_s_decay', type=float, default=0.9999)
    parser.add_argument('--wubu_initial_c', type=float, default=0.1);
    parser.add_argument('--wubu_c_phi_influence', type=bool_type, default=True)
    parser.add_argument('--wubu_num_phi_scaffold_points', type=int, default=3); parser.add_argument('--wubu_phi_scaffold_init_scale', type=float, default=0.001)
    parser.add_argument('--wubu_use_gaussian_rungs', type=bool_type, default=True)
    parser.add_argument('--wubu_base_gaussian_std_dev', type=float, default=0.005); parser.add_argument('--wubu_gaussian_std_dev_decay', type=float, default=0.999995)
    parser.add_argument('--wubu_rung_affinity_temp', type=float, default=0.01); parser.add_argument('--wubu_rung_modulation_strength', type=float, default=0.001)
    parser.add_argument('--wubu_t_tilde_scale', type=float, default=1.00001)
    parser.add_argument('--wubu_micro_transform_type', type=str, default="mlp", choices=["mlp", "linear", "identity"])
    parser.add_argument('--wubu_micro_transform_hidden_factor', type=float, default=0.5)
    parser.add_argument('--wubu_use_quaternion_so4', type=bool_type, default=True)
    parser.add_argument('--wubu_scaffold_co_rotation_mode', type=str, default="none", choices=["none", "matrix_only"])

    # WuBu Advanced Micro-Transform Features
    parser.add_argument('--wubu_enable_isp', type=bool_type, default=True, help="Enable Internal Sub-Processing in MicroTransforms")
    parser.add_argument('--wubu_enable_stateful', type=bool_type, default=True, help="Enable Stateful MicroTransforms")
    parser.add_argument('--wubu_enable_hypernet', type=bool_type, default=True, help="Enable Hypernetwork Modulation in MicroTransforms")
    parser.add_argument('--wubu_internal_state_factor', type=float, default=0.5, help="Factor for internal state dim relative to base_micro_dim")
    parser.add_argument('--wubu_hypernet_strength', type=float, default=0.01, help="Strength of hypernetwork modulation")

    # WuBu Stack Specifics
    parser.add_argument('--enc_wubu_num_virtual_levels', type=int, default=1000); parser.add_argument('--enc_wubu_base_micro_dim', type=int, default=4); parser.add_argument('--enc_wubu_num_physical_stages', type=int, default=20)
    parser.add_argument('--gen_wubu_num_virtual_levels', type=int, default=2000); parser.add_argument('--gen_wubu_base_micro_dim', type=int, default=4); parser.add_argument('--gen_wubu_num_physical_stages', type=int, default=40)
    parser.add_argument('--disc_wubu_num_virtual_levels', type=int, default=500); parser.add_argument('--disc_wubu_base_micro_dim', type=int, default=4); parser.add_argument('--disc_wubu_num_physical_stages', type=int, default=10)

    # Logging & Saving
    parser.add_argument('--wandb', type=bool_type, default=True);
    parser.add_argument('--wandb_project',type=str,default='WuBuCPUGenV1Adv')
    parser.add_argument('--wandb_run_name',type=str,default=None); parser.add_argument('--log_interval',type=int, default=5)
    parser.add_argument('--save_interval',type=int, default=1); parser.add_argument('--wandb_log_train_recon_interval', type=int, default=20)
    parser.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=5)
    parser.add_argument('--num_val_samples_to_log', type=int, default=2)
    parser.add_argument('--data_fraction', type=float, default=0.1);
    parser.add_argument('--show_train_progress_bar', type=bool_type, default=True)
    parser.add_argument('--getitem_log_interval', type=int, default=100, help="Log VideoFrameDataset __getitem__ time every N calls per worker (0 to disable).")
    parser.add_argument('--getitem_slow_threshold', type=float, default=1.0, help="Log __getitem__ time if it exceeds this threshold in seconds (0 to disable).")


    # Checkpoint loading behavior
    parser.add_argument('--load_checkpoint_reset_epoch', type=bool_type, default=False, help="If True, use saved epoch as start; otherwise, resume from epoch+1.")

    # Validation specific arguments
    parser.add_argument('--val_fraction', type=float, default=0.1, help="Fraction of data to use for validation (0 to disable).")
    parser.add_argument('--val_batch_size', type=int, default=None, help="Batch size for validation (defaults to train batch_size).")
    parser.add_argument('--validate_every_n_epochs', type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument('--wandb_log_val_recon_interval_epochs', type=int, default=1, help="Log validation reconstructions to WandB every N validation-performing epochs.")

    parsed_args = parser.parse_args()
    if parsed_args.val_batch_size is None:
        parsed_args.val_batch_size = parsed_args.batch_size
    return parsed_args

def main_cpu():
    args = parse_cpu_arguments()
    if args.show_train_progress_bar == False: 
        global tqdm
        def tqdm_disabled_wrapper(*args_tqdm, **kwargs_tqdm):
            if len(args_tqdm) > 0 and hasattr(args_tqdm[0], '__iter__'): return args_tqdm[0] 
            return None 
        tqdm = tqdm_disabled_wrapper


    device = torch.device("cpu")
    global DEBUG_MICRO_TRANSFORM_INTERNALS
    # Assuming 'debug_micro_transforms' is not an arg, keep default behavior for now
    # if hasattr(args, 'debug_micro_transforms') and args.debug_micro_transforms: 
    #     DEBUG_MICRO_TRANSFORM_INTERNALS = True
    #     logger_wubu_cpu_gen.setLevel(logging.DEBUG)
    # else:
    logger_wubu_cpu_gen.setLevel(logging.INFO)

    logging.basicConfig(level=logger_wubu_cpu_gen.level, format='%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s', force=True)
    logger_wubu_cpu_gen.info(f"--- WuBuCPUOnlyGenV1 (Advanced FDQWR) ---")
    logger_wubu_cpu_gen.info(f"Effective Args: {vars(args)}")
    random.seed(args.seed); os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubucpu_adv_{datetime.now().strftime('%y%m%d_%H%M')}"
        try:
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
            logger_wubu_cpu_gen.info(f"WandB initialized: Run '{run_name}', Project '{args.wandb_project}'")
        except Exception as e_wandb: logger_wubu_cpu_gen.error(f"WandB initialization failed: {e_wandb}", exc_info=True); args.wandb = False

    video_config = { "num_channels": args.num_channels, "num_input_frames": args.num_input_frames, "num_predict_frames": args.num_predict_frames }
    gaad_config = { "num_regions": args.gaad_num_regions, "min_size_px": args.gaad_min_size_px }

    model = WuBuCPUNet(args, video_config, gaad_config).to(device)
    discriminator = WuBuCPUDiscriminator(args, video_config, gaad_config).to(device)

    if args.wandb and WANDB_AVAILABLE and wandb.run: 
        # Changed log="all" to log="gradients" to reduce data volume from parameter histograms
        wandb.watch(model, log="gradients", log_freq=args.log_interval * 10, log_graph=False)
        wandb.watch(discriminator, log="gradients", log_freq=args.log_interval * 10, log_graph=False)

    trainer = CPUTrainer(model, discriminator, args)
    start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint) if args.load_checkpoint else (0,0)
    
    try:
        trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt: logger_wubu_cpu_gen.info("Training interrupted by user.")
    except Exception as e: logger_wubu_cpu_gen.error(f"Training loop crashed: {e}", exc_info=True)
    finally:
        logger_wubu_cpu_gen.info("Finalizing run...")
        if hasattr(trainer, 'current_epoch'): 
             trainer._save_checkpoint(epoch=trainer.current_epoch, is_final=True)
        if args.wandb and WANDB_AVAILABLE and wandb.run: wandb.finish()
        logger_wubu_cpu_gen.info("WuBuCPUOnlyGenV1 (Advanced FDQWR) script finished.")

if __name__ == "__main__":
    main_cpu()
