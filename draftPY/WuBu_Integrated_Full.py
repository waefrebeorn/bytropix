# WuBu_Integrated_Full.py
# Consolidating Fractal Depth, Gaussian Rungs, Phi-Scaffolds,
# and Quaternion SO(4) / Generalized SO(n) Rotations.
# Incorporating advanced intra-4D processing concepts with verification aids.
# Version: 2025-05-20.08 - Corrected Grad Check Print Logic

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict

# --- Constants ---
EPS = 1e-5
PHI = (1 + math.sqrt(5)) / 2
TAN_VEC_CLAMP_VAL = 1e4
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e8
MIN_WUBU_LEVEL_SCALE = EPS
MAX_WUBU_LEVEL_SCALE = 10.0

# --- Basic Logging Setup ---
logger_wubu_integrated = logging.getLogger("WuBuFractalIntegratedFull")
if not logger_wubu_integrated.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_wubu_integrated.addHandler(handler)
    logger_wubu_integrated.setLevel(logging.INFO)

# --- Global flag for detailed micro-transform debugging ---
DEBUG_MICRO_TRANSFORM_INTERNALS = False

# --- Utility Functions ---
def init_weights_general(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        if m.dim() > 1: nn.init.xavier_uniform_(m.data)
        else: nn.init.normal_(m.data, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', True)):
            if hasattr(m, 'weight') and m.weight is not None: nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor:
    return F.softplus(param_unconstrained) + min_val

def print_tensor_stats(tensor: Optional[torch.Tensor], name: str, micro_level_idx: Optional[int] = None, stage_idx: Optional[int] = None, enabled: bool = True, batch_idx_to_print: int = 0):
    if not enabled or tensor is None or not DEBUG_MICRO_TRANSFORM_INTERNALS:
        return
    if tensor.numel() == 0:
        # print(f"DEBUG STATS ({name}): Tensor is empty.") # Reduced noise
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
    print(prefix +
          f"Sh:{tensor.shape},Dt:{tensor.dtype},"
          f"Min:{item_tensor.min().item():.2e},Max:{item_tensor.max().item():.2e},"
          f"Mu:{item_tensor.mean().item():.2e},Std:{item_tensor.std().item():.2e},"
          f"Nrm:{torch.linalg.norm(item_tensor.float()).item():.2e}")

# --- Quaternion Math ---
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
            logger_wubu_integrated.warning_once(f"Quat axis-angle: Cannot broadcast axis (shape {axis.shape}) to align with angle_rad (shape {angle_rad.shape}). Using default.")
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
        logger_wubu_integrated.error(f"Quat component extraction failed. q1_eff: {q1_eff.shape}, q2_eff: {q2_eff.shape}. Error: {e}")
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

# --- Hyperbolic Geometry Utilities ---
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
    def exponential_map(v: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_exponential_map(v, c_scalar, scale_scalar=1.0, eps=eps)
    @staticmethod
    def logarithmic_map(y: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_logarithmic_map(y, c_scalar, scale_scalar=1.0, eps=eps)

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

# --- Core WuBu Rotational and Transformational Components ---
class SkewSymmetricMatrix(nn.Module):
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
                logger_wubu_integrated.warning_once(f"SkewSymMtx: Mismatch triu_idx ({indices.shape[1]}) vs params ({self.params.shape[0]}) for dim {self.n_dim}.")
                num_to_fill = min(indices.shape[1], self.params.shape[0])
                if num_to_fill > 0: X[indices[0,:num_to_fill], indices[1,:num_to_fill]] = self.params[:num_to_fill]
        X = X - X.T
        try: R = torch.linalg.matrix_exp(torch.clamp(X, -5, 5))
        except Exception as e:
            logger_wubu_integrated.error(f"matrix_exp error: {e}. Using identity. X norm: {X.norm().item() if X.numel()>0 else 'N/A'}", exc_info=True)
            R = torch.eye(self.n_dim, device=X.device, dtype=X.dtype)
        return R

class QuaternionSO4From6Params(nn.Module):
    def __init__(self, small_init_std: float = 0.001):
        super().__init__()
        self.so3_params_for_p = nn.Parameter(torch.randn(3) * small_init_std)
        self.so3_params_for_q = nn.Parameter(torch.randn(3) * small_init_std)
    def _so3_params_to_unit_quaternion(self, so3_params: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.norm(so3_params, dim=-1)
        identity_q_val = torch.zeros(4, device=so3_params.device, dtype=so3_params.dtype)
        if identity_q_val.numel() > 0 : identity_q_val[0] = 1.0
        if angle < EPS : return identity_q_val
        axis = so3_params / (angle.unsqueeze(-1) + EPS)
        return quaternion_from_axis_angle(angle, axis)
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._so3_params_to_unit_quaternion(self.so3_params_for_p), self._so3_params_to_unit_quaternion(self.so3_params_for_q)

class FractalMicroTransformRungs(nn.Module):
    def __init__(self, dim: int = 4, transform_type: str = "mlp",
                 hidden_dim_factor: float = 1.0, # dropout: float = 0.0, # Dropout not used
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
                if dim > 0 : logger_wubu_integrated.warning_once("HyperNet modulation enabled but no suitable MLP found. Disabling for this micro-transform.")
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
        v_scaled = v * w_modulation_factor * (torch.tanh(self.v_scaler_param) + 1.0) # ensure v_scaler_param factor is positive
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
            if x_tan.shape[-1] != 4: logger_wubu_integrated.error_once(f"Quat rot needs dim 4, got {x_tan.shape[-1]}. Skip."); return x_tan, None
            p_b, q_b = (p_quat.unsqueeze(0).expand_as(x_tan), q_quat.unsqueeze(0).expand_as(x_tan)) if x_tan.dim() > 1 and p_quat.dim() == 1 else (p_quat, q_quat)
            rotated_x_tan = quaternion_multiply(quaternion_multiply(p_b, x_tan), q_b)
        elif self.rotation_generator_matrix is not None and isinstance(self.rotation_generator_matrix, SkewSymmetricMatrix):
            R_matrix_this_step = self.rotation_generator_matrix().to(x_tan.device, x_tan.dtype)
            if x_tan.dim() == 2: rotated_x_tan = torch.einsum('ij,bj->bi', R_matrix_this_step, x_tan)
            elif x_tan.dim() == 1: rotated_x_tan = torch.matmul(R_matrix_this_step, x_tan)
            elif x_tan.dim() == 3: rotated_x_tan = torch.einsum('ij,bnj->bni', R_matrix_this_step, x_tan)
            else: rotated_x_tan = x_tan; logger_wubu_integrated.warning_once(f"MicroTrans apply_rot: Unexpected x_tan dim {x_tan.dim()}. Skip.")
        elif isinstance(self.rotation_generator_matrix, nn.Identity): rotated_x_tan = x_tan
        else: rotated_x_tan = x_tan; logger_wubu_integrated.error_once("Rot module misconfigured.")
        return rotated_x_tan, R_matrix_this_step

    def _apply_non_rotational_map(self, x_in: torch.Tensor, original_main_tan: torch.Tensor, micro_level_idx: Optional[int], stage_idx: Optional[int]) -> torch.Tensor:
        if not self.non_rotational_map_layers or isinstance(self.non_rotational_map_layers[0], nn.Identity):
            return x_in
        x_processed = x_in # Use x_in which is already rotated_x_intermediate
        first_linear_layer = self.non_rotational_map_layers[0]
        if self.enable_hypernetwork_modulation and self.hyper_bias_generator is not None and isinstance(first_linear_layer, nn.Linear):
            dynamic_bias_offset = self.hyper_bias_generator(original_main_tan) # Modulate based on pre-ISP, pre-rotation main_tan
            if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 10 == 0:
                print_tensor_stats(dynamic_bias_offset, "Hyper_BiasOffset", micro_level_idx, stage_idx)
            effective_bias = (first_linear_layer.bias + dynamic_bias_offset * self.hyper_mod_strength) if first_linear_layer.bias is not None else (dynamic_bias_offset * self.hyper_mod_strength)
            # Apply first linear layer out-of-place
            x_processed = F.linear(x_processed, first_linear_layer.weight, effective_bias)
            # Apply subsequent layers
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
        
        x_for_hypernet_mod = main_tan # Use original input for hypernet modulation source
        
        x_intermediate = self._apply_internal_quaternion_sub_processing(main_tan, micro_level_idx, stage_idx) if self.enable_internal_sub_processing else main_tan
        print_tensor_stats(x_intermediate, "MF_PostISP", micro_level_idx, stage_idx)
        
        rotated_x, R_matrix = self.apply_rotation(x_intermediate)
        print_tensor_stats(rotated_x, "MF_PostRot", micro_level_idx, stage_idx)
        if R_matrix is not None: print_tensor_stats(R_matrix, "MF_R_matrix", micro_level_idx, stage_idx)
        
        mapped_x = self._apply_non_rotational_map(rotated_x, x_for_hypernet_mod, micro_level_idx, stage_idx)
        print_tensor_stats(mapped_x, "MF_PostMap", micro_level_idx, stage_idx)
        
        next_state, final_tan = current_internal_state_in, mapped_x
        if self.enable_stateful_micro_transform and current_internal_state_in is not None and self.state_influence_mlp and self.state_update_gate_mlp:
            print_tensor_stats(current_internal_state_in, "STF_In_State", micro_level_idx, stage_idx)
            influence = self.state_influence_mlp(current_internal_state_in)
            print_tensor_stats(influence, "STF_Influence", micro_level_idx, stage_idx)
            final_tan = mapped_x + influence # Apply state influence here
            
            # Update state based on the state-influenced tangent vector
            state_update_in = torch.cat([final_tan, current_internal_state_in], dim=-1)
            delta = self.state_update_gate_mlp(state_update_in)
            print_tensor_stats(delta, "STF_Delta", micro_level_idx, stage_idx)
            next_state = torch.tanh(current_internal_state_in + delta) # Additive update is fine, tanh keeps it bounded
            print_tensor_stats(next_state, "STF_Out_State", micro_level_idx, stage_idx)

        if scaffold_modulation is not None:
            mod = scaffold_modulation
            if final_tan.dim()>1 and mod.dim()==1 and self.dim>0: mod=mod.unsqueeze(0).expand_as(final_tan)
            elif final_tan.dim()==1 and mod.dim()==2 and mod.shape[0]==1 and self.dim>0: mod=mod.squeeze(0)
            if final_tan.shape==mod.shape: final_tan += mod
            else: logger_wubu_integrated.debug_once("Scaffold mod shape mismatch.")
        print_tensor_stats(final_tan, "MF_PostScaff", micro_level_idx, stage_idx)
        
        clamped_tan = torch.clamp(final_tan, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        print_tensor_stats(clamped_tan, "MF_Out_Clamp", micro_level_idx, stage_idx)
        if DEBUG_MICRO_TRANSFORM_INTERNALS and self.debug_counter % 20 == 0: print("-" * 20)
        return clamped_tan, R_matrix, next_state


class FractalDepthQuaternionWuBuRungs(nn.Module):
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
                 scaffold_co_rotation_mode: str = "matrix_only",
                 enable_internal_sub_processing: bool = True,
                 enable_stateful_micro_transform: bool = True,
                 enable_hypernetwork_modulation: bool = True,
                 internal_state_dim_factor: float = 1.0,
                 hyper_mod_strength: float = 0.01
                 ):
        super().__init__()
        self.logger_fractal_rungs = logger_wubu_integrated
        self.num_virtual_micro_levels = num_virtual_micro_levels
        self.base_micro_level_dim = base_micro_level_dim
        self.enable_stateful_micro_transform_globally = enable_stateful_micro_transform

        if self.base_micro_level_dim != 4 and use_quaternion_so4_micro:
            self.logger_fractal_rungs.warning("use_quaternion_so4_micro=True but base_dim!=4. Forced off.")
            use_quaternion_so4_micro = False
        if self.base_micro_level_dim != 4 and enable_internal_sub_processing:
            self.logger_fractal_rungs.warning("enable_internal_sub_processing=True but base_dim!=4. Forced off.")
            enable_internal_sub_processing = False

        self.num_physical_transform_stages = max(1, num_physical_transform_stages)
        self.micro_levels_per_stage = max(1, self.num_virtual_micro_levels // self.num_physical_transform_stages)
        self.initial_s, self.s_decay_factor = initial_s, s_decay_factor_per_micro_level
        self.initial_c_base, self.c_phi_influence = initial_c_base, c_phi_influence
        self.min_curvature, self.t_tilde_activation_scale = EPS, t_tilde_activation_scale
        self.scaffold_co_rotation_mode = scaffold_co_rotation_mode

        if use_quaternion_so4_micro and scaffold_co_rotation_mode == "matrix_only":
            self.logger_fractal_rungs.warning_once("Scaffold co-rotation 'matrix_only' may not work with quat micro-transforms if they don't output R_matrix.")

        self.input_projection = nn.Linear(input_dim, self.base_micro_level_dim) if input_dim != self.base_micro_level_dim and input_dim > 0 and self.base_micro_level_dim > 0 else nn.Identity()
        self.input_layernorm = nn.LayerNorm(self.base_micro_level_dim) if self.base_micro_level_dim > 0 else nn.Identity()

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
        self.log_stage_curvatures_unconstrained = nn.ParameterList(
            [nn.Parameter(torch.tensor(math.log(math.expm1(max(EPS*10,initial_c_base)))))
             for _ in range(self.num_physical_transform_stages)]
        )
        self.num_phi_scaffold_points_per_stage = num_phi_scaffold_points_per_stage
        if self.num_phi_scaffold_points_per_stage > 0 and self.base_micro_level_dim > 0:
            self.phi_scaffold_base_tangent_vectors = nn.ParameterList([
                nn.Parameter(torch.randn(num_phi_scaffold_points_per_stage, self.base_micro_level_dim) * phi_scaffold_init_scale_factor)
                for _ in range(self.num_physical_transform_stages)])
        else: self.phi_scaffold_base_tangent_vectors = None
        self.use_gaussian_rungs = use_gaussian_rungs
        self.base_gaussian_std_dev_factor_rung, self.gaussian_std_dev_decay_factor_rung = base_gaussian_std_dev_factor_rung, gaussian_std_dev_decay_factor_rung
        self.rung_affinity_temperature, self.rung_modulation_strength = max(EPS, rung_affinity_temperature), rung_modulation_strength
        self.output_projection = nn.Linear(self.base_micro_level_dim, output_dim) if self.base_micro_level_dim != output_dim and self.base_micro_level_dim > 0 and output_dim > 0 else nn.Identity()
        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger_fractal_rungs.info(f"FDQRungs Init: {self.num_virtual_micro_levels} virtLvl ({self.base_micro_level_dim}D) "
                                 f"over {self.num_physical_transform_stages} physStgs. Params: {param_count:,}. "
                                 f"AdvMicro: SubP={enable_internal_sub_processing}, StateF={self.enable_stateful_micro_transform_globally}, HyperM={enable_hypernetwork_modulation}")

    def get_s_c_gsigma_at_micro_level(self, micro_level_idx: int, stage_idx: int) -> Tuple[float, float, float]:
        s_i = self.initial_s * (self.s_decay_factor ** micro_level_idx)
        s_i = max(EPS, s_i)
        stage_c_unconstrained = self.log_stage_curvatures_unconstrained[stage_idx]
        c_i = get_constrained_param_val(stage_c_unconstrained, self.min_curvature).item()
        if self.c_phi_influence and self.micro_levels_per_stage > 0 :
            micro_idx_in_stage = micro_level_idx % self.micro_levels_per_stage
            phi_exp = (micro_idx_in_stage % 4) - 1.5
            c_i *= (PHI ** phi_exp)
        c_i = max(self.min_curvature, c_i)
        sigma_gauss_i_skin = (self.base_gaussian_std_dev_factor_rung / max(s_i, EPS)) * \
                             (self.gaussian_std_dev_decay_factor_rung ** micro_level_idx)
        sigma_gauss_i_skin = max(EPS*100, sigma_gauss_i_skin)
        return s_i, c_i, sigma_gauss_i_skin

    def _propagate_scaffold_points(self, base_scaffold_tan_vectors_stage: torch.Tensor,
                                   accumulated_R_matrix_stage: Optional[torch.Tensor],
                                   current_s_i: float, initial_s_for_stage: float) -> torch.Tensor:
        propagated_scaffold = base_scaffold_tan_vectors_stage
        if accumulated_R_matrix_stage is not None and self.scaffold_co_rotation_mode == "matrix_only":
            propagated_scaffold = torch.einsum('ij,kj->ki', accumulated_R_matrix_stage.to(propagated_scaffold.device, propagated_scaffold.dtype), propagated_scaffold)
        return propagated_scaffold * (initial_s_for_stage / max(current_s_i, EPS))

    def forward(self, x_input: torch.Tensor, show_progress: bool = False, progress_desc: str = "FractalLevels") -> torch.Tensor:
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
        else: raise ValueError(f"FDQRungs forward expects 2D/3D input, got {input_original_dim_fwd}D.")

        if self.base_micro_level_dim == 0:
             out_feat_zero_dim = self.output_projection(current_v_tan_fwd)
             if input_original_dim_fwd == 3 and B_orig_fwd!=-1: return out_feat_zero_dim.reshape(B_orig_fwd, S_orig_fwd, -1 if out_feat_zero_dim.shape[-1]>0 else 0)
             return out_feat_zero_dim

        proj_in_feats = self.input_projection.in_features if hasattr(self.input_projection, 'in_features') else -1
        if not isinstance(self.input_projection, nn.Identity) and current_v_tan_fwd.shape[-1] != proj_in_feats :
            raise ValueError(f"Input dim {current_v_tan_fwd.shape[-1]} mismatch with input_projection {proj_in_feats}")

        current_v_tan_fwd = self.input_projection(current_v_tan_fwd)
        current_v_tan_fwd = self.input_layernorm(current_v_tan_fwd)

        accumulated_R_for_stage: Optional[torch.Tensor] = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype) if self.base_micro_level_dim > 0 else None
        s_at_stage_start_val: float = self.initial_s
        batched_internal_micro_state: Optional[torch.Tensor] = None
        # Initialize state based on the first physical_micro_transform's internal_state_dim
        if self.enable_stateful_micro_transform_globally and self.base_micro_level_dim > 0 and \
           len(self.physical_micro_transforms) > 0:
            first_micro_transform = self.physical_micro_transforms[0]
            if hasattr(first_micro_transform, 'enable_stateful_micro_transform') and \
               first_micro_transform.enable_stateful_micro_transform and first_micro_transform.internal_state_dim > 0:
                batched_internal_micro_state = torch.zeros(current_batch_size, first_micro_transform.internal_state_dim,
                                                           device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)

        micro_level_iterator = range(self.num_virtual_micro_levels)
        tqdm_iterator = None
        if show_progress:
            try: from tqdm import tqdm; tqdm_iterator = tqdm(micro_level_iterator, desc=progress_desc, total=self.num_virtual_micro_levels, leave=False, dynamic_ncols=True, disable=not show_progress)
            except ImportError: logger_wubu_integrated.warning_once("TQDM not found. Progress bar disabled.")
            if tqdm_iterator: micro_level_iterator = tqdm_iterator


        for micro_i_fwd in micro_level_iterator:
            current_stage_idx_fwd = micro_i_fwd // self.micro_levels_per_stage
            micro_idx_in_stage_fwd = micro_i_fwd % self.micro_levels_per_stage
            physical_transform_module_fwd = self.physical_micro_transforms[current_stage_idx_fwd]

            if micro_idx_in_stage_fwd == 0: # Start of a new physical stage's processing block
                if self.base_micro_level_dim > 0: accumulated_R_for_stage = torch.eye(self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype)
                s_at_stage_start_val, _, _ = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)
                # Reset/re-initialize batched_internal_micro_state for the new physical stage
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

        if tqdm_iterator: tqdm_iterator.close()
        output_features_fwd = self.output_projection(current_v_tan_fwd)
        if input_original_dim_fwd == 3 and B_orig_fwd!=-1:
            final_out_dim = output_features_fwd.shape[-1] if output_features_fwd.numel() > 0 else 0
            return output_features_fwd.reshape(B_orig_fwd, S_orig_fwd, final_out_dim)
        return output_features_fwd

def analyze_model_parameters(model: nn.Module):
    print("\n--- Model Parameter Analysis ---")
    total_params = 0
    param_counts_by_major_module = defaultdict(int)
    adv_micro_params_totals = defaultdict(int) # Totals for advanced components across all stages

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num_params = param.numel()
        total_params += num_params
        parts = name.split('.')

        if parts[0] == "physical_micro_transforms":
            param_counts_by_major_module["physical_micro_transforms_total"] += num_params
            # Check for advanced components within a physical_micro_transform stage
            # e.g., physical_micro_transforms.0.w_modulator.0.weight -> w_modulator
            # e.g., physical_micro_transforms.0.v_scaler_param -> v_scaler_param
            component_name_in_micro_transform = parts[2] # e.g. 'w_modulator', 'v_scaler_param', etc.
            if component_name_in_micro_transform == "w_modulator": adv_micro_params_totals["w_modulator_total"] += num_params
            elif component_name_in_micro_transform == "v_scaler_param": adv_micro_params_totals["v_scaler_param_total"] += num_params
            elif component_name_in_micro_transform == "state_update_gate_mlp": adv_micro_params_totals["state_update_gate_mlp_total"] += num_params
            elif component_name_in_micro_transform == "state_influence_mlp": adv_micro_params_totals["state_influence_mlp_total"] += num_params
            elif component_name_in_micro_transform == "hyper_bias_generator": adv_micro_params_totals["hyper_bias_generator_total"] += num_params
            elif component_name_in_micro_transform == "rotation_param_generator": adv_micro_params_totals["rotation_param_generator_total"] += num_params
            elif component_name_in_micro_transform == "rotation_generator_matrix" and parts[3] == "params": adv_micro_params_totals["rotation_generator_matrix_total"] += num_params
            elif component_name_in_micro_transform == "non_rotational_map_layers": adv_micro_params_totals["non_rotational_map_main_total"] += num_params
            # else: it's a parameter of a sub-module already categorized (e.g. w_modulator.0.weight)
        elif parts[0] == "input_projection": param_counts_by_major_module["input_projection"] += num_params
        elif parts[0] == "input_layernorm": param_counts_by_major_module["input_layernorm"] += num_params
        elif parts[0] == "output_projection": param_counts_by_major_module["output_projection"] += num_params
        elif parts[0] == "log_stage_curvatures_unconstrained": param_counts_by_major_module["log_stage_curvatures"] += num_params
        elif parts[0] == "phi_scaffold_base_tangent_vectors": param_counts_by_major_module["phi_scaffolds"] += num_params
        else: param_counts_by_major_module[f"other_{parts[0]}"] += num_params

    print(f"Total Trainable Parameters: {total_params:,}")
    print("Parameters by Major Module Type:")
    for name, count in sorted(param_counts_by_major_module.items()): print(f"  {name}: {count:,}")
    if adv_micro_params_totals:
        print("Total Parameters within ALL Physical Micro-Transforms (Aggregated Advanced Components):")
        for name, count in sorted(adv_micro_params_totals.items()): print(f"  {name}: {count:,}")
    print("--- End Parameter Analysis ---")

# Example Usage
if __name__ == '__main__':
    DEBUG_MICRO_TRANSFORM_INTERNALS = False
    logger_wubu_integrated.setLevel(logging.INFO)

    print("\n--- Testing FractalDepthQuaternionWuBuRungs (Advanced Intra-4D) ---")
    batch_size, in_dim_test, out_dim_test = 4, 16, 8
    num_virtual_levels_target, num_physical_stages_target = 100, 5

    adv_config = {"enable_internal_sub_processing": True, "enable_stateful_micro_transform": True, "enable_hypernetwork_modulation": True}
    print(f"Advanced Features Config: {adv_config}")

    fractal_model_config_adv = {
        "input_dim": in_dim_test, "output_dim": out_dim_test,
        "num_virtual_micro_levels": num_virtual_levels_target,
        "base_micro_level_dim": 4, "num_physical_transform_stages": num_physical_stages_target,
        "initial_s": 1.0, "s_decay_factor_per_micro_level": 0.99,
        "initial_c_base": 0.1, "c_phi_influence": True,
        "num_phi_scaffold_points_per_stage": 2, "phi_scaffold_init_scale_factor": 0.01,
        "use_gaussian_rungs": True, "base_gaussian_std_dev_factor_rung": 0.01,
        "gaussian_std_dev_decay_factor_rung": 0.999,
        "rung_affinity_temperature": 0.05, "rung_modulation_strength": 0.005,
        "t_tilde_activation_scale": 1.0, "micro_transform_type": "mlp",
        "micro_transform_hidden_factor": 0.5, "use_quaternion_so4_micro": True,
        "scaffold_co_rotation_mode": "none",
        "internal_state_dim_factor": 0.5, "hyper_mod_strength": 0.01, **adv_config
    }
    import time
    try: from tqdm import tqdm; TQDM_AVAILABLE = True
    except ImportError: TQDM_AVAILABLE = False; print("TQDM library not found. Progress bar will not be shown.")
    device_test = torch.device("cpu")
    print(f"Running test on device: {device_test}")
    print("Instantiating model with advanced features...")
    fractal_wubu_model_adv = FractalDepthQuaternionWuBuRungs(**fractal_model_config_adv).to(device_test)

    analyze_model_parameters(fractal_wubu_model_adv)

    test_input_2d_adv = torch.randn(batch_size, in_dim_test, device=device_test)
    dummy_target = torch.randn(batch_size, out_dim_test, device=device_test)
    print("\nStarting forward pass...")
    fractal_wubu_model_adv.train()
    start_time_fwd_2d_adv = time.time()
    output_2d_adv = fractal_wubu_model_adv(test_input_2d_adv, show_progress=TQDM_AVAILABLE, progress_desc="AdvTestFWD")
    end_time_fwd_2d_adv = time.time()
    print(f"Forward pass completed in {end_time_fwd_2d_adv - start_time_fwd_2d_adv:.3f} seconds.")
    print(f"Output shape: {output_2d_adv.shape}, Target shape: {dummy_target.shape}")
    assert output_2d_adv.shape == dummy_target.shape

    print("\nCalculating dummy loss and performing backward pass...")
    optimizer = torch.optim.SGD(fractal_wubu_model_adv.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = F.mse_loss(output_2d_adv, dummy_target)
    loss.backward()
    print(f"Dummy MSE Loss: {loss.item():.4f}")

    print("\n--- Gradient Check for Advanced Micro-Transform Parameters ---")
    found_any_adv_grads_overall = False
    for i, micro_transform_stage in enumerate(fractal_wubu_model_adv.physical_micro_transforms):
        print(f"  Physical Stage {i}:")
        adv_components_to_check = {
            "w_modulator": micro_transform_stage.w_modulator,
            "v_scaler_param": micro_transform_stage.v_scaler_param,
            "state_update_gate_mlp": micro_transform_stage.state_update_gate_mlp,
            "state_influence_mlp": micro_transform_stage.state_influence_mlp,
            "hyper_bias_generator": micro_transform_stage.hyper_bias_generator
        }
        stage_had_any_significant_grad = False
        for comp_name, component in adv_components_to_check.items():
            if component is None: continue
            
            component_itself_had_grad = False
            if isinstance(component, nn.Parameter):
                if component.grad is not None and torch.linalg.norm(component.grad.float()).item() > EPS:
                    print(f"    {comp_name} (Direct Param): Grad norm = {torch.linalg.norm(component.grad.float()).item():.3e}")
                    stage_had_any_significant_grad = True; component_itself_had_grad = True
            elif isinstance(component, nn.Module) and list(component.parameters()):
                params_in_submodule_with_grad = False
                for sub_param_name, param in component.named_parameters():
                    if param.grad is not None and torch.linalg.norm(param.grad.float()).item() > EPS:
                        print(f"      {comp_name}.{sub_param_name}: Grad norm = {torch.linalg.norm(param.grad.float()).item():.3e}")
                        stage_had_any_significant_grad = True; component_itself_had_grad = True; params_in_submodule_with_grad = True
                if not params_in_submodule_with_grad:
                    print(f"    {comp_name} (Module): No significant gradients in its parameters.")
            
            if not component_itself_had_grad and (isinstance(component, nn.Parameter) or (isinstance(component, nn.Module) and list(component.parameters()))):
                 print(f"    {comp_name}: No significant gradient detected for this component or its parameters.")
            
            if component_itself_had_grad: found_any_adv_grads_overall = True # If any component in any stage has grad
        
        if not stage_had_any_significant_grad:
             print(f"    No significant gradients found for ANY advanced components in this stage.")


    if found_any_adv_grads_overall: print("SUCCESS: Significant gradients found in at least one advanced micro-transform component.")
    else: print("WARNING: No significant gradients detected in params of advanced micro-transform components.")
    print("\n--- Basic Test Finished ---")