# WuBu_Integrated_Full.py
# Consolidating Fractal Depth, Gaussian Rungs, Phi-Scaffolds,
# and Quaternion SO(4) / Generalized SO(n) Rotations.
# Based on concepts by WuBu, Transcribed & Elaborated by Gemini
# Version: 2025-05-20.03 - Fully Integrated with Robust Geo Utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# --- Constants ---
EPS = 1e-5  # Small epsilon for numerical stability
PHI = (1 + math.sqrt(5)) / 2 # The Golden Ratio
TAN_VEC_CLAMP_VAL = 1e4  # Max norm for tangent vectors before expmap
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e8 # Max squared norm for poincare_clip
MIN_WUBU_LEVEL_SCALE = EPS # Min scale for sigmoid-scaled parameterization
MAX_WUBU_LEVEL_SCALE = 10.0 # Max scale for sigmoid-scaled parameterization

# --- Basic Logging Setup ---
logger_wubu_integrated = logging.getLogger("WuBuFractalIntegratedFull")
if not logger_wubu_integrated.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_wubu_integrated.addHandler(handler)
    logger_wubu_integrated.setLevel(logging.INFO) # Default INFO, can be changed

# --- Utility Functions (Essential for Initialization & Params) ---
def init_weights_general(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        if m.dim() > 1: nn.init.xavier_uniform_(m.data) # Use .data for parameters
        else: nn.init.normal_(m.data, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', True)):
            if hasattr(m, 'weight') and m.weight is not None: nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor:
    return F.softplus(param_unconstrained) + min_val

# --- Quaternion Math (Essential) ---
def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    if angle_rad.dim() == 0: angle_rad = angle_rad.unsqueeze(0)
    if axis.dim() == 1 and angle_rad.shape[0] > 1 : axis = axis.unsqueeze(0).expand(angle_rad.shape[0], -1)
    elif axis.dim() == 1 and angle_rad.shape[0] == 1: axis = axis.unsqueeze(0)
    elif axis.dim() == 0 and angle_rad.shape[0] == 1:
        axis_normalized = torch.zeros(angle_rad.shape[0], 3, device=axis.device, dtype=axis.dtype)
        if axis_normalized.numel() > 0: axis_normalized[:,0] = 1.0
    else:
        axis_normalized = F.normalize(axis, p=2, dim=-1)
    
    angle_half = angle_rad / 2.0
    q_w = torch.cos(angle_half).unsqueeze(-1)
    q_xyz = axis_normalized * torch.sin(angle_half).unsqueeze(-1)
    return torch.cat([q_w, q_xyz], dim=-1)

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Ensure q1 and q2 are broadcastable for batched operations.
    # If q1 is (B, N, 4) and q2 is (N, 4), q2 needs to be (1, N, 4) or (B, N, 4)
    # This simple broadcasting might not cover all cases, depends on expected input shapes.
    # Assuming inputs are either (..., 4) and (..., 4) with compatible ...
    # or one is (4,) and the other is (B, ..., 4).
    
    q1_dim = q1.dim()
    q2_dim = q2.dim()

    if q1_dim < q2_dim and q1.shape[-1] == 4:
        # Try to unsqueeze q1 to match q2's batch dimensions
        num_unsqueeze = q2_dim - q1_dim
        q1_expanded = q1
        for _ in range(num_unsqueeze):
            q1_expanded = q1_expanded.unsqueeze(0)
        if q1_expanded.shape[:-1] != q2.shape[:-1]: # Check if broadcastable after unsqueeze
             q1_expanded = q1.unsqueeze(0).expand_as(q2) # Fallback to direct expand
        q1 = q1_expanded
    elif q2_dim < q1_dim and q2.shape[-1] == 4:
        num_unsqueeze = q1_dim - q2_dim
        q2_expanded = q2
        for _ in range(num_unsqueeze):
            q2_expanded = q2_expanded.unsqueeze(0)
        if q2_expanded.shape[:-1] != q1.shape[:-1]:
            q2_expanded = q2.unsqueeze(0).expand_as(q1)
        q2 = q2_expanded

    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.cat([w, x, y, z], dim=-1)

def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)

# --- Hyperbolic Geometry Utilities (Your Robust Versions) ---
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
        if c_scalar <= 0: 
            self.c = 0.0; self.sqrt_c = 0.0; self.radius = float('inf')
        else: 
            self.c = c_scalar; self.sqrt_c = math.sqrt(self.c); self.radius = 1.0 / (self.sqrt_c + EPS)
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
        if n_dim <= 1:
            self.num_params = 0
            self.register_parameter('params', None)
        else:
            self.num_params = n_dim * (n_dim - 1) // 2
            self.params = nn.Parameter(torch.randn(self.num_params) * 0.001) 

    def forward(self) -> torch.Tensor:
        dev = self.params.device if self.params is not None else torch.device('cpu')
        dtype = self.params.dtype if self.params is not None else torch.float32
        if self.n_dim <= 1 or self.params is None:
            return torch.eye(self.n_dim, device=dev, dtype=dtype)
        X = torch.zeros(self.n_dim, self.n_dim, device=dev, dtype=dtype)
        indices = torch.triu_indices(self.n_dim, self.n_dim, offset=1, device=dev)
        if X.numel() > 0 and indices.numel() > 0 and self.params.numel() > 0: # Check if params can fill indices
             if indices.shape[1] == self.params.shape[0]: # Correct number of upper triangle elements
                X[indices[0], indices[1]] = self.params
             else:
                logger_wubu_integrated.warning_once(f"SkewSymmetricMatrix: Mismatch between triu_indices ({indices.shape[1]}) and params ({self.params.shape[0]}) for dim {self.n_dim}. Filling partially or not at all.")
                # Handle partial fill if possible, or log error and return identity
                num_to_fill = min(indices.shape[1], self.params.shape[0])
                if num_to_fill > 0:
                    X[indices[0,:num_to_fill], indices[1,:num_to_fill]] = self.params[:num_to_fill]


        X = X - X.T
        try:
            R = torch.linalg.matrix_exp(X) 
        except Exception as e:
            logger_wubu_integrated.error_once(f"torch.linalg.matrix_exp error: {e}. Using identity.", exc_info=True)
            R = torch.eye(self.n_dim, device=X.device, dtype=X.dtype)
        return R

class QuaternionSO4From6Params(nn.Module):
    def __init__(self, small_init_std: float = 0.001):
        super().__init__()
        self.so3_params_for_p = nn.Parameter(torch.randn(3) * small_init_std)
        self.so3_params_for_q = nn.Parameter(torch.randn(3) * small_init_std)

    def _so3_params_to_unit_quaternion(self, so3_params: torch.Tensor) -> torch.Tensor:
        angle = torch.norm(so3_params, p=2, dim=-1)
        if angle < EPS:
            identity_q = torch.zeros(4, device=so3_params.device, dtype=so3_params.dtype)
            identity_q[0] = 1.0
            return identity_q
        axis = so3_params / angle
        return quaternion_from_axis_angle(angle.unsqueeze(0), axis.unsqueeze(0)).squeeze(0)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p_unit_quat = self._so3_params_to_unit_quaternion(self.so3_params_for_p)
        q_unit_quat = self._so3_params_to_unit_quaternion(self.so3_params_for_q)
        return p_unit_quat, q_unit_quat

class FractalMicroTransformRungs(nn.Module):
    def __init__(self, dim: int = 4, transform_type: str = "mlp", 
                 hidden_dim_factor: float = 1.0, dropout: float = 0.0,
                 use_quaternion_so4: bool = True):
        super().__init__()
        self.dim = dim
        self.use_quaternion_so4 = use_quaternion_so4 and (dim == 4)

        if self.use_quaternion_so4:
            self.rotation_param_generator = QuaternionSO4From6Params()
            self.rotation_generator_matrix = None # Not used if quat path active
        elif dim > 0:
            self.rotation_generator_matrix = SkewSymmetricMatrix(dim)
            self.rotation_param_generator = None # Not used if matrix path active
        else:
            self.rotation_generator_matrix = nn.Identity() # type: ignore
            self.rotation_param_generator = None # type: ignore
            
        mlp_hidden_dim = max(dim, int(dim * hidden_dim_factor)) if dim > 0 else 0
        if transform_type == 'mlp' and dim > 0 and mlp_hidden_dim > 0:
            self.non_rotational_map = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim), nn.GELU(),
                nn.Linear(mlp_hidden_dim, dim)
            )
        elif transform_type == 'linear' and dim > 0:
             self.non_rotational_map = nn.Linear(dim, dim)
        else:
            self.non_rotational_map = nn.Identity()
        
        self.apply(init_weights_general)

    def apply_rotation(self, x_tan: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        R_matrix_this_step: Optional[torch.Tensor] = None
        
        if self.use_quaternion_so4 and self.rotation_param_generator is not None:
            p_quat, q_quat = self.rotation_param_generator()
            p_quat = p_quat.to(x_tan.device, x_tan.dtype) # Ensure same device and dtype
            q_quat = q_quat.to(x_tan.device, x_tan.dtype)

            if x_tan.dim() == 1: x_tan_b = x_tan.unsqueeze(0)
            else: x_tan_b = x_tan
            
            p_b = p_quat if p_quat.dim() == x_tan_b.dim() else p_quat.unsqueeze(0).expand_as(x_tan_b)
            q_b = q_quat if q_quat.dim() == x_tan_b.dim() else q_quat.unsqueeze(0).expand_as(x_tan_b)
            
            px = quaternion_multiply(p_b, x_tan_b)
            pxq = quaternion_multiply(px, q_b)
            
            rotated_x_tan = pxq.squeeze(0) if x_tan.dim() == 1 else pxq
            # R_matrix_this_step remains None for quaternion path currently
        elif self.rotation_generator_matrix is not None and isinstance(self.rotation_generator_matrix, SkewSymmetricMatrix):
            R_matrix_this_step = self.rotation_generator_matrix()
            if x_tan.dim() == 2: rotated_x_tan = torch.einsum('ij,bj->bi', R_matrix_this_step, x_tan)
            elif x_tan.dim() == 1: rotated_x_tan = torch.matmul(R_matrix_this_step, x_tan)
            elif x_tan.dim() == 3: rotated_x_tan = torch.einsum('ij,bnj->bni', R_matrix_this_step, x_tan)
            else: rotated_x_tan = x_tan
        elif isinstance(self.rotation_generator_matrix, nn.Identity):
            rotated_x_tan = x_tan
        else:
            rotated_x_tan = x_tan
            logger_wubu_integrated.error_once("FractalMicroTransformRungs: Rotation module misconfigured or None.")
        return rotated_x_tan, R_matrix_this_step

    def forward(self, 
                main_tan: torch.Tensor, 
                scaffold_modulation: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        rotated_main_tan, R_matrix_step = self.apply_rotation(main_tan)
        mapped_main_tan = self.non_rotational_map(rotated_main_tan)

        final_main_tan = mapped_main_tan
        if scaffold_modulation is not None:
            if main_tan.dim() == 2 and scaffold_modulation.dim() == 1 and self.dim > 0:
                 scaffold_modulation = scaffold_modulation.unsqueeze(0).expand_as(mapped_main_tan)
            elif main_tan.dim() == 1 and scaffold_modulation.dim() == 2 and scaffold_modulation.shape[0] == 1 and self.dim > 0:
                 scaffold_modulation = scaffold_modulation.squeeze(0)
            
            if mapped_main_tan.shape == scaffold_modulation.shape:
                final_main_tan = mapped_main_tan + scaffold_modulation
            else:
                 logger_wubu_integrated.debug_once(f"MicroTransform: Scaffold modulation shape {scaffold_modulation.shape} "
                                               f"mismatched with mapped_main_tan {mapped_main_tan.shape}. Skipping modulation.")
            
        clamped_final_tan = torch.clamp(final_main_tan, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        return clamped_final_tan, R_matrix_step


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
                 scaffold_co_rotation_mode: str = "matrix_only" 
                 ):
        super().__init__()
        self.logger_fractal_rungs = logger_wubu_integrated
        self.num_virtual_micro_levels = num_virtual_micro_levels
        self.base_micro_level_dim = base_micro_level_dim
        if self.base_micro_level_dim != 4 and use_quaternion_so4_micro:
            self.logger_fractal_rungs.warning("use_quaternion_so4_micro=True but base_dim!=4. Quat SO(4) path forced off in FractalMicroTransformRungs.")
            use_quaternion_so4_micro = False 

        self.num_physical_transform_stages = max(1, num_physical_transform_stages)
        self.micro_levels_per_stage = max(1, self.num_virtual_micro_levels // self.num_physical_transform_stages)
        
        self.initial_s = initial_s
        self.s_decay_factor = s_decay_factor_per_micro_level
        self.initial_c_base = initial_c_base
        self.c_phi_influence = c_phi_influence
        self.min_curvature = EPS
        self.t_tilde_activation_scale = t_tilde_activation_scale
        self.scaffold_co_rotation_mode = scaffold_co_rotation_mode
        
        if use_quaternion_so4_micro and scaffold_co_rotation_mode == "matrix_only":
            self.logger_fractal_rungs.warning("Scaffold co-rotation 'matrix_only' requested with quaternion micro-transforms. "
                                        "Scaffolds will NOT co-rotate via matrix if quaternion path is active in micro-transform "
                                        "(as R_matrix_from_micro_step will be None). Consider 'none' for scaffold_co_rotation_mode, "
                                        "or modify FractalMicroTransformRungs to output equivalent matrix for quaternion rotations.")
            
        self.input_projection = nn.Linear(input_dim, self.base_micro_level_dim) if input_dim != self.base_micro_level_dim and input_dim > 0 and self.base_micro_level_dim > 0 else nn.Identity()
        self.input_layernorm = nn.LayerNorm(self.base_micro_level_dim) if self.base_micro_level_dim > 0 else nn.Identity()

        self.physical_micro_transforms = nn.ModuleList(
            [FractalMicroTransformRungs(dim=self.base_micro_level_dim,
                                        transform_type=micro_transform_type,
                                        hidden_dim_factor=micro_transform_hidden_factor,
                                        use_quaternion_so4=use_quaternion_so4_micro)
             for _ in range(self.num_physical_transform_stages)]
        )
        self.log_stage_curvatures_unconstrained = nn.ParameterList(
            [nn.Parameter(torch.tensor(math.log(math.expm1(max(EPS,initial_c_base))))) 
             for _ in range(self.num_physical_transform_stages)]
        )
        
        self.num_phi_scaffold_points_per_stage = num_phi_scaffold_points_per_stage
        if self.num_phi_scaffold_points_per_stage > 0 and self.base_micro_level_dim > 0:
            self.phi_scaffold_base_tangent_vectors = nn.ParameterList([
                nn.Parameter(torch.randn(num_phi_scaffold_points_per_stage, self.base_micro_level_dim) * phi_scaffold_init_scale_factor)
                for _ in range(self.num_physical_transform_stages)
            ])
        else:
            self.phi_scaffold_base_tangent_vectors = None # type: ignore

        self.use_gaussian_rungs = use_gaussian_rungs
        self.base_gaussian_std_dev_factor_rung = base_gaussian_std_dev_factor_rung
        self.gaussian_std_dev_decay_factor_rung = gaussian_std_dev_decay_factor_rung
        self.rung_affinity_temperature = max(EPS, rung_affinity_temperature)
        self.rung_modulation_strength = rung_modulation_strength

        self.output_projection = nn.Linear(self.base_micro_level_dim, output_dim) if self.base_micro_level_dim != output_dim and self.base_micro_level_dim > 0 and output_dim > 0 else nn.Identity()
        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger_fractal_rungs.info(f"FractalDepthRungs Init: {self.num_virtual_micro_levels} virt Lvl ({self.base_micro_level_dim}D) "
                                 f"over {self.num_physical_transform_stages} phys Stages. Params: {param_count:,}. "
                                 f"QuatSO4: {use_quaternion_so4_micro}, Rungs: {self.use_gaussian_rungs}, Scaffolds: {self.num_phi_scaffold_points_per_stage > 0}, "
                                 f"Scaffold Co-Rot: {self.scaffold_co_rotation_mode}")

    def get_s_c_gsigma_at_micro_level(self, micro_level_idx: int, stage_idx: int) -> Tuple[float, float, float]:
        s_i = self.initial_s * (self.s_decay_factor ** micro_level_idx)
        s_i = max(EPS, s_i)
        stage_c_unconstrained = self.log_stage_curvatures_unconstrained[stage_idx]
        c_i = get_constrained_param_val(stage_c_unconstrained, self.min_curvature).item()
        if self.c_phi_influence:
            micro_idx_in_stage = micro_level_idx % self.micro_levels_per_stage
            phi_exp = (micro_idx_in_stage % 4) - 1.5 
            c_i *= (PHI ** phi_exp)
        c_i = max(self.min_curvature, c_i)
        sigma_gauss_i_skin = (self.base_gaussian_std_dev_factor_rung / s_i) * \
                             (self.gaussian_std_dev_decay_factor_rung ** micro_level_idx)
        sigma_gauss_i_skin = max(EPS*100, sigma_gauss_i_skin)
        return s_i, c_i, sigma_gauss_i_skin

    def _propagate_scaffold_points(self, 
                                   base_scaffold_tan_vectors_stage: torch.Tensor, 
                                   accumulated_R_matrix_stage: Optional[torch.Tensor], 
                                   current_s_i: float, initial_s_for_stage: float
                                  ) -> torch.Tensor:
        propagated_scaffold = base_scaffold_tan_vectors_stage
        if accumulated_R_matrix_stage is not None and self.scaffold_co_rotation_mode == "matrix_only":
            propagated_scaffold = torch.einsum('ij,kj->ki', accumulated_R_matrix_stage, propagated_scaffold)
        magnitude_scale_factor = initial_s_for_stage / max(current_s_i, EPS)
        scaled_propagated_scaffold_tan = propagated_scaffold * magnitude_scale_factor
        return scaled_propagated_scaffold_tan

    def forward(self, x_input: torch.Tensor, show_progress: bool = False, progress_desc: str = "FractalLevels") -> torch.Tensor: # Added show_progress
        input_original_dim_fwd = x_input.dim()
        B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = -1, -1, -1
        current_batch_size_effective: int

        if input_original_dim_fwd == 3:
            B_orig_fwd, S_orig_fwd, d_in_runtime_fwd = x_input.shape
            current_v_tan_fwd = x_input.reshape(B_orig_fwd * S_orig_fwd, d_in_runtime_fwd)
            # current_batch_size_effective = B_orig_fwd * S_orig_fwd # Not needed if batch_size is 1 for test
        elif input_original_dim_fwd == 2:
            # current_batch_size_effective, d_in_runtime_fwd = x_input.shape # Not needed if batch_size is 1
            current_v_tan_fwd = x_input
        else:
            raise ValueError(f"FractalDepthRungs forward expects 2D or 3D input, got {input_original_dim_fwd}D.")

        if self.base_micro_level_dim == 0: 
             out_feat_zero_dim = self.output_projection(current_v_tan_fwd)
             return out_feat_zero_dim.reshape(B_orig_fwd, S_orig_fwd, -1) if input_original_dim_fwd == 3 and B_orig_fwd != -1 else out_feat_zero_dim

        if not isinstance(self.input_projection, nn.Identity) and current_v_tan_fwd.shape[-1] != self.input_projection.in_features:
            raise ValueError(f"Input dim {current_v_tan_fwd.shape[-1]} mismatch with input_projection in_features {self.input_projection.in_features}")

        current_v_tan_fwd = self.input_projection(current_v_tan_fwd)
        current_v_tan_fwd = self.input_layernorm(current_v_tan_fwd)

        accumulated_R_for_stage: torch.Tensor = torch.eye(
            self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype
        )
        s_at_stage_start_val: float = self.initial_s
        
        # --- TQDM Integration ---
        micro_level_iterator = range(self.num_virtual_micro_levels)
        if show_progress:
            from tqdm import tqdm # Import tqdm locally if used
            # Disable tqdm if not main process in DDP, or if globally disabled
            # For this test, assuming single process, so always enable if show_progress is True.
            # In a DDP setup, only rank 0 should show tqdm.
            # is_main_process = (torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True)
            # disable_tqdm = not is_main_process or os.getenv('CI') == 'true' # Example
            micro_level_iterator = tqdm(micro_level_iterator, desc=progress_desc, 
                                        total=self.num_virtual_micro_levels,
                                        leave=False, dynamic_ncols=True,
                                        disable=not show_progress) # Redundant disable but clear

        for micro_i_fwd in micro_level_iterator: # Use the iterator
            current_stage_idx_fwd = micro_i_fwd // self.micro_levels_per_stage
            micro_idx_in_stage_fwd = micro_i_fwd % self.micro_levels_per_stage
            
            physical_transform_module_fwd = self.physical_micro_transforms[current_stage_idx_fwd]
            
            if micro_idx_in_stage_fwd == 0: 
                accumulated_R_for_stage = torch.eye(
                    self.base_micro_level_dim, device=current_v_tan_fwd.device, dtype=current_v_tan_fwd.dtype
                )
                s_at_stage_start_val, _, _ = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)

            s_i_fwd, _c_i_fwd, sigma_gauss_skin_i_fwd = self.get_s_c_gsigma_at_micro_level(micro_i_fwd, current_stage_idx_fwd)
            
            scaffold_modulation_input: Optional[torch.Tensor] = None
            if self.use_gaussian_rungs and self.phi_scaffold_base_tangent_vectors is not None and \
               self.num_phi_scaffold_points_per_stage > 0 and self.base_micro_level_dim > 0:
                
                base_scaffolds_for_current_stage = self.phi_scaffold_base_tangent_vectors[current_stage_idx_fwd].to(
                    current_v_tan_fwd.device, current_v_tan_fwd.dtype
                )
                
                R_for_scaffold_prop = accumulated_R_for_stage if self.scaffold_co_rotation_mode == "matrix_only" else None
                propagated_scaffold_tan_at_i = self._propagate_scaffold_points(
                    base_scaffolds_for_current_stage, R_for_scaffold_prop, 
                    s_i_fwd, s_at_stage_start_val 
                ) 

                diffs_v_scaffold = current_v_tan_fwd.unsqueeze(1) - propagated_scaffold_tan_at_i.unsqueeze(0)
                sq_distances_v_scaffold = torch.sum(diffs_v_scaffold**2, dim=-1)
                combined_var_denom_affinity = 2 * (sigma_gauss_skin_i_fwd**2) + EPS * 100
                
                affinity_scores_v_scaffold = torch.exp(-sq_distances_v_scaffold / combined_var_denom_affinity) 
                affinity_weights_v_scaffold = F.softmax(affinity_scores_v_scaffold / self.rung_affinity_temperature, dim=-1)
                
                weighted_scaffold_avg_influence = torch.einsum('bn,nd->bd', 
                                                              affinity_weights_v_scaffold, 
                                                              propagated_scaffold_tan_at_i)
                scaffold_modulation_input = self.rung_modulation_strength * weighted_scaffold_avg_influence

            transformed_v_tan_micro, R_matrix_from_micro_this_step = physical_transform_module_fwd(
                current_v_tan_fwd,
                scaffold_modulation=scaffold_modulation_input
            )
            
            if self.t_tilde_activation_scale != 1.0:
                transformed_v_tan_micro = transformed_v_tan_micro * self.t_tilde_activation_scale
            
            current_v_tan_fwd = transformed_v_tan_micro
            
            if R_matrix_from_micro_this_step is not None and self.scaffold_co_rotation_mode == "matrix_only":
                 accumulated_R_for_stage = torch.matmul(R_matrix_from_micro_this_step, accumulated_R_for_stage)

            # Reduced logging frequency for very deep loops, now controlled by tqdm
            if show_progress and isinstance(micro_level_iterator, tqdm): # Check if it's a tqdm iterator
                if micro_i_fwd > 0 and (micro_i_fwd % (max(1, self.num_virtual_micro_levels // 100)) == 0): # Log ~100 times
                    log_dict = {
                        "s": f"{s_i_fwd:.2e}",
                        "c": f"{_c_i_fwd:.2e}",
                        "σ_skin": f"{sigma_gauss_skin_i_fwd:.2e}"
                    }
                    if scaffold_modulation_input is not None and 'affinity_weights_v_scaffold' in locals() and affinity_weights_v_scaffold is not None:
                        log_dict["aff_w_max"] = f"{affinity_weights_v_scaffold.max().item():.2e}"
                    micro_level_iterator.set_postfix(log_dict)

        # Close tqdm progress bar if it was used
        if show_progress and isinstance(micro_level_iterator, tqdm):
            micro_level_iterator.close()

        final_v_tan_fwd = current_v_tan_fwd
        output_features_fwd = self.output_projection(final_v_tan_fwd)

        if input_original_dim_fwd == 3 and B_orig_fwd != -1:
            final_output_dim = output_features_fwd.shape[-1]
            return output_features_fwd.reshape(B_orig_fwd, S_orig_fwd, final_output_dim)
            
        return output_features_fwd
# Example Usage (Conceptual - requires robust PoincareBall & HyperbolicUtils)
if __name__ == '__main__':
    logger_wubu_integrated.setLevel(logging.INFO) 
    
    print("\n--- Testing FractalDepthQuaternionWuBuRungs (ONE MILLION Micro-Levels) ---")
    batch_size, in_dim_test, out_dim_test = 1, 64, 32 
    
    num_virtual_levels_target = 1000000
    num_physical_stages_target = 10 

    if num_virtual_levels_target % num_physical_stages_target != 0:
        # ... (adjust num_physical_stages_target logic as before) ...
        effective_micro_per_stage = num_virtual_levels_target // num_physical_stages_target # Integer division
        logger_wubu_integrated.info(
            f"Note: {num_virtual_levels_target} virtual levels with {num_physical_stages_target} physical stages. "
            f"Effective micro-levels per stage: {effective_micro_per_stage}. "
        )


    fractal_model_config_1M = {
        "input_dim": in_dim_test, 
        "output_dim": out_dim_test,
        "num_virtual_micro_levels": num_virtual_levels_target,
        "base_micro_level_dim": 4,
        "num_physical_transform_stages": num_physical_stages_target,
        "initial_s": 1.0, 
        "s_decay_factor_per_micro_level": 0.999999, 
        "initial_c_base": 0.1, 
        "c_phi_influence": True,
        "num_phi_scaffold_points_per_stage": 3,
        "phi_scaffold_init_scale_factor": 0.001,
        "use_gaussian_rungs": True, 
        "base_gaussian_std_dev_factor_rung": 0.005,
        "gaussian_std_dev_decay_factor_rung": 0.9999995,
        "rung_affinity_temperature": 0.01, 
        "rung_modulation_strength": 0.001,  
        "t_tilde_activation_scale": 1.00001, 
        "micro_transform_type": "mlp",       
        "micro_transform_hidden_factor": 0.5, 
        "use_quaternion_so4_micro": True,     
        "scaffold_co_rotation_mode": "none"   
    }
    
    print(f"Configuring for {num_virtual_levels_target} virtual levels with {num_physical_stages_target} physical transform stages.")
    actual_micro_per_stage = num_virtual_levels_target // num_physical_stages_target
    print(f"Each physical transform module will be effectively reused up to ~{actual_micro_per_stage} times (average).")
    
    import time
    # Try to import tqdm, if not available, progress bar won't show
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        print("TQDM library not found. Progress bar will not be shown.")
    
    device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running test on device: {device_test}")

    try:
        print("Instantiating MILLION-level model...")
        start_time_inst_1M = time.time()
        fractal_wubu_model_1M = FractalDepthQuaternionWuBuRungs(**fractal_model_config_1M).to(device_test) # type: ignore
        fractal_wubu_model_1M.eval() 
        end_time_inst_1M = time.time()
        print(f"MILLION-level Model instantiated in {end_time_inst_1M - start_time_inst_1M:.2f} seconds.")
        param_count_1M = sum(p.numel() for p in fractal_wubu_model_1M.parameters())
        print(f"  Actual parameter count: {param_count_1M:,}")

        test_input_2d_1M = torch.randn(batch_size, in_dim_test, device=device_test)
        print(f"Input 2D shape for MILLION test: {test_input_2d_1M.shape}")
        
        print("Starting forward pass (2D, MILLION levels)... THIS WILL TAKE A SIGNIFICANT AMOUNT OF TIME...")
        logger_wubu_integrated.info("Log level for WuBu module is INFO for this run.") # Keep this
        
        start_time_fwd_2d_1M = time.time()
        with torch.no_grad(): 
            # Pass show_progress=True if TQDM is available
            output_2d_1M = fractal_wubu_model_1M(test_input_2d_1M, show_progress=TQDM_AVAILABLE, progress_desc="MillionLevelsFWD")
        end_time_fwd_2d_1M = time.time()
        
        elapsed_time_fwd_2d_1M = end_time_fwd_2d_1M - start_time_fwd_2d_1M
        print(f"\nForward pass (2D, MILLION levels) completed in {elapsed_time_fwd_2d_1M:.2f} seconds.") # Added newline for tqdm
        if elapsed_time_fwd_2d_1M > 0 and num_virtual_levels_target > 0 and batch_size > 0:
             time_per_level_ms = (elapsed_time_fwd_2d_1M / num_virtual_levels_target / batch_size) * 1000
             print(f"  Approx. time per micro-level per sample: {time_per_level_ms:.6f} ms")
        
        print(f"Output 2D shape: {output_2d_1M.shape}, Expected: ({batch_size}, {out_dim_test})")
        assert output_2d_1M.shape == (batch_size, out_dim_test)
        
        output_sum_1M = torch.sum(output_2d_1M).item()
        print(f"Output 2D sum (check for NaN/Inf): {output_sum_1M}")
        if not torch.isfinite(output_2d_1M).all():
            print("ERROR: Output 2D (MILLION levels) contains NaN/Inf!")
            num_nans = torch.isnan(output_2d_1M).sum().item()
            num_infs = torch.isinf(output_2d_1M).sum().item()
            print(f"  NaNs: {num_nans}, Infs: {num_infs}")
            if num_nans > 0 or num_infs > 0: print("   First few elements of output:", output_2d_1M.flatten()[:10])
        else:
            print("Output 2D (MILLION levels) is finite.")
            print(f"   Output min: {output_2d_1M.min().item()}, max: {output_2d_1M.max().item()}, mean: {output_2d_1M.mean().item()}")
        
        print("\nFractalDepthQuaternionWuBuRungs (MILLION Micro-Levels) basic test finished.")

    except Exception as e:
        print(f"Error during FractalDepthQuaternionWuBuRungs (MILLION levels) test: {e}")
        import traceback
        traceback.print_exc()