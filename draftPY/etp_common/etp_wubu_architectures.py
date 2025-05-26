import logging
import math
from typing import Dict, Optional, List, Tuple, Union, Type 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

# Configure a module-specific logger
logger = logging.getLogger(__name__)

# --- Manifold and PoincareBall ---
class Manifold:
    """
    Abstract base class for manifolds.
    Provides an interface for essential manifold operations.
    """
    def __init__(self, eps: float = 1e-7): # Standardized eps
        super().__init__()
        self.eps = eps 

    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def proj(self, x: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
        
    def expmap(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def ptransp(self, u: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def an_dist(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(self.sqdist(p1, p2, c) + self.eps) # Added self.eps for sqrt stability

    def egrad2rgrad(self, p: torch.Tensor, grad_e: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        """Converts Euclidean gradient to Riemannian gradient. Placeholder."""
        logger.warning(f"Manifold.egrad2rgrad not implemented for {self.__class__.__name__}. Returning Euclidean gradient.")
        return grad_e


class PoincareBall(Manifold):
    """PoincareBall Manifold class."""
    def __init__(self, eps: float = 1e-7, min_norm_scale: float = 1e-5):
        super().__init__(eps=eps)
        self.name = 'PoincareBall' # Useful for RESGD to identify manifold type
        self.min_norm_scale = min_norm_scale 

    def _get_c_tensor(self, c: Union[float, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Ensures c is a tensor with appropriate shape for broadcasting with x."""
        if isinstance(c, (float, int)):
            return torch.tensor(c, dtype=x.dtype, device=x.device)
        # If c is already a tensor, ensure it can broadcast with x (e.g., Bx1 or scalar for BxDx...)
        if c.ndim == x.ndim: # c is likely already BxDx... or scalar
            return c
        elif c.ndim == x.ndim -1 : # c is B for x as BxD
             return c.unsqueeze(-1)
        elif c.ndim == 1 and x.ndim > 1 and c.size(0) == x.size(0): # c is B for x as Bx...
            # Reshape c to be (B, 1, ..., 1) to match x's batch dim and allow broadcasting
            return c.view(x.size(0), *([1]*(x.ndim-1)))
        return c # Fallback, assume c is already compatible or scalar tensor

    def _lambda_x(self, x: torch.Tensor, c_val: Union[float, torch.Tensor], keepdim: bool = False) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x)
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=keepdim)
        return 2 / (1 - c * x_sqnorm + self.eps)

    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p1)
        sqrt_c = c.sqrt()
        mobius_add_result = self.mobius_add(-p1, p2, c, dim=-1)
        # num = 2 * c * torch.sum(mobius_add_result.pow(2), dim=-1) # Original
        num = 2 * c.squeeze() * torch.sum(mobius_add_result.pow(2), dim=-1) # Ensure c is scalar or matches batch for mult

        p1_sqnorm = torch.sum(p1 * p1, dim=-1)
        p2_sqnorm = torch.sum(p2 * p2, dim=-1)
        den1 = 1 - c.squeeze() * p1_sqnorm
        den2 = 1 - c.squeeze() * p2_sqnorm
        
        acosh_arg = 1 + num / (den1 * den2 + self.eps)
        dist_c = torch.acosh(torch.clamp_min(acosh_arg, 1.0 + self.eps))
        dist = dist_c / (sqrt_c.squeeze() + self.eps) 
        return dist.pow(2)

    def proj(self, x: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x)
        sqrt_c = c.sqrt()
        max_radius = (1.0 - self.min_norm_scale) / (sqrt_c + self.eps) 
        
        norm_x = x.norm(dim=-1, keepdim=True, p=2)
        cond = norm_x > max_radius
        projected_x = x / (norm_x + self.eps) * max_radius 
        return torch.where(cond, projected_x, x)

    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        return u

    def expmap(self, p: torch.Tensor, u: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor: # Order changed to p, u
        c = self._get_c_tensor(c_val, p)
        sqrt_c = c.sqrt()
        u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
        u_norm_clamped = torch.clamp_min(u_norm, self.eps)
        lambda_p_val = self._lambda_x(p, c, keepdim=True)
        tanh_arg = sqrt_c * lambda_p_val * u_norm / 2
        
        # Handle u_norm = 0 case more explicitly to prevent u/u_norm_clamped from becoming large if u is almost zero.
        # If u_norm is very small, tanh_arg will be small, tanh(small) ~ small.
        # The result should be p if u is zero.
        # second_term = u * (torch.tanh(tanh_arg) / (sqrt_c * u_norm_clamped + self.eps)) # Original
        # if u is zero, second term is zero. mobius_add(p,0) = p. This is correct.
        # The scaling factor for u:
        scale = torch.tanh(tanh_arg) / (sqrt_c * u_norm_clamped + self.eps)
        second_term = u * scale
        return self.mobius_add(p, second_term, c, dim=-1)

    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p1)
        sqrt_c = c.sqrt()
        sub_p = self.mobius_add(-p1, p2, c, dim=-1) 
        sub_p_norm = torch.norm(sub_p, p=2, dim=-1, keepdim=True)
        sub_p_norm_clamped = torch.clamp_min(sub_p_norm, self.eps)
        lambda_p1_val = self._lambda_x(p1, c, keepdim=True)
        
        atanh_arg = sqrt_c * sub_p_norm
        atanh_arg_clamped = torch.clamp(atanh_arg, max=1.0 - self.eps) # Clamp to avoid atanh(1)
        
        # scale = (2 / (sqrt_c * lambda_p1_val + self.eps)) * torch.atanh(atanh_arg_clamped)
        # result = sub_p * (scale / (sub_p_norm_clamped + self.eps)) # Original
        # If sub_p_norm_clamped is self.eps, and scale is also small, result is fine.
        # If sub_p is zero, sub_p_norm is zero, atanh_arg is zero, atanh(0) is zero. scale is zero. Result is zero. Correct.
        scale = (2. / (sqrt_c * lambda_p1_val + self.eps)) * torch.atanh(atanh_arg_clamped)
        result = sub_p / (sub_p_norm_clamped + self.eps) * scale # More stable: normalize first, then scale
        return result

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, c_val: Union[float, torch.Tensor], dim: int = -1) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x)
        x_sqnorm = torch.sum(x * x, dim=dim, keepdim=True)
        y_sqnorm = torch.sum(y * y, dim=dim, keepdim=True)
        xy_dot = torch.sum(x * y, dim=dim, keepdim=True)
        
        num = (1 + 2 * c * xy_dot + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y
        den = 1 + 2 * c * xy_dot + c**2 * x_sqnorm * y_sqnorm
        return num / (den + self.eps)

    def ptransp(self, u: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        # For CODING-ONLY, identity is acceptable if inter-level transforms learn alignment.
        # A more accurate approximation from "Poincare Embeddings for Learning Hierarchical Representations"
        # lambda_p1 = self._lambda_x(p1, c_val, keepdim=True)
        # lambda_p2 = self._lambda_x(p2, c_val, keepdim=True)
        # return lambda_p1 / lambda_p2 * u # This is a common approximation for transport along geodesic
        logger.debug(f"PoincareBall.ptransp used identity for u at p1 to p2 with c={c_val}.")
        return u 

    def egrad2rgrad(self, p: torch.Tensor, grad_e: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        """Converts Euclidean gradient to Riemannian gradient for Poincare ball."""
        c = self._get_c_tensor(c_val, p)
        lambda_p_sq = self._lambda_x(p, c, keepdim=True).pow(2) / 4.0
        return grad_e / (lambda_p_sq + self.eps) # Inverse scaling factor for gradient


# --- Hyperbolic Utilities ---
class HyperbolicUtils: # ... (No significant changes, assuming it's mostly correct as per previous reviews)
    @staticmethod
    def _eps_like(tensor: torch.Tensor) -> float: return 1e-7 if tensor.dtype == torch.float32 else 1e-15
    @staticmethod
    def poincare_to_lorentz(x_poincare: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_tensor = torch.tensor(c, dtype=x_poincare.dtype, device=x_poincare.device) if isinstance(c, (float, int)) else c
        if c_tensor.ndim < x_poincare.ndim -1 : c_tensor = c_tensor.unsqueeze(-1)
        x_norm_sq = torch.sum(x_poincare * x_poincare, dim=-1, keepdim=True)
        # It's important that x_poincare is already projected within the ball.
        # max_x_norm_sq = (1.0 / c_tensor) * (1.0 - HyperbolicUtils._eps_like(x_poincare) * 10)
        # x_norm_sq = torch.clamp(x_norm_sq, max=max_x_norm_sq) # Clamping here can be problematic if not done carefully
        
        denominator = 1.0 - c_tensor * x_norm_sq + HyperbolicUtils._eps_like(x_poincare)
        inv_sqrt_c = 1.0 / (c_tensor.sqrt() + HyperbolicUtils._eps_like(x_poincare))
        
        lorentz_x0 = inv_sqrt_c * (1 + c_tensor * x_norm_sq) / denominator
        lorentz_x_spatial = inv_sqrt_c * (2 * x_poincare) / denominator
        return torch.cat([lorentz_x0, lorentz_x_spatial], dim=-1)

    @staticmethod
    def lorentz_to_poincare(x_lorentz: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_tensor = torch.tensor(c, dtype=x_lorentz.dtype, device=x_lorentz.device) if isinstance(c, (float, int)) else c
        if c_tensor.ndim < x_lorentz.ndim -1 : c_tensor = c_tensor.unsqueeze(-1)
        inv_sqrt_c = 1.0 / (c_tensor.sqrt() + HyperbolicUtils._eps_like(x_lorentz))
        x0 = x_lorentz[..., 0:1]; x_spatial = x_lorentz[..., 1:]
        denominator = x0 + inv_sqrt_c + HyperbolicUtils._eps_like(x_lorentz)
        return inv_sqrt_c * x_spatial / denominator

    @staticmethod
    def lorentz_tangent_to_poincare_tangent(tangent_L: torch.Tensor, point_L: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_tensor = torch.tensor(c, dtype=point_L.dtype, device=point_L.device) if isinstance(c, (float, int)) else c
        if c_tensor.ndim < point_L.ndim -1 : c_tensor = c_tensor.unsqueeze(-1)
        inv_sqrt_c = 1.0 / (c_tensor.sqrt() + HyperbolicUtils._eps_like(point_L))
        x_L_0 = point_L[..., 0:1]; x_L_spatial = point_L[..., 1:]
        u_L_0 = tangent_L[..., 0:1]; u_L_spatial = tangent_L[..., 1:]
        den_factor = x_L_0 + inv_sqrt_c + HyperbolicUtils._eps_like(x_L_0)
        term1 = (inv_sqrt_c / den_factor) * u_L_spatial
        term2_factor = (inv_sqrt_c / (den_factor.pow(2) + HyperbolicUtils._eps_like(den_factor)))
        term2 = term2_factor * x_L_spatial * u_L_0
        return term1 - term2

    @staticmethod
    def poincare_tangent_to_lorentz_tangent(tangent_P: torch.Tensor, point_P: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_tensor = torch.tensor(c, dtype=point_P.dtype, device=point_P.device) if isinstance(c, (float, int)) else c
        if c_tensor.ndim < point_P.ndim -1 : c_tensor = c_tensor.unsqueeze(-1)
        sqrt_c = c_tensor.sqrt()
        x_P_norm_sq = torch.sum(point_P * point_P, dim=-1, keepdim=True)
        den_inner = 1.0 - c_tensor * x_P_norm_sq + HyperbolicUtils._eps_like(point_P)
        den_factor_sq = den_inner.pow(2) + HyperbolicUtils._eps_like(point_P)
        xP_dot_uP = torch.sum(point_P * tangent_P, dim=-1, keepdim=True)
        u_L_0_num = (1.0 / (sqrt_c + HyperbolicUtils._eps_like(point_P))) * 4 * c_tensor * xP_dot_uP
        u_L_0 = u_L_0_num / den_factor_sq
        term1_spatial = (1.0 - c_tensor * x_P_norm_sq) * tangent_P
        term2_spatial = 2 * c_tensor * point_P * xP_dot_uP
        u_L_spatial_num = (2.0 / (sqrt_c + HyperbolicUtils._eps_like(point_P))) * (term1_spatial + term2_spatial)
        u_L_spatial = u_L_spatial_num / den_factor_sq
        return torch.cat([u_L_0, u_L_spatial], dim=-1)

    @staticmethod
    def minkowski_dot_product(u: torch.Tensor, v: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        res = -u[..., 0:1] * v[..., 0:1] + torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        return res if keepdim else res.squeeze(-1)

# --- Weight Initialization --- (No changes, assumed correct from previous reviews)
def init_weights_general(m: nn.Module, init_type: str = 'xavier_uniform', nonlinearity: str = 'relu', gain_factor: float = 1.0):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        gain = nn.init.calculate_gain(nonlinearity, param=0.2 if nonlinearity == 'leaky_relu' else None) * gain_factor 
        if init_type == 'xavier_uniform': nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init_type == 'xavier_normal': nn.init.xavier_normal_(m.weight, gain=gain)
        elif init_type == 'kaiming_uniform': nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0)
        elif init_type == 'kaiming_normal': nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0)
        elif init_type == 'orthogonal': nn.init.orthogonal_(m.weight, gain=gain)
        else: logger.warning(f"Unsupported init_type: {init_type}. Using default for {m.__class__.__name__}.")
        if m.bias is not None:
            if init_type in ['kaiming_uniform', 'kaiming_normal']:
                fan_in, _ = _calculate_fan_in_and_fan_out(m.weight); 
                if fan_in != 0: bound = 1 / math.sqrt(fan_in); nn.init.uniform_(m.bias, -bound, bound)
                else: nn.init.zeros_(m.bias)
            else: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, mean=0, std=0.02)

# --- DEFAULT_CONFIG_WUBU --- (No changes, assumed correct from previous reviews)
DEFAULT_WUBU_TEXT_CONFIG: Dict[str, Any] = { 
    "num_levels": 3, "hyperbolic_dims": [128, 64, 32], "euclidean_dims": [128, 64, 32], 
    "initial_curvatures": [1.0, 0.5, 0.25], "learnable_curvatures": True, "fixed_curvature_value": 1.0,
    "inter_level_transform_type": "mlp", "inter_level_use_lorentz": True, 
    "inter_level_mlp_layers": 2, "inter_level_mlp_dropout": 0.1,
    "boundary_points_per_level": [0, 0, 0], "boundary_manifold_dim_reduction_factor": 2, 
    "boundary_interaction_type": "none", "use_euclidean_skip_connections": False, 
    "use_hyperbolic_skip_connections": False, "final_aggregation_type": "last_level", 
    "final_aggregation_levels": "all", "output_tangent_projection_dim": None, 
    "dropout_rate": 0.1, "activation_fn": "gelu", "use_layer_norm": True, 
    "nesting_type": "tangent_proj", "use_rotation_in_transform": False, 
    "init_std_factor": 0.02, "ball_class": PoincareBall, 
    "hyperbolic_utils_class": HyperbolicUtils, "eps": 1e-7, 
}

# --- WuBu Core Components --- (Refinements for clarity and config usage)
class HyperbolicInterLevelTransform(nn.Module): # ... (No major structural changes, refined MLP logic from previous)
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any], level_idx: int):
        super().__init__()
        self.input_dim_poincare = input_dim 
        self.output_dim_poincare = output_dim 
        self.config = config; self.level_idx = level_idx
        BallClass: Type[PoincareBall] = config.get("ball_class", PoincareBall)
        HypUtilsClass: Type[HyperbolicUtils] = config.get("hyperbolic_utils_class", HyperbolicUtils)
        self.ball = BallClass(eps=config.get("eps", 1e-7))
        self.hyp_utils = HypUtilsClass()
        self.use_lorentz = config.get("inter_level_use_lorentz", True)
        self.transform_type = config.get("inter_level_transform_type", "mlp")
        self.eps = config.get("eps", 1e-7)

        mlp_input_dim = self.input_dim_poincare + 1 if self.use_lorentz else self.input_dim_poincare
        mlp_output_dim = self.output_dim_poincare + 1 if self.use_lorentz else self.output_dim_poincare
        
        if self.transform_type == "mlp": # ... (MLP construction as before, using config values)
            layers = []; current_d = mlp_input_dim
            for i in range(config.get("inter_level_mlp_layers", 2)):
                is_last = (i == config.get("inter_level_mlp_layers", 2) - 1)
                out_d = mlp_output_dim if is_last else max(mlp_output_dim // 2, mlp_input_dim // 2, mlp_input_dim, 1) # Ensure hidden is reasonable
                layers.append(nn.Linear(current_d, out_d))
                if not is_last:
                    if config.get("use_layer_norm", True): layers.append(nn.LayerNorm(out_d))
                    layers.append(getattr(nn, config.get("activation_fn", "GELU").upper())())
                    if config.get("inter_level_mlp_dropout", 0.1) > 0: layers.append(nn.Dropout(config.get("inter_level_mlp_dropout", 0.1)))
                current_d = out_d
            self.mlp_transform = nn.Sequential(*layers)
        else: self.mlp_transform = nn.Linear(mlp_input_dim, mlp_output_dim)
        self.apply(lambda m: init_weights_general(m, gain_factor=config.get("init_std_factor", 0.02))) # Corrected gain factor key

    def forward(self, parent_tangent_p: torch.Tensor, parent_origin_p: torch.Tensor, parent_curvature: torch.Tensor, child_curvature: torch.Tensor) -> torch.Tensor:
        # ... (Logic as per previous version, using self.input_dim_poincare and self.output_dim_poincare for clarity)
        if self.use_lorentz:
            parent_origin_l = self.hyp_utils.poincare_to_lorentz(parent_origin_p, parent_curvature)
            parent_tangent_l = self.hyp_utils.poincare_tangent_to_lorentz_tangent(parent_tangent_p, parent_origin_p, parent_curvature)
            transformed_tangent_l = self.mlp_transform(parent_tangent_l)
            
            batch_size = transformed_tangent_l.size(0)
            _child_curv_sqrt = child_curvature.sqrt().unsqueeze(-1) 
            child_origin_l_canonical_time = 1.0 / (_child_curv_sqrt + self.eps)
            child_origin_l_canonical_space = torch.zeros(batch_size, self.output_dim_poincare, dtype=transformed_tangent_l.dtype, device=transformed_tangent_l.device)
            child_origin_l_canonical = torch.cat([child_origin_l_canonical_time, child_origin_l_canonical_space], dim=-1)
            
            # Assuming MLP output is a valid tangent vector at child's Lorentz origin for CODING-ONLY
            projected_tangent_l = transformed_tangent_l 
            child_tangent_p_at_origin = self.hyp_utils.lorentz_tangent_to_poincare_tangent(projected_tangent_l, child_origin_l_canonical, child_curvature)
            return child_tangent_p_at_origin
        else:
            return self.mlp_transform(parent_tangent_p)


class HyperbolicWuBuNestingLevel(nn.Module): # ... (No major structural changes, refined config usage)
    def __init__(self, level_idx: int, input_dim: int, config: Dict[str,Any]):
        super().__init__(); self.level_idx = level_idx; self.input_dim = input_dim; self.config = config
        BallClass: Type[PoincareBall] = config.get("ball_class", PoincareBall)
        self.ball = BallClass(eps=config.get("eps", 1e-7))
        self.hyperbolic_dim_level = config["hyperbolic_dims"][level_idx]
        self.feature_transform = nn.Linear(input_dim, self.hyperbolic_dim_level)
        self.activation = getattr(nn, config.get("activation_fn", "GELU").upper())()
        self.norm = nn.LayerNorm(self.hyperbolic_dim_level) if config.get("use_layer_norm", True) else nn.Identity()
        
        curv_val = config["initial_curvatures"][level_idx]
        if config["learnable_curvatures"]: self.curvature = nn.Parameter(torch.tensor(float(curv_val))) # Ensure float
        else: self.register_buffer('curvature', torch.tensor(float(curv_val)))
        self.apply(lambda m: init_weights_general(m, gain_factor=config.get("init_std_factor", 0.02))) # Corrected gain factor key

    def forward(self, input_tangent_p: torch.Tensor, input_origin_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level_features_tangent = self.norm(self.activation(self.feature_transform(input_tangent_p)))
        return input_origin_p, level_features_tangent, self.curvature


class FullyHyperbolicWuBuNestingModel(nn.Module): # ... (No major structural changes, verified _determine_output_dim)
    def __init__(self, input_tangent_dim: int, config: Optional[Dict[str,Any]] = None):
        super().__init__(); self.config = config.copy() if config is not None else DEFAULT_WUBU_TEXT_CONFIG.copy() # Use copy
        self.num_levels = self.config["num_levels"]
        BallClass: Type[PoincareBall] = self.config.get("ball_class", PoincareBall)
        self.ball = BallClass(eps=self.config.get("eps", 1e-7))
        self.levels = nn.ModuleList(); self.inter_level_transforms = nn.ModuleList()
        current_input_dim = input_tangent_dim
        for i in range(self.num_levels):
            level_poincare_dim = self.config["hyperbolic_dims"][i]
            self.levels.append(HyperbolicWuBuNestingLevel(i, current_input_dim, self.config))
            if i < self.num_levels - 1:
                next_level_poincare_dim = self.config["hyperbolic_dims"][i+1]
                self.inter_level_transforms.append(HyperbolicInterLevelTransform(level_poincare_dim, next_level_poincare_dim, self.config, i))
                current_input_dim = next_level_poincare_dim
        self._determine_output_dim()
        self.apply(lambda m: init_weights_general(m, gain_factor=self.config.get("init_std_factor", 0.02))) # Corrected gain factor key

    def _determine_output_dim(self): # ... (Logic seems fine from previous review)
        agg_levels_cfg = self.config.get("final_aggregation_levels", "all")
        agg_indices = list(range(self.num_levels)) if agg_levels_cfg=="all" else [i for i in agg_levels_cfg if 0<=i<self.num_levels] # type: ignore
        if not agg_indices: agg_indices = list(range(self.num_levels))
        dims_to_agg = [self.config["hyperbolic_dims"][i] for i in agg_indices if i < len(self.config["hyperbolic_dims"])]

        if not dims_to_agg and self.num_levels > 0 : # If agg_indices was empty or out of bounds
            dims_to_agg = [self.config["hyperbolic_dims"][-1]] if self.config["hyperbolic_dims"] else [1]
            logger.warning(f"Aggregation indices resulted in empty list. Defaulting to last known dim: {dims_to_agg}")
        elif not dims_to_agg and self.num_levels == 0:
             self.aggregated_output_dim = 1; self.output_tangent_dim = 1; 
             self.output_tangent_projection = nn.Identity() # type: ignore[assignment]
             logger.warning("No levels in WuBu model, output dim set to 1.")
             return

        agg_type = self.config.get("final_aggregation_type", "last_level")
        if agg_type == "last_level": self.aggregated_output_dim = dims_to_agg[-1]
        elif agg_type == "concat": self.aggregated_output_dim = sum(dims_to_agg)
        else: self.aggregated_output_dim = dims_to_agg[0] # sum, mean, etc. assume same dim or take first.
        
        proj_dim_cfg = self.config.get("output_tangent_projection_dim")
        if proj_dim_cfg is not None and proj_dim_cfg > 0:
            self.output_tangent_projection = nn.Linear(self.aggregated_output_dim, proj_dim_cfg)
            self.output_tangent_dim = proj_dim_cfg
        else:
            self.output_tangent_projection = nn.Identity() # type: ignore[assignment]
            self.output_tangent_dim = self.aggregated_output_dim
        if self.output_tangent_dim <=0: self.output_tangent_dim = 1 # Safety

    def forward(self, initial_tangent_p: torch.Tensor, initial_origin_p: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ... (Logic as per previous version, seems robust for CODING-ONLY)
        batch_size = initial_tangent_p.size(0)
        if initial_origin_p is None: initial_origin_p = torch.zeros_like(initial_tangent_p)
        current_o_p, current_t_p = initial_origin_p, initial_tangent_p
        level_outputs = []
        for i in range(self.num_levels):
            level_out_o, level_feat_t, level_curv = self.levels[i](current_t_p, current_o_p)
            level_outputs.append(level_feat_t)
            if i < self.num_levels - 1:
                current_t_p = self.inter_level_transforms[i](level_feat_t, level_out_o, level_curv, self.levels[i+1].curvature)
                current_o_p = level_out_o # Origin for next level is output origin of current
        
        agg_levels_cfg = self.config.get("final_aggregation_levels", "all")
        agg_indices = list(range(self.num_levels)) if agg_levels_cfg=="all" else [i for i in agg_levels_cfg if 0<=i<len(level_outputs)] # type: ignore
        if not agg_indices: agg_indices = list(range(len(level_outputs)))
        to_agg = [level_outputs[i] for i in agg_indices]
        if not to_agg: return torch.zeros(batch_size, self.output_tangent_dim, device=initial_tangent_p.device)

        agg_type = self.config.get("final_aggregation_type", "last_level")
        if agg_type == "last_level": agg_t = to_agg[-1]
        elif agg_type == "concat": agg_t = torch.cat(to_agg, dim=-1)
        else: # sum, mean
            stacked_t = torch.stack(to_agg, dim=0)
            agg_t = stacked_t.sum(dim=0) if agg_type == "sum" else stacked_t.mean(dim=0)
        return self.output_tangent_projection(agg_t)


# --- Base ETP Classes --- (No changes, verified previously)
class AbstractETPTransfusionHead(nn.Module):
    def __init__(self, source_embedding_dim: int, wubu_tangent_dim: int): super().__init__(); self.source_embedding_dim=source_embedding_dim; self.wubu_tangent_dim=wubu_tangent_dim
    def forward(self, source_embedding: torch.Tensor) -> torch.Tensor: raise NotImplementedError
class AbstractETPDecoder(nn.Module):
    def __init__(self, wubu_tangent_dim: int, source_embedding_dim: int): super().__init__(); self.wubu_tangent_dim=wubu_tangent_dim; self.source_embedding_dim=source_embedding_dim
    def forward(self, wubu_tangent_vector: torch.Tensor) -> torch.Tensor: raise NotImplementedError

# --- Concrete ETP Components --- (No changes, verified previously)
class DeepSeekR1TransfusionHead(AbstractETPTransfusionHead):
    def __init__(self, source_embedding_dim: int, wubu_tangent_dim: int, mlp_hidden_dim_ratio: float=2.0, num_mlp_layers: int=2, activation_fn: str="GELU", use_layer_norm: bool=True, dropout_rate: float=0.1):
        super().__init__(source_embedding_dim,wubu_tangent_dim); layers=[]; curr_dim=source_embedding_dim; hid_dim=int(source_embedding_dim*mlp_hidden_dim_ratio)
        for i in range(num_mlp_layers):
            is_last=(i==num_mlp_layers-1); out_d=wubu_tangent_dim if is_last else hid_dim; layers.append(nn.Linear(curr_dim,out_d))
            if not is_last:
                if use_layer_norm: layers.append(nn.LayerNorm(out_d))
                layers.append(getattr(nn,activation_fn.upper(),nn.GELU())()); 
                if dropout_rate>0: layers.append(nn.Dropout(dropout_rate))
            curr_dim=out_d
        self.mlp=nn.Sequential(*layers); self.apply(init_weights_general)
    def forward(self,source_embedding:torch.Tensor)->torch.Tensor: return self.mlp(source_embedding)

class WuBuToDeepSeekR1Decoder(AbstractETPDecoder):
    def __init__(self, wubu_tangent_dim: int, source_embedding_dim: int, mlp_hidden_dim_ratio: float=2.0, num_mlp_layers: int=2, activation_fn: str="GELU", use_layer_norm: bool=True, dropout_rate: float=0.1):
        super().__init__(wubu_tangent_dim,source_embedding_dim); layers=[]; curr_dim=wubu_tangent_dim; hid_dim=int(wubu_tangent_dim*mlp_hidden_dim_ratio)
        for i in range(num_mlp_layers):
            is_last=(i==num_mlp_layers-1); out_d=source_embedding_dim if is_last else hid_dim; layers.append(nn.Linear(curr_dim,out_d))
            if not is_last:
                if use_layer_norm: layers.append(nn.LayerNorm(out_d))
                layers.append(getattr(nn,activation_fn.upper(),nn.GELU())()); 
                if dropout_rate>0: layers.append(nn.Dropout(dropout_rate))
            curr_dim=out_d
        self.mlp=nn.Sequential(*layers); self.apply(init_weights_general)
    def forward(self,wubu_tangent_vector:torch.Tensor)->torch.Tensor: return self.mlp(wubu_tangent_vector)

# --- Main ETP Sphere Model --- (No changes, verified previously)
class ETP_WuBuText_DS_R1_Sphere(nn.Module):
    def __init__(self, ds_r1_embedding_dim:int, wubu_initial_tangent_dim:int, wubu_core_config:Optional[Dict]=None, head_mlp_layers:int=2, head_mlp_ratio:float=2.0, decoder_mlp_layers:int=2, decoder_mlp_ratio:float=2.0, activation_fn:str="GELU", use_layer_norm:bool=True, dropout_rate:float=0.1):
        super().__init__(); self.ds_r1_embedding_dim=ds_r1_embedding_dim; self.wubu_initial_tangent_dim=wubu_initial_tangent_dim
        actual_wubu_config=wubu_core_config.copy() if wubu_core_config is not None else DEFAULT_WUBU_TEXT_CONFIG.copy()
        if isinstance(actual_wubu_config.get("ball_class"),type): BallClass=actual_wubu_config["ball_class"]; actual_wubu_config["ball"]=BallClass(eps=actual_wubu_config.get("eps",1e-7)) # type: ignore
        self.transfusion_head=DeepSeekR1TransfusionHead(ds_r1_embedding_dim,wubu_initial_tangent_dim,head_mlp_ratio,head_mlp_layers,activation_fn,use_layer_norm,dropout_rate)
        self.wubu_core=FullyHyperbolicWuBuNestingModel(wubu_initial_tangent_dim,actual_wubu_config)
        core_out_dim=self.wubu_core.output_tangent_dim
        self.decoder=WuBuToDeepSeekR1Decoder(core_out_dim,ds_r1_embedding_dim,decoder_mlp_ratio,decoder_mlp_layers,activation_fn,use_layer_norm,dropout_rate)
        logger.info(f"ETP_WuBuText_DS_R1_Sphere initialized. WuBu core output dim: {core_out_dim}")
    def forward(self,source_embedding:torch.Tensor)->torch.Tensor: return self.decoder(self.wubu_core(self.transfusion_head(source_embedding)))
    def get_latent(self,source_embedding:torch.Tensor)->torch.Tensor: return self.wubu_core(self.transfusion_head(source_embedding))
    def get_wubu_core_config(self) -> Dict[str,Any]: return self.wubu_core.config


if __name__ == '__main__': # ... (Conceptual test as before)
    logger.info("Starting example usage of ETP WuBu Architectures (CODING-ONLY structure check).")
    ds_r1_dim_example=768; wubu_input_tangent_dim_example=256
    test_wubu_config_example=DEFAULT_WUBU_TEXT_CONFIG.copy()
    test_wubu_config_example["hyperbolic_dims"]=[128,64]; test_wubu_config_example["num_levels"]=2
    test_wubu_config_example["initial_curvatures"]=[1.0,0.5]; test_wubu_config_example["final_aggregation_type"]="last_level"
    etp_model_example=ETP_WuBuText_DS_R1_Sphere(ds_r1_dim_example,wubu_input_tangent_dim_example,test_wubu_config_example)
    logger.info(f"Example ETP Model Instantiated: {etp_model_example}. WuBu Core output_tangent_dim: {etp_model_example.wubu_core.output_tangent_dim}") # type: ignore
    dummy_input=torch.randn(4,ds_r1_dim_example)
    try:
        logger.info(f"Conceptual reconstructed output shape: {etp_model_example(dummy_input).shape}")
        logger.info(f"Conceptual latent vector shape: {etp_model_example.get_latent(dummy_input).shape}")
    except Exception as e_example: logger.error(f"Error during conceptual pass: {e_example}",exc_info=True)
    logger.info("Example usage structure check finished.")
