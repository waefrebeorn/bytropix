# Combined ETP Script (Focus: Phase 1 - Reconstruction)
# This script integrates components from etp_common, trainer_phase1.py, and run_phase1.py.

# =====================================================================
# Python Imports and Global Setup
# =====================================================================
import argparse
import json
import logging
import math
import os
import random
import sys # For sys.exit in example script main
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union, Type, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.init import _calculate_fan_in_and_fan_out
# Spectral norm is not directly used by Phase 1 models, but init_weights_general might reference it.
# If not, this could be removed for Phase 1. For now, keeping it for init_weights_general.
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer # For etp_embedding_extractor

# --- Setup Root Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
script_logger = logging.getLogger("ETP_Combined_Phase1_Script")

# --- Constants ---
EPS = 1e-7
PHI = (1 + math.sqrt(5)) / 2
TAN_VEC_CLAMP_VAL = 1e4
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e8
MIN_WUBU_LEVEL_SCALE = EPS
MAX_WUBU_LEVEL_SCALE = 10.0

# --- Conditional Imports & Availability Flags ---
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    script_logger.warning("h5py library not found globally. HDF5 file support will be disabled if any component relies on it directly.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None # type: ignore
    WANDB_AVAILABLE = False
    script_logger.info("wandb not found globally. WandB logging will be disabled if any component tries to use it.")

# =====================================================================
# Content from: /draftPY/etp_common/etp_wubu_architectures.py
# (Identical to Phase 2, as ETP_Sphere model is still the core)
# =====================================================================
logger_wubu_arch = logging.getLogger("etp_wubu_architectures_p1")

class Manifold:
    def __init__(self, eps: float = 1e-7): self.eps = eps
    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def proj(self, x: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def expmap(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def ptransp(self, u: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def an_dist(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: return torch.sqrt(self.sqdist(p1, p2, c) + self.eps)
    def egrad2rgrad(self, p: torch.Tensor, grad_e: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        logger_wubu_arch.warning(f"Manifold.egrad2rgrad not implemented for {self.__class__.__name__}. Returning Euclidean gradient.")
        return grad_e

class PoincareBall(Manifold):
    def __init__(self, eps: float = 1e-7, min_norm_scale: float = 1e-5):
        super().__init__(eps=eps); self.name = 'PoincareBall'; self.min_norm_scale = min_norm_scale
    def _get_c_tensor(self, c: Union[float, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if isinstance(c, (float, int)): return torch.tensor(c, dtype=x.dtype, device=x.device)
        if c.ndim == x.ndim: return c
        elif c.ndim == x.ndim -1 : return c.unsqueeze(-1)
        elif c.ndim == 1 and x.ndim > 1 and c.size(0) == x.size(0): return c.view(x.size(0), *([1]*(x.ndim-1)))
        return c
    def _lambda_x(self, x: torch.Tensor, c_val: Union[float, torch.Tensor], keepdim: bool = False) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x); x_sqnorm = torch.sum(x * x, dim=-1, keepdim=keepdim)
        return 2 / (1 - c * x_sqnorm + self.eps)
    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p1); sqrt_c = c.sqrt()
        mobius_add_result = self.mobius_add(-p1, p2, c, dim=-1)
        num = 2 * c.squeeze() * torch.sum(mobius_add_result.pow(2), dim=-1)
        p1_sqnorm = torch.sum(p1 * p1, dim=-1); p2_sqnorm = torch.sum(p2 * p2, dim=-1)
        den1 = 1 - c.squeeze() * p1_sqnorm; den2 = 1 - c.squeeze() * p2_sqnorm
        acosh_arg = 1 + num / (den1 * den2 + self.eps)
        dist_c = torch.acosh(torch.clamp_min(acosh_arg, 1.0 + self.eps))
        dist = dist_c / (sqrt_c.squeeze() + self.eps); return dist.pow(2)
    def proj(self, x: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x); sqrt_c = c.sqrt()
        max_radius = (1.0 - self.min_norm_scale) / (sqrt_c + self.eps)
        norm_x = x.norm(dim=-1, keepdim=True, p=2); cond = norm_x > max_radius
        projected_x = x / (norm_x + self.eps) * max_radius; return torch.where(cond, projected_x, x)
    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: return u
    def expmap(self, p: torch.Tensor, u: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p); sqrt_c = c.sqrt()
        u_norm = torch.norm(u, p=2, dim=-1, keepdim=True); u_norm_clamped = torch.clamp_min(u_norm, self.eps)
        lambda_p_val = self._lambda_x(p, c, keepdim=True); tanh_arg = sqrt_c * lambda_p_val * u_norm / 2
        scale = torch.tanh(tanh_arg) / (sqrt_c * u_norm_clamped + self.eps)
        second_term = u * scale; return self.mobius_add(p, second_term, c, dim=-1)
    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p1); sqrt_c = c.sqrt()
        sub_p = self.mobius_add(-p1, p2, c, dim=-1)
        sub_p_norm = torch.norm(sub_p, p=2, dim=-1, keepdim=True); sub_p_norm_clamped = torch.clamp_min(sub_p_norm, self.eps)
        lambda_p1_val = self._lambda_x(p1, c, keepdim=True); atanh_arg = sqrt_c * sub_p_norm
        atanh_arg_clamped = torch.clamp(atanh_arg, max=1.0 - self.eps)
        scale = (2. / (sqrt_c * lambda_p1_val + self.eps)) * torch.atanh(atanh_arg_clamped)
        result = sub_p / (sub_p_norm_clamped + self.eps) * scale; return result
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, c_val: Union[float, torch.Tensor], dim: int = -1) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x); x_sqnorm = torch.sum(x * x, dim=dim, keepdim=True)
        y_sqnorm = torch.sum(y * y, dim=dim, keepdim=True); xy_dot = torch.sum(x * y, dim=dim, keepdim=True)
        num = (1 + 2 * c * xy_dot + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y
        den = 1 + 2 * c * xy_dot + c**2 * x_sqnorm * y_sqnorm; return num / (den + self.eps)
    def ptransp(self, u: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        logger_wubu_arch.debug(f"PoincareBall.ptransp used identity for u at p1 to p2 with c={c_val}.")
        return u
    def egrad2rgrad(self, p: torch.Tensor, grad_e: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p); lambda_p_sq = self._lambda_x(p, c, keepdim=True).pow(2) / 4.0
        return grad_e / (lambda_p_sq + self.eps)

class HyperbolicUtils:
    @staticmethod
    def _eps_like(tensor: torch.Tensor) -> float: return 1e-7 if tensor.dtype == torch.float32 else 1e-15
    @staticmethod
    def poincare_to_lorentz(x_poincare: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_tensor = torch.tensor(c, dtype=x_poincare.dtype, device=x_poincare.device) if isinstance(c, (float, int)) else c
        if c_tensor.ndim < x_poincare.ndim -1 : c_tensor = c_tensor.unsqueeze(-1)
        x_norm_sq = torch.sum(x_poincare * x_poincare, dim=-1, keepdim=True)
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
        term2 = term2_factor * x_L_spatial * u_L_0; return term1 - term2
    @staticmethod
    def poincare_tangent_to_lorentz_tangent(tangent_P: torch.Tensor, point_P: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_tensor = torch.tensor(c, dtype=point_P.dtype, device=point_P.device) if isinstance(c, (float, int)) else c
        if c_tensor.ndim < point_P.ndim -1 : c_tensor = c_tensor.unsqueeze(-1)
        sqrt_c = c_tensor.sqrt(); x_P_norm_sq = torch.sum(point_P * point_P, dim=-1, keepdim=True)
        den_inner = 1.0 - c_tensor * x_P_norm_sq + HyperbolicUtils._eps_like(point_P)
        den_factor_sq = den_inner.pow(2) + HyperbolicUtils._eps_like(point_P)
        xP_dot_uP = torch.sum(point_P * tangent_P, dim=-1, keepdim=True)
        u_L_0_num = (1.0 / (sqrt_c + HyperbolicUtils._eps_like(point_P))) * 4 * c_tensor * xP_dot_uP
        u_L_0 = u_L_0_num / den_factor_sq
        term1_spatial = (1.0 - c_tensor * x_P_norm_sq) * tangent_P; term2_spatial = 2 * c_tensor * point_P * xP_dot_uP
        u_L_spatial_num = (2.0 / (sqrt_c + HyperbolicUtils._eps_like(point_P))) * (term1_spatial + term2_spatial)
        u_L_spatial = u_L_spatial_num / den_factor_sq; return torch.cat([u_L_0, u_L_spatial], dim=-1)
    @staticmethod
    def minkowski_dot_product(u: torch.Tensor, v: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        res = -u[..., 0:1] * v[..., 0:1] + torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        return res if keepdim else res.squeeze(-1)

def init_weights_general(m: nn.Module, init_type: str = 'xavier_uniform', nonlinearity: str = 'relu', gain_factor: float = 1.0):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        gain = nn.init.calculate_gain(nonlinearity, param=0.2 if nonlinearity == 'leaky_relu' else None) * gain_factor
        if init_type == 'xavier_uniform': nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init_type == 'xavier_normal': nn.init.xavier_normal_(m.weight, gain=gain)
        elif init_type == 'kaiming_uniform': nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0)
        elif init_type == 'kaiming_normal': nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0)
        elif init_type == 'orthogonal': nn.init.orthogonal_(m.weight, gain=gain)
        else: logger_wubu_arch.warning(f"Unsupported init_type: {init_type}. Using default for {m.__class__.__name__}.")
        if m.bias is not None:
            if init_type in ['kaiming_uniform', 'kaiming_normal']:
                fan_in, _ = _calculate_fan_in_and_fan_out(m.weight);
                if fan_in != 0: bound = 1 / math.sqrt(fan_in); nn.init.uniform_(m.bias, -bound, bound)
                else: nn.init.zeros_(m.bias)
            else: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, mean=0, std=0.02)

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

class HyperbolicInterLevelTransform(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any], level_idx: int):
        super().__init__(); self.input_dim_poincare = input_dim; self.output_dim_poincare = output_dim
        self.config = config; self.level_idx = level_idx
        BallClass: Type[PoincareBall] = config.get("ball_class", PoincareBall)
        HypUtilsClass: Type[HyperbolicUtils] = config.get("hyperbolic_utils_class", HyperbolicUtils)
        self.ball = BallClass(eps=config.get("eps", 1e-7)); self.hyp_utils = HypUtilsClass()
        self.use_lorentz = config.get("inter_level_use_lorentz", True); self.transform_type = config.get("inter_level_transform_type", "mlp")
        self.eps = config.get("eps", 1e-7)
        mlp_input_dim = self.input_dim_poincare + 1 if self.use_lorentz else self.input_dim_poincare
        mlp_output_dim = self.output_dim_poincare + 1 if self.use_lorentz else self.output_dim_poincare
        if self.transform_type == "mlp":
            layers = []; current_d = mlp_input_dim
            for i in range(config.get("inter_level_mlp_layers", 2)):
                is_last = (i == config.get("inter_level_mlp_layers", 2) - 1)
                out_d = mlp_output_dim if is_last else max(mlp_output_dim // 2, mlp_input_dim // 2, mlp_input_dim, 1)
                layers.append(nn.Linear(current_d, out_d))
                if not is_last:
                    if config.get("use_layer_norm", True): layers.append(nn.LayerNorm(out_d))
                    layers.append(getattr(nn, config.get("activation_fn", "GELU").upper())())
                    if config.get("inter_level_mlp_dropout", 0.1) > 0: layers.append(nn.Dropout(config.get("inter_level_mlp_dropout", 0.1)))
                current_d = out_d
            self.mlp_transform = nn.Sequential(*layers)
        else: self.mlp_transform = nn.Linear(mlp_input_dim, mlp_output_dim)
        self.apply(lambda m: init_weights_general(m, gain_factor=config.get("init_std_factor", 0.02)))
    def forward(self, parent_tangent_p: torch.Tensor, parent_origin_p: torch.Tensor, parent_curvature: torch.Tensor, child_curvature: torch.Tensor) -> torch.Tensor:
        if self.use_lorentz:
            parent_origin_l = self.hyp_utils.poincare_to_lorentz(parent_origin_p, parent_curvature)
            parent_tangent_l = self.hyp_utils.poincare_tangent_to_lorentz_tangent(parent_tangent_p, parent_origin_p, parent_curvature)
            transformed_tangent_l = self.mlp_transform(parent_tangent_l); batch_size = transformed_tangent_l.size(0)
            _child_curv_sqrt = child_curvature.sqrt().unsqueeze(-1)
            child_origin_l_canonical_time = 1.0 / (_child_curv_sqrt + self.eps)
            child_origin_l_canonical_space = torch.zeros(batch_size, self.output_dim_poincare, dtype=transformed_tangent_l.dtype, device=transformed_tangent_l.device)
            child_origin_l_canonical = torch.cat([child_origin_l_canonical_time, child_origin_l_canonical_space], dim=-1)
            projected_tangent_l = transformed_tangent_l # Assuming MLP output is valid tangent vector
            child_tangent_p_at_origin = self.hyp_utils.lorentz_tangent_to_poincare_tangent(projected_tangent_l, child_origin_l_canonical, child_curvature)
            return child_tangent_p_at_origin
        else: return self.mlp_transform(parent_tangent_p)

class HyperbolicWuBuNestingLevel(nn.Module):
    def __init__(self, level_idx: int, input_dim: int, config: Dict[str,Any]):
        super().__init__(); self.level_idx = level_idx; self.input_dim = input_dim; self.config = config
        BallClass: Type[PoincareBall] = config.get("ball_class", PoincareBall)
        self.ball = BallClass(eps=config.get("eps", 1e-7))
        self.hyperbolic_dim_level = config["hyperbolic_dims"][level_idx]
        self.feature_transform = nn.Linear(input_dim, self.hyperbolic_dim_level)
        self.activation = getattr(nn, config.get("activation_fn", "GELU").upper())()
        self.norm = nn.LayerNorm(self.hyperbolic_dim_level) if config.get("use_layer_norm", True) else nn.Identity()
        curv_val = config["initial_curvatures"][level_idx]
        if config["learnable_curvatures"]: self.curvature = nn.Parameter(torch.tensor(float(curv_val)))
        else: self.register_buffer('curvature', torch.tensor(float(curv_val)))
        self.apply(lambda m: init_weights_general(m, gain_factor=config.get("init_std_factor", 0.02)))
    def forward(self, input_tangent_p: torch.Tensor, input_origin_p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level_features_tangent = self.norm(self.activation(self.feature_transform(input_tangent_p)))
        return input_origin_p, level_features_tangent, self.curvature

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_tangent_dim: int, config: Optional[Dict[str,Any]] = None):
        super().__init__(); self.config = config.copy() if config is not None else DEFAULT_WUBU_TEXT_CONFIG.copy()
        self.num_levels = self.config["num_levels"]
        BallClass: Type[PoincareBall] = self.config.get("ball_class", PoincareBall)
        self.ball = BallClass(eps=self.config.get("eps", 1e-7))
        self.levels = nn.ModuleList(); self.inter_level_transforms = nn.ModuleList(); current_input_dim = input_tangent_dim
        for i in range(self.num_levels):
            level_poincare_dim = self.config["hyperbolic_dims"][i]
            self.levels.append(HyperbolicWuBuNestingLevel(i, current_input_dim, self.config))
            if i < self.num_levels - 1:
                next_level_poincare_dim = self.config["hyperbolic_dims"][i+1]
                self.inter_level_transforms.append(HyperbolicInterLevelTransform(level_poincare_dim, next_level_poincare_dim, self.config, i))
                current_input_dim = next_level_poincare_dim
        self._determine_output_dim()
        self.apply(lambda m: init_weights_general(m, gain_factor=self.config.get("init_std_factor", 0.02)))
    def _determine_output_dim(self):
        agg_levels_cfg = self.config.get("final_aggregation_levels", "all")
        agg_indices = list(range(self.num_levels)) if agg_levels_cfg=="all" else [i for i in agg_levels_cfg if 0<=i<self.num_levels] # type: ignore
        if not agg_indices: agg_indices = list(range(self.num_levels))
        dims_to_agg = [self.config["hyperbolic_dims"][i] for i in agg_indices if i < len(self.config["hyperbolic_dims"])]
        if not dims_to_agg and self.num_levels > 0 :
            dims_to_agg = [self.config["hyperbolic_dims"][-1]] if self.config["hyperbolic_dims"] else [1]
            logger_wubu_arch.warning(f"Aggregation indices resulted in empty list. Defaulting to last known dim: {dims_to_agg}")
        elif not dims_to_agg and self.num_levels == 0:
             self.aggregated_output_dim = 1; self.output_tangent_dim = 1; self.output_tangent_projection = nn.Identity(); logger_wubu_arch.warning("No levels in WuBu model, output dim set to 1."); return # type: ignore
        agg_type = self.config.get("final_aggregation_type", "last_level")
        if agg_type == "last_level": self.aggregated_output_dim = dims_to_agg[-1]
        elif agg_type == "concat": self.aggregated_output_dim = sum(dims_to_agg)
        else: self.aggregated_output_dim = dims_to_agg[0]
        proj_dim_cfg = self.config.get("output_tangent_projection_dim")
        if proj_dim_cfg is not None and proj_dim_cfg > 0:
            self.output_tangent_projection = nn.Linear(self.aggregated_output_dim, proj_dim_cfg); self.output_tangent_dim = proj_dim_cfg
        else: self.output_tangent_projection = nn.Identity(); self.output_tangent_dim = self.aggregated_output_dim # type: ignore
        if self.output_tangent_dim <=0: self.output_tangent_dim = 1 # Safety
    def forward(self, initial_tangent_p: torch.Tensor, initial_origin_p: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = initial_tangent_p.size(0)
        if initial_origin_p is None: initial_origin_p = torch.zeros_like(initial_tangent_p)
        current_o_p, current_t_p = initial_origin_p, initial_tangent_p; level_outputs = []
        for i in range(self.num_levels):
            level_out_o, level_feat_t, level_curv = self.levels[i](current_t_p, current_o_p)
            level_outputs.append(level_feat_t)
            if i < self.num_levels - 1:
                current_t_p = self.inter_level_transforms[i](level_feat_t, level_out_o, level_curv, self.levels[i+1].curvature)
                current_o_p = level_out_o
        agg_levels_cfg = self.config.get("final_aggregation_levels", "all")
        agg_indices = list(range(self.num_levels)) if agg_levels_cfg=="all" else [i for i in agg_levels_cfg if 0<=i<len(level_outputs)] # type: ignore
        if not agg_indices: agg_indices = list(range(len(level_outputs)))
        to_agg = [level_outputs[i] for i in agg_indices]
        if not to_agg: return torch.zeros(batch_size, self.output_tangent_dim, device=initial_tangent_p.device)
        agg_type = self.config.get("final_aggregation_type", "last_level")
        if agg_type == "last_level": agg_t = to_agg[-1]
        elif agg_type == "concat": agg_t = torch.cat(to_agg, dim=-1)
        else: stacked_t = torch.stack(to_agg, dim=0); agg_t = stacked_t.sum(dim=0) if agg_type == "sum" else stacked_t.mean(dim=0)
        return self.output_tangent_projection(agg_t)

class AbstractETPTransfusionHead(nn.Module):
    def __init__(self, source_embedding_dim: int, wubu_tangent_dim: int): super().__init__(); self.source_embedding_dim=source_embedding_dim; self.wubu_tangent_dim=wubu_tangent_dim
    def forward(self, source_embedding: torch.Tensor) -> torch.Tensor: raise NotImplementedError
class AbstractETPDecoder(nn.Module):
    def __init__(self, wubu_tangent_dim: int, source_embedding_dim: int): super().__init__(); self.wubu_tangent_dim=wubu_tangent_dim; self.source_embedding_dim=source_embedding_dim
    def forward(self, wubu_tangent_vector: torch.Tensor) -> torch.Tensor: raise NotImplementedError

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

class ETP_WuBuText_DS_R1_Sphere(nn.Module):
    def __init__(self, ds_r1_embedding_dim:int, wubu_initial_tangent_dim:int, wubu_core_config:Optional[Dict]=None, head_mlp_layers:int=2, head_mlp_ratio:float=2.0, decoder_mlp_layers:int=2, decoder_mlp_ratio:float=2.0, activation_fn:str="GELU", use_layer_norm:bool=True, dropout_rate:float=0.1):
        super().__init__(); self.ds_r1_embedding_dim=ds_r1_embedding_dim; self.wubu_initial_tangent_dim=wubu_initial_tangent_dim
        actual_wubu_config=wubu_core_config.copy() if wubu_core_config is not None else DEFAULT_WUBU_TEXT_CONFIG.copy()
        if isinstance(actual_wubu_config.get("ball_class"),type): BallClass=actual_wubu_config["ball_class"]; actual_wubu_config["ball"]=BallClass(eps=actual_wubu_config.get("eps",1e-7)) # type: ignore
        self.transfusion_head=DeepSeekR1TransfusionHead(ds_r1_embedding_dim,wubu_initial_tangent_dim,head_mlp_ratio,head_mlp_layers,activation_fn,use_layer_norm,dropout_rate)
        self.wubu_core=FullyHyperbolicWuBuNestingModel(wubu_initial_tangent_dim,actual_wubu_config)
        core_out_dim=self.wubu_core.output_tangent_dim
        self.decoder=WuBuToDeepSeekR1Decoder(core_out_dim,ds_r1_embedding_dim,decoder_mlp_ratio,decoder_mlp_layers,activation_fn,use_layer_norm,dropout_rate)
        logger_wubu_arch.info(f"ETP_WuBuText_DS_R1_Sphere initialized. WuBu core output dim: {core_out_dim}")
    def forward(self,source_embedding:torch.Tensor)->torch.Tensor: return self.decoder(self.wubu_core(self.transfusion_head(source_embedding)))
    def get_latent(self,source_embedding:torch.Tensor)->torch.Tensor: return self.wubu_core(self.transfusion_head(source_embedding))
    def get_wubu_core_config(self) -> Dict[str,Any]: return self.wubu_core.config
# End of etp_wubu_architectures.py content
# =====================================================================

# =====================================================================
# Content from: /draftPY/etp_common/etp_losses.py
# (Slightly trimmed for Phase 1: Adversarial losses not directly used by trainer)
# =====================================================================
logger_losses_p1 = logging.getLogger("etp_losses_p1")

def calculate_reconstruction_loss_p1( # Renamed for clarity in combined script
    reconstructed_embeddings: torch.Tensor,
    original_embeddings: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    if reconstructed_embeddings.shape != original_embeddings.shape:
        logger_losses_p1.warning(f"Shape mismatch in reconstruction loss: Reconstructed {reconstructed_embeddings.shape}, Original {original_embeddings.shape}.")
    if loss_type == "mse": return F.mse_loss(reconstructed_embeddings, original_embeddings)
    elif loss_type == "cosine":
        sim_dim = 1 if reconstructed_embeddings.ndim > 1 else 0 # dim=0 for single vector comparison
        cosine_sim = F.cosine_similarity(reconstructed_embeddings, original_embeddings, dim=sim_dim)
        return (1 - cosine_sim).mean()
    else:
        logger_losses_p1.error(f"Unsupported reconstruction loss type: {loss_type}. Defaulting to MSE.")
        return F.mse_loss(reconstructed_embeddings, original_embeddings)

def _pairwise_similarity_p1(batch: torch.Tensor, metric: str = "cosine") -> torch.Tensor: # Renamed
    if batch.ndim != 2: raise ValueError(f"Input batch must be 2D (N, D), got {batch.shape}")
    N, D = batch.shape
    if N == 0: return torch.empty((0, 0), device=batch.device, dtype=batch.dtype)
    if N == 1: return torch.ones((1,1), device=batch.device, dtype=batch.dtype) if metric=="cosine" else (batch @ batch.T)
    if metric == "cosine": batch_normalized = F.normalize(batch, p=2, dim=1); return batch_normalized @ batch_normalized.T
    elif metric == "dot": return batch @ batch.T
    else: raise ValueError(f"Unsupported similarity metric: {metric}")

def _normalize_matrix_p1(matrix: torch.Tensor) -> torch.Tensor: # Renamed
    if matrix.numel() == 0 : return matrix
    if matrix.numel() == 1: return matrix - matrix.mean()
    mean = matrix.mean(); std = matrix.std()
    if std < 1e-8: return matrix - mean
    return (matrix - mean) / std

def calculate_vector_space_preservation_loss_p1( # Renamed
    source_batch: torch.Tensor,
    wubu_latent_batch: torch.Tensor,
    similarity_metric: str = "cosine",
    normalize_similarity_matrices: bool = True
) -> torch.Tensor:
    if source_batch.shape[0] != wubu_latent_batch.shape[0]:
        logger_losses_p1.error(f"Batch size mismatch VSP: Src N={source_batch.shape[0]}, WuBu N={wubu_latent_batch.shape[0]}.")
        return torch.tensor(0.0, device=source_batch.device, requires_grad=True)
    if source_batch.shape[0] <= 1: return torch.tensor(0.0, device=source_batch.device, requires_grad=True)
    sim_source = _pairwise_similarity_p1(source_batch, metric=similarity_metric)
    sim_wubu = _pairwise_similarity_p1(wubu_latent_batch, metric=similarity_metric)
    if normalize_similarity_matrices: sim_source = _normalize_matrix_p1(sim_source); sim_wubu = _normalize_matrix_p1(sim_wubu)
    return F.mse_loss(sim_source, sim_wubu)
# Adversarial loss functions omitted as they are not used by ETPTrainerPhase1
# End of etp_losses.py (trimmed for Phase 1) content
# =====================================================================

# =====================================================================
# Content from: /draftPY/etp_common/etp_embedding_extractor.py
# (Identical to Phase 2)
# =====================================================================
logger_embedding_extractor_p1 = logging.getLogger("etp_embedding_extractor_p1")
def extract_ds_r1_sentence_embeddings_p1( # Renamed
    texts: List[str], model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device: str = "cuda" if torch.cuda.is_available() else "cpu", batch_size: int = 32
) -> List[np.ndarray]:
    logger_embedding_extractor_p1.info(f"Loading model and tokenizer: {model_name}")
    try:
        if "prajjwal1/bert-tiny" in model_name and device == "cpu": pass
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.to(device); model.eval()
    except Exception as e:
        logger_embedding_extractor_p1.error(f"Error loading model or tokenizer '{model_name}': {e}")
        if "prajjwal1/bert-tiny" in model_name and isinstance(e, OSError): logger_embedding_extractor_p1.error("Bert-tiny not cached/accessible.")
        raise
    all_embeddings: List[np.ndarray] = []
    logger_embedding_extractor_p1.info(f"Processing {len(texts)} texts in batches of {batch_size} on {device}.")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs); last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']; expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * expanded_mask, 1)
                sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
                mean_pooled_embeddings = sum_embeddings / sum_mask
            all_embeddings.extend([emb.cpu().numpy() for emb in mean_pooled_embeddings])
        except Exception as e: logger_embedding_extractor_p1.error(f"Error processing batch {i // batch_size + 1}: {e}"); raise
    logger_embedding_extractor_p1.info(f"Successfully extracted embeddings for {len(all_embeddings)} sentences from model {model_name}.")
    return all_embeddings

def save_embeddings_p1(embeddings: List[np.ndarray], output_path: str, output_format: str = "numpy_list"): # Renamed
    logger_embedding_extractor_p1.info(f"Saving {len(embeddings)} embeddings to {output_path} in {output_format} format.")
    if output_format == "numpy_list":
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir); logger_embedding_extractor_p1.info(f"Created directory: {output_dir}")
            np.savez_compressed(output_path, *embeddings)
            logger_embedding_extractor_p1.info(f"Embeddings successfully saved to {output_path}.")
        except Exception as e: logger_embedding_extractor_p1.error(f"Error saving with np.savez_compressed: {e}"); raise
    elif output_format == "hdf5":
        try:
            if not H5PY_AVAILABLE: logger_embedding_extractor_p1.error("h5py not installed for HDF5 save."); raise ImportError("h5py required.")
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
            with h5py.File(output_path, 'w') as hf:
                for i, emb in enumerate(embeddings): hf.create_dataset(f"embedding_{i}", data=emb)
            logger_embedding_extractor_p1.info(f"Embeddings saved to {output_path} in HDF5 format.")
        except ImportError: raise
        except Exception as e: logger_embedding_extractor_p1.error(f"Error saving HDF5: {e}"); raise
    else: logger_embedding_extractor_p1.error(f"Unsupported output format: {output_format}"); raise ValueError(f"Unsupported output format: {output_format}")
# End of etp_embedding_extractor.py content
# =====================================================================

# =====================================================================
# Content from: /draftPY/etp_common/etp_datasets.py
# (Identical to Phase 2)
# =====================================================================
logger_datasets_p1 = logging.getLogger("etp_datasets_p1")
def load_embeddings_from_file_p1(filepath: str) -> List[np.ndarray]: # Renamed
    if not os.path.exists(filepath) and not os.path.isdir(filepath):
        logger_datasets_p1.error(f"File or directory not found: {filepath}"); raise FileNotFoundError(f"File or directory not found: {filepath}")
    embeddings: List[np.ndarray] = []; file_extension = os.path.splitext(filepath)[1].lower()
    logger_datasets_p1.info(f"Attempting to load embeddings from {filepath} (type: {file_extension if file_extension else 'directory'})")
    if file_extension == ".npz":
        try:
            with np.load(filepath, allow_pickle=True) as data:
                if len(data.files) == 1 and isinstance(data[data.files[0]], (list, np.ndarray)) and data[data.files[0]].dtype == 'object':
                    loaded_obj = data[data.files[0]]
                    if isinstance(loaded_obj, list): embeddings = [np.asarray(emb) for emb in loaded_obj]
                    elif loaded_obj.ndim > 0 and isinstance(loaded_obj[0], np.ndarray): embeddings = [np.asarray(emb) for emb in loaded_obj]
                    else: embeddings = [np.asarray(loaded_obj)]
                elif len(data.files) > 0 and all(f.startswith('arr_') for f in data.files):
                    embeddings = [data[f] for f in sorted(data.files, key=lambda x: int(x.split('_')[1]))]
                elif len(data.files) > 0 : embeddings = [data[f] for f in data.files]
                else: logger_datasets_p1.warning(f"NPZ file {filepath} empty/unexpected structure."); return []
                embeddings = [np.asarray(emb) for emb in embeddings if emb is not None]
                logger_datasets_p1.info(f"Loaded {len(embeddings)} embeddings from NPZ {filepath}.")
        except Exception as e: logger_datasets_p1.error(f"Error loading NPZ {filepath}: {e}"); raise ValueError(f"Could not load NPZ {filepath}: {e}")
    elif file_extension in [".hdf5", ".h5"]:
        if not H5PY_AVAILABLE: logger_datasets_p1.error("h5py required for HDF5."); raise ImportError("h5py required.")
        try:
            with h5py.File(filepath, 'r') as hf:
                dataset_keys = sorted(list(hf.keys()), key=lambda x: int(x.split('_')[1]) if x.startswith('embedding_') and x.split('_')[1].isdigit() else 0)
                for key in dataset_keys: embeddings.append(hf[key][:])
            logger_datasets_p1.info(f"Loaded {len(embeddings)} embeddings from HDF5 {filepath}.")
        except Exception as e: logger_datasets_p1.error(f"Error loading HDF5 {filepath}: {e}"); raise ValueError(f"Could not load HDF5 {filepath}: {e}")
    elif os.path.isdir(filepath):
        logger_datasets_p1.info(f"{filepath} is a directory. Loading .npy files.")
        try:
            npy_files = sorted([f for f in os.listdir(filepath) if f.endswith(".npy")])
            if not npy_files: logger_datasets_p1.warning(f"No .npy files in {filepath}."); return []
            for f_name in npy_files: embeddings.append(np.load(os.path.join(filepath, f_name), allow_pickle=True))
            logger_datasets_p1.info(f"Loaded {len(embeddings)} embeddings from .npy files in {filepath}.")
        except Exception as e: logger_datasets_p1.error(f"Error loading .npy from {filepath}: {e}"); raise ValueError(f"Could not load .npy from {filepath}: {e}")
    else: logger_datasets_p1.error(f"Unsupported file type: {file_extension} for {filepath}"); raise ValueError(f"Unsupported file type: {file_extension}.")
    return embeddings

class DeepSeekR1EmbeddingDataset_P1(Dataset): # Renamed
    def __init__(self, embeddings_file_A: str, embeddings_file_B: str):
        logger_datasets_p1.info(f"Initializing DeepSeekR1EmbeddingDataset_P1: {embeddings_file_A}, {embeddings_file_B}")
        try:
            self.embeddings_A = load_embeddings_from_file_p1(embeddings_file_A)
            self.embeddings_B = load_embeddings_from_file_p1(embeddings_file_B)
        except Exception as e: logger_datasets_p1.error(f"Failed loading embeddings: {e}"); raise
        if not self.embeddings_A: logger_datasets_p1.warning(f"File A ({embeddings_file_A}) empty.")
        if not self.embeddings_B: logger_datasets_p1.warning(f"File B ({embeddings_file_B}) empty.")
        if not self.embeddings_A and not self.embeddings_B: logger_datasets_p1.error("Both sources empty.")
        elif not self.embeddings_A or not self.embeddings_B: logger_datasets_p1.warning("One source empty.")
    def __len__(self) -> int:
        len_A = len(self.embeddings_A) if self.embeddings_A else 0; len_B = len(self.embeddings_B) if self.embeddings_B else 0
        if len_A == 0 and len_B == 0: return 0
        return max(len_A, len_B)
    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, None]]:
        len_A = len(self.embeddings_A) if self.embeddings_A else 0; len_B = len(self.embeddings_B) if self.embeddings_B else 0
        if len_A == 0 and len_B == 0: logger_datasets_p1.warning("Get item from empty dataset."); return {'source_A': None, 'source_B': None}
        emb_A = self.embeddings_A[idx % len_A] if len_A > 0 else None; emb_B = self.embeddings_B[idx % len_B] if len_B > 0 else None
        if emb_A is not None: emb_A = np.asarray(emb_A, dtype=np.float32)
        if emb_B is not None: emb_B = np.asarray(emb_B, dtype=np.float32)
        return {'source_A': emb_A, 'source_B': emb_B}
# End of etp_datasets.py content
# =====================================================================

# =====================================================================
# Content from: /draftPY/etp_common/optimizer_utils.py
# (Identical to Phase 2)
# =====================================================================
logger_optimizer_utils_p1 = logging.getLogger("etp_optimizer_utils_p1")
DEFAULT_CONFIG_QLEARN_HYBRID_P1: Dict[str, Any] = { # Renamed
    "q_table_size": 10, "num_lr_actions": 5, "lr_change_factors": [0.5, 0.9, 1.0, 1.1, 1.5],
    "learning_rate_q": 0.1, "discount_factor_q": 0.9, "exploration_rate_q": 0.1,
    "lr_min": 1e-7, "lr_max": 1e-1, "metric_history_len": 5,
    "loss_min": 0.0, "loss_max": 10.0, "grad_stats_window": 20,
}
class GradientStats_P1: # Renamed
    def __init__(self, window_size: int = 100, device: Optional[torch.device] = None):
        self.window_size = window_size; self.device = device if device else torch.device("cpu")
        self.grad_norms: Deque[float]=deque(maxlen=window_size); self.grad_means: Deque[float]=deque(maxlen=window_size)
        self.grad_stds: Deque[float]=deque(maxlen=window_size); self.update_magnitudes: Deque[float]=deque(maxlen=window_size)
        self.ewma_norm=0.0; self.ewma_mean=0.0; self.ewma_std=0.0; self.alpha=0.1
    @torch.no_grad()
    def update(self, params_with_grad: List[torch.nn.Parameter], current_lr: Optional[float] = None) -> None:
        if not params_with_grad: return
        valid_grads = [p.grad.data.view(-1) for p in params_with_grad if p.grad is not None]
        if not valid_grads: return
        all_grads_flat = torch.cat(valid_grads);
        if all_grads_flat.numel() == 0: return
        all_grads_flat = all_grads_flat.to(self.device)
        current_grad_norm = torch.norm(all_grads_flat,p=2).item(); self.grad_norms.append(current_grad_norm)
        self.ewma_norm = self.alpha * current_grad_norm + (1-self.alpha) * self.ewma_norm
        current_grad_mean=all_grads_flat.mean().item(); current_grad_std=all_grads_flat.std().item()
        self.grad_means.append(current_grad_mean); self.grad_stds.append(current_grad_std)
        self.ewma_mean=self.alpha*current_grad_mean+(1-self.alpha)*self.ewma_mean; self.ewma_std=self.alpha*current_grad_std+(1-self.alpha)*self.ewma_std
        if current_lr is not None: self.update_magnitudes.append(current_lr * current_grad_norm)
    def get_stats(self) -> Dict[str, float]:
        return {"grad_norm_current":self.grad_norms[-1] if self.grad_norms else 0.0,"grad_norm_ewma":self.ewma_norm,
                "grad_mean_current":self.grad_means[-1] if self.grad_means else 0.0,"grad_mean_ewma":self.ewma_mean,
                "grad_std_current":self.grad_stds[-1] if self.grad_stds else 0.0,"grad_std_ewma":self.ewma_std,
                "update_magnitude_current":self.update_magnitudes[-1] if self.update_magnitudes else 0.0}
    def state_dict(self) -> Dict[str, Any]: return {"grad_norms":list(self.grad_norms),"grad_means":list(self.grad_means),"grad_stds":list(self.grad_stds),"update_magnitudes":list(self.update_magnitudes),"ewma_norm":self.ewma_norm,"ewma_mean":self.ewma_mean,"ewma_std":self.ewma_std,"window_size":self.window_size,"alpha":self.alpha}
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.window_size=state_dict.get("window_size",self.window_size); self.grad_norms=deque(state_dict.get("grad_norms",[]),maxlen=self.window_size); self.grad_means=deque(state_dict.get("grad_means",[]),maxlen=self.window_size); self.grad_stds=deque(state_dict.get("grad_stds",[]),maxlen=self.window_size); self.update_magnitudes=deque(state_dict.get("update_magnitudes",[]),maxlen=self.window_size); self.ewma_norm=state_dict.get("ewma_norm",0.0); self.ewma_mean=state_dict.get("ewma_mean",0.0); self.ewma_std=state_dict.get("ewma_std",0.0); self.alpha=state_dict.get("alpha",self.alpha)

class HAKMEMQController_P1: # Renamed
    def __init__(self, initial_lr: float, config: Optional[Dict[str, Any]] = None, logger_suffix: str = ""):
        self.config = config if config is not None else DEFAULT_CONFIG_QLEARN_HYBRID_P1.copy()
        self.initial_lr=initial_lr; self.current_lr=initial_lr; self.logger_suffix=logger_suffix
        self.q_table_size=int(self.config.get("q_table_size",10)); self.q_table_size=max(1,self.q_table_size)
        self.num_actions=int(self.config.get("num_lr_actions",5)); self.lr_change_factors=self.config.get("lr_change_factors",[0.5,0.9,1.0,1.1,1.5])
        if self.num_actions!=len(self.lr_change_factors): self.num_actions=len(self.lr_change_factors)
        self.q_table=torch.zeros((self.q_table_size,self.num_actions))
        self.learning_rate_q=float(self.config.get("learning_rate_q",0.1)); self.discount_factor_q=float(self.config.get("discount_factor_q",0.9)); self.exploration_rate_q=float(self.config.get("exploration_rate_q",0.1))
        self.lr_min=float(self.config.get("lr_min",1e-7)); self.lr_max=float(self.config.get("lr_max",1e-1))
        self.loss_history:Deque[float]=deque(maxlen=int(self.config.get("metric_history_len",5))); self.loss_min=float(self.config.get("loss_min",0.0)); self.loss_max=float(self.config.get("loss_max",10.0))
        if self.loss_min>=self.loss_max: self.loss_max=self.loss_min+10.0
        self.grad_stats=GradientStats_P1(window_size=int(self.config.get("grad_stats_window",20)))
        self.last_action_idx:Optional[int]=None; self.last_state_idx:Optional[int]=None
        logger_optimizer_utils_p1.info(f"HAKMEMQController_P1 ({self.logger_suffix}) init. LR: {self.initial_lr:.2e}, Q-Table: {self.q_table.shape}")
    def _discretize_value(self,value:float,min_val:float,max_val:float,num_bins:int)->int:
        if num_bins<=0:return 0; If value<=min_val:return 0; If value>=max_val:return num_bins-1
        bin_size=(max_val-min_val)/num_bins; If bin_size<=0:return num_bins//2
        return min(int((value-min_val)/bin_size),num_bins-1)
    def _get_current_state_idx(self,current_loss_val:Optional[float])->int:
        if current_loss_val is not None:return self._discretize_value(current_loss_val,self.loss_min,self.loss_max,self.q_table_size)
        return self.q_table_size//2
    def choose_action(self,current_loss_val:Optional[float]=None,params_with_grad:Optional[List[torch.nn.Parameter]]=None)->float:
        if params_with_grad:self.grad_stats.update(params_with_grad,self.current_lr)
        self.last_state_idx=self._get_current_state_idx(current_loss_val)
        if random.random()<self.exploration_rate_q:self.last_action_idx=random.randint(0,self.num_actions-1)
        else:with torch.no_grad():self.last_action_idx=torch.argmax(self.q_table[self.last_state_idx]).item()
        self.current_lr=max(self.lr_min,min(self.current_lr*self.lr_change_factors[self.last_action_idx],self.lr_max)); return self.current_lr
    def log_reward(self,reward:float,current_loss_val:Optional[float]=None)->None:
        if self.last_state_idx is None or self.last_action_idx is None:return
        current_q=self.q_table[self.last_state_idx,self.last_action_idx]; next_state_idx=self._get_current_state_idx(current_loss_val)
        with torch.no_grad():max_next_q=torch.max(self.q_table[next_state_idx]).item()
        new_q=current_q+self.learning_rate_q*(reward+self.discount_factor_q*max_next_q-current_q)
        self.q_table[self.last_state_idx,self.last_action_idx]=new_q
        if current_loss_val is not None:self.loss_history.append(current_loss_val)
    def get_current_lr(self)->float:return self.current_lr
    def state_dict(self)->Dict[str,Any]:return {"current_lr":self.current_lr,"q_table":self.q_table.tolist(),"loss_history":list(self.loss_history),"last_action_idx":self.last_action_idx,"last_state_idx":self.last_state_idx,"initial_lr":self.initial_lr,"config":self.config,"grad_stats_state_dict":self.grad_stats.state_dict()}
    def load_state_dict(self,state_dict:Dict[str,Any])->None:
        self.current_lr=state_dict.get("current_lr",self.initial_lr)
        if "q_table" in state_dict:
            loaded_q=torch.tensor(state_dict["q_table"])
            if loaded_q.shape==self.q_table.shape:self.q_table=loaded_q
            else:logger_optimizer_utils_p1.warning(f"HAKMEM ({self.logger_suffix}): Q-table shape mismatch. Not loading.")
        self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.config.get("metric_history_len",5)); self.last_action_idx=state_dict.get("last_action_idx"); self.last_state_idx=state_dict.get("last_state_idx")
        if "grad_stats_state_dict" in state_dict:self.grad_stats.load_state_dict(state_dict["grad_stats_state_dict"])

class RiemannianEnhancedSGD_P1(optim.Optimizer): # Renamed
    def __init__(self, params: Any, lr: float = 1e-3, q_learning_config: Optional[Dict[str,Any]] = None, q_logger_suffix: str = "", **kwargs: Any):
        param_list = list(params) if not isinstance(params, list) else params
        if not param_list: param_list = [torch.nn.Parameter(torch.zeros(1))]
        defaults = dict(lr=lr, **kwargs); super().__init__(param_list, defaults)
        self.q_controller: Optional[HAKMEMQController_P1] = None
        if q_learning_config: self.q_controller = HAKMEMQController_P1(initial_lr=lr, config=q_learning_config, logger_suffix=f"RESGD_P1_{q_logger_suffix}")
    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]: # type: ignore[override]
        loss: Optional[torch.Tensor] = None
        if closure is not None: with torch.enable_grad(): loss = closure()
        params_with_grad_for_q = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        if self.q_controller:
            current_loss_val = loss.item() if loss is not None else None
            new_lr = self.q_controller.choose_action(current_loss_val=current_loss_val, params_with_grad=params_with_grad_for_q)
            if abs(new_lr - self.param_groups[0]['lr']) > 1e-9 :
                 logger_optimizer_utils_p1.info(f"RESGD_P1 ({self.q_controller.logger_suffix}) LR updated by Q-Ctrl to: {new_lr:.2e}")
                 for group in self.param_groups: group['lr'] = new_lr
        for group in self.param_groups:
            lr_group = group['lr']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data; If grad.is_sparse: raise RuntimeError('RiemannianSGD does not support sparse gradients for now.')
                if hasattr(p, 'manifold') and p.manifold is not None:
                    manifold = p.manifold # type: Manifold
                    p_on_manifold = p.data
                    if hasattr(manifold, 'proj') and callable(manifold.proj): p_on_manifold = manifold.proj(p_on_manifold, group.get('c', getattr(manifold,'c_val',1.0)))
                    riemannian_grad: torch.Tensor
                    if hasattr(manifold, 'egrad2rgrad') and callable(manifold.egrad2rgrad):
                        curvature_val = group.get('c', getattr(manifold,'c_val',1.0)) # type: ignore
                        riemannian_grad = manifold.egrad2rgrad(p_on_manifold, grad, curvature_val)
                    else: logger_optimizer_utils_p1.warning(f"P:{p.shape} manifold no egrad2rgrad. Euclidean grad."); riemannian_grad = grad
                    update_vec = -lr_group * riemannian_grad; new_p_val: torch.Tensor
                    if hasattr(manifold, 'expmap') and callable(manifold.expmap):
                        curvature_val = group.get('c', getattr(manifold,'c_val',1.0)) # type: ignore
                        new_p_val = manifold.expmap(p_on_manifold, update_vec, curvature_val)
                    else: logger_optimizer_utils_p1.warning(f"P:{p.shape} manifold no expmap. Euclidean update."); new_p_val = p_on_manifold + update_vec
                    if hasattr(manifold, 'proj') and callable(manifold.proj): p.data.copy_(manifold.proj(new_p_val, group.get('c', getattr(manifold,'c_val',1.0))))
                    else: p.data.copy_(new_p_val)
                else: p.data.add_(grad, alpha=-lr_group)
        return loss
    def get_q_controller(self) -> Optional[HAKMEMQController_P1]: return self.q_controller
    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        if self.q_controller: state_dict['q_controller'] = self.q_controller.state_dict()
        return state_dict
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self.q_controller and 'q_controller' in state_dict: self.q_controller.load_state_dict(state_dict['q_controller'])
# End of optimizer_utils.py content
# =====================================================================

# =====================================================================
# Content from: /draftPY/etp_phase1_reconstruction/trainer_phase1.py
# =====================================================================
logger_trainer_phase1 = logging.getLogger("etp_trainer_phase1")
# Imports from etp_common are now direct class usages.

class ETPTrainerPhase1:
    def __init__(self,
                 etp_sphere_model: ETP_WuBuText_DS_R1_Sphere,
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 lr_sphere_wubu_core: float, lr_sphere_mlps: float,
                 optimizer_kwargs_wubu_core: Optional[Dict[str, Any]] = None,
                 optimizer_kwargs_mlps: Optional[Dict[str, Any]] = None,
                 lambda_rec: float = 1.0, lambda_vsp: float = 0.0,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 epochs: int = 10, grad_accum_steps: int = 1, use_amp: bool = True,
                 global_max_grad_norm: float = 1.0, q_controller_enabled: bool = True,
                 q_config_sphere_wubu_core: Optional[Dict[str, Any]] = None,
                 q_config_sphere_mlps: Optional[Dict[str, Any]] = None,
                 checkpoint_dir: str = "checkpoints_etp_phase1",
                 log_interval: int = 50, save_interval: int = 500, val_interval_epochs: int = 1,
                 wandb_project: Optional[str] = None, wandb_run_name: Optional[str] = None,
                 best_val_metric_name: str = "val_loss_rec", best_val_metric_higher_is_better: bool = False):

        self.etp_sphere_model = etp_sphere_model.to(device)
        self.train_loader = train_loader; self.val_loader = val_loader
        self.lr_sphere_wubu_core = lr_sphere_wubu_core; self.lr_sphere_mlps = lr_sphere_mlps
        self.optimizer_kwargs_wubu_core = optimizer_kwargs_wubu_core or {}; self.optimizer_kwargs_mlps = optimizer_kwargs_mlps or {}
        self.lambda_rec = lambda_rec; self.lambda_vsp = lambda_vsp
        self.device = device; self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps if grad_accum_steps > 0 else 1
        self.use_amp = use_amp if self.device.type == 'cuda' else False
        self.global_max_grad_norm = global_max_grad_norm if global_max_grad_norm > 0 else -1.0
        self.q_controller_enabled = q_controller_enabled
        _default_q_config_local_p1 = DEFAULT_CONFIG_QLEARN_HYBRID_P1.copy() # Use renamed config
        self.q_config_sphere_wubu_core = q_config_sphere_wubu_core if q_config_sphere_wubu_core is not None else _default_q_config_local_p1.copy()
        self.q_config_sphere_mlps = q_config_sphere_mlps if q_config_sphere_mlps is not None else _default_q_config_local_p1.copy()
        self.checkpoint_dir = Path(checkpoint_dir); self.log_interval = log_interval
        self.save_interval = save_interval; self.val_interval_epochs = val_interval_epochs
        self.current_epoch = 0; self.global_step = 0
        self.best_val_metric = float('-inf') if best_val_metric_higher_is_better else float('inf')
        self.best_val_metric_name = best_val_metric_name; self.best_val_metric_higher_is_better = best_val_metric_higher_is_better
        self.wandb_run = None
        if wandb_project and WANDB_AVAILABLE and wandb is not None: # Check WANDB_AVAILABLE
            try:
                self.wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=self._get_config_dict()) # type: ignore
                if self.wandb_run: wandb.watch(self.etp_sphere_model, log="all", log_freq=max(1,log_interval*5)) # type: ignore
            except Exception as e_wandb_ph1: logger_trainer_phase1.error(f"WandB init failed P1: {e_wandb_ph1}. Disabling.") ; self.wandb_run = None
        self._setup_optimizers_and_q_controllers()
        self.scaler_sphere = GradScaler(enabled=self.use_amp)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger_trainer_phase1.info(f"ETPTrainerPhase1 initialized. Device: {self.device}, AMP: {self.use_amp}")

    def _get_config_dict(self) -> Dict[str, Any]:
        return {"phase": 1, "lr_sphere_wubu_core": self.lr_sphere_wubu_core, "lr_sphere_mlps": self.lr_sphere_mlps,
                "lambda_rec": self.lambda_rec, "lambda_vsp": self.lambda_vsp, "epochs": self.epochs,
                "grad_accum_steps": self.grad_accum_steps, "use_amp": self.use_amp, "global_max_grad_norm": self.global_max_grad_norm,
                "q_controller_enabled": self.q_controller_enabled, "best_val_metric_name": self.best_val_metric_name,
                "best_val_metric_higher_is_better": self.best_val_metric_higher_is_better}

    def _setup_optimizers_and_q_controllers(self) -> None:
        wubu_core_params_ids = set(id(p) for p in self.etp_sphere_model.wubu_core.parameters())
        wubu_core_params_list = [p for p in self.etp_sphere_model.wubu_core.parameters() if p.requires_grad]
        mlp_params_list = [p for n, p in self.etp_sphere_model.named_parameters() if p.requires_grad and id(p) not in wubu_core_params_ids]
        self.q_controllers: Dict[str, Optional[HAKMEMQController_P1]] = {"sphere_wubu_core": None, "sphere_mlps": None} # Use renamed QController
        self.optimizer_sphere_wubu_core = RiemannianEnhancedSGD_P1( # Use renamed RESGD
            wubu_core_params_list if wubu_core_params_list else [nn.Parameter(torch.zeros(1))], lr=self.lr_sphere_wubu_core,
            q_learning_config=self.q_config_sphere_wubu_core if self.q_controller_enabled else None,
            optimizer_type="etp_sphere_wubu_core_phase1", q_logger_suffix="SphereWuBuCoreP1", **self.optimizer_kwargs_wubu_core) # type: ignore
        if self.q_controller_enabled and hasattr(self.optimizer_sphere_wubu_core, 'get_q_controller'):
            self.q_controllers["sphere_wubu_core"] = self.optimizer_sphere_wubu_core.get_q_controller()
        self.optimizer_sphere_mlps = optim.AdamW(mlp_params_list if mlp_params_list else [nn.Parameter(torch.zeros(1))], lr=self.lr_sphere_mlps, **self.optimizer_kwargs_mlps)
        if self.q_controller_enabled: self.q_controllers["sphere_mlps"] = HAKMEMQController_P1(initial_lr=self.lr_sphere_mlps, config=self.q_config_sphere_mlps, logger_suffix="SphereMLPsP1")
        logger_trainer_phase1.info("Phase 1 Optimizers and Q-Controllers (if specified) set up.")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        embeddings_A = batch['source_A'].to(self.device, non_blocking=True)
        raw_losses_dict: Dict[str, float] = {}; loss_total_tensor: Optional[torch.Tensor] = None
        self.etp_sphere_model.train()
        for param in self.etp_sphere_model.parameters(): param.requires_grad = True
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            reconstructed_A = self.etp_sphere_model(embeddings_A)
            loss_rec_tensor = calculate_reconstruction_loss_p1(reconstructed_A, embeddings_A) # Use renamed loss
            raw_losses_dict['loss_rec'] = loss_rec_tensor.item(); loss_total_tensor = self.lambda_rec * loss_rec_tensor
            if self.lambda_vsp > 0:
                latents_A = self.etp_sphere_model.get_latent(embeddings_A)
                loss_vsp_tensor = calculate_vector_space_preservation_loss_p1(embeddings_A, latents_A) # Use renamed loss
                raw_losses_dict['loss_vsp'] = loss_vsp_tensor.item(); loss_total_tensor = loss_total_tensor + (self.lambda_vsp * loss_vsp_tensor) # type: ignore
        if loss_total_tensor is not None: raw_losses_dict['loss_total_phase1'] = loss_total_tensor.item()
        return raw_losses_dict, loss_total_tensor

    def train_epoch(self) -> Dict[str,float]:
        epoch_losses_sum = defaultdict(float); num_batches_this_epoch = 0; batch_times: List[float] = []
        self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True); self.optimizer_sphere_mlps.zero_grad(set_to_none=True)
        for batch_idx, batch_data in enumerate(self.train_loader): # type: ignore
            start_time = time.time()
            step_raw_losses, loss_total_tensor = self._train_step(batch_data) # type: ignore
            if loss_total_tensor is not None: self.scaler_sphere.scale(loss_total_tensor / self.grad_accum_steps).backward() # type: ignore
            for k, v in step_raw_losses.items(): epoch_losses_sum[k] += v
            num_batches_this_epoch +=1; batch_times.append(time.time() - start_time)
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.q_controller_enabled:
                    qc_mlps = self.q_controllers.get("sphere_mlps")
                    if qc_mlps:
                        new_lr_mlps = qc_mlps.choose_action(current_loss_val=step_raw_losses.get('loss_total_phase1'))
                        if new_lr_mlps != self.optimizer_sphere_mlps.param_groups[0]['lr']:
                            logger_trainer_phase1.info(f"QController (SphereMLPsP1): Updating LR to {new_lr_mlps:.2e}")
                            for pg in self.optimizer_sphere_mlps.param_groups: pg['lr'] = new_lr
                if self.global_max_grad_norm > 0:
                    self.scaler_sphere.unscale_(self.optimizer_sphere_wubu_core); self.scaler_sphere.unscale_(self.optimizer_sphere_mlps)
                    torch.nn.utils.clip_grad_norm_(self.etp_sphere_model.parameters(), self.global_max_grad_norm)
                self.scaler_sphere.step(self.optimizer_sphere_wubu_core); self.scaler_sphere.step(self.optimizer_sphere_mlps); self.scaler_sphere.update()
                self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True); self.optimizer_sphere_mlps.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.log_interval > 0 and self.global_step % self.log_interval == 0:
                    avg_bt = sum(batch_times)/len(batch_times) if batch_times else 0; batch_times=[]
                    avg_losses = {f"train_p1/{k}": v_sum/num_batches_this_epoch for k,v_sum in epoch_losses_sum.items()}
                    log_metrics = {**avg_losses, "train_p1/lr_wubu_core": self.optimizer_sphere_wubu_core.param_groups[0]['lr'], "train_p1/lr_mlps": self.optimizer_sphere_mlps.param_groups[0]['lr'], "train_p1/avg_batch_time_ms": avg_bt*1000, "progress/global_step": self.global_step, "progress/epoch": self.current_epoch+1}
                    logger_trainer_phase1.info(f"Epoch {self.current_epoch+1} | Step {self.global_step} | " + " | ".join([f"{k.split('/')[-1]}:{v:.4f}" for k,v in avg_losses.items() if v is not None]))
                    if self.wandb_run: self.wandb_run.log(log_metrics, step=self.global_step) # type: ignore
                if self.save_interval > 0 and self.global_step % self.save_interval == 0: self._save_checkpoint(is_best=False, reason="interval_step_p1")
        self.current_epoch += 1
        return {k: v / num_batches_this_epoch if num_batches_this_epoch > 0 else 0.0 for k,v in epoch_losses_sum.items()}

    def validate_epoch(self) -> Dict[str, float]:
        if not self.val_loader: return {"val_p1_no_loader": 0.0}
        self.etp_sphere_model.eval(); val_losses_sum = defaultdict(float); num_val_batches = 0
        val_losses_sum.update({'val_p1_loss_vsp': 0.0}) # Initialize
        with torch.no_grad():
            for batch_data in self.val_loader: # type: ignore
                embeddings_A = batch_data['source_A'].to(self.device, non_blocking=True) # type: ignore
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    reconstructed_A = self.etp_sphere_model(embeddings_A)
                    val_losses_sum['val_p1_loss_rec'] += calculate_reconstruction_loss_p1(reconstructed_A, embeddings_A).item() # Use renamed
                    if self.lambda_vsp > 0:
                        latents_A = self.etp_sphere_model.get_latent(embeddings_A)
                        val_losses_sum['val_p1_loss_vsp'] += calculate_vector_space_preservation_loss_p1(embeddings_A, latents_A).item() # Use renamed
                num_val_batches +=1
        avg_val_losses = {k: v / num_val_batches if num_val_batches > 0 else 0.0 for k,v in val_losses_sum.items()}
        avg_val_losses['val_p1_combined_loss'] = avg_val_losses.get('val_p1_loss_rec', 0.0) + avg_val_losses.get('val_p1_loss_vsp', 0.0)
        if self.best_val_metric_name not in avg_val_losses: avg_val_losses[self.best_val_metric_name] = avg_val_losses.get('val_p1_loss_rec', float('inf') if not self.best_val_metric_higher_is_better else float('-inf'))
        logger_trainer_phase1.info(f"Validation P1 Epoch {self.current_epoch}: " + " | ".join([f"{k}:{v:.4f}" for k,v in avg_val_losses.items()]))
        if self.wandb_run:
            wandb_log = {f"val_p1/{k.replace('val_p1_','')}": v for k,v in avg_val_losses.items()}; wandb_log["progress/epoch"] = self.current_epoch
            self.wandb_run.log(wandb_log, step=self.global_step) # type: ignore
        return avg_val_losses

    def _save_checkpoint(self, is_best: bool = False, reason: str = "") -> None:
        name_parts = ["checkpoint_p1", reason] if reason else ["checkpoint_p1"]; If is_best: name_parts.append("best")
        name_parts.extend([f"epoch{self.current_epoch}", f"step{self.global_step}"]); filename = "_".join(filter(None, name_parts)) + ".pth.tar"
        filepath = self.checkpoint_dir / filename
        checkpoint_data: Dict[str, Any] = {'phase': 1, 'epoch': self.current_epoch, 'global_step': self.global_step,
            'etp_sphere_model_state_dict': self.etp_sphere_model.state_dict(),
            'optimizer_sphere_wubu_core_state_dict': self.optimizer_sphere_wubu_core.state_dict(),
            'optimizer_sphere_mlps_state_dict': self.optimizer_sphere_mlps.state_dict(),
            'scaler_sphere_state_dict': self.scaler_sphere.state_dict(), 'best_val_metric': self.best_val_metric,
            'best_val_metric_name': self.best_val_metric_name, 'best_val_metric_higher_is_better': self.best_val_metric_higher_is_better}
        for qc_name, qc_instance in self.q_controllers.items():
            if qc_instance: checkpoint_data[f'q_controller_{qc_name}_state_dict'] = qc_instance.state_dict()
        logger_trainer_phase1.info(f"Saving Phase 1 checkpoint to {filepath} (CODING-ONLY: Actual save commented out).")
        # torch.save(checkpoint_data, filepath) # Actual save

    def load_checkpoint(self, path: str, load_optimizers: bool = True, load_q_controllers: bool = True) -> None:
        filepath = Path(path); logger_trainer_phase1.info(f"Attempting to load Phase 1 checkpoint from {filepath} (CODING-ONLY: No actual file read).")
        ckpt: Dict[str, Any] = {'phase':1, 'epoch': 0, 'global_step': 0, 'best_val_metric': float('inf'), 'best_val_metric_name': self.best_val_metric_name, 'best_val_metric_higher_is_better': self.best_val_metric_higher_is_better}
        # self.etp_sphere_model.load_state_dict(ckpt.get('etp_sphere_model_state_dict', {}))
        self.current_epoch = ckpt.get('epoch', 0); self.global_step = ckpt.get('global_step', 0); self.best_val_metric = ckpt.get('best_val_metric', self.best_val_metric)
        # ... (conceptual load of other states)
        if load_q_controllers:
            for qc_name, qc_instance in self.q_controllers.items():
                if qc_instance and f'q_controller_{qc_name}_state_dict' in ckpt: pass # Conceptual: qc_instance.load_state_dict(...)
        logger_trainer_phase1.info(f"Phase 1 Checkpoint conceptually loaded. Resuming from epoch {self.current_epoch + 1}.")

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        if resume_from_checkpoint: self.load_checkpoint(resume_from_checkpoint)
        initial_completed_epochs = self.current_epoch
        logger_trainer_phase1.info(f"Starting Phase 1 training. Target epochs: {self.epochs}. Current completed: {initial_completed_epochs}.")
        for epoch_iter in range(initial_completed_epochs, self.epochs):
            logger_trainer_phase1.info(f"Commencing Phase 1 Epoch {self.current_epoch + 1}/{self.epochs}")
            epoch_train_losses = self.train_epoch()
            if self.q_controller_enabled:
                for opt_name, _ in self.q_controllers.items(): # Iterate through configured Q-controllers
                    qc = self.q_controllers.get(opt_name)
                    if qc: qc.log_reward(-epoch_train_losses.get('loss_total_phase1', float('inf')))
            if self.val_loader and (self.current_epoch % self.val_interval_epochs == 0 or self.current_epoch == self.epochs):
                val_metrics = self.validate_epoch()
                current_val_metric = val_metrics.get(self.best_val_metric_name, float('-inf') if self.best_val_metric_higher_is_better else float('inf'))
                is_better = (current_val_metric > self.best_val_metric) if self.best_val_metric_higher_is_better else (current_val_metric < self.best_val_metric)
                if is_better:
                    self.best_val_metric = current_val_metric; logger_trainer_phase1.info(f"New best P1 val metric ({self.best_val_metric_name}): {self.best_val_metric:.4f}.")
                    self._save_checkpoint(is_best=True, reason=f"best_val_p1_{self.best_val_metric_name.replace('val_p1_','')}")
            if self.save_interval == 0: self._save_checkpoint(is_best=False, reason="end_of_epoch_p1")
        logger_trainer_phase1.info(f"Phase 1 Training completed after {self.current_epoch} epochs.")
        if self.wandb_run and hasattr(self.wandb_run, 'finish'): self.wandb_run.finish() # type: ignore
# End of trainer_phase1.py content
# =====================================================================

# =====================================================================
# Main Script Logic (from /draftPY/etp_phase1_reconstruction/run_phase1.py)
# =====================================================================
logger_run_phase1 = logging.getLogger("etp_run_phase1")

def parse_arguments_phase1() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETP WuBuText DS-R1 - Phase 1 (Reconstruction)")
    parser.add_argument("--embeddings_file_A", required=True, type=str, help="Path to target embeddings for Corpus A (reconstruction target)")
    parser.add_argument("--embeddings_file_B", required=True, type=str, help="Path to (potentially unused) embeddings for Corpus B (dataset class might need it)")
    parser.add_argument("--ds_r1_embedding_dim", type=int, default=768); parser.add_argument("--wubu_initial_tangent_dim", type=int, default=256)
    parser.add_argument("--head_mlp_layers", type=int, default=2); parser.add_argument("--decoder_mlp_layers", type=int, default=2)
    parser.add_argument("--wubu_core_config_json", type=str, default=None, help="Path to JSON for WuBu core config")
    parser.add_argument("--lambda_rec", type=float, default=1.0); parser.add_argument("--lambda_vsp", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--use_amp", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--global_max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_sphere_wubu_core", type=float, default=1e-4); parser.add_argument("--lr_sphere_mlps", type=float, default=1e-4)
    parser.add_argument("--optimizer_kwargs_wubu_core_json", type=str, default='{}'); parser.add_argument("--optimizer_kwargs_mlps_json", type=str, default='{}')
    parser.add_argument("--q_controller_enabled", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--q_config_sphere_wubu_core_json", type=str, default=None); parser.add_argument("--q_config_sphere_mlps_json", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="etp_phase1_reconstruction/checkpoints_phase1_rec"); parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=20); parser.add_argument("--save_interval", type=int, default=500); parser.add_argument("--val_interval_epochs", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="ETP_Phase1_Reconstruction"); parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()
    for json_arg_name_suffix in ['optimizer_kwargs_wubu_core', 'optimizer_kwargs_mlps']:
        json_arg_name_full = f"{json_arg_name_suffix}_json"; json_str_val = getattr(args, json_arg_name_full)
        try: setattr(args, json_arg_name_suffix, json.loads(json_str_val))
        except json.JSONDecodeError as e:
            logger_run_phase1.error(f"Error parsing JSON for --{json_arg_name_full.replace('_','-')}: '{json_str_val}'. Error: {e}")
            setattr(args, json_arg_name_suffix, {})
    return args

def main_phase1(): # Renamed for clarity within combined script
    args = parse_arguments_phase1()
    logger_run_phase1.info("Starting ETP WuBuText DS-R1 - Phase 1 Runner (Combined Script)")
    logger_run_phase1.info(f"Phase 1 Parsed Arguments (sample): epochs={args.epochs}, lambda_rec={args.lambda_rec}")
    run_device = torch.device(args.device); logger_run_phase1.info(f"Device for Phase 1: {run_device}")

    effective_wubu_core_config = DEFAULT_WUBU_TEXT_CONFIG.copy()
    if args.wubu_core_config_json:
        try:
            with open(args.wubu_core_config_json, 'r') as f_wubu: effective_wubu_core_config.update(json.load(f_wubu))
            logger_run_phase1.info(f"Loaded custom WuBu core config for Phase 1 from: {args.wubu_core_config_json}")
        except Exception as e_wubu_json: logger_run_phase1.warning(f"Error with WuBu JSON {args.wubu_core_config_json}: {e_wubu_json}. Using default.")

    q_controller_configs: Dict[str, Optional[Dict[str, Any]]] = {}
    for qc_name_key in ["sphere_wubu_core", "sphere_mlps"]:
        json_path_val = getattr(args, f"q_config_{qc_name_key}_json", None)
        if json_path_val:
            try:
                with open(json_path_val, 'r') as f_qc: q_controller_configs[qc_name_key] = json.load(f_qc)
                logger_run_phase1.info(f"Loaded Q-Controller config for {qc_name_key} (Phase 1) from: {json_path_val}")
            except Exception as e_qc_json: logger_run_phase1.warning(f"Error with Q-Ctrl JSON for {qc_name_key} ({json_path_val}): {e_qc_json}. Defaults used."); q_controller_configs[qc_name_key] = None
        else: q_controller_configs[qc_name_key] = None

    logger_run_phase1.info("Instantiating Dataset and DataLoader for Phase 1...")
    dataset_instance = DeepSeekR1EmbeddingDataset_P1(args.embeddings_file_A, args.embeddings_file_B) # Use renamed
    safe_batch_size = max(1, args.batch_size)
    train_loader_instance = DataLoader(dataset_instance, batch_size=safe_batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader_instance = None # Or configure a validation split if available
    logger_run_phase1.info("Dataset and DataLoader for Phase 1 instantiated.")

    etp_sphere_model_instance = ETP_WuBuText_DS_R1_Sphere(
        ds_r1_embedding_dim=args.ds_r1_embedding_dim, wubu_initial_tangent_dim=args.wubu_initial_tangent_dim,
        wubu_core_config=effective_wubu_core_config, head_mlp_layers=args.head_mlp_layers, decoder_mlp_layers=args.decoder_mlp_layers)
    logger_run_phase1.info("ETP_WuBuText_DS_R1_Sphere for Phase 1 instantiated.")

    trainer_instance = ETPTrainerPhase1(
        etp_sphere_model=etp_sphere_model_instance, train_loader=train_loader_instance, val_loader=val_loader_instance,
        lr_sphere_wubu_core=args.lr_sphere_wubu_core, lr_sphere_mlps=args.lr_sphere_mlps,
        optimizer_kwargs_wubu_core=args.optimizer_kwargs_wubu_core, optimizer_kwargs_mlps=args.optimizer_kwargs_mlps,
        lambda_rec=args.lambda_rec, lambda_vsp=args.lambda_vsp,
        device=run_device, epochs=args.epochs, grad_accum_steps=args.grad_accum_steps, use_amp=args.use_amp,
        global_max_grad_norm=args.global_max_grad_norm, q_controller_enabled=args.q_controller_enabled,
        q_config_sphere_wubu_core=q_controller_configs["sphere_wubu_core"], q_config_sphere_mlps=q_controller_configs["sphere_mlps"],
        checkpoint_dir=args.checkpoint_dir, log_interval=args.log_interval,
        save_interval=args.save_interval, val_interval_epochs=args.val_interval_epochs,
        wandb_project=args.wandb_project, wandb_run_name=args.wandb_run_name)
    logger_run_phase1.info("ETPTrainerPhase1 instantiated.")

    logger_run_phase1.info("Calling ETPTrainerPhase1.train()...")
    trainer_instance.train(resume_from_checkpoint=args.load_checkpoint)
    logger_run_phase1.info("Phase 1 training process finished.")

if __name__ == '__main__':
    script_logger.info("Starting combined ETP script (Phase 1 focused).")
    main_phase1()
    script_logger.info("Combined ETP script (Phase 1) execution completed.")