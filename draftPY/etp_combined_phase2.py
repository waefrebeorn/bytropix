# =====================================================================
# Python Imports and Global Setup
# =====================================================================
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union, Type, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler as AmpGradScaler # For older torch if needed
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
script_logger = logging.getLogger("ETP_Combined_Universal_Script")

EPS = 1e-7
PHI = (1 + math.sqrt(5)) / 2
TAN_VEC_CLAMP_VAL = 1e4
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e8
MIN_WUBU_LEVEL_SCALE = EPS
MAX_WUBU_LEVEL_SCALE = 10.0

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    script_logger.warning("h5py library not found. HDF5 support disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    script_logger.info("wandb not found. WandB logging disabled.")

# =====================================================================
# Common Architectures (WuBu, Discriminator)
# =====================================================================
logger_wubu_arch = logging.getLogger("etp_wubu_architectures")

class Manifold:
    def __init__(self, eps: float = EPS): self.eps = eps
    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def proj(self, x: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def expmap(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def ptransp(self, u: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: raise NotImplementedError
    def an_dist(self, p1: torch.Tensor, p2: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: return torch.sqrt(self.sqdist(p1, p2, c) + self.eps)
    def egrad2rgrad(self, p: torch.Tensor, grad_e: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: return grad_e

class PoincareBall(Manifold):
    def __init__(self, eps: float = EPS, min_norm_scale: float = 1e-5):
        super().__init__(eps=eps); self.name = 'PoincareBall'; self.min_norm_scale = min_norm_scale
    def _get_c_tensor(self, c: Union[float, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if isinstance(c, (float, int)): return torch.tensor(c, dtype=x.dtype, device=x.device)
        if c.ndim == x.ndim: return c
        elif c.ndim == x.ndim - 1: return c.unsqueeze(-1)
        elif c.ndim == 1 and x.ndim > 1 and c.size(0) == x.size(0): return c.view(x.size(0), *([1]*(x.ndim-1)))
        return c
    def _lambda_x(self, x: torch.Tensor, c_val: Union[float, torch.Tensor], keepdim: bool = False) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x); x_sqnorm = torch.sum(x * x, dim=-1, keepdim=keepdim)
        return 2 / (1 - c * x_sqnorm + self.eps)
    def sqdist(self, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p1); sqrt_c = c.sqrt(); mobius_add_result = self.mobius_add(-p1, p2, c, dim=-1)
        c_squeezed = c.squeeze() if c.ndim > 0 else c # Ensure c is scalar or batch-wise for products
        num = 2 * c_squeezed * torch.sum(mobius_add_result.pow(2), dim=-1)
        p1_sqnorm = torch.sum(p1 * p1, dim=-1); p2_sqnorm = torch.sum(p2 * p2, dim=-1)
        den1 = 1 - c_squeezed * p1_sqnorm; den2 = 1 - c_squeezed * p2_sqnorm
        acosh_arg = 1 + num / (den1 * den2 + self.eps)
        dist_c = torch.acosh(torch.clamp_min(acosh_arg, 1.0 + self.eps))
        sqrt_c_squeezed = sqrt_c.squeeze() if sqrt_c.ndim > 0 else sqrt_c
        dist = dist_c / (sqrt_c_squeezed + self.eps)
        return dist.pow(2)
    def proj(self, x: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x); sqrt_c = c.sqrt()
        max_radius = (1.0 - self.min_norm_scale) / (sqrt_c + self.eps); norm_x = x.norm(dim=-1, keepdim=True, p=2)
        cond = norm_x > max_radius; projected_x = x / (norm_x + self.eps) * max_radius
        return torch.where(cond, projected_x, x)
    def proj_tan(self, u: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor: return u
    def expmap(self, p: torch.Tensor, u: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p); sqrt_c = c.sqrt(); u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
        u_norm_clamped = torch.clamp_min(u_norm, self.eps); lambda_p_val = self._lambda_x(p, c, keepdim=True)
        tanh_arg = sqrt_c * lambda_p_val * u_norm / 2
        scale = torch.tanh(tanh_arg) / (sqrt_c * u_norm_clamped + self.eps)
        second_term = u * scale
        return self.mobius_add(p, second_term, c, dim=-1)
    def logmap(self, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p1); sqrt_c = c.sqrt(); sub_p = self.mobius_add(-p1, p2, c, dim=-1)
        sub_p_norm = torch.norm(sub_p, p=2, dim=-1, keepdim=True); sub_p_norm_clamped = torch.clamp_min(sub_p_norm, self.eps)
        lambda_p1_val = self._lambda_x(p1, c, keepdim=True); atanh_arg = sqrt_c * sub_p_norm
        atanh_arg_clamped = torch.clamp(atanh_arg, max=1.0 - self.eps, min=-(1.0-self.eps)) # Clamp both sides
        scale = (2. / (sqrt_c * lambda_p1_val + self.eps)) * torch.atanh(atanh_arg_clamped)
        result = sub_p / (sub_p_norm_clamped + self.eps) * scale
        return result
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, c_val: Union[float, torch.Tensor], dim: int = -1) -> torch.Tensor:
        c = self._get_c_tensor(c_val, x); x_sqnorm = torch.sum(x * x, dim=dim, keepdim=True)
        y_sqnorm = torch.sum(y * y, dim=dim, keepdim=True); xy_dot = torch.sum(x * y, dim=dim, keepdim=True)
        num = (1 + 2 * c * xy_dot + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y
        den = 1 + 2 * c * xy_dot + c**2 * x_sqnorm * y_sqnorm
        return num / (den + self.eps)
    def ptransp(self, u: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor: return u
    def egrad2rgrad(self, p: torch.Tensor, grad_e: torch.Tensor, c_val: Union[float, torch.Tensor]) -> torch.Tensor:
        c = self._get_c_tensor(c_val, p); lambda_p_sq = self._lambda_x(p, c, keepdim=True).pow(2) / 4.0
        return grad_e / (lambda_p_sq + self.eps)

class HyperbolicUtils:
    @staticmethod
    def _eps_like(tensor: torch.Tensor) -> float: return EPS if tensor.dtype == torch.float32 else 1e-15
    @staticmethod
    def poincare_to_lorentz(x_p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_ = PoincareBall()._get_c_tensor(c, x_p); x_norm_sq = torch.sum(x_p * x_p, dim=-1, keepdim=True)
        den = 1.0 - c_ * x_norm_sq + HyperbolicUtils._eps_like(x_p); inv_sqrt_c = 1.0 / (c_.sqrt() + HyperbolicUtils._eps_like(x_p))
        x0 = inv_sqrt_c * (1 + c_ * x_norm_sq) / den; xs = inv_sqrt_c * (2 * x_p) / den
        return torch.cat([x0, xs], dim=-1)
    @staticmethod
    def lorentz_to_poincare(x_l: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_ = PoincareBall()._get_c_tensor(c, x_l); inv_sqrt_c = 1.0 / (c_.sqrt() + HyperbolicUtils._eps_like(x_l))
        x0, xs = x_l[..., 0:1], x_l[..., 1:]; den = x0 + inv_sqrt_c + HyperbolicUtils._eps_like(x_l)
        return inv_sqrt_c * xs / den
    @staticmethod
    def lorentz_tangent_to_poincare_tangent(u_l: torch.Tensor, x_l: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_ = PoincareBall()._get_c_tensor(c, x_l); inv_sqrt_c = 1.0 / (c_.sqrt() + HyperbolicUtils._eps_like(x_l))
        x0, xs = x_l[..., 0:1], x_l[..., 1:]; u0, us = u_l[..., 0:1], u_l[..., 1:]
        den_factor = x0 + inv_sqrt_c + HyperbolicUtils._eps_like(x0)
        term1 = (inv_sqrt_c / den_factor) * us
        term2 = (inv_sqrt_c / (den_factor.pow(2) + HyperbolicUtils._eps_like(den_factor))) * xs * u0
        return term1 - term2
    @staticmethod
    def poincare_tangent_to_lorentz_tangent(u_p: torch.Tensor, x_p: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
        c_ = PoincareBall()._get_c_tensor(c, x_p); sqrt_c = c_.sqrt(); x_p_norm_sq = torch.sum(x_p * x_p, dim=-1, keepdim=True)
        den_inner = 1.0 - c_ * x_p_norm_sq + HyperbolicUtils._eps_like(x_p); den_sq = den_inner.pow(2) + HyperbolicUtils._eps_like(x_p)
        xp_dot_up = torch.sum(x_p * u_p, dim=-1, keepdim=True); inv_s_c_eps = 1.0/(sqrt_c + HyperbolicUtils._eps_like(x_p))
        u0 = (inv_s_c_eps * 4 * c_ * xp_dot_up) / den_sq
        us_num = (2.0 * inv_s_c_eps) * ((1.0 - c_ * x_p_norm_sq) * u_p + 2 * c_ * x_p * xp_dot_up)
        us = us_num / den_sq
        return torch.cat([u0, us], dim=-1)
    @staticmethod
    def minkowski_dot_product(u: torch.Tensor, v: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        res = -u[..., 0:1] * v[..., 0:1] + torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        return res if keepdim else res.squeeze(-1)

def init_weights_general(m: nn.Module, init_type: str = 'xavier_uniform', nonlinearity: str = 'relu', gain_factor: float = 1.0):
    # ... (condensed init_weights_general from previous correct versions)
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        gain = nn.init.calculate_gain(nonlinearity, param=0.2 if nonlinearity == 'leaky_relu' else None) * gain_factor
        if init_type == 'xavier_uniform': nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init_type == 'xavier_normal': nn.init.xavier_normal_(m.weight, gain=gain)
        elif init_type == 'kaiming_uniform': nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0, mode='fan_in')
        elif init_type == 'kaiming_normal': nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0, mode='fan_in')
        elif init_type == 'orthogonal': nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            if init_type in ['kaiming_uniform', 'kaiming_normal']:
                fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0: bound = 1 / math.sqrt(fan_in); nn.init.uniform_(m.bias, -bound, bound)
                else: nn.init.zeros_(m.bias)
            else: nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        if m.weight is not None: nn.init.ones_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, mean=0, std=0.02)

DEFAULT_WUBU_TEXT_CONFIG: Dict[str, Any] = {
    "num_levels": 3, "hyperbolic_dims": [128, 64, 32], "process_sequences": False,
    "initial_curvatures": [1.0, 0.5, 0.25], "learnable_curvatures": True,
    "inter_level_transform_type": "mlp", "inter_level_use_lorentz": True,
    "inter_level_mlp_layers": 2, "inter_level_mlp_dropout": 0.1,
    "final_aggregation_type": "last_level", "final_aggregation_levels": "all",
    "output_tangent_projection_dim": None, "dropout_rate": 0.1,
    "activation_fn": "gelu", "use_layer_norm": True, "init_std_factor": 0.02,
    "ball_class": PoincareBall, "hyperbolic_utils_class": HyperbolicUtils, "eps": EPS,
}

class HyperbolicInterLevelTransform(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any], level_idx: int):
        super().__init__(); self.input_dim_poincare=input_dim; self.output_dim_poincare=output_dim; self.config=config; self.level_idx=level_idx
        BallClass: Type[PoincareBall] = config.get("ball_class", PoincareBall); HypUtilsClass: Type[HyperbolicUtils] = config.get("hyperbolic_utils_class", HyperbolicUtils)
        self.ball=BallClass(eps=config.get("eps", EPS)); self.hyp_utils=HypUtilsClass(); self.use_lorentz=config.get("inter_level_use_lorentz", True)
        self.transform_type=config.get("inter_level_transform_type", "mlp"); self.eps=config.get("eps", EPS)
        mlp_in_dim = (input_dim+1 if self.use_lorentz else input_dim)
        mlp_out_dim = (output_dim+1 if self.use_lorentz else output_dim)
        if self.transform_type == "mlp":
            layers = []; curr_d = mlp_in_dim; num_layers = config.get("inter_level_mlp_layers", 2)
            if num_layers == 0: self.mlp_transform = nn.Linear(mlp_in_dim, mlp_out_dim)
            else:
                for i in range(num_layers):
                    is_last = (i == num_layers - 1); hid_d = max(mlp_out_dim//2, curr_d//2, 1); out_d = mlp_out_dim if is_last else hid_d
                    layers.append(nn.Linear(curr_d, out_d))
                    if not is_last:
                        if config.get("use_layer_norm",True): layers.append(nn.LayerNorm(out_d))
                        layers.append(getattr(nn,config.get("activation_fn","GELU").upper())())
                        if config.get("inter_level_mlp_dropout",0.1)>0: layers.append(nn.Dropout(config.get("inter_level_mlp_dropout",0.1)))
                    curr_d=out_d
                self.mlp_transform = nn.Sequential(*layers)
        else: self.mlp_transform = nn.Linear(mlp_in_dim, mlp_out_dim)
        self.apply(lambda m: init_weights_general(m, gain_factor=config.get("init_std_factor", 0.02)))

    def forward(self, parent_tangent: torch.Tensor, parent_origin: torch.Tensor, parent_c: torch.Tensor, child_c: torch.Tensor) -> torch.Tensor:
        parent_c = self.ball._get_c_tensor(parent_c, parent_tangent).squeeze(); child_c = self.ball._get_c_tensor(child_c, parent_tangent).squeeze()
        if self.use_lorentz:
            parent_o_l = self.hyp_utils.poincare_to_lorentz(parent_origin, parent_c)
            parent_t_l = self.hyp_utils.poincare_tangent_to_lorentz_tangent(parent_tangent, parent_origin, parent_c)
            transformed_t_l = self.mlp_transform(parent_t_l)
            expected_lorentz_dim = self.output_dim_poincare + 1
            if transformed_t_l.shape[-1] != expected_lorentz_dim: # Basic dim check
                raise ValueError(f"HLT Lorentz: Dim mismatch. MLP out {transformed_t_l.shape[-1]}, expected {expected_lorentz_dim}")

            bs = transformed_t_l.size(0); sqrt_child_c = child_c.sqrt(); time_val = 1.0 / (sqrt_child_c + self.eps)
            child_o_l_time = time_val.expand(bs,1).to(transformed_t_l.device, transformed_t_l.dtype) if time_val.ndim==0 else time_val.unsqueeze(-1).to(transformed_t_l.device, transformed_t_l.dtype)
            child_o_l_space = torch.zeros(bs, self.output_dim_poincare, dtype=transformed_t_l.dtype, device=transformed_t_l.device)
            child_o_l_canonical = torch.cat([child_o_l_time, child_o_l_space], dim=-1)
            
            mink_dot_vx = self.hyp_utils.minkowski_dot_product(transformed_t_l, child_o_l_canonical, keepdim=True)
            mink_dot_xx = -1.0 / (child_c + self.eps); mink_dot_xx = mink_dot_xx.unsqueeze(-1) if mink_dot_xx.ndim == child_o_l_canonical.ndim-2 else mink_dot_xx
            projected_t_l = transformed_t_l - (mink_dot_vx / (mink_dot_xx + self.hyp_utils._eps_like(mink_dot_xx))) * child_o_l_canonical
            return self.hyp_utils.lorentz_tangent_to_poincare_tangent(projected_t_l, child_o_l_canonical, child_c)
        else:
            transformed_t_p = self.mlp_transform(parent_tangent)
            if transformed_t_p.shape[-1] != self.output_dim_poincare:
                raise ValueError(f"HLT Poincare: Dim mismatch. MLP out {transformed_t_p.shape[-1]}, expected {self.output_dim_poincare}")
            return transformed_t_p

class HyperbolicWuBuNestingLevel(nn.Module):
    def __init__(self, level_idx: int, input_dim: int, config: Dict[str, Any]):
        super().__init__(); self.level_idx=level_idx; self.input_dim=input_dim; self.config=config
        self.process_sequences = config.get("process_sequences", False) # If true, operates token-wise
        self.hyperbolic_dim_level = config["hyperbolic_dims"][level_idx]
        self.feature_transform = nn.Linear(input_dim, self.hyperbolic_dim_level)
        self.activation = getattr(nn, config.get("activation_fn", "GELU").upper())()
        self.norm = nn.LayerNorm(self.hyperbolic_dim_level) if config.get("use_layer_norm", True) else nn.Identity()
        curv_val = config["initial_curvatures"][level_idx]
        self.curvature_param = nn.Parameter(torch.tensor(float(curv_val))) if config["learnable_curvatures"] else torch.tensor(float(curv_val))
        if not config["learnable_curvatures"]: self.register_buffer('curvature_param_fixed', self.curvature_param)
        self.apply(lambda m: init_weights_general(m, gain_factor=config.get("init_std_factor", 0.02)))

    @property
    def curvature(self) -> torch.Tensor:
        cur_param = self.curvature_param if self.config["learnable_curvatures"] else self.curvature_param_fixed
        return torch.clamp(cur_param, min=self.config.get("eps", EPS))

    def forward(self, input_tangent: torch.Tensor, input_origin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_tangent: (B, D_in) or (B, S, D_in) if process_sequences
        # input_origin: (B, D_in) or (B, S, D_in)
        x = self.feature_transform(input_tangent) # (B, D_level) or (B, S, D_level)
        x = self.activation(x)
        x = self.norm(x) # Operates on last dim
        # Output tangent features are at the origin of this level's manifold
        output_origin = torch.zeros_like(x)
        return output_origin, x, self.curvature

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(); self.config = config.copy() if config is not None else DEFAULT_WUBU_TEXT_CONFIG.copy()
        self.num_levels = self.config["num_levels"]
        self.process_sequences = self.config.get("process_sequences", False) # If true, expects (B,S,D) inputs
        self.levels = nn.ModuleList(); self.inter_level_transforms = nn.ModuleList()
        curr_level_in_dim = input_dim
        for i in range(self.num_levels):
            level_hyper_dim = self.config["hyperbolic_dims"][i]
            self.levels.append(HyperbolicWuBuNestingLevel(i, curr_level_in_dim, self.config))
            if i < self.num_levels - 1:
                next_level_hyper_dim = self.config["hyperbolic_dims"][i+1]
                self.inter_level_transforms.append(HyperbolicInterLevelTransform(level_hyper_dim, next_level_hyper_dim, self.config, i))
                curr_level_in_dim = next_level_hyper_dim
        self._determine_output_dim()
        self.apply(lambda m: init_weights_general(m, gain_factor=self.config.get("init_std_factor", 0.02)))

    def _determine_output_dim(self):
        # ... (condensed _determine_output_dim from previous correct versions)
        agg_levels_cfg = self.config.get("final_aggregation_levels", "all")
        if isinstance(agg_levels_cfg, str) and agg_levels_cfg.lower() == "all": agg_indices = list(range(self.num_levels))
        elif isinstance(agg_levels_cfg, Iterable) and not isinstance(agg_levels_cfg, str): agg_indices = [i for i in agg_levels_cfg if isinstance(i,int) and 0<=i<self.num_levels]
        else: agg_indices = list(range(self.num_levels))
        if not agg_indices and self.num_levels > 0: agg_indices = list(range(self.num_levels))
        
        dims_to_agg = [self.config["hyperbolic_dims"][i] for i in agg_indices if i < len(self.config["hyperbolic_dims"])]
        if not dims_to_agg:
            self.aggregated_output_dim=1; self.output_tangent_dim=1; self.output_tangent_projection=nn.Identity(); return

        agg_type = self.config.get("final_aggregation_type", "last_level")
        if agg_type == "last_level": self.aggregated_output_dim = dims_to_agg[-1]
        elif agg_type == "concat": self.aggregated_output_dim = sum(dims_to_agg)
        else: self.aggregated_output_dim = dims_to_agg[0] # sum/mean assume same dim or projected first

        proj_dim = self.config.get("output_tangent_projection_dim")
        if proj_dim and isinstance(proj_dim, int) and proj_dim > 0:
            self.output_tangent_projection = nn.Linear(self.aggregated_output_dim, proj_dim); self.output_tangent_dim = proj_dim
        else: self.output_tangent_projection = nn.Identity(); self.output_tangent_dim = self.aggregated_output_dim
        if self.output_tangent_dim <= 0: self.output_tangent_dim = 1
        
    def forward(self, initial_tangent: torch.Tensor, initial_origin: Optional[torch.Tensor] = None, output_level_indices: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        curr_tangent = initial_tangent
        curr_origin = initial_origin if initial_origin is not None else torch.zeros_like(initial_tangent)
        
        level_outputs = {} # To store intermediate level outputs if requested
        all_level_feature_tangents = [] # To store all level outputs for final aggregation

        for i in range(self.num_levels):
            level_origin, level_feat_tangent, level_c = self.levels[i](curr_tangent, curr_origin)
            all_level_feature_tangents.append(level_feat_tangent)
            if output_level_indices and i in output_level_indices:
                level_outputs[f'wubu_level_{i}_output'] = level_feat_tangent
            
            if i < self.num_levels - 1:
                next_level_c = self.levels[i+1].curvature
                curr_tangent = self.inter_level_transforms[i](level_feat_tangent, level_origin, level_c, next_level_c)
                curr_origin = torch.zeros_like(curr_tangent) # Next level input tangent is at its origin
        
        # Aggregation
        agg_levels_cfg = self.config.get("final_aggregation_levels", "all")
        if isinstance(agg_levels_cfg, str) and agg_levels_cfg.lower() == "all": agg_indices = list(range(len(all_level_feature_tangents)))
        elif isinstance(agg_levels_cfg, Iterable) and not isinstance(agg_levels_cfg,str): agg_indices = [idx for idx in agg_levels_cfg if isinstance(idx,int) and 0 <= idx < len(all_level_feature_tangents)]
        else: agg_indices = list(range(len(all_level_feature_tangents)))
        if not agg_indices and all_level_feature_tangents: agg_indices = list(range(len(all_level_feature_tangents)))

        to_agg = [all_level_feature_tangents[idx] for idx in agg_indices]
        if not to_agg: # Should not happen if there are levels
            final_agg_tangent = torch.zeros(initial_tangent.size(0), self.output_tangent_dim, dtype=initial_tangent.dtype, device=initial_tangent.device)
        else:
            agg_type = self.config.get("final_aggregation_type", "last_level")
            if agg_type == "last_level": agg_t = to_agg[-1]
            elif agg_type == "concat":
                if self.process_sequences: # Concat along feature dim for sequences (B, S, D_sum)
                    agg_t = torch.cat([t.view(t.shape[0], t.shape[1], -1) for t in to_agg], dim=2)
                else: # Concat along feature dim for vectors (B, D_sum)
                    agg_t = torch.cat(to_agg, dim=-1)
            else: # sum, mean
                try:
                    stacked_t = torch.stack(to_agg, dim=0) # (num_agg, B, S, D) or (num_agg, B, D)
                    agg_t = stacked_t.sum(dim=0) if agg_type == "sum" else stacked_t.mean(dim=0)
                except RuntimeError: agg_t = to_agg[-1] # Fallback if stacking fails (e.g. varying dims)
            final_agg_tangent = self.output_tangent_projection(agg_t)
        
        level_outputs['final_aggregated_tangent'] = final_agg_tangent
        return level_outputs

class AbstractETPTransfusionHead(nn.Module):
    def __init__(self, source_dim: int, tangent_dim: int): super().__init__(); self.source_dim=source_dim; self.tangent_dim=tangent_dim
    def forward(self, source_embedding: torch.Tensor) -> torch.Tensor: raise NotImplementedError

class AbstractETPDecoder(nn.Module):
    def __init__(self, tangent_dim: int, target_dim: int): super().__init__(); self.tangent_dim=tangent_dim; self.target_dim=target_dim
    def forward(self, tangent_vector: torch.Tensor) -> torch.Tensor: raise NotImplementedError

class DeepSeekR1TransfusionHead(AbstractETPTransfusionHead):
    def __init__(self, source_embedding_dim: int, wubu_initial_tangent_dim: int, mlp_hidden_dim_ratio: float=2.0, num_mlp_layers: int=2, activation_fn: str="GELU", use_layer_norm: bool=True, dropout_rate: float=0.1):
        super().__init__(source_embedding_dim, wubu_initial_tangent_dim)
        # ... (condensed MLP construction from previous correct versions)
        layers=[]; curr_dim=source_embedding_dim; hid_dim=int(source_embedding_dim*mlp_hidden_dim_ratio)
        for i in range(num_mlp_layers):
            is_last=(i==num_mlp_layers-1); out_d=wubu_initial_tangent_dim if is_last else hid_dim; layers.append(nn.Linear(curr_dim,out_d))
            if not is_last:
                if use_layer_norm: layers.append(nn.LayerNorm(out_d))
                layers.append(getattr(nn,activation_fn.upper(),nn.GELU())());
                if dropout_rate>0: layers.append(nn.Dropout(dropout_rate))
            curr_dim=out_d
        self.mlp=nn.Sequential(*layers) if layers else nn.Linear(curr_dim, wubu_initial_tangent_dim) # Handle 0 layers
        self.apply(init_weights_general)
    def forward(self,source_embedding:torch.Tensor)->torch.Tensor: return self.mlp(source_embedding)

class WuBuToDeepSeekR1Decoder(AbstractETPDecoder):
    def __init__(self, wubu_tangent_dim: int, source_embedding_dim: int, mlp_hidden_dim_ratio: float=2.0, num_mlp_layers: int=2, activation_fn: str="GELU", use_layer_norm: bool=True, dropout_rate: float=0.1):
        super().__init__(wubu_tangent_dim, source_embedding_dim)
        # ... (condensed MLP construction from previous correct versions)
        layers=[]; curr_dim=wubu_tangent_dim; hid_dim=int(wubu_tangent_dim*mlp_hidden_dim_ratio)
        for i in range(num_mlp_layers):
            is_last=(i==num_mlp_layers-1); out_d=source_embedding_dim if is_last else hid_dim; layers.append(nn.Linear(curr_dim,out_d))
            if not is_last:
                if use_layer_norm: layers.append(nn.LayerNorm(out_d))
                layers.append(getattr(nn,activation_fn.upper(),nn.GELU())());
                if dropout_rate>0: layers.append(nn.Dropout(dropout_rate))
            curr_dim=out_d
        self.mlp=nn.Sequential(*layers) if layers else nn.Linear(curr_dim, source_embedding_dim)
        self.apply(init_weights_general)
    def forward(self,wubu_tangent_vector:torch.Tensor)->torch.Tensor: return self.mlp(wubu_tangent_vector)

class ETP_WuBu_Model(nn.Module): # Renamed for generality
    def __init__(self, source_embedding_dim:int, wubu_initial_tangent_dim:int,
                 wubu_core_config:Optional[Dict]=None,
                 transfusion_head_config:Optional[Dict]=None,
                 main_decoder_config:Optional[Dict]=None,
                 aux_decoder_configs:Optional[Dict[str, Dict]]=None, # e.g. {'level_0_to_teacher_X': {'out_dim':D_teacher, ...}}
                 num_aux_decoders_to_create: int = 0, # Alternative if aux_decoder_configs not detailed
                 ):
        super().__init__()
        self.source_embedding_dim = source_embedding_dim
        self.wubu_initial_tangent_dim = wubu_initial_tangent_dim
        
        _th_cfg = transfusion_head_config or {}
        self.transfusion_head = DeepSeekR1TransfusionHead(source_embedding_dim, wubu_initial_tangent_dim,
                                                          mlp_hidden_dim_ratio=_th_cfg.get("mlp_hidden_dim_ratio", 2.0),
                                                          num_mlp_layers=_th_cfg.get("num_mlp_layers", 2),
                                                          activation_fn=_th_cfg.get("activation_fn", "GELU"))
        
        self.wubu_core_config = wubu_core_config.copy() if wubu_core_config is not None else DEFAULT_WUBU_TEXT_CONFIG.copy()
        self.wubu_core = FullyHyperbolicWuBuNestingModel(wubu_initial_tangent_dim, self.wubu_core_config)
        
        core_final_out_dim = self.wubu_core.output_tangent_dim
        _md_cfg = main_decoder_config or {}
        self.main_decoder = WuBuToDeepSeekR1Decoder(core_final_out_dim, source_embedding_dim, # Main decoder reconstructs source dim
                                                 mlp_hidden_dim_ratio=_md_cfg.get("mlp_hidden_dim_ratio", 2.0),
                                                 num_mlp_layers=_md_cfg.get("num_mlp_layers", 2),
                                                 activation_fn=_md_cfg.get("activation_fn", "GELU"))
        
        self.aux_decoders = nn.ModuleDict()
        if aux_decoder_configs:
            for name, cfg in aux_decoder_configs.items():
                # cfg must specify 'wubu_level_idx_source' and 'teacher_target_dim'
                wubu_level_idx = cfg.get("wubu_level_idx_source")
                if wubu_level_idx is None or not (0 <= wubu_level_idx < self.wubu_core_config["num_levels"]):
                    logger_wubu_arch.error(f"Invalid wubu_level_idx_source: {wubu_level_idx} for aux_decoder {name}. Skipping.")
                    continue
                
                wubu_level_dim = self.wubu_core_config["hyperbolic_dims"][wubu_level_idx]
                teacher_target_dim = cfg.get("teacher_target_dim", source_embedding_dim) # Default to source_embedding_dim

                self.aux_decoders[name] = WuBuToDeepSeekR1Decoder(
                    wubu_level_dim, teacher_target_dim,
                    mlp_hidden_dim_ratio=cfg.get("mlp_hidden_dim_ratio", 1.0), # Smaller aux decoders
                    num_mlp_layers=cfg.get("num_mlp_layers", 1),
                    activation_fn=cfg.get("activation_fn", "GELU")
                )
        elif num_aux_decoders_to_create > 0 and self.wubu_core_config["num_levels"] > 0 : # Fallback if only num_aux is given
             num_to_create = min(num_aux_decoders_to_create, self.wubu_core_config["num_levels"])
             for i in range(num_to_create):
                 level_idx = i # or some other selection logic
                 wubu_level_dim = self.wubu_core_config["hyperbolic_dims"][level_idx]
                 # Assume aux decoders reconstruct to source_embedding_dim by default
                 self.aux_decoders[f'aux_decoder_level_{level_idx}'] = WuBuToDeepSeekR1Decoder(
                     wubu_level_dim, source_embedding_dim, 
                     mlp_hidden_dim_ratio=1.0, num_mlp_layers=1
                 )


        logger_wubu_arch.info(f"ETP_WuBu_Model initialized. WuBu core final out dim: {core_final_out_dim}. Aux decoders: {list(self.aux_decoders.keys())}")

    def forward(self, source_embedding: torch.Tensor, 
                output_level_indices: Optional[List[int]] = None, # For DM mode
                run_aux_decoders: bool = False # For DM mode
                ) -> Dict[str, torch.Tensor]:
        
        # Input source_embedding can be (B, D_src) or (B, S, D_src)
        # Transfusion head might need to handle sequence if wubu_core.process_sequences is True
        is_sequence_input = source_embedding.ndim == 3 and self.wubu_core.process_sequences
        
        if is_sequence_input:
            batch_size, seq_len, _ = source_embedding.shape
            # Reshape to (B*S, D_src) for token-wise processing by transfusion_head if it's a simple MLP
            # Or modify transfusion_head to be Conv1D or apply MLP token-wise
            # Assuming simple MLP for now, so process token-wise then reshape back
            initial_tangent = self.transfusion_head(source_embedding.reshape(batch_size * seq_len, -1))
            initial_tangent = initial_tangent.view(batch_size, seq_len, -1) # (B, S, D_wubu_init)
        else: # Pooled input (B, D_src)
            initial_tangent = self.transfusion_head(source_embedding) # (B, D_wubu_init)

        wubu_core_outputs = self.wubu_core(initial_tangent, output_level_indices=output_level_indices)
        # wubu_core_outputs is a dict like {'wubu_level_X_output': ..., 'final_aggregated_tangent': ...}
        # Each output is (B, D_level) or (B, S, D_level)

        final_aggregated_latent = wubu_core_outputs['final_aggregated_tangent']
        reconstructed_embedding = self.main_decoder(final_aggregated_latent) # (B, D_src) or (B,S,D_src) if decoder handles sequences

        results = {'reconstructed_embedding': reconstructed_embedding,
                   'final_wubu_latent': final_aggregated_latent}
        results.update(wubu_core_outputs) # Add all intermediate wubu outputs

        if run_aux_decoders and self.aux_decoders:
            for name, aux_decoder_module in self.aux_decoders.items():
                # Determine which wubu_level output this aux_decoder corresponds to from its name or config
                # Assuming name format 'aux_decoder_level_X_to_teacher_Y' or 'aux_decoder_level_X'
                level_idx_str = name.split('_')[2] if "level_" in name else None # Very basic parsing
                if level_idx_str and level_idx_str.isdigit():
                    wubu_level_key = f'wubu_level_{level_idx_str}_output'
                    if wubu_level_key in wubu_core_outputs:
                        wubu_level_data = wubu_core_outputs[wubu_level_key]
                        results[f'aux_reconstruction_{name}'] = aux_decoder_module(wubu_level_data)
                    else:
                        logger_wubu_arch.warning(f"Could not find wubu output '{wubu_level_key}' for aux_decoder '{name}'.")
                else:
                     logger_wubu_arch.warning(f"Could not parse level index from aux_decoder name '{name}'.")
        return results

    def get_latent(self, source_embedding: torch.Tensor) -> torch.Tensor: # For ALA mode compatibility
        # Returns only the final aggregated latent
        is_sequence_input = source_embedding.ndim == 3 and self.wubu_core.process_sequences
        if is_sequence_input:
            batch_size, seq_len, _ = source_embedding.shape
            initial_tangent = self.transfusion_head(source_embedding.reshape(batch_size * seq_len, -1))
            initial_tangent = initial_tangent.view(batch_size, seq_len, -1)
        else:
            initial_tangent = self.transfusion_head(source_embedding)
        
        wubu_core_outputs = self.wubu_core(initial_tangent) # No output_level_indices
        return wubu_core_outputs['final_aggregated_tangent']

class LatentDiscriminatorMLP(nn.Module):
    # ... (condensed from previous correct versions)
    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, activation_fn: str = "leaky_relu", use_spectral_norm: bool = False):
        super().__init__(); self.input_dim=input_dim; self.use_spectral_norm=use_spectral_norm
        h_dims = hidden_dims or ([max(1, input_dim//i if i > 0 else input_dim) for i in [1,2]] if input_dim > 2 else [max(4,input_dim*2), max(2,input_dim)])
        if not h_dims: h_dims = [max(1, input_dim//2)]
        self.hidden_dims = [h for h in h_dims if h > 0]

        layers = []; curr_d = input_dim
        for h_dim in self.hidden_dims:
            lin = nn.Linear(curr_d, h_dim); layers.append(spectral_norm(lin) if use_spectral_norm else lin)
            if activation_fn == "leaky_relu": layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation_fn == "relu": layers.append(nn.ReLU(inplace=True))
            # ... other activations or default
            else: layers.append(nn.GELU()) # Defaulting to GELU if not common ones
            curr_d = h_dim
        out_lin = nn.Linear(curr_d, 1); layers.append(spectral_norm(out_lin) if use_spectral_norm else out_lin)
        self.mlp = nn.Sequential(*layers)
        self.apply(lambda m: init_weights_general(m, nonlinearity=activation_fn if activation_fn in ["leaky_relu","relu"] else "linear"))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim: raise ValueError(f"Input D ({x.shape[-1]}) != Discriminator D ({self.input_dim})")
        return self.mlp(x)

# =====================================================================
# Common Losses
# =====================================================================
logger_losses = logging.getLogger("etp_losses")
def calculate_reconstruction_loss(rec_emb: torch.Tensor, orig_emb: torch.Tensor, type: str="mse") -> torch.Tensor:
    # ... (condensed from previous correct versions)
    if rec_emb.shape != orig_emb.shape: logger_losses.warning(f"Shape mismatch rec: {rec_emb.shape}, orig: {orig_emb.shape}");
    if type == "mse": return F.mse_loss(rec_emb, orig_emb)
    if type == "cosine":
        sim_dim = -1 if rec_emb.ndim > 1 else 0
        return (1 - F.cosine_similarity(rec_emb, orig_emb, dim=sim_dim)).mean()
    return F.mse_loss(rec_emb, orig_emb) # Default
def calculate_vsp_loss(src_batch: torch.Tensor, wubu_batch: torch.Tensor, metric: str="cosine", norm_sim: bool=True) -> torch.Tensor:
    # ... (condensed from previous correct versions, using _pairwise_similarity, _normalize_matrix)
    if src_batch.shape[0] != wubu_batch.shape[0] or src_batch.shape[0] <=1: return torch.tensor(0.0, device=src_batch.device, requires_grad=True)
    sim_s = _pairwise_similarity(src_batch, metric); sim_w = _pairwise_similarity(wubu_batch, metric)
    if norm_sim: sim_s=_normalize_matrix(sim_s); sim_w=_normalize_matrix(sim_w)
    return F.mse_loss(sim_s, sim_w)
def calculate_ala_loss_disc(d_out_A: torch.Tensor, d_out_B_detached: torch.Tensor, type: str="bce") -> torch.Tensor:
    # ... (condensed from previous correct versions)
    if type == "bce":
        return (F.binary_cross_entropy_with_logits(d_out_A, torch.ones_like(d_out_A)) + \
               F.binary_cross_entropy_with_logits(d_out_B_detached, torch.zeros_like(d_out_B_detached))) / 2
    return torch.tensor(0.0) # Placeholder for other GAN losses
def calculate_ala_loss_gen(d_out_A_for_g: torch.Tensor, d_out_B_for_g: torch.Tensor, type: str="bce") -> torch.Tensor:
    # ... (condensed from previous correct versions)
    # Assumes generator wants D to classify both as source A (target 1 for D)
    if type == "bce":
        return (F.binary_cross_entropy_with_logits(d_out_A_for_g, torch.ones_like(d_out_A_for_g)) + \
               F.binary_cross_entropy_with_logits(d_out_B_for_g, torch.ones_like(d_out_B_for_g))) / 2
    return torch.tensor(0.0)

# --- Helper for VSP Loss ---
def _pairwise_similarity(batch: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    if batch.ndim != 2: raise ValueError(f"Input batch must be 2D (N, D), but got shape {batch.shape}")
    N, D = batch.shape;
    if N == 0: return torch.empty((0,0), device=batch.device, dtype=batch.dtype)
    if N == 1: return torch.ones((1,1), device=batch.device, dtype=batch.dtype) if metric=="cosine" else (batch @ batch.T)
    if metric == "cosine": batch_norm = F.normalize(batch, p=2, dim=1); return batch_norm @ batch_norm.T
    elif metric == "dot": return batch @ batch.T
    else: raise ValueError(f"Unsupported similarity: {metric}")
def _normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0 : return matrix
    if matrix.numel() == 1: return matrix - matrix.mean()
    mean = matrix.mean(); std = matrix.std()
    return (matrix - mean) / (std + EPS if std < EPS else std)


# =====================================================================
# Datasets
# =====================================================================
logger_datasets = logging.getLogger("etp_datasets")
# _safe_to_float32_numpy_array_local (keep as is)
def _safe_to_float32_numpy_array_local(item: Any, fp_log: str, item_log: str, logger_use: logging.Logger) -> Optional[np.ndarray]:
    if item is None: return None
    try:
        if isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.number): return item.astype(np.float32)
        arr = np.asarray(item, dtype=np.float32)
        if arr.dtype == np.object_: logger_use.warning(f"Skip '{item_log}' from {fp_log} (object array)."); return None
        return arr
    except (ValueError, TypeError): logger_use.warning(f"Skip '{item_log}' from {fp_log} (conversion error)."); return None

# load_embeddings_from_file (for pooled embeddings - keep as is for ALA mode)
def load_pooled_embeddings_from_file(filepath: str) -> List[np.ndarray]: # Renamed for clarity
    # ... (implementation from previous correct version, focused on loading lists of vectors) ...
    # This version assumes the NPZ stores embeddings as arr_0, arr_1 ... or a single 'stacked_array'
    # or HDF5 stores embedding_0, embedding_1 ... or 'stacked_embeddings'
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")
    embeddings: List[np.ndarray] = []
    ext = os.path.splitext(filepath)[1].lower()
    logger_datasets.info(f"Loading POOLED embeddings from {filepath} (type: {ext})")
    if ext == ".npz":
        with np.load(filepath, allow_pickle=True) as data:
            if 'stacked_array' in data:
                stack = _safe_to_float32_numpy_array_local(data['stacked_array'], filepath, 'stacked_array', logger_datasets)
                if stack is not None and stack.ndim > 1: embeddings.extend([stack[i] for i in range(stack.shape[0])])
            else: # Fallback to arr_0, arr_1, ...
                keys = sorted([k for k in data.files if k.startswith('arr_') and k[4:].isdigit()], key=lambda x: int(x[4:]))
                if not keys: keys = sorted(data.files) # If no arr_i, try all sorted
                for key in keys:
                    if key == 'metadata': continue # Skip metadata if stored this way
                    emb = _safe_to_float32_numpy_array_local(data[key], filepath, key, logger_datasets)
                    if emb is not None: embeddings.append(emb)
    elif ext in [".h5", ".hdf5"] and H5PY_AVAILABLE:
        with h5py.File(filepath, 'r') as hf:
            if "stacked_embeddings" in hf:
                stack = _safe_to_float32_numpy_array_local(hf["stacked_embeddings"][:], filepath, "stacked_embeddings", logger_datasets)
                if stack is not None and stack.ndim > 1: embeddings.extend([stack[i] for i in range(stack.shape[0])])
            else:
                keys = sorted([k for k in hf.keys() if k.startswith('embedding_') and k.split('_')[1].isdigit()], key=lambda x: int(x.split('_')[1]))
                for key in keys:
                    emb = _safe_to_float32_numpy_array_local(hf[key][:], filepath, key, logger_datasets)
                    if emb is not None: embeddings.append(emb)
    else: raise ValueError(f"Unsupported file type for pooled embeddings: {ext}")
    logger_datasets.info(f"Loaded {len(embeddings)} pooled embeddings from {filepath}.")
    return embeddings

class PooledEmbeddingDataset(Dataset): # For ALA mode
    def __init__(self, embeddings_file_A: str, embeddings_file_B: str):
        logger_datasets.info(f"Initializing PooledEmbeddingDataset: A='{embeddings_file_A}', B='{embeddings_file_B}'")
        self.embeddings_A = load_pooled_embeddings_from_file(embeddings_file_A)
        self.embeddings_B = load_pooled_embeddings_from_file(embeddings_file_B)
        if not self.embeddings_A or not self.embeddings_B: logger_datasets.warning("One or both pooled embedding sources are empty!")
        self.length = min(len(self.embeddings_A), len(self.embeddings_B)) if self.embeddings_A and self.embeddings_B else 0
        if self.length == 0: logger_datasets.error("Effective length of PooledEmbeddingDataset is 0.")
    def __len__(self) -> int: return self.length
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        if idx >= self.length: raise IndexError("Index out of bounds for PooledEmbeddingDataset.")
        return {'source_A': self.embeddings_A[idx], 'source_B': self.embeddings_B[idx]}

class DissectedDeepSeekNPZDataset(Dataset): # For DM mode
    def __init__(self, dissected_npz_file: str, target_teacher_layers: List[int], seq_len_for_pooling: Optional[int]=None):
        logger_datasets.info(f"Initializing DissectedDeepSeekNPZDataset from: {dissected_npz_file}")
        self.npz_file = dissected_npz_file
        self.target_teacher_layers = sorted(list(set(target_teacher_layers))) # Ensure unique and sorted
        self.seq_len_for_pooling = seq_len_for_pooling # If WuBu levels are pooled, teacher states also need pooling

        with np.load(self.npz_file, allow_pickle=True) as data:
            self.metadata = json.loads(data['metadata'].item()) if 'metadata' in data else {}
            self.num_items = self.metadata.get("num_embeddings_extracted", 0) # num_texts
            if self.num_items == 0: # Try to infer from keys if metadata incomplete
                example_key = next((k for k in data.files if k.startswith("hidden_state_layer_0_item_")), None)
                if example_key:
                    self.num_items = sum(1 for k in data.files if k.startswith("hidden_state_layer_0_item_"))
                    logger_datasets.warning(f"Inferred num_items={self.num_items} from NPZ keys as metadata was incomplete.")
            self.ds_hidden_size = self.metadata.get("embedding_dimension", -1) # This is teacher's hidden_size
            if self.ds_hidden_size == -1 or self.ds_hidden_size == "N/A (no embeddings extracted)":
                # Try to infer from an actual array
                example_key = next((k for k in data.files if k.startswith("hidden_state_layer_0_item_")), None)
                if example_key and example_key in data:
                    self.ds_hidden_size = data[example_key].shape[-1]
                    logger_datasets.warning(f"Inferred ds_hidden_size={self.ds_hidden_size} from NPZ data.")
                else:
                    raise ValueError("Could not determine ds_hidden_size from dissected NPZ metadata or data.")


        if self.num_items == 0: logger_datasets.error("DissectedNPZDataset: No items found based on metadata or keys.")
        logger_datasets.info(f"DissectedNPZDataset: Found {self.num_items} items. Targeting teacher layers: {self.target_teacher_layers}. Teacher hidden_size: {self.ds_hidden_size}.")

    def __len__(self) -> int: return self.num_items

    def __getitem__(self, item_idx: int) -> Dict[str, Any]:
        if item_idx >= self.num_items: raise IndexError("Index out of bounds for DissectedDeepSeekNPZDataset.")
        
        # Lazily load data for the current item_idx to save memory
        with np.load(self.npz_file, allow_pickle=True) as data:
            # Initial input for WuBu_Mimic (DeepSeek-R1 layer 0 embeddings)
            # This is a sequence (seq_len, D_teacher)
            initial_input_key = f"hidden_state_layer_0_item_{item_idx}"
            if initial_input_key not in data: raise KeyError(f"Key {initial_input_key} not found in {self.npz_file}")
            initial_input_embedding_seq = _safe_to_float32_numpy_array_local(data[initial_input_key], self.npz_file, initial_input_key, logger_datasets)
            if initial_input_embedding_seq is None: raise ValueError(f"Failed to load {initial_input_key}")

            item_data = {'initial_input_embedding_sequence': initial_input_embedding_seq}

            target_intermediate_states = {}
            for layer_l in self.target_teacher_layers:
                key = f"hidden_state_layer_{layer_l}_item_{item_idx}"
                if key not in data: logger_datasets.warning(f"Target teacher key {key} not found. Skipping."); continue
                state_seq = _safe_to_float32_numpy_array_local(data[key], self.npz_file, key, logger_datasets)
                if state_seq is None: continue
                
                if self.seq_len_for_pooling is not None: # If WuBu levels are pooled, pool teacher states
                    # Assuming state_seq is (S, D_teacher). Pool over S.
                    target_intermediate_states[f'teacher_layer_{layer_l}_pooled'] = np.mean(state_seq, axis=0)
                else: # WuBu levels process sequences, keep teacher states as sequences
                    target_intermediate_states[f'teacher_layer_{layer_l}_sequence'] = state_seq
            item_data['target_teacher_intermediate_states'] = target_intermediate_states

            # Target for main decoder (e.g., final DeepSeek-R1 layer output)
            # For simplicity, let's assume we reconstruct the pooled initial input for now in DM mode
            # This means the main_decoder of WuBu model tries to autoencode the pooled version of its own input
            if self.seq_len_for_pooling is not None:
                 item_data['target_final_reconstruction_pooled'] = np.mean(initial_input_embedding_seq, axis=0)
            else: # If main decoder also reconstructs sequences
                 item_data['target_final_reconstruction_sequence'] = initial_input_embedding_seq

        return item_data

# =====================================================================
# Optimizer Utils
# =====================================================================
logger_optimizer_utils = logging.getLogger("etp_optimizer_utils")
DEFAULT_CONFIG_QLEARN_HYBRID: Dict[str, Any] = {
    "q_table_size": 10, "num_lr_actions": 5, "lr_change_factors": [0.5, 0.9, 1.0, 1.1, 1.5],
    "learning_rate_q": 0.1, "discount_factor_q": 0.9, "exploration_rate_q": 0.1,
    "lr_min": 1e-7, "lr_max": 1e-1, "metric_history_len": 5,
    "loss_min": 0.0, "loss_max": 10.0, "grad_stats_window": 20,
}
class GradientStats: # ... (condensed from previous correct versions)
    def __init__(self, window_size: int = 20, device: Optional[torch.device] = None):
        self.window_size=window_size; self.device=device or torch.device("cpu"); self.alpha=0.1
        self.grad_norms:Deque[float]=deque(maxlen=window_size); self.grad_means:Deque[float]=deque(maxlen=window_size)
        self.grad_stds:Deque[float]=deque(maxlen=window_size); self.update_magnitudes:Deque[float]=deque(maxlen=window_size)
        self.ewma_norm, self.ewma_mean, self.ewma_std = 0.0, 0.0, 0.0
    @torch.no_grad()
    def update(self, params:List[nn.Parameter], lr:Optional[float]=None):
        grads=[p.grad.data.view(-1) for p in params if p.grad is not None];
        if not grads: return
        flat_grads=torch.cat(grads).to(self.device);
        if flat_grads.numel()==0: return
        norm=torch.norm(flat_grads,p=2).item(); self.grad_norms.append(norm); self.ewma_norm=self.alpha*norm+(1-self.alpha)*self.ewma_norm
        mean=flat_grads.mean().item(); std=flat_grads.std().item(); self.grad_means.append(mean); self.grad_stds.append(std)
        self.ewma_mean=self.alpha*mean+(1-self.alpha)*self.ewma_mean; self.ewma_std=self.alpha*std+(1-self.alpha)*self.ewma_std
        if lr: self.update_magnitudes.append(lr*norm)
    def get_stats(self): return {"grad_norm_ewma":self.ewma_norm, "grad_mean_ewma":self.ewma_mean, "grad_std_ewma":self.ewma_std}
    def state_dict(self): return {k:list(v) if isinstance(v,deque) else v for k,v in self.__dict__.items() if k!="device"}
    def load_state_dict(self, s):
        for k,v in s.items():
            if k!="device" and hasattr(self,k):
                setattr(self, k, deque(v, maxlen=self.window_size) if isinstance(getattr(self,k),deque) else v)

class HAKMEMQController: # ... (condensed from previous correct versions)
    def __init__(self, initial_lr: float, config: Optional[Dict[str, Any]]=None, logger_suffix: str=""):
        self.cfg=config or DEFAULT_CONFIG_QLEARN_HYBRID.copy(); self.lr=initial_lr; self.init_lr=initial_lr; self.suffix=logger_suffix
        self.q_size=max(1,int(self.cfg["q_table_size"])); self.n_actions=len(self.cfg["lr_change_factors"])
        self.q_table=torch.zeros((self.q_size,self.n_actions)); self.lr_q=float(self.cfg["learning_rate_q"]); self.gamma_q=float(self.cfg["discount_factor_q"])
        self.eps_q=float(self.cfg["exploration_rate_q"]); self.lr_min=float(self.cfg["lr_min"]); self.lr_max=float(self.cfg["lr_max"])
        self.losses:Deque[float]=deque(maxlen=int(self.cfg["metric_history_len"])); self.loss_min=float(self.cfg["loss_min"]); self.loss_max=float(self.cfg["loss_max"])
        if self.loss_min>=self.loss_max: self.loss_max=self.loss_min+10.0
        self.grad_stats=GradientStats(window_size=int(self.cfg["grad_stats_window"]))
        self.last_a_idx:Optional[int]=None; self.last_s_idx:Optional[int]=None
    def _discretize(self,v,min_v,max_v,bins): return 0 if v<=min_v else bins-1 if v>=max_v else min(int((v-min_v)/((max_v-min_v)/bins)),bins-1) if bins>0 and max_v>min_v else bins//2
    def _get_state(self,loss): return self._discretize(loss,self.loss_min,self.loss_max,self.q_size) if loss is not None else self.q_size//2
    def choose_action(self,loss:Optional[float]=None,params:Optional[List[nn.Parameter]]=None)->float:
        if params: self.grad_stats.update(params,self.lr)
        self.last_s_idx=self._get_state(loss)
        self.last_a_idx=random.randint(0,self.n_actions-1) if random.random()<self.eps_q else torch.argmax(self.q_table[self.last_s_idx]).item()
        self.lr=max(self.lr_min,min(self.lr*self.cfg["lr_change_factors"][self.last_a_idx],self.lr_max))
        return self.lr
    def log_reward(self,reward:float,loss:Optional[float]=None):
        if self.last_s_idx is None or self.last_a_idx is None: return
        q_curr=self.q_table[self.last_s_idx,self.last_a_idx]; next_s_idx=self._get_state(loss)
        q_next_max=torch.max(self.q_table[next_s_idx]).item()
        self.q_table[self.last_s_idx,self.last_a_idx]=q_curr+self.lr_q*(reward+self.gamma_q*q_next_max-q_curr)
        if loss: self.losses.append(loss)
    def get_current_lr(self): return self.lr
    def state_dict(self): return {k:v.tolist() if isinstance(v,torch.Tensor) else list(v) if isinstance(v,deque) else v for k,v in self.__dict__.items() if k not in ["grad_stats"]} | {"grad_stats_state_dict": self.grad_stats.state_dict()}
    def load_state_dict(self,s):
        for k,v in s.items():
            if k=="grad_stats_state_dict": self.grad_stats.load_state_dict(v)
            elif hasattr(self,k): setattr(self,k,torch.tensor(v) if isinstance(getattr(self,k),torch.Tensor) else deque(v,maxlen=self.cfg["metric_history_len"]) if isinstance(getattr(self,k),deque) else v)

class RiemannianEnhancedSGD(optim.Optimizer): # ... (condensed from previous correct versions)
    def __init__(self, params: Any, lr: float=1e-3, q_cfg: Optional[Dict]=None, q_suffix: str="", **kwargs):
        p_list = list(params) if not isinstance(params,list) else params; p_list = p_list or [nn.Parameter(torch.zeros(1))]
        super().__init__(p_list, dict(lr=lr,**kwargs))
        self.q_ctrl: Optional[HAKMEMQController] = HAKMEMQController(lr,q_cfg,f"RESGD_{q_suffix}") if q_cfg else None
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        if self.q_ctrl:
            params_g = [p for grp in self.param_groups for p in grp['params'] if p.grad is not None]
            new_lr = self.q_ctrl.choose_action(loss.item() if loss else None, params_g)
            if abs(new_lr - self.param_groups[0]['lr']) > EPS:
                for grp in self.param_groups: grp['lr'] = new_lr
        for grp in self.param_groups:
            lr = grp['lr']
            for p in grp['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if hasattr(p,'manifold') and p.manifold:
                    manifold:Manifold = p.manifold; pt = manifold.proj(p.data, grp.get('c',1.0)) if hasattr(manifold,'proj') else p.data
                    rgrad = manifold.egrad2rgrad(pt, grad, grp.get('c',1.0)) if hasattr(manifold,'egrad2rgrad') else grad
                    update = -lr * rgrad
                    new_p = manifold.expmap(pt, update, grp.get('c',1.0)) if hasattr(manifold,'expmap') else pt + update
                    p.data.copy_(manifold.proj(new_p, grp.get('c',1.0)) if hasattr(manifold,'proj') else new_p)
                else: p.data.add_(grad, alpha=-lr)
        return loss
    def get_q_controller(self): return self.q_ctrl
    def state_dict(self): s=super().state_dict(); s['q_controller']=self.q_ctrl.state_dict() if self.q_ctrl else None; return s
    def load_state_dict(self,s): super().load_state_dict(s); self.q_ctrl.load_state_dict(s['q_controller']) if self.q_ctrl and 'q_controller' in s else None

# =====================================================================
# Trainer (Flexible for ALA or DM)
# =====================================================================
logger_trainer = logging.getLogger("etp_trainer_universal")

class ETPUniversalTrainer:
    def __init__(self, args: argparse.Namespace, # Pass all args for flexibility
                 etp_model: ETP_WuBu_Model,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 discriminator_model: Optional[LatentDiscriminatorMLP] = None, # Only for ALA
                 ):
        self.args = args; self.etp_model = etp_model.to(args.device); self.train_loader = train_loader; self.val_loader = val_loader
        self.discriminator_model = discriminator_model.to(args.device) if discriminator_model and args.training_mode == "ala" else None
        
        self.lambda_rec = args.lambda_rec; self.lambda_vsp = args.lambda_vsp
        if args.training_mode == "ala": self.lambda_ala = args.lambda_ala
        elif args.training_mode == "dissection_mimicry":
            self.lambda_distill_intermediate = json.loads(args.lambda_distill_intermediate_json) if isinstance(args.lambda_distill_intermediate_json, str) else args.lambda_distill_intermediate_json
            self.lambda_distill_final = args.lambda_distill_final

        self.device = args.device; self.epochs = args.epochs; self.grad_accum_steps = args.grad_accum_steps
        self.use_amp = args.use_amp if self.device.type == 'cuda' else False
        self.global_max_grad_norm = args.global_max_grad_norm
        
        self.checkpoint_dir = Path(args.checkpoint_dir); self.log_interval=args.log_interval
        self.save_interval=args.save_interval; self.val_interval_epochs=args.val_interval_epochs
        
        self.current_epoch = 0; self.global_step = 0
        self.best_val_metric = float('-inf') if args.best_val_metric_higher_is_better else float('inf')
        self.best_val_metric_name = args.best_val_metric_name
        self.best_val_metric_higher_is_better = args.best_val_metric_higher_is_better
        
        self._setup_optimizers_and_q_controllers()
        self.scaler_etp = AmpGradScaler(enabled=self.use_amp)
        if self.discriminator_model: self.scaler_discriminator = AmpGradScaler(enabled=self.use_amp)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._init_wandb()
        logger_trainer.info(f"ETPUniversalTrainer initialized for mode: {args.training_mode}. Device: {self.device}, AMP: {self.use_amp}")

    def _init_wandb(self):
        self.wandb_run = None
        if self.args.wandb_enabled and WANDB_AVAILABLE and wandb:
            wandb_config = vars(self.args).copy() # Log all parsed args
            # Add more specific configs if needed, e.g., model structure
            wandb_config["effective_wubu_core_config"] = self.etp_model.wubu_core_config
            
            init_kwargs = {"project": self.args.wandb_project, "name": self.args.wandb_run_name, "config": wandb_config, "reinit": True}
            if self.args.wandb_run_id_to_resume: init_kwargs.update({"id": self.args.wandb_run_id_to_resume, "resume": "allow"})
            
            try:
                self.wandb_run = wandb.init(**init_kwargs)
                if self.wandb_run:
                    logger_trainer.info(f"W&B run: {self.wandb_run.name} (ID: {self.wandb_run.id}). URL: {self.wandb_run.url}")
                    wandb.watch(self.etp_model, log="all", log_freq=max(1, self.log_interval*5))
                    if self.discriminator_model: wandb.watch(self.discriminator_model, log="all", log_freq=max(1,self.log_interval*5))
            except Exception as e: logger_trainer.error(f"W&B init failed: {e}. Logging disabled.", exc_info=True); self.wandb_run = None
        else: logger_trainer.info("W&B logging disabled by args or availability.")


    def _setup_optimizers_and_q_controllers(self):
        # ETP Model Optimizers
        wubu_core_ids = set(id(p) for p in self.etp_model.wubu_core.parameters())
        aux_dec_ids = set(id(p) for n,m in self.etp_model.named_modules() if "aux_decoder" in n for p in m.parameters())
        
        wubu_core_params = [p for p in self.etp_model.wubu_core.parameters() if p.requires_grad]
        # MLPs include transfusion_head, main_decoder, and aux_decoders
        mlp_params = [p for id_p, p in {(id(param), param) for name, param in self.etp_model.named_parameters() if param.requires_grad} 
                        if id_p not in wubu_core_ids]


        self.optimizer_etp_wubu_core = RiemannianEnhancedSGD(wubu_core_params or [nn.Parameter(torch.zeros(1,device=self.device))], lr=self.args.lr_sphere_wubu_core,
                                                             q_cfg=json.loads(self.args.q_config_sphere_wubu_core_json) if self.args.q_controller_enabled and self.args.q_config_sphere_wubu_core_json else (DEFAULT_CONFIG_QLEARN_HYBRID if self.args.q_controller_enabled else None),
                                                             q_suffix="ETPWuBuCore", **json.loads(self.args.optimizer_kwargs_wubu_core_json))
        self.optimizer_etp_mlps = optim.AdamW(mlp_params or [nn.Parameter(torch.zeros(1,device=self.device))], lr=self.args.lr_sphere_mlps, **json.loads(self.args.optimizer_kwargs_mlps_json))

        self.q_controllers = {"etp_wubu_core": self.optimizer_etp_wubu_core.get_q_controller(), "etp_mlps": None}
        if self.args.q_controller_enabled:
            self.q_controllers["etp_mlps"] = HAKMEMQController(self.args.lr_sphere_mlps, json.loads(self.args.q_config_sphere_mlps_json) if self.args.q_config_sphere_mlps_json else DEFAULT_CONFIG_QLEARN_HYBRID, "ETPMLPs")

        if self.args.training_mode == "ala" and self.discriminator_model:
            disc_params = list(self.discriminator_model.parameters())
            self.optimizer_discriminator = optim.AdamW(disc_params or [nn.Parameter(torch.zeros(1,device=self.device))], lr=self.args.lr_discriminator, **json.loads(self.args.optimizer_kwargs_discriminator_json))
            if self.args.q_controller_enabled:
                self.q_controllers["discriminator"] = HAKMEMQController(self.args.lr_discriminator, json.loads(self.args.q_config_discriminator_json) if self.args.q_config_discriminator_json else DEFAULT_CONFIG_QLEARN_HYBRID, "Discriminator")
        else:
            self.optimizer_discriminator = None # Explicitly None

        logger_trainer.info(f"Optimizers and Q-Controllers for mode '{self.args.training_mode}' set up.")

    def _train_step_ala(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # ... (ALA train step logic from original ETPTrainerPhase2._train_step) ...
        # Returns raw_losses_dict, loss_d_tensor, loss_g_total_tensor
        emb_A = batch['source_A'].to(self.device, non_blocking=True)
        emb_B = batch['source_B'].to(self.device, non_blocking=True)
        losses, loss_d, loss_g = {}, None, None
        ac_dtype = torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16

        # Train Discriminator
        self.etp_model.eval(); self.discriminator_model.train()
        for p in self.etp_model.parameters(): p.requires_grad=False
        for p in self.discriminator_model.parameters(): p.requires_grad=True
        with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=ac_dtype):
            lat_A_d = self.etp_model.get_latent(emb_A).detach(); lat_B_d = self.etp_model.get_latent(emb_B).detach()
            d_out_A = self.discriminator_model(lat_A_d); d_out_B = self.discriminator_model(lat_B_d)
            loss_d = calculate_ala_loss_disc(d_out_A, d_out_B)
        if loss_d is not None: losses['loss_d_ala'] = loss_d.item()

        # Train ETP Model (Generator)
        self.etp_model.train(); self.discriminator_model.eval()
        for p in self.etp_model.parameters(): p.requires_grad=True
        for p in self.discriminator_model.parameters(): p.requires_grad=False
        with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=ac_dtype):
            etp_outputs_A = self.etp_model(emb_A) # Get full outputs for rec/vsp
            etp_outputs_B = self.etp_model(emb_B) # Only need latent from B for ALA loss with D
            
            lat_A_g = etp_outputs_A['final_wubu_latent']
            lat_B_g = etp_outputs_B['final_wubu_latent'] # Or self.etp_model.get_latent(emb_B)
            
            d_out_A_g = self.discriminator_model(lat_A_g)
            d_out_B_g = self.discriminator_model(lat_B_g) # Pass B's latent through D
            
            loss_g_ala = calculate_ala_loss_gen(d_out_A_g, d_out_B_g) # G wants D to confuse A and B, or make both look like target (1)
            loss_rec = calculate_reconstruction_loss(etp_outputs_A['reconstructed_embedding'], emb_A)
            loss_vsp = calculate_vsp_loss(emb_A, lat_A_g)
            loss_g = self.lambda_ala * loss_g_ala + self.lambda_rec * loss_rec + self.lambda_vsp * loss_vsp
        
        if loss_g is not None: losses['loss_g_total_ala'] = loss_g.item()
        if loss_g_ala is not None: losses['loss_g_ala'] = loss_g_ala.item()
        if loss_rec is not None: losses['loss_rec_ala'] = loss_rec.item()
        if loss_vsp is not None: losses['loss_vsp_ala'] = loss_vsp.item()
        
        for p in self.etp_model.parameters(): p.requires_grad=True # ensure all are back on for next iter
        for p in self.discriminator_model.parameters(): p.requires_grad=True
        return losses, loss_d, loss_g

    def _train_step_dm(self, batch: Dict[str, Any]) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        # Returns raw_losses_dict, total_etp_model_loss_tensor
        initial_input_seq = batch['initial_input_embedding_sequence'].to(self.device) # (B,S,D_teacher)
        target_intermediate_pooled = {k:v.to(self.device) for k,v in batch['target_teacher_intermediate_states'].items()} # Dict of (B,D_teacher)
        target_final_pooled = batch['target_final_reconstruction_pooled'].to(self.device) # (B,D_teacher)
        
        losses, total_etp_loss = {}, torch.tensor(0.0, device=self.device)
        ac_dtype = torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16

        self.etp_model.train()
        with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=ac_dtype):
            # Forward pass ETP_WuBu_Model
            # Determine which WuBu levels to get outputs from based on aux_decoder names or config
            # Assuming student_levels_to_match_teacher_keys is defined in __init__ or args
            # e.g., self.student_teacher_map = {'wubu_level_1_output': 'teacher_layer_5_pooled', ...}
            
            wubu_level_indices_to_output = []
            if hasattr(self.args, 'student_wubu_levels_json') and self.args.student_wubu_levels_json:
                wubu_level_indices_to_output = json.loads(self.args.student_wubu_levels_json)

            etp_outputs = self.etp_model(source_embedding=initial_input_seq, # Pass sequence
                                         output_level_indices=wubu_level_indices_to_output,
                                         run_aux_decoders=True)

            # Intermediate Distillation Losses
            for student_level_idx in wubu_level_indices_to_output:
                aux_decoder_key = f'aux_decoder_level_{student_level_idx}' # Simplified name matching
                # Or a more robust mapping from student level to teacher layer key
                teacher_target_key = self.args.dm_student_teacher_map.get(f'wubu_level_{student_level_idx}_output')

                if aux_decoder_key in self.etp_model.aux_decoders and \
                   f'aux_reconstruction_{aux_decoder_key}' in etp_outputs and \
                   teacher_target_key and teacher_target_key in target_intermediate_pooled:
                    
                    reconstructed_teacher_repr = etp_outputs[f'aux_reconstruction_{aux_decoder_key}']
                    target_teacher_repr = target_intermediate_pooled[teacher_target_key]
                    
                    # Ensure shapes match for loss (e.g. if aux_decoder outputs sequence but target is pooled)
                    if reconstructed_teacher_repr.ndim == 3 and target_teacher_repr.ndim == 2 and self.etp_model.wubu_core.process_sequences:
                        reconstructed_teacher_repr = torch.mean(reconstructed_teacher_repr, dim=1) # Pool student aux output
                    
                    distill_loss = F.mse_loss(reconstructed_teacher_repr, target_teacher_repr)
                    lambda_val = self.lambda_distill_intermediate.get(teacher_target_key, 1.0) # Get specific lambda
                    total_etp_loss += lambda_val * distill_loss
                    losses[f'loss_distill_{teacher_target_key}'] = distill_loss.item()

            # Final Reconstruction Loss (main decoder)
            # Main decoder reconstructs pooled initial input
            loss_final_rec = calculate_reconstruction_loss(etp_outputs['reconstructed_embedding'], target_final_pooled)
            total_etp_loss += self.lambda_distill_final * loss_final_rec
            losses['loss_final_rec_dm'] = loss_final_rec.item()
            
            # Optional VSP loss on final WuBu latent vs pooled initial input
            if self.lambda_vsp > 0:
                # If initial_input_seq is (B,S,D), pool it for VSP comparison with final_wubu_latent (B,D_wubu)
                pooled_initial_input = torch.mean(initial_input_seq, dim=1) if initial_input_seq.ndim == 3 else initial_input_seq
                loss_vsp = calculate_vsp_loss(pooled_initial_input, etp_outputs['final_wubu_latent'])
                total_etp_loss += self.lambda_vsp * loss_vsp
                losses['loss_vsp_dm'] = loss_vsp.item()

        if total_etp_loss is not None : losses['loss_etp_total_dm'] = total_etp_loss.item()
        return losses, total_etp_loss


    def train_epoch(self) -> Dict[str, float]:
        # ... (train_epoch setup: epoch_losses_sum, num_batches_this_epoch, batch_times, zero_grad) ...
        epoch_losses_sum=defaultdict(float); num_batches=0; batch_times_q=deque(maxlen=20)
        self.optimizer_etp_wubu_core.zero_grad(set_to_none=True)
        self.optimizer_etp_mlps.zero_grad(set_to_none=True)
        if self.optimizer_discriminator: self.optimizer_discriminator.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(self.train_loader):
            start_t = time.time()
            # Common batch validation (ensure essential keys are present)
            if self.args.training_mode == "ala" and ('source_A' not in batch_data or 'source_B' not in batch_data):
                logger_trainer.warning(f"ALA mode: Batch {batch_idx+1} missing source_A/B. Skipping."); continue
            if self.args.training_mode == "dissection_mimicry" and ('initial_input_embedding_sequence' not in batch_data or \
                                                                  'target_teacher_intermediate_states' not in batch_data or \
                                                                  'target_final_reconstruction_pooled' not in batch_data): # or sequence
                logger_trainer.warning(f"DM mode: Batch {batch_idx+1} missing required dissected data. Skipping."); continue

            if self.args.training_mode == "ala":
                step_losses, loss_d, loss_g = self._train_step_ala(batch_data)
                if loss_d: self.scaler_discriminator.scale(loss_d / self.grad_accum_steps).backward()
                if loss_g: self.scaler_etp.scale(loss_g / self.grad_accum_steps).backward()
            elif self.args.training_mode == "dissection_mimicry":
                step_losses, loss_etp_total = self._train_step_dm(batch_data)
                if loss_etp_total: self.scaler_etp.scale(loss_etp_total / self.grad_accum_steps).backward()
            else: raise ValueError(f"Unknown training_mode: {self.args.training_mode}")

            for k,v in step_losses.items(): epoch_losses_sum[k] += v
            num_batches+=1; batch_times_q.append(time.time()-start_t)

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Q-Controller updates (simplified)
                if self.args.q_controller_enabled:
                    if self.q_controllers["etp_mlps"]: # For ETP MLPs (transfusion, main/aux decoders)
                        loss_key = 'loss_g_total_ala' if self.args.training_mode == "ala" else 'loss_etp_total_dm'
                        params = [p for grp in self.optimizer_etp_mlps.param_groups for p in grp['params'] if p.grad is not None]
                        new_lr = self.q_controllers["etp_mlps"].choose_action(step_losses.get(loss_key), params)
                        if new_lr != self.optimizer_etp_mlps.param_groups[0]['lr']:
                            for grp in self.optimizer_etp_mlps.param_groups: grp['lr'] = new_lr
                    if self.args.training_mode == "ala" and self.q_controllers.get("discriminator"):
                        params = [p for grp in self.optimizer_discriminator.param_groups for p in grp['params'] if p.grad is not None]
                        new_lr = self.q_controllers["discriminator"].choose_action(step_losses.get('loss_d_ala'), params)
                        if new_lr != self.optimizer_discriminator.param_groups[0]['lr']:
                             for grp in self.optimizer_discriminator.param_groups: grp['lr'] = new_lr
                
                # Grad clipping
                if self.global_max_grad_norm > 0:
                    self.scaler_etp.unscale_(self.optimizer_etp_wubu_core); self.scaler_etp.unscale_(self.optimizer_etp_mlps)
                    torch.nn.utils.clip_grad_norm_(self.etp_model.parameters(), self.global_max_grad_norm)
                    if self.optimizer_discriminator:
                        self.scaler_discriminator.unscale_(self.optimizer_discriminator)
                        torch.nn.utils.clip_grad_norm_(self.discriminator_model.parameters(), self.global_max_grad_norm)
                
                # Optimizer steps
                self.scaler_etp.step(self.optimizer_etp_wubu_core); self.scaler_etp.step(self.optimizer_etp_mlps)
                if self.optimizer_discriminator: self.scaler_discriminator.step(self.optimizer_discriminator)
                
                self.scaler_etp.update()
                if self.optimizer_discriminator: self.scaler_discriminator.update()

                self.optimizer_etp_wubu_core.zero_grad(set_to_none=True); self.optimizer_etp_mlps.zero_grad(set_to_none=True)
                if self.optimizer_discriminator: self.optimizer_discriminator.zero_grad(set_to_none=True)
                
                self.global_step += 1
                # Logging
                if self.log_interval > 0 and self.global_step % self.log_interval == 0:
                    avg_bt = sum(batch_times_q)/len(batch_times_q) if batch_times_q else 0
                    avg_losses = {f"train/{k}": v_sum/num_batches for k,v_sum in epoch_losses_sum.items()}
                    log_msg_parts = [f"{k.split('/')[-1]}:{v:.4f}" for k,v in avg_losses.items()]
                    log_msg_parts.append(f"LR_core:{self.optimizer_etp_wubu_core.param_groups[0]['lr']:.1e}")
                    log_msg_parts.append(f"LR_mlp:{self.optimizer_etp_mlps.param_groups[0]['lr']:.1e}")
                    if self.optimizer_discriminator: log_msg_parts.append(f"LR_D:{self.optimizer_discriminator.param_groups[0]['lr']:.1e}")
                    logger_trainer.info(f"E{self.current_epoch+1} S{self.global_step} | {' | '.join(log_msg_parts)}")
                    if self.wandb_run:
                        wandb_metrics = {**avg_losses, "progress/epoch":self.current_epoch+1,
                                         "train/lr_etp_core":self.optimizer_etp_wubu_core.param_groups[0]['lr'],
                                         "train/lr_etp_mlp":self.optimizer_etp_mlps.param_groups[0]['lr']}
                        if self.optimizer_discriminator: wandb_metrics["train/lr_disc"] = self.optimizer_discriminator.param_groups[0]['lr']
                        self.wandb_run.log(wandb_metrics, step=self.global_step)
                # Saving
                if self.save_interval > 0 and self.global_step % self.save_interval == 0: self._save_checkpoint(reason="interval_step")
        
        self.current_epoch += 1
        return {k:v/num_batches if num_batches > 0 else 0.0 for k,v in epoch_losses_sum.items()}

    def validate_epoch(self) -> Dict[str, float]:
        if not self.val_loader: return {"val_no_loader": 0.0}
        logger_trainer.info(f"--- Validating Epoch {self.current_epoch} ---")
        self.etp_model.eval()
        if self.discriminator_model: self.discriminator_model.eval()
        
        val_losses = defaultdict(float); num_val_batches = 0
        ac_dtype = torch.bfloat16 if self.device.type=='cuda' and torch.cuda.is_bf16_supported() else torch.float16

        with torch.no_grad():
            for batch_data in self.val_loader:
                if self.args.training_mode == "ala":
                    if 'source_A' not in batch_data or 'source_B' not in batch_data: continue
                    emb_A=batch_data['source_A'].to(self.device); emb_B=batch_data['source_B'].to(self.device)
                    with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=ac_dtype):
                        etp_out_A = self.etp_model(emb_A); lat_A = etp_out_A['final_wubu_latent']
                        val_losses['val/loss_rec'] += calculate_reconstruction_loss(etp_out_A['reconstructed_embedding'], emb_A).item()
                        val_losses['val/loss_vsp'] += calculate_vsp_loss(emb_A, lat_A).item()
                        if self.discriminator_model:
                            lat_B = self.etp_model.get_latent(emb_B)
                            d_out_A = self.discriminator_model(lat_A); d_out_B = self.discriminator_model(lat_B)
                            # Simplified D accuracy: (preds_A==1 + preds_B==0) / total
                            acc = ((torch.sigmoid(d_out_A)>0.5).sum() + (torch.sigmoid(d_out_B)<0.5).sum()).item() / (len(d_out_A)*2.0+EPS)
                            val_losses['val/disc_accuracy'] += acc
                elif self.args.training_mode == "dissection_mimicry":
                    if 'initial_input_embedding_sequence' not in batch_data: continue # Basic check
                    initial_input_seq = batch_data['initial_input_embedding_sequence'].to(self.device)
                    target_intermediate_pooled = {k:v.to(self.device) for k,v in batch_data['target_teacher_intermediate_states'].items()}
                    target_final_pooled = batch_data['target_final_reconstruction_pooled'].to(self.device)
                    with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=ac_dtype):
                        wubu_level_indices_to_output = json.loads(self.args.student_wubu_levels_json) if self.args.student_wubu_levels_json else []
                        etp_outputs = self.etp_model(initial_input_seq, output_level_indices=wubu_level_indices_to_output, run_aux_decoders=True)
                        
                        total_distill_loss_val = 0.0
                        for student_level_idx in wubu_level_indices_to_output:
                            aux_rec_key = f'aux_reconstruction_aux_decoder_level_{student_level_idx}' # Match ETP_Model naming
                            teacher_target_key = self.args.dm_student_teacher_map.get(f'wubu_level_{student_level_idx}_output')
                            if aux_rec_key in etp_outputs and teacher_target_key and teacher_target_key in target_intermediate_pooled:
                                rec_teacher = etp_outputs[aux_rec_key]
                                target_teacher = target_intermediate_pooled[teacher_target_key]
                                if rec_teacher.ndim == 3 and target_teacher.ndim == 2: rec_teacher = torch.mean(rec_teacher, dim=1)
                                distill_loss = F.mse_loss(rec_teacher, target_teacher)
                                total_distill_loss_val += distill_loss.item()
                                val_losses[f'val/loss_distill_{teacher_target_key}'] += distill_loss.item()
                        if wubu_level_indices_to_output: val_losses['val/loss_distill_avg'] += total_distill_loss_val / len(wubu_level_indices_to_output)
                        
                        loss_final_rec = calculate_reconstruction_loss(etp_outputs['reconstructed_embedding'], target_final_pooled)
                        val_losses['val/loss_final_rec_dm'] += loss_final_rec.item()
                        if self.lambda_vsp > 0:
                            pooled_initial_input = torch.mean(initial_input_seq, dim=1) if initial_input_seq.ndim == 3 else initial_input_seq
                            val_losses['val/loss_vsp_dm'] += calculate_vsp_loss(pooled_initial_input, etp_outputs['final_wubu_latent']).item()
                num_val_batches += 1
        
        avg_val = {k:v/num_val_batches if num_val_batches>0 else 0.0 for k,v in val_losses.items()}
        # Define combined metric based on mode
        if self.args.training_mode == "ala":
            avg_val['val/combined_ala'] = avg_val.get('val/loss_rec',0) + avg_val.get('val/loss_vsp',0) - avg_val.get('val/disc_accuracy',0)
        elif self.args.training_mode == "dissection_mimicry":
            avg_val['val/combined_dm'] = avg_val.get('val/loss_distill_avg',0) + avg_val.get('val/loss_final_rec_dm',0) + avg_val.get('val/loss_vsp_dm',0)

        logger_trainer.info(f"Validation E{self.current_epoch}: {' | '.join([f'{k}:{v:.4f}' for k,v in avg_val.items()])}")
        if self.wandb_run: self.wandb_run.log({**avg_val, "progress/epoch": self.current_epoch}, step=self.global_step)
        return avg_val

    def _save_checkpoint(self, is_best: bool = False, reason: str = ""):
        # ... (condensed from previous correct versions, includes wandb_run_id)
        name = f"ckpt_{self.args.training_mode}_{reason}{'_best' if is_best else ''}_ep{self.current_epoch}_gs{self.global_step}.pth.tar"
        fp = self.checkpoint_dir / name
        state = {'phase': self.args.training_mode, 'epoch':self.current_epoch, 'global_step':self.global_step,
                 'etp_model_state_dict':self.etp_model.state_dict(),
                 'optimizer_etp_wubu_core_state_dict':self.optimizer_etp_wubu_core.state_dict(),
                 'optimizer_etp_mlps_state_dict':self.optimizer_etp_mlps.state_dict(),
                 'scaler_etp_state_dict':self.scaler_etp.state_dict(),
                 'best_val_metric':self.best_val_metric, 'best_val_metric_name':self.best_val_metric_name,
                 'wandb_run_id': self.wandb_run.id if self.wandb_run else None}
        if self.optimizer_discriminator: state['discriminator_model_state_dict']=self.discriminator_model.state_dict()
        if self.optimizer_discriminator: state['optimizer_discriminator_state_dict']=self.optimizer_discriminator.state_dict()
        if hasattr(self, 'scaler_discriminator'): state['scaler_discriminator_state_dict']=self.scaler_discriminator.state_dict()
        for qc_name, qc_inst in self.q_controllers.items():
            if qc_inst: state[f'q_controller_{qc_name}_state_dict'] = qc_inst.state_dict()
        torch.save(state, fp); logger_trainer.info(f"Checkpoint saved to {fp}")
        if self.wandb_run: # Log to W&B
            try:
                aliases=["latest", f"ep{self.current_epoch}"] + ([reason.replace(" ","_")] if reason else []) + (["best"] if is_best else [])
                art_name = f"ckpt_{self.args.training_mode}_ep{self.current_epoch}_gs{self.global_step}"
                art = wandb.Artifact(art_name, type="model_checkpoint", metadata=vars(self.args)); art.add_file(str(fp))
                self.wandb_run.log_artifact(art, aliases=aliases); logger_trainer.info(f"Logged artifact to W&B: {art_name}")
            except Exception as e_art: logger_trainer.error(f"W&B artifact logging failed: {e_art}")

    def load_checkpoint(self, path: str):
        # ... (condensed load_checkpoint, handles ALA/DM/Phase1 init) ...
        fp = Path(path)
        if not fp.exists(): logger_trainer.warning(f"Checkpoint {fp} not found."); return
        logger_trainer.info(f"Loading checkpoint from {fp}")
        ckpt = torch.load(fp, map_location=self.device, weights_only=False)
        
        ckpt_phase_mode = ckpt.get('phase', 1 if "discriminator" not in path.lower() else "ala") # Guess mode if not in ckpt

        if 'etp_model_state_dict' in ckpt: self.etp_model.load_state_dict(ckpt['etp_model_state_dict'], strict=False) # strict=False for aux_decoders
        elif 'etp_sphere_model_state_dict' in ckpt: self.etp_model.load_state_dict(ckpt['etp_sphere_model_state_dict'], strict=False)
        else: logger_trainer.warning("No ETP model state_dict found in checkpoint.")

        if ckpt_phase_mode == self.args.training_mode or (ckpt_phase_mode == 2 and self.args.training_mode == "ala"): # Resuming same mode
            logger_trainer.info(f"Resuming {self.args.training_mode} mode from checkpoint.")
            if self.discriminator_model and 'discriminator_model_state_dict' in ckpt: self.discriminator_model.load_state_dict(ckpt['discriminator_model_state_dict'])
            if self.optimizer_etp_wubu_core and 'optimizer_etp_wubu_core_state_dict' in ckpt: self.optimizer_etp_wubu_core.load_state_dict(ckpt['optimizer_etp_wubu_core_state_dict'])
            if self.optimizer_etp_mlps and 'optimizer_etp_mlps_state_dict' in ckpt: self.optimizer_etp_mlps.load_state_dict(ckpt['optimizer_etp_mlps_state_dict'])
            if self.optimizer_discriminator and 'optimizer_discriminator_state_dict' in ckpt: self.optimizer_discriminator.load_state_dict(ckpt['optimizer_discriminator_state_dict'])
            if hasattr(self,'scaler_etp') and 'scaler_etp_state_dict' in ckpt: self.scaler_etp.load_state_dict(ckpt['scaler_etp_state_dict'])
            if hasattr(self,'scaler_discriminator') and 'scaler_discriminator_state_dict' in ckpt: self.scaler_discriminator.load_state_dict(ckpt['scaler_discriminator_state_dict'])
            for qc_name, qc_inst in self.q_controllers.items():
                if qc_inst and f'q_controller_{qc_name}_state_dict' in ckpt: qc_inst.load_state_dict(ckpt[f'q_controller_{qc_name}_state_dict'])
            self.current_epoch = ckpt.get('epoch',0); self.global_step = ckpt.get('global_step',0)
            self.best_val_metric = ckpt.get('best_val_metric', self.best_val_metric)
            # W&B resume
            ckpt_wandb_id = ckpt.get('wandb_run_id')
            if ckpt_wandb_id and self.wandb_run is None and self.args.wandb_enabled and WANDB_AVAILABLE and wandb:
                self.args.wandb_run_id_to_resume = ckpt_wandb_id # Set for _init_wandb if called after this
                logger_trainer.info(f"Set wandb_run_id_to_resume from checkpoint: {ckpt_wandb_id}")


        elif ckpt_phase_mode == 1 : # Loading Phase 1 ETP model to start current mode
            logger_trainer.info(f"Initialized ETP model from Phase 1 checkpoint for {self.args.training_mode} mode. Training state reset.")
            # Reset epoch, step, best_metric for new training phase
            self.current_epoch = 0; self.global_step = 0
            self.best_val_metric = float('-inf') if self.args.best_val_metric_higher_is_better else float('inf')
        else:
            logger_trainer.warning(f"Checkpoint mode '{ckpt_phase_mode}' doesn't match current mode '{self.args.training_mode}' and isn't Phase 1. ETP model loaded, but training state reset.")
            self.current_epoch = 0; self.global_step = 0
            self.best_val_metric = float('-inf') if self.args.best_val_metric_higher_is_better else float('inf')


    def train(self, resume_from_checkpoint: Optional[str] = None):
        if resume_from_checkpoint: self.load_checkpoint(resume_from_checkpoint)
        # Re-init W&B if load_checkpoint might have set wandb_run_id_to_resume
        if self.args.wandb_enabled and self.args.wandb_run_id_to_resume and self.wandb_run is None :
            logger_trainer.info("Re-initializing W&B due to run ID found in checkpoint.")
            self._init_wandb() # Try to init/resume

        start_epoch = self.current_epoch
        logger_trainer.info(f"Starting training for mode '{self.args.training_mode}'. Target epochs: {self.epochs}. Start epoch: {start_epoch + 1}")
        for _ in range(start_epoch, self.epochs):
            logger_trainer.info(f"Commencing Epoch {self.current_epoch + 1}/{self.epochs}")
            train_losses = self.train_epoch() # Increments self.current_epoch

            # Log Q-controller rewards
            if self.args.q_controller_enabled:
                loss_key_g = 'loss_g_total_ala' if self.args.training_mode == "ala" else 'loss_etp_total_dm'
                if self.q_controllers["etp_wubu_core"]: self.q_controllers["etp_wubu_core"].log_reward(-train_losses.get(loss_key_g, float('inf')), train_losses.get(loss_key_g))
                if self.q_controllers["etp_mlps"]: self.q_controllers["etp_mlps"].log_reward(-train_losses.get(loss_key_g, float('inf')), train_losses.get(loss_key_g))
                if self.args.training_mode == "ala" and self.q_controllers.get("discriminator"):
                    self.q_controllers["discriminator"].log_reward(-train_losses.get('loss_d_ala', float('inf')), train_losses.get('loss_d_ala'))
            
            # Validation
            if self.val_loader and (self.current_epoch % self.val_interval_epochs == 0 or self.current_epoch == self.epochs):
                val_metrics = self.validate_epoch()
                current_val = val_metrics.get(self.best_val_metric_name, float('inf') if not self.best_val_metric_higher_is_better else float('-inf'))
                is_better = (current_val > self.best_val_metric) if self.best_val_metric_higher_is_better else (current_val < self.best_val_metric)
                if is_better: self.best_val_metric = current_val; self._save_checkpoint(is_best=True, reason=f"best_val_{self.args.training_mode}")
            
            if self.save_interval == 0 and self.epochs > 0: self._save_checkpoint(reason=f"end_of_epoch_{self.args.training_mode}")
        
        logger_trainer.info(f"Training finished after {self.current_epoch} epochs.")
        if self.wandb_run: self.wandb_run.finish()

# =====================================================================
# Main Script Logic
# =====================================================================
def parse_arguments_universal() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETP Universal Trainer (ALA or Dissection Mimicry)")
    # Common args
    parser.add_argument("--training_mode", type=str, required=True, choices=["ala", "dissection_mimicry"], help="Training mode.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load (Phase 1 or same mode Phase 2).")
    parser.add_argument("--checkpoint_dir", type=str, default="etp_universal_checkpoints")
    # ETP Model Structure
    parser.add_argument("--ds_r1_embedding_dim", type=int, default=1536, help="Dim of source embeddings (e.g., DeepSeek-R1).")
    parser.add_argument("--wubu_initial_tangent_dim", type=int, default=256)
    parser.add_argument("--wubu_core_config_json", type=str, default=None)
    parser.add_argument("--transfusion_head_config_json", type=str, default='{}')
    parser.add_argument("--main_decoder_config_json", type=str, default='{}')
    # Optimizers & Training
    parser.add_argument("--lr_sphere_wubu_core", type=float, default=1e-4); parser.add_argument("--lr_sphere_mlps", type=float, default=1e-4)
    parser.add_argument("--optimizer_kwargs_wubu_core_json", type=str, default='{}'); parser.add_argument("--optimizer_kwargs_mlps_json", type=str, default='{}')
    parser.add_argument("--grad_accum_steps", type=int, default=1); parser.add_argument("--use_amp", type=lambda x: (str(x).lower()=='true'), default=True)
    parser.add_argument("--global_max_grad_norm", type=float, default=1.0)
    # Q-Learning
    parser.add_argument("--q_controller_enabled", type=lambda x: (str(x).lower()=='true'), default=True)
    parser.add_argument("--q_config_sphere_wubu_core_json", type=str, default=None); parser.add_argument("--q_config_sphere_mlps_json", type=str, default=None)
    # Logging & Saving
    parser.add_argument("--log_interval", type=int, default=50); parser.add_argument("--save_interval", type=int, default=500); parser.add_argument("--val_interval_epochs", type=int, default=1)
    parser.add_argument("--best_val_metric_name", type=str, default="val/combined_loss") # Default, may need mode-specific
    parser.add_argument("--best_val_metric_higher_is_better", type=lambda x: (str(x).lower()=='true'), default=False)
    # W&B
    parser.add_argument("--wandb_enabled", type=lambda x: (str(x).lower()=='true'), default=True)
    parser.add_argument("--wandb_project", type=str, default="ETP_Universal"); parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id_to_resume", type=str, default=None)

    # --- ALA Mode Specific Arguments ---
    ala_group = parser.add_argument_group("ALA Mode Specific")
    ala_group.add_argument("--embeddings_file_A", type=str, help="Path to pooled embeddings for Corpus A (ALA mode).")
    ala_group.add_argument("--embeddings_file_B", type=str, help="Path to pooled embeddings for Corpus B (ALA mode).")
    ala_group.add_argument("--lr_discriminator", type=float, default=2e-4)
    ala_group.add_argument("--optimizer_kwargs_discriminator_json", type=str, default='{}')
    ala_group.add_argument("--q_config_discriminator_json", type=str, default=None)
    ala_group.add_argument("--disc_input_dim_override", type=int, default=None, help="Manually set discriminator input dim if needed.")
    ala_group.add_argument("--disc_hidden_dims_json", type=str, default='[256, 128]')
    ala_group.add_argument("--disc_activation_fn", type=str, default="leaky_relu")
    ala_group.add_argument("--disc_use_spectral_norm", type=lambda x: (str(x).lower()=='true'), default=True)
    ala_group.add_argument("--lambda_ala", type=float, default=0.1)
    ala_group.add_argument("--lambda_rec", type=float, default=1.0) # Common, but can be ALA specific too
    ala_group.add_argument("--lambda_vsp", type=float, default=0.01) # Common

    # --- Dissection Mimicry (DM) Mode Specific Arguments ---
    dm_group = parser.add_argument_group("Dissection Mimicry Mode Specific")
    dm_group.add_argument("--dissected_data_file_train", type=str, help="Path to NPZ with dissected states for training (DM mode).")
    dm_group.add_argument("--dissected_data_file_val", type=str, default=None, help="Path to NPZ with dissected states for validation (DM mode).")
    dm_group.add_argument("--dm_target_teacher_layers_json", type=str, default="[]", help="JSON list of teacher layer indices to target for distillation.")
    dm_group.add_argument("--dm_student_wubu_levels_json", type=str, default="[]", help="JSON list of student WuBu level indices to use for distillation.")
    dm_group.add_argument("--dm_aux_decoder_config_json", type=str, default='{"num_mlp_layers":1, "mlp_hidden_dim_ratio":1.0}', help="JSON config for aux decoders.")
    dm_group.add_argument("--dm_lambda_distill_intermediate_json", type=str, default='{}', help="JSON dict of lambdas for intermediate distill losses, e.g. {'teacher_layer_5':0.1}")
    dm_group.add_argument("--dm_lambda_distill_final", type=float, default=1.0, help="Lambda for final reconstruction in DM mode.")
    dm_group.add_argument("--dm_wubu_core_processes_sequences", type=lambda x: (str(x).lower()=='true'), default=False, help="If WuBu core levels process sequences (DM mode).")
    dm_group.add_argument("--dm_pool_teacher_states", type=lambda x: (str(x).lower()=='true'), default=True, help="If teacher states should be pooled for DM loss (if WuBu levels are pooled).")


    args = parser.parse_args()
    args.device = torch.device(args.device)
    
    # Parse all JSON string args
    for arg_name in dir(args):
        if arg_name.endswith("_json"):
            json_str_val = getattr(args, arg_name)
            actual_arg_name = arg_name[:-len("_json")]
            try:
                setattr(args, actual_arg_name, json.loads(json_str_val) if json_str_val else ({} if "kwargs" in actual_arg_name or "config" in actual_arg_name else []))
            except json.JSONDecodeError as e:
                script_logger.error(f"Error parsing JSON for --{arg_name.replace('_','-')}: '{json_str_val}'. Error: {e}")
                setattr(args, actual_arg_name, {} if "kwargs" in actual_arg_name or "config" in actual_arg_name else [])
    
    # Mode-specific validation
    if args.training_mode == "ala" and (not args.embeddings_file_A or not args.embeddings_file_B):
        parser.error("For ALA mode, --embeddings_file_A and --embeddings_file_B are required.")
    if args.training_mode == "dissection_mimicry" and not args.dissected_data_file_train:
        parser.error("For Dissection Mimicry mode, --dissected_data_file_train is required.")
    if args.training_mode == "dissection_mimicry":
        if len(args.dm_target_teacher_layers) != len(args.dm_student_wubu_levels) and (args.dm_target_teacher_layers and args.dm_student_wubu_levels):
             parser.error("In DM mode, --dm_target_teacher_layers_json and --dm_student_wubu_levels_json must have the same number of elements if both provided.")
        # Create dm_student_teacher_map
        args.dm_student_teacher_map = {}
        for i, s_level_idx in enumerate(args.dm_student_wubu_levels):
            if i < len(args.dm_target_teacher_layers):
                t_layer_key = f'teacher_layer_{args.dm_target_teacher_layers[i]}' # Key used in dataset
                t_layer_key_pooled = f'teacher_layer_{args.dm_target_teacher_layers[i]}_{"pooled" if args.dm_pool_teacher_states else "sequence"}'
                args.dm_student_teacher_map[f'wubu_level_{s_level_idx}_output'] = t_layer_key_pooled


    # Override W&B project/run name if mode-specific alternatives make sense
    if not args.wandb_run_name:
        args.wandb_run_name = f"{args.training_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not args.wandb_project.endswith(f"_{args.training_mode.upper()}"):
        args.wandb_project = f"{args.wandb_project}_{args.training_mode.upper()}"


    return args

def main_universal():
    args = parse_arguments_universal()
    script_logger.info(f"Starting ETP Universal Trainer in '{args.training_mode}' mode.")
    script_logger.info(f"Parsed Arguments (sample): epochs={args.epochs}, device={args.device}, batch_size={args.batch_size}")

    # --- WuBu Core Config ---
    effective_wubu_core_config = DEFAULT_WUBU_TEXT_CONFIG.copy()
    if args.wubu_core_config_json: # Assuming it's a path
        try:
            with open(args.wubu_core_config_json, 'r') as f: effective_wubu_core_config.update(json.load(f))
        except Exception as e: script_logger.warning(f"Error loading WuBu core JSON: {e}. Using defaults.")
    
    # Override process_sequences in wubu_core_config for DM mode
    if args.training_mode == "dissection_mimicry":
        effective_wubu_core_config["process_sequences"] = args.dm_wubu_core_processes_sequences
        script_logger.info(f"DM Mode: WuBu core process_sequences set to {args.dm_wubu_core_processes_sequences}")
    else: # ALA mode assumes pooled inputs to wubu_core
        effective_wubu_core_config["process_sequences"] = False


    # --- ETP Model Instantiation ---
    aux_decoder_configs_for_etp_model = None
    if args.training_mode == "dissection_mimicry":
        aux_decoder_configs_for_etp_model = {}
        for i, student_level_idx in enumerate(args.dm_student_wubu_levels):
            teacher_layer_idx = args.dm_target_teacher_layers[i]
            # Name aux decoders consistently, e.g., mapping student WuBu level to a teacher characteristic
            aux_decoder_name = f'aux_decoder_level_{student_level_idx}_to_teacher_layer_{teacher_layer_idx}'
            aux_decoder_configs_for_etp_model[aux_decoder_name] = {
                "wubu_level_idx_source": student_level_idx,
                "teacher_target_dim": args.ds_r1_embedding_dim, # Assuming all teacher layers have this dim
                **args.dm_aux_decoder_config # Base config for MLP structure
            }
        script_logger.info(f"DM Mode: Aux decoder configs prepared: {list(aux_decoder_configs_for_etp_model.keys())}")


    etp_model = ETP_WuBu_Model(
        source_embedding_dim=args.ds_r1_embedding_dim,
        wubu_initial_tangent_dim=args.wubu_initial_tangent_dim,
        wubu_core_config=effective_wubu_core_config,
        transfusion_head_config=args.transfusion_head_config,
        main_decoder_config=args.main_decoder_config,
        aux_decoder_configs=aux_decoder_configs_for_etp_model
    )

    # --- Discriminator (ALA mode only) ---
    discriminator = None
    if args.training_mode == "ala":
        disc_input_dim = args.disc_input_dim_override if args.disc_input_dim_override else etp_model.wubu_core.output_tangent_dim
        discriminator = LatentDiscriminatorMLP(
            input_dim=disc_input_dim, hidden_dims=args.disc_hidden_dims,
            activation_fn=args.disc_activation_fn, use_spectral_norm=args.disc_use_spectral_norm
        )

    # --- Datasets and DataLoaders ---
    train_loader, val_loader = None, None
    if args.training_mode == "ala":
        train_dataset = PooledEmbeddingDataset(args.embeddings_file_A, args.embeddings_file_B)
        # val_dataset = PooledEmbeddingDataset(args.val_embeddings_file_A, args.val_embeddings_file_B) # If you add val files
    elif args.training_mode == "dissection_mimicry":
        train_dataset = DissectedDeepSeekNPZDataset(args.dissected_data_file_train, args.dm_target_teacher_layers,
                                                    seq_len_for_pooling=args.ds_r1_embedding_dim if args.dm_pool_teacher_states and not args.dm_wubu_core_processes_sequences else None)
        if args.dissected_data_file_val:
            val_dataset = DissectedDeepSeekNPZDataset(args.dissected_data_file_val, args.dm_target_teacher_layers,
                                                      seq_len_for_pooling=args.ds_r1_embedding_dim if args.dm_pool_teacher_states and not args.dm_wubu_core_processes_sequences else None)
        else: val_dataset = None
    else:
        raise ValueError(f"Invalid training_mode: {args.training_mode}")

    if len(train_dataset) == 0:
        script_logger.error("Training dataset is empty. Exiting."); sys.exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(args.device.type=='cuda'), drop_last=True)
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(args.device.type=='cuda'))
    else: val_loader = None


    # --- Trainer ---
    trainer = ETPUniversalTrainer(args, etp_model, train_loader, val_loader, discriminator)
    trainer.train(resume_from_checkpoint=args.load_checkpoint)
    script_logger.info(f"Training in mode '{args.training_mode}' finished.")


if __name__ == '__main__':
    script_logger.info("Starting ETP Universal Combined Script.")
    main_universal()
    script_logger.info("ETP Universal Combined Script execution completed.")

