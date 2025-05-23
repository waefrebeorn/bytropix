
# WuBuGAADHybridGen_v0.2.py
# VAE-GAN Hybrid Model with GAAD-WuBu Regional Hyperbolic Latent Space
# Incorporates Optical Flow for Motion Encoding Branch.
# Operating directly on GAAD-defined regions with WuBu nesting.
# LAST UPDATE: Refactored from Diffusion to VAE-GAN (v0.1 internal rev from Diff v0.10.1)
# THIS VERSION: Incorporating DFT for regional appearance features.

# =====================================================================
# Python Imports and Setup
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
from torch.nn.utils import spectral_norm
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from tqdm import tqdm

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
try:
    import torchvision.io as video_io
    VIDEO_IO_AVAILABLE = True
except ImportError:
    video_io = None
    VIDEO_IO_AVAILABLE = False
    print("Warn: torchvision.io unavailable.")
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None
    IMAGEIO_AVAILABLE = False
    print("Warn: imageio unavailable.")

from torchvision.ops import roi_align
from torchvision.utils import save_image

# --- Optical Flow Import ---
try:
    import torchvision.models.optical_flow as tv_flow
    OPTICAL_FLOW_AVAILABLE = True
    FLOW_MODELS = {
        'raft_large': (tv_flow.Raft_Large_Weights.DEFAULT, tv_flow.raft_large),
        'raft_small': (tv_flow.Raft_Small_Weights.DEFAULT, tv_flow.raft_small),
    }
except ImportError:
    tv_flow = None
    OPTICAL_FLOW_AVAILABLE = False
    FLOW_MODELS = {}
    print("Warning: torchvision.models.optical_flow not available. Motion branch will be disabled if selected.")


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
logger = logging.getLogger("WuBuGAADHybridGenV01DFT") # Renamed logger
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
DEFAULT_CONFIG_QLEARN_HYBRID = {
    "q_learning_rate": 0.01, "discount_factor": 0.90, "epsilon_start": 0.5,
    "epsilon_min": 0.05, "epsilon_decay": 0.9995,
    "lr_scale_options": [0.8, 0.9, 1.0, 1.1, 1.2],
    "momentum_scale_options": [0.95, 0.98, 1.0, 1.01, 1.02],
    "max_q_table_size": 20000, "state_history_len": 5,
    "reward_clipping": (-2.0, 2.0), "q_value_clipping": (-30.0, 30.0)
}

# =====================================================================
# DFT Utilities (New)
# =====================================================================
class DFTUtils:
    @staticmethod
    def compute_2d_dft_features(
        patches: torch.Tensor,
        norm_scale: float = 10.0,
        out_feature_dim_first: bool = False
    ) -> torch.Tensor:
        """
        Computes 2D DFT features for a batch of image patches.
        Args:
            patches (torch.Tensor): Input patches, shape (B_flat, C, H_patch, W_patch).
                                    B_flat could be Batch * NumRegions * NumFrames.
            norm_scale (float): Value to scale real/imag components for normalization.
            out_feature_dim_first (bool): If true, output is (B_flat, FeatureDim).
                                          If false, output is (B_flat, C, 2, H_patch, W_patch//2+1) - NOT SUPPORTED YET for this path
        Returns:
            torch.Tensor: DFT features, shape (B_flat, C * 2 * H_patch * (W_patch//2+1)).
        """
        B_flat, C, H_patch, W_patch = patches.shape
        if not torch.is_floating_point(patches):
            patches = patches.float() / 255.0 # Assuming uint8 input if not float

        dft_coeffs_complex = torch.fft.rfft2(patches, dim=(-2, -1), norm="ortho") # (B_flat, C, H_patch, W_patch//2+1)

        dft_real = dft_coeffs_complex.real
        dft_imag = dft_coeffs_complex.imag

        # Normalize (simple scaling)
        dft_real_norm = dft_real / norm_scale
        dft_imag_norm = dft_imag / norm_scale

        # Concatenate real and imaginary parts and flatten
        # (B_flat, C, H_patch, W_coeffs_one_sided), (B_flat, C, H_patch, W_coeffs_one_sided)
        # -> (B_flat, C, 2, H_patch, W_coeffs_one_sided) -> (B_flat, C * 2 * H_patch * W_coeffs_one_sided)
        
        # Stack real and imag along a new dimension, then flatten
        # Target: (B_flat, D_dft_features)
        # D_dft_features = C * 2 * H_patch * (W_patch//2+1)
        stacked_coeffs = torch.stack([dft_real_norm, dft_imag_norm], dim=2) # (B_flat, C, 2, H_patch, W_patch//2+1)
        
        if out_feature_dim_first:
             flattened_dft_features = stacked_coeffs.reshape(B_flat, -1) # (B_flat, D_dft_features)
        else: # For direct use without PatchEmbed, or if PatchEmbed expects this shape
            # This path is less likely for the current encoder setup which uses PatchEmbed on flattened features.
            # Keeping the option for future flexibility or direct use in Generator's regional head if it outputs this shape.
            # For this VAE-GAN refactor, encoder will likely use out_feature_dim_first=True
            # and generator will output features that reconstruct to this `stacked_coeffs` shape.
            # However, the method signature returns the flattened version.
            logging.getLogger("WuBuGAADHybridGenV01DFT.DFTUtils").warning("compute_2d_dft_features: out_feature_dim_first=False requested but typically returns flattened features. Returning flattened.")
            flattened_dft_features = stacked_coeffs.reshape(B_flat, -1)

        return flattened_dft_features

    @staticmethod
    def reconstruct_patches_from_2d_dft(
        dft_features_norm_flat: torch.Tensor, # (B_flat, NumRegions, FeatureDim) or (B_flat_total_regions, FeatureDim)
        norm_scale: float,
        num_channels: int,
        target_patch_h: int,
        target_patch_w: int, # Original patch width before rfft2
    ) -> torch.Tensor:
        """
        Reconstructs image patches from normalized flat DFT features.
        Args:
            dft_features_norm_flat (torch.Tensor): Normalized flat DFT features.
                Shape (B_total_regions, C * 2 * H_patch * (W_patch//2+1)).
            norm_scale (float): Scale used for normalization.
            num_channels (int): Number of image channels (e.g., 3 for RGB).
            target_patch_h (int): Height of the original patches.
            target_patch_w (int): Width of the original patches.
        Returns:
            torch.Tensor: Reconstructed pixel patches, shape (B_total_regions, C, H_patch, W_patch).
                          Values will be in [0, 1] approx.
        """
        B_total_regions = dft_features_norm_flat.shape[0]
        W_coeffs_one_sided = target_patch_w // 2 + 1
        
        expected_feat_dim = num_channels * 2 * target_patch_h * W_coeffs_one_sided
        if dft_features_norm_flat.shape[1] != expected_feat_dim:
            raise ValueError(f"DFT feature dim mismatch. Expected {expected_feat_dim}, got {dft_features_norm_flat.shape[1]}")

        # Reshape flat features back to (B_total_regions, C, 2, H_patch, W_coeffs_one_sided)
        stacked_coeffs_norm = dft_features_norm_flat.view(
            B_total_regions, num_channels, 2, target_patch_h, W_coeffs_one_sided
        )

        dft_real_norm = stacked_coeffs_norm[:, :, 0, :, :]
        dft_imag_norm = stacked_coeffs_norm[:, :, 1, :, :]

        # Unnormalize
        dft_real = dft_real_norm * norm_scale
        dft_imag = dft_imag_norm * norm_scale

        # Combine to complex
        dft_coeffs_complex = torch.complex(dft_real, dft_imag) # (B_total_regions, C, H_patch, W_coeffs_one_sided)

        # Inverse DFT
        # irfft2 needs the original W, not W_coeffs_one_sided for the 's' parameter
        reconstructed_patches = torch.fft.irfft2(dft_coeffs_complex, s=(target_patch_h, target_patch_w), dim=(-2, -1), norm="ortho")
        # Output is (B_total_regions, C, H_patch, W_patch)
        
        # Values should be roughly in a range that can be clamped to [0,1] if input was normalized [0,1]
        # For VAE-GAN, output is typically [-1, 1] from Tanh, so recon loss on DFT might be on features
        # derived from [-1, 1] pixels. Here, let's assume the IDFT gives values that can be scaled to [0,1]
        # The trainer will handle if [-1,1] is needed.
        return reconstructed_patches # Values might not be in [0,1] yet.

# =====================================================================
# Image Assembly Utility (New)
# =====================================================================
class ImageAssemblyUtils:
    @staticmethod
    def assemble_frames_from_patches(
        patches_batch: torch.Tensor, # (B, N_frames, N_regions, C, H_patch, W_patch)
        bboxes_batch: torch.Tensor,  # (B, N_frames, N_regions, 4) [x1, y1, x2, y2]
        target_image_size: Tuple[int, int], # (H_img, W_img)
        output_range: Tuple[float, float] = (0.0, 1.0) # e.g. (0,1) or (-1,1)
    ) -> torch.Tensor:
        """
        Assembles full frames from batches of regional patches.
        Handles resizing patches to bbox sizes and averages overlapping regions.
        """
        B, N_f, N_r, C, H_patch, W_patch = patches_batch.shape
        H_img, W_img = target_image_size
        device = patches_batch.device
        dtype = patches_batch.dtype

        all_assembled_frames = torch.zeros(B, N_f, C, H_img, W_img, device=device, dtype=dtype)

        for b_idx in range(B):
            for f_idx in range(N_f):
                canvas = torch.zeros(C, H_img, W_img, device=device, dtype=dtype)
                count_map = torch.zeros(1, H_img, W_img, device=device, dtype=dtype) # Use float for count_map sums

                for r_idx in range(N_r):
                    patch = patches_batch[b_idx, f_idx, r_idx] # (C, H_patch, W_patch)
                    x1, y1, x2, y2 = bboxes_batch[b_idx, f_idx, r_idx].round().int().tolist()

                    # Ensure valid bbox coordinates
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(W_img, x2), min(H_img, y2)

                    if x1_c >= x2_c or y1_c >= y2_c:
                        continue # Skip empty or invalid region

                    target_h_bbox = y2_c - y1_c
                    target_w_bbox = x2_c - x1_c

                    if target_h_bbox <=0 or target_w_bbox <=0: continue


                    # Resize patch to fit the bounding box dimensions
                    if H_patch != target_h_bbox or W_patch != target_w_bbox:
                        # TF.resize expects (C, H, W) or (B, C, H, W)
                        resized_patch = TF.resize(patch, [target_h_bbox, target_w_bbox], antialias=True)
                    else:
                        resized_patch = patch
                    
                    # Add to canvas and update count map
                    canvas[:, y1_c:y2_c, x1_c:x2_c] += resized_patch
                    count_map[:, y1_c:y2_c, x1_c:x2_c] += 1.0
                
                # Average overlapping regions
                # Add EPS to count_map before division to avoid div by zero for unpainted areas
                # Unpainted areas will remain 0 in canvas, and 0/EPS will also be 0.
                assembled_frame = torch.where(count_map > 0, canvas / (count_map + EPS), canvas)
                all_assembled_frames[b_idx, f_idx] = assembled_frame
        
        # Clamp to output range if specified
        if output_range is not None:
            all_assembled_frames = torch.clamp(all_assembled_frames, min=output_range[0], max=output_range[1])
            
        return all_assembled_frames




# =====================================================================
# Geometric, Optimizer, WuBu Core Components (Largely Unchanged)
# =====================================================================
class HyperbolicUtils: # ... (No changes from original, keep as is)
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
    def exponential_map(v: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor: return HyperbolicUtils.scale_aware_exponential_map(v, c_scalar, scale_scalar=1.0, eps=eps)
    @staticmethod
    def logarithmic_map(y: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor: return HyperbolicUtils.scale_aware_logarithmic_map(y, c_scalar, scale_scalar=1.0, eps=eps)




class Manifold: # ... (No changes from original, keep as is)
    def __init__(self, c_scalar=0.0): self.c = float(c_scalar)
    def proju(self, p: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def logmap0(self, p: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5): raise NotImplementedError
    def expmap(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor: return self.proju(p + dp) if self.c > 0 else p + dp
    @property
    def name(self) -> str: return self.__class__.__name__

class PoincareBall(Manifold): # ... (No changes from original, keep as is)
    def __init__(self, c_scalar: float = 1.0):
        super().__init__(c_scalar)
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0: self.c = 0.0; self.k = 0.; self.sqrt_c = 0.; self.radius = float('inf')
        else: self.c = c_scalar; self.k = -self.c; self.sqrt_c = math.sqrt(self.c); self.radius = 1. / self.sqrt_c
        self.max_norm = self.radius * (1. - EPS * 10) if self.c > 0 and self.radius != float('inf') else float('inf')
        self._name = f'PoincareBall(c={self.c:.3g})'; self.logger = logging.getLogger(f"WuBuGAADHybridGenV01DFT.Poincare")


    @property
    def name(self) -> str: return self._name
    def proju(self, x: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.poincare_clip(x, self.c, radius=1., eps=EPS * 10)
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.exponential_map(dp, self.c, eps=EPS)
    def logmap0(self, p: torch.Tensor) -> torch.Tensor: return HyperbolicUtils.logarithmic_map(p, self.c, eps=EPS)
    def expmap0_scaled(self, dp: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_exponential_map(dp, self.c, scale_scalar=scale_scalar, eps=EPS)
    def logmap0_scaled(self, p: torch.Tensor, scale_scalar: float) -> torch.Tensor: return HyperbolicUtils.scale_aware_logarithmic_map(p, self.c, scale_scalar=scale_scalar, eps=EPS)
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        if self.c <= 0: return dp
        p_projected = self.proju(p); p_norm_sq = torch.sum(p_projected.pow(2), dim=-1, keepdim=True)
        max_sq_norm_val = self.max_norm**2 if self.radius != float('inf') else float('inf')
        p_norm_sq_clamped = torch.clamp(p_norm_sq, min=0.0, max=max_sq_norm_val)
        term_inside_paren = 1. - self.c * p_norm_sq_clamped; lambda_p_factor = term_inside_paren / 2.0
        riemannian_scaling_factor = lambda_p_factor.pow(2); final_factor = torch.clamp(riemannian_scaling_factor, min=EPS); r_grad = final_factor * dp
        if not torch.isfinite(r_grad).all(): self.logger.warning(f"Non-finite Riemannian gradient in egrad2rgrad for P:{p.shape}, c={self.c}. Factor:{final_factor.mean().item() if final_factor.numel()>0 else 'N/A'}. Fallback to Euclidean grad."); return dp
        return r_grad
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad():
            w.data.uniform_(-irange, irange)
            if self.c > 0 : w.data = self.expmap0(w.data); w.data = self.proju(w.data)

def init_weights_general(m): # ... (No changes from original, keep as is)
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', False)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.zeros_(m.bias) if m.bias is not None else None

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor: return F.softplus(param_unconstrained) + min_val
class BoundaryManifoldHyperbolic(nn.Module): # ... (No changes from original, keep as is)
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold_c: float):
        super().__init__(); self.level_idx = level_idx; self.num_points = num_points; self.point_dim = point_dim; self.current_manifold_c = initial_manifold_c
        if num_points > 0 and point_dim > 0: self.hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim)); PoincareBall(initial_manifold_c).init_weights(self.hyperbolic_points_params, irange=1e-3); setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(initial_manifold_c))
        else: self.register_parameter('hyperbolic_points_params', None)
    def set_current_manifold_c(self, c_scalar: float): self.current_manifold_c = c_scalar; setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(c_scalar)) if self.hyperbolic_points_params is not None else None
    def get_points(self) -> Optional[torch.Tensor]: return PoincareBall(self.current_manifold_c).proju(self.hyperbolic_points_params) if self.hyperbolic_points_params is not None else None

def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor: # ... (No changes)
    axis = F.normalize(axis, p=2, dim=-1); angle_half = angle_rad / 2.0; q_w = torch.cos(angle_half); q_xyz = axis * torch.sin(angle_half); return torch.cat([q_w, q_xyz], dim=-1)
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor: # ... (No changes)
    w1, x1, y1, z1 = q1.unbind(-1); w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2; return torch.stack([w, x, y, z], dim=-1)
def quaternion_apply_to_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: # ... (No changes)
    v_quat = F.pad(v, (1, 0), value=0); q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1); rotated_v_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj); return rotated_v_quat[..., 1:]

class HyperbolicInterLevelTransform(nn.Module): # ... (No changes from original, keep as is)
    def __init__(self, in_dim: int, out_dim: int, initial_c_in: float, initial_c_out: float, transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1, use_rotation: bool = False, phi_influence_rotation_init: bool = False, level_idx_for_phi: int = 0):
        super().__init__(); self.in_dim, self.out_dim, self.transform_type = in_dim, out_dim, transform_type.lower(); self.use_rotation = use_rotation; self.rotation_module = None; self.phi_influence_rotation_init = phi_influence_rotation_init; current_logger=logging.getLogger("WuBuGAADHybridGenV01DFT.HILT")
        if self.use_rotation and self.in_dim > 0:
            if self.in_dim == 4 and self.phi_influence_rotation_init: self.rot_axis_param = nn.Parameter(torch.randn(3)); self.rot_angle_unconstrained = nn.Parameter(torch.tensor(0.0)); self.phi_angle_scale = PHI**(level_idx_for_phi % 5 - 2) * (math.pi / 4); current_logger.info(f"L{level_idx_for_phi} (4D): Quat rot. Scale: {self.phi_angle_scale:.3f}")
            elif self.in_dim == 2 and self.phi_influence_rotation_init: self.rot_angle_unconstrained_2d = nn.Parameter(torch.tensor(0.0)); self.phi_angle_scale_2d = PHI**(level_idx_for_phi % 3) * (math.pi / 3); current_logger.info(f"L{level_idx_for_phi} (2D): SO(2) rot. Scale: {self.phi_angle_scale_2d:.3f}")
            else: self.rotation_module = nn.Linear(self.in_dim, self.in_dim, bias=False); nn.init.eye_(self.rotation_module.weight) if self.in_dim > 0 else None
        mlp_hidden_dim = hidden_dim if hidden_dim and hidden_dim > 0 else max(16, (in_dim + out_dim) // 2)
        if self.transform_type == 'mlp' and all(d > 0 for d in [in_dim, out_dim, mlp_hidden_dim]): self.non_rotational_map = nn.Sequential(nn.Linear(in_dim, mlp_hidden_dim), nn.LayerNorm(mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim, out_dim))
        elif self.transform_type == 'linear' and in_dim > 0 and out_dim > 0: self.non_rotational_map = nn.Linear(in_dim, out_dim)
        else: self.non_rotational_map = nn.Identity(); current_logger.info(f"L{level_idx_for_phi}: Using Identity transform as in_dim={in_dim} or out_dim={out_dim} or hidden_dim={mlp_hidden_dim} is non-positive.")
        self.apply(init_weights_general)
    def _apply_rotation(self, x_tan: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_tan is None or not self.use_rotation: return x_tan; B_maybe = x_tan.shape[0] if x_tan.dim() > 1 else 1
        if self.in_dim == 4 and self.phi_influence_rotation_init and hasattr(self, 'rot_axis_param'): angle = F.softplus(self.rot_angle_unconstrained) * self.phi_angle_scale; current_axis = self.rot_axis_param.to(x_tan.device).unsqueeze(0).expand(B_maybe, -1); angle_b = angle.unsqueeze(0).expand(B_maybe, 1); q_rot = quaternion_from_axis_angle(angle_b, current_axis); return quaternion_apply_to_vector(q_rot, x_tan)
        elif self.in_dim == 2 and self.phi_influence_rotation_init and hasattr(self, 'rot_angle_unconstrained_2d'): angle_2d = F.softplus(self.rot_angle_unconstrained_2d) * self.phi_angle_scale_2d; cos_a = torch.cos(angle_2d); sin_a = torch.sin(angle_2d); x_comp = x_tan[..., 0]; y_comp = x_tan[..., 1]; x_rot = x_comp * cos_a - y_comp * sin_a; y_rot = x_comp * sin_a + y_comp * cos_a; return torch.stack([x_rot, y_rot], dim=-1)
        return self.rotation_module(x_tan) if self.rotation_module else x_tan
    def forward(self, point_in: torch.Tensor, boundaries_in: Optional[torch.Tensor], descriptor_in: Optional[torch.Tensor], current_c_in: float, current_c_out: float, current_s_in: Optional[float]=None, current_s_out: Optional[float]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        m_in, m_out = PoincareBall(current_c_in), PoincareBall(current_c_out)
        tan_main = m_in.logmap0(point_in); tan_bound = m_in.logmap0(boundaries_in) if boundaries_in is not None else None; tan_desc = m_in.logmap0(descriptor_in) if descriptor_in is not None else None
        tan_main_rot = self._apply_rotation(tan_main); tan_bound_rot = self._apply_rotation(tan_bound); tan_desc_rot = self._apply_rotation(tan_desc)
        def apply_map_and_clamp(tan_vec): return torch.clamp(self.non_rotational_map(tan_vec), -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL) if tan_vec is not None else None
        tan_main_out_clamped = apply_map_and_clamp(tan_main_rot); tan_bound_out_clamped = apply_map_and_clamp(tan_bound_rot); tan_desc_out_clamped = apply_map_and_clamp(tan_desc_rot)
        default_out_shape = (point_in.shape[0], self.out_dim) if point_in.dim() > 1 else (self.out_dim,)
        expmap_main_out = m_out.expmap0(tan_main_out_clamped) if tan_main_out_clamped is not None else m_out.expmap0(torch.zeros(default_out_shape, device=point_in.device, dtype=point_in.dtype))
        expmap_bound_out = m_out.expmap0(tan_bound_out_clamped) if tan_bound_out_clamped is not None else None
        expmap_desc_out = m_out.expmap0(tan_desc_out_clamped) if tan_desc_out_clamped is not None else None
        return (expmap_main_out, expmap_bound_out, expmap_desc_out)

class HyperbolicWuBuNestingLevel(nn.Module): # ... (No changes from original, keep as is)
    def __init__(self, level_idx: int, dim: int, config: Dict, initial_curvature_val_base: float):
        super().__init__()
        self.level_idx, self.dim, self.config = level_idx, dim, config; self.logger = logging.getLogger(f"WuBuGAADHybridGenV01DFT.Level{self.level_idx}")
        current_logger = self.logger
        self.phi_influence_curvature = config.get("phi_influence_curvature", False)
        self.initial_curvature_val = initial_curvature_val_base * (PHI**(level_idx % 4 - 1.5) if self.phi_influence_curvature else 1.0)
        current_logger.info(f"InitialC={self.initial_curvature_val:.2f}"+(f" (PhiBase {initial_curvature_val_base:.2f})" if self.phi_influence_curvature else ""))
        self.use_ld = config.get("use_level_descriptors", True); self.use_spread = config.get("use_level_spread", True)
        self.dropout_rate = config.get("dropout", 0.1); self.ld_init_scale = config.get("level_descriptor_init_scale", 1e-5)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean")
        self.min_curvature = max(EPS, config.get("curvature_min_value", EPS)); self.min_scale = max(EPS, config.get("scale_min_value", EPS)); self.min_spread = max(EPS, config.get("spread_min_value", EPS))
        def _init_unconstrained_param_sigmoid_scaled(target_val, min_val_range, max_val_range):
            if not (min_val_range < max_val_range): current_logger.warning(f"SigmoidInit: Invalid range [{min_val_range}, {max_val_range}]. Init unconstrained to 0."); return torch.tensor(0.0)
            clamped_target_val = torch.clamp(torch.as_tensor(target_val, dtype=torch.float), min_val_range + EPS, max_val_range - EPS).item(); initial_sigmoid_target = (clamped_target_val - min_val_range) / (max_val_range - min_val_range); initial_sigmoid_target_clamped = max(EPS, min(initial_sigmoid_target, 1.0 - EPS)); unconstrained_val = math.log(initial_sigmoid_target_clamped / (1.0 - initial_sigmoid_target_clamped)); return torch.tensor(unconstrained_val)
        def _init_unconstrained_param_softplus(target_val, min_val): val_for_softplus = max(float(target_val), min_val + EPS) - min_val; return torch.tensor(math.log(math.expm1(val_for_softplus)) if val_for_softplus > 1e-6 else math.log(val_for_softplus + EPS))
        param_init_args = {'learn_c': ("learnable_curvature", self.initial_curvature_val, self.min_curvature, 'log_curvature_unconstrained', 'softplus'), 'learn_s': ("learnable_scales", "initial_scales", (MIN_WUBU_LEVEL_SCALE, MAX_WUBU_LEVEL_SCALE), 'log_scale_unconstrained', 'sigmoid_scaled'), 'learn_spread': ("learnable_spread", "initial_spread_values", self.min_spread, 'log_spread_unconstrained', 'softplus')}
        for key, (learn_flag_name, init_val_name_or_direct, min_or_range_val_local, param_name, init_type) in param_init_args.items():
            if key == 'learn_spread' and not self.use_spread: self.register_parameter(param_name, None); continue
            learn_flag = config.get(learn_flag_name, True); default_list_val = [1.0] if key == 'learn_s' else [0.1] if key == 'learn_spread' else [self.initial_curvature_val]
            if isinstance(init_val_name_or_direct, str): init_list = config.get(init_val_name_or_direct, default_list_val); init_val = init_list[level_idx] if level_idx < len(init_list) else (init_list[-1] if init_list else default_list_val[0])
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
        self.comb_in_dim = self.dim + (self.dim if self.relative_vector_aggregation not in ['none', None] else 0) + (self.dim if self.use_ld else 0)
        comb_h_dims_cfg = config.get("tangent_input_combination_dims", [max(16, self.comb_in_dim // 2)]) if self.comb_in_dim > 0 else []; comb_h_dims = comb_h_dims_cfg if isinstance(comb_h_dims_cfg, list) else [comb_h_dims_cfg]
        layers = []; in_d = self.comb_in_dim
        if self.dim > 0 and self.comb_in_dim > 0:
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
        ref_param = next(iter(self.parameters()), None)
        if ref_param is None and isinstance(self.tangent_combiner, nn.Sequential) and list(self.tangent_combiner.parameters()): ref_param = next(iter(self.tangent_combiner.parameters()), None)
        ref_device = ref_param.device if ref_param is not None else torch.device('cpu'); ref_dtype = ref_param.dtype if ref_param is not None else torch.float; return torch.tensor(self.min_spread, device=ref_device, dtype=ref_dtype)
    def forward(self, point_in_hyperbolic: torch.Tensor, relative_vectors_tangent_in: Optional[torch.Tensor], descriptor_point_in_hyperbolic: Optional[torch.Tensor], sigma_in_scalar_tensor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if point_in_hyperbolic.dim() != 2: raise ValueError(f"WuBuLevel forward expects 2D input (B', D), got {point_in_hyperbolic.dim()}D shape {point_in_hyperbolic.shape}")
        B_prime, D_in = point_in_hyperbolic.shape; dev = point_in_hyperbolic.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else point_in_hyperbolic.dtype
        if self.dim == 0: dummy_out_shape = (B_prime, 0); dummy_dtype_dev = {'device': dev, 'dtype': dtype_to_use}; current_spread_tensor = self.get_current_spread_scalar_tensor().to(dtype_to_use); return (torch.zeros(dummy_out_shape, **dummy_dtype_dev), torch.zeros(dummy_out_shape, **dummy_dtype_dev), None, None, current_spread_tensor)
        current_c_val = self.get_current_curvature_scalar(); current_s_val = self.get_current_scale_scalar(); current_sigma_out_tensor = self.get_current_spread_scalar_tensor(); current_manifold_obj = PoincareBall(c_scalar=current_c_val)
        if self.level_descriptor_param is not None and hasattr(self.level_descriptor_param, 'manifold'): setattr(self.level_descriptor_param, 'manifold', PoincareBall(c_scalar=current_c_val))
        if self.boundary_manifold_module is not None: self.boundary_manifold_module.set_current_manifold_c(current_c_val)
        point_in_proj = current_manifold_obj.proju(point_in_hyperbolic.to(dtype_to_use)); tan_main_component = current_manifold_obj.logmap0(point_in_proj); tan_rel_component = torch.zeros_like(tan_main_component); ld_point_self_hyperbolic = None
        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None]:
            if relative_vectors_tangent_in.shape[0] != B_prime: raise ValueError(f"RelVec shape mismatch: {relative_vectors_tangent_in.shape[0]} != B' {B_prime}")
            tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)
        if self.use_ld and self.level_descriptor_param is not None: ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use))
        tan_desc_prev_level_component = torch.zeros_like(tan_main_component)
        if descriptor_point_in_hyperbolic is not None and self.use_ld:
            if descriptor_point_in_hyperbolic.shape[0] != B_prime: raise ValueError(f"DescIn shape mismatch: {descriptor_point_in_hyperbolic.shape[0]} != B' {B_prime}")
            desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use)); tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)
        inputs_for_combiner = [tan_main_component]
        if self.relative_vector_aggregation not in ['none', None]: inputs_for_combiner.append(tan_rel_component)
        if self.use_ld: inputs_for_combiner.append(tan_desc_prev_level_component)
        if not inputs_for_combiner: combined_tangent_features = torch.zeros(B_prime, self.dim, device=dev, dtype=dtype_to_use)
        elif len(inputs_for_combiner) > 1: combined_tangent_features = torch.cat(inputs_for_combiner, dim=-1)
        else: combined_tangent_features = inputs_for_combiner[0]
        if self.comb_in_dim > 0:
            if combined_tangent_features.shape[-1] < self.comb_in_dim: padding_size = self.comb_in_dim - combined_tangent_features.shape[-1]; combined_tangent_features = F.pad(combined_tangent_features, (0, padding_size)) if padding_size > 0 else combined_tangent_features
            elif combined_tangent_features.shape[-1] > self.comb_in_dim: self.logger.warning(f"Tangent Combiner input dim {combined_tangent_features.shape[-1]} > expected {self.comb_in_dim}. Truncating."); combined_tangent_features = combined_tangent_features[..., :self.comb_in_dim]
        elif combined_tangent_features.shape[-1] > 0 and self.comb_in_dim == 0: B_prime_local = combined_tangent_features.shape[0]; self.logger.warning(f"Tangent Combiner expects 0-dim input (self.comb_in_dim=0), but got {combined_tangent_features.shape[-1]} features. Forcing to (Batch={B_prime_local}, 0)."); combined_tangent_features = torch.empty(B_prime_local, 0, device=combined_tangent_features.device, dtype=combined_tangent_features.dtype)
        v_combined_tangent_processed = self.tangent_combiner(combined_tangent_features); v_final_for_expmap_unclamped = v_combined_tangent_processed * current_s_val
        if self.use_flow and self.tangent_flow_module is not None: flow_effect = self.tangent_flow_module(v_combined_tangent_processed) * self.flow_scale_val; v_final_for_expmap_unclamped = v_final_for_expmap_unclamped + flow_effect
        scaled_output_tangent_for_expmap = torch.clamp(v_final_for_expmap_unclamped, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL); point_this_level_out_hyperbolic = current_manifold_obj.expmap0(scaled_output_tangent_for_expmap); tangent_out_for_aggregation = v_combined_tangent_processed.to(dtype_to_use)
        boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype=dtype_to_use, device=dev) if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None else None
        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None: descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B_prime, -1).to(dtype=dtype_to_use) if ld_point_self_hyperbolic.dim() == 1 else ld_point_self_hyperbolic.to(dtype=dtype_to_use)
        output_dtype = point_in_hyperbolic.dtype
        return (point_this_level_out_hyperbolic.to(dtype=output_dtype), tangent_out_for_aggregation.to(dtype=output_dtype), descriptor_point_out_for_transform_hyperbolic.to(dtype=output_dtype) if descriptor_point_out_for_transform_hyperbolic is not None else None, boundary_points_this_level_hyperbolic.to(dtype=output_dtype) if boundary_points_this_level_hyperbolic is not None else None, current_sigma_out_tensor.to(dtype=output_dtype))

class FullyHyperbolicWuBuNestingModel(nn.Module): # ... (No changes from original, keep as is)
    def __init__(self, input_tangent_dim: int, output_tangent_dim: int, config: Dict):
        super().__init__(); current_logger=logging.getLogger("WuBuGAADHybridGenV01DFT.WuBuModel"); self.input_tangent_dim, self.output_tangent_dim, self.config = input_tangent_dim, output_tangent_dim, config; self.num_levels = config.get("num_levels", 3); assert self.num_levels >= 0; self.hyperbolic_dims_list = config.get("hyperbolic_dims", []); self.initial_curvatures_list = config.get("initial_curvatures", []); self.dropout_val = config.get("dropout", 0.1); self.relative_vector_aggregation_mode = config.get("relative_vector_aggregation", "mean"); self.aggregation_method_mode = config.get("aggregation_method", "concat_tangent"); assert self.aggregation_method_mode == "concat_tangent"; self.use_rotation_in_transform_flag = config.get("use_rotation_in_transform", False); self.phi_influence_rotation_init = config.get("phi_influence_rotation_init", False)
        first_level_dim = self.hyperbolic_dims_list[0] if self.num_levels > 0 and self.hyperbolic_dims_list else 0
        self.input_tangent_projection = nn.Linear(input_tangent_dim, first_level_dim) if input_tangent_dim > 0 and first_level_dim > 0 and input_tangent_dim != first_level_dim else nn.Identity()
        self.input_tangent_layernorm = nn.LayerNorm(first_level_dim) if first_level_dim > 0 else nn.Identity()
        self.levels_modulelist = nn.ModuleList(); self.transforms_modulelist = nn.ModuleList()
        if self.num_levels > 0:
            for i in range(self.num_levels):
                if i < len(self.hyperbolic_dims_list) and i < len(self.initial_curvatures_list): self.levels_modulelist.append(HyperbolicWuBuNestingLevel(i, self.hyperbolic_dims_list[i], self.config, self.initial_curvatures_list[i]))
                else: current_logger.error(f"Level {i} skipped: Config lists too short (dims:{len(self.hyperbolic_dims_list)}, curves:{len(self.initial_curvatures_list)})"); break
            num_transforms_needed = max(0, len(self.levels_modulelist) - 1)
            if num_transforms_needed > 0:
                transform_types_list = config.get("transform_types", ["linear"] * num_transforms_needed); transform_hidden_dims_list = config.get("transform_hidden_dims", [None] * num_transforms_needed)
                for i in range(num_transforms_needed):
                    if i + 1 < len(self.levels_modulelist) and i + 1 < len(self.hyperbolic_dims_list) and i + 1 < len(self.initial_curvatures_list): self.transforms_modulelist.append(HyperbolicInterLevelTransform(self.hyperbolic_dims_list[i], self.hyperbolic_dims_list[i+1], self.initial_curvatures_list[i], self.initial_curvatures_list[i+1], transform_types_list[i] if i < len(transform_types_list) else "linear", transform_hidden_dims_list[i] if i < len(transform_hidden_dims_list) else None, self.dropout_val, self.use_rotation_in_transform_flag, self.phi_influence_rotation_init, level_idx_for_phi=i))
                    else: current_logger.warning(f"Skipping transform {i} to {i+1} due to insufficient config/levels for next level.")
        actual_output_dims_from_levels = [d for d_idx, d in enumerate(self.hyperbolic_dims_list[:len(self.levels_modulelist)]) if d > 0]; aggregated_tangent_dim_val = sum(actual_output_dims_from_levels) if actual_output_dims_from_levels else input_tangent_dim
        self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, output_tangent_dim) if aggregated_tangent_dim_val > 0 and output_tangent_dim > 0 and aggregated_tangent_dim_val != output_tangent_dim else nn.Identity()
        self.apply(init_weights_general); param_count = sum(p.numel() for p in self.parameters() if p.requires_grad); current_logger.info(f"Levels: {len(self.levels_modulelist)}. Params: {param_count:,}. InDim {input_tangent_dim}, AggDim {aggregated_tangent_dim_val}, OutDim {output_tangent_dim}")
    def forward(self, x_initial_tangent_in: torch.Tensor) -> torch.Tensor:
        input_dim = x_initial_tangent_in.dim(); B_orig, S_orig, D_orig = -1, -1, -1; B_prime = -1
        if input_dim == 3: B_orig, S_orig, D_orig = x_initial_tangent_in.shape; x_proc = x_initial_tangent_in.reshape(B_orig * S_orig, D_orig); B_prime_for_levels = B_orig * S_orig
        elif input_dim == 2: B_prime, D_orig = x_initial_tangent_in.shape; x_proc = x_initial_tangent_in; B_prime_for_levels = B_prime
        else: raise ValueError(f"WuBuModel expects 2D/3D input, got {input_dim}D")
        if D_orig != self.input_tangent_dim: raise ValueError(f"Input feature dim {D_orig} != model input_tangent_dim {self.input_tangent_dim}")
        if self.num_levels == 0 or not self.levels_modulelist: return self.output_tangent_projection(x_proc).reshape(B_orig, S_orig, -1) if input_dim==3 else self.output_tangent_projection(x_proc)
        dev = x_proc.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else x_proc.dtype; x_proc = x_proc.to(dtype_to_use)
        current_tangent_projected = self.input_tangent_projection(x_proc); current_tangent_for_level0 = self.input_tangent_layernorm(current_tangent_projected)
        level0_module = self.levels_modulelist[0]; c0_val = level0_module.get_current_curvature_scalar(); m0_obj = PoincareBall(c_scalar=c0_val)
        current_point_repr_hyperbolic = m0_obj.expmap0(current_tangent_for_level0) if self.hyperbolic_dims_list[0] > 0 else torch.empty(B_prime_for_levels, 0, device=dev, dtype=dtype_to_use)
        level_tangent_outputs_for_aggregation = []; aggregated_relative_vectors_from_prev_transform = None; descriptor_from_prev_transform_hyperbolic = None; sigma_from_prev_level_tensor = torch.tensor(0.0, device=dev, dtype=dtype_to_use)
        for i, level_module in enumerate(self.levels_modulelist):
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation, descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic, sigma_out_of_level_tensor) = level_module(current_point_repr_hyperbolic, aggregated_relative_vectors_from_prev_transform, descriptor_from_prev_transform_hyperbolic, sigma_from_prev_level_tensor)
            if self.hyperbolic_dims_list[i] > 0: level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)
            if i < len(self.levels_modulelist) - 1:
                if i >= len(self.transforms_modulelist): logging.getLogger("WuBuGAADHybridGenV01DFT.WuBuModel").warning(f"Missing transform L{i}->L{i+1}. Stop."); break
                transform_module = self.transforms_modulelist[i]; next_level_module = self.levels_modulelist[i+1]
                c_in_for_transform = level_module.get_current_curvature_scalar(); c_out_for_transform = next_level_module.get_current_curvature_scalar()
                (point_transformed_to_next_level_hyperbolic, boundaries_transformed_to_next_level_hyperbolic, descriptor_transformed_to_next_level_hyperbolic) = transform_module(point_out_of_level_hyperbolic, boundary_points_of_level_hyperbolic, descriptor_generated_by_level_hyperbolic, c_in_for_transform, c_out_for_transform)
                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic; descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic; sigma_from_prev_level_tensor = sigma_out_of_level_tensor; aggregated_relative_vectors_from_prev_transform = None
                if boundaries_transformed_to_next_level_hyperbolic is not None and self.relative_vector_aggregation_mode not in ['none', None] and self.hyperbolic_dims_list[i+1] > 0 and current_point_repr_hyperbolic.shape[-1] > 0 :
                    manifold_next_level_obj = PoincareBall(c_scalar=c_out_for_transform); tan_main_next_level = manifold_next_level_obj.logmap0(current_point_repr_hyperbolic); tan_bounds_next_level = manifold_next_level_obj.logmap0(boundaries_transformed_to_next_level_hyperbolic); tan_bounds_next_level_expanded = tan_bounds_next_level.unsqueeze(0).expand(B_prime_for_levels, -1, -1); relative_tangent_vectors = tan_main_next_level.unsqueeze(1) - tan_bounds_next_level_expanded; agg_mode = self.relative_vector_aggregation_mode
                    if agg_mode == "mean": agg_rel_vec = torch.mean(relative_tangent_vectors, dim=1)
                    elif agg_mode == "sum": agg_rel_vec = torch.sum(relative_tangent_vectors, dim=1)
                    elif agg_mode == "max_norm": norms = torch.norm(relative_tangent_vectors, p=2, dim=-1); best_idx = torch.argmax(norms, dim=1, keepdim=True); best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, relative_tangent_vectors.shape[-1]); agg_rel_vec = torch.gather(relative_tangent_vectors, 1, best_idx_expanded).squeeze(1)
                    else: agg_rel_vec = None
                    aggregated_relative_vectors_from_prev_transform = torch.zeros_like(tan_main_next_level) if agg_rel_vec is not None and not torch.isfinite(agg_rel_vec).all() else agg_rel_vec
        compatible_tangent_outputs = [t_val.to(dtype_to_use) for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation) if t_val is not None and t_idx < len(self.hyperbolic_dims_list) and self.hyperbolic_dims_list[t_idx] > 0 and torch.isfinite(t_val).all()]
        if not compatible_tangent_outputs: out_zeros = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use); return out_zeros.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim == 3 else out_zeros
        aggregated_tangent_final = torch.cat(compatible_tangent_outputs, dim=-1); final_output_flat = self.output_tangent_projection(aggregated_tangent_final); final_output_flat = torch.nan_to_num(final_output_flat, nan=0.0) if not torch.isfinite(final_output_flat).all() else final_output_flat
        return final_output_flat.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim == 3 else final_output_flat

class GradientStats: # ... (No changes from original, keep as is)
    def __init__(self): self.reset()
    def reset(self): self.total_params_updated=0; self.total_finite_grads_processed=0; self.total_non_finite_grads_encountered=0; self.params_skipped_due_non_finite_grad=0; self.max_grad_norm_observed=0.; self.step_summary={}
    def record_param_grad(self, grad_is_finite: bool, original_norm_if_finite: Optional[float] = None):
        if grad_is_finite: self.total_finite_grads_processed += 1; self.max_grad_norm_observed = max(self.max_grad_norm_observed, original_norm_if_finite if original_norm_if_finite is not None else 0.0)
        else: self.total_non_finite_grads_encountered += 1; self.params_skipped_due_non_finite_grad += 1
    def finalize_step_stats(self, num_params_in_optimizer_step: int): self.total_params_updated=num_params_in_optimizer_step-self.params_skipped_due_non_finite_grad; self.step_summary={"params_in_step":num_params_in_optimizer_step, "params_updated":self.total_params_updated, "params_skipped_non_finite_grad":self.params_skipped_due_non_finite_grad, "initial_finite_grads":self.total_finite_grads_processed, "initial_non_finite_grads":self.total_non_finite_grads_encountered, "max_finite_grad_norm_observed":self.max_grad_norm_observed}
    def get_step_summary_for_logging(self) -> dict: return self.step_summary.copy()

class HAKMEMQController: # ... (No changes from original, keep as is)
    def __init__(self, q_learning_rate: float = 0.01, discount_factor: float = 0.90, epsilon_start: float = 0.5, epsilon_min: float = 0.05, epsilon_decay: float = 0.9995, lr_scale_options: list[float] | None = None, momentum_scale_options: list[float] | None = None, lambda_kl_scale_options: list[float] | None = None, max_q_table_size: int = 25000, state_history_len: int = 5, lambda_kl_state_history_len: int = 5, reward_clipping: tuple[float, float] | None = (-2.0, 2.0), q_value_clipping: tuple[float, float] | None = (-30.0, 30.0)):
        self.q_table: dict[tuple, dict[str, np.ndarray]] = {}; self.alpha = q_learning_rate; self.gamma = discount_factor; self.epsilon_start = epsilon_start; self.epsilon = self.epsilon_start; self.epsilon_min = epsilon_min; self.epsilon_decay = epsilon_decay; self.reward_clipping = reward_clipping; self.q_value_clipping = q_value_clipping; self.current_lambda_kl: float = 0.0001
        _lr_options = lr_scale_options if lr_scale_options is not None else [0.8, 0.9, 1.0, 1.1, 1.2]; _mom_options = momentum_scale_options if momentum_scale_options is not None else [0.95, 0.98, 1.0, 1.01, 1.02]; _lkl_options = lambda_kl_scale_options if lambda_kl_scale_options is not None else [0.94, 0.97, 1.0, 1.03, 1.06]
        self.action_ranges = {'lr_scale': np.array(_lr_options, dtype=np.float32), 'momentum_scale': np.array(_mom_options, dtype=np.float32), 'lambda_kl_scale': np.array(_lkl_options, dtype=np.float32)}; self.num_actions = {p_type: len(actions) for p_type, actions in self.action_ranges.items()}
        self.state_history_len = max(3, state_history_len); self.loss_g_total_hist = deque(maxlen=self.state_history_len); self.loss_g_recon_hist = deque(maxlen=self.state_history_len); self.loss_g_kl_hist = deque(maxlen=self.state_history_len); self.loss_g_adv_hist = deque(maxlen=self.state_history_len); self.loss_d_total_hist = deque(maxlen=self.state_history_len); self.loss_d_real_hist = deque(maxlen=self.state_history_len); self.loss_d_fake_hist = deque(maxlen=self.state_history_len)
        self.lambda_kl_state_history_len = max(2, lambda_kl_state_history_len); self.interval_avg_recon_hist = deque(maxlen=self.lambda_kl_state_history_len); self.interval_avg_kl_div_hist = deque(maxlen=self.lambda_kl_state_history_len); self.interval_avg_d_total_hist = deque(maxlen=self.lambda_kl_state_history_len); self.interval_val_metric_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.prev_lr_mom_state: tuple | None = None; self.prev_lr_mom_action: dict[str, float] | None = None; self.prev_lambda_kl_state: tuple | None = None; self.prev_lambda_kl_action: dict[str, float] | None = None; self.reward_hist = deque(maxlen=100)
        self.max_q_table_size = max_q_table_size; self.q_table_access_count: dict[tuple, int] = defaultdict(int); self.q_table_creation_time: dict[tuple, float] = {}; self.q_table_last_access_time: dict[tuple, float] = {}
        self.reward_weights = {"g_recon_improvement": 2.5, "g_adv_improvement": 1.2, "g_kl_control_penalty": 0.3, "g_loss_stability": 0.1, "d_balance_target": 1.5, "d_real_low_bonus": 0.7, "d_fake_low_meaningful_bonus": 0.7, "d_loss_stability": 0.1, "gan_balance_g_bonus": 0.3, "gan_balance_d_penalty": 0.3, "oscillation_penalty": 0.25, "extreme_loss_penalty": 0.75, "lambda_kl_recon_focus": 1.5, "lambda_kl_kl_target_range": 1.0, "lambda_kl_val_metric_improvement": 2.0, "lambda_kl_stability_penalty": 0.5}
        self.logger = logging.getLogger(f"WuBuGAADHybridGenV01DFT.QController"); self.logger.info(f"HAKMEMQController (LR/Mom + Scheduled LambdaKL) initialized. Eps: {self.epsilon_start:.2f}->{self.epsilon_min:.2f}"); self._internal_step_counter = 0
    def _get_trend_bin(self, history: deque, current_val: float | None, relative_to_median: bool = True, value_scale_for_diff:float = 1.0) -> int:
        if current_val is None or not np.isfinite(current_val): return 2
        valid_history = [h for h in history if np.isfinite(h)];
        if not valid_history: return 2
        prev_median = np.median(valid_history); diff = current_val - prev_median
        if relative_to_median: denominator_scale = abs(prev_median); denominator_scale = max(abs(current_val), value_scale_for_diff * 0.01 + EPS) if denominator_scale < value_scale_for_diff * 0.01 + EPS else denominator_scale; denominator = denominator_scale + EPS; relative_diff = diff / denominator
        else: denominator = value_scale_for_diff + EPS; relative_diff = diff / denominator
        if relative_diff < -0.15: return 0
        if relative_diff < -0.02: return 1
        if relative_diff <= 0.02: return 2
        if relative_diff <= 0.15: return 3
        return 4
    def _update_loss_histories(self, current_losses: dict[str, float]):
        loss_map = {'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist, 'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist, 'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist, 'loss_d_fake': self.loss_d_fake_hist}
        for name, deq in loss_map.items(): loss_val = current_losses.get(name); deq.append(loss_val) if loss_val is not None and np.isfinite(loss_val) else None
    def get_lr_mom_state(self, current_losses: dict[str, float], current_lr: float, current_momentum: float, is_generator_q: bool) -> tuple | None:
        self._internal_step_counter +=1; self._update_loss_histories(current_losses)
        required_keys_g = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']; required_keys_d = ['loss_d_total', 'loss_g_total', 'loss_d_real', 'loss_d_fake', 'loss_g_adv']
        required_keys = required_keys_g if is_generator_q else required_keys_d
        if not all(key in current_losses and np.isfinite(current_losses[key]) for key in required_keys): return None
        if is_generator_q:
            s_g_total_trend = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total']); s_d_total_trend_opp = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total']); s_g_recon_trend = self._get_trend_bin(self.loss_g_recon_hist, current_losses['loss_g_recon'])
            kl_val, recon_val = current_losses['loss_g_kl'], current_losses['loss_g_recon']; s_kl_problem = 0
            if (self.current_lambda_kl * kl_val > self.reward_weights.get("g_recon_improvement", 2.0) * recon_val * 2.0 and recon_val > 0.10 and self.current_lambda_kl > 0.0005): s_kl_problem = 1
            elif kl_val > 150.0 and recon_val > 0.15 and self.current_lambda_kl > 0.005: s_kl_problem = 2
            s_g_adv_level = np.digitize(current_losses['loss_g_adv'], [0.3, 0.6, 1.0]).item(); s_lr_bin = np.digitize(current_lr, [1e-5, 5e-5, 2e-4]).item(); s_mom_bin = np.digitize(current_momentum, [0.85, 0.95]).item()
            state_tuple = ("LRM_G", s_g_total_trend, s_d_total_trend_opp, s_g_recon_trend, s_kl_problem, s_g_adv_level, s_lr_bin, s_mom_bin, np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item())
        else:
            s_d_total_trend = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total']); s_g_total_trend_opp = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total']); s_d_balance_bin = np.digitize(current_losses['loss_d_total'], [0.35, 0.65, 0.85]).item(); s_d_fake_vs_real_ratio_bin = np.digitize(current_losses['loss_d_fake'] / (current_losses['loss_d_real'] + EPS), [0.8, 1.2, 2.0]).item()
            s_lr_bin = np.digitize(current_lr, [1e-5, 5e-5, 2e-4]).item(); s_mom_bin = np.digitize(current_momentum, [0.85, 0.95]).item()
            state_tuple = ("LRM_D", s_d_total_trend, s_g_total_trend_opp, s_d_balance_bin, s_d_fake_vs_real_ratio_bin, np.digitize(current_losses.get('loss_g_adv', 0.7), [0.2, 0.5]).item(), s_lr_bin, s_mom_bin, np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item())
        self._ensure_q_state_exists(state_tuple); return state_tuple
    def get_lambda_kl_state(self, interval_metrics: dict[str, float | None]) -> tuple | None:
        required_keys = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric', 'current_lambda_kl_val']
        if not all(key in interval_metrics and interval_metrics[key] is not None and np.isfinite(interval_metrics[key]) for key in required_keys): self.logger.debug(f"LambdaKL QState: Insufficient/non-finite interval metrics. Need: {required_keys}, Got: {interval_metrics}"); return None # type: ignore
        self.interval_avg_recon_hist.append(float(interval_metrics['avg_recon'])); self.interval_avg_kl_div_hist.append(float(interval_metrics['avg_kl_div'])); self.interval_avg_d_total_hist.append(float(interval_metrics['avg_d_total'])); self.interval_val_metric_hist.append(float(interval_metrics['val_metric'])) # type: ignore
        s_interval_recon_trend = self._get_trend_bin(self.interval_avg_recon_hist, float(interval_metrics['avg_recon'])); s_interval_kl_trend = self._get_trend_bin(self.interval_avg_kl_div_hist, float(interval_metrics['avg_kl_div'])); s_interval_val_metric_trend = self._get_trend_bin(self.interval_val_metric_hist, float(interval_metrics['val_metric'])) # type: ignore
        s_current_lambda_kl_bin = np.digitize(float(interval_metrics['current_lambda_kl_val']), [0.0005, 0.005, 0.05]).item(); s_interval_d_balance_bin = np.digitize(float(interval_metrics['avg_d_total']), [0.35, 0.65, 0.85]).item() # type: ignore
        state_tuple = ("LKL", s_interval_recon_trend, s_interval_kl_trend, s_interval_val_metric_trend, s_current_lambda_kl_bin, s_interval_d_balance_bin, np.digitize(self.epsilon, [self.epsilon_min * 2, self.epsilon_start * 0.6]).item())
        self._ensure_q_state_exists(state_tuple); return state_tuple
    def _ensure_q_state_exists(self, state_tuple: tuple):
        current_time = time.time(); self.q_table_access_count[state_tuple] += 1; self.q_table_last_access_time[state_tuple] = current_time
        if state_tuple not in self.q_table: self.q_table[state_tuple] = {p_type: np.zeros(n_actions, dtype=np.float32) for p_type, n_actions in self.num_actions.items()}; self.q_table_creation_time[state_tuple] = current_time; self._manage_q_table_size()
    def choose_action(self, state: tuple | None, mode: str = 'lr_mom') -> dict[str, float]:
        default_actions = {'lr_scale': 1.0, 'momentum_scale': 1.0, 'lambda_kl_scale': 1.0}; action_types_to_choose = []
        if mode == 'lr_mom': action_types_to_choose = ['lr_scale', 'momentum_scale']
        elif mode == 'lambda_kl': action_types_to_choose = ['lambda_kl_scale']
        else: raise ValueError(f"Invalid mode for choose_action: {mode}")
        if state is None or state not in self.q_table: return {k: default_actions[k] for k in action_types_to_choose}
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay); chosen_actions = {}
        for param_type in action_types_to_choose:
            q_values = self.q_table[state].get(param_type); action_space = self.action_ranges[param_type]
            if q_values is None: self.logger.error(f"Q-values for {param_type} missing in state {state}. Choosing default."); chosen_actions[param_type] = default_actions[param_type]; continue
            if random.random() < self.epsilon: chosen_idx = random.randrange(len(action_space))
            else:
                finite_q = q_values[np.isfinite(q_values)]
                if finite_q.size > 0: best_q_val = np.max(finite_q); best_indices_options = np.where(np.isclose(q_values, best_q_val) & np.isfinite(q_values))[0]; chosen_idx = random.choice(best_indices_options) if best_indices_options.size > 0 else random.randrange(len(action_space))
                else: chosen_idx = random.randrange(len(action_space)); self.logger.warning(f"State {state}, PType {param_type}: All Q-vals non-finite. Random action.")
            chosen_actions[param_type] = float(action_space[chosen_idx])
        if mode == 'lr_mom': self.prev_lr_mom_action = chosen_actions.copy()
        elif mode == 'lambda_kl': self.prev_lambda_kl_action = chosen_actions.copy()
        return chosen_actions
    def update_q_values(self, state: tuple, action: dict[str, float], reward: float, next_state: tuple | None, mode: str = 'lr_mom'):
        if state not in self.q_table: return
        if self.reward_clipping: reward = np.clip(reward, self.reward_clipping[0], self.reward_clipping[1])
        self.reward_hist.append(reward); action_types_to_update = list(action.keys())
        for param_type in action_types_to_update:
            chosen_scale_value = action[param_type]; action_idx_arr = np.where(np.isclose(self.action_ranges[param_type], chosen_scale_value))[0]
            if not action_idx_arr.size: continue
            action_idx = action_idx_arr[0]; current_q_value = self.q_table[state][param_type][action_idx]; max_future_q = 0.0
            if next_state is not None and next_state in self.q_table and param_type in self.q_table[next_state]: next_q_vals = self.q_table[next_state][param_type]; max_future_q = np.max(next_q_vals[np.isfinite(next_q_vals)]) if np.any(np.isfinite(next_q_vals)) else 0.0
            td_target = reward + self.gamma * max_future_q; td_error = td_target - current_q_value; new_q_value = current_q_value + self.alpha * td_error
            if np.isfinite(new_q_value): self.q_table[state][param_type][action_idx] = np.clip(new_q_value, self.q_value_clipping[0], self.q_value_clipping[1]) if self.q_value_clipping else new_q_value
    def _manage_q_table_size(self):
        if len(self.q_table) <= self.max_q_table_size: return
        num_to_prune = len(self.q_table) - self.max_q_table_size; current_time = time.time()
        state_scores = {s_tuple: (self.q_table_access_count.get(s_tuple, 1) * (1.0 + np.log1p((current_time - self.q_table_creation_time.get(s_tuple, current_time)) / 86400.0)) * (1.0 / (1.0 + np.log1p((current_time - self.q_table_last_access_time.get(s_tuple, current_time)) / 3600.0)))) for s_tuple in self.q_table.keys()}
        sorted_states_for_pruning = sorted(state_scores.keys(), key=lambda s: state_scores[s]); pruned_count = 0
        for i in range(num_to_prune):
            if i < len(sorted_states_for_pruning): s_rm = sorted_states_for_pruning[i]; self.q_table.pop(s_rm, None); self.q_table_access_count.pop(s_rm, None); self.q_table_creation_time.pop(s_rm, None); self.q_table_last_access_time.pop(s_rm, None); pruned_count +=1
        if pruned_count > 0: self.logger.info(f"Pruned {pruned_count} Q-table entries. New size: {len(self.q_table)}.")
    def compute_lr_mom_reward(self, current_losses: dict[str, float], is_generator_q: bool) -> float:
        total_reward = 0.0; w = self.reward_weights
        for loss_name, loss_val in current_losses.items():
            if not np.isfinite(loss_val): total_reward -= w["extreme_loss_penalty"] * 5; current_losses[loss_name] = 100.0
            elif abs(loss_val) > 500: total_reward -= w["extreme_loss_penalty"] * (abs(loss_val) / 500.0); current_losses[loss_name] = np.sign(loss_val) * 500
        def get_prev_median(hist_deque, current_val_fallback): valid_hist = [v for v in hist_deque if np.isfinite(v)]; return np.median(valid_hist[:-1]) if len(valid_hist) > 1 else (valid_hist[0] if len(valid_hist) == 1 else current_val_fallback)
        if is_generator_q:
            loss_g_recon = current_losses.get('loss_g_recon', 1.0); prev_g_recon = get_prev_median(self.loss_g_recon_hist, loss_g_recon); recon_improvement = prev_g_recon - loss_g_recon; recon_scale = 1.0 + math.log1p(max(0, loss_g_recon - 0.02) * 20); total_reward += np.tanh(recon_improvement / (abs(prev_g_recon) + 0.01 + EPS) * recon_scale) * w["g_recon_improvement"]
            loss_g_adv = current_losses.get('loss_g_adv', 0.7); prev_g_adv = get_prev_median(self.loss_g_adv_hist, loss_g_adv); adv_improvement = prev_g_adv - loss_g_adv; total_reward += np.tanh(adv_improvement / (abs(prev_g_adv) + EPS)) * w["g_adv_improvement"]
            loss_g_kl = current_losses.get('loss_g_kl', 0.0)
            if loss_g_kl > 100.0 and self.current_lambda_kl >= 0.0005 and loss_g_recon > 0.1: total_reward -= w["g_kl_control_penalty"] * min(1.0, (loss_g_kl - 100.0) / 200.0)
            loss_d_total = current_losses.get('loss_d_total', 0.7)
            if 0.4 < loss_d_total < 0.75: total_reward += w["gan_balance_g_bonus"]
            elif loss_d_total <= 0.3: total_reward -= w["gan_balance_g_bonus"] * 1.5
            loss_g_total = current_losses.get('loss_g_total', 1.0); prev_g_total = get_prev_median(self.loss_g_total_hist, loss_g_total); g_total_improvement = prev_g_total - loss_g_total; total_reward += np.tanh(g_total_improvement / (abs(prev_g_total) + EPS)) * w["g_loss_stability"]
        else:
            loss_d_total = current_losses.get('loss_d_total', 0.7)
            if 0.4 < loss_d_total < 0.65: total_reward += w["d_balance_target"]
            elif loss_d_total < 0.3: total_reward -= w["d_balance_target"] * 0.5 
            elif loss_d_total > 0.8: total_reward -= w["d_balance_target"] * 0.75
            loss_d_real = current_losses.get('loss_d_real', 0.7)
            if loss_d_real < 0.3: total_reward += w["d_real_low_bonus"] * (0.3 - loss_d_real) / 0.3
            loss_d_fake = current_losses.get('loss_d_fake', 0.7); loss_g_adv_opp = current_losses.get('loss_g_adv', 0.7)
            if loss_d_fake < 0.3 and loss_g_adv_opp > 0.4: total_reward += w["d_fake_low_meaningful_bonus"] * (0.3 - loss_d_fake) / 0.3
            if loss_g_adv_opp < 0.25: total_reward -= w["gan_balance_d_penalty"]
            prev_d_total = get_prev_median(self.loss_d_total_hist, loss_d_total); d_total_improvement = prev_d_total - loss_d_total; total_reward += np.tanh(d_total_improvement / (abs(prev_d_total) + EPS)) * w["d_loss_stability"]
        if len(self.reward_hist) >= self.state_history_len:
            recent_rewards = list(self.reward_hist)[-self.state_history_len:]; sign_flips = 0
            for i in range(len(recent_rewards) - 1):
                if np.sign(recent_rewards[i]) != np.sign(recent_rewards[i+1]) and abs(recent_rewards[i]) > 0.05 and abs(recent_rewards[i+1]) > 0.05: sign_flips += 1
            if sign_flips >= (self.state_history_len // 2) : total_reward -= w["oscillation_penalty"] * (sign_flips / self.state_history_len)
        if self.reward_clipping: total_reward = np.clip(total_reward, self.reward_clipping[0], self.reward_clipping[1])
        return float(total_reward)
    def compute_lambda_kl_reward(self, interval_metrics: dict[str, float | None], prev_interval_metrics: dict[str, float | None] | None) -> float:
        total_reward = 0.0; w = self.reward_weights; _prev_metrics = prev_interval_metrics if prev_interval_metrics is not None else {}
        current_val_metric = interval_metrics.get('val_metric'); prev_val_metric = _prev_metrics.get('val_metric', current_val_metric)
        if current_val_metric is not None and prev_val_metric is not None and np.isfinite(current_val_metric) and np.isfinite(prev_val_metric): val_metric_change = float(current_val_metric) - float(prev_val_metric); total_reward += np.tanh(val_metric_change * 5.0) * w["lambda_kl_val_metric_improvement"]
        current_avg_recon = interval_metrics.get('avg_recon'); prev_avg_recon = _prev_metrics.get('avg_recon', current_avg_recon)
        if current_avg_recon is not None and prev_avg_recon is not None and np.isfinite(current_avg_recon) and np.isfinite(prev_avg_recon): recon_change = float(prev_avg_recon) - float(current_avg_recon); recon_penalty_factor = 1.0 if recon_change >= -0.05 else (1.0 + abs(recon_change * 10)); total_reward += np.tanh(recon_change * 10.0 / recon_penalty_factor) * w["lambda_kl_recon_focus"]
        current_kl_div = interval_metrics.get('avg_kl_div'); prev_kl_div = _prev_metrics.get('avg_kl_div', current_kl_div)
        if current_kl_div is not None and prev_kl_div is not None and np.isfinite(current_kl_div) and np.isfinite(prev_kl_div):
            if float(current_kl_div) > 100: kl_div_decrease = float(prev_kl_div) - float(current_kl_div); total_reward += np.tanh(kl_div_decrease / 50.0) * w["lambda_kl_kl_target_range"]
            elif float(current_kl_div) < 20 and current_avg_recon is not None and float(current_avg_recon) > 0.05 : total_reward -= w["lambda_kl_kl_target_range"] * 0.5
        current_avg_d_total = interval_metrics.get('avg_d_total'); prev_avg_d_total = _prev_metrics.get('avg_d_total', current_avg_d_total)
        if current_avg_d_total is not None and prev_avg_d_total is not None and np.isfinite(current_avg_d_total) and np.isfinite(prev_avg_d_total): d_total_stability_change = abs(float(current_avg_d_total) - float(prev_avg_d_total)); total_reward -= w["lambda_kl_stability_penalty"] * (d_total_stability_change / 0.2) if d_total_stability_change > 0.2 else 0.0
        current_lambda_kl_val = interval_metrics.get('current_lambda_kl_val')
        if current_lambda_kl_val is not None and float(current_lambda_kl_val) > 0.5 and current_avg_recon is not None and float(current_avg_recon) > 0.15: total_reward -= 0.5
        if self.logger.isEnabledFor(logging.DEBUG): log_mets = {k: f'{v:.3f}' if isinstance(v, (float, np.float32, np.float64)) and np.isfinite(v) else str(v) for k,v in interval_metrics.items()}; self.logger.debug(f"LambdaKL_Rew: Raw={total_reward:.3f}. IntervalMet: {log_mets}")
        if self.reward_clipping: total_reward = np.clip(total_reward, self.reward_clipping[0], self.reward_clipping[1])
        return float(total_reward)
    def set_current_lambda_kl(self, lambda_kl_val: float):
        if np.isfinite(lambda_kl_val): self.current_lambda_kl = float(lambda_kl_val)
        else: self.logger.warning(f"Attempted to set non-finite lambda_kl: {lambda_kl_val}")
    def get_info(self) -> dict:
        q_mem_mb = 0.0;
        try: q_mem_mb = sum(sys.getsizeof(s_tuple) + sum(q_vals.nbytes + sys.getsizeof(p_type) for p_type, q_vals in q_actions.items()) for s_tuple, q_actions in self.q_table.items()) / (1024**2) if self.q_table else 0.0
        except Exception: q_mem_mb = -1.0
        avg_reward_recent = np.mean(list(self.reward_hist)) if self.reward_hist else 0.0
        return {"epsilon": round(self.epsilon, 4), "q_table_size": len(self.q_table), "q_table_mem_mb_approx": round(q_mem_mb, 3), "last_lr_mom_action": self.prev_lr_mom_action if self.prev_lr_mom_action else "None", "last_lambda_kl_action": self.prev_lambda_kl_action if self.prev_lambda_kl_action else "None", f"avg_reward_last_{len(self.reward_hist) if self.reward_hist.maxlen is None else self.reward_hist.maxlen}": round(avg_reward_recent, 3)}
    def set_initial_losses(self, losses: dict[str, float], is_generator_q: bool):
        loss_map_init = {'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,'loss_d_fake': self.loss_d_fake_hist}
        relevant_keys = (['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total'] if is_generator_q else ['loss_d_total', 'loss_d_real', 'loss_d_fake', 'loss_g_total', 'loss_g_adv'])
        for name in relevant_keys: val = losses.get(name); loss_map_init[name].append(val) if val is not None and np.isfinite(val) else None
        for deq_obj in loss_map_init.values():
            if deq_obj:
                 while len(deq_obj) < self.state_history_len: deq_obj.appendleft(deq_obj[0])
    def set_initial_lambda_kl_metrics(self, interval_metrics: dict[str, float | None]):
        metric_map = {'avg_recon': self.interval_avg_recon_hist, 'avg_kl_div': self.interval_avg_kl_div_hist, 'avg_d_total': self.interval_avg_d_total_hist, 'val_metric': self.interval_val_metric_hist}
        for name, deq in metric_map.items(): val = interval_metrics.get(name); deq.append(float(val)) if val is not None and np.isfinite(val) else None
        for deq_obj in metric_map.values():
            if deq_obj:
                while len(deq_obj) < self.lambda_kl_state_history_len: deq_obj.appendleft(deq_obj[0])

class RiemannianEnhancedSGD(torch.optim.Optimizer): # ... (No changes from original, keep as is)
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.01, max_grad_norm_risgd: float = 1.0, q_learning_config: Optional[Dict] = None, optimizer_type: str = "generator"):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}"); defaults = dict(lr=lr, initial_lr=lr, momentum=momentum, initial_momentum=momentum, weight_decay=weight_decay); super().__init__(params, defaults)
        self.optimizer_type = optimizer_type.lower(); assert self.optimizer_type in ["generator", "discriminator"]
        if isinstance(q_learning_config, dict): self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(**q_learning_config.copy())
        else: self.q_controller = None
        self.logger = logging.getLogger(f"WuBuGAADHybridGenV01DFT.RiSGD.{self.optimizer_type.capitalize()}"); self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled for {self.optimizer_type} optimizer.")
        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf'); self._step_count_internal = 0; self.grad_stats = GradientStats()
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad: self.state.setdefault(p, {})
    def zero_grad(self, set_to_none: bool = True): super().zero_grad(set_to_none=set_to_none)
    def q_controller_update_and_set_hyperparams(self, avg_losses_dict: Dict[str, Optional[float]], current_lambda_kl_value: Optional[float] = None):
        if not self.q_controller: return
        finite_losses_for_q_state: Dict[str, float] = {k: v for k, v in avg_losses_dict.items() if v is not None and np.isfinite(v)}
        is_gen_q = (self.optimizer_type == "generator"); required_keys = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total'] if is_gen_q else ['loss_d_total', 'loss_g_total', 'loss_g_adv', 'loss_d_real', 'loss_d_fake']
        if not all(key in finite_losses_for_q_state for key in required_keys): self.logger.debug(f"QCtrl ({self.optimizer_type}): Insufficient finite losses for LR/Mom state. Skipping Q-update. Need: {required_keys}, Got: {list(finite_losses_for_q_state.keys())}"); return
        if hasattr(self.q_controller, 'set_current_lambda_kl') and current_lambda_kl_value is not None: self.q_controller.set_current_lambda_kl(current_lambda_kl_value)
        current_lr_for_q_state = self.param_groups[0]['lr']; current_mom_for_q_state = self.param_groups[0]['momentum']
        q_state_current = self.q_controller.get_lr_mom_state(finite_losses_for_q_state, current_lr_for_q_state, current_mom_for_q_state, is_generator_q=is_gen_q)
        if self.q_controller.prev_lr_mom_state is not None and self.q_controller.prev_lr_mom_action is not None and q_state_current is not None:
            reward = self.q_controller.compute_lr_mom_reward(finite_losses_for_q_state, is_generator_q=is_gen_q)
            if np.isfinite(reward): self.q_controller.update_q_values(self.q_controller.prev_lr_mom_state, self.q_controller.prev_lr_mom_action, reward, q_state_current, mode='lr_mom')
        elif q_state_current is not None and hasattr(self.q_controller, 'set_initial_losses'): self.q_controller.set_initial_losses(finite_losses_for_q_state, is_generator_q=is_gen_q)
        self.q_controller.prev_lr_mom_state = q_state_current; action_for_upcoming_step = self.q_controller.choose_action(q_state_current, mode='lr_mom')
        if action_for_upcoming_step:
            for group in self.param_groups: base_lr = group['initial_lr']; base_mom = group['initial_momentum']; group['lr'] = float(np.clip(base_lr * action_for_upcoming_step.get('lr_scale', 1.0), 1e-8, 1.0)); group['momentum'] = float(np.clip(base_mom * action_for_upcoming_step.get('momentum_scale', 1.0), 0.0, 0.999))
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad
                if not torch.isfinite(grad).all(): self.logger.warning(f"Optimizer step: Non-finite gradient for param shape {p.shape} ({self.optimizer_type}). Skipping update."); self.state[p].pop('momentum_buffer', None); continue
                if self.max_grad_norm_risgd > 0 and self.max_grad_norm_risgd != float('inf') : param_grad_norm = grad.norm().item(); grad.mul_(self.max_grad_norm_risgd / (param_grad_norm + EPS)) if param_grad_norm > self.max_grad_norm_risgd else None
                manifold: Optional[Manifold] = getattr(p, 'manifold', None)
                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    p_projected_on_manifold = manifold.proju(p); grad_eff = grad.clone(); grad_eff.add_(p, alpha=weight_decay) if weight_decay != 0 else None
                    try: riemannian_grad = manifold.egrad2rgrad(p_projected_on_manifold, grad_eff)
                    except Exception as e_egrad: self.logger.error(f"egrad2rgrad failed for P:{p.shape} (c={manifold.c:.2e}): {e_egrad}. Skipping."); self.state[p].pop('momentum_buffer', None); continue
                    if not torch.isfinite(riemannian_grad).all(): self.logger.warning(f"Non-finite Riemannian grad P:{p.shape} (c={manifold.c:.2e}). Skipping."); self.state[p].pop('momentum_buffer', None); continue
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0: buf = torch.clone(riemannian_grad).detach() if buf is None or buf.shape != riemannian_grad.shape else buf.mul_(momentum).add_(riemannian_grad); self.state[p]['momentum_buffer'] = buf
                    else: buf = riemannian_grad
                    if not torch.isfinite(buf).all(): self.logger.warning(f"Non-finite momentum buffer P:{p.shape} (c={manifold.c:.2e}). Resetting."); buf.zero_(); self.state[p]['momentum_buffer'] = buf
                    expmap_tangent_vector = buf.mul(-lr)
                    if not torch.isfinite(expmap_tangent_vector).all(): self.logger.warning(f"Non-finite tangent vector for expmap P:{p.shape} (c={manifold.c:.2e}). Skipping update."); continue
                    try:
                        new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
                        if not torch.isfinite(new_p_candidate).all(): self.logger.warning(f"Expmap non-finite P:{p.shape} (c={manifold.c:.2e}). Projecting & zeroing momentum."); p.data = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0)); self.state[p].get('momentum_buffer', torch.zeros_like(buf)).zero_()
                        else: p.data = manifold.proju(new_p_candidate)
                    except Exception as e_expmap: self.logger.error(f"Expmap failed P:{p.shape} (c={manifold.c:.2e}): {e_expmap}. Zeroing momentum."); self.state[p].get('momentum_buffer', torch.zeros_like(buf)).zero_(); continue
                    if not torch.isfinite(p.data).all(): self.logger.error(f"P:{p.shape} (c={manifold.c:.2e}) non-finite post-update. Resetting."); p.data = manifold.expmap0(torch.zeros_like(p.data, device=p.device)); self.state[p].get('momentum_buffer', torch.zeros_like(buf)).zero_()
                else:
                    grad_eff_euc = grad.clone(); grad_eff_euc.add_(p, alpha=weight_decay) if weight_decay != 0 else None
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0: buf = torch.clone(grad_eff_euc).detach() if buf is None or buf.shape != grad_eff_euc.shape else buf.mul_(momentum).add_(grad_eff_euc); self.state[p]['momentum_buffer'] = buf
                    else: buf = grad_eff_euc
                    if not torch.isfinite(buf).all(): self.logger.warning(f"Non-finite Euclidean momentum P:{p.shape}. Resetting."); buf.zero_(); self.state[p]['momentum_buffer'] = buf
                    p.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all(): self.logger.warning(f"Euclidean P:{p.shape} non-finite. Clamping & zeroing momentum."); p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e5, neginf=-1e5); self.state[p].get('momentum_buffer', torch.zeros_like(buf)).zero_()
        self._step_count_internal += 1; return loss
    def get_q_controller_info(self) -> Dict: return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}
    def get_gradient_stats_summary_optimizer_view(self) -> Dict: return self.grad_stats.get_step_summary_for_logging()

# =====================================================================
# GAAD Components (Unchanged)
# =====================================================================
def golden_subdivide_rect_fixed_n(frame_dims:Tuple[int,int], num_regions_target:int, device='cpu', dtype=torch.float, min_size_px=5) -> torch.Tensor: # ... (No changes)
    W, H = frame_dims; all_rects = [[0,0,W,H]]; rect_queue = deque([(0,0,W,H,0)])
    while rect_queue and len(all_rects) < num_regions_target * 3:
        x_off, y_off, w_curr, h_curr, depth = rect_queue.popleft()
        if min(w_curr, h_curr) < min_size_px or depth > 6 : continue
        is_landscape = w_curr > h_curr + EPS; is_portrait = h_curr > w_curr + EPS
        if is_landscape: cut_w = w_curr / PHI; r1_w, r2_w = cut_w, w_curr - cut_w; all_rects.append([x_off, y_off, x_off + r1_w, y_off + h_curr]) if r1_w >= min_size_px and h_curr >= min_size_px else None; rect_queue.append((x_off, y_off, r1_w, h_curr, depth + 1)) if r1_w >= min_size_px and h_curr >= min_size_px else None; all_rects.append([x_off + r1_w, y_off, x_off + r1_w + r2_w, y_off + h_curr]) if r2_w >= min_size_px and h_curr >= min_size_px else None; rect_queue.append((x_off + r1_w, y_off, r2_w, h_curr, depth + 1)) if r2_w >= min_size_px and h_curr >= min_size_px else None
        elif is_portrait: cut_h = h_curr / PHI; r1_h, r2_h = cut_h, h_curr - cut_h; all_rects.append([x_off, y_off, x_off + w_curr, y_off + r1_h]) if w_curr >= min_size_px and r1_h >= min_size_px else None; rect_queue.append((x_off, y_off, w_curr, r1_h, depth + 1)) if w_curr >= min_size_px and r1_h >= min_size_px else None; all_rects.append([x_off, y_off + r1_h, x_off + w_curr, y_off + r1_h + r2_h]) if w_curr >= min_size_px and r2_h >= min_size_px else None; rect_queue.append((x_off, y_off + r1_h, w_curr, r2_h, depth + 1)) if w_curr >= min_size_px and r2_h >= min_size_px else None
        elif abs(w_curr - h_curr) < EPS and w_curr > min_size_px * PHI : cut_w = w_curr / PHI; r1_w, r2_w = cut_w, w_curr - cut_w; all_rects.append([x_off, y_off, x_off + r1_w, y_off + h_curr]) if r1_w >= min_size_px and h_curr >= min_size_px else None; rect_queue.append((x_off, y_off, r1_w, h_curr, depth + 1)) if r1_w >= min_size_px and h_curr >= min_size_px else None; all_rects.append([x_off + r1_w, y_off, x_off + r1_w + r2_w, y_off + h_curr]) if r2_w >= min_size_px and h_curr >= min_size_px else None; rect_queue.append((x_off + r1_w, y_off, r2_w, h_curr, depth + 1)) if r2_w >= min_size_px and h_curr >= min_size_px else None
    unique_valid_rects_tensors = []; seen_hashes = set()
    for r_coords in all_rects:
        if r_coords[0] >= r_coords[2] - EPS or r_coords[1] >= r_coords[3] - EPS: continue
        r_tensor = torch.tensor(r_coords, dtype=dtype, device=device); r_hashable = tuple(round(c, 3) for c in r_coords)
        if r_hashable not in seen_hashes: unique_valid_rects_tensors.append(r_tensor); seen_hashes.add(r_hashable)
    unique_valid_rects_tensors.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True); selected_rects = unique_valid_rects_tensors[:num_regions_target]
    if len(selected_rects) < num_regions_target: padding_box = selected_rects[-1] if selected_rects else torch.tensor([0,0,float(W),float(H)],dtype=dtype,device=device); selected_rects.extend([padding_box.clone() for _ in range(num_regions_target - len(selected_rects))])
    return torch.stack(selected_rects)

def phi_spiral_patch_centers_fixed_n(frame_dims:Tuple[int,int], num_centers:int, device='cpu', dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]: # ... (No changes)
    W, H = frame_dims; centers_xy = []; scale_factors = []; cx, cy = W / 2.0, H / 2.0
    if num_centers <= 0: return torch.empty(0,2,device=device,dtype=dtype), torch.empty(0,1,device=device,dtype=dtype)
    centers_xy.append([cx, cy]); scale_factors.append(0.25); num_spiral_points_to_generate = num_centers - 1
    if num_spiral_points_to_generate <= 0: return (torch.tensor(centers_xy, dtype=dtype, device=device), torch.tensor(scale_factors, dtype=dtype, device=device).unsqueeze(-1)) if num_centers == 1 else (torch.empty(0,2,device=device,dtype=dtype), torch.empty(0,1,device=device,dtype=dtype))
    a = 0.05 * min(W, H); b = math.log(PHI) / (math.pi / 2); angle_step = PHI * 2 * math.pi / num_spiral_points_to_generate if num_spiral_points_to_generate > 0 else 0; current_angle = 0.0
    for i in range(num_spiral_points_to_generate):
        r = min(a * math.exp(b * current_angle), max(W,H) * 0.6); x = max(0.0, min(cx + r * math.cos(current_angle), float(W))); y = max(0.0, min(cy + r * math.sin(current_angle), float(H)))
        centers_xy.append([x, y]); scale_factors.append(max(0.05, 0.20 * math.exp(-0.5 * r / (min(W,H)*0.1)))); current_angle += angle_step
    if len(centers_xy) < num_centers: num_to_pad = num_centers - len(centers_xy); last_xy = centers_xy[-1] if centers_xy else [cx,cy]; last_scale = scale_factors[-1] if scale_factors else 0.1; centers_xy.extend([last_xy] * num_to_pad); scale_factors.extend([last_scale] * num_to_pad)
    return torch.tensor(centers_xy[:num_centers], dtype=dtype, device=device), torch.tensor(scale_factors[:num_centers], dtype=dtype, device=device).unsqueeze(-1)

# =====================================================================
# Architectural Components (v0.1 - VAE-GAN Refactor + DFT)
# =====================================================================

class RegionalPatchExtractor(nn.Module): # ... (No changes, but its output interpretation changes if DFT is used)
    def __init__(self, patch_output_size: Optional[Tuple[int, int]] = None, feature_extractor: Optional[nn.Module] = None, feature_map_spatial_scale: float = 1.0, roi_align_output_size: Optional[Tuple[int, int]] = None, use_roi_align: bool = False):
        super().__init__(); self.patch_output_size = patch_output_size; self.feature_extractor = feature_extractor; self.feature_map_spatial_scale = feature_map_spatial_scale; self.roi_align_output_size = roi_align_output_size; self.use_roi_align = use_roi_align; current_logger=logging.getLogger("WuBuGAADHybridGenV01DFT.PatchExtract"); self.resize_transform=None
        if self.use_roi_align:
            if self.feature_extractor is None or self.roi_align_output_size is None: raise ValueError("feature_extractor and roi_align_output_size needed for use_roi_align=True")
            current_logger.info(f"Using RoIAlign. Output: {roi_align_output_size}, FeatMapScale: {feature_map_spatial_scale:.2f}")
        else:
            if self.patch_output_size is None: raise ValueError("patch_output_size needed for use_roi_align=False")
            current_logger.info(f"Using Pixel Patches. Resizing to: {patch_output_size}")
            self.resize_transform = T.Resize(patch_output_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    def forward(self, images: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
        # Returns (B_flat, NumRegions_flat_total, C_patch, H_patch, W_patch)
        # or (B_per_gpu, NumRegions_per_frame, C_patch, H_patch, W_patch) if not flattened in caller.
        # This model's RegionalVAEEncoder flattens images and bboxes before calling, so output is:
        # (B * N_frames, NumRegions, C_patch, H_patch, W_patch)
        B_img_orig, NumCh_img, H_img_orig, W_img_orig = images.shape # Input images already flat: (B*N_frames, C, H, W)
        B_bboxes, NumRegions_bboxes, _ = bboxes_batch.shape # Input bboxes: (B*N_frames, NumReg, 4)
        if B_img_orig != B_bboxes:
            raise ValueError(f"Batch size mismatch between images ({B_img_orig}) and bboxes ({B_bboxes}) in PatchExtractor")

        device = images.device; original_images_dtype = images.dtype; compute_dtype = torch.float32 if images.dtype == torch.uint8 else images.dtype; images_for_processing = images.to(compute_dtype)

        if self.use_roi_align and self.feature_extractor is not None and self.roi_align_output_size is not None:
            feature_maps = self.feature_extractor(images_for_processing); # (B*N_frames, C_feat, H_feat, W_feat)
            h_feat, w_feat = feature_maps.shape[2:]; max_w_feat_scalar=float(w_feat); max_h_feat_scalar=float(h_feat);
            
            # bboxes_batch is (B*N_frames, NumReg, 4)
            # We need to create batch_indices for roi_align: (TotalRoIs, 1)
            # TotalRoIs = (B*N_frames) * NumRegions
            all_rois_list = []
            for i in range(B_img_orig): # Iterate through the "flattened batch" index
                current_bboxes_scaled = bboxes_batch[i].to(torch.float32) * self.feature_map_spatial_scale
                current_bboxes_scaled[:,0]=torch.clamp(current_bboxes_scaled[:,0],min=0.0,max=max_w_feat_scalar-EPS); current_bboxes_scaled[:,1]=torch.clamp(current_bboxes_scaled[:,1],min=0.0,max=max_h_feat_scalar-EPS); min_for_x2=current_bboxes_scaled[:,0]; current_bboxes_scaled[:,2]=torch.clamp(current_bboxes_scaled[:,2],max=max_w_feat_scalar); current_bboxes_scaled[:,2]=torch.maximum(current_bboxes_scaled[:,2],min_for_x2); min_for_y2=current_bboxes_scaled[:,1]; current_bboxes_scaled[:,3]=torch.clamp(current_bboxes_scaled[:,3],max=max_h_feat_scalar); current_bboxes_scaled[:,3]=torch.maximum(current_bboxes_scaled[:,3],min_for_y2);
                
                batch_indices_for_this_image_in_flat_batch = torch.full((NumRegions_bboxes, 1), float(i), device=device, dtype=current_bboxes_scaled.dtype)
                all_rois_list.append(torch.cat([batch_indices_for_this_image_in_flat_batch, current_bboxes_scaled], dim=1))

            all_rois_for_align = torch.cat(all_rois_list, dim=0) # Shape: ((B*N_frames)*NumReg, 5)

            try: aligned_features_flat = roi_align(feature_maps, all_rois_for_align, output_size=self.roi_align_output_size, spatial_scale=1.0, aligned=True)
            except Exception as e_roi: logging.getLogger("WuBuGAADHybridGenV01DFT.PatchExtract").error(f"RoIAlign failed: {e_roi}. FeatMap:{feature_maps.shape}, RoIs:{all_rois_for_align.shape}, Output:{self.roi_align_output_size}"); raise e_roi
            
            # Reshape aligned_features_flat from ((B*N_frames)*NumReg, C_feat, H_roi, W_roi)
            # to (B*N_frames, NumReg, C_feat, H_roi, W_roi)
            C_feat=feature_maps.shape[1]; H_roi, W_roi = self.roi_align_output_size
            aligned_features_reshaped = aligned_features_flat.view(B_img_orig, NumRegions_bboxes, C_feat, H_roi, W_roi)
            return aligned_features_reshaped.to(original_images_dtype)
        else: # Pixel patches
            # images_for_processing: (B*N_frames, C, H_img, W_img)
            # bboxes_batch: (B*N_frames, NumReg, 4)
            all_patches_collected = [] # Will be list of (NumReg, C, H_patch, W_patch) tensors
            patch_h_out, patch_w_out = self.patch_output_size # type: ignore
            
            for i in range(B_img_orig): # Iterate through B*N_frames
                single_image_from_flat_batch = images_for_processing[i] # (C, H_img, W_img)
                single_image_bboxes = bboxes_batch[i] # (NumReg, 4)
                
                current_image_patches = []
                for r in range(NumRegions_bboxes):
                    x1,y1,x2,y2 = single_image_bboxes[r].round().int().tolist()
                    x1_c,y1_c=max(0,x1),max(0,y1)
                    x2_c,y2_c=min(W_img_orig,x2),min(H_img_orig,y2)
                    
                    if x1_c >= x2_c or y1_c >= y2_c:
                        patch = torch.zeros((images.shape[1], patch_h_out, patch_w_out), device=device, dtype=compute_dtype)
                    else:
                        patch = single_image_from_flat_batch[:, y1_c:y2_c, x1_c:x2_c] # (C, H_reg, W_reg)
                        patch = self.resize_transform(patch) if self.resize_transform else patch # (C, H_patch, W_patch)
                    current_image_patches.append(patch)
                all_patches_collected.append(torch.stack(current_image_patches)) # Stack regions for this image
            
            final_patches_tensor = torch.stack(all_patches_collected) # Stack images (from flat batch)
            # Shape: (B*N_frames, NumReg, C, H_patch, W_patch)
            return final_patches_tensor.to(original_images_dtype)

class PatchEmbed(nn.Module):
    def __init__(self, patch_feature_dim: int, embed_dim: int):
        super().__init__(); self.proj = nn.Linear(patch_feature_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B_flat_total_regions, patch_feature_dim) if DFT features are pre-flattened.
        # Or (B_flat_frames, NumReg, patch_feature_dim) -> needs reshape before proj
        if x.dim() == 3: # (BatchOfFrames, NumRegions, FeatureDim)
            B_frames, N_reg, D_feat = x.shape
            x = x.view(B_frames * N_reg, D_feat)
            out = self.proj(x)
            return out.view(B_frames, N_reg, -1) # Reshape back if needed by caller
        elif x.dim() == 2: # (BatchOfAllRegions, FeatureDim)
            return self.proj(x)
        else:
            raise ValueError(f"PatchEmbed input x has unsupported dimension: {x.dim()}")

# Renamed from RegionalHyperbolicEncoder
class RegionalVAEEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_config: Dict, latent_dim: int):
        super().__init__(); self.args = args; self.video_config = video_config; self.gaad_config = gaad_config; self.wubu_s_config = wubu_s_config; self.latent_dim = latent_dim; self.image_size = (args.image_h, args.image_w); self.num_appearance_regions = gaad_config['num_regions']; self.decomposition_type = gaad_config['decomposition_type']; self.gaad_min_size_px = gaad_config.get('min_size_px', 5); current_logger=logging.getLogger("WuBuGAADHybridGenV01DFT.EncoderVAE"); self.feature_extractor: Optional[nn.Module] = None; patch_input_channels_for_dft = self.video_config['num_channels']

        # Determine patch dimensions for DFT
        if args.use_dft_features_appearance:
            self.enc_patch_h_for_dft = args.dft_patch_size_h
            self.enc_patch_w_for_dft = args.dft_patch_size_w
            # If DFT is used, RegionalPatchExtractor should output patches of this size.
            # If RoIAlign is used with DFT, its output should be dft_patch_size.
            # If pixel patches are used with DFT, they should be resized to dft_patch_size.
            roi_align_output_size_for_dft = (args.dft_patch_size_h, args.dft_patch_size_w)
            pixel_patch_output_size_for_dft = (args.dft_patch_size_h, args.dft_patch_size_w)
            current_logger.info(f"DFT App Enc: Using patch size {self.enc_patch_h_for_dft}x{self.enc_patch_w_for_dft} for DFT input.")
        else: # Original non-DFT path
            self.enc_patch_h_for_dft = 0 # Not used
            self.enc_patch_w_for_dft = 0 # Not used

        use_roi_align_effective = args.encoder_use_roi_align
        if args.use_dft_features_appearance and args.encoder_use_roi_align:
            # If DFT uses RoIAlign, the shallow CNN output becomes input to DFT
            self.feature_extractor = nn.Sequential(nn.Conv2d(self.video_config['num_channels'], args.encoder_shallow_cnn_channels, kernel_size=3, stride=1, padding=1), nn.GroupNorm(8, args.encoder_shallow_cnn_channels), nn.GELU())
            patch_input_channels_for_dft = args.encoder_shallow_cnn_channels
            current_logger.info(f"DFT App Enc: RoIAlign ON. ShallowCNN (Ch:{patch_input_channels_for_dft}) -> RoIAlign (Size:{roi_align_output_size_for_dft}) -> DFT.")
            self.patch_extractor = RegionalPatchExtractor(
                feature_extractor=self.feature_extractor,
                roi_align_output_size=roi_align_output_size_for_dft, # DFT patch size
                use_roi_align=True
            )
        elif args.use_dft_features_appearance and not args.encoder_use_roi_align:
            # DFT with pixel patches, resized to dft_patch_size
            patch_input_channels_for_dft = self.video_config['num_channels']
            current_logger.info(f"DFT App Enc: RoIAlign OFF. Pixel Patches -> Resize (Size:{pixel_patch_output_size_for_dft}) -> DFT.")
            self.patch_extractor = RegionalPatchExtractor(
                patch_output_size=pixel_patch_output_size_for_dft, # DFT patch size
                use_roi_align=False
            )
        elif not args.use_dft_features_appearance and args.encoder_use_roi_align:
            # Original RoIAlign path (no DFT)
            self.feature_extractor = nn.Sequential(nn.Conv2d(self.video_config['num_channels'], args.encoder_shallow_cnn_channels, kernel_size=3, stride=1, padding=1), nn.GroupNorm(8, args.encoder_shallow_cnn_channels), nn.GELU())
            patch_input_channels_non_dft = args.encoder_shallow_cnn_channels
            roi_align_output_size_non_dft = (args.encoder_roi_align_output_h, args.encoder_roi_align_output_w)
            current_logger.info(f"App Enc (No DFT): RoIAlign ON (OutCh:{patch_input_channels_non_dft}, RoISize:{roi_align_output_size_non_dft})")
            self.patch_extractor = RegionalPatchExtractor(
                feature_extractor=self.feature_extractor,
                roi_align_output_size=roi_align_output_size_non_dft,
                use_roi_align=True
            )
            _patch_h_eff = roi_align_output_size_non_dft[0]
            _patch_w_eff = roi_align_output_size_non_dft[1]
            patch_feature_dim_for_embed = patch_input_channels_non_dft * _patch_h_eff * _patch_w_eff
        else: # Original Pixel Patch path (no DFT, no RoIAlign)
            patch_input_channels_non_dft = self.video_config['num_channels']
            pixel_patch_size_non_dft = (args.encoder_pixel_patch_size, args.encoder_pixel_patch_size)
            current_logger.info(f"App Enc (No DFT): Pixel Patches (Resize: {pixel_patch_size_non_dft})")
            self.patch_extractor = RegionalPatchExtractor(
                patch_output_size=pixel_patch_size_non_dft,
                use_roi_align=False
            )
            _patch_h_eff = pixel_patch_size_non_dft[0]
            _patch_w_eff = pixel_patch_size_non_dft[1]
            patch_feature_dim_for_embed = patch_input_channels_non_dft * _patch_h_eff * _patch_w_eff

        if args.use_dft_features_appearance:
            # DFT features are C * 2 * H_dft * (W_dft//2+1)
            self.dft_w_coeffs_one_sided = self.enc_patch_w_for_dft // 2 + 1
            patch_feature_dim_for_embed = patch_input_channels_for_dft * 2 * self.enc_patch_h_for_dft * self.dft_w_coeffs_one_sided
            current_logger.info(f"DFT App Enc: PatchEmbed input dim: {patch_feature_dim_for_embed} (from {patch_input_channels_for_dft}ch, {self.enc_patch_h_for_dft}x{self.enc_patch_w_for_dft} patches)")

        self.patch_embed = PatchEmbed(patch_feature_dim_for_embed, args.encoder_initial_tangent_dim)
        self.wubu_s = FullyHyperbolicWuBuNestingModel(input_tangent_dim=args.encoder_initial_tangent_dim, output_tangent_dim=video_config['wubu_s_output_dim'], config=wubu_s_config)
        self.wubu_s_final_hyp_dim = wubu_s_config['hyperbolic_dims'][-1] if wubu_s_config['num_levels'] > 0 and wubu_s_config['hyperbolic_dims'] else 0
        self.wubu_s_final_curvature = 1.0
        if wubu_s_config['num_levels'] > 0 and self.wubu_s_final_hyp_dim > 0:
            last_level_idx = wubu_s_config['num_levels'] - 1
            try: temp_level = HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_s_final_hyp_dim, wubu_s_config, wubu_s_config['initial_curvatures'][last_level_idx]); self.wubu_s_final_curvature = temp_level.get_current_curvature_scalar(); del temp_level; current_logger.info(f"WuBu-S final C est: {self.wubu_s_final_curvature:.3f}")
            except IndexError: current_logger.error(f"Index error WuBu-S L{last_level_idx}. Default C=1.0."); self.wubu_s_final_curvature = 1.0

        self.wubu_t_input_dim = video_config['wubu_s_output_dim']
        self.wubu_m_output_dim = video_config.get('wubu_m_output_dim', 0)
        if args.use_wubu_motion_branch and self.wubu_m_output_dim > 0:
            self.wubu_t_input_dim += self.wubu_m_output_dim
            current_logger.info(f"VAE Enc: WuBu-M feats (dim {self.wubu_m_output_dim}) for WuBu-T.")
        elif args.use_wubu_motion_branch: current_logger.warning("VAE Enc: WuBu-M active but dim 0.")

        self.wubu_t_config = _configure_wubu_stack(args, "wubu_t")
        self.wubu_t: Optional[FullyHyperbolicWuBuNestingModel] = None
        if self.wubu_t_config and self.wubu_t_config['num_levels'] > 0 and self.wubu_t_input_dim > 0:
             self.wubu_t_output_dim = self.wubu_t_config['hyperbolic_dims'][-1] if self.wubu_t_config['hyperbolic_dims'] else 0
             self.wubu_t = FullyHyperbolicWuBuNestingModel(input_tangent_dim=self.wubu_t_input_dim, output_tangent_dim=self.wubu_t_output_dim, config=self.wubu_t_config)
             current_logger.info(f"VAE Enc WuBu-T: InDim {self.wubu_t_input_dim}, OutDim {self.wubu_t_output_dim}")
             self.fc_mu = nn.Linear(self.wubu_t_output_dim, self.latent_dim)
             self.fc_logvar = nn.Linear(self.wubu_t_output_dim, self.latent_dim)
        else:
             current_logger.warning("VAE Enc WuBu-T disabled. Latent from direct projection.")
             self.fc_mu = nn.Linear(self.wubu_t_input_dim, self.latent_dim)
             self.fc_logvar = nn.Linear(self.wubu_t_input_dim, self.latent_dim)
        self.apply(init_weights_general)

    def forward(self, frames_pixels: torch.Tensor, motion_features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # frames_pixels: (B, N_total_sample_frames, C, H, W)
        # motion_features: (B, N_pairs, NumMotionRegions, D_motion_out)
        # Returns: mu, logvar, gaad_bboxes_all_frames (B, N_total_sample_frames, NumReg, 4), regional_app_features (tangent or dft)
        
        B, N_frames_total_sample, C_img, H_img, W_img = frames_pixels.shape
        device = frames_pixels.device
        dtype_model = next(self.parameters()).dtype
        
        frames_pixels_flat = frames_pixels.reshape(B * N_frames_total_sample, C_img, H_img, W_img)
        gaad_bboxes_list = []
        for b_idx in range(B): # Iterate per item in original batch
            frame_bboxes_for_sequence = []
            for f_idx in range(N_frames_total_sample): # For each frame in the sequence for this batch item
                # Generate GAAD bboxes for THIS specific frame (H_img, W_img might be dynamic if model supported it)
                frame_dims = (W_img, H_img); max_w_scalar=float(W_img); max_h_scalar=float(H_img);
                if self.decomposition_type == "hybrid":
                    num_subdivide=self.num_appearance_regions//2; num_spiral=self.num_appearance_regions-num_subdivide; bboxes_for_item=[]
                    if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims,num_subdivide,device,dtype_model,self.gaad_min_size_px))
                    if num_spiral > 0:
                         spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype_model); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype_model); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); bboxes_for_item.append(spiral_bboxes_current)
                    single_frame_bboxes = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_appearance_regions, dtype=dtype_model, device=device)
                elif self.decomposition_type == "spiral": # ... (GAAD spiral)
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_appearance_regions, device, dtype_model); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(self.num_appearance_regions, 4, device=device, dtype=dtype_model); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); single_frame_bboxes = spiral_bboxes_current
                else: # subdivide
                    single_frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims,self.num_appearance_regions,device,dtype_model,self.gaad_min_size_px)
                
                # Pad/truncate bboxes for this frame
                if single_frame_bboxes.shape[0] < self.num_appearance_regions: num_to_pad=self.num_appearance_regions-single_frame_bboxes.shape[0]; padding_box=single_frame_bboxes[-1:].clone() if single_frame_bboxes.shape[0]>0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]],dtype=dtype_model,device=device); padding=padding_box.repeat(num_to_pad,1); single_frame_bboxes=torch.cat([single_frame_bboxes, padding], dim=0)
                elif single_frame_bboxes.shape[0] > self.num_appearance_regions: single_frame_bboxes=single_frame_bboxes[:self.num_appearance_regions]
                frame_bboxes_for_sequence.append(single_frame_bboxes)
            gaad_bboxes_list.append(torch.stack(frame_bboxes_for_sequence)) # (N_frames_total_sample, NumReg, 4)
        
        gaad_bboxes_full_batch_sequences = torch.stack(gaad_bboxes_list) # (B, N_frames_total_sample, NumReg, 4)
        gaad_bboxes_flat_for_patch_extractor = gaad_bboxes_full_batch_sequences.reshape(B * N_frames_total_sample, self.num_appearance_regions, 4)

        # extracted_patches: (B*N_frames, NumRegions, C_patch_in, H_patch, W_patch)
        extracted_patches = self.patch_extractor(frames_pixels_flat, gaad_bboxes_flat_for_patch_extractor)
        B_flat_frames, NumReg_patch, C_patch_in, H_patch, W_patch = extracted_patches.shape

        if self.args.use_dft_features_appearance:
            # Reshape for DFT: ( (B*N_frames)*NumReg, C_patch_in, H_patch, W_patch )
            patches_for_dft_flat = extracted_patches.reshape(B_flat_frames * NumReg_patch, C_patch_in, H_patch, W_patch)
            # dft_features_flat: ( (B*N_frames)*NumReg, D_dft_features )
            dft_features_flat = DFTUtils.compute_2d_dft_features(
                patches_for_dft_flat,
                norm_scale=self.args.dft_norm_scale_video,
                out_feature_dim_first=True
            )
            # Pass to PatchEmbed: ( (B*N_frames)*NumReg, D_dft_features ) -> ( (B*N_frames)*NumReg, encoder_initial_tangent_dim )
            initial_tangent_vectors_flat_regions = self.patch_embed(dft_features_flat)
            # Store DFT features if needed by trainer for recon loss (this is pre-WuBuS)
            # For recon loss, we'd need DFT features for *target* frames, using bboxes from generator
            # So, `regional_app_features_tangent` will be the output of WuBu-S for the VAE path,
            # and the trainer will recompute target DFT features.
            # Let's store the dft_features_flat for now if needed for debugging or direct target later.
            # The generator will produce DFT features *after* its WuBu-G equivalent.
            # So, for VAE, this `dft_features_flat` *is* the representation we want to reconstruct if loss is in DFT domain.
            # But it's for all input frames.
            regional_features_for_wubu_s_input = initial_tangent_vectors_flat_regions # These are tangent vectors for WuBu-S
            # Keep a reference to the raw DFT features from input patches for potential direct use as recon target
            # Reshape to (B, N_frames_total_sample, NumReg_patch, D_dft_features)
            raw_dft_features_from_input = dft_features_flat.view(B, N_frames_total_sample, NumReg_patch, -1)

        else: # Original non-DFT path
            patches_for_embed = extracted_patches.reshape(B_flat_frames * NumReg_patch, -1)
            initial_tangent_vectors_flat_regions = self.patch_embed(patches_for_embed)
            regional_features_for_wubu_s_input = initial_tangent_vectors_flat_regions
            raw_dft_features_from_input = None # No DFT features to store

        wubu_s_output_tangent_flat = self.wubu_s(regional_features_for_wubu_s_input)
        D_out_s = wubu_s_output_tangent_flat.shape[-1]
        # regional_app_features_tangent: (B, N_frames_total_sample, NumReg_patch, D_out_s)
        regional_app_features_tangent = wubu_s_output_tangent_flat.reshape(B, N_frames_total_sample, NumReg_patch, D_out_s)

        agg_app_features = torch.mean(regional_app_features_tangent, dim=2) # (B, N_frames_total_sample, D_out_s)

        wubu_t_input_features = agg_app_features
        if motion_features is not None and self.args.use_wubu_motion_branch:
             motion_features_tangent = motion_features.to(dtype_model) # Assume motion_encoder already outputs tangent-like features
             N_pairs_motion = motion_features_tangent.shape[1] # (B, N_pairs, NumReg_motion, D_out_m)
             agg_motion_features_per_pair = torch.mean(motion_features_tangent, dim=2) # (B, N_pairs, D_out_m)
             
             # Align N_pairs with N_frames_total_sample for WuBu-T input
             # WuBu-T expects features for each of the N_frames_total_sample.
             # Motion features are for pairs. We need to map N_pairs features to N_frames_total_sample features.
             # Simplest: use motion[t] for frame[t+1]. Frame 0 has no preceding motion.
             # Or, average motion around frame t. For now, let's try a simple forward fill.
             aligned_motion_for_wubut = torch.zeros(B, N_frames_total_sample, agg_motion_features_per_pair.shape[-1], device=device, dtype=dtype_model)
             if N_pairs_motion > 0:
                 # Fill from frame 1 onwards
                 len_to_copy = min(N_pairs_motion, N_frames_total_sample -1)
                 aligned_motion_for_wubut[:, 1 : 1 + len_to_copy, :] = agg_motion_features_per_pair[:, :len_to_copy, :]
                 # If more motion pairs than needed, they are ignored.
                 # If fewer, later frames in aligned_motion_for_wubut will have zero motion features for those time steps.
                 # Or forward fill last valid motion:
                 if N_pairs_motion < N_frames_total_sample -1 and N_pairs_motion > 0:
                      last_valid_motion = agg_motion_features_per_pair[:, -1, :].unsqueeze(1) # (B, 1, D_out_m)
                      aligned_motion_for_wubut[:, 1 + N_pairs_motion :, :] = last_valid_motion.expand(-1, N_frames_total_sample - (1+N_pairs_motion), -1)


             wubu_t_input_features = torch.cat([agg_app_features, aligned_motion_for_wubut], dim=-1)

        if self.wubu_t:
            temporal_features_sequence = self.wubu_t(wubu_t_input_features) # (B, N_frames_total_sample, wubu_t_output_dim)
            final_temporal_feature = temporal_features_sequence[:, -1, :] # Take last feature from total sample sequence
        else:
            final_temporal_feature = torch.mean(wubu_t_input_features, dim=1)

        mu = self.fc_mu(final_temporal_feature)
        logvar = self.fc_logvar(final_temporal_feature)
        
        # Return `gaad_bboxes_full_batch_sequences` for the trainer to use.
        # Return `raw_dft_features_from_input` if DFT used, else None. This is for recon target.
        return mu, logvar, gaad_bboxes_full_batch_sequences, raw_dft_features_from_input


# --- FiLM Layer (Helper for GAAD modulation) --- (No changes)
class FiLMLayer(nn.Module): # ... (No changes from original, keep as is)
    def __init__(self, channels: int, condition_dim: int):
        super().__init__(); self.channels = channels; self.condition_dim = condition_dim; self.to_gamma_beta = nn.Linear(condition_dim, channels * 2)
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(condition); gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        if x.dim() == 5: gamma = gamma.view(-1, self.channels, 1, 1, 1); beta = beta.view(-1, self.channels, 1, 1, 1)
        elif x.dim() == 4: gamma = gamma.view(-1, self.channels, 1, 1); beta = beta.view(-1, self.channels, 1, 1)
        else: raise ValueError(f"FiLMLayer input x has unsupported dimension: {x.dim()}")
        return (1 + gamma) * x + beta

class RegionalGeneratorDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, latent_dim: int): # wubu_s_output_dim removed
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.image_size = (args.image_h, args.image_w)
        self.num_regions = gaad_config['num_regions']
        self.num_img_channels = video_config['num_channels']
        self.latent_dim = latent_dim
        self.num_predict_frames = video_config["num_predict_frames"]
        self.logger = logging.getLogger("WuBuGAADHybridGenV01DFT.Generator")

        min_target_dim = min(self.image_size[0], self.image_size[1])
        if min_target_dim <= 8: self.gen_init_spatial_res = 1; self.gen_num_upsampling_layers = int(math.log2(min_target_dim)) if min_target_dim > 0 and math.log2(min_target_dim).is_integer() else max(1, int(math.ceil(math.log2(min_target_dim))))
        elif min_target_dim <= 32: self.gen_init_spatial_res = 2; self.gen_num_upsampling_layers = int(math.log2(min_target_dim / 2)) if (min_target_dim/2)>0 and math.log2(min_target_dim/2).is_integer() else max(1, int(math.ceil(math.log2(min_target_dim/2))))
        else: self.gen_init_spatial_res = 4; self.gen_num_upsampling_layers = int(math.log2(min_target_dim / 4)) if (min_target_dim/4)>0 and math.log2(min_target_dim/4).is_integer() else max(1, int(math.ceil(math.log2(min_target_dim/4))))
        
        calculated_final_res = self.gen_init_spatial_res * (2**self.gen_num_upsampling_layers)
        if calculated_final_res != min_target_dim: self.logger.warning(f"Gen calculated final res {calculated_final_res} vs target {min_target_dim}.")

        self.gen_init_channels = min(512, max(128, self.latent_dim * 2))
        self.gen_temporal_kernel_size = getattr(args, 'gen_temporal_kernel_size', 3)

        self.fc_expand_latent = nn.Linear(self.latent_dim, self.gen_init_channels * self.num_predict_frames * self.gen_init_spatial_res * self.gen_init_spatial_res)

        self.gaad_condition_dim = max(32, self.latent_dim // 4)
        if self.num_regions > 0 and getattr(args, 'gen_use_gaad_film_condition', True): # Added arg check
            self.bbox_feature_dim = 4
            hidden_bbox_embed_dim = max(self.gaad_condition_dim, self.num_regions * self.bbox_feature_dim // 2)
            self.frame_gaad_embedder = nn.Sequential(nn.Linear(self.num_regions * self.bbox_feature_dim, hidden_bbox_embed_dim), nn.GELU(), nn.Linear(hidden_bbox_embed_dim, self.gaad_condition_dim))
            self.logger.info(f"Generator GAAD-FiLM enabled, cond_dim: {self.gaad_condition_dim}")
        else:
            self.frame_gaad_embedder = None
            self.logger.info("Generator GAAD-FiLM disabled.")

        self.upsample_blocks = nn.ModuleList()
        current_channels = self.gen_init_channels
        padding_temp = self.gen_temporal_kernel_size // 2
        min_gen_channels_final_block = max(32, self.num_img_channels * 8) # Channels before regional head

        for i in range(self.gen_num_upsampling_layers):
            out_channels = max(min_gen_channels_final_block, current_channels // 2) if i < self.gen_num_upsampling_layers -1 else min_gen_channels_final_block
            block = nn.ModuleDict(); block['conv_transpose'] = nn.ConvTranspose3d(current_channels, out_channels, kernel_size=(self.gen_temporal_kernel_size, 4, 4), stride=(1, 2, 2), padding=(padding_temp, 1, 1), bias=False)
            block['norm'] = nn.InstanceNorm3d(out_channels, affine=True);
            if self.frame_gaad_embedder is not None: block['film'] = FiLMLayer(out_channels, self.gaad_condition_dim)
            block['activation'] = nn.GELU(); self.upsample_blocks.append(block); current_channels = out_channels
        
        # current_channels is now C_feat for the dense feature volume.
        # This volume has shape (B, C_feat, N_pred_frames, H_final_feat, W_final_feat)
        # H_final_feat, W_final_feat are results of upsampling gen_init_spatial_res
        self.final_dense_feature_channels = current_channels

        if args.use_dft_features_appearance: # DFT Path for Generator Head
            self.gen_patch_h_for_dft = args.dft_patch_size_h
            self.gen_patch_w_for_dft = args.dft_patch_size_w
            self.dft_w_coeffs_one_sided_gen = self.gen_patch_w_for_dft // 2 + 1
            
            # RoIAlign output size for extracting regional features from dense map
            # This should match the patch size for which DFT coeffs are predicted.
            self.gen_roi_align_output_spatial = (self.gen_patch_h_for_dft, self.gen_patch_w_for_dft)
            
            # MLP to project RoIAlign'd features to DFT coefficients
            # Input dim: C_feat * H_patch_dft * W_patch_dft
            roi_feat_dim = self.final_dense_feature_channels * self.gen_patch_h_for_dft * self.gen_patch_w_for_dft
            # Output dim: num_img_channels * 2 (real/imag) * H_patch_dft * (W_patch_dft//2+1)
            dft_coeff_output_dim_per_region = self.num_img_channels * 2 * self.gen_patch_h_for_dft * self.dft_w_coeffs_one_sided_gen
            
            # Using a small MLP for this projection
            hidden_mlp_dft = max(dft_coeff_output_dim_per_region, roi_feat_dim // 2)
            self.to_dft_coeffs_mlp = nn.Sequential(
                nn.Linear(roi_feat_dim, hidden_mlp_dft),
                nn.GELU(),
                nn.Linear(hidden_mlp_dft, dft_coeff_output_dim_per_region)
                # No final activation here, DFT coeffs can be positive/negative
            )
            self.logger.info(f"Generator DFT Head: RoIAlign spatial output {self.gen_roi_align_output_spatial}, Projects from {roi_feat_dim} to {dft_coeff_output_dim_per_region} DFT coeffs per region.")
        else: # Original Pixel Output Path
            final_conv_padding_spatial = 1 if getattr(args, 'gen_final_conv_kernel_spatial', 3) > 1 else 0
            self.final_conv_pixel = nn.Conv3d(current_channels, self.num_img_channels, kernel_size=(self.gen_temporal_kernel_size, getattr(args, 'gen_final_conv_kernel_spatial', 3), getattr(args, 'gen_final_conv_kernel_spatial', 3)), padding=(padding_temp, final_conv_padding_spatial, final_conv_padding_spatial))
            self.final_activation_pixel = nn.Tanh()
            self.logger.info(f"Generator Pixel Head: FinalConv output channels {self.num_img_channels}")

        self.apply(init_weights_general)

    def _normalize_bboxes(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x1, y1, x2, y2 = bboxes.unbind(-1); img_W = float(W) if W > 0 else 1.0; img_H = float(H) if H > 0 else 1.0
        norm_cx = ((x1 + x2) / 2.0) / img_W; norm_cy = ((y1 + y2) / 2.0) / img_H
        norm_w = (x2 - x1) / img_W; norm_h = (y2 - y1) / img_H # If y from top, y2-y1 is correct
        return torch.stack([norm_cx, norm_cy, norm_w, norm_h], dim=-1)

    def forward(self, latent_code: torch.Tensor, gaad_bboxes_for_decode: Optional[torch.Tensor]) -> torch.Tensor:
        # gaad_bboxes_for_decode: (B, num_predict_frames, NumRegions, 4)
        # Output:
        #   If DFT: (B, num_predict_frames, NumRegions, num_img_channels, 2, H_dft_patch, W_dft_coeffs)
        #   If Pixel: (B, num_predict_frames, num_img_channels, H_img, W_img)
        B = latent_code.shape[0]; device = latent_code.device; dtype_in = latent_code.dtype

        x = self.fc_expand_latent(latent_code)
        x = x.view(B, self.gen_init_channels, self.num_predict_frames, self.gen_init_spatial_res, self.gen_init_spatial_res).to(dtype_in)

        sequence_condition = None
        if self.frame_gaad_embedder is not None:
            if gaad_bboxes_for_decode is not None and \
               (gaad_bboxes_for_decode.shape[0] != B or \
                gaad_bboxes_for_decode.shape[1] != self.num_predict_frames or \
                gaad_bboxes_for_decode.shape[2] != self.num_regions):
                self.logger.warning(f"Gen GAAD bbox shape mismatch. Expected (B={B}, N_pred={self.num_predict_frames}, NumReg={self.num_regions}, 4), got {gaad_bboxes_for_decode.shape}. Zero cond.")
                frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)
            elif gaad_bboxes_for_decode is not None:
                norm_bboxes = self._normalize_bboxes(gaad_bboxes_for_decode.to(dtype_in), self.image_size[0], self.image_size[1])
                norm_bboxes_flat = norm_bboxes.view(B * self.num_predict_frames, -1)
                frame_conditions_flat = self.frame_gaad_embedder(norm_bboxes_flat)
            else: frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)
            if frame_conditions_flat is not None:
                frame_conditions_reshaped = frame_conditions_flat.view(B, self.num_predict_frames, self.gaad_condition_dim)
                sequence_condition = torch.mean(frame_conditions_reshaped, dim=1).to(dtype_in)

        for block_idx, block in enumerate(self.upsample_blocks):
            x = block['conv_transpose'](x); x = block['norm'](x)
            if 'film' in block and sequence_condition is not None: x = block['film'](x, sequence_condition)
            x = block['activation'](x)
        # x is now dense feature volume: (B, C_feat, N_pred_frames, H_final_feat, W_final_feat)
        
        if self.args.use_dft_features_appearance: # DFT Output Path
            if gaad_bboxes_for_decode is None:
                # This should ideally not happen if DFT is on, as bboxes are needed for regional DFTs
                self.logger.error("Generator DFT path: gaad_bboxes_for_decode is None. Cannot produce regional DFTs.")
                # Fallback: create dummy zero DFT coeffs
                return torch.zeros(B, self.num_predict_frames, self.num_regions,
                                   self.num_img_channels, 2, self.gen_patch_h_for_dft, self.dft_w_coeffs_one_sided_gen,
                                   device=device, dtype=dtype_in)

            all_frames_regional_dft_coeffs = []
            C_feat_dense = x.shape[1]
            # Spatial scale factor from dense feature map to original image coords for bboxes
            # H_final_feat is spatial height of x. self.image_size[0] is target image H.
            # spatial_scale_for_roi = H_final_feat / self.image_size[0] # Incorrect. Bboxes are in original image coords.
            # RoIAlign's spatial_scale arg maps bbox coords from input image to feature map.
            # So, if bboxes are in original image coords (0, H_img), and feature map is (0, H_feat),
            # scale = H_feat / H_img.
            # x shape: (B, C_feat, D_frames, H_feat, W_feat)
            H_final_feat = x.shape[-2]
            W_final_feat = x.shape[-1]
            
            # The bboxes passed (gaad_bboxes_for_decode) are in original image coordinates.
            # We need to scale them for RoIAlign on the feature map `x`.
            # The `spatial_scale` for RoIAlign should be `feat_map_dim / original_img_dim`.
            # Since roi_align operates on 2D feature maps (per frame), we extract frame by frame.

            for f_idx in range(self.num_predict_frames):
                frame_feature_map = x[:, :, f_idx, :, :] # (B, C_feat, H_final_feat, W_final_feat)
                frame_bboxes = gaad_bboxes_for_decode[:, f_idx, :, :] # (B, NumRegions, 4)
                
                # Prepare bboxes for roi_align: needs batch_idx prepended
                rois_for_frame_list = []
                for b_s_idx in range(B): # iterate through samples in batch
                    # Bboxes are in original image space (H_img, W_img)
                    # We need to scale them for the feature map (H_final_feat, W_final_feat)
                    # This is incorrect. RoIAlign takes spatial_scale parameter. BBoxes should be in original coord.
                    # scaled_bboxes_for_roialign = frame_bboxes[b_s_idx] * spatial_scale_factor_h_w (This is not how roi_align works)
                    
                    batch_indices = torch.full((self.num_regions, 1), float(b_s_idx), device=device, dtype=dtype_in)
                    rois_for_frame_list.append(torch.cat([batch_indices, frame_bboxes[b_s_idx]], dim=1))
                
                all_rois_this_frame = torch.cat(rois_for_frame_list, dim=0) # (B*NumRegions, 5)
                
                # RoIAlign on the frame's feature map
                # spatial_scale for roi_align is feat_map_dim / input_img_dim
                # Here, input_img_dim is self.image_size[0] or [1]
                # feat_map_dim is H_final_feat or W_final_feat
                # Since bboxes are for H_img, W_img, and roi_align takes one scale,
                # This assumes isotropic scaling or feature map aspect ratio matches image.
                # Let's use an effective scale if feature map might be non-square relative to image
                # For simplicity, assume roi_align handles aspect if output_size is square.
                # The default behavior is that bboxes are in image pixel coordinates.
                # spatial_scale converts these to feature map coordinates.
                # spatial_scale = feature_map_size / input_image_size
                # Using H for scale. This might need adjustment if aspect ratios differ wildly.
                roi_spatial_scale = H_final_feat / self.image_size[0]

                regional_feats_from_roi = roi_align(
                    frame_feature_map,              # (B, C_feat, H_final_feat, W_final_feat)
                    all_rois_this_frame,            # (B*NumRegions, 5)
                    output_size=self.gen_roi_align_output_spatial, # (H_dft_patch, W_dft_patch)
                    spatial_scale=roi_spatial_scale,
                    aligned=True
                )
                # regional_feats_from_roi: (B*NumRegions, C_feat, H_dft_patch, W_dft_patch)
                
                # Flatten and project to DFT coeffs
                regional_feats_flat = regional_feats_from_roi.reshape(B * self.num_regions, -1)
                dft_coeffs_flat_for_frame = self.to_dft_coeffs_mlp(regional_feats_flat)
                # dft_coeffs_flat_for_frame: (B*NumRegions, num_img_channels * 2 * H_dft * W_dft_coeffs)
                
                # Reshape to structured DFT coeffs
                frame_dft_structured = dft_coeffs_flat_for_frame.view(
                    B, self.num_regions, self.num_img_channels, 2,
                    self.gen_patch_h_for_dft, self.dft_w_coeffs_one_sided_gen
                )
                all_frames_regional_dft_coeffs.append(frame_dft_structured)
            
            # Stack along frame dimension
            output_dft_coeffs = torch.stack(all_frames_regional_dft_coeffs, dim=1)
            # Final shape: (B, N_pred_frames, NumRegions, NumImgChannels, 2, H_dft, W_dft_coeffs)
            return output_dft_coeffs.to(dtype_in)

        else: # Original Pixel Output Path
            x = self.final_conv_pixel(x)
            generated_frames_sequence_pixels = self.final_activation_pixel(x)
            generated_frames_sequence_pixels = generated_frames_sequence_pixels.permute(0, 2, 1, 3, 4) # (B, N_pred, C, H, W)

            final_h_actual, final_w_actual = generated_frames_sequence_pixels.shape[-2:]
            if final_h_actual != self.image_size[0] or final_w_actual != self.image_size[1]:
                self.logger.debug(f"Gen pixel output {final_h_actual}x{final_w_actual} vs target {self.image_size}. Adaptive pool.")
                temp_permuted_for_pool = generated_frames_sequence_pixels.permute(0, 2, 1, 3, 4)
                pooled = F.adaptive_avg_pool3d(temp_permuted_for_pool, (self.num_predict_frames, self.image_size[0], self.image_size[1]))
                generated_frames_sequence_pixels = pooled.permute(0, 2, 1, 3, 4)
            return generated_frames_sequence_pixels.to(dtype_in)


class RegionalDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_config = gaad_config
        self.disc_config = disc_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV01DFT.Discriminator")

        self.image_size = (args.image_h, args.image_w)
        self.num_channels = video_config['num_channels']
        self.num_frames_to_discriminate = video_config.get("num_predict_frames", 1)
        self.num_frames_to_discriminate = max(1, self.num_frames_to_discriminate)
        self.num_regions = self.gaad_config.get('num_regions', 0)
        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", getattr(args, 'disc_apply_spectral_norm', True))
        self.use_gaad_film_condition = disc_config.get("use_gaad_film_condition", getattr(args, 'disc_use_gaad_film_condition', False)) and self.num_regions > 0

        if self.use_gaad_film_condition:
            self.gaad_condition_dim = disc_config.get("gaad_condition_dim_disc", getattr(args, 'disc_gaad_condition_dim_disc', 64))
            self.bbox_feature_dim = 4 # (cx, cy, w, h) normalized
            hidden_bbox_embed_dim = max(self.gaad_condition_dim, self.num_regions * self.bbox_feature_dim // 2)
            self.frame_gaad_embedder_disc = nn.Sequential(
                nn.Linear(self.num_regions * self.bbox_feature_dim, hidden_bbox_embed_dim),
                nn.GELU(),
                nn.Linear(hidden_bbox_embed_dim, self.gaad_condition_dim)
            )
            self.logger.info(f"Disc GAAD-FiLM ENABLED. Cond Dim: {self.gaad_condition_dim}")
        else:
            self.frame_gaad_embedder_disc = None
            self.gaad_condition_dim = 0 # Effectively no FiLM conditioning
            self.logger.info("Disc GAAD-FiLM DISABLED.")

        self.disc_type = self.disc_config.get("type", getattr(args, 'discriminator_type', "spatio_temporal_cnn"))
        self.logger.info(f"Init Disc Type: {self.disc_type}")

        if self.disc_type == "spatio_temporal_cnn":
            min_input_dim = min(self.image_size[0], self.image_size[1])
            num_spatial_downsamples_target = int(math.log2(min_input_dim / 4)) if min_input_dim >= 8 else 1
            max_possible_downsamples = int(math.log2(min_input_dim)) if min_input_dim > 0 else 0
            num_spatial_downsamples = max(1, min(num_spatial_downsamples_target, max_possible_downsamples))

            base_disc_channels = disc_config.get("base_disc_channels", getattr(args, 'disc_base_disc_channels', 64))
            cnn3d_channels_list = [base_disc_channels * (2**i) for i in range(num_spatial_downsamples)]
            max_disc_channels = disc_config.get("max_disc_channels", getattr(args, 'disc_max_disc_channels', 512))
            cnn3d_channels_list = [min(c, max_disc_channels) for c in cnn3d_channels_list]
            cnn3d_channels_list = [base_disc_channels] if not cnn3d_channels_list else cnn3d_channels_list

            temporal_kernel_size = disc_config.get("temporal_kernel_size", getattr(args, 'disc_temporal_kernel_size', 3))
            default_temporal_stride = 1
            layers = []
            in_c = self.num_channels
            current_d_dim = self.num_frames_to_discriminate
            current_h_dim = self.image_size[0]
            current_w_dim = self.image_size[1]

            for i, out_c in enumerate(cnn3d_channels_list):
                can_halve_spatial = current_h_dim >= 8 and current_w_dim >= 8
                spatial_stride = 2 if can_halve_spatial and i < num_spatial_downsamples else 1
                
                apply_temporal_stride_val = 2
                can_stride_temporally = current_d_dim > temporal_kernel_size and current_d_dim >= apply_temporal_stride_val
                actual_temporal_stride = apply_temporal_stride_val if can_stride_temporally and i == 0 else default_temporal_stride # Stride temporally only in first layer if possible
                
                current_t_kernel = min(temporal_kernel_size, current_d_dim) if current_d_dim > 1 else 1
                current_t_padding = current_t_kernel // 2 if current_t_kernel > 1 else 0

                block = nn.ModuleDict()
                conv_layer = nn.Conv3d(in_c, out_c, kernel_size=(current_t_kernel, 4, 4), stride=(actual_temporal_stride, spatial_stride, spatial_stride), padding=(current_t_padding, 1, 1), bias=False)
                block['conv'] = spectral_norm(conv_layer) if self.apply_spectral_norm else conv_layer
                if self.apply_spectral_norm and i == 0: self.logger.info("Applying Spectral Norm to Disc Conv3D.")
                
                block['norm'] = nn.InstanceNorm3d(out_c, affine=not self.use_gaad_film_condition) # If FiLM is used, InstanceNorm's affine should be False or FiLM handles scale/shift
                if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None:
                    block['film'] = FiLMLayer(out_c, self.gaad_condition_dim)
                else:
                    block['film'] = None
                block['activation'] = nn.LeakyReLU(0.2, inplace=True)
                layers.append(block)
                in_c = out_c

                # Update effective dimensions after this layer
                if current_d_dim > 0: current_d_dim = (current_d_dim + 2 * current_t_padding - (current_t_kernel -1) -1 ) // actual_temporal_stride + 1
                if current_h_dim > 0: current_h_dim = (current_h_dim + 2 * 1 - (4-1) -1 ) // spatial_stride + 1 # Assuming padding 1, kernel 4 for spatial
                if current_w_dim > 0: current_w_dim = (current_w_dim + 2 * 1 - (4-1) -1 ) // spatial_stride + 1
                current_d_dim = max(1, current_d_dim); current_h_dim = max(1, current_h_dim); current_w_dim = max(1, current_w_dim)

            self.feature_extractor_blocks = nn.ModuleList(layers)

            # Calculate output shape for FC layers
            _device_for_shape_calc = torch.device('cpu')
            if len(list(self.parameters())) > 0:
                _device_for_shape_calc = next(self.parameters()).device
            if hasattr(self.args, 'device') and self.args.device == 'cuda' and not torch.cuda.is_available():
                 _device_for_shape_calc = torch.device('cpu') # Fallback if CUDA specified but not available
            
            test_input = torch.randn(1, self.num_channels, self.num_frames_to_discriminate, self.image_size[0], self.image_size[1]).to(_device_for_shape_calc)
            dummy_sequence_condition_disc = None
            original_embedder_device = None
            if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None:
                if len(list(self.frame_gaad_embedder_disc.parameters())) > 0:
                    original_embedder_device = next(self.frame_gaad_embedder_disc.parameters()).device
                    if original_embedder_device != _device_for_shape_calc: self.frame_gaad_embedder_disc.to(_device_for_shape_calc)
                dummy_bboxes_norm = torch.rand(1, self.num_frames_to_discriminate, self.num_regions, self.bbox_feature_dim).to(_device_for_shape_calc)
                norm_bboxes_flat_dummy = dummy_bboxes_norm.view(1 * self.num_frames_to_discriminate, -1)
                frame_cond_flat_dummy = self.frame_gaad_embedder_disc(norm_bboxes_flat_dummy)
                frame_cond_reshaped_dummy = frame_cond_flat_dummy.view(1, self.num_frames_to_discriminate, self.gaad_condition_dim)
                dummy_sequence_condition_disc = torch.mean(frame_cond_reshaped_dummy, dim=1)
                if original_embedder_device and original_embedder_device != _device_for_shape_calc:
                    self.frame_gaad_embedder_disc.to(original_embedder_device)

            temp_features = test_input
            original_fe_devices = {}
            for i_fe, block_module_item_fe in enumerate(self.feature_extractor_blocks):
                if len(list(block_module_item_fe.parameters())) > 0:
                    original_fe_devices[i_fe] = next(block_module_item_fe.parameters()).device
                    if original_fe_devices[i_fe] != _device_for_shape_calc: block_module_item_fe.to(_device_for_shape_calc)
            
            with torch.no_grad():
                for block_module_item_fe in self.feature_extractor_blocks:
                    temp_features = block_module_item_fe['conv'](temp_features)
                    temp_features = block_module_item_fe['norm'](temp_features)
                    if 'film' in block_module_item_fe and block_module_item_fe['film'] is not None and dummy_sequence_condition_disc is not None:
                        temp_features = block_module_item_fe['film'](temp_features, dummy_sequence_condition_disc)
                    temp_features = block_module_item_fe['activation'](temp_features)
            
            final_feature_map_shape_pre_pool = temp_features.shape
            self.adaptive_pool = nn.AdaptiveAvgPool3d((max(1, final_feature_map_shape_pre_pool[2]), 1, 1)) # Pool spatially, keep temporal if > 1
            self.adaptive_pool.to(_device_for_shape_calc) # Ensure pool layer is on correct device for test
            
            with torch.no_grad():
                pooled_features_test = self.adaptive_pool(temp_features.to(_device_for_shape_calc)) # Ensure temp_features is on device
            
            # Restore original devices for feature extractor blocks
            for i_fe, block_module_item_fe in enumerate(self.feature_extractor_blocks):
                if i_fe in original_fe_devices and original_fe_devices[i_fe] and original_fe_devices[i_fe] != _device_for_shape_calc:
                    block_module_item_fe.to(original_fe_devices[i_fe])

            final_flattened_dim = max(1, pooled_features_test.numel() // pooled_features_test.shape[0]) # B should be 1 here
            min_hidden_fc_dim = getattr(args, 'disc_min_hidden_fc_dim', 128)
            max_hidden_fc_dim = getattr(args, 'disc_max_hidden_fc_dim', 512)

            if final_flattened_dim <= min_hidden_fc_dim * 1.5 and final_flattened_dim > 0 :
                fc_layer = nn.Linear(final_flattened_dim, 1)
                self.final_fc_layers = spectral_norm(fc_layer) if self.apply_spectral_norm else fc_layer
                self.logger.info(f"Disc final FC: Direct {final_flattened_dim} to 1.")
            elif final_flattened_dim > 0 :
                hidden_fc_dim = min(max(min_hidden_fc_dim, final_flattened_dim // 2), max_hidden_fc_dim)
                fc1 = nn.Linear(final_flattened_dim, hidden_fc_dim)
                fc2 = nn.Linear(hidden_fc_dim, 1)
                if self.apply_spectral_norm:
                    self.final_fc_layers = nn.Sequential(spectral_norm(fc1), nn.LeakyReLU(0.2, inplace=True), spectral_norm(fc2))
                else:
                    self.final_fc_layers = nn.Sequential(fc1, nn.LeakyReLU(0.2, inplace=True), fc2)
                self.logger.info(f"Disc final FC: {final_flattened_dim} -> {hidden_fc_dim} -> 1.")
            else:
                self.logger.error(f"Disc final_flattened_dim invalid: {final_flattened_dim}. Defaulting small FC.")
                self.final_fc_layers = nn.Linear(1,1) # Should not happen
            
            self.logger.info(f"SpatioTemporalCNN Disc: Frames={self.num_frames_to_discriminate}, CNN3D_Ch={cnn3d_channels_list}, FinalFeatMap(prepool)={final_feature_map_shape_pre_pool}, PooledFeatMap={pooled_features_test.shape}, FlattenedDimForFC={final_flattened_dim}, FiLM={self.use_gaad_film_condition}, SN={self.apply_spectral_norm}")

        elif self.disc_type == "regional_cnn":
            self.logger.warning("Using 'regional_cnn' discriminator (operates on first frame of sequence only).")
            self.patch_size = disc_config.get("patch_size", getattr(args, 'disc_patch_size', 16))
            self.resize_transform_regional = T.Resize((self.patch_size, self.patch_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True) if T is not None else None
            
            cnn_channels_2d = disc_config.get("cnn_channels_2d", getattr(args, 'disc_cnn_channels_2d', [64, 128, 256]))
            layers_2d = []
            in_c_2d = self.num_channels
            for i_2d, out_c_2d in enumerate(cnn_channels_2d):
                conv2d_layer = nn.Conv2d(in_c_2d, out_c_2d, kernel_size=4, stride=2, padding=1, bias=False)
                layers_2d.append(spectral_norm(conv2d_layer) if self.apply_spectral_norm else conv2d_layer)
                if self.apply_spectral_norm and i_2d == 0: self.logger.info("Applying SN to RegionalDisc Conv2D.")
                layers_2d.extend([nn.InstanceNorm2d(out_c_2d, affine=True), nn.LeakyReLU(0.2, inplace=True)])
                in_c_2d = out_c_2d
            
            _temp_regional_feature_extractor_2d = nn.Sequential(*layers_2d)
            _dev_calc_reg = torch.device('cpu')
            if len(list(self.parameters())) > 0: _dev_calc_reg = next(self.parameters()).device
            if hasattr(self.args, 'device') and self.args.device == 'cuda' and not torch.cuda.is_available(): _dev_calc_reg = torch.device('cpu')

            test_input_2d = torch.randn(1, self.num_channels, self.patch_size, self.patch_size).to(_dev_calc_reg)
            orig_reg_fe_dev = None
            if len(list(_temp_regional_feature_extractor_2d.parameters())) > 0:
                orig_reg_fe_dev = next(_temp_regional_feature_extractor_2d.parameters()).device
                if orig_reg_fe_dev != _dev_calc_reg: _temp_regional_feature_extractor_2d.to(_dev_calc_reg)
            
            with torch.no_grad():
                dummy_out_reg_fe = _temp_regional_feature_extractor_2d(test_input_2d)
                h_f, w_f = dummy_out_reg_fe.shape[-2:]
            
            if orig_reg_fe_dev and orig_reg_fe_dev != _dev_calc_reg:
                _temp_regional_feature_extractor_2d.to(orig_reg_fe_dev)

            self.regional_feature_extractor_2d = _temp_regional_feature_extractor_2d
            final_feature_dim_per_region = max(1, cnn_channels_2d[-1] * h_f * w_f if cnn_channels_2d else 0)
            fc_regional = nn.Linear(final_feature_dim_per_region, 1)
            self.final_fc_layers_regional = spectral_norm(fc_regional) if self.apply_spectral_norm else fc_regional
            
            self.gaad_min_size_px_regional = self.gaad_config.get('min_size_px', 5)
            self.decomposition_type_regional = self.gaad_config.get('decomposition_type', "hybrid")
        else:
            raise ValueError(f"Unsupported discriminator type: '{self.disc_type}'")
        
        self.apply(init_weights_general)

    def _normalize_bboxes_disc(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # Input bboxes: (..., 4) with [x1, y1, x2, y2]
        # Output: (..., 4) with [norm_cx, norm_cy, norm_w, norm_h]
        x1, y1, x2, y2 = bboxes.unbind(-1)
        img_W = float(W) if W > 0 else 1.0
        img_H = float(H) if H > 0 else 1.0
        
        norm_cx = ((x1 + x2) / 2.0) / img_W
        norm_cy = ((y1 + y2) / 2.0) / img_H
        norm_w = (x2 - x1).abs() / img_W # Ensure width is positive
        norm_h = (y2 - y1).abs() / img_H # Ensure height is positive
        
        return torch.stack([norm_cx, norm_cy, norm_w, norm_h], dim=-1)

    def forward(self, frames_pixels: torch.Tensor, gaad_bboxes_for_disc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_seq, C, H, W = frames_pixels.shape
        device = frames_pixels.device
        dtype_in = frames_pixels.dtype

        # Ensure correct number of frames for discriminator input
        if N_seq < self.num_frames_to_discriminate:
            padding_needed = self.num_frames_to_discriminate - N_seq
            last_frame_repeated = frames_pixels[:, -1:, ...].repeat(1, padding_needed, 1, 1, 1)
            frames_to_process = torch.cat([frames_pixels, last_frame_repeated], dim=1)
        elif N_seq > self.num_frames_to_discriminate:
            frames_to_process = frames_pixels[:, :self.num_frames_to_discriminate, ...]
        else:
            frames_to_process = frames_pixels

        sequence_condition_disc = None
        if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None:
            if gaad_bboxes_for_disc is not None:
                N_bboxes_seq = gaad_bboxes_for_disc.shape[1]
                bboxes_to_process_for_film = gaad_bboxes_for_disc

                if N_bboxes_seq != self.num_frames_to_discriminate:
                    if N_bboxes_seq < self.num_frames_to_discriminate:
                        bbox_padding = self.num_frames_to_discriminate - N_bboxes_seq
                        last_bbox_set_repeated = gaad_bboxes_for_disc[:, -1:, ...].repeat(1, bbox_padding, 1, 1)
                        bboxes_to_process_for_film = torch.cat([gaad_bboxes_for_disc, last_bbox_set_repeated], dim=1)
                    else: # N_bboxes_seq > self.num_frames_to_discriminate
                        bboxes_to_process_for_film = gaad_bboxes_for_disc[:, :self.num_frames_to_discriminate, ...]
                
                # Validate shape before proceeding
                if (bboxes_to_process_for_film.shape[0] != B or
                    bboxes_to_process_for_film.shape[1] != self.num_frames_to_discriminate or
                    (self.num_regions > 0 and bboxes_to_process_for_film.shape[2] != self.num_regions)):
                    self.logger.warning(f"Disc GAAD bbox shape mismatch for FiLM. Exp (B={B}, N={self.num_frames_to_discriminate}, Reg={self.num_regions if self.num_regions > 0 else 'Any'}), Got {bboxes_to_process_for_film.shape}. Zero cond.")
                    frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
                elif self.num_regions > 0 : # num_regions check implies bboxes_to_process_for_film.shape[2] == self.num_regions
                    norm_bboxes_disc = self._normalize_bboxes_disc(bboxes_to_process_for_film.to(dtype_in), H, W)
                    norm_bboxes_flat_disc = norm_bboxes_disc.view(B * self.num_frames_to_discriminate, -1) # Flatten (NumReg * 4)
                    frame_conditions_flat_disc = self.frame_gaad_embedder_disc(norm_bboxes_flat_disc)
                else: # self.num_regions is 0, but FiLM is somehow active (should be prevented by init logic)
                    self.logger.warning("FiLM active but num_regions is 0 in forward. Zero cond.")
                    frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
            else: # gaad_bboxes_for_disc is None
                self.logger.debug("Disc GAAD Embedder active but no bboxes provided. Zero cond for FiLM.")
                frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
            
            if frame_conditions_flat_disc is not None: # Should always be true due to fallbacks
                frame_conditions_reshaped_disc = frame_conditions_flat_disc.view(B, self.num_frames_to_discriminate, self.gaad_condition_dim)
                sequence_condition_disc = torch.mean(frame_conditions_reshaped_disc, dim=1).to(dtype_in) # Average FiLM condition over frames

        if self.disc_type == "spatio_temporal_cnn":
            features = frames_to_process.permute(0, 2, 1, 3, 4).to(dtype_in) # (B, C, N_disc_frames, H, W)
            for block_module in self.feature_extractor_blocks:
                features = block_module['conv'](features)
                features = block_module['norm'](features)
                if block_module.get('film') is not None and sequence_condition_disc is not None:
                    features = block_module['film'](features, sequence_condition_disc)
                features = block_module['activation'](features)
            
            features = self.adaptive_pool(features) # Output (B, C_final, D_pooled, 1, 1)
            features_flat = features.view(B, -1) # Flatten for FC
            logits = self.final_fc_layers(features_flat)
            return logits.to(dtype_in)

        elif self.disc_type == "regional_cnn":
            if self.resize_transform_regional is None:
                raise RuntimeError("resize_transform_regional not initialized in RegionalDiscriminator with type 'regional_cnn'.")

            frame_to_disc_regional = frames_to_process[:, 0, ...].to(dtype_in) # Use first frame
            gaad_bboxes_list_regional = []

            for b_idx in range(B):
                frame_dims_regional = (W,H)
                max_w_scalar = float(W)
                max_h_scalar = float(H)
                
                # Determine frame_bboxes_reg based on decomposition type
                current_frame_bboxes_reg = None
                if 'golden_subdivide_rect_fixed_n' not in globals() or 'phi_spiral_patch_centers_fixed_n' not in globals():
                    self.logger.error("GAAD utils (golden_subdivide_rect_fixed_n or phi_spiral_patch_centers_fixed_n) not found for 'regional_cnn' discriminator.")
                    if self.num_regions > 0:
                        current_frame_bboxes_reg = torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_regions, dtype=dtype_in, device=device)
                    else:
                        current_frame_bboxes_reg = torch.empty(0,4,dtype=dtype_in, device=device)
                elif self.num_regions == 0:
                    current_frame_bboxes_reg = torch.empty(0,4,dtype=dtype_in, device=device)
                elif self.decomposition_type_regional == "hybrid":
                    num_subdivide = self.num_regions // 2
                    num_spiral = self.num_regions - num_subdivide
                    bboxes_for_item = []
                    if num_subdivide > 0:
                        bboxes_for_item.append(
                            golden_subdivide_rect_fixed_n(frame_dims_regional, num_subdivide, device, dtype_in, self.gaad_min_size_px_regional)
                        )
                    if num_spiral > 0:
                        spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims_regional, num_spiral, device, dtype_in)
                        patch_base_size = min(frame_dims_regional)
                        spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype_in)
                        patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0
                        patch_ws = patch_hs # Assuming square patches
                        val_x1 = spiral_centers[:,0] - patch_ws; val_y1 = spiral_centers[:,1] - patch_hs
                        val_x2 = spiral_centers[:,0] + patch_ws; val_y2 = spiral_centers[:,1] + patch_hs
                        
                        spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar - EPS)
                        spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar - EPS)
                        min_for_x2 = spiral_bboxes_current[:,0] + EPS
                        spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar)
                        spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)
                        min_for_y2 = spiral_bboxes_current[:,1] + EPS
                        spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar)
                        spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2)
                        bboxes_for_item.append(spiral_bboxes_current)
                    
                    if bboxes_for_item:
                        current_frame_bboxes_reg = torch.cat(bboxes_for_item, dim=0)
                    elif self.num_regions > 0:
                        current_frame_bboxes_reg = torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_regions, dtype=dtype_in, device=device)
                    else:
                        current_frame_bboxes_reg = torch.empty(0,4,dtype=dtype_in, device=device)
                
                elif self.decomposition_type_regional == "spiral":
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims_regional, self.num_regions, device, dtype_in)
                    patch_base_size = min(frame_dims_regional)
                    spiral_bboxes_current = torch.zeros(self.num_regions, 4, device=device, dtype=dtype_in)
                    patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0
                    patch_ws = patch_hs
                    val_x1 = spiral_centers[:,0] - patch_ws; val_y1 = spiral_centers[:,1] - patch_hs
                    val_x2 = spiral_centers[:,0] + patch_ws; val_y2 = spiral_centers[:,1] + patch_hs
                    
                    spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar - EPS)
                    spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar - EPS)
                    min_for_x2 = spiral_bboxes_current[:,0] + EPS
                    spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar)
                    spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)
                    min_for_y2 = spiral_bboxes_current[:,1] + EPS
                    spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar)
                    spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2)
                    current_frame_bboxes_reg = spiral_bboxes_current
                else: # Default to subdivide
                    current_frame_bboxes_reg = golden_subdivide_rect_fixed_n(frame_dims_regional, self.num_regions, device, dtype_in, self.gaad_min_size_px_regional)

                # Pad or truncate bboxes for this frame to match self.num_regions
                if self.num_regions > 0 and current_frame_bboxes_reg.shape[0] < self.num_regions:
                    num_to_pad = self.num_regions - current_frame_bboxes_reg.shape[0]
                    if current_frame_bboxes_reg.shape[0] > 0:
                        padding_box = current_frame_bboxes_reg[-1:].clone()
                    else: 
                        padding_box = torch.tensor([[0,0,max_w_scalar,max_h_scalar]],dtype=dtype_in,device=device)
                    padding = padding_box.repeat(num_to_pad,1)
                    current_frame_bboxes_reg = torch.cat([current_frame_bboxes_reg, padding], dim=0)
                elif self.num_regions > 0 and current_frame_bboxes_reg.shape[0] > self.num_regions:
                    current_frame_bboxes_reg = current_frame_bboxes_reg[:self.num_regions]
                
                gaad_bboxes_list_regional.append(current_frame_bboxes_reg)
            
            if self.num_regions == 0: # Should be caught by init if regional_cnn and num_regions=0
                self.logger.warning("RegionalDisc 'regional_cnn' with num_regions=0. Returning zero logits.")
                return torch.zeros(B, 1, device=device, dtype=dtype_in)
            
            gaad_bboxes_batch_regional = torch.stack(gaad_bboxes_list_regional) # (B, NumRegions, 4)
            all_regional_features = []

            for b_idx in range(B):
                batch_region_features = []
                for r_idx in range(self.num_regions):
                    x1,y1,x2,y2 = gaad_bboxes_batch_regional[b_idx,r_idx].round().int().tolist()
                    x1_c = max(0,x1); y1_c = max(0,y1)
                    x2_c = min(W,x2); y2_c = min(H,y2)
                    
                    if x1_c >= x2_c or y1_c >= y2_c: # Handle empty or invalid region
                        feat_dim_input_to_fc = self.final_fc_layers_regional.in_features
                        # Create zero features matching the input dimension of the FC layer
                        patch_features_reg = torch.zeros(1, feat_dim_input_to_fc, device=device, dtype=dtype_in)
                    else:
                        patch = frame_to_disc_regional[b_idx, :, y1_c:y2_c, x1_c:x2_c]
                        resized_patch = self.resize_transform_regional(patch).unsqueeze(0) # Add batch dim for Conv2d
                        patch_features_extracted = self.regional_feature_extractor_2d(resized_patch)
                        patch_features_reg = patch_features_extracted.view(1, -1) # Flatten for FC layer
                    
                    batch_region_features.append(patch_features_reg)
                all_regional_features.append(torch.cat(batch_region_features, dim=0)) # (NumRegions, feat_dim_per_region)
            
            regional_features_tensor = torch.stack(all_regional_features) # (B, NumRegions, feat_dim_per_region)
            aggregated_features = torch.mean(regional_features_tensor, dim=1) # Aggregate over regions (B, feat_dim_per_region)
            logits = self.final_fc_layers_regional(aggregated_features)
            return logits.to(dtype_in)
        else:
            raise NotImplementedError(f"Discriminator forward not implemented for type {self.disc_type}")




# Constants and Default Configs (ensure EPS is defined if used, it is defined in the full script)
# EPS = 1e-5 # This should be at the top of the main script.

# --- Optical Flow Import --- (Ensure this is present earlier in the file)
try:
    import torchvision.models.optical_flow as tv_flow
    OPTICAL_FLOW_AVAILABLE = True
    FLOW_MODELS = {
        'raft_large': (tv_flow.Raft_Large_Weights.DEFAULT, tv_flow.raft_large),
        'raft_small': (tv_flow.Raft_Small_Weights.DEFAULT, tv_flow.raft_small),
    }
except ImportError:
    tv_flow = None
    OPTICAL_FLOW_AVAILABLE = False
    FLOW_MODELS = {}
    # print("Warning: torchvision.models.optical_flow not available. Motion branch will be disabled if selected.") # Already handled by logger


class RegionalHyperbolicMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_motion_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_m_config = wubu_m_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV01DFT.EncoderM")
        self.enabled = args.use_wubu_motion_branch and wubu_m_config is not None and gaad_motion_config is not None and OPTICAL_FLOW_AVAILABLE
        self.flow_net = None

        if self.enabled:
            if args.optical_flow_net_type not in FLOW_MODELS:
                self.logger.error(f"Optical flow model '{args.optical_flow_net_type}' not found. Disabling motion branch.")
                self.enabled = False
            else:
                weights, model_builder = FLOW_MODELS[args.optical_flow_net_type]
                try:
                    self.flow_net = model_builder(weights=weights)
                    if args.freeze_flow_net:
                        self.flow_net.eval()
                        for param in self.flow_net.parameters():
                            param.requires_grad = False
                    self.logger.info(f"Loaded OptFlow Net: {args.optical_flow_net_type} (Frozen: {args.freeze_flow_net})")
                except Exception as e:
                    self.logger.error(f"Failed to load optflow model '{args.optical_flow_net_type}': {e}. Disabling motion.", exc_info=True)
                    self.flow_net = None
                    self.enabled = False
        else:
            self.logger.info("Motion Encoder branch DISABLED (config/availability).")

        if not self.enabled:
            self.wubu_m_final_curvature = 1.0 # Default value if not enabled
            return

        self.image_size = (args.image_h, args.image_w)
        self.num_motion_regions = gaad_motion_config['num_regions'] # type: ignore [index]
        self.motion_decomposition_type = gaad_motion_config['decomposition_type'] # type: ignore [index]
        self.gaad_min_size_px = gaad_motion_config.get('min_size_px', 5) # type: ignore [union-attr]

        self.flow_stats_components = args.flow_stats_components
        self.flow_stats_dim = 0
        if 'mag_mean' in self.flow_stats_components: self.flow_stats_dim += 1
        if 'angle_mean' in self.flow_stats_components: self.flow_stats_dim += 2 # cos, sin
        if 'mag_std' in self.flow_stats_components: self.flow_stats_dim += 1
        if 'angle_std' in self.flow_stats_components: self.flow_stats_dim += 1
        if self.flow_stats_dim == 0:
            self.logger.warning("No flow stats selected for motion encoder. This branch will output zeros if no features.")
        
        self.motion_feature_embed = nn.Linear(self.flow_stats_dim, args.encoder_initial_tangent_dim) if self.flow_stats_dim > 0 else nn.Identity()
        
        if self.wubu_m_config is not None and self.flow_stats_dim > 0 :
            self.wubu_m = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=args.encoder_initial_tangent_dim, 
                output_tangent_dim=video_config['wubu_m_output_dim'], 
                config=wubu_m_config
            )
            self.wubu_m_final_hyp_dim = wubu_m_config['hyperbolic_dims'][-1] if wubu_m_config['num_levels'] > 0 and wubu_m_config['hyperbolic_dims'] else 0
            self.wubu_m_final_curvature = 1.0
            if wubu_m_config['num_levels'] > 0 and self.wubu_m_final_hyp_dim > 0:
                last_level_idx = wubu_m_config['num_levels'] - 1
                try:
                    # Temporarily create a level to get its curvature
                    temp_level_m = HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_m_final_hyp_dim, wubu_m_config, wubu_m_config['initial_curvatures'][last_level_idx])
                    self.wubu_m_final_curvature = temp_level_m.get_current_curvature_scalar()
                    del temp_level_m
                    self.logger.info(f"WuBu-M final C estimated: {self.wubu_m_final_curvature:.3f}")
                except IndexError:
                    self.logger.error(f"Index error accessing WuBu-M config for level {last_level_idx}. Defaulting C=1.0.")
                    self.wubu_m_final_curvature = 1.0
            # Set manifold for output projection if hyperbolic
            if self.wubu_m_final_hyp_dim > 0 and hasattr(self.wubu_m, 'output_tangent_projection'):
                 for p in self.wubu_m.output_tangent_projection.parameters(): # type: ignore
                    setattr(p, 'manifold', PoincareBall(self.wubu_m_final_curvature))
        elif self.wubu_m_config is not None and self.flow_stats_dim == 0:
             self.logger.warning("WuBu-M configured, but flow_stats_dim=0. WuBu-M will be ineffective or bypassed.")
             self.wubu_m = None # type: ignore
             self.wubu_m_final_hyp_dim=0
             self.wubu_m_final_curvature=1.0
        else: # wubu_m_config is None
             self.logger.error("MotionEncoder: wubu_m_config is None, cannot initialize WuBu-M. Disabling motion branch.")
             self.wubu_m = None # type: ignore
             self.wubu_m_final_hyp_dim=0
             self.wubu_m_final_curvature=1.0
             self.enabled = False # Ensure it's disabled if WuBu-M cannot be set up

        self.apply(init_weights_general)

    def _get_motion_gaad_bboxes(self, analysis_map: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        B_eff, _, H, W = analysis_map.shape
        all_batch_bboxes = []
        for i in range(B_eff):
            frame_dims = (W, H)
            max_w_scalar = float(W)
            max_h_scalar = float(H)
            frame_bboxes = None # Initialize

            if self.motion_decomposition_type == "hybrid":
                num_subdivide = self.num_motion_regions // 2
                num_spiral = self.num_motion_regions - num_subdivide
                bboxes_for_item = []
                if num_subdivide > 0:
                    bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims, num_subdivide, device, dtype, self.gaad_min_size_px))
                if num_spiral > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype)
                    patch_base_size = min(frame_dims)
                    spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype)
                    patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0
                    patch_ws = patch_hs # Assuming square patches
                    val_x1 = spiral_centers[:,0] - patch_ws; val_y1 = spiral_centers[:,1] - patch_hs
                    val_x2 = spiral_centers[:,0] + patch_ws; val_y2 = spiral_centers[:,1] + patch_hs
                    
                    spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar - EPS)
                    spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar - EPS)
                    min_for_x2 = spiral_bboxes_current[:,0] + EPS
                    spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar)
                    spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)
                    min_for_y2 = spiral_bboxes_current[:,1] + EPS
                    spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar)
                    spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2) # TYPO FIX HERE
                    bboxes_for_item.append(spiral_bboxes_current)
                
                if bboxes_for_item:
                    frame_bboxes = torch.cat(bboxes_for_item, dim=0)
                elif self.num_motion_regions > 0: # No items but regions expected
                    frame_bboxes = torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_motion_regions, dtype=dtype, device=device)
                else: # No items and no regions expected
                    frame_bboxes = torch.empty(0,4,dtype=dtype, device=device)

            elif self.motion_decomposition_type == "spiral":
                spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_motion_regions, device, dtype)
                patch_base_size = min(frame_dims)
                spiral_bboxes_current = torch.zeros(self.num_motion_regions, 4, device=device, dtype=dtype)
                patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0
                patch_ws = patch_hs
                val_x1 = spiral_centers[:,0] - patch_ws; val_y1 = spiral_centers[:,1] - patch_hs
                val_x2 = spiral_centers[:,0] + patch_ws; val_y2 = spiral_centers[:,1] + patch_hs
                
                spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar - EPS)
                spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar - EPS)
                min_for_x2 = spiral_bboxes_current[:,0] + EPS
                spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar)
                spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)
                min_for_y2 = spiral_bboxes_current[:,1] + EPS
                spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar)
                spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2) # TYPO FIX HERE
                frame_bboxes = spiral_bboxes_current
            else: # Default to subdivide
                frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims, self.num_motion_regions, device, dtype, self.gaad_min_size_px)

            # Pad or truncate bboxes for this frame to match self.num_motion_regions
            if frame_bboxes.shape[0] < self.num_motion_regions:
                num_to_pad = self.num_motion_regions - frame_bboxes.shape[0]
                if frame_bboxes.shape[0] > 0:
                    padding_box = frame_bboxes[-1:].clone()
                else: 
                    padding_box = torch.tensor([[0,0,max_w_scalar,max_h_scalar]], dtype=dtype, device=device)
                padding = padding_box.repeat(num_to_pad, 1)
                frame_bboxes = torch.cat([frame_bboxes, padding], dim=0)
            elif frame_bboxes.shape[0] > self.num_motion_regions:
                frame_bboxes = frame_bboxes[:self.num_motion_regions]
            
            all_batch_bboxes.append(frame_bboxes)
        return torch.stack(all_batch_bboxes)

    def _extract_flow_statistics(self, flow_field: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        B, _, H, W = flow_field.shape
        N_reg = bboxes.shape[1]
        device = flow_field.device
        dtype = flow_field.dtype
        all_stats = torch.zeros(B, N_reg, self.flow_stats_dim, device=device, dtype=dtype)

        for b in range(B):
            for r in range(N_reg):
                x1, y1, x2, y2 = bboxes[b, r].round().int().tolist()
                x1_c = max(0, x1); y1_c = max(0, y1)
                x2_c = min(W, x2); y2_c = min(H, y2)

                if x1_c >= x2_c or y1_c >= y2_c: continue # Skip empty region

                region_flow = flow_field[b, :, y1_c:y2_c, x1_c:x2_c] # (2, H_reg, W_reg)
                flow_dx = region_flow[0, ...].flatten()
                flow_dy = region_flow[1, ...].flatten()

                if flow_dx.numel() == 0: continue

                stat_idx = 0
                magnitudes = torch.sqrt(flow_dx**2 + flow_dy**2)
                if 'mag_mean' in self.flow_stats_components:
                    all_stats[b, r, stat_idx] = torch.mean(magnitudes)
                    stat_idx += 1
                if 'mag_std' in self.flow_stats_components:
                    all_stats[b, r, stat_idx] = torch.std(magnitudes) if magnitudes.numel() > 1 else 0.0
                    stat_idx += 1
                
                angles = torch.atan2(flow_dy, flow_dx)
                if 'angle_mean' in self.flow_stats_components:
                    all_stats[b,r,stat_idx] = torch.mean(torch.cos(angles)) # Store cos(angle_mean)
                    stat_idx+=1
                    all_stats[b,r,stat_idx] = torch.mean(torch.sin(angles)) # Store sin(angle_mean)
                    stat_idx+=1
                if 'angle_std' in self.flow_stats_components:
                    # For angular std, it's more complex. A simple std might not be ideal due to wrap-around.
                    # For now, keeping it simple, but this could be improved (e.g., circular standard deviation).
                    angle_std = torch.std(angles) if angles.numel() > 1 else 0.0
                    all_stats[b,r,stat_idx] = angle_std if torch.isfinite(angle_std) else 0.0
                    stat_idx+=1
        return all_stats

    def forward(self, frames_pixels: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.enabled or self.flow_net is None :
            return None
        
        B, N_frames, C, H, W = frames_pixels.shape
        device = frames_pixels.device
        original_dtype = frames_pixels.dtype
        # Use a consistent compute dtype, preferably from model parameters
        compute_dtype = next(self.parameters(), torch.tensor(0.0, device=device)).dtype

        if N_frames < 2:
            self.logger.debug(f"Not enough frames ({N_frames}) for optical flow. Skipping motion branch.")
            return None

        num_pairs = N_frames - 1
        all_motion_features_list = []
        all_motion_bboxes_list = []

        flow_context = torch.no_grad() if self.args.freeze_flow_net else contextlib.nullcontext()
        
        with flow_context:
            for i in range(num_pairs):
                # RAFT expects CHW, float32, [0, 255]
                # Input frames_pixels are assumed to be normalized, e.g. [-1, 1] or [0,1]
                # Denormalize for RAFT if necessary
                frame_t_orig_norm = frames_pixels[:, i+1, ...]
                frame_t_minus_1_orig_norm = frames_pixels[:, i, ...]

                # Assuming input [-1, 1] normalized by T.Normalize([0.5]*C, [0.5]*C)
                # Convert to [0, 255]
                frame_t_raft = ((frame_t_orig_norm * 0.5 + 0.5) * 255.0).to(torch.float32)
                frame_t_minus_1_raft = ((frame_t_minus_1_orig_norm * 0.5 + 0.5) * 255.0).to(torch.float32)
                
                try:
                    # Ensure flow net is on the correct device
                    self.flow_net = self.flow_net.to(device) # type: ignore [union-attr]
                    flow_predictions = self.flow_net(frame_t_minus_1_raft, frame_t_raft) # type: ignore [operator]
                    flow_field = flow_predictions[-1].to(compute_dtype) # Last prediction is usually the best
                except Exception as e_flow:
                    self.logger.error(f"Optical flow computation failed for pair {i} (Shapes: {frame_t_minus_1_raft.shape}, {frame_t_raft.shape}): {e_flow}", exc_info=True)
                    return None

                # Use flow magnitude for GAAD bbox generation for motion regions
                flow_magnitude = torch.sqrt(flow_field[:, 0:1, :, :]**2 + flow_field[:, 1:2, :, :]**2)
                motion_gaad_bboxes_batch = self._get_motion_gaad_bboxes(flow_magnitude, device, compute_dtype) # (B_eff, NumMotionReg, 4)

                if self.flow_stats_dim > 0:
                    flow_stats = self._extract_flow_statistics(flow_field, motion_gaad_bboxes_batch) # (B_eff, NumMotionReg, flow_stats_dim)
                    flow_stats_flat = flow_stats.reshape(B * self.num_motion_regions, self.flow_stats_dim)
                    initial_motion_tangent_vectors_flat = self.motion_feature_embed(flow_stats_flat)
                else: # No flow stats selected, create zero vectors
                    initial_motion_tangent_vectors_flat = torch.zeros(B * self.num_motion_regions, self.args.encoder_initial_tangent_dim, device=device, dtype=compute_dtype)

                if self.wubu_m is None: # If WuBu-M is not configured (e.g. flow_stats_dim was 0)
                    motion_features_pair_flat = initial_motion_tangent_vectors_flat # Use raw embedded stats
                else:
                    wubu_m_output_tangent_flat = self.wubu_m(initial_motion_tangent_vectors_flat)
                    # Output of WuBu-M is tangent. For VAE encoder, it's better to keep it tangent.
                    # If it were for direct use or another type of model, expmap might be needed.
                    # The VAE encoder's WuBu-T will expect tangent inputs.
                    motion_features_pair_flat = wubu_m_output_tangent_flat
                    # If we wanted hyperbolic points:
                    # motion_features_pair_flat = PoincareBall(self.wubu_m_final_curvature).expmap0(wubu_m_output_tangent_flat) if self.wubu_m_final_hyp_dim > 0 else wubu_m_output_tangent_flat

                motion_features_pair = motion_features_pair_flat.reshape(B, self.num_motion_regions, -1)
                all_motion_features_list.append(motion_features_pair)
                all_motion_bboxes_list.append(motion_gaad_bboxes_batch)

        if not all_motion_features_list:
            self.logger.warning("No motion features were generated (list is empty).")
            return None

        final_motion_features = torch.stack(all_motion_features_list, dim=1).to(original_dtype) # (B, NumPairs, NumMotionReg, D_motion_out)
        final_motion_bboxes = torch.stack(all_motion_bboxes_list, dim=1).to(original_dtype) # (B, NumPairs, NumMotionReg, 4)
        
        return final_motion_features, final_motion_bboxes         





# =====================================================================
# VAE-GAN Model (WuBuGAADHybridGenNet)
# =====================================================================

class WuBuGAADHybridGenNet(nn.Module):
    """Combines Encoder and Generator for VAE-GAN. Supports DFT features for appearance."""
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_appearance_config: Dict, gaad_motion_config: Optional[Dict], wubu_s_config: Dict, wubu_t_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_appearance_config = gaad_appearance_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_s_config = wubu_s_config
        self.wubu_t_config = wubu_t_config
        self.wubu_m_config = wubu_m_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV01DFT.MainNet")

        self.latent_dim = args.latent_dim

        self.encoder = RegionalVAEEncoder(args, video_config, gaad_appearance_config, wubu_s_config, self.latent_dim)
        self.motion_encoder: Optional[RegionalHyperbolicMotionEncoder] = None
        if args.use_wubu_motion_branch:
             temp_motion_encoder = RegionalHyperbolicMotionEncoder(args, video_config, gaad_motion_config, wubu_m_config)
             if temp_motion_encoder.enabled: self.motion_encoder = temp_motion_encoder; self.logger.info("Motion Encoder Branch Activated.")
             else: self.logger.warning("Motion branch requested but disabled."); args.use_wubu_motion_branch = False; self.wubu_m_config = None; self.gaad_motion_config = None; video_config['wubu_m_output_dim'] = 0

        self.generator = RegionalGeneratorDecoder(args, video_config, gaad_appearance_config, self.latent_dim) # Removed wubu_s_output_dim

        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"WuBuGAADHybridGenNet Initialized (DFT Appearance: {args.use_dft_features_appearance}): {param_count:,} params.")

    def encode(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Output: mu, logvar, gaad_bboxes_all_frames, regional_app_features_for_recon_target (raw DFT if used, else None)
        motion_features = None
        if self.motion_encoder is not None and self.motion_encoder.enabled:
            motion_output_tuple = self.motion_encoder(frames_pixels)
            if motion_output_tuple is not None:
                motion_features, _ = motion_output_tuple

        mu, logvar, gaad_bboxes_all_frames, regional_app_features_for_recon_target = self.encoder(frames_pixels, motion_features)
        return mu, logvar, gaad_bboxes_all_frames, regional_app_features_for_recon_target

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, gaad_bboxes_for_decode: torch.Tensor) -> torch.Tensor:
        # Output:
        #   If DFT: (B, N_pred, NumReg, NumImgCh, 2, H_dft, W_dft_coeffs)
        #   If Pixel: (B, N_pred, C, H_img, W_img)
        return self.generator(z, gaad_bboxes_for_decode)

    def forward(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Output:
        #   recon_output (DFT coeffs or Pixels), mu, logvar, bboxes_for_decode, regional_app_features_for_recon_target (raw DFT if used, else None)

        mu, logvar, gaad_bboxes_all_input, regional_app_features_for_recon_target = self.encode(frames_pixels)
        z = self.reparameterize(mu, logvar)

        num_input_f = self.video_config["num_input_frames"]
        num_predict_f = self.video_config["num_predict_frames"]
        expected_total_bbox_frames = frames_pixels.shape[1]

        if gaad_bboxes_all_input.shape[1] != expected_total_bbox_frames :
             self.logger.warning(f"MainNet Fwd: Encoder bbox sets ({gaad_bboxes_all_input.shape[1]}) != input frames ({expected_total_bbox_frames}).")

        if gaad_bboxes_all_input.shape[1] < num_input_f + num_predict_f:
            self.logger.error(f"MainNet Fwd: Not enough bbox sets from encoder ({gaad_bboxes_all_input.shape[1]}) for input ({num_input_f}) + pred ({num_predict_f}).")
            if gaad_bboxes_all_input.shape[1] >= num_predict_f:
                 decoder_bboxes_selected = gaad_bboxes_all_input[:, -num_predict_f:, ...]
            elif gaad_bboxes_all_input.shape[1] > 0:
                 decoder_bboxes_selected = gaad_bboxes_all_input
            else: raise ValueError("MainNet Fwd: Encoder returned no GAAD bboxes.")
        else:
            decoder_bboxes_selected = gaad_bboxes_all_input[:, num_input_f : num_input_f + num_predict_f, ...]

        if decoder_bboxes_selected.shape[1] != num_predict_f:
            self.logger.warning(f"MainNet Fwd: Sliced decoder_bboxes has {decoder_bboxes_selected.shape[1]} frames, gen expects {num_predict_f}. Padding/truncating.")
            if decoder_bboxes_selected.shape[1] < num_predict_f:
                if decoder_bboxes_selected.shape[1] == 0: raise ValueError("MainNet Fwd: No bboxes for decoder post-slice.")
                num_to_pad = num_predict_f - decoder_bboxes_selected.shape[1]
                padding_slice = decoder_bboxes_selected[:, -1:, ...].repeat(1, num_to_pad, 1, 1)
                decoder_bboxes_selected = torch.cat([decoder_bboxes_selected, padding_slice], dim=1)
            else: decoder_bboxes_selected = decoder_bboxes_selected[:, :num_predict_f, ...]

        recon_output = self.decode(z, decoder_bboxes_selected) # DFT coeffs or Pixels
        return recon_output, mu, logvar, decoder_bboxes_selected, regional_app_features_for_recon_target

# =====================================================================
# Dataset (Unchanged from Diffusion version)
# =====================================================================
class VideoFrameDataset(Dataset): # ... (No changes from original, keep as is)
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int], frame_skip: int = 1, data_fraction: float = 1.0):
        super().__init__(); self.video_path = video_path; self.num_frames_total = num_frames_total; self.image_size = image_size; self.frame_skip = frame_skip; current_logger=logging.getLogger("WuBuGAADHybridGenV01DFT.Dataset")
        if not os.path.isfile(self.video_path): current_logger.error(f"Video file not found: {self.video_path}"); raise FileNotFoundError(f"Video file not found: {self.video_path}")
        current_logger.info(f"Attempting to load entire video into RAM: {self.video_path}"); self.video_frames_in_ram = None; self.source_video_fps = 30.0; read_success = False
        if VIDEO_IO_AVAILABLE and video_io is not None:
            try: video_data = video_io.read_video(self.video_path, output_format="TCHW", pts_unit="sec"); self.video_frames_in_ram = video_data[0].contiguous(); self.source_video_fps = video_data[2].get('video_fps', 30.0); read_success = True
            except Exception as e_tv: current_logger.warning(f"torchvision.io failed to read {self.video_path}: {e_tv}. Trying imageio...")
        else: current_logger.warning("torchvision.io not available. Trying imageio...")
        if not read_success and IMAGEIO_AVAILABLE and imageio is not None:
            try: reader = imageio.get_reader(self.video_path); meta = reader.get_meta_data(); self.source_video_fps = meta.get('fps', 30.0); frames = [torch.from_numpy(frame_np).permute(2, 0, 1) for frame_np in reader]; self.video_frames_in_ram = torch.stack(frames).contiguous(); reader.close(); read_success = True
            except Exception as e_ii: current_logger.error(f"imageio also failed to read {self.video_path}: {e_ii}", exc_info=True); raise RuntimeError(f"Failed to load video '{self.video_path}' using both torchvision and imageio.") from e_ii
        elif not read_success: raise RuntimeError(f"Failed to load video '{self.video_path}'. Neither torchvision.io nor imageio could read it or are available.")
        ram_usage_gb = self.video_frames_in_ram.nbytes / (1024**3); current_logger.info(f"Loaded video into RAM. Shape: {self.video_frames_in_ram.shape}, Dtype: {self.video_frames_in_ram.dtype}, FPS: {self.source_video_fps:.2f}. Est RAM: {ram_usage_gb:.2f} GB.")
        self.resize_transform = T.Resize(self.image_size, antialias=True); self.normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]); self.num_disk_frames = self.video_frames_in_ram.shape[0]; self.samples = []; required_span_len = (self.num_frames_total - 1) * self.frame_skip + 1
        if self.num_disk_frames >= required_span_len: [self.samples.append(i) for i in range(self.num_disk_frames - required_span_len + 1)]
        else: current_logger.warning(f"Not enough frames ({self.num_disk_frames}) in '{self.video_path}' for span {required_span_len}.")
        if data_fraction < 1.0 and len(self.samples) > 1: num_to_keep = max(1, int(len(self.samples) * data_fraction)); self.samples = random.sample(self.samples, num_to_keep); current_logger.info(f"Using {data_fraction*100:.2f}% of samples: {len(self.samples)} samples.")
        if not self.samples: current_logger.error(f"VideoFrameDataset: No valid samples from '{self.video_path}'.");
        else: current_logger.info(f"VideoFrameDataset RAM init. Frames:{self.num_disk_frames}. Samples:{len(self.samples)}. Len:{self.num_frames_total} (skip {self.frame_skip}).")
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> torch.Tensor:
        start_frame_idx_in_ram = self.samples[idx]; frames_for_sample = []
        for i in range(self.num_frames_total):
            actual_frame_idx_in_ram = start_frame_idx_in_ram + i * self.frame_skip
            if actual_frame_idx_in_ram < self.num_disk_frames:
                try: frame_tensor_chw_uint8 = self.video_frames_in_ram[actual_frame_idx_in_ram]; resized_frame_tensor = self.resize_transform(frame_tensor_chw_uint8); frame_float_01 = resized_frame_tensor.float() / 255.0; transformed_frame = self.normalize_transform(frame_float_01); frames_for_sample.append(transformed_frame)
                except Exception as e: logging.getLogger("WuBuGAADHybridGenV01DFT.Dataset").error(f"Error transforming frame {actual_frame_idx_in_ram} sample {idx}: {e}", exc_info=True); raise e
            else: logging.getLogger("WuBuGAADHybridGenV01DFT.Dataset").error(f"Frame index {actual_frame_idx_in_ram} out of bounds (total: {self.num_disk_frames}). Sample: {idx}"); raise IndexError("Frame index out of bounds.")
        if len(frames_for_sample) != self.num_frames_total: logging.getLogger("WuBuGAADHybridGenV01DFT.Dataset").error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total} for sample {idx}"); raise ValueError("Incorrect number of frames loaded for sample.")
        return torch.stack(frames_for_sample)

# =====================================================================
# VAE-GAN Trainer (Modified for DFT)
# =====================================================================

class HybridTrainer:
    def __init__(self, model: "WuBuGAADHybridGenNet", discriminator: "RegionalDiscriminator", optimizer_enc_gen: torch.optim.Optimizer, optimizer_disc: torch.optim.Optimizer, device: torch.device, train_loader: DataLoader, val_loader: Optional[DataLoader], args: argparse.Namespace, rank: int, world_size: int, ddp_active: bool):
        self.model = model
        self.discriminator = discriminator
        self.optimizer_enc_gen = optimizer_enc_gen
        self.optimizer_disc = optimizer_disc
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.ddp_active = ddp_active
        self.am_main_process = (rank == 0)
        self.logger = logging.getLogger("WuBuGAADHybridGenV01DFT.Trainer")
        self.video_config = getattr(model, 'video_config', {})
        self.gaad_appearance_config = getattr(model, 'gaad_appearance_config', {})
        
        self.lambda_recon = args.lambda_recon
        self.lambda_kl = args.lambda_kl
        self.lambda_gan = args.lambda_gan
        
        self.scaler_enc_gen = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.scaler_disc = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric_val = -float('inf') if args.val_primary_metric in ["avg_val_psnr", "avg_val_ssim"] else float('inf')
        self.last_val_metrics: Dict[str, Any] = {}
        
        if self.am_main_process:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
        self.lpips_loss_fn: Optional[lpips.LPIPS] = None
        self.ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None
        
        if self.am_main_process and self.args.use_lpips_for_verification:
             if LPIPS_AVAILABLE and lpips is not None:
                 try:
                     self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)
                     self.logger.info("LPIPS metric enabled.")
                 except Exception as e_lpips:
                     self.logger.warning(f"LPIPS init failed: {e_lpips}. Disabling LPIPS.")
                     self.lpips_loss_fn = None
             else:
                 self.logger.warning("LPIPS library unavailable. Skipping LPIPS metric.")
                 
        if self.am_main_process and TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
             try:
                 self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device) # Assuming data is in [0,1] for SSIM
                 self.logger.info("SSIM metric enabled.")
             except Exception as e:
                 self.logger.warning(f"SSIM metric init failed: {e}. Skipping SSIM metric.")
                 self.ssim_metric = None
        elif self.am_main_process:
             self.logger.warning("torchmetrics SSIM unavailable. Skipping SSIM metric.")
             
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        if self.grad_accum_steps > 1 and self.am_main_process:
            self.logger.info(f"Gradient accumulation steps: {self.grad_accum_steps}.")
            
        self.fixed_noise_for_sampling: Optional[torch.Tensor] = None
        
        self.lambda_kl_update_interval = getattr(args, 'lambda_kl_update_interval', 0)
        self.lambda_kl_q_controller: Optional[HAKMEMQController] = None
        if self.args.q_controller_enabled and self.lambda_kl_update_interval > 0:
            q_cfg_lambda_kl = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
            self.lambda_kl_q_controller = HAKMEMQController(**q_cfg_lambda_kl)
            if self.am_main_process:
                self.logger.info(f"Separate Lambda_KL Q-Control ENABLED. Interval: {self.lambda_kl_update_interval}. Config: {q_cfg_lambda_kl}")
            if hasattr(self.lambda_kl_q_controller, 'set_current_lambda_kl'):
                 self.lambda_kl_q_controller.set_current_lambda_kl(self.lambda_kl)
        elif self.am_main_process:
            self.logger.info("Lambda_KL Q-Control DISABLED (interval <= 0 or Q-controller globally disabled).")
            
        self.interval_metrics_accum = defaultdict(float)
        self.interval_steps_count = 0
        self.prev_interval_metrics_for_lambda_kl_reward: Optional[Dict[str, Union[float, None]]] = None
        self.min_lambda_kl_q_control = getattr(args, 'min_lambda_kl_q_control', 1e-6)
        self.max_lambda_kl_q_control = getattr(args, 'max_lambda_kl_q_control', 1.0)

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def _compute_recon_loss(self, recon_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        if recon_features.shape != target_features.shape:
            self.logger.error(f"Reconstruction loss shape mismatch: Recon {recon_features.shape}, Target {target_features.shape}")
            # Attempt to reshape if total elements and last dimension match (e.g. B,N,R,D vs B*N*R,D)
            if recon_features.numel() == target_features.numel() and \
               (recon_features.ndim > 1 and target_features.ndim > 1 and recon_features.size(-1) == target_features.size(-1)):
                feature_dim = recon_features.size(-1)
                recon_features = recon_features.reshape(-1, feature_dim)
                target_features = target_features.reshape(-1, feature_dim)
                self.logger.warning("Recon loss: Reshaped tensors due to initial shape mismatch.")
            else:
                self.logger.error("Recon loss: Cannot resolve shape mismatch. Returning high loss.")
                return torch.tensor(1000.0, device=recon_features.device, dtype=recon_features.dtype)
        
        return F.mse_loss(recon_features, target_features)

    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, frames_to_log: Optional[torch.Tensor], num_frames_per_sequence_to_log: int = 1, num_sequences_to_log_max: int = 2):
        if not (self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run and frames_to_log is not None and frames_to_log.numel() > 0):
            return
        
        B_log, N_seq_log, C_log, H_log, W_log = frames_to_log.shape
        num_to_actually_log_sequences = min(B_log, num_sequences_to_log_max)
        num_frames_to_log_this_seq = min(N_seq_log, num_frames_per_sequence_to_log)
        wandb_images_for_log = []

        for b_idx in range(num_to_actually_log_sequences):
            for frame_idx_in_seq in range(num_frames_to_log_this_seq):
                frame_tensor = frames_to_log[b_idx, frame_idx_in_seq, ...].cpu().float()
                # Assuming pixel values are in [-1,1], normalize to [0,1] for wandb.Image
                img_0_1 = (frame_tensor.clamp(-1,1) + 1) / 2.0
                caption = f"{tag_prefix} S{b_idx} F{frame_idx_in_seq} Ep{self.current_epoch+1} GStep{self.global_step}"
                wandb_images_for_log.append(wandb.Image(img_0_1, caption=caption))
        
        if wandb_images_for_log:
            wandb.log({f"samples/{tag_prefix}": wandb_images_for_log}, step=self.global_step)

    def _train_discriminator_step(self, real_frames_full: torch.Tensor, m_ref: "WuBuGAADHybridGenNet", d_ref: "RegionalDiscriminator") -> Dict[str, torch.Tensor]:
        B = real_frames_full.shape[0]
        device = real_frames_full.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0, device=device)).dtype # Get dtype from model

        frames_for_d_processing = real_frames_full[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
        
        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model)
        fake_labels = torch.zeros(B, 1, device=device, dtype=dtype_model)
        losses_d_micro: Dict[str, torch.Tensor] = {}

        # Set requires_grad for discriminator and generator/encoder
        for p in d_ref.parameters():
            p.requires_grad = True
        for p in m_ref.parameters():
            p.requires_grad = False
            
        gaad_bboxes_for_d_real_cond = None
        if d_ref.use_gaad_film_condition:
            with torch.no_grad(): # Encoder pass should not update model params here
                _, _, gaad_bboxes_from_encoder_full, _ = m_ref.encode(real_frames_full.to(device, dtype_model))
            if gaad_bboxes_from_encoder_full is not None:
                gaad_bboxes_for_d_real_cond = gaad_bboxes_from_encoder_full[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
        
        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            real_logits = d_ref(frames_for_d_processing, gaad_bboxes_for_d_real_cond)
            loss_d_real = self.adversarial_loss(real_logits, real_labels)

            with torch.no_grad(): # Generate fake frames, no grad for generator here
                fake_output_gen, _, _, bboxes_used_for_fake_gen, _ = m_ref(real_frames_full)
                
                fake_frames_pixels_full_sequence: torch.Tensor
                if self.args.use_dft_features_appearance:
                    B_gen, N_pred_gen, NumReg_gen, C_gen, _, H_dft_gen, W_dft_coeffs_gen = fake_output_gen.shape
                    fake_dft_coeffs_flat = fake_output_gen.reshape(B_gen * N_pred_gen * NumReg_gen, C_gen * 2 * H_dft_gen * W_dft_coeffs_gen)
                    
                    reconstructed_patches_flat = DFTUtils.reconstruct_patches_from_2d_dft(
                        fake_dft_coeffs_flat, self.args.dft_norm_scale_video,
                        m_ref.generator.num_img_channels, m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft
                    )
                    reconstructed_patches_structured = reconstructed_patches_flat.view(
                        B_gen, N_pred_gen, NumReg_gen, C_gen, m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft
                    )
                    fake_frames_pixels_full_sequence = ImageAssemblyUtils.assemble_frames_from_patches(
                        reconstructed_patches_structured, bboxes_used_for_fake_gen,
                        self.args.image_h_w_tuple, # type: ignore [attr-defined]
                        output_range=(-1.0, 1.0) 
                    )
                else: 
                    fake_frames_pixels_full_sequence = fake_output_gen

                fake_frames_for_d_processing = fake_frames_pixels_full_sequence[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
                gaad_bboxes_for_d_fake_cond = None
                if d_ref.use_gaad_film_condition and bboxes_used_for_fake_gen is not None:
                    gaad_bboxes_for_d_fake_cond = bboxes_used_for_fake_gen[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)

            fake_logits = d_ref(fake_frames_for_d_processing.detach(), gaad_bboxes_for_d_fake_cond) # Detach fake frames
            loss_d_fake = self.adversarial_loss(fake_logits, fake_labels)
            
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total_scaled_for_accum_micro = loss_d_total_micro / self.grad_accum_steps
            
        self.scaler_disc.scale(loss_d_total_scaled_for_accum_micro).backward()
        
        losses_d_micro['loss_d_real_micro'] = loss_d_real.detach()
        losses_d_micro['loss_d_fake_micro'] = loss_d_fake.detach()
        losses_d_micro['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d_micro

    def _train_generator_step(self, real_frames_full: torch.Tensor, m_ref: "WuBuGAADHybridGenNet", d_ref: "RegionalDiscriminator") -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        B = real_frames_full.shape[0]
        device = real_frames_full.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0, device=device)).dtype

        real_labels = torch.ones(B, 1, device=device, dtype=dtype_model)
        losses_g_micro: Dict[str, torch.Tensor] = {}
        recon_pixel_frames_for_log: Optional[torch.Tensor] = None

        # Set requires_grad for discriminator and generator/encoder
        for p in d_ref.parameters():
            p.requires_grad = False
        for p in m_ref.parameters():
            p.requires_grad = True

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            recon_output_gen, mu, logvar, bboxes_used_by_decoder, target_app_features_from_encoder = m_ref(real_frames_full.to(device, dtype_model))
            
            start_idx_target = self.video_config.get("num_input_frames", 0)
            num_predict_cfg = self.video_config.get("num_predict_frames", 1)

            loss_recon: torch.Tensor
            if self.args.use_dft_features_appearance:
                if target_app_features_from_encoder is None:
                     self.logger.error("G_Step DFT: target_app_features_from_encoder is None. Cannot compute DFT recon loss.")
                     loss_recon = torch.tensor(1000.0, device=device, dtype=dtype_model)
                else:
                    target_dft_for_loss = target_app_features_from_encoder[:, start_idx_target : start_idx_target + num_predict_cfg, ...]
                    B_rec, N_pred_rec, NumReg_rec, C_rec, _, H_dft_rec, W_dft_coeffs_rec = recon_output_gen.shape
                    recon_dft_flat_for_loss = recon_output_gen.reshape(B_rec, N_pred_rec, NumReg_rec, -1)
                    
                    comp_len_dft = min(recon_dft_flat_for_loss.shape[1], target_dft_for_loss.shape[1], num_predict_cfg)
                    if comp_len_dft <= 0:
                         self.logger.error(f"G_Step DFT: Cannot compute recon loss (comp_len_dft={comp_len_dft}).")
                         loss_recon = torch.tensor(1000.0, device=device, dtype=dtype_model)
                    else:
                         loss_recon = self._compute_recon_loss(
                             recon_dft_flat_for_loss[:, :comp_len_dft, ...],
                             target_dft_for_loss[:, :comp_len_dft, ...]
                         )
            else: # Pixel-based reconstruction
                actual_pred_len_for_loss = min(recon_output_gen.shape[1], real_frames_full.shape[1] - start_idx_target, num_predict_cfg)
                if actual_pred_len_for_loss <= 0:
                    self.logger.error(f"G_Step Pixel: Cannot compute recon loss (actual_pred_len={actual_pred_len_for_loss}).")
                    loss_recon = torch.tensor(1000.0, device=device, dtype=dtype_model)
                else:
                    target_frames_for_recon = real_frames_full[:, start_idx_target : start_idx_target + actual_pred_len_for_loss, ...].to(device, dtype_model)
                    recon_frames_for_loss_calc = recon_output_gen[:, :actual_pred_len_for_loss, ...]
                    loss_recon = self._compute_recon_loss(recon_frames_for_loss_calc, target_frames_for_recon)
            
            loss_kl = self._compute_kl_loss(mu, logvar)

            fake_pixel_frames_for_g_adv_full_seq: torch.Tensor
            if self.args.use_dft_features_appearance:
                B_g, N_pred_g, NumReg_g, C_g, _, H_dft_g, W_dft_coeffs_g = recon_output_gen.shape
                recon_dft_coeffs_flat_adv = recon_output_gen.reshape(B_g * N_pred_g * NumReg_g, C_g * 2 * H_dft_g * W_dft_coeffs_g)
                
                reconstructed_patches_flat_adv = DFTUtils.reconstruct_patches_from_2d_dft(
                    recon_dft_coeffs_flat_adv, self.args.dft_norm_scale_video,
                    m_ref.generator.num_img_channels, m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft
                )
                reconstructed_patches_structured_adv = reconstructed_patches_flat_adv.view(
                    B_g, N_pred_g, NumReg_g, C_g, m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft
                )
                fake_pixel_frames_for_g_adv_full_seq = ImageAssemblyUtils.assemble_frames_from_patches(
                    reconstructed_patches_structured_adv, bboxes_used_by_decoder,
                    self.args.image_h_w_tuple, # type: ignore [attr-defined]
                    output_range=(-1.0, 1.0)
                )
                if self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and \
                   ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0 or self.global_step == 0) :
                    recon_pixel_frames_for_log = fake_pixel_frames_for_g_adv_full_seq.detach().clone()
            else: 
                fake_pixel_frames_for_g_adv_full_seq = recon_output_gen
                if self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and \
                   ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0 or self.global_step == 0) :
                    recon_pixel_frames_for_log = fake_pixel_frames_for_g_adv_full_seq.detach().clone()

            fake_frames_for_g_adv_d_input = fake_pixel_frames_for_g_adv_full_seq[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
            gaad_bboxes_for_g_adv_cond = None
            if d_ref.use_gaad_film_condition and bboxes_used_by_decoder is not None:
                gaad_bboxes_for_g_adv_cond = bboxes_used_by_decoder[:, :d_ref.num_frames_to_discriminate, ...].to(device, dtype_model)
            
            fake_logits_gen = d_ref(fake_frames_for_g_adv_d_input, gaad_bboxes_for_g_adv_cond) # No detach here
            loss_g_adv = self.adversarial_loss(fake_logits_gen, real_labels)
            
            loss_g_total_micro = (self.lambda_recon * loss_recon + self.lambda_kl * loss_kl + self.lambda_gan * loss_g_adv)
            loss_g_total_scaled_for_accum_micro = loss_g_total_micro / self.grad_accum_steps
            
        self.scaler_enc_gen.scale(loss_g_total_scaled_for_accum_micro).backward()
        
        losses_g_micro['loss_recon_micro'] = loss_recon.detach()
        losses_g_micro['loss_kl_micro'] = loss_kl.detach()
        losses_g_micro['loss_g_adv_micro'] = loss_g_adv.detach()
        losses_g_micro['loss_g_total_micro'] = loss_g_total_micro.detach()
        return losses_g_micro, recon_pixel_frames_for_log

    def train(self, start_epoch:int=0, initial_global_step:int=0):
        self.global_step = initial_global_step
        self.current_epoch = start_epoch
        
        if self.am_main_process:
            self.logger.info(f"Starting training. DFT App: {self.args.use_dft_features_appearance}. Epochs: {self.args.epochs}, StartEpoch: {start_epoch}, GStep: {initial_global_step}, L_KL_init: {self.lambda_kl:.3e}")
        
        if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and self.args.num_val_samples_to_log > 0:
            m_ref_temp = self.model.module if self.ddp_active else self.model
            default_dtype = next(iter(m_ref_temp.parameters()), torch.tensor(0.0, device=self.device)).dtype
            self.fixed_noise_for_sampling = torch.randn(self.args.num_val_samples_to_log, self.args.latent_dim, device=self.device, dtype=default_dtype)
            self.logger.info(f"Created fixed noise tensor: {self.fixed_noise_for_sampling.shape}")
            
        accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
        accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0
        log_interval_accum_losses = defaultdict(float)
        log_interval_items_processed = 0
        
        m_ref = self.model.module if self.ddp_active else self.model
        d_ref = self.discriminator.module if self.ddp_active else self.discriminator
        
        opt_gen_q_controller = getattr(self.optimizer_enc_gen, 'q_controller', None)
        opt_disc_q_controller = getattr(self.optimizer_disc, 'q_controller', None)
        lambda_kl_q_ctrl_ref = self.lambda_kl_q_controller
        
        if opt_gen_q_controller and hasattr(opt_gen_q_controller, 'set_current_lambda_kl'):
            opt_gen_q_controller.set_current_lambda_kl(self.lambda_kl)
        if opt_disc_q_controller and hasattr(opt_disc_q_controller, 'set_current_lambda_kl'):
            opt_disc_q_controller.set_current_lambda_kl(self.lambda_kl)
            
        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process:
                self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} (lambda_kl: {self.lambda_kl:.3e})...")
            if self.ddp_active and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch) # type: ignore
                
            m_ref.train()
            d_ref.train()
            self.optimizer_disc.zero_grad(set_to_none=True)
            self.optimizer_enc_gen.zero_grad(set_to_none=True)
            
            num_batches_epoch = len(self.train_loader)
            prog_bar = tqdm(self.train_loader, desc=f"E{epoch+1}", disable=not self.am_main_process or os.getenv('CI') == 'true', dynamic_ncols=True, total=num_batches_epoch)
            
            for batch_idx, batch_frames_raw in enumerate(prog_bar):
                batch_frames = batch_frames_raw.to(self.device)
                batch_size_micro = batch_frames.size(0)

                # --- Train Discriminator ---
                losses_d_micro = self._train_discriminator_step(batch_frames, m_ref, d_ref)
                if torch.isfinite(losses_d_micro['loss_d_total_micro']):
                    d_total_val = losses_d_micro['loss_d_total_micro'].item()
                    accum_d_total_q += d_total_val
                    self.interval_metrics_accum['d_total'] += d_total_val
                if torch.isfinite(losses_d_micro['loss_d_real_micro']):
                    accum_d_real_q += losses_d_micro['loss_d_real_micro'].item()
                if torch.isfinite(losses_d_micro['loss_d_fake_micro']):
                    accum_d_fake_q += losses_d_micro['loss_d_fake_micro'].item()
                for k,v_tensor in losses_d_micro.items():
                    if torch.isfinite(v_tensor):
                        log_interval_accum_losses[k.replace('_micro','_agg')] += v_tensor.item() * batch_size_micro
                
                # --- Train Generator ---
                losses_g_micro, recon_pixel_frames_for_logging = self._train_generator_step(batch_frames, m_ref, d_ref)
                if torch.isfinite(losses_g_micro['loss_g_total_micro']):
                    accum_g_total_q += losses_g_micro['loss_g_total_micro'].item()
                if torch.isfinite(losses_g_micro['loss_recon_micro']):
                    recon_val = losses_g_micro['loss_recon_micro'].item()
                    accum_g_recon_q += recon_val
                    self.interval_metrics_accum['recon'] += recon_val
                if torch.isfinite(losses_g_micro['loss_kl_micro']):
                    kl_val = losses_g_micro['loss_kl_micro'].item()
                    accum_g_kl_q += kl_val
                    self.interval_metrics_accum['kl_div'] += kl_val
                if torch.isfinite(losses_g_micro['loss_g_adv_micro']):
                    accum_g_adv_q += losses_g_micro['loss_g_adv_micro'].item()
                for k,v_tensor in losses_g_micro.items():
                    if torch.isfinite(v_tensor):
                        log_interval_accum_losses[k.replace('_micro','_agg')] += v_tensor.item() * batch_size_micro
                
                log_interval_items_processed += batch_size_micro
                self.interval_steps_count += 1
                
                # --- Optimizer Step after grad_accum_steps ---
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if hasattr(self.optimizer_disc, 'grad_stats'):
                        self.optimizer_disc.grad_stats.finalize_step_stats(sum(p.numel() for grp in self.optimizer_disc.param_groups for p in grp['params'] if p.requires_grad)) # type: ignore
                    if hasattr(self.optimizer_enc_gen, 'grad_stats'):
                        self.optimizer_enc_gen.grad_stats.finalize_step_stats(sum(p.numel() for grp in self.optimizer_enc_gen.param_groups for p in grp['params'] if p.requires_grad)) # type: ignore

                    avg_g_total_macro = accum_g_total_q / self.grad_accum_steps
                    avg_g_recon_macro = accum_g_recon_q / self.grad_accum_steps
                    avg_g_kl_macro = accum_g_kl_q / self.grad_accum_steps
                    avg_g_adv_macro = accum_g_adv_q / self.grad_accum_steps
                    avg_d_total_macro = accum_d_total_q / self.grad_accum_steps
                    avg_d_real_macro = accum_d_real_q / self.grad_accum_steps
                    avg_d_fake_macro = accum_d_fake_q / self.grad_accum_steps
                    
                    # Discriminator optimizer step
                    for p in d_ref.parameters(): p.requires_grad = True
                    for p in m_ref.parameters(): p.requires_grad = False
                    if opt_disc_q_controller and hasattr(self.optimizer_disc, 'q_controller_update_and_set_hyperparams'):
                        losses_d_q_lr_mom = {'loss_g_total': avg_g_total_macro, 'loss_g_adv': avg_g_adv_macro, 'loss_d_total': avg_d_total_macro, 'loss_d_real': avg_d_real_macro, 'loss_d_fake': avg_d_fake_macro}
                        self.optimizer_disc.q_controller_update_and_set_hyperparams(losses_d_q_lr_mom, self.lambda_kl) # type: ignore
                    if self.args.global_max_grad_norm > 0:
                        self.scaler_disc.unscale_(self.optimizer_disc)
                        torch.nn.utils.clip_grad_norm_(d_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_disc.step(self.optimizer_disc)
                    self.scaler_disc.update()
                    
                    # Generator optimizer step
                    for p in d_ref.parameters(): p.requires_grad = False
                    for p in m_ref.parameters(): p.requires_grad = True
                    if opt_gen_q_controller and hasattr(self.optimizer_enc_gen, 'q_controller_update_and_set_hyperparams'):
                        losses_g_q_lr_mom = {'loss_g_total': avg_g_total_macro, 'loss_g_recon': avg_g_recon_macro, 'loss_g_kl': avg_g_kl_macro, 'loss_g_adv': avg_g_adv_macro, 'loss_d_total': avg_d_total_macro}
                        self.optimizer_enc_gen.q_controller_update_and_set_hyperparams(losses_g_q_lr_mom, self.lambda_kl) # type: ignore
                    if self.args.global_max_grad_norm > 0:
                        self.scaler_enc_gen.unscale_(self.optimizer_enc_gen)
                        torch.nn.utils.clip_grad_norm_(m_ref.parameters(), self.args.global_max_grad_norm)
                    self.scaler_enc_gen.step(self.optimizer_enc_gen)
                    self.scaler_enc_gen.update()
                    
                    self.optimizer_disc.zero_grad(set_to_none=True)
                    self.optimizer_enc_gen.zero_grad(set_to_none=True)
                    self.global_step += 1
                    
                    # Reset accumulators for Q-controller
                    accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
                    accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0
                    
                    # Lambda_KL Q-Controller Update
                    if lambda_kl_q_ctrl_ref is not None and self.lambda_kl_update_interval > 0 and \
                       self.global_step > 0 and self.global_step % self.lambda_kl_update_interval == 0 and \
                       self.interval_steps_count > 0 :
                        
                        current_interval_metrics: Dict[str, Union[float, None]] = {
                            'avg_recon': self.interval_metrics_accum['recon'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_kl_div': self.interval_metrics_accum['kl_div'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_d_total': self.interval_metrics_accum['d_total'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'val_metric': self.last_val_metrics.get(self.args.val_primary_metric), # From last validation
                            'current_lambda_kl_val': self.lambda_kl
                        }
                        if self.am_main_process:
                            log_mets_str = {k: (f'{v:.4f}' if isinstance(v, float) else str(v)) for k, v in current_interval_metrics.items()}
                            self.logger.info(f"GStep {self.global_step}: LKL_QCtrl. lambda_kl: {self.lambda_kl:.4e}. IntervalMets: {log_mets_str}. PrevLKLState: {lambda_kl_q_ctrl_ref.prev_lambda_kl_state}. PrevLKLAction: {lambda_kl_q_ctrl_ref.prev_lambda_kl_action}")
                        
                        q_state_lambda_kl = lambda_kl_q_ctrl_ref.get_lambda_kl_state(current_interval_metrics)
                        if self.am_main_process and q_state_lambda_kl is not None:
                             q_vals_str = str(lambda_kl_q_ctrl_ref.q_table.get(q_state_lambda_kl, {}).get('lambda_kl_scale'))
                             self.logger.info(f"  New LKL Q-State: {q_state_lambda_kl}. QVals(LKL_scale): {q_vals_str}")
                        
                        if lambda_kl_q_ctrl_ref.prev_lambda_kl_state is not None and \
                           lambda_kl_q_ctrl_ref.prev_lambda_kl_action is not None and \
                           q_state_lambda_kl is not None and \
                           self.prev_interval_metrics_for_lambda_kl_reward is not None:
                            reward_for_lambda_kl = lambda_kl_q_ctrl_ref.compute_lambda_kl_reward(current_interval_metrics, self.prev_interval_metrics_for_lambda_kl_reward)
                            if self.am_main_process: self.logger.info(f"  LKL Q-Ctrl reward: {reward_for_lambda_kl:.3f}")
                            lambda_kl_q_ctrl_ref.update_q_values(lambda_kl_q_ctrl_ref.prev_lambda_kl_state, lambda_kl_q_ctrl_ref.prev_lambda_kl_action, reward_for_lambda_kl, q_state_lambda_kl, mode='lambda_kl')
                        elif q_state_lambda_kl is not None and hasattr(lambda_kl_q_ctrl_ref, 'set_initial_lambda_kl_metrics'):
                            lambda_kl_q_ctrl_ref.set_initial_lambda_kl_metrics(current_interval_metrics) # type: ignore
                        
                        if q_state_lambda_kl is not None:
                            lambda_kl_action_dict = lambda_kl_q_ctrl_ref.choose_action(q_state_lambda_kl, mode='lambda_kl')
                            chosen_scale = lambda_kl_action_dict.get('lambda_kl_scale', 1.0)
                            if self.am_main_process: self.logger.info(f"  LKL Q-Ctrl CHOSE scale: {chosen_scale:.2f} (Eps: {lambda_kl_q_ctrl_ref.epsilon:.3f})")
                            
                            self.prev_interval_metrics_for_lambda_kl_reward = current_interval_metrics.copy()
                            new_lambda_kl_val = self.lambda_kl * chosen_scale
                            self.lambda_kl = float(np.clip(new_lambda_kl_val, self.min_lambda_kl_q_control, self.max_lambda_kl_q_control))
                            if self.am_main_process: self.logger.info(f"GStep {self.global_step}: LKL_QCtrl updated trainer's self.lambda_kl to {self.lambda_kl:.4e} (scale: {chosen_scale:.2f})")
                            
                            lambda_kl_q_ctrl_ref.prev_lambda_kl_state = q_state_lambda_kl
                            lambda_kl_q_ctrl_ref.prev_lambda_kl_action = lambda_kl_action_dict

                        # Update lambda_kl in other controllers if they use it
                        newly_set_lambda_kl_for_q_ctrl = self.lambda_kl
                        if opt_gen_q_controller and hasattr(opt_gen_q_controller, 'set_current_lambda_kl'):
                            opt_gen_q_controller.set_current_lambda_kl(newly_set_lambda_kl_for_q_ctrl)
                        if opt_disc_q_controller and hasattr(opt_disc_q_controller, 'set_current_lambda_kl'):
                            opt_disc_q_controller.set_current_lambda_kl(newly_set_lambda_kl_for_q_ctrl)
                        if lambda_kl_q_ctrl_ref and hasattr(lambda_kl_q_ctrl_ref, 'set_current_lambda_kl'):
                            lambda_kl_q_ctrl_ref.set_current_lambda_kl(newly_set_lambda_kl_for_q_ctrl)
                            
                        self.interval_metrics_accum = defaultdict(float)
                        self.interval_steps_count = 0
                        
                    # Logging
                    if self.global_step > 0 and self.global_step % self.args.log_interval == 0 and log_interval_items_processed > 0 and self.am_main_process:
                        log_metrics_train = {f"train/{k.replace('_agg','')}":v/log_interval_items_processed for k,v in log_interval_accum_losses.items()}
                        lr_g = self.optimizer_enc_gen.param_groups[0]['lr']
                        lr_d = self.optimizer_disc.param_groups[0]['lr']
                        log_metrics_train.update({"train/lr_gen":lr_g,"train/lr_disc":lr_d,"epoch_frac":epoch+((batch_idx+1)/(num_batches_epoch or 1)),"global_step":self.global_step,"train/lambda_kl_eff":self.lambda_kl})
                        
                        if opt_gen_q_controller and hasattr(opt_gen_q_controller, 'get_info'):
                            log_metrics_train.update({f"q_ctrl_gen/{k.replace('_','')}":v for k,v in opt_gen_q_controller.get_info().items()})
                        if opt_disc_q_controller and hasattr(opt_disc_q_controller, 'get_info'):
                            log_metrics_train.update({f"q_ctrl_disc/{k.replace('_','')}":v for k,v in opt_disc_q_controller.get_info().items()})
                        if lambda_kl_q_ctrl_ref and hasattr(lambda_kl_q_ctrl_ref, 'get_info'):
                            log_metrics_train.update({f"q_ctrl_lkl/{k.replace('_','')}":v for k,v in lambda_kl_q_ctrl_ref.get_info().items()})
                            
                        gt = log_metrics_train.get('train/loss_g_total',-1.0); dt = log_metrics_train.get('train/loss_d_total',-1.0)
                        gr = log_metrics_train.get('train/loss_recon',-1.0); gk = log_metrics_train.get('train/loss_kl',-1.0)
                        ga = log_metrics_train.get('train/loss_g_adv',-1.0); dr = log_metrics_train.get('train/loss_d_real',-1.0); df = log_metrics_train.get('train/loss_d_fake',-1.0)
                        
                        qeg_lrmom_eps = log_metrics_train.get('q_ctrl_gen/epsilon',-1.0); qed_lrmom_eps = log_metrics_train.get('q_ctrl_disc/epsilon',-1.0)
                        qelkl_eps = log_metrics_train.get('q_ctrl_lkl/epsilon', -1.0)
                        
                        qag_lrmom_act = log_metrics_train.get('q_ctrl_gen/lastlrmomaction',{}); qad_lrmom_act = log_metrics_train.get('q_ctrl_disc/lastlrmomaction',{})
                        qal_lkl_act = log_metrics_train.get('q_ctrl_lkl/lastlambdaklaction',{})
                        
                        qslg_lr = qag_lrmom_act.get('lr_scale',1.0) if isinstance(qag_lrmom_act,dict) else 1.0
                        qsld_lr = qad_lrmom_act.get('lr_scale',1.0) if isinstance(qad_lrmom_act,dict) else 1.0
                        qslkl_lkl = qal_lkl_act.get('lambda_kl_scale',1.0) if isinstance(qal_lkl_act,dict) else 1.0
                        
                        log_str=(f"E{epoch+1} S{self.global_step} | G_tot:{gt:.3f}(Rec:{gr:.3f} KL:{gk:.3f} Adv:{ga:.3f}) | D_tot:{dt:.3f}(R:{dr:.3f} F:{df:.3f}) | "
                                 f"LR(G/D):{lr_g:.1e}/{lr_d:.1e} | Q_Eps(LRM G/D):{qeg_lrmom_eps:.2f}/{qed_lrmom_eps:.2f} Q_Scl(LRM G/D):{qslg_lr:.2f}/{qsld_lr:.2f} | "
                                 f"Q_Eps(LKL):{qelkl_eps:.2f} Q_Scl(LKL):{qslkl_lkl:.2f} LKL_eff:{self.lambda_kl:.2e}")
                        
                        prog_bar.set_postfix_str(f"G:{gt:.2f} D:{dt:.2f} Rec:{gr:.3f} LKL:{self.lambda_kl:.1e}",refresh=True)
                        self.logger.info(log_str)
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                            wandb.log(log_metrics_train, step=self.global_step)
                            
                        log_interval_accum_losses = defaultdict(float)
                        log_interval_items_processed = 0
                    
                    # Log reconstructed training samples
                    if self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and self.global_step > 0 and \
                       self.global_step % self.args.wandb_log_train_recon_interval == 0 and recon_pixel_frames_for_logging is not None:
                        
                        self._log_samples_to_wandb("train_recon", recon_pixel_frames_for_logging, 
                                                   recon_pixel_frames_for_logging.shape[1], self.args.num_val_samples_to_log)
                        
                        num_input_frames = self.video_config.get("num_input_frames", 0)
                        self._log_samples_to_wandb("train_context", batch_frames[:, :num_input_frames, ...], 
                                                   num_input_frames, self.args.num_val_samples_to_log)
                        
                        s_idx_target_log = num_input_frames
                        gt_len_log = min(self.video_config.get("num_predict_frames", 1), batch_frames.shape[1] - s_idx_target_log)
                        if gt_len_log > 0:
                            self._log_samples_to_wandb("train_ground_truth", batch_frames[:, s_idx_target_log : s_idx_target_log + gt_len_log, ...], 
                                                       gt_len_log, self.args.num_val_samples_to_log)
                    
                    # Log samples from fixed noise
                    if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and self.global_step > 0 and \
                       self.global_step % self.args.wandb_log_fixed_noise_samples_interval == 0 and self.fixed_noise_for_sampling is not None:
                        m_ref.eval()
                        with torch.no_grad():
                            num_fs = self.fixed_noise_for_sampling.shape[0]
                            npf_cfg = self.video_config.get("num_predict_frames",1)
                            nra_cfg = self.gaad_appearance_config.get("num_regions",0)
                            fd_cfg = self.args.image_h_w_tuple # type: ignore [attr-defined]
                            
                            dbb_list_fixed = []
                            if nra_cfg > 0:
                                for _ in range(num_fs):
                                    dbb_list_fixed.append(golden_subdivide_rect_fixed_n(fd_cfg, nra_cfg, 
                                                                                       device=self.device, dtype=self.fixed_noise_for_sampling.dtype, 
                                                                                       min_size_px=self.gaad_appearance_config.get('min_size_px',5)
                                                                                      ).unsqueeze(0).repeat(npf_cfg,1,1)) # (1,Nframes,Nregions,4) -> (Nframes,Nregions,4)
                            dbb_batch_fixed = torch.stack(dbb_list_fixed) if dbb_list_fixed else None
                            
                            if dbb_batch_fixed is not None or nra_cfg == 0: # Proceed if bboxes generated or not needed
                                gen_output_fixed = m_ref.decode(self.fixed_noise_for_sampling, dbb_batch_fixed)
                                fixed_noise_samples_pixels: torch.Tensor
                                if self.args.use_dft_features_appearance:
                                    B_fix, N_fix, R_fix, C_fix, _, H_dft_fix, W_dftc_fix = gen_output_fixed.shape
                                    gen_dft_flat_fix = gen_output_fixed.reshape(B_fix*N_fix*R_fix, C_fix*2*H_dft_fix*W_dftc_fix)
                                    patches_flat_fix = DFTUtils.reconstruct_patches_from_2d_dft(
                                        gen_dft_flat_fix, self.args.dft_norm_scale_video, 
                                        m_ref.generator.num_img_channels, m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft
                                    )
                                    patches_struct_fix = patches_flat_fix.view(B_fix,N_fix,R_fix,C_fix, m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft)
                                    fixed_noise_samples_pixels = ImageAssemblyUtils.assemble_frames_from_patches(
                                        patches_struct_fix, dbb_batch_fixed, fd_cfg, output_range=(-1.0,1.0)
                                    )
                                else:
                                    fixed_noise_samples_pixels = gen_output_fixed
                                
                                self._log_samples_to_wandb("fixed_noise_generated", fixed_noise_samples_pixels, 
                                                           fixed_noise_samples_pixels.shape[1], num_fs)
                        m_ref.train() # Set back to train mode
                        
                    # Intermediate checkpoint saving
                    if self.args.save_interval > 0 and self.global_step > 0 and self.global_step % self.args.save_interval == 0 and self.am_main_process:
                        self._save_checkpoint(is_intermediate=True,metrics={'train_loss_g_total_macro':avg_g_total_macro,'train_loss_d_total_macro':avg_d_total_macro})
            
            # End of Epoch
            if self.am_main_process:
                final_avg_g_loss = log_interval_accum_losses['loss_g_total_agg'] / log_interval_items_processed if log_interval_items_processed > 0 else (log_metrics_train.get('train/loss_g_total', float('nan')) if 'log_metrics_train' in locals() else float('nan')) # type: ignore
                final_avg_d_loss = log_interval_accum_losses['loss_d_total_agg'] / log_interval_items_processed if log_interval_items_processed > 0 else (log_metrics_train.get('train/loss_d_total', float('nan')) if 'log_metrics_train' in locals() else float('nan')) # type: ignore
                self.logger.info(f"Epoch {epoch+1} finished. Approx Avg Loss (G/D): {final_avg_g_loss:.4f}/{final_avg_d_loss:.4f}, LKL_eff:{self.lambda_kl:.3e}")
                if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                    wandb.log({"epoch":epoch+1, 
                               "epoch_avg_train_loss_g_approx":final_avg_g_loss if np.isfinite(final_avg_g_loss) else -1.0, 
                               "epoch_avg_train_loss_d_approx":final_avg_d_loss if np.isfinite(final_avg_d_loss) else -1.0, 
                               "epoch_lambda_kl": self.lambda_kl}, 
                              step=self.global_step)
                              
            # Validation
            if self.val_loader and self.am_main_process:
                val_metrics = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if val_metrics:
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                         wandb.log({f"val/{k}":v for k,v in val_metrics.items()}, step=self.global_step)
                    
                    metric_to_check = self.args.val_primary_metric
                    default_val = float('inf') if metric_to_check not in ["avg_val_psnr","avg_val_ssim"] else -float('inf')
                    current_val_for_best = val_metrics.get(metric_to_check, default_val)
                    
                    is_better = (current_val_for_best > self.best_val_metric_val) if metric_to_check in ["avg_val_psnr","avg_val_ssim"] else (current_val_for_best < self.best_val_metric_val)
                    if is_better and np.isfinite(current_val_for_best):
                        self.logger.info(f"New best val metric ({metric_to_check}): {current_val_for_best:.4f} (prev: {self.best_val_metric_val:.4f}). Saving best checkpoint.")
                        self.best_val_metric_val = current_val_for_best
                        self._save_checkpoint(is_best=True, metrics=val_metrics)
            
            # Save regular checkpoint at end of epoch if not intermediate-saved
            if self.am_main_process:
                epoch_end_metrics = self.last_val_metrics.copy() if self.last_val_metrics else {}
                if 'final_avg_g_loss' in locals() and np.isfinite(final_avg_g_loss): # type: ignore
                    epoch_end_metrics["epoch_end_train_loss_g_avg_approx"] = final_avg_g_loss # type: ignore
                if 'final_avg_d_loss' in locals() and np.isfinite(final_avg_d_loss): # type: ignore
                    epoch_end_metrics["epoch_end_train_loss_d_avg_approx"] = final_avg_d_loss # type: ignore
                self._save_checkpoint(metrics=epoch_end_metrics) # Saves last/epoch checkpoint


    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Optional[Dict[str, float]]:
        if not self.val_loader or not self.am_main_process:
            return None
        
        m_ref = self.model.module if self.ddp_active else self.model
        m_ref.eval() # Set model to evaluation mode
        
        total_recon_mse_sum = 0.0
        total_psnr_sum = 0.0
        total_ssim_sum = 0.0
        total_lpips_sum = 0.0
        total_comp_frames_pixel_metric = 0 # For PSNR, SSIM, LPIPS
        total_comp_dft_regions = 0 # For DFT MSE
        
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0, device=self.device)).dtype
        logged_samples_count_this_val = 0

        for batch_idx_val, batch_frames_raw in enumerate(tqdm(self.val_loader, desc="Validating", disable=not self.am_main_process or os.getenv('CI') == 'true', dynamic_ncols=True)):
            real_full_pixel = batch_frames_raw.to(self.device, dtype_m)
            B,N_total_sample,C_img,H_img,W_img = real_full_pixel.shape
            
            recon_output_gen, _, _, bboxes_used_by_decoder, target_app_features_from_encoder = m_ref(real_full_pixel)
            
            n_cond = self.video_config.get("num_input_frames",0)
            n_pred_cfg = self.video_config.get("num_predict_frames",1)

            pred_pixels_val: torch.Tensor # To store pixel-space predictions
            
            if self.args.use_dft_features_appearance:
                if target_app_features_from_encoder is None:
                    self.logger.warning("Val DFT: target_app_features_from_encoder is None. Skipping batch.")
                    continue
                
                target_dft_for_loss = target_app_features_from_encoder[:, n_cond : n_cond + n_pred_cfg, ...]
                recon_dft_flat_for_loss = recon_output_gen.reshape(B, recon_output_gen.shape[1], recon_output_gen.shape[2], -1)

                comp_len_dft = min(recon_dft_flat_for_loss.shape[1], target_dft_for_loss.shape[1], n_pred_cfg)
                if comp_len_dft <= 0: continue

                mse_batch_dft = F.mse_loss(
                    recon_dft_flat_for_loss[:, :comp_len_dft, ...],
                    target_dft_for_loss[:, :comp_len_dft, ...]
                )
                if torch.isfinite(mse_batch_dft):
                    total_recon_mse_sum += mse_batch_dft.item() * B * comp_len_dft * recon_dft_flat_for_loss.shape[2]
                total_comp_dft_regions += B * comp_len_dft * recon_dft_flat_for_loss.shape[2]
                
                B_dft, N_pred_dft_actual, R_dft, C_dft, _, H_dft_patch, W_dftc = recon_output_gen.shape
                dft_coeffs_flat_val = recon_output_gen.reshape(B_dft * N_pred_dft_actual * R_dft, C_dft * 2 * H_dft_patch * W_dftc)
                patches_flat_val = DFTUtils.reconstruct_patches_from_2d_dft(
                    dft_coeffs_flat_val, self.args.dft_norm_scale_video, 
                    C_img, # Assuming generator.num_img_channels == C_img
                    m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft
                )
                patches_struct_val = patches_flat_val.view(B_dft, N_pred_dft_actual, R_dft, C_img, m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft)
                
                pred_pixels_val = ImageAssemblyUtils.assemble_frames_from_patches(
                    patches_struct_val[:, :comp_len_dft, ...], 
                    bboxes_used_by_decoder[:, :comp_len_dft, ...], # Ensure bboxes match the length used for patches
                    self.args.image_h_w_tuple, # type: ignore [attr-defined]
                    output_range=(-1.0, 1.0)
                )
            else: 
                pred_pixels_val = recon_output_gen
            
            # Ground truth for pixel metrics
            # Ensure gt_pixels_val matches the number of predicted frames from pred_pixels_val
            num_predicted_frames_for_metrics = pred_pixels_val.shape[1]
            gt_pixels_val = real_full_pixel[:, n_cond : n_cond + num_predicted_frames_for_metrics, ...]
            comp_len_pixels = gt_pixels_val.shape[1] # Should be same as num_predicted_frames_for_metrics

            if comp_len_pixels <=0: continue

            # Normalize to [0,1] for pixel metrics if they are in [-1,1]
            pred_01 = (pred_pixels_val.clamp(-1,1)+1)/2.0
            gt_01 = (gt_pixels_val.clamp(-1,1)+1)/2.0
            
            # Flatten frames for metric calculation (B*N_pred, C, H, W)
            pred_flat_pix = pred_01.reshape(-1,C_img,H_img,W_img)
            gt_flat_pix = gt_01.reshape(-1,C_img,H_img,W_img)
            curr_batch_flat_pixel_frames = pred_flat_pix.shape[0]

            if not self.args.use_dft_features_appearance: # If pixel VAE, MSE is on pixels
                 mse_batch_pixel = F.mse_loss(pred_flat_pix, gt_flat_pix, reduction='mean') # MSE on [0,1] pixels
                 if torch.isfinite(mse_batch_pixel):
                     total_recon_mse_sum += mse_batch_pixel.item() * curr_batch_flat_pixel_frames

            # PSNR (on [0,1] pixels)
            mse_for_psnr = F.mse_loss(pred_flat_pix, gt_flat_pix).item()
            psnr_batch_avg_pix = 10 * math.log10(1.0 / (mse_for_psnr + EPS)) if mse_for_psnr > EPS else 100.0
            if np.isfinite(psnr_batch_avg_pix):
                total_psnr_sum += psnr_batch_avg_pix * curr_batch_flat_pixel_frames
            
            if self.ssim_metric:
                try:
                    ssim_batch_avg_pix = self.ssim_metric(pred_flat_pix, gt_flat_pix) # SSIM expects [0,1]
                    if torch.isfinite(ssim_batch_avg_pix):
                         total_ssim_sum += ssim_batch_avg_pix.item() * curr_batch_flat_pixel_frames
                except Exception as e_ssim:
                    self.logger.debug(f"SSIM calculation failed in validation: {e_ssim}")
            if self.lpips_loss_fn:
                try:
                    # LPIPS expects images in [-1,1] range
                    lpips_batch_frames_pix = self.lpips_loss_fn(pred_pixels_val.reshape(-1,C_img,H_img,W_img), 
                                                                gt_pixels_val.reshape(-1,C_img,H_img,W_img)) # Use original [-1,1]
                    if torch.isfinite(lpips_batch_frames_pix.sum()):
                        total_lpips_sum += lpips_batch_frames_pix.sum().item() # LPIPS is per image, sum then average later
                except Exception as e_lpips:
                    self.logger.debug(f"LPIPS calculation failed in validation: {e_lpips}")
            total_comp_frames_pixel_metric += curr_batch_flat_pixel_frames
            
            # Log samples to WandB during validation
            if logged_samples_count_this_val < num_val_samples_to_log and self.args.wandb and WANDB_AVAILABLE and wandb.run:
                num_seq_log_batch = min(B, num_val_samples_to_log - logged_samples_count_this_val)
                num_f_log_seq = min(comp_len_pixels, 3) # Log up to 3 frames per sequence
                if num_seq_log_batch > 0 and num_f_log_seq > 0:
                    self._log_samples_to_wandb("val_context", real_full_pixel[:num_seq_log_batch, :n_cond, ...], min(n_cond, 3), num_seq_log_batch)
                    self._log_samples_to_wandb("val_predicted", pred_pixels_val[:num_seq_log_batch, :num_f_log_seq, ...], num_f_log_seq, num_seq_log_batch)
                    self._log_samples_to_wandb("val_ground_truth", gt_pixels_val[:num_seq_log_batch, :num_f_log_seq, ...], num_f_log_seq, num_seq_log_batch)
                logged_samples_count_this_val += num_seq_log_batch
        
        m_ref.train() # Set model back to training mode
        
        avg_mse_key = "avg_val_recon_mse_dft" if self.args.use_dft_features_appearance else "avg_val_recon_mse_pixel"
        avg_mse_val = (total_recon_mse_sum / total_comp_dft_regions if self.args.use_dft_features_appearance and total_comp_dft_regions > 0
                       else total_recon_mse_sum / total_comp_frames_pixel_metric if not self.args.use_dft_features_appearance and total_comp_frames_pixel_metric > 0
                       else float('inf'))
        
        avg_psnr_val = total_psnr_sum / total_comp_frames_pixel_metric if total_comp_frames_pixel_metric > 0 else 0.0
        avg_ssim_val = total_ssim_sum / total_comp_frames_pixel_metric if total_comp_frames_pixel_metric > 0 and self.ssim_metric else 0.0
        avg_lpips_val = total_lpips_sum / total_comp_frames_pixel_metric if total_comp_frames_pixel_metric > 0 and self.lpips_loss_fn else float('inf')
        
        metrics={avg_mse_key: avg_mse_val, "avg_val_psnr":avg_psnr_val, "avg_val_ssim":avg_ssim_val, "avg_val_lpips":avg_lpips_val}
        self.last_val_metrics = metrics
        self.logger.info(f"Validation Metrics (Ep {self.current_epoch+1}, GStep {self.global_step}): {avg_mse_key}:{avg_mse_val:.4f}, PSNR:{avg_psnr_val:.2f}, SSIM:{avg_ssim_val:.4f}, LPIPS:{avg_lpips_val:.4f}")
        return metrics

    def _save_checkpoint(self, is_intermediate: bool=False, metrics:Optional[Dict[str, Any]]=None, is_best:bool=False):
        if not self.am_main_process: return
        m_s = self.model.module if self.ddp_active else self.model
        d_s = self.discriminator.module if self.ddp_active else self.discriminator
        
        data = {
            'global_step': self.global_step, 
            'epoch': self.current_epoch,
            'model_state_dict': m_s.state_dict(), 
            'discriminator_state_dict': d_s.state_dict(),
            'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict() if self.args.use_amp and self.device.type == 'cuda' else None,
            'scaler_disc_state_dict': self.scaler_disc.state_dict() if self.args.use_amp and self.device.type == 'cuda' else None,
            'args': vars(self.args), 
            'metrics': metrics if metrics else self.last_val_metrics,
            'video_config': self.video_config, # Added back video_config
            'best_val_metric_val': self.best_val_metric_val,
            'current_lambda_kl': self.lambda_kl # Save current lambda_kl
        }
        
        qg_obj = getattr(self.optimizer_enc_gen,'q_controller',None)
        qd_obj = getattr(self.optimizer_disc,'q_controller',None)
        
        if qg_obj and hasattr(qg_obj,'q_table'):
            data['q_controller_enc_gen_state'] = {
                'q_table': qg_obj.q_table, 'epsilon': qg_obj.epsilon,
                'prev_lr_mom_state': qg_obj.prev_lr_mom_state, 'prev_lr_mom_action': qg_obj.prev_lr_mom_action,
                'loss_histories': {
                    'g_total': list(qg_obj.loss_g_total_hist), 'g_recon': list(qg_obj.loss_g_recon_hist),
                    'g_kl': list(qg_obj.loss_g_kl_hist), 'g_adv': list(qg_obj.loss_g_adv_hist),
                    'd_total': list(qg_obj.loss_d_total_hist), 'd_real': list(qg_obj.loss_d_real_hist),
                    'd_fake': list(qg_obj.loss_d_fake_hist)
                }
            }
        if qd_obj and hasattr(qd_obj,'q_table'):
            data['q_controller_disc_state'] = {
                'q_table': qd_obj.q_table, 'epsilon': qd_obj.epsilon,
                'prev_lr_mom_state': qd_obj.prev_lr_mom_state, 'prev_lr_mom_action': qd_obj.prev_lr_mom_action,
                 'loss_histories': {
                    'g_total': list(qd_obj.loss_g_total_hist), 'g_recon': list(qd_obj.loss_g_recon_hist),
                    'g_kl': list(qd_obj.loss_g_kl_hist), 'g_adv': list(qd_obj.loss_g_adv_hist),
                    'd_total': list(qd_obj.loss_d_total_hist), 'd_real': list(qd_obj.loss_d_real_hist),
                    'd_fake': list(qd_obj.loss_d_fake_hist)
                }
            }
        if self.lambda_kl_q_controller and hasattr(self.lambda_kl_q_controller, 'q_table'):
            data['q_controller_lambda_kl_state'] = {
                'q_table': self.lambda_kl_q_controller.q_table,
                'epsilon': self.lambda_kl_q_controller.epsilon,
                'prev_lambda_kl_state': self.lambda_kl_q_controller.prev_lambda_kl_state,
                'prev_lambda_kl_action': self.lambda_kl_q_controller.prev_lambda_kl_action,
                'interval_histories': {
                    'avg_recon': list(self.lambda_kl_q_controller.interval_avg_recon_hist),
                    'avg_kl_div': list(self.lambda_kl_q_controller.interval_avg_kl_div_hist),
                    'avg_d_total': list(self.lambda_kl_q_controller.interval_avg_d_total_hist),
                    'val_metric': list(self.lambda_kl_q_controller.interval_val_metric_hist)
                },
                'reward_hist': list(self.lambda_kl_q_controller.reward_hist),
                'q_table_access_count': dict(self.lambda_kl_q_controller.q_table_access_count),
                'q_table_creation_time': self.lambda_kl_q_controller.q_table_creation_time,
                'q_table_last_access_time': self.lambda_kl_q_controller.q_table_last_access_time
            }
            
        fprefix = "wubugaad_hybridgen_dft_ckpt_v01" # Using consistent prefix
        if is_best:
            fp = os.path.join(self.args.checkpoint_dir, f"{fprefix}_best.pt")
        elif is_intermediate:
            fp = os.path.join(self.args.checkpoint_dir, f"{fprefix}_step{self.global_step}.pt")
        else: # Regular end-of-epoch save
            fp = os.path.join(self.args.checkpoint_dir, f"{fprefix}_ep{self.current_epoch+1}_step{self.global_step}.pt")
            
        try:
            torch.save(data, fp)
            self.logger.info(f"Checkpoint saved: {os.path.basename(fp)}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint to {fp}: {e}", exc_info=True)

    def load_checkpoint(self, checkpoint_path:str) -> Tuple[int,int]:
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint file {checkpoint_path} not found. Starting fresh.")
            return 0,0
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}. Starting fresh.", exc_info=True)
            return 0,0

        m_load = self.model.module if self.ddp_active else self.model
        d_load = self.discriminator.module if self.ddp_active else self.discriminator

        try:
            m_load.load_state_dict(ckpt['model_state_dict'], strict=self.args.load_strict)
            self.logger.info("Model state_dict loaded.")
        except Exception as e:
            self.logger.error(f"Error loading model state_dict: {e}", exc_info=not self.args.load_strict)
        try:
            d_load.load_state_dict(ckpt['discriminator_state_dict'], strict=self.args.load_strict)
            self.logger.info("Discriminator state_dict loaded.")
        except Exception as e:
            self.logger.error(f"Error loading discriminator state_dict: {e}", exc_info=not self.args.load_strict)

        if 'optimizer_enc_gen_state_dict' in ckpt and self.optimizer_enc_gen:
            try:
                self.optimizer_enc_gen.load_state_dict(ckpt['optimizer_enc_gen_state_dict'])
                self.logger.info("Optimizer Enc/Gen state loaded.")
            except Exception as e: self.logger.warning(f"Could not load Optimizer Enc/Gen state: {e}")
        if 'optimizer_disc_state_dict' in ckpt and self.optimizer_disc:
            try:
                self.optimizer_disc.load_state_dict(ckpt['optimizer_disc_state_dict'])
                self.logger.info("Optimizer Disc state loaded.")
            except Exception as e: self.logger.warning(f"Could not load Optimizer Disc state: {e}")

        if self.args.use_amp and self.device.type == 'cuda':
            if 'scaler_enc_gen_state_dict' in ckpt and self.scaler_enc_gen and ckpt['scaler_enc_gen_state_dict'] is not None:
                self.scaler_enc_gen.load_state_dict(ckpt['scaler_enc_gen_state_dict'])
            if 'scaler_disc_state_dict' in ckpt and self.scaler_disc and ckpt['scaler_disc_state_dict'] is not None:
                self.scaler_disc.load_state_dict(ckpt['scaler_disc_state_dict'])
        
        def _load_q_controller_state(q_ctrl_obj, state_dict_key_in_ckpt):
            if q_ctrl_obj and state_dict_key_in_ckpt in ckpt and ckpt[state_dict_key_in_ckpt]:
                q_state = ckpt[state_dict_key_in_ckpt]
                try:
                    q_ctrl_obj.q_table = q_state.get('q_table', {})
                    q_ctrl_obj.epsilon = q_state.get('epsilon', q_ctrl_obj.epsilon_start)
                    q_ctrl_obj.prev_lr_mom_state = q_state.get('prev_lr_mom_state')
                    q_ctrl_obj.prev_lr_mom_action = q_state.get('prev_lr_mom_action')
                    q_ctrl_obj.prev_lambda_kl_state = q_state.get('prev_lambda_kl_state') # For LKL controller
                    q_ctrl_obj.prev_lambda_kl_action = q_state.get('prev_lambda_kl_action') # For LKL controller

                    if 'loss_histories' in q_state:
                        lh = q_state['loss_histories']
                        q_ctrl_obj.loss_g_total_hist = deque(lh.get('g_total',[]), maxlen=q_ctrl_obj.state_history_len)
                        q_ctrl_obj.loss_g_recon_hist = deque(lh.get('g_recon',[]), maxlen=q_ctrl_obj.state_history_len)
                        q_ctrl_obj.loss_g_kl_hist = deque(lh.get('g_kl',[]), maxlen=q_ctrl_obj.state_history_len)
                        q_ctrl_obj.loss_g_adv_hist = deque(lh.get('g_adv',[]), maxlen=q_ctrl_obj.state_history_len)
                        q_ctrl_obj.loss_d_total_hist = deque(lh.get('d_total',[]), maxlen=q_ctrl_obj.state_history_len)
                        q_ctrl_obj.loss_d_real_hist = deque(lh.get('d_real',[]), maxlen=q_ctrl_obj.state_history_len)
                        q_ctrl_obj.loss_d_fake_hist = deque(lh.get('d_fake',[]), maxlen=q_ctrl_obj.state_history_len)
                    if 'interval_histories' in q_state: # For LKL controller
                        ih = q_state['interval_histories']
                        q_ctrl_obj.interval_avg_recon_hist = deque(ih.get('avg_recon',[]), maxlen=q_ctrl_obj.lambda_kl_state_history_len)
                        q_ctrl_obj.interval_avg_kl_div_hist = deque(ih.get('avg_kl_div',[]), maxlen=q_ctrl_obj.lambda_kl_state_history_len)
                        q_ctrl_obj.interval_avg_d_total_hist = deque(ih.get('avg_d_total',[]), maxlen=q_ctrl_obj.lambda_kl_state_history_len)
                        q_ctrl_obj.interval_val_metric_hist = deque(ih.get('val_metric',[]), maxlen=q_ctrl_obj.lambda_kl_state_history_len)
                    
                    q_ctrl_obj.reward_hist = deque(q_state.get('reward_hist', []), maxlen=q_ctrl_obj.reward_hist.maxlen if q_ctrl_obj.reward_hist else 100)
                    q_ctrl_obj.q_table_access_count = defaultdict(int, q_state.get('q_table_access_count', {}))
                    q_ctrl_obj.q_table_creation_time = q_state.get('q_table_creation_time', {})
                    q_ctrl_obj.q_table_last_access_time = q_state.get('q_table_last_access_time', {})
                    self.logger.info(f"Q-Controller state for '{state_dict_key_in_ckpt}' loaded.")
                except Exception as e_qc_load:
                    self.logger.warning(f"Could not fully load Q-Controller state for '{state_dict_key_in_ckpt}': {e_qc_load}", exc_info=True)

        _load_q_controller_state(getattr(self.optimizer_enc_gen, 'q_controller', None), 'q_controller_enc_gen_state')
        _load_q_controller_state(getattr(self.optimizer_disc, 'q_controller', None), 'q_controller_disc_state')
        _load_q_controller_state(self.lambda_kl_q_controller, 'q_controller_lambda_kl_state')

        loaded_gs = ckpt.get('global_step', 0)
        loaded_ep = ckpt.get('epoch', 0)
        # If loading strict, resume from the exact epoch. Otherwise, start next epoch if global step > 0.
        next_ep_start = loaded_ep + 1 if (loaded_gs > 0 and not self.args.load_strict) else loaded_ep
        
        default_best_val = -float('inf') if self.args.val_primary_metric in ["avg_val_psnr", "avg_val_ssim"] else float('inf')
        self.best_val_metric_val = ckpt.get('best_val_metric_val', default_best_val)
        
        # Load lambda_kl value from checkpoint if present, otherwise use arg
        loaded_lambda_kl_from_ckpt = float(ckpt.get('current_lambda_kl', self.args.lambda_kl))
        self.lambda_kl = loaded_lambda_kl_from_ckpt if (self.args.load_checkpoint and 'current_lambda_kl' in ckpt) else self.args.lambda_kl
        self.logger.info(f"Effective self.lambda_kl after load/init: {self.lambda_kl:.4e}")

        # Ensure Q-controllers are updated with the loaded/current lambda_kl
        for q_ctrl in [getattr(self.optimizer_enc_gen, 'q_controller', None), 
                       getattr(self.optimizer_disc, 'q_controller', None), 
                       self.lambda_kl_q_controller]:
            if q_ctrl and hasattr(q_ctrl, 'set_current_lambda_kl'):
                q_ctrl.set_current_lambda_kl(self.lambda_kl)
        
        self.logger.info(f"Resuming training. GlobalStep: {loaded_gs}, NextEpochStart: {next_ep_start}. BestVal({self.args.val_primary_metric}): {self.best_val_metric_val:.4f}. LambdaKL: {self.lambda_kl:.4e}")
        return loaded_gs, next_ep_start

    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        m_ref = self.model.module if self.ddp_active else self.model
        m_ref.eval() # Set to evaluation mode
        dev = self.device
        dtype_m = next(iter(m_ref.parameters()), torch.tensor(0.0, device=self.device)).dtype
        lat_dim = self.args.latent_dim

        if noise is None:
            z = torch.randn(num_samples, lat_dim, device=dev, dtype=dtype_m)
        else:
            z = noise.to(device=dev, dtype=dtype_m)
        
        if z.shape[0] != num_samples:
            num_samples = z.shape[0] # Adjust if provided noise has different batch size
            self.logger.warning(f"Number of samples adjusted to noise shape: {num_samples}")

        npf_cfg = self.video_config.get("num_predict_frames",1)
        nra_cfg = self.gaad_appearance_config.get("num_regions",0)
        fd_cfg = self.args.image_h_w_tuple # type: ignore [attr-defined]
        
        dbb_list = []
        if nra_cfg > 0:
            for _ in range(num_samples):
                # Generate GAAD bboxes for each sample in the batch, for all predicted frames
                # Assuming GAAD bboxes are same for all predicted frames for a given sample for simplicity here
                single_sample_bboxes_for_all_pred_frames = golden_subdivide_rect_fixed_n(
                    fd_cfg, nra_cfg, device=dev, dtype=dtype_m, 
                    min_size_px=self.gaad_appearance_config.get('min_size_px',5)
                ).unsqueeze(0).repeat(npf_cfg,1,1) # Shape (N_pred_frames, NumRegions, 4)
                dbb_list.append(single_sample_bboxes_for_all_pred_frames)
        
        dbb_batch = torch.stack(dbb_list) if dbb_list else None # Shape (NumSamples, N_pred_frames, NumRegions, 4)
        
        self.logger.info(f"Sampling {num_samples} sequences (DFT App: {self.args.use_dft_features_appearance})...")
        
        gen_frames_pixels: Optional[torch.Tensor] = None
        if dbb_batch is not None or nra_cfg == 0: # Proceed if bboxes generated or not needed (nra_cfg=0)
            gen_output = m_ref.decode(z, dbb_batch) # Output: DFT coeffs or Pixels
            
            if self.args.use_dft_features_appearance:
                B_s, N_s, R_s, C_s, _, H_dft_s, W_dftc_s = gen_output.shape
                dft_flat_s = gen_output.reshape(B_s*N_s*R_s, C_s*2*H_dft_s*W_dftc_s)
                patches_flat_s = DFTUtils.reconstruct_patches_from_2d_dft(
                    dft_flat_s, self.args.dft_norm_scale_video, 
                    m_ref.generator.num_img_channels, 
                    m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft
                )
                patches_struct_s = patches_flat_s.view(B_s,N_s,R_s,C_s,m_ref.generator.gen_patch_h_for_dft, m_ref.generator.gen_patch_w_for_dft)
                gen_frames_pixels = ImageAssemblyUtils.assemble_frames_from_patches(
                    patches_struct_s, dbb_batch, fd_cfg, output_range=(-1.0,1.0) # type: ignore
                )
            else: # Generator outputs pixels directly
                gen_frames_pixels = gen_output
            self.logger.info("Sampling finished.")
        else:
            self.logger.warning("Sampling skipped: BBoxes required for GAAD but not generated (nra_cfg > 0 and dbb_batch is None).")
            
        m_ref.train() # Set back to training mode after sampling
        return gen_frames_pixels




# =====================================================================
# Arg Parsing and Main Execution Logic (Added DFT args)
# =====================================================================
def seed_worker_init_fn(worker_id, base_seed, rank, world_size): # ... (No changes)
     worker_seed = base_seed + worker_id + rank * world_size; random.seed(worker_seed); np.random.seed(worker_seed); torch.manual_seed(worker_seed)
def seed_everything(seed:int,rank:int=0,world_size:int=1): # ... (No changes)
    actual_seed = seed + rank; random.seed(actual_seed); np.random.seed(actual_seed); torch.manual_seed(actual_seed); torch.cuda.manual_seed_all(actual_seed) if torch.cuda.is_available() else None

def _configure_wubu_stack(args: argparse.Namespace, prefix: str) -> Optional[Dict]: # ... (No changes)
    if prefix == "wubu_m" and not (args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE): return None
    config = DEFAULT_CONFIG_WUBU.copy(); num_levels_val = getattr(args, f"{prefix}_num_levels", 0); config["num_levels"] = num_levels_val
    if num_levels_val == 0: [ setattr(config, key, [] if key not in ["tangent_input_combination_dims"] else [DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0]]) for key in ["hyperbolic_dims", "initial_curvatures", "initial_scales", "initial_spread_values", "boundary_points_per_level", "transform_types", "transform_hidden_dims", "tangent_input_combination_dims"]]; return config
    config["hyperbolic_dims"] = getattr(args, f"{prefix}_hyperbolic_dims", DEFAULT_CONFIG_WUBU["hyperbolic_dims"]); config["initial_curvatures"] = getattr(args, f"{prefix}_initial_curvatures", DEFAULT_CONFIG_WUBU["initial_curvatures"]); config["use_rotation_in_transform"] = getattr(args, f"{prefix}_use_rotation", DEFAULT_CONFIG_WUBU["use_rotation_in_transform"]); config["phi_influence_curvature"] = getattr(args, f"{prefix}_phi_influence_curvature", DEFAULT_CONFIG_WUBU["phi_influence_curvature"]); config["phi_influence_rotation_init"] = getattr(args, f"{prefix}_phi_influence_rotation_init", DEFAULT_CONFIG_WUBU["phi_influence_rotation_init"]); config["dropout"] = args.wubu_dropout
    def _ensure_list_len(cfg_dict, key, target_len, default_fill_list):
        current_val = cfg_dict.get(key, []); is_list_orig = isinstance(current_val, list); current_list_val = current_val if is_list_orig else [current_val]
        base_default = default_fill_list[0] if default_fill_list else (1.0 if "scales" in key or "curvatures" in key else (0.1 if "spread" in key else ("linear" if "types" in key else 32))); fill_val = current_list_val[-1] if current_list_val else base_default
        if len(current_list_val) < target_len: cfg_dict[key] = (current_list_val + [fill_val]*(target_len-len(current_list_val)))[:target_len]
        elif len(current_list_val) > target_len: cfg_dict[key] = current_list_val[:target_len]
        if not is_list_orig and target_len == 1 and isinstance(cfg_dict[key], list): cfg_dict[key] = cfg_dict[key][0]
    for key_chk, default_key in [("hyperbolic_dims", "hyperbolic_dims"), ("initial_curvatures", "initial_curvatures"), ("initial_scales", "initial_scales"), ("initial_spread_values", "initial_spread_values"), ("boundary_points_per_level", "boundary_points_per_level")]: _ensure_list_len(config, key_chk, num_levels_val, DEFAULT_CONFIG_WUBU[default_key])
    if not isinstance(config.get("tangent_input_combination_dims"), list): config["tangent_input_combination_dims"] = [config.get("tangent_input_combination_dims", DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0])]
    num_transitions = max(0, num_levels_val-1)
    if num_transitions > 0: _ensure_list_len(config,"transform_types",num_transitions,DEFAULT_CONFIG_WUBU["transform_types"]); _ensure_list_len(config,"transform_hidden_dims",num_transitions,DEFAULT_CONFIG_WUBU["transform_hidden_dims"])
    else: config["transform_types"]=[]; config["transform_hidden_dims"]=[]
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu-GAAD Regional VAE-GAN w/ OptFlow & DFT (v0.1-DFT)")
    # --- Data and DDP --- (No changes here)
    parser.add_argument('--video_data_path', type=str, default="demo_video_data_dir"); parser.add_argument('--local_rank', type=int, default=-1); parser.add_argument('--epochs', type=int, default=100); parser.add_argument('--batch_size', type=int, default=4); parser.add_argument('--image_h', type=int, default=64); parser.add_argument('--image_w', type=int, default=64); parser.add_argument('--num_channels', type=int, default=3); parser.add_argument('--num_input_frames', type=int, default=3); parser.add_argument('--num_predict_frames', type=int, default=1); parser.add_argument('--frame_skip', type=int, default=1); parser.add_argument('--seed',type=int, default=42); parser.add_argument('--num_workers',type=int, default=2); parser.add_argument('--checkpoint_dir',type=str, default='wubugaad_hybridgen_dft_checkpoints_v01'); parser.add_argument('--load_checkpoint', type=str, default=None); parser.add_argument('--load_strict', action='store_true'); parser.add_argument('--wandb',action='store_true'); parser.add_argument('--wandb_project',type=str,default='WuBuGAADHybridGenV01DFT'); parser.add_argument('--wandb_run_name',type=str,default=None); parser.add_argument('--log_interval',type=int, default=50); parser.add_argument('--save_interval',type=int, default=1000)
    # --- GAAD --- (No changes here)
    parser.add_argument('--gaad_num_regions', type=int, default=12); parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"]); parser.add_argument('--gaad_min_size_px', type=int, default=4)
    # --- Motion Branch --- (No changes here)
    parser.add_argument('--use_wubu_motion_branch', action='store_true'); parser.add_argument('--gaad_motion_num_regions', type=int, default=8); parser.add_argument('--gaad_motion_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"]); parser.add_argument('--optical_flow_net_type', type=str, default='raft_small', choices=list(FLOW_MODELS.keys()) if OPTICAL_FLOW_AVAILABLE else []); parser.add_argument('--freeze_flow_net', action='store_true'); parser.add_argument('--flow_stats_components', nargs='+', type=str, default=['mag_mean', 'angle_mean'])
    # --- Encoder Architecture --- (No changes here, DFT patch size is separate)
    parser.add_argument('--latent_dim', type=int, default=256); parser.add_argument('--encoder_use_roi_align', action='store_true'); parser.add_argument('--encoder_shallow_cnn_channels', type=int, default=32); parser.add_argument('--encoder_roi_align_output_h', type=int, default=4); parser.add_argument('--encoder_roi_align_output_w', type=int, default=4); parser.add_argument('--encoder_pixel_patch_size', type=int, default=16); parser.add_argument('--encoder_initial_tangent_dim', type=int, default=128)
    # --- DFT Specific Args (NEW) ---
    parser.add_argument('--use_dft_features_appearance', action='store_true', help="Use DFT features for appearance branch.")
    parser.add_argument('--dft_patch_size_h', type=int, default=16, help="Target patch height for 2D DFT.")
    parser.add_argument('--dft_patch_size_w', type=int, default=16, help="Target patch width for 2D DFT.")
    parser.add_argument('--dft_norm_scale_video', type=float, default=20.0, help="Normalization scale for DFT real/imag components (video).")
    # --- Generator Architecture ---
    parser.add_argument('--gen_temporal_kernel_size', type=int, default=3); parser.add_argument('--gen_final_conv_kernel_spatial', type=int, default=3)
    parser.add_argument('--gen_use_gaad_film_condition', action='store_true', default=True, help="Enable GAAD-FiLM in Generator (default True).") # New arg for G
    # --- Discriminator Architecture --- (No changes here)
    parser.add_argument('--discriminator_type', type=str, default="spatio_temporal_cnn", choices=["spatio_temporal_cnn", "regional_cnn"]); parser.add_argument('--disc_apply_spectral_norm', action='store_true'); parser.add_argument('--disc_base_disc_channels', type=int, default=64); parser.add_argument('--disc_max_disc_channels', type=int, default=512); parser.add_argument('--disc_temporal_kernel_size', type=int, default=3); parser.add_argument('--disc_min_hidden_fc_dim', type=int, default=128); parser.add_argument('--disc_max_hidden_fc_dim', type=int, default=512); parser.add_argument('--disc_use_gaad_film_condition', action='store_true'); parser.add_argument('--disc_gaad_condition_dim_disc', type=int, default=64); parser.add_argument('--disc_patch_size', type=int, default=16); parser.add_argument('--disc_cnn_channels_2d', nargs='+', type=int, default=[64, 128, 256])
    # --- WuBu Stacks --- (No changes here)
    parser.add_argument('--wubu_dropout', type=float, default=0.1); parser.add_argument('--wubu_s_num_levels', type=int, default=2); parser.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[64,32]); parser.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0,0.8]); parser.add_argument('--wubu_s_use_rotation', action='store_true'); parser.add_argument('--wubu_s_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true'); parser.add_argument('--wubu_m_num_levels', type=int, default=1); parser.add_argument('--wubu_m_hyperbolic_dims', nargs='+', type=int, default=[32]); parser.add_argument('--wubu_m_initial_curvatures', nargs='+', type=float, default=[0.7]); parser.add_argument('--wubu_m_use_rotation', action='store_true'); parser.add_argument('--wubu_m_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_m_phi_influence_rotation_init', action='store_true'); parser.add_argument('--wubu_t_num_levels', type=int, default=1); parser.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[128]); parser.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[0.5]); parser.add_argument('--wubu_t_use_rotation', action='store_true'); parser.add_argument('--wubu_t_phi_influence_curvature', action='store_true'); parser.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true'); parser.add_argument('--wubu_s_output_dim', type=int, default=32); parser.add_argument('--wubu_m_output_dim', type=int, default=32)
    # --- Training --- (No changes here)
    parser.add_argument('--lambda_recon', type=float, default=10.0); parser.add_argument('--lambda_kl', type=float, default=0.1); parser.add_argument('--lambda_gan', type=float, default=1.0); parser.add_argument('--learning_rate_gen',type=float,default=1e-4); parser.add_argument('--learning_rate_disc',type=float,default=1e-4); parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0); parser.add_argument('--global_max_grad_norm',type=float,default=5.0); parser.add_argument('--q_controller_enabled',action='store_true'); parser.add_argument('--grad_accum_steps',type=int, default=1); parser.add_argument('--use_amp', action='store_true'); parser.add_argument('--detect_anomaly',action='store_true'); parser.add_argument('--log_grad_norm', action='store_true'); parser.add_argument('--lambda_kl_update_interval', type=int, default=2); parser.add_argument('--min_lambda_kl_q_control', type=float, default=1e-6); parser.add_argument('--max_lambda_kl_q_control', type=float, default=0.5)
    # --- Validation & Sampling --- (No changes here)
    parser.add_argument('--wandb_log_train_recon_interval', type=int, default=0); parser.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=0); parser.add_argument('--use_lpips_for_verification', action='store_true'); parser.add_argument('--validation_video_path', type=str, default=None); parser.add_argument('--validation_split_fraction', type=float, default=0.1); parser.add_argument('--val_block_size', type=int, default=20); parser.add_argument('--val_primary_metric', type=str, default="avg_val_psnr", choices=["avg_val_recon_mse_dft", "avg_val_recon_mse_pixel", "avg_val_psnr", "avg_val_ssim", "avg_val_lpips"]); parser.add_argument('--num_val_samples_to_log', type=int, default=2); parser.add_argument('--demo_num_samples', type=int, default=4)
    parsed_args = parser.parse_args()
    parsed_args.image_h_w_tuple = (parsed_args.image_h, parsed_args.image_w) # Store for convenience
    if parsed_args.use_wubu_motion_branch and not OPTICAL_FLOW_AVAILABLE: parser.error("Motion branch needs optical flow, but torchvision.models.optical_flow unavailable.")
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.optical_flow_net_type not in FLOW_MODELS: parser.error(f"Optical flow net type '{parsed_args.optical_flow_net_type}' not in available: {list(FLOW_MODELS.keys())}")
    def validate_wubu_config_for_argparse(args_obj, prefix_str, parser_ref, is_motion_branch_active): # ... (No changes)
        num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0); is_motion = prefix_str == "wubu_m"
        if num_levels > 0 and (not is_motion or is_motion_branch_active):
            for suffix, attr_name in [("hyperbolic_dims", f"{prefix_str}_hyperbolic_dims"), ("initial_curvatures", f"{prefix_str}_initial_curvatures")]:
                val_list = getattr(args_obj, attr_name); is_list=isinstance(val_list, list); val_list = val_list if is_list else [val_list]
                if len(val_list) != num_levels:
                    if len(val_list) == 1 and num_levels > 1: setattr(args_obj, attr_name, [val_list[0]] * num_levels)
                    elif not val_list and suffix=="initial_curvatures": setattr(args_obj, attr_name, [1.0] * num_levels)
                    elif not val_list and suffix=="hyperbolic_dims": setattr(args_obj, attr_name, [getattr(args_obj, 'latent_dim', 32)//num_levels if num_levels>0 else []]*num_levels)
                    else: parser_ref.error(f"{prefix_str}: Length mismatch {attr_name} ({len(val_list)}) vs num_levels ({num_levels})")
    validate_wubu_config_for_argparse(parsed_args, "wubu_s", parser, True); validate_wubu_config_for_argparse(parsed_args, "wubu_t", parser, True); validate_wubu_config_for_argparse(parsed_args, "wubu_m", parser, parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE)
    if parsed_args.wubu_s_num_levels > 0 and parsed_args.wubu_s_hyperbolic_dims: parsed_args.wubu_s_output_dim = parsed_args.wubu_s_hyperbolic_dims[-1]
    else: parsed_args.wubu_s_output_dim = parsed_args.encoder_initial_tangent_dim; parsed_args.wubu_s_num_levels = 0
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.wubu_m_num_levels > 0 and parsed_args.wubu_m_hyperbolic_dims: parsed_args.wubu_m_output_dim = parsed_args.wubu_m_hyperbolic_dims[-1]
    else: parsed_args.wubu_m_output_dim = 0; parsed_args.wubu_m_num_levels = 0
    valid_stats = {'mag_mean', 'angle_mean', 'mag_std', 'angle_std'};
    if any(s not in valid_stats for s in parsed_args.flow_stats_components): parser.error(f"Invalid flow_stats_components. Allowed: {valid_stats}. Got: {parsed_args.flow_stats_components}")
    return parsed_args

def main():
    args = parse_arguments()
    ddp_active = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    
    if ddp_active:
        rank=int(os.environ["RANK"])
        local_rank=int(os.environ["LOCAL_RANK"])
        world_size=int(os.environ["WORLD_SIZE"])
        init_process_group(backend="nccl") # Consider "gloo" for CPU-only or mixed environments
        device=torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank=0
        local_rank=0
        world_size=1
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            
    am_main_process = (rank == 0)
    base_logger_name = "WuBuGAADHybridGenV01DFT" # Consistent logger name
    
    # Clear existing handlers for the root logger and specific logger to avoid duplicate logs with DDP
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers): # Iterate over a copy
        root_logger.removeHandler(h)
    specific_logger = logging.getLogger(base_logger_name)
    for h in list(specific_logger.handlers): # Iterate over a copy
        specific_logger.removeHandler(h)
        
    log_level = logging.INFO if am_main_process else logging.WARNING
    logging.basicConfig(level=log_level, 
                        format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s', 
                        force=True) # force=True helps in DDP setups
                        
    current_logger_main = logging.getLogger(f"{base_logger_name}.Main")
    current_logger_main.info(f"--- {base_logger_name} (Rank {rank}/{world_size}, Device {device}, DDP Active: {ddp_active}, AMP: {args.use_amp}, DFT Appearance: {args.use_dft_features_appearance}) ---")
    
    seed_everything(args.seed, rank, world_size)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        current_logger_main.warning("Autograd anomaly detection ENABLED.")
        
    if am_main_process:
        current_logger_main.info(f"Effective Args: {vars(args)}")
        
    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubudft_v0.1_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        try:
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args), 
                       resume="allow", id=wandb.util.generate_id() if wandb.run is None else wandb.run.id)
            current_logger_main.info(f"WandB initialized: Run Name '{run_name}', Project '{args.wandb_project}'")
        except Exception as e_wandb:
            current_logger_main.error(f"WandB initialization failed: {e_wandb}", exc_info=True)
            args.wandb = False # Disable wandb if init fails
            
    video_config = {
        "image_size": args.image_h_w_tuple, 
        "num_channels": args.num_channels, 
        "num_input_frames": args.num_input_frames, 
        "num_predict_frames": args.num_predict_frames, 
        "wubu_s_output_dim": args.wubu_s_output_dim, 
        "wubu_m_output_dim": args.wubu_m_output_dim 
    }
    gaad_appearance_config = {
        "num_regions": args.gaad_num_regions, 
        "decomposition_type": args.gaad_decomposition_type, 
        "min_size_px": args.gaad_min_size_px
    }
    gaad_motion_config = {
        "num_regions": args.gaad_motion_num_regions, 
        "decomposition_type": args.gaad_motion_decomposition_type, 
        "min_size_px": args.gaad_min_size_px
    } if args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE else None
    
    wubu_s_config = _configure_wubu_stack(args, "wubu_s")
    wubu_t_config = _configure_wubu_stack(args, "wubu_t")
    wubu_m_config = _configure_wubu_stack(args, "wubu_m")
    
    discriminator_config = {
        "type": args.discriminator_type, 
        "apply_spectral_norm": args.disc_apply_spectral_norm, 
        "use_gaad_film_condition": args.disc_use_gaad_film_condition, 
        "gaad_condition_dim_disc": args.disc_gaad_condition_dim_disc, 
        "base_disc_channels": args.disc_base_disc_channels, 
        "max_disc_channels": args.disc_max_disc_channels, 
        "temporal_kernel_size": args.disc_temporal_kernel_size, 
        "patch_size": args.disc_patch_size, 
        "cnn_channels_2d": args.disc_cnn_channels_2d 
    }
    if am_main_process:
        current_logger_main.info(f"VideoCfg:{video_config}\nGAADAppCfg:{gaad_appearance_config}\nGAADMotCfg:{gaad_motion_config}\nWuBuS:{wubu_s_config}\nWuBuT:{wubu_t_config}\nWuBuM:{wubu_m_config}\nDiscCfg:{discriminator_config}")

    model = WuBuGAADHybridGenNet(args, video_config, gaad_appearance_config, gaad_motion_config, wubu_s_config, wubu_t_config, wubu_m_config).to(device)
    discriminator = RegionalDiscriminator(args, video_config, gaad_appearance_config, discriminator_config).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)
        wandb.watch(discriminator, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)

    if ddp_active:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # find_unused_parameters can be True during dev
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False) # Usually False for Discriminator
        
    q_cfg_gen = DEFAULT_CONFIG_QLEARN_HYBRID.copy() if args.q_controller_enabled else None
    q_cfg_disc = DEFAULT_CONFIG_QLEARN_HYBRID.copy() if args.q_controller_enabled else None
    if am_main_process and q_cfg_gen: current_logger_main.info(f"Q-Controller config for Generator: {q_cfg_gen}")
    if am_main_process and q_cfg_disc: current_logger_main.info(f"Q-Controller config for Discriminator: {q_cfg_disc}")
        
    optimizer_enc_gen = RiemannianEnhancedSGD(model.parameters(), lr=args.learning_rate_gen, q_learning_config=q_cfg_gen, max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type="generator")
    optimizer_disc = RiemannianEnhancedSGD(discriminator.parameters(), lr=args.learning_rate_disc, q_learning_config=q_cfg_disc, max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type="discriminator")

    actual_video_path = args.video_data_path
    demo_file_name = "dummy_video_hybridgen_dft_v01.mp4" # Consistent demo file name
    if "demo_video_data" in args.video_data_path: # If using default demo path structure
        actual_video_path = os.path.join(args.video_data_path, demo_file_name)

    if am_main_process:
        os.makedirs(os.path.dirname(actual_video_path), exist_ok=True)
        if "demo_video_data" in args.video_data_path and not os.path.exists(actual_video_path):
            if IMAGEIO_AVAILABLE and imageio is not None:
                current_logger_main.info(f"Creating dummy video: {actual_video_path}...")
                min_raw_frames_needed = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                num_dummy_frames = max(100, min_raw_frames_needed + 50) # Ensure enough frames
                dummy_h, dummy_w = int(args.image_h), int(args.image_w)
                try:
                    with imageio.get_writer(actual_video_path, fps=15, quality=8, macro_block_size=16) as video_writer: # Use with statement
                        for _ in range(num_dummy_frames):
                            video_writer.append_data(np.random.randint(0,255, (dummy_h,dummy_w,args.num_channels), dtype=np.uint8))
                    current_logger_main.info(f"Dummy video with {num_dummy_frames} frames created at {actual_video_path}.")
                except Exception as e_imageio_write:
                    current_logger_main.error(f"Error creating dummy video: {e_imageio_write}", exc_info=True)
            else:
                current_logger_main.error("imageio library not available. Cannot create dummy video.")
    
    if ddp_active: torch.distributed.barrier() # Ensure all processes sync after potential file creation

    if not os.path.isfile(actual_video_path):
        current_logger_main.error(f"Video path '{actual_video_path}' does not point to a file. Exiting.")
        sys.exit(1)
        
    total_frames_per_sample = args.num_input_frames + args.num_predict_frames
    try:
        full_dataset = VideoFrameDataset(video_path=actual_video_path, num_frames_total=total_frames_per_sample, 
                                         image_size=args.image_h_w_tuple, frame_skip=args.frame_skip)
    except Exception as e:
        current_logger_main.error(f"Failed to initialize main Dataset from '{actual_video_path}': {e}", exc_info=True)
        sys.exit(1)
        
    if not full_dataset or len(full_dataset) == 0:
        current_logger_main.error(f"Main dataset from '{actual_video_path}' is empty. Exiting.")
        sys.exit(1)

    train_dataset, val_dataset = full_dataset, None
    num_total_samples = len(full_dataset)

    if args.validation_video_path and os.path.isfile(args.validation_video_path):
        try:
            val_dataset_candidate = VideoFrameDataset(video_path=args.validation_video_path, num_frames_total=total_frames_per_sample, 
                                                      image_size=args.image_h_w_tuple, frame_skip=args.frame_skip)
            if len(val_dataset_candidate) > 0:
                val_dataset = val_dataset_candidate
                current_logger_main.info(f"Using separate validation video: {args.validation_video_path}, Samples: {len(val_dataset)}")
            else:
                current_logger_main.warning(f"Validation video '{args.validation_video_path}' is empty. Attempting split from training data.")
        except Exception as e:
            current_logger_main.warning(f"Could not load validation dataset from '{args.validation_video_path}': {e}. Attempting split from training data.")

    if val_dataset is None and args.validation_split_fraction > 0.0 and num_total_samples > 10: # Min samples for a meaningful split
        val_block_size = args.val_block_size
        if val_block_size <= 0 or num_total_samples < val_block_size * 2: # Use random split if block size invalid or too small dataset
            num_val = int(num_total_samples * args.validation_split_fraction)
            num_train = num_total_samples - num_val
            if num_train > 0 and num_val > 0:
                train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], 
                                                                            generator=torch.Generator().manual_seed(args.seed + rank))
                current_logger_main.info(f"Split main dataset (random): Train={len(train_dataset)}, Val={len(val_dataset if val_dataset else [])}") # type: ignore
            else:
                current_logger_main.warning("Random split failed (not enough samples for train/val). No validation set.")
                train_dataset = full_dataset # Use full dataset for training
        else: # Use block split
            target_val_samples = int(num_total_samples * args.validation_split_fraction)
            num_val_blocks = max(1, target_val_samples // val_block_size)
            max_possible_block_starts = num_total_samples - val_block_size + 1
            rng_val_split = random.Random(args.seed + 7) # Use a different seed for split consistency
            
            block_start_indices = sorted(rng_val_split.sample(range(max_possible_block_starts), min(num_val_blocks, max_possible_block_starts)))
            val_indices_set = set()
            for start_idx in block_start_indices:
                val_indices_set.update(range(start_idx, min(start_idx + val_block_size, num_total_samples)))
            
            all_indices = list(range(num_total_samples))
            train_indices = sorted(list(set(all_indices) - val_indices_set))
            val_indices = sorted(list(val_indices_set))
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
                val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
                current_logger_main.info(f"Split main dataset (block): Train={len(train_dataset)}, Val={len(val_dataset)}")
            else:
                current_logger_main.warning("Block split failed (not enough samples for train/val). No validation set.")
                train_dataset = full_dataset

    if am_main_process:
        current_logger_main.info(f"Final dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}")

    worker_init_fn_seeded = functools.partial(seed_worker_init_fn, base_seed=args.seed, rank=rank, world_size=world_size) if args.num_workers > 0 else None
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_active else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), # Shuffle if not using DDP sampler
                              num_workers=args.num_workers, sampler=train_sampler, 
                              pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded, drop_last=True)

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, sampler=val_sampler, 
                                pin_memory=(device.type == 'cuda'), drop_last=False, worker_init_fn=worker_init_fn_seeded)
                                
    trainer = HybridTrainer(model, discriminator, optimizer_enc_gen, optimizer_disc, device, 
                            train_loader, val_loader, args, rank, world_size, ddp_active)
                            
    start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint) if args.load_checkpoint else (0,0)
    
    try:
        trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt:
        current_logger_main.info(f"Rank {rank}: Training interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        current_logger_main.error(f"Rank {rank}: Training loop crashed: {e}", exc_info=True)
    finally:
        if am_main_process:
            current_logger_main.info("Finalizing run...")
            final_metrics_to_save = trainer.last_val_metrics.copy() if trainer.last_val_metrics else {}
            final_metrics_to_save['best_val_metric_val_at_end'] = trainer.best_val_metric_val
            trainer._save_checkpoint(metrics=final_metrics_to_save) # Save final checkpoint

        if am_main_process and args.epochs > 0 and hasattr(trainer, 'sample') and trainer.global_step > 0:
            current_logger_main.info("Generating final demo samples...")
            try:
                # Model for sampling should be the unwrapped model if DDP was active
                # The trainer's .sample() method handles m_ref = self.model.module if self.ddp_active
                pred_pixels = trainer.sample(num_samples=args.demo_num_samples)

                if pred_pixels is not None and pred_pixels.numel() > 0 and pred_pixels.shape[0] > 0:
                    save_dir = os.path.join(args.checkpoint_dir, "demo_samples_hybrid_dft_v01")
                    os.makedirs(save_dir, exist_ok=True)
                    num_frames_to_save_per_sample = min(pred_pixels.shape[1], 3) # Save up to 3 frames
                    
                    for b_idx in range(min(args.demo_num_samples, pred_pixels.shape[0])):
                        for frame_s_idx in range(num_frames_to_save_per_sample):
                            save_image(
                                (pred_pixels[b_idx, frame_s_idx].cpu().clamp(-1, 1) + 1) / 2.0, # Normalize to [0,1]
                                os.path.join(save_dir, f"demo_s{b_idx}_f{frame_s_idx}_ep{trainer.current_epoch+1}_gs{trainer.global_step}.png")
                            )
                    current_logger_main.info(f"Saved demo sample frames to {save_dir}")

                    if args.wandb and WANDB_AVAILABLE and wandb.run:
                        wb_imgs_final_demo = []
                        for b_idx in range(min(args.demo_num_samples, pred_pixels.shape[0])):
                            for frame_s_idx in range(num_frames_to_save_per_sample):
                                wb_imgs_final_demo.append(wandb.Image(
                                    (pred_pixels[b_idx, frame_s_idx].cpu().float().clamp(-1,1)+1)/2.0, 
                                    caption=f"FinalSample S{b_idx} F{frame_s_idx} Ep{trainer.current_epoch+1} Gs{trainer.global_step}"
                                ))
                        if wb_imgs_final_demo:
                            wandb.log({"demo_samples_final": wb_imgs_final_demo}, step=trainer.global_step)
                else:
                    current_logger_main.info("No demo samples generated or pred_pixels was None/empty.")
            except Exception as e_demo:
                current_logger_main.error(f"Demo sampling or saving error: {e_demo}", exc_info=True)
        
        if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
            wandb.finish()
            
        if ddp_active and is_initialized():
            destroy_process_group()
            
        current_logger_main.info(f"Rank {rank}: {base_logger_name} (v0.1-DFT) script finished.")

if __name__ == "__main__":
    # Check for optical flow availability early if motion branch is requested
    # This specific check might be better handled inside parse_arguments if it can raise parser.error
    # For now, keeping it simple here.
    temp_args_check = sys.argv[1:] # Basic check
    use_motion_branch_requested = False
    for arg_idx, arg_val in enumerate(temp_args_check):
        if arg_val == '--use_wubu_motion_branch':
            use_motion_branch_requested = True
            break
        elif arg_val.startswith('--use_wubu_motion_branch='): # Handles --use_wubu_motion_branch=true (though action='store_true' doesn't use this)
             if arg_val.split('=')[1].lower() == 'true':
                 use_motion_branch_requested = True
             break
             
    if use_motion_branch_requested and not OPTICAL_FLOW_AVAILABLE:
        print("FATAL ERROR: Motion branch (--use_wubu_motion_branch) requested, but torchvision.models.optical_flow is unavailable. Please install it or disable the motion branch.")
        sys.exit(1)
    main()