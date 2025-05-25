# WuBuGAADHybridGen_v0.3.py
# VAE-GAN Hybrid Model with GAAD-WuBu Regional Hyperbolic Latent Space
# Incorporates Optical Flow for Motion Encoding Branch.
# Incorporates DCT + DFT for regional appearance features.
# Incorporates advanced training heuristics and dual discriminator switching.
# LAST UPDATE: Refactored from v0.2 (Video DFT) to v0.3 (Video DFT+DCT, Adv Training)
# THIS VERSION: Incorporating DCT alongside DFT, advanced trainer from WuBuSpecTrans_v0.1.1

# =====================================================================
# Python Imports and Setup
# =====================================================================
import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SubsetRandomSampler
import numpy as np
import heapq
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
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import spectral_norm
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from tqdm import tqdm
import inspect
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

# --- DCT Import ---
try:
    from torch_dct import dct_2d, idct_2d
    TORCH_DCT_AVAILABLE = True
except ImportError:
    dct_2d, idct_2d = None, None
    TORCH_DCT_AVAILABLE = False
    print("CRITICAL WARNING: torch-dct library not found. DCT/IDCT operations will fail. Install with 'pip install torch-dct'.")


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
logger = logging.getLogger("WuBuGAADHybridGenV03") # Renamed logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')


# Constants and Default Configs
EPS = 1e-5 # Video uses larger values typically, but audio used smaller. Keeping video default for now.
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
    "boundary_points_per_level": [4], # Video usually has boundary points for spatial WuBu
    "tangent_input_combination_dims": [32],
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
DEFAULT_CONFIG_QLEARN_HYBRID = { # From audio script
    "q_learning_rate": 0.01,
    "discount_factor": 0.90,
    "epsilon_start": 0.6, # Higher start from audio
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9995,
    "lr_scale_options": [0.7, 0.85, 1.0, 1.15, 1.3], # From audio
    "momentum_scale_options": [0.9, 0.95, 0.99, 1.0, 1.01], # From audio
    "max_q_table_size": 15000, # From audio
    "state_history_len": 7, # From audio
    "lambda_kl_state_history_len": 7, # From audio
    "reward_clipping": (-2.5, 2.5), # From audio
    "q_value_clipping": (-35.0, 35.0), # From audio
    "num_probation_steps": None, # To be set dynamically if desired
    "lkl_num_probation_steps": None # To be set dynamically if desired
}

# =====================================================================
# Spectral Transform Utilities (DFT + DCT)
# =====================================================================
class SpectralTransformUtils:
    @staticmethod
    def compute_2d_dft_features(
        patches: torch.Tensor,
        norm_scale: float = 10.0,
        fft_norm_type: str = "ortho"
    ) -> torch.Tensor:
        """
        Computes 2D DFT features (real and imaginary parts) for a batch of image patches.
        Args:
            patches (torch.Tensor): Input patches, shape (B_flat, C, H_patch, W_patch).
            norm_scale (float): Value for tanh normalization scaling.
            fft_norm_type (str): "backward", "ortho", or "forward" for torch.fft.
        Returns:
            torch.Tensor: DFT features, shape (B_flat, C * 2 * H_patch * (W_patch//2+1)).
        """
        B_flat, C, H_patch, W_patch = patches.shape
        if not torch.is_floating_point(patches):
            patches_float = patches.float() / 255.0
        else:
            patches_float = patches # Assume already in a good range if float

        dft_coeffs_complex = torch.fft.rfft2(patches_float, dim=(-2, -1), norm=fft_norm_type) # (B_flat, C, H_patch, W_patch//2+1)
        dft_real = dft_coeffs_complex.real
        dft_imag = dft_coeffs_complex.imag

        # Tanh normalization
        norm_real = torch.tanh(dft_real / norm_scale)
        norm_imag = torch.tanh(dft_imag / norm_scale)
        
        stacked_coeffs = torch.stack([norm_real, norm_imag], dim=2) # (B_flat, C, 2, H_patch, W_patch//2+1)
        flattened_dft_features = stacked_coeffs.reshape(B_flat, -1) # (B_flat, D_dft_features)
        return flattened_dft_features

    @staticmethod
    def reconstruct_patches_from_2d_dft(
        dft_features_norm_flat: torch.Tensor,
        norm_scale: float,
        num_channels: int,
        target_patch_h: int,
        target_patch_w: int,
        fft_norm_type: str = "ortho"
    ) -> torch.Tensor:
        """
        Reconstructs image patches from normalized flat DFT features.
        Args:
            dft_features_norm_flat (torch.Tensor): Shape (B_total_regions, C * 2 * H_patch * (W_patch//2+1)).
            norm_scale (float): Scale used for tanh normalization.
        Returns:
            torch.Tensor: Reconstructed pixel patches, shape (B_total_regions, C, H_patch, W_patch).
                          Values will be approximately in the original input range of DFT (e.g. [0,1] or [-1,1] if input to DFT was so).
        """
        B_total_regions = dft_features_norm_flat.shape[0]
        W_coeffs_one_sided = target_patch_w // 2 + 1
        
        expected_feat_dim = num_channels * 2 * target_patch_h * W_coeffs_one_sided
        if dft_features_norm_flat.shape[1] != expected_feat_dim:
            raise ValueError(f"DFT feature dim mismatch. Expected {expected_feat_dim}, got {dft_features_norm_flat.shape[1]}")

        stacked_coeffs_norm = dft_features_norm_flat.view(
            B_total_regions, num_channels, 2, target_patch_h, W_coeffs_one_sided
        )
        norm_real = stacked_coeffs_norm[:, :, 0, :, :]
        norm_imag = stacked_coeffs_norm[:, :, 1, :, :]

        # Unnormalize (atanh)
        input_dtype = norm_real.dtype
        compute_dtype = torch.float32 if input_dtype in [torch.float16, torch.bfloat16] else input_dtype
        
        eps_clamp_atanh = torch.finfo(compute_dtype).eps * 8
        upper_b_atanh = torch.tensor(1.0 - eps_clamp_atanh, dtype=compute_dtype, device=norm_real.device)
        lower_b_atanh = torch.tensor(-1.0 + eps_clamp_atanh, dtype=compute_dtype, device=norm_real.device)
        
        dft_real_unscaled = torch.atanh(torch.clamp(norm_real.to(compute_dtype), min=lower_b_atanh, max=upper_b_atanh)) * norm_scale
        dft_imag_unscaled = torch.atanh(torch.clamp(norm_imag.to(compute_dtype), min=lower_b_atanh, max=upper_b_atanh)) * norm_scale
        
        dft_real = dft_real_unscaled.to(input_dtype)
        dft_imag = dft_imag_unscaled.to(input_dtype)

        dft_coeffs_complex = torch.complex(dft_real, dft_imag)
        reconstructed_patches = torch.fft.irfft2(dft_coeffs_complex, s=(target_patch_h, target_patch_w), dim=(-2, -1), norm=fft_norm_type)
        return reconstructed_patches

    @staticmethod
    def compute_2d_dct_features(
        patches: torch.Tensor,
        norm_type: str = "tanh", # "none", "global_scale", "tanh"
        norm_global_scale: float = 200.0,
        norm_tanh_scale: float = 60.0
    ) -> torch.Tensor:
        """
        Computes 2D DCT-II features for a batch of image patches.
        Args:
            patches (torch.Tensor): Input patches, shape (B_flat, C, H_patch, W_patch).
            norm_type (str): Normalization type.
            norm_global_scale (float): Scale for 'global_scale' normalization.
            norm_tanh_scale (float): Scale for 'tanh' normalization.
        Returns:
            torch.Tensor: DCT features, shape (B_flat, C * H_patch * W_patch).
        """
        if not TORCH_DCT_AVAILABLE or dct_2d is None:
            logger.error("compute_2d_dct_features: torch-dct not available or dct_2d is None. Returning zeros.")
            return torch.zeros(patches.shape[0], patches.shape[1]*patches.shape[2]*patches.shape[3], device=patches.device, dtype=patches.dtype)

        B_flat, C, H_patch, W_patch = patches.shape
        if not torch.is_floating_point(patches):
            patches_float = patches.float() / 255.0
        else:
            patches_float = patches

        # dct_2d expects (..., H, W)
        transformed_coeffs_list = []
        for c_idx in range(C):
            channel_patches = patches_float[:, c_idx, :, :] # (B_flat, H_patch, W_patch)
            dct_coeffs_channel = dct_2d(channel_patches.float(), norm='ortho') # Use float for DCT
            
            if norm_type == "none": norm_coeffs_ch = dct_coeffs_channel
            elif norm_type == "global_scale": norm_coeffs_ch = dct_coeffs_channel / norm_global_scale
            elif norm_type == "tanh": norm_coeffs_ch = torch.tanh(dct_coeffs_channel / norm_tanh_scale)
            else: norm_coeffs_ch = dct_coeffs_channel # Default to no norm if type unknown
            transformed_coeffs_list.append(norm_coeffs_ch)
        
        # Stack along channel dimension and then flatten
        stacked_norm_coeffs = torch.stack(transformed_coeffs_list, dim=1) # (B_flat, C, H_patch, W_patch)
        flattened_dct_features = stacked_norm_coeffs.reshape(B_flat, -1)
        return flattened_dct_features

    @staticmethod
    def reconstruct_patches_from_2d_dct(
        dct_features_norm_flat: torch.Tensor,
        num_channels: int,
        target_patch_h: int,
        target_patch_w: int,
        norm_type: str = "tanh",
        norm_global_scale: float = 200.0,
        norm_tanh_scale: float = 60.0
    ) -> torch.Tensor:
        """
        Reconstructs image patches from normalized flat DCT features.
        """
        if not TORCH_DCT_AVAILABLE or idct_2d is None:
            logger.error("reconstruct_patches_from_2d_dct: torch-dct not available or idct_2d is None. Returning zeros.")
            return torch.zeros(dct_features_norm_flat.shape[0], num_channels, target_patch_h, target_patch_w, device=dct_features_norm_flat.device, dtype=dct_features_norm_flat.dtype)

        B_total_regions = dct_features_norm_flat.shape[0]
        expected_feat_dim = num_channels * target_patch_h * target_patch_w
        if dct_features_norm_flat.shape[1] != expected_feat_dim:
            raise ValueError(f"DCT feature dim mismatch. Expected {expected_feat_dim}, got {dct_features_norm_flat.shape[1]}")

        # Reshape to (B_total_regions, C, H_patch, W_patch)
        norm_coeffs_structured = dct_features_norm_flat.view(
            B_total_regions, num_channels, target_patch_h, target_patch_w
        )
        
        unnorm_coeffs_list = []
        for c_idx in range(num_channels):
            norm_coeffs_ch = norm_coeffs_structured[:, c_idx, :, :]
            unnorm_coeffs_ch: torch.Tensor
            if norm_type == "none": unnorm_coeffs_ch = norm_coeffs_ch
            elif norm_type == "global_scale": unnorm_coeffs_ch = norm_coeffs_ch * norm_global_scale
            elif norm_type == "tanh":
                input_dtype = norm_coeffs_ch.dtype
                compute_dtype = torch.float32 if input_dtype in [torch.float16, torch.bfloat16] else input_dtype
                eps_clamp_atanh = torch.finfo(compute_dtype).eps * 8
                upper_b_atanh = torch.tensor(1.0 - eps_clamp_atanh, dtype=compute_dtype, device=norm_coeffs_ch.device)
                lower_b_atanh = torch.tensor(-1.0 + eps_clamp_atanh, dtype=compute_dtype, device=norm_coeffs_ch.device)
                
                unscaled_ch = torch.atanh(torch.clamp(norm_coeffs_ch.to(compute_dtype), min=lower_b_atanh, max=upper_b_atanh)) * norm_tanh_scale
                unnorm_coeffs_ch = unscaled_ch.to(input_dtype)
            else: unnorm_coeffs_ch = norm_coeffs_ch
            unnorm_coeffs_list.append(unnorm_coeffs_ch)

        unnorm_coeffs_stacked_c = torch.stack(unnorm_coeffs_list, dim=1) # (B_total_regions, C, H, W)
        
        # idct_2d expects (..., H, W)
        reconstructed_patches_list_c = []
        for c_idx in range(num_channels):
            channel_unnorm_coeffs = unnorm_coeffs_stacked_c[:, c_idx, :, :]
            # Ensure float32 for idct_2d if current dtype is not
            input_dtype_idct = channel_unnorm_coeffs.dtype
            compute_dtype_idct = torch.float32 if input_dtype_idct != torch.float32 else input_dtype_idct
            
            recons_ch = idct_2d(channel_unnorm_coeffs.to(compute_dtype_idct), norm='ortho')
            reconstructed_patches_list_c.append(recons_ch.to(input_dtype_idct)) # Convert back
        
        reconstructed_patches = torch.stack(reconstructed_patches_list_c, dim=1)
        return reconstructed_patches

# =====================================================================
# Image Assembly Utility
# =====================================================================
class ImageAssemblyUtils:
    @staticmethod
    def assemble_frames_from_patches(
        patches_batch: torch.Tensor, # (B, N_frames, N_regions, C, H_patch, W_patch)
        bboxes_batch: torch.Tensor,  # (B, N_frames, N_regions, 4) [x1, y1, x2, y2]
        target_image_size: Tuple[int, int], # (H_img, W_img)
        output_range: Tuple[float, float] = (0.0, 1.0) # e.g. (0,1) or (-1,1)
    ) -> torch.Tensor:
        B, N_f, N_r, C, H_patch, W_patch = patches_batch.shape
        H_img, W_img = target_image_size
        device = patches_batch.device
        dtype = patches_batch.dtype

        all_assembled_frames = torch.zeros(B, N_f, C, H_img, W_img, device=device, dtype=dtype)

        for b_idx in range(B):
            for f_idx in range(N_f):
                canvas = torch.zeros(C, H_img, W_img, device=device, dtype=dtype)
                count_map = torch.zeros(1, H_img, W_img, device=device, dtype=dtype) 

                for r_idx in range(N_r):
                    patch = patches_batch[b_idx, f_idx, r_idx] # (C, H_patch, W_patch)
                    x1, y1, x2, y2 = bboxes_batch[b_idx, f_idx, r_idx].round().int().tolist()

                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(W_img, x2), min(H_img, y2)
                    if x1_c >= x2_c or y1_c >= y2_c: continue 

                    target_h_bbox = y2_c - y1_c
                    target_w_bbox = x2_c - x1_c
                    if target_h_bbox <=0 or target_w_bbox <=0: continue

                    if H_patch != target_h_bbox or W_patch != target_w_bbox:
                        resized_patch = TF.resize(patch, [target_h_bbox, target_w_bbox], antialias=True)
                    else:
                        resized_patch = patch
                    
                    canvas[:, y1_c:y2_c, x1_c:x2_c] += resized_patch
                    count_map[:, y1_c:y2_c, x1_c:x2_c] += 1.0
                
                assembled_frame = torch.where(count_map > 0, canvas / (count_map + EPS), canvas)
                all_assembled_frames[b_idx, f_idx] = assembled_frame
        
        if output_range is not None:
            all_assembled_frames = torch.clamp(all_assembled_frames, min=output_range[0], max=output_range[1])
            
        return all_assembled_frames

# =====================================================================
# Geometric, Optimizer, WuBu Core Components (Largely Unchanged from audio script)
# =====================================================================


class HyperbolicUtils: # Copied from audio script
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

class Manifold: # Copied from audio script
    def __init__(self, c_scalar=0.0): self.c = float(c_scalar)
    def proju(self, p: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def logmap0(self, p: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5): raise NotImplementedError
    def expmap(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor: return self.proju(p + dp) if self.c > 0 else p + dp
    @property
    def name(self) -> str: return self.__class__.__name__

class PoincareBall(Manifold): # Copied from audio script
    def __init__(self, c_scalar: float = 1.0):
        super().__init__(c_scalar)
        self.logger = logging.getLogger("WuBuGAADHybridGenV03.PoincareBall") # Updated logger name
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0: self.c = 0.0; self.k = 0.; self.sqrt_c = 0.; self.radius = float('inf')
        else: self.c = c_scalar; self.k = -self.c; self.sqrt_c = math.sqrt(self.c); self.radius = 1. / self.sqrt_c
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
        p_projected = self.proju(p); p_norm_sq = torch.sum(p_projected.pow(2), dim=-1, keepdim=True)
        max_sq_norm_val = self.max_norm**2 if self.radius != float('inf') else float('inf'); p_norm_sq_clamped = torch.clamp(p_norm_sq, min=0.0, max=max_sq_norm_val)
        term_inside_paren = 1. - self.c * p_norm_sq_clamped; lambda_p_factor = term_inside_paren / 2.0
        riemannian_scaling_factor = lambda_p_factor.pow(2); final_factor = torch.clamp(riemannian_scaling_factor, min=EPS); r_grad = final_factor * dp
        if not torch.isfinite(r_grad).all():
            dp_norm_str = dp.norm().item() if torch.isfinite(dp).all() else 'NaN'
            p_norm_sq_str = p_norm_sq.mean().item() if p_norm_sq.numel()>0 and torch.isfinite(p_norm_sq).all() else 'NaN'
            p_proj_norm_str = p_projected.norm().item() if torch.isfinite(p_projected).all() else 'NaN'
            factor_str = final_factor.mean().item() if final_factor.numel()>0 else 'N/A'
            self.logger.warning(f"Non-finite Riemannian gradient computed in egrad2rgrad for param shape {p.shape}, c={self.c}. Factor: {factor_str}, dp_norm: {dp_norm_str}. Input p_norm_sq: {p_norm_sq_str}. Projected p norm: {p_proj_norm_str}")
            return dp
        return r_grad
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad():
            w.data.uniform_(-irange, irange)
            if self.c > 0 : w.data = self.expmap0(w.data); w.data = self.proju(w.data)

def init_weights_general(m): # Copied from audio script
    if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', False)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu'); nn.init.zeros_(m.bias) if m.bias is not None else None

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor: return F.softplus(param_unconstrained) + min_val

class BoundaryManifoldHyperbolic(nn.Module): # Copied from audio script
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold_c: float):
        super().__init__(); self.level_idx = level_idx; self.num_points = num_points; self.point_dim = point_dim; self.current_manifold_c = initial_manifold_c
        if num_points > 0 and point_dim > 0: self.hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim)); PoincareBall(initial_manifold_c).init_weights(self.hyperbolic_points_params, irange=1e-3); setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(initial_manifold_c))
        else: self.register_parameter('hyperbolic_points_params', None)
    def set_current_manifold_c(self, c_scalar: float): self.current_manifold_c = c_scalar; setattr(self.hyperbolic_points_params, 'manifold', PoincareBall(c_scalar)) if self.hyperbolic_points_params is not None else None
    def get_points(self) -> Optional[torch.Tensor]: return PoincareBall(self.current_manifold_c).proju(self.hyperbolic_points_params) if self.hyperbolic_points_params is not None else None

def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor: # Copied from audio script
    axis = F.normalize(axis, p=2, dim=-1); angle_half = angle_rad / 2.0; q_w = torch.cos(angle_half); q_xyz = axis * torch.sin(angle_half); return torch.cat([q_w, q_xyz], dim=-1)
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor: # Copied from audio script
    w1, x1, y1, z1 = q1.unbind(-1); w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2; return torch.stack([w, x, y, z], dim=-1)
def quaternion_apply_to_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: # Copied from audio script
    v_quat = F.pad(v, (1, 0), value=0); q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1); rotated_v_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj); return rotated_v_quat[..., 1:]

class HyperbolicInterLevelTransform(nn.Module): # Copied from audio script
    def __init__(self, in_dim: int, out_dim: int, initial_c_in: float, initial_c_out: float, transform_type: str, hidden_dim: Optional[int] = None, dropout: float = 0.1, use_rotation: bool = False, phi_influence_rotation_init: bool = False, level_idx_for_phi: int = 0):
        super().__init__(); self.in_dim, self.out_dim, self.transform_type = in_dim, out_dim, transform_type.lower(); self.use_rotation = use_rotation; self.rotation_module = None; self.phi_influence_rotation_init = phi_influence_rotation_init; current_logger=logging.getLogger("WuBuGAADHybridGenV03.HILT")
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

class HyperbolicWuBuNestingLevel(nn.Module): # Copied from audio script
    def __init__(self, level_idx: int, dim: int, config: Dict, initial_curvature_val_base: float):
        super().__init__(); self.level_idx, self.dim, self.config = level_idx, dim, config; self.logger = logging.getLogger(f"WuBuGAADHybridGenV03.Level{self.level_idx}")
        current_logger = self.logger
        self.phi_influence_curvature = config.get("phi_influence_curvature", False)

        # Effective exponent for PHI scaling, e.g., for level_idx=0 -> -1.5, level_idx=1 -> -0.5, level_idx=2 -> 0.5, level_idx=3 -> 1.5, level_idx=4 -> -1.5 ...
        # This creates a cycle of 4 scaling factors relative to PHI.
        # An exemplary calculation of this formula has been formally verified.
        phi_scaling_exponent = (level_idx % 4) - 1.5 
        self.initial_curvature_val = initial_curvature_val_base * (PHI**phi_scaling_exponent if self.phi_influence_curvature else 1.0)
        
        phi_base_str = f" (PhiBase {initial_curvature_val_base:.2f})" if self.phi_influence_curvature else ""; current_logger.info(f"InitialC={self.initial_curvature_val:.2f}{phi_base_str}")
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
        num_bounds_list = config.get("boundary_points_per_level", [0]) # Video default was [4]
        num_boundaries_val = num_bounds_list[level_idx] if level_idx < len(num_bounds_list) else (num_bounds_list[-1] if num_bounds_list else 0)
        if self.dim > 0 and num_boundaries_val > 0: self.boundary_manifold_module = BoundaryManifoldHyperbolic(level_idx, num_boundaries_val, dim, initial_manifold_c=self.initial_curvature_val)
        else: self.boundary_manifold_module = None
        self.comb_in_dim = self.dim
        if self.relative_vector_aggregation not in ['none', None] and num_boundaries_val > 0: self.comb_in_dim += self.dim
        if self.use_ld: self.comb_in_dim += self.dim
        comb_h_dims_cfg = config.get("tangent_input_combination_dims", [max(16, self.comb_in_dim // 2)]) if self.comb_in_dim > 0 else []; comb_h_dims = comb_h_dims_cfg if isinstance(comb_h_dims_cfg, list) else [comb_h_dims_cfg]; layers = []; in_d = self.comb_in_dim
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
        if relative_vectors_tangent_in is not None and self.relative_vector_aggregation not in ['none', None] and self.boundary_manifold_module is not None and self.boundary_manifold_module.num_points > 0:
            if relative_vectors_tangent_in.shape[0] != B_prime: raise ValueError(f"RelVec shape mismatch: {relative_vectors_tangent_in.shape[0]} != B' {B_prime}")
            tan_rel_component = relative_vectors_tangent_in.to(dtype_to_use)
        if self.use_ld and self.level_descriptor_param is not None: ld_point_self_hyperbolic = current_manifold_obj.proju(self.level_descriptor_param.to(dtype_to_use))
        tan_desc_prev_level_component = torch.zeros_like(tan_main_component)
        if descriptor_point_in_hyperbolic is not None and self.use_ld:
            if descriptor_point_in_hyperbolic.shape[0] != B_prime: raise ValueError(f"DescIn shape mismatch: {descriptor_point_in_hyperbolic.shape[0]} != B' {B_prime}")
            desc_in_proj = current_manifold_obj.proju(descriptor_point_in_hyperbolic.to(dtype_to_use)); tan_desc_prev_level_component = current_manifold_obj.logmap0(desc_in_proj)
        inputs_for_combiner = [tan_main_component]
        if self.relative_vector_aggregation not in ['none', None] and self.boundary_manifold_module is not None and self.boundary_manifold_module.num_points > 0: inputs_for_combiner.append(tan_rel_component)
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
        boundary_points_this_level_hyperbolic = None
        if self.boundary_manifold_module and self.boundary_manifold_module.get_points() is not None: boundary_points_this_level_hyperbolic = self.boundary_manifold_module.get_points().to(dtype=dtype_to_use, device=dev)
        descriptor_point_out_for_transform_hyperbolic = None
        if ld_point_self_hyperbolic is not None: descriptor_point_out_for_transform_hyperbolic = ld_point_self_hyperbolic.unsqueeze(0).expand(B_prime, -1).to(dtype=dtype_to_use) if ld_point_self_hyperbolic.dim() == 1 else ld_point_self_hyperbolic.to(dtype=dtype_to_use)
        output_dtype = point_in_hyperbolic.dtype
        point_out = point_this_level_out_hyperbolic.to(dtype=output_dtype); tangent_out = tangent_out_for_aggregation.to(dtype=output_dtype); desc_out = descriptor_point_out_for_transform_hyperbolic.to(dtype=output_dtype) if descriptor_point_out_for_transform_hyperbolic is not None else None; bound_out = boundary_points_this_level_hyperbolic.to(dtype=output_dtype) if boundary_points_this_level_hyperbolic is not None else None; sigma_out = current_sigma_out_tensor.to(dtype=output_dtype)
        return (point_out, tangent_out, desc_out, bound_out, sigma_out)

class FullyHyperbolicWuBuNestingModel(nn.Module): # Copied from audio script
    def __init__(self, input_tangent_dim: int, output_tangent_dim: int, config: Dict):
        super().__init__(); current_logger=logging.getLogger("WuBuGAADHybridGenV03.WuBuModel"); self.input_tangent_dim, self.output_tangent_dim, self.config = input_tangent_dim, output_tangent_dim, config; self.num_levels = config.get("num_levels", 3); assert self.num_levels >= 0; self.hyperbolic_dims_list = config.get("hyperbolic_dims", []); self.initial_curvatures_list = config.get("initial_curvatures", []); self.dropout_val = config.get("dropout", 0.1); self.relative_vector_aggregation_mode = config.get("relative_vector_aggregation", "mean"); self.aggregation_method_mode = config.get("aggregation_method", "concat_tangent"); assert self.aggregation_method_mode == "concat_tangent"; self.use_rotation_in_transform_flag = config.get("use_rotation_in_transform", False); self.phi_influence_rotation_init = config.get("phi_influence_rotation_init", False)
        first_level_dim = self.hyperbolic_dims_list[0] if self.num_levels > 0 and self.hyperbolic_dims_list else 0
        if input_tangent_dim > 0 and first_level_dim > 0 and input_tangent_dim != first_level_dim: self.input_tangent_projection = nn.Linear(input_tangent_dim, first_level_dim)
        else: self.input_tangent_projection = nn.Identity()
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
                    if i + 1 < len(self.levels_modulelist) and i + 1 < len(self.hyperbolic_dims_list) and i + 1 < len(self.initial_curvatures_list): t_type = transform_types_list[i] if i < len(transform_types_list) else "linear"; t_hidden = transform_hidden_dims_list[i] if i < len(transform_hidden_dims_list) else None; self.transforms_modulelist.append(HyperbolicInterLevelTransform(self.hyperbolic_dims_list[i], self.hyperbolic_dims_list[i+1], self.initial_curvatures_list[i], self.initial_curvatures_list[i+1], t_type, t_hidden, self.dropout_val, self.use_rotation_in_transform_flag, self.phi_influence_rotation_init, level_idx_for_phi=i))
                    else: current_logger.warning(f"Skipping transform {i} to {i+1} due to insufficient config/levels for next level.")
        actual_output_dims_from_levels = [d for d_idx, d in enumerate(self.hyperbolic_dims_list[:len(self.levels_modulelist)]) if d > 0]; aggregated_tangent_dim_val = sum(actual_output_dims_from_levels) if actual_output_dims_from_levels else input_tangent_dim
        if aggregated_tangent_dim_val > 0 and output_tangent_dim > 0 and aggregated_tangent_dim_val != output_tangent_dim: self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, output_tangent_dim)
        else: self.output_tangent_projection = nn.Identity()
        self.apply(init_weights_general); param_count = sum(p.numel() for p in self.parameters() if p.requires_grad); current_logger.info(f"Levels: {len(self.levels_modulelist)}. Params: {param_count:,}. InDim {input_tangent_dim}, AggDim {aggregated_tangent_dim_val}, OutDim {output_tangent_dim}")
    def forward(self, x_initial_tangent_in: torch.Tensor) -> torch.Tensor:
        input_dim = x_initial_tangent_in.dim(); B_orig, S_orig, D_orig = -1, -1, -1
        if input_dim == 3: B_orig, S_orig, D_orig = x_initial_tangent_in.shape; x_proc = x_initial_tangent_in.reshape(B_orig * S_orig, D_orig); B_prime_for_levels = B_orig * S_orig
        elif input_dim == 2: B_prime, D_orig = x_initial_tangent_in.shape; x_proc = x_initial_tangent_in; B_prime_for_levels = B_prime
        else: raise ValueError(f"WuBuModel expects 2D/3D input, got {input_dim}D")
        if D_orig != self.input_tangent_dim: raise ValueError(f"Input feature dim {D_orig} != model input_tangent_dim {self.input_tangent_dim}")
        if self.num_levels == 0 or not self.levels_modulelist: out_proj = self.output_tangent_projection(x_proc); return out_proj.reshape(B_orig, S_orig, -1) if input_dim==3 else out_proj
        dev = x_proc.device; ref_param_for_dtype = next(iter(self.parameters()), None); dtype_to_use = ref_param_for_dtype.dtype if ref_param_for_dtype is not None else x_proc.dtype; x_proc = x_proc.to(dtype_to_use)
        current_tangent_projected = self.input_tangent_projection(x_proc); current_tangent_for_level0 = self.input_tangent_layernorm(current_tangent_projected)
        level0_module = self.levels_modulelist[0]; c0_val = level0_module.get_current_curvature_scalar(); m0_obj = PoincareBall(c_scalar=c0_val)
        if self.hyperbolic_dims_list[0] > 0: current_point_repr_hyperbolic = m0_obj.expmap0(current_tangent_for_level0)
        else: current_point_repr_hyperbolic = torch.empty(B_prime_for_levels, 0, device=dev, dtype=dtype_to_use)
        level_tangent_outputs_for_aggregation = []; aggregated_relative_vectors_from_prev_transform = None; descriptor_from_prev_transform_hyperbolic = None; sigma_from_prev_level_tensor = torch.tensor(0.0, device=dev, dtype=dtype_to_use)
        for i, level_module in enumerate(self.levels_modulelist):
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation, descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic, sigma_out_of_level_tensor) = level_module(current_point_repr_hyperbolic, aggregated_relative_vectors_from_prev_transform, descriptor_from_prev_transform_hyperbolic, sigma_from_prev_level_tensor)
            if self.hyperbolic_dims_list[i] > 0: level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)
            if i < len(self.levels_modulelist) - 1:
                if i >= len(self.transforms_modulelist): logging.getLogger("WuBuGAADHybridGenV03.WuBuModel").warning(f"Missing transform L{i}->L{i+1}. Stop."); break
                transform_module = self.transforms_modulelist[i]; next_level_module = self.levels_modulelist[i+1]; c_in_for_transform = level_module.get_current_curvature_scalar(); c_out_for_transform = next_level_module.get_current_curvature_scalar()
                (point_transformed_to_next_level_hyperbolic, boundaries_transformed_to_next_level_hyperbolic, descriptor_transformed_to_next_level_hyperbolic) = transform_module(point_out_of_level_hyperbolic, boundary_points_of_level_hyperbolic, descriptor_generated_by_level_hyperbolic, c_in_for_transform, c_out_for_transform)
                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic; descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic; sigma_from_prev_level_tensor = sigma_out_of_level_tensor; aggregated_relative_vectors_from_prev_transform = None
                valid_boundary_conditions = (boundaries_transformed_to_next_level_hyperbolic is not None and self.relative_vector_aggregation_mode not in ['none', None] and self.hyperbolic_dims_list[i+1] > 0 and current_point_repr_hyperbolic.shape[-1] > 0 and self.levels_modulelist[i].boundary_manifold_module is not None and self.levels_modulelist[i].boundary_manifold_module.num_points > 0)
                if valid_boundary_conditions:
                    manifold_next_level_obj = PoincareBall(c_scalar=c_out_for_transform); tan_main_next_level = manifold_next_level_obj.logmap0(current_point_repr_hyperbolic); tan_bounds_next_level = manifold_next_level_obj.logmap0(boundaries_transformed_to_next_level_hyperbolic); tan_bounds_next_level_expanded = tan_bounds_next_level.unsqueeze(0).expand(B_prime_for_levels, -1, -1); relative_tangent_vectors = tan_main_next_level.unsqueeze(1) - tan_bounds_next_level_expanded; agg_mode = self.relative_vector_aggregation_mode
                    if agg_mode == "mean": agg_rel_vec = torch.mean(relative_tangent_vectors, dim=1)
                    elif agg_mode == "sum": agg_rel_vec = torch.sum(relative_tangent_vectors, dim=1)
                    elif agg_mode == "max_norm": norms = torch.norm(relative_tangent_vectors, p=2, dim=-1); best_idx = torch.argmax(norms, dim=1, keepdim=True); best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, relative_tangent_vectors.shape[-1]); agg_rel_vec = torch.gather(relative_tangent_vectors, 1, best_idx_expanded).squeeze(1)
                    else: agg_rel_vec = None
                    if agg_rel_vec is not None and not torch.isfinite(agg_rel_vec).all(): agg_rel_vec = torch.zeros_like(tan_main_next_level)
                    aggregated_relative_vectors_from_prev_transform = agg_rel_vec
        compatible_tangent_outputs = [t_val.to(dtype_to_use) for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation) if t_val is not None and t_idx < len(self.hyperbolic_dims_list) and self.hyperbolic_dims_list[t_idx] > 0 and torch.isfinite(t_val).all()]
        if not compatible_tangent_outputs: out_zeros = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use); return out_zeros.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim == 3 else out_zeros
        aggregated_tangent_final = torch.cat(compatible_tangent_outputs, dim=-1); final_output_flat = self.output_tangent_projection(aggregated_tangent_final)
        if not torch.isfinite(final_output_flat).all(): final_output_flat = torch.nan_to_num(final_output_flat, nan=0.0)
        return final_output_flat.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim == 3 else final_output_flat

class GradientStats: # Copied from audio script
    def __init__(self): self.reset()
    def reset(self): self.total_params_updated=0; self.total_finite_grads_processed=0; self.total_non_finite_grads_encountered=0; self.params_skipped_due_non_finite_grad=0; self.max_grad_norm_observed=0.; self.step_summary={}
    def record_param_grad(self, grad_is_finite: bool, original_norm_if_finite: Optional[float] = None):
        if grad_is_finite: self.total_finite_grads_processed += 1; self.max_grad_norm_observed = max(self.max_grad_norm_observed, original_norm_if_finite if original_norm_if_finite is not None else 0.0)
        else: self.total_non_finite_grads_encountered += 1; self.params_skipped_due_non_finite_grad += 1
    def finalize_step_stats(self, num_params_in_optimizer_step: int): self.total_params_updated=num_params_in_optimizer_step-self.params_skipped_due_non_finite_grad; self.step_summary={"params_in_step":num_params_in_optimizer_step, "params_updated":self.total_params_updated, "params_skipped_non_finite_grad":self.params_skipped_due_non_finite_grad, "initial_finite_grads":self.total_finite_grads_processed, "initial_non_finite_grads":self.total_non_finite_grads_encountered, "max_finite_grad_norm_observed":self.max_grad_norm_observed}
    def get_step_summary_for_logging(self) -> dict: return self.step_summary.copy()


class HAKMEMQController:
    def __init__(self, q_learning_rate: float = 0.01, discount_factor: float = 0.90,
                 epsilon_start: float = 0.5, epsilon_min: float = 0.05, epsilon_decay: float = 0.9995,
                 lr_scale_options: list[float] | None = None,
                 momentum_scale_options: list[float] | None = None,
                 lambda_kl_scale_options: list[float] | None = None,
                 max_q_table_size: int = 25000,
                 state_history_len: int = 5,
                 lambda_kl_state_history_len: int = 5,
                 reward_clipping: tuple[float, float] | None = (-2.0, 2.0),
                 q_value_clipping: tuple[float, float] | None = (-30.0, 30.0),
                 logger_name_suffix: Optional[str] = None,
                 num_probation_steps: Optional[int] = None,
                 lkl_num_probation_steps: Optional[int] = None
                ):
        self.q_table: dict[tuple, dict[str, np.ndarray]] = {}; self.alpha = q_learning_rate; self.gamma = discount_factor; self.epsilon_start = epsilon_start; self.epsilon = self.epsilon_start; self.epsilon_min = epsilon_min; self.epsilon_decay = epsilon_decay; self.reward_clipping = reward_clipping; self.q_value_clipping = q_value_clipping; self.current_lambda_kl: float = 0.0001
        _lr_options = lr_scale_options if lr_scale_options is not None else [0.8, 0.9, 1.0, 1.1, 1.2]; _mom_options = momentum_scale_options if momentum_scale_options is not None else [0.95, 0.98, 1.0, 1.01, 1.02]; _lkl_options = lambda_kl_scale_options if lambda_kl_scale_options is not None else [0.94, 0.97, 1.0, 1.03, 1.06]
        self.action_ranges = {'lr_scale': np.array(_lr_options, dtype=np.float32), 'momentum_scale': np.array(_mom_options, dtype=np.float32), 'lambda_kl_scale': np.array(_lkl_options, dtype=np.float32)}; self.num_actions = {p_type: len(actions) for p_type, actions in self.action_ranges.items()}
        self.state_history_len = max(3, state_history_len); self.loss_g_total_hist = deque(maxlen=self.state_history_len); self.loss_g_recon_hist = deque(maxlen=self.state_history_len); self.loss_g_kl_hist = deque(maxlen=self.state_history_len); self.loss_g_adv_hist = deque(maxlen=self.state_history_len); self.loss_d_total_hist = deque(maxlen=self.state_history_len); self.loss_d_real_hist = deque(maxlen=self.state_history_len); self.loss_d_fake_hist = deque(maxlen=self.state_history_len)
        self.lambda_kl_state_history_len = max(2, lambda_kl_state_history_len); self.interval_avg_recon_hist = deque(maxlen=self.lambda_kl_state_history_len); self.interval_avg_kl_div_hist = deque(maxlen=self.lambda_kl_state_history_len); self.interval_avg_d_total_hist = deque(maxlen=self.lambda_kl_state_history_len); self.interval_val_metric_hist = deque(maxlen=self.lambda_kl_state_history_len)
        self.prev_lr_mom_state: tuple | None = None; self.prev_lr_mom_action: dict[str, float] | None = None; self.prev_lambda_kl_state: tuple | None = None; self.prev_lambda_kl_action: dict[str, float] | None = None; self.reward_hist = deque(maxlen=100)
        self.max_q_table_size = max_q_table_size; self.q_table_access_count: dict[tuple, int] = defaultdict(int); self.q_table_creation_time: dict[tuple, float] = {}; self.q_table_last_access_time: dict[tuple, float] = {}
        self.reward_weights = {"g_recon_improvement": 2.5, "g_adv_improvement": 1.2, "g_kl_control_penalty": 0.3, "g_loss_stability": 0.1, "d_balance_target": 1.5, "d_real_low_bonus": 0.7, "d_fake_low_meaningful_bonus": 0.7, "d_loss_stability": 0.1, "gan_balance_g_bonus": 0.3, "gan_balance_d_penalty": 0.3, "oscillation_penalty": 0.25, "extreme_loss_penalty": 0.75, "lambda_kl_recon_focus": 1.5, "lambda_kl_kl_target_range": 1.0, "lambda_kl_val_metric_improvement": 2.0, "lambda_kl_stability_penalty": 0.5}
        
        base_logger_name_q = "WuBuGAADHybridGenV03.QController"
        effective_logger_name_q = f"{base_logger_name_q}.{logger_name_suffix}" if logger_name_suffix else base_logger_name_q
        self.logger = logging.getLogger(effective_logger_name_q)
        self.logger.info(f"HAKMEMQController ({effective_logger_name_q}) initialized. Eps: {self.epsilon_start:.2f}->{self.epsilon_min:.2f}")
        self._internal_step_counter = 0

        self.num_probation_steps = num_probation_steps
        self.lkl_num_probation_steps = lkl_num_probation_steps
        self.on_probation = False
        self.current_probation_step = 0
        self.lkl_on_probation = False
        self.lkl_current_probation_step = 0
        self.epsilon_boost_active_steps = 0 
        self.epsilon_boost_target = self.epsilon_start
        self.original_epsilon_before_boost = self.epsilon_start

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
        
        if hasattr(self, 'epsilon_boost_active_steps') and self.epsilon_boost_active_steps > 0:
            current_epsilon_for_choice = self.epsilon_boost_target
            self.epsilon_boost_active_steps -= 1
            if self.epsilon_boost_active_steps == 0:
                self.logger.info(f"Q-Ctrl exploration boost finished. Epsilon reverting to {self.original_epsilon_before_boost:.3f} (then decays).")
                self.epsilon = self.original_epsilon_before_boost 
        else: 
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            current_epsilon_for_choice = self.epsilon
            
        chosen_actions = {}
        for param_type in action_types_to_choose:
            q_values = self.q_table[state].get(param_type); action_space = self.action_ranges[param_type]
            if q_values is None: self.logger.error(f"Q-values for {param_type} missing in state {state}. Choosing default."); chosen_actions[param_type] = default_actions[param_type]; continue
            
            if random.random() < current_epsilon_for_choice: 
                chosen_idx = random.randrange(len(action_space))
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
    def reset_q_learning_state(self, reset_q_table: bool = True, reset_epsilon: bool = True, context_msg: str = "Generic Reset", start_probation: bool = False):
        if reset_q_table: self.q_table.clear(); self.q_table_access_count.clear(); self.q_table_creation_time.clear(); self.q_table_last_access_time.clear(); self.logger.info(f"Q-table cleared. Context: {context_msg}")
        if reset_epsilon: self.epsilon = self.epsilon_start; self.logger.info(f"Epsilon reset to start value: {self.epsilon:.3f}. Context: {context_msg}")
        self.prev_lr_mom_state = None; self.prev_lr_mom_action = None; self.prev_lambda_kl_state = None; self.prev_lambda_kl_action = None; self.reward_hist.clear()
        self.on_probation = False; self.current_probation_step = 0; self.lkl_on_probation = False; self.lkl_current_probation_step = 0
        if start_probation: self.start_probation(); self.start_lkl_probation()
    def start_probation(self):
        if self.num_probation_steps is None or self.num_probation_steps <=0: self.on_probation=False; return
        self.on_probation = True; self.current_probation_step = 0; self.logger.info(f"LR/Mom Q-Controller entering probation for {self.num_probation_steps} steps.")
    def start_lkl_probation(self):
        if self.lkl_num_probation_steps is None or self.lkl_num_probation_steps <=0: self.lkl_on_probation=False; return
        self.lkl_on_probation = True; self.lkl_current_probation_step = 0; self.logger.info(f"Lambda_KL Q-Controller entering probation for {self.lkl_num_probation_steps} LKL updates.")
    def force_exploration_boost(self, num_steps: int, target_epsilon: float):
        self.epsilon_boost_active_steps = num_steps; self.epsilon_boost_target = target_epsilon; self.original_epsilon_before_boost = self.epsilon; self.logger.info(f"Q-Ctrl exploration boosted for {num_steps} steps to eps {target_epsilon:.2f}.")



class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[nn.Parameter],
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 weight_decay: float = 0.01,
                 max_grad_norm_risgd: float = 1.0,
                 q_learning_config: Optional[Dict] = None,
                 optimizer_type: str = "generator", # This is the key argument for the type
                 q_logger_suffix: Optional[str] = None # New argument for Q-controller's logger
                ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            initial_lr=lr, 
            momentum=momentum,
            initial_momentum=momentum, 
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        self.optimizer_type = optimizer_type.lower()
        # The check is now correct as `optimizer_type` will be "generator" or "discriminator"
        if self.optimizer_type not in ["generator", "discriminator"]:
            raise ValueError("optimizer_type must be 'generator' or 'discriminator'")

        self.logger = logging.getLogger(f"WuBuGAADHybridGenV03.RiSGD.{self.optimizer_type.capitalize()}") # Use V03

        if isinstance(q_learning_config, dict):
            q_cfg_for_internal_q = q_learning_config.copy()
            # Use the new q_logger_suffix if provided, otherwise use the optimizer_type for Q-controller logger
            effective_q_logger_suffix = q_logger_suffix if q_logger_suffix else self.optimizer_type.capitalize()
            self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(
                **q_cfg_for_internal_q,
                logger_name_suffix=effective_q_logger_suffix # Pass to HAKMEMQ
            )
        else:
            self.q_controller = None
        
        self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled for {self.optimizer_type} optimizer.")

        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count_internal = 0
        self.grad_stats = GradientStats()

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state.setdefault(p, {})
    # ... (rest of RiemannianEnhancedSGD methods remain the same) ...
    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)

    def q_controller_update_and_set_hyperparams(self,
                                                avg_losses_dict: Dict[str, Optional[float]],
                                                current_lambda_kl_value: Optional[float] = None
                                               ):
        if not self.q_controller:
            return

        finite_losses_for_q_state: Dict[str, float] = {
            k: v for k, v in avg_losses_dict.items() if v is not None and np.isfinite(v)
        }

        is_gen_q = (self.optimizer_type == "generator")
        if is_gen_q:
            required_keys = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        else: # Discriminator
            required_keys = ['loss_d_total', 'loss_g_total', 'loss_g_adv', 'loss_d_real', 'loss_d_fake']


        if not all(key in finite_losses_for_q_state for key in required_keys):
            self.logger.debug(f"QCtrl ({self.optimizer_type}): Insufficient finite losses for LR/Mom state. Skipping Q-update. "
                              f"Need: {required_keys}, Got: {list(finite_losses_for_q_state.keys())}")
            return

        if hasattr(self.q_controller, 'set_current_lambda_kl') and current_lambda_kl_value is not None:
            self.q_controller.set_current_lambda_kl(current_lambda_kl_value)

        current_lr_for_q_state = self.param_groups[0]['lr']
        current_mom_for_q_state = self.param_groups[0]['momentum']

        q_state_current = self.q_controller.get_lr_mom_state(
            finite_losses_for_q_state, current_lr_for_q_state, current_mom_for_q_state,
            is_generator_q=is_gen_q
        )

        if self.q_controller.prev_lr_mom_state is not None and \
           self.q_controller.prev_lr_mom_action is not None and \
           q_state_current is not None:
            reward = self.q_controller.compute_lr_mom_reward(finite_losses_for_q_state, is_generator_q=is_gen_q)
            if np.isfinite(reward):
                self.q_controller.update_q_values(
                    self.q_controller.prev_lr_mom_state,
                    self.q_controller.prev_lr_mom_action,
                    reward,
                    q_state_current,
                    mode='lr_mom'
                )
        elif q_state_current is not None and hasattr(self.q_controller, 'set_initial_losses'):
            # Initialize history if it's the first valid state
            self.q_controller.set_initial_losses(finite_losses_for_q_state, is_generator_q=is_gen_q)

        self.q_controller.prev_lr_mom_state = q_state_current
        action_for_upcoming_step = self.q_controller.choose_action(q_state_current, mode='lr_mom')
        # self.q_controller.prev_lr_mom_action is set internally by choose_action

        if action_for_upcoming_step:
            for group in self.param_groups:
                base_lr = group['initial_lr']
                base_mom = group['initial_momentum']

                group['lr'] = float(np.clip(base_lr * action_for_upcoming_step.get('lr_scale', 1.0), 1e-8, 1.0))
                group['momentum'] = float(np.clip(base_mom * action_for_upcoming_step.get('momentum_scale', 1.0), 0.0, 0.999))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            # initial_lr = group['initial_lr'] # For logging or reference if needed

            for p in group['params']:
                if p.grad is None:
                    continue
                if not p.requires_grad:
                    self.logger.warning(f"Parameter {p.shape} has grad but requires_grad is False. Skipping.")
                    continue

                grad = p.grad

                if not torch.isfinite(grad).all():
                    self.logger.warning(f"Optimizer step: Non-finite gradient for param shape {p.shape} "
                                        f"({self.optimizer_type}). Skipping update for this parameter.")
                    self.state[p].pop('momentum_buffer', None)
                    continue

                if self.max_grad_norm_risgd > 0 and self.max_grad_norm_risgd != float('inf') :
                    param_grad_norm = grad.norm().item()
                    if param_grad_norm > self.max_grad_norm_risgd:
                        clip_coef = self.max_grad_norm_risgd / (param_grad_norm + EPS) # Ensure EPS is defined
                        grad.mul_(clip_coef)

                manifold: Optional[Manifold] = getattr(p, 'manifold', None)

                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    p_projected_on_manifold = manifold.proju(p)
                    grad_eff = grad.clone()
                    if weight_decay != 0:
                        grad_eff.add_(p, alpha=weight_decay)

                    try:
                        riemannian_grad = manifold.egrad2rgrad(p_projected_on_manifold, grad_eff)
                    except Exception as e_egrad:
                        self.logger.error(f"egrad2rgrad failed for P:{p.shape} (c={manifold.c:.2e}): {e_egrad}. Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue

                    if not torch.isfinite(riemannian_grad).all():
                        self.logger.warning(f"Non-finite Riemannian grad for P:{p.shape} (c={manifold.c:.2e}). Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue

                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None or buf.shape != riemannian_grad.shape: # Handle new buffer or shape mismatch
                            buf = torch.clone(riemannian_grad).detach()
                        else:
                            buf.mul_(momentum).add_(riemannian_grad)
                        self.state[p]['momentum_buffer'] = buf
                    else:
                        buf = riemannian_grad

                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite momentum buffer for P:{p.shape} (c={manifold.c:.2e}). Resetting buffer.")
                        buf.zero_()
                        self.state[p]['momentum_buffer'] = buf # Store reset buffer

                    expmap_tangent_vector = buf.mul(-lr)
                    if not torch.isfinite(expmap_tangent_vector).all():
                        self.logger.warning(f"Non-finite tangent vector for expmap P:{p.shape} (c={manifold.c:.2e}). Skipping param update.")
                        continue

                    try:
                        new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
                        if not torch.isfinite(new_p_candidate).all():
                            self.logger.warning(f"Expmap resulted in non-finite P:{p.shape} (c={manifold.c:.2e}). Projecting and zeroing momentum.")
                            p.data = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0))
                            if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()
                        else:
                            p.data = manifold.proju(new_p_candidate)
                    except Exception as e_expmap:
                        self.logger.error(f"Expmap failed for P:{p.shape} (c={manifold.c:.2e}): {e_expmap}. Zeroing momentum.")
                        if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()
                        continue

                    if not torch.isfinite(p.data).all():
                        self.logger.error(f"Parameter P:{p.shape} (c={manifold.c:.2e}) became non-finite after update. Resetting to origin.")
                        p.data = manifold.expmap0(torch.zeros_like(p.data, device=p.device))
                        if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()

                else: # Euclidean Parameter Update
                    grad_eff_euc = grad.clone()
                    if weight_decay != 0:
                        grad_eff_euc.add_(p, alpha=weight_decay)

                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None or buf.shape != grad_eff_euc.shape: # Handle new buffer or shape mismatch
                            buf = torch.clone(grad_eff_euc).detach()
                        else:
                            buf.mul_(momentum).add_(grad_eff_euc)
                        self.state[p]['momentum_buffer'] = buf
                    else:
                        buf = grad_eff_euc

                    if not torch.isfinite(buf).all():
                        self.logger.warning(f"Non-finite Euclidean momentum buffer for P:{p.shape}. Resetting buffer.")
                        buf.zero_()
                        self.state[p]['momentum_buffer'] = buf # Store reset buffer

                    p.add_(buf, alpha=-lr)

                    if not torch.isfinite(p.data).all():
                        self.logger.warning(f"Euclidean P:{p.shape} became non-finite. Clamping and zeroing momentum.")
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1e5, neginf=-1e5)
                        if self.state[p].get('momentum_buffer') is not None: self.state[p]['momentum_buffer'].zero_()

        self._step_count_internal += 1
        return loss

    def get_q_controller_info(self) -> Dict:
        return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}

    def get_gradient_stats_summary_optimizer_view(self) -> Dict:
        return self.grad_stats.get_step_summary_for_logging()


# =====================================================================
# GAAD Components 
# =====================================================================
def golden_subdivide_rect_fixed_n(frame_dims:Tuple[int,int], num_regions_target:int, device='cpu', dtype=torch.float, min_size_px=5) -> torch.Tensor: # Copied from audio script
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

def phi_spiral_patch_centers_fixed_n(frame_dims:Tuple[int,int], num_centers:int, device='cpu', dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]: # Copied from audio script
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


def get_rect_energy(analysis_map_slice: torch.Tensor, rect_coords: List[float], min_size_px_energy: int = 1) -> float:
    """Calculates average energy in a rectangle of an analysis map."""
    x1, y1, x2, y2 = [int(round(c)) for c in rect_coords]
    H_map, W_map = analysis_map_slice.shape

    # Clamp coordinates to be within map boundaries
    x1_c, y1_c = max(0, x1), max(0, y1)
    x2_c, y2_c = min(W_map, x2), min(H_map, y2)

    if x1_c >= x2_c or y1_c >= y2_c or (x2_c - x1_c) < min_size_px_energy or (y2_c - y1_c) < min_size_px_energy:
        return 0.0 # Or a very small epsilon to avoid division by zero if area is used

    region = analysis_map_slice[y1_c:y2_c, x1_c:x2_c]
    if region.numel() == 0:
        return 0.0
    return torch.mean(region).item() # Using mean energy

def golden_subdivide_rect_fixed_n_motion_aware(
    frame_dims: Tuple[int, int],
    analysis_map_slice: torch.Tensor, # (H, W) tensor for the current frame/item
    num_regions_target: int,
    device: torch.device, # Changed to torch.device type
    dtype: torch.dtype,   # Changed to torch.dtype type
    min_size_px: int = 5,
    max_depth: int = 6,
    energy_threshold_factor: float = 0.1, # Factor of max energy to consider subdividing
    prioritize_high_energy: bool = True # If true, subdivide high energy regions more
) -> torch.Tensor:
    W, H = frame_dims
    
    # Priority queue stores (-energy, depth, x_off, y_off, w_curr, h_curr)
    # Negative energy because heapq is a min-heap, we want to pop highest energy first
    # Add a unique counter to break ties in priority for stable sorting
    tie_breaker_counter = 0
    initial_energy = get_rect_energy(analysis_map_slice, [0,0,W,H])
    
    # Max energy on the map for relative thresholding
    max_map_energy = torch.max(analysis_map_slice).item() if analysis_map_slice.numel() > 0 else 1.0
    min_energy_to_consider_subdivision = max_map_energy * energy_threshold_factor

    pq_item = (-initial_energy if prioritize_high_energy else initial_energy, 0, tie_breaker_counter, 0.0, 0.0, float(W), float(H))
    rect_priority_queue = [pq_item]
    heapq.heapify(rect_priority_queue)
    
    all_generated_rects_with_energy = [] # Store (energy, [coords])

    while rect_priority_queue and len(all_generated_rects_with_energy) < num_regions_target * 3: # Generate more to select from
        if not rect_priority_queue: break
        
        neg_energy_or_energy, depth, _, x_off, y_off, w_curr, h_curr = heapq.heappop(rect_priority_queue)
        current_energy = -neg_energy_or_energy if prioritize_high_energy else neg_energy_or_energy
        
        # Add the current rect itself to the list of candidates
        # We add it *before* checking subdivision conditions if it's valid sized
        if w_curr >= min_size_px and h_curr >= min_size_px:
            all_generated_rects_with_energy.append((current_energy, [x_off, y_off, x_off + w_curr, y_off + h_curr]))

        if min(w_curr, h_curr) < min_size_px or depth >= max_depth:
            continue
        # Only subdivide if energy is significant or depth is low
        if prioritize_high_energy and current_energy < min_energy_to_consider_subdivision and depth > 1 : # Don't stop too early for low depth
            continue


        is_landscape = w_curr > h_curr + EPS
        is_portrait = h_curr > w_curr + EPS
        children_rects_coords = []

        if is_landscape:
            cut_w = w_curr / PHI
            r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px: children_rects_coords.append({'x':x_off, 'y':y_off, 'w':r1_w, 'h':h_curr})
            if r2_w >= min_size_px: children_rects_coords.append({'x':x_off + r1_w, 'y':y_off, 'w':r2_w, 'h':h_curr})
        elif is_portrait:
            cut_h = h_curr / PHI
            r1_h, r2_h = cut_h, h_curr - cut_h
            if r1_h >= min_size_px: children_rects_coords.append({'x':x_off, 'y':y_off, 'w':w_curr, 'h':r1_h})
            if r2_h >= min_size_px: children_rects_coords.append({'x':x_off, 'y':y_off + r1_h, 'w':w_curr, 'h':r2_h})
        elif abs(w_curr - h_curr) < EPS and w_curr > min_size_px * PHI : # Approx square, subdivide like landscape
            cut_w = w_curr / PHI
            r1_w, r2_w = cut_w, w_curr - cut_w
            if r1_w >= min_size_px: children_rects_coords.append({'x':x_off, 'y':y_off, 'w':r1_w, 'h':h_curr})
            if r2_w >= min_size_px: children_rects_coords.append({'x':x_off + r1_w, 'y':y_off, 'w':r2_w, 'h':h_curr})
        
        for child_r_dict in children_rects_coords:
            ch_x, ch_y, ch_w, ch_h = child_r_dict['x'], child_r_dict['y'], child_r_dict['w'], child_r_dict['h']
            if ch_w >= min_size_px and ch_h >= min_size_px :
                 child_energy = get_rect_energy(analysis_map_slice, [ch_x, ch_y, ch_x + ch_w, ch_y + ch_h])
                 tie_breaker_counter += 1
                 heapq.heappush(rect_priority_queue, (-child_energy if prioritize_high_energy else child_energy, depth + 1, tie_breaker_counter, ch_x, ch_y, ch_w, ch_h))

    # Selection strategy: prioritize high energy, then smaller area for detail
    # We stored (energy, [coords]). Higher raw energy is better.
    # Sort by energy (descending), then by area (ascending for smaller boxes in high energy)
    unique_valid_rects_tensors = []
    seen_hashes = set()
    # Sort existing generated rects before selection
    all_generated_rects_with_energy.sort(key=lambda item: (-item[0], (item[1][2]-item[1][0])*(item[1][3]-item[1][1]))) # -energy for descending, area for ascending

    for energy, r_coords in all_generated_rects_with_energy:
        if len(unique_valid_rects_tensors) >= num_regions_target: break
        # Ensure coords are valid floats for tensor conversion
        r_coords_float = [float(c) for c in r_coords]
        if r_coords_float[0] >= r_coords_float[2] - EPS or r_coords_float[1] >= r_coords_float[3] - EPS: continue
        
        r_tensor = torch.tensor(r_coords_float, dtype=dtype, device=device)
        # Use more robust hashing based on rounded float coordinates
        r_hashable = tuple(round(c, 2) for c in r_coords_float) # Round to 2 decimal places for hashing
        if r_hashable not in seen_hashes:
            unique_valid_rects_tensors.append(r_tensor)
            seen_hashes.add(r_hashable)
            
    selected_rects = unique_valid_rects_tensors[:num_regions_target]

    if not selected_rects and num_regions_target > 0: # If no rects were selected at all
        initial_rect_coords = [0.0, 0.0, float(W), float(H)]
        selected_rects = [torch.tensor(initial_rect_coords, dtype=dtype, device=device)]


    if len(selected_rects) < num_regions_target:
        padding_box = selected_rects[-1].clone() if selected_rects else torch.tensor([0.0,0.0,float(W),float(H)],dtype=dtype,device=device)
        selected_rects.extend([padding_box.clone() for _ in range(num_regions_target - len(selected_rects))])
    
    return torch.stack(selected_rects)
    

def find_motion_hotspots(analysis_map_slice: torch.Tensor, num_hotspots: int = 1, min_distance: float = 10.0, blur_sigma: float = 1.5) -> List[Tuple[float, float]]:
    """Finds N distinct motion hotspots (peaks) in an analysis map."""
    if analysis_map_slice.numel() == 0: return [(analysis_map_slice.shape[1]/2, analysis_map_slice.shape[0]/2)] * num_hotspots
    
    # Optional: Blur the map to find smoother peaks
    if blur_sigma > 0:
        blurred_map = TF.gaussian_blur(analysis_map_slice.unsqueeze(0).unsqueeze(0), kernel_size=5, sigma=blur_sigma).squeeze()
    else:
        blurred_map = analysis_map_slice

    hotspots = []
    temp_map = blurred_map.clone()
    H, W = temp_map.shape

    for _ in range(num_hotspots):
        if temp_map.numel() == 0 or torch.all(temp_map <= 0): break # Stop if no more positive energy
        
        max_val, max_idx_flat = torch.max(temp_map.flatten(), 0)
        if max_val.item() <= EPS: break # Stop if max energy is too low

        peak_y = (max_idx_flat // W).item()
        peak_x = (max_idx_flat % W).item()
        hotspots.append((float(peak_x), float(peak_y)))

        # Suppress this peak and its surroundings to find the next distinct peak
        y_min = max(0, int(peak_y - min_distance))
        y_max = min(H, int(peak_y + min_distance + 1))
        x_min = max(0, int(peak_x - min_distance))
        x_max = min(W, int(peak_x + min_distance + 1))
        temp_map[y_min:y_max, x_min:x_max] = 0 # Suppress region
        
    # If not enough hotspots found, fill with map center
    if not hotspots: hotspots = [(W/2, H/2)] # Default to center if no hotspots
    while len(hotspots) < num_hotspots:
        hotspots.append(hotspots[-1]) # Pad with the last found hotspot or center
        
    return hotspots[:num_hotspots]


def phi_spiral_patch_centers_fixed_n_motion_aware(
    frame_dims: Tuple[int, int],
    analysis_map_slice: Optional[torch.Tensor], # (H, W) tensor for current frame/item, CAN BE NONE
    num_centers: int,
    device: torch.device,
    dtype: torch.dtype,
    num_spiral_arms_per_hotspot: int = 3, # e.g. 3-5 arms per hotspot
    points_per_arm: int = 5,      # e.g. 5-8 points per arm
    base_patch_scale_factor: float = 0.25, # Overall scale for patches
    motion_scale_influence: float = 0.5, # How much motion energy affects patch size (0 to 1)
                                         # 0: no effect, 1: fully inverse proportional
    hotspot_blur_sigma: float = 2.0,
    min_spiral_patch_scale: float = 0.03,
    max_spiral_patch_scale: float = 0.30
) -> Tuple[torch.Tensor, torch.Tensor]:
    W, H = frame_dims
    all_centers_xy_list = []
    all_scale_factors_list = []

    if num_centers <= 0:
        return torch.empty(0, 2, device=device, dtype=dtype), torch.empty(0, 1, device=device, dtype=dtype)

    # Determine hotspots for spiral origins
    if analysis_map_slice is not None and analysis_map_slice.numel() > 0 :
        # Estimate number of distinct hotspots needed based on num_centers and arms
        num_hotspots_to_find = max(1, math.ceil(num_centers / (num_spiral_arms_per_hotspot * points_per_arm if num_spiral_arms_per_hotspot * points_per_arm > 0 else 1) ))
        num_hotspots_to_find = min(num_hotspots_to_find, num_centers, 5) # Cap max hotspots
        
        hotspot_coords_list = find_motion_hotspots(analysis_map_slice, num_hotspots=num_hotspots_to_find, blur_sigma=hotspot_blur_sigma)
    else: # Fallback to geometric center if no analysis map
        hotspot_coords_list = [(W / 2.0, H / 2.0)]

    points_generated_so_far = 0
    for hotspot_idx, (cx, cy) in enumerate(hotspot_coords_list):
        if points_generated_so_far >= num_centers: break

        # Add hotspot itself as a center if it's the first one and we need a central point
        if hotspot_idx == 0 and points_generated_so_far < num_centers:
             all_centers_xy_list.append([cx, cy])
             # Scale for central point can be larger or fixed
             local_energy_at_hotspot = analysis_map_slice[int(cy), int(cx)].item() if analysis_map_slice is not None and 0<=int(cy)<H and 0<=int(cx)<W else 0.1
             max_map_energy = torch.max(analysis_map_slice).item() if analysis_map_slice is not None and analysis_map_slice.numel()>0 else 1.0
             norm_energy = min(1.0, local_energy_at_hotspot / (max_map_energy + EPS) ) if max_map_energy > EPS else 0.1
             
             scale_val = base_patch_scale_factor * (1.0 - motion_scale_influence * norm_energy) # Smaller if high energy
             scale_val = max(min_spiral_patch_scale, min(max_spiral_patch_scale, scale_val))
             all_scale_factors_list.append(scale_val)
             points_generated_so_far += 1
        
        a = 0.03 * min(W, H) # Initial radius for spiral from this hotspot
        b = math.log(PHI) / (math.pi / 2) # Growth rate
        
        for arm_idx in range(num_spiral_arms_per_hotspot):
            if points_generated_so_far >= num_centers: break
            angle_offset = (2 * math.pi / num_spiral_arms_per_hotspot) * arm_idx
            
            for pt_idx_on_arm in range(points_per_arm):
                if points_generated_so_far >= num_centers: break
                
                # More points closer to center of spiral by varying angle step or using logspace for theta
                theta_local = (pt_idx_on_arm + 1) * (PHI * math.pi / (2*points_per_arm)) # Spread points out on arm
                
                r = min(a * math.exp(b * theta_local), max(W, H) * 0.4) # Cap radius
                actual_angle = angle_offset + theta_local
                
                x = max(0.0, min(cx + r * math.cos(actual_angle), float(W - 1)))
                y = max(0.0, min(cy + r * math.sin(actual_angle), float(H - 1)))
                all_centers_xy_list.append([x, y])

                # Scale factor based on distance from this spiral's origin (cx,cy) and local motion energy
                local_energy_at_xy = analysis_map_slice[int(y), int(x)].item() if analysis_map_slice is not None else 0.1
                norm_energy_xy = min(1.0, local_energy_at_xy / (max_map_energy + EPS) ) if analysis_map_slice is not None and max_map_energy > EPS else 0.1

                # Foveated + motion-influenced scaling
                # Smaller patches further out (foveation) AND smaller patches in high motion
                foveation_decay = math.exp(-0.8 * r / (min(W,H)*0.1)) # Stronger decay for foveation
                scale_val_xy = base_patch_scale_factor * foveation_decay * (1.0 - motion_scale_influence * norm_energy_xy)
                scale_val_xy = max(min_spiral_patch_scale, min(max_spiral_patch_scale, scale_val_xy))
                all_scale_factors_list.append(scale_val_xy)
                points_generated_so_far += 1
    
    # Pad if not enough centers generated
    if len(all_centers_xy_list) < num_centers:
        num_to_pad = num_centers - len(all_centers_xy_list)
        last_xy = all_centers_xy_list[-1] if all_centers_xy_list else [W / 2.0, H / 2.0]
        last_scale = all_scale_factors_list[-1] if all_scale_factors_list else base_patch_scale_factor
        all_centers_xy_list.extend([last_xy] * num_to_pad)
        all_scale_factors_list.extend([last_scale] * num_to_pad)

    final_centers = torch.tensor(all_centers_xy_list[:num_centers], dtype=dtype, device=device)
    final_scales = torch.tensor(all_scale_factors_list[:num_centers], dtype=dtype, device=device).unsqueeze(-1)
    
    return final_centers, final_scales

# =====================================================================
# Architectural Components (v0.3 - VAE-GAN Refactor + DFT + DCT)
# =====================================================================


# --- FiLMLayer (Unchanged) ---
class FiLMLayer(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__(); self.channels = channels; self.condition_dim = condition_dim; self.to_gamma_beta = nn.Linear(condition_dim, channels * 2)
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_gamma_beta(condition); gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        # Ensure gamma and beta are broadcastable to x
        if x.dim() == 5: # (B, C, D_frames, H, W) for 3D conv
            gamma = gamma.view(-1, self.channels, 1, 1, 1); beta = beta.view(-1, self.channels, 1, 1, 1)
        elif x.dim() == 4: # (B, C, H, W) for 2D conv (or per-frame features)
            gamma = gamma.view(-1, self.channels, 1, 1); beta = beta.view(-1, self.channels, 1, 1)
        else:
            raise ValueError(f"FiLMLayer input x has unsupported dimension: {x.dim()}")
        return (1 + gamma) * x + beta


class RegionalGeneratorDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, latent_dim: int):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.image_size = args.image_h_w_tuple # Ensure this is set in parse_arguments
        self.num_regions = gaad_config['num_regions']
        self.num_img_channels = video_config['num_channels']
        self.latent_dim = latent_dim
        self.num_predict_frames = video_config["num_predict_frames"]
        self.logger = logging.getLogger("WuBuGAADHybridGenV03.Generator")

        # Determine initial spatial resolution and number of upsampling layers
        min_target_dim = min(self.image_size[0], self.image_size[1])
        if min_target_dim <= 8: self.gen_init_spatial_res = 1
        elif min_target_dim <= 32: self.gen_init_spatial_res = 2
        else: self.gen_init_spatial_res = 4
        
        # Calculate num_upsampling_layers ensuring it's an integer or handle non-power-of-2
        target_upsample_factor = min_target_dim / self.gen_init_spatial_res
        if target_upsample_factor > 0 and math.log2(target_upsample_factor).is_integer():
            self.gen_num_upsampling_layers = int(math.log2(target_upsample_factor))
        else:
            # If not a power of 2, use ceil and potentially an adaptive pool at the end
            self.gen_num_upsampling_layers = max(1, int(math.ceil(math.log2(target_upsample_factor))) if target_upsample_factor > 0 else 1)

        calculated_final_res = self.gen_init_spatial_res * (2**self.gen_num_upsampling_layers)
        self.needs_final_adaptive_pool = (calculated_final_res != min_target_dim)
        if self.needs_final_adaptive_pool:
            self.logger.warning(f"Gen calculated final res {calculated_final_res} vs target {min_target_dim}. Final adaptive pool will be used for pixel output.")

        self.gen_init_channels = min(512, max(128, self.latent_dim * 2)) # From audio script, seems reasonable
        self.gen_temporal_kernel_size = getattr(args, 'gen_temporal_kernel_size', 3)

        self.fc_expand_latent = nn.Linear(self.latent_dim, self.gen_init_channels * self.num_predict_frames * self.gen_init_spatial_res * self.gen_init_spatial_res)

        self.gaad_condition_dim = max(32, self.latent_dim // 4)
        if self.num_regions > 0 and getattr(args, 'gen_use_gaad_film_condition', True):
            self.bbox_feature_dim = 4 # (cx, cy, w, h) normalized
            hidden_bbox_embed_dim = max(self.gaad_condition_dim, self.num_regions * self.bbox_feature_dim // 2)
            self.frame_gaad_embedder = nn.Sequential(
                nn.Linear(self.num_regions * self.bbox_feature_dim, hidden_bbox_embed_dim),
                nn.GELU(),
                nn.Linear(hidden_bbox_embed_dim, self.gaad_condition_dim)
            )
            self.logger.info(f"Generator GAAD-FiLM enabled, cond_dim: {self.gaad_condition_dim}")
        else:
            self.frame_gaad_embedder = None
            self.logger.info("Generator GAAD-FiLM disabled.")

        # Upsampling blocks (3D Convolutions)
        self.upsample_blocks = nn.ModuleList()
        current_channels = self.gen_init_channels
        padding_temp = self.gen_temporal_kernel_size // 2
        # Aim for a reasonable number of channels before splitting to regional heads or final pixel conv
        min_gen_channels_final_block = max(32, self.num_img_channels * 8) 

        for i in range(self.gen_num_upsampling_layers):
            out_channels = max(min_gen_channels_final_block, current_channels // 2) if i < self.gen_num_upsampling_layers -1 else min_gen_channels_final_block
            block = nn.ModuleDict()
            block['conv_transpose'] = nn.ConvTranspose3d(
                current_channels, out_channels,
                kernel_size=(self.gen_temporal_kernel_size, 4, 4), # (D, H, W)
                stride=(1, 2, 2), # Upsample spatially, not temporally here
                padding=(padding_temp, 1, 1),
                bias=False # Bias usually False with Norm layers
            )
            block['norm'] = nn.InstanceNorm3d(out_channels, affine=True) # affine=True is common
            if self.frame_gaad_embedder is not None:
                block['film'] = FiLMLayer(out_channels, self.gaad_condition_dim)
            block['activation'] = nn.GELU()
            self.upsample_blocks.append(block)
            current_channels = out_channels
        
        self.final_dense_feature_channels = current_channels
        
        # --- Spectral Feature Prediction Heads ---
        # These heads operate on RoIAlign'd features from the dense feature volume
        self.gen_patch_h_spectral = args.spectral_patch_size_h
        self.gen_patch_w_spectral = args.spectral_patch_size_w
        self.gen_roi_align_output_spatial = (self.gen_patch_h_spectral, self.gen_patch_w_spectral)
        # Dimension of RoIAlign'd features (flattened) that will be input to MLPs
        roi_feat_dim_for_heads = self.final_dense_feature_channels * self.gen_patch_h_spectral * self.gen_patch_w_spectral

        self.to_dft_coeffs_mlp: Optional[nn.Module] = None
        self.to_dct_coeffs_mlp: Optional[nn.Module] = None

        if args.use_dft_features_appearance:
            dft_w_coeffs_one_sided_gen = self.gen_patch_w_spectral // 2 + 1
            # Output dim: C * 2 (real/imag) * H_patch * W_coeffs_one_sided
            dft_coeff_output_dim_per_region = self.num_img_channels * 2 * self.gen_patch_h_spectral * dft_w_coeffs_one_sided_gen
            hidden_mlp_dft = max(dft_coeff_output_dim_per_region // 2, roi_feat_dim_for_heads // 2, 128) # Sensible hidden dim
            self.to_dft_coeffs_mlp = nn.Sequential(
                nn.Linear(roi_feat_dim_for_heads, hidden_mlp_dft), nn.GELU(),
                nn.Linear(hidden_mlp_dft, dft_coeff_output_dim_per_region)
                # No final activation here, DFT coeffs from Tanh normalization have their own range.
            )
            self.logger.info(f"Generator DFT Head: RoIAlign spatial output {self.gen_roi_align_output_spatial}, Projects {roi_feat_dim_for_heads} to {dft_coeff_output_dim_per_region} DFT coeffs/region.")
        
        if args.use_dct_features_appearance:
            # Output dim: C * H_patch * W_patch
            dct_coeff_output_dim_per_region = self.num_img_channels * self.gen_patch_h_spectral * self.gen_patch_w_spectral
            hidden_mlp_dct = max(dct_coeff_output_dim_per_region // 2, roi_feat_dim_for_heads // 2, 128)
            self.to_dct_coeffs_mlp = nn.Sequential(
                nn.Linear(roi_feat_dim_for_heads, hidden_mlp_dct), nn.GELU(),
                nn.Linear(hidden_mlp_dct, dct_coeff_output_dim_per_region)
                # No final activation here for DCT either if it's Tanh normalized from encoder.
            )
            self.logger.info(f"Generator DCT Head: RoIAlign spatial output {self.gen_roi_align_output_spatial}, Projects {roi_feat_dim_for_heads} to {dct_coeff_output_dim_per_region} DCT coeffs/region.")

        # Fallback to Pixel output if NO spectral features are selected
        if not args.use_dft_features_appearance and not args.use_dct_features_appearance:
            self.logger.warning("Generator: No spectral features (DFT or DCT) selected for output. Defaulting to pixel output path.")
            final_conv_padding_spatial = 1 if getattr(args, 'gen_final_conv_kernel_spatial', 3) > 1 else 0
            self.final_conv_pixel = nn.Conv3d(current_channels, self.num_img_channels, 
                                              kernel_size=(self.gen_temporal_kernel_size, 
                                                           getattr(args, 'gen_final_conv_kernel_spatial', 3), 
                                                           getattr(args, 'gen_final_conv_kernel_spatial', 3)), 
                                              padding=(padding_temp, final_conv_padding_spatial, final_conv_padding_spatial))
            self.final_activation_pixel = nn.Tanh()
            self.logger.info(f"Generator Pixel Head (Fallback): FinalConv output channels {self.num_img_channels}")
        
        self.apply(init_weights_general)

    def _normalize_bboxes(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
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

    def forward(self, latent_code: torch.Tensor, gaad_bboxes_for_decode: Optional[torch.Tensor]
               ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Output: predicted_pixel_frames (Optional), predicted_dft_coeffs (Optional), predicted_dct_coeffs (Optional)
        # One of them MUST be non-None. If DFT/DCT are off, pixel frames are returned.
        # If DFT/DCT are on, they are returned, and pixel frames will be None from this function.
        
        B = latent_code.shape[0]
        device = latent_code.device
        dtype_in = latent_code.dtype

        x = self.fc_expand_latent(latent_code)
        x = x.view(B, self.gen_init_channels, self.num_predict_frames, self.gen_init_spatial_res, self.gen_init_spatial_res).to(dtype_in)

        sequence_condition = None
        if self.frame_gaad_embedder is not None:
            # Validate gaad_bboxes_for_decode shape
            if gaad_bboxes_for_decode is not None and \
               (gaad_bboxes_for_decode.shape[0] != B or \
                gaad_bboxes_for_decode.shape[1] != self.num_predict_frames or \
                gaad_bboxes_for_decode.shape[2] != self.num_regions):
                self.logger.warning(
                    f"Generator GAAD bbox shape mismatch. Expected (B={B}, N_pred={self.num_predict_frames}, NumReg={self.num_regions}, 4), "
                    f"got {gaad_bboxes_for_decode.shape if gaad_bboxes_for_decode is not None else 'None'}. Using zero condition for FiLM."
                )
                frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)
            elif gaad_bboxes_for_decode is not None:
                # Bboxes are (B, N_pred, NumReg, 4)
                norm_bboxes = self._normalize_bboxes(gaad_bboxes_for_decode.to(dtype_in), self.image_size[0], self.image_size[1])
                norm_bboxes_flat = norm_bboxes.view(B * self.num_predict_frames, -1) # Flatten NumReg and bbox coords
                frame_conditions_flat = self.frame_gaad_embedder(norm_bboxes_flat)
            else: # No bboxes provided but FiLM is active
                self.logger.debug("Generator GAAD Embedder active but no bboxes provided. Using zero condition for FiLM.")
                frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)
            
            if frame_conditions_flat is not None: # Should always be true due to fallbacks
                frame_conditions_reshaped = frame_conditions_flat.view(B, self.num_predict_frames, self.gaad_condition_dim)
                sequence_condition = torch.mean(frame_conditions_reshaped, dim=1).to(dtype_in) # Average FiLM condition over frames

        # Upsampling blocks
        for block_idx, block in enumerate(self.upsample_blocks):
            x = block['conv_transpose'](x)
            x = block['norm'](x)
            if 'film' in block and block['film'] is not None and sequence_condition is not None:
                x = block['film'](x, sequence_condition)
            x = block['activation'](x)
        # x is now dense feature volume: (B, C_feat, N_pred_frames, H_final_feat, W_final_feat)
        
        output_dft_coeffs: Optional[torch.Tensor] = None
        output_dct_coeffs: Optional[torch.Tensor] = None
        output_pixel_frames: Optional[torch.Tensor] = None

        if self.args.use_dft_features_appearance or self.args.use_dct_features_appearance:
            if gaad_bboxes_for_decode is None:
                self.logger.error("Generator spectral path: gaad_bboxes_for_decode is None. Cannot produce regional spectral features.")
                # Create dummy zero coeffs if needed
                if self.args.use_dft_features_appearance:
                    dft_w_coeffs_gen = self.gen_patch_w_spectral // 2 + 1
                    output_dft_coeffs = torch.zeros(B, self.num_predict_frames, self.num_regions, self.num_img_channels, 2, self.gen_patch_h_spectral, dft_w_coeffs_gen, device=device, dtype=dtype_in)
                if self.args.use_dct_features_appearance:
                    output_dct_coeffs = torch.zeros(B, self.num_predict_frames, self.num_regions, self.num_img_channels, self.gen_patch_h_spectral, self.gen_patch_w_spectral, device=device, dtype=dtype_in)
                return output_pixel_frames, output_dft_coeffs, output_dct_coeffs

            all_frames_regional_dft_coeffs_list = [] if self.args.use_dft_features_appearance and self.to_dft_coeffs_mlp else None
            all_frames_regional_dct_coeffs_list = [] if self.args.use_dct_features_appearance and self.to_dct_coeffs_mlp else None
            
            H_final_feat, W_final_feat = x.shape[-2], x.shape[-1]
            # RoIAlign spatial_scale maps bbox coords from input image (H_img, W_img) to feature map (H_final_feat, W_final_feat)
            roi_spatial_scale = H_final_feat / self.image_size[0] # Assuming H scale is representative

            for f_idx in range(self.num_predict_frames):
                frame_feature_map = x[:, :, f_idx, :, :] # (B, C_feat, H_final_feat, W_final_feat)
                frame_bboxes = gaad_bboxes_for_decode[:, f_idx, :, :] # (B, NumRegions, 4)
                
                rois_for_frame_list = []
                for b_s_idx in range(B): # iterate through samples in batch
                    batch_indices = torch.full((self.num_regions, 1), float(b_s_idx), device=device, dtype=dtype_in)
                    rois_for_frame_list.append(torch.cat([batch_indices, frame_bboxes[b_s_idx]], dim=1))
                all_rois_this_frame = torch.cat(rois_for_frame_list, dim=0) # (B*NumRegions, 5)
                
                regional_feats_from_roi = roi_align(
                    frame_feature_map, all_rois_this_frame,
                    output_size=self.gen_roi_align_output_spatial, # (H_spectral, W_spectral)
                    spatial_scale=roi_spatial_scale, aligned=True
                ) # (B*NumRegions, C_feat, H_spectral, W_spectral)
                regional_feats_flat = regional_feats_from_roi.reshape(B * self.num_regions, -1) # Flatten for MLPs

                if self.args.use_dft_features_appearance and self.to_dft_coeffs_mlp is not None and all_frames_regional_dft_coeffs_list is not None:
                    dft_coeffs_flat_for_frame = self.to_dft_coeffs_mlp(regional_feats_flat)
                    dft_w_coeffs_one_sided_gen = self.gen_patch_w_spectral // 2 + 1
                    frame_dft_structured = dft_coeffs_flat_for_frame.view(
                        B, self.num_regions, self.num_img_channels, 2, # Real/Imag
                        self.gen_patch_h_spectral, dft_w_coeffs_one_sided_gen
                    )
                    all_frames_regional_dft_coeffs_list.append(frame_dft_structured)
                
                if self.args.use_dct_features_appearance and self.to_dct_coeffs_mlp is not None and all_frames_regional_dct_coeffs_list is not None:
                    dct_coeffs_flat_for_frame = self.to_dct_coeffs_mlp(regional_feats_flat)
                    frame_dct_structured = dct_coeffs_flat_for_frame.view(
                        B, self.num_regions, self.num_img_channels,
                        self.gen_patch_h_spectral, self.gen_patch_w_spectral
                    )
                    all_frames_regional_dct_coeffs_list.append(frame_dct_structured)
            
            if all_frames_regional_dft_coeffs_list:
                output_dft_coeffs = torch.stack(all_frames_regional_dft_coeffs_list, dim=1).to(dtype_in)
            if all_frames_regional_dct_coeffs_list:
                output_dct_coeffs = torch.stack(all_frames_regional_dct_coeffs_list, dim=1).to(dtype_in)
        
        else: # Fallback to Pixel Output Path (No spectral features selected by args)
            if hasattr(self, 'final_conv_pixel') and hasattr(self, 'final_activation_pixel'):
                x = self.final_conv_pixel(x)
                generated_frames_sequence_pixels = self.final_activation_pixel(x)
                # Permute to (B, N_pred, C, H, W)
                output_pixel_frames = generated_frames_sequence_pixels.permute(0, 2, 1, 3, 4).to(dtype_in)

                # Adaptive pooling if spatial dimensions don't match target image_size
                final_h_actual, final_w_actual = output_pixel_frames.shape[-2:]
                if final_h_actual != self.image_size[0] or final_w_actual != self.image_size[1] or self.needs_final_adaptive_pool:
                    self.logger.debug_once(f"Generator pixel output {final_h_actual}x{final_w_actual} vs target {self.image_size}. Applying adaptive pool.")
                    # AdaptiveAvgPool3d expects (B, C, D, H, W)
                    temp_permuted_for_pool = output_pixel_frames.permute(0, 2, 1, 3, 4) # (B, C, N_pred, H_curr, W_curr)
                    pooled = F.adaptive_avg_pool3d(temp_permuted_for_pool, (self.num_predict_frames, self.image_size[0], self.image_size[1]))
                    output_pixel_frames = pooled.permute(0, 2, 1, 3, 4) # Back to (B, N_pred, C, H_target, W_target)
            else:
                self.logger.error("Generator: Fallback pixel path missing final_conv_pixel or final_activation_pixel.")
                # This should not happen if constructor logic is correct.
                # Return None for pixel frames and ensure spectral are also None or zeros.
                output_pixel_frames = None 

        return output_pixel_frames, output_dft_coeffs, output_dct_coeffs

# --- Helper Self-Attention Module for Discriminators (from audio script) ---
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, k_reduction_factor=8, use_spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // k_reduction_factor)

        conv_fn = functools.partial(nn.Conv2d, kernel_size=1, padding=0, bias=False)
        self.query_conv = conv_fn(self.in_channels, self.inter_channels)
        self.key_conv = conv_fn(self.in_channels, self.inter_channels)
        self.value_conv = conv_fn(self.in_channels, self.in_channels)
        self.out_conv = conv_fn(self.in_channels, self.in_channels) # Output back to in_channels

        if use_spectral_norm:
            self.query_conv = spectral_norm(self.query_conv)
            self.key_conv = spectral_norm(self.key_conv)
            self.value_conv = spectral_norm(self.value_conv)
            self.out_conv = spectral_norm(self.out_conv)

        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scale factor
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        proj_query = self.query_conv(x).view(B, self.inter_channels, -1).permute(0, 2, 1) # (B, HW, inter_channels)
        proj_key = self.key_conv(x).view(B, self.inter_channels, -1) # (B, inter_channels, HW)
        energy = torch.bmm(proj_query, proj_key) # (B, HW, HW)
        attention = self.softmax(energy) # (B, HW, HW)
        
        proj_value = self.value_conv(x).view(B, C, -1) # (B, C, HW)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # (B, C, HW)
        out = out.view(B, C, H, W) # Reshape back
        
        out = self.out_conv(out) # Final 1x1 conv
        return self.gamma * out + x # Add residual connection scaled by gamma



# --- Helper for Convolution Output Dimension Calculation ---
def _calculate_conv_output_dim(input_dim: int, kernel_size: int, padding: int, stride: int, dilation: int = 1) -> int:
    if input_dim <= 0: return 0
    return (input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

# --- 3D Self-Attention Module ---
class SelfAttention3D(nn.Module):
    """ Self attention Layer for 3D Tensors (B, C, D, H, W) """
    def __init__(self, in_dim: int, k_reduction_factor: int = 8, use_spectral_norm: bool = False):
        super().__init__()
        self.chanel_in = in_dim
        self.inter_channels = max(1, in_dim // k_reduction_factor)

        conv_fn_3d = functools.partial(nn.Conv3d, kernel_size=1, padding=0, bias=False)

        self.query_conv = conv_fn_3d(self.chanel_in, self.inter_channels)
        self.key_conv = conv_fn_3d(self.chanel_in, self.inter_channels)
        self.value_conv = conv_fn_3d(self.chanel_in, self.chanel_in) # Value projects to full channels
        self.out_conv = conv_fn_3d(self.chanel_in, self.chanel_in)   # Final 1x1x1 conv

        if use_spectral_norm:
            self.query_conv = spectral_norm(self.query_conv)
            self.key_conv = spectral_norm(self.key_conv)
            self.value_conv = spectral_norm(self.value_conv)
            self.out_conv = spectral_norm(self.out_conv)

        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scale factor for residual connection
        self.softmax  = nn.Softmax(dim=-1) # Apply softmax on the last dimension of energy

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
            inputs :
                x : input feature maps (B, C, D, H, W)
            returns :
                out : self attention value + input feature
                attention: (B, N, N) (N = D*H*W)
        """
        B, C, D_in, H_in, W_in = x.size()
        N_spatial_temporal = D_in * H_in * W_in

        proj_query = self.query_conv(x).view(B, self.inter_channels, N_spatial_temporal).permute(0, 2, 1) # B, N, C_inter
        proj_key = self.key_conv(x).view(B, self.inter_channels, N_spatial_temporal) # B, C_inter, N
        
        energy = torch.bmm(proj_query, proj_key) # B, N, N (transpose check: (N, C_inter) * (C_inter, N) = (N,N))
        attention = self.softmax(energy) # B, N, N
        
        proj_value = self.value_conv(x).view(B, C, N_spatial_temporal) # B, C, N
        
        out_att = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, N ( (C,N) * (N,N) = (C,N) )
        out_att = out_att.view(B, C, D_in, H_in, W_in) # Reshape back
        
        out_final_conv = self.out_conv(out_att)
        out = self.gamma * out_final_conv + x # Add residual scaled by gamma
        return out #, attention # Usually only return out, attention map is for debugging

class _SingleScaleVideoDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, disc_config: Dict, scale_index: int = 0):
        super().__init__()
        self.args = args
        self.scale_index = scale_index
        self.apply_spectral_norm = disc_config.get(f"video_d_scale{scale_index}_apply_sn",
                                                   disc_config.get("apply_spectral_norm",
                                                                   getattr(args, 'disc_apply_spectral_norm', True)))
        self.logger = logging.getLogger(f"WuBuGAADHybridGenV03.SingleVideoD.Scale{scale_index}.{id(self)}")

        img_h_initial = args.image_h
        img_w_initial = args.image_w
        
        img_h_eff = img_h_initial // (2**scale_index)
        img_w_eff = img_w_initial // (2**scale_index)
        num_frames_eff = disc_config.get(f"video_d_scale{scale_index}_num_frames",
                                        video_config.get("num_predict_frames", 1))

        base_ch = disc_config.get("base_disc_channels", getattr(args, 'disc_base_disc_channels', 64))
        max_ch = disc_config.get("max_disc_channels", getattr(args, 'disc_max_disc_channels', 512))
        
        target_final_dim_config = disc_config.get("target_video_disc_final_feature_dim", getattr(args, 'disc_target_final_feature_dim', [4,4]))
        if isinstance(target_final_dim_config, int): target_final_dim_h = target_final_dim_w = target_final_dim_config
        elif isinstance(target_final_dim_config, (list, tuple)) and len(target_final_dim_config) == 2: target_final_dim_h, target_final_dim_w = target_final_dim_config
        else: target_final_dim_h = target_final_dim_w = 4

        max_downs_limit = disc_config.get("max_video_disc_downsample_layers", getattr(args, 'max_video_disc_downsample_layers', 5))
        
        parent_prefix = disc_config.get("parent_discriminator_arg_prefix", "video_d")
        self.use_attention_in_video_scale = getattr(args, f"{parent_prefix}_scale{scale_index}_use_attention",
                                            getattr(args, f"{parent_prefix}_use_attention", False)) # Read general arg if scale-specific is missing
        self.attention_after_layer_idx = getattr(args, f"{parent_prefix}_scale{scale_index}_attention_idx",
                                            getattr(args, f"{parent_prefix}_attention_idx", 2)) # e.g. after 2nd conv block (0-indexed)
        self.attention_k_reduction_factor = getattr(args, f"{parent_prefix}_attention_k_reduction", 8)


        cnn_layers_list = []
        in_c = video_config.get("num_channels", 3)
        curr_h, curr_w, curr_d = img_h_eff, img_w_eff, num_frames_eff

        if curr_h <= 0 or curr_w <= 0 or curr_d <= 0:
            self.logger.warning(f"SingleVideoD Scale {scale_index}: Effective input dims ({curr_d}x{curr_h}x{curr_w}) non-positive. Using Identity layers.")
            self.feature_extractor = nn.Identity()
            self.final_conv_in_channels = in_c
            self.final_conv = nn.Identity()
            return

        num_actual_conv_layers = 0
        K_SPATIAL, P_SPATIAL, DIL_SPATIAL = 4, 1, 1 # Kernel, Padding, Dilation for spatial
        
        temporal_kernel_size_config = disc_config.get("temporal_kernel_size", getattr(args, 'disc_temporal_kernel_size', 3))
        
        for i_loop_idx in range(max_downs_limit):
            if num_actual_conv_layers >= max_downs_limit: break
            # Stop if target spatial dims are met AND temporal is 1 (or non-stridable)
            spatial_target_met = (curr_h <= target_final_dim_h and curr_w <= target_final_dim_w)
            temporal_target_met = (curr_d == 1) # or if further temporal striding isn't planned/possible
            if spatial_target_met and temporal_target_met: break
            if curr_h <= 1 and curr_w <= 1 and curr_d == 1: break # Already 1x1x1

            out_c = min(base_ch * (2**num_actual_conv_layers), max_ch)
            
            spatial_stride_val = 2 if (curr_h > target_final_dim_h or curr_w > target_final_dim_w) and (curr_h > 1 or curr_w > 1) else 1
            temporal_stride_val = 2 if curr_d > 1 and num_actual_conv_layers < disc_config.get("num_temporal_stride_layers_disc", 1) else 1
            
            eff_temporal_kernel = min(temporal_kernel_size_config, curr_d) if curr_d > 1 else 1
            eff_temporal_padding = eff_temporal_kernel // 2 if eff_temporal_kernel > 1 else 0
            if eff_temporal_kernel > curr_d and curr_d > 0: eff_temporal_kernel = curr_d; eff_temporal_padding = 0

            conv_l = nn.Conv3d(in_c, out_c,
                               kernel_size=(eff_temporal_kernel, K_SPATIAL, K_SPATIAL),
                               stride=(temporal_stride_val, spatial_stride_val, spatial_stride_val),
                               padding=(eff_temporal_padding, P_SPATIAL, P_SPATIAL),
                               bias=False)
            
            if self.apply_spectral_norm: cnn_layers_list.append(spectral_norm(conv_l))
            else: cnn_layers_list.append(conv_l)
            
            cnn_layers_list.append(nn.InstanceNorm3d(out_c, affine=True))
            cnn_layers_list.append(nn.LeakyReLU(0.2, inplace=True))
            
            in_c = out_c # Update in_c for the next layer (or attention/final_conv)
            
            # Update current dimensions using the _calculate_conv_output_dim helper
            next_d = _calculate_conv_output_dim(curr_d, eff_temporal_kernel, eff_temporal_padding, temporal_stride_val)
            next_h = _calculate_conv_output_dim(curr_h, K_SPATIAL, P_SPATIAL, spatial_stride_val, DIL_SPATIAL)
            next_w = _calculate_conv_output_dim(curr_w, K_SPATIAL, P_SPATIAL, spatial_stride_val, DIL_SPATIAL)

            if next_d == 0 or next_h == 0 or next_w == 0:
                self.logger.warning(f"SingleVideoD Scale {scale_index} Layer {num_actual_conv_layers}: Calculated next dims zero ({next_d}x{next_h}x{next_w}) from ({curr_d}x{curr_h}x{curr_w}). Stopping downsampling.")
                break
            curr_d, curr_h, curr_w = next_d, next_h, next_w
            
            # Add Self-Attention block if configured
            if self.use_attention_in_video_scale and num_actual_conv_layers == self.attention_after_layer_idx and in_c > 0:
                self.logger.info(f"  SingleVideoD Scale {scale_index} Layer {num_actual_conv_layers}: Adding 3D Self-Attention. InChannels: {in_c}, K_reduction: {self.attention_k_reduction_factor}")
                # `in_c` here is the `out_c` of the preceding conv layer
                attention_block = SelfAttention3D(in_dim=in_c, k_reduction_factor=self.attention_k_reduction_factor, use_spectral_norm=self.apply_spectral_norm)
                cnn_layers_list.append(attention_block)
                # Note: Attention block does not change channel count or spatial/temporal dimensions.

            num_actual_conv_layers += 1 # Count actual conv blocks added (attention is separate)


        self.feature_extractor = nn.Sequential(*cnn_layers_list) if cnn_layers_list else nn.Identity()
        self.final_conv_in_channels = in_c # Channels after last conv/attention
        
        final_kernel_d_eff = curr_d if curr_d > 0 else 1
        final_kernel_h_eff = curr_h if curr_h > 0 else 1
        final_kernel_w_eff = curr_w if curr_w > 0 else 1
        
        self.final_conv = nn.Conv3d(self.final_conv_in_channels, 1,
                                    kernel_size=(final_kernel_d_eff, final_kernel_h_eff, final_kernel_w_eff),
                                    stride=1, padding=0, bias=True)
        if self.apply_spectral_norm: self.final_conv = spectral_norm(self.final_conv)

        self.logger.info(f"SingleVideoD Scale {scale_index} constructed. Input Eff: {num_frames_eff}x{img_h_eff}x{img_w_eff}. Actual Conv Layers: {num_actual_conv_layers}. Final FeatMap Dims (before final_conv): {curr_d}x{curr_h}x{curr_w}. Final Conv InChannels: {self.final_conv_in_channels}.")

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(self.feature_extractor, nn.Identity) and isinstance(self.final_conv, nn.Identity):
            self.logger.debug_once(f"SingleVideoD Scale {self.scale_index}: Forward pass through Identity layers.")
            dummy_logits = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            # Return input x as "features" but ensure C matches what final_conv_in_channels would have been if active
            dummy_features_channels = self.final_conv_in_channels if self.final_conv_in_channels > 0 else x.size(1)
            dummy_features_shape = (x.size(0), dummy_features_channels) + x.shape[2:]
            dummy_features = torch.zeros(dummy_features_shape, device=x.device, dtype=x.dtype) if return_features else torch.empty(0, device=x.device, dtype=x.dtype)
            if return_features and dummy_features_channels != x.size(1) : # If input channels don't match expected out_channels
                self.logger.warning_once(f"SingleVideoD Scale {self.scale_index} (Identity Path): Mismatch in feature channels. Input C={x.size(1)}, Expected D.final_conv_in_channels={dummy_features_channels}. Returning features based on expected.")
            elif return_features: dummy_features = x

            return (dummy_logits, dummy_features) if return_features else dummy_logits

        features = self.feature_extractor(x)
        patch_logits_map = self.final_conv(features)
        logits = patch_logits_map.squeeze()
        if logits.ndim == 0: logits = logits.unsqueeze(0)

        return (logits, features) if return_features else logits

# =====================================================================
# Architectural Components (v0.3 - VAE-GAN Refactor + DFT + DCT)
# =====================================================================

class RegionalPatchExtractor(nn.Module): # Unchanged from v0.2
    def __init__(self, patch_output_size: Optional[Tuple[int, int]] = None, feature_extractor: Optional[nn.Module] = None, feature_map_spatial_scale: float = 1.0, roi_align_output_size: Optional[Tuple[int, int]] = None, use_roi_align: bool = False):
        super().__init__(); self.patch_output_size = patch_output_size; self.feature_extractor = feature_extractor; self.feature_map_spatial_scale = feature_map_spatial_scale; self.roi_align_output_size = roi_align_output_size; self.use_roi_align = use_roi_align; current_logger=logging.getLogger("WuBuGAADHybridGenV03.PatchExtract"); self.resize_transform=None
        if self.use_roi_align:
            if self.feature_extractor is None or self.roi_align_output_size is None: raise ValueError("feature_extractor and roi_align_output_size needed for use_roi_align=True")
            current_logger.info(f"Using RoIAlign. Output: {roi_align_output_size}, FeatMapScale: {feature_map_spatial_scale:.2f}")
        else:
            if self.patch_output_size is None: raise ValueError("patch_output_size needed for use_roi_align=False")
            current_logger.info(f"Using Pixel Patches. Resizing to: {patch_output_size}")
            self.resize_transform = T.Resize(self.patch_output_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    def forward(self, images: torch.Tensor, bboxes_batch: torch.Tensor) -> torch.Tensor:
        B_img_orig, NumCh_img, H_img_orig, W_img_orig = images.shape
        B_bboxes, NumRegions_bboxes, _ = bboxes_batch.shape
        if B_img_orig != B_bboxes: raise ValueError(f"Batch size mismatch images ({B_img_orig}) vs bboxes ({B_bboxes})")
        device = images.device; original_images_dtype = images.dtype; compute_dtype = torch.float32 if images.dtype == torch.uint8 else images.dtype; images_for_processing = images.to(compute_dtype)
        if self.use_roi_align and self.feature_extractor is not None and self.roi_align_output_size is not None:
            feature_maps = self.feature_extractor(images_for_processing); h_feat, w_feat = feature_maps.shape[2:]; max_w_feat_scalar=float(w_feat); max_h_feat_scalar=float(h_feat);
            all_rois_list = []
            for i in range(B_img_orig):
                current_bboxes_scaled = bboxes_batch[i].to(torch.float32) * self.feature_map_spatial_scale
                current_bboxes_scaled[:,0]=torch.clamp(current_bboxes_scaled[:,0],min=0.0,max=max_w_feat_scalar-EPS); current_bboxes_scaled[:,1]=torch.clamp(current_bboxes_scaled[:,1],min=0.0,max=max_h_feat_scalar-EPS); min_for_x2=current_bboxes_scaled[:,0]; current_bboxes_scaled[:,2]=torch.clamp(current_bboxes_scaled[:,2],max=max_w_feat_scalar); current_bboxes_scaled[:,2]=torch.maximum(current_bboxes_scaled[:,2],min_for_x2); min_for_y2=current_bboxes_scaled[:,1]; current_bboxes_scaled[:,3]=torch.clamp(current_bboxes_scaled[:,3],max=max_h_feat_scalar); current_bboxes_scaled[:,3]=torch.maximum(current_bboxes_scaled[:,3],min_for_y2);
                batch_indices_for_this_image_in_flat_batch = torch.full((NumRegions_bboxes, 1), float(i), device=device, dtype=current_bboxes_scaled.dtype)
                all_rois_list.append(torch.cat([batch_indices_for_this_image_in_flat_batch, current_bboxes_scaled], dim=1))
            all_rois_for_align = torch.cat(all_rois_list, dim=0)
            try: aligned_features_flat = roi_align(feature_maps, all_rois_for_align, output_size=self.roi_align_output_size, spatial_scale=1.0, aligned=True)
            except Exception as e_roi: logging.getLogger("WuBuGAADHybridGenV03.PatchExtract").error(f"RoIAlign failed: {e_roi}. FeatMap:{feature_maps.shape}, RoIs:{all_rois_for_align.shape}, Output:{self.roi_align_output_size}"); raise e_roi
            C_feat=feature_maps.shape[1]; H_roi, W_roi = self.roi_align_output_size
            aligned_features_reshaped = aligned_features_flat.view(B_img_orig, NumRegions_bboxes, C_feat, H_roi, W_roi)
            return aligned_features_reshaped.to(original_images_dtype)
        else:
            all_patches_collected = []; patch_h_out, patch_w_out = self.patch_output_size # type: ignore
            for i in range(B_img_orig):
                single_image_from_flat_batch = images_for_processing[i]; single_image_bboxes = bboxes_batch[i]
                current_image_patches = []
                for r in range(NumRegions_bboxes):
                    x1,y1,x2,y2 = single_image_bboxes[r].round().int().tolist(); x1_c,y1_c=max(0,x1),max(0,y1); x2_c,y2_c=min(W_img_orig,x2),min(H_img_orig,y2)
                    if x1_c >= x2_c or y1_c >= y2_c: patch = torch.zeros((images.shape[1], patch_h_out, patch_w_out), device=device, dtype=compute_dtype)
                    else: patch = single_image_from_flat_batch[:, y1_c:y2_c, x1_c:x2_c]; patch = self.resize_transform(patch) if self.resize_transform else patch
                    current_image_patches.append(patch)
                all_patches_collected.append(torch.stack(current_image_patches))
            final_patches_tensor = torch.stack(all_patches_collected)
            return final_patches_tensor.to(original_images_dtype)

class PatchEmbed(nn.Module): # Unchanged from v0.2
    def __init__(self, patch_feature_dim: int, embed_dim: int):
        super().__init__(); self.proj = nn.Linear(patch_feature_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3: B_frames, N_reg, D_feat = x.shape; x = x.view(B_frames * N_reg, D_feat); out = self.proj(x); return out.view(B_frames, N_reg, -1)
        elif x.dim() == 2: return self.proj(x)
        else: raise ValueError(f"PatchEmbed input x has unsupported dimension: {x.dim()}")


class RegionalVAEEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, wubu_s_config: Dict, latent_dim: int):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_config = gaad_config
        # self.wubu_s_config = wubu_s_config # Not stored if only used for sub-module
        self.latent_dim = latent_dim
        self.image_size = args.image_h_w_tuple # Ensure this is set in parse_arguments
        self.num_appearance_regions = gaad_config['num_regions']
        self.decomposition_type = gaad_config['decomposition_type']
        self.gaad_min_size_px = gaad_config.get('min_size_px', 5)
        current_logger=logging.getLogger("WuBuGAADHybridGenV03.EncoderVAE")
        self.feature_extractor: Optional[nn.Module] = None # For RoIAlign path

        self.channels_for_target_spectral = self.video_config['num_channels']
        
        # Patch extractors for DFT and DCT (pixel-based for target, potentially feature-based for WuBu input)
        self.enc_patch_h_spectral = args.spectral_patch_size_h
        self.enc_patch_w_spectral = args.spectral_patch_size_w
        spectral_patch_output_size = (args.spectral_patch_size_h, args.spectral_patch_size_w)
        
        # Patch extractor for TARGET DFT and DCT features (always from pixels)
        self.target_spectral_patch_extractor = RegionalPatchExtractor(
            patch_output_size=spectral_patch_output_size, use_roi_align=False
        )
        current_logger.info(f"Encoder: Target Spectral Extractor (Pixel Patches -> Resize {spectral_patch_output_size}).")

        patch_channels_for_wubus_input: int
        if args.encoder_use_roi_align:
            self.feature_extractor = nn.Sequential( # Shallow CNN before RoIAlign
                nn.Conv2d(self.video_config['num_channels'], args.encoder_shallow_cnn_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, args.encoder_shallow_cnn_channels), nn.GELU()
            )
            patch_channels_for_wubus_input = args.encoder_shallow_cnn_channels
            self.wubus_input_patch_extractor = RegionalPatchExtractor(
                feature_extractor=self.feature_extractor,
                roi_align_output_size=spectral_patch_output_size, # RoIAlign output matches spectral patch size
                use_roi_align=True
            )
            current_logger.info(f"Encoder (WuBu-S Input): RoIAlign ON. ShallowCNN (Ch:{args.encoder_shallow_cnn_channels}) -> RoIAlign (Size:{spectral_patch_output_size}) -> Spectral Transforms.")
        else:
            patch_channels_for_wubus_input = self.video_config['num_channels']
            self.wubus_input_patch_extractor = RegionalPatchExtractor(
                patch_output_size=spectral_patch_output_size, use_roi_align=False
            )
            current_logger.info(f"Encoder (WuBu-S Input): RoIAlign OFF. Pixel Patches -> Resize (Size:{spectral_patch_output_size}) -> Spectral Transforms.")

        # Calculate feature dimension after spectral transforms
        dft_w_coeffs_one_sided = self.enc_patch_w_spectral // 2 + 1
        single_dft_feature_dim = patch_channels_for_wubus_input * 2 * self.enc_patch_h_spectral * dft_w_coeffs_one_sided
        single_dct_feature_dim = patch_channels_for_wubus_input * self.enc_patch_h_spectral * self.enc_patch_w_spectral
        
        # Combined feature dimension for PatchEmbed
        patch_feature_dim_for_embed = 0
        if args.use_dft_features_appearance: patch_feature_dim_for_embed += single_dft_feature_dim
        if args.use_dct_features_appearance: patch_feature_dim_for_embed += single_dct_feature_dim
        
        if patch_feature_dim_for_embed == 0:
            current_logger.error("Encoder: No spectral features (DFT or DCT) selected for appearance. This will likely fail.")
            # Fallback to pixel features if NO spectral features selected (original v0.1 path)
            if args.encoder_use_roi_align:
                 patch_feature_dim_for_embed = patch_channels_for_wubus_input * spectral_patch_output_size[0] * spectral_patch_output_size[1]
            else: # Pixel patches without RoIAlign
                 patch_feature_dim_for_embed = self.video_config['num_channels'] * spectral_patch_output_size[0] * spectral_patch_output_size[1]
            current_logger.warning(f"Encoder: Falling back to pixel-based features (dim: {patch_feature_dim_for_embed}) as no spectral features selected.")

        current_logger.info(f"PatchEmbed input dim: {patch_feature_dim_for_embed}")
        self.patch_embed = PatchEmbed(patch_feature_dim_for_embed, args.encoder_initial_tangent_dim)

        self.wubu_s = FullyHyperbolicWuBuNestingModel(input_tangent_dim=args.encoder_initial_tangent_dim, output_tangent_dim=video_config['wubu_s_output_dim'], config=wubu_s_config)
        
        self.wubu_t_input_dim = video_config['wubu_s_output_dim']
        self.wubu_m_output_dim = video_config.get('wubu_m_output_dim', 0) # Get current, possibly 0 if motion disabled
        if args.use_wubu_motion_branch and self.wubu_m_output_dim > 0:
            self.wubu_t_input_dim += self.wubu_m_output_dim
            current_logger.info(f"VAE Enc: Including WuBu-M features (dim {self.wubu_m_output_dim}) for WuBu-T input.")
        elif args.use_wubu_motion_branch:
            current_logger.warning("VAE Enc: Motion branch enabled but wubu_m_output_dim is 0. Not included in WuBu-T.")

        self.wubu_t_config = _configure_wubu_stack(args, "wubu_t")
        self.wubu_t: Optional[FullyHyperbolicWuBuNestingModel] = None
        fc_input_dim_for_latent: int
        if self.wubu_t_config and self.wubu_t_config['num_levels'] > 0 and self.wubu_t_input_dim > 0:
             wubu_t_output_dim_config = self.wubu_t_config['hyperbolic_dims'][-1] if self.wubu_t_config['hyperbolic_dims'] else 0
             if wubu_t_output_dim_config == 0: wubu_t_output_dim_config = self.wubu_t_input_dim
             self.wubu_t = FullyHyperbolicWuBuNestingModel(input_tangent_dim=self.wubu_t_input_dim, output_tangent_dim=wubu_t_output_dim_config, config=self.wubu_t_config)
             fc_input_dim_for_latent = wubu_t_output_dim_config
             current_logger.info(f"VAE Enc WuBu-T Enabled: InputDim {self.wubu_t_input_dim}, OutputDim {fc_input_dim_for_latent}")
        else:
             current_logger.warning("VAE Enc WuBu-T disabled. Latent space from direct projection of aggregated features.")
             fc_input_dim_for_latent = self.wubu_t_input_dim
        
        if fc_input_dim_for_latent <=0:
            current_logger.error(f"FC input dim for latent space is {fc_input_dim_for_latent}. This is invalid. Check WuBu configurations or input dimensions.")
            # Fallback to ensure FC layers can be created, though model might not train.
            fc_input_dim_for_latent = self.latent_dim 
        
        self.fc_mu = nn.Linear(fc_input_dim_for_latent, self.latent_dim)
        self.fc_logvar = nn.Linear(fc_input_dim_for_latent, self.latent_dim)

        self.apply(init_weights_general)

    def forward(self, frames_pixels: torch.Tensor, motion_features: Optional[torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Output: mu, logvar, gaad_bboxes_all_frames, target_dft_features, target_dct_features
        B, N_frames_total_sample, C_img, H_img, W_img = frames_pixels.shape
        device = frames_pixels.device
        dtype_model = next(self.parameters()).dtype

        frames_pixels_flat = frames_pixels.reshape(B * N_frames_total_sample, C_img, H_img, W_img)
        # GAAD Bbox Generation (copied from v0.2)
        gaad_bboxes_list = []
        for b_idx in range(B):
            frame_bboxes_for_sequence = []
            for f_idx in range(N_frames_total_sample):
                frame_dims = (W_img, H_img); max_w_scalar=float(W_img); max_h_scalar=float(H_img)
                if self.decomposition_type == "hybrid":
                    num_subdivide=self.num_appearance_regions//2; num_spiral=self.num_appearance_regions-num_subdivide; bboxes_for_item=[]
                    if num_subdivide > 0: bboxes_for_item.append(golden_subdivide_rect_fixed_n(frame_dims,num_subdivide,device,dtype_model,self.gaad_min_size_px))
                    if num_spiral > 0:
                         spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, num_spiral, device, dtype_model); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype_model); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); bboxes_for_item.append(spiral_bboxes_current)
                    single_frame_bboxes = torch.cat(bboxes_for_item, dim=0) if bboxes_for_item else torch.tensor([[0,0,max_w_scalar,max_h_scalar]]*self.num_appearance_regions, dtype=dtype_model, device=device)
                elif self.decomposition_type == "spiral":
                     spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n(frame_dims, self.num_appearance_regions, device, dtype_model); patch_base_size = min(frame_dims); spiral_bboxes_current = torch.zeros(self.num_appearance_regions, 4, device=device, dtype=dtype_model); patch_hs = float(patch_base_size) * spiral_scales[:,0] / 2.0; patch_ws = patch_hs; val_x1=spiral_centers[:,0]-patch_ws; val_y1=spiral_centers[:,1]-patch_hs; val_x2=spiral_centers[:,0]+patch_ws; val_y2=spiral_centers[:,1]+patch_hs; spiral_bboxes_current[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_current[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS); min_for_x2=spiral_bboxes_current[:,0]+EPS; spiral_bboxes_current[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_current[:,2]=torch.maximum(spiral_bboxes_current[:,2],min_for_x2); min_for_y2=spiral_bboxes_current[:,1]+EPS; spiral_bboxes_current[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_current[:,3]=torch.maximum(spiral_bboxes_current[:,3],min_for_y2); single_frame_bboxes = spiral_bboxes_current
                else: single_frame_bboxes = golden_subdivide_rect_fixed_n(frame_dims,self.num_appearance_regions,device,dtype_model,self.gaad_min_size_px)
                if single_frame_bboxes.shape[0] < self.num_appearance_regions: num_to_pad=self.num_appearance_regions-single_frame_bboxes.shape[0]; padding_box=single_frame_bboxes[-1:].clone() if single_frame_bboxes.shape[0]>0 else torch.tensor([[0,0,max_w_scalar,max_h_scalar]],dtype=dtype_model,device=device); padding=padding_box.repeat(num_to_pad,1); single_frame_bboxes=torch.cat([single_frame_bboxes, padding], dim=0)
                elif single_frame_bboxes.shape[0] > self.num_appearance_regions: single_frame_bboxes=single_frame_bboxes[:self.num_appearance_regions]
                frame_bboxes_for_sequence.append(single_frame_bboxes)
            gaad_bboxes_list.append(torch.stack(frame_bboxes_for_sequence))
        gaad_bboxes_full_batch_sequences = torch.stack(gaad_bboxes_list)
        gaad_bboxes_flat_for_patch_extractor = gaad_bboxes_full_batch_sequences.reshape(B * N_frames_total_sample, self.num_appearance_regions, 4)

        # --- Extract TARGET spectral features (from original pixels) ---
        target_pixel_patches = self.target_spectral_patch_extractor(frames_pixels_flat, gaad_bboxes_flat_for_patch_extractor)
        # target_pixel_patches: (B*N_frames, NumReg, C_target, H_spectral, W_spectral)
        B_flat_target, NReg_target, C_target_spectral, H_spec_target, W_spec_target = target_pixel_patches.shape
        
        target_dft_features_flat: Optional[torch.Tensor] = None
        target_dct_features_flat: Optional[torch.Tensor] = None

        if self.args.use_dft_features_appearance:
            target_dft_features_flat = SpectralTransformUtils.compute_2d_dft_features(
                target_pixel_patches.reshape(B_flat_target * NReg_target, C_target_spectral, H_spec_target, W_spec_target),
                norm_scale=self.args.dft_norm_scale_video,
                fft_norm_type=self.args.dft_fft_norm
            ) # ( (B*N_frames)*NumReg, D_dft )
        
        if self.args.use_dct_features_appearance:
            target_dct_features_flat = SpectralTransformUtils.compute_2d_dct_features(
                target_pixel_patches.reshape(B_flat_target * NReg_target, C_target_spectral, H_spec_target, W_spec_target),
                norm_type=self.args.dct_norm_type,
                norm_global_scale=self.args.dct_norm_global_scale,
                norm_tanh_scale=self.args.dct_norm_tanh_scale
            ) # ( (B*N_frames)*NumReg, D_dct )

        # --- Extract INPUT spectral features for WuBu-S (potentially from RoIAlign) ---
        patches_for_wubus_input_path = self.wubus_input_patch_extractor(frames_pixels_flat, gaad_bboxes_flat_for_patch_extractor)
        B_flat_wubus, NReg_wubus, C_wubus_in, H_spec_wubus, W_spec_wubus = patches_for_wubus_input_path.shape
        
        wubus_dft_input_flat: Optional[torch.Tensor] = None
        wubus_dct_input_flat: Optional[torch.Tensor] = None
        
        if self.args.use_dft_features_appearance:
            wubus_dft_input_flat = SpectralTransformUtils.compute_2d_dft_features(
                patches_for_wubus_input_path.reshape(B_flat_wubus * NReg_wubus, C_wubus_in, H_spec_wubus, W_spec_wubus),
                norm_scale=self.args.dft_norm_scale_video,
                fft_norm_type=self.args.dft_fft_norm
            )
        if self.args.use_dct_features_appearance:
            wubus_dct_input_flat = SpectralTransformUtils.compute_2d_dct_features(
                patches_for_wubus_input_path.reshape(B_flat_wubus * NReg_wubus, C_wubus_in, H_spec_wubus, W_spec_wubus),
                norm_type=self.args.dct_norm_type,
                norm_global_scale=self.args.dct_norm_global_scale,
                norm_tanh_scale=self.args.dct_norm_tanh_scale
            )

        # Concatenate DFT and DCT features for PatchEmbed if both are used
        combined_spectral_features_for_embed_list = []
        if wubus_dft_input_flat is not None: combined_spectral_features_for_embed_list.append(wubus_dft_input_flat)
        if wubus_dct_input_flat is not None: combined_spectral_features_for_embed_list.append(wubus_dct_input_flat)
        
        if not combined_spectral_features_for_embed_list: # Fallback if no spectral features
            self.logger.warning_once("Encoder: No spectral features generated for WuBu-S input. Using raw reshaped patches. This path may be problematic if PatchEmbed expects specific dim.")
            combined_spectral_features_flat = patches_for_wubus_input_path.reshape(B_flat_wubus * NReg_wubus, -1)
        else:
            combined_spectral_features_flat = torch.cat(combined_spectral_features_for_embed_list, dim=-1)

        initial_tangent_vectors_flat_regions = self.patch_embed(combined_spectral_features_flat)
        wubu_s_output_tangent_flat = self.wubu_s(initial_tangent_vectors_flat_regions)
        D_out_s = wubu_s_output_tangent_flat.shape[-1]
        
        regional_app_features_tangent = wubu_s_output_tangent_flat.reshape(
            B, N_frames_total_sample, self.num_appearance_regions, D_out_s
        )
        agg_app_features = torch.mean(regional_app_features_tangent, dim=2) # (B, N_frames, D_out_s)

        wubu_t_input_features = agg_app_features
        if motion_features is not None and self.args.use_wubu_motion_branch:
             motion_features_tangent = motion_features.to(dtype_model) # (B, N_pairs, N_motion_reg, D_motion)
             N_pairs_motion = motion_features_tangent.shape[1]
             agg_motion_features_per_pair = torch.mean(motion_features_tangent, dim=2) # (B, N_pairs, D_motion)
             aligned_motion_for_wubut = torch.zeros(B, N_frames_total_sample, agg_motion_features_per_pair.shape[-1], device=device, dtype=dtype_model)
             if N_pairs_motion > 0:
                 len_to_copy = min(N_pairs_motion, N_frames_total_sample -1)
                 aligned_motion_for_wubut[:, 1 : 1 + len_to_copy, :] = agg_motion_features_per_pair[:, :len_to_copy, :]
                 if N_pairs_motion < N_frames_total_sample -1 and N_pairs_motion > 0: # Pad if motion seq shorter
                      last_valid_motion = agg_motion_features_per_pair[:, -1, :].unsqueeze(1)
                      if N_frames_total_sample - (1+N_pairs_motion) > 0:
                        aligned_motion_for_wubut[:, 1 + N_pairs_motion :, :] = last_valid_motion.expand(-1, N_frames_total_sample - (1+N_pairs_motion), -1)
             wubu_t_input_features = torch.cat([agg_app_features, aligned_motion_for_wubut], dim=-1)

        if self.wubu_t:
            temporal_features_sequence = self.wubu_t(wubu_t_input_features) # (B, N_frames, D_out_t)
            final_temporal_feature = temporal_features_sequence[:, -1, :] # Use last frame's temporal feature
        else:
            final_temporal_feature = torch.mean(wubu_t_input_features, dim=1) # Avg over time if no WuBu-T

        mu = self.fc_mu(final_temporal_feature)
        logvar = self.fc_logvar(final_temporal_feature)

        # Reshape target features for trainer: (B, N_frames, NumReg, D_spectral_target)
        final_target_dft = target_dft_features_flat.view(B, N_frames_total_sample, self.num_appearance_regions, -1) if target_dft_features_flat is not None else None
        final_target_dct = target_dct_features_flat.view(B, N_frames_total_sample, self.num_appearance_regions, -1) if target_dct_features_flat is not None else None
        
        return mu, logvar, gaad_bboxes_full_batch_sequences, final_target_dft, final_target_dct


class RegionalGeneratorDecoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, latent_dim: int):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.image_size = args.image_h_w_tuple # Ensure this is set in parse_arguments
        self.num_regions = gaad_config['num_regions']
        self.num_img_channels = video_config['num_channels']
        self.latent_dim = latent_dim
        self.num_predict_frames = video_config["num_predict_frames"]
        self.logger = logging.getLogger("WuBuGAADHybridGenV03.Generator")

        # Determine initial spatial resolution and number of upsampling layers
        min_target_dim = min(self.image_size[0], self.image_size[1])
        if min_target_dim <= 8: self.gen_init_spatial_res = 1
        elif min_target_dim <= 32: self.gen_init_spatial_res = 2
        else: self.gen_init_spatial_res = 4
        
        target_upsample_factor = min_target_dim / self.gen_init_spatial_res if self.gen_init_spatial_res > 0 else 0
        if target_upsample_factor > 0 and math.log2(target_upsample_factor).is_integer():
            self.gen_num_upsampling_layers = int(math.log2(target_upsample_factor))
        else:
            self.gen_num_upsampling_layers = max(1, int(math.ceil(math.log2(target_upsample_factor))) if target_upsample_factor > 0 else 1)

        calculated_final_res = self.gen_init_spatial_res * (2**self.gen_num_upsampling_layers)
        self.needs_final_adaptive_pool = (calculated_final_res != min_target_dim)
        if self.needs_final_adaptive_pool:
            self.logger.warning(f"Gen calculated final res {calculated_final_res} vs target {min_target_dim}. Final adaptive pool will be used for pixel output path if active.")

        self.gen_init_channels = min(512, max(128, self.latent_dim * 2))
        self.gen_temporal_kernel_size = getattr(args, 'gen_temporal_kernel_size', 3)

        self.fc_expand_latent = nn.Linear(self.latent_dim, self.gen_init_channels * self.num_predict_frames * self.gen_init_spatial_res * self.gen_init_spatial_res)

        self.gaad_condition_dim = max(32, self.latent_dim // 4)
        if self.num_regions > 0 and getattr(args, 'gen_use_gaad_film_condition', True):
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
        min_gen_channels_final_block = max(32, self.num_img_channels * 8) 

        for i in range(self.gen_num_upsampling_layers):
            out_channels = max(min_gen_channels_final_block, current_channels // 2) if i < self.gen_num_upsampling_layers -1 else min_gen_channels_final_block
            block = nn.ModuleDict(); block['conv_transpose'] = nn.ConvTranspose3d(current_channels, out_channels, kernel_size=(self.gen_temporal_kernel_size, 4, 4), stride=(1, 2, 2), padding=(padding_temp, 1, 1), bias=False)
            block['norm'] = nn.InstanceNorm3d(out_channels, affine=True);
            if self.frame_gaad_embedder is not None: block['film'] = FiLMLayer(out_channels, self.gaad_condition_dim)
            block['activation'] = nn.GELU(); self.upsample_blocks.append(block); current_channels = out_channels
        
        self.final_dense_feature_channels = current_channels
        
        # --- Spectral Feature Prediction Heads ---
        self.gen_patch_h_spectral = args.spectral_patch_size_h
        self.gen_patch_w_spectral = args.spectral_patch_size_w
        self.gen_roi_align_output_spatial = (self.gen_patch_h_spectral, self.gen_patch_w_spectral)
        roi_feat_dim_for_heads = self.final_dense_feature_channels * self.gen_patch_h_spectral * self.gen_patch_w_spectral

        self.to_dft_coeffs_mlp: Optional[nn.Module] = None
        self.to_dct_coeffs_mlp: Optional[nn.Module] = None

        if args.use_dft_features_appearance:
            dft_w_coeffs_one_sided_gen = self.gen_patch_w_spectral // 2 + 1
            dft_coeff_output_dim_per_region = self.num_img_channels * 2 * self.gen_patch_h_spectral * dft_w_coeffs_one_sided_gen
            hidden_mlp_dft = max(dft_coeff_output_dim_per_region // 2, roi_feat_dim_for_heads // 2, 128)
            self.to_dft_coeffs_mlp = nn.Sequential(
                nn.Linear(roi_feat_dim_for_heads, hidden_mlp_dft), nn.GELU(),
                nn.Linear(hidden_mlp_dft, dft_coeff_output_dim_per_region),
                nn.Tanh() # Output normalized DFT features (real/imag parts in [-1,1])
            )
            self.logger.info(f"Gen DFT Head: RoIAlign spatial {self.gen_roi_align_output_spatial}, Projects {roi_feat_dim_for_heads} to {dft_coeff_output_dim_per_region} DFT coeffs/region. Output Tanh.")
        
        if args.use_dct_features_appearance:
            dct_coeff_output_dim_per_region = self.num_img_channels * self.gen_patch_h_spectral * self.gen_patch_w_spectral
            hidden_mlp_dct = max(dct_coeff_output_dim_per_region // 2, roi_feat_dim_for_heads // 2, 128)
            self.to_dct_coeffs_mlp = nn.Sequential(
                nn.Linear(roi_feat_dim_for_heads, hidden_mlp_dct), nn.GELU(),
                nn.Linear(hidden_mlp_dct, dct_coeff_output_dim_per_region),
                nn.Tanh() # Output normalized DCT features in [-1,1] if args.dct_norm_type == 'tanh'
            )
            self.logger.info(f"Gen DCT Head: RoIAlign spatial {self.gen_roi_align_output_spatial}, Projects {roi_feat_dim_for_heads} to {dct_coeff_output_dim_per_region} DCT coeffs/region. Output Tanh.")

        # Fallback to Pixel output if NO spectral features are selected
        if not args.use_dft_features_appearance and not args.use_dct_features_appearance:
            self.logger.warning("Generator: No spectral features (DFT or DCT) selected for output. Defaulting to pixel output path.")
            final_conv_padding_spatial = 1 if getattr(args, 'gen_final_conv_kernel_spatial', 3) > 1 else 0
            self.final_conv_pixel = nn.Conv3d(current_channels, self.num_img_channels, 
                                              kernel_size=(self.gen_temporal_kernel_size, 
                                                           getattr(args, 'gen_final_conv_kernel_spatial', 3), 
                                                           getattr(args, 'gen_final_conv_kernel_spatial', 3)), 
                                              padding=(padding_temp, final_conv_padding_spatial, final_conv_padding_spatial))
            self.final_activation_pixel = nn.Tanh() # Output pixels in [-1, 1]
            self.logger.info(f"Generator Pixel Head (Fallback): FinalConv output channels {self.num_img_channels}")
        
        self.apply(init_weights_general)

    def _normalize_bboxes(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor: # Unchanged
        x1, y1, x2, y2 = bboxes.unbind(-1); img_W = float(W) if W > 0 else 1.0; img_H = float(H) if H > 0 else 1.0
        norm_cx = ((x1 + x2) / 2.0) / img_W; norm_cy = ((y1 + y2) / 2.0) / img_H
        norm_w = (x2 - x1).abs() / img_W; norm_h = (y2 - y1).abs() / img_H; return torch.stack([norm_cx, norm_cy, norm_w, norm_h], dim=-1)

    def forward(self, latent_code: torch.Tensor, gaad_bboxes_for_decode: Optional[torch.Tensor]
               ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Output: predicted_pixel_frames (Optional), predicted_dft_coeffs (Optional), predicted_dct_coeffs (Optional)
        B = latent_code.shape[0]; device = latent_code.device; dtype_in = latent_code.dtype
        x = self.fc_expand_latent(latent_code)
        x = x.view(B, self.gen_init_channels, self.num_predict_frames, self.gen_init_spatial_res, self.gen_init_spatial_res).to(dtype_in)
        sequence_condition = None
        if self.frame_gaad_embedder is not None:
            if gaad_bboxes_for_decode is not None and (gaad_bboxes_for_decode.shape[0] != B or gaad_bboxes_for_decode.shape[1] != self.num_predict_frames or gaad_bboxes_for_decode.shape[2] != self.num_regions):
                self.logger.warning_once(f"Gen GAAD bbox shape mismatch. Expected (B={B}, N_pred={self.num_predict_frames}, NumReg={self.num_regions}, 4), got {gaad_bboxes_for_decode.shape if gaad_bboxes_for_decode is not None else 'None'}. Zero cond."); frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)
            elif gaad_bboxes_for_decode is not None: norm_bboxes = self._normalize_bboxes(gaad_bboxes_for_decode.to(dtype_in), self.image_size[0], self.image_size[1]); norm_bboxes_flat = norm_bboxes.view(B * self.num_predict_frames, -1); frame_conditions_flat = self.frame_gaad_embedder(norm_bboxes_flat)
            else: frame_conditions_flat = torch.zeros(B * self.num_predict_frames, self.gaad_condition_dim, device=device, dtype=dtype_in)
            if frame_conditions_flat is not None: frame_conditions_reshaped = frame_conditions_flat.view(B, self.num_predict_frames, self.gaad_condition_dim); sequence_condition = torch.mean(frame_conditions_reshaped, dim=1).to(dtype_in)
        for block_idx, block in enumerate(self.upsample_blocks):
            x = block['conv_transpose'](x); x = block['norm'](x)
            if 'film' in block and block['film'] is not None and sequence_condition is not None: x = block['film'](x, sequence_condition)
            x = block['activation'](x)
        
        output_dft_coeffs: Optional[torch.Tensor] = None
        output_dct_coeffs: Optional[torch.Tensor] = None
        output_pixel_frames: Optional[torch.Tensor] = None

        if self.args.use_dft_features_appearance or self.args.use_dct_features_appearance:
            if gaad_bboxes_for_decode is None:
                self.logger.error("Generator spectral path: gaad_bboxes_for_decode is None. Cannot produce regional spectral features.")
                if self.args.use_dft_features_appearance: dft_w_coeffs_gen = self.gen_patch_w_spectral // 2 + 1; output_dft_coeffs = torch.zeros(B, self.num_predict_frames, self.num_regions, self.num_img_channels, 2, self.gen_patch_h_spectral, dft_w_coeffs_gen, device=device, dtype=dtype_in)
                if self.args.use_dct_features_appearance: output_dct_coeffs = torch.zeros(B, self.num_predict_frames, self.num_regions, self.num_img_channels, self.gen_patch_h_spectral, self.gen_patch_w_spectral, device=device, dtype=dtype_in)
                return output_pixel_frames, output_dft_coeffs, output_dct_coeffs

            all_frames_regional_dft_coeffs_list = [] if self.args.use_dft_features_appearance and self.to_dft_coeffs_mlp else None
            all_frames_regional_dct_coeffs_list = [] if self.args.use_dct_features_appearance and self.to_dct_coeffs_mlp else None
            
            H_final_feat, W_final_feat = x.shape[-2], x.shape[-1]
            roi_spatial_scale = H_final_feat / self.image_size[0]

            for f_idx in range(self.num_predict_frames):
                frame_feature_map = x[:, :, f_idx, :, :]; frame_bboxes = gaad_bboxes_for_decode[:, f_idx, :, :]
                rois_for_frame_list = []
                for b_s_idx in range(B): batch_indices = torch.full((self.num_regions, 1), float(b_s_idx), device=device, dtype=dtype_in); rois_for_frame_list.append(torch.cat([batch_indices, frame_bboxes[b_s_idx]], dim=1))
                all_rois_this_frame = torch.cat(rois_for_frame_list, dim=0)
                regional_feats_from_roi = roi_align(frame_feature_map, all_rois_this_frame, output_size=self.gen_roi_align_output_spatial, spatial_scale=roi_spatial_scale, aligned=True)
                regional_feats_flat = regional_feats_from_roi.reshape(B * self.num_regions, -1)

                if self.args.use_dft_features_appearance and self.to_dft_coeffs_mlp is not None and all_frames_regional_dft_coeffs_list is not None:
                    dft_coeffs_flat_for_frame = self.to_dft_coeffs_mlp(regional_feats_flat)
                    dft_w_coeffs_one_sided_gen = self.gen_patch_w_spectral // 2 + 1
                    frame_dft_structured = dft_coeffs_flat_for_frame.view(B, self.num_regions, self.num_img_channels, 2, self.gen_patch_h_spectral, dft_w_coeffs_one_sided_gen)
                    all_frames_regional_dft_coeffs_list.append(frame_dft_structured)
                
                if self.args.use_dct_features_appearance and self.to_dct_coeffs_mlp is not None and all_frames_regional_dct_coeffs_list is not None:
                    dct_coeffs_flat_for_frame = self.to_dct_coeffs_mlp(regional_feats_flat)
                    frame_dct_structured = dct_coeffs_flat_for_frame.view(B, self.num_regions, self.num_img_channels, self.gen_patch_h_spectral, self.gen_patch_w_spectral)
                    all_frames_regional_dct_coeffs_list.append(frame_dct_structured)
            
            if all_frames_regional_dft_coeffs_list: output_dft_coeffs = torch.stack(all_frames_regional_dft_coeffs_list, dim=1).to(dtype_in)
            if all_frames_regional_dct_coeffs_list: output_dct_coeffs = torch.stack(all_frames_regional_dct_coeffs_list, dim=1).to(dtype_in)
        
        else: # Fallback to Pixel Output Path
            if hasattr(self, 'final_conv_pixel') and hasattr(self, 'final_activation_pixel'):
                x_px = self.final_conv_pixel(x) # type: ignore
                output_pixel_frames_raw = self.final_activation_pixel(x_px) # type: ignore
                output_pixel_frames = output_pixel_frames_raw.permute(0, 2, 1, 3, 4).to(dtype_in) # (B, N_pred, C, H, W)
                if self.needs_final_adaptive_pool:
                    self.logger.debug_once(f"Generator pixel output needs adaptive pool. Current shape: {output_pixel_frames.shape[-2:]}")
                    temp_permuted_for_pool = output_pixel_frames.permute(0, 2, 1, 3, 4) # (B, C, N_pred, H_curr, W_curr)
                    pooled = F.adaptive_avg_pool3d(temp_permuted_for_pool, (self.num_predict_frames, self.image_size[0], self.image_size[1]))
                    output_pixel_frames = pooled.permute(0, 2, 1, 3, 4)
            else: self.logger.error("Generator: Fallback pixel path missing final_conv_pixel or final_activation_pixel.")
        return output_pixel_frames, output_dft_coeffs, output_dct_coeffs


class VideoDiscriminatorWrapper(nn.Module):
    """
    Wrapper to manage different discriminator architectures (pixel-based, feature-based).
    This adapts the structure from `AudioSpecDiscriminator` for video.
    """
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.disc_config_orig = disc_config.copy() # Store original config for this instance

        # `architecture_variant` now guides which sub-discriminator to build.
        # `input_type` from disc_config might be legacy or for the 'default' variant.
        self.architecture_variant = disc_config.get("architecture_variant", "default_pixel_cnn")
        self.logger = logging.getLogger(f"WuBuGAADHybridGenV03.VideoDiscWrapper.{self.architecture_variant}.{id(self)}")
        
        self.actual_discriminator_module: Optional[nn.Module] = None
        self.effective_input_type_for_trainer: str = "unknown" # To be set by the actual D module

        if self.architecture_variant == "global_wubu_video_feature":
            self.logger.info(f"Wrapper: Instantiating GlobalWuBuVideoFeatureDiscriminator.")
            # Pass the specific 'disc_config' intended for this variant
            self.actual_discriminator_module = GlobalWuBuVideoFeatureDiscriminator(args, video_config, gaad_config, disc_config)
        elif self.architecture_variant == "default_pixel_cnn": # This is the original RegionalDiscriminator
            self.logger.info(f"Wrapper: Instantiating default RegionalDiscriminator (pixel-based CNN).")
            # The original RegionalDiscriminator doesn't need a separate sub-config dict as it reads from args/main_disc_config
            self.actual_discriminator_module = RegionalDiscriminator(args, video_config, gaad_config, disc_config)
        # Add other variants like a MultiScalePixelDiscriminator if needed
        # elif self.architecture_variant == "multi_scale_pixel_cnn":
        #     self.actual_discriminator_module = MultiScaleVideoDiscriminator(args, video_config, disc_config)
        else:
            self.logger.error(f"Wrapper: Unknown architecture_variant '{self.architecture_variant}'. Check config.")
            # Fallback or raise error
            raise ValueError(f"Unsupported discriminator architecture_variant: {self.architecture_variant}")

        if self.actual_discriminator_module is not None and hasattr(self.actual_discriminator_module, 'effective_input_type_for_trainer'):
            self.effective_input_type_for_trainer = self.actual_discriminator_module.effective_input_type_for_trainer
        elif self.actual_discriminator_module is not None and isinstance(self.actual_discriminator_module, RegionalDiscriminator):
            # RegionalDiscriminator (default_pixel_cnn) always takes assembled pixels
            self.effective_input_type_for_trainer = "assembled_pixels"
        else:
            self.logger.warning(f"Effective input type for trainer not explicitly set by D module '{self.architecture_variant}'. Inferring.")
            if self.architecture_variant == "global_wubu_video_feature":
                self.effective_input_type_for_trainer = "regional_spectral_features_combined" # (DFT+DCT)
            elif self.architecture_variant == "default_pixel_cnn":
                self.effective_input_type_for_trainer = "assembled_pixels"
            else:
                self.effective_input_type_for_trainer = "unknown"

        active_module_name = self.actual_discriminator_module.__class__.__name__ if self.actual_discriminator_module else "Wrapper_Error"
        self.logger.info(f"VideoDiscriminatorWrapper: Active D Module: {active_module_name}, Effective input for trainer: '{self.effective_input_type_for_trainer}'")

    def forward(self, main_input_data: torch.Tensor,
                gaad_bboxes_cond: Optional[torch.Tensor] = None, # For pixel-based D's FiLM
                aux_input_data: Optional[torch.Tensor] = None, # For multi-modal Ds, e.g. raw video stats
                return_features: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        if self.actual_discriminator_module is None:
            self.logger.error("VideoDiscriminatorWrapper: actual_discriminator_module is None. Cannot forward.")
            dummy_logits = torch.zeros(main_input_data.size(0), device=main_input_data.device, dtype=main_input_data.dtype)
            dummy_features = torch.zeros(main_input_data.size(0), 1, device=main_input_data.device, dtype=main_input_data.dtype)
            return (dummy_logits, dummy_features) if return_features else dummy_logits

        # Delegate to the actual discriminator module
        # The actual module's forward signature needs to be compatible with what the trainer provides
        if isinstance(self.actual_discriminator_module, RegionalDiscriminator):
            # RegionalDiscriminator (default_pixel_cnn) takes (frames_pixels, gaad_bboxes_for_disc)
            return self.actual_discriminator_module(main_input_data, gaad_bboxes_cond) # Does not use return_features or aux_input
        
        elif isinstance(self.actual_discriminator_module, GlobalWuBuVideoFeatureDiscriminator):
            # GlobalWuBuVideoFeatureDiscriminator takes (regional_features_input, return_features)
            # It doesn't use gaad_bboxes_cond or aux_input_data in its current form.
            return self.actual_discriminator_module(main_input_data, return_features=return_features)
        
        # Example for a hypothetical MultiModalVideoDiscriminator (if you add one)
        # elif isinstance(self.actual_discriminator_module, MultiModalVideoDiscriminator):
        #     # This D might take assembled_pixels (main_input_data), gaad_bboxes (gaad_bboxes_cond for FiLM on CNN part),
        #     # and raw_video_stats (aux_input_data)
        #     return self.actual_discriminator_module(main_input_data, gaad_bboxes_cond, aux_input_data, return_features)
            
        else:
            self.logger.error(f"VideoDiscriminatorWrapper: Forward pass not implemented for module type {type(self.actual_discriminator_module)}. Using direct call.")
            # Fallback attempt: direct call if signature matches. This might fail.
            try:
                if hasattr(self.actual_discriminator_module, 'forward') and callable(getattr(self.actual_discriminator_module, 'forward')):
                     # Try to call with all possible arguments, the D should ignore what it doesn't need
                    if return_features:
                         if "aux_input_data" in inspect.signature(self.actual_discriminator_module.forward).parameters:
                              return self.actual_discriminator_module(main_input_data, gaad_bboxes_cond, aux_input_data, return_features=True)
                         elif "gaad_bboxes_cond" in inspect.signature(self.actual_discriminator_module.forward).parameters:
                              return self.actual_discriminator_module(main_input_data, gaad_bboxes_cond, return_features=True)
                         else:
                              return self.actual_discriminator_module(main_input_data, return_features=True)
                    else:
                         if "aux_input_data" in inspect.signature(self.actual_discriminator_module.forward).parameters:
                              return self.actual_discriminator_module(main_input_data, gaad_bboxes_cond, aux_input_data)
                         elif "gaad_bboxes_cond" in inspect.signature(self.actual_discriminator_module.forward).parameters:
                              return self.actual_discriminator_module(main_input_data, gaad_bboxes_cond)
                         else:
                              return self.actual_discriminator_module(main_input_data)
            except Exception as e_fwd_fallback:
                 self.logger.error(f"Fallback forward call failed for {type(self.actual_discriminator_module)}: {e_fwd_fallback}")
                 dummy_logits = torch.zeros(main_input_data.size(0), device=main_input_data.device, dtype=main_input_data.dtype)
                 dummy_features = torch.zeros(main_input_data.size(0), 1, device=main_input_data.device, dtype=main_input_data.dtype)
                 return (dummy_logits, dummy_features) if return_features else dummy_logits
        # This default return should only be hit if the specific D type isn't handled above.
        # For a properly configured setup, one of the `isinstance` blocks should handle it.
        # Fallback just in case.
        self.logger.error("Unhandled discriminator type in wrapper forward. Returning zeros.")
        dummy_logits = torch.zeros(main_input_data.size(0), device=main_input_data.device, dtype=main_input_data.dtype)
        dummy_features = torch.zeros(main_input_data.size(0), 1, device=main_input_data.device, dtype=main_input_data.dtype)
        return (dummy_logits, dummy_features) if return_features else dummy_logits

class GlobalWuBuVideoFeatureDiscriminator(nn.Module):
    """
    Discriminator that operates on concatenated regional DFT+DCT features.
    Adapted from GlobalWuBuDCTDiscriminator in the audio script.
    """
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger(f"WuBuGAADHybridGenV03.GlobalWuBuVideoFeatureD.{id(self)}")
        
        # This D takes features directly, so this is what the trainer should prepare
        self.effective_input_type_for_trainer = "regional_spectral_features_combined" # DFT + DCT

        self.num_gaad_regions = gaad_config['num_regions']
        self.patch_h_spectral = args.spectral_patch_size_h
        self.patch_w_spectral = args.spectral_patch_size_w
        self.num_img_channels = video_config['num_channels']

        # Define how many frames of features this specific discriminator variant processes.
        # This attribute will be used by the HybridTrainer to determine how many frames of features to pass.
        self.num_frames_to_discriminate = disc_config.get(
            "num_frames_input_for_global_wubu_d", # Check specific disc_config first
            getattr(args, 'video_global_wubu_d_num_frames_input', args.num_predict_frames) 
        )
        self.logger.info(f"GlobalWuBuVideoFeatureD configured to process features from {self.num_frames_to_discriminate} frame(s).")


        # Calculate input dimension per region PER FRAME based on what VAE encoder outputs
        features_per_region_from_g_dft_per_frame = 0
        if args.use_dft_features_appearance:
            dft_w_coeffs_one_sided = self.patch_w_spectral // 2 + 1
            features_per_region_from_g_dft_per_frame = self.num_img_channels * 2 * self.patch_h_spectral * dft_w_coeffs_one_sided
        
        features_per_region_from_g_dct_per_frame = 0
        if args.use_dct_features_appearance:
            features_per_region_from_g_dct_per_frame = self.num_img_channels * self.patch_h_spectral * self.patch_w_spectral
        
        # This is the dimension of combined (DFT+DCT) features for ONE region for ONE frame
        self.num_features_per_region_per_frame_input = features_per_region_from_g_dft_per_frame + features_per_region_from_g_dct_per_frame
        if self.num_features_per_region_per_frame_input == 0:
             self.logger.error("GlobalWuBuVideoFeatureD: num_features_per_region_per_frame_input is 0. This D will not work.")
             self.num_features_per_region_per_frame_input = 1 # Dummy value
        
        # The total input dimension expected by the initial_projection layer.
        # It's (NumRegions * FeaturesPerRegionPerFrame * NumFramesThisDProcesses)
        self.total_input_feature_dim_for_projection = (
            self.num_gaad_regions * 
            self.num_features_per_region_per_frame_input * 
            self.num_frames_to_discriminate
        )
        # self.total_input_feature_dim was the old name, let's keep it for compatibility if needed by logs,
        # but total_input_feature_dim_for_projection is more descriptive for the Linear layer.
        self.total_input_feature_dim = self.total_input_feature_dim_for_projection


        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", getattr(args, 'disc_apply_spectral_norm', True))
        
        # Global stats aux input (mean/std of input features) - optional
        self.use_global_stats_aux = disc_config.get("disc_use_global_stats_aux_video_global_wubu", 
                                                    getattr(args, 'disc_use_global_stats_aux_video_global_wubu', False))
        self.global_stats_mlp: Optional[nn.Module] = None
        current_projection_input_dim = self.total_input_feature_dim_for_projection # Use the correctly calculated dim for the Linear layer

        if self.use_global_stats_aux:
            self.num_global_stats_outputs = 2 # Mean, Std
            stats_mlp_hidden_dim_key = "disc_global_stats_mlp_hidden_dim_video_global_wubu"
            self.global_stats_mlp_hidden_dim = disc_config.get(stats_mlp_hidden_dim_key, getattr(args, stats_mlp_hidden_dim_key, 64))
            if self.num_global_stats_outputs > 0 and self.global_stats_mlp_hidden_dim > 0:
                self.global_stats_mlp = nn.Sequential(
                    nn.Linear(self.num_global_stats_outputs, self.global_stats_mlp_hidden_dim), nn.LeakyReLU(0.2, True),
                    nn.Linear(self.global_stats_mlp_hidden_dim, self.global_stats_mlp_hidden_dim)
                )
                if self.apply_spectral_norm:
                    self.global_stats_mlp[0] = spectral_norm(self.global_stats_mlp[0]) # type: ignore
                    self.global_stats_mlp[2] = spectral_norm(self.global_stats_mlp[2]) # type: ignore
                current_projection_input_dim += self.global_stats_mlp_hidden_dim # Add to the input of the main projection
            else: self.use_global_stats_aux = False
        
        # Input tangent dim for the WuBu stack
        self.global_wubu_input_tangent_dim = getattr(args, 'video_global_wubu_d_input_tangent_dim', 512)
        
        if current_projection_input_dim > 0 and self.global_wubu_input_tangent_dim > 0:
            self.initial_projection = nn.Linear(current_projection_input_dim, self.global_wubu_input_tangent_dim)
            self.initial_layernorm = nn.LayerNorm(self.global_wubu_input_tangent_dim)
        else:
            self.logger.warning(f"GlobalWuBuVideoFeatureD: Initial projection has zero input ({current_projection_input_dim}) or output ({self.global_wubu_input_tangent_dim}) dim. Using Identity.")
            self.initial_projection = nn.Identity(); self.initial_layernorm = nn.Identity()

        # WuBu stack configuration
        wubu_config = _configure_wubu_stack(args, "wubu_d_global_video") # New prefix for video D
        self.wubu_output_dim = getattr(args, 'video_global_wubu_d_output_feature_dim', 256)

        if wubu_config and wubu_config.get("num_levels", 0) > 0 and self.global_wubu_input_tangent_dim > 0 :
            if wubu_config.get('hyperbolic_dims') and wubu_config['hyperbolic_dims'][-1] > 0:
                self.wubu_output_dim = wubu_config['hyperbolic_dims'][-1]
            self.wubu_stack = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=self.global_wubu_input_tangent_dim,
                output_tangent_dim=self.wubu_output_dim,
                config=wubu_config
            )
            self.logger.info(f"GlobalWuBuVideoFeatureD: WuBu stack active ({wubu_config.get('num_levels')} levels). Output dim: {self.wubu_output_dim}")
        else:
            self.logger.warning(f"GlobalWuBuVideoFeatureD: WuBu stack not configured or input_tangent_dim is 0. Using MLP fallback.")
            if self.global_wubu_input_tangent_dim > 0 and self.wubu_output_dim > 0:
                self.wubu_stack = nn.Sequential(
                    nn.Linear(self.global_wubu_input_tangent_dim, self.global_wubu_input_tangent_dim * 2),
                    nn.LeakyReLU(0.2, True), nn.LayerNorm(self.global_wubu_input_tangent_dim * 2),
                    nn.Linear(self.global_wubu_input_tangent_dim * 2, self.wubu_output_dim))
            else: self.wubu_stack = nn.Identity()

        # Final decision layer
        if self.wubu_output_dim > 0:
            self.final_decision_layer = nn.Linear(self.wubu_output_dim, 1)
            if self.apply_spectral_norm: self.final_decision_layer = spectral_norm(self.final_decision_layer)
        else:
            self.logger.error("GlobalWuBuVideoFeatureD: final_decision_layer input dim is 0. Using Identity.")
            self.final_decision_layer = nn.Identity()
        
        self.apply(init_weights_general)
        self.logger.info(f"GlobalWuBuVideoFeatureD initialized. Expects features from {self.num_frames_to_discriminate} frames. Total input dim for projection: {self.total_input_feature_dim_for_projection}. Features per region per frame: {self.num_features_per_region_per_frame_input}")

    def _calculate_global_feature_stats_for_video_features(self, regional_features_input_to_stats: torch.Tensor) -> torch.Tensor:
        # regional_features_input_to_stats: (B, NumRegions, NumFramesThisDProcesses * FeaturesPerRegionPerFrame)
        # This input is already flattened over frames for each region.
        # We need to calculate stats over the entire feature vector *per region*, then average those stats.
        # OR average regions first, then stats. Let's average regions first.

        if regional_features_input_to_stats.numel() == 0: 
            return torch.zeros(regional_features_input_to_stats.shape[0], 2, device=regional_features_input_to_stats.device, dtype=regional_features_input_to_stats.dtype)
        
        # Reshape to (B, NumRegions, NumFrames, FeaturesPerRegionPerFrame) to average over regions & frames correctly if desired.
        # Current input is (B, NumRegions, NumFramesThisDProcesses * FeaturesPerRegionPerFrame)
        # For simplicity, the audio version averaged over the already combined feature vector.
        # Let's average across regions first, resulting in (B, NumFrames * FeaturesPerRegionPerFrame)
        mean_features_across_regions = torch.mean(regional_features_input_to_stats, dim=1) # (B, NumFramesThisDProcesses * FeaturesPerRegionPerFrame)
        
        # Then calculate mean and std of this single averaged (over regions) feature vector.
        mean_stat = torch.mean(mean_features_across_regions, dim=1) # (B,)
        std_stat = torch.std(mean_features_across_regions, dim=1)   # (B,)
        std_stat = torch.max(std_stat, torch.tensor(EPS, device=std_stat.device, dtype=std_stat.dtype)) # Avoid NaN/Inf
        return torch.stack([mean_stat, std_stat], dim=-1) # (B, 2)

    def forward(self, regional_spectral_features_input: torch.Tensor, # (B, NumRegions, NumFramesThisDProcesses * FeaturesPerRegionPerFrame)
                return_features: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = regional_spectral_features_input.shape[0]
        # The input regional_spectral_features_input is expected to be:
        # (Batch, NumRegions, NumFramesThisDProcesses * FeaturesPerRegionPerFrame)
        # as prepared by the HybridTrainer.

        if regional_spectral_features_input.numel() == 0 or self.num_features_per_region_per_frame_input == 0:
            self.logger.warning_once("GlobalWuBuVideoFeatureD: Forward called with empty input or zero feature_per_region_per_frame dim. Returning zeros.")
            dummy_logits = torch.zeros(B, device=regional_spectral_features_input.device, dtype=regional_spectral_features_input.dtype)
            dummy_feat_ret_dim = self.wubu_output_dim if self.wubu_output_dim > 0 else 1
            dummy_features_ret = torch.zeros(B, dummy_feat_ret_dim, device=regional_spectral_features_input.device,dtype=regional_spectral_features_input.dtype)
            return (dummy_logits, dummy_features_ret) if return_features else dummy_logits

        # Flatten all regional and frame features for the initial_projection layer
        # Input is (B, NumRegions, NumFrames * FeatPerRegPerFrame)
        # Reshape to (B, NumRegions * NumFrames * FeatPerRegPerFrame)
        flat_all_batch_features = regional_spectral_features_input.reshape(B, -1) 
        
        # Check if the flattened dimension matches what the initial_projection layer expects (excluding aux stats for now)
        if flat_all_batch_features.shape[1] != self.total_input_feature_dim_for_projection:
             self.logger.error_once(f"GlobalWuBuVideoFeatureD forward: Shape mismatch for initial_projection. "
                               f"Input flat features dim: {flat_all_batch_features.shape[1]}, "
                               f"Expected (NumReg * FeatPerRegPerFrame * NumFramesD): {self.total_input_feature_dim_for_projection}. "
                               f"Input regional_spectral_features_input shape: {regional_spectral_features_input.shape}. "
                               f"D Config: NumReg={self.num_gaad_regions}, FeatPerRegPerFrame={self.num_features_per_region_per_frame_input}, NumFramesD={self.num_frames_to_discriminate}")
             # This is a critical error if it occurs.
             # Fallback to returning zeros if shapes don't match.
             dummy_logits = torch.zeros(B, device=regional_spectral_features_input.device, dtype=regional_spectral_features_input.dtype)
             dummy_feat_ret_dim = self.wubu_output_dim if self.wubu_output_dim > 0 else 1
             dummy_features_ret = torch.zeros(B, dummy_feat_ret_dim, device=regional_spectral_features_input.device,dtype=regional_spectral_features_input.dtype)
             return (dummy_logits, dummy_features_ret) if return_features else dummy_logits
        
        current_input_to_projection = flat_all_batch_features
        if self.use_global_stats_aux and self.global_stats_mlp is not None:
            # Calculate stats based on the input regional_spectral_features_input
            # The input to _calculate_global_feature_stats_for_video_features is (B, NumRegions, NumFramesThisDProcesses * FeaturesPerRegionPerFrame)
            stats = self._calculate_global_feature_stats_for_video_features(regional_spectral_features_input.detach())
            projected_stats = self.global_stats_mlp(stats)
            current_input_to_projection = torch.cat([flat_all_batch_features, projected_stats], dim=-1)

        projected_tangent = self.initial_projection(current_input_to_projection)
        projected_tangent_norm = self.initial_layernorm(projected_tangent)
        
        wubu_out_features = self.wubu_stack(projected_tangent_norm) # (B, wubu_output_dim)
        
        if isinstance(self.final_decision_layer, nn.Identity): # Should not happen if wubu_output_dim > 0
            logits = torch.mean(wubu_out_features, dim=-1) if wubu_out_features.numel() > 0 else torch.zeros(B, device=wubu_out_features.device, dtype=wubu_out_features.dtype)
        else:
            logits = self.final_decision_layer(wubu_out_features).squeeze(-1) # (B,)

        return (logits, wubu_out_features) if return_features else logits

# --- RegionalDiscriminator (original from v0.2, now one of the options in VideoDiscriminatorWrapper) ---
# This is effectively the "default_pixel_cnn" variant.
class RegionalDiscriminator(nn.Module): # Unchanged from v0.2, but now used as a sub-module by wrapper
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_config: Dict, disc_config: Dict):
        super().__init__()
        self.args = args; self.video_config = video_config; self.gaad_config = gaad_config; self.disc_config = disc_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV03.RegionalPixelD")
        self.image_size = args.image_h_w_tuple; self.num_channels = video_config['num_channels']
        self.num_frames_to_discriminate = max(1, video_config.get("num_predict_frames", 1))
        self.num_regions = self.gaad_config.get('num_regions', 0)
        self.apply_spectral_norm = disc_config.get("apply_spectral_norm", getattr(args, 'disc_apply_spectral_norm', True))
        self.use_gaad_film_condition = disc_config.get("use_gaad_film_condition", getattr(args, 'disc_use_gaad_film_condition', False)) and self.num_regions > 0
        self.effective_input_type_for_trainer = "assembled_pixels" # This D always sees pixels

        if self.use_gaad_film_condition:
            self.gaad_condition_dim = disc_config.get("gaad_condition_dim_disc", getattr(args, 'disc_gaad_condition_dim_disc', 64))
            self.bbox_feature_dim = 4; hidden_bbox_embed_dim = max(self.gaad_condition_dim, self.num_regions * self.bbox_feature_dim // 2)
            self.frame_gaad_embedder_disc = nn.Sequential(nn.Linear(self.num_regions * self.bbox_feature_dim, hidden_bbox_embed_dim), nn.GELU(), nn.Linear(hidden_bbox_embed_dim, self.gaad_condition_dim))
            self.logger.info(f"RegionalPixelD GAAD-FiLM ENABLED. Cond Dim: {self.gaad_condition_dim}")
        else: self.frame_gaad_embedder_disc = None; self.gaad_condition_dim = 0; self.logger.info("RegionalPixelD GAAD-FiLM DISABLED.")
        
        # This D only supports "spatio_temporal_cnn" type. "regional_cnn" (2D) is not suitable for video sequences directly.
        # The disc_config "type" for this D itself should be spatio_temporal_cnn.
        # If disc_config had a "type" for a sub-component, that would be different.
        self.disc_type_internal = "spatio_temporal_cnn" # This D is always this type

        min_input_dim = min(self.image_size[0], self.image_size[1]); num_spatial_downsamples_target = int(math.log2(min_input_dim / 4)) if min_input_dim >= 8 else 1
        max_possible_downsamples = int(math.log2(min_input_dim)) if min_input_dim > 0 else 0; num_spatial_downsamples = max(1, min(num_spatial_downsamples_target, max_possible_downsamples))
        base_disc_channels = disc_config.get("base_disc_channels", getattr(args, 'disc_base_disc_channels', 64)); cnn3d_channels_list = [base_disc_channels * (2**i) for i in range(num_spatial_downsamples)]
        max_disc_channels = disc_config.get("max_disc_channels", getattr(args, 'disc_max_disc_channels', 512)); cnn3d_channels_list = [min(c, max_disc_channels) for c in cnn3d_channels_list];
        if not cnn3d_channels_list: cnn3d_channels_list = [base_disc_channels]
        temporal_kernel_size = disc_config.get("temporal_kernel_size", getattr(args, 'disc_temporal_kernel_size', 3)); default_temporal_stride = 1
        layers = []; in_c = self.num_channels; current_d_dim = self.num_frames_to_discriminate; current_h_dim = self.image_size[0]; current_w_dim = self.image_size[1]
        for i, out_c in enumerate(cnn3d_channels_list):
            can_halve_spatial = current_h_dim >= 8 and current_w_dim >= 8; spatial_stride = 2 if can_halve_spatial and i < num_spatial_downsamples else 1
            apply_temporal_stride_val = 2; can_stride_temporally = current_d_dim > temporal_kernel_size and current_d_dim >= apply_temporal_stride_val
            actual_temporal_stride = apply_temporal_stride_val if can_stride_temporally and i == 0 else default_temporal_stride
            current_t_kernel = min(temporal_kernel_size, current_d_dim) if current_d_dim > 1 else 1; current_t_padding = current_t_kernel // 2 if current_t_kernel > 1 else 0
            block = nn.ModuleDict(); conv_layer = nn.Conv3d(in_c, out_c, kernel_size=(current_t_kernel, 4, 4), stride=(actual_temporal_stride, spatial_stride, spatial_stride), padding=(current_t_padding, 1, 1), bias=False)
            block['conv'] = spectral_norm(conv_layer) if self.apply_spectral_norm else conv_layer
            if self.apply_spectral_norm and i == 0: self.logger.info("Applying Spectral Norm to RegionalPixelD Conv3D layers.")
            block['norm'] = nn.InstanceNorm3d(out_c, affine=not (self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None))
            if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None: block['film'] = FiLMLayer(out_c, self.gaad_condition_dim)
            block['activation'] = nn.LeakyReLU(0.2, inplace=True); layers.append(block); in_c = out_c
            if current_d_dim > 0: current_d_dim = (current_d_dim + 2 * current_t_padding - (current_t_kernel -1) -1 ) // actual_temporal_stride + 1
            if current_h_dim > 0: current_h_dim = (current_h_dim + 2 * 1 - (4-1) -1 ) // spatial_stride + 1
            if current_w_dim > 0: current_w_dim = (current_w_dim + 2 * 1 - (4-1) -1 ) // spatial_stride + 1
            current_d_dim = max(1, current_d_dim); current_h_dim = max(1, current_h_dim); current_w_dim = max(1, current_w_dim)
        self.feature_extractor_blocks = nn.ModuleList(layers)
        
        # Dummy pass for shape calculation (simplified from audio script)
        _device_for_shape_calc = torch.device('cpu'); test_input_shape = (1, self.num_channels, self.num_frames_to_discriminate, self.image_size[0], self.image_size[1])
        test_input = torch.randn(test_input_shape).to(_device_for_shape_calc)
        temp_features = test_input
        with torch.no_grad():
            for block_module_item_fe in self.feature_extractor_blocks:
                # Create dummy FiLM condition if needed, on CPU
                dummy_sequence_condition_disc_cpu = None
                if 'film' in block_module_item_fe and self.use_gaad_film_condition and self.frame_gaad_embedder_disc:
                    # Minimal FiLM condition for shape calculation
                    dummy_sequence_condition_disc_cpu = torch.randn(1, self.gaad_condition_dim, device=_device_for_shape_calc) # (B=1, CondDim)
                
                temp_features = block_module_item_fe['conv'](temp_features.to(_device_for_shape_calc))
                temp_features = block_module_item_fe['norm'](temp_features)
                if 'film' in block_module_item_fe and dummy_sequence_condition_disc_cpu is not None:
                    temp_features = block_module_item_fe['film'](temp_features, dummy_sequence_condition_disc_cpu)
                temp_features = block_module_item_fe['activation'](temp_features)
        final_feature_map_shape_pre_pool = temp_features.shape
        self.adaptive_pool = nn.AdaptiveAvgPool3d((max(1, final_feature_map_shape_pre_pool[2]), 1, 1)) # Pool spatially, keep temporal if > 1
        with torch.no_grad(): pooled_features_test = self.adaptive_pool(temp_features.to(_device_for_shape_calc))
        final_flattened_dim = max(1, pooled_features_test.numel() // pooled_features_test.shape[0])
        
        min_hidden_fc_dim = getattr(args, 'disc_min_hidden_fc_dim', 128); max_hidden_fc_dim = getattr(args, 'disc_max_hidden_fc_dim', 512)
        if final_flattened_dim <= min_hidden_fc_dim * 1.5 and final_flattened_dim > 0 : fc_layer = nn.Linear(final_flattened_dim, 1); self.final_fc_layers = spectral_norm(fc_layer) if self.apply_spectral_norm else fc_layer
        elif final_flattened_dim > 0 :
            hidden_fc_dim = min(max(min_hidden_fc_dim, final_flattened_dim // 2), max_hidden_fc_dim)
            fc1 = nn.Linear(final_flattened_dim, hidden_fc_dim); fc2 = nn.Linear(hidden_fc_dim, 1)
            self.final_fc_layers = nn.Sequential(spectral_norm(fc1), nn.LeakyReLU(0.2, inplace=True), spectral_norm(fc2)) if self.apply_spectral_norm else nn.Sequential(fc1, nn.LeakyReLU(0.2, inplace=True), fc2)
        else: self.final_fc_layers = nn.Linear(1,1)
        self.logger.info(f"RegionalPixelD (SpatioTemporalCNN): Frames={self.num_frames_to_discriminate}, Channels={cnn3d_channels_list}, FinalFeatMap(pre-pool)={final_feature_map_shape_pre_pool}, Pooled={pooled_features_test.shape}, FlattenedDimFC={final_flattened_dim}")
        self.apply(init_weights_general)

    def _normalize_bboxes_disc(self, bboxes: torch.Tensor, H: int, W: int) -> torch.Tensor: # Unchanged
        x1, y1, x2, y2 = bboxes.unbind(-1); img_W = float(W) if W > 0 else 1.0; img_H = float(H) if H > 0 else 1.0
        norm_cx = ((x1 + x2) / 2.0) / img_W; norm_cy = ((y1 + y2) / 2.0) / img_H
        norm_w = (x2 - x1).abs() / img_W; norm_h = (y2 - y1).abs() / img_H; return torch.stack([norm_cx, norm_cy, norm_w, norm_h], dim=-1)

    def forward(self, frames_pixels: torch.Tensor, gaad_bboxes_for_disc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_seq, C, H, W = frames_pixels.shape; device = frames_pixels.device; dtype_in = frames_pixels.dtype
        if N_seq < self.num_frames_to_discriminate:
            padding_needed = self.num_frames_to_discriminate - N_seq
            last_frame_repeated = frames_pixels[:, -1:, ...].repeat(1, padding_needed, 1, 1, 1)
            frames_to_process = torch.cat([frames_pixels, last_frame_repeated], dim=1)
        elif N_seq > self.num_frames_to_discriminate: frames_to_process = frames_pixels[:, :self.num_frames_to_discriminate, ...]
        else: frames_to_process = frames_pixels
        sequence_condition_disc = None
        if self.use_gaad_film_condition and self.frame_gaad_embedder_disc is not None:
            if gaad_bboxes_for_disc is not None:
                N_bboxes_seq = gaad_bboxes_for_disc.shape[1]; bboxes_to_process_for_film = gaad_bboxes_for_disc
                if N_bboxes_seq != self.num_frames_to_discriminate:
                    if N_bboxes_seq < self.num_frames_to_discriminate: bbox_padding = self.num_frames_to_discriminate - N_bboxes_seq; last_bbox_set_repeated = gaad_bboxes_for_disc[:, -1:, ...].repeat(1, bbox_padding, 1, 1); bboxes_to_process_for_film = torch.cat([gaad_bboxes_for_disc, last_bbox_set_repeated], dim=1)
                    else: bboxes_to_process_for_film = gaad_bboxes_for_disc[:, :self.num_frames_to_discriminate, ...]
                if (bboxes_to_process_for_film.shape[0] != B or bboxes_to_process_for_film.shape[1] != self.num_frames_to_discriminate or (self.num_regions > 0 and bboxes_to_process_for_film.shape[2] != self.num_regions)):
                    self.logger.warning_once(f"RegionalPixelD GAAD bbox shape mismatch for FiLM. Zero cond."); frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
                elif self.num_regions > 0 : norm_bboxes_disc = self._normalize_bboxes_disc(bboxes_to_process_for_film.to(dtype_in), H, W); norm_bboxes_flat_disc = norm_bboxes_disc.view(B * self.num_frames_to_discriminate, -1); frame_conditions_flat_disc = self.frame_gaad_embedder_disc(norm_bboxes_flat_disc)
                else: frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
            else: frame_conditions_flat_disc = torch.zeros(B * self.num_frames_to_discriminate, self.gaad_condition_dim, device=device, dtype=dtype_in)
            if frame_conditions_flat_disc is not None: frame_conditions_reshaped_disc = frame_conditions_flat_disc.view(B, self.num_frames_to_discriminate, self.gaad_condition_dim); sequence_condition_disc = torch.mean(frame_conditions_reshaped_disc, dim=1).to(dtype_in)
        
        features = frames_to_process.permute(0, 2, 1, 3, 4).to(dtype_in) # (B, C, N_disc_frames, H, W)
        for block_module in self.feature_extractor_blocks:
            features = block_module['conv'](features); features = block_module['norm'](features)
            if 'film' in block_module and block_module['film'] is not None and sequence_condition_disc is not None: features = block_module['film'](features, sequence_condition_disc)
            features = block_module['activation'](features)
        features = self.adaptive_pool(features); features_flat = features.view(B, -1)
        logits = self.final_fc_layers(features_flat)
        return logits.to(dtype_in)


# =====================================================================
# VAE-GAN Model (WuBuGAADHybridGenNet) - Updated
# =====================================================================

class WuBuGAADHybridGenNet(nn.Module):
    """
    Combines Encoder and Generator for VAE-GAN. Supports DFT+DCT features for appearance.
    Refined for clarity in bbox and feature flow.
    """
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_appearance_config: Dict,
                 gaad_motion_config: Optional[Dict], wubu_s_config: Dict, wubu_t_config: Optional[Dict],
                 wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_appearance_config = gaad_appearance_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV03.MainNet") # Consistent logger name

        self.latent_dim = args.latent_dim
        self.use_dft_features_appearance = args.use_dft_features_appearance
        self.use_dct_features_appearance = args.use_dct_features_appearance

        # --- Initialize Components ---
        self.encoder = RegionalVAEEncoder(args, video_config, gaad_appearance_config, wubu_s_config, self.latent_dim)
        
        self.motion_encoder: Optional[RegionalHyperbolicMotionEncoder] = None
        if args.use_wubu_motion_branch:
            # Assuming RegionalHyperbolicMotionEncoder is defined and has an 'enabled' attribute
            temp_motion_encoder = RegionalHyperbolicMotionEncoder(args, video_config, gaad_motion_config, wubu_m_config)
            if temp_motion_encoder.enabled:
                self.motion_encoder = temp_motion_encoder
                self.logger.info("Motion Encoder Branch Activated.")
            else:
                self.logger.warning("Motion branch requested but disabled in sub-module or due to missing dependencies. Forcing args.use_wubu_motion_branch to False for this run.")
                # Update args to reflect that motion branch is effectively off if sub-module is disabled
                # This is important because other parts of the model (like RegionalVAEEncoder)
                # might adjust their input dimensions based on args.use_wubu_motion_branch.
                # This needs careful handling: either the encoder itself checks motion_encoder.enabled,
                # or we ensure args reflects the true state.
                # For now, let's assume RegionalVAEEncoder will handle a None motion_features input correctly.
                # A more robust way is for the encoder to take self.motion_encoder as an argument.
                # Or, as done in your original code, args.use_wubu_motion_branch can be set to False.
                # However, modifying args post-init can be tricky if other components already read it.
                # It's cleaner if components can handle a disabled sub-component gracefully.
                # Let's assume the encoder's forward method handles motion_features being None.

        self.generator = RegionalGeneratorDecoder(args, video_config, gaad_appearance_config, self.latent_dim)

        self.apply(init_weights_general) # Assuming init_weights_general is defined
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"WuBuGAADHybridGenNet Initialized (DFT:{args.use_dft_features_appearance}, DCT:{args.use_dct_features_appearance}, Motion:{args.use_wubu_motion_branch and self.motion_encoder is not None}): {param_count:,} params.")

    def encode(self, frames_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Output: mu, logvar, gaad_bboxes_for_all_input_frames, target_dft_features_all_input, target_dct_features_all_input
        # The bboxes and target features returned by encode should correspond to *all* frames_pixels passed in.
        
        motion_features_for_encoder: Optional[torch.Tensor] = None
        # motion_bboxes_from_encoder: Optional[torch.Tensor] = None # Not directly used by this class's forward

        if self.motion_encoder is not None and self.motion_encoder.enabled:
            # frames_pixels is (B, N_total_sample_frames, C, H, W)
            motion_output_tuple = self.motion_encoder(frames_pixels)
            if motion_output_tuple is not None:
                motion_features_for_encoder, _ = motion_output_tuple # Second item is motion_bboxes
                # Ensure motion_features_for_encoder is correctly shaped for the encoder if needed
                # E.g., (B, N_pairs, N_motion_reg, D_motion)
        
        # The encoder's forward should handle if motion_features_for_encoder is None
        mu, logvar, gaad_bboxes_all_input_f, target_dft_all_input_f, target_dct_all_input_f = self.encoder(frames_pixels, motion_features_for_encoder)
        
        return mu, logvar, gaad_bboxes_all_input_f, target_dft_all_input_f, target_dct_all_input_f

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Ensure eps is on the same device and dtype as std
        return mu + eps * std

    def decode(self, z: torch.Tensor, gaad_bboxes_for_decode: torch.Tensor
              ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Output: recon_pixel_frames (Optional), recon_dft_coeffs (Optional), recon_dct_coeffs (Optional)
        # gaad_bboxes_for_decode should be (B, num_predict_frames, num_regions, 4)
        return self.generator(z, gaad_bboxes_for_decode)

    def forward(self, frames_pixels: torch.Tensor # Expected shape (B, N_total_sample_frames, C, H, W)
               ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], # Reconstructions
                          torch.Tensor, torch.Tensor, # mu, logvar
                          torch.Tensor, # bboxes_used_for_decoding
                          Optional[torch.Tensor], Optional[torch.Tensor]]: # Target spectral features for loss
        """
        Full forward pass for training.
        Output Tuple:
            - recon_pixel_frames_gen (Optional[Tensor]): (B, N_pred, C, H, W) or None
            - recon_dft_coeffs_gen (Optional[Tensor]): (B, N_pred, N_reg, D_dft_flat_per_reg) or similar, or None
            - recon_dct_coeffs_gen (Optional[Tensor]): (B, N_pred, N_reg, D_dct_flat_per_reg) or similar, or None
            - mu (Tensor): (B, latent_dim)
            - logvar (Tensor): (B, latent_dim)
            - bboxes_used_for_decoding (Tensor): (B, N_pred, N_reg, 4) - BBoxes corresponding to the frames the decoder generated.
            - target_dft_features_for_loss (Optional[Tensor]): (B, N_pred, N_reg, D_dft_flat_per_reg) or None
            - target_dct_features_for_loss (Optional[Tensor]): (B, N_pred, N_reg, D_dct_flat_per_reg) or None
        """
        B, N_total_sample_frames, _, _, _ = frames_pixels.shape
        device = frames_pixels.device
        dtype_model = next(self.parameters()).dtype if hasattr(self, 'parameters') and next(self.parameters(), None) is not None else frames_pixels.dtype


        # 1. Encode the entire input sequence to get latent parameters and TARGET spectral features
        #    gaad_bboxes_all_input_frames is (B, N_total_sample_frames, num_app_regions, 4)
        #    target_dft_all_input is (B, N_total_sample_frames, num_app_regions, D_dft_flat)
        mu, logvar, gaad_bboxes_all_input_frames, target_dft_all_input, target_dct_all_input = self.encode(frames_pixels)
        
        # 2. Sample from the latent space
        z = self.reparameterize(mu, logvar)

        # 3. Determine which GAAD bounding boxes to pass to the decoder/generator
        #    The generator predicts `num_predict_frames`.
        #    These frames are typically those *after* `num_input_frames` from the input sequence.
        num_input_frames_conditioning = self.video_config.get("num_input_frames", 0)
        num_frames_to_predict_by_gen = self.video_config.get("num_predict_frames", 1)

        # Ensure we have enough frames in the input sequence for the desired operation
        if N_total_sample_frames < num_input_frames_conditioning + num_frames_to_predict_by_gen:
            self.logger.error(f"MainNet Fwd: Input sequence length ({N_total_sample_frames}) is less than "
                              f"num_input_frames ({num_input_frames_conditioning}) + num_predict_frames ({num_frames_to_predict_by_gen}). "
                              f"Cannot select appropriate bboxes/targets for decoder/loss.")
            # Handle error: maybe return Nones or raise exception
            # For now, let's try to proceed with what we have, but this is a config issue.
            # This indicates a mismatch between dataset sampling and model config.
            # If num_input_frames_conditioning is 0, we predict all N_total_sample_frames.
            actual_predict_start_idx = num_input_frames_conditioning if N_total_sample_frames > num_input_frames_conditioning else 0
            actual_num_frames_for_decoder = min(num_frames_to_predict_by_gen, N_total_sample_frames - actual_predict_start_idx)
            if actual_num_frames_for_decoder <= 0:
                raise ValueError(f"MainNet Fwd: Misconfiguration or insufficient input frames. Effective frames for decoder is {actual_num_frames_for_decoder}.")
            self.logger.warning(f"  Adjusting to decode {actual_num_frames_for_decoder} frames starting from index {actual_predict_start_idx} of input bboxes/targets.")
        else:
            actual_predict_start_idx = num_input_frames_conditioning
            actual_num_frames_for_decoder = num_frames_to_predict_by_gen
            
        # Slice the GAAD bboxes from the *encoder's output* that correspond to the frames
        # the *decoder* will generate.
        # gaad_bboxes_all_input_frames shape: (B, N_total_sample_frames, num_app_regions, 4)
        bboxes_for_decoding_input = gaad_bboxes_all_input_frames[
            :,
            actual_predict_start_idx : actual_predict_start_idx + actual_num_frames_for_decoder,
            ...
        ]

        # Ensure the selected bboxes match the number of frames the generator is configured to predict.
        # The generator itself is built expecting to produce `self.num_predict_frames`.
        # If actual_num_frames_for_decoder is less, we need to give the generator bboxes for its full prediction window,
        # potentially by padding or by ensuring the config is consistent.
        # For now, assume RegionalGeneratorDecoder is robust to `gaad_bboxes_for_decode` having fewer frames
        # than its internal `self.num_predict_frames` if it uses that for its ConvTranspose3d output shape.
        # However, it's better if `bboxes_for_decoding_input` always matches `generator.num_predict_frames`.
        # The generator's `self.num_predict_frames` *should* be `actual_num_frames_for_decoder`.
        # If `self.video_config["num_predict_frames"]` is the single source of truth for generator output length,
        # then `actual_num_frames_for_decoder` must match it.
        if bboxes_for_decoding_input.shape[1] != self.generator.num_predict_frames:
            self.logger.warning_once(f"MainNet Fwd: Shape mismatch for decoder bboxes. "
                                f"Selected {bboxes_for_decoding_input.shape[1]} frames of bboxes, "
                                f"but generator.num_predict_frames is {self.generator.num_predict_frames}. "
                                f"This might occur if N_total_sample_frames was too short. "
                                f"Attempting to use available bboxes; decoder might misbehave if temporal dim of bboxes doesn't match its output.")
            # If fewer bboxes than generator expects, could pad. If more, could truncate.
            # This situation ideally indicates a dataset/config mismatch.
            # For now, pass what we sliced. Generator needs to be robust or this is an error.
            if bboxes_for_decoding_input.shape[1] < self.generator.num_predict_frames and bboxes_for_decoding_input.shape[1] > 0:
                num_to_pad_bbox = self.generator.num_predict_frames - bboxes_for_decoding_input.shape[1]
                padding_slice_bbox = bboxes_for_decoding_input[:, -1:, ...].repeat(1, num_to_pad_bbox, 1, 1)
                bboxes_for_decoding_input = torch.cat([bboxes_for_decoding_input, padding_slice_bbox], dim=1)
                self.logger.warning_once(f"  Padded decoder bboxes to {bboxes_for_decoding_input.shape[1]} frames.")
            elif bboxes_for_decoding_input.shape[1] > self.generator.num_predict_frames:
                bboxes_for_decoding_input = bboxes_for_decoding_input[:, :self.generator.num_predict_frames, ...]
                self.logger.warning_once(f"  Truncated decoder bboxes to {bboxes_for_decoding_input.shape[1]} frames.")


        # 4. Decode the latent sample `z` using the selected bboxes
        #    The decoder should generate `self.generator.num_predict_frames` worth of content.
        recon_pixel_frames_gen, recon_dft_coeffs_gen, recon_dct_coeffs_gen = self.decode(z, bboxes_for_decoding_input)

        # 5. Select the corresponding TARGET spectral features for the reconstruction loss.
        #    These targets must align with the frames the generator *attempted* to reconstruct.
        #    And their number of frames must match the generator's output frames.
        target_dft_for_loss: Optional[torch.Tensor] = None
        if target_dft_all_input is not None:
            target_dft_for_loss = target_dft_all_input[
                :,
                actual_predict_start_idx : actual_predict_start_idx + actual_num_frames_for_decoder,
                ...
            ]
            # If generator output frame count differs from actual_num_frames_for_decoder (due to fixed generator.num_predict_frames)
            # and we padded/truncated bboxes_for_decoding_input, we need to adjust target_dft_for_loss similarly
            # to match recon_dft_coeffs_gen.shape[1]
            if recon_dft_coeffs_gen is not None and target_dft_for_loss.shape[1] != recon_dft_coeffs_gen.shape[1]:
                self.logger.warning_once(f"MainNet Fwd: Target DFT frames ({target_dft_for_loss.shape[1]}) "
                                     f"!= Recon DFT frames ({recon_dft_coeffs_gen.shape[1]}). Adjusting target for loss.")
                if target_dft_for_loss.shape[1] < recon_dft_coeffs_gen.shape[1] and target_dft_for_loss.shape[1] > 0: # Pad target
                    num_pad_target_dft = recon_dft_coeffs_gen.shape[1] - target_dft_for_loss.shape[1]
                    pad_slice_target_dft = target_dft_for_loss[:, -1:, ...].repeat(1, num_pad_target_dft, 1, 1)
                    target_dft_for_loss = torch.cat([target_dft_for_loss, pad_slice_target_dft], dim=1)
                elif target_dft_for_loss.shape[1] > recon_dft_coeffs_gen.shape[1]: # Truncate target
                    target_dft_for_loss = target_dft_for_loss[:, :recon_dft_coeffs_gen.shape[1], ...]


        target_dct_for_loss: Optional[torch.Tensor] = None
        if target_dct_all_input is not None:
            target_dct_for_loss = target_dct_all_input[
                :,
                actual_predict_start_idx : actual_predict_start_idx + actual_num_frames_for_decoder,
                ...
            ]
            if recon_dct_coeffs_gen is not None and target_dct_for_loss.shape[1] != recon_dct_coeffs_gen.shape[1]:
                self.logger.warning_once(f"MainNet Fwd: Target DCT frames ({target_dct_for_loss.shape[1]}) "
                                     f"!= Recon DCT frames ({recon_dct_coeffs_gen.shape[1]}). Adjusting target for loss.")
                if target_dct_for_loss.shape[1] < recon_dct_coeffs_gen.shape[1] and target_dct_for_loss.shape[1] > 0: # Pad target
                    num_pad_target_dct = recon_dct_coeffs_gen.shape[1] - target_dct_for_loss.shape[1]
                    pad_slice_target_dct = target_dct_for_loss[:, -1:, ...].repeat(1, num_pad_target_dct, 1, 1)
                    target_dct_for_loss = torch.cat([target_dct_for_loss, pad_slice_target_dct], dim=1)
                elif target_dct_for_loss.shape[1] > recon_dct_coeffs_gen.shape[1]: # Truncate target
                    target_dct_for_loss = target_dct_for_loss[:, :recon_dct_coeffs_gen.shape[1], ...]


        # bboxes_used_by_decoder refers to the bboxes that align with the *generator's output length*
        # which is `self.generator.num_predict_frames`. This is `bboxes_for_decoding_input` after any padding/truncation.
        return (
            recon_pixel_frames_gen, recon_dft_coeffs_gen, recon_dct_coeffs_gen,
            mu, logvar,
            bboxes_for_decoding_input, # These are the bboxes corresponding to G's output
            target_dft_for_loss, target_dct_for_loss
        )

# =====================================================================

class RegionalHyperbolicMotionEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, video_config: Dict, gaad_motion_config: Optional[Dict], wubu_m_config: Optional[Dict]):
        super().__init__()
        self.args = args
        self.video_config = video_config
        self.gaad_motion_config = gaad_motion_config
        self.wubu_m_config = wubu_m_config
        self.logger = logging.getLogger("WuBuGAADHybridGenV03.EncoderM")
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

        self.image_size = args.image_h_w_tuple
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
                    temp_level_m = HyperbolicWuBuNestingLevel(last_level_idx, self.wubu_m_final_hyp_dim, wubu_m_config, wubu_m_config['initial_curvatures'][last_level_idx])
                    self.wubu_m_final_curvature = temp_level_m.get_current_curvature_scalar()
                    del temp_level_m
                    self.logger.info(f"WuBu-M final C estimated: {self.wubu_m_final_curvature:.3f}")
                except IndexError:
                    self.logger.error(f"Index error accessing WuBu-M config for level {last_level_idx}. Defaulting C=1.0.")
                    self.wubu_m_final_curvature = 1.0
            if self.wubu_m_final_hyp_dim > 0 and hasattr(self.wubu_m, 'output_tangent_projection'):
                 for p in self.wubu_m.output_tangent_projection.parameters(): # type: ignore
                    setattr(p, 'manifold', PoincareBall(self.wubu_m_final_curvature))
        elif self.wubu_m_config is not None and self.flow_stats_dim == 0:
             self.logger.warning("WuBu-M configured, but flow_stats_dim=0. WuBu-M will be ineffective or bypassed.")
             self.wubu_m = None # type: ignore
             self.wubu_m_final_hyp_dim=0
             self.wubu_m_final_curvature=1.0
        else: 
             self.logger.error("MotionEncoder: wubu_m_config is None, cannot initialize WuBu-M. Disabling motion branch.")
             self.wubu_m = None # type: ignore
             self.wubu_m_final_hyp_dim=0
             self.wubu_m_final_curvature=1.0
             self.enabled = False 

        self.apply(init_weights_general)

    def _get_motion_gaad_bboxes(self, analysis_map: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # analysis_map is expected to be (B_eff, 1, H, W) e.g., flow_magnitude
        B_eff, _, H, W = analysis_map.shape
        all_batch_bboxes_list = [] # Use a list to collect tensors for each batch item

        for i in range(B_eff):
            # Get the single channel (H, W) slice for the current batch item
            current_analysis_map_slice = analysis_map[i, 0, :, :] # Shape (H, W)
            
            frame_dims_tuple = (W, H) # Consistent (Width, Height) for GAAD functions
            max_w_scalar, max_h_scalar = float(W), float(H)
            frame_bboxes_for_item_list = [] # Bboxes for current item in batch

            if self.motion_decomposition_type == "hybrid":
                num_subdivide = self.num_motion_regions // 2
                num_spiral = self.num_motion_regions - num_subdivide
                
                if num_subdivide > 0:
                    subdivide_bboxes = golden_subdivide_rect_fixed_n_motion_aware(
                        frame_dims_tuple, current_analysis_map_slice, num_subdivide,
                        device, dtype, self.gaad_min_size_px,
                        prioritize_high_energy=self.args.motion_gaad_gas_prioritize_high_energy # New arg
                    )
                    frame_bboxes_for_item_list.append(subdivide_bboxes)
                
                if num_spiral > 0:
                    spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n_motion_aware(
                        frame_dims_tuple, current_analysis_map_slice, num_spiral, device, dtype,
                        num_spiral_arms_per_hotspot=self.args.motion_gaad_psp_arms_per_hotspot, # New arg
                        points_per_arm=self.args.motion_gaad_psp_pts_per_arm,                   # New arg
                        motion_scale_influence=self.args.motion_gaad_psp_motion_scale_influence # New arg
                    )
                    # Convert centers and scales to [x1, y1, x2, y2] bboxes
                    patch_base_size_for_spiral = min(frame_dims_tuple) # Or some other reference
                    patch_hs = float(patch_base_size_for_spiral) * spiral_scales[:,0] / 2.0 # Half-height
                    patch_ws = patch_hs # Assume square patches from spiral centers for simplicity
                    
                    val_x1 = spiral_centers[:,0] - patch_ws
                    val_y1 = spiral_centers[:,1] - patch_hs
                    val_x2 = spiral_centers[:,0] + patch_ws
                    val_y2 = spiral_centers[:,1] + patch_hs
                    
                    spiral_bboxes_current = torch.zeros(num_spiral, 4, device=device, dtype=dtype)
                    spiral_bboxes_current[:,0] = torch.clamp(val_x1, min=0.0, max=max_w_scalar - EPS)
                    spiral_bboxes_current[:,1] = torch.clamp(val_y1, min=0.0, max=max_h_scalar - EPS)
                    min_for_x2 = spiral_bboxes_current[:,0] + EPS # Ensure x2 > x1
                    spiral_bboxes_current[:,2] = torch.clamp(val_x2, max=max_w_scalar)
                    spiral_bboxes_current[:,2] = torch.maximum(spiral_bboxes_current[:,2], min_for_x2)
                    min_for_y2 = spiral_bboxes_current[:,1] + EPS # Ensure y2 > y1
                    spiral_bboxes_current[:,3] = torch.clamp(val_y2, max=max_h_scalar)
                    spiral_bboxes_current[:,3] = torch.maximum(spiral_bboxes_current[:,3], min_for_y2)
                    frame_bboxes_for_item_list.append(spiral_bboxes_current)

            elif self.motion_decomposition_type == "spiral":
                spiral_centers, spiral_scales = phi_spiral_patch_centers_fixed_n_motion_aware(
                    frame_dims_tuple, current_analysis_map_slice, self.num_motion_regions, device, dtype,
                    num_spiral_arms_per_hotspot=self.args.motion_gaad_psp_arms_per_hotspot,
                    points_per_arm=self.args.motion_gaad_psp_pts_per_arm,
                    motion_scale_influence=self.args.motion_gaad_psp_motion_scale_influence
                )
                patch_base_size_for_spiral = min(frame_dims_tuple)
                patch_hs = float(patch_base_size_for_spiral) * spiral_scales[:,0] / 2.0
                patch_ws = patch_hs
                val_x1 = spiral_centers[:,0] - patch_ws; val_y1 = spiral_centers[:,1] - patch_hs
                val_x2 = spiral_centers[:,0] + patch_ws; val_y2 = spiral_centers[:,1] + patch_hs
                spiral_bboxes_final = torch.zeros(self.num_motion_regions, 4, device=device, dtype=dtype)
                spiral_bboxes_final[:,0]=torch.clamp(val_x1,min=0.0,max=max_w_scalar-EPS); spiral_bboxes_final[:,1]=torch.clamp(val_y1,min=0.0,max=max_h_scalar-EPS)
                min_for_x2=spiral_bboxes_final[:,0]+EPS; spiral_bboxes_final[:,2]=torch.clamp(val_x2,max=max_w_scalar); spiral_bboxes_final[:,2]=torch.maximum(spiral_bboxes_final[:,2],min_for_x2)
                min_for_y2=spiral_bboxes_final[:,1]+EPS; spiral_bboxes_final[:,3]=torch.clamp(val_y2,max=max_h_scalar); spiral_bboxes_final[:,3]=torch.maximum(spiral_bboxes_final[:,3],min_for_y2)
                frame_bboxes_for_item_list.append(spiral_bboxes_final)
                
            else: # "subdivide"
                subdivide_bboxes = golden_subdivide_rect_fixed_n_motion_aware(
                    frame_dims_tuple, current_analysis_map_slice, self.num_motion_regions,
                    device, dtype, self.gaad_min_size_px,
                    prioritize_high_energy=self.args.motion_gaad_gas_prioritize_high_energy
                )
                frame_bboxes_for_item_list.append(subdivide_bboxes)

            # Concatenate bboxes if hybrid produced multiple sets
            if frame_bboxes_for_item_list:
                single_item_bboxes_concatenated = torch.cat(frame_bboxes_for_item_list, dim=0)
            elif self.num_motion_regions > 0 : # Fallback if list is empty but regions expected
                 self.logger.warning_once(f"Motion GAAD for item {i} produced no bboxes. Defaulting to full frame region(s).")
                 single_item_bboxes_concatenated = torch.tensor([[0.0, 0.0, max_w_scalar, max_h_scalar]] * self.num_motion_regions, dtype=dtype, device=device)
            else: # No regions expected, empty tensor
                 single_item_bboxes_concatenated = torch.empty((0,4), dtype=dtype, device=device)


            # Ensure correct number of regions for this item (padding/truncating)
            if single_item_bboxes_concatenated.shape[0] < self.num_motion_regions:
                num_to_pad = self.num_motion_regions - single_item_bboxes_concatenated.shape[0]
                padding_box_coords = single_item_bboxes_concatenated[-1:].clone() if single_item_bboxes_concatenated.shape[0] > 0 else torch.tensor([[0.0,0.0,max_w_scalar,max_h_scalar]], dtype=dtype, device=device)
                padding_tensor = padding_box_coords.repeat(num_to_pad, 1)
                final_item_bboxes = torch.cat([single_item_bboxes_concatenated, padding_tensor], dim=0)
            elif single_item_bboxes_concatenated.shape[0] > self.num_motion_regions:
                # If hybrid produced too many, select based on some criteria (e.g. smallest area first, or random)
                # For now, just truncate. A more sophisticated selection could be added.
                # Example: if from GAS and PSP, maybe interleave or take best from each.
                # Current cat just appends. Sorting by area/energy before truncate would be better.
                final_item_bboxes = single_item_bboxes_concatenated[:self.num_motion_regions]
            else:
                final_item_bboxes = single_item_bboxes_concatenated
            
            all_batch_bboxes_list.append(final_item_bboxes)

        if not all_batch_bboxes_list: # Should not happen if B_eff > 0
             return torch.empty((B_eff, self.num_motion_regions if self.num_motion_regions > 0 else 0, 4), device=device, dtype=dtype)
             
        return torch.stack(all_batch_bboxes_list) # Shape (B_eff, num_motion_regions, 4)
        
    def _extract_flow_statistics(self, flow_field: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        B, _, H, W = flow_field.shape; N_reg = bboxes.shape[1]; device = flow_field.device; dtype = flow_field.dtype
        all_stats = torch.zeros(B, N_reg, self.flow_stats_dim, device=device, dtype=dtype)
        for b in range(B):
            for r in range(N_reg):
                x1, y1, x2, y2 = bboxes[b, r].round().int().tolist(); x1_c = max(0, x1); y1_c = max(0, y1); x2_c = min(W, x2); y2_c = min(H, y2)
                if x1_c >= x2_c or y1_c >= y2_c: continue 
                region_flow = flow_field[b, :, y1_c:y2_c, x1_c:x2_c]; flow_dx = region_flow[0, ...].flatten(); flow_dy = region_flow[1, ...].flatten()
                if flow_dx.numel() == 0: continue
                stat_idx = 0; magnitudes = torch.sqrt(flow_dx**2 + flow_dy**2)
                if 'mag_mean' in self.flow_stats_components: all_stats[b, r, stat_idx] = torch.mean(magnitudes); stat_idx += 1
                if 'mag_std' in self.flow_stats_components: all_stats[b, r, stat_idx] = torch.std(magnitudes) if magnitudes.numel() > 1 else 0.0; stat_idx += 1
                angles = torch.atan2(flow_dy, flow_dx)
                if 'angle_mean' in self.flow_stats_components: all_stats[b,r,stat_idx] = torch.mean(torch.cos(angles)); stat_idx+=1; all_stats[b,r,stat_idx] = torch.mean(torch.sin(angles)); stat_idx+=1
                if 'angle_std' in self.flow_stats_components: angle_std = torch.std(angles) if angles.numel() > 1 else 0.0; all_stats[b,r,stat_idx] = angle_std if torch.isfinite(angle_std) else 0.0; stat_idx+=1
        return all_stats

    def forward(self, frames_pixels: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.enabled or self.flow_net is None : return None
        B, N_frames, C, H, W = frames_pixels.shape; device = frames_pixels.device; original_dtype = frames_pixels.dtype
        compute_dtype = next(self.parameters(), torch.tensor(0.0, device=device)).dtype
        if N_frames < 2: self.logger.debug(f"Not enough frames ({N_frames}) for optical flow. Skipping motion branch."); return None
        num_pairs = N_frames - 1; all_motion_features_list = []; all_motion_bboxes_list = []
        flow_context = torch.no_grad() if self.args.freeze_flow_net else contextlib.nullcontext()
        with flow_context:
            for i in range(num_pairs):
                frame_t_orig_norm = frames_pixels[:, i+1, ...]; frame_t_minus_1_orig_norm = frames_pixels[:, i, ...]
                frame_t_raft = ((frame_t_orig_norm * 0.5 + 0.5) * 255.0).to(torch.float32)
                frame_t_minus_1_raft = ((frame_t_minus_1_orig_norm * 0.5 + 0.5) * 255.0).to(torch.float32)
                try:
                    self.flow_net = self.flow_net.to(device) # type: ignore [union-attr]
                    flow_predictions = self.flow_net(frame_t_minus_1_raft, frame_t_raft) # type: ignore [operator]
                    flow_field = flow_predictions[-1].to(compute_dtype)
                except Exception as e_flow: self.logger.error(f"Optical flow computation failed for pair {i}: {e_flow}", exc_info=True); return None
                flow_magnitude = torch.sqrt(flow_field[:, 0:1, :, :]**2 + flow_field[:, 1:2, :, :]**2)
                motion_gaad_bboxes_batch = self._get_motion_gaad_bboxes(flow_magnitude, device, compute_dtype)
                if self.flow_stats_dim > 0: flow_stats = self._extract_flow_statistics(flow_field, motion_gaad_bboxes_batch); flow_stats_flat = flow_stats.reshape(B * self.num_motion_regions, self.flow_stats_dim); initial_motion_tangent_vectors_flat = self.motion_feature_embed(flow_stats_flat)
                else: initial_motion_tangent_vectors_flat = torch.zeros(B * self.num_motion_regions, self.args.encoder_initial_tangent_dim, device=device, dtype=compute_dtype)
                if self.wubu_m is None: motion_features_pair_flat = initial_motion_tangent_vectors_flat
                else: motion_features_pair_flat = self.wubu_m(initial_motion_tangent_vectors_flat)
                motion_features_pair = motion_features_pair_flat.reshape(B, self.num_motion_regions, -1)
                all_motion_features_list.append(motion_features_pair); all_motion_bboxes_list.append(motion_gaad_bboxes_batch)
        if not all_motion_features_list: self.logger.warning("No motion features were generated."); return None
        final_motion_features = torch.stack(all_motion_features_list, dim=1).to(original_dtype)
        final_motion_bboxes = torch.stack(all_motion_bboxes_list, dim=1).to(original_dtype)
        return final_motion_features, final_motion_bboxes

# =====================================================================
# Dataset (Unchanged from v0.2)
# =====================================================================
class VideoFrameDataset(Dataset): 
    def __init__(self, video_path: str, num_frames_total: int, image_size: Tuple[int, int], frame_skip: int = 1, data_fraction: float = 1.0):
        super().__init__(); self.video_path = video_path; self.num_frames_total = num_frames_total; self.image_size = image_size; self.frame_skip = frame_skip; current_logger=logging.getLogger("WuBuGAADHybridGenV03.Dataset")
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
                except Exception as e: logging.getLogger("WuBuGAADHybridGenV03.Dataset").error(f"Error transforming frame {actual_frame_idx_in_ram} sample {idx}: {e}", exc_info=True); raise e
            else: logging.getLogger("WuBuGAADHybridGenV03.Dataset").error(f"Frame index {actual_frame_idx_in_ram} out of bounds (total: {self.num_disk_frames}). Sample: {idx}"); raise IndexError("Frame index out of bounds.")
        if len(frames_for_sample) != self.num_frames_total: logging.getLogger("WuBuGAADHybridGenV03.Dataset").error(f"Loaded {len(frames_for_sample)} frames, expected {self.num_frames_total} for sample {idx}"); raise ValueError("Incorrect number of frames loaded for sample.")
        return torch.stack(frames_for_sample)

# =====================================================================
# VAE-GAN Trainer (Major Updates for v0.3)
# =====================================================================
class HybridTrainer:
    def __init__(self,
                 model: "WuBuGAADHybridGenNet",
                 discriminator_primary: "VideoDiscriminatorWrapper",
                 discriminator_alternative: "VideoDiscriminatorWrapper",
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 args: argparse.Namespace,
                 rank: int,
                 world_size: int,
                 ddp_active: bool):

        self.model = model
        self.discriminator_primary_obj = discriminator_primary
        self.discriminator_alternative_obj = discriminator_alternative
        
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.ddp_active = ddp_active
        self.am_main_process = (rank == 0)
        self.logger = logging.getLogger("WuBuGAADHybridGenV03.Trainer")
        
        self.m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        
        self.video_config = getattr(self.m_ref, 'video_config', {})
        self.gaad_appearance_config = getattr(self.m_ref, 'gaad_appearance_config', {})

        # --- Loss Lambdas ---
        self.lambda_recon_dft = args.lambda_recon_dft
        self.lambda_recon_dct = args.lambda_recon_dct
        self.lambda_kl_base = args.lambda_kl 
        self.lambda_gan_base = args.lambda_gan 
        self.lambda_kl = self.lambda_kl_base 
        self.lambda_gan = self.lambda_gan_base

        # --- Optimizers and Q-Controllers ---
        q_cfg_gen_shared = DEFAULT_CONFIG_QLEARN_HYBRID.copy() if args.q_controller_enabled else None
        if q_cfg_gen_shared: q_cfg_gen_shared.update(getattr(args, 'q_config_gen', {}))

        m_ref_for_q_suffix_name = self.m_ref.__class__.__name__ if hasattr(self.m_ref, '__class__') else 'Model'
        d_pri_ref_for_q_suffix_name = self.discriminator_primary_obj.architecture_variant if hasattr(self.discriminator_primary_obj, 'architecture_variant') else 'DiscPri'
        d_alt_ref_for_q_suffix_name = self.discriminator_alternative_obj.architecture_variant if hasattr(self.discriminator_alternative_obj, 'architecture_variant') else 'DiscAlt'


        self.optimizer_enc_gen: RiemannianEnhancedSGD = RiemannianEnhancedSGD(
            self.model.parameters(), lr=self.args.learning_rate_gen,
            q_learning_config=q_cfg_gen_shared.copy() if q_cfg_gen_shared else None,
            max_grad_norm_risgd=self.args.risgd_max_grad_norm,
            optimizer_type="generator",
            q_logger_suffix=f"Gen.{m_ref_for_q_suffix_name}"
        )
        self.q_controller_gen = getattr(self.optimizer_enc_gen, 'q_controller', None)

        q_cfg_disc_shared = DEFAULT_CONFIG_QLEARN_HYBRID.copy() if args.q_controller_enabled else None
        if q_cfg_disc_shared: q_cfg_disc_shared.update(getattr(args, 'q_config_disc', {}))

        self.optimizer_disc_primary = RiemannianEnhancedSGD(
            self.discriminator_primary_obj.parameters(), lr=args.learning_rate_disc,
            q_learning_config=q_cfg_disc_shared.copy() if q_cfg_disc_shared else None,
            max_grad_norm_risgd=args.risgd_max_grad_norm,
            optimizer_type="discriminator",
            q_logger_suffix=f"DiscPri.{d_pri_ref_for_q_suffix_name}"
        )
        self.q_controller_d_primary = getattr(self.optimizer_disc_primary, 'q_controller', None)
        
        lr_disc_alt = getattr(args, 'learning_rate_disc_alt', args.learning_rate_disc)
        self.optimizer_disc_alternative = RiemannianEnhancedSGD(
            self.discriminator_alternative_obj.parameters(), lr=lr_disc_alt,
            q_learning_config=q_cfg_disc_shared.copy() if q_cfg_disc_shared else None,
            max_grad_norm_risgd=args.risgd_max_grad_norm,
            optimizer_type="discriminator",
            q_logger_suffix=f"DiscAlt.{d_alt_ref_for_q_suffix_name}"
        )
        self.q_controller_d_alt = getattr(self.optimizer_disc_alternative, 'q_controller', None)

        self.active_discriminator_key = 'primary' 
        if self.args.enable_heuristic_disc_switching and self.args.initial_disc_type:
            primary_is_pixel_like = "pixels" in self.discriminator_primary_obj.effective_input_type_for_trainer.lower()
            primary_is_feature_like = "features" in self.discriminator_primary_obj.effective_input_type_for_trainer.lower() or \
                                      "spectral" in self.discriminator_primary_obj.effective_input_type_for_trainer.lower()
            alt_is_pixel_like = "pixels" in self.discriminator_alternative_obj.effective_input_type_for_trainer.lower()
            alt_is_feature_like = "features" in self.discriminator_alternative_obj.effective_input_type_for_trainer.lower() or \
                                  "spectral" in self.discriminator_alternative_obj.effective_input_type_for_trainer.lower()

            if self.args.initial_disc_type == 'pixel':
                if primary_is_pixel_like: self.active_discriminator_key = 'primary'
                elif alt_is_pixel_like: self.active_discriminator_key = 'alternative'
            elif self.args.initial_disc_type == 'feature':
                if primary_is_feature_like: self.active_discriminator_key = 'primary'
                elif alt_is_feature_like: self.active_discriminator_key = 'alternative'
        
        self.active_discriminator: nn.Module 
        self.optimizer_disc_active: RiemannianEnhancedSGD
        self.active_disc_effective_trainer_input_type: str
        self.q_controller_d_active: Optional[HAKMEMQController] = None
        self._update_active_discriminator_pointers() 
        
        self.scaler_enc_gen = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.scaler_disc_active = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')

        self.global_step = 0; self.current_epoch = 0
        self.is_val_metric_higher_better = self.args.val_primary_metric in ["avg_val_psnr", "avg_val_ssim"]
        self.best_val_metric_val = -float('inf') if self.is_val_metric_higher_better else float('inf')
        self.last_val_metrics: Dict[str, Any] = {}
        self.prev_interval_metrics_for_lambda_kl_reward: Optional[Dict[str, Union[float, None]]] = None
        if self.am_main_process: os.makedirs(args.checkpoint_dir, exist_ok=True)

        self.lpips_loss_fn = None; self.ssim_metric = None
        if self.am_main_process:
            if self.args.use_lpips_for_verification and LPIPS_AVAILABLE and lpips is not None:
                try: self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device) 
                except Exception as e: self.logger.warning(f"LPIPS init failed: {e}")
            if TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
                try: self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                except Exception as e: self.logger.warning(f"SSIM init failed: {e}")
        
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))
        self.fixed_noise_for_sampling: Optional[torch.Tensor] = None

        self.lambda_kl_update_interval = getattr(args, 'lambda_kl_update_interval', 0)
        self.lambda_kl_q_controller: Optional[HAKMEMQController] = None
        if self.args.q_controller_enabled and self.lambda_kl_update_interval > 0:
            q_cfg_lkl = DEFAULT_CONFIG_QLEARN_HYBRID.copy(); q_cfg_lkl.update(getattr(args, 'q_config_lkl', {}))
            q_cfg_lkl["lambda_kl_scale_options"] = getattr(args, 'q_lkl_scale_options', [0.85, 0.95, 1.0, 1.05, 1.15])
            # Ensure probation step keys are correctly passed if HAKMEMQController expects them
            # These were added to DEFAULT_CONFIG_QLEARN_HYBRID
            q_cfg_lkl['num_probation_steps'] = getattr(args, 'q_lkl_lr_mom_probation_steps', q_cfg_lkl.get('state_history_len', 7) + 3)
            q_cfg_lkl['lkl_num_probation_steps'] = getattr(args, 'q_lkl_action_probation_steps', max(3, q_cfg_lkl.get('lambda_kl_state_history_len', 7) + 2))
            
            self.lambda_kl_q_controller = HAKMEMQController(**q_cfg_lkl, logger_name_suffix="LambdaKL")
            if self.am_main_process: self.logger.info(f"Lambda_KL Q-Control ENABLED. Update interval: {self.lambda_kl_update_interval} global steps.")
            if hasattr(self.lambda_kl_q_controller, 'set_current_lambda_kl'): self.lambda_kl_q_controller.set_current_lambda_kl(self.lambda_kl)

        self.interval_metrics_accum: Dict[str, float] = defaultdict(float)
        self.interval_steps_count = 0
        self.min_lambda_kl_q_control = getattr(args, 'min_lambda_kl_q_control', 1e-7)
        self.max_lambda_kl_q_control = getattr(args, 'max_lambda_kl_q_control', 0.5)

        self.enable_heuristic_interventions = getattr(args, 'enable_heuristic_interventions', False) 
        self.enable_heuristic_disc_switching = getattr(args, 'enable_heuristic_disc_switching', False)
        self.heuristic_check_interval = args.heuristic_check_interval if args.heuristic_check_interval is not None else \
                                        (args.disc_switch_check_interval if self.enable_heuristic_disc_switching else args.log_interval)
        self.disc_switch_min_steps_between = args.disc_switch_min_steps_between
        self.disc_switch_problem_state_count_thresh = args.disc_switch_problem_state_count_thresh
        self.steps_since_last_d_switch = 0
        self.consecutive_trigger_primary_to_alt_count = 0 
        self.consecutive_trigger_alt_to_primary_count = 0
        self.consecutive_heuristic_trigger_counts: Dict[str, int] = defaultdict(int)
        self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS = getattr(args, 'heuristic_short_term_history_len', 7)
        self.avg_g_recon_hist_for_stagnation = deque(maxlen=self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS)
        self.q_data_derived_g_recon_hist = deque(maxlen=self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS)
        self.rec_features_stagnant = False 

        self.D_STRONG_THRESH = getattr(args, 'heuristic_d_strong_thresh', 0.25) 
        self.D_WEAK_THRESH = getattr(args, 'heuristic_d_weak_thresh', 1.0)    
        self.D_VERY_WEAK_THRESH = getattr(args, 'heuristic_d_very_weak_thresh', 1.8) 
        self.G_STALLED_THRESH = getattr(args, 'heuristic_g_stalled_thresh', 1.5) 
        self.G_WINNING_THRESH = getattr(args, 'heuristic_g_winning_thresh', 0.2) 
        self.G_VERY_MUCH_WINNING_THRESH = getattr(args, 'heuristic_g_very_much_winning_thresh', 0.05)
        self.KL_HIGH_THRESH = getattr(args, 'heuristic_kl_high_thresh', 25.0) 
        self.RECON_STAGNATION_IMPROVEMENT_THRESH_REL = getattr(args, 'heuristic_recon_stagnation_improvement_thresh_rel', 0.001)
        self.TARGET_GOOD_RECON_THRESH_HEURISTIC = getattr(args, 'target_good_recon_thresh_heuristic_video', 0.015) 
        self.Q_REWARD_STAGNATION_THRESH = getattr(args, 'heuristic_q_reward_stagnation_thresh', -0.25)
        self.HEURISTIC_TRIGGER_COUNT_THRESH = getattr(args, 'heuristic_trigger_count_thresh', 2) 

        self.heuristic_vae_feature_match_active = False
        self.heuristic_penalize_g_easy_win_active = False
        self.heuristic_boost_active_d_lr_active = False
        self.heuristic_force_d_q_explore_active = False # Initialize this attribute
        self.heuristic_override_lambda_recon_factor = 1.0
        self.heuristic_override_lambda_kl_factor = 1.0 
        self.heuristic_override_lambda_gan_factor = 1.0 
        self.lambda_feat_match_heuristic = getattr(args, 'lambda_feat_match_heuristic_video', 0.1) 
        self.lambda_g_easy_win_penalty_heuristic = getattr(args, 'lambda_g_easy_win_penalty_heuristic_video', 1.0)
        self.heuristic_active_d_lr_boost_factor = getattr(args, 'heuristic_active_d_lr_boost_factor', 1.8)
        self.heuristic_d_q_explore_boost_epsilon = getattr(args, 'heuristic_d_q_explore_boost_epsilon', 0.7)
        self.heuristic_d_q_explore_duration = getattr(args, 'heuristic_d_q_explore_duration', 10)
        
        if self.am_main_process:
             self.logger.info(f"HybridTrainer initialized. Initial Active D: '{self.active_discriminator_key}' (Effective Input: '{self.active_disc_effective_trainer_input_type}'). Heuristics {'ENABLED' if self.enable_heuristic_interventions else 'DISABLED'}.")

    def _update_active_discriminator_pointers(self):
        if self.active_discriminator_key == 'primary':
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_effective_trainer_input_type = self.discriminator_primary_obj.effective_input_type_for_trainer 
            self.q_controller_d_active = self.q_controller_d_primary
        elif self.active_discriminator_key == 'alternative':
            self.active_discriminator = self.discriminator_alternative_obj
            self.optimizer_disc_active = self.optimizer_disc_alternative
            self.active_disc_effective_trainer_input_type = self.discriminator_alternative_obj.effective_input_type_for_trainer 
            self.q_controller_d_active = self.q_controller_d_alt
        else:
            self.logger.error(f"Invalid active_discriminator_key: {self.active_discriminator_key}. Defaulting to primary.")
            self.active_discriminator_key = 'primary' 
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_effective_trainer_input_type = self.discriminator_primary_obj.effective_input_type_for_trainer 
            self.q_controller_d_active = self.q_controller_d_primary

        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_arch_variant = getattr(d_ref_active, 'architecture_variant', 'unknown_variant')
        
        if self.am_main_process:
            self.logger.info(f"Active Discriminator is now: '{self.active_discriminator_key}' (Arch Variant: '{active_d_arch_variant}', Effective Trainer Input: '{self.active_disc_effective_trainer_input_type}')")


    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div.mean()


    def _compute_recon_loss(self, 
                            recon_dft: Optional[torch.Tensor], target_dft: Optional[torch.Tensor],
                            recon_dct: Optional[torch.Tensor], target_dct: Optional[torch.Tensor],
                            recon_pixels: Optional[torch.Tensor], target_pixels: Optional[torch.Tensor]
                           ) -> torch.Tensor:
        total_recon_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32) 
        num_losses_counted = 0

        if self.args.use_dft_features_appearance and recon_dft is not None and target_dft is not None:
            recon_dft_for_loss = recon_dft
            if recon_dft.ndim == 7 and target_dft.ndim == 4: # Standard case from current Gen/Enc
                B, N, R, C_dft, two, H, W_coeff = recon_dft.shape
                recon_dft_for_loss = recon_dft.reshape(B, N, R, -1)
            
            if recon_dft_for_loss.shape == target_dft.shape:
                loss_dft = F.mse_loss(recon_dft_for_loss.float(), target_dft.float())
                total_recon_loss += self.args.lambda_recon_dft * loss_dft
                num_losses_counted +=1
            else: 
                if not hasattr(self, '_logged_dft_mismatch_warning'): # Log only once
                    self.logger.warning(f"Recon DFT shape mismatch: Recon effective {recon_dft_for_loss.shape}, Target {target_dft.shape}. Original recon_dft: {recon_dft.shape}")
                    setattr(self, '_logged_dft_mismatch_warning', True)
        
        if self.args.use_dct_features_appearance and recon_dct is not None and target_dct is not None:
            recon_dct_for_loss = recon_dct
            if recon_dct.ndim == 6 and target_dct.ndim == 4: # Standard case
                B, N, R, C_dct, H, W = recon_dct.shape
                recon_dct_for_loss = recon_dct.reshape(B, N, R, -1)

            if recon_dct_for_loss.shape == target_dct.shape:
                loss_dct = F.mse_loss(recon_dct_for_loss.float(), target_dct.float())
                total_recon_loss += self.args.lambda_recon_dct * loss_dct
                num_losses_counted +=1
            else: 
                if not hasattr(self, '_logged_dct_mismatch_warning'): # Log only once
                    self.logger.warning(f"Recon DCT shape mismatch: Recon effective {recon_dct_for_loss.shape}, Target {target_dct.shape}. Original recon_dct: {recon_dct.shape}")
                    setattr(self, '_logged_dct_mismatch_warning', True)

        if not (self.args.use_dft_features_appearance or self.args.use_dct_features_appearance): 
            if recon_pixels is not None and target_pixels is not None:
                if recon_pixels.shape == target_pixels.shape:
                    loss_pixel = F.mse_loss(recon_pixels.float(), target_pixels.float())
                    total_recon_loss += self.args.lambda_recon * loss_pixel 
                    num_losses_counted +=1
                else: 
                    if not hasattr(self, '_logged_pixel_mismatch_warning'):
                        self.logger.warning(f"Recon Pixel shape mismatch: Recon {recon_pixels.shape}, Target {target_pixels.shape}")
                        setattr(self, '_logged_pixel_mismatch_warning', True)
            elif recon_pixels is None and target_pixels is not None : 
                 self.logger.error("Pixel reconstruction mode, but G did not output pixels. This is a bug.")
                 return torch.tensor(1000.0, device=self.device, dtype=torch.float32) 
        
        return total_recon_loss / num_losses_counted if num_losses_counted > 0 else total_recon_loss


    @torch.no_grad()


    def _assemble_pixels_from_spectral(self,
                                       predicted_dft_coeffs: Optional[torch.Tensor], # Expected (B, N_gen_frames, N_reg, C, 2, H_p_spectral, W_p_coeff_spectral) OR (B, N_gen_frames, N_reg, D_dft_flat)
                                       predicted_dct_coeffs: Optional[torch.Tensor], # Expected (B, N_gen_frames, N_reg, C, H_p_spectral, W_p_spectral) OR (B, N_gen_frames, N_reg, D_dct_flat)
                                       gaad_bboxes: torch.Tensor, # Expected (B, N_gen_frames, N_reg, 4) - These bboxes MUST align with N_gen_frames of spectral coeffs
                                       target_image_height: int,
                                       target_image_width: int,
                                       num_image_channels_target: int, # C_img_target
                                       output_range: Tuple[float, float] = (-1.0, 1.0)
                                      ) -> Optional[torch.Tensor]:
        # This function now takes target H, W, C instead of full target_pixel_shape
        # to avoid ambiguity if N_pred_target from target_pixel_shape doesn't match N_gen_frames.
        # The number of frames in the output will be determined by N_gen_frames from spectral_coeffs.

        # Ensure self.model and self.args are accessible if needed for parameters
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        
        # Get generator's configured spectral patch size and number of channels
        # These are used by SpectralTransformUtils.reconstruct_...
        # Assuming these are attributes of the generator or accessible via m_ref.generator
        gen_num_img_channels = getattr(m_ref.generator, 'num_img_channels', num_image_channels_target)
        gen_patch_h_spectral = getattr(m_ref.generator, 'gen_patch_h_spectral', self.args.spectral_patch_size_h)
        gen_patch_w_spectral = getattr(m_ref.generator, 'gen_patch_w_spectral', self.args.spectral_patch_size_w)

        assembled_pixels_from_dft: Optional[torch.Tensor] = None
        assembled_pixels_from_dct: Optional[torch.Tensor] = None

        # --- DFT Path ---
        if predicted_dft_coeffs is not None and self.args.use_dft_features_appearance:
            # Current generator output for DFT is 7D: (B, N_gen_frames, N_reg, C, 2, H_p, W_p_coeff)
            if predicted_dft_coeffs.ndim == 7:
                B_dft, N_frames_dft, N_reg_dft, C_dft_in, two_dft, H_p_dft, W_p_c_dft = predicted_dft_coeffs.shape
                # Reshape for SpectralTransformUtils: (TotalPatches, FlatSpectralDim)
                dft_flat_for_recon = predicted_dft_coeffs.reshape(
                    B_dft * N_frames_dft * N_reg_dft,
                    C_dft_in * two_dft * H_p_dft * W_p_c_dft # This is D_dft_flat_per_region
                )
            elif predicted_dft_coeffs.ndim == 4: # Handle older flat format if necessary
                B_dft, N_frames_dft, N_reg_dft, _ = predicted_dft_coeffs.shape # D_dft_flat already last dim
                dft_flat_for_recon = predicted_dft_coeffs.reshape(B_dft * N_frames_dft * N_reg_dft, -1)
            else:
                self.logger.error(f"_assemble_pixels: DFT coeffs have unexpected ndim: {predicted_dft_coeffs.ndim}. Shape: {predicted_dft_coeffs.shape}")
                return None # Critical error

            # Ensure gaad_bboxes temporal dimension matches N_frames_dft
            bboxes_for_dft_assembly = gaad_bboxes
            if N_frames_dft != gaad_bboxes.shape[1]:
                 self.logger.warning_once(f"Assemble DFT: N_frames in DFT coeffs ({N_frames_dft}) != N_frames in bboxes ({gaad_bboxes.shape[1]}). Adjusting bboxes to match DFT coeffs frame count.")
                 if N_frames_dft < gaad_bboxes.shape[1]: # DFT has fewer frames than bboxes
                     bboxes_for_dft_assembly = gaad_bboxes[:, :N_frames_dft, ...]
                 elif N_frames_dft > gaad_bboxes.shape[1] and gaad_bboxes.shape[1] > 0: # DFT has more frames, pad bboxes
                     num_pad_bbox_dft = N_frames_dft - gaad_bboxes.shape[1]
                     pad_slice_bbox_dft = gaad_bboxes[:, -1:, ...].repeat(1, num_pad_bbox_dft, 1, 1)
                     bboxes_for_dft_assembly = torch.cat([gaad_bboxes, pad_slice_bbox_dft], dim=1)
                 # If gaad_bboxes.shape[1] == 0 and N_frames_dft > 0, this is an issue handled by caller or ImageAssemblyUtils

            try:
                patches_from_dft_flat = SpectralTransformUtils.reconstruct_patches_from_2d_dft(
                    dft_flat_for_recon, self.args.dft_norm_scale_video,
                    gen_num_img_channels, # Use generator's config for channels
                    gen_patch_h_spectral, gen_patch_w_spectral,
                    fft_norm_type=self.args.dft_fft_norm
                ) # Output: (TotalPatches, C_gen, H_gen_p, W_gen_p)

                patches_from_dft_structured = patches_from_dft_flat.view(
                    B_dft, N_frames_dft, N_reg_dft, gen_num_img_channels,
                    gen_patch_h_spectral, gen_patch_w_spectral
                )

                assembled_pixels_from_dft = ImageAssemblyUtils.assemble_frames_from_patches(
                    patches_from_dft_structured, bboxes_for_dft_assembly,
                    (target_image_height, target_image_width), output_range=output_range
                ) # Output: (B_dft, N_frames_dft, C_gen, H_img_target, W_img_target)
            except Exception as e_dft_recon: # Catching general Exception as ValueError might be too specific
                 self.logger.error(f"Error during DFT patch reconstruction/assembly: {e_dft_recon}. DFT_flat_shape: {dft_flat_for_recon.shape}", exc_info=True)
                 assembled_pixels_from_dft = None # Ensure it's None on failure

        # --- DCT Path ---
        if predicted_dct_coeffs is not None and self.args.use_dct_features_appearance:
            # Current generator output for DCT is 6D: (B, N_gen_frames, N_reg, C, H_p, W_p)
            if predicted_dct_coeffs.ndim == 6:
                B_dct, N_frames_dct, N_reg_dct, C_dct_in, H_p_dct, W_p_dct = predicted_dct_coeffs.shape
                dct_flat_for_recon = predicted_dct_coeffs.reshape(
                    B_dct * N_frames_dct * N_reg_dct,
                    C_dct_in * H_p_dct * W_p_dct # This is D_dct_flat_per_region
                )
            elif predicted_dct_coeffs.ndim == 4: # Handle older flat format
                B_dct, N_frames_dct, N_reg_dct, _ = predicted_dct_coeffs.shape
                dct_flat_for_recon = predicted_dct_coeffs.reshape(B_dct * N_frames_dct * N_reg_dct, -1)
            else:
                self.logger.error(f"_assemble_pixels: DCT coeffs have unexpected ndim: {predicted_dct_coeffs.ndim}. Shape: {predicted_dct_coeffs.shape}")
                return None

            bboxes_for_dct_assembly = gaad_bboxes
            if N_frames_dct != gaad_bboxes.shape[1]:
                 self.logger.warning_once(f"Assemble DCT: N_frames in DCT coeffs ({N_frames_dct}) != N_frames in bboxes ({gaad_bboxes.shape[1]}). Adjusting bboxes to match DCT coeffs frame count.")
                 if N_frames_dct < gaad_bboxes.shape[1]:
                     bboxes_for_dct_assembly = gaad_bboxes[:, :N_frames_dct, ...]
                 elif N_frames_dct > gaad_bboxes.shape[1] and gaad_bboxes.shape[1] > 0:
                     num_pad_bbox_dct = N_frames_dct - gaad_bboxes.shape[1]
                     pad_slice_bbox_dct = gaad_bboxes[:, -1:, ...].repeat(1, num_pad_bbox_dct, 1, 1)
                     bboxes_for_dct_assembly = torch.cat([gaad_bboxes, pad_slice_bbox_dct], dim=1)

            try:
                patches_from_dct_flat = SpectralTransformUtils.reconstruct_patches_from_2d_dct(
                    dct_flat_for_recon,
                    gen_num_img_channels,
                    gen_patch_h_spectral, gen_patch_w_spectral,
                    norm_type=self.args.dct_norm_type,
                    norm_global_scale=self.args.dct_norm_global_scale,
                    norm_tanh_scale=self.args.dct_norm_tanh_scale
                ) # Output: (TotalPatches, C_gen, H_gen_p, W_gen_p)

                patches_from_dct_structured = patches_from_dct_flat.view(
                    B_dct, N_frames_dct, N_reg_dct, gen_num_img_channels,
                    gen_patch_h_spectral, gen_patch_w_spectral
                )
                assembled_pixels_from_dct = ImageAssemblyUtils.assemble_frames_from_patches(
                    patches_from_dct_structured, bboxes_for_dct_assembly,
                    (target_image_height, target_image_width), output_range=output_range
                ) # Output: (B_dct, N_frames_dct, C_gen, H_img_target, W_img_target)
            except Exception as e_dct_recon:
                 self.logger.error(f"Error during DCT patch reconstruction/assembly: {e_dct_recon}. DCT_flat_shape: {dct_flat_for_recon.shape}", exc_info=True)
                 assembled_pixels_from_dct = None


        # --- Combine DFT and DCT (if both available) ---
        if assembled_pixels_from_dft is not None and assembled_pixels_from_dct is not None:
            # Ensure they have the same number of frames before averaging
            # This should be guaranteed if N_frames_dft and N_frames_dct are derived from generator's single N_pred
            if assembled_pixels_from_dft.shape[1] != assembled_pixels_from_dct.shape[1]:
                self.logger.warning_once(f"DFT ({assembled_pixels_from_dft.shape[1]}f) and DCT ({assembled_pixels_from_dct.shape[1]}f) "
                                     f"assembled pixels have different frame counts after processing. This is unexpected. "
                                     f"Will attempt to average over min_frames or return only one.")
                min_frames_avg = min(assembled_pixels_from_dft.shape[1], assembled_pixels_from_dct.shape[1])
                if min_frames_avg > 0:
                    return (assembled_pixels_from_dft[:, :min_frames_avg, ...] + assembled_pixels_from_dct[:, :min_frames_avg, ...]) / 2.0
                else: # If one became zero-frame, return the other if valid
                    return assembled_pixels_from_dft if assembled_pixels_from_dft.shape[1] > 0 else assembled_pixels_from_dct
            else: # Frame counts match
                return (assembled_pixels_from_dft + assembled_pixels_from_dct) / 2.0
        elif assembled_pixels_from_dft is not None:
            return assembled_pixels_from_dft
        elif assembled_pixels_from_dct is not None:
            return assembled_pixels_from_dct
        else:
            self.logger.warning_once("_assemble_pixels_from_spectral: Neither DFT nor DCT path yielded assembled pixels.")
            return None



    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, frames_to_log: Optional[torch.Tensor], num_frames_per_sequence_to_log: int = 1, num_sequences_to_log_max: int = 2):
        if not (self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb.run and frames_to_log is not None and frames_to_log.numel() > 0):
            return
        
        if frames_to_log.ndim == 4: 
            frames_to_log = frames_to_log.unsqueeze(0) 
        elif frames_to_log.ndim < 4 or frames_to_log.ndim > 5:
            self.logger.warning(f"WandB log samples: frames_to_log has unexpected shape {frames_to_log.shape}. Expected 4D or 5D. Skipping.")
            return

        B_log, N_seq_log, C_log, H_log, W_log = frames_to_log.shape
        num_to_actually_log_sequences = min(B_log, num_sequences_to_log_max)
        num_frames_to_log_this_seq = min(N_seq_log, num_frames_per_sequence_to_log)
        
        wandb_images_for_log = []
        for b_idx in range(num_to_actually_log_sequences):
            for frame_idx_in_seq in range(num_frames_to_log_this_seq):
                frame_tensor = frames_to_log[b_idx, frame_idx_in_seq, ...].cpu().float()
                if frame_tensor.shape[0] == 1 and C_log == 1: 
                    frame_tensor_display = frame_tensor.repeat(3,1,1) 
                elif frame_tensor.shape[0] != 3 and C_log !=1 : 
                    self.logger.warning(f"WandB: Frame has {frame_tensor.shape[0]} channels, expected 1 or 3. Taking first 3 or 1.")
                    if frame_tensor.shape[0] > 3: frame_tensor_display = frame_tensor[:3,...]
                    elif frame_tensor.shape[0] == 2: frame_tensor_display = frame_tensor.repeat(2,1,1)[:3,...] 
                    else: frame_tensor_display = frame_tensor 
                else: 
                    frame_tensor_display = frame_tensor

                img_0_1 = (frame_tensor_display.clamp(-1,1) + 1) / 2.0 
                caption = f"{tag_prefix} S{b_idx} F{frame_idx_in_seq} Ep{self.current_epoch+1} GStep{self.global_step}"
                wandb_images_for_log.append(wandb.Image(img_0_1, caption=caption))
        
        if wandb_images_for_log:
            try:
                wandb.log({f"samples_video/{tag_prefix}": wandb_images_for_log}, step=self.global_step)
            except Exception as e_wandb_log_vid:
                self.logger.error(f"WandB video sample log fail for {tag_prefix}: {e_wandb_log_vid}", exc_info=True)

    def _get_q_controller_data_for_heuristics(self) -> Dict[str, Any]:
        q_data: Dict[str, Any] = {'gen': {'is_valid': False}, 'active_d': {'is_valid': False}, 'lkl': {'is_valid': False}}
        controllers_map = { 'gen': self.q_controller_gen, 'active_d': self.q_controller_d_active, 'lkl': self.lambda_kl_q_controller }
        hist_names_g_d = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']
        hist_names_lkl = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric'] 
        
        for key, controller in controllers_map.items():
            if controller:
                q_data.setdefault(key, {}).update({'is_valid': True})
                q_data[key]['epsilon'] = controller.epsilon
                q_data[key]['on_probation'] = getattr(controller, 'on_probation', False) or getattr(controller, 'lkl_on_probation', False)
                
                if hasattr(controller, 'reward_hist') and controller.reward_hist:
                    valid_rewards = [r for r in list(controller.reward_hist)[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:] if r is not None and np.isfinite(r)]
                    q_data[key]['reward_median_short_term'] = np.median(valid_rewards) if valid_rewards else 0.0 
                else: q_data[key]['reward_median_short_term'] = 0.0

                current_hist_names_for_controller = hist_names_g_d if key in ['gen', 'active_d'] else hist_names_lkl
                hist_attr_prefix = "loss_" if key in ['gen', 'active_d'] else "interval_" 

                for lname in current_hist_names_for_controller:
                    hist_attr_name = f"{hist_attr_prefix}{lname}_hist" 
                    
                    if hasattr(controller, hist_attr_name):
                        hist_deque = getattr(controller, hist_attr_name)
                        if hist_deque:
                            finite_hist_values = [v for v in list(hist_deque) if v is not None and np.isfinite(v)]                             
                            val_for_trend = finite_hist_values[-1] if finite_hist_values else None
                            short_term_finite_values = [v for v in list(hist_deque)[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:] if v is not None and np.isfinite(v)]
                            median_val = np.median(short_term_finite_values) if short_term_finite_values else np.nan
                            q_data[key][f"{lname}_median_short_term"] = median_val
                            q_data[key][f"{lname}_trend_short_term"] = controller._get_trend_bin(hist_deque, val_for_trend) if val_for_trend is not None and hasattr(controller, '_get_trend_bin') else 2 
                        else: q_data[key][f"{lname}_median_short_term"] = np.nan; q_data[key][f"{lname}_trend_short_term"] = 2
                    else: q_data[key][f"{lname}_median_short_term"] = np.nan; q_data[key][f"{lname}_trend_short_term"] = 2
            else: q_data.setdefault(key, {}).update({'is_valid': False})

        if q_data.get('gen', {}).get('is_valid') and q_data['gen'].get('g_recon_median_short_term') is not None and np.isfinite(q_data['gen']['g_recon_median_short_term']):
            current_feature_recon_median = q_data['gen']['g_recon_median_short_term']
            self.q_data_derived_g_recon_hist.append(current_feature_recon_median) 
            self.avg_g_recon_hist_for_stagnation.append(current_feature_recon_median) 
            if len(self.avg_g_recon_hist_for_stagnation) >= max(2, self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS // 2):
                hist_vals_for_stag_check = list(self.avg_g_recon_hist_for_stagnation)
                past_relevant_history = hist_vals_for_stag_check[:-1] 
                if past_relevant_history:
                    past_recon_median = np.median(past_relevant_history)
                    improvement = past_recon_median - current_feature_recon_median
                    threshold_improvement = abs(past_recon_median) * self.RECON_STAGNATION_IMPROVEMENT_THRESH_REL
                    is_stagnant_improvement = improvement < threshold_improvement
                    is_loss_still_high = current_feature_recon_median > (self.TARGET_GOOD_RECON_THRESH_HEURISTIC * 1.25) 
                    self.rec_features_stagnant = is_stagnant_improvement and is_loss_still_high
                else: self.rec_features_stagnant = False
            else: self.rec_features_stagnant = False 
        else: self.rec_features_stagnant = False 
        return q_data

    def _check_and_perform_disc_switch(self, is_g_winning_hard: bool, is_d_very_weak: bool,
                                     is_d_q_stagnant:bool, is_d_strong: bool, is_g_stalled_adv: bool,
                                     current_g_kl_median: float, log_msgs_list: List[str]) -> bool:
        if not self.enable_heuristic_disc_switching: return False
        if self.steps_since_last_d_switch < self.disc_switch_min_steps_between: return False

        switched_this_cycle = False
        current_active_is_primary = (self.active_discriminator_key == 'primary')
        
        primary_is_pixel_like = "pixels" in getattr(self.discriminator_primary_obj, 'effective_input_type_for_trainer', "unknown").lower()
        primary_is_feature_like = "features" in getattr(self.discriminator_primary_obj, 'effective_input_type_for_trainer', "unknown").lower() or \
                                  "spectral" in getattr(self.discriminator_primary_obj, 'effective_input_type_for_trainer', "unknown").lower()
        alt_is_pixel_like = "pixels" in getattr(self.discriminator_alternative_obj, 'effective_input_type_for_trainer', "unknown").lower()
        alt_is_feature_like = "features" in getattr(self.discriminator_alternative_obj, 'effective_input_type_for_trainer', "unknown").lower() or \
                              "spectral" in getattr(self.discriminator_alternative_obj, 'effective_input_type_for_trainer', "unknown").lower()

        trigger_primary_to_alt = (current_active_is_primary and
                                  (primary_is_pixel_like and alt_is_feature_like) and 
                                  is_g_winning_hard and
                                  (is_d_very_weak or is_d_q_stagnant) and
                                  self.rec_features_stagnant)
        if trigger_primary_to_alt:
            self.consecutive_trigger_primary_to_alt_count += 1
            log_msgs_list.append(f"HEURISTIC D-SWITCH trigger (P->A): Count {self.consecutive_trigger_primary_to_alt_count}/{self.disc_switch_problem_state_count_thresh}. GWinHard:{is_g_winning_hard}, DWeak:{is_d_very_weak}, DQStag:{is_d_q_stagnant}, RecStag:{self.rec_features_stagnant}")
        else: self.consecutive_trigger_primary_to_alt_count = 0

        trigger_alt_to_primary = (not current_active_is_primary and 
                                  (alt_is_feature_like and primary_is_pixel_like) and 
                                  is_d_strong and
                                  is_g_stalled_adv and
                                  (current_g_kl_median < self.KL_HIGH_THRESH * 1.5)) 
        if trigger_alt_to_primary:
            self.consecutive_trigger_alt_to_primary_count += 1
            log_msgs_list.append(f"HEURISTIC D-SWITCH trigger (A->P): Count {self.consecutive_trigger_alt_to_primary_count}/{self.disc_switch_problem_state_count_thresh}. DStrong:{is_d_strong}, GStallAdv:{is_g_stalled_adv}, KLMed:{current_g_kl_median:.2f}")
        else: self.consecutive_trigger_alt_to_primary_count = 0

        if self.consecutive_trigger_primary_to_alt_count >= self.disc_switch_problem_state_count_thresh:
            if self.active_discriminator_key == 'primary': 
                self.active_discriminator_key = 'alternative'
                self._update_active_discriminator_pointers(); switched_this_cycle = True
                log_msgs_list.append(f"HEURISTIC: SWITCHED Discriminator from Primary to Alternative.")
                if self.q_controller_d_active and hasattr(self.q_controller_d_active, 'reset_q_learning_state'): self.q_controller_d_active.reset_q_learning_state(True, True, "Switched to Alt D by Heuristic", True)
            self.consecutive_trigger_primary_to_alt_count = 0 
        
        elif self.consecutive_trigger_alt_to_primary_count >= self.disc_switch_problem_state_count_thresh:
            if self.active_discriminator_key == 'alternative': 
                self.active_discriminator_key = 'primary'
                self._update_active_discriminator_pointers(); switched_this_cycle = True
                log_msgs_list.append(f"HEURISTIC: SWITCHED Discriminator from Alternative to Primary.")
                if self.q_controller_d_active and hasattr(self.q_controller_d_active, 'reset_q_learning_state'): self.q_controller_d_active.reset_q_learning_state(True, True, "Switched to Pri D by Heuristic", True)
            self.consecutive_trigger_alt_to_primary_count = 0

        if switched_this_cycle:
            self.steps_since_last_d_switch = 0; self.consecutive_heuristic_trigger_counts = defaultdict(int) 
            self.q_data_derived_g_recon_hist.clear(); self.avg_g_recon_hist_for_stagnation.clear()
            self.rec_features_stagnant = False 
            log_msgs_list.append("HEURISTIC: D switched, general heuristic counters & recon history reset.")
        return switched_this_cycle

    def _evaluate_training_state_and_apply_heuristics(self):
        if not self.am_main_process: return
        if not self.enable_heuristic_interventions:
            if hasattr(self, 'global_step') and self.global_step > 0 and self.heuristic_check_interval > 0 and self.global_step % (self.heuristic_check_interval * 10) == 0:
                 self.logger.info(f"GStep {self.global_step}: Heuristic interventions globally DISABLED.")
            self.heuristic_vae_feature_match_active = False; self.heuristic_penalize_g_easy_win_active = False
            self.heuristic_boost_active_d_lr_active = False; self.heuristic_force_d_q_explore_active = False
            self.heuristic_override_lambda_recon_factor = 1.0; self.heuristic_override_lambda_kl_factor = 1.0
            self.heuristic_override_lambda_gan_factor = 1.0
            self.lambda_kl = self.lambda_kl_base * self.heuristic_override_lambda_kl_factor 
            all_q_controllers_to_sync_lkl_heur_off = [self.q_controller_gen, self.q_controller_d_primary, self.q_controller_d_alt, self.lambda_kl_q_controller]
            for q_ctrl_heur_sync_off in all_q_controllers_to_sync_lkl_heur_off:
                if q_ctrl_heur_sync_off and hasattr(q_ctrl_heur_sync_off, 'set_current_lambda_kl'):
                    q_ctrl_heur_sync_off.set_current_lambda_kl(self.lambda_kl)
            return

        if self.global_step == 0 or (self.heuristic_check_interval > 0 and self.global_step % (self.heuristic_check_interval * 5) == 0): 
            self.logger.info(f"GStep {self.global_step}: Evaluating training state for heuristics.")
        
        q_data = self._get_q_controller_data_for_heuristics()
        gen_q, active_d_q = q_data.get('gen', {}), q_data.get('active_d', {})
        log_msgs = []
        
        current_lambda_recon_factor = 1.0; current_lambda_kl_factor = 1.0
        current_lambda_gan_factor = self.heuristic_override_lambda_gan_factor 
        current_boost_active_d_lr = False; current_force_d_q_explore = False
        current_penalize_g_easy_win = False; current_vae_feature_match = False

        g_adv_median = gen_q.get('g_adv_median_short_term', 0.7) if gen_q.get('is_valid') and np.isfinite(gen_q.get('g_adv_median_short_term', np.nan)) else 0.7
        d_total_median = active_d_q.get('d_total_median_short_term', 0.7) if active_d_q.get('is_valid') and np.isfinite(active_d_q.get('d_total_median_short_term', np.nan)) else 0.7
        d_q_reward_median = active_d_q.get('reward_median_short_term', 0.0) if active_d_q.get('is_valid') and np.isfinite(active_d_q.get('reward_median_short_term', np.nan)) else 0.0
        g_kl_median_val = gen_q.get('g_kl_median_short_term', 0.0) if gen_q.get('is_valid') and np.isfinite(gen_q.get('g_kl_median_short_term', np.nan)) else 0.0

        is_g_dominating_very_much = g_adv_median < self.G_VERY_MUCH_WINNING_THRESH
        is_d_very_weak = d_total_median > self.D_VERY_WEAK_THRESH
        is_d_q_learner_stagnant = d_q_reward_median < self.Q_REWARD_STAGNATION_THRESH
        is_d_strong = d_total_median < self.D_STRONG_THRESH
        is_g_stalled_adv = g_adv_median > self.G_STALLED_THRESH
        
        active_d_is_feature_based = "features" in self.active_disc_effective_trainer_input_type.lower() or \
                                  "spectral" in self.active_disc_effective_trainer_input_type.lower()
        switched_d_this_cycle = False
        if self.enable_heuristic_disc_switching:
            switched_d_this_cycle = self._check_and_perform_disc_switch(is_g_dominating_very_much, is_d_very_weak, is_d_q_learner_stagnant, is_d_strong, is_g_stalled_adv, g_kl_median_val, log_msgs)
        
        if switched_d_this_cycle:
            self.consecutive_heuristic_trigger_counts = defaultdict(int)
            current_lambda_gan_factor = 1.0; current_lambda_recon_factor = 1.0; current_lambda_kl_factor = 1.0
            log_msgs.append("HEURISTIC: D switched, other heuristic factors reset for this cycle.")
        else: 
            condition_gan_rebalance = is_g_dominating_very_much and (is_d_very_weak or is_d_q_learner_stagnant) and self.rec_features_stagnant
            if condition_gan_rebalance:
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] += 1
                if self.consecutive_heuristic_trigger_counts['gan_rebalance'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_penalize_g_easy_win = True
                    current_lambda_recon_factor = getattr(self.args, 'heuristic_recon_boost_factor_video', 1.2) 
                    current_lambda_gan_factor = min(self.heuristic_override_lambda_gan_factor * 1.05, getattr(self.args, 'heuristic_max_lambda_gan_factor', 1.3))
                    if is_d_q_learner_stagnant: current_boost_active_d_lr = True; current_force_d_q_explore = True
                    log_msgs.append(f"HEURISTIC: GAN REBALANCE triggered. PenalizeG:{current_penalize_g_easy_win}, LRecF:{current_lambda_recon_factor:.2f}, LGanF:{current_lambda_gan_factor:.2f}, D_LRB:{current_boost_active_d_lr}, D_QE:{current_force_d_q_explore}")
            else: 
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] = 0
                if self.heuristic_override_lambda_gan_factor > 1.0: current_lambda_gan_factor = max(self.heuristic_override_lambda_gan_factor * 0.98, 1.0)

            condition_vae_feat_match = (active_d_is_feature_based and self.lambda_feat_match_heuristic > 0 and not is_g_dominating_very_much and not is_d_very_weak and (is_d_strong or not is_d_q_learner_stagnant) and self.rec_features_stagnant)
            if condition_vae_feat_match:
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] += 1
                if self.consecutive_heuristic_trigger_counts['vae_feat_match'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_vae_feature_match = True
                    if self.lambda_kl * self.heuristic_override_lambda_kl_factor < 1e-4 : current_lambda_kl_factor = 1.5 
                    current_lambda_gan_factor = max(self.heuristic_override_lambda_gan_factor * 0.95, getattr(self.args, 'heuristic_min_lambda_gan_factor', 0.7))
                    log_msgs.append(f"HEURISTIC: VAE FEATURE MATCH triggered. LKLF:{current_lambda_kl_factor:.2f}, LGanF:{current_lambda_gan_factor:.2f}")
            else: 
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] = 0
                if not active_d_is_feature_based and self.heuristic_vae_feature_match_active: log_msgs.append(f"HEURISTIC: Disabling VAE FM as Active D ('{self.active_disc_effective_trainer_input_type}') is not feature-based.")
                if self.heuristic_override_lambda_gan_factor < 1.0: current_lambda_gan_factor = min(self.heuristic_override_lambda_gan_factor * 1.02, 1.0) 
            
            is_kl_too_high = g_kl_median_val > self.KL_HIGH_THRESH
            condition_kl_reduction = is_kl_too_high and self.rec_features_stagnant
            if condition_kl_reduction:
                self.consecutive_heuristic_trigger_counts['kl_reduction'] +=1
                if self.consecutive_heuristic_trigger_counts['kl_reduction'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_lambda_kl_factor = max(0.5, current_lambda_kl_factor * 0.7) 
                    current_lambda_recon_factor = getattr(self.args, 'heuristic_recon_boost_factor_video', 1.2) 
                    log_msgs.append(f"HEURISTIC: KL HIGH & RECON STAGNANT. LKLF:{current_lambda_kl_factor:.2f}, LRecF:{current_lambda_recon_factor:.2f}")
            else: 
                self.consecutive_heuristic_trigger_counts['kl_reduction'] = 0
                if self.heuristic_override_lambda_kl_factor < 1.0 and not condition_vae_feat_match: 
                    current_lambda_kl_factor = min(1.0, self.heuristic_override_lambda_kl_factor * 1.02)

        self.heuristic_penalize_g_easy_win_active = current_penalize_g_easy_win
        self.heuristic_override_lambda_recon_factor = current_lambda_recon_factor
        self.heuristic_boost_active_d_lr_active = current_boost_active_d_lr
        self.heuristic_vae_feature_match_active = current_vae_feature_match
        self.heuristic_override_lambda_kl_factor = current_lambda_kl_factor
        self.heuristic_override_lambda_gan_factor = current_lambda_gan_factor 
        self.heuristic_force_d_q_explore_active = current_force_d_q_explore # Assign back
        
        self.lambda_kl = self.lambda_kl_base * self.heuristic_override_lambda_kl_factor 
        all_q_controllers_to_sync_lkl_heur = [self.q_controller_gen, self.q_controller_d_primary, self.q_controller_d_alt, self.lambda_kl_q_controller]
        for q_ctrl_heur_sync in all_q_controllers_to_sync_lkl_heur:
            if q_ctrl_heur_sync and hasattr(q_ctrl_heur_sync, 'set_current_lambda_kl'):
                q_ctrl_heur_sync.set_current_lambda_kl(self.lambda_kl)

        if self.heuristic_force_d_q_explore_active and self.q_controller_d_active and hasattr(self.q_controller_d_active, 'force_exploration_boost'): 
            self.q_controller_d_active.force_exploration_boost(self.heuristic_d_q_explore_duration, self.heuristic_d_q_explore_boost_epsilon)
            log_msgs.append(f"HEURISTIC: Active D Q-Ctrl exploration boosted for {self.heuristic_d_q_explore_duration} steps to eps {self.heuristic_d_q_explore_boost_epsilon:.2f}.")
        
        if log_msgs and self.am_main_process: 
            for msg in log_msgs: self.logger.info(msg)




    def _train_discriminator_step(self, real_frames_full_sequence: torch.Tensor, m_ref: "WuBuGAADHybridGenNet") -> Dict[str, torch.Tensor]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_trainer_input_type = self.active_disc_effective_trainer_input_type # From _update_active_discriminator_pointers
        
        B = real_frames_full_sequence.shape[0]
        device = real_frames_full_sequence.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0, device=device)).dtype

        # Determine how many frames the active D processes
        num_frames_for_active_d = getattr(d_ref_active, 'num_frames_to_discriminate', self.args.num_predict_frames)
        num_input_frames_conditioning = self.video_config.get("num_input_frames", 0)

        real_labels = torch.ones(B, device=device, dtype=dtype_model)
        fake_labels = torch.zeros(B, device=device, dtype=dtype_model)
        losses_d_micro: Dict[str, torch.Tensor] = {}

        for p in d_ref_active.parameters(): p.requires_grad = True
        for p in m_ref.parameters(): p.requires_grad = False
        
        # --- Prepare Real Data for Discriminator ---
        real_input_for_d_main: torch.Tensor
        gaad_bboxes_for_d_real_cond: Optional[torch.Tensor] = None
        
        # Real data always starts as pixels. If D needs features, we encode.
        # We need target spectral features (from encoder) if D is feature-based.
        # We need GAAD bboxes from encoder if D is pixel-based AND uses FiLM.
        # The GAAD bboxes should correspond to the *frames the D will see*.
        # If D sees predicted frames, these are from index num_input_frames_conditioning onwards.
        
        with torch.no_grad(): # Encoder pass for targets and bboxes for D conditioning
            _, _, gaad_bboxes_all_input_enc, target_dft_all_input_enc, target_dct_all_input_enc = m_ref.encode(real_frames_full_sequence.to(device, dtype_model))

        # Select the slice of real data (pixels or features) that the D will process.
        # This typically corresponds to the "prediction window" of the original sequence.
        start_idx_for_d_real_data = num_input_frames_conditioning
        end_idx_for_d_real_data = num_input_frames_conditioning + num_frames_for_active_d
        
        if end_idx_for_d_real_data > real_frames_full_sequence.shape[1]:
            self.logger.warning_once(f"D Real Path: Not enough frames in input sequence ({real_frames_full_sequence.shape[1]}) "
                                f"to cover D's window ({num_frames_for_active_d} frames after {num_input_frames_conditioning} context frames). "
                                f"Slicing available frames. This might cause issues if D expects fixed length.")
            end_idx_for_d_real_data = real_frames_full_sequence.shape[1]
            # num_frames_for_active_d might need to be temporarily adjusted for this batch if D is flexible,
            # or padding applied if D requires fixed length. For now, we slice what's available.
            # The D itself might pad if it receives fewer frames than its `num_frames_to_discriminate` config.

        if active_d_trainer_input_type == "assembled_pixels":
            real_input_for_d_main = real_frames_full_sequence[:, start_idx_for_d_real_data:end_idx_for_d_real_data, ...].to(device, dtype_model)
            # Check if the D module uses GAAD FiLM
            d_module_for_check = d_ref_active.actual_discriminator_module if isinstance(d_ref_active, VideoDiscriminatorWrapper) else d_ref_active
            if hasattr(d_module_for_check, 'use_gaad_film_condition') and d_module_for_check.use_gaad_film_condition:
                if gaad_bboxes_all_input_enc is not None:
                    gaad_bboxes_for_d_real_cond = gaad_bboxes_all_input_enc[:, start_idx_for_d_real_data:end_idx_for_d_real_data, ...].to(device, dtype_model)
                else:
                    self.logger.warning_once("D Real Path (Pixel D with FiLM): Encoder did not return GAAD bboxes. FiLM condition will be None.")
        
        elif active_d_trainer_input_type == "regional_spectral_features_combined":
            real_features_list_for_d = []
            if self.args.use_dft_features_appearance and target_dft_all_input_enc is not None:
                real_features_list_for_d.append(target_dft_all_input_enc[:, start_idx_for_d_real_data:end_idx_for_d_real_data, ...])
            if self.args.use_dct_features_appearance and target_dct_all_input_enc is not None:
                real_features_list_for_d.append(target_dct_all_input_enc[:, start_idx_for_d_real_data:end_idx_for_d_real_data, ...])
            
            if not real_features_list_for_d:
                raise ValueError("D training (real, feature-D): No target spectral features from encoder for feature-based D.")
            
            concatenated_real_features_for_d = torch.cat(real_features_list_for_d, dim=-1) # (B, N_active_D, N_reg, D_combined_flat)
            
            # Reshape for GlobalWuBuVideoFeatureDiscriminator: (B, N_reg, N_active_D * D_combined_flat)
            B_real_feat, N_active_D_real, N_reg_real, _ = concatenated_real_features_for_d.shape
            real_input_for_d_main_permuted = concatenated_real_features_for_d.permute(0, 2, 1, 3) # (B, N_reg, N_active_D, D_combined_flat)
            real_input_for_d_main = real_input_for_d_main_permuted.reshape(B_real_feat, N_reg_real, -1)
        else:
            raise ValueError(f"D training (real): Unsupported active_d_trainer_input_type: {active_d_trainer_input_type}")

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            # --- Real Path Forward ---
            # The active discriminator d_ref_active should handle if num_frames received is less than its configured num_frames_to_discriminate (e.g. by padding)
            real_logits = d_ref_active(real_input_for_d_main, gaad_bboxes_cond=gaad_bboxes_for_d_real_cond)
            loss_d_real = self.adversarial_loss(real_logits.squeeze(-1) if real_logits.ndim > 1 and real_logits.shape[-1] == 1 else real_logits, real_labels)

            # --- Fake Path ---
            fake_input_for_d_main: torch.Tensor
            fake_gaad_bboxes_for_d_cond: Optional[torch.Tensor] = None
            
            with torch.no_grad(): # Generate fake samples
                # m_ref.forward() returns reconstructions aligned with the *prediction window*
                # and bboxes_used_by_decoder corresponding to that prediction window.
                fake_pixel_output_gen, fake_dft_output_gen, fake_dct_output_gen, \
                _, _, bboxes_used_by_decoder_for_fake, _, _ = m_ref(real_frames_full_sequence)
                # bboxes_used_by_decoder_for_fake is (B, N_gen_predict_frames, N_reg, 4)

            # Now, prepare these generated fakes for the active discriminator
            if active_d_trainer_input_type == "assembled_pixels":
                if fake_pixel_output_gen is not None: # G generated pixels directly
                    assembled_fake_pixels = fake_pixel_output_gen
                else: # G generated spectral, need to assemble
                    if bboxes_used_by_decoder_for_fake is None:
                        raise RuntimeError("D training (fake, pixel-D): bboxes_used_by_decoder_for_fake is None from G, cannot assemble pixels.")
                    
                    # Target shape for assembly should match the number of frames G generated
                    # and the image H, W from args.
                    N_gen_predict_frames = fake_dft_output_gen.shape[1] if fake_dft_output_gen is not None else \
                                          (fake_dct_output_gen.shape[1] if fake_dct_output_gen is not None else 0)
                    if N_gen_predict_frames == 0:
                        raise RuntimeError("D training (fake, pixel-D): G generated no spectral frames to assemble.")

                    assembled_fake_pixels = self._assemble_pixels_from_spectral(
                        fake_dft_output_gen, fake_dct_output_gen, bboxes_used_by_decoder_for_fake,
                        target_image_height=self.args.image_h, target_image_width=self.args.image_w,
                        num_image_channels_target=self.video_config['num_channels']
                    )
                    if assembled_fake_pixels is None:
                        raise RuntimeError("Failed to assemble pixels from G's spectral output for D (pixel-based).")
                
                # Slice the assembled fake pixels for the number of frames D processes
                fake_input_for_d_main = assembled_fake_pixels[:, :num_frames_for_active_d, ...]
                
                # Prepare GAAD bboxes for FiLM if D uses it
                d_module_for_check_fake = d_ref_active.actual_discriminator_module if isinstance(d_ref_active, VideoDiscriminatorWrapper) else d_ref_active
                if hasattr(d_module_for_check_fake, 'use_gaad_film_condition') and d_module_for_check_fake.use_gaad_film_condition:
                    if bboxes_used_by_decoder_for_fake is not None:
                        fake_gaad_bboxes_for_d_cond = bboxes_used_by_decoder_for_fake[:, :num_frames_for_active_d, ...]
                    else:
                        self.logger.warning_once("D Fake Path (Pixel D with FiLM): G did not return bboxes_used_by_decoder. FiLM condition will be None.")
            
            elif active_d_trainer_input_type == "regional_spectral_features_combined":
                fake_features_list_for_d = []
                if self.args.use_dft_features_appearance and fake_dft_output_gen is not None:
                    # fake_dft_output_gen is (B, N_gen_pred, N_reg, C, 2, H_p, W_p_coeff) or (B, N_gen_pred, N_reg, D_flat)
                    # Reshape to (B, N_gen_pred, N_reg, D_flat_dft_per_reg)
                    if fake_dft_output_gen.ndim == 7:
                        B_fk_dft, N_fk_dft, R_fk_dft, C_fk, two_fk, Hp_fk, Wpc_fk = fake_dft_output_gen.shape
                        reshaped_fk_dft = fake_dft_output_gen.reshape(B_fk_dft, N_fk_dft, R_fk_dft, -1)
                    elif fake_dft_output_gen.ndim == 4: reshaped_fk_dft = fake_dft_output_gen
                    else: raise ValueError(f"Unexpected ndim {fake_dft_output_gen.ndim} for fake_dft_output_gen")
                    fake_features_list_for_d.append(reshaped_fk_dft)

                if self.args.use_dct_features_appearance and fake_dct_output_gen is not None:
                    # fake_dct_output_gen is (B, N_gen_pred, N_reg, C, H_p, W_p) or (B, N_gen_pred, N_reg, D_flat)
                    if fake_dct_output_gen.ndim == 6:
                        B_fk_dct, N_fk_dct, R_fk_dct, C_fk_d, Hp_fk_d, Wp_fk_d = fake_dct_output_gen.shape
                        reshaped_fk_dct = fake_dct_output_gen.reshape(B_fk_dct, N_fk_dct, R_fk_dct, -1)
                    elif fake_dct_output_gen.ndim == 4: reshaped_fk_dct = fake_dct_output_gen
                    else: raise ValueError(f"Unexpected ndim {fake_dct_output_gen.ndim} for fake_dct_output_gen")
                    fake_features_list_for_d.append(reshaped_fk_dct)

                if not fake_features_list_for_d:
                    raise ValueError("D training (fake, feature-D): No spectral features from G for feature-based D.")
                
                concatenated_fake_features_for_d = torch.cat(fake_features_list_for_d, dim=-1) # (B, N_gen_pred, N_reg, D_combined_flat)
                
                # Slice for the frames D processes
                fake_input_for_d_main_sliced_frames = concatenated_fake_features_for_d[:, :num_frames_for_active_d, ...]
                
                # Reshape for GlobalWuBuVideoFeatureDiscriminator: (B, N_reg, N_active_D_fakes * D_combined_flat)
                B_fake_feat, N_active_D_fakes, N_reg_fakes, _ = fake_input_for_d_main_sliced_frames.shape
                fake_input_for_d_main_permuted = fake_input_for_d_main_sliced_frames.permute(0, 2, 1, 3)
                fake_input_for_d_main = fake_input_for_d_main_permuted.reshape(B_fake_feat, N_reg_fakes, -1)
            else:
                raise ValueError(f"D training (fake): Unsupported active_d_trainer_input_type: {active_d_trainer_input_type}")
            
            # --- Fake Path Forward ---
            fake_logits = d_ref_active(fake_input_for_d_main.detach(), gaad_bboxes_cond=fake_gaad_bboxes_for_d_cond)
            loss_d_fake = self.adversarial_loss(fake_logits.squeeze(-1) if fake_logits.ndim > 1 and fake_logits.shape[-1] == 1 else fake_logits, fake_labels)
            
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total_scaled_for_accum_micro = loss_d_total_micro / self.grad_accum_steps
            
        self.scaler_disc_active.scale(loss_d_total_scaled_for_accum_micro).backward()
        
        losses_d_micro['loss_d_real_micro'] = loss_d_real.detach()
        losses_d_micro['loss_d_fake_micro'] = loss_d_fake.detach()
        losses_d_micro['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d_micro


    def _train_generator_step(self, real_frames_full_sequence: torch.Tensor, m_ref: "WuBuGAADHybridGenNet") -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_trainer_input_type = self.active_disc_effective_trainer_input_type

        B = real_frames_full_sequence.shape[0]
        device = real_frames_full_sequence.device
        dtype_model = next(iter(m_ref.parameters()), torch.tensor(0.0, device=device)).dtype
        real_labels_for_g = torch.ones(B, device=device, dtype=dtype_model)
        losses_g_micro: Dict[str, torch.Tensor] = {}
        assembled_pixels_for_log: Optional[torch.Tensor] = None # For logging generated samples

        for p in d_ref_active.parameters(): p.requires_grad = False
        for p in m_ref.parameters(): p.requires_grad = True

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            # m_ref.forward() returns reconstructions and targets aligned with the *prediction window*
            # bboxes_used_by_decoder is (B, N_gen_predict_frames, N_reg, 4)
            # target_dft/dct_for_loss are (B, N_gen_predict_frames, N_reg, D_flat_spectral)
            recon_pixel_frames_gen, recon_dft_coeffs_gen, recon_dct_coeffs_gen, \
            mu, logvar, bboxes_used_by_decoder, \
            target_dft_features_for_loss, target_dct_features_for_loss = m_ref(real_frames_full_sequence.to(device, dtype_model))

            num_gen_predict_frames = self.video_config.get("num_predict_frames", 1) # How many frames G generates
            
            # target_pixels_for_loss should also correspond to the generator's output window
            num_input_f_cond = self.video_config.get("num_input_frames", 0)
            target_pixels_for_loss = real_frames_full_sequence[
                :, num_input_f_cond : num_input_f_cond + num_gen_predict_frames, ...
            ].to(device, dtype_model) if recon_pixel_frames_gen is not None else None

            loss_recon_raw = self._compute_recon_loss(
                recon_dft_coeffs_gen, target_dft_features_for_loss,
                recon_dct_coeffs_gen, target_dct_features_for_loss,
                recon_pixel_frames_gen, target_pixels_for_loss
            )
            loss_kl_raw = self._compute_kl_loss(mu, logvar)
            
            loss_recon_eff = self.args.lambda_recon * self.heuristic_override_lambda_recon_factor * loss_recon_raw
            loss_kl_eff = self.lambda_kl * self.heuristic_override_lambda_kl_factor * loss_kl_raw

            # --- Adversarial Loss for Generator ---
            adv_input_for_d_main: torch.Tensor
            adv_gaad_bboxes_for_d_cond: Optional[torch.Tensor] = None
            
            # Determine how many frames the active D will process from G's output
            num_frames_for_active_d_from_gen = getattr(d_ref_active, 'num_frames_to_discriminate', num_gen_predict_frames)
            
            d_module_for_check = d_ref_active.actual_discriminator_module if isinstance(d_ref_active, VideoDiscriminatorWrapper) else d_ref_active


            if active_d_trainer_input_type == "assembled_pixels":
                if recon_pixel_frames_gen is not None: # G generated pixels directly
                    assembled_pixels_for_log = recon_pixel_frames_gen # For logging
                    adv_input_for_d_main = recon_pixel_frames_gen[:, :num_frames_for_active_d_from_gen, ...]
                else: # G generated spectral, need to assemble
                    if bboxes_used_by_decoder is None:
                        raise RuntimeError("G training (adv, pixel-D): bboxes_used_by_decoder is None from G's forward, cannot assemble pixels.")
                    
                    assembled_pixels_for_log = self._assemble_pixels_from_spectral(
                        recon_dft_coeffs_gen, recon_dct_coeffs_gen, bboxes_used_by_decoder,
                        target_image_height=self.args.image_h, target_image_width=self.args.image_w,
                        num_image_channels_target=self.video_config['num_channels']
                    )
                    if assembled_pixels_for_log is None:
                        raise RuntimeError("Failed to get/assemble pixels for pixel-based D in G step.")
                    adv_input_for_d_main = assembled_pixels_for_log[:, :num_frames_for_active_d_from_gen, ...]
                
                if hasattr(d_module_for_check, 'use_gaad_film_condition') and d_module_for_check.use_gaad_film_condition: # type: ignore
                    if bboxes_used_by_decoder is not None:
                        adv_gaad_bboxes_for_d_cond = bboxes_used_by_decoder[:, :num_frames_for_active_d_from_gen, ...]
            
            elif active_d_trainer_input_type == "regional_spectral_features_combined":
                adv_features_list_g = []
                if self.args.use_dft_features_appearance and recon_dft_coeffs_gen is not None:
                    if recon_dft_coeffs_gen.ndim == 7: B_g_dft, N_g_dft, R_g_dft, C_g, _, _, _ = recon_dft_coeffs_gen.shape; reshaped_g_dft = recon_dft_coeffs_gen.reshape(B_g_dft, N_g_dft, R_g_dft, -1)
                    elif recon_dft_coeffs_gen.ndim == 4: reshaped_g_dft = recon_dft_coeffs_gen
                    else: raise ValueError(f"Unexpected ndim {recon_dft_coeffs_gen.ndim} for recon_dft_coeffs_gen in G step")
                    adv_features_list_g.append(reshaped_g_dft)
                if self.args.use_dct_features_appearance and recon_dct_coeffs_gen is not None:
                    if recon_dct_coeffs_gen.ndim == 6: B_g_dct, N_g_dct, R_g_dct, C_g_d, _, _ = recon_dct_coeffs_gen.shape; reshaped_g_dct = recon_dct_coeffs_gen.reshape(B_g_dct, N_g_dct, R_g_dct, -1)
                    elif recon_dct_coeffs_gen.ndim == 4: reshaped_g_dct = recon_dct_coeffs_gen
                    else: raise ValueError(f"Unexpected ndim {recon_dct_coeffs_gen.ndim} for recon_dct_coeffs_gen in G step")
                    adv_features_list_g.append(reshaped_g_dct)
                
                if not adv_features_list_g: raise ValueError("G training (adv, feature-D): No spectral features from G for D.")
                
                concatenated_adv_features_g = torch.cat(adv_features_list_g, dim=-1) # (B, N_gen_pred, N_reg, D_combined_flat)
                
                adv_input_for_d_main_sliced_frames = concatenated_adv_features_g[:, :num_frames_for_active_d_from_gen, ...]
                B_g_feat, N_active_D_g, N_reg_g, _ = adv_input_for_d_main_sliced_frames.shape
                adv_input_for_d_main_permuted = adv_input_for_d_main_sliced_frames.permute(0, 2, 1, 3)
                adv_input_for_d_main = adv_input_for_d_main_permuted.reshape(B_g_feat, N_reg_g, -1)
                
                # Also assemble pixels if needed for logging, even if D is feature-based
                if self.am_main_process and self.args.wandb_log_train_recon_interval > 0: # Check if logging needed
                     if bboxes_used_by_decoder is not None:
                        assembled_pixels_for_log = self._assemble_pixels_from_spectral(
                            recon_dft_coeffs_gen, recon_dct_coeffs_gen, bboxes_used_by_decoder,
                            target_image_height=self.args.image_h, target_image_width=self.args.image_w,
                            num_image_channels_target=self.video_config['num_channels']
                        )
            else:
                raise ValueError(f"G training (adv): Unsupported active_d_trainer_input_type: {active_d_trainer_input_type}")

            # --- Adversarial Forward and Loss ---
            d_output_for_adv = d_ref_active(adv_input_for_d_main, gaad_bboxes_cond=adv_gaad_bboxes_for_d_cond, return_features=self.heuristic_vae_feature_match_active)
            fake_logits_for_g: torch.Tensor
            features_from_d_for_g_feat_match: Optional[torch.Tensor] = None
            if isinstance(d_output_for_adv, tuple) and len(d_output_for_adv) > 1 and isinstance(d_output_for_adv[0], torch.Tensor) and isinstance(d_output_for_adv[1], torch.Tensor):
                fake_logits_for_g, features_from_d_for_g_feat_match = d_output_for_adv
            elif isinstance(d_output_for_adv, torch.Tensor):
                fake_logits_for_g = d_output_for_adv
            else: raise TypeError(f"Unexpected output type from discriminator: {type(d_output_for_adv)}")


            loss_g_adv_raw = self.adversarial_loss(fake_logits_for_g.squeeze(-1) if fake_logits_for_g.ndim > 1 and fake_logits_for_g.shape[-1] == 1 else fake_logits_for_g, real_labels_for_g)
            loss_g_adv_eff = self.lambda_gan * self.heuristic_override_lambda_gan_factor * loss_g_adv_raw
            loss_g_total_micro = loss_recon_eff + loss_kl_eff + loss_g_adv_eff

            # --- Heuristic Feature Matching Loss ---
            if self.heuristic_vae_feature_match_active and features_from_d_for_g_feat_match is not None and self.lambda_feat_match_heuristic > 0:
                with torch.no_grad(): # Get features from D for REAL data
                    real_input_for_d_fm_main: torch.Tensor
                    real_gaad_bboxes_for_d_fm_cond: Optional[torch.Tensor] = None
                    
                    # Slicing real data for D's input window (feature matching part)
                    start_idx_fm_real = num_input_f_cond
                    end_idx_fm_real = num_input_f_cond + num_frames_for_active_d_from_gen # Match G's output window length
                    if end_idx_fm_real > real_frames_full_sequence.shape[1]: # Adjust if input too short
                        end_idx_fm_real = real_frames_full_sequence.shape[1]

                    if active_d_trainer_input_type == "assembled_pixels":
                        real_input_for_d_fm_main = real_frames_full_sequence[:, start_idx_fm_real:end_idx_fm_real, ...]
                        if hasattr(d_module_for_check, 'use_gaad_film_condition') and d_module_for_check.use_gaad_film_condition: # type: ignore
                             _, _, gaad_bboxes_real_enc_fm_all, _, _ = m_ref.encode(real_frames_full_sequence)
                             if gaad_bboxes_real_enc_fm_all is not None:
                                 real_gaad_bboxes_for_d_fm_cond = gaad_bboxes_real_enc_fm_all[:, start_idx_fm_real:end_idx_fm_real, ...]
                    elif active_d_trainer_input_type == "regional_spectral_features_combined":
                        # Use target_dft/dct_features_for_loss which are already aligned with G's output window
                        target_features_list_fm = []
                        if self.args.use_dft_features_appearance and target_dft_features_for_loss is not None:
                            target_features_list_fm.append(target_dft_features_for_loss[:, :num_frames_for_active_d_from_gen, ...]) # Slice to D's input length
                        if self.args.use_dct_features_appearance and target_dct_features_for_loss is not None:
                            target_features_list_fm.append(target_dct_features_for_loss[:, :num_frames_for_active_d_from_gen, ...])
                        
                        if not target_features_list_fm: raise ValueError("G FM: No target spectral features for D feature matching (real path).")
                        
                        concatenated_target_features_fm = torch.cat(target_features_list_fm, dim=-1)
                        B_fm_real, N_active_D_fm, N_reg_fm, _ = concatenated_target_features_fm.shape
                        real_input_for_d_fm_main_permuted = concatenated_target_features_fm.permute(0, 2, 1, 3)
                        real_input_for_d_fm_main = real_input_for_d_fm_main_permuted.reshape(B_fm_real, N_reg_fm, -1)
                    else:
                        # This case should be caught by earlier checks on active_d_trainer_input_type
                        real_input_for_d_fm_main = torch.empty(0, device=device, dtype=dtype_model)
                    
                    target_features_d: Optional[torch.Tensor] = None
                    if real_input_for_d_fm_main.numel() > 0 and real_input_for_d_fm_main.shape[0] > 0:
                        # Ensure real_input_for_d_fm_main has correct number of frames for D
                        # If D processes fewer frames than G generates, slice input for D.
                        # The `num_frames_for_active_d_from_gen` should match what D expects.
                        # If `real_input_for_d_fm_main` was prepared based on G's output length, and D takes less,
                        # then D itself must handle the slicing or we slice here.
                        # Assuming D handles it or `num_frames_for_active_d_from_gen` matches what D needs.
                        target_d_output_fm = d_ref_active(real_input_for_d_fm_main, gaad_bboxes_cond=real_gaad_bboxes_for_d_fm_cond, return_features=True)
                        if isinstance(target_d_output_fm, tuple) and len(target_d_output_fm) > 1 and isinstance(target_d_output_fm[1], torch.Tensor):
                            target_features_d = target_d_output_fm[1]
                        else:
                            self.logger.warning_once("G FM: D did not return features for real data during feature matching.")
                
                if target_features_d is not None and features_from_d_for_g_feat_match.shape == target_features_d.shape:
                    loss_g_feat_match = F.mse_loss(features_from_d_for_g_feat_match, target_features_d.detach())
                    loss_g_total_micro += self.lambda_feat_match_heuristic * loss_g_feat_match
                    losses_g_micro['loss_g_feat_match_micro'] = loss_g_feat_match.detach()
                elif target_features_d is not None:
                    self.logger.warning_once(f"G FM shapes mismatch: Fake_D_feat {features_from_d_for_g_feat_match.shape}, Real_D_feat {target_features_d.shape}")

            # --- Heuristic G Easy Win Penalty ---
            if self.heuristic_penalize_g_easy_win_active:
                # Use raw recon loss for this check, not the scaled one
                if loss_g_adv_raw.item() < self.G_WINNING_THRESH and loss_recon_raw.item() > self.TARGET_GOOD_RECON_THRESH_HEURISTIC:
                    denominator_penalty = loss_g_adv_raw.item() + getattr(self.args, 'g_easy_win_penalty_eps_denom', 1e-4)
                    penalty_val = (loss_recon_raw.item() - self.TARGET_GOOD_RECON_THRESH_HEURISTIC) / max(EPS, denominator_penalty) * self.lambda_g_easy_win_penalty_heuristic
                    penalty_clamped = torch.clamp(torch.tensor(penalty_val, device=device, dtype=dtype_model), 0, getattr(self.args, 'max_g_easy_win_penalty_abs', 20.0))
                    loss_g_total_micro += penalty_clamped
                    losses_g_micro['loss_g_easy_win_penalty_micro'] = penalty_clamped.detach()

            loss_g_total_scaled_for_accum_micro = loss_g_total_micro / self.grad_accum_steps
            
        self.scaler_enc_gen.scale(loss_g_total_scaled_for_accum_micro).backward()
        
        losses_g_micro['loss_recon_micro'] = loss_recon_raw.detach()
        losses_g_micro['loss_kl_micro'] = loss_kl_raw.detach()
        losses_g_micro['loss_g_adv_micro'] = loss_g_adv_raw.detach()
        losses_g_micro['loss_g_total_micro'] = loss_g_total_micro.detach()

        # Prepare pixels for logging if needed (even if D is feature-based)
        final_pixels_for_logging = assembled_pixels_for_log # This was already prepared
        if final_pixels_for_logging is None and \
           self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and self.global_step > 0 and \
           ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0):
            if bboxes_used_by_decoder is not None:
                final_pixels_for_logging = self._assemble_pixels_from_spectral(
                    recon_dft_coeffs_gen.detach() if recon_dft_coeffs_gen is not None else None,
                    recon_dct_coeffs_gen.detach() if recon_dct_coeffs_gen is not None else None,
                    bboxes_used_by_decoder.detach(), # bboxes_used_by_decoder corresponds to G's output frames
                    target_image_height=self.args.image_h, target_image_width=self.args.image_w,
                    num_image_channels_target=self.video_config['num_channels']
                )
            else:
                 self.logger.warning_once("Cannot assemble pixels for logging in G-step (D is feature-based path) as bboxes_used_by_decoder is None.")

        return losses_g_micro, final_pixels_for_logging.detach() if final_pixels_for_logging is not None else None



    @staticmethod # Make it static as it's a utility
    def get_scale_from_action_value(action_dict: Optional[Dict[str, float]], key: str, default_value: float = 1.0) -> float:
        if action_dict is None: return default_value
        val = action_dict.get(key)
        return float(val) if val is not None and np.isfinite(val) else default_value



    def train(self, start_epoch: int = 0, initial_global_step: int = 0):
        self.global_step = initial_global_step
        self.current_epoch = start_epoch
        
        m_ref_dtype_check = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        model_param_list_for_dtype = list(m_ref_dtype_check.parameters())
        first_param_dtype = model_param_list_for_dtype[0] if model_param_list_for_dtype else None
        dtype_model = first_param_dtype.dtype if first_param_dtype is not None else torch.float32
        if self.am_main_process: self.logger.info(f"Trainer determined model dtype: {dtype_model}")

        if self.am_main_process:
            d_ref_active_log = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
            active_d_arch_variant_log = getattr(d_ref_active_log, 'architecture_variant', 'unknown_variant')
            self.logger.info(f"Starting training. DFT:{self.args.use_dft_features_appearance}, DCT:{self.args.use_dct_features_appearance}. "
                             f"Epochs: {self.args.epochs}, StartEpoch: {start_epoch}, GStep: {initial_global_step}, "
                             f"L_KL_base: {self.lambda_kl_base:.3e}. "
                             f"Initial Active D: {self.active_discriminator_key} (Arch: {active_d_arch_variant_log}, EffIn: {self.active_disc_effective_trainer_input_type})")
        
        if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and \
           self.args.num_val_samples_to_log > 0 and self.fixed_noise_for_sampling is None:
            if self.args.latent_dim > 0:
                self.fixed_noise_for_sampling = torch.randn(
                    self.args.num_val_samples_to_log, self.args.latent_dim, device=self.device, dtype=dtype_model
                )
            else:
                self.logger.warning("Cannot create fixed_noise_for_sampling as latent_dim is 0 or negative.")
            
        # Accumulators for logging over self.args.log_interval GLOBAL steps
        log_interval_global_steps_accum_losses = defaultdict(float)
        log_interval_global_steps_items_processed = 0 # Counts items over multiple global steps for averaging
        
        all_q_controllers_to_sync_lkl = [self.q_controller_gen, self.q_controller_d_primary, self.q_controller_d_alt, self.lambda_kl_q_controller]
        current_effective_lambda_kl = self.lambda_kl_base * self.heuristic_override_lambda_kl_factor
        for q_ctrl in all_q_controllers_to_sync_lkl:
            if q_ctrl and hasattr(q_ctrl, 'set_current_lambda_kl'):
                q_ctrl.set_current_lambda_kl(current_effective_lambda_kl)

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process:
                 d_ref_active_ep_log = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                 active_d_arch_variant_ep_log = getattr(d_ref_active_ep_log, 'architecture_variant', 'unknown_variant')
                 self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting "
                                  f"(L_KL_eff: {self.lambda_kl_base * self.heuristic_override_lambda_kl_factor:.3e}, "
                                  f"LRecF: {self.heuristic_override_lambda_recon_factor:.2f}, "
                                  f"LGanF: {self.heuristic_override_lambda_gan_factor:.2f}, "
                                  f"ActD: {self.active_discriminator_key} "
                                  f"[Arch:{active_d_arch_variant_ep_log}, EffIn:{self.active_disc_effective_trainer_input_type}]).")
            
            if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
                
            self.m_ref.train()
            if self.active_discriminator_key == 'primary':
                self.discriminator_primary_obj.train()
                if self.discriminator_alternative_obj is not None: self.discriminator_alternative_obj.eval()
            else: # 'alternative'
                self.discriminator_alternative_obj.train()
                if self.discriminator_primary_obj is not None: self.discriminator_primary_obj.eval()

            num_batches_epoch = len(self.train_loader)
            prog_bar_desc = f"E{epoch+1} ActD:{self.active_discriminator_key[0]}"
            prog_bar = tqdm(self.train_loader, desc=prog_bar_desc, 
                            disable=not self.am_main_process or os.getenv('CI') == 'true' or getattr(self.args, 'disable_train_tqdm', False),
                            dynamic_ncols=True, total=num_batches_epoch)
            
            accum_g_total_q_cycle, accum_g_recon_q_cycle, accum_g_kl_q_cycle, accum_g_adv_q_cycle = 0.0, 0.0, 0.0, 0.0
            accum_d_total_q_cycle, accum_d_real_q_cycle, accum_d_fake_q_cycle = 0.0, 0.0, 0.0
            num_micro_steps_in_accum_cycle = 0
            
            # Zero grads at the beginning of each accumulation cycle
            if self.optimizer_disc_active: self.optimizer_disc_active.zero_grad(set_to_none=True)
            if self.optimizer_enc_gen: self.optimizer_enc_gen.zero_grad(set_to_none=True)

            for batch_idx, batch_frames_raw in enumerate(prog_bar):
                batch_frames_full_sequence = batch_frames_raw.to(device=self.device, dtype=dtype_model, non_blocking=True)
                batch_size_micro = batch_frames_full_sequence.size(0)
                self.steps_since_last_d_switch += 1

                # --- Discriminator Training Micro-Step ---
                for p in self.active_discriminator.parameters(): p.requires_grad = True
                for p in self.m_ref.parameters(): p.requires_grad = False
                losses_d_micro = self._train_discriminator_step(batch_frames_full_sequence, self.m_ref)
                
                for k, v_tensor in losses_d_micro.items():
                    if torch.isfinite(v_tensor): 
                        val = v_tensor.item(); accum_key = k.replace('_micro', '_agg')
                        log_interval_global_steps_accum_losses[accum_key] += val * batch_size_micro
                        if k == 'loss_d_total_micro': self.interval_metrics_accum['d_total'] += val; accum_d_total_q_cycle += val
                        elif k == 'loss_d_real_micro': accum_d_real_q_cycle += val
                        elif k == 'loss_d_fake_micro': accum_d_fake_q_cycle += val
                
                # --- Generator/Encoder Training Micro-Step ---
                for p in self.m_ref.parameters(): p.requires_grad = True
                for p in self.active_discriminator.parameters(): p.requires_grad = False
                losses_g_micro, assembled_pixels_for_logging = self._train_generator_step(batch_frames_full_sequence, self.m_ref)
                
                for k, v_tensor in losses_g_micro.items():
                    if torch.isfinite(v_tensor): 
                        val = v_tensor.item(); accum_key = k.replace('_micro', '_agg')
                        log_interval_global_steps_accum_losses[accum_key] += val * batch_size_micro
                        if k == 'loss_recon_micro': self.interval_metrics_accum['recon_spectral'] += val; accum_g_recon_q_cycle += val
                        elif k == 'loss_kl_micro':  self.interval_metrics_accum['kl_div'] += val; accum_g_kl_q_cycle += val
                        elif k == 'loss_g_adv_micro': accum_g_adv_q_cycle += val
                        elif k == 'loss_g_total_micro': accum_g_total_q_cycle += val
                        elif k == 'loss_g_feat_match_micro' and self.heuristic_vae_feature_match_active:
                            log_interval_global_steps_accum_losses['loss_g_feat_match_eff_contrib_agg'] += (self.lambda_feat_match_heuristic * val * batch_size_micro)
                        elif k == 'loss_g_easy_win_penalty_micro' and self.heuristic_penalize_g_easy_win_active:
                            log_interval_global_steps_accum_losses['loss_g_easy_win_penalty_eff_contrib_agg'] += (val * batch_size_micro)
                
                log_interval_global_steps_items_processed += batch_size_micro
                self.interval_steps_count += 1
                num_micro_steps_in_accum_cycle +=1
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    d_active_ref_for_opt_step = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                    
                    if hasattr(self.optimizer_disc_active, 'grad_stats') and hasattr(self.optimizer_disc_active.grad_stats, 'finalize_step_stats'): # type: ignore
                         self.optimizer_disc_active.grad_stats.finalize_step_stats(sum(p.numel() for p in d_active_ref_for_opt_step.parameters() if p.requires_grad and p.grad is not None)) # type: ignore
                    if hasattr(self.optimizer_enc_gen, 'grad_stats') and hasattr(self.optimizer_enc_gen.grad_stats, 'finalize_step_stats'): # type: ignore
                         self.optimizer_enc_gen.grad_stats.finalize_step_stats(sum(p.numel() for p in self.m_ref.parameters() if p.requires_grad and p.grad is not None)) # type: ignore

                    avg_losses_for_q_cycle: Dict[str, Optional[float]] = {
                        'loss_g_total': accum_g_total_q_cycle / num_micro_steps_in_accum_cycle if num_micro_steps_in_accum_cycle > 0 else None,
                        'loss_g_recon': accum_g_recon_q_cycle / num_micro_steps_in_accum_cycle if num_micro_steps_in_accum_cycle > 0 else None,
                        'loss_g_kl': accum_g_kl_q_cycle / num_micro_steps_in_accum_cycle if num_micro_steps_in_accum_cycle > 0 else None,
                        'loss_g_adv': accum_g_adv_q_cycle / num_micro_steps_in_accum_cycle if num_micro_steps_in_accum_cycle > 0 else None,
                        'loss_d_total': accum_d_total_q_cycle / num_micro_steps_in_accum_cycle if num_micro_steps_in_accum_cycle > 0 else None,
                        'loss_d_real': accum_d_real_q_cycle / num_micro_steps_in_accum_cycle if num_micro_steps_in_accum_cycle > 0 else None,
                        'loss_d_fake': accum_d_fake_q_cycle / num_micro_steps_in_accum_cycle if num_micro_steps_in_accum_cycle > 0 else None,
                    }
                    
                    # --- Update Discriminator ---
                    for p_opt_d in d_active_ref_for_opt_step.parameters(): p_opt_d.requires_grad = True
                    for p_opt_g_frozen_for_d_step in self.m_ref.parameters(): p_opt_g_frozen_for_d_step.requires_grad = False

                    if self.q_controller_d_active and hasattr(self.optimizer_disc_active, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_disc_active.q_controller_update_and_set_hyperparams(avg_losses_for_q_cycle, self.lambda_kl) # type: ignore
                        if self.heuristic_boost_active_d_lr_active: 
                            for group in self.optimizer_disc_active.param_groups: group['lr'] = float(np.clip(group['lr'] * self.heuristic_active_d_lr_boost_factor, 1e-8, 1.0))
                    
                    if self.args.global_max_grad_norm > 0:
                        self.scaler_disc_active.unscale_(self.optimizer_disc_active)
                        torch.nn.utils.clip_grad_norm_(d_active_ref_for_opt_step.parameters(), self.args.global_max_grad_norm)
                    
                    self.scaler_disc_active.step(self.optimizer_disc_active)
                    self.scaler_disc_active.update()
                    
                    # --- Update Generator/Encoder ---
                    for p_opt_g in self.m_ref.parameters(): p_opt_g.requires_grad = True
                    for p_opt_d_frozen_for_g_step in d_active_ref_for_opt_step.parameters(): p_opt_d_frozen_for_g_step.requires_grad = False
                        
                    if self.q_controller_gen and hasattr(self.optimizer_enc_gen, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_enc_gen.q_controller_update_and_set_hyperparams(avg_losses_for_q_cycle, self.lambda_kl) # type: ignore

                    if self.args.global_max_grad_norm > 0:
                        self.scaler_enc_gen.unscale_(self.optimizer_enc_gen)
                        torch.nn.utils.clip_grad_norm_(self.m_ref.parameters(), self.args.global_max_grad_norm)
                    
                    self.scaler_enc_gen.step(self.optimizer_enc_gen)
                    self.scaler_enc_gen.update()
                    
                    # Zero grads for the START of the NEXT accumulation cycle
                    if self.optimizer_disc_active: self.optimizer_disc_active.zero_grad(set_to_none=True)
                    if self.optimizer_enc_gen: self.optimizer_enc_gen.zero_grad(set_to_none=True)
                    
                    self.global_step += 1
                    
                    accum_g_total_q_cycle, accum_g_recon_q_cycle, accum_g_kl_q_cycle, accum_g_adv_q_cycle = 0.0, 0.0, 0.0, 0.0
                    accum_d_total_q_cycle, accum_d_real_q_cycle, accum_d_fake_q_cycle = 0.0, 0.0, 0.0
                    num_micro_steps_in_accum_cycle = 0
                    
                    if self.global_step > 0:
                        if self.am_main_process:
                            if self.heuristic_check_interval > 0 and self.global_step % self.heuristic_check_interval == 0:
                                self._evaluate_training_state_and_apply_heuristics()
                            
                            if self.lambda_kl_q_controller is not None and self.lambda_kl_update_interval > 0 and \
                               self.global_step % self.lambda_kl_update_interval == 0 and self.interval_steps_count > 0:
                                current_interval_metrics: Dict[str, Union[float, None]] = {
                                    'avg_recon': self.interval_metrics_accum['recon_spectral'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                                    'avg_kl_div': self.interval_metrics_accum['kl_div'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                                    'avg_d_total': self.interval_metrics_accum['d_total'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                                    'val_metric': self.last_val_metrics.get(self.args.val_primary_metric), 
                                    'current_lambda_kl_val': self.lambda_kl_base
                                }
                                prog_bar.write(f"GStep {self.global_step}: LKL Q-Ctrl. LKL_Base: {self.lambda_kl_base:.4e}. Metrics: {{ {', '.join([f'{k}: {v:.3f}' if isinstance(v,float) and np.isfinite(v) else f'{k}: {str(v)}' for k,v in current_interval_metrics.items()])} }}")
                                q_state_lambda_kl = self.lambda_kl_q_controller.get_lambda_kl_state(current_interval_metrics)
                                if self.lambda_kl_q_controller.prev_lambda_kl_state is not None and \
                                   self.lambda_kl_q_controller.prev_lambda_kl_action is not None and \
                                   q_state_lambda_kl is not None and \
                                   self.prev_interval_metrics_for_lambda_kl_reward is not None:
                                    reward_for_lambda_kl = self.lambda_kl_q_controller.compute_lambda_kl_reward(current_interval_metrics, self.prev_interval_metrics_for_lambda_kl_reward)
                                    prog_bar.write(f"  LKL Q-Ctrl reward: {reward_for_lambda_kl:.3f}")
                                    self.lambda_kl_q_controller.update_q_values(self.lambda_kl_q_controller.prev_lambda_kl_state, self.lambda_kl_q_controller.prev_lambda_kl_action, reward_for_lambda_kl, q_state_lambda_kl, mode='lambda_kl')
                                elif q_state_lambda_kl is not None and hasattr(self.lambda_kl_q_controller, 'set_initial_lambda_kl_metrics'):
                                     self.lambda_kl_q_controller.set_initial_lambda_kl_metrics(current_interval_metrics)
                                
                                if q_state_lambda_kl is not None:
                                    lambda_kl_action_dict = self.lambda_kl_q_controller.choose_action(q_state_lambda_kl, mode='lambda_kl')
                                    chosen_scale = HybridTrainer.get_scale_from_action_value(lambda_kl_action_dict, 'lambda_kl_scale', 1.0)
                                    self.lambda_kl_base = float(np.clip(self.lambda_kl_base * chosen_scale, self.min_lambda_kl_q_control, self.max_lambda_kl_q_control)) 
                                    prog_bar.write(f"  LKL_Q-Ctrl CHOSE scale: {chosen_scale:.3f}. New LKL_Base: {self.lambda_kl_base:.4e}")
                                    self.lambda_kl_q_controller.prev_lambda_kl_state = q_state_lambda_kl
                                    self.lambda_kl_q_controller.prev_lambda_kl_action = lambda_kl_action_dict
                                
                                self.prev_interval_metrics_for_lambda_kl_reward = current_interval_metrics.copy()
                                self.lambda_kl = self.lambda_kl_base * self.heuristic_override_lambda_kl_factor
                                for q_ctrl_opt_sync in all_q_controllers_to_sync_lkl: 
                                    if q_ctrl_opt_sync and hasattr(q_ctrl_opt_sync, 'set_current_lambda_kl'):
                                        q_ctrl_opt_sync.set_current_lambda_kl(self.lambda_kl)
                                self.interval_metrics_accum = defaultdict(float); self.interval_steps_count = 0

                            if self.args.log_interval > 0 and self.global_step % self.args.log_interval == 0 and \
                               log_interval_global_steps_items_processed > 0 :
                                current_log_metrics_wandb: Dict[str, Any] = {}
                                for k_log, v_sum_log in log_interval_global_steps_accum_losses.items(): 
                                    wandb_key_base = k_log.replace('_eff_contrib_agg', '_eff_contrib').replace('_agg', '')
                                    current_log_metrics_wandb[f"train/{wandb_key_base}"] = v_sum_log / log_interval_global_steps_items_processed
                                
                                avg_raw_recon_feat_log = current_log_metrics_wandb.get('train/loss_recon', float('nan'))
                                avg_raw_kl_log = current_log_metrics_wandb.get('train/loss_kl', float('nan'))
                                avg_raw_g_adv_log = current_log_metrics_wandb.get('train/loss_g_adv', float('nan'))
                                avg_raw_d_total_log = current_log_metrics_wandb.get('train/loss_d_total', float('nan'))
                                avg_raw_d_real_log = current_log_metrics_wandb.get('train/loss_d_real', float('nan'))
                                avg_raw_d_fake_log = current_log_metrics_wandb.get('train/loss_d_fake', float('nan'))

                                eff_recon_log = avg_raw_recon_feat_log * self.args.lambda_recon * self.heuristic_override_lambda_recon_factor
                                eff_kl_log = avg_raw_kl_log * self.lambda_kl
                                eff_gan_log = avg_raw_g_adv_log * self.lambda_gan_base * self.heuristic_override_lambda_gan_factor
                                current_log_metrics_wandb.update({ "train/lambda_recon_eff_contrib_calc": eff_recon_log, "train/lambda_kl_eff_contrib_calc": eff_kl_log, "train/lambda_gan_eff_contrib_calc": eff_gan_log, "train/lambda_kl_eff_actual_val": self.lambda_kl, "train/lambda_kl_base_val_actual": self.lambda_kl_base, "train/lambda_recon_factor_heur": self.heuristic_override_lambda_recon_factor, "train/lambda_kl_factor_heur": self.heuristic_override_lambda_kl_factor, "train/lambda_gan_factor_heur": self.heuristic_override_lambda_gan_factor,})
                                loss_g_feat_match_contrib_log = current_log_metrics_wandb.get('train/loss_g_feat_match_eff_contrib', 0.0)
                                loss_g_easy_win_penalty_contrib_log = current_log_metrics_wandb.get('train/loss_g_easy_win_penalty_eff_contrib', 0.0)
                                calculated_g_total_for_log = eff_recon_log + eff_kl_log + eff_gan_log + loss_g_feat_match_contrib_log + loss_g_easy_win_penalty_contrib_log
                                current_log_metrics_wandb["train/loss_g_total_calculated_log"] = calculated_g_total_for_log
                                lr_g_log = self.optimizer_enc_gen.param_groups[0]['lr'] if self.optimizer_enc_gen else -1.0
                                lr_d_active_log = self.optimizer_disc_active.param_groups[0]['lr'] if self.optimizer_disc_active else -1.0
                                
                                d_ref_log_console = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                                active_d_arch_variant_console_log = getattr(d_ref_log_console, 'architecture_variant', 'unk')[:7]
                                active_d_eff_in_console_log = self.active_disc_effective_trainer_input_type.split('_')[0][:4]
                                current_log_metrics_wandb.update({"train/lr_gen": lr_g_log, f"train/lr_disc_{self.active_discriminator_key}_{active_d_arch_variant_console_log[:3]}_{active_d_eff_in_console_log[:3]}": lr_d_active_log, "epoch_frac": epoch + ((batch_idx + 1) / max(1, num_batches_epoch)), "global_step": self.global_step, f"active_disc_is_primary_val": 1 if self.active_discriminator_key == 'primary' else 0})
                                q_controller_info_map_log = {"gen": self.q_controller_gen, f"d_{self.active_discriminator_key}": self.q_controller_d_active, "lkl": self.lambda_kl_q_controller }
                                for prefix_log, controller_obj_log in q_controller_info_map_log.items(): 
                                    if controller_obj_log and hasattr(controller_obj_log, 'get_info'):
                                        for k_info_log, v_info_log in controller_obj_log.get_info().items(): clean_k_info_log = ''.join(c if c.isalnum() or c in ['_', '/'] else '_' for c in str(k_info_log)).lower().replace('lrmom','').replace('lambdakl',''); current_log_metrics_wandb[f"q_info/{prefix_log}/{clean_k_info_log}"] = v_info_log
                                        if hasattr(controller_obj_log, 'prev_lr_mom_action') and controller_obj_log.prev_lr_mom_action: current_log_metrics_wandb[f"q_actions/{prefix_log}/lr_scale"] = HybridTrainer.get_scale_from_action_value(controller_obj_log.prev_lr_mom_action, 'lr_scale'); current_log_metrics_wandb[f"q_actions/{prefix_log}/mom_scale"] = HybridTrainer.get_scale_from_action_value(controller_obj_log.prev_lr_mom_action, 'momentum_scale')
                                        if hasattr(controller_obj_log, 'prev_lambda_kl_action') and controller_obj_log.prev_lambda_kl_action: current_log_metrics_wandb[f"q_actions/{prefix_log}/lkl_scale"] = HybridTrainer.get_scale_from_action_value(controller_obj_log.prev_lambda_kl_action, 'lambda_kl_scale')
                                current_log_metrics_wandb.update({"heuristic/vae_fm_active_val": 1 if self.heuristic_vae_feature_match_active else 0, "heuristic/pen_g_ez_win_val": 1 if self.heuristic_penalize_g_easy_win_active else 0, "heuristic/d_lr_boost_active_val": 1 if self.heuristic_boost_active_d_lr_active else 0, "heuristic/lrec_factor_val": self.heuristic_override_lambda_recon_factor, "heuristic/lkl_factor_val": self.heuristic_override_lambda_kl_factor, "heuristic/lgan_factor_val": self.heuristic_override_lambda_gan_factor, "heuristic/lambda_feat_match_heuristic_val": self.lambda_feat_match_heuristic if self.heuristic_vae_feature_match_active else 0.0, "heuristic/lambda_g_easy_win_penalty_heuristic_val": self.lambda_g_easy_win_penalty_heuristic if self.heuristic_penalize_g_easy_win_active else 0.0, "heuristic/active_d_lr_boost_factor_applied": self.heuristic_active_d_lr_boost_factor if self.heuristic_boost_active_d_lr_active else 1.0, "heuristic/d_q_explore_active_val": 1 if (self.q_controller_d_active and hasattr(self.q_controller_d_active, 'epsilon_boost_active_steps') and self.q_controller_d_active.epsilon_boost_active_steps > 0) else 0, "heuristic/rec_features_stagnant_val": 1 if self.rec_features_stagnant else 0, "disc_switch/steps_since_last_switch": self.steps_since_last_d_switch, "disc_switch/p_to_a_trigger_count": self.consecutive_trigger_primary_to_alt_count, "disc_switch/a_to_p_trigger_count": self.consecutive_trigger_alt_to_primary_count })
                                for k_heur_trig, v_heur_trig in self.consecutive_heuristic_trigger_counts.items(): current_log_metrics_wandb[f"heuristic/trigger_count/{k_heur_trig}"] = v_heur_trig
                                if self.global_step % (self.args.log_interval * 5) == 0:
                                    if self.optimizer_enc_gen and hasattr(self.optimizer_enc_gen, 'get_gradient_stats_summary_optimizer_view'): current_log_metrics_wandb.update({f"grad_stats/gen/{k}": v for k,v in self.optimizer_enc_gen.get_gradient_stats_summary_optimizer_view().items()}) # type: ignore
                                    if self.optimizer_disc_active and hasattr(self.optimizer_disc_active, 'get_gradient_stats_summary_optimizer_view'): current_log_metrics_wandb.update({f"grad_stats/d_active_{self.active_discriminator_key}/{k}": v for k,v in self.optimizer_disc_active.get_gradient_stats_summary_optimizer_view().items()}) # type: ignore
                                
                                gen_q_eps_str = f"{self.q_controller_gen.epsilon:.2f}" if self.q_controller_gen else "N/A"
                                d_act_q_eps_str = f"{self.q_controller_d_active.epsilon:.2f}" if self.q_controller_d_active else "N/A"
                                lkl_q_eps_str = f"{self.lambda_kl_q_controller.epsilon:.2f}" if self.lambda_kl_q_controller else "N/A"
                                gen_q_last_lr_s = HybridTrainer.get_scale_from_action_value(getattr(self.q_controller_gen, 'prev_lr_mom_action', None), 'lr_scale', -1.0)
                                d_q_last_lr_s = HybridTrainer.get_scale_from_action_value(getattr(self.q_controller_d_active, 'prev_lr_mom_action', None), 'lr_scale', -1.0)
                                lkl_q_last_lkl_s = HybridTrainer.get_scale_from_action_value(getattr(self.lambda_kl_q_controller, 'prev_lambda_kl_action', None), 'lambda_kl_scale', -1.0)
                                q_scales_str = f"QSc(G:{gen_q_last_lr_s:.1f}|D:{d_q_last_lr_s:.1f}|LKL:{lkl_q_last_lkl_s:.1f})"
                                heur_flags_short = [ f"FM(x{self.lambda_feat_match_heuristic:.1f})" if self.heuristic_vae_feature_match_active else "", f"GPEW(x{self.lambda_g_easy_win_penalty_heuristic:.1f})" if self.heuristic_penalize_g_easy_win_active else "", f"DLRB(x{self.heuristic_active_d_lr_boost_factor:.1f})" if self.heuristic_boost_active_d_lr_active else "", f"DQE({self.q_controller_d_active.epsilon_boost_active_steps})" if self.q_controller_d_active and hasattr(self.q_controller_d_active, 'epsilon_boost_active_steps') and self.q_controller_d_active.epsilon_boost_active_steps > 0 else "", "RecStag" if self.rec_features_stagnant else "" ]
                                heur_flags_console_str = f"H:[{','.join(filter(None, heur_flags_short))}]" if any(filter(None, heur_flags_short)) else ""
                                lambda_factors_str = f"LF(R:{self.heuristic_override_lambda_recon_factor:.1f}|K:{self.heuristic_override_lambda_kl_factor:.1f}|A:{self.heuristic_override_lambda_gan_factor:.1f})"
                                log_str_console = (f"E{epoch+1} S{self.global_step} ActD:{self.active_discriminator_key[0]}:{active_d_arch_variant_console_log[:3]}:{active_d_eff_in_console_log[:3]} "
                                    f"| G_Tot:{calculated_g_total_for_log:.2f}(R:{eff_recon_log:.2f}[{avg_raw_recon_feat_log:.2f}] K:{eff_kl_log:.2f}[{avg_raw_kl_log:.2f}] A:{eff_gan_log:.2f}[{avg_raw_g_adv_log:.2f}]"
                                    + (f" FM:{loss_g_feat_match_contrib_log:.2f}" if self.heuristic_vae_feature_match_active and loss_g_feat_match_contrib_log!=0 else "")
                                    + (f" GPen:{loss_g_easy_win_penalty_contrib_log:.2f}" if self.heuristic_penalize_g_easy_win_active and loss_g_easy_win_penalty_contrib_log!=0 else "")
                                    + f") | D_Tot:{avg_raw_d_total_log:.2f}(Rl:{avg_raw_d_real_log:.2f} Fk:{avg_raw_d_fake_log:.2f})"
                                    f" | LR(G/D):{lr_g_log:.1e}/{lr_d_active_log:.1e} | LKL_eff:{self.lambda_kl:.2e} "
                                    f"Qε(G:{gen_q_eps_str} D:{d_act_q_eps_str} L:{lkl_q_eps_str}) {q_scales_str} "
                                    f"DSw(P>A:{self.consecutive_trigger_primary_to_alt_count},A>P:{self.consecutive_trigger_alt_to_primary_count}|St:{self.steps_since_last_d_switch}) " # Shortened DSw
                                    f"{lambda_factors_str} {heur_flags_console_str}")
                                
                                prog_bar.set_postfix_str(f"ActD:{self.active_discriminator_key[0]} G:{calculated_g_total_for_log:.2f} D:{avg_raw_d_total_log:.2f} RecFRaw:{avg_raw_recon_feat_log:.3f}", refresh=True)
                                prog_bar.write(log_str_console)
                                if self.args.wandb and WANDB_AVAILABLE and wandb.run: wandb.log(current_log_metrics_wandb, step=self.global_step)
                                
                                log_interval_global_steps_accum_losses = defaultdict(float)
                                log_interval_global_steps_items_processed = 0
                            
                            if assembled_pixels_for_logging is not None and self.args.wandb_log_train_recon_interval > 0 and \
                               self.global_step % self.args.wandb_log_train_recon_interval == 0:
                                self._log_samples_to_wandb("train_recon_pixels", assembled_pixels_for_logging,
                                   num_frames_per_sequence_to_log=min(assembled_pixels_for_logging.shape[1], getattr(self.args, 'wandb_log_num_frames_train', 3)),
                                   num_sequences_to_log_max=self.args.num_val_samples_to_log)
                                num_input_frames_log = self.video_config.get("num_input_frames", 0)
                                if num_input_frames_log > 0:
                                    self._log_samples_to_wandb("train_context_pixels", batch_frames_full_sequence[:, :num_input_frames_log, ...],
                                       num_frames_per_sequence_to_log=min(num_input_frames_log, getattr(self.args, 'wandb_log_num_frames_context',3)),
                                       num_sequences_to_log_max=self.args.num_val_samples_to_log)
                                s_idx_target_log = num_input_frames_log
                                gt_len_log = min(self.video_config.get("num_predict_frames", 1), batch_frames_full_sequence.shape[1] - s_idx_target_log)
                                if gt_len_log > 0:
                                    self._log_samples_to_wandb("train_ground_truth_pixels", batch_frames_full_sequence[:, s_idx_target_log : s_idx_target_log + gt_len_log, ...],
                                       num_frames_per_sequence_to_log=gt_len_log, # Log all predicted GT frames for this sequence
                                       num_sequences_to_log_max=self.args.num_val_samples_to_log)

                            if self.fixed_noise_for_sampling is not None and self.args.wandb_log_fixed_noise_samples_interval > 0 and \
                               self.global_step % self.args.wandb_log_fixed_noise_samples_interval == 0:
                                fixed_noise_pixels = self.sample(self.args.num_val_samples_to_log, noise=self.fixed_noise_for_sampling)
                                if fixed_noise_pixels is not None:
                                    self._log_samples_to_wandb("fixed_noise_generated_pixels", fixed_noise_pixels,
                                       num_frames_per_sequence_to_log=min(fixed_noise_pixels.shape[1], getattr(self.args, 'wandb_log_num_frames_fixed', 3)),
                                       num_sequences_to_log_max=self.args.num_val_samples_to_log)
                            
                            if self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0:
                                self._save_checkpoint(is_intermediate=True, metrics=avg_losses_for_q_cycle if 'avg_losses_for_q_cycle' in locals() else None) # type: ignore
            
            # --- End of Epoch Actions ---
            if self.am_main_process:
                final_avg_g_loss_eoe = log_interval_global_steps_accum_losses['loss_g_total_agg'] / log_interval_global_steps_items_processed if log_interval_global_steps_items_processed > 0 else \
                                     (avg_losses_for_q_cycle.get('loss_g_total', float('nan')) if 'avg_losses_for_q_cycle' in locals() and avg_losses_for_q_cycle is not None else float('nan')) # type: ignore
                final_avg_d_loss_eoe = log_interval_global_steps_accum_losses['loss_d_total_agg'] / log_interval_global_steps_items_processed if log_interval_global_steps_items_processed > 0 else \
                                     (avg_losses_for_q_cycle.get('loss_d_total', float('nan')) if 'avg_losses_for_q_cycle' in locals() and avg_losses_for_q_cycle is not None else float('nan')) # type: ignore
                
                self.logger.info(f"Epoch {epoch+1} finished. Approx Avg Loss (G/D): {final_avg_g_loss_eoe:.4f}/{final_avg_d_loss_eoe:.4f}, LKL_Eff:{self.lambda_kl_base * self.heuristic_override_lambda_kl_factor:.3e}")
                if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                    wandb.log({"epoch": epoch + 1, 
                               "epoch_avg_train_loss_g_approx": final_avg_g_loss_eoe if np.isfinite(final_avg_g_loss_eoe) else -1.0, 
                               "epoch_avg_train_loss_d_approx": final_avg_d_loss_eoe if np.isfinite(final_avg_d_loss_eoe) else -1.0, 
                               "epoch_lambda_kl_eff_eoe": self.lambda_kl_base * self.heuristic_override_lambda_kl_factor,
                               "epoch_lambda_kl_base_eoe": self.lambda_kl_base}, 
                              step=self.global_step)
                              
            validation_interval_epochs = getattr(self.args, 'validation_interval_epochs', 1)
            if self.val_loader and self.am_main_process and validation_interval_epochs > 0 and (epoch + 1) % validation_interval_epochs == 0:
                val_metrics_eoe = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log)
                if val_metrics_eoe:
                    finite_val_metrics_eoe = {f"val/{k_val}": v_val for k_val, v_val in val_metrics_eoe.items() if v_val is not None and np.isfinite(v_val)}
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                        wandb.log(finite_val_metrics_eoe, step=self.global_step)
                    
                    metric_to_check_eoe = self.args.val_primary_metric
                    current_val_for_best_eoe: Optional[float] = val_metrics_eoe.get(metric_to_check_eoe)

                    if current_val_for_best_eoe is not None and np.isfinite(current_val_for_best_eoe):
                        is_better_eoe_flag = False # Default to False
                        if not np.isfinite(self.best_val_metric_val): is_better_eoe_flag = True # First valid metric is always best
                        elif self.is_val_metric_higher_better: is_better_eoe_flag = (current_val_for_best_eoe > self.best_val_metric_val)
                        else: is_better_eoe_flag = (current_val_for_best_eoe < self.best_val_metric_val)
                        
                        if is_better_eoe_flag:
                            prog_bar.write(f"New best val metric ({metric_to_check_eoe}): {current_val_for_best_eoe:.4f} (prev: {self.best_val_metric_val:.4f}). Saving best checkpoint.")
                            self.best_val_metric_val = current_val_for_best_eoe
                            self._save_checkpoint(is_best=True, metrics=val_metrics_eoe)
            
            save_epoch_interval_epochs = getattr(self.args, 'save_epoch_interval', 1)
            if self.am_main_process and save_epoch_interval_epochs > 0 and (epoch + 1) % save_epoch_interval_epochs == 0:
                already_saved_as_best_this_epoch = 'is_better_eoe_flag' in locals() and locals().get('is_better_eoe_flag', False)
                is_last_micro_batch_of_epoch = (batch_idx + 1) == num_batches_epoch
                was_optimizer_step_this_micro_batch = (batch_idx + 1) % self.grad_accum_steps == 0
                already_saved_as_intermediate_this_step = self.args.save_interval > 0 and self.global_step > 0 and \
                                                        self.global_step % self.args.save_interval == 0 and \
                                                        is_last_micro_batch_of_epoch and was_optimizer_step_this_micro_batch
                
                if not (already_saved_as_best_this_epoch or already_saved_as_intermediate_this_step):
                    eoe_metrics_for_save = self.last_val_metrics.copy() if self.last_val_metrics else {}
                    g_loss_save = final_avg_g_loss_eoe if 'final_avg_g_loss_eoe' in locals() and np.isfinite(final_avg_g_loss_eoe) else -1.0 # type: ignore
                    d_loss_save = final_avg_d_loss_eoe if 'final_avg_d_loss_eoe' in locals() and np.isfinite(final_avg_d_loss_eoe) else -1.0 # type: ignore
                    eoe_metrics_for_save["epoch_end_train_g_total_approx"] = g_loss_save
                    eoe_metrics_for_save["epoch_end_train_d_total_approx"] = d_loss_save
                    
                    self.logger.info(f"Saving end-of-epoch checkpoint for epoch {epoch+1} at GStep {self.global_step}.")
                    self._save_checkpoint(metrics=eoe_metrics_for_save)




    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1) -> Optional[Dict[str, float]]:
        if not self.val_loader or not self.am_main_process:
            return None
        
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        original_training_mode_m = m_ref.training
        m_ref.eval() # Set model to evaluation mode

        # Initialize accumulators for metrics
        total_recon_dft_mse_sum = 0.0; num_dft_loss_batches = 0
        total_recon_dct_mse_sum = 0.0; num_dct_loss_batches = 0
        total_pixel_mse_sum = 0.0; num_pixel_recon_loss_batches = 0 # For pixel-based recon loss (if G outputs pixels)
        
        total_psnr_sum = 0.0; total_ssim_sum = 0.0; total_lpips_sum = 0.0
        num_pixel_metric_eval_frames_count = 0 # Counts individual frames evaluated for pixel metrics

        # Determine model's expected dtype more robustly
        model_param_iter = iter(m_ref.parameters())
        first_param = next(model_param_iter, None)
        dtype_m = first_param.dtype if first_param is not None else torch.float32
        
        logged_samples_count_this_val = 0

        for batch_idx_val, batch_frames_raw in enumerate(
            tqdm(self.val_loader, desc="Validating", 
                 disable=(not self.am_main_process or os.getenv('CI') == 'true' or getattr(self.args, 'disable_val_tqdm', False)), 
                 dynamic_ncols=True)
        ):
            real_full_pixel_sequence = batch_frames_raw.to(self.device, dtype=dtype_m)
            B, N_total_sample, C_img, H_img_actual, W_img_actual = real_full_pixel_sequence.shape # Actual H, W from data
            
            # Model forward pass:
            # recon_pixel_gen, recon_dft_gen, recon_dct_gen are (B, N_gen_pred, ...)
            # bboxes_used_by_decoder is (B, N_gen_pred, N_reg, 4)
            # target_dft/dct_for_loss are (B, N_gen_pred, N_reg, D_spectral_flat)
            recon_pixel_gen, recon_dft_gen, recon_dct_gen, \
            _mu_val, _logvar_val, bboxes_used_by_decoder, \
            target_dft_for_loss, target_dct_for_loss = m_ref(real_full_pixel_sequence)
            
            # Number of frames G is configured to predict, should match recon_...gen.shape[1]
            num_gen_predict_frames = self.video_config.get("num_predict_frames", 1) 
            num_input_f_conditioning = self.video_config.get("num_input_frames", 0)

            # --- Spectral Reconstruction Loss Calculation (if applicable) ---
            if self.args.use_dft_features_appearance and recon_dft_gen is not None and target_dft_for_loss is not None:
                if recon_dft_gen.shape == target_dft_for_loss.shape:
                    loss_dft_batch = F.mse_loss(recon_dft_gen.float(), target_dft_for_loss.float())
                    if torch.isfinite(loss_dft_batch): 
                        total_recon_dft_mse_sum += loss_dft_batch.item() * B # Accumulate sum of losses
                        num_dft_loss_batches += B # Count items
                else:
                    self.logger.warning_once(f"Val DFT shape mismatch: Recon {recon_dft_gen.shape}, Target {target_dft_for_loss.shape}")
            
            if self.args.use_dct_features_appearance and recon_dct_gen is not None and target_dct_for_loss is not None:
                if recon_dct_gen.shape == target_dct_for_loss.shape:
                    loss_dct_batch = F.mse_loss(recon_dct_gen.float(), target_dct_for_loss.float())
                    if torch.isfinite(loss_dct_batch): 
                        total_recon_dct_mse_sum += loss_dct_batch.item() * B
                        num_dct_loss_batches += B
                else:
                    self.logger.warning_once(f"Val DCT shape mismatch: Recon {recon_dct_gen.shape}, Target {target_dct_for_loss.shape}")
            
            # --- Pixel-space Metrics Calculation ---
            # Ground truth pixels for the prediction window
            gt_pixels_for_metrics = real_full_pixel_sequence[
                :, num_input_f_conditioning : num_input_f_conditioning + num_gen_predict_frames, ...
            ] # (B, N_gen_pred, C, H_actual, W_actual)
            
            predicted_pixels_for_metrics: Optional[torch.Tensor] = None
            if recon_pixel_gen is not None: # Generator outputted pixels directly
                predicted_pixels_for_metrics = recon_pixel_gen
            elif self.args.use_dft_features_appearance or self.args.use_dct_features_appearance: # Assemble from spectral
                if bboxes_used_by_decoder is None:
                    self.logger.error_once("Validate: bboxes_used_by_decoder is None from model.forward(), cannot assemble pixels for validation metrics.")
                else:
                    predicted_pixels_for_metrics = self._assemble_pixels_from_spectral(
                        recon_dft_gen, recon_dct_gen, bboxes_used_by_decoder,
                        target_image_height=self.args.image_h,         # Use configured target H
                        target_image_width=self.args.image_w,          # Use configured target W
                        num_image_channels_target=self.video_config['num_channels']
                        # output_range is defaulted in _assemble_pixels_from_spectral
                    )
            
            if predicted_pixels_for_metrics is not None:
                # Ensure predicted frames and ground truth frames for metrics have the same temporal length
                # This should be num_gen_predict_frames
                N_frames_eval_metrics = predicted_pixels_for_metrics.shape[1]
                gt_pixels_for_metrics_eff = gt_pixels_for_metrics[:, :N_frames_eval_metrics, ...]

                if predicted_pixels_for_metrics.shape == gt_pixels_for_metrics_eff.shape:
                    # Pixel MSE Loss (only if this is the primary reconstruction target, i.e., not using spectral)
                    if not (self.args.use_dft_features_appearance or self.args.use_dct_features_appearance):
                        pixel_mse_batch = F.mse_loss(predicted_pixels_for_metrics.float(), gt_pixels_for_metrics_eff.float())
                        if torch.isfinite(pixel_mse_batch): 
                            total_pixel_mse_sum += pixel_mse_batch.item() * B
                            num_pixel_recon_loss_batches += B
                    
                    # --- Standard Pixel-Space Quality Metrics (PSNR, SSIM, LPIPS) ---
                    # Reshape to (B*N_frames_eval, C, H, W) for metric functions
                    pred_for_metrics_flat = predicted_pixels_for_metrics.reshape(-1, C_img, H_img_actual, W_img_actual)
                    gt_for_metrics_flat = gt_pixels_for_metrics_eff.reshape(-1, C_img, H_img_actual, W_img_actual)
                    
                    # Normalize to [0,1] for PSNR/SSIM if they are in [-1,1]
                    pred_01 = (pred_for_metrics_flat.clamp(-1,1) + 1) / 2.0
                    gt_01 = (gt_for_metrics_flat.clamp(-1,1) + 1) / 2.0
                    
                    current_batch_num_indiv_frames_for_metric = pred_for_metrics_flat.shape[0] # B * N_frames_eval_metrics

                    # PSNR (per frame, then sum)
                    mse_for_psnr_per_frame = F.mse_loss(pred_01, gt_01, reduction='none').mean(dim=[1,2,3]) # MSE per frame
                    psnr_per_frame = 10 * torch.log10(1.0 / (mse_for_psnr_per_frame + EPS))
                    psnr_per_frame_clamped = torch.clamp(psnr_per_frame, 0, 100) # Clip PSNR to a reasonable range
                    if torch.isfinite(psnr_per_frame_clamped.sum()):
                        total_psnr_sum += psnr_per_frame_clamped.sum().item()
                    
                    # SSIM (per frame, then sum)
                    if self.ssim_metric:
                        try:
                            # ssim_metric expects (N, C, H, W) and data_range (default 1.0 if input is [0,1])
                            ssim_val_batch = self.ssim_metric(pred_01, gt_01) # Returns (N,)
                            if torch.isfinite(ssim_val_batch.sum()):
                                total_ssim_sum += ssim_val_batch.sum().item()
                        except Exception as e_ssim_val:
                            self.logger.debug(f"Val SSIM calculation failed: {e_ssim_val}")
                    
                    # LPIPS (per frame, then sum)
                    if self.lpips_loss_fn:
                        try:
                            # LPIPS expects (N, C, H, W) in range [-1, 1] typically
                            lpips_pred_input = pred_for_metrics_flat # Already in [-1,1] before 0-1 normalization for PSNR/SSIM
                            lpips_gt_input = gt_for_metrics_flat

                            if C_img == 1 and self.video_config['num_channels'] == 1: # Ensure LPIPS gets 3 channels if it expects it (common for AlexNet based)
                                lpips_pred_input = lpips_pred_input.repeat(1,3,1,1)
                                lpips_gt_input = lpips_gt_input.repeat(1,3,1,1)
                            
                            lpips_val_batch = self.lpips_loss_fn(lpips_pred_input, lpips_gt_input) # Returns (N,1,1,1), squeeze
                            if torch.isfinite(lpips_val_batch.sum()):
                                total_lpips_sum += lpips_val_batch.sum().item()
                        except Exception as e_lpips_val:
                            self.logger.debug(f"Val LPIPS calculation failed: {e_lpips_val}")
                    
                    num_pixel_metric_eval_frames_count += current_batch_num_indiv_frames_for_metric

                else: # Shape mismatch between predicted and GT pixels for metrics
                    self.logger.warning_once(f"Val Pixel Metric shape mismatch: Pred {predicted_pixels_for_metrics.shape}, GT_eff {gt_pixels_for_metrics_eff.shape}")

                # Log predicted and GT samples to WandB
                if logged_samples_count_this_val < num_val_samples_to_log and self.args.wandb and WANDB_AVAILABLE and wandb.run is not None:
                    num_sequences_to_log_this_batch = min(B, num_val_samples_to_log - logged_samples_count_this_val)
                    num_frames_to_log_per_seq = min(predicted_pixels_for_metrics.shape[1], 3) # Log up to 3 predicted frames

                    if num_sequences_to_log_this_batch > 0 and num_frames_to_log_per_seq > 0:
                        # Context frames from input
                        num_context_to_log = min(num_input_f_conditioning, 3)
                        if num_input_f_conditioning > 0 and num_context_to_log > 0:
                             self._log_samples_to_wandb("val_context_frames",
                                                       real_full_pixel_sequence[:num_sequences_to_log_this_batch, :num_context_to_log, ...],
                                                       num_frames_per_sequence_to_log=num_context_to_log, # Pass correct arg name
                                                       num_sequences_to_log_max=num_sequences_to_log_this_batch)
                        
                        # Predicted frames
                        self._log_samples_to_wandb("val_predicted_frames",
                                                   predicted_pixels_for_metrics[:num_sequences_to_log_this_batch, :num_frames_to_log_per_seq, ...],
                                                   num_frames_per_sequence_to_log=num_frames_to_log_per_seq,
                                                   num_sequences_to_log_max=num_sequences_to_log_this_batch)
                        
                        # Ground truth frames corresponding to prediction
                        self._log_samples_to_wandb("val_ground_truth_frames",
                                                   gt_pixels_for_metrics_eff[:num_sequences_to_log_this_batch, :num_frames_to_log_per_seq, ...],
                                                   num_frames_per_sequence_to_log=num_frames_to_log_per_seq,
                                                   num_sequences_to_log_max=num_sequences_to_log_this_batch)
                    logged_samples_count_this_val += num_sequences_to_log_this_batch
        
        m_ref.train(original_training_mode_m) # Restore model's original training mode
        
        metrics = {}
        if num_dft_loss_batches > 0: metrics["avg_val_recon_mse_dft"] = total_recon_dft_mse_sum / num_dft_loss_batches
        else: metrics["avg_val_recon_mse_dft"] = float('nan') # Report NaN if no batches
            
        if num_dct_loss_batches > 0: metrics["avg_val_recon_mse_dct"] = total_recon_dct_mse_sum / num_dct_loss_batches
        else: metrics["avg_val_recon_mse_dct"] = float('nan')

        if num_pixel_recon_loss_batches > 0: metrics["avg_val_recon_mse_pixel"] = total_pixel_mse_sum / num_pixel_recon_loss_batches
        else: metrics["avg_val_recon_mse_pixel"] = float('nan')
            
        if num_pixel_metric_eval_frames_count > 0:
            metrics["avg_val_psnr"] = total_psnr_sum / num_pixel_metric_eval_frames_count
            if self.ssim_metric: metrics["avg_val_ssim"] = total_ssim_sum / num_pixel_metric_eval_frames_count
            else: metrics["avg_val_ssim"] = float('nan')
            if self.lpips_loss_fn: metrics["avg_val_lpips"] = total_lpips_sum / num_pixel_metric_eval_frames_count
            else: metrics["avg_val_lpips"] = float('nan')
        else: # No frames evaluated for pixel metrics
            metrics["avg_val_psnr"] = float('nan')
            metrics["avg_val_ssim"] = float('nan')
            metrics["avg_val_lpips"] = float('nan')
        
        self.last_val_metrics = {k: v for k, v in metrics.items() if v is not None and np.isfinite(v)} # Store only valid metrics
        
        log_str_val = f"Validation Metrics (Ep {self.current_epoch+1}, GStep {self.global_step}, ActiveD: {self.active_discriminator_key}): "
        log_str_val += ", ".join([f"{k}:{v:.4f}" for k,v in self.last_val_metrics.items()])
        self.logger.info(log_str_val)
        
        return self.last_val_metrics # Return only valid metrics
    
    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        m_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        original_mode_m = m_ref.training; m_ref.eval() # Set to eval mode
        dev = self.device
        
        # Determine model's expected dtype
        # Use a more robust way to get dtype if model has no parameters yet (e.g. for an empty shell before loading ckpt)
        model_param_iter = iter(m_ref.parameters())
        first_param = next(model_param_iter, None)
        dtype_m = first_param.dtype if first_param is not None else torch.float32 # Default to float32 if no params

        if self.args.latent_dim <= 0:
            self.logger.error("Sample: Latent dim is 0 or negative, cannot generate noise for sampling.")
            m_ref.train(original_mode_m) # Restore training mode
            return None

        if noise is None:
            z = torch.randn(num_samples, self.args.latent_dim, device=dev, dtype=dtype_m)
        else:
            z = noise.to(device=dev, dtype=dtype_m)
            num_samples = z.shape[0] # Update num_samples if noise is provided

        num_predict_f_sample = self.video_config.get("num_predict_frames", 1)
        num_regions_sample = self.gaad_appearance_config.get("num_regions", 0)
        # Ensure image_h_w_tuple is correctly sourced (e.g., from args)
        img_dims_sample = self.args.image_h_w_tuple # Expected (H, W)

        sample_gaad_bboxes_batch: Optional[torch.Tensor] = None
        if num_regions_sample > 0:
            sample_bboxes_list_for_batch = []
            gaad_app_decomp_type = self.gaad_appearance_config.get("decomposition_type", "hybrid")
            gaad_app_min_px = self.gaad_appearance_config.get("min_size_px", 5)
            
            # For sampling, use static GAAD (not motion-aware as there's no input flow_magnitude map)
            for _ in range(num_samples):
                single_sample_bboxes_for_item_list = []
                current_frame_dims_for_gaad = (img_dims_sample[1], img_dims_sample[0]) # (W, H) for GAAD funcs

                if gaad_app_decomp_type == "hybrid":
                    num_sub = num_regions_sample // 2
                    num_spi = num_regions_sample - num_sub
                    if num_sub > 0:
                        single_sample_bboxes_for_item_list.append(
                            golden_subdivide_rect_fixed_n(current_frame_dims_for_gaad, num_sub, 
                                                          device=dev, dtype=dtype_m, min_size_px=gaad_app_min_px)
                        )
                    if num_spi > 0:
                        centers, scales = phi_spiral_patch_centers_fixed_n(current_frame_dims_for_gaad, num_spi, 
                                                                           device=dev, dtype=dtype_m)
                        # Convert centers/scales to bboxes [x1,y1,x2,y2]
                        patch_base_size_s = min(current_frame_dims_for_gaad)
                        sb_curr = torch.zeros(num_spi, 4, device=dev, dtype=dtype_m)
                        patch_hs_half = float(patch_base_size_s) * scales[:,0] / 2.0
                        patch_ws_half = patch_hs_half # Assuming square patches from spiral centers

                        val_x1_s = centers[:,0] - patch_ws_half
                        val_y1_s = centers[:,1] - patch_hs_half
                        val_x2_s = centers[:,0] + patch_ws_half
                        val_y2_s = centers[:,1] + patch_hs_half
                        
                        sb_curr[:,0]=torch.clamp(val_x1_s,min=0.0,max=float(current_frame_dims_for_gaad[0])-EPS)
                        sb_curr[:,1]=torch.clamp(val_y1_s,min=0.0,max=float(current_frame_dims_for_gaad[1])-EPS)
                        sb_curr[:,2]=torch.clamp(val_x2_s,max=float(current_frame_dims_for_gaad[0]))
                        sb_curr[:,2]=torch.maximum(sb_curr[:,2], sb_curr[:,0]+EPS) # Ensure x2 > x1
                        sb_curr[:,3]=torch.clamp(val_y2_s,max=float(current_frame_dims_for_gaad[1]))
                        sb_curr[:,3]=torch.maximum(sb_curr[:,3], sb_curr[:,1]+EPS) # Ensure y2 > y1
                        single_sample_bboxes_for_item_list.append(sb_curr)
                    
                    single_item_bboxes_for_frame = torch.cat(single_sample_bboxes_for_item_list, dim=0) if single_sample_bboxes_for_item_list else \
                                                   torch.tensor([[0,0,float(current_frame_dims_for_gaad[0]), float(current_frame_dims_for_gaad[1])]]*num_regions_sample, dtype=dtype_m, device=dev)

                elif gaad_app_decomp_type == "spiral":
                    centers, scales = phi_spiral_patch_centers_fixed_n(current_frame_dims_for_gaad, num_regions_sample, device=dev, dtype=dtype_m)
                    patch_base_size_s=min(current_frame_dims_for_gaad); sb_curr=torch.zeros(num_regions_sample,4,device=dev,dtype=dtype_m)
                    patch_hs_half=float(patch_base_size_s)*scales[:,0]/2.0; patch_ws_half=patch_hs_half
                    val_x1_s=centers[:,0]-patch_ws_half; val_y1_s=centers[:,1]-patch_hs_half; val_x2_s=centers[:,0]+patch_ws_half; val_y2_s=centers[:,1]+patch_hs_half
                    sb_curr[:,0]=torch.clamp(val_x1_s,min=0.0,max=float(current_frame_dims_for_gaad[0])-EPS); sb_curr[:,1]=torch.clamp(val_y1_s,min=0.0,max=float(current_frame_dims_for_gaad[1])-EPS)
                    sb_curr[:,2]=torch.clamp(val_x2_s,max=float(current_frame_dims_for_gaad[0])); sb_curr[:,2]=torch.maximum(sb_curr[:,2], sb_curr[:,0]+EPS)
                    sb_curr[:,3]=torch.clamp(val_y2_s,max=float(current_frame_dims_for_gaad[1])); sb_curr[:,3]=torch.maximum(sb_curr[:,3], sb_curr[:,1]+EPS)
                    single_item_bboxes_for_frame = sb_curr
                else: # "subdivide"
                    single_item_bboxes_for_frame = golden_subdivide_rect_fixed_n(current_frame_dims_for_gaad, num_regions_sample, 
                                                                                device=dev, dtype=dtype_m, min_size_px=gaad_app_min_px)
                
                # Ensure correct number of regions for this item after generation
                if single_item_bboxes_for_frame.shape[0] < num_regions_sample:
                    num_pad_s = num_regions_sample - single_item_bboxes_for_frame.shape[0]
                    pad_box_s = single_item_bboxes_for_frame[-1:].clone() if single_item_bboxes_for_frame.shape[0]>0 else \
                                torch.tensor([[0,0,float(current_frame_dims_for_gaad[0]),float(current_frame_dims_for_gaad[1])]], dtype=dtype_m,device=dev)
                    single_item_bboxes_for_frame = torch.cat([single_item_bboxes_for_frame, pad_box_s.repeat(num_pad_s,1)], dim=0)
                elif single_item_bboxes_for_frame.shape[0] > num_regions_sample:
                    single_item_bboxes_for_frame = single_item_bboxes_for_frame[:num_regions_sample]
                
                # Repeat these bboxes for each predicted frame
                sample_bboxes_list_for_batch.append(single_item_bboxes_for_frame.unsqueeze(0).repeat(num_predict_f_sample, 1, 1))
            
            if sample_bboxes_list_for_batch:
                sample_gaad_bboxes_batch = torch.stack(sample_bboxes_list_for_batch) # (B, N_pred_frames, N_reg, 4)
            elif num_samples > 0 : # Fallback if list is empty but samples expected
                 self.logger.warning("Sample: GAAD bbox list empty for sampling. Creating dummy full-frame bboxes.")
                 dummy_bbox_s = torch.tensor([0,0,float(img_dims_sample[1]),float(img_dims_sample[0])], dtype=dtype_m, device=dev) # W, H
                 sample_gaad_bboxes_batch = dummy_bbox_s.view(1,1,1,4).expand(num_samples, num_predict_f_sample, num_regions_sample, 4)
        
        self.logger.info(f"Sampling {num_samples} sequences (DFT:{self.args.use_dft_features_appearance}, DCT:{self.args.use_dct_features_appearance}). Z-shape: {z.shape}. Bboxes shape: {sample_gaad_bboxes_batch.shape if sample_gaad_bboxes_batch is not None else 'None'}")
        
        # The decoder expects bboxes to match the number of frames it internally generates (m_ref.generator.num_predict_frames)
        # sample_gaad_bboxes_batch should already be (B, N_gen_predict_frames, N_reg, 4) due to repeat(num_predict_f_sample,...)
        gen_pixels, gen_dft, gen_dct = m_ref.decode(z, sample_gaad_bboxes_batch) # type: ignore
        
        final_pixel_samples: Optional[torch.Tensor] = None
        if gen_pixels is not None:
            final_pixel_samples = gen_pixels
        elif self.args.use_dft_features_appearance or self.args.use_dct_features_appearance:
            if sample_gaad_bboxes_batch is None:
                self.logger.error("Sample: Cannot assemble spectral to pixels because sample_gaad_bboxes_batch is None.")
                m_ref.train(original_mode_m)
                return None

            final_pixel_samples = self._assemble_pixels_from_spectral(
                gen_dft, gen_dct, sample_gaad_bboxes_batch, # sample_gaad_bboxes_batch has N_pred_frames temporal dim
                target_image_height=self.args.image_h,
                target_image_width=self.args.image_w,
                num_image_channels_target=self.video_config['num_channels']
                # output_range is defaulted in _assemble_pixels_from_spectral
            )
        
        if final_pixel_samples is not None:
            self.logger.info(f"Sampling finished. Output pixel shape: {final_pixel_samples.shape}")
        else:
            self.logger.warning("Sampling resulted in no pixel output (final_pixel_samples is None).")
            
        m_ref.train(original_mode_m) # Restore training mode
        return final_pixel_samples


    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict[str, Any]] = None, is_best: bool = False):
        if not self.am_main_process: return
        m_s = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_primary_s = self.discriminator_primary_obj.module if self.ddp_active and hasattr(self.discriminator_primary_obj, 'module') else self.discriminator_primary_obj
        d_alt_s = self.discriminator_alternative_obj.module if self.ddp_active and hasattr(self.discriminator_alternative_obj, 'module') else self.discriminator_alternative_obj
        
        def get_q_state_from_controller_or_optimizer(obj_with_q_controller): # From audio trainer
            if obj_with_q_controller is None: return None
            q_ctrl_to_save: Optional[HAKMEMQController] = None
            if isinstance(obj_with_q_controller, HAKMEMQController): q_ctrl_to_save = obj_with_q_controller
            elif hasattr(obj_with_q_controller, 'q_controller'): q_ctrl_to_save = getattr(obj_with_q_controller, 'q_controller', None)
            if not q_ctrl_to_save or not hasattr(q_ctrl_to_save, 'q_table'): return None
            state = {'q_table': q_ctrl_to_save.q_table, 'epsilon': q_ctrl_to_save.epsilon, 'prev_lr_mom_state': q_ctrl_to_save.prev_lr_mom_state, 'prev_lr_mom_action': q_ctrl_to_save.prev_lr_mom_action, 'prev_lambda_kl_state': q_ctrl_to_save.prev_lambda_kl_state, 'prev_lambda_kl_action': q_ctrl_to_save.prev_lambda_kl_action, 'reward_hist': list(q_ctrl_to_save.reward_hist), 'q_table_access_count': dict(q_ctrl_to_save.q_table_access_count), 'q_table_creation_time': q_ctrl_to_save.q_table_creation_time, 'q_table_last_access_time': q_ctrl_to_save.q_table_last_access_time, 'on_probation': getattr(q_ctrl_to_save, 'on_probation', False), 'current_probation_step': getattr(q_ctrl_to_save, 'current_probation_step', 0), 'lkl_on_probation': getattr(q_ctrl_to_save, 'lkl_on_probation', False), 'lkl_current_probation_step': getattr(q_ctrl_to_save, 'lkl_current_probation_step', 0)}
            if hasattr(q_ctrl_to_save, 'loss_g_total_hist'): q_hist_names = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake']; state['loss_histories'] = {hname: list(getattr(q_ctrl_to_save, f"loss_{hname}_hist")) for hname in q_hist_names if hasattr(q_ctrl_to_save, f"loss_{hname}_hist")}
            if hasattr(q_ctrl_to_save, 'interval_avg_recon_hist'): q_lkl_hist_names = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']; state['interval_histories'] = {hname: list(getattr(q_ctrl_to_save, f"interval_{hname}_hist")) for hname in q_lkl_hist_names if hasattr(q_ctrl_to_save, f"interval_{hname}_hist")}
            return state

        data_to_save = {
            'global_step': self.global_step, 'epoch': self.current_epoch,
            'model_state_dict': m_s.state_dict(),
            'discriminator_primary_state_dict': d_primary_s.state_dict(),
            'discriminator_alternative_state_dict': d_alt_s.state_dict(),
            'active_discriminator_key': self.active_discriminator_key,
            'active_disc_effective_trainer_input_type': self.active_disc_effective_trainer_input_type, 
            'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict() if self.optimizer_enc_gen else None,
            'optimizer_disc_primary_state_dict': self.optimizer_disc_primary.state_dict(),
            'optimizer_disc_alternative_state_dict': self.optimizer_disc_alternative.state_dict(),
            'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict(),
            'scaler_disc_active_state_dict': self.scaler_disc_active.state_dict(), 
            'args': vars(self.args), 'metrics': metrics if metrics is not None else self.last_val_metrics.copy(),
            'best_val_metric_val': self.best_val_metric_val, 'current_lambda_kl_base': self.lambda_kl_base, # Save base KL
            'prev_interval_metrics_for_lambda_kl_reward': self.prev_interval_metrics_for_lambda_kl_reward,
            'steps_since_last_d_switch': self.steps_since_last_d_switch,
            'consecutive_trigger_primary_to_alt_count': self.consecutive_trigger_primary_to_alt_count,
            'consecutive_trigger_alt_to_primary_count': self.consecutive_trigger_alt_to_primary_count,
            'consecutive_heuristic_trigger_counts': dict(self.consecutive_heuristic_trigger_counts),
            'q_data_derived_g_recon_hist': list(self.q_data_derived_g_recon_hist),
            'avg_g_recon_hist_for_stagnation': list(self.avg_g_recon_hist_for_stagnation),
            'heuristic_vae_feature_match_active': self.heuristic_vae_feature_match_active,
            'heuristic_penalize_g_easy_win_active': self.heuristic_penalize_g_easy_win_active,
            'heuristic_boost_active_d_lr_active': self.heuristic_boost_active_d_lr_active,
            'heuristic_force_d_q_explore_active': self.heuristic_force_d_q_explore_active, 
            'heuristic_override_lambda_recon_factor': self.heuristic_override_lambda_recon_factor,
            'heuristic_override_lambda_kl_factor': self.heuristic_override_lambda_kl_factor,
            'heuristic_override_lambda_gan_factor': self.heuristic_override_lambda_gan_factor,
        }
        data_to_save['q_controller_enc_gen_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_gen)
        data_to_save['q_controller_disc_primary_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_d_primary)
        data_to_save['q_controller_disc_alternative_state'] = get_q_state_from_controller_or_optimizer(self.q_controller_d_alt)
        data_to_save['q_controller_lambda_kl_state'] = get_q_state_from_controller_or_optimizer(self.lambda_kl_q_controller)

        fprefix = "wubugaad_hybridgen_v03_dft_dct"
        if is_best: fp_str = f"{fprefix}_best_ep{self.current_epoch + 1}_step{self.global_step}.pt" 
        elif is_intermediate: fp_str = f"{fprefix}_step{self.global_step}.pt"
        else: fp_str = f"{fprefix}_ep{self.current_epoch + 1}_step{self.global_step}.pt"
        fp = Path(self.args.checkpoint_dir) / fp_str
        try: torch.save(data_to_save, fp); self.logger.info(f"Checkpoint saved: {fp.name}")
        except Exception as e: self.logger.error(f"Error saving checkpoint {fp}: {e}", exc_info=True)

    def load_checkpoint(self, checkpoint_path_str: str) -> Tuple[int, int]:
        checkpoint_path = Path(checkpoint_path_str)

        # Get references to Q-controllers for convenience
        q_ctrl_gen = getattr(self.optimizer_enc_gen, 'q_controller', None)
        q_ctrl_d_pri = self.q_controller_d_primary
        q_ctrl_d_alt = self.q_controller_d_alt
        q_ctrl_lkl = self.lambda_kl_q_controller
        all_q_controllers_list = [qc for qc in [q_ctrl_gen, q_ctrl_d_pri, q_ctrl_d_alt, q_ctrl_lkl] if qc is not None]

        global_manual_flush_requested = getattr(HAKMEMQController, 'MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD', False)
        effective_reset_request_for_q = global_manual_flush_requested or self.args.reset_q_controllers_on_load

        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint file {checkpoint_path} not found. Starting fresh.")
            if self.args.reset_lkl_q_controller_on_load:
                self.lambda_kl_base = float(self.args.lambda_kl)
                self.lambda_kl = self.lambda_kl_base # Effective lambda_kl starts as base
                self.logger.info(f"Checkpoint not found, --reset_lkl_q_controller_on_load is True. Setting self.lambda_kl_base and self.lambda_kl to args.lambda_kl: {self.lambda_kl_base:.2e}")

            for qc_obj in all_q_controllers_list:
                is_lkl_and_reset_lkl_arg = (qc_obj == q_ctrl_lkl and self.args.reset_lkl_q_controller_on_load)
                perform_reset_for_this_specific_controller = effective_reset_request_for_q or is_lkl_and_reset_lkl_arg
                self._load_q_state_helper_inner(qc_obj, None,
                                                perform_manual_flush_for_this_controller=perform_reset_for_this_specific_controller,
                                                is_associated_optimizer_state_loaded=False)
                if is_lkl_and_reset_lkl_arg and qc_obj is not None:
                    qc_obj.set_current_lambda_kl(self.lambda_kl)

            if global_manual_flush_requested and not self.args.reset_q_controllers_on_load:
                HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
            return 0, 0

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}. Starting fresh.", exc_info=True)
            if self.args.reset_lkl_q_controller_on_load:
                self.lambda_kl_base = float(self.args.lambda_kl); self.lambda_kl = self.lambda_kl_base
                self.logger.info(f"Checkpoint load failed, --reset_lkl_q_controller_on_load is True. Setting self.lambda_kl_base and self.lambda_kl to args.lambda_kl: {self.lambda_kl_base:.2e}")
            for qc_obj in all_q_controllers_list:
                is_lkl_and_reset_lkl_arg = (qc_obj == q_ctrl_lkl and self.args.reset_lkl_q_controller_on_load)
                perform_reset_for_this_specific_controller = effective_reset_request_for_q or is_lkl_and_reset_lkl_arg
                self._load_q_state_helper_inner(qc_obj, None, perform_manual_flush_for_this_specific_controller, False)
                if is_lkl_and_reset_lkl_arg and qc_obj is not None: qc_obj.set_current_lambda_kl(self.lambda_kl)
            if global_manual_flush_requested and not self.args.reset_q_controllers_on_load: HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
            return 0, 0

        # --- Load Model States ---
        m_load = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_primary_load = self.discriminator_primary_obj.module if self.ddp_active and hasattr(self.discriminator_primary_obj, 'module') else self.discriminator_primary_obj
        d_alt_load = self.discriminator_alternative_obj.module if self.ddp_active and hasattr(self.discriminator_alternative_obj, 'module') else self.discriminator_alternative_obj

        model_loaded_ok, disc_primary_model_loaded_ok, disc_alt_model_loaded_ok = False, False, False

        try:
            if 'model_state_dict' in ckpt: m_load.load_state_dict(ckpt['model_state_dict'], strict=self.args.load_strict); model_loaded_ok = True; self.logger.info("Main model state_dict loaded.")
            else: self.logger.warning("Main model_state_dict not found in checkpoint.")
        except Exception as e: self.logger.error(f"Error loading main model state_dict: {e}", exc_info=not self.args.load_strict)

        if 'discriminator_primary_state_dict' in ckpt and d_primary_load:
            try: d_primary_load.load_state_dict(ckpt['discriminator_primary_state_dict'], strict=self.args.load_strict); disc_primary_model_loaded_ok = True; self.logger.info(f"Primary D state_dict loaded.")
            except Exception as e: self.logger.error(f"Error loading D_primary state_dict: {e}", exc_info=not self.args.load_strict)
        elif not d_primary_load: self.logger.warning("discriminator_primary_obj is None, cannot load its state.")
        else: self.logger.warning("discriminator_primary_state_dict not found in checkpoint.")

        if 'discriminator_alternative_state_dict' in ckpt and d_alt_load:
            try: d_alt_load.load_state_dict(ckpt['discriminator_alternative_state_dict'], strict=self.args.load_strict); disc_alt_model_loaded_ok = True; self.logger.info(f"Alternative D state_dict loaded.")
            except Exception as e: self.logger.error(f"Error loading D_alternative state_dict: {e}", exc_info=not self.args.load_strict)
        elif not d_alt_load: self.logger.warning("discriminator_alternative_obj is None, cannot load its state.")
        else: self.logger.warning("discriminator_alternative_state_dict not found in checkpoint.")

        # --- Load Metrics and Basic Training State ---
        self.is_val_metric_higher_better = self.args.val_primary_metric in ["avg_val_psnr", "avg_val_ssim"]
        default_best_val = -float('inf') if self.is_val_metric_higher_better else float('inf')
        self.best_val_metric_val = ckpt.get('best_val_metric_val', default_best_val)
        self.last_val_metrics = ckpt.get('metrics', {}).copy() if ckpt.get('metrics') is not None else {}

        # --- Load Lambda_KL and Heuristic Factors ---
        if not self.args.reset_lkl_q_controller_on_load:
            self.lambda_kl_base = float(ckpt.get('current_lambda_kl_base', self.args.lambda_kl))
        else:
            self.lambda_kl_base = float(self.args.lambda_kl)
            self.logger.info(f"Trainer's self.lambda_kl_base set to args.lambda_kl: {self.lambda_kl_base:.2e} due to --reset_lkl_q_controller_on_load.")

        self.heuristic_override_lambda_kl_factor = ckpt.get('heuristic_override_lambda_kl_factor', 1.0)
        self.lambda_kl = self.lambda_kl_base * self.heuristic_override_lambda_kl_factor

        self.heuristic_override_lambda_recon_factor = ckpt.get('heuristic_override_lambda_recon_factor', 1.0)
        self.heuristic_override_lambda_gan_factor = ckpt.get('heuristic_override_lambda_gan_factor', 1.0)
        self.prev_interval_metrics_for_lambda_kl_reward = ckpt.get('prev_interval_metrics_for_lambda_kl_reward')

        # --- Load Global Step and Epoch ---
        loaded_gs = ckpt.get('global_step', 0)
        loaded_ep = ckpt.get('epoch', 0)
        next_ep_start = loaded_ep + 1 if model_loaded_ok and loaded_gs > 0 and loaded_ep < self.args.epochs else loaded_ep
        if getattr(self.args, 'force_start_epoch_on_load', None) is not None:
            next_ep_start = self.args.force_start_epoch_on_load
            loaded_gs = getattr(self.args, 'force_start_gstep_on_load', 0 if self.args.force_start_epoch_on_load is not None else loaded_gs)
            if self.am_main_process: self.logger.info(f"CKPT Load: Overriding start epoch to {next_ep_start} and GStep to {loaded_gs} due to force_start args.")

        # --- Load Discriminator Switching and Heuristic States ---
        saved_active_disc_key = ckpt.get('active_discriminator_key', 'primary')
        saved_active_disc_effective_type = ckpt.get('active_disc_effective_trainer_input_type', 'unknown_in_ckpt')
        target_active_key_for_this_resume = saved_active_disc_key
        forced_switch_on_resume = False

        if self.args.enable_heuristic_disc_switching and self.args.initial_disc_type:
            user_prefers_pixel_like = (self.args.initial_disc_type == 'pixel')
            user_prefers_feature_like = (self.args.initial_disc_type == 'feature')
            current_args_implied_active_key: Optional[str] = None

            d_primary_is_pixel_like = "pixels" in self.discriminator_primary_obj.effective_input_type_for_trainer # type: ignore
            d_primary_is_feature_like = "features" in self.discriminator_primary_obj.effective_input_type_for_trainer # type: ignore
            d_alt_is_pixel_like = "pixels" in self.discriminator_alternative_obj.effective_input_type_for_trainer # type: ignore
            d_alt_is_feature_like = "features" in self.discriminator_alternative_obj.effective_input_type_for_trainer # type: ignore

            if user_prefers_pixel_like:
                if d_primary_is_pixel_like: current_args_implied_active_key = 'primary'
                elif d_alt_is_pixel_like: current_args_implied_active_key = 'alternative'
            elif user_prefers_feature_like:
                if d_primary_is_feature_like: current_args_implied_active_key = 'primary'
                elif d_alt_is_feature_like: current_args_implied_active_key = 'alternative'

            if current_args_implied_active_key is not None and current_args_implied_active_key != saved_active_disc_key:
                if self.am_main_process: self.logger.warning(f"LOAD_CKPT_OVERRIDE: Ckpt active D was '{saved_active_disc_key}' (Type: '{saved_active_disc_effective_type}'). Current initial_disc_type ('{self.args.initial_disc_type}') implies '{current_args_implied_active_key}'. FORCING active D to '{current_args_implied_active_key}'.")
                target_active_key_for_this_resume = current_args_implied_active_key; forced_switch_on_resume = True
            elif current_args_implied_active_key is None and self.am_main_process:
                self.logger.warning(f"LOAD_CKPT_WARNING: initial_disc_type ('{self.args.initial_disc_type}') did not match D types (Pri: {self.discriminator_primary_obj.effective_input_type_for_trainer}, Alt: {self.discriminator_alternative_obj.effective_input_type_for_trainer}). Using active D from ckpt: '{saved_active_disc_key}'.")

        self.active_discriminator_key = target_active_key_for_this_resume
        self._update_active_discriminator_pointers()

        # --- Load Optimizer States ---
        opt_g_loaded_ok, opt_d_primary_loaded_ok, opt_d_alt_loaded_ok = False, False, False
        if self.optimizer_enc_gen and 'optimizer_enc_gen_state_dict' in ckpt and ckpt['optimizer_enc_gen_state_dict'] is not None:
            if model_loaded_ok:
                try: self.optimizer_enc_gen.load_state_dict(ckpt['optimizer_enc_gen_state_dict']); opt_g_loaded_ok = True
                except Exception as e: self.logger.warning(f"Could not load Opt_Gen state: {e}. It will start fresh.")
            else: self.logger.warning("Main model failed to load, Opt_Gen will start fresh.")
        if self.optimizer_enc_gen:
            for group in self.optimizer_enc_gen.param_groups: group['initial_lr'] = self.args.learning_rate_gen; group['initial_momentum'] = self.optimizer_enc_gen.defaults.get('momentum', 0.9)

        if self.optimizer_disc_primary and 'optimizer_disc_primary_state_dict' in ckpt and ckpt['optimizer_disc_primary_state_dict'] is not None:
            if disc_primary_model_loaded_ok:
                try: self.optimizer_disc_primary.load_state_dict(ckpt['optimizer_disc_primary_state_dict']); opt_d_primary_loaded_ok = True
                except Exception as e: self.logger.warning(f"Could not load Opt_D_Primary state: {e}. It will start fresh.")
            else: self.logger.warning("D_Primary model failed to load, Opt_D_Primary will start fresh.")
        if self.optimizer_disc_primary:
            for group in self.optimizer_disc_primary.param_groups: group['initial_lr'] = self.args.learning_rate_disc; group['initial_momentum'] = self.optimizer_disc_primary.defaults.get('momentum', 0.9)

        lr_disc_alt_load = getattr(self.args, 'learning_rate_disc_alt', self.args.learning_rate_disc)
        if self.optimizer_disc_alternative and 'optimizer_disc_alternative_state_dict' in ckpt and ckpt['optimizer_disc_alternative_state_dict'] is not None:
            if disc_alt_model_loaded_ok:
                try: self.optimizer_disc_alternative.load_state_dict(ckpt['optimizer_disc_alternative_state_dict']); opt_d_alt_loaded_ok = True
                except Exception as e: self.logger.warning(f"Could not load Opt_D_Alt state: {e}. It will start fresh.")
            else: self.logger.warning("D_Alternative model failed to load, Opt_D_Alt will start fresh.")
        if self.optimizer_disc_alternative:
            for group in self.optimizer_disc_alternative.param_groups: group['initial_lr'] = lr_disc_alt_load; group['initial_momentum'] = self.optimizer_disc_alternative.defaults.get('momentum', 0.9)

        # --- Load Q-Controller States ---
        self._load_q_state_helper_inner(q_ctrl_gen, ckpt.get('q_controller_enc_gen_state'), effective_reset_request_for_q, opt_g_loaded_ok)
        self._load_q_state_helper_inner(q_ctrl_d_pri, ckpt.get('q_controller_disc_primary_state'), effective_reset_request_for_q, opt_d_primary_loaded_ok)
        self._load_q_state_helper_inner(q_ctrl_d_alt, ckpt.get('q_controller_disc_alternative_state'), effective_reset_request_for_q, opt_d_alt_loaded_ok)

        if self.args.reset_lkl_q_controller_on_load and q_ctrl_lkl is not None:
            self.logger.info(f"FORCE RESETTING Lambda_KL Q-Controller due to --reset_lkl_q_controller_on_load.")
            self.lambda_kl_base = float(self.args.lambda_kl); self.lambda_kl = self.lambda_kl_base
            q_ctrl_lkl.reset_q_learning_state(reset_q_table=True, reset_epsilon=True, context_msg="LKL Q-Ctrl Force Reset on Load by Arg", start_probation=True)
            self.logger.info(f"Trainer's self.lambda_kl_base (set to args value): {self.lambda_kl_base:.2e} after LKL Q-Ctrl reset."); q_ctrl_lkl.set_current_lambda_kl(self.lambda_kl); self.prev_interval_metrics_for_lambda_kl_reward = None
        else: self._load_q_state_helper_inner(q_ctrl_lkl, ckpt.get('q_controller_lambda_kl_state'), effective_reset_request_for_q, True)

        # --- Reset Active Discriminator's Q-Controller if Forced Switch ---
        if forced_switch_on_resume:
            active_d_q_to_reset = self.q_controller_d_active
            if active_d_q_to_reset:
                if self.am_main_process: self.logger.warning(f"Due to resume override, resetting Q-controller for newly FORCED active D: '{self.active_discriminator_key}' (Effective Input Type: {self.active_disc_effective_trainer_input_type}).")
                active_d_q_to_reset.reset_q_learning_state(True, True, f"Forced D switch to {self.active_discriminator_key} on Resume Override", True)

            self.steps_since_last_d_switch = 0
            self.consecutive_trigger_primary_to_alt_count = 0; self.consecutive_trigger_alt_to_primary_count = 0
            self.consecutive_heuristic_trigger_counts = defaultdict(int)
            self.q_data_derived_g_recon_hist.clear(); self.rec_features_stagnant = False
            self.avg_g_recon_hist_for_stagnation.clear()
            if self.am_main_process: self.logger.info("Heuristic switching counters and short-term recon history reset due to forced D switch on resume.")
        else:
            self.steps_since_last_d_switch = ckpt.get('steps_since_last_d_switch', 0)
            self.consecutive_trigger_primary_to_alt_count = ckpt.get('consecutive_trigger_primary_to_alt_count', 0)
            self.consecutive_trigger_alt_to_primary_count = ckpt.get('consecutive_trigger_alt_to_primary_count', 0)
            self.consecutive_heuristic_trigger_counts = defaultdict(int, ckpt.get('consecutive_heuristic_trigger_counts', {}))
            if 'q_data_derived_g_recon_hist' in ckpt and ckpt['q_data_derived_g_recon_hist'] is not None:
                try: self.q_data_derived_g_recon_hist.clear(); self.q_data_derived_g_recon_hist.extend(list(ckpt['q_data_derived_g_recon_hist']))
                except TypeError: self.logger.warning(f"Could not extend deque q_data_derived_g_recon_hist from checkpoint.")
            if 'avg_g_recon_hist_for_stagnation' in ckpt and ckpt['avg_g_recon_hist_for_stagnation'] is not None:
                try: self.avg_g_recon_hist_for_stagnation.clear(); self.avg_g_recon_hist_for_stagnation.extend(list(ckpt['avg_g_recon_hist_for_stagnation']))
                except TypeError: self.logger.warning(f"Could not extend deque avg_g_recon_hist_for_stagnation from checkpoint.")

        self.heuristic_vae_feature_match_active = ckpt.get('heuristic_vae_feature_match_active', False)
        self.heuristic_penalize_g_easy_win_active = ckpt.get('heuristic_penalize_g_easy_win_active', False)
        self.heuristic_boost_active_d_lr_active = ckpt.get('heuristic_boost_active_d_lr_active', False)
        # self.heuristic_force_d_q_explore_active is handled by Q-controller's probation/boost state loading

        if global_manual_flush_requested and not self.args.reset_q_controllers_on_load:
            HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
            if self.am_main_process: self.logger.info("Global MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD applied and reset.")
        elif self.args.reset_q_controllers_on_load and self.am_main_process:
             self.logger.info("Global Q-controller reset triggered by --reset_q_controllers_on_load argument for all applicable Q-controllers.")

        if self.args.use_amp and self.device.type == 'cuda':
            if 'scaler_enc_gen_state_dict' in ckpt and self.scaler_enc_gen and ckpt['scaler_enc_gen_state_dict'] is not None:
                try: self.scaler_enc_gen.load_state_dict(ckpt['scaler_enc_gen_state_dict'])
                except Exception as e_sc_g: self.logger.warning(f"Could not load scaler_enc_gen state: {e_sc_g}")
            if 'scaler_disc_active_state_dict' in ckpt and self.scaler_disc_active and ckpt['scaler_disc_active_state_dict'] is not None:
                try: self.scaler_disc_active.load_state_dict(ckpt['scaler_disc_active_state_dict'])
                except Exception as e_sc_d: self.logger.warning(f"Could not load scaler_disc_active state: {e_sc_d}")

        for q_ctrl_sync in all_q_controllers_list:
            if q_ctrl_sync and hasattr(q_ctrl_sync, 'set_current_lambda_kl'):
                q_ctrl_sync.set_current_lambda_kl(self.lambda_kl)

        d_ref_resume_log = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        active_d_arch_variant_resume_log = getattr(d_ref_resume_log, 'architecture_variant', 'unknown_variant')
        self.logger.info(
            f"Resuming training. GlobalStep: {loaded_gs}, NextEpochStart: {next_ep_start}. "
            f"ActiveD upon resume: '{self.active_discriminator_key}' (Arch:{active_d_arch_variant_resume_log}, EffIn:{self.active_disc_effective_trainer_input_type}). "
            f"Effective Lambda_KL (base*factor): {self.lambda_kl:.4e}"
        )
        return loaded_gs, next_ep_start

    def _load_q_state_helper_inner(self, q_controller_instance: Optional["HAKMEMQController"],
                                   q_state_from_ckpt: Optional[Dict],
                                   perform_manual_flush_for_this_controller: bool,
                                   is_associated_optimizer_state_loaded: bool):
        """
        Helper to load or reset state for a single Q-controller.
        Args:
            q_controller_instance: The HAKMEMQController instance.
            q_state_from_ckpt: The Q-controller state dict from the checkpoint, if available.
            perform_manual_flush_for_this_controller: If True, forces a full reset of this Q-controller.
            is_associated_optimizer_state_loaded: If False, Q-controller should start on probation
                                                  even if loading its state, as the optimizer is fresh.
        """
        if q_controller_instance is None:
            return

        q_ctrl_name = q_controller_instance.logger.name.split('.')[-1] if q_controller_instance.logger else "UnknownQCtrl"

        if perform_manual_flush_for_this_controller:
            self.logger.info(f"Q-Ctrl Load Helper: Manual flush requested for '{q_ctrl_name}'. Resetting and starting probation.")
            q_controller_instance.reset_q_learning_state(True, True, f"Manual Flush for {q_ctrl_name} on Load", True)
            return

        if q_state_from_ckpt is not None:
            try:
                q_controller_instance.q_table = q_state_from_ckpt.get('q_table', {})
                q_controller_instance.epsilon = float(q_state_from_ckpt.get('epsilon', q_controller_instance.epsilon_start))
                q_controller_instance.prev_lr_mom_state = q_state_from_ckpt.get('prev_lr_mom_state')
                q_controller_instance.prev_lr_mom_action = q_state_from_ckpt.get('prev_lr_mom_action')
                q_controller_instance.prev_lambda_kl_state = q_state_from_ckpt.get('prev_lambda_kl_state')
                q_controller_instance.prev_lambda_kl_action = q_state_from_ckpt.get('prev_lambda_kl_action')

                # Load histories
                if 'reward_hist' in q_state_from_ckpt and q_state_from_ckpt['reward_hist'] is not None:
                    q_controller_instance.reward_hist = deque(q_state_from_ckpt['reward_hist'], maxlen=q_controller_instance.reward_hist.maxlen)
                if 'loss_histories' in q_state_from_ckpt:
                    for hname, hlist in q_state_from_ckpt['loss_histories'].items():
                        hist_attr = getattr(q_controller_instance, f"loss_{hname}_hist", None)
                        if hist_attr is not None: hist_attr.clear(); hist_attr.extend(hlist)
                if 'interval_histories' in q_state_from_ckpt:
                    for hname, hlist in q_state_from_ckpt['interval_histories'].items():
                        hist_attr = getattr(q_controller_instance, f"interval_{hname}_hist", None)
                        if hist_attr is not None: hist_attr.clear(); hist_attr.extend(hlist)

                # Load Q-table stats
                q_controller_instance.q_table_access_count = defaultdict(int, q_state_from_ckpt.get('q_table_access_count', {}))
                q_controller_instance.q_table_creation_time = q_state_from_ckpt.get('q_table_creation_time', {})
                q_controller_instance.q_table_last_access_time = q_state_from_ckpt.get('q_table_last_access_time', {})

                # Load probation state
                q_controller_instance.on_probation = q_state_from_ckpt.get('on_probation', False)
                q_controller_instance.current_probation_step = q_state_from_ckpt.get('current_probation_step', 0)
                q_controller_instance.lkl_on_probation = q_state_from_ckpt.get('lkl_on_probation', False)
                q_controller_instance.lkl_current_probation_step = q_state_from_ckpt.get('lkl_current_probation_step', 0)

                if not is_associated_optimizer_state_loaded:
                    self.logger.info(f"Q-Ctrl Load Helper ('{q_ctrl_name}'): Optimizer state not loaded. Forcing Q-controller into probation.")
                    q_controller_instance.start_probation() # Start probation if optimizer is fresh

                self.logger.info(f"Q-Ctrl Load Helper: State loaded for '{q_ctrl_name}'. Epsilon: {q_controller_instance.epsilon:.3f}. Probation LR/Mom: {q_controller_instance.on_probation} ({q_controller_instance.current_probation_step}), LKL: {q_controller_instance.lkl_on_probation} ({q_controller_instance.lkl_current_probation_step}).")

            except Exception as e_q_load:
                self.logger.error(f"Q-Ctrl Load Helper: Error loading state for '{q_ctrl_name}': {e_q_load}. Resetting and starting probation.", exc_info=True)
                q_controller_instance.reset_q_learning_state(True, True, f"Load Fail for {q_ctrl_name}", True)
        else:
            self.logger.info(f"Q-Ctrl Load Helper: No Q-state in checkpoint for '{q_ctrl_name}'. Resetting and starting probation.")
            q_controller_instance.reset_q_learning_state(True, True, f"No Q-State in Ckpt for {q_ctrl_name}", True)

    @staticmethod
    def get_scale_from_action_value(action_dict: Optional[Dict[str, float]], key: str, default_value: float = 1.0) -> float:
        if action_dict is None: return default_value
        val = action_dict.get(key)
        return float(val) if val is not None and np.isfinite(val) else default_value

# =====================================================================
# Utility functions for DDP & main execution
# =====================================================================
def seed_everything(seed: int, rank: int = 0, world_size: int = 1):
    effective_seed = seed + rank # Offset seed by rank for DDP
    random.seed(effective_seed)
    os.environ['PYTHONHASHSEED'] = str(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed) # For multi-GPU
        # The following two lines are often debated; enable if strict determinism is paramount
        # torch.backends.cudnn.deterministic = True 
        # torch.backends.cudnn.benchmark = False # Can hinder performance if set to False
    # if rank == 0:
    #     logging.getLogger("WuBuGAADHybridGenV03.Seed").info(f"Global seed set to {seed}, effective seed for rank {rank} is {effective_seed}")

def seed_worker_init_fn(worker_id: int, base_seed: int, rank: int, world_size: int):
    worker_seed = base_seed + rank * world_size + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # if worker_id == 0 and rank == 0: # Log only for one worker on main process
    #    logging.getLogger("WuBuGAADHybridGenV03.Seed").debug(f"Worker {worker_id} on rank {rank} initialized with seed {worker_seed}")


# =====================================================================
# Arg Parsing and Main Execution Logic (Updated for v0.3)
# =====================================================================
def _configure_wubu_stack(args: argparse.Namespace, prefix: str) -> Dict: # From audio script
    config = DEFAULT_CONFIG_WUBU.copy()
    num_levels_val = getattr(args, f"{prefix}_num_levels", 0)
    config["num_levels"] = num_levels_val
    if num_levels_val == 0:
        for key in ["hyperbolic_dims", "initial_curvatures", "initial_scales", "initial_spread_values", "boundary_points_per_level", "transform_types", "transform_hidden_dims"]: config[key] = []
        config["tangent_input_combination_dims"] = [DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0]]; return config
    config["hyperbolic_dims"] = getattr(args, f"{prefix}_hyperbolic_dims", DEFAULT_CONFIG_WUBU["hyperbolic_dims"]); config["initial_curvatures"] = getattr(args, f"{prefix}_initial_curvatures", DEFAULT_CONFIG_WUBU["initial_curvatures"])
    config["use_rotation_in_transform"] = getattr(args, f"{prefix}_use_rotation", DEFAULT_CONFIG_WUBU["use_rotation_in_transform"]); config["phi_influence_curvature"] = getattr(args, f"{prefix}_phi_influence_curvature", DEFAULT_CONFIG_WUBU["phi_influence_curvature"]); config["phi_influence_rotation_init"] = getattr(args, f"{prefix}_phi_influence_rotation_init", DEFAULT_CONFIG_WUBU["phi_influence_rotation_init"]); config["dropout"] = args.wubu_dropout
    def _ensure_list_len(cfg_dict, key, target_len, default_config_value_for_key_ref):
        arg_name_for_key = f"{prefix}_{key}"; current_val_from_args = getattr(args, arg_name_for_key, None)
        current_val = default_config_value_for_key_ref if current_val_from_args is None else current_val_from_args
        is_list_orig = isinstance(current_val, list); current_list_val = current_val if is_list_orig else [current_val]
        base_default_for_fill = default_config_value_for_key_ref[0] if isinstance(default_config_value_for_key_ref, list) and default_config_value_for_key_ref else (default_config_value_for_key_ref if not isinstance(default_config_value_for_key_ref, list) else (1.0 if "scales" in key or "curvatures" in key else (0.1 if "spread" in key else ("linear" if "types" in key else 32))))
        fill_val = current_list_val[-1] if current_list_val else base_default_for_fill
        if len(current_list_val) < target_len: cfg_dict[key] = (current_list_val + [fill_val]*(target_len-len(current_list_val)))[:target_len]
        elif len(current_list_val) > target_len: cfg_dict[key] = current_list_val[:target_len]
        else: cfg_dict[key] = current_list_val
    for key_chk in ["hyperbolic_dims", "initial_curvatures", "initial_scales", "initial_spread_values", "boundary_points_per_level"]: _ensure_list_len(config, key_chk, num_levels_val, DEFAULT_CONFIG_WUBU.get(key_chk, []))
    if "boundary_points_per_level" in config and num_levels_val > 0 and not hasattr(args, f"{prefix}_boundary_points_per_level"): config["boundary_points_per_level"] = [DEFAULT_CONFIG_WUBU["boundary_points_per_level"][0]] * num_levels_val # Default for video is [4]
    if not isinstance(config.get("tangent_input_combination_dims"), list): config["tangent_input_combination_dims"] = [config.get("tangent_input_combination_dims", DEFAULT_CONFIG_WUBU["tangent_input_combination_dims"][0])]
    num_transitions = max(0, num_levels_val-1)
    if num_transitions > 0: _ensure_list_len(config,"transform_types",num_transitions,DEFAULT_CONFIG_WUBU["transform_types"]); _ensure_list_len(config,"transform_hidden_dims",num_transitions,DEFAULT_CONFIG_WUBU["transform_hidden_dims"])
    else: config["transform_types"]=[]; config["transform_hidden_dims"]=[]
    return config

def validate_wubu_config_for_argparse(args_obj, prefix_str, parser_ref): # From audio script
    num_levels = getattr(args_obj, f"{prefix_str}_num_levels", 0)
    if num_levels > 0:
        for suffix, attr_name_fmt in [("hyperbolic_dims", "{prefix}_hyperbolic_dims"), ("initial_curvatures", "{prefix}_initial_curvatures")]:
            attr_name = attr_name_fmt.format(prefix=prefix_str)
            val_list = getattr(args_obj, attr_name)
            if not isinstance(val_list, list):
                val_list = [val_list]
            setattr(args_obj, attr_name, val_list)
            if len(val_list) != num_levels:
                if len(val_list) == 1 and num_levels > 1:
                    setattr(args_obj, attr_name, [val_list[0]] * num_levels)
                elif not val_list:
                    default_val = 1.0 if "curvatures" in suffix else \
                                  (max(1, getattr(args_obj, 'latent_dim', 32*num_levels)//num_levels if num_levels > 0 else 32) \
                                   if "dims" in suffix else None)
                    if default_val is not None:
                        setattr(args_obj, attr_name, [default_val] * num_levels)
                    else:
                        parser_ref.error(f"{attr_name} empty and no clear default for {num_levels} levels.")
                else:
                    parser_ref.error(f"{attr_name} length {len(val_list)} != num_levels {num_levels}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBu-GAAD Regional VAE-GAN w/ OptFlow & DFT+DCT (v0.3)")
    
    # --- Group: Core Paths and DDP/General Setup ---
    core_group = parser.add_argument_group('Core Paths and DDP/General Setup')
    core_group.add_argument('--video_data_path', type=str, default="demo_video_data_dir", help="Path to video data directory or a single video file.")
    core_group.add_argument('--checkpoint_dir',type=str, default='wubugaad_hybridgen_v03_checkpoints', help="Directory to save checkpoints.")
    core_group.add_argument('--load_checkpoint', type=str, default=None, help="Path to a checkpoint to load and resume training.")
    core_group.add_argument('--load_strict', action='store_true', default=False, help="Whether to strictly enforce that the keys in checkpoint match the keys returned by model.state_dict().") # Changed default
    core_group.add_argument('--local_rank', type=int, default=-1, help="Local rank for DDP. Default -1 means not using DDP explicitly through this arg.")
    core_group.add_argument('--seed',type=int, default=1337, help="Random seed for reproducibility.")
    core_group.add_argument('--num_workers',type=int, default=4, help="Number of worker processes for DataLoader.")
    core_group.add_argument('--use_amp', action='store_true', default=True, help="Enable Automatic Mixed Precision (AMP) training.")
    core_group.add_argument('--detect_anomaly',action='store_true', default=False, help="Enable PyTorch's autograd anomaly detection.")
    core_group.add_argument('--ddp_find_unused_params_d', action='store_true', default=True, help="Set find_unused_parameters=True for Discriminator DDP.")
    core_group.add_argument('--ddp_find_unused_params_g', action='store_true', default=True, help="Set find_unused_parameters=True for Generator/Encoder DDP.")

    # --- Group: Training Hyperparameters ---
    train_hp_group = parser.add_argument_group('Training Hyperparameters')
    train_hp_group.add_argument('--epochs', type=int, default=1000, help="Total number of training epochs.")
    train_hp_group.add_argument('--batch_size', type=int, default=8, help="Batch size per GPU.")
    train_hp_group.add_argument('--grad_accum_steps',type=int, default=1, help="Number of steps to accumulate gradients before an optimizer step.")
    train_hp_group.add_argument('--learning_rate_gen',type=float,default=1e-4, help="Learning rate for the generator/encoder optimizer.")
    train_hp_group.add_argument('--learning_rate_disc',type=float,default=2e-4, help="Learning rate for the primary discriminator optimizer.")
    train_hp_group.add_argument('--learning_rate_disc_alt',type=float,default=1.5e-4, help="Learning rate for the alternative discriminator optimizer.")
    train_hp_group.add_argument('--risgd_max_grad_norm',type=float,default=2.0, help="Max grad norm for RiemannianSGD internal clipping (per parameter).")
    train_hp_group.add_argument('--global_max_grad_norm',type=float,default=5.0, help="Global max grad norm for clipping all model parameters before optimizer step.")

    # --- Group: Loss Weights & VAE Structure ---
    loss_group = parser.add_argument_group('Loss Weights & VAE Structure')
    loss_group.add_argument('--lambda_recon', type=float, default=10.0, help="Global reconstruction lambda (used if only pixels or only one spectral type is fallback).")
    loss_group.add_argument('--lambda_recon_dft', type=float, default=7.0, help="Weight for DFT reconstruction loss.")
    loss_group.add_argument('--lambda_recon_dct', type=float, default=7.0, help="Weight for DCT reconstruction loss.")
    loss_group.add_argument('--lambda_kl', type=float, default=0.001, help="Base weight for KL divergence loss (can be adapted by Q-controller).")
    loss_group.add_argument('--lambda_gan', type=float, default=1.0, help="Base weight for GAN adversarial loss for the generator (can be adapted by heuristics).")
    loss_group.add_argument('--latent_dim', type=int, default=512, help="Dimensionality of the VAE latent space.")

    # --- Group: Video & GAAD Config ---
    video_gaad_group = parser.add_argument_group('Video & GAAD Config')
    video_gaad_group.add_argument('--image_h', type=int, default=64, help="Target height for video frames.")
    video_gaad_group.add_argument('--image_w', type=int, default=64, help="Target width for video frames.")
    video_gaad_group.add_argument('--num_channels', type=int, default=3, help="Number of channels in input/output video frames (e.g., 3 for RGB).")
    video_gaad_group.add_argument('--num_input_frames', type=int, default=4, help="Number of input frames for conditioning/context.")
    video_gaad_group.add_argument('--num_predict_frames', type=int, default=1, help="Number of future frames to predict.")
    video_gaad_group.add_argument('--frame_skip', type=int, default=1, help="Number of frames to skip between selected frames in a sequence.")
    video_gaad_group.add_argument('--gaad_num_regions', type=int, default=16, help="Number of GAAD regions for appearance features.")
    video_gaad_group.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="GAAD decomposition method.")
    video_gaad_group.add_argument('--gaad_min_size_px', type=int, default=4, help="Minimum size (pixels) for a GAAD region.")

    # --- Group: Spectral Transforms (DFT & DCT) ---
    spectral_group = parser.add_argument_group('Spectral Transforms (DFT & DCT)')
    spectral_group.add_argument('--use_dft_features_appearance', action='store_true', default=True, help="Enable DFT features for appearance.")
    spectral_group.add_argument('--use_dct_features_appearance', action='store_true', default=True, help="Enable DCT features for appearance.")
    spectral_group.add_argument('--spectral_patch_size_h', type=int, default=16, help="Target patch height for DFT/DCT.")
    spectral_group.add_argument('--spectral_patch_size_w', type=int, default=16, help="Target patch width for DFT/DCT.")
    spectral_group.add_argument('--dft_norm_scale_video', type=float, default=50.0, help="Tanh normalization scale for DFT components (video).")
    spectral_group.add_argument('--dft_fft_norm', type=str, default="ortho", choices=["backward", "ortho", "forward"], help="Normalization mode for torch.fft.")
    spectral_group.add_argument('--dct_norm_type', type=str, default="tanh", choices=["none", "global_scale", "tanh"], help="Normalization type for DCT components.")
    spectral_group.add_argument('--dct_norm_global_scale', type=float, default=150.0, help="Global scale for DCT if 'global_scale' norm_type.")
    spectral_group.add_argument('--dct_norm_tanh_scale', type=float, default=40.0, help="Tanh scale for DCT if 'tanh' norm_type.")

    # --- Group: Motion Branch ---
    motion_group = parser.add_argument_group('Motion Branch')
    motion_group.add_argument('--use_wubu_motion_branch', action='store_true', default=True, help="Enable WuBu motion encoding branch.")
    motion_group.add_argument('--gaad_motion_num_regions', type=int, default=8, help="Number of GAAD regions for motion features.")
    motion_group.add_argument('--gaad_motion_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="GAAD method for motion.")
    motion_group.add_argument('--optical_flow_net_type', type=str, default='raft_small', choices=list(FLOW_MODELS.keys()) if OPTICAL_FLOW_AVAILABLE else [], help="Type of optical flow network.")
    motion_group.add_argument('--freeze_flow_net', action='store_true', default=True, help="Freeze weights of the optical flow network.")
    motion_group.add_argument('--flow_stats_components', nargs='+', type=str, default=['mag_mean', 'angle_mean', 'mag_std'], help="Which optical flow statistics to compute per region.")
    motion_group.add_argument('--motion_gaad_gas_prioritize_high_energy', action='store_true', default=True,
                              help="For GAS motion regions, prioritize subdividing areas with high motion energy.")
    motion_group.add_argument('--motion_gaad_gas_energy_thresh_factor', type=float, default=0.05,
                              help="Factor of max motion energy; regions below this (and deep enough) won't be subdivided by GAS.")
    motion_group.add_argument('--motion_gaad_psp_arms_per_hotspot', type=int, default=3,
                              help="Number of spiral arms to generate from each motion hotspot for PSP.")
    motion_group.add_argument('--motion_gaad_psp_pts_per_arm', type=int, default=4,
                              help="Number of points (patch centers) per spiral arm for PSP.")
    motion_group.add_argument('--motion_gaad_psp_motion_scale_influence', type=float, default=0.6,
                              help="Influence of local motion energy on PSP patch size (0=none, 1=fully inverse).")
    motion_group.add_argument('--motion_gaad_psp_hotspot_blur_sigma', type=float, default=2.5,
                              help="Gaussian blur sigma for `analysis_map` before finding PSP hotspots.")
    motion_group.add_argument('--motion_gaad_psp_min_scale', type=float, default=0.02)
    motion_group.add_argument('--motion_gaad_psp_max_scale', type=float, default=0.35)
    # --- Group: Encoder Architecture ---
    enc_arch_group = parser.add_argument_group('Encoder Architecture')
    enc_arch_group.add_argument('--encoder_use_roi_align', action='store_true', default=False, help="Use RoIAlign for patch extraction in encoder instead of direct pixel cropping.")
    enc_arch_group.add_argument('--encoder_shallow_cnn_channels', type=int, default=32, help="Number of channels for shallow CNN if RoIAlign is used.")
    enc_arch_group.add_argument('--encoder_initial_tangent_dim', type=int, default=256, help="Initial tangent space dimension for WuBu-S input from PatchEmbed.")

    # --- Group: Generator Architecture ---
    gen_arch_group = parser.add_argument_group('Generator Architecture')
    gen_arch_group.add_argument('--gen_temporal_kernel_size', type=int, default=3, help="Temporal kernel size in generator's 3D conv layers.")
    gen_arch_group.add_argument('--gen_final_conv_kernel_spatial', type=int, default=3, help="Spatial kernel size of the final pixel conv layer (if pixel fallback is used).")
    gen_arch_group.add_argument('--gen_use_gaad_film_condition', action='store_true', default=True, help="Enable GAAD-based FiLM conditioning in the generator.")

    # --- Group: General Discriminator Settings (used by specific D variants if not overridden) ---
    disc_general_group = parser.add_argument_group('General Discriminator Settings')
    disc_general_group.add_argument('--disc_apply_spectral_norm', action='store_true', default=True)
    # For default_pixel_cnn (RegionalDiscriminator)
    disc_general_group.add_argument('--disc_base_disc_channels', type=int, default=64) 
    disc_general_group.add_argument('--disc_max_disc_channels', type=int, default=512)  
    disc_general_group.add_argument('--disc_temporal_kernel_size', type=int, default=3) 
    disc_general_group.add_argument('--disc_target_final_feature_dim', nargs='+', type=int, default=[4,4]) 
    disc_general_group.add_argument('--max_video_disc_downsample_layers', type=int, default=5) 
    disc_general_group.add_argument('--disc_use_gaad_film_condition', action='store_true', default=True) 
    disc_general_group.add_argument('--disc_gaad_condition_dim_disc', type=int, default=64) 
    
    # --- Group: Discriminator Architecture Variants & Switching ---
    disc_variant_group = parser.add_argument_group('Discriminator Architecture Variants & Switching')
    disc_variant_group.add_argument('--primary_disc_architecture_variant', type=str, default="default_pixel_cnn", choices=["default_pixel_cnn", "global_wubu_video_feature"]) # Add more as implemented
    disc_variant_group.add_argument('--alt_disc_architecture_variant', type=str, default="global_wubu_video_feature", choices=["default_pixel_cnn", "global_wubu_video_feature"])
    disc_variant_group.add_argument('--enable_heuristic_disc_switching', action='store_true', default=True)
    disc_variant_group.add_argument('--initial_disc_type', type=str, default='pixel', choices=['pixel', 'feature'], help="Hint for which D type to prefer initially if switching is on ('pixel' for pixel-based D, 'feature' for feature-based D).")
    disc_variant_group.add_argument('--disc_switch_check_interval', type=int, default=25)
    disc_variant_group.add_argument('--disc_switch_min_steps_between', type=int, default=50)
    disc_variant_group.add_argument('--disc_switch_problem_state_count_thresh', type=int, default=2)

    # --- Group: GlobalWuBuVideoFeatureDiscriminator Specifics ---
    gwvf_d_group = parser.add_argument_group('Global WuBu Video Feature Discriminator Specifics')
    gwvf_d_group.add_argument('--video_global_wubu_d_input_tangent_dim', type=int, default=512)
    gwvf_d_group.add_argument('--video_global_wubu_d_output_feature_dim', type=int, default=256)
    gwvf_d_group.add_argument('--disc_use_global_stats_aux_video_global_wubu', action='store_true', default=True)
    gwvf_d_group.add_argument('--disc_global_stats_mlp_hidden_dim_video_global_wubu', type=int, default=64)

    # --- WuBu Stack Configs ---
    parser.add_argument('--wubu_dropout', type=float, default=0.1)
    # WuBu-S (Encoder Spatial/Appearance)
    wubu_s_group = parser.add_argument_group('WuBu-S (Encoder Appearance)')
    wubu_s_group.add_argument('--wubu_s_num_levels', type=int, default=3); wubu_s_group.add_argument('--wubu_s_hyperbolic_dims', nargs='+', type=int, default=[128, 96, 64]); wubu_s_group.add_argument('--wubu_s_initial_curvatures', nargs='+', type=float, default=[1.0, 0.8, 0.6]); wubu_s_group.add_argument('--wubu_s_output_dim', type=int, default=64) 
    wubu_s_group.add_argument('--wubu_s_use_rotation', action='store_true', default=False); wubu_s_group.add_argument('--wubu_s_phi_influence_curvature', action='store_true', default=False); wubu_s_group.add_argument('--wubu_s_phi_influence_rotation_init', action='store_true', default=False)
    # WuBu-M (Encoder Motion)
    wubu_m_group = parser.add_argument_group('WuBu-M (Encoder Motion)')
    wubu_m_group.add_argument('--wubu_m_num_levels', type=int, default=2); wubu_m_group.add_argument('--wubu_m_hyperbolic_dims', nargs='+', type=int, default=[64, 48]); wubu_m_group.add_argument('--wubu_m_initial_curvatures', nargs='+', type=float, default=[0.7, 0.5]); wubu_m_group.add_argument('--wubu_m_output_dim', type=int, default=48)
    wubu_m_group.add_argument('--wubu_m_use_rotation', action='store_true', default=False); wubu_m_group.add_argument('--wubu_m_phi_influence_curvature', action='store_true', default=False); wubu_m_group.add_argument('--wubu_m_phi_influence_rotation_init', action='store_true', default=False)
    # WuBu-T (Encoder Temporal)
    wubu_t_group = parser.add_argument_group('WuBu-T (Encoder Temporal)')
    wubu_t_group.add_argument('--wubu_t_num_levels', type=int, default=2); wubu_t_group.add_argument('--wubu_t_hyperbolic_dims', nargs='+', type=int, default=[256, 128]); wubu_t_group.add_argument('--wubu_t_initial_curvatures', nargs='+', type=float, default=[0.5, 0.3])
    wubu_t_group.add_argument('--wubu_t_use_rotation', action='store_true', default=False); wubu_t_group.add_argument('--wubu_t_phi_influence_curvature', action='store_true', default=False); wubu_t_group.add_argument('--wubu_t_phi_influence_rotation_init', action='store_true', default=False)
    # WuBu-D-Global-Video (For Feature Discriminator)
    wubu_d_global_vid_group = parser.add_argument_group('WuBu-D-Global-Video (Feature Discriminator)')
    wubu_d_global_vid_group.add_argument('--wubu_d_global_video_num_levels', type=int, default=3)
    wubu_d_global_vid_group.add_argument('--wubu_d_global_video_hyperbolic_dims', nargs='+', type=int, default=[256, 128, 64])
    wubu_d_global_vid_group.add_argument('--wubu_d_global_video_initial_curvatures', nargs='+', type=float, default=[0.8, 0.6, 0.4])
    wubu_d_global_vid_group.add_argument('--wubu_d_global_video_use_rotation', action='store_true', default=False)
    wubu_d_global_vid_group.add_argument('--wubu_d_global_video_phi_influence_curvature', action='store_true', default=False)
    wubu_d_global_vid_group.add_argument('--wubu_d_global_video_phi_influence_rotation_init', action='store_true', default=False)

    # Common WuBu params (can be overridden by stack-specific if needed, but often shared)
    for prefix in ["wubu_s", "wubu_m", "wubu_t", "wubu_d_global_video"]:
        group = parser.add_argument_group(f'WuBu Common Params for {prefix.upper()}')
        group.add_argument(f'--{prefix}_initial_scales', nargs='+', type=float, default=[1.0])
        group.add_argument(f'--{prefix}_initial_spread_values', nargs='+', type=float, default=[0.1])
        group.add_argument(f'--{prefix}_boundary_points_per_level', nargs='+', type=int, default=[4 if prefix=="wubu_s" else 0]) 

    # --- Group: Q-Learning Controller & Heuristics ---
    q_heur_group = parser.add_argument_group('Q-Learning Controller & Heuristics')
    q_heur_group.add_argument('--q_controller_enabled',action='store_true', default=True)
    q_heur_group.add_argument('--q_config_gen_json', type=str, default=None, help="JSON string or path to JSON for Generator Q-controller overrides.")
    q_heur_group.add_argument('--q_config_disc_json', type=str, default=None, help="JSON string or path to JSON for Discriminator Q-controller overrides.")
    q_heur_group.add_argument('--q_config_lkl_json', type=str, default=None, help="JSON string or path to JSON for LambdaKL Q-controller overrides.")
    q_heur_group.add_argument('--reset_q_controllers_on_load', action='store_true', default=False) 
    q_heur_group.add_argument('--reset_lkl_q_controller_on_load', action='store_true', default=False) 
    q_heur_group.add_argument('--lambda_kl_update_interval', type=int, default=10) 
    q_heur_group.add_argument('--min_lambda_kl_q_control', type=float, default=1e-5) 
    q_heur_group.add_argument('--max_lambda_kl_q_control', type=float, default=0.1)  
    q_heur_group.add_argument('--q_lkl_scale_options', nargs='+', type=float, default=[0.75, 0.9, 1.0, 1.1, 1.25]) 
    q_heur_group.add_argument('--q_lkl_lr_mom_probation_steps', type=int, default=15)
    q_heur_group.add_argument('--q_lkl_action_probation_steps', type=int, default=15)
    q_heur_group.add_argument('--enable_heuristic_interventions', action='store_true', default=True)
    q_heur_group.add_argument('--heuristic_check_interval', type=int, default=10)
    q_heur_group.add_argument('--heuristic_short_term_history_len', type=int, default=7)
    q_heur_group.add_argument('--heuristic_trigger_count_thresh', type=int, default=2)
    q_heur_group.add_argument('--heuristic_d_strong_thresh', type=float, default=0.20)
    q_heur_group.add_argument('--heuristic_d_weak_thresh', type=float, default=1.2)
    q_heur_group.add_argument('--heuristic_d_very_weak_thresh', type=float, default=2.0)
    q_heur_group.add_argument('--heuristic_g_stalled_thresh', type=float, default=1.8)
    q_heur_group.add_argument('--heuristic_g_winning_thresh', type=float, default=0.15)
    q_heur_group.add_argument('--heuristic_g_very_much_winning_thresh', type=float, default=0.03)
    q_heur_group.add_argument('--heuristic_kl_high_thresh', type=float, default=15.0)
    q_heur_group.add_argument('--heuristic_recon_stagnation_improvement_thresh_rel', type=float, default=0.001)
    q_heur_group.add_argument('--target_good_recon_thresh_heuristic_video', type=float, default=0.015)
    q_heur_group.add_argument('--heuristic_q_reward_stagnation_thresh', type=float, default=-0.3)
    q_heur_group.add_argument('--heuristic_recon_boost_factor_video', type=float, default=1.5)
    q_heur_group.add_argument('--lambda_feat_match_heuristic_video', type=float, default=0.1)
    q_heur_group.add_argument('--lambda_g_easy_win_penalty_heuristic_video', type=float, default=1.0)
    q_heur_group.add_argument('--g_easy_win_penalty_eps_denom', type=float, default=1e-5)
    q_heur_group.add_argument('--max_g_easy_win_penalty_abs', type=float, default=10.0)
    q_heur_group.add_argument('--heuristic_active_d_lr_boost_factor', type=float, default=2.0)
    q_heur_group.add_argument('--heuristic_d_q_explore_boost_epsilon', type=float, default=0.75)
    q_heur_group.add_argument('--heuristic_d_q_explore_duration', type=int, default=15)
    q_heur_group.add_argument('--heuristic_min_lambda_gan_factor', type=float, default=0.5)
    q_heur_group.add_argument('--heuristic_max_lambda_gan_factor', type=float, default=1.5)
    q_heur_group.add_argument('--force_start_epoch_on_load', type=int, default=None)
    q_heur_group.add_argument('--force_start_gstep_on_load', type=int, default=None)

    # --- Group: Logging, Sampling, Validation & Checkpointing ---
    log_group = parser.add_argument_group('Logging and Saving')
    log_group.add_argument('--wandb',action='store_true', default=True); log_group.add_argument('--wandb_project',type=str,default='WuBuGAADHybridGenV03'); log_group.add_argument('--wandb_run_name',type=str,default=None); log_group.add_argument('--log_interval',type=int, default=10); log_group.add_argument('--save_interval',type=int, default=1000); log_group.add_argument('--save_epoch_interval', type=int, default=1); log_group.add_argument('--validation_interval_epochs', type=int, default=1); log_group.add_argument('--disable_val_tqdm', action='store_true', default=False)
    log_group.add_argument('--wandb_log_train_recon_interval', type=int, default=50); log_group.add_argument('--train_target_log_freq_multiplier', type=int, default=5); log_group.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=100); log_group.add_argument('--use_lpips_for_verification', action='store_true', default=True); log_group.add_argument('--validation_video_path', type=str, default=None); log_group.add_argument('--validation_split_fraction', type=float, default=0.05); log_group.add_argument('--val_block_size', type=int, default=10); log_group.add_argument('--val_primary_metric', type=str, default="avg_val_psnr", choices=["avg_val_recon_mse_dft", "avg_val_recon_mse_dct", "avg_val_recon_mse_pixel", "avg_val_psnr", "avg_val_ssim", "avg_val_lpips"]); log_group.add_argument('--num_val_samples_to_log', type=int, default=4); log_group.add_argument('--demo_num_samples', type=int, default=4)
    log_group.add_argument('--data_fraction', type=float, default=1.0, help="Fraction of the dataset to use (0.0 to 1.0).")
    
    parsed_args = parser.parse_args()
    parsed_args.image_h_w_tuple = (parsed_args.image_h, parsed_args.image_w)
    if parsed_args.use_wubu_motion_branch and not OPTICAL_FLOW_AVAILABLE: parser.error("Motion branch needs optical flow, but torchvision.models.optical_flow unavailable.")
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.optical_flow_net_type not in FLOW_MODELS: parser.error(f"Optical flow net type '{parsed_args.optical_flow_net_type}' not in available: {list(FLOW_MODELS.keys())}")
    if (parsed_args.use_dft_features_appearance or parsed_args.use_dct_features_appearance) and (parsed_args.spectral_patch_size_h <= 0 or parsed_args.spectral_patch_size_w <= 0): parser.error("If using spectral features (DFT/DCT), --spectral_patch_size_h and --spectral_patch_size_w must be positive.")
    if not (parsed_args.use_dft_features_appearance or parsed_args.use_dct_features_appearance): logging.warning("Neither DFT nor DCT features are enabled for appearance. Generator will default to pixel output if configured for it.")

    wubu_prefixes_to_validate = ["wubu_s", "wubu_m", "wubu_t", "wubu_d_global_video"]
    for prefix in wubu_prefixes_to_validate:
        num_levels_attr = f"{prefix}_num_levels"; num_levels_val = getattr(parsed_args, num_levels_attr, 0)
        if num_levels_val > 0: validate_wubu_config_for_argparse(parsed_args, prefix, parser)
    
    if parsed_args.wubu_s_num_levels > 0 and parsed_args.wubu_s_hyperbolic_dims: parsed_args.wubu_s_output_dim = parsed_args.wubu_s_hyperbolic_dims[-1]
    else: parsed_args.wubu_s_output_dim = parsed_args.encoder_initial_tangent_dim; parsed_args.wubu_s_num_levels = 0
    if parsed_args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE and parsed_args.wubu_m_num_levels > 0 and parsed_args.wubu_m_hyperbolic_dims: parsed_args.wubu_m_output_dim = parsed_args.wubu_m_hyperbolic_dims[-1]
    else: parsed_args.wubu_m_output_dim = 0; parsed_args.wubu_m_num_levels = 0
    valid_stats = {'mag_mean', 'angle_mean', 'mag_std', 'angle_std'};
    if any(s not in valid_stats for s in parsed_args.flow_stats_components): parser.error(f"Invalid flow_stats_components. Allowed: {valid_stats}. Got: {parsed_args.flow_stats_components}")
    
    # Process JSON config overrides for Q-controllers
    for q_config_key in ['q_config_gen', 'q_config_disc', 'q_config_lkl']:
        json_arg_val = getattr(parsed_args, f"{q_config_key}_json", None)
        override_dict = {}
        if json_arg_val:
            if os.path.isfile(json_arg_val):
                try:
                    with open(json_arg_val, 'r') as f: override_dict = json.load(f)
                except Exception as e: logging.warning(f"Could not load Q-config JSON from file {json_arg_val}: {e}")
            else:
                try: override_dict = json.loads(json_arg_val)
                except Exception as e: logging.warning(f"Could not parse Q-config JSON string for {q_config_key}: {e}")
        setattr(parsed_args, q_config_key, override_dict) # Store the dict

    return parsed_args

def main():
    args = parse_arguments() # Assumes parse_arguments() is defined as provided earlier
    ddp_active = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    if ddp_active:
        rank=int(os.environ["RANK"]); local_rank=int(os.environ["LOCAL_RANK"]); world_size=int(os.environ["WORLD_SIZE"])
        init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        device=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == 'cuda': torch.cuda.set_device(device)
    else: rank=0; local_rank=0; world_size=1; device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda' and torch.cuda.is_available(): torch.cuda.set_device(device)
    am_main_process = (rank == 0)
    base_logger_name = "WuBuGAADHybridGenV03"
    root_logger = logging.getLogger(); [root_logger.removeHandler(h) for h in list(root_logger.handlers)]; specific_logger = logging.getLogger(base_logger_name); [specific_logger.removeHandler(h) for h in list(specific_logger.handlers)]
    log_level = logging.INFO if am_main_process else logging.WARNING; logging.basicConfig(level=log_level, format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s', force=True)
    current_logger_main = logging.getLogger(f"{base_logger_name}.Main")
    current_logger_main.info(f"--- {base_logger_name} (R{rank}/{world_size}, Dev {device}, DDP:{ddp_active}, AMP:{args.use_amp}, DFT:{args.use_dft_features_appearance}, DCT:{args.use_dct_features_appearance}) ---")
    seed_everything(args.seed, rank, world_size)
    if args.detect_anomaly: torch.autograd.set_detect_anomaly(True); current_logger_main.warning("Autograd anomaly detection ENABLED.")
    if am_main_process: current_logger_main.info(f"Effective Args: {vars(args)}")
    if am_main_process and args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name if args.wandb_run_name else f"wubuvid_v03_{datetime.now().strftime('%y%m%d_%H%M')}"
        try: wandb.init(project=args.wandb_project, name=run_name, config=vars(args), resume="allow", id=wandb.util.generate_id() if wandb.run is None else wandb.run.id); current_logger_main.info(f"WandB initialized: Run '{run_name}', Project '{args.wandb_project}'")
        except Exception as e_wandb: current_logger_main.error(f"WandB initialization failed: {e_wandb}", exc_info=True); args.wandb = False
            
    video_config = {"image_size": args.image_h_w_tuple, "num_channels": args.num_channels, "num_input_frames": args.num_input_frames, "num_predict_frames": args.num_predict_frames, "wubu_s_output_dim": args.wubu_s_output_dim, "wubu_m_output_dim": args.wubu_m_output_dim }
    gaad_appearance_config = {"num_regions": args.gaad_num_regions, "decomposition_type": args.gaad_decomposition_type, "min_size_px": args.gaad_min_size_px}
    gaad_motion_config = {"num_regions": args.gaad_motion_num_regions, "decomposition_type": args.gaad_motion_decomposition_type, "min_size_px": args.gaad_min_size_px} if args.use_wubu_motion_branch and OPTICAL_FLOW_AVAILABLE else None
    wubu_s_config_enc = _configure_wubu_stack(args, "wubu_s"); wubu_t_config = _configure_wubu_stack(args, "wubu_t"); wubu_m_config = _configure_wubu_stack(args, "wubu_m")

    model = WuBuGAADHybridGenNet(args, video_config, gaad_appearance_config, gaad_motion_config, wubu_s_config_enc, wubu_t_config, wubu_m_config).to(device)
    
    # --- Discriminator Setup ---
    def _get_disc_sub_config(args_ref: argparse.Namespace, variant_name: str) -> Dict:
        cfg = {"architecture_variant": variant_name, "apply_spectral_norm": args_ref.disc_apply_spectral_norm}
        # Common settings that might be used by multiple variants
        cfg["base_disc_channels"] = args_ref.disc_base_disc_channels
        cfg["max_disc_channels"] = args_ref.disc_max_disc_channels
        cfg["temporal_kernel_size"] = args_ref.disc_temporal_kernel_size
        
        if variant_name == "default_pixel_cnn":
            cfg.update({ # Specific to RegionalDiscriminator (pixel-based)
                         "target_video_disc_final_feature_dim": args_ref.disc_target_final_feature_dim,
                         "max_video_disc_downsample_layers": args_ref.max_video_disc_downsample_layers, 
                         "use_gaad_film_condition": args_ref.disc_use_gaad_film_condition,
                         "gaad_condition_dim_disc": args_ref.disc_gaad_condition_dim_disc})
        elif variant_name == "global_wubu_video_feature":
            cfg.update({"disc_use_global_stats_aux_video_global_wubu": args_ref.disc_use_global_stats_aux_video_global_wubu,
                        "disc_global_stats_mlp_hidden_dim_video_global_wubu": args_ref.disc_global_stats_mlp_hidden_dim_video_global_wubu})
        # Add elif for other variants (e.g., MultiScaleVideoDiscriminator if implemented)
        # elif variant_name == "multi_scale_pixel_cnn":
        #    cfg.update({ ... args for multi_scale_pixel_cnn ...})
        return cfg

    primary_disc_actual_config = _get_disc_sub_config(args, args.primary_disc_architecture_variant)
    discriminator_primary = VideoDiscriminatorWrapper(args, video_config, gaad_appearance_config, primary_disc_actual_config).to(device)
    
    alt_disc_actual_config = _get_disc_sub_config(args, args.alt_disc_architecture_variant)
    discriminator_alternative = VideoDiscriminatorWrapper(args, video_config, gaad_appearance_config, alt_disc_actual_config).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)
        wandb.watch(discriminator_primary, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)
        wandb.watch(discriminator_alternative, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False)

    if ddp_active:
        model = DDP(model, device_ids=[local_rank] if device.type=='cuda' else None, output_device=local_rank if device.type=='cuda' else None, find_unused_parameters=args.ddp_find_unused_params_g)
        discriminator_primary = DDP(discriminator_primary, device_ids=[local_rank] if device.type=='cuda' else None, output_device=local_rank if device.type=='cuda' else None, find_unused_parameters=args.ddp_find_unused_params_d)
        discriminator_alternative = DDP(discriminator_alternative, device_ids=[local_rank] if device.type=='cuda' else None, output_device=local_rank if device.type=='cuda' else None, find_unused_parameters=args.ddp_find_unused_params_d)
        
    actual_video_path = args.video_data_path; demo_file_name = "dummy_video_hybridgen_v03.mp4"
    if "demo_video_data" in args.video_data_path: actual_video_path = str(Path(args.video_data_path) / demo_file_name)
    if am_main_process:
        Path(actual_video_path).parent.mkdir(parents=True, exist_ok=True)
        if "demo_video_data" in str(args.video_data_path) and not Path(actual_video_path).exists():
            if IMAGEIO_AVAILABLE and imageio is not None:
                current_logger_main.info(f"Creating dummy video: {actual_video_path}...")
                min_raw_frames_needed = (args.num_input_frames + args.num_predict_frames -1) * args.frame_skip + 1
                num_dummy_frames = max(100, min_raw_frames_needed + 50); dummy_h, dummy_w = int(args.image_h), int(args.image_w)
                try:
                    with imageio.get_writer(actual_video_path, fps=15, quality=8, macro_block_size=16) as video_writer:
                        for _ in range(num_dummy_frames): video_writer.append_data(np.random.randint(0,255, (dummy_h,dummy_w,args.num_channels), dtype=np.uint8))
                    current_logger_main.info(f"Dummy video with {num_dummy_frames} frames created at {actual_video_path}.")
                except Exception as e_imageio_write: current_logger_main.error(f"Error creating dummy video: {e_imageio_write}", exc_info=True)
            else: current_logger_main.error("imageio library not available. Cannot create dummy video.")
    if ddp_active: torch.distributed.barrier()
    if not Path(actual_video_path).is_file(): current_logger_main.error(f"Video path '{actual_video_path}' is not a file. Exiting."); sys.exit(1)
        
    total_frames_per_sample = args.num_input_frames + args.num_predict_frames
    try: full_dataset = VideoFrameDataset(video_path=actual_video_path, num_frames_total=total_frames_per_sample, image_size=args.image_h_w_tuple, frame_skip=args.frame_skip, data_fraction=args.data_fraction)
    except Exception as e: current_logger_main.error(f"Failed to initialize main Dataset from '{actual_video_path}': {e}", exc_info=True); sys.exit(1)
    if not full_dataset or len(full_dataset) == 0: current_logger_main.error(f"Main dataset from '{actual_video_path}' is empty. Exiting."); sys.exit(1)

    train_dataset, val_dataset = full_dataset, None; num_total_samples = len(full_dataset)
    val_video_files_list = [] 
    if args.validation_video_path:
        val_dir_path_obj_vid = Path(args.validation_video_path)
        if val_dir_path_obj_vid.is_dir(): [val_video_files_list.extend([str(p) for p in val_dir_path_obj_vid.rglob(ext)]) for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]] 
        elif val_dir_path_obj_vid.is_file(): val_video_files_list.append(str(val_dir_path_obj_vid))
    if val_video_files_list:
        try:
            val_dataset_candidate = VideoFrameDataset(video_path=val_video_files_list[0], num_frames_total=total_frames_per_sample, image_size=args.image_h_w_tuple, frame_skip=args.frame_skip, data_fraction=1.0) 
            if len(val_dataset_candidate) > 0: val_dataset = val_dataset_candidate; current_logger_main.info(f"Using separate validation video: {val_video_files_list[0]}, Samples: {len(val_dataset)}")
            else: current_logger_main.warning(f"Validation video '{val_video_files_list[0]}' is empty. Splitting from train.")
        except Exception as e: current_logger_main.warning(f"Could not load validation dataset from '{val_video_files_list[0]}': {e}. Splitting from train.")
    if val_dataset is None and args.validation_split_fraction > 0.0 and num_total_samples > 10:
        num_val = int(num_total_samples * args.validation_split_fraction); num_train = num_total_samples - num_val
        if num_train > 0 and num_val > 0: train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [num_train, num_val], generator=torch.Generator().manual_seed(args.seed + rank))
        else: current_logger_main.warning("Random split failed. No validation set."); train_dataset = full_dataset
    if am_main_process: current_logger_main.info(f"Final dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}")

    worker_init_fn_seeded = functools.partial(seed_worker_init_fn, base_seed=args.seed, rank=rank, world_size=world_size) if args.num_workers > 0 else None
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_active else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded, drop_last=True if ddp_active else False)
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=(device.type == 'cuda'), drop_last=False, worker_init_fn=worker_init_fn_seeded)
                                
    trainer = HybridTrainer(model, discriminator_primary, discriminator_alternative, device, train_loader, val_loader, args, rank, world_size, ddp_active)
                            
    start_global_step, start_epoch = trainer.load_checkpoint(args.load_checkpoint) if args.load_checkpoint else (0,0)
    if start_global_step == 0 and start_epoch == 0 and args.q_controller_enabled: 
        if am_main_process: trainer.logger.info("Fresh run. Initializing Q-controllers on probation.")
        controllers_to_init_fresh = [trainer.q_controller_gen, trainer.q_controller_d_primary, trainer.q_controller_d_alt, trainer.lambda_kl_q_controller]
        initial_dummy_losses = {'loss_g_total':1.0,'loss_g_recon':1.0,'loss_g_kl':0.1,'loss_g_adv':0.7, 'loss_d_total':0.7,'loss_d_real':0.7,'loss_d_fake':0.7}
        for i_qc, qc_obj in enumerate(controllers_to_init_fresh):
            if qc_obj:
                is_gen_q = (i_qc == 0); qc_obj.reset_q_learning_state(True, True, "Fresh Run Q-Init", True) 
                if hasattr(qc_obj, 'set_initial_losses'): qc_obj.set_initial_losses(initial_dummy_losses, is_generator_q=is_gen_q if i_qc < 3 else False) 
                if qc_obj == trainer.lambda_kl_q_controller and hasattr(qc_obj, 'set_initial_lambda_kl_metrics'):
                    initial_val_metric_val = 0.0 if trainer.is_val_metric_higher_better else 1.0
                    initial_lkl_metrics = {'avg_recon':1.0,'avg_kl_div':0.1,'avg_d_total':0.7,'val_metric':initial_val_metric_val,'current_lambda_kl_val':trainer.lambda_kl_base}
                    qc_obj.set_initial_lambda_kl_metrics(initial_lkl_metrics)

    try:
        trainer.train(start_epoch=start_epoch, initial_global_step=start_global_step)
    except KeyboardInterrupt: current_logger_main.info(f"Rank {rank}: Training interrupted by user (KeyboardInterrupt).")
    except Exception as e: current_logger_main.error(f"Rank {rank}: Training loop crashed: {e}", exc_info=True)
    finally:
        if am_main_process:
            current_logger_main.info("Finalizing run...")
            final_metrics_to_save = trainer.last_val_metrics.copy() if hasattr(trainer, 'last_val_metrics') and trainer.last_val_metrics else {}
            if hasattr(trainer, 'best_val_metric_val'): final_metrics_to_save['best_val_metric_val_at_end'] = trainer.best_val_metric_val
            if hasattr(trainer, '_save_checkpoint') and callable(getattr(trainer, '_save_checkpoint')) : trainer._save_checkpoint(metrics=final_metrics_to_save) 
            else: current_logger_main.error("CRITICAL: trainer object does not have _save_checkpoint in finally block!")
            if args.epochs > 0 and hasattr(trainer, 'sample') and hasattr(trainer, 'global_step') and trainer.global_step > 0 and args.demo_num_samples > 0:
                current_logger_main.info("Generating final demo samples...")
                try:
                    pred_pixels = trainer.sample(num_samples=args.demo_num_samples)
                    if pred_pixels is not None and pred_pixels.numel() > 0 and pred_pixels.shape[0] > 0:
                        save_dir = Path(args.checkpoint_dir) / f"demo_samples_v03_{args.primary_disc_architecture_variant}_{args.alt_disc_architecture_variant}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        num_frames_to_save_per_sample = min(pred_pixels.shape[1], 3)
                        for b_idx in range(min(args.demo_num_samples, pred_pixels.shape[0])):
                            for frame_s_idx in range(num_frames_to_save_per_sample):
                                save_image((pred_pixels[b_idx, frame_s_idx].cpu().clamp(-1, 1) + 1) / 2.0, save_dir / f"demo_s{b_idx}_f{frame_s_idx}_ep{trainer.current_epoch+1}_gs{trainer.global_step}.png")
                        current_logger_main.info(f"Saved demo sample frames to {save_dir}")
                        if args.wandb and WANDB_AVAILABLE and wandb.run: trainer._log_samples_to_wandb("final_demo_video", pred_pixels, pred_pixels.shape[1], args.demo_num_samples)
                    else: current_logger_main.info("No demo samples generated or pred_pixels was None/empty.")
                except Exception as e_demo: current_logger_main.error(f"Demo sampling or saving error: {e_demo}", exc_info=True)
        if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run: wandb.finish()
        if ddp_active and is_initialized(): destroy_process_group()
        current_logger_main.info(f"Rank {rank}: {base_logger_name} (v0.3) script finished.")


if __name__ == "__main__":
    use_motion_branch_requested = '--use_wubu_motion_branch' in sys.argv
    if use_motion_branch_requested and not OPTICAL_FLOW_AVAILABLE:
        print("FATAL ERROR: Motion branch (--use_wubu_motion_branch) requested, but torchvision.models.optical_flow is unavailable. Please install it or disable the motion branch.")
        sys.exit(1)
    if ('--use_dct_features_appearance' in sys.argv or '--use_dct_features_appearance=True' in sys.argv) and not TORCH_DCT_AVAILABLE:
        print("FATAL ERROR: DCT features (--use_dct_features_appearance) requested, but torch-dct is unavailable. Please install it or disable DCT.")
        sys.exit(1)
    main()


