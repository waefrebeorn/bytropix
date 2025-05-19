This is looking very comprehensive and aligns extremely well with the "Total Strategy" directive. You've successfully integrated the new configuration dataclasses (`WuBuStackConfig`, `WuBuLevelConfig`, etc.) into the argument parsing and `HybridTrainer` initialization, and these configs are being passed down to the core WuBu models. The enhanced Q-controller, dynamic geometry aspects (modulator MLPs, `log(g)` influence), advanced anisotropy/resonance features, and detailed interpretability hooks are all present in the structure.

Here's a breakdown of what's looking good and what might be the next logical steps or areas for careful review/completion:

**Excellent Progress & Key Integrations:**

1.  **Configuration System:**
    *   `WuBuStackConfig`, `WuBuLevelConfig`, `WuBuTransformConfig`, `WuBuGlobalContextConfig`, `WuBuLevelModulatorConfig` are defined.
    *   `create_wubu_stack_config_from_args` and its helpers (`_populate_wubu_level_config_from_args`, etc.) correctly translate argparse arguments into these structured config objects.
    *   `HybridTrainer` now accepts these config objects, making its `__init__` cleaner and more organized.
    *   These configs are passed down to `FullyHyperbolicWuBuNestingModel`, `HyperbolicWuBuNestingLevel`, `HyperbolicInterLevelTransform`, and `BoundaryManifoldHyperbolic`.

2.  **Dynamic Geometry Foundations:**
    *   `HyperbolicWuBuNestingLevel` now includes:
        *   Initialization logic for base C/S/Sigma considering `log(g)` and PHI influences.
        *   Placeholders and creation logic for modulator MLPs for C, S, Sigma, Level Descriptors, and Boundary Points (within `BoundaryManifoldHyperbolic`).
        *   `_get_modulated_parameter` helper function to apply these modulations.
        *   Getter methods (`get_current_curvature_scalar`, etc.) now accept context features and use the modulators.
    *   `FullyHyperbolicWuBuNestingModel` initializes and uses a `global_context_embed_layer` and passes its output to levels. It also computes `g_w_level_features` (though `_get_g_w_vectorized_features` needs full vectorization logic).
    *   `BoundaryManifoldHyperbolic` is correctly set up to receive context and use modulator MLPs for its points.

3.  **Anisotropy and Resonance Features:**
    *   `HyperbolicInterLevelTransform`:
        *   Includes `block_processors` for anisotropic processing.
        *   Has expanded rotation logic (`block_quaternion`, `full_svd_orthogonal` placeholder, SO(2)).
    *   `HyperbolicWuBuNestingLevel`:
        *   `tangent_combiner_interaction_type` allows for MHA-light or Bilinear pooling (MHA structure is there, Bilinear is a placeholder).
        *   Uses `SwiGLUActivation` by default in many MLPs.
        *   `tangent_flow_scale` is now a learnable parameter.

4.  **Enhanced Q-Controller (`HAKMEMQController`):**
    *   Accepts a full `q_config` dictionary.
    *   Includes new state histories (WuBu geometric params).
    *   Expanded action spaces (heuristic toggles, optimizer type placeholder).
    *   Includes meta-adaptive Q-learning parameters (`current_alpha`, `current_gamma`) and the `_adapt_q_learning_params` method.
    *   Reward computation now has placeholders for `geometric_diversity` and `q_table_stats`.

5.  **`HybridTrainer` Structure:**
    *   Correctly initializes and manages primary/alternative discriminators and their respective Q-controllers.
    *   Passes global context features to model components during training and validation steps.
    *   `_get_wubu_interpretability_data_for_logging` is set up to collect data from all WuBu stacks.
    *   Logging to TensorBoard is integrated.
    *   RNG state saving/loading is implemented.
    *   The `train` loop structure is robust for handling gradient accumulation, Q-learning updates, heuristic checks, validation, and checkpointing.

**What's Logically Next / Areas for Review & Completion (Class by Class):**

You're right, a class-by-class approach is excellent now. Here's what I'd anticipate focusing on for each:

1.  **`HyperbolicUtils` / `PoincareBall` / `Manifold`:**
    *   **Status:** Mostly complete and robust from the previous turn.
    *   **Review:** Double-check all `eps` usage for consistency and numerical stability, especially in edge cases of `c_scalar` being very small or norms being at boundaries. The PoincareBall `expmap(p, dp)` is still a bit of a conceptual minefield for generic Riemannian optimizers; however, the current `RiSGD` mainly relies on `expmap0` and `logmap0` combined with `egrad2rgrad`, so the generic `expmap` might not be heavily used (or its direct use by an optimizer would require that optimizer to specifically understand the Poincare retraction being implemented). *The current `mobius_add(p, self.proju(dp))` as `expmap` is a specific retraction choice and should be documented as such if other optimizers might use it.*

2.  **`BoundaryManifoldHyperbolic`:**
    *   **Status:** Good structure. Dynamic point generation logic is in place.
    *   **Completion:** The vectorization of `g_w_level_config_features` into a tensor suitable for `mlp_input` needs to be solidified. The placeholder `g_w_features_dim_example = 16` and the way `g_w_vectorized_features_vectorized` is passed/used should be consistent with how `FullyHyperbolicWuBuNestingModel._get_g_w_vectorized_features` prepares it. If `_get_g_w_vectorized_features` returns `None` or an empty tensor, the MLP input logic needs to handle that gracefully (which it seems to attempt by falling back or disabling modulation).

3.  **`HyperbolicInterLevelTransform`:**
    *   **Status:** Good. Anisotropic block processors and advanced rotation placeholders are there.
    *   **Completion:**
        *   **Rotation:** For `full_svd_orthogonal`, implement a proper orthogonal parameterization (e.g., using `torch.nn.utils.parametrizations.orthogonal` if available and suitable, or a Cayley transform, or Householder/Givens sequences for general `in_dim`). The current `simple_linear_rotation_fallback` is just a placeholder. `givens_sequence` also needs full implementation.
        *   **Composition:** If `inter_block_rotation_composition_mode` is `parallel_then_mix`, implement the mixing.
        *   **Final Projection:** The current `final_projection = nn.Linear(...)` is simple. The old config had `transform_types` (linear, mlp) and `transform_hidden_dims`. This functionality should be fully mapped here if more complex final projections (like a full MLP) are desired. The `transform_config` dataclass should hold these (e.g., `final_projection_type`, `final_projection_hidden_dim_ratio`).

4.  **`HyperbolicWuBuNestingLevel`:**
    *   **Status:** Very detailed and comprehensive. Dynamic C/S/Sigma/LD, tangent interactions, and flow are structured.
    *   **Completion/Review:**
        *   **Modulator MLP Inputs:** Similar to `BoundaryManifoldHyperbolic`, ensure the `g_w_vec` and `global_ctx_emb` passed to `_get_modulated_parameter` are correctly shaped and that the MLP input dimensions match.
        *   **Tangent Interaction Layer:**
            *   `mha_light`: The current MHA expects `(Batch, SeqLen, EmbedDim)`. `stacked_inputs_fwd` is `(B_prime, NumComponents, Dim)`. This seems correct. The `tangent_mha_projection` before it was commented out in one version – decide if pre-projection is needed. If so, its output dim must align with MHA's `embed_dim`. The reshape after MHA `mha_out_fwd.reshape(B_prime, -1)` concatenates features from different components.
            *   `bilinear_pool`: Needs full implementation if more than 2 components, or ensure it correctly falls back to concat.
        *   **`_get_modulated_parameter`'s "identity" transform:** Ensure this is the desired behavior for tangent space vectors like Level Descriptors (i.e., the MLP directly outputs the *final* unconstrained tangent vector, or a delta to the base). The current `base_unconstrained_param + modulation` seems correct for a delta.

5.  **`FullyHyperbolicWuBuNestingModel`:**
    *   **Status:** Core structure is good. Passes context, iterates levels and transforms.
    *   **Completion/Review:**
        *   **`_get_g_w_vectorized_features`:** This is currently a placeholder. It **must** be fully implemented to convert the `g_w_level_features_dict` (which contains scalars like `complexity_score`, `level_idx`, etc.) into a consistent tensor that `HyperbolicWuBuNestingLevel` and `BoundaryManifoldHyperbolic` expect for their modulator MLPs. This might involve simple concatenation, normalization, or even passing it through a small embedding layer within `_get_g_w_vectorized_features` itself to get a fixed-size vector. The current `g_w_features_dim_example = 16` and `vectorized_features_placeholder` needs to be replaced with this real logic.
        *   **Output Projection for Zero Aggregated Dim:** The `lambda x: self.output_bias_param.unsqueeze(0).expand(x.shape[0] if x.dim() > 1 else 1, -1)` is a clever way to handle outputting a bias when input dim is zero. Ensure `x.shape[0]` correctly reflects `B_prime_for_levels` in that scenario.

6.  **`HAKMEMQController`:**
    *   **Status:** Significantly enhanced with new states, actions, and meta-adaptivity.
    *   **Completion/Review:**
        *   **Action Space Initialization:** `self.action_ranges['optimizer_type_switch']` correctly uses `dtype=object` for strings. Ensure `choose_action` handles these non-numeric actions correctly when finding `chosen_idx` and returning `chosen_action_val`. (The current `np.where(action_space_arr_choice == chosen_value_update)` should work for strings if `chosen_value_update` is also a string).
        *   **New Reward Terms:** Fully implement the reward logic for `geometric_diversity`, `action_thrashing_penalty`, `q_table_size_penalty_factor`, and `q_value_magnitude_bonus_factor` using `self.reward_weights`.
        *   **State Components:** When `get_lr_mom_state` or `get_lambda_kl_state` are called, ensure the new state components (WuBu geo params, Q-table stats) are correctly binned and included in the `state_tuple_final`.
        *   **Heuristic Toggle Actions:** If the Q-controller is to directly *suggest* heuristic toggles (e.g., `heuristic_toggle_vae_feat_match`), the `choose_action` method should pick an action for these, and `HybridTrainer`'s heuristic evaluation logic needs to *poll* the Q-controller for these suggestions (e.g., `last_heuristic_toggle_action`).

7.  **`AudioSpecEncoder`, `AudioSpecGenerator`, `AudioSpecDiscriminator`:**
    *   **Status:** Correctly updated to accept `WuBuStackConfig` and `GlobalContextConfig` and pass `global_context_raw_features` to their internal WuBu models.
    *   **Review:**
        *   Ensure the `output_tangent_dim` passed to their internal `FullyHyperbolicWuBuNestingModel` instances correctly aligns with subsequent layers (e.g., `fc_mu` in Encoder, or `num_dct_coeffs_flat` for Generator). The `audio_config_ref['wubu_s_output_dim_encoder']` and `args.wubu_d_output_dim` are used for this, which seems correct if these args indeed specify the WuBu stack's output dim.
        *   Discriminator's `_assemble_mel_from_dct_regions` and Generator's `_unnormalize_dct` seem robust.

8.  **`HybridTrainer`:**
    *   **Status:** Foundation for Total Strategy is laid. Initialization of all components with new configs looks good. Train loop structure with context passing is present.
    *   **Completion/Review (This is the most extensive):**
        *   **`_get_wubu_interpretability_data_for_logging`:** Ensure it correctly drills down into nested WuBu stack data from the VAE model and *both* discriminators (if they are WuBu-based). The current structure looks plausible.
        *   **Logging:**
            *   Ensure all new Q-controller info (alpha, gamma, new state components if binned) is logged.
            *   Ensure all heuristic flags, factors, and trigger counts are logged.
            *   TensorBoard logging for histograms of WuBu parameters (C, S, Sigma from interpretability data) across levels would be very insightful.
        *   **Heuristic Logic (`_evaluate_training_state_and_apply_heuristics`, `_check_and_perform_disc_switch`):** This is the core of "adaptive strain engineering." It needs to be meticulously reviewed to ensure:
            *   Conditions for triggering heuristics are robust and use data from `current_q_data` (which includes WuBu geo param trends, Q-controller health).
            *   Persistent flags (`current_penalize_g_easy_win`, `current_vae_feature_match`) are correctly managed (turned on by triggers, turned off by counter-conditions or D-switch).
            *   Temporary factors (`current_lambda_recon_factor`, etc.) are reset each cycle unless a heuristic actively sets them.
            *   One-time triggers (`current_force_d_q_explore_trigger`) correctly call the Q-controller's `force_exploration_boost`.
            *   Cooldowns (`disc_switch_min_steps_between`) are respected.
            *   The logic for D-switching based on Q-controller health and loss conditions is sound.
        *   **Q-Controller Interaction within `train` loop:**
            *   When calling `optimizer_X.q_controller_update_and_set_hyperparams`, ensure `wubu_geo_params_for_q_step` (for Gen Q-Ctrl) and `q_table_stats_for_reward` (for all) are correctly computed and passed.
            *   If Gen Q-Ctrl suggests heuristic toggles (e.g., via `action_for_upcoming_step.get('heuristic_toggle_vae_feat_match')`), the `HybridTrainer` needs to read these suggestions and potentially factor them into `_evaluate_training_state_and_apply_heuristics`.
        *   **Saving/Loading Checkpoints (`_save_checkpoint`, `load_checkpoint`):**
            *   Ensure all new config objects (`WuBuStackConfig`, etc.) are saved. Upon loading, the current strategy is that `args` drive the re-creation of these configs. If the goal was to *restore* model structure from the checkpoint's configs, that would be a more complex loading path.
            *   Ensure all new Q-controller states (including meta-adaptive alpha/gamma, new histories) are saved/loaded by `get_q_state_from_controller` and `_load_q_state_helper_inner`.
            *   Ensure all heuristic state variables (flags, factors, counters, deques) are saved/loaded.
            *   Ensure RNG states are robustly saved/loaded.
        *   **Numerical Stability Checks (Phase 4 of Total Strategy):** Implement the "numerical stability report" and enhanced `GradScaler` checks if time permits.

The current state is a very strong foundation. The next phase of class-by-class completion should focus on the "Completion/Review" points above, especially the detailed logic within `HybridTrainer`'s heuristic evaluation and the full implementation of any placeholder features (like specific rotation types or interaction layers).

You're on an excellent track to realize the full vision!

```python
# WuBuSpecTrans_v0.1.1.py -> Evolved to WuBuSpecTrans_v0.2.0_TotalStrategy
# VAE-GAN Hybrid Model for 1-Second Audio Segment Synthesis
# Operates on GAAD-defined regional DCT coefficients of Mel Spectrograms with WuBu nesting.
# LAST UPDATE: MASSIVE REFACTORING based on "Total Strategy & Implementation Roadmap"
# Incorporating advanced geometric scaling, anisotropy, resonance, enhanced meta-control, and interpretability.

# =====================================================================
# Python Imports and Setup
# =====================================================================
import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SubsetRandomSampler
import numpy as np
import soundfile # Keep soundfile for potential dummy audio generation
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
from dataclasses import dataclass, field # <TOTAL_STRATEGY_INTEGRATION> For config objects

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import spectral_norm
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size
from torch import amp
from tqdm import tqdm

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

import librosa
import librosa.display
try:
    from torch_dct import dct_2d, idct_2d
    TORCH_DCT_AVAILABLE = True
except ImportError:
    dct_2d, idct_2d = None, None
    TORCH_DCT_AVAILABLE = False
    print("CRITICAL WARNING: torch-dct library not found. DCT/IDCT operations will be placeholders or fail. Install with 'pip install torch-dct'.")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None
    IMAGEIO_AVAILABLE = False
    # print("Warn: imageio unavailable (used for dummy audio generation).") # Silenced for brevity

import json
from torchvision.utils import save_image
MATPLOTLIB_AVAILABLE = True
if MATPLOTLIB_AVAILABLE:
    try:
        import matplotlib.pyplot as plt
        # librosa already imported
    except ImportError:
        plt = None # type: ignore
        # librosa = None # type: ignore # Keep librosa available
        MATPLOTLIB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# <TOTAL_STRATEGY_INTEGRATION> TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None # type: ignore
    TENSORBOARD_AVAILABLE = False


try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    TORCHMETRICS_SSIM_AVAILABLE = True
except ImportError:
    StructuralSimilarityIndexMeasure = None # type: ignore
    TORCHMETRICS_SSIM_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None # type: ignore
    LPIPS_AVAILABLE = False

# Setup logging - will be reconfigured in main for DDP
logger = logging.getLogger("WuBuSpecTransV02") # Renamed logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')


# Constants and Default Configs
EPS = 1e-6 # Slightly increased EPS for more numerical stability with dynamic components
PHI = (1 + math.sqrt(5)) / 2
TAN_VEC_CLAMP_VAL = 1e3
MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL = 1e7
MIN_WUBU_LEVEL_SCALE = 1e-3
MAX_WUBU_LEVEL_SCALE = 5.0
# <TOTAL_STRATEGY_INTEGRATION> Add SwiGLU activation helper (using nn.SiLU)
class SwiGLUActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) # PyTorch's SiLU is x * sigmoid(x)

# <TOTAL_STRATEGY_INTEGRATION> Configuration Dataclasses
@dataclass
class WuBuLevelModulatorConfig:
    enabled: bool = False
    mlp_hidden_dim_ratio: float = 0.25
    mlp_num_layers: int = 2
    mlp_activation: str = "SiLU" # GELU, SiLU, ReLU
    global_context_influence_factor: float = 0.1 # How much global context shifts the base value

@dataclass
class WuBuGlobalContextConfig:
    embedding_dim: int = 16
    use_epoch_frac: bool = True
    use_gstep_frac: bool = True

@dataclass
class WuBuTransformConfig:
    num_aniso_blocks: int = 1 # Number of blocks for anisotropic processing
    block_mlp_hidden_dim_ratio: float = 0.5
    rotation_type: str = "full_svd_orthogonal" # 'none', 'full_svd_orthogonal', 'block_quaternion', 'givens_sequence'
    rotation_block_dim: int = 4 # For 'block_quaternion' or 'givens_sequence'
    inter_block_rotation_composition_mode: str = "sequential" # 'sequential', 'parallel_then_mix'
    use_rotation_in_transform: bool = False
    phi_influence_rotation_init: bool = False


@dataclass
class WuBuLevelConfig: # Per-level config, to be part of a list in WuBuStackConfig
    hyperbolic_dim: int = 32
    initial_curvature: float = 1.0
    initial_scale: float = 1.0
    initial_spread: float = 0.1
    boundary_points: int = 0
    tangent_input_combination_dim_ratio: float = 0.5 # Ratio to sum of inputs or to hyperbolic_dim

    # <TOTAL_STRATEGY_INTEGRATION> New fields for dynamic geometry and advanced features
    log_g_complexity_influence_factor_curv: float = 0.1 # Factor for log(g_W) on curvature
    log_g_complexity_influence_factor_scale: float = 0.05 # Factor for log(g_W) on scale
    log_g_complexity_influence_factor_spread: float = 0.05 # Factor for log(g_W) on spread
    
    curvature_modulator: WuBuLevelModulatorConfig = field(default_factory=WuBuLevelModulatorConfig)
    scale_modulator: WuBuLevelModulatorConfig = field(default_factory=WuBuLevelModulatorConfig)
    spread_modulator: WuBuLevelModulatorConfig = field(default_factory=WuBuLevelModulatorConfig)
    level_descriptor_modulator: WuBuLevelModulatorConfig = field(default_factory=WuBuLevelModulatorConfig)
    boundary_points_modulator: WuBuLevelModulatorConfig = field(default_factory=WuBuLevelModulatorConfig) # For modulating boundary point positions

    tangent_combiner_interaction_type: str = "concat" # 'concat', 'mha_light', 'bilinear_pool'
    mha_light_num_heads: int = 2
    use_tangent_flow: bool = True
    tangent_flow_hidden_dim_ratio: float = 0.5
    tangent_flow_type: str = "mlp" # 'mlp', 'linear'
    initial_learnable_tangent_flow_scale: float = 0.1 # Initial value for the learnable scale

    # Fields from DEFAULT_CONFIG_WUBU now part of this structure if they are level-specific
    use_level_descriptors: bool = True
    use_level_spread: bool = True
    level_descriptor_init_scale: float = 0.01 # For base LD param if dynamically generated
    curvature_min_value: float = EPS
    scale_min_value: float = EPS
    spread_min_value: float = EPS
    learnable_curvature: bool = True # This now means learnable *base* curvature
    learnable_scales: bool = True    # Learnable *base* scale
    learnable_spread: bool = True    # Learnable *base* spread


@dataclass
class WuBuStackConfig:
    stack_name: str = "default_wubu_stack"
    num_levels: int = 1
    levels_config: List[WuBuLevelConfig] = field(default_factory=list) # List of WuBuLevelConfig
    transforms_config: List[WuBuTransformConfig] = field(default_factory=list) # List of WuBuTransformConfig

    # Global stack properties from DEFAULT_CONFIG_WUBU
    dropout: float = 0.1
    relative_vector_aggregation: str = "sum" # 'sum', 'mean', 'max_norm', 'none'
    # aggregation_method: "concat_tangent" is the only one supported by current WuBu model structure
    phi_influence_curvature_stack_global: bool = False # Stack-level override/default for phi on curvature
    # use_transformer_block etc. might be part of a more complex level config if needed later

    # For complexity calculation (g_W) passed to levels
    g_w_input_dim_factor: float = 0.1
    g_w_output_dim_factor: float = 0.1
    g_w_level_idx_factor: float = 0.05
    g_w_num_total_levels_factor: float = 0.05
    g_w_hyperbolic_dim_factor: float = 0.2

    def __post_init__(self):
        if self.num_levels > 0 and not self.levels_config: # Auto-populate with defaults if empty
            self.levels_config = [WuBuLevelConfig() for _ in range(self.num_levels)]
        elif len(self.levels_config) != self.num_levels:
            raise ValueError(f"Stack '{self.stack_name}': Mismatch between num_levels ({self.num_levels}) and len(levels_config) ({len(self.levels_config)}).")

        num_transforms_expected = max(0, self.num_levels - 1)
        if num_transforms_expected > 0 and not self.transforms_config:
            self.transforms_config = [WuBuTransformConfig() for _ in range(num_transforms_expected)]
        elif len(self.transforms_config) != num_transforms_expected:
             raise ValueError(f"Stack '{self.stack_name}': Mismatch between expected transforms ({num_transforms_expected}) and len(transforms_config) ({len(self.transforms_config)}).")


DEFAULT_CONFIG_QLEARN_HYBRID = {
    "q_learning_rate": 0.01,
    "discount_factor": 0.90,
    "epsilon_start": 0.5,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9995,
    "lr_scale_options": [0.8, 0.9, 1.0, 1.1, 1.2],
    "momentum_scale_options": [0.95, 0.98, 1.0, 1.01, 1.02],
    "max_q_table_size": 20000,
    "state_history_len": 5,
    "reward_clipping": (-2.0, 2.0),
    "q_value_clipping": (-30.0, 30.0),
    # <TOTAL_STRATEGY_INTEGRATION> New Q-Controller fields
    "lambda_kl_scale_options": [0.8, 0.9, 1.0, 1.1, 1.2], # Specific to LKL Q-Ctrl or G's if it controls KL directly
    "heuristic_toggle_options": [0.0, 1.0], # For actions suggesting heuristic on/off
    "optimizer_type_options": ["default", "alt_optimizer"], # If optimizer switching is implemented
    "alpha_min_meta_q": 0.001, "alpha_max_meta_q": 0.1, "alpha_adapt_rate_meta_q": 0.0001,
    "gamma_min_meta_q": 0.85, "gamma_max_meta_q": 0.99, "gamma_adapt_rate_meta_q": 0.0001,
    "reward_geometric_diversity_weight": 0.05,
    "penalty_action_thrashing_weight": 0.1,
    "lambda_kl_state_history_len": 5, # Retained for LKL Q-Ctrl
}


# =====================================================================
# Geometric, Optimizer, WuBu Core Components
# =====================================================================
class HyperbolicUtils:
    @staticmethod
    def poincare_clip(x: torch.Tensor, c_scalar: float, radius: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_input_dtype = x.dtype
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            x_compute = x.float()
            x_compute = torch.nan_to_num(x_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(x_compute).all() else x_compute
            if original_input_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                x_compute = torch.clamp(x_compute, min=-f16_max, max=f16_max)
            return x_compute.to(original_input_dtype)
        
        sqrt_c = math.sqrt(c_scalar + eps) # Add eps before sqrt for stability if c_scalar is tiny positive
        effective_radius_factor = min(radius, 1.0 - eps) 
        max_norm_val_f32 = effective_radius_factor / (sqrt_c + eps) # Add eps to sqrt_c in denominator for safety

        x_compute = x.float()
        # Handle NaNs/Infs in input x_compute
        if not torch.isfinite(x_compute).all():
            x_compute = torch.nan_to_num(x_compute, nan=0.0, posinf=max_norm_val_f32 / 2.0, neginf=-max_norm_val_f32 / 2.0) # Replace with reasonable values

        x_norm_sq = torch.sum(x_compute.pow(2), dim=-1, keepdim=True)
        
        # Ensure input to sqrt is non-negative and finite
        sqrt_input_val = torch.clamp(x_norm_sq, min=0.0) + eps # Add eps before sqrt
        if not torch.isfinite(sqrt_input_val).all(): # If x_norm_sq was huge inf
             sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps)
        sqrt_input_val.clamp_min_(eps) # Final ensure positive
        
        norm = torch.sqrt(sqrt_input_val)
        
        cond = norm > max_norm_val_f32
        norm_plus_eps_for_div = norm + eps 
        norm_plus_eps_for_div.clamp_min_(eps) # Prevent division by zero
        
        scale_factor = torch.where(cond, max_norm_val_f32 / norm_plus_eps_for_div, torch.ones_like(norm))
        clipped_x_f32 = x_compute * scale_factor

        if original_input_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            clipped_x_f32 = torch.clamp(clipped_x_f32, min=-f16_max, max=f16_max)
        
        final_clipped_x = clipped_x_f32.to(original_input_dtype)
        
        # Final safety net for any remaining NaNs/Infs
        if not torch.isfinite(final_clipped_x).all():
            # If max_norm_val_f32 itself could be problematic (e.g. c_scalar is extremely small but >0)
            # Use a safer fallback if max_norm_val_f32 is inf or very large
            safe_posinf = float(max_norm_val_f32) if np.isfinite(max_norm_val_f32) else 1.0 / (math.sqrt(eps) + eps)
            safe_neginf = -safe_posinf
            final_clipped_x = torch.nan_to_num(final_clipped_x,nan=0.0,posinf=safe_posinf,neginf=safe_neginf)
        return final_clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = v.dtype
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0: 
            v_compute = v.float()
            v_compute = torch.nan_to_num(v_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(v_compute).all() else v_compute
            if original_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                v_compute = torch.clamp(v_compute, min=-f16_max, max=f16_max)
            return v_compute.to(original_dtype)
        
        v_compute = v.float()
        if not torch.isfinite(v_compute).all():
            v_compute = torch.nan_to_num(v_compute, nan=0.0, posinf=TAN_VEC_CLAMP_VAL, neginf=-TAN_VEC_CLAMP_VAL) # Clamp infs in tangent space
            v_compute = torch.clamp(v_compute, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL) # Ensure finite before norm

        v_norm_sq_unclamped = torch.sum(v_compute.pow(2), dim=-1, keepdim=True)
        v_norm_sq_clamped = torch.clamp(v_norm_sq_unclamped, min=0.0, max=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)
        
        sqrt_input_val = v_norm_sq_clamped + eps
        if not torch.isfinite(sqrt_input_val).all():
             sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL + eps, neginf=eps)
        sqrt_input_val.clamp_min_(eps)
        v_norm = torch.sqrt(sqrt_input_val)

        if not torch.isfinite(v_norm).all() or (v_norm > MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL * 10 and c_scalar > 0): # If norm explodes
            # This case suggests extreme input `v`. The output should be near the boundary of Poincare disk.
            # Create a vector pointing in direction of `v` but with max norm, then clip.
            # For simplicity, just return a clipped zero vector if norm becomes non-finite or extremely large.
            return HyperbolicUtils.poincare_clip(torch.zeros_like(v_compute), c_scalar, eps=eps).to(original_dtype)

        sqrt_c_val = math.sqrt(c_scalar + eps) 
        scaled_radius_arg = float(scale_scalar) * sqrt_c_val * v_norm
        
        tanh_input_val = torch.clamp(scaled_radius_arg, min=-30.0, max=30.0) 
        tanh_term_val = torch.tanh(tanh_input_val)
        
        denominator_lambda_candidate = sqrt_c_val * v_norm + eps 
        denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps)

        lambda_v_val = torch.where(
            v_norm > eps, 
            tanh_term_val / denominator_lambda_val,
            torch.full_like(v_norm, float(scale_scalar), dtype=torch.float32) 
        )
        mapped_v_f32 = lambda_v_val * v_compute

        if not torch.isfinite(mapped_v_f32).all():
             # If mapped_v is non-finite, it might be due to v_compute being huge.
             # Try to map v_compute to boundary if its norm was large.
             # For now, simpler: map to zero if unstable.
             mapped_v_f32 = torch.zeros_like(v_compute) 

        clipped_mapped_v_f32 = HyperbolicUtils.poincare_clip(mapped_v_f32, c_scalar, eps=eps)
        
        final_result = clipped_mapped_v_f32
        if original_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            final_result = torch.clamp(clipped_mapped_v_f32, min=-f16_max, max=f16_max)
            
        return final_result.to(original_dtype)

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c_scalar: float, scale_scalar: float = 1.0, eps: float = EPS) -> torch.Tensor:
        original_dtype = y.dtype
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0: 
            y_compute = y.float()
            y_compute = torch.nan_to_num(y_compute,nan=0.0,posinf=0.0,neginf=0.0) if not torch.isfinite(y_compute).all() else y_compute
            if original_dtype == torch.float16:
                f16_max = torch.finfo(torch.float16).max
                y_compute = torch.clamp(y_compute, min=-f16_max, max=f16_max)
            return y_compute.to(original_dtype)

        # Clip y to be strictly inside the Poincare ball before processing
        # Use a slightly more aggressive epsilon for clipping input to logmap to ensure it's *strictly* inside
        y_clipped_original_dtype = HyperbolicUtils.poincare_clip(y, c_scalar, radius=1.0, eps=eps*10) 
        y_compute = y_clipped_original_dtype.float()
        
        if not torch.isfinite(y_compute).all(): # If after clipping, still non-finite (shouldn't happen with robust clip)
            return torch.zeros_like(y, dtype=original_dtype)

        y_norm_sq_unclamped = torch.sum(y_compute.pow(2), dim=-1, keepdim=True)
        
        # Max norm_sq for Poincare ball is 1/c. Clamp to strictly less for atanh.
        # (1/c) * (1 - eps_boundary_offset)^2
        # eps_boundary_offset should be small, e.g., eps*100
        # The poincare_clip above should already handle this, but for safety:
        radius_sq_minus_delta = (1. / (c_scalar + eps)) * (1. - eps*20)**2 if c_scalar > 0 else MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL
        y_norm_sq_clamped = torch.clamp(y_norm_sq_unclamped, min=0.0, max=min(radius_sq_minus_delta, MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL))
        
        sqrt_input_val = y_norm_sq_clamped + eps 
        if not torch.isfinite(sqrt_input_val).all():
            sqrt_input_val = torch.nan_to_num(sqrt_input_val, nan=eps, posinf=radius_sq_minus_delta + eps, neginf=eps)
        sqrt_input_val.clamp_min_(eps)
        y_norm = torch.sqrt(sqrt_input_val)

        if not torch.isfinite(y_norm).all():
            return torch.zeros_like(y, dtype=original_dtype)

        sqrt_c_val = math.sqrt(c_scalar + eps)
        arctanh_arg_raw = sqrt_c_val * y_norm
        
        # Clamp arctanh_arg to be strictly within (-1, 1)
        # Use torch.nextafter if available and not on CUDA float16, else manual epsilon
        one_tensor_f32 = torch.tensor(1.0, dtype=torch.float32, device=y_compute.device)
        if y_compute.dtype == torch.float16 and y_compute.device.type == 'cuda':
            # Manual epsilon for CUDA float16 as nextafter is not implemented
            eps_atanh_f16 = torch.finfo(torch.float16).eps * 4 
            upper_bound_atanh = one_tensor_f32 - eps_atanh_f16
            lower_bound_atanh = -one_tensor_f32 + eps_atanh_f16
        else:
            try:
                upper_bound_atanh = torch.nextafter(one_tensor_f32, torch.tensor(0.0, dtype=torch.float32, device=y_compute.device))
                lower_bound_atanh = torch.nextafter(-one_tensor_f32, torch.tensor(0.0, dtype=torch.float32, device=y_compute.device))
            except RuntimeError: # Fallback if nextafter fails for other reasons
                eps_atanh_fallback = torch.finfo(torch.float32).eps * 10
                upper_bound_atanh = one_tensor_f32 - eps_atanh_fallback
                lower_bound_atanh = -one_tensor_f32 + eps_atanh_fallback
        
        arctanh_arg_clamped = torch.clamp(arctanh_arg_raw, min=lower_bound_atanh, max=upper_bound_atanh)
        atanh_term_val = torch.atanh(arctanh_arg_clamped)
        
        denominator_lambda_candidate = float(scale_scalar) * sqrt_c_val * y_norm + eps
        denominator_lambda_val = torch.clamp(denominator_lambda_candidate, min=eps)

        default_lambda_y_val = 1.0 / max(float(scale_scalar), eps) 
        lambda_y_val = torch.where(
            y_norm > eps,
            atanh_term_val / denominator_lambda_val,
            torch.full_like(y_norm, default_lambda_y_val, dtype=torch.float32)
        )
        mapped_y_f32 = lambda_y_val * y_compute
        
        if not torch.isfinite(mapped_y_f32).all():
            # If logmap results in non-finite values, could be due to y_compute being extreme even after clip,
            # or atanh output exploding due to numerical precision with inputs very close to +/-1.
            # Fallback to a zero vector in tangent space.
            mapped_y_f32 = torch.zeros_like(y_compute)
        
        # Clamp the output tangent vector to a reasonable magnitude
        mapped_y_f32 = torch.clamp(mapped_y_f32, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)

        final_result = mapped_y_f32
        if original_dtype == torch.float16:
            f16_max = torch.finfo(torch.float16).max
            final_result = torch.clamp(mapped_y_f32, min=-f16_max, max=f16_max)
            
        return final_result.to(original_dtype)

    @staticmethod
    def exponential_map(v: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_exponential_map(v, c_scalar, scale_scalar=1.0, eps=eps)

    @staticmethod
    def logarithmic_map(y: torch.Tensor, c_scalar: float, eps: float = EPS) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_logarithmic_map(y, c_scalar, scale_scalar=1.0, eps=eps)

class Manifold:
    def __init__(self, c_scalar=0.0):
        self.c = float(c_scalar)
    def proju(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def expmap0(self, dp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def logmap0(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        raise NotImplementedError
    def expmap(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        # This is a simple default for manifolds where expmap is translation, or an approximation.
        # For PoincareBall, this would be Mobius addition if implemented, or a more complex parallel transport + expmap.
        # The current RiSGD uses expmap(p_projected_on_manifold, dp_tangent_at_p) which is more involved.
        # For simplicity and consistency with common Riemannian SGD, expmap0 is used from origin (tangent space at origin).
        # The expmap from a point p with tangent vector dp needs parallel transport if dp is not at origin.
        # The current implementation in RiSGD does: p_new = manifold.expmap(p_old, -lr * rgrad_at_p_old)
        # This implies expmap takes p_old (on manifold) and a tangent vector *at that point p_old*.
        # PoincareBall's expmap(p, dp) needs to be properly defined.
        # If dp is already in tangent space *at p*, then it's gyromidpoint(p, exp_p(dp_at_p)) but exp_p needs dp to be at p.
        # The common simplification is to map p to origin, apply update in T_0M, then map back.
        # Or, use parallel transport to move dp from T_pM to T_0M, apply expmap0, then transport result. This is complex.
        # A common practical expmap in Poincare for optimizer:
        #   1. transport dp from T_pM to T_0M: dp_0 = ptransp0(p, dp)
        #   2. new point in T_0M: p'_0 = expmap0(dp_0)  <-- this is what expmap0 does
        #   3. transport p'_0 from T_0M to p: p_new = ptransp(origin, p, p'_0)
        # This is too complex for a generic Manifold.expmap.
        # The RiSGD uses: new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
        # where expmap_tangent_vector IS ASSUMED TO BE IN T_{p_projected}M.
        # For Poincare, this is Mobius addition: p (+)_c dp_at_p
        # For now, keeping the simple Euclidean addition as a placeholder if c=0, otherwise error.
        if self.c > 0:
            # Proper Mobius addition should be implemented here if this expmap is to be general.
            # For now, let's assume the RiSGD will use expmap0 after mapping to origin or handle it specifically.
            # This generic expmap might not be directly used by the current RiSGD structure which uses expmap0.
            # If it *is* used, it needs a proper implementation for Poincare ball.
            # Simplified placeholder:
            # log_p_at_origin = self.logmap0(p)
            # updated_tangent_at_origin = log_p_at_origin + dp # This assumes dp is also at origin, which is wrong.
            # return self.expmap0(updated_tangent_at_origin)
            raise NotImplementedError("General expmap(p, dp) for PoincareBall needs Mobius addition or proper parallel transport logic.")
        return self.proju(p + dp) # Euclidean case

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PoincareBall(Manifold):
    def __init__(self, c_scalar: float = 1.0):
        super().__init__(c_scalar)
        self.logger = logging.getLogger(f"{logger.name}.PoincareBall") 
        c_scalar = float(max(c_scalar, 0.0))
        if c_scalar <= 0:
            self.c = 0.0; self.k = 0.; self.sqrt_c = 0.; self.radius = float('inf')
        else:
            self.c = c_scalar; self.k = -self.c
            # Ensure c+EPS is positive before sqrt
            sqrt_c_arg = self.c + EPS
            if sqrt_c_arg < 0: # Should not happen if c_scalar >=0
                 self.logger.error(f"Negative sqrt_c_arg {sqrt_c_arg} with c={self.c}. Setting sqrt_c to 0.")
                 self.sqrt_c = 0.0
            else: self.sqrt_c = math.sqrt(sqrt_c_arg)
            
            if self.sqrt_c < EPS: # Effectively Euclidean if sqrt_c is too small
                self.radius = float('inf')
                # self.c = 0.0 # Can also reset c to 0 if sqrt_c is negligible
            else:
                self.radius = 1. / self.sqrt_c
        
        # max_norm should be strictly less than radius for numerical stability with atanh etc.
        self.max_norm = self.radius * (1. - EPS * 100) if self.c > 0 and self.radius != float('inf') else float('inf') # Increased gap
        self._name = f'PoincareBall(c={self.c:.3g})'

    @property
    def name(self) -> str: return self._name

    def proju(self, x: torch.Tensor) -> torch.Tensor:
        # Use a slightly larger epsilon for projection to ensure points are strictly inside for subsequent logmap
        return HyperbolicUtils.poincare_clip(x, self.c, radius=1., eps=EPS * 10) 

    def expmap0(self, dp: torch.Tensor) -> torch.Tensor:
        return HyperbolicUtils.exponential_map(dp, self.c, eps=EPS)

    def logmap0(self, p: torch.Tensor) -> torch.Tensor:
        return HyperbolicUtils.logarithmic_map(p, self.c, eps=EPS)

    def expmap0_scaled(self, dp: torch.Tensor, scale_scalar: float) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_exponential_map(dp, self.c, scale_scalar=scale_scalar, eps=EPS)

    def logmap0_scaled(self, p: torch.Tensor, scale_scalar: float) -> torch.Tensor:
        return HyperbolicUtils.scale_aware_logarithmic_map(p, self.c, scale_scalar=scale_scalar, eps=EPS)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Poincare ball addition: x (+) y
        if self.c <= 0: return x + y # Euclidean addition

        x_norm_sq = torch.sum(x.pow(2), dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y.pow(2), dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

        # Denominator: 1 + 2c <x,y> + c^2 ||x||^2 ||y||^2
        # Clamp intermediate terms to prevent overflow if x,y norms are large (close to radius)
        # Though inputs should ideally be clipped by proju before mobius_add.
        x_norm_sq_c = torch.clamp(x_norm_sq, 0, self.radius**2 * (1-EPS)**2 if self.radius != float('inf') else MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)
        y_norm_sq_c = torch.clamp(y_norm_sq, 0, self.radius**2 * (1-EPS)**2 if self.radius != float('inf') else MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)

        denominator = 1 + 2 * self.c * xy_dot + self.c**2 * x_norm_sq_c * y_norm_sq_c
        denominator.clamp_min_(EPS) # Avoid division by zero

        # Numerator: (1 + 2c <x,y> + c ||y||^2)x + (1 - c ||x||^2)y
        # The formula from HNN paper (Ganea et al.) is slightly different for numerator factor of x
        # Using formula: ( (1 + 2c<x,y> + c||y||^2)x + (1-c||x||^2)y ) / den
        # Or from Nickel & Kiela: ((1+c*dot(x,y))x + (1-c*norm(x)^2)y)/(1+c*dot(x,y) - c^2*norm(x)^2*norm(y)^2/ (1+c*dot(x,y)))
        # Let's use the more standard one: x +_c y = ((1+2c<x,y>+c||y||^2_c)x + (1-c||x||^2_c)y) / (1+2c<x,y>+c^2||x||^2_c||y||^2_c)
        
        num_factor_x = (1 + 2 * self.c * xy_dot + self.c * y_norm_sq_c)
        num_factor_y = (1 - self.c * x_norm_sq_c)
        
        numerator = num_factor_x * x + num_factor_y * y
        result = numerator / denominator
        
        return self.proju(result) # Project result to ensure it's on the manifold

    def expmap(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at point p (on manifold) with tangent vector dp (at T_pM).
        This requires parallel transport of dp to origin, expmap0, then transport back.
        A common approximation or alternative is using Mobius addition formulation if dp is small.
        For optimizer step: p_new = p_old (+) (-lr * rgrad_at_p_old_transported_to_p_old)
        If dp is already a tangent vector *at p*, properly scaled (e.g. -lr * grad_riemannian),
        the operation is: p_new = exp_p(dp)
        Which is equivalent to ptransp_0_to_p ( expmap_0 ( ptransp_p_to_0 (dp) ) )
        A more direct formula for exp_p(v) is (from some sources, needs verification for Poincare):
        exp_p(v) = p +_c tanh(sqrt(c) * lambda_p * ||v||_p / (2*||v||_p) ) * v / lambda_p
        where lambda_p = 2 / (1 - c ||p||^2)
        
        For now, using the RiSGD common pattern:
        1. logmap p to origin -> get p_tan0
        2. Assume dp is a Riemmanian gradient *at p*, which needs to be transported to origin.
           Or, if dp is already the -lr*rgrad vector in T_pM, it needs to be used in exp_p(dp).
        The current RiSGD implementation computes `expmap_tangent_vector = buf.mul(-lr)` where `buf` is
        the momentum buffer containing Riemannian gradients (or their accumulation).
        This `expmap_tangent_vector` is meant to be in T_pM.
        So we need exp_p(expmap_tangent_vector).
        
        A standard way to implement exp_p(v_p) for Poincare ball:
        v_0 = self.ptransp_p_to_0(p, v_p)  // Parallel transport v_p from T_pM to T_0M
        exp_v_0 = self.expmap0(v_0)      // Exp map from origin
        result = self.mobius_add(p, exp_v_0) // This is not quite right. Mobius add is p (+)_c v.
                                            // If exp_v_0 is on the manifold, it's not a tangent vector anymore.

        Let's use a known formula for exp_p(v) in Poincare Ball (from Ganea et al. supplementary, or similar):
        If v is in T_p M, then exp_p(v) = p (+) (tanh(sqrt(c)/2 * lambda_p * ||v||) * v / (sqrt(c) * ||v||) )
        where lambda_p = 2 / (1 - c ||p||^2). ||v|| is Euclidean norm of v.
        This can be simplified if ||v||_hyp = lambda_p/2 * ||v||_euc
        Then, exp_p(v) = p (+)_c (tanh(sqrt(c) * ||v||_hyp) * v / (sqrt(c) * ||v||_hyp * lambda_p/2))
                     = p (+)_c (tanh(sqrt(c) * ||v||_hyp) / (sqrt(c) * ||v||_hyp) * (v * 2/lambda_p)) ??? -> this is getting messy

        A simpler, more common formulation for optimizer updates (assuming dp is the scaled Riemannian gradient in T_pM):
        new_p = p (+)_c GyroVector( tanh( sqrt(c) * ||dp_g||_p / 2 ) / (sqrt(c) * ||dp_g||_p ) * dp_g_euclidean )
        where dp_g is the Riemannian gradient. dp_g_euclidean is its Euclidean representation.
        ||dp_g||_p is its hyperbolic norm.

        The formulation from "Riemannian Adaptive Optimization Methods" (Equations 10, 14 for Poincare):
        Retraction R_x(v) = proju ( x (+)_c ( ( (1-c||x||^2)^2 / 4 ) * v ) )
        This is simpler and often used. Here, v is the scaled Riemannian gradient.
        This seems to be using scaled Euclidean gradient as input `v` to the Mobius addition
        with a specific scaling factor related to lambda_x^2.

        Let's stick to the expmap0 for updates after transporting gradient, as implied by RiSGD paper logic:
        p_new = expmap_p( -lr * grad_R(p) )
              = p (+) (-lr * grad_R(p))  -- If using Mobius addition as retraction.
        The optimizer uses `manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)`
        If `expmap_tangent_vector` is considered a vector in T_pM, then Mobius addition is a valid retraction.
        
        So, `expmap(p, dp)` will be implemented as `mobius_add(p, dp)`.
        Ensure dp is appropriately scaled if it's a Riem. gradient.
        The `expmap_tangent_vector` in RiSGD is already `-lr * rgrad`.
        """
        if self.c <= 0: # Euclidean case
            return self.proju(p + dp)
        
        # Assuming dp is a tangent vector at p, scaled appropriately (e.g., -lr * Riemannian gradient at p)
        # Perform Mobius addition: p (+)_c dp
        # We must ensure dp is small enough or that Mobius addition is well-behaved.
        # The original optimizer step was: new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
        # This implies expmap_tangent_vector is the vector to "add" to p_projected_on_manifold.
        
        # A common retraction R_p(v) for optimization is p (+)_c v
        # However, the 'v' in R_p(v) = p (+)_c v is a tangent vector at p that has been scaled.
        # The mobius_add method takes two points on the manifold.
        # This requires careful interpretation.
        # If dp is a tangent vector in T_pM:
        #   1. Map p to origin: p_0 = logmap0(p) -> this gives tangent at origin
        #   2. Transport dp from T_pM to T_0M -> dp_0
        #   3. Update in T_0M: updated_dp_0 = dp_0 (this is where -lr * grad would be)
        #   4. Map back to manifold: expmap0(updated_dp_0) -> new point relative to origin
        #   5. Transport this new point from origin frame to p frame.
        # This is complex.

        # Let's assume the RiSGD's `expmap_tangent_vector` IS the `dp` here and is already in T_pM.
        # A common retraction is R_x(v) = x (+)_c v_scaled_for_retraction
        # For Poincare, from Nickel & Kiela "Learning Continuous Hierarchies...", eq (4) for exp_x(v) seems to be x (+)_c v.
        # But that 'v' might be an already transformed tangent vector.

        # If `dp` is the "update vector" in the tangent space *at p*, then
        # `exp_p(dp)` maps this tangent vector to a point on the manifold.
        # The `mobius_add(x,y)` takes two points `x, y` on the manifold.
        # The usage in optimizer: `p.data = manifold.expmap(p_projected, tangent_update_vector_at_p)`
        # This means `expmap` here must be `exp_p(tangent_update_vector_at_p)`.
        
        # Using the formula for exp_p(v) from Ganea et al. "Hyperbolic Neural Networks" App D.1:
        # exp_p(v) = mobius_add(p, tanh(sqrt(c) * lambda_p_v_norm / 2) * (v / (sqrt(c) * lambda_p_v_norm)))
        # where lambda_p_v_norm = (2 / (1 - c ||p||^2)) * ||v||_euc.
        # This is complex.
        
        # Let's simplify: assume `dp` has been appropriately scaled and represents the "target point" in some sense
        # relative to `p`, such that Mobius addition is the correct combination.
        # This is often the case if `dp` is small (like `-lr * grad`).
        # The critical part is that `dp` for `mobius_add(p, dp)` should represent a point on the manifold,
        # not a tangent vector.
        
        # If the optimizer intends `manifold.expmap(point, tangent_at_point)`:
        # Then this method should implement the actual exponential map exp_point(tangent_at_point).
        # For now, let's assume the optimizer structure is using expmap0 and handles transport implicitly
        # or that `expmap` here is a retraction like `p + dp` (Euclidean) or `p (+)_c dp_tangent_scaled_and_expmapped_at_origin`.
        
        # Sticking to the structure from RiemannianSGD paper by Bonnabel (Algorithm 1)
        # x_{k+1} = R_{x_k} (-alpha_k * grad f(x_k))
        # where R is a retraction. For Poincare, a common retraction is Mobius addition after scaling v.
        # R_x(v) = x (+)_c ( ( (1-c||x||^2)^2 / 4 ) * v ) -- from Becigneul & Ganea "Riemannian AdaGrad"
        # Here `v` is the Euclidean representation of the Riemannian gradient.
        # The `expmap_tangent_vector` from RiSGD is already `momentum_buffer * -lr`.
        # This `momentum_buffer` contains accumulated Riemannian gradients.
        # So `expmap_tangent_vector` IS the `-alpha_k * grad f(x_k)` term (in Euclidean coords).

        # Let p_manifold = p (already projected)
        # Let v_euclidean_update = dp (this is `expmap_tangent_vector` from optimizer)

        p_norm_sq = torch.sum(p.pow(2), dim=-1, keepdim=True)
        # Clamp norm_sq to be < 1/c for stability of lambda_p
        p_norm_sq_c = torch.clamp(p_norm_sq, 0, self.radius**2 * (1-EPS*10)**2 if self.radius != float('inf') else MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL)

        # This is lambda_p in some notations, or part of the scaling factor for retraction
        # lambda_p_sq_factor = (1 - self.c * p_norm_sq_c)**2 / 4.0  -- This is (lambda_p / 2)^-2 * (1/c) or similar.
        # No, this factor is from RAdam paper for their specific retraction.
        # The actual term for the retraction R_x(v) = x (+)_c ((lambda_x)^-1 v) for some lambda_x.
        # Ganea uses lambda_x = 2/(1-c||x||^2). So v_transformed = ( (1-c||x||^2)/2 ) * v
        
        # For exp_p(v_tangent_at_p):
        # v_tangent_at_p is dp.
        # lambda_p_factor = 2.0 / (1.0 - self.c * p_norm_sq_c + EPS) # Ensure denominator > 0
        # v_norm_euclidean = dp.norm(dim=-1, keepdim=True) + EPS
        # arg_tanh = torch.clamp(self.sqrt_c * lambda_p_factor * v_norm_euclidean / 2.0, max=15.0) # Clamp tanh arg
        # mobius_vec_factor = torch.tanh(arg_tanh) / (self.sqrt_c * v_norm_euclidean + EPS) # Avoid div by zero
        # mobius_vec_on_manifold = mobius_vec_factor * dp
        
        # The above `mobius_vec_on_manifold` is actually exp_0(ptranps_p_to_0(dp)). This is a point on the manifold.
        # Then the actual exp_p(dp) = ptransp_0_to_p(p, mobius_vec_on_manifold)
        # This involves Mobius addition of p with mobius_vec_on_manifold *if* mobius_vec_on_manifold is correctly transported.
        # The most robust for optimizer is:
        #   1. `rgrad = manifold.egrad2rgrad(p, egrad)` (gives rgrad in T_p M, Euclidean coords)
        #   2. `update_vec = -lr * rgrad`
        #   3. `p_new = R_p(update_vec)` where R is a retraction.
        #      A common R_p(v) is `mobius_add(p, expmap0(ptransp_p_to_0(v)))`.
        #      Or, simplified R_p(v) = `mobius_add(p, v_scaled)` where v_scaled is simpler.
        
        # Given the RiSGD structure, it is most likely that `dp` IS the vector in T_pM
        # and `expmap(p, dp)` should perform the exponential map exp_p(dp).
        # For now, let's use a simple retraction R_p(dp) = mobius_add(p, self.proju(dp * (1-self.c *p_norm_sq_c)/2 ))
        # This is an approximation. A proper exp_p(v) for Poincare is more complex.
        # Reverting to a simple structure: use expmap0 after appropriate transformations if possible.
        # The optimizer will likely use expmap0 after transforming gradients.
        # This expmap(p,dp) should ideally not be called if optimizer uses logmap0/expmap0 strategy.
        # If it *is* called, it must be correct.
        # Fallback: if dp is small, p + dp is a first-order approx.
        # For hyperbolic, mobius_add(p, dp_point_on_manifold_near_origin) is used.
        # If dp is tangent vector, must map to point first.
        
        # Using retraction from " RiemannianProximalMethods Alg 2": R_x(u) = mobius_add(x, u).
        # Here u needs to be point on manifold. So dp (tangent) must be mapped to manifold first via expmap0.
        # This means exp_p(dp) = mobius_add(p, expmap0(dp_transported_to_origin)) if dp is already transported.
        # If dp is in T_pM, then need ptransp_p_to_0(dp).

        self.logger.warning("expmap(p,dp) called on PoincareBall. This assumes dp is a point on the manifold to be added via Mobius addition, or is a T_0M vector. If dp is T_pM, this is incorrect.")
        # Assuming dp is a small vector that has been mapped by expmap0 (i.e., it's a point near origin)
        # and we want to "add" it to p using Mobius addition as a retraction.
        return self.mobius_add(p, self.proju(dp)) # proju(dp) to ensure dp is also in ball if it's a point

    def egrad2rgrad(self, p: torch.Tensor, dp: torch.Tensor) -> torch.Tensor:
        # dp is the Euclidean gradient
        if self.c <= 0: return dp # Euclidean case
        
        # Project p onto the manifold first for safety, though it should already be.
        p_proj = self.proju(p)
        p_norm_sq = torch.sum(p_proj.pow(2), dim=-1, keepdim=True)
        
        # Clamp norm_sq to be strictly less than 1/c for stability of lambda_p
        # self.max_norm ensures this if p_proj is used.
        # max_r_sq = self.radius**2 * (1 - EPS*10)**2 if self.radius != float('inf') else MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL
        # Using self.max_norm which is (1/sqrt_c) * (1-eps_factor)
        # So max_norm_sq is (1/c) * (1-eps_factor)^2
        max_norm_sq_val = self.max_norm**2 if self.max_norm != float('inf') else MAX_HYPERBOLIC_SQ_NORM_CLAMP_VAL
        p_norm_sq_clamped = torch.clamp(p_norm_sq, 0, max_norm_sq_val)

        lambda_p = 2.0 / (1.0 - self.c * p_norm_sq_clamped + EPS) # Add EPS to denominator
        lambda_p_sq_inv_c = (lambda_p / 2.0).pow(2) # This is ( (1-c||p||^2)/2 )^2 / c -- no, this is (1/sqrt(c) * (1-c||p||^2)/2 )^2
                                                 # The factor is ( (1-c||p||^2)/2 )^2
        
        scaling_factor = ( (1.0 - self.c * p_norm_sq_clamped) / 2.0 ).pow(2)
        # Ensure scaling_factor is positive and not too small
        scaling_factor_clamped = torch.clamp(scaling_factor, min=EPS*EPS) # Min value related to EPS squared

        r_grad = scaling_factor_clamped * dp
        
        if not torch.isfinite(r_grad).all():
            dp_norm_str = dp.norm().item() if torch.isfinite(dp).all() else 'NaN'
            p_norm_sq_str = p_norm_sq.mean().item() if p_norm_sq.numel()>0 and torch.isfinite(p_norm_sq).all() else 'NaN'
            p_proj_norm_str = p_proj.norm().item() if torch.isfinite(p_proj).all() else 'NaN'
            factor_str = scaling_factor_clamped.mean().item() if scaling_factor_clamped.numel()>0 and torch.isfinite(scaling_factor_clamped).all() else 'NaN_or_Empty'
            self.logger.warning(f"Non-finite Riemannian gradient in egrad2rgrad for P:{p.shape}, c={self.c}. Factor: {factor_str}, dp_norm: {dp_norm_str}. p_norm_sq: {p_norm_sq_str}. p_proj_norm: {p_proj_norm_str}")
            # Fallback to Euclidean gradient, possibly scaled down
            return dp * EPS # Heavily dampened Euclidean gradient as fallback
        return r_grad

    def init_weights(self, w: nn.Parameter, irange: float = 1e-5):
        with torch.no_grad():
            w.data.uniform_(-irange, irange) # Initialize in tangent space at origin
            if self.c > 0 :
                # Map from tangent space at origin to the manifold
                w.data = self.expmap0(w.data) 
                # Ensure projection after mapping, though expmap0 should already be clipped by poincare_clip
                w.data = self.proju(w.data)

def init_weights_general(m): # Unchanged
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, 'elementwise_affine', getattr(m, 'affine', False)): # Compatibility
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)

def get_constrained_param_val(param_unconstrained: nn.Parameter, min_val: float = EPS) -> torch.Tensor: # Unchanged
    return F.softplus(param_unconstrained) + min_val

# <TOTAL_STRATEGY_INTEGRATION> BoundaryManifoldHyperbolic

class BoundaryManifoldHyperbolic(nn.Module):
    def __init__(self, level_idx: int, num_points: int, point_dim: int, initial_manifold_c_base: float,
                 level_config: WuBuLevelConfig, 
                 g_w_level_config_features: Dict[str, Any], # Contains 'vectorized_features_dim' and potentially pre-computed 'vectorized_features' tensor
                 global_context_embedding_dim: int 
                 ):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim
        self.initial_manifold_c_base = initial_manifold_c_base 
        self.current_manifold_c = initial_manifold_c_base 
        self.level_config = level_config 
        self.g_w_level_config_features = g_w_level_config_features 
        self.global_context_embedding_dim = global_context_embedding_dim
        self.logger = logging.getLogger(f"{logger.name}.BoundaryManifold.L{level_idx}")

        self.point_generator_mlps = nn.ModuleList()
        self.base_boundary_point_params_unconstrained = nn.ParameterList()
        # Ensure static_hyperbolic_points_params is registered as None if not used.
        self.register_parameter('static_hyperbolic_points_params', None)


        if num_points > 0 and point_dim > 0:
            mod_config = self.level_config.boundary_points_modulator
            if mod_config.enabled:
                self.logger.info(f"Boundary points (N={num_points}, Dim={point_dim}) for L{level_idx} WILL BE DYNAMICALLY MODULATED.")
                
                # Determine MLP input dimension for modulators
                g_w_vectorized_dim_from_config = self.g_w_level_config_features.get("vectorized_features_dim", 0)
                if g_w_vectorized_dim_from_config <= 0:
                    self.logger.warning(f"L{level_idx} BoundaryManifold: g_w_vectorized_features_dim is {g_w_vectorized_dim_from_config}. Dynamic modulation might be ineffective if context is not passed or dim is zero.")
                
                # Each point's modulator MLP might also take a unique point embedding if desired,
                # but for now, use shared g_w_vec + global_ctx_emb for all points in this manifold.
                mlp_input_dim = g_w_vectorized_dim_from_config + self.global_context_embedding_dim
                if mlp_input_dim <= 0:
                    self.logger.warning(f"L{level_idx} BoundaryManifold: Total MLP input dim for dynamic points is {mlp_input_dim}. Modulation might be disabled or ineffective.")
                    # Fallback: disable modulation if no input context for MLP
                    mod_config.enabled = False # Override if no input context

                for i in range(num_points):
                    base_param = nn.Parameter(torch.Tensor(point_dim))
                    # Initialize in tangent space at origin, then map to manifold if c>0
                    # For simplicity, init_weights does expmap0 if c>0
                    PoincareBall(self.initial_manifold_c_base).init_weights(base_param, irange=self.level_config.level_descriptor_init_scale) # Using ld_init_scale for boundary point init too
                    self.base_boundary_point_params_unconstrained.append(base_param)

                    if mod_config.enabled and mlp_input_dim > 0:
                        hidden_size = max(8, int(point_dim * mod_config.mlp_hidden_dim_ratio))
                        if hidden_size == 0 and point_dim > 0: hidden_size = point_dim # Ensure hidden_size > 0 if point_dim > 0
                        
                        layers = []
                        if mlp_input_dim > 0 and hidden_size > 0:
                            layers.append(nn.Linear(mlp_input_dim, hidden_size))
                            act_fn_choice = SwiGLUActivation() if mod_config.mlp_activation == "SiLU" else (nn.GELU() if mod_config.mlp_activation == "GELU" else nn.ReLU())
                            
                            for _ in range(max(0, mod_config.mlp_num_layers - 1)): # Ensure num_layers >= 1
                                layers.extend([act_fn_choice, nn.Linear(hidden_size, hidden_size)])
                            layers.extend([act_fn_choice, nn.Linear(hidden_size, point_dim)]) # Output is delta for tangent space
                            self.point_generator_mlps.append(nn.Sequential(*layers))
                        else: # Not enough dims for a proper MLP
                             self.point_generator_mlps.append(nn.Identity())
                    else: 
                        self.point_generator_mlps.append(nn.Identity()) # No modulation if disabled or no input dim
            
            # This 'else' block handles the case where mod_config.enabled was false from the start.
            if not mod_config.enabled: # If modulation is NOT enabled (either initially or due to fallback)
                self.logger.info(f"Boundary points (N={num_points}, Dim={point_dim}) for L{level_idx} are STATIC nn.Parameters.")
                self.static_hyperbolic_points_params = nn.Parameter(torch.Tensor(num_points, point_dim))
                PoincareBall(self.initial_manifold_c_base).init_weights(self.static_hyperbolic_points_params, irange=self.level_config.level_descriptor_init_scale)
                setattr(self.static_hyperbolic_points_params, 'manifold', PoincareBall(self.initial_manifold_c_base))
                # Ensure point_generator_mlps and base_boundary_point_params_unconstrained are empty if static
                self.point_generator_mlps = nn.ModuleList()
                self.base_boundary_point_params_unconstrained = nn.ParameterList()

        # Ensure attributes exist even if num_points or point_dim is 0
        if not hasattr(self, 'point_generator_mlps'): self.point_generator_mlps = nn.ModuleList()
        if not hasattr(self, 'base_boundary_point_params_unconstrained'): self.base_boundary_point_params_unconstrained = nn.ParameterList()
        if not hasattr(self, 'static_hyperbolic_points_params'): self.register_parameter('static_hyperbolic_points_params', None)


    def set_current_manifold_c(self, c_scalar: float):
        self.current_manifold_c = float(max(c_scalar, 0.0)) # Ensure non-negative
        if hasattr(self, 'static_hyperbolic_points_params') and self.static_hyperbolic_points_params is not None:
            # The manifold attribute of the parameter itself is mainly for RiemannianSGD.
            # The actual projection/expmap uses self.current_manifold_c.
            # For consistency, update it, but it's less critical than self.current_manifold_c.
            if getattr(self.static_hyperbolic_points_params, 'manifold', None) is not None : # Check if it has manifold attr
                 current_param_manifold = getattr(self.static_hyperbolic_points_params, 'manifold')
                 if isinstance(current_param_manifold, PoincareBall): current_param_manifold.c = self.current_manifold_c # Update existing one
                 else: setattr(self.static_hyperbolic_points_params, 'manifold', PoincareBall(self.current_manifold_c)) # Set new one
            else: # If no manifold attribute was set
                 setattr(self.static_hyperbolic_points_params, 'manifold', PoincareBall(self.current_manifold_c))


    def get_points(self, g_w_features_vectorized: Optional[torch.Tensor] = None, 
                   global_context_embedding: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.num_points == 0 or self.point_dim == 0:
            return None

        mod_cfg = self.level_config.boundary_points_modulator
        manifold_to_use = PoincareBall(self.current_manifold_c)

        if mod_cfg.enabled and self.point_generator_mlps and len(self.point_generator_mlps) == self.num_points:
            # Check if context is available for modulation
            g_w_vec_dim_expected = self.g_w_level_config_features.get("vectorized_features_dim", 0)
            
            context_available_for_mlp = (
                g_w_features_vectorized is not None and
                global_context_embedding is not None and
                g_w_features_vectorized.shape[-1] == g_w_vec_dim_expected and
                global_context_embedding.shape[-1] == self.global_context_embedding_dim and
                (g_w_vec_dim_expected + self.global_context_embedding_dim) > 0 # Ensure MLP has inputs
            )

            if not context_available_for_mlp:
                self.logger.debug(f"L{self.level_idx} BoundaryManifold: Dynamic points enabled but context missing or malformed. Returning base points. "
                                  f"g_w_vec shape: {g_w_features_vectorized.shape if g_w_features_vectorized is not None else 'None'} (expected dim {g_w_vec_dim_expected}), "
                                  f"global_ctx_emb shape: {global_context_embedding.shape if global_context_embedding is not None else 'None'} (expected dim {self.global_context_embedding_dim})")
                # Fallback: generate points from base parameters only, no modulation
                generated_points_fallback = []
                for i in range(self.num_points):
                    if i < len(self.base_boundary_point_params_unconstrained):
                        base_tangent = self.base_boundary_point_params_unconstrained[i]
                        # base_tangent is already in tangent space. Map to manifold.
                        point_on_manifold = manifold_to_use.expmap0(base_tangent)
                        generated_points_fallback.append(point_on_manifold)
                    else: # Should not happen if initialized correctly
                        self.logger.error(f"L{self.level_idx} BoundaryManifold: Mismatch in num_points and base_params len during fallback.")
                        return None
                return torch.stack(generated_points_fallback) if generated_points_fallback else None

            # Prepare MLP input: [g_w_features_vectorized, global_context_embedding]
            # Ensure context tensors are 1D (or 2D with batch_size=1) before cat for MLP
            g_w_vec_mlp = g_w_features_vectorized.view(1, -1) if g_w_features_vectorized.dim() > 1 else g_w_features_vectorized.unsqueeze(0)
            global_ctx_mlp = global_context_embedding.view(1, -1) if global_context_embedding.dim() > 1 else global_context_embedding.unsqueeze(0)
            
            # Move to device of MLP parameters (usually same as base_params)
            dev_mlp = next(self.point_generator_mlps[0].parameters()).device
            mlp_input = torch.cat([g_w_vec_mlp.to(dev_mlp), global_ctx_mlp.to(dev_mlp)], dim=-1)
            
            generated_points_dynamic = []
            for i in range(self.num_points):
                base_tangent_param = self.base_boundary_point_params_unconstrained[i]
                modulator_mlp = self.point_generator_mlps[i]
                
                tangent_delta = torch.zeros_like(base_tangent_param) # Default to no change
                if not isinstance(modulator_mlp, nn.Identity): # If there's an actual MLP
                    tangent_delta = modulator_mlp(mlp_input).squeeze(0) * mod_cfg.global_context_influence_factor
                
                effective_tangent = base_tangent_param + tangent_delta
                # Clamp effective tangent vector before expmap
                effective_tangent_clamped = torch.clamp(effective_tangent, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
                
                point_on_manifold = manifold_to_use.expmap0(effective_tangent_clamped)
                generated_points_dynamic.append(point_on_manifold)
            
            return torch.stack(generated_points_dynamic) if generated_points_dynamic else None
            
        elif hasattr(self, 'static_hyperbolic_points_params') and self.static_hyperbolic_points_params is not None:
            # Static points are already on the manifold (or should be after init_weights)
            # proju ensures they stay there if c changes or if they were slightly off.
            return manifold_to_use.proju(self.static_hyperbolic_points_params)
        
        # This case should ideally not be reached if num_points > 0 and point_dim > 0
        # It means neither dynamic nor static points were properly configured/initialized.
        self.logger.error(f"L{self.level_idx} BoundaryManifold: No method to get points. mod_enabled={mod_cfg.enabled}, "
                          f"num_mlps={len(self.point_generator_mlps)}, static_params_exist={hasattr(self, 'static_hyperbolic_points_params') and self.static_hyperbolic_points_params is not None}. "
                          f"Num_points={self.num_points}, Point_dim={self.point_dim}")
        return None

    def get_interpretability_data(self, g_w_features_vectorized: Optional[torch.Tensor] = None, 
                               global_context_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "level_idx": self.level_idx,
            "num_points": self.num_points,
            "point_dim": self.point_dim,
            "current_manifold_c": self.current_manifold_c,
            "is_dynamic": self.level_config.boundary_points_modulator.enabled
        }
        points_h = self.get_points(g_w_features_vectorized, global_context_embedding)
        if points_h is not None:
            data["points_hyperbolic_mean_norm"] = points_h.norm(dim=-1).mean().item()
            data["points_hyperbolic_max_norm"] = points_h.norm(dim=-1).max().item()
            if self.num_points > 0 and self.point_dim > 0: # Check to avoid error on empty tensor
                data["points_hyperbolic_std_coords_mean"] = points_h.view(self.num_points, self.point_dim).std(dim=0).mean().item()

        # Log norms of modulator MLP weights if they exist
        if self.level_config.boundary_points_modulator.enabled and self.point_generator_mlps:
            mod_mlp_weight_norms = []
            for i, mlp in enumerate(self.point_generator_mlps):
                if not isinstance(mlp, nn.Identity):
                    # Example: norm of the first linear layer's weights
                    first_linear_layer = next(filter(lambda m: isinstance(m, nn.Linear), mlp.modules()), None)
                    if first_linear_layer:
                        mod_mlp_weight_norms.append(first_linear_layer.weight.norm().item())
            if mod_mlp_weight_norms:
                data["modulator_mlp_first_layer_weight_norm_mean"] = np.mean(mod_mlp_weight_norms)
        
        return data

# Quaternion functions (largely unchanged, but ensure device consistency and normalization)
def quaternion_from_axis_angle(angle_rad: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    # axis: (B, 3) or (3), angle_rad: (B, 1) or (1)
    # Ensure axis is normalized
    axis_norm = F.normalize(axis, p=2, dim=-1, eps=EPS) # Add EPS for stability
    angle_half = angle_rad / 2.0
    q_w = torch.cos(angle_half)
    q_xyz = axis_norm * torch.sin(angle_half)
    # Ensure proper broadcasting if axis was (3) and angle_rad was (B,1)
    if q_w.dim() == 2 and q_xyz.dim() == 1 and q_w.shape[0] == q_xyz.shape[0]: # Should not happen with correct inputs
        q_xyz = q_xyz.unsqueeze(0).expand_as(q_w.expand(-1,3)) # Error in logic before
    elif q_w.dim() > q_xyz.dim() and q_xyz.shape[-1] == 3: # (B,1) and (3) case
         q_xyz = q_xyz.unsqueeze(0).expand(q_w.shape[0], -1)
    elif q_xyz.dim() > q_w.dim() and q_w.shape[-1] == 1: # (B,3) and (1) case
         q_w = q_w.unsqueeze(0).expand(q_xyz.shape[0], -1)


    return torch.cat([q_w, q_xyz], dim=-1)

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # q1, q2: (..., 4)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_apply_to_vector(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: (..., 4), v: (..., 3)
    # Pad v to be a "pure quaternion" (0, x, y, z)
    v_quat_shape = list(v.shape)
    v_quat_shape[-1] +=1
    v_quat = torch.zeros(v_quat_shape, device=v.device, dtype=v.dtype)
    v_quat[..., 1:] = v
    # q_conj: (w, -x, -y, -z)
    q_conj_parts = [-q[..., 1], -q[..., 2], -q[..., 3]]
    q_conj = torch.cat([q[..., :1], torch.stack(q_conj_parts, dim=-1)], dim=-1)
    
    rotated_v_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    return rotated_v_quat[..., 1:] # Return the vector part

# <TOTAL_STRATEGY_INTEGRATION> HyperbolicInterLevelTransform
class HyperbolicInterLevelTransform(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, initial_c_in_base: float, initial_c_out_base: float,
                 transform_config: WuBuTransformConfig, # <TOTAL_STRATEGY_INTEGRATION>
                 level_idx_for_phi: int = 0, dropout_val: float = 0.1): # <TOTAL_STRATEGY_INTEGRATION>
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.transform_config = transform_config
        self.level_idx_for_phi = level_idx_for_phi # For PHI-influenced rotation init
        self.dropout_val = dropout_val
        self.logger = logging.getLogger(f"{logger.name}.HILT.L{level_idx_for_phi}")

        self.current_c_in = initial_c_in_base # Will be updated by parent WuBu model if C is dynamic
        self.current_c_out = initial_c_out_base

        # Anisotropic Block Processors
        self.block_processors = nn.ModuleList()
        self.num_aniso_blocks = max(1, self.transform_config.num_aniso_blocks)
        if self.in_dim > 0 and self.num_aniso_blocks > 0 and self.in_dim % self.num_aniso_blocks == 0:
            self.block_dim_in = self.in_dim // self.num_aniso_blocks
            for _ in range(self.num_aniso_blocks):
                block_hidden_dim = max(1, int(self.block_dim_in * self.transform_config.block_mlp_hidden_dim_ratio))
                # Using SiLU as SwiGLU activation for simplicity as per notes
                self.block_processors.append(nn.Sequential(
                    nn.Linear(self.block_dim_in, block_hidden_dim),
                    SwiGLUActivation(), # nn.SiLU(),
                    nn.Dropout(self.dropout_val),
                    nn.Linear(block_hidden_dim, self.block_dim_in),
                    nn.LayerNorm(self.block_dim_in) # LayerNorm per block output
                ))
            self.processed_block_total_dim = self.in_dim
        else:
            if self.in_dim > 0:
                self.logger.warning(f"L{level_idx_for_phi} HILT: in_dim {self.in_dim} not divisible by num_aniso_blocks {self.num_aniso_blocks}. Using single block (Identity).")
            self.num_aniso_blocks = 1
            self.block_dim_in = self.in_dim
            self.block_processors.append(nn.Identity()) # Fallback
            self.processed_block_total_dim = self.in_dim

        # Rotation Module (Advanced)
        self.rotation_module_parts = nn.ParameterDict() # For learnable rotation parameters
        self.use_rotation = self.transform_config.use_rotation_in_transform
        self.rotation_type = self.transform_config.rotation_type if self.use_rotation else "none"
        self.phi_influence_rotation_init = self.transform_config.phi_influence_rotation_init

        if self.use_rotation and self.in_dim > 0 :
            self.rotation_block_dim = self.transform_config.rotation_block_dim
            if self.rotation_type == "block_quaternion" and self.in_dim % 3 == 0 and self.rotation_block_dim == 3: # Use dim 3 blocks for quaternion on vectors
                num_rot_blocks = self.in_dim // 3
                for i in range(num_rot_blocks):
                    # Angle (unconstrained) + Axis (3D unconstrained)
                    self.rotation_module_parts[f'rot_block_{i}_angle_un'] = nn.Parameter(torch.tensor(0.0))
                    self.rotation_module_parts[f'rot_block_{i}_axis_un'] = nn.Parameter(torch.randn(3))
                    if self.phi_influence_rotation_init:
                        # phi_scale = PHI**((level_idx_for_phi + i) % 5 - 2) * (math.pi / (4 + i*0.5)) # Varying scale
                        phi_scale = PHI**(self.level_idx_for_phi % 4 - 1.5) * (math.pi / 4) # Consistent with old
                        with torch.no_grad(): self.rotation_module_parts[f'rot_block_{i}_angle_un'].data += phi_scale * (torch.rand(1).item() - 0.5) * 0.1
            elif self.rotation_type == "full_svd_orthogonal" and self.in_dim > 0:
                # Standard learnable orthogonal matrix using nn.Linear then QR/SVD parametrization (complex)
                # For now, a simple linear layer and rely on spectral norm or other regularizers if needed outside.
                # Or, use torch.nn.utils.parametrizations.orthogonal (if PyTorch version supports it well)
                # Fallback to a simple Linear for now, actual orthogonalization would be separate.
                self.simple_linear_rotation_fallback = nn.Linear(self.in_dim, self.in_dim, bias=False)
                if self.in_dim > 0: nn.init.eye_(self.simple_linear_rotation_fallback.weight)
                self.logger.warning(f"L{level_idx_for_phi} HILT: 'full_svd_orthogonal' rotation type is complex. Using Linear layer as placeholder. Proper orthogonal parametrization needed for true SVD method.")
            elif self.in_dim == 2 and self.rotation_type != "none": # Special SO(2) case if in_dim is 2
                 self.rotation_module_parts['rot_2d_angle_un'] = nn.Parameter(torch.tensor(0.0))
                 if self.phi_influence_rotation_init:
                      phi_scale_2d = PHI**(self.level_idx_for_phi % 3) * (math.pi / 3)
                      with torch.no_grad(): self.rotation_module_parts['rot_2d_angle_un'].data += phi_scale_2d * (torch.rand(1).item() - 0.5) * 0.1
            else: # No specific rotation or fallback
                self.rotation_type = "none" # Ensure it's none if not handled
                self.logger.info(f"L{level_idx_for_phi} HILT: Rotation disabled or in_dim ({self.in_dim}) not suitable for configured rotation_type '{self.rotation_type}'.")

        # Final Non-Rotational Mapping (processes output of block processors + rotation)
        # The input to this is self.processed_block_total_dim (which is self.in_dim)
        if self.processed_block_total_dim > 0 and self.out_dim > 0:
            # This is equivalent to the old "non_rotational_map" being 'mlp' or 'linear'
            # based on transform_types and transform_hidden_dims in the old system.
            # For now, assume a simple linear projection after anisotropy and rotation.
            # The "transform_type" and "hidden_dim" from old config need to be mapped here.
            # Let's use a simple Linear layer for now, assuming block_processors handle complexity.
            self.final_projection = nn.Linear(self.processed_block_total_dim, self.out_dim)
        else:
            self.final_projection = nn.Identity()
            if self.out_dim > 0:
                 self.logger.info(f"L{level_idx_for_phi} HILT: Using Identity for final_projection as processed_block_total_dim is 0.")


        self.apply(init_weights_general)

    def _apply_rotation(self, x_tan: torch.Tensor) -> torch.Tensor:
        if not self.use_rotation or self.rotation_type == "none" or self.in_dim == 0:
            return x_tan
        
        B_shape = x_tan.shape[:-1] # Keep all leading batch/sequence dimensions

        if self.rotation_type == "block_quaternion" and self.in_dim % 3 == 0 and self.rotation_block_dim == 3:
            num_rot_blocks = self.in_dim // 3
            x_tan_blocks = x_tan.reshape(*B_shape, num_rot_blocks, 3)
            rotated_blocks = []
            for i in range(num_rot_blocks):
                angle_un = self.rotation_module_parts[f'rot_block_{i}_angle_un']
                axis_un = self.rotation_module_parts[f'rot_block_{i}_axis_un']
                
                angle = F.softplus(angle_un) # Ensure positive angle, can be adapted
                if self.phi_influence_rotation_init: # If PHI used, scale might already be set.
                     phi_scale_factor_runtime = PHI**(self.level_idx_for_phi % 4 - 1.5) * (math.pi / 4)
                     angle = angle * phi_scale_factor_runtime # Apply at runtime if phi influence desired beyond init

                current_axis = axis_un.unsqueeze(0).expand(x_tan_blocks.shape[0] if x_tan_blocks.dim() > 2 else 1, -1) # B, 3 or 1, 3
                angle_b = angle.unsqueeze(0) # 1,1 or B,1. Ensure it matches batch for quaternion_from_axis_angle
                if angle_b.dim() == 1: angle_b = angle_b.unsqueeze(0) # Ensure 2D for broadcasting with B if B exists
                if x_tan_blocks.dim() > 2 and angle_b.shape[0] == 1 : angle_b = angle_b.expand(x_tan_blocks.shape[0],-1)


                q_rot = quaternion_from_axis_angle(angle_b.to(x_tan.device), current_axis.to(x_tan.device))
                rotated_block = quaternion_apply_to_vector(q_rot, x_tan_blocks[..., i, :])
                rotated_blocks.append(rotated_block)
            return torch.stack(rotated_blocks, dim=-2).reshape_as(x_tan) # (..., num_rot_blocks, 3) -> (..., in_dim)

        elif self.in_dim == 2 and 'rot_2d_angle_un' in self.rotation_module_parts:
            angle_un_2d = self.rotation_module_parts['rot_2d_angle_un']
            angle_2d = F.softplus(angle_un_2d) # Or direct use if range desired
            if self.phi_influence_rotation_init:
                phi_scale_2d_runtime = PHI**(self.level_idx_for_phi % 3) * (math.pi / 3)
                angle_2d = angle_2d * phi_scale_2d_runtime

            cos_a = torch.cos(angle_2d); sin_a = torch.sin(angle_2d)
            x_comp = x_tan[..., 0]; y_comp = x_tan[..., 1]
            x_rot = x_comp * cos_a - y_comp * sin_a
            y_rot = x_comp * sin_a + y_comp * cos_a
            return torch.stack([x_rot, y_rot], dim=-1)

        elif hasattr(self, 'simple_linear_rotation_fallback'):
            return self.simple_linear_rotation_fallback(x_tan)
        
        return x_tan # Fallback if no rotation applied

    def forward(self, point_in: torch.Tensor, boundaries_in: Optional[torch.Tensor],
                descriptor_in: Optional[torch.Tensor], current_c_in_val: float, current_c_out_val: float,
                # current_s_in, current_s_out are not used in original transform, can be removed or used for scaling
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.current_c_in = current_c_in_val
        self.current_c_out = current_c_out_val
        m_in, m_out = PoincareBall(self.current_c_in), PoincareBall(self.current_c_out)

        def _process_vector(hyper_vec: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if hyper_vec is None: return None
            tan_vec = m_in.logmap0(hyper_vec)

            # 1. Anisotropic Block Processing
            if self.num_aniso_blocks > 1 and self.in_dim > 0 and tan_vec.shape[-1] == self.in_dim:
                tan_vec_blocks = tan_vec.reshape(*tan_vec.shape[:-1], self.num_aniso_blocks, self.block_dim_in)
                processed_blocks = []
                for i in range(self.num_aniso_blocks):
                    processed_blocks.append(self.block_processors[i](tan_vec_blocks[..., i, :]))
                tan_vec_processed_aniso = torch.cat(processed_blocks, dim=-1)
            elif self.block_processors and not isinstance(self.block_processors[0], nn.Identity): # Single block processor
                tan_vec_processed_aniso = self.block_processors[0](tan_vec)
            else: # No block processing or identity
                tan_vec_processed_aniso = tan_vec
            
            # 2. Rotation
            tan_vec_rotated = self._apply_rotation(tan_vec_processed_aniso)
            
            # 3. Final Projection
            tan_vec_projected = self.final_projection(tan_vec_rotated)
            
            return torch.clamp(tan_vec_projected, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)

        tan_main_out_clamped = _process_vector(point_in)
        tan_bound_out_clamped = _process_vector(boundaries_in) if boundaries_in is not None else None
        tan_desc_out_clamped = _process_vector(descriptor_in) if descriptor_in is not None else None

        default_out_shape = (point_in.shape[0], self.out_dim) if point_in.dim() > 1 else (self.out_dim,)
        dtype_out = point_in.dtype
        dev_out = point_in.device

        expmap_main_out = m_out.expmap0(tan_main_out_clamped) if tan_main_out_clamped is not None else \
                          m_out.expmap0(torch.zeros(default_out_shape, device=dev_out, dtype=dtype_out))
        expmap_bound_out = m_out.expmap0(tan_bound_out_clamped) if tan_bound_out_clamped is not None else None
        expmap_desc_out = m_out.expmap0(tan_desc_out_clamped) if tan_desc_out_clamped is not None else None
        
        return expmap_main_out, expmap_bound_out, expmap_desc_out

    def get_interpretability_data(self) -> Dict[str, Any]:
        data = {"c_in": self.current_c_in, "c_out": self.current_c_out, "rotation_type": self.rotation_type}
        if hasattr(self, 'simple_linear_rotation_fallback'):
             data["rotation_fallback_linear_weight_norm"] = self.simple_linear_rotation_fallback.weight.norm().item()
        for name, param in self.rotation_module_parts.items():
            data[f"rotation_param_{name}"] = param.data.detach().cpu().numpy()
        # TODO: Add singular values of block processors / final_projection if computationally feasible
        return data

# <TOTAL_STRATEGY_INTEGRATION> HyperbolicWuBuNestingLevel
# WuBuSpecTrans_v0.2.0_TotalStrategy (Continued)

class HyperbolicWuBuNestingLevel(nn.Module):
    def __init__(self, level_idx: int,
                 level_config: WuBuLevelConfig, 
                 stack_config: WuBuStackConfig, 
                 g_w_level_features: Dict[str, Any], 
                 global_context_embedding_dim: int 
                 ):
        super().__init__()
        self.level_idx = level_idx
        self.level_config = level_config
        self.stack_config = stack_config
        self.g_w_level_features = g_w_level_features 
        self.global_context_embedding_dim = global_context_embedding_dim
        self.logger = logging.getLogger(f"{logger.name}.WuBuLevel.L{level_idx}")
        self.dim = self.level_config.hyperbolic_dim

        # Initial curvature, scale, spread (base values)
        initial_c = self.level_config.initial_curvature
        initial_s = self.level_config.initial_scale
        initial_sigma = self.level_config.initial_spread
        
        # Apply log(g) influence to base initial values
        complexity_score = self.g_w_level_features.get("complexity_score", 0.0)
        # Ensure log_g_eff is calculated safely, even if complexity_score is 0 or very small
        log_g_eff = math.log(max(complexity_score, 1e-8) + EPS) # Use 1e-8 to avoid log(EPS) if score is 0

        initial_c = initial_c * (1 + self.level_config.log_g_complexity_influence_factor_curv * log_g_eff)
        initial_s = initial_s * (1 + self.level_config.log_g_complexity_influence_factor_scale * log_g_eff)
        initial_sigma = initial_sigma * (1 + self.level_config.log_g_complexity_influence_factor_spread * log_g_eff)

        # Apply PHI influence (if stack_config.phi_influence_curvature_stack_global is true)
        if self.stack_config.phi_influence_curvature_stack_global:
            initial_c = initial_c * (PHI**(level_idx % 4 - 1.5)) # Ensure calculation is float
        
        self.logger.info(f"L{level_idx} Dim:{self.dim}, Initial C:{initial_c:.3f} S:{initial_s:.3f} Sigma:{initial_sigma:.3f} (after g_W/PHI mods)")

        # Min values from level_config
        self.min_curvature = max(EPS, self.level_config.curvature_min_value)
        self.min_scale = max(EPS, self.level_config.scale_min_value)
        self.min_spread = max(EPS, self.level_config.spread_min_value)

        # Base unconstrained parameters
        def _init_unconstrained_softplus(target_val: float, min_val: float) -> torch.Tensor:
            # Ensure target_val is float for calculations
            target_val_float = float(target_val)
            val_for_softplus = max(target_val_float, min_val + EPS) - min_val
            # Softplus inverse: log(exp(y) - 1)
            # If val_for_softplus is very small, expm1(val_for_softplus) approx val_for_softplus
            if val_for_softplus < 1e-6: # Threshold for approximation
                return torch.tensor(math.log(val_for_softplus + EPS)) # Add EPS for safety if val_for_softplus is zero
            else:
                return torch.tensor(math.log(math.expm1(val_for_softplus)))
        
        def _init_unconstrained_sigmoid_scaled(target_val: float, min_r: float, max_r: float) -> torch.Tensor:
            target_val_float = float(target_val)
            # Clamp target_val to be strictly within (min_r, max_r) for stability of inverse sigmoid
            clamped_target = torch.clamp(torch.as_tensor(target_val_float, dtype=torch.float32), min_r + EPS, max_r - EPS).item()
            if (max_r - min_r) < EPS: # Avoid division by zero if min_r and max_r are too close
                self.logger.warning(f"L{level_idx} Sigmoid init: min_r ({min_r}) and max_r ({max_r}) are too close. Defaulting unconstrained to 0.")
                return torch.tensor(0.0)
            initial_sig_target = (clamped_target - min_r) / (max_r - min_r)
            # Ensure initial_sig_target is strictly within (0,1) for logit (inverse sigmoid)
            initial_sig_target_clamped = max(EPS, min(initial_sig_target, 1.0 - EPS))
            return torch.logit(torch.tensor(initial_sig_target_clamped, dtype=torch.float32), eps=None) # Use PyTorch's logit

        if self.level_config.learnable_curvature:
            self.base_log_curvature_unconstrained = nn.Parameter(_init_unconstrained_softplus(initial_c, self.min_curvature))
        else: self.register_buffer('base_log_curvature_unconstrained', _init_unconstrained_softplus(initial_c, self.min_curvature))

        if self.level_config.learnable_scales:
            self.base_log_scale_unconstrained = nn.Parameter(_init_unconstrained_sigmoid_scaled(initial_s, MIN_WUBU_LEVEL_SCALE, MAX_WUBU_LEVEL_SCALE))
        else: self.register_buffer('base_log_scale_unconstrained', _init_unconstrained_sigmoid_scaled(initial_s, MIN_WUBU_LEVEL_SCALE, MAX_WUBU_LEVEL_SCALE))
        
        self.base_log_spread_unconstrained: Optional[Union[nn.Parameter, torch.Tensor]]
        if self.level_config.use_level_spread:
            if self.level_config.learnable_spread:
                self.base_log_spread_unconstrained = nn.Parameter(_init_unconstrained_softplus(initial_sigma, self.min_spread))
            else: self.register_buffer('base_log_spread_unconstrained', _init_unconstrained_softplus(initial_sigma, self.min_spread))
        else: self.register_parameter('base_log_spread_unconstrained', None)

        # Modulator MLPs for C, S, Spread
        g_w_vec_dim = self.g_w_level_features.get("vectorized_features_dim", 0)
        mod_mlp_input_dim = g_w_vec_dim + self.global_context_embedding_dim

        def _create_modulator_mlp(param_name: str, mod_cfg: WuBuLevelModulatorConfig, output_dim: int = 1, mlp_input_dim_override: Optional[int] = None) -> nn.Module:
            current_mlp_input_dim = mlp_input_dim_override if mlp_input_dim_override is not None else mod_mlp_input_dim
            if not mod_cfg.enabled or current_mlp_input_dim <= 0: 
                return nn.Identity()
            
            hidden_dim = max(1, int(current_mlp_input_dim * mod_cfg.mlp_hidden_dim_ratio))
            if hidden_dim == 0 and output_dim > 0 : hidden_dim = output_dim # Ensure hidden_dim is at least output_dim if output_dim > 0
            if hidden_dim == 0 and output_dim == 0: return nn.Identity() # Cannot make MLP

            layers = [nn.Linear(current_mlp_input_dim, hidden_dim)]
            act_fn_choice = SwiGLUActivation() if mod_cfg.mlp_activation == "SiLU" else (nn.GELU() if mod_cfg.mlp_activation == "GELU" else nn.ReLU())
            
            # Ensure at least one hidden layer before output if mlp_num_layers > 1
            # If mlp_num_layers is 1, it's Input -> Hidden -> Output.
            # If mlp_num_layers is 2, it's Input -> Hidden -> Activation -> Hidden -> Activation -> Output
            num_hidden_layers = max(0, mod_cfg.mlp_num_layers -1) # Number of intermediate hidden layers
            
            for _ in range(num_hidden_layers):
                layers.extend([act_fn_choice, nn.Linear(hidden_dim, hidden_dim)])
            
            layers.extend([act_fn_choice, nn.Linear(hidden_dim, output_dim)]) # Final layer to output_dim
            self.logger.info(f"L{self.level_idx} {param_name} modulator MLP: In:{current_mlp_input_dim}, Hidden:{hidden_dim} (x{num_hidden_layers + 1} blocks), Out:{output_dim}, Act:{mod_cfg.mlp_activation}")
            return nn.Sequential(*layers)

        self.curvature_modulator_mlp = _create_modulator_mlp("Curvature", self.level_config.curvature_modulator)
        self.scale_modulator_mlp = _create_modulator_mlp("Scale", self.level_config.scale_modulator)
        
        self.spread_modulator_mlp: nn.Module = nn.Identity()
        if self.level_config.use_level_spread:
            self.spread_modulator_mlp = _create_modulator_mlp("Spread", self.level_config.spread_modulator)

        # Level Descriptor (potentially dynamic)
        self.use_ld = self.level_config.use_level_descriptors
        self.base_level_descriptor_unconstrained: Optional[nn.Parameter] = None
        self.level_descriptor_modulator_mlp: nn.Module = nn.Identity()
        if self.use_ld and self.dim > 0:
            self.base_level_descriptor_unconstrained = nn.Parameter(torch.Tensor(self.dim)) # This is in Tangent space
            PoincareBall(c_scalar=initial_c).init_weights(self.base_level_descriptor_unconstrained, irange=self.level_config.level_descriptor_init_scale)
            self.level_descriptor_modulator_mlp = _create_modulator_mlp("LevelDescriptor", self.level_config.level_descriptor_modulator, output_dim=self.dim)
        else:
            self.register_parameter('base_level_descriptor_unconstrained', None)

        # Boundary Manifold (dynamic points handled within)
        num_boundaries = self.level_config.boundary_points
        self.boundary_manifold_module: Optional[BoundaryManifoldHyperbolic] = None
        if self.dim > 0 and num_boundaries > 0:
            self.boundary_manifold_module = BoundaryManifoldHyperbolic(
                self.level_idx, num_boundaries, self.dim, initial_c, 
                self.level_config, 
                self.g_w_level_features, 
                self.global_context_embedding_dim
            )

        # Tangent Combiner with advanced interactions
        comb_in_dim_calc = self.dim # tan_main
        if self.stack_config.relative_vector_aggregation not in ['none', None] and self.boundary_manifold_module is not None and self.boundary_manifold_module.num_points > 0 :
            comb_in_dim_calc += self.dim # tan_rel
        if self.use_ld and self.dim > 0: # Check dim > 0 for LD
            comb_in_dim_calc += self.dim # tan_desc_prev
        self.actual_comb_in_dim = comb_in_dim_calc

        interaction_output_dim = self.actual_comb_in_dim # Default if no specific interaction layer
        self.tangent_interaction_layer: nn.Module = nn.Identity()

        if self.actual_comb_in_dim > 0 and self.level_config.tangent_combiner_interaction_type != "concat":
            num_tangent_components = (1 if self.dim > 0 else 0) + \
                                 (1 if self.stack_config.relative_vector_aggregation != 'none' and self.boundary_manifold_module and self.boundary_manifold_module.num_points > 0 and self.dim > 0 else 0) + \
                                 (1 if self.use_ld and self.dim > 0 else 0)

            if self.level_config.tangent_combiner_interaction_type == "mha_light" and self.dim > 0 and num_tangent_components > 1:
                # MHA operates on a sequence of features.
                # Each component (main, rel, desc) has self.dim features.
                # Input to MHA: (Batch, NumComponents, self.dim)
                # Output from MHA: (Batch, NumComponents, self.dim) -> flatten to (Batch, NumComponents * self.dim)
                self.tangent_mha_layer = nn.MultiheadAttention(
                    embed_dim=self.dim, num_heads=max(1, self.level_config.mha_light_num_heads), # Ensure num_heads >= 1
                    dropout=self.stack_config.dropout, batch_first=True 
                )
                # The output dim after MHA and reshape will be num_tangent_components * self.dim
                interaction_output_dim = num_tangent_components * self.dim
                self.tangent_interaction_layer = self.tangent_mha_layer # Assign for use in forward
                self.logger.info(f"L{self.level_idx} Tangent Combiner using MHA-light. Num components: {num_tangent_components}, Dim per comp: {self.dim}, Heads: {self.level_config.mha_light_num_heads}. Interaction output dim: {interaction_output_dim}")
            
            elif self.level_config.tangent_combiner_interaction_type == "bilinear_pool" and self.dim > 0 and num_tangent_components > 1:
                # Example: if 2 components [A, B], bilinear A.T W B. If 3 [A,B,C], sum of pairwise?
                # This requires a more complex setup. For now, a simplified placeholder.
                # If num_components is 2, use nn.Bilinear. If >2, could use pairwise or a factorized pooling.
                if num_tangent_components == 2: # A, B
                    # Output dim for bilinear is configurable. Let's set it to self.dim.
                    self.bilinear_layer = nn.Bilinear(self.dim, self.dim, self.dim)
                    interaction_output_dim = self.dim
                    self.tangent_interaction_layer = self.bilinear_layer # Assign
                    self.logger.info(f"L{self.level_idx} Tangent Combiner using Bilinear. Interaction output dim: {interaction_output_dim}")
                else: # Fallback to concat if >2 components for bilinear or not implemented
                    self.logger.warning(f"L{self.level_idx} Bilinear pooling for >2 components not fully implemented, falling back to concat for interaction.")
                    self.level_config.tangent_combiner_interaction_type = "concat" # Change type
                    interaction_output_dim = self.actual_comb_in_dim # Revert to sum of dims

            else: # Unknown interaction type or not applicable
                 self.level_config.tangent_combiner_interaction_type = "concat" # Fallback
                 interaction_output_dim = self.actual_comb_in_dim

        # MLP stack for tangent_combiner (after interaction)
        combiner_mlp_layers = []
        current_mlp_in_dim_for_combiner = interaction_output_dim

        if current_mlp_in_dim_for_combiner > 0 and self.dim > 0:
            hidden_dim_combiner_mlp = max(16, int(current_mlp_in_dim_for_combiner * self.level_config.tangent_input_combination_dim_ratio))
            if hidden_dim_combiner_mlp > 0:
                combiner_mlp_layers.extend([nn.Linear(current_mlp_in_dim_for_combiner, hidden_dim_combiner_mlp), SwiGLUActivation(), nn.Dropout(self.stack_config.dropout)])
                combiner_mlp_layers.append(nn.Linear(hidden_dim_combiner_mlp, self.dim))
                if self.dim > 0: combiner_mlp_layers.append(nn.LayerNorm(self.dim))
            else: # Direct projection if hidden dim is 0
                combiner_mlp_layers.append(nn.Linear(current_mlp_in_dim_for_combiner, self.dim))
                if self.dim > 0: combiner_mlp_layers.append(nn.LayerNorm(self.dim))
        
        self.tangent_combiner_mlp = nn.Sequential(*combiner_mlp_layers) if combiner_mlp_layers else nn.Identity()

        # Tangent Flow (SwiGLU, learnable scale)
        self.use_flow = self.level_config.use_tangent_flow
        self.tangent_flow_module: Optional[nn.Module] = None
        self.log_tangent_flow_scale_unconstrained: Optional[nn.Parameter] = None

        if self.use_flow and self.dim > 0:
            flow_h_dim = max(16, int(self.dim * self.level_config.tangent_flow_hidden_dim_ratio))
            flow_type = self.level_config.tangent_flow_type.lower()
            flow_mlp_layers = []
            if flow_type == 'mlp' and flow_h_dim > 0 and self.dim > 0:
                flow_mlp_layers.extend([nn.Linear(self.dim, flow_h_dim), SwiGLUActivation(), nn.Dropout(self.stack_config.dropout), nn.Linear(flow_h_dim, self.dim)])
            elif flow_type == 'linear' and self.dim > 0:
                flow_mlp_layers.append(nn.Linear(self.dim, self.dim))
            
            if flow_mlp_layers:
                self.tangent_flow_module = nn.Sequential(*flow_mlp_layers)
                self.log_tangent_flow_scale_unconstrained = nn.Parameter(
                    _init_unconstrained_softplus(self.level_config.initial_learnable_tangent_flow_scale, EPS)
                )
        
        self.apply(init_weights_general)


    def _get_modulated_parameter(self, base_unconstrained_param: torch.Tensor,
                                modulator_mlp: nn.Module, mod_cfg: WuBuLevelModulatorConfig,
                                g_w_vec: Optional[torch.Tensor], global_ctx_emb: Optional[torch.Tensor],
                                min_val_or_range: Union[float, Tuple[float, float]], # Updated for sigmoid_scaled
                                transform_fn: str) -> torch.Tensor:
        current_val_unconstrained = base_unconstrained_param # Start with the base unconstrained value

        if mod_cfg.enabled and not isinstance(modulator_mlp, nn.Identity) and \
           g_w_vec is not None and global_ctx_emb is not None and \
           g_w_vec.numel() > 0 and global_ctx_emb.numel() > 0: # Check if context tensors are non-empty
            
            # Ensure context tensors are 2D (Batch=1, Dim) for MLP input
            g_w_vec_mlp_in = g_w_vec.view(1, -1) if g_w_vec.dim() == 1 else g_w_vec
            global_ctx_emb_mlp_in = global_ctx_emb.view(1, -1) if global_ctx_emb.dim() == 1 else global_ctx_emb

            # Assuming g_w_vec and global_ctx_emb are already on the correct device or will be moved.
            dev_mlp = next(modulator_mlp.parameters()).device # Device of the MLP
            
            # Check if MLP input layer expects concatenated dim
            expected_mlp_in_dim = -1
            if isinstance(modulator_mlp, nn.Sequential) and len(modulator_mlp) > 0 and isinstance(modulator_mlp[0], nn.Linear):
                expected_mlp_in_dim = modulator_mlp[0].in_features
            
            actual_concat_dim = g_w_vec_mlp_in.shape[-1] + global_ctx_emb_mlp_in.shape[-1]

            if expected_mlp_in_dim != -1 and actual_concat_dim == expected_mlp_in_dim :
                mlp_input = torch.cat([g_w_vec_mlp_in.to(dev_mlp), global_ctx_emb_mlp_in.to(dev_mlp)], dim=-1)
                modulation = modulator_mlp(mlp_input) # Output shape should match base_unconstrained_param (e.g., (1) or (D))
                
                # Ensure modulation has same shape as base_unconstrained_param for broadcasting or element-wise op
                if modulation.shape != base_unconstrained_param.shape:
                    modulation = modulation.view_as(base_unconstrained_param) # Attempt to reshape

                current_val_unconstrained = base_unconstrained_param + modulation * mod_cfg.global_context_influence_factor
            else:
                self.logger.debug(f"L{self.level_idx} Modulator MLP input dim mismatch or invalid. Expected {expected_mlp_in_dim}, got concat_dim {actual_concat_dim}. Skipping modulation.")
        
        # Apply transform function (softplus, sigmoid_scaled, or identity)
        if transform_fn == "softplus":
            if not isinstance(min_val_or_range, (float, int)):
                raise TypeError(f"min_val_or_range must be float for softplus, got {type(min_val_or_range)}")
            return F.softplus(current_val_unconstrained) + min_val_or_range
        elif transform_fn == "sigmoid_scaled":
            if not isinstance(min_val_or_range, tuple) or len(min_val_or_range) != 2:
                raise TypeError(f"min_val_or_range must be a tuple (min_r, max_r) for sigmoid_scaled, got {min_val_or_range}")
            min_r, max_r = min_val_or_range
            scaled_sigmoid = torch.sigmoid(current_val_unconstrained)
            return min_r + (max_r - min_r) * scaled_sigmoid
        elif transform_fn == "identity": # For tangent space vectors like Level Descriptor
            return current_val_unconstrained # No transformation, just the (modulated) unconstrained value
        else:
            self.logger.error(f"Unknown transform_fn: {transform_fn}. Defaulting to softplus.")
            if not isinstance(min_val_or_range, (float, int)): min_val_or_range = EPS # Fallback min_val
            return F.softplus(current_val_unconstrained) + min_val_or_range


    def get_current_curvature_scalar(self, g_w_vec: Optional[torch.Tensor]=None, global_ctx_emb: Optional[torch.Tensor]=None) -> float:
        val = self._get_modulated_parameter(
            self.base_log_curvature_unconstrained, self.curvature_modulator_mlp,
            self.level_config.curvature_modulator, g_w_vec, global_ctx_emb,
            self.min_curvature, "softplus"
        )
        return val.item()

    def get_current_scale_scalar(self, g_w_vec: Optional[torch.Tensor]=None, global_ctx_emb: Optional[torch.Tensor]=None) -> float:
        val = self._get_modulated_parameter(
            self.base_log_scale_unconstrained, self.scale_modulator_mlp,
            self.level_config.scale_modulator, g_w_vec, global_ctx_emb,
            (MIN_WUBU_LEVEL_SCALE, MAX_WUBU_LEVEL_SCALE), "sigmoid_scaled"
        )
        return val.item()

    def get_current_spread_scalar_tensor(self, g_w_vec: Optional[torch.Tensor]=None, global_ctx_emb: Optional[torch.Tensor]=None) -> torch.Tensor:
        if not self.level_config.use_level_spread or self.base_log_spread_unconstrained is None:
            ref_param = next(iter(self.parameters()), None) # type: ignore # Iterating over self.parameters()
            dev = ref_param.device if ref_param is not None else torch.device('cpu')
            dtype = ref_param.dtype if ref_param is not None else torch.float32
            return torch.tensor(self.min_spread, device=dev, dtype=dtype)

        return self._get_modulated_parameter(
            self.base_log_spread_unconstrained, self.spread_modulator_mlp, # type: ignore
            self.level_config.spread_modulator, g_w_vec, global_ctx_emb,
            self.min_spread, "softplus"
        )

    def get_current_level_descriptor_hyperbolic(self, current_manifold: PoincareBall,
                                              g_w_vec: Optional[torch.Tensor]=None, global_ctx_emb: Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
        if not self.use_ld or self.dim == 0 or self.base_level_descriptor_unconstrained is None:
            return None
        
        effective_tangent_ld = self._get_modulated_parameter(
            self.base_level_descriptor_unconstrained, self.level_descriptor_modulator_mlp,
            self.level_config.level_descriptor_modulator, g_w_vec, global_ctx_emb,
            0.0, "identity" 
        ) 
        effective_tangent_ld_clamped = torch.clamp(effective_tangent_ld, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        hyperbolic_ld = current_manifold.expmap0(effective_tangent_ld_clamped)
        return current_manifold.proju(hyperbolic_ld)

    def get_current_tangent_flow_scale(self, g_w_vec: Optional[torch.Tensor]=None, global_ctx_emb: Optional[torch.Tensor]=None) -> float:
        if not self.use_flow or self.log_tangent_flow_scale_unconstrained is None:
            return 0.0 
        # Assume flow scale is NOT dynamically modulated for now, just learnable base.
        # If it were dynamic, it would use _get_modulated_parameter like C, S, Sigma.
        return (F.softplus(self.log_tangent_flow_scale_unconstrained) + EPS).item()


    def forward(self, point_in_hyperbolic: torch.Tensor, 
                relative_vectors_tangent_in: Optional[torch.Tensor],
                descriptor_point_in_hyperbolic: Optional[torch.Tensor], 
                sigma_in_scalar_tensor: Optional[torch.Tensor] = None, 
                g_w_vectorized_features_for_level: Optional[torch.Tensor] = None, 
                global_context_embedding_for_level: Optional[torch.Tensor] = None  
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        
        # Determine batch size (B_prime) and original input format if input was (B,S,D)
        B_orig_fmt, S_orig_fmt = -1, -1
        if point_in_hyperbolic.dim() == 3:
            B_orig_fmt, S_orig_fmt, D_in_orig = point_in_hyperbolic.shape
            B_prime = B_orig_fmt * S_orig_fmt
            point_in_hyperbolic = point_in_hyperbolic.reshape(B_prime, D_in_orig)
            if relative_vectors_tangent_in is not None and relative_vectors_tangent_in.dim() == 3:
                 relative_vectors_tangent_in = relative_vectors_tangent_in.reshape(B_prime, -1)
            if descriptor_point_in_hyperbolic is not None and descriptor_point_in_hyperbolic.dim() == 3:
                 descriptor_point_in_hyperbolic = descriptor_point_in_hyperbolic.reshape(B_prime, -1)
        elif point_in_hyperbolic.dim() == 2:
            B_prime, _ = point_in_hyperbolic.shape
        else:
            raise ValueError(f"L{self.level_idx} forward expects 2D or 3D input, got {point_in_hyperbolic.dim()}D shape {point_in_hyperbolic.shape}")

        dev = point_in_hyperbolic.device
        ref_param_for_dtype_fwd = next(iter(self.parameters()), None) # type: ignore
        dtype_to_use_fwd = ref_param_for_dtype_fwd.dtype if ref_param_for_dtype_fwd is not None else point_in_hyperbolic.dtype

        if self.dim == 0: 
            dummy_out_shape = (B_prime, 0)
            dummy_dtype_dev_fwd = {'device': dev, 'dtype': dtype_to_use_fwd}
            current_spread_tensor_fwd = self.get_current_spread_scalar_tensor(
                g_w_vectorized_features_for_level, global_context_embedding_for_level
            ).to(dtype_to_use_fwd)
            # Ensure output shape matches original input if it was (B,S,D)
            if B_orig_fmt != -1:
                return (torch.zeros(B_orig_fmt, S_orig_fmt, 0, **dummy_dtype_dev_fwd), 
                        torch.zeros(B_orig_fmt, S_orig_fmt, 0, **dummy_dtype_dev_fwd),
                        None, None, current_spread_tensor_fwd)
            else:
                return (torch.zeros(dummy_out_shape, **dummy_dtype_dev_fwd), torch.zeros(dummy_out_shape, **dummy_dtype_dev_fwd),
                        None, None, current_spread_tensor_fwd)

        current_c_val_fwd = self.get_current_curvature_scalar(g_w_vectorized_features_for_level, global_context_embedding_for_level)
        current_s_val_fwd = self.get_current_scale_scalar(g_w_vectorized_features_for_level, global_context_embedding_for_level)
        current_sigma_out_tensor_fwd = self.get_current_spread_scalar_tensor(g_w_vectorized_features_for_level, global_context_embedding_for_level)
        current_manifold_obj_fwd = PoincareBall(c_scalar=current_c_val_fwd)

        boundary_points_this_level_hyperbolic_fwd = None
        if self.boundary_manifold_module:
            self.boundary_manifold_module.set_current_manifold_c(current_c_val_fwd)
            boundary_points_this_level_hyperbolic_fwd = self.boundary_manifold_module.get_points(
                g_w_vectorized_features_for_level, global_context_embedding_for_level
            )
            if boundary_points_this_level_hyperbolic_fwd is not None:
                 boundary_points_this_level_hyperbolic_fwd = boundary_points_this_level_hyperbolic_fwd.to(dtype=dtype_to_use_fwd, device=dev)

        point_in_proj_fwd = current_manifold_obj_fwd.proju(point_in_hyperbolic.to(dtype=dtype_to_use_fwd))
        tan_main_component_fwd = current_manifold_obj_fwd.logmap0(point_in_proj_fwd)

        tan_rel_component_fwd = None
        if relative_vectors_tangent_in is not None and \
           self.stack_config.relative_vector_aggregation not in ['none', None] and \
           self.boundary_manifold_module and self.boundary_manifold_module.num_points > 0:
            tan_rel_component_fwd = relative_vectors_tangent_in.to(dtype_to_use_fwd)
            if tan_rel_component_fwd.shape[0] != B_prime or tan_rel_component_fwd.shape[-1] != self.dim :
                self.logger.error(f"L{self.level_idx} RelVec shape/dim mismatch. Got {tan_rel_component_fwd.shape}, expected B_prime={B_prime}, Dim={self.dim}. Forcing zeros.")
                tan_rel_component_fwd = torch.zeros_like(tan_main_component_fwd)


        ld_point_self_hyperbolic_fwd = self.get_current_level_descriptor_hyperbolic(
            current_manifold_obj_fwd, g_w_vectorized_features_for_level, global_context_embedding_for_level
        )
        if ld_point_self_hyperbolic_fwd is not None: ld_point_self_hyperbolic_fwd = ld_point_self_hyperbolic_fwd.to(dtype=dtype_to_use_fwd)

        tan_desc_prev_level_component_fwd = None
        if descriptor_point_in_hyperbolic is not None and self.use_ld:
            desc_in_proj_fwd = current_manifold_obj_fwd.proju(descriptor_point_in_hyperbolic.to(dtype=dtype_to_use_fwd))
            tan_desc_prev_level_component_fwd = current_manifold_obj_fwd.logmap0(desc_in_proj_fwd)
            if tan_desc_prev_level_component_fwd.shape[0] != B_prime or tan_desc_prev_level_component_fwd.shape[-1] != self.dim:
                self.logger.error(f"L{self.level_idx} DescIn shape/dim mismatch. Got {tan_desc_prev_level_component_fwd.shape}, expected B_prime={B_prime}, Dim={self.dim}. Forcing zeros.")
                tan_desc_prev_level_component_fwd = torch.zeros_like(tan_main_component_fwd)


        inputs_for_interaction_fwd = []
        if tan_main_component_fwd is not None: inputs_for_interaction_fwd.append(tan_main_component_fwd)
        if tan_rel_component_fwd is not None: inputs_for_interaction_fwd.append(tan_rel_component_fwd)
        if tan_desc_prev_level_component_fwd is not None: inputs_for_interaction_fwd.append(tan_desc_prev_level_component_fwd)

        if not inputs_for_interaction_fwd:
            combined_tangent_features_for_mlp_fwd = torch.zeros(B_prime, self.dim, device=dev, dtype=dtype_to_use_fwd)
        else:
            if self.level_config.tangent_combiner_interaction_type == "mha_light" and isinstance(self.tangent_interaction_layer, nn.MultiheadAttention) and len(inputs_for_interaction_fwd) > 1:
                stacked_inputs_fwd = torch.stack(inputs_for_interaction_fwd, dim=1) # (B_prime, NumComponents, Dim)
                mha_out_fwd, _ = self.tangent_interaction_layer(stacked_inputs_fwd, stacked_inputs_fwd, stacked_inputs_fwd) # Q=K=V
                combined_tangent_features_for_mlp_fwd = mha_out_fwd.reshape(B_prime, -1) 
            elif self.level_config.tangent_combiner_interaction_type == "bilinear_pool" and isinstance(self.tangent_interaction_layer, nn.Bilinear) and len(inputs_for_interaction_fwd) == 2:
                combined_tangent_features_for_mlp_fwd = self.tangent_interaction_layer(inputs_for_interaction_fwd[0], inputs_for_interaction_fwd[1])
            else: # Default to concatenation
                combined_tangent_features_for_mlp_fwd = torch.cat(inputs_for_interaction_fwd, dim=-1)
            
            current_dim_after_interaction_fwd = combined_tangent_features_for_mlp_fwd.shape[-1]
            expected_dim_for_mlp_fwd = 0
            if isinstance(self.tangent_combiner_mlp, nn.Sequential) and len(self.tangent_combiner_mlp) > 0 and isinstance(self.tangent_combiner_mlp[0], nn.Linear):
                 expected_dim_for_mlp_fwd = self.tangent_combiner_mlp[0].in_features
            
            if expected_dim_for_mlp_fwd > 0 and current_dim_after_interaction_fwd != expected_dim_for_mlp_fwd:
                if current_dim_after_interaction_fwd < expected_dim_for_mlp_fwd:
                    padding_size_fwd = expected_dim_for_mlp_fwd - current_dim_after_interaction_fwd
                    combined_tangent_features_for_mlp_fwd = F.pad(combined_tangent_features_for_mlp_fwd, (0, padding_size_fwd))
                else:
                    combined_tangent_features_for_mlp_fwd = combined_tangent_features_for_mlp_fwd[..., :expected_dim_for_mlp_fwd]
            elif expected_dim_for_mlp_fwd == 0 and current_dim_after_interaction_fwd > 0:
                 combined_tangent_features_for_mlp_fwd = torch.empty(B_prime, 0, device=dev, dtype=dtype_to_use_fwd)

        v_combined_tangent_processed_fwd = self.tangent_combiner_mlp(combined_tangent_features_for_mlp_fwd)
        v_final_for_expmap_unclamped_fwd = v_combined_tangent_processed_fwd * current_s_val_fwd

        if self.use_flow and self.tangent_flow_module is not None:
            current_flow_scale_fwd = self.get_current_tangent_flow_scale(g_w_vectorized_features_for_level, global_context_embedding_for_level)
            flow_effect_fwd = self.tangent_flow_module(v_combined_tangent_processed_fwd) * current_flow_scale_fwd
            v_final_for_expmap_unclamped_fwd = v_final_for_expmap_unclamped_fwd + flow_effect_fwd
        
        scaled_output_tangent_for_expmap_fwd = torch.clamp(v_final_for_expmap_unclamped_fwd, -TAN_VEC_CLAMP_VAL, TAN_VEC_CLAMP_VAL)
        point_this_level_out_hyperbolic_fwd = current_manifold_obj_fwd.expmap0(scaled_output_tangent_for_expmap_fwd)
        tangent_out_for_aggregation_fwd = v_combined_tangent_processed_fwd.to(dtype_to_use_fwd)

        descriptor_point_out_for_transform_hyperbolic_fwd = None
        if ld_point_self_hyperbolic_fwd is not None:
            # Expand if it's a single descriptor for the whole batch
            if ld_point_self_hyperbolic_fwd.dim() == 1: # (D)
                descriptor_point_out_for_transform_hyperbolic_fwd = ld_point_self_hyperbolic_fwd.unsqueeze(0).expand(B_prime, -1).to(dtype=dtype_to_use_fwd)
            elif ld_point_self_hyperbolic_fwd.shape[0] == 1 and B_prime > 1: # (1, D)
                 descriptor_point_out_for_transform_hyperbolic_fwd = ld_point_self_hyperbolic_fwd.expand(B_prime, -1).to(dtype=dtype_to_use_fwd)
            else: # Already (B_prime, D)
                 descriptor_point_out_for_transform_hyperbolic_fwd = ld_point_self_hyperbolic_fwd.to(dtype=dtype_to_use_fwd)


        original_input_dtype_fwd = point_in_hyperbolic.dtype 
        def _format_output_final(tensor_val_fwd: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor_val_fwd is None: return None
            if B_orig_fmt != -1 and S_orig_fmt != -1: # If original input was (B,S,D)
                # Ensure output dimension matches if it's not None
                # For boundary points, the second dim is num_boundaries, not S_orig_fmt
                if tensor_val_fwd.shape[0] == B_prime and (tensor_val_fwd is not boundary_points_this_level_hyperbolic_fwd):
                    return tensor_val_fwd.reshape(B_orig_fmt, S_orig_fmt, -1).to(dtype=original_input_dtype_fwd)
                # Boundary points are (NumBoundaries, Dim), not batched further by S_orig_fmt
                # Descriptor could be (B_prime, Dim) or (NumDesc, Dim) - here it's (B_prime, Dim) for transform
            return tensor_val_fwd.to(dtype=original_input_dtype_fwd)

        # Special handling for boundary_points_this_level_hyperbolic_fwd as it's not per-batch-item
        # It's (NumBoundaries, Dim) and should remain so, just ensure dtype.
        formatted_boundary_points = boundary_points_this_level_hyperbolic_fwd.to(dtype=original_input_dtype_fwd) if boundary_points_this_level_hyperbolic_fwd is not None else None

        return (_format_output_final(point_this_level_out_hyperbolic_fwd),
                _format_output_final(tangent_out_for_aggregation_fwd),
                _format_output_final(descriptor_point_out_for_transform_hyperbolic_fwd),
                formatted_boundary_points, 
                current_sigma_out_tensor_fwd.to(dtype=original_input_dtype_fwd))

    def get_interpretability_data(self, g_w_vec: Optional[torch.Tensor]=None, global_ctx_emb: Optional[torch.Tensor]=None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "level_idx": self.level_idx,
            "hyperbolic_dim": self.dim,
            "current_c": self.get_current_curvature_scalar(g_w_vec, global_ctx_emb),
            "current_s": self.get_current_scale_scalar(g_w_vec, global_ctx_emb),
        }
        current_spread_val = self.get_current_spread_scalar_tensor(g_w_vec, global_ctx_emb)
        data["current_sigma"] = current_spread_val.item() if torch.is_tensor(current_spread_val) else float(current_spread_val) # Ensure scalar
        
        if self.level_config.curvature_modulator.enabled and not isinstance(self.curvature_modulator_mlp, nn.Identity):
            data["c_base_unconstrained"] = self.base_log_curvature_unconstrained.item()
            # Could add norm of modulation if g_w_vec and global_ctx_emb are passed to compute it here
        if self.level_config.scale_modulator.enabled and not isinstance(self.scale_modulator_mlp, nn.Identity):
            data["s_base_unconstrained"] = self.base_log_scale_unconstrained.item()
        if self.level_config.use_level_spread and self.base_log_spread_unconstrained is not None and \
           self.level_config.spread_modulator.enabled and not isinstance(self.spread_modulator_mlp, nn.Identity):
            data["sigma_base_unconstrained"] = self.base_log_spread_unconstrained.item() # type: ignore

        if self.use_ld and self.base_level_descriptor_unconstrained is not None:
            current_manifold_for_interp = PoincareBall(data["current_c"])
            ld_h_interp = self.get_current_level_descriptor_hyperbolic(current_manifold_for_interp, g_w_vec, global_ctx_emb)
            if ld_h_interp is not None: 
                data["level_descriptor_hyper_norm_mean"] = ld_h_interp.norm(dim=-1).mean().item() # If LD is batched
                if self.level_config.level_descriptor_modulator.enabled and not isinstance(self.level_descriptor_modulator_mlp, nn.Identity):
                     data["ld_base_unconstrained_norm"] = self.base_level_descriptor_unconstrained.norm().item()
        
        if self.boundary_manifold_module and self.boundary_manifold_module.num_points > 0:
            # Pass context to get potentially dynamic boundary points for interpretation
            bounds_h_interp = self.boundary_manifold_module.get_points(g_w_vec, global_ctx_emb)
            if bounds_h_interp is not None: 
                data["boundary_points_hyper_mean_norm"] = bounds_h_interp.norm(dim=-1).mean().item()
                if self.level_config.boundary_points_modulator.enabled : # Log base params if dynamic
                    base_norms = [bp.norm().item() for bp in self.boundary_manifold_module.base_boundary_point_params_unconstrained]
                    if base_norms: data["boundary_base_params_mean_norm"] = np.mean(base_norms)


        if self.use_flow and self.log_tangent_flow_scale_unconstrained is not None:
            data["tangent_flow_scale_effective"] = self.get_current_tangent_flow_scale(g_w_vec, global_ctx_emb)
            data["tangent_flow_scale_base_unconstrained"] = self.log_tangent_flow_scale_unconstrained.item()
        
        # Log norms of key learnable weights within the level
        if not isinstance(self.tangent_combiner_mlp, nn.Identity) and len(self.tangent_combiner_mlp) > 0:
            first_linear_combiner = next(filter(lambda m: isinstance(m, nn.Linear), self.tangent_combiner_mlp.modules()), None)
            if first_linear_combiner: data["tangent_combiner_mlp_L0_w_norm"] = first_linear_combiner.weight.norm().item()
        
        if self.tangent_flow_module is not None and not isinstance(self.tangent_flow_module, nn.Identity):
             first_linear_flow = next(filter(lambda m: isinstance(m, nn.Linear), self.tangent_flow_module.modules()), None)
             if first_linear_flow: data["tangent_flow_module_L0_w_norm"] = first_linear_flow.weight.norm().item()
        
        return data
        
# <TOTAL_STRATEGY_INTEGRATION> FullyHyperbolicWuBuNestingModel
# WuBuSpecTrans_v0.2.0_TotalStrategy (Continued)

class FullyHyperbolicWuBuNestingModel(nn.Module):
    def __init__(self, input_tangent_dim: int, output_tangent_dim: int,
                 wubu_stack_config: WuBuStackConfig, 
                 global_context_config: WuBuGlobalContextConfig 
                 ):
        super().__init__()
        self.input_tangent_dim = input_tangent_dim
        self.output_tangent_dim = output_tangent_dim
        self.wubu_stack_config = wubu_stack_config
        self.global_context_config = global_context_config
        self.num_levels = self.wubu_stack_config.num_levels
        self.logger = logging.getLogger(f"{logger.name}.WuBuModel.{self.wubu_stack_config.stack_name}")

        self.global_context_embedding_dim = 0
        self.global_context_embed_layer: nn.Module = nn.Identity() # Default to Identity

        # Check if any level in the stack actually uses dynamic modulation
        any_dynamic_modulation_enabled = False
        if self.wubu_stack_config.levels_config: # Ensure levels_config is not empty
            any_dynamic_modulation_enabled = any(
                lvl_cfg.curvature_modulator.enabled or \
                lvl_cfg.scale_modulator.enabled or \
                lvl_cfg.spread_modulator.enabled or \
                lvl_cfg.level_descriptor_modulator.enabled or \
                lvl_cfg.boundary_points_modulator.enabled
                for lvl_cfg in self.wubu_stack_config.levels_config
            )

        if any_dynamic_modulation_enabled:
            ctx_input_dim = 0
            if self.global_context_config.use_epoch_frac: ctx_input_dim +=1
            if self.global_context_config.use_gstep_frac: ctx_input_dim +=1
            
            if ctx_input_dim > 0 and self.global_context_config.embedding_dim > 0:
                # Using SwiGLU (SiLU based) as per strategy
                self.global_context_embed_layer = nn.Sequential(
                    nn.Linear(ctx_input_dim, self.global_context_config.embedding_dim),
                    SwiGLUActivation(), 
                    nn.Linear(self.global_context_config.embedding_dim, self.global_context_config.embedding_dim)
                )
                self.global_context_embedding_dim = self.global_context_config.embedding_dim
                self.logger.info(f"Global Context Embedding Layer ENABLED for WuBu stack '{self.wubu_stack_config.stack_name}'. Raw Ctx Dim: {ctx_input_dim}, Emb Dim: {self.global_context_embedding_dim}")
            else:
                 self.logger.info(f"Global Context Embedding for WuBu stack '{self.wubu_stack_config.stack_name}' is NOT USED (raw_ctx_dim={ctx_input_dim} or emb_dim={self.global_context_config.embedding_dim} is zero).")
        else:
            self.logger.info(f"No dynamic geometry modulators enabled in WuBu stack '{self.wubu_stack_config.stack_name}'. Global context embedding layer is Identity.")

        first_level_dim = self.wubu_stack_config.levels_config[0].hyperbolic_dim if self.num_levels > 0 and self.wubu_stack_config.levels_config else 0
        if input_tangent_dim > 0 and first_level_dim > 0 and input_tangent_dim != first_level_dim:
            self.input_tangent_projection = nn.Linear(input_tangent_dim, first_level_dim)
        else: self.input_tangent_projection = nn.Identity()
        
        self.input_tangent_layernorm = nn.LayerNorm(first_level_dim) if first_level_dim > 0 else nn.Identity()

        self.levels_modulelist = nn.ModuleList()
        self.transforms_modulelist = nn.ModuleList()

        if self.num_levels > 0:
            if not self.wubu_stack_config.levels_config or len(self.wubu_stack_config.levels_config) != self.num_levels:
                raise ValueError(f"WuBu Stack '{self.wubu_stack_config.stack_name}': levels_config length mismatch or empty.")

            for i in range(self.num_levels):
                level_cfg = self.wubu_stack_config.levels_config[i]
                
                # Calculate g_W features for this level
                g_w_complexity_score = (
                    self.input_tangent_dim * self.wubu_stack_config.g_w_input_dim_factor +
                    level_cfg.hyperbolic_dim * self.wubu_stack_config.g_w_hyperbolic_dim_factor +
                    float(i) * self.wubu_stack_config.g_w_level_idx_factor + # Ensure float for calculation
                    float(self.num_levels) * self.wubu_stack_config.g_w_num_total_levels_factor
                )
                
                # For g_w_vectorized_features, create a simple vector of these components.
                # The dimension must match what modulators expect.
                # For simplicity, let's make this vector explicit here.
                # This part is crucial and might need a more sophisticated embedding if complex g_W features are used.
                g_w_features_list_for_vec = [
                    g_w_complexity_score, 
                    float(i), 
                    float(self.num_levels), 
                    float(self.input_tangent_dim), 
                    float(level_cfg.hyperbolic_dim)
                ]
                # Pad or truncate to a fixed dimension if modulators expect that.
                # For now, assume modulators can handle a variable-length g_w feature vector or the first element (score).
                # The _get_g_w_vectorized_features method below will handle creating the actual tensor.
                # The 'vectorized_features_dim' in g_w_level_features_dict will be set by the length of this list.
                
                g_w_level_features_dict = {
                    "complexity_score": g_w_complexity_score,
                    "level_idx": float(i),
                    "num_total_levels": float(self.num_levels),
                    "input_dim_to_stack": float(self.input_tangent_dim),
                    "level_hyperbolic_dim": float(level_cfg.hyperbolic_dim),
                    "vectorized_features_list": g_w_features_list_for_vec, # Store the list of raw features
                    "vectorized_features_dim": len(g_w_features_list_for_vec), # This will be the dim of the tensor created by _get_g_w_vectorized_features
                }

                self.levels_modulelist.append(
                    HyperbolicWuBuNestingLevel(i, level_cfg, self.wubu_stack_config,
                                             g_w_level_features_dict, self.global_context_embedding_dim)
                )
            
            num_transforms_needed = max(0, self.num_levels - 1)
            if num_transforms_needed > 0:
                if not self.wubu_stack_config.transforms_config or len(self.wubu_stack_config.transforms_config) != num_transforms_needed:
                     raise ValueError(f"WuBu Stack '{self.wubu_stack_config.stack_name}': transforms_config length mismatch or empty.")

                for i in range(num_transforms_needed):
                    transform_cfg = self.wubu_stack_config.transforms_config[i]
                    level_i_cfg = self.wubu_stack_config.levels_config[i]
                    level_i_plus_1_cfg = self.wubu_stack_config.levels_config[i+1]
                    self.transforms_modulelist.append(
                        HyperbolicInterLevelTransform(
                            level_i_cfg.hyperbolic_dim, level_i_plus_1_cfg.hyperbolic_dim,
                            level_i_cfg.initial_curvature, level_i_plus_1_cfg.initial_curvature, 
                            transform_cfg, level_idx_for_phi=i, dropout_val=self.wubu_stack_config.dropout
                        )
                    )
        
        aggregated_tangent_dim_val = sum(
            lvl_cfg.hyperbolic_dim for lvl_cfg in self.wubu_stack_config.levels_config if lvl_cfg.hyperbolic_dim > 0
        ) if self.wubu_stack_config.levels_config else self.input_tangent_dim

        if aggregated_tangent_dim_val > 0 and self.output_tangent_dim > 0 and aggregated_tangent_dim_val != self.output_tangent_dim:
            self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, self.output_tangent_dim)
        elif self.output_tangent_dim == 0 and aggregated_tangent_dim_val > 0 : 
            self.output_tangent_projection = nn.Linear(aggregated_tangent_dim_val, 0) 
        elif aggregated_tangent_dim_val == 0 and self.output_tangent_dim > 0:
            # This case is problematic: no features to project from.
            # Output will be zeros if projection expects >0 input.
            # It's better to make output_tangent_projection an Identity if input is 0 and output is 0.
            # If input is 0 and output >0, this needs a learnable bias or similar.
            self.logger.warning(f"Stack '{self.wubu_stack_config.stack_name}': Aggregated tangent dim is 0, but output_tangent_dim is {self.output_tangent_dim}. Output projection might be ill-defined.")
            if self.output_tangent_dim == 0: self.output_tangent_projection = nn.Identity()
            else: # Create a learnable bias vector if output_dim > 0 and input_dim = 0
                  # This is unusual but handles the case.
                self.output_bias_param = nn.Parameter(torch.Tensor(self.output_tangent_dim))
                nn.init.zeros_(self.output_bias_param)
                self.output_tangent_projection = lambda x: self.output_bias_param.unsqueeze(0).expand(x.shape[0] if x.dim() > 1 else 1, -1) # type: ignore

        else: self.output_tangent_projection = nn.Identity()
        
        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Stack '{self.wubu_stack_config.stack_name}': {self.num_levels} levels. Params: {param_count:,}. InDim {self.input_tangent_dim}, AggDim {aggregated_tangent_dim_val}, OutDim {self.output_tangent_dim}")


    def _get_g_w_vectorized_features(self, level_module_ref: HyperbolicWuBuNestingLevel) -> Optional[torch.Tensor]:
        """
        Creates a vectorized tensor from the g_w_level_features dictionary of a level.
        This tensor is used as input to modulator MLPs.
        """
        features_dict = level_module_ref.g_w_level_features
        raw_feature_list = features_dict.get("vectorized_features_list")
        expected_dim = features_dict.get("vectorized_features_dim", 0)

        if not raw_feature_list or expected_dim == 0:
            # If no raw features or expected dim is 0, modulator MLPs won't have g_w input.
            # Return None or an empty tensor based on what modulator MLPs expect for disabled g_w input.
            # Modulator MLPs should handle None or 0-dim g_w_vec gracefully.
            return torch.empty(0, device=next(self.parameters()).device) # type: ignore

        # Convert the list of scalar features to a tensor
        try:
            # Ensure all elements in raw_feature_list are numeric before converting
            numeric_features = [float(f) for f in raw_feature_list if isinstance(f, (int, float))]
            if len(numeric_features) != len(raw_feature_list):
                self.logger.warning(f"Stack '{self.wubu_stack_config.stack_name}', Level {level_module_ref.level_idx}: Non-numeric features in g_w_vectorized_features_list. Using only numeric ones.")
            
            if not numeric_features: # If all were non-numeric or list was empty
                return torch.empty(0, device=next(self.parameters()).device) # type: ignore

            g_w_tensor = torch.tensor(numeric_features, dtype=torch.float32, device=next(self.parameters()).device) # type: ignore
        except Exception as e:
            self.logger.error(f"Error vectorizing g_w_features for L{level_module_ref.level_idx}: {e}. Returning empty.")
            return torch.empty(0, device=next(self.parameters()).device) # type: ignore

        # Ensure the tensor dimension matches 'vectorized_features_dim' if it was pre-calculated
        # This might involve padding or truncation if the list length changed.
        # For now, assume the list length directly defines the dim.
        if g_w_tensor.shape[0] != expected_dim:
            self.logger.warning(f"Stack '{self.wubu_stack_config.stack_name}', Level {level_module_ref.level_idx}: g_w_tensor dim {g_w_tensor.shape[0]} != expected_dim {expected_dim}. This might cause issues with modulator MLPs.")
            # Fallback: if dim mismatch, return empty to signal modulator MLP to not use g_w features.
            # This is safer than passing a mismatched tensor.
            # Or, pad/truncate if a fixed dim is strictly required by MLPs.
            # For now, rely on MLP input dim checks.
            if g_w_tensor.shape[0] < expected_dim:
                g_w_tensor = F.pad(g_w_tensor, (0, expected_dim - g_w_tensor.shape[0]))
            else:
                g_w_tensor = g_w_tensor[:expected_dim]
        
        return g_w_tensor


    def forward(self, x_initial_tangent_in: torch.Tensor, global_context_raw_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dim_orig_fmt = x_initial_tangent_in.dim()
        B_orig, S_orig, D_orig_fmt = -1, -1, -1 # For 3D input (Batch, Sequence, Dim)
        B_prime_for_levels: int # Effective batch size for level processing

        if input_dim_orig_fmt == 3: 
            B_orig, S_orig, D_orig_fmt = x_initial_tangent_in.shape
            x_proc = x_initial_tangent_in.reshape(B_orig * S_orig, D_orig_fmt)
            B_prime_for_levels = B_orig * S_orig
        elif input_dim_orig_fmt == 2: 
            B_prime_for_levels, D_orig_fmt = x_initial_tangent_in.shape
            x_proc = x_initial_tangent_in
        else: raise ValueError(f"WuBuModel expects 2D/3D input, got {input_dim_orig_fmt}D shape {x_initial_tangent_in.shape}")
        
        if D_orig_fmt != self.input_tangent_dim:
            raise ValueError(f"Input feature dim {D_orig_fmt} != model input_tangent_dim {self.input_tangent_dim}")

        if self.num_levels == 0 or not self.levels_modulelist:
            # If output_tangent_projection is a lambda for bias (when aggregated_tangent_dim_val == 0)
            if callable(self.output_tangent_projection) and not isinstance(self.output_tangent_projection, nn.Module):
                 out_proj = self.output_tangent_projection(x_proc) # Pass x_proc for batch size expansion of bias
            else: out_proj = self.output_tangent_projection(x_proc)

            return out_proj.reshape(B_orig, S_orig, -1) if input_dim_orig_fmt == 3 and B_orig != -1 else out_proj

        dev = x_proc.device
        # Determine dtype from parameters if possible, else from input
        ref_param_for_dtype_fwd_stack = next(iter(self.parameters()), None) # type: ignore
        dtype_to_use_fwd_stack = ref_param_for_dtype_fwd_stack.dtype if ref_param_for_dtype_fwd_stack is not None else x_proc.dtype
        x_proc = x_proc.to(dtype=dtype_to_use_fwd_stack)

        current_global_context_embedding: Optional[torch.Tensor] = None
        if not isinstance(self.global_context_embed_layer, nn.Identity) and global_context_raw_features is not None:
            # Move raw features to device and ensure correct dtype for embed_layer
            processed_global_ctx_raw_features = global_context_raw_features.to(device=dev, dtype=dtype_to_use_fwd_stack)
            current_global_context_embedding = self.global_context_embed_layer(processed_global_ctx_raw_features)
            
            # Context embedding should be (1, EmbDim) or (EmbDim) for per-level/stack parameters,
            # or (B_prime, EmbDim) if it's meant to be per-item.
            # Current assumption: context is for the whole stack/level, not per batch item.
            if current_global_context_embedding.dim() > 1 and current_global_context_embedding.shape[0] == B_prime_for_levels and B_prime_for_levels > 1:
                 # If context became batched (e.g. if raw_features were per-item), average it for global stack/level params
                 current_global_context_embedding = current_global_context_embedding.mean(dim=0, keepdim=True)
            elif current_global_context_embedding.dim() == 0: # Scalar output
                 current_global_context_embedding = current_global_context_embedding.unsqueeze(0) # Make it (1,)

        current_tangent_projected = self.input_tangent_projection(x_proc)
        current_tangent_for_level0 = self.input_tangent_layernorm(current_tangent_projected)
        
        level0_module_ref = self.levels_modulelist[0]
        g_w_vec_l0_fwd = self._get_g_w_vectorized_features(level0_module_ref)
        c0_val_fwd = level0_module_ref.get_current_curvature_scalar(g_w_vec_l0_fwd, current_global_context_embedding)
        m0_obj_fwd = PoincareBall(c_scalar=c0_val_fwd)

        current_point_repr_hyperbolic: torch.Tensor
        if self.wubu_stack_config.levels_config[0].hyperbolic_dim > 0:
            current_point_repr_hyperbolic = m0_obj_fwd.expmap0(current_tangent_for_level0)
        else: # Handle 0-dim hyperbolic space for a level
            current_point_repr_hyperbolic = torch.empty(B_prime_for_levels, 0, device=dev, dtype=dtype_to_use_fwd_stack)
        
        level_tangent_outputs_for_aggregation: List[torch.Tensor] = []
        aggregated_relative_vectors_from_prev_transform: Optional[torch.Tensor] = None
        descriptor_from_prev_transform_hyperbolic: Optional[torch.Tensor] = None
        # Initialize sigma_from_prev_level_tensor correctly for the first level
        # This could be a learnable param or a fixed small value if not passed.
        # For now, use 0.0, but it should align with how sigma is used (e.g. if it's noise std).
        sigma_from_prev_level_tensor = torch.tensor(0.0, device=dev, dtype=dtype_to_use_fwd_stack)

        for i, level_module_fwd in enumerate(self.levels_modulelist):
            g_w_vec_level_i_fwd = self._get_g_w_vectorized_features(level_module_fwd)
            (point_out_of_level_hyperbolic, tangent_out_of_level_for_aggregation,
             descriptor_generated_by_level_hyperbolic, boundary_points_of_level_hyperbolic,
             sigma_out_of_level_tensor) = level_module_fwd(
                current_point_repr_hyperbolic, 
                aggregated_relative_vectors_from_prev_transform,
                descriptor_from_prev_transform_hyperbolic, 
                sigma_from_prev_level_tensor,
                g_w_vec_level_i_fwd, current_global_context_embedding
            )
            
            if self.wubu_stack_config.levels_config[i].hyperbolic_dim > 0 and tangent_out_of_level_for_aggregation is not None:
                level_tangent_outputs_for_aggregation.append(tangent_out_of_level_for_aggregation)

            if i < self.num_levels - 1: # If not the last level
                if i >= len(self.transforms_modulelist): # Should not happen if config is correct
                    self.logger.error(f"Missing transform L{i}->L{i+1} for stack '{self.wubu_stack_config.stack_name}'. Stopping propagation at this level."); break
                
                transform_module_fwd = self.transforms_modulelist[i]
                next_level_module_fwd = self.levels_modulelist[i+1]
                
                g_w_vec_next_level_fwd = self._get_g_w_vectorized_features(next_level_module_fwd)
                c_in_for_transform_fwd = level_module_fwd.get_current_curvature_scalar(g_w_vec_level_i_fwd, current_global_context_embedding)
                c_out_for_transform_fwd = next_level_module_fwd.get_current_curvature_scalar(g_w_vec_next_level_fwd, current_global_context_embedding)

                (point_transformed_to_next_level_hyperbolic,
                 boundaries_transformed_to_next_level_hyperbolic,
                 descriptor_transformed_to_next_level_hyperbolic
                ) = transform_module_fwd(
                    point_out_of_level_hyperbolic, boundary_points_of_level_hyperbolic,
                    descriptor_generated_by_level_hyperbolic, 
                    c_in_for_transform_fwd, c_out_for_transform_fwd
                )
                current_point_repr_hyperbolic = point_transformed_to_next_level_hyperbolic
                descriptor_from_prev_transform_hyperbolic = descriptor_transformed_to_next_level_hyperbolic
                sigma_from_prev_level_tensor = sigma_out_of_level_tensor 

                aggregated_relative_vectors_from_prev_transform = None 
                next_level_dim_fwd = self.wubu_stack_config.levels_config[i+1].hyperbolic_dim
                
                # Check if the *source* level (current `level_module_fwd`) had boundary points configured,
                # as `boundaries_transformed_to_next_level_hyperbolic` originates from them.
                source_level_had_boundaries = level_module_fwd.boundary_manifold_module is not None and \
                                              level_module_fwd.boundary_manifold_module.num_points > 0
                
                valid_boundary_conditions_for_rel_vec_fwd = (
                    boundaries_transformed_to_next_level_hyperbolic is not None and
                    self.wubu_stack_config.relative_vector_aggregation not in ['none', None] and
                    next_level_dim_fwd > 0 and
                    current_point_repr_hyperbolic.shape[-1] > 0 and # Point to compare against
                    source_level_had_boundaries # Ensure boundaries actually existed to be transformed
                )

                if valid_boundary_conditions_for_rel_vec_fwd:
                    manifold_next_level_obj_fwd = PoincareBall(c_scalar=c_out_for_transform_fwd)
                    tan_main_next_level_fwd = manifold_next_level_obj_fwd.logmap0(current_point_repr_hyperbolic)
                    tan_bounds_next_level_fwd = manifold_next_level_obj_fwd.logmap0(boundaries_transformed_to_next_level_hyperbolic) # type: ignore
                    
                    tan_bounds_next_level_expanded_fwd = tan_bounds_next_level_fwd.unsqueeze(0).expand(B_prime_for_levels, -1, -1)
                    relative_tangent_vectors_fwd = tan_main_next_level_fwd.unsqueeze(1) - tan_bounds_next_level_expanded_fwd
                    
                    agg_mode_fwd = self.wubu_stack_config.relative_vector_aggregation
                    agg_rel_vec_fwd: Optional[torch.Tensor] = None
                    if agg_mode_fwd == "mean": agg_rel_vec_fwd = torch.mean(relative_tangent_vectors_fwd, dim=1)
                    elif agg_mode_fwd == "sum": agg_rel_vec_fwd = torch.sum(relative_tangent_vectors_fwd, dim=1)
                    elif agg_mode_fwd == "max_norm":
                        norms_fwd = torch.norm(relative_tangent_vectors_fwd, p=2, dim=-1) 
                        best_idx_fwd = torch.argmax(norms_fwd, dim=1, keepdim=True) 
                        best_idx_expanded_fwd = best_idx_fwd.unsqueeze(-1).expand(-1, -1, relative_tangent_vectors_fwd.shape[-1])
                        agg_rel_vec_fwd = torch.gather(relative_tangent_vectors_fwd, 1, best_idx_expanded_fwd).squeeze(1)
                    
                    if agg_rel_vec_fwd is not None and not torch.isfinite(agg_rel_vec_fwd).all():
                        agg_rel_vec_fwd = torch.zeros_like(tan_main_next_level_fwd) 
                    aggregated_relative_vectors_from_prev_transform = agg_rel_vec_fwd
        
        compatible_tangent_outputs_fwd = [
            t_val for t_idx, t_val in enumerate(level_tangent_outputs_for_aggregation)
            if t_val is not None and self.wubu_stack_config.levels_config[t_idx].hyperbolic_dim > 0 and torch.isfinite(t_val).all()
        ]

        if not compatible_tangent_outputs_fwd:
            # If output_tangent_projection is a lambda for bias
            if callable(self.output_tangent_projection) and not isinstance(self.output_tangent_projection, nn.Module):
                # Create dummy input for lambda to get correct batch size for bias
                dummy_input_for_bias = torch.empty(B_prime_for_levels, 0, device=dev, dtype=dtype_to_use_fwd_stack)
                out_zeros_fwd = self.output_tangent_projection(dummy_input_for_bias)
            else: # Standard nn.Identity or nn.Linear(0, D)
                out_zeros_fwd = torch.zeros((B_prime_for_levels, self.output_tangent_dim), device=dev, dtype=dtype_to_use_fwd_stack)

            return out_zeros_fwd.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim_orig_fmt == 3 and B_orig !=-1 else out_zeros_fwd
        
        aggregated_tangent_final_fwd = torch.cat(compatible_tangent_outputs_fwd, dim=-1)
        
        final_output_flat_fwd: torch.Tensor
        if callable(self.output_tangent_projection) and not isinstance(self.output_tangent_projection, nn.Module):
             final_output_flat_fwd = self.output_tangent_projection(aggregated_tangent_final_fwd) # Pass actual aggregate if bias expansion needs it
        else: final_output_flat_fwd = self.output_tangent_projection(aggregated_tangent_final_fwd)
        
        if not torch.isfinite(final_output_flat_fwd).all():
            final_output_flat_fwd = torch.nan_to_num(final_output_flat_fwd, nan=0.0, posinf=TAN_VEC_CLAMP_VAL, neginf=-TAN_VEC_CLAMP_VAL)
            
        return final_output_flat_fwd.reshape(B_orig, S_orig, self.output_tangent_dim) if input_dim_orig_fmt == 3 and B_orig != -1 else final_output_flat_fwd

    def get_interpretability_data_stack(self, global_context_raw_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        stack_data: Dict[str, Any] = {"stack_name": self.wubu_stack_config.stack_name, "levels_interp": [], "transforms_interp": []} # Renamed keys
        
        current_global_context_embedding_interp: Optional[torch.Tensor] = None
        if not isinstance(self.global_context_embed_layer, nn.Identity) and global_context_raw_features is not None:
            # Ensure context is on the same device as model parameters for embed_layer
            dev_interp = next(self.parameters()).device # type: ignore
            dtype_interp = next(self.parameters()).dtype # type: ignore
            processed_global_ctx_raw_interp = global_context_raw_features.to(device=dev_interp, dtype=dtype_interp)
            current_global_context_embedding_interp = self.global_context_embed_layer(processed_global_ctx_raw_interp)
            if current_global_context_embedding_interp.dim() > 1 and current_global_context_embedding_interp.shape[0] > 1 :
                 current_global_context_embedding_interp = current_global_context_embedding_interp.mean(dim=0, keepdim=True)
            elif current_global_context_embedding_interp.dim() == 0:
                 current_global_context_embedding_interp = current_global_context_embedding_interp.unsqueeze(0)


        for i, level_module_interp in enumerate(self.levels_modulelist):
            g_w_vec_interp = self._get_g_w_vectorized_features(level_module_interp)
            stack_data["levels_interp"].append(level_module_interp.get_interpretability_data(g_w_vec_interp, current_global_context_embedding_interp))
        
        for i, transform_module_interp in enumerate(self.transforms_modulelist):
            if hasattr(transform_module_interp, 'get_interpretability_data'):
                stack_data["transforms_interp"].append(transform_module_interp.get_interpretability_data())
            else: # Should not happen if HILT has the method
                 stack_data["transforms_interp"].append({"transform_idx": i, "error": "get_interpretability_data not found"})
        
        # Add global stack parameters if any are directly part of this model (e.g. global_context_embed_layer weights)
        if not isinstance(self.global_context_embed_layer, nn.Identity):
            first_linear_global_ctx = next(filter(lambda m: isinstance(m, nn.Linear), self.global_context_embed_layer.modules()), None)
            if first_linear_global_ctx:
                stack_data["global_context_embed_L0_w_norm"] = first_linear_global_ctx.weight.norm().item()

        return stack_data

# GradientStats: Unchanged

# <TOTAL_STRATEGY_INTEGRATION> HAKMEMQController - Major Overhaul

class HAKMEMQController:
    MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False # Class variable

    def __init__(self, q_config: Dict, associated_component_name: str = "UnnamedComponent"):
        self.config = q_config 
        self.q_table: Dict[tuple, Dict[str, np.ndarray]] = {}
        
        self.base_alpha = float(self.config.get("q_learning_rate", 0.01))
        self.base_gamma = float(self.config.get("discount_factor", 0.90))
        self.current_alpha = self.base_alpha
        self.current_gamma = self.base_gamma
        
        self.epsilon_start = float(self.config.get("epsilon_start", 0.6))
        self.epsilon = self.epsilon_start
        self.epsilon_min = float(self.config.get("epsilon_min", 0.05))
        self.epsilon_decay = float(self.config.get("epsilon_decay", 0.9995))
        
        reward_clip_tuple = self.config.get("reward_clipping", (-2.5, 2.5))
        self.reward_clipping: Optional[Tuple[float, float]] = (float(reward_clip_tuple[0]), float(reward_clip_tuple[1])) if reward_clip_tuple else None
        
        q_clip_tuple = self.config.get("q_value_clipping", (-35.0, 35.0))
        self.q_value_clipping: Optional[Tuple[float, float]] = (float(q_clip_tuple[0]), float(q_clip_tuple[1])) if q_clip_tuple else None
        
        self.current_lambda_kl: float = 0.0001 

        self.action_ranges: Dict[str, np.ndarray] = {
            'lr_scale': np.array(self.config.get("lr_scale_options", [0.7, 0.85, 1.0, 1.15, 1.3]), dtype=np.float32),
            'momentum_scale': np.array(self.config.get("momentum_scale_options", [0.9, 0.95, 0.99, 1.0, 1.01]), dtype=np.float32),
            'lambda_kl_scale': np.array(self.config.get("lambda_kl_scale_options", [0.80, 0.90, 1.0, 1.10, 1.20]), dtype=np.float32),
            'heuristic_toggle_generic': np.array(self.config.get("heuristic_toggle_options", [0.0, 1.0]), dtype=np.float32),
        }
        # Optimizer type switch needs careful handling of dtype if it's strings
        opt_type_options = self.config.get("optimizer_type_options", ["default"])
        if not isinstance(opt_type_options, list) or not all(isinstance(o, str) for o in opt_type_options) :
            opt_type_options = ["default"] # Fallback
        self.action_ranges['optimizer_type_switch'] = np.array(opt_type_options, dtype=object)

        self.num_actions: Dict[str, int] = {p_type: len(actions) for p_type, actions in self.action_ranges.items()}

        self.state_history_len = max(3, int(self.config.get("state_history_len", 7)))
        self.loss_g_total_hist: deque = deque(maxlen=self.state_history_len)
        self.loss_g_recon_hist: deque = deque(maxlen=self.state_history_len)
        self.loss_g_kl_hist: deque = deque(maxlen=self.state_history_len)
        self.loss_g_adv_hist: deque = deque(maxlen=self.state_history_len)
        self.loss_d_total_hist: deque = deque(maxlen=self.state_history_len)
        self.loss_d_real_hist: deque = deque(maxlen=self.state_history_len)
        self.loss_d_fake_hist: deque = deque(maxlen=self.state_history_len)

        self.wubu_avg_curvature_hist: deque = deque(maxlen=self.state_history_len)
        self.wubu_avg_scale_hist: deque = deque(maxlen=self.state_history_len)
        self.wubu_avg_spread_hist: deque = deque(maxlen=self.state_history_len)
        self.wubu_var_curvature_hist: deque = deque(maxlen=self.state_history_len)

        self.lambda_kl_state_history_len = max(3, int(self.config.get("lambda_kl_state_history_len", 7)))
        self.interval_avg_recon_hist: deque = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_avg_kl_div_hist: deque = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_avg_d_total_hist: deque = deque(maxlen=self.lambda_kl_state_history_len)
        self.interval_val_metric_hist: deque = deque(maxlen=self.lambda_kl_state_history_len)

        self.prev_lr_mom_state: Optional[tuple] = None
        self.prev_lr_mom_action: Optional[Dict[str, Any]] = None
        self.prev_lambda_kl_state: Optional[tuple] = None
        self.prev_lambda_kl_action: Optional[Dict[str, Any]] = None
        self.prev_heuristic_toggle_state: Optional[tuple] = None
        self.prev_heuristic_toggle_action: Optional[Dict[str, Any]] = None

        self.reward_hist: deque = deque(maxlen=150) 
        self.max_q_table_size = int(self.config.get("max_q_table_size", 15000))
        self.q_table_access_count: Dict[tuple, int] = defaultdict(int)
        self.q_table_creation_time: Dict[tuple, float] = {}
        self.q_table_last_access_time: Dict[tuple, float] = {}
        
        # Reward Weights - Use a copy to avoid modifying the template dict if it's shared
        default_rw_template = {
            "g_recon_improvement": 3.0, "g_adv_improvement": 1.5,
            "g_kl_control_penalty_ratio_trigger": 0.75, "g_kl_control_penalty_abs_trigger_low": 30.0,
            "g_kl_control_penalty_abs_trigger_high": 100.0, "g_kl_control_penalty": 0.4,
            "g_loss_stability": 0.15, "g_easy_win_adv_thresh": 0.1, "g_easy_win_recon_thresh": 0.15,
            "g_easy_win_recon_bad_penalty": 0.75, "d_balance_target": 1.8, "d_real_low_bonus": 0.8,
            "d_fake_low_meaningful_bonus": 0.8, "d_misclassifies_fake_penalty": 1.2,
            "d_loss_stability": 0.15, "d_very_weak_penalty": 1.0, "gan_balance_g_bonus": 0.4,
            "gan_balance_d_penalty": 0.4, "extreme_gan_imbalance_penalty_g": 1.2,
            "g_stagnation_penalty_for_d": 0.3, "oscillation_penalty": 0.3,
            "extreme_loss_penalty": 1.0, "q_learner_stagnation_penalty_trigger": -0.3,
            "q_learner_stagnation_penalty": 0.25, "lambda_kl_recon_focus": 1.8,
            "lambda_kl_kl_target_range_low": 15.0, "lambda_kl_kl_target_range_high": 80.0,
            "lambda_kl_kl_target_range_bonus": 1.2, "lambda_kl_val_metric_improvement": 2.2,
            "lambda_kl_stability_penalty": 0.6, "lambda_kl_too_high_recon_bad_penalty": 0.7,
            "g_stagnation_adv_high_thresh": 1.8, "g_stagnation_d_strong_thresh": 0.2, # New from prev turn
            # Total Strategy additions
            "geometric_diversity": 0.05, "action_thrashing_penalty": -0.1,
            "q_table_size_penalty_factor": -0.001, "q_value_magnitude_bonus_factor": 0.01
        }
        self.reward_weights = default_rw_template.copy()
        # Override with any reward_weights provided in self.config
        if "reward_weights" in self.config and isinstance(self.config["reward_weights"], dict):
            self.reward_weights.update(self.config["reward_weights"])


        self.num_probation_steps = int(self.config.get("num_probation_steps", self.state_history_len + 3))
        self.current_probation_step = 0; self.on_probation = False
        
        # lkl_num_probation_steps from config should be for LKL-specific action probation
        self.lkl_num_probation_steps = int(self.config.get("lkl_num_probation_steps", max(3, self.lambda_kl_state_history_len + 2)))
        self.lkl_current_probation_step = 0; self.lkl_on_probation = False

        self.logger = logging.getLogger(f"{logger.name}.QController.{associated_component_name}.{id(self)}")
        if HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD:
             self.logger.warning(f"Global FLUSH_Q_TABLES ON. Q-table will be cleared on load if applicable for this instance.")
        self.logger.info(f"Initialized for '{associated_component_name}'. Eps: {self.epsilon_start:.2f}->{self.epsilon_min:.2f}. LR/Mom Probation: {self.num_probation_steps} steps. LKL-Action Probation (if applicable): {self.lkl_num_probation_steps} steps.")
        
        self._internal_step_counter = 0
        self.original_epsilon_before_boost: Optional[float] = None
        self.epsilon_boost_active_steps: int = 0
        self.boosted_epsilon_value: Optional[float] = None

        self.alpha_min = float(self.config.get("alpha_min_meta_q", 0.001))
        self.alpha_max = float(self.config.get("alpha_max_meta_q", 0.1))
        self.alpha_adapt_rate = float(self.config.get("alpha_adapt_rate_meta_q", 0.0001))
        self.gamma_min = float(self.config.get("gamma_min_meta_q", 0.85))
        self.gamma_max = float(self.config.get("gamma_max_meta_q", 0.99))
        self.gamma_adapt_rate = float(self.config.get("gamma_adapt_rate_meta_q", 0.0001))
        self.long_term_reward_avg_hist: deque = deque(maxlen=max(50, self.reward_hist.maxlen // 2))


    def start_probation(self):
        # This method applies to the primary (LR/Mom) probation of this Q-controller instance.
        # If this instance is *also* the LKL controller, its LKL probation might be started separately
        # or tied to this, depending on design. The current structure implies distinct probation types.
        if not self.on_probation: # Primary LR/Mom probation
            self.logger.info(f"Q-Ctrl ({self.logger.name}, LR/Mom Aspect) entering probation ({self.num_probation_steps} steps).")
            self.on_probation = True; self.current_probation_step = 0
        
        # If this controller instance is also used for Lambda_KL actions (e.g. if it's the LKL Q-Ctrl)
        # then `lkl_on_probation` would be relevant for those specific actions.
        # Let's assume `start_probation` is generic and can trigger LKL aspect if applicable.
        # Typically, the HybridTrainer would call start_probation on the *specific* LKL Q-controller.
        # For a generic Q-controller, this refers to its main action probation.
        # If this is the LKL Q-controller, `lkl_on_probation` applies to `lambda_kl_scale` actions.
        if "LKL" in self.logger.name or "LambdaKL" in self.logger.name: # Heuristic to check if this is LKL Q-Ctrl
            if not self.lkl_on_probation:
                self.logger.info(f"Q-Ctrl ({self.logger.name}, LambdaKL Action Aspect) entering probation ({self.lkl_num_probation_steps} steps).")
                self.lkl_on_probation = True; self.lkl_current_probation_step = 0


    def _tick_probation_lr_mom(self):
        if self.on_probation:
            self.current_probation_step += 1
            if self.current_probation_step >= self.num_probation_steps:
                self.logger.info(f"Q-Ctrl ({self.logger.name}, LR/Mom Aspect) probation ended after {self.current_probation_step} steps.")
                self.on_probation = False; self.current_probation_step = 0
    
    def _tick_probation_lkl(self): # Applies if this controller handles LKL actions
        if self.lkl_on_probation:
            self.lkl_current_probation_step += 1
            if self.lkl_current_probation_step >= self.lkl_num_probation_steps:
                self.logger.info(f"Q-Ctrl ({self.logger.name}, LambdaKL Action Aspect) probation ended after {self.lkl_current_probation_step} steps.")
                self.lkl_on_probation = False; self.lkl_current_probation_step = 0

    def force_exploration_boost(self, duration_steps: int = 5, boost_epsilon_to: float = 0.6):
        # Check primary probation first. If specific actions (like LKL) have their own probation,
        # this boost should ideally only apply if *all* relevant probations for the controller are off.
        # For now, simplified: if `on_probation` (main actions) is true, no boost.
        if self.on_probation : 
            self.logger.debug(f"Q-Ctrl ({self.logger.name}): Exploration boost requested but controller (main aspect) is on probation. Ignoring.")
            return
        
        if self.epsilon_boost_active_steps > 0: 
             self.epsilon_boost_active_steps = max(self.epsilon_boost_active_steps, duration_steps) 
             self.logger.info(f"Q-Ctrl ({self.logger.name}): Exploration boost extended to {self.epsilon_boost_active_steps} total steps.")
        else: 
            self.original_epsilon_before_boost = self.epsilon
            self.boosted_epsilon_value = max(self.epsilon, boost_epsilon_to) 
            self.epsilon = self.boosted_epsilon_value # type: ignore # Mypy might complain about float vs Optional[float]
            self.epsilon_boost_active_steps = duration_steps
            self.logger.info(f"Q-Ctrl ({self.logger.name}): Exploration boost ACTIVATED. Epsilon: {self.epsilon:.3f} for {duration_steps} steps.")

    def _tick_exploration_boost(self):
        if hasattr(self, 'epsilon_boost_active_steps') and self.epsilon_boost_active_steps > 0:
            self.epsilon_boost_active_steps -= 1
            if self.epsilon_boost_active_steps == 0:
                if self.original_epsilon_before_boost is not None:
                    self.epsilon = self.original_epsilon_before_boost
                    self.logger.info(f"Q-Ctrl ({self.logger.name}): Exploration boost ENDED. Epsilon restored to {self.epsilon:.3f}.")
                else: 
                    self.logger.error(f"Q-Ctrl ({self.logger.name}): Exploration boost ended, but original_epsilon was None!")
                    self.epsilon = self.epsilon_start 
                self.original_epsilon_before_boost = None
                self.boosted_epsilon_value = None

    def reset_q_learning_state(self, reset_q_table: bool = True, reset_epsilon: bool = True,
                               context_msg: str = "Q-Ctrl Reset", start_probation: bool = False):
        self.logger.info(f"{context_msg} ({self.logger.name}): Resetting Q-Controller state. Reset Q-table: {reset_q_table}, Reset Epsilon: {reset_epsilon}, Start Probation: {start_probation}")
        history_deques_to_clear: List[deque] = [
            self.loss_g_total_hist, self.loss_g_recon_hist, self.loss_g_kl_hist,
            self.loss_g_adv_hist, self.loss_d_total_hist, self.loss_d_real_hist,
            self.loss_d_fake_hist, self.interval_avg_recon_hist, self.interval_avg_kl_div_hist,
            self.interval_avg_d_total_hist, self.interval_val_metric_hist, self.reward_hist,
            self.wubu_avg_curvature_hist, self.wubu_avg_scale_hist, self.wubu_avg_spread_hist,
            self.wubu_var_curvature_hist, self.long_term_reward_avg_hist
        ]
        for deq_to_clear in history_deques_to_clear: deq_to_clear.clear()
        
        self.prev_lr_mom_state = None; self.prev_lr_mom_action = None
        self.prev_lambda_kl_state = None; self.prev_lambda_kl_action = None
        self.prev_heuristic_toggle_state = None; self.prev_heuristic_toggle_action = None
        
        if reset_epsilon: 
            self.epsilon = self.epsilon_start
            self.logger.info(f"{context_msg} ({self.logger.name}): Epsilon reset to {self.epsilon_start:.2f}")
        
        if reset_q_table:
            self.logger.info(f"{context_msg} ({self.logger.name}): Clearing Q-table and related stats.")
            self.q_table.clear(); self.q_table_access_count.clear()
            self.q_table_creation_time.clear(); self.q_table_last_access_time.clear()
        
        if start_probation: self.start_probation() # This will trigger both LR/Mom and LKL probation if applicable
        else: # Explicitly turn off if not starting probation
            self.on_probation = False; self.current_probation_step = 0
            self.lkl_on_probation = False; self.lkl_current_probation_step = 0
            
        self._internal_step_counter = 0
        self.epsilon_boost_active_steps = 0
        self.current_alpha = self.base_alpha # Reset meta-adaptive params too
        self.current_gamma = self.base_gamma


    def _get_trend_bin(self, history: deque, current_val: Optional[float], 
                       relative_to_median: bool = True, value_scale_for_diff:float = 1.0,
                       thresholds: List[float] = [-0.15, -0.02, 0.02, 0.15]) -> int:
        if current_val is None or not np.isfinite(current_val): return (len(thresholds) + 1) // 2
        valid_history = [h for h in history if h is not None and np.isfinite(h)]
        if not valid_history: return (len(thresholds) + 1) // 2

        # Use median of all but the last element if history is long enough, else median of all
        if len(valid_history) > self.state_history_len // 2 + 1 and len(valid_history) > 1:
            prev_ref = np.median(valid_history[:-1]) 
        else:
            prev_ref = np.median(valid_history)
        
        diff = current_val - prev_ref
        if relative_to_median:
            # More robust denominator to avoid division by zero or tiny numbers
            denominator_val = max(abs(prev_ref), abs(current_val), value_scale_for_diff * 0.01) + EPS 
            relative_diff = diff / denominator_val
        else: # Absolute difference scaled
            relative_diff = diff / (value_scale_for_diff + EPS)

        for i, th in enumerate(thresholds):
            if relative_diff < th: return i
        return len(thresholds) # Index for "greater than last threshold"

    def _update_loss_histories(self, current_losses: Dict[str, float], wubu_geo_params: Optional[Dict[str,float]] = None):
        loss_map_update = { 
            'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist 
        }
        for name_update, deq_update in loss_map_update.items():
            loss_val_update = current_losses.get(name_update)
            if loss_val_update is not None and np.isfinite(loss_val_update): deq_update.append(loss_val_update)

        if wubu_geo_params:
            geo_map_update = {
                'avg_curvature': self.wubu_avg_curvature_hist, 'avg_scale': self.wubu_avg_scale_hist,
                'avg_spread': self.wubu_avg_spread_hist, 'var_curvature': self.wubu_var_curvature_hist
            }
            for name_geo_update, deq_geo_update in geo_map_update.items():
                val_geo_update = wubu_geo_params.get(name_geo_update)
                if val_geo_update is not None and np.isfinite(val_geo_update): deq_geo_update.append(val_geo_update)
    
    def get_lr_mom_state(self, current_losses: Dict[str, float], current_lr: float,
                         current_momentum: float, is_generator_q: bool,
                         wubu_geo_params: Optional[Dict[str,float]] = None
                         ) -> Optional[tuple]:
        self._internal_step_counter +=1
        self._update_loss_histories(current_losses, wubu_geo_params)

        req_keys_g_state = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        req_keys_d_state = ['loss_d_total', 'loss_g_total', 'loss_d_real', 'loss_d_fake', 'loss_g_adv']
        required_keys_state = req_keys_g_state if is_generator_q else req_keys_d_state
        
        if not all(key_state in current_losses and np.isfinite(current_losses[key_state]) for key_state in required_keys_state):
            self.logger.debug(f"LR/Mom QState ({self.logger.name}): Insufficient/non-finite. Need: {required_keys_state}.")
            return None
        if not (np.isfinite(current_lr) and np.isfinite(current_momentum)):
            self.logger.debug(f"LR/Mom QState ({self.logger.name}): Non-finite LR/Mom.")
            return None

        s_wubu_c_trend_state, s_wubu_s_trend_state, s_wubu_var_c_level_state = 2, 2, 2 
        if wubu_geo_params and is_generator_q:
            s_wubu_c_trend_state = self._get_trend_bin(self.wubu_avg_curvature_hist, wubu_geo_params.get('avg_curvature'), value_scale_for_diff=0.2)
            s_wubu_s_trend_state = self._get_trend_bin(self.wubu_avg_scale_hist, wubu_geo_params.get('avg_scale'), value_scale_for_diff=0.1)
            s_wubu_var_c_level_state = np.digitize(wubu_geo_params.get('var_curvature', 0.0), [0.01, 0.1, 0.5]).item()
        
        q_table_size_bin_state = np.digitize(len(self.q_table), [self.max_q_table_size * 0.25, self.max_q_table_size * 0.75]).item()
        avg_q_mag_val_state = 0.0
        if self.q_table:
            all_q_vals_state = [q_val for state_actions in self.q_table.values() for q_vals_for_param in state_actions.values() for q_val in q_vals_for_param if np.isfinite(q_val)]
            if all_q_vals_state: avg_q_mag_val_state = np.mean(np.abs(all_q_vals_state))
        q_value_mag_bin_state = np.digitize(avg_q_mag_val_state, [1.0, 5.0, 15.0]).item()

        if is_generator_q:
            s_g_total_trend_state = self._get_trend_bin(self.loss_g_total_hist, current_losses['loss_g_total'])
            s_d_total_level_opp_state = np.digitize(current_losses['loss_d_total'], [0.15, 0.4, 0.7, 1.2]).item()
            s_g_recon_trend_state = self._get_trend_bin(self.loss_g_recon_hist, current_losses['loss_g_recon'])
            s_g_recon_level_state = np.digitize(current_losses['loss_g_recon'], [0.02, 0.08, 0.2, 0.5]).item()
            kl_val_state, recon_val_state = current_losses['loss_g_kl'], current_losses['loss_g_recon']
            s_kl_problem_state = 0 
            rw_state = self.reward_weights
            if (self.current_lambda_kl * kl_val_state > rw_state.get("g_kl_control_penalty_ratio_trigger", 0.75) * recon_val_state and
                recon_val_state > 0.03 and self.current_lambda_kl > 1e-5): s_kl_problem_state = 1
            elif kl_val_state > rw_state.get("g_kl_control_penalty_abs_trigger_high", 100.0): s_kl_problem_state = 2
            s_g_adv_level_state = np.digitize(current_losses['loss_g_adv'], [0.05, 0.2, 0.6, 1.5]).item()
            s_lr_bin_state = np.digitize(current_lr, [5e-6, 2e-5, 1e-4, 5e-4]).item()
            s_mom_bin_state = np.digitize(current_momentum, [0.8, 0.9, 0.97]).item()
            eps_bin_state = np.digitize(self.epsilon, [self.epsilon_min * 1.2, self.epsilon_start * 0.3, self.epsilon_start * 0.7]).item()

            state_tuple_final = ("LRM_G", s_g_total_trend_state, s_d_total_level_opp_state, s_g_recon_trend_state, s_g_recon_level_state,
                           s_kl_problem_state, s_g_adv_level_state, s_lr_bin_state, s_mom_bin_state, eps_bin_state,
                           s_wubu_c_trend_state, s_wubu_s_trend_state, s_wubu_var_c_level_state, 
                           q_table_size_bin_state, q_value_mag_bin_state)
        else: 
            s_d_total_trend_state = self._get_trend_bin(self.loss_d_total_hist, current_losses['loss_d_total'])
            s_g_adv_level_opp_state = np.digitize(current_losses.get('loss_g_adv', 0.7), [0.05, 0.2, 0.6, 1.5]).item()
            s_d_total_level_state = np.digitize(current_losses['loss_d_total'], [0.1, 0.3, 0.7, 1.2, 2.0]).item()
            d_fake_val_state = current_losses['loss_d_fake']; d_real_val_state = current_losses['loss_d_real']
            s_d_fake_level_state = np.digitize(d_fake_val_state, [0.1, 0.5, 1.0, 2.0]).item()
            s_d_real_level_state = np.digitize(d_real_val_state, [0.05, 0.2, 0.5, 0.8]).item()
            s_lr_bin_state = np.digitize(current_lr, [5e-6, 2e-5, 1e-4, 5e-4]).item()
            s_mom_bin_state = np.digitize(current_momentum, [0.8, 0.9, 0.97]).item()
            eps_bin_state = np.digitize(self.epsilon, [self.epsilon_min * 1.2, self.epsilon_start * 0.3, self.epsilon_start * 0.7]).item()

            state_tuple_final = ("LRM_D", s_d_total_trend_state, s_g_adv_level_opp_state, s_d_total_level_state,
                           s_d_fake_level_state, s_d_real_level_state, s_lr_bin_state, s_mom_bin_state, eps_bin_state,
                           q_table_size_bin_state, q_value_mag_bin_state)

        self._ensure_q_state_exists(state_tuple_final)
        return state_tuple_final

    def get_lambda_kl_state(self, interval_metrics: Dict[str, Optional[float]]) -> Optional[tuple]:
        required_keys_lkl = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric', 'current_lambda_kl_val']
        valid_metrics_lkl = True; current_metrics_for_hist_lkl: Dict[str, float] = {}
        for key_lkl in required_keys_lkl:
            val_lkl = interval_metrics.get(key_lkl)
            if val_lkl is None or not np.isfinite(val_lkl): valid_metrics_lkl = False; break
            current_metrics_for_hist_lkl[key_lkl] = float(val_lkl)
        if not valid_metrics_lkl: return None

        self.interval_avg_recon_hist.append(current_metrics_for_hist_lkl['avg_recon'])
        self.interval_avg_kl_div_hist.append(current_metrics_for_hist_lkl['avg_kl_div'])
        self.interval_avg_d_total_hist.append(current_metrics_for_hist_lkl['avg_d_total'])
        if current_metrics_for_hist_lkl['val_metric'] is not None and np.isfinite(current_metrics_for_hist_lkl['val_metric']):
             self.interval_val_metric_hist.append(current_metrics_for_hist_lkl['val_metric'])

        s_interval_recon_trend_lkl = self._get_trend_bin(self.interval_avg_recon_hist, current_metrics_for_hist_lkl['avg_recon'], value_scale_for_diff=0.1)
        s_interval_kl_trend_lkl = self._get_trend_bin(self.interval_avg_kl_div_hist, current_metrics_for_hist_lkl['avg_kl_div'], value_scale_for_diff=5.0) 
        s_interval_val_metric_trend_lkl = self._get_trend_bin(self.interval_val_metric_hist, current_metrics_for_hist_lkl['val_metric'], value_scale_for_diff=0.05) 
        
        s_current_lambda_kl_bin_lkl = np.digitize(current_metrics_for_hist_lkl['current_lambda_kl_val'], [1e-5, 1e-4, 0.001, 0.01, 0.1]).item() 
        s_interval_d_balance_level_lkl = np.digitize(current_metrics_for_hist_lkl['avg_d_total'], [0.15, 0.4, 0.7, 1.2]).item() 
        eps_bin_lkl = np.digitize(self.epsilon, [self.epsilon_min * 1.2, self.epsilon_start * 0.3, self.epsilon_start * 0.7]).item()
        
        q_table_size_bin_lkl_state = np.digitize(len(self.q_table), [self.max_q_table_size * 0.25, self.max_q_table_size * 0.75]).item()
        avg_q_mag_val_lkl_state = 0.0
        if self.q_table:
            all_q_vals_lkl_state = [q_val for state_actions in self.q_table.values() for q_vals_for_param in state_actions.values() for q_val in q_vals_for_param if np.isfinite(q_val)]
            if all_q_vals_lkl_state: avg_q_mag_val_lkl_state = np.mean(np.abs(all_q_vals_lkl_state))
        q_value_mag_bin_lkl_state = np.digitize(avg_q_mag_val_lkl_state, [1.0, 5.0, 15.0]).item()

        state_tuple_lkl_final = ("LKL", s_interval_recon_trend_lkl, s_interval_kl_trend_lkl, s_interval_val_metric_trend_lkl,
                       s_current_lambda_kl_bin_lkl, s_interval_d_balance_level_lkl, eps_bin_lkl,
                       q_table_size_bin_lkl_state, q_value_mag_bin_lkl_state)
        self._ensure_q_state_exists(state_tuple_lkl_final)
        return state_tuple_lkl_final

    def _ensure_q_state_exists(self, state_tuple: tuple):
        current_time = time.time()
        self.q_table_access_count[state_tuple] += 1
        self.q_table_last_access_time[state_tuple] = current_time
        if state_tuple not in self.q_table:
            # Determine relevant action types for this state type
            # Example: LKL states only need 'lambda_kl_scale' actions
            # LRM_G states need 'lr_scale', 'momentum_scale', 'heuristic_toggle_generic'
            # LRM_D states need 'lr_scale', 'momentum_scale'
            relevant_action_types_for_state: List[str] = []
            if state_tuple[0] == "LRM_G":
                relevant_action_types_for_state = ['lr_scale', 'momentum_scale', 'heuristic_toggle_generic']
                # Add 'optimizer_type_switch' if implemented and G controls it
                # if 'optimizer_type_switch' in self.action_ranges:
                #     relevant_action_types_for_state.append('optimizer_type_switch')
            elif state_tuple[0] == "LRM_D":
                relevant_action_types_for_state = ['lr_scale', 'momentum_scale']
            elif state_tuple[0] == "LKL":
                relevant_action_types_for_state = ['lambda_kl_scale']
            else: # Fallback: initialize all known action types (original behavior)
                relevant_action_types_for_state = list(self.action_ranges.keys())

            self.q_table[state_tuple] = {}
            for p_type in relevant_action_types_for_state:
                if p_type in self.num_actions: # Ensure this action type is defined
                    self.q_table[state_tuple][p_type] = np.zeros(self.num_actions[p_type], dtype=np.float32)
                else:
                    self.logger.warning(f"Action type '{p_type}' for state {state_tuple[0]} not in self.num_actions. Q-value init skipped for this type.")
            
            self.q_table_creation_time[state_tuple] = current_time
            self._manage_q_table_size() 

    def choose_action(self, state: Optional[tuple], mode: str = 'lr_mom') -> Dict[str, Any]:
        self._tick_exploration_boost()
        default_actions: Dict[str, Any] = {
            'lr_scale': 1.0, 'momentum_scale': 1.0, 'lambda_kl_scale': 1.0,
            'heuristic_toggle_generic': 0.0, # Default to 'off' or 'no change'
            'optimizer_type_switch': self.action_ranges['optimizer_type_switch'][0] if 'optimizer_type_switch' in self.action_ranges and len(self.action_ranges['optimizer_type_switch']) > 0 else "default"
        }
        action_types_to_choose_for_mode: List[str] = []
        chosen_actions_final: Dict[str, Any] = {}

        if mode == 'lr_mom':
            self._tick_probation_lr_mom() # Ticks primary probation
            action_types_to_choose_for_mode = ['lr_scale', 'momentum_scale']
            if state and state[0] == "LRM_G": # Generator's Q-controller
                 if 'heuristic_toggle_generic' in self.action_ranges: action_types_to_choose_for_mode.append('heuristic_toggle_generic')
                 # if 'optimizer_type_switch' in self.action_ranges: action_types_to_choose_for_mode.append('optimizer_type_switch')
            if self.on_probation: # Primary probation
                chosen_actions_final = {k: default_actions[k] for k in action_types_to_choose_for_mode}
                return chosen_actions_final
        elif mode == 'lambda_kl':
            self._tick_probation_lkl() # Ticks LKL-action probation
            action_types_to_choose_for_mode = ['lambda_kl_scale']
            if self.lkl_on_probation: # LKL-action probation
                chosen_actions_final = {'lambda_kl_scale': 1.0}
                return chosen_actions_final
        else: raise ValueError(f"Invalid mode for choose_action: {mode}")

        if state is None or state not in self.q_table:
            self.logger.warning(f"Q-Ctrl ({self.logger.name}, Mode {mode}): State None or not in Q-table. Using default actions for this mode.")
            chosen_actions_final = {k: default_actions[k] for k in action_types_to_choose_for_mode}
            return chosen_actions_final

        if not (hasattr(self, 'epsilon_boost_active_steps') and self.epsilon_boost_active_steps > 0):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        for param_type_choice in action_types_to_choose_for_mode:
            if param_type_choice not in self.q_table[state]: 
                self.logger.warning(f"State {state}, PType {param_type_choice} ({self.logger.name}): Q-values missing for this param_type. Choosing default.")
                chosen_actions_final[param_type_choice] = default_actions[param_type_choice]
                continue

            q_values_arr_choice = self.q_table[state].get(param_type_choice) 
            action_space_arr_choice = self.action_ranges[param_type_choice]  
            if q_values_arr_choice is None or not isinstance(q_values_arr_choice, np.ndarray) or len(q_values_arr_choice) != len(action_space_arr_choice):
                self.logger.error(f"Q-values for {param_type_choice} malformed in state {state}. Q_vals: {q_values_arr_choice}, ActionSpaceLen: {len(action_space_arr_choice)}. Choosing default.")
                chosen_actions_final[param_type_choice] = default_actions[param_type_choice]
                continue
            
            if random.random() < self.epsilon: 
                chosen_idx_choice = random.randrange(len(action_space_arr_choice))
            else:
                finite_q_indices_choice = np.where(np.isfinite(q_values_arr_choice))[0]
                if finite_q_indices_choice.size > 0:
                    best_q_val_among_finite_choice = np.max(q_values_arr_choice[finite_q_indices_choice])
                    best_indices_options_choice = np.where(
                        np.isclose(q_values_arr_choice, best_q_val_among_finite_choice) & np.isfinite(q_values_arr_choice)
                    )[0]
                    chosen_idx_choice = random.choice(best_indices_options_choice) if best_indices_options_choice.size > 0 else random.randrange(len(action_space_arr_choice))
                else: 
                    chosen_idx_choice = random.randrange(len(action_space_arr_choice))
                    self.logger.warning(f"State {state}, PType {param_type_choice} ({self.logger.name}): All Q-vals non-finite. Random action.")
            
            chosen_action_val_final = action_space_arr_choice[chosen_idx_choice]
            if isinstance(chosen_action_val_final, np.floating): chosen_actions_final[param_type_choice] = float(chosen_action_val_final)
            else: chosen_actions_final[param_type_choice] = chosen_action_val_final
        return chosen_actions_final

    def _adapt_q_learning_params(self):
        if not self.reward_hist: return
        # Add current short-term avg to long-term history (use median for robustness)
        current_short_term_reward_median = np.median(list(self.reward_hist)) if self.reward_hist else 0.0
        self.long_term_reward_avg_hist.append(current_short_term_reward_median)
        
        if len(self.long_term_reward_avg_hist) < self.long_term_reward_avg_hist.maxlen // 2: return 

        long_term_median_reward = np.median(list(self.long_term_reward_avg_hist))
        
        # Adapt alpha
        if long_term_median_reward > self.reward_weights.get("q_learner_stagnation_penalty_trigger", -0.3) + 0.2: # If consistently better than stagnation trigger
            self.current_alpha = max(self.alpha_min, self.current_alpha - self.alpha_adapt_rate)
        elif long_term_median_reward < self.reward_weights.get("q_learner_stagnation_penalty_trigger", -0.3) - 0.1: # If consistently worse
            self.current_alpha = min(self.alpha_max, self.current_alpha + self.alpha_adapt_rate)
            
        # Adapt gamma
        reward_variance_short_term = np.var(list(self.reward_hist)) if len(self.reward_hist) > 1 else 0.0
        if long_term_median_reward > 0.1 and reward_variance_short_term < 0.75: # Relatively stable positive rewards
             self.current_gamma = min(self.gamma_max, self.current_gamma + self.gamma_adapt_rate)
        elif long_term_median_reward < -0.1 or reward_variance_short_term > 1.25: # Unstable or negative rewards
             self.current_gamma = max(self.gamma_min, self.current_gamma - self.gamma_adapt_rate)


    def update_q_values(self, state: tuple, action: Dict[str, Any], reward: float,
                        next_state: Optional[tuple], mode: str = 'lr_mom'):
        if state not in self.q_table:
            self.logger.warning(f"Updating Q for non-existent state ({self.logger.name}): {state}. Ensuring state exists.")
            self._ensure_q_state_exists(state)
            if state not in self.q_table: 
                self.logger.error(f"Failed to ensure state {state} for Q-update ({self.logger.name}). Update aborted for this step."); return
        
        if self.reward_clipping: reward = np.clip(reward, self.reward_clipping[0], self.reward_clipping[1])
        self.reward_hist.append(reward)
        self._adapt_q_learning_params()

        for param_type_update, chosen_value_update in action.items():
            if param_type_update not in self.action_ranges or param_type_update not in self.q_table[state]: 
                self.logger.debug(f"Q-Update ({self.logger.name}): param_type '{param_type_update}' not in action_ranges or Q-table for state. Skipping.")
                continue
            
            # Find index of chosen_value_update in self.action_ranges[param_type_update]
            action_space_for_type = self.action_ranges[param_type_update]
            try:
                # Handle string actions (like optimizer_type_switch) vs numeric actions
                if isinstance(action_space_for_type[0], str):
                    action_idx_arr_update = np.where(action_space_for_type == str(chosen_value_update))[0]
                else: # Assume numeric
                    action_idx_arr_update = np.where(np.isclose(action_space_for_type.astype(float), float(chosen_value_update)))[0]
            except Exception as e_idx:
                self.logger.error(f"Error finding action_idx for PType {param_type_update}, Val {chosen_value_update}: {e_idx}. Skipping.")
                continue

            if not action_idx_arr_update.size:
                self.logger.warning(f"Q-Update ({self.logger.name}): Action value '{chosen_value_update}' for type '{param_type_update}' not found in action_ranges. Values: {action_space_for_type}. Skipping.")
                continue
            action_idx_update = action_idx_arr_update[0]
            
            current_q_update = self.q_table[state][param_type_update][action_idx_update]
            max_future_q_update = 0.0
            if next_state and next_state in self.q_table and param_type_update in self.q_table[next_state]:
                next_q_vals_update = self.q_table[next_state][param_type_update]
                if np.any(np.isfinite(next_q_vals_update)): max_future_q_update = np.nanmax(next_q_vals_update[np.isfinite(next_q_vals_update)])
            
            td_target_update = reward + self.current_gamma * max_future_q_update
            new_q_update = current_q_update + self.current_alpha * (td_target_update - current_q_update)
            
            if np.isfinite(new_q_update):
                if self.q_value_clipping: new_q_update = np.clip(new_q_update, self.q_value_clipping[0], self.q_value_clipping[1])
                self.q_table[state][param_type_update][action_idx_update] = new_q_update
            # else: # If new_q is not finite, do not update (keeps old Q-value)
            #     self.logger.warning(f"Q-Update ({self.logger.name}): New Q-value for state {state}, action_type {param_type_update}, idx {action_idx_update} was non-finite. Q-value not updated.")

    def _manage_q_table_size(self):
        if len(self.q_table) <= self.max_q_table_size: return
        num_to_prune_final = len(self.q_table) - self.max_q_table_size
        current_time_final = time.time()
        
        # Enhanced scoring: favor states that are older (more time to learn) but also recently accessed.
        # Higher score is better to keep.
        state_scores_final: Dict[tuple, float] = {}
        for s_tuple_score in self.q_table.keys():
            access_count = self.q_table_access_count.get(s_tuple_score, 1)
            # Age: time since creation (normalize by hour, log scale)
            age_hours = (current_time_final - self.q_table_creation_time.get(s_tuple_score, current_time_final)) / 3600.0
            age_score_factor = 1.0 + np.log1p(age_hours) # Favor older states
            
            # Recency: time since last access (normalize by 10 mins, log scale, inverse relationship)
            inactivity_minutes = (current_time_final - self.q_table_last_access_time.get(s_tuple_score, current_time_final)) / 600.0
            recency_score_factor = 1.0 / (1.0 + np.log1p(inactivity_minutes) * 0.5) # Penalize inactivity, less aggressively
            
            # Optional: Q-value magnitude contribution (states with stronger Q-values might be more important)
            # q_mag_factor = 1.0
            # if s_tuple_score in self.q_table:
            #     all_q_vals_for_state = [q for q_actions in self.q_table[s_tuple_score].values() for q in q_actions if np.isfinite(q)]
            #     if all_q_vals_for_state:
            #         avg_abs_q = np.mean(np.abs(all_q_vals_for_state))
            #         q_mag_factor = 1.0 + np.log1p(avg_abs_q / 5.0) # Slight bonus for higher avg magnitude (e.g. >5)

            state_scores_final[s_tuple_score] = access_count * age_score_factor * recency_score_factor # * q_mag_factor
            
        sorted_states_for_pruning_final = sorted(state_scores_final.keys(), key=lambda s: state_scores_final[s])
        pruned_count_final = 0
        for i in range(num_to_prune_final):
            if i < len(sorted_states_for_pruning_final):
                s_rm_final = sorted_states_for_pruning_final[i]
                self.q_table.pop(s_rm_final, None); self.q_table_access_count.pop(s_rm_final, None)
                self.q_table_creation_time.pop(s_rm_final, None); self.q_table_last_access_time.pop(s_rm_final, None)
                pruned_count_final +=1
        if pruned_count_final > 0: self.logger.info(f"Pruned {pruned_count_final} Q-table entries. New size: {len(self.q_table)}.")


    def compute_lr_mom_reward(self, current_losses: Dict[str, float], is_generator_q: bool,
                              wubu_geo_params: Optional[Dict[str,float]] = None, 
                              q_table_stats: Optional[Dict[str,float]] = None 
                              ) -> float:
        # This method needs to be fully implemented using self.reward_weights
        # and considering all new state components (wubu_geo_params, q_table_stats)
        # For brevity, the detailed logic from the previous turn is assumed here.
        # The key is that it accesses self.reward_weights for specific term weights.
        
        # Simplified structure:
        total_reward_val = 0.0
        rw = self.reward_weights # Use the instance's reward_weights
        
        # --- Original Loss-based Rewards (Copied from previous turn, ensure rw keys exist) ---
        losses_to_use_reward = {k: (100.0 if not np.isfinite(v) else np.clip(v, -500, 500)) 
                                for k,v in current_losses.items()}
        if any(not np.isfinite(v) for v in current_losses.values()):
            total_reward_val -= rw.get("extreme_loss_penalty", 1.0) * 2.5

        def get_prev_median_reward(hist_deque: deque, current_val_fallback: float) -> float:
            valid_hist = [v for v in hist_deque if v is not None and np.isfinite(v)]
            if not valid_hist: return current_val_fallback
            if len(valid_hist) > self.state_history_len // 2 + 1 and len(valid_hist) > 1 :
                 return np.median(valid_hist[:-1]) if len(valid_hist) > 1 else valid_hist[0]
            return np.median(valid_hist) if valid_hist else current_val_fallback

        if is_generator_q:
            # G Recon Improvement
            loss_g_recon_reward = losses_to_use_reward.get('loss_g_recon', 1.0)
            prev_g_recon_reward = get_prev_median_reward(self.loss_g_recon_hist, loss_g_recon_reward)
            recon_improvement_reward = prev_g_recon_reward - loss_g_recon_reward
            recon_scale_reward = 1.0 / (loss_g_recon_reward + 0.01) 
            total_reward_val += np.tanh(recon_improvement_reward * recon_scale_reward * 15.0) * rw.get("g_recon_improvement", 3.0)
            # G Adv Improvement
            loss_g_adv_reward = losses_to_use_reward.get('loss_g_adv', 0.7)
            prev_g_adv_reward = get_prev_median_reward(self.loss_g_adv_hist, loss_g_adv_reward)
            adv_improvement_reward = prev_g_adv_reward - loss_g_adv_reward
            total_reward_val += np.tanh(adv_improvement_reward / (abs(prev_g_adv_reward) + 0.05 + EPS)) * rw.get("g_adv_improvement", 1.5)
            # KL Control Penalty
            loss_g_kl_reward = losses_to_use_reward.get('loss_g_kl', 0.0)
            if (self.current_lambda_kl * loss_g_kl_reward > rw.get("g_kl_control_penalty_ratio_trigger", 0.75) * loss_g_recon_reward and
                loss_g_recon_reward > 0.03 and loss_g_kl_reward > rw.get("g_kl_control_penalty_abs_trigger_low", 30.0)):
                total_reward_val -= rw.get("g_kl_control_penalty", 0.4) * (1 + min(1.0, (loss_g_kl_reward - rw.get("g_kl_control_penalty_abs_trigger_low", 30.0)) / 100.0))
            # GAN Balance (G perspective)
            loss_d_total_opp_reward = losses_to_use_reward.get('loss_d_total', 0.7)
            if 0.2 < loss_d_total_opp_reward < 0.8: total_reward_val += rw.get("gan_balance_g_bonus", 0.4)
            elif loss_d_total_opp_reward <= 0.1: total_reward_val -= rw.get("extreme_gan_imbalance_penalty_g", 1.2) * 2.0
            # G Easy Win Penalty
            if loss_g_adv_reward < rw.get("g_easy_win_adv_thresh", 0.1) and loss_g_recon_reward > rw.get("g_easy_win_recon_thresh", 0.15):
                total_reward_val -= rw.get("g_easy_win_recon_bad_penalty", 0.75) * (loss_g_recon_reward / (rw.get("g_easy_win_recon_thresh", 0.15)+EPS))
        else: # Discriminator Q
            loss_d_total_reward = losses_to_use_reward.get('loss_d_total', 0.7)
            if 0.2 < loss_d_total_reward < 0.8: total_reward_val += rw.get("d_balance_target", 1.8)
            elif loss_d_total_reward < 0.1: total_reward_val -= rw.get("d_balance_target", 1.8) * 1.5 
            elif loss_d_total_reward > 1.5: total_reward_val -= rw.get("d_very_weak_penalty", 1.0) * (1 + min(1.0, (loss_d_total_reward - 1.5)/1.0))
            # ... other D loss components (d_real, d_fake, GAN balance from D side) ...
            loss_d_real_reward = losses_to_use_reward.get('loss_d_real', 0.7)
            if loss_d_real_reward < 0.15: total_reward_val += rw.get("d_real_low_bonus", 0.8) * 1.5
            loss_d_fake_reward = losses_to_use_reward.get('loss_d_fake', 0.7)
            loss_g_adv_opp_reward = losses_to_use_reward.get('loss_g_adv', 0.7) 
            if loss_d_fake_reward < 0.15 and loss_g_adv_opp_reward > 0.7: 
                 total_reward_val += rw.get("d_fake_low_meaningful_bonus",0.8) * 1.5
            elif loss_d_fake_reward > 2.0 and loss_g_adv_opp_reward < 0.1: 
                total_reward_val -= rw.get("d_misclassifies_fake_penalty", 1.2) * 2.0
            if loss_g_adv_opp_reward < 0.05: total_reward_val -= rw.get("gan_balance_d_penalty", 0.4) * 2.0
            if len(self.loss_g_adv_hist) >= max(3, self.state_history_len-1): 
                g_adv_hist_for_check_reward = list(self.loss_g_adv_hist)[-max(3, self.state_history_len//2):]
                if g_adv_hist_for_check_reward and np.median(g_adv_hist_for_check_reward) > rw.get("g_stagnation_adv_high_thresh", 1.8): 
                    if loss_d_total_reward < rw.get("g_stagnation_d_strong_thresh", 0.2): 
                        total_reward_val -= rw.get("g_stagnation_penalty_for_d", 0.3)
        
        # --- New Reward Components from Total Strategy ---
        if is_generator_q and wubu_geo_params:
            var_c_reward = wubu_geo_params.get('var_curvature', 0.0)
            if 0.05 < var_c_reward < 0.5: total_reward_val += rw.get("geometric_diversity", 0.05) * 0.5
            elif var_c_reward < 0.01 : total_reward_val -= rw.get("geometric_diversity", 0.05) * 0.2

        if q_table_stats:
            q_size_ratio_reward = q_table_stats.get('q_table_size', 0) / (self.max_q_table_size + EPS)
            if q_size_ratio_reward > 0.9:
                 total_reward_val += rw.get("q_table_size_penalty_factor", -0.001) * (q_size_ratio_reward - 0.9) * 10.0
            avg_q_mag_reward = q_table_stats.get('avg_q_value_magnitude', 0.0)
            if avg_q_mag_reward > 10.0:
                total_reward_val += rw.get("q_value_magnitude_bonus_factor", 0.01) * min(1.0, (avg_q_mag_reward - 10.0) / 10.0)

        # --- Oscillation and Stagnation Penalties (Original) ---
        if len(self.reward_hist) >= self.state_history_len: # Use state_history_len for consistency
            recent_q_rewards_osc = list(self.reward_hist)[-max(5, self.state_history_len//2):] 
            if len(recent_q_rewards_osc) > 2 : 
                sign_flips_osc = 0
                for i_osc in range(len(recent_q_rewards_osc) - 1):
                    if (np.sign(recent_q_rewards_osc[i_osc]) != np.sign(recent_q_rewards_osc[i_osc+1]) and
                        abs(recent_q_rewards_osc[i_osc]) > 0.15 and abs(recent_q_rewards_osc[i_osc+1]) > 0.15):
                        sign_flips_osc += 1
                if sign_flips_osc >= (len(recent_q_rewards_osc) // 2) :
                    total_reward_val -= rw.get("oscillation_penalty", 0.3) * (sign_flips_osc / (len(recent_q_rewards_osc) -1 + EPS)) # Add EPS to denom

        # Stagnation penalty based on a longer history of rewards (e.g. last 15 *steps* not epochs)
        # Assuming reward_hist stores rewards per Q-update step.
        if len(self.reward_hist) >= 15: 
            if np.median(list(self.reward_hist)[-15:]) < rw.get("q_learner_stagnation_penalty_trigger", -0.3):
                total_reward_val -= rw.get("q_learner_stagnation_penalty", 0.25)

        if self.reward_clipping:
            total_reward_val = np.clip(total_reward_val, self.reward_clipping[0], self.reward_clipping[1])
        return float(total_reward_val)


    def compute_lambda_kl_reward(self, interval_metrics: Dict[str, Optional[float]],
                                 prev_interval_metrics: Optional[Dict[str, Optional[float]]]) -> float:
        total_reward_lkl = 0.0; rw_lkl = self.reward_weights; prev_metrics_lkl = prev_interval_metrics or {}
        
        required_lkl = ['val_metric', 'avg_recon', 'avg_kl_div', 'avg_d_total', 'current_lambda_kl_val']
        current_finite_metrics_lkl: Dict[str, float] = {}
        for key_lkl_req in required_lkl:
            val_lkl_req = interval_metrics.get(key_lkl_req)
            if val_lkl_req is None or not np.isfinite(val_lkl_req):
                self.logger.warning(f"LKL_Rew ({self.logger.name}): Metric '{key_lkl_req}' missing/non-finite. Reward may be impacted."); return -0.2 
            current_finite_metrics_lkl[key_lkl_req] = float(val_lkl_req)

        # Validation metric improvement
        val_metric_current = current_finite_metrics_lkl['val_metric']
        val_metric_prev = float(prev_metrics_lkl.get('val_metric', val_metric_current)) # Default to current if no prev
        val_metric_improvement = val_metric_prev - val_metric_current # Assumes lower is better for val_metric
        # If higher is better for val_metric, this needs to be flipped or handled based on metric type.
        # For now, assuming lower is better (e.g. LPIPS, MSE).
        total_reward_lkl += np.tanh(val_metric_improvement * 8.0) * rw_lkl.get("lambda_kl_val_metric_improvement",2.2)

        # Reconstruction focus
        recon_current = current_finite_metrics_lkl['avg_recon']
        recon_prev = float(prev_metrics_lkl.get('avg_recon', recon_current))
        recon_improvement_lkl = recon_prev - recon_current # Lower recon loss is better
        recon_penalty_factor_lkl = 1.0 if recon_improvement_lkl >= -0.02 else (1.0 + abs(recon_improvement_lkl * 20))
        total_reward_lkl += np.tanh(recon_improvement_lkl * 15.0 / recon_penalty_factor_lkl) * rw_lkl.get("lambda_kl_recon_focus",1.8)

        # KL divergence target range
        kl_low_target = rw_lkl.get("lambda_kl_kl_target_range_low", 15.0)
        kl_high_target = rw_lkl.get("lambda_kl_kl_target_range_high", 80.0)
        kl_div_current = current_finite_metrics_lkl['avg_kl_div']
        kl_target_bonus = rw_lkl.get("lambda_kl_kl_target_range_bonus",1.2)
        if kl_div_current < kl_low_target and current_finite_metrics_lkl['avg_recon'] > 0.04 : # If KL too low and recon not perfect
            total_reward_lkl -= kl_target_bonus * (1.0 - kl_div_current/(kl_low_target+EPS)) * 0.75 # Penalize being too low
        elif kl_div_current > kl_high_target: # If KL too high
            kl_div_prev = float(prev_metrics_lkl.get('avg_kl_div', kl_div_current))
            kl_decrease_from_prev = kl_div_prev - kl_div_current
            total_reward_lkl += np.tanh(kl_decrease_from_prev / (kl_high_target * 0.5 + EPS)) * kl_target_bonus # Reward decreasing it
        else: # KL is in target range
            total_reward_lkl += kl_target_bonus * 0.25 

        # D_total stability
        d_total_current = current_finite_metrics_lkl['avg_d_total']
        d_total_prev = float(prev_metrics_lkl.get('avg_d_total', d_total_current))
        d_total_change_abs = abs(d_total_current - d_total_prev)
        if d_total_change_abs > 0.25: 
            total_reward_lkl -= rw_lkl.get("lambda_kl_stability_penalty",0.6) * (d_total_change_abs / (0.25 + EPS)) * 1.5 # Penalize large D changes

        # Penalty if lambda_kl is high but recon is bad
        current_lambda_kl_val_lkl = current_finite_metrics_lkl['current_lambda_kl_val']
        if current_lambda_kl_val_lkl > 0.1 and current_finite_metrics_lkl['avg_recon'] > 0.1:
             total_reward_lkl -= rw_lkl.get("lambda_kl_too_high_recon_bad_penalty", 0.7)

        if self.logger.isEnabledFor(logging.DEBUG):
            log_mets_debug_lkl = {k_lkl: f'{v_lkl:.3f}' if isinstance(v_lkl, (float, np.float32)) and np.isfinite(v_lkl) else str(v_lkl) for k_lkl,v_lkl in interval_metrics.items()}
            self.logger.debug(f"LKL_Rew ({self.logger.name}): Raw={total_reward_lkl:.3f}. IntervalMet: {log_mets_debug_lkl}")

        return float(np.clip(total_reward_lkl, self.reward_clipping[0], self.reward_clipping[1])) if self.reward_clipping else float(total_reward_lkl)


    def set_current_lambda_kl(self, lambda_kl_val: float):
        if np.isfinite(lambda_kl_val): self.current_lambda_kl = float(lambda_kl_val)
        else: self.logger.warning(f"Attempt to set non-finite lambda_kl ({self.logger.name}): {lambda_kl_val}")

    def get_info(self) -> Dict[str, Any]:
        q_mem_mb_info = 0.0
        try:
            if self.q_table:
                q_mem_mb_info = sum( sys.getsizeof(s_tuple_info) + sum(q_vals_info.nbytes + sys.getsizeof(p_type_info) for p_type_info, q_vals_info in q_actions_info.items())
                    for s_tuple_info, q_actions_info in self.q_table.items() ) / (1024**2)
        except Exception as e_mem_info: self.logger.error(f"Error Q-table mem ({self.logger.name}): {e_mem_info}"); q_mem_mb_info = -1.0

        avg_reward_recent_info = np.mean(list(self.reward_hist)) if self.reward_hist else 0.0
        
        info_dict_final: Dict[str, Any] = {
            "epsilon": round(self.epsilon, 4), "q_table_size": len(self.q_table),
            "q_table_mem_mb_approx": round(q_mem_mb_info, 3),
            "last_lr_mom_action": self.prev_lr_mom_action if self.prev_lr_mom_action is not None else "None",
            "last_lambda_kl_action": self.prev_lambda_kl_action if self.prev_lambda_kl_action is not None else "None",
            "last_heuristic_toggle_action": self.prev_heuristic_toggle_action if self.prev_heuristic_toggle_action is not None else "None",
            f"avg_reward_last_{self.reward_hist.maxlen}": round(avg_reward_recent_info, 3),
            "on_probation_lr_mom": self.on_probation, "probation_step_lr_mom": self.current_probation_step if self.on_probation else -1,
            "on_probation_lkl_action": self.lkl_on_probation, "probation_step_lkl_action": self.lkl_current_probation_step if self.lkl_on_probation else -1,
            "current_q_alpha": round(self.current_alpha, 5), 
            "current_q_gamma": round(self.current_gamma, 4)
        }
        if hasattr(self, 'epsilon_boost_active_steps') and self.epsilon_boost_active_steps > 0:
            info_dict_final["epsilon_boost_active_for_steps"] = self.epsilon_boost_active_steps
        return info_dict_final

    def set_initial_losses(self, losses: Dict[str, float], is_generator_q: bool, 
                           wubu_geo_params: Optional[Dict[str,float]] = None):
        loss_map_init_set = { 
            'loss_g_total': self.loss_g_total_hist, 'loss_g_recon': self.loss_g_recon_hist,
            'loss_g_kl': self.loss_g_kl_hist, 'loss_g_adv': self.loss_g_adv_hist,
            'loss_d_total': self.loss_d_total_hist, 'loss_d_real': self.loss_d_real_hist,
            'loss_d_fake': self.loss_d_fake_hist 
        }
        relevant_keys_set = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total'] if is_generator_q \
                        else ['loss_d_total', 'loss_d_real', 'loss_d_fake', 'loss_g_total', 'loss_g_adv']
        
        needs_init_set = any(not loss_map_init_set[name_set] for name_set in relevant_keys_set if name_set in loss_map_init_set)
        if needs_init_set:
            self.logger.info(f"Initializing Q-Ctrl loss histories for {'G' if is_generator_q else 'D'} ({self.logger.name}).")
            for name_set_fill in relevant_keys_set:
                deq_set_fill = loss_map_init_set.get(name_set_fill)
                if deq_set_fill is not None and not deq_set_fill: 
                    val_set_fill = losses.get(name_set_fill)
                    fill_val_set = val_set_fill if val_set_fill is not None and np.isfinite(val_set_fill) else 1.0
                    if val_set_fill is None or not np.isfinite(val_set_fill): self.logger.warning(f"Missing/non-finite '{name_set_fill}' for Q-hist init. Using {fill_val_set}.")
                    for _ in range(self.state_history_len): deq_set_fill.append(fill_val_set)

        if wubu_geo_params and is_generator_q:
            geo_map_init_set = {
                'avg_curvature': self.wubu_avg_curvature_hist, 'avg_scale': self.wubu_avg_scale_hist,
                'avg_spread': self.wubu_avg_spread_hist, 'var_curvature': self.wubu_var_curvature_hist
            }
            needs_geo_init_set = any(not deq_geo_set for deq_geo_set in geo_map_init_set.values())
            if needs_geo_init_set:
                self.logger.info(f"Initializing Q-Ctrl WuBu geometric param histories for G ({self.logger.name}).")
                for name_geo_set, deq_geo_fill in geo_map_init_set.items():
                    if not deq_geo_fill: 
                        val_geo_set = wubu_geo_params.get(name_geo_set)
                        fill_val_geo_set = val_geo_set if val_geo_set is not None and np.isfinite(val_geo_set) else 0.1 
                        if val_geo_set is None or not np.isfinite(val_geo_set): self.logger.warning(f"Missing/non-finite '{name_geo_set}' for WuBu geo Q-hist init. Using {fill_val_geo_set}.")
                        for _ in range(self.state_history_len): deq_geo_fill.append(fill_val_geo_set)
                        
    def set_initial_lambda_kl_metrics(self, interval_metrics: Dict[str, Optional[float]]):
        metric_map_set = { 'avg_recon': self.interval_avg_recon_hist, 'avg_kl_div': self.interval_avg_kl_div_hist,
            'avg_d_total': self.interval_avg_d_total_hist, 'val_metric': self.interval_val_metric_hist }
        needs_init_any_set = any(not deq_metric_set for deq_metric_set in metric_map_set.values())
        if needs_init_any_set:
            self.logger.info(f"Initializing Q-Ctrl Lambda_KL interval metrics histories ({self.logger.name}).")
            for name_metric_set, deq_metric_fill in metric_map_set.items():
                if not deq_metric_fill: 
                    val_metric_set = interval_metrics.get(name_metric_set)
                    default_val_metric_set = 1.0 
                    fill_val_metric_set = float(val_metric_set) if val_metric_set is not None and np.isfinite(val_metric_set) else default_val_metric_set
                    if val_metric_set is None or not np.isfinite(val_metric_set): self.logger.warning(f"Missing/non-finite '{name_metric_set}' for LKL Q-hist init. Using {fill_val_metric_set}.")
                    for _ in range(self.lambda_kl_state_history_len): deq_metric_fill.append(fill_val_metric_set)

# RiemannianEnhancedSGD: Largely unchanged, but uses the overhauled Q-Controller and its config
class RiemannianEnhancedSGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 1e-3, momentum: float = 0.9,
                 weight_decay: float = 0.01, max_grad_norm_risgd: float = 1.0,
                 q_controller_config_dict: Optional[Dict] = None, # <TOTAL_STRATEGY_INTEGRATION> Pass full dict
                 optimizer_type: str = "generator",
                 # <TOTAL_STRATEGY_INTEGRATION> For optimizer type switching (experimental)
                 # alt_optimizer_class: Optional[Type[torch.optim.Optimizer]] = None, 
                 # alt_optimizer_params: Optional[Dict] = None
                 ):
        # ... (original init checks) ...
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, initial_lr=lr, momentum=momentum, initial_momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.full_optimizer_type_str = optimizer_type.lower()
        if "generator" in self.full_optimizer_type_str: self.core_optimizer_role = "generator"
        elif "discriminator" in self.full_optimizer_type_str: self.core_optimizer_role = "discriminator"
        else:
            temp_logger = logging.getLogger(f"{logger.name}.RiSGD.InitCheck")
            temp_logger.warning(f"Unclear core role from optimizer_type '{optimizer_type}'. Defaulting to 'generator' for Q-role.")
            self.core_optimizer_role = "generator"

        if q_controller_config_dict: # <TOTAL_STRATEGY_INTEGRATION> Use passed dict
            self.q_controller: Optional[HAKMEMQController] = HAKMEMQController(
                q_config=q_controller_config_dict, 
                associated_component_name=self.full_optimizer_type_str
            )
        else: self.q_controller = None
            
        self.logger = logging.getLogger(f"{logger.name}.RiSGD.{self.full_optimizer_type_str.replace('_', ' ').title().replace(' ', '')}")
        self.logger.info(f"Q-Controller {'en' if self.q_controller else 'dis'}abled for {self.full_optimizer_type_str} (Core Role: {self.core_optimizer_role}).")
        
        self.max_grad_norm_risgd = float(max_grad_norm_risgd) if max_grad_norm_risgd > 0 else float('inf')
        self._step_count_internal = 0
        self.grad_stats = GradientStats() # Unchanged usage
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad: self.state.setdefault(p, {})
        
        # Placeholder for optimizer type switching
        # self.current_optimizer_impl = self # Initially points to RiSGD itself
        # self.alt_optimizer_class = alt_optimizer_class
        # self.alt_optimizer_params = alt_optimizer_params
        # self.alt_optimizer_instance = None # To be created if switched

    def zero_grad(self, set_to_none: bool = True): # Unchanged
        # if self.current_optimizer_impl is not self: self.alt_optimizer_instance.zero_grad(set_to_none=set_to_none)
        # else: 
        super().zero_grad(set_to_none=set_to_none)

    def q_controller_update_and_set_hyperparams(self, avg_losses_dict: Dict[str, Optional[float]],
                                                current_lambda_kl_value: Optional[float] = None,
                                                wubu_geo_params_for_q: Optional[Dict[str, float]] = None): # <TOTAL_STRATEGY_INTEGRATION>
        if not self.q_controller: return
        finite_losses_for_q_state: Dict[str, float] = {
            k: v for k, v in avg_losses_dict.items() if v is not None and np.isfinite(v)
        }
        is_gen_q = (self.core_optimizer_role == "generator")
        
        req_keys_g = ['loss_g_total', 'loss_g_recon', 'loss_g_kl', 'loss_g_adv', 'loss_d_total']
        req_keys_d = ['loss_d_total', 'loss_g_total', 'loss_g_adv', 'loss_d_real', 'loss_d_fake']
        required_keys = req_keys_g if is_gen_q else req_keys_d
        if not all(key in finite_losses_for_q_state for key in required_keys):
            self.logger.debug(f"QCtrl ({self.full_optimizer_type_str}): Insufficient finite losses for LR/Mom state. Skipping Q-update.")
            return
            
        if hasattr(self.q_controller, 'set_current_lambda_kl') and current_lambda_kl_value is not None:
            self.q_controller.set_current_lambda_kl(current_lambda_kl_value)
            
        current_lr_for_q_state = self.param_groups[0]['lr']
        current_mom_for_q_state = self.param_groups[0]['momentum']
        
        # <TOTAL_STRATEGY_INTEGRATION> Pass wubu_geo_params to get_lr_mom_state
        q_state_current = self.q_controller.get_lr_mom_state(
            finite_losses_for_q_state, current_lr_for_q_state, current_mom_for_q_state, 
            is_generator_q=is_gen_q, wubu_geo_params=wubu_geo_params_for_q
        )
        
        # <TOTAL_STRATEGY_INTEGRATION> Get Q-table stats for reward calculation
        q_table_stats_for_reward: Optional[Dict[str,float]] = None
        if q_state_current: # If state is valid
            q_table_stats_for_reward = {
                'q_table_size': float(len(self.q_controller.q_table)),
                'avg_q_value_magnitude': 0.0
            }
            if self.q_controller.q_table:
                all_q_vals_stats = [q_val for state_actions in self.q_controller.q_table.values() 
                                    for q_vals_for_param in state_actions.values() 
                                    for q_val in q_vals_for_param if np.isfinite(q_val)]
                if all_q_vals_stats: q_table_stats_for_reward['avg_q_value_magnitude'] = float(np.mean(np.abs(all_q_vals_stats)))


        if self.q_controller.prev_lr_mom_state is not None and \
           self.q_controller.prev_lr_mom_action is not None and q_state_current is not None:
            # <TOTAL_STRATEGY_INTEGRATION> Pass wubu_geo_params and q_table_stats to reward
            reward = self.q_controller.compute_lr_mom_reward(
                finite_losses_for_q_state, is_generator_q=is_gen_q,
                wubu_geo_params=wubu_geo_params_for_q, q_table_stats=q_table_stats_for_reward
            )
            if np.isfinite(reward):
                self.q_controller.update_q_values(
                    self.q_controller.prev_lr_mom_state, self.q_controller.prev_lr_mom_action,
                    reward, q_state_current, mode='lr_mom'
                )
        elif q_state_current is not None and hasattr(self.q_controller, 'set_initial_losses'):
             self.q_controller.set_initial_losses(finite_losses_for_q_state, is_generator_q=is_gen_q, wubu_geo_params=wubu_geo_params_for_q) # <TOTAL_STRATEGY_INTEGRATION>
             
        self.q_controller.prev_lr_mom_state = q_state_current
        # <TOTAL_STRATEGY_INTEGRATION> action_for_upcoming_step can now contain more action types
        action_for_upcoming_step = self.q_controller.choose_action(q_state_current, mode='lr_mom')
        self.q_controller.prev_lr_mom_action = action_for_upcoming_step # Store the chosen action dict

        if action_for_upcoming_step:
            # Apply LR/Momentum scales
            for group in self.param_groups:
                base_lr = group['initial_lr']
                base_mom = group['initial_momentum']
                group['lr'] = float(np.clip(base_lr * action_for_upcoming_step.get('lr_scale', 1.0), 1e-8, 1.0))
                group['momentum'] = float(np.clip(base_mom * action_for_upcoming_step.get('momentum_scale', 1.0), 0.0, 0.999))
            
            # <TOTAL_STRATEGY_INTEGRATION> Handle other Q-controlled actions (e.g., heuristic toggles, optimizer switch)
            # These would typically be handled by HybridTrainer based on these suggestions.
            # For now, RiSGD primarily handles LR/Mom. If it were to handle optimizer switching itself:
            # if 'optimizer_type_switch' in action_for_upcoming_step and action_for_upcoming_step['optimizer_type_switch'] != "default":
            #     self._try_switch_optimizer_impl(action_for_upcoming_step['optimizer_type_switch'])
            pass # HybridTrainer will poll Q-controller for these suggestions


    @torch.no_grad()
    def step(self, closure=None): # Logic for RiSGD step itself is largely unchanged
        # if self.current_optimizer_impl is not self and self.alt_optimizer_instance:
        #     return self.alt_optimizer_instance.step(closure)
        
        loss = closure() if closure is not None else None
        # ... (original RiSGD step logic, using self.param_groups, self.state, etc.) ...
        # This part is complex and assumed to be mostly correct.
        # Key is that it uses self.param_groups[0]['lr'] etc. which are set by Q-controller.
        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad
                if not torch.isfinite(grad).all():
                    self.logger.warning(f"Optimizer step: Non-finite gradient for param shape {p.shape} ({self.full_optimizer_type_str}). Skipping update.")
                    self.state[p].pop('momentum_buffer', None) # Clear momentum for this unstable param
                    continue
                if self.max_grad_norm_risgd > 0 and self.max_grad_norm_risgd != float('inf'):
                    param_grad_norm = grad.norm().item()
                    if param_grad_norm > self.max_grad_norm_risgd:
                        grad.mul_(self.max_grad_norm_risgd / (param_grad_norm + EPS))
                
                manifold: Optional[Manifold] = getattr(p, 'manifold', None) # type: ignore
                if isinstance(manifold, PoincareBall) and manifold.c > 0:
                    p_projected_on_manifold = manifold.proju(p) # Project p before using in egrad or expmap
                    grad_eff = grad.clone() # Make a copy to modify
                    if weight_decay != 0: grad_eff.add_(p, alpha=weight_decay) # Add weight decay to grad
                    try:
                        riemannian_grad = manifold.egrad2rgrad(p_projected_on_manifold, grad_eff)
                    except Exception as e_egrad:
                        self.logger.error(f"egrad2rgrad failed for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}): {e_egrad}. Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue
                    
                    if not torch.isfinite(riemannian_grad).all():
                        self.logger.warning(f"Non-finite Riemannian grad for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Skipping param.")
                        self.state[p].pop('momentum_buffer', None)
                        continue

                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None: buf = torch.clone(riemannian_grad).detach()
                        else:
                            if buf.shape == riemannian_grad.shape: buf.mul_(momentum).add_(riemannian_grad)
                            else: buf = torch.clone(riemannian_grad).detach() # Shape mismatch, reset buffer
                        self.state[p]['momentum_buffer'] = buf
                    else: buf = riemannian_grad # No momentum
                    
                    if not torch.isfinite(buf).all(): # Check momentum buffer
                        self.logger.warning(f"Non-finite momentum buffer for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Resetting.")
                        buf.zero_(); self.state[p]['momentum_buffer'] = buf # Reset if non-finite
                        
                    expmap_tangent_vector = buf.mul(-lr) # dp = -lr * riemannian_grad_with_momentum
                    if not torch.isfinite(expmap_tangent_vector).all():
                        self.logger.warning(f"Non-finite tangent vector for expmap P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Skipping.")
                        continue # Skip update for this param

                    try: # Try to update the parameter
                        new_p_candidate = manifold.expmap(p_projected_on_manifold, expmap_tangent_vector)
                        if not torch.isfinite(new_p_candidate).all():
                            self.logger.warning(f"Expmap resulted in non-finite P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}). Projecting and zeroing momentum.")
                            p.data = manifold.proju(torch.nan_to_num(new_p_candidate, nan=0.0))
                            if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                        else:
                            p.data = manifold.proju(new_p_candidate) # Project result back to ball
                    except Exception as e_expmap:
                        self.logger.error(f"Expmap failed for P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}): {e_expmap}. Zeroing momentum.")
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                        continue # Skip if expmap fails catastrophically
                    
                    if not torch.isfinite(p.data).all(): # Final check on parameter
                        self.logger.error(f"Parameter P:{p.shape} (c={manifold.c:.2e}, opt: {self.full_optimizer_type_str}) became non-finite AFTER update. Resetting to origin.")
                        p.data = manifold.expmap0(torch.zeros_like(p.data, device=p.device)) # Reset to origin
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()
                else: # Euclidean parameter update
                    grad_eff_euc = grad.clone()
                    if weight_decay != 0: grad_eff_euc.add_(p, alpha=weight_decay)
                    buf = self.state[p].get('momentum_buffer')
                    if momentum != 0:
                        if buf is None: buf = torch.clone(grad_eff_euc).detach()
                        else:
                            if buf.shape == grad_eff_euc.shape: buf.mul_(momentum).add_(grad_eff_euc)
                            else: buf = torch.clone(grad_eff_euc).detach() # Shape mismatch
                        self.state[p]['momentum_buffer'] = buf
                    else: buf = grad_eff_euc
                    
                    if not torch.isfinite(buf).all(): # Check Euclidean momentum buffer
                        self.logger.warning(f"Non-finite Euclidean momentum buffer for P:{p.shape} (opt: {self.full_optimizer_type_str}). Resetting.")
                        buf.zero_(); self.state[p]['momentum_buffer'] = buf

                    p.add_(buf, alpha=-lr)
                    if not torch.isfinite(p.data).all(): # Final check for Euclidean params
                        self.logger.warning(f"Euclidean P:{p.shape} (opt: {self.full_optimizer_type_str}) became non-finite. Clamping and zeroing momentum.")
                        # Use a large but finite clamp value for Euclidean parameters if they go wild
                        euclidean_clamp_val = 1e6 
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=euclidean_clamp_val, neginf=-euclidean_clamp_val)
                        p.data.clamp_(-euclidean_clamp_val, euclidean_clamp_val)
                        if 'momentum_buffer' in self.state[p]: self.state[p]['momentum_buffer'].zero_()

        self._step_count_internal += 1
        return loss

    def get_q_controller_info(self) -> Dict: # Unchanged
        return self.q_controller.get_info() if self.q_controller else {"Q-Controller": "Disabled"}
    def get_gradient_stats_summary_optimizer_view(self) -> Dict: # Unchanged
        return self.grad_stats.get_step_summary_for_logging()

# GAAD Components (golden_subdivide_rect_fixed_n, phi_spiral_patch_centers_fixed_n): Unchanged for now.

# Architectural Components (RegionExtractor, DCTCoeffEmbed): Unchanged.

# AudioSpecEncoder, AudioSpecGenerator, AudioSpecDiscriminator:
# Need to accept and use WuBuStackConfig and GlobalContextConfig.
# Their internal FullyHyperbolicWuBuNestingModel instances will be initialized with these.
# For brevity, I'll show the change in __init__ and forward call to WuBuModel.

# WuBuSpecTrans_v0.2.0_TotalStrategy (Continued)

class AudioSpecEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config_ref: Dict, gaad_config_ref: Dict, # Renamed for clarity
                 wubu_s_stack_config: WuBuStackConfig, 
                 global_context_cfg: WuBuGlobalContextConfig, 
                 latent_dim: int):
        super().__init__()
        self.args = args
        self.audio_config_ref = audio_config_ref # Storing reference passed by main model
        self.gaad_config_ref = gaad_config_ref   # Storing reference
        self.wubu_s_stack_config = wubu_s_stack_config
        self.global_context_cfg = global_context_cfg
        self.latent_dim = latent_dim
        self.logger = logging.getLogger(f"{logger.name}.Encoder")

        self.num_gaad_regions = self.gaad_config_ref['num_regions']
        self.region_proc_size = (args.region_proc_size_t, args.region_proc_size_f)
        self.region_extractor = RegionalSpectrogramRegionExtractor(region_proc_size=self.region_proc_size)
        self.num_dct_coeffs_flat = self.region_proc_size[0] * self.region_proc_size[1]
        
        # Ensure encoder_initial_tangent_dim matches the input_tangent_dim expected by WuBu-S stack
        # This is now implicitly handled by wubu_s_stack_config.input_tangent_dim if it were part of the config dataclass.
        # For now, args.encoder_initial_tangent_dim is the source.
        self.dct_coeff_embed = DCTCoeffEmbed(
            num_dct_coeffs_per_region=self.num_dct_coeffs_flat,
            embed_dim=args.encoder_initial_tangent_dim # This should be input_tangent_dim for WuBu-S
        )
        
        # The output_tangent_dim for WuBu-S should align with wubu_s_output_dim_encoder from audio_config_ref
        # which is then used for fc_mu and fc_logvar.
        self.wubu_s_encoder = FullyHyperbolicWuBuNestingModel(
            input_tangent_dim=args.encoder_initial_tangent_dim,
            output_tangent_dim=self.audio_config_ref['wubu_s_output_dim_encoder'], 
            wubu_stack_config=self.wubu_s_stack_config,
            global_context_config=self.global_context_cfg
        )
        self.fc_mu = nn.Linear(self.audio_config_ref['wubu_s_output_dim_encoder'], self.latent_dim)
        self.fc_logvar = nn.Linear(self.audio_config_ref['wubu_s_output_dim_encoder'], self.latent_dim)
        
        self.apply(init_weights_general)
        self.logger.info(f"AudioSpecEncoder initialized with WuBuStack '{self.wubu_s_stack_config.stack_name}'. "
                         f"Region Proc Size (T,F): {self.region_proc_size}, DCT Coeffs/Region: {self.num_dct_coeffs_flat}, "
                         f"WuBu-S InDim: {args.encoder_initial_tangent_dim}, WuBu-S OutDim (target): {self.audio_config_ref['wubu_s_output_dim_encoder']}")

    def _apply_dct_and_normalize(self, region_patches: torch.Tensor) -> torch.Tensor:
        B, N_Reg, C, F_p, T_p = region_patches.shape
        patches_for_dct = region_patches.reshape(-1, F_p, T_p) # Assuming C=1 for Mel

        if dct_2d is None or not TORCH_DCT_AVAILABLE:
            self.logger.error("dct_2d function is not available. Cannot perform DCT. Returning zeros.")
            return torch.zeros_like(patches_for_dct).reshape(B, N_Reg, C, F_p, T_p) # Reshape zeros

        dct_coeffs = dct_2d(patches_for_dct) 

        norm_dct_coeffs: torch.Tensor
        if self.args.dct_norm_type == "none":
            norm_dct_coeffs = dct_coeffs
        elif self.args.dct_norm_type == "global_scale":
            norm_dct_coeffs = dct_coeffs / self.args.dct_norm_global_scale
        elif self.args.dct_norm_type == "tanh":
            norm_dct_coeffs = torch.tanh(dct_coeffs / self.args.dct_norm_tanh_scale)
        else: # Default to global scale for safety
            self.logger.warning(f"Unknown DCT norm type: {self.args.dct_norm_type}. Using global_scale.")
            norm_dct_coeffs = dct_coeffs / self.args.dct_norm_global_scale
        
        return norm_dct_coeffs.reshape(B, N_Reg, C, F_p, T_p)


    def forward(self, mel_spectrogram: torch.Tensor, 
                global_context_raw_features: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C_spec, H_spec, W_spec = mel_spectrogram.shape
        device = mel_spectrogram.device
        dtype = mel_spectrogram.dtype

        gaad_bboxes_list = []
        for b_idx in range(B):
            spec_dims = (W_spec, H_spec) # (Time, Freq) for GAAD function
            # gaad_config_ref is used here
            bboxes_current_spec = golden_subdivide_rect_fixed_n(
                spec_dims, self.gaad_config_ref['num_regions'], 
                device, dtype, self.gaad_config_ref.get('min_size_px',5)
            )
            gaad_bboxes_list.append(bboxes_current_spec)
        gaad_bboxes_batch = torch.stack(gaad_bboxes_list)

        processed_regions = self.region_extractor(mel_spectrogram, gaad_bboxes_batch)
        # Assuming C_spec is 1 for Mel spectrograms after region_extractor if input was (B,1,H,W)
        # If region_extractor outputs (B, NumRegions, C_spec, F_proc, T_proc)
        # and C_spec might not be 1 if other inputs are used one day.
        
        norm_dct_coeffs_structured = self._apply_dct_and_normalize(processed_regions)
        
        # Squeeze channel dim C_spec if it's 1. dct_coeff_embed expects (..., num_dct_coeffs_flat)
        # norm_dct_coeffs_structured is (B, NumRegions, C_spec, F_proc, T_proc)
        if norm_dct_coeffs_structured.shape[2] != 1:
            self.logger.warning(f"Encoder: Expected C_spec=1 for DCT coeffs, got {norm_dct_coeffs_structured.shape[2]}. Averaging over channel dim.")
            norm_dct_coeffs_squeezed = torch.mean(norm_dct_coeffs_structured, dim=2)
        else:
            norm_dct_coeffs_squeezed = norm_dct_coeffs_structured.squeeze(2)
            
        flat_norm_dct_coeffs = norm_dct_coeffs_squeezed.reshape(B, self.num_gaad_regions, -1)
        embedded_dct_coeffs = self.dct_coeff_embed(flat_norm_dct_coeffs)
        
        wubu_s_input = embedded_dct_coeffs.reshape(B * self.num_gaad_regions, -1)
        wubu_s_features_flat = self.wubu_s_encoder(wubu_s_input, global_context_raw_features)
        
        wubu_s_features_structured = wubu_s_features_flat.reshape(B, self.num_gaad_regions, -1)
        aggregated_features = torch.mean(wubu_s_features_structured, dim=1)
        
        mu = self.fc_mu(aggregated_features)
        logvar = self.fc_logvar(aggregated_features)
        
        # norm_dct_coeffs_squeezed is (B, NumRegions, F_proc, T_proc)
        return mu, logvar, norm_dct_coeffs_squeezed, gaad_bboxes_batch


class AudioSpecGenerator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config_ref: Dict, gaad_config_ref: Dict, # Renamed
                 wubu_g_stack_config: WuBuStackConfig, 
                 global_context_cfg: WuBuGlobalContextConfig, 
                 latent_dim: int):
        super().__init__()
        self.args = args
        self.audio_config_ref = audio_config_ref
        self.gaad_config_ref = gaad_config_ref
        self.wubu_g_stack_config = wubu_g_stack_config
        self.global_context_cfg = global_context_cfg
        self.latent_dim = latent_dim
        self.logger = logging.getLogger(f"{logger.name}.Generator")

        self.num_gaad_regions = self.gaad_config_ref['num_regions']
        self.region_proc_size = (args.region_proc_size_t, args.region_proc_size_f)
        self.num_dct_coeffs_flat = self.region_proc_size[0] * self.region_proc_size[1]
        
        # initial_gen_wubu_dim for fc_expand_latent should match input_tangent_dim of WuBu-G
        # args.encoder_initial_tangent_dim might not be right if WuBu-G has different input needs.
        # Assume for now it's a shared initial tangent dim for simplicity, or it should be a new arg.
        # Let's use a dedicated arg if available, else fallback.
        self.initial_gen_wubu_dim = getattr(args, 'generator_initial_tangent_dim', args.encoder_initial_tangent_dim)

        self.fc_expand_latent = nn.Linear(
            self.latent_dim,
            self.num_gaad_regions * self.initial_gen_wubu_dim
        )
        
        self.wubu_generator: nn.Module
        if self.wubu_g_stack_config.num_levels == 0:
             self.logger.warning(f"Generator WuBuStack '{self.wubu_g_stack_config.stack_name}' has 0 levels. Using MLP fallback.")
             self.wubu_generator = nn.Sequential(
                 nn.Linear(self.initial_gen_wubu_dim, self.initial_gen_wubu_dim * 2), SwiGLUActivation(),
                 nn.LayerNorm(self.initial_gen_wubu_dim * 2),
                 nn.Linear(self.initial_gen_wubu_dim * 2, self.num_dct_coeffs_flat)
             )
        else:
            self.wubu_generator = FullyHyperbolicWuBuNestingModel(
                input_tangent_dim=self.initial_gen_wubu_dim, 
                output_tangent_dim=self.num_dct_coeffs_flat, # WuBu-G directly outputs final DCT coeffs
                wubu_stack_config=self.wubu_g_stack_config,
                global_context_config=self.global_context_cfg
            )
        
        if self.args.dct_norm_type == "tanh": self.final_activation = nn.Tanh()
        else: self.final_activation = nn.Identity() # For global_scale or none
        
        self.apply(init_weights_general)
        self.logger.info(f"AudioSpecGenerator initialized with WuBuStack '{self.wubu_g_stack_config.stack_name}'. "
                         f"WuBu-G InDim (per region): {self.initial_gen_wubu_dim}, WuBu-G OutDim (DCT coeffs): {self.num_dct_coeffs_flat}")

    @staticmethod
    def _unnormalize_dct(norm_dct_coeffs: torch.Tensor, args_ref: argparse.Namespace) -> torch.Tensor:
        if args_ref.dct_norm_type == "none": 
            return norm_dct_coeffs
        elif args_ref.dct_norm_type == "global_scale":
            # Ensure input is finite before scaling
            if not torch.isfinite(norm_dct_coeffs).all(): 
                norm_dct_coeffs = torch.nan_to_num(norm_dct_coeffs, nan=0.0, posinf=1.0, neginf=-1.0) # Clamp to a typical normalized range
            
            scaled_output = norm_dct_coeffs * args_ref.dct_norm_global_scale
            # Clamp output to prevent extreme values if input was already large or scale is large
            if not torch.isfinite(scaled_output).all():
                finfo_unnorm = torch.finfo(scaled_output.dtype)
                # Use a very large but finite clamp value, e.g. 10x TAN_VEC_CLAMP_VAL or half of finfo.max
                safe_max_val_unnorm = min(finfo_unnorm.max / 2 if finfo_unnorm.max < float('inf') else TAN_VEC_CLAMP_VAL * 100.0, TAN_VEC_CLAMP_VAL * 100.0) 
                scaled_output = torch.clamp(
                    torch.nan_to_num(scaled_output, nan=0.0, posinf=safe_max_val_unnorm, neginf=-safe_max_val_unnorm),
                    min=-safe_max_val_unnorm, max=safe_max_val_unnorm
                )
            return scaled_output
        elif args_ref.dct_norm_type == "tanh":
            if not torch.is_tensor(norm_dct_coeffs): 
                # This should not happen if called from within the model pipeline
                logging.getLogger(f"{logger.name}.AudioSpecGenerator").error("_unnormalize_dct (tanh): Input not a tensor. Returning input as is.")
                return norm_dct_coeffs # type: ignore 
            
            # Ensure input to atanh is within (-1, 1)
            if not torch.isfinite(norm_dct_coeffs).all():
                # If input has NaNs/Infs, clamp them to a value safely within (-1,1) for atanh
                norm_dct_coeffs = torch.nan_to_num(norm_dct_coeffs, nan=0.0, posinf=0.999, neginf=-0.999) # Example safe clamp
            
            input_dtype_unnorm = norm_dct_coeffs.dtype
            device_unnorm = norm_dct_coeffs.device
            one_tensor_unnorm = torch.tensor(1.0, dtype=torch.float32, device=device_unnorm) # Compute intermediate in float32 for atanh
            
            # Determine epsilon for clamping based on input dtype for atanh
            # Using a slightly larger epsilon than global EPS for atanh input clamping
            eps_for_atanh_clamp: torch.Tensor
            if input_dtype_unnorm == torch.float16: 
                eps_for_atanh_clamp = torch.tensor(1e-3, dtype=torch.float32, device=device_unnorm) # float16 needs a larger gap from +/-1
            elif input_dtype_unnorm == torch.bfloat16:
                eps_for_atanh_clamp = torch.tensor(1e-2, dtype=torch.float32, device=device_unnorm) # bfloat16 also
            else: # float32 or float64
                eps_for_atanh_clamp = torch.tensor(EPS * 10, dtype=torch.float32, device=device_unnorm)

            # Clamp input to atanh to be strictly within (-1+eps, 1-eps)
            strict_upper_bound_unnorm = one_tensor_unnorm - eps_for_atanh_clamp
            strict_lower_bound_unnorm = -one_tensor_unnorm + eps_for_atanh_clamp
            
            # Use nextafter if available for better precision, especially for float32/64
            # Note: nextafter might not be implemented for CUDA float16/bfloat16 by all PyTorch versions
            can_use_nextafter = not ((input_dtype_unnorm == torch.float16 or input_dtype_unnorm == torch.bfloat16) and device_unnorm.type == 'cuda')
            if can_use_nextafter:
                try:
                    strict_upper_bound_unnorm = torch.nextafter(one_tensor_unnorm, torch.tensor(0.0, dtype=torch.float32, device=device_unnorm))
                    strict_lower_bound_unnorm = torch.nextafter(-one_tensor_unnorm, torch.tensor(0.0, dtype=torch.float32, device=device_unnorm))
                except RuntimeError: pass # Fallback to epsilon clamping if nextafter fails

            clamped_for_atanh_unnorm = torch.clamp(norm_dct_coeffs.to(torch.float32), min=strict_lower_bound_unnorm, max=strict_upper_bound_unnorm)
            
            atanh_output_unnorm = torch.atanh(clamped_for_atanh_unnorm)
            unscaled_dct_intermediate_unnorm = atanh_output_unnorm * args_ref.dct_norm_tanh_scale

            # Final clamp for the output (unnormalized DCTs)
            if not torch.isfinite(unscaled_dct_intermediate_unnorm).all():
                # Max value for DCTs could be related to dynamic range of original signal.
                # For now, use a large but finite clamp like TAN_VEC_CLAMP_VAL * scale_factor
                final_output_clamp_val_unnorm = TAN_VEC_CLAMP_VAL * args_ref.dct_norm_tanh_scale * 0.5 # Heuristic clamp
                unscaled_dct_intermediate_unnorm = torch.nan_to_num(
                    unscaled_dct_intermediate_unnorm, nan=0.0, 
                    posinf=final_output_clamp_val_unnorm, neginf=-final_output_clamp_val_unnorm
                )
                unscaled_dct_intermediate_unnorm = torch.clamp(unscaled_dct_intermediate_unnorm, -final_output_clamp_val_unnorm, final_output_clamp_val_unnorm)

            return unscaled_dct_intermediate_unnorm.to(input_dtype_unnorm)
        else:
            logging.getLogger(f"{logger.name}.AudioSpecGenerator").error(f"_unnormalize_dct: Unknown DCT norm type '{args_ref.dct_norm_type}'. Using global_scale as fallback.")
            return (norm_dct_coeffs / args_ref.dct_norm_global_scale) # Fallback, assuming it was tanh normalized


    def forward(self, latent_code: torch.Tensor,
                global_context_raw_features: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B = latent_code.shape[0]
        expanded_z = self.fc_expand_latent(latent_code) # (B, NumRegions * initial_gen_wubu_dim)
        # Reshape for WuBu: (B * NumRegions, initial_gen_wubu_dim)
        wubu_gen_input = expanded_z.view(B * self.num_gaad_regions, self.initial_gen_wubu_dim)
        
        generated_flat_dct_coeffs: torch.Tensor
        if isinstance(self.wubu_generator, FullyHyperbolicWuBuNestingModel):
            # If global context is per-batch item, it needs to be expanded for B * NumRegions
            # Current global_context_raw_features is assumed (1, D_ctx) or (D_ctx) by WuBuModel's embedder.
            # If WuBuModel's forward handles batched context internally, this is fine.
            # For now, assume global_context_raw_features is passed as is, and WuBuModel handles it.
            generated_flat_dct_coeffs = self.wubu_generator(wubu_gen_input, global_context_raw_features)
        else: # MLP fallback
            generated_flat_dct_coeffs = self.wubu_generator(wubu_gen_input)

        generated_flat_dct_coeffs_activated = self.final_activation(generated_flat_dct_coeffs)

        F_proc, T_proc = self.region_proc_size[1], self.region_proc_size[0] # Freq, Time for DCT block
        generated_dct_structured = generated_flat_dct_coeffs_activated.view(
            B, self.num_gaad_regions, F_proc, T_proc
        )
        return generated_dct_structured


class AudioSpecDiscriminator(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config_ref: Dict, gaad_config_ref: Dict,
                 discriminator_specific_config: Dict, 
                 wubu_d_stack_config_if_dct: Optional[WuBuStackConfig], 
                 global_context_cfg_if_wubu_d: Optional[WuBuGlobalContextConfig] 
                 ):
        super().__init__()
        self.args = args
        self.audio_config_ref = audio_config_ref
        self.gaad_config_ref = gaad_config_ref
        self.disc_specific_config = discriminator_specific_config
        self.wubu_d_stack_config = wubu_d_stack_config_if_dct 
        self.global_context_cfg_wubu_d = global_context_cfg_if_wubu_d
        
        # Ensure unique logger name for each D instance if multiple are created
        unique_id_for_logger = str(id(self))[-6:] # Short unique ID
        self.logger = logging.getLogger(f"{logger.name}.Discriminator.{self.disc_specific_config.get('input_type', 'unknown')}_{unique_id_for_logger}")

        self.num_gaad_regions = self.gaad_config_ref['num_regions']
        self.region_proc_size_t = args.region_proc_size_t
        self.region_proc_size_f = args.region_proc_size_f
        self.num_dct_coeffs_flat = self.region_proc_size_t * self.region_proc_size_f

        self.input_type = self.disc_specific_config.get("input_type", "mel")
        self.apply_spectral_norm = self.disc_specific_config.get("apply_spectral_norm", True)
        self.logger.info(f"Initializing Discriminator instance (Type: '{self.input_type}', SpectralNorm: {self.apply_spectral_norm}).")

        self.feature_extractor_module: nn.Module
        self.final_decision_layer: nn.Module

        if self.input_type == "dct":
            # Determine input dim for WuBu-D's embedder
            # Use a specific arg if available, else fallback to encoder_initial_tangent_dim
            wubu_d_embed_input_dim = getattr(args, 'wubu_d_embed_input_dim', args.encoder_initial_tangent_dim)
            self.dct_coeff_embed_disc = DCTCoeffEmbed(
                num_dct_coeffs_per_region=self.num_dct_coeffs_flat,
                embed_dim=wubu_d_embed_input_dim
            )
            # Output dim of WuBu-D stack (before final linear layer to 1 logit)
            # This should come from args.wubu_d_output_dim or similar, specific to D.
            wubu_d_stack_output_dim = getattr(args, 'wubu_d_output_dim', 64) 

            if self.wubu_d_stack_config is None or self.wubu_d_stack_config.num_levels == 0 or self.global_context_cfg_wubu_d is None:
                self.logger.warning(f"D-DCT: WuBu-D stack config invalid or global_ctx_cfg missing for '{self.wubu_d_stack_config.stack_name if self.wubu_d_stack_config else 'N/A'}'. Using MLP fallback (In: {wubu_d_embed_input_dim}, Out: {wubu_d_stack_output_dim}).")
                self.feature_extractor_module = nn.Sequential(
                    nn.Linear(wubu_d_embed_input_dim, wubu_d_embed_input_dim * 2), nn.LeakyReLU(0.2, True), 
                    nn.LayerNorm(wubu_d_embed_input_dim * 2),
                    nn.Linear(wubu_d_embed_input_dim * 2, wubu_d_stack_output_dim)
                )
            else:
                self.logger.info(f"D-DCT: Initializing WuBuNestingModel (In: {wubu_d_embed_input_dim}, Out: {wubu_d_stack_output_dim}) with stack '{self.wubu_d_stack_config.stack_name}'.")
                self.feature_extractor_module = FullyHyperbolicWuBuNestingModel(
                    input_tangent_dim=wubu_d_embed_input_dim,
                    output_tangent_dim=wubu_d_stack_output_dim,
                    wubu_stack_config=self.wubu_d_stack_config,
                    global_context_config=self.global_context_cfg_wubu_d
                )
            self.final_decision_layer = nn.Linear(wubu_d_stack_output_dim, 1)

        elif self.input_type == "mel": 
            n_mels_d = args.n_mels 
            n_time_d = self.audio_config_ref.get("num_time_frames_for_1s_segment", 87) # Default from training logs
            
            base_ch_d = self.disc_specific_config.get("base_disc_channels", 64)
            max_ch_d = self.disc_specific_config.get("max_disc_channels", 512)
            target_final_dim_cnn_d = self.disc_specific_config.get("target_mel_disc_final_feature_dim", 4)

            cnn_layers_d: List[nn.Module] = []
            in_c_d = 1 
            curr_h_d, curr_w_d = n_mels_d, n_time_d
            
            num_downsamples_d = 0
            max_downs_limit_d = 5 # Max 5 downsampling conv layers
            temp_h_d, temp_w_d = curr_h_d, curr_w_d
            
            while temp_h_d > target_final_dim_cnn_d and temp_w_d > target_final_dim_cnn_d and num_downsamples_d < max_downs_limit_d :
                # Kernel 4, Stride 2, Padding 1: H_out = (H_in - 4 + 2*1)/2 + 1 = H_in/2
                next_h_d = (temp_h_d - 4 + 2*1) // 2 + 1
                next_w_d = (temp_w_d - 4 + 2*1) // 2 + 1
                if next_h_d < 1 or next_w_d < 1: break 
                temp_h_d, temp_w_d = next_h_d, next_w_d
                num_downsamples_d +=1
            num_downsamples_d = max(1, num_downsamples_d) # Ensure at least one layer

            self.logger.info(f"D-Mel: Initial Mel (H,W): ({n_mels_d},{n_time_d}). Aiming for ~{target_final_dim_cnn_d}x{target_final_dim_cnn_d} features. Calculated num_downsamples: {num_downsamples_d}")

            for i_d in range(num_downsamples_d):
                out_c_d = min(base_ch_d * (2**i_d), max_ch_d)
                conv_l_d = nn.Conv2d(in_c_d, out_c_d, kernel_size=4, stride=2, padding=1, bias=False) # Bias often false with Norm layers
                if self.apply_spectral_norm: cnn_layers_d.append(spectral_norm(conv_l_d))
                else: cnn_layers_d.append(conv_l_d)
                cnn_layers_d.append(nn.InstanceNorm2d(out_c_d, affine=True)) # InstanceNorm common in GANs
                cnn_layers_d.append(nn.LeakyReLU(0.2, inplace=True))
                in_c_d = out_c_d
                curr_h_d = (curr_h_d - 4 + 2*1) // 2 + 1 
                curr_w_d = (curr_w_d - 4 + 2*1) // 2 + 1
                self.logger.debug(f"  D-Mel CNN layer {i_d+1}: out_c={out_c_d}, new feature map (H,W)=({curr_h_d},{curr_w_d})")
            
            self.feature_extractor_module = nn.Sequential(*cnn_layers_d)
            
            # Final conv layer for patch-wise logits. Kernel 3, Stride 1, Padding 1 preserves dimensions.
            # Or Kernel 4, Stride 1, Padding 0 if further reduction is desired without GAP (PatchGAN style).
            # Let's use a kernel that matches the final feature map size for a single logit if not PatchGAN.
            # For PatchGAN, output is a map of logits.
            # Current AudioSpecDiscriminator in v0.1.1 uses mean of patch_logits_map.
            # So, a final conv to 1 channel, then mean.
            final_conv_patch_d = nn.Conv2d(in_c_d, 1, kernel_size=3, stride=1, padding=1, bias=False) # Preserves H,W from feature_extractor
            if self.apply_spectral_norm: self.final_decision_layer = spectral_norm(final_conv_patch_d)
            else: self.final_decision_layer = final_conv_patch_d
            self.logger.info(f"D-Mel: CNN feature extractor output HxW before final decision: ({curr_h_d}x{curr_w_d}), Channels: {in_c_d}. Final decision layer produces 1 channel map to be averaged.")

        else:
            raise ValueError(f"Unsupported discriminator input_type: {self.input_type}")

        self.apply(init_weights_general)
        self.logger.info(f"Discriminator instance (ID: {unique_id_for_logger}) fully initialized. Total Params: {sum(p.numel() for p in self.parameters()):,}")


    def _assemble_mel_from_dct_regions(self, dct_regions: torch.Tensor, gaad_bboxes: torch.Tensor,
                                       target_mel_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        B_asm, N_Reg_asm, F_p_asm, T_p_asm = dct_regions.shape
        _, C_target_asm, H_target_asm, W_target_asm = target_mel_shape # B, C, H, W
        device_asm = dct_regions.device; dtype_asm = dct_regions.dtype

        if idct_2d is None or not TORCH_DCT_AVAILABLE:
            self.logger.error("idct_2d function is not available. Cannot assemble Mel. Returning zeros.")
            return torch.zeros(target_mel_shape, device=device_asm, dtype=dtype_asm)

        dct_regions_flat_asm = dct_regions.reshape(-1, F_p_asm, T_p_asm)
        spatial_regions_flat_asm = idct_2d(dct_regions_flat_asm) # (B*N_Reg, F_p, T_p)
        # Reshape back to (B, N_Reg, F_p, T_p)
        spatial_regions_asm = spatial_regions_flat_asm.reshape(B_asm, N_Reg_asm, F_p_asm, T_p_asm)

        assembled_mel_canvas_asm = torch.zeros(target_mel_shape, device=device_asm, dtype=dtype_asm)
        counts_canvas_asm = torch.zeros(target_mel_shape, device=device_asm, dtype=dtype_asm) + EPS # Avoid div by zero

        for b_item in range(B_asm):
            for r_item in range(N_Reg_asm):
                # Bboxes are [x1, y1, x2, y2] where x=Time (W_target), y=Freq (H_target)
                t1_box, f1_box, t2_box, f2_box = gaad_bboxes[b_item, r_item].round().int().tolist()
                
                # Clamp coordinates to be within target canvas dimensions
                t1_c_asm = max(0, t1_box); f1_c_asm = max(0, f1_box)
                t2_c_asm = min(W_target_asm, t2_box); f2_c_asm = min(H_target_asm, f2_box)
                
                # Skip if region is invalid or zero-size after clamping
                if t1_c_asm >= t2_c_asm or f1_c_asm >= f2_c_asm: continue

                current_spatial_region_asm = spatial_regions_asm[b_item, r_item, :, :] # (F_p, T_p)
                # Add Channel and Batch dims for TF.resize: (1, 1, F_p, T_p)
                current_spatial_region_asm_unsqueezed = current_spatial_region_asm.unsqueeze(0).unsqueeze(0) 
                
                target_h_bbox_asm, target_w_bbox_asm = f2_c_asm - f1_c_asm, t2_c_asm - t1_c_asm
                if target_h_bbox_asm <= 0 or target_w_bbox_asm <= 0: continue # Should be caught by above, but safety
                
                # TF.resize expects (C, H_out, W_out) for size -> (target_h_bbox_asm, target_w_bbox_asm)
                # Input to TF.resize is (N, C, H_in, W_in) -> (1, 1, F_p_asm, T_p_asm)
                # Output will be (1, 1, target_h_bbox_asm, target_w_bbox_asm)
                try:
                    resized_region_asm = TF.resize(current_spatial_region_asm_unsqueezed, 
                                                (target_h_bbox_asm, target_w_bbox_asm),
                                                interpolation=T.InterpolationMode.BILINEAR, antialias=True)
                except Exception as e_resize: # Catch potential errors in resize (e.g. if F_p_asm or T_p_asm is 0)
                    self.logger.error(f"Error resizing region for D assembly: {e_resize}. Region shape: {current_spatial_region_asm_unsqueezed.shape}, Target: ({target_h_bbox_asm},{target_w_bbox_asm}). Skipping region.")
                    continue
                
                # Add to canvas: indices are [b, channel, freq_slice, time_slice]
                # assembled_mel_canvas_asm is (B, C_target, H_target, W_target)
                # C_target should be 1 for Mels.
                assembled_mel_canvas_asm[b_item, 0, f1_c_asm:f2_c_asm, t1_c_asm:t2_c_asm] += resized_region_asm.squeeze(0).squeeze(0) # Remove N, C dims
                counts_canvas_asm[b_item, 0, f1_c_asm:f2_c_asm, t1_c_asm:t2_c_asm] += 1.0
        
        # Average overlapping regions
        assembled_mel_canvas_final = assembled_mel_canvas_asm / counts_canvas_asm
        return assembled_mel_canvas_final


    def forward(self, input_data: torch.Tensor,
                gaad_bboxes_for_assembly: Optional[torch.Tensor] = None,
                target_mel_shape_for_assembly: Optional[Tuple[int,int,int,int]] = None,
                global_context_raw_features: Optional[torch.Tensor] = None, 
                return_features: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B_fwd = input_data.shape[0]
        features_intermediate_fwd: Optional[torch.Tensor] = None
        logits_fwd: torch.Tensor
        
        if self.input_type == "dct":
            # input_data is (B, NumRegions, F_proc, T_proc)
            # Reshape to (B, NumRegions, F_proc * T_proc) for embedder
            flat_dct_coeffs_fwd = input_data.reshape(B_fwd, self.num_gaad_regions, -1)
            embedded_coeffs_fwd = self.dct_coeff_embed_disc(flat_dct_coeffs_fwd)
            # Reshape for WuBu model: (B * NumRegions, embed_dim)
            wubu_d_input_fwd = embedded_coeffs_fwd.reshape(B_fwd * self.num_gaad_regions, -1)
            
            extracted_features_flat_fwd: torch.Tensor
            if isinstance(self.feature_extractor_module, FullyHyperbolicWuBuNestingModel):
                extracted_features_flat_fwd = self.feature_extractor_module(wubu_d_input_fwd, global_context_raw_features)
            else: # MLP fallback
                extracted_features_flat_fwd = self.feature_extractor_module(wubu_d_input_fwd)
                
            # Aggregate features over regions: (B * NumRegions, feat_dim) -> (B, NumRegions, feat_dim) -> (B, feat_dim)
            features_aggregated_fwd = extracted_features_flat_fwd.reshape(B_fwd, self.num_gaad_regions, -1).mean(dim=1)
            features_intermediate_fwd = features_aggregated_fwd # Store for potential feature matching loss
            logits_fwd = self.final_decision_layer(features_aggregated_fwd)

        elif self.input_type == "mel": 
            mel_input_for_d_fwd: torch.Tensor
            if input_data.ndim == 4 and input_data.shape[1] == self.num_gaad_regions: # DCTs provided, need assembly
                if gaad_bboxes_for_assembly is None or target_mel_shape_for_assembly is None:
                    raise ValueError("GAAD bboxes and target_mel_shape needed for D (mel type) with DCT region inputs.")
                # Unnormalize DCTs first (as Generator output is normalized, D on Mels expects unnormalized effect)
                unnorm_dct_coeffs_fwd = AudioSpecGenerator._unnormalize_dct(input_data, self.args)
                mel_input_for_d_fwd = self._assemble_mel_from_dct_regions(unnorm_dct_coeffs_fwd, gaad_bboxes_for_assembly, target_mel_shape_for_assembly)
            elif input_data.ndim == 4 and input_data.shape[1] == 1: # Full Mel spectrogram provided
                mel_input_for_d_fwd = input_data
            else:
                raise ValueError(f"Unsupported input_data shape {input_data.shape} for Discriminator (mel type). Expects (B,NumRegions,F,T) for DCTs or (B,1,F_mel,T_mel) for Mels.")

            cnn_feature_map_fwd = self.feature_extractor_module(mel_input_for_d_fwd) 
            features_intermediate_fwd = cnn_feature_map_fwd # CNN feature map
            patch_logits_map_fwd = self.final_decision_layer(cnn_feature_map_fwd) 
            # Average patch logits to get a single score per item in batch
            logits_fwd = torch.mean(patch_logits_map_fwd, dim=[2,3], keepdim=False) 
        else:
            raise NotImplementedError(f"Discriminator forward not implemented for input_type: {self.input_type}")
        
        # Ensure logits are (B) shape if they became (B,1) due to Linear layer
        if logits_fwd.ndim > 1 and logits_fwd.shape[1] == 1 and logits_fwd.numel() == B_fwd : 
            logits_fwd = logits_fwd.squeeze(1)
        elif logits_fwd.ndim == 0 and B_fwd == 1: # If single item batch and mean reduced to scalar
            logits_fwd = logits_fwd.unsqueeze(0)

        if return_features:
            if features_intermediate_fwd is None: # Should be set by DCT or Mel path
                self.logger.warning("return_features=True but intermediate features are None. Returning logits as features.")
                # Fallback: use logits as features if nothing else, though not ideal
                # Ensure detached clone if logits are used as features
                return logits_fwd, logits_fwd.detach().clone() if logits_fwd is not None else torch.empty(0, device=logits_fwd.device, dtype=logits_fwd.dtype) # type: ignore
            return logits_fwd, features_intermediate_fwd
        return logits_fwd

    def get_interpretability_data_discriminator(self, global_context_raw_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        data_interp_d: Dict[str, Any] = {"disc_instance_id": str(id(self))[-6:], "disc_type": self.input_type}
        if self.input_type == "dct" and isinstance(self.feature_extractor_module, FullyHyperbolicWuBuNestingModel):
            data_interp_d["wubu_d_stack_interpretability"] = self.feature_extractor_module.get_interpretability_data_stack(global_context_raw_features)
        elif self.input_type == "mel" and isinstance(self.feature_extractor_module, nn.Sequential):
            # Log norms of conv layer weights for CNN D
            conv_layer_weight_norms = []
            for i, layer in enumerate(self.feature_extractor_module):
                if isinstance(layer, (nn.Conv2d, spectral_norm)):
                    weight_to_log = layer.weight if isinstance(layer, nn.Conv2d) else layer.module.weight_orig # For SN
                    conv_layer_weight_norms.append({f"cnn_L{i}_w_norm": weight_to_log.norm().item()})
            if conv_layer_weight_norms:
                data_interp_d["cnn_feature_extractor_weights"] = conv_layer_weight_norms
            # Final decision layer weight norm
            final_dec_layer_weight_to_log = self.final_decision_layer.weight if isinstance(self.final_decision_layer, nn.Conv2d) else \
                                       (self.final_decision_layer.module.weight_orig if isinstance(self.final_decision_layer, spectral_norm) else None)
            if final_dec_layer_weight_to_log is not None:
                data_interp_d["final_decision_layer_w_norm"] = final_dec_layer_weight_to_log.norm().item()

        return data_interp_d


class WuBuSpecTransNet(nn.Module):
    def __init__(self, args: argparse.Namespace, audio_config_ref: Dict, gaad_config_ref: Dict,
                 wubu_s_enc_stack_cfg: WuBuStackConfig, 
                 wubu_g_gen_stack_cfg: WuBuStackConfig, 
                 global_ctx_cfg: WuBuGlobalContextConfig 
                 ):
        super().__init__()
        self.args = args
        self.audio_config_ref = audio_config_ref
        self.gaad_config_ref = gaad_config_ref
        self.wubu_s_enc_stack_cfg = wubu_s_enc_stack_cfg
        self.wubu_g_gen_stack_cfg = wubu_g_gen_stack_cfg
        self.global_ctx_cfg = global_ctx_cfg
        self.logger = logging.getLogger(f"{logger.name}.MainNet")
        self.latent_dim = args.latent_dim

        self.encoder = AudioSpecEncoder(args, audio_config_ref, gaad_config_ref, 
                                        wubu_s_enc_stack_cfg, global_ctx_cfg, self.latent_dim)
        self.generator = AudioSpecGenerator(args, audio_config_ref, gaad_config_ref, 
                                          wubu_g_gen_stack_cfg, global_ctx_cfg, self.latent_dim)
        
        self.apply(init_weights_general)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"WuBuSpecTransNet Initialized. Total VAE/Gen Params: {param_count:,}. "
                         f"Encoder WuBu: '{wubu_s_enc_stack_cfg.stack_name}', Generator WuBu: '{wubu_g_gen_stack_cfg.stack_name}'.")

    def encode(self, mel_spectrogram: torch.Tensor, 
               global_context_raw_features: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: mu, logvar, norm_dct_coeffs_target (from encoder), gaad_bboxes_from_enc
        return self.encoder(mel_spectrogram, global_context_raw_features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps_noise = torch.randn_like(std)
        return mu + eps_noise * std

    def decode(self, z: torch.Tensor,
               global_context_raw_features: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        # Output: (B, NumRegions, F_proc, T_proc) - normalized DCTs
        return self.generator(z, global_context_raw_features)

    def forward(self, mel_spectrogram: torch.Tensor,
                global_context_raw_features: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, norm_dct_coeffs_target, gaad_bboxes_from_enc = self.encode(mel_spectrogram, global_context_raw_features)
        z = self.reparameterize(mu, logvar)
        recon_norm_dct_coeffs = self.decode(z, global_context_raw_features) 
        
        return recon_norm_dct_coeffs, mu, logvar, gaad_bboxes_from_enc, norm_dct_coeffs_target

    def get_interpretability_data_vae(self, global_context_raw_features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        data_interp_vae: Dict[str, Any] = {}
        if hasattr(self.encoder, 'wubu_s_encoder') and isinstance(self.encoder.wubu_s_encoder, FullyHyperbolicWuBuNestingModel):
            data_interp_vae["encoder_wubu_s_stack"] = self.encoder.wubu_s_encoder.get_interpretability_data_stack(global_context_raw_features)
        else: data_interp_vae["encoder_wubu_s_stack"] = {"status": "Not WuBu or get_interpretability_data_stack not found"}
            
        if hasattr(self.generator, 'wubu_generator') and isinstance(self.generator.wubu_generator, FullyHyperbolicWuBuNestingModel):
            data_interp_vae["generator_wubu_g_stack"] = self.generator.wubu_generator.get_interpretability_data_stack(global_context_raw_features)
        else: data_interp_vae["generator_wubu_g_stack"] = {"status": "Not WuBu or get_interpretability_data_stack not found"}
            
        return data_interp_vae



# AudioSegmentDataset: Unchanged.

# HybridTrainer: This class will see massive changes to integrate all heuristic, Q-learning, and logging enhancements.
# For brevity, I will outline the key structural changes and new methods.
# The full implementation would be very extensive.

# WuBuSpecTrans_v0.2.0_TotalStrategy (Continued)

class HybridTrainer:
    def __init__(self,
                 model: "WuBuSpecTransNet", device: torch.device,
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 args: argparse.Namespace,
                 wubu_s_enc_cfg: WuBuStackConfig, wubu_g_gen_cfg: WuBuStackConfig,
                 wubu_d_pri_cfg: Optional[WuBuStackConfig], wubu_d_alt_cfg: Optional[WuBuStackConfig],
                 global_ctx_cfg: WuBuGlobalContextConfig,
                 q_learn_cfg_template: Dict, 
                 disc_pri_specific_cfg: Dict, disc_alt_specific_cfg: Dict,
                 rank: int, world_size: int, ddp_active: bool):

        self.model = model 
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.ddp_active = ddp_active
        self.am_main_process = (rank == 0)
        self.logger = logging.getLogger(f"{logger.name}.Trainer") 

        self.wubu_s_enc_cfg = wubu_s_enc_cfg
        self.wubu_g_gen_cfg = wubu_g_gen_cfg
        self.wubu_d_pri_cfg = wubu_d_pri_cfg 
        self.wubu_d_alt_cfg = wubu_d_alt_cfg 
        self.global_ctx_cfg = global_ctx_cfg
        self.q_learn_cfg_template = q_learn_cfg_template 
        self.disc_pri_specific_cfg = disc_pri_specific_cfg
        self.disc_alt_specific_cfg = disc_alt_specific_cfg   

        q_cfg_gen_actual = self.q_learn_cfg_template.copy() if args.q_controller_enabled else None
        if q_cfg_gen_actual: # Customize for Generator Q-Controller if needed
            q_cfg_gen_actual["action_ranges_override"] = { # Example: Gen Q-Ctrl can also suggest heuristic toggles
                'heuristic_toggle_vae_feat_match': np.array([0.0, 1.0], dtype=np.float32),
                'heuristic_toggle_g_easy_win_penalty': np.array([0.0, 1.0], dtype=np.float32),
            }
            # Ensure these new action types are also in the main q_learn_cfg_template if they are standard now
            # Or handle them as special additions here. For now, assuming HAKMEMQController can take action_ranges_override.
        
        self.optimizer_enc_gen = RiemannianEnhancedSGD(
            self.model.parameters(),
            lr=self.args.learning_rate_gen,
            q_controller_config_dict=q_cfg_gen_actual, # Pass the potentially customized config
            max_grad_norm_risgd=self.args.risgd_max_grad_norm,
            optimizer_type="generator_vae" 
        )
        if self.am_main_process: self.logger.info("Optimizer_Enc_Gen initialized.")
        self.q_controller_gen = getattr(self.optimizer_enc_gen, 'q_controller', None)

        if self.am_main_process: self.logger.info("Initializing Discriminators and their Optimizers...")
        m_ref_for_configs_init = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        audio_cfg_for_d_init = getattr(m_ref_for_configs_init, 'audio_config_ref', {})
        gaad_cfg_for_d_init = getattr(m_ref_for_configs_init, 'gaad_config_ref', {})

        self.discriminator_primary_obj = AudioSpecDiscriminator(
            args, audio_cfg_for_d_init, gaad_cfg_for_d_init,
            self.disc_pri_specific_cfg, self.wubu_d_pri_cfg, self.global_ctx_cfg
        ).to(device)
        self.discriminator_alternative_obj = AudioSpecDiscriminator(
            args, audio_cfg_for_d_init, gaad_cfg_for_d_init,
            self.disc_alt_specific_cfg, self.wubu_d_alt_cfg, self.global_ctx_cfg
        ).to(device)

        self.primary_disc_actual_type = self.disc_pri_specific_cfg.get("input_type", "unknown_primary_type")
        self.alternative_disc_actual_type = self.disc_alt_specific_cfg.get("input_type", "unknown_alt_type")
        if self.am_main_process:
            self.logger.info(f"Primary D (Type: '{self.primary_disc_actual_type}') initialized. Params: {sum(p.numel() for p in self.discriminator_primary_obj.parameters()):,}")
            self.logger.info(f"Alternative D (Type: '{self.alternative_disc_actual_type}') initialized. Params: {sum(p.numel() for p in self.discriminator_alternative_obj.parameters()):,}")

        if self.ddp_active:
            local_rank_ddp_init = self.args.local_rank if hasattr(self.args, 'local_rank') and self.args.local_rank != -1 else rank
            ddp_find_unused_d_init = getattr(self.args, 'ddp_find_unused_params_d', False)
            self.discriminator_primary_obj = DDP(self.discriminator_primary_obj, device_ids=[local_rank_ddp_init], output_device=local_rank_ddp_init, find_unused_parameters=ddp_find_unused_d_init)
            self.discriminator_alternative_obj = DDP(self.discriminator_alternative_obj, device_ids=[local_rank_ddp_init], output_device=local_rank_ddp_init, find_unused_parameters=ddp_find_unused_d_init)
            if self.am_main_process: self.logger.info(f"Discriminators DDP wrapped (find_unused_parameters={ddp_find_unused_d_init}).")

        q_cfg_disc_pri_actual_init = self.q_learn_cfg_template.copy() if args.q_controller_enabled else None
        q_cfg_disc_alt_actual_init = self.q_learn_cfg_template.copy() if args.q_controller_enabled else None
        
        lr_disc_alt_init = getattr(args, 'learning_rate_disc_alt', args.learning_rate_disc)
        self.optimizer_disc_primary = RiemannianEnhancedSGD(
            self.discriminator_primary_obj.parameters(), lr=args.learning_rate_disc,
            q_controller_config_dict=q_cfg_disc_pri_actual_init,
            max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type=f"discriminator_primary_{self.primary_disc_actual_type}"
        )
        self.optimizer_disc_alternative = RiemannianEnhancedSGD(
            self.discriminator_alternative_obj.parameters(), lr=lr_disc_alt_init,
            q_controller_config_dict=q_cfg_disc_alt_actual_init,
            max_grad_norm_risgd=args.risgd_max_grad_norm, optimizer_type=f"discriminator_alt_{self.alternative_disc_actual_type}"
        )
        if self.am_main_process: self.logger.info("Discriminator optimizers initialized.")

        self.q_controller_d_primary = getattr(self.optimizer_disc_primary, 'q_controller', None)
        self.q_controller_d_alt = getattr(self.optimizer_disc_alternative, 'q_controller', None)
        
        self.active_discriminator_key = 'primary' 
        if self.args.enable_heuristic_disc_switching:
            initial_disc_type_arg_init = args.initial_disc_type if args.initial_disc_type is not None else args.disc_input_type
            if initial_disc_type_arg_init == self.primary_disc_actual_type: self.active_discriminator_key = 'primary'
            elif initial_disc_type_arg_init == self.alternative_disc_actual_type: self.active_discriminator_key = 'alternative'
            else:
                 if self.am_main_process: self.logger.warning(f"Mismatch in initial active D mapping. Defaulting to 'primary'.")
        
        self.active_discriminator: nn.Module = self.discriminator_primary_obj 
        self.optimizer_disc_active: RiemannianEnhancedSGD = self.optimizer_disc_primary 
        self.active_disc_actual_type: str = self.primary_disc_actual_type 
        self.q_controller_d_active: Optional[HAKMEMQController] = self.q_controller_d_primary 
        self._update_active_discriminator_pointers()
        
        self.lambda_recon = args.lambda_recon; self.lambda_kl = args.lambda_kl; self.lambda_gan = args.lambda_gan
        self.scaler_enc_gen = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.scaler_disc = amp.GradScaler(enabled=args.use_amp and device.type == 'cuda')
        self.global_step = 0; self.current_epoch = 0
        self.is_val_metric_higher_better = self.args.val_primary_metric in ["avg_val_ssim_mel", "avg_val_psnr_mel"]
        self.best_val_metric_val = -float('inf') if self.is_val_metric_higher_better else float('inf')
        self.last_val_metrics: Dict[str, Any] = {}
        self.prev_interval_metrics_for_lambda_kl_reward: Optional[Dict[str, Union[float, None]]] = None
        if self.am_main_process: os.makedirs(args.checkpoint_dir, exist_ok=True)

        self.lpips_loss_fn = None; self.ssim_metric = None
        if self.am_main_process:
            if self.args.use_lpips_for_mel_verification and LPIPS_AVAILABLE and lpips is not None:
                try: self.lpips_loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)
                except Exception as e_lpips_init: self.logger.warning(f"LPIPS init failed: {e_lpips_init}")
            if TORCHMETRICS_SSIM_AVAILABLE and StructuralSimilarityIndexMeasure is not None:
                try: self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                except Exception as e_ssim_init: self.logger.warning(f"SSIM init failed: {e_ssim_init}")
        
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))
        self.fixed_noise_for_sampling = None

        self.lambda_kl_update_interval = getattr(args, 'lambda_kl_update_interval', 0)
        self.lambda_kl_q_controller = None
        if self.args.q_controller_enabled and self.lambda_kl_update_interval > 0:
            q_cfg_lkl_actual_init = self.q_learn_cfg_template.copy()
            q_cfg_lkl_actual_init["lambda_kl_scale_options"] = getattr(args, 'q_lkl_scale_options', [0.85, 0.95, 1.0, 1.05, 1.15])
            q_cfg_lkl_actual_init['num_probation_steps'] = getattr(args, 'q_lkl_action_probation_steps', 
                                                               max(3, q_cfg_lkl_actual_init.get('lambda_kl_state_history_len', 5) + 2))
            self.lambda_kl_q_controller = HAKMEMQController(q_cfg_lkl_actual_init, associated_component_name="LambdaKL_Global")
            if self.am_main_process: self.logger.info(f"Separate Lambda_KL Q-Control ENABLED. Update interval: {self.lambda_kl_update_interval} global steps.")
            if hasattr(self.lambda_kl_q_controller, 'set_current_lambda_kl'): 
                self.lambda_kl_q_controller.set_current_lambda_kl(self.lambda_kl)

        self.interval_metrics_accum = defaultdict(float)
        self.interval_steps_count = 0
        self.min_lambda_kl_q_control = getattr(args, 'min_lambda_kl_q_control', 1e-7)
        self.max_lambda_kl_q_control = getattr(args, 'max_lambda_kl_q_control', 0.5)

        self.enable_heuristic_interventions = getattr(args, 'enable_heuristic_interventions', True)
        self.enable_heuristic_disc_switching = args.enable_heuristic_disc_switching
        self.heuristic_check_interval = args.heuristic_check_interval if args.heuristic_check_interval is not None else \
                                        (args.disc_switch_check_interval if self.enable_heuristic_disc_switching else args.log_interval)
        
        self.disc_switch_min_steps_between = args.disc_switch_min_steps_between
        self.disc_switch_problem_state_count_thresh = args.disc_switch_problem_state_count_thresh
        self.steps_since_last_d_switch = 0
        self.consecutive_trigger_primary_to_alt_count = 0 
        self.consecutive_trigger_alt_to_primary_count = 0
        self.consecutive_heuristic_trigger_counts = defaultdict(int)

        self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS = getattr(args, 'heuristic_short_term_history_len', 7)
        self.q_data_derived_g_recon_hist = deque(maxlen=self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS)
        self.avg_g_recon_hist_for_stagnation = self.q_data_derived_g_recon_hist 
        self.rec_dct_stagnant = False 

        self.D_STRONG_THRESH = getattr(args, 'heuristic_d_strong_thresh', 0.25) 
        self.D_WEAK_THRESH = getattr(args, 'heuristic_d_weak_thresh', 1.0)    
        self.D_VERY_WEAK_THRESH = getattr(args, 'heuristic_d_very_weak_thresh', 1.8) 
        self.G_STALLED_THRESH = getattr(args, 'heuristic_g_stalled_thresh', 1.5) 
        self.G_WINNING_THRESH = getattr(args, 'heuristic_g_winning_thresh', 0.2) 
        self.G_VERY_MUCH_WINNING_THRESH = getattr(args, 'heuristic_g_very_much_winning_thresh', 0.05)
        self.KL_HIGH_THRESH = getattr(args, 'heuristic_kl_high_thresh', 25.0) 
        self.RECON_STAGNATION_IMPROVEMENT_THRESH_REL = getattr(args, 'heuristic_recon_stagnation_improvement_thresh_rel', 0.001)
        self.TARGET_GOOD_RECON_THRESH_HEURISTIC = getattr(args, 'target_good_recon_thresh_heuristic', 0.03) 
        self.Q_REWARD_STAGNATION_THRESH = getattr(args, 'heuristic_q_reward_stagnation_thresh', -0.25)
        self.HEURISTIC_TRIGGER_COUNT_THRESH = getattr(args, 'heuristic_trigger_count_thresh', 2) 

        self.heuristic_vae_feature_match_active = False
        self.heuristic_penalize_g_easy_win_active = False
        self.heuristic_boost_active_d_lr_active = False
        self.heuristic_force_d_q_explore_active = False

        self.heuristic_override_lambda_recon_factor = 1.0
        self.heuristic_override_lambda_kl_factor = 1.0 
        self.lambda_feat_match_heuristic = getattr(args, 'lambda_feat_match_heuristic', 0.75)
        self.lambda_g_easy_win_penalty_heuristic = getattr(args, 'lambda_g_easy_win_penalty_heuristic', 1.5)
        self.heuristic_active_d_lr_boost_factor = getattr(args, 'heuristic_active_d_lr_boost_factor', 1.8)
        self.heuristic_d_q_explore_boost_epsilon = getattr(args, 'heuristic_d_q_explore_boost_epsilon', 0.7)
        self.heuristic_d_q_explore_duration = getattr(args, 'heuristic_d_q_explore_duration', 10)

        self.tb_writer = None
        if self.am_main_process and TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            # Create a unique run name for TensorBoard if WandB is not used or fails
            tb_run_name = datetime.now().strftime('%y%m%d_%H%M%S_local')
            if args.wandb and WANDB_AVAILABLE and wandb.run and wandb.run.name:
                tb_run_name = wandb.run.name 
            
            tb_log_dir_path = Path(self.args.checkpoint_dir) / "tensorboard_logs_v020" / tb_run_name
            self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir_path))
            self.logger.info(f"TensorBoard logging to: {tb_log_dir_path}")

        self.rng_states_for_checkpoint = {}

        if self.am_main_process:
             self.logger.info(f"HybridTrainer fully initialized. Initial Active D: '{self.active_discriminator_key}' (Type: '{self.active_disc_actual_type}'). Heuristics {'ENABLED' if self.enable_heuristic_interventions else 'DISABLED'}.")
        self._sync_lambda_kl_to_all_q_controllers()


    def _sync_lambda_kl_to_all_q_controllers(self):
        """Updates the internal lambda_kl value for all relevant Q-Controllers."""
        controllers_to_update_sync = [
            self.q_controller_gen, 
            self.q_controller_d_primary, 
            self.q_controller_d_alt, 
            self.lambda_kl_q_controller
        ]
        effective_l_kl_sync = self.lambda_kl * self.heuristic_override_lambda_kl_factor
        for qc_sync in controllers_to_update_sync:
            if qc_sync and hasattr(qc_sync, 'set_current_lambda_kl'):
                qc_sync.set_current_lambda_kl(effective_l_kl_sync) 

    def _update_active_discriminator_pointers(self):
        if self.active_discriminator_key == 'primary':
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_actual_type = self.primary_disc_actual_type
            self.q_controller_d_active = self.q_controller_d_primary
        elif self.active_discriminator_key == 'alternative':
            self.active_discriminator = self.discriminator_alternative_obj
            self.optimizer_disc_active = self.optimizer_disc_alternative
            self.active_disc_actual_type = self.alternative_disc_actual_type
            self.q_controller_d_active = self.q_controller_d_alt
        else: 
            self.logger.error(f"CRITICAL: Invalid active_discriminator_key: {self.active_discriminator_key}. Defaulting to primary.")
            self.active_discriminator_key = 'primary' # Fallback
            self.active_discriminator = self.discriminator_primary_obj
            self.optimizer_disc_active = self.optimizer_disc_primary
            self.active_disc_actual_type = self.primary_disc_actual_type
            self.q_controller_d_active = self.q_controller_d_primary
        
        if self.am_main_process: # Ensure logger is used after it's confirmed to exist
            getattr(self, 'logger', logging.getLogger(f"{logger.name}.Trainer")).info(
                f"Active Discriminator set to: '{self.active_discriminator_key}' (Actual Type: '{self.active_disc_actual_type}')"
            )


    def _get_global_context_raw_features(self) -> Optional[torch.Tensor]:
        if not (self.global_ctx_cfg.use_epoch_frac or self.global_ctx_cfg.use_gstep_frac):
            return None
        features_ctx: List[float] = []
        if self.global_ctx_cfg.use_epoch_frac:
            epoch_frac_ctx = (float(self.current_epoch) + 1.0) / max(1.0, float(self.args.epochs))
            features_ctx.append(epoch_frac_ctx)
        if self.global_ctx_cfg.use_gstep_frac:
            batches_per_epoch_est_ctx = float(len(self.train_loader)) if self.train_loader and len(self.train_loader) > 0 else 500.0
            total_expected_gsteps_ctx = (float(self.args.epochs) * batches_per_epoch_est_ctx) / max(1.0, float(self.grad_accum_steps))
            gstep_frac_ctx = float(self.global_step) / max(1.0, total_expected_gsteps_ctx)
            features_ctx.append(gstep_frac_ctx)
        
        if not features_ctx: return None
        # Ensure output is (1, D_raw_ctx) for nn.Linear in embed_layer
        return torch.tensor(features_ctx, device=self.device, dtype=torch.float32).unsqueeze(0)


    def _get_wubu_interpretability_data_for_logging(self, global_ctx_raw_feats: Optional[torch.Tensor]) -> Dict[str, Any]:
        log_data_interp: Dict[str, Any] = {}
        m_ref_interp_log = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        
        if hasattr(m_ref_interp_log, 'get_interpretability_data_vae'):
            vae_interp_data_log = m_ref_interp_log.get_interpretability_data_vae(global_ctx_raw_feats)
            for stack_key_log, stack_interp_data_log in vae_interp_data_log.items(): 
                # stack_key_log is e.g. "encoder_wubu_s_stack"
                # stack_interp_data_log contains "levels_interp" and "transforms_interp"
                if "levels_interp" in stack_interp_data_log:
                    for level_idx_log, level_data_log in enumerate(stack_interp_data_log["levels_interp"]):
                        for param_name_log, val_log in level_data_log.items():
                            log_data_interp[f"wubu_interp/{stack_key_log}/L{level_idx_log}/{param_name_log}"] = val_log
                if "transforms_interp" in stack_interp_data_log:
                     for trans_idx_log, trans_data_log in enumerate(stack_interp_data_log["transforms_interp"]):
                        for param_name_log, val_log in trans_data_log.items():
                            log_data_interp[f"wubu_interp/{stack_key_log}/T{trans_idx_log}/{param_name_log}"] = val_log

        disc_map_interp_log = {
            "disc_primary": (self.discriminator_primary_obj, self.disc_pri_specific_cfg.get("input_type")),
            "disc_alt": (self.discriminator_alternative_obj, self.disc_alt_specific_cfg.get("input_type"))
        }
        for disc_name_log, (disc_obj_log, disc_type_log) in disc_map_interp_log.items():
            if disc_type_log == "dct": 
                d_ref_interp_log = disc_obj_log.module if self.ddp_active and hasattr(disc_obj_log, 'module') else disc_obj_log
                if hasattr(d_ref_interp_log, 'get_interpretability_data_discriminator'):
                    disc_interp_data_outer_log = d_ref_interp_log.get_interpretability_data_discriminator(global_ctx_raw_feats) # type: ignore
                    if "wubu_d_stack_interpretability" in disc_interp_data_outer_log:
                        stack_interp_data_d_log = disc_interp_data_outer_log["wubu_d_stack_interpretability"]
                        if "levels_interp" in stack_interp_data_d_log:
                             for level_idx_d_log, level_data_d_log in enumerate(stack_interp_data_d_log["levels_interp"]):
                                for param_name_d_log, val_d_log in level_data_d_log.items():
                                    log_data_interp[f"wubu_interp/{disc_name_log}_wubu_d/L{level_idx_d_log}/{param_name_d_log}"] = val_d_log
                        if "transforms_interp" in stack_interp_data_d_log:
                            for trans_idx_d_log, trans_data_d_log in enumerate(stack_interp_data_d_log["transforms_interp"]):
                                for param_name_d_log, val_d_log in trans_data_d_log.items():
                                    log_data_interp[f"wubu_interp/{disc_name_log}_wubu_d/T{trans_idx_d_log}/{param_name_d_log}"] = val_d_log
        return log_data_interp


    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_div_comp = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_div_comp.mean()

    def _compute_recon_loss(self, recon_norm_dcts: torch.Tensor, target_norm_dcts: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are same dtype for mse_loss
        return F.mse_loss(recon_norm_dcts.to(target_norm_dcts.dtype), target_norm_dcts)

    @torch.no_grad()
    def _log_samples_to_wandb(self, tag_prefix: str, mel_spectrograms_to_log: Optional[torch.Tensor],
                              num_sequences_to_log_max: int = 2):
        if not (self.am_main_process and self.args.wandb and WANDB_AVAILABLE and wandb is not None and wandb.run): return
        if mel_spectrograms_to_log is None or mel_spectrograms_to_log.numel() == 0: 
            self.logger.debug(f"Skipping WandB image log for {tag_prefix} due to None or empty data.")
            return

        B_log_wandb, C_log_wandb, H_log_wandb, W_log_wandb = mel_spectrograms_to_log.shape
        if C_log_wandb != 1: 
            self.logger.warning(f"Mel spectrograms for logging ({tag_prefix}) have {C_log_wandb} channels, expected 1. Taking first channel.")
            mel_spectrograms_to_log = mel_spectrograms_to_log[:,0:1,:,:]

        num_to_actually_log_wandb = min(B_log_wandb, num_sequences_to_log_max)
        wandb_images_for_log_final = []
        
        for b_idx_wandb in range(num_to_actually_log_wandb):
            mel_tensor_wandb = mel_spectrograms_to_log[b_idx_wandb, 0, ...].cpu().float() # Ensure float for display
            fig_iter_wandb, ax_iter_wandb = None, None
            try:
                if MATPLOTLIB_AVAILABLE and plt is not None and librosa is not None and librosa.display is not None:
                    aspect_ratio_wandb = W_log_wandb / H_log_wandb if H_log_wandb > 0 and W_log_wandb > 0 else 1.0
                    fig_width_wandb = max(5, min(15, int(5 * aspect_ratio_wandb))); fig_height_wandb = max(4, min(10, int(fig_width_wandb / aspect_ratio_wandb if aspect_ratio_wandb > 0 else fig_width_wandb)))
                    fig_iter_wandb, ax_iter_wandb = plt.subplots(1, 1, figsize=(fig_width_wandb, fig_height_wandb))
                    
                    fmax_val_wandb = self.args.fmax if self.args.fmax is not None and self.args.fmax > self.args.fmin else self.args.sample_rate / 2.0
                    img_wandb = librosa.display.specshow(mel_tensor_wandb.numpy(), ax=ax_iter_wandb,
                                             sr=self.args.sample_rate, hop_length=self.args.hop_length,
                                             x_axis='time', y_axis='mel', fmin=self.args.fmin, fmax=fmax_val_wandb, cmap='magma')
                    fig_iter_wandb.colorbar(img_wandb, ax=ax_iter_wandb, format='%+.2f (norm val)')
                    ax_iter_wandb.set_title(f"{tag_prefix} S{b_idx_wandb} Ep{self.current_epoch+1} GStep{self.global_step}")
                    wandb_images_for_log_final.append(wandb.Image(fig_iter_wandb))
                else: raise RuntimeError("Matplotlib/Librosa display unavailable for logging WandB image.")
            except Exception as e_disp_wandb:
                self.logger.debug(f"Librosa display failed for {tag_prefix} S{b_idx_wandb}: {e_disp_wandb}. Falling back to raw image for WandB.")
                img_0_1_wandb = (mel_tensor_wandb.clamp(-1,1) + 1) / 2.0 
                caption_wandb = f"{tag_prefix} S{b_idx_wandb} Ep{self.current_epoch+1} GStep{self.global_step} (raw_fallback)"
                wandb_images_for_log_final.append(wandb.Image(img_0_1_wandb, caption=caption_wandb))
            finally:
                if fig_iter_wandb is not None and plt is not None: plt.close(fig_iter_wandb)

        if wandb_images_for_log_final:
            try:
                wandb.log({f"samples_mel/{tag_prefix}": wandb_images_for_log_final}, step=self.global_step)
            except Exception as e_wandb_log_final:
                self.logger.error(f"Failed to log images to WandB for {tag_prefix}: {e_wandb_log_final}", exc_info=True)


    def _train_discriminator_step(self, real_mel_spectrograms: torch.Tensor,
                                  m_ref: "WuBuSpecTransNet", # DDP unwrapped model
                                  global_ctx_raw_features: Optional[torch.Tensor]
                                 ) -> Dict[str, torch.Tensor]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        B = real_mel_spectrograms.shape[0]; device = real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters())).dtype

        real_labels = torch.ones(B, device=device, dtype=dtype_model) # For BCEWithLogitsLoss if logits are (B)
        fake_labels = torch.zeros(B, device=device, dtype=dtype_model)
        losses_d_micro: Dict[str, torch.Tensor] = {}

        for p in d_ref_active.parameters(): p.requires_grad = True
        for p in m_ref.parameters(): p.requires_grad = False
        self.optimizer_disc_active.zero_grad(set_to_none=True) # Use active optimizer

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            # Real data pass
            real_input_for_d: torch.Tensor
            if d_ref_active.input_type == "mel": # type: ignore
                real_input_for_d = real_mel_spectrograms.to(device, dtype=dtype_model)
                real_logits = d_ref_active(real_input_for_d, global_context_raw_features=global_ctx_raw_features) # Pass context if D uses it (WuBu-D)
            elif d_ref_active.input_type == "dct": # type: ignore
                with torch.no_grad(): # Encoder part of VAE does not need grads for D step
                    # Pass global_ctx for encoder's WuBu-S if it's dynamic
                    _, _, real_norm_dcts_target, _ = m_ref.encode(real_mel_spectrograms.to(device, dtype=dtype_model), global_ctx_raw_features)
                real_input_for_d = real_norm_dcts_target.to(device, dtype=dtype_model)
                real_logits = d_ref_active(real_input_for_d, global_context_raw_features=global_ctx_raw_features)
            else: raise ValueError(f"Unsupported D input type: {d_ref_active.input_type}") # type: ignore
            loss_d_real = self.adversarial_loss(real_logits.squeeze(), real_labels)

            # Fake data pass (using VAE reconstruction G(E(X)))
            with torch.no_grad(): # VAE does not need grads for D step here
                # Pass global_ctx for encoder's WuBu-S and generator's WuBu-G
                fake_norm_dct_coeffs, _, _, gaad_bboxes_for_assembly, _ = m_ref(real_mel_spectrograms.to(device, dtype=dtype_model), global_ctx_raw_features)

            fake_input_for_d: torch.Tensor
            if d_ref_active.input_type == "mel": # type: ignore
                unnorm_fake_dcts = AudioSpecGenerator._unnormalize_dct(fake_norm_dct_coeffs.detach(), self.args)
                # _assemble_mel_from_dct_regions is a method of AudioSpecDiscriminator
                fake_mel_input_for_d = d_ref_active._assemble_mel_from_dct_regions(unnorm_fake_dcts, gaad_bboxes_for_assembly.detach(), real_mel_spectrograms.shape) # type: ignore
                fake_logits = d_ref_active(fake_mel_input_for_d.detach(), global_context_raw_features=global_ctx_raw_features)
            elif d_ref_active.input_type == "dct": # type: ignore
                fake_input_for_d = fake_norm_dct_coeffs.to(device, dtype=dtype_model).detach()
                fake_logits = d_ref_active(fake_input_for_d, global_context_raw_features=global_ctx_raw_features)
            else: raise ValueError(f"Unsupported D input type (fake): {d_ref_active.input_type}") # type: ignore
            loss_d_fake = self.adversarial_loss(fake_logits.squeeze(), fake_labels)
            
            loss_d_total_micro = (loss_d_real + loss_d_fake) * 0.5
            loss_d_total_scaled_for_accum_micro = loss_d_total_micro / self.grad_accum_steps

        self.scaler_disc.scale(loss_d_total_scaled_for_accum_micro).backward()

        losses_d_micro['loss_d_real_micro'] = loss_d_real.detach()
        losses_d_micro['loss_d_fake_micro'] = loss_d_fake.detach()
        losses_d_micro['loss_d_total_micro'] = loss_d_total_micro.detach()
        return losses_d_micro


    def _train_generator_step(self, real_mel_spectrograms: torch.Tensor,
                              m_ref: "WuBuSpecTransNet", # DDP unwrapped model
                              global_ctx_raw_features: Optional[torch.Tensor]
                             ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        d_ref_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        B = real_mel_spectrograms.shape[0]; device = real_mel_spectrograms.device
        dtype_model = next(iter(m_ref.parameters())).dtype
        real_labels_for_g = torch.ones(B, device=device, dtype=dtype_model) # For BCEWithLogitsLoss
        losses_g_micro: Dict[str, torch.Tensor] = {}
        recon_mel_for_log: Optional[torch.Tensor] = None

        for p in d_ref_active.parameters(): p.requires_grad = False
        for p in m_ref.parameters(): p.requires_grad = True
        self.optimizer_enc_gen.zero_grad(set_to_none=True)

        with amp.autocast(device_type=self.device.type, enabled=self.args.use_amp):
            # Pass global_ctx to VAE (model forward)
            recon_norm_dct_coeffs, mu, logvar, gaad_bboxes_from_enc, target_norm_dct_coeffs = \
                m_ref(real_mel_spectrograms.to(device,dtype=dtype_model), global_ctx_raw_features)

            loss_recon_raw = self._compute_recon_loss(recon_norm_dct_coeffs, target_norm_dct_coeffs)
            loss_kl_raw = self._compute_kl_loss(mu, logvar)
            
            # Effective lambda weights (can be modulated by heuristics)
            current_effective_lambda_kl = self.lambda_kl * self.heuristic_override_lambda_kl_factor
            current_effective_lambda_recon = self.lambda_recon * self.heuristic_override_lambda_recon_factor

            loss_recon_eff = current_effective_lambda_recon * loss_recon_raw
            loss_kl_eff = current_effective_lambda_kl * loss_kl_raw
            
            fake_logits_for_g: torch.Tensor
            features_for_g_feat_match: Optional[torch.Tensor] = None

            if d_ref_active.input_type == "mel": # type: ignore
                unnorm_recon_dcts_for_adv = AudioSpecGenerator._unnormalize_dct(recon_norm_dct_coeffs, self.args)
                recon_mel_for_adv = d_ref_active._assemble_mel_from_dct_regions(unnorm_recon_dcts_for_adv, gaad_bboxes_from_enc, real_mel_spectrograms.shape) # type: ignore
                if self.heuristic_vae_feature_match_active:
                    # Pass context to D if it uses WuBu
                    output_d = d_ref_active(recon_mel_for_adv, return_features=True, global_context_raw_features=global_ctx_raw_features)
                    fake_logits_for_g, features_for_g_feat_match = output_d if isinstance(output_d, tuple) else (output_d, None)
                else:
                    fake_logits_for_g = d_ref_active(recon_mel_for_adv, return_features=False, global_context_raw_features=global_ctx_raw_features) # type: ignore
                # Logging recon_mel_for_adv (assembled from G's DCTs)
                if self.am_main_process and self.args.wandb_log_train_recon_interval > 0 and \
                   ((self.global_step + 1) % self.args.wandb_log_train_recon_interval == 0 or self.global_step == 0):
                    recon_mel_for_log = recon_mel_for_adv.detach().clone()
            elif d_ref_active.input_type == "dct": # type: ignore
                if self.heuristic_vae_feature_match_active:
                    output_d = d_ref_active(recon_norm_dct_coeffs, return_features=True, global_context_raw_features=global_ctx_raw_features) # type: ignore
                    fake_logits_for_g, features_for_g_feat_match = output_d if isinstance(output_d, tuple) else (output_d, None)
                else:
                    fake_logits_for_g = d_ref_active(recon_norm_dct_coeffs, return_features=False, global_context_raw_features=global_ctx_raw_features) # type: ignore
            else: raise ValueError(f"Unsupported D input type for G: {d_ref_active.input_type}") # type: ignore

            loss_g_adv_raw = self.adversarial_loss(fake_logits_for_g.squeeze(), real_labels_for_g)
            loss_g_adv_eff = self.lambda_gan * loss_g_adv_raw
            loss_g_total_micro = loss_recon_eff + loss_kl_eff + loss_g_adv_eff

            # VAE Feature Matching (Heuristic)
            if self.heuristic_vae_feature_match_active and features_for_g_feat_match is not None:
                with torch.no_grad(): # Target features from D(real_data)
                    target_d_output_for_feat_match: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                    if d_ref_active.input_type == "mel": # type: ignore
                        target_d_output_for_feat_match = d_ref_active(real_mel_spectrograms.to(device, dtype=dtype_model), return_features=True, global_context_raw_features=global_ctx_raw_features) # type: ignore
                    else: # DCT input for D
                        target_d_output_for_feat_match = d_ref_active(target_norm_dct_coeffs.to(device, dtype=dtype_model).detach(), return_features=True, global_context_raw_features=global_ctx_raw_features) # type: ignore
                    target_features_d = target_d_output_for_feat_match[1] if isinstance(target_d_output_for_feat_match, tuple) else None
                
                if target_features_d is not None:
                    # Ensure feature shapes match for MSE (e.g., by GAP if they are maps)
                    if features_for_g_feat_match.ndim > 2 and target_features_d.ndim > 2 : 
                        features_for_g_feat_match = torch.mean(features_for_g_feat_match, dim=list(range(2, features_for_g_feat_match.ndim)))
                        target_features_d = torch.mean(target_features_d, dim=list(range(2, target_features_d.ndim)))
                    
                    if features_for_g_feat_match.shape == target_features_d.shape:
                        loss_g_feat_match = F.mse_loss(features_for_g_feat_match, target_features_d.detach())
                        loss_g_total_micro += self.lambda_feat_match_heuristic * loss_g_feat_match
                        losses_g_micro['loss_g_feat_match_micro'] = loss_g_feat_match.detach()
                    else: self.logger.warning(f"Feat Match shapes mismatch: G_feat {features_for_g_feat_match.shape}, D_feat_target {target_features_d.shape}")

            # Penalize G for Easy Wins (Heuristic)
            if self.heuristic_penalize_g_easy_win_active:
                if loss_g_adv_raw.item() < self.G_WINNING_THRESH and loss_recon_raw.item() > self.TARGET_GOOD_RECON_THRESH_HEURISTIC:
                    # Penalty proportional to how bad recon is, scaled by how easy adv win was
                    penalty_factor = (loss_recon_raw.item() - self.TARGET_GOOD_RECON_THRESH_HEURISTIC) / (loss_g_adv_raw.item() + EPS)
                    penalty_g_easy_win_val = penalty_factor * self.lambda_g_easy_win_penalty_heuristic
                    loss_g_total_micro += penalty_g_easy_win_val # Add to total loss
                    losses_g_micro['loss_g_easy_win_penalty_micro'] = torch.tensor(penalty_g_easy_win_val, device=device, dtype=dtype_model)

            loss_g_total_scaled_for_accum_micro = loss_g_total_micro / self.grad_accum_steps

        self.scaler_enc_gen.scale(loss_g_total_scaled_for_accum_micro).backward()

        losses_g_micro['loss_recon_micro'] = loss_recon_raw.detach()
        losses_g_micro['loss_kl_micro'] = loss_kl_raw.detach()
        losses_g_micro['loss_g_adv_micro'] = loss_g_adv_raw.detach()
        losses_g_micro['loss_g_total_micro'] = loss_g_total_micro.detach()
        return losses_g_micro, recon_mel_for_log



    def _get_q_controller_data_for_heuristics(self) -> Dict[str, Any]: 
        q_data: Dict[str, Any] = {'gen': {'is_valid': False}, 'active_d': {'is_valid': False}, 'lkl': {'is_valid': False}}
        controllers_map = { 'gen': self.q_controller_gen, 'active_d': self.q_controller_d_active, 'lkl': self.lambda_kl_q_controller }
        hist_names_g_d = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake', 
                          'wubu_avg_curvature', 'wubu_avg_scale', 'wubu_var_curvature'] # Added WuBu hists
        hist_names_lkl = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']

        for key, controller in controllers_map.items():
            if controller:
                q_data[key]['is_valid'] = True
                q_data[key]['epsilon'] = controller.epsilon
                q_data[key]['current_q_alpha'] = controller.current_alpha # For logging
                q_data[key]['current_q_gamma'] = controller.current_gamma # For logging
                q_data[key]['on_probation'] = getattr(controller, 'on_probation', False) or \
                                              (hasattr(controller, 'lkl_on_probation') and getattr(controller, 'lkl_on_probation', False))
                
                reward_hist_to_use = controller.reward_hist
                q_data[key]['reward_median_short_term'] = np.median(list(reward_hist_to_use)[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:]) \
                                                          if len(reward_hist_to_use) >= self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS else \
                                                          (np.median(list(reward_hist_to_use)) if reward_hist_to_use else 0.0)
                
                current_hist_names_set = hist_names_g_d if key in ['gen', 'active_d'] else hist_names_lkl
                hist_prefix_map = {'gen': "loss_", 'active_d': "loss_", 'lkl': "interval_"}
                # Handle WuBu specific prefix for gen
                if key == 'gen': hist_prefix_map['gen_wubu'] = "" # No prefix for wubu_avg_curvature_hist etc.

                for lname in current_hist_names_set:
                    hist_attr_name_candidate = f"{hist_prefix_map.get(key, 'loss_')}{lname}_hist"
                    if "wubu" in lname and key=='gen': # Special handling for wubu hists in gen Q-Ctrl
                        hist_attr_name_candidate = f"{lname}_hist" # e.g., wubu_avg_curvature_hist

                    if hasattr(controller, hist_attr_name_candidate):
                        hist_deque = getattr(controller, hist_attr_name_candidate)
                        val_for_trend = list(hist_deque)[-1] if hist_deque else None
                        median_val = None
                        if len(hist_deque) >= self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:
                            median_val = np.median(list(hist_deque)[-self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS:])
                        elif hist_deque: median_val = np.median(list(hist_deque))
                        
                        q_data[key][f"{lname}_median_short_term"] = median_val
                        q_data[key][f"{lname}_trend_short_term"] = controller._get_trend_bin(hist_deque, val_for_trend) if val_for_trend is not None else 2 # Neutral trend if no current val
        
        # Update global recon stagnation flag
        if q_data['gen']['is_valid'] and q_data['gen'].get('g_recon_median_short_term') is not None:
            self.q_data_derived_g_recon_hist.append(q_data['gen']['g_recon_median_short_term'])
            if len(self.q_data_derived_g_recon_hist) >= max(2, self.SHORT_TERM_LOSS_HISTORY_LEN_FOR_HEURISTICS // 2):
                current_recon = self.q_data_derived_g_recon_hist[-1]
                past_recon_avg = np.mean(list(self.q_data_derived_g_recon_hist)[:-1]) if len(self.q_data_derived_g_recon_hist) > 1 else current_recon
                self.rec_dct_stagnant = (past_recon_avg - current_recon) < (past_recon_avg * self.RECON_STAGNATION_IMPROVEMENT_THRESH_REL)
            else: self.rec_dct_stagnant = False
        else: self.rec_dct_stagnant = False # Default if no data

        return q_data

    def _evaluate_training_state_and_apply_heuristics(self, current_q_data: Dict[str, Any]):
        if not self.am_main_process or not self.enable_heuristic_interventions:
            # Reset all flags and factors if heuristics are globally disabled or not on main process
            self.heuristic_vae_feature_match_active = False
            self.heuristic_penalize_g_easy_win_active = False
            self.heuristic_boost_active_d_lr_active = False
            # self.heuristic_force_d_q_explore_active is a trigger, so no need to reset its "active" state here
            self.heuristic_override_lambda_recon_factor = 1.0
            self.heuristic_override_lambda_kl_factor = 1.0
            return

        gen_q, active_d_q = current_q_data.get('gen', {}), current_q_data.get('active_d', {})
        log_msgs_heuristics = [] # Collect messages for this heuristic evaluation cycle

        # Persist flags that should remain active until explicitly turned off by a counter-condition
        current_penalize_g_easy_win = self.heuristic_penalize_g_easy_win_active 
        current_vae_feature_match = self.heuristic_vae_feature_match_active
        
        # Factors and triggers are typically reset/re-evaluated each cycle unless logic dictates persistence
        current_lambda_recon_factor = 1.0 
        current_lambda_kl_factor = 1.0    
        current_boost_active_d_lr = False
        current_force_d_q_explore_trigger = False # This is a trigger for one-time action

        # Extract key conditions from Q-data
        g_adv_median = gen_q.get('g_adv_median_short_term', 0.7) # Default to neutral if missing
        d_total_median = active_d_q.get('d_total_median_short_term', 0.7)
        d_q_reward_median = active_d_q.get('reward_median_short_term', 0.0)
        is_d_q_on_probation = active_d_q.get('on_probation', True) # Assume probation if data missing (conservative)
        
        is_g_dominating_very_much = g_adv_median < self.G_VERY_MUCH_WINNING_THRESH
        is_d_very_weak = d_total_median > self.D_VERY_WEAK_THRESH
        is_d_q_learner_stagnant = d_q_reward_median < self.Q_REWARD_STAGNATION_THRESH and not is_d_q_on_probation
        is_d_strong = d_total_median < self.D_STRONG_THRESH
        is_g_stalled_adv = g_adv_median > self.G_STALLED_THRESH

        # --- 1. Discriminator Switch (Highest Priority if enabled) ---
        switched_d_this_cycle = False
        if self.enable_heuristic_disc_switching:
            switched_d_this_cycle = self._check_and_perform_disc_switch(
                is_g_dominating_adv=is_g_dominating_very_much, is_d_weak_overall=is_d_very_weak,
                is_d_struggling_q=is_d_q_learner_stagnant, is_d_strong_overall=is_d_strong,
                is_g_stalled_adv=is_g_stalled_adv,
                current_g_kl_median = gen_q.get('g_kl_median_short_term', 0.0), # Raw KL
                log_msgs_ref = log_msgs_heuristics # Pass list to append messages
            )
        
        if switched_d_this_cycle:
            # Reset other heuristic counters and potentially persistent flags if D changed
            self.consecutive_heuristic_trigger_counts = defaultdict(int)
            current_penalize_g_easy_win = False # Reset these on D switch
            current_vae_feature_match = False
        else:
            # --- 2. If No D-Switch, Evaluate Other Heuristics ---
            
            # A. GAN Rebalancing (G dominating, D weak/stagnant, VAE recon also stagnant)
            condition_gan_rebalance = is_g_dominating_very_much and \
                                      (is_d_very_weak or is_d_q_learner_stagnant) and \
                                      self.rec_dct_stagnant # VAE recon needs help too

            if condition_gan_rebalance:
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] += 1
                if self.consecutive_heuristic_trigger_counts['gan_rebalance'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_penalize_g_easy_win = True # Persists until GAN rebalances
                    current_lambda_recon_factor = self.args.heuristic_recon_boost_factor # Temp boost
                    if is_d_q_learner_stagnant: # If D's Q-learner is also stuck
                        current_boost_active_d_lr = True # Temp boost
                        current_force_d_q_explore_trigger = True # One-time trigger
                    log_msgs_heuristics.append(f"HEURISTIC: GAN REBALANCE ACTIVE. PenalizeG:{current_penalize_g_easy_win}, LRecFactor:{current_lambda_recon_factor:.2f}, D_LR_Boost:{current_boost_active_d_lr}, D_Q_ExploreTrig:{current_force_d_q_explore_trigger}")
            else: # Condition not met, decrement counter towards turning off persistent flags
                self.consecutive_heuristic_trigger_counts['gan_rebalance'] = 0
                if self.heuristic_penalize_g_easy_win_active: # If it was on, turn it off
                    current_penalize_g_easy_win = False
                    log_msgs_heuristics.append("HEURISTIC: GAN REBALANCE condition no longer met. PenalizeG turned OFF.")


            # B. VAE Feature Matching Boost (D is strong/healthy, but VAE recon is stagnant)
            #    This means D has good features, VAE should try to match them.
            condition_vae_feat_match = (not is_g_dominating_very_much and not is_d_very_weak and 
                                        (is_d_strong or not is_d_q_learner_stagnant) and # D is doing okay
                                        self.rec_dct_stagnant) # But VAE recon is stuck

            if condition_vae_feat_match:
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] += 1
                if self.consecutive_heuristic_trigger_counts['vae_feat_match'] >= self.HEURISTIC_TRIGGER_COUNT_THRESH:
                    current_vae_feature_match = True # Persists
                    # If KL is already very low, give it a slight nudge up to allow more expressive latent space
                    # for better feature matching.
                    effective_kl_weight = self.lambda_kl * self.heuristic_override_lambda_kl_factor # Use current effective
                    if effective_kl_weight < self.args.min_lambda_kl_q_control * 5: # If KL weight is tiny
                        current_lambda_kl_factor = 1.5 # Temporarily boost base KL factor
                    log_msgs_heuristics.append(f"HEURISTIC: VAE FEATURE MATCH ACTIVE. LKL_Factor:{current_lambda_kl_factor:.2f}")
            else:
                self.consecutive_heuristic_trigger_counts['vae_feat_match'] = 0
                if self.heuristic_vae_feature_match_active: # If it was on, turn it off
                    current_vae_feature_match = False
                    log_msgs_heuristics.append("HEURISTIC: VAE FEATURE MATCH condition no longer met. Turned OFF.")
            
            # TODO: Add more heuristics from the paper's "Adaptive Strain Engineering" concept if time permits
            # e.g., direct modulation of WuBu params based on Q-data (very advanced)

        # --- Update Global Heuristic State Flags and Factors based on decisions this cycle ---
        self.heuristic_penalize_g_easy_win_active = current_penalize_g_easy_win
        self.heuristic_override_lambda_recon_factor = current_lambda_recon_factor # This is often temporary
        self.heuristic_boost_active_d_lr_active = current_boost_active_d_lr # This is often temporary
        self.heuristic_vae_feature_match_active = current_vae_feature_match
        self.heuristic_override_lambda_kl_factor = current_lambda_kl_factor # This can be temporary

        # Apply one-time triggers
        if current_force_d_q_explore_trigger and self.q_controller_d_active:
            self.q_controller_d_active.force_exploration_boost(
                duration_steps=self.heuristic_d_q_explore_duration,
                boost_epsilon_to=self.heuristic_d_q_explore_boost_epsilon
            )
            log_msgs_heuristics.append(f"HEURISTIC: Active D Q-Controller exploration boosted for {self.heuristic_d_q_explore_duration} steps to eps {self.heuristic_d_q_explore_boost_epsilon:.2f}.")
        
        # Log all collected heuristic messages for this cycle
        if log_msgs_heuristics and self.am_main_process:
            for msg in log_msgs_heuristics: self.logger.info(msg) # Use trainer's logger

        # After applying heuristics, sync effective lambda_kl to Q-controllers
        self._sync_lambda_kl_to_all_q_controllers()


    def _check_and_perform_disc_switch(self, 
                                         is_g_dominating_adv: bool, is_d_weak_overall: bool, is_d_struggling_q: bool,
                                         is_d_strong_overall: bool, is_g_stalled_adv: bool,
                                         current_g_kl_median: float, # Raw KL median
                                         log_msgs_ref: List[str]) -> bool:
        if not self.enable_heuristic_disc_switching or self.steps_since_last_d_switch < self.disc_switch_min_steps_between:
            return False

        switched_this_check = False
        # Effective KL weight for check = base_lambda_kl * heuristic_factor
        effective_kl_weight_for_check = self.lambda_kl * self.heuristic_override_lambda_kl_factor
        
        # Condition A: D is too strong, G is stalled, and VAE is either high KL or stagnant recon.
        # This suggests the current D might be too easy for G to fool (if G_adv still high) or too hard,
        # and G cannot improve VAE aspects. Try a different D.
        condition_A = (is_d_strong_overall and is_g_stalled_adv and 
                       (effective_kl_weight_for_check * current_g_kl_median > self.KL_HIGH_THRESH or self.rec_dct_stagnant))

        if condition_A:
            self.consecutive_trigger_primary_to_alt_count += 1
            if self.consecutive_trigger_primary_to_alt_count >= self.disc_switch_problem_state_count_thresh:
                if self.active_discriminator_key == 'primary':
                    target_key = 'alternative'
                    log_msgs_ref.append(f"DISC_SWITCH_HEURISTIC (Cond A): D_strong & G_stalled_adv & (HighEffKL or ReconStagnant). Switching Primary -> Alternative.")
                    self.active_discriminator_key = target_key
                    switched_this_check = True
                # else: self.consecutive_trigger_primary_to_alt_count = 0 # Already Alt, condition not for switching back this way
        else: self.consecutive_trigger_primary_to_alt_count = 0

        # Condition B: G is dominating, current D is very weak and its Q-learner is struggling,
        # AND VAE recon is also not great (stagnant). This D isn't providing useful signal.
        condition_B = (is_g_dominating_adv and is_d_weak_overall and is_d_struggling_q and self.rec_dct_stagnant)
        
        if not switched_this_check and condition_B: # Only if not already switched by A
            self.consecutive_trigger_alt_to_primary_count += 1
            if self.consecutive_trigger_alt_to_primary_count >= self.disc_switch_problem_state_count_thresh:
                if self.active_discriminator_key == 'alternative':
                    target_key = 'primary'
                    log_msgs_ref.append(f"DISC_SWITCH_HEURISTIC (Cond B): G_dominating_adv & D_very_weak & D_Q_struggling & ReconStagnant. Switching Alternative -> Primary.")
                    self.active_discriminator_key = target_key
                    switched_this_check = True
                # else: self.consecutive_trigger_alt_to_primary_count = 0 # Already Primary
        elif not switched_this_check: self.consecutive_trigger_alt_to_primary_count = 0
        
        if switched_this_check:
            self._update_active_discriminator_pointers() # Apply the switch
            # Reset Q-controller of the newly activated D to give it a fresh start
            if self.q_controller_d_active:
                self.q_controller_d_active.reset_q_learning_state(
                    reset_q_table=True, reset_epsilon=True, 
                    context_msg=f"DSwitch to {self.active_discriminator_key}", 
                    start_probation=True
                )
            self.steps_since_last_d_switch = 0
            self.consecutive_trigger_primary_to_alt_count = 0 # Reset both counters after a switch
            self.consecutive_trigger_alt_to_primary_count = 0
            self.rec_dct_stagnant = False # Reset VAE stagnation flag as D context changed
            self.q_data_derived_g_recon_hist.clear()
            log_msgs_ref.append(f"  --> Post D-Switch: Heuristic flags & counters reset. New active D: '{self.active_discriminator_key}' (type: {self.active_disc_actual_type}).")
        return switched_this_check


    def train(self, start_epoch: int = 0, initial_global_step: int = 0): 
        self.global_step = initial_global_step
        self.current_epoch = start_epoch
        
        if self.am_main_process:
            self.logger.info(f"Starting training. Epochs: {self.args.epochs}, StartEpoch: {start_epoch}, InitialGStep: {initial_global_step}")
            self.logger.info(f"Initial Active D: {self.active_discriminator_key} (Type: {self.active_disc_actual_type})")

        # Fixed noise for sampling (if latent_dim > 0)
        if self.am_main_process and self.args.wandb_log_fixed_noise_samples_interval > 0 and \
           self.args.num_val_samples_to_log > 0 and self.fixed_noise_for_sampling is None and self.args.latent_dim > 0:
            m_ref_temp_fn = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
            default_dtype_fn = torch.float32; try: default_dtype_fn = next(iter(m_ref_temp_fn.parameters())).dtype
            except StopIteration: self.logger.warning("Model has no parameters for fixed noise dtype; using float32.")
            self.fixed_noise_for_sampling = torch.randn(self.args.num_val_samples_to_log, self.args.latent_dim, device=self.device, dtype=default_dtype_fn)
            self.logger.info(f"Created fixed noise tensor: {self.fixed_noise_for_sampling.shape} on {self.device}")

        log_interval_accum_losses: Dict[str, float] = defaultdict(float)
        log_interval_items_processed = 0
        m_ref_train = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        
        self._sync_lambda_kl_to_all_q_controllers() # Initial sync

        for epoch in range(start_epoch, self.args.epochs):
            self.current_epoch = epoch
            if self.am_main_process:
                eff_lkl = self.lambda_kl * self.heuristic_override_lambda_kl_factor
                self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} starting (L_KL_eff: {eff_lkl:.3e}, LRecF: {self.heuristic_override_lambda_recon_factor:.2f}, ActD: {self.active_discriminator_key} [{self.active_disc_actual_type}]).")
            if self.ddp_active and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            m_ref_train.train(); self.active_discriminator.train() # Set active D to train mode
            
            num_batches_epoch = len(self.train_loader)
            prog_bar = tqdm(self.train_loader, desc=f"E{epoch+1}", disable=not self.am_main_process, dynamic_ncols=True, total=num_batches_epoch)
            
            accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
            accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0
            # For WuBu geo param averaging for Q-Ctrl
            accum_wubu_avg_c_q, accum_wubu_avg_s_q, accum_wubu_var_c_q = 0.0, 0.0, 0.0
            num_wubu_param_samples_q = 0

            for batch_idx, batch_mel_segments in enumerate(prog_bar):
                batch_mel_segments = batch_mel_segments.to(self.device)
                batch_size_micro = batch_mel_segments.size(0)
                self.steps_since_last_d_switch +=1 

                # Get current global context features (epoch/step based)
                current_global_ctx_raw_features = self._get_global_context_raw_features()

                # Train Discriminator
                losses_d_micro = self._train_discriminator_step(batch_mel_segments, m_ref_train, current_global_ctx_raw_features)
                for k, v_tensor in losses_d_micro.items():
                    if torch.isfinite(v_tensor): val = v_tensor.item()
                    else: val = 100.0 # High penalty for non-finite
                    if np.isfinite(val):
                        accum_key = k.replace('_micro', '_agg'); log_interval_accum_losses[accum_key] += val * batch_size_micro
                        if k == 'loss_d_total_micro': accum_d_total_q += val; self.interval_metrics_accum['d_total'] += val
                        elif k == 'loss_d_real_micro': accum_d_real_q += val
                        elif k == 'loss_d_fake_micro': accum_d_fake_q += val
                
                # Train Generator/VAE
                losses_g_micro, recon_mel_for_logging = self._train_generator_step(batch_mel_segments, m_ref_train, current_global_ctx_raw_features)
                for k, v_tensor in losses_g_micro.items():
                    if torch.isfinite(v_tensor): val = v_tensor.item()
                    else: val = 100.0
                    if np.isfinite(val):
                        accum_key = k.replace('_micro', '_agg'); log_interval_accum_losses[accum_key] += val * batch_size_micro
                        if k == 'loss_g_total_micro': accum_g_total_q += val
                        elif k == 'loss_recon_micro': accum_g_recon_q += val; self.interval_metrics_accum['recon_dct'] += val
                        elif k == 'loss_kl_micro': accum_g_kl_q += val; self.interval_metrics_accum['kl_div'] += val
                        elif k == 'loss_g_adv_micro': accum_g_adv_q += val
                        # Store other heuristic losses if present (feat_match, easy_win_penalty)
                        elif k in ['loss_g_feat_match_micro', 'loss_g_easy_win_penalty_micro']:
                             log_interval_accum_losses[accum_key] += val * batch_size_micro


                log_interval_items_processed += batch_size_micro
                self.interval_steps_count += 1

                # Accumulate WuBu geometric params for Q-Ctrl if dynamic geometry is on in Gen's WuBu stacks
                # This requires getting interpretability data, which can be slow. Do it less frequently?
                # Or make interpretability data getter very lightweight for just C/S/Spread.
                # For now, assume it's efficient enough for Q-Ctrl state.
                if self.q_controller_gen and (self.wubu_s_enc_cfg.num_levels > 0 or self.wubu_g_gen_cfg.num_levels > 0) and \
                   any(lvl.curvature_modulator.enabled or lvl.scale_modulator.enabled for stack_cfg in [self.wubu_s_enc_cfg, self.wubu_g_gen_cfg] if stack_cfg for lvl in stack_cfg.levels_config):
                    
                    with torch.no_grad(): # No grads needed for interpretability data
                        interp_data_vae = m_ref_train.get_interpretability_data_vae(current_global_ctx_raw_features)
                    all_c, all_s = [], []
                    for stack_name_key, stack_data in interp_data_vae.items():
                        if "levels" in stack_data:
                            for level_data_interp in stack_data["levels"]:
                                if "current_c" in level_data_interp and np.isfinite(level_data_interp["current_c"]): all_c.append(level_data_interp["current_c"])
                                if "current_s" in level_data_interp and np.isfinite(level_data_interp["current_s"]): all_s.append(level_data_interp["current_s"])
                    if all_c: accum_wubu_avg_c_q += np.mean(all_c)
                    if all_s: accum_wubu_avg_s_q += np.mean(all_s)
                    if len(all_c) > 1: accum_wubu_var_c_q += np.var(all_c)
                    num_wubu_param_samples_q +=1


                # Optimizer steps and Q-learning updates after grad_accum_steps
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    # Finalize D grad stats, clip, step, update scaler, zero grad
                    if hasattr(self.optimizer_disc_active, 'grad_stats'):
                        num_disc_params = sum(p.numel() for grp in self.optimizer_disc_active.param_groups for p in grp['params'] if p.requires_grad and p.grad is not None)
                        self.optimizer_disc_active.grad_stats.finalize_step_stats(num_disc_params)
                    if self.args.global_max_grad_norm > 0:
                        self.scaler_disc.unscale_(self.optimizer_disc_active)
                        d_to_clip_step = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                        torch.nn.utils.clip_grad_norm_([p for p in d_to_clip_step.parameters() if p.grad is not None], self.args.global_max_grad_norm)
                    self.scaler_disc.step(self.optimizer_disc_active)
                    self.scaler_disc.update()
                    self.optimizer_disc_active.zero_grad(set_to_none=True)

                    # Finalize G grad stats, clip, step, update scaler, zero grad
                    if hasattr(self.optimizer_enc_gen, 'grad_stats'):
                        num_gen_params = sum(p.numel() for grp in self.optimizer_enc_gen.param_groups for p in grp['params'] if p.requires_grad and p.grad is not None)
                        self.optimizer_enc_gen.grad_stats.finalize_step_stats(num_gen_params)
                    if self.args.global_max_grad_norm > 0:
                        self.scaler_enc_gen.unscale_(self.optimizer_enc_gen)
                        torch.nn.utils.clip_grad_norm_([p for p in m_ref_train.parameters() if p.grad is not None], self.args.global_max_grad_norm)
                    self.scaler_enc_gen.step(self.optimizer_enc_gen)
                    self.scaler_enc_gen.update()
                    self.optimizer_enc_gen.zero_grad(set_to_none=True)
                    
                    self.global_step += 1

                    # Prepare average losses for Q-Controllers
                    avg_losses_for_q_step = {
                        'loss_g_total': accum_g_total_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_g_recon': accum_g_recon_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_g_kl': accum_g_kl_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_g_adv': accum_g_adv_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_d_total': accum_d_total_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_d_real': accum_d_real_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0,
                        'loss_d_fake': accum_d_fake_q / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0
                    }
                    for k_q_chk_step, v_q_chk_step in avg_losses_for_q_step.items(): 
                        if not np.isfinite(v_q_chk_step): 
                            avg_losses_for_q_step[k_q_chk_step] = 1.0 # Fallback for non-finite
                    
                    wubu_geo_params_for_q_step: Optional[Dict[str,float]] = None
                    if num_wubu_param_samples_q > 0:
                        wubu_geo_params_for_q_step = {
                            'avg_curvature': accum_wubu_avg_c_q / num_wubu_param_samples_q,
                            'avg_scale': accum_wubu_avg_s_q / num_wubu_param_samples_q,
                            'var_curvature': accum_wubu_var_c_q / num_wubu_param_samples_q, # This is avg of variances, or variance of means. Be careful.
                                                                                           # For now, assume it's sum of variances / count
                        }

                    # Update Q-Controllers
                    if self.q_controller_d_active and hasattr(self.optimizer_disc_active, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_disc_active.q_controller_update_and_set_hyperparams(
                            avg_losses_for_q_step, 
                            self.lambda_kl * self.heuristic_override_lambda_kl_factor, # Pass effective L_KL
                            # D's Q-Ctrl doesn't typically use WuBu geo params directly, unless D is WuBu based and needs it
                        )
                        # Apply LR boost if heuristic is active (this is a one-step boost)
                        if self.heuristic_boost_active_d_lr_active:
                            for group in self.optimizer_disc_active.param_groups:
                                boosted_lr = group['lr'] * self.heuristic_active_d_lr_boost_factor
                                group['lr'] = float(np.clip(boosted_lr, 1e-8, 1.0))
                            if self.am_main_process: self.logger.info(f"HEURISTIC (Applied): Active D LR boosted to {self.optimizer_disc_active.param_groups[0]['lr']:.2e}")
                            self.heuristic_boost_active_d_lr_active = False # Reset after applying


                    if self.q_controller_gen and hasattr(self.optimizer_enc_gen, 'q_controller_update_and_set_hyperparams'):
                        self.optimizer_enc_gen.q_controller_update_and_set_hyperparams(
                            avg_losses_for_q_step, 
                            self.lambda_kl * self.heuristic_override_lambda_kl_factor,
                            wubu_geo_params_for_q_step # Pass WuBu geo params to Gen's Q-Ctrl
                        )
                    
                    # Reset accumulators for Q-Ctrl inputs
                    accum_g_total_q, accum_g_recon_q, accum_g_kl_q, accum_g_adv_q = 0.0, 0.0, 0.0, 0.0
                    accum_d_total_q, accum_d_real_q, accum_d_fake_q = 0.0, 0.0, 0.0
                    accum_wubu_avg_c_q, accum_wubu_avg_s_q, accum_wubu_var_c_q = 0.0, 0.0, 0.0
                    num_wubu_param_samples_q = 0

                    # Heuristic evaluation and intervention
                    if self.global_step > 0 and self.global_step % self.heuristic_check_interval == 0:
                        q_data_for_heuristics = self._get_q_controller_data_for_heuristics()
                        self._evaluate_training_state_and_apply_heuristics(q_data_for_heuristics) 

                    # Lambda_KL Q-Controller update
                    if self.lambda_kl_q_controller is not None and self.lambda_kl_update_interval > 0 and \
                       self.global_step > 0 and self.global_step % self.lambda_kl_update_interval == 0 and \
                       self.interval_steps_count > 0:
                        
                        current_interval_metrics_lkl: Dict[str, Union[float, None]] = {
                            'avg_recon': self.interval_metrics_accum['recon_dct'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_kl_div': self.interval_metrics_accum['kl_div'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'avg_d_total': self.interval_metrics_accum['d_total'] / self.interval_steps_count if self.interval_steps_count > 0 else None,
                            'val_metric': self.last_val_metrics.get(self.args.val_primary_metric), # Use last known val metric
                            'current_lambda_kl_val': self.lambda_kl # Current base lambda_kl
                        }
                        # ... (LKL Q-Ctrl update logic: get_state, compute_reward, update_q_values, choose_action, update self.lambda_kl) ...
                        # This is complex and involves interaction with self.lambda_kl_q_controller.
                        # For brevity, assume this logic block correctly updates self.lambda_kl.
                        # An important part is syncing the new self.lambda_kl (after Q-Ctrl action) to other Q-controllers.
                        # Example of the core LKL Q-Ctrl interaction:
                        q_state_lkl = self.lambda_kl_q_controller.get_lambda_kl_state(current_interval_metrics_lkl)
                        if self.lambda_kl_q_controller.prev_lambda_kl_state is not None and \
                           self.lambda_kl_q_controller.prev_lambda_kl_action is not None and \
                           q_state_lkl is not None and self.prev_interval_metrics_for_lambda_kl_reward is not None:
                            reward_lkl = self.lambda_kl_q_controller.compute_lambda_kl_reward(current_interval_metrics_lkl, self.prev_interval_metrics_for_lambda_kl_reward)
                            self.lambda_kl_q_controller.update_q_values(self.lambda_kl_q_controller.prev_lambda_kl_state, self.lambda_kl_q_controller.prev_lambda_kl_action, reward_lkl, q_state_lkl, mode='lambda_kl')
                        elif q_state_lkl is not None and hasattr(self.lambda_kl_q_controller, 'set_initial_lambda_kl_metrics'):
                            self.lambda_kl_q_controller.set_initial_lambda_kl_metrics(current_interval_metrics_lkl)
                        
                        if q_state_lkl is not None:
                            lkl_action_dict = self.lambda_kl_q_controller.choose_action(q_state_lkl, mode='lambda_kl')
                            chosen_lkl_scale = lkl_action_dict.get('lambda_kl_scale', 1.0)
                            new_base_lambda_kl_val = self.lambda_kl * chosen_lkl_scale 
                            self.lambda_kl = float(np.clip(new_base_lambda_kl_val, self.min_lambda_kl_q_control, self.max_lambda_kl_q_control))
                            self.lambda_kl_q_controller.prev_lambda_kl_state = q_state_lkl # Store current state as previous
                            self.lambda_kl_q_controller.prev_lambda_kl_action = lkl_action_dict # Store action

                        self.prev_interval_metrics_for_lambda_kl_reward = current_interval_metrics_lkl.copy()
                        self.interval_metrics_accum = defaultdict(float); self.interval_steps_count = 0
                        self._sync_lambda_kl_to_all_q_controllers() # Sync new effective L_KL

                    # Logging to WandB and TensorBoard
                    if self.global_step > 0 and self.args.log_interval > 0 and \
                       (self.global_step % self.args.log_interval == 0) and \
                       log_interval_items_processed > 0 and self.am_main_process:
                        
                        # Prepare metrics for logging
                        current_log_metrics_wandb_tb: Dict[str, Any] = {
                            f"train_loss/{k.replace('_agg', '')}": v / log_interval_items_processed
                            for k, v in log_interval_accum_losses.items()
                        }
                        eff_lkl_log = self.lambda_kl * self.heuristic_override_lambda_kl_factor
                        eff_lrec_log = self.lambda_recon * self.heuristic_override_lambda_recon_factor
                        current_log_metrics_wandb_tb["train_params/lambda_recon_eff"] = eff_lrec_log
                        current_log_metrics_wandb_tb["train_params/lambda_kl_eff"] = eff_lkl_log
                        current_log_metrics_wandb_tb["train_params/lambda_kl_base"] = self.lambda_kl
                        # ... (add LRs, epoch_frac, active_disc_info, Q-Ctrl info, heuristic flags to current_log_metrics_wandb_tb) ...
                        # This part is extensive and involves formatting many pieces of info.
                        # For example, retrieving info from each Q-controller.
                        # Q-Ctrl Info
                        q_controllers_for_log = {
                            "QGen": self.q_controller_gen, 
                            f"QD_{self.active_discriminator_key[:3].upper()}": self.q_controller_d_active, 
                            "QLKL": self.lambda_kl_q_controller
                        }
                        for q_prefix, q_ctrl_obj in q_controllers_for_log.items():
                            if q_ctrl_obj:
                                q_info = q_ctrl_obj.get_info()
                                for k_qinfo, v_qinfo in q_info.items():
                                     # Sanitize key for wandb/tb: remove spaces, special chars
                                    clean_k_qinfo = ''.join(c if c.isalnum() else '_' for c in str(k_qinfo)).lower()
                                    current_log_metrics_wandb_tb[f"q_info/{q_prefix}/{clean_k_qinfo}"] = v_qinfo
                        
                        # Heuristic flags
                        current_log_metrics_wandb_tb["heuristic_flags/vae_fm_active"] = 1 if self.heuristic_vae_feature_match_active else 0
                        # ... add all other heuristic flags and factors ...
                        current_log_metrics_wandb_tb["heuristic_factors/lrec_override"] = self.heuristic_override_lambda_recon_factor
                        current_log_metrics_wandb_tb["heuristic_factors/lkl_override"] = self.heuristic_override_lambda_kl_factor


                        # WuBu Interpretability Data Logging (less frequent)
                        interpretability_log_interval_eff = max(self.args.log_interval * 5, 100) # Example: every 5 log intervals or 100 steps
                        if self.global_step % interpretability_log_interval_eff == 0:
                             wubu_interp_log_data = self._get_wubu_interpretability_data_for_logging(current_global_ctx_raw_features)
                             current_log_metrics_wandb_tb.update(wubu_interp_log_data)
                        
                        if self.args.wandb and WANDB_AVAILABLE and wandb.run:
                            wandb.log(current_log_metrics_wandb_tb, step=self.global_step)
                        if self.tb_writer:
                            for k_tb, v_tb in current_log_metrics_wandb_tb.items():
                                if isinstance(v_tb, (int, float, np.number)): # TensorBoard only logs scalars well for generic add_scalar
                                    self.tb_writer.add_scalar(k_tb, v_tb, self.global_step)
                                # For histograms (e.g. WuBu params), use add_histogram if data is suitable
                            # Example: Logging WuBu param histograms to TensorBoard
                            # if "wubu_interp/encoder_wubu_s_stack/L0/current_c" in current_log_metrics_wandb_tb: # Check if data exists
                            #    # This needs actual list of C values, not just one scalar.
                            #    # self.tb_writer.add_histogram("wubu_curvatures/Encoder_L0", np.array([level_data["current_c"]...]), self.global_step)
                            pass


                        # Console Log (simplified example)
                        gt_console = current_log_metrics_wandb_tb.get('train_loss/loss_g_total',-1.0)
                        dt_console = current_log_metrics_wandb_tb.get('train_loss/loss_d_total',-1.0)
                        log_str_console = f"E{epoch+1} S{self.global_step} ActD:{self.active_disc_actual_type[0].upper()} G:{gt_console:.2f} D:{dt_console:.2f} LKL_eff:{eff_lkl_log:.1e}"
                        prog_bar.set_postfix_str(log_str_console, refresh=True)
                        # prog_bar.write(log_str_console_final_detailed) # A more detailed version for console

                        log_interval_accum_losses = defaultdict(float); log_interval_items_processed = 0

                    # Log reconstructed samples and fixed noise samples (uses self._log_samples_to_wandb)
                    if recon_mel_for_logging is not None and self.am_main_process and \
                       self.args.wandb_log_train_recon_interval > 0 and self.global_step > 0 and \
                       (self.global_step % self.args.wandb_log_train_recon_interval == 0):
                        self._log_samples_to_wandb("train_recon_mel", recon_mel_for_logging, self.args.num_val_samples_to_log)
                        if self.global_step % (self.args.wandb_log_train_recon_interval * getattr(self.args, 'train_target_log_freq_multiplier', 5)) == 0 :
                           self._log_samples_to_wandb("train_target_mel", batch_mel_segments, self.args.num_val_samples_to_log)
                    
                    if self.fixed_noise_for_sampling is not None and self.am_main_process and \
                       self.args.wandb_log_fixed_noise_samples_interval > 0 and self.global_step > 0 and \
                       (self.global_step % self.args.wandb_log_fixed_noise_samples_interval == 0):
                        m_ref_train.eval(); self.active_discriminator.eval() 
                        with torch.no_grad():
                            # Get DDP unwrapped active discriminator for its _assemble method
                            d_ref_sample_unwrapped_fn = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
                            generated_norm_dcts_fixed = m_ref_train.decode(self.fixed_noise_for_sampling, current_global_ctx_raw_features) # Pass context
                            unnorm_dcts_fixed = AudioSpecGenerator._unnormalize_dct(generated_norm_dcts_fixed, self.args)
                            
                            # Get canonical bboxes (as in sample method)
                            m_configs_fn = m_ref_train.module if self.ddp_active else m_ref_train # type: ignore
                            audio_cfg_fn = getattr(m_configs_fn, 'audio_config_ref', {})
                            gaad_cfg_fn = getattr(m_configs_fn, 'gaad_config_ref', {})
                            spec_time_fn = audio_cfg_fn.get("num_time_frames_for_1s_segment", 86)
                            spec_mels_fn = self.args.n_mels; spec_dims_canon_fn = (spec_time_fn, spec_mels_fn)
                            num_fixed_samples_fn = self.fixed_noise_for_sampling.shape[0]
                            bboxes_list_fixed_fn = [golden_subdivide_rect_fixed_n(spec_dims_canon_fn, gaad_cfg_fn['num_regions'], self.device, self.fixed_noise_for_sampling.dtype, gaad_cfg_fn.get('min_size_px',5)) for _ in range(num_fixed_samples_fn)]
                            bboxes_batch_fixed_fn = torch.stack(bboxes_list_fixed_fn)
                            target_mel_shape_fixed_fn = (num_fixed_samples_fn, 1, spec_mels_fn, spec_time_fn)
                            
                            fixed_noise_mels_gen = d_ref_sample_unwrapped_fn._assemble_mel_from_dct_regions(unnorm_dcts_fixed, bboxes_batch_fixed_fn, target_mel_shape_fixed_fn) # type: ignore
                            self._log_samples_to_wandb("fixed_noise_generated_mel", fixed_noise_mels_gen, num_fixed_samples_fn)
                        m_ref_train.train(); self.active_discriminator.train() 

                    # Checkpointing
                    if self.args.save_interval > 0 and self.global_step > 0 and \
                       (self.global_step % self.args.save_interval == 0) and self.am_main_process:
                        avg_g_total_current_interval = avg_losses_for_q_step.get('loss_g_total', -1.0)
                        avg_d_total_current_interval = avg_losses_for_q_step.get('loss_d_total', -1.0)
                        chkpt_metrics_interval = {
                            'train_loss_g_total_macro': avg_g_total_current_interval if np.isfinite(avg_g_total_current_interval) else -1.0,
                            'train_loss_d_total_macro': avg_d_total_current_interval if np.isfinite(avg_d_total_current_interval) else -1.0
                        }
                        self._capture_rng_states() # Capture before save
                        self._save_checkpoint(is_intermediate=True, metrics=chkpt_metrics_interval)
            
            # End of Epoch Validation and Checkpointing
            validation_interval_epochs_actual = getattr(self.args, 'validation_interval_epochs', 1) 
            if self.val_loader and self.am_main_process and validation_interval_epochs_actual > 0 and \
               (epoch + 1) % validation_interval_epochs_actual == 0:
                
                val_metrics_eoe_current = self.validate(num_val_samples_to_log=self.args.num_val_samples_to_log, global_ctx_raw_features_val=current_global_ctx_raw_features) 
                if val_metrics_eoe_current: 
                    if self.args.wandb and WANDB_AVAILABLE and wandb.run: 
                        wandb.log({f"val_metrics_eoe/{k_val_eoe}": v_val_eoe for k_val_eoe, v_val_eoe in val_metrics_eoe_current.items()}, step=self.global_step)
                    if self.tb_writer:
                        for k_tb_val, v_tb_val in val_metrics_eoe_current.items():
                            if isinstance(v_tb_val, (int,float, np.number)): self.tb_writer.add_scalar(f"val_metrics_eoe/{k_tb_val}", v_tb_val, self.global_step)
                    
                    metric_to_check_eoe_val = self.args.val_primary_metric
                    current_val_for_best_eoe_val: float = val_metrics_eoe_current.get(metric_to_check_eoe_val, self.best_val_metric_val) 
                    is_better_eoe_val = (current_val_for_best_eoe_val > self.best_val_metric_val) if self.is_val_metric_higher_better \
                                      else (current_val_for_best_eoe_val < self.best_val_metric_val)
                    if is_better_eoe_val and np.isfinite(current_val_for_best_eoe_val):
                        prog_bar.write(f"New best val metric ({metric_to_check_eoe_val}): {current_val_for_best_eoe_val:.4f} (prev: {self.best_val_metric_val:.4f}). Saving best checkpoint.")
                        self.best_val_metric_val = current_val_for_best_eoe_val
                        self._capture_rng_states()
                        self._save_checkpoint(is_best=True, metrics=val_metrics_eoe_current)
            
            save_epoch_interval_actual = getattr(self.args, 'save_epoch_interval_epochs', 1) # Renamed in args
            if self.am_main_process and save_epoch_interval_actual > 0 and (epoch + 1) % save_epoch_interval_actual == 0:
                is_better_eoe_defined = 'is_better_eoe_val' in locals() and locals().get('is_better_eoe_val', False)
                already_saved_as_best_ep = is_better_eoe_defined and np.isfinite(locals().get('current_val_for_best_eoe_val', float('inf')))
                is_last_batch_ep = batch_idx == num_batches_epoch -1 if num_batches_epoch > 0 else False
                already_saved_as_intermediate_ep = self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0 and is_last_batch_ep
                if not (already_saved_as_best_ep or already_saved_as_intermediate_ep):
                    eoe_metrics_for_save_ep = self.last_val_metrics.copy() if self.last_val_metrics else {}
                    # Add latest training losses if available from avg_losses_for_q_step
                    if 'avg_losses_for_q_step' in locals() and avg_losses_for_q_step:
                        eoe_metrics_for_save_ep["epoch_end_train_g_total_approx"] = avg_losses_for_q_step.get('loss_g_total', -1.0)
                        eoe_metrics_for_save_ep["epoch_end_train_d_total_approx"] = avg_losses_for_q_step.get('loss_d_total', -1.0)
                    self._capture_rng_states()
                    self._save_checkpoint(metrics=eoe_metrics_for_save_ep) # Saves as regular epoch end

        # End of training loop
        if self.tb_writer: self.tb_writer.close()


    @torch.no_grad()
    def validate(self, num_val_samples_to_log: int = 1, global_ctx_raw_features_val: Optional[torch.Tensor] = None) -> Optional[Dict[str, float]]: # Pass context
        if not self.val_loader or not self.am_main_process: return None
        m_ref_val = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_ref_val_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        
        original_training_mode_m = m_ref_val.training
        original_training_mode_d = d_ref_val_active.training # type: ignore
        m_ref_val.eval(); d_ref_val_active.eval() # type: ignore

        # ... (rest of validation logic as before, but pass global_ctx_raw_features_val to model and D calls) ...
        # Example modification for model call in validation:
        # recon_norm_dcts, _, _, gaad_bboxes_from_enc, target_norm_dcts_for_loss = \
        #    m_ref_val(real_mel_segments, global_ctx_raw_features_val) # Pass context
        # ... (rest of metric calculations) ...
        total_recon_dct_mse_sum = 0.0; total_mel_mse_sum = 0.0; total_psnr_mel_sum = 0.0
        total_ssim_mel_sum = 0.0; total_lpips_mel_sum = 0.0; total_items_evaluated = 0
        dtype_m_val = next(iter(m_ref_val.parameters()), torch.tensor(0.0, device=self.device)).dtype
        logged_samples_count_this_val_run = 0

        for batch_idx_val_run, batch_real_mel_segments_val in enumerate(
            tqdm(self.val_loader, desc="Validating", disable=not self.am_main_process or os.getenv('CI') == 'true' or getattr(self.args, 'disable_val_tqdm', False), dynamic_ncols=True)
        ):
            real_mel_segments_val = batch_real_mel_segments_val.to(self.device, dtype=dtype_m_val)
            B_val, _, H_mel_val, W_mel_val = real_mel_segments_val.shape

            recon_norm_dcts_val, _, _, gaad_bboxes_from_enc_val, target_norm_dcts_for_loss_val = \
                m_ref_val(real_mel_segments_val, global_ctx_raw_features_val) # Pass context
            
            loss_recon_dct_batch_val = self._compute_recon_loss(recon_norm_dcts_val, target_norm_dcts_for_loss_val)
            if torch.isfinite(loss_recon_dct_batch_val): total_recon_dct_mse_sum += loss_recon_dct_batch_val.item() * B_val
            
            unnorm_recon_dcts_val = AudioSpecGenerator._unnormalize_dct(recon_norm_dcts_val, self.args)
            # D's _assemble method is part of the D class instance
            recon_mel_assembled_val = d_ref_val_active._assemble_mel_from_dct_regions(unnorm_recon_dcts_val, gaad_bboxes_from_enc_val, real_mel_segments_val.shape) # type: ignore
            
            if recon_mel_assembled_val.shape == real_mel_segments_val.shape:
                loss_mel_mse_batch_val = F.mse_loss(recon_mel_assembled_val, real_mel_segments_val, reduction='mean')
                if torch.isfinite(loss_mel_mse_batch_val):
                    total_mel_mse_sum += loss_mel_mse_batch_val.item() * B_val
                    mse_val_current = loss_mel_mse_batch_val.item()
                    psnr_val_current = 10 * math.log10(1.0 / (mse_val_current + EPS)) if mse_val_current > EPS else 100.0 
                    total_psnr_mel_sum += psnr_val_current * B_val
                
                recon_mel_01_val = (recon_mel_assembled_val.clamp(-1,1)+1)/2.0
                real_mel_01_val = (real_mel_segments_val.clamp(-1,1)+1)/2.0

                if self.ssim_metric:
                    try: ssim_val_batch = self.ssim_metric(recon_mel_01_val, real_mel_01_val); total_ssim_mel_sum += ssim_val_batch.item() * B_val
                    except Exception as e_ssim_val: self.logger.debug(f"Val SSIM failed: {e_ssim_val}")
                if self.lpips_loss_fn:
                    try:
                        rec_lpips_val = recon_mel_assembled_val.repeat(1,3,1,1) if recon_mel_assembled_val.shape[1]==1 else recon_mel_assembled_val
                        real_lpips_val = real_mel_segments_val.repeat(1,3,1,1) if real_mel_segments_val.shape[1]==1 else real_mel_segments_val
                        lpips_val_batch = self.lpips_loss_fn(rec_lpips_val, real_lpips_val); total_lpips_mel_sum += lpips_val_batch.sum().item() # Sum over batch items for LPIPS
                    except Exception as e_lpips_val: self.logger.debug(f"Val LPIPS failed: {e_lpips_val}")
            
            total_items_evaluated += B_val
            if logged_samples_count_this_val_run < num_val_samples_to_log and self.args.wandb and WANDB_AVAILABLE and wandb.run:
                num_to_log_now_val = min(B_val, num_val_samples_to_log - logged_samples_count_this_val_run)
                if num_to_log_now_val > 0:
                    self._log_samples_to_wandb("val_recon_mel", recon_mel_assembled_val[:num_to_log_now_val], num_to_log_now_val)
                    self._log_samples_to_wandb("val_target_mel", real_mel_segments_val[:num_to_log_now_val], num_to_log_now_val)
                logged_samples_count_this_val_run += num_to_log_now_val
        
        m_ref_val.train(original_training_mode_m)
        d_ref_val_active.train(original_training_mode_d) # type: ignore

        if total_items_evaluated == 0: return None
        val_metrics_final = {
            "avg_val_recon_dct_mse": total_recon_dct_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_mel_mse": total_mel_mse_sum / total_items_evaluated if total_items_evaluated > 0 else float('inf'),
            "avg_val_psnr_mel": total_psnr_mel_sum / total_items_evaluated if total_items_evaluated > 0 else 0.0,
            "avg_val_ssim_mel": total_ssim_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.ssim_metric else 0.0,
            "avg_val_lpips_mel": total_lpips_mel_sum / total_items_evaluated if total_items_evaluated > 0 and self.lpips_loss_fn else float('inf')
        }
        self.last_val_metrics = val_metrics_final # Update trainer's record of last val metrics
        self.logger.info(f"Validation Metrics (Ep {self.current_epoch+1}, GStep {self.global_step}, ActiveD: {self.active_discriminator_key}): " + 
                         ", ".join([f"{k_val_log}:{v_val_log:.4f}" for k_val_log,v_val_log in val_metrics_final.items()]))
        return val_metrics_final


    def _capture_rng_states(self):
        self.rng_states_for_checkpoint = {
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_cpu_random_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            self.rng_states_for_checkpoint['torch_cuda_random_state'] = torch.cuda.get_rng_state_all() # All GPUs

    def _load_rng_states(self, rng_states_from_ckpt: Dict):
        try:
            if 'python_random_state' in rng_states_from_ckpt: random.setstate(rng_states_from_ckpt['python_random_state'])
            if 'numpy_random_state' in rng_states_from_ckpt: np.random.set_state(rng_states_from_ckpt['numpy_random_state'])
            if 'torch_cpu_random_state' in rng_states_from_ckpt: torch.set_rng_state(rng_states_from_ckpt['torch_cpu_random_state'])
            if torch.cuda.is_available() and 'torch_cuda_random_state' in rng_states_from_ckpt:
                torch.cuda.set_rng_state_all(rng_states_from_ckpt['torch_cuda_random_state'])
            self.logger.info("RNG states loaded from checkpoint for reproducibility.")
        except Exception as e_rng:
            self.logger.warning(f"Could not load RNG states from checkpoint: {e_rng}. Reproducibility might be affected.")

    def _save_checkpoint(self, is_intermediate: bool = False, metrics: Optional[Dict[str, Any]] = None, is_best: bool = False): # Save new configs and states
        if not self.am_main_process: return
        m_s = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_primary_s = self.discriminator_primary_obj.module if self.ddp_active and hasattr(self.discriminator_primary_obj, 'module') else self.discriminator_primary_obj
        d_alt_s = self.discriminator_alternative_obj.module if self.ddp_active and hasattr(self.discriminator_alternative_obj, 'module') else self.discriminator_alternative_obj

        def get_q_state_from_controller(qc_obj: Optional[HAKMEMQController]):
            if qc_obj is None: return None
            # Same detailed state saving as before (q_table, epsilon, histories, etc.)
            # For brevity, assume this helper correctly serializes the Q-controller state
            # from the HAKMEMQController thoughts
            state = { 'q_table': qc_obj.q_table, 'epsilon': qc_obj.epsilon,
                     'prev_lr_mom_state': qc_obj.prev_lr_mom_state, 'prev_lr_mom_action': qc_obj.prev_lr_mom_action,
                     'prev_lambda_kl_state': qc_obj.prev_lambda_kl_state, 'prev_lambda_kl_action': qc_obj.prev_lambda_kl_action,
                     'prev_heuristic_toggle_state': getattr(qc_obj, 'prev_heuristic_toggle_state', None), # Save new
                     'prev_heuristic_toggle_action': getattr(qc_obj, 'prev_heuristic_toggle_action', None), # Save new
                     'reward_hist': list(qc_obj.reward_hist),
                     'q_table_access_count': dict(qc_obj.q_table_access_count),
                     'q_table_creation_time': qc_obj.q_table_creation_time, 
                     'q_table_last_access_time': qc_obj.q_table_last_access_time,
                     'on_probation': getattr(qc_obj, 'on_probation', False), 
                     'current_probation_step': getattr(qc_obj, 'current_probation_step', 0),
                     'lkl_on_probation': getattr(qc_obj, 'lkl_on_probation', False), 
                     'lkl_current_probation_step': getattr(qc_obj, 'lkl_current_probation_step', 0),
                     'current_alpha': qc_obj.current_alpha, 'current_gamma': qc_obj.current_gamma, # Save meta-adaptive params
                     'long_term_reward_avg_hist': list(qc_obj.long_term_reward_avg_hist)
                    }
            # Loss histories
            loss_hist_names = ['g_total', 'g_recon', 'g_kl', 'g_adv', 'd_total', 'd_real', 'd_fake', 
                               'wubu_avg_curvature', 'wubu_avg_scale', 'wubu_var_curvature']
            state['loss_histories'] = {hname: list(getattr(qc_obj, f"{hname}_hist" if "wubu" in hname else f"loss_{hname}_hist")) 
                                       for hname in loss_hist_names if hasattr(qc_obj, f"{hname}_hist" if "wubu" in hname else f"loss_{hname}_hist")}
            # Interval histories (for LKL controller)
            if hasattr(qc_obj, 'interval_avg_recon_hist'):
                lkl_hist_names = ['avg_recon', 'avg_kl_div', 'avg_d_total', 'val_metric']
                state['interval_histories'] = {hname: list(getattr(qc_obj, f"interval_{hname}_hist")) 
                                               for hname in lkl_hist_names if hasattr(qc_obj, f"interval_{hname}_hist")}
            return state

        data = {
            'global_step': self.global_step, 'epoch': self.current_epoch,
            'model_state_dict': m_s.state_dict(),
            'discriminator_primary_state_dict': d_primary_s.state_dict(), # type: ignore
            'discriminator_alternative_state_dict': d_alt_s.state_dict(), # type: ignore
            'active_discriminator_key': self.active_discriminator_key,
            'active_disc_actual_type': self.active_disc_actual_type,
            'optimizer_enc_gen_state_dict': self.optimizer_enc_gen.state_dict(),
            'optimizer_disc_primary_state_dict': self.optimizer_disc_primary.state_dict(),
            'optimizer_disc_alternative_state_dict': self.optimizer_disc_alternative.state_dict(),
            'scaler_enc_gen_state_dict': self.scaler_enc_gen.state_dict(),
            'scaler_disc_state_dict': self.scaler_disc.state_dict(),
            'args': vars(self.args), 'metrics': metrics if metrics is not None else self.last_val_metrics.copy(),
            'best_val_metric_val': self.best_val_metric_val, 'current_lambda_kl': self.lambda_kl,
            'prev_interval_metrics_for_lambda_kl_reward': self.prev_interval_metrics_for_lambda_kl_reward,
            
            # Save full config objects <TOTAL_STRATEGY_INTEGRATION>
            'wubu_s_enc_cfg': self.wubu_s_enc_cfg, 'wubu_g_gen_cfg': self.wubu_g_gen_cfg,
            'wubu_d_pri_cfg': self.wubu_d_pri_cfg, 'wubu_d_alt_cfg': self.wubu_d_alt_cfg,
            'global_ctx_cfg': self.global_ctx_cfg, 'q_learn_cfg_template': self.q_learn_cfg_template,
            'disc_pri_specific_cfg': self.disc_pri_specific_cfg, 
            'disc_alt_specific_cfg': self.disc_alt_specific_cfg,

            # Heuristic states
            'steps_since_last_d_switch': self.steps_since_last_d_switch,
            'consecutive_trigger_primary_to_alt_count': self.consecutive_trigger_primary_to_alt_count,
            'consecutive_trigger_alt_to_primary_count': self.consecutive_trigger_alt_to_primary_count,
            'consecutive_heuristic_trigger_counts': dict(self.consecutive_heuristic_trigger_counts),
            'q_data_derived_g_recon_hist': list(self.q_data_derived_g_recon_hist), # Also avg_g_recon_hist_for_stagnation
            'rec_dct_stagnant_flag': self.rec_dct_stagnant, # Save the flag
            
            'heuristic_vae_feature_match_active': self.heuristic_vae_feature_match_active,
            'heuristic_penalize_g_easy_win_active': self.heuristic_penalize_g_easy_win_active,
            'heuristic_boost_active_d_lr_active': self.heuristic_boost_active_d_lr_active,
            # 'heuristic_force_d_q_explore_active' is a trigger, its effect is in Q-Ctrl's epsilon boost state
            'heuristic_override_lambda_recon_factor': self.heuristic_override_lambda_recon_factor,
            'heuristic_override_lambda_kl_factor': self.heuristic_override_lambda_kl_factor,

            # Q-Controller states
            'q_controller_enc_gen_state': get_q_state_from_controller(self.q_controller_gen),
            'q_controller_disc_primary_state': get_q_state_from_controller(self.q_controller_d_primary),
            'q_controller_disc_alternative_state': get_q_state_from_controller(self.q_controller_d_alt),
            'q_controller_lambda_kl_state': get_q_state_from_controller(self.lambda_kl_q_controller),
            
            # RNG States for reproducibility <TOTAL_STRATEGY_INTEGRATION>
            'rng_states': self.rng_states_for_checkpoint,
        }
        
        fprefix = "wubuspectrans_ckpt_v020" # Updated version
        if is_best: fp_str = f"{fprefix}_best_ep{self.current_epoch + 1}_step{self.global_step}.pt"
        elif is_intermediate: fp_str = f"{fprefix}_step{self.global_step}.pt"
        else: fp_str = f"{fprefix}_ep{self.current_epoch + 1}_step{self.global_step}.pt"
        fp = Path(self.args.checkpoint_dir) / fp_str
        try: torch.save(data, fp); self.logger.info(f"Checkpoint saved: {fp.name}")
        except Exception as e_save: self.logger.error(f"Save CKPT error {fp}: {e_save}", exc_info=True)


    def load_checkpoint(self, checkpoint_path_str: str) -> Tuple[int, int]: # Major updates for new configs/states
        # ... (File existence and basic ckpt loading as before) ...
        # Key changes:
        # 1. Load new config objects (WuBuStackConfig, etc.) and potentially compare with current args.
        #    For now, assume args drive the config struct, ckpt might just be for record.
        #    If configs from ckpt were to be used, careful handling of mismatches with current code is needed.
        #    Simplest: current args + _configure_wubu_stack in main() defines the model structure.
        #    CKPT's saved configs are for reference or if one day we load configs directly.
        # 2. Load all new Q-controller states (including meta-adaptive params, new hists).
        # 3. Load all new heuristic states and RNG states.
        # 4. The _load_q_state_helper_inner needs to handle the richer Q-controller state.
        
        checkpoint_path = Path(checkpoint_path_str)
        global_manual_flush_q = getattr(HAKMEMQController, 'MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD', False)
        effective_reset_q_request = global_manual_flush_q or self.args.reset_q_controllers_on_load

        if not checkpoint_path.exists():
            self.logger.warning(f"CKPT {checkpoint_path} not found. Starting fresh.")
            # Initialize Q-controllers to fresh state (as if no checkpoint)
            all_qcs_to_init = [self.q_controller_gen, self.q_controller_d_primary, self.q_controller_d_alt, self.lambda_kl_q_controller]
            for qc_init in all_qcs_to_init:
                if qc_init: self._load_q_state_helper_inner(qc_init, None, effective_reset_q_request, False)
            if global_manual_flush_q and not self.args.reset_q_controllers_on_load: HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
            return 0, 0

        try: ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e_load_main:
            self.logger.error(f"Failed to load CKPT {checkpoint_path}: {e_load_main}. Starting fresh.", exc_info=True)
            all_qcs_to_init_err = [self.q_controller_gen, self.q_controller_d_primary, self.q_controller_d_alt, self.lambda_kl_q_controller]
            for qc_init_err in all_qcs_to_init_err:
                if qc_init_err: self._load_q_state_helper_inner(qc_init_err, None, effective_reset_q_request, False)
            if global_manual_flush_q and not self.args.reset_q_controllers_on_load: HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
            return 0, 0
        
        self.logger.info(f"Loaded CKPT: {checkpoint_path}")

        # Load basic progress, metrics, lambda_kl
        self.is_val_metric_higher_better = self.args.val_primary_metric in ["avg_val_ssim_mel", "avg_val_psnr_mel"] # Re-evaluate based on current args
        default_best_val_load = -float('inf') if self.is_val_metric_higher_better else float('inf')
        self.best_val_metric_val = ckpt.get('best_val_metric_val', default_best_val_load)
        self.last_val_metrics = ckpt.get('metrics', {}).copy() if ckpt.get('metrics') is not None else {}
        self.prev_interval_metrics_for_lambda_kl_reward = ckpt.get('prev_interval_metrics_for_lambda_kl_reward')
        self.lambda_kl = float(ckpt.get('current_lambda_kl', self.args.lambda_kl))

        loaded_gs_val = ckpt.get('global_step', 0)
        loaded_ep_val = ckpt.get('epoch', 0)
        next_ep_start_val = loaded_ep_val + 1 if self.args.load_strict and loaded_gs_val > 0 and loaded_ep_val < self.args.epochs else loaded_ep_val
        if getattr(self.args, 'force_start_epoch_on_load', None) is not None:
            next_ep_start_val = self.args.force_start_epoch_on_load
            loaded_gs_val = getattr(self.args, 'force_start_gstep_on_load', 0 if self.args.force_start_epoch_on_load is not None else loaded_gs_val)
            if self.am_main_process: self.logger.info(f"CKPT Load: Overriding start epoch to {next_ep_start_val} and GStep to {loaded_gs_val} due to force_start args.")
        
        # RNG States (load before model/optimizer states if they use RNG in init, though unlikely here)
        if 'rng_states' in ckpt and ckpt['rng_states'] is not None: self._load_rng_states(ckpt['rng_states'])
        else: self.logger.warning("RNG states not found in checkpoint. Reproducibility from this point may vary slightly.")


        # Model state dicts (same loading logic)
        m_load_ref = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_pri_load_ref = self.discriminator_primary_obj.module if self.ddp_active and hasattr(self.discriminator_primary_obj, 'module') else self.discriminator_primary_obj
        d_alt_load_ref = self.discriminator_alternative_obj.module if self.ddp_active and hasattr(self.discriminator_alternative_obj, 'module') else self.discriminator_alternative_obj
        model_loaded_ok_val, disc_pri_loaded_ok_val, disc_alt_loaded_ok_val = False, False, False
        try:
            if 'model_state_dict' in ckpt: m_load_ref.load_state_dict(ckpt['model_state_dict'], strict=self.args.load_strict); model_loaded_ok_val = True
        except Exception as e_mload: self.logger.error(f"Error loading main model state_dict: {e_mload}", exc_info=not self.args.load_strict)
        if 'discriminator_primary_state_dict' in ckpt and d_pri_load_ref:
            try: d_pri_load_ref.load_state_dict(ckpt['discriminator_primary_state_dict'], strict=self.args.load_strict); disc_pri_loaded_ok_val = True # type: ignore
            except Exception as e_dpload: self.logger.error(f"Error loading D_primary state_dict: {e_dpload}", exc_info=not self.args.load_strict)
        if 'discriminator_alternative_state_dict' in ckpt and d_alt_load_ref:
            try: d_alt_load_ref.load_state_dict(ckpt['discriminator_alternative_state_dict'], strict=self.args.load_strict); disc_alt_loaded_ok_val = True # type: ignore
            except Exception as e_daload: self.logger.error(f"Error loading D_alternative state_dict: {e_daload}", exc_info=not self.args.load_strict)


        # Active Discriminator (same override logic)
        saved_active_d_key_val = ckpt.get('active_discriminator_key', 'primary')
        target_active_key_resume_val = saved_active_d_key_val
        forced_d_switch_resume_val = False
        if self.args.enable_heuristic_disc_switching: # And initial_disc_type from args implies a switch
            initial_disc_type_arg_val = self.args.initial_disc_type if self.args.initial_disc_type is not None else self.args.disc_input_type
            current_args_implied_active_key_val = None
            if initial_disc_type_arg_val == self.primary_disc_actual_type: current_args_implied_active_key_val = 'primary'
            elif initial_disc_type_arg_val == self.alternative_disc_actual_type: current_args_implied_active_key_val = 'alternative'
            if current_args_implied_active_key_val is not None and current_args_implied_active_key_val != saved_active_d_key_val:
                if self.am_main_process: self.logger.warning(f"LOAD_CKPT_OVERRIDE: Args imply active D '{current_args_implied_active_key_val}', ckpt had '{saved_active_d_key_val}'. FORCING D.")
                target_active_key_resume_val = current_args_implied_active_key_val; forced_d_switch_resume_val = True
        self.active_discriminator_key = target_active_key_resume_val
        self._update_active_discriminator_pointers()


        # Optimizer states (same logic, relies on model_loaded_ok flags)
        opt_g_loaded_ok_val, opt_dp_loaded_ok_val, opt_da_loaded_ok_val = False, False, False
        # ... (optimizer loading logic for G, D_pri, D_alt, setting initial_lr/momentum)...
        if self.optimizer_enc_gen and ckpt.get('optimizer_enc_gen_state_dict'):
            if model_loaded_ok_val:
                try: self.optimizer_enc_gen.load_state_dict(ckpt['optimizer_enc_gen_state_dict']); opt_g_loaded_ok_val = True
                except Exception as e_og: self.logger.warning(f"Could not load Opt_Gen state: {e_og}")
        if self.optimizer_enc_gen: # Set initial_lr/mom for Q-Ctrl base
            for grp in self.optimizer_enc_gen.param_groups: grp['initial_lr'] = self.args.learning_rate_gen; grp['initial_momentum'] = self.optimizer_enc_gen.defaults.get('momentum',0.9)

        if self.optimizer_disc_primary and ckpt.get('optimizer_disc_primary_state_dict'):
            if disc_pri_loaded_ok_val:
                try: self.optimizer_disc_primary.load_state_dict(ckpt['optimizer_disc_primary_state_dict']); opt_dp_loaded_ok_val = True
                except Exception as e_odp: self.logger.warning(f"Could not load Opt_D_Pri state: {e_odp}")
        if self.optimizer_disc_primary:
            for grp in self.optimizer_disc_primary.param_groups: grp['initial_lr'] = self.args.learning_rate_disc; grp['initial_momentum'] = self.optimizer_disc_primary.defaults.get('momentum',0.9)
        
        lr_d_alt_load_val = getattr(self.args, 'learning_rate_disc_alt', self.args.learning_rate_disc)
        if self.optimizer_disc_alternative and ckpt.get('optimizer_disc_alternative_state_dict'):
            if disc_alt_loaded_ok_val:
                try: self.optimizer_disc_alternative.load_state_dict(ckpt['optimizer_disc_alternative_state_dict']); opt_da_loaded_ok_val = True
                except Exception as e_oda: self.logger.warning(f"Could not load Opt_D_Alt state: {e_oda}")
        if self.optimizer_disc_alternative:
            for grp in self.optimizer_disc_alternative.param_groups: grp['initial_lr'] = lr_d_alt_load_val; grp['initial_momentum'] = self.optimizer_disc_alternative.defaults.get('momentum',0.9)


        # Q-Controller states (using the helper)
        self._load_q_state_helper_inner(self.q_controller_gen, ckpt.get('q_controller_enc_gen_state'), effective_reset_q_request, opt_g_loaded_ok_val)
        self._load_q_state_helper_inner(self.q_controller_d_primary, ckpt.get('q_controller_disc_primary_state'), effective_reset_q_request, opt_dp_loaded_ok_val)
        self._load_q_state_helper_inner(self.q_controller_d_alt, ckpt.get('q_controller_disc_alternative_state'), effective_reset_q_request, opt_da_loaded_ok_val)
        self._load_q_state_helper_inner(self.lambda_kl_q_controller, ckpt.get('q_controller_lambda_kl_state'), effective_reset_q_request, True) # LKL Q-Ctrl not tied to specific opt's model load status

        if forced_d_switch_resume_val and self.q_controller_d_active:
            if self.am_main_process: self.logger.warning(f"Due to resume override, resetting Q-controller for newly FORCED active D: '{self.active_discriminator_key}'.")
            self.q_controller_d_active.reset_q_learning_state(True, True, f"Forced D switch to {self.active_discriminator_key} on Resume Override", True)
            self.steps_since_last_d_switch = 0; self.consecutive_trigger_primary_to_alt_count = 0; self.consecutive_trigger_alt_to_primary_count = 0
            self.consecutive_heuristic_trigger_counts = defaultdict(int); self.q_data_derived_g_recon_hist.clear(); self.rec_dct_stagnant = False

        if global_manual_flush_q and not self.args.reset_q_controllers_on_load: HAKMEMQController.MANUALLY_FLUSH_Q_TABLES_ON_NEXT_LOAD = False
        elif self.args.reset_q_controllers_on_load and self.am_main_process: self.logger.info("Global Q-controller reset triggered by --reset_q_controllers_on_load.")


        # Scalers (same logic)
        if self.args.use_amp and self.device.type == 'cuda':
            if ckpt.get('scaler_enc_gen_state_dict') and self.scaler_enc_gen: self.scaler_enc_gen.load_state_dict(ckpt['scaler_enc_gen_state_dict'])
            if ckpt.get('scaler_disc_state_dict') and self.scaler_disc: self.scaler_disc.load_state_dict(ckpt['scaler_disc_state_dict'])
        
        self._sync_lambda_kl_to_all_q_controllers() # Sync loaded lambda_kl

        # Heuristic states (load unless forced_d_switch_resume reset them)
        if not forced_d_switch_resume_val:
            self.steps_since_last_d_switch = ckpt.get('steps_since_last_d_switch', 0)
            self.consecutive_trigger_primary_to_alt_count = ckpt.get('consecutive_trigger_primary_to_alt_count', 0)
            self.consecutive_trigger_alt_to_primary_count = ckpt.get('consecutive_trigger_alt_to_primary_count', 0)
            self.consecutive_heuristic_trigger_counts = defaultdict(int, ckpt.get('consecutive_heuristic_trigger_counts', {}))
            if 'q_data_derived_g_recon_hist' in ckpt and ckpt['q_data_derived_g_recon_hist'] is not None:
                try: self.q_data_derived_g_recon_hist.clear(); self.q_data_derived_g_recon_hist.extend(list(ckpt['q_data_derived_g_recon_hist']))
                except TypeError: self.logger.warning(f"Could not extend deque q_data_derived_g_recon_hist from checkpoint.")
            self.rec_dct_stagnant = ckpt.get('rec_dct_stagnant_flag', False) # Load the flag

        self.heuristic_vae_feature_match_active = ckpt.get('heuristic_vae_feature_match_active', False)
        self.heuristic_penalize_g_easy_win_active = ckpt.get('heuristic_penalize_g_easy_win_active', False)
        self.heuristic_boost_active_d_lr_active = ckpt.get('heuristic_boost_active_d_lr_active', False)
        # self.heuristic_force_d_q_explore_active is a trigger, its state is within Q-Ctrl's epsilon_boost_active_steps
        self.heuristic_override_lambda_recon_factor = ckpt.get('heuristic_override_lambda_recon_factor', 1.0)
        self.heuristic_override_lambda_kl_factor = ckpt.get('heuristic_override_lambda_kl_factor', 1.0)

        self.logger.info(f"Resuming training. GlobalStep: {loaded_gs_val}, NextEpochStart: {next_ep_start_val}. ActiveD: '{self.active_discriminator_key}'. L_KL_base: {self.lambda_kl:.4e}")
        return loaded_gs_val, next_ep_start_val


    @staticmethod
    def get_scale_from_action_value(action_val: Union[Dict, str, None], scale_key: str, default: float = 1.0) -> float: # Unchanged
        if isinstance(action_val, dict): return action_val.get(scale_key, default)
        return default

    @torch.no_grad()
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None, global_ctx_raw_features_sample: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]: # Pass context
        m_ref_sample = self.model.module if self.ddp_active and hasattr(self.model, 'module') else self.model
        d_ref_sample_active = self.active_discriminator.module if self.ddp_active and hasattr(self.active_discriminator, 'module') else self.active_discriminator
        
        original_mode_m_sample = m_ref_sample.training; original_mode_d_sample = d_ref_sample_active.training # type: ignore
        m_ref_sample.eval(); d_ref_sample_active.eval() # type: ignore

        dev_sample = self.device
        dtype_m_sample = next(iter(m_ref_sample.parameters()), torch.tensor(0.0, device=self.device)).dtype
        lat_dim_sample = self.args.latent_dim

        if noise is None: z_sample = torch.randn(num_samples, lat_dim_sample, device=dev_sample, dtype=dtype_m_sample)
        else: z_sample = noise.to(device=dev_sample, dtype=dtype_m_sample); num_samples = z_sample.shape[0]

        # Pass context to generator's decode method
        generated_norm_dcts_sample = m_ref_sample.decode(z_sample, global_ctx_raw_features_sample)
        unnorm_dcts_for_assembly_sample = AudioSpecGenerator._unnormalize_dct(generated_norm_dcts_sample, self.args)
        
        # Get canonical bboxes (as before)
        audio_cfg_sample = getattr(m_ref_sample, 'audio_config_ref', {})
        gaad_cfg_sample = getattr(m_ref_sample, 'gaad_config_ref', {})
        spec_time_sample = audio_cfg_sample.get("num_time_frames_for_1s_segment", 86)
        spec_mels_sample = self.args.n_mels
        spec_dims_canonical_sample = (spec_time_sample, spec_mels_sample)
        canonical_bboxes_list_sample = [golden_subdivide_rect_fixed_n(spec_dims_canonical_sample, gaad_cfg_sample['num_regions'], dev_sample, dtype_m_sample, gaad_cfg_sample.get('min_size_px', 5)) for _ in range(num_samples)]
        canonical_gaad_bboxes_batch_sample = torch.stack(canonical_bboxes_list_sample)
        target_mel_shape_for_sample_out = (num_samples, 1, spec_mels_sample, spec_time_sample)
        
        generated_mel_spectrograms_sample = d_ref_sample_active._assemble_mel_from_dct_regions(unnorm_dcts_for_assembly_sample, canonical_gaad_bboxes_batch_sample, target_mel_shape_for_sample_out) # type: ignore
        
        m_ref_sample.train(original_mode_m_sample)
        d_ref_sample_active.train(original_mode_d_sample) # type: ignore
        return generated_mel_spectrograms_sample



# =====================================================================
# Arg Parsing and Main Execution Logic (Updated for Config Dataclasses)
# =====================================================================

def seed_worker_init_fn(worker_id: int, base_seed: int, rank: int, world_size: int):
     # Corrected worker_seed calculation for DDP to ensure unique seeds across all workers globally
     # This was a subtle point, original might have collisions if multiple nodes.
     # A common way: base_seed + rank * (num_workers_per_node_approx * some_large_prime) + worker_id
     # Simpler for now:
     worker_seed = base_seed + (rank * 1000) + worker_id # Ensure rank has a larger impact
     random.seed(worker_seed)
     np.random.seed(worker_seed)
     torch.manual_seed(worker_seed)
     if torch.cuda.is_available():
         torch.cuda.manual_seed_all(worker_seed) # Important for CUDA operations in workers

def seed_everything(seed:int,rank:int=0,world_size:int=1): # world_size unused here but kept for signature
    actual_seed = seed + rank # Offset seed by rank for DDP
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    # Potentially set deterministic algorithms if desired (can impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def _populate_wubu_level_config_from_args(args: argparse.Namespace, prefix: str, level_idx: int, num_total_levels: int) -> WuBuLevelConfig:
    """Helper to populate a single WuBuLevelConfig from argparse."""
    cfg = WuBuLevelConfig() # Start with defaults

    def get_arg_list_val(attr_name, default_list_val):
        raw_val = getattr(args, f"{prefix}_{attr_name}", default_list_val)
        if not isinstance(raw_val, list): raw_val = [raw_val] # Ensure list
        # Cycle or truncate/pad to num_total_levels, then pick level_idx
        # This logic ensures that if user provides 1 value, it's used for all levels.
        # If they provide N values for M levels (N < M), last value is repeated.
        # If N > M, extra values are ignored.
        if not raw_val: # Empty list from args
            effective_list = default_list_val
        elif len(raw_val) < num_total_levels:
            effective_list = raw_val + [raw_val[-1]] * (num_total_levels - len(raw_val))
        else:
            effective_list = raw_val
        return effective_list[level_idx] if level_idx < len(effective_list) else effective_list[-1]

    cfg.hyperbolic_dim = get_arg_list_val("hyperbolic_dims", WuBuLevelConfig().hyperbolic_dim) # Pass default from dataclass
    cfg.initial_curvature = get_arg_list_val("initial_curvatures", WuBuLevelConfig().initial_curvature)
    cfg.initial_scale = get_arg_list_val("initial_scales", WuBuLevelConfig().initial_scale) # Need initial_scales in argparse
    cfg.initial_spread = get_arg_list_val("initial_spread_values", WuBuLevelConfig().initial_spread) # Need initial_spread_values
    cfg.boundary_points = get_arg_list_val("boundary_points_per_level", WuBuLevelConfig().boundary_points) # Need boundary_points_per_level

    # log_g factors (scalar from args, applies to all levels of this stack)
    cfg.log_g_complexity_influence_factor_curv = getattr(args, f"{prefix}_log_g_factor_curv", cfg.log_g_complexity_influence_factor_curv)
    cfg.log_g_complexity_influence_factor_scale = getattr(args, f"{prefix}_log_g_factor_scale", cfg.log_g_complexity_influence_factor_scale)
    cfg.log_g_complexity_influence_factor_spread = getattr(args, f"{prefix}_log_g_factor_spread", cfg.log_g_complexity_influence_factor_spread)

    # Modulator configs (scalar from args, applies to all levels of this stack)
    # Curvature Modulator
    cfg.curvature_modulator.enabled = getattr(args, f"{prefix}_mod_c_enabled", cfg.curvature_modulator.enabled)
    cfg.curvature_modulator.mlp_hidden_dim_ratio = getattr(args, f"{prefix}_mod_c_mlp_ratio", cfg.curvature_modulator.mlp_hidden_dim_ratio)
    # ... (similarly for other modulator params and for scale, spread, ld, boundary_points modulators) ...
    # Example for LD modulator:
    cfg.level_descriptor_modulator.enabled = getattr(args, f"{prefix}_mod_ld_enabled", cfg.level_descriptor_modulator.enabled)

    cfg.tangent_combiner_interaction_type = getattr(args, f"{prefix}_tangent_combiner_interaction", cfg.tangent_combiner_interaction_type)
    cfg.mha_light_num_heads = getattr(args, f"{prefix}_mha_heads", cfg.mha_light_num_heads)
    cfg.use_tangent_flow = getattr(args, f"{prefix}_use_tangent_flow", cfg.use_tangent_flow)
    cfg.initial_learnable_tangent_flow_scale = getattr(args, f"{prefix}_tangent_flow_init_scale", cfg.initial_learnable_tangent_flow_scale)

    cfg.use_level_descriptors = getattr(args, f"{prefix}_use_level_descriptors", cfg.use_level_descriptors)
    # ... and so on for all fields in WuBuLevelConfig that should be settable via argparse ...

    return cfg

def _populate_wubu_transform_config_from_args(args: argparse.Namespace, prefix: str, transform_idx: int, num_total_transforms: int) -> WuBuTransformConfig:
    cfg = WuBuTransformConfig()
    
    cfg.num_aniso_blocks = getattr(args, f"{prefix}_transform_num_aniso_blocks", cfg.num_aniso_blocks)
    # ... similarly for other transform config fields ...
    cfg.use_rotation_in_transform = getattr(args, f"{prefix}_use_rotation", cfg.use_rotation_in_transform) # This was wubu_s_use_rotation
    cfg.phi_influence_rotation_init = getattr(args, f"{prefix}_phi_influence_rotation_init", cfg.phi_influence_rotation_init) # This was wubu_s_phi_rot_init
    cfg.rotation_type = getattr(args, f"{prefix}_transform_rotation_type", cfg.rotation_type)
    cfg.rotation_block_dim = getattr(args, f"{prefix}_transform_rotation_block_dim", cfg.rotation_block_dim)
    
    return cfg


def create_wubu_stack_config_from_args(args: argparse.Namespace, prefix: str) -> WuBuStackConfig:
    """Creates a WuBuStackConfig object from parsed argparse arguments."""
    stack_cfg = WuBuStackConfig(stack_name=prefix)
    stack_cfg.num_levels = getattr(args, f"{prefix}_num_levels", 0)
    
    # Global stack properties from args
    stack_cfg.dropout = getattr(args, "wubu_dropout", stack_cfg.dropout) # Global wubu_dropout
    stack_cfg.relative_vector_aggregation = getattr(args, f"{prefix}_rel_vec_agg", stack_cfg.relative_vector_aggregation)
    stack_cfg.phi_influence_curvature_stack_global = getattr(args, f"{prefix}_phi_influence_curvature", stack_cfg.phi_influence_curvature_stack_global) # Was wubu_s_phi_curvature
    
    # g_W factors from args (these are stack-global, not per-level from args)
    stack_cfg.g_w_input_dim_factor = getattr(args, f"{prefix}_gw_in_dim_factor", stack_cfg.g_w_input_dim_factor)
    # ... (all other g_w factors) ...

    if stack_cfg.num_levels > 0:
        for i in range(stack_cfg.num_levels):
            level_specific_cfg = _populate_wubu_level_config_from_args(args, prefix, i, stack_cfg.num_levels)
            stack_cfg.levels_config.append(level_specific_cfg)
        
        num_transforms = max(0, stack_cfg.num_levels - 1)
        for i in range(num_transforms):
            transform_specific_cfg = _populate_wubu_transform_config_from_args(args, prefix, i, num_transforms)
            stack_cfg.transforms_config.append(transform_specific_cfg)
    
    # Call post_init validation (or it will be called automatically if WuBuStackConfig is a dataclass)
    try:
        stack_cfg.__post_init__() # Manual call if not auto
    except ValueError as e:
        # This indicates a fundamental mismatch that should be caught by argparse or earlier validation
        # For now, re-raise or log critically.
        raise ValueError(f"Configuration error for WuBu stack '{prefix}': {e}")

    return stack_cfg


def parse_arguments():
    parser = argparse.ArgumentParser(description="WuBuSpecTrans_v0.2.0: Total Strategy Audio VAE-GAN")

    # --- Group: Core Paths and DDP/General Setup ---
    # ... (same as v0.1.1)
    parser.add_argument('--audio_dir_path', type=str, default="demo_audio_data_dir", help="Path to directory containing audio files or a single audio file.")
    parser.add_argument('--checkpoint_dir',type=str, default='wubuspectrans_checkpoints_v020', help="Directory for checkpoints.") # Updated default
    parser.add_argument('--load_checkpoint', type=str, default=None, help="Path to checkpoint to load.")
    parser.add_argument('--load_strict', action='store_true', help="Use strict=True when loading model state_dict.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DDP local rank (set by launch utility).")
    parser.add_argument('--seed',type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--num_workers',type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument('--use_amp', action='store_true', help="Enable Automatic Mixed Precision training.")
    parser.add_argument('--detect_anomaly',action='store_true', help="Enable PyTorch autograd anomaly detection (for debugging).")
    parser.add_argument('--ddp_find_unused_params_d', action='store_true', help="Set find_unused_parameters=True for DDP wrapped Discriminators.")


    # --- Group: Training Hyperparameters ---
    # ... (same as v0.1.1)
    parser.add_argument('--epochs', type=int, default=1500, help="Total training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per GPU.")
    parser.add_argument('--grad_accum_steps',type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument('--learning_rate_gen',type=float,default=1e-4, help="Learning rate for Generator/VAE.")
    parser.add_argument('--learning_rate_disc',type=float,default=1e-4, help="Learning rate for the primary Discriminator.")
    parser.add_argument('--learning_rate_disc_alt',type=float,default=None, help="Specific LR for alt Discriminator (defaults to learning_rate_disc).")
    parser.add_argument('--risgd_max_grad_norm',type=float,default=1.0, help="Max grad norm for Riemannian SGD per-parameter clipping.")
    parser.add_argument('--global_max_grad_norm',type=float,default=5.0, help="Global gradient clipping norm for optimizers (0 to disable).")


    # --- Group: Loss Weights ---
    # ... (same as v0.1.1)
    parser.add_argument('--lambda_recon', type=float, default=10.0, help="Weight for VAE reconstruction loss.")
    parser.add_argument('--lambda_kl', type=float, default=0.01, help="Initial base weight for VAE KL divergence loss.")
    parser.add_argument('--lambda_gan', type=float, default=1.0, help="Weight for GAN adversarial loss (Generator part).")


    # --- Group: Audio Processing & Dataset ---
    # ... (same as v0.1.1)
    parser.add_argument('--sample_rate', type=int, default=22050, help="Target sample rate.")
    parser.add_argument('--n_fft', type=int, default=1024, help="FFT window size.")
    parser.add_argument('--hop_length', type=int, default=256, help="Hop length for STFT.")
    parser.add_argument('--n_mels', type=int, default=128, help="Number of Mel bands.")
    parser.add_argument('--fmin', type=float, default=30.0, help="Minimum frequency for Mel bands.")
    parser.add_argument('--fmax', type=float, default=None, help="Maximum frequency for Mel bands (None for sr/2).")
    parser.add_argument('--segment_duration_sec', type=float, default=1.0, help="Duration of audio segments.")
    parser.add_argument('--segment_overlap_sec', type=float, default=0.0, help="Overlap between audio segments.")
    parser.add_argument('--db_norm_min', type=float, default=-80.0, help="Min dB for Mel spectrogram normalization.")
    parser.add_argument('--db_norm_max', type=float, default=0.0, help="Max dB for Mel spectrogram normalization.")
    parser.add_argument('--preload_audio_dataset_to_ram', action='store_true', help="Preload audio dataset (as Mels) into RAM.")
    parser.add_argument('--validation_audio_dir_path', type=str, default=None, help="Path to separate validation audio files.")
    parser.add_argument('--validation_split_fraction', type=float, default=0.1, help="Fraction of main dataset for validation.")


    # --- Group: GAAD (Spectrogram Regions) & DCT Processing ---
    # ... (same as v0.1.1)
    parser.add_argument('--gaad_num_regions', type=int, default=10, help="Number of GAAD regions.")
    parser.add_argument('--gaad_decomposition_type', type=str, default="hybrid", choices=["spiral", "subdivide", "hybrid"], help="GAAD type.")
    parser.add_argument('--gaad_min_size_px', type=int, default=4, help="Min GAAD region size.")
    parser.add_argument('--region_proc_size_t', type=int, default=16, help="Time dim of processed GAAD region (DCT block).")
    parser.add_argument('--region_proc_size_f', type=int, default=16, help="Frequency dim of processed GAAD region (DCT block).")
    parser.add_argument('--dct_norm_type', type=str, default="tanh", choices=["none", "global_scale", "tanh"], help="DCT normalization.")
    parser.add_argument('--dct_norm_global_scale', type=float, default=100.0, help="Global scaling for DCT if global_scale.")
    parser.add_argument('--dct_norm_tanh_scale', type=float, default=30.0, help="Scaling before tanh for DCT if tanh.")


    # --- Group: Model Architecture (VAE & Discriminator Base) ---
    # ... (same as v0.1.1, but some WuBu output dims might be derived from WuBuStackConfig if not specified)
    parser.add_argument('--latent_dim', type=int, default=256, help="VAE latent space dimensionality.")
    parser.add_argument('--encoder_initial_tangent_dim', type=int, default=128, help="Input tangent dim to WuBu-S in Encoder (becomes input_tangent_dim for wubu_s_stack_config).")
    # output_dim for WuBu-S is now implicitly defined by the last level of wubu_s_stack_config and its output_tangent_projection
    # Or, can add an explicit --wubu_s_final_output_dim if needed, which would be output_tangent_dim for the stack.
    # Same for wubu_d_output_dim. For now, assume these are set by the stack's design or a separate output_projection.
    # For simplicity, let's keep these as they were, they'll be the target output_tangent_dim for the WuBu stacks.
    parser.add_argument('--wubu_s_output_dim_encoder', type=int, default=128, help="Target output dim for WuBu-S encoder stack.")
    parser.add_argument('--wubu_d_output_dim', type=int, default=64, help="Target output dim for WuBu-D stack (if DCT D).")

    parser.add_argument('--disc_input_type', type=str, default="mel", choices=["mel", "dct"], help="Default D input type if not overridden.")
    parser.add_argument('--disc_apply_spectral_norm', action='store_true', help="Apply spectral norm to D's conv/linear layers.")
    parser.add_argument('--disc_base_disc_channels', type=int, default=64, help="Base channels for D's CNN (Mel input).")
    parser.add_argument('--disc_max_disc_channels', type=int, default=512, help="Max channels for D's CNN (Mel input).")
    parser.add_argument('--disc_target_final_feature_dim', type=int, default=4, help="Target spatial dim of D's CNN feature map (Mel input).")


    # --- Group: WuBu Stack Configurations (Now more detailed for Total Strategy) ---
    parser.add_argument('--wubu_dropout', type=float, default=0.1, help="Global dropout for WuBu layers.")

    wubu_prefixes = ["wubu_s", "wubu_g", "wubu_d_pri", "wubu_d_alt"] # Primary and Alt D stacks
    for prefix in wubu_prefixes:
        # Core stack structure
        parser.add_argument(f'--{prefix}_num_levels', type=int, default=2 if prefix in ["wubu_s", "wubu_g"] else 1, help=f"Num levels for {prefix} stack.")
        parser.add_argument(f'--{prefix}_hyperbolic_dims', nargs='+', type=int, default=[64,32] if prefix=="wubu_s" else ([128,256] if prefix=="wubu_g" else [64]), help=f"List of hyperbolic dims for {prefix} levels.")
        parser.add_argument(f'--{prefix}_initial_curvatures', nargs='+', type=float, default=[1.0,0.8] if prefix=="wubu_s" else ([0.8,1.0] if prefix=="wubu_g" else [0.7]), help=f"List of initial curvatures for {prefix} levels.")
        parser.add_argument(f'--{prefix}_initial_scales', nargs='+', type=float, default=[1.0], help=f"List of initial scales for {prefix} levels (cycles if shorter).")
        parser.add_argument(f'--{prefix}_initial_spread_values', nargs='+', type=float, default=[0.1], help=f"List of initial spread values for {prefix} levels (cycles).")
        parser.add_argument(f'--{prefix}_boundary_points_per_level', nargs='+', type=int, default=[0], help=f"List of boundary points for {prefix} levels (cycles).")
        parser.add_argument(f'--{prefix}_rel_vec_agg', type=str, default="sum", choices=["sum", "mean", "max_norm", "none"], help=f"Relative vector aggregation for {prefix}.")
        parser.add_argument(f'--{prefix}_phi_influence_curvature', action='store_true', help=f"Enable PHI influence on curvature for {prefix} stack.")
        
        # Log-g complexity influence
        parser.add_argument(f'--{prefix}_log_g_factor_curv', type=float, default=0.0, help=f"Factor for log(g_W) on curvature for {prefix} (0 to disable).")
        parser.add_argument(f'--{prefix}_log_g_factor_scale', type=float, default=0.0, help=f"Factor for log(g_W) on scale for {prefix}.")
        parser.add_argument(f'--{prefix}_log_g_factor_spread', type=float, default=0.0, help=f"Factor for log(g_W) on spread for {prefix}.")
        # g_W calculation factors (stack-global)
        parser.add_argument(f'--{prefix}_gw_in_dim_factor', type=float, default=0.1, help=f"g_W factor for input_dim to {prefix} stack.")
        # ... add other gw factors if they are stack-specific, or use global ones below.

        # Per-level modulator MLP enable flags (scalar, applies to all levels if stack uses dynamic)
        parser.add_argument(f'--{prefix}_mod_c_enabled', action='store_true', help=f"Enable dynamic curvature modulation for {prefix}.")
        parser.add_argument(f'--{prefix}_mod_s_enabled', action='store_true', help=f"Enable dynamic scale modulation for {prefix}.")
        parser.add_argument(f'--{prefix}_mod_sigma_enabled', action='store_true', help=f"Enable dynamic spread modulation for {prefix}.")
        parser.add_argument(f'--{prefix}_mod_ld_enabled', action='store_true', help=f"Enable dynamic level descriptor modulation for {prefix}.")
        parser.add_argument(f'--{prefix}_mod_bpts_enabled', action='store_true', help=f"Enable dynamic boundary points modulation for {prefix}.")
        # Shared modulator MLP hyperparams (can be overridden per modulator type if needed by adding more args)
        parser.add_argument(f'--{prefix}_mod_mlp_ratio', type=float, default=0.25, help=f"Hidden dim ratio for modulator MLPs in {prefix}.")

        # Per-level tangent processing
        parser.add_argument(f'--{prefix}_tangent_combiner_interaction', type=str, default="concat", choices=["concat", "mha_light", "bilinear_pool"], help=f"Interaction in tangent_combiner for {prefix}.")
        parser.add_argument(f'--{prefix}_mha_heads', type=int, default=2, help=f"Num heads for MHA-light if used in {prefix}.")
        parser.add_argument(f'--{prefix}_use_tangent_flow', action='store_true', help=f"Use tangent flow in {prefix} levels.")
        parser.add_argument(f'--{prefix}_tangent_flow_init_scale', type=float, default=0.1, help=f"Initial learnable scale for tangent flow in {prefix}.")
        parser.add_argument(f'--{prefix}_use_level_descriptors', action='store_true', default=True, help=f"Use level descriptors in {prefix}.") # Default true

        # Inter-level Transform (applies to all transforms in this stack)
        parser.add_argument(f'--{prefix}_transform_num_aniso_blocks', type=int, default=1, help=f"Num anisotropic blocks in transforms for {prefix}.")
        parser.add_argument(f'--{prefix}_use_rotation', action='store_true', help=f"Enable rotation in transforms for {prefix}.")
        parser.add_argument(f'--{prefix}_phi_influence_rotation_init', action='store_true', help=f"Enable PHI influence on rotation init for {prefix} transforms.")
        parser.add_argument(f'--{prefix}_transform_rotation_type', type=str, default="full_svd_orthogonal", help=f"Rotation type for {prefix} transforms.")
        parser.add_argument(f'--{prefix}_transform_rotation_block_dim', type=int, default=4, help=f"Rotation block dim for {prefix} transforms.")

    # Global context embedding config (shared by all WuBu stacks if they use dynamic components)
    parser.add_argument('--global_ctx_emb_dim', type=int, default=16, help="Dim for global context embedding.")
    parser.add_argument('--global_ctx_no_epoch_frac', action='store_false', dest='global_ctx_use_epoch_frac', help="Disable epoch fraction in global context.")
    parser.add_argument('--global_ctx_no_gstep_frac', action='store_false', dest='global_ctx_use_gstep_frac', help="Disable gstep fraction in global context.")
    parser.set_defaults(global_ctx_use_epoch_frac=True, global_ctx_use_gstep_frac=True)


    # --- Group: Q-Learning Controller (General & Lambda_KL) ---
    # ... (same as v0.1.1, potentially add new Q-Ctrl specific args for meta-adaptivity, new actions)
    parser.add_argument('--q_controller_enabled',action='store_true', help="Enable HAKMEMQController.")
    parser.add_argument('--reset_q_controllers_on_load', action='store_true', help="Force reset Q-controllers on load.")
    # Q-template params (can be overridden by specific Q-Ctrl type if needed)
    parser.add_argument('--q_base_lr', type=float, default=0.01)
    parser.add_argument('--q_base_gamma', type=float, default=0.90)
    parser.add_argument('--q_base_eps_start', type=float, default=0.5)
    # ... other q_learn_cfg_template fields ...

    parser.add_argument('--lambda_kl_update_interval', type=int, default=100, help="Global steps between Lambda_KL Q-controller updates.")
    parser.add_argument('--min_lambda_kl_q_control', type=float, default=1e-7, help="Min value for base lambda_kl by Q-ctrl.")
    parser.add_argument('--max_lambda_kl_q_control', type=float, default=0.2, help="Max value for base lambda_kl by Q-ctrl.")
    parser.add_argument('--q_lkl_scale_options', nargs='+', type=float, default=[0.80, 0.90, 1.0, 1.10, 1.20], help="Scale options for LKL Q-ctrl.")
    parser.add_argument('--q_lkl_action_probation_steps', type=int, default=None, help="Probation steps for LKL Q-Ctrl's lambda_kl scaling action.")


    # --- Group: Heuristic Interventions & Discriminator Switching ---
    # ... (largely same as v0.1.1, ensure all relevant args are present)
    parser.add_argument('--enable_heuristic_interventions', action='store_true', help="Globally enable/disable advanced heuristic interventions.")
    parser.add_argument('--enable_heuristic_disc_switching', action='store_true', help="Enable heuristic D switching.")
    parser.add_argument('--initial_disc_type', type=str, default=None, choices=['mel', 'dct'], help="Force initial active D type (overrides disc_input_type if switching on).")
    parser.add_argument('--heuristic_check_interval', type=int, default=None, help="Global steps between heuristic checks (default: log_interval).")
    # ... (all other heuristic threshold and action param args from v0.1.1) ...
    parser.add_argument('--heuristic_short_term_history_len', type=int, default=7)
    parser.add_argument('--heuristic_trigger_count_thresh', type=int, default=2)
    parser.add_argument('--disc_switch_check_interval', type=int, default=50) # Redundant if heuristic_check_interval covers it
    parser.add_argument('--disc_switch_min_steps_between', type=int, default=250)
    parser.add_argument('--disc_switch_problem_state_count_thresh', type=int, default=2)
    parser.add_argument('--heuristic_d_strong_thresh', type=float, default=0.25)
    parser.add_argument('--heuristic_d_weak_thresh', type=float, default=1.0)
    parser.add_argument('--heuristic_d_very_weak_thresh', type=float, default=1.8)
    parser.add_argument('--heuristic_g_stalled_thresh', type=float, default=1.5)
    parser.add_argument('--heuristic_g_winning_thresh', type=float, default=0.2)
    parser.add_argument('--heuristic_g_very_much_winning_thresh', type=float, default=0.05)
    parser.add_argument('--heuristic_kl_high_thresh', type=float, default=25.0)
    parser.add_argument('--heuristic_recon_stagnation_improvement_thresh_rel', type=float, default=0.001)
    parser.add_argument('--target_good_recon_thresh_heuristic', type=float, default=0.03)
    parser.add_argument('--heuristic_q_reward_stagnation_thresh', type=float, default=-0.25)
    parser.add_argument('--heuristic_recon_boost_factor', type=float, default=1.8)
    parser.add_argument('--lambda_feat_match_heuristic', type=float, default=0.75)
    parser.add_argument('--lambda_g_easy_win_penalty_heuristic', type=float, default=1.5)
    parser.add_argument('--heuristic_active_d_lr_boost_factor', type=float, default=1.8)
    parser.add_argument('--heuristic_d_q_explore_boost_epsilon', type=float, default=0.7)
    parser.add_argument('--heuristic_d_q_explore_duration', type=int, default=10)
    parser.add_argument('--force_start_epoch_on_load', type=int, default=None)
    parser.add_argument('--force_start_gstep_on_load', type=int, default=None)


    # --- Group: Logging, Sampling, Validation & Checkpointing ---
    # ... (same as v0.1.1)
    parser.add_argument('--log_interval',type=int, default=20, help="Log training stats every N global steps.")
    parser.add_argument('--save_interval',type=int, default=500, help="Save intermediate checkpoint every N global steps (0 to disable).")
    parser.add_argument('--save_epoch_interval_epochs', type=int, default=1, help="Save checkpoint every N epochs (0 to disable this type of save).") # Renamed for clarity
    parser.add_argument('--validation_interval_epochs', type=int, default=1, help="Run validation every N epochs (0 to disable).")
    parser.add_argument('--disable_val_tqdm', action='store_true', help="Disable tqdm progress bar during validation.")
    parser.add_argument('--wandb',action='store_true', help="Enable WandB logging.")
    parser.add_argument('--wandb_project',type=str,default='WuBuSpecTransV020_Total', help="WandB project name.") # Updated name
    parser.add_argument('--wandb_run_name',type=str,default=None, help="WandB run name (auto-generated if None).")
    parser.add_argument('--wandb_log_train_recon_interval', type=int, default=100, help="Log train recon Mel to WandB every N global steps.")
    parser.add_argument('--train_target_log_freq_multiplier', type=int, default=5, help="Log train target mels N times less frequently than train recon mels.")
    parser.add_argument('--wandb_log_fixed_noise_samples_interval', type=int, default=250, help="Log fixed noise generated Mel to WandB every N global steps.")
    parser.add_argument('--use_lpips_for_mel_verification', action='store_true', help="Use LPIPS for Mel quality during validation.")
    parser.add_argument('--val_primary_metric', type=str, default="avg_val_lpips_mel",
                        choices=["avg_val_recon_dct_mse", "avg_val_mel_mse", "avg_val_psnr_mel", "avg_val_ssim_mel", "avg_val_lpips_mel"],
                        help="Primary metric for choosing best checkpoint.")
    parser.add_argument('--num_val_samples_to_log', type=int, default=3, help="Number of validation samples to log to WandB.")
    parser.add_argument('--demo_num_samples', type=int, default=5, help="Number of demo Mel spectrograms to generate at end.")


    parsed_args = parser.parse_args()

    if not TORCH_DCT_AVAILABLE: # Should be checked before this point but good fallback
        parser.error("torch-dct library is required but not found. Please install it: 'pip install torch-dct'")

    if parsed_args.heuristic_check_interval is None: # Default heuristic_check_interval
        parsed_args.heuristic_check_interval = parsed_args.disc_switch_check_interval if parsed_args.enable_heuristic_disc_switching else parsed_args.log_interval
    
    if parsed_args.enable_heuristic_disc_switching and parsed_args.initial_disc_type is None:
        parsed_args.initial_disc_type = parsed_args.disc_input_type

    # Note: `validate_wubu_config_for_argparse` used in v0.1.1 to ensure list lengths match num_levels
    # This is now handled by `_populate_wubu_level_config_from_args` and `create_wubu_stack_config_from_args`.
    # The old validation can be removed or adapted if explicit pre-check of args is still desired.
    # For now, assuming the new config creation process handles this.

    return parsed_args


def main():
    args = parse_arguments()
    ddp_active = "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE",1)) > 1
    if ddp_active:
        rank=int(os.environ["RANK"])
        local_rank=int(os.environ["LOCAL_RANK"])
        world_size=int(os.environ["WORLD_SIZE"])
        init_process_group(backend="nccl", init_method="env://") # Ensure init_method for robustness
        device=torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank=0; local_rank=0; world_size=1
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda' and torch.cuda.is_available(): torch.cuda.set_device(device)

    am_main_process = (rank == 0)
    base_logger_name = "WuBuSpecTransV02" # Updated version
    root_logger_main = logging.getLogger() # Get root logger
    for handler in root_logger_main.handlers[:]: root_logger_main.removeHandler(handler) # Clear existing handlers
    # Setup basicConfig again, this time it will be the only one
    log_level_main = logging.INFO if am_main_process else logging.WARNING
    logging.basicConfig(level=log_level_main,
                        format=f'%(asctime)s R{rank} %(name)s:%(lineno)d %(levelname)s %(message)s',
                        force=True) 
    
    current_logger_main_exec = logging.getLogger(f"{base_logger_name}.MainExec") # More specific name
    current_logger_main_exec.info(f"--- {base_logger_name} (R{rank}/{world_size}, Dev {device}, DDP:{ddp_active}, AMP:{args.use_amp}) ---")
    seed_everything(args.seed, rank, world_size)

    if args.detect_anomaly: torch.autograd.set_detect_anomaly(True); current_logger_main_exec.warning("Autograd anomaly detection ENABLED.")
    if am_main_process: current_logger_main_exec.info(f"Effective Args: {vars(args)}")

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb is not None: # Check wandb too
        run_name_main = args.wandb_run_name if args.wandb_run_name else f"wubuspectrans_v020_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        try:
            current_wandb_run_id = wandb.util.generate_id() if wandb.run is None else wandb.run.id # type: ignore
            wandb.init(project=args.wandb_project, name=run_name_main, config=vars(args), resume="allow", id=current_wandb_run_id) # type: ignore
            current_logger_main_exec.info(f"WandB initialized for run: {run_name_main} (ID: {current_wandb_run_id}), Project: {args.wandb_project}")
        except Exception as e_wandb_main: current_logger_main_exec.error(f"WandB initialization failed: {e_wandb_main}", exc_info=True); args.wandb = False

    # --- Create Config Dataclasses from args ---
    global_ctx_cfg_main = WuBuGlobalContextConfig(
        embedding_dim=args.global_ctx_emb_dim,
        use_epoch_frac=args.global_ctx_use_epoch_frac,
        use_gstep_frac=args.global_ctx_use_gstep_frac
    )
    wubu_s_enc_cfg_main = create_wubu_stack_config_from_args(args, "wubu_s")
    wubu_g_gen_cfg_main = create_wubu_stack_config_from_args(args, "wubu_g")
    # For Discriminators, they might not use WuBu (e.g. CNN). Check num_levels.
    wubu_d_pri_cfg_main: Optional[WuBuStackConfig] = None
    if getattr(args, "wubu_d_pri_num_levels", 0) > 0: # Check if WuBu is configured for primary D
        wubu_d_pri_cfg_main = create_wubu_stack_config_from_args(args, "wubu_d_pri")
    wubu_d_alt_cfg_main: Optional[WuBuStackConfig] = None
    if getattr(args, "wubu_d_alt_num_levels", 0) > 0: # Check for alternative D
        wubu_d_alt_cfg_main = create_wubu_stack_config_from_args(args, "wubu_d_alt")

    # Discriminator specific configs (CNN params, input_type) - these are simpler dicts
    disc_pri_specific_cfg_main = {
        "input_type": args.initial_disc_type if args.initial_disc_type is not None else args.disc_input_type, # Determine primary D's type
        "apply_spectral_norm": args.disc_apply_spectral_norm,
        "base_disc_channels": args.disc_base_disc_channels,
        "max_disc_channels": args.disc_max_disc_channels,
        "target_mel_disc_final_feature_dim": args.disc_target_final_feature_dim
    }
    alt_disc_type_main = 'mel' if disc_pri_specific_cfg_main["input_type"] == 'dct' else 'dct' # Simple toggle for alt
    disc_alt_specific_cfg_main = {
        "input_type": alt_disc_type_main,
        "apply_spectral_norm": args.disc_apply_spectral_norm, # Use same spectral norm policy for now
        "base_disc_channels": args.disc_base_disc_channels, # Same CNN params for simplicity
        "max_disc_channels": args.disc_max_disc_channels,
        "target_mel_disc_final_feature_dim": args.disc_target_final_feature_dim
    }

    # Q-Learning Config Template (from DEFAULT_CONFIG_QLEARN_HYBRID and args)
    q_learn_cfg_template_main = DEFAULT_CONFIG_QLEARN_HYBRID.copy()
    q_learn_cfg_template_main["q_learning_rate"] = args.q_base_lr # Example of overriding from args
    # ... (populate other fields of q_learn_cfg_template_main from args if needed) ...
    q_learn_cfg_template_main["lambda_kl_scale_options"] = args.q_lkl_scale_options # Specific for LKL Q-Ctrl
    if args.q_lkl_action_probation_steps is not None:
        q_learn_cfg_template_main["lkl_num_probation_steps"] = args.q_lkl_action_probation_steps # For LKL Q-Ctrl
    
    # Audio and GAAD configs for model instantiation (simpler dicts, as before)
    segment_samples_main = int(args.segment_duration_sec * args.sample_rate)
    num_time_frames_main = math.ceil(segment_samples_main / args.hop_length)
    audio_config_main = {
        "sample_rate": args.sample_rate, "n_fft": args.n_fft, "hop_length": args.hop_length,
        "n_mels": args.n_mels, "fmin": args.fmin, "fmax": args.fmax,
        "segment_duration_sec": args.segment_duration_sec,
        "region_proc_size_t": args.region_proc_size_t, "region_proc_size_f": args.region_proc_size_f,
        "wubu_s_output_dim_encoder": args.wubu_s_output_dim_encoder, # Target for WuBu-S stack
        "wubu_d_output_dim": args.wubu_d_output_dim, # Target for WuBu-D stack
        "num_time_frames_for_1s_segment": num_time_frames_main,
    }
    gaad_config_main = {
        "num_regions": args.gaad_num_regions, "decomposition_type": args.gaad_decomposition_type,
        "min_size_px": args.gaad_min_size_px
    }
    if am_main_process:
        current_logger_main_exec.info(f"GlobalCtxCfg: {global_ctx_cfg_main}")
        current_logger_main_exec.info(f"WuBuS_Enc_Cfg: {wubu_s_enc_cfg_main.stack_name}, Levels: {wubu_s_enc_cfg_main.num_levels}")
        current_logger_main_exec.info(f"WuBuG_Gen_Cfg: {wubu_g_gen_cfg_main.stack_name}, Levels: {wubu_g_gen_cfg_main.num_levels}")
        if wubu_d_pri_cfg_main: current_logger_main_exec.info(f"WuBuD_Pri_Cfg: {wubu_d_pri_cfg_main.stack_name}, Levels: {wubu_d_pri_cfg_main.num_levels}")
        else: current_logger_main_exec.info(f"Primary D is not WuBu-based (Type: {disc_pri_specific_cfg_main['input_type']}).")
        if wubu_d_alt_cfg_main: current_logger_main_exec.info(f"WuBuD_Alt_Cfg: {wubu_d_alt_cfg_main.stack_name}, Levels: {wubu_d_alt_cfg_main.num_levels}")
        else: current_logger_main_exec.info(f"Alternative D is not WuBu-based (Type: {disc_alt_specific_cfg_main['input_type']}).")

    # Model (VAE components)
    model_main = WuBuSpecTransNet(args, audio_config_main, gaad_config_main, 
                                  wubu_s_enc_cfg_main, wubu_g_gen_cfg_main, global_ctx_cfg_main).to(device)

    if am_main_process and args.wandb and WANDB_AVAILABLE and wandb.run:
        wandb.watch(model_main, log="all", log_freq=max(100, args.log_interval * 10), log_graph=False) # type: ignore

    if ddp_active: model_main = DDP(model_main, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) # find_unused can be slow

    # Dataset loading (same as before)
    # ... (audio_files_list, dummy audio creation, Dataset instantiation, split, DataLoaders) ...
    audio_files_list_main = []
    audio_dir_path_obj_main = Path(args.audio_dir_path)
    if audio_dir_path_obj_main.is_dir():
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]: audio_files_list_main.extend([str(p) for p in audio_dir_path_obj_main.rglob(ext)])
    elif audio_dir_path_obj_main.is_file(): audio_files_list_main.append(str(audio_dir_path_obj_main))
    if not audio_files_list_main and "demo_audio_data" in args.audio_dir_path: # Create dummy if path suggests demo and is empty
        if am_main_process:
            demo_dir_main = Path(args.audio_dir_path); demo_dir_main.mkdir(parents=True, exist_ok=True)
            dummy_audio_path_main = demo_dir_main / "dummy_sine_wubuspectrans_v020.wav"
            if not dummy_audio_path_main.exists():
                current_logger_main_exec.info(f"Attempting to create dummy audio: {dummy_audio_path_main}...")
                try:
                    # soundfile already imported
                    sr_dummy_main = args.sample_rate; duration_dummy_main = 5.0
                    t_dummy_main = np.linspace(0, duration_dummy_main, int(sr_dummy_main * duration_dummy_main), endpoint=False)
                    wav_dummy_main = (0.3*np.sin(2*np.pi*220.0*t_dummy_main) + 0.2*np.sin(2*np.pi*440.0*t_dummy_main)).astype(np.float32)
                    soundfile.write(str(dummy_audio_path_main), wav_dummy_main, sr_dummy_main)
                    current_logger_main_exec.info(f"Dummy audio created: {dummy_audio_path_main}")
                    audio_files_list_main.append(str(dummy_audio_path_main))
                except Exception as e_dummy_main: current_logger_main_exec.error(f"Error creating dummy audio: {e_dummy_main}", exc_info=True)
        if ddp_active: torch.distributed.barrier() # Ensure dummy file creation is seen by all ranks if main created it
    if not audio_files_list_main: current_logger_main_exec.error(f"No audio files found in '{args.audio_dir_path}'. Exiting."); sys.exit(1)
    current_logger_main_exec.info(f"Found {len(audio_files_list_main)} audio files for main dataset pool.")

    try: full_dataset_main = AudioSegmentDataset(audio_file_paths=audio_files_list_main, args=args, preload_to_ram=args.preload_audio_dataset_to_ram)
    except Exception as e_ds_main: current_logger_main_exec.error(f"Failed to initialize main Dataset: {e_ds_main}", exc_info=True); sys.exit(1)
    if not full_dataset_main or len(full_dataset_main) == 0: current_logger_main_exec.error("Main dataset is empty. Exiting."); sys.exit(1)

    train_dataset_main: Union[AudioSegmentDataset, SubsetRandomSampler] = full_dataset_main # type: ignore
    val_dataset_main: Optional[Union[AudioSegmentDataset, SubsetRandomSampler]] = None # type: ignore
    num_total_samples_main = len(full_dataset_main)
    val_audio_files_list_main = []
    if args.validation_audio_dir_path:
        val_dir_path_obj_main = Path(args.validation_audio_dir_path)
        if val_dir_path_obj_main.is_dir():
            for ext_val in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]: val_audio_files_list_main.extend([str(p) for p in val_dir_path_obj_main.rglob(ext_val)])
        elif val_dir_path_obj_main.is_file(): val_audio_files_list_main.append(str(val_dir_path_obj_main))
    if val_audio_files_list_main:
        try:
            val_ds_candidate_main = AudioSegmentDataset(audio_file_paths=val_audio_files_list_main, args=args, preload_to_ram=args.preload_audio_dataset_to_ram)
            if len(val_ds_candidate_main) > 0: val_dataset_main = val_ds_candidate_main; current_logger_main_exec.info(f"Using separate validation dir: {len(val_dataset_main)} segments.")
            else: current_logger_main_exec.warning(f"Validation dir {args.validation_audio_dir_path} yielded 0 segments.")
        except Exception as e_val_ds: current_logger_main_exec.warning(f"Could not load validation dataset from '{args.validation_audio_dir_path}': {e_val_ds}.")
    if val_dataset_main is None and args.validation_split_fraction > 0.0 and num_total_samples_main > 10 :
        num_val_main = int(num_total_samples_main * args.validation_split_fraction)
        num_train_main = num_total_samples_main - num_val_main
        if num_train_main > 0 and num_val_main > 0:
            train_dataset_main, val_dataset_main = torch.utils.data.random_split(full_dataset_main, [num_train_main, num_val_main], generator=torch.Generator().manual_seed(args.seed + rank))
            current_logger_main_exec.info(f"Split main dataset: Train={len(train_dataset_main)}, Val={len(val_dataset_main)}") # type: ignore
        else: current_logger_main_exec.warning("Random split for validation resulted in 0 samples."); val_dataset_main = None; train_dataset_main = full_dataset_main
    if am_main_process: current_logger_main_exec.info(f"Final dataset sizes: Train={len(train_dataset_main)}, Val={len(val_dataset_main) if val_dataset_main else 0}") # type: ignore

    worker_init_fn_seeded_main = functools.partial(seed_worker_init_fn, base_seed=args.seed, rank=rank, world_size=world_size) if args.num_workers > 0 else None
    train_sampler_main = DistributedSampler(train_dataset_main, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if ddp_active else None # type: ignore
    train_loader_main = DataLoader(train_dataset_main, batch_size=args.batch_size, shuffle=(train_sampler_main is None), num_workers=args.num_workers, sampler=train_sampler_main, pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded_main, drop_last=True) # drop_last=True common for DDP
    val_loader_main = None
    if val_dataset_main and len(val_dataset_main) > 0: # type: ignore
        val_sampler_main = DistributedSampler(val_dataset_main, num_replicas=world_size, rank=rank, shuffle=False) if ddp_active else None # type: ignore
        val_loader_main = DataLoader(val_dataset_main, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler_main, pin_memory=(device.type == 'cuda'), worker_init_fn=worker_init_fn_seeded_main, drop_last=False) # type: ignore


    trainer = HybridTrainer(model=model_main, device=device,
                            train_loader=train_loader_main, val_loader=val_loader_main, args=args,
                            wubu_s_enc_cfg=wubu_s_enc_cfg_main, wubu_g_gen_cfg=wubu_g_gen_cfg_main,
                            wubu_d_pri_cfg=wubu_d_pri_cfg_main, wubu_d_alt_cfg=wubu_d_alt_cfg_main,
                            global_ctx_cfg=global_ctx_cfg_main,
                            q_learn_cfg_template=q_learn_cfg_template_main,
                            disc_pri_specific_cfg=disc_pri_specific_cfg_main,
                            disc_alt_specific_cfg=disc_alt_specific_cfg_main,
                            rank=rank, world_size=world_size, ddp_active=ddp_active)

    start_global_step_main, start_epoch_main = 0, 0
    if args.load_checkpoint:
        start_global_step_main, start_epoch_main = trainer.load_checkpoint(args.load_checkpoint)
    else: # New run without checkpoint
        if am_main_process: trainer.logger.info("Starting a new run from scratch. Q-controllers initializing fresh.")
        # Initialize Q-controller histories if not loading checkpoint
        initial_dummy_losses_main = { 'loss_g_total': 1.0, 'loss_g_recon': 1.0, 'loss_g_kl': 0.1, 'loss_g_adv': 0.7,
                                  'loss_d_total': 0.7, 'loss_d_real': 0.7, 'loss_d_fake': 0.7 }
        initial_dummy_wubu_geo_main = {'avg_curvature': 1.0, 'avg_scale': 1.0, 'avg_spread': 0.1, 'var_curvature': 0.05}
        if trainer.q_controller_gen: trainer.q_controller_gen.set_initial_losses(initial_dummy_losses_main, True, initial_dummy_wubu_geo_main)
        if trainer.q_controller_d_primary: trainer.q_controller_d_primary.set_initial_losses(initial_dummy_losses_main, False)
        if trainer.q_controller_d_alt: trainer.q_controller_d_alt.set_initial_losses(initial_dummy_losses_main, False)
        if trainer.lambda_kl_q_controller:
            initial_lkl_metrics_main = { 'avg_recon': 1.0, 'avg_kl_div': 0.1, 'avg_d_total': 0.7, 'val_metric': 1.0, 'current_lambda_kl_val': trainer.lambda_kl }
            trainer.lambda_kl_q_controller.set_initial_lambda_kl_metrics(initial_lkl_metrics_main)
            trainer.lambda_kl_q_controller.start_probation() # Explicitly start probation if new run

    try:
        trainer.train(start_epoch=start_epoch_main, initial_global_step=start_global_step_main)
    except KeyboardInterrupt: current_logger_main_exec.info(f"Rank {rank}: Training interrupted by user.")
    except Exception as e_train_loop: current_logger_main_exec.error(f"Rank {rank}: Training loop crashed: {e_train_loop}", exc_info=True)
    finally:
        if am_main_process:
            current_logger_main_exec.info("Finalizing run and saving final checkpoint...")
            trainer._capture_rng_states() # Capture RNG before final save
            final_metrics_to_save_main = trainer.last_val_metrics.copy() if trainer.last_val_metrics else {}
            final_metrics_to_save_main['best_val_metric_val_at_end'] = trainer.best_val_metric_val
            trainer._save_checkpoint(metrics=final_metrics_to_save_main) # Default is not intermediate, not best (epoch end)

            if args.epochs > 0 and hasattr(trainer, 'sample') and trainer.global_step > 0 and args.demo_num_samples > 0:
                current_logger_main_exec.info("Generating final demo samples (Mel Spectrograms)...")
                try:
                    final_global_ctx_raw = trainer._get_global_context_raw_features() # Get context at end of training
                    generated_mels_final = trainer.sample(num_samples=args.demo_num_samples, global_ctx_raw_features_sample=final_global_ctx_raw)
                    if generated_mels_final is not None and generated_mels_final.numel() > 0:
                        save_dir_final = Path(args.checkpoint_dir) / "demo_samples_mel_spectrograms_v020"
                        save_dir_final.mkdir(parents=True, exist_ok=True)
                        for b_idx_final in range(min(args.demo_num_samples, generated_mels_final.shape[0])):
                            mel_to_save_final = (generated_mels_final[b_idx_final, 0].cpu().clamp(-1,1) + 1) / 2.0
                            save_image_path_final = save_dir_final / f"demo_mel_sample_{b_idx_final}_ep{trainer.current_epoch+1}_final.png"
                            save_image(mel_to_save_final, str(save_image_path_final))
                        current_logger_main_exec.info(f"Saved demo Mel spectrogram images to {save_dir_final}")
                        if args.wandb and WANDB_AVAILABLE and wandb.run:
                            trainer._log_samples_to_wandb("final_demo_mel", generated_mels_final, args.demo_num_samples)
                except Exception as e_demo_final: current_logger_main_exec.error(f"Demo Mel sampling/saving error: {e_demo_final}", exc_info=True)
            
            if trainer.tb_writer: trainer.tb_writer.close() # Close TensorBoard writer
            if args.wandb and WANDB_AVAILABLE and wandb.run: wandb.finish() # type: ignore

        if ddp_active and is_initialized(): destroy_process_group()
        current_logger_main_exec.info(f"Rank {rank}: {base_logger_name} (v0.2.0 TotalStrategy) script finished.")

if __name__ == "__main__":
    # This check is primarily for torch-dct, which is critical.
    if not TORCH_DCT_AVAILABLE:
        # Logger might not be fully set up here, so print.
        print("CRITICAL ERROR: torch-dct library not found or failed to import. WuBuSpecTrans cannot run without it. Please install with 'pip install torch-dct'.")
        sys.exit(1)
    main()

        