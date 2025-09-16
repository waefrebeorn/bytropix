# =================================================================================================
#
#                    PHASE 2.5: VQ-GAN TOKENIZER TRAINING
#
#     A Deterministic, Physics-Informed Framework for Structured Media Synthesis
#                   (Advanced Perceptual Loss & PID Control)
#
# =================================================================================================

import os
import sys
import argparse
import pickle
import time
import math
import signal
from pathlib import Path
from typing import Any, NamedTuple, Optional, Dict, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import atexit

# --- Environment and JAX Setup ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / ".jax_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at: {cache_dir} ---")

    def _jax_shutdown():
        """Function to be called at script exit to ensure JAX cleans up."""
        import jax
        jax.clear_caches()
        print("--- JAX cache finalized. Script exiting. ---")
    atexit.register(_jax_shutdown)

except NameError:
    cache_dir = Path.home() / ".jax_cache_global"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at (fallback global): {cache_dir} ---")

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
import optax
import chex
from flax import linen as nn
from flax.training import train_state
from flax.training.train_state import TrainState
from functools import partial
from tqdm import tqdm
from PIL import Image

# --- GUI & Monitoring Dependencies ---
try:
    from rich_pixels import Pixels
except ImportError:
    print("[Warning] `rich-pixels` not found. Visual preview in GUI will be disabled. Run: pip install rich-pixels")
    Pixels = None
try:

    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.align import Align
    from rich.table import Table
    from rich.text import Text
    import pynvml
    pynvml.nvmlInit()
except ImportError:
    print("[FATAL] Required dependencies missing: tensorflow, rich, pynvml, chex. Please install them.")
    sys.exit(1)


# --- JAX Configuration ---
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)


# =================================================================================================
# 1. CORE PHYSICS/AE MODEL DEFINITIONS (From Phase 1/2)
# =================================================================================================

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

class PathModulator(nn.Module):
    latent_grid_size: int
    input_image_size: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        x = images
        features = 32
        current_dim = self.input_image_size
        i = 0
        while (current_dim // 2) >= self.latent_grid_size and (current_dim // 2) > 0:
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x)
            x = nn.gelu(x)
            features *= 2
            current_dim //= 2
            i += 1
        if current_dim != self.latent_grid_size:
            x = nn.Conv(features, (1, 1), name="pre_resize_projection", dtype=self.dtype)(x)
            x = nn.gelu(x)
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x)
        x = nn.gelu(x)
        path_params = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1)

class TopologicalObserver(nn.Module):
    d_model: int
    num_path_steps: int = 16
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = path_params_grid.shape
        path_params = path_params_grid.reshape(B, H * W, 3)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
        chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path)
        m = jnp.stack([jnp.mean(t_co_steps.real,-1), jnp.std(t_co_steps.real,-1), jnp.mean(t_co_steps.imag,-1), jnp.std(t_co_steps.imag,-1)], -1)
        return nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(m).reshape(B, H, W, self.d_model)

class PositionalEncoding(nn.Module):
    num_freqs: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x*freq) for freq in freqs for f in (jnp.sin, jnp.cos)], -1)

class CoordinateDecoder(nn.Module):
    d_model: int
    num_freqs: int=10
    mlp_width: int=256
    mlp_depth: int=4
    dtype: Any=jnp.float32

    @nn.compact
    def __call__(self, feature_grid, coords):
        B, H, W, _ = feature_grid.shape
        enc_coords = PositionalEncoding(self.num_freqs, self.dtype)(coords)
        pyramid = [feature_grid] + [jax.image.resize(feature_grid, (B,H//(2**i),W//(2**i),self.d_model),'bilinear') for i in range(1,3)]
        sampled_feats = []
        for level_grid in pyramid:
            level_shape = jnp.array(level_grid.shape[1:3], dtype=self.dtype)
            coords_rescaled = (coords + 1)/2 * (level_shape - 1)
            def sample_one(grid):
                grid_chw = grid.transpose(2,0,1)
                return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw).T
            sampled_feats.append(jax.vmap(sample_one)(level_grid))
        mlp_in = jnp.concatenate([jnp.repeat(enc_coords[None,...],B,0), jnp.concatenate(sampled_feats, -1)], -1)
        h = mlp_in
        for i in range(self.mlp_depth):
            h = nn.gelu(nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h))
            if i == self.mlp_depth // 2: h = jnp.concatenate([h, mlp_in], -1)
        return nn.tanh(nn.Dense(3, name="mlp_out", dtype=self.dtype)(h))

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int
    latent_grid_size: int
    input_image_size: int
    dtype: Any = jnp.float32

    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(self.d_model, name="coord_decoder", dtype=self.dtype)

    def decode(self, path_params, coords):
        return self.coord_decoder(self.observer(path_params), coords)


# =================================================================================================
# 2. ADVANCED TOKENIZER TOOLKIT
# =================================================================================================

Q_CONTROLLER_CONFIG_NORMAL = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-5, "lr_max": 1e-3, "metric_history_len": 5000, "loss_min": 0.01, "loss_max": 0.5, "exploration_rate_q": 0.3, "min_exploration_rate": 0.05, "exploration_decay": 0.9995, "trend_window": 420, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": 10.0, "stagnation_penalty": -2.0, "warmup_steps": 420, "warmup_lr_start": 1e-6}
Q_CONTROLLER_CONFIG_FINETUNE = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.8, 0.95, 1.0, 1.05, 1.2], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-6, "lr_max": 5e-4, "metric_history_len": 5000, "loss_min": .001, "loss_max": 0.1, "exploration_rate_q": 0.15, "min_exploration_rate": 0.02, "exploration_decay": 0.9998, "warmup_steps": 200, "warmup_lr_start": 1e-7, "trend_window": 420, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": 10.0, "stagnation_penalty": -2.0}

class JaxHakmemQController:
    """A generalized Q-Learning agent to dynamically control any hyperparameter."""
    def __init__(self, initial_value:float, config:Dict[str,Any], param_name: str = "LR"):
        self.param_name = param_name
        self.config = config
        self.initial_value = initial_value
        self.current_value = initial_value
        self.q_table_size = int(self.config["q_table_size"])
        self.num_actions = int(self.config["num_lr_actions"])
        self.action_factors = self.config["lr_change_factors"]
        self.q_table = np.zeros((self.q_table_size, self.num_actions), dtype=np.float32)
        self.learning_rate_q = float(self.config["learning_rate_q"])
        self.discount_factor_q = float(self.config["discount_factor_q"])
        self.value_min = float(self.config["lr_min"])
        self.value_max = float(self.config["lr_max"])
        self.metric_history = deque(maxlen=int(self.config["metric_history_len"]))
        self.metric_min = float(self.config["loss_min"])
        self.metric_max = float(self.config["loss_max"])
        self.last_action_idx: Optional[int] = None
        self.last_state_idx: Optional[int] = None
        self.initial_exploration_rate = float(self.config["exploration_rate_q"])
        self.exploration_rate_q = self.initial_exploration_rate
        self.min_exploration_rate = float(self.config["min_exploration_rate"])
        self.exploration_decay = float(self.config["exploration_decay"])
        self.status: str = "STARTING"
        self.last_reward: float = 0.0
        self.trend_window = int(config["trend_window"])
        self.trend_history = deque(maxlen=self.trend_window)
        self.improve_threshold = float(config["improve_threshold"])
        self.regress_threshold = float(config["regress_threshold"])
        self.regress_penalty = float(config["regress_penalty"])
        self.stagnation_penalty = float(config["stagnation_penalty"])
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.warmup_start_val = float(config.get("warmup_lr_start", 1e-7))
        self._step_count = 0
        print(f"--- Q-Controller ({self.param_name}) initialized. Warmup: {self.warmup_steps} steps. Trend Window: {self.trend_window} steps ---")

    def _discretize_value(self, value: float) -> int:
        if not np.isfinite(value): return self.q_table_size // 2
        if value <= self.metric_min: return 0
        if value >= self.metric_max: return self.q_table_size - 1
        return min(int((value - self.metric_min) / ((self.metric_max - self.metric_min) / self.q_table_size)), self.q_table_size - 1)

    def _get_current_state_idx(self) -> int:
        if not self.metric_history: return self.q_table_size // 2
        return self._discretize_value(np.mean(list(self.metric_history)[-5:]))

    def choose_action(self) -> float:
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            alpha = self._step_count / self.warmup_steps
            self.current_value = self.warmup_start_val * (1 - alpha) + self.initial_value * alpha
            self.status = f"WARMUP ({self.param_name}) {self._step_count}/{self.warmup_steps}"
            return self.current_value
        
        self.last_state_idx = self._get_current_state_idx()
        if np.random.rand() < self.exploration_rate_q:
            self.last_action_idx = np.random.randint(0, self.num_actions)
        else:
            self.last_action_idx = np.argmax(self.q_table[self.last_state_idx]).item()
        
        self.current_value = np.clip(self.current_value * self.action_factors[self.last_action_idx], self.value_min, self.value_max)
        self.current_lr = self.current_value # Alias for UI
        return self.current_value

    def update_q_value(self, metric_value: float):
        self.metric_history.append(metric_value)
        self.trend_history.append(metric_value)
        if self._step_count <= self.warmup_steps: return
        if self.last_state_idx is None or self.last_action_idx is None: return
        
        reward = self._calculate_reward()
        self.last_reward = reward
        current_q = self.q_table[self.last_state_idx, self.last_action_idx]
        next_state_idx = self._get_current_state_idx()
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor_q * np.max(self.q_table[next_state_idx]) - current_q)
        self.q_table[self.last_state_idx, self.last_action_idx] = new_q
        self.exploration_rate_q = max(self.min_exploration_rate, self.exploration_rate_q * self.exploration_decay)

    def _calculate_reward(self):
        if len(self.trend_history) < self.trend_window:
            self.status = f"WARMING UP (TREND) {len(self.trend_history)}/{self.trend_window}"; return 0.0
        
        slope = np.polyfit(np.arange(self.trend_window), np.array(self.trend_history), 1)[0]
        
        if slope < -self.improve_threshold:
            self.status = f"IMPROVING (S={slope:.2e})"; reward = abs(slope) * 1000
        elif slope > self.regress_threshold:
            self.status = f"REGRESSING (S={slope:.2e})"; reward = -abs(slope) * 1000 - self.regress_penalty
        else:
            self.status = f"STAGNATED (S={slope:.2e})"; reward = self.stagnation_penalty
        return reward

    def state_dict(self) -> Dict[str, Any]:
        return {"current_value": self.current_value, "q_table": self.q_table.tolist(), "metric_history": list(self.metric_history), "exploration_rate_q": self.exploration_rate_q, "trend_history": list(self.trend_history), "_step_count": self._step_count}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_value = state_dict.get("current_value", self.current_value)
        self.q_table = np.array(state_dict.get("q_table", self.q_table.tolist()), dtype=np.float32)
        self.metric_history = deque(state_dict.get("metric_history", []), maxlen=self.metric_history.maxlen)
        self.exploration_rate_q = state_dict.get("exploration_rate_q", self.initial_exploration_rate)
        self.trend_history = deque(state_dict.get("trend_history", []), maxlen=self.trend_window)
        self._step_count = state_dict.get("_step_count", 0)

# --- VQ-GAN Model Definitions ---
class VectorQuantizer(nn.Module):
    num_codes: int
    code_dim: int
    beta: float = 0.25

    @nn.compact
    def __call__(self, z):
        codebook = self.param('codebook', nn.initializers.uniform(), (self.code_dim, self.num_codes))
        z_flat = z.reshape(-1, self.code_dim)
        d = jnp.sum(z_flat**2, axis=1, keepdims=True) - 2 * jnp.dot(z_flat, codebook) + jnp.sum(codebook**2, axis=0, keepdims=True)
        indices = jnp.argmin(d, axis=1)
        z_q = codebook.T[indices].reshape(z.shape)
        commitment_loss = self.beta * jnp.mean((jax.lax.stop_gradient(z_q) - z)**2)
        codebook_loss = jnp.mean((z_q - jax.lax.stop_gradient(z))**2)
        z_q_ste = z + jax.lax.stop_gradient(z_q - z)
        return {"quantized": z_q_ste, "indices": indices.reshape(z.shape[:-1]), "loss": commitment_loss + codebook_loss}

    def lookup(self, indices):
        codebook = self.variables['params']['codebook']
        return codebook.T[indices]

class LatentTokenizerVQGAN(nn.Module):
    num_codes: int
    code_dim: int
    latent_grid_size: int
    dtype: Any = jnp.float32

    def setup(self):
        self.enc_conv1 = nn.Conv(128, (3,3), (2,2), 'SAME', name="enc_conv1", dtype=self.dtype)
        self.enc_conv2 = nn.Conv(256, (3,3), (2,2), 'SAME', name="enc_conv2", dtype=self.dtype)
        self.enc_proj = nn.Conv(self.code_dim, (1,1), name="enc_proj", dtype=self.dtype)
        self.vq = VectorQuantizer(self.num_codes, self.code_dim, name="vq")
        self.dec_convT1 = nn.ConvTranspose(256, (3,3), (2,2), 'SAME', name="dec_convT1", dtype=self.dtype)
        self.dec_convT2 = nn.ConvTranspose(3, (3,3), (2,2), 'SAME', name="dec_convT2", dtype=self.dtype)

    def __call__(self, path_params_grid):
        target_size = self.latent_grid_size // 4
        h = nn.gelu(self.enc_conv1(path_params_grid))
        h = nn.gelu(self.enc_conv2(h))
        z_e = self.enc_proj(h)
        assert z_e.shape[1] == target_size and z_e.shape[2] == target_size, f"Incorrect spatial dim: {z_e.shape}"
        vq_out = self.vq(z_e)
        z_q = vq_out["quantized"]
        p_r = self.dec_convT2(nn.gelu(self.dec_convT1(z_q)))
        return {"reconstructed_path_params": p_r, "indices": vq_out["indices"], "vq_loss": vq_out["loss"], "pre_quant_latents": z_e}

    def encode(self, path_params_grid):
        h = nn.gelu(self.enc_conv1(path_params_grid))
        h = nn.gelu(self.enc_conv2(h))
        z_e = self.enc_proj(h)
        return self.vq(z_e)["indices"]

    def decode(self, indices):
        z_q = self.vq.lookup(indices)
        h_r = nn.gelu(self.dec_convT1(z_q))
        return self.dec_convT2(h_r)

class PatchDiscriminator(nn.Module):
    num_filters: int = 64
    num_layers: int = 3
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.num_filters, (4, 4), (2, 2), 'SAME', dtype=self.dtype)(x)
        h = nn.leaky_relu(h, 0.2)
        for i in range(1, self.num_layers):
            nf = self.num_filters * (2**i)
            h = nn.Conv(nf, (4, 4), (2, 2), 'SAME', dtype=self.dtype)(h)
            h = nn.LayerNorm(dtype=self.dtype)(h)
            h = nn.leaky_relu(h, 0.2)
        return nn.Conv(1, (4, 4), (1, 1), 'SAME', dtype=self.dtype)(h)

class GANTrainStates(NamedTuple):
    generator: TrainState
    discriminator: TrainState

# --- Perceptual Loss Suite ---

@jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    p = jnp.exp(logp)
    ent = -jnp.sum(p * logp, axis=-1)
    diff = logp + ent[..., None]
    varent = jnp.sum(p * diff**2, axis=-1)
    return ent, varent

@partial(jax.jit, static_argnames=('max_lag',))
def calculate_autocorrelation_features(patches: jnp.ndarray, max_lag: int = 8) -> jnp.ndarray:
    patches_gray = jnp.mean(patches, axis=-1, keepdims=True)
    patches_centered = patches_gray - jnp.mean(patches_gray, axis=(1, 2), keepdims=True)
    norm_factor = jnp.var(patches_centered, axis=(1, 2), keepdims=True) + 1e-6
    lags_x = jnp.arange(1, max_lag + 1)
    lags_y = jnp.arange(1, max_lag + 1)
    def _calculate_correlation_at_lag(lag_x, lag_y):
        shifted = jnp.roll(patches_centered, (lag_y, lag_x), axis=(1, 2))
        covariance = jnp.mean(patches_centered * shifted, axis=(1, 2, 3))
        return covariance / jnp.squeeze(norm_factor)
    corr_h = jax.vmap(_calculate_correlation_at_lag, in_axes=(0, None))(lags_x, 0)
    corr_v = jax.vmap(_calculate_correlation_at_lag, in_axes=(None, 0))(0, lags_y)
    corr_d = jax.vmap(_calculate_correlation_at_lag, in_axes=(0, 0))(lags_x, lags_y)
    return jnp.concatenate([corr_h.T, corr_v.T, corr_d.T], axis=-1)

def _compute_sobel_magnitude(patches: jnp.ndarray, kernel_x: jnp.ndarray, kernel_y: jnp.ndarray) -> jnp.ndarray:
    mag_channels = []
    k_x_4d = kernel_x[..., None, None]
    k_y_4d = kernel_y[..., None, None]
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    for i in range(patches.shape[-1]):
        channel_slice = patches[..., i]
        image_4d = channel_slice[..., None]
        grad_x = jax.lax.conv_general_dilated(image_4d.astype(jnp.float32), k_x_4d, window_strides=(1, 1), padding='SAME', dimension_numbers=dimension_numbers)
        grad_y = jax.lax.conv_general_dilated(image_4d.astype(jnp.float32), k_y_4d, window_strides=(1, 1), padding='SAME', dimension_numbers=dimension_numbers)
        magnitude = jnp.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        mag_channels.append(jnp.squeeze(magnitude, axis=-1))
    mag_per_channel = jnp.stack(mag_channels, axis=-1)
    return jnp.linalg.norm(mag_per_channel, axis=-1)

@jit
def calculate_edge_loss(patches1: jnp.ndarray, patches2: jnp.ndarray) -> jnp.ndarray:
    sobel_x_kernel = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
    sobel_y_kernel = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32)
    mag1 = _compute_sobel_magnitude(patches1, sobel_x_kernel, sobel_y_kernel)
    mag2 = _compute_sobel_magnitude(patches2, sobel_x_kernel, sobel_y_kernel)
    return jnp.mean(jnp.abs(mag1 - mag2))

@jit
def calculate_color_covariance_loss(patches1: jnp.ndarray, patches2: jnp.ndarray) -> jnp.ndarray:
    def get_gram_matrix(patches):
        features = patches.reshape(patches.shape[0], -1, patches.shape[-1])
        gram = jax.vmap(lambda x: x.T @ x)(features)
        return gram / (features.shape[1] * features.shape[2])
    gram1 = get_gram_matrix(patches1)
    gram2 = get_gram_matrix(patches2)
    return jnp.mean(jnp.abs(gram1 - gram2))

@jit
def calculate_ssim_loss(patches1: jnp.ndarray, patches2: jnp.ndarray, max_val: float = 2.0) -> jnp.ndarray:
    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2
    patches1_gray = jnp.mean(patches1, axis=-1)
    patches2_gray = jnp.mean(patches2, axis=-1)
    mu1 = jnp.mean(patches1_gray, axis=(1, 2))
    mu2 = jnp.mean(patches2_gray, axis=(1, 2))
    var1 = jnp.var(patches1_gray, axis=(1, 2))
    var2 = jnp.var(patches2_gray, axis=(1, 2))
    mu1_mu2 = mu1 * mu2
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    covar = jnp.mean(patches1_gray * patches2_gray, axis=(1,2)) - mu1_mu2
    numerator = (2 * mu1_mu2 + C1) * (2 * covar + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (var1 + var2 + C2)
    ssim = numerator / denominator
    return jnp.mean(1.0 - ssim)

@partial(jax.jit, static_argnames=('num_moments',))
def calculate_moments(patches, num_moments=4):
    flat_patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
    mean = jnp.mean(flat_patches, axis=1)
    var = jnp.var(flat_patches, axis=1)
    if num_moments <= 2: return jnp.concatenate([mean, var], axis=-1)
    centered = flat_patches - mean[:, None, :]
    std_dev = jnp.sqrt(var + 1e-6)
    skew = jnp.mean((centered / std_dev[:, None, :])**3, axis=1)
    if num_moments <= 3: return jnp.concatenate([mean, var, skew], axis=-1)
    kurt = jnp.mean((centered / std_dev[:, None, :])**4, axis=1)
    return jnp.concatenate([mean, var, skew, kurt], axis=-1)

@jit
def fft_magnitude_log(patches):
    def fft_on_slice(patch_2d): return jnp.log(jnp.abs(jnp.fft.fft2(patch_2d)) + 1e-6)
    return jax.vmap(jax.vmap(fft_on_slice, in_axes=-1, out_axes=-1))(patches)

@partial(jax.vmap, in_axes=(0, 0, 0, None, None), out_axes=0)
def _extract_patches_vmapped(image, x_coords, y_coords, patch_size, c):
    def get_patch(x, y):
         return jax.lax.dynamic_slice(image, (y, x, 0), (patch_size, patch_size, c))
    return jax.vmap(get_patch)(x_coords, y_coords)

class JAXMultiMetricPerceptualLoss:
    def __init__(self, num_patches=64, patch_size=32):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self._calculate_losses_jit = partial(jax.jit, static_argnames=('batch_size',))(self._calculate_losses)

    def _calculate_losses(self, img1, img2, key, batch_size: int) -> Dict[str, jnp.ndarray]:
        _, h, w, c = img1.shape
        x_coords = jax.random.randint(key, (batch_size, self.num_patches), 0, w - self.patch_size)
        y_coords = jax.random.randint(key, (batch_size, self.num_patches), 0, h - self.patch_size)
        patches1 = _extract_patches_vmapped(img1, x_coords, y_coords, self.patch_size, c)
        patches2 = _extract_patches_vmapped(img2, x_coords, y_coords, self.patch_size, c)
        patches1 = patches1.reshape(-1, self.patch_size, self.patch_size, c)
        patches2 = patches2.reshape(-1, self.patch_size, self.patch_size, c)
        
        scales = [1.0, 0.5]
        all_losses = {'moment': [], 'fft': [], 'autocorr': [], 'edge': [], 'color_cov': [], 'ssim': []}

        for scale in scales:
            new_size = int(self.patch_size * scale)
            if new_size < 16: continue
            p1 = jax.image.resize(patches1, (patches1.shape[0], new_size, new_size, c), 'bilinear')
            p2 = jax.image.resize(patches2, (patches2.shape[0], new_size, new_size, c), 'bilinear')
            all_losses['moment'].append(jnp.mean(jnp.abs(calculate_moments(p1) - calculate_moments(p2))))
            all_losses['fft'].append(jnp.mean(jnp.abs(fft_magnitude_log(jnp.mean(p1, axis=-1, keepdims=True)) - fft_magnitude_log(jnp.mean(p2, axis=-1, keepdims=True)))))
            all_losses['autocorr'].append(jnp.mean(jnp.abs(calculate_autocorrelation_features(p1) - calculate_autocorrelation_features(p2))))
            all_losses['edge'].append(calculate_edge_loss(p1, p2))
            all_losses['color_cov'].append(calculate_color_covariance_loss(p1, p2))
            all_losses['ssim'].append(calculate_ssim_loss(p1, p2))

        return {k: jnp.mean(jnp.array(v)) for k, v in all_losses.items() if v}

    def __call__(self, img1, img2, key):
        if img1.ndim != 4 or img2.ndim != 4:
            raise ValueError(f"Inputs must be 4D tensors, got {img1.shape} and {img2.shape}")
        return self._calculate_losses_jit(img1, img2, key, batch_size=img1.shape[0])

# =================================================================================================
# 3. DATA PREPARATION
# =================================================================================================

def create_raw_dataset(data_dir: str):
    console = Console()
    data_p = Path(data_dir)
    record_file = data_p / "data_512x512.tfrecord"
    info_file = data_p / "dataset_info.pkl"

    if not record_file.exists() or not info_file.exists():
        sys.exit(f"[FATAL] TFRecord file ({record_file}) or info file ({info_file}) not found.")

    console.print("--- Loading dataset info for robust pairing... ---", style="yellow")
    with open(info_file, 'rb') as f: info = pickle.load(f)
    image_paths_from_source = info.get('image_paths')
    if not image_paths_from_source:
        sys.exit("[FATAL] 'image_paths' key not found in dataset_info.pkl. Please re-run the `prepare-data` command from your Phase 1/2 script after updating it to save the path list.")

    image_ds = tf.data.TFRecordDataset(str(record_file))
    
    # We only need images, but we parse both to keep the function consistent.
    # The text path is a dummy here.
    dummy_text_ds = tf.data.Dataset.from_tensor_slices([""] * len(image_paths_from_source))
    paired_ds = tf.data.Dataset.zip((image_ds, dummy_text_ds))
    
    def _parse(img_proto, _):
        features = {'image': tf.io.FixedLenFeature([], tf.string)}
        parsed_features = tf.io.parse_single_example(img_proto, features)
        img = tf.io.decode_jpeg(parsed_features['image'], 3)
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        return img, "" # Return dummy text

    return paired_ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

def prepare_tokenizer_data(args):
    """Pre-computes the path_params latents for the entire dataset to be used in tokenizer training."""
    console = Console()
    console.print("--- 🧠 STEP 0: Preparing Latent Dataset for Tokenizer ---", style="bold yellow")
    output_path = Path(args.data_dir) / f"tokenizer_latents_{args.basename}.pkl"
    if output_path.exists():
        console.print(f"✅ Tokenizer latent data already exists at [green]{output_path}[/green]. Skipping preparation.")
        return

    p1_path = Path(f"{args.basename}_{args.d_model}d_512.pkl")
    if not p1_path.exists(): sys.exit(f"[FATAL] Phase 1 model not found: {p1_path}")
    with open(p1_path, 'rb') as f: p1_checkpoint = pickle.load(f)
    p1_params = p1_checkpoint['params']

    p1_encoder = PathModulator(args.latent_grid_size, 512, jnp.float32)
    p1_encoder_fn = jit(lambda i: p1_encoder.apply({'params': p1_params['modulator']}, i))

    raw_ds = create_raw_dataset(args.data_dir).map(lambda img, txt: img).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    all_latents = []
    
    console.print("Processing raw images into latents for the tokenizer...")
    num_samples = create_raw_dataset(args.data_dir).reduce(0, lambda x, _: x + 1).numpy()
    
    for img_batch in tqdm(raw_ds.as_numpy_iterator(), desc="Preprocessing", total=math.ceil(num_samples / args.batch_size)):
        latents_np = np.asarray(p1_encoder_fn(img_batch))
        all_latents.append(latents_np)

    final_latents = np.concatenate(all_latents, axis=0)
    
    console.print(f"Processed {len(final_latents)} images. Saving latents to [green]{output_path}[/green]...")
    with open(output_path, 'wb') as f:
        pickle.dump({'latents': final_latents}, f)
    console.print("✅ Preparation complete.")

# =================================================================================================
# 4. TOKENIZER TRAINER
# =================================================================================================

class AdvancedTrainer:
    """Base class for training with advanced toolkit features."""
    def __init__(self, args):
        self.args = args
        self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.loss_history = deque(maxlen=200)
        
        # Safely check for 'use_q_controller' attribute, defaulting to True for tokenizer
        if getattr(self.args, 'use_q_controller', True):
            is_finetune = getattr(self.args, 'finetune', False)
            q_config = Q_CONTROLLER_CONFIG_FINETUNE if is_finetune else Q_CONTROLLER_CONFIG_NORMAL
            self.q_controller = JaxHakmemQController(initial_value=self.args.lr, config=q_config)
        else:
            self.q_controller = None

    def _get_gpu_stats(self):
        try:
            h=pynvml.nvmlDeviceGetHandleByIndex(0)
            m=pynvml.nvmlDeviceGetMemoryInfo(h)
            u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except Exception: return "N/A", "N/A"
    
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"
        hist=np.array(list(data))
        if len(hist)<2: return " "*w
        hist=hist[-w:]
        min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s))
        indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

class PIDLambdaController:
    """Dynamically balances loss weights using a PID controller for each component."""
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]]):
        self.targets = targets
        self.base_weights = base_weights
        self.gains = gains
        self.integral_error = {k: 0.0 for k in targets.keys()}
        self.last_error = {k: 0.0 for k in targets.keys()}
        self.derivative = {k: 0.0 for k in targets.keys()}

    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        final_lambdas = self.base_weights.copy()
        for name, target in self.targets.items():
            metric_key = next((k for k in last_metrics if k.endswith(name)), None)
            if metric_key is None: continue

            kp, ki, kd = self.gains[name]
            current_loss = last_metrics.get(metric_key, target)
            error = current_loss - target
            
            self.integral_error[name] += error
            self.integral_error[name] = np.clip(self.integral_error[name], -5.0, 5.0)
            self.derivative[name] = error - self.last_error[name]
            
            adjustment = (kp * error) + (ki * self.integral_error[name]) + (kd * self.derivative[name])
            multiplier = np.exp(adjustment)
            
            calculated_lambda = self.base_weights[name] * multiplier
            self.last_error[name] = error
            final_lambdas[name] = np.clip(calculated_lambda, 0.2, 5.0)
        return final_lambdas

    def state_dict(self):
        return {'integral_error': self.integral_error, 'last_error': self.last_error}
    
    def load_state_dict(self, state):
        self.integral_error = state.get('integral_error', self.integral_error)
        self.last_error = state.get('last_error', self.last_error)

class TokenizerTrainer(AdvancedTrainer):
    """The complete TokenizerTrainer with PID control and interactive GUI."""
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.generator_model = LatentTokenizerVQGAN(args.num_codes, args.code_dim, args.latent_grid_size, self.dtype)
        self.discriminator_model = PatchDiscriminator(dtype=self.dtype)
        self.perceptual_loss_fn = JAXMultiMetricPerceptualLoss()

        # GAN Balancer & Lockout State
        self.d_loss_ema = 0.5
        self.d_loss_ema_alpha = 0.05
        self.d_loss_target_min = 0.3
        self.d_loss_target_max = 0.6
        self.g_lr_multiplier = 1.0
        self.d_lr_multiplier = 1.0
        self.d_lockout_steps = 0
        self.d_lockout_threshold = 0.009
        self.d_lockout_duration = 5

        pid_gains = {
            'l1': (1.5, 0.01, 2.0), 'vq': (1.8, 0.01, 2.5),
            'moment': (1.0, 0.01, 1.0), 'fft': (1.2, 0.01, 1.5), 'autocorr': (2.5, 0.02, 3.5),
            'edge': (2.8, 0.02, 4.0), 'color_cov': (2.0, 0.02, 2.5), 'ssim': (3.0, 0.03, 3.0)
        }
        self.lambda_controller = PIDLambdaController(
            targets={'l1': 0.05, 'vq': 0.1, 'moment': 0.2, 'fft': 0.5, 'autocorr': 0.1, 'edge': 0.1, 'color_cov': 0.05, 'ssim': 0.02},
            base_weights={'l1': 2.0, 'vq': 1.5, 'adv': 0.5, 'moment': 0.5, 'fft': 0.5, 'autocorr': 2.0, 'edge': 2.5, 'color_cov': 2.5, 'ssim': 3.0},
            gains=pid_gains
        )
        
        self.ui_lock = threading.Lock()
        self.param_count = 0
        
        self.hist_len = 400
        self.g_loss_hist = deque(maxlen=self.hist_len)
        self.d_loss_hist = deque(maxlen=self.hist_len)
        self.l1_hist = deque(maxlen=self.hist_len)
        self.ssim_hist = deque(maxlen=self.hist_len)
        self.vq_hist = deque(maxlen=self.hist_len)
        self.varent_hist = deque(maxlen=self.hist_len)
        self.last_metrics_for_ui = {}
        self.current_lambdas_for_ui = {}
        self.p1_params = None
        self.p1_decoder_model = None
        self.preview_latents = None
        self.current_preview_np = None
        self.current_recon_np = None
        self.rendered_original_preview = None
        self.rendered_recon_preview = None

    def get_geometric_boosts(self, path_params_batch: jnp.ndarray):
        avg_radius = jnp.mean(path_params_batch[..., 2])
        complexity_factor = avg_radius / (jnp.pi / 2.0)
        return jnp.exp(complexity_factor)

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            root_layout = Layout(name="root")
            root_layout.split_column(Layout(name="header", size=3), Layout(name="main_content", ratio=1), Layout(self.progress, name="footer", size=3))
            root_layout["main_content"].split_row(Layout(name="left_column", ratio=2, minimum_size=50), Layout(name="right_column", ratio=3))
            
            left_stack = Layout(name="left_stack")
            right_stack = Layout(name="right_stack")
            left_stack.split_column(Layout(name="stats", minimum_size=6), Layout(name="gan_balancer", minimum_size=6), Layout(name="q_controller", minimum_size=5), Layout(name="pid_controller", ratio=1, minimum_size=12))
            right_stack.split_column(Layout(name="live_trends", minimum_size=15), Layout(name="live_preview", ratio=1, minimum_size=10))

            precision_str = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"🧬 [bold]Tokenizer Trainer[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            root_layout["header"].update(Panel(Align.center(header_text), style="bold blue", title="[dim]wubumind.ai[/dim]", title_align="right"))

            g_loss, d_loss = self.last_metrics_for_ui.get('g_loss', 0), self.last_metrics_for_ui.get('d_loss', 0)
            l1, ssim, vq, edge = (self.last_metrics_for_ui.get(k,0) for k in ['l1','ssim','vq','edge'])
            stats_tbl = Table.grid(expand=True); stats_tbl.add_column(style="dim",width=14); stats_tbl.add_column()
            stats_tbl.add_row("G / D Loss", f"[cyan]{g_loss:.3f}[/] / [magenta]{d_loss:.3f}[/]"); stats_tbl.add_row("L1 / VQ", f"[green]{l1:.2e}[/] / [green]{vq:.2e}[/]"); stats_tbl.add_row("SSIM / Edge", f"[yellow]{ssim:.2e}[/] / [yellow]{edge:.2e}[/]")
            mem, util = self._get_gpu_stats(); stats_tbl.add_row("GPU Mem / Util", f"[yellow]{mem}[/] / [yellow]{util}[/]")
            left_stack["stats"].update(Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue"))
            
            gan_balancer_tbl = Table.grid(expand=True); gan_balancer_tbl.add_column(style="dim", width=12); gan_balancer_tbl.add_column(style="yellow")
            gan_balancer_tbl.add_row("D Loss EMA", f"{self.d_loss_ema:.3f}"); gan_balancer_tbl.add_row("G LR Mult", f"{self.g_lr_multiplier:.2f}x"); gan_balancer_tbl.add_row("D LR Mult", f"{self.d_lr_multiplier:.2f}x")
            if self.d_lockout_steps > 0: gan_balancer_tbl.add_row("Status", Text(f"LOCKED ({self.d_lockout_steps})", style="bold red"))
            left_stack["gan_balancer"].update(Panel(gan_balancer_tbl, title="[bold]⚖️ GAN Balancer[/]", border_style="yellow"))
            
            q_panel_content = Align.center("[dim]Q-Ctrl Off[/dim]")
            if self.q_controller:
                q_tbl = Table.grid(expand=True); q_tbl.add_column(style="dim",width=12); q_tbl.add_column()
                status_short = self.q_controller.status.split(' ')[0]
                status_emoji, color = ("😎","green") if "IMPROVING" in status_short else (("🤔","yellow") if "STAGNATED" in status_short else (("😠","red") if "REGRESSING" in status_short else (("🐣","blue") if "WARMUP" in status_short else ("🤖","dim"))))
                q_tbl.add_row("Base LR", f"[{color}]{self.q_controller.current_lr:.2e}[/] {status_emoji}"); q_tbl.add_row("Reward", f"{self.q_controller.last_reward:+.2e}"); q_tbl.add_row("Exploration", f"{self.q_controller.exploration_rate_q:.2e}")
                q_panel_content = q_tbl
            left_stack["q_controller"].update(Panel(q_panel_content, title="[bold]🤖 Q-Controller[/]", border_style="green"))
            
            pid_internals_tbl = Table("Loss", "Error", "Integral", "Deriv", "Mult", "Final λ", title_style="bold yellow")
            for name in self.lambda_controller.targets:
                error = self.lambda_controller.last_error.get(name, 0.0); integral = self.lambda_controller.integral_error.get(name, 0.0); derivative = self.lambda_controller.derivative.get(name, 0.0)
                multiplier = np.exp((self.lambda_controller.gains[name][0] * error) + (self.lambda_controller.gains[name][1] * integral) + (self.lambda_controller.gains[name][2] * derivative))
                pid_internals_tbl.add_row(name.capitalize(), f"{error:+.2e}", f"{integral:+.2e}", f"{derivative:+.2e}", f"{multiplier:.2e}", f"{self.current_lambdas_for_ui.get(name, 0):.2e}")
            left_stack["pid_controller"].update(Panel(pid_internals_tbl, title="[bold]🧠 PID Controller Internals[/]", border_style="yellow"))

            spark_w = 40
            graphs = [Panel(Align.center(f"{self._get_sparkline(hist, spark_w)}"), title=name, height=3, border_style=color) for name, hist, color in [("G Loss", self.g_loss_hist, "cyan"), ("D Loss", self.d_loss_hist, "magenta"), ("SSIM", self.ssim_hist, "green"), ("VQ", self.vq_hist, "yellow")]]
            right_stack["live_trends"].update(Panel(Group(*graphs), title="[bold]📈 Live Trends[/]"))
            
            if self.rendered_recon_preview is None and self.current_recon_np is not None:
                 if Pixels: self.rendered_recon_preview = Align.center(Pixels.from_image(Image.fromarray(self.current_recon_np)))

            preview_content = Align.center("...Waiting for first validation...")
            if self.rendered_original_preview and self.rendered_recon_preview:
                preview_table = Table.grid(expand=True); preview_table.add_column(ratio=1); preview_table.add_column(ratio=1)
                preview_table.add_row(Text("Original", justify="center"), Text("Reconstruction", justify="center")); preview_table.add_row(self.rendered_original_preview, self.rendered_recon_preview)
                preview_content = preview_table
            
            right_stack["live_preview"].update(Panel(preview_content, title="[bold]🖼️ Live Validation Preview[/]", border_style="green"))

            root_layout["left_column"].update(left_stack); root_layout["right_column"].update(right_stack)
            return root_layout

    def train(self):
        console = Console()
        background_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Tokenizer_BG_Worker')
        active_preview_future = None
        
        console.print("--- Loading Phase 1 AE for visual validation... ---", style="yellow")
        try:
            p1_path = next(Path('.').glob(f"{self.args.basename}_{self.args.d_model}d_512.pkl"))
        except StopIteration:
            sys.exit(f"[FATAL] Could not find Phase 1 AE model needed for previews.")
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']
        self.p1_decoder_model = TopologicalCoordinateGenerator(self.args.d_model, self.args.latent_grid_size, 512, dtype=jnp.float32)
        
        data_path = Path(self.args.data_dir) / f"tokenizer_latents_{self.args.basename}.pkl"
        with open(data_path, 'rb') as f: latents_data = pickle.load(f)['latents']
        np.random.seed(self.args.seed); shuffled_indices = np.random.permutation(len(latents_data))
        val_split_idx = int(len(latents_data) * 0.01)
        train_data, val_data = latents_data[shuffled_indices[val_split_idx:]], latents_data[shuffled_indices[:val_split_idx]]
        self.preview_latents = jnp.asarray(val_data[:4], dtype=jnp.float32)
        
        @partial(jit, static_argnames=('resolution','patch_size'))
        def render_image(params, path_params, resolution=128, patch_size=64):
             coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
             coord_chunks = jnp.array_split(coords, (resolution**2)//(patch_size**2))
             pixels_list = [self.p1_decoder_model.apply({'params': params}, path_params, c, method=self.p1_decoder_model.decode) for c in coord_chunks]
             return jnp.concatenate(pixels_list, axis=1).reshape(path_params.shape[0], resolution, resolution, 3)
        
        original_preview_batch = render_image(self.p1_params, self.preview_latents)
        self.current_preview_np = np.array(((original_preview_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        if Pixels: self.rendered_original_preview = Align.center(Pixels.from_image(Image.fromarray(self.current_preview_np)))
        
        console.print(f"✅ Data split: {len(train_data)} training samples, {len(val_data)} validation samples.")
        steps_per_epoch = math.ceil(len(train_data) / self.args.batch_size)
        
        key = jax.random.PRNGKey(self.args.seed)
        g_key, d_key, self.train_key = jax.random.split(key, 3)
        gen_optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=self.args.lr, b1=0.5, b2=0.9)
        disc_optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=self.args.lr * 0.8, b1=0.5, b2=0.9)
        dummy_input = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), self.dtype)
        gen_params = self.generator_model.init(g_key, dummy_input)['params']; disc_params = self.discriminator_model.init(d_key, dummy_input)['params']
        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, gen_params, 0)
        gen_state = TrainState.create(apply_fn=self.generator_model.apply, params=gen_params, tx=gen_optimizer)
        disc_state = TrainState.create(apply_fn=self.discriminator_model.apply, params=disc_params, tx=disc_optimizer)
        states = GANTrainStates(generator=gen_state, discriminator=disc_state)
        ckpt_path = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_final.pkl"); ckpt_path_best = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_best.pkl")
        best_val_loss, start_epoch, global_step = float('inf'), 0, 0
        
        last_metrics = {'g_loss': 0.0, 'd_loss': 0.5, 'l1': 0.05, 'vq': 0.1, 'moment': 0.2, 'fft': 0.5, 'autocorr': 0.1, 'edge': 0.1, 'color_cov': 0.05, 'ssim': 0.02, 'varentropy': 0.0}

        if ckpt_path.exists():
            console.print(f"--- Resuming training state from: [green]{ckpt_path}[/green] ---")
            with open(ckpt_path, 'rb') as f: ckpt = pickle.load(f)
            gen_opt_state, disc_opt_state = ckpt['gen_opt_state'], ckpt['disc_opt_state']
            start_epoch = ckpt.get('epoch', 0)
            last_metrics.update(ckpt.get('last_metrics', {}))
            global_step = ckpt.get('global_step', start_epoch * steps_per_epoch)
            if self.q_controller and 'q_controller_state' in ckpt: self.q_controller.load_state_dict(ckpt['q_controller_state']); console.print(f"🤖 Q-Controller state restored.")
            if 'pid_controller_state' in ckpt: self.lambda_controller.load_state_dict(ckpt['pid_controller_state']); console.print(f"🧠 PID Controller state restored.")
            if ckpt_path_best.exists():
                with open(ckpt_path_best, 'rb') as f_best: best_ckpt = pickle.load(f_best); gen_params = best_ckpt['params']; best_val_loss = best_ckpt.get('val_loss', float('inf'))
            else: gen_params = ckpt['gen_params']
            states = GANTrainStates(generator=states.generator.replace(params=gen_params, opt_state=gen_opt_state), discriminator=states.discriminator.replace(params=ckpt['disc_params'], opt_state=disc_opt_state))
            console.print(f"✅ Resuming session from epoch {start_epoch + 1}, step {global_step}. Best val loss: {best_val_loss:.4f}")
        
        @partial(jit, static_argnames=('gen_apply_fn', 'disc_apply_fn', 'd_is_locked_out'))
        def train_step(states, batch, key, lambdas, gen_apply_fn, disc_apply_fn, d_is_locked_out: bool):
            (lambda_l1, lambda_vq, lambda_adv, lambda_stink, lambda_moment, lambda_fft, lambda_autocorr, lambda_edge, lambda_color_cov, lambda_ssim) = lambdas

            def generator_loss_fn(p):
                gen_output = gen_apply_fn({'params': p}, batch)
                recon = gen_output['reconstructed_path_params']
                l1_loss = jnp.mean(jnp.abs(batch - recon))
                vq_loss = gen_output['vq_loss']
                adv_loss = jnp.mean((disc_apply_fn({'params': states.discriminator.params}, recon) - 1)**2)
                perceptual_losses = self.perceptual_loss_fn(batch, recon, key)
                _, varent = ent_varent(gen_output['pre_quant_latents'].reshape(-1, gen_output['pre_quant_latents'].shape[-1]))
                varentropy_loss = jnp.mean(varent)
                total_loss = (lambda_l1 * l1_loss) + (lambda_vq * vq_loss) + (lambda_adv * adv_loss) + (lambda_stink * varentropy_loss) + (lambda_moment * perceptual_losses['moment']) + (lambda_fft * perceptual_losses['fft']) + (lambda_autocorr * perceptual_losses['autocorr']) + (lambda_edge * perceptual_losses['edge']) + (lambda_color_cov * perceptual_losses['color_cov']) + (lambda_ssim * perceptual_losses['ssim'])
                all_metrics = {'l1': l1_loss, 'vq': vq_loss, 'adv': adv_loss, 'varentropy': varentropy_loss}
                all_metrics.update(perceptual_losses)
                return total_loss, all_metrics

            (g_loss_total, metrics), g_grads = jax.value_and_grad(generator_loss_fn, has_aux=True)(states.generator.params)
            new_gen_state = states.generator.apply_gradients(grads=g_grads)
            
            def discriminator_loss_fn(disc_params):
                def compute_d_loss():
                    recon = gen_state.apply_fn({'params': new_gen_state.params}, batch)['reconstructed_path_params']
                    loss_real = jnp.mean((disc_apply_fn({'params': disc_params}, batch) - 1)**2)
                    loss_fake = jnp.mean(disc_apply_fn({'params': disc_params}, jax.lax.stop_gradient(recon))**2)
                    return ((loss_real + loss_fake) * 0.5).astype(jnp.float32)
                return jax.lax.cond(d_is_locked_out, lambda: 0.0, compute_d_loss)
            
            d_loss, d_grads = jax.value_and_grad(discriminator_loss_fn)(states.discriminator.params)
            new_disc_state = states.discriminator.apply_gradients(grads=d_grads)
            metrics['g_loss'] = g_loss_total
            metrics['d_loss'] = d_loss
            return GANTrainStates(generator=new_gen_state, discriminator=new_disc_state), metrics
            
        @partial(jit, static_argnames=('apply_fn',))
        def eval_step(gen_params, apply_fn, batch, key):
            out = apply_fn({'params': gen_params}, batch)
            l1_loss = jnp.mean(jnp.abs(out['reconstructed_path_params']-batch))
            perceptual_losses = self.perceptual_loss_fn(out['reconstructed_path_params'], batch, key)
            return (l1_loss + perceptual_losses['ssim'] + perceptual_losses['edge']).astype(jnp.float32)

        @partial(jit, static_argnames=('gen_apply_fn',))
        def generate_preview(gen_params, gen_apply_fn, p1_params, preview_latents_batch):
            recon_path_params = gen_apply_fn({'params': gen_params}, preview_latents_batch)['reconstructed_path_params']
            return render_image(p1_params, recon_path_params)
        
        def _update_preview_task(gen_params, gen_apply_fn, p1_params, preview_latents_batch):
            recon_batch = generate_preview(gen_params, gen_apply_fn, p1_params, preview_latents_batch)
            recon_batch.block_until_ready()
            recon_np = np.array(((recon_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
            with self.ui_lock:
                self.current_recon_np = recon_np
                if Pixels: self.rendered_recon_preview = Align.center(Pixels.from_image(Image.fromarray(self.current_recon_np)))

        console.print("[bold yellow]🚀 JIT compiling GAN training step...[/bold yellow]")
        dummy_batch = jnp.asarray(train_data[:self.args.batch_size], dtype=self.dtype)
        compile_key, self.train_key = jax.random.split(self.train_key)
        dummy_lambda_dict = self.lambda_controller(last_metrics)
        dummy_lambdas = (dummy_lambda_dict['l1'], dummy_lambda_dict['vq'], dummy_lambda_dict['adv'], 0.2, dummy_lambda_dict['moment'], dummy_lambda_dict['fft'], dummy_lambda_dict['autocorr'], dummy_lambda_dict['edge'], dummy_lambda_dict['color_cov'], dummy_lambda_dict['ssim'])
        states, _ = train_step(states, dummy_batch, compile_key, dummy_lambdas, self.generator_model.apply, self.discriminator_model.apply, d_is_locked_out=False)
        if len(val_data) > 0: eval_step(states.generator.params, self.generator_model.apply, jnp.asarray(val_data[:self.args.batch_size], dtype=self.dtype), jax.random.split(self.train_key)[0])
        generate_preview(states.generator.params, self.generator_model.apply, self.p1_params, self.preview_latents[:1])
        console.print("[green]✅ Compilation complete.[/green]")

        self.progress = Progress(TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[val_loss]:.2e}[/]"), BarColumn(), "•", TextColumn("Step {task.fields[step]}/{task.fields[steps_per_epoch]}"), "•", TimeRemainingColumn(), TextColumn("Ctrl+C to Exit"))
        epoch_task = self.progress.add_task("epochs", total=self.args.epochs, completed=start_epoch, val_loss=best_val_loss, step=0, steps_per_epoch=steps_per_epoch)
        rng = np.random.default_rng(self.args.seed)
        train_indices_shuffler = np.arange(len(train_data))
        last_ui_update_time = 0.0
        UI_UPDATE_INTERVAL_SECS = 0.25

        try:
            with Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for epoch in range(start_epoch, self.args.epochs):
                    if self.should_shutdown: break
                    rng.shuffle(train_indices_shuffler)
                    for step_in_epoch in range(steps_per_epoch):
                        if self.should_shutdown: break
                        
                        start_idx = step_in_epoch * self.args.batch_size
                        end_idx = start_idx + self.args.batch_size
                        if start_idx >= len(train_indices_shuffler): continue
                        batch_indices = train_indices_shuffler[start_idx:end_idx]
                        train_batch = jnp.asarray(train_data[batch_indices], dtype=self.dtype)
                        
                        perc_boost = self.get_geometric_boosts(train_batch)
                        lambda_dict = self.lambda_controller(last_metrics)
                        for k in self.lambda_controller.targets.keys():
                            if k not in ['l1', 'vq', 'adv']: lambda_dict[k] *= perc_boost
                        self.current_lambdas_for_ui = lambda_dict
                        current_lambdas = (lambda_dict['l1'], lambda_dict['vq'], lambda_dict['adv'], 0.2, lambda_dict['moment'], lambda_dict['fft'], lambda_dict['autocorr'], lambda_dict['edge'], lambda_dict['color_cov'], lambda_dict['ssim'])
                        
                        base_lr = self.args.lr
                        if self.q_controller: base_lr = self.q_controller.choose_action()

                        self.d_loss_ema = (1 - self.d_loss_ema_alpha) * self.d_loss_ema + self.d_loss_ema_alpha * last_metrics.get('d_loss', 0.5)
                        d_is_locked_out = False
                        if self.d_lockout_steps > 0: self.d_lockout_steps -= 1; d_is_locked_out = True
                        elif self.d_loss_ema < self.d_lockout_threshold: self.d_lockout_steps = self.d_lockout_duration; d_is_locked_out = True
                        
                        self.g_lr_multiplier = 1.0; self.d_lr_multiplier = 1.0
                        if not d_is_locked_out:
                            if self.d_loss_ema < self.d_loss_target_min: self.d_lr_multiplier = 0.5; self.g_lr_multiplier = 1.5
                            elif self.d_loss_ema > self.d_loss_target_max: self.d_lr_multiplier = 1.5; self.g_lr_multiplier = 0.5
                        
                        states.generator.opt_state.hyperparams['learning_rate'] = base_lr * self.g_lr_multiplier
                        states.discriminator.opt_state.hyperparams['learning_rate'] = (base_lr * 0.8) * self.d_lr_multiplier
                        
                        step_key, self.train_key = jax.random.split(self.train_key)
                        states, metrics = train_step(states, train_batch, step_key, current_lambdas, self.generator_model.apply, self.discriminator_model.apply, d_is_locked_out=d_is_locked_out)
                        metrics_cpu = {k: v.item() for k, v in metrics.items()}
                        last_metrics = metrics_cpu
                        self.last_metrics_for_ui = metrics_cpu

                        with self.ui_lock:
                            self.g_loss_hist.append(metrics_cpu['g_loss']); self.d_loss_hist.append(metrics_cpu['d_loss']); self.l1_hist.append(metrics_cpu['l1']); self.ssim_hist.append(metrics_cpu.get('ssim',0.0)); self.vq_hist.append(metrics_cpu['vq']); self.varent_hist.append(metrics_cpu['varentropy'])
                        
                        if self.q_controller: self.q_controller.update_q_value(metrics_cpu['g_loss'])
                        
                        global_step += 1
                        self.progress.update(epoch_task, step=step_in_epoch + 1)
                        
                        if global_step % self.args.eval_every == 0:
                            if not (active_preview_future and not active_preview_future.done()):
                                host_gen_params = jax.device_get(states.generator.params)
                                active_preview_future = background_executor.submit(_update_preview_task, host_gen_params, self.generator_model.apply, self.p1_params, self.preview_latents)
                            
                            if len(val_data) > 0:
                                eval_key, self.train_key = jax.random.split(self.train_key)
                                eval_keys = jax.random.split(eval_key, (len(val_data) // self.args.batch_size) + 1)
                                val_losses = [eval_step(states.generator.params, self.generator_model.apply, jnp.asarray(val_data[i:i+self.args.batch_size], dtype=self.dtype), eval_keys[j]) for j, i in enumerate(range(0, len(val_data), self.args.batch_size))]
                                val_loss = np.mean([v.item() for v in val_losses]) if val_losses else float('inf')
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    console.print(f"\n[bold magenta]🏆 New best val loss: {best_val_loss:.2e} @ step {global_step}. Saving...[/bold magenta]")
                                    with open(ckpt_path_best, 'wb') as f: pickle.dump({'params': states.generator.params, 'val_loss': best_val_loss, 'epoch': epoch, 'global_step': global_step}, f)
                                self.progress.update(epoch_task, val_loss=best_val_loss)

                        current_time = time.time()
                        if current_time - last_ui_update_time > UI_UPDATE_INTERVAL_SECS:
                            live.update(self._generate_layout()); last_ui_update_time = current_time

                    self.progress.update(epoch_task, advance=1)
                    
                    host_state_to_save = jax.device_get(states)
                    q_state_to_save = self.q_controller.state_dict() if self.q_controller else None
                    pid_state_to_save = self.lambda_controller.state_dict()
                    data_to_save = {'gen_params': host_state_to_save.generator.params, 'gen_opt_state': host_state_to_save.generator.opt_state, 'disc_params': host_state_to_save.discriminator.params, 'disc_opt_state': host_state_to_save.discriminator.opt_state, 'epoch': epoch, 'global_step': global_step, 'q_controller_state': q_state_to_save, 'last_metrics': last_metrics, 'pid_controller_state': pid_state_to_save}
                    with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
        finally:
            console.print(f"\n[yellow]--- Training loop exited. Waiting for background tasks to finish... ---[/yellow]")
            background_executor.shutdown(wait=True)
            final_epoch_count = self.args.epochs if 'epoch' not in locals() else epoch
            console.print(f"\n[yellow]--- Training session finished at epoch {final_epoch_count+1}. Saving final state... ---[/yellow]")
            
            if 'states' in locals():
                host_state_final = jax.device_get(states)
                q_state_to_save = self.q_controller.state_dict() if self.q_controller else None
                pid_state_to_save = self.lambda_controller.state_dict()
                final_data = {'gen_params': host_state_final.generator.params, 'gen_opt_state': host_state_final.generator.opt_state, 'disc_params': host_state_final.discriminator.params, 'disc_opt_state': host_state_final.discriminator.opt_state, 'epoch': final_epoch_count, 'global_step': global_step, 'q_controller_state': q_state_to_save, 'last_metrics': last_metrics, 'pid_controller_state': pid_state_to_save}
                with open(ckpt_path, 'wb') as f: pickle.dump(final_data, f)
                console.print(f"✅ Final resume-state saved to [green]{ckpt_path}[/green]")

            config = {'num_codes': self.args.num_codes, 'code_dim': self.args.code_dim, 'latent_grid_size': self.args.latent_grid_size}
            config_path = Path(str(ckpt_path).replace("_final.pkl", "_config.pkl"))
            with open(config_path, 'wb') as f: pickle.dump(config, f)
            console.print(f"✅ Config saved to [green]{config_path}[/green]")
            if ckpt_path_best.exists(): console.print(f"👑 Best model (by validation) remains at [bold magenta]{ckpt_path_best}[/bold magenta]")

# =================================================================================================
# 5. MAIN EXECUTION BLOCK
# =================================================================================================

def run_tokenizer_check(args):
    """Loads a trained tokenizer and reconstructs a single image to test its quality."""
    console = Console()
    console.print("--- 🔬 Running Tokenizer Sanity Check ---", style="bold yellow")

    try:
        tok_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_*_config.pkl"))
        tok_path_best = Path(str(tok_config_path).replace("_config.pkl", "_best.pkl"))
        if not tok_path_best.exists(): tok_path_best = Path(str(tok_config_path).replace("_config.pkl", "_final.pkl"))
    except StopIteration:
        sys.exit(f"[FATAL] Could not find tokenizer config/model for basename '{args.basename}'.")
    
    with open(tok_config_path, 'rb') as f: tok_config = pickle.load(f)
    console.print(f"-> Loading tokenizer from [green]{tok_path_best}[/green]")
    with open(tok_path_best, 'rb') as f: tok_ckpt = pickle.load(f); tok_params = tok_ckpt.get('params', tok_ckpt.get('gen_params'))

    dtype = jnp.float32
    tokenizer = LatentTokenizerVQGAN(**tok_config, dtype=dtype)
    
    @jit
    def reconstruct(params, path_params_grid):
        return tokenizer.apply({'params': params}, path_params_grid)['reconstructed_path_params']

    console.print("-> Loading Phase 1 AE to get input latents...", style="dim")
    try:
        p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
        p1_d_model = int(p1_path.stem.split('_')[-2].replace('d',''))
    except StopIteration:
        sys.exit(f"[FATAL] Could not find Phase 1 model for basename '{args.basename}'.")
    
    with open(p1_path, 'rb') as f: p1_params = pickle.load(f)['params']
    p1_modulator = PathModulator(tok_config['latent_grid_size'], 512, dtype=dtype)
    
    @jit
    def get_path_params(params, image):
        return p1_modulator.apply({'params': params['modulator']}, image)

    console.print(f"-> Processing image: {args.image_path}")
    img = Image.open(args.image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    img_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
    img_batch = jnp.expand_dims(img_np, axis=0)
    
    path_params_grid = get_path_params(p1_params, img_batch)
    recon_path_params = reconstruct(tok_params, path_params_grid)
    
    console.print("-> Rendering final image from tokenizer's reconstruction...")
    p1_decoder_model = TopologicalCoordinateGenerator(p1_d_model, tok_config['latent_grid_size'], 512, dtype=dtype)
    
    @partial(jit, static_argnames=('resolution','patch_size'))
    def render_image(params, path_params, resolution=512, patch_size=256):
        coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
        coord_chunks = jnp.array_split(coords, (resolution**2)//(patch_size**2))
        pixels_list = [p1_decoder_model.apply({'params': params}, path_params, c, method=p1_decoder_model.decode) for c in coord_chunks]
        return jnp.concatenate(pixels_list, axis=1).reshape(path_params.shape[0], resolution, resolution, 3)

    final_image_batch = render_image(p1_params, recon_path_params)
    recon_np = np.array(((final_image_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
    
    comparison_img = Image.new('RGB', (1034, 512), (20, 20, 20))
    comparison_img.paste(img, (5, 0))
    comparison_img.paste(Image.fromarray(recon_np), (512 + 17, 0))
    
    comparison_img.save(args.output_path)
    console.print(f"✅ Sanity check complete. Comparison saved to [green]{args.output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Phase 2.5: VQ-GAN Tokenizer Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--basename', type=str, required=True, help="Base name for the model set (e.g., 'laion_ae'). Used to find/save all related files.")
    
    p_prep_tok = subparsers.add_parser("prepare-data", help="Pre-process images into latents for the tokenizer.", parents=[base_parser])
    p_prep_tok.add_argument('--data-dir', type=str, required=True)
    p_prep_tok.add_argument('--d-model', type=int, required=True)
    p_prep_tok.add_argument('--latent-grid-size', type=int, required=True)
    p_prep_tok.add_argument('--batch-size', type=int, default=128)

    p_tok = subparsers.add_parser("train", help="Train the Latent Tokenizer (VQ-GAN).", parents=[base_parser])
    p_tok.add_argument('--data-dir', type=str, required=True)
    p_tok.add_argument('--d-model', type=int, required=True)
    p_tok.add_argument('--latent-grid-size', type=int, required=True)
    p_tok.add_argument('--epochs', type=int, default=100)
    p_tok.add_argument('--batch-size', type=int, default=128)
    p_tok.add_argument('--lr', type=float, default=3e-4)
    p_tok.add_argument('--eval-every', type=int, default=1000, help="Run validation and update preview every N steps.")
    p_tok.add_argument('--num-codes', type=int, default=3072)
    p_tok.add_argument('--code-dim', type=int, default=256)
    p_tok.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 precision for training.")
    p_tok.add_argument('--seed', type=int, default=42)
    
    p_check_tok = subparsers.add_parser("check", help="Reconstruct an image to check tokenizer quality.", parents=[base_parser])
    p_check_tok.add_argument('--image-path', type=str, required=True)
    p_check_tok.add_argument('--output-path', type=str, default="tokenizer_recon_check.png")
    
    args = parser.parse_args()
    
    if args.command == "prepare-data": prepare_tokenizer_data(args)
    elif args.command == "train": TokenizerTrainer(args).train()
    elif args.command == "check": run_tokenizer_check(args)
    else: parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console = Console()
        console.print("\n[bold red]FATAL ERROR ENCOUNTERED[/bold red]")
        console.print_exception(show_locals=False)
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write("A fatal error occurred. Full traceback:\n")
            f.write(traceback.format_exc())
        console.print("\n[yellow]Full traceback has been written to [bold]crash_log.txt[/bold][/yellow]")