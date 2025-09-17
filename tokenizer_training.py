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
import threading
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

# --- [FIX] Legacy class definitions for loading Phase 1/2 checkpoints ---
class CustomTrainState(train_state.TrainState):
    """A custom train state that allows passing extra arguments to the optimizer for Sentinel."""
    def apply_gradients(self, *, grads: Any, **kwargs) -> "CustomTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, **kwargs)
        new_params = optax.apply_updates(self.params, updates)
        known_keys = self.__dataclass_fields__.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_keys}
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **filtered_kwargs)

class SentinelState(NamedTuple):
    """State for the Sentinel optimizer component."""
    sign_history: chex.ArrayTree
    dampened_count: Optional[jnp.ndarray] = None
    dampened_pct: Optional[jnp.ndarray] = None
# --- End of fix ---
# =================================================================================================
# 2. ADVANCED TOKENIZER TOOLKIT
# =================================================================================================

# --- [FIX] Legacy class definitions for loading Phase 1/2 checkpoints ---
class CustomTrainState(train_state.TrainState):
    def apply_gradients(self, *, grads: Any, **kwargs) -> "CustomTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, **kwargs)
        new_params = optax.apply_updates(self.params, updates)
        known_keys = self.__dataclass_fields__.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_keys}
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **filtered_kwargs)

class SentinelState(NamedTuple):
    # Replaces sign_history with a much more efficient EMA
    sign_ema: chex.ArrayTree
    dampened_count: Optional[jnp.ndarray] = None
    dampened_pct: Optional[jnp.ndarray] = None
# --- End of fix ---

# A new TrainState that will hold the Q-Controller's state
class GeneratorTrainState(CustomTrainState):
    q_state: 'QControllerState'

def sentinel(decay: float = 0.9, oscillation_threshold: float = 0.5) -> optax.GradientTransformation:
    """A faster, EMA-based Sentinel optimizer."""
    def init_fn(params):
        sign_ema = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), params)
        return SentinelState(sign_ema=sign_ema, dampened_count=jnp.array(0), dampened_pct=jnp.array(0.0))

    def update_fn(updates, state, params=None, **kwargs):
        dampening_factor = kwargs.get('dampening_factor', 1.0)
        current_sign = jax.tree_util.tree_map(jnp.sign, updates)
        new_sign_ema = jax.tree_util.tree_map(lambda ema, sign: ema * decay + sign * (1 - decay), state.sign_ema, current_sign)
        is_oscillating = jax.tree_util.tree_map(lambda ema: jnp.abs(ema) < oscillation_threshold, new_sign_ema)
        
        def apply_dampening():
            dampening_mask = jax.tree_util.tree_map(lambda is_osc: jnp.where(is_osc, dampening_factor, 1.0), is_oscillating)
            dampened_updates = jax.tree_util.tree_map(lambda u, m: u * m, updates, dampening_mask)
            num_oscillating = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.sum, is_oscillating)))
            total_params = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, params)))
            dampened_pct = num_oscillating / (total_params + 1e-8)
            return dampened_updates, num_oscillating, dampened_pct

        def skip_dampening():
            return updates, jnp.array(0), jnp.array(0.0)

        # lax.cond is JIT-friendly
        dampened_updates, num_oscillating, dampened_pct = jax.lax.cond(dampening_factor < 1.0, apply_dampening, skip_dampening)
        new_state = SentinelState(sign_ema=new_sign_ema, dampened_count=num_oscillating, dampened_pct=dampened_pct)
        return dampened_updates, new_state
    return optax.GradientTransformation(init_fn, update_fn)

# --- JAX-Native Q-Controller ---
Q_CONTROLLER_CONFIG_TOKENIZER = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-6, "lr_max": 1e-3, "metric_history_len": 5000, "loss_min": 0.1, "loss_max": 5.0, "exploration_rate_q": 0.3, "min_exploration_rate": 0.05, "exploration_decay": 0.9995, "trend_window": 420, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": 10.0, "stagnation_penalty": -2.0, "warmup_steps": 420, "warmup_lr_start": 1e-6}

from dataclasses import dataclass, replace

@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class QControllerState:
    q_table: chex.Array; metric_history: chex.Array; trend_history: chex.Array
    current_value: jnp.ndarray; exploration_rate: jnp.ndarray; step_count: jnp.ndarray
    last_action_idx: jnp.ndarray; last_reward: jnp.ndarray; status_code: jnp.ndarray
    def tree_flatten(self): return (self.q_table, self.metric_history, self.trend_history, self.current_value, self.exploration_rate, self.step_count, self.last_action_idx, self.last_reward, self.status_code), None
    @classmethod
    def tree_unflatten(cls, aux_data, children): return cls(*children)

def init_q_controller(config):
    return QControllerState(
        q_table=jnp.zeros((config["q_table_size"], config["num_lr_actions"]), dtype=jnp.float32),
        metric_history=jnp.full((config["metric_history_len"],), (config["loss_min"] + config["loss_max"]) / 2, dtype=jnp.float32),
        trend_history=jnp.zeros((config["trend_window"],), dtype=jnp.float32),
        current_value=jnp.array(config["warmup_lr_start"], dtype=jnp.float32),
        exploration_rate=jnp.array(config["exploration_rate_q"], dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32),
        last_action_idx=jnp.array(-1, dtype=jnp.int32),
        last_reward=jnp.array(0.0, dtype=jnp.float32),
        status_code=jnp.array(0, dtype=jnp.int32)
    )

@jax.jit
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey):
    config = Q_CONTROLLER_CONFIG_TOKENIZER
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config["warmup_steps"]
        new_value = config["warmup_lr_start"] * (1 - alpha) + config["lr_max"] * 0.5 * alpha
        return replace(state, current_value=new_value, step_count=state.step_count + 1, status_code=jnp.array(0)) # Status 0: WARMUP
    def regular_action():
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config["metric_history_len"] - 5, 5))
        state_idx = jnp.clip(((metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        explore_key, action_key = jax.random.split(key)
        def explore(): return jax.random.randint(action_key, (), 0, config["num_lr_actions"])
        def exploit(): return jnp.argmax(state.q_table[state_idx])
        action_idx = jax.lax.cond(jax.random.uniform(explore_key) < state.exploration_rate, explore, exploit)
        selected_factor = jnp.array(config["lr_change_factors"])[action_idx]
        new_value = jnp.clip(state.current_value * selected_factor, config["lr_min"], config["lr_max"])
        return replace(state, current_value=new_value, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config["warmup_steps"], warmup_action, regular_action)

@jax.jit
def q_controller_update(state: QControllerState, metric_value: float):
    config = Q_CONTROLLER_CONFIG_TOKENIZER
    new_metric_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value)
    new_trend_history = jnp.roll(state.trend_history, -1).at[-1].set(metric_value)
    def perform_update(st):
        x = jnp.arange(config["trend_window"], dtype=jnp.float32)
        y = new_trend_history
        A = jnp.vstack([x, jnp.ones_like(x)]).T
        slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        status_code, reward = jax.lax.cond(
            slope < -config["improve_threshold"], lambda: (jnp.array(1), abs(slope) * 1000.0), # Status 1: IMPROVING
            lambda: jax.lax.cond(
                slope > config["regress_threshold"], lambda: (jnp.array(3), -abs(slope) * 1000.0 - config["regress_penalty"]), # Status 3: REGRESSING
                lambda: (jnp.array(2), config["stagnation_penalty"]) # Status 2: STAGNATED
            )
        )
        old_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config["metric_history_len"]-5, 5))
        last_state_idx = jnp.clip(((old_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        new_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(new_metric_history, config["metric_history_len"]-5, 5))
        next_state_idx = jnp.clip(((new_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        current_q = st.q_table[last_state_idx, st.last_action_idx]
        max_next_q = jnp.max(st.q_table[next_state_idx])
        new_q = current_q + config["learning_rate_q"] * (reward + config["discount_factor_q"] * max_next_q - current_q)
        new_q_table = st.q_table.at[last_state_idx, st.last_action_idx].set(new_q)
        new_exp_rate = jnp.maximum(config["min_exploration_rate"], st.exploration_rate * config["exploration_decay"])
        return replace(st, q_table=new_q_table, exploration_rate=new_exp_rate, last_reward=reward, status_code=status_code)
    can_update = (state.step_count > config["warmup_steps"]) & (state.step_count > config["trend_window"]) & (state.last_action_idx >= 0)
    new_state = jax.lax.cond(can_update, perform_update, lambda s: s, state)
    return replace(new_state, metric_history=new_metric_history, trend_history=new_trend_history)


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
# Place this before the TokenizerTrainer class
class GeneratorTrainState(CustomTrainState):
    q_state: QControllerState
    
class AdvancedTrainer:
    """Base class for training with advanced toolkit features."""

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
        self.state = {
            'integral_error': {k: 0.0 for k in targets.keys()},
            'last_error': {k: 0.0 for k in targets.keys()},
        }

    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        # Start with a clean dictionary that will hold the new lambdas
        final_lambdas = {}
        
        # Iterate over all possible lambdas defined in base_weights
        for name, base_weight in self.base_weights.items():
            
            # Default to the base weight. This ensures every key has a float value.
            final_lambdas[name] = float(base_weight)

            # Check if this lambda is supposed to be dynamically controlled
            if name in self.targets:
                metric_key = name
                raw_value = last_metrics.get(metric_key)
                
                if raw_value is None:
                    continue # Skip update if metric not present, will use base_weight

                try:
                    current_loss = float(raw_value)
                except (TypeError, ValueError):
                    continue

                kp, ki, kd = self.gains[name]
                target = self.targets[name]
                error = current_loss - target
                
                self.state['integral_error'][name] += error
                self.state['integral_error'][name] = np.clip(self.state['integral_error'][name], -5.0, 5.0)
                derivative = error - self.state['last_error'][name]
                
                adjustment = (kp * error) + (ki * self.state['integral_error'][name]) + (kd * derivative)
                multiplier = np.exp(adjustment)
                
                calculated_lambda = self.base_weights[name] * multiplier
                
                self.state['last_error'][name] = error
                
                # **THE CRITICAL FIX**: Ensure the final updated value is also a standard float.
                final_lambdas[name] = float(np.clip(calculated_lambda, 0.2, 5.0))
                
        return final_lambdas
    
    def get_sanitized_state_for_ui(self) -> dict:
        return {
            'last_error': {k: float(v) for k, v in self.state['last_error'].items()},
            'integral_error': {k: float(v) for k, v in self.state['integral_error'].items()},
        }

    def state_dict(self):
        return self.state
    
    def load_state_dict(self, state):
        # Ensure loaded state is also sanitized
        self.state['integral_error'] = state.get('integral_error', {k: 0.0 for k in self.targets.keys()})
        self.state['last_error'] = state.get('last_error', {k: 0.0 for k in self.targets.keys()})       
        
        
class TokenizerTrainer(AdvancedTrainer):
    def __init__(self, args):
        # The super().__init__(args) call is now correctly REMOVED.
        self.args = args
        self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.generator_model = LatentTokenizerVQGAN(args.num_codes, args.code_dim, args.latent_grid_size, self.dtype)
        self.discriminator_model = PatchDiscriminator(dtype=self.dtype)
        self.perceptual_loss_fn = JAXMultiMetricPerceptualLoss()

        self.d_loss_ema_alpha = 0.05
        self.d_loss_target_min = 0.3
        self.d_loss_target_max = 0.6
        self.d_lockout_threshold = 0.002 # Made less sensitive
        self.d_lockout_duration = 5

        pid_gains = {
            'l1': (1.5, 0.01, 2.0), 'vq': (1.8, 0.01, 2.5),
            'moment': (1.0, 0.01, 1.0), 'fft': (1.2, 0.01, 1.5), 'autocorr': (2.5, 0.02, 3.5),
            'edge': (2.8, 0.02, 4.0), 'color_cov': (2.0, 0.02, 2.5), 'ssim': (3.0, 0.03, 3.0)
        }
        self.lambda_controller = PIDLambdaController(
            targets={'l1': 0.05, 'vq': 0.1, 'moment': 0.2, 'fft': 0.5, 'autocorr': 0.1, 'edge': 0.1, 'color_cov': 0.05, 'ssim': 0.02},
            base_weights={'l1': 10.0, 'vq': 1.5, 'adv': 0.5, 'moment': 0.5, 'fft': 0.5, 'autocorr': 2.0, 'edge': 7.5, 'color_cov': 2.5, 'ssim': 10.0},
            gains=pid_gains
        )
        
        # Python-side state for UI and non-JIT logic
        self.ui_lock = threading.Lock()
        self.param_count = 0
        self.hist_len = 400
        self.g_loss_hist = deque(maxlen=self.hist_len); self.d_loss_hist = deque(maxlen=self.hist_len)
        self.l1_hist = deque(maxlen=self.hist_len); self.ssim_hist = deque(maxlen=self.hist_len)
        self.vq_hist = deque(maxlen=self.hist_len); self.varent_hist = deque(maxlen=self.hist_len)
        self.last_metrics_for_ui = {}
        self.current_lambdas_for_ui = {}
        self.p1_params = None; self.p1_decoder_model = None
        self.preview_latents = None; self.current_preview_np = None
        self.current_recon_np = None; self.rendered_original_preview = None
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
            gan_balancer_tbl.add_row("D Loss EMA", f"{self.last_metrics_for_ui.get('d_loss_ema', 0.5):.3f}"); gan_balancer_tbl.add_row("G LR Mult", f"{self.last_metrics_for_ui.get('g_lr_mult', 1.0):.2f}x"); gan_balancer_tbl.add_row("D LR Mult", f"{self.last_metrics_for_ui.get('d_lr_mult', 1.0):.2f}x")
            if self.last_metrics_for_ui.get('d_lockout_steps', 0) > 0: gan_balancer_tbl.add_row("Status", Text(f"LOCKED ({int(self.last_metrics_for_ui.get('d_lockout_steps', 0))})", style="bold red"))
            left_stack["gan_balancer"].update(Panel(gan_balancer_tbl, title="[bold]⚖️ GAN Balancer[/]", border_style="yellow"))
            
            q_tbl = Table.grid(expand=True); q_tbl.add_column(style="dim",width=12); q_tbl.add_column()
            q_status_code = int(self.last_metrics_for_ui.get('q_status_code', 0))
            status_map = {0: ("WARMUP", "blue", "🐣"), 1: ("IMPROVING", "green", "😎"), 2: ("STAGNATED", "yellow", "🤔"), 3: ("REGRESSING", "red", "😠")}
            q_status_str, q_color, q_emoji = status_map.get(q_status_code, ("N/A", "dim", "🤖"))
            q_tbl.add_row("Base LR", f"[{q_color}]{self.last_metrics_for_ui.get('lr', 0.0):.2e}[/] {q_emoji}"); q_tbl.add_row("Reward", f"{self.last_metrics_for_ui.get('q_reward', 0.0):+.2e}"); q_tbl.add_row("Exploration", f"{self.last_metrics_for_ui.get('q_explore_rate', 0.0):.2e}")
            q_panel_content = q_tbl
            left_stack["q_controller"].update(Panel(q_panel_content, title="[bold]🤖 Q-Controller[/]", border_style="green"))
            
            pid_internals_tbl = Table("Loss", "Error", "Integral", "Deriv", "Mult", "Final λ", title_style="bold yellow")
            
            # Get the sanitized, float-only state from the controller
            sanitized_pid_state = self.lambda_controller.get_sanitized_state_for_ui()
            
            for name in self.lambda_controller.targets:
                # Read from the sanitized state dictionary
                error = sanitized_pid_state['last_error'].get(name, 0.0)
                integral = sanitized_pid_state['integral_error'].get(name, 0.0)
                
                # The derivative is still a fresh calculation based on the new error
                # We need a way to store the previous last_error to calculate it.
                # For simplicity in the UI, we can just show the current error again or 0.
                # Let's calculate it properly by storing one more piece of state.
                # Or even simpler, let's just show what we have.
                # The PID controller itself calculates the derivative correctly.
                # The UI just needs to display it.
                # Let's just display the error and integral for now, which are the most important state variables.
                
                pid_internals_tbl.add_row(
                    name.capitalize(),
                    f"{error:+.2e}",
                    f"{integral:+.2e}",
                    "-", # Derivative is transient, not stored
                    "-", # Multiplier is transient, not stored
                    f"{self.current_lambdas_for_ui.get(name, 0.0):.2e}"
                )
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
        
        def lr_schedule_fn(step): return self.args.lr # Placeholder
        gen_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule_fn))
        disc_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule_fn))
        
        dummy_input = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), self.dtype)
        gen_params = self.generator_model.init(g_key, dummy_input)['params']
        disc_params = self.discriminator_model.init(d_key, dummy_input)['params']
        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, gen_params, 0)
        
        gen_state = GeneratorTrainState.create(apply_fn=self.generator_model.apply, params=gen_params, tx=gen_optimizer, q_state=init_q_controller(Q_CONTROLLER_CONFIG_TOKENIZER))
        disc_state = CustomTrainState.create(apply_fn=self.discriminator_model.apply, params=disc_params, tx=disc_optimizer)
        states = GANTrainStates(generator=gen_state, discriminator=disc_state)
        
        ckpt_path = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_final.pkl"); ckpt_path_best = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_best.pkl")
        best_val_loss, start_epoch, global_step = float('inf'), 0, 0
        
        if ckpt_path.exists():
            console.print(f"--- Resuming training state from: [green]{ckpt_path}[/green] ---")
            with open(ckpt_path, 'rb') as f: ckpt = pickle.load(f)
            
            q_state_from_ckpt = ckpt.get('q_controller_state')
            if isinstance(q_state_from_ckpt, dict):
                console.print("--- Found old dictionary-based Q-Controller state. Converting to new dataclass format... ---")
                default_q_state = init_q_controller(Q_CONTROLLER_CONFIG_TOKENIZER)
                q_state_loaded = QControllerState(
                    q_table=jnp.array(q_state_from_ckpt.get('q_table', default_q_state.q_table)),
                    metric_history=jnp.array(list(q_state_from_ckpt.get('metric_history', default_q_state.metric_history))),
                    trend_history=jnp.array(list(q_state_from_ckpt.get('trend_history', default_q_state.trend_history))),
                    current_value=jnp.array(q_state_from_ckpt.get('current_value', default_q_state.current_value)),
                    exploration_rate=jnp.array(q_state_from_ckpt.get('exploration_rate_q', default_q_state.exploration_rate)),
                    step_count=jnp.array(q_state_from_ckpt.get('_step_count', default_q_state.step_count)),
                    last_action_idx=default_q_state.last_action_idx,
                    last_reward=default_q_state.last_reward,
                    status_code=default_q_state.status_code
                )
            elif isinstance(q_state_from_ckpt, QControllerState):
                 q_state_loaded = q_state_from_ckpt
            else:
                 q_state_loaded = gen_state.q_state
            
            console.print("--- Resetting optimizer state to match current definition... ---")
            gen_state = GeneratorTrainState.create(apply_fn=gen_state.apply_fn, params=ckpt['gen_params'], tx=gen_optimizer, q_state=q_state_loaded)
            disc_state = CustomTrainState.create(apply_fn=disc_state.apply_fn, params=ckpt['disc_params'], tx=disc_optimizer)
            
            states = GANTrainStates(generator=gen_state, discriminator=disc_state)
            start_epoch = ckpt.get('epoch', 0)
            global_step = ckpt.get('global_step', start_epoch * steps_per_epoch)
            if 'pid_controller_state' in ckpt: self.lambda_controller.load_state_dict(ckpt['pid_controller_state'])
            if ckpt_path_best.exists():
                with open(ckpt_path_best, 'rb') as f_best: best_ckpt = pickle.load(f_best)
                states = states._replace(generator=states.generator.replace(params=best_ckpt['params']))
                best_val_loss = best_ckpt.get('val_loss', float('inf'))
            console.print(f"✅ Resuming session from epoch {start_epoch + 1}, step {global_step}.")
        
        @jit 
        def train_step(states, batch, key, lambdas, d_loss_ema):
            g_key, d_key, q_key = jax.random.split(key, 3)
            new_q_state = q_controller_choose_action(states.generator.q_state, q_key)
            g_lr = new_q_state.current_value
            d_lr = g_lr * 0.4

            d_is_locked_out = d_loss_ema < self.d_lockout_threshold
            d_lockout_steps_indicator = jax.lax.cond(d_is_locked_out, lambda: jnp.array(self.d_lockout_duration, dtype=jnp.int32), lambda: jnp.array(0, dtype=jnp.int32))
            
            g_lr_mult, d_lr_mult = 1.0, 1.0
            def rebalance_lrs():
                g_mult = jax.lax.cond(d_loss_ema < self.d_loss_target_min, lambda: 1.5, lambda: 0.5)
                d_mult = jax.lax.cond(d_loss_ema > self.d_loss_target_max, lambda: 1.5, lambda: 0.5)
                return g_mult, d_mult
            g_lr_mult, d_lr_mult = jax.lax.cond(d_is_locked_out, lambda: (1.0, 0.0), rebalance_lrs)

            (lambda_l1, lambda_vq, lambda_adv, lambda_stink, lambda_moment, lambda_fft, lambda_autocorr, lambda_edge, lambda_color_cov, lambda_ssim) = lambdas

            def generator_loss_fn(p):
                gen_output = states.generator.apply_fn({'params': p}, batch)
                recon = gen_output['reconstructed_path_params']
                l1_loss = jnp.mean(jnp.abs(batch - recon))
                vq_loss = gen_output['vq_loss']
                adv_loss = jnp.mean((states.discriminator.apply_fn({'params': states.discriminator.params}, recon) - 1)**2)
                perceptual_losses = self.perceptual_loss_fn(batch, recon, g_key)
                _, varent = ent_varent(gen_output['pre_quant_latents'].reshape(-1, gen_output['pre_quant_latents'].shape[-1]))
                varentropy_loss = jnp.mean(varent)
                total_loss = (lambda_l1 * l1_loss) + (lambda_vq * vq_loss) + (lambda_adv * adv_loss) + (lambda_stink * varentropy_loss) + (lambda_moment * perceptual_losses['moment']) + (lambda_fft * perceptual_losses['fft']) + (lambda_autocorr * perceptual_losses['autocorr']) + (lambda_edge * perceptual_losses['edge']) + (lambda_color_cov * perceptual_losses['color_cov']) + (lambda_ssim * perceptual_losses['ssim'])
                all_metrics = {'l1': l1_loss, 'vq': vq_loss, 'adv': adv_loss, 'varentropy': varentropy_loss}
                all_metrics.update(perceptual_losses)
                return total_loss, all_metrics
            
            (g_loss_total, metrics), g_grads = jax.value_and_grad(generator_loss_fn, has_aux=True)(states.generator.params)
            new_gen_state = states.generator.apply_gradients(grads=g_grads, learning_rate=(g_lr * g_lr_mult))
            
            def discriminator_loss_fn(disc_params):
                def compute_d_loss():
                    recon = states.generator.apply_fn({'params': new_gen_state.params}, batch)['reconstructed_path_params']
                    loss_real = jnp.mean((states.discriminator.apply_fn({'params': disc_params}, batch) - 1)**2)
                    loss_fake = jnp.mean(states.discriminator.apply_fn({'params': disc_params}, jax.lax.stop_gradient(recon))**2)
                    return (loss_real + loss_fake) * 0.5
                return jax.lax.cond(d_is_locked_out, lambda: jnp.array(0.0, dtype=self.dtype), compute_d_loss)
            
            d_loss, d_grads = jax.value_and_grad(discriminator_loss_fn)(states.discriminator.params)
            new_disc_state = states.discriminator.apply_gradients(grads=d_grads, learning_rate=(d_lr * d_lr_mult))

            final_q_state = q_controller_update(new_q_state, g_loss_total)
            new_d_loss_ema = d_loss_ema * (1 - self.d_loss_ema_alpha) + d_loss * self.d_loss_ema_alpha
            
            metrics.update({'g_loss': g_loss_total, 'd_loss': d_loss, 'lr': g_lr, 'q_status_code': final_q_state.status_code, 'q_reward': final_q_state.last_reward, 'q_explore_rate': final_q_state.exploration_rate, 'd_loss_ema': new_d_loss_ema, 'g_lr_mult': g_lr_mult, 'd_lr_mult': d_lr_mult, 'd_lockout_steps': d_lockout_steps_indicator})
            return GANTrainStates(generator=new_gen_state.replace(q_state=final_q_state), discriminator=new_disc_state), metrics

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
        
        console.print("[bold yellow]🚀 JIT compiling GAN training step...[/bold yellow]")
        dummy_batch = jnp.asarray(train_data[:self.args.batch_size], dtype=self.dtype); compile_key, self.train_key = jax.random.split(self.train_key)
        dummy_lambda_dict = self.lambda_controller(self.last_metrics_for_ui)
        lambda_keys = ['l1', 'vq', 'adv', 'stink', 'moment', 'fft', 'autocorr', 'edge', 'color_cov', 'ssim']
        dummy_lambdas = tuple(dummy_lambda_dict.get(k, 0.2 if k == 'stink' else 0.0) for k in lambda_keys)
        states, _ = train_step(states, dummy_batch, compile_key, dummy_lambdas, jnp.array(0.5))
        console.print("[green]✅ Compilation complete.[/green]")
        
        self.progress = Progress(TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[val_loss]:.2e}[/]"), BarColumn(), "•", TextColumn("Step {task.fields[step]}/{task.fields[steps_per_epoch]}"), "•", TimeRemainingColumn(), TextColumn("Ctrl+C to Exit"))
        epoch_task = self.progress.add_task("epochs", total=self.args.epochs, completed=start_epoch, val_loss=best_val_loss, step=global_step % steps_per_epoch, steps_per_epoch=steps_per_epoch)
        rng = np.random.default_rng(self.args.seed); train_indices_shuffler = np.arange(len(train_data))
        last_ui_update_time = 0.0; UI_UPDATE_INTERVAL_SECS = 0.25
        d_loss_ema_py = 0.5

        try:
            with Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for epoch in range(start_epoch, self.args.epochs + 1):
                    if self.should_shutdown: break
                    rng.shuffle(train_indices_shuffler)
                    for step_in_epoch in range(global_step % steps_per_epoch, steps_per_epoch):
                        if self.should_shutdown: break
                        
                        start_idx = step_in_epoch * self.args.batch_size; end_idx = start_idx + self.args.batch_size
                        if start_idx >= len(train_indices_shuffler): continue
                        batch_indices = train_indices_shuffler[start_idx:end_idx]; train_batch = jnp.asarray(train_data[batch_indices], dtype=self.dtype)
                        
                        perc_boost = self.get_geometric_boosts(train_batch)
                        # --- [FIX] ---
                        # Convert the JAX array to a standard Python float before using it in the lambda dictionary.
                        # This prevents the dictionary from being "infected" with JAX arrays, which the UI cannot format.
                        perc_boost_py = perc_boost.item()
                        
                        lambda_dict = self.lambda_controller(self.last_metrics_for_ui)
                        for k in self.lambda_controller.targets.keys():
                            if k not in ['l1', 'vq', 'adv']:
                                lambda_dict[k] *= perc_boost_py
                        # --- [END FIX] ---

                        self.current_lambdas_for_ui = lambda_dict
                        current_lambdas = tuple(lambda_dict.get(k, 0.2 if k == 'stink' else 0.0) for k in lambda_keys)
                        
                        step_key, self.train_key = jax.random.split(self.train_key)
                        states, metrics = train_step(states, train_batch, step_key, current_lambdas, jnp.array(d_loss_ema_py))
                        
                        metrics_cpu = jax.device_get(metrics)
                        self.last_metrics_for_ui = {k: v.item() for k, v in metrics_cpu.items()}
                        d_loss_ema_py = self.last_metrics_for_ui['d_loss_ema']
                        
                        with self.ui_lock:
                            self.g_loss_hist.append(self.last_metrics_for_ui['g_loss']); self.d_loss_hist.append(self.last_metrics_for_ui['d_loss']); self.l1_hist.append(self.last_metrics_for_ui['l1']); self.ssim_hist.append(self.last_metrics_for_ui.get('ssim',0.0)); self.vq_hist.append(self.last_metrics_for_ui['vq']); self.varent_hist.append(self.last_metrics_for_ui['varentropy'])
                        
                        global_step += 1
                        self.progress.update(epoch_task, step=step_in_epoch + 1)
                        
                        if global_step % self.args.eval_every == 0:
                            if not (active_preview_future and not active_preview_future.done()):
                                host_gen_params = jax.device_get(states.generator.params)
                                active_preview_future = background_executor.submit(self._update_preview_task, host_gen_params, self.generator_model.apply, self.p1_params, self.preview_latents)
                            
                            if len(val_data) > 0:
                                eval_key, self.train_key = jax.random.split(self.train_key)
                                val_batch_size = self.args.batch_size
                                eval_keys = jax.random.split(eval_key, (len(val_data) // val_batch_size) + 1)
                                val_losses = [eval_step(states.generator.params, self.generator_model.apply, jnp.asarray(val_data[i:i+val_batch_size], dtype=self.dtype), eval_keys[j]) for j, i in enumerate(range(0, len(val_data), val_batch_size)) if len(val_data[i:i+val_batch_size]) == val_batch_size]
                                val_loss = np.mean([v.item() for v in val_losses]) if val_losses else float('inf')
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    console.print(f"\n[bold magenta]🏆 New best val loss: {best_val_loss:.2e} @ step {global_step}. Saving...[/bold magenta]")
                                    with open(ckpt_path_best, 'wb') as f: pickle.dump({'params': jax.device_get(states.generator.params), 'val_loss': best_val_loss, 'epoch': epoch, 'global_step': global_step}, f)
                                self.progress.update(epoch_task, val_loss=best_val_loss)

                        current_time = time.time()
                        if current_time - last_ui_update_time > UI_UPDATE_INTERVAL_SECS: live.update(self._generate_layout()); last_ui_update_time = current_time

                    self.progress.update(epoch_task, advance=1, step=0)
                    global_step = (epoch + 1) * steps_per_epoch
                    
                    host_state_to_save = jax.device_get(states)
                    pid_state_to_save = self.lambda_controller.state_dict()
                    data_to_save = {'gen_params': host_state_to_save.generator.params, 'gen_opt_state': host_state_to_save.generator.opt_state, 'disc_params': host_state_to_save.discriminator.params, 'disc_opt_state': host_state_to_save.discriminator.opt_state, 'q_controller_state': host_state_to_save.generator.q_state, 'epoch': epoch, 'global_step': global_step, 'pid_controller_state': pid_state_to_save}
                    with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)

        finally:
            console.print(f"\n[yellow]--- Training loop exited. Waiting for background tasks to finish... ---[/yellow]")
            background_executor.shutdown(wait=True)
            final_epoch_count = epoch if 'epoch' in locals() else 0
            if 'states' in locals():
                console.print(f"\n[yellow]--- Training session finished at epoch {final_epoch_count+1}. Saving final state... ---[/yellow]")
                host_state_final = jax.device_get(states)
                pid_state_to_save = self.lambda_controller.state_dict()
                final_data = {'gen_params': host_state_final.generator.params, 'gen_opt_state': host_state_final.generator.opt_state, 'disc_params': host_state_final.discriminator.params, 'disc_opt_state': host_state_final.discriminator.opt_state, 'q_controller_state': host_state_final.generator.q_state, 'epoch': final_epoch_count, 'global_step': global_step, 'pid_controller_state': pid_state_to_save}
                with open(ckpt_path, 'wb') as f: pickle.dump(final_data, f)
                console.print(f"✅ Final resume-state saved to [green]{ckpt_path}[/green]")

            config = {'num_codes': self.args.num_codes, 'code_dim': self.args.code_dim, 'latent_grid_size': self.args.latent_grid_size}
            config_path = Path(str(ckpt_path).replace("_final.pkl", "_config.pkl"))
            with open(config_path, 'wb') as f: pickle.dump(config, f)
            console.print(f"✅ Config saved to [green]{config_path}[/green]")
            if ckpt_path_best.exists(): console.print(f"👑 Best model (by validation) remains at [bold magenta]{ckpt_path_best}[/bold magenta]")


    def _update_preview_task(self, gen_params, gen_apply_fn, p1_params, preview_latents_batch):
        recon_batch = generate_preview(gen_params, gen_apply_fn, p1_params, preview_latents_batch)
        recon_batch.block_until_ready()
        recon_np = np.array(((recon_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        with self.ui_lock:
            self.current_recon_np = recon_np
            if Pixels:
                self.rendered_recon_preview = Align.center(Pixels.from_image(Image.fromarray(self.current_recon_np)))
                
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
