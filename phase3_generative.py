# =================================================================================================
#
#                    PHASE 3: TEXT-TO-IMAGE GENERATIVE FRAMEWORK (V5.1 - FINAL)
#
#     A Deterministic, Physics-Informed Framework for Structured Media Synthesis
#                   (Upgraded with Full CLIP Coherence Objective)
#
# =================================================================================================

import os
import sys
import argparse
import pickle
import time
import math
import platform
import threading
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Optional, Dict, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import atexit
from dataclasses import dataclass, field, replace

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
from flax.linen import remat
from tqdm import tqdm
from PIL import Image

# --- Dependency Checks ---
try:
    from rich_pixels import Pixels
except ImportError:
    Pixels = None
try:
    # Suppress verbose logging from transformers
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import clip
    import torch
    from transformers import FlaxCLIPVisionModel, FlaxCLIPModel
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.align import Align
    from rich.table import Table
    from rich.text import Text
    import pynvml
    pynvml.nvmlInit()
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("[FATAL] Required dependencies missing: tensorflow, clip-by-openai, torch, transformers, rich, pynvml. Please install them.")
    sys.exit(1)

if platform.system() == "Windows":
    import msvcrt
else:
    import tty, termios, select

jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)


# =================================================================================================
# 1. CORE MODEL DEFINITIONS
# =================================================================================================


def hsv_to_rgb(h, s, v):
    """Differentiable HSV to RGB conversion for JAX."""
    i = jnp.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    
    i = jnp.asarray(i, dtype=jnp.int32) % 6
    
    # Use lax.switch for conditional logic in JAX
    rgb = jax.lax.switch(i, [
        lambda v, t, p: jnp.stack([v, t, p], axis=-1),
        lambda v, t, p: jnp.stack([q, v, p], axis=-1),
        lambda v, t, p: jnp.stack([p, v, t], axis=-1),
        lambda v, t, p: jnp.stack([p, q, v], axis=-1),
        lambda v, t, p: jnp.stack([t, p, v], axis=-1),
        lambda v, t, p: jnp.stack([v, p, q], axis=-1),
    ], v, t, p)
    return rgb


class SingularityShader(nn.Module):
    """
    A differentiable shader that implements the co-polarized singular phase
    mechanism described in Nature Communications (doi: 10.1038/s41467-025-60956-2).
    It translates learned physical parameters (delta, chi) into RGB colors
    by mapping the phase and amplitude of the complex transmittance to HSV color space.
    """
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, control_params: jnp.ndarray) -> jnp.ndarray:
        # --- 1. Interpret the 3-channel input from the CoordinateDecoder ---
        # The decoder output is in [-1, 1] due to nn.tanh. We map these to
        # physically meaningful ranges for delta, chi, and a brightness scalar.
        
        # Channel 0 -> Delta (Retardance): Map to [0, 4π].
        # This gives the model two full 2π cycles to work with, allowing for
        # more complex phase landscapes. This channel will become the primary
        # control for the final output phase.
        delta = (control_params[..., 0] * 0.5 + 0.5) * 4.0 * jnp.pi

        # Channel 1 -> Chi (Ellipticity): Map to [0, π/2].
        # The paper shows the most efficient path is near chi = π/4. We give the model the
        # full range to learn the optimal path that balances efficiency and effect.
        chi = (control_params[..., 1] * 0.5 + 0.5) * (jnp.pi / 2.0)

        # Channel 2 -> Brightness Scalar: Map to [0, 1].
        # An additional degree of freedom for the model to control overall brightness.
        brightness_scalar = control_params[..., 2] * 0.5 + 0.5

        # --- 2. Calculate the Complex Transmittance using the paper's formula ---
        t_co = PoincareSphere.calculate_co_polarized_transmittance(delta, chi)
        
        # --- 3. Decompose the complex value into Amplitude and Phase ---
        # The amplitude |t_co| determines the physical efficiency/brightness.
        amplitude = jnp.abs(t_co)
        
        # The angle of t_co is the topologically protected geometric phase.
        phase = jnp.angle(t_co) # This will be in [-π, π]

        # --- 4. Map Physics (Amplitude, Phase) to Color (HSV) ---
        # Hue: The phase directly controls the color. We map [-π, π] to [0, 1].
        hue = (phase + jnp.pi) / (2 * jnp.pi)
        
        # Saturation: We can keep it high for vibrant colors.
        saturation = jnp.ones_like(hue) * 0.95
        
        # Value (Brightness): This is determined by the physical transmittance amplitude.
        value = amplitude
        
        # --- 5. Convert HSV to RGB ---
        # This is a standard, differentiable conversion.
        rendered_hsv = hsv_to_rgb(hue, saturation, value)
        
        # --- 6. Apply the final learned brightness control ---
        final_rgb = rendered_hsv * brightness_scalar[..., None]
        
        return jnp.asarray(final_rgb, dtype=self.dtype)
 
class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        x = images; features = 32; current_dim = self.input_image_size; i = 0
        while (current_dim // 2) >= self.latent_grid_size and (current_dim // 2) > 0:
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            features *= 2; current_dim //= 2; i += 1
        if current_dim != self.latent_grid_size:
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        path_params = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1)

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
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
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x*freq) for freq in freqs for f in (jnp.sin, jnp.cos)], -1)

class CoordinateDecoder(nn.Module):
    d_model: int; num_freqs: int=10; mlp_width: int=256; mlp_depth: int=4; dtype: Any=jnp.float32
    
    @nn.compact
    def __call__(self, feature_grid, coords, command_vector: Optional[jnp.ndarray] = None):
        B, H, W, _ = feature_grid.shape; enc_coords = PositionalEncoding(self.num_freqs, self.dtype)(coords)
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

        if command_vector is not None:
            tiled_command = jnp.repeat(command_vector[:, None, :], mlp_in.shape[1], axis=1)

        for i in range(self.mlp_depth):
            # The Dense layer is defined here thanks to @nn.compact
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            
            if command_vector is not None:
                # Define the modulation network for this specific layer.
                # @nn.compact allows us to do this here. We access self.mlp_width.
                modulation = nn.Sequential([
                    nn.Dense(self.mlp_width, dtype=self.dtype, name=f"mod_dense_{i}"),
                    nn.silu,
                    nn.Dense(self.mlp_width * 2, dtype=self.dtype, name=f"mod_out_{i}")
                ])(tiled_command)
                
                scale, shift = jnp.split(modulation, 2, axis=-1)
                
                # Apply Feature-Wise Linear Modulation (FiLM)
                h = h * (1 + scale) + shift

            h = nn.gelu(h)
            if i == self.mlp_depth // 2: h = jnp.concatenate([h, mlp_in], -1)
            
        return nn.tanh(nn.Dense(3, name="mlp_out", dtype=self.dtype)(h))
        
        
class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(self.d_model, name="coord_decoder", dtype=self.dtype)

    def __call__(self, images):
        path_params = self.modulator(images)
        return self.observer(path_params)

    def decode(self, path_params, coords, command_vector: Optional[jnp.ndarray] = None):
        """
        Decodes path parameters at given coordinates, now with optional text conditioning.
        """
        return self.coord_decoder(self.observer(path_params), coords, command_vector)

    def get_features(self, path_params):
        """A new method to correctly access the observer submodule within an apply context."""
        return self.observer(path_params)

# =================================================================================================
# 2. ADVANCED TRAINING TOOLKIT (FULL SUITE)
# =================================================================================================
class CustomTrainState(train_state.TrainState):
    def apply_gradients(self, *, grads: Any, **kwargs) -> "CustomTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, **kwargs)
        new_params = optax.apply_updates(self.params, updates)
        known_keys = self.__dataclass_fields__.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_keys}
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **filtered_kwargs)

class InteractivityState:
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_prompt_change = 0
        self.sentinel_dampening_log_factor = 0.0
        self.shutdown_event = threading.Event()
        self.force_save = False
    def get_and_reset_preview_change(self):
        with self.lock: change = self.preview_prompt_change; self.preview_prompt_change = 0; return change
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def update_sentinel_factor(self, direction):
        with self.lock: self.sentinel_dampening_log_factor = np.clip(self.sentinel_dampening_log_factor + direction * 0.5, -3.0, 0.0)
    def get_sentinel_factor(self):
        with self.lock: return 10**self.sentinel_dampening_log_factor
    def set_shutdown(self): self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    if platform.system() == "Windows":
        # ... (windows logic remains the same, but check H/P directions)
        while not shared_state.shutdown_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x03' or key == b'q': shared_state.set_shutdown(); break
                elif key == b's': 
                    with shared_state.lock: shared_state.force_save = True
                elif key == b'\xe0':
                    arrow = msvcrt.getch()
                    if arrow == b'K': shared_state.preview_prompt_change = -1 # Left
                    elif arrow == b'M': shared_state.preview_prompt_change = 1  # Right
                    # FIX: Up arrow should increase dampening (positive direction)
                    elif arrow == b'H': shared_state.update_sentinel_factor(1.0) # Up
                    # FIX: Down arrow should decrease dampening (negative direction)
                    elif arrow == b'P': shared_state.update_sentinel_factor(-1.0) # Down
            time.sleep(0.05)
    else: # Linux/Mac
        fd = sys.stdin.fileno(); old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not shared_state.shutdown_event.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char == '\x03' or char == 'q': shared_state.set_shutdown(); break
                    elif char == 's': 
                        with shared_state.lock: shared_state.force_save = True
                    elif char == '\x1b':
                        next_chars = sys.stdin.read(2)
                        # FIX: Up arrow should increase dampening (positive direction)
                        if next_chars == '[A': shared_state.update_sentinel_factor(1.0)
                        # FIX: Down arrow should decrease dampening (negative direction)
                        elif next_chars == '[B': shared_state.update_sentinel_factor(-1.0)
                        elif next_chars == '[C': shared_state.preview_prompt_change = 1
                        elif next_chars == '[D': shared_state.preview_prompt_change = -1
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# In InteractivityState, update the factor change logic
    def update_sentinel_factor(self, direction):
        # Direction +1.0 moves towards 0.001 (more dampening)
        # Direction -1.0 moves towards 1.0 (less dampening)
        with self.lock: self.sentinel_dampening_log_factor = np.clip(self.sentinel_dampening_log_factor - direction * 0.5, -3.0, 0.0)
        
def get_sentinel_lever_ascii(log_factor: float):
    # Map the log_factor range [0.0, -3.0] to an index range [0, 6]
    # Ensure the value is clipped to prevent floating point errors from going out of bounds
    clipped_factor = np.clip(log_factor, -3.0, 0.0)
    
    # Calculate the fractional position and scale it to the number of indices
    # We use -clipped_factor because the range is inverted (0 is max, -3 is min)
    idx = int((-clipped_factor / 3.0) * 6)
    
    # Final safety clip to handle any edge cases
    idx = np.clip(idx, 0, 6)
    
    lever_bars = ["│         │"] * 7
    lever_bars[idx] = "│█████████│"
    labels = ["1.0 (Off)", " ", "0.1", " ", "0.01", " ", "0.001"]
    return "\n".join([f" {labels[i]:<10} {lever_bars[i]}" for i in range(7)])
    
class SentinelState(NamedTuple):
    # Replaces sign_history with a much more efficient EMA
    sign_ema: chex.ArrayTree
    dampened_count: Optional[jnp.ndarray] = None
    dampened_pct: Optional[jnp.ndarray] = None

def sentinel(decay: float = 0.9, oscillation_threshold: float = 0.5) -> optax.GradientTransformation:
    """A faster, EMA-based Sentinel optimizer."""
    def init_fn(params):
        sign_ema = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t), params)
        return SentinelState(sign_ema=sign_ema, dampened_count=jnp.array(0), dampened_pct=jnp.array(0.0))

    def update_fn(updates, state, params=None, **kwargs):
        dampening_factor = kwargs.get('dampening_factor', 1.0)
        
        # Get the sign of the current gradient update
        current_sign = jax.tree_util.tree_map(jnp.sign, updates)
        
        # Update the Exponential Moving Average of the signs
        new_sign_ema = jax.tree_util.tree_map(
            lambda ema, sign: ema * decay + sign * (1 - decay),
            state.sign_ema,
            current_sign
        )
        
        # An oscillation is detected if the EMA of the signs is close to zero
        is_oscillating = jax.tree_util.tree_map(
            lambda ema: jnp.abs(ema) < oscillation_threshold,
            new_sign_ema
        )
        
        # If dampening is off (factor is 1.0), we can skip the expensive parts
        def apply_dampening():
            dampening_mask = jax.tree_util.tree_map(lambda is_osc: jnp.where(is_osc, dampening_factor, 1.0), is_oscillating)
            dampened_updates = jax.tree_util.tree_map(lambda u, m: u * m, updates, dampening_mask)
            num_oscillating = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.sum, is_oscillating)))
            total_params = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, params)))
            dampened_pct = num_oscillating / (total_params + 1e-8)
            return dampened_updates, num_oscillating, dampened_pct

        def skip_dampening():
            return updates, jnp.array(0), jnp.array(0.0)

        # Use lax.cond to avoid computation when dampening is off.
        dampened_updates, num_oscillating, dampened_pct = jax.lax.cond(
            dampening_factor < 1.0,
            apply_dampening,
            skip_dampening
        )

        new_state = SentinelState(sign_ema=new_sign_ema, dampened_count=num_oscillating, dampened_pct=dampened_pct)
        return dampened_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


Q_CONTROLLER_CONFIG_MANIFOLD = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": (0.9, 0.95, 1.0, 1.05, 1.1), "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-6, "lr_max": 5e-4, "metric_history_len": 5000, "loss_min": 2.0, "loss_max": 10.0, "exploration_rate_q": 0.2, "min_exploration_rate": 0.05, "exploration_decay": 0.9995, "trend_window": 500, "improve_threshold": 1e-4, "regress_threshold": 1e-5, "regress_penalty": 10.0, "stagnation_penalty": -2.0, "warmup_steps": 1000, "warmup_lr_start": 1e-6}

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
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
    config = Q_CONTROLLER_CONFIG_MANIFOLD
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config["warmup_steps"]
        new_value = config["warmup_lr_start"] * (1 - alpha) + config["lr_max"] * 0.1 * alpha
        return replace(state, current_value=new_value, step_count=state.step_count + 1, status_code=jnp.array(0))
    def regular_action():
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config["metric_history_len"] - 5, 5))
        state_idx = jnp.clip(((metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        explore_key, action_key = jax.random.split(key)
        def explore(): return jax.random.randint(action_key, (), 0, config["num_lr_actions"])
        def exploit(): return jnp.argmax(state.q_table[state_idx])
        action_idx = jax.lax.cond(jax.random.uniform(explore_key) < state.exploration_rate, explore, exploit)
        selected_factor = jax.lax.switch(action_idx, [lambda: f for f in config["lr_change_factors"]])
        new_value = jnp.clip(state.current_value * selected_factor, config["lr_min"], config["lr_max"])
        return replace(state, current_value=new_value, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config["warmup_steps"], warmup_action, regular_action)

@jax.jit
def q_controller_update(state: QControllerState, metric_value: float):
    config = Q_CONTROLLER_CONFIG_MANIFOLD
    new_metric_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value)
    new_trend_history = jnp.roll(state.trend_history, -1).at[-1].set(metric_value)
    def perform_update(st):
        x = jnp.arange(config["trend_window"], dtype=jnp.float32)
        y = new_trend_history
        A = jnp.vstack([x, jnp.ones_like(x)]).T
        slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        status_code, reward = jax.lax.cond(
            slope < -config["improve_threshold"], lambda: (jnp.array(1), abs(slope) * 1000.0),
            lambda: jax.lax.cond(
                slope > config["regress_threshold"], lambda: (jnp.array(3), -abs(slope) * 1000.0 - config["regress_penalty"]),
                lambda: (jnp.array(2), config["stagnation_penalty"])
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

class PatchDiscriminator(nn.Module):
    num_filters: int = 64; num_layers: int = 3; dtype: Any = jnp.bfloat16
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

@partial(jax.vmap, in_axes=(0, 0, 0, None, None), out_axes=0)
def _extract_patches_vmapped(image, x_coords, y_coords, patch_size, c):
    def get_patch(x, y):
         return jax.lax.dynamic_slice(image, (y, x, 0), (patch_size, patch_size, c))
    return jax.vmap(get_patch)(x_coords, y_coords)

@partial(jit, static_argnames=('num_moments',))
def calculate_moments(patches, num_moments=4):
    flat = patches.reshape(patches.shape[0], -1, patches.shape[-1])
    mean = jnp.mean(flat, axis=1)
    var = jnp.var(flat, axis=1)
    if num_moments <= 2: return jnp.concatenate([mean, var], axis=-1)
    std_dev = jnp.sqrt(var + 1e-6)
    centered = flat - mean[:, None, :]
    skew = jnp.mean((centered / std_dev[:, None, :])**3, axis=1)
    if num_moments <= 3: return jnp.concatenate([mean, var, skew], axis=-1)
    kurt = jnp.mean((centered / std_dev[:, None, :])**4, axis=1)
    return jnp.concatenate([mean, var, skew, kurt], axis=-1)

@jit
def calculate_ssim_loss(p1, p2, max_val=2.0):
    C1, C2 = (0.01 * max_val)**2, (0.03 * max_val)**2
    p1g, p2g = jnp.mean(p1, axis=-1), jnp.mean(p2, axis=-1)
    mu1, mu2 = jnp.mean(p1g, axis=(1,2)), jnp.mean(p2g, axis=(1,2))
    var1, var2 = jnp.var(p1g, axis=(1,2)), jnp.var(p2g, axis=(1,2))
    covar = jnp.mean(p1g * p2g, axis=(1,2)) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    den = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
    return jnp.mean(1.0 - (num / den))

class JAXMultiMetricPerceptualLoss:
    def __init__(self, num_patches=64, patch_size=32):
        self.num_patches, self.patch_size = num_patches, patch_size
        self._calculate_losses_jit = partial(jit, static_argnames=('batch_size',))(self._calculate_losses)
    def _calculate_losses(self, img1, img2, key, batch_size: int):
        _, h, w, c = img1.shape
        x = jax.random.randint(key, (batch_size, self.num_patches), 0, w - self.patch_size)
        y = jax.random.randint(key, (batch_size, self.num_patches), 0, h - self.patch_size)
        p1 = _extract_patches_vmapped(img1, x, y, self.patch_size, c).reshape(-1, self.patch_size, self.patch_size, c)
        p2 = _extract_patches_vmapped(img2, x, y, self.patch_size, c).reshape(-1, self.patch_size, self.patch_size, c)
        losses = {}
        for scale in [1.0, 0.5]:
            sz = int(self.patch_size * scale)
            if sz < 8: continue
            ps1 = jax.image.resize(p1, (p1.shape[0], sz, sz, c), 'bilinear')
            ps2 = jax.image.resize(p2, (p2.shape[0], sz, sz, c), 'bilinear')
            losses[f'moment_s{scale}'] = jnp.mean(jnp.abs(calculate_moments(ps1) - calculate_moments(ps2)))
            losses[f'ssim_s{scale}'] = calculate_ssim_loss(ps1, ps2)
        return losses
    def __call__(self, img1, img2, key): return self._calculate_losses_jit(img1, img2, key, batch_size=img1.shape[0])

class PIDLambdaController:
    def __init__(self, targets, base_weights, gains):
        self.targets, self.base_weights, self.gains = targets, base_weights, gains
        self.integral_error = {k: 0.0 for k in targets.keys()}
        self.last_error = {k: 0.0 for k in targets.keys()}
    def __call__(self, last_metrics):
        lambdas = self.base_weights.copy()
        for name, target in self.targets.items():
            key = f'loss/{name}'
            if key not in last_metrics: continue
            kp, ki, kd = self.gains[name]
            error = last_metrics[key] - target
            self.integral_error[name] = np.clip(self.integral_error[name] + error, -5.0, 5.0)
            derivative = error - self.last_error[name]
            adjustment = np.exp((kp * error) + (ki * self.integral_error[name]) + (kd * derivative))
            lambdas[name] = np.clip(self.base_weights[name] * adjustment, 0.1, 10.0)
            self.last_error[name] = error
        return lambdas

@jax.jit
def gumbel_softmax_straight_through(logits, key, tau=1.0):
    gumbel_noise = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
    y_soft = jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)
    y_hard_one_hot = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1], dtype=logits.dtype)
    return y_soft + jax.lax.stop_gradient(y_hard_one_hot - y_soft)

@partial(jax.jit, static_argnames=('target_size',))
def encode_image_for_clip(image_batch, depth_map, target_size=224):
    resized = jax.image.resize(image_batch, (image_batch.shape[0], target_size, target_size, 3), 'bilinear')
    depth_resized = jax.image.resize(depth_map[..., None], (depth_map.shape[0], target_size, target_size, 1), 'bilinear')
    depth_salience = depth_resized / (jnp.max(depth_resized, axis=(1,2,3), keepdims=True) + 1e-6)
    weighted_image = resized * (0.5 + 0.5 * depth_salience)
    renormalized = (weighted_image + 1.0) * 0.5
    clip_mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 1, 3)
    clip_std = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 1, 3)
    
    # Produces a tensor of shape (B, H, W, C)
    normalized_image_nhwc = (renormalized - clip_mean) / clip_std
    
    # FIX: Transpose from NHWC (Flax default) to NCHW (Transformers CLIP model expectation)
    # (B, 224, 224, 3) -> (B, 3, 224, 224)
    return jnp.transpose(normalized_image_nhwc, (0, 3, 1, 2))

def calculate_clip_similarity_loss(img_embeddings, text_embeddings, temp=0.07):
    img_embeddings /= jnp.linalg.norm(img_embeddings, axis=-1, keepdims=True) + 1e-6
    text_embeddings /= jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True) + 1e-6
    logits = jnp.einsum('bd,cd->bc', img_embeddings, text_embeddings) / temp
    labels = jnp.arange(img_embeddings.shape[0])
    loss_i = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    loss_t = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
    return (loss_i + loss_t) / 2.0

def calculate_compositional_clip_loss(feature_grid_embeddings, text_embeddings, k_patches=1024):
    """
    Calculates a spatially-aware CLIP loss to promote compositionality. It compares
    the text embedding to the top-k most similar patches in the feature grid,
    encouraging all parts of the image to be relevant to the prompt.
    """
    B, H, W, D = feature_grid_embeddings.shape
    num_patches = H * W
    patches_flat = feature_grid_embeddings.reshape(B, num_patches, D)

    patches_norm = patches_flat / (jnp.linalg.norm(patches_flat, axis=-1, keepdims=True) + 1e-6)
    text_norm = text_embeddings / (jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True) + 1e-6)

    # Calculate cosine similarity of each patch with its corresponding text prompt
    similarities = jnp.einsum('bpd,bd->bp', patches_norm, text_norm) # Shape: (B, num_patches)

    # Select the top-k most similar patches for each item in the batch.
    # This helps the model focus on generating the most relevant objects for the prompt.
    top_k_similarities, _ = jax.lax.top_k(similarities, k=min(k_patches, num_patches))

    # The loss is 1 minus the mean of these top similarities. Minimizing this loss maximizes similarity.
    return (1.0 - top_k_similarities).mean()



class CLIPImageProjector(nn.Module):
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x): return nn.Dense(512, dtype=self.dtype, name="clip_proj")(x)


# =================================================================================================
# 3. VQ-GAN TOKENIZER & State Management
# =================================================================================================

class VectorQuantizer(nn.Module):
    num_codes: int; code_dim: int; beta: float = 0.25
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
    def lookup(self, indices): return self.variables['params']['codebook'].T[indices]

class LatentTokenizerVQGAN(nn.Module):
    num_codes: int; code_dim: int; latent_grid_size: int; dtype: Any = jnp.float32
    def setup(self):
        self.enc_conv1 = nn.Conv(128, (3,3), (2,2), 'SAME', name="enc_conv1", dtype=self.dtype)
        self.enc_conv2 = nn.Conv(256, (3,3), (2,2), 'SAME', name="enc_conv2", dtype=self.dtype)
        self.enc_proj = nn.Conv(self.code_dim, (1,1), name="enc_proj", dtype=self.dtype)
        self.vq = VectorQuantizer(self.num_codes, self.code_dim, name="vq")
        self.dec_convT1 = nn.ConvTranspose(256, (3,3), (2,2), 'SAME', name="dec_convT1", dtype=self.dtype)
        self.dec_convT2 = nn.ConvTranspose(3, (3,3), (2,2), 'SAME', name="dec_convT2", dtype=self.dtype)
    def __call__(self, path_params_grid):
        h = nn.gelu(self.enc_conv1(path_params_grid)); h = nn.gelu(self.enc_conv2(h))
        z_e = self.enc_proj(h)
        vq_out = self.vq(z_e); z_q = vq_out["quantized"]
        p_r = self.dec_convT2(nn.gelu(self.dec_convT1(z_q)))
        return {"reconstructed_path_params": p_r, "indices": vq_out["indices"], "vq_loss": vq_out["loss"], "pre_quant_latents": z_e}
    def encode(self, path_params_grid):
        h = nn.gelu(self.enc_conv1(path_params_grid)); h = nn.gelu(self.enc_conv2(h))
        return self.vq(self.enc_proj(h))["indices"]
    def decode(self, indices):
        z_q = self.vq.lookup(indices)
        return self.dec_convT2(nn.gelu(self.dec_convT1(z_q)))

    def decode_from_quantized(self, z_q):
        """Decodes directly from a grid of quantized vectors."""
        return self.dec_convT2(nn.gelu(self.dec_convT1(z_q)))
        
class ConductorTrainState(CustomTrainState):
    q_state: QControllerState
    ema_params: Any

class GANTrainStates(NamedTuple):
    conductor: ConductorTrainState
    discriminator: CustomTrainState
    clip_projector: CustomTrainState
    painter: CustomTrainState  


# =================================================================================================
# 4. DATA PREPARATION
# =================================================================================================
class TokenizerConfig(NamedTuple): num_codes: int; code_dim: int; latent_grid_size: int; dtype: Any
class P1Config(NamedTuple): d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any

# =================================================================================================
# 5. MANIFOLD CONDUCTOR & FINAL OBJECTIVE TRAINER
# =================================================================================================

class JenkinsBlock(nn.Module):
    d_model: int; num_heads: int; dtype: Any
    @nn.compact
    def __call__(self, x, cond, train=True):
        attn = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype, deterministic=not train)
        mlp = nn.Sequential([nn.Dense(self.d_model*4,dtype=self.dtype), nn.gelu, nn.Dense(self.d_model,dtype=self.dtype)])
        norm1, norm2 = nn.LayerNorm(dtype=self.dtype), nn.LayerNorm(dtype=self.dtype)
        ada_ln_mod = nn.Sequential([nn.silu, nn.Dense(self.d_model * 6, dtype=self.dtype)])
        mod = ada_ln_mod(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)
        res_x = x; x = norm1(x) * (1 + scale_msa) + shift_msa
        x = res_x + gate_msa * attn(x)
        res_x = x; x = norm2(x) * (1 + scale_mlp) + shift_mlp
        x = res_x + gate_mlp * mlp(x)
        return x

RematJenkinsBlock = remat(JenkinsBlock, static_argnums=(3,))

class ManifoldConductor(nn.Module):
    num_codes: int; d_model: int; num_heads: int; num_layers: int; latent_grid_size: int; dtype: Any
    def setup(self):
        grid_dim = (self.latent_grid_size // 4)**2
        self.pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (1, grid_dim, self.d_model), self.dtype)
        self.mask_token_embed = self.param('mask_token', nn.initializers.normal(0.02), (1, 1, self.d_model), self.dtype)
        self.command_proj = nn.Dense(self.d_model, dtype=self.dtype)
        self.token_embedding = nn.Embed(self.num_codes + 2, self.d_model, dtype=self.dtype)
        self.salience_embedding = nn.Embed(self.num_codes, self.d_model, dtype=self.dtype)
        self.output_proj = nn.Sequential([nn.LayerNorm(dtype=self.dtype), nn.Dense(self.num_codes, dtype=self.dtype)])
        self.blocks = [RematJenkinsBlock(self.d_model, self.num_heads, self.dtype, name=f'block_{i}') for i in range(self.num_layers)]

    def __call__(self, input_indices, command_vector, salience_indices, train=True):
        B, H, W = input_indices.shape
        num_tokens = H * W
        x = self.token_embedding(input_indices).reshape(B, num_tokens, -1)
        mask = (input_indices.reshape(B, num_tokens) == self.num_codes + 1)
        x = jnp.where(mask[..., None], self.mask_token_embed, x)
        salience_cond = self.salience_embedding(salience_indices).reshape(B, num_tokens, -1)
        if self.pos_embed.shape[1] != num_tokens:
             pos_embed = jax.image.resize(self.pos_embed, (self.pos_embed.shape[0], num_tokens, self.pos_embed.shape[2]), 'bilinear')
        else:
            pos_embed = self.pos_embed
        x += salience_cond + pos_embed
        cond = self.command_proj(command_vector)[:, None, :]
        for block in self.blocks: x = block(x, cond, train)
        return self.output_proj(x).reshape(B, H, W, self.num_codes)

class AdvancedTrainer:
    def __init__(self, args):
        self.args = args; self.interactive_state = InteractivityState(); self.loss_hist = deque(maxlen=400)
    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except Exception: return "N/A", "N/A"
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

class ManifoldTrainer(AdvancedTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.console = Console()
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.console.print(f"--- [MODE] Manifold Conductor (Final Objective) [bold green]ACTIVE[/bold green]. ---", style="bold yellow")
        
        self.p1_params, self.p1_model, _ = self._load_phase1(args)
        self.tok_params, self.tokenizer, _ = self._load_tokenizer(args)
        
        self.console.print("--- Loading Full CLIP Model for pixel-level loss... ---")
        self.full_clip_model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32", dtype=self.dtype)
        self.clip_params = self.full_clip_model.params

        self.uncond_embedding = jax.device_put(jnp.zeros((1, 512), dtype=self.dtype))
        self.conductor_model = ManifoldConductor(num_codes=args.num_codes, d_model=args.d_model_cond, num_heads=args.num_heads, num_layers=args.num_layers, latent_grid_size=args.latent_grid_size, dtype=self.dtype)
        self.discriminator_model = PatchDiscriminator(dtype=jnp.float32)
        self.clip_projector = CLIPImageProjector(dtype=self.dtype)
        
        self.perceptual_loss_fn = JAXMultiMetricPerceptualLoss(num_patches=64, patch_size=16)
        
        self.pid_controller = PIDLambdaController(
            targets={'mae': 0.1, 'gan_g': 0.5, 'perceptual': 0.2, 'clip': 0.25, 'patch_clip': 0.25, 'global_clip': 0.3},
            base_weights={'mae': 1.0, 'gan_g': 1.0, 'perceptual': 0.5, 'clip': 1.0, 'patch_clip': 1.5, 'global_clip': 1.0},
            gains={'mae': (0.1, 0.01, 0.05), 'gan_g': (0.2, 0.02, 0.1), 'perceptual': (0.2, 0.02, 0.1), 
                   'clip': (0.25, 0.03, 0.1), 'patch_clip': (0.3, 0.04, 0.15), 'global_clip': (0.2, 0.03, 0.1)}
        )
        self.last_metrics_for_ui = {}; self.ui_lock = threading.Lock(); self.loss_hist = deque(maxlen=400)
        self.lambdas = self.pid_controller.base_weights.copy()
        self.validation_prompts = ["a red cup on a table", "a photorealistic orange cat", "a blue ball"]
        clip_model_torch, _ = clip.load("ViT-B/32", device=_clip_device)
        with torch.no_grad():
            tokens = clip.tokenize(self.validation_prompts).to(_clip_device)
            self.validation_embeddings = jax.device_put(clip_model_torch.encode_text(tokens).cpu().numpy().astype(self.dtype))
        
        self.q_status_str = "WARMUP"; self.sentinel_pct = 0.0
        self.rendered_preview = None
        self.current_preview_prompt_idx = 0

    def _load_phase1(self, args):
        p1_path = next(Path('.').glob(f"{args.basename}_{args.d_model}d_512.pkl"))
        with open(p1_path, 'rb') as f: p1_params = pickle.load(f)['params']
        p1_config = P1Config(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=512, dtype=jnp.float32)
        p1_model = TopologicalCoordinateGenerator(**p1_config._asdict())
        return jax.device_put(p1_params), p1_model, p1_config

    def _load_tokenizer(self, args):
        tok_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_{args.num_codes}c_gan_config.pkl"))
        tok_ckpt_path = Path(str(tok_config_path).replace("_config.pkl", "_best.pkl"))
        if not tok_ckpt_path.exists(): tok_ckpt_path = Path(str(tok_ckpt_path).replace("_best.pkl", "_final.pkl"))
        with open(tok_config_path, 'rb') as f: config_dict = pickle.load(f)
        with open(tok_ckpt_path, 'rb') as f: ckpt_data = pickle.load(f); params = ckpt_data.get('gen_params', ckpt_data.get('params'))
        tok_config = TokenizerConfig(**config_dict, dtype=jnp.float32)
        tokenizer = LatentTokenizerVQGAN(**tok_config._asdict())
        return jax.device_put(params), tokenizer, tok_config
        
    def _generate_layout(self):
        layout = Layout()
        layout.split(Layout(name="header", size=3), Layout(ratio=1, name="main"), Layout(size=3, name="footer"))
        mem, util = self._get_gpu_stats()
        loss = self.last_metrics_for_ui.get('loss/total_g', 0.0)
        lr = self.last_metrics_for_ui.get('lr', 0.0)
        header_text = f"🌋 [bold]Manifold Conductor[/] | G-Loss: [yellow]{loss:.4f}[/] | LR: [cyan]{lr:.2e}[/] | GPU: {mem} / {util}"
        layout["header"].update(Panel(Align.center(header_text), style="bold red", title="[dim]wubumind.ai[/dim]", title_align="right"))
        main_grid = Table.grid(expand=True); main_grid.add_column(ratio=2); main_grid.add_column(ratio=3)
        left_grid = Table.grid(expand=True)
        left_grid.add_row(Panel(Align.center(self._get_sparkline(self.loss_hist, 50)), title="Loss Sparkline", border_style="cyan", height=5))
        sentinel_lever_text = get_sentinel_lever_ascii(self.interactive_state.sentinel_dampening_log_factor)
        sentinel_panel = Panel(Group(Text(f"Dampened: {self.sentinel_pct:.2%}", justify="center"), Text(sentinel_lever_text, justify="center")), title="Sentinel (↑/↓)", border_style="yellow")
        q_panel = Panel(Text(self.q_status_str, justify="center"), title="Q-Controller Status", border_style="green")
        lambda_table = Table(title="PID Lambdas", show_header=False, box=None)
        lambda_table.add_column(style="magenta"); lambda_table.add_column(justify="right")
        for k, v in self.lambdas.items(): lambda_table.add_row(k, f"{v:.2f}")
        loss_table = Table(title="Losses", show_header=False, box=None)
        loss_table.add_column(style="cyan"); loss_table.add_column(justify="right")
        
        for k in ['mae', 'gan_g', 'perceptual', 'clip', 'patch_clip', 'global_clip', 'total_d']:
            value = self.last_metrics_for_ui.get(f'loss/{k}', 0.0)
            try: formatted_value = f"{value:.3f}"
            except (ValueError, TypeError): formatted_value = "N/A"
            loss_table.add_row(k, formatted_value)
            
        left_grid.add_row(Panel(Group(q_panel, sentinel_panel, lambda_table, loss_table)))
        preview_content = Align.center("...Awaiting First Generation...")
        if self.rendered_preview and Pixels:
             prompt_text = Text(f"Prompt: \"{self.validation_prompts[self.current_preview_prompt_idx]}\"", justify="center")
             preview_content = Align.center(Group(prompt_text, self.rendered_preview))
        main_grid.add_row(left_grid, Panel(preview_content, title="Live Preview", border_style="magenta"))
        layout["main"].update(main_grid); layout["footer"].update(self.progress); return layout
    
    def _ui_refresher_thread(self, live: Live):
        target_fps = 15.0
        sleep_duration = 1.0 / target_fps
        while not self.interactive_state.shutdown_event.is_set():
            try:
                live.update(self._generate_layout(), refresh=True)
                time.sleep(sleep_duration)
            except Exception: break

    def _save_checkpoint(self, state: GANTrainStates, global_step: int, ckpt_path: Path):
        try:
            save_data = {
                'conductor_state': {
                    'step': state.conductor.step,
                    'params': jax.device_get(state.conductor.params),
                    'opt_state': jax.device_get(state.conductor.opt_state),
                    'q_state': jax.device_get(state.conductor.q_state),
                    'ema_params': jax.device_get(state.conductor.ema_params)
                },
                'discriminator_state': {
                    'step': state.discriminator.step,
                    'params': jax.device_get(state.discriminator.params),
                    'opt_state': jax.device_get(state.discriminator.opt_state)
                },
                'clip_projector_state': {
                    'step': state.clip_projector.step,
                    'params': jax.device_get(state.clip_projector.params),
                    'opt_state': jax.device_get(state.clip_projector.opt_state)
                },
                'painter_state': {
                    'step': state.painter.step,
                    'params': jax.device_get(state.painter.params),
                    'opt_state': jax.device_get(state.painter.opt_state)
                },
                'global_step': global_step,
                'pid_controller_state': {
                    'integral_error': self.pid_controller.integral_error,
                    'last_error': self.pid_controller.last_error
                },
                'loss_hist': list(self.loss_hist)
            }
            
            with open(ckpt_path, 'wb') as f:
                pickle.dump(save_data, f)
            self.console.log(f"💾 Checkpoint saved to [green]{ckpt_path}[/green] at step {global_step}")

            final_ckpt_path = Path(str(ckpt_path).replace("_checkpoint.pkl", "_final.pkl"))
            final_save_data = {
                'conductor_ema_params': jax.device_get(state.conductor.ema_params),
                'p1_params': jax.device_get(state.painter.params),
                'clip_projector_params': jax.device_get(state.clip_projector.params)
            }
            with open(final_ckpt_path, 'wb') as f:
                 pickle.dump(final_save_data, f)
            self.console.log(f"💾 Final generator weights saved to [green]{final_ckpt_path}[/green]")

        except Exception as e:
            self.console.print(f"[bold red]Error saving checkpoint: {e}[/bold red]")

    def _load_checkpoint(self, ckpt_path: Path, optimizers: Dict) -> Tuple[Optional[GANTrainStates], int]:
        if not ckpt_path.exists():
            self.console.print("--- 🏁 No checkpoint found. Starting new training run. ---")
            return None, 0
        
        try:
            self.console.print(f"--- 🚀 Found checkpoint at [green]{ckpt_path}[/green]. Resuming training... ---")
            with open(ckpt_path, 'rb') as f:
                ckpt_data = pickle.load(f)

            cond_data = ckpt_data['conductor_state']
            
            base_cond_state = TrainState.create(
                apply_fn=self.conductor_model.apply,
                params=cond_data['params'],
                tx=optimizers['conductor']
            )
            restored_cond_state = ConductorTrainState(
                step=base_cond_state.step, apply_fn=base_cond_state.apply_fn,
                params=base_cond_state.params, tx=base_cond_state.tx,
                opt_state=base_cond_state.opt_state, q_state=cond_data['q_state'],
                ema_params=cond_data['ema_params']
            ).replace(step=cond_data['step'], opt_state=cond_data['opt_state'])

            disc_data = ckpt_data['discriminator_state']
            restored_disc_state = CustomTrainState.create(
                apply_fn=self.discriminator_model.apply,
                params=disc_data['params'], tx=optimizers['discriminator']
            ).replace(step=disc_data['step'], opt_state=disc_data['opt_state'])

            clip_proj_data = ckpt_data['clip_projector_state']
            restored_clip_proj_state = CustomTrainState.create(
                apply_fn=self.clip_projector.apply,
                params=clip_proj_data['params'], tx=optimizers['clip_projector']
            ).replace(step=clip_proj_data['step'], opt_state=clip_proj_data['opt_state'])
            
            painter_data = ckpt_data['painter_state']
            restored_painter_state = CustomTrainState.create(
                apply_fn=self.p1_model.apply,
                params=painter_data['params'], tx=optimizers['painter']
            ).replace(step=painter_data['step'], opt_state=painter_data['opt_state'])

            state = GANTrainStates(
                conductor=restored_cond_state, discriminator=restored_disc_state, 
                clip_projector=restored_clip_proj_state, painter=restored_painter_state
            )
            
            global_step = ckpt_data['global_step']
            self.pid_controller.integral_error = ckpt_data['pid_controller_state']['integral_error']
            self.pid_controller.last_error = ckpt_data['pid_controller_state']['last_error']
            self.loss_hist = deque(ckpt_data['loss_hist'], maxlen=400)
            
            self.console.print(f"--- ✅ Resumed from step {global_step}. ---")
            return state, global_step
        except Exception as e:
            self.console.print(f"[bold red]Error loading checkpoint: {e}. Starting fresh.[/bold red]")
            return None, 0
            
    def train(self):
        if self.args.batch_size < 2:
            self.console.print("[bold red]FATAL ERROR: Batch size must be >= 2 for contrastive CLIP loss.[/bold red]")
            sys.exit(1)
        
        tokenized_data_path = Path(self.args.data_dir) / f"pretokenized_{self.args.basename}_{self.args.num_codes}c.npz"
        if tokenized_data_path.exists():
            data = np.load(tokenized_data_path)
            target_indices_np, salience_indices_np = data['target_indices'], data['salience_indices']
            embeddings, latents = data['embeddings'], data['latents']
            num_train_samples = len(embeddings)
        else:
            self.console.print("--- ⚠️ No pre-tokenized data found. Generating now... ---")
            paired_data_path = Path(self.args.data_dir) / f"paired_data_synced_{self.args.basename}.pkl"
            depth_path = Path(self.args.data_dir) / f"depth_maps_{self.args.basename}_{self.args.depth_layers}l.npy"
            alpha_path = Path(self.args.data_dir) / f"alpha_maps_{self.args.basename}.npy"
            if not all(p.exists() for p in [paired_data_path, depth_path, alpha_path]): sys.exit(f"FATAL: Missing data files.")
            with open(paired_data_path, 'rb') as f: data_raw = pickle.load(f)
            latents, embeddings = np.asarray(data_raw['latents']), np.asarray(data_raw['embeddings'])
            depth_maps = np.load(depth_path).astype(np.float32)
            alpha_maps = np.squeeze(np.load(alpha_path).astype(np.float32))
            num_train_samples = len(latents)
            @partial(jit, static_argnames=('latent_grid_size', 'depth_layers'))
            def _preprocess_batch_fn(latents_batch, depth_batch, alpha_batch, latent_grid_size, depth_layers):
                salience_maps = (depth_batch / depth_layers) * 0.5 + (alpha_batch / 255.0) * 0.5
                resized_salience = jax.image.resize(salience_maps[..., None], (salience_maps.shape[0], latent_grid_size, latent_grid_size, 1), 'bilinear')
                salience_for_tok = jnp.repeat(resized_salience, 3, axis=-1)
                return self.tokenizer.apply({'params': self.tok_params}, latents_batch, method=self.tokenizer.encode), \
                       self.tokenizer.apply({'params': self.tok_params}, salience_for_tok, method=self.tokenizer.encode)
            _ = _preprocess_batch_fn(jnp.zeros((2, self.args.latent_grid_size, self.args.latent_grid_size, 3), dtype=jnp.float32), jnp.zeros((2, self.args.latent_grid_size, self.args.latent_grid_size), dtype=jnp.float32), jnp.zeros((2, self.args.latent_grid_size, self.args.latent_grid_size), dtype=jnp.float32), self.args.latent_grid_size, self.args.depth_layers)
            preprocess_batch_size, token_grid_shape = 256, (self.args.latent_grid_size // 4, self.args.latent_grid_size // 4)
            target_indices_np, salience_indices_np = np.zeros((num_train_samples, *token_grid_shape), dtype=np.int32), np.zeros_like(target_indices_np, dtype=np.int32)
            for i in tqdm(range(0, num_train_samples, preprocess_batch_size), desc="Pre-Tokenizing"):
                end_idx = min(i + preprocess_batch_size, num_train_samples)
                chunks = jax.device_put(latents[i:end_idx]), jax.device_put(depth_maps[i:end_idx]), jax.device_put(alpha_maps[i:end_idx])
                target_indices_np[i:end_idx], salience_indices_np[i:end_idx] = jax.device_get(_preprocess_batch_fn(*chunks, self.args.latent_grid_size, self.args.depth_layers))
            np.savez_compressed(tokenized_data_path, target_indices=target_indices_np, salience_indices=salience_indices_np, embeddings=embeddings, latents=latents)
            
        train_ds = tf.data.Dataset.from_tensor_slices((target_indices_np, salience_indices_np, embeddings, latents)).shuffle(num_train_samples, seed=self.args.seed).repeat().batch(self.args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        train_iterator = train_ds.as_numpy_iterator()

        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
        preview_executor = ThreadPoolExecutor(max_workers=1)
        key = jax.random.PRNGKey(self.args.seed)
        
        state_holder = {}
        def lr_schedule_fn(step): q_state = state_holder.get("q_state"); return 0.0 if q_state is None else q_state.current_value
        conductor_optimizer = optax.chain(optax.clip_by_global_norm(1.0), sentinel(), optax.adamw(learning_rate=lr_schedule_fn, b1=0.9, b2=0.95))
        discriminator_optimizer = optax.chain(optax.clip_by_global_norm(1.0), sentinel(), optax.adamw(learning_rate=lr_schedule_fn, b1=0.9, b2=0.95))
        clip_projector_optimizer = optax.adamw(learning_rate=1e-5, b1=0.9, b2=0.95)
        painter_optimizer = optax.adamw(learning_rate=1e-5, b1=0.9, b2=0.95)
        
        ckpt_path = Path(f"manifold_{self.args.basename}_{self.args.num_codes}c_checkpoint.pkl")
        
        optimizers = {'conductor': conductor_optimizer, 'discriminator': discriminator_optimizer, 'clip_projector': clip_projector_optimizer, 'painter': painter_optimizer}
        state, global_step = self._load_checkpoint(ckpt_path, optimizers)
        
        if state is None:
            def merge_params(base_params, pretrained_params):
                merged = base_params.copy()
                for key, value in pretrained_params.items():
                    if key in merged and isinstance(merged.get(key), dict) and isinstance(value, dict): merged[key] = merge_params(merged[key], value)
                    elif key in merged: merged[key] = value
                return merged

            key, cond_key, disc_key, clip_key, painter_key = jax.random.split(key, 5)
            q_controller_state = init_q_controller(Q_CONTROLLER_CONFIG_MANIFOLD)
            H_tok, W_tok = self.args.latent_grid_size // 4, self.args.latent_grid_size // 4
            initialized_p1_params = self.p1_model.init({'params': painter_key}, jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3)), jnp.zeros((1, 2)), jnp.zeros((1, 512)), method=self.p1_model.decode)['params']
            self.p1_params = merge_params(initialized_p1_params, self.p1_params)
            conductor_params = self.conductor_model.init({'params': cond_key}, jnp.zeros((1, H_tok, W_tok), dtype=jnp.int32), jnp.zeros((1, 512), dtype=self.dtype), jnp.zeros((1, H_tok, W_tok), dtype=jnp.int32))['params']
            discriminator_params = self.discriminator_model.init(disc_key, jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), dtype=self.discriminator_model.dtype))['params']
            clip_projector_params = self.clip_projector.init(clip_key, jnp.zeros((1, self.args.d_model), dtype=self.dtype))['params']
            conductor_state = ConductorTrainState.create(apply_fn=self.conductor_model.apply, params=conductor_params, tx=conductor_optimizer, q_state=q_controller_state, ema_params=conductor_params)
            discriminator_state = CustomTrainState.create(apply_fn=self.discriminator_model.apply, params=discriminator_params, tx=discriminator_optimizer)
            clip_projector_state = CustomTrainState.create(apply_fn=self.clip_projector.apply, params=clip_projector_params, tx=clip_projector_optimizer)
            painter_state = CustomTrainState.create(apply_fn=self.p1_model.apply, params=self.p1_params, tx=painter_optimizer)
            state = GANTrainStates(conductor=conductor_state, discriminator=discriminator_state, clip_projector=clip_projector_state, painter=painter_state)
        
        state_holder["q_state"] = state.conductor.q_state

        # --- DEFINITIVE FIX: Define train_step as a closure inside train() ---
        @partial(jit, static_argnames=('train',))
        def train_step(state: GANTrainStates, batch_data, key, lambdas, dampening_factor, train):
            
            patch_render_size, global_render_size = 96, 224
            
            @partial(jit, static_argnames=('patch_render_size', 'global_render_size'))
            def jitted_generate_pixels(painter_params, path_params_reconstructed, batch_patch_coords, global_coords, final_command, patch_render_size, global_render_size):
                B = path_params_reconstructed.shape[0]
                render_patch_vmapped = jax.vmap(lambda pp, coords, cmd: self.p1_model.apply({'params': painter_params}, pp[None, ...], coords, command_vector=cmd[None, ...], method=self.p1_model.decode).squeeze(0))
                rendered_patches = render_patch_vmapped(path_params_reconstructed, batch_patch_coords, final_command)
                rendered_patches = rendered_patches.reshape(B, patch_render_size, patch_render_size, 3)

                rendered_global = self.p1_model.apply({'params': painter_params}, path_params_reconstructed, global_coords, command_vector=final_command, method=self.p1_model.decode)
                rendered_global = rendered_global.reshape(B, global_render_size, global_render_size, 3)
                return rendered_patches, rendered_global

            # --- FIX: `has_aux=True` is not a valid argument for `jit`. Removed it. ---
            @jit
            def jitted_loss_and_path_params(conductor_params, train_key, target_indices, final_command, salience_indices):
                B, H_tok, W_tok = target_indices.shape
                num_to_keep = int(H_tok * W_tok * (1 - self.args.mask_ratio))
                ids_shuffle = jnp.argsort(jax.random.uniform(train_key, (B, H_tok * W_tok)), axis=1)
                ids_keep = ids_shuffle[:, :num_to_keep]
                masked_indices = jnp.full((B, H_tok * W_tok), self.args.num_codes + 1, dtype=jnp.int32)
                visible_indices = jnp.take_along_axis(target_indices.reshape(B, H_tok*W_tok), ids_keep, axis=1)
                input_indices = masked_indices.at[jnp.arange(B)[:, None], ids_keep].set(visible_indices).reshape(B, H_tok, W_tok)
                predicted_logits = self.conductor_model.apply({'params': conductor_params}, input_indices, final_command, salience_indices, train=train)
                loss_mask = jnp.ones((B, H_tok * W_tok)).at[jnp.arange(B)[:, None], ids_keep].set(0)
                mae_loss = (optax.softmax_cross_entropy_with_integer_labels(predicted_logits.reshape(-1, self.args.num_codes), target_indices.reshape(-1)).reshape(B, -1) * loss_mask).sum() / (loss_mask.sum() + 1e-9)
                gumbel_ste = gumbel_softmax_straight_through(predicted_logits, gumbel_key)
                visible_one_hot = jax.nn.one_hot(visible_indices, self.args.num_codes, dtype=self.dtype)
                full_grid_ste = jnp.zeros_like(gumbel_ste).reshape(B, -1, self.args.num_codes).at[jnp.arange(B)[:, None], ids_keep].set(visible_one_hot)
                pred_mask = jnp.ones((B, H_tok * W_tok), dtype=jnp.bool_).at[jnp.arange(B)[:, None], ids_keep].set(False)
                full_grid_ste = jnp.where(pred_mask[..., None], gumbel_ste.reshape(B, -1, self.args.num_codes), full_grid_ste).reshape(B, H_tok, W_tok, -1)
                z_q_ste = jnp.einsum('bhwc,dc->bhwd', full_grid_ste, self.tok_params['vq']['codebook'])
                path_params_reconstructed = self.tokenizer.apply({'params': self.tok_params}, z_q_ste, method=self.tokenizer.decode_from_quantized)
                return mae_loss, path_params_reconstructed
                
            # --- The Bridge ---
            target_indices, salience_indices, command_vector, path_params_grid_real = batch_data
            key, q_key, train_key, perc_key, gumbel_key, render_key = jax.random.split(key, 6)
            q_state = q_controller_choose_action(state.conductor.q_state, q_key)

            is_unconditional = jax.random.uniform(train_key, (self.args.batch_size, 1)) < 0.1
            final_command = jnp.where(is_unconditional, self.uncond_embedding, command_vector)

            # 1. First part of forward pass (JIT)
            mae_loss, path_params_reconstructed = jitted_loss_and_path_params(state.conductor.params, train_key, target_indices, final_command, salience_indices)

            # 2. Render pixels (JIT)
            global_coords = jnp.mgrid[-1:1:global_render_size*1j, -1:1:global_render_size*1j].transpose(1, 2, 0).reshape(-1, 2)
            patch_width = 0.5; rand_centers = jax.random.uniform(render_key, (self.args.batch_size, 2), minval=-(1-patch_width), maxval=(1-patch_width))
            def create_patch_coords(center):
                cx, cy = center[1], center[0]; ax = jnp.linspace(cx - patch_width/2, cx + patch_width/2, patch_render_size); ay = jnp.linspace(cy - patch_width/2, cy + patch_width/2, patch_render_size)
                return jnp.stack(jnp.meshgrid(ax, ay), axis=-1).reshape(-1, 2)
            batch_patch_coords = jax.vmap(create_patch_coords)(rand_centers)
            rendered_patches, rendered_global = jitted_generate_pixels(state.painter.params, path_params_reconstructed, batch_patch_coords, global_coords, final_command, patch_render_size, global_render_size)

            # 3. Get CLIP embeddings in plain Python (outside any JIT)
            patch_clip_input = encode_image_for_clip(rendered_patches, jnp.zeros_like(rendered_patches[..., 0]))
            global_clip_input = encode_image_for_clip(rendered_global, jnp.zeros_like(rendered_global[..., 0]))
            patch_image_embeddings = self.full_clip_model.get_image_features(pixel_values=patch_clip_input, params=self.clip_params)
            global_image_embeddings = self.full_clip_model.get_image_features(pixel_values=global_clip_input, params=self.clip_params)

            # 4. Calculate final loss and gradients (JIT)
            def final_loss_fn(conductor_params, clip_proj_params, painter_params):
                # We need to recompute the path_params inside the grad scope
                _, path_params_reconstructed_grad = jitted_loss_and_path_params(conductor_params, train_key, target_indices, final_command, salience_indices)
                reconstructed_features = self.p1_model.apply({'params': painter_params}, path_params_reconstructed_grad, method=self.p1_model.get_features)
                projected_feature_grid = state.clip_projector.apply_fn({'params': clip_proj_params}, reconstructed_features)
                feature_clip_loss = calculate_compositional_clip_loss(projected_feature_grid, command_vector)
                perceptual_losses = self.perceptual_loss_fn(path_params_reconstructed_grad, path_params_grid_real, perc_key)
                perceptual_loss = jnp.mean(jnp.array(list(perceptual_losses.values())))
                d_fake_logits = state.discriminator.apply_fn({'params': state.discriminator.params}, path_params_reconstructed_grad)
                gan_g_loss = jnp.mean((d_fake_logits - 1.0)**2)
                patch_clip_loss = calculate_clip_similarity_loss(patch_image_embeddings, final_command)
                global_clip_loss = calculate_clip_similarity_loss(global_image_embeddings, final_command)
                total_loss = (lambdas['mae'] * mae_loss + lambdas['gan_g'] * gan_g_loss + lambdas['perceptual'] * perceptual_loss + lambdas['clip'] * feature_clip_loss + lambdas['patch_clip'] * patch_clip_loss + lambdas['global_clip'] * global_clip_loss)
                metrics = {'loss/total_g': total_loss, 'loss/mae': mae_loss, 'loss/gan_g': gan_g_loss, 'loss/perceptual': perceptual_loss, 'loss/clip': feature_clip_loss, 'loss/patch_clip': patch_clip_loss, 'loss/global_clip': global_clip_loss}
                return total_loss, metrics

            grad_fn = jax.value_and_grad(final_loss_fn, argnums=(0, 1, 2), has_aux=True)
            (g_loss, g_metrics), (g_grads, clip_proj_grads, painter_grads) = grad_fn(state.conductor.params, state.clip_projector.params, state.painter.params)
            
            # --- Rest of the step is standard ---
            def discriminator_loss_fn(discriminator_params):
                d_real = state.discriminator.apply_fn({'params': discriminator_params}, path_params_grid_real)
                d_fake = state.discriminator.apply_fn({'params': discriminator_params}, jax.lax.stop_gradient(path_params_reconstructed))
                loss_real, loss_fake = jnp.mean((d_real - 0.9)**2), jnp.mean((d_fake - 0.1)**2)
                return (loss_real + loss_fake) * 0.5, {'loss/total_d': (loss_real + loss_fake) * 0.5}

            (d_loss, d_metrics), d_grads = jax.value_and_grad(discriminator_loss_fn, has_aux=True)(state.discriminator.params)
            
            new_cond_state = state.conductor.replace(q_state=q_state).apply_gradients(grads=g_grads, dampening_factor=dampening_factor)
            new_disc_state = state.discriminator.apply_gradients(grads=d_grads, dampening_factor=dampening_factor)
            new_clip_proj_state = state.clip_projector.apply_gradients(grads=clip_proj_grads)
            new_painter_state = state.painter.apply_gradients(grads=painter_grads)
            
            q_state = q_controller_update(q_state, g_loss)
            new_ema = jax.tree_util.tree_map(lambda ema, p: ema*0.999 + p*(1-0.999), state.conductor.ema_params, new_cond_state.params)
            metrics = {**g_metrics, **d_metrics, 'sentinel_pct': new_cond_state.opt_state[1].dampened_pct, 'lr': q_state.current_value, 'q_status': q_state.status_code}
            
            return GANTrainStates(conductor=new_cond_state.replace(ema_params=new_ema, q_state=q_state), discriminator=new_disc_state, clip_projector=new_clip_proj_state, painter=new_painter_state), metrics

        self.console.print(f"--- JIT Compiling Train Step... ---")
        state_holder["q_state"] = state.conductor.q_state
        jitted_train_step = train_step
        _, _ = jitted_train_step(state, next(train_iterator), key, self.lambdas, jnp.array(1.0, dtype=self.dtype), train=True)
        self.console.print("--- ✅ Compilation Complete. ---")

        steps_per_epoch = num_train_samples // self.args.batch_size
        self.progress = Progress(BarColumn(), TextColumn("{task.description}"))
        epoch_task = self.progress.add_task("Epoch Progress", total=steps_per_epoch)
        start_epoch, step_in_epoch_start = (global_step // steps_per_epoch) + 1, global_step % steps_per_epoch
        active_preview_future, ui_thread = None, None
        
        try:
            with Live(self._generate_layout(), screen=True, redirect_stderr=False, refresh_per_second=20) as live:
                ui_thread = threading.Thread(target=self._ui_refresher_thread, args=(live,), daemon=True)
                ui_thread.start()
                
                for epoch in range(start_epoch, self.args.epochs + 1):
                    self.progress.reset(epoch_task, total=steps_per_epoch, description=f"Epoch {epoch}/{self.args.epochs}", completed=step_in_epoch_start)
                    for step_in_epoch in range(step_in_epoch_start, steps_per_epoch):
                        if self.interactive_state.shutdown_event.is_set(): break
                        batch_data = next(train_iterator)
                        key, step_key = jax.random.split(key)
                        damp_factor_py = self.interactive_state.get_sentinel_factor()
                        current_damp_factor_jnp = jnp.array(damp_factor_py, dtype=self.dtype)
                        self.lambdas = self.pid_controller(self.last_metrics_for_ui)
                        
                        state_holder["q_state"] = state.conductor.q_state
                        
                        state, metrics = jitted_train_step(state, batch_data, step_key, self.lambdas, current_damp_factor_jnp, train=True)
                        
                        with self.ui_lock:
                            self.last_metrics_for_ui = jax.device_get(metrics)
                            self.loss_hist.append(self.last_metrics_for_ui['loss/total_g'])
                            self.sentinel_pct = self.last_metrics_for_ui.get('sentinel_pct', 0.0)
                            q_status_code = int(self.last_metrics_for_ui.get('q_status', 0))
                            self.q_status_str = {0: "WARMUP", 1: "IMPROVING", 2: "STAGNATED", 3: "REGRESSING"}[q_status_code]

                        self.progress.update(epoch_task, advance=1)
                        if global_step > 0 and global_step % self.args.eval_every == 0 and (active_preview_future is None or active_preview_future.done()):
                            key, preview_key = jax.random.split(key)
                            active_preview_future = preview_executor.submit(self._update_preview_task, state, preview_key)
                        
                        if self.interactive_state.get_and_reset_force_save():
                            self._save_checkpoint(state, global_step, ckpt_path)
                        global_step += 1
                    
                    step_in_epoch_start = 0
                    if self.interactive_state.shutdown_event.is_set(): break

                    self.console.log(f"Epoch {epoch} complete. Saving checkpoint...")
                    self._save_checkpoint(state, global_step, ckpt_path)
                
        finally:
            self.interactive_state.shutdown_event.set()
            if ui_thread and ui_thread.is_alive(): ui_thread.join(timeout=1.0)
            preview_executor.shutdown(wait=True)
            self.console.print(f"\n--- Training ended. Saving final checkpoint... ---")
            self._save_checkpoint(state, global_step, ckpt_path)
            self.console.print("✅ Final save complete.")



        
    def _update_preview_task(self, state: GANTrainStates, key: chex.PRNGKey):
        prompt_idx_change = self.interactive_state.get_and_reset_preview_change()
        with self.ui_lock:
            if prompt_idx_change != 0: self.current_preview_prompt_idx = (self.current_preview_prompt_idx + prompt_idx_change) % len(self.validation_prompts)
            command_vector = self.validation_embeddings[self.current_preview_prompt_idx][None, :]
        
        dummy_salience_latents = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3))
        dummy_salience_indices = self.tokenizer.apply({'params': self.tok_params}, dummy_salience_latents, method=self.tokenizer.encode)

        image_batch = _jitted_manifold_inference(
            state.conductor.ema_params, self.conductor_model, self.p1_model, self.tokenizer,
            state.painter.params, self.tok_params,
            command_vector, self.uncond_embedding,
            dummy_salience_indices, key, resolution=128, patch_size=64, num_steps=12,
            grid_size=self.args.latent_grid_size//4, num_codes=self.args.num_codes, guidance_scale=4.0
        )
        
        img_np = np.array(((image_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        if Pixels:
            with self.ui_lock: self.rendered_preview = Pixels.from_image(Image.fromarray(img_np))





# =================================================================================================
# 6. GENERATION & INFERENCE
# =================================================================================================
@partial(jit, static_argnames=('manifold_model', 'p1_model', 'tok_model', 'resolution', 'patch_size', 'num_steps', 'grid_size', 'num_codes'))
def _jitted_manifold_inference(manifold_params, manifold_model, p1_model, tok_model, p1_params, tok_params, command_vector, uncond_vector, salience_indices, key, resolution, patch_size, num_steps, grid_size, num_codes, guidance_scale):
    B = command_vector.shape[0]; H = W = grid_size; MASK_TOKEN_ID = num_codes + 1
    def render(indices):
        path_params_grid = tok_model.apply({'params': tok_params}, indices, method=tok_model.decode)
        coords = jnp.mgrid[-1:1:resolution*1j, -1:1:resolution*1j].transpose(1, 2, 0).reshape(-1, 2)
        coord_chunks = jnp.array_split(coords, (resolution*resolution) // (patch_size*patch_size))
        
        # --- COHERENCE ENHANCEMENT MODIFICATION ---
        # We must pass the text `command_vector` to the painter's decode method.
        # This activates the FiLM conditioning layers in the CoordinateDecoder,
        # giving the painter direct, high-level guidance on what it should be rendering.
        # This dramatically improves object coherence.
        pixel_chunks = [p1_model.apply(
            {'params': p1_params}, 
            path_params_grid, 
            c, 
            command_vector=command_vector,
            method=p1_model.decode
        ) for c in coord_chunks]
        # --- END MODIFICATION ---

        pixels = jnp.concatenate(pixel_chunks, axis=1)
        return pixels.reshape(B, resolution, resolution, 3)

    def loop_body(i, carry):
        indices, known_mask, key = carry
        logits_cond = manifold_model.apply({'params': manifold_params}, indices, command_vector, salience_indices, train=False)
        logits_uncond = manifold_model.apply({'params': manifold_params}, indices, uncond_vector, salience_indices, train=False)
        logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        probs = jax.nn.softmax(logits, axis=-1)
        confidences = jnp.where(known_mask, -1.0, jnp.max(probs, axis=-1))
        
        ratio = (i + 1) / num_steps
        mask_ratio = jnp.cos(ratio * (jnp.pi / 2))
        
        num_unknown = H * W - jnp.sum(known_mask)
        num_to_unmask = jnp.ceil(num_unknown * (1.0 - mask_ratio)).astype(jnp.int32)
        num_to_unmask = jnp.maximum(0, jnp.minimum(num_unknown, num_to_unmask))

        # --- FINAL, ROBUST FIX: Masking with jnp.where ---
        def unmask_branch(operands):
            indices_op, known_mask_op = operands
            
            indices_flat = indices_op.reshape(B, H * W)
            known_mask_flat = known_mask_op.reshape(B, H * W)
            
            # 1. Get indices sorted by confidence
            sorted_confidence_indices = jnp.argsort(confidences.reshape(B, H * W), axis=-1)[:, ::-1]
            
            # 2. Get the model's prediction for each token
            predicted_ids = jnp.argmax(logits, axis=-1).reshape(B, H * W)
            
            # 3. Create a boolean mask for the top k positions to unmask
            mask_range = jnp.arange(H * W)
            update_mask = mask_range < num_to_unmask
            
            # 4. Scatter the boolean `update_mask` to the actual token positions using the sorted indices.
            # `scatter_mask` will be True at the token positions with the highest confidence.
            scatter_mask = jnp.zeros_like(known_mask_flat, dtype=jnp.bool_).at[jnp.arange(B)[:, None], sorted_confidence_indices].set(update_mask)

            # 5. Use jnp.where to conditionally update the indices and known_mask.
            # This avoids non-concrete boolean indexing.
            updated_indices_flat = jnp.where(scatter_mask, predicted_ids, indices_flat)
            new_known_mask_flat = jnp.where(scatter_mask, True, known_mask_flat)

            return updated_indices_flat.reshape(B, H, W), new_known_mask_flat.reshape(B, H, W)

        def noop_branch(operands):
            indices_op, known_mask_op = operands
            return indices_op, known_mask_op

        updated_indices, new_known_mask = jax.lax.cond(
            num_to_unmask > 0,
            unmask_branch,
            noop_branch,
            (indices, known_mask)
        )
        # --- END FIX ---
        
        return updated_indices, new_known_mask, key
        
    initial_indices = jnp.full((B, H, W), MASK_TOKEN_ID, dtype=jnp.int32)
    initial_known_mask = jnp.zeros_like(initial_indices, dtype=jnp.bool_)
    final_indices, _, _ = jax.lax.fori_loop(0, num_steps, loop_body, (initial_indices, initial_known_mask, key))
    return render(final_indices)
 
 
 
 
 
class Generator:
    def __init__(self, args):
        self.args, self.console = args, Console()
        self.console.print("--- 🧠 Loading Manifold Conductor for Generation ---", style="bold green")
        self.dtype = jnp.float32

        _, self.p1_model, _ = ManifoldTrainer._load_phase1(self, args)
        self.tok_params, self.tokenizer, _ = ManifoldTrainer._load_tokenizer(self, args)
        self.conductor_model = ManifoldConductor(num_codes=args.num_codes, d_model=args.d_model_cond, num_heads=args.num_heads, num_layers=args.num_layers, latent_grid_size=args.latent_grid_size, dtype=self.dtype)
        self.clip_projector = CLIPImageProjector(dtype=self.dtype)

        ckpt_path = Path(f"manifold_{args.basename}_{args.num_codes}c_final.pkl")
        if not ckpt_path.exists(): sys.exit(f"FATAL: Checkpoint not found: {ckpt_path}")
        self.console.print(f"-> Loading weights from [green]{ckpt_path}[/green]")
        
        with open(ckpt_path, 'rb') as f:
            ckpt_data = pickle.load(f)
            self.conductor_params = ckpt_data['conductor_ema_params']
            self.p1_params = ckpt_data['p1_params']
            if 'clip_projector_params' not in ckpt_data: sys.exit("FATAL: clip_projector_params not found. Please retrain/save a new model.")
            self.clip_projector_params = ckpt_data['clip_projector_params']
            
        self.uncond_embedding = jnp.zeros((1, 512), dtype=self.dtype)
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        self.console.print("--- 🚀 JIT Compiling Manifold inference pipeline... ---")
        _ = self.generate("a test compile run", 42, 4.0, 12, True, refinement_steps=0) # Compile run
        self.console.print("--- ✅ Compilation Complete ---")

    @partial(jit, static_argnames=('self', 'grid_size'))
    def _get_refined_salience_map(self, draft_image_batch, command_vector, grid_size):
        """JIT-compiled function to perform one step of salience map refinement."""
        # The modulator expects input in the range [-1, 1]
        draft_image_batch_norm = (draft_image_batch * 2.0) - 1.0
        
        # 1. Encode the draft image into the latent manifold
        path_params_grid = self.p1_model.apply({'params': self.p1_params}, draft_image_batch_norm)
        
        # 2. Extract features from this new latent grid
        feature_canvas = self.p1_model.apply({'params': self.p1_params}, path_params_grid, method=self.p1_model.get_features)
        
        # 3. Project features into CLIP-space
        projected_features = self.clip_projector.apply({'params': self.clip_projector_params}, feature_canvas)
        
        # 4. Perform CLIP similarity scan
        patches_norm = projected_features / (jnp.linalg.norm(projected_features, axis=-1, keepdims=True) + 1e-6)
        text_norm = command_vector / (jnp.linalg.norm(command_vector, axis=-1, keepdims=True) + 1e-6)
        similarity_heatmap = jnp.einsum('bhwd,bd->bhw', patches_norm, text_norm[:, None, :])
        
        # 5. Normalize and enhance the heatmap to create the final map
        similarity_heatmap = jnp.squeeze(similarity_heatmap, axis=0)
        h_min, h_max = similarity_heatmap.min(), similarity_heatmap.max()
        normalized_map = (similarity_heatmap - h_min) / (h_max - h_min + 1e-6)
        return jnp.power(normalized_map, 2.5)

    def generate(self, prompt: str, seed: int, guidance_scale: float, decoding_steps: int, _compile_run: bool = False, salience_map: Optional[np.ndarray] = None, output_filename_prefix: str = "MANIFOLD", refinement_steps: int = 3):
        if not _compile_run: self.console.print(f"--- 🎨 Orchestrating shaders for: \"[italic yellow]{prompt}[/italic yellow]\" ---")
        key = jax.random.PRNGKey(seed)
        with torch.no_grad():
            tokens = clip.tokenize([prompt]).to(_clip_device)
            command_vector = self.clip_model.encode_text(tokens).cpu().numpy().astype(self.dtype)
        
        grid_size = self.args.latent_grid_size
        
        if salience_map is not None:
             self.console.print(f"--- Salience: Using user-provided map, skipping refinement. ---")
             final_salience_map = jnp.array(salience_map)
        else:
            # --- ITERATIVE MANIFOLD REFINEMENT LOOP ---
            # Initial salience map is still generated from the prompt as a starting point.
            # We use a generic latent grid for this very first pass.
            self.console.print("--- [Pass 0] Salience: Auto-generating initial map from prompt... ---")
            generic_path_params = jnp.zeros((1, grid_size, grid_size, 3), dtype=self.dtype)
            feature_canvas = self.p1_model.apply({'params': self.p1_params}, generic_path_params, method=self.p1_model.get_features)
            projected_features = self.clip_projector.apply({'params': self.clip_projector_params}, feature_canvas)
            patches_norm = projected_features / (jnp.linalg.norm(projected_features, axis=-1, keepdims=True) + 1e-6)
            text_norm = command_vector / (jnp.linalg.norm(command_vector, axis=-1, keepdims=True) + 1e-6)
            similarity_heatmap = jnp.einsum('bhwd,bd->bhw', patches_norm, text_norm[:, None, :])
            h_min, h_max = similarity_heatmap.min(), similarity_heatmap.max()
            normalized_map = (similarity_heatmap - h_min) / (h_max - h_min + 1e-6)
            current_salience_map = jnp.power(jnp.squeeze(normalized_map), 2.5)

            for i in range(refinement_steps):
                self.console.print(f"--- [Pass {i+1}/{refinement_steps}] Refining composition... ---")
                key, gen_key = jax.random.split(key)
                
                # Tokenize the current salience map
                salience_latents = jnp.repeat(current_salience_map[None, ..., None], 3, axis=-1)
                salience_indices = self.tokenizer.apply({'params': self.tok_params}, salience_latents, method=self.tokenizer.encode)

                # Generate a low-resolution draft image (e.g., 256x256) for speed
                draft_image_batch = _jitted_manifold_inference(
                    self.conductor_params, self.conductor_model, self.p1_model, self.tokenizer, 
                    self.p1_params, self.tok_params, command_vector, self.uncond_embedding, 
                    salience_indices, gen_key, resolution=256, patch_size=128, num_steps=decoding_steps, 
                    grid_size=self.args.latent_grid_size//4, num_codes=self.args.num_codes, guidance_scale=guidance_scale
                )
                
                # The output image is [-1, 1], we need [0, 1] for the refinement encoder
                draft_image_0_1 = draft_image_batch * 0.5 + 0.5
                
                # Refine the salience map based on this new draft
                current_salience_map = self._get_refined_salience_map(draft_image_0_1, command_vector, grid_size)
            
            final_salience_map = current_salience_map
        
        # --- FINAL RENDER ---
        self.console.print("--- [Final Pass] Rendering full resolution image... ---")
        salience_latents = jnp.repeat(final_salience_map[None, ..., None], 3, axis=-1)
        salience_indices = self.tokenizer.apply({'params': self.tok_params}, salience_latents, method=self.tokenizer.encode)

        image_batch = _jitted_manifold_inference(
            self.conductor_params, self.conductor_model, self.p1_model, self.tokenizer, 
            self.p1_params, self.tok_params, command_vector, self.uncond_embedding, 
            salience_indices, key, resolution=512, patch_size=256, num_steps=decoding_steps, 
            grid_size=self.args.latent_grid_size//4, num_codes=self.args.num_codes, guidance_scale=guidance_scale
        )
        
        image_batch.block_until_ready()
        if _compile_run: return
        
        img_np = np.array(((image_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        filename = f"{output_filename_prefix}_{Path(self.args.basename).stem}_{prompt.replace(' ', '_')[:40]}_{seed}.png"
        Image.fromarray(img_np).save(filename); self.console.print(f"✅ Image saved to [green]{filename}[/green]")

# =================================================================================================
# 7. MAIN EXECUTION BLOCK (FINAL)
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 3: Manifold Conductor", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--basename', type=str, required=True, help="Base name for the model set.")
    
    manifold_base_parser = argparse.ArgumentParser(add_help=False, parents=[base_parser])
    manifold_base_parser.add_argument('--d-model', type=int, required=True, help="d_model of Phase 1 AE.")
    manifold_base_parser.add_argument('--latent-grid-size', type=int, required=True, help="Latent grid size of Phase 1 AE.")
    manifold_base_parser.add_argument('--num-codes', type=int, default=4096)
    manifold_base_parser.add_argument('--d-model-cond', type=int, default=1024, help="d_model for the Conductor.")
    manifold_base_parser.add_argument('--num-layers', type=int, default=16)
    manifold_base_parser.add_argument('--num-heads', type=int, default=16)
    
    p_manifold = subparsers.add_parser("train-manifold", help="Train the Manifold Conductor with the final objective.", parents=[manifold_base_parser])
    p_manifold.add_argument('--data-dir', type=str, required=True)
    p_manifold.add_argument('--epochs', type=int, default=200)
    p_manifold.add_argument('--batch-size', type=int, default=8)
    p_manifold.add_argument('--eval-every', type=int, default=500)
    p_manifold.add_argument('--use-bfloat16', action='store_true')
    p_manifold.add_argument('--seed', type=int, default=42)
    p_manifold.add_argument('--mask-ratio', type=float, default=0.75)
    p_manifold.add_argument('--depth-layers', type=int, required=True)

    p_gen = subparsers.add_parser("generate", help="Generate an image using the Manifold Conductor.", parents=[manifold_base_parser])
    p_gen.add_argument('--prompt', type=str, required=True); p_gen.add_argument('--seed', type=int, default=lambda: int(time.time()))
    p_gen.add_argument('--guidance-scale', type=float, default=4.0); p_gen.add_argument('--decoding-steps', type=int, default=16)

    args = parser.parse_args()
    if args.command == "generate": args.seed = args.seed() if callable(args.seed) else args.seed

    if args.command == "train-manifold": ManifoldTrainer(args).train()
    elif args.command == "generate": Generator(args).generate(args.prompt, args.seed, args.guidance_scale, args.decoding_steps)
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console = Console()
        console.print("\n[bold red]FATAL ERROR ENCOUNTERED[/bold red]")
        console.print_exception(show_locals=False)
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write(traceback.format_exc())
        console.print("\n[yellow]Full traceback written to [bold]crash_log.txt[/bold][/yellow]")
