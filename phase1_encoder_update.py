# =================================================================================================
#
#                    PHASE 1: ON-THE-FLY, HYBRID-LOSS AUTOENCODER
#
# (Final architecture using just-in-time data processing, eliminating the need for TFRecords.
#  This is the most robust, flexible, and efficient version.)
#
# =================================================================================================

import os
# --- Environment Setup for JAX/TensorFlow ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
# --- JAX Configuration ---
from pathlib import Path
import platform
import atexit

# --- Robust and Explicit Cache Setup ---
try:
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / ".jax_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at: {cache_dir} ---")
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

    def _jax_shutdown():
        print("\n--- Script ending. Waiting for JAX to finalize cache... ---")
        import jax
        jax.clear_caches()
        print("--- JAX cache finalized. ---")
    atexit.register(_jax_shutdown)

except NameError:
    cache_dir = Path.home() / ".jax_cache_global"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at (fallback global): {cache_dir} ---")

# --- Core Imports ---
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, common_utils
from flax.jax_utils import replicate, unreplicate
from flax import struct
import optax
import numpy as np
import pickle
import time
from typing import Any, Sequence, Dict, Tuple, Optional
import sys
import argparse
import signal
import threading
from functools import partial
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import jax.scipy.signal
import imageio
from dataclasses import dataclass
from jax import jit

# --- Dependency Checks and Imports ---
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
    from rich.text import Text
    import pynvml; pynvml.nvmlInit()
    from tqdm import tqdm
    import chex
except ImportError as e:
    print(f"[FATAL] A required dependency is missing. Please install it. Error: {e}")
    sys.exit(1)

try:
    from rich_pixels import Pixels
except ImportError:
    print("[Warning] `rich-pixels` not found. Visual preview in GUI will be disabled. Run: pip install rich-pixels")
    Pixels = None

CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_threefry_partitionable', True)


# =================================================================================================
# 1. ADVANCED TRAINING TOOLKIT
# =================================================================================================
class InteractivityState:
    """Manages user input during training for interactivity without pausing the loop."""
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_index_change = 0
        self.shutdown_event, self.force_save = threading.Event(), False

    def get_and_reset_preview_change(self):
        with self.lock: change = self.preview_index_change; self.preview_index_change = 0; return change
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def set_shutdown(self): self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    """A thread to listen for keyboard commands."""
    print("--- Key listener started. Controls: [←/→] Preview | [s] Force Save | [q] Quit ---")
    if platform.system() == "Windows": import msvcrt # type: ignore
    else: import select, sys, tty, termios; fd, old_settings = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    try:
        if platform.system() != "Windows": tty.setcbreak(sys.stdin.fileno())
        while not shared_state.shutdown_event.is_set():
            if platform.system() == "Windows":
                if msvcrt.kbhit(): key = msvcrt.getch()
                else: time.sleep(0.05); continue
            else:
                if select.select([sys.stdin], [], [], 0.05)[0]: key = sys.stdin.read(1)
                else: continue
            with shared_state.lock:
                if key in [b'q', 'q', b'\x03', '\x03']: shared_state.set_shutdown(); break
                elif key in [b's', 's']: shared_state.force_save = True
                elif key == b'\xe0' or key == '\x1b': # Arrow keys
                    arrow = msvcrt.getch() if platform.system() == "Windows" else sys.stdin.read(2)
                    if arrow in [b'K', '[D']: shared_state.preview_index_change = -1
                    elif arrow in [b'M', '[C']: shared_state.preview_index_change = 1
    finally:
        if platform.system() != "Windows": termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

class PIDControllerState(struct.PyTreeNode):
    integral_error: chex.Array; last_error: chex.Array; step_count: chex.Array

@struct.dataclass
class PIDControllerConfig:
    loss_names: Tuple[str, ...]=struct.field(pytree_node=False); base_weights: Tuple[float, ...]=struct.field(pytree_node=False)
    targets: Sequence[Tuple[str, float]]=struct.field(pytree_node=False); gains: Sequence[Tuple[str, Tuple[float, float, float]]]=struct.field(pytree_node=False)
    warmup_steps: int=struct.field(pytree_node=False, default=500); integral_clip: Tuple[float, float]=struct.field(pytree_node=False, default=(-5.0, 5.0))
    lambda_clip: Tuple[float, float]=struct.field(pytree_node=False, default=(0.1, 20.0))

def init_pid_controller(config: PIDControllerConfig) -> PIDControllerState:
    num_losses = len(config.loss_names)
    return PIDControllerState(integral_error=jnp.zeros((num_losses,)), last_error=jnp.zeros((num_losses,)), step_count=jnp.array(0))

@partial(jax.jit, static_argnames=('config',))
def pid_controller_update(state: PIDControllerState, current_metrics: chex.Array, config: PIDControllerConfig) -> Tuple[PIDControllerState, chex.Array]:
    targets_dict=dict(config.targets); gains_dict=dict(config.gains)
    base_weights=jnp.array(config.base_weights); targets=jnp.array([targets_dict.get(n,0.0) for n in config.loss_names])
    gains_kp=jnp.array([gains_dict.get(n,(0,0,0))[0] for n in config.loss_names]); gains_ki=jnp.array([gains_dict.get(n,(0,0,0))[1] for n in config.loss_names]); gains_kd=jnp.array([gains_dict.get(n,(0,0,0))[2] for n in config.loss_names])
    error = current_metrics - targets
    integral_error = jnp.clip(state.integral_error + error, config.integral_clip[0], config.integral_clip[1])
    derivative = error - state.last_error
    adjustment = (gains_kp * error) + (gains_ki * integral_error) + (gains_kd * derivative)
    # --- PATCH: Clip adjustment to prevent jnp.exp from overflowing to inf ---
    adjustment = jnp.clip(adjustment, -20.0, 20.0)
    multiplier = jnp.exp(adjustment)
    calculated_lambdas = base_weights * multiplier
    final_lambdas = jnp.clip(calculated_lambdas, config.lambda_clip[0], config.lambda_clip[1])
    new_state = state.replace(integral_error=integral_error, last_error=error, step_count=state.step_count + 1)
    is_warmed_up = new_state.step_count > config.warmup_steps
    final_weights = jnp.where(is_warmed_up, final_lambdas, base_weights)
    return new_state, final_weights

class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; current_lr: jnp.ndarray
    exploration_rate: jnp.ndarray; step_count: jnp.ndarray; last_action_idx: jnp.ndarray; status_code: jnp.ndarray

@dataclass(frozen=True)
class QControllerConfig:
    num_lr_actions: int = 5; lr_change_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1; lr_min: float = 1e-6; lr_max: float = 1e-3
    metric_history_len: int = 100; exploration_rate_q: float = 0.3; min_exploration_rate: float = 0.05; exploration_decay: float = 0.9998
    warmup_steps: int = 500; warmup_lr_start: float = 1e-6

def init_q_controller(config: QControllerConfig) -> QControllerState:
    return QControllerState(q_table=jnp.zeros(config.num_lr_actions), metric_history=jnp.zeros(config.metric_history_len),
                            current_lr=jnp.array(config.warmup_lr_start), exploration_rate=jnp.array(config.exploration_rate_q),
                            step_count=jnp.array(0), last_action_idx=jnp.array(-1), status_code=jnp.array(0))

@partial(jax.jit, static_argnames=('config', 'target_lr'))
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey, config: QControllerConfig, target_lr: float) -> QControllerState:
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config.warmup_steps
        lr = config.warmup_lr_start * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=lr, step_count=state.step_count + 1, status_code=jnp.array(0)) # 0: WARMUP
    def regular_action():
        explore, act = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore) < state.exploration_rate,
            lambda: jax.random.randint(act, (), 0, config.num_lr_actions),
            lambda: jnp.argmax(state.q_table))
        new_lr = jnp.clip(state.current_lr * jnp.array(config.lr_change_factors)[action_idx], config.lr_min, config.lr_max)
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config.warmup_steps, warmup_action, regular_action)

@partial(jax.jit, static_argnames=('config',))
def q_controller_update(state: QControllerState, metric_value: chex.Array, config: QControllerConfig) -> QControllerState:
    new_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value); st = state.replace(metric_history=new_history)
    def perform_q_update(s: QControllerState) -> QControllerState:
        reward = -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 10, 10))
        is_improving = reward > -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 20, 10))
        status = jax.lax.select(is_improving, jnp.array(1), jnp.array(2)) # 1: IMPROVING, 2: STAGNATED
        old_q = s.q_table[s.last_action_idx]
        new_q = old_q + config.learning_rate_q * (reward - old_q)
        return s.replace(q_table=s.q_table.at[s.last_action_idx].set(new_q),
                         exploration_rate=jnp.maximum(config.min_exploration_rate, s.exploration_rate * config.exploration_decay),
                         status_code=status)
    can_update = (st.step_count > config.warmup_steps) & (st.last_action_idx >= 0)
    return jax.lax.cond(can_update, perform_q_update, lambda s: s, st)


class CustomTrainState(train_state.TrainState):
    """A train state that also tracks EMA parameters, PID controller, and Q-controller state."""
    ema_params: Any
    pid_controller_state: PIDControllerState
    q_controller_state: QControllerState
    # Add this type hint for clarity
    opt_state: optax.OptState
    
    
@partial(jax.jit)
def rgb_to_ycber(image: jnp.ndarray) -> jnp.ndarray:
    rgb_01=(image+1.0)/2.0; r,g,b=rgb_01[...,0],rgb_01[...,1],rgb_01[...,2]
    y=0.299*r+0.587*g+0.114*b; cb=-0.168736*r-0.331264*g+0.5*b; cr=0.5*r-0.418688*g-0.081312*b
    return jnp.stack([y*2.0-1.0,cb*2.0,cr*2.0],axis=-1)

@partial(jax.jit)
def ycber_to_rgb(image: jnp.ndarray) -> jnp.ndarray:
    y_scaled,cb_scaled,cr_scaled=image[...,0],image[...,1],image[...,2]
    y=(y_scaled+1.0)/2.0; cb=cb_scaled/2.0; cr=cr_scaled/2.0
    r=y+1.402*cr; g=y-0.344136*cb-0.714136*cr; b=y+1.772*cb
    return jnp.clip(jnp.stack([r,g,b],axis=-1),0.0,1.0)*2.0-1.0

# =================================================================================================
# 2. MODEL DEFINITIONS
# =================================================================================================

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2); imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images_ycber: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, features, current_dim, i = images_ycber, 32, self.input_image_size, 0
        context_vectors = []
        while (current_dim // 2) >= self.latent_grid_size and (current_dim // 2) > 0:
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            context_vectors.append(jnp.mean(x, axis=(1, 2)))
            features *= 2; current_dim //= 2; i += 1
        context_vector = jnp.concatenate(context_vectors, axis=-1)
        if current_dim != self.latent_grid_size:
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        def create_head(name: str, input_features: jnp.ndarray):
            h = nn.Conv(128, (1, 1), name=f"{name}_head_conv1", dtype=self.dtype)(input_features); h = nn.gelu(h)
            def radius_bias_init_head(key, shape, dtype=jnp.float32):
                return jnp.zeros(shape, dtype).at[2].set(-1.0)
            params_raw = nn.Conv(3, (1, 1), name=f"{name}_head_out", dtype=self.dtype, bias_init=radius_bias_init_head)(h)
            delta = nn.tanh(params_raw[..., 0]) * jnp.pi
            chi   = nn.tanh(params_raw[..., 1]) * (jnp.pi / 4.0)
            radius = nn.sigmoid(params_raw[..., 2]) * (jnp.pi / 2.0)
            return delta, chi, radius
        delta_y, chi_y, radius_y = create_head("y_luma", x)
        delta_cb, chi_cb, radius_cb = create_head("cb_chroma", x)
        delta_cr, chi_cr, radius_cr = create_head("cr_chroma", x)
        path_params = jnp.stack([
            delta_y, chi_y, radius_y,
            delta_cb, chi_cb, radius_cb,
            delta_cr, chi_cr, radius_cr
        ], axis=-1)
        return path_params, context_vector

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = path_params_grid.shape
        path_params = path_params_grid.reshape(B, H * W, 9)
        def get_channel_stats(delta_c, chi_c, radius):
            theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
            delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
            chi_path   = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
            t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path) + 1e-8
            phase_steps = jnp.angle(t_co_steps); amplitude_steps = jnp.abs(t_co_steps)
            safe_std = jnp.sqrt(jnp.maximum(0., jnp.var(amplitude_steps, axis=-1)))
            return jnp.stack([jnp.mean(phase_steps, axis=-1), jnp.ptp(phase_steps, axis=-1), jnp.mean(amplitude_steps, axis=-1), safe_std, radius], axis=-1)
        stats = jnp.concatenate([
            get_channel_stats(path_params[..., 0], path_params[..., 1], path_params[..., 2]),
            get_channel_stats(path_params[..., 3], path_params[..., 4], path_params[..., 5]),
            get_channel_stats(path_params[..., 6], path_params[..., 7], path_params[..., 8])
        ], axis=-1)
        feature_grid = nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(stats)
        return feature_grid.reshape(B, H, W, self.d_model)

class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)

class FiLMLayer(nn.Module):
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x, context):
        context_proj = nn.Dense(x.shape[-1] * 2, name="context_proj", dtype=self.dtype)(context)
        gamma, beta = context_proj[..., :x.shape[-1]], context_proj[..., x.shape[-1]:]
        return x * (gamma[:, None, :] + 1) + beta[:, None, :]

class CoordinateDecoder(nn.Module):
    d_model: int; num_freqs: int = 10; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.remat
    def _mlp_block(self, h: jnp.ndarray, context_vector: jnp.ndarray) -> jnp.ndarray:
        film_layer = FiLMLayer(dtype=self.dtype)
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = film_layer(h, context_vector)
            h = nn.gelu(h)
        return nn.Dense(3, name="mlp_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(h)
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray, oracle_pixels: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = feature_grid.shape
        encoded_coords = PositionalEncoding(self.num_freqs, dtype=self.dtype)(coords)
        pyramid = [feature_grid] + [jax.image.resize(feature_grid, (B, H//(2**i), W//(2**i), self.d_model), 'bilinear') for i in range(1, 3)]
        all_sampled_features = []
        for level_grid in pyramid:
            level_shape = jnp.array(level_grid.shape[1:3], dtype=self.dtype)
            coords_rescaled = (coords + 1) / 2 * (level_shape - 1)
            def sample_one_image_level(single_level_grid):
                grid_chw = single_level_grid.transpose(2, 0, 1)
                return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw).T
            all_sampled_features.append(jax.vmap(sample_one_image_level)(level_grid))
        concatenated_features = jnp.concatenate(all_sampled_features, axis=-1)
        encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)
        mlp_input = jnp.concatenate([encoded_coords_tiled, concatenated_features, oracle_pixels], axis=-1)
        output = self._mlp_block(mlp_input, context_vector)
        return nn.tanh(output)

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    cheat_prob: float = 0.9
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(self.d_model, name="coord_decoder", dtype=self.dtype)
    def __call__(self, images_rgb, coords, rngs=None):
        path_params, context_vector = self.encode(images_rgb)
        coords_rescaled = (coords + 1) / 2 * (jnp.array(images_rgb.shape[1:3], dtype=self.dtype) - 1)
        def sample_oracle_pixels(single_image):
             return jax.vmap(lambda c: jax.scipy.ndimage.map_coordinates(c, coords_rescaled.T, order=1, mode='reflect'))(single_image.transpose(2,0,1)).T
        oracle_pixels = jax.vmap(sample_oracle_pixels)(images_rgb)
        if self.is_mutable_collection('dropout'):
            dropout_rng = self.make_rng('dropout') if rngs is None or 'dropout' not in rngs else rngs['dropout']
            dropout_mask = jax.random.bernoulli(dropout_rng, 1.0 - self.cheat_prob, oracle_pixels.shape)
            oracle_pixels = jnp.where(dropout_mask, oracle_pixels, jnp.zeros_like(oracle_pixels))
        pixels_ycber = self.coord_decoder(self.observer(path_params), context_vector, coords, oracle_pixels)
        return ycber_to_rgb(pixels_ycber)
    def encode(self, images_rgb):
        return self.modulator(rgb_to_ycber(images_rgb))

# =================================================================================================
# 3. DATA HANDLING (ON-THE-FLY) - CORRECTED VERSION
# =================================================================================================

def create_on_the_fly_dataset(image_dir: str, image_size: int, is_training: bool):
    """
    Creates an efficient on-the-fly TensorFlow dataset pipeline.

    This corrected version caches only the lightweight image file paths, not the
    full-sized decoded images, to prevent the memory leak observed during training.
    """
    base_path = Path(image_dir).resolve()
    path_cache_file = base_path / "image_path_cache.pkl"

    if path_cache_file.exists():
        print(f"--- Found path cache file. Loading file list from: {path_cache_file} ---")
        with open(path_cache_file, 'rb') as f: image_paths = pickle.load(f)
    else:
        print(f"--- No path cache found. Scanning directory: {base_path} ---")
        print("    (This will be slow on the first run, but fast afterwards.)")
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        image_paths = [str(p) for p in base_path.rglob('*') if p.suffix.lower() in image_extensions]
        if image_paths:
            print(f"--- Found {len(image_paths)} images. Saving path list to cache... ---")
            with open(path_cache_file, 'wb') as f: pickle.dump(image_paths, f)

    if not image_paths:
        print(f"[FATAL] No images found in '{base_path}' or its cache.")
        sys.exit(1)

    num_samples = len(image_paths)
    print(f"--- Creating dataset with {num_samples} images. ---")

    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    # --- MEMORY LEAK FIX ---
    # The key change is the order of operations. We now cache the lightweight
    # file paths *before* mapping the expensive image loading function.

    if is_training:
        # 1. Cache the list of file paths. This is very small and fits in RAM.
        ds = ds.cache()
        # 2. Repeat the dataset of file paths indefinitely for multiple epochs.
        ds = ds.repeat()
        # 3. Shuffle the file paths. A large buffer ensures good randomness across epochs.
        #    Using min() prevents creating a buffer larger than the dataset itself.
        #    reshuffle_each_iteration=True is the default and desired behavior.
        shuffle_buffer = min(num_samples, 20000)
        print(f"--- Using shuffle buffer of {shuffle_buffer} for training. ---")
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    else:
        # For validation, we don't need to shuffle or repeat, but caching the
        # file paths still speeds up subsequent validation runs.
        ds = ds.cache()

    @tf.function
    def _process_path(file_path):
        img = tf.io.decode_image(tf.io.read_file(file_path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        # Lanczos3 is high quality but can be slow; consider BILINEAR for speed if needed.
        resized_img = tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.LANCZOS3)
        return tf.cast(resized_img, tf.float32) / 127.5 - 1.0

    # 4. Map the image loading function *after* caching and shuffling.
    # This ensures that images are loaded from disk on-the-fly. AUTOTUNE
    # will parallelize this to keep the GPU fed.
    ds = ds.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)

    # The .batch() and .prefetch() calls will be applied later in the train() method.
    # The problematic .cache() call after .map() has been removed.

    return ds, num_samples








# =================================================================================================
# 4. TRAINER CLASS
# =================================================================================================
class ImageTrainer:
    def __init__(self, args):
        self.args = args; self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=self.dtype, cheat_prob=args.cheat_prob)
        self.interactive_state = InteractivityState(); self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.num_devices = jax.local_device_count(); self.ui_lock = threading.Lock()
        self.last_metrics = {}; self.current_lambdas = {}; self.current_preview_np, self.current_recon_np = None, None
        self.param_count = 0; self.loss_hist = deque(maxlen=200); self.steps_per_sec = 0.0
        self.preview_images_device, self.rendered_original_preview, self.rendered_recon_preview = None, None, None
        self.is_validating, self.validation_progress = False, None
        self.current_q_lr = 0.0; self.current_q_status = 0

    def _ssim_loss(self, img1, img2, C1=0.01**2, C2=0.03**2):
        mu1=nn.avg_pool(img1,(8,8),(1,1),'VALID'); mu2=nn.avg_pool(img2,(8,8),(1,1),'VALID')
        mu1_sq,mu2_sq,mu1_mu2=mu1**2,mu2**2,mu1*mu2
        sigma1_sq=nn.avg_pool(img1**2,(8,8),(1,1),'VALID')-mu1_sq; sigma2_sq=nn.avg_pool(img2**2,(8,8),(1,1),'VALID')-mu2_sq
        sigma12=nn.avg_pool(img1*img2,(8,8),(1,1),'VALID')-mu1_mu2
        ssim_map=((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        return 1.0 - jnp.mean(ssim_map)

    def _ms_gdl_loss(self, y_true, y_pred, scales=4):
        def _gradient_diff(true, pred):
            return jnp.mean(jnp.abs((true[:,1:,:]-true[:,:-1,:])-(pred[:,1:,:]-pred[:,:-1,:]))) + jnp.mean(jnp.abs((true[:,:,1:]-true[:,:,:-1])-(pred[:,:,1:]-pred[:,:,:-1])))
        loss = 0.0
        for i in range(scales):
            loss += _gradient_diff(y_true, y_pred)
            if i < scales-1:
                y_true=jax.image.resize(y_true,(y_true.shape[0],y_true.shape[1]//2,y_true.shape[2]//2,y_true.shape[3]),"bilinear")
                y_pred=jax.image.resize(y_pred,(y_pred.shape[0],y_pred.shape[1]//2,y_pred.shape[2]//2,y_pred.shape[3]),"bilinear")
        return loss / scales

    def _laplacian_pyramid_loss(self, y_true, y_pred, levels=3):
        loss = 0.0
        for i in range(levels):
            h,w=y_true.shape[1],y_true.shape[2]
            down_true=jax.image.resize(y_true,(y_true.shape[0],h//2,w//2,y_true.shape[3]),'linear'); down_pred=jax.image.resize(y_pred,(y_pred.shape[0],h//2,w//2,y_pred.shape[3]),'linear')
            up_true=jax.image.resize(down_true,(down_true.shape[0],h,w,y_true.shape[3]),'linear'); up_pred=jax.image.resize(down_pred,(down_pred.shape[0],h,w,y_pred.shape[3]),'linear')
            loss+=jnp.mean(jnp.abs((y_true-up_true)-(y_pred-up_pred)))
            y_true,y_pred=down_true,down_pred
        return loss+jnp.mean(jnp.abs(y_true-y_pred))

    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB",f"{u.gpu}%"
        except Exception: return "N/A","N/A"

    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " "*w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

    def _save_checkpoint(self, state, epoch, global_step, best_val_loss, path):
        unrep_state = unreplicate(state)
        data = {'epoch': epoch, 'global_step': global_step, 'best_val_loss': best_val_loss,
                'params': jax.device_get(unrep_state.params), 'ema_params': jax.device_get(unrep_state.ema_params),
                'opt_state': jax.device_get(unrep_state.opt_state), 'pid_controller_state': jax.device_get(unrep_state.pid_controller_state)}
        with open(path, 'wb') as f: pickle.dump(data, f)
        console = Console(); console.print(f"\n--- 💾 Checkpoint saved for epoch {epoch+1} / step {global_step} ---")

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            layout = Layout(name="root")
            bottom_widgets = [self.progress]
            if self.is_validating and self.validation_progress: bottom_widgets.append(self.validation_progress)
            layout.split(Layout(name="header",size=3), Layout(ratio=1,name="main"), Layout(Group(*bottom_widgets),name="footer",size=5))
            layout["main"].split_row(Layout(name="left",minimum_size=60), Layout(name="right",ratio=1))
            precision = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"🧠⚡ [bold]Topological AE (On-the-Fly)[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision}"
            layout["header"].update(Panel(Align.center(header_text),style="bold magenta",title="[dim]wubumind.ai[/dim]",title_align="right"))
            stats_tbl = Table.grid(expand=True,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
            mem,util=self._get_gpu_stats()
            q_status_map = {0: "[cyan]Warmup[/]", 1: "[green]Improving[/]", 2: "[yellow]Stagnated[/]"}
            q_status_text = q_status_map.get(self.current_q_status, "[red]Unknown[/]")
            stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🚀")
            stats_tbl.add_row("Learning Rate", f"[green]{self.current_q_lr:.2e}[/]")
            stats_tbl.add_row("LR Controller", q_status_text)
            stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            loss_table = Table(show_header=False,box=None); loss_table.add_column(style="cyan",width=10); loss_table.add_column(justify="right",style="white",width=10); loss_table.add_column(justify="right",style="yellow")
            loss_table.add_row("[bold]Metric[/bold]","[bold]Value[/bold]","[bold]λ (PID)[/bold]")
            for name in self.pid_config.loss_names:
                value=self.last_metrics.get(name); formatted_value=f"{float(value):.4f}" if value is not None else "---"
                loss_table.add_row(name.replace('_','-').title(),formatted_value,f"{self.current_lambdas.get(name,0.0):.2f}")
            loss_panel=Panel(loss_table,title="[bold]📉 Hybrid Loss[/]",border_style="cyan")
            layout["left"].update(Group(Panel(stats_tbl,title="[bold]📊 Core Stats[/]",border_style="blue"),loss_panel))
            spark_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_hist,60)}[/]"),title="Total Loss History",height=3,border_style="cyan")
            preview_content=Text("...",justify="center")
            if self.rendered_original_preview:
                recon_panel=self.rendered_recon_preview or Text("[dim]Waiting...[/dim]",justify="center")
                prev_tbl=Table.grid(expand=True,padding=(0,1)); prev_tbl.add_column(); prev_tbl.add_column()
                prev_tbl.add_row(Panel(self.rendered_original_preview,title="Original 📸",border_style="dim"), Panel(recon_panel,title="Reconstruction ✨",border_style="dim"))
                preview_content=prev_tbl
            right_panel_group=Group(spark_panel,Panel(preview_content,title="[bold]🖼️ Live Preview (←/→)[/]",border_style="green"))
            layout["right"].update(right_panel_group)
            return layout

    @partial(jit, static_argnames=('self', 'resolution'))
    def generate_preview(self, gen_ema_params, preview_image_batch, resolution=128):
        coords = jnp.mgrid[-1:1:resolution*1j, -1:1:resolution*1j].transpose(1, 2, 0).reshape(-1, 2)
        recon_pixels_flat = self.model.clone(cheat_prob=0.0).apply({'params': gen_ema_params}, preview_image_batch, coords, rngs={'dropout': jax.random.PRNGKey(0)})
        return recon_pixels_flat.reshape(preview_image_batch.shape[0], resolution, resolution, 3)

    def _update_preview_task(self, gen_ema_params, preview_image_batch):
        recon_batch = self.generate_preview(gen_ema_params, preview_image_batch)
        recon_batch.block_until_ready()
        recon_np = np.array(((recon_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        with self.ui_lock:
            self.current_recon_np = recon_np
            if Pixels:
                term_w=64; h,w,_=self.current_recon_np.shape; term_h=int(term_w*(h/w)*0.5)
                self.rendered_recon_preview=Pixels.from_image(Image.fromarray(self.current_recon_np).resize((term_w,term_h),Image.LANCZOS))

    def train(self):
            console = Console()
            key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
            console.print("--- Setting up ON-THE-FLY data pipeline... ---")
            dataset, num_samples = create_on_the_fly_dataset(self.args.data_dir, self.args.image_size, is_training=True)
            val_dataset, _ = create_on_the_fly_dataset(self.args.data_dir, self.args.image_size, is_training=False)
            REBATCH_SIZE = 100
            dataset = dataset.batch(self.args.batch_size*self.num_devices*REBATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            preview_dataset = val_dataset.batch(50).prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset.batch(self.args.batch_size * self.num_devices, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            train_iterator = iter(tfds.as_numpy(dataset))
            console.print("--- Pre-loading data for asynchronous previews... ---")
            preview_batch_tf = next(iter(preview_dataset.take(1)), None)
            if preview_batch_tf is None: console.print("[bold red]FATAL: Validation dataset is empty.[/bold red]"); sys.exit(1)
            preview_buffer_host = np.array(preview_batch_tf)
            self.preview_images_device = jax.device_put(preview_buffer_host); current_preview_idx = 0
            self.current_preview_np = ((preview_buffer_host[current_preview_idx] * 0.5 + 0.5) * 255).astype(np.uint8)
            if Pixels:
                term_w=64; h,w,_=self.current_preview_np.shape; term_h=int(term_w*(h/w)*0.5)
                self.rendered_original_preview = Pixels.from_image(Image.fromarray(self.current_preview_np).resize((term_w, term_h), Image.LANCZOS))
            console.print(f"--- 🚀 Performance Mode: Rebatching {REBATCH_SIZE} steps per data load. ---")
            steps_per_epoch = num_samples // (self.args.batch_size * self.num_devices) if self.args.batch_size*self.num_devices>0 else 0
            total_steps = self.args.epochs * steps_per_epoch if steps_per_epoch > 0 else 0
            console.print(f"📈 Training for {self.args.epochs} epochs ({total_steps} total steps).")
            p_full_coords = replicate(jnp.mgrid[-1:1:self.args.image_size*1j, -1:1:self.args.image_size*1j].transpose(1, 2, 0).reshape(-1, 2))
            train_key = jax.random.PRNGKey(self.args.seed)
            optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr, b1=0.9, b2=0.95))
            loss_names = ('l1_img', 'ssim', 'ms_gdl', 'laplacian')
            self.pid_config = PIDControllerConfig(loss_names=loss_names, base_weights=(10.0, 5.0, 2.0, 1.5), targets=(('l1_img',0.08),('ssim',0.12)), gains=(('l1_img',(0.1,0.01,0.05)),('ssim',(0.2,0.02,0.05))), warmup_steps=1000)
            self.q_config = QControllerConfig(warmup_lr_start=1e-7, warmup_steps=1000)
            pid_state = init_pid_controller(self.pid_config); q_state = init_q_controller(self.q_config)
            start_epoch, global_step, best_val_loss = 0, 0, float('inf')
            ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl"); ckpt_path_best = Path(f"{self.args.basename}_{self.args.d_model}d_512_best.pkl")
            
            if ckpt_path.exists():
                console.print(f"--- Resuming from {ckpt_path} ---")
                with open(ckpt_path, 'rb') as f: data = pickle.load(f)
                params = data['params']; ema_params = data.get('ema_params', params); pid_state = data.get('pid_controller_state', pid_state)
                start_epoch=data.get('epoch',0); global_step=data.get('global_step',0); best_val_loss=float(data.get('best_val_loss',float('inf')))
                q_state = q_state.replace(step_count=jnp.array(global_step))
                state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=ema_params, pid_controller_state=pid_state, q_controller_state=q_state)
                try:
                    state = state.replace(opt_state=data['opt_state'])
                    console.print("-- Optimizer state loaded successfully from checkpoint. --")
                except (TypeError, ValueError, KeyError):
                    console.print(f"[bold yellow]-- WARNING: Could not load optimizer state. Resetting optimizer state. --[/bold yellow]")
            else:
                console.print("--- Initializing new model ---")
                with jax.default_device(CPU_DEVICE):
                    params = self.model.init({'params':jax.random.PRNGKey(0),'dropout':jax.random.PRNGKey(1)}, jnp.zeros((1,512,512,3),self.dtype), jnp.zeros((1024,2),self.dtype))['params']
                state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=params, pid_controller_state=pid_state, q_controller_state=q_state)
            
            p_state = replicate(state)
            self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, unreplicate(state.params), 0)
    
            @partial(jax.pmap, axis_name='devices', donate_argnums=(0,), static_broadcasted_argnums=(4,5))
            def train_step(state: CustomTrainState, batch: chex.Array, coords: chex.Array, train_rng: chex.PRNGKey, pid_cfg: PIDControllerConfig, q_cfg: QControllerConfig):
                q_rng, loss_rng = jax.random.split(jnp.squeeze(train_rng))
                new_q_state_pre = q_controller_choose_action(state.q_controller_state, q_rng, q_cfg, self.args.lr)
                current_lr = new_q_state_pre.current_lr
                
                def loss_fn(params):
                    _,dropout_rng,hard_step_rng=jax.random.split(loss_rng,3)
                    model_dynamic = self.model.clone(cheat_prob=jnp.where(jax.random.bernoulli(hard_step_rng,0.1), 0.0, self.args.cheat_prob))
                    recon_full = model_dynamic.apply({'params':params}, batch, coords, rngs={'dropout':dropout_rng}).reshape(batch.shape)
                    
                    # ============================ FIX STARTS HERE ============================
                    # The original line was too long and used a Python list with jnp.dot, causing a TypeError.
                    # It has been broken down and corrected.
                    l1 = jnp.mean(jnp.abs(recon_full - batch))
                    
                    # 1. Convert the Python list to a JAX array.
                    # 2. Match the dtype of the batch for mixed-precision compatibility.
                    lum_weights = jnp.array([.299, .587, .114], dtype=batch.dtype)
                    
                    batch_lum = jnp.dot(batch, lum_weights)[..., None]
                    recon_lum = jnp.dot(recon_full, lum_weights)[..., None]
                    
                    ssim = self._ssim_loss(batch_lum, recon_lum)
                    gdl = self._ms_gdl_loss(batch_lum, recon_lum)
                    lap = self._laplacian_pyramid_loss(batch, recon_full)
                    # ============================= FIX ENDS HERE =============================

                    metrics_arr=jnp.stack([l1,ssim,gdl,lap]); new_pid_state,weights=pid_controller_update(state.pid_controller_state,metrics_arr,pid_cfg)
                    total_loss = jnp.dot(metrics_arr, weights)
                    metrics={'total_raw':total_loss,'l1_img':l1,'ssim':ssim,'ms_gdl':gdl,'laplacian':lap}
                    return jnp.nan_to_num(total_loss,nan=1e5),(metrics,new_pid_state)
    
                (loss,(metrics,new_pid_state)),grads=jax.value_and_grad(loss_fn,has_aux=True)(state.params)
                grads=jax.lax.pmean(grads,'devices'); metrics=jax.lax.pmean(metrics,'devices')
                new_pid_state=jax.tree_util.tree_map(lambda x:jax.lax.pmean(x,'devices'),new_pid_state)
                updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, learning_rate=current_lr)
                new_params = optax.apply_updates(state.params, updates)
                new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1 - 0.999), state.ema_params, new_params)
                final_q_state = q_controller_update(new_q_state_pre, metrics['total_raw'], q_cfg)
                new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, ema_params=new_ema_params, pid_controller_state=new_pid_state, q_controller_state=final_q_state)
                return new_state, metrics, current_lr, final_q_state.status_code
                
                
                
            @partial(jax.pmap, axis_name='devices')
            def eval_step(ema_params, batch, coords):
                recon=self.model.clone(cheat_prob=0.0).apply({'params':ema_params},batch,coords,rngs={'dropout':jax.random.PRNGKey(0)}).reshape(batch.shape)
                return jax.lax.pmean(jnp.mean(jnp.abs(recon-batch)),'devices')
    
            console.print("--- Compiling JAX functions (one-time cost)... ---")
            compile_batch = common_utils.shard(next(train_iterator).reshape(REBATCH_SIZE,self.args.batch_size*self.num_devices,self.args.image_size,self.args.image_size,3)[0].astype(self.dtype))
            p_state, _, _, _ = train_step(p_state, compile_batch, p_full_coords, common_utils.shard_prng_key(train_key), self.pid_config, self.q_config)
            eval_step(p_state.ema_params, compile_batch, p_full_coords).block_until_ready()
            self.generate_preview(unreplicate(p_state.ema_params), self.preview_images_device[0:1]).block_until_ready()
            console.print("--- Compilation complete. Starting training. ---")
            spinner_column = TextColumn("🍓", style="magenta")
            self.progress=Progress(spinner_column,TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[best_val]:.4f}[/]"),BarColumn(),"•",TextColumn("Step {task.fields[step]}/{task.fields[steps_per_epoch]}"),"•",TimeRemainingColumn())
            epoch_task=self.progress.add_task("epochs",total=self.args.epochs,completed=start_epoch,best_val=float(best_val_loss),step=global_step%steps_per_epoch if steps_per_epoch>0 else 0,steps_per_epoch=steps_per_epoch)
            last_step_time, last_ui_update_time = time.time(), time.time()
            live = Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
            try:
                live.start()
                with ThreadPoolExecutor(max_workers=1) as async_pool:
                    active_preview_future=None
                    while global_step < total_steps:
                        if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                        super_batch=next(train_iterator).reshape(REBATCH_SIZE,self.args.batch_size*self.num_devices,self.args.image_size,self.args.image_size,3)
                        for batch_np in super_batch:
                            if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                            spinner_column.style="magenta" if(global_step//2)%2==0 else "blue"
                            train_key,step_key=jax.random.split(train_key)
                            p_state,metrics,q_lr,q_status=train_step(p_state,common_utils.shard(batch_np.astype(self.dtype)),p_full_coords,common_utils.shard_prng_key(step_key),self.pid_config, self.q_config)
                            global_step+=1; current_epoch=global_step//steps_per_epoch if steps_per_epoch>0 else 0; step_in_epoch=global_step%steps_per_epoch if steps_per_epoch>0 else 0
                            time_now=time.time(); self.steps_per_sec=1.0/(time_now-last_step_time+1e-6); last_step_time=time_now
                            self.progress.update(epoch_task,completed=current_epoch,step=step_in_epoch+1)
                            if(time_now - last_ui_update_time) > (1.0/15.0):
                                metrics_unrep=unreplicate(metrics); _,current_weights_unrep=pid_controller_update(unreplicate(p_state.pid_controller_state),jnp.zeros(len(self.pid_config.loss_names)),self.pid_config)
                                with self.ui_lock:
                                    self.last_metrics=jax.device_get(metrics_unrep); self.current_lambdas=dict(zip(self.pid_config.loss_names,jax.device_get(current_weights_unrep)))
                                    if self.last_metrics and 'total_raw' in self.last_metrics and np.isfinite(self.last_metrics['total_raw']): self.loss_hist.append(self.last_metrics['total_raw'])
                                    self.current_q_lr = float(unreplicate(q_lr)); self.current_q_status = int(unreplicate(q_status))
                                if active_preview_future is None or active_preview_future.done():
                                    if active_preview_future: active_preview_future.result()
                                    active_preview_future=async_pool.submit(self._update_preview_task,unreplicate(p_state.ema_params),self.preview_images_device[current_preview_idx:current_preview_idx+1])
                                preview_change = self.interactive_state.get_and_reset_preview_change()
                                if preview_change != 0:
                                    current_preview_idx = (current_preview_idx+preview_change)%self.preview_images_device.shape[0]
                                    self.current_preview_np=((preview_buffer_host[current_preview_idx]*0.5+0.5)*255).astype(np.uint8)
                                    if Pixels:
                                        term_w=64;h,w,_=self.current_preview_np.shape;term_h=int(term_w*(h/w)*0.5)
                                        self.rendered_original_preview=Pixels.from_image(Image.fromarray(self.current_preview_np).resize((term_w,term_h),Image.LANCZOS))
                                live.update(self._generate_layout(),refresh=True); last_ui_update_time=time_now
                            if global_step > 0 and global_step % self.args.eval_every == 0:
                                with self.ui_lock:
                                    self.is_validating=True; VAL_STEPS=50
                                    self.validation_progress=Progress(TextColumn("[cyan]Validating..."),BarColumn(),TextColumn("{task.completed}/{task.total}"))
                                    val_task=self.validation_progress.add_task("val",total=VAL_STEPS)
                                live.update(self._generate_layout(),refresh=True)
                                val_iterator=val_dataset.take(VAL_STEPS).as_numpy_iterator(); val_losses=[]
                                for val_batch in val_iterator:
                                    loss=eval_step(p_state.ema_params,common_utils.shard(val_batch.astype(self.dtype)),p_full_coords)
                                    val_losses.append(jax.device_get(unreplicate(loss)))
                                    with self.ui_lock: self.validation_progress.update(val_task,advance=1)
                                    live.update(self._generate_layout(),refresh=True)
                                val_loss=np.mean(val_losses) if val_losses else float('inf')
                                if val_loss<best_val_loss:
                                    best_val_loss=val_loss
                                    self.progress.update(epoch_task, best_val=float(best_val_loss))
                                    self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path_best)
                                console.print(f"\n[bold yellow]-- Validation complete. L1 Loss: {float(val_loss):.4f} (Best: {float(best_val_loss):.4f}) --[/bold yellow]")
                                with self.ui_lock: self.is_validating=False; self.validation_progress=None
                                live.update(self._generate_layout(),refresh=True)
                            
                            # --- CORRECTED INDENTATION ---
                            if self.interactive_state.get_and_reset_force_save() or (global_step > 0 and steps_per_epoch > 0 and global_step % steps_per_epoch == 0 and (current_epoch + 1) % self.args.save_every == 0):
                                self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path)
            finally:
                live.stop(); print("\n--- Training loop finished. ---")
                self.interactive_state.set_shutdown()
                if key_listener_thread.is_alive(): key_listener_thread.join()
                if 'p_state' in locals() and 'global_step' in locals():
                    print("--- Saving final model state... ---")
                    current_epoch = global_step//steps_per_epoch if steps_per_epoch>0 else 0
                    self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path)
                    print("--- ✅ Final state saved. ---")
    




    
# =================================================================================================
# 5. MAIN EXECUTION BLOCK
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Topological AE - On-the-Fly Hybrid Loss Trainer")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the directory with SOURCE images (JPG, PNG, etc.).")
    parser.add_argument('--basename', type=str, required=True, help="Basename for model files.")
    parser.add_argument('--d-model', type=int, default=128, help="Model dimension.")
    parser.add_argument('--latent-grid-size', type=int, default=16, help="Size of the latent grid.")
    parser.add_argument('--image-size', type=int, default=512, help="Image resolution.")
    parser.add_argument('--epochs', type=int, default=100, help="Total number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size PER DEVICE.")
    parser.add_argument('--lr', type=float, default=2e-4, help="Target learning rate for the Q-Controller.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 mixed precision.")
    parser.add_argument('--save-every', type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument('--eval-every', type=int, default=200, help="Run validation every N global steps.")
    parser.add_argument('--cheat-prob', type=float, default=0.9, help="Probability of providing oracle pixels to the decoder (teacher forcing).")
    args = parser.parse_args()
    tf.config.set_visible_devices([], 'GPU')
    ImageTrainer(args).train()

if __name__ == "__main__":
    main()