import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
from pathlib import Path
import platform
import atexit
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
import math
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from flax import linen as nn
from flax.training import train_state, common_utils
from flax.jax_utils import replicate, unreplicate
from flax import struct
import optax
import numpy as np
import pickle
import time
from typing import Any, Sequence, Tuple, Optional
import sys
import argparse
import signal
import threading
from functools import partial
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from dataclasses import dataclass
from jax import jit
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
    from rich.text import Text
    import pynvml; pynvml.nvmlInit()
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
def rgb_to_hsl_jax(rgb: chex.Array) -> chex.Array:
    epsilon = 1e-8; r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = jnp.maximum(jnp.maximum(r, g), b); cmin = jnp.minimum(jnp.minimum(r, g), b); delta = cmax - cmin
    l = (cmax + cmin) / 2.0
    s = jnp.where(delta < epsilon, 0.0, delta / (1.0 - jnp.abs(2.0 * l - 1.0) + epsilon))
    h = jnp.zeros_like(r)
    h = jnp.where(cmax == r, (((g - b) / (delta + epsilon))) % 6.0, h)
    h = jnp.where(cmax == g, (((b - r) / (delta + epsilon)) + 2.0), h)
    h = jnp.where(cmax == b, (((r - g) / (delta + epsilon)) + 4.0), h)
    h = h / 6.0; h = jnp.where(delta < epsilon, 0.0, h)
    return jnp.stack([h, s, l], axis=-1)
def circular_l1_loss(pred: chex.Array, target: chex.Array) -> chex.Array:
    diff = jnp.abs(pred - target); return jnp.minimum(diff, 1.0 - diff)
def rgb_to_grayscale_jax(rgb: chex.Array) -> chex.Array:
    return jnp.dot(rgb[...,:3], jnp.array([0.2989, 0.5870, 0.1140]))
@partial(jit, static_argnames=('shape',))
def generate_ellipse_mask(coords, key, shape):
    h, w = shape; key, cx_k, cy_k, rx_k, ry_k, rot_k = jax.random.split(key, 6)
    cx = jax.random.uniform(cx_k, (), minval=-0.5, maxval=0.5); cy = jax.random.uniform(cy_k, (), minval=-0.5, maxval=0.5)
    rx = jax.random.uniform(rx_k, (), minval=0.2, maxval=0.8); ry = jax.random.uniform(ry_k, (), minval=0.2, maxval=0.8)
    theta = jax.random.uniform(rot_k, (), minval=0, maxval=2*jnp.pi)
    x, y = coords[..., 0] - cx, coords[..., 1] - cy
    x_rot = x * jnp.cos(theta) - y * jnp.sin(theta); y_rot = x * jnp.sin(theta) + y * jnp.cos(theta)
    mask = ((x_rot**2 / rx**2) + (y_rot**2 / ry**2) < 1.0).astype(jnp.float32)
    return mask.reshape(shape)
@partial(jit, static_argnames=('shape',))
def generate_rect_mask(coords, key, shape):
    h, w = shape; key, cx_k, cy_k, rw_k, rh_k, rot_k = jax.random.split(key, 6)
    cx = jax.random.uniform(cx_k, (), minval=-0.5, maxval=0.5); cy = jax.random.uniform(cy_k, (), minval=-0.5, maxval=0.5)
    rw = jax.random.uniform(rw_k, (), minval=0.2, maxval=0.8); rh = jax.random.uniform(rh_k, (), minval=0.2, maxval=0.8)
    theta = jax.random.uniform(rot_k, (), minval=0, maxval=2*jnp.pi)
    x, y = coords[..., 0] - cx, coords[..., 1] - cy
    x_rot = x * jnp.cos(theta) - y * jnp.sin(theta); y_rot = x * jnp.sin(theta) + y * jnp.cos(theta)
    mask = (jnp.abs(x_rot) < rw / 2.0) & (jnp.abs(y_rot) < rh / 2.0)
    return mask.astype(jnp.float32).reshape(shape)
@partial(jit, static_argnames=('shape', 'num_ops_min', 'num_ops_max'))
def generate_complex_mask(key, shape, num_ops_min=3, num_ops_max=7):
    h, w = shape
    coords = jnp.mgrid[-1:1:h*1j, -1:1:w*1j].transpose(1, 2, 0).reshape(-1, 2)
    key, num_ops_key = jax.random.split(key)
    num_ops = jax.random.randint(num_ops_key, (), num_ops_min, num_ops_max)
    final_mask = jnp.zeros(shape)
    def body_fn(i, state):
        key, current_mask = state
        key, op_key, shape_key, prim_key = jax.random.split(key, 4)
        primitive_idx = jax.random.randint(prim_key, (), 0, 2)
        new_shape_mask = jax.lax.cond(primitive_idx == 0,
            lambda: generate_ellipse_mask(coords, shape_key, shape),
            lambda: generate_rect_mask(coords, shape_key, shape))
        op_idx = jax.random.randint(op_key, (), 0, 3)
        union_mask = jnp.maximum(current_mask, new_shape_mask)
        diff_mask = jnp.clip(current_mask - new_shape_mask, 0, 1)
        inter_mask = jnp.minimum(current_mask, new_shape_mask)
        op_masks = jnp.stack([union_mask, diff_mask, inter_mask])
        return key, jax.lax.select(i == 0, new_shape_mask, op_masks[op_idx])
    _, final_mask = jax.lax.fori_loop(0, num_ops, body_fn, (key, final_mask))
    kernel = jnp.ones((5, 5)) / 25.0
    return convolve2d(final_mask, kernel, mode='same')
@partial(jit, static_argnames=('shape',))
def create_synthetic_rgba_texture_batch(key, fg_batch, bg_batch, shape):
    B, H, W, C = fg_batch.shape
    key, mask_key, texture_key = jax.random.split(key, 3)
    alpha_mask = jax.vmap(generate_complex_mask, in_axes=(0, None))(jax.random.split(mask_key, B), shape)[..., None]
    fg_lum = rgb_to_grayscale_jax(fg_batch * 0.5 + 0.5)[..., None]
    kernel = jnp.ones((1, 1, 5, 5)) / 25.0
    blurred_lum = jax.lax.conv_general_dilated(fg_lum.transpose(0,3,1,2), kernel, (1,1), 'SAME').transpose(0,2,3,1)
    high_freq_texture = jnp.clip(fg_lum - blurred_lum, -0.5, 0.5) + 0.5
    checkerboard = jnp.mgrid[0:H, 0:W].sum(axis=0) % 2
    checkerboard_bcast = checkerboard[None, ..., None]
    gt_structure_channel = alpha_mask * checkerboard_bcast + high_freq_texture * (1 - checkerboard_bcast)
    target_rgb = fg_batch * alpha_mask + bg_batch * (1.0 - alpha_mask)
    return jnp.concatenate([target_rgb, gt_structure_channel], axis=-1), checkerboard[None,...,None]
class InteractivityState:
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_index_change, self.force_save = 0, False
        self.shutdown_event = threading.Event()
    def get_and_reset_preview_change(self):
        with self.lock: change = self.preview_index_change; self.preview_index_change = 0; return change
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def set_shutdown(self): self.shutdown_event.set()
def listen_for_keys(shared_state: InteractivityState):
    print("--- Key listener started. Controls: [←/→] Preview | [s] Force Save | [q] Quit ---")
    if platform.system() == "Windows": import msvcrt
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
                elif key == b'\xe0' or key == '\x1b':
                    arrow = msvcrt.getch() if platform.system() == "Windows" else sys.stdin.read(2)
                    if arrow in [b'K', '[D']: shared_state.preview_index_change = -1
                    elif arrow in [b'M', '[C']: shared_state.preview_index_change = 1
    finally:
        if platform.system() != "Windows": termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; current_lr: jnp.ndarray
    exploration_rate: jnp.ndarray; step_count: jnp.ndarray; last_action_idx: jnp.ndarray; status_code: jnp.ndarray
@dataclass(frozen=True)
class QControllerConfig:
    num_lr_actions: int = 5; lr_change_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1; lr_min: float = 1e-4; lr_max: float = 5e-2
    metric_history_len: int = 100; exploration_rate_q: float = 0.3; min_exploration_rate: float = 0.05; exploration_decay: float = 0.9998
    warmup_steps: int = 500; warmup_lr_start: float = 1e-6
def init_q_controller(config: QControllerConfig) -> QControllerState:
    return QControllerState(q_table=jnp.zeros(config.num_lr_actions), metric_history=jnp.zeros(config.metric_history_len),
                            current_lr=jnp.array(config.warmup_lr_start), exploration_rate=jnp.array(config.exploration_rate_q),
                            step_count=jnp.array(0), last_action_idx=jnp.array(-1, dtype=jnp.int32), status_code=jnp.array(0, dtype=jnp.int32))
@partial(jit, static_argnames=('config', 'target_lr'))
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey, config: QControllerConfig, target_lr: float) -> QControllerState:
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config.warmup_steps
        lr = config.warmup_lr_start * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=lr, step_count=state.step_count + 1, status_code=jnp.array(0, dtype=jnp.int32), last_action_idx=jnp.array(-1, dtype=jnp.int32))
    def regular_action():
        explore, act = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore) < jnp.squeeze(state.exploration_rate),
            lambda: jax.random.randint(act, (), 0, config.num_lr_actions, dtype=jnp.int32),
            lambda: jnp.argmax(state.q_table).astype(jnp.int32))
        new_lr = jnp.clip(state.current_lr * jnp.array(config.lr_change_factors)[action_idx], config.lr_min, config.lr_max)
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx, status_code=jnp.array(0, dtype=jnp.int32))
    return jax.lax.cond(jnp.squeeze(state.step_count) < config.warmup_steps, warmup_action, regular_action)
@partial(jit, static_argnames=('config',))
def q_controller_update(state: QControllerState, metric_value: chex.Array, config: QControllerConfig) -> QControllerState:
    new_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value)
    st = state.replace(metric_history=new_history)
    def perform_q_update(s: QControllerState) -> QControllerState:
        reward = -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 10, 10))
        is_improving = reward > -jnp.mean(jax.lax.dynamic_slice_in_dim(s.metric_history, config.metric_history_len - 20, 10))
        status = jax.lax.select(is_improving, jnp.array(1, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32))
        old_q = s.q_table[s.last_action_idx]
        new_q = old_q + config.learning_rate_q * (reward - old_q)
        return s.replace(q_table=s.q_table.at[s.last_action_idx].set(new_q),
                         exploration_rate=jnp.maximum(config.min_exploration_rate, s.exploration_rate * config.exploration_decay),
                         status_code=status)
    can_update = (jnp.squeeze(st.step_count) > config.warmup_steps) & (jnp.squeeze(st.last_action_idx) >= 0)
    return jax.lax.cond(can_update, perform_q_update, lambda s: s, st)
class CustomTrainState(train_state.TrainState):
    ema_params: Any; q_controller_state: QControllerState; opt_state: optax.OptState
class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2); imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part
class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images_rgb: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, features, current_dim, i = images_rgb, 32, self.input_image_size, 0
        context_vectors = []
        while (current_dim // 2) >= self.latent_grid_size and (current_dim // 2) > 0:
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            context_vectors.append(jnp.mean(x, axis=(1, 2)))
            features *= 2; current_dim //= 2; i += 1
        context_vector = jnp.concatenate(context_vectors, axis=-1)
        if current_dim != self.latent_grid_size:
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        path_params_raw = nn.Conv(3, (1, 1), name="path_params_head", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params_raw[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params_raw[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params_raw[..., 2]) * (jnp.pi / 2.0)
        path_params = jnp.stack([delta_c, chi_c, radius], axis=-1)
        return path_params, context_vector
class TopologicalObserver(nn.Module):
    d_model: int; latent_dim: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = path_params_grid.shape; L = H * W
        path_params = path_params_grid.reshape(B, L, 3)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
        chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path) + 1e-8
        path_real_mean = jnp.mean(t_co_steps.real, axis=-1); path_real_std = jnp.std(t_co_steps.real, axis=-1)
        path_imag_mean = jnp.mean(t_co_steps.imag, axis=-1); path_imag_std = jnp.std(t_co_steps.imag, axis=-1)
        complex_measurement = jnp.stack([path_real_mean, path_real_std, path_imag_mean, path_imag_std], axis=-1)
        base_features = nn.Dense(self.d_model, name="base_feature_projector", dtype=self.dtype)(complex_measurement)
        detail_generator = nn.Sequential([
            nn.Dense(self.d_model * 2, name="expander_hidden", dtype=self.dtype), nn.gelu,
            nn.Dense(self.d_model, name="expander_output", dtype=self.dtype)])
        detail_features = detail_generator(base_features)
        virtual_expanded_features = jnp.concatenate([base_features, detail_features], axis=-1)
        final_latent_features = nn.Dense(self.latent_dim, name="final_latent_projector", dtype=self.dtype)(virtual_expanded_features)
        return final_latent_features.reshape(B, H, W, self.latent_dim)
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
    latent_dim: int; num_freqs: int = 10; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.remat
    def _mlp_block(self, h: jnp.ndarray, context_vector: jnp.ndarray) -> jnp.ndarray:
        film_layer = FiLMLayer(dtype=self.dtype)
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = film_layer(h, context_vector)
            h = nn.gelu(h)
        mlp_output = nn.Dense(4, name="mlp_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(h)
        rgb = nn.tanh(mlp_output[..., :3]); structure = nn.sigmoid(mlp_output[..., 3:4])
        return jnp.concatenate([rgb, structure], axis=-1)
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = feature_grid.shape
        encoded_coords = PositionalEncoding(self.num_freqs, dtype=self.dtype)(coords)
        pyramid = [feature_grid] + [jax.image.resize(feature_grid, (B, H//(2**i), W//(2**i), self.latent_dim), 'bilinear') for i in range(1, 3)]
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
        mlp_input = jnp.concatenate([encoded_coords_tiled, concatenated_features], axis=-1)
        return self._mlp_block(mlp_input, context_vector)
class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_dim: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, latent_dim=self.latent_dim, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(latent_dim=self.latent_dim, name="coord_decoder", dtype=self.dtype)
    def __call__(self, images_rgb, coords):
        path_params, context_vector = self.encode(images_rgb)
        feature_grid = self.observer(path_params)
        pixels_rgba_struct = self.coord_decoder(feature_grid, context_vector, coords)
        return pixels_rgba_struct
    def encode(self, images_rgb) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.modulator(images_rgb)
    def decode(self, path_params: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        feature_grid = self.observer(path_params)
        pixels_rgba_struct = self.coord_decoder(feature_grid, context_vector, coords)
        return pixels_rgba_struct
def create_on_the_fly_dataset(image_dir: str, image_size: int, is_training: bool):
    base_path = Path(image_dir).resolve()
    path_cache_file = base_path / "image_path_cache.pkl"
    if path_cache_file.exists():
        print(f"--- Found path cache file. Loading file list from: {path_cache_file} ---")
        with open(path_cache_file, 'rb') as f: image_paths = pickle.load(f)
    else:
        print(f"--- No path cache found. Scanning directory: {base_path} ---")
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        image_paths = [str(p) for p in base_path.rglob('*') if p.suffix.lower() in image_extensions]
        if image_paths:
            print(f"--- Found {len(image_paths)} images. Saving path list to cache... ---")
            with open(path_cache_file, 'wb') as f: pickle.dump(image_paths, f)
    if not image_paths: print(f"[FATAL] No images found in '{base_path}' or its cache."); sys.exit(1)
    num_samples = len(image_paths)
    print(f"--- Creating dataset with {num_samples} images. ---")
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    if is_training:
        ds = ds.cache().repeat().shuffle(buffer_size=min(num_samples, 20000))
        print(f"--- Using shuffle buffer of {min(num_samples, 20000)} for training. ---")
    else:
        ds = ds.cache()
    @tf.function
    def _process_path(file_path):
        img = tf.io.decode_image(tf.io.read_file(file_path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return tf.cast(tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.LANCZOS3), tf.float32) / 127.5 - 1.0
    return ds.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE), num_samples
class ImageTrainer:
    def __init__(self, args):
        self.args = args; self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_dim=args.latent_dim, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=self.dtype)
        self.interactive_state = InteractivityState(); self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.num_devices = jax.local_device_count(); self.ui_lock = threading.Lock()
        self.last_metrics = {}; self.current_preview_np, self.current_recon_np = None, None
        self.param_count = 0; self.loss_hist = deque(maxlen=200); self.steps_per_sec = 0.0
        self.preview_images_device, self.rendered_original_preview, self.rendered_recon_preview = None, None, None
        self.is_validating, self.validation_progress = False, None
        self.current_q_lr = 0.0; self.current_q_status = 0
        self.pixels_per_step = args.pixels_per_step
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
        unrep_state = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, unrep_state)
        data = {'epoch': epoch, 'global_step': global_step, 'best_val_loss': best_val_loss,
                'params': jax.device_get(unrep_state.params), 'ema_params': jax.device_get(unrep_state.ema_params),
                'opt_state': jax.device_get(unrep_state.opt_state), 'q_controller_state': jax.device_get(unrep_state.q_controller_state)}
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
            header_text = f"🧠⚡ [bold]Topological AE (Scene Decomposer)[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d->{self.args.latent_dim}Ld[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision}"
            layout["header"].update(Panel(Align.center(header_text),style="bold magenta",title="[dim]wubumind.ai[/dim]",title_align="right"))
            stats_tbl = Table.grid(expand=True,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
            mem,util=self._get_gpu_stats()
            q_status_map = {0: "[cyan]Warmup[/]", 1: "[green]Improving[/]", 2: "[yellow]Stagnated[/]"}
            q_status_text = q_status_map.get(self.current_q_status, "[red]Unknown[/]")
            stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🚀")
            stats_tbl.add_row("Learning Rate", f"[green]{self.current_q_lr:.2e}[/]")
            stats_tbl.add_row("LR Controller", q_status_text)
            stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            loss_raw = self.last_metrics.get('composite_loss', 0.0)
            loss_val = float(loss_raw)
            if not np.isfinite(loss_val): loss_emoji, loss_color, loss_val = "🔥", "bold red", 0.0
            elif loss_val < 0.05:   loss_emoji, loss_color = "👌", "bold green"
            elif loss_val < 0.15: loss_emoji, loss_color = "👍", "bold yellow"
            else:                 loss_emoji, loss_color = "😟", "bold red"
            if loss_raw == 0.0: loss_emoji, loss_color = "...", "dim"
            loss_panel=Panel(Align.center(f"[{loss_color}]{loss_val:.5f}[/] {loss_emoji}"), title="[bold]📉 Composite Scene Loss[/]", border_style="cyan")
            layout["left"].update(Group(Panel(stats_tbl,title="[bold]📊 Core Stats[/]",border_style="blue"),loss_panel))
            spark_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_hist,60)}[/]"),title="Loss History (Composite)",height=3,border_style="cyan")
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
        input_rgb_for_preview = preview_image_batch[..., :3]
        path_params, context_vector = self.model.apply({'params': gen_ema_params}, input_rgb_for_preview, method=self.model.encode)
        coords = jnp.mgrid[-1:1:resolution*1j, -1:1:resolution*1j].transpose(1, 2, 0).reshape(-1, 2)
        recon_pixels = self.model.apply({'params': gen_ema_params}, path_params, context_vector, coords, method=self.model.decode)
        recon_rgb = recon_pixels[..., :3]
        return recon_rgb.reshape(preview_image_batch.shape[0], resolution, resolution, 3)
    def _update_preview_task(self, gen_ema_params, preview_image_batch):
        recon_batch_rgb = self.generate_preview(gen_ema_params, preview_image_batch)
        recon_batch_rgb.block_until_ready()
        recon_rgb_np = np.array(((recon_batch_rgb[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        with self.ui_lock:
            self.current_recon_np = recon_rgb_np
            if Pixels:
                term_w=64; h,w,_=self.current_recon_np.shape; term_h=int(term_w*(h/w)*0.5)
                self.rendered_recon_preview=Pixels.from_image(Image.fromarray(self.current_recon_np).resize((term_w,term_h),Image.LANCZOS))
    def _compute_loss_and_metrics(self, params, target_rgba, loss_rng, pixels_per_step):
        B, H, W, _ = target_rgba.shape
        sampled_coords = jax.random.uniform(loss_rng, shape=(B, pixels_per_step, 2), minval=-1.0, maxval=1.0)
        coords_rescaled = (sampled_coords + 1) / 2 * jnp.array([H - 1, W - 1])
        def sample_one_image(image, coords):
            return jax.vmap(lambda c: jax.scipy.ndimage.map_coordinates(c, coords.T, order=1, mode='reflect'))(image.transpose(2,0,1)).T
        sampled_targets = jax.vmap(sample_one_image)(target_rgba, coords_rescaled)
        input_rgb_to_model = target_rgba[..., :3]
        recon_patch = self.model.apply({'params': params}, input_rgb_to_model, sampled_coords)
        pred_rgb, pred_struct = recon_patch[..., :3], recon_patch[..., 3:4]
        gt_rgb, gt_struct = sampled_targets[..., :3], sampled_targets[..., 3:4]
        targets_hsl = rgb_to_hsl_jax((gt_rgb + 1.0) / 2.0)
        recon_hsl = rgb_to_hsl_jax((pred_rgb + 1.0) / 2.0)
        loss_h = jnp.mean(circular_l1_loss(recon_hsl[..., 0], targets_hsl[..., 0]))
        loss_sl = jnp.mean(jnp.abs(recon_hsl[..., 1:] - targets_hsl[..., 1:]))
        loss_hsl = loss_h + loss_sl
        loss_structure = jnp.mean(jnp.abs(pred_struct - gt_struct))
        loss = 1.0 * loss_hsl + 1.0 * loss_structure
        metrics = {'composite_loss': loss, 'hsl_loss': loss_hsl, 'structure_loss': loss_structure}
        return loss.astype(jnp.float32), metrics
    def train(self):
            console = Console()
            key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
            console.print("--- Setting up ON-THE-FLY data pipeline... ---")
            dataset, num_samples = create_on_the_fly_dataset(self.args.data_dir, self.args.image_size, is_training=True)
            val_dataset, num_val_samples = create_on_the_fly_dataset(self.args.data_dir, self.args.image_size, is_training=False)
            REBATCH_SIZE = 100
            dataset = dataset.batch(self.args.batch_size * self.num_devices * REBATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            preview_dataset = val_dataset.batch(50).prefetch(tf.data.AUTOTUNE)
            val_dataset = val_dataset.batch(self.args.batch_size * self.num_devices, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
            train_iterator = iter(tfds.as_numpy(dataset))
            val_iterator = iter(tfds.as_numpy(val_dataset))
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
            total_steps = self.args.epochs * steps_per_epoch if steps_per_epoch > 0 else 1
            console.print(f"📈 Training for {self.args.epochs} epochs ({total_steps} total steps).")
            train_key = jax.random.PRNGKey(self.args.seed)
            eval_key = jax.random.PRNGKey(self.args.seed + 1)
            optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr, b1=0.9, b2=0.95))
            self.q_config = QControllerConfig(warmup_lr_start=1e-7, warmup_steps=1000)
            q_state = init_q_controller(self.q_config)
            start_epoch, global_step, best_val_loss = 0, 0, float('inf')
            ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_{self.args.image_size}.pkl"); ckpt_path_best = Path(f"{self.args.basename}_{self.args.d_model}d_{self.args.image_size}_best.pkl")
            if ckpt_path.exists():
                console.print(f"--- Resuming from {ckpt_path} ---")
                with open(ckpt_path, 'rb') as f: data = pickle.load(f)
                params = jax.tree_util.tree_map(lambda x: x.astype(self.dtype), data['params'])
                ema_params = jax.tree_util.tree_map(lambda x: x.astype(self.dtype), data.get('ema_params', params))
                q_state = data.get('q_controller_state', q_state)
                start_epoch = data.get('epoch', 0); global_step = data.get('global_step', 0); best_val_loss = float(data.get('best_val_loss', float('inf')))
                q_state = q_state.replace(step_count=jnp.array(global_step))
                state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=ema_params, q_controller_state=q_state)
                try:
                    opt_state_loaded = jax.tree_util.tree_map(lambda x: x.astype(self.dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, data['opt_state'])
                    state = state.replace(opt_state=opt_state_loaded); console.print("-- Optimizer state loaded successfully from checkpoint. --")
                except (KeyError, TypeError, ValueError):
                    console.print("[bold yellow]-- WARNING: No/mismatched optimizer state in ckpt. Initializing new state. --[/bold yellow]")
            else:
                console.print("--- Initializing new model ---")
                with jax.default_device(CPU_DEVICE):
                    dummy_img = jnp.zeros((1,self.args.image_size,self.args.image_size,3), self.dtype)
                    dummy_coords = jnp.zeros((1024,2), self.dtype)
                    params = self.model.init(jax.random.PRNGKey(0), dummy_img, dummy_coords)['params']
                state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=params, q_controller_state=q_state)
            p_state = replicate(state)
            self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, unreplicate(state.params), 0)
            @partial(jax.pmap, axis_name='devices', donate_argnums=(0,), static_broadcasted_argnums=(3,4))
            def train_step(state: CustomTrainState, batch_pair: chex.Array, train_rng: chex.PRNGKey, q_cfg: QControllerConfig, pixels_per_step: int):
                q_rng, loss_rng, synth_rng = jax.random.split(jnp.squeeze(train_rng), 3)
                fg_batch, bg_batch = batch_pair[:, 0], batch_pair[:, 1]
                B, H, W, C = fg_batch.shape
                target_rgba, _ = create_synthetic_rgba_texture_batch(synth_rng, fg_batch, bg_batch, (H, W))
                def loss_fn(params):
                    return self._compute_loss_and_metrics(params, target_rgba, loss_rng, pixels_per_step)
                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                grad_leaves = jax.tree_util.tree_leaves(grads); grads_are_valid = jax.lax.pmin(jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in grad_leaves])), 'devices')
                def apply_updates(operands):
                    state, grads, metrics, q_rng = operands
                    grads = jax.lax.pmean(grads, 'devices'); metrics = jax.lax.pmean(metrics, 'devices')
                    new_q_state_pre = q_controller_choose_action(state.q_controller_state, q_rng, q_cfg, self.args.lr)
                    current_lr = new_q_state_pre.current_lr
                    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, learning_rate=current_lr)
                    new_params = optax.apply_updates(state.params, updates)
                    new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1 - 0.999), state.ema_params, new_params)
                    final_q_state = q_controller_update(new_q_state_pre, metrics['composite_loss'], q_cfg)
                    return state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, ema_params=new_ema_params, q_controller_state=final_q_state), current_lr
                new_state, updated_lr = jax.lax.cond(grads_are_valid, apply_updates, lambda s: (s[0], s[0].q_controller_state.current_lr), (state, grads, metrics, q_rng))
                return new_state, jax.lax.pmean(metrics, 'devices'), updated_lr, jax.lax.pmean(new_state.q_controller_state.status_code, 'devices')
            @partial(jax.pmap, axis_name='devices') 
            def validation_step(ema_params: Any, batch_pair: chex.Array, eval_rng: chex.PRNGKey):
                synth_rng, loss_rng = jax.random.split(jnp.squeeze(eval_rng), 2)
                fg_batch, bg_batch = batch_pair[:, 0], batch_pair[:, 1]
                B, H, W, _ = fg_batch.shape
                target_rgba, _ = create_synthetic_rgba_texture_batch(synth_rng, fg_batch, bg_batch, (H, W))
                _, metrics = self._compute_loss_and_metrics(ema_params, target_rgba, loss_rng, self.pixels_per_step)
                return jax.lax.pmean(metrics, 'devices')
            console.print("--- Compiling JAX functions (one-time cost)... ---")
            compile_super_batch = next(train_iterator).reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, self.args.image_size, self.args.image_size, 3)
            compile_images = compile_super_batch[0]
            compile_batch_pair = common_utils.shard(np.stack([compile_images, compile_images], axis=1))
            p_state, _, _, _ = train_step(p_state, compile_batch_pair, common_utils.shard_prng_key(train_key), self.q_config, self.pixels_per_step)
            validation_step(p_state.ema_params, compile_batch_pair, common_utils.shard_prng_key(eval_key))['composite_loss'].block_until_ready()
            self.generate_preview(unreplicate(p_state.ema_params), self.preview_images_device[0:1]).block_until_ready()
            console.print("--- Compilation complete. Starting training. ---")
            spinner_column = TextColumn("🍓", style="magenta")
            self.progress=Progress(spinner_column,TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[best_val]:.4f}[/]"),BarColumn(),"•",TextColumn("Step {task.fields[step]}/{task.fields[steps_per_epoch]}"),"•",TimeRemainingColumn())
            self.validation_progress = Progress(TextColumn("[bold yellow]Validating..."), BarColumn(), TextColumn("{task.completed}/{task.total}"))
            epoch_task=self.progress.add_task("epochs",total=self.args.epochs,completed=start_epoch,best_val=float(best_val_loss),step=global_step%steps_per_epoch if steps_per_epoch>0 else 0,steps_per_epoch=steps_per_epoch)
            val_task = self.validation_progress.add_task("val_batches", total=10, visible=False)
            last_step_time, last_ui_update_time = time.time(), time.time()
            live = Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
            try:
                live.start()
                with ThreadPoolExecutor(max_workers=1) as async_pool:
                    active_preview_future=None
                    while global_step < total_steps:
                        if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                        super_batch=next(train_iterator).reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, self.args.image_size,self.args.image_size,3)
                        for batch_idx in range(REBATCH_SIZE):
                            if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                            images_np = super_batch[batch_idx]
                            batch_pair_np = np.stack([images_np, images_np], axis=1)
                            spinner_column.style="magenta" if(global_step//2)%2==0 else "blue"
                            train_key,step_key=jax.random.split(train_key)
                            p_state,metrics,q_lr,q_status=train_step(p_state,common_utils.shard(batch_pair_np.astype(self.dtype)),common_utils.shard_prng_key(step_key),self.q_config, self.pixels_per_step)
                            global_step+=1; current_epoch=global_step//steps_per_epoch if steps_per_epoch>0 else 0; step_in_epoch=global_step%steps_per_epoch if steps_per_epoch>0 else 0
                            time_now=time.time(); self.steps_per_sec=1.0/(time_now-last_step_time+1e-6); last_step_time=time_now
                            self.progress.update(epoch_task,completed=current_epoch,step=step_in_epoch+1)
                            if(time_now - last_ui_update_time) > (1.0/15.0):
                                metrics_unrep=unreplicate(metrics)
                                with self.ui_lock:
                                    self.last_metrics=jax.device_get(metrics_unrep)
                                    if self.last_metrics and 'composite_loss' in self.last_metrics and np.isfinite(self.last_metrics['composite_loss']): self.loss_hist.append(self.last_metrics['composite_loss'])
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
                            if self.interactive_state.get_and_reset_force_save() or (global_step > 0 and global_step % self.args.save_every_steps == 0):
                                self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path)
                            if global_step > 0 and global_step % self.args.eval_every == 0:
                                self.is_validating = True; self.validation_progress.update(val_task, completed=0, total=10, visible=True)
                                live.update(self._generate_layout(), refresh=True)
                                val_metrics = []
                                for val_step in range(10):
                                    try: val_images_np = next(val_iterator)
                                    except StopIteration: val_iterator = iter(tfds.as_numpy(val_dataset)); val_images_np = next(val_iterator)
                                    val_batch_pair_np = np.stack([val_images_np, val_images_np], axis=1)
                                    eval_key, step_eval_key = jax.random.split(eval_key)
                                    metrics = validation_step(p_state.ema_params, common_utils.shard(val_batch_pair_np.astype(self.dtype)), common_utils.shard_prng_key(step_eval_key))
                                    val_metrics.append(unreplicate(metrics))
                                    self.validation_progress.update(val_task, advance=1); live.update(self._generate_layout(), refresh=True)
                                avg_val_loss = np.mean([m['composite_loss'] for m in val_metrics])
                                console.print(f"\n[bold green]-- ✅ Validation @ step {global_step}. Avg Loss: {avg_val_loss:.4f} (Best: {best_val_loss:.4f}) --[/bold green]")
                                if avg_val_loss < best_val_loss:
                                    best_val_loss = avg_val_loss
                                    console.print(f"[bold magenta]-- 🎉 New best validation loss! Saving best model. --[/bold magenta]")
                                    self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path_best)
                                self.progress.update(epoch_task, best_val=float(best_val_loss))
                                self.is_validating = False; self.validation_progress.update(val_task, visible=False)
                                live.update(self._generate_layout(), refresh=True)
            finally:
                live.stop(); print("\n--- Training loop finished. ---")
                self.interactive_state.set_shutdown()
                if key_listener_thread.is_alive(): key_listener_thread.join()
                if 'p_state' in locals() and 'global_step' in locals():
                    print("--- Saving final model state... ---")
                    current_epoch = global_step//steps_per_epoch if steps_per_epoch>0 else 0
                    self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path)
                    print("--- ✅ Final state saved. ---")
def main():
    parser = argparse.ArgumentParser(description="Topological AE - Scene Decomposer Trainer")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the directory with SOURCE images (JPG, PNG, etc.).")
    parser.add_argument('--basename', type=str, required=True, help="Basename for model files.")
    parser.add_argument('--d-model', type=int, default=64, help="Core model dimension (the efficient 'shader').")
    parser.add_argument('--latent-dim', type=int, default=96, help="Final latent feature dimension (the rich 'G-Buffer').")
    parser.add_argument('--latent-grid-size', type=int, default=96, help="Size of the latent grid.")
    parser.add_argument('--image-size', type=int, default=512, help="Image resolution.")
    parser.add_argument('--epochs', type=int, default=100, help="Total number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size PER DEVICE.")
    parser.add_argument('--lr', type=float, default=2e-4, help="Target learning rate for the Q-Controller.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 mixed precision.")
    parser.add_argument('--save-every-steps', type=int, default=2000, help="Save a checkpoint every N global steps.")
    parser.add_argument('--eval-every', type=int, default=5000, help="Run validation every N global steps.")
    parser.add_argument('--pixels-per-step', type=int, default=8192, help="Number of random pixels to sample for the loss calculation each step.")
    args = parser.parse_args()
    tf.config.set_visible_devices([], 'GPU')
    ImageTrainer(args).train()
if __name__ == "__main__":
    main()