import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
from pathlib import Path
import platform
import atexit
import signal
import sys
import argparse
import pickle
import time
import threading
import math
import copy
from functools import partial
from typing import Any, Tuple, List, Sequence
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state, common_utils
from flax.core import freeze
from flax.jax_utils import replicate, unreplicate
import chex
from PIL import Image
import torch
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
from transformers import SiglipTextModel, SiglipTokenizer
from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
from rich.text import Text
import pynvml; pynvml.nvmlInit()
if platform.system() != "Windows": import select, tty, termios
try:
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / ".jax_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at: {cache_dir} ---")
    def _jax_shutdown():
        print("\n--- Script ending. Waiting for JAX to finalize cache... ---")
        import jax; jax.clear_caches(); print("--- JAX cache finalized. ---")
    atexit.register(_jax_shutdown)
except NameError:
    pass
try:
    from rich_pixels import Pixels
except ImportError:
    Pixels = None; print("⚠️ [MISSING] `rich-pixels`. Preview disabled. `pip install rich-pixels`")
from typing import NamedTuple
def rgb_to_hsl_jax(rgb: chex.Array) -> chex.Array:
    rgb = (rgb + 1.0) / 2.0  
    epsilon = 1e-8
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = jnp.maximum(jnp.maximum(r, g), b)
    cmin = jnp.minimum(jnp.minimum(r, g), b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2.0
    s = jnp.where(delta < epsilon, 0.0, delta / (1.0 - jnp.abs(2.0 * l - 1.0) + epsilon))
    h = jnp.zeros_like(r)
    h = jnp.where(cmax == r, (((g - b) / (delta + epsilon))) % 6.0, h)
    h = jnp.where(cmax == g, (((b - r) / (delta + epsilon)) + 2.0), h)
    h = jnp.where(cmax == b, (((r - g) / (delta + epsilon)) + 4.0), h)
    h = h / 6.0
    h = jnp.where(delta < epsilon, 0.0, h)
    return jnp.stack([h, s, l], axis=-1)
def circular_l1_loss(pred: chex.Array, target: chex.Array) -> chex.Array:
    diff = jnp.abs(pred - target)
    return jnp.minimum(diff, 1.0 - diff)
class ToroidalState(NamedTuple):
  pass
def toroidal_gradient_transform() -> optax.GradientTransformation:
    def init_fn(params: optax.Params) -> ToroidalState:
        return ToroidalState()
    def update_fn(updates: optax.Updates, state: ToroidalState, params: optax.Params | None = None) -> tuple[optax.Updates, ToroidalState]:
        boundary = 2 * jnp.pi
        def wrap_gradient(g: jnp.ndarray) -> jnp.ndarray:
            if g is not None:
                return jnp.mod(g + jnp.pi, boundary) - jnp.pi
            return g
        wrapped_updates = jax.tree_util.tree_map(wrap_gradient, updates)
        return wrapped_updates, state
    return optax.GradientTransformation(init_fn, update_fn)
class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; current_lr: jnp.ndarray
    exploration_rate: jnp.ndarray; step_count: jnp.ndarray; last_action_idx: jnp.ndarray; status_code: jnp.ndarray
@struct.dataclass
class QControllerConfig:
    num_lr_actions: int = 5; lr_change_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1; lr_min: float = 1e-6; lr_max: float = 1e-3
    metric_history_len: int = 100; exploration_rate_q: float = 0.3; min_exploration_rate: float = 0.05; exploration_decay: float = 0.9998
    warmup_steps: int = 2000; warmup_lr_start: float = 1e-7
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
class CoordinateDecoder(nn.Module):
    latent_dim: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        class PositionalEncoding(nn.Module):
            num_freqs: int = 10
            dtype: Any = jnp.float32
            @nn.compact
            def __call__(self, x):
                f = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
                return jnp.concatenate([x] + [fn(x * fr) for fr in f for fn in (jnp.sin, jnp.cos)], -1)
        class FiLMLayer(nn.Module):
            dtype: Any = jnp.float32
            @nn.compact
            def __call__(self, x, context):
                context_proj = nn.Dense(x.shape[-1] * 2, name="context_proj", dtype=self.dtype)(context)
                gamma, beta = context_proj[..., :x.shape[-1]], context_proj[..., x.shape[-1]:]
                return x * (gamma[:, None, :] + 1) + beta[:, None, :]
        B, H, W, _ = feature_grid.shape
        encoded_coords = PositionalEncoding(dtype=self.dtype)(coords)
        pyramid = [feature_grid] + [
            jax.image.resize(feature_grid, (B, H // (2**i), W // (2**i), feature_grid.shape[-1]), 'bilinear')
            for i in range(1, 3)
        ]
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
        h = mlp_input
        film_layer = FiLMLayer(dtype=self.dtype)
        for i in range(4):
            h = nn.Dense(256, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = film_layer(h, context_vector)
            h = nn.gelu(h)
        mlp_output = nn.Dense(4, name="mlp_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(h)
        return jnp.concatenate([nn.tanh(mlp_output[..., :3]), nn.sigmoid(mlp_output[..., 3:4])], axis=-1)
class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_dim: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, latent_dim=self.latent_dim, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(latent_dim=self.latent_dim, name="coord_decoder", dtype=self.dtype)
    def __call__(self, images_rgb, coords):
        path_params, context_vector = self.encode(images_rgb)
        feature_grid = self.observer(path_params)
        return self.coord_decoder(feature_grid, context_vector, coords)
    def encode(self, images_rgb) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.modulator(images_rgb)
    def decode(self, path_params: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        feature_grid = self.observer(path_params)
        return self.coord_decoder(feature_grid, context_vector, coords)
def _modulation(x, shift, scale): return x * (1 + scale[:, None, :]) + shift[:, None, :]
class TimestepEmbedding(nn.Module):
    dim: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, t):
        half = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None, :]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if self.dim % 2:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
        x = nn.Dense(self.dim, dtype=self.dtype)(emb); x = nn.silu(x); x = nn.Dense(self.dim, dtype=self.dtype)(x); return x
class RMSNorm(nn.Module):
    dim: int; eps: float = 1e-6; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        x = x.astype(x_dtype)
        scale = self.param('scale', nn.initializers.ones, self.dim, self.dtype)
        return x * scale
class SwiGLUFFN(nn.Module):
    hidden_size: int; mlp_ratio: float; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        mlp_hidden_size = int(self.hidden_size * self.mlp_ratio)
        ffn_hidden_size = int(2 * mlp_hidden_size / 3)
        w12 = nn.Dense(ffn_hidden_size * 2, dtype=self.dtype, name="w12")(x)
        w1, w2 = jnp.split(w12, 2, axis=-1)
        gated_act = nn.silu(w1) * w2
        w3 = nn.Dense(self.hidden_size, dtype=self.dtype, kernel_init=nn.initializers.zeros, name="w3")(gated_act)
        return w3
class RotaryEmbedding(nn.Module):
    dim: int
    dtype: Any = jnp.float32
    @staticmethod
    def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[-2]
        head_dim = x.shape[-1]
        if self.dim != head_dim:
            raise ValueError(f"RotaryEmbedding dim ({self.dim}) does not match input head_dim ({head_dim})")
        theta = 10000.0
        freqs = 1.0 / (theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs_cis = jnp.einsum("i,j->ij", t, freqs)
        sin, cos = jnp.sin(freqs_cis), jnp.cos(freqs_cis)
        sin = jnp.repeat(sin, 2, axis=-1)
        cos = jnp.repeat(cos, 2, axis=-1)
        return (x * cos) + (self.rotate_half(x) * sin)
class LightningDiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    attention_window_size: int
    mlp_ratio: float = 4.0
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x, c):
        B, L, D = x.shape
        head_dim = self.hidden_size // self.num_heads
        adaLN_params = nn.Sequential([
            nn.silu,
            nn.Dense(6 * self.hidden_size, dtype=self.dtype, kernel_init=nn.initializers.zeros)
        ], name="adaLN_modulation")(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(adaLN_params, 6, axis=-1)
        res_x = x
        x_norm = RMSNorm(self.hidden_size, name="norm1")(x)
        x_mod = _modulation(x_norm, shift_msa, scale_msa)
        num_windows = L // self.attention_window_size
        x_windows = x_mod.reshape(B * num_windows, self.attention_window_size, D)
        qkv = nn.Dense(self.hidden_size * 3, use_bias=True, dtype=self.dtype, name="qkv")(x_windows)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B * num_windows, self.attention_window_size, self.num_heads, head_dim)
        k = k.reshape(B * num_windows, self.attention_window_size, self.num_heads, head_dim)
        v = v.reshape(B * num_windows, self.attention_window_size, self.num_heads, head_dim)
        rope = RotaryEmbedding(dim=head_dim, dtype=self.dtype)
        q, k = rope(q), rope(k)
        attn_out = nn.dot_product_attention(q, k, v, deterministic=False)
        attn_out = attn_out.reshape(B, L, D)
        attn_proj = nn.Dense(self.hidden_size, name="out_proj", kernel_init=nn.initializers.zeros)(attn_out)
        x = res_x + gate_msa[:, None, :] * attn_proj
        res_x = x
        x_norm = RMSNorm(self.hidden_size, name="norm2")(x)
        x_mod = _modulation(x_norm, shift_mlp, scale_mlp)
        mlp_out = SwiGLUFFN(self.hidden_size, self.mlp_ratio, dtype=self.dtype)(x_mod)
        x = res_x + gate_mlp[:, None, :] * mlp_out
        return x
class DenoisingStudent(nn.Module):
    num_layers: int; hidden_size: int; num_heads: int
    latent_grid_size: int; latent_context_dim: int; text_seq_len: int
    attention_window_size: int
    use_remat: bool = False
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, noisy_path_params, noisy_context, text_context, timesteps):
        B, H, W, _ = noisy_path_params.shape
        L = H * W
        t_emb = TimestepEmbedding(self.hidden_size, dtype=self.dtype, name="t_embed")(timesteps)
        text_proj = nn.Dense(self.hidden_size, dtype=self.dtype, name="text_proj")(text_context)
        context_proj = nn.Dense(self.hidden_size, dtype=self.dtype, name="context_proj")(noisy_context)
        path_params_proj = nn.Conv(
            self.hidden_size, kernel_size=(1, 1), name="path_params_proj", dtype=self.dtype
        )(noisy_path_params)
        c = t_emb + context_proj + jnp.mean(text_proj, axis=1)
        x = path_params_proj.reshape(B, L, self.hidden_size)
        pad_len = (self.attention_window_size - L % self.attention_window_size) % self.attention_window_size
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0)))
        Block = nn.remat(LightningDiTBlock) if self.use_remat else LightningDiTBlock
        for i in range(self.num_layers):
            x = Block(
                num_heads=self.num_heads, hidden_size=self.hidden_size,
                attention_window_size=self.attention_window_size,
                dtype=self.dtype, name=f"dit_block_{i}"
            )(x, c)
        x = RMSNorm(self.hidden_size, name="final_norm")(x)
        if pad_len > 0:
            x = x[:, :L, :]
        pred_path_noise_flat = nn.Dense(3, dtype=self.dtype, kernel_init=nn.initializers.zeros, name="out_path")(x)
        pred_path_noise = pred_path_noise_flat.reshape(B, H, W, 3)
        pred_context_noise = jnp.zeros_like(noisy_context)
        return pred_path_noise, pred_context_noise
class InteractivityState:
    def __init__(self):
        self.lock=threading.Lock(); self.preview_idx_change,self.force_save,self.shutdown=0,False,threading.Event()
    def get_preview_change(self):
        with self.lock: c=self.preview_idx_change; self.preview_idx_change=0; return c
    def get_force_save(self):
        with self.lock: s=self.force_save; self.force_save=False; return s
    def set_shutdown(self): self.shutdown.set()
def listen_for_keys(state:InteractivityState):
    print("--- Controls: [←/→] Preview | [s] Save | [q] Quit ---")
    if platform.system()=="Windows": import msvcrt
    else: fd,old=sys.stdin.fileno(),termios.tcgetattr(sys.stdin.fileno())
    try:
        if platform.system()!="Windows": tty.setcbreak(sys.stdin.fileno())
        while not state.shutdown.is_set():
            if (platform.system()=="Windows" and msvcrt.kbhit()) or (platform.system()!="Windows" and select.select([sys.stdin],[],[],0.05)[0]):
                k=msvcrt.getch() if platform.system()=="Windows" else sys.stdin.read(1)
                if k in [b'q','q',b'\x03','\x03']: state.set_shutdown(); break
                elif k in [b's','s']:
                    with state.lock: state.force_save=True
                elif k==b'\xe0' or k=='\x1b':
                    a=msvcrt.getch() if platform.system()=="Windows" else sys.stdin.read(2)
                    with state.lock: state.preview_idx_change = -1 if a in [b'K','[D'] else 1
            else: time.sleep(0.05)
    finally:
        if platform.system()!="Windows": termios.tcsetattr(fd,termios.TCSADRAIN,old)
class StudentTrainState(train_state.TrainState):
    ema_params: Any
    q_controller_state: QControllerState
@partial(jit, static_argnames=['teacher_model'])
def get_clean_latents(teacher_params: Any, images: jnp.ndarray, teacher_model: nn.Module):
    return teacher_model.apply({'params': teacher_params}, images, method=teacher_model.encode)
@partial(jax.pmap, axis_name='devices', donate_argnums=(0,), static_broadcasted_argnums=(5, 6, 8, 9, 10, 11))
def train_step(state: StudentTrainState, images: jnp.ndarray, text_embeds: jnp.ndarray,
               teacher_params: Any, key: chex.PRNGKey,
               student_model: nn.Module, teacher_model: nn.Module,
               alphas_cumprod: jnp.ndarray, num_train_timesteps: int,
               q_config: QControllerConfig, target_lr: float, hsl_loss_weight: float):
    noise_key, time_key, dropout_key, q_key, coord_key = jax.random.split(key, 5)
    B = images.shape[0]
    path_params, context_vector = get_clean_latents(teacher_params, images, teacher_model)
    def loss_fn(params):
        timesteps = jax.random.randint(time_key, (B,), 0, num_train_timesteps)
        path_noise = jax.random.normal(jax.random.fold_in(noise_key, 1), path_params.shape, dtype=path_params.dtype)
        context_noise = jax.random.normal(jax.random.fold_in(noise_key, 2), context_vector.shape, dtype=context_vector.dtype)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        def match_shape(value, broadcast_array):
            return value.reshape(value.shape[0], *((1,) * (broadcast_array.ndim - 1)))
        noisy_path_params = match_shape(sqrt_alpha_prod, path_params) * path_params + \
                            match_shape(sqrt_one_minus_alpha_prod, path_params) * path_noise
        noisy_context_vector = match_shape(sqrt_alpha_prod, context_vector) * context_vector + \
                               match_shape(sqrt_one_minus_alpha_prod, context_vector) * context_noise
        predicted_path_noise, _ = student_model.apply(
            {'params': params}, noisy_path_params, noisy_context_vector,
            text_context=text_embeds, timesteps=timesteps, rngs={'dropout': dropout_key}
        )
        loss_path = jnp.mean(jnp.abs(path_noise - predicted_path_noise))
        pred_x0_path = (noisy_path_params - match_shape(sqrt_one_minus_alpha_prod, path_params) * predicted_path_noise) / match_shape(sqrt_alpha_prod, path_params)
        pred_x0_path = jnp.clip(pred_x0_path, -math.pi, math.pi)
        num_pixels = 64 * 64
        coords = jax.random.uniform(coord_key, (num_pixels, 2), minval=-1.0, maxval=1.0)
        pred_pixels = teacher_model.apply(
            {'params': jax.lax.stop_gradient(teacher_params)},
            path_params=pred_x0_path,
            context_vector=jax.lax.stop_gradient(context_vector),
            coords=coords,
            method=teacher_model.decode
        )
        pred_rgb = pred_pixels[..., :3]
        coords_rescaled = (coords + 1) / 2 * (jnp.array(images.shape[1:3]) - 1)
        def sample_one_image(image, coords_for_one):
            img_chw = image.transpose(2, 0, 1)
            return jax.vmap(lambda c: jax.scipy.ndimage.map_coordinates(c, coords_for_one.T, order=1, mode='reflect'))(img_chw).T
        sampled_gt_pixels = jax.vmap(sample_one_image)(images, jnp.expand_dims(coords_rescaled, 0))
        gt_rgb = sampled_gt_pixels[..., :3]
        pred_hsl = rgb_to_hsl_jax(pred_rgb)
        gt_hsl = rgb_to_hsl_jax(gt_rgb)
        loss_h = jnp.mean(circular_l1_loss(pred_hsl[..., 0], gt_hsl[..., 0]))
        loss_sl = jnp.mean(jnp.abs(pred_hsl[..., 1:] - gt_hsl[..., 1:]))
        loss_hsl = loss_h + loss_sl
        total_loss = loss_path + hsl_loss_weight * loss_hsl
        return total_loss.astype(jnp.float32), {
            'loss/total': total_loss,
            'loss/denoising_l1': loss_path,
            'loss/hsl_perceptual': loss_hsl,
        }
    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    synced_grads = jax.lax.pmean(grads, 'devices')
    grad_norm = optax.global_norm(synced_grads)
    grads_are_valid = jnp.isfinite(grad_norm)
    synced_metrics = jax.lax.pmean(metrics, 'devices')
    synced_metrics['grad_norm'] = grad_norm
    synced_metrics['is_finite'] = grads_are_valid.astype(jnp.float32)
    def apply_updates(state, grads, metrics, q_key):
        new_q_state_pre = q_controller_choose_action(state.q_controller_state, q_key, q_config, target_lr)
        current_lr = new_q_state_pre.current_lr
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, learning_rate=current_lr)
        new_params = optax.apply_updates(state.params, updates)
        new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1 - 0.999), state.ema_params, new_params)
        final_q_state = q_controller_update(new_q_state_pre, metrics['loss/total'], q_config)
        new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, ema_params=new_ema_params, q_controller_state=final_q_state)
        return new_state, current_lr
    def skip_updates(state, grads, metrics, q_key):
        return state, state.q_controller_state.current_lr
    new_state, updated_lr = jax.lax.cond(
        grads_are_valid, apply_updates, skip_updates,
        state, synced_grads, synced_metrics, q_key
    )
    return new_state, synced_metrics, updated_lr, jax.lax.pmean(new_state.q_controller_state.status_code, 'devices')
def _get_and_cache_verified_pairs(source_path: Path, console: Console):
    txt_list_cache_file = source_path / "phase3_file_list_cache.pkl"
    verified_pairs_cache_file = source_path / "phase3_verified_pairs_cache.pkl"
    if txt_list_cache_file.exists():
        console.print(f"--- ⚡ Found .txt file list cache. Loading from [green]{txt_list_cache_file}[/green] ---")
        with open(txt_list_cache_file, 'rb') as f:
            current_txt_files = pickle.load(f)
    else:
        console.print(f"--- 🏃 No .txt list cache. Scanning [cyan]{source_path}[/cyan]... ---")
        current_txt_files = []
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), console=console) as progress:
            task = progress.add_task("[yellow]Scanning for .txt files...", total=None)
            for entry in os.scandir(source_path):
                progress.update(task, advance=1)
                if entry.is_file() and entry.name.endswith('.txt'):
                    current_txt_files.append(entry.path)
        console.print(f"--- ✅ Scan complete. Found {len(current_txt_files)} text files. Caching list... ---")
        with open(txt_list_cache_file, 'wb') as f:
            pickle.dump(current_txt_files, f)
    if not current_txt_files:
        return [], []
    if verified_pairs_cache_file.exists():
        console.print(f"--- ⚡ Found verified pairs cache. Validating... ---")
        with open(verified_pairs_cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        if set(cached_data.get('source_txt_files', [])) == set(current_txt_files):
            console.print(f"--- ✅ Cache is valid! Loading {len(cached_data['all_pairs'])} pairs directly. ---")
            return cached_data['all_pairs'], cached_data['missing_embeddings']
        else:
            console.print(f"--- ⚠️ Dataset changed. Re-verifying all file pairs. ---")
    console.print(f"--- 🤝 Verifying {len(current_txt_files)} file pairs in parallel... ---")
    all_pairs = []
    missing_embeddings = []
    def _process_path(txt_path_str: str):
        txt_path = Path(txt_path_str)
        img_path = next((txt_path.with_suffix(ext) for ext in ['.jpg', '.png', '.webp', '.jpeg'] if txt_path.with_suffix(ext).exists()), None)
        if not img_path: return None
        npy_path = txt_path.with_suffix('.npy')
        pair_data = {'img': str(img_path), 'txt': txt_path_str, 'npy': str(npy_path)}
        missing = None if npy_path.exists() else {'txt': txt_path_str, 'npy': str(npy_path)}
        return (pair_data, missing)
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 4) as executor:
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), console=console) as progress:
            task = progress.add_task("[green]Verifying pairs...", total=len(current_txt_files))
            futures = [executor.submit(_process_path, path_str) for path_str in current_txt_files]
            for future in futures:
                result = future.result()
                if result:
                    pair_data, missing_data = result
                    all_pairs.append(pair_data)
                    if missing_data: missing_embeddings.append(missing_data)
                progress.update(task, advance=1)
    console.print(f"--- ✅ Verification complete. Caching results... ---")
    with open(verified_pairs_cache_file, 'wb') as f:
        pickle.dump({
            'source_txt_files': current_txt_files,
            'all_pairs': all_pairs,
            'missing_embeddings': missing_embeddings
        }, f)
    return all_pairs, missing_embeddings
class GenerativeTrainer:
    def __init__(self, args):
        self.args = args; self.console = Console(); self.interactive_state = InteractivityState()
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.num_devices = jax.local_device_count()
        self.console.print(f"--- 🧑‍🏫 Loading Teacher AE from: [cyan]{args.teacher_model_path}[/cyan] ---")
        with open(args.teacher_model_path, 'rb') as f: teacher_data = pickle.load(f)
        self.teacher_model = TopologicalCoordinateGenerator(
            d_model=args.teacher_d_model, latent_dim=args.teacher_latent_dim,
            latent_grid_size=args.teacher_latent_grid, input_image_size=args.teacher_img_size,
            dtype=self.dtype)
        self.teacher_params = freeze(jax.tree_util.tree_map(lambda x: x.astype(self.dtype), teacher_data['ema_params']))
        self.p_teacher_params = jax.device_put_replicated(self.teacher_params, jax.local_devices())
        self.console.print("--- ✅ Teacher AE loaded and weights are REPLICATED and FROZEN. ---")
        c, f = args.teacher_img_size, 32
        context_dims = []
        while (c // 2) >= args.teacher_latent_grid and (c // 2) > 0:
            context_dims.append(f)
            f *= 2
            c //= 2
        self.latent_context_dim = sum(context_dims)
        self.text_encoder_id="google/siglip-base-patch16-224"; self.SEQ_LEN=64; self.EMBED_DIM=768
        self.student_model = DenoisingStudent(
            num_layers=args.num_body_layers, hidden_size=args.body_width, num_heads=args.body_heads,
            latent_grid_size=args.teacher_latent_grid,
            latent_context_dim=self.latent_context_dim,
            text_seq_len=self.SEQ_LEN, use_remat=args.use_remat,
            attention_window_size=args.attention_window_size,
            dtype=self.dtype)
        self.console.print(f"--- 🧠 Initialized Student (Latent Grid: {args.teacher_latent_grid}x{args.teacher_latent_grid}, Remat: {args.use_remat}, Attn Window: {args.attention_window_size}) ---")
        self.ui_lock = threading.Lock()
        self.ckpt_dir = Path("./student_checkpoints"); self.ckpt_dir.mkdir(exist_ok=True)
        self.eval_dir = self.ckpt_dir / "eval_images"; self.eval_dir.mkdir(exist_ok=True)
        self.ckpt_path = self.ckpt_dir / f"{args.basename}_student.pkl"
        self.best_ckpt_path = self.ckpt_dir / f"{args.basename}_student_best.pkl"
        self.num_train_timesteps = 1000
        beta_start=0.0001; beta_end=0.02
        betas = jnp.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=jnp.float32)
        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(alphas, axis=0)
        self.p_alphas_cumprod_for_training = jax.device_put_replicated(self.alphas_cumprod, jax.local_devices())
        self.dataset_size = 0; self.steps_per_epoch = 0; self.total_steps = 0
        self.loss_hist = deque(maxlen=200)
        self.last_metrics, self.steps_per_sec, self.current_q_lr, self.current_q_status = {}, 0.0, 0.0, 0
        self.preview_prompt, self.rendered_preview = "...", None
        self.best_loss = float('inf')
        self.last_best_save_time = 0
        self.best_save_cooldown = 60
        self.active_best_save_future = None
        self.last_preview_time = 0
        self.q_config = QControllerConfig()
        self.null_text_embed = None
    def _create_dataset(self, aligned_pairs_with_embeddings: List[Tuple[str, str]], apply_augmentations: bool):
        img_paths, npy_paths = zip(*aligned_pairs_with_embeddings)
        ds = tf.data.Dataset.from_tensor_slices((list(img_paths), list(npy_paths)))
        null_text_embed_tf = tf.constant(self.null_text_embed, dtype=tf.float16)
        @tf.function
        def _process(img_path, npy_path):
            img_raw = tf.io.read_file(img_path)
            img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
            img = tf.cast(img, tf.float32)
            if apply_augmentations:
                new_size = int(self.args.teacher_img_size * 1.1)
                img = tf.image.resize(img, [new_size, new_size], method=tf.image.ResizeMethod.LANCZOS3)
                img = tf.image.random_crop(img, size=[self.args.teacher_img_size, self.args.teacher_img_size, 3])
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, max_delta=0.1)
                img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
                img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
                img = tf.clip_by_value(img, 0.0, 255.0)
            else:
                img = tf.image.resize(img, [self.args.teacher_img_size, self.args.teacher_img_size], method=tf.image.ResizeMethod.LANCZOS3)
            img = img / 127.5 - 1.0
            npy_raw = tf.io.read_file(npy_path)
            NPY_HEADER_LEN = 128
            embed_raw = tf.strings.substr(npy_raw, NPY_HEADER_LEN, -1)
            text_embed = tf.io.decode_raw(embed_raw, out_type=tf.float16)
            img.set_shape((self.args.teacher_img_size, self.args.teacher_img_size, 3))
            text_embed.set_shape((self.SEQ_LEN * self.EMBED_DIM,))
            text_embed = tf.reshape(text_embed, (self.SEQ_LEN, self.EMBED_DIM))
            if self.args.cfg_drop_rate > 0 and apply_augmentations:
                should_drop = tf.random.uniform(()) < self.args.cfg_drop_rate
                text_embed = tf.cond(should_drop, lambda: null_text_embed_tf, lambda: text_embed)
            return img, text_embed
        return ds.map(_process, tf.data.AUTOTUNE)
    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{pynvml.nvmlDeviceGetUtilizationRates(h).gpu}%"
        except: return "N/A", "N/A"
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " "*w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)
    def _generate_layout(self, progress, epoch, global_step) -> Layout:
        with self.ui_lock:
            layout=Layout(); layout.split(Layout(name="h",size=3),Layout(name="m"),Layout(name="f",size=3))
            layout["m"].split_row(Layout(name="l"), Layout(name="r"))
            aug_status = "[bold green]ON[/]" if epoch < self.args.aug_curriculum_epochs else "[dim]OFF[/]"
            header_text = f"🚀💡 [bold]Generative Student[/] | Epoch: {epoch+1}/{self.args.epochs} | Augs: {aug_status} | Step: {global_step} | SPS: {self.steps_per_sec:.2f}"
            layout["h"].update(Panel(Align.center(header_text),title="[b]wubumind.ai[/b]"))
            mem, util = self._get_gpu_stats()
            loss = float(self.last_metrics.get('loss/total', 0.0))
            denoising_loss = float(self.last_metrics.get('loss/denoising_l1', 0.0))
            hsl_loss = float(self.last_metrics.get('loss/hsl_perceptual', 0.0))
            best_loss = float(self.best_loss)
            q_status_map = {0: "[cyan]Warmup[/]", 1: "[green]Improving[/]", 2: "[yellow]Stagnated[/]"}
            q_status_text = q_status_map.get(self.current_q_status, "[red]Unknown[/]")
            s_tbl=Table.grid(expand=True); s_tbl.add_row("GPU Mem/Util",f"[y]{mem}[/]/[y]{util}[/]")
            s_tbl.add_row("Total Loss", f"[c]{loss:.4f}[/]")
            s_tbl.add_row("  ├─ Denoise L1", f"[dim]{denoising_loss:.4f}[/dim]")
            s_tbl.add_row("  └─ HSL Perceptual", f"[dim]{hsl_loss:.4f}[/dim]")
            s_tbl.add_row("Best Loss", f"[magenta]{best_loss:.4f}[/]")
            s_tbl.add_row("Learning Rate", f"[green]{self.current_q_lr:.2e}[/]"); s_tbl.add_row("LR Controller", q_status_text)
            spark_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_hist,50)}[/]"),title="Loss History",height=3,border_style="cyan")
            layout["l"].update(Group(Panel(s_tbl,title="[b]📊 Stats[/]"), spark_panel))
            img_panel=Panel(Align.center(self.rendered_preview or Text("...",justify="center")),title="Generated Image (Live)")
            prompt_panel=Panel(Text(self.preview_prompt,justify="center"),title="[b]Live Preview Prompt (←/→)[/b]")
            layout["r"].update(Group(prompt_panel, img_panel))
            layout["f"].update(progress)
            return layout
    def _get_preview_jitted_func(self, resolution, num_steps, cfg_scale):
        @partial(jit, static_argnames=('self',))
        def _generate_preview_jitted(self, student_ema_params, text_embeds, null_text_embed, key, alphas_cumprod):
            B = text_embeds.shape[0]
            uncond_embeds = jnp.repeat(null_text_embed[None, ...], B, axis=0)
            ts = jnp.linspace(1.0, 1.0 / self.num_train_timesteps, num_steps + 1)
            timesteps = (ts * (self.num_train_timesteps - 1)).astype(jnp.int32)
            path_key, ctx_key = jax.random.split(key)
            path_shape = (B, self.args.teacher_latent_grid, self.args.teacher_latent_grid, 3)
            ctx_shape = (B, self.latent_context_dim)
            noisy_path = jax.random.normal(path_key, path_shape, dtype=self.dtype)
            noisy_ctx = jax.random.normal(ctx_key, ctx_shape, dtype=self.dtype)
            def denoise_step(carry, t_and_s):
                (x_path, x_ctx), key = carry
                t_idx, s_idx = t_and_s
                t_val = jnp.repeat(t_idx, B)
                x_path_cfg = jnp.concatenate([x_path, x_path], axis=0)
                x_ctx_cfg = jnp.concatenate([x_ctx, x_ctx], axis=0)
                t_cfg = jnp.concatenate([t_val, t_val], axis=0)
                text_embeds_cfg = jnp.concatenate([text_embeds, uncond_embeds], axis=0)
                pred_path_noise, _ = self.student_model.apply(
                    {'params': student_ema_params}, x_path_cfg, x_ctx_cfg, text_embeds_cfg, t_cfg
                )
                pred_path_noise_cond, pred_path_noise_uncond = jnp.split(pred_path_noise, 2, axis=0)
                guided_path_noise = pred_path_noise_uncond + cfg_scale * (pred_path_noise_cond - pred_path_noise_uncond)
                alpha_t, alpha_s = alphas_cumprod[t_idx], alphas_cumprod[s_idx]
                sigma_t, sigma_s = (1 - alpha_t)**0.5, (1 - alpha_s)**0.5
                pred_x0_path = (x_path - sigma_t * guided_path_noise) / alpha_t
                lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
                lambda_s = jnp.log(alpha_s) - jnp.log(sigma_s)
                h = lambda_t - lambda_s
                next_x_path = (sigma_s / sigma_t) * x_path - (alpha_s * (jnp.exp(-h) - 1)) * pred_x0_path
                next_x_ctx = (alpha_s / alpha_t)**0.5 * x_ctx
                return ((next_x_path.astype(self.dtype), next_x_ctx.astype(self.dtype)), key), None
            t_pairs = jnp.stack([timesteps[:-1], timesteps[1:]], axis=1)
            (final_state, _), _ = jax.lax.scan(denoise_step, ((noisy_path, noisy_ctx), key), t_pairs)
            final_path_params, final_ctx = final_state
            coords=jnp.mgrid[-1:1:resolution*1j,-1:1:resolution*1j].transpose(1,2,0).reshape(-1,2)
            pixels=self.teacher_model.apply({'params':self.teacher_params}, path_params=final_path_params, context_vector=final_ctx, coords=coords, method=self.teacher_model.decode)
            return pixels[...,:3].reshape(B,resolution,resolution,3)
        return _generate_preview_jitted
    def _save_checkpoint(self, state_to_save: StudentTrainState, path: Path, global_step: int, is_best: bool = False):
        data_to_save = {
            'params': jax.device_get(unreplicate(state_to_save.params)),
            'ema_params': jax.device_get(unreplicate(state_to_save.ema_params)),
            'opt_state': jax.device_get(unreplicate(state_to_save.opt_state)),
            'q_controller_state': jax.device_get(unreplicate(state_to_save.q_controller_state)),
            'global_step': global_step, 'best_loss': self.best_loss
        }
        with open(path, 'wb') as f: pickle.dump(data_to_save, f)
        prefix = "🏆 New best model" if is_best else "💾 Checkpoint"
        best_loss_val = float(self.best_loss)
        self.console.print(f"\n--- {prefix} saved at step {global_step} (Loss: {best_loss_val:.4f}) ---")
    def _run_save_best_task(self, unreplicated_state: StudentTrainState, global_step: int):
        self._save_checkpoint(replicate(unreplicated_state), self.best_ckpt_path, global_step, is_best=True)
    def _run_preview_task(self, unreplicated_state: StudentTrainState, eval_samples: List[Tuple], global_step: int):
        prompts = [p for _, _, p in eval_samples]
        text_embeds = np.stack([e for _, e, _ in eval_samples])
        eval_key = jax.random.PRNGKey(global_step)
        jitted_preview_func = self._get_preview_jitted_func(128, self.args.num_inference_steps, self.args.cfg_scale)
        preview_image = jitted_preview_func(
            self, unreplicated_state.ema_params, text_embeds, self.null_text_embed, eval_key,
            self.alphas_cumprod
        )
        preview_image.block_until_ready()
        preview_image_np = ((np.array(preview_image) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)[0]
        if Pixels:
            try:
                term_w=64; h,w,_=preview_image_np.shape; term_h=int(term_w*(h/w)*0.5)
                rendered_preview = Pixels.from_image(Image.fromarray(preview_image_np).resize((term_w,term_h),Image.LANCZOS))
            except Exception as e: rendered_preview = Text(f"[red]Preview Error: {e}[/red]")
        else: rendered_preview = Text("[dim]Preview disabled[/dim]")
        with self.ui_lock:
            self.preview_prompt = prompts[0]; self.rendered_preview = rendered_preview
        Image.fromarray(preview_image_np).save(self.eval_dir / f"preview_step_{global_step:07d}.png")
        self._save_checkpoint(replicate(unreplicated_state), self.ckpt_path, global_step)
    def _precompile_and_warmup(self, p_state: StudentTrainState, dummy_dataset_pairs: List, preview_buffer: List) -> StudentTrainState:
        self.console.print("--- 🔥 Compiling and warming up JAX functions (one-time cost)... ---")
        self.console.print("--- Compiling `train_step`... ---")
        dummy_ds = self._create_dataset(dummy_dataset_pairs, False).batch(self.args.batch_size * self.num_devices)
        dummy_batch_tf = next(iter(dummy_ds))
        dummy_batch_np = (dummy_batch_tf[0].numpy(), dummy_batch_tf[1].numpy())
        p_state, _, _, _ = train_step(p_state, *common_utils.shard(dummy_batch_np), self.p_teacher_params,
                                      common_utils.shard_prng_key(jax.random.PRNGKey(0)),
                                      self.student_model, self.teacher_model,
                                      self.p_alphas_cumprod_for_training,
                                      self.num_train_timesteps,
                                      self.q_config, self.args.lr,
                                      self.args.hsl_loss_weight)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), p_state)
        self.console.print("--- ✅ `train_step` compiled. ---")
        self.console.print("--- Compiling `_generate_preview_jitted`... ---")
        unrep_state = unreplicate(p_state)
        dummy_text_embed = preview_buffer[0][1][None,:]
        dummy_key = jax.random.PRNGKey(0)
        jitted_preview_func = self._get_preview_jitted_func(64, self.args.num_inference_steps, self.args.cfg_scale)
        preview_result = jitted_preview_func(
            self, unrep_state.ema_params, dummy_text_embed, self.null_text_embed, dummy_key,
            self.alphas_cumprod
        )
        preview_result.block_until_ready()
        self.console.print("--- ✅ `_generate_preview_jitted` compiled. ---")
        self.console.print("--- ✅ All functions compiled. Starting training. ---")
        return p_state
    def train(self):
        k_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); k_thread.start()
        source_path = Path(self.args.data_dir).resolve()
        self.console.print(f"--- 🧠 Pre-computing null text embedding... ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_model_for_null = SiglipTextModel.from_pretrained(self.text_encoder_id).to(device)
        tokenizer_for_null = SiglipTokenizer.from_pretrained(self.text_encoder_id)
        with torch.no_grad():
            inputs = tokenizer_for_null([""], padding="max_length", max_length=self.SEQ_LEN, return_tensors="pt").to(device)
            self.null_text_embed = text_model_for_null(**inputs).last_hidden_state.squeeze(0).cpu().numpy().astype(np.float16)
        del text_model_for_null, tokenizer_for_null
        self.console.print("--- ✅ Null text embedding cached. ---")
        all_pairs, missing_embeddings = _get_and_cache_verified_pairs(source_path, self.console)
        if not all_pairs: self.console.print(f"[bold red]FATAL: No valid image/text pairs found in {source_path}.[/bold red]"); sys.exit(1)
        if missing_embeddings:
            self.console.print(f"--- 🧠 Found [yellow]{len(missing_embeddings)}[/] missing text embeddings. Pre-computing now... ---")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            text_model = SiglipTextModel.from_pretrained(self.text_encoder_id).to(device)
            tokenizer = SiglipTokenizer.from_pretrained(self.text_encoder_id)
            text_model.eval()
            with Progress(SpinnerColumn(), *Progress.get_default_columns(), console=self.console) as progress:
                task = progress.add_task("[green]Embedding prompts...", total=len(missing_embeddings))
                for item in missing_embeddings:
                    with open(item['txt'], 'r', encoding='utf-8') as f: prompt = f.read().strip()
                    with torch.no_grad():
                        inputs = tokenizer([prompt], padding="max_length", max_length=self.SEQ_LEN, return_tensors="pt").to(device)
                        embedding = text_model(**inputs).last_hidden_state.squeeze(0).cpu().numpy()
                    np.save(item['npy'], embedding.astype(np.float16))
                    progress.update(task, advance=1)
            del text_model, tokenizer
            self.console.print("--- ✅ All text embeddings are now cached. ---")
        self.dataset_size = len(all_pairs)
        self.steps_per_epoch = self.dataset_size // (self.args.batch_size * self.num_devices)
        self.total_steps = self.args.epochs * self.steps_per_epoch
        self.console.print(f"--- 📊 Dataset: [cyan]{self.dataset_size}[/] samples | [cyan]{self.steps_per_epoch}[/] steps/epoch | [cyan]{self.total_steps}[/] total steps ---")
        aligned_pairs_for_ds = [(p['img'], p['npy']) for p in all_pairs]
        preview_buffer = [(None, np.load(p['npy']), open(p['txt'], 'r').read().strip()) for p in all_pairs[:50]]
        q_state = init_q_controller(self.q_config)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            toroidal_gradient_transform(),
            optax.adamw(self.args.lr, b1=0.9, b2=0.98)
        )
        global_step = 0
        if self.ckpt_path.exists():
            self.console.print(f"--- Resuming from checkpoint: {self.ckpt_path} ---")
            with open(self.ckpt_path, 'rb') as f: data=pickle.load(f)
            params = jax.tree_util.tree_map(lambda x: x.astype(self.dtype), data['params'])
            ema_params = jax.tree_util.tree_map(lambda x: x.astype(self.dtype), data.get('ema_params', params))
            self.console.print("[yellow]⚠️ Re-initializing optimizer state to match current code.[/yellow]")
            opt_state = optimizer.init(params)
            q_state = data.get('q_controller_state', q_state)
            global_step = data.get('global_step', 0)
            self.best_loss = float(data.get('best_loss', float('inf')))
            state = StudentTrainState(step=global_step, apply_fn=None, params=params, tx=optimizer, opt_state=opt_state, ema_params=ema_params, q_controller_state=q_state)
        else:
            self.console.print("--- Initializing new model from scratch ---")
            key=jax.random.PRNGKey(self.args.seed)
            dummy_path = jnp.zeros((1, self.args.teacher_latent_grid, self.args.teacher_latent_grid, 3), self.dtype)
            dummy_ctx = jnp.zeros((1, self.latent_context_dim), self.dtype)
            dummy_text = jnp.zeros((1, self.SEQ_LEN, self.EMBED_DIM), self.dtype)
            dummy_ts = jnp.zeros((1,), jnp.int32)
            params=self.student_model.init({'params':key,'dropout':key}, dummy_path, dummy_ctx, dummy_text, dummy_ts)['params']
            ema_params = copy.deepcopy(params)
            state = StudentTrainState.create(apply_fn=None, params=params, tx=optimizer, ema_params=ema_params, q_controller_state=q_state)
        p_state = replicate(state)
        p_state = self._precompile_and_warmup(p_state, aligned_pairs_for_ds[:self.args.batch_size*self.num_devices], preview_buffer)
        progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeRemainingColumn())
        main_task = progress.add_task("Total Progress", total=self.total_steps, completed=global_step)
        live = Live(self._generate_layout(progress, 0, 0), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
        try:
            live.start()
            start_epoch = global_step // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
            current_preview_idx = 0
            with ThreadPoolExecutor(max_workers=2) as pool:
                active_preview_future = None
                for epoch in range(start_epoch, self.args.epochs):
                    is_aug_phase = epoch < self.args.aug_curriculum_epochs
                    permuted_pairs = np.random.permutation(aligned_pairs_for_ds).tolist()
                    epoch_dataset = self._create_dataset(permuted_pairs, apply_augmentations=is_aug_phase)
                    REBATCH_SIZE = 100
                    super_batch_size = self.args.batch_size * self.num_devices * REBATCH_SIZE
                    epoch_dataset = epoch_dataset.batch(super_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
                    for rebatched_images, rebatched_embeds in tfds.as_numpy(epoch_dataset):
                        if global_step >= self.total_steps or self.interactive_state.shutdown.is_set(): break
                        rebatched_images = rebatched_images.reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, *rebatched_images.shape[1:])
                        rebatched_embeds = rebatched_embeds.reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, *rebatched_embeds.shape[1:])
                        for i in range(REBATCH_SIZE):
                            if global_step >= self.total_steps or self.interactive_state.shutdown.is_set(): break
                            start_time = time.time()
                            train_key = jax.random.PRNGKey(global_step)
                            p_state, metrics, q_lr, q_status = train_step(
                                p_state, common_utils.shard(rebatched_images[i]),
                                common_utils.shard(rebatched_embeds[i]), self.p_teacher_params,
                                common_utils.shard_prng_key(train_key), self.student_model, self.teacher_model,
                                self.p_alphas_cumprod_for_training,
                                self.num_train_timesteps,
                                self.q_config, self.args.lr,
                                self.args.hsl_loss_weight
                            )
                            global_step += 1
                            preview_change = self.interactive_state.get_preview_change()
                            if preview_change != 0:
                                current_preview_idx = (current_preview_idx + preview_change) % len(preview_buffer)
                                self.last_preview_time = 0
                                with self.ui_lock:
                                    self.preview_prompt = preview_buffer[current_preview_idx][2]
                                    self.rendered_preview = Text("🔄 New prompt selected. Generating...", justify="center")
                            metrics_unrep = unreplicate(metrics)
                            now = time.time()
                            with self.ui_lock:
                                self.last_metrics = jax.device_get(metrics_unrep)
                                current_loss = self.last_metrics.get('loss/total', float('inf'))
                                if 'loss/total' in self.last_metrics and np.isfinite(self.last_metrics['loss/total']):
                                    self.loss_hist.append(self.last_metrics['loss/total'])
                                self.current_q_lr = float(unreplicate(q_lr)); self.current_q_status = int(unreplicate(q_status))
                                self.steps_per_sec = 1.0 / (now - start_time + 1e-9)
                                if np.isfinite(current_loss) and float(current_loss) < self.best_loss:
                                    self.best_loss = float(current_loss)
                                    if (now - self.last_best_save_time > self.best_save_cooldown) and \
                                       (self.active_best_save_future is None or self.active_best_save_future.done()):
                                        if self.active_best_save_future: self.active_best_save_future.result()
                                        self.last_best_save_time = now
                                        jax.tree_util.tree_map(lambda x: x.block_until_ready(), p_state)
                                        unrep_state = unreplicate(p_state)
                                        self.active_best_save_future = pool.submit(self._run_save_best_task, unrep_state, global_step)
                            if (now - self.last_preview_time > self.args.preview_every_seconds):
                                if active_preview_future is None or active_preview_future.done():
                                    if active_preview_future: active_preview_future.result()
                                    self.console.print(f"\n--- 🚀 Starting background preview at step {global_step}... ---")
                                    self.last_preview_time = now
                                    jax.tree_util.tree_map(lambda x: x.block_until_ready(), p_state)
                                    unrep_state = unreplicate(p_state)
                                    eval_samples = [preview_buffer[current_preview_idx]]
                                    active_preview_future = pool.submit(self._run_preview_task, unrep_state, eval_samples, global_step)
                            if self.interactive_state.get_force_save():
                                jax.tree_util.tree_map(lambda x: x.block_until_ready(), p_state)
                                self._save_checkpoint(p_state, self.ckpt_path, global_step)
                            progress.update(main_task, completed=global_step, description=f"Epoch {epoch+1}/{self.args.epochs}")
                            live.update(self._generate_layout(progress, epoch, global_step), refresh=True)
                    if self.interactive_state.shutdown.is_set(): break
        finally:
            live.stop(); self.interactive_state.set_shutdown(); k_thread.join(1)
            self.console.print("--- Saving final model... ---")
            if 'p_state' in locals():
                self.console.print("--- Waiting for final step to complete on device... ---")
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), p_state)
                self._save_checkpoint(p_state, self.ckpt_path, global_step)
            self.console.print(f"--- ✅ Final model saved at step {global_step}. ---")
def main():
    parser = argparse.ArgumentParser(description="Train a Generative Student model with high-speed optimizations.")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the directory with SOURCE image-text pairs.")
    parser.add_argument('--basename', type=str, required=True, help="Basename for the student model checkpoint.")
    parser.add_argument('--teacher-model-path', type=str, required=True, help="Path to the trained Teacher AE model (.pkl).")
    parser.add_argument('--teacher-d-model', type=int, default=64, help="d_model of the loaded teacher.")
    parser.add_argument('--teacher-latent-dim', type=int, default=96, help="latent_dim of the loaded teacher.")
    parser.add_argument('--teacher-latent-grid', type=int, default=96, help="latent_grid_size of the loaded teacher.")
    parser.add_argument('--teacher-img-size', type=int, default=512, help="image_size of the loaded teacher.")
    parser.add_argument('--epochs', type=int, default=200, help="Total number of epochs to train for.")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size PER DEVICE.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Target learning rate for the Q-Controller.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 precision.")
    parser.add_argument('--use-remat', action='store_true', help="Use gradient checkpointing to save memory.")
    parser.add_argument('--preview-every-seconds', type=int, default=120, help="Run preview generation and save a checkpoint every N seconds.")
    parser.add_argument('--num-inference-steps', type=int, default=20, help="Number of steps for the DPM-Solver++ sampler during preview.")
    parser.add_argument('--num-body-layers', type=int, default=8, help="Transformer layers in the student model.")
    parser.add_argument('--body-width', type=int, default=512, help="Hidden dimension of the student model.")
    parser.add_argument('--body-heads', type=int, default=8, help="Attention heads in the student model.")
    parser.add_argument('--aug-curriculum-epochs', type=int, default=25, help="Number of initial epochs to train WITH augmentations.")
    parser.add_argument('--attention-window-size', type=int, default=144, help="Size of the attention window for self-attention. Must be a divisor of latent_grid_size^2.")
    parser.add_argument('--cfg-drop-rate', type=float, default=0.15, help="Dropout rate for text conditioning (for CFG). 0.1 = 10% chance to drop.")
    parser.add_argument('--cfg-scale', type=float, default=7.5, help="Classifier-Free Guidance scale.")
    parser.add_argument('--hsl-loss-weight', type=float, default=0.5, help="Weight for the HSL perceptual loss.")
    args = parser.parse_args()
    if (args.teacher_latent_grid ** 2) % args.attention_window_size != 0:
        raise ValueError(f"Attention window size ({args.attention_window_size}) must be a divisor of the total number of latent tokens ({args.teacher_latent_grid**2}).")
    trainer = GenerativeTrainer(args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n--- Training interrupted by user. ---"); sys.exit(0)
if __name__ == "__main__":
    main()