import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import platform
import atexit
import signal
from queue import Queue, Empty
from dataclasses import dataclass, replace
try:
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / ".jax_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at: {cache_dir} ---")
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
import sys
import argparse
import pickle
import time
import threading
import math
from functools import partial
from typing import Any, Tuple, Dict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import optax
import einops
from flax import linen as nn
from flax.training import train_state
from flax.core import freeze, unfreeze
from flax import struct
import chex
from PIL import Image
print("--- Checking Core Dependencies ---")
try:
    from rich_pixels import Pixels; print("✅ [FOUND]   rich_pixels")
except ImportError:
    Pixels = None; print("⚠️ [MISSING] `rich-pixels`. Preview disabled. `pip install rich-pixels`")
dependencies = [("tensorflow","tensorflow"), ("tensorflow_datasets","tensorflow-datasets"), ("rich.console","rich"), ("pynvml","nvidia-ml-py"), ("transformers", "transformers"), ("torch", "torch")]
missing = []
for module, package in dependencies:
    try:
        __import__(module.split('.')[0])
        print(f"✅ [FOUND]   {module}")
    except (ImportError, FutureWarning):
        missing.append(package)
        print(f"❌ [MISSING] {module} (Requires: {package})")
if missing:
    print(f"\n[FATAL] Missing dependencies. Please run: pip install {' '.join(missing)}")
    sys.exit(1)
print("--- All dependencies verified. Proceeding with full imports. ---")
import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
from rich.text import Text
import pynvml; pynvml.nvmlInit()
from tqdm import tqdm
from transformers import SiglipTextModel, SiglipTokenizer
import torch
if platform.system() != "Windows": import select, tty, termios
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False)
class InteractivityState:
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
    print("--- Key listener started. Controls: [←/→] Preview | [s] Save | [q] Quit ---")
    if platform.system() == "Windows": import msvcrt
    else: fd, old_settings = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    try:
        if platform.system() != "Windows": tty.setcbreak(sys.stdin.fileno())
        while not shared_state.shutdown_event.is_set():
            if platform.system() == "Windows":
                if msvcrt.kbhit(): key = msvcrt.getch()
                else: time.sleep(0.05); continue
            else:
                if select.select([sys.stdin], [], [], 0.05)[0]: key = sys.stdin.read(1)
                else: continue
            if key in [b'q', 'q', b'\x03', '\x03']: shared_state.set_shutdown(); break
            elif key in [b's', 's']:
                with shared_state.lock: shared_state.force_save = True
            elif key == b'\xe0' or key == '\x1b':
                arrow = msvcrt.getch() if platform.system() == "Windows" else sys.stdin.read(2)
                if arrow in [b'K', '[D']:
                    with shared_state.lock: shared_state.preview_index_change = -1
                elif arrow in [b'M', '[C']:
                    with shared_state.lock: shared_state.preview_index_change = 1
    finally:
        if platform.system() != "Windows": termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
class PIDLambdaController:
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]]):
        self.targets = targets; self.base_weights = base_weights; self.gains = gains
        self.state = {'integral_error': {k: 0.0 for k in targets.keys()}, 'last_error': {k: 0.0 for k in targets.keys()}}
    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        final_lambdas = {}
        for name, base_weight in self.base_weights.items():
            final_lambdas[name] = float(base_weight)
            if name in self.targets:
                raw_value = last_metrics.get(name);
                if raw_value is None: continue
                try: current_loss = float(raw_value)
                except (TypeError, ValueError): continue
                kp, ki, kd = self.gains[name]; target = self.targets[name]; error = current_loss - target
                self.state['integral_error'][name] += error; self.state['integral_error'][name] = np.clip(self.state['integral_error'][name], -5.0, 5.0)
                derivative = error - self.state['last_error'][name]; adjustment = (kp * error) + (ki * self.state['integral_error'][name]) + (kd * derivative)
                final_lambdas[name] = float(np.clip(self.base_weights[name] * np.exp(adjustment), 0.1, 10.0)); self.state['last_error'][name] = error
        return final_lambdas
    def adjust_target(self, metric_name: str, new_target: float, decay: float = 0.1):
        if metric_name in self.targets: self.targets[metric_name] = (1 - decay) * self.targets[metric_name] + decay * new_target
    def state_dict(self): return {'targets': self.targets, 'base_weights': self.base_weights, 'gains': self.gains, 'state': self.state}
    def load_state_dict(self, data): self.targets = data.get('targets', self.targets); self.state = data.get('state', self.state)
Q_CONTROLLER_CONFIG_DISTILLATION = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 5e-5, "lr_max": 5e-3, "metric_history_len": 500, "loss_min": 0.001, "loss_max": 1.0, "exploration_rate_q": 0.3, "min_exploration_rate": 0.05, "exploration_decay": 0.9998, "trend_window": 100, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": -2.0, "stagnation_penalty": -0.5, "warmup_steps": 500, "warmup_lr_start": 1e-4}
@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class QControllerState:
    q_table: chex.Array; metric_history: chex.Array; trend_history: chex.Array; current_value: jnp.ndarray; exploration_rate: jnp.ndarray
    step_count: jnp.ndarray; last_action_idx: jnp.ndarray; last_reward: jnp.ndarray; status_code: jnp.ndarray
    def tree_flatten(self): return (self.q_table, self.metric_history, self.trend_history, self.current_value, self.exploration_rate, self.step_count, self.last_action_idx, self.last_reward, self.status_code), None
    @classmethod
    def tree_unflatten(cls, aux_data, children): return cls(*children)
def init_q_controller(config):
    return QControllerState(q_table=jnp.zeros((config["q_table_size"], config["num_lr_actions"]), dtype=jnp.float32), metric_history=jnp.full((config["metric_history_len"],), (config["loss_min"] + config["loss_max"]) / 2, dtype=jnp.float32), trend_history=jnp.zeros((config["trend_window"],), dtype=jnp.float32), current_value=jnp.array(config["warmup_lr_start"], dtype=jnp.float32), exploration_rate=jnp.array(config["exploration_rate_q"], dtype=jnp.float32), step_count=jnp.array(0, dtype=jnp.int32), last_action_idx=jnp.array(-1, dtype=jnp.int32), last_reward=jnp.array(0.0, dtype=jnp.float32), status_code=jnp.array(0, dtype=jnp.int32))
@jit
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey):
    config = Q_CONTROLLER_CONFIG_DISTILLATION
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config["warmup_steps"]; new_value = config["warmup_lr_start"] * (1 - alpha) + config["lr_max"] * 0.5 * alpha
        return replace(state, current_value=new_value, step_count=state.step_count + 1, status_code=jnp.array(0))
    def regular_action():
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config["metric_history_len"] - 5, 5)); state_idx = jnp.clip(((metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        explore_key, action_key = jax.random.split(key); action_idx = jax.lax.cond(jax.random.uniform(explore_key) < state.exploration_rate, lambda: jax.random.randint(action_key, (), 0, config["num_lr_actions"]), lambda: jnp.argmax(state.q_table[state_idx]))
        new_value = jnp.clip(state.current_value * jnp.array(config["lr_change_factors"])[action_idx], config["lr_min"], config["lr_max"]); return replace(state, current_value=new_value, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config["warmup_steps"], warmup_action, regular_action)
@jit
def q_controller_update(state: QControllerState, metric_value: float):
    config = Q_CONTROLLER_CONFIG_DISTILLATION
    metric_value_f32 = metric_value.astype(jnp.float32)
    new_metric_history = jnp.roll(state.metric_history, -1).at[-1].set(metric_value_f32); new_trend_history = jnp.roll(state.trend_history, -1).at[-1].set(metric_value_f32)
    def perform_update(st):
        x = jnp.arange(config["trend_window"], dtype=jnp.float32); y = new_trend_history; A = jnp.vstack([x, jnp.ones_like(x)]).T; slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        status_code, reward = jax.lax.cond(slope < -config["improve_threshold"], lambda: (jnp.array(1), abs(slope) * 1000.0), lambda: jax.lax.cond(slope > config["regress_threshold"], lambda: (jnp.array(3), config["regress_penalty"]), lambda: (jnp.array(2), config["stagnation_penalty"])))
        old_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config["metric_history_len"]-5, 5)); last_state_idx = jnp.clip(((old_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        new_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(new_metric_history, config["metric_history_len"]-5, 5)); next_state_idx = jnp.clip(((new_metric_mean - config["loss_min"]) / ((config["loss_max"] - config["loss_min"]) / config["q_table_size"])).astype(jnp.int32), 0, config["q_table_size"] - 1)
        current_q = st.q_table[last_state_idx, st.last_action_idx]; max_next_q = jnp.max(st.q_table[next_state_idx]); new_q = current_q + config["learning_rate_q"] * (reward + config["discount_factor_q"] * max_next_q - current_q)
        new_q_table = st.q_table.at[last_state_idx, st.last_action_idx].set(new_q.astype(st.q_table.dtype)); new_exp_rate = jnp.maximum(config["min_exploration_rate"], st.exploration_rate * config["exploration_decay"])
        return replace(st, q_table=new_q_table, exploration_rate=new_exp_rate, last_reward=reward.astype(st.last_reward.dtype), status_code=status_code)
    can_update = (state.step_count > config["warmup_steps"]) & (state.step_count > config["trend_window"]) & (state.last_action_idx >= 0)
    new_state = jax.lax.cond(can_update, perform_update, lambda s: s, state)
    return replace(new_state, metric_history=new_metric_history, trend_history=new_trend_history)
class DistillationTrainState(train_state.TrainState):
    q_state: QControllerState
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
        x, features = images_rgb, 32; context_vectors = []
        num_downsamples = int(np.log2(self.input_image_size / self.latent_grid_size))
        for i in range(num_downsamples):
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            context_vectors.append(jnp.mean(x, axis=(1, 2))); features *= 2
        context_vector = jnp.concatenate(context_vectors, axis=-1)
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        path_params_raw = nn.Conv(3, (1, 1), name="path_params_head", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params_raw[..., 0]) * jnp.pi; chi_c = nn.tanh(path_params_raw[..., 1]) * (jnp.pi / 4.0); radius = nn.sigmoid(path_params_raw[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1), context_vector
class TopologicalObserver(nn.Module): 
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = path_params_grid.shape; L = H * W; path_params = path_params_grid.reshape(B, L, 3)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta); chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path) + 1e-8
        path_real_mean = jnp.mean(t_co_steps.real, axis=-1); path_real_std = jnp.std(t_co_steps.real, axis=-1); path_imag_mean = jnp.mean(t_co_steps.imag, axis=-1); path_imag_std = jnp.std(t_co_steps.imag, axis=-1)
        complex_measurement = jnp.stack([path_real_mean, path_real_std, path_imag_mean, path_imag_std], axis=-1)
        feature_vectors = nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(complex_measurement); return feature_vectors.reshape(B, H, W, self.d_model)
class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi; return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)
class FiLMLayer(nn.Module):
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x, context):
        context_proj = nn.Dense(x.shape[-1] * 2, name="context_proj", dtype=self.dtype)(context); gamma, beta = context_proj[..., :x.shape[-1]], context_proj[..., x.shape[-1]:]
        return x * (gamma[:, None, :] + 1) + beta[:, None, :]
class CoordinateDecoder(nn.Module): 
    d_model: int; num_freqs: int = 10; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.remat
    def _mlp_block(self, h: jnp.ndarray, context_vector: jnp.ndarray) -> jnp.ndarray:
        film_layer = FiLMLayer(dtype=self.dtype)
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h); h = film_layer(h, context_vector); h = nn.gelu(h)
        return nn.Dense(3, name="mlp_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(h)
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = feature_grid.shape; encoded_coords = PositionalEncoding(self.num_freqs, dtype=self.dtype)(coords)
        pyramid = [feature_grid] + [jax.image.resize(feature_grid, (B, H//(2**i), W//(2**i), self.d_model), 'bilinear') for i in range(1, 3)]; all_sampled_features = []
        for level_grid in pyramid:
            level_shape = jnp.array(level_grid.shape[1:3], dtype=self.dtype); coords_rescaled = (coords + 1) / 2 * (level_shape - 1)
            def sample_one_image_level(single_level_grid):
                grid_chw = single_level_grid.transpose(2, 0, 1); return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw).T
            all_sampled_features.append(jax.vmap(sample_one_image_level)(level_grid))
        concatenated_features = jnp.concatenate(all_sampled_features, axis=-1); encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)
        mlp_input = jnp.concatenate([encoded_coords_tiled, concatenated_features], axis=-1); return nn.tanh(self._mlp_block(mlp_input, context_vector))
def _modulation(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    if shift.ndim == 2:
        shift = shift[:, None, :]
    if scale.ndim == 2:
        scale = scale[:, None, :]
    return x * (1 + scale) + shift
class RMSNorm(nn.Module):
    hidden_size: int
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        rms_x = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.epsilon)
        x = x / rms_x
        scale = self.param('scale', nn.initializers.ones, (self.hidden_size,), self.dtype)
        return x.astype(self.dtype) * scale
class SwiGLUFFNBlock(nn.Module):
    hidden_size: int
    mlp_dim: int
    dropout: float = 0.0
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        mlp_dim_gated = int(2 / 3 * self.mlp_dim)
        x12 = nn.Dense(mlp_dim_gated * 2, dtype=self.dtype, name="w12")(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        gated_act = nn.silu(x1) * x2
        x3 = nn.Dense(self.hidden_size, dtype=self.dtype, name="w3")(gated_act)
        return nn.Dropout(rate=self.dropout)(x3, deterministic=deterministic)
class VisionRotaryEmbedder(nn.Module):
    hidden_size: int
    seq_len: int 
    theta: float = 10000.0
    dtype: Any = jnp.float32
    @staticmethod
    def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
        x = einops.rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x[..., 0], x[..., 1]
        x = jnp.stack((-x2, x1), axis=-1)
        return einops.rearrange(x, '... d r -> ... (d r)')
    def setup(self):
        freqs = 1.0 / (self.theta ** (jnp.arange(0, self.hidden_size, 2, dtype=jnp.float32) / self.hidden_size))
        indices = jnp.arange(self.seq_len, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', indices, freqs)
        freqs = einops.repeat(freqs, '... n -> ... (n r)', r=2)
        self.freqs_cos = jnp.cos(freqs).astype(self.dtype)
        self.freqs_sin = jnp.sin(freqs).astype(self.dtype)
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return t * self.freqs_cos[None, None, :, :] + self.rotate_half(t) * self.freqs_sin[None, None, :, :]
class Attention(nn.Module):
    num_heads: int; hidden_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x: jnp.ndarray, rope: VisionRotaryEmbedder) -> jnp.ndarray:
        B, L, _ = x.shape
        head_dim = self.hidden_size // self.num_heads
        qkv = nn.Dense(self.hidden_size * 3, use_bias=True, dtype=self.dtype, name="qkv")(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, head_dim)
        qkv = jnp.swapaxes(qkv, 1, 3) 
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = rope(q); k = rope(k)
        w = jnp.matmul(q, jnp.swapaxes(k, -2, -1) / math.sqrt(head_dim))
        w = nn.softmax(w, axis=-1)
        x_attn = jnp.matmul(w, v)
        out = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj")(jnp.swapaxes(x_attn, 1, 2).reshape(B, L, self.hidden_size))
        return out
class CrossAttention(nn.Module):
    num_heads: int; hidden_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        B, L_x, _ = x.shape; _, L_c, _ = context.shape; head_dim = self.hidden_size // self.num_heads
        q = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="q_proj")(x)
        kv = nn.Dense(self.hidden_size * 2, use_bias=True, dtype=self.dtype, name="kv_proj")(context)
        q = q.reshape(B, L_x, self.num_heads, head_dim); q = jnp.swapaxes(q, 1, 2)
        k, v = jnp.split(kv, 2, axis=-1)
        k = k.reshape(B, L_c, self.num_heads, head_dim); v = v.reshape(B, L_c, self.num_heads, head_dim)
        k = jnp.swapaxes(k, 1, 2); v = jnp.swapaxes(v, 1, 2)
        w = jnp.matmul(q, jnp.swapaxes(k, -2, -1) / math.sqrt(head_dim)); w = nn.softmax(w, axis=-1)
        x_attn = jnp.matmul(w, v)
        out = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj")(jnp.swapaxes(x_attn, 1, 2).reshape(B, L_x, self.hidden_size))
        return out
class LightningDDTBlock(nn.Module):
    hidden_size: int; num_heads: int; mlp_ratio: float; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, text_context: jnp.ndarray, rope: VisionRotaryEmbedder, deterministic: bool) -> jnp.ndarray:
        adaLN_mod = nn.Sequential([nn.silu, nn.Dense(8 * self.hidden_size, dtype=self.dtype, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)], name="adaLN_mod")(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_cross, _ = jnp.split(adaLN_mod, 8, axis=-1)
        attn_out = Attention(num_heads=self.num_heads, hidden_size=self.hidden_size, dtype=self.dtype)(_modulation(RMSNorm(self.hidden_size, dtype=self.dtype, name="norm1")(x), shift_msa, scale_msa), rope=rope)
        x = x + gate_msa[:, None, :] * attn_out
        cross_attn_out = CrossAttention(num_heads=self.num_heads, hidden_size=self.hidden_size, dtype=self.dtype, name="cross_attn")(RMSNorm(self.hidden_size, dtype=self.dtype, name="norm_cross")(x), context=text_context)
        x = x + gate_cross[:, None, :] * cross_attn_out
        mlp_out = SwiGLUFFNBlock(hidden_size=self.hidden_size, mlp_dim=int(self.hidden_size * self.mlp_ratio), dtype=self.dtype, name="mlp")(_modulation(RMSNorm(self.hidden_size, dtype=self.dtype, name="norm2")(x), shift_mlp, scale_mlp), deterministic=deterministic)
        x = x + gate_mlp[:, None, :] * mlp_out
        return x
class GenerativeStudent(nn.Module):
    latent_grid_size: int; num_body_layers: int; body_width: int; body_heads: int; num_head_layers: int; head_width: int; head_heads: int
    tread_start_layer: int; tread_end_layer: int; tread_selection_rate: float; context_dim: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, command_vector: jnp.ndarray, deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B, L, C = command_vector.shape
        text_context = nn.Dense(self.body_width, dtype=self.dtype, name="text_context_proj")(command_vector)
        s = self.param('spatial_query_embed', nn.initializers.normal(0.02), (1, L, self.body_width), self.dtype)
        s = jnp.repeat(s, B, axis=0)
        c = jnp.mean(text_context, axis=1)
        rope_body = VisionRotaryEmbedder(self.body_width // self.body_heads, L, dtype=self.dtype, name="rope_body")
        is_tread_active = not deterministic and (0 <= self.tread_start_layer < self.tread_end_layer <= self.num_body_layers)
        routed_x, mask = None, None
        for i in range(self.num_body_layers):
            if is_tread_active and i == self.tread_start_layer:
                mask = jax.random.uniform(self.make_rng('dropout'), (s.shape[0], s.shape[1], 1)) > self.tread_selection_rate
                routed_x = s; s = s * mask
            s = LightningDDTBlock(self.body_width, self.body_heads, 4.0, dtype=self.dtype, name=f"body_block_{i}")(s, c, text_context, rope_body, deterministic)
            if is_tread_active and i == self.tread_end_layer - 1:
                s = s + routed_x * (1.0 - mask); routed_x, mask = None, None
        predicted_context_vector = nn.Dense(self.context_dim, name="context_head", dtype=self.dtype)(c)
        x_head = nn.Dense(self.head_width, dtype=self.dtype, name="body_to_head_proj")(s)
        text_context_head = nn.Dense(self.head_width, dtype=self.dtype, name="text_context_head_proj")(text_context)
        c_head = nn.Dense(self.head_width, dtype=self.dtype, name="cond_head_proj")(c)
        rope_head = VisionRotaryEmbedder(self.head_width // self.head_heads, L, dtype=self.dtype, name="rope_head")
        for i in range(self.num_head_layers):
            x_head = LightningDDTBlock(self.head_width, self.head_heads, 4.0, dtype=self.dtype, name=f"head_block_{i}")(x_head, c_head, text_context_head, rope_head, deterministic)
        x_out = RMSNorm(self.head_width, dtype=self.dtype, name="head_out_norm")(x_head)
        start_res = int(np.sqrt(L)); x_out = x_out.reshape(B, start_res, start_res, self.head_width)
        num_upsamples = int(np.log2(self.latent_grid_size / start_res)); channels = self.head_width
        for i in range(num_upsamples):
            channels = max(channels // 2, 64); x_out = nn.ConvTranspose(features=channels, kernel_size=(4, 4), strides=(2, 2), padding='SAME', dtype=self.dtype, name=f"upsample_conv_{i}")(x_out)
            x_out = RMSNorm(channels, dtype=self.dtype, name=f"ln_{i}")(x_out); x_out = nn.gelu(x_out)
        path_params_raw = nn.Conv(features=3, kernel_size=(3, 3), padding='SAME', dtype=self.dtype, name="output_conv")(x_out)
        delta_c = nn.tanh(path_params_raw[..., 0]) * jnp.pi; chi_c = nn.tanh(path_params_raw[..., 1]) * (jnp.pi / 4.0); radius = nn.sigmoid(path_params_raw[..., 2]) * (jnp.pi / 2.0)
        predicted_path_params = jnp.stack([delta_c, chi_c, radius], axis=-1)
        return predicted_path_params, predicted_context_vector
def create_distill_dataset_cache(data_dir: str, text_encoder_id: str):
    console = Console(); source_path = Path(data_dir).resolve(); cache_dir = source_path / ".distill_cache"; cache_dir.mkdir(exist_ok=True); manifest_file = source_path / "distill_manifest.pkl"
    console.print(f"--- 🔍 Scanning for aligned image-text pairs in [cyan]{source_path}[/cyan]... ---"); image_paths = sorted(list(source_path.rglob('*.jpg')) + list(source_path.rglob('*.png')) + list(source_path.rglob('*.webp')))
    aligned_pairs = [(str(img_path), str(img_path.with_suffix('.txt'))) for img_path in image_paths if img_path.with_suffix('.txt').exists()]
    if not aligned_pairs: console.print(f"[bold red]FATAL: No aligned image/.txt pairs found in {source_path}.[/bold red]"); sys.exit(1)
    console.print(f"--- ✅ Found {len(aligned_pairs)} aligned pairs. ---"); console.print(f"--- 🧠 Loading Text Encoder: [yellow]{text_encoder_id}[/yellow]... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"; text_model = SiglipTextModel.from_pretrained(text_encoder_id).to(device); tokenizer = SiglipTokenizer.from_pretrained(text_encoder_id); text_model.eval(); manifest = []
    for img_path_str, txt_path_str in tqdm(aligned_pairs, desc="Creating Embeddings"):
        try:
            with open(txt_path_str, 'r', encoding='utf-8') as f: prompt = f.read().strip()
            embedding_filename = f"{Path(img_path_str).stem}.npy"; embedding_path = cache_dir / embedding_filename
            if not embedding_path.exists():
                with torch.no_grad(): inputs = tokenizer([prompt], padding="max_length", max_length=64, return_tensors="pt").to(device); embedding = text_model(**inputs).last_hidden_state.squeeze(0).cpu().numpy(); np.save(embedding_path, embedding.astype(np.float16))
            manifest.append({'image_path': img_path_str, 'prompt': prompt, 'embedding_path': str(embedding_path)})
        except Exception as e: console.print(f"⚠️ Skipping {img_path_str} due to error: {e}")
    with open(manifest_file, 'wb') as f: pickle.dump(manifest, f); console.print(f"\n--- 🎉 Data preparation complete! Manifest saved to [green]{manifest_file}[/green] ---")
def create_distill_dataset(data_dir: str, image_size: int, batch_size: int, dtype: Any):
    console = Console(); source_path = Path(data_dir).resolve(); manifest_file = source_path / "distill_manifest.pkl"
    if not manifest_file.exists(): console.print(f"[bold red]FATAL: Manifest file not found. Run `prepare-data` first.[/bold red]"); sys.exit(1)
    with open(manifest_file, 'rb') as f: manifest = pickle.load(f); image_paths, embedding_paths, prompts = [item['image_path'] for item in manifest], [item['embedding_path'] for item in manifest], [item['prompt'] for item in manifest]; num_samples = len(manifest)
    console.print(f"--- 💿 Loading {num_samples} samples from manifest file. ---"); ds = tf.data.Dataset.from_tensor_slices((image_paths, embedding_paths, prompts)).shuffle(buffer_size=min(num_samples, 20000)).repeat()
    def _load_embedding_py(path): return np.load(path.numpy().decode('utf-8')).astype(np.float16)
    @tf.function
    def _process_triplet(image_path, embedding_path, prompt):
        img_raw = tf.io.read_file(image_path); img = tf.io.decode_image(img_raw, channels=3, expand_animations=False); img.set_shape([None, None, 3]); img = tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.LANCZOS3); img = tf.cast(img, dtype) / 127.5 - 1.0
        emb = tf.py_function(_load_embedding_py, [embedding_path], tf.float16); emb = tf.reshape(emb, (64, 768)); emb = tf.cast(emb, dtype); return emb, prompt, img
    return ds.map(_process_triplet, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE), num_samples
@partial(jit, static_argnames=('teacher_model', 'student_model'))
def train_step(state: DistillationTrainState, command_vec: jnp.ndarray, gt_images: jnp.ndarray, teacher_model: nn.Module, student_model: nn.Module, teacher_params: Any,
               loss_weights: Dict[str, float], key: chex.PRNGKey):
    """
    Performs a single training step using pure L1 loss for distilling the student model from the teacher.
    This version removes perceptual losses to focus solely on matching the teacher's latent outputs.
    """
    q_key, dropout_key = jax.random.split(key, 2)
    new_q_state_pre_update = q_controller_choose_action(state.q_state, q_key); current_lr = new_q_state_pre_update.current_value
    def combined_loss_fn(student_params, d_key: chex.PRNGKey):
        # Teacher forward pass to get target latent representations (path_params and context_vector).
        target_path_params, teacher_context_vector = teacher_model.apply({'params': teacher_params}, gt_images)
        target_path_params_sg = jax.lax.stop_gradient(target_path_params)
        teacher_context_vector_sg = jax.lax.stop_gradient(teacher_context_vector)

        # Student forward pass to get predicted latent representations.
        predicted_path_params, predicted_context_vector = student_model.apply({'params': student_params}, command_vec, deterministic=False, rngs={'dropout': d_key})
        
        # Calculate pure L1 losses between teacher and student outputs.
        l1_latent_loss = jnp.mean(jnp.abs(predicted_path_params - target_path_params_sg))
        context_loss = jnp.mean(jnp.abs(predicted_context_vector - teacher_context_vector_sg))
        
        # Total loss is a weighted sum of the L1 losses. The PID controller adjusts the weights.
        total_loss_val = (loss_weights.get('l1_latent', 1.0) * l1_latent_loss + 
                          loss_weights.get('context', 0.1) * context_loss)
        
        metrics = {'loss/total': total_loss_val, 'loss/l1_latent': l1_latent_loss, 'loss/context': context_loss}
        return jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), (total_loss_val, metrics))

    (total_loss, metrics), grads = jax.value_and_grad(combined_loss_fn, has_aux=True)(state.params, dropout_key)
    grad_norm = optax.global_norm(grads)
    def apply_update_branch(st, g, lr):
        updates, new_opt_state = st.tx.update(g, st.opt_state, st.params, learning_rate=lr)
        new_params = optax.apply_updates(st.params, updates)
        return new_params, new_opt_state
    def skip_update_branch(st, g, lr): return st.params, st.opt_state
    is_finite_grad = jnp.isfinite(grad_norm)
    new_params, new_opt_state = jax.lax.cond(is_finite_grad, apply_update_branch, skip_update_branch, state, grads, current_lr)
    final_q_state = q_controller_update(new_q_state_pre_update, total_loss)
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, q_state=final_q_state)
    metrics['grad_norm'] = grad_norm
    metrics['is_finite'] = is_finite_grad.astype(jnp.float32)
    return new_state, metrics
class DistillationTrainer:
    def __init__(self, args, physics_model_params):
        self.args = args; self.console = Console(); self.interactive_state = InteractivityState(); self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.teacher_model = PathModulator(latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=self.dtype); self.renderer_observer = TopologicalObserver(d_model=args.d_model, dtype=self.dtype)
        self.renderer_decoder = CoordinateDecoder(d_model=args.d_model, dtype=self.dtype); self.teacher_params = freeze(physics_model_params['modulator']); self.renderer_params = freeze({'observer': physics_model_params['observer'], 'coord_decoder': physics_model_params['coord_decoder']})
        self.console.print("--- ✅ Teacher and Renderer models are configured with FROZEN weights. ---")
        num_downsamples = int(np.log2(self.args.image_size / self.args.latent_grid_size)); self.context_dim = sum([32 * (2**i) for i in range(num_downsamples)])
        self.student_model = GenerativeStudent(latent_grid_size=args.latent_grid_size, num_body_layers=args.num_body_layers, body_width=args.body_width, body_heads=args.body_heads, num_head_layers=args.num_head_layers, head_width=args.head_width, head_heads=args.head_heads, tread_start_layer=args.tread_start_layer, tread_end_layer=args.tread_end_layer, tread_selection_rate=args.tread_selection_rate, context_dim=self.context_dim, dtype=self.dtype)
        self.console.print(f"--- 🧠 Initialized [bold cyan]Cross-Attention Student[/] with TREAD & DiTDH architecture ---"); self.console.print(f"    - Body: {args.num_body_layers} layers, {args.body_width} width"); self.console.print(f"    - Head: {args.num_head_layers} layers, {args.head_width} width"); self.console.print(f"    - Context Dim: {self.context_dim}")
        tread_active = (0 <= args.tread_start_layer < args.tread_end_layer <= args.num_body_layers); self.console.print(f"    - [bold green]TREAD ACTIVE[/]: Layers {args.tread_start_layer}-{args.tread_end_layer-1}, Rate {args.tread_selection_rate*100:.0f}%" if tread_active else "    - [dim]TREAD INACTIVE[/]")
        self.text_encoder_id = "google/siglip-base-patch16-224"; self.SEQUENCE_LENGTH = 64; self.EXPECTED_EMBEDDING_DIM = 768
        self.lambda_controller = PIDLambdaController(targets={'l1_latent': 0.01, 'context': 0.01}, base_weights={'l1_latent': 1.0, 'context': 0.1}, gains={'l1_latent': (2.0, 0.02, 3.0), 'context': (2.0, 0.02, 3.0)})
        self.current_lambdas = self.lambda_controller.base_weights; self.param_count = 0; self.last_metrics = {}; self.steps_per_sec = 0.0; self.ui_lock = threading.Lock(); self.preview_prompt = "..."; self.rendered_preview = None; self.best_val_loss = float('inf')
        self.loss_hists = {'total': deque(maxlen=200), 'l1_latent': deque(maxlen=200), 'context': deque(maxlen=200)}; self.global_step = 0
        ckpt_dir = Path("./checkpoints"); ckpt_dir.mkdir(exist_ok=True); self.ckpt_path = ckpt_dir / f"{self.args.basename}_{self.args.d_model}d_distill.pkl"; self.ckpt_path_best = ckpt_dir / f"{self.args.basename}_{self.args.d_model}d_distill_best.pkl"; self.console.print(f"--- Checkpoints will be saved to: [cyan]{self.ckpt_path.parent.resolve()}[/cyan] ---")
    def _save_checkpoint(self, state: DistillationTrainState, global_step: int, best_val_loss: float, path: Path):
        state_cpu = jax.device_get(state); state_cpu = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16 else x, state_cpu)
        data = {'global_step': global_step, 'best_val_loss': best_val_loss, 'params': state_cpu.params, 'opt_state': state_cpu.opt_state, 'q_state': state_cpu.q_state, 'pid_state': self.lambda_controller.state_dict()}
        with open(path, 'wb') as f: pickle.dump(data, f); self.console.print(f"\n--- 💾 Checkpoint saved to [green]{path.name}[/green] at step {global_step} ---")
    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB",f"{u.gpu}%"
        except: return "N/A","N/A"
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2:return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v):return " "*w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1); return "".join(s[i] for i in indices)
    def _generate_layout(self):
        layout = Layout(); layout.split(Layout(name="header", size=3), Layout(ratio=1, name="main"), Layout(name="footer", size=3)); layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        tread_active = (0 <= self.args.tread_start_layer < self.args.tread_end_layer <= self.args.num_body_layers); tread_status = f"[bold green]ACTIVE ({self.args.tread_start_layer}→{self.args.tread_end_layer-1} @ {self.args.tread_selection_rate*100:.0f}%)[/]" if tread_active else "[dim]INACTIVE[/]"
        header = f"🚀💡 [bold]Distillation w/ Cross-Attention (Pure L1)[/] | Step: {self.global_step} | SPS: [blue]{self.steps_per_sec:.2f}[/] | TREAD: {tread_status}"; layout["header"].update(Panel(Align.center(header)))
        stats_tbl=Table.grid(expand=True,padding=(0,1)); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem/Util",f"[yellow]{mem}[/] / [yellow]{util}[/]")
        grad_norm = self.last_metrics.get('grad_norm', 0.0); is_finite = self.last_metrics.get('is_finite', 1.0)
        grad_status = f"[bold green]OK[/]" if is_finite > 0.5 else f"[bold red]SKIPPED (NaN/Inf)[/]"
        stats_tbl.add_row("Grad Norm", f"[cyan]{grad_norm:.3f}[/] | {grad_status}")
        q_state = self.last_metrics.get('q_state', None)
        if q_state:
            q_status_map = {0: ("[cyan]Warmup[/]","🐣"), 1: ("[green]Improving[/]","😎"), 2: ("[yellow]Stagnated[/]","🤔"), 3: ("[red]Regressing[/]","😠")}; status_str, emoji = q_status_map.get(int(q_state.status_code), ("N/A", "🤖"))
            stats_tbl.add_row("LR (Q-Ctrl)", f"[green]{float(q_state.current_value):.2e}[/] {status_str} {emoji}"); stats_tbl.add_row("Q-Reward", f"{float(q_state.last_reward):+.2f}")
        loss_tbl = Table("Metric", "Value", "λ (Weight)", "Target", title="[bold]📉 L1 Distillation Losses[/]", border_style="cyan");
        for name in ['total', 'l1_latent', 'context']:
            val = self.last_metrics.get(f'loss/{name}', 0.0); weight = self.current_lambdas.get(name, 0.0); target = self.lambda_controller.targets.get(name, 'N/A')
            loss_tbl.add_row(name, f"{val:.4f}", f"{weight:.2f}", f"{target:.3f}" if isinstance(target, float) else target)
        layout["left"].update(Group(Panel(stats_tbl, title="[bold]📊 Core Stats[/]"), loss_tbl))
        sparks = Group(*[Panel(Align.center(f"[cyan]{self._get_sparkline(hist, 50)}[/]"),title=f"{name} Loss History",height=3,border_style="dim") for name, hist in self.loss_hists.items()]); prompt_panel=Panel(Text(self.preview_prompt,justify="center",overflow="fold"),title="[bold]Live Preview Prompt (←/→)[/]",border_style="green")
        img_render=self.rendered_preview or Text("...",justify="center"); img_panel=Panel(Align.center(img_render),title="Generated Image (Live Student)")
        layout["right"].update(Group(sparks,prompt_panel,img_panel)); layout["footer"].update(self.progress); return layout
    @partial(jit, static_argnames=('self', 'resolution'))
    def _generate_preview_jitted(self, student_params, command_vec, resolution):
        coords = jnp.mgrid[-1:1:resolution*1j, -1:1:resolution*1j].transpose(1, 2, 0).reshape(-1, 2)
        predicted_path_params, predicted_context_vector = self.student_model.apply({'params': student_params}, command_vec, deterministic=True)
        feature_grid = self.renderer_observer.apply({'params': self.renderer_params['observer']}, predicted_path_params)
        pixels = self.renderer_decoder.apply({'params': self.renderer_params['coord_decoder']}, feature_grid, predicted_context_vector, coords)
        return pixels.reshape(command_vec.shape[0], resolution, resolution, 3)
    def _update_preview_task(self, student_params_device, preview_data):
        emb, prompt_bytes, _ = preview_data; img_batch = self._generate_preview_jitted(student_params_device, emb, 128); img_batch.block_until_ready()
        local_preview_np = ((np.array(img_batch[0]) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8); prompt_str = prompt_bytes[0].decode('utf-8')
        with self.ui_lock:
            self.preview_prompt = prompt_str
            if Pixels and np.all(np.isfinite(local_preview_np)):
                try: term_w=64; h,w,_=local_preview_np.shape; term_h=int(term_w*(h/w)*0.5); self.rendered_preview=Pixels.from_image(Image.fromarray(local_preview_np).resize((term_w,term_h),Image.LANCZOS))
                except Exception as e: self.rendered_preview=Text(f"[red]Preview Error: {e}[/red]")
    def train(self):
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start(); signal.signal(signal.SIGINT, lambda s, f: self.interactive_state.set_shutdown())
        self.console.print("--- Initializing optimized tf.data pipeline... ---"); REBATCH_SIZE=100
        dataset, num_samples = create_distill_dataset(self.args.data_dir, self.args.image_size, self.args.batch_size * REBATCH_SIZE, self.dtype)
        train_iterator = iter(tfds.as_numpy(dataset)); steps_per_epoch = num_samples // self.args.batch_size; total_steps = self.args.epochs * steps_per_epoch
        self.console.print(f"--- Data pipeline ready. Training for {self.args.epochs} epochs ({total_steps} total steps). ---"); self.console.print(f"--- 🚀 [bold green]PURE L1 MODE ENABLED:[/bold green] Training to match teacher latent outputs directly. ---")
        val_dataset, _ = create_distill_dataset(self.args.data_dir, self.args.image_size, self.args.batch_size, self.dtype); preview_data_buffer = list(val_dataset.take(50).as_numpy_iterator())
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=self.args.lr, b1=0.9, b2=0.98)); start_step=0
        if self.ckpt_path.exists():
            self.console.print(f"--- Resuming from checkpoint: [cyan]{self.ckpt_path}[/cyan] ---")
            with open(self.ckpt_path, 'rb') as f: data=pickle.load(f)
            student_params=data['params']; opt_state_loaded=data['opt_state']; start_step=data.get('global_step',0); self.best_val_loss=float(data.get('best_val_loss',float('inf')))
            q_state = data.get('q_state', init_q_controller(Q_CONTROLLER_CONFIG_DISTILLATION)); self.lambda_controller.load_state_dict(data.get('pid_state', {}))
            state=DistillationTrainState.create(apply_fn=None,params=student_params,tx=optimizer, q_state=q_state); state=state.replace(opt_state=opt_state_loaded)
        else:
            self.console.print("--- No checkpoint found. Initializing new student model... ---")
            key = jax.random.PRNGKey(self.args.seed); dummy_command_vec = jnp.zeros((1,self.SEQUENCE_LENGTH,self.EXPECTED_EMBEDDING_DIM), self.dtype)
            student_params = self.student_model.init({'params': key, 'dropout': key}, dummy_command_vec, deterministic=True)['params']
            state = DistillationTrainState.create(apply_fn=None, params=student_params, tx=optimizer, q_state=init_q_controller(Q_CONTROLLER_CONFIG_DISTILLATION))
        state = jax.tree_util.tree_map(lambda x: x.astype(self.dtype) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x, state)
        jitted_train_step = partial(train_step, teacher_model=self.teacher_model, student_model=self.student_model)
        with self.ui_lock: self.param_count=sum(p.size for p in jax.tree_util.tree_leaves(state.params)); self.global_step=start_step
        self.progress=Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"))
        main_task=self.progress.add_task(f"Epoch {(self.global_step//steps_per_epoch)+1}/{self.args.epochs}",total=total_steps,completed=self.global_step)
        live = Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
        try:
            with ThreadPoolExecutor(max_workers=1) as async_pool:
                active_preview_future = None; self.console.print("--- Compiling JAX functions (one-time cost)... ---")
                (emb_compile, _, img_compile) = next(train_iterator); train_key = jax.random.PRNGKey(self.args.seed); key_1, key_2 = jax.random.split(train_key)
                state, _ = jitted_train_step(state, emb_compile[:self.args.batch_size], img_compile[:self.args.batch_size], teacher_params=self.teacher_params, loss_weights=self.current_lambdas, key=key_1)
                state, _ = jitted_train_step(state, emb_compile[:self.args.batch_size], img_compile[:self.args.batch_size], teacher_params=self.teacher_params, loss_weights=self.current_lambdas, key=key_2)
                self.console.print("--- ✅ Compilation complete. Starting training. ---"); live.start(); last_time, last_ui_update_time = time.time(), 0.0; preview_idx = 0
                while self.global_step < total_steps:
                    if self.interactive_state.shutdown_event.is_set(): break
                    super_command_vecs, _, super_gt_images = next(train_iterator)
                    for i in range(REBATCH_SIZE):
                        if self.global_step >= total_steps or self.interactive_state.shutdown_event.is_set(): break
                        command_vec, gt_images = super_command_vecs[i*self.args.batch_size:(i+1)*self.args.batch_size], super_gt_images[i*self.args.batch_size:(i+1)*self.args.batch_size]
                        time_now=time.time(); self.steps_per_sec=self.args.batch_size/(time_now-last_time+1e-9); last_time=time_now; train_key, step_key = jax.random.split(train_key)
                        self.current_lambdas = self.lambda_controller({k.replace('loss/',''):v for k,v in self.last_metrics.items()})
                        state, metrics = jitted_train_step(state, command_vec, gt_images, teacher_params=self.teacher_params, loss_weights=self.current_lambdas, key=step_key)
                        self.global_step+=1;
                        if self.global_step % steps_per_epoch == 0: self.progress.update(main_task, description=f"Epoch {(self.global_step//steps_per_epoch)+1}/{self.args.epochs}")
                        self.progress.update(main_task, completed=self.global_step)
                        if time_now-last_ui_update_time > 1.0/15.0:
                            metrics_np = jax.device_get(metrics)
                            with self.ui_lock:
                                self.last_metrics={k:v.item() for k,v in metrics_np.items()}; self.last_metrics['q_state'] = jax.device_get(state.q_state)
                                for name, hist in self.loss_hists.items():
                                    if f'loss/{name}' in self.last_metrics and np.isfinite(self.last_metrics[f'loss/{name}']): hist.append(self.last_metrics[f'loss/{name}'])
                            if active_preview_future is None or active_preview_future.done():
                                preview_idx = (preview_idx + self.interactive_state.get_and_reset_preview_change()) % len(preview_data_buffer); p_emb,p_prompt,_=preview_data_buffer[preview_idx]
                                active_preview_future = async_pool.submit(self._update_preview_task, state.params, (p_emb[:1], p_prompt[:1], None))
                            live.update(self._generate_layout(), refresh=True); last_ui_update_time=time_now
                        force_save=self.interactive_state.get_and_reset_force_save()
                        if force_save or (self.global_step > 0 and self.global_step % self.args.eval_every == 0): self._save_checkpoint(state, self.global_step, self.best_val_loss, self.ckpt_path_best if force_save else self.ckpt_path)
        finally:
            live.stop(); self.interactive_state.set_shutdown()
            if key_listener_thread.is_alive(): key_listener_thread.join(timeout=1)
            self.console.print("\n--- Training finished. Saving final model state... ---")
            if 'state' in locals() and hasattr(self, 'global_step'): self._save_checkpoint(state, self.global_step, self.best_val_loss, self.ckpt_path)
def main():
    parser = argparse.ArgumentParser(description="Distill a generative model using pure L1 loss.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--basename', type=str, default='my_modelv3', help="Basename for model files.")
    parent_parser.add_argument('--d-model', type=int, default=64, help="Model dimension of Phase 1/2 AE.")
    parent_parser.add_argument('--latent-grid-size', type=int, default=128, help="Latent grid size of Phase 1/2 AE.")
    parent_parser.add_argument('--image-size', type=int, default=512, help="Image resolution for training data.")
    
    p_prep = subparsers.add_parser("prepare-data", help="Create manifest and pre-compute text embeddings for training.")
    p_prep.add_argument('--data-dir', type=str, required=True, help="Directory containing image-text pairs.")
    p_prep.add_argument('--text-encoder-id', type=str, default="google/siglip-base-patch16-224", help="HuggingFace ID of the text encoder.")
    
    p_train = subparsers.add_parser("train", help="Distill knowledge from a teacher model using pure L1 loss.", parents=[parent_parser])
    p_train.add_argument('--data-dir', type=str, default='coco_prepared_for_phase3', help="Path to the prepared dataset directory.")
    p_train.add_argument('--epochs', type=int, default=500, help="Total number of training epochs.")
    p_train.add_argument('--batch-size', type=int, default=4, help="Per-device batch size.")
    p_train.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the AdamW optimizer.")
    p_train.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    p_train.add_argument('--use-bfloat16', action='store_true', default=True, help="Use BFloat16 for mixed-precision training.")
    p_train.add_argument('--eval-every', type=int, default=1000, help="Save a checkpoint every N steps.")
    p_train.add_argument('--physics-model-path', type=str, default="my_modelv3_64d_512_best.pkl", help="Path to the frozen teacher model (.pkl).")
    
    p_train.add_argument('--num-body-layers', type=int, default=8, help="Number of transformer layers in the student model's body.")
    p_train.add_argument('--body-width', type=int, default=512, help="Hidden dimension of the student model's body.")
    p_train.add_argument('--body-heads', type=int, default=8, help="Number of attention heads in the student model's body.")
    p_train.add_argument('--num-head-layers', type=int, default=4, help="Number of transformer layers in the student model's head.")
    p_train.add_argument('--head-width', type=int, default=512, help="Hidden dimension of the student model's head.")
    p_train.add_argument('--head-heads', type=int, default=8, help="Number of attention heads in the student model's head.")
    p_train.add_argument('--tread-start-layer', type=int, default=-1, help="Start layer for TREAD token routing (inclusive). Set to < 0 to disable.")
    p_train.add_argument('--tread-end-layer', type=int, default=4, help="End layer for TREAD token routing (exclusive). Should be > start_layer.")
    p_train.add_argument('--tread-selection-rate', type=float, default=0.33, help="Fraction of tokens to ROUTE (shortcut) during TREAD.")
    
    args = parser.parse_args()
    if args.command == "train":
        if args.tread_start_layer >= 0 and not (0 <= args.tread_start_layer < args.tread_end_layer <= args.num_body_layers):
            parser.error(f"Invalid TREAD config: must be 0 <= start ({args.tread_start_layer}) < end ({args.tread_end_layer}) <= num_body_layers ({args.num_body_layers}).")
        if not (0.0 < args.tread_selection_rate < 1.0):
            parser.error(f"--tread-selection-rate must be between 0 and 1.")
            
    if args.command == "prepare-data":
        create_distill_dataset_cache(args.data_dir, args.text_encoder_id)
    elif args.command == "train":
        physics_model_path = Path(args.physics_model_path)
        if not physics_model_path.exists():
            Console().print(f"[bold red]FATAL: Pre-trained Physics Model not found: {physics_model_path}[/bold red]"), sys.exit(1)
        with open(physics_model_path, 'rb') as f:
            p_data = pickle.load(f)
        physics_model_params = p_data.get('ema_params', p_data.get('params'))
        trainer = DistillationTrainer(args, physics_model_params)
        try:
            trainer.train()
        except KeyboardInterrupt:
            Console().print("\n--- Training interrupted by user. ---"), sys.exit(0)

if __name__ == "__main__":
    main()