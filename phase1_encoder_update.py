# =================================================================================================
#
#                    PHASE 1: DETERMINISTIC, PHYSICS-INFORMED AUTOENCODER
#
#       (Upgraded with Epoch-Based Training, PID Loss Scaling & Performance Optimizations)
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

# --- Robust and Explicit Cache Path Setup ---
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
from typing import Any, Sequence, Dict, NamedTuple, Optional, Tuple
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
    import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
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
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_index_change = 0
        self.sentinel_dampening_log_factor = 0.0
        self.shutdown_event = threading.Event()
        self.force_save = False
    def get_and_reset_preview_change(self):
        with self.lock: change = self.preview_index_change; self.preview_index_change = 0; return change
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def update_sentinel_factor(self, direction):
        with self.lock: self.sentinel_dampening_log_factor = np.clip(self.sentinel_dampening_log_factor + direction * 0.5, -3.0, 0.0)
    def get_sentinel_factor(self):
        with self.lock: return 10**self.sentinel_dampening_log_factor
    def set_shutdown(self): self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    print("--- Key listener started. Controls: [←/→] Preview | [↑/↓] Sentinel | [s] Force Save | [q] Quit ---")
    if platform.system() == "Windows":
        import msvcrt
        while not shared_state.shutdown_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in [b'q', b'\x03']: shared_state.set_shutdown(); break
                elif key == b's':
                    with shared_state.lock: shared_state.force_save = True
                elif key == b'\xe0':
                    arrow = msvcrt.getch()
                    if arrow == b'K': shared_state.preview_index_change = -1
                    elif arrow == b'M': shared_state.preview_index_change = 1
                    elif arrow == b'H': shared_state.update_sentinel_factor(1)
                    elif arrow == b'P': shared_state.update_sentinel_factor(-1)
            time.sleep(0.05)
    else:
        import select, sys, tty, termios
        fd = sys.stdin.fileno(); old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not shared_state.shutdown_event.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char in ['q', '\x03']: shared_state.set_shutdown(); break
                    elif char == 's':
                        with shared_state.lock: shared_state.force_save = True
                    elif char == '\x1b':
                        next_chars = sys.stdin.read(2)
                        if next_chars == '[A': shared_state.update_sentinel_factor(1)
                        elif next_chars == '[B': shared_state.update_sentinel_factor(-1)
                        elif next_chars == '[C': shared_state.preview_index_change = 1
                        elif next_chars == '[D': shared_state.preview_index_change = -1
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_sentinel_lever_ascii(log_factor: float):
    clipped_factor = np.clip(log_factor, -3.0, 0.0); idx = int((-clipped_factor / 3.0) * 6)
    idx = np.clip(idx, 0, 6); lever_bars = ["│         │"] * 7; lever_bars[idx] = "│█████████│"
    labels = ["1.0 (Off)", " ", "0.1", " ", "0.01", " ", "0.001"]
    return "\n".join([f" {labels[i]:<10} {lever_bars[i]}" for i in range(7)])

class QControllerState(struct.PyTreeNode):
    q_table: chex.Array
    metric_history: chex.Array
    trend_history: chex.Array
    current_lr: jnp.ndarray
    exploration_rate: jnp.ndarray
    step_count: jnp.ndarray
    last_action_idx: jnp.ndarray
    last_reward: jnp.ndarray
    status_code: jnp.ndarray

@dataclass(frozen=True)
class QControllerConfig:
    q_table_size: int = 100
    num_lr_actions: int = 5
    lr_change_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1
    discount_factor_q: float = 0.9
    lr_min: float = 1e-5
    lr_max: float = 8e-4
    metric_history_len: int = 420
    loss_min: float = 0.05
    loss_max: float = 1.0
    exploration_rate_q: float = 0.3
    min_exploration_rate: float = 0.05
    exploration_decay: float = 0.9995
    trend_window: int = 420
    improve_threshold: float = 1e-5
    regress_threshold: float = 1e-6
    regress_penalty: float = 10.0
    stagnation_penalty: float = -2.0
    warmup_steps: int = 420
    warmup_lr_start: float = 1e-6

Q_CONTROLLER_CONFIG = QControllerConfig()

def init_q_controller(config: QControllerConfig, initial_lr):
    return QControllerState(
        q_table=jnp.zeros((config.q_table_size, config.num_lr_actions), dtype=jnp.float32),
        metric_history=jnp.full((config.metric_history_len,), (config.loss_min + config.loss_max) / 2, dtype=jnp.float32),
        trend_history=jnp.zeros((config.trend_window,), dtype=jnp.float32),
        current_lr=jnp.array(config.warmup_lr_start, dtype=jnp.float32),
        exploration_rate=jnp.array(config.exploration_rate_q, dtype=jnp.float32),
        step_count=jnp.array(0, dtype=jnp.int32), last_action_idx=jnp.array(-1, dtype=jnp.int32),
        last_reward=jnp.array(0.0, dtype=jnp.float32), status_code=jnp.array(0, dtype=jnp.int32)
    )

@partial(jax.jit, static_argnames=('config', 'target_lr'))
def q_controller_choose_action(state: QControllerState, key: chex.PRNGKey, config: QControllerConfig, target_lr: float):
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config.warmup_steps
        new_lr = config.warmup_lr_start * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, status_code=jnp.array(0))
    def regular_action():
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config.metric_history_len - 10, 10))
        state_idx = jnp.clip(((metric_mean - config.loss_min) / ((config.loss_max - config.loss_min) / config.q_table_size)).astype(jnp.int32), 0, config.q_table_size - 1)
        explore_key, action_key = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore_key) < state.exploration_rate,
            lambda: jax.random.randint(action_key, (), 0, config.num_lr_actions),
            lambda: jnp.argmax(state.q_table[state_idx]))
        change_factor = jnp.array(config.lr_change_factors)[action_idx]
        new_lr = jnp.clip(state.current_lr * change_factor, config.lr_min, config.lr_max)
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config.warmup_steps, warmup_action, regular_action)

@partial(jax.jit, static_argnames=('config',))
def q_controller_update(state: QControllerState, metric_value: float, config: QControllerConfig):
    state_with_new_history = state.replace(
        metric_history=jnp.roll(state.metric_history, -1).at[-1].set(metric_value),
        trend_history=jnp.roll(state.trend_history, -1).at[-1].set(metric_value)
    )
    def perform_q_table_update(st: QControllerState) -> QControllerState:
        x = jnp.arange(config.trend_window, dtype=jnp.float32); y = st.trend_history
        A = jnp.vstack([x, jnp.ones_like(x)]).T; slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        status_code, reward = jax.lax.cond(slope < -config.improve_threshold,
            lambda: (jnp.array(1), abs(slope) * 1000.0), # Improving
            lambda: jax.lax.cond(slope > config.regress_threshold,
                lambda: (jnp.array(3), -abs(slope) * 1000.0 - config.regress_penalty), # Regressing
                lambda: (jnp.array(2), config.stagnation_penalty))) # Stagnated
        last_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config.metric_history_len - 11, 10))
        last_state_idx = jnp.clip(((last_metric_mean - config.loss_min) / ((config.loss_max - config.loss_min) / config.q_table_size)).astype(jnp.int32), 0, config.q_table_size - 1)
        new_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config.metric_history_len - 10, 10))
        next_state_idx = jnp.clip(((new_metric_mean - config.loss_min) / ((config.loss_max - config.loss_min) / config.q_table_size)).astype(jnp.int32), 0, config.q_table_size - 1)
        current_q = st.q_table[last_state_idx, st.last_action_idx]
        max_next_q = jnp.max(st.q_table[next_state_idx])
        new_q = current_q + config.learning_rate_q * (reward + config.discount_factor_q * max_next_q - current_q)
        return st.replace(
            q_table=st.q_table.at[last_state_idx, st.last_action_idx].set(new_q),
            exploration_rate=jnp.maximum(config.min_exploration_rate, st.exploration_rate * config.exploration_decay),
            last_reward=reward, status_code=status_code)
    can_update = (state_with_new_history.step_count > config.warmup_steps) & (state_with_new_history.step_count > config.trend_window) & (state_with_new_history.last_action_idx >= 0)
    return jax.lax.cond(can_update, perform_q_table_update, lambda s: s, state_with_new_history)

class SentinelState(NamedTuple):
    sign_ema: chex.ArrayTree; dampened_count: Optional[jnp.ndarray] = None; dampened_pct: Optional[jnp.ndarray] = None

def sentinel(decay: float = 0.9, oscillation_threshold: float = 0.5) -> optax.GradientTransformation:
    def init_fn(params):
        sign_ema = jax.tree_util.tree_map(jnp.zeros_like, params)
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
        dampened_updates, num_oscillating, dampened_pct = jax.lax.cond(
            dampening_factor < 1.0, apply_dampening, lambda: (updates, jnp.array(0), jnp.array(0.0)))
        new_state = SentinelState(sign_ema=new_sign_ema, dampened_count=num_oscillating, dampened_pct=dampened_pct)
        return dampened_updates, new_state
    return optax.GradientTransformation(init_fn, update_fn)

class PIDLambdaController:
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]], warmup_steps: int = 500):
        self.targets = targets
        self.base_weights = base_weights
        self.gains = gains
        self.warmup_steps = warmup_steps
        self.state = {
            'integral_error': {k: 0.0 for k in targets.keys()},
            'last_error': {k: 0.0 for k in targets.keys()},
        }
    def __call__(self, last_metrics: Dict[str, float], global_step: int) -> Dict[str, float]:
        if global_step < self.warmup_steps:
            return self.base_weights
        final_lambdas = {}
        for name, base_weight in self.base_weights.items():
            final_lambdas[name] = float(base_weight)
            if name in self.targets:
                metric_key = f'loss/{name}'
                raw_value = last_metrics.get(metric_key)
                if raw_value is None: continue
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
                final_lambdas[name] = float(np.clip(calculated_lambda, 0.1, 20.0))
        return final_lambdas
    def state_dict(self): return self.state
    def load_state_dict(self, state):
        self.state['integral_error'] = state.get('integral_error', {k: 0.0 for k in self.targets.keys()})
        self.state['last_error'] = state.get('last_error', {k: 0.0 for k in self.targets.keys()})

class CustomTrainState(train_state.TrainState):
    ema_params: Any; q_controller_state: Optional[QControllerState] = None
    def apply_gradients(self, *, grads: Any, **kwargs) -> "CustomTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, **kwargs)
        new_params = optax.apply_updates(self.params, updates)
        new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1.0 - 0.999), self.ema_params, new_params)
        known_keys = self.__dataclass_fields__.keys(); filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_keys}
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, ema_params=new_ema_params, **filtered_kwargs)

@partial(jax.jit)
def rgb_to_ycber(image: jnp.ndarray) -> jnp.ndarray:
    rgb_01 = (image + 1.0) / 2.0
    r, g, b = rgb_01[..., 0], rgb_01[..., 1], rgb_01[..., 2]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b
    y_scaled = y * 2.0 - 1.0
    cb_scaled = cb * 2.0
    cr_scaled = cr * 2.0
    return jnp.stack([y_scaled, cb_scaled, cr_scaled], axis=-1)

@partial(jax.jit)
def ycber_to_rgb(image: jnp.ndarray) -> jnp.ndarray:
    y_scaled, cb_scaled, cr_scaled = image[..., 0], image[..., 1], image[..., 2]
    y = (y_scaled + 1.0) / 2.0
    cb = cb_scaled / 2.0
    cr = cr_scaled / 2.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb_01 = jnp.stack([r, g, b], axis=-1)
    return jnp.clip(rgb_01, 0.0, 1.0) * 2.0 - 1.0

# =================================================================================================
# 2. PERCEPTUAL LOSS & MODEL DEFINITIONS
# =================================================================================================
@jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    p = jnp.exp(logp); ent = -jnp.sum(p * logp, axis=-1); diff = logp + ent[..., None]; varent = jnp.sum(p * diff**2, axis=-1)
    return ent, varent

@partial(jax.jit, static_argnames=('max_lag',))
def calculate_autocorrelation_features(patches: jnp.ndarray, max_lag: int = 8) -> jnp.ndarray:
    # --- DEFINITIVE FIX: Force float32 precision for stability ---
    patches_f32 = patches.astype(jnp.float32)
    patches_gray = jnp.mean(patches_f32, axis=-1, keepdims=True)
    patches_centered = patches_gray - jnp.mean(patches_gray, axis=(1, 2), keepdims=True)
    norm_factor = jnp.var(patches_centered, axis=(1, 2), keepdims=True) + 1e-5
    lags_x, lags_y = jnp.arange(1, max_lag + 1), jnp.arange(1, max_lag + 1)
    def _calculate_correlation_at_lag(lag_x, lag_y):
        shifted = jnp.roll(patches_centered, (lag_y, lag_x), axis=(1, 2))
        covariance = jnp.mean(patches_centered * shifted, axis=(1, 2, 3))
        return covariance / (jnp.squeeze(norm_factor) + 1e-8)
    corr_h = jax.vmap(_calculate_correlation_at_lag, in_axes=(0, None))(lags_x, 0)
    corr_v = jax.vmap(_calculate_correlation_at_lag, in_axes=(None, 0))(0, lags_y)
    corr_d = jax.vmap(_calculate_correlation_at_lag, in_axes=(0, 0))(lags_x, lags_y)
    return jnp.concatenate([corr_h.T, corr_v.T, corr_d.T], axis=-1)

def _compute_sobel_magnitude(patches: jnp.ndarray, kernel_x: jnp.ndarray, kernel_y: jnp.ndarray) -> jnp.ndarray:
    # --- DEFINITIVE FIX: Force float32 precision for stability ---
    patches_f32 = patches.astype(jnp.float32)
    mag_channels = []
    k_x_4d, k_y_4d = kernel_x[..., None, None], kernel_y[..., None, None]
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    for i in range(patches_f32.shape[-1]):
        channel_slice = patches_f32[..., i][..., None]
        grad_x = jax.lax.conv_general_dilated(channel_slice, k_x_4d, (1, 1), 'SAME', dimension_numbers=dimension_numbers)
        grad_y = jax.lax.conv_general_dilated(channel_slice, k_y_4d, (1, 1), 'SAME', dimension_numbers=dimension_numbers)
        squared_mag = grad_x**2 + grad_y**2
        mag_channels.append(jnp.squeeze(jnp.sqrt(jnp.maximum(squared_mag, 0.) + 1e-6), axis=-1))
    return jnp.linalg.norm(jnp.stack(mag_channels, axis=-1), axis=-1)

@jit
def calculate_edge_loss(patches1: jnp.ndarray, patches2: jnp.ndarray) -> jnp.ndarray:
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], jnp.float32)
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], jnp.float32)
    mag1 = _compute_sobel_magnitude(patches1, sobel_x, sobel_y)
    mag2 = _compute_sobel_magnitude(patches2, sobel_x, sobel_y)
    return jnp.mean(jnp.abs(mag1 - mag2))

@jit
def calculate_color_covariance_loss(patches1: jnp.ndarray, patches2: jnp.ndarray) -> jnp.ndarray:
    # --- DEFINITIVE FIX: Force float32 precision for stability ---
    p1_f32, p2_f32 = patches1.astype(jnp.float32), patches2.astype(jnp.float32)
    def get_gram_matrix(patches):
        features = patches.reshape(patches.shape[0], -1, patches.shape[-1])
        return jax.vmap(lambda x: x.T @ x)(features) / (features.shape[1] * features.shape[2])
    return jnp.mean(jnp.abs(get_gram_matrix(p1_f32) - get_gram_matrix(p2_f32)))

@jit
def calculate_ssim_loss(patches1: jnp.ndarray, patches2: jnp.ndarray, max_val: float = 2.0) -> jnp.ndarray:
    # --- DEFINITIVE FIX: Force float32 precision for stability ---
    p1_f32, p2_f32 = patches1.astype(jnp.float32), patches2.astype(jnp.float32)
    C1, C2 = (0.01 * max_val)**2, (0.03 * max_val)**2
    p1_gray, p2_gray = jnp.mean(p1_f32, axis=-1), jnp.mean(p2_f32, axis=-1)
    mu1, mu2 = jnp.mean(p1_gray, (1, 2)), jnp.mean(p2_gray, (1, 2))
    var1, var2 = jnp.var(p1_gray, (1, 2)), jnp.var(p2_gray, (1, 2))
    covar = jnp.mean(p1_gray * p2_gray, (1,2)) - (mu1 * mu2)
    numerator = (2 * mu1 * mu2 + C1) * (2 * covar + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
    ssim = numerator / (denominator + 1e-8)
    return jnp.mean(1.0 - ssim)

@partial(jax.jit, static_argnames=('num_moments',))
def calculate_moments(patches, num_moments=4):
    # --- DEFINITIVE FIX: Force float32 precision for stability ---
    patches_f32 = patches.astype(jnp.float32)
    flat = patches_f32.reshape(patches_f32.shape[0], -1, patches_f32.shape[-1])
    mean, var = jnp.mean(flat, axis=1), jnp.var(flat, axis=1)
    if num_moments <= 2: return jnp.concatenate([mean, var], axis=-1)
    std_dev = jnp.sqrt(jnp.maximum(var, 1e-8))
    norm_dev = (flat - mean[:, None, :]) / (std_dev[:, None, :] + 1e-8)
    skew = jnp.mean(norm_dev**3, axis=1)
    if num_moments <= 3: return jnp.concatenate([mean, var, skew], axis=-1)
    kurt = jnp.mean(norm_dev**4, axis=1)
    return jnp.concatenate([mean, var, skew, kurt], axis=-1)

@jit
def fft_magnitude_log(patches):
    # --- DEFINITIVE FIX: Force float32 precision for stability ---
    patches_f32 = patches.astype(jnp.float32)
    return jax.vmap(lambda p: jnp.log(jnp.abs(jnp.fft.fft2(p)) + 1e-5))(patches_f32)

@partial(jax.vmap, in_axes=(0, 0, 0, None, None), out_axes=0)
def _extract_patches_vmapped(image, x_coords, y_coords, patch_size, c): return jax.vmap(lambda x, y: jax.lax.dynamic_slice(image, (y, x, 0), (patch_size, patch_size, c)))(x_coords, y_coords)
class JAXMultiMetricPerceptualLoss:
    def __init__(self, num_patches=64, patch_size=32):
        self.num_patches, self.patch_size = num_patches, patch_size; self._calculate_losses_jit = partial(jax.jit, static_argnames=('batch_size',))(self._calculate_losses)
    def _calculate_losses(self, img1, img2, key, batch_size: int) -> Dict[str, jnp.ndarray]:
        _, h, w, c = img1.shape; key1, key2 = jax.random.split(key)
        x_coords = jax.random.randint(key1, (batch_size, self.num_patches), 0, w - self.patch_size)
        y_coords = jax.random.randint(key2, (batch_size, self.num_patches), 0, h - self.patch_size)
        p1 = _extract_patches_vmapped(img1, x_coords, y_coords, self.patch_size, c).reshape(-1, self.patch_size, self.patch_size, c)
        p2 = _extract_patches_vmapped(img2, x_coords, y_coords, self.patch_size, c).reshape(-1, self.patch_size, self.patch_size, c)
        losses = {}; losses['l1'] = jnp.mean(jnp.abs(p1 - p2)); losses['moment'] = jnp.mean(jnp.abs(calculate_moments(p1) - calculate_moments(p2)))
        losses['fft'] = jnp.mean(jnp.abs(fft_magnitude_log(jnp.mean(p1,-1)) - fft_magnitude_log(jnp.mean(p2,-1)))); losses['autocorr'] = jnp.mean(jnp.abs(calculate_autocorrelation_features(p1) - calculate_autocorrelation_features(p2)))
        losses['edge'] = calculate_edge_loss(p1, p2); losses['color_cov'] = calculate_color_covariance_loss(p1, p2); losses['ssim'] = calculate_ssim_loss(p1, p2)
        return {f'loss/{k}': v for k, v in losses.items()}
    def __call__(self, img1, img2, key): return self._calculate_losses_jit(img1, img2, key, batch_size=img1.shape[0])

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        # --- DEFINITIVE FIX: Force float32 precision for this sensitive physical calculation ---
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2); imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images_ycber: jnp.ndarray) -> jnp.ndarray:
        x, features, current_dim, i = images_ycber, 32, self.input_image_size, 0
        while (current_dim // 2) >= self.latent_grid_size and (current_dim // 2) > 0:
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            features *= 2; current_dim //= 2; i += 1
        if current_dim != self.latent_grid_size:
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        def create_head(name: str, input_features: jnp.ndarray):
            h = nn.Conv(128, (1, 1), name=f"{name}_head_conv1", dtype=self.dtype)(input_features); h = nn.gelu(h)
            def radius_bias_init_head(key, shape, dtype=jnp.float32):
                biases = jnp.zeros(shape, dtype)
                return biases.at[2].set(-1.0)
            params_raw = nn.Conv(3, (1, 1), name=f"{name}_head_out", dtype=self.dtype, bias_init=radius_bias_init_head)(h)
            delta = nn.tanh(params_raw[..., 0]) * jnp.pi
            chi   = nn.tanh(params_raw[..., 1]) * (jnp.pi / 4.0)
            radius = nn.sigmoid(params_raw[..., 2]) * (jnp.pi / 2.0)
            return delta, chi, radius
        delta_y, chi_y, radius_y = create_head("y_luma", x)
        delta_cb, chi_cb, radius_cb = create_head("cb_chroma", x)
        delta_cr, chi_cr, radius_cr = create_head("cr_chroma", x)
        return jnp.stack([
            delta_y, chi_y, radius_y,
            delta_cb, chi_cb, radius_cb,
            delta_cr, chi_cr, radius_cr
        ], axis=-1)

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
            phase_steps = jnp.angle(t_co_steps)
            amplitude_steps = jnp.abs(t_co_steps)
            # --- BULLETPROOF FIX: Replace jnp.std with a manually stabilized version. ---
            # This prevents sqrt(negative_variance) which is a common source of NaNs in mixed precision.
            safe_std = jnp.sqrt(jnp.maximum(0., jnp.var(amplitude_steps, axis=-1)))
            return jnp.stack([
                jnp.mean(phase_steps, axis=-1), jnp.ptp(phase_steps, axis=-1),
                jnp.mean(amplitude_steps, axis=-1), safe_std,
                radius
            ], axis=-1)
        params_y = (path_params[..., 0], path_params[..., 1], path_params[..., 2])
        params_cb = (path_params[..., 3], path_params[..., 4], path_params[..., 5])
        params_cr = (path_params[..., 6], path_params[..., 7], path_params[..., 8])
        stats_y = get_channel_stats(*params_y)
        stats_cb = get_channel_stats(*params_cb)
        stats_cr = get_channel_stats(*params_cr)
        all_stats = jnp.concatenate([stats_y, stats_cb, stats_cr], axis=-1)
        feature_grid = nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(all_stats)
        return feature_grid.reshape(B, H, W, self.d_model)

class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)
class CoordinateDecoder(nn.Module):
    d_model: int; num_freqs: int = 10; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.remat
    def _mlp_block(self, h: jnp.ndarray, mlp_input_for_skip: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = nn.gelu(h)
            if i == self.mlp_depth // 2:
                h = jnp.concatenate([h, mlp_input_for_skip], axis=-1)
        return nn.Dense(3, name="mlp_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(h)
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = feature_grid.shape
        pos_encoder = PositionalEncoding(self.num_freqs, dtype=self.dtype)
        encoded_coords = pos_encoder(coords)
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
        mlp_input = jnp.concatenate([encoded_coords_tiled, concatenated_features], axis=-1)
        output = self._mlp_block(mlp_input, mlp_input)
        return nn.tanh(output)

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(self.d_model, name="coord_decoder", dtype=self.dtype)
    def __call__(self, images_rgb, coords):
        images_ycber = rgb_to_ycber(images_rgb)
        path_params = self.modulator(images_ycber)
        feature_grid = self.observer(path_params)
        pixels_ycber = self.coord_decoder(feature_grid, coords)
        pixels_rgb = ycber_to_rgb(pixels_ycber)
        return pixels_rgb
    def encode(self, images_rgb):
        images_ycber = rgb_to_ycber(images_rgb)
        return self.modulator(images_ycber)
    def decode_from_path_params(self, path_params, coords):
        feature_grid = self.observer(path_params)
        pixels_ycber = self.coord_decoder(feature_grid, coords)
        return ycber_to_rgb(pixels_ycber)

# =================================================================================================
# 3. DATA HANDLING
# =================================================================================================
def prepare_data(image_dir: str):
    base_path=Path(image_dir); record_file=base_path/"data_512x512.tfrecord"; info_file=base_path/"dataset_info.pkl"
    if not record_file.exists():
        image_paths = sorted([p for p in base_path.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
        if not image_paths: print(f"[FATAL] No images found in {image_dir}."), sys.exit(1)
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for path in tqdm(image_paths, "Processing Images"):
                try:
                    img = Image.open(path).convert("RGB").resize((512,512),Image.Resampling.LANCZOS)
                    ex=tf.train.Example(features=tf.train.Features(feature={'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(np.array(img),quality=95).numpy()]))}))
                    writer.write(ex.SerializeToString())
                except Exception as e: print(f"Skipping {path}: {e}")
        with open(info_file, 'wb') as f: pickle.dump({'num_samples': len(image_paths)}, f)
        print(f"✅ Data preparation complete.")
    else: print("✅ TFRecord file already exists.")

def create_dataset(image_dir: str, is_training: bool = True):
    record_file = Path(image_dir)/"data_512x512.tfrecord"; info_file = Path(image_dir)/"dataset_info.pkl"
    if not record_file.exists(): raise FileNotFoundError(f"{record_file} not found. Run 'prepare-data' first.")
    with open(info_file, 'rb') as f: num_samples = pickle.load(f)['num_samples']
    def _parse(proto): return tf.io.decode_jpeg(tf.io.parse_single_example(proto,{'image':tf.io.FixedLenFeature([],tf.string)})['image'], 3)
    ds = tf.data.TFRecordDataset(str(record_file)).map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training: ds = ds.shuffle(1024); ds = ds.repeat()
    return ds, num_samples

def prepare_image_for_eval_tf(img_uint8: tf.Tensor) -> tf.Tensor:
    img = tf.cast(img_uint8, tf.float32) / 255.0; return img * 2.0 - 1.0
def apply_augmentations_tf(img_uint8: tf.Tensor) -> tf.Tensor:
    img = tf.cast(img_uint8, tf.float32) / 255.0; img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1); img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.clip_by_value(img, 0.0, 1.0); return img * 2.0 - 1.0

# =================================================================================================
# 4. TRAINER CLASS
# =================================================================================================
class ImageTrainer:
    def __init__(self, args):
        self.args = args
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=self.dtype)
        self.interactive_state = InteractivityState()
        self.loss_calculator = JAXMultiMetricPerceptualLoss()
        pid_gains = {
            'l1': (0.8, 0.01, 1.0), 'ssim': (1.5, 0.02, 2.0), 'edge': (1.2, 0.01, 1.5),
            'moment': (0.5, 0.005, 0.8), 'color_cov': (0.7, 0.005, 1.0),
            'autocorr': (0.6, 0.005, 0.9), 'fft': (0.4, 0.005, 0.5)
        }
        self.lambda_controller = PIDLambdaController(
            targets={'l1': 0.01, 'ssim': 0.1, 'edge': 0.15, 'moment': 0.15, 'color_cov': 0.02, 'autocorr': 0.15, 'fft': 0.1},
            base_weights={'l1': 1.0, 'ssim': 0.8, 'edge': 1.0, 'moment': 1.0, 'color_cov': 0.9, 'autocorr': 0.3, 'fft': 0.4},
            gains=pid_gains,
            warmup_steps=500
        )
        self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.num_devices = jax.local_device_count()
        self.ui_lock = threading.Lock()
        self.last_metrics = {}
        self.current_lambdas = {}
        self.current_preview_np, self.current_recon_np = None, None
        self.param_count = 0
        self.loss_hist = deque(maxlen=200)
        self.steps_per_sec = 0.0
        self.preview_images_device = None
        self.rendered_original_preview = None
        self.rendered_recon_preview = None
        self.is_validating = False
        self.validation_progress = None

    def _get_gpu_stats(self):
        try:
            h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except Exception: return "N/A", "N/A"

    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

    def _save_checkpoint(self, state, epoch, global_step, best_val_loss, path):
        unrep_state = unreplicate(state)
        data = {
            'epoch': epoch, 'global_step': global_step, 'best_val_loss': best_val_loss,
            'params': jax.device_get(unrep_state.params),
            'ema_params': jax.device_get(unrep_state.ema_params),
            'opt_state': jax.device_get(unrep_state.opt_state),
            'pid_controller_state': self.lambda_controller.state_dict()
        }
        if self.args.use_q_controller and unrep_state.q_controller_state:
            data['q_controller_state'] = jax.device_get(unrep_state.q_controller_state)
        with open(path, 'wb') as f: pickle.dump(data, f)
        console = Console()
        console.print(f"\n--- 💾 Checkpoint saved for epoch {epoch+1} / step {global_step} ---")

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            layout = Layout(name="root")
            bottom_widgets = [self.progress]
            if self.is_validating and self.validation_progress:
                bottom_widgets.append(self.validation_progress)
            layout.split(
                Layout(name="header", size=3),
                Layout(ratio=1, name="main"),
                Layout(Group(*bottom_widgets), name="footer", size=5)
            )
            layout["main"].split_row(Layout(name="left", minimum_size=60), Layout(name="right", ratio=1))
            precision = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"🧠⚡ [bold]Topological AE[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision}"
            layout["header"].update(Panel(Align.center(header_text), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right"))
            stats_tbl = Table.grid(expand=True, padding=(0,1)); stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right")
            mem, util = self._get_gpu_stats()
            stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🚀")
            if self.args.use_q_controller:
                lr_value = self.last_metrics.get('learning_rate', 0.0)
                stats_tbl.add_row("Learning Rate", f"[green]{float(lr_value):.2e}[/]")
            stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]")
            stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            loss_table = Table(show_header=False, box=None); loss_table.add_column(style="cyan", width=10); loss_table.add_column(justify="right", style="white", width=10); loss_table.add_column(justify="right", style="yellow")
            loss_table.add_row("[bold]Metric[/bold]", "[bold]Value[/bold]", "[bold]λ (PID)[/bold]")
            loss_order = ['l1', 'ssim', 'edge', 'moment', 'color_cov', 'autocorr', 'fft']
            for key in loss_order:
                metric_name = f'loss/{key}'; value = self.last_metrics.get(metric_name)
                formatted_value = f"{float(value):.4f}" if value is not None else "---"
                loss_table.add_row(key.title(), formatted_value, f"{self.current_lambdas.get(key, 0.0):.2f}")
            loss_panel = Panel(loss_table, title="[bold]📉 Perceptual Loss[/]", border_style="cyan")
            q_status_code = int(self.last_metrics.get('q_status', 0))
            q_status_str = {0: "[blue]WARMUP", 1: "[green]IMPROVING", 2: "[yellow]STAGNATED", 3: "[red]REGRESSING"}.get(q_status_code, "[dim]N/A[/dim]") if self.args.use_q_controller else "[dim]Disabled[/dim]"
            q_panel = Panel(Align.center(Text(q_status_str, justify="center")), title="[bold]🤖 Q-Controller[/]", border_style="green", height=3)
            sentinel_panel = Panel(Group(Text(f"Dampened: {self.last_metrics.get('sentinel_pct', 0.0):.2%}", justify="center"), Text(get_sentinel_lever_ascii(self.interactive_state.sentinel_dampening_log_factor), justify="center")), title="[bold]🕹️ Sentinel (↑/↓)[/]", border_style="yellow")
            layout["left"].update(Group(Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue"), loss_panel, q_panel, sentinel_panel))
            spark_panel = Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_hist, 60)}[/]"), title=f"Total Loss History", height=3, border_style="cyan")
            preview_content = Text("...", justify="center")
            if self.rendered_original_preview:
                recon_panel = self.rendered_recon_preview or Text("[dim]Waiting...[/dim]", justify="center")
                original_panel = Panel(self.rendered_original_preview, title="Original 📸", border_style="dim")
                reconstruction_panel = Panel(recon_panel, title="Reconstruction ✨", border_style="dim")
                prev_tbl = Table.grid(expand=True, padding=(0,1)); prev_tbl.add_column(); prev_tbl.add_column()
                prev_tbl.add_row(original_panel, reconstruction_panel)
                preview_content = prev_tbl
            right_panel_group = Group(spark_panel, Panel(preview_content, title="[bold]🖼️ Live Preview (←/→)[/]", border_style="green"))
            layout["right"].update(right_panel_group)
            return layout

    @partial(jit, static_argnames=('self', 'resolution'))
    def generate_preview(self, gen_ema_params, preview_image_batch, resolution=128):
        coords = jnp.mgrid[-1:1:resolution*1j, -1:1:resolution*1j].transpose(1, 2, 0).reshape(-1, 2)
        recon_pixels_flat = self.model.apply({'params': gen_ema_params}, preview_image_batch, coords)
        return recon_pixels_flat.reshape(preview_image_batch.shape[0], resolution, resolution, 3)

    def _update_preview_task(self, gen_ema_params, preview_image_batch):
        recon_batch = self.generate_preview(gen_ema_params, preview_image_batch)
        recon_batch.block_until_ready()
        recon_np = np.array(((recon_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        with self.ui_lock:
            self.current_recon_np = recon_np
            if Pixels:
                term_w = 64; h, w, _ = self.current_recon_np.shape; term_h = int(term_w * (h / w) * 0.5)
                recon_img = Image.fromarray(self.current_recon_np).resize((term_w, term_h), Image.LANCZOS)
                self.rendered_recon_preview = Pixels.from_image(recon_img)

    def train(self):
        console = Console()
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
        console.print("--- Setting up data pipeline and epoch structure... ---")
        dataset, num_samples = create_dataset(str(self.args.data_dir), is_training=True)
        dataset = dataset.map(apply_augmentations_tf, num_parallel_calls=tf.data.AUTOTUNE)
        REBATCH_SIZE = 100
        super_batch_size = self.args.batch_size * self.num_devices * REBATCH_SIZE
        dataset = dataset.batch(super_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        train_iterator = iter(tfds.as_numpy(dataset))
        console.print(f"--- 🚀 Performance Mode: Rebatching {REBATCH_SIZE} steps per data load. ---")
        steps_per_epoch = num_samples // (self.args.batch_size * self.num_devices) if self.args.batch_size * self.num_devices > 0 else 0
        total_steps = self.args.epochs * steps_per_epoch
        console.print(f"📈 Training for {self.args.epochs} epochs ({total_steps} total steps).")
        val_dataset, _ = create_dataset(str(self.args.data_dir), is_training=False)
        val_dataset = val_dataset.map(prepare_image_for_eval_tf, num_parallel_calls=tf.data.AUTOTUNE).batch(self.args.batch_size * self.num_devices, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        console.print("--- Pre-loading data for asynchronous previews... ---")
        preview_buffer_host = [img for img in create_dataset(str(self.args.data_dir), is_training=False)[0].take(50).as_numpy_iterator()]
        self.preview_images_device = jax.device_put(np.stack([prepare_image_for_eval_tf(img).numpy() for img in preview_buffer_host]))
        current_preview_idx = 0
        self.current_preview_np = ((preview_buffer_host[current_preview_idx] / 255.0) * 255).astype(np.uint8)
        if Pixels:
            term_w = 64; h, w, _ = self.current_preview_np.shape; term_h = int(term_w*(h/w)*0.5)
            orig_img = Image.fromarray(self.current_preview_np).resize((term_w, term_h), Image.LANCZOS)
            self.rendered_original_preview = Pixels.from_image(orig_img)
        H, W = self.args.image_size, self.args.image_size
        full_coords_host = jnp.mgrid[-1:1:H*1j, -1:1:W*1j].transpose(1, 2, 0).reshape(-1, 2)
        p_full_coords = replicate(full_coords_host)
        train_key = jax.random.PRNGKey(self.args.seed)
        optimizer_components = []
        if self.args.use_sentinel:
            optimizer_components.append(sentinel())
        adamw_constructor = optax.inject_hyperparams(optax.adamw)
        adamw_instance = adamw_constructor(learning_rate=self.args.lr, b1=0.9, b2=0.95)
        optimizer_components.append(adamw_instance)
        optimizer = optax.chain(*optimizer_components)
        start_epoch, global_step, best_val_loss = 0, 0, float('inf')
        ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        ckpt_path_best = Path(f"{self.args.basename}_{self.args.d_model}d_512_best.pkl")
        if ckpt_path.exists():
            console.print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path, 'rb') as f: data = pickle.load(f)
            params = data['params']; ema_params = data.get('ema_params', params)
            q_state = None
            if self.args.use_q_controller:
                saved_q_state_data = data.get('q_controller_state')
                if saved_q_state_data:
                    is_new_format = hasattr(saved_q_state_data, 'status_code') and isinstance(getattr(saved_q_state_data, 'status_code', None), jnp.ndarray)
                    if isinstance(saved_q_state_data, QControllerState) and is_new_format:
                        q_state = saved_q_state_data
                    else:
                        console.print("[yellow]-- Incompatible or old Q-Controller state format detected. Re-initializing. --[/yellow]")
                        q_state = init_q_controller(Q_CONTROLLER_CONFIG, self.args.lr)
                else:
                    console.print("[yellow]-- Q-controller state not found in checkpoint, initializing a new one. --[/yellow]")
                    q_state = init_q_controller(Q_CONTROLLER_CONFIG, self.args.lr)
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=ema_params, q_controller_state=q_state).replace(opt_state=data['opt_state'])
            start_epoch = data.get('epoch', 0); global_step = data.get('global_step', start_epoch * steps_per_epoch)
            best_val_loss = data.get('best_val_loss', float('inf'))
            if 'pid_controller_state' in data: self.lambda_controller.load_state_dict(data['pid_controller_state'])
        else:
            console.print("--- Initializing new model ---")
            with jax.default_device(CPU_DEVICE):
                dummy_images=jnp.zeros((1,512,512,3),self.dtype); dummy_coords=jnp.zeros((1024,2),self.dtype)
                params = self.model.init(jax.random.PRNGKey(0), dummy_images, dummy_coords)['params']
            q_state = init_q_controller(Q_CONTROLLER_CONFIG, self.args.lr) if self.args.use_q_controller else None
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=params, q_controller_state=q_state)
        if self.args.use_q_controller and state.q_controller_state is not None:
            console.print(f"--- Synchronizing Q-Controller step count to global_step: {global_step} ---")
            synced_q_state = state.q_controller_state.replace(step_count=jnp.array(global_step, dtype=jnp.int32))
            state = state.replace(q_controller_state=synced_q_state)
        p_state = replicate(state)
        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, unreplicate(state.params), 0)

        @partial(jax.pmap, axis_name='devices', in_axes=(0, 0, 0, None, 0))
        def train_step_and_grad(state, batch, coords, lambdas_tuple, key):
            def loss_fn(params):
                recon_full_flat = self.model.apply({'params': params}, batch, coords)
                recon_full = recon_full_flat.reshape(batch.shape)
                loss_key, _ = jax.random.split(key)
                metrics = self.loss_calculator(batch, recon_full, loss_key)
                lambda_l1, lambda_ssim, lambda_edge, lambda_moment, lambda_color_cov, lambda_autocorr, lambda_fft = lambdas_tuple
                total_loss = (lambda_l1 * metrics['loss/l1'] +
                              lambda_ssim * metrics['loss/ssim'] +
                              lambda_edge * metrics['loss/edge'] +
                              lambda_moment * metrics['loss/moment'] +
                              lambda_color_cov * metrics['loss/color_cov'] +
                              lambda_autocorr * metrics['loss/autocorr'] +
                              lambda_fft * metrics['loss/fft'])
                metrics['loss/total_raw'] = total_loss
                return total_loss, metrics
            f32_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), state.params)
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(f32_params)
            grads = optax.clip_by_global_norm(1.0).update(grads, None, None)[0]
            grad_leaves = jax.tree_util.tree_leaves(grads)
            are_grads_stable = jnp.all(jnp.array([jnp.all(jnp.isfinite(g)) for g in grad_leaves]))
            is_loss_stable = jnp.isfinite(loss)
            is_stable = are_grads_stable & is_loss_stable
            grads_if_stable = grads
            grads_if_unstable = jax.tree_util.tree_map(jnp.zeros_like, state.params)
            final_grads = jax.lax.cond(is_stable, lambda: grads_if_stable, lambda: grads_if_unstable)
            metrics['stability/is_stable'] = is_stable.astype(jnp.float32)
            final_grads = jax.lax.pmean(final_grads, 'devices')
            metrics = jax.lax.pmean(metrics, 'devices')
            return final_grads, metrics

        # --- FIX: `update_step` now returns the learning rate for logging ---
        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,), in_axes=(0, 0, None, 0))
        def update_step(state, grads, damp_factor, key):
            apply_grad_kwargs = {'dampening_factor': damp_factor}
            new_q_state = state.q_controller_state
            lr_for_metric = jnp.array(self.args.lr) # Default LR if Q-Controller is off

            if self.args.use_q_controller and state.q_controller_state is not None:
                q_state_slice = q_controller_choose_action(state.q_controller_state, key, Q_CONTROLLER_CONFIG, self.args.lr)
                apply_grad_kwargs['learning_rate'] = q_state_slice.current_lr
                new_q_state = q_state_slice
                lr_for_metric = new_q_state.current_lr

            new_state = state.apply_gradients(grads=grads, **apply_grad_kwargs)

            if self.args.use_q_controller:
                new_state = new_state.replace(q_controller_state=new_q_state)
            return new_state, lr_for_metric

        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,), in_axes=(0, 0))
        def q_update_step(state, total_loss):
            safe_loss = jnp.nan_to_num(total_loss)
            if self.args.use_q_controller and state.q_controller_state is not None:
                new_q_state = q_controller_update(state.q_controller_state, safe_loss, Q_CONTROLLER_CONFIG)
                # Also log the status code from the update
                q_status = new_q_state.status_code
                return state.replace(q_controller_state=new_q_state), q_status
            return state, jnp.array(-1) # Return dummy status

        @partial(jax.pmap, axis_name='devices', in_axes=(0, 0, 0, None))
        def eval_step(params, batch, coords, lambdas_tuple):
            recon_flat = self.model.apply({'params': params}, batch, coords); recon = recon_flat.reshape(batch.shape)
            metrics = self.loss_calculator(batch, recon, jax.random.PRNGKey(0))
            lambda_l1, lambda_ssim, lambda_edge, lambda_moment, lambda_color_cov, lambda_autocorr, lambda_fft = lambdas_tuple
            total_loss = (lambda_l1 * metrics['loss/l1'] + lambda_ssim * metrics['loss/ssim'] + lambda_edge * metrics['loss/edge'] + lambda_moment * metrics['loss/moment'] + lambda_color_cov * metrics['loss/color_cov'] + lambda_autocorr * metrics['loss/autocorr'] + lambda_fft * metrics['loss/fft'])
            return jax.lax.pmean(total_loss, 'devices')

        console.print("--- Compiling JAX functions (one-time cost)... ---")
        try:
            compile_super_batch_np = next(train_iterator)
        except StopIteration:
            console.print("[bold red]FATAL: Dataset is too small for one super-batch. Decrease REBATCH_SIZE or add more data.[/bold red]")
            return
        compile_super_batch = compile_super_batch_np.reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, self.args.image_size, self.args.image_size, 3)
        compile_batch_np = compile_super_batch[0]
        sharded_compile_batch = common_utils.shard(compile_batch_np.astype(self.dtype)); compile_key = jax.random.split(train_key, self.num_devices)
        lambda_keys_in_order = ['l1', 'ssim', 'edge', 'moment', 'color_cov', 'autocorr', 'fft']
        compile_lambdas_dict = self.lambda_controller(self.last_metrics, global_step)
        compile_lambdas_tuple = tuple(compile_lambdas_dict[key] for key in lambda_keys_in_order)
        compile_grads, compile_metrics = train_step_and_grad(p_state, sharded_compile_batch, p_full_coords, compile_lambdas_tuple, compile_key)
        p_state, _ = update_step(p_state, compile_grads, 1.0, compile_key)
        p_state, _ = q_update_step(p_state, compile_metrics['loss/total_raw'])
        eval_step(p_state.ema_params, sharded_compile_batch, p_full_coords, compile_lambdas_tuple).block_until_ready()
        self.generate_preview(unreplicate(p_state.ema_params), self.preview_images_device[0:1]).block_until_ready()
        console.print("--- Compilation complete. Starting training. ---")
        spinner_column = TextColumn("🍓", style="magenta")
        self.progress = Progress(spinner_column, TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[val_loss]:.4f}[/]"), BarColumn(), "•", TextColumn("Step {task.fields[step]}/{task.fields[steps_per_epoch]}"), "•", TimeRemainingColumn())
        epoch_task = self.progress.add_task("epochs", total=self.args.epochs, completed=start_epoch, val_loss=best_val_loss, step=global_step % steps_per_epoch if steps_per_epoch > 0 else 0, steps_per_epoch=steps_per_epoch)
        last_step_time, last_ui_update_time = time.time(), time.time()
        UI_REFRESH_RATE, SYNC_EVERY_N_STEPS = 15.0, 100
        live = Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
        try:
            live.start()
            with ThreadPoolExecutor(max_workers=1) as async_pool:
                active_preview_future = None
                while global_step < total_steps:
                    if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                    try:
                        super_batch_np = next(train_iterator)
                    except StopIteration:
                        console.print("[yellow]Data iterator exhausted. This shouldn't happen with .repeat(). Exiting.[/yellow]")
                        break
                    super_batch = super_batch_np.reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, self.args.image_size, self.args.image_size, 3)
                    for batch_np in super_batch:
                        if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                        spinner_column.style = "magenta" if (global_step // 2) % 2 == 0 else "blue"
                        sharded_batch = common_utils.shard(batch_np.astype(self.dtype)); train_key, step_key = jax.random.split(train_key)
                        sharded_keys = jax.random.split(step_key, self.num_devices)
                        self.current_lambdas = self.lambda_controller(self.last_metrics, global_step)
                        lambdas_for_jit = tuple(self.current_lambdas[key] for key in lambda_keys_in_order)

                        grads, metrics = train_step_and_grad(p_state, sharded_batch, p_full_coords, lambdas_for_jit, sharded_keys)
                        damp_factor = self.interactive_state.get_sentinel_factor()
                        # --- FIX: Capture the returned learning rate ---
                        p_state, current_lr_metric = update_step(p_state, grads, damp_factor, sharded_keys)
                        metrics['learning_rate'] = current_lr_metric # Add to metrics dict

                        is_stable = unreplicate(metrics['stability/is_stable']) == 1.0
                        if is_stable:
                           # --- FIX: Capture the q_status metric ---
                           p_state, q_status_metric = q_update_step(p_state, metrics['loss/total_raw'])
                           metrics['q_status'] = q_status_metric

                        if global_step > 0 and global_step % SYNC_EVERY_N_STEPS == 0: jax.tree_util.tree_map(lambda x: x.block_until_ready(), p_state)
                        global_step += 1
                        current_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
                        step_in_epoch = global_step % steps_per_epoch if steps_per_epoch > 0 else 0
                        time_now = time.time(); self.steps_per_sec = 1.0/(time_now - last_step_time + 1e-6); last_step_time = time_now
                        self.progress.update(epoch_task, completed=current_epoch, step=step_in_epoch + 1)

                        if (time_now - last_ui_update_time) > (1.0 / UI_REFRESH_RATE):
                            metrics_unrep = unreplicate(metrics)
                            with self.ui_lock:
                                self.last_metrics = jax.device_get(metrics_unrep)
                                if self.last_metrics and 'loss/total_raw' in self.last_metrics:
                                    if np.isfinite(self.last_metrics['loss/total_raw']):
                                        self.loss_hist.append(self.last_metrics['loss/total_raw'])
                            if active_preview_future is not None and active_preview_future.done():
                                active_preview_future.result()
                                active_preview_future = None
                            if active_preview_future is None:
                                safe_ema_params_copy = unreplicate(p_state.ema_params)
                                active_preview_future = async_pool.submit(
                                    self._update_preview_task,
                                    safe_ema_params_copy,
                                    self.preview_images_device[current_preview_idx:current_preview_idx+1]
                                )
                            preview_change = self.interactive_state.get_and_reset_preview_change()
                            if preview_change != 0:
                                current_preview_idx = (current_preview_idx + preview_change) % len(preview_buffer_host)
                                self.current_preview_np = ((preview_buffer_host[current_preview_idx] / 255.0) * 255).astype(np.uint8)
                                if Pixels:
                                    term_w = 64; h, w, _ = self.current_preview_np.shape; term_h = int(term_w*(h/w)*0.5)
                                    orig_img = Image.fromarray(self.current_preview_np).resize((term_w, term_h), Image.LANCZOS)
                                    self.rendered_original_preview = Pixels.from_image(orig_img)
                            live.update(self._generate_layout(), refresh=True); last_ui_update_time = time_now

                        if global_step > 0 and global_step % self.args.eval_every == 0:
                            with self.ui_lock:
                                self.is_validating = True
                                VAL_STEPS = 50
                                self.validation_progress = Progress(TextColumn("[cyan]Validating..."), BarColumn(), TextColumn("{task.completed}/{task.total}"))
                                val_task = self.validation_progress.add_task("val", total=VAL_STEPS)
                            live.update(self._generate_layout(), refresh=True)
                            val_iterator = val_dataset.take(VAL_STEPS).as_numpy_iterator()
                            val_losses = []
                            current_lambdas_dict = self.lambda_controller(self.last_metrics, global_step)
                            lambdas_for_eval = tuple(current_lambdas_dict[key] for key in lambda_keys_in_order)
                            for i, val_batch in enumerate(val_iterator):
                                sharded_val_batch = common_utils.shard(val_batch.astype(self.dtype))
                                loss = eval_step(p_state.ema_params, sharded_val_batch, p_full_coords, lambdas_for_eval)
                                val_losses.append(unreplicate(loss))
                                with self.ui_lock: self.validation_progress.update(val_task, advance=1)
                                live.update(self._generate_layout(), refresh=True)
                            val_loss = np.mean(val_losses) if val_losses else float('inf')
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                self.progress.update(epoch_task, val_loss=best_val_loss)
                                self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path_best)
                            console.print(f"\n[bold yellow]-- Validation complete. Loss: {val_loss:.4f} (Best: {best_val_loss:.4f}) --[/bold yellow]")
                            with self.ui_lock:
                                self.is_validating = False
                                self.validation_progress = None
                            live.update(self._generate_layout(), refresh=True)

                        if self.interactive_state.get_and_reset_force_save() or (global_step > 0 and steps_per_epoch > 0 and global_step % steps_per_epoch == 0 and (current_epoch + 1) % self.args.save_every == 0) :
                           self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path)
        finally:
            live.stop(); print("\n--- Training loop finished. ---")
            self.interactive_state.set_shutdown(); key_listener_thread.join()
            if 'p_state' in locals() and 'global_step' in locals():
                print("--- Saving final model state... ---")
                current_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
                self._save_checkpoint(p_state, current_epoch, global_step, best_val_loss, ckpt_path)
                print("--- ✅ Final state saved. ---")

# =================================================================================================
# 5. MAIN EXECUTION BLOCK
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Topological AE - Advanced Trainer"); subparsers = parser.add_subparsers(dest="command", required=True); parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--basename', type=str, required=True, help="Basename for model files."); parent_parser.add_argument('--d-model', type=int, default=128, help="Model dimension."); parent_parser.add_argument('--latent-grid-size', type=int, default=16, help="Size of the latent grid."); parent_parser.add_argument('--image-size', type=int, default=512, help="Image resolution.")
    p_prep = subparsers.add_parser("prepare-data", help="Convert images to TFRecords."); p_prep.add_argument('--data-dir', type=str, required=True)
    p_train = subparsers.add_parser("train", help="Train the model with advanced tools.", parents=[parent_parser]); p_train.add_argument('--data-dir', type=str, required=True, help="Path to a directory with TFRecords."); p_train.add_argument('--epochs', type=int, default=100, help="Total number of training epochs."); p_train.add_argument('--batch-size', type=int, default=4, help="Batch size PER DEVICE."); p_train.add_argument('--lr', type=float, default=2e-4)
    p_train.add_argument('--seed', type=int, default=42); p_train.add_argument('--use-q-controller', action='store_true', help="Enable adaptive LR via Q-Learning."); p_train.add_argument('--use-sentinel', action='store_true', help="Enable Sentinel optimizer."); p_train.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 mixed precision."); p_train.add_argument('--save-every', type=int, default=1, help="Save checkpoint every N epochs."); p_train.add_argument('--eval-every', type=int, default=200, help="Run validation every N global steps.")
    args = parser.parse_args()
    if args.command == "prepare-data": prepare_data(args.data_dir)
    elif args.command == "train": ImageTrainer(args).train()
    else: print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()