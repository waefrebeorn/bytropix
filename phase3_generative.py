# =================================================================================================
#
#        PHASE 3: DIRECT KNOWLEDGE DISTILLATION (Upgraded with Full Phase 1 Toolkit)
#
#       Distills a text encoder's knowledge into a tiny, self-contained, physics-informed
#       text-to-image generator using a self-referential consistency loss, supervised by
#       a full, dynamically-weighted perceptual loss suite.
#
#       >>> WORKFLOW: Use `coco_preprocessor.py` first, then run `prepare-distill-data` on its output. <<<
#
# =================================================================================================

import os
# --- Environment Setup for JAX/TensorFlow ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
import sys
import argparse
import pickle
import time
import threading
from functools import partial
from typing import Any, Dict, NamedTuple, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from flax.core import freeze, unfreeze
from flax import struct
import chex
from PIL import Image
from dataclasses import dataclass

# --- Dependency Checks ---
print("--- Checking Core Dependencies ---")
try:
    from rich_pixels import Pixels
    print("✅ [FOUND]   rich_pixels")
except ImportError:
    Pixels = None
    print("⚠️ [MISSING] `rich-pixels`. Preview disabled. `pip install rich-pixels`")

dependencies = [("tensorflow","tensorflow"),("tensorflow_datasets","tensorflow-datasets"),("rich.console","rich"),("pynvml","nvidia-ml-py"), ("transformers", "transformers"), ("torch", "torch")]
missing = []
for module, package in dependencies:
    try: __import__(module.split('.')[0]); print(f"✅ [FOUND]   {module}")
    except ImportError: missing.append(package); print(f"❌ [MISSING] {module} (Requires: {package})")
if missing: print(f"\n[FATAL] Missing dependencies. Please run: pip install {' '.join(missing)}"), sys.exit(1)
print("--- All dependencies verified. Proceeding with full imports. ---\n")

import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
from rich.text import Text
import pynvml; pynvml.nvmlInit()
from tqdm import tqdm
# Only import transformers if needed for data prep
try:
    from transformers import SiglipImageProcessor, SiglipTextModel, SiglipTokenizer
    import torch
except ImportError:
    SiglipImageProcessor, SiglipTextModel, SiglipTokenizer, torch = None, None, None, None


if platform.system() != "Windows": import select, tty, termios
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False)

# =================================================================================================
# 1. ADVANCED TRAINING TOOLKIT (PORTED & INTEGRATED)
# =================================================================================================
class InteractivityState:
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_index_change, self.sentinel_dampening_log_factor = 0, 0.0
        self.shutdown_event, self.force_save = threading.Event(), False
    def get_and_reset_preview_change(self):
        with self.lock: change = self.preview_index_change; self.preview_index_change = 0; return change
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def update_sentinel_factor(self, direction):
        with self.lock: self.sentinel_dampening_log_factor = np.clip(self.sentinel_dampening_log_factor + direction*0.5, -3.0, 0.0)
    def get_sentinel_factor(self):
        with self.lock: return 10**self.sentinel_dampening_log_factor
    def set_shutdown(self): self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    print("--- Key listener started. Controls: [←/→] Preview | [↑/↓] Sentinel | [s] Force Save | [q] Quit ---")
    if platform.system() == "Windows": import msvcrt # type: ignore
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
            elif key == b'\xe0' or key == '\x1b': # Arrow keys
                arrow = msvcrt.getch() if platform.system() == "Windows" else sys.stdin.read(2)
                if arrow in [b'K', '[D']: shared_state.preview_index_change = -1
                elif arrow in [b'M', '[C']: shared_state.preview_index_change = 1
                elif arrow in [b'H', '[A']: shared_state.update_sentinel_factor(1)
                elif arrow in [b'P', '[B']: shared_state.update_sentinel_factor(-1)
    finally:
        if platform.system() != "Windows": termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_sentinel_lever_ascii(log_factor: float):
    idx = np.clip(int((-np.clip(log_factor, -3.0, 0.0) / 3.0) * 6), 0, 6)
    bars = ["│         │"] * 7; bars[idx] = "│█████████│"
    labels = ["1.0 (Off)", " ", "0.1", " ", "0.01", " ", "0.001"]
    return "\n".join([f" {labels[i]:<10} {bars[i]}" for i in range(7)])

class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; trend_history: chex.Array
    current_lr: jnp.ndarray; exploration_rate: jnp.ndarray; step_count: jnp.ndarray
    last_action_idx: jnp.ndarray; last_reward: jnp.ndarray; status_code: jnp.ndarray; trend_slope: jnp.ndarray

@dataclass(frozen=True)
class QControllerConfig:
    q_table_size: int = 100; num_lr_actions: int = 5
    lr_change_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1; discount_factor_q: float = 0.9
    lr_min: float = 1e-6; lr_max: float = 1e-3
    metric_history_len: int = 100; loss_min: float = 0.01; loss_max: float = 5.0
    exploration_rate_q: float = 0.3; min_exploration_rate: float = 0.05; exploration_decay: float = 0.9998
    trend_window: int = 50; improve_threshold: float = 1e-5; regress_threshold: float = 1e-6
    regress_penalty: float = 5.0; stagnation_penalty: float = -1.0
    warmup_steps: int = 500; warmup_lr_start: float = 1e-6

Q_CONTROLLER_CONFIG = QControllerConfig()

def init_q_controller(config: QControllerConfig, initial_lr):
    return QControllerState(
        q_table=jnp.zeros((config.q_table_size, config.num_lr_actions)),
        metric_history=jnp.full((config.metric_history_len,), (config.loss_min + config.loss_max) / 2),
        trend_history=jnp.zeros((config.trend_window,)), current_lr=jnp.array(config.warmup_lr_start),
        exploration_rate=jnp.array(config.exploration_rate_q), step_count=jnp.array(0),
        last_action_idx=jnp.array(-1), last_reward=jnp.array(0.0), status_code=jnp.array(0), trend_slope=jnp.array(0.0))

@partial(jax.jit, static_argnames=('config', 'target_lr'))
def q_controller_choose_action(state, key, config, target_lr):
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config.warmup_steps
        lr = config.warmup_lr_start * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=lr, step_count=state.step_count + 1, status_code=jnp.array(0))
    def regular_action():
        mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config.metric_history_len - 10, 10))
        state_idx = jnp.clip(((mean - config.loss_min) / ((config.loss_max-config.loss_min)/config.q_table_size)).astype(jnp.int32), 0, config.q_table_size-1)
        explore, act = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore) < state.exploration_rate,
            lambda: jax.random.randint(act, (), 0, config.num_lr_actions), lambda: jnp.argmax(state.q_table[state_idx]))
        new_lr = jnp.clip(state.current_lr * config.lr_change_factors[action_idx], config.lr_min, config.lr_max)
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config.warmup_steps, warmup_action, regular_action)

@partial(jax.jit, static_argnames=('config',))
def q_controller_update(state, metric_value, config):
    st_new = state.replace(metric_history=jnp.roll(state.metric_history, -1).at[-1].set(metric_value),
                           trend_history=jnp.roll(state.trend_history, -1).at[-1].set(metric_value))
    def perform_q_update(st):
        x = jnp.arange(config.trend_window, dtype=jnp.float32); y = st.trend_history
        A = jnp.vstack([x, jnp.ones_like(x)]).T; slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        status, reward = jax.lax.cond(slope < -config.improve_threshold, lambda: (jnp.array(1), -slope*1000.),
            lambda: jax.lax.cond(slope > config.regress_threshold, lambda: (jnp.array(3), -slope*1000. - config.regress_penalty),
                lambda: (jnp.array(2), config.stagnation_penalty)))
        last_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config.metric_history_len-20, 10))
        new_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(st.metric_history, config.metric_history_len-10, 10))
        last_idx = jnp.clip(((last_mean - config.loss_min)/((config.loss_max-config.loss_min)/config.q_table_size)).astype(jnp.int32),0,config.q_table_size-1)
        next_idx = jnp.clip(((new_mean - config.loss_min)/((config.loss_max-config.loss_min)/config.q_table_size)).astype(jnp.int32),0,config.q_table_size-1)
        q = st.q_table[last_idx,st.last_action_idx]; max_q = jnp.max(st.q_table[next_idx])
        new_q = q + config.learning_rate_q * (reward + config.discount_factor_q * max_q - q)
        return st.replace(q_table=st.q_table.at[last_idx, st.last_action_idx].set(new_q),
            exploration_rate=jnp.maximum(config.min_exploration_rate, st.exploration_rate * config.exploration_decay),
            last_reward=reward, status_code=status, trend_slope=slope)
    can_update = (st_new.step_count > config.warmup_steps) & (st_new.step_count > config.trend_window) & (st_new.last_action_idx >= 0)
    return jax.lax.cond(can_update, perform_q_update, lambda s: s, st_new)

class SentinelState(NamedTuple):
    sign_ema: chex.ArrayTree; dampened_pct: jnp.ndarray

def sentinel(decay: float = 0.9, oscillation_threshold: float = 0.5) -> optax.GradientTransformation:
    def init_fn(params): return SentinelState(jax.tree_util.tree_map(jnp.zeros_like, params), jnp.array(0.0))
    def update_fn(updates, state, params=None, **kwargs):
        damp_factor = kwargs.get('dampening_factor', 1.0)
        signs = jax.tree_util.tree_map(jnp.sign, updates)
        new_ema = jax.tree_util.tree_map(lambda ema, s: ema * decay + s * (1-decay), state.sign_ema, signs)
        is_osc = jax.tree_util.tree_map(lambda ema: jnp.abs(ema) < oscillation_threshold, new_ema)
        mask = jax.tree_util.tree_map(lambda osc: jnp.where(osc, damp_factor, 1.0), is_osc)
        damp_updates = jax.tree_util.tree_map(lambda u, m: u * m, updates, mask)
        num_damp = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.sum, jax.tree_util.tree_map(lambda x: x < 1.0, mask))))
        total = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, params)))
        pct = num_damp / (total + 1e-8)
        return damp_updates, SentinelState(new_ema, pct)
    return optax.GradientTransformation(init_fn, update_fn)

class PIDLambdaController:
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]], warmup_steps: int = 500):
        self.targets, self.base_weights, self.gains, self.warmup_steps = targets, base_weights, gains, warmup_steps
        self.state = {'integral_error': {k: 0.0 for k in targets}, 'last_error': {k: 0.0 for k in targets}}
    def __call__(self, last_metrics: Dict[str, float], step: int) -> Dict[str, float]:
        if step < self.warmup_steps: return self.base_weights
        lambdas = {}
        for name, base in self.base_weights.items():
            lambdas[name] = float(base)
            if name in self.targets:
                val = last_metrics.get(f'loss/{name}');
                if val is None: continue
                try: current_loss = float(val)
                except (TypeError, ValueError): continue
                kp, ki, kd = self.gains[name]; target = self.targets[name]; error = current_loss - target
                self.state['integral_error'][name] += error; self.state['integral_error'][name] = np.clip(self.state['integral_error'][name], -5.0, 5.0)
                derivative = error - self.state['last_error'][name]
                adj = (kp * error) + (ki * self.state['integral_error'][name]) + (kd * derivative)
                lambdas[name] = float(np.clip(base * np.exp(adj), 0.1, 20.0)); self.state['last_error'][name] = error
        return lambdas
    def state_dict(self): return self.state
    def load_state_dict(self, state):
        self.state['integral_error'] = state.get('integral_error', {k: 0.0 for k in self.targets})
        self.state['last_error'] = state.get('last_error', {k: 0.0 for k in self.targets})

class CustomTrainState(train_state.TrainState):
    ema_params: Any; q_controller_state: QControllerState
    def apply_gradients(self, *, grads: Any, **kwargs) -> "CustomTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, **kwargs)
        new_params = optax.apply_updates(self.params, updates)
        new_ema = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1-0.999), self.ema_params, new_params)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, ema_params=new_ema)

# =================================================================================================
# 2. MODEL DEFINITIONS & PERCEPTUAL LOSS TOOLKIT
# =================================================================================================
@partial(jax.jit)
def rgb_to_ycber(image: jnp.ndarray) -> jnp.ndarray:
    rgb_01 = (image + 1.0) / 2.0; r,g,b = rgb_01[...,0], rgb_01[...,1], rgb_01[...,2]
    y = 0.299*r + 0.587*g + 0.114*b; cb = -0.168736*r - 0.331264*g + 0.5*b; cr = 0.5*r - 0.418688*g - 0.081312*b
    return jnp.stack([y, cb, cr], axis=-1)

@partial(jax.jit)
def ycber_to_rgb(image: jnp.ndarray) -> jnp.ndarray:
    y,cb,cr = image[...,0], image[...,1], image[...,2]; r = y + 1.402 * cr; g = y - 0.344136*cb - 0.714136*cr; b = y + 1.772 * cb
    return jnp.clip(jnp.stack([r,g,b], axis=-1), 0., 1.) * 2. - 1.

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2); imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part
@partial(jax.jit)
def rgb_to_ycber_sym(image: jnp.ndarray) -> jnp.ndarray:
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr =  0.5 * r - 0.418688 * g - 0.081312 * b
    return jnp.stack([y, cb * 2.0, cr * 2.0], axis=-1)

@partial(jax.jit)
def ycber_sym_to_rgb(image: jnp.ndarray) -> jnp.ndarray:
    y, cb, cr = image[..., 0], image[..., 1], image[..., 2]
    cb_unscaled = cb / 2.0
    cr_unscaled = cr / 2.0
    r = y + 1.402 * cr_unscaled
    g = y - 0.344136 * cb_unscaled - 0.714136 * cr_unscaled
    b = y + 1.772 * cb_unscaled
    return jnp.clip(jnp.stack([r, g, b], axis=-1), -1.0, 1.0)
class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images_rgb: jnp.ndarray) -> jnp.ndarray:
        images_ycber = rgb_to_ycber_sym(images_rgb)
        x, features, current_dim, i = images_ycber, 32, self.input_image_size, 0
        while (current_dim // 2) >= self.latent_grid_size and (current_dim // 2) > 0:
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            features *= 2; current_dim //= 2; i += 1
        if current_dim != self.latent_grid_size:
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        def create_head(name: str, input_features: jnp.ndarray):
            h = nn.Conv(128, (1, 1), name=f"{name}_head_conv1", dtype=self.dtype)(input_features); h = nn.gelu(h)
            params_raw = nn.Conv(3, (1, 1), name=f"{name}_head_out", dtype=self.dtype, bias_init=lambda k,s,d:jnp.zeros(s,d).at[2].set(-1.0))(h)
            return nn.tanh(params_raw[...,0])*jnp.pi, nn.tanh(params_raw[...,1])*(jnp.pi/4.0), nn.sigmoid(params_raw[...,2])*(jnp.pi/2.0)
        d_y, c_y, r_y = create_head("y_luma", x); d_cb, c_cb, r_cb = create_head("cb_chroma", x); d_cr, c_cr, r_cr = create_head("cr_chroma", x)
        return jnp.stack([d_y, c_y, r_y, d_cb, c_cb, r_cb, d_cr, c_cr, r_cr], axis=-1)

class CoordinateDecoder(nn.Module):
    d_model: int; num_freqs: int = 10; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.remat
    def _mlp_block(self, h, skip):
        for i in range(self.mlp_depth):
            h=nn.Dense(self.mlp_width,name=f"mlp_{i}",dtype=self.dtype)(h); h=nn.gelu(h)
            if i==self.mlp_depth//2: h=jnp.concatenate([h,skip],axis=-1)
        return nn.Dense(3, name="mlp_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(h)
    @nn.compact
    def __call__(self, feat_grid, coords):
        B,H,W,C=feat_grid.shape; enc_coords=PositionalEncoding(self.num_freqs,self.dtype)(coords)
        pyramid=[feat_grid]+[jax.image.resize(feat_grid,(B,H//(2**i),W//(2**i),C),'bilinear')for i in range(1,3)]
        s_feats=[]
        for level in pyramid:
            shape=jnp.array(level.shape[1:3],self.dtype); coords_r=(coords+1)/2*(shape-1)
            def sample(single): return jax.vmap(lambda g:jax.scipy.ndimage.map_coordinates(g,coords_r.T,order=1,mode='reflect'))(single.transpose(2,0,1)).T
            s_feats.append(jax.vmap(sample)(level))
        all_feats=jnp.concatenate(s_feats,axis=-1)
        mlp_in=jnp.concatenate([jnp.repeat(enc_coords[None,...],B,0),all_feats],axis=-1)
        return ycber_sym_to_rgb(nn.tanh(self._mlp_block(mlp_in,mlp_in)))
class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = path_params_grid.shape; p = path_params_grid.reshape(B, H*W, 9)
        def get_stats(d_c, chi_c, r):
            t = jnp.linspace(0, 2*jnp.pi, self.num_path_steps, dtype=jnp.float32)
            d_p = d_c[...,None]+r[...,None]*jnp.cos(t); chi_p = chi_c[...,None]+r[...,None]*jnp.sin(t)
            t_co = PoincareSphere.calculate_co_polarized_transmittance(d_p, chi_p) + 1e-8
            phase=jnp.angle(t_co); amp=jnp.abs(t_co)
            safe_std = jnp.sqrt(jnp.maximum(0., jnp.var(amp, axis=-1)))
            return jnp.stack([jnp.mean(phase,-1), jnp.ptp(phase,-1), jnp.mean(amp,-1), safe_std, r], -1).astype(self.dtype)
        s_y,s_cb,s_cr=get_stats(p[...,0],p[...,1],p[...,2]),get_stats(p[...,3],p[...,4],p[...,5]),get_stats(p[...,6],p[...,7],p[...,8])
        all_s = jnp.concatenate([s_y,s_cb,s_cr],axis=-1)
        return nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(all_s).reshape(B,H,W,self.d_model)
class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        f=2.**jnp.arange(self.num_freqs,dtype=self.dtype)*jnp.pi
        return jnp.concatenate([x]+[fn(x*fr) for fr in f for fn in (jnp.sin,jnp.cos)],axis=-1)
class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(self.d_model, name="coord_decoder", dtype=self.dtype)
    def encode(self, images_rgb): return self.modulator(images_rgb)
    def decode_from_path_params(self, path_params, coords): return self.coord_decoder(self.observer(path_params), coords)
class TextToPathParamsProjector(nn.Module):
    latent_grid_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, command_vector: jnp.ndarray) -> jnp.ndarray:
        pooled = jnp.mean(command_vector, axis=1)
        x = nn.Dense(4*4*1024,dtype=self.dtype)(pooled); x=nn.gelu(x); x=x.reshape(-1,4,4,1024)
        for feats in [512, 256, 128, 64]:
            x = nn.ConvTranspose(feats,(4,4),(2,2),'SAME',dtype=self.dtype)(x); x=nn.gelu(x)
        p = nn.Conv(9,(1,1),dtype=self.dtype,bias_init=lambda k,s,d:jnp.zeros(s,d).at[jnp.array([2,5,8])].set(0.))(x)
        d_y,c_y,r_y=nn.tanh(p[...,0])*jnp.pi,nn.tanh(p[...,1])*(jnp.pi/4.),nn.sigmoid(p[...,2])*(jnp.pi/2.)
        d_cb,c_cb,r_cb=nn.tanh(p[...,3])*jnp.pi,nn.tanh(p[...,4])*(jnp.pi/4.),nn.sigmoid(p[...,5])*(jnp.pi/2.)
        d_cr,c_cr,r_cr=nn.tanh(p[...,6])*jnp.pi,nn.tanh(p[...,7])*(jnp.pi/4.),nn.sigmoid(p[...,8])*(jnp.pi/2.)
        return jnp.stack([d_y,c_y,r_y,d_cb,c_cb,r_cb,d_cr,c_cr,r_cr],axis=-1)
@partial(jax.jit, static_argnames=('max_lag',))
def calculate_autocorrelation_features(patches: jnp.ndarray, max_lag: int = 8) -> jnp.ndarray:
    patches_f32 = patches.astype(jnp.float32); patches_gray = jnp.mean(patches_f32, axis=-1, keepdims=True)
    patches_centered = patches_gray - jnp.mean(patches_gray, axis=(1, 2), keepdims=True)
    norm_factor = jnp.var(patches_centered, axis=(1, 2), keepdims=True) + 1e-5; lags = jnp.arange(1, max_lag + 1)
    def _corr(lag_x, lag_y): return jnp.mean(patches_centered*jnp.roll(patches_centered,(lag_y,lag_x),axis=(1,2)),(1,2,3))/(jnp.squeeze(norm_factor)+1e-8)
    corr_h, corr_v, corr_d = jax.vmap(_corr, (0,None))(lags,0), jax.vmap(_corr, (None,0))(0,lags), jax.vmap(_corr, (0,0))(lags,lags)
    return jnp.concatenate([corr_h.T, corr_v.T, corr_d.T], axis=-1)
def _compute_sobel_magnitude(patches: jnp.ndarray, kx: jnp.ndarray, ky: jnp.ndarray) -> jnp.ndarray:
    patches_f32 = patches.astype(jnp.float32)
    def apply_sobel_on_slice(img_slice_2d):
        grad_x = jax.scipy.signal.convolve2d(img_slice_2d, kx, mode='same', boundary='fill', fillvalue=0)
        grad_y = jax.scipy.signal.convolve2d(img_slice_2d, ky, mode='same', boundary='fill', fillvalue=0)
        return jnp.sqrt(jnp.maximum(grad_x**2 + grad_y**2, 0.) + 1e-6)
    patches_nchw = jnp.transpose(patches_f32, (0, 3, 1, 2))
    magnitudes_nchw = jax.vmap(jax.vmap(apply_sobel_on_slice))(patches_nchw)
    magnitudes_nhwc = jnp.transpose(magnitudes_nchw, (0, 2, 3, 1))
    return jnp.linalg.norm(magnitudes_nhwc, axis=-1)
@jit
def calculate_edge_loss(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    sobel_x, sobel_y = jnp.array([[-1,0,1],[-2,0,2],[-1,0,1]],jnp.float32), jnp.array([[-1,-2,-1],[0,0,0],[1,2,1]],jnp.float32)
    return jnp.mean(jnp.abs(_compute_sobel_magnitude(p1,sobel_x,sobel_y)-_compute_sobel_magnitude(p2,sobel_x,sobel_y)))
@jit
def calculate_color_covariance_loss(p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    p1_f32, p2_f32 = p1.astype(jnp.float32), p2.astype(jnp.float32)
    def get_gram(p): feat = p.reshape(p.shape[0],-1,p.shape[-1]); return jax.vmap(lambda x:x.T@x)(feat)/(feat.shape[1]*feat.shape[2])
    return jnp.mean(jnp.abs(get_gram(p1_f32)-get_gram(p2_f32)))
@jit
def calculate_ssim_loss(p1: jnp.ndarray, p2: jnp.ndarray, max_val: float = 2.0) -> jnp.ndarray:
    p1_f32,p2_f32=p1.astype(jnp.float32),p2.astype(jnp.float32); C1,C2=(0.01*max_val)**2,(0.03*max_val)**2
    p1g,p2g=jnp.mean(p1_f32,-1),jnp.mean(p2_f32,-1); mu1,mu2=jnp.mean(p1g,(1,2)),jnp.mean(p2g,(1,2))
    var1,var2=jnp.var(p1g,(1,2)),jnp.var(p2g,(1,2)); cov=jnp.mean(p1g*p2g,(1,2))-(mu1*mu2)
    num,den=(2*mu1*mu2+C1)*(2*cov+C2),(mu1**2+mu2**2+C1)*(var1+var2+C2)
    return jnp.mean(1.0-num/(den+1e-8))
@partial(jax.jit, static_argnames=('num_moments',))
def calculate_moments(patches, num_moments=4):
    p_f32 = patches.astype(jnp.float32); flat = p_f32.reshape(p_f32.shape[0], -1, p_f32.shape[-1])
    mean, var = jnp.mean(flat, axis=1), jnp.var(flat, axis=1);
    if num_moments<=2: return jnp.concatenate([mean,var],-1)
    std=jnp.sqrt(jnp.maximum(var,1e-8)); norm_dev=(flat-mean[:,None,:])/(std[:,None,:]+1e-8)
    skew=jnp.mean(norm_dev**3,1);
    if num_moments<=3: return jnp.concatenate([mean,var,skew],-1)
    kurt=jnp.mean(norm_dev**4,1); return jnp.concatenate([mean,var,skew,kurt],-1)
@jit
def fft_magnitude_log(patches): return jax.vmap(lambda p: jnp.log(jnp.abs(jnp.fft.fft2(p))+1e-5))(patches.astype(jnp.float32))
@partial(jax.vmap, in_axes=(0,0,0,None,None), out_axes=0)
def _extract_patches_vmapped(img, x, y, ps, c): return jax.vmap(lambda i,j:jax.lax.dynamic_slice(img,(j,i,0),(ps,ps,c)))(x,y)
class JAXMultiMetricPerceptualLoss:
    def __init__(self, num_patches=64, patch_size=64):
        self.num_patches, self.patch_size = num_patches, patch_size
        self._calculate_losses_jit = partial(jax.jit, static_argnames=('batch_size',))(self._calculate_losses)
    def _calculate_losses(self, img1, img2, key, batch_size: int):
        _, h, w, c = img1.shape; k1, k2 = jax.random.split(key)
        x=jax.random.randint(k1,(batch_size,self.num_patches),0,w-self.patch_size); y=jax.random.randint(k2,(batch_size,self.num_patches),0,h-self.patch_size)
        p1=_extract_patches_vmapped(img1,x,y,self.patch_size,c).reshape(-1,self.patch_size,self.patch_size,c)
        p2=_extract_patches_vmapped(img2,x,y,self.patch_size,c).reshape(-1,self.patch_size,self.patch_size,c)
        l={'l1':jnp.mean(jnp.abs(p1-p2)),'moment':jnp.mean(jnp.abs(calculate_moments(p1)-calculate_moments(p2))),
           'fft':jnp.mean(jnp.abs(fft_magnitude_log(jnp.mean(p1,-1))-fft_magnitude_log(jnp.mean(p2,-1)))),
           'autocorr':jnp.mean(jnp.abs(calculate_autocorrelation_features(p1)-calculate_autocorrelation_features(p2))),
           'edge':calculate_edge_loss(p1,p2),'color_cov':calculate_color_covariance_loss(p1,p2),'ssim':calculate_ssim_loss(p1,p2)}
        return {f'loss/{k}': v for k, v in l.items()}
    def __call__(self, img1, img2, key): return self._calculate_losses_jit(img1, img2, key, batch_size=img1.shape[0])

# =================================================================================================
# 3. UNIFIED & ROBUST DATA PREPARATION
# =================================================================================================
def prepare_distill_data(source_dir: str, target_dir: str, text_encoder_id: str):
    console = Console()
    if not all([SiglipImageProcessor, SiglipTextModel, SiglipTokenizer, torch]):
        console.print("[bold red]FATAL: `transformers` and `torch` are required. Please install them.[/bold red]")
        sys.exit(1)

    source_path, target_path = Path(source_dir), Path(target_dir)
    target_path.mkdir(exist_ok=True)
    console.print(f"--- 🔍 Scanning for aligned image-text pairs in [cyan]{source_path}[/cyan]... ---")
    image_paths = sorted(list(source_path.rglob('*.jpg')) + list(source_path.rglob('*.png')) + list(source_path.rglob('*.webp')))
    
    aligned_pairs = []
    for img_path in tqdm(image_paths, desc="Verifying pairs"):
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists(): aligned_pairs.append((img_path, txt_path))

    if not aligned_pairs:
        console.print(f"[bold red]FATAL: No aligned .jpg/.txt pairs found in {source_path}.[/bold red]"); sys.exit(1)

    console.print(f"--- ✅ Found {len(aligned_pairs)} aligned pairs. ---")
    console.print(f"--- 🧠 Loading Text Encoder: [yellow]{text_encoder_id}[/yellow]... ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_model = SiglipTextModel.from_pretrained(text_encoder_id).to(device)
    tokenizer = SiglipTokenizer.from_pretrained(text_encoder_id)
    text_model.eval()

    safe_name = text_encoder_id.replace('/','_')
    distill_record_file = target_path / f"distill_data_{safe_name}.tfrecord"
    image_record_file = target_path / "data_512x512.tfrecord"
    console.print(f"--- ✍️ Writing aligned TFRecords to [cyan]{target_path}[/cyan]... ---")
    
    with tf.io.TFRecordWriter(str(image_record_file)) as img_writer, \
         tf.io.TFRecordWriter(str(distill_record_file)) as distill_writer:
        for img_path, txt_path in tqdm(aligned_pairs, desc="Processing and Writing"):
            try:
                img = Image.open(img_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
                img_bytes = tf.io.encode_jpeg(np.array(img), quality=95).numpy()
                img_writer.write(tf.train.Example(features=tf.train.Features(feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))})).SerializeToString())
                
                with open(txt_path, 'r', encoding='utf-8') as f: prompt = f.read().strip()
                with torch.no_grad():
                    inputs = tokenizer([prompt], padding="max_length", max_length=64, return_tensors="pt").to(device)
                    embedding = text_model(**inputs).last_hidden_state.squeeze(0).cpu().numpy().astype(np.float16)
                
                distill_writer.write(tf.train.Example(features=tf.train.Features(feature={
                    'embedding': tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding.tobytes()])),
                    'prompt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[prompt.encode('utf-8')]))
                })).SerializeToString())
            except Exception as e:
                console.print(f"[yellow]Skipping {img_path.name} due to error: {e}[/yellow]")

    console.print(f"\n--- 🎉 Unified data preparation complete! ---")
    console.print(f"✅ Image TFRecord: [green]{image_record_file}[/green]")
    console.print(f"✅ Distill TFRecord: [green]{distill_record_file}[/green]")

# =================================================================================================
# 4. DISTILLATION TRAINER
# =================================================================================================
class DistillationTrainer:
    def __init__(self, args):
        self.args = args; self.console = Console()
        self.interactive_state = InteractivityState()
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.loss_calculator = JAXMultiMetricPerceptualLoss(patch_size=64)
        
        # PID Controller for Perceptual Sub-Losses
        sub_loss_pid_gains = {'l1':(0.8,0.01,1.0),'ssim':(1.5,0.02,2.0),'edge':(1.2,0.01,1.5),'moment':(0.5,0.005,0.8),
                              'color_cov':(0.7,0.005,1.0),'autocorr':(0.6,0.005,0.9),'fft':(0.4,0.005,0.5)}
        self.sub_loss_lambda_controller = PIDLambdaController(
            targets={'l1':0.01,'ssim':0.1,'edge':0.15,'moment':0.15,'color_cov':0.02,'autocorr':0.15,'fft':0.1},
            base_weights={'l1':1.0,'ssim':0.8,'edge':1.0,'moment':1.0,'color_cov':0.9,'autocorr':0.3,'fft':0.4},
            gains=sub_loss_pid_gains, warmup_steps=1000)

        # PID Controller for Main Loss Components
        main_loss_pid_gains = {'echo':(0.6,0.01,0.8),'perceptual':(0.5,0.01,0.7),'gt_patch':(0.7,0.01,0.9),'diversity':(0.5,0.005,0.6)}
        self.main_loss_lambda_controller = PIDLambdaController(
            targets={'echo': 0.1, 'perceptual': 0.25, 'gt_patch': 0.50, 'diversity': 0.1},
            base_weights={'echo': 1.0, 'perceptual': 0.5, 'gt_patch': 0.5, 'diversity': 0.05},
            gains=main_loss_pid_gains, warmup_steps=1000)
        
        self.text_encoder_id = "google/siglip-base-patch16-224"; self.SEQUENCE_LENGTH=64; self.EXPECTED_EMBEDDING_DIM=768
        self.param_count = 0; self.last_metrics = {}; self.loss_hist = deque(maxlen=200); self.current_lambdas = {}
        self.steps_per_sec = 0.0; self.ui_lock = threading.Lock(); self.rendered_preview = None; self.current_prompt = "..."

    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB",f"{u.gpu}%"
        except: return "N/A","N/A"

    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist) < 2: return " " * w
        hist = hist[-w:]; min_v, max_v = hist.min(), hist.max()
        if max_v == min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1); return "".join(s[i] for i in indices)

    def _generate_layout(self):
        with self.ui_lock:
            layout = Layout(name="root"); layout.split(Layout(name="header",size=3), Layout(ratio=1,name="main"), Layout(name="footer",size=3))
            layout["main"].split_row(Layout(name="left",minimum_size=60), Layout(name="right",ratio=1))
            precision = "[bold purple]BF16[/]" if self.args.use_bfloat16 else "[dim]FP32[/]"
            header = f"🧠⚡ [bold]Echo+GT Distillation[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision}"
            layout["header"].update(Panel(Align.center(header), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right"))

            stats_tbl = Table.grid(expand=True,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
            mem,util = self._get_gpu_stats(); stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🚀"); stats_tbl.add_row("GPU Mem/Util", f"[yellow]{mem}[/] / [yellow]{util}[/]")
            lr = self.last_metrics.get('learning_rate', self.args.lr); stats_tbl.add_row("Learning Rate", f"[green]{float(lr):.2e}[/]")
            
            left_panels = [Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue")]

            loss_table = Table(show_header=False, box=None, padding=(0,1)); loss_table.add_column(style="cyan",width=10); loss_table.add_column(justify="right",style="white",width=10); loss_table.add_column(justify="right",style="yellow")
            loss_table.add_row("[bold]Metric[/bold]", "[bold]Value[/bold]", "[bold]λ (PID)[/bold]")
            loss_keys = ['echo', 'perceptual', 'gt_patch', 'diversity', 'l1', 'ssim', 'edge', 'moment', 'color_cov', 'autocorr', 'fft']
            for key in loss_keys:
                value = self.last_metrics.get(f'loss/{key}', 0.0);
                display_name = key.capitalize().replace('_', ' ')
                if key == 'gt_patch': display_name = "GT Patch"
                if key in ['perceptual', 'gt_patch']:
                    loss_table.add_row(f"[bold]{display_name}[/bold]", f"{value:.4f}", f"{self.current_lambdas.get(key, 0.0):.2f}")
                elif key in ['echo', 'diversity']:
                    loss_table.add_row(f"[bold]{display_name}[/bold]", f"{value:.4f}", f"{self.current_lambdas.get(key, 0.0):.2f}")
                else:
                    loss_table.add_row(display_name, f"{value:.4f}", f"{self.current_lambdas.get(key, 0.0):.2f}")
            loss_panel = Panel(loss_table, title="[bold]Loss Components[/]", border_style="cyan")
            left_panels.append(loss_panel)

            if self.args.use_q_controller:
                q_code=int(self.last_metrics.get('q_status',0)); q_status={0:"[blue]WARMUP",1:"[green]IMPROVING",2:"[yellow]STAGNATED",3:"[red]REGRESSING"}.get(q_code,"[dim]N/A[/dim]")
                q_panel = Panel(Align.center(q_status), title="[bold]Autonomous LR Scheduler 🧠[/]", border_style="green", height=3)
                left_panels.append(q_panel)

            if self.args.use_sentinel:
                sentinel_panel = Panel(Group(Text(f"Dampened: {self.last_metrics.get('sentinel_pct',0.0):.2%}",justify="center"),Text(get_sentinel_lever_ascii(self.interactive_state.sentinel_dampening_log_factor),justify="center")), title="[bold]🕹️ Sentinel (↑/↓)[/]", border_style="yellow")
                left_panels.append(sentinel_panel)

            layout["left"].update(Group(*left_panels))

            total_loss = self.last_metrics.get('loss/total', 0.0)
            spark = Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_hist,60)}[/]"), title=f"Total Loss: {total_loss:.4f}", height=3, border_style="cyan")
            preview = self.rendered_preview or Text("...", justify="center")
            prompt = Panel(Text(self.current_prompt, justify="center"), title="[bold]Live Preview (←/→)[/]", border_style="green")
            layout["right"].update(Group(spark, prompt, Panel(preview, title="Generated Image 👨‍🎓")))
            layout["footer"].update(self.progress); return layout

    @partial(jit, static_argnames=('self','student_encoder','fixed_decoder_model','resolution'))
    def _generate_student_image(self, student_params, fixed_decoder_params, command_vec, student_encoder, fixed_decoder_model, resolution):
         paths = student_encoder.apply({'params': student_params}, command_vec)
         coords = jnp.mgrid[-1:1:resolution*1j,-1:1:resolution*1j].transpose(1,2,0).reshape(-1,2)
         pixels = fixed_decoder_model.apply({'params': fixed_decoder_params}, paths, coords, method='decode_from_path_params')
         return pixels.reshape(paths.shape[0], resolution, resolution, 3)

    @partial(jit, static_argnames=('self','fixed_decoder_model','resolution'))
    def _generate_target_image(self, fixed_decoder_params, path_params, fixed_decoder_model, resolution):
         coords = jnp.mgrid[-1:1:resolution*1j,-1:1:resolution*1j].transpose(1,2,0).reshape(-1,2)
         pixels = fixed_decoder_model.apply({'params': fixed_decoder_params}, path_params, coords, method='decode_from_path_params')
         return pixels.reshape(path_params.shape[0], resolution, resolution, 3)

    def _update_preview_task(self, student_ema_params, fixed_decoder_params, preview_batch, student_encoder, fixed_decoder_model):
        emb_p, prompt_p, _ = preview_batch
        img_p = self._generate_student_image(student_ema_params, fixed_decoder_params, emb_p, student_encoder, fixed_decoder_model, 128)
        img_p.block_until_ready()
        img_np = ((np.array(img_p[0])*0.5+0.5)*255).clip(0,255).astype(np.uint8)
        with self.ui_lock:
            self.current_prompt = prompt_p[0].decode('utf-8')
            if Pixels:
                term_w=64; h,w,_=img_np.shape; term_h=int(term_w*(h/w)*0.5)
                self.rendered_preview = Pixels.from_image(Image.fromarray(img_np).resize((term_w, term_h), Image.LANCZOS))

    def train(self):
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()

        safe_name = self.text_encoder_id.replace('/','_')
        distill_record_file = Path(self.args.data_dir) / f"distill_data_{safe_name}.tfrecord"
        image_record_file = Path(self.args.data_dir) / "data_512x512.tfrecord"

        if not distill_record_file.exists() or not image_record_file.exists():
            self.console.print(f"[bold red]FATAL: TFRecord files not found![/bold red]")
            self.console.print(f"Please run the `prepare-distill-data` command on your high-quality dataset first.")
            sys.exit(1)
        
        num_records = sum(1 for _ in tf.data.TFRecordDataset(str(distill_record_file)))
        super_bs = self.args.batch_size * self.args.rebatch_size
        steps_per_epoch=num_records//super_bs; 
        total_steps=steps_per_epoch*self.args.epochs
        self.console.print(f"--- 🚀 Performance Mode: Rebatching {self.args.rebatch_size} steps per data load. ---")
        self.console.print(f"--- Found {num_records} aligned samples. Total steps: {total_steps} ({steps_per_epoch} steps/epoch) ---")

        emb_shape = (self.SEQUENCE_LENGTH, self.EXPECTED_EMBEDDING_DIM)

        def _parse_distill(proto):
            features = {'embedding': tf.io.FixedLenFeature([], tf.string), 'prompt': tf.io.FixedLenFeature([], tf.string)}
            p = tf.io.parse_single_example(proto, features)
            emb = tf.cast(tf.reshape(tf.io.decode_raw(p['embedding'], tf.float16), emb_shape), self.dtype)
            return emb, p['prompt']

        def _parse_image(proto):
            features = {'image': tf.io.FixedLenFeature([], tf.string)}
            p = tf.io.parse_single_example(proto, features)
            img = tf.io.decode_jpeg(p['image'], channels=3)
            img = tf.image.resize(img, [self.args.image_size, self.args.image_size], method='area')
            return tf.cast(img, self.dtype) / 127.5 - 1.0

        distill_ds = tf.data.TFRecordDataset(str(distill_record_file)).map(_parse_distill, num_parallel_calls=tf.data.AUTOTUNE)
        image_ds = tf.data.TFRecordDataset(str(image_record_file)).map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        combined_ds = tf.data.Dataset.zip((distill_ds, image_ds))
        ds_base = combined_ds.map(lambda x, y: (x[0], x[1], y), num_parallel_calls=tf.data.AUTOTUNE)
        
        p1_glob = list(Path('.').glob(f"{self.args.basename}_{self.args.d_model}d_*_best.pkl"))
        if not p1_glob: self.console.print(f"[bold red]FATAL: Phase 1 model not found![/bold red] Searched for [cyan]'{self.args.basename}_{self.args.d_model}d_*_best.pkl'[/cyan]."), sys.exit(1)
        self.console.print(f"--- Loading Phase 1 weights from: [cyan]{p1_glob[0]}[/cyan] ---")
        with open(p1_glob[0], 'rb') as f: p1_data = pickle.load(f)

        fixed_physics_model = TopologicalCoordinateGenerator(self.args.d_model,self.args.latent_grid_size,self.args.image_size,self.dtype)
        fixed_physics_params = freeze(p1_data['ema_params'])

        student_encoder = TextToPathParamsProjector(self.args.latent_grid_size, self.dtype)
        optimizer_chain=[optax.clip_by_global_norm(1.0)]
        if self.args.use_sentinel: optimizer_chain.append(sentinel())
        optimizer_chain.append(optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr if not self.args.use_q_controller else Q_CONTROLLER_CONFIG.lr_min))
        optimizer=optax.chain(*optimizer_chain)

        key=jax.random.PRNGKey(self.args.seed); start_step=0; ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d_distilled_t2i.ckpt.pkl")

        if ckpt_path.exists():
            self.console.print(f"--- Resuming from checkpoint: [cyan]{ckpt_path}[/cyan] ---")
            with open(ckpt_path, 'rb') as f: data = pickle.load(f)
            params=data['params']; ema_params=data.get('ema_params',params); start_step=data.get('step',0)
            q_state = data.get('q_controller_state', init_q_controller(Q_CONTROLLER_CONFIG, self.args.lr))
            if 'pid_controller_state' in data: self.sub_loss_lambda_controller.load_state_dict(data['pid_controller_state'])
            if 'main_pid_controller_state' in data: self.main_loss_lambda_controller.load_state_dict(data['main_pid_controller_state'])
            state = CustomTrainState.create(apply_fn=student_encoder.apply, params=params, tx=optimizer, ema_params=ema_params, q_controller_state=q_state)
            state = state.replace(opt_state=data['opt_state'], step=start_step)
        else:
            self.console.print("--- Initializing new model state ---")
            dummy_input = jnp.zeros((1,self.SEQUENCE_LENGTH,self.EXPECTED_EMBEDDING_DIM),self.dtype)
            params = student_encoder.init(key, dummy_input)['params']
            q_state = init_q_controller(Q_CONTROLLER_CONFIG, self.args.lr)
            state = CustomTrainState.create(apply_fn=student_encoder.apply, params=params, tx=optimizer, ema_params=params, q_controller_state=q_state)
        with self.ui_lock: self.param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))

        @partial(jit, static_argnames=('loss_calculator_static','student_encoder_static','physics_model_static','image_size','rebatch_size','batch_size'))
        def train_super_step(state, super_command_vec, super_gt_images, damp, main_lambdas_tuple, sub_lambdas_tuple, key, loss_calculator_static, student_encoder_static, physics_model_static, image_size, rebatch_size, batch_size):
            def body(i, carry):
                state, key, metrics_sum = carry
                vec = jax.lax.dynamic_slice_in_dim(super_command_vec, i*batch_size, batch_size, axis=0)
                gt_image_batch = jax.lax.dynamic_slice_in_dim(super_gt_images, i*batch_size, batch_size, axis=0)
                g_key, q_key, key = jax.random.split(key, 3)

                def loss_fn(p):
                    loss_key, perc_key, gt_perc_key = jax.random.split(g_key, 3)
                    path_intent = student_encoder_static.apply({'params': p}, vec)
                    img_student = self._generate_student_image(p, fixed_physics_params, vec, student_encoder_static, physics_model_static, image_size)
                    
                    path_echo = physics_model_static.apply({'params': fixed_physics_params}, img_student, method='encode')
                    img_target = self._generate_target_image(fixed_physics_params, path_echo, physics_model_static, image_size)
                    echo_loss = jnp.mean(jnp.abs(path_intent - path_echo))
                    perceptual_losses = loss_calculator_static(img_student, jax.lax.stop_gradient(img_target), perc_key)
                    gt_perceptual_losses = loss_calculator_static(img_student, jax.lax.stop_gradient(gt_image_batch), gt_perc_key)

                    l_l1,l_ssim,l_edge,l_moment,l_ccov,l_acorr,l_fft = sub_lambdas_tuple
                    perceptual_loss = (l_l1*perceptual_losses['loss/l1'] + l_ssim*perceptual_losses['loss/ssim'] +
                                       l_edge*perceptual_losses['loss/edge'] + l_moment*perceptual_losses['loss/moment'] +
                                       l_ccov*perceptual_losses['loss/color_cov'] + l_acorr*perceptual_losses['loss/autocorr'] +
                                       l_fft*perceptual_losses['loss/fft'])
                    
                    gt_patch_loss = (l_l1*gt_perceptual_losses['loss/l1'] + l_ssim*gt_perceptual_losses['loss/ssim'] +
                                     l_edge*gt_perceptual_losses['loss/edge'] + l_moment*gt_perceptual_losses['loss/moment'] +
                                     l_ccov*gt_perceptual_losses['loss/color_cov'] + l_acorr*gt_perceptual_losses['loss/autocorr'] +
                                     l_fft*gt_perceptual_losses['loss/fft'])

                    radii = path_intent[..., jnp.array([2, 5, 8])]; mean_radius = jnp.mean(radii)
                    radius_diversity_loss = 1.0 / (mean_radius + 1e-6)

                    l_echo, l_perc, l_gt, l_div = main_lambdas_tuple
                    total_loss = (l_echo * echo_loss) + (l_perc * perceptual_loss) + (l_gt * gt_patch_loss) + (l_div * radius_diversity_loss)

                    all_metrics = {
                        'loss/total': total_loss, 'loss/echo': echo_loss, 
                        'loss/diversity': radius_diversity_loss, 'loss/perceptual': perceptual_loss,
                        'loss/gt_patch': gt_patch_loss
                    }
                    all_metrics.update(perceptual_losses)
                    return total_loss, all_metrics
                
                (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                
                new_q, lr = state.q_controller_state, jnp.array(self.args.lr)
                if self.args.use_q_controller:
                    new_q = q_controller_choose_action(state.q_controller_state, q_key, Q_CONTROLLER_CONFIG, self.args.lr); lr = new_q.current_lr
                
                new_state = state.apply_gradients(grads=grads, dampening_factor=damp, learning_rate=lr).replace(q_controller_state=new_q)
                
                if self.args.use_q_controller:
                    safe_loss = jnp.nan_to_num(loss, nan=Q_CONTROLLER_CONFIG.loss_max, posinf=Q_CONTROLLER_CONFIG.loss_max)
                    new_state = new_state.replace(q_controller_state=q_controller_update(new_state.q_controller_state, safe_loss, Q_CONTROLLER_CONFIG))

                metrics = {**aux_metrics, 'learning_rate': lr}
                if self.args.use_sentinel: metrics['sentinel_pct'] = new_state.opt_state[1].dampened_pct
                if self.args.use_q_controller: metrics.update({k: getattr(new_state.q_controller_state, k) for k in ['status_code','last_reward','trend_slope']})
                return new_state, key, jax.tree_util.tree_map(lambda x,y:x+y, metrics_sum, metrics)

            loss_keys = ['total','echo','perceptual', 'gt_patch', 'diversity','l1','ssim','edge','moment','color_cov','autocorr','fft']
            initial_metrics = {f'loss/{k}':0. for k in loss_keys}; initial_metrics['learning_rate'] = 0.
            if self.args.use_sentinel: initial_metrics['sentinel_pct'] = 0.
            if self.args.use_q_controller: initial_metrics.update({'status_code':0.,'last_reward':0.,'trend_slope':0.})

            final_state, _, total_metrics = jax.lax.fori_loop(0, rebatch_size, body, (state, key, initial_metrics))
            avg_metrics = jax.tree_util.tree_map(lambda x: x/rebatch_size, total_metrics)
            if self.args.use_q_controller:
                avg_metrics['q_status'],avg_metrics['q_reward'],avg_metrics['q_trend'] = avg_metrics.pop('status_code'),avg_metrics.pop('last_reward'),avg_metrics.pop('trend_slope')
            return final_state, avg_metrics

        preview_data_buffer = [next(iter(ds_base.take(50).batch(1).as_numpy_iterator()))]
        preview_idx = 0
        
        start_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
        global_step = start_step

        self.progress = Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[epochs]}"), BarColumn(), "•", TextColumn("Step {task.completed}/{task.total}"))
        main_task = self.progress.add_task("train", total=total_steps, completed=start_step, epoch=start_epoch+1, epochs=self.args.epochs)
        
        last_time=time.time(); live=Live(self._generate_layout(),screen=True,redirect_stderr=False,vertical_overflow="crop",auto_refresh=False)
        
        try:
            live.start()
            with ThreadPoolExecutor(max_workers=1) as async_pool:
                active_preview_future = None
                for epoch in range(start_epoch, self.args.epochs):
                    if self.interactive_state.shutdown_event.is_set(): break
                    self.console.print(f"\n--- Epoch {epoch+1}/{self.args.epochs} --- Shuffling data with seed {self.args.seed + epoch}... ---")

                    ds = ds_base.shuffle(buffer_size=num_records, seed=self.args.seed + epoch)
                    ds = ds.batch(super_bs, drop_remainder=True)
                    train_iterator = iter(tfds.as_numpy(ds.prefetch(2)))
                    
                    for step_in_epoch in range(steps_per_epoch):
                        if self.interactive_state.shutdown_event.is_set(): break
                        
                        try: super_vec, super_prompts, super_gt_images = next(train_iterator)
                        except StopIteration: break

                        damp = self.interactive_state.get_sentinel_factor(); key, step_key = jax.random.split(key)
                        
                        current_sub_lambdas = self.sub_loss_lambda_controller(self.last_metrics, global_step)
                        current_main_lambdas = self.main_loss_lambda_controller(self.last_metrics, global_step)
                        self.current_lambdas = {**current_main_lambdas, **current_sub_lambdas}

                        sub_lambda_keys = ['l1', 'ssim', 'edge', 'moment', 'color_cov', 'autocorr', 'fft']
                        main_lambda_keys = ['echo', 'perceptual', 'gt_patch', 'diversity']
                        sub_lambdas_for_jit = tuple(current_sub_lambdas[k] for k in sub_lambda_keys)
                        main_lambdas_for_jit = tuple(current_main_lambdas[k] for k in main_lambda_keys)
                        
                        if self.args.use_q_controller:
                            state = state.replace(q_controller_state=state.q_controller_state.replace(step_count=jnp.array(global_step, dtype=jnp.int32)))
                        
                        state, metrics = train_super_step(state, super_vec, super_gt_images, damp, main_lambdas_for_jit, sub_lambdas_for_jit, step_key, self.loss_calculator, student_encoder, fixed_physics_model, self.args.image_size, self.args.rebatch_size, self.args.batch_size)
                        jax.tree_util.tree_map(lambda x: x.block_until_ready(), (state, metrics))

                        time_now=time.time(); self.steps_per_sec=self.args.rebatch_size/(time_now-last_time+1e-6); last_time=time_now
                        global_step += self.args.rebatch_size
                        self.progress.update(main_task, completed=global_step, epoch=epoch+1)
                        
                        metrics_np = jax.device_get(jax.tree_util.tree_map(np.asarray, metrics))
                        with self.ui_lock:
                            self.last_metrics = {k:v.item() if hasattr(v,'item') else v for k,v in metrics_np.items()}
                            if np.isfinite(self.last_metrics['loss/total']): self.loss_hist.append(self.last_metrics['loss/total'])

                        preview_change = self.interactive_state.get_and_reset_preview_change()
                        if preview_change != 0: preview_idx = (preview_idx + preview_change) % len(preview_data_buffer)
                        
                        if active_preview_future is None or active_preview_future.done():
                             if active_preview_future: active_preview_future.result()
                             active_preview_future = async_pool.submit(self._update_preview_task, state.ema_params, fixed_physics_params, preview_data_buffer[preview_idx], student_encoder, fixed_physics_model)
                        
                        live.update(self._generate_layout(), refresh=True)

                        if self.interactive_state.get_and_reset_force_save() or (global_step > 0 and (global_step % self.args.save_every < self.args.rebatch_size)):
                            self.console.print(f"\n--- 💾 Saving checkpoint at step {global_step}... ---")
                            data_to_save = {'params': jax.device_get(state.params), 'ema_params': jax.device_get(state.ema_params),
                                            'opt_state': jax.device_get(state.opt_state), 'q_controller_state': jax.device_get(state.q_controller_state),
                                            'pid_controller_state': self.sub_loss_lambda_controller.state_dict(), 
                                            'main_pid_controller_state': self.main_loss_lambda_controller.state_dict(), 'step': global_step}
                            with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
        finally:
            live.stop(); self.interactive_state.set_shutdown(); key_listener_thread.join(timeout=1)
            self.console.print("\n--- Training finished. ---")
            if 'state' in locals():
                if not self.interactive_state.shutdown_event.is_set():
                    self.console.print("\n--- Saving final model... ---")
                    clean_params={'student_encoder_params':unfreeze(jax.device_get(state.ema_params)),'fixed_physics_params':fixed_physics_params}
                    final_path = Path(f"{self.args.basename}_{self.args.d_model}d_distilled_t2i_echo_gt.pkl")
                    with open(final_path, 'wb') as f: pickle.dump(clean_params, f)
                    self.console.print(f"✅ Self-contained T2I model saved to [green]{final_path}[/green]")
                else:
                    self.console.print(f"\n--- 💾 Saving final checkpoint due to interruption... ---")
                    data = {'params':jax.device_get(state.params),'ema_params':jax.device_get(state.ema_params),
                            'opt_state':jax.device_get(state.opt_state),'q_controller_state':jax.device_get(state.q_controller_state),
                            'pid_controller_state': self.sub_loss_lambda_controller.state_dict(), 
                            'main_pid_controller_state': self.main_loss_lambda_controller.state_dict(), 'step':global_step}
                    with open(ckpt_path, 'wb') as f: pickle.dump(data, f)
                    self.console.print(f"✅ Checkpoint saved to [cyan]{ckpt_path}[/cyan].")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Echo+GT Distillation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--basename',type=str,required=True,help="Basename for model files.")
    parent_parser.add_argument('--d-model',type=int,default=64,help="Model dimension of Phase 1 AE.")
    parent_parser.add_argument('--latent-grid-size',type=int,default=64,help="Latent grid size of Phase 1 AE.")
    parent_parser.add_argument('--image-size',type=int,default=512,help="Image resolution.")

    p_prep = subparsers.add_parser("prepare-distill-data", help="Create aligned TFRecords from a folder of image/.txt pairs (e.g., from coco_preprocessor.py).")
    p_prep.add_argument('--source-dir', type=str, required=True, help="Directory with ALIGNED images and .txt files.")
    p_prep.add_argument('--target-dir', type=str, required=True, help="Directory to save the final TFRecord files.")
    p_prep.add_argument('--text-encoder-id', type=str, default="google/siglip-base-patch16-224", help="Hugging Face ID of the text encoder.")
    
    p_train = subparsers.add_parser("train", help="Distill knowledge using Echo and Ground Truth Loss.", parents=[parent_parser])
    p_train.add_argument('--data-dir',type=str,required=True,help="Directory containing the ALIGNED TFRecord files.")
    p_train.add_argument('--epochs',type=int,default=100)
    p_train.add_argument('--batch-size',type=int,default=1,help="Size of mini-batches processed on GPU at once.")
    p_train.add_argument('--rebatch-size',type=int,default=50,help="Number of mini-batches per super-batch to reduce Python overhead.")
    p_train.add_argument('--lr',type=float,default=2e-4,help="Target learning rate for the Autonomous Scheduler.")
    p_train.add_argument('--seed',type=int,default=42)
    p_train.add_argument('--use-bfloat16',action='store_true')
    p_train.add_argument('--use-q-controller',action='store_true',help="Enable Autonomous LR Scheduler.")
    p_train.add_argument('--use-sentinel',action='store_true',help="Enable Sentinel optimizer.")
    p_train.add_argument('--save-every',type=int,default=2000,help="Save a checkpoint every N steps.")

    args = parser.parse_args()
    if args.command == "prepare-distill-data":
        prepare_distill_data(args.source_dir, args.target_dir, args.text_encoder_id)
    elif args.command == "train":
        trainer = DistillationTrainer(args)
        trainer.train()
    else:
        print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()