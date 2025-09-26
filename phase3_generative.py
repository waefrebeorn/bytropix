# =================================================================================================
#
#        PHASE 3: HYPER-OPTIMIZED ECHO+GT DISTILLATION (AUTONOMOUS VERSION)
#
#       Distills a text encoder's knowledge into a tiny, self-contained, text-to-image
#       generator. This version features autonomous difficulty regularization and
#       validation-based saving for peak performance and robustness.
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
from dataclasses import dataclass, field

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
        self.preview_index_change = 0
        self.shutdown_event, self.force_save = threading.Event(), False
        self.gt_blast_active = False # TOGGLE state
        self.rerun_batch_request = False # ONE-SHOT event

    def get_and_reset_preview_change(self):
        with self.lock: change = self.preview_index_change; self.preview_index_change = 0; return change
    def get_and_reset_force_save(self):
        with self.lock: save = self.force_save; self.force_save = False; return save
    def toggle_gt_blast(self):
        with self.lock: self.gt_blast_active = not self.gt_blast_active
    def get_and_reset_rerun_request(self):
        with self.lock: rerun = self.rerun_batch_request; self.rerun_batch_request = False; return rerun
    def set_shutdown(self): self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    print("--- Key listener started. Controls: [←/→] Preview | [s] Save | [g] Toggle GT Blast | [b] Rerun Batch | [q] Quit ---")
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
            elif key in [b'g', 'g']:
                shared_state.toggle_gt_blast()
            elif key in [b'b', 'b']:
                with shared_state.lock: shared_state.rerun_batch_request = True
            elif key == b'\xe0' or key == '\x1b': # Arrow keys
                arrow = msvcrt.getch() if platform.system() == "Windows" else sys.stdin.read(2)
                if arrow in [b'K', '[D']: shared_state.preview_index_change = -1
                elif arrow in [b'M', '[C']: shared_state.preview_index_change = 1
    finally:
        if platform.system() != "Windows": termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; trend_history: chex.Array
    current_lr: jnp.ndarray; exploration_rate: jnp.ndarray; step_count: jnp.ndarray
    last_action_idx: jnp.ndarray; last_reward: jnp.ndarray; status_code: jnp.ndarray; trend_slope: jnp.ndarray

@dataclass(frozen=True)
class QControllerConfig:
    q_table_size: int = 100; num_lr_actions: int = 5
    lr_change_factors: Tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1)
    learning_rate_q: float = 0.1; discount_factor_q: float = 0.9; lr_min: float = 1e-6; lr_max: float = 1e-3
    metric_history_len: int = 100; loss_min: float = 0.01; loss_max: float = 5.0
    exploration_rate_q: float = 0.3; min_exploration_rate: float = 0.05; exploration_decay: float = 0.9998
    trend_window: int = 50; improve_threshold: float = 1e-5; regress_threshold: float = 1e-6
    regress_penalty: float = 5.0; stagnation_penalty: float = -1.0
    warmup_steps: int = 500; warmup_lr_start: float = 1e-6

@partial(jax.jit, static_argnames=('config', 'target_lr'))
def q_controller_choose_action(state, key, config: QControllerConfig, target_lr):
    def warmup_action():
        alpha = state.step_count.astype(jnp.float32) / config.warmup_steps
        lr = config.warmup_lr_start * (1 - alpha) + target_lr * alpha
        return state.replace(current_lr=lr, step_count=state.step_count + 1, status_code=jnp.array(0))
    def regular_action():
        mean = jnp.mean(jax.lax.dynamic_slice_in_dim(state.metric_history, config.metric_history_len - 10, 10))
        state_idx = jnp.clip(((mean - config.loss_min) / ((config.loss_max - config.loss_min) / config.q_table_size)).astype(jnp.int32), 0, config.q_table_size - 1)
        explore, act = jax.random.split(key)
        action_idx = jax.lax.cond(jax.random.uniform(explore) < state.exploration_rate,
            lambda: jax.random.randint(act, (), 0, config.num_lr_actions),
            lambda: jnp.argmax(state.q_table[state_idx]))
        lr_factors_arr = jnp.array(config.lr_change_factors); lr_multiplier = lr_factors_arr[action_idx]
        new_lr = jnp.clip(state.current_lr * lr_multiplier, config.lr_min, config.lr_max)
        return state.replace(current_lr=new_lr, step_count=state.step_count + 1, last_action_idx=action_idx)
    return jax.lax.cond(state.step_count < config.warmup_steps, warmup_action, regular_action)

def init_q_controller(config: QControllerConfig, initial_lr):
    return QControllerState(q_table=jnp.zeros((config.q_table_size, config.num_lr_actions)),
        metric_history=jnp.full((config.metric_history_len,), (config.loss_min + config.loss_max) / 2),
        trend_history=jnp.zeros((config.trend_window,)), current_lr=jnp.array(config.warmup_lr_start),
        exploration_rate=jnp.array(config.exploration_rate_q), step_count=jnp.array(0),
        last_action_idx=jnp.array(-1), last_reward=jnp.array(0.0), status_code=jnp.array(0), trend_slope=jnp.array(0.0))

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

class PIDLambdaController:
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]], warmup_steps: int = 500, gt_flood_steps: int = 0, ease_in_steps: int = 0):
        self.targets, self.base_weights, self.gains = targets, base_weights, gains
        self.warmup_steps, self.gt_flood_steps, self.ease_in_steps = warmup_steps, gt_flood_steps, ease_in_steps
        self.state = {'integral_error': {k: 0.0 for k in targets}, 'last_error': {k: 0.0 for k in targets}}
    def __call__(self, last_metrics: Dict[str, float], step: int) -> Dict[str, float]:
        if self.gt_flood_steps > 0 and step < self.gt_flood_steps:
            flooded_weights = {k: 0.01 for k in self.base_weights};
            if 'gt_patch' in flooded_weights: flooded_weights['gt_patch'] = 50.0
            return flooded_weights
        pid_lambdas = {}
        for name, base in self.base_weights.items():
            pid_lambdas[name] = float(base)
            if name in self.targets:
                val = last_metrics.get(f'loss/{name}');
                if val is None: continue
                try: current_loss = float(val)
                except (TypeError, ValueError): continue
                kp, ki, kd = self.gains[name]; target = self.targets[name]; error = current_loss - target
                self.state['integral_error'][name] += error; self.state['integral_error'][name] = np.clip(self.state['integral_error'][name], -5.0, 5.0)
                derivative = error - self.state['last_error'][name]
                adj = (kp * error) + (ki * self.state['integral_error'][name]) + (kd * derivative)
                pid_lambdas[name] = float(np.clip(base * np.exp(adj), 0.1, 50.0)); self.state['last_error'][name] = error
        if self.ease_in_steps > 0 and step >= self.gt_flood_steps and step < (self.gt_flood_steps + self.ease_in_steps):
            progress = (step - self.gt_flood_steps) / self.ease_in_steps; eased_lambdas = {}
            for name, pid_val in pid_lambdas.items(): eased_lambdas[name] = 0.01 * (1.0 - progress) + pid_val * progress
            return eased_lambdas
        if step < self.warmup_steps: return self.base_weights
        return pid_lambdas
    def state_dict(self): return self.state
    def load_state_dict(self, state):
        self.state['integral_error'] = state.get('integral_error', {k: 0.0 for k in self.targets})
        self.state['last_error'] = state.get('last_error', {k: 0.0 for k in self.targets})

@struct.dataclass
class PatchBufferState:
    patches1_buffer: chex.Array; patches2_buffer: chex.Array

class PatchBuffer:
    def __init__(self, buffer_size: int, patch_shape: Tuple[int, ...]):
        full_buffer_shape = (buffer_size,) + patch_shape
        self.state = PatchBufferState(
            patches1_buffer=jnp.zeros(full_buffer_shape, dtype=jnp.float32),
            patches2_buffer=jnp.zeros(full_buffer_shape, dtype=jnp.float32))
    @partial(jit, static_argnames=('self',))
    def update_and_get(self, state: PatchBufferState, new_p1: chex.Array, new_p2: chex.Array) -> Tuple[PatchBufferState, chex.Array, chex.Array]:
        num_new_patches = new_p1.shape[0]
        p1_rolled = jnp.roll(state.patches1_buffer, shift=-num_new_patches, axis=0)
        p2_rolled = jnp.roll(state.patches2_buffer, shift=-num_new_patches, axis=0)
        start_index = state.patches1_buffer.shape[0] - num_new_patches
        new_p1_buffer = jax.lax.dynamic_update_slice_in_dim(p1_rolled, new_p1.astype(jnp.float32), start_index, axis=0)
        new_p2_buffer = jax.lax.dynamic_update_slice_in_dim(p2_rolled, new_p2.astype(jnp.float32), start_index, axis=0)
        new_state = PatchBufferState(patches1_buffer=new_p1_buffer, patches2_buffer=new_p2_buffer)
        return new_state, new_p1_buffer, new_p2_buffer

class CustomTrainState(train_state.TrainState):
    q_controller_state: QControllerState
    patch_buffer_state: PatchBufferState
    def apply_gradients(self, *, grads: Any, **kwargs) -> "CustomTrainState":
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, **kwargs)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)

# =================================================================================================
# 2. MODEL DEFINITIONS & PERCEPTUAL LOSS TOOLKIT
# =================================================================================================
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
    cb_unscaled, cr_unscaled = cb / 2.0, cr / 2.0
    r = y + 1.402 * cr_unscaled
    g = y - 0.344136 * cb_unscaled - 0.714136 * cr_unscaled
    b = y + 1.772 * cb_unscaled
    return jnp.clip(jnp.stack([r, g, b], axis=-1), -1.0, 1.0)
class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2); imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part
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
        x = nn.Dense(4 * 4 * 1024, dtype=self.dtype, name="input_proj")(pooled); x = nn.gelu(x)
        x = x.reshape(-1, 4, 4, 1024)
        features = [512, 256, 128, 64]; current_res = 4
        for i, feats in enumerate(features):
            if current_res >= self.latent_grid_size: break
            x = nn.ConvTranspose(feats, (4, 4), (2, 2), 'SAME', dtype=self.dtype, name=f"upsample_{i}")(x)
            x = nn.LayerNorm(dtype=self.dtype)(x); x = nn.gelu(x); current_res *= 2
        p = nn.Conv(9, (3, 3), padding='SAME', dtype=self.dtype, name="output_conv", bias_init=lambda k,s,d:jnp.zeros(s,d).at[jnp.array([2,5,8])].set(0.))(x)
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
    return jnp.linalg.norm(jnp.transpose(magnitudes_nchw, (0, 2, 3, 1)), axis=-1)
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
@partial(jit, static_argnames=('num_samples', 'patch_size'))
def golden_spiral_sample(key: chex.PRNGKey, h: int, w: int, num_samples: int, patch_size: int) -> Tuple[chex.Array, chex.Array]:
    golden_angle = jnp.pi * (3. - jnp.sqrt(5.)); i = jnp.arange(num_samples, dtype=jnp.float32)
    offset_key, rotation_key = jax.random.split(key); offset = jax.random.uniform(offset_key) * num_samples
    theta = golden_angle * (i + offset); radius = jnp.sqrt(i / num_samples)
    rotation_angle = jax.random.uniform(rotation_key, minval=0, maxval=2*jnp.pi)
    x = 0.5 + radius * jnp.cos(theta + rotation_angle); y = 0.5 + radius * jnp.sin(theta + rotation_angle)
    x_coords = jnp.clip((x * w).astype(jnp.int32), 0, w - patch_size)
    y_coords = jnp.clip((y * h).astype(jnp.int32), 0, h - patch_size)
    return y_coords, x_coords

class JAXMultiMetricPerceptualLoss:
    def __init__(self, patch_size=32):
        self.patch_size = patch_size
        image_area = 512 * 512; patch_area = patch_size * patch_size
        self.num_patches = int((image_area * 0.25) / patch_area)
    @partial(jit, static_argnames=('self', 'use_buffer'))
    def __call__(self, img1, img2, key, patch_buffer_state=None, use_buffer=False):
        _, h, w, c = img1.shape; batch_size = img1.shape[0]; keys = jax.random.split(key, batch_size)
        @partial(jax.vmap, in_axes=(0, 0, 0))
        def get_patches_for_image(single_img1, single_img2, k):
            y_coords, x_coords = golden_spiral_sample(k, h, w, self.num_patches, self.patch_size)
            p1 = jax.vmap(lambda y, x: jax.lax.dynamic_slice(single_img1, (y, x, 0), (self.patch_size, self.patch_size, c)))(y_coords, x_coords)
            p2 = jax.vmap(lambda y, x: jax.lax.dynamic_slice(single_img2, (y, x, 0), (self.patch_size, self.patch_size, c)))(y_coords, x_coords)
            return p1, p2
        p1_current, p2_current = get_patches_for_image(img1, img2, keys)
        p1_current = p1_current.reshape(-1, self.patch_size, self.patch_size, c)
        p2_current = p2_current.reshape(-1, self.patch_size, self.patch_size, c)
        p1_for_loss, p2_for_loss = p1_current, p2_current
        if use_buffer and patch_buffer_state is not None:
            p1_for_loss = jnp.concatenate([p1_current, patch_buffer_state.patches1_buffer], axis=0)
            p2_for_loss = jnp.concatenate([p2_current, patch_buffer_state.patches2_buffer], axis=0)
        losses = {'l1': jnp.mean(jnp.abs(p1_for_loss - p2_for_loss)),
            'moment': jnp.mean(jnp.abs(calculate_moments(p1_for_loss) - calculate_moments(p2_for_loss))),
            'fft': jnp.mean(jnp.abs(fft_magnitude_log(jnp.mean(p1_for_loss, -1)) - fft_magnitude_log(jnp.mean(p2_for_loss, -1)))),
            'autocorr': jnp.mean(jnp.abs(calculate_autocorrelation_features(p1_for_loss) - calculate_autocorrelation_features(p2_for_loss))),
            'edge': calculate_edge_loss(p1_for_loss, p2_for_loss),
            'color_cov': calculate_color_covariance_loss(p1_for_loss, p2_for_loss),
            'ssim': calculate_ssim_loss(p1_for_loss, p2_for_loss)}
        return {f'loss/{k}': v for k, v in losses.items()}, (p1_current, p2_current)

class WeightAverager:
    def __init__(self, buffer_size=16):
        self.buffer_size = buffer_size; self.buffer = None; self.current_idx = 0; self.steps_since_init = 0
    def init_buffer(self, example_params):
        self.buffer = jax.tree_util.tree_map(lambda x: jnp.zeros((self.buffer_size,) + x.shape, dtype=x.dtype), example_params)
    def update(self, new_params):
        if self.buffer is None: self.init_buffer(new_params)
        self.buffer = jax.tree_util.tree_map(lambda buf, new: buf.at[self.current_idx].set(new), self.buffer, new_params)
        self.current_idx = (self.current_idx + 1) % self.buffer_size; self.steps_since_init += 1
    @partial(jit, static_argnames=('self',))
    def _get_averaged_params_jitted(self, buffer, num_valid_entries):
        def create_and_apply_mask(buf):
            mask_1d = jnp.arange(self.buffer_size) < num_valid_entries
            mask = mask_1d.reshape((self.buffer_size,) + (1,) * (buf.ndim - 1))
            return jnp.sum(buf * mask, axis=0)
        summed_params = jax.tree_util.tree_map(create_and_apply_mask, buffer)
        averaged_params = jax.tree_util.tree_map(lambda s: s / jnp.maximum(1, num_valid_entries), summed_params)
        return averaged_params
    def get_averaged_params(self):
        if self.buffer is None: return None
        num_valid_entries = jnp.minimum(self.steps_since_init, self.buffer_size)
        return self._get_averaged_params_jitted(self.buffer, num_valid_entries)

def apply_blur_schedule(global_step: jnp.ndarray, schedule: Tuple[int, int, int, float, float]) -> jnp.ndarray:
    warmup_start, warmup_end, cooldown_end, max_sigma, min_sigma = schedule; step = global_step.astype(jnp.float32)
    is_in_warmup = (step >= warmup_start) & (step < warmup_end); is_in_cooldown = (step >= warmup_end) & (step < cooldown_end)
    warmup_progress = jnp.clip((step - warmup_start) / (warmup_end - warmup_start + 1e-6), 0.0, 1.0)
    cooldown_progress = jnp.clip((step - warmup_end) / (cooldown_end - warmup_end + 1e-6), 0.0, 1.0)
    sigma = jnp.where(is_in_warmup, min_sigma + warmup_progress * (max_sigma - min_sigma),
        jnp.where(is_in_cooldown, max_sigma - cooldown_progress * (max_sigma - min_sigma), min_sigma))
    return sigma

def apply_gaussian_blur(images: jnp.ndarray, sigma: jnp.ndarray, key: chex.PRNGKey) -> jnp.ndarray:
    MAX_KERNEL_SIZE = 11
    def _blur_op(img_batch):
        radius = (MAX_KERNEL_SIZE - 1) // 2; x = jnp.arange(-radius, radius + 1).astype(jnp.float32)
        kernel_1d = jnp.exp(-0.5 * (x / sigma)**2); current_radius = jnp.ceil(sigma * 2)
        mask = jnp.abs(x) <= current_radius; masked_kernel = kernel_1d * mask
        kernel_1d_normalized_f32 = masked_kernel / (jnp.sum(masked_kernel) + 1e-6)
        kernel_1d_normalized = kernel_1d_normalized_f32.astype(img_batch.dtype)
        num_channels = img_batch.shape[-1]
        kernel_h = jnp.tile(kernel_1d_normalized.reshape(1, MAX_KERNEL_SIZE, 1, 1), (1, 1, 1, num_channels))
        kernel_v = jnp.tile(kernel_1d_normalized.reshape(MAX_KERNEL_SIZE, 1, 1, 1), (1, 1, 1, num_channels))
        img_nchw = img_batch.transpose(0, 3, 1, 2); dn = ('NCHW', 'HWIO', 'NCHW')
        blurred_h = jax.lax.conv_general_dilated(img_nchw, kernel_h, window_strides=(1, 1), padding='SAME', feature_group_count=num_channels, dimension_numbers=dn)
        blurred_v = jax.lax.conv_general_dilated(blurred_h, kernel_v, window_strides=(1, 1), padding='SAME', feature_group_count=num_channels, dimension_numbers=dn)
        return blurred_v.transpose(0, 2, 3, 1)
    return jax.lax.cond(sigma > 0.01, _blur_op, lambda x: x, images)

# =================================================================================================
# 3. UNIFIED & ROBUST DATA PREPARATION
# =================================================================================================
def prepare_distill_data(source_dir: str, target_dir: str, text_encoder_id: str):
    console = Console()
    if not all([SiglipImageProcessor, SiglipTextModel, SiglipTokenizer, torch]):
        console.print("[bold red]FATAL: `transformers` and `torch` are required. Please install them.[/bold red]"); sys.exit(1)
    source_path, target_path = Path(source_dir), Path(target_dir); target_path.mkdir(exist_ok=True)
    console.print(f"--- 🔍 Scanning for aligned image-text pairs in [cyan]{source_path}[/cyan]... ---")
    image_paths = sorted(list(source_path.rglob('*.jpg')) + list(source_path.rglob('*.png')) + list(source_path.rglob('*.webp')))
    aligned_pairs = [];
    for img_path in tqdm(image_paths, desc="Verifying pairs"):
        if (txt_path := img_path.with_suffix('.txt')).exists(): aligned_pairs.append((img_path, txt_path))
    if not aligned_pairs: console.print(f"[bold red]FATAL: No aligned .jpg/.txt pairs found in {source_path}.[/bold red]"); sys.exit(1)
    console.print(f"--- ✅ Found {len(aligned_pairs)} aligned pairs. ---")
    console.print(f"--- 🧠 Loading Text Encoder: [yellow]{text_encoder_id}[/yellow]... ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_model = SiglipTextModel.from_pretrained(text_encoder_id).to(device)
    tokenizer = SiglipTokenizer.from_pretrained(text_encoder_id); text_model.eval()
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
# 4. DISTILLATION TRAINER & HYPER-OPTIMIZED TRAINING STEP
# =================================================================================================
@partial(jax.jit, static_argnames=('trainer', 'image_size', 'q_config_static', 'physics_model_static', 'student_encoder_static', 'blur_schedule'))
def train_step(state, global_step, super_command_vec, super_gt_images, main_lambdas_tuple, sub_lambdas_tuple, key, trainer, image_size, q_config_static, physics_model_static, student_encoder_static, blur_schedule):
    lambda_gt_patch, lambda_echo, lambda_perceptual, lambda_diversity = main_lambdas_tuple
    lambda_l1, lambda_ssim, lambda_edge, lambda_moment, lambda_color_cov, lambda_autocorr, lambda_fft = sub_lambdas_tuple
    grad_key, q_key, teacher_key, student_key, blur_key = jax.random.split(key, 5)

    def loss_fn(params):
        student_path_params = student_encoder_static.apply({'params': params}, super_command_vec)
        student_img = trainer._generate_target_image(trainer.fixed_physics_params, student_path_params, physics_model_static, image_size)
        teacher_path_params = student_path_params + jax.random.normal(teacher_key, student_path_params.shape) * 1e-2
        teacher_img = trainer._generate_target_image(trainer.fixed_physics_params, teacher_path_params, physics_model_static, image_size)
        echo_img = jax.lax.stop_gradient(teacher_img)
        perceptual_metrics_gt, (p1_new, p2_new) = trainer.loss_calculator(student_img,super_gt_images,student_key,patch_buffer_state=state.patch_buffer_state,use_buffer=True)
        blur_sigma = apply_blur_schedule(global_step, blur_schedule)
        gt_patch_loss = perceptual_metrics_gt['loss/l1']
        total_perceptual_loss = ( lambda_ssim * perceptual_metrics_gt['loss/ssim'] + lambda_edge * perceptual_metrics_gt['loss/edge'] + lambda_moment * perceptual_metrics_gt['loss/moment'] + lambda_color_cov * perceptual_metrics_gt['loss/color_cov'] + lambda_autocorr * perceptual_metrics_gt['loss/autocorr'] + lambda_fft * perceptual_metrics_gt['loss/fft'] )
        echo_loss = jnp.mean(jnp.abs(student_img - echo_img))
        diversity_loss = -jnp.mean(jnp.std(student_path_params, axis=(1, 2)))
        total_loss = ( lambda_gt_patch * gt_patch_loss + lambda_perceptual * total_perceptual_loss + lambda_echo * echo_loss + lambda_diversity * diversity_loss )
        metrics = { 'loss/total': total_loss, 'loss/gt_patch': gt_patch_loss, 'loss/perceptual': total_perceptual_loss, 'loss/echo': echo_loss, 'loss/diversity': diversity_loss, 'blur_sigma': blur_sigma, 'latent_mean': jnp.mean(student_path_params), 'latent_std': jnp.mean(jnp.std(student_path_params, axis=(1,2,3))) }
        metrics.update(perceptual_metrics_gt)
        return total_loss, (metrics, student_img, p1_new, p2_new)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, generated_image, p1_new, p2_new)), grads = grad_fn(state.params)
    apply_kwargs = {}; new_q_state = state.q_controller_state
    if trainer.args.use_q_controller:
        q_state_slice = q_controller_choose_action(state.q_controller_state, q_key, q_config_static, trainer.args.lr)
        apply_kwargs['learning_rate'] = q_state_slice.current_lr; new_q_state = q_state_slice
    new_state = state.apply_gradients(grads=grads, **apply_kwargs)
    new_patch_buffer_state, _, _ = trainer.patch_buffer.update_and_get(state.patch_buffer_state, p1_new, p2_new)
    if trainer.args.use_q_controller:
        new_q_state = q_controller_update(new_q_state, metrics['loss/total'], q_config_static)
        new_state = new_state.replace(q_controller_state=new_q_state, patch_buffer_state=new_patch_buffer_state)
        metrics['q_status'] = new_q_state.status_code; metrics['learning_rate'] = new_q_state.current_lr
    else:
        new_state = new_state.replace(patch_buffer_state=new_patch_buffer_state)
        metrics['learning_rate'] = jnp.array(trainer.args.lr)
    return new_state, metrics, generated_image

# --- NEW: JIT-compiled validation step ---
@partial(jax.jit, static_argnames=('trainer', 'image_size', 'physics_model_static', 'student_encoder_static'))
def validation_step(student_params, command_vec, gt_images, trainer, image_size, physics_model_static, student_encoder_static):
    student_path_params = student_encoder_static.apply({'params': student_params}, command_vec)
    student_img = trainer._generate_target_image(trainer.fixed_physics_params, student_path_params, physics_model_static, image_size)
    l1_loss = jnp.mean(jnp.abs(student_img - gt_images))
    return l1_loss

class DistillationTrainer:
    def __init__(self, args, fixed_physics_params):
        self.args = args; self.console = Console()
        self.fixed_physics_params = fixed_physics_params
        self.interactive_state = InteractivityState()
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.loss_calculator = JAXMultiMetricPerceptualLoss(patch_size=32)
        PATCH_BUFFER_SIZE = 4096; PATCH_SHAPE = (self.loss_calculator.patch_size, self.loss_calculator.patch_size, 3)
        self.patch_buffer = PatchBuffer(buffer_size=PATCH_BUFFER_SIZE, patch_shape=PATCH_SHAPE)
        self.console.print(f"--- 🧠 Perceptual Memory: Patch buffer enabled (size: [cyan]{PATCH_BUFFER_SIZE}[/cyan]). ---")
        self.weight_averager = WeightAverager(buffer_size=16)
        self.console.print(f"--- 💡 DFT-Inspired Smoothing: Averaging last [cyan]16[/cyan] weight states for previews. ---")
        self.blur_schedule = (1, 10000, 10000, 1.0, 0.0)
        self.console.print(f"--- Curriculum: Progressive blur active from steps {self.blur_schedule[0]}-{self.blur_schedule[2]} (max sigma: {self.blur_schedule[3]}). ---")
        self.physics_model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=self.dtype)
        if args.use_q_controller: self.q_controller_config = QControllerConfig()
        sub_loss_pid_gains = {'l1':(0.8,0.01,1.0),'ssim':(1.5,0.02,2.0),'edge':(1.2,0.01,1.5),'moment':(0.5,0.005,0.8),'color_cov':(0.7,0.005,1.0),'autocorr':(0.6,0.005,0.9),'fft':(0.4,0.005,0.5)}
        self.sub_loss_lambda_controller = PIDLambdaController(targets={'l1':0.01,'ssim':0.1,'edge':0.15,'moment':0.15,'color_cov':0.02,'autocorr':0.15,'fft':0.1}, base_weights={'l1':1.5,'ssim':1.2,'edge':1.0,'moment':0.8,'color_cov':0.9,'autocorr':0.2,'fft':0.2}, gains=sub_loss_pid_gains, warmup_steps=1000)
        main_loss_pid_gains = {'gt_patch': (1.5, 0.05, 1.2), 'echo': (0.5, 0.01, 0.7), 'perceptual': (0.4, 0.01, 0.6), 'diversity': (0.5, 0.005, 0.6)}
        self.main_loss_lambda_controller = PIDLambdaController(targets={'gt_patch': 0.05, 'echo': 0.25, 'perceptual': 0.35, 'diversity': -0.6}, base_weights={'gt_patch': 15.0, 'echo': 0.4, 'perceptual': 0.4, 'diversity': 0.02}, gains=main_loss_pid_gains, warmup_steps=500, gt_flood_steps=5000)
        self.text_encoder_id = "google/siglip-base-patch16-224"; self.SEQUENCE_LENGTH=64; self.EXPECTED_EMBEDDING_DIM=768
        self.param_count = 0; self.last_metrics = {}; self.loss_hist = deque(maxlen=200); self.current_lambdas = {}
        self.steps_per_sec = 0.0; self.ui_lock = threading.Lock()
        self.live_preview_image_np = None; self.live_preview_prompt = "..."
        self.current_step_image_np = None; self.current_prompt = "..."
        
        # --- NEW: Autonomous Regulation State ---
        self.intervention_status = ""
        self.adr_active = False
        self.adr_buffer = []
        self.adr_idx = 0
        self.adr_loss_history = deque(maxlen=50) # For averaging
        self.HIGH_LOSS_THRESHOLD = 30.0
        self.LOW_LOSS_THRESHOLD = 13.0
        self.ADR_BUFFER_SIZE = 50
        self.best_val_score = -1.0
        self.current_val_score = -1.0

        self.jitted_train_step = partial(train_step, trainer=self, image_size=self.args.image_size,
            q_config_static=self.q_controller_config if self.args.use_q_controller else None,
            physics_model_static=self.physics_model,
            student_encoder_static=TextToPathParamsProjector(self.args.latent_grid_size,self.dtype),
            blur_schedule=self.blur_schedule)
        self.jitted_validation_step = partial(validation_step, trainer=self, image_size=self.args.image_size,
            physics_model_static=self.physics_model,
            student_encoder_static=TextToPathParamsProjector(self.args.latent_grid_size, self.dtype))

    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB",f"{u.gpu}%"
        except: return "N/A","N/A"
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist) < 2: return " " * w
        hist = hist[-w:]; min_v, max_v = hist.min(), hist.max();
        if max_v == min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1); return "".join(s[i] for i in indices)
        
    def _run_validation(self, student_params_avg, validation_set):
        self.console.print("\n--- 🤖 Running validation... ---")
        total_loss = 0.0
        for val_emb, _, val_gt in tqdm(validation_set, desc="Validating", leave=False):
            loss = self.jitted_validation_step(student_params_avg, val_emb, val_gt)
            total_loss += loss
        avg_loss = total_loss / len(validation_set)
        score = 1.0 - avg_loss # Higher is better
        return float(score)

    def _generate_layout(self):
        with self.ui_lock:
            layout = Layout(name="root"); layout.split(Layout(name="header",size=3), Layout(ratio=1,name="main"), Layout(name="footer",size=3))
            layout["main"].split_row(Layout(name="left",minimum_size=60), Layout(name="right",ratio=1))
            header = f"🚀⚡ [bold]Hyper-Optimized Distillation[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/]"
            layout["header"].update(Panel(Align.center(header), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right"))
            stats_tbl = Table.grid(expand=True,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
            mem,util = self._get_gpu_stats(); stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🚀"); stats_tbl.add_row("GPU Mem/Util", f"[yellow]{mem}[/] / [yellow]{util}[/]")
            lr = self.last_metrics.get('learning_rate', self.args.lr); stats_tbl.add_row("Learning Rate", f"[green]{float(lr):.2e}[/]")
            blur_sigma = self.last_metrics.get('blur_sigma', 0.0); stats_tbl.add_row("Blur Sigma (σ)", f"[magenta]{float(blur_sigma):.2f}[/]")
            
            # --- NEW: Validation Score Display ---
            if self.best_val_score > -1:
                stats_tbl.add_row("Val Score", f"[cyan]{self.current_val_score:.4f}[/] (Best: [bold green]{self.best_val_score:.4f}[/])")
            else:
                stats_tbl.add_row("Val Score", "[dim]N/A[/]")

            left_panels = [Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue")]
            loss_table = Table(show_header=False, box=None, padding=(0,1)); loss_table.add_column(style="cyan",width=12); loss_table.add_column(justify="right",style="white",width=10); loss_table.add_column(justify="right",style="yellow")
            loss_table.add_row("[bold]Metric[/bold]", "[bold]Value[/bold]", "[bold]λ (PID)[/bold]")
            loss_keys = ['gt_patch', 'echo', 'perceptual', 'diversity', 'l1', 'ssim', 'edge', 'moment', 'color_cov', 'autocorr', 'fft']
            for key in loss_keys:
                value = self.last_metrics.get(f'loss/{key}', 0.0); display_name = key.replace('_gt', ' GT').capitalize(); is_main = key in ['gt_patch', 'echo', 'perceptual', 'diversity']
                if is_main: display_name_styled = f"[bold]{display_name}[/bold]"
                else: display_name_styled = display_name
                loss_table.add_row(display_name_styled, f"{value:.4f}", f"{self.current_lambdas.get(key, 0.0):.2f}")
            loss_panel = Panel(loss_table, title="[bold]Loss Components[/]", border_style="cyan"); left_panels.append(loss_panel)
            if self.args.use_q_controller:
                q_code=int(self.last_metrics.get('q_status',0)); q_status={0:"[blue]WARMUP",1:"[green]IMPROVING",2:"[yellow]STAGNATED",3:"[red]REGRESSING"}.get(q_code,"[dim]N/A[/dim]")
                q_panel = Panel(Align.center(q_status), title="[bold]Autonomous LR Scheduler 🧠[/]", border_style="green", height=3); left_panels.append(q_panel)
            if self.intervention_status:
                intervention_panel = Panel(Align.center(f"[bold yellow]{self.intervention_status}[/bold yellow]"), title="[bold red]🔴 Live Interventions[/bold red]", border_style="red", height=3)
                left_panels.append(intervention_panel)
            layout["left"].update(Group(*left_panels))
            total_loss = self.last_metrics.get('loss/total', 0.0)
            spark = Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_hist,60)}[/]"), title=f"Total Loss: {total_loss:.4f}", height=3, border_style="cyan")
            prompt_panel = Panel(Text(self.current_prompt, justify="center"), title="[bold]Live Prompt (Current Batch)[/]", border_style="green")
            current_img_render = Text("...", justify="center")
            if self.current_step_image_np is not None and Pixels:
                term_w=48; h,w,_=self.current_step_image_np.shape; term_h=int(term_w*(h/w)*0.5)
                current_img_render = Pixels.from_image(Image.fromarray(self.current_step_image_np).resize((term_w, term_h), Image.NEAREST))
            current_img_panel = Panel(Align.center(current_img_render), title="Generated Image (Live Weights)")
            live_prompt_panel = Panel(Text(self.live_preview_prompt, justify="center"), title="[bold]Live Preview Prompt (←/→)[/]", border_style="yellow")
            live_img_render = Text("...", justify="center")
            if self.live_preview_image_np is not None and Pixels:
                term_w=48; h,w,_=self.live_preview_image_np.shape; term_h=int(term_w*(h/w)*0.5)
                live_img_render = Pixels.from_image(Image.fromarray(self.live_preview_image_np).resize((term_w, term_h), Image.LANCZOS))
            live_img_panel = Panel(Align.center(live_img_render), title="Live Preview Image (Averaged Weights)")
            layout["right"].update(Group(spark, prompt_panel, current_img_panel, live_prompt_panel, live_img_panel))
            layout["footer"].update(self.progress); return layout

    def _generate_target_image(self, fixed_decoder_params, path_params, fixed_decoder_model, resolution):
         coords = jnp.mgrid[-1:1:resolution*1j,-1:1:resolution*1j].transpose(1,2,0).reshape(-1,2)
         pixels = fixed_decoder_model.apply({'params': fixed_decoder_params}, path_params, coords, method='decode_from_path_params')
         return pixels.reshape(path_params.shape[0], resolution, resolution, 3)
    @partial(jit, static_argnames=('self','student_encoder','resolution'))
    def _generate_preview_jitted(self, student_params, command_vec, student_encoder, resolution):
         paths = student_encoder.apply({'params': student_params}, command_vec)
         coords = jnp.mgrid[-1:1:resolution*1j,-1:1:resolution*1j].transpose(1,2,0).reshape(-1,2)
         pixels = self.physics_model.apply({'params': self.fixed_physics_params}, paths, coords, method='decode_from_path_params')
         return pixels.reshape(paths.shape[0], resolution, resolution, 3)
    def _update_live_preview_task(self, student_params_avg, preview_data, student_encoder):
        val_emb, val_prompt, _ = preview_data
        val_img = self._generate_preview_jitted(student_params_avg, val_emb, student_encoder, 128)
        val_img.block_until_ready()
        with self.ui_lock:
            self.live_preview_prompt = val_prompt[0].decode('utf-8')
            self.live_preview_image_np = ((np.array(val_img[0])*0.5+0.5)*255).clip(0,255).astype(np.uint8)

    def train(self):
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
        safe_name = self.text_encoder_id.replace('/','_'); distill_record_file = Path(self.args.data_dir)/f"distill_data_{safe_name}.tfrecord"; image_record_file = Path(self.args.data_dir)/"data_512x512.tfrecord"
        if not distill_record_file.exists() or not image_record_file.exists(): self.console.print("[bold red]FATAL: TFRecord files not found![/bold red]"); sys.exit(1)
        num_records = sum(1 for _ in tf.data.TFRecordDataset(str(distill_record_file))); super_bs = self.args.batch_size*self.args.rebatch_size; steps_per_epoch=num_records//super_bs; total_steps=steps_per_epoch*self.args.epochs if steps_per_epoch > 0 else 0
        self.console.print(f"--- 🚀 Super-Batch size: {super_bs}. ---"); self.console.print(f"--- Found {num_records} samples. Total steps: {total_steps} ({steps_per_epoch} steps/epoch) ---")
        emb_shape = (self.SEQUENCE_LENGTH, self.EXPECTED_EMBEDDING_DIM)
        def _parse_distill(proto): features={'embedding':tf.io.FixedLenFeature([],tf.string),'prompt':tf.io.FixedLenFeature([],tf.string)}; p=tf.io.parse_single_example(proto,features); emb=tf.cast(tf.reshape(tf.io.decode_raw(p['embedding'],tf.float16),emb_shape),self.dtype); return emb, p['prompt']
        def _parse_image(proto): features={'image':tf.io.FixedLenFeature([],tf.string)}; p=tf.io.parse_single_example(proto,features); img=tf.io.decode_jpeg(p['image'],channels=3); img=tf.image.resize(img,[self.args.image_size,self.args.image_size],method='area'); return tf.cast(img,self.dtype)/127.5-1.0
        ds_base = tf.data.Dataset.zip((tf.data.TFRecordDataset(str(distill_record_file)), tf.data.TFRecordDataset(str(image_record_file))))
        def _parse_combined(distill_proto, image_proto): emb,prompt=_parse_distill(distill_proto); img=_parse_image(image_proto); return emb,prompt,img
        
        # --- NEW: Create Validation Set ---
        all_preview_data = [_parse_combined(*item) for item in list(ds_base.take(50).as_numpy_iterator())]
        preview_data_buffer = [(np.expand_dims(e,0),np.expand_dims(p,0),np.expand_dims(i,0)) for e,p,i in all_preview_data[:40]]
        validation_set = [(np.expand_dims(e,0),np.expand_dims(p,0),np.expand_dims(i,0)) for e,p,i in all_preview_data[40:]]
        self.console.print(f"--- 📊 Validation set created with {len(validation_set)} samples. ---")

        WARMUP_STEPS = 10000; WARMUP_BATCHES_COUNT = 100
        self.console.print(f"--- 🧠 [bold yellow]Staggered Bloom Warmup[/] enabled for the first {WARMUP_STEPS} steps. ---"); self.console.print(f"--- Pre-loading {WARMUP_BATCHES_COUNT} batches for the curriculum... ---")
        warmup_batches_tf = list(ds_base.shuffle(10000, seed=self.args.seed).map(_parse_combined).batch(super_bs, drop_remainder=True).take(WARMUP_BATCHES_COUNT).as_numpy_iterator())
        warmup_batches_np = [(e, p, i) for e, p, i in warmup_batches_tf]; self.console.print(f"--- ✅ {len(warmup_batches_np)} batches loaded into memory for warmup. ---")
        student_encoder = TextToPathParamsProjector(self.args.latent_grid_size,self.dtype)
        lr_for_init = self.args.lr if not self.args.use_q_controller else self.q_controller_config.lr_min
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adamw)(learning_rate=lr_for_init))
        key = jax.random.PRNGKey(self.args.seed); ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_distilled_t2i.ckpt")
        best_ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_distilled_t2i_best.pkl")
        self.console.print("--- Initializing model state... ---")
        dummy_input = jnp.zeros((1,self.SEQUENCE_LENGTH,self.EXPECTED_EMBEDDING_DIM),self.dtype)
        state = CustomTrainState.create(apply_fn=student_encoder.apply,params=student_encoder.init(key, dummy_input)['params'],tx=optimizer,q_controller_state=init_q_controller(self.q_controller_config, self.args.lr) if self.args.use_q_controller else None,patch_buffer_state=self.patch_buffer.state)
        start_step = 0
        if ckpt_path.exists():
            self.console.print(f"--- Loading checkpoint: [cyan]{ckpt_path}[/cyan] ---");
            with open(ckpt_path,'rb') as f: data=pickle.load(f)
            start_step=data.get('step',0); self.best_val_score=data.get('best_val_score', -1.0)
            patch_buffer_state_from_ckpt = data.get('patch_buffer_state', self.patch_buffer.state)
            state=state.replace(params=data['params'], step=start_step, opt_state=data['opt_state'], patch_buffer_state=patch_buffer_state_from_ckpt)
            if self.args.use_q_controller and 'q_controller_state' in data: state = state.replace(q_controller_state=data['q_controller_state'])
            self.console.print(f"--- Resuming from step {start_step} (Best Val Score: {self.best_val_score:.4f}) ---")
        else: self.console.print("--- No checkpoint found. Starting from scratch. ---")
        with self.ui_lock: self.param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
        preview_idx = 0; global_step = start_step
        self.progress = Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[epochs]}"),BarColumn(),"•",TextColumn("Step {task.completed}/{task.total}"))
        main_task = self.progress.add_task("train",total=total_steps,completed=start_step,epoch=global_step//steps_per_epoch+1 if steps_per_epoch>0 else 1,epochs=self.args.epochs)
        last_time, last_ui_update_time, last_val_time = time.time(), 0.0, time.time()
        main_ds_iterator = iter(tfds.as_numpy(ds_base.shuffle(10000, seed=self.args.seed).map(_parse_combined).repeat().batch(super_bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)))
        live = Live(self._generate_layout(),screen=True,redirect_stderr=False,vertical_overflow="crop",auto_refresh=False)
        try:
            live.start()
            with ThreadPoolExecutor(max_workers=1) as async_pool:
                active_preview_future = None
                while global_step < total_steps:
                    if self.interactive_state.shutdown_event.is_set(): break
                    
                    # --- AUTONOMOUS DIFFICULTY REGULATOR (ADR) LOGIC ---
                    if not self.adr_active and (self.last_metrics.get('loss/total', 0) > self.HIGH_LOSS_THRESHOLD and global_step > WARMUP_STEPS):
                        self.adr_active = True
                        self.adr_buffer = [next(main_ds_iterator) for _ in range(self.ADR_BUFFER_SIZE)]
                        self.adr_idx = 0
                        self.adr_loss_history.clear()
                    
                    current_status = ""
                    if self.adr_active:
                        super_vec, super_prompts, super_gt_images = self.adr_buffer[self.adr_idx]
                        self.adr_idx = (self.adr_idx + 1) % self.ADR_BUFFER_SIZE
                        avg_adr_loss = np.mean(self.adr_loss_history) if self.adr_loss_history else self.HIGH_LOSS_THRESHOLD
                        current_status = f"ADR ACTIVE 🔁 (Loss: {avg_adr_loss:.2f} / Target: <{self.LOW_LOSS_THRESHOLD})"
                        if avg_adr_loss < self.LOW_LOSS_THRESHOLD:
                            self.adr_active = False
                            current_status = "ADR Deactivated ✅"
                    else: # Normal operation
                        if global_step < WARMUP_STEPS: # curriculum learning
                            # ... (warmup logic remains the same)
                            stage = min((global_step * 10) // WARMUP_STEPS, 9); frontier_start_idx, frontier_end_idx = stage * 10, (stage + 1) * 10; reinforcement_end_idx = stage * 10
                            frontier_indices = np.random.randint(frontier_start_idx, frontier_end_idx, size=int(0.75 * super_bs)) if super_bs > 1 else [np.random.randint(frontier_start_idx, frontier_end_idx)]
                            if reinforcement_end_idx > 0: chosen_batch_indices = np.concatenate([frontier_indices, np.random.randint(0, reinforcement_end_idx, size=int(0.25 * super_bs)) if super_bs > 1 else [np.random.randint(0, reinforcement_end_idx)]])
                            else: chosen_batch_indices = np.random.randint(0, frontier_end_idx, size=super_bs)
                            np.random.shuffle(chosen_batch_indices); vecs, prompts, gts = [], [], [];
                            for i in chosen_batch_indices: e, p, img = warmup_batches_np[i % WARMUP_BATCHES_COUNT]; vecs.append(e); prompts.append(p); gts.append(img)
                            super_vec, super_prompts, super_gt_images = np.concatenate(vecs), np.concatenate(prompts), np.concatenate(gts)
                        else:
                            try: super_vec, super_prompts, super_gt_images = next(main_ds_iterator)
                            except StopIteration: break
                    
                    with self.interactive_state.lock: is_gt_blast_active = self.interactive_state.gt_blast_active
                    if is_gt_blast_active:
                        current_main_lambdas = {'gt_patch': 100.0, 'echo': 0.01, 'perceptual': 0.01, 'diversity': 0.0}
                        current_sub_lambdas = self.sub_loss_lambda_controller(self.last_metrics, global_step)
                        current_status += " | GT BLAST ACTIVE! 💥"
                    else:
                        current_sub_lambdas = self.sub_loss_lambda_controller(self.last_metrics, global_step)
                        current_main_lambdas = self.main_loss_lambda_controller(self.last_metrics, global_step)
                    self.intervention_status = current_status.strip(" | ")

                    key, step_key = jax.random.split(key)
                    self.current_lambdas = {**current_main_lambdas, **current_sub_lambdas}
                    sub_lambda_keys = ['l1','ssim','edge','moment','color_cov','autocorr','fft']; main_lambda_keys = ['gt_patch', 'echo', 'perceptual', 'diversity']
                    sub_lambdas_for_jit = tuple(current_sub_lambdas.get(k,0.0) for k in sub_lambda_keys); main_lambdas_for_jit = tuple(current_main_lambdas.get(k,0.0) for k in main_lambda_keys)
                    if self.args.use_q_controller: state = state.replace(q_controller_state=state.q_controller_state.replace(step_count=jnp.array(global_step,dtype=jnp.int32)))
                    state, metrics, generated_image_batch = self.jitted_train_step(state=state, global_step=jnp.array(global_step),super_command_vec=super_vec, super_gt_images=super_gt_images,main_lambdas_tuple=main_lambdas_for_jit, sub_lambdas_tuple=sub_lambdas_for_jit,key=step_key)
                    self.weight_averager.update(state.params)
                    jax.device_get(metrics); time_now=time.time()
                    self.steps_per_sec = super_bs/(time_now-last_time+1e-6); last_time=time_now
                    global_step += 1; epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
                    self.progress.update(main_task, completed=global_step, epoch=epoch+1)
                    metrics_np = jax.tree_util.tree_map(np.asarray, metrics)

                    if self.adr_active: self.adr_loss_history.append(metrics_np['loss/total'])
                    
                    if time_now - last_ui_update_time > 0.2:
                        last_ui_update_time = time_now
                        with self.ui_lock:
                            self.last_metrics = {k:v.item() for k,v in metrics_np.items()}
                            if 'loss/total' in self.last_metrics and np.isfinite(self.last_metrics['loss/total']): self.loss_hist.append(self.last_metrics['loss/total'])
                            self.current_prompt = super_prompts[0].decode('utf-8')
                            self.current_step_image_np = ((np.array(generated_image_batch[0])*0.5+0.5)*255).clip(0,255).astype(np.uint8)
                        preview_idx += self.interactive_state.get_and_reset_preview_change(); preview_idx %= len(preview_data_buffer)
                        if active_preview_future is None or active_preview_future.done():
                            if active_preview_future: active_preview_future.result()
                            if (averaged_params := self.weight_averager.get_averaged_params()) is not None:
                                avg_params_device = jax.device_put(averaged_params)
                                student_encoder_ref = TextToPathParamsProjector(self.args.latent_grid_size,self.dtype)
                                active_preview_future = async_pool.submit(self._update_live_preview_task, avg_params_device, preview_data_buffer[preview_idx], student_encoder_ref)
                        live.update(self._generate_layout(), refresh=True)

                    # --- NEW: Validation and Best Model Saving Logic ---
                    if time_now - last_val_time > self.args.save_every:
                        last_val_time = time_now
                        if (avg_params := self.weight_averager.get_averaged_params()) is not None:
                            avg_params_device = jax.device_put(avg_params)
                            self.current_val_score = self._run_validation(avg_params_device, validation_set)
                            if self.current_val_score > self.best_val_score:
                                self.best_val_score = self.current_val_score
                                self.console.print(f"\n--- ⭐ New Best Validation Score: {self.best_val_score:.4f}! Saving model... ---")
                                best_model_data = {'student_encoder_params': unfreeze(jax.device_get(avg_params_device)), 'fixed_physics_params': self.fixed_physics_params}
                                with open(best_ckpt_path, 'wb') as f: pickle.dump(best_model_data, f)
                            else:
                                self.console.print(f"\n--- Validation score: {self.current_val_score:.4f} (Best: {self.best_val_score:.4f}) ---")
                        # Also save a regular checkpoint
                        self.console.print(f"--- 💾 Saving regular checkpoint at step {global_step}... ---")
                        data_to_save = {'params': jax.device_get(state.params), 'opt_state': jax.device_get(state.opt_state), 'q_controller_state': jax.device_get(state.q_controller_state), 'patch_buffer_state': jax.device_get(state.patch_buffer_state), 'best_val_score': self.best_val_score, 'step': global_step}
                        with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
        finally:
            live.stop(); self.interactive_state.set_shutdown(); key_listener_thread.join(timeout=1)
            self.console.print("\n--- Training finished. ---")
            if 'state' in locals():
                if not self.interactive_state.shutdown_event.is_set():
                    self.console.print("\n--- Saving final model... ---")
                    final_averaged_params = self.weight_averager.get_averaged_params()
                    final_path_avg = Path(f"{self.args.basename}_{self.args.d_model}d_distilled_t2i_final.pkl")
                    with open(final_path_avg,'wb') as f: pickle.dump({'student_encoder_params':unfreeze(jax.device_get(final_averaged_params)),'fixed_physics_params':self.fixed_physics_params},f)
                    self.console.print(f"✅ Final averaged weights model saved to [green]{final_path_avg}[/green]")
                    self.console.print(f"🏆 Best validation model saved to [yellow]{best_ckpt_path}[/yellow]")
                else:
                    self.console.print(f"\n--- 💾 Saving final checkpoint due to interruption... ---")
                    data={'params':jax.device_get(state.params),'opt_state':jax.device_get(state.opt_state),'q_controller_state':jax.device_get(state.q_controller_state), 'patch_buffer_state': jax.device_get(state.patch_buffer_state), 'best_val_score':self.best_val_score, 'step':global_step}
                    with open(ckpt_path,'wb') as f: pickle.dump(data,f)
                    self.console.print(f"✅ Checkpoint saved to [cyan]{ckpt_path}[/cyan].")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Hyper-Optimized Echo+GT Distillation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--basename',type=str,required=True,help="Basename for model files (must match Phase 1).")
    parent_parser.add_argument('--d-model',type=int,default=64,help="Model dimension of Phase 1 AE.")
    parent_parser.add_argument('--latent-grid-size',type=int,default=64,help="Latent grid size of Phase 1 AE.")
    parent_parser.add_argument('--image-size',type=int,default=512,help="Image resolution for training data.")
    p_prep = subparsers.add_parser("prepare-distill-data", help="Create aligned TFRecords from a folder of image/.txt pairs.")
    p_prep.add_argument('--source-dir', type=str, required=True, help="Directory with ALIGNED images and .txt files.")
    p_prep.add_argument('--target-dir', type=str, required=True, help="Directory to save the final TFRecord files.")
    p_prep.add_argument('--text-encoder-id', type=str, default="google/siglip-base-patch16-224", help="Hugging Face ID of the text encoder.")
    p_train = subparsers.add_parser("train", help="Distill knowledge using Echo and a super-charged Ground Truth Loss.", parents=[parent_parser])
    p_train.add_argument('--data-dir',type=str,required=True,help="Directory containing the ALIGNED TFRecord files.")
    p_train.add_argument('--epochs',type=int,default=100)
    p_train.add_argument('--batch-size',type=int,default=1,help="Size of mini-batches processed on GPU at once.")
    p_train.add_argument('--rebatch-size',type=int,default=1,help="Effective batch size is batch-size * rebatch-size.")
    p_train.add_argument('--lr',type=float,default=2e-4,help="Target learning rate for the Autonomous Scheduler.")
    p_train.add_argument('--seed',type=int,default=42)
    p_train.add_argument('--use-bfloat16',action='store_true', help="Use BFloat16 for training to save memory and increase speed.")
    p_train.add_argument('--use-q-controller',action='store_true',help="Enable Autonomous LR Scheduler.")
    p_train.add_argument('--save-every',type=int,default=300,help="Run validation and save checkpoint every N seconds.")
    args = parser.parse_args()
    if args.command == "prepare-distill-data":
        prepare_distill_data(args.source_dir, args.target_dir, args.text_encoder_id)
    elif args.command == "train":
        p1_glob = list(Path('.').glob(f"{args.basename}_{args.d_model}d_*_best.pkl"))
        if not p1_glob:
            print(f"[bold red]FATAL: Phase 1 model not found![/bold red] Searched for [cyan]'{args.basename}_{args.d_model}d_*_best.pkl'[/cyan]."), sys.exit(1)
        print(f"--- Loading Phase 1 weights from: [cyan]{p1_glob[0]}[/cyan] ---")
        with open(p1_glob[0], 'rb') as f: p1_data = pickle.load(f)
        fixed_physics_params = freeze(p1_data['ema_params'])
        trainer = DistillationTrainer(args, fixed_physics_params)
        trainer.train()
    else:
        print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
