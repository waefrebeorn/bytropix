
import os
# --- Environment Setup for JAX/TensorFlow ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
# --- JAX Configuration ---
from pathlib import Path
import platform

try:
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    cache_dir = script_dir / ".jax_cache"
    cache_dir.mkdir(exist_ok=True) # Ensure the directory exists
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at: {cache_dir} ---")
except NameError:
    # This can happen in interactive environments (like Jupyter) where __file__ is not defined.
    # We can fall back to a default location in the user's home directory.
    cache_dir = Path.home() / ".jax_cache_global"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at (fallback global): {cache_dir} ---")
# Conditional imports for keyboard listening
if platform.system() == "Windows":
    import msvcrt
else:
    import tty, termios, select
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, common_utils
from flax.jax_utils import replicate, unreplicate
import optax
import numpy as np
import pickle
import time
from typing import Any, Sequence, Dict, NamedTuple, Optional
import sys
import struct
import argparse
import signal

import threading
from functools import partial

from collections import deque
from PIL import Image
import jax.scipy.ndimage
import imageio
from flax.traverse_util import path_aware_map

CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_threefry_partitionable', True)
# --- Dependency Checks and Imports ---
try:
    import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
    from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
    from rich.text import Text
    import pynvml; pynvml.nvmlInit()
    from tqdm import tqdm
    import chex
except ImportError:
    print("[FATAL] A required dependency is missing (tensorflow, rich, pynvml, tqdm, chex). Please install them.")
    sys.exit(1)

# --- Generative Command & GUI Preview Dependencies ---
try:
    import clip
    import torch
    _clip_device = "cuda" if "cuda" in str(jax.devices()[0]).lower() else "cpu"
except ImportError:
    print("[Warning] `clip-by-openai` or `torch` not found. Generative commands will not be available.")
    clip, torch = None, None
try:
    from rich_pixels import Pixels
except ImportError:
    print("[Warning] `rich-pixels` not found. Visual preview in GUI will be disabled. Run: pip install rich-pixels")
    Pixels = None

# --- PHASE 2: Video Dependencies ---
try:
    import cv2
except ImportError:
    print("[Warning] `opencv-python` not found. Video processing commands will not be available. Run: pip install opencv-python")
    cv2 = None


# =================================================================================================
# 1. ADVANCED TRAINING TOOLKIT
# =================================================================================================
# --- NEW: Interactive Control System ---

class InteractivityState:
    """A thread-safe class to hold shared state for interactive controls."""
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_index_change = 0  # -1 for prev, 1 for next, 0 for no change
        self.sentinel_dampening_log_factor = -1.0  # Represents 0.1, log10 scale
        self.shutdown_event = threading.Event()

    def get_and_reset_preview_change(self):
        with self.lock:
            change = self.preview_index_change
            self.preview_index_change = 0
            return change

    def update_sentinel_factor(self, direction):
        with self.lock:
            # Update on a log scale: -3=0.001, -2=0.01, -1=0.1, 0=1.0
            self.sentinel_dampening_log_factor = np.clip(self.sentinel_dampening_log_factor + direction * 0.5, -3.0, 0.0)

    def get_sentinel_factor(self):
        with self.lock:
            return 10**self.sentinel_dampening_log_factor

    def set_shutdown(self):
        self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    """A cross-platform, non-blocking key listener that runs in a separate thread."""
    if platform.system() == "Windows":
        while not shared_state.shutdown_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\xe0': # Arrow key prefix
                    arrow = msvcrt.getch()
                    if arrow == b'K': # Left
                        with shared_state.lock: shared_state.preview_index_change = -1
                    elif arrow == b'M': # Right
                        with shared_state.lock: shared_state.preview_index_change = 1
                    elif arrow == b'H': # Up
                        shared_state.update_sentinel_factor(1)
                    elif arrow == b'P': # Down
                        shared_state.update_sentinel_factor(-1)
            time.sleep(0.05)
    else: # Linux/macOS
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not shared_state.shutdown_event.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char == '\x1b': # ESC sequence
                        next_chars = sys.stdin.read(2)
                        if next_chars == '[A': # Up
                            shared_state.update_sentinel_factor(1)
                        elif next_chars == '[B': # Down
                            shared_state.update_sentinel_factor(-1)
                        elif next_chars == '[C': # Right
                            with shared_state.lock: shared_state.preview_index_change = 1
                        elif next_chars == '[D': # Left
                            with shared_state.lock: shared_state.preview_index_change = -1
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def get_sentinel_lever_ascii(log_factor: float):
    """Generates an ASCII art lever for the UI."""
    levels = np.linspace(-3.0, 0.0, 7) # 7 levels from 0.001 to 1.0
    idx = np.digitize(log_factor, levels, right=True)
    lever = ["│         │"] * 7
    lever[6-idx] = "│█████████│"
    labels = ["1.0", " ", "0.1", " ", "0.01", " ", "0.001"]
    lines = [f" {labels[i]:<6} {lever[i]}" for i in range(7)]
    return "\n".join(lines)

class CustomTrainState(train_state.TrainState):
    """A custom train state that allows passing extra arguments to the optimizer.
    
    The default `TrainState.apply_gradients` passes all `**kwargs` to both the
    optimizer's `update` method and its own `replace` method. This causes an
    error if an argument is meant only for the optimizer. This custom state
    filters out arguments that are not part of the state's fields before
    calling `replace`.
    """
    def apply_gradients(self, *, grads: Any, **kwargs) -> "CustomTrainState":
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, **kwargs
        )
        new_params = optax.apply_updates(self.params, updates)

        # Filter kwargs to only include actual fields of the TrainState
        # before calling replace. This is the crucial fix.
        known_keys = self.__dataclass_fields__.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_keys}
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **filtered_kwargs,
        )
        
class SentinelState(NamedTuple):
    sign_history: chex.ArrayTree; dampened_count: Optional[jnp.ndarray] = None; dampened_pct: Optional[jnp.ndarray] = None

def sentinel(history_len: int = 5, oscillation_threshold: int = 3) -> optax.GradientTransformation:
    """
    An Optax component that dampens oscillating gradients.
    It is designed to work inside an `optax.chain` where extra arguments are passed
    to the top-level `update` call.
    """
    def init_fn(params):
        sign_history = jax.tree_util.tree_map(lambda t: jnp.zeros((history_len,) + t.shape, dtype=jnp.int8), params)
        return SentinelState(sign_history=sign_history, dampened_count=jnp.array(0), dampened_pct=jnp.array(0.0))

    def update_fn(updates, state, params=None, **kwargs):
        # Safely get 'dampening_factor' from kwargs. Default to 1.0 (no effect) if not present.
        # This makes the component robust inside a chain.
        dampening_factor = kwargs.get('dampening_factor', 1.0)

        new_sign_history = jax.tree_util.tree_map(lambda old_hist, new_sign: jnp.roll(old_hist, shift=-1, axis=0).at[history_len-1].set(new_sign.astype(jnp.int8)), state.sign_history, jax.tree_util.tree_map(jnp.sign, updates))
        is_oscillating = jax.tree_util.tree_map(lambda hist: jnp.sum(jnp.abs(jnp.diff(hist, axis=0)), axis=0) >= oscillation_threshold, new_sign_history)
        dampening_mask = jax.tree_util.tree_map(lambda is_osc: jnp.where(is_osc, dampening_factor, 1.0), is_oscillating)
        dampened_updates = jax.tree_util.tree_map(lambda u, m: u * m, updates, dampening_mask)
        num_oscillating = jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x), is_oscillating, 0)
        total_params = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, params, 0)
        new_state = SentinelState(sign_history=new_sign_history, dampened_count=num_oscillating, dampened_pct=(num_oscillating / (total_params + 1e-8)))
        return dampened_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)

Q_CONTROLLER_CONFIG_NORMAL = {
    "q_table_size": 100,
    "num_lr_actions": 5,
    "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1],
    "learning_rate_q": 0.1,
    "discount_factor_q": 0.9,
    "lr_min": 1e-4, "lr_max": 8.5e-4,
    "metric_history_len": 5000,
    "loss_min": 0.05, "loss_max": 1.5,
    "exploration_rate_q": 0.3,
    "min_exploration_rate": 0.05,
    "exploration_decay": 0.9995,
    "trend_window": 420,
    "improve_threshold": 1e-5,
    "regress_threshold": 1e-6,
    "regress_penalty": 10.0,
    "stagnation_penalty": -2.0,
    # --- NEW WARMUP PARAMETERS ---
    "warmup_steps": 420,
    "warmup_lr_start": 1e-6
}


Q_CONTROLLER_CONFIG_FINETUNE = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.8, 0.95, 1.0, 1.05, 1.2], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-4, "lr_max": 9.5e-4, "metric_history_len": 5000, "loss_min": .001, "loss_max": 0.1, "exploration_rate_q": 0.15, "min_exploration_rate": 0.02, "exploration_decay": 0.9998, "target_pixel_loss": 0.01}
Q_AUG_CONFIG = {
    "q_table_size": 100,
    "num_lr_actions": 4,
    "lr_change_factors": [], # DUMMY KEY TO SATISFY PARENT CLASS INIT
    "learning_rate_q": 0.05,
    "discount_factor_q": 0.9,
    "lr_min": 0, "lr_max": 0,
    "metric_history_len": 5000,
    "loss_min": 0.01,
    "loss_max": 0.4,
    "exploration_rate_q": 0.5,
    "min_exploration_rate": 0.1,
    "exploration_decay": 0.9996,
    "trend_window": 420,
    "improve_threshold": 1e-5,
    "regress_threshold": 1e-6,
    "regress_penalty": 5.0,
    "stagnation_penalty": -1.0,
    "warmup_steps": 420,
    "warmup_lr_start": 1e-6
}
class JaxHakmemQController:
    """A Q-Learning agent to dynamically control the learning rate with a warmup phase."""
    def __init__(self,initial_lr:float,config:Dict[str,Any]):
        self.config=config;
        self.initial_lr = initial_lr # Store the target LR
        self.current_lr=initial_lr;
        self.q_table_size=int(self.config["q_table_size"]);
        self.num_actions=int(self.config["num_lr_actions"]);
        self.lr_change_factors=self.config["lr_change_factors"];
        self.q_table=np.zeros((self.q_table_size,self.num_actions),dtype=np.float32);
        self.learning_rate_q=float(self.config["learning_rate_q"]);
        self.discount_factor_q=float(self.config["discount_factor_q"]);
        self.lr_min=float(self.config["lr_min"]);
        self.lr_max=float(self.config["lr_max"]);
        self.loss_history=deque(maxlen=int(self.config["metric_history_len"]));
        self.loss_min=float(self.config["loss_min"]);
        self.loss_max=float(self.config["loss_max"]);
        self.last_action_idx:Optional[int]=None;
        self.last_state_idx:Optional[int]=None;
        self.initial_exploration_rate = float(self.config["exploration_rate_q"]);
        self.exploration_rate_q = self.initial_exploration_rate;
        self.min_exploration_rate = float(self.config["min_exploration_rate"]);
        self.exploration_decay = float(self.config["exploration_decay"]);
        self.status: str = "STARTING";
        self.last_reward: float = 0.0
        self.trend_window = int(config["trend_window"]);
        self.pixel_loss_trend_history = deque(maxlen=self.trend_window);
        self.improve_threshold = float(config["improve_threshold"]);
        self.regress_threshold = float(config["regress_threshold"]);
        self.regress_penalty = float(config["regress_penalty"]);
        self.stagnation_penalty = float(config["stagnation_penalty"]);
        self.last_slope: float = 0.0

        # --- NEW WARMUP STATE ---
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.warmup_lr_start = float(config.get("warmup_lr_start", 1e-7))
        self._step_count = 0

        print(f"--- Q-Controller initialized. LR Warmup: {self.warmup_steps} steps. Trend Window: {self.trend_window} steps ---")

    def _discretize_value(self,value:float) -> int:
        if not np.isfinite(value): return self.q_table_size // 2
        if value<=self.loss_min: return 0
        if value>=self.loss_max: return self.q_table_size-1
        bin_size=(self.loss_max-self.loss_min)/self.q_table_size; return min(int((value-self.loss_min)/bin_size),self.q_table_size-1)

    def _get_current_state_idx(self) -> int:
        if not self.loss_history: return self.q_table_size//2
        avg_loss=np.mean(list(self.loss_history)[-5:]); return self._discretize_value(avg_loss)

    def choose_action(self) -> float:
        self._step_count += 1

        # --- WARMUP LOGIC ---
        if self._step_count <= self.warmup_steps:
            # Linearly interpolate LR from start_lr to initial_lr
            alpha = self._step_count / self.warmup_steps
            self.current_lr = self.warmup_lr_start * (1 - alpha) + self.initial_lr * alpha
            self.status = f"WARMUP (LR) {self._step_count}/{self.warmup_steps}"
            return self.current_lr

        # --- STANDARD Q-LEARNING LOGIC (after warmup) ---
        self.last_state_idx = self._get_current_state_idx()
        if np.random.rand() < self.exploration_rate_q:
            self.last_action_idx = np.random.randint(0, self.num_actions)
        else:
            self.last_action_idx = np.argmax(self.q_table[self.last_state_idx]).item()

        change_factor = self.lr_change_factors[self.last_action_idx]
        self.current_lr = np.clip(self.current_lr * change_factor, self.lr_min, self.lr_max)
        return self.current_lr

    def update_q_value(self, total_loss:float):
        self.loss_history.append(total_loss)

        # Do not update Q-table during LR warmup, but DO update trend history
        if self._step_count <= self.warmup_steps:
            self.pixel_loss_trend_history.append(total_loss)
            return

        if self.last_state_idx is None or self.last_action_idx is None: return

        reward = self._calculate_reward(total_loss); self.last_reward = reward
        current_q = self.q_table[self.last_state_idx, self.last_action_idx]
        next_state_idx = self._get_current_state_idx()
        max_next_q = np.max(self.q_table[next_state_idx])
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor_q * max_next_q - current_q)
        self.q_table[self.last_state_idx, self.last_action_idx] = new_q
        self.exploration_rate_q = max(self.min_exploration_rate, self.exploration_rate_q * self.exploration_decay)

    def _calculate_reward(self, current_loss):
        self.pixel_loss_trend_history.append(current_loss)
        if len(self.pixel_loss_trend_history) < self.trend_window:
            self.status = f"WARMING UP (TREND) {len(self.pixel_loss_trend_history)}/{self.trend_window}"
            return 0.0

        loss_window = np.array(self.pixel_loss_trend_history); slope = np.polyfit(np.arange(self.trend_window), loss_window, 1)[0]; self.last_slope = slope
        if slope < -self.improve_threshold: self.status = f"IMPROVING (S={slope:.2e})"; reward = abs(slope) * 1000
        elif slope > self.regress_threshold: self.status = f"REGRESSING (S={slope:.2e})"; reward = -abs(slope) * 1000 - self.regress_penalty
        else: self.status = f"STAGNATED (S={slope:.2e})"; reward = self.stagnation_penalty
        return reward

    def state_dict(self)->Dict[str,Any]:
        # Also save the step count to resume warmup correctly
        return {"current_lr":self.current_lr, "q_table":self.q_table.tolist(), "loss_history":list(self.loss_history),
                "exploration_rate_q":self.exploration_rate_q, "pixel_loss_trend_history": list(self.pixel_loss_trend_history),
                "_step_count": self._step_count}

    def load_state_dict(self,state_dict:Dict[str,Any]):
        self.current_lr=state_dict.get("current_lr",self.current_lr)
        self.q_table=np.array(state_dict.get("q_table",self.q_table.tolist()),dtype=np.float32)
        self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.loss_history.maxlen)
        self.exploration_rate_q=state_dict.get("exploration_rate_q", self.initial_exploration_rate)
        self.pixel_loss_trend_history=deque(state_dict.get("pixel_loss_trend_history",[]),maxlen=self.trend_window)
        self._step_count=state_dict.get("_step_count", 0)



class QAugmentationController(JaxHakmemQController):
    """A specialized Q-Controller to dynamically select data augmentation profiles."""
    def __init__(self, config:Dict[str,Any]):
        super().__init__(initial_lr=1.0, config=config)
        print(f"--- Q-Augmentation Controller initialized. ---")

        self.action_profiles = [
            {"name": "Off",    "flip_prob": 0.0, "bright_delta": 0.0,  "contrast_factor": 0.0,  "sat_factor": 0.0,  "hue_delta": 0.0, "noise_stddev": 0.0},
            {"name": "Low",    "flip_prob": 0.5, "bright_delta": 0.05, "contrast_factor": 0.1,  "sat_factor": 0.1,  "hue_delta": 0.02,"noise_stddev": 0.01},
            {"name": "Medium", "flip_prob": 0.5, "bright_delta": 0.1,  "contrast_factor": 0.15, "sat_factor": 0.2,  "hue_delta": 0.08,"noise_stddev": 0.02},
            {"name": "High",   "flip_prob": 0.5, "bright_delta": 0.15, "contrast_factor": 0.25, "sat_factor": 0.3,  "hue_delta": 0.1, "noise_stddev": 0.025},
        ]

        self.num_actions = len(self.action_profiles)
        self.q_table = np.zeros((self.q_table_size, self.num_actions), dtype=np.float32)
        self.current_action_profile = self.action_profiles[0]

    def choose_action(self) -> Dict[str, Any]:
        self.last_state_idx = self._get_current_state_idx()

        if np.random.rand() < self.exploration_rate_q:
            self.last_action_idx = np.random.randint(0, self.num_actions)
        else:
            self.last_action_idx = np.argmax(self.q_table[self.last_state_idx]).item()

        self.current_action_profile = self.action_profiles[self.last_action_idx]
        return self.current_action_profile

# =================================================================================================
# 2. MATHEMATICAL & MODEL FOUNDATIONS
# =================================================================================================

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        # --- [SCIENTIFIC PIVOT] ---
        # This equation is now aligned with Eq. (2) from the provided research paper.
        # It describes the complex transmittance for a co-polarized channel by encircling
        # a singularity on the Poincare sphere, which is a topologically protected mechanism.
        # Original (slightly incorrect): real_part = jnp.cos(delta/2) * jnp.cos(chi)
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part
        # --- [END SCIENTIFIC PIVOT] ---

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        if self.latent_grid_size > self.input_image_size:
            raise ValueError(f"latent_grid_size must be smaller than or equal to input_image_size. Got {self.latent_grid_size}.")

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
            x = jax.image.resize(x,
                                 (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]),
                                 'bilinear')

        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x)
        x = nn.gelu(x)
        path_params = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1)

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = path_params_grid.shape
        L = H * W
        path_params = path_params_grid.reshape(B, L, C)
        
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
        chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path)
        
        # --- Enriched 4D measurement to break the bottleneck ---
        path_real_mean = jnp.mean(t_co_steps.real, axis=-1)
        path_real_std = jnp.std(t_co_steps.real, axis=-1)
        path_imag_mean = jnp.mean(t_co_steps.imag, axis=-1)
        path_imag_std = jnp.std(t_co_steps.imag, axis=-1)
        complex_measurement = jnp.stack([path_real_mean, path_real_std, path_imag_mean, path_imag_std], axis=-1)

        feature_vectors = nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(complex_measurement)
        return feature_vectors.reshape(B, H, W, self.d_model)

class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)

class CoordinateDecoder(nn.Module):
    d_model: int; num_freqs: int = 10; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = feature_grid.shape
        pos_encoder = PositionalEncoding(self.num_freqs, dtype=self.dtype)
        encoded_coords = pos_encoder(coords)

        # Create a feature pyramid
        num_levels = 3 # Base, 2x down, 4x down
        pyramid = [feature_grid]
        for i in range(1, num_levels):
            downsampled = jax.image.resize(feature_grid, (B, H//(2**i), W//(2**i), self.d_model), 'bilinear')
            pyramid.append(downsampled)

        # Sample from all pyramid levels
        all_sampled_features = []
        for i, level_grid in enumerate(pyramid):
            level_shape = jnp.array(level_grid.shape[1:3], dtype=self.dtype)
            coords_rescaled = (coords + 1) / 2 * (level_shape - 1)

            def sample_one_image_level(single_level_grid):
                grid_chw = single_level_grid.transpose(2, 0, 1)
                sampled_channels = jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw)
                return sampled_channels.T

            sampled_features = jax.vmap(sample_one_image_level)(level_grid)
            all_sampled_features.append(sampled_features)

        # Concatenate features from all scales
        concatenated_features = jnp.concatenate(all_sampled_features, axis=-1)

        encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)

        mlp_input = jnp.concatenate([encoded_coords_tiled, concatenated_features], axis=-1)
        h = mlp_input

        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = nn.gelu(h)
            if i == self.mlp_depth // 2:
                h = jnp.concatenate([h, mlp_input], axis=-1)

        output_pixels = nn.Dense(3, name="mlp_out", dtype=self.dtype)(h)
        return nn.tanh(output_pixels)
        
class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(d_model=self.d_model, name="coord_decoder", dtype=self.dtype)
    def __call__(self, images, coords):
        path_params = self.modulator(images); feature_grid = self.observer(path_params); return self.coord_decoder(feature_grid, coords), path_params
    def decode(self, path_params, coords):
        feature_grid = self.observer(path_params); return self.coord_decoder(feature_grid, coords)

# --- [ARCHITECTURAL REFACTOR] ---
# Renamed from LatentCorrectionNetwork to DynamicsPredictor
# Its role is now to predict the change (Δp) in latent parameters based on flow.
class DynamicsPredictor(nn.Module):
    """Predicts the change in latent parameters (Δp) based on motion."""
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, p_warped: jnp.ndarray, flow_latent_res: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([p_warped, flow_latent_res], axis=-1); features = 32
        x = nn.Conv(features, (3, 3), padding='SAME', name="dyn_conv_1", dtype=self.dtype)(x); x = nn.gelu(x)
        x = nn.Conv(features * 2, (3, 3), padding='SAME', name="dyn_conv_2", dtype=self.dtype)(x); x = nn.gelu(x)
        x = nn.Conv(features * 2, (3, 3), padding='SAME', name="dyn_conv_3", dtype=self.dtype)(x); x = nn.gelu(x)
        # Predict the change in (delta, chi, radius). Initialize to predict zero change.
        delta_p = nn.Conv(3, (1, 1), name="delta_p_out", dtype=self.dtype,
                          kernel_init=nn.initializers.zeros)(x)
        return nn.tanh(delta_p) * 0.25 # Constrain the change to a reasonable range
# --- [END ARCHITECTURAL REFACTOR] ---



# =================================================================================================
# 3. DATA HANDLING
# =================================================================================================

# =================================================================================================
# 3. DATA HANDLING (With Scene Cut Detection)
# =================================================================================================

def prepare_data(image_dir: str):
    base_path = Path(image_dir); record_file = base_path/"data_512x512.tfrecord"; info_file=base_path/"dataset_info.pkl"
    if record_file.exists(): print(f"✅ Data files found in {image_dir}. Skipping preparation."); return
    print(f"--- Preparing 512x512 data from {image_dir} ---")
    image_paths = sorted([p for p in base_path.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
    if not image_paths: print(f"[FATAL] No images found in {image_dir}."), sys.exit(1)
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for path in tqdm(image_paths, "Processing Images"):
            try:
                img = Image.open(path).convert("RGB").resize((512,512),Image.Resampling.LANCZOS)
                img_bytes = tf.io.encode_jpeg(np.array(img),quality=95).numpy()
                ex=tf.train.Example(features=tf.train.Features(feature={'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
                writer.write(ex.SerializeToString())
            except Exception as e: print(f"Skipping {path}: {e}")
    with open(info_file,'wb') as f: pickle.dump({'num_samples':len(image_paths)},f)
    print(f"✅ Data preparation complete.")

# --- [SOLUTION] create_video_dataset is now scene-aware ---
def create_video_dataset(data_dir: str, batch_size: int, clip_len: int, is_training: bool = True):
    """Creates a TF dataset that yields clips of frames, respecting scene cuts."""
    data_p = Path(data_dir)
    frames_p = data_p / "frames"
    info_file = data_p / "dataset_info.pkl"
    if not frames_p.is_dir() or not info_file.exists():
        raise FileNotFoundError(f"Prepared data not found at {data_p}. Run 'prepare-video-data' first.")

    with open(info_file, 'rb') as f:
        dataset_info = pickle.load(f)
    
    scene_starts = dataset_info.get('scene_starts')
    if scene_starts is None:
        raise ValueError("'scene_starts' not found in dataset_info.pkl. Please re-run 'prepare-video-data'.")

    # Create a list of all valid clips (as lists of file paths)
    all_clips = []
    file_paths = sorted([str(p) for p in frames_p.glob("*.jpg")])
    
    for i in range(len(scene_starts) - 1):
        start_frame_idx = scene_starts[i]
        # The end of the scene is the frame just before the start of the next scene
        end_frame_idx = scene_starts[i+1]
        scene_paths = file_paths[start_frame_idx:end_frame_idx]

        # Generate all possible overlapping clips within this single scene
        for j in range(len(scene_paths) - 1): # -1 to ensure at least 2 frames per clip
            # Take a full clip if possible, otherwise take a shorter clip until the end of the scene
            clip = scene_paths[j : j + clip_len]
            if len(clip) > 1: # Only add clips with at least one P-frame
                all_clips.append(clip)

    print(f"--- Created {len(all_clips)} scene-aware video clips. ---")

    def _parse_and_pad(paths):
        # Decode all images in the clip
        imgs = tf.map_fn(lambda p: (tf.cast(tf.io.decode_jpeg(tf.io.read_file(p), 3), tf.float32) / 127.5) - 1.0, 
                         paths, fn_output_signature=tf.TensorSpec(shape=[512, 512, 3], dtype=tf.float32))
        
        # Pad the clip to the maximum length `clip_len` if it's shorter
        num_frames = tf.shape(imgs)[0]
        padding_needed = clip_len - num_frames
        
        # Pad with zeros. We will also return the true length to create a mask in the training step.
        paddings = [[0, padding_needed], [0, 0], [0, 0], [0, 0]]
        imgs_padded = tf.pad(imgs, paddings)
        
        imgs_padded.set_shape([clip_len, 512, 512, 3])
        return imgs_padded, num_frames # Return the true length of the clip

    ds = tf.data.Dataset.from_generator(lambda: all_clips, output_signature=tf.TensorSpec(shape=(None,), dtype=tf.string))
    
    if is_training:
        ds = ds.shuffle(len(all_clips))

    ds = ds.map(_parse_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# --- [SOLUTION] prepare_video_data now performs scene cut detection ---
def prepare_video_data(video_path: str, data_dir: str):
    if cv2 is None: raise ImportError("OpenCV is required for video preparation.")
    video_p, data_p = Path(video_path), Path(data_dir); data_p.mkdir(exist_ok=True)
    frames_dir = data_p / "frames"
    if (data_p/"prep_complete.flag").exists(): print(f"✅ Video data already prepared in {data_dir}. Skipping."); return
    frames_dir.mkdir(exist_ok=True)
    
    print(f"--- Preparing video data from {video_path} into {data_dir} ---"); cap = cv2.VideoCapture(str(video_p)); frames = []
    print("Step 1/4: Reading and saving frames...")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count), desc="Reading Frames"):
        ret, frame = cap.read();
        if not ret: break
        frame_np = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512, 512), Image.Resampling.LANCZOS))
        frames.append(frame_np)
        Image.fromarray(frame_np).save(frames_dir / f"frame_{i:05d}.jpg", quality=95)
    cap.release()

    print("Step 2/4: Detecting scene cuts...")
    # A simple threshold-based scene detector using color histogram differences.
    # We use a lower resolution for speed.
    prvs_hist = cv2.calcHist([cv2.resize(frames[0], (128, 128))], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(prvs_hist, prvs_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    scene_starts = [0] # The first frame is always the start of a scene.
    threshold = 0.8 # Correlation threshold. Lower value = more sensitive to cuts.
    
    for i in tqdm(range(1, len(frames)), desc="Analyzing Scenes"):
        nxt_hist = cv2.calcHist([cv2.resize(frames[i], (128, 128))], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(nxt_hist, nxt_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        diff = cv2.compareHist(prvs_hist, nxt_hist, cv2.HISTCMP_CORREL)
        if diff < threshold:
            scene_starts.append(i)
        prvs_hist = nxt_hist
    
    # Add the total number of frames as the final "scene start" to mark the end of the last scene.
    scene_starts.append(len(frames))
    print(f"✅ Found {len(scene_starts) - 1} distinct scenes.")

    # Note: Optical flow is no longer pre-calculated as it depends on the clips generated by the dataset pipeline.
    # It will be calculated on-the-fly during training.
    print("Step 3/4: Skipping pre-calculation of optical flow.")

    print("Step 4/4: Saving dataset info...")
    with open(data_p/"dataset_info.pkl", 'wb') as f:
        pickle.dump({'num_frames': len(frames), 'scene_starts': scene_starts}, f)
    
    (data_p/"prep_complete.flag").touch(); print("✅ Video data preparation complete.")


def create_dataset(image_dir: str, is_training: bool = True):
    """Creates a dataset that yields raw uint8 image tensors from a TFRecord file."""
    record_file = Path(image_dir)/"data_512x512.tfrecord"
    if not record_file.exists(): raise FileNotFoundError(f"{record_file} not found. Run 'prepare-data' first.")

    def _parse(proto):
        f={'image':tf.io.FixedLenFeature([],tf.string)}; p=tf.io.parse_single_example(proto,f)
        return tf.io.decode_jpeg(p['image'], 3)

    ds = tf.data.TFRecordDataset(str(record_file)).map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(1024)
    return ds
def create_dataset_from_frames(data_dir: str, is_training: bool = True):
    """Creates a dataset that yields raw uint8 image tensors from a directory of JPG files."""
    base_p = Path(data_dir)
    frames_p = base_p / "frames"

    if base_p.name == "frames" and base_p.is_dir():
        frames_p = base_p
    elif not frames_p.is_dir():
        raise FileNotFoundError(f"Frames directory not found at {frames_p}. Run 'prepare-video-data' first.")

    def _parse(path):
        img_bytes = tf.io.read_file(path)
        return tf.io.decode_jpeg(img_bytes, 3)

    ds = tf.data.Dataset.list_files(str(frames_p / "*.jpg"), shuffle=is_training)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def apply_augmentations_tf(img_uint8: tf.Tensor, profile: Dict[str, any]) -> tf.Tensor:
    """
    Applies a full augmentation pipeline to a raw uint8 image tensor.
    """
    img = tf.cast(img_uint8, tf.float32) / 255.0
    if profile["flip_prob"] > 0 and tf.random.uniform(()) < profile["flip_prob"]:
        img = tf.image.flip_left_right(img)
    if profile["bright_delta"] > 0:
        img = tf.image.random_brightness(img, max_delta=profile["bright_delta"])
    if profile["contrast_factor"] > 0:
        img = tf.image.random_contrast(img, lower=1.0-profile["contrast_factor"], upper=1.0+profile["contrast_factor"])
    if profile["sat_factor"] > 0:
        img = tf.image.random_saturation(img, lower=1.0-profile["sat_factor"], upper=1.0+profile["sat_factor"])
    if profile["hue_delta"] > 0:
        img = tf.image.random_hue(img, max_delta=profile["hue_delta"])
    img = tf.clip_by_value(img, 0.0, 1.0)
    if profile["noise_stddev"] > 0:
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=profile["noise_stddev"], dtype=tf.float32)
        img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img * 2.0 - 1.0




# =================================================================================================
# 4. ADVANCED TRAINING FRAMEWORK
# =================================================================================================

class AdvancedTrainer:
    """Base class for training with advanced toolkit features."""
    def __init__(self, args, model):
        self.args = args; self.model = model; self.key = jax.random.PRNGKey(args.seed); self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.num_devices = jax.local_device_count()
        self.recon_loss_history = deque(maxlen=200)
        if self.args.use_q_controller:
            q_config = Q_CONTROLLER_CONFIG_FINETUNE if args.finetune else Q_CONTROLLER_CONFIG_NORMAL
            self.q_controller = JaxHakmemQController(initial_lr=self.args.lr, config=q_config)
        else: self.q_controller = None

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

    def _update_preview_panel(self, panel, original_img, recon_img):
        if Pixels is None:
            panel.renderable = Align.center(Text("Install `rich-pixels` for previews", style="yellow")); return panel
        term_width = 64; h, w, _ = original_img.shape; term_height = int(term_width * (h / w) * 0.5)
        original_pil = Image.fromarray(original_img).resize((term_width, term_height), Image.Resampling.LANCZOS)
        recon_pil = Image.fromarray(recon_img).resize((term_width, term_height), Image.Resampling.LANCZOS)
        original_pix = Pixels.from_image(original_pil); recon_pix = Pixels.from_image(recon_pil)
        preview_table = Table.grid(expand=True); preview_table.add_column(ratio=1); preview_table.add_column(ratio=1)
        preview_table.add_row(Text("Original", justify="center"), Text("Reconstruction", justify="center")); preview_table.add_row(original_pix, recon_pix)
        panel.renderable = preview_table; return panel

    def _save_checkpoint(self, p_state, epoch, ckpt_path):
        data = {'params': jax.device_get(unreplicate(p_state.params)), 'opt_state': jax.device_get(unreplicate(p_state.opt_state)), 'epoch': epoch}
        if self.q_controller: data['q_controller_state'] = self.q_controller.state_dict()
        with open(ckpt_path, 'wb') as f: pickle.dump(data, f)

    def train(self):
        raise NotImplementedError("Subclasses must implement the train method.")

class ImageTrainer(AdvancedTrainer):
    def __init__(self, args):
        # --- [SOLUTION] Use bfloat16 if specified ---
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        model = TopologicalCoordinateGenerator(
            d_model=args.d_model, 
            latent_grid_size=args.latent_grid_size, 
            input_image_size=args.image_size,
            dtype=self.dtype
        )
        super().__init__(args, model)
        self.sentinel_dampen_history = deque(maxlen=200)
        self.interactive_state = InteractivityState()

        if args.use_q_aug_controller:
            self.q_aug_controller = QAugmentationController(config=Q_AUG_CONFIG)
        else:
            self.q_aug_controller = QAugmentationController(config=Q_AUG_CONFIG)
            self.q_aug_controller.exploration_rate_q = 0.0
            self.q_aug_controller.q_table = np.full((Q_AUG_CONFIG["q_table_size"], len(self.q_aug_controller.action_profiles)), -1e9, dtype=np.float32)
            self.q_aug_controller.q_table[:, 0] = 0.0
            
        self.ui_lock = threading.Lock()
        self.last_loss_val = 0.0
        self.current_preview_np = None
        self.current_recon_np = None
        # --- NEW UI STATE ---
        self.spinner_chars = ["🧠", "⚡", "💾", "📈", "🧠", "⚡", "💽", "📉"]
        self.spinner_idx = 0
        self.param_count = 0
        self.steps_per_sec = 0.0

    def _save_checkpoint(self, p_state, epoch, ckpt_path):
        data = {'params': jax.device_get(unreplicate(p_state.params)), 'opt_state': jax.device_get(unreplicate(p_state.opt_state)), 'epoch': epoch}
        if self.q_controller: data['q_controller_state'] = self.q_controller.state_dict()
        if self.args.use_q_aug_controller: data['q_aug_controller_state'] = self.q_aug_controller.state_dict()
        with open(ckpt_path, 'wb') as f: pickle.dump(data, f)

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            # --- Main Layout Structure ---
            layout = Layout(name="root")
            layout.split(
                Layout(name="header", size=3),
                Layout(ratio=1, name="main"),
                Layout(self.progress, size=3)
            )
            # --- Responsive Main Panel ---
            layout["main"].split_row(
                Layout(name="left", minimum_size=55),
                Layout(name="right", ratio=1)
            )

            # --- 1. Header ---
            spinner = self.spinner_chars[self.spinner_idx]
            precision_str = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"{spinner} [bold]Topological Autoencoder Training[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            layout["header"].update(Panel(Align.center(header_text), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right"))

            # --- 2. Left Column (Information) ---
            
            # 2a. Core Stats Panel
            stats_tbl = Table.grid(expand=True, padding=(0, 1))
            stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right")
            loss_val = self.last_loss_val
            if loss_val < 0.05:   loss_emoji, loss_color = "👌", "bold green"
            elif loss_val < 0.15: loss_emoji, loss_color = "👍", "bold yellow"
            else:                 loss_emoji, loss_color = "😟", "bold red"
            stats_tbl.add_row("Image Loss (L1)", f"[{loss_color}]{loss_val:.5f}[/] {loss_emoji}")
            stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🏃💨")
            mem, util = self._get_gpu_stats()
            stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            # --- [SOLUTION] GUI FIX: Use a fixed height for the panel to prevent jumping ---
            stats_panel = Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue", height=6)
            
            # 2b. Q-Controllers Panel
            q_table = Table.grid(expand=True, padding=(0, 1))
            q_table.add_column("Ctrl", style="bold cyan", width=6)
            q_table.add_column("Metric", style="dim", width=10)
            q_table.add_column("Value", justify="left")
            
            num_q_controllers = 0
            if self.q_controller:
                num_q_controllers += 1
                status = self.q_controller.status
                if "IMPROVING" in status:  status_emoji, color = "😎", "bold green"
                elif "STAGNATED" in status: status_emoji, color = "🤔", "bold yellow"
                elif "REGRESSING" in status:status_emoji, color = "😠", "bold red"
                elif "WARMUP" in status:    status_emoji, color = "🐣", "bold blue"
                else:                       status_emoji, color = "🤖", "dim"
                q_table.add_row("🧠 LR", "Status", f"[{color}]{status}[/] {status_emoji}")
                q_table.add_row("", "Reward", f"{self.q_controller.last_reward:+.2f}")
                q_table.add_row("", "Slope", f"{self.q_controller.last_slope:.2e}")
            
            if self.args.use_q_aug_controller:
                if num_q_controllers > 0: q_table.add_row() # Add a separator line
                num_q_controllers += 1
                aug_name, aug_status = self.q_aug_status if hasattr(self, 'q_aug_status') else ("N/A", "INIT")
                if "IMPROVING" in aug_status:  status_emoji, color = "😎", "bold green"
                elif "STAGNATED" in aug_status: status_emoji, color = "🤔", "bold yellow"
                elif "REGRESSING" in aug_status:status_emoji, color = "😠", "bold red"
                else:                          status_emoji, color = "🤖", "dim"
                q_table.add_row("🎨 Aug", "Profile", f"[yellow]{aug_name}[/]")
                q_table.add_row("", "Status", f"[{color}]{aug_status}[/] {status_emoji}")
                q_table.add_row("", "Reward", f"{self.q_aug_controller.last_reward:+.2f}")
                q_table.add_row("", "Slope", f"{self.q_aug_controller.last_slope:.2e}")

            # --- [SOLUTION] GUI FIX: Calculate a fixed height based on content ---
            q_panel_height = 8 if (self.q_controller and self.args.use_q_aug_controller) else 5
            if num_q_controllers > 0:
                q_panel = Panel(q_table, title="[bold]🤖 Q-Controllers[/]", border_style="green", height=q_panel_height)
            else:
                q_panel = Panel(Align.center("[dim]Q-Controllers Disabled[/dim]"), title="[bold]🤖 Q-Controllers[/]", border_style="dim", height=q_panel_height)

            # 2c. Sentinel Panel
            if self.args.use_sentinel:
                sentinel_layout = Layout()
                log_factor = self.interactive_state.sentinel_dampening_log_factor
                if log_factor <= -2.5:    rocket = "🚀"
                elif log_factor <= -1.5:  rocket = "✈️"
                else:                     rocket = "🛸"
                lever_panel = Panel(get_sentinel_lever_ascii(log_factor), title=f"Dampen {rocket}", title_align="left")
                status_str = self.sentinel_status_str if hasattr(self, 'sentinel_status_str') else "Initializing..."
                # --- [SOLUTION] GUI FIX: Fix inner panel height ---
                status_panel = Panel(Align.center(Text(status_str)), title="Status 🚦", title_align="left", height=4)
                sentinel_layout.split_row(Layout(lever_panel), Layout(status_panel))
                # --- [SOLUTION] GUI FIX: Fix outer panel height ---
                sentinel_panel = Panel(sentinel_layout, title="[bold]🕹️ Sentinel Interactive[/]", border_style="yellow", height=11)
                left_panels = [stats_panel, q_panel, sentinel_panel]
            else:
                left_panels = [stats_panel, q_panel]

            layout["left"].update(Group(*left_panels))

            # --- 3. Right Column (Visuals) ---
            spark_w = 40
            recon_panel = Panel(Align.center(f"[cyan]{self._get_sparkline(self.recon_loss_history, spark_w)}[/]"), title=f"Reconstruction Loss (L1)", height=3, border_style="cyan")
            graph_panels = [recon_panel]
            if self.args.use_sentinel:
                sentinel_panel = Panel(Align.center(f"[magenta]{self._get_sparkline(self.sentinel_dampen_history, spark_w)}[/]"), title="Sentinel Dampening %", height=3, border_style="magenta")
                graph_panels.append(sentinel_panel)
            graphs_group = Panel(Group(*graph_panels), title="[bold]📉 Trends[/]")

            preview_content = "..."
            if self.current_preview_np is not None and self.current_recon_np is not None:
                if Pixels is None:
                    preview_content = Align.center(Text("Install `rich-pixels` for previews", style="yellow"))
                else:
                    term_width = 64; h, w, _ = self.current_preview_np.shape; term_height = int(term_width * (h / w) * 0.5)
                    original_pil = Image.fromarray(self.current_preview_np).resize((term_width, term_height), Image.Resampling.LANCZOS)
                    recon_pil = Image.fromarray(self.current_recon_np).resize((term_width, term_height), Image.Resampling.LANCZOS)
                    preview_table = Table.grid(expand=True); preview_table.add_column(ratio=1); preview_table.add_column(ratio=1)
                    preview_table.add_row(Text("Original 📸", justify="center"), Text("Reconstruction ✨", justify="center"))
                    preview_table.add_row(Pixels.from_image(original_pil), Pixels.from_image(recon_pil))
                    preview_content = preview_table
            preview_panel = Panel(preview_content, title="[bold]🖼️ Live Preview[/]", border_style="green", height=20)

            layout["right"].split(
                graphs_group,
                preview_panel,
                Layout(Align.center(Text("←/→: Change Preview | ↑/↓: Adjust Sentinel", style="dim")), size=1)
            )
            return layout

    def train(self):
        train_key = jax.random.PRNGKey(self.args.seed)
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True)
        key_listener_thread.start()
        ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        components = [optax.clip_by_global_norm(1.0)]
        if self.args.use_sentinel: components.append(sentinel())
        base_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr)
        optimizer = optax.chain(*components, base_optimizer)
        
        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path,'rb') as f: data=pickle.load(f)
            params = data['params']
            state_template = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
            if 'opt_state' in data and jax.tree_util.tree_structure(state_template.opt_state) == jax.tree_util.tree_structure(data['opt_state']):
                state = state_template.replace(opt_state=data['opt_state']); print("--- Full optimizer state loaded successfully. ---")
            else:
                state = state_template; print("[bold yellow]Warning: Optimizer state mismatch. Re-initializing optimizer.[/bold yellow]")
            start_epoch = data.get('epoch', 0) + 1
            if self.q_controller and 'q_controller_state' in data: self.q_controller.load_state_dict(data['q_controller_state']); print("--- Q-Controller state loaded. ---")
            if self.args.use_q_aug_controller and 'q_aug_controller_state' in data: self.q_aug_controller.load_state_dict(data['q_aug_controller_state']); print("--- Q-Augmentation state loaded. ---")
        else:
            print("--- Initializing new model ---")
            with jax.default_device(CPU_DEVICE):
                dummy_images = jnp.zeros((1, 512, 512, 3), dtype=self.dtype); 
                dummy_coords = jnp.zeros((1024, 2), dtype=self.dtype)
                params = self.model.init(jax.random.PRNGKey(0), dummy_images, dummy_coords)['params']
            state = CustomTrainState.create(apply_fn=self.model.apply,params=params,tx=optimizer); start_epoch=0
        
        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, unreplicate(state.params), 0)
        
        p_state = replicate(state)
        unbatched_pipelines = []
        total_samples = 0
        global_batch_size = self.args.batch_size * self.num_devices
        if self.args.tfrecord_dir:
            pipeline_tf = create_dataset(self.args.tfrecord_dir)
            unbatched_pipelines.append(pipeline_tf)
            with open(Path(self.args.tfrecord_dir)/"dataset_info.pkl", 'rb') as f: total_samples += pickle.load(f)['num_samples']
        if self.args.video_frames_dir:
            pipeline_frames = create_dataset_from_frames(self.args.video_frames_dir)
            unbatched_pipelines.append(pipeline_frames)
            with open(Path(self.args.video_frames_dir)/"dataset_info.pkl", 'rb') as f:
                info = pickle.load(f)
                total_samples += info.get('num_samples', info.get('num_frames', 0))
        if not unbatched_pipelines: raise ValueError("FATAL: No data source provided.")
        if len(unbatched_pipelines) > 1: base_dataset = tf.data.Dataset.sample_from_datasets(unbatched_pipelines)
        else: base_dataset = unbatched_pipelines[0]
        raw_image_pipeline = base_dataset.repeat().prefetch(tf.data.AUTOTUNE)
        it = raw_image_pipeline.as_numpy_iterator()
        steps_per_epoch = total_samples // global_batch_size
        print("--- Caching a buffer of raw images for interactive preview... ---")
        preview_buffer = [next(it) for _ in range(50)]; current_preview_idx = 0
        
        PIXELS_PER_STEP = 8192
        @partial(jax.jit, static_argnames=('resolution', 'patch_size'))
        def generate_preview(params, image_batch, resolution=128, patch_size=64):
            latent_params = self.model.apply({'params': params}, image_batch, method=lambda m, i: m.modulator(i))
            x = jnp.linspace(-1, 1, resolution); full_coords = jnp.stack(jnp.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
            pixels = [self.model.apply({'params': params}, latent_params, c, method=lambda m, p, c: m.decode(p, c)) for c in jnp.array_split(full_coords, (resolution**2)//(patch_size**2))]
            return jnp.concatenate(pixels, axis=1).reshape(image_batch.shape[0], resolution, resolution, 3)

        @partial(jax.pmap,
                 axis_name='devices',
                 in_axes=(0, 0, None, 0),
                 donate_argnums=(0,1))
        def train_step_sentinel(state, batch, dampening_factor_arg, key):
            def loss_fn(params):
                feature_grid = self.model.apply({'params': params}, batch, method=lambda m, i: m.observer(m.modulator(i)))
                B, H, W, C = batch.shape
                coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, W), jnp.linspace(-1, 1, H), indexing='ij'), axis=-1).reshape(-1, 2)
                target_pixels_all = batch.reshape(B, H * W, C)
                random_indices = jax.random.randint(key, shape=(B, PIXELS_PER_STEP), minval=0, maxval=H*W)
                batch_indices = jnp.arange(B)[:, None]
                sampled_coords = coords[random_indices]
                sampled_targets = target_pixels_all[batch_indices, random_indices]
                recon_patch = self.model.apply({'params': params}, feature_grid, sampled_coords, method=lambda m, fg, c: m.coord_decoder(fg, c))
                loss = jnp.sum(jnp.abs(sampled_targets - recon_patch)) / PIXELS_PER_STEP
                # --- [SOLUTION] Cast to float32 for stable loss calculation if using bfloat16 ---
                return loss.astype(jnp.float32)
            
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            grads = jax.lax.pmean(grads, 'devices')
            return state.apply_gradients(grads=grads, dampening_factor=dampening_factor_arg), jax.lax.pmean(loss, 'devices')

        @partial(jax.pmap, axis_name='devices', in_axes=(0, 0, 0), donate_argnums=(0,1))
        def train_step_no_sentinel(state, batch, key):
            def loss_fn(params):
                feature_grid = self.model.apply({'params': params}, batch, method=lambda m, i: m.observer(m.modulator(i)))
                B, H, W, C = batch.shape
                coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, W), jnp.linspace(-1, 1, H), indexing='ij'), axis=-1).reshape(-1, 2)
                target_pixels_all = batch.reshape(B, H * W, C)
                random_indices = jax.random.randint(key, shape=(B, PIXELS_PER_STEP), minval=0, maxval=H*W)
                batch_indices = jnp.arange(B)[:, None]
                sampled_coords = coords[random_indices]
                sampled_targets = target_pixels_all[batch_indices, random_indices]
                recon_patch = self.model.apply({'params': params}, feature_grid, sampled_coords, method=lambda m, fg, c: m.coord_decoder(fg, c))
                loss = jnp.sum(jnp.abs(sampled_targets - recon_patch)) / PIXELS_PER_STEP
                # --- [SOLUTION] Cast to float32 for stable loss calculation if using bfloat16 ---
                return loss.astype(jnp.float32)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            grads = jax.lax.pmean(grads, 'devices')
            return state.apply_gradients(grads=grads), jax.lax.pmean(loss, 'devices')
        
        print("--- Compiling JAX functions (one-time cost)... ---")
        preview_compile_batch = np.stack([apply_augmentations_tf(preview_buffer[0], self.q_aug_controller.action_profiles[0]).numpy()])
        p_state_for_compile = replicate(state)
        sharded_batch_for_compile = common_utils.shard(preview_compile_batch.astype(self.dtype))
        compile_key = jax.random.split(train_key, self.num_devices)
        if self.args.use_sentinel:
            train_step_sentinel(p_state_for_compile, sharded_batch_for_compile, 1.0, compile_key)
        
        train_step_no_sentinel(replicate(state), sharded_batch_for_compile, compile_key)
        
        unreplicated_compile_batch = unreplicate(sharded_batch_for_compile)
        generate_preview(unreplicate(p_state.params), unreplicated_compile_batch)
        
        print("--- Compilation complete. Starting training. ---")

        self.progress = Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn(), TextColumn("LR: {task.fields[lr]:.2e}"))
        epoch_task = self.progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs, lr=self.args.lr)
        
        epoch_for_save = start_epoch
        last_step_time = time.time()
        last_ui_update_time = time.time()
        UI_REFRESH_RATE = 15.0

        live = Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
        live.start()
        try:
            for epoch in range(start_epoch, self.args.epochs):
                epoch_for_save = epoch
                self.progress.update(epoch_task, completed=0, epoch=epoch+1)
                for step in range(steps_per_epoch):
                    if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                    
                    train_key, step_key = jax.random.split(train_key)
                    
                    batch_list = []
                    for _ in range(global_batch_size):
                        raw_img_np = next(it)
                        aug_profile = self.q_aug_controller.choose_action()
                        batch_list.append(apply_augmentations_tf(raw_img_np, aug_profile).numpy())
                    batch = np.stack(batch_list).astype(self.dtype) # Cast to correct dtype
                    sharded_batch = common_utils.shard(batch)
                    sharded_keys = jax.random.split(step_key, self.num_devices)

                    opt_state_unrep = unreplicate(p_state.opt_state)
                    if self.q_controller:
                        current_lr = self.q_controller.choose_action()
                        for i, sub_state in enumerate(opt_state_unrep):
                            if isinstance(sub_state, tuple) and sub_state and hasattr(sub_state[-1], 'hyperparams') and 'learning_rate' in sub_state[-1].hyperparams:
                                opt_state_unrep[i][-1].hyperparams['learning_rate'] = jnp.asarray(current_lr)
                                break
                        p_state = p_state.replace(opt_state=replicate(opt_state_unrep))
                    else: current_lr = self.args.lr
                    
                    if self.args.use_sentinel:
                        damp_factor = self.interactive_state.get_sentinel_factor()
                        p_state, loss = train_step_sentinel(p_state, sharded_batch, damp_factor, sharded_keys)
                    else:
                        p_state, loss = train_step_no_sentinel(p_state, sharded_batch, sharded_keys)
                    
                    time_now = time.time()
                    self.steps_per_sec = 1.0 / (time_now - last_step_time + 1e-6)
                    last_step_time = time_now

                    with self.ui_lock:
                        loss_val = unreplicate(loss)
                        self.last_loss_val = loss_val
                        self.recon_loss_history.append(loss_val)
                        if self.q_controller:
                            self.q_controller.update_q_value(loss_val)
                            self.q_controller_status = self.q_controller.status
                        if self.args.use_q_aug_controller:
                            self.q_aug_controller.update_q_value(loss_val)
                            self.q_aug_status = (self.q_aug_controller.current_action_profile["name"], self.q_aug_controller.status)
                        if self.args.use_sentinel:
                            unrep_state = unreplicate(p_state)
                            if unrep_state.step > 0:
                                sentinel_state = next((s for s in unrep_state.opt_state if isinstance(s, SentinelState)), None)
                                if sentinel_state:
                                    self.sentinel_dampen_history.append(sentinel_state.dampened_pct)
                                    self.sentinel_status_str = f"[cyan]{sentinel_state.dampened_count:,} dampened ({sentinel_state.dampened_pct:.3%})[/]"
                                else: self.sentinel_status_str = "[bold red]State not found![/]"
                            else: self.sentinel_status_str = "[dim]Initializing...[/]"

                    self.progress.update(epoch_task, advance=1, lr=current_lr)
                    
                    if (time_now - last_ui_update_time) > (1.0 / UI_REFRESH_RATE):
                        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
                        preview_change = self.interactive_state.get_and_reset_preview_change()
                        if preview_change != 0:
                            current_preview_idx = (current_preview_idx + preview_change) % len(preview_buffer)
                        
                        preview_batch_f32 = np.stack([apply_augmentations_tf(preview_buffer[current_preview_idx], self.q_aug_controller.action_profiles[0]).numpy()])
                        recon = generate_preview(unreplicate(p_state.params), preview_batch_f32[0:1].astype(self.dtype))
                        
                        with self.ui_lock:
                            self.current_preview_np = np.array(((preview_batch_f32[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
                            self.current_recon_np = np.array((np.asarray(recon[0], dtype=np.float32) * 0.5 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
                        
                        live.update(self._generate_layout(), refresh=True)
                        last_ui_update_time = time_now

                if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                if jax.process_index() == 0:
                    self._save_checkpoint(p_state, epoch, ckpt_path)
                    live.console.print(f"--- :floppy_disk: Checkpoint saved for epoch {epoch+1} ---")
        finally:
            live.stop()
            print("\n--- Training loop finished. Cleaning up resources... ---")
            
            self.interactive_state.set_shutdown()
            key_listener_thread.join()
            
            print("--- Cleanup complete. The program will now exit, and JAX will save the cache. ---")
            
        if jax.process_index() == 0 and 'p_state' in locals():
            print("\n--- Saving final model state... ---")
            self._save_checkpoint(p_state, epoch_for_save, ckpt_path)
            print("--- :floppy_disk: Final state saved. ---")
     
class VideoCodecModel(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    
    def setup(self):
        self.image_codec = TopologicalCoordinateGenerator(d_model=self.d_model, latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, name="image_codec", dtype=self.dtype)
        self.dynamics_predictor = DynamicsPredictor(name="dynamics_predictor", dtype=self.dtype)

    def encode(self, images):
        """Runs the image encoder's modulator."""
        return self.image_codec.modulator(images)

    def predict_dynamics(self, p_warped, flow_latent):
        """Runs the dynamics predictor network."""
        return self.dynamics_predictor(p_warped, flow_latent)

    # This part is already correct from the previous fix.
    @partial(nn.remat, static_argnums=(1,))
    def unroll_step(self, loss_resolution, p_prev_reconstructed, xs):
        """
        Performs one step of the video frame prediction process.
        This function is decorated with @nn.remat for gradient checkpointing.
        `loss_resolution` is marked as a static argument.
        """
        flow_t, frame_true_t = xs
        
        def warp_latents(single_latent, single_flow):
            grid_size = single_latent.shape[0]
            grid_y, grid_x = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size), indexing='ij')
            coords = jnp.stack([grid_y, grid_x], axis=-1)
            new_coords = jnp.reshape(coords + single_flow, (grid_size**2, 2)).T
            warped_channels = [jax.scipy.ndimage.map_coordinates(single_latent[..., c], new_coords, order=1, mode='reflect').reshape(grid_size, grid_size) for c in range(3)]
            return jnp.stack(warped_channels, axis=-1)

        flow_latent = jax.image.resize(flow_t, (self.latent_grid_size, self.latent_grid_size, 2), 'bilinear')
        
        p_warped_t = warp_latents(p_prev_reconstructed, flow_latent)
        delta_p = self.predict_dynamics(p_warped_t, flow_latent)
        p_current_reconstructed = p_warped_t + delta_p
        
        x_coords_low_res = jnp.linspace(-1, 1, loss_resolution)
        low_res_coords = jnp.stack(jnp.meshgrid(x_coords_low_res, x_coords_low_res, indexing='ij'), axis=-1).reshape(-1, 2)
        
        recon_pixels = self.image_codec.decode(
            p_current_reconstructed[None, ...], 
            low_res_coords
        ).reshape(loss_resolution, loss_resolution, 3)
        
        target_pixels = jax.image.resize(frame_true_t, (loss_resolution, loss_resolution, 3), 'bilinear')
        loss = jnp.mean(jnp.abs(recon_pixels - target_pixels))
        
        return p_current_reconstructed, loss

    # This part is also correct.
    def __call__(self, frames, flows, loss_resolution):
        """
        The main entry point for the video model during training.
        """
        B, T, H, W, C = frames.shape
        
        def scan_over_clip(initial_frame, flow_clip, frame_clip):
            p0 = self.encode(initial_frame[None, ...])[0]
            
            scan_fn = partial(self.unroll_step, loss_resolution)

            inputs = (flow_clip, frame_clip[1:])
            final_latent, losses = jax.lax.scan(scan_fn, p0, inputs)
            
            full_res_coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,H), jnp.linspace(-1,1,W), indexing='ij'),-1).reshape(-1,2)
            patch_size_sq = (H//2)**2
            
            pixels = [self.image_codec.decode(final_latent[None,...], c) for c in jnp.array_split(full_res_coords, (H*W)//patch_size_sq)]
            last_recon_frame = jnp.concatenate(pixels, axis=1).reshape(H,W,C)
            
            return jnp.mean(losses), last_recon_frame

        batch_losses, last_recon_frames = jax.vmap(scan_over_clip)(frames[:, 0], flows, frames)
        
        return jnp.mean(batch_losses).astype(jnp.float32), last_recon_frames



class VideoTrainer(AdvancedTrainer):
    # ... (init and _generate_layout are unchanged from the previous version) ...
    def __init__(self, args):
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        model = VideoCodecModel(
            d_model=args.d_model, 
            latent_grid_size=args.latent_grid_size, 
            input_image_size=args.image_size,
            dtype=self.dtype
        )
        super().__init__(args, model)
        self.sentinel_dampen_history = deque(maxlen=200)
        self.interactive_state = InteractivityState()
        self.ui_lock = threading.Lock()
        self.last_loss_val = 0.0
        self.current_preview_np = None
        self.current_recon_np = None
        self.spinner_chars = ["🎬", "🎞️", "計算中", "📈", "🎥", "💾", "処理中", "📉"]
        self.spinner_idx = 0
        self.param_count = 0
        self.steps_per_sec = 0.0

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            layout = Layout(name="root")
            layout.split(
                Layout(name="header", size=3),
                Layout(ratio=1, name="main"),
                Layout(self.progress, size=3)
            )
            layout["main"].split_row(Layout(name="left", minimum_size=55), Layout(name="right", ratio=1))

            spinner = self.spinner_chars[self.spinner_idx]
            precision_str = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"{spinner} [bold]Video Dynamics Training[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/] | Trainable: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            layout["header"].update(Panel(Align.center(header_text), style="bold blue", title="[dim]wubumind.ai[/dim]", title_align="right"))

            stats_tbl = Table.grid(expand=True, padding=(0, 1))
            stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right")
            loss_val = self.last_loss_val
            if loss_val < 0.05:   loss_emoji, loss_color = "👌", "bold green"
            elif loss_val < 0.15: loss_emoji, loss_color = "👍", "bold yellow"
            else:                 loss_emoji, loss_color = "😟", "bold red"
            stats_tbl.add_row("Frame Loss (L1)", f"[{loss_color}]{loss_val:.5f}[/] {loss_emoji}")
            stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🏃💨")
            mem, util = self._get_gpu_stats()
            stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            stats_panel = Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue", height=6)

            q_table = Table.grid(expand=True, padding=(0, 1))
            q_table.add_column("Ctrl", style="bold cyan", width=6)
            q_table.add_column("Metric", style="dim", width=10)
            q_table.add_column("Value", justify="left")
            
            if self.q_controller:
                status = self.q_controller.status
                if "IMPROVING" in status:  status_emoji, color = "😎", "bold green"
                elif "STAGNATED" in status: status_emoji, color = "🤔", "bold yellow"
                elif "REGRESSING" in status:status_emoji, color = "😠", "bold red"
                elif "WARMUP" in status:    status_emoji, color = "🐣", "bold blue"
                else:                       status_emoji, color = "🤖", "dim"
                q_table.add_row("🧠 LR", "Status", f"[{color}]{status}[/] {status_emoji}")
                q_table.add_row("", "Reward", f"{self.q_controller.last_reward:+.2f}")
                q_table.add_row("", "Slope", f"{self.q_controller.last_slope:.2e}")
                q_panel = Panel(q_table, title="[bold]🤖 Q-Controller[/]", border_style="green", height=5)
            else:
                q_panel = Panel(Align.center("[dim]Q-Controller Disabled[/dim]"), title="[bold]🤖 Q-Controller[/]", border_style="dim", height=5)

            if self.args.use_sentinel:
                sentinel_layout = Layout()
                log_factor = self.interactive_state.sentinel_dampening_log_factor
                if log_factor <= -2.5:    rocket = "🚀"
                elif log_factor <= -1.5:  rocket = "✈️"
                else:                     rocket = "🛸"
                lever_panel = Panel(get_sentinel_lever_ascii(log_factor), title=f"Dampen {rocket}", title_align="left")
                status_str = getattr(self, 'sentinel_status_str', "Initializing...")
                status_panel = Panel(Align.center(Text(status_str)), title="Status 🚦", title_align="left", height=4)
                sentinel_layout.split_row(Layout(lever_panel), Layout(status_panel))
                sentinel_panel_widget = Panel(sentinel_layout, title="[bold]🕹️ Sentinel Interactive[/]", border_style="yellow", height=11)
                left_panels = [stats_panel, q_panel, sentinel_panel_widget]
            else:
                left_panels = [stats_panel, q_panel]
            layout["left"].update(Group(*left_panels))
            
            spark_w = 40
            recon_panel = Panel(Align.center(f"[cyan]{self._get_sparkline(self.recon_loss_history, spark_w)}[/]"), title=f"Frame Reconstruction Loss (L1)", height=3, border_style="cyan")
            graph_panels = [recon_panel]
            if self.args.use_sentinel:
                sentinel_graph_panel = Panel(Align.center(f"[magenta]{self._get_sparkline(self.sentinel_dampen_history, spark_w)}[/]"), title="Sentinel Dampening %", height=3, border_style="magenta")
                graph_panels.append(sentinel_graph_panel)
            graphs_group = Panel(Group(*graph_panels), title="[bold]📉 Trends[/]")

            preview_content = "..."
            if self.current_preview_np is not None and self.current_recon_np is not None:
                if Pixels is None:
                    preview_content = Align.center(Text("Install `rich-pixels` for previews", style="yellow"))
                else:
                    term_width = 64; h, w, _ = self.current_preview_np.shape; term_height = int(term_width * (h / w) * 0.5)
                    original_pil = Image.fromarray(self.current_preview_np).resize((term_width, term_height), Image.Resampling.LANCZOS)
                    recon_pil = Image.fromarray(self.current_recon_np).resize((term_width, term_height), Image.Resampling.LANCZOS)
                    preview_table = Table.grid(expand=True); preview_table.add_column(ratio=1); preview_table.add_column(ratio=1)
                    preview_table.add_row(Text("Original Frame 📸", justify="center"), Text("Predicted Frame ✨", justify="center"))
                    preview_table.add_row(Pixels.from_image(original_pil), Pixels.from_image(recon_pil))
                    preview_content = preview_table
            preview_panel_widget = Panel(preview_content, title="[bold]🖼️ Live Preview[/]", border_style="green", height=20)

            layout["right"].split(
                graphs_group,
                preview_panel_widget,
                Layout(Align.center(Text("↑/↓: Adjust Sentinel", style="dim")), size=1)
            )
            return layout

    def train(self):
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True)
        key_listener_thread.start()

        phase1_ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        phase2_ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_512_video_dynamics.pkl")
        if not phase1_ckpt_path.exists(): print(f"[FATAL] Phase 1 checkpoint not found at {phase1_ckpt_path}. Run 'train' first."), sys.exit(1)
        print("--- Loading Phase 1 (frozen) model ---");
        with open(phase1_ckpt_path, 'rb') as f: phase1_data = pickle.load(f)

        components = [optax.clip_by_global_norm(1.0)]
        if self.args.use_sentinel:
            components.append(sentinel())

        base_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr)
        trainable_optimizer = optax.chain(*components, base_optimizer)

        loaded_opt_state = None
        if phase2_ckpt_path.exists():
            print(f"--- Resuming Video Dynamics training from {phase2_ckpt_path} ---")
            with open(phase2_ckpt_path, 'rb') as f: phase2_data = pickle.load(f)
            params = {'image_codec': phase1_data['params'], 'dynamics_predictor': phase2_data['params']}
            if 'opt_state' in phase2_data:
                 loaded_opt_state = phase2_data['opt_state']
                 print("--- Optimizer state found in checkpoint. ---")
            else:
                 print("[bold yellow]Warning: No optimizer state in checkpoint. Re-initializing.[/bold yellow]")
            start_epoch = phase2_data.get('epoch', 0) + 1
            if self.q_controller and 'q_controller_state' in phase2_data: self.q_controller.load_state_dict(phase2_data['q_controller_state']); print("--- Q-Controller state loaded. ---")
        else:
            print("--- Initializing new Phase 2 model for Video Dynamics Training ---")
            with jax.default_device(CPU_DEVICE):
                dummy_latent = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), dtype=self.dtype)
                dummy_flow = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 2), dtype=self.dtype)
                dynamics_predictor_params = self.model.init(jax.random.PRNGKey(0), dummy_latent, dummy_flow, method=lambda m, p, f: m.predict_dynamics(p,f))['params']
            params = {'image_codec': phase1_data['params'], 'dynamics_predictor': dynamics_predictor_params['dynamics_predictor']}
            start_epoch = 0

        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, params['dynamics_predictor'], 0)

        def partition_fn(path, param):
            if path and path[0] == 'dynamics_predictor': return 'trainable'
            return 'frozen'
        param_partitions = path_aware_map(partition_fn, params)
        final_optimizer = optax.multi_transform({'trainable': trainable_optimizer, 'frozen': optax.set_to_zero()}, param_partitions)
        
        state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=final_optimizer)

        if loaded_opt_state is not None:
             state = state.replace(opt_state=loaded_opt_state); print("--- Full partitioned optimizer state loaded successfully. ---")
        
        p_state = replicate(state)
        
        # --- [SOLUTION] train_step now accepts `true_clip_lengths` to handle masking ---
        @partial(jax.pmap, axis_name='devices', in_axes=(0, 0, 0, None, None), static_broadcasted_argnums=(3, 4))
        def train_step_video(state, frames, true_clip_lengths, loss_resolution, dampening_factor):
            
            def calculate_flow_for_batch_py(frames_u8_jax, true_clip_lengths_np):
                # This function now gets padded clips, but optical flow should be fine
                # as it calculates frame-by-frame. The invalid flow for padded frames
                # will be ignored by the loss mask.
                frames_u8_np = np.asarray(frames_u8_jax)
                B, T, H, W, _ = frames_u8_np.shape
                all_flows = []
                for b in range(B):
                    clip_flows = []
                    # Use the argument that was passed into the function
                    true_len = true_clip_lengths_np[b]
                    if true_len <= 1:
                        # If clip is too short, just append zeros for flow
                        all_flows.append(np.zeros((T - 1, H, W, 2), dtype=self.dtype))
                        continue
                
                    prvs_gray = cv2.cvtColor(frames_u8_np[b, 0], cv2.COLOR_RGB2GRAY)
                    for t in range(1, int(true_len)): # Cast to int for safety
                        nxt_gray = cv2.cvtColor(frames_u8_np[b, t], cv2.COLOR_RGB2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(prvs_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        clip_flows.append(flow)
                        prvs_gray = nxt_gray
                    # Pad the flows if the clip was shorter than max length
                    stacked_flows = np.array(clip_flows)
                    
                    # Pad the flows if the clip was shorter than max length
                    num_flows = stacked_flows.shape[0] if stacked_flows.ndim == 4 else 0
                    padding_needed = (T - 1) - num_flows
                    
                    if padding_needed > 0:
                        padding = np.zeros((padding_needed, H, W, 2), dtype=stacked_flows.dtype)
                        # If there were no flows to begin with, the result is just the padding
                        if num_flows == 0:
                            final_clip_flows = padding
                        else:
                            # Concatenate the stacked flows and the padding along the time axis (axis=0)
                            final_clip_flows = np.concatenate([stacked_flows, padding], axis=0)
                    else:
                        final_clip_flows = stacked_flows
                    
                    all_flows.append(final_clip_flows)
                return np.stack(all_flows).astype(self.dtype)

            frames_u8 = ((frames * 0.5 + 0.5) * 255).astype(jnp.uint8)
            flow_shape = jax.ShapeDtypeStruct(
                (frames.shape[0], frames.shape[1] - 1, frames.shape[2], frames.shape[3], 2),
                self.dtype
            )
            # Pass true_clip_lengths into the callback
            batch_flows = jax.pure_callback(calculate_flow_for_batch_py, flow_shape, frames_u8, true_clip_lengths)
            batch_flows = jax.lax.stop_gradient(batch_flows)

            def loss_fn(params):
                # The model's __call__ function now needs to accept the true clip lengths to apply a mask
                loss, last_recon_frames = state.apply_fn(
                    {'params': params},
                    frames,
                    batch_flows,
                    loss_resolution,
                    true_clip_lengths # Pass the lengths to the model
                )
                return loss, last_recon_frames

            (loss, recon_frames), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, 'devices')
            return state.apply_gradients(grads=grads, dampening_factor=dampening_factor), jax.lax.pmean(loss, 'devices'), jax.lax.pmean(recon_frames, 'devices')
        
        dataset = create_video_dataset(self.args.data_dir, self.args.batch_size * self.num_devices, self.args.clip_len)
        
        # --- Create a single iterator instance ---
        it = dataset.as_numpy_iterator()

        with open(Path(self.args.data_dir)/"dataset_info.pkl",'rb') as f:
            dataset_info = pickle.load(f)
        
        # The number of steps per epoch is now determined by the number of clips generated
        num_clips = sum(len(list(range(dataset_info['scene_starts'][i], dataset_info['scene_starts'][i+1]))) -1 for i in range(len(dataset_info['scene_starts'])-1))
        steps_per_epoch = num_clips // (self.args.batch_size * self.num_devices)

        print("--- Compiling JAX Video Dynamics training function (this may take a moment)... ---")
        if self.args.use_remat: print("--- [VRAM STRATEGY] Gradient Checkpointing (remat) is ENABLED inside the model. ---")
        if self.dtype == jnp.bfloat16: print("--- [VRAM STRATEGY] BFloat16 Mixed Precision is ENABLED. ---")

        dummy_batch, dummy_lengths = next(it)
        train_step_video(p_state, common_utils.shard(dummy_batch.astype(self.dtype)), common_utils.shard(dummy_lengths), self.args.loss_resolution, 1.0)
        print("--- Compilation complete. Starting Video Dynamics training. ---")
        
        self.progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn(), TextColumn("LR: {task.fields[lr]:.2e}"))
        epoch_task=self.progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch,total_epochs=self.args.epochs, lr=self.args.lr)
        epoch_for_save = start_epoch
        
        last_step_time = time.time()
        last_ui_update_time = time.time()
        UI_REFRESH_RATE = 15.0

        live = Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="crop", auto_refresh=False)
        live.start()
        try:
            for epoch in range(start_epoch, self.args.epochs):
                epoch_for_save = epoch
                self.progress.update(epoch_task, completed=0, epoch=epoch+1)
                for step in range(steps_per_epoch):
                    if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break

                    if self.q_controller:
                        current_lr = self.q_controller.choose_action()
                        opt_state_unrep = unreplicate(p_state.opt_state)
                        opt_state_unrep.inner_states['trainable'].inner_state[-1].hyperparams['learning_rate'] = jnp.asarray(current_lr)
                        p_state = p_state.replace(opt_state=replicate(opt_state_unrep))
                    else: current_lr = self.args.lr

                    batch_np, true_lengths_np = next(it)
                    
                    damp_factor = self.interactive_state.get_sentinel_factor() if self.args.use_sentinel else 1.0

                    p_state, loss, recon_frames = train_step_video(p_state, common_utils.shard(batch_np.astype(self.dtype)), common_utils.shard(true_lengths_np), self.args.loss_resolution, damp_factor)
                    
                    time_now = time.time()
                    self.steps_per_sec = 1.0 / (time_now - last_step_time + 1e-6)
                    last_step_time = time_now

                    with self.ui_lock:
                        loss_val = unreplicate(loss)
                        self.last_loss_val = loss_val
                        self.recon_loss_history.append(loss_val)
                        if self.q_controller:
                            self.q_controller.update_q_value(loss_val)
                        if self.args.use_sentinel:
                            unrep_state = unreplicate(p_state)
                            if unrep_state.step > 0:
                                sentinel_full_state = unrep_state.opt_state.inner_states['trainable'].inner_state
                                sentinel_state = next((s for s in sentinel_full_state if isinstance(s, SentinelState)), None)
                                if sentinel_state:
                                    self.sentinel_dampen_history.append(sentinel_state.dampened_pct)
                                    self.sentinel_status_str = f"[cyan]{sentinel_state.dampened_count:,} dampened ({sentinel_state.dampened_pct:.3%})[/]"
                                else: self.sentinel_status_str = "[bold red]State not found![/]"
                            else: self.sentinel_status_str = "[dim]Initializing...[/]"

                    self.progress.update(epoch_task, advance=1, lr=current_lr)

                    if (time_now - last_ui_update_time) > (1.0 / UI_REFRESH_RATE):
                        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
                        
                        with self.ui_lock:
                            # Display the last *valid* frame of the first clip in the batch
                            last_valid_idx = true_lengths_np[0] - 1
                            self.current_preview_np = np.array((batch_np[0][last_valid_idx]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                            self.current_recon_np = np.array((np.asarray(unreplicate(recon_frames)[0], dtype=np.float32)*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                        
                        live.update(self._generate_layout(), refresh=True)
                        last_ui_update_time = time_now

                if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                if jax.process_index() == 0:
                    state_to_save = unreplicate(p_state)
                    data_to_save = {'params': jax.device_get(state_to_save.params['dynamics_predictor']), 
                                    'opt_state': jax.device_get(state_to_save.opt_state), 'epoch': epoch}
                    if self.q_controller: data_to_save['q_controller_state'] = self.q_controller.state_dict()
                    with open(phase2_ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
                    live.console.print(f"--- :floppy_disk: Phase 2 checkpoint saved for epoch {epoch+1} ---")
        finally:
            live.stop()
            print("\n--- Training loop finished. Cleaning up resources... ---")
            self.interactive_state.set_shutdown()
            key_listener_thread.join()
            print("--- Cleanup complete. The program will now exit, and JAX will save the cache. ---")

        if jax.process_index() == 0 and 'p_state' in locals():
            print("\n--- Saving final model state... ---")
            state_to_save = unreplicate(p_state)
            data_to_save = {'params': jax.device_get(state_to_save.params['dynamics_predictor']), 
                            'opt_state': jax.device_get(state_to_save.opt_state), 'epoch': epoch_for_save}
            if self.q_controller: data_to_save['q_controller_state'] = self.q_controller.state_dict()
            with open(phase2_ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
            print(f"--- :floppy_disk: Final Phase 2 state saved to {phase2_ckpt_path}. ---")
            
# --- [SOLUTION] VideoCodecModel needs to be updated to handle masked loss ---
class VideoCodecModel(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    
    def setup(self):
        self.image_codec = TopologicalCoordinateGenerator(d_model=self.d_model, latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, name="image_codec", dtype=self.dtype)
        self.dynamics_predictor = DynamicsPredictor(name="dynamics_predictor", dtype=self.dtype)

    def encode(self, images):
        return self.image_codec.modulator(images)

    def predict_dynamics(self, p_warped, flow_latent):
        return self.dynamics_predictor(p_warped, flow_latent)

    @partial(nn.remat, static_argnums=(1,))
    def unroll_step(self, loss_resolution, p_prev_reconstructed, xs):
        flow_t, frame_true_t = xs
        def warp_latents(single_latent, single_flow):
            grid_size = single_latent.shape[0]
            grid_y, grid_x = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size), indexing='ij')
            coords = jnp.stack([grid_y, grid_x], axis=-1)
            new_coords = jnp.reshape(coords + single_flow, (grid_size**2, 2)).T
            warped_channels = [jax.scipy.ndimage.map_coordinates(single_latent[..., c], new_coords, order=1, mode='reflect').reshape(grid_size, grid_size) for c in range(3)]
            return jnp.stack(warped_channels, axis=-1)
        flow_latent = jax.image.resize(flow_t, (self.latent_grid_size, self.latent_grid_size, 2), 'bilinear')
        p_warped_t = warp_latents(p_prev_reconstructed, flow_latent)
        delta_p = self.predict_dynamics(p_warped_t, flow_latent)
        p_current_reconstructed = p_warped_t + delta_p
        x_coords_low_res = jnp.linspace(-1, 1, loss_resolution)
        low_res_coords = jnp.stack(jnp.meshgrid(x_coords_low_res, x_coords_low_res, indexing='ij'), axis=-1).reshape(-1, 2)
        recon_pixels = self.image_codec.decode(p_current_reconstructed[None, ...], low_res_coords).reshape(loss_resolution, loss_resolution, 3)
        target_pixels = jax.image.resize(frame_true_t, (loss_resolution, loss_resolution, 3), 'bilinear')
        loss = jnp.mean(jnp.abs(recon_pixels - target_pixels))
        return p_current_reconstructed, loss

    def __call__(self, frames, flows, loss_resolution, true_clip_lengths):
        B, T, H, W, C = frames.shape
        
        def scan_over_clip(initial_frame, flow_clip, frame_clip, true_len):
            p0 = self.encode(initial_frame[None, ...])[0]
            scan_fn = partial(self.unroll_step, loss_resolution)
            inputs = (flow_clip, frame_clip[1:])
            final_latent, losses = jax.lax.scan(scan_fn, p0, inputs)
            
            # --- [SOLUTION] Create and apply a mask to the losses ---
            # The loss is for T-1 frames. True length is for T frames.
            # So, the number of valid P-frames is true_len - 1.
            mask = jnp.arange(T - 1) < (true_len - 1)
            masked_loss = jnp.sum(losses * mask) / jnp.maximum(1.0, jnp.sum(mask))

            full_res_coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,H), jnp.linspace(-1,1,W), indexing='ij'),-1).reshape(-1,2)
            patch_size_sq = (H//2)**2
            pixels = [self.image_codec.decode(final_latent[None,...], c) for c in jnp.array_split(full_res_coords, (H*W)//patch_size_sq)]
            last_recon_frame = jnp.concatenate(pixels, axis=1).reshape(H,W,C)
            
            return masked_loss, last_recon_frame

        batch_losses, last_recon_frames = jax.vmap(scan_over_clip)(frames[:, 0], flows, frames, true_clip_lengths)
        
        return jnp.mean(batch_losses).astype(jnp.float32), last_recon_frames    
# --- [SOLUTION] VideoCodecModel needs to be updated to handle masked loss ---
class VideoCodecModel(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    
    def setup(self):
        self.image_codec = TopologicalCoordinateGenerator(d_model=self.d_model, latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, name="image_codec", dtype=self.dtype)
        self.dynamics_predictor = DynamicsPredictor(name="dynamics_predictor", dtype=self.dtype)

    def encode(self, images):
        return self.image_codec.modulator(images)

    def predict_dynamics(self, p_warped, flow_latent):
        return self.dynamics_predictor(p_warped, flow_latent)

    @partial(nn.remat, static_argnums=(1,))
    def unroll_step(self, loss_resolution, p_prev_reconstructed, xs):
        flow_t, frame_true_t = xs
        def warp_latents(single_latent, single_flow):
            grid_size = single_latent.shape[0]
            grid_y, grid_x = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size), indexing='ij')
            coords = jnp.stack([grid_y, grid_x], axis=-1)
            new_coords = jnp.reshape(coords + single_flow, (grid_size**2, 2)).T
            warped_channels = [jax.scipy.ndimage.map_coordinates(single_latent[..., c], new_coords, order=1, mode='reflect').reshape(grid_size, grid_size) for c in range(3)]
            return jnp.stack(warped_channels, axis=-1)
        flow_latent = jax.image.resize(flow_t, (self.latent_grid_size, self.latent_grid_size, 2), 'bilinear')
        p_warped_t = warp_latents(p_prev_reconstructed, flow_latent)
        delta_p = self.predict_dynamics(p_warped_t, flow_latent)
        p_current_reconstructed = p_warped_t + delta_p
        x_coords_low_res = jnp.linspace(-1, 1, loss_resolution)
        low_res_coords = jnp.stack(jnp.meshgrid(x_coords_low_res, x_coords_low_res, indexing='ij'), axis=-1).reshape(-1, 2)
        recon_pixels = self.image_codec.decode(p_current_reconstructed[None, ...], low_res_coords).reshape(loss_resolution, loss_resolution, 3)
        target_pixels = jax.image.resize(frame_true_t, (loss_resolution, loss_resolution, 3), 'bilinear')
        loss = jnp.mean(jnp.abs(recon_pixels - target_pixels))
        return p_current_reconstructed, loss

    def __call__(self, frames, flows, loss_resolution, true_clip_lengths):
        B, T, H, W, C = frames.shape
        
        def scan_over_clip(initial_frame, flow_clip, frame_clip, true_len):
            p0 = self.encode(initial_frame[None, ...])[0]
            scan_fn = partial(self.unroll_step, loss_resolution)
            inputs = (flow_clip, frame_clip[1:])
            final_latent, losses = jax.lax.scan(scan_fn, p0, inputs)
            
            # --- [SOLUTION] Create and apply a mask to the losses ---
            # The loss is for T-1 frames. True length is for T frames.
            # So, the number of valid P-frames is true_len - 1.
            mask = jnp.arange(T - 1) < (true_len - 1)
            masked_loss = jnp.sum(losses * mask) / jnp.maximum(1.0, jnp.sum(mask))

            full_res_coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,H), jnp.linspace(-1,1,W), indexing='ij'),-1).reshape(-1,2)
            patch_size_sq = (H//2)**2
            pixels = [self.image_codec.decode(final_latent[None,...], c) for c in jnp.array_split(full_res_coords, (H*W)//patch_size_sq)]
            last_recon_frame = jnp.concatenate(pixels, axis=1).reshape(H,W,C)
            
            return masked_loss, last_recon_frame

        batch_losses, last_recon_frames = jax.vmap(scan_over_clip)(frames[:, 0], flows, frames, true_clip_lengths)
        
        return jnp.mean(batch_losses).astype(jnp.float32), last_recon_frames



# =================================================================================================
# 5. COMPRESSION & GENERATION LOGIC
# =================================================================================================

class Compressor:
    def __init__(self, args):
        self.args = args
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        if not model_path.exists(): print(f"[FATAL] Model file not found at {model_path}. Train a model first."), sys.exit(1)
        if jax.process_index() == 0: print(f"--- Loading compressor model from {model_path} ---")
        with open(model_path, 'rb') as f: self.params = pickle.load(f)['params']

    @partial(jax.jit, static_argnames=('self',))
    def _encode(self, image_batch):
        return self.model.apply({'params': self.params}, image_batch, method=lambda m, i: m.modulator(i))
    @partial(jax.jit, static_argnames=('self', 'resolution', 'patch_size'))
    def _decode_batched(self, latent_batch, resolution=512, patch_size=256):
        x = jnp.linspace(-1, 1, resolution); grid_x, grid_y = jnp.meshgrid(x, x, indexing='ij')
        full_coords = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        pixels = [self.model.apply({'params': self.params}, latent_batch, c, method=lambda m, p, c: m.decode(p, c)) for c in jnp.array_split(full_coords, (resolution*resolution)//(patch_size*patch_size))]
        return jnp.concatenate(pixels, axis=1).reshape(latent_batch.shape[0], resolution, resolution, 3)

    def compress(self):
        image_path = Path(self.args.image_path);
        img = Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        img_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0; image_batch = jnp.expand_dims(img_np, axis=0)
        latent_grid = self._encode(image_batch); latent_grid_uint8 = self._quantize_latents(latent_grid)
        output_path = Path(self.args.output_path); np.save(output_path, latent_grid_uint8)
        original_size = image_path.stat().st_size; compressed_size = output_path.stat().st_size
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        print(f"✅ Image compressed to {output_path}\n   Original: {original_size/1024:.2f} KB, Compressed: {compressed_size/1024:.2f} KB, Ratio: {ratio:.2f}x")
    def decompress(self):
        compressed_path = Path(self.args.compressed_path); latent_grid_uint8 = np.load(compressed_path)
        latent_grid = self._dequantize_latents(latent_grid_uint8); latent_batch = jnp.expand_dims(latent_grid, axis=0)
        print("--- Decompressing (rendering 512x512 image)... ---")
        reconstruction_batch = self._decode_batched(latent_batch); recon_np = np.array(reconstruction_batch[0])
        recon_np = ((recon_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8); recon_img = Image.fromarray(recon_np)
        output_path = Path(self.args.output_path); recon_img.save(output_path)
        print(f"✅ Decompressed 512x512 image saved to {output_path}")

    def _quantize_latents(self, latent_grid_float):
        params = latent_grid_float[0]; delta, chi, radius = params[..., 0], params[..., 1], params[..., 2]
        delta_norm=(delta/jnp.pi)*0.5+0.5; chi_norm=(chi/(jnp.pi/4.0))*0.5+0.5; radius_norm=radius/(jnp.pi/2.0)
        num_bins=256
        return np.stack([np.array(jnp.round(p*(num_bins-1)),dtype=np.uint8) for p in [delta_norm,chi_norm,radius_norm]],axis=-1)
    def _dequantize_latents(self, latent_grid_uint8):
        num_bins=256; latent_grid_float_norm=jnp.asarray(latent_grid_uint8,dtype=jnp.float32)/(num_bins-1)
        delta_norm,chi_norm,radius_norm = latent_grid_float_norm[...,0],latent_grid_float_norm[...,1],latent_grid_float_norm[...,2]
        delta=(delta_norm-0.5)*2.0*jnp.pi; chi=(chi_norm-0.5)*2.0*(jnp.pi/4.0); radius=radius_norm*(jnp.pi/2.0)
        return jnp.stack([delta,chi,radius],axis=-1)


class VideoCompressor(Compressor):
    # --- [ARCHITECTURAL REFACTOR] ---
    # The entire VideoCompressor is refactored for the new dynamics prediction model.
    def __init__(self, args):
        super().__init__(args)
        self.video_model = VideoCodecModel(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size)

        phase2_model_path = Path(f"{self.args.basename}_{self.args.d_model}d_512_video_dynamics.pkl")
        if not phase2_model_path.exists():
            print(f"[FATAL] Phase 2 model for video dynamics not found at {phase2_model_path}. Run 'train-video' first.")
            sys.exit(1)

        print(f"--- Loading Phase 2 Video Dynamics model from {phase2_model_path} ---")
        with open(phase2_model_path, 'rb') as f: phase2_params = pickle.load(f)['params']
        self.full_params = jax.device_put({'image_codec': self.params, 'dynamics_predictor': phase2_params})

        def warp_latents_standalone(single_latent, single_flow):
            H, W, _ = single_latent.shape
            grid_y, grid_x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'); coords = jnp.stack([grid_y, grid_x], axis=-1)
            new_coords = jnp.reshape(coords + single_flow, (H * W, 2)).T
            warped_channels = [jax.scipy.ndimage.map_coordinates(single_latent[..., c], new_coords, order=1, mode='reflect').reshape(H, W) for c in range(3)]
            return jnp.stack(warped_channels, axis=-1)
        self._vmapped_warp_fn = jax.jit(jax.vmap(warp_latents_standalone))

    def _quantize_flow(self, flow_np):
        flow_scaled = (flow_np + 32.0) * (65535.0 / 64.0)
        return np.clip(flow_scaled, 0, 65535).astype(np.uint16)

    def _dequantize_flow(self, flow_uint16):
        flow_scaled = flow_uint16.astype(np.float32)
        return (flow_scaled * (64.0 / 65535.0)) - 32.0
    
    # Quantization for Δp, the "tiny Hamiltons"
    def _quantize_delta_p(self, delta_p_float):
        # Assumes delta_p is in [-0.25, 0.25] from the network's tanh * 0.25
        norm_grid = (delta_p_float / 0.5) + 0.5; num_bins = 256
        return np.array(jnp.round(jnp.clip(norm_grid, 0, 1) * (num_bins - 1)), dtype=np.uint8)

    def _dequantize_delta_p(self, delta_p_uint8):
        num_bins=256; norm_grid = jnp.asarray(delta_p_uint8, dtype=jnp.float32) / (num_bins - 1)
        return (norm_grid - 0.5) * 0.5

    @partial(jax.jit, static_argnames=('self',))
    def _decode_and_convert_jitted(self, latent_batch):
        recon = self._decode_batched(latent_batch)
        return ((recon[0] * 0.5 + 0.5) * 255).astype(jnp.uint8)

    def video_compress(self):
        video_p, output_path = Path(self.args.video_path), Path(self.args.output_path)
        cap = cv2.VideoCapture(str(video_p))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or num_frames == 0: sys.exit(f"[FATAL] Could not open video file: {video_p}")

        print(f"--- Compressing video with dynamics predictor: {output_path} ---")

        encoded_frames_data = []

        @partial(jax.jit, static_argnames=('self',))
        def _predict_p_frame_jit(self, p_prev_recon, flow_np):
            flow_latent = jax.image.resize(jnp.expand_dims(flow_np, 0), (1, self.args.latent_grid_size, self.args.latent_grid_size, 2), 'bilinear')
            p_warped = self._vmapped_warp_fn(p_prev_recon, flow_latent)
            delta_p = self.video_model.apply({'params': self.full_params}, p_warped, flow_latent, method='predict_dynamics')
            p_next_recon = p_warped + delta_p
            return p_next_recon, delta_p

        pbar = tqdm(total=num_frames, desc="Compressing Frames")
        ret, frame = cap.read()
        if not ret: sys.exit("[FATAL] Could not read first frame.")

        # --- I-Frame ---
        prvs_frame_np = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512, 512), Image.Resampling.LANCZOS))
        frame_normalized = (prvs_frame_np.astype(np.float32) / 127.5) - 1.0
        p_current_latent_gpu = self._encode(jnp.expand_dims(frame_normalized, 0))
        quantized_iframe = self._quantize_latents(p_current_latent_gpu)
        encoded_frames_data.append(('I', quantized_iframe.tobytes()))
        pbar.update(1)

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_np = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512, 512), Image.Resampling.LANCZOS))
            
            # --- P-Frame ---
            prvs_gray = cv2.cvtColor(prvs_frame_np, cv2.COLOR_RGB2GRAY)
            nxt_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            flow_np = cv2.calcOpticalFlowFarneback(prvs_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Predict next latent and the delta_p that was used
            p_next_latent_gpu, delta_p_gpu = _predict_p_frame_jit(self, p_current_latent_gpu, flow_np)

            # Quantize the flow and the predicted change in parameters (Δp)
            quantized_flow = self._quantize_flow(flow_np)
            quantized_delta_p = self._quantize_delta_p(np.asarray(delta_p_gpu)[0])
            encoded_frames_data.append(('P', quantized_flow.tobytes() + quantized_delta_p.tobytes()))
            
            p_current_latent_gpu = p_next_latent_gpu
            prvs_frame_np = frame_np
            pbar.update(1)

        pbar.close(); cap.release()

        with open(output_path, 'wb') as f:
            f.write(struct.pack('<HI', self.args.latent_grid_size, len(encoded_frames_data)))
            for frame_type, data in encoded_frames_data:
                frame_type_byte = b'\x01' if frame_type == 'I' else b'\x00'
                f.write(frame_type_byte)
                f.write(struct.pack('<I', len(data)))
                f.write(data)

        original_size = video_p.stat().st_size; compressed_size = output_path.stat().st_size
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        i_count = sum(1 for type, _ in encoded_frames_data if type == 'I')
        print(f"✅ Compression complete to {output_path} ({compressed_size/1024**2:.2f} MB). Ratio: {ratio:.2f}x")
        print(f"   Frame breakdown: I-Frames: {i_count}, P-Frames: {len(encoded_frames_data) - i_count}")

    def video_decompress(self):
        input_path, output_path = Path(self.args.input_path), Path(self.args.output_path)
        print(f"--- Decompressing video with dynamics predictor: {input_path} ---")

        with open(input_path, 'rb') as f:
            latent_grid_size, num_frames = struct.unpack('<HI', f.read(6))
            compressed_frames_data = []
            for _ in range(num_frames):
                frame_type = 'I' if f.read(1) == b'\x01' else 'P'
                data_length = struct.unpack('<I', f.read(4))[0]
                compressed_frames_data.append((frame_type, f.read(data_length)))

        @partial(jax.jit, static_argnames=('self',))
        def _reconstruct_p_frame_jit(self, p_prev_recon, flow_np, delta_p_np):
            flow_latent = jax.image.resize(jnp.expand_dims(flow_np, 0), (1, self.args.latent_grid_size, self.args.latent_grid_size, 2), 'bilinear')
            p_warped = self._vmapped_warp_fn(p_prev_recon, flow_latent)
            p_current = p_warped + jnp.expand_dims(delta_p_np, 0)
            return p_current

        current_latent_gpu = None
        writer = imageio.get_writer(output_path, fps=30)

        flow_data_len = 512 * 512 * 2 * 2 # uint16 = 2 bytes
        delta_p_data_len = latent_grid_size * latent_grid_size * 3

        for frame_type, data in tqdm(compressed_frames_data, desc="Decoding Frames"):
            if frame_type == 'I':
                quantized_latent = np.frombuffer(data, dtype=np.uint8).reshape((latent_grid_size, latent_grid_size, 3))
                current_latent_gpu = jax.device_put(jnp.expand_dims(self._dequantize_latents(quantized_latent), 0))
            elif frame_type == 'P':
                flow_bytes = data[:flow_data_len]
                delta_p_bytes = data[flow_data_len:]

                flow_uint16 = np.frombuffer(flow_bytes, dtype=np.uint16).reshape((512, 512, 2))
                flow_np = self._dequantize_flow(flow_uint16)

                delta_p_uint8 = np.frombuffer(delta_p_bytes, dtype=np.uint8).reshape((latent_grid_size, latent_grid_size, 3))
                delta_p_np = self._dequantize_delta_p(delta_p_uint8)

                current_latent_gpu = _reconstruct_p_frame_jit(self, current_latent_gpu, flow_np, delta_p_np)

            decoded_frame_np = np.asarray(self._decode_and_convert_jitted(current_latent_gpu))
            writer.append_data(decoded_frame_np)

        writer.close(); print(f"✅ Video decompressed to {output_path}")
    # --- [END ARCHITECTURAL REFACTOR] ---


class Generator(Compressor):
    def __init__(self, args):
        super().__init__(args)
        if clip is None: print("[FATAL] CLIP and PyTorch are required."), sys.exit(1)
        self.latent_db_path = Path(self.args.image_dir) / f"latent_database_{self.args.latent_grid_size}grid.pkl"
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)

    def _get_latent_for_text(self, text):
        if not self.latent_db_path.exists(): print(f"[FATAL] Latent DB not found. Run 'build-db' first."), sys.exit(1)
        with open(self.latent_db_path, 'rb') as f: db = pickle.load(f)
        image_features = db['clip_features'].to(_clip_device)
        with torch.no_grad(): text_features = self.clip_model.encode_text(clip.tokenize([text]).to(_clip_device))
        image_features /= image_features.norm(dim=-1,keepdim=True); text_features /= text_features.norm(dim=-1,keepdim=True)
        similarity = (100.0*image_features@text_features.T).softmax(dim=0); best_idx = similarity.argmax().item()
        print(f"--- Best match for '{text}' is image #{best_idx} ---")
        return jnp.asarray(db['latents'][best_idx])

    def build_db(self):
        print("--- Building latent and CLIP feature database ---")
        dataset_gen = create_dataset(self.args.image_dir, is_training=False)
        dataset_gen = dataset_gen.map(lambda img_u8: (tf.cast(img_u8, tf.float32) / 127.5) - 1.0)
        dataset_gen = dataset_gen.batch(self.args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        all_latents, all_clip_features = [], []
        for batch_np in tqdm(dataset_gen.as_numpy_iterator(), desc="Encoding Images"):
            latents = self._encode(jnp.asarray(batch_np)); all_latents.append(np.array(latents))
            with torch.no_grad():
                batch_torch = torch.from_numpy(batch_np).to(_clip_device).permute(0, 3, 1, 2)
                image_features = self.clip_model.encode_image(batch_torch); all_clip_features.append(image_features.cpu())
        db = {'latents': np.concatenate(all_latents), 'clip_features': torch.cat(all_clip_features)}
        with open(self.latent_db_path, 'wb') as f: pickle.dump(db, f)
        print(f"✅ DB with {len(db['latents'])} entries saved to {self.latent_db_path}")

    def generate(self):
        latent_grid = self._get_latent_for_text(self.args.prompt)
        print(f"--- Generating 512x512 image for prompt: '{self.args.prompt}' ---")
        reconstruction = self._decode_batched(jnp.expand_dims(latent_grid, 0))
        recon_np = np.array(reconstruction[0]); recon_np = ((recon_np*0.5+0.5).clip(0,1)*255).astype(np.uint8)
        img = Image.fromarray(recon_np)
        save_path = f"GEN_{''.join(c for c in self.args.prompt if c.isalnum())[:50]}.png"; img.save(save_path)
        print(f"✅ Image saved to {save_path}")

    def animate(self):
        print(f"--- Creating animation from '{self.args.start}' to '{self.args.end}' ---")
        latent_start = self._get_latent_for_text(self.args.start); latent_end = self._get_latent_for_text(self.args.end)
        frames = []
        for i in tqdm(range(self.args.steps), desc="Generating Frames"):
            alpha = i / (self.args.steps - 1)
            interp_latent = latent_start * (1 - alpha) + latent_end * alpha
            reconstruction = self._decode_batched(jnp.expand_dims(interp_latent, 0))
            recon_np = np.array(reconstruction[0]); recon_np = ((recon_np*0.5+0.5).clip(0,1)*255).astype(np.uint8)
            frames.append(Image.fromarray(recon_np))
        start_name, end_name = ''.join(c for c in self.args.start if c.isalnum())[:20], ''.join(c for c in self.args.end if c.isalnum())[:20]
        save_path = f"ANIM_{start_name}_to_{end_name}.gif"
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
        print(f"✅ Animation saved to {save_path}")


# =================================================================================================
# 6. MAIN EXECUTION BLOCK
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Topological Coordinate Generator for High-Resolution Images & Video (Advanced Trainer)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--basename', type=str, required=True, help="Basename for model files (e.g., 'my_model').")
    parent_parser.add_argument('--d-model', type=int, default=128, help="Model dimension.")
    parent_parser.add_argument('--latent-grid-size', type=int, default=16, help="Size of the latent grid (e.g., 16 for 16x16).")
    parent_parser.add_argument('--image-size', type=int, default=512, help="Image resolution the model is trained on.")

    p_prep = subparsers.add_parser("prepare-data", help="Convert images to TFRecords."); p_prep.add_argument('--image-dir', type=str, required=True)

    p_train = subparsers.add_parser("train", help="PHASE 1: Train the static image AE with advanced tools.", parents=[parent_parser])
    p_train.add_argument('--tfrecord-dir', type=str, default=None, help="Path to a directory with TFRecords (e.g., from 'prepare-data').")
    p_train.add_argument('--video-frames-dir', type=str, default=None, help="Path to a prepared video data directory (e.g., from 'prepare-video-data').")
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--batch-size', type=int, default=4, help="Batch size PER DEVICE.")
    p_train.add_argument('--lr', type=float, default=2e-4)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--use-q-controller', action='store_true', help="Enable adaptive LR via Q-Learning.")
    p_train.add_argument('--use-q-aug-controller', action='store_true', help="Enable adaptive Data Augmentation via Q-Learning.")
    p_train.add_argument('--use-sentinel', action='store_true', help="Enable Sentinel optimizer to dampen oscillations.")
    p_train.add_argument('--finetune', action='store_true', help="Use finetuning Q-Controller config (not typically used for phase 1).")
    # --- [SOLUTION] Add bfloat16 argument for both trainers ---
    p_train.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 mixed precision to save VRAM.")


    p_prep_vid = subparsers.add_parser("prepare-video-data", help="Extract frames and optical flow from a video.")
    p_prep_vid.add_argument('--video-path', type=str, required=True); p_prep_vid.add_argument('--data-dir', type=str, required=True)
    p_train_vid = subparsers.add_parser("train-video", help="PHASE 2: Train the video dynamics model with advanced tools.", parents=[parent_parser])
    p_train_vid.add_argument('--data-dir', type=str, required=True); p_train_vid.add_argument('--epochs', type=int, default=200); p_train_vid.add_argument('--batch-size', type=int, default=2, help="Batch size PER DEVICE for video clips.")
    p_train_vid.add_argument('--lr', type=float, default=1e-4); p_train_vid.add_argument('--seed', type=int, default=42); p_train_vid.add_argument('--clip-len', type=int, default=8)
    p_train_vid.add_argument('--use-q-controller', action='store_true', help="Enable adaptive LR via Q-Learning."); p_train_vid.add_argument('--use-sentinel', action='store_true', help="Enable Sentinel optimizer to dampen oscillations.")
    p_train_vid.add_argument('--finetune', action='store_true', help="Use finetuning Q-Controller config.")
    p_train_vid.add_argument('--loss-resolution', type=int, default=256, help="Intermediate resolution for reconstruction loss during training (e.g., 128, 256).")
    p_train_vid.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 mixed precision to save VRAM.")
    # --- [SOLUTION] Add remat argument for video trainer ---
    p_train_vid.add_argument('--use-remat', action='store_true', help="Use gradient checkpointing (remat) to save VRAM during video training.")

    
    p_comp = subparsers.add_parser("compress", help="Compress a single image to a file.", parents=[parent_parser]); p_comp.add_argument('--image-path', type=str, required=True); p_comp.add_argument('--output-path', type=str, required=True)
    p_dcomp = subparsers.add_parser("decompress", help="Decompress a file to an image.", parents=[parent_parser]); p_dcomp.add_argument('--compressed-path', type=str, required=True); p_dcomp.add_argument('--output-path', type=str, required=True)

    p_vcomp = subparsers.add_parser("video-compress", help="Compress a video to a single efficient file.", parents=[parent_parser])
    p_vcomp.add_argument('--video-path', type=str, required=True)
    p_vcomp.add_argument('--output-path', type=str, required=True, help="Path to the output .wubu file.")

    p_vdcomp = subparsers.add_parser("video-decompress", help="Decompress a custom video file.", parents=[parent_parser])
    p_vdcomp.add_argument('--input-path', type=str, required=True, help="Path to the compressed .wubu file.")
    p_vdcomp.add_argument('--output-path', type=str, required=True, help="Path for the output video (e.g., video.mp4).")

    p_db = subparsers.add_parser("build-db", help="Build a latent database for generative tasks.", parents=[parent_parser]); p_db.add_argument('--image-dir', type=str, required=True); p_db.add_argument('--batch-size', type=int, default=16)
    p_gen = subparsers.add_parser("generate", help="Generate an image from a text prompt.", parents=[parent_parser]); p_gen.add_argument('--image-dir', type=str, required=True); p_gen.add_argument('--prompt', type=str, required=True)
    p_anim = subparsers.add_parser("animate", help="Create an animation between two prompts.", parents=[parent_parser]); p_anim.add_argument('--image-dir', type=str, required=True); p_anim.add_argument('--start', type=str, required=True); p_anim.add_argument('--end', type=str, required=True); p_anim.add_argument('--steps', type=int, default=60)

    args = parser.parse_args()
    if args.command == "prepare-data": prepare_data(args.image_dir)
    elif args.command == "train": ImageTrainer(args).train()
    elif args.command == "prepare-video-data": prepare_video_data(args.video_path, args.data_dir)
    elif args.command == "train-video": VideoTrainer(args).train()
    elif args.command == "compress": Compressor(args).compress()
    elif args.command == "decompress": Compressor(args).decompress()
    elif args.command == "video-compress": VideoCompressor(args).video_compress()
    elif args.command == "video-decompress": VideoCompressor(args).video_decompress()
    elif args.command == "build-db": Generator(args).build_db()
    elif args.command == "generate": Generator(args).generate()
    elif args.command == "animate": Generator(args).animate()
if __name__ == "__main__":
    main()
    print("\n--- Program finished normally. ---")
