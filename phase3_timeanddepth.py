# =================================================================================================
#
#                    PHASE 3: TEXT-TO-IMAGE GENERATIVE FRAMEWORK
#
#     A Deterministic, Physics-Informed Framework for Structured Media Synthesis
#                   (Upgraded with Advanced Training Toolkit)
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
import signal
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Optional, Dict, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor # ### OPTIMIZATION 2 ###
import atexit

# --- Environment and JAX Setup ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    # Get the directory where the script is located.
    script_dir = Path(__file__).parent.resolve()
    cache_dir = script_dir / ".jax_cache"
    cache_dir.mkdir(exist_ok=True) # Ensure the directory exists
    
    # Set the environment variable for JAX to find the cache.
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    
    print(f"--- JAX persistent cache enabled at: {cache_dir} ---")
    
    # This ensures JAX waits for the cache to be written before exiting.
    def _jax_shutdown():
        """Function to be called at script exit to ensure JAX cleans up."""
        # This is a blocking call that waits for cache writes.
        jax.clear_caches()
        print("--- JAX cache finalized. Script exiting. ---")
        
    # Register the function to run when the script exits.
    atexit.register(_jax_shutdown)

except NameError:
    # This fallback is for interactive environments like Jupyter.
    cache_dir = Path.home() / ".jax_cache_global"
    cache_dir.mkdir(exist_ok=True)
    os.environ['JAX_PERSISTENT_CACHE_PATH'] = str(cache_dir)
    print(f"--- JAX persistent cache enabled at (fallback global): {cache_dir} ---")
# 
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
import optax
import chex
from flax import linen as nn
from flax.linen import initializers
from flax.linen import dot_product_attention
import jax.lax # Import the lax module
from flax.training import train_state
from flax.training.train_state import TrainState
from flax.jax_utils import unreplicate, replicate
from flax.training.common_utils import shard
from flax.linen import remat
from functools import partial
from tqdm import tqdm
from PIL import Image
try:
    from flash_attn_jax import flash_mha
except ImportError:
    print("[FATAL] Required dependency `flash-attn-jax` is missing. Please run: pip install flash-attn-jax, pip install einops")
    sys.exit(1)
# --- Dependency Checks ---
# =================================================================================================
# GUI PREVIEW DEPENDENCIES (Place near other imports)
# =================================================================================================
try:
    from rich_pixels import Pixels
except ImportError:
    print("[Warning] `rich-pixels` not found. Visual preview in GUI will be disabled. Run: pip install rich-pixels")
    Pixels = None
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import clip
    import torch
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
    _clip_device = "cuda" if "cuda" in str(jax.devices()[0]).lower() else "cpu"
except ImportError:
    print("[FATAL] Required dependencies missing: tensorflow, clip-by-openai, torch, rich, pynvml, chex. Please install them.")
    sys.exit(1)

# Conditional imports for keyboard listening
if platform.system() == "Windows":
    import msvcrt
else:
    import tty, termios, select

# --- JAX Configuration ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False)


# =================================================================================================
# 1. CORE MODEL & TOOLKIT DEFINITIONS (Copied from Phase 1/2 for parameter loading)
# =================================================================================================

# --- [FIX] Custom TrainState & SentinelState definitions moved here to prevent pickle loading errors ---
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
    sign_history: chex.ArrayTree; dampened_count: Optional[jnp.ndarray] = None; dampened_pct: Optional[jnp.ndarray] = None

# --- Core Physics/AE Model Definitions ---
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
            x = nn.Conv(features, (1, 1), name="pre_resize_projection", dtype=self.dtype)(x); x = nn.gelu(x)
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
        B, H, W, _ = path_params_grid.shape; path_params = path_params_grid.reshape(B, H * W, 3)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta); chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
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
    def __call__(self, feature_grid, coords):
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
        for i in range(self.mlp_depth):
            h = nn.gelu(nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h))
            if i == self.mlp_depth // 2: h = jnp.concatenate([h, mlp_in], -1)
        return nn.tanh(nn.Dense(3, name="mlp_out", dtype=self.dtype)(h))

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(self.d_model, name="coord_decoder", dtype=self.dtype)
    def decode(self, path_params, coords):
        return self.coord_decoder(self.observer(path_params), coords)


# =================================================================================================
# 2. ADVANCED TRAINING TOOLKIT
# =================================================================================================
# =================================================================================================
# INTERACTIVE GUI HELPERS (Place before the ConductorTrainer class)
# =================================================================================================
# In the INTERACTIVE GUI HELPERS section, update InteractivityState

class InteractivityState:
    """A thread-safe class to hold shared state for interactive controls."""
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_prompt_change = 0
        self.sentinel_dampening_log_factor = -1.0
        self.shutdown_event = threading.Event()
        self.toggle_profiler = False # [PROFILING] New state for the profiler

    def get_and_reset_preview_change(self):
        with self.lock:
            change = self.preview_prompt_change
            self.preview_prompt_change = 0
            return change
            
    # [PROFILING] New method to safely get and reset the profiler toggle
    def get_and_reset_profiler_toggle(self):
        with self.lock:
            toggle = self.toggle_profiler
            self.toggle_profiler = False
            return toggle

    def update_sentinel_factor(self, direction):
        with self.lock:
            self.sentinel_dampening_log_factor = np.clip(self.sentinel_dampening_log_factor + direction * 0.5, -3.0, 0.0)

    def get_sentinel_factor(self):
        with self.lock:
            return 10**self.sentinel_dampening_log_factor

    def set_shutdown(self):
        self.shutdown_event.set()
        
# In the INTERACTIVE GUI HELPERS section, update listen_for_keys

def listen_for_keys(shared_state: InteractivityState):
    """A cross-platform, non-blocking key listener that runs in a separate thread."""
    if platform.system() == "Windows":
        while not shared_state.shutdown_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'p': # [PROFILING] Toggle profiler
                    with shared_state.lock: shared_state.toggle_profiler = True
                elif key == b'\xe0': # Arrow key prefix
                    arrow = msvcrt.getch()
                    if arrow == b'K': shared_state.preview_prompt_change = -1 # Left
                    elif arrow == b'M': shared_state.preview_prompt_change = 1 # Right
                    elif arrow == b'H': shared_state.update_sentinel_factor(1) # Up
                    elif arrow == b'P': shared_state.update_sentinel_factor(-1) # Down
            time.sleep(0.05)
    else: # Linux/macOS
        fd = sys.stdin.fileno(); old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not shared_state.shutdown_event.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char == 'p': # [PROFILING] Toggle profiler
                        with shared_state.lock: shared_state.toggle_profiler = True
                    elif char == '\x1b': # ESC sequence
                        next_chars = sys.stdin.read(2)
                        if next_chars == '[A': shared_state.update_sentinel_factor(1)    # Up
                        elif next_chars == '[B': shared_state.update_sentinel_factor(-1) # Down
                        elif next_chars == '[C':                                          # Right
                             with shared_state.lock: shared_state.preview_prompt_change = 1
                        elif next_chars == '[D':                                          # Left
                             with shared_state.lock: shared_state.preview_prompt_change = -1
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
def get_sentinel_lever_ascii(log_factor: float):
    """Generates an ASCII art lever for the UI."""
    levels = np.linspace(-3.0, 0.0, 7); idx = np.digitize(log_factor, levels, right=True)
    lever = ["│         │"] * 7; lever[6-idx] = "│█████████│"
    labels = ["1.0", " ", "0.1", " ", "0.01", " ", "0.001"]
    return "\n".join([f" {labels[i]:<6} {lever[i]}" for i in range(7)])

def sentinel(history_len: int = 5, oscillation_threshold: int = 3) -> optax.GradientTransformation:
    """An Optax component that dampens oscillating gradients."""
    def init_fn(params):
        sign_history = jax.tree_util.tree_map(lambda t: jnp.zeros((history_len,) + t.shape, dtype=jnp.int8), params)
        return SentinelState(sign_history=sign_history, dampened_count=jnp.array(0), dampened_pct=jnp.array(0.0))
    def update_fn(updates, state, params=None, **kwargs):
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

Q_CONTROLLER_CONFIG_NORMAL = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.9, 0.95, 1.0, 1.05, 1.1], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-5, "lr_max": 1e-3, "metric_history_len": 5000, "loss_min": 0.01, "loss_max": 0.5, "exploration_rate_q": 0.3, "min_exploration_rate": 0.05, "exploration_decay": 0.9995, "trend_window": 420, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": 10.0, "stagnation_penalty": -2.0, "warmup_steps": 420, "warmup_lr_start": 1e-6}
Q_CONTROLLER_CONFIG_FINETUNE = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.8, 0.95, 1.0, 1.05, 1.2], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-6, "lr_max": 5e-4, "metric_history_len": 5000, "loss_min": .001, "loss_max": 0.1, "exploration_rate_q": 0.15, "min_exploration_rate": 0.02, "exploration_decay": 0.9998, "warmup_steps": 200, "warmup_lr_start": 1e-7, "trend_window": 420, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": 10.0, "stagnation_penalty": -2.0}

class JaxHakmemQController:
    """A generalized Q-Learning agent to dynamically control any hyperparameter."""
    def __init__(self, initial_value:float, config:Dict[str,Any], param_name: str = "LR"):
        self.param_name = param_name
        self.config = config
        self.initial_value = initial_value
        self.current_value = initial_value
        self.q_table_size = int(self.config["q_table_size"])
        self.num_actions = int(self.config["num_actions"])
        self.action_factors = self.config["action_factors"]
        self.q_table = np.zeros((self.q_table_size, self.num_actions), dtype=np.float32)
        self.learning_rate_q = float(self.config["learning_rate_q"])
        self.discount_factor_q = float(self.config["discount_factor_q"])
        self.value_min = float(self.config["value_min"])
        self.value_max = float(self.config["value_max"])
        self.metric_history = deque(maxlen=int(self.config["metric_history_len"]))
        self.metric_min = float(self.config["metric_min"])
        self.metric_max = float(self.config["metric_max"])
        self.last_action_idx: Optional[int] = None
        self.last_state_idx: Optional[int] = None
        self.initial_exploration_rate = float(self.config["exploration_rate_q"])
        self.exploration_rate_q = self.initial_exploration_rate
        self.min_exploration_rate = float(self.config["min_exploration_rate"])
        self.exploration_decay = float(self.config["exploration_decay"])
        self.status: str = "STARTING"
        self.last_reward: float = 0.0
        self.trend_window = int(config["trend_window"])
        self.trend_history = deque(maxlen=self.trend_window)
        self.improve_threshold = float(config["improve_threshold"])
        self.regress_threshold = float(config["regress_threshold"])
        self.regress_penalty = float(config["regress_penalty"])
        self.stagnation_penalty = float(config["stagnation_penalty"])
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.warmup_start_val = float(config.get("warmup_start_val", 1e-7))
        self._step_count = 0
        print(f"--- Q-Controller ({self.param_name}) initialized. Warmup: {self.warmup_steps} steps. Trend Window: {self.trend_window} steps ---")

    def _discretize_value(self, value: float) -> int:
        if not np.isfinite(value): return self.q_table_size // 2
        if value <= self.metric_min: return 0
        if value >= self.metric_max: return self.q_table_size - 1
        return min(int((value - self.metric_min) / ((self.metric_max - self.metric_min) / self.q_table_size)), self.q_table_size - 1)

    def _get_current_state_idx(self) -> int:
        if not self.metric_history: return self.q_table_size // 2
        return self._discretize_value(np.mean(list(self.metric_history)[-5:]))

    def choose_action(self) -> float:
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            alpha = self._step_count / self.warmup_steps
            self.current_value = self.warmup_start_val * (1 - alpha) + self.initial_value * alpha
            self.status = f"WARMUP ({self.param_name}) {self._step_count}/{self.warmup_steps}"
            return self.current_value
        
        self.last_state_idx = self._get_current_state_idx()
        if np.random.rand() < self.exploration_rate_q:
            self.last_action_idx = np.random.randint(0, self.num_actions)
        else:
            self.last_action_idx = np.argmax(self.q_table[self.last_state_idx]).item()
        
        self.current_value = np.clip(self.current_value * self.action_factors[self.last_action_idx], self.value_min, self.value_max)
        return self.current_value

    def update_q_value(self, metric_value: float):
        self.metric_history.append(metric_value)
        self.trend_history.append(metric_value)
        if self._step_count <= self.warmup_steps: return
        if self.last_state_idx is None or self.last_action_idx is None: return
        
        reward = self._calculate_reward()
        self.last_reward = reward
        current_q = self.q_table[self.last_state_idx, self.last_action_idx]
        next_state_idx = self._get_current_state_idx()
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor_q * np.max(self.q_table[next_state_idx]) - current_q)
        self.q_table[self.last_state_idx, self.last_action_idx] = new_q
        self.exploration_rate_q = max(self.min_exploration_rate, self.exploration_rate_q * self.exploration_decay)

    def _calculate_reward(self):
        if len(self.trend_history) < self.trend_window:
            self.status = f"WARMING UP (TREND) {len(self.trend_history)}/{self.trend_window}"; return 0.0
        
        slope = np.polyfit(np.arange(self.trend_window), np.array(self.trend_history), 1)[0]
        
        if slope < -self.improve_threshold:
            self.status = f"IMPROVING (S={slope:.2e})"; reward = abs(slope) * 1000
        elif slope > self.regress_threshold:
            self.status = f"REGRESSING (S={slope:.2e})"; reward = -abs(slope) * 1000 - self.regress_penalty
        else:
            self.status = f"STAGNATED (S={slope:.2e})"; reward = self.stagnation_penalty
        return reward

    def state_dict(self) -> Dict[str, Any]:
        return {"current_value": self.current_value, "q_table": self.q_table.tolist(), "metric_history": list(self.metric_history), "exploration_rate_q": self.exploration_rate_q, "trend_history": list(self.trend_history), "_step_count": self._step_count}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_value = state_dict.get("current_value", self.current_value)
        self.q_table = np.array(state_dict.get("q_table", self.q_table.tolist()), dtype=np.float32)
        self.metric_history = deque(state_dict.get("metric_history", []), maxlen=self.metric_history.maxlen)
        self.exploration_rate_q = state_dict.get("exploration_rate_q", self.initial_exploration_rate)
        self.trend_history = deque(state_dict.get("trend_history", []), maxlen=self.trend_window)
        self._step_count = state_dict.get("_step_count", 0)

# =================================================================================================
# FINAL, CHECKPOINT-COMPATIBLE VECTOR QUANTIZER
# =================================================================================================
class VectorQuantizer(nn.Module):
    num_codes: int
    code_dim: int
    beta: float = 0.25

    @nn.compact
    def __call__(self, z):
        # --- This is the original, correct way your VQ was defined ---
        # It creates a parameter named 'codebook' directly in the 'vq' scope.
        codebook = self.param(
            'codebook',
            nn.initializers.uniform(),
            (self.code_dim, self.num_codes)
        )

        z_flat = z.reshape(-1, self.code_dim)
        
        # Distances from z to embeddings
        d = jnp.sum(z_flat**2, axis=1, keepdims=True) - 2 * jnp.dot(z_flat, codebook) + jnp.sum(codebook**2, axis=0, keepdims=True)
        indices = jnp.argmin(d, axis=1)
        
        # Quantize using the codebook
        z_q = codebook.T[indices].reshape(z.shape)

        # VQ Losses
        commitment_loss = self.beta * jnp.mean((jax.lax.stop_gradient(z_q) - z)**2)
        codebook_loss = jnp.mean((z_q - jax.lax.stop_gradient(z))**2)
        
        # Straight-through estimator
        z_q_ste = z + jax.lax.stop_gradient(z_q - z)
        
        return {
            "quantized": z_q_ste,
            "indices": indices.reshape(z.shape[:-1]),
            "loss": commitment_loss + codebook_loss
        }

    def lookup(self, indices):
        """A dedicated method to look up embedding vectors from integer indices."""
        # This method now correctly looks for the 'codebook' parameter.
        codebook = self.variables['params']['codebook']
        return codebook.T[indices]

# =================================================================================================
# FINAL, CORRECTED TOKENIZER - No changes needed here now
# =================================================================================================
class LatentTokenizerVQ(nn.Module):
    num_codes: int
    code_dim: int
    latent_grid_size: int
    dtype: Any = jnp.float32

    def setup(self):
        self.enc_conv1 = nn.Conv(128, (3,3), (2,2), 'SAME', name="enc_conv1", dtype=self.dtype)
        self.enc_conv2 = nn.Conv(256, (3,3), (2,2), 'SAME', name="enc_conv2", dtype=self.dtype)
        self.enc_proj = nn.Conv(self.code_dim, (1,1), name="enc_proj", dtype=self.dtype)
        
        # This now uses the checkpoint-compatible VectorQuantizer
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
        h_r = nn.gelu(self.dec_convT1(z_q))
        p_r = self.dec_convT2(h_r)
        return {"reconstructed_path_params": p_r, "indices": vq_out["indices"], "vq_loss": vq_out["loss"]}

    def encode(self, path_params_grid):
        h = nn.gelu(self.enc_conv1(path_params_grid))
        h = nn.gelu(self.enc_conv2(h))
        z_e = self.enc_proj(h)
        return self.vq(z_e)["indices"]

    def decode(self, indices):
        # This correctly calls the new lookup method, which finds the 'codebook' param.
        z_q = self.vq.lookup(indices)
        h_r = nn.gelu(self.dec_convT1(z_q))
        return self.dec_convT2(h_r)
        
# =================================================================================================
# ADVANCED ENTROPIC SAMPLER (DSlider) - Self-Contained Integration
# Source: Entropix, adapted for high-performance image generation.
# =================================================================================================
from dataclasses import dataclass, field, fields

# --- [THE DEFINITIVE SOLUTION] ---
# Explicit PyTree registration correctly separates dynamic JAX arrays from static, hashable Python values.

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DSThreshold:
    # Explicitly define fields in the order they will be unflattened
    bilinear: jnp.ndarray
    linear_state_ent: jnp.ndarray
    linear_state_std: jnp.ndarray
    weight: float
    bias: float
    linear_naked_ent: float
    linear_naked_varent: float

    def tree_flatten(self):
        # Children are JAX arrays (dynamic data)
        children = (self.bilinear, self.linear_state_ent, self.linear_state_std)
        # Aux_data are Python primitives (static data)
        aux_data = (self.weight, self.bias, self.linear_naked_ent, self.linear_naked_varent)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        bilinear, linear_state_ent, linear_state_std = children
        weight, bias, linear_naked_ent, linear_naked_varent = aux_data
        return cls(bilinear, linear_state_ent, linear_state_std, weight, bias, linear_naked_ent, linear_naked_varent)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DSConfig:
    # --- Dynamic Children (JAX arrays or other PyTrees) ---
    dirichlet_support: jnp.ndarray
    outlier_threshold: DSThreshold
    argmax_threshold: DSThreshold
    dirichlet_threshold: DSThreshold
    target_entropy: DSThreshold
    # --- Static Aux Data (Python primitives) ---
    outlier_topk: int
    noise_floor: float
    emwa_ent_naked_coeff: float
    emwa_varent_naked_coeff: float
    emwa_topk_ent_naked_coeff: float
    emwa_temp_coeff: float
    emwa_logp_base: float
    emwa_logp_exp_factor: float
    emwa_dir_ent_coeff: float
    emwa_ent_scaffold_coeff: float
    emwa_varent_scaffold_coeff: float
    token_cross_ent_naked_coeff: float
    token_cross_ent_scaffold_coeff: float
    token_cross_var_naked_coeff: float
    token_cross_var_scaffold_coeff: float
    perturb_base_coeff: float
    perturb_exp_coeff: float

    def tree_flatten(self):
        children = (self.dirichlet_support, self.outlier_threshold, self.argmax_threshold, self.dirichlet_threshold, self.target_entropy)
        aux_data = (self.outlier_topk, self.noise_floor, self.emwa_ent_naked_coeff, self.emwa_varent_naked_coeff, self.emwa_topk_ent_naked_coeff,
                    self.emwa_temp_coeff, self.emwa_logp_base, self.emwa_logp_exp_factor, self.emwa_dir_ent_coeff, self.emwa_ent_scaffold_coeff,
                    self.emwa_varent_scaffold_coeff, self.token_cross_ent_naked_coeff, self.token_cross_ent_scaffold_coeff,
                    self.token_cross_var_naked_coeff, self.token_cross_var_scaffold_coeff, self.perturb_base_coeff, self.perturb_exp_coeff)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (dirichlet_support, outlier_threshold, argmax_threshold, dirichlet_threshold, target_entropy) = children
        (outlier_topk, noise_floor, emwa_ent_naked_coeff, emwa_varent_naked_coeff, emwa_topk_ent_naked_coeff,
         emwa_temp_coeff, emwa_logp_base, emwa_logp_exp_factor, emwa_dir_ent_coeff, emwa_ent_scaffold_coeff,
         emwa_varent_scaffold_coeff, token_cross_ent_naked_coeff, token_cross_ent_scaffold_coeff,
         token_cross_var_naked_coeff, token_cross_var_scaffold_coeff, perturb_base_coeff, perturb_exp_coeff) = aux_data
        return cls(dirichlet_support, outlier_threshold, argmax_threshold, dirichlet_threshold, target_entropy,
                   outlier_topk, noise_floor, emwa_ent_naked_coeff, emwa_varent_naked_coeff, emwa_topk_ent_naked_coeff,
                   emwa_temp_coeff, emwa_logp_base, emwa_logp_exp_factor, emwa_dir_ent_coeff, emwa_ent_scaffold_coeff,
                   emwa_varent_scaffold_coeff, token_cross_ent_naked_coeff, token_cross_ent_scaffold_coeff,
                   token_cross_var_naked_coeff, token_cross_var_scaffold_coeff, perturb_base_coeff, perturb_exp_coeff)


def DEFAULT_DS_CONFIG():
    return DSConfig(
        outlier_topk=16,
        dirichlet_support=jnp.arange(1, 257),
        noise_floor=-18.42068,
        emwa_ent_naked_coeff=0.01,
        emwa_varent_naked_coeff=0.01,
        emwa_topk_ent_naked_coeff=0.01,
        emwa_temp_coeff=0.01,
        emwa_logp_base=2.0,
        emwa_logp_exp_factor=1.0,
        emwa_dir_ent_coeff=0.01,
        emwa_ent_scaffold_coeff=0.01,
        emwa_varent_scaffold_coeff=0.01,
        token_cross_ent_naked_coeff=0.01,
        token_cross_ent_scaffold_coeff=0.01,
        token_cross_var_naked_coeff=0.01,
        token_cross_var_scaffold_coeff=0.01,
        perturb_base_coeff=0.5,
        perturb_exp_coeff=0.1,
        outlier_threshold=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.zeros(4), linear_state_std=jnp.zeros(4), weight=1.0, bias=0.5, linear_naked_ent=0.0, linear_naked_varent=0.0),
        argmax_threshold=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.zeros(4), linear_state_std=jnp.zeros(4), weight=1.0, bias=-0.5, linear_naked_ent=0.0, linear_naked_varent=0.0),
        dirichlet_threshold=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.zeros(4), linear_state_std=jnp.zeros(4), weight=1.0, bias=-0.5, linear_naked_ent=0.0, linear_naked_varent=0.0),
        target_entropy=DSThreshold(bilinear=jnp.zeros((4,4)), linear_state_ent=jnp.array([0., 0., 0., 1.0]), linear_state_std=jnp.zeros(4), weight=0.0, bias=0.5, linear_naked_ent=0.0, linear_naked_varent=0.0)
    )

@dataclass
class SamplerLogicConfig:
  low_naked_entropy_threshold = 0.3
  high_naked_entropy_threshold = 2.5
  low_naked_varentropy_threshold = 1.2
  high_naked_varentropy_threshold = 2.5

# --- Core State and Math Kernels (unchanged) ---
EPS = 1e-8
MIN_TEMP = 0.1
MAX_TEMP = 10.0

class DSState(NamedTuple):
  emwa_dir: jnp.ndarray; emwa_logp_on_supp: jnp.ndarray; emwa_temp: jnp.ndarray
  emwa_ent_scaffold: jnp.ndarray; emwa_ent_naked: jnp.ndarray; emwa_varent_scaffold: jnp.ndarray
  emwa_varent_naked: jnp.ndarray; token_cross_ent_scaffold: jnp.ndarray
  token_cross_ent_naked: jnp.ndarray; token_cross_var_scaffold: jnp.ndarray
  token_cross_var_naked: jnp.ndarray; emwa_dir_ent: jnp.ndarray; emwa_topk_ent_naked: jnp.ndarray

@jax.jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  p = jnp.exp(logp); ent = -jnp.sum(p * logp, axis=-1)
  diff = logp + ent[..., None]; varent = jnp.sum(p * diff**2, axis=-1)
  return ent, varent

@jax.jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float) -> jnp.ndarray:
  shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
  normalized = shifted - jax.nn.logsumexp(shifted + EPS, axis=-1, keepdims=True)
  return jnp.where(normalized < noise_floor, jnp.log(EPS), normalized)

@jax.jit
def dirichlet_log_likelihood_from_logprob(logprobs: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
  return (jnp.sum((alpha - 1.0) * logprobs, axis=-1) - jax.scipy.special.gammaln(jnp.sum(alpha, axis=-1)) + jnp.sum(jax.scipy.special.gammaln(alpha), axis=-1))

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def fit_dirichlet(target_values, init_alpha=None, initial_lr=1.2, decay_alpha=0.1, decay_beta=2.0, decay_gamma=0.25, decay_nu=0.75, max_iters=140, tol=1e-4, dtype: jnp.dtype = jnp.bfloat16):
  batch_shape=target_values.shape[:-1]; n=target_values.shape[-1]; min_lr=1e-8
  target_values=target_values.astype(jnp.float32)
  if init_alpha is None: init_alpha=jnp.ones((*batch_shape, n), dtype=jnp.float32)
  def halley_update(alpha,target_values):
    p1=jax.scipy.special.polygamma(1,alpha);p2=jax.scipy.special.polygamma(2,alpha);S=jnp.sum(alpha,axis=-1,keepdims=True);s1=jax.scipy.special.polygamma(1,S);s2=jax.scipy.special.polygamma(2,S);p1_inv=1./p1;sum_p1_inv=jnp.sum(p1_inv,axis=-1,keepdims=True);denom=jnp.where(jnp.abs(1.-s1*sum_p1_inv)<1e-12,1e-12,1.-s1*sum_p1_inv);coeff=s1/denom;error=jax.scipy.special.digamma(alpha)-jax.scipy.special.digamma(S)-target_values;temp=p1_inv*error;sum_temp=jnp.sum(temp,axis=-1,keepdims=True);J_inv_error=temp+coeff*sum_temp*p1_inv;sum_J_inv_error=jnp.sum(J_inv_error,axis=-1,keepdims=True);H_J_inv_error=p2*J_inv_error-s2*sum_J_inv_error;temp2=p1_inv*H_J_inv_error;sum_temp2=jnp.sum(temp2,axis=-1,keepdims=True);J_inv_H_J_inv_error=temp2+coeff*sum_temp2*p1_inv
    return -J_inv_error+.5*J_inv_H_J_inv_error
  def scan_body(carry, _):
    alpha,converged,error_norm,step=carry;S=jnp.sum(alpha,axis=-1,keepdims=True);error=jax.scipy.special.digamma(alpha)-jax.scipy.special.digamma(S)-target_values;error_norm=jnp.linalg.norm(error,axis=-1);new_converged=converged|(error_norm<tol);lr=jnp.maximum(initial_lr*jnp.exp(-decay_alpha*(step**decay_nu))*jnp.abs(jnp.cos(decay_beta/(step**decay_gamma))),min_lr);delta_alpha=jnp.clip(lr[...,None]*halley_update(alpha,target_values),-.5*alpha,.5*alpha);new_alpha=jnp.where(new_converged[...,None],alpha,jnp.maximum(alpha+delta_alpha,alpha/2))
    return (new_alpha,new_converged,error_norm,step+1),None
  init_state=(init_alpha,jnp.zeros(batch_shape,dtype=jnp.bool_),jnp.full(batch_shape,jnp.inf),jnp.ones(batch_shape,dtype=jnp.int32));(final_alpha,final_converged,_,final_step),_=jax.lax.scan(scan_body,init_state,None,length=max_iters)
  return final_alpha.astype(dtype),final_step-1,final_converged

@partial(jax.jit, static_argnames=("bsz", "dtype"))
def initialize_state(logits: jax.Array, bsz: int, config: DSConfig, dtype=jnp.bfloat16) -> DSState:
    # --- [THE DEFINITIVE FIX] ---
    # The initial logits from the model have a sequence length dimension (L=1),
    # e.g., shape (B, 1, V). This extra dimension corrupts the shape of all
    # initial state metrics (e.g., entropy becomes (B, 1) instead of (B,)).
    # We must squeeze this dimension out *before* any calculations.
    if logits.ndim == 3:
        logits = logits.squeeze(1)
    # Now logits has the correct shape (B, V).

    logprobs = normalize_logits(logits, config.noise_floor)
    ent, varent = ent_varent(logprobs) # ent and varent will now have the correct shape (B,)

    topk_logits, topk_indices = jax.lax.top_k(logprobs, config.outlier_topk)
    topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
    topk_ent, _ = ent_varent(topk_logprobs)
    logprobs_on_supp = normalize_logits(logits[..., config.dirichlet_support], config.noise_floor)
    initial_dir, _, _ = fit_dirichlet(jnp.mean(logprobs_on_supp, axis=0, keepdims=True))
    avg_dir_ent = dirichlet_log_likelihood_from_logprob(logprobs_on_supp, initial_dir).mean()
    topk_token_logprobs = jnp.take_along_axis(logprobs, topk_indices, axis=-1)
    
    # All metrics are now correctly shaped, so the initial state will be correct.
    single_state = DSState(
        emwa_dir=initial_dir, 
        emwa_logp_on_supp=jnp.mean(logprobs_on_supp, axis=0, keepdims=True), 
        emwa_temp=jnp.ones((1,), dtype=dtype), 
        emwa_ent_scaffold=ent, 
        emwa_ent_naked=ent, 
        emwa_varent_scaffold=jnp.zeros((1,), dtype=dtype), 
        emwa_varent_naked=varent, 
        token_cross_ent_scaffold=ent, 
        token_cross_ent_naked=-topk_token_logprobs.mean(), 
        token_cross_var_scaffold=jnp.zeros((1,), dtype=dtype), 
        token_cross_var_naked=topk_token_logprobs.var(), 
        emwa_dir_ent=avg_dir_ent, 
        emwa_topk_ent_naked=topk_ent
    )
    return jax.tree_util.tree_map(lambda x: x.repeat(bsz, axis=0), single_state)

@jax.jit
def update_emwa(new: jax.Array, old: jax.Array, coeff: float | jax.Array) -> jax.Array:
  return coeff * new + (1 - coeff) * old

@jax.jit
def adaptive_dirichlet_step(key: jax.random.PRNGKey, state: DSState, logits: jnp.ndarray, config: DSConfig):
    dtype = logits.dtype; bsz, vsz = logits.shape; output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
    naked_log_probs = normalize_logits(logits, config.noise_floor)
    naked_ent, naked_varent = ent_varent(naked_log_probs)
    new_emwa_ent_naked = update_emwa(naked_ent, state.emwa_ent_naked, config.emwa_ent_naked_coeff)
    new_emwa_varent_naked = update_emwa(naked_varent, state.emwa_varent_naked, config.emwa_varent_naked_coeff)
    topk_logits, topk_indices = jax.lax.top_k(naked_log_probs, config.outlier_topk)
    topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
    naked_topk_ent, _ = ent_varent(topk_logprobs)
    new_emwa_topk_ent_naked = update_emwa(naked_topk_ent, state.emwa_topk_ent_naked, config.emwa_topk_ent_naked_coeff)
    argmax_threshold = config.argmax_threshold.weight * state.emwa_topk_ent_naked + config.argmax_threshold.bias
    argmax_mask = (naked_topk_ent < argmax_threshold)
    argmax_indices = jnp.argmax(topk_logprobs, axis=-1)
    argmax_tokens = jnp.take_along_axis(topk_indices, argmax_indices[:, None], axis=-1).squeeze(1)
    output_tokens = jnp.where(argmax_mask, argmax_tokens, output_tokens)
    inlier_sampling_mask = ~argmax_mask
    inlier_sampling_temp = jnp.ones_like(state.emwa_temp)
    inlier_choices = jax.random.categorical(key, topk_logprobs / inlier_sampling_temp[:, None])
    inlier_tokens = jnp.take_along_axis(topk_indices, inlier_choices[:, None], axis=-1).squeeze(1)
    output_tokens = jnp.where(inlier_sampling_mask, inlier_tokens, output_tokens)
    scaffold_ent, scaffold_varent = naked_ent, naked_varent
    naked_token_logprob = jnp.take_along_axis(naked_log_probs, output_tokens[:,None], axis=-1).squeeze(-1)
    scaffold_token_logprob = naked_token_logprob
    new_state = state._replace(emwa_ent_naked=new_emwa_ent_naked, emwa_varent_naked=new_emwa_varent_naked, emwa_topk_ent_naked=new_emwa_topk_ent_naked)
    return new_state, output_tokens, naked_ent, naked_varent, scaffold_ent, scaffold_varent, naked_token_logprob, scaffold_token_logprob

@jax.jit
def dslider_sampler_step(key: jax.random.PRNGKey, state: DSState, logits: jnp.ndarray, config: DSConfig):
  cfg = SamplerLogicConfig()
  main_key, resample_key = jax.random.split(key)

  # --- Step 1: Propose initial tokens for all items in the batch ---
  (proposed_state, proposed_token, naked_ent, naked_varent, *_) = adaptive_dirichlet_step(main_key, state, logits, config)

  # --- Step 2: Identify which items need resampling (High Entropy High Variance) ---
  is_hehv = (naked_ent > cfg.high_naked_entropy_threshold) & (naked_varent > cfg.high_naked_varentropy_threshold)

  # --- Step 3: *Unconditionally* perform resampling for all items ---
  # This is more JIT-friendly than lax.cond with a non-static predicate.
  # The .at[] operation correctly handles batching by creating a new masked array.
  masked_logits = logits.at[jnp.arange(logits.shape[0]), proposed_token].set(-1e9)
  (resampled_state, resampled_token, *_) = adaptive_dirichlet_step(resample_key, proposed_state, masked_logits, config)

  # --- Step 4: Selectively combine the results using the `is_hehv` mask ---
  # jnp.where correctly handles selecting between the two pre-computed results.
  final_token = jnp.where(is_hehv, resampled_token, proposed_token)

  # For the state, we must also use `where`. We broadcast the 1D `is_hehv` mask 
  # to match the shape of each leaf in the state PyTree.
  final_state = jax.tree_util.tree_map(
      lambda original, resampled: jnp.where(is_hehv.reshape(-1, *([1] * (original.ndim - 1))), resampled, original),
      proposed_state,
      resampled_state
  )

  return final_token, final_state




# Place these new helper functions near your other JAX math kernels.
@partial(jax.jit, static_argnames=('max_lag',))
def calculate_autocorrelation_features(patches: jnp.ndarray, max_lag: int = 8) -> jnp.ndarray:
    """
    Calculates a feature vector based on the spatial autocorrelation of a batch of patches.
    This is sensitive to the spatial structure and texture fabric.
    """
    # We operate on grayscale patches for texture analysis.
    patches_gray = jnp.mean(patches, axis=-1, keepdims=True)
    
    # Center the data by removing the mean (we only care about variance and structure)
    patches_centered = patches_gray - jnp.mean(patches_gray, axis=(1, 2), keepdims=True)
    
    # Calculate the variance as a normalization factor
    norm_factor = jnp.var(patches_centered, axis=(1, 2), keepdims=True) + 1e-6

    # Define a set of lags (offsets) to check.
    # We'll check horizontal, vertical, and both diagonal directions.
    lags_x = jnp.arange(1, max_lag + 1)
    lags_y = jnp.arange(1, max_lag + 1)

    def _calculate_correlation_at_lag(lag_x, lag_y):
        # Roll the patch to create the shifted version
        shifted = jnp.roll(patches_centered, (lag_y, lag_x), axis=(1, 2))
        
        # Calculate the covariance between original and shifted
        covariance = jnp.mean(patches_centered * shifted, axis=(1, 2, 3))
        
        # Normalize to get the correlation coefficient
        return covariance / jnp.squeeze(norm_factor)

    # Vmap for efficiency across different lags.
    # We create a feature vector from correlations at different spatial offsets.
    
    # Horizontal correlations
    corr_h = jax.vmap(_calculate_correlation_at_lag, in_axes=(0, None))(lags_x, 0)
    
    # Vertical correlations
    corr_v = jax.vmap(_calculate_correlation_at_lag, in_axes=(None, 0))(0, lags_y)
    
    # Diagonal correlations
    corr_d = jax.vmap(_calculate_correlation_at_lag, in_axes=(0, 0))(lags_x, lags_y)
    
    # Concatenate all features into a single vector per patch
    # Transpose is needed to get shape (num_patches, num_features)
    return jnp.concatenate([corr_h.T, corr_v.T, corr_d.T], axis=-1)


def _compute_sobel_magnitude(patches: jnp.ndarray, kernel_x: jnp.ndarray, kernel_y: jnp.ndarray) -> jnp.ndarray:
    """A pure, non-nested helper to compute Sobel gradient magnitude for a batch of patches."""
    mag_channels = []
    
    # Pre-shape the kernels for convolution: (H, W, In_Channels, Out_Channels)
    k_x_4d = kernel_x[..., None, None]
    k_y_4d = kernel_y[..., None, None]

    # Explicitly define the tensor layouts for the convolution operation.
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

    for i in range(patches.shape[-1]):
        channel_slice = patches[..., i]
        image_4d = channel_slice[..., None]
        
        # --- [THE FIX] ---
        # Use the full, low-level convolution function that accepts dimension_numbers.
        grad_x = jax.lax.conv_general_dilated(
            image_4d.astype(jnp.float32), 
            k_x_4d, 
            window_strides=(1, 1), 
            padding='SAME', 
            dimension_numbers=dimension_numbers
        )
        grad_y = jax.lax.conv_general_dilated(
            image_4d.astype(jnp.float32), 
            k_y_4d, 
            window_strides=(1, 1), 
            padding='SAME', 
            dimension_numbers=dimension_numbers
        )
        
        magnitude = jnp.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        mag_channels.append(jnp.squeeze(magnitude, axis=-1))
    
    mag_per_channel = jnp.stack(mag_channels, axis=-1)
    return jnp.linalg.norm(mag_per_channel, axis=-1)


@jax.jit
def calculate_edge_loss(patches1: jnp.ndarray, patches2: jnp.ndarray) -> jnp.ndarray:
    """
    [DEFINITIVE FIX v3] Calculates loss based on Sobel gradients.
    This version uses a top-level helper function, completely removing the nested `def`
    that was confusing the JIT compiler.
    """
    # Define Sobel kernels once.
    sobel_x_kernel = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
    sobel_y_kernel = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32)

    # Call the clean, non-nested helper function for each set of patches.
    mag1 = _compute_sobel_magnitude(patches1, sobel_x_kernel, sobel_y_kernel)
    mag2 = _compute_sobel_magnitude(patches2, sobel_x_kernel, sobel_y_kernel)
    
    # Return the final loss.
    return jnp.mean(jnp.abs(mag1 - mag2))



@jax.jit
def calculate_color_covariance_loss(patches1: jnp.ndarray, patches2: jnp.ndarray) -> jnp.ndarray:
    """Calculates loss based on the Gram matrix of color channels to enforce color style."""
    def get_gram_matrix(patches):
        # N, H, W, C -> N, H*W, C
        features = patches.reshape(patches.shape[0], -1, patches.shape[-1])
        # N, C, H*W @ N, H*W, C -> N, C, C
        gram = jax.vmap(lambda x: x.T @ x)(features)
        # Normalize by number of elements
        return gram / (features.shape[1] * features.shape[2])

    gram1 = get_gram_matrix(patches1)
    gram2 = get_gram_matrix(patches2)

    return jnp.mean(jnp.abs(gram1 - gram2))

@jax.jit
def calculate_ssim_loss(patches1: jnp.ndarray, patches2: jnp.ndarray, max_val: float = 2.0) -> jnp.ndarray:
    """Calculates the structural dissimilarity (1 - SSIM) loss."""
    # SSIM constants
    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2
    
    # We operate on grayscale
    patches1_gray = jnp.mean(patches1, axis=-1)
    patches2_gray = jnp.mean(patches2, axis=-1)

    mu1 = jnp.mean(patches1_gray, axis=(1, 2))
    mu2 = jnp.mean(patches2_gray, axis=(1, 2))
    
    var1 = jnp.var(patches1_gray, axis=(1, 2))
    var2 = jnp.var(patches2_gray, axis=(1, 2))
    
    # Reshape for broadcasting
    mu1_mu2 = mu1 * mu2
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    
    # Covariance
    covar = jnp.mean(patches1_gray * patches2_gray, axis=(1,2)) - mu1_mu2
    
    # SSIM formula components
    numerator = (2 * mu1_mu2 + C1) * (2 * covar + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (var1 + var2 + C2)
    
    ssim = numerator / denominator
    
    # We want to minimize dissimilarity
    return jnp.mean(1.0 - ssim)
@partial(jax.jit, static_argnames=('num_moments',))
def calculate_moments(patches, num_moments=4):
    flat_patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
    mean = jnp.mean(flat_patches, axis=1)
    var = jnp.var(flat_patches, axis=1)
    if num_moments <= 2: return jnp.concatenate([mean, var], axis=-1)
    centered = flat_patches - mean[:, None, :]; std_dev = jnp.sqrt(var + 1e-6)
    skew = jnp.mean((centered / std_dev[:, None, :])**3, axis=1)
    if num_moments <= 3: return jnp.concatenate([mean, var, skew], axis=-1)
    kurt = jnp.mean((centered / std_dev[:, None, :])**4, axis=1)
    return jnp.concatenate([mean, var, skew, kurt], axis=-1)

@jax.jit
def fft_magnitude_log(patches):
    def fft_on_slice(patch_2d): return jnp.log(jnp.abs(jnp.fft.fft2(patch_2d)) + 1e-6)
    return jax.vmap(jax.vmap(fft_on_slice, in_axes=-1, out_axes=-1))(patches)

@partial(jax.vmap, in_axes=(0, 0, 0, None, None), out_axes=0)
def _extract_patches_vmapped(image, x_coords, y_coords, patch_size, c):
    """A static, vmapped helper to extract patches from a single image."""
    def get_patch(x, y):
         return jax.lax.dynamic_slice(image, (y, x, 0), (patch_size, patch_size, c))
    return jax.vmap(get_patch)(x_coords, y_coords)

# REPLACE your JAXPerceptualLoss class with this new, upgraded version.

class JAXMultiMetricPerceptualLoss:
    """
    A stateless, physics-informed diagnostic suite for generative models.
    It returns a dictionary of unweighted loss components for external PID control.
    """
    def __init__(self, num_patches=64, patch_size=32):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self._calculate_losses_jit = partial(jax.jit, static_argnames=('batch_size',))(self._calculate_losses)

    def _calculate_losses(self, img1, img2, key, batch_size: int) -> Dict[str, jnp.ndarray]:
        """The core, JIT-compatible loss calculation logic."""
        _, h, w, c = img1.shape
        x_coords = jax.random.randint(key, (batch_size, self.num_patches), 0, w - self.patch_size)
        y_coords = jax.random.randint(key, (batch_size, self.num_patches), 0, h - self.patch_size)

        patches1 = _extract_patches_vmapped(img1, x_coords, y_coords, self.patch_size, c)
        patches2 = _extract_patches_vmapped(img2, x_coords, y_coords, self.patch_size, c)

        patches1 = patches1.reshape(-1, self.patch_size, self.patch_size, c)
        patches2 = patches2.reshape(-1, self.patch_size, self.patch_size, c)
        
        # --- Run all diagnostics across multiple scales ---
        scales = [1.0, 0.5] # Run on full and half-res patches
        all_losses = {
            'moment': [], 'fft': [], 'autocorr': [],
            'edge': [], 'color_cov': [], 'ssim': []
        }

        for scale in scales:
            new_size = int(self.patch_size * scale)
            if new_size < 16: continue

            p1 = jax.image.resize(patches1, (patches1.shape[0], new_size, new_size, c), 'bilinear')
            p2 = jax.image.resize(patches2, (patches2.shape[0], new_size, new_size, c), 'bilinear')

            # --- Append measurements from each diagnostic to the list ---
            all_losses['moment'].append(jnp.mean(jnp.abs(calculate_moments(p1) - calculate_moments(p2))))
            all_losses['fft'].append(jnp.mean(jnp.abs(fft_magnitude_log(jnp.mean(p1, axis=-1, keepdims=True)) - fft_magnitude_log(jnp.mean(p2, axis=-1, keepdims=True)))))
            all_losses['autocorr'].append(jnp.mean(jnp.abs(calculate_autocorrelation_features(p1) - calculate_autocorrelation_features(p2))))
            all_losses['edge'].append(calculate_edge_loss(p1, p2))
            all_losses['color_cov'].append(calculate_color_covariance_loss(p1, p2))
            all_losses['ssim'].append(calculate_ssim_loss(p1, p2))

        # --- Average the results across scales for a final report ---
        final_losses = {k: jnp.mean(jnp.array(v)) for k, v in all_losses.items() if v}
        return final_losses

    def __call__(self, img1, img2, key):
        if img1.ndim != 4 or img2.ndim != 4:
            raise ValueError(f"Inputs must be 4D tensors, got {img1.shape} and {img2.shape}")
        return self._calculate_losses_jit(img1, img2, key, batch_size=img1.shape[0])      
        

# Place this with your other mathematical/utility functions

def _rbf_kernel(x: jnp.ndarray, y: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """Computes the RBF (Gaussian) kernel between two sets of vectors."""
    # x: (N, D), y: (M, D)
    # The squared Euclidean distance between all pairs of vectors
    dist_sq = jnp.sum(x**2, 1).reshape(-1, 1) + jnp.sum(y**2, 1) - 2 * jnp.dot(x, y.T)
    # The kernel matrix
    return jnp.exp(-dist_sq / (2 * sigma**2))

def calculate_mmd_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the MMD between two batches of samples.
    [DEFINITIVE FIX v2] This version is robust to any batch size, including 1.
    It uses the universally compatible `jnp.where` function for safe division,
    avoiding both `TypeError` on older JAX versions and `NaN` values.
    
    Args:
        x: Samples from the first distribution (e.g., real data), shape (B, L, D).
        y: Samples from the second distribution (e.g., generated data), shape (B, L, D).
    """
    # Reshape from (B, L, D) to (B, L*D) to treat each sequence as a single vector
    B, L, D = x.shape
    x_flat = x.reshape(B, L * D)
    y_flat = y.reshape(B, L * D)

    # The three components of the MMD^2 formula
    k_xx = _rbf_kernel(x_flat, x_flat)
    k_yy = _rbf_kernel(y_flat, y_flat)
    k_xy = _rbf_kernel(x_flat, y_flat)
    
    # Create a mask to identify the diagonal elements.
    mask = jnp.eye(B, dtype=jnp.bool_)
    
    # --- Robust Mean Calculation for Off-Diagonal Elements ---
    # 1. Use jnp.where to create matrices with zeros on the diagonal.
    off_diag_k_xx = jnp.where(mask, 0., k_xx)
    off_diag_k_yy = jnp.where(mask, 0., k_yy)

    # 2. Sum these matrices.
    sum_off_diag_xx = jnp.sum(off_diag_k_xx)
    sum_off_diag_yy = jnp.sum(off_diag_k_yy)

    # 3. Calculate the number of off-diagonal elements: B * (B - 1).
    num_off_diag = B * (B - 1)

    # 4. Perform safe division using jnp.where.
    # If num_off_diag > 0, calculate sum / num_off_diag. Otherwise, return 0.0.
    mean_xx = jnp.where(num_off_diag > 0, sum_off_diag_xx / num_off_diag, 0.0)
    mean_yy = jnp.where(num_off_diag > 0, sum_off_diag_yy / num_off_diag, 0.0)
    
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    # The mean of k_xy is always safe as it's not excluding any elements.
    loss = mean_xx + mean_yy - 2 * jnp.mean(k_xy)

    return loss


@jax.jit
def gumbel_softmax_straight_through(logits, key, tau=1.0):
    """
    Gumbel-Softmax Straight-Through Estimator.
    Returns the STE tensor for differentiable backprop and the hard integer tokens.
    """
    gumbel_noise = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
    y_soft = jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)
    
    y_hard = jnp.argmax(y_soft, axis=-1)
    y_hard_one_hot = jax.nn.one_hot(y_hard, logits.shape[-1], dtype=logits.dtype)
    
    y_ste = jax.lax.stop_gradient(y_hard_one_hot - y_soft) + y_soft
    
    return y_ste, y_hard # Return both
    
    
@partial(jax.jit, static_argnames=('target_size',))
def encode_image_for_clip(image_batch, target_size=224):
    """
    Prepares a generated image batch for CLIP by resizing and normalizing.
    Input shape: (B, H, W, C), values in [-1, 1]
    """
    # JAX's resize expects (B, H, W, C)
    resized_images = jax.image.resize(image_batch, (image_batch.shape[0], target_size, target_size, 3), 'bilinear')
    
    # Denormalize from [-1, 1] to [0, 1]
    renormalized = (resized_images * 0.5) + 0.5
    
    # Normalize with CLIP's specific mean/std
    clip_mean = jnp.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 1, 3)
    clip_std = jnp.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 1, 3)
    
    final_images = (renormalized - clip_mean) / clip_std
    return final_images



def calculate_clip_similarity_loss(seq_embeddings: jnp.ndarray, text_embeddings: jnp.ndarray, temperature: float = 0.07) -> jnp.ndarray:
    """
    Computes a contrastive loss to maximize similarity between sequence embeddings
    and corresponding text embeddings.
    Args:
        seq_embeddings: L2-normalized embeddings from the generated sequence (B, D).
        text_embeddings: L2-normalized embeddings from the text prompt (B, D).
        temperature: The temperature parameter for the softmax.
    """
    # Calculate the cosine similarity matrix between all pairs in the batch
    logits = jnp.einsum('bd,cd->bc', seq_embeddings, text_embeddings) / temperature
    
    # The labels are the diagonal elements, as seq_embeddings[i] should match text_embeddings[i]
    labels = jnp.arange(seq_embeddings.shape[0])
    
    # We compute the loss in both directions (image-to-text and text-to-image) and average them
    loss_i = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    loss_t = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
    
    return (loss_i + loss_t) / 2.0
    """
    Computes a contrastive loss to maximize similarity between sequence embeddings
    and corresponding text embeddings.
    """
    @nn.compact
    def __call__(self, seq_embeddings: jnp.ndarray, text_embeddings: jnp.ndarray, temperature: float = 0.07) -> jnp.ndarray:
        """
        Args:
            seq_embeddings: L2-normalized embeddings from the generated sequence (B, D).
            text_embeddings: L2-normalized embeddings from the text prompt (B, D).
            temperature: The temperature parameter for the softmax.
        """
        # Calculate the cosine similarity matrix between all pairs in the batch
        logits = jnp.einsum('bd,cd->bc', seq_embeddings, text_embeddings) / temperature
        
        # The labels are the diagonal elements, as seq_embeddings[i] should match text_embeddings[i]
        labels = jnp.arange(seq_embeddings.shape[0])
        
        # We compute the loss in both directions (image-to-text and text-to-image) and average them
        loss_i = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        loss_t = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
        
        return (loss_i + loss_t) / 2.0       
# --- NEW VQ-GAN MODEL DEFINITIONS ---
class PatchDiscriminator(nn.Module):
    num_filters: int = 64; num_layers: int = 3; dtype: Any = jnp.float32
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

# Rename to reflect its new GAN-based training
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
        target_size = self.latent_grid_size // 4
        h = nn.gelu(self.enc_conv1(path_params_grid)); h = nn.gelu(self.enc_conv2(h))
        z_e = self.enc_proj(h)
        assert z_e.shape[1] == target_size and z_e.shape[2] == target_size, f"Incorrect spatial dim: {z_e.shape}"
        vq_out = self.vq(z_e); z_q = vq_out["quantized"]
        p_r = self.dec_convT2(nn.gelu(self.dec_convT1(z_q)))
        # [THE CHANGE] Also return the pre-quantization latents for the "stink field" loss
        return {"reconstructed_path_params": p_r, "indices": vq_out["indices"], "vq_loss": vq_out["loss"], "pre_quant_latents": z_e}

    def encode(self, path_params_grid):
        h = nn.gelu(self.enc_conv1(path_params_grid)); h = nn.gelu(self.enc_conv2(h))
        z_e = self.enc_proj(h); return self.vq(z_e)["indices"]

    def decode(self, indices):
        z_q = self.vq.lookup(indices); h_r = nn.gelu(self.dec_convT1(z_q)); return self.dec_convT2(h_r)

# --- State holder for GAN training ---
class GANTrainStates(NamedTuple):
    generator: TrainState
    discriminator: TrainState


   


# ==============================================================================
# FINAL, CORRECTED ATTENTION: Using pre-allocated KV cache for jax.lax.scan
# ==============================================================================


class StandardAttention(nn.Module):
    """
    [DEBUGGING VERSION] A multi-head attention module with ALL remat/checkpointing
    temporarily disabled to isolate tracer errors.
    """
    num_heads: int
    dtype: Any = jnp.float32
    num_positions: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None, decode: bool = False, index: Optional[jnp.ndarray] = None):
        is_self_attn = context is None
        
        d_model = x.shape[-1]
        head_dim = d_model // self.num_heads
        
        q_proj = nn.Dense(d_model, name="query", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        k_proj = nn.Dense(d_model, name="key", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        v_proj = nn.Dense(d_model, name="value", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        out_proj = nn.Dense(d_model, name="out", dtype=self.dtype, kernel_init=initializers.xavier_uniform())

        kv_source = x if is_self_attn else context
        q = q_proj(x)
        
        if decode:
            if not is_self_attn:
                k = k_proj(kv_source)
                v = v_proj(kv_source)
            else:
                if self.num_positions is None: raise ValueError("num_positions must be provided for self-attention.")
                cache_k = self.variable('cache', 'cached_key', lambda: jnp.zeros((x.shape[0], self.num_positions, d_model), dtype=self.dtype))
                cache_v = self.variable('cache', 'cached_value', lambda: jnp.zeros((x.shape[0], self.num_positions, d_model), dtype=self.dtype))
                k_new = k_proj(kv_source)
                v_new = v_proj(kv_source)
                valid_index = index if index is not None else 0
                k_update_slice = k_new[:, 0, :]
                v_update_slice = v_new[:, 0, :]
                k = cache_k.value.at[:, valid_index, :].set(k_update_slice)
                v = cache_v.value.at[:, valid_index, :].set(v_update_slice)
                if self.is_mutable_collection('cache'):
                    cache_k.value = k
                    cache_v.value = v
        else:
            k = k_proj(kv_source)
            v = v_proj(kv_source)

        B, L_q, _ = q.shape
        L_kv = k.shape[1]
        
        q_heads = q.reshape(B, L_q, self.num_heads, head_dim)
        k_heads = k.reshape(B, L_kv, self.num_heads, head_dim)
        v_heads = v.reshape(B, L_kv, self.num_heads, head_dim)
        
        # [THE DEFINITIVE FIX]
        # Direct, un-wrapped call. No remat.
        attn_output = dot_product_attention(q_heads, k_heads, v_heads, mask=mask, dtype=self.dtype)
        
        return out_proj(attn_output.reshape(B, L_q, d_model))


class TransformerBlock(nn.Module):
    """
    [FINAL, GRAFF-INFORMED] Transformer block. The MLP layers now use a
    symmetric weight matrix, allowing the block's dynamics to be interpreted
    as a gradient flow of a learnable energy function, learning attractive
    (smoothing) and repulsive (sharpening) forces as per the GRAFF paper.
    """
    num_heads: int; d_model: int; num_positions: int; dtype: Any = jnp.float32

    def setup(self):
        self.ln1 = nn.LayerNorm(dtype=self.dtype)
        self.sa = StandardAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            num_positions=self.num_positions,
            name='sa'
        )
        self.ln2 = nn.LayerNorm(dtype=self.dtype)
        self.ca = StandardAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            num_positions=self.num_positions,
            name='ca'
        )
        self.ln3 = nn.LayerNorm(dtype=self.dtype)
        
        # [THE GRAFF FIX]
        # Instead of a standard Dense layer, we create the parameters for the
        # MLP and enforce symmetry on the kernel.
        self.mlp_w1_raw = self.param('mlp_w1_raw', initializers.xavier_uniform(), (self.d_model, self.d_model * 4), self.dtype)
        self.mlp_b1 = self.param('mlp_b1', initializers.zeros, (self.d_model * 4,), self.dtype)
        
        self.mlp_w2_raw = self.param('mlp_w2_raw', initializers.xavier_uniform(), (self.d_model * 4, self.d_model), self.dtype)
        self.mlp_b2 = self.param('mlp_b2', initializers.zeros, (self.d_model,), self.dtype)

    def __call__(self, x, context, mask, decode: bool = False):
        sa_input = self.ln1(x)
        x = x + self.sa(sa_input, mask=mask, decode=decode, index=None)

        ca_input = self.ln2(x)
        x = x + self.ca(ca_input, context=context, mask=None, decode=decode, index=None)

        ffn_input = self.ln3(x)
        
        # Enforce symmetry on the weight matrices to align with Gradient Flow formalism
        w1 = (self.mlp_w1_raw + self.mlp_w1_raw.T) / 2 if self.d_model == self.d_model * 4 else self.mlp_w1_raw
        w2 = (self.mlp_w2_raw + self.mlp_w2_raw.T) / 2 if self.d_model * 4 == self.d_model else self.mlp_w2_raw
        
        h = nn.gelu(ffn_input @ w1 + self.mlp_b1)
        h = h @ w2 + self.mlp_b2
        
        x = x + h
        return x
        
        
class GenerativeConductor(nn.Module):
    # Core architectural params
    num_codes: int; d_model: int; num_heads: int; num_layers: int
    
    # Spacetime grid and patch dimensions
    time_size: int = 4
    depth_size: int = 4
    height_size: int = 24
    width_size: int = 24
    
    time_patch_size: int = 2
    depth_patch_size: int = 2
    height_patch_size: int = 4
    width_patch_size: int = 4
    
    # Other params
    clip_dim: int = 512
    uncond_drop_rate: float = 0.1
    mask_ratio: float = 0.5
    dtype: Any = jnp.float32

    @property
    def MASK_TOKEN_ID(self):
        return self.num_codes + 1

    @property
    def vocab_size(self):
        return self.num_codes + 2

    def setup(self):
        self.grid_t = self.time_size // self.time_patch_size
        self.grid_z = self.depth_size // self.depth_patch_size
        self.grid_y = self.height_size // self.height_patch_size
        self.grid_x = self.width_size // self.width_patch_size
        self.num_patches = self.grid_t * self.grid_z * self.grid_y * self.grid_x
        self.patch_dim = self.time_patch_size * self.depth_patch_size * self.height_patch_size * self.width_patch_size

        self.uncond_embedding = self.param('uncond_embedding', nn.initializers.normal(0.02), (1, self.clip_dim), self.dtype)
        self.token_embedding = nn.Embed(self.vocab_size, self.d_model, name='token_embedding', dtype=self.dtype)
        self.patch_projection = nn.Dense(self.d_model, name='patch_projection', dtype=self.dtype)
        self.pos_embed_t = self.param('pos_embed_t', nn.initializers.normal(0.02), (self.grid_t, self.d_model), self.dtype)
        self.pos_embed_z = self.param('pos_embed_z', nn.initializers.normal(0.02), (self.grid_z, self.d_model), self.dtype)
        self.pos_embed_y = self.param('pos_embed_y', nn.initializers.normal(0.02), (self.grid_y, self.d_model), self.dtype)
        self.pos_embed_x = self.param('pos_embed_x', nn.initializers.normal(0.02), (self.grid_x, self.d_model), self.dtype)
        self.text_projection = nn.Dense(self.d_model, name='text_projection', dtype=self.dtype)
        self.norm = nn.LayerNorm(dtype=self.dtype)
        self.output_head = nn.Dense(self.patch_dim * self.vocab_size, name='output_head', dtype=self.dtype)
        self.mmd_projection = nn.Dense(self.clip_dim, name='mmd_projection', dtype=self.dtype)
        self.blocks = [
            TransformerBlock(self.num_heads, self.d_model, self.num_patches, self.dtype, name=f'block_{i}')
            for i in range(self.num_layers)
        ]

    def _prepare_inputs(self, tokens_4d, text_emb):
        x = tokens_4d.reshape(
            -1,
            self.grid_t, self.time_patch_size,
            self.grid_z, self.depth_patch_size,
            self.grid_y, self.height_patch_size,
            self.grid_x, self.width_patch_size,
            *tokens_4d.shape[5:]
        )
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6, 8, *range(9, tokens_4d.ndim + 4))
        
        if tokens_4d.dtype == jnp.int32 or tokens_4d.dtype == jnp.int64:
            x = x.reshape(-1, self.num_patches, self.patch_dim)
            x_emb = self.token_embedding(x)
        else:
            x = x.reshape(-1, self.num_patches, self.patch_dim, self.vocab_size)
            x_emb = x @ self.token_embedding.embedding
    
        x = x_emb.reshape(-1, self.num_patches, self.patch_dim * self.d_model)
        x = self.patch_projection(x) 
    
        pos_embed_4d = (
            self.pos_embed_t[:, None, None, None, :] +
            self.pos_embed_z[None, :, None, None, :] +
            self.pos_embed_y[None, None, :, None, :] +
            self.pos_embed_x[None, None, None, :, :]
        ).reshape(self.num_patches, self.d_model)
        
        x = x + pos_embed_4d[None, :, :]
        
        ctx = self.text_projection(text_emb)[:, None, :]
        return x, ctx
        
    def __call__(self, target_tokens_4d, text_emb, train: bool = True):
        key = self.make_rng('dropout')

        if target_tokens_4d.dtype == jnp.int32:
            rand = jax.random.uniform(key, target_tokens_4d.shape)
            num_to_mask = int(self.mask_ratio * target_tokens_4d.size / target_tokens_4d.shape[0])
            rand_flat = rand.reshape(rand.shape[0], -1)
            should_mask_flat = jnp.argsort(rand_flat, axis=-1) < num_to_mask
            should_mask = should_mask_flat.reshape(target_tokens_4d.shape)
            input_tokens = jnp.where(should_mask, self.MASK_TOKEN_ID, target_tokens_4d)
        else:
            input_tokens = target_tokens_4d
            should_mask = jnp.zeros(target_tokens_4d.shape[:5], dtype=bool) 
        
        x, ctx = self._prepare_inputs(input_tokens, text_emb)

        for block in self.blocks:
            x = block(x, ctx, mask=None, decode=not train)
        
        x = self.norm(x)
        x = self.output_head(x)

        logits_4d = x.reshape(
            -1, self.grid_t, self.grid_z, self.grid_y, self.grid_x,
            self.time_patch_size, self.depth_patch_size, self.height_patch_size, self.width_patch_size,
            self.vocab_size
        )
        logits_4d = logits_4d.transpose(0, 1, 5, 2, 6, 3, 7, 4, 8, 9)
        final_logits = logits_4d.reshape(-1, self.time_size, self.depth_size, self.height_size, self.width_size, self.vocab_size)

        if target_tokens_4d.dtype == jnp.int32:
            all_losses = optax.softmax_cross_entropy(
                jax.nn.one_hot(target_tokens_4d, self.vocab_size), 
                final_logits
            )
            masked_losses = jnp.where(should_mask, all_losses, 0)
            loss = masked_losses.sum() / jnp.maximum(should_mask.sum(), 1)
        else:
            loss = jnp.array(0.0, dtype=self.dtype)

        return loss, final_logits
        
    def get_features(self, tokens_4d, text_emb, train: bool = True):
        # [THE FIX] `get_features` needs its own dropout key, so we must add `make_rng`.
        # This was a latent bug.
        _ = self.make_rng('dropout') # This is just to ensure the stream is requested.

        x, ctx = self._prepare_inputs(tokens_4d, text_emb)
    
        for block in self.blocks:
            x = block(x, ctx, mask=None, decode=not train)
            
        x_norm = self.norm(x)
        
        projected_patch_embeddings = self.mmd_projection(x_norm)
    
        x_out = self.output_head(x_norm)
    
        logits_4d = x_out.reshape(
            -1, self.grid_t, self.grid_z, self.grid_y, self.grid_x,
            self.time_patch_size, self.depth_patch_size, self.height_patch_size, self.width_patch_size,
            self.vocab_size
        )
        logits_4d = logits_4d.transpose(0, 1, 5, 2, 6, 3, 7, 4, 8, 9)
        final_logits = logits_4d.reshape(-1, self.time_size, self.depth_size, self.height_size, self.width_size, self.vocab_size)
    
        return final_logits, projected_patch_embeddings

    # [THE FIX] A new method dedicated to initialization to trace all code paths.
    def init_forward_pass(self, tokens_4d, text_emb, train: bool = True):
        # Call the default __call__ method
        loss, _ = self.__call__(tokens_4d, text_emb, train=train)
        
        # Also call the get_features method to ensure its layers are initialized
        _, _ = self.get_features(tokens_4d, text_emb, train=train)
        
        # Return a dummy value, as only the side-effect of parameter creation matters
        return loss




# =================================================================================================
# 4. DATA PREPARATION (Corrected for Robust Pairing)
# =================================================================================================

def create_raw_dataset(data_dir: str):
    """
    [ROBUST VERSION] Creates a perfectly aligned (image, text) dataset.
    
    This function avoids the 'zip' misalignment issue by using the definitive list of
    image paths that were used to create the TFRecord (saved in dataset_info.pkl).
    It derives the corresponding text file paths from this list, guaranteeing
    that each image from the TFRecord is paired with its correct caption.
    """
    console = Console()
    data_p = Path(data_dir)
    record_file = data_p / "data_512x512.tfrecord"
    info_file = data_p / "dataset_info.pkl"

    if not record_file.exists() or not info_file.exists():
        sys.exit(f"[FATAL] TFRecord file ({record_file}) or info file ({info_file}) not found.")

    console.print("--- Loading dataset info for robust pairing... ---", style="yellow")
    with open(info_file, 'rb') as f:
        info = pickle.load(f)
    
    image_paths_from_source = info.get('image_paths')
    if not image_paths_from_source:
        sys.exit("[FATAL] 'image_paths' key not found in dataset_info.pkl. Please re-run the `prepare-data` command from your Phase 1/2 script after updating it to save the path list.")

    # Derive text file paths from the definitive image path list
    text_paths = []
    for img_path_str in image_paths_from_source:
        p = Path(img_path_str)
        text_p = p.with_suffix('.txt')
        if text_p.exists():
            text_paths.append(str(text_p))
        else:
            # This case is unlikely if you just created the dataset, but handles it.
            console.print(f"[bold yellow]Warning:[/bold yellow] Missing text file for image: {p.name}", style="yellow")
            # We add a placeholder to keep the lists aligned, which will be filtered out later.
            text_paths.append("DUMMY_PATH_FOR_MISSING_TEXT")

    # Create the two datasets that are now guaranteed to be aligned
    image_ds = tf.data.TFRecordDataset(str(record_file))
    text_files_ds = tf.data.Dataset.from_tensor_slices(text_paths)

    # Zip them together. Now the alignment is correct.
    paired_ds = tf.data.Dataset.zip((image_ds, text_files_ds))

    # Filter out any pairs where the text file was missing
    paired_ds = paired_ds.filter(lambda img_proto, text_path: text_path != "DUMMY_PATH_FOR_MISSING_TEXT")

    num_images_in_record = info.get('num_samples')
    num_texts_found = len([p for p in text_paths if p != "DUMMY_PATH_FOR_MISSING_TEXT"])

    console.print(f"Found {num_images_in_record} images in TFRecord and {num_texts_found} corresponding .txt files.")
    if num_images_in_record != num_texts_found:
        console.print("[bold yellow]Note:[/bold yellow] The number of images and found texts differ slightly. This is expected if some text files were missing.")

    # The rest of the parsing logic remains the same
    def _parse(img_proto, text_path):
        features = {'image': tf.io.FixedLenFeature([], tf.string)}
        parsed_features = tf.io.parse_single_example(img_proto, features)
        img = tf.io.decode_jpeg(parsed_features['image'], 3)
        img = (tf.cast(img, tf.float32) / 127.5) - 1.0
        text = tf.io.read_file(text_path)
        return img, text

    return paired_ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

def prepare_paired_data(args):
    console = Console()
    console.print("--- 🧠 STEP 0: Preparing Paired (Latent, Text Embedding) Dataset ---", style="bold yellow")
    output_path = Path(args.data_dir) / f"paired_data_{args.basename}.pkl"
    if output_path.exists():
        console.print(f"✅ Paired data already exists at [green]{output_path}[/green]. Skipping preparation.")
        return

    p1_path = Path(f"{args.basename}_{args.d_model}d_512.pkl")
    if not p1_path.exists(): sys.exit(f"[FATAL] Phase 1 model not found: {p1_path}")
    with open(p1_path, 'rb') as f: p1_checkpoint = pickle.load(f)
    p1_params = p1_checkpoint['params']

    p1_encoder = PathModulator(args.latent_grid_size, 512, jnp.float32)
    p1_encoder_fn = jax.jit(lambda i: p1_encoder.apply({'params': p1_params['modulator']}, i))

    clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
    
    # Use the new robust dataset creation function
    raw_ds = create_raw_dataset(args.data_dir).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    all_latents, all_embeddings = [], []
    
    console.print("Processing raw data into (latent, embedding) pairs...")
    for img_batch, text_batch in tqdm(raw_ds.as_numpy_iterator(), desc="Preprocessing"):
        latents_np = np.asarray(p1_encoder_fn(img_batch))
        all_latents.append(latents_np)
        text_list = [t.decode('utf-8').strip() for t in text_batch]
        tokens = clip.tokenize(text_list, truncate=True).to(_clip_device)
        with torch.no_grad(): text_features = clip_model.encode_text(tokens)
        all_embeddings.append(text_features.cpu().numpy())

    final_latents = np.concatenate(all_latents, axis=0)
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    
    console.print(f"Processed {len(final_latents)} pairs. Saving to [green]{output_path}[/green]...")
    with open(output_path, 'wb') as f:
        pickle.dump({'latents': final_latents, 'embeddings': final_embeddings}, f)
    console.print("✅ Preparation complete.")
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

    p1_encoder = PathModulator(args.latent_grid_size, 512, jnp.float32) # Use float32 for high quality preprocessing
    p1_encoder_fn = jax.jit(lambda i: p1_encoder.apply({'params': p1_params['modulator']}, i))

    # We only need the images for this step
    raw_ds = create_raw_dataset(args.data_dir).map(lambda img, txt: img).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    all_latents = []
    
    console.print("Processing raw images into latents for the tokenizer...")
    # Get total number of samples for a better progress bar
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
# 5. PHASE 3 TRAINERS (UPGRADED)
# =================================================================================================

class AdvancedTrainer:
    """Base class for training with advanced toolkit features."""
    def __init__(self, args):
        self.args = args
        self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.num_devices = jax.local_device_count()
        self.loss_history = deque(maxlen=200)

        # --- [FIX] Robust, Backward-Compatible Agent Initialization ---
        # Use getattr() to safely check for the existence of the arguments.
        # If 'use_q_controller' doesn't exist on args (like in Chimera), default to False.
        # This check is now only relevant for older trainers like TokenizerTrainer.
        if getattr(self.args, 'use_q_controller', False):
            # Also safely check for 'finetune', defaulting to False.
            is_finetune = getattr(self.args, 'finetune', False)
            q_config = Q_CONTROLLER_CONFIG_FINETUNE if is_finetune else Q_CONTROLLER_CONFIG_NORMAL
            self.q_controller = JaxHakmemQController(initial_lr=self.args.lr, config=q_config)
        else:
            self.q_controller = None

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




# In TokenizerTrainer

class PIDLambdaController:
    """
    An intelligent agent for the Structuralist clone.
    Dynamically balances the weights between next-token prediction (Cross-Entropy)
    and global distributional coherence (MMD).
    """
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]]):
        self.targets = targets
        self.base_weights = base_weights
        self.gains = gains
        # Initialize state for only the keys this controller cares about: ce_loss and mmd_loss
        self.integral_error = {k: 0.0 for k in targets.keys()}
        self.last_error = {k: 0.0 for k in targets.keys()}
        self.derivative = {k: 0.0 for k in targets.keys()}

    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Takes the last step's metrics and returns a dictionary of new lambda weights.
        Example last_metrics: {'Structuralist/CE_Loss': 0.9, 'Structuralist/MMD_Loss': 0.02}
        """
        final_lambdas = self.base_weights.copy()
        
        # --- Dynamically calculate lambdas for CE and MMD ---
        for name, target in self.targets.items():
            # The metric from the UI will have a prefix like "Structuralist/"
            metric_key = next((k for k in last_metrics if k.endswith(name)), None)
            if metric_key is None: continue # Skip if the metric isn't available yet

            kp, ki, kd = self.gains[name]
            current_loss = last_metrics.get(metric_key, target)
            error = current_loss - target
            
            self.integral_error[name] += error
            self.integral_error[name] = np.clip(self.integral_error[name], -5.0, 5.0) # Anti-windup
            self.derivative[name] = error - self.last_error[name]
            
            adjustment = (kp * error) + (ki * self.integral_error[name]) + (kd * self.derivative[name])
            multiplier = np.exp(adjustment)
            
            calculated_lambda = self.base_weights[name] * multiplier
            self.last_error[name] = error
            
            # Use a general, stable clipping range for these lambdas
            final_lambdas[name] = np.clip(calculated_lambda, 0.2, 5.0)

        return final_lambdas

    def state_dict(self):
        return {'integral_error': self.integral_error, 'last_error': self.last_error}
    
    def load_state_dict(self, state):
        self.integral_error = state.get('integral_error', self.integral_error)
        self.last_error = state.get('last_error', self.last_error)        
        
        
class TokenizerTrainer(AdvancedTrainer):
    """
    [MISSION CONTROL] The complete TokenizerTrainer, featuring the stateful PID
    controller and a dense, multi-graph, interactive GUI with ASYNCHRONOUS and
    THROTTLED-RENDER LIVE PREVIEW for maximum performance.
    [UPGRADED] with a full suite of perceptual loss diagnostics.
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.generator_model = LatentTokenizerVQGAN(args.num_codes, args.code_dim, args.latent_grid_size, self.dtype)
        self.discriminator_model = PatchDiscriminator(dtype=self.dtype)
        
        # --- [SYSTEMS RETROFIT 1] Instantiate the new multi-metric diagnostic suite ---
        self.perceptual_loss_fn = JAXMultiMetricPerceptualLoss()

        # GAN Balancer & Lockout State
        self.d_loss_ema = 0.5; self.d_loss_ema_alpha = 0.05
        self.d_loss_target_min = 0.3; self.d_loss_target_max = 0.6
        self.g_lr_multiplier = 1.0; self.d_lr_multiplier = 1.0
        self.d_lockout_steps = 0; self.d_lockout_threshold = 0.009
        self.d_lockout_duration = 5

        # --- [SYSTEMS RETROFIT 2] Reconfigure the PID Controller for the full diagnostic suite ---
        pid_gains = {
            'l1': (1.5, 0.01, 2.0), 'vq': (1.8, 0.01, 2.5),
            'moment': (1.0, 0.01, 1.0), 'fft': (1.2, 0.01, 1.5), 'autocorr': (2.5, 0.02, 3.5),
            'edge': (2.8, 0.02, 4.0), 'color_cov': (2.0, 0.02, 2.5), 'ssim': (3.0, 0.03, 3.0)
        }
        self.lambda_controller = PIDLambdaController(
            targets={'l1': 0.05, 'vq': 0.1, 'moment': 0.2, 'fft': 0.5, 'autocorr': 0.1, 'edge': 0.1, 'color_cov': 0.05, 'ssim': 0.02},
            base_weights={'l1': 2.0, 'vq': 1.5, 'adv': 0.5, 'moment': 0.5, 'fft': 0.5, 'autocorr': 2.0, 'edge': 2.5, 'color_cov': 2.5, 'ssim': 3.0},
            gains=pid_gains
        )
        
        self.interactive_state = InteractivityState()
        self.ui_lock = threading.Lock()
        self.param_count = 0
        
        # UI State (can be expanded later if needed)
        self.hist_len = 400
        self.g_loss_hist = deque(maxlen=self.hist_len); self.d_loss_hist = deque(maxlen=self.hist_len)
        self.l1_hist = deque(maxlen=self.hist_len); self.ssim_hist = deque(maxlen=self.hist_len) # Changed from perc
        self.vq_hist = deque(maxlen=self.hist_len); self.varent_hist = deque(maxlen=self.hist_len)
        self.last_metrics_for_ui = {}; self.current_lambdas_for_ui = {}
        self.p1_params = None; self.p1_decoder_model = None
        self.preview_latents = None
        self.current_preview_np = None
        self.current_recon_np = None
        self.rendered_original_preview = None
        self.rendered_recon_preview = None

    def get_geometric_boosts(self, path_params_batch: jnp.ndarray):
        avg_radius = jnp.mean(path_params_batch[..., 2])
        complexity_factor = avg_radius / (jnp.pi / 2.0)
        # This boost now applies to all perceptual terms, not just one.
        # We will apply it inside the train loop.
        return jnp.exp(complexity_factor)

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            root_layout = Layout(name="root")
            root_layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main_content", ratio=1),
                Layout(self.progress, name="footer", size=3)
            )
            root_layout["main_content"].split_row(
                Layout(name="left_column", ratio=2, minimum_size=50),
                Layout(name="right_column", ratio=3)
            )
            
            left_stack = Layout(name="left_stack")
            right_stack = Layout(name="right_stack")
            left_stack.split_column(
                Layout(name="stats", minimum_size=6),
                Layout(name="gan_balancer", minimum_size=6),
                Layout(name="q_controller", minimum_size=5),
                Layout(name="pid_controller", ratio=1, minimum_size=12) # Increased size for more rows
            )
            right_stack.split_column(
                Layout(name="live_trends", minimum_size=15),
                Layout(name="live_preview", ratio=1, minimum_size=10)
            )

            precision_str = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"🧬 [bold]Tokenizer Trainer[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            root_layout["header"].update(Panel(Align.center(header_text), style="bold blue", title="[dim]wubumind.ai[/dim]", title_align="right"))

            g_loss, d_loss = self.last_metrics_for_ui.get('g_loss', 0), self.last_metrics_for_ui.get('d_loss', 0)
            l1, ssim, vq, edge = (self.last_metrics_for_ui.get(k,0) for k in ['l1','ssim','vq','edge'])
            stats_tbl = Table.grid(expand=True); stats_tbl.add_column(style="dim",width=14); stats_tbl.add_column()
            stats_tbl.add_row("G / D Loss", f"[cyan]{g_loss:.3f}[/] / [magenta]{d_loss:.3f}[/]")
            stats_tbl.add_row("L1 / VQ", f"[green]{l1:.2e}[/] / [green]{vq:.2e}[/]")
            stats_tbl.add_row("SSIM / Edge", f"[yellow]{ssim:.2e}[/] / [yellow]{edge:.2e}[/]")
            mem, util = self._get_gpu_stats(); stats_tbl.add_row("GPU Mem / Util", f"[yellow]{mem}[/] / [yellow]{util}[/]")
            left_stack["stats"].update(Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue"))
            
            gan_balancer_tbl = Table.grid(expand=True); gan_balancer_tbl.add_column(style="dim", width=12); gan_balancer_tbl.add_column(style="yellow")
            gan_balancer_tbl.add_row("D Loss EMA", f"{self.d_loss_ema:.3f}")
            gan_balancer_tbl.add_row("G LR Mult", f"{self.g_lr_multiplier:.2f}x")
            gan_balancer_tbl.add_row("D LR Mult", f"{self.d_lr_multiplier:.2f}x")
            if self.d_lockout_steps > 0:
                lockout_text = Text(f"LOCKED ({self.d_lockout_steps})", style="bold red")
                gan_balancer_tbl.add_row("Status", lockout_text)
            left_stack["gan_balancer"].update(Panel(gan_balancer_tbl, title="[bold]⚖️ GAN Balancer[/]", border_style="yellow"))
            
            q_panel_content = Align.center("[dim]Q-Ctrl Off[/dim]")
            if self.q_controller:
                q_tbl = Table.grid(expand=True); q_tbl.add_column(style="dim",width=12); q_tbl.add_column()
                status_full = self.q_controller.status; status_short = status_full.split(' ')[0]
                status_emoji, color = ("😎","green") if "IMPROVING" in status_short else (("🤔","yellow") if "STAGNATED" in status_short else (("😠","red") if "REGRESSING" in status_short else (("🐣","blue") if "WARMUP" in status_short else ("🤖","dim"))))
                q_tbl.add_row("Base LR", f"[{color}]{self.q_controller.current_lr:.2e}[/] {status_emoji}")
                q_tbl.add_row("Reward", f"{self.q_controller.last_reward:+.2e}"); q_tbl.add_row("Exploration", f"{self.q_controller.exploration_rate_q:.2e}")
                q_panel_content = q_tbl
            left_stack["q_controller"].update(Panel(q_panel_content, title="[bold]🤖 Q-Controller[/]", border_style="green"))
            
            pid_internals_tbl = Table("Loss", "Error", "Integral", "Deriv", "Mult", "Final λ", title_style="bold yellow")
            for name in self.lambda_controller.targets:
                error = self.lambda_controller.last_error.get(name, 0.0)
                integral = self.lambda_controller.integral_error.get(name, 0.0)
                derivative = self.lambda_controller.derivative.get(name, 0.0)
                multiplier = np.exp((self.lambda_controller.gains[name][0] * error) + (self.lambda_controller.gains[name][1] * integral) + (self.lambda_controller.gains[name][2] * derivative))
                pid_internals_tbl.add_row(name.capitalize(), f"{error:+.2e}", f"{integral:+.2e}", f"{derivative:+.2e}", f"{multiplier:.2e}", f"{self.current_lambdas_for_ui.get(name, 0):.2e}")
            left_stack["pid_controller"].update(Panel(pid_internals_tbl, title="[bold]🧠 PID Controller Internals[/]", border_style="yellow"))

            spark_w = 40
            g_loss_panel = Panel(Align.center(f"{self._get_sparkline(self.g_loss_hist, spark_w)}"), title="G Loss", height=3, border_style="cyan")
            d_loss_panel = Panel(Align.center(f"{self._get_sparkline(self.d_loss_hist, spark_w)}"), title="D Loss", height=3, border_style="magenta")
            ssim_panel = Panel(Align.center(f"{self._get_sparkline(self.ssim_hist, spark_w)}"), title="SSIM", height=3, border_style="green")
            vq_panel = Panel(Align.center(f"{self._get_sparkline(self.vq_hist, spark_w)}"), title="VQ", height=3, border_style="yellow")
            graphs = [g_loss_panel, d_loss_panel, ssim_panel, vq_panel]
            right_stack["live_trends"].update(Panel(Group(*graphs), title="[bold]📈 Live Trends[/]"))
            
            if self.rendered_recon_preview is None and self.current_recon_np is not None:
                 if Pixels: self.rendered_recon_preview = Align.center(Pixels.from_image(Image.fromarray(self.current_recon_np)))

            preview_content = Align.center("...Waiting for first validation...")
            if self.rendered_original_preview and self.rendered_recon_preview:
                preview_table = Table.grid(expand=True); preview_table.add_column(ratio=1); preview_table.add_column(ratio=1)
                preview_table.add_row(Text("Original", justify="center"), Text("Reconstruction", justify="center"))
                preview_table.add_row(self.rendered_original_preview, self.rendered_recon_preview)
                preview_content = preview_table
            
            right_stack["live_preview"].update(Panel(preview_content, title="[bold]🖼️ Live Validation Preview[/]", border_style="green"))

            root_layout["left_column"].update(left_stack)
            root_layout["right_column"].update(right_stack)
            
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
        
        @partial(jax.jit, static_argnames=('resolution','patch_size'))
        def render_image(params, path_params, resolution=128, patch_size=64):
             coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
             coord_chunks = jnp.array_split(coords, (resolution**2)//(patch_size**2))
             pixels_list = [self.p1_decoder_model.apply({'params': params}, path_params, c, method=self.p1_decoder_model.decode) for c in coord_chunks]
             return jnp.concatenate(pixels_list, axis=1).reshape(path_params.shape[0], resolution, resolution, 3)
        
        original_preview_batch = render_image(self.p1_params, self.preview_latents)
        self.current_preview_np = np.array(((original_preview_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        if Pixels:
            self.rendered_original_preview = Align.center(Pixels.from_image(Image.fromarray(self.current_preview_np)))
        
        console.print(f"✅ Data split: {len(train_data)} training samples, {len(val_data)} validation samples.")
        steps_per_epoch = math.ceil(len(train_data) / self.args.batch_size)
        
        key = jax.random.PRNGKey(self.args.seed)
        g_key, d_key, self.train_key = jax.random.split(key, 3)
        gen_optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=self.args.lr, b1=0.5, b2=0.9)
        disc_optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=self.args.lr * 0.8, b1=0.5, b2=0.9)
        dummy_input = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), self.dtype)
        gen_params = self.generator_model.init(g_key, dummy_input)['params']; disc_params = self.discriminator_model.init(d_key, dummy_input)['params']
        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, gen_params, 0)
        gen_state = TrainState.create(apply_fn=self.generator_model.apply, params=gen_params, tx=gen_optimizer)
        disc_state = TrainState.create(apply_fn=self.discriminator_model.apply, params=disc_params, tx=disc_optimizer)
        states = GANTrainStates(generator=gen_state, discriminator=disc_state)
        ckpt_path = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_final.pkl"); ckpt_path_best = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_best.pkl")
        best_val_loss, start_epoch, global_step = float('inf'), 0, 0
        
        # --- [SYSTEMS RETROFIT 3] Initialize last_metrics with all new keys ---
        last_metrics = {
            'g_loss': 0.0, 'd_loss': 0.5, 'l1': 0.05, 'vq': 0.1, 
            'moment': 0.2, 'fft': 0.5, 'autocorr': 0.1, 'edge': 0.1, 
            'color_cov': 0.05, 'ssim': 0.02, 'varentropy': 0.0
        }

        if ckpt_path.exists():
            console.print(f"--- Resuming training state from: [green]{ckpt_path}[/green] ---")
            with open(ckpt_path, 'rb') as f: ckpt = pickle.load(f)
            gen_opt_state, disc_opt_state = ckpt['gen_opt_state'], ckpt['disc_opt_state']
            start_epoch = ckpt.get('epoch', 0)
            # Make sure loaded metrics have all keys for the new controller
            loaded_metrics = ckpt.get('last_metrics', {})
            last_metrics.update(loaded_metrics)
            global_step = ckpt.get('global_step', start_epoch * steps_per_epoch)
            if self.q_controller and 'q_controller_state' in ckpt: self.q_controller.load_state_dict(ckpt['q_controller_state']); console.print(f"🤖 Q-Controller state restored.")
            if 'pid_controller_state' in ckpt: self.lambda_controller.load_state_dict(ckpt['pid_controller_state']); console.print(f"🧠 PID Controller state restored.")
            if ckpt_path_best.exists():
                console.print(f"--- Loading BEST generator weights from: [bold magenta]{ckpt_path_best}[/bold magenta] ---")
                with open(ckpt_path_best, 'rb') as f_best: best_ckpt = pickle.load(f_best); gen_params = best_ckpt['params']; best_val_loss = best_ckpt.get('val_loss', float('inf'))
            else: gen_params = ckpt['gen_params']
            states = GANTrainStates(generator=states.generator.replace(params=gen_params, opt_state=gen_opt_state), discriminator=states.discriminator.replace(params=ckpt['disc_params'], opt_state=disc_opt_state))
            console.print(f"✅ Resuming session from epoch {start_epoch + 1}, step {global_step}. Best val loss: {best_val_loss:.4f}")
        
        # --- [SYSTEMS RETROFIT 4] Update the train_step function ---
        @partial(jax.jit, static_argnames=('gen_apply_fn', 'disc_apply_fn', 'd_is_locked_out'))
        def train_step(states, batch, key, lambdas, gen_apply_fn, disc_apply_fn, d_is_locked_out: bool):
            # Unpack all lambdas. The PID controller now provides all of them.
            (lambda_l1, lambda_vq, lambda_adv, lambda_stink, 
             lambda_moment, lambda_fft, lambda_autocorr, lambda_edge, 
             lambda_color_cov, lambda_ssim) = lambdas

            def generator_loss_fn(p):
                gen_output = gen_apply_fn({'params': p}, batch)
                recon = gen_output['reconstructed_path_params']
                
                # --- Core Losses ---
                l1_loss = jnp.mean(jnp.abs(batch - recon))
                vq_loss = gen_output['vq_loss']
                adv_loss = jnp.mean((disc_apply_fn({'params': states.discriminator.params}, recon) - 1)**2)
                
                # --- Perceptual Suite ---
                perceptual_losses = self.perceptual_loss_fn(batch, recon, key)

                # --- Stink Field (Varentropy) ---
                z_e = gen_output['pre_quant_latents']
                _, varent = ent_varent(z_e.reshape(-1, z_e.shape[-1]))
                varentropy_loss = jnp.mean(varent)

                # --- The Grand Unification of Losses ---
                total_loss = (lambda_l1 * l1_loss) + \
                             (lambda_vq * vq_loss) + \
                             (lambda_adv * adv_loss) + \
                             (lambda_stink * varentropy_loss) + \
                             (lambda_moment * perceptual_losses['moment']) + \
                             (lambda_fft * perceptual_losses['fft']) + \
                             (lambda_autocorr * perceptual_losses['autocorr']) + \
                             (lambda_edge * perceptual_losses['edge']) + \
                             (lambda_color_cov * perceptual_losses['color_cov']) + \
                             (lambda_ssim * perceptual_losses['ssim'])

                # --- Prepare the full metrics dictionary for the PID and UI ---
                all_metrics = {'l1': l1_loss, 'vq': vq_loss, 'adv': adv_loss, 'varentropy': varentropy_loss}
                all_metrics.update(perceptual_losses) # Merge the dictionaries

                return total_loss, all_metrics

            (g_loss_total, metrics), g_grads = jax.value_and_grad(generator_loss_fn, has_aux=True)(states.generator.params)
            new_gen_state = states.generator.apply_gradients(grads=g_grads)
            
            def discriminator_loss_fn(disc_params):
                def compute_d_loss():
                    recon = gen_state.apply_fn({'params': new_gen_state.params}, batch)['reconstructed_path_params']
                    loss_real = jnp.mean((disc_apply_fn({'params': disc_params}, batch) - 1)**2)
                    loss_fake = jnp.mean(disc_apply_fn({'params': disc_params}, jax.lax.stop_gradient(recon))**2)
                    return ((loss_real + loss_fake) * 0.5).astype(jnp.float32)
                return jax.lax.cond(d_is_locked_out, lambda: 0.0, compute_d_loss)
            
            d_loss, d_grads = jax.value_and_grad(discriminator_loss_fn)(states.discriminator.params)
            new_disc_state = states.discriminator.apply_gradients(grads=d_grads)
            
            metrics['g_loss'] = g_loss_total
            metrics['d_loss'] = d_loss
            return GANTrainStates(generator=new_gen_state, discriminator=new_disc_state), metrics
            
        @partial(jax.jit, static_argnames=('apply_fn',))
        def eval_step(gen_params, apply_fn, batch, key):
            out = apply_fn({'params': gen_params}, batch)
            l1_loss = jnp.mean(jnp.abs(out['reconstructed_path_params']-batch))
            perceptual_losses = self.perceptual_loss_fn(out['reconstructed_path_params'], batch, key)
            # A good validation loss combines a raw pixel metric with a structural one.
            return (l1_loss + perceptual_losses['ssim'] + perceptual_losses['edge']).astype(jnp.float32)

        @partial(jax.jit, static_argnames=('gen_apply_fn',))
        def generate_preview(gen_params, gen_apply_fn, p1_params, preview_latents_batch):
            recon_path_params = gen_apply_fn({'params': gen_params}, preview_latents_batch)['reconstructed_path_params']
            return render_image(p1_params, recon_path_params)
        
        def _update_preview_task(gen_params, gen_apply_fn, p1_params, preview_latents_batch):
            recon_batch = generate_preview(gen_params, gen_apply_fn, p1_params, preview_latents_batch)
            recon_batch.block_until_ready()
            recon_np = np.array(((recon_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
            
            with self.ui_lock:
                self.current_recon_np = recon_np
                if Pixels:
                    self.rendered_recon_preview = Align.center(Pixels.from_image(Image.fromarray(self.current_recon_np)))

        console.print("[bold yellow]🚀 JIT compiling GAN training step...[/bold yellow]")
        dummy_batch = jnp.asarray(train_data[:self.args.batch_size], dtype=self.dtype); compile_key, self.train_key = jax.random.split(self.train_key)
        
        # JIT compile with the full lambda tuple
        dummy_lambda_dict = self.lambda_controller(last_metrics)
        dummy_lambdas = (
            dummy_lambda_dict['l1'], dummy_lambda_dict['vq'], dummy_lambda_dict['adv'], 0.2,
            dummy_lambda_dict['moment'], dummy_lambda_dict['fft'], dummy_lambda_dict['autocorr'],
            dummy_lambda_dict['edge'], dummy_lambda_dict['color_cov'], dummy_lambda_dict['ssim']
        )
        states, _ = train_step(states, dummy_batch, compile_key, dummy_lambdas, self.generator_model.apply, self.discriminator_model.apply, d_is_locked_out=False)
        if len(val_data) > 0:
            compile_key, self.train_key = jax.random.split(self.train_key)
            eval_step(states.generator.params, self.generator_model.apply, jnp.asarray(val_data[:self.args.batch_size], dtype=self.dtype), compile_key)
        generate_preview(states.generator.params, self.generator_model.apply, self.p1_params, self.preview_latents[:1])
        console.print("[green]✅ Compilation complete.[/green]")

        self.progress = Progress(TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[val_loss]:.2e}[/]"), BarColumn(), "•", TextColumn("Step {task.fields[step]}/{task.fields[steps_per_epoch]}"), "•", TimeRemainingColumn(), TextColumn("Ctrl+C to Exit"))
        epoch_task = self.progress.add_task("epochs", total=self.args.epochs, completed=start_epoch, val_loss=best_val_loss, step=0, steps_per_epoch=steps_per_epoch)
        
        rng = np.random.default_rng(self.args.seed); train_indices_shuffler = np.arange(len(train_data))
        last_ui_update_time = 0.0; UI_UPDATE_INTERVAL_SECS = 0.25

        try:
            with Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for epoch in range(start_epoch, self.args.epochs):
                    if self.should_shutdown: break
                    rng.shuffle(train_indices_shuffler)
                    for step_in_epoch in range(steps_per_epoch):
                        if self.should_shutdown: break
                        
                        start_idx = step_in_epoch * self.args.batch_size; end_idx = start_idx + self.args.batch_size
                        if start_idx >= len(train_indices_shuffler): continue
                        batch_indices = train_indices_shuffler[start_idx:end_idx]; train_batch = jnp.asarray(train_data[batch_indices], dtype=self.dtype)
                        
                        # --- [SYSTEMS RETROFIT 5] Create the full lambda tuple ---
                        perc_boost = self.get_geometric_boosts(train_batch)
                        lambda_dict = self.lambda_controller(last_metrics)
                        
                        # Apply geometric boost to all perceptual terms
                        for k in self.lambda_controller.targets.keys():
                            if k not in ['l1', 'vq', 'adv']:
                                lambda_dict[k] *= perc_boost
                        
                        self.current_lambdas_for_ui = lambda_dict
                        
                        lambda_stink = 0.2
                        # The order here MUST match the unpacking order in train_step
                        current_lambdas = (
                            lambda_dict['l1'], lambda_dict['vq'], lambda_dict['adv'], lambda_stink,
                            lambda_dict['moment'], lambda_dict['fft'], lambda_dict['autocorr'],
                            lambda_dict['edge'], lambda_dict['color_cov'], lambda_dict['ssim']
                        )
                        
                        base_lr = self.args.lr
                        if self.q_controller: base_lr = self.q_controller.choose_action()

                        self.d_loss_ema = (1 - self.d_loss_ema_alpha) * self.d_loss_ema + self.d_loss_ema_alpha * last_metrics.get('d_loss', 0.5)

                        d_is_locked_out = False
                        if self.d_lockout_steps > 0: self.d_lockout_steps -= 1; d_is_locked_out = True
                        elif self.d_loss_ema < self.d_lockout_threshold: self.d_lockout_steps = self.d_lockout_duration; d_is_locked_out = True
                        
                        self.g_lr_multiplier = 1.0; self.d_lr_multiplier = 1.0
                        if not d_is_locked_out:
                            if self.d_loss_ema < self.d_loss_target_min: self.d_lr_multiplier = 0.5; self.g_lr_multiplier = 1.5
                            elif self.d_loss_ema > self.d_loss_target_max: self.d_lr_multiplier = 1.5; self.g_lr_multiplier = 0.5
                        
                        states.generator.opt_state.hyperparams['learning_rate'] = base_lr * self.g_lr_multiplier
                        states.discriminator.opt_state.hyperparams['learning_rate'] = (base_lr * 0.8) * self.d_lr_multiplier
                        
                        step_key, self.train_key = jax.random.split(self.train_key)
                        
                        states, metrics = train_step(states, train_batch, step_key, current_lambdas, self.generator_model.apply, self.discriminator_model.apply, d_is_locked_out=d_is_locked_out)
                        metrics_cpu = {k: v.item() for k, v in metrics.items()}
                        last_metrics = metrics_cpu; self.last_metrics_for_ui = metrics_cpu

                        with self.ui_lock:
                            self.g_loss_hist.append(metrics_cpu['g_loss']); self.d_loss_hist.append(metrics_cpu['d_loss'])
                            self.l1_hist.append(metrics_cpu['l1']); self.ssim_hist.append(metrics_cpu.get('ssim',0.0))
                            self.vq_hist.append(metrics_cpu['vq']); self.varent_hist.append(metrics_cpu['varentropy'])
                        
                        if self.q_controller: self.q_controller.update_q_value(metrics_cpu['g_loss'])
                        
                        global_step += 1
                        self.progress.update(epoch_task, step=step_in_epoch + 1)
                        
                        if global_step % self.args.eval_every == 0:
                            if not (active_preview_future and not active_preview_future.done()):
                                host_gen_params = jax.device_get(states.generator.params)
                                active_preview_future = background_executor.submit(_update_preview_task, host_gen_params, self.generator_model.apply, self.p1_params, self.preview_latents)
                            
                            if len(val_data) > 0:
                                eval_key, self.train_key = jax.random.split(self.train_key)
                                eval_keys = jax.random.split(eval_key, (len(val_data) // self.args.batch_size) + 1)
                                val_losses = [eval_step(states.generator.params, self.generator_model.apply, jnp.asarray(val_data[i:i+self.args.batch_size], dtype=self.dtype), eval_keys[j]) for j, i in enumerate(range(0, len(val_data), self.args.batch_size))]
                                val_loss = np.mean([v.item() for v in val_losses]) if val_losses else float('inf')
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    console.print(f"\n[bold magenta]🏆 New best val loss: {best_val_loss:.2e} @ step {global_step}. Saving...[/bold magenta]")
                                    with open(ckpt_path_best, 'wb') as f: pickle.dump({'params': states.generator.params, 'val_loss': best_val_loss, 'epoch': epoch, 'global_step': global_step}, f)
                                self.progress.update(epoch_task, val_loss=best_val_loss)

                        current_time = time.time()
                        if current_time - last_ui_update_time > UI_UPDATE_INTERVAL_SECS:
                            live.update(self._generate_layout()); last_ui_update_time = current_time

                    self.progress.update(epoch_task, advance=1)
                    
                    host_state_to_save = jax.device_get(states)
                    q_state_to_save = self.q_controller.state_dict() if self.q_controller else None
                    pid_state_to_save = self.lambda_controller.state_dict()
                    data_to_save = {'gen_params': host_state_to_save.generator.params, 
                                    'gen_opt_state': host_state_to_save.generator.opt_state, 
                                    'disc_params': host_state_to_save.discriminator.params, 
                                    'disc_opt_state': host_state_to_save.discriminator.opt_state, 
                                    'epoch': epoch, 'global_step': global_step, 
                                    'q_controller_state': q_state_to_save, 
                                    'last_metrics': last_metrics, 
                                    'pid_controller_state': pid_state_to_save}
                    with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
        finally:
            console.print(f"\n[yellow]--- Training loop exited. Waiting for background tasks to finish... ---[/yellow]")
            background_executor.shutdown(wait=True)
            final_epoch_count = self.args.epochs if 'epoch' not in locals() else epoch
            console.print(f"\n[yellow]--- Training session finished at epoch {final_epoch_count+1}. Saving final state... ---[/yellow]")
            
            if 'states' in locals():
                host_state_final = jax.device_get(states)
                q_state_to_save = self.q_controller.state_dict() if self.q_controller else None
                pid_state_to_save = self.lambda_controller.state_dict()
                final_data = {'gen_params': host_state_final.generator.params,
                              'gen_opt_state': host_state_final.generator.opt_state,
                              'disc_params': host_state_final.discriminator.params,
                              'disc_opt_state': host_state_final.discriminator.opt_state,
                              'epoch': final_epoch_count, 'global_step': global_step,
                              'q_controller_state': q_state_to_save,
                              'last_metrics': last_metrics,
                              'pid_controller_state': pid_state_to_save}
                with open(ckpt_path, 'wb') as f: pickle.dump(final_data, f)
                console.print(f"✅ Final resume-state saved to [green]{ckpt_path}[/green]")

            config = {'num_codes': self.args.num_codes, 'code_dim': self.args.code_dim, 'latent_grid_size': self.args.latent_grid_size}
            config_path = Path(str(ckpt_path).replace("_final.pkl", "_config.pkl"))
            with open(config_path, 'wb') as f: pickle.dump(config, f)
            console.print(f"✅ Config saved to [green]{config_path}[/green]")
            if ckpt_path_best.exists(): console.print(f"👑 Best model (by validation) remains at [bold magenta]{ckpt_path_best}[/bold magenta]")


# =================================================================================================
# NEW: CHIMERA-AWARE GUI LAYOUT MANAGER
# =================================================================================================

class GenerationalLayoutManager:
    """
    [NEW] A dedicated layout manager for the "No Teacher" Generational Ecosystem.
    Visualizes the different tiers of agents (babies, kids, etc.).
    """
    def __init__(self, trainer_instance):
        self.trainer = trainer_instance
        self.console = Console()

    def generate_layout(self) -> Layout:
        trainer = self.trainer
        with trainer.ui_lock:
            # --- Header ---
            precision_str = "MIXED" # Reflects the multi-precision nature of the population
            header_text = f"🧬 [bold]GENERATIONAL WORLD MODEL[/] | Total Params: [yellow]{trainer.param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            header_panel = Panel(Align.center(header_text), style="bold red", title="[dim]wubumind.ai[/dim]", title_align="right")

            # --- Main Content Area ---
            main_layout = Layout(name="main")
            main_layout.split_row(
                Layout(name="agent_tiers", ratio=2),
                Layout(name="system_preview", ratio=1)
            )

            # --- Agent Tier Panels ---
            tier_panels = []
            tier_colors = {"babies": "cyan", "kids": "magenta", "students": "yellow", "polisher": "green"}
            
            for tier_name, config in trainer.population_tiers.items():
                tier_table = Table(box=None, expand=True, padding=(0,1))
                tier_table.add_column("Agent", style="dim", width=4)
                tier_table.add_column("LR", width=8)
                tier_table.add_column("Status")

                for i in range(config['count']):
                    controller = trainer.agent_controllers[tier_name]['lr_controllers'][i]
                    status_short = controller.status.split(' ')[0]
                    if "IMPROVING" in status_short: s_emoji, s_color = "😎", "green"
                    elif "STAGNATED" in status_short: s_emoji, s_color = "🤔", "yellow"
                    else: s_emoji, s_color = "😠", "red"

                    tier_table.add_row(f"{i}", f"{controller.current_value:.2e}", f"[{s_color}]{status_short}[/] {s_emoji}")

                panel_title = f"[bold {tier_colors.get(tier_name, 'white')}]{tier_name.capitalize()} Tier[/] ({config['count']}x {config['d_model']}d, {config['num_layers']}L)"
                tier_panels.append(Panel(tier_table, title=panel_title, border_style=tier_colors.get(tier_name, 'white')))

            main_layout["agent_tiers"].split_column(*tier_panels)
            
            # --- Right Column (System, Preview) ---
            right_column_layout = Layout(name="right_column")
            
            global_stats_tbl = Table.grid(expand=True, padding=(0,1)); global_stats_tbl.add_column(style="dim", width=11); global_stats_tbl.add_column()
            global_stats_tbl.add_row("Steps/sec", f"[blue]{trainer.steps_per_sec:.2f}[/] 🏃💨"); mem, util = trainer._get_gpu_stats()
            global_stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); global_stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            global_stats_panel = Panel(global_stats_tbl, title="[bold]📊 System Stats[/]", border_style="blue", height=5)
            
            current_prompt = trainer.validation_prompts[trainer.current_preview_prompt_idx]
            prompt_text = Text(f"Prompt #{trainer.current_preview_prompt_idx+1}: \"{current_prompt}\"", justify="center")
            preview_content = trainer.rendered_preview if trainer.rendered_preview else Align.center("...Awaiting First Generation...")
            preview_group = Group(prompt_text, preview_content)
            preview_panel = Panel(preview_group, title="[bold]🖼️ World Model Preview[/]", border_style="green")
            
            right_column_layout.split_column(global_stats_panel, preview_panel)
            main_layout["system_preview"].update(right_column_layout)

            # --- Footer (Progress Bar) ---
            root_layout = Layout(name="root")
            root_layout.split_column(
                Layout(header_panel, name="header", size=3),
                Layout(main_layout, name="main", ratio=1),
                Layout(trainer.progress, name="progress", size=3)
            )
            return root_layout


            
class BaseConductorTrainer(AdvancedTrainer):
    """
    Base class for Conductor training. Handles all common setup, including model loading,
    data preparation, and JIT compilation of validation/preview functions.
    """
    def __init__(self, args):
        super().__init__(args)
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        
        console = Console()
        self.preview_resolution = 96 if args.vram_saver_mode else 128

        try:
            tok_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_*_config.pkl"))
        except StopIteration:
            sys.exit(f"[FATAL] VQ-GAN tokenizer config not found for basename '{args.basename}'.")
        
        tok_path_best = Path(str(tok_config_path).replace("_config.pkl", "_best.pkl"))
        tok_path_final = Path(str(tok_config_path).replace("_config.pkl", "_final.pkl"))
        tok_ckpt_path = tok_path_best if tok_path_best.exists() else tok_path_final
        if not tok_ckpt_path.exists(): sys.exit(f"[FATAL] No VQ-GAN tokenizer model found.")
        
        console.print(f"--- Loading VQ-GAN tokenizer from [green]{tok_ckpt_path}[/green] ---")
        with open(tok_ckpt_path,'rb') as f:
             tok_ckpt = pickle.load(f)
             self.tok_params = tok_ckpt.get('params', tok_ckpt.get('gen_params'))
             codebook_cpu = self.tok_params['vq']['codebook'].T
             self.tokenizer_codebook = jax.device_put(codebook_cpu.astype(self.dtype))
        
        with open(tok_config_path, 'rb') as f: self.tok_config = pickle.load(f)
        
        try:
            p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
            p1_d_model = int(p1_path.stem.split('_')[-2].replace('d', ''))
        except (StopIteration, ValueError):
            sys.exit(f"[FATAL] Could not find a unique Phase 1 model file for basename '{args.basename}'.")

        console.print(f"--- Loading Phase 1 AE from: [green]{p1_path}[/green] (d_model={p1_d_model}) ---")
        self.p1_model = TopologicalCoordinateGenerator(p1_d_model, self.tok_config['latent_grid_size'], 512, self.dtype)
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']

        self.token_map_size = (self.tok_config['latent_grid_size'] // 4) ** 2
        self.cond_config = {
            'num_codes': args.num_codes,

            'd_model': args.d_model_cond,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'clip_dim': 512
        }

        self.tokenizer = LatentTokenizerVQGAN(**self.tok_config, dtype=self.dtype)
        self.model = GenerativeConductor(**self.cond_config, mask_ratio=args.mask_ratio, dtype=self.dtype)
        
        self.interactive_state = InteractivityState()
        self.loss_history = deque(maxlen=200)
        self.spinner_chars = ["🧠", "⚡", "💾", "📈", "🧠", "⚡", "💽", "📉"]
        self.spinner_idx, self.param_count, self.steps_per_sec = 0, 0, 0.0
        self.ui_lock = threading.Lock()
        
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        self.validation_prompts = ["a red cup on a table", "a photorealistic orange cat", "a blue ball", "green grass with a single tree", "a purple sports car"]
        with torch.no_grad():
            text_tokens = clip.tokenize(self.validation_prompts).to(_clip_device)
            self.validation_embeddings = self.clip_model.encode_text(text_tokens).cpu().numpy()
        
        self.current_preview_prompt_idx = 0
        self.rendered_preview = None # Unified attribute for the final Rich renderable

    def _save_cpu_data_task(self, cpu_data_to_save, path):
        with open(path, 'wb') as f: pickle.dump(cpu_data_to_save, f)

    def _get_common_train_setup(self):
        console = Console()
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
        
        preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='PreviewGenerator')

        tokenized_data_path = Path(self.args.data_dir) / f"tokenized_data_{self.args.basename}_{self.args.num_codes}c.npz"
        
        if not tokenized_data_path.exists():
            if getattr(self.args, 'debug_fast_init', False):
                sys.exit(f"[FATAL] --debug-fast-init requires tokenized data to exist at {tokenized_data_path}")
                
            console.print("--- 🧠 STEP 1: Pre-tokenized data not found. Creating it now... ---", style="bold yellow")
            data_path = Path(self.args.data_dir) / f"paired_data_{self.args.basename}.pkl";
            if not data_path.exists(): sys.exit(f"[FATAL] Paired data not found at {data_path}")
            with open(data_path, 'rb') as f: data = pickle.load(f)
            jit_tokenizer_encode = jax.jit(lambda p, l: self.tokenizer.apply({'params': p}, l, method=self.tokenizer.encode))
            all_tokens_list = []; tokenization_batch_size = self.args.batch_size * 8; latents_jnp = jnp.asarray(data['latents'])
            for i in tqdm(range(0, len(latents_jnp), tokenization_batch_size), desc="Tokenizing"):
                tokens_2d = jit_tokenizer_encode(self.tok_params, latents_jnp[i:i + tokenization_batch_size]); all_tokens_list.append(tokens_2d.reshape(tokens_2d.shape[0], -1))
            all_tokens_flat = np.array(jnp.concatenate(all_tokens_list, axis=0)); all_embeddings = data['embeddings']
            del data, all_tokens_list, latents_jnp
            np.random.seed(self.args.seed); shuffled_indices = np.random.permutation(len(all_tokens_flat))
            val_split_idx = int(len(all_tokens_flat) * 0.02); train_indices, val_indices = shuffled_indices[val_split_idx:], shuffled_indices[:val_split_idx]
            train_tokens, train_embeddings = all_tokens_flat[train_indices], all_embeddings[train_indices]; val_tokens, val_embeddings = all_tokens_flat[val_indices], all_embeddings[val_indices]
            # Save with a consistent type for streaming
            np.savez_compressed(tokenized_data_path, 
                                train_tokens=train_tokens.astype(np.int32), 
                                train_embeddings=train_embeddings.astype(np.float32), 
                                val_tokens=val_tokens.astype(np.int32), 
                                val_embeddings=val_embeddings.astype(np.float32))
        else:
            if getattr(self.args, 'debug_fast_init', False):
                console.print("--- [DEBUG] FAST INIT: Assuming tokenized data is present. ---", style="bold red")
            else:
                 console.print(f"--- ✅ Found pre-tokenized data at [green]{tokenized_data_path}[/green] ---", style="bold yellow")

        # [THE FIX] Define a Python generator for true streaming
        def _data_generator(path, split_prefix):
            # np.load uses memory-mapping, which is VRAM-efficient.
            with np.load(path) as data:
                tokens = data[f'{split_prefix}_tokens']
                embeddings = data[f'{split_prefix}_embeddings']
                for i in range(len(tokens)):
                    yield tokens[i], embeddings[i]

        # Get the shapes and types from the file for the signature
        with np.load(tokenized_data_path) as data:
            num_train_samples = data['train_tokens'].shape[0]
            num_val_samples = data['val_tokens'].shape[0]
            token_shape = data['train_tokens'].shape[1:]
            embedding_shape = data['train_embeddings'].shape[1:]

        output_signature = (
            tf.TensorSpec(shape=token_shape, dtype=tf.int32),
            tf.TensorSpec(shape=embedding_shape, dtype=tf.float32)
        )

        # [THE FIX] Create the dataset using the generator
        train_ds = tf.data.Dataset.from_generator(
            lambda: _data_generator(str(tokenized_data_path), 'train'),
            output_signature=output_signature
        )
        
        train_ds = train_ds.shuffle(10000, seed=self.args.seed).repeat().batch(self.args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        train_iterator = train_ds.as_numpy_iterator()

        val_ds = None
        if num_val_samples > 0:
             val_ds = tf.data.Dataset.from_generator(
                lambda: _data_generator(str(tokenized_data_path), 'val'),
                output_signature=output_signature
             )
             val_ds = val_ds.batch(self.args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
             console.print(f"✅ Validation dataset prepared with {num_val_samples} samples (true streaming).")

        return key_listener_thread, preview_executor, train_iterator, val_ds, num_train_samples
        
    def train(self):
        raise NotImplementedError("Trainer subclasses must implement the `train` method.")
        
        



# Place near the top with other class definitions.
class StaticAgentData(NamedTuple):
    kids: Tuple[GenerativeConductor, ...]
    students: Tuple[GenerativeConductor, ...]
    polisher: Tuple[GenerativeConductor, ...]

class GenerationalConductorTrainer(BaseConductorTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.console = Console()
        self.console.print("--- [MODE] Generational 'No Teacher' Spatiotemporal Ecosystem is ACTIVE. ---", style="bold yellow")
        
        self.population_tiers = {
            "kids":     {"count": 2, "d_model": 256, "num_layers": 4,  "num_heads": 8,  "dtype": jnp.bfloat16},
            "students": {"count": 1, "d_model": 512, "num_layers": 8,  "num_heads": 8,  "dtype": jnp.float32},
            "polisher": {"count": 1, "d_model": 768, "num_layers": 12, "num_heads": 12, "dtype": jnp.float32}
        }
        
        self.agent_population = {tier: {'models': [], 'states': []} for tier in self.population_tiers}
        self.agent_controllers = {tier: {'lr_controllers': []} for tier in self.population_tiers}

        q_lr_config = {"q_table_size":100,"num_actions":5,"action_factors":[0.9,0.95,1.0,1.05,1.1],"learning_rate_q":0.1,"discount_factor_q":0.9,"value_min":1e-6,"value_max":1e-3,"metric_history_len":5000,"metric_min":0.1,"metric_max":8.0,"exploration_rate_q":0.3,"min_exploration_rate":0.05,"exploration_decay":0.9995,"trend_window":420,"improve_threshold":1e-4,"regress_threshold":1e-5,"regress_penalty":10.0,"stagnation_penalty":-2.0,"warmup_steps":420,"warmup_start_val":1e-6}

        for tier_name, config in self.population_tiers.items():
            for i in range(config['count']):
                model = GenerativeConductor(
                    num_codes=args.num_codes, d_model=config['d_model'], num_heads=config['num_heads'],
                    num_layers=config['num_layers'], mask_ratio=args.mask_ratio, dtype=config['dtype']
                )
                self.agent_population[tier_name]['models'].append(model)
                lr_controller = JaxHakmemQController(initial_value=args.lr, config=q_lr_config, param_name=f"{tier_name}_{i}/LR")
                self.agent_controllers[tier_name]['lr_controllers'].append(lr_controller)
        
        self.lambda_world_mmd = 0.2
        self.lambda_refinement = 0.5
        self.lambda_mlm = 0.2

        self.ui_lock = threading.Lock()
        self.last_metrics_for_ui = {} 
        self.layout_manager = GenerationalLayoutManager(self)

    def _initialize_states(self):
        key = jax.random.PRNGKey(self.args.seed)
        total_params = 0
        
        for tier_name, tier_data in self.agent_population.items():
            tier_data['states'] = []
            for i in range(self.population_tiers[tier_name]['count']):
                model = tier_data['models'][i]
                key, init_key = jax.random.split(key)
                dummy_tokens_4d = jnp.zeros((1, model.time_size, model.depth_size, model.height_size, model.width_size), jnp.int32)
                dummy_embeddings = jnp.zeros((1, 512), self.dtype)
                
                # [THE FIX] Call model.init using the new method to ensure all params are created.
                params = model.init(
                    {'params': init_key, 'dropout': init_key}, 
                    dummy_tokens_4d, 
                    dummy_embeddings, 
                    train=True,
                    method=model.init_forward_pass
                )['params']
                
                total_params += jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, params, 0)
                optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr))
                state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
                tier_data['states'].append(state)
        self.param_count = total_params
        
        
    @staticmethod
    def _static_generational_pass(all_params, tok_params, p1_params, text_emb, key,
                                  agent_models, lambdas, tokenizer, p1_model, static_tok_grid_size):
        B = text_emb.shape[0]
        lambda_mlm, lambda_refinement, lambda_world_mmd = lambdas
        
        p_model = agent_models.polisher[0]
        canvas_shape = (B, p_model.time_size, p_model.depth_size, p_model.height_size, p_model.width_size)
        
        initial_tokens_int = jnp.full(canvas_shape, p_model.MASK_TOKEN_ID, dtype=jnp.int32)
        initial_tokens_onehot = jax.nn.one_hot(initial_tokens_int, p_model.vocab_size, dtype=jnp.float32)

        total_loss = jnp.array(0.0, dtype=jnp.float32)
        metrics = {}
        all_logits = {}

        key, dropout_key = jax.random.split(key)

        # --- Tier 1: Kids ---
        kid_models = agent_models.kids
        kid_params_list = all_params['kids']
        kid_logits_list = [m.apply({'params': p}, initial_tokens_onehot, text_emb, train=True, rngs={'dropout': dropout_key})[1] for m, p in zip(kid_models, kid_params_list)]
        kid_logits_ensembled = jnp.mean(jnp.stack(kid_logits_list, axis=0), axis=0)
        all_logits['kids'] = kid_logits_ensembled
        key, ste_key = jax.random.split(key)
        kid_ste, kid_tokens_hard = gumbel_softmax_straight_through(kid_logits_ensembled, ste_key)
        
        # [THE FIX] one_hot must use vocab_size to match the logits shape
        mlm_loss_kids = optax.softmax_cross_entropy(kid_logits_ensembled, jax.nn.one_hot(kid_tokens_hard, p_model.vocab_size)).mean()
        total_loss += lambda_mlm * mlm_loss_kids
        metrics['loss/mlm_kids'] = mlm_loss_kids
        
        # --- Tier 2: Students ---
        student_models = agent_models.students
        student_params_list = all_params['students']
        student_logits_list = [m.apply({'params': p}, kid_ste, text_emb, train=True, rngs={'dropout': dropout_key})[1] for m, p in zip(student_models, student_params_list)]
        student_logits_ensembled = jnp.mean(jnp.stack(student_logits_list, axis=0), axis=0)
        all_logits['students'] = student_logits_ensembled
        key, ste_key = jax.random.split(key)
        student_ste, student_tokens_hard = gumbel_softmax_straight_through(student_logits_ensembled, ste_key)
        
        # [THE FIX] one_hot must use vocab_size
        mlm_loss_students = optax.softmax_cross_entropy(student_logits_ensembled, jax.nn.one_hot(student_tokens_hard, p_model.vocab_size)).mean()
        total_loss += lambda_mlm * mlm_loss_students
        metrics['loss/mlm_students'] = mlm_loss_students
        
        # --- Tier 3: Polisher ---
        polisher_model = agent_models.polisher[0]
        polisher_params = all_params['polisher'][0]
        polisher_logits, polisher_patch_embeddings = polisher_model.apply(
            {'params': polisher_params}, student_ste, text_emb, train=True, rngs={'dropout': dropout_key}, method=polisher_model.get_features
        )
        all_logits['polisher'] = polisher_logits
        key, ste_key = jax.random.split(key)
        _, polisher_tokens_hard = gumbel_softmax_straight_through(polisher_logits, ste_key)
        
        # [THE FIX] one_hot must use vocab_size
        mlm_loss_polisher = optax.softmax_cross_entropy(polisher_logits, jax.nn.one_hot(polisher_tokens_hard, p_model.vocab_size)).mean()
        total_loss += lambda_mlm * mlm_loss_polisher
        metrics['loss/mlm_polisher'] = mlm_loss_polisher
        
        # [THE FIX] one_hot must use vocab_size for refinement targets
        refinement_loss_student = optax.softmax_cross_entropy(all_logits['students'], jax.lax.stop_gradient(jax.nn.one_hot(polisher_tokens_hard, p_model.vocab_size))).mean()
        refinement_loss_kid = optax.softmax_cross_entropy(all_logits['kids'], jax.lax.stop_gradient(jax.nn.one_hot(student_tokens_hard, p_model.vocab_size))).mean()
        total_refinement_loss = refinement_loss_student + refinement_loss_kid
        total_loss += lambda_refinement * total_refinement_loss
        metrics['loss/refinement'] = total_refinement_loss

        @partial(jax.jit, static_argnames=('resolution','patch_size'))
        def _render(p1_params, path_params, resolution=512, patch_size=256):
            coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
            coord_chunks = jnp.array_split(coords,(resolution**2)//(patch_size**2))
            pixels = [p1_model.apply({'params':p1_params}, path_params, c, method=p1_model.decode) for c in coord_chunks]
            return jnp.concatenate(pixels, axis=1).reshape(B, resolution, resolution, 3)

        middle_frame_tokens_2d = polisher_tokens_hard[:, canvas_shape[1] // 2, 0, :, :].reshape(B, static_tok_grid_size, static_tok_grid_size)
        path_params = tokenizer.apply({'params': tok_params}, middle_frame_tokens_2d, method=tokenizer.decode)
        rendered_image = _render(p1_params, path_params)
        text_emb_expanded = jnp.tile(text_emb[:, None, :], (1, polisher_patch_embeddings.shape[1], 1))
        mmd_loss = calculate_mmd_loss(polisher_patch_embeddings, text_emb_expanded)
        total_loss += lambda_world_mmd * mmd_loss
        metrics['loss/world_mmd'] = mmd_loss
        
        metrics['loss/total'] = total_loss
        return total_loss, (metrics, rendered_image)
        
    def train(self):
        self.console.print("--- [PROJECT CHIMERA] Activating Generational Conductor Training. ---", style="bold red")
        key = jax.random.PRNGKey(self.args.seed)
        
        key_listener_thread, preview_executor, train_iterator, val_ds, num_train_samples = self._get_common_train_setup()
        self._initialize_states()
        
        all_states = {tier: self.agent_population[tier]['states'] for tier in self.population_tiers}
        
        ckpt_path = Path(f"chimera_{self.args.basename}_{self.args.num_codes}c_final.pkl"); start_step = 0
        if ckpt_path.exists():
            self.console.print(f"--- Resuming training from [green]{ckpt_path}[/green] ---")
            with open(ckpt_path, 'rb') as f: ckpt = pickle.load(f)
            start_step = ckpt.get('global_step', 0)
            loaded_states_raw = ckpt.get('all_states_raw')
            
            # [THE FINAL FIX] Replace the buggy tree_map with robust manual reconstruction.
            if loaded_states_raw:
                restored_states = {}
                for tier_name, initial_tier_states in all_states.items():
                    restored_tier_states = []
                    if tier_name in loaded_states_raw:
                        for i, state in enumerate(initial_tier_states):
                            if i < len(loaded_states_raw[tier_name]):
                                loaded_state_dict = loaded_states_raw[tier_name][i]
                                # Create a new TrainState from the loaded components
                                new_state = state.replace(
                                    step=loaded_state_dict['step'],
                                    params=loaded_state_dict['params'],
                                    opt_state=loaded_state_dict['opt_state']
                                )
                                restored_tier_states.append(new_state)
                            else:
                                restored_tier_states.append(state)
                    else:
                        restored_tier_states = initial_tier_states
                    restored_states[tier_name] = restored_tier_states
                all_states = restored_states

            for tier in self.population_tiers:
                for i, controller in enumerate(self.agent_controllers[tier]['lr_controllers']):
                    if f'{tier}_{i}_q_controller' in ckpt: controller.load_state_dict(ckpt[f'{tier}_{i}_q_controller'])
        
        agent_models = StaticAgentData(
            kids=tuple(self.agent_population['kids']['models']),
            students=tuple(self.agent_population['students']['models']),
            polisher=tuple(self.agent_population['polisher']['models'])
        )
        lambdas = (self.lambda_mlm, self.lambda_refinement, self.lambda_world_mmd)
        static_tok_grid_size = self.tok_config['latent_grid_size'] // 4
        tok_params = self.tok_params
        p1_params = self.p1_params

        @partial(jax.jit, 
                 static_argnames=('agent_models', 'lambdas', 'tokenizer', 'p1_model', 'static_tok_grid_size'))
        def jit_grad_and_loss_fn(all_params, tok_params, p1_params, text_emb, key, 
                                 agent_models, lambdas, tokenizer, p1_model, static_tok_grid_size):
            grad_fn = jax.value_and_grad(self._static_generational_pass, argnums=0, has_aux=True, allow_int=True)
            (loss, (metrics, image)), grads = grad_fn(all_params, tok_params, p1_params, text_emb, key, 
                                                      agent_models, lambdas, tokenizer, p1_model, static_tok_grid_size)
            return loss, metrics, image, grads

        if not getattr(self.args, 'debug_fast_init', False):
            self.console.print("--- 🚀 JIT Compiling full generational training step... (This will take a while) ---")
            dummy_batch = next(train_iterator)[1][:1] 
            key, compile_key = jax.random.split(key)
            all_params = {tier: [state.params for state in states_list] for tier, states_list in all_states.items()}
            _, _, _, _ = jit_grad_and_loss_fn(
                all_params, tok_params, p1_params, dummy_batch, compile_key,
                agent_models, lambdas, self.tokenizer, self.p1_model, static_tok_grid_size
            )
            self.console.print("✅ Compilation complete.")
        else:
            self.console.print("--- [DEBUG] FAST INIT: SKIPPING JIT COMPILATION. ---", style="bold red")
        
        self.progress = Progress(TextColumn("[bold]Step {task.completed}/{task.total} [green]Total Loss: {task.fields[loss]:.3f}[/]"), BarColumn(), "•", TextColumn("Steps/sec: {task.fields[sps]:.2f}"), "•", TimeRemainingColumn(), TextColumn("Ctrl+C to Exit"))
        train_task_total = self.args.epochs * (num_train_samples // self.args.batch_size)
        train_task = self.progress.add_task("training", total=train_task_total, completed=start_step, loss=0.0, sps=0.0)

        step_times = deque(maxlen=100)
        try:
            with Live(self.layout_manager.generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for step in range(start_step, train_task_total):
                    if self.should_shutdown: break
                    
                    start_time = time.time()
                    _, text_emb_batch = next(train_iterator)
                    
                    for tier_name in self.agent_controllers:
                        for i, controller in enumerate(self.agent_controllers[tier_name]['lr_controllers']):
                            new_lr = controller.choose_action()
                            
                            original_opt_state = all_states[tier_name][i].opt_state
                            inner_optimizer_state = original_opt_state[0]
                            hyperparams_state = original_opt_state[1]

                            new_hyperparams_dict = dict(hyperparams_state.hyperparams)
                            new_hyperparams_dict['learning_rate'] = new_lr
                            
                            new_hyperparams_state = type(hyperparams_state)(
                                count=hyperparams_state.count,
                                hyperparams=new_hyperparams_dict,
                                hyperparams_states=hyperparams_state.hyperparams_states,
                                inner_state=hyperparams_state.inner_state
                            )
                            
                            new_opt_state = (inner_optimizer_state, new_hyperparams_state)
                            
                            all_states[tier_name][i] = all_states[tier_name][i].replace(
                                opt_state=new_opt_state
                            )

                    key, step_key = jax.random.split(key)
                    
                    all_params = {tier: [state.params for state in states_list] for tier, states_list in all_states.items()}
                    
                    loss, metrics, rendered_image, grads = jit_grad_and_loss_fn(
                        all_params, tok_params, p1_params, text_emb_batch, step_key,
                        agent_models, lambdas, self.tokenizer, self.p1_model, static_tok_grid_size
                    )

                    # --- [THE FIX] Manually iterate and apply gradients ---
                    # The structure of `grads` is Dict[tier, List[grad_pytree]].
                    # The structure of `all_states` is Dict[tier, List[TrainState]].
                    # This loop correctly pairs each TrainState with its corresponding grad_pytree.
                    new_states = {}
                    for tier_name, tier_grads in grads.items():
                        new_tier_states = []
                        for i, grad in enumerate(tier_grads):
                            current_state = all_states[tier_name][i]
                            new_state = current_state.apply_gradients(grads=grad)
                            new_tier_states.append(new_state)
                        new_states[tier_name] = new_tier_states
                    all_states = new_states
                    
                    metrics_cpu = {k: v.item() for k, v in metrics.items()}
                    total_loss_val = metrics_cpu.get('loss/total', 0.0)
                    
                    for tier_controllers in self.agent_controllers.values():
                        for controller in tier_controllers['lr_controllers']:
                            controller.update_q_value(total_loss_val)

                    step_times.append(time.time() - start_time)
                    self.steps_per_sec = 1.0 / (sum(step_times) / len(step_times)) if step_times else 0.0
                    self.progress.update(train_task, advance=1, loss=total_loss_val, sps=self.steps_per_sec)
                    
                    if step % self.args.eval_every == 0:
                        img_np = np.array(((rendered_image[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
                        with self.ui_lock:
                            if Pixels: self.rendered_preview = Align.center(Pixels.from_image(Image.fromarray(img_np)))

                    if step % 50 == 0:
                        live.update(self.layout_manager.generate_layout())

                    if step % 2000 == 0 and step > 0:
                        data_to_save = {'global_step': step}
                        all_states_raw = {
                            tier: [{'step': s.step, 'params': s.params, 'opt_state': s.opt_state} for s in states_list]
                            for tier, states_list in jax.device_get(all_states).items()
                        }
                        data_to_save['all_states_raw'] = all_states_raw
                        for tier_name, tier_controllers in self.agent_controllers.items():
                            for i, controller in enumerate(tier_controllers['lr_controllers']):
                                data_to_save[f'{tier_name}_{i}_q_controller'] = controller.state_dict()
                        with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
        
        finally:
            self.console.print("--- Training finished. Saving final state. ---")
            if 'all_states' in locals():
                final_step = self.progress.tasks[0].completed
                data_to_save = {'global_step': final_step}
                
                all_states_raw = {
                    tier: [{'step': s.step, 'params': s.params, 'opt_state': s.opt_state} for s in states_list]
                    for tier, states_list in jax.device_get(all_states).items()
                }

                data_to_save['all_states_raw'] = all_states_raw
                for tier_name, tier_controllers in self.agent_controllers.items():
                    for i, controller in enumerate(tier_controllers['lr_controllers']):
                        data_to_save[f'{tier_name}_{i}_q_controller'] = controller.state_dict()
                with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
                self.console.print(f"✅ Final checkpoint saved to [green]{ckpt_path}[/green]")







# =================================================================================================
# =================================================================================================
# 6. GENERATION & INFERENCE 
# =================================================================================================

class Generator:
    def __init__(self, args):
        self.args = args
        self.console = Console()
        self.console.print("--- 🧠 Loading Full Generative Stack (Masked Bidirectional) ---", style="bold yellow")
        
        try:
            conductor_config_path = next(Path('.').glob(f"chimera_{args.basename}_*_config.pkl"))
            tokenizer_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_*_config.pkl"))
            p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
        except StopIteration:
            sys.exit(f"[FATAL] Could not find required config or model files for basename '{args.basename}'.")

        with open(conductor_config_path, 'rb') as f: self.cond_config = pickle.load(f)
        with open(tokenizer_config_path, 'rb') as f: self.tok_config = pickle.load(f)

        self.dtype = jnp.float32
        p1_d_model = int(p1_path.stem.split('_')[-2].replace('d',''))
        self.p1_model = TopologicalCoordinateGenerator(p1_d_model, self.tok_config['latent_grid_size'], 512, self.dtype)
        self.tokenizer = LatentTokenizerVQGAN(**self.tok_config, dtype=self.dtype)
        self.conductor = GenerativeConductor(**self.cond_config, mask_ratio=0.5, dtype=self.dtype) # mask_ratio is a dummy here
        
        cond_ckpt_path = Path(str(conductor_config_path).replace("_config.pkl", "_best.pkl"))
        if not cond_ckpt_path.exists(): cond_ckpt_path = Path(str(cond_ckpt_path).replace("_best.pkl", "_final.pkl"))
        tok_ckpt_path = Path(str(tokenizer_config_path).replace("_config.pkl", "_best.pkl"))
        if not tok_ckpt_path.exists(): tok_ckpt_path = Path(str(tok_ckpt_path).replace("_best.pkl", "_final.pkl"))

        self.console.print(f"-> Loading Phase 1 AE from: [green]{p1_path}[/green]")
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']
        self.console.print(f"-> Loading Tokenizer from: [green]{tok_ckpt_path}[/green]")
        with open(tok_ckpt_path, 'rb') as f: tok_data = pickle.load(f); self.tok_params = tok_data.get('params', tok_data.get('gen_params'))
        self.console.print(f"-> Loading Conductor from: [green]{cond_ckpt_path}[/green]")
        with open(cond_ckpt_path, 'rb') as f: self.cond_params = pickle.load(f)['master_params']
        
        self.uncond_embedding = self.cond_params['uncond_embedding']
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        self.token_map_size = (self.tok_config['latent_grid_size'] // 4) ** 2

        self.console.print("--- 🚀 JIT Compiling non-autoregressive inference kernels... ---")
        self._jit_compile_functions()
        self.console.print("✅ All kernels compiled.")

    def _jit_compile_functions(self):
        @partial(jax.jit, static_argnames=('model_apply_fn', 'num_steps', 'guidance_scale'))
        def _non_autoregressive_cfg_decode_jit(
            model_apply_fn, params, key, cond_emb, uncond_emb, guidance_scale, num_steps
        ):
            B, L = 1, self.token_map_size
            MASK_ID = self.conductor.MASK_TOKEN_ID
            
            t = jnp.linspace(0, 1, num_steps + 1)
            mask_schedule = jnp.cos(t * jnp.pi / 2)

            def loop_body(i, carry):
                tokens, key = carry
                
                stacked_tokens = jnp.concatenate([tokens, tokens], axis=0)
                stacked_emb = jnp.concatenate([cond_emb, uncond_emb], axis=0)
                logits_stacked = model_apply_fn({'params': params}, stacked_tokens, stacked_emb, train=False, method=self.conductor.get_features)[0]

                logits_cond, logits_uncond = logits_stacked[0:1], logits_stacked[1:2]
                cfg_logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
                
                key, sample_key = jax.random.split(key)
                sampled_tokens = jax.random.categorical(sample_key, cfg_logits)

                unknown_mask = (tokens == MASK_ID)
                probs = jax.nn.softmax(cfg_logits, axis=-1)
                confidence = jnp.max(probs, axis=-1)
                confidence = jnp.where(unknown_mask, confidence, -1.0) # only consider confidence of unknown tokens
                
                num_to_unmask = jnp.floor(L * (mask_schedule[i] - mask_schedule[i+1])).astype(jnp.int32)
                
                _, indices_to_unmask = jax.lax.top_k(confidence, num_to_unmask)
                
                mask_for_updates = jnp.zeros_like(tokens, dtype=jnp.bool_).at[jnp.arange(B)[:,None], indices_to_unmask].set(True)
                
                new_tokens = jnp.where(mask_for_updates, sampled_tokens, tokens)
                
                return new_tokens, key

            initial_tokens = jnp.full((B, L), MASK_ID, dtype=jnp.int32)
            final_tokens, _ = jax.lax.fori_loop(0, num_steps, loop_body, (initial_tokens, key))
            return final_tokens

        self._decode_fn = _non_autoregressive_cfg_decode_jit
        
        @partial(jax.jit, static_argnames=('resolution','patch_size'))
        def _render_fn(p1_params, path_params, resolution=512, patch_size=256):
            coords=jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
            coord_chunks=jnp.array_split(coords,(resolution**2)//(patch_size**2));pixels_list=[self.p1_model.apply({'params':p1_params},path_params,c,method=self.p1_model.decode)for c in coord_chunks]
            return jnp.concatenate(pixels_list,axis=1).reshape(path_params.shape[0],resolution,resolution,3)
        self._render_fn_jit = _render_fn
        self._decode_tokens_fn = jax.jit(lambda i: self.tokenizer.apply({'params':self.tok_params}, i, method=self.tokenizer.decode))

    def generate(self, prompt: str, seed: int, guidance_scale: float, decoding_steps: int):
        self.console.print(f"--- 🎨 Generating image for prompt: \"[italic yellow]{prompt}[/italic yellow]\" ---")
        self.console.print(f"-> Seed: {seed}, CFG Scale: {guidance_scale}, Steps: {decoding_steps}")
        key = jax.random.PRNGKey(seed)
        
        with torch.no_grad():
            cond_text_emb = self.clip_model.encode_text(clip.tokenize([prompt]).to(_clip_device)).cpu().numpy().astype(self.dtype)
        
        self.console.print("1/3: Generating token canvas (Non-Autoregressive)...")
        final_tokens = self._decode_fn(self.conductor.apply, self.cond_params, key, cond_text_emb, self.uncond_embedding, guidance_scale, decoding_steps)
        final_tokens.block_until_ready()
        
        self.console.print("2/3: Decoding tokens with Tokenizer...")
        grid_dim = self.tok_config['latent_grid_size'] // 4; token_grid = final_tokens.reshape(1, grid_dim, grid_dim)
        path_params = self._decode_tokens_fn(token_grid)

        self.console.print("3/3: Rendering final 512x512 image...")
        recon_batch = self._render_fn_jit(self.p1_params, path_params)
        recon_np = np.array(((recon_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        
        filename = f"GEN_{Path(self.args.basename).stem}_{prompt.replace(' ', '_')[:40]}_{seed}_cfg{guidance_scale:.1f}_s{decoding_steps}.png"
        Image.fromarray(recon_np).save(filename)
        self.console.print(f"✅ Image saved to [green]{filename}[/green]")





# =================================================================================================
# 7. MAIN EXECUTION BLOCK
# =================================================================================================

# [ADD] Place the new function definition before main() so it can be called.
def run_tokenizer_check(args):
    """Loads a trained tokenizer and reconstructs a single image to test its quality."""
    console = Console()
    console.print("--- 🔬 Running Tokenizer Sanity Check ---", style="bold yellow")

    # --- Load Tokenizer Config and Params ---
    try:
        tok_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_*_config.pkl"))
        tok_path_best = Path(str(tok_config_path).replace("_config.pkl", "_best.pkl"))
        if not tok_path_best.exists():
            tok_path_best = Path(str(tok_config_path).replace("_config.pkl", "_final.pkl"))
    except StopIteration:
        sys.exit(f"[FATAL] Could not find tokenizer config/model for basename '{args.basename}'.")
    
    with open(tok_config_path, 'rb') as f: tok_config = pickle.load(f)

    console.print(f"-> Loading tokenizer from [green]{tok_path_best}[/green]")
    with open(tok_path_best, 'rb') as f:
        tok_ckpt = pickle.load(f)
        tok_params = tok_ckpt.get('params', tok_ckpt.get('gen_params'))

    # --- Initialize Model ---
    dtype = jnp.float32 # Use high precision for check
    tokenizer = LatentTokenizerVQGAN(**tok_config, dtype=dtype)
    
    # --- JIT the Reconstruction Function ---
    @jax.jit
    def reconstruct(params, path_params_grid):
        # The __call__ method of the VQGAN performs a full encode-decode pass
        return tokenizer.apply({'params': params}, path_params_grid)['reconstructed_path_params']

    # --- Load and Preprocess Phase 1 Latents ---
    console.print("-> Loading Phase 1 AE to get input latents...", style="dim")
    try:
        p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
        p1_d_model = int(p1_path.stem.split('_')[-2].replace('d',''))
    except StopIteration:
        sys.exit(f"[FATAL] Could not find Phase 1 model for basename '{args.basename}'.")
    
    with open(p1_path, 'rb') as f: p1_params = pickle.load(f)['params']
    p1_modulator = PathModulator(tok_config['latent_grid_size'], 512, dtype=dtype)
    
    @jax.jit
    def get_path_params(params, image):
        return p1_modulator.apply({'params': params['modulator']}, image)

    # --- Process the Image ---
    console.print(f"-> Processing image: {args.image_path}")
    img = Image.open(args.image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    img_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
    img_batch = jnp.expand_dims(img_np, axis=0)

    # 1. Image -> Path Params (Phase 1 Encoder)
    path_params_grid = get_path_params(p1_params, img_batch)
    
    # 2. Path Params -> Reconstructed Path Params (Tokenizer Autoencoder)
    recon_path_params = reconstruct(tok_params, path_params_grid)
    
    # 3. Reconstructed Path Params -> Image (Phase 1 Decoder)
    console.print("-> Rendering final image from tokenizer's reconstruction...")
    p1_decoder_model = TopologicalCoordinateGenerator(p1_d_model, tok_config['latent_grid_size'], 512, dtype=dtype)
    
    @partial(jax.jit, static_argnames=('resolution','patch_size'))
    def render_image(params, path_params, resolution=512, patch_size=256):
        coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
        coord_chunks = jnp.array_split(coords, (resolution**2)//(patch_size**2))
        pixels_list = [p1_decoder_model.apply({'params': params}, path_params, c, method=p1_decoder_model.decode) for c in coord_chunks]
        return jnp.concatenate(pixels_list, axis=1).reshape(path_params.shape[0], resolution, resolution, 3)

    final_image_batch = render_image(p1_params, recon_path_params)
    
    # --- Save Output ---
    recon_np = np.array(((final_image_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
    
    # Create a comparison image
    comparison_img = Image.new('RGB', (1034, 512), (20, 20, 20))
    comparison_img.paste(img, (5, 0))
    comparison_img.paste(Image.fromarray(recon_np), (512 + 17, 0))
    
    comparison_img.save(args.output_path)
    console.print(f"✅ Sanity check complete. Comparison saved to [green]{args.output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Generative Framework (Project Chimera)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--basename', type=str, required=True, help="Base name for the model set (e.g., 'laion_ae'). Used to find/save all related files.")

    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 precision for training where applicable.")
    train_parser.add_argument('--seed', type=int, default=42)
    
    p_prep = subparsers.add_parser("prepare-paired-data", help="Pre-process (image, text) -> (latent, embedding).", parents=[base_parser])
    p_prep.add_argument('--data-dir', type=str, required=True); p_prep.add_argument('--d-model', type=int, required=True); p_prep.add_argument('--latent-grid-size', type=int, required=True); p_prep.add_argument('--batch-size', type=int, default=128)

    p_prep_tok = subparsers.add_parser("prepare-tokenizer-data", help="Pre-process images into latents for the tokenizer.", parents=[base_parser])
    p_prep_tok.add_argument('--data-dir', type=str, required=True); p_prep_tok.add_argument('--d-model', type=int, required=True); p_prep_tok.add_argument('--latent-grid-size', type=int, required=True); p_prep_tok.add_argument('--batch-size', type=int, default=128)

    p_tok = subparsers.add_parser("train-tokenizer", help="Train the Latent Tokenizer (VQ-GAN).", parents=[base_parser, train_parser])
    p_tok.add_argument('--data-dir', type=str, required=True); p_tok.add_argument('--d-model', type=int, required=True); p_tok.add_argument('--latent-grid-size', type=int, required=True)
    p_tok.add_argument('--epochs', type=int, default=100); p_tok.add_argument('--batch-size', type=int, default=128); p_tok.add_argument('--lr', type=float, default=3e-4)
    p_tok.add_argument('--eval-every', type=int, default=1000, help="Run validation and update preview every N steps.")
    p_tok.add_argument('--num-codes', type=int, default=3072); p_tok.add_argument('--code-dim', type=int, default=256)

    p_gen_eco = subparsers.add_parser("train-generational", help="Train the Spatiotemporal World Model with the Generational Ecosystem.", parents=[base_parser, train_parser])
    p_gen_eco.add_argument('--data-dir', type=str, required=True)
    p_gen_eco.add_argument('--latent-grid-size', type=int, required=True, help="Grid size of the Phase 1 AE latents (e.g., 96).")
    p_gen_eco.add_argument('--epochs', type=int, default=200)
    p_gen_eco.add_argument('--batch-size', type=int, default=4, help="Batch size for the entire generational pass.")
    p_gen_eco.add_argument('--lr', type=float, default=1e-4, help="Base learning rate for all agent Q-Controllers.")
    p_gen_eco.add_argument('--eval-every', type=int, default=500, help="Run async validation and preview every N global steps.")
    p_gen_eco.add_argument('--num-codes', type=int, default=3072)
    p_gen_eco.add_argument('--mask-ratio', type=float, default=0.5, help="Base mask ratio for the training objective.")
    p_gen_eco.add_argument('--vram-saver-mode', action='store_true', help="Enable VRAM saving optimizations, like smaller preview resolutions.")
    p_gen_eco.add_argument('--debug-fast-init', action='store_true', help="DEBUG: Skip tokenization and assume cache is valid for faster startup.")
    p_gen_eco.add_argument('--d-model-cond', type=int, default=768, help="[DEPRECATED] Model size is now defined by tiers.")
    p_gen_eco.add_argument('--num-layers', type=int, default=12, help="[DEPRECATED] Num layers is now defined by tiers.")
    p_gen_eco.add_argument('--num-heads', type=int, default=12, help="[DEPRECATED] Num heads is now defined by tiers.")
    
    inference_parser = argparse.ArgumentParser(add_help=False)
    p_gen = subparsers.add_parser("generate", help="Generate an image from a text prompt.", parents=[base_parser, inference_parser])
    p_gen.add_argument('--prompt', type=str, required=True)
    p_gen.add_argument('--seed', type=int, default=lambda: int(time.time()))
    p_gen.add_argument('--guidance-scale', type=float, default=4.0, help="Classifier-Free Guidance scale for masked models.")
    p_gen.add_argument('--decoding-steps', type=int, default=12, help="Number of iterative steps for non-autoregressive decoding.")
    
    p_check_tok = subparsers.add_parser("check-tokenizer", help="Reconstruct an image to check tokenizer quality.", parents=[base_parser])
    p_check_tok.add_argument('--image-path', type=str, required=True)
    p_check_tok.add_argument('--output-path', type=str, default="tokenizer_recon_check.png")
    
    args = parser.parse_args()
    
    if args.command in ["generate"]: args.seed = args.seed() if callable(args.seed) else args.seed
    
    if args.command == "train-generational":
        if args.latent_grid_size % 24 != 0:
            sys.exit(f"[FATAL] latent_grid_size ({args.latent_grid_size}) must be divisible by 24 for the spatiotemporal patcher.")

    if args.command == "prepare-paired-data": prepare_paired_data(args)
    elif args.command == "prepare-tokenizer-data": prepare_tokenizer-data(args)
    elif args.command == "train-tokenizer": TokenizerTrainer(args).train()
    elif args.command == "check-tokenizer": run_tokenizer_check(args)
    elif args.command == "train-generational": GenerationalConductorTrainer(args).train()
    elif args.command == "generate": Generator(args).generate(args.prompt, args.seed, args.guidance_scale, args.decoding_steps)   
    
if __name__ == "__main__":
    # [THE FIX] Add a top-level exception handler to log any crash
    try:
        main()
    except Exception as e:
        console = Console()
        console.print("\n[bold red]FATAL ERROR ENCOUNTERED[/bold red]")
        # Use Rich's traceback for pretty printing to the console
        console.print_exception(show_locals=False)
        # Also write the raw, unfiltered traceback to a log file
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write("A fatal error occurred. Full traceback:\n")
            f.write(traceback.format_exc())
        console.print("\n[yellow]Full traceback has been written to [bold]crash_log.txt[/bold][/yellow]")