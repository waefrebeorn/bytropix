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
import itertools
from dataclasses import dataclass, field, fields, replace
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
class InteractivityState:
    """A thread-safe class to hold shared state for interactive controls."""
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_prompt_change = 0
        self.sentinel_dampening_log_factor = -1.0
        self.shutdown_event = threading.Event()
        # --- [OPTIMIZATION] Profiler toggle removed ---
        self.force_save = False
        self.force_validate = False

    def get_and_reset_preview_change(self):
        with self.lock:
            change = self.preview_prompt_change
            self.preview_prompt_change = 0
            return change

    # --- [OPTIMIZATION] Profiler method removed ---

    def get_and_reset_force_save(self):
        with self.lock:
            save = self.force_save
            self.force_save = False
            return save

    def get_and_reset_force_validate(self):
        with self.lock:
            validate = self.force_validate
            self.force_validate = False
            return validate

    def update_sentinel_factor(self, direction):
        with self.lock:
            self.sentinel_dampening_log_factor = np.clip(self.sentinel_dampening_log_factor + direction * 0.5, -3.0, 0.0)

    def get_sentinel_factor(self):
        with self.lock:
            return 10**self.sentinel_dampening_log_factor

    def set_shutdown(self):
        self.shutdown_event.set()

def listen_for_keys(shared_state: InteractivityState):
    """
    [UPGRADED] A cross-platform, non-blocking key listener that now handles
    graceful shutdown (q, Ctrl+C), saving (s), and validation (v).
    """
    if platform.system() == "Windows":
        while not shared_state.shutdown_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x03' or key == b'q':  # Ctrl+C or 'q'
                    shared_state.set_shutdown()
                    break
                # --- [OPTIMIZATION] Profiler key 'p' removed ---
                elif key == b's':
                    with shared_state.lock: shared_state.force_save = True
                elif key == b'v':
                    with shared_state.lock: shared_state.force_validate = True
                elif key == b'\xe0': # Arrow key prefix
                    arrow = msvcrt.getch()
                    if arrow == b'K': shared_state.preview_prompt_change = -1 # Left
                    elif arrow == b'M': shared_state.preview_prompt_change = 1 # Right
            time.sleep(0.05)
    else: # Linux/macOS
        fd = sys.stdin.fileno(); old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not shared_state.shutdown_event.is_set():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char == '\x03' or char == 'q': # Ctrl+C or 'q'
                        shared_state.set_shutdown()
                        break
                    # --- [OPTIMIZATION] Profiler key 'p' removed ---
                    elif char == 's':
                        with shared_state.lock: shared_state.force_save = True
                    elif char == 'v':
                        with shared_state.lock: shared_state.force_validate = True
                    elif char == '\x1b': # ESC sequence
                        next_chars = sys.stdin.read(2)
                        if next_chars == '[C':
                             with shared_state.lock: shared_state.preview_prompt_change = 1
                        elif next_chars == '[D':
                             with shared_state.lock: shared_state.preview_prompt_change = -1
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
           
            
            
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
        self.q_table_size = int(config["q_table_size"])
        self.num_actions = int(config["num_lr_actions"])
        # [THE FIX] Convert to a tuple to ensure it's hashable and static
        self.action_factors = tuple(config["lr_change_factors"])
        self.q_table = np.zeros((self.q_table_size, self.num_actions), dtype=np.float32)
        self.learning_rate_q = float(config["learning_rate_q"])
        self.discount_factor_q = float(config["discount_factor_q"])
        self.value_min = float(config["lr_min"])
        self.value_max = float(config["lr_max"])
        self.metric_history = deque(maxlen=int(config["metric_history_len"]))
        self.metric_min = float(config["loss_min"])
        self.metric_max = float(config["loss_max"])
        self.last_action_idx: Optional[int] = None
        self.last_state_idx: Optional[int] = None
        self.initial_exploration_rate = float(config["exploration_rate_q"])
        self.exploration_rate_q = self.initial_exploration_rate
        self.min_exploration_rate = float(config["min_exploration_rate"])
        self.exploration_decay = float(config["exploration_decay"])
        self.status: str = "STARTING"
        self.last_reward: float = 0.0
        self.trend_window = int(config["trend_window"])
        self.trend_history = deque(maxlen=self.trend_window)
        self.improve_threshold = float(config["improve_threshold"])
        self.regress_threshold = float(config["regress_threshold"])
        self.regress_penalty = float(config["regress_penalty"])
        self.stagnation_penalty = float(config["stagnation_penalty"])
        self.warmup_steps = int(config.get("warmup_steps", 0))
        self.warmup_start_val = float(config.get("warmup_lr_start"))
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
        
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QController:
    # --- Dynamic State (Children) ---
    q_table: chex.Array
    metric_history: chex.Array
    trend_history: chex.Array
    current_value: jnp.ndarray
    exploration_rate: jnp.ndarray
    step_count: jnp.ndarray
    last_reward: jnp.ndarray
    status_code: jnp.ndarray

    # --- Static Config (Aux Data) ---
    initial_value: float = field(metadata={"static": True})
    warmup_start_val: float = field(metadata={"static": True})
    q_table_size: int = field(metadata={"static": True})
    num_actions: int = field(metadata={"static": True})
    # [THE DEFINITIVE FIX] This must be a hashable Tuple, not a JAX array.
    action_factors: Tuple[float, ...] = field(metadata={"static": True})
    learning_rate_q: float = field(metadata={"static": True})
    discount_factor_q: float = field(metadata={"static": True})
    value_min: float = field(metadata={"static": True})
    value_max: float = field(metadata={"static": True})
    metric_min: float = field(metadata={"static": True})
    metric_max: float = field(metadata={"static": True})
    min_exploration_rate: float = field(metadata={"static": True})
    exploration_decay: float = field(metadata={"static": True})
    trend_window: int = field(metadata={"static": True})
    improve_threshold: float = field(metadata={"static": True})
    regress_threshold: float = field(metadata={"static": True})
    regress_penalty: float = field(metadata={"static": True})
    stagnation_penalty: float = field(metadata={"static": True})
    warmup_steps: int = field(metadata={"static": True})

    def tree_flatten(self):
        children = (self.q_table, self.metric_history, self.trend_history, self.current_value,
                    self.exploration_rate, self.step_count, self.last_reward, self.status_code)
        aux_data = (self.initial_value, self.warmup_start_val, self.q_table_size, self.num_actions,
                    self.action_factors, self.learning_rate_q, self.discount_factor_q,
                    self.value_min, self.value_max, self.metric_min, self.metric_max,
                    self.min_exploration_rate, self.exploration_decay, self.trend_window,
                    self.improve_threshold, self.regress_threshold, self.regress_penalty,
                    self.stagnation_penalty, self.warmup_steps)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
        
        
@jax.jit
def q_controller_choose_action(controller: QController, key: chex.PRNGKey) -> Tuple[jnp.ndarray, QController, jnp.ndarray]:
    """Pure JAX function to choose an action and update the step count."""
    
    def warmup_action():
        alpha = controller.step_count / controller.warmup_steps
        new_value = controller.warmup_start_val * (1 - alpha) + controller.initial_value * alpha
        new_controller = replace(controller, current_value=new_value, step_count=controller.step_count + 1, status_code=jnp.array(0))
        return new_value, new_controller, jnp.array(-1)

    def regular_action():
        metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(controller.metric_history, -5, 5))
        state_idx = jnp.clip(
            ((metric_mean - controller.metric_min) / ((controller.metric_max - controller.metric_min) / controller.q_table_size)).astype(jnp.int32),
            0, controller.q_table_size - 1
        )
        
        explore_key, action_key = jax.random.split(key)
        
        def explore():
            return jax.random.randint(action_key, (), 0, controller.num_actions)
        
        def exploit():
            return jnp.argmax(controller.q_table[state_idx])
            
        action_idx = jax.lax.cond(jax.random.uniform(explore_key) < controller.exploration_rate, explore, exploit)
        
        # --- [THE DEFINITIVE FIX] ---
        # Cannot index a static tuple with a dynamic tracer. Use lax.switch.
        selected_factor = jax.lax.switch(
            action_idx,
            [lambda: f for f in controller.action_factors]
        )
        new_value = jnp.clip(controller.current_value * selected_factor, controller.value_min, controller.value_max)
        # -----------------------------
        
        new_controller = replace(controller, current_value=new_value, step_count=controller.step_count + 1)
        return new_value, new_controller, action_idx

    return jax.lax.cond(controller.step_count <= controller.warmup_steps, warmup_action, regular_action)
    
@jax.jit
def q_controller_update(controller: QController, metric_value: float, last_action_idx: int) -> QController:
    """Pure JAX function to update the Q-table based on a new metric."""
    
    new_metric_history = jnp.roll(controller.metric_history, -1).at[-1].set(metric_value)
    new_trend_history = jnp.roll(controller.trend_history, -1).at[-1].set(metric_value)
    
    def no_update():
        # [THE FIX] Use dataclasses.replace instead of ._replace
        return replace(controller, metric_history=new_metric_history, trend_history=new_trend_history)

    def perform_update():
        x = jnp.arange(controller.trend_window)
        y = new_trend_history
        A = jnp.vstack([x, jnp.ones(len(x))]).T
        slope, _ = jnp.linalg.lstsq(A, y, rcond=None)[0]
        
        reward = jax.lax.cond(
            slope < -controller.improve_threshold,
            lambda: abs(slope) * 1000.0,
            lambda: jax.lax.cond(
                slope > controller.regress_threshold,
                lambda: -abs(slope) * 1000.0 - controller.regress_penalty,
                lambda: controller.stagnation_penalty
            )
        )
        
        status_code = jax.lax.cond(
            slope < -controller.improve_threshold, lambda: 1,
            lambda: jax.lax.cond(slope > controller.regress_threshold, lambda: 3, lambda: 2)
        )
        
        old_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(controller.metric_history, -5, 5))
        last_state_idx = jnp.clip(
            ((old_metric_mean - controller.metric_min) / ((controller.metric_max - controller.metric_min) / controller.q_table_size)).astype(jnp.int32),
            0, controller.q_table_size - 1
        )
        
        new_metric_mean = jnp.mean(jax.lax.dynamic_slice_in_dim(new_metric_history, -5, 5))
        next_state_idx = jnp.clip(
            ((new_metric_mean - controller.metric_min) / ((controller.metric_max - controller.metric_min) / controller.q_table_size)).astype(jnp.int32),
            0, controller.q_table_size - 1
        )

        current_q = controller.q_table[last_state_idx, last_action_idx]
        max_next_q = jnp.max(controller.q_table[next_state_idx])
        new_q = current_q + controller.learning_rate_q * (reward + controller.discount_factor_q * max_next_q - current_q)
        
        new_q_table = controller.q_table.at[last_state_idx, last_action_idx].set(new_q)
        
        new_exploration_rate = jnp.maximum(controller.min_exploration_rate, controller.exploration_rate * controller.exploration_decay)
        
        # [THE FIX] Use dataclasses.replace instead of ._replace
        return replace(
            controller,
            q_table=new_q_table,
            metric_history=new_metric_history,
            trend_history=new_trend_history,
            exploration_rate=new_exploration_rate,
            last_reward=reward,
            status_code=status_code
        )
        
    can_update = (controller.step_count > controller.warmup_steps) & \
                 (controller.step_count > controller.trend_window) & \
                 (last_action_idx >= 0)
                 
    return jax.lax.cond(can_update, perform_update, no_update)


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
# ATTENTION MECH
# ==============================================================================


class StandardAttention(nn.Module):
    """
    [DEFINITIVELY CORRECTED v3] The internal layer attributes are now marked
    with `field(init=False)` to prevent the dataclass transformer from adding
    them to the __init__ method, resolving the TypeError at class definition.
    """
    # --- Constructor Arguments (Processed by dataclasses) ---
    num_heads: int
    d_model: int
    dtype: Any = jnp.float32

    # --- Internal Module State (Ignored by __init__) ---
    # [THE FIX] Mark these as `init=False` so they don't become constructor arguments.
    q_proj: nn.Dense = field(init=False)
    k_proj: nn.Dense = field(init=False)
    v_proj: nn.Dense = field(init=False)
    out_proj: nn.Dense = field(init=False)

    def setup(self):
        """
        Layers are defined as instance attributes. This is now fully compatible
        with the dataclass-generated __init__ method.
        """
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = nn.Dense(self.d_model, name="query", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        self.k_proj = nn.Dense(self.d_model, name="key", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        self.v_proj = nn.Dense(self.d_model, name="value", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        self.out_proj = nn.Dense(self.d_model, name="out", dtype=self.dtype, kernel_init=initializers.xavier_uniform())

    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, 
                 k_context: Optional[jnp.ndarray] = None, 
                 v_context: Optional[jnp.ndarray] = None,
                 mask: Optional[jnp.ndarray] = None):
        
        is_self_attn = context is None and k_context is None
        B, L_q, _ = x.shape
        head_dim = self.d_model // self.num_heads

        q = self.q_proj(x)
        
        if is_self_attn:
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            if k_context is not None and v_context is not None:
                k, v = k_context, v_context
            elif context is not None:
                k = self.k_proj(context)
                v = self.v_proj(context)
            else:
                raise ValueError("Cross-attention requires either a context or a k/v_context cache.")
        
        L_kv = k.shape[1]
        
        q_heads = q.reshape(B, L_q, self.num_heads, head_dim)
        k_heads = k.reshape(B, L_kv, self.num_heads, head_dim)
        v_heads = v.reshape(B, L_kv, self.num_heads, head_dim)
        
        attn_output = dot_product_attention(q_heads, k_heads, v_heads, mask=mask, dtype=self.dtype)
        
        return self.out_proj(attn_output.reshape(B, L_q, self.d_model))

class StatelessAttention(nn.Module):
    """
    [THE FIX] A simplified, JIT-safe attention mechanism specifically for
    cross-attention with a pre-computed cache. It has no internal conditional
    logic, making it robust to JAX tracing during inference.
    """
    num_heads: int
    d_model: int
    dtype: Any = jnp.float32

    # Layers are defined here but used by the TransformerBlock to create caches
    q_proj: nn.Dense = field(init=False)
    k_proj: nn.Dense = field(init=False)
    v_proj: nn.Dense = field(init=False)
    out_proj: nn.Dense = field(init=False)

    def setup(self):
        """Initializes all projection layers. The k_proj and v_proj layers
        will be called externally by the TransformerBlock to create the cache."""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = nn.Dense(self.d_model, name="query", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        self.k_proj = nn.Dense(self.d_model, name="key", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        self.v_proj = nn.Dense(self.d_model, name="value", dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        self.out_proj = nn.Dense(self.d_model, name="out", dtype=self.dtype, kernel_init=initializers.xavier_uniform())

    def __call__(self, x: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None):
        """
        Performs attention using an explicitly provided key and value tensor.
        This method is stateless and contains no Python branching.
        """
        B, L_q, _ = x.shape
        _, L_kv, _ = k.shape
        head_dim = self.d_model // self.num_heads

        # Query is always computed from the main input 'x'.
        q_heads = self.q_proj(x).reshape(B, L_q, self.num_heads, head_dim)
        
        # K and V are passed in directly and just need reshaping.
        k_heads = k.reshape(B, L_kv, self.num_heads, head_dim)
        v_heads = v.reshape(B, L_kv, self.num_heads, head_dim)
        
        attn_output = dot_product_attention(q_heads, k_heads, v_heads, mask=mask, dtype=self.dtype)
        
        return self.out_proj(attn_output.reshape(B, L_q, self.d_model))
   
class TransformerBlock(nn.Module):
    """
    [UPGRADED & ROBUST] The transformer block now features a third attention mechanism,
    the "Synaptic Bridge," and is robust to its absence, allowing it to be used
    by both base conductors and the advanced Polisher.
    """
    num_heads: int; d_model: int; num_positions: int; dtype: Any = jnp.float32

    def setup(self):
        self.ln1 = nn.LayerNorm(dtype=self.dtype)
        self.sa = StandardAttention(
            num_heads=self.num_heads, d_model=self.d_model, dtype=self.dtype, name='sa'
        )
        self.ln2 = nn.LayerNorm(dtype=self.dtype)
        self.ca = StatelessAttention(
            num_heads=self.num_heads, d_model=self.d_model, dtype=self.dtype, name='ca'
        )
        self.ln_bridge = nn.LayerNorm(dtype=self.dtype)
        self.sa_bridge = StatelessAttention(
            num_heads=self.num_heads, d_model=self.d_model, dtype=self.dtype, name='sa_bridge'
        )
        self.ln_mlp = nn.LayerNorm(dtype=self.dtype)
        
        self.mlp_w1_raw = self.param('mlp_w1_raw', initializers.xavier_uniform(), (self.d_model, self.d_model * 4), self.dtype)
        self.mlp_b1 = self.param('mlp_b1', initializers.zeros, (self.d_model * 4,), self.dtype)
        
        self.mlp_w2_raw = self.param('mlp_w2_raw', initializers.xavier_uniform(), (self.d_model * 4, self.d_model), self.dtype)
        self.mlp_b2 = self.param('mlp_b2', initializers.zeros, (self.d_model,), self.dtype)

    def __call__(self, x, context, mask, k_context_cache, v_context_cache, k_bridge_cache=None, v_bridge_cache=None):
        # 1. Self-attention (internal thought)
        x = x + self.sa(self.ln1(x), mask=mask)

        # 2. Cross-attention to text prompt (guidance)
        x = x + self.ca(x=self.ln2(x), k=k_context_cache, v=v_context_cache, mask=None)

        # 3. [THE FIX] Synaptic Bridge attention is now optional and only runs if caches are provided.
        # This allows the same block to be used by base conductors (who pass None) and the Polisher.
        if k_bridge_cache is not None and v_bridge_cache is not None:
            x = x + self.sa_bridge(x=self.ln_bridge(x), k=k_bridge_cache, v=v_bridge_cache, mask=None)

        # 4. MLP block (processing)
        ffn_input = self.ln_mlp(x)
        w1 = (self.mlp_w1_raw + self.mlp_w1_raw.T) / 2 if self.d_model == self.d_model * 4 else self.mlp_w1_raw
        w2 = (self.mlp_w2_raw + self.mlp_w2_raw.T) / 2 if self.d_model * 4 == self.d_model else self.mlp_w2_raw
        h = nn.gelu(ffn_input @ w1 + self.mlp_b1)
        h = h @ w2 + self.mlp_b2
        
        x = x + h
        return x



class BaseGenerativeConductor(nn.Module):
    """A base class to hold all common logic, avoiding code duplication."""
    num_codes: int
    d_model: int
    num_heads: int
    num_layers: int
    dtype: Any

    depth_size: int = 4
    height_size: int = 24
    width_size: int = 6
    
    depth_patch_size: int = 2
    height_patch_size: int = 4
    width_patch_size: int = 2
    clip_dim: int = 512

    @property
    def MASK_TOKEN_ID(self):
        return self.num_codes + 1

    @property
    def vocab_size(self):
        return self.num_codes + 2

    def setup(self):
        self.grid_z = self.depth_size // self.depth_patch_size
        self.grid_y = self.height_size // self.height_patch_size
        self.grid_x = self.width_size // self.width_patch_size
        
        self.num_patches = self.grid_z * self.grid_y * self.grid_x
        self.patch_dim = self.depth_patch_size * self.height_patch_size * self.width_patch_size

        self.uncond_embedding = self.param('uncond_embedding', nn.initializers.normal(0.02), (1, self.clip_dim), self.dtype)
        self.token_embedding = nn.Embed(self.vocab_size, self.d_model, name='token_embedding', dtype=self.dtype)
        self.patch_projection = nn.Dense(self.d_model, name='patch_projection', dtype=self.dtype)
        
        self.input_norm = nn.LayerNorm(name='input_norm', dtype=self.dtype)
        
        # --- [THE SMARTER BROADCASTING FIX] ---
        # Create learnable embeddings to explicitly signal the data's origin (tier).
        # This provides crucial context to the Polisher's synaptic bridge attention.
        self.kid_tier_embedding = self.param('kid_tier_embedding', nn.initializers.normal(0.02), (1, 1, self.d_model), self.dtype)
        self.student_tier_embedding = self.param('student_tier_embedding', nn.initializers.normal(0.02), (1, 1, self.d_model), self.dtype)
        # ---

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

    def _embed_and_finalize(self, x_emb, text_emb):
        x = x_emb.reshape(-1, self.num_patches, self.patch_dim * self.d_model)
        x = self.patch_projection(x)
        x = self.input_norm(x)
        
        pos_embed_3d = (self.pos_embed_z[:, None, None, :] + self.pos_embed_y[None, :, None, :] + self.pos_embed_x[None, None, :, :]).reshape(self.num_patches, self.d_model)
        x = x + pos_embed_3d[None, :, :]
        ctx = self.text_projection(text_emb)[:, None, :]
        return x, ctx

    def _prepare_inputs_float(self, tokens_3d_float, text_emb):
        x_spat = tokens_3d_float.reshape(-1, self.grid_z, self.depth_patch_size, self.grid_y, self.height_patch_size, self.grid_x, self.width_patch_size, self.vocab_size)
        x_spat = x_spat.transpose(0, 1, 3, 5, 2, 4, 6, 7)
        x_patch = x_spat.reshape(-1, self.num_patches, self.patch_dim, self.vocab_size)
        x_emb = x_patch @ self.token_embedding.embedding
        return self._embed_and_finalize(x_emb, text_emb)
    
    def _prepare_bridge_context(self, kid_ste, student_ste):
        """Prepares the concatenated Key/Value cache for the Synaptic Bridge."""
        if hasattr(self, 'uncond_embedding'):
            dummy_emb = self.uncond_embedding
        else:
            dummy_emb = jnp.zeros((1, self.clip_dim), dtype=self.dtype)

        kid_emb = self._prepare_inputs_float(kid_ste, dummy_emb)[0]
        student_emb = self._prepare_inputs_float(student_ste, dummy_emb)[0]
        
        # --- [THE SMARTER BROADCASTING FIX] ---
        # Add the tier-specific embedding to each patch sequence. This explicitly
        # labels the data for the Polisher.
        kid_emb = kid_emb + self.kid_tier_embedding
        student_emb = student_emb + self.student_tier_embedding
        # ---
        
        bridge_context = jnp.concatenate([kid_emb, student_emb], axis=1)
        return bridge_context
        
    def __call__(self, input_tokens, text_emb, train: bool = True):
        x, ctx = self._prepare_inputs_float(input_tokens, text_emb)
        kv_caches = [(block.ca.k_proj(ctx), block.ca.v_proj(ctx)) for block in self.blocks]
        for i, block in enumerate(self.blocks):
            k_cache, v_cache = kv_caches[i]
            x = block(x, ctx, mask=None, k_context_cache=k_cache, v_context_cache=v_cache, k_bridge_cache=None, v_bridge_cache=None)
        x = self.norm(x)
        x = self.output_head(x)
        
        logits_3d = x.reshape(-1, self.grid_z, self.grid_y, self.grid_x, self.depth_patch_size, self.height_patch_size, self.width_patch_size, self.vocab_size)
        logits_3d = logits_3d.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        final_logits = logits_3d.reshape(-1, self.depth_size, self.height_size, self.width_size, self.vocab_size)
        return final_logits
        
    def get_features(self, tokens_3d, text_emb, train: bool = True):
        rngs = {}
        if train:
            rngs['dropout'] = self.make_rng('dropout')

        x, ctx = self._prepare_inputs_float(tokens_3d, text_emb)
        kv_caches = [(block.ca.k_proj(ctx), block.ca.v_proj(ctx)) for block in self.blocks]

        for i, block in enumerate(self.blocks):
            k_cache, v_cache = kv_caches[i]
            x = block(x, ctx, mask=None, k_context_cache=k_cache, v_context_cache=v_cache, k_bridge_cache=None, v_bridge_cache=None)
            
        x_norm = self.norm(x)
        projected_patch_embeddings = self.mmd_projection(x_norm)
        x_out = self.output_head(x_norm)

        logits_3d = x_out.reshape(-1, self.grid_z, self.grid_y, self.grid_x, self.depth_patch_size, self.height_patch_size, self.width_patch_size, self.vocab_size)
        logits_3d = logits_3d.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        final_logits = logits_3d.reshape(-1, self.depth_size, self.height_size, self.width_size, self.vocab_size)
        return final_logits, projected_patch_embeddings

    def init_forward_pass(self, tokens_3d, text_emb, train: bool = True):
        _ = self.__call__(tokens_3d, text_emb, train=train)
        _, _ = self.get_features(tokens_3d, text_emb, train=train)
        return jnp.array(0.0)      
        
# Define separate, explicit classes for each tier.
class KidConductor(BaseGenerativeConductor): pass
class StudentConductor(BaseGenerativeConductor): pass
class PolisherConductor(BaseGenerativeConductor): pass

# --- [THE BEST OF BOTH WORLDS] ---
# Create a type alias for backwards compatibility with StaticAgentData.
GenerativeConductor = BaseGenerativeConductor



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
        # --- [THE FIX] REMOVE these two lines ---
        # self.should_shutdown = False
        # signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        
        # This instance will now be the single source of truth for interactivity
        self.interactive_state = InteractivityState()
        
        self.num_devices = jax.local_device_count()
        self.loss_history = deque(maxlen=200)

        # Q-Controller is now handled by the specific trainer subclass
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
    [UPGRADED] with a full suite of perceptual loss diagnostics and a JAX-native Q-Controller.
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # The 'use_q_controller' argument is now specific to this trainer
        self.use_q_controller = getattr(args, 'use_q_controller', True)
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.generator_model = LatentTokenizerVQGAN(args.num_codes, args.code_dim, args.latent_grid_size, self.dtype)
        self.discriminator_model = PatchDiscriminator(dtype=self.dtype)
        
        self.perceptual_loss_fn = JAXMultiMetricPerceptualLoss()

        # GAN Balancer & Lockout State
        self.d_loss_ema = 0.5; self.d_loss_ema_alpha = 0.05
        self.d_loss_target_min = 0.3; self.d_loss_target_max = 0.6
        self.d_lockout_steps = 0; self.d_lockout_threshold = 0.009
        self.d_lockout_duration = 5

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
        
        # --- [Q-CTRL REWORK] Initialize both Python (UI) and JAX (compute) controllers ---
        self.q_controller_jax = None
        if self.use_q_controller:
            is_finetune = getattr(self.args, 'finetune', False)
            q_config = Q_CONTROLLER_CONFIG_FINETUNE if is_finetune else Q_CONTROLLER_CONFIG_NORMAL
            self.q_controller = JaxHakmemQController(initial_value=self.args.lr, config=q_config)
            
            self.q_controller_jax = QController(
                q_table=jnp.array(self.q_controller.q_table),
                metric_history=jnp.zeros(self.q_controller.metric_history.maxlen, dtype=jnp.float32),
                trend_history=jnp.zeros(self.q_controller.trend_history.maxlen, dtype=jnp.float32),
                current_value=jnp.array(self.q_controller.current_value),
                exploration_rate=jnp.array(self.q_controller.exploration_rate_q),
                step_count=jnp.array(self.q_controller._step_count),
                last_reward=jnp.array(self.q_controller.last_reward),
                status_code=jnp.array(0),
                initial_value=self.q_controller.initial_value,
                warmup_start_val=self.q_controller.warmup_start_val,
                q_table_size=self.q_controller.q_table_size,
                num_actions=self.q_controller.num_actions,
                action_factors=jnp.array(self.q_controller.action_factors),
                learning_rate_q=self.q_controller.learning_rate_q,
                discount_factor_q=self.q_controller.discount_factor_q,
                value_min=self.q_controller.value_min,
                value_max=self.q_controller.value_max,
                metric_min=self.q_controller.metric_min,
                metric_max=self.q_controller.metric_max,
                min_exploration_rate=self.q_controller.min_exploration_rate,
                exploration_decay=self.q_controller.exploration_decay,
                trend_window=self.q_controller.trend_window,
                improve_threshold=self.q_controller.improve_threshold,
                regress_threshold=self.q_controller.regress_threshold,
                regress_penalty=self.q_controller.regress_penalty,
                stagnation_penalty=self.q_controller.stagnation_penalty,
                warmup_steps=self.q_controller.warmup_steps
            )

        self.interactive_state = InteractivityState()
        self.ui_lock = threading.Lock()
        self.param_count = 0
        
        self.hist_len = 400
        self.g_loss_hist = deque(maxlen=self.hist_len); self.d_loss_hist = deque(maxlen=self.hist_len)
        self.l1_hist = deque(maxlen=self.hist_len); self.ssim_hist = deque(maxlen=self.hist_len)
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
                Layout(name="pid_controller", ratio=1, minimum_size=12)
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
                q_tbl.add_row("Base LR", f"[{color}]{self.q_controller.current_value:.2e}[/] {status_emoji}")
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
        # --- [THE FIX] The key listener is started using the unified interactive_state ---
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True)
        key_listener_thread.start()
        
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
        gen_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr, b1=0.5, b2=0.9)
        disc_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr * 0.8, b1=0.5, b2=0.9)
        dummy_input = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), self.dtype)
        gen_params = self.generator_model.init(g_key, dummy_input)['params']; disc_params = self.discriminator_model.init(d_key, dummy_input)['params']
        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, gen_params, 0)
        gen_state = TrainState.create(apply_fn=self.generator_model.apply, params=gen_params, tx=gen_optimizer)
        disc_state = TrainState.create(apply_fn=self.discriminator_model.apply, params=disc_params, tx=disc_optimizer)
        states = GANTrainStates(generator=gen_state, discriminator=disc_state)
        
        current_q_state = self.q_controller_jax

        ckpt_path = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_final.pkl"); ckpt_path_best = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_gan_best.pkl")
        best_val_loss, start_epoch, global_step = float('inf'), 0, 0
        
        last_metrics = { 'g_loss': 0.0, 'd_loss': 0.5, 'l1': 0.05, 'vq': 0.1, 'moment': 0.2, 'fft': 0.5, 'autocorr': 0.1, 'edge': 0.1, 'color_cov': 0.05, 'ssim': 0.02, 'varentropy': 0.0 }

        if ckpt_path.exists():
            console.print(f"--- Resuming training state from: [green]{ckpt_path}[/green] ---")
            with open(ckpt_path, 'rb') as f: ckpt = pickle.load(f)
            gen_opt_state, disc_opt_state = ckpt['gen_opt_state'], ckpt['disc_opt_state']
            start_epoch = ckpt.get('epoch', 0)
            last_metrics.update(ckpt.get('last_metrics', {}))
            global_step = ckpt.get('global_step', start_epoch * steps_per_epoch)
            if self.q_controller and 'q_controller_jax' in ckpt:
                current_q_state = ckpt['q_controller_jax']
                console.print(f"🤖 JAX Q-Controller state restored.")
            if 'pid_controller_state' in ckpt: self.lambda_controller.load_state_dict(ckpt['pid_controller_state']); console.print(f"🧠 PID Controller state restored.")
            if ckpt_path_best.exists():
                console.print(f"--- Loading BEST generator weights from: [bold magenta]{ckpt_path_best}[/bold magenta] ---")
                with open(ckpt_path_best, 'rb') as f_best: best_ckpt = pickle.load(f_best); gen_params = best_ckpt['params']; best_val_loss = best_ckpt.get('val_loss', float('inf'))
            else: gen_params = ckpt['gen_params']
            states = GANTrainStates(generator=states.generator.replace(params=gen_params, opt_state=gen_opt_state), discriminator=states.discriminator.replace(params=ckpt['disc_params'], opt_state=disc_opt_state))
            console.print(f"✅ Resuming session from epoch {start_epoch + 1}, step {global_step}. Best val loss: {best_val_loss:.4f}")
        
        @partial(jax.jit, static_argnames=('gen_apply_fn', 'disc_apply_fn', 'd_is_locked_out', 'use_q_controller'))
        def train_step(states, q_controller_state, batch, key, lambdas, d_loss_ema, g_lr_mult, d_lr_mult, gen_apply_fn, disc_apply_fn, d_is_locked_out: bool, use_q_controller: bool):
            q_key, loss_key = jax.random.split(key)
            
            def get_q_lr():
                lr, new_q_state, action_idx = q_controller_choose_action(q_controller_state, q_key)
                return lr, new_q_state, action_idx
            def get_default_lr():
                return jnp.array(self.args.lr), q_controller_state, jnp.array(-1, dtype=jnp.int32)

            base_lr, new_q_state_after_action, action_idx = jax.lax.cond(
                use_q_controller, get_q_lr, get_default_lr
            )

            (lambda_l1, lambda_vq, lambda_adv, lambda_stink, 
             lambda_moment, lambda_fft, lambda_autocorr, lambda_edge, 
             lambda_color_cov, lambda_ssim) = lambdas

            def generator_loss_fn(p):
                gen_output = gen_apply_fn({'params': p}, batch)
                recon = gen_output['reconstructed_path_params']
                l1_loss = jnp.mean(jnp.abs(batch - recon))
                vq_loss = gen_output['vq_loss']
                adv_loss = jnp.mean((disc_apply_fn({'params': states.discriminator.params}, recon) - 1)**2)
                perceptual_losses = self.perceptual_loss_fn(batch, recon, loss_key)
                z_e = gen_output['pre_quant_latents']
                _, varent = ent_varent(z_e.reshape(-1, z_e.shape[-1]))
                varentropy_loss = jnp.mean(varent)
                total_loss = (lambda_l1 * l1_loss) + (lambda_vq * vq_loss) + (lambda_adv * adv_loss) + (lambda_stink * varentropy_loss) + (lambda_moment * perceptual_losses['moment']) + (lambda_fft * perceptual_losses['fft']) + (lambda_autocorr * perceptual_losses['autocorr']) + (lambda_edge * perceptual_losses['edge']) + (lambda_color_cov * perceptual_losses['color_cov']) + (lambda_ssim * perceptual_losses['ssim'])
                all_metrics = {'l1': l1_loss, 'vq': vq_loss, 'adv': adv_loss, 'varentropy': varentropy_loss}
                all_metrics.update(perceptual_losses)
                return total_loss, all_metrics

            (g_loss_total, metrics), g_grads = jax.value_and_grad(generator_loss_fn, has_aux=True)(states.generator.params)
            
            # Update LR in optimizer state before applying gradients
            new_gen_opt_state = states.generator.opt_state
            new_gen_opt_state.hyperparams['learning_rate'] = base_lr * g_lr_mult
            new_gen_state = states.generator.replace(opt_state=new_gen_opt_state).apply_gradients(grads=g_grads)
            
            def discriminator_loss_fn(disc_params):
                def compute_d_loss():
                    recon = gen_state.apply_fn({'params': new_gen_state.params}, batch)['reconstructed_path_params']
                    loss_real = jnp.mean((disc_apply_fn({'params': disc_params}, batch) - 1)**2)
                    loss_fake = jnp.mean(disc_apply_fn({'params': disc_params}, jax.lax.stop_gradient(recon))**2)
                    return ((loss_real + loss_fake) * 0.5).astype(jnp.float32)
                return jax.lax.cond(d_is_locked_out, lambda: 0.0, compute_d_loss)
            
            d_loss, d_grads = jax.value_and_grad(discriminator_loss_fn)(states.discriminator.params)
            new_disc_opt_state = states.discriminator.opt_state
            new_disc_opt_state.hyperparams['learning_rate'] = (base_lr * 0.8) * d_lr_mult
            new_disc_state = states.discriminator.replace(opt_state=new_disc_opt_state).apply_gradients(grads=d_grads)
            
            def update_q_state():
                return q_controller_update(new_q_state_after_action, g_loss_total, action_idx)
            def identity_q_state():
                return new_q_state_after_action
            final_q_state = jax.lax.cond(use_q_controller, update_q_state, identity_q_state)

            metrics['g_loss'] = g_loss_total; metrics['d_loss'] = d_loss
            return GANTrainStates(generator=new_gen_state, discriminator=new_disc_state), metrics, final_q_state

        @partial(jax.jit, static_argnames=('apply_fn',))
        def eval_step(gen_params, apply_fn, batch, key):
            out = apply_fn({'params': gen_params}, batch)
            l1_loss = jnp.mean(jnp.abs(out['reconstructed_path_params']-batch))
            perceptual_losses = self.perceptual_loss_fn(out['reconstructed_path_params'], batch, key)
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
        
        dummy_lambda_dict = self.lambda_controller(last_metrics)
        dummy_lambdas = tuple(dummy_lambda_dict.values())
        states, _, _ = train_step(states, current_q_state, dummy_batch, compile_key, dummy_lambdas, 0.5, 1.0, 1.0, self.generator_model.apply, self.discriminator_model.apply, d_is_locked_out=False, use_q_controller=self.use_q_controller)
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
                    # --- [THE FIX] Unified shutdown check ---
                    if self.interactive_state.shutdown_event.is_set(): break
                    rng.shuffle(train_indices_shuffler)
                    for step_in_epoch in range(steps_per_epoch):
                        # --- [THE FIX] Unified shutdown check ---
                        if self.interactive_state.shutdown_event.is_set(): break
                        
                        start_idx = step_in_epoch * self.args.batch_size; end_idx = start_idx + self.args.batch_size
                        if start_idx >= len(train_indices_shuffler): continue
                        batch_indices = train_indices_shuffler[start_idx:end_idx]; train_batch = jnp.asarray(train_data[batch_indices], dtype=self.dtype)
                        
                        perc_boost = self.get_geometric_boosts(train_batch)
                        lambda_dict = self.lambda_controller(last_metrics)
                        for k in self.lambda_controller.targets.keys():
                            if k not in ['l1', 'vq', 'adv']: lambda_dict[k] *= perc_boost
                        self.current_lambdas_for_ui = lambda_dict
                        current_lambdas = tuple(lambda_dict.values())
                        
                        self.d_loss_ema = (1 - self.d_loss_ema_alpha) * self.d_loss_ema + self.d_loss_ema_alpha * last_metrics.get('d_loss', 0.5)

                        d_is_locked_out = False
                        if self.d_lockout_steps > 0: self.d_lockout_steps -= 1; d_is_locked_out = True
                        elif self.d_loss_ema < self.d_lockout_threshold: self.d_lockout_steps = self.d_lockout_duration; d_is_locked_out = True
                        
                        self.g_lr_multiplier = 1.0; self.d_lr_multiplier = 1.0
                        if not d_is_locked_out:
                            if self.d_loss_ema < self.d_loss_target_min: self.d_lr_multiplier = 0.5; self.g_lr_multiplier = 1.5
                            elif self.d_loss_ema > self.d_loss_target_max: self.d_lr_multiplier = 1.5; self.g_lr_multiplier = 0.5
                        
                        step_key, self.train_key = jax.random.split(self.train_key)
                        
                        states, metrics, new_q_state = train_step(states, current_q_state, train_batch, step_key, current_lambdas, self.d_loss_ema, self.g_lr_multiplier, self.d_lr_multiplier, self.generator_model.apply, self.discriminator_model.apply, d_is_locked_out=d_is_locked_out, use_q_controller=self.use_q_controller)
                        current_q_state = new_q_state
                        
                        metrics_cpu = {k: v.item() for k, v in metrics.items()}
                        last_metrics = metrics_cpu; self.last_metrics_for_ui = metrics_cpu

                        with self.ui_lock:
                            self.g_loss_hist.append(metrics_cpu['g_loss']); self.d_loss_hist.append(metrics_cpu['d_loss'])
                            self.l1_hist.append(metrics_cpu['l1']); self.ssim_hist.append(metrics_cpu.get('ssim',0.0))
                            self.vq_hist.append(metrics_cpu['vq']); self.varent_hist.append(metrics_cpu['varentropy'])
                        
                        if self.q_controller:
                            q_state_host = jax.device_get(current_q_state)
                            self.q_controller.current_value = q_state_host.current_value.item()
                            self.q_controller.exploration_rate_q = q_state_host.exploration_rate.item()
                            self.q_controller.last_reward = q_state_host.last_reward.item()
                            self.q_controller._step_count = q_state_host.step_count.item()
                            status_map = {0: "WARMUP", 1: "IMPROVING", 2: "STAGNATED", 3: "REGRESSING"}
                            self.q_controller.status = status_map[q_state_host.status_code.item()]
                        
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
                    pid_state_to_save = self.lambda_controller.state_dict()
                    data_to_save = {'gen_params': host_state_to_save.generator.params, 'gen_opt_state': host_state_to_save.generator.opt_state, 'disc_params': host_state_to_save.discriminator.params, 'disc_opt_state': host_state_to_save.discriminator.opt_state, 'epoch': epoch, 'global_step': global_step, 'last_metrics': last_metrics, 'pid_controller_state': pid_state_to_save}
                    if self.q_controller:
                        data_to_save['q_controller_jax'] = jax.device_get(current_q_state)
                    with open(ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
        finally:
            console.print(f"\n[yellow]--- Training loop exited. Waiting for background tasks to finish... ---[/yellow]")
            # --- [THE FIX] Ensure all threads are signaled to stop ---
            self.interactive_state.shutdown_event.set()
            background_executor.shutdown(wait=True)
            final_epoch_count = self.args.epochs if 'epoch' not in locals() else epoch
            console.print(f"\n[yellow]--- Training session finished at epoch {final_epoch_count+1}. Saving final state... ---[/yellow]")
            
            if 'states' in locals():
                host_state_final = jax.device_get(states)
                pid_state_to_save = self.lambda_controller.state_dict()
                final_data = {'gen_params': host_state_final.generator.params, 'gen_opt_state': host_state_final.generator.opt_state, 'disc_params': host_state_final.discriminator.params, 'disc_opt_state': host_state_final.discriminator.opt_state, 'epoch': final_epoch_count, 'global_step': global_step, 'last_metrics': last_metrics, 'pid_controller_state': pid_state_to_save}
                if self.q_controller:
                    final_data['q_controller_jax'] = jax.device_get(current_q_state)
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



@dataclass
class TimingStats:
    total: float = 0.0
    data_loading: float = 0.0
    host_prep: float = 0.0
    device_execution: float = 0.0
    host_post: float = 0.0


class GenerationalLayoutManager:
    """
    [OPTIMIZED] A state-aware layout manager. It now renders a "Compiling"
    status panel and is robust to the progress bar not being initialized
    during the JIT phase. Profiler and detailed timing panels removed for speed.
    """
    def __init__(self, trainer_instance):
        self.trainer = trainer_instance
        self.console = Console()

    def generate_layout(self) -> Layout:
        trainer = self.trainer
        with trainer.ui_lock:
            if trainer.ui_state.get("is_compiling", False):
                from rich.spinner import Spinner
                spinner = Spinner("dots", text=" JIT compiling full training pipeline...")
                info_text = Text(
                    "This may take several minutes on first run.\n"
                    "The UI will become fully interactive once compilation is complete.\n\n"
                    "Press 'q' or Ctrl+C to quit.",
                    justify="center", style="dim"
                )
                compiling_panel = Panel(
                    Align.center(Group(spinner, "\n", info_text)),
                    title="[bold yellow]🚀 System Initializing[/]",
                    border_style="yellow",
                    height=15
                )
                return Layout(compiling_panel)

            precision_str = "MIXED"
            param_count = trainer.ui_state.get("param_count", 0)
            header_text = f"🧬 [bold]GENERATIONAL WORLD MODEL[/] | Total Params: [yellow]{param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            header_panel = Panel(Align.center(header_text), style="bold red", title="[dim]wubumind.ai[/dim]", title_align="right")

            main_layout = Layout(name="main")
            main_layout.split_row(
                Layout(name="agent_tiers", ratio=2),
                Layout(name="system_preview", ratio=1)
            )

            tier_panels = []
            tier_colors = {"kids": "cyan", "students": "magenta", "polisher": "green"}
            agent_controllers = trainer.ui_state.get("agent_controllers", {})
            for tier_name, config in trainer.population_tiers.items():
                tier_table = Table.grid(expand=True, padding=(0, 1)); tier_table.add_column("Agent", style="dim", width=12); tier_table.add_column("LR", width=10); tier_table.add_column("Status")
                controller = agent_controllers.get(tier_name)
                if controller:
                    status_short = controller.status.split(' ')[0]
                    if "IMPROVING" in status_short: s_emoji, s_color = "😎", "green"
                    elif "STAGNATED" in status_short: s_emoji, s_color = "🤔", "yellow"
                    elif "WARMUP" in status_short: s_emoji, s_color = "🐣", "blue"
                    else: s_emoji, s_color = "😠", "red"
                    
                    for i in range(config['count']):
                        agent_name = f"{tier_name[:-1]}_{i+1}" if tier_name.endswith('s') else f"{tier_name}_{i+1}"
                        tier_table.add_row(agent_name, f"{controller.current_value:.2e}", f"[{s_color}]{status_short}[/] {s_emoji}")
                
                panel_title = f"[bold {tier_colors.get(tier_name, 'white')}]{tier_name.capitalize()} Tier[/] ({config['count']}x {config['d_model']}d, {config['num_layers']}L)"
                tier_panels.append(Panel(tier_table, title=panel_title, border_style=tier_colors.get(tier_name, 'white')))
            main_layout["agent_tiers"].split_column(*tier_panels)
            
            right_column_layout = Layout(name="right_column")
            
            system_stats_tbl = Table.grid(expand=True, padding=(0,1)); system_stats_tbl.add_column(style="dim", width=11); system_stats_tbl.add_column()
            steps_per_sec = trainer.ui_state.get("steps_per_sec", 0.0)
            system_stats_tbl.add_row("Steps/sec", f"[blue]{steps_per_sec:.2f}[/] 🏃💨"); mem, util = trainer._get_gpu_stats()
            system_stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); system_stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            system_stats_panel = Panel(system_stats_tbl, title="[bold]📊 System Stats[/]", border_style="blue", height=5)
            
            current_preview_prompt_idx = trainer.ui_state.get("current_preview_prompt_idx", 0)
            validation_prompts = trainer.ui_state.get("validation_prompts", [])
            current_prompt = validation_prompts[current_preview_prompt_idx] if validation_prompts else "N/A"
            prompt_text = Text(f"Prompt #{current_preview_prompt_idx+1}: \"{current_prompt}\"", justify="center", style="dim")
            
            rendered_preview = trainer.ui_state.get("rendered_preview")
            if rendered_preview:
                # --- [THE FIX] Wrap the Pixels object to make it scalable. ---
                # `Group` allows `Align` to measure its contents and scale them down if needed.
                # `fit=True` tells Align to perform this scaling.
                preview_content = Align.center(Group(rendered_preview), fit=True)
            else:
                preview_content = Align.center("...Awaiting First Generation...")

            preview_group = Group(prompt_text, "\n", preview_content)
            
            preview_panel = Panel(preview_group, title="[bold]🖼️ World Model Preview[/]")
            
            log_messages = trainer.ui_state.get("log_messages", deque(maxlen=5))
            log_text = Text("\n".join(log_messages), no_wrap=True)
            log_panel = Panel(log_text, title="[bold]📝 Logs[/]", border_style="dim", height=7)
            
            right_column_layout.split_column(system_stats_panel, preview_panel, log_panel)
            main_layout["system_preview"].update(right_column_layout)

            root_layout = Layout(name="root")
            
            if hasattr(trainer, 'progress'):
                root_layout.split_column(Layout(header_panel, name="header", size=3), Layout(main_layout, name="main", ratio=1), Layout(trainer.progress, name="progress", size=3))
            else:
                root_layout.split_column(Layout(header_panel, name="header", size=3), Layout(main_layout, name="main", ratio=1))
                
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
        
        # [THE FIX] Load tokenizer config and create a static, JIT-compatible NamedTuple
        with open(tok_config_path, 'rb') as f: 
            tok_config_dict = pickle.load(f)
        tok_config_dict['dtype'] = self.dtype
        self.tok_config = TokenizerConfig(**tok_config_dict)
        
        try:
            p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
            p1_d_model = int(p1_path.stem.split('_')[-2].replace('d', ''))
        except (StopIteration, ValueError):
            sys.exit(f"[FATAL] Could not find a unique Phase 1 model file for basename '{args.basename}'.")

        console.print(f"--- Loading Phase 1 AE from: [green]{p1_path}[/green] (d_model={p1_d_model}) ---")
        
        # [THE FIX] Create static, JIT-compatible NamedTuple for Phase 1 config
        self.p1_config = P1Config(d_model=p1_d_model, latent_grid_size=self.tok_config.latent_grid_size, input_image_size=512, dtype=self.dtype)
        self.p1_model = TopologicalCoordinateGenerator(**self.p1_config._asdict())
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']

        # [THE FIX] Instantiate the tokenizer model using the static config
        self.tokenizer = LatentTokenizerVQGAN(**self.tok_config._asdict())
        
        # [REMOVED] Redundant/vestigial model and config initializations
        
        self.interactive_state = InteractivityState()
        
        self.ui_state = {
            "steps_per_sec": 0.0,
            "param_count": 0,
            "agent_controllers": {},
            "rendered_preview": None,
            "current_preview_prompt_idx": 0,
            "validation_prompts": ["a red cup on a table", "a photorealistic orange cat", "a blue ball", "green grass with a single tree", "a purple sports car"],
            "timing_stats": TimingStats(),
            "profiling_active": False,
            "log_messages": deque(maxlen=5),
        }
        self.ui_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        with torch.no_grad():
            text_tokens = clip.tokenize(self.ui_state["validation_prompts"]).to(_clip_device)
            self.validation_embeddings = self.clip_model.encode_text(text_tokens).cpu().numpy()
        
        self.current_preview_prompt_idx = 0
        self.rendered_preview = None

    def _save_cpu_data_task(self, cpu_data_to_save, path):
        with open(path, 'wb') as f: pickle.dump(cpu_data_to_save, f)

    def _get_common_train_setup(self):
        # --- [THE FIX] This method no longer takes `num_pipeline_steps` ---
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

        def _data_generator(path, split_prefix):
            with np.load(path) as data:
                tokens = data[f'{split_prefix}_tokens']
                embeddings = data[f'{split_prefix}_embeddings']
                for i in range(len(tokens)):
                    yield tokens[i], embeddings[i]

        with np.load(tokenized_data_path) as data:
            num_train_samples = data['train_tokens'].shape[0]
            num_val_samples = data['val_tokens'].shape[0]
            token_shape = data['train_tokens'].shape[1:]
            embedding_shape = data['train_embeddings'].shape[1:]

        output_signature = (
            tf.TensorSpec(shape=token_shape, dtype=tf.int32),
            tf.TensorSpec(shape=embedding_shape, dtype=tf.float32)
        )

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





class PolisherConductor(BaseGenerativeConductor):
    """
    A specialized Conductor that utilizes the Synaptic Bridge. It overrides the
    base forward pass to manage three sources of information: its own canvas,
    the text prompt, and the combined proposals from the Kid and Student tiers.
    """
    def __call__(self, input_tokens, text_emb, kid_ste, student_ste, train: bool = True):
        # The polisher's main input (`input_tokens`) is the Student's output.
        x, ctx = self._prepare_inputs_float(input_tokens, text_emb)

        # Prepare the standard key/value cache for cross-attention to the text prompt.
        kv_caches = [(block.ca.k_proj(ctx), block.ca.v_proj(ctx)) for block in self.blocks]

        # --- [NEW] Prepare the Synaptic Bridge cache ---
        bridge_context = self._prepare_bridge_context(kid_ste, student_ste)
        bridge_kv_caches = [(block.sa_bridge.k_proj(bridge_context), block.sa_bridge.v_proj(bridge_context)) for block in self.blocks]
        # ---

        for i, block in enumerate(self.blocks):
            k_cache, v_cache = kv_caches[i]
            k_bridge, v_bridge = bridge_kv_caches[i]
            x = block(
                x, ctx, mask=None,
                k_context_cache=k_cache, v_context_cache=v_cache,
                k_bridge_cache=k_bridge, v_bridge_cache=v_bridge
            )

        x = self.norm(x)
        x = self.output_head(x)
        
        logits_3d = x.reshape(-1, self.grid_z, self.grid_y, self.grid_x, self.depth_patch_size, self.height_patch_size, self.width_patch_size, self.vocab_size)
        logits_3d = logits_3d.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        final_logits = logits_3d.reshape(-1, self.depth_size, self.height_size, self.width_size, self.vocab_size)
        return final_logits

    def get_features(self, input_tokens, text_emb, kid_ste, student_ste, train: bool = True):
        # This method must also be overridden to match the new __call__ signature.
        x, ctx = self._prepare_inputs_float(input_tokens, text_emb)
        kv_caches = [(block.ca.k_proj(ctx), block.ca.v_proj(ctx)) for block in self.blocks]
        bridge_context = self._prepare_bridge_context(kid_ste, student_ste)
        bridge_kv_caches = [(block.sa_bridge.k_proj(bridge_context), block.sa_bridge.v_proj(bridge_context)) for block in self.blocks]

        for i, block in enumerate(self.blocks):
            k_cache, v_cache = kv_caches[i]
            k_bridge, v_bridge = bridge_kv_caches[i]
            x = block(
                x, ctx, mask=None,
                k_context_cache=k_cache, v_context_cache=v_cache,
                k_bridge_cache=k_bridge, v_bridge_cache=v_bridge
            )
            
        x_norm = self.norm(x)
        projected_patch_embeddings = self.mmd_projection(x_norm)
        x_out = self.output_head(x_norm)

        logits_3d = x_out.reshape(-1, self.grid_z, self.grid_y, self.grid_x, self.depth_patch_size, self.height_patch_size, self.width_patch_size, self.vocab_size)
        logits_3d = logits_3d.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        final_logits = logits_3d.reshape(-1, self.depth_size, self.height_size, self.width_size, self.vocab_size)
        return final_logits, projected_patch_embeddings
        
    # --- [THE FIX] ---
    # The Polisher requires its own initialization method to ensure the synaptic
    # bridge layers are created. Its signature matches the specialized __call__.
    def init_forward_pass(self, input_tokens, text_emb, kid_ste, student_ste, train: bool = True):
        _ = self.__call__(input_tokens, text_emb, kid_ste, student_ste, train=train)
        _, _ = self.get_features(input_tokens, text_emb, kid_ste, student_ste, train=train)
        return jnp.array(0.0)


# Define immutable, hashable NamedTuples to hold static model configurations.
# This is the correct way to pass complex configuration to a JIT'd function.
# Place near the top with other class definitions.
class StaticAgentData(NamedTuple):
    kids: GenerativeConductor
    students: GenerativeConductor
    polisher: GenerativeConductor
    tokenizer: LatentTokenizerVQGAN
    p1_model: TopologicalCoordinateGenerator
# [NEW] Define a structure for just the serializable parts of the state
class SerializableStackedState(NamedTuple):
    step: chex.Array
    params: chex.ArrayTree
    opt_state: any

class StackedTrainState(NamedTuple):
    step: chex.Array
    params: chex.ArrayTree
    # tx: Any # Removed
    opt_state: Any

# [NEW] Define a NamedTuple for the pipeline's carry-over state
class PipelineState(NamedTuple):
    all_states: Tuple[StackedTrainState, ...]
    q_controllers: Tuple[QController, ...] # Correctly uses the QController PyTree
    # [NEW] Add the state for the Kids' token discriminator
    kids_d_state: TrainState
    prev_kid_ste: chex.Array
    prev_student_ste: chex.Array    

class ConductorConfig(NamedTuple):
    num_codes: int
    d_model: int
    num_layers: int
    num_heads: int
    dtype: Any

class TokenizerConfig(NamedTuple):
    num_codes: int
    code_dim: int
    latent_grid_size: int
    dtype: Any
    
class P1Config(NamedTuple):
    d_model: int
    latent_grid_size: int
    input_image_size: int
    dtype: Any

class PatchConfig(NamedTuple):
    mask_token_id: int
    depth: int
    height: int
    width: int
    patch_size_d: int
    patch_size_h: int
    patch_size_w: int



@partial(jax.jit, static_argnames=('num_patches', 'patch_size'))
def _extract_random_logit_patches(key, grid, num_patches, patch_size):
    """
    Extracts a batch of random patches from a 4D (B, H, W, V) logit/logprob grid.
    This is the core of Asymmetric Patchwise Knowledge Distillation.
    """
    B, H, W, V = grid.shape
    
    # Generate random top-left corners for each patch
    h_coords = jax.random.randint(key, (B, num_patches), 0, H - patch_size + 1)
    w_coords = jax.random.randint(key, (B, num_patches), 0, W - patch_size + 1)

    # vmap over the batch dimension
    def get_patches_for_batch_item(single_grid, h_c, w_c):
        # vmap over the number of patches
        def get_one_patch(h, w):
            return jax.lax.dynamic_slice(single_grid, (h, w, 0), (patch_size, patch_size, V))
        return jax.vmap(get_one_patch)(h_c, w_c)

    all_patches = jax.vmap(get_patches_for_batch_item)(grid, h_coords, w_coords)
    # Reshape from (B, num_patches, pH, pW, V) to (B * num_patches, pH, pW, V)
    return all_patches.reshape(-1, patch_size, patch_size, V)
@partial(jax.jit, static_argnames=('num_patches', 'patch_size_d', 'patch_size_h', 'patch_size_w'))
def _extract_random_patches(key, grid, num_patches, patch_size_d, patch_size_h, patch_size_w):
    """Extracts a batch of random patches from a 5D one-hot token grid."""
    B, D, H, W, C = grid.shape
    
    # Generate random top-left-front corners for each patch
    d_coords = jax.random.randint(key, (B, num_patches), 0, D - patch_size_d + 1)
    h_coords = jax.random.randint(key, (B, num_patches), 0, H - patch_size_h + 1)
    w_coords = jax.random.randint(key, (B, num_patches), 0, W - patch_size_w + 1)

    # vmap over the batch dimension
    def get_patches_for_batch_item(single_grid, d_c, h_c, w_c):
        # vmap over the number of patches
        def get_one_patch(d, h, w):
            return jax.lax.dynamic_slice(single_grid, (d, h, w, 0), (patch_size_d, patch_size_h, patch_size_w, C))
        return jax.vmap(get_one_patch)(d_c, h_c, w_c)

    all_patches = jax.vmap(get_patches_for_batch_item)(grid, d_coords, h_coords, w_coords)
    # Reshape from (B, num_patches, pD, pH, pW, C) to (B * num_patches, pD, pH, pW, C)
    return all_patches.reshape(-1, patch_size_d, patch_size_h, patch_size_w, C)

@partial(jax.jit, static_argnames=(
    'pipeline_step_fn', 'static_pipeline_args'
))
def _jitted_pipeline_train_step(initial_pipeline_state, xs_for_scan, pipeline_step_fn, static_pipeline_args):
    """
    A pure, JIT-compilable wrapper around the jax.lax.scan call.
    All static configuration is passed explicitly.
    """
    scan_body = partial(pipeline_step_fn, static_args=static_pipeline_args)
    # --- [THE FIX] ---
    # The `scan` will output a `PipelineState` where `all_states` is a tuple.
    # We must handle this structure after the call.
    final_state_tuple, stacked_outputs = jax.lax.scan(scan_body, initial_pipeline_state, xs_for_scan)
    return final_state_tuple, stacked_outputs


@partial(jax.jit, static_argnames=('scan_body',))
def _jitted_scan_wrapper(scan_body, initial_state, xs):
    """A pure JIT wrapper that executes a given scan_body function."""
    final_state, stacked_outputs = jax.lax.scan(scan_body, initial_state, xs)
    return final_state, stacked_outputs




@partial(jax.jit, static_argnames=(
    'agent_apply_fns', 'agent_optimizers', 'kids_d_optimizer_tx', 'polisher_vocab_size', 
    'polisher_depth_size', 'polisher_height_size', 'polisher_width_size', 'polisher_mask_token_id', 
    'num_patches', 'patch_size_d', 'patch_size_h', 'patch_size_w',
    'distill_num_patches', 'distill_patch_size', 'distill_temp'
))
def _jitted_standalone_pipeline_step(
    pipeline_state, xs, lambdas, gan_balancer_params,
    agent_apply_fns, agent_optimizers, kids_d_optimizer_tx: optax.GradientTransformation,
    polisher_vocab_size, polisher_depth_size, polisher_height_size, polisher_width_size, polisher_mask_token_id,
    num_patches, patch_size_d, patch_size_h, patch_size_w,
    distill_num_patches, distill_patch_size, distill_temp
):
    """
    [DEFINITIVE STABILITY FIX] Implements a robust, temperature-scaled distillation
    loss using softmax_cross_entropy with soft labels. This is applied to BOTH
    the self-distillation (mlm_loss) and refinement_loss, preventing NaN errors
    and fully realizing the user's insight to "soft-teach" all model components.
    """
    all_states_tuple, q_controllers_tuple, kids_d_state, prev_kid_ste, prev_student_ste = pipeline_state.all_states, pipeline_state.q_controllers, pipeline_state.kids_d_state, pipeline_state.prev_kid_ste, pipeline_state.prev_student_ste
    
    (real_tokens_flat, text_emb, key) = xs
    d_is_locked_out, g_lr_mult, d_lr_mult = gan_balancer_params

    real_tokens_3d = real_tokens_flat.reshape(polisher_depth_size, polisher_height_size, polisher_width_size)
    real_tokens_onehot = jax.nn.one_hot(real_tokens_3d[None, ...], polisher_vocab_size)

    tier_names = ("kids", "students", "polisher")
    q_children = [c.tree_flatten()[0] for c in q_controllers_tuple]; stacked_q_children_by_type = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *q_children)
    stacked_q_controller = QController.tree_unflatten(q_controllers_tuple[0].tree_flatten()[1], stacked_q_children_by_type)
    q_key, loss_key, d_patch_key = jax.random.split(key, 3); q_keys = jax.random.split(q_key, len(tier_names))
    lrs_stacked, new_q_controllers_stacked, last_action_indices_stacked = jax.vmap(q_controller_choose_action)(stacked_q_controller, q_keys)
    
    lambda_mlm, lambda_refinement, lambda_world_mmd, lambda_kids_adv, lambda_r1 = lambdas
        
    def loss_fn(all_params_tuple):
        all_params = {name: params for name, params in zip(tier_names, all_params_tuple)}; from jax import checkpoint
        keys = jax.random.split(loss_key, 7)
        key_dropout, key_gumbel_kids, key_gumbel_students, key_gumbel_polisher, key_distill_k, key_distill_s, key_distill_p = keys
        
        canvas_shape_3d = (text_emb.shape[0], polisher_depth_size, polisher_height_size, polisher_width_size)
        initial_tokens_onehot = jax.nn.one_hot(jnp.full(canvas_shape_3d, polisher_mask_token_id, dtype=jnp.int32), polisher_vocab_size)
        
        kids_apply_fn, students_apply_fn, polisher_apply_fn, kids_d_apply_fn = agent_apply_fns
        
        kid_logits_stack = checkpoint(jax.vmap(lambda p: kids_apply_fn({'params': p}, initial_tokens_onehot, text_emb, train=True, rngs={'dropout': key_dropout})))(all_params['kids'])
        kid_logits_ensembled = jnp.mean(kid_logits_stack, axis=0)
        new_kid_ste, _ = gumbel_softmax_straight_through(kid_logits_ensembled, key_gumbel_kids)
        
        student_logits_stack = checkpoint(jax.vmap(lambda p: students_apply_fn({'params': p}, new_kid_ste, text_emb, train=True, rngs={'dropout': key_dropout})))(all_params['students'])
        student_logits_ensembled = jnp.mean(student_logits_stack, axis=0)
        new_student_ste, _ = gumbel_softmax_straight_through(student_logits_ensembled, key_gumbel_students)

        polisher_logits_stack, polisher_embeddings_stack = checkpoint(jax.vmap(lambda p: polisher_apply_fn({'params': p}, new_student_ste, text_emb, new_kid_ste, new_student_ste, train=True, rngs={'dropout': key_dropout}, method=PolisherConductor.get_features)))(all_params['polisher'])
        
        polisher_logits, polisher_embeddings = jnp.mean(polisher_logits_stack, axis=0), jnp.mean(polisher_embeddings_stack, axis=0)
        polisher_ste, _ = gumbel_softmax_straight_through(polisher_logits, key_gumbel_polisher)

        fake_patches_polisher = _extract_random_patches(d_patch_key, polisher_ste, num_patches, patch_size_d, patch_size_h, patch_size_w)
        d_fake_output = kids_d_apply_fn({'params': kids_d_state.params}, fake_patches_polisher)
        loss_kids_adv = jnp.mean((d_fake_output - 1)**2)
        
        # --- Numerically Stable Distillation Helper ---
        def _distillation_loss(student_logits, teacher_logits):
            teacher_probs = jax.lax.stop_gradient(jax.nn.softmax(teacher_logits / distill_temp))
            # The T^2 factor keeps the gradient magnitude consistent across temperatures
            loss = optax.softmax_cross_entropy(student_logits / distill_temp, teacher_probs).mean()
            return loss * (distill_temp ** 2)

        # --- Self-Distillation for MLM Loss ---
        loss_mlm = (_distillation_loss(kid_logits_ensembled, kid_logits_ensembled) + 
                    _distillation_loss(student_logits_ensembled, student_logits_ensembled) +
                    _distillation_loss(polisher_logits, polisher_logits))
        
        # --- Asymmetric Patchwise Knowledge Distillation for Refinement Loss ---
        B, D, H, W, V = student_logits_ensembled.shape
        student_logits_flat = student_logits_ensembled.reshape(B, D*H, W, V)
        polisher_logits_flat = polisher_logits.reshape(B, D*H, W, V)
        kid_logits_flat = kid_logits_ensembled.reshape(B, D*H, W, V)

        student_patches = _extract_random_logit_patches(key_distill_s, student_logits_flat, distill_num_patches, distill_patch_size)
        polisher_patches_teacher = _extract_random_logit_patches(key_distill_p, polisher_logits_flat, distill_num_patches, distill_patch_size)
        kid_patches = _extract_random_logit_patches(key_distill_k, kid_logits_flat, distill_num_patches, distill_patch_size)
        student_patches_teacher = _extract_random_logit_patches(key_distill_s, student_logits_flat, distill_num_patches, distill_patch_size)

        distill_students_to_polisher = _distillation_loss(student_patches, polisher_patches_teacher)
        distill_kids_to_students = _distillation_loss(kid_patches, student_patches_teacher)
        
        loss_refinement = distill_students_to_polisher + distill_kids_to_students

        text_emb_tiled = jnp.tile(text_emb[:, None, :], (1, polisher_embeddings.shape[1], 1))
        loss_mmd = calculate_mmd_loss(polisher_embeddings, text_emb_tiled)
        
        total_loss = (lambda_mlm * loss_mlm) + (lambda_refinement * loss_refinement) + (lambda_world_mmd * loss_mmd) + (lambda_kids_adv * loss_kids_adv)
        
        return total_loss, (new_kid_ste, new_student_ste, polisher_ste, loss_mlm, loss_refinement, loss_mmd, loss_kids_adv, polisher_logits_stack)

    (loss, (new_kid_ste, new_student_ste, polisher_ste, mlm_loss, ref_loss, mmd_loss, kids_adv_loss, polisher_logits_stack)), grads_tuple = jax.value_and_grad(loss_fn, has_aux=True)(tuple(s.params for s in all_states_tuple))
    
    def kids_d_loss_fn(d_params):
        d_apply_fn = agent_apply_fns[-1]
        
        real_patches = _extract_random_patches(d_patch_key, real_tokens_onehot, num_patches, patch_size_d, patch_size_h, patch_size_w)
        fake_patches_kids = _extract_random_patches(d_patch_key, jax.lax.stop_gradient(new_kid_ste), num_patches, patch_size_d, patch_size_h, patch_size_w)
        fake_patches_students = _extract_random_patches(d_patch_key, jax.lax.stop_gradient(new_student_ste), num_patches, patch_size_d, patch_size_h, patch_size_w)
        fake_patches_polisher = _extract_random_patches(d_patch_key, jax.lax.stop_gradient(polisher_ste), num_patches, patch_size_d, patch_size_h, patch_size_w)

        (_, real_grads) = jax.value_and_grad(lambda x: jnp.sum(d_apply_fn({'params': d_params}, x)))(real_patches)
        r1_penalty = 0.5 * lambda_r1 * jnp.mean(jnp.sum(jnp.square(real_grads), axis=[1,2,3,4]))

        d_fake_kids = d_apply_fn({'params': d_params}, fake_patches_kids)
        d_fake_students = d_apply_fn({'params': d_params}, fake_patches_students)
        d_fake_polisher = d_apply_fn({'params': d_params}, fake_patches_polisher)
        
        loss_fake_avg = (jnp.mean(d_fake_kids**2) + jnp.mean(d_fake_students**2) + jnp.mean(d_fake_polisher**2)) / 3.0
        
        loss_d = 0.5 * (jnp.mean((d_apply_fn({'params': d_params}, real_patches) - 1)**2) + loss_fake_avg) + r1_penalty
        return loss_d
    
    d_loss_val, d_grads = jax.value_and_grad(kids_d_loss_fn)(kids_d_state.params)
    
    d_grads_no_update = jax.tree_util.tree_map(jnp.zeros_like, d_grads)
    final_d_grads = jax.lax.cond(d_is_locked_out, lambda: d_grads_no_update, lambda: d_grads)
    new_kids_d_state = kids_d_state.apply_gradients(grads=final_d_grads)

    new_states_list = []
    for i in range(len(tier_names)):
        state, grad, lr = all_states_tuple[i], grads_tuple[i], lrs_stacked[i]
        final_lr = lr * g_lr_mult
        learning_rate_tree = jax.tree_util.tree_map(lambda _: final_lr, state.params)
        updates, new_opt_state = agent_optimizers[i].update(grad, state.opt_state, state.params, learning_rate=learning_rate_tree)
        new_params = optax.apply_updates(state.params, updates)
        new_states_list.append(state._replace(step=state.step + 1, params=new_params, opt_state=new_opt_state))
    
    final_q_controllers_stacked = jax.vmap(q_controller_update, in_axes=(0, None, 0))(new_q_controllers_stacked, loss, last_action_indices_stacked)
    final_q_controllers_list = [jax.tree_util.tree_map(lambda x: x[i], final_q_controllers_stacked) for i in range(len(tier_names))]
    
    new_carry = PipelineState(
        all_states=tuple(new_states_list), 
        q_controllers=tuple(final_q_controllers_list), 
        kids_d_state=new_kids_d_state,
        prev_kid_ste=new_kid_ste.astype(jnp.float32), 
        prev_student_ste=new_student_ste.astype(jnp.float32)
    )
    
    metrics = {
        'loss/total': loss, 'loss/mlm': mlm_loss, 'loss/refinement': ref_loss,
        'loss/world_mmd': mmd_loss, 'loss/kids_adv': kids_adv_loss, 'loss/kids_d': d_loss_val
    }
    return new_carry, (loss, metrics, polisher_logits_stack[0])



@partial(jax.jit, static_argnames=('apply_discriminator_fn', 'patch_config', 'vocab_size', 'top_k'))
def _gradient_free_discriminator_guidance(
    tokens,  # The current canvas (B, L)
    logits,  # The logits to be guided (B, L, V)
    discriminator_params,
    apply_discriminator_fn,
    patch_config: "PatchConfig",
    vocab_size,
    top_k,
    key
):
    """
    Applies discriminator guidance without using gradients. It evaluates the top_k
    candidate tokens for each position and re-weighs them based on the discriminator's score.
    """
    B, L, V = logits.shape
    original_dtype = logits.dtype # Store the original dtype

    unknown_mask = (tokens == patch_config.mask_token_id)
    
    # Perform probability calculations in float32 for stability
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
    
    hypothetical_tokens = jnp.where(
        unknown_mask[..., None], 
        top_k_indices, 
        tokens[..., None]
    )
    hypothetical_tokens = hypothetical_tokens.transpose(2, 0, 1)

    def score_canvas(canvas):
        canvas_3d = canvas.reshape(B, patch_config.depth, patch_config.height, patch_config.width)
        one_hot_canvas = jax.nn.one_hot(canvas_3d, vocab_size)
        patches = _extract_random_patches(
            key, one_hot_canvas, num_patches=32, 
            patch_size_d=patch_config.patch_size_d,
            patch_size_h=patch_config.patch_size_h,
            patch_size_w=patch_config.patch_size_w
        )
        # Discriminator is float32, so this is fine
        return jnp.mean(apply_discriminator_fn(discriminator_params, patches))

    scores = jax.vmap(score_canvas)(hypothetical_tokens)
    
    score_weights = jax.nn.softmax(scores).reshape(1, 1, top_k)
    guided_top_k_probs = top_k_probs * score_weights
    guided_top_k_probs /= (jnp.sum(guided_top_k_probs, axis=-1, keepdims=True) + 1e-9)
    
    # --- [FIX FOR WARNING] ---
    # Create the final_probs array with the target dtype from the start.
    final_probs = jnp.zeros_like(probs, dtype=original_dtype)
    # Cast the guided probabilities to the correct dtype before scattering.
    guided_top_k_probs_casted = guided_top_k_probs.astype(original_dtype)
    
    batch_indices, seq_indices = jnp.meshgrid(jnp.arange(B), jnp.arange(L), indexing='ij')
    final_probs = final_probs.at[batch_indices[..., None], seq_indices[..., None], top_k_indices].set(guided_top_k_probs_casted)
    
    # one_hot will produce the correct dtype if the input has the correct dtype
    final_probs = jnp.where(
        unknown_mask[..., None],
        final_probs,
        jax.nn.one_hot(tokens, vocab_size, dtype=original_dtype)
    )

    return jnp.log(final_probs.astype(jnp.float32) + 1e-9).astype(original_dtype)





@partial(jax.jit, static_argnames=(
    'num_steps', 'guidance_scale', 'resolution', 'grid_shape', 'mask_token_id', 'vocab_size',
    'kid_config', 'student_config', 'polisher_config', 'tok_config', 'p1_config',
    'd_guidance_scale', 'd_guidance_top_k' 
))
def _jitted_inference_pipeline(
    stacked_agent_params, tok_params_tree, p1_params_tree, key, cond_emb, uncond_emb,
    guidance_scale, num_steps, resolution, grid_shape, mask_token_id, vocab_size,
    kid_config: ConductorConfig, student_config: ConductorConfig, polisher_config: ConductorConfig,
    tok_config: TokenizerConfig, p1_config: P1Config,
    d_guidance_scale: float,
    d_guidance_top_k: int
):
    if cond_emb.ndim == 3: cond_emb = jnp.squeeze(cond_emb, axis=1)
    if uncond_emb.ndim == 3: uncond_emb = jnp.squeeze(uncond_emb, axis=1)

    depth_size, height_size, width_size = grid_shape
    L = depth_size * height_size * width_size; MASK_ID = mask_token_id; B = cond_emb.shape[0]
    mask_schedule = jnp.cos(jnp.linspace(0, 1, num_steps + 1) * jnp.pi / 2)
    
    def apply_kid(params, tokens, emb, rng):
        return KidConductor(**kid_config._asdict()).apply({'params': params}, tokens, emb, train=False, rngs={'dropout': rng})
    def apply_student(params, tokens, emb, rng):
        return StudentConductor(**student_config._asdict()).apply({'params': params}, tokens, emb, train=False, rngs={'dropout': rng})
    def apply_polisher(params, input_tokens, emb, kid_ste, student_ste, rng):
        return PolisherConductor(**polisher_config._asdict()).apply({'params': params}, input_tokens, emb, kid_ste, student_ste, train=False, rngs={'dropout': rng})
    def apply_discriminator(params, patches):
        return PatchDiscriminator(dtype=jnp.float32).apply({'params': params}, patches)

    vmapped_kids_apply = jax.vmap(apply_kid, in_axes=(0, None, None, None))
    vmapped_students_apply = jax.vmap(apply_student, in_axes=(0, None, None, None))
    vmapped_polisher_apply = jax.vmap(apply_polisher, in_axes=(0, None, None, None, None, None))

    masked_canvas_shape_3d = (B, depth_size, height_size, width_size)
    masked_canvas_onehot = jax.nn.one_hot(jnp.full(masked_canvas_shape_3d, MASK_ID, dtype=jnp.int32), vocab_size)

    pm_ref = KidConductor(**kid_config._asdict())
    patch_config_for_guidance = PatchConfig(
        mask_token_id=MASK_ID,
        depth=depth_size, height=height_size, width=width_size,
        patch_size_d=pm_ref.depth_patch_size,
        patch_size_h=pm_ref.height_patch_size,
        patch_size_w=pm_ref.width_patch_size,
    )

    def loop_body(i, carry):
        tokens, key = carry
        key, kids_key, students_key, polisher_key, dropout_key, d_guidance_key = jax.random.split(key, 6)
        
        kids_logits = jnp.mean(vmapped_kids_apply(stacked_agent_params['kids'], masked_canvas_onehot, cond_emb, dropout_key), axis=0)
        kid_ste, _ = gumbel_softmax_straight_through(kids_logits, kids_key)
        student_logits = jnp.mean(vmapped_students_apply(stacked_agent_params['students'], kid_ste, cond_emb, dropout_key), axis=0)
        student_ste, _ = gumbel_softmax_straight_through(student_logits, students_key)
        logits_cond = jnp.mean(vmapped_polisher_apply(stacked_agent_params['polisher'], student_ste, cond_emb, kid_ste, student_ste, dropout_key), axis=0)
        logits_uncond = jnp.mean(vmapped_polisher_apply(stacked_agent_params['polisher'], student_ste, uncond_emb, kid_ste, student_ste, dropout_key), axis=0)
        
        cfg_logits = (logits_uncond + guidance_scale * (logits_cond - logits_uncond))
        
        V = cfg_logits.shape[-1]
        
        def apply_guidance(lgts):
            # The guidance function returns a flat (B, L, V) tensor.
            guided_logits_flat = _gradient_free_discriminator_guidance(
                tokens, lgts.reshape(B, L, V), stacked_agent_params['kids_d'],
                apply_discriminator, patch_config_for_guidance, vocab_size,
                d_guidance_top_k, d_guidance_key
            )
            # --- [THE FIX] ---
            # 1. Reshape back to the original 5D shape.
            # 2. Cast to the original dtype to match the false_fun branch.
            return guided_logits_flat.reshape(lgts.shape).astype(lgts.dtype)

        final_logits = jax.lax.cond(
            d_guidance_scale > 0,
            apply_guidance,
            lambda x: x,
            cfg_logits
        )

        final_logits_flat = final_logits.reshape(B, L, -1)
        sampled_tokens = jax.random.categorical(polisher_key, final_logits_flat).reshape(B, L)
        
        unknown_mask = (tokens == MASK_ID)
        probs = jax.nn.softmax(final_logits_flat, axis=-1)
        confidence = jnp.max(probs, axis=-1)
        confidence = jnp.where(unknown_mask, confidence, -1.0)
        
        num_to_unmask = jnp.floor(L * (mask_schedule[i] - mask_schedule[i+1])).astype(jnp.int32)
        
        ranks = jnp.argsort(jnp.argsort(confidence, axis=-1), axis=-1)
        mask_for_updates = ranks >= (L - num_to_unmask)
        
        new_tokens = jnp.where(mask_for_updates, sampled_tokens, tokens)
        
        return new_tokens, key

    initial_tokens = jnp.full((B, L), MASK_ID, dtype=jnp.int32)
    final_tokens, _ = jax.lax.fori_loop(0, num_steps, loop_body, (initial_tokens, key))
    
    token_grid_3d = final_tokens.reshape(B, depth_size, height_size, width_size)
    B_dim, D, H, W = token_grid_3d.shape
    token_grid_2d = token_grid_3d.reshape(B_dim, D * H, W)
    
    path_params = LatentTokenizerVQGAN(**tok_config._asdict()).apply({'params': tok_params_tree}, token_grid_2d, method='decode')
    
    patch_size = resolution // 2 if resolution >= 64 else resolution
    coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution), jnp.linspace(-1,1,resolution), indexing='ij'),-1).reshape(-1,2)
    coord_chunks=jnp.array_split(coords,(resolution**2)//(patch_size**2))
    pixels_list=[TopologicalCoordinateGenerator(**p1_config._asdict()).apply({'params': p1_params_tree}, path_params, c, method='decode') for c in coord_chunks]
    return jnp.concatenate(pixels_list,axis=1).reshape(B, resolution, resolution, 3)



 
# =================================================================================================
# [NEW] UNIFIED MODEL ARCHITECTURE CONFIGURATION
# =================================================================================================

POPULATION_TIERS_CONFIG = {
    "kids":     {"count": 4, "d_model": 48, "num_layers": 6,  "num_heads": 6,  "dtype": jnp.bfloat16},
    "students": {"count": 2, "d_model": 64, "num_layers": 8,  "num_heads": 8,  "dtype": jnp.bfloat16},
    "polisher": {"count": 1, "d_model": 96, "num_layers": 12, "num_heads": 12, "dtype": jnp.bfloat16}
}    
    




class GenerationalConductorTrainer(BaseConductorTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.console = Console()
        self.console.print("--- [MODE] Generational 'No Teacher' Spatiotemporal Ecosystem is ACTIVE. ---", style="bold yellow")
        self.console.print("--- [ENHANCEMENT] Kids Tier Token Discriminator is [bold green]ONLINE[/bold green]. ---")
        self.console.print("--- [ENHANCEMENT] R1 Gradient Penalty is [bold green]ACTIVE[/bold green]. ---")
        self.console.print("--- [ENHANCEMENT] Asymmetric Patchwise Knowledge Distillation is [bold green]ACTIVE[/bold green]. ---")
        self.console.print("--- [ENHANCEMENT] PID Controller for Loss Balancing is [bold green]ACTIVE[/bold green]. ---")
        self.console.print("--- [ENHANCEMENT] Dynamic GAN Balancer & Lockout is [bold green]ACTIVE[/bold green]. ---")
        # --- [THE FIX] Added Temperature Scaling to prevent NaN loss in distillation ---
        self.console.print("--- [STABILITY] Distillation Temperature Scaling is [bold green]ACTIVE[/bold green]. ---")


        self.population_tiers = POPULATION_TIERS_CONFIG
        
        self.num_pipeline_steps = 16
        self.agent_models = {}
        self.agent_controllers = {}
        self.agent_optimizers = {}
        self.agent_configs = {}
        
        pm_ref = BaseGenerativeConductor(num_codes=1, d_model=1, num_heads=1, num_layers=1, dtype=self.dtype)
        self.num_d_patches = 32
        self.d_patch_size = (pm_ref.depth_patch_size, pm_ref.height_patch_size, pm_ref.width_patch_size)

        self.distill_num_patches = 64
        self.distill_patch_size = 4
        self.distill_temp = 2.5 # The temperature for softening distributions
        
        q_lr_config = Q_CONTROLLER_CONFIG_NORMAL

        tier_class_map = { "kids": KidConductor, "students": StudentConductor, "polisher": PolisherConductor }
        for tier_name in self.population_tiers.keys():
            config = self.population_tiers[tier_name]
            model_constructor_args = {k: v for k, v in config.items() if k != 'count'}; model_constructor_args['num_codes'] = args.num_codes
            ModelClass = tier_class_map[tier_name]
            self.agent_models[tier_name] = ModelClass(**model_constructor_args)
            config_fields = set(ConductorConfig._fields)
            static_config_args = {k: v for k, v in model_constructor_args.items() if k in config_fields}
            self.agent_configs[tier_name] = ConductorConfig(**static_config_args)
            self.agent_controllers[tier_name] = JaxHakmemQController(initial_value=args.lr, config=q_lr_config, param_name=f"{tier_name}/LR")
        
        self.kids_d_model = PatchDiscriminator(dtype=jnp.float32)
        
        self.lambda_controller = PIDLambdaController(
            targets={'mlm': 0.8, 'refinement': 0.4, 'world_mmd': 0.05, 'kids_adv': 0.7, 'kids_d': 0.3},
            base_weights={'mlm': 1.0, 'refinement': 1.0, 'world_mmd': 0.25, 'kids_adv': 0.1, 'r1': 1.0, 'kids_d': 0.0},
            gains={'mlm': (1.0, 0.01, 0.5), 'refinement': (1.0, 0.01, 0.5), 'world_mmd': (1.5, 0.02, 1.0), 'kids_adv': (1.2, 0.01, 0.8), 'kids_d': (1.0, 0.01, 0.5)}
        )
        self.last_metrics = {}
        
        self.d_loss_ema = 0.5
        self.d_loss_ema_alpha = 0.05
        self.d_loss_target_min = 0.3
        self.d_loss_target_max = 0.6
        self.d_lockout_steps = 0
        self.d_lockout_threshold = 0.05 
        self.d_lockout_duration = 10 
        
        with self.ui_lock: self.ui_state["agent_controllers"] = self.agent_controllers

        self.stacked_states = {} 
        self.layout_manager = GenerationalLayoutManager(self)
        self.q_controllers_jax = {}
        for tier_name in self.population_tiers.keys():
            controller = self.agent_controllers[tier_name]
            self.q_controllers_jax[tier_name] = QController(q_table=jnp.array(controller.q_table), metric_history=jnp.zeros(controller.metric_history.maxlen), trend_history=jnp.zeros(controller.trend_history.maxlen), current_value=jnp.array(controller.current_value), exploration_rate=jnp.array(controller.exploration_rate_q), step_count=jnp.array(controller._step_count), last_reward=jnp.array(controller.last_reward), status_code=jnp.array(0), initial_value=controller.initial_value, warmup_start_val=controller.warmup_start_val, q_table_size=controller.q_table_size, num_actions=controller.num_actions, action_factors=controller.action_factors, learning_rate_q=controller.learning_rate_q, discount_factor_q=controller.discount_factor_q, value_min=controller.value_min, value_max=controller.value_max, metric_min=controller.metric_min, metric_max=controller.metric_max, min_exploration_rate=controller.min_exploration_rate, exploration_decay=controller.exploration_decay, trend_window=controller.trend_window, improve_threshold=controller.improve_threshold, regress_threshold=controller.regress_threshold, regress_penalty=controller.regress_penalty, stagnation_penalty=controller.stagnation_penalty, warmup_steps=controller.warmup_steps)
        
        self.active_preview_future = None  
        
    def _initialize_states(self):
        key = jax.random.PRNGKey(self.args.seed); total_params = 0
        for tier_name, model in self.agent_models.items():
            config = self.population_tiers[tier_name]
            base_optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=self.args.lr))
            dummy_input_shape = (1, model.depth_size, model.height_size, model.width_size, model.vocab_size)
            dummy_input = jnp.zeros(dummy_input_shape, dtype=model.dtype)
            
            # --- [THE FIX] ---
            # We now use a conditional to call the correct `init` signature for the Polisher.
            if tier_name == "polisher":
                # Provide dummy STE tensors to the Polisher's specialized init method.
                dummy_ste_shape = (1, model.depth_size, model.height_size, model.width_size, model.vocab_size)
                dummy_ste = jnp.zeros(dummy_ste_shape, dtype=model.dtype)
                init_args = (dummy_input, jnp.zeros((1, 512), self.dtype), dummy_ste, dummy_ste)
                init_kwargs = {'train': True, 'method': model.init_forward_pass}
            else:
                # Kids and Students use the simpler, base init method.
                init_args = (dummy_input, jnp.zeros((1, 512), self.dtype))
                init_kwargs = {'train': True, 'method': model.init_forward_pass}

            params_struct = model.init({'params': key, 'dropout': key}, *init_args, **init_kwargs)['params']

            vmapped_optimizer = optax.multi_transform({'agent': base_optimizer}, jax.tree_util.tree_map(lambda _: 'agent', params_struct))
            self.agent_optimizers[tier_name] = vmapped_optimizer
            
            # This logic must also be applied when creating the stacked parameters for multiple agents.
            individual_params = []
            for i in range(config['count']):
                agent_key = jax.random.split(key, i + 2)[-1]
                agent_params = model.init({'params': agent_key, 'dropout': agent_key}, *init_args, **init_kwargs)['params']
                individual_params.append(agent_params)
            
            if config['count'] > 0: total_params += jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, individual_params[0], 0) * config['count']
            stacked_params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *individual_params)
            self.stacked_states[tier_name] = StackedTrainState(step=jnp.array(0), params=stacked_params, opt_state=vmapped_optimizer.init(stacked_params))
        
        d_key, key = jax.random.split(key)
        pm = self.agent_models['polisher']
        dummy_patch_shape = (1, self.d_patch_size[0], self.d_patch_size[1], self.d_patch_size[2], pm.vocab_size)
        d_params = self.kids_d_model.init(d_key, jnp.zeros(dummy_patch_shape))['params']
        self.kids_d_optimizer = optax.adamw(learning_rate=self.args.lr * 2.0, b1=0.5, b2=0.9)
        self.kids_d_state = TrainState.create(apply_fn=self.kids_d_model.apply, params=d_params, tx=self.kids_d_optimizer)
        total_params += jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, d_params, 0)
        
        with self.ui_lock: self.ui_state["param_count"] = total_params
        
    def _generate_preview_task(self, all_agent_stacked_states, key):
        prompt_idx = self.ui_state["current_preview_prompt_idx"]
        cond_emb = self.validation_embeddings[prompt_idx:prompt_idx+1]
        stacked_params = {tier: state.params for tier, state in all_agent_stacked_states.items()}
        uncond_emb = jnp.mean(stacked_params['polisher']['uncond_embedding'], axis=0, keepdims=False)
        pm = self.agent_models['polisher']
        grid_shape = (pm.depth_size, pm.height_size, pm.width_size)
        image_pixels_batch = _jitted_inference_pipeline(stacked_params, self.tok_params, self.p1_params, key, cond_emb, uncond_emb, guidance_scale=2.0, num_steps=8, resolution=self.preview_resolution, grid_shape=grid_shape, mask_token_id=pm.MASK_TOKEN_ID, vocab_size=pm.vocab_size, kid_config=self.agent_configs['kids'], student_config=self.agent_configs['students'], polisher_config=self.agent_configs['polisher'], tok_config=self.tok_config, p1_config=self.p1_config)
        image_pixels_batch.block_until_ready()
        img_np = np.array(((image_pixels_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        with self.ui_lock:
            if Pixels: self.ui_state["rendered_preview"] = Pixels.from_image(Image.fromarray(img_np))
            
    def _save_checkpoint(self, path: Path, global_step: int, states_to_save: Dict, d_state_to_save: TrainState, q_controllers_to_save: Dict):
        serializable_states = {tier: {'step': s.step, 'params': s.params} for tier, s in jax.device_get(states_to_save).items()}
        serializable_d_state = {'step': d_state_to_save.step, 'params': d_state_to_save.params}
        data_to_save = {
            'global_step': global_step, 'stacked_params_and_steps': serializable_states,
            'kids_d_params_and_step': serializable_d_state, 'q_controllers_jax': jax.device_get(q_controllers_to_save),
            'pid_controller_state': self.lambda_controller.state_dict(), 'last_metrics': self.last_metrics,
            'd_loss_ema': self.d_loss_ema,
        }
        with open(path, 'wb') as f: pickle.dump(data_to_save, f)
        with self.ui_lock:
            log_messages = self.ui_state.get("log_messages", deque(maxlen=5)); log_messages.append(f"✅ Checkpoint saved to {path.name}"); self.ui_state["log_messages"] = log_messages

    def train(self):
        key = jax.random.PRNGKey(self.args.seed)
        _key_listener_thread, preview_executor, train_iterator, _, num_train_samples = self._get_common_train_setup()
        self._initialize_states()
        
        initial_dynamic_states = {tier: StackedTrainState(step=s.step, params=s.params, opt_state=s.opt_state) for tier, s in self.stacked_states.items()}
        initial_q_controllers = self.q_controllers_jax
        initial_kids_d_state = self.kids_d_state
        
        ckpt_path = Path(f"chimera_{self.args.basename}_{self.args.num_codes}c_final.pkl")
        best_ckpt_path = Path(f"chimera_{self.args.basename}_{self.args.num_codes}c_best.pkl")
        start_step = 0

        if ckpt_path.exists():
            self.console.print(f"--- Resuming training from [green]{ckpt_path}[/green] ---")
            with open(ckpt_path, 'rb') as f: ckpt = pickle.load(f)
            start_step = ckpt.get('global_step', 0)
            if 'stacked_params_and_steps' in ckpt:
                for tier, s_state in ckpt['stacked_params_and_steps'].items():
                    reinitialized_opt_state = self.agent_optimizers[tier].init(s_state['params'])
                    initial_dynamic_states[tier] = StackedTrainState(step=s_state['step'], params=s_state['params'], opt_state=reinitialized_opt_state)
            if 'q_controllers_jax' in ckpt: initial_q_controllers = ckpt['q_controllers_jax']
            if 'kids_d_params_and_step' in ckpt:
                loaded_d_data = ckpt['kids_d_params_and_step']
                initial_kids_d_state = TrainState(step=loaded_d_data['step'], apply_fn=self.kids_d_model.apply, params=loaded_d_data['params'], tx=self.kids_d_optimizer, opt_state=self.kids_d_optimizer.init(loaded_d_data['params']))
            if 'pid_controller_state' in ckpt: self.lambda_controller.load_state_dict(ckpt['pid_controller_state'])
            self.last_metrics = ckpt.get('last_metrics', {})
            self.d_loss_ema = ckpt.get('d_loss_ema', 0.5)
    
        polisher_model = self.agent_models['polisher']
        self.tok_params, self.p1_params = jax.device_put(self.tok_params), jax.device_put(self.p1_params)
        
        B = self.args.batch_size
        canvas_shape_onehot = (B, polisher_model.depth_size, polisher_model.height_size, polisher_model.width_size, polisher_model.vocab_size)
        initial_kid_ste, initial_student_ste = jnp.zeros(canvas_shape_onehot, dtype=jnp.float32), jnp.zeros(canvas_shape_onehot, dtype=jnp.float32)
        
        global_step = start_step
        
        self.progress = Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("Epoch Step {task.completed}/{task.total}"), "•", TextColumn("Global: {task.fields[global_step]}"), "•", TextColumn("Loss: {task.fields[loss]:.3f}"), "•", TextColumn("[magenta]D(Kids): {task.fields[d_loss]:.3f}[/]"), "•", TextColumn("SPS: {task.fields[sps]:.2f}"))
        
        data_loading_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='DataLoader')
        def _get_next_batch_task(): return next(train_iterator)
        
        try:
            with Live(self.layout_manager.generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                def ui_render_loop(live_manager):
                    while not self.interactive_state.shutdown_event.is_set():
                        preview_change = self.interactive_state.get_and_reset_preview_change()
                        if preview_change != 0:
                            with self.ui_lock: num_prompts = len(self.ui_state["validation_prompts"]); self.ui_state["current_preview_prompt_idx"] = (self.ui_state["current_preview_prompt_idx"] + preview_change) % num_prompts
                        if self.active_preview_future and self.active_preview_future.done():
                            try: self.active_preview_future.result(); self.active_preview_future = None 
                            except Exception as e:
                                with self.ui_lock: log_messages = self.ui_state.get("log_messages", deque(maxlen=5)); log_messages.append(f"[bold red]Preview task failed![/]"); self.ui_state["log_messages"] = log_messages; self.active_preview_future = None
                        live_manager.update(self.layout_manager.generate_layout()); time.sleep(1.0 / 15)

                render_thread = threading.Thread(target=ui_render_loop, args=(live,), daemon=True); render_thread.start()

                jitted_pipeline_fn = None
                with self.ui_lock: self.ui_state["is_compiling"] = True
                
                def compile_task():
                    nonlocal jitted_pipeline_fn
                    agent_apply_fns_tuple = tuple(self.agent_models[tier].apply for tier in ["kids", "students", "polisher"]) + (self.kids_d_model.apply,)
                    agent_optimizers_tuple = tuple(self.agent_optimizers[tier] for tier in ["kids", "students", "polisher"])
                    
                    # --- [THE BUG FIX] ---
                    # The distill_temp value is now correctly passed into the partial function.
                    scan_body = partial(_jitted_standalone_pipeline_step, 
                                        agent_apply_fns=agent_apply_fns_tuple, agent_optimizers=agent_optimizers_tuple, 
                                        kids_d_optimizer_tx=self.kids_d_optimizer, polisher_vocab_size=polisher_model.vocab_size, 
                                        polisher_depth_size=polisher_model.depth_size, polisher_height_size=polisher_model.height_size, 
                                        polisher_width_size=polisher_model.width_size, polisher_mask_token_id=polisher_model.MASK_TOKEN_ID, 
                                        num_patches=self.num_d_patches, patch_size_d=self.d_patch_size[0], 
                                        patch_size_h=self.d_patch_size[1], patch_size_w=self.d_patch_size[2],
                                        distill_num_patches=self.distill_num_patches,
                                        distill_patch_size=self.distill_patch_size,
                                        distill_temp=self.distill_temp)

                    @jax.jit
                    def run_scan(initial_state, xs_for_scan, lambdas_for_scan):
                        def scan_fn_wrapper(carry_state, scan_slice):
                            tokens, text_emb, key, balancer = scan_slice
                            new_state, metrics_tuple = scan_body(carry_state, (tokens, text_emb, key), lambdas_for_scan, balancer)
                            return new_state, metrics_tuple
                        
                        return jax.lax.scan(scan_fn_wrapper, initial_state, xs_for_scan)

                    jitted_pipeline_fn = run_scan
                    
                    tier_names_tuple = ("kids", "students", "polisher")
                    dummy_states_tuple = tuple(initial_dynamic_states[name] for name in tier_names_tuple)
                    dummy_q_controllers_tuple = tuple(initial_q_controllers[name] for name in tier_names_tuple)
                    dummy_pipeline_state = PipelineState(all_states=dummy_states_tuple, q_controllers=dummy_q_controllers_tuple, kids_d_state=initial_kids_d_state, prev_kid_ste=initial_kid_ste, prev_student_ste=initial_student_ste)
                    
                    dummy_keys = jax.random.split(key, self.num_pipeline_steps)
                    dummy_tokens_flat = jnp.zeros((self.num_pipeline_steps, B, polisher_model.depth_size * polisher_model.height_size * polisher_model.width_size), dtype=jnp.int32)
                    dummy_text_emb = jnp.zeros((self.num_pipeline_steps, B, 512), dtype=jnp.float32)
                    dummy_balancer_params = jnp.array([False, 1.0, 1.0])
                    dummy_balancer_params_stacked = jnp.repeat(dummy_balancer_params[None, :], self.num_pipeline_steps, axis=0)
                    
                    dummy_xs = (dummy_tokens_flat, dummy_text_emb, dummy_keys, dummy_balancer_params_stacked)
                    
                    dummy_lambdas = (1.0, 1.0, 0.25, 0.1, 1.0)
                    jitted_pipeline_fn(dummy_pipeline_state, dummy_xs, dummy_lambdas)

                compile_thread = threading.Thread(target=compile_task); compile_thread.start(); compile_thread.join()
                with self.ui_lock: self.ui_state["is_compiling"] = False

                if jitted_pipeline_fn is None:
                    self.console.print("[bold red]FATAL: JIT compilation failed or was interrupted.[/bold red]"); self.interactive_state.set_shutdown()
                else:
                    self.console.print("--- ✅ Compilation complete. Starting training... ---", style="bold green")
                    steps_per_epoch = num_train_samples // B if B > 0 else 1; total_epochs = self.args.epochs; start_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
                    
                    prefetch_buffer = deque(maxlen=2)
                    self.console.print("--- 🚀 Priming data prefetch buffer... ---")
                    for _ in range(2): prefetch_buffer.append(data_loading_executor.submit(_get_next_batch_task))

                    try:
                        for epoch in range(start_epoch, total_epochs):
                            if self.interactive_state.shutdown_event.is_set(): break
                            epoch_task = self.progress.add_task(f"Epoch {epoch + 1}/{total_epochs}", total=steps_per_epoch, loss=0.0, d_loss=0.0, sps=0.0, global_step=global_step)
            
                            for step_in_epoch in range(steps_per_epoch):
                                if self.interactive_state.shutdown_event.is_set(): break
                                
                                next_batch_future = prefetch_buffer.popleft(); real_tokens, text_emb = next_batch_future.result(); prefetch_buffer.append(data_loading_executor.submit(_get_next_batch_task))

                                t_loop_start = time.perf_counter()
                                
                                lambda_dict = self.lambda_controller(self.last_metrics)
                                current_lambdas = (lambda_dict['mlm'], lambda_dict['refinement'], lambda_dict['world_mmd'], lambda_dict['kids_adv'], lambda_dict['r1'])

                                self.d_loss_ema = (1 - self.d_loss_ema_alpha) * self.d_loss_ema + self.d_loss_ema_alpha * self.last_metrics.get('loss/kids_d', 0.5)
                                d_is_locked_out_py = False
                                if self.d_lockout_steps > 0: self.d_lockout_steps -= 1; d_is_locked_out_py = True
                                elif self.d_loss_ema < self.d_lockout_threshold: self.d_lockout_steps = self.d_lockout_duration; d_is_locked_out_py = True
                                
                                g_lr_mult = 1.0; d_lr_mult = 1.0
                                if not d_is_locked_out_py:
                                    if self.d_loss_ema < self.d_loss_target_min: d_lr_mult = 0.5; g_lr_mult = 1.5
                                    elif self.d_loss_ema > self.d_loss_target_max: d_lr_mult = 1.5; g_lr_mult = 0.5
                                
                                gan_balancer_params = jnp.array([d_is_locked_out_py, g_lr_mult, d_lr_mult])
                                gan_balancer_params_stacked = jnp.repeat(gan_balancer_params[None, :], self.num_pipeline_steps, axis=0)

                                tier_names_tuple = ("kids", "students", "polisher")
                                current_pipeline_state = PipelineState(all_states=tuple(initial_dynamic_states[name] for name in tier_names_tuple), q_controllers=tuple(initial_q_controllers[name] for name in tier_names_tuple), kids_d_state=initial_kids_d_state, prev_kid_ste=initial_kid_ste, prev_student_ste=initial_student_ste)
                                
                                key, *step_keys = jax.random.split(key, self.num_pipeline_steps + 1)
                                scan_keys = jnp.stack(step_keys)
                                scan_tokens = jnp.repeat(real_tokens[None, ...], self.num_pipeline_steps, axis=0)
                                scan_text_emb = jnp.repeat(text_emb[None, ...], self.num_pipeline_steps, axis=0)
                                xs_for_scan = (scan_tokens, scan_text_emb, scan_keys, gan_balancer_params_stacked)
                                
                                final_pipeline_state, stacked_outputs = jitted_pipeline_fn(current_pipeline_state, xs_for_scan, current_lambdas)
                                
                                stacked_metrics = stacked_outputs[1]
                                stacked_metrics['loss/total'][-1].block_until_ready()
                                total_step_time = time.perf_counter() - t_loop_start
                                
                                initial_dynamic_states = {name: state for name, state in zip(tier_names_tuple, final_pipeline_state.all_states)}
                                initial_q_controllers = {name: q_state for name, q_state in zip(tier_names_tuple, final_pipeline_state.q_controllers)}
                                initial_kids_d_state = final_pipeline_state.kids_d_state
                                
                                self.last_metrics = {k: v[-1].item() for k, v in stacked_metrics.items()}
                                
                                q_states_host = jax.device_get(initial_q_controllers)
                                with self.ui_lock:
                                    for tier_name, state in q_states_host.items():
                                        controller = self.agent_controllers[tier_name]
                                        controller.current_value = state.current_value.item(); controller.exploration_rate_q = state.exploration_rate.item(); controller.last_reward = state.last_reward.item(); controller._step_count = state.step_count.item(); controller.status = {0: "WARMUP", 1: "IMPROVING", 2: "STAGNATED", 3: "REGRESSING"}[state.status_code.item()]
                                
                                avg_loss = self.last_metrics['loss/total']
                                avg_d_loss = self.last_metrics['loss/kids_d']
                                global_step += self.num_pipeline_steps
                                sps = self.num_pipeline_steps / (total_step_time + 1e-9)
                                
                                with self.ui_lock: self.ui_state["steps_per_sec"] = sps
                                self.progress.update(epoch_task, advance=1, loss=avg_loss, d_loss=avg_d_loss, sps=sps, global_step=global_step)
                                
                                if (self.interactive_state.get_and_reset_force_validate() or (self.args.eval_every > 0 and (global_step // self.num_pipeline_steps) % self.args.eval_every == 0)) and not (self.active_preview_future and not self.active_preview_future.done()):
                                    with self.ui_lock: log_messages = self.ui_state.get("log_messages", deque(maxlen=5)); log_messages.append(f"[{time.strftime('%H:%M:%S')}] Spawning preview..."); self.ui_state["log_messages"] = log_messages
                                    host_states = jax.device_get(initial_dynamic_states); key, preview_key = jax.random.split(key)
                                    self.active_preview_future = preview_executor.submit(self._generate_preview_task, host_states, preview_key)
                                
                                if self.interactive_state.get_and_reset_force_save():
                                    with self.ui_lock: log_messages = self.ui_state.get("log_messages", deque(maxlen=5)); log_messages.append(f"[yellow]Forcing save to best...[/]"); self.ui_state["log_messages"] = log_messages
                                    self._save_checkpoint(best_ckpt_path, global_step, initial_dynamic_states, initial_kids_d_state, initial_q_controllers)

                                time.sleep(0.001)

                            self.progress.remove_task(epoch_task)
                            self._save_checkpoint(ckpt_path, global_step, initial_dynamic_states, initial_kids_d_state, initial_q_controllers)

                    except KeyboardInterrupt:
                        self.console.print("\n[bold yellow]Keyboard interrupt caught. Initiating graceful shutdown...[/]"); self.interactive_state.set_shutdown()
        finally:
            self.interactive_state.set_shutdown(); preview_executor.shutdown(wait=True); data_loading_executor.shutdown(wait=True)
            self.console.print("\n--- Training finished. Saving final state. ---")
            if 'initial_dynamic_states' in locals() and 'global_step' in locals():
                self._save_checkpoint(ckpt_path, global_step, initial_dynamic_states, initial_kids_d_state, initial_q_controllers)
   
   
   
# =================================================================================================
# 6. GENERATION & INFERENCE 
# =================================================================================================


class Generator:
    def __init__(self, args):
        self.args = args
        self.console = Console()
        self.console.print("--- 🧠 Loading Full Generative Stack (Chimera Ensemble) ---", style="bold yellow")
        
        try:
            conductor_ckpt_path = next(Path('.').glob(f"chimera_{args.basename}_*_final.pkl"))
            tokenizer_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_*_config.pkl"))
            p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
        except StopIteration:
            sys.exit(f"[FATAL] Could not find required model files for basename '{args.basename}'.")

        with open(tokenizer_config_path, 'rb') as f: self.tok_config_dict = pickle.load(f)

        self.dtype = jnp.float32
        p1_d_model = int(p1_path.stem.split('_')[-2].replace('d',''))
        
        self.p1_config = P1Config(d_model=p1_d_model, latent_grid_size=self.tok_config_dict['latent_grid_size'], input_image_size=512, dtype=self.dtype)
        
        self.tok_config_dict['dtype'] = self.dtype
        self.tok_config = TokenizerConfig(**self.tok_config_dict)
        
        self.population_configs = {}
        for tier_name, config in POPULATION_TIERS_CONFIG.items():
            model_config = {k: v for k, v in config.items() if k != 'count'}
            model_config['num_codes'] = self.tok_config.num_codes
            self.population_configs[tier_name] = ConductorConfig(**model_config)
        
        tok_ckpt_path = Path(str(tokenizer_config_path).replace("_config.pkl", "_best.pkl"))
        if not tok_ckpt_path.exists(): tok_ckpt_path = Path(str(tok_ckpt_path).replace("_best.pkl", "_final.pkl"))

        self.console.print(f"-> Loading Phase 1 AE from: [green]{p1_path}[/green]")
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']
        self.console.print(f"-> Loading Tokenizer from: [green]{tok_ckpt_path}[/green]")
        with open(tok_ckpt_path, 'rb') as f: tok_data = pickle.load(f); self.tok_params = tok_data.get('params', tok_data.get('gen_params'))
        
        self.console.print(f"-> Loading Chimera state from: [green]{conductor_ckpt_path}[/green]")
        with open(conductor_ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            self.stacked_params = {
                tier_name: state_data['params']
                for tier_name, state_data in ckpt['stacked_params_and_steps'].items()
            }
            # --- [NEW] Load the Kid Discriminator parameters ---
            if 'kids_d_params_and_step' in ckpt:
                self.stacked_params['kids_d'] = ckpt['kids_d_params_and_step']['params']
                self.console.print("-> [bold green]Loaded Kid Discriminator for guidance.[/bold green]")
            else:
                self.console.print("[bold yellow]Warning: Kid Discriminator params not found in checkpoint. Guidance disabled.[/bold yellow]")
                # Disable guidance if params are missing
                self.args.d_guidance_scale = 0.0

        self.uncond_embedding = jnp.mean(self.stacked_params['polisher']['uncond_embedding'], axis=0)
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        self.polisher_model_ref = PolisherConductor(**self.population_configs['polisher']._asdict())

        self.console.print("--- 🚀 JIT Compiling inference kernels... ---")
        self.console.print("✅ All kernels compiled.")

    def generate(self, prompt: str, seed: int, guidance_scale: float, decoding_steps: int, d_guidance_scale: float, d_guidance_top_k: int):
        self.console.print(f"--- 🎨 Generating image for prompt: \"[italic yellow]{prompt}[/italic yellow]\" ---")
        self.console.print(f"-> Seed: {seed}, CFG Scale: {guidance_scale}, Steps: {decoding_steps}")
        if d_guidance_scale > 0:
            self.console.print(f"-> [bold green]Discriminator Guidance ACTIVE[/]: Scale={d_guidance_scale}, Top-K={d_guidance_top_k}")

        key = jax.random.PRNGKey(seed)
        
        with torch.no_grad():
            cond_text_emb = self.clip_model.encode_text(clip.tokenize([prompt]).to(_clip_device)).cpu().numpy().astype(self.dtype)
        
        uncond_emb_batch = jnp.repeat(self.uncond_embedding[None, :], cond_text_emb.shape[0], axis=0)

        self.console.print("1/3: Generating token canvas via deep ensemble pipeline...")
        
        pm = self.polisher_model_ref
        grid_shape = (pm.depth_size, pm.height_size, pm.width_size)
        
        image_batch = _jitted_inference_pipeline(
            self.stacked_params, 
            self.tok_params,
            self.p1_params,
            key, cond_text_emb, uncond_emb_batch, 
            guidance_scale, decoding_steps, resolution=512,
            grid_shape=grid_shape,
            mask_token_id=pm.MASK_TOKEN_ID,
            vocab_size=pm.vocab_size,
            kid_config=self.population_configs['kids'],
            student_config=self.population_configs['students'],
            polisher_config=self.population_configs['polisher'],
            tok_config=self.tok_config,
            p1_config=self.p1_config,
            d_guidance_scale=d_guidance_scale,
            d_guidance_top_k=d_guidance_top_k
        )
        image_batch.block_until_ready()
        
        self.console.print("2/3: Finalizing image array...")
        recon_np = np.array(((image_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        
        self.console.print("3/3: Saving to disk...")
        filename = f"GEN_{Path(self.args.basename).stem}_{prompt.replace(' ', '_')[:40]}_{seed}_cfg{guidance_scale:.1f}_dg{d_guidance_scale:.1f}_s{decoding_steps}.png"
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
    p_gen.add_argument('--d-guidance-scale', type=float, default=150.0, help="Scale of the discriminator guidance. Set to 0 to disable.")
    p_gen.add_argument('--d-guidance-top-k', type=int, default=8, help="Number of top candidate tokens to evaluate with the discriminator for guidance.")

    
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
    elif args.command == "generate": 
        Generator(args).generate(
            args.prompt, 
            args.seed, 
            args.guidance_scale, 
            args.decoding_steps, 
            args.d_guidance_scale, 
            args.d_guidance_top_k
        )
    
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