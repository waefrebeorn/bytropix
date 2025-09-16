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

class PIDLambdaController:
    """
    An intelligent agent to dynamically balance multiple loss terms.
    """
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]]):
        self.targets = targets
        self.base_weights = base_weights
        self.gains = gains
        self.integral_error = {k: 0.0 for k in targets.keys()}
        self.last_error = {k: 0.0 for k in targets.keys()}
        self.derivative = {k: 0.0 for k in targets.keys()}

    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        final_lambdas = self.base_weights.copy()
        for name, target in self.targets.items():
            metric_key = f'loss/{name}'
            if metric_key not in last_metrics: continue

            kp, ki, kd = self.gains[name]
            current_loss = last_metrics.get(metric_key, target)
            error = current_loss - target
            
            self.integral_error[name] += error
            self.integral_error[name] = np.clip(self.integral_error[name], -5.0, 5.0)
            self.derivative[name] = error - self.last_error[name]
            
            adjustment = (kp * error) + (ki * self.integral_error[name]) + (kd * self.derivative[name])
            multiplier = np.exp(adjustment)
            
            calculated_lambda = self.base_weights.get(name, 0.0) * multiplier
            self.last_error[name] = error
            
            final_lambdas[name] = np.clip(calculated_lambda, 0.1, 10.0)

        return final_lambdas

    def state_dict(self):
        return {'integral_error': self.integral_error, 'last_error': self.last_error}
    
    def load_state_dict(self, state):
        self.integral_error = state.get('integral_error', self.integral_error)
        self.last_error = state.get('last_error', self.last_error)
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

    def decode(self, path_params_grid):
        """Decodes from a grid of continuous path parameters (shader blend)."""
        h_r = nn.gelu(self.dec_convT1(path_params_grid))
        return self.dec_convT2(h_r)

# --- State holder for GAN training ---
class GANTrainStates(NamedTuple):
    generator: TrainState
    discriminator: TrainState


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



def prepare_jenkins_data(args):
    """
    Pre-tokenizes the entire dataset and saves it to a TFRecord file for efficient,
    streaming access during Jenkins training. This avoids loading all latents into memory.
    """
    console = Console()
    output_path = Path(args.data_dir) / f"jenkins_data_{args.basename}_{args.num_codes}c.tfrecord"
    if output_path.exists():
        console.print(f"✅ Jenkins TFRecord data already exists at [green]{output_path}[/green]. Skipping.")
        return

    console.print("--- 🧠 Preparing Jenkins TFRecord Dataset ---", style="bold yellow")
    
    try:
        tok_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_{args.num_codes}c_gan_config.pkl"))
    except StopIteration:
        sys.exit(f"[FATAL] Could not find Jenkins-compatible tokenizer config file for pattern: tokenizer_{args.basename}_{args.num_codes}c_gan_config.pkl")
        
    tok_ckpt_path = Path(str(tok_config_path).replace("_config.pkl", "_best.pkl"))
    if not tok_ckpt_path.exists(): 
        tok_ckpt_path = Path(str(tok_ckpt_path).replace("_best.pkl", "_final.pkl"))
    
    if not tok_ckpt_path.exists():
        sys.exit(f"[FATAL] Found tokenizer config but could not find matching .pkl weights at {tok_ckpt_path}")

    with open(tok_config_path, 'rb') as f: config_dict = pickle.load(f)
    with open(tok_ckpt_path, 'rb') as f: 
        ckpt_data = pickle.load(f)
        tok_params = ckpt_data.get('gen_params', ckpt_data.get('params'))
        if tok_params is None:
            sys.exit(f"[FATAL] Could not find 'gen_params' or 'params' key in tokenizer checkpoint: {tok_ckpt_path}")

    tok_config = TokenizerConfig(**config_dict, dtype=jnp.float32)
    tokenizer = LatentTokenizerVQGAN(**tok_config._asdict())
    
    data_path = Path(args.data_dir) / f"paired_data_{args.basename}.pkl"
    if not data_path.exists(): sys.exit(f"FATAL: Paired data not found at {data_path}.")
    with open(data_path, 'rb') as f: data = pickle.load(f)

    # --- [THE FOOLPROOF FIX] ---
    # Explicitly check the dimension of the loaded embeddings.
    embedding_dim = data['embeddings'].shape[1]
    if embedding_dim != 512:
        console.print(f"[bold red]FATAL DATA MISMATCH![/bold red]")
        console.print(f"The file [cyan]{data_path}[/cyan] contains embeddings of dimension [yellow]{embedding_dim}[/yellow].")
        console.print("The Jenkins architecture requires 512-dimensional CLIP ViT-B/32 embeddings.")
        console.print("Please delete the file and re-run the `prepare-paired-data` command to fix this.")
        sys.exit(1)
    console.print(f"Verified embeddings have correct dimension: {embedding_dim}")

    jit_tokenizer_encode = jax.jit(lambda p, l: tokenizer.apply({'params': p}, l, method=tokenizer.encode))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_example(tokens, embedding):
        feature = {
            'tokens': _bytes_feature(tokens.tobytes()),
            'embedding': _bytes_feature(embedding.tobytes()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    console.print(f"Writing tokenized data to [green]{output_path}[/green]...")
    chunk_size = 256
    num_samples = len(data['latents'])
    
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for i in tqdm(range(0, num_samples, chunk_size), desc="Tokenizing and Writing"):
            latents_chunk = jnp.asarray(data['latents'][i:i + chunk_size])
            embeddings_chunk = data['embeddings'][i:i + chunk_size]
            
            tokens_chunk = jit_tokenizer_encode(tok_params, latents_chunk)
            tokens_chunk_cpu = np.array(tokens_chunk, dtype=np.int32)
            
            for j in range(len(tokens_chunk_cpu)):
                example = create_example(tokens_chunk_cpu[j], embeddings_chunk[j])
                writer.write(example.SerializeToString())
                
    console.print("✅ Jenkins data preparation complete.")


# Define immutable, hashable NamedTuples to hold static model configurations.
class TokenizerConfig(NamedTuple):
    num_codes: int; code_dim: int; latent_grid_size: int; dtype: Any
class P1Config(NamedTuple):
    d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any

# =================================================================================================
# 5. JENKINS SHADER CONDUCTOR (REPLACES GENERATIONAL CONDUCTOR)
# =================================================================================================
class JenkinsTrainState(TrainState):
    """A custom train state to hold both the active and EMA weights for Jenkins."""
    ema_params: Any

       
class JenkinsBlock(nn.Module):
    """A DiT-style block with adaptive layer norm and gated attention/MLP layers."""
    d_model: int
    num_heads: int
    dtype: Any
    
    @nn.compact
    def __call__(self, x, cond, train=True):
        # We need to define the layers inside __call__ for remat to work correctly
        # when layers are defined in a list comprehension in the parent.
        # The 'deterministic' flag is now correctly handled because 'train' will be static.
        attn = nn.SelfAttention(num_heads=self.num_heads, dtype=self.dtype, deterministic=not train)
        mlp = nn.Sequential([
            nn.Dense(self.d_model * 4, dtype=self.dtype),
            nn.gelu,
            nn.Dense(self.d_model, dtype=self.dtype),
        ])
        norm1 = nn.LayerNorm(dtype=self.dtype)
        norm2 = nn.LayerNorm(dtype=self.dtype)
        ada_ln_mod = nn.Sequential([
            nn.silu,
            nn.Dense(self.d_model * 6, dtype=self.dtype)
        ])

        mod = ada_ln_mod(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)

        res_x = x
        x = norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * attn(x)
        x = res_x + x
        
        res_x = x
        x = norm2(x) * (1 + scale_mlp) + shift_mlp
        x = res_x + gate_mlp * mlp(x)
        return x

# --- [THE DEFINITIVE FIX] ---
# The __call__ method of JenkinsBlock is called with (self, x, cond, train).
# The arguments are indexed starting from 0 (self). So 'train' is at index 3.
# We must tell remat that the argument at index 3 is static.
RematJenkinsBlock = remat(JenkinsBlock, static_argnums=(3,))



class ManifoldConductor(nn.Module):
    """
    The Manifold Conductor. It orchestrates a field of physics shaders based on a
    multi-modal command vector and a spatial "Salience Blueprint" derived from
    ground-truth depth and alpha maps, using a Masked Autoencoder objective.
    """
    num_codes: int; d_model: int; num_heads: int; num_layers: int
    latent_grid_size: int; dtype: Any
    
    def setup(self):
        grid_dim = self.latent_grid_size * self.latent_grid_size
        self.pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (1, grid_dim, self.d_model), self.dtype)
        
        # A learnable embedding for the [MASK] token
        self.mask_token_embed = self.param('mask_token', nn.initializers.normal(0.02), (1, 1, self.d_model), self.dtype)
        
        self.command_proj = nn.Dense(self.d_model, dtype=self.dtype)
        
        # Main embedding layer for shader tokens (num_codes) plus MASK and unused tokens
        self.token_embedding = nn.Embed(self.num_codes + 2, self.d_model, dtype=self.dtype)
        
        # Dedicated embedding layer for the Salience Blueprint's tokens
        self.salience_embedding = nn.Embed(self.num_codes, self.d_model, dtype=self.dtype)
        
        self.output_proj = nn.Sequential([
            nn.LayerNorm(dtype=self.dtype),
            nn.Dense(self.num_codes, dtype=self.dtype)
        ])
        
        self.blocks = [RematJenkinsBlock(self.d_model, self.num_heads, self.dtype, name=f'block_{i}') for i in range(self.num_layers)]

    def __call__(self, input_indices, command_vector, salience_indices, train=True):
        B, H, W = input_indices.shape
        
        # Embed the masked input indices, which are a mix of real tokens and MASK_TOKEN
        x = self.token_embedding(input_indices).reshape(B, H * W, -1)
        
        # Find where the MASK token is and replace its generic embedding with the learned MASK embedding
        mask_token_id = self.num_codes + 1 # As defined in the trainer
        mask = (input_indices.reshape(B, H*W) == mask_token_id)
        x = jnp.where(mask[..., None], self.mask_token_embed, x)
        
        # Embed the Salience Blueprint and add it as spatial conditioning
        salience_cond = self.salience_embedding(salience_indices.reshape(B, H * W))
        x += salience_cond
        x += self.pos_embed
        
        # Project text command and use as global conditioning for the DiT blocks
        c_emb = self.command_proj(command_vector)
        cond = c_emb[:, None, :] # Add sequence dimension for broadcasting
        
        for block in self.blocks:
            x = block(x, cond, train)
            
        new_shader_logits = self.output_proj(x)
        return new_shader_logits.reshape(B, H, W, self.num_codes)


class AdvancedTrainer:
    """Base class for training with advanced toolkit features."""
    def __init__(self, args):
        self.args = args
        self.interactive_state = InteractivityState()
        self.num_devices = jax.local_device_count()
        self.loss_history = deque(maxlen=200)

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

@partial(jax.jit, static_argnames=('train_step_fn', 'num_scan_steps', 'train'))
def jitted_scanned_train_step(
    state: "JenkinsTrainState", # Use string annotation to avoid forward declaration issues
    super_batch_latents, 
    super_batch_commands, 
    lambdas, # Add lambdas as an argument
    keys,
    train_step_fn,
    num_scan_steps,
    train
):
    """
    JIT-compiles a loop that executes the core train_step multiple times.
    This version is updated to work with the PID-controlled JenkinsTrainer.
    """
    def scan_body(carry_state, xs):
        # Unpack the data for a single step
        batch_latents, command_vector, key = xs
        
        # Execute one iteration of the original training logic, passing the lambdas
        new_state, metrics = train_step_fn(carry_state, batch_latents, command_vector, lambdas, key, train=train)
        
        # The new state is carried to the next iteration. Metrics are collected.
        return new_state, metrics

    # Run the scan over the "super-batch" of data
    final_state, collected_metrics = jax.lax.scan(
        scan_body,
        state,
        (super_batch_latents, super_batch_commands, keys)
    )
    
    return final_state, collected_metrics


class JenkinsTrainer(AdvancedTrainer):
    """
    The trainer for the Manifold Conductor. Implements Masked Shader Reconstruction,
    conditioned on a pre-computed "Salience Blueprint" derived from ground-truth
    depth and alpha maps.
    """
    def __init__(self, args):
        super().__init__(args)
        self.console = Console()
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.console.print(f"--- [MODE] Manifold Conductor (Salience-Conditioned) is [bold green]ACTIVE[/bold green]. ---", style="bold yellow")

        self.p1_params, self.p1_model, self.p1_config = self._load_phase1(args)
        self.tok_params, self.tokenizer, self.tok_config = self._load_tokenizer(args)
        
        self.uncond_embedding = jax.device_put(jnp.zeros((1, 512), dtype=self.dtype))

        self.model = ManifoldConductor(
            num_codes=args.num_codes,
            d_model=args.d_model_cond,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            latent_grid_size=args.latent_grid_size // 4,
            dtype=self.dtype
        )
        self.last_metrics_for_ui = {}
        self.ui_lock = threading.Lock()
        self.loss_hist = deque(maxlen=400)
        self.rendered_preview = None
        self.current_preview_prompt_idx = 0
        self.validation_prompts = ["a red cup on a table", "a photorealistic orange cat", "a blue ball", "green grass with a single tree", "a purple sports car"]
        self.clip_model_torch, _ = clip.load("ViT-B/32", device=_clip_device)
        with torch.no_grad():
            text_tokens = clip.tokenize(self.validation_prompts).to(_clip_device)
            self.validation_embeddings = jax.device_put(torch.from_numpy(self.clip_model_torch.encode_text(text_tokens).cpu().numpy()).to(torch.float32).numpy().astype(self.dtype))

    def _load_phase1(self, args):
        p1_path = next(Path('.').glob(f"{args.basename}_{args.d_model}d_512.pkl"))
        with open(p1_path, 'rb') as f: p1_params = pickle.load(f)['params']
        p1_config = P1Config(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=512, dtype=jnp.float32)
        p1_model = TopologicalCoordinateGenerator(**p1_config._asdict())
        return jax.device_put(p1_params), p1_model, p1_config

    def _load_tokenizer(self, args):
        try:
            tok_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_{args.num_codes}c_gan_config.pkl"))
        except StopIteration:
            sys.exit(f"[FATAL] Could not find tokenizer config file for pattern: tokenizer_{args.basename}_{args.num_codes}c_gan_config.pkl")
        tok_ckpt_path = Path(str(tok_config_path).replace("_config.pkl", "_best.pkl"))
        if not tok_ckpt_path.exists(): tok_ckpt_path = Path(str(tok_ckpt_path).replace("_best.pkl", "_final.pkl"))
        with open(tok_config_path, 'rb') as f: config_dict = pickle.load(f)
        with open(tok_ckpt_path, 'rb') as f: ckpt_data = pickle.load(f); params = ckpt_data.get('gen_params', ckpt_data.get('params'))
        tok_config = TokenizerConfig(**config_dict, dtype=jnp.float32)
        tokenizer = LatentTokenizerVQGAN(**tok_config._asdict())
        return jax.device_put(params), tokenizer, tok_config

    def _get_common_train_setup(self):
        console = Console()
        paired_data_path = Path(self.args.data_dir) / f"paired_data_{self.args.basename}.pkl"
        depth_path = Path(self.args.data_dir) / f"depth_maps_{self.args.basename}_{self.args.depth_layers}l.npy"
        alpha_path = Path(self.args.data_dir) / f"alpha_maps_{self.args.basename}.npy"
        
        if not all([paired_data_path.exists(), depth_path.exists(), alpha_path.exists()]):
            sys.exit(f"FATAL: Missing data files. Please run `prepare-paired-data` and `prepare_manifold_data.py`.")
            
        console.print("--- Loading all data into memory for Manifold training... ---")
        with open(paired_data_path, 'rb') as f: data = pickle.load(f)
        latents, embeddings = np.asarray(data['latents'], dtype=np.float32), np.asarray(data['embeddings'], dtype=np.float32)
        depth_maps = np.load(depth_path).astype(np.float32)
        alpha_maps = np.load(alpha_path).astype(np.float32)
        num_train_samples = len(latents)
        
        train_ds = tf.data.Dataset.from_tensor_slices((latents, embeddings, depth_maps, alpha_maps))
        train_ds = train_ds.shuffle(num_train_samples, seed=self.args.seed).repeat().batch(self.args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True)
        key_listener_thread.start()
        
        return key_listener_thread, train_ds.as_numpy_iterator(), num_train_samples
            
    def _generate_layout(self):
        layout = Layout()
        layout.split(Layout(name="header", size=3), Layout(ratio=1, name="main"), Layout(size=3, name="footer"))
        mem, util = self._get_gpu_stats()
        loss = self.last_metrics_for_ui.get('loss/total', 0.0)
        header_text = f"🌋 [bold]Manifold Conductor[/] | Masked Shader Loss: [yellow]{loss:.4f}[/] | GPU: {mem} / {util}"
        layout["header"].update(Panel(Align.center(header_text), style="bold red", title="[dim]wubumind.ai[/dim]", title_align="right"))
        main_table = Table.grid(expand=True); main_table.add_column(ratio=1); main_table.add_column(ratio=2)
        loss_panel = Panel(Align.center(self._get_sparkline(self.loss_hist, 50)), title="Loss", height=5, border_style="cyan")
        preview_content = Align.center("...Awaiting First Generation...")
        if self.rendered_preview and Pixels:
             preview_content = Align.center(Group(Text(f"Prompt: \"{self.validation_prompts[self.current_preview_prompt_idx]}\"", justify="center"), self.rendered_preview), fit=True)
        main_table.add_row(loss_panel, Panel(preview_content, title="Live Preview"))
        layout["main"].update(main_table)
        layout["footer"].update(self.progress)
        return layout

    def train(self):
        key_listener_thread, train_iterator, num_train_samples = self._get_common_train_setup()
        preview_executor = ThreadPoolExecutor(max_workers=1)
        
        key = jax.random.PRNGKey(self.args.seed)
        key, init_key = jax.random.split(key)

        optimizer = optax.adamw(self.args.lr, b1=0.9, b2=0.95)
        
        H = W = self.args.latent_grid_size // 4
        dummy_indices = jnp.zeros((1, H, W), dtype=jnp.int32)
        dummy_command = jnp.zeros((1, 512), dtype=self.dtype)
        
        params = self.model.init({'params': init_key, 'dropout': init_key}, dummy_indices, dummy_command, dummy_indices, train=False)['params']
        state = JenkinsTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=params)

        jit_encoder = jax.jit(lambda p, grid: self.tokenizer.apply({'params': p}, grid, method=self.tokenizer.encode))
        
        @partial(jax.jit, static_argnames=('train',))
        def train_step(state: JenkinsTrainState, path_params_grid, command_vector, depth_map, alpha_map, key, train: bool):
            
            target_indices = jit_encoder(self.tok_params, path_params_grid)
            
            salience_map = (depth_map / self.args.depth_layers) * 0.5 + (alpha_map / 255.0) * 0.5
            salience_indices = jit_encoder(self.tok_params, jnp.repeat(salience_map[..., None], 3, axis=-1))

            B, H, W = target_indices.shape
            num_to_keep = int(H * W * (1 - self.args.mask_ratio))
            
            key, mask_key, dropout_key = jax.random.split(key, 3)
            
            noise = jax.random.uniform(mask_key, (B, H * W))
            ids_shuffle = jnp.argsort(noise, axis=1)
            ids_keep = ids_shuffle[:, :num_to_keep]

            MASK_TOKEN = self.args.num_codes + 1
            masked_indices = jnp.full((B, H * W), MASK_TOKEN, dtype=jnp.int32)
            
            visible_indices = jnp.take_along_axis(target_indices.reshape(B, H*W), ids_keep, axis=1)
            masked_indices = masked_indices.at[jnp.arange(B)[:, None], ids_keep].set(visible_indices)
            input_indices = masked_indices.reshape(B,H,W)

            def loss_fn(params):
                is_unconditional = jax.random.uniform(dropout_key, (B, 1)) < 0.1
                final_command_vector = jnp.where(is_unconditional, self.uncond_embedding, command_vector)

                predicted_logits = state.apply_fn(
                    {'params': params, 'dropout': dropout_key}, 
                    input_indices, final_command_vector, salience_indices, train=train
                )
                
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=predicted_logits.reshape(-1, self.args.num_codes),
                    labels=target_indices.reshape(-1)
                ).reshape(B, H*W)
                
                loss_mask = jnp.ones((B, H*W), dtype=jnp.int32).at[jnp.arange(B)[:, None], ids_keep].set(0)
                final_loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)
                
                return final_loss, {'loss/total': final_loss}

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            new_state = state.apply_gradients(grads=grads)
            
            new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1 - 0.999), state.ema_params, new_state.params)
            new_state = new_state.replace(ema_params=new_ema_params)
            
            return new_state, metrics

        self.console.print(f"--- JIT Compiling Manifold train step... ---")
        dummy_latents = jnp.zeros((self.args.batch_size, self.args.latent_grid_size, self.args.latent_grid_size, 3), dtype=jnp.float32)
        dummy_commands = jnp.zeros((self.args.batch_size, 512), dtype=self.dtype)
        dummy_depth = jnp.zeros((self.args.batch_size, 512, 512), dtype=jnp.float32)
        dummy_alpha = jnp.zeros((self.args.batch_size, 512, 512), dtype=jnp.float32)
        state, _ = train_step(state, dummy_latents, dummy_commands, dummy_depth, dummy_alpha, key, train=True)
        self.console.print("--- ✅ Compilation complete. ---")

        steps_per_epoch = num_train_samples // self.args.batch_size
        total_steps = steps_per_epoch * self.args.epochs
        self.progress = Progress(BarColumn(), TextColumn("{task.description}"))
        task = self.progress.add_task("Training...", total=total_steps)
        
        active_preview_future = None
        global_step = 0
        try:
            with Live(self._generate_layout(), screen=True, redirect_stderr=False) as live:
                while global_step < total_steps:
                    if self.interactive_state.shutdown_event.is_set(): break
                    
                    latents, commands, depths, alphas = next(train_iterator)
                    key, step_key = jax.random.split(key)
                    
                    state, metrics = train_step(state, latents, commands, depths, alphas, step_key, train=True)
                    
                    with self.ui_lock:
                        self.last_metrics_for_ui = {k: v.item() for k,v in metrics.items()}
                        self.loss_hist.append(self.last_metrics_for_ui['loss/total'])
                    
                    self.progress.update(task, advance=1, description=f"Step {global_step+1}/{total_steps}")
                    
                    # Preview logic will need to be updated for MAE-style inference
                    # if global_step > 0 and global_step % self.args.eval_every == 0: ...

                    live.update(self._generate_layout())
                    global_step += 1
        finally:
            self.interactive_state.shutdown_event.set()
            preview_executor.shutdown(wait=True)
            ckpt_path = Path(f"manifold_{args.basename}_{args.num_codes}c.pkl")
            self.console.print(f"\n--- Training finished. Saving final EMA state to {ckpt_path} ---")
            final_state_host = jax.device_get(state)
            with open(ckpt_path, 'wb') as f:
                pickle.dump({'ema_params': final_state_host.ema_params, 'step': final_state_host.step}, f)
            self.console.print("✅ Save complete.")

    # Note: _update_preview_task and the Generator class need to be updated for the new MAE inference logic.
    def _update_preview_task(self, ema_params, key):
        prompt_idx_change = self.interactive_state.get_and_reset_preview_change()
        with self.ui_lock:
            if prompt_idx_change != 0:
                self.current_preview_prompt_idx = (self.current_preview_prompt_idx + prompt_idx_change) % len(self.validation_prompts)
            command_vector = self.validation_embeddings[self.current_preview_prompt_idx][None, :]

        # --- [THE ALIGNMENT FIX] ---
        # Previews use a lower resolution for speed, suitable for the terminal UI.
        resolution = 128
        patch_size = 64

        image_batch = _jitted_jenkins_inference(
            ema_params, self.model.apply, self.p1_params, self.p1_model, self.tok_params, self.tokenizer,
            command_vector, self.uncond_embedding, key,
            resolution=resolution, patch_size=patch_size, num_steps=20,
            grid_size=self.args.latent_grid_size//4, num_codes=self.args.num_codes,
            guidance_scale=4.0
        )
        image_batch.block_until_ready()
        img_np = np.array(((image_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        
        if Pixels:
            with self.ui_lock:
                self.rendered_preview = Pixels.from_image(Image.fromarray(img_np))



# =================================================================================================
# 6. GENERATION & INFERENCE 
# =================================================================================================

@partial(jax.jit, static_argnames=('jenkins_apply_fn', 'p1_apply_fn', 'tok_apply_fn', 'resolution', 'patch_size', 'num_steps', 'grid_size', 'num_codes'))
def _jitted_jenkins_inference(
    jenkins_params, jenkins_apply_fn, p1_params, p1_apply_fn, tok_params, tok_apply_fn,
    command_vector, uncond_vector, key,
    resolution, patch_size, num_steps, grid_size, num_codes, guidance_scale
):
    B = command_vector.shape[0]
    
    def render(shader_weights, codebook):
        path_params = shader_weights @ codebook
        # Use the passed-in apply functions
        full_latent_grid = tok_apply_fn({'params': tok_params}, path_params, method='decode')
        coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
        coord_chunks = jnp.array_split(coords, (resolution**2)//(patch_size**2))
        pixels_list=[p1_apply_fn({'params': p1_params}, full_latent_grid, c, method='decode') for c in coord_chunks]
        return jnp.concatenate(pixels_list,axis=1).reshape(B, resolution, resolution, 3)

    def loop_body(i, carry):
        shader_grid, key = carry
        t = jnp.ones((B,)) * (1.0 - i / num_steps)
        
        logits_cond = jenkins_apply_fn({'params': jenkins_params}, shader_grid, command_vector, t, train=False)
        logits_uncond = jenkins_apply_fn({'params': jenkins_params}, shader_grid, uncond_vector, t, train=False)
        
        logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        pred_shader_weights = jax.nn.softmax(logits, axis=-1)
        
        alpha = 1.0 / (num_steps - i + 1e-6) # Added epsilon for stability
        new_shader_grid = (1 - alpha) * shader_grid + alpha * pred_shader_weights
        
        return new_shader_grid, key

    initial_shaders = jnp.ones((B, grid_size, grid_size, num_codes), dtype=command_vector.dtype) / num_codes
    
    final_shaders, _ = jax.lax.fori_loop(0, num_steps, loop_body, (initial_shaders, key))
    
    codebook = tok_params['vq']['codebook'].T.astype(command_vector.dtype)
    return render(final_shaders, codebook)
    
class Generator:
    def __init__(self, args):
        self.args = args
        self.console = Console()
        self.console.print("--- 🧠 Loading Jenkins Shader Conductor V2 ---", style="bold green")
        
        self.dtype = jnp.float32
        
        p1_params_host, self.p1_model, _ = JenkinsTrainer._load_phase1(self, args)
        self.p1_params = jax.device_get(p1_params_host)
        
        tok_params_host, self.tokenizer, _ = JenkinsTrainer._load_tokenizer(self, args)
        self.tok_params = jax.device_get(tok_params_host)
        
        self.model = JenkinsConductor(
            num_codes=args.num_codes, d_model=args.d_model_cond, num_heads=args.num_heads,
            num_layers=args.num_layers, latent_grid_size=args.latent_grid_size // 4, dtype=self.dtype
        )
        
        ckpt_path = Path(f"jenkins_{args.basename}_{args.num_codes}c.pkl")
        if not ckpt_path.exists():
            sys.exit(f"[FATAL] Could not find Jenkins checkpoint file at: {ckpt_path}")
            
        self.console.print(f"-> Loading Jenkins weights from [green]{ckpt_path}[/green]")
        with open(ckpt_path, 'rb') as f:
            ckpt_data = pickle.load(f)
            self.params = jax.device_get(ckpt_data['ema_params'])
            
        self.uncond_embedding = jnp.zeros((1, 512), dtype=self.dtype)
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        
        self.console.print("--- 🚀 JIT Compiling Jenkins inference pipeline... ---")
        _ = self.generate("a test", 42, 4.0, 2, True)
        self.console.print("--- ✅ Compilation Complete ---")

    def generate(self, prompt: str, seed: int, guidance_scale: float, decoding_steps: int, _compile_run=False):
        if not _compile_run:
            self.console.print(f"--- 🎨 Orchestrating shaders for: \"[italic yellow]{prompt}[/italic yellow]\" ---")
        
        key = jax.random.PRNGKey(seed)
        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(_clip_device)
            command_vector = self.clip_model.encode_text(text_tokens).cpu().numpy().astype(np.float32)
        
        resolution = 512
        patch_size = 256
        
        # --- [THE FIX] ---
        # Pass the .apply methods of the models, not the model objects themselves.
        image_batch = _jitted_jenkins_inference(
            self.params, self.model.apply, 
            self.p1_params, self.p1_model.apply, 
            self.tok_params, self.tokenizer.apply,
            command_vector, self.uncond_embedding, key,
            resolution=resolution, patch_size=patch_size, num_steps=decoding_steps,
            grid_size=self.args.latent_grid_size//4, num_codes=self.args.num_codes,
            guidance_scale=guidance_scale
        )
        image_batch.block_until_ready()
        
        if _compile_run: return None
        
        img_np = np.array(((image_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        filename = f"JENKINS_{Path(self.args.basename).stem}_{prompt.replace(' ', '_')[:40]}_{seed}.png"
        Image.fromarray(img_np).save(filename)
        self.console.print(f"✅ Image saved to [green]{filename}[/green]")

# =================================================================================================
# 7. MAIN EXECUTION BLOCK
# =================================================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Jenkins Shader Conductor", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--basename', type=str, required=True, help="Base name for the model set (e.g., 'laion_ae').")
    
    # --- Data Prep Commands ---
    p_prep = subparsers.add_parser("prepare-paired-data", help="Pre-process (image, text) -> (latent, embedding).", parents=[base_parser])
    p_prep.add_argument('--data-dir', type=str, required=True); p_prep.add_argument('--d-model', type=int, required=True); p_prep.add_argument('--latent-grid-size', type=int, required=True); p_prep.add_argument('--batch-size', type=int, default=128)

    p_prep_tok = subparsers.add_parser("prepare-tokenizer-data", help="Pre-process images into latents for the tokenizer.", parents=[base_parser])
    p_prep_tok.add_argument('--data-dir', type=str, required=True); p_prep_tok.add_argument('--d-model', type=int, required=True); p_prep_tok.add_argument('--latent-grid-size', type=int, required=True); p_prep_tok.add_argument('--batch-size', type=int, default=128)

    # --- Tokenizer Commands ---
    p_tok = subparsers.add_parser("train-tokenizer", help="Train the Latent Tokenizer (VQ-GAN).", parents=[base_parser])
    p_tok.add_argument('--data-dir', type=str, required=True); p_tok.add_argument('--d-model', type=int, required=True); p_tok.add_argument('--latent-grid-size', type=int, required=True)
    p_tok.add_argument('--epochs', type=int, default=100); p_tok.add_argument('--batch-size', type=int, default=128); p_tok.add_argument('--lr', type=float, default=3e-4)
    p_tok.add_argument('--eval-every', type=int, default=1000); p_tok.add_argument('--num-codes', type=int, default=3072); p_tok.add_argument('--code-dim', type=int, default=256)
    p_tok.add_argument('--use-bfloat16', action='store_true')

    p_check_tok = subparsers.add_parser("check-tokenizer", help="Reconstruct an image to check tokenizer quality.", parents=[base_parser])
    p_check_tok.add_argument('--image-path', type=str, required=True); p_check_tok.add_argument('--output-path', type=str, default="tokenizer_recon_check.png")

    # --- Jenkins Conductor Commands ---
    jenkins_base_parser = argparse.ArgumentParser(add_help=False, parents=[base_parser])
    jenkins_base_parser.add_argument('--num-codes', type=int, default=3072)
    jenkins_base_parser.add_argument('--d-model-cond', type=int, default=1024, help="d_model for the Jenkins Conductor.")
    jenkins_base_parser.add_argument('--num-layers', type=int, default=16, help="Number of layers in the Jenkins Conductor.")
    jenkins_base_parser.add_argument('--num-heads', type=int, default=16, help="Number of attention heads.")
    
    p_jenkins = subparsers.add_parser("train-jenkins", help="Train the Jenkins Shader Conductor.", parents=[jenkins_base_parser])
    p_jenkins.add_argument('--data-dir', type=str, required=True)
    p_jenkins.add_argument('--d-model', type=int, required=True)
    p_jenkins.add_argument('--latent-grid-size', type=int, required=True)
    p_jenkins.add_argument('--epochs', type=int, default=100); p_jenkins.add_argument('--batch-size', type=int, default=16)
    p_jenkins.add_argument('--lr', type=float, default=1e-4); p_jenkins.add_argument('--eval-every', type=int, default=500)
    p_jenkins.add_argument('--use-bfloat16', action='store_true'); p_jenkins.add_argument('--seed', type=int, default=42)

    p_gen = subparsers.add_parser("generate", help="Generate an image using Jenkins.", parents=[jenkins_base_parser])
    p_gen.add_argument('--prompt', type=str, required=True); p_gen.add_argument('--seed', type=int, default=lambda: int(time.time()))
    p_gen.add_argument('--guidance-scale', type=float, default=4.0); p_gen.add_argument('--decoding-steps', type=int, default=20)
    p_gen.add_argument('--d-model', type=int, required=True, help="The d_model of the Phase 1 AE used for training.")
    p_gen.add_argument('--latent-grid-size', type=int, required=True, help="The latent_grid_size of the Phase 1 AE used for training.")

    args = parser.parse_args()
    
    if args.command == "generate": args.seed = args.seed() if callable(args.seed) else args.seed

    if args.command == "prepare-paired-data": prepare_paired_data(args)
    elif args.command == "prepare-tokenizer-data": prepare_tokenizer_data(args)
    elif args.command == "train-tokenizer": TokenizerTrainer(args).train()
    elif args.command == "check-tokenizer": run_tokenizer_check(args)
    elif args.command == "train-jenkins": JenkinsTrainer(args).train()
    elif args.command == "generate": Generator(args).generate(args.prompt, args.seed, args.guidance_scale, args.decoding_steps)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console = Console()
        console.print("\n[bold red]FATAL ERROR ENCOUNTERED[/bold red]")
        console.print_exception(show_locals=False)
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write("A fatal error occurred. Full traceback:\n")
            f.write(traceback.format_exc())
        console.print("\n[yellow]Full traceback has been written to [bold]crash_log.txt[/bold][/yellow]")