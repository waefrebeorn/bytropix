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
class InteractivityState:
    """A thread-safe class to hold shared state for interactive controls."""
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_prompt_change = 0  # -1 for prev, 1 for next
        self.sentinel_dampening_log_factor = -1.0
        self.shutdown_event = threading.Event()

    def get_and_reset_preview_change(self):
        with self.lock:
            change = self.preview_prompt_change
            self.preview_prompt_change = 0
            return change

    def update_sentinel_factor(self, direction):
        with self.lock:
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
                    if char == '\x1b': # ESC sequence
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
    """A Q-Learning agent to dynamically control the learning rate."""
    def __init__(self,initial_lr:float,config:Dict[str,Any]):
        self.config=config; self.initial_lr=initial_lr; self.current_lr=initial_lr; self.q_table_size=int(self.config["q_table_size"]); self.num_actions=int(self.config["num_lr_actions"]); self.lr_change_factors=self.config["lr_change_factors"]; self.q_table=np.zeros((self.q_table_size,self.num_actions),dtype=np.float32); self.learning_rate_q=float(self.config["learning_rate_q"]); self.discount_factor_q=float(self.config["discount_factor_q"]); self.lr_min=float(self.config["lr_min"]); self.lr_max=float(self.config["lr_max"]); self.loss_history=deque(maxlen=int(self.config["metric_history_len"])); self.loss_min=float(self.config["loss_min"]); self.loss_max=float(self.config["loss_max"]); self.last_action_idx:Optional[int]=None; self.last_state_idx:Optional[int]=None; self.initial_exploration_rate = float(self.config["exploration_rate_q"]); self.exploration_rate_q = self.initial_exploration_rate; self.min_exploration_rate = float(self.config["min_exploration_rate"]); self.exploration_decay = float(self.config["exploration_decay"]); self.status: str = "STARTING"; self.last_reward: float = 0.0; self.trend_window = int(config["trend_window"]); self.trend_history = deque(maxlen=self.trend_window); self.improve_threshold = float(config["improve_threshold"]); self.regress_threshold = float(config["regress_threshold"]); self.regress_penalty = float(config["regress_penalty"]); self.stagnation_penalty = float(config["stagnation_penalty"]); self.last_slope: float = 0.0; self.warmup_steps = int(config.get("warmup_steps", 0)); self.warmup_lr_start = float(config.get("warmup_lr_start", 1e-7)); self._step_count = 0
        print(f"--- Q-Controller initialized. LR Warmup: {self.warmup_steps} steps. Trend Window: {self.trend_window} steps ---")
    def _discretize_value(self,value:float) -> int:
        if not np.isfinite(value): return self.q_table_size // 2
        if value<=self.loss_min: return 0
        if value>=self.loss_max: return self.q_table_size-1
        return min(int((value-self.loss_min)/((self.loss_max-self.loss_min)/self.q_table_size)),self.q_table_size-1)
    def _get_current_state_idx(self) -> int:
        if not self.loss_history: return self.q_table_size//2
        return self._discretize_value(np.mean(list(self.loss_history)[-5:]))
    def choose_action(self) -> float:
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            alpha = self._step_count / self.warmup_steps
            self.current_lr = self.warmup_lr_start * (1 - alpha) + self.initial_lr * alpha
            self.status = f"WARMUP (LR) {self._step_count}/{self.warmup_steps}"
            return self.current_lr
        self.last_state_idx = self._get_current_state_idx()
        if np.random.rand() < self.exploration_rate_q: self.last_action_idx = np.random.randint(0, self.num_actions)
        else: self.last_action_idx = np.argmax(self.q_table[self.last_state_idx]).item()
        self.current_lr = np.clip(self.current_lr * self.lr_change_factors[self.last_action_idx], self.lr_min, self.lr_max)
        return self.current_lr
    def update_q_value(self, total_loss:float):
        self.loss_history.append(total_loss); self.trend_history.append(total_loss)
        if self._step_count <= self.warmup_steps: return
        if self.last_state_idx is None or self.last_action_idx is None: return
        reward = self._calculate_reward(total_loss); self.last_reward = reward
        current_q = self.q_table[self.last_state_idx, self.last_action_idx]
        next_state_idx = self._get_current_state_idx()
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor_q * np.max(self.q_table[next_state_idx]) - current_q)
        self.q_table[self.last_state_idx, self.last_action_idx] = new_q
        self.exploration_rate_q = max(self.min_exploration_rate, self.exploration_rate_q * self.exploration_decay)
    def _calculate_reward(self, current_loss):
        if len(self.trend_history) < self.trend_window:
            self.status = f"WARMING UP (TREND) {len(self.trend_history)}/{self.trend_window}"; return 0.0
        slope = np.polyfit(np.arange(self.trend_window), np.array(self.trend_history), 1)[0]; self.last_slope = slope
        if slope < -self.improve_threshold: self.status = f"IMPROVING (S={slope:.2e})"; reward = abs(slope) * 1000
        elif slope > self.regress_threshold: self.status = f"REGRESSING (S={slope:.2e})"; reward = -abs(slope) * 1000 - self.regress_penalty
        else: self.status = f"STAGNATED (S={slope:.2e})"; reward = self.stagnation_penalty
        return reward
    def state_dict(self)->Dict[str,Any]:
        return {"current_lr":self.current_lr, "q_table":self.q_table.tolist(), "loss_history":list(self.loss_history), "exploration_rate_q":self.exploration_rate_q, "trend_history": list(self.trend_history), "_step_count": self._step_count}
    def load_state_dict(self,state_dict:Dict[str,Any]):
        self.current_lr=state_dict.get("current_lr",self.current_lr); self.q_table=np.array(state_dict.get("q_table",self.q_table.tolist()),dtype=np.float32); self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.loss_history.maxlen); self.exploration_rate_q=state_dict.get("exploration_rate_q", self.initial_exploration_rate); self.trend_history=deque(state_dict.get("trend_history",[]),maxlen=self.trend_window); self._step_count=state_dict.get("_step_count", 0)


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
    A robust multi-head attention module designed to work with a pre-allocated
    Key-Value (KV) cache for efficient autoregressive generation within jax.lax.scan.
    """
    num_heads: int
    dtype: Any = jnp.float32
    # --- [SOLUTION] Add num_positions to the module's signature ---
    # This makes it an expected parameter during initialization.
    # It's optional because it's only needed for decoding cache setup.
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
        
        if decode and is_self_attn:
            # --- [SOLUTION] Ensure num_positions is available ---
            if self.num_positions is None:
                raise ValueError("num_positions must be provided to StandardAttention during decoding.")

            # Initialize the pre-allocated cache using the provided num_positions.
            cache_k = self.variable('cache', 'cached_key', lambda: jnp.zeros((x.shape[0], self.num_positions, d_model), dtype=self.dtype))
            cache_v = self.variable('cache', 'cached_value', lambda: jnp.zeros((x.shape[0], self.num_positions, d_model), dtype=self.dtype))

            k_new = k_proj(kv_source)
            v_new = v_proj(kv_source)

            # Write the new key/value into the cache at the current index.
            cache_k.value = cache_k.value.at[:, index, :].set(k_new.squeeze(axis=1))
            cache_v.value = cache_v.value.at[:, index, :].set(v_new.squeeze(axis=1))
            
            k = cache_k.value
            v = cache_v.value
        else:
            k = k_proj(kv_source)
            v = v_proj(kv_source)

        B, L_q, _ = q.shape
        _, L_kv, _ = k.shape
        
        q_heads = q.reshape(B, L_q, self.num_heads, head_dim)
        k_heads = k.reshape(B, L_kv, self.num_heads, head_dim)
        v_heads = v.reshape(B, L_kv, self.num_heads, head_dim)

        attn_output = dot_product_attention(q_heads, k_heads, v_heads, mask=mask, dtype=self.dtype)

        return out_proj(attn_output.reshape(B, L_q, d_model))


class TransformerBlock(nn.Module):
    """
    [FIXED] Transformer block that correctly defines all submodules in setup()
    to ensure a static parameter structure for JAX compatibility.
    """
    num_heads: int; d_model: int; num_positions: int; dtype: Any = jnp.float32

    def setup(self):
        # Define all layers and modules here, once.
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
            name='ca'
            # num_positions is not needed for cross-attention as it doesn't cache.
        )
        self.ln3 = nn.LayerNorm(dtype=self.dtype)
        self.mlp_dense1 = nn.Dense(self.d_model * 4, dtype=self.dtype, kernel_init=initializers.xavier_uniform())
        self.mlp_dense2 = nn.Dense(self.d_model, dtype=self.dtype, kernel_init=initializers.xavier_uniform())

    def __call__(self, x, context, mask, decode: bool = False, index: Optional[jnp.ndarray] = None):
        # Use the pre-defined modules from self.
        sa_input = self.ln1(x)
        x = x + self.sa(sa_input, mask=mask, decode=decode, index=index)

        ca_input = self.ln2(x)
        x = x + self.ca(ca_input, context=context)

        ffn_input = self.ln3(x)
        h = nn.gelu(self.mlp_dense1(ffn_input))
        h = self.mlp_dense2(h)
        x = x + h
        return x



# In GenerativeConductor class

class GenerativeConductor(nn.Module):
    num_codes: int; num_positions: int; d_model: int; num_heads: int; num_layers: int; clip_dim: int
    # [ADD] Add a dropout rate for the text condition
    uncond_drop_rate: float = 0.1
    dtype: Any = jnp.float32

    def setup(self):
        """
        Defines the entire static structure of the model.
        """
        # [ADD] Create a learnable parameter for the unconditional embedding.
        # This will represent the "null prompt".
        self.uncond_embedding = self.param(
            'uncond_embedding',
            nn.initializers.normal(0.02),
            (1, self.clip_dim), self.dtype
        )
        self.token_embedding = nn.Embed(self.num_codes + 1, self.d_model, name='token_embedding', dtype=self.dtype)
        self.pos_embedding = self.param('pos_embedding', nn.initializers.normal(.02), (self.num_positions, self.d_model), self.dtype)
        self.text_projection = nn.Dense(self.d_model, name='text_projection', dtype=self.dtype)
        self.logit_head = nn.Dense(self.num_codes, name='logit_head', dtype=self.dtype)
        self.norm = nn.LayerNorm(dtype=self.dtype)
        
        self.blocks = [
            TransformerBlock(self.num_heads, self.d_model, self.num_positions, self.dtype, name=f'block_{i}')
            for i in range(self.num_layers)
        ]

    def __call__(self, tokens, text_emb, train: bool = False, decode: bool = False, index: Optional[jnp.ndarray] = None):
        B, L = tokens.shape
        
        # [ADD] The core CFG training logic
        if train:
            # Get a dropout RNG stream
            key = self.make_rng('dropout')
            
            # Create a random mask for which batch items to drop the prompt for
            should_drop = jax.random.bernoulli(key, self.uncond_drop_rate, (B, 1))
            
            # Use the mask to select between the real text embedding and the learned unconditional one
            text_emb = jnp.where(should_drop, self.uncond_embedding, text_emb)

        tok_emb = self.token_embedding(tokens)
        
        if decode:
            chex.assert_rank(index, 0)
            positional_embedding = self.pos_embedding[index]
            x = tok_emb + positional_embedding[None, None, :]
        else:
            positional_embedding = self.pos_embedding[:L]
            x = tok_emb + positional_embedding[None, :, :]

        x = x.astype(self.dtype)
        ctx = self.text_projection(text_emb)[:,None,:]
        
        if decode:
            mask = nn.make_attention_mask(
                jnp.ones((B, 1), dtype=jnp.bool_),
                jnp.arange(self.num_positions) < (index + 1),
                dtype=jnp.bool_
            )
        else:
            mask = nn.make_causal_mask(jnp.ones((B, L), dtype=jnp.bool_))
        
        for block in self.blocks:
            x = block(x, context=ctx, mask=mask, decode=decode, index=index)
            
        return self.logit_head(self.norm(x))







     
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





# In TokenizerTrainer

class PIDLambdaController:
    """
    A PID controller to dynamically balance the generator's reconstruction loss weights.
    [UPGRADED] to handle a generic dictionary of losses for the full diagnostic suite.
    """
    def __init__(self, targets: Dict[str, float], base_weights: Dict[str, float], gains: Dict[str, Tuple[float, float, float]]):
        self.targets = targets
        self.base_weights = base_weights
        self.gains = gains
        # Initialize state for all potential keys to prevent errors during training
        all_keys = set(targets.keys()) | set(base_weights.keys())
        self.integral_error = {k: 0.0 for k in all_keys}
        self.last_error = {k: 0.0 for k in all_keys}
        self.derivative = {k: 0.0 for k in all_keys}

    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        final_lambdas = {}
        
        # Dynamically calculate lambdas for all targets
        for name, target in self.targets.items():
            kp, ki, kd = self.gains[name]
            current_loss = last_metrics.get(name, target) # Use target as default if metric not present yet
            error = current_loss - target
            
            self.integral_error[name] += error
            # Clamp the integral term to prevent wind-up
            self.integral_error[name] = np.clip(self.integral_error[name], -5.0, 5.0)
            self.derivative[name] = error - self.last_error[name]
            
            adjustment = (kp * error) + (ki * self.integral_error[name]) + (kd * self.derivative[name])
            multiplier = np.exp(adjustment)
            
            calculated_lambda = self.base_weights[name] * multiplier
            self.last_error[name] = error
            
            # Apply specific clipping rules for stability
            if name == 'l1':
                final_lambdas[name] = np.clip(calculated_lambda, 0.2, 5.0)
            elif name == 'vq':
                final_lambdas[name] = np.clip(calculated_lambda, 0.5, 10.0)
            elif name == 'ssim':
                 final_lambdas[name] = np.clip(calculated_lambda, 0.5, 30.0)
            else: # General clipping for other perceptual terms
                final_lambdas[name] = np.clip(calculated_lambda, 0.1, 20.0)

        # Handle fixed-weight losses like 'adv'
        if 'adv' in self.base_weights:
            final_lambdas['adv'] = self.base_weights['adv']

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
            base_weights={'l1': 1.0, 'vq': 1.5, 'adv': 0.5, 'moment': 0.5, 'fft': 0.5, 'autocorr': 2.0, 'edge': 2.5, 'color_cov': 1.0, 'ssim': 3.0},
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
# UPGRADE: ConductorTrainer with CFG Training Loop
# =================================================================================================
class ConductorTrainer(AdvancedTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        
        console = Console()

        self.ds_config = DEFAULT_DS_CONFIG()
        self.preview_resolution = 96 if args.vram_saver_mode else 128

        tok_config_path = Path(f"tokenizer_{args.basename}_{args.num_codes}c_gan_config.pkl")
        if not tok_config_path.exists(): sys.exit(f"[FATAL] VQ-GAN tokenizer config not found: {tok_config_path}. Run the new 'train-tokenizer' first.")
        
        tok_path_best = Path(str(tok_config_path).replace("_config.pkl", "_best.pkl"))
        tok_path_final = Path(str(tok_config_path).replace("_config.pkl", "_final.pkl"))

        if tok_path_best.exists():
            console.print(f"--- Loading BEST VQ-GAN tokenizer from [green]{tok_path_best}[/green] ---")
            with open(tok_path_best,'rb') as f: self.tok_params = pickle.load(f)['params']
        elif tok_path_final.exists():
            console.print(f"--- Loading FINAL VQ-GAN tokenizer from [yellow]{tok_path_final}[/yellow] ---")
            with open(tok_path_final,'rb') as f: self.tok_params = pickle.load(f)['gen_params']
        else: sys.exit(f"[FATAL] No VQ-GAN tokenizer model found. Train tokenizer first.")
        
        with open(tok_config_path, 'rb') as f: self.tok_config = pickle.load(f)
        
        try:
            p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
            p1_d_model = int(p1_path.stem.split('_')[-2].replace('d', ''))
        except (StopIteration, ValueError):
            sys.exit(f"[FATAL] Could not find or parse a unique Phase 1 model file matching: '{args.basename}_*d_512.pkl'")

        console.print(f"--- Loading Phase 1 AE from: [green]{p1_path}[/green] (d_model={p1_d_model}) ---")
        self.p1_model = TopologicalCoordinateGenerator(p1_d_model, self.tok_config['latent_grid_size'], 512, self.dtype)
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']

        self.tokenizer = LatentTokenizerVQGAN(**self.tok_config, dtype=self.dtype)
        self.token_map_size = (self.tok_config['latent_grid_size'] // 4) ** 2
        self.model = GenerativeConductor(
            num_codes=args.num_codes + 1,
            num_positions=self.token_map_size + 1, 
            d_model=args.d_model_cond, 
            num_heads=args.num_heads, 
            num_layers=args.num_layers, 
            clip_dim=512, 
            dtype=self.dtype
        )
        
        self.interactive_state = InteractivityState()
        self.loss_history = deque(maxlen=200); self.sentinel_dampen_history = deque(maxlen=200)
        self.spinner_chars = ["🧠", "⚡", "💾", "📈", "🧠", "⚡", "💽", "📉"]
        self.spinner_idx, self.param_count, self.steps_per_sec = 0, 0, 0.0
        self.ui_lock = threading.Lock()
        
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        self.validation_prompts = [
            "red cup",
            "orange cat",
            "blue ball",
            "green grass with tree",
            "purple car",
        ]
        self.current_preview_prompt_idx = 0; self.current_preview_image_np = None


    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            console = Console()
            
            # --- Header ---
            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
            spinner = self.spinner_chars[self.spinner_idx]
            precision_str = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"{spinner} [bold]Generative Conductor[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            header_panel = Panel(Align.center(header_text), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right")

            # --- Left Column (Fixed Height Components) ---
            stats_tbl = Table.grid(expand=True, padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
            loss_val=self.loss_history[-1] if self.loss_history else 0; loss_emoji, color = ("👌","green") if loss_val < 2.5 else (("👍","yellow") if loss_val < 4.0 else ("😟","red"))
            stats_tbl.add_row("X-Entropy Loss", f"[{color}]{loss_val:.4f}[/] {loss_emoji}"); stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🏃💨")
            mem, util = self._get_gpu_stats(); stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            stats_panel = Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue")
            
            q_panel = Panel(Align.center("[dim]Q-Ctrl Off[/dim]"), title="[bold]🤖 Q-Controller[/]", border_style="dim")
            if self.q_controller:
                q_table = Table.grid(expand=True, padding=(0,1)); q_table.add_column("Ctrl", style="bold cyan", width=6); q_table.add_column("Metric", style="dim", width=10); q_table.add_column("Value", justify="left")
                status_full = self.q_controller.status; status_short = status_full.split(' ')[0]
                status_emoji, color = ("😎","green") if "IMPROVING" in status_short else (("🤔","yellow") if "STAGNATED" in status_short else (("😠","red") if "REGRESSING" in status_short else (("🐣","blue") if "WARMUP" in status_short else ("🤖","dim"))))
                q_table.add_row("🧠 LR", "Status", f"[{color}]{status_short}[/] {status_emoji}"); q_table.add_row("", "Reward", f"{self.q_controller.last_reward:+.2f}")
                q_panel = Panel(q_table, title="[bold]🤖 Q-Controller[/]", border_style="green")

            # [GUI FIX] Integrate footer text cleanly into the left column
            footer_text = Text("←/→: Change Preview | ↑/↓: Adjust Sentinel | Ctrl+C to Exit", style="dim", justify="center")
            
            left_column_panels = [stats_panel, q_panel]
            if self.args.use_sentinel:
                sentinel_layout = Layout(); log_factor = self.interactive_state.sentinel_dampening_log_factor
                lever_panel = Panel(get_sentinel_lever_ascii(log_factor), title="Dampen 🚀", title_align="left")
                status_str_padded = f"{getattr(self, 'sentinel_pct', 0.0): >7.2%}"; status_str = f"Dampened: {status_str_padded}"
                status_panel = Panel(Align.center(Text(status_str)), title="Status 🚦", height=4); sentinel_layout.split_row(Layout(lever_panel), Layout(status_panel))
                sentinel_panel = Panel(sentinel_layout, title="[bold]🕹️ Sentinel Interactive[/]", border_style="yellow")
                left_column_panels.append(sentinel_panel)

            left_column = Layout(Group(*left_column_panels, Panel(footer_text)))

            # --- Right Column (Trends + Preview) ---
            right_column_layout = Layout(name="right_col")
            
            # Define a fixed height for the trend graphs
            trends_height = 5 if not self.args.use_sentinel else 8

            spark_w = max(10, console.width - 45) # Flexible width
            loss_spark = Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_history, spark_w)}[/]"), title="Loss Trend", height=3, border_style="cyan")
            graph_panels = [loss_spark]
            if self.args.use_sentinel:
                sentinel_spark = Panel(Align.center(f"[magenta]{self._get_sparkline(self.sentinel_dampen_history, spark_w)}[/]"), title="Sentinel Dampening %", height=3, border_style="magenta")
                graph_panels.append(sentinel_spark)
            trends_panel = Panel(Group(*graph_panels), title="[bold]📉 Trends[/]", height=trends_height)

            # --- Preview Panel (The Star of the Show) ---
            current_prompt = self.validation_prompts[self.current_preview_prompt_idx]
            prompt_text = Text(f"Prompt #{self.current_preview_prompt_idx+1}: \"{current_prompt}\"", justify="center", no_wrap=False, overflow="fold")
            
            preview_content = Align.center("...Waiting for first validation step...")
            if self.current_preview_image_np is not None:
                # [GUI FIX] Let Pixels render inside an Align.center directly. This is the most robust way.
                if Pixels:
                    preview_content = Align.center(Pixels.from_image(Image.fromarray(self.current_preview_image_np)))
                else:
                    preview_content = Align.center(Text("Install `rich-pixels`", style="yellow"))

            # The preview panel is now a simple group of the prompt and the content.
            preview_group = Group(prompt_text, preview_content)
            preview_panel = Panel(preview_group, title="[bold]🖼️ Live Generation Preview[/]", border_style="green")
            
            # [GUI FIX] Explicitly split the right column. Trends get a fixed size, Preview gets the rest.
            right_column_layout.split_column(
                Layout(trends_panel, size=trends_height),
                Layout(preview_panel, ratio=1, name="preview_area")
            )

            # --- Final Layout Assembly ---
            layout = Layout(name="root")
            # The footer is gone from the main split.
            layout.split(
                Layout(header_panel, name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(self.progress, name="progress", size=3)
            )
            layout["main"].split_row(Layout(left_column, name="left", size=42), right_column_layout)
            return layout









    def train(self):
        console = Console()
        key_listener_thread = threading.Thread(target=listen_for_keys, args=(self.interactive_state,), daemon=True); key_listener_thread.start()
        
        checkpoint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='CheckpointSaver')
        active_save_future = None

        def _save_cpu_data_task(cpu_data_to_save, path):
            with open(path, 'wb') as f: pickle.dump(cpu_data_to_save, f)

        with torch.no_grad():
            text_tokens = clip.tokenize(self.validation_prompts).to(_clip_device)
            self.validation_embeddings = self.clip_model.encode_text(text_tokens).cpu().numpy()

        tokenized_data_path = Path(self.args.data_dir) / f"tokenized_data_{self.args.basename}_{self.args.num_codes}c.npz"
        if tokenized_data_path.exists():
            console.print(f"--- ✅ Found pre-tokenized data. Loading from [green]{tokenized_data_path}[/green] ---", style="bold yellow")
            with np.load(tokenized_data_path) as data: train_tokens, train_embeddings, val_tokens, val_embeddings = data['train_tokens'], data['train_embeddings'], data['val_tokens'], data['val_embeddings']
        else:
            console.print("--- 🧠 STEP 1: Pre-tokenized data not found. Creating it now... ---", style="bold yellow")
            data_path = Path(self.args.data_dir) / f"paired_data_{self.args.basename}.pkl";
            if not data_path.exists(): sys.exit(f"[FATAL] Paired data not found at {data_path}")
            with open(data_path, 'rb') as f: data = pickle.load(f)
            jit_tokenizer_encode = jax.jit(lambda p, l: self.tokenizer.apply({'params': p}, l, method=self.tokenizer.encode))
            all_tokens_list = []; tokenization_batch_size = self.args.batch_size * self.num_devices * 8; latents_jnp = jnp.asarray(data['latents'])
            for i in tqdm(range(0, len(latents_jnp), tokenization_batch_size), desc="Tokenizing"):
                tokens_2d = jit_tokenizer_encode(self.tok_params, latents_jnp[i:i + tokenization_batch_size]); all_tokens_list.append(tokens_2d.reshape(tokens_2d.shape[0], -1))
            all_tokens_flat = np.array(jnp.concatenate(all_tokens_list, axis=0)); all_embeddings = data['embeddings']
            del data, all_tokens_list, latents_jnp
            np.random.seed(self.args.seed); shuffled_indices = np.random.permutation(len(all_tokens_flat))
            val_split_idx = int(len(all_tokens_flat) * 0.02); train_indices, val_indices = shuffled_indices[val_split_idx:], shuffled_indices[:val_split_idx]
            train_tokens, train_embeddings = all_tokens_flat[train_indices], all_embeddings[train_indices]; val_tokens, val_embeddings = all_tokens_flat[val_indices], all_embeddings[val_indices]
            np.savez_compressed(tokenized_data_path, train_tokens=train_tokens, train_embeddings=train_embeddings, val_tokens=val_tokens, val_embeddings=val_embeddings)
        
        train_ds = tf.data.Dataset.from_tensor_slices((train_tokens, train_embeddings)).shuffle(10000, seed=self.args.seed).repeat().batch(self.args.batch_size * self.num_devices, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        train_iterator = train_ds.as_numpy_iterator()

        key = jax.random.PRNGKey(self.args.seed)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), sentinel() if self.args.use_sentinel else optax.identity(), optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr))
        
        ckpt_path_final = Path(f"conductor_{self.args.basename}_{self.args.num_layers}l_final.pkl"); ckpt_path_best = Path(f"conductor_{self.args.basename}_{self.args.num_layers}l_best.pkl")
        best_val_loss, start_step = float('inf'), 0
        
        def get_initial_variables(init_key):
            dummy_tokens = jnp.zeros((1, 1), jnp.int32)
            dummy_embeddings = jnp.zeros((1, 512), self.dtype)
            return self.model.init({'params': init_key, 'dropout': init_key}, dummy_tokens, dummy_embeddings, decode=True, index=0)

        if ckpt_path_final.exists():
            with open(ckpt_path_final, 'rb') as f: ckpt = pickle.load(f)
            with jax.default_device(CPU_DEVICE): params_template = get_initial_variables(key)['params']
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params_template, tx=optimizer)
            state = state.replace(params=ckpt['params'], opt_state=ckpt['opt_state'], step=ckpt.get('step',0))
            start_step = state.step; console.print(f"✅ Resuming from step {start_step + 1}")
            if self.q_controller and ckpt.get('q_controller_state'): self.q_controller.load_state_dict(ckpt['q_controller_state']);
            if ckpt_path_best.exists():
                with open(ckpt_path_best, 'rb') as f_best: best_val_loss = pickle.load(f_best).get('val_loss', float('inf'))
        else:
            with jax.default_device(CPU_DEVICE): params = get_initial_variables(key)['params']
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

        self.param_count = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, state.params, 0)
        p_state = replicate(state)
        
        console.print("--- 🧠 STEP 3: JIT Compiling training and validation kernels (one-time cost)... ---")
        
        
# In ConductorTrainer.train()

        @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(5,))
        def train_step_fn(state, batch, dropout_key, lr, damp_factor, num_codes):
            tokens_flat, embeddings = batch
            input_tokens = jnp.concatenate([jnp.full((tokens_flat.shape[0], 1), num_codes), tokens_flat], axis=1)[:, :-1]
            
            def loss_fn(p):
                # [UPDATE] Pass the dropout RNG stream to the model's apply_fn
                logits = state.apply_fn({'params': p}, input_tokens, embeddings, train=True, rngs={'dropout': dropout_key})
                
                ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, tokens_flat).mean()

                logits_flat = logits.reshape(-1, logits.shape[-1])
                log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
                _, varent = ent_varent(log_probs)
                varentropy_loss = varent.mean()
                
                lambda_varentropy = 0.01 
                total_loss = ce_loss + (lambda_varentropy * varentropy_loss)
                
                return total_loss.astype(jnp.float32), total_loss.astype(jnp.float32)

            (loss, loss_for_reporting), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            grads = jax.lax.pmean(grads, 'batch')
            loss_reported = jax.lax.pmean(loss_for_reporting, 'batch')
            
            new_state = state.apply_gradients(grads=grads, dampening_factor=damp_factor, learning_rate=lr)
            
            sentinel_dampened_pct = 0.0
            if self.args.use_sentinel:
                if len(new_state.opt_state) > 1 and isinstance(new_state.opt_state[1], SentinelState):
                    sentinel_state = new_state.opt_state[1]
                    sentinel_dampened_pct = sentinel_state.dampened_pct
            
            return new_state, loss_reported, sentinel_dampened_pct
            
        @partial(jax.jit, static_argnames=('apply_fn', 'num_codes'))
        def run_validation_batch(params, val_tokens_batch, val_embeddings_batch, apply_fn, num_codes):
            input_tokens = jnp.concatenate([jnp.full((val_tokens_batch.shape[0], 1), num_codes), val_tokens_batch], axis=1)[:, :-1]
            logits = apply_fn({'params': params}, input_tokens, val_embeddings_batch, train=False)
            return optax.softmax_cross_entropy_with_integer_labels(logits, val_tokens_batch).mean()
        
        @partial(jax.jit, static_argnames=('tokenizer_apply_fn', 'p1_model_apply_fn', 'resolution', 'num_chunks', 'grid_dim'))
        def _render_from_tokens_jit(tokenizer_apply_fn, tok_params, p1_model_apply_fn, p1_params, full_tokens, resolution, num_chunks, grid_dim):
            token_grid = full_tokens.reshape(full_tokens.shape[0], grid_dim, grid_dim)
            path_params = tokenizer_apply_fn({'params': tok_params}, token_grid, method=LatentTokenizerVQGAN.decode)
            coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, resolution), jnp.linspace(-1, 1, resolution), indexing='ij'), -1).reshape(-1, 2)
            coord_chunks = jnp.array_split(coords, num_chunks, axis=0)
            def scan_body(carry, chunk):
                pixels = p1_model_apply_fn({'params': p1_params}, path_params, chunk, method=TopologicalCoordinateGenerator.decode)
                return carry, pixels
            _, pixel_chunks = jax.lax.scan(scan_body, None, jnp.stack(coord_chunks))
            pixels_batched = jnp.concatenate(pixel_chunks, axis=1)
            img = pixels_batched.reshape(full_tokens.shape[0], resolution, resolution, 3)
            return ((img * 0.5 + 0.5) * 255).astype(jnp.uint8)

        @partial(jax.jit, static_argnames=('model_apply_fn', 'num_steps', 'bos_token_id'))
        def _dslider_autoregressive_sample_jit(model_apply_fn, variables, initial_ds_state, text_emb, key, ds_config, num_steps, bos_token_id):
            def scan_body(carry, xs_slice):
                current_vars, last_token, current_ds_state = carry
                step_index, key_step = xs_slice
                logits, new_mutable_vars = model_apply_fn(current_vars, last_token, text_emb, train=False, decode=True, index=step_index, mutable=['cache'])
                output_vars = {'params': current_vars['params'], 'cache': new_mutable_vars['cache']}
                next_token, new_ds_state = dslider_sampler_step(key_step, current_ds_state, logits.squeeze(1), ds_config)
                return (output_vars, next_token[:, None], new_ds_state), next_token
            initial_token = jnp.full((text_emb.shape[0], 1), bos_token_id, dtype=jnp.int32)
            keys = jax.random.split(key, num_steps)
            initial_carry = (variables, initial_token, initial_ds_state)
            xs = (jnp.arange(num_steps), keys)
            _, generated_tokens_collection = jax.lax.scan(scan_body, initial_carry, xs)
            return generated_tokens_collection.transpose()

        @partial(jax.jit, static_argnames=('num_steps', 'resolution'))
        def _generate_validation_preview(conductor_params, initial_cache, text_emb, key, ds_config, num_steps, resolution):
            variables = {'params': conductor_params, 'cache': initial_cache}
            bos_token = jnp.full((text_emb.shape[0], 1), self.args.num_codes, dtype=jnp.int32)
            initial_logits, updated_mutable_state = self.model.apply(variables, bos_token, text_emb, train=False, decode=True, index=0, mutable=['cache'])
            variables_for_scan = {'params': conductor_params, 'cache': updated_mutable_state['cache']}
            initial_ds_state = initialize_state(initial_logits, bsz=text_emb.shape[0], config=ds_config, dtype=self.dtype)
            final_tokens = _dslider_autoregressive_sample_jit(self.model.apply, variables_for_scan, initial_ds_state, text_emb, key, ds_config, num_steps, self.args.num_codes)
            grid_dim = self.tok_config['latent_grid_size'] // 4
            return _render_from_tokens_jit(self.tokenizer.apply, self.tok_params, self.p1_model.apply, self.p1_params, final_tokens, resolution, 16, grid_dim)
        
        clean_state_for_compile = jax.device_get(unreplicate(p_state))
        if len(val_tokens) > 0:
            val_batch_size = self.args.batch_size * self.num_devices; MAX_VAL_SAMPLES = 1024
            num_val_to_keep = min(len(val_tokens), MAX_VAL_SAMPLES); num_val_to_keep = (num_val_to_keep // val_batch_size) * val_batch_size
            val_cpu_batches = list(zip(np.split(val_tokens[:num_val_to_keep], num_val_to_keep // val_batch_size), np.split(val_embeddings[:num_val_to_keep], num_val_to_keep // val_batch_size))) if num_val_to_keep > 0 else None
            if val_cpu_batches:
                dummy_val_batch = jax.device_put(val_cpu_batches[0][0]), jax.device_put(val_cpu_batches[0][1])
                run_validation_batch(clean_state_for_compile.params, *dummy_val_batch, self.model.apply, self.args.num_codes)
                with jax.default_device(CPU_DEVICE): initial_cache = get_initial_variables(key)['cache']
                _generate_validation_preview(clean_state_for_compile.params, initial_cache, jnp.zeros((1, 512), dtype=self.dtype), key, self.ds_config, self.token_map_size, self.preview_resolution)
        
        dummy_batch = next(train_iterator); sharded_batch = shard(dummy_batch); dummy_keys = jax.random.split(key, self.num_devices)
        p_state, _, _ = train_step_fn(p_state, sharded_batch, dummy_keys, replicate(jnp.array(self.args.lr)), replicate(jnp.array(1.0)), self.args.num_codes)
        console.print("[green]✅ Compilation complete. Starting training...[/green]")
        
        self.progress = Progress(TextColumn("[bold]Step {task.completed}/{task.total}"), BarColumn(), "[p.p.]{task.percentage:>3.1f}%", "•", TextColumn("Loss: {task.fields[loss]:.4f}"), "•", TextColumn("Val: {task.fields[val_loss]:.4f}"), "•", TextColumn("[bold green]Best: {task.fields[best_val_loss]:.4f}[/]"), TimeRemainingColumn())
        task_id = self.progress.add_task("train", total=self.args.steps, completed=start_step, loss=0, val_loss=best_val_loss, best_val_loss=best_val_loss)
        SAVE_EVERY_N_STEPS = 2000; last_step_time = time.time(); current_step = start_step; 
        with jax.default_device(CPU_DEVICE): initial_inference_cache = get_initial_variables(key)['cache']
        
        last_val_loss = best_val_loss
        
        try:
            with Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for step in range(start_step, self.args.steps):
                    current_step = step
                    if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                    
                    batch = next(train_iterator); sharded_batch = shard(batch)
                    key, train_step_key, preview_key = jax.random.split(key, 3); sharded_keys = jax.random.split(train_step_key, self.num_devices)
                    
                    lr = self.q_controller.choose_action() if self.q_controller else self.args.lr
                    damp_factor = self.interactive_state.get_sentinel_factor() if self.args.use_sentinel else 1.0
                    
                    p_lr = replicate(jnp.array(lr)); p_damp = replicate(jnp.array(damp_factor))
                    
                    p_state, p_loss, p_sentinel_pct = train_step_fn(p_state, sharded_batch, sharded_keys, p_lr, p_damp, self.args.num_codes)
                    
                    loss_val = unreplicate(p_loss).item(); sentinel_val = unreplicate(p_sentinel_pct).item()
                    with self.ui_lock: self.loss_history.append(loss_val)
                    if self.args.use_sentinel: self.sentinel_dampen_history.append(sentinel_val); self.sentinel_pct = sentinel_val
                    if self.q_controller: self.q_controller.update_q_value(loss_val)

                    preview_prompt_change = self.interactive_state.get_and_reset_preview_change()
                    if preview_prompt_change != 0:
                        with self.ui_lock: self.current_preview_prompt_idx = (self.current_preview_prompt_idx + preview_prompt_change) % len(self.validation_prompts)

                    if (step + 1) % self.args.eval_every == 0 and val_cpu_batches:
                        unrep_state_for_eval = jax.device_get(unreplicate(p_state))
                        
                        val_losses = [run_validation_batch(unrep_state_for_eval.params, jax.device_put(tokens), jax.device_put(embs), self.model.apply, self.args.num_codes) for tokens, embs in val_cpu_batches]
                        last_val_loss = np.mean([v.item() for v in val_losses])
                        
                        if last_val_loss < best_val_loss:
                            best_val_loss = last_val_loss
                            if not (active_save_future and not active_save_future.done()):
                                console.print(f"\n[bold magenta]🏆 New best val loss: {best_val_loss:.4f} @ step {step+1}. Saving in background...[/bold magenta]")
                                data_for_save = {'params': unrep_state_for_eval.params, 'val_loss': best_val_loss, 'step': step+1}
                                active_save_future = checkpoint_executor.submit(_save_cpu_data_task, data_for_save, ckpt_path_best)
                        
                        text_emb = jnp.expand_dims(self.validation_embeddings[self.current_preview_prompt_idx], 0)
                        preview_img_array = _generate_validation_preview(unrep_state_for_eval.params, initial_inference_cache, text_emb, preview_key, self.ds_config, self.token_map_size, self.preview_resolution)
                        preview_img_array.block_until_ready()
                        with self.ui_lock:
                            self.current_preview_image_np = np.asarray(preview_img_array[0])

                    self.progress.update(task_id, advance=1, loss=loss_val, val_loss=last_val_loss, best_val_loss=best_val_loss)
                    
                    if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                        if not (active_save_future and not active_save_future.done()):
                            host_state_unreplicated = jax.device_get(unreplicate(p_state))
                            data_for_save = {'params': host_state_unreplicated.params, 'opt_state': host_state_unreplicated.opt_state, 'step': step + 1, 'q_controller_state': self.q_controller.state_dict() if self.q_controller else None}
                            active_save_future = checkpoint_executor.submit(_save_cpu_data_task, data_for_save, ckpt_path_final)

                    current_time = time.time(); self.steps_per_sec = 1.0 / (current_time - last_step_time + 1e-9); last_step_time = current_time
                    
                    # [GUI FIX] Update every step, but throttle the actual console render call
                    live.update(self._generate_layout(), refresh=True)
        finally:
            console.print(f"\n[yellow]--- Training loop exited at step {current_step + 1}. Waiting for final save... ---[/yellow]")
            checkpoint_executor.shutdown(wait=True)
            if 'p_state' in locals():
                host_state = jax.device_get(unreplicate(p_state))
                final_data_to_save = {'params': host_state.params, 'opt_state': host_state.opt_state, 'step': current_step + 1, 'q_controller_state': self.q_controller.state_dict() if self.q_controller else None}
                with open(ckpt_path_final, 'wb') as f: pickle.dump(final_data_to_save, f)
                console.print(f"✅ Final resume-state saved to [green]{ckpt_path_final}[/green]")
            
            config = { 'num_codes': self.args.num_codes, 'num_positions': self.token_map_size + 1, 'd_model': self.args.d_model_cond, 'num_heads': self.args.num_heads, 'num_layers': self.args.num_layers, 'clip_dim': 512 }
            config_path = Path(f"conductor_{self.args.basename}_{self.args.num_layers}l_config.pkl")
            with open(config_path, 'wb') as f: pickle.dump(config, f)
            console.print(f"✅ Config saved to [green]{config_path}[/green]")
            if ckpt_path_best.exists(): console.print(f"👑 Best model (by validation) remains at [bold magenta]{ckpt_path_best}[/bold magenta]")
            self.interactive_state.set_shutdown(); key_listener_thread.join()









# =================================================================================================
# 6. GENERATION & INFERENCE (PARAMETER AGNOSTIC)
# =================================================================================================

class Generator:
    """
    An advanced, parameter-agnostic inference engine for the Phase 3 stack.
    
    This class loads all necessary trained components (Phase 1 AE, Tokenizer, Conductor)
    based on a single basename and provides high-level methods for text-to-image
    generation and editing.
    
    The key upgrade is the implementation of Classifier-Free Guidance (CFG) in the
    autoregressive sampling loop, which dramatically improves text-image alignment.
    """
    def __init__(self, args):
        self.args = args
        self.console = Console()
        self.console.print("--- 🧠 Loading Full Generative Stack (CFG-Enabled) ---", style="bold yellow")
        
        # --- Discover and Load Model Configurations ---
        try:
            conductor_config_path = next(Path('.').glob(f"conductor_{args.basename}_*_config.pkl"))
            tokenizer_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_*_config.pkl"))
            # [FIX] Use a more robust glob pattern to find the Phase 1 model regardless of d_model.
            p1_path = next(Path('.').glob(f"{args.basename}_*d_512.pkl"))
        except StopIteration:
            sys.exit(f"[FATAL] Could not find required config or model files for basename '{args.basename}'. Please train the models first.")

        with open(conductor_config_path, 'rb') as f: self.cond_config = pickle.load(f)
        with open(tokenizer_config_path, 'rb') as f: self.tok_config = pickle.load(f)

        # --- Initialize Models with Correct Architectures ---
        self.dtype = jnp.float32  # Use FP32 for inference for max quality
        p1_d_model = int(p1_path.stem.split('_')[-2].replace('d',''))

        self.p1_model = TopologicalCoordinateGenerator(p1_d_model, self.tok_config['latent_grid_size'], 512, self.dtype)
        # [FIX] Instantiate the exact same model class used for training (LatentTokenizerVQGAN).
        self.tokenizer = LatentTokenizerVQGAN(**self.tok_config, dtype=self.dtype)
        self.conductor = GenerativeConductor(**self.cond_config, dtype=self.dtype)
        
        # --- Load Trained Parameters ---
        cond_ckpt_path = Path(str(conductor_config_path).replace("_config.pkl", "_best.pkl"))
        if not cond_ckpt_path.exists():
            cond_ckpt_path = Path(str(cond_ckpt_path).replace("_best.pkl", "_final.pkl"))
        
        tok_ckpt_path = Path(str(tokenizer_config_path).replace("_config.pkl", "_best.pkl"))
        if not tok_ckpt_path.exists():
            tok_ckpt_path = Path(str(tok_ckpt_path).replace("_best.pkl", "_final.pkl"))

        self.console.print(f"-> Loading Phase 1 AE from: [green]{p1_path}[/green]")
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']
        
        self.console.print(f"-> Loading Tokenizer from: [green]{tok_ckpt_path}[/green]")
        with open(tok_ckpt_path, 'rb') as f:
            # [FIX] Robustly load tokenizer params from either 'best' or 'final' checkpoint format.
            tok_ckpt = pickle.load(f)
            self.tok_params = tok_ckpt.get('params', tok_ckpt.get('gen_params'))

        self.console.print(f"-> Loading Conductor from: [green]{cond_ckpt_path}[/green]")
        with open(cond_ckpt_path, 'rb') as f: self.cond_params = pickle.load(f)['params']
        
        # --- CRITICAL FOR CFG: Extract the learned unconditional embedding ---
        self.uncond_embedding = self.cond_params['uncond_embedding']

        self.console.print(f"✅ Models and configs loaded for basename [cyan]'{args.basename}'[/cyan]")

        # --- Load CLIP and JIT Compile Core Functions ---
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        self.console.print("✅ [CLIP] Text Encoder loaded.")
        
        self.ds_config = DEFAULT_DS_CONFIG()
        self.token_map_size = (self.tok_config['latent_grid_size'] // 4) ** 2

        self.console.print("--- 🚀 JIT Compiling inference kernels (this may take a moment)... ---")
        self._jit_compile_functions()
        self.console.print("✅ All kernels compiled.")

    def _jit_compile_functions(self):
        """JIT compiles all necessary functions for fast inference."""
        
        # --- The core CFG sampling loop ---
        @partial(jax.jit, static_argnames=('model_apply_fn', 'num_steps', 'bos_token_id', 'guidance_scale'))
        def _cfg_dslider_autoregressive_sample_jit(
            model_apply_fn, variables, initial_ds_state,
            cond_emb, uncond_emb, key, ds_config,
            guidance_scale, num_steps, bos_token_id
        ):
            # 1. Stack conditional and unconditional embeddings for a single parallel forward pass
            text_emb = jnp.concatenate([cond_emb, uncond_emb], axis=0) # Batch size is now 2
            
            # 2. Duplicate the initial state (cache, DSState) for the two parallel runs
            doubled_vars = jax.tree_util.tree_map(lambda x: jnp.concatenate([x,x], axis=0), variables)
            doubled_ds_state = jax.tree_util.tree_map(lambda x: jnp.concatenate([x,x], axis=0), initial_ds_state)
            
            def scan_body(carry, xs_slice):
                current_vars, last_token, current_ds_state = carry
                step_index, key_step = xs_slice

                # 3. Run model ONCE on the stacked batch of size 2
                logits, new_mutable_vars = model_apply_fn(
                    current_vars, last_token, text_emb, train=False, decode=True, index=step_index, mutable=['cache']
                )
                output_vars = {'params': current_vars['params'], 'cache': new_mutable_vars['cache']}

                # 4. Split the output logits back into conditional and unconditional
                logits_cond, logits_uncond = logits[0:1], logits[1:2]
                
                # 5. Apply the CFG formula - THIS IS THE MAGIC
                cfg_logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)

                # 6. Sample using ONLY the CFG-enhanced logits. We only need the DS state for the conditional branch.
                ds_state_cond = jax.tree_util.tree_map(lambda x: x[0:1], current_ds_state)
                next_token, new_ds_state_cond = dslider_sampler_step(key_step, ds_state_cond, cfg_logits.squeeze(1), ds_config)
                
                # 7. For the next loop iteration, both branches need a token and a state. We feed the same token
                #    to both and update both states with the new conditional state.
                new_ds_state_doubled = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, x], axis=0), new_ds_state_cond)
                next_token_doubled = jnp.concatenate([next_token, next_token], axis=0)
                
                return (output_vars, next_token_doubled[:, None], new_ds_state_doubled), next_token

            # Prepare initial inputs for the scan loop
            initial_token = jnp.full((2, 1), bos_token_id, dtype=jnp.int32)
            keys = jax.random.split(key, num_steps)
            initial_carry = (doubled_vars, initial_token, doubled_ds_state)
            xs = (jnp.arange(num_steps), keys)
            
            # Run the scan
            _, generated_tokens_collection = jax.lax.scan(scan_body, initial_carry, xs)
            
            # Transpose from (L, B) to (B, L) and return only the conditional branch's result
            generated_tokens = generated_tokens_collection.transpose()
            return generated_tokens[0:1, :]

        # --- Helper for rendering ---
        @partial(jax.jit, static_argnames=('resolution', 'patch_size'))
        def _render_fn(p1_params, path_params, resolution=512, patch_size=256):
            coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
            coord_chunks = jnp.array_split(coords, (resolution**2)//(patch_size**2))
            pixels_list = [self.p1_model.apply({'params': p1_params}, path_params, c, method=self.p1_model.decode) for c in coord_chunks]
            return jnp.concatenate(pixels_list, axis=1).reshape(path_params.shape[0], resolution, resolution, 3)
            
        # --- Store jitted functions ---
        self._cfg_sampler = _cfg_dslider_autoregressive_sample_jit
        self._render_fn = _render_fn
        # [FIX] This now correctly calls the method on the LatentTokenizerVQGAN instance
        self._decode_tokens_fn = jax.jit(lambda i: self.tokenizer.apply({'params':self.tok_params}, i, method=self.tokenizer.decode))
        self._get_initial_vars_fn = jax.jit(lambda k: self.conductor.init({'params':k, 'dropout':k}, jnp.zeros((1,1),jnp.int32), jnp.zeros((1,512), self.dtype), decode=True, index=0))

    def generate(self, prompt: str, seed: int, guidance_scale: float):
        self.console.print(f"--- 🎨 Generating image for prompt: \"[italic yellow]{prompt}[/italic yellow]\" ---")
        self.console.print(f"-> Using Seed: {seed}, CFG Scale: {guidance_scale}")
        key = jax.random.PRNGKey(seed)
        
        # 1. Prepare inputs
        key, init_key, sample_key = jax.random.split(key, 3)
        with torch.no_grad():
            cond_text_emb = self.clip_model.encode_text(clip.tokenize([prompt]).to(_clip_device)).cpu().numpy().astype(self.dtype)
        
        # Get initial model state (params + empty cache)
        initial_vars = self._get_initial_vars_fn(init_key)
        initial_vars['params'] = self.cond_params # Overwrite with loaded params
        
        # 2. Run one step to get initial logits for the DSState
        bos_token = jnp.full((1, 1), self.cond_config['num_codes'], dtype=jnp.int32)
        initial_logits, updated_vars = self.conductor.apply(initial_vars, bos_token, cond_text_emb, decode=True, index=0, mutable=['cache'])
        initial_ds_state = initialize_state(initial_logits, bsz=1, config=self.ds_config, dtype=self.dtype)

        self.console.print("1/3: Composing scene with Conductor (CFG Sampling)...")
        # 3. Run the main CFG sampling loop
        final_tokens = self._cfg_sampler(
            self.conductor.apply, updated_vars, initial_ds_state,
            cond_text_emb, self.uncond_embedding,
            sample_key, self.ds_config, guidance_scale,
            self.token_map_size, self.cond_config['num_codes']
        )
        
        self.console.print("2/3: Decoding tokens with Tokenizer...")
        grid_dim = self.tok_config['latent_grid_size'] // 4
        token_grid = final_tokens.reshape(1, grid_dim, grid_dim)
        path_params = self._decode_tokens_fn(token_grid)

        self.console.print("3/3: Rendering final 512x512 image...")
        recon_batch = self._render_fn(self.p1_params, path_params)
        recon_np = np.array(((recon_batch[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        
        filename = f"GEN_{Path(self.args.basename).stem}_{prompt.replace(' ', '_')[:40]}_{seed}_cfg{guidance_scale:.1f}.png"
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
    parser = argparse.ArgumentParser(description="Phase 3 Generative Framework (Advanced Trainer)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--basename', type=str, required=True, help="Base name for the model set (e.g., 'laion_ae_96d'). Used to find/save all related files.")

    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 precision for training."); train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--use-q-controller', action='store_true', help="Enable adaptive LR via Q-Learning.")
    train_parser.add_argument('--use-sentinel', action='store_true', help="Enable Sentinel optimizer to dampen oscillations.")
    train_parser.add_argument('--finetune', action='store_true', help="Use finetuning Q-Controller config.")
    
    p_prep = subparsers.add_parser("prepare-paired-data", help="Pre-process (image, text) -> (latent, embedding).", parents=[base_parser])
    p_prep.add_argument('--data-dir', type=str, required=True); p_prep.add_argument('--d-model', type=int, required=True); p_prep.add_argument('--latent-grid-size', type=int, required=True); p_prep.add_argument('--batch-size', type=int, default=128)

    p_prep_tok = subparsers.add_parser("prepare-tokenizer-data", help="Pre-process images into latents for the tokenizer.", parents=[base_parser])
    p_prep_tok.add_argument('--data-dir', type=str, required=True); p_prep_tok.add_argument('--d-model', type=int, required=True); p_prep_tok.add_argument('--latent-grid-size', type=int, required=True); p_prep_tok.add_argument('--batch-size', type=int, default=128)

    p_tok = subparsers.add_parser("train-tokenizer", help="Train the Latent Tokenizer (VQ-VAE).", parents=[base_parser, train_parser])
    p_tok.add_argument('--data-dir', type=str, required=True); p_tok.add_argument('--d-model', type=int, required=True); p_tok.add_argument('--latent-grid-size', type=int, required=True)
    p_tok.add_argument('--epochs', type=int, default=100, help="Number of training epochs.") ; p_tok.add_argument('--batch-size', type=int, default=128); p_tok.add_argument('--lr', type=float, default=3e-4)
    # [NEW ARGUMENT]
    p_tok.add_argument('--eval-every', type=int, default=1000, help="Run validation and update preview every N steps.")
    p_tok.add_argument('--num-codes', type=int, default=8192); p_tok.add_argument('--code-dim', type=int, default=256)

    p_cond = subparsers.add_parser("train-conductor", help="Train the Generative Conductor (Transformer).", parents=[base_parser, train_parser])
    p_cond.add_argument('--data-dir', type=str, required=True); p_cond.add_argument('--latent-grid-size', type=int, required=True)
    p_cond.add_argument('--steps', type=int, default=1500000); p_cond.add_argument('--batch-size', type=int, default=32, help="Batch size PER DEVICE."); p_cond.add_argument('--lr', type=float, default=1e-4)
    p_cond.add_argument('--eval-every', type=int, default=500, help="Run validation every N steps.")
    p_cond.add_argument('--num-codes', type=int, default=8192); p_cond.add_argument('--code-dim', type=int, default=256)
    p_cond.add_argument('--num-layers', type=int, default=12); p_cond.add_argument('--d-model-cond', type=int, default=768); p_cond.add_argument('--num-heads', type=int, default=12)
    p_cond.add_argument('--vram-saver-mode', action='store_true', help="Enable aggressive VRAM saving optimizations (e.g., lower-res previews). Recommended for <= 8GB GPUs.")
    
    inference_parser = argparse.ArgumentParser(add_help=False); inference_parser.add_argument('--temp', type=float, default=0.9); inference_parser.add_argument('--top-k', type=int, default=256)
    p_gen = subparsers.add_parser("generate", help="Generate an image from a text prompt.", parents=[base_parser, inference_parser])
    p_gen.add_argument('--prompt', type=str, required=True)
    p_gen.add_argument('--seed', type=int, default=lambda: int(time.time()))
    p_gen.add_argument('--guidance-scale', type=float, default=7.5, help="Classifier-Free Guidance scale. Higher values mean stronger prompt adherence.")
    
    p_check_tok = subparsers.add_parser("check-tokenizer", help="Reconstruct an image through the tokenizer to check its quality.", parents=[base_parser])
    p_check_tok.add_argument('--image-path', type=str, required=True)
    p_check_tok.add_argument('--output-path', type=str, default="tokenizer_recon_check.png")
    
    args = parser.parse_args()
    
    if args.command in ["generate"]: args.seed = args.seed() if callable(args.seed) else args.seed
    if args.command == "train-conductor":
        if args.d_model_cond % args.num_heads != 0: sys.exit(f"[FATAL] Conductor d_model_cond ({args.d_model_cond}) must be divisible by num_heads ({args.num_heads}).")
        if args.latent_grid_size % 4 != 0: sys.exit(f"[FATAL] latent_grid_size ({args.latent_grid_size}) must be divisible by 4 due to tokenizer architecture.")

    # --- COMMAND DISPATCH ---
    if args.command == "prepare-paired-data": prepare_paired_data(args)
    elif args.command == "prepare-tokenizer-data": prepare_tokenizer_data(args)
    elif args.command == "train-tokenizer": TokenizerTrainer(args).train()
    elif args.command == "train-conductor": ConductorTrainer(args).train()
    elif args.command == "check-tokenizer": run_tokenizer_check(args)
    elif args.command == "generate": Generator(args).generate(args.prompt, args.seed, args.guidance_scale)
    
if __name__ == "__main__":
    main()
