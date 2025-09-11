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
import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
from flax import linen as nn
from flax.training import train_state
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









   
import math
from flax.linen import initializers
from flax.linen import dot_product_attention

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


import jax.lax # Import the lax module

class GenerativeConductor(nn.Module):
    num_codes: int; num_positions: int; d_model: int; num_heads: int; num_layers: int; clip_dim: int
    uncond_drop_rate: float = 0.1
    dtype: Any = jnp.float32

    def setup(self):
        """
        Defines the entire static structure of the model.
        """
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
        
        if train:
            key = self.make_rng('dropout')
            
            # --- [THE DEFINITIVE FIX] ---
            # The JAX tracer is incorrectly resolving `self.uncond_drop_rate` to a
            # type object (<class 'jnp.bfloat16'>) instead of its numerical value
            # when the module's dtype is bfloat16. To fix this, we bypass the
            # problematic instance attribute lookup (`self.`) and access the static
            # value directly from the class definition (`GenerativeConductor.`).
            # This retrieves the original Python float 0.1, which JAX can correctly handle.
            p = GenerativeConductor.uncond_drop_rate
            should_drop = jax.random.bernoulli(key, p, (B, 1))
            
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
# =================================================================================================
# FINAL, PRODUCTION-READY TokenizerTrainer with Clean Metrics
# This version ensures that both the train and eval steps compute and return
# the total loss, reconstruction loss, and VQ loss. The UI is updated
# to display this full, informative breakdown.
# =================================================================================================
class TokenizerTrainer(AdvancedTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.model = LatentTokenizerVQ(args.num_codes, args.code_dim, args.latent_grid_size, self.dtype)

    def train(self):
        console = Console()
        data_path = Path(self.args.data_dir) / f"tokenizer_latents_{self.args.basename}.pkl"
        if not data_path.exists(): sys.exit(f"[FATAL] Tokenizer latent data not found: {data_path}. Run 'prepare-tokenizer-data' first.")
        
        console.print(f"--- Loading all pre-computed latents into RAM from [green]{data_path}[/green] ---")
        with open(data_path, 'rb') as f: latents_data = pickle.load(f)['latents']
        
        np.random.seed(self.args.seed); shuffled_indices = np.random.permutation(len(latents_data))
        val_split_idx = int(len(latents_data) * 0.01)
        train_indices = shuffled_indices[val_split_idx:]; val_indices = shuffled_indices[:val_split_idx]
        train_data = latents_data[train_indices]; val_data = latents_data[val_indices]
        console.print(f"✅ Data split: {len(train_data)} training samples, {len(val_data)} validation samples.")
        
        key = jax.random.PRNGKey(self.args.seed)
        
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr))
        
        ckpt_path_final = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_final.pkl")
        ckpt_path_best = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_best.pkl")
        best_val_loss = float('inf'); start_step = 0

        if ckpt_path_final.exists():
            console.print(f"--- Resuming training from checkpoint: [green]{ckpt_path_final}[/green] ---")
            with open(ckpt_path_final, 'rb') as f: checkpoint_data = pickle.load(f)
            with jax.default_device(CPU_DEVICE):
                params_template = self.model.init(key, jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), self.dtype))['params']
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params_template, tx=optimizer)
            state = state.replace(params=checkpoint_data['params'], opt_state=checkpoint_data['opt_state'])
            start_step = checkpoint_data.get('step', 0)
            console.print(f"✅ Resuming from step {start_step + 1}")
            if self.q_controller and 'q_controller_state' in checkpoint_data:
                self.q_controller.load_state_dict(checkpoint_data['q_controller_state']); console.print("✅ Q-Controller state loaded.")
            if ckpt_path_best.exists():
                with open(ckpt_path_best, 'rb') as f_best: best_val_loss = pickle.load(f_best).get('val_loss', float('inf')); console.print(f"✅ Best validation loss loaded: {best_val_loss:.4f}")
        else:
            console.print("--- Initializing new model from scratch ---")
            with jax.default_device(CPU_DEVICE):
                params = self.model.init(key, jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3), self.dtype))['params']
            state = CustomTrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)

        # --- [CLEANUP] JIT functions now return all 3 metrics ---
        @partial(jax.jit, static_argnames=('apply_fn',))
        def train_step_fn(state, apply_fn, batch):
            def loss_fn(p):
                out = apply_fn({'params': p}, batch)
                recon_loss = jnp.mean(jnp.abs(out['reconstructed_path_params'] - batch))
                vq_loss = out['vq_loss']
                total_loss = recon_loss + vq_loss
                return total_loss.astype(jnp.float32), (recon_loss, vq_loss)
            
            (total_loss, (recon_loss, vq_loss)), g = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            new_state = state.apply_gradients(grads=g)
            return new_state, total_loss, recon_loss, vq_loss

        @partial(jax.jit, static_argnames=('apply_fn',))
        def eval_step_fn(params, apply_fn, batch):
            out = apply_fn({'params': params}, batch)
            recon_loss = jnp.mean(jnp.abs(out['reconstructed_path_params'] - batch))
            vq_loss = out['vq_loss']
            total_loss = recon_loss + vq_loss
            return total_loss.astype(jnp.float32), recon_loss.astype(jnp.float32), vq_loss.astype(jnp.float32)

        console.print("[bold yellow]🚀 JIT compiling (first run)...[/bold yellow]")
        dummy_batch = jnp.asarray(train_data[:self.args.batch_size], dtype=self.dtype)
        state, _, _, _ = train_step_fn(state, self.model.apply, dummy_batch)
        eval_step_fn(state.params, self.model.apply, dummy_batch); console.print("[green]✅ Compilation complete.[/green]")
        
        # --- [CLEANUP] UI now displays the full breakdown ---
        progress = Progress(
            TextColumn("[bold blue]Training..."), BarColumn(), "[p.p.]{task.percentage:>3.0f}%", "•",
            TextColumn("Loss(R/VQ): {task.fields[loss]:.3f} ({task.fields[recon]:.3f}/{task.fields[vq]:.3f})"), "•",
            TextColumn("Val(R/VQ): {task.fields[val_loss]:.3f} ({task.fields[val_recon]:.3f}/{task.fields[val_vq]:.3f})"), "•",
            TextColumn("[bold green]Best: {task.fields[best_val_loss]:.4f}[/]"),
            TimeElapsedColumn()
        )
        task_id = progress.add_task(
            "train", total=self.args.steps, completed=start_step,
            loss=0, recon=0, vq=0,
            val_loss=0, val_recon=0, val_vq=0,
            best_val_loss=best_val_loss
        )
        
        rng = np.random.default_rng(self.args.seed); train_indices_shuffler = np.arange(len(train_data)); current_idx = 0
        EVAL_EVERY_N_STEPS = 500; SAVE_EVERY_N_STEPS = 2000
        
        current_step = start_step
        try:
            with Live(progress, console=console, screen=False, vertical_overflow="visible") as live:
                for step in range(start_step, self.args.steps):
                    current_step = step
                    if self.should_shutdown: break

                    if current_idx + self.args.batch_size > len(train_data):
                        rng.shuffle(train_indices_shuffler); current_idx = 0
                    
                    batch_indices = train_indices_shuffler[current_idx : current_idx + self.args.batch_size]
                    train_batch = jnp.asarray(train_data[batch_indices], dtype=self.dtype); current_idx += self.args.batch_size

                    if self.q_controller:
                        lr = self.q_controller.choose_action(); state.opt_state[-1].hyperparams['learning_rate'] = jnp.asarray(lr)
                    
                    state, l, rl, vql = train_step_fn(state, self.model.apply, train_batch)
                    
                    loss_val, recon_val, vq_val = l.item(), rl.item(), vql.item()
                    if self.q_controller: self.q_controller.update_q_value(loss_val)

                    # Get previous validation values to keep them stable between eval runs
                    p_fields = progress.tasks[task_id].fields
                    val_l, val_rl, val_vql = p_fields['val_loss'], p_fields['val_recon'], p_fields['val_vq']

                    if (step + 1) % EVAL_EVERY_N_STEPS == 0:
                        val_metrics = [eval_step_fn(state.params, self.model.apply, jnp.asarray(val_data[i:i+self.args.batch_size], dtype=self.dtype)) for i in range(0, len(val_data), self.args.batch_size) if i < len(val_data)]
                        if val_metrics:
                            val_l_cpu, val_rl_cpu, val_vql_cpu = zip(*[(m[0].item(), m[1].item(), m[2].item()) for m in val_metrics])
                            val_l, val_rl, val_vql = np.mean(val_l_cpu), np.mean(val_rl_cpu), np.mean(val_vql_cpu)
                        
                        if val_l < best_val_loss:
                            best_val_loss = val_l
                            console.print(f"[bold magenta]🏆 New best val loss: {best_val_loss:.4f} @ step {step+1}. Saving...[/bold magenta]")
                            with open(ckpt_path_best, 'wb') as f: pickle.dump({'params': state.params, 'val_loss': best_val_loss, 'step': step+1}, f)

                    progress.update(task_id, advance=1, loss=loss_val, recon=recon_val, vq=vq_val,
                                    val_loss=val_l, val_recon=val_rl, val_vq=val_vql,
                                    best_val_loss=best_val_loss)
                    
                    if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                        data_to_save = {'params': state.params, 'opt_state': state.opt_state, 'step': step + 1, 'q_controller_state': self.q_controller.state_dict() if self.q_controller else None}
                        with open(ckpt_path_final, 'wb') as f: pickle.dump(data_to_save, f)
        finally:
            console.print(f"\n[yellow]--- Training loop exited at step {current_step + 1}. Saving final state... ---[/yellow]")
            final_data_to_save = {'params': state.params, 'opt_state': state.opt_state, 'step': current_step + 1, 'q_controller_state': self.q_controller.state_dict() if self.q_controller else None}
            with open(ckpt_path_final, 'wb') as f: pickle.dump(final_data_to_save, f)
            console.print(f"✅ Final resume-state saved to [green]{ckpt_path_final}[/green]")
            config = {'num_codes': self.args.num_codes, 'code_dim': self.args.code_dim, 'latent_grid_size': self.args.latent_grid_size}
            config_path = Path(f"tokenizer_{self.args.basename}_{self.args.num_codes}c_config.pkl")
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

        # --- DSlider and VRAM Saver Config ---
        self.ds_config = DEFAULT_DS_CONFIG()
        if args.vram_saver_mode:
            console.print("--- [bold yellow]VRAM Saver Mode ENABLED[/bold yellow]. Using lower-resolution previews. ---")
            self.preview_resolution = 96
        else:
            self.preview_resolution = 128

        tok_path_best = Path(f"tokenizer_{args.basename}_{args.num_codes}c_best.pkl")
        tok_path_final = Path(f"tokenizer_{args.basename}_{args.num_codes}c_final.pkl")
        tok_config_path = Path(f"tokenizer_{args.basename}_{args.num_codes}c_config.pkl")

        if tok_path_best.exists():
            console.print(f"--- Loading BEST tokenizer from [green]{tok_path_best}[/green] ---")
            with open(tok_path_best,'rb') as f: self.tok_params = pickle.load(f)['params']
        elif tok_path_final.exists():
            console.print(f"--- Loading FINAL tokenizer from [yellow]{tok_path_final}[/yellow] ---")
            with open(tok_path_final,'rb') as f: self.tok_params = pickle.load(f)['params']
        else: sys.exit(f"[FATAL] No tokenizer model found. Train tokenizer first.")
        
        if not tok_config_path.exists(): sys.exit(f"[FATAL] Tokenizer config not found: {tok_config_path}")
        with open(tok_config_path, 'rb') as f: self.tok_config = pickle.load(f)

        try:
            p1_paths = list(Path('.').glob(f"{args.basename}_*d_512.pkl"))
            if not p1_paths: raise StopIteration
            if len(p1_paths) > 1:
                console.print(f"[bold red]FATAL: Ambiguous Phase 1 model. Found multiple matches for '{args.basename}_*d_512.pkl':[/]")
                for p in p1_paths: console.print(f"- {p}")
                sys.exit(1)
            p1_path = p1_paths[0]
            p1_d_model_str = p1_path.stem.split('_')[-2]
            p1_d_model = int(p1_d_model_str.replace('d', ''))

        except (StopIteration, IndexError, ValueError):
            sys.exit(f"[FATAL] Could not find or parse a unique Phase 1 model file matching the pattern: '{args.basename}_*d_512.pkl'")

        console.print(f"--- Discovered and loading Phase 1 AE from: [green]{p1_path}[/green] (d_model={p1_d_model}) ---")
        self.p1_model = TopologicalCoordinateGenerator(p1_d_model, self.tok_config['latent_grid_size'], 512, self.dtype)
        with open(p1_path, 'rb') as f: self.p1_params = pickle.load(f)['params']

        self.tokenizer = LatentTokenizerVQ(**self.tok_config, dtype=self.dtype)
        self.token_map_size = (self.tok_config['latent_grid_size'] // 4) ** 2
        self.model = GenerativeConductor(args.num_codes, self.token_map_size + 1, args.d_model_cond, args.num_heads, args.num_layers, 512, self.dtype)
        
        self.interactive_state = InteractivityState()
        self.loss_history = deque(maxlen=200)
        self.sentinel_dampen_history = deque(maxlen=200)
        self.spinner_chars = ["🧠", "⚡", "💾", "📈", "🧠", "⚡", "💽", "📉"]
        self.spinner_idx = 0; self.param_count = 0; self.steps_per_sec = 0.0
        self.ui_lock = threading.Lock()
        
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        
        self.validation_prompts = [
            "A photorealistic portrait of an ancient warrior queen",
            "A cute Corgi puppy playing in a field of flowers, impressionist painting",
            "A stunning sports car racing through a neon-lit city at night, synthwave",
            "A serene Japanese zen garden with a cherry blossom tree, watercolor",
        ]
        self.current_preview_prompt_idx = 0
        self.current_preview_image_np = None

    def _generate_layout(self) -> Layout:
        with self.ui_lock:
            console = Console()
            width = console.width
            LAYOUT_BREAKPOINT = 120

            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
            spinner = self.spinner_chars[self.spinner_idx]
            precision_str = "[bold purple]BF16[/]" if self.dtype == jnp.bfloat16 else "[dim]FP32[/]"
            header_text = f"{spinner} [bold]Generative Conductor[/] | Params: [yellow]{self.param_count/1e6:.2f}M[/] | Precision: {precision_str}"
            header_panel = Panel(Align.center(header_text), style="bold magenta", title="[dim]wubumind.ai[/dim]", title_align="right")

            stats_tbl = Table.grid(expand=True, padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
            loss_val=self.loss_history[-1] if self.loss_history else 0; loss_emoji, color = ("👌","green") if loss_val < 2.5 else (("👍","yellow") if loss_val < 4.0 else ("😟","red"))
            stats_tbl.add_row("X-Entropy Loss", f"[{color}]{loss_val:.4f}[/] {loss_emoji}"); stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/] 🏃💨")
            mem, util = self._get_gpu_stats(); stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
            stats_panel = Panel(stats_tbl, title="[bold]📊 Core Stats[/]", border_style="blue", height=5)

            q_table = Table.grid(expand=True, padding=(0,1)); q_table.add_column("Ctrl", style="bold cyan", width=6); q_table.add_column("Metric", style="dim", width=10); q_table.add_column("Value", justify="left")
            if self.q_controller:
                status_full = self.q_controller.status
                status_short = status_full.split(' ')[0] # Extract main status word for tidiness
                status_emoji, color = ("😎","green") if "IMPROVING" in status_short else (("🤔","yellow") if "STAGNATED" in status_short else (("😠","red") if "REGRESSING" in status_short else (("🐣","blue") if "WARMUP" in status_short else ("🤖","dim"))))
                q_table.add_row("🧠 LR", "Status", f"[{color}]{status_short}[/] {status_emoji}"); q_table.add_row("", "Reward", f"{self.q_controller.last_reward:+.2f}")
                q_panel = Panel(q_table, title="[bold]🤖 Q-Controller[/]", border_style="green", height=4)
            else: q_panel = Panel(Align.center("[dim]Q-Ctrl Off[/dim]"), title="[bold]🤖 Q-Controller[/]", border_style="dim", height=4)

            sentinel_panel_widget = None
            if self.args.use_sentinel:
                sentinel_layout = Layout()
                # --- FIX: Corrected attribute name ---
                log_factor = self.interactive_state.sentinel_dampening_log_factor
                lever_panel = Panel(get_sentinel_lever_ascii(log_factor), title="Dampen 🚀", title_align="left")
                status_str_padded = f"{getattr(self, 'sentinel_pct', 0.0): >7.2%}"
                status_str = f"Dampened: {status_str_padded}"
                status_panel = Panel(Align.center(Text(status_str)), title="Status 🚦", height=4)
                sentinel_layout.split_row(Layout(lever_panel), Layout(status_panel))
                sentinel_panel_widget = Panel(sentinel_layout, title="[bold]🕹️ Sentinel Interactive[/]", border_style="yellow", height=11)

            spark_w = max(10, (width // 2 if width >= LAYOUT_BREAKPOINT else width) - 10)
            loss_spark = Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_history, spark_w)}[/]"), title="Loss Trend", height=3, border_style="cyan")
            graph_panels = [loss_spark]
            if self.args.use_sentinel:
                sentinel_spark = Panel(Align.center(f"[magenta]{self._get_sparkline(self.sentinel_dampen_history, spark_w)}[/]"), title="Sentinel Dampening %", height=3, border_style="magenta")
                graph_panels.append(sentinel_spark)
            trends_group = Panel(Group(*graph_panels), title="[bold]📉 Trends[/]")

            left_col_width = 50
            if width < LAYOUT_BREAKPOINT:
                available_width = width - 4
            else:
                available_width = width - left_col_width - 4
            max_preview_width = 90
            term_width = max(20, min(available_width, max_preview_width))
            term_height = int(term_width * 1.0 * 0.5)

            preview_renderable = None
            if self.current_preview_image_np is not None:
                if Pixels is None:
                    preview_renderable = Align.center(Text("Install `rich-pixels` for previews", style="yellow"))
                else:
                    pil_img = Image.fromarray(self.current_preview_image_np).resize((term_width, term_height), Image.Resampling.LANCZOS)
                    preview_renderable = Pixels.from_image(pil_img)
            else:
                waiting_text = Align.center("...Waiting for first validation step...")
                # --- [THE DEFINITIVE FIX] ---
                # Changed the incorrect keyword `minimum_height` to the correct `minimum_size`.
                # This reserves vertical space and PREVENTS the GUI from "jumping".
                preview_renderable = Layout(waiting_text, minimum_size=term_height)

            current_prompt = self.validation_prompts[self.current_preview_prompt_idx]
            prompt_text = Text(f"Prompt #{self.current_preview_prompt_idx+1}: \"{current_prompt}\"", justify="center")
            preview_panel = Panel(Group(prompt_text, Align.center(preview_renderable)), title="[bold]🖼️ Live Generation Preview[/]", border_style="green")

            layout = Layout(name="root")
            layout.split(
                Layout(header_panel, name="header", size=3),
                Layout(name="main"),
                self.progress,
                Layout(Align.center(Text("←/→: Change Preview | ↑/↓: Adjust Sentinel  |  Ctrl+C to Exit", style="dim")), name="footer", size=1)
            )

            left_column_panels = [stats_panel, q_panel]
            if sentinel_panel_widget: left_column_panels.append(sentinel_panel_widget)
            
            if width < LAYOUT_BREAKPOINT:
                layout["main"].split(Group(*left_column_panels), trends_group, preview_panel)
            else:
                left_column_panels.append(trends_group)
                layout["main"].split_row(
                    Layout(Group(*left_column_panels), name="left", size=left_col_width),
                    Layout(preview_panel, name="right", ratio=1)
                )

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
        
        @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(4,))
        def train_step_fn(state, batch, dropout_key, damp_factor, num_codes, lr):
            opt_state_list = list(state.opt_state); hp_state = opt_state_list[-1]; new_hp_state = hp_state._replace(hyperparams={'learning_rate': lr}); opt_state_list[-1] = new_hp_state; state = state.replace(opt_state=tuple(opt_state_list))
            tokens_flat, embeddings = batch; input_tokens = jnp.concatenate([jnp.full((tokens_flat.shape[0], 1), num_codes), tokens_flat], axis=1)[:, :-1]
            def loss_fn(p):
                logits = state.apply_fn({'params': p}, input_tokens, embeddings, train=True, rngs={'dropout': dropout_key})
                return optax.softmax_cross_entropy_with_integer_labels(logits, tokens_flat).mean().astype(jnp.float32)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            grads = jax.lax.pmean(grads, 'batch'); loss = jax.lax.pmean(loss, 'batch')
            new_state = state.apply_gradients(grads=grads, dampening_factor=damp_factor)
            sentinel_dampened_pct = 0.0
            if self.args.use_sentinel:
                sentinel_state = new_state.opt_state[1]
                if isinstance(sentinel_state, SentinelState): sentinel_dampened_pct = sentinel_state.dampened_pct
            return new_state, loss, sentinel_dampened_pct

        @partial(jax.jit, static_argnames=('apply_fn', 'num_codes'))
        def run_validation_batch(params, val_tokens_batch, val_embeddings_batch, apply_fn, num_codes):
            input_tokens = jnp.concatenate([jnp.full((val_tokens_batch.shape[0], 1), num_codes), val_tokens_batch], axis=1)[:, :-1]
            logits = apply_fn({'params': params}, input_tokens, val_embeddings_batch, train=False)
            return optax.softmax_cross_entropy_with_integer_labels(logits, val_tokens_batch).mean()
        
        @partial(jax.jit, static_argnames=('tokenizer_apply_fn', 'p1_model_apply_fn', 'resolution', 'num_chunks', 'grid_dim'))
        def _render_from_tokens_jit(tokenizer_apply_fn, tok_params, p1_model_apply_fn, p1_params, full_tokens, resolution, num_chunks, grid_dim):
            token_grid = full_tokens.reshape(full_tokens.shape[0], grid_dim, grid_dim)
            path_params = tokenizer_apply_fn({'params': tok_params}, token_grid, method=LatentTokenizerVQ.decode)
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

                logits, new_mutable_vars = model_apply_fn(
                    current_vars, last_token, text_emb, train=False, decode=True, index=step_index, mutable=['cache']
                )
                
                output_vars = {'params': current_vars['params'], 'cache': new_mutable_vars['cache']}

                next_token, new_ds_state = dslider_sampler_step(
                    key_step, current_ds_state, logits.squeeze(1), ds_config
                )
                
                return (output_vars, next_token[:, None], new_ds_state), next_token

            initial_token = jnp.full((text_emb.shape[0], 1), bos_token_id, dtype=jnp.int32)
            keys = jax.random.split(key, num_steps)
            initial_carry = (variables, initial_token, initial_ds_state)
            xs = (jnp.arange(num_steps), keys)
            
            _, generated_tokens_collection = jax.lax.scan(scan_body, initial_carry, xs)
            
            # --- [THE DEFINITIVE FIX] ---
            # The output of scan is (sequence_length, batch_size), e.g., (576, 1).
            # We must transpose it to (batch_size, sequence_length), e.g., (1, 576).
            # The previous `.squeeze()` call was incorrectly removing the batch dimension.
            generated_tokens = generated_tokens_collection.transpose()
            
            return generated_tokens

        @partial(jax.jit, static_argnames=('num_steps', 'resolution'))
        def _generate_validation_preview(conductor_params, initial_cache, text_emb, key, ds_config, num_steps, resolution):
            # The initial state contains both params and the empty cache
            variables = {'params': conductor_params, 'cache': initial_cache}
            
            bos_token = jnp.full((text_emb.shape[0], 1), self.args.num_codes, dtype=jnp.int32)
            
            # Run one step to populate the cache for the first token (BOS)
            initial_logits, updated_mutable_state = self.model.apply(variables, bos_token, text_emb, train=False, decode=True, index=0, mutable=['cache'])
            
            variables_for_scan = {'params': conductor_params, 'cache': updated_mutable_state['cache']}
            
            initial_ds_state = initialize_state(initial_logits, bsz=text_emb.shape[0], config=ds_config, dtype=self.dtype)
            
            # This now receives a correctly shaped (B, L) tensor.
            final_tokens = _dslider_autoregressive_sample_jit(
                self.model.apply, variables_for_scan, initial_ds_state, text_emb, 
                key, ds_config, num_steps, self.args.num_codes
            )
            
            grid_dim = self.tok_config['latent_grid_size'] // 4
            # This call will now succeed because final_tokens has the correct shape.
            return _render_from_tokens_jit(
                self.tokenizer.apply, self.tok_params, self.p1_model.apply, self.p1_params, 
                final_tokens, resolution, 16, grid_dim
            )
        
        clean_state_for_compile = jax.device_get(unreplicate(p_state))
        
        val_cpu_batches = None
        if len(val_tokens) > 0:
            val_batch_size = self.args.batch_size * self.num_devices
            MAX_VAL_SAMPLES = 1024
            num_val_to_keep = min(len(val_tokens), MAX_VAL_SAMPLES)
            num_val_to_keep = (num_val_to_keep // val_batch_size) * val_batch_size
            
            if num_val_to_keep > 0:
                val_tokens_cpu, val_embeddings_cpu = val_tokens[:num_val_to_keep], val_embeddings[:num_val_to_keep]
                val_cpu_batches = list(zip(np.split(val_tokens_cpu, num_val_to_keep // val_batch_size), np.split(val_embeddings_cpu, num_val_to_keep // val_batch_size)))
                
                dummy_val_batch = jax.device_put(val_cpu_batches[0][0]), jax.device_put(val_cpu_batches[0][1])
                run_validation_batch(clean_state_for_compile.params, *dummy_val_batch, self.model.apply, self.args.num_codes)
                
                with jax.default_device(CPU_DEVICE): initial_cache = get_initial_variables(key)['cache']
                dummy_text_emb = jnp.zeros((1, 512), dtype=self.dtype)
                _generate_validation_preview(clean_state_for_compile.params, initial_cache, dummy_text_emb, key, self.ds_config, self.token_map_size, self.preview_resolution)
                del dummy_val_batch, initial_cache, dummy_text_emb
        
        dummy_batch = next(train_iterator); sharded_batch = shard(dummy_batch)
        dummy_keys = jax.random.split(key, self.num_devices)
        train_step_fn(p_state, sharded_batch, dummy_keys, replicate(1.0), self.args.num_codes, replicate(self.args.lr))
        
        del clean_state_for_compile, dummy_batch, sharded_batch, dummy_keys
        
        console.print("[green]✅ Compilation complete. Starting training...[/green]")
        
        self.progress = Progress(TextColumn("[bold]Step {task.completed}/{task.total}"), BarColumn(), "[p.p.]{task.percentage:>3.1f}%", "•", TextColumn("Loss: {task.fields[loss]:.4f}"), "•", TextColumn("Val: {task.fields[val_loss]:.4f}"), "•", TextColumn("[bold green]Best: {task.fields[best_val_loss]:.4f}[/]"), TimeRemainingColumn())
        task_id = self.progress.add_task("train", total=self.args.steps, completed=start_step, loss=0, val_loss=0, best_val_loss=best_val_loss)
        SAVE_EVERY_N_STEPS = 2000
        last_step_time = time.time(); current_step = start_step; last_ui_update_time = 0.0; UI_UPDATE_INTERVAL_SECS = 0.25
        validation_future = None; preview_future = None; val_batch_idx = 0
        with jax.default_device(CPU_DEVICE): initial_inference_cache = get_initial_variables(key)['cache']

        try:
            with Live(self._generate_layout(), screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for step in range(start_step, self.args.steps):
                    current_step = step
                    if self.should_shutdown or self.interactive_state.shutdown_event.is_set(): break
                    
                    if validation_future is not None:
                        val_loss_val = validation_future.item()
                        self.progress.update(task_id, val_loss=val_loss_val)
                        if val_loss_val < best_val_loss:
                            best_val_loss = val_loss_val
                            if not (active_save_future and not active_save_future.done()):
                                console.print(f"\n[bold magenta]🏆 New best val loss: {best_val_loss:.4f} @ step {step}. Saving in background...[/bold magenta]")
                                host_state_unreplicated = jax.device_get(unreplicate(p_state)); data_for_save = {'params': host_state_unreplicated.params, 'val_loss': best_val_loss, 'step': step}
                                active_save_future = checkpoint_executor.submit(_save_cpu_data_task, data_for_save, ckpt_path_best)
                        validation_future = None

                    if preview_future is not None:
                        preview_future.block_until_ready()
                        with self.ui_lock:
                            self.current_preview_image_np = np.asarray(preview_future[0])
                        preview_future = None

                    batch = next(train_iterator); sharded_batch = shard(batch)
                    key, train_step_key, preview_key = jax.random.split(key, 3); sharded_keys = jax.random.split(train_step_key, self.num_devices)
                    lr = self.q_controller.choose_action() if self.q_controller else self.args.lr
                    damp_factor = self.interactive_state.get_sentinel_factor() if self.args.use_sentinel else 1.0
                    
                    p_state, p_loss, p_sentinel_pct = train_step_fn(p_state, sharded_batch, sharded_keys, replicate(damp_factor), self.args.num_codes, replicate(lr))
                    
                    loss_val = unreplicate(p_loss).item()
                    sentinel_val = unreplicate(p_sentinel_pct).item()

                    del batch, sharded_batch, sharded_keys, p_loss, p_sentinel_pct

                    with self.ui_lock: 
                        self.loss_history.append(loss_val)
                        if self.args.use_sentinel: self.sentinel_dampen_history.append(sentinel_val); self.sentinel_pct = sentinel_val
                    if self.q_controller: self.q_controller.update_q_value(loss_val)

                    preview_prompt_change = self.interactive_state.get_and_reset_preview_change()
                    if preview_prompt_change != 0:
                        with self.ui_lock: self.current_preview_prompt_idx = (self.current_preview_prompt_idx + preview_prompt_change) % len(self.validation_prompts)

                    if (step + 1) % self.args.eval_every == 0 and val_cpu_batches is not None:
                        unrep_state_for_eval = jax.device_get(unreplicate(p_state))
                        
                        tokens_batch_cpu, embeddings_batch_cpu = val_cpu_batches[val_batch_idx]
                        val_batch_gpu = jax.device_put(tokens_batch_cpu), jax.device_put(embeddings_batch_cpu)
                        validation_future = run_validation_batch(unrep_state_for_eval.params, *val_batch_gpu, self.model.apply, self.args.num_codes)
                        val_batch_idx = (val_batch_idx + 1) % len(val_cpu_batches)
                        
                        if preview_future is None:
                            prompt_idx = self.current_preview_prompt_idx
                            text_emb = jnp.expand_dims(self.validation_embeddings[prompt_idx], 0)
                            preview_future = _generate_validation_preview(unrep_state_for_eval.params, initial_inference_cache, text_emb, preview_key, self.ds_config, self.token_map_size, self.preview_resolution)
                        
                        del unrep_state_for_eval, val_batch_gpu, text_emb

                    self.progress.update(task_id, advance=1, loss=loss_val, best_val_loss=best_val_loss)
                    
                    if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                        if not (active_save_future and not active_save_future.done()):
                            host_state_unreplicated = jax.device_get(unreplicate(p_state))
                            q_state = self.q_controller.state_dict() if self.q_controller else None
                            data_for_save = {'params': host_state_unreplicated.params, 'opt_state': host_state_unreplicated.opt_state, 'step': step + 1, 'q_controller_state': q_state}
                            active_save_future = checkpoint_executor.submit(_save_cpu_data_task, data_for_save, ckpt_path_final)

                    current_time = time.time()
                    self.steps_per_sec = 1.0 / (current_time - last_step_time + 1e-9); last_step_time = current_time
                    if current_time - last_ui_update_time > UI_UPDATE_INTERVAL_SECS: live.update(self._generate_layout()); last_ui_update_time = current_time
        finally:
            console.print(f"\n[yellow]--- Training loop exited at step {current_step + 1}. Waiting for final save... ---[/yellow]")
            checkpoint_executor.shutdown(wait=True)
            if 'p_state' in locals():
                host_state = jax.device_get(unreplicate(p_state))
                final_data_to_save = {'params': host_state.params, 'opt_state': host_state.opt_state, 'step': current_step + 1, 'q_controller_state': self.q_controller.state_dict() if self.q_controller else None}
                with open(ckpt_path_final, 'wb') as f: pickle.dump(final_data_to_save, f)
                console.print(f"✅ Final resume-state saved to [green]{ckpt_path_final}[/green]")
                del p_state, host_state
            
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
    def __init__(self, args):
        self.args = args; console = Console()
        console.print("--- 🧠 Loading Full Generative Stack (Agnostic Mode) ---", style="bold yellow")
        try:
            conductor_config_path = next(Path('.').glob(f"conductor_{args.basename}_*_config.pkl"))
            tokenizer_config_path = next(Path('.').glob(f"tokenizer_{args.basename}_*_config.pkl"))
        except StopIteration:
            sys.exit(f"[FATAL] Could not find required config files for basename '{args.basename}'. Please train the models first.")

        with open(conductor_config_path, 'rb') as f: self.cond_config = pickle.load(f)
        with open(tokenizer_config_path, 'rb') as f: self.tok_config = pickle.load(f)
        
        self.dtype = jnp.float32 # Use FP32 for inference for max quality
        p1_d_model_str = args.basename.split('_')[-1]
        try: p1_d_model = int(p1_d_model_str.replace('d',''))
        except (ValueError, IndexError): sys.exit(f"[FATAL] Could not parse d_model from basename '{args.basename}'. Expected format like '..._96d'.")
        
        self.p1_model = TopologicalCoordinateGenerator(p1_d_model, self.tok_config['latent_grid_size'], 512, self.dtype)
        self.tokenizer = LatentTokenizerVQ(**self.tok_config, dtype=self.dtype)
        self.conductor = GenerativeConductor(**self.cond_config, dtype=self.dtype)
        self.p1_encoder = PathModulator(self.tok_config['latent_grid_size'], 512, self.dtype)
        
        p1_path = next(Path('.').glob(f"{args.basename}_512.pkl"))
        tok_path = Path(str(tokenizer_config_path).replace("_config.pkl", ".pkl"))
        cond_path = Path(str(conductor_config_path).replace("_config.pkl", ".pkl"))
        
        with open(p1_path, 'rb') as f: self.p1_params=pickle.load(f)['params']
        with open(tok_path,'rb') as f: self.tok_params=pickle.load(f)['params']
        with open(cond_path,'rb') as f: self.cond_params=pickle.load(f)['params']
        console.print(f"✅ Models and configs loaded for basename [cyan]'{args.basename}'[/cyan]")

        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device); console.print("✅ [CLIP] Text Encoder loaded.")
        
        self._jit_get_logits = jax.jit(lambda t,e: self.conductor.apply({'params':self.cond_params},t,e))
        self._jit_decode_tokens = jax.jit(lambda i: self.tokenizer.apply({'params':self.tok_params}, i, method=self.tokenizer.decode))
        self._jit_encode_tokens = jax.jit(lambda p: self.tokenizer.apply({'params':self.tok_params}, p, method=self.tokenizer.encode))
        self.p1_encoder_fn = jax.jit(lambda i: self.p1_encoder.apply({'params':self.p1_params['modulator']}, i))

        @partial(jax.jit, static_argnames=('resolution','patch_size'))
        def _render(params, latent_batch, resolution=512, patch_size=256):
            coords = jnp.stack(jnp.meshgrid(jnp.linspace(-1,1,resolution),jnp.linspace(-1,1,resolution),indexing='ij'),-1).reshape(-1,2)
            pixels = [self.p1_model.apply({'params':params}, latent_batch, c, method=self.p1_model.decode) for c in jnp.array_split(coords, (resolution**2)//(patch_size**2))]
            return jnp.concatenate(pixels, axis=1).reshape(latent_batch.shape[0], resolution, resolution, 3)
        self._render_fn = _render

    def _sample(self, key, text_emb, num_steps, initial_tokens, temp=1.0, top_k=50):
        tokens = initial_tokens
        for _ in range(num_steps):
            key, subkey = jax.random.split(key)
            logits = self._jit_get_logits(tokens, text_emb)[:, -1, :] / temp
            top_k_logits, top_k_indices = jax.lax.top_k(logits, k=top_k)
            next_token_idx = jax.random.categorical(subkey, top_k_logits)
            next_token = jnp.take_along_axis(top_k_indices, next_token_idx[:, None], axis=-1)
            tokens = jnp.concatenate([tokens, next_token], axis=1)
        return tokens

    def generate(self, prompt, seed):
        console = Console(); console.print(f"--- 🎨 Generating image for prompt: \"[italic yellow]{prompt}[/italic yellow]\" ---")
        key = jax.random.PRNGKey(seed)
        with torch.no_grad(): text_emb = self.clip_model.encode_text(clip.tokenize([prompt]).to(_clip_device)).cpu().numpy().astype(jnp.float32)
        console.print("1/3: Composing scene with Conductor..."); initial_tokens = jnp.full((1, 1), self.cond_config['num_codes'], dtype=jnp.int32); tokens_with_bos = self._sample(key, text_emb, self.cond_config['num_positions'] - 1, initial_tokens, temp=self.args.temp, top_k=self.args.top_k)
        console.print("2/3: Decoding tokens with Tokenizer..."); token_grid = tokens_with_bos[:, 1:].reshape(1, self.tok_config['latent_grid_size']//4, self.tok_config['latent_grid_size']//4); path_params = self._jit_decode_tokens(token_grid)
        console.print("3/3: Rendering final 512x512 image..."); recon_np = np.array(((self._render_fn(self.p1_params, path_params)[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        filename = f"GEN_{Path(self.args.basename).stem}_{prompt.replace(' ', '_')[:40]}_{seed}.png"; Image.fromarray(recon_np).save(filename)
        console.print(f"✅ Image saved to [green]{filename}[/green]")
        
    def edit(self, source_image_path, new_prompt, seed):
        console = Console(); key = jax.random.PRNGKey(seed); console.print(f"--- ✏️ Editing [cyan]{source_image_path}[/cyan] with prompt: \"[italic yellow]{new_prompt}[/italic yellow]\" ---")
        console.print("1/4: Encoding source image..."); img = Image.open(source_image_path).convert("RGB").resize((512,512)); path_params = self.p1_encoder_fn(jnp.expand_dims((np.array(img, dtype=np.float32)/127.5)-1.0, 0)); source_tokens_flat = self._jit_encode_tokens(path_params).reshape(1, -1)
        with torch.no_grad(): text_emb = self.clip_model.encode_text(clip.tokenize([new_prompt]).to(_clip_device)).cpu().numpy().astype(jnp.float32)
        console.print("2/4: Re-composing scene with new prompt..."); num_to_keep = (self.cond_config['num_positions'] - 1) // 2; initial_tokens = jnp.concatenate([jnp.full((1,1), self.cond_config['num_codes']), source_tokens_flat[:, :num_to_keep]], axis=1); composed_tokens = self._sample(key, text_emb, (self.cond_config['num_positions']-1) - num_to_keep, initial_tokens, temp=self.args.temp, top_k=self.args.top_k)
        console.print("3/4: Decoding new tokens..."); token_grid = composed_tokens[:, 1:].reshape(1, self.tok_config['latent_grid_size']//4, self.tok_config['latent_grid_size']//4); new_path_params = self._jit_decode_tokens(token_grid)
        console.print("4/4: Rendering edited 512x512 image..."); recon_np = np.array(((self._render_fn(self.p1_params, new_path_params)[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8))
        filename = f"EDIT_{Path(self.args.basename).stem}_{Path(source_image_path).stem}_{new_prompt.replace(' ', '_')[:20]}_{seed}.png"; Image.fromarray(recon_np).save(filename)
        console.print(f"✅ Edited image saved to [green]{filename}[/green]")

# =================================================================================================
# 7. MAIN EXECUTION BLOCK
# =================================================================================================
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
    
    # --- PARSER DEFINITIONS (MODIFIED) ---
    p_prep = subparsers.add_parser("prepare-paired-data", help="Pre-process (image, text) -> (latent, embedding).", parents=[base_parser])
    p_prep.add_argument('--data-dir', type=str, required=True); p_prep.add_argument('--d-model', type=int, required=True); p_prep.add_argument('--latent-grid-size', type=int, required=True); p_prep.add_argument('--batch-size', type=int, default=128)

    # New parser for tokenizer preprocessing
    p_prep_tok = subparsers.add_parser("prepare-tokenizer-data", help="Pre-process images into latents for the tokenizer.", parents=[base_parser])
    p_prep_tok.add_argument('--data-dir', type=str, required=True); p_prep_tok.add_argument('--d-model', type=int, required=True); p_prep_tok.add_argument('--latent-grid-size', type=int, required=True); p_prep_tok.add_argument('--batch-size', type=int, default=128)

    p_tok = subparsers.add_parser("train-tokenizer", help="Train the Latent Tokenizer (VQ-VAE).", parents=[base_parser, train_parser])
    p_tok.add_argument('--data-dir', type=str, required=True); p_tok.add_argument('--d-model', type=int, required=True); p_tok.add_argument('--latent-grid-size', type=int, required=True)
    p_tok.add_argument('--steps', type=int, default=150000); p_tok.add_argument('--batch-size', type=int, default=128); p_tok.add_argument('--lr', type=float, default=3e-4)
    p_tok.add_argument('--num-codes', type=int, default=8192); p_tok.add_argument('--code-dim', type=int, default=256)

    p_cond = subparsers.add_parser("train-conductor", help="Train the Generative Conductor (Transformer).", parents=[base_parser, train_parser])
    p_cond.add_argument('--data-dir', type=str, required=True); p_cond.add_argument('--latent-grid-size', type=int, required=True)
    p_cond.add_argument('--steps', type=int, default=1500000); p_cond.add_argument('--batch-size', type=int, default=32, help="Batch size PER DEVICE."); p_cond.add_argument('--lr', type=float, default=1e-4)
    p_cond.add_argument('--eval-every', type=int, default=500, help="Run validation every N steps.") # <-- ADD THIS LINE
    p_cond.add_argument('--num-codes', type=int, default=8192); p_cond.add_argument('--code-dim', type=int, default=256)
    p_cond.add_argument('--num-layers', type=int, default=12); p_cond.add_argument('--d-model-cond', type=int, default=768); p_cond.add_argument('--num-heads', type=int, default=12)
    p_cond.add_argument('--vram-saver-mode', action='store_true', help="Enable aggressive VRAM saving optimizations (e.g., lower-res previews). Recommended for <= 8GB GPUs.")
    
    inference_parser = argparse.ArgumentParser(add_help=False); inference_parser.add_argument('--temp', type=float, default=0.9); inference_parser.add_argument('--top-k', type=int, default=256)
    p_gen = subparsers.add_parser("generate", help="Generate an image from a text prompt.", parents=[base_parser, inference_parser]); p_gen.add_argument('--prompt', type=str, required=True); p_gen.add_argument('--seed', type=int, default=lambda: int(time.time()))
    p_edit = subparsers.add_parser("edit", help="Edit an image using a new text prompt.", parents=[base_parser, inference_parser]); p_edit.add_argument('--source-image', type=str, required=True); p_edit.add_argument('--prompt', type=str, required=True); p_edit.add_argument('--seed', type=int, default=lambda: int(time.time()))
    
    args = parser.parse_args()
    
    if args.command in ["generate", "edit"]: args.seed = args.seed() if callable(args.seed) else args.seed
    if args.command == "train-conductor":
        if args.d_model_cond % args.num_heads != 0: sys.exit(f"[FATAL] Conductor d_model_cond ({args.d_model_cond}) must be divisible by num_heads ({args.num_heads}).")
        if args.latent_grid_size % 4 != 0: sys.exit(f"[FATAL] latent_grid_size ({args.latent_grid_size}) must be divisible by 4 due to tokenizer architecture.")

    # --- COMMAND DISPATCH (MODIFIED) ---
    if args.command == "prepare-paired-data": prepare_paired_data(args)
    elif args.command == "prepare-tokenizer-data": prepare_tokenizer_data(args) # New command
    elif args.command == "train-tokenizer": TokenizerTrainer(args).train()
    elif args.command == "train-conductor": ConductorTrainer(args).train()
    elif args.command == "generate": Generator(args).generate(args.prompt, args.seed)
    elif args.command == "edit": Generator(args).edit(args.source_image, args.prompt, args.seed)

if __name__ == "__main__":
    main()
