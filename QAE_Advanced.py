import os
# --- Environment Setup for JAX/TensorFlow ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
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
import platform
import threading
from functools import partial
from pathlib import Path
from collections import deque
from PIL import Image
import jax.scipy.ndimage
import imageio
# You will need this import, you can place it with the other imports
# at the top of the file or inside the function as shown.
from flax.traverse_util import path_aware_map

# --- JAX Configuration ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_threefry_partitionable', True)

# Conditional imports for keyboard listening
if platform.system() == "Windows":
    import msvcrt
else:
    import tty, termios, select

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

class SentinelState(NamedTuple):
    sign_history: chex.ArrayTree; dampened_count: Optional[jnp.ndarray] = None; dampened_pct: Optional[jnp.ndarray] = None

def sentinel(dampening_factor: float = 0.1, history_len: int = 5, oscillation_threshold: int = 3) -> optax.GradientTransformation:
    """An Optax component to dampen oscillating gradients."""
    def init_fn(params):
        sign_history = jax.tree_util.tree_map(lambda t: jnp.zeros((history_len,) + t.shape, dtype=jnp.int8), params)
        return SentinelState(sign_history=sign_history, dampened_count=jnp.array(0), dampened_pct=jnp.array(0.0))
    def update_fn(updates, state, params=None, **kwargs):
        new_sign_history = jax.tree_util.tree_map(lambda old_hist, new_sign: jnp.roll(old_hist, shift=-1, axis=0).at[history_len-1].set(new_sign.astype(jnp.int8)), state.sign_history, jax.tree_util.tree_map(jnp.sign, updates))
        is_oscillating = jax.tree_util.tree_map(lambda hist: jnp.sum(jnp.abs(jnp.diff(hist, axis=0)), axis=0) >= oscillation_threshold, new_sign_history)
        dampening_mask = jax.tree_util.tree_map(lambda is_osc: jnp.where(is_osc, dampening_factor, 1.0), is_oscillating)
        dampened_updates = jax.tree_util.tree_map(lambda u, m: u * m, updates, dampening_mask)
        num_oscillating = jax.tree_util.tree_reduce(lambda acc, x: acc + jnp.sum(x), is_oscillating, 0)
        total_params = jax.tree_util.tree_reduce(lambda acc, x: acc + x.size, params, 0)
        new_state = SentinelState(sign_history=new_sign_history, dampened_count=num_oscillating, dampened_pct=(num_oscillating / (total_params + 1e-8)))
        return dampened_updates, new_state
    return optax.GradientTransformation(init_fn, update_fn)

Q_CONTROLLER_CONFIG_NORMAL = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.7, 0.9, 1.0, 1.1, 1.3], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-7, "lr_max": 2e-3, "metric_history_len": 5000, "loss_min": 0.05, "loss_max": 1.5, "exploration_rate_q": 0.3, "min_exploration_rate": 0.05, "exploration_decay": 0.9995, "trend_window": 777, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": 10.0, "stagnation_penalty": -2.0}
Q_CONTROLLER_CONFIG_FINETUNE = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.8, 0.95, 1.0, 1.05, 1.2], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-8, "lr_max": 1e-2, "metric_history_len": 5000, "loss_min": 0.0, "loss_max": 0.5, "exploration_rate_q": 0.15, "min_exploration_rate": 0.02, "exploration_decay": 0.9998, "target_pixel_loss": 0.01}

class JaxHakmemQController:
    """A Q-Learning agent to dynamically control the learning rate."""
    def __init__(self,initial_lr:float,config:Dict[str,Any]):
        self.config=config; self.current_lr=initial_lr; self.q_table_size=int(self.config["q_table_size"]); self.num_actions=int(self.config["num_lr_actions"]); self.lr_change_factors=self.config["lr_change_factors"]; self.q_table=np.zeros((self.q_table_size,self.num_actions),dtype=np.float32); self.learning_rate_q=float(self.config["learning_rate_q"]); self.discount_factor_q=float(self.config["discount_factor_q"]); self.lr_min=float(self.config["lr_min"]); self.lr_max=float(self.config["lr_max"]); self.loss_history=deque(maxlen=int(self.config["metric_history_len"])); self.loss_min=float(self.config["loss_min"]); self.loss_max=float(self.config["loss_max"]); self.last_action_idx:Optional[int]=None; self.last_state_idx:Optional[int]=None; self.initial_exploration_rate = float(self.config["exploration_rate_q"]); self.exploration_rate_q = self.initial_exploration_rate; self.min_exploration_rate = float(self.config["min_exploration_rate"]); self.exploration_decay = float(self.config["exploration_decay"]); self.status: str = "STARTING"; self.last_reward: float = 0.0
        self.trend_window = int(config["trend_window"]); self.pixel_loss_trend_history = deque(maxlen=self.trend_window); self.improve_threshold = float(config["improve_threshold"]); self.regress_threshold = float(config["regress_threshold"]); self.regress_penalty = float(config["regress_penalty"]); self.stagnation_penalty = float(config["stagnation_penalty"]); self.last_slope: float = 0.0; print(f"--- Q-Controller initialized in 3-STATE SEARCH mode. Trend Window: {self.trend_window} steps ---")
    def _discretize_value(self,value:float) -> int:
        if not np.isfinite(value): return self.q_table_size // 2
        if value<=self.loss_min: return 0
        if value>=self.loss_max: return self.q_table_size-1
        bin_size=(self.loss_max-self.loss_min)/self.q_table_size; return min(int((value-self.loss_min)/bin_size),self.q_table_size-1)
    def _get_current_state_idx(self) -> int:
        if not self.loss_history: return self.q_table_size//2
        avg_loss=np.mean(list(self.loss_history)[-5:]); return self._discretize_value(avg_loss)
    def choose_action(self) -> float:
        self.last_state_idx=self._get_current_state_idx()
        if np.random.rand()<self.exploration_rate_q: self.last_action_idx=np.random.randint(0,self.num_actions)
        else: self.last_action_idx=np.argmax(self.q_table[self.last_state_idx]).item()
        change_factor=self.lr_change_factors[self.last_action_idx]; self.current_lr=np.clip(self.current_lr*change_factor,self.lr_min,self.lr_max)
        return self.current_lr
    def update_q_value(self, total_loss:float):
        self.loss_history.append(total_loss)
        if self.last_state_idx is None or self.last_action_idx is None: return
        reward = self._calculate_reward(total_loss); self.last_reward = reward
        current_q = self.q_table[self.last_state_idx, self.last_action_idx]; next_state_idx = self._get_current_state_idx(); max_next_q = np.max(self.q_table[next_state_idx])
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor_q * max_next_q - current_q); self.q_table[self.last_state_idx, self.last_action_idx] = new_q
        self.exploration_rate_q = max(self.min_exploration_rate, self.exploration_rate_q * self.exploration_decay)
    def _calculate_reward(self, current_loss):
        self.pixel_loss_trend_history.append(current_loss)
        if len(self.pixel_loss_trend_history) < self.trend_window: return 0.0
        loss_window = np.array(self.pixel_loss_trend_history); slope = np.polyfit(np.arange(self.trend_window), loss_window, 1)[0]; self.last_slope = slope
        if slope < -self.improve_threshold: self.status = f"IMPROVING (S={slope:.2e})"; reward = abs(slope) * 1000
        elif slope > self.regress_threshold: self.status = f"REGRESSING (S={slope:.2e})"; reward = -abs(slope) * 1000 - self.regress_penalty
        else: self.status = f"STAGNATED (S={slope:.2e})"; reward = self.stagnation_penalty
        return reward
    def state_dict(self)->Dict[str,Any]:
        return {"current_lr":self.current_lr,"q_table":self.q_table.tolist(),"loss_history":list(self.loss_history), "exploration_rate_q":self.exploration_rate_q, "pixel_loss_trend_history": list(self.pixel_loss_trend_history)}
    def load_state_dict(self,state_dict:Dict[str,Any]):
        self.current_lr=state_dict.get("current_lr",self.current_lr); self.q_table=np.array(state_dict.get("q_table",self.q_table.tolist()),dtype=np.float32); self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.loss_history.maxlen); self.exploration_rate_q=state_dict.get("exploration_rate_q", self.initial_exploration_rate)
        self.pixel_loss_trend_history=deque(state_dict.get("pixel_loss_trend_history",[]),maxlen=self.trend_window)

# =================================================================================================
# 2. MATHEMATICAL & MODEL FOUNDATIONS
# =================================================================================================

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2) * jnp.cos(chi_f32)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        is_power_of_two = self.latent_grid_size > 0 and (self.latent_grid_size & (self.latent_grid_size - 1) == 0)
        if not is_power_of_two or self.latent_grid_size > self.input_image_size: raise ValueError(f"latent_grid_size must be a power of two. Got {self.latent_grid_size}.")
        num_downsamples = int(math.log2(self.input_image_size / self.latent_grid_size)); x = images; features = 32
        for i in range(num_downsamples):
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x); features *= 2
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        path_params = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params[..., 0]) * jnp.pi; chi_c = nn.tanh(path_params[..., 1]) * (jnp.pi / 4.0); radius = nn.sigmoid(path_params[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1)

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = path_params_grid.shape; L = H * W; path_params = path_params_grid.reshape(B, L, C)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta); chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path)
        accumulated_t_co = jnp.cumprod(t_co_steps, axis=-1)[:, :, -1]
        complex_measurement = jnp.stack([accumulated_t_co.real, accumulated_t_co.imag], axis=-1)
        feature_vectors = nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(complex_measurement)
        return feature_vectors.reshape(B, H, W, self.d_model)

class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)

class CoordinateDecoder(nn.Module):
    d_model: int; num_freqs: int = 10; mlp_width: int = 128; mlp_depth: int = 3; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        B = feature_grid.shape[0]; pos_encoder = PositionalEncoding(self.num_freqs, dtype=self.dtype); encoded_coords = pos_encoder(coords)
        coords_rescaled = (coords + 1) / 2 * (jnp.array(feature_grid.shape[1:3], dtype=self.dtype) - 1)
        def sample_one_image(single_feature_grid):
            grid_chw = single_feature_grid.transpose(2, 0, 1); sampled_channels = jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw); return sampled_channels.T
        sampled_features = jax.vmap(sample_one_image)(feature_grid); encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)
        mlp_input = jnp.concatenate([encoded_coords_tiled, sampled_features], axis=-1); h = mlp_input
        for i in range(self.mlp_depth): h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h); h = nn.gelu(h)
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

class LatentCorrectionNetwork(nn.Module):
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, p_warped: jnp.ndarray, flow_latent_res: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([p_warped, flow_latent_res], axis=-1); features = 16
        x = nn.Conv(features, (3, 3), padding='SAME', name="corr_conv_1", dtype=self.dtype)(x); x = nn.gelu(x)
        x = nn.Conv(features, (3, 3), padding='SAME', name="corr_conv_2", dtype=self.dtype)(x); x = nn.gelu(x)
        residual_grid = nn.Conv(3, (1, 1), name="residual_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(x)
        return nn.tanh(residual_grid) * 0.1

class VideoCodecModel(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int = 512; dtype: Any = jnp.float32
    def setup(self):
        self.image_codec = TopologicalCoordinateGenerator(d_model=self.d_model, latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, name="image_codec", dtype=self.dtype)
        self.correction_net = LatentCorrectionNetwork(name="correction_net", dtype=self.dtype)


# =================================================================================================
# 3. DATA HANDLING
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

def create_dataset(image_dir: str, batch_size: int, is_training: bool = True):
    record_file = Path(image_dir)/"data_512x512.tfrecord"
    if not record_file.exists(): raise FileNotFoundError(f"{record_file} not found. Run 'prepare-data' first.")
    def _parse(proto):
        f={'image':tf.io.FixedLenFeature([],tf.string)}; p=tf.io.parse_single_example(proto,f)
        img=(tf.cast(tf.io.decode_jpeg(p['image'],3),tf.float32)/127.5)-1.0; img.set_shape([512,512,3]); return img
    ds = tf.data.TFRecordDataset(str(record_file)).map(_parse,num_parallel_calls=tf.data.AUTOTUNE)
    if is_training: ds = ds.shuffle(512).repeat()
    return ds.batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)

def prepare_video_data(video_path: str, data_dir: str):
    if cv2 is None: raise ImportError("OpenCV is required for video preparation.")
    video_p, data_p = Path(video_path), Path(data_dir); data_p.mkdir(exist_ok=True)
    frames_dir, flow_dir = data_p / "frames", data_p / "flow"
    if (data_p/"prep_complete.flag").exists(): print(f"✅ Video data already prepared in {data_dir}. Skipping."); return
    frames_dir.mkdir(exist_ok=True); flow_dir.mkdir(exist_ok=True)
    print(f"--- Preparing video data from {video_path} into {data_dir} ---"); cap = cv2.VideoCapture(str(video_p)); frames = []
    print("Step 1/3: Reading frames...")
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read();
        if not ret: break
        frames.append(np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512, 512), Image.Resampling.LANCZOS)))
    cap.release(); print("Step 2/3: Saving frames...")
    for i, frame_np in enumerate(tqdm(frames)): Image.fromarray(frame_np).save(frames_dir / f"frame_{i:05d}.jpg", quality=95)
    print("Step 3/3: Calculating and saving optical flow..."); prvs = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in tqdm(range(1, len(frames))):
        nxt = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        np.save(flow_dir / f"flow_{i-1:05d}.npy", flow.astype(np.float16)); prvs = nxt
    with open(data_p/"dataset_info.pkl", 'wb') as f: pickle.dump({'num_frames': len(frames)}, f)
    (data_p/"prep_complete.flag").touch(); print("✅ Video data preparation complete.")

# Replace the old create_video_dataset function with this one

def create_video_dataset(data_dir: str, batch_size: int, clip_len: int):
    data_p = Path(data_dir)
    frames_dir = data_p / "frames"
    with open(data_p/"dataset_info.pkl", 'rb') as f:
        num_frames = pickle.load(f)['num_frames']

    def generator():
        while True:
            start_idx = np.random.randint(0, num_frames - clip_len)
            frames = []
            for i in range(clip_len):
                frame_path = frames_dir / f"frame_{start_idx+i:05d}.jpg"
                frame = (np.array(Image.open(frame_path), dtype=np.float32) / 127.5) - 1.0
                frames.append(frame)
            # Only yield the frames now
            yield np.stack(frames, axis=0)

    # The output signature is now just one tensor
    output_signature = tf.TensorSpec(shape=(clip_len, 512, 512, 3), dtype=tf.float32)
    return tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
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
        model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size)
        super().__init__(args, model)

    def train(self):
        ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        components = [optax.clip_by_global_norm(1.0)]
        if self.args.use_sentinel: components.append(sentinel())
        base_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr)
        optimizer = optax.chain(*components, base_optimizer)

        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path,'rb') as f: data=pickle.load(f)
            params = data['params']
            state_template = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
            if 'opt_state' in data and jax.tree_util.tree_structure(state_template.opt_state) == jax.tree_util.tree_structure(data['opt_state']):
                state = state_template.replace(opt_state=data['opt_state']); print("--- Full optimizer state loaded successfully. ---")
            else:
                state = state_template; print("[bold yellow]Warning: Optimizer state mismatch. Re-initializing optimizer.[/bold yellow]")
            start_epoch = data.get('epoch', 0) + 1
            if self.q_controller and 'q_controller_state' in data:
                self.q_controller.load_state_dict(data['q_controller_state']); print("--- Q-Controller state loaded. ---")
        else:
            print("--- Initializing new model ---")
            with jax.default_device(CPU_DEVICE):
                dummy_images = jnp.zeros((1, 512, 512, 3)); dummy_coords = jnp.zeros((1024, 2))
                params = self.model.init(jax.random.PRNGKey(0), dummy_images, dummy_coords)['params']
            state = train_state.TrainState.create(apply_fn=self.model.apply,params=params,tx=optimizer); start_epoch=0

        p_state = replicate(state)

        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,1))
        def train_step(state, batch):
            image_res = batch.shape[1]; x = jnp.linspace(-1, 1, image_res); coords = jnp.stack(jnp.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
            target_pixels = batch.reshape(batch.shape[0], -1, 3)
            def loss_fn(params):
                feature_grid = self.model.apply({'params': params}, batch, method=lambda m, i: m.observer(m.modulator(i)))
                num_pixels = coords.shape[0]; decoder_patch_size = 8192
                coords_patched = jnp.array_split(coords, (num_pixels + decoder_patch_size -1) // decoder_patch_size, axis=0)
                targets_patched = jnp.array_split(target_pixels, (num_pixels + decoder_patch_size -1) // decoder_patch_size, axis=1)
                total_recon_error = 0.0
                for coord_patch, target_patch in zip(coords_patched, targets_patched):
                    recon_patch = self.model.apply({'params': params}, feature_grid, coord_patch, method=lambda m, fg, c: m.coord_decoder(fg, c))
                    total_recon_error += jnp.sum(jnp.abs(target_patch - recon_patch))
                return total_recon_error / num_pixels
            loss, grads = jax.value_and_grad(loss_fn)(state.params); grads = jax.lax.pmean(grads, 'devices')
            return state.apply_gradients(grads=grads), jax.lax.pmean(loss, 'devices')

        @partial(jax.jit, static_argnames=('resolution', 'patch_size'))
        def generate_preview(params, image_batch, resolution=128, patch_size=64):
            latent_params = self.model.apply({'params': params}, image_batch, method=lambda m, i: m.modulator(i))
            x = jnp.linspace(-1, 1, resolution); full_coords = jnp.stack(jnp.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
            pixels = [self.model.apply({'params': params}, latent_params, c, method=lambda m, p, c: m.decode(p, c)) for c in jnp.array_split(full_coords, (resolution**2)//(patch_size**2))]
            return jnp.concatenate(pixels, axis=1).reshape(1, resolution, resolution, 3)

        dataset = create_dataset(self.args.image_dir, self.args.batch_size*self.num_devices); preview_batch = next(dataset.as_numpy_iterator())[0:1]; it = dataset.as_numpy_iterator()
        with open(Path(self.args.image_dir)/"dataset_info.pkl",'rb') as f: steps_per_epoch = pickle.load(f)['num_samples'] // (self.args.batch_size*self.num_devices)

        print("--- Compiling JAX functions (one-time cost)... ---")
        generate_preview(unreplicate(p_state.params), preview_batch); p_state, _ = train_step(p_state, common_utils.shard(np.repeat(preview_batch, self.num_devices, axis=0)))
        print("--- Compilation complete. Starting training. ---")

        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Phase 1: Static Image AE[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2)); layout["left"].split(Layout(name="stats"), Layout(name="preview"))
        preview_panel = Panel("...", title="[bold]🔎 Preview[/]", border_style="green", height=18); layout["left"]["preview"].update(preview_panel)
        progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn(), TextColumn("LR: {task.fields[lr]:.2e}"))
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs, lr=self.args.lr); layout['footer'].update(progress)
        epoch_for_save = start_epoch
        with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            try:
                for epoch in range(start_epoch, self.args.epochs):
                    epoch_for_save = epoch; progress.update(epoch_task, completed=0, epoch=epoch+1)
                    for step in range(steps_per_epoch):
                        if self.should_shutdown: break
                        if self.q_controller:
                            current_lr = self.q_controller.choose_action()
                            opt_state_unrep = unreplicate(p_state.opt_state)
                            opt_state_unrep[-1].hyperparams['learning_rate'] = jnp.asarray(current_lr)
                            p_state = p_state.replace(opt_state=replicate(opt_state_unrep))
                        else: current_lr = self.args.lr
                        p_state, loss = train_step(p_state, common_utils.shard(next(it))); loss_val = unreplicate(loss)
                        if self.q_controller: self.q_controller.update_q_value(loss_val)
                        if step % 2 == 0:
                            self.recon_loss_history.append(loss_val)
                            stats_tbl=Table(show_header=False,box=None,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
                            stats_tbl.add_row("Image Loss",f"[bold green]{loss_val:.4f}[/]"); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util",f"[yellow]{util}[/]")
                            if self.q_controller:
                                status_word = self.q_controller.status.split(" ")[0]
                                status_map = {"IMPROVING": "bold green", "STAGNATED": "bold yellow", "REGRESSING": "bold red", "STARTING": "dim"}
                                color = status_map.get(status_word, "dim"); stats_tbl.add_row("Q-Ctrl Status", f"[{color}]{self.q_controller.status}[/]")
                            if self.args.use_sentinel:
                                try: dampened_pct = unreplicate(p_state.opt_state)[-2][1].dampened_pct; stats_tbl.add_row("Sentinel Dampen", f"[cyan]{dampened_pct:.3%}[/]")
                                except (IndexError, AttributeError): pass
                            layout["left"]["stats"].update(Panel(stats_tbl,title="[bold]📊 Stats[/]",border_style="blue"))
                            spark_w = max(1, (live.console.width*2//3)-10)
                            recon_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.recon_loss_history,spark_w)}[/]"),title=f"Reconstruction Loss (L1)",height=3, border_style="cyan")
                            layout["right"].update(Panel(recon_panel,title="[bold]📉 Losses[/]",border_style="magenta"))
                        if step > 0 and step % 25 == 0:
                            recon = generate_preview(unreplicate(p_state.params), preview_batch)
                            orig_np = np.array((preview_batch[0]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                            recon_np = np.array((recon[0]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                            self._update_preview_panel(preview_panel, orig_np, recon_np)
                        progress.update(epoch_task, advance=1, lr=current_lr)
                    if self.should_shutdown: break
                    if jax.process_index() == 0:
                        self._save_checkpoint(p_state, epoch, ckpt_path)
                        live.console.print(f"--- :floppy_disk: Checkpoint saved for epoch {epoch+1} ---")
            except Exception as e: live.stop(); print("\n[bold red]FATAL: Training loop crashed![/bold red]"); raise e
        if jax.process_index() == 0 and 'p_state' in locals():
            print("\n--- Training finished or interrupted. Saving final state... ---")
            self._save_checkpoint(p_state, epoch_for_save, ckpt_path)
            print("--- :floppy_disk: Final state saved. ---")


class VideoTrainer(AdvancedTrainer):
    def __init__(self, args):
        model = VideoCodecModel(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size)
        super().__init__(args, model)

    def train(self):
        phase1_ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        phase2_ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_512_video_autoregressive.pkl")
        if not phase1_ckpt_path.exists(): print(f"[FATAL] Phase 1 checkpoint not found at {phase1_ckpt_path}. Run 'train' first."), sys.exit(1)
        print("--- Loading Phase 1 (frozen) model ---");
        with open(phase1_ckpt_path, 'rb') as f: phase1_data = pickle.load(f)

        components = [optax.clip_by_global_norm(1.0)]
        if self.args.use_sentinel: components.append(sentinel())
        base_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr)
        trainable_optimizer = optax.chain(*components, base_optimizer)

        loaded_opt_state = None
        if phase2_ckpt_path.exists():
            print(f"--- Resuming Auto-Regressive training from {phase2_ckpt_path} ---")
            with open(phase2_ckpt_path, 'rb') as f: phase2_data = pickle.load(f)
            params = {'image_codec': phase1_data['params'], 'correction_net': phase2_data['params']}
            if 'opt_state' in phase2_data:
                 loaded_opt_state = phase2_data['opt_state']
                 print("--- Optimizer state found in checkpoint. ---")
            else:
                 print("[bold yellow]Warning: No optimizer state in checkpoint. Re-initializing.[/bold yellow]")
            start_epoch = phase2_data.get('epoch', 0) + 1
            if self.q_controller and 'q_controller_state' in phase2_data: self.q_controller.load_state_dict(phase2_data['q_controller_state']); print("--- Q-Controller state loaded. ---")
        else:
            print("--- Initializing new Phase 2 model for Auto-Regressive Training ---")
            with jax.default_device(CPU_DEVICE):
                dummy_latent = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 3))
                dummy_flow = jnp.zeros((1, self.args.latent_grid_size, self.args.latent_grid_size, 2))
                correction_net_params = self.model.init(jax.random.PRNGKey(0), dummy_latent, dummy_flow, method=lambda m, p, f: m.correction_net(p,f))['params']
            params = {'image_codec': phase1_data['params'], 'correction_net': correction_net_params['correction_net']}
            start_epoch = 0

        def partition_fn(path, param):
            if path and path[0] == 'correction_net': return 'trainable'
            return 'frozen'
        param_partitions = path_aware_map(partition_fn, params)
        final_optimizer = optax.multi_transform({'trainable': trainable_optimizer, 'frozen': optax.set_to_zero()}, param_partitions)
        state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=final_optimizer)

        if loaded_opt_state is not None:
            if jax.tree_util.tree_structure(state.opt_state) == jax.tree_util.tree_structure(loaded_opt_state):
                state = state.replace(opt_state=loaded_opt_state); print("--- Full partitioned optimizer state loaded successfully. ---")
            else:
                print("[bold yellow]Warning: Optimizer state structure mismatch. Re-initializing optimizer.[/bold yellow]")

        p_state = replicate(state)

        def calculate_flow_py(prvs_gray, nxt_gray):
            prvs_np = np.asarray(prvs_gray)
            nxt_np = np.asarray(nxt_gray)
            flow = cv2.calcOpticalFlowFarneback(prvs_np, nxt_np, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            return flow.astype(np.float32)

        @partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(2, 3, 4))
        def train_step_video(state, frames, clip_len, num_patch_pixels, loss_resolution, pmap_sampling_key):
            
            def loss_fn(params):
                @jax.custom_jvp
                def optical_flow_callback(prvs_gray, nxt_gray):
                    return jax.pure_callback(
                        calculate_flow_py,
                        jax.ShapeDtypeStruct((loss_resolution, loss_resolution, 2), jnp.float32),
                        prvs_gray, nxt_gray,
                        vmap_method='sequential'
                    )

                @optical_flow_callback.defjvp
                def optical_flow_callback_jvp(primals, tangents):
                    prvs_gray, nxt_gray = primals
                    primal_out = optical_flow_callback(prvs_gray, nxt_gray)
                    tangent_out = jnp.zeros_like(primal_out)
                    return primal_out, tangent_out

                def _single_example_loss_fn(frames_single, params, pmap_sampling_key_inner):
                    H, W = self.args.image_size, self.args.image_size
                    x_coords_full = jnp.linspace(-1, 1, H)
                    full_coords = jnp.stack(jnp.meshgrid(x_coords_full, x_coords_full, indexing='ij'), axis=-1).reshape(-1, 2)
                    
                    def rgb_to_grayscale_jax(rgb_image):
                        weights = jnp.array([0.299, 0.587, 0.114])
                        return jnp.dot(rgb_image, weights)
                    
                    patch_indices = jax.random.choice(pmap_sampling_key_inner, H * W, shape=(num_patch_pixels,), replace=False)
                    coords_patch = full_coords[patch_indices]

                    target_pixels_full = frames_single.reshape(clip_len, H * W, 3)
                    target_pixels_patch = target_pixels_full[:, patch_indices, :]

                    def warp_latents(single_latent, single_flow):
                        grid_size = single_latent.shape[0]
                        grid_y, grid_x = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size), indexing='ij'); coords = jnp.stack([grid_y, grid_x], axis=-1)
                        new_coords = jnp.reshape(coords + single_flow, (grid_size**2, 2)).T
                        warped_channels = [jax.scipy.ndimage.map_coordinates(single_latent[..., c], new_coords, order=1, mode='reflect').reshape(grid_size, grid_size) for c in range(3)]
                        return jnp.stack(warped_channels, axis=-1)
                    
                    frame0 = jnp.expand_dims(frames_single[0], axis=0)
                    p_recon0 = self.model.apply({'params': params}, frame0, method=lambda m, i: m.image_codec.modulator(i))
                    recon0_patch = self.model.apply({'params': params}, p_recon0, coords_patch, method=lambda m, p, c: m.image_codec.decode(p, c))
                    loss0 = jnp.mean(jnp.abs(target_pixels_patch[0] - recon0_patch))

                    def unroll_step(carry, xs):
                        p_prev_recon, recon_prev_low_res = carry
                        frame_target_t, target_patch_t = xs

                        prvs_gray = rgb_to_grayscale_jax(recon_prev_low_res)
                        target_t_low_res = jax.image.resize(frame_target_t, (loss_resolution, loss_resolution, 3), 'bilinear')
                        nxt_gray = rgb_to_grayscale_jax(target_t_low_res)
                        
                        flow = optical_flow_callback(prvs_gray, nxt_gray)
                        
                        flow_latent = jax.image.resize(flow, (self.args.latent_grid_size, self.args.latent_grid_size, 2), 'bilinear')
                        
                        p_warped = warp_latents(jnp.squeeze(p_prev_recon, axis=0), flow_latent)
                        p_warped_batched = jnp.expand_dims(p_warped, 0)
                        flow_latent_batched = jnp.expand_dims(flow_latent, 0)
                        
                        p_residual = self.model.apply({'params': params}, p_warped_batched, flow_latent_batched, method=lambda m, p, f: m.correction_net(p,f))
                        p_current_recon = p_warped_batched + p_residual
                        
                        recon_patch_t = self.model.apply({'params': params}, p_current_recon, coords_patch, method=lambda m, p, c: m.image_codec.decode(p, c))
                        loss_t = jnp.mean(jnp.abs(target_patch_t - recon_patch_t))

                        x_coords_low_res = jnp.linspace(-1, 1, loss_resolution)
                        low_res_coords = jnp.stack(jnp.meshgrid(x_coords_low_res, x_coords_low_res, indexing='ij'), axis=-1).reshape(-1, 2)
                        recon_current_low_res = self.model.apply({'params': params}, p_current_recon, low_res_coords, method=lambda m, p, c: m.image_codec.decode(p, c)).reshape(loss_resolution, loss_resolution, 3)
                        
                        return (p_current_recon, recon_current_low_res), loss_t

                    x_coords_low_res = jnp.linspace(-1, 1, loss_resolution)
                    low_res_coords = jnp.stack(jnp.meshgrid(x_coords_low_res, x_coords_low_res, indexing='ij'), axis=-1).reshape(-1, 2)
                    recon0_low_res = self.model.apply({'params': params}, p_recon0, low_res_coords, method=lambda m, p, c: m.image_codec.decode(p, c)).reshape(loss_resolution, loss_resolution, 3)

                    initial_carry = (p_recon0, recon0_low_res)
                    frames_rest = frames_single[1:]
                    target_patches_rest = target_pixels_patch[1:]

                    final_carry, p_frame_losses = jax.lax.scan(unroll_step, initial_carry, (frames_rest, target_patches_rest))
                    
                    total_loss = loss0 + jnp.mean(p_frame_losses)
                    return total_loss / clip_len, recon0_low_res
                
                device_keys = jax.random.split(pmap_sampling_key, frames.shape[0])
                batch_losses, recon_frames = jax.vmap(_single_example_loss_fn, in_axes=(0, None, 0))(frames, params, device_keys)
                return jnp.mean(batch_losses), recon_frames

            (loss, recon_frames), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, 'devices')
            return state.apply_gradients(grads=grads), jax.lax.pmean(loss, 'devices'), jax.lax.pmean(recon_frames, 'devices')
        
        # --- Main Training Loop ---
        dataset = create_video_dataset(self.args.data_dir, self.args.batch_size, self.args.clip_len)
        it = dataset.as_numpy_iterator()
        with open(Path(self.args.data_dir)/"dataset_info.pkl",'rb') as f: num_frames = pickle.load(f)['num_frames']
        steps_per_epoch = (num_frames // self.args.clip_len) // (self.args.batch_size * self.num_devices)

        print("--- Compiling JAX auto-regressive video function (vmap-scan)... ---")
        dummy_keys = jax.random.split(jax.random.PRNGKey(0), self.num_devices)
        p_state, _, _ = train_step_video(p_state, common_utils.shard(next(it)), self.args.clip_len, self.args.num_patch_pixels, self.args.loss_resolution, dummy_keys)
        print("--- Compilation complete. Starting auto-regressive training. ---")

        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Phase 2: Video Dynamics (Auto-Regressive)[/] | Base: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2)); layout["left"].split(Layout(name="stats"), Layout(name="preview"))
        preview_panel = Panel("...", title="[bold]🔎 Video Preview[/]", border_style="green", height=18); layout["left"]["preview"].update(preview_panel)
        progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn(), TextColumn("LR: {task.fields[lr]:.2e}"))
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch,total_epochs=self.args.epochs, lr=self.args.lr); layout['footer'].update(progress)
        epoch_for_save = start_epoch
        with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            try:
                for epoch in range(start_epoch, self.args.epochs):
                    epoch_for_save = epoch; progress.update(epoch_task, completed=0, epoch=epoch+1)
                    for step in range(steps_per_epoch):
                        if self.should_shutdown: break

                        if self.q_controller:
                            current_lr = self.q_controller.choose_action()
                            opt_state_unrep = unreplicate(p_state.opt_state)
                            opt_state_unrep.inner_states['trainable'].inner_state[-1].hyperparams['learning_rate'] = jnp.asarray(current_lr)
                            p_state = p_state.replace(opt_state=replicate(opt_state_unrep))
                        else: current_lr = self.args.lr

                        batch_np = next(it)
                        self.key, sampling_key = jax.random.split(self.key)
                        device_keys = jax.random.split(sampling_key, self.num_devices)
                        p_state, loss, recon_frames = train_step_video(p_state, common_utils.shard(batch_np), self.args.clip_len, self.args.num_patch_pixels, self.args.loss_resolution, device_keys)
                        
                        loss_val = unreplicate(loss)
                        if self.q_controller: self.q_controller.update_q_value(loss_val)

                        if step % 2 == 0:
                            self.recon_loss_history.append(loss_val)
                            stats_tbl=Table(show_header=False,box=None,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
                            stats_tbl.add_row("Video Loss",f"[bold green]{loss_val:.4f}[/]"); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util",f"[yellow]{util}[/]")
                            if self.q_controller:
                                status_word = self.q_controller.status.split(" ")[0]
                                status_map = {"IMPROVING": "bold green", "STAGNATED": "bold yellow", "REGRESSING": "bold red", "STARTING": "dim"}
                                color = status_map.get(status_word, "dim"); stats_tbl.add_row("Q-Ctrl Status", f"[{color}]{self.q_controller.status}[/]")
                            layout["left"]["stats"].update(Panel(stats_tbl,title="[bold]📊 Stats[/]",border_style="blue"))
                            spark_w = max(1, (live.console.width*2//3)-10)
                            recon_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.recon_loss_history,spark_w)}[/]"),title=f"Video Reconstruction Loss (L1)",height=3, border_style="cyan")
                            layout["right"].update(Panel(recon_panel,title="[bold]📉 Losses[/]",border_style="magenta"))
                        if step > 0 and step % 25 == 0:
                            # --- THE FIX IS HERE ---
                            # 1. Compare the same frame (the first one) for both
                            orig_frame = np.array((batch_np[0][0]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                            # 2. Use the correct indexing for the reconstructed frame
                            recon_frame_np = np.array((unreplicate(recon_frames)[0]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                            # --- END OF FIX ---
                            self._update_preview_panel(preview_panel, orig_frame, recon_frame_np)
                        progress.update(epoch_task, advance=1, lr=current_lr)
                    if self.should_shutdown: break
                    if jax.process_index() == 0:
                        state_to_save = unreplicate(p_state); data_to_save = {'params':jax.device_get(state_to_save.params['correction_net']), 'opt_state':jax.device_get(state_to_save.opt_state), 'epoch':epoch}
                        if self.q_controller: data_to_save['q_controller_state'] = self.q_controller.state_dict()
                        with open(phase2_ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
                        live.console.print(f"--- :floppy_disk: Phase 2 checkpoint saved for epoch {epoch+1} ---")
            except Exception as e: live.stop(); print("\n[bold red]FATAL: Training loop crashed![/bold red]"); raise e
        if jax.process_index() == 0 and 'p_state' in locals():
            print("\n--- Training finished or interrupted. Saving final state... ---")
            state_to_save = unreplicate(p_state); data_to_save = {'params':jax.device_get(state_to_save.params['correction_net']), 'opt_state':jax.device_get(state_to_save.opt_state), 'epoch':epoch_for_save}
            if self.q_controller: data_to_save['q_controller_state'] = self.q_controller.state_dict()
            with open(phase2_ckpt_path, 'wb') as f: pickle.dump(data_to_save, f)
            print("--- :floppy_disk: Final Phase 2 state saved. ---")




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
    def __init__(self, args):
        super().__init__(args)
        self.video_model = VideoCodecModel(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size)
        # --- FIX: Point to the new auto-regressive model file ---
        phase2_model_path = Path(f"{self.args.basename}_{self.args.d_model}d_512_video_autoregressive.pkl")
        if not phase2_model_path.exists(): print(f"[FATAL] Auto-regressive Phase 2 model not found at {phase2_model_path}. Run 'train-video' first."), sys.exit(1)
        
        print(f"--- Loading Auto-Regressive Phase 2 model from {phase2_model_path} ---")
        with open(phase2_model_path, 'rb') as f: phase2_params = pickle.load(f)['params']
        self.full_params = jax.device_put({'image_codec': self.params, 'correction_net': phase2_params})

        def warp_latents_standalone(single_latent, single_flow):
            H, W, _ = single_latent.shape
            grid_y, grid_x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'); coords = jnp.stack([grid_y, grid_x], axis=-1)
            new_coords = jnp.reshape(coords + single_flow, (H * W, 2)).T
            warped_channels = [jax.scipy.ndimage.map_coordinates(single_latent[..., c], new_coords, order=1, mode='reflect').reshape(H, W) for c in range(3)]
            return jnp.stack(warped_channels, axis=-1)
        self._vmapped_warp_fn = jax.jit(jax.vmap(warp_latents_standalone))

    @partial(jax.jit, static_argnames=('self',))
    def _process_p_frame_batch(self, p_initial_gpu, flow_batch_gpu, params):
        def unroll_step(p_prev, flow_current):
            flow_latent = jax.image.resize(jnp.expand_dims(flow_current, 0), (1, self.args.latent_grid_size, self.args.latent_grid_size, 2), 'bilinear')
            p_warped = self._vmapped_warp_fn(p_prev, flow_latent)
            residual = self.video_model.apply({'params': params}, p_warped, flow_latent, method=lambda m,p,f:m.correction_net(p,f))
            p_current = p_warped + residual
            return p_current, residual
        final_latent, residual_batch = jax.lax.scan(unroll_step, p_initial_gpu, flow_batch_gpu)
        return jnp.squeeze(residual_batch, axis=1), final_latent

    def _quantize_residuals(self, residual_grid_float):
        norm_grid = (residual_grid_float / 0.1) * 0.5 + 0.5; num_bins = 256
        return np.array(jnp.round(norm_grid * (num_bins - 1)), dtype=np.uint8)
    def _dequantize_residuals(self, residual_grid_uint8):
        num_bins=256; norm_grid = jnp.asarray(residual_grid_uint8, dtype=jnp.float32) / (num_bins - 1)
        return (norm_grid - 0.5) * 2.0 * 0.1

    def video_compress(self):
        batch_size = self.args.batch_size
        video_p, output_path = Path(self.args.video_path), Path(self.args.output_path)
        
        cap = cv2.VideoCapture(str(video_p))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or num_frames == 0:
            print(f"[FATAL] Could not open video file: {video_p}"); sys.exit(1)

        print(f"--- Compressing video to single file: {output_path} ---")

        ret, frame = cap.read()
        if not ret: print("[FATAL] Could not read first frame."); sys.exit(1)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512,512),Image.Resampling.LANCZOS)
        frame_np = (np.array(img,dtype=np.float32)/127.5)-1.0
        p0 = self._encode(jnp.expand_dims(frame_np, 0))
        quantized_iframe = self._quantize_latents(p0)
        
        p_prev_gpu = jax.device_put(p0)
        prvs_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        
        pbar = tqdm(total=num_frames-1, desc="Encoding P-Frames")
        frame_idx = 1
        all_quantized_p_frames = []

        while frame_idx < num_frames:
            frames_batch, flow_batch = [], []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret: break
                img_next = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512,512),Image.Resampling.LANCZOS)
                frames_batch.append(np.array(img_next))
            if not frames_batch: break

            current_prvs_gray = prvs_gray
            for next_frame_rgb in frames_batch:
                nxt_gray = cv2.cvtColor(next_frame_rgb, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(current_prvs_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_batch.append(flow)
                current_prvs_gray = nxt_gray
            
            residual_batch_gpu, p_final_gpu = self._process_p_frame_batch(p_prev_gpu, jax.device_put(np.stack(flow_batch)), self.full_params)
            
            for res_frame in np.array(residual_batch_gpu):
                all_quantized_p_frames.append(self._quantize_residuals(res_frame))

            p_prev_gpu = p_final_gpu; prvs_gray = current_prvs_gray
            pbar.update(len(frames_batch)); frame_idx += len(frames_batch)
            
        pbar.close(); cap.release()

        with open(output_path, 'wb') as f:
            header = struct.pack('<HII', self.args.latent_grid_size, quantized_iframe.nbytes, len(all_quantized_p_frames))
            f.write(header)
            f.write(quantized_iframe.tobytes())
            if all_quantized_p_frames:
                f.write(np.stack(all_quantized_p_frames).tobytes())
        
        original_size = video_p.stat().st_size
        compressed_size = output_path.stat().st_size
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        print(f"✅ Video compressed to {output_path} ({compressed_size/1024**2:.2f} MB). Ratio: {ratio:.2f}x")

    def video_decompress(self):
        input_path, output_path = Path(self.args.input_path), Path(self.args.output_path)
        print(f"--- Decompressing single file (Auto-Regressive): {input_path} ---")

        with open(input_path, 'rb') as f:
            header_bytes = f.read(10)
            latent_grid_size, i_frame_data_length, num_p_frames = struct.unpack('<HII', header_bytes)
            iframe_bytes = f.read(i_frame_data_length)
            quantized_iframe = np.frombuffer(iframe_bytes, dtype=np.uint8).reshape((latent_grid_size, latent_grid_size, 3))
            p_current = jax.device_put(jnp.expand_dims(self._dequantize_latents(quantized_iframe), 0))
            p_frame_bytes_per_frame = latent_grid_size * latent_grid_size * 3
            all_p_frame_bytes = f.read(num_p_frames * p_frame_bytes_per_frame)
            quantized_pframes = np.frombuffer(all_p_frame_bytes, dtype=np.uint8).reshape((num_p_frames, latent_grid_size, latent_grid_size, 3))

        writer = imageio.get_writer(output_path, fps=30)
        
        frame_recon_np = self._decode_and_convert(p_current)
        writer.append_data(frame_recon_np)
        prvs_gray = cv2.cvtColor(frame_recon_np, cv2.COLOR_RGB2GRAY)

        @jax.jit
        def clamp_latents(p):
            delta, chi, radius = p[..., 0], p[..., 1], p[..., 2]
            delta_c = jnp.clip(delta, -jnp.pi, jnp.pi)
            chi_c = jnp.clip(chi, -jnp.pi / 4.0, jnp.pi / 4.0)
            radius_c = jnp.clip(radius, 0, jnp.pi / 2.0)
            return jnp.stack([delta_c, chi_c, radius_c], axis=-1)

        for i in tqdm(range(num_p_frames), desc="Decoding P-Frames"):
            residual = jax.device_put(jnp.expand_dims(self._dequantize_residuals(quantized_pframes[i]), 0))
            
            # --- AUTO-REGRESSIVE DECODING LOGIC ---
            # 1. Decode the current frame to get the `next` gray image for flow calculation
            current_recon_np = self._decode_and_convert(p_current)
            nxt_gray = cv2.cvtColor(current_recon_np, cv2.COLOR_RGB2GRAY)

            # 2. Calculate flow between the previous decoded frame and the current one
            flow = cv2.calcOpticalFlowFarneback(prvs_gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_gpu = jax.device_put(jnp.expand_dims(flow, 0))
            flow_latent = jax.image.resize(flow_gpu, (1, self.args.latent_grid_size, self.args.latent_grid_size, 2), 'bilinear')
            
            # 3. Warp the previous latent, apply residual, and clamp
            p_warped = self._vmapped_warp_fn(p_current, flow_latent)
            p_current = p_warped + residual
            p_current = clamp_latents(p_current)
            
            # 4. Decode the final, stabilized latent for this frame and write to video
            final_frame_np = self._decode_and_convert(p_current)
            writer.append_data(final_frame_np)

            # 5. Update the `previous` gray frame for the next iteration's flow calculation
            prvs_gray = cv2.cvtColor(final_frame_np, cv2.COLOR_RGB2GRAY)
            # --- END OF LOGIC ---
            
        writer.close(); print(f"✅ Video decompressed to {output_path}")

    def _decode_and_convert(self, latent_batch):
        recon = self._decode_batched(latent_batch)
        return np.array((recon[0]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)


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
        dataset = create_dataset(self.args.image_dir, self.args.batch_size, is_training=False)
        all_latents, all_clip_features = [], []
        for batch_np in tqdm(dataset.as_numpy_iterator(), desc="Encoding Images"):
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
    p_train.add_argument('--image-dir', type=str, required=True); p_train.add_argument('--epochs', type=int, default=100); p_train.add_argument('--batch-size', type=int, default=4, help="Batch size PER DEVICE.")
    p_train.add_argument('--lr', type=float, default=2e-4); p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--use-q-controller', action='store_true', help="Enable adaptive LR via Q-Learning."); p_train.add_argument('--use-sentinel', action='store_true', help="Enable Sentinel optimizer to dampen oscillations.")
    p_train.add_argument('--finetune', action='store_true', help="Use finetuning Q-Controller config (not typically used for phase 1).")

    p_prep_vid = subparsers.add_parser("prepare-video-data", help="Extract frames and optical flow from a video.")
    p_prep_vid.add_argument('--video-path', type=str, required=True); p_prep_vid.add_argument('--data-dir', type=str, required=True)
    p_train_vid = subparsers.add_parser("train-video", help="PHASE 2: Train the video dynamics model with advanced tools.", parents=[parent_parser])
    p_train_vid.add_argument('--data-dir', type=str, required=True); p_train_vid.add_argument('--epochs', type=int, default=200); p_train_vid.add_argument('--batch-size', type=int, default=2, help="Batch size PER DEVICE for video clips.")
    p_train_vid.add_argument('--lr', type=float, default=1e-4); p_train_vid.add_argument('--seed', type=int, default=42); p_train_vid.add_argument('--clip-len', type=int, default=8)
    p_train_vid.add_argument('--use-q-controller', action='store_true', help="Enable adaptive LR via Q-Learning."); p_train_vid.add_argument('--use-sentinel', action='store_true', help="Enable Sentinel optimizer to dampen oscillations.")
    p_train_vid.add_argument('--finetune', action='store_true', help="Use finetuning Q-Controller config.")
    p_train_vid.add_argument('--loss-resolution', type=int, default=256, help="Intermediate resolution for optical flow calculation during training (e.g., 128, 256).")
    p_train_vid.add_argument('--num-patch-pixels', type=int, default=8192, help="Number of pixels to sample for patched loss calculation.")


    p_comp = subparsers.add_parser("compress", help="Compress a single image to a file.", parents=[parent_parser]); p_comp.add_argument('--image-path', type=str, required=True); p_comp.add_argument('--output-path', type=str, required=True)
    p_dcomp = subparsers.add_parser("decompress", help="Decompress a file to an image.", parents=[parent_parser]); p_dcomp.add_argument('--compressed-path', type=str, required=True); p_dcomp.add_argument('--output-path', type=str, required=True)
    
    # --- UPDATED VIDEO COMPRESSION/DECOMPRESSION ARGUMENTS ---
    p_vcomp = subparsers.add_parser("video-compress", help="Compress a video to a single efficient file.", parents=[parent_parser])
    p_vcomp.add_argument('--video-path', type=str, required=True)
    p_vcomp.add_argument('--output-path', type=str, required=True, help="Path to the output .wubu file.")
    p_vcomp.add_argument('--batch-size', type=int, default=32, help="Number of frames to process in a batch for faster compression.")

    p_vdcomp = subparsers.add_parser("video-decompress", help="Decompress a custom video file.", parents=[parent_parser])
    p_vdcomp.add_argument('--input-path', type=str, required=True, help="Path to the compressed .wubu file.")
    p_vdcomp.add_argument('--output-path', type=str, required=True, help="Path for the output video (e.g., video.mp4).")
    # --- END OF UPDATES ---

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
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"), sys.exit(0)
