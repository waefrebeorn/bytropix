import os
import sys
import argparse
import signal
import platform
import threading
import time
import random
import pickle
from pathlib import Path
from collections import deque
from functools import partial
from typing import Any, Dict, Optional, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.ndimage
from flax import linen as nn
from flax.training import train_state, common_utils
from flax.jax_utils import replicate, unreplicate
import optax
import numpy as np
import torch
from PIL import Image

# Conditional imports for keyboard listening
if platform.system() == "Windows":
    import msvcrt
else:
    import tty, termios, select

try:
    import chex
except ImportError:
    print("[FATAL] Missing dependency 'chex'. Please run: pip install chex"), sys.exit(1)

# --- JAX Configuration & Dependency Checks ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_threefry_partitionable', True)
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.console import Console
    import pynvml
    pynvml.nvmlInit()
    from tqdm import tqdm
    import clip
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    print("[FATAL] A required dependency is missing. Please install: tensorflow, rich, pynvml, tqdm, ftfy, clip"), sys.exit(1)

# =================================================================================================
# Optimizer and Foundational Model Components
# =================================================================================================
DTYPE = jnp.float32
class SentinelState(NamedTuple):
    sign_history: chex.ArrayTree; dampened_count: Optional[jnp.ndarray] = None; dampened_pct: Optional[jnp.ndarray] = None
def sentinel(dampening_factor: float = 0.1, history_len: int = 5, oscillation_threshold: int = 3) -> optax.GradientTransformation:
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
class ImageEncoder(nn.Module):
    d_model: int; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, (4, 4), (2, 2))(images); x = nn.gelu(x); x = nn.LayerNorm()(x)
        x = nn.Conv(64, (4, 4), (2, 2))(x); x = nn.gelu(x); x = nn.LayerNorm()(x)
        x = nn.Conv(128, (3, 3), padding='SAME')(x); x = nn.gelu(x); x = nn.LayerNorm()(x)
        return nn.Dense(self.d_model)(jnp.mean(x, axis=(1, 2)))
class TextEncoder(nn.Module):
    d_model: int; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, text_embeds: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.d_model * 2)(text_embeds); h = nn.gelu(h); h = nn.LayerNorm()(h)
        return nn.Dense(self.d_model)(h)
class LatentMapDecoder(nn.Module):
    d_model: int; map_size: int = 16; map_channels: int = 64; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(512)(z); h = nn.gelu(h)
        h = nn.Dense(self.map_size * self.map_size * self.map_channels)(h)
        return h.reshape(-1, self.map_size, self.map_size, self.map_channels)
class MathematicalPerceptualLoss(nn.Module):
    @nn.compact
    def __call__(self, gen_img, real_img, style_weight=1e6, fft_weight=1.0, color_weight=1.0):
        def gram_matrix(x): b, h, w, c = x.shape; x = x.reshape((b, h * w, c)); return jnp.einsum('bic,bjc->bij', x, x) / (h * w)
        def color_loss(x1, x2): m1, s1 = jnp.mean(x1, (1,2)), jnp.std(x1, (1,2)); m2, s2 = jnp.mean(x2, (1,2)), jnp.std(x2, (1,2)); return jnp.mean(jnp.abs(m1-m2)) + jnp.mean(jnp.abs(s1-s2))
        def fft_loss(x1, x2): return jnp.mean(jnp.abs(jnp.abs(jnp.fft.fftn(x1,(1,2))) - jnp.abs(jnp.fft.fftn(x2,(1,2)))))
        return (style_weight * jnp.mean(jnp.abs(gram_matrix(gen_img) - gram_matrix(real_img))) + color_weight * color_loss(gen_img, real_img) + fft_weight * fft_loss(gen_img, real_img))

# =================================================================================================
# The Wubu-Poincaré Architecture (Corrected Implementation)
# =================================================================================================
class PositionalEncoding(nn.Module):
    num_freqs: int
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs) * jnp.pi
        return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)
class PoincareCoordinateDecoder(nn.Module):
    d_model: int; map_channels: int; num_freqs: int = 12; mlp_width: int = 256; mlp_depth: int = 4
    
    def setup(self):
        mlp_layers = []
        input_dim = (2 * self.num_freqs + 1) * 2 + self.d_model + self.map_channels
        for i in range(self.mlp_depth):
            mlp_layers.append(nn.Dense(self.mlp_width, name=f"mlp_{i}"))
            mlp_layers.append(nn.gelu)
        mlp_layers.append(nn.Dense(3, name="mlp_out"))
        self.coordinate_mlp = nn.Sequential(mlp_layers)
        self.pos_encoder = PositionalEncoding(self.num_freqs)
        
    def __call__(self, z: jnp.ndarray, z_grid: jnp.ndarray, resolution: int) -> jnp.ndarray:
        B = z.shape[0]
        x = jnp.linspace(-1, 1, resolution)
        y = jnp.linspace(-1, 1, resolution)
        grid_x, grid_y = jnp.meshgrid(x, y, indexing='ij')
        coords = jnp.stack([grid_x, grid_y], axis=-1).reshape(resolution * resolution, 2)
        coords_rescaled = (coords + 1) / 2 * (z_grid.shape[1] - 1)
        z_grid_for_sampling = z_grid.transpose(0, 3, 1, 2)
        def sample_one_channel(grid_2d, coords_2d): 
            return jax.scipy.ndimage.map_coordinates(grid_2d, coords_2d.T, order=1, mode='reflect')
        sampled_features_batched = jax.vmap(jax.vmap(sample_one_channel, in_axes=(0, None)), in_axes=(0, None))(
            z_grid_for_sampling, coords_rescaled
        )
        sampled_features = sampled_features_batched.transpose(0, 2, 1)
        encoded_coords = self.pos_encoder(coords)
        encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)
        z_tiled = jnp.repeat(z[:, None, :], resolution * resolution, axis=1)
        mlp_input = jnp.concatenate([encoded_coords_tiled, z_tiled, sampled_features], axis=-1)
        output_pixels = self.coordinate_mlp(mlp_input)
        output_image = output_pixels.reshape(B, resolution, resolution, 3)
        return nn.tanh(output_image)

class Chimera(nn.Module):
    d_model: int; map_channels: int = 64; resolutions: tuple = (64, 128, 256, 512); dtype: Any = DTYPE
    def setup(self):
        self.image_encoder = ImageEncoder(d_model=self.d_model)
        self.text_encoder = TextEncoder(d_model=self.d_model)
        self.latent_map_decoder = LatentMapDecoder(d_model=self.d_model, map_channels=self.map_channels)
        self.coord_decoder = PoincareCoordinateDecoder(d_model=self.d_model, map_channels=self.map_channels, name='poincare_decoder')
        self.perceptual_loss_fn = MathematicalPerceptualLoss()

    def __call__(self, batch):
        z_image = self.image_encoder(batch['images'][64])
        z_text = self.text_encoder(batch['clip_text_embeddings'])
        z_grid = self.latent_map_decoder(z_image)
        recons = {res: self.coord_decoder(z_image, z_grid, res) for res in self.resolutions}
        return recons, z_image, z_text

    def encode_text_only(self, text_embeds):
        return self.text_encoder(text_embeds)

    def decode_from_z(self, z, resolution=512):
        z_grid = self.latent_map_decoder(z)
        return self.coord_decoder(z, z_grid, resolution)

# =================================================================================================
# Training and Generation Infrastructure (RESTRUCTURED)
# =================================================================================================

# --- STEP 1: A PMAPPED function for the SHARED ENCODERS ---
@partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(2, 3))
def p_shared_step(state, batch, model_apply_fn, w_align):
    def loss_fn(params):
        z_image = model_apply_fn({'params': params}, batch['images'][64], method=lambda m, x: m.image_encoder(x))
        z_text = model_apply_fn({'params': params}, batch['clip_text_embeddings'], method=lambda m, x: m.text_encoder(x))
        loss_align = jnp.mean((z_image - z_text)**2)
        loss_reg = 1e-4 * (jnp.mean(z_image**2) + jnp.mean(z_text**2))
        total_loss = w_align * loss_align + loss_reg
        return total_loss, (z_image, z_text, loss_align)

    (loss, (z_image, z_text, align_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    grads = jax.lax.pmean(grads, axis_name='devices')
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {'total_loss': loss, 'align_loss': align_loss}
    return new_state, z_image, jax.lax.pmean(metrics, axis_name='devices')


# --- STEP 2: A PMAPPED function for a SINGLE GENERATOR update ---
@partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(4, 5, 6), donate_argnums=(0,))
def p_generator_step(state, batch, z_image, dropout_key, resolution, model_apply_fn, w_perceptual):
    target_image = batch['images'][resolution]
    z_image_detached = jax.lax.stop_gradient(z_image)

    def generator_loss_fn(params):
        z_grid = model_apply_fn({'params': params}, z_image_detached, method=lambda m, x: m.latent_map_decoder(x))
        recon = model_apply_fn({'params': params}, z_image_detached, z_grid, resolution, method=lambda m, z, zg, res: m.coord_decoder(z, zg, res))
        
        B, H, W, C = target_image.shape
        patch_size = 64
        
        if H > patch_size:
            def random_crop(key, img):
                corner_x = jax.random.randint(key, (), 0, W - patch_size + 1)
                corner_y = jax.random.randint(key, (), 0, H - patch_size + 1)
                return jax.lax.dynamic_slice(img, (corner_y, corner_x, 0), (patch_size, patch_size, C))

            batch_keys = jax.random.split(dropout_key, B)
            target_patches = jax.vmap(random_crop)(batch_keys, target_image)
            recon_patches = jax.vmap(random_crop)(batch_keys, recon)
            target, pred = target_patches, recon_patches
        else:
            target, pred = target_image, recon

        loss = jnp.mean(jnp.abs(target - pred))
        perceptual_loss = 0.0
        if resolution == 512:
            perceptual_loss = model_apply_fn({'params': params}, pred, target, method=lambda m, g, r: m.perceptual_loss_fn(g, r))
            loss += w_perceptual * perceptual_loss
            
        aux = {'recon': recon, f'pixel_loss_{resolution}': jnp.mean(jnp.abs(target_image - recon)), 'perceptual_loss': perceptual_loss}
        return loss, aux

    (loss, aux_outputs), grads = jax.value_and_grad(generator_loss_fn, has_aux=True)(state.params)
    
    grads = jax.lax.pmean(grads, axis_name='devices')
    new_state = state.apply_gradients(grads=grads)
    
    metrics = {
        'total_loss': loss,
        f'pixel_loss_{resolution}': aux_outputs[f'pixel_loss_{resolution}'],
        'perceptual_loss': aux_outputs['perceptual_loss'] if resolution == 512 else 0.0
    }
    return new_state, jax.lax.pmean(metrics, axis_name='devices')

def prepare_data(image_dir: str):
    base_path = Path(image_dir); resolutions = [64, 128, 256, 512]
    record_files = {res: base_path / f"data_{res}x{res}.tfrecord" for res in resolutions}
    text_emb_file = base_path / "clip_text_embeddings.pkl"; info_file = base_path / "dataset_info.pkl"
    if all(f.exists() for f in record_files.values()) and text_emb_file.exists() and info_file.exists(): print(f"✅ All necessary data files exist in {image_dir}. Skipping."); return
    print(f"--- Preparing multi-resolution data from {image_dir} ---")
    image_paths = sorted([p for p in base_path.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
    text_paths = [p.with_suffix('.txt') for p in image_paths]; valid_pairs = [(img, txt) for img, txt in zip(image_paths, text_paths) if txt.exists()]
    if not valid_pairs: print(f"[FATAL] No matching image/text pairs found in {image_dir}."), sys.exit(1)
    image_paths, text_paths = zip(*valid_pairs); print(f"Found {len(image_paths)} matching image-text pairs. Processing...")
    writers = {res: tf.io.TFRecordWriter(str(f)) for res, f in record_files.items()}
    for img_path in tqdm(image_paths, desc="Writing TFRecords"):
        try:
            img = Image.open(img_path).convert("RGB")
            for res, writer in writers.items():
                img_resized = img.resize((res, res), Image.Resampling.LANCZOS)
                img_bytes = tf.io.encode_jpeg(np.array(img_resized), quality=98).numpy()
                ex = tf.train.Example(features=tf.train.Features(feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
                writer.write(ex.SerializeToString())
        except Exception as e: print(f"\n[Warning] Skipping {img_path} due to error: {e}"); continue
    for writer in writers.values(): writer.close()
    clip_model, _ = clip.load("ViT-B/32", device=_clip_device); all_clip_text_embs = []
    for i in tqdm(range(0, len(text_paths), 256), desc="Processing Text Batches"):
        captions = [p.read_text().strip() for p in text_paths[i:i+256]]
        text_tokens = clip.tokenize(captions, truncate=True).to(_clip_device)
        with torch.no_grad(): all_clip_text_embs.append(clip_model.encode_text(text_tokens).cpu().numpy())
    with open(text_emb_file, 'wb') as f: pickle.dump(np.concatenate(all_clip_text_embs).astype(np.float32), f)
    with open(info_file, 'wb') as f: pickle.dump({'num_samples': len(image_paths)}, f)
    print(f"✅ Multi-resolution data preparation complete.")
def create_dataset(image_dir: str, batch_size: int):
    base_path = Path(image_dir); resolutions = [64, 128, 256, 512]
    record_files = {res: base_path / f"data_{res}x{res}.tfrecord" for res in resolutions}
    text_emb_file = base_path / "clip_text_embeddings.pkl"
    if not all(f.exists() for f in record_files.values()) or not text_emb_file.exists(): raise FileNotFoundError("Dataset files not found. Run 'prepare-data' first.")
    with open(text_emb_file, 'rb') as f: clip_text_embs = pickle.load(f)
    def _parse(proto, res):
        p = tf.io.parse_single_example(proto, {'image': tf.io.FixedLenFeature([], tf.string)})
        img = (tf.cast(tf.io.decode_jpeg(p['image'], 3), tf.float32) / 127.5) - 1.0
        img.set_shape([res, res, 3]); return img
    img_datasets = {res: tf.data.TFRecordDataset(str(f)).map(partial(_parse, res=res), num_parallel_calls=tf.data.AUTOTUNE) for res, f in record_files.items()}
    text_emb_ds = tf.data.Dataset.from_tensor_slices(clip_text_embs)
    full_ds = tf.data.Dataset.zip(tuple(img_datasets[res] for res in resolutions) + (text_emb_ds,))
    def to_dict(*args):
        imgs = args[:-1]; txt = args[-1]
        return {'images': {res: img for res, img in zip(resolutions, imgs)}, 'clip_text_embeddings': txt}
    return full_ds.shuffle(1024).repeat().batch(batch_size, drop_remainder=True).map(to_dict).prefetch(tf.data.AUTOTUNE)
Q_CONTROLLER_CONFIG_NORMAL = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.7, 0.9, 1.0, 1.1, 1.3], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-7, "lr_max": 5e-1, "metric_history_len": 5000, "loss_min": 0.05, "loss_max": 1.5, "exploration_rate_q": 0.3, "min_exploration_rate": 0.05, "exploration_decay": 0.9995, "trend_window": 777, "improve_threshold": 1e-5, "regress_threshold": 1e-6, "regress_penalty": 10.0, "stagnation_penalty": -2.0}
Q_CONTROLLER_CONFIG_FINETUNE = {"q_table_size": 100, "num_lr_actions": 5, "lr_change_factors": [0.8, 0.95, 1.0, 1.05, 1.2], "learning_rate_q": 0.1, "discount_factor_q": 0.9, "lr_min": 1e-8, "lr_max": 1e-2, "metric_history_len": 5000, "loss_min": 0.0, "loss_max": 0.5, "exploration_rate_q": 0.15, "min_exploration_rate": 0.02, "exploration_decay": 0.9998, "target_pixel_loss": 0.01}
class JaxHakmemQController:
    def __init__(self,initial_lr:float,config:Dict[str,Any], is_finetune:bool=False):
        self.config=config; self.current_lr=initial_lr; self.is_finetune = is_finetune; self.q_table_size=int(self.config["q_table_size"]); self.num_actions=int(self.config["num_lr_actions"]); self.lr_change_factors=self.config["lr_change_factors"]; self.q_table=np.zeros((self.q_table_size,self.num_actions),dtype=np.float32); self.learning_rate_q=float(self.config["learning_rate_q"]); self.discount_factor_q=float(self.config["discount_factor_q"]); self.lr_min=float(self.config["lr_min"]); self.lr_max=float(self.config["lr_max"]); self.loss_history=deque(maxlen=int(self.config["metric_history_len"])); self.loss_min=float(self.config["loss_min"]); self.loss_max=float(self.config["loss_max"]); self.last_action_idx:Optional[int]=None; self.last_state_idx:Optional[int]=None; self.initial_exploration_rate = float(self.config["exploration_rate_q"]); self.exploration_rate_q = self.initial_exploration_rate; self.min_exploration_rate = float(self.config["min_exploration_rate"]); self.exploration_decay = float(self.config["exploration_decay"]); self.status: str = "STARTING"; self.last_reward: float = 0.0
        if self.is_finetune: self.target_pixel_loss = float(config["target_pixel_loss"]); self.last_pixel_loss:Optional[float] = None; print(f"--- Q-Controller initialized in GOAL-SEEKING mode. Target: {self.target_pixel_loss} ---")
        else: self.trend_window = int(config["trend_window"]); self.pixel_loss_trend_history = deque(maxlen=self.trend_window); self.improve_threshold = float(config["improve_threshold"]); self.regress_threshold = float(config["regress_threshold"]); self.regress_penalty = float(config["regress_penalty"]); self.stagnation_penalty = float(config["stagnation_penalty"]); self.last_slope: float = 0.0; print(f"--- Q-Controller initialized in 3-STATE SEARCH mode. Trend Window: {self.trend_window} steps ---")
    def _discretize_value(self,value:float) -> int:
        if value<=self.loss_min: return 0
        if value>=self.loss_max: return self.q_table_size-1
        bin_size=(self.loss_max-self.loss_min)/self.q_table_size; return min(int((value-self.loss_min)/bin_size),self.q_table_size-1)
    def _get_current_state_idx(self) -> int:
        if not self.loss_history: return self.q_table_size//2
        avg_loss=np.mean(list(self.loss_history)[-5:]); return self._discretize_value(avg_loss)
    def choose_action(self) -> float:
        self.last_state_idx=self._get_current_state_idx()
        if random.random()<self.exploration_rate_q: self.last_action_idx=random.randint(0,self.num_actions-1)
        else: self.last_action_idx=np.argmax(self.q_table[self.last_state_idx]).item()
        change_factor=self.lr_change_factors[self.last_action_idx]; self.current_lr=np.clip(self.current_lr*change_factor,self.lr_min,self.lr_max)
        return self.current_lr
    def update_q_value(self, total_loss:float, pixel_loss_64:float):
        self.loss_history.append(total_loss)
        if self.last_state_idx is None or self.last_action_idx is None: return
        reward = self._calculate_reward(pixel_loss_64); self.last_reward = reward
        current_q = self.q_table[self.last_state_idx, self.last_action_idx]; next_state_idx = self._get_current_state_idx(); max_next_q = np.max(self.q_table[next_state_idx])
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor_q * max_next_q - current_q); self.q_table[self.last_state_idx, self.last_action_idx] = new_q
    def _calculate_reward(self, pixel_loss_64):
        if self.is_finetune:
            if self.last_pixel_loss is None: self.last_pixel_loss = pixel_loss_64; return 0.0
            reward = 1.0 / (abs(pixel_loss_64 - self.target_pixel_loss) + 1e-5)
            if pixel_loss_64 > self.last_pixel_loss: reward -= 50.0; self.status = "REGRESSING"
            else: self.status = "CONVERGING"
            self.last_pixel_loss = pixel_loss_64; return reward
        else:
            self.pixel_loss_trend_history.append(pixel_loss_64)
            if len(self.pixel_loss_trend_history) < self.trend_window: return 0.0
            loss_window = np.array(self.pixel_loss_trend_history); slope = np.polyfit(np.arange(self.trend_window), loss_window, 1)[0]; self.last_slope = slope
            if slope < -self.improve_threshold: self.status = "IMPROVING"; reward = abs(slope) * 1000; self.exploration_rate_q = max(self.min_exploration_rate, self.exploration_rate_q * self.exploration_decay)
            elif slope > self.regress_threshold: self.status = "REGRESSING"; reward = -abs(slope) * 1000 - self.regress_penalty; self.exploration_rate_q = self.initial_exploration_rate
            else: self.status = "STAGNATED"; reward = self.stagnation_penalty; self.exploration_rate_q = self.initial_exploration_rate / 2
            return reward
    def state_dict(self)->Dict[str,Any]:
        state = {"current_lr":self.current_lr,"q_table":self.q_table.tolist(),"loss_history":list(self.loss_history), "exploration_rate_q":self.exploration_rate_q}
        if self.is_finetune: state["last_pixel_loss"] = self.last_pixel_loss
        else: state["pixel_loss_trend_history"] = list(self.pixel_loss_trend_history)
        return state
    def load_state_dict(self,state_dict:Dict[str,Any]):
        self.current_lr=state_dict.get("current_lr",self.current_lr); self.q_table=np.array(state_dict.get("q_table",self.q_table.tolist()),dtype=np.float32); self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.loss_history.maxlen); self.exploration_rate_q=state_dict.get("exploration_rate_q", self.initial_exploration_rate)
        if self.is_finetune: self.last_pixel_loss=state_dict.get("last_pixel_loss", None)
        else: self.pixel_loss_trend_history=deque(state_dict.get("pixel_loss_trend_history",[]),maxlen=self.trend_window)

class Trainer:
    def __init__(self, args):
        self.args = args; self.num_devices = jax.local_device_count(); self.should_shutdown=False; signal.signal(signal.SIGINT, self.request_shutdown); self.key = jax.random.PRNGKey(args.seed)
        self.metric_histories = {f'pixel_loss_{res}': deque(maxlen=200) for res in [64, 128, 256, 512]}
        self.metric_histories.update({'align_loss': deque(maxlen=200), 'dampened_pct': deque(maxlen=200), 'perceptual_loss': deque(maxlen=200)})
        self.model = Chimera(d_model=args.d_model, dtype=DTYPE)
        if self.args.use_q_controller:
            q_config = Q_CONTROLLER_CONFIG_FINETUNE if args.finetune else Q_CONTROLLER_CONFIG_NORMAL
            self.q_controller = JaxHakmemQController(initial_lr=self.args.lr, config=q_config, is_finetune=args.finetune); threading.Thread(target=self._listen_for_boost, daemon=True).start()
        else: self.q_controller = None
    def _get_char_non_blocking(self):
        if platform.system() == "Windows":
            if msvcrt.kbhit(): return msvcrt.getch().decode('utf-8', errors='ignore')
        else:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                fd = sys.stdin.fileno(); old_settings = termios.tcgetattr(fd)
                try: tty.setraw(sys.stdin.fileno()); return sys.stdin.read(1)
                finally: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    def _listen_for_boost(self):
        while not self.should_shutdown:
            try:
                char = self._get_char_non_blocking();
                if char and char.lower() == 'b': self.boost_requested = True
            except Exception: pass
            time.sleep(0.05)
    def request_shutdown(self, signum, frame): print("\n--- Shutdown requested. Saving after this step... ---"); self.should_shutdown = True
    def _save_checkpoint(self, p_state, epoch, ckpt_path):
        data = {'params': jax.device_get(unreplicate(p_state.params)), 'opt_state': jax.device_get(unreplicate(p_state.opt_state)), 'epoch': epoch }
        if self.q_controller: data['q_controller_state'] = self.q_controller.state_dict()
        with open(ckpt_path, 'wb') as f: pickle.dump(data, f)
    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except: return "N/A", "N/A"
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array([val for val in data if np.isfinite(val)])
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v: return "".join([s[0]]*len(hist))
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1); return "".join(s[i] for i in indices)
    def train(self):
        ckpt_path = Path(f"{self.args.basename}_{self.args.d_model}d_wubu.pkl")
        components = [optax.clip_by_global_norm(1.0)]
        if self.args.use_sentinel: components.append(sentinel())
        tx = optax.chain(*components)
        optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.args.lr)
        full_optimizer = optax.chain(tx, optimizer)

        w_align = 1.0 if not self.args.finetune else 0.01

        if ckpt_path.exists():
            print(f"--- Resuming from checkpoint: {ckpt_path} ---")
            with open(ckpt_path, 'rb') as f: data = pickle.load(f)
            params = data['params']; state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=full_optimizer)
            if 'opt_state' in data and jax.tree_util.tree_structure(state.opt_state)==jax.tree_util.tree_structure(data['opt_state']):
                state=state.replace(opt_state=data['opt_state']); print("--- Optimizer state loaded. ---")
            else: print("[bold yellow]Warning: Optimizer state mismatch. Re-initializing.[/bold yellow]")
            start_epoch = data.get('epoch', -1) + 1
            if self.q_controller and 'q_controller_state' in data: self.q_controller.load_state_dict(data['q_controller_state'])
        else:
            print("--- Initializing new model from scratch... ---")
            init_key = jax.random.PRNGKey(self.args.seed)
            dummy_batch = {'images': {res: jnp.zeros((1, res, res, 3)) for res in [64,128,256,512]}, 'clip_text_embeddings': jnp.zeros((1, 512))}
            params = self.model.init({'params': init_key}, dummy_batch)['params']
            state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=full_optimizer); start_epoch=0
            print("--- Initialization complete. ---")

        p_state = replicate(state)
        dataset = create_dataset(self.args.image_dir, self.args.batch_size * self.num_devices)
        it = dataset.as_numpy_iterator()
        with open(Path(self.args.image_dir)/"dataset_info.pkl",'rb') as f: num_samples = pickle.load(f)['num_samples']
        steps_per_epoch = num_samples // (self.args.batch_size * self.num_devices)
        
        print(">>> Compiling training steps on GPU... This will be fast. <<<")
        jit_batch = next(it)
        self.key, shared_key, *gen_keys = jax.random.split(self.key, 2 + len(self.model.resolutions))
        
        print("--- Compiling shared step... ---")
        p_state_after_shared, z_image_for_compile, _ = p_shared_step(p_state, common_utils.shard(jit_batch), self.model.apply, w_align)
        
        print("--- Compiling generator steps sequentially... ---")
        # METICULOUS FIX: We must thread the state through the compilation loop to respect donation.
        p_state_for_gen_compile = p_state_after_shared
        for i, res in enumerate(self.model.resolutions):
            print(f"Compiling for {res}x{res}...")
            p_gen_key = jax.random.split(gen_keys[i], self.num_devices)
            # Pass the valid state, and capture the new valid state for the next iteration.
            p_state_for_gen_compile, _ = p_generator_step(p_state_for_gen_compile, common_utils.shard(jit_batch), z_image_for_compile, p_gen_key, res, self.model.apply, self.args.perceptual_weight)

        print(">>> Compilation successful! Starting training. <<<")
        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Project Chimera (Wubu-Poincaré Architecture)[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2)); progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn(), TextColumn("LR: {task.fields[lr]:.2e}"))
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs, lr=0.0); layout['footer'].update(progress)
        epoch_for_save = start_epoch
        try:
            with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                for epoch in range(start_epoch, self.args.epochs):
                    epoch_for_save = epoch
                    progress.update(epoch_task, completed=0, epoch=epoch + 1)
                    for step in range(steps_per_epoch):
                        if self.should_shutdown: break
                        
                        batch = common_utils.shard(next(it))
                        self.key, shared_key, *gen_keys = jax.random.split(self.key, 2 + len(self.model.resolutions))
                        
                        current_lr = self.q_controller.choose_action() if self.q_controller else self.args.lr
                        opt_state_unrep = unreplicate(p_state.opt_state)
                        opt_state_unrep[1].hyperparams['learning_rate'] = jnp.asarray(current_lr)
                        p_state = p_state.replace(opt_state=replicate(opt_state_unrep))
                        
                        metrics = {}
                        total_loss_step = 0.0

                        p_state, z_image, shared_metrics = p_shared_step(p_state, batch, self.model.apply, w_align)
                        metrics.update(unreplicate(shared_metrics))
                        total_loss_step += metrics['total_loss']

                        for i, res in enumerate(self.model.resolutions):
                            p_gen_keys = jax.random.split(gen_keys[i], self.num_devices)
                            p_state, gen_metrics = p_generator_step(p_state, batch, z_image, p_gen_keys, res, self.model.apply, self.args.perceptual_weight)
                            unrep_gen_metrics = unreplicate(gen_metrics)
                            metrics.update(unrep_gen_metrics)
                            total_loss_step += unrep_gen_metrics['total_loss']
                        
                        metrics['total_loss'] = total_loss_step

                        if step % 10 == 0:
                            m = metrics; m['lr'] = current_lr
                            if self.q_controller: self.q_controller.update_q_value(m['total_loss'], m.get('pixel_loss_64', 0.0))
                            progress.update(epoch_task, lr=m['lr'])
                            for k, v in self.metric_histories.items(): v.append(m.get(k, 0.0))

                            stats=Table(show_header=False,box=None,padding=(0,1)); stats.add_column(style="dim",width=15); stats.add_column(justify="right"); stats.add_row("Total Loss",f"[bold green]{m['total_loss']:.4f}[/]"); mem,util=self._get_gpu_stats(); stats.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats.add_row("GPU Util",f"[yellow]{util}[/]");
                            if self.args.use_sentinel: 
                                try:
                                    dampened_pct = unreplicate(p_state.opt_state)[0][1].dampened_pct
                                    stats.add_row("Sentinel Dampen", f"[cyan]{dampened_pct:.3%}[/]")
                                except (IndexError, AttributeError):
                                    pass
                            if self.q_controller:
                                status_map = {"IMPROVING": "bold green", "STAGNATED": "bold yellow", "REGRESSING": "bold red", "CONVERGING": "bold cyan", "STARTING": "dim"}; status = self.q_controller.status; color = status_map.get(status, "dim"); stats.add_row("Q-Ctrl Status", f"[{color}]{status}[/]")
                            layout["left"].update(Panel(stats,title="[bold]📊 Stats[/]"))
                            spark_w = max(1, (live.console.width*2//3)-25); losses=Table(show_header=False,box=None,padding=(0,1)); losses.add_column(style="dim",width=15); losses.add_column(width=10, justify="right"); losses.add_column(ratio=1)
                            for res in [64, 128, 256, 512]: losses.add_row(f"Pixel Loss {res}", f"{m.get(f'pixel_loss_{res}', 0.0):.4f}", f"[yellow]{self._get_sparkline(self.metric_histories[f'pixel_loss_{res}'], spark_w)}")
                            losses.add_row("Align Loss", f"[magenta]{m.get('align_loss', 0.0):.4f}[/magenta]", f"[magenta]{self._get_sparkline(self.metric_histories['align_loss'], spark_w)}"); losses.add_row("Perceptual Loss", f"[cyan]{m.get('perceptual_loss', 0.0):.4f}[/]", f"[cyan]{self._get_sparkline(self.metric_histories['perceptual_loss'], spark_w)}"); layout["right"].update(Panel(losses, title="[bold]📉 Losses[/]"))

                        progress.update(epoch_task,advance=1)
                    if self.should_shutdown: break
                    if (epoch + 1) % 5 == 0: self._save_checkpoint(p_state, epoch, ckpt_path); live.console.print(f"--- :floppy_disk: Checkpoint saved for epoch {epoch+1} ---")
        finally:
            print("\n--- Training loop finished. Saving final state... ---"); self._save_checkpoint(p_state, epoch_for_save, ckpt_path); print(f"--- :floppy_disk: Final state for epoch {epoch_for_save+1} saved. ---")
class Generator:
    def __init__(self, args):
        self.args = args; self.model = Chimera(d_model=args.d_model, dtype=DTYPE)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d_wubu.pkl")
        if not model_path.exists(): print(f"[FATAL] Model file not found at {model_path}."), sys.exit(1)
        with open(model_path, 'rb') as f: data = pickle.load(f); self.params = data['params']
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
        for param in self.clip_model.parameters(): param.requires_grad = False
    @partial(jax.jit, static_argnames=('self', 'resolution'))
    def _decode(self, params, z, resolution): return self.model.apply({'params': params}, z, resolution=resolution, method=self.model.decode_from_z)
    @partial(jax.jit, static_argnames=('self',))
    def _encode_text(self, params, text_embed): return self.model.apply({'params': params}, text_embed, method=self.model.encode_text_only)
    def _save_image(self, tensor, path_str):
        img_array = np.array((jax.device_get(tensor[0]) * 0.5 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
        Image.fromarray(img_array).save(path_str); print(f"✅ Image saved to {path_str}")
    def _save_gif(self, frames, path_str, duration=100):
        if not frames: print("[ERROR] No frames to save for GIF."); return
        frames[0].save(path_str, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f"✅ GIF saved to {path_str}")
    def generate(self):
        print(f"--- Generating initial image for prompt: '{self.args.prompt}' ---")
        with torch.no_grad():
            text_embed = self.clip_model.encode_text(clip.tokenize([self.args.prompt]).to(_clip_device)).cpu().numpy().astype(jnp.float32)
        z = self._encode_text(self.params, text_embed); img_tensor = self._decode(self.params, z, self.args.resolution)
        self._save_image(img_tensor, f"GEN_{''.join(c for c in self.args.prompt if c.isalnum())[:50]}_{self.args.seed}.png")
    def refine(self):
        print(f"--- Refining for prompt: '{self.args.prompt}' ({self.args.steps} steps) ---")
        with torch.no_grad():
            text_tokens = clip.tokenize([self.args.prompt]).to(_clip_device)
            target_text_features = self.clip_model.encode_text(text_tokens).float()
            initial_text_embed = target_text_features.cpu().numpy().astype(jnp.float32)
        z_base = self._encode_text(self.params, initial_text_embed); z_delta = jnp.zeros_like(z_base)
        optimizer = optax.adam(learning_rate=self.args.guidance_strength); opt_state = optimizer.init(z_delta)
        @partial(jax.jit, static_argnames=('self',))
        def refinement_step(z_delta, opt_state):
            def loss_fn(delta):
                generated_img_jax = self._decode(self.params, z_base + delta, self.args.resolution)
                img_torch = torch.from_numpy(np.array(generated_img_jax)).to(_clip_device).permute(0,3,1,2).float()
                img_features = self.clip_model.encode_image(self.clip_preprocess((img_torch+1)/2))
                return - (torch.cosine_similarity(target_text_features, img_features, dim=-1)).mean()
            loss_value, grads = jax.value_and_grad(loss_fn)(z_delta)
            updates, new_opt_state = optimizer.apply_updates(grads, opt_state); new_z_delta = optax.apply_updates(z_delta, updates)
            return new_z_delta, new_opt_state, loss_value
        frames = []; pbar = tqdm(range(self.args.steps), desc="Refining Latent")
        for i in pbar:
            z_delta, opt_state, loss = refinement_step(z_delta, opt_state); pbar.set_postfix({"CLIP Loss": f"{loss.item():.4f}"})
            if self.args.save_gif:
                img_tensor = self._decode(self.params, z_base + z_delta, self.args.resolution)
                frames.append(Image.fromarray(np.array((jax.device_get(img_tensor[0])*0.5+0.5).clip(0,1)*255, dtype=np.uint8)))
        print("--- Refinement complete. Generating final image... ---")
        final_img = self._decode(self.params, z_base + z_delta, self.args.resolution); prompt_name = ''.join(c for c in self.args.prompt if c.isalnum())[:50]
        self._save_image(final_img, f"REFINED_{prompt_name}_{self.args.seed}.png")
        if self.args.save_gif and frames: self._save_gif(frames, f"REFINED_PROCESS_{prompt_name}_{self.args.seed}.gif", duration=200)
    def blend(self):
        print(f"--- Blending '{self.args.base}' ({1-self.args.strength:.0%}) with '{self.args.modifier}' ({self.args.strength:.0%}) ---")
        with torch.no_grad():
            z_base = self._encode_text(self.params, self.clip_model.encode_text(clip.tokenize([self.args.base]).to(_clip_device)).cpu().numpy().astype(jnp.float32))
            z_modifier = self._encode_text(self.params, self.clip_model.encode_text(clip.tokenize([self.args.modifier]).to(_clip_device)).cpu().numpy().astype(jnp.float32))
        z_blended = z_base * (1 - jnp.clip(self.args.strength, 0.0, 1.0)) + z_modifier * jnp.clip(self.args.strength, 0.0, 1.0)
        print("--- Generating blended image... ---"); blended_img = self._decode(self.params, z_blended, self.args.resolution)
        base_name = ''.join(c for c in self.args.base if c.isalnum())[:20]; mod_name = ''.join(c for c in self.args.modifier if c.isalnum())[:20]
        self._save_image(blended_img, f"BLEND_{base_name}_{mod_name}_{self.args.seed}.png")
    def animate(self):
        print(f"--- Creating transform GIF from '{self.args.start}' to '{self.args.end}' ({self.args.steps} steps) ---")
        with torch.no_grad():
            z_start = self._encode_text(self.params, self.clip_model.encode_text(clip.tokenize([self.args.start]).to(_clip_device)).cpu().numpy().astype(jnp.float32))
            z_end = self._encode_text(self.params, self.clip_model.encode_text(clip.tokenize([self.args.end]).to(_clip_device)).cpu().numpy().astype(jnp.float32))
        frames = []
        for i in tqdm(range(self.args.steps), desc="Generating Frames"):
            alpha = i / (self.args.steps - 1)
            z_interp = z_start * (1 - alpha) + z_end * alpha
            img_tensor = self._decode(self.params, z_interp, self.args.resolution)
            frames.append(Image.fromarray(np.array((jax.device_get(img_tensor[0])*0.5+0.5).clip(0,1)*255, dtype=np.uint8)))
        start_name = ''.join(c for c in self.args.start if c.isalnum())[:20]; end_name = ''.join(c for c in self.args.end if c.isalnum())[:20]
        self._save_gif(frames, f"ANIM_{start_name}_to_{end_name}_{self.args.seed}.gif", duration=80)
def main():
    parser = argparse.ArgumentParser(description="Project Chimera (Wubu-Poincaré Architecture): A Parallel Coordinate Generative Model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_prep = subparsers.add_parser("prepare-data", help="Prepare multi-resolution images and CLIP text embeddings."); p_prep.add_argument('--image-dir', type=str, required=True)
    p_train = subparsers.add_parser("train", help="Train the memory-efficient parallel generator.")
    p_train.add_argument('--image-dir', type=str, required=True); p_train.add_argument('--basename', type=str, required=True); p_train.add_argument('--d-model', type=int, default=256)
    p_train.add_argument('--epochs', type=int, default=10000); p_train.add_argument('--batch-size', type=int, default=8, help="Global batch size across all devices."); p_train.add_argument('--lr', type=float, default=3e-4); p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--perceptual-weight', type=float, default=0.1, help="Weight for the Perceptual Loss.")
    p_train.add_argument('--use-q-controller', action='store_true'); p_train.add_argument('--finetune', action='store_true'); p_train.add_argument('--use-sentinel', action='store_true')
    for p in [subparsers.add_parser(c) for c in ["generate", "refine", "blend", "animate"]]:
        p.add_argument('--basename', type=str, required=True); p.add_argument('--d-model', type=int, default=256); p.add_argument('--seed', type=int, default=lambda: int(time.time()))
        p.add_argument('--resolution', type=int, default=512, help="Output resolution for generation.")
    p_gen = subparsers.choices["generate"]; p_gen.add_argument('--prompt', type=str, required=True)
    p_refine = subparsers.choices["refine"]; p_refine.add_argument('--prompt', type=str, required=True); p_refine.add_argument('--steps', type=int, default=10); p_refine.add_argument('--guidance-strength', type=float, default=0.05); p_refine.add_argument('--save-gif', action='store_true', help="Save the refinement process as a GIF.")
    p_blend = subparsers.choices["blend"]; p_blend.add_argument('--base', type=str, required=True); p_blend.add_argument('--modifier', type=str, required=True); p_blend.add_argument('--strength', type=float, default=0.5)
    p_animate = subparsers.choices["animate"]; p_animate.add_argument('--start', type=str, required=True); p_animate.add_argument('--end', type=str, required=True); p_animate.add_argument('--steps', type=int, default=60)
    args = parser.parse_args()
    if 'seed' in args and callable(args.seed): args.seed = args.seed()
    if args.command == "train" and args.batch_size % jax.local_device_count() != 0: print(f"[FATAL] Global batch size ({args.batch_size}) must be divisible by the number of devices ({jax.local_device_count()})."), sys.exit(1)
    if args.command == "prepare-data": prepare_data(args.image_dir)
    elif args.command == "train": Trainer(args).train()
    else: getattr(Generator(args), args.command)()
if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"); sys.exit(0)