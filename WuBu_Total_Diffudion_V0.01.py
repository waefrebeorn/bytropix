import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, common_utils
from flax.jax_utils import replicate, unreplicate
import optax
import numpy as np
from tqdm import tqdm
import pickle
from typing import Any, Tuple, Dict, Optional, List
import sys
import argparse
from collections import deque
import signal
from functools import partial
import random
import time
from pathlib import Path
import torch
from PIL import Image

# --- JAX Configuration ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_default_matmul_precision', 'bfloat16')
jax.config.update('jax_threefry_partitionable', True)

# --- Dependency Checks & Reusable Components (Unchanged) ---
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError: print("[FATAL] `tensorflow` not found."), sys.exit(1)
try:
    from sklearn.neighbors import BallTree, NearestNeighbors
except ImportError: print("[FATAL] `scikit-learn` not found."), sys.exit(1)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError: print("[FATAL] `scikit-learn` not found."), sys.exit(1)
try:
    import clip
except ImportError: print("[INFO] `clip` not found."), (clip := None)

class PoincareBall: # ... (code unchanged)
    EPS = 1e-7
    @staticmethod
    def project(x):
        x_f32 = x.astype(jnp.float32); norm_sq = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True); max_norm = 1.0 - PoincareBall.EPS
        projected_f32 = jnp.where(norm_sq >= 1.0, x_f32 / jnp.sqrt(norm_sq + PoincareBall.EPS) * max_norm, x_f32); return projected_f32.astype(x.dtype)
    @staticmethod
    def mobius_add(x, y, c):
        x_f32, y_f32, c_f32 = x.astype(jnp.float32), y.astype(jnp.float32), c.astype(jnp.float32)
        x2, y2, xy = jnp.sum(x_f32*x_f32, -1, keepdims=True), jnp.sum(y_f32*y_f32, -1, keepdims=True), jnp.sum(x_f32*y_f32, -1, keepdims=True)
        num = (1 + 2 * c_f32 * xy + c_f32 * y2) * x_f32 + (1 - c_f32 * x2) * y_f32
        den = 1 + 2 * c_f32 * xy + c_f32**2 * x2 * y2; return PoincareBall.project(num / den.clip(PoincareBall.EPS)).astype(x.dtype)
    @staticmethod
    def logmap0(y, c):
        y_f32, c_f32 = y.astype(jnp.float32), c.astype(jnp.float32); sqrt_c = jnp.sqrt(c_f32).clip(PoincareBall.EPS)
        y_norm = jnp.linalg.norm(y_f32, axis=-1, keepdims=True); safe_y_norm = y_norm.clip(PoincareBall.EPS, 1.0 - PoincareBall.EPS)
        direction = y_f32 / safe_y_norm; magnitude = jnp.arctanh(safe_y_norm) / sqrt_c
        result = jnp.where(y_norm < PoincareBall.EPS, jnp.zeros_like(y_f32), magnitude * direction); return result.astype(y.dtype)
    @staticmethod
    def expmap0(v, c):
        v_f32, c_f32 = v.astype(jnp.float32), c.astype(jnp.float32); sqrt_c = jnp.sqrt(c_f32).clip(PoincareBall.EPS)
        v_norm = jnp.linalg.norm(v_f32, axis=-1, keepdims=True); safe_v_norm = v_norm.clip(PoincareBall.EPS)
        direction = v_f32 / safe_v_norm; magnitude = jnp.tanh(sqrt_c * safe_v_norm) / sqrt_c
        result = jnp.where(v_norm < PoincareBall.EPS, jnp.zeros_like(v_f32), PoincareBall.project(magnitude * direction)); return result.astype(v.dtype)

# --- All nn.Module classes are unchanged ---
class ComplexEmbedding(nn.Module): # ...
    num_embeddings: int; features: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Embed(self.num_embeddings, self.features, name="real_embed", dtype=self.dtype)(x), nn.Embed(self.num_embeddings, self.features, name="imag_embed", dtype=self.dtype)(x)
class ComplexLayerNorm(nn.Module):
    dtype: Any = jnp.float32 # Keep LayerNorm ops in float32 for stability

    @nn.compact
    def __call__(self, x_complex: Tuple[jnp.ndarray, jnp.ndarray]):
        real, imag = x_complex
        # LayerNorm parameters (scale, bias) will be float32
        real_norm = nn.LayerNorm(dtype=self.dtype, name="real_ln")(real)
        imag_norm = nn.LayerNorm(dtype=self.dtype, name="imag_ln")(imag)
        # tanh is a safe activation. Cast back to input dtype if needed, but tanh output is fine.
        return nn.tanh(real_norm), nn.tanh(imag_norm)

class GalacticNavigator(nn.Module):
    d_model_total: int
    num_patches: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, patch_embeddings: jnp.ndarray) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        d_model_comp = self.d_model_total // 2
        patch_proj = nn.Dense(self.d_model_total, dtype=self.dtype, name="patch_proj")(patch_embeddings)
        
        pos_indices = jnp.arange(self.num_patches)
        pos_embed_r, pos_embed_i = ComplexEmbedding(self.num_patches, d_model_comp, name="pos_embed", dtype=self.dtype)(pos_indices)
        
        # Add dims for the leading (anchor,pos,neg) and (batch) axes for correct broadcasting
        # Shape becomes (1, 1, num_patches, d_model_comp)
        pos_embed_r = pos_embed_r[None, None, :, :]
        pos_embed_i = pos_embed_i[None, None, :, :]
        
        h_r = patch_proj[..., :d_model_comp] + pos_embed_r
        h_i = patch_proj[..., d_model_comp:] + pos_embed_i
        
        return {
            'syn': ComplexLayerNorm(name="norm_syn")((h_r, h_i)),
            'sem': ComplexLayerNorm(name="norm_sem")((h_r, h_i)),
            'exe': ComplexLayerNorm(name="norm_exe")((h_r, h_i))
        }
        
        
        
        
        
class SinusoidalPosEmb(nn.Module): # ...
    dim: int
    @nn.compact
    def __call__(self, time): half_dim=self.dim//2; embeddings=jnp.log(10000)/(half_dim-1); embeddings=jnp.exp(jnp.arange(half_dim)*-embeddings); embeddings=time[:,None]*embeddings[None,:]; return jnp.concatenate([jnp.sin(embeddings),jnp.cos(embeddings)],axis=-1)
class DenoisingResnetBlock(nn.Module): # ...
    out_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x, time_emb, cake_cond):
        in_dim = x.shape[-1]
        h = nn.GroupNorm(num_groups=min(in_dim//4, 32))(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_dim, kernel_size=(3,3), padding=1, dtype=self.dtype)(h)
        time_cond = nn.Dense(self.out_dim, dtype=self.dtype)(nn.swish(time_emb))
        cake_cond_proj = nn.Dense(self.out_dim, dtype=self.dtype)(nn.swish(cake_cond.astype(self.dtype)))
        h = h + time_cond[:, None, None, :] + cake_cond_proj[:, None, None, :]
        h = nn.GroupNorm(num_groups=min(h.shape[-1]//4, 32))(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_dim, kernel_size=(3,3), padding=1, dtype=self.dtype)(h)
        res_conn = nn.Conv(self.out_dim, kernel_size=(1,1), dtype=self.dtype)(x) if in_dim != self.out_dim else x
        return h + res_conn
class Downsample(nn.Module): # ...
    dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Conv(self.dim, kernel_size=(4,4), strides=(2,2), padding=1, dtype=self.dtype)(x)
class Upsample(nn.Module): # ...
    dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): B, H, W, C = x.shape; x = jax.image.resize(x, (B, H*2, W*2, C), 'nearest'); return nn.Conv(self.dim, kernel_size=(3,3), padding=1, dtype=self.dtype)(x)
# --- AFTER ---
class GalacticDenoisingUNet(nn.Module):
    d_model: int
    @nn.compact
    def __call__(self, noisy_image_patches, time, cake_conditioning):
        B, N, D = noisy_image_patches.shape; num_patches_side = int(np.sqrt(N)); x = noisy_image_patches.reshape(B, num_patches_side, num_patches_side, D)
        cake_cond_norm = nn.LayerNorm(dtype=jnp.float32, name="cake_cond_ln")(cake_conditioning)
        time_emb = SinusoidalPosEmb(self.d_model)(time)
        # RematResnetBlock = nn.remat(DenoisingResnetBlock) # <-- LINE REMOVED
        dims = [256, 512, 768]; skips = []; h = nn.Conv(dims[0], kernel_size=(3,3), padding=1, name='in_conv')(x)
        for i, dim_out in enumerate(dims):
            # Directly instantiate DenoisingResnetBlock instead of the remat version
            h = DenoisingResnetBlock(out_dim=dim_out, name=f'down_res_{i}')(h, time_emb, cake_cond_norm); skips.append(h)
            if i < len(dims) - 1: h = Downsample(dim=dim_out, name=f'downsample_{i}')(h)
        # Apply the same change to the middle and upsampling blocks
        h = DenoisingResnetBlock(out_dim=dims[-1], name='mid_res1')(h, time_emb, cake_cond_norm)
        h = DenoisingResnetBlock(out_dim=dims[-1], name='mid_res2')(h, time_emb, cake_cond_norm)
        for i, dim_out in enumerate(reversed(dims)):
            if i > 0: h = Upsample(dim=dim_out, name=f'upsample_{i-1}')(h)
            h = jnp.concatenate([h, skips.pop()], axis=-1)
            h = DenoisingResnetBlock(out_dim=dim_out, name=f'up_res_{i}')(h, time_emb, cake_cond_norm)
        out = nn.Conv(D, kernel_size=(3,3), padding=1, dtype=jnp.float32, name='out_conv')(h)
        return out.reshape(B, N, D)





class ImagePatchEncoder(nn.Module): # ...
    patch_size: int; in_channels: int; embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Conv(self.embed_dim,kernel_size=(self.patch_size,self.patch_size),strides=(self.patch_size,self.patch_size),dtype=self.dtype,name="conv_encoder")(x)
class ImagePatchDecoder(nn.Module): # ...
    patch_size: int; out_channels: int; embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(self.embed_dim,kernel_size=(self.patch_size,self.patch_size),strides=(self.patch_size,self.patch_size),dtype=self.dtype,name="ct_decoder")(x)
        x = nn.Conv(self.out_channels,kernel_size=(3,3),padding=1,dtype=jnp.float32,name="final_conv")(x); return nn.tanh(x)
def _extract(a, t, x_shape): b, *_ = t.shape; res = a[t]; return res.reshape(b, *((1,) * (len(x_shape) - 1)))
class DiffusionProcess: # ...
    def __init__(self, timesteps, beta_schedule='cosine'):
        self.timesteps=timesteps
        if beta_schedule=="cosine": s=0.008; x=jnp.linspace(0,timesteps,timesteps+1); ac=jnp.cos(((x/timesteps)+s)/(1+s)*jnp.pi*0.5)**2; self.betas=jnp.clip(1-(ac[1:]/ac[:-1]),0.0001,0.9999)
        else: self.betas = jnp.linspace(1e-4, 0.02, timesteps)
        self.alphas=1.-self.betas; self.alphas_cumprod=jnp.cumprod(self.alphas,axis=0); self.alphas_cumprod_prev=jnp.pad(self.alphas_cumprod[:-1],(1,0),constant_values=1.); self.sqrt_alphas_cumprod=jnp.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod=jnp.sqrt(1.-self.alphas_cumprod)
    def q_sample(self,x_start,t,noise): return _extract(self.sqrt_alphas_cumprod,t,x_start.shape)*x_start+_extract(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape)*noise

# --- Data loading functions are unchanged ---
def _get_or_create_caption_pairs(image_dir: str): # ...
    cache_file = Path(image_dir) / "caption_pairs.pkl";
    if cache_file.exists():
        with open(cache_file, "rb") as f: data = pickle.load(f); return data['pairs'], data['all_paths']
    image_paths = sorted([p for p in Path(image_dir).rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    captions, valid_paths = [], []
    for img_path in tqdm(image_paths, desc="Reading captions"):
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists(): captions.append(txt_path.read_text(encoding='utf-8', errors='ignore').strip()); valid_paths.append(str(img_path))
    if not valid_paths: raise ValueError(f"No images with matching .txt files found in {image_dir}")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000); caption_vectors = vectorizer.fit_transform(captions)
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine'); neighbors.fit(caption_vectors)
    _, indices = neighbors.kneighbors(caption_vectors)
    anchor_positive_pairs = [(valid_paths[i], valid_paths[indices[i][1]]) for i in range(len(valid_paths))]
    data_to_cache = {'pairs': anchor_positive_pairs, 'all_paths': valid_paths}
    with open(cache_file, "wb") as f: pickle.dump(data_to_cache, f)
    return anchor_positive_pairs, valid_paths




def create_laion_dataset(image_pairs: List[Tuple[str, str]], all_image_paths: List[str], image_size: int, batch_size: int, cache: bool = True):
    """
    Creates a highly optimized and robust tf.data pipeline for triplets.
    V5: Handles corrupted images by skipping them.
    """
    def _safe_load_and_preprocess_image(path):
        """
        Wraps the loading function in a try-catch block within TensorFlow.
        Returns a blank image and a boolean `False` if loading fails.
        """
        try:
            image_bytes = tf.io.read_file(path)
            image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC)
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = (tf.cast(image, tf.float32) / 127.5) - 1.0
            return image, True # Return the image and a success flag
        except tf.errors.InvalidArgumentError:
            # This error is caught if tf.image.decode_image fails.
            print(f"\n[Data Warning] Skipping corrupted image: {path.numpy().decode('utf-8')}\n")
            # Return a dummy image and a failure flag.
            # The dummy image must have the correct shape and type.
            return tf.zeros([image_size, image_size, 3], dtype=tf.float32), False

    def load_and_filter_triplet(pair, neg_path):
        """
        Loads all three images and checks their success flags.
        """
        anchor_path, positive_path = pair
        
        # Use a TensorFlow py_function to wrap our safe loader.
        # This allows us to use python-level try-except logic and printing.
        anchor_img, anchor_ok = tf.py_function(_safe_load_and_preprocess_image, [anchor_path], [tf.float32, tf.bool])
        pos_img, pos_ok = tf.py_function(_safe_load_and_preprocess_image, [positive_path], [tf.float32, tf.bool])
        neg_img, neg_ok = tf.py_function(_safe_load_and_preprocess_image, [neg_path], [tf.float32, tf.bool])
        
        # Ensure the shapes are correctly set after py_function
        anchor_img.set_shape([image_size, image_size, 3])
        pos_img.set_shape([image_size, image_size, 3])
        neg_img.set_shape([image_size, image_size, 3])
        
        # A triplet is valid only if all three images loaded correctly.
        all_ok = tf.logical_and(tf.logical_and(anchor_ok, pos_ok), neg_ok)
        
        return (anchor_img, pos_img, neg_img), all_ok

    num_samples = len(image_pairs)
    print(f"--- Building PURE TF DATA pipeline with {num_samples} anchor-positive pairs. ---")
    
    # --- The pipeline structure remains the same ---
    anchor_paths = [p[0] for p in image_pairs]
    positive_paths = [p[1] for p in image_pairs]
    ds_pairs = tf.data.Dataset.from_tensor_slices((anchor_paths, positive_paths))
    ds_negs = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds_negs = ds_negs.shuffle(buffer_size=10000).repeat()
    ds = tf.data.Dataset.zip((ds_pairs, ds_negs))
    ds = ds.shuffle(buffer_size=5000)

    # --- The changes are in the mapping and filtering ---
    
    # 1. Map the new safe loading function.
    # Each element will now be ((anchor, pos, neg), all_ok_flag).
    ds = ds.map(load_and_filter_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 2. Filter out the bad triplets.
    # This keeps only the elements where the all_ok_flag is True.
    ds = ds.filter(lambda images, all_ok: all_ok)
    
    # 3. After filtering, we only have the images left.
    # Remap to discard the now-unnecessary boolean flag.
    ds = ds.map(lambda images, all_ok: images)

    # The rest of the pipeline is the same.
    if cache:
        image_dir_name = Path(image_pairs[0][0]).parent.name
        cache_filename = f"./{image_dir_name}_{image_size}px.tfcache"
        print(f"--- Caching dataset to file: '{cache_filename}'. First epoch will be slow. Subsequent epochs will be very fast. ---")
        ds = ds.cache(cache_filename)
        
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds, num_samples 
# NEW: Dynamic Loss Balancer
class DynamicLossController:
    # ... (code unchanged from v0.28)
    def __init__(self, loss_names: List[str], history_len: int = 100, adjustment_strength: float = 0.05):
        self.weights = {name: 1.0 for name in loss_names}
        self.loss_histories = {name: deque(maxlen=history_len) for name in loss_names}
        self.strength = adjustment_strength
        print(f"--- Dynamic Loss Balancer initialized. Tracking: {loss_names} ---")
    def update(self, current_losses: Dict[str, np.ndarray]):
        for name, loss in current_losses.items():
            if np.isfinite(loss): self.loss_histories[name].append(float(loss))
        avg_losses = {name: np.mean(history) if history else 1.0 for name, history in self.loss_histories.items()}
        total_avg_loss = sum(avg_losses.values())
        if total_avg_loss == 0: return self.weights
        target_proportion = 1.0 / len(self.weights)
        for name in self.weights:
            current_proportion = avg_losses[name] / total_avg_loss
            error = current_proportion - target_proportion
            self.weights[name] += self.strength * error
        current_sum = sum(self.weights.values()); target_sum = len(self.weights)
        self.weights = {name: w * (target_sum / current_sum) for name, w in self.weights.items()}
        return self.weights

class GalacticDiffusionFunnel:
    def __init__(self, config):
        self.config = config
        self.key = jax.random.PRNGKey(config['seed'])
        self.params = {}
        self.should_shutdown = False
        signal.signal(signal.SIGINT, self._handle_sigint)
        self.d_model = config['d_model']
        self.models = {
            'patch_encoder': ImagePatchEncoder(patch_size=config['patch_size'], in_channels=config['channels'], embed_dim=self.d_model, dtype=jnp.bfloat16),
            'navigator': GalacticNavigator(d_model_total=self.d_model, num_patches=config['num_patches_side']**2, dtype=jnp.bfloat16),
            'denoiser': GalacticDenoisingUNet(d_model=self.d_model) # Inner dtype is bfloat16
        }
        self.diffusion = DiffusionProcess(timesteps=config['diffusion_timesteps'])
        self.num_devices = jax.local_device_count()

    def _handle_sigint(self, s, f):
        print("\n--- SIGINT received. Shutting down after this epoch... ---")
        self.should_shutdown = True

    def _init_params(self):
        # ... (This method is fine, no changes needed)
        if self.params: return
        print("--- Initializing model parameters on CPU... ---")
        weights_file = Path(f"{self.config['basename']}.weights.pkl")
        if weights_file.exists():
            print(f"--- Loading weights from {weights_file} ---")
            with open(weights_file, 'rb') as f:
                saved_data = pickle.load(f)
                if saved_data.get('d_model') == self.config['d_model']:
                    self.params = saved_data['params']
                    print("--- Weights loaded successfully. ---")
                    return
                print(f"[!!] d_model changed from {saved_data.get('d_model')} to {self.config['d_model']}. Re-initializing.")
        with jax.default_device(CPU_DEVICE):
            self.key, enc_key, nav_key, den_key = jax.random.split(self.key, 4)
            dummy_image = jnp.zeros((1, self.config['image_size'], self.config['image_size'], 3), jnp.bfloat16)
            print("Initializing Patch Encoder...")
            self.params['patch_encoder'] = self.models['patch_encoder'].init(enc_key, dummy_image)['params']
            dummy_patches = self.models['patch_encoder'].apply({'params': self.params['patch_encoder']}, dummy_image).reshape(1, -1, self.config['d_model'])
            print("Initializing Navigator...")
            self.params['navigator'] = self.models['navigator'].init(nav_key, dummy_patches)['params']
            dummy_time = jnp.array([0]); dummy_cond = jnp.zeros((1, self.config['d_model']))
            print("Initializing Denoiser...")
            self.params['denoiser'] = self.models['denoiser'].init(den_key, dummy_patches, dummy_time, dummy_cond)['params']
        print("--- CPU Initialization Complete. VRAM should be clear. ---")


    def train(self, image_dir, epochs, batch_size, cache_dataset=True):
        self._init_params()
        if batch_size % self.num_devices != 0: raise ValueError(f"Batch size must be divisible by {self.num_devices}")
        
        print(f"--- Starting STABILIZED Training on {self.num_devices} devices ---")
        
        # <<< FIX IS HERE: DEFINE `image_pairs` and `all_paths` BEFORE USING THEM >>>
        image_pairs, all_paths = _get_or_create_caption_pairs(image_dir)
        
        dataset, num_samples = create_laion_dataset(
            image_pairs, all_paths, self.config['image_size'], batch_size, cache=cache_dataset
        )
        steps_per_epoch = num_samples // batch_size
        total_steps = epochs * steps_per_epoch
        warmup_steps = min(5000, int(0.1 * total_steps)) # 10% of total steps or 5k, whichever is smaller

        print(f"--- Training Info ---")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps:     {total_steps}")
        print(f"Warmup steps:    {warmup_steps}")
        
        # <<< GAMEPLAN: STABILIZED OPTIMIZER >>>
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-7,
            peak_value=self.config['learning_rate'],
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=1e-6
        )

        optimizer = optax.chain(
            # Adaptive clipping is much more robust than global norm clipping for explosions.
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=1e-2, b1=0.9, b2=0.95)
        )
        
        state = train_state.TrainState.create(apply_fn=None, params=self.params, tx=optimizer)
        loss_controller = DynamicLossController(['triplet', 'denoise'])

        @partial(jax.pmap, axis_name='batch')
        def p_train_step(state, batch, key, triplet_w, denoise_w):
            
            def loss_fn(params):
                anchor_img, positive_img, negative_img = batch
                B = anchor_img.shape[0]
                
                # --- bfloat16 NN ops ---
                anchor_patches = self.models['patch_encoder'].apply({'params': params['patch_encoder']}, anchor_img).reshape(B, -1, self.d_model)
                positive_patches = self.models['patch_encoder'].apply({'params': params['patch_encoder']}, positive_img).reshape(B, -1, self.d_model)
                negative_patches = self.models['patch_encoder'].apply({'params': params['patch_encoder']}, negative_img).reshape(B, -1, self.d_model)
                
                all_patches = jnp.stack([anchor_patches, positive_patches, negative_patches])
                h_all = self.models['navigator'].apply({'params': params['navigator']}, all_patches)
                h_anchor, h_pos, h_neg = [jax.tree.map(lambda x: x[i], h_all) for i in range(3)]
                
                # --- float32 hyperbolic ops and loss ---
                triplet_loss = 0.0
                margin = 0.5
                for m in ['syn', 'sem', 'exe']:
                    # Cast to float32 for stable distance calculation
                    ah_r, ah_i = jnp.mean(h_anchor[m][0], 1).astype(jnp.float32), jnp.mean(h_anchor[m][1], 1).astype(jnp.float32)
                    ph_r, ph_i = jnp.mean(h_pos[m][0], 1).astype(jnp.float32), jnp.mean(h_pos[m][1], 1).astype(jnp.float32)
                    nh_r, nh_i = jnp.mean(h_neg[m][0], 1).astype(jnp.float32), jnp.mean(h_neg[m][1], 1).astype(jnp.float32)
                    dist_pos = jnp.sum((ah_r - ph_r)**2 + (ah_i - ph_i)**2, axis=-1)
                    dist_neg = jnp.sum((ah_r - nh_r)**2 + (ah_i - nh_i)**2, axis=-1)
                    triplet_loss += jnp.mean(jnp.maximum(0, dist_pos - dist_neg + margin))
                
                key_t, key_noise = jax.random.split(key)
                t = jax.random.randint(key_t, (B,), 0, self.diffusion.timesteps)
                noise = jax.random.normal(key_noise, anchor_patches.shape)
                noisy_patches = self.diffusion.q_sample(anchor_patches, t, noise)
                
                sem_r, sem_i = h_anchor['sem']
                cake_cond = jnp.concatenate([sem_r, sem_i], axis=-1).mean(axis=1)
                
                # Denoising path
                pred_noise = self.models['denoiser'].apply({'params': params['denoiser']}, noisy_patches, t, cake_cond)
                
                # <<< GAMEPLAN: STABLE LOSS CALCULATION >>>
                # Cast predictions and noise to float32 before squaring
                denoising_loss = jnp.mean((pred_noise.astype(jnp.float32) - noise.astype(jnp.float32))**2)

                # Combine losses in float32
                total_loss = (triplet_loss * triplet_w) + (denoising_loss * denoise_w)
                
                return total_loss, (triplet_loss, denoising_loss)

            # Compute gradients
            (loss, (triplet_loss, denoising_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            # Synchronize gradients across devices
            grads = jax.lax.pmean(grads, axis_name='batch')
            
            # <<< GAMEPLAN: GRADIENT MONITORING >>>
            grad_norm = optax.global_norm(grads)
            
            # Apply updates
            new_state = state.apply_gradients(grads=grads)
            
            metrics = {
                'loss': loss, 
                'triplet': triplet_loss, 
                'denoise': denoising_loss,
                'grad_norm': grad_norm,
                'lr': lr_schedule(state.step) # Log current learning rate
            }
            return new_state, jax.lax.pmean(metrics, axis_name='batch')

        # --- JIT WARM-UP (Corrected from previous step) ---
        print("\n--- JIT Compiling unified training step... This WILL take several minutes. Please wait. ---")
        start_jit = time.time()
        p_state = replicate(state)
        dummy_batch_shape = (batch_size // self.num_devices, self.config['image_size'], self.config['image_size'], 3)
        dummy_batch = (jnp.zeros(dummy_batch_shape, jnp.bfloat16),
                       jnp.zeros(dummy_batch_shape, jnp.bfloat16),
                       jnp.zeros(dummy_batch_shape, jnp.bfloat16))
        
        sharded_dummy_batch = common_utils.shard(dummy_batch)
        dummy_key = common_utils.shard_prng_key(self.key)
        dummy_triplet_w = jnp.ones(self.num_devices)
        dummy_denoise_w = jnp.ones(self.num_devices)
        
        _, jit_metrics_output = p_train_step(p_state, sharded_dummy_batch, dummy_key, dummy_triplet_w, dummy_denoise_w)
        jit_metrics_output['loss'].block_until_ready()
        print(f"--- Compilation finished in {time.time() - start_jit:.2f}s. Starting training. ---")
        
        for epoch in range(epochs):
            if self.should_shutdown: break
            print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
            pbar = tqdm(dataset.as_numpy_iterator(), total=steps_per_epoch, desc=f"Epoch {epoch+1}")
            
            last_step_metrics = {'triplet': 1.0, 'denoise': 1.0}

            for batch in pbar:
                if self.should_shutdown: break
                
                sharded_batch = common_utils.shard(batch)
                self.key, *step_keys = jax.random.split(self.key, self.num_devices + 1)
                sharded_keys = np.array(step_keys)
                
                weights = loss_controller.update(last_step_metrics)
                triplet_w_sharded = jnp.full((self.num_devices,), weights['triplet'])
                denoise_w_sharded = jnp.full((self.num_devices,), weights['denoise'])

                p_state, metrics = p_train_step(p_state, sharded_batch, sharded_keys, triplet_w_sharded, denoise_w_sharded)
                
                metrics = unreplicate(metrics)
                
                # Check for non-finite values and stop if they occur
                if not np.isfinite(metrics['loss']):
                    print("\n[FATAL] Loss is no longer finite. Halting training.")
                    print(f"Metrics: {metrics}")
                    self.save_weights() # Save weights for debugging
                    return # Exit training

                last_step_metrics = {k: v for k, v in metrics.items() if k in ['triplet', 'denoise']}

                pbar.set_postfix(
                    loss=f"{metrics['loss']:.3f}",
                    denoise=f"{metrics['denoise']:.3f}",
                    gnorm=f"{metrics['grad_norm']:.3f}",
                    lr=f"{metrics['lr']:.1e}"
                )

            self.params = unreplicate(p_state).params
            self.save_weights()
            if self.should_shutdown: break
            
    def save_weights(self):
        print(f"--- Saving weights to {self.config['basename']}.weights.pkl ---")
        data_to_save = {'params': jax.device_get(self.params), 'd_model': self.config['d_model']}
        with open(f"{self.config['basename']}.weights.pkl", 'wb') as f: pickle.dump(data_to_save, f)


def main():
    parser = argparse.ArgumentParser(description="GDF v0.30 (Stabilized Trainer)");
    parser.add_argument('command', nargs='?', default='train', choices=['train'], help="Action to perform.")
    parser.add_argument('--basename', type=str, default="gdf_model", help="Basename for model files.")
    parser.add_argument('--image_dir', type=str, default="./images/", help="Path to image directory.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=None, help="Global batch size. Defaults to 8 * num_gpus.")
    parser.add_argument('--d-model', type=int, default=512, help="Model embedding dimension.")
    parser.add_argument('--learning-rate', type=float, default=2e-4, help="Peak learning rate.") # Increased slightly as schedule is more stable
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--no-cache', action='store_false', dest='cache_dataset', 
                        help="Disable dataset caching. Caching is on by default for performance.")
    
    args = parser.parse_args()
    num_devices = jax.local_device_count()
    if args.batch_size is None:
        # A larger default batch size is better for stability and throughput
        args.batch_size = max(1, 8 * num_devices)
        print(f"[INFO] --batch-size not set. Defaulting to {args.batch_size} ({num_devices} devices * 8).")

    patch_size = 32
    image_size = 512
    MODEL_CONFIG = {
        'basename': f"{args.basename}_{args.d_model}d",
        'image_size': image_size, 'patch_size': patch_size, 
        'num_patches_side': image_size // patch_size, 
        'channels': 3, 'd_model': args.d_model, 
        'diffusion_timesteps': 1000,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
    }
    
    print(f"--- Galactic Diffusion Funnel v0.30 on {num_devices} device(s) ---")
    gdf = GalacticDiffusionFunnel(MODEL_CONFIG)
    
    if args.command == 'train':
        gdf.train(args.image_dir, args.epochs, args.batch_size, cache_dataset=args.cache_dataset)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- Program terminated by user. ---")
        sys.exit(0)
