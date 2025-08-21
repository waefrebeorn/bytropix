import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
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
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_default_matmul_precision', 'bfloat16')
jax.config.update('jax_threefry_partitionable', True)

# --- Dependency Checks ---
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError: print("[FATAL] `tensorflow` not found. `pip install tensorflow-cpu` is required."), sys.exit(1)
try:
    from sklearn.neighbors import BallTree, NearestNeighbors
except ImportError: print("[FATAL] `scikit-learn` not found. `pip install scikit-learn`."), sys.exit(1)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError: print("[FATAL] `scikit-learn` not found. `pip install scikit-learn`."), sys.exit(1)
try:
    import clip
except ImportError: print("[INFO] `clip` not found (`pip install git+https://github.com/openai/CLIP.git`). Text-to-image prompting will be disabled."), (clip := None)

# --- Re-usable Components ---
class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def _dist_sq(x, y, c):
        x_f32, y_f32, c_f32 = x.astype(jnp.float32), y.astype(jnp.float32), c.astype(jnp.float32)
        sqrt_c = jnp.sqrt(c_f32); mobius_add_result = PoincareBall.mobius_add(-x_f32, y_f32, c_f32)
        norm_val = jnp.linalg.norm(mobius_add_result, axis=-1).clip(PoincareBall.EPS, 1.0 - PoincareBall.EPS)
        dist = 2. / sqrt_c * jnp.arctanh(sqrt_c * norm_val); return (dist**2).astype(x.dtype)
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

class JaxHakmemQController:
    def __init__(self,initial_lr:float,config:Dict[str,Any],logger_suffix:str=""):
        self.config=config; self.current_lr=initial_lr; self.logger_suffix=logger_suffix; self.q_table_size=int(self.config["q_table_size"]); self.num_actions=int(self.config["num_lr_actions"]); self.lr_change_factors=np.array(self.config["lr_change_factors"], dtype=np.float32); self.q_table=np.zeros((self.q_table_size,self.num_actions),dtype=np.float32); self.learning_rate_q=float(self.config["learning_rate_q"]); self.discount_factor_q=float(self.config["discount_factor_q"]); self.exploration_rate_q=float(self.config["exploration_rate_q"]); self.lr_min=float(self.config["lr_min"]); self.lr_max=float(self.config["lr_max"]); self.loss_history=deque(maxlen=int(self.config["metric_history_len"])); self.loss_min=float(self.config["loss_min"]); self.loss_max=float(self.config["loss_max"]); self.last_action_idx:Optional[int]=None; self.last_state_idx:Optional[int]=None; self.short_term_window=5
        print(f"--- HAKMEM Q-Controller ({self.logger_suffix}) initialized (Trend-Aware). LR: {self.current_lr:.2e}, Q-Table: {self.q_table.shape} ---")
    def _discretize_value(self,value:float) -> int:
        if not np.isfinite(value): return self.q_table_size // 2
        if value<=self.loss_min: return 0;
        if value>=self.loss_max: return self.q_table_size-1
        bin_size=(self.loss_max-self.loss_min)/self.q_table_size; return min(int((value-self.loss_min)/bin_size),self.q_table_size-1)
    def _get_current_state_idx(self) -> int:
        if not self.loss_history: return self.q_table_size//2
        avg_loss=np.mean(list(self.loss_history)[-self.short_term_window:]); return self._discretize_value(avg_loss)
    def choose_action(self):
        self.last_state_idx = self._get_current_state_idx()
        if random.random() < self.exploration_rate_q: self.last_action_idx = random.randint(0, self.num_actions - 1)
        else: self.last_action_idx = np.argmax(self.q_table[self.last_state_idx]).item()
        change_factor = self.lr_change_factors[self.last_action_idx]; self.current_lr = np.clip(self.current_lr * change_factor, self.lr_min, self.lr_max); return self.current_lr
    def update_q_value(self,current_loss:float):
        if not np.isfinite(current_loss): return
        self.loss_history.append(current_loss)
        if self.last_state_idx is None or self.last_action_idx is None or len(self.loss_history)<self.loss_history.maxlen: return
        history_arr=np.array(self.loss_history); long_term_avg=np.mean(history_arr); short_term_avg=np.mean(history_arr[-self.short_term_window:])
        reward=long_term_avg-short_term_avg
        current_q=self.q_table[self.last_state_idx,self.last_action_idx]; next_state_idx=self._get_current_state_idx(); max_next_q=np.max(self.q_table[next_state_idx])
        new_q=current_q+self.learning_rate_q*(reward+self.discount_factor_q*max_next_q-current_q); self.q_table[self.last_state_idx,self.last_action_idx]=new_q
    def state_dict(self)->Dict[str,Any]: return {"current_lr":self.current_lr,"q_table":self.q_table.tolist(),"loss_history":list(self.loss_history),"last_action_idx":self.last_action_idx,"last_state_idx":self.last_state_idx}
    def load_state_dict(self,state_dict:Dict[str,Any]): self.current_lr=state_dict.get("current_lr",self.current_lr); self.q_table=np.array(state_dict.get("q_table",self.q_table.tolist()),dtype=np.float32); self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.loss_history.maxlen); self.last_action_idx=state_dict.get("last_action_idx"); self.last_state_idx=state_dict.get("last_state_idx")

### --- DATA LOADING (Unchanged, as it's efficient) --- ###
def _get_or_create_caption_pairs(image_dir: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    cache_file = Path(image_dir) / "caption_pairs.pkl"
    if cache_file.exists():
        print(f"--- Loading cached caption pairs from {cache_file} ---")
        with open(cache_file, "rb") as f: data = pickle.load(f)
        return data['pairs'], data['all_paths']
    print(f"--- No cache found. Scanning for images and captions in {image_dir} ---")
    image_paths = sorted([p for p in Path(image_dir).rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    captions, valid_paths = [], []
    for img_path in tqdm(image_paths, desc="Reading captions"):
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists():
            captions.append(txt_path.read_text(encoding='utf-8', errors='ignore').strip())
            valid_paths.append(str(img_path))
    if not valid_paths: raise ValueError(f"No images with matching .txt files found in {image_dir}")
    print(f"Found {len(valid_paths)} images with corresponding captions.")
    print("\n--- Vectorizing captions with TF-IDF to find similar pairs ---")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    caption_vectors = vectorizer.fit_transform(captions)
    print("\n--- Building nearest neighbors index for captions ---")
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
    neighbors.fit(caption_vectors)
    _, indices = neighbors.kneighbors(caption_vectors)
    print("\n--- Creating and saving anchor-positive pairs ---")
    anchor_positive_pairs = [(valid_paths[i], valid_paths[indices[i][1]]) for i in range(len(valid_paths))]
    data_to_cache = {'pairs': anchor_positive_pairs, 'all_paths': valid_paths}
    with open(cache_file, "wb") as f: pickle.dump(data_to_cache, f)
    print(f"\n--- Data Preparation Complete! Saved pairs to {cache_file} ---")
    return anchor_positive_pairs, valid_paths

def create_laion_dataset(image_pairs: List[Tuple[str, str]], all_image_paths: List[str], image_size: int, batch_size: int, for_navigator: bool) -> Tuple[tf.data.Dataset, int]:
    def _load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size], method='bicubic')
        image = tf.clip_by_value(image, 0.0, 255.0)
        return (tf.cast(image, tf.float32) / 127.5) - 1.0
    if for_navigator:
        num_samples = len(image_pairs)
        print(f"--- Building NAVIGATOR tf.data pipeline with {num_samples} anchor-positive pairs. ---")
        def generator():
            pairs = list(image_pairs)
            random.shuffle(pairs)
            for anchor_path, positive_path in pairs:
                while True:
                    negative_path = random.choice(all_image_paths)
                    if negative_path != anchor_path and negative_path != positive_path: break
                yield (anchor_path, positive_path, negative_path)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),) * 3)
        dataset = dataset.map(lambda a, p, n: (_load_and_preprocess_image(a), _load_and_preprocess_image(p), _load_and_preprocess_image(n)), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        num_samples = len(all_image_paths)
        print(f"--- Building DENOISER/CONSTRUCT tf.data pipeline with {num_samples} images. ---")
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        dataset = path_ds.map(_load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, num_samples


# --- Diffusion and Model Definitions ---
def _extract(a, t, x_shape): b, *_ = t.shape; res = a[t]; return res.reshape(b, *((1,) * (len(x_shape) - 1)))
class DiffusionProcess:
    def __init__(self, timesteps, beta_schedule='cosine'):
        self.timesteps=timesteps
        if beta_schedule=="cosine": s=0.008; x=jnp.linspace(0,timesteps,timesteps+1); ac=jnp.cos(((x/timesteps)+s)/(1+s)*jnp.pi*0.5)**2; self.betas=jnp.clip(1-(ac[1:]/ac[:-1]),0.0001,0.9999)
        else: self.betas = jnp.linspace(1e-4, 0.02, timesteps)
        self.alphas=1.-self.betas; self.alphas_cumprod=jnp.cumprod(self.alphas,axis=0); self.alphas_cumprod_prev=jnp.pad(self.alphas_cumprod[:-1],(1,0),constant_values=1.); self.sqrt_alphas_cumprod=jnp.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod=jnp.sqrt(1.-self.alphas_cumprod)
    def q_sample(self,x_start,t,noise): return _extract(self.sqrt_alphas_cumprod,t,x_start.shape)*x_start+_extract(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape)*noise
class ComplexEmbedding(nn.Module):
    num_embeddings: int; features: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Embed(self.num_embeddings, self.features, name="real_embed", dtype=self.dtype)(x), nn.Embed(self.num_embeddings, self.features, name="imag_embed", dtype=self.dtype)(x)
class ComplexLayerNorm(nn.Module):
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x_complex: Tuple[jnp.ndarray, jnp.ndarray]): real, imag = x_complex; return nn.tanh(nn.LayerNorm(dtype=self.dtype, name="real_ln")(real)), nn.tanh(nn.LayerNorm(dtype=self.dtype, name="imag_ln")(imag))
class GalacticNavigator(nn.Module):
    d_model_total: int; num_patches: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, patch_embeddings: jnp.ndarray) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        d_model_comp = self.d_model_total // 2
        patch_proj = nn.Dense(self.d_model_total, dtype=self.dtype, name="patch_proj")(patch_embeddings)
        pos_indices = jnp.arange(self.num_patches)
        pos_embed_r, pos_embed_i = ComplexEmbedding(self.num_patches, d_model_comp, name="pos_embed", dtype=self.dtype)(pos_indices)
        h_r = patch_proj[:, :, :d_model_comp] + pos_embed_r[None, :, :]
        h_i = patch_proj[:, :, d_model_comp:] + pos_embed_i[None, :, :]
        return {
            'syn': ComplexLayerNorm(name="norm_syn")((h_r, h_i)),
            'sem': ComplexLayerNorm(name="norm_sem")((h_r, h_i)),
            'exe': ComplexLayerNorm(name="norm_exe")((h_r, h_i))
        }
class SinusoidalPosEmb(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, time): half_dim=self.dim//2; embeddings=jnp.log(10000)/(half_dim-1); embeddings=jnp.exp(jnp.arange(half_dim)*-embeddings); embeddings=time[:,None]*embeddings[None,:]; return jnp.concatenate([jnp.sin(embeddings),jnp.cos(embeddings)],axis=-1)
class DenoisingResnetBlock(nn.Module):
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
class Downsample(nn.Module):
    dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Conv(self.dim, kernel_size=(4,4), strides=(2,2), padding=1, dtype=self.dtype)(x)
class Upsample(nn.Module):
    dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): B, H, W, C = x.shape; x = jax.image.resize(x, (B, H*2, W*2, C), 'nearest'); return nn.Conv(self.dim, kernel_size=(3,3), padding=1, dtype=self.dtype)(x)

# OPTIMIZATION: U-Net refactored to use gradient checkpointing and a standard upsampling path.
class GalacticDenoisingUNet(nn.Module):
    d_model: int
    @nn.compact
    def __call__(self, noisy_image_patches, time, cake_conditioning):
        B, N, D = noisy_image_patches.shape
        num_patches_side = int(np.sqrt(N))
        x = noisy_image_patches.reshape(B, num_patches_side, num_patches_side, D)
        time_emb = SinusoidalPosEmb(self.d_model)(time)
        
        # Define a rematerialized (checkpointed) ResNet block to save VRAM
        RematResnetBlock = nn.remat(DenoisingResnetBlock)

        dims = [256, 512, 768]
        skips = []
        h = nn.Conv(dims[0], kernel_size=(3,3), padding=1, name='in_conv')(x)
        
        # Downsampling path
        for i, dim_out in enumerate(dims):
            h = RematResnetBlock(out_dim=dim_out, name=f'down_res_{i}')(h, time_emb, cake_conditioning)
            skips.append(h)
            if i < len(dims) - 1:
                h = Downsample(dim=dim_out, name=f'downsample_{i}')(h)
        
        # Middle path
        h = RematResnetBlock(out_dim=dims[-1], name='mid_res1')(h, time_emb, cake_conditioning)
        h = RematResnetBlock(out_dim=dims[-1], name='mid_res2')(h, time_emb, cake_conditioning)
        
        # Upsampling path (Corrected architecture)
        for i, dim_out in enumerate(reversed(dims)):
            if i > 0: # Upsample before concatenation for all but the first up-block
                h = Upsample(dim=dim_out, name=f'upsample_{i-1}')(h)
            
            h = jnp.concatenate([h, skips.pop()], axis=-1)
            h = RematResnetBlock(out_dim=dim_out, name=f'up_res_{i}')(h, time_emb, cake_conditioning)
            
        out = nn.Conv(D, kernel_size=(3,3), padding=1, dtype=jnp.float32, name='out_conv')(h)
        return out.reshape(B, N, D)

class ImagePatchEncoder(nn.Module):
    patch_size: int; in_channels: int; embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Conv(self.embed_dim,kernel_size=(self.patch_size,self.patch_size),strides=(self.patch_size,self.patch_size),dtype=self.dtype,name="conv_encoder")(x)
class ImagePatchDecoder(nn.Module):
    patch_size: int; out_channels: int; embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(self.embed_dim,kernel_size=(self.patch_size,self.patch_size),strides=(self.patch_size,self.patch_size),dtype=self.dtype,name="ct_decoder")(x)
        x = nn.Conv(self.out_channels,kernel_size=(3,3),padding=1,dtype=jnp.float32,name="final_conv")(x); return nn.tanh(x)

# --- Main System Class ---
class GalacticDiffusionFunnel:
    def __init__(self, config):
        self.config, self.key = config, jax.random.PRNGKey(42); self.manifolds = ['syn', 'sem', 'exe']; self.params = {}
        self.ball_tree = {m: None for m in self.manifolds}; self.H_sphere_metadata = {m: [] for m in self.manifolds}
        self.should_shutdown = False; signal.signal(signal.SIGINT, self._handle_sigint)
        patch_size=config['patch_size']; num_patches=config['num_patches_side']**2
        self.navigator=GalacticNavigator(d_model_total=config['d_model'],num_patches=num_patches)
        self.denoiser=GalacticDenoisingUNet(d_model=config['d_model']); self.diffusion=DiffusionProcess(timesteps=config['diffusion_timesteps'])
        self.patch_encoder=ImagePatchEncoder(patch_size=patch_size,in_channels=config['channels'],embed_dim=config['d_model'])
        self.patch_decoder=ImagePatchDecoder(patch_size=patch_size,out_channels=config['channels'],embed_dim=config['d_model'])
        self.q_controller = None; self.train_state = None
        if clip: print("--- Loading CLIP model for text-to-image guidance... ---"); self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu"); self.clip_model.eval()
        else: self.clip_model = None
        self.probe_vocabulary = ["person","man","woman","child","animal","dog","cat","bird","tree","flower","grass","sky","cloud","sun","moon","water","ocean","river","mountain","field","road","building","house","car","boat","day","night","bright","dark","red","blue","green","yellow","orange","purple","white","black","gray","brown","fur","feathers","leaves","sand","rock","wood","metal","close-up","wide shot","landscape","portrait","still life","abstract","running","flying","swimming","standing","sitting","eating","sleeping","happy","sad","calm","stormy"]

    def _handle_sigint(self, s, f):
        print("\n--- SIGINT received. Shutting down and saving progress... ---")
        self.should_shutdown = True
        if self.train_state is not None: self.save_weights()

    def _init_models(self, mode: str):
        """
        Initializes models in a lazy, mode-aware fashion to avoid unnecessary JIT compilation.
        Also sets up the training state if required.
        """
        print(f"--- Configuring models for '{mode}' mode... ---")
        self.load_weights()

        # --- Step 1: Initialize common components ---
        print("--- JIT compiling models for first use. This may take a moment... ---")
        
        self.key, enc_key, nav_key, den_key, dec_key = jax.random.split(self.key, 5)
        dummy_image = jnp.zeros((1, self.config['image_size'], self.config['image_size'], self.config['channels']), dtype=jnp.bfloat16)
        
        # Encoder is almost always needed, so we initialize it first.
        if 'patch_encoder' not in self.params:
            print("Initializing Patch Encoder...")
            self.params['patch_encoder'] = self.patch_encoder.init(enc_key, dummy_image)['params']
        
        # --- Step 2: Set up mode-specific components ---
        is_training_mode = mode in ['navigator', 'denoiser']
        
        if is_training_mode:
            self.q_controller = JaxHakmemQController(self.config[f'learning_rate_{mode}'], self.config, mode.capitalize())
            self.load_weights() # Reload to get Q-controller state if it exists

            dummy_patch_grid = self.patch_encoder.apply({'params': self.params['patch_encoder']}, dummy_image)
            dummy_patch_flat = dummy_patch_grid.reshape(1, -1, self.config['d_model'])
            
            # ** THE CRITICAL FIX IS HERE **
            # We ensure the specific model to be trained has its parameters initialized
            # right before we create its training state. This is the most direct logic.
            if mode == 'navigator' and 'navigator' not in self.params:
                print("Initializing Galactic Navigator...")
                self.params['navigator'] = self.navigator.init(nav_key, dummy_patch_flat)['params']
            
            if mode == 'denoiser':
                # Denoiser training also requires a navigator for conditioning.
                if 'navigator' not in self.params:
                    print("Initializing Galactic Navigator (for Denoiser conditioning)...")
                    self.params['navigator'] = self.navigator.init(nav_key, dummy_patch_flat)['params']
                if 'denoiser' not in self.params:
                    print("Initializing Galactic Denoising U-Net...")
                    dummy_time = jnp.array([0])
                    dummy_cake_cond = jnp.zeros((1, self.config['d_model']))
                    self.params['denoiser'] = self.denoiser.init(den_key, dummy_patch_flat, dummy_time, dummy_cake_cond)['params']
            
            # Now, we are CERTAIN the parameters exist.
            if mode not in self.params:
                raise RuntimeError(f"FATAL: Logic error. Parameters for '{mode}' should have been initialized but were not.")

            model_map = {'navigator': self.navigator, 'denoiser': self.denoiser}
            target_model = model_map[mode]
            target_params = self.params[mode]
            
            base_optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=self.q_controller.current_lr, weight_decay=0.01)
            tx = optax.chain(optax.clip_by_global_norm(1.0), base_optimizer)
            self.train_state = train_state.TrainState.create(apply_fn=target_model.apply, params=target_params, tx=tx)

        # For non-training modes, initialize as needed
        elif mode == 'generate':
            if 'denoiser' not in self.params or 'patch_decoder' not in self.params:
                 # Re-run the full dummy data path for generation
                dummy_patch_grid = self.patch_encoder.apply({'params': self.params['patch_encoder']}, dummy_image)
                dummy_patch_flat = dummy_patch_grid.reshape(1, -1, self.config['d_model'])
                dummy_time = jnp.array([0])
                dummy_cake_cond = jnp.zeros((1, self.config['d_model']))
                if 'denoiser' not in self.params: self.params['denoiser'] = self.denoiser.init(den_key, dummy_patch_flat, dummy_time, dummy_cake_cond)['params']
                if 'patch_decoder' not in self.params: self.params['patch_decoder'] = self.patch_decoder.init(dec_key, dummy_patch_grid)['params']

        print("--- JIT compilation and model setup complete. ---")

        p_enc = sum(x.size for x in jax.tree.leaves(self.params.get('patch_encoder', {})))
        p_nav = sum(x.size for x in jax.tree.leaves(self.params.get('navigator', {})))
        p_den = sum(x.size for x in jax.tree.leaves(self.params.get('denoiser', {})))
        print(f"--- Model Config Summary. Encoder: {p_enc:,} | Navigator: {p_nav:,} | Denoiser: {p_den:,} params loaded/initialized. ---")


        
            
            
    def train_navigator(self, image_dir, epochs, batch_size):
        self._init_models('navigator')
        image_pairs, all_paths = _get_or_create_caption_pairs(image_dir)
        dataset, num_samples = create_laion_dataset(image_pairs, all_paths, self.config['image_size'], batch_size, for_navigator=True)
        
        # OPTIMIZATION: Process anchor/positive/negative separately to reduce peak VRAM
        @partial(jax.jit, static_argnames=['margin', 'manifold_loss_weights'])
        def train_step(state, enc_params, batch, margin, learning_rate, manifold_loss_weights):
            anchor_img, positive_img, negative_img = batch
            
            def loss_fn(nav_params):
                # Process each part of the triplet separately to avoid a large concatenated tensor
                anchor_patches = self.patch_encoder.apply({'params': enc_params}, anchor_img).reshape(anchor_img.shape[0], -1, self.config['d_model'])
                positive_patches = self.patch_encoder.apply({'params': enc_params}, positive_img).reshape(positive_img.shape[0], -1, self.config['d_model'])
                negative_patches = self.patch_encoder.apply({'params': enc_params}, negative_img).reshape(negative_img.shape[0], -1, self.config['d_model'])

                h_anchor = state.apply_fn({'params': nav_params}, anchor_patches)
                h_pos = state.apply_fn({'params': nav_params}, positive_patches)
                h_neg = state.apply_fn({'params': nav_params}, negative_patches)
                
                total_loss_step = 0.0
                for i, m in enumerate(self.manifolds):
                    ah_r, ah_i = jnp.mean(h_anchor[m][0], 1), jnp.mean(h_anchor[m][1], 1)
                    ph_r, ph_i = jnp.mean(h_pos[m][0], 1), jnp.mean(h_pos[m][1], 1)
                    nh_r, nh_i = jnp.mean(h_neg[m][0], 1), jnp.mean(h_neg[m][1], 1)
                    dist_pos = jnp.sum((ah_r - ph_r)**2 + (ah_i - ph_i)**2, axis=-1)
                    dist_neg = jnp.sum((ah_r - nh_r)**2 + (ah_i - nh_i)**2, axis=-1)
                    loss = jnp.mean(jnp.maximum(0, dist_pos - dist_neg + margin))
                    total_loss_step += loss * manifold_loss_weights[i]
                return total_loss_step
                
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, learning_rate=learning_rate)
            new_state = state.replace(step=state.step + 1, params=optax.apply_updates(state.params, updates), opt_state=new_opt_state)
            return new_state, {'loss': loss, 'grad_norm': optax.global_norm(grads)}
        
        manifold_weights_tuple = tuple(self.config['manifold_loss_weights'][m] for m in self.manifolds)
        for epoch in range(epochs):
            if self.should_shutdown: break
            print(f"\n--- Starting Navigator Epoch {epoch+1}/{epochs} ---")
            metrics_history = deque(maxlen=50); start_time = time.time(); total_steps = 0
            pbar = tqdm(dataset.as_numpy_iterator(), desc=f"Epoch {epoch+1}", total=num_samples // batch_size)
            for batch in pbar:
                if self.should_shutdown: break
                new_lr = self.q_controller.choose_action()
                self.train_state, metrics = train_step(self.train_state, self.params['patch_encoder'], batch, margin=0.5, learning_rate=new_lr, manifold_loss_weights=manifold_weights_tuple)
                self.q_controller.update_q_value(metrics['loss'].item()); metrics_history.append(jax.device_get(metrics)); total_steps += 1
                avg_metrics = {k: np.mean([m[k] for m in metrics_history]) for k in metrics_history[0].keys()}
                pbar.set_postfix(loss=f"{avg_metrics['loss']:.4f}", lr=f"{new_lr:.2e}", grad=f"{avg_metrics['grad_norm']:.2f}", sps=f"{total_steps/(time.time()-start_time+1e-6):.2f}")
        if not self.should_shutdown: print("\n--- Navigator training complete. ---"); self.params['navigator'] = jax.device_get(self.train_state.params); self.save_weights()
    
    def train_denoiser(self, image_dir, epochs, batch_size):
        self._init_models('denoiser')
        image_pairs, all_paths = _get_or_create_caption_pairs(image_dir)
        dataset, num_samples = create_laion_dataset(image_pairs, all_paths, self.config['image_size'], batch_size, for_navigator=False)
        
        # NOTE: Denoiser train step is already reasonably efficient. The main saving comes from the checkpointed U-Net.
        @jax.jit
        def train_step(state, nav_params, enc_params, batch, key, learning_rate):
            def loss_fn(denoiser_params):
                key_noise, key_time = jax.random.split(key); t = jax.random.randint(key_time, (batch.shape[0],), 0, self.diffusion.timesteps)
                clean_patches_grid = self.patch_encoder.apply({'params': enc_params}, batch)
                clean_patches_flat = clean_patches_grid.reshape(clean_patches_grid.shape[0],-1,self.config['d_model'])
                noise = jax.random.normal(key_noise, clean_patches_flat.shape); noisy_patches = self.diffusion.q_sample(clean_patches_flat, t, noise)
                nav_states = self.navigator.apply({'params': nav_params}, clean_patches_flat)
                sem_state_r, sem_state_i = nav_states['sem']; cake_cond = jnp.concatenate([sem_state_r, sem_state_i], axis=-1).mean(axis=1)
                predicted_noise_patches = state.apply_fn({'params': denoiser_params}, noisy_patches, t, cake_cond)
                return jnp.mean((predicted_noise_patches - noise)**2)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, learning_rate=learning_rate)
            new_state = state.replace(step=state.step + 1, params=optax.apply_updates(state.params, updates), opt_state=new_opt_state)
            return new_state, {'loss': loss, 'grad_norm': optax.global_norm(grads)}
        for epoch in range(epochs):
            if self.should_shutdown: break
            print(f"\n--- Starting Denoiser Epoch {epoch+1}/{epochs} ---")
            metrics_history = deque(maxlen=50); start_time = time.time(); total_steps = 0
            pbar = tqdm(dataset.as_numpy_iterator(), desc=f"Epoch {epoch+1}", total=num_samples // batch_size)
            for batch in pbar:
                if self.should_shutdown: break
                self.key, subkey = jax.random.split(self.key); new_lr = self.q_controller.choose_action()
                self.train_state, metrics = train_step(self.train_state, self.params['navigator'], self.params['patch_encoder'], batch, subkey, new_lr)
                self.q_controller.update_q_value(metrics['loss'].item()); metrics_history.append(jax.device_get(metrics)); total_steps += 1
                avg_metrics = {k: np.mean([m[k] for m in metrics_history]) for k in metrics_history[0].keys()}
                pbar.set_postfix(loss=f"{avg_metrics['loss']:.4f}", lr=f"{new_lr:.2e}", grad=f"{avg_metrics['grad_norm']:.2f}", sps=f"{total_steps/(time.time()-start_time+1e-6):.2f}")
        if not self.should_shutdown: print("\n--- Denoiser training complete. ---"); self.params['denoiser'] = jax.device_get(self.train_state.params); self.save_weights()

    def construct(self, image_dir):
        self._init_models('construct')
        if 'navigator' not in self.params: raise FileNotFoundError("Navigator must be trained first.")
        print("--- Constructing Visual Funnel Cake... ---")
        image_pairs, all_paths = _get_or_create_caption_pairs(image_dir)
        dataset, num_samples = create_laion_dataset(image_pairs, all_paths, self.config['image_size'], self.config['batch_size'], for_navigator=False)
        
        all_points = {m: [] for m in self.manifolds}
        
        @jax.jit
        def get_manifold_points(img_batch):
            patches = self.patch_encoder.apply({'params': self.params['patch_encoder']}, img_batch).reshape(img_batch.shape[0], -1, self.config['d_model'])
            nav_states = self.navigator.apply({'params': self.params['navigator']}, patches)
            results = {}
            for m in self.manifolds:
                c = jnp.array(self.config['manifold_curvatures'][m])
                r_points, i_points = nav_states[m]
                r_tangent = jax.vmap(PoincareBall.logmap0, in_axes=(0, None))(r_points, c).mean(axis=1)
                i_tangent = jax.vmap(PoincareBall.logmap0, in_axes=(0, None))(i_points, c).mean(axis=1)
                r_mean = jax.vmap(PoincareBall.expmap0, in_axes=(0, None))(r_tangent, c)
                i_mean = jax.vmap(PoincareBall.expmap0, in_axes=(0, None))(i_tangent, c)
                results[m] = jnp.concatenate([r_mean, i_mean], axis=-1)
            return results

        # OPTIMIZATION: Stream processing batch-by-batch to avoid storing all images in RAM.
        pbar = tqdm(dataset.as_numpy_iterator(), desc="Processing frames for cake", total=num_samples // self.config['batch_size'])
        for batch in pbar:
            # Process manifold points for the current batch
            mean_points = get_manifold_points(batch)
            for m in self.manifolds:
                all_points[m].append(jax.device_get(mean_points[m]))

            # Process CLIP embeddings for the current batch
            if self.clip_model:
                with torch.no_grad():
                    pil_images = [Image.fromarray(((img * 0.5 + 0.5) * 255).astype(np.uint8)) for img in jax.device_get(batch)]
                    processed_images = torch.stack([self.clip_preprocess(p) for p in pil_images])
                    embeddings = self.clip_model.encode_image(processed_images)
                    self.H_sphere_metadata['sem'].extend([{'clip_embedding': emb} for emb in embeddings.cpu().numpy()])
        
        for m in self.manifolds:
            if not all_points[m]: print(f"--- Manifold '{m}' has no points. Skipping. ---"); continue
            points = np.concatenate(all_points[m], axis=0)
            self.ball_tree[m] = BallTree(points, leaf_size=40)
            print(f"--- Manifold '{m}' constructed with {len(points)} points. ---")
        self.save_cake()
        
    def generate(self, prompt: str, num_frames: int = 1):
        if not hasattr(self, 'ball_tree') or not self.ball_tree.get('sem'): self.load_cake()
        if not self.clip_model: print("[ERROR] CLIP model not loaded."), sys.exit(1)
        print(f"--- Generating image for prompt: '\033[1;32m{prompt}\033[0m' ---")
        with torch.no_grad(): text_features = self.clip_model.encode_text(clip.tokenize([prompt])).cpu().numpy()
        sem_metadata = self.H_sphere_metadata.get('sem')
        if not sem_metadata or 'clip_embedding' not in sem_metadata[0] or self.ball_tree['sem'] is None: print("[ERROR] Cake metadata/BallTree missing CLIP embeddings."), sys.exit(1)
        clip_embeddings = np.stack([m['clip_embedding'] for m in sem_metadata]); dists = np.linalg.norm(clip_embeddings - text_features, axis=1); closest_idx = np.argmin(dists)
        cake_cond_syn = np.array(self.ball_tree['syn'].data[closest_idx]); cake_cond_sem = np.array(self.ball_tree['sem'].data[closest_idx]); cake_cond_exe = np.array(self.ball_tree['exe'].data[closest_idx])
        cake_cond = (jnp.array(cake_cond_syn * self.config['manifold_guidance_weights']['syn']) + jnp.array(cake_cond_sem * self.config['manifold_guidance_weights']['sem']) + jnp.array(cake_cond_exe * self.config['manifold_guidance_weights']['exe']))
        num_patches_side = self.config['num_patches_side']; patch_shape = (num_frames, num_patches_side**2, self.config['d_model'])
        self.key, subkey = jax.random.split(self.key); noisy_patches = jax.random.normal(subkey, patch_shape)
        @jax.jit
        def denoise_step(params, patches_t, t, cake_cond):
            pred_noise = self.denoiser.apply({'params': params}, patches_t, t, cake_cond); 
            alpha_t = _extract(self.diffusion.alphas,t,patches_t.shape)
            alpha_cumprod_t = _extract(self.diffusion.alphas_cumprod, t, patches_t.shape)
            sqrt_one_minus_alpha_cumprod = jnp.sqrt(1. - alpha_cumprod_t)
            # Denoising step using DDIM-like formulation for stability
            pred_x0 = (patches_t - sqrt_one_minus_alpha_cumprod * pred_noise) / jnp.sqrt(alpha_cumprod_t)
            return (1/jnp.sqrt(alpha_t))*(patches_t-((1-alpha_t)/sqrt_one_minus_alpha_cumprod)*pred_noise)
        
        for i in tqdm(reversed(range(self.diffusion.timesteps)), desc="Denoising", total=self.diffusion.timesteps):
            if self.should_shutdown: break
            t = jnp.full((num_frames,), i); noisy_patches = denoise_step(self.params['denoiser'], noisy_patches, t, jnp.repeat(cake_cond[None,:], num_frames, axis=0))
        
        final_patch_grid = noisy_patches.reshape(num_frames, num_patches_side, num_patches_side, self.config['d_model'])
        final_image = self.patch_decoder.apply({'params': self.params['patch_decoder']}, final_patch_grid)
        final_image_np = (np.asarray(final_image)[0] * 0.5 + 0.5).clip(0, 1)
        save_path = f"generated_{prompt.replace(' ','_')[:30]}.png"; Image.fromarray((final_image_np * 255).astype(np.uint8)).save(save_path)
        print(f"--- Image saved to {save_path} ---")

    def list_embeddings(self):
        print("--- Listing discovered concepts from Funnel Cake... ---")
        if not self.clip_model: print("[ERROR] CLIP model is required for this feature."); return
        self.load_cake()
        sem_metadata = self.H_sphere_metadata.get('sem')
        if not sem_metadata or 'clip_embedding' not in sem_metadata[0]: print("[ERROR] Cake is missing required CLIP embeddings. Please run 'construct' first."); return
        image_features = np.stack([m['clip_embedding'] for m in sem_metadata])
        with torch.no_grad():
            text_tokens = clip.tokenize(self.probe_vocabulary); text_features = self.clip_model.encode_text(text_tokens).cpu().numpy()
        image_features /= np.linalg.norm(image_features, axis=1, keepdims=True); text_features /= np.linalg.norm(text_features, axis=1, keepdims=True)
        similarity = image_features @ text_features.T
        best_word_indices = np.argmax(similarity, axis=1)
        discovered_concepts = sorted(list(set(self.probe_vocabulary[i] for i in best_word_indices)))
        print("--- Discovered Concepts ---\n" + ", ".join(discovered_concepts))

    def save_weights(self):
        mode = "denoiser" if self.train_state.apply_fn == self.denoiser.apply else "navigator"
        self.params[mode] = jax.device_get(self.train_state.params)
        print(f"--- Saving weights & Q-Controller state to {self.config['basename']}.weights.pkl ---")
        checkpoint = {'params': jax.device_get(self.params)}
        if self.q_controller: checkpoint['q_controller_state'] = self.q_controller.state_dict()
        with open(f"{self.config['basename']}.weights.pkl", 'wb') as f: pickle.dump(checkpoint, f)

    def load_weights(self):
        fpath = f"{self.config['basename']}.weights.pkl"
        if os.path.exists(fpath):
            print(f"--- Loading weights & Q-Controller state from {fpath} ---")
            with open(fpath, 'rb') as f:
                checkpoint = pickle.load(f)
                self.params = checkpoint.get('params', {})
                if q_state := checkpoint.get('q_controller_state'):
                    if hasattr(self, 'q_controller') and self.q_controller is not None: self.q_controller.load_state_dict(q_state)
    
    def save_cake(self):
        print(f"--- Saving cake to {self.config['basename']}.cake ---")
        with open(f"{self.config['basename']}.cake", 'wb') as f: pickle.dump({'ball_tree': self.ball_tree,'H_sphere_metadata': self.H_sphere_metadata}, f)
    
    def load_cake(self):
        fpath = f"{self.config['basename']}.cake"
        if os.path.exists(fpath):
            print(f"--- Loading cake from {fpath} ---")
            with open(fpath, 'rb') as f: data = pickle.load(f); self.ball_tree = data['ball_tree']; self.H_sphere_metadata = data['H_sphere_metadata']
        else: print(f"[ERROR] Cake file not found at {fpath}. Please run 'construct' first."), sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Galactic Diffusion Funnel v0.18 (Optimized for LAION)"); 
    parser.add_argument('command', nargs='?', default='generate', choices=['train_navigator', 'train_denoiser', 'construct', 'generate'], help="The main command to execute.")
    parser.add_argument('--basename', type=str, default="gdf_laion_512px_model", help="Basename for model files.")
    parser.add_argument('--image_dir', type=str, default="./laion_aesthetics_512/", help="Path to the root LAION image directory (containing subfolders like 00000, 00001, etc.).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size. Adjust based on VRAM (e.g., 1-8 for 512px on a 24GB GPU).")
    parser.add_argument('--listemb', action='store_true', help="List discovered concepts from the cake and exit.")
    args = parser.parse_args()

    patch_size = 32
    image_size = 512
    num_patches_side = image_size // patch_size
    MODEL_CONFIG = {
        'basename': args.basename, 'image_size': image_size, 'patch_size': patch_size, 
        'num_patches_side': num_patches_side, 'channels': 3, 'd_model': 1024, 
        'diffusion_timesteps': 1000, 'batch_size': args.batch_size,
        'learning_rate_navigator': 1e-4, 'learning_rate_denoiser': 2e-4, 
        'manifold_curvatures':{'syn':5.0,'sem':1.0,'exe':0.1},
        'manifold_loss_weights':{'syn':0.2,'sem':0.6,'exe':0.2},
        'manifold_guidance_weights':{'syn':0.1,'sem':0.8,'exe':0.1}, 
    }
    QLEARN_CONFIG = {
        "q_table_size":100, "num_lr_actions":5, "lr_change_factors":[0.8, 0.95, 1.0, 1.05, 1.2],
        "learning_rate_q":0.1, "discount_factor_q":0.9, "exploration_rate_q":0.1, "lr_min":1e-7, "lr_max":5e-3, 
        "metric_history_len":500, "loss_min":0.0, "loss_max":2.5 
    }
    FULL_CONFIG = {**MODEL_CONFIG, **QLEARN_CONFIG}

    print("--- Galactic Diffusion Funnel v0.18 (Optimized for LAION) ---")
    gdf = GalacticDiffusionFunnel(FULL_CONFIG)
    
    if args.listemb: gdf.list_embeddings(); sys.exit(0)

    if not os.path.isdir(args.image_dir) and args.command in ['train_navigator', 'train_denoiser', 'construct']:
        print(f"[FATAL] Image directory '{args.image_dir}' not found."); sys.exit(1)
    
    if args.command == 'train_navigator': gdf.train_navigator(args.image_dir, args.epochs, args.batch_size)
    elif args.command == 'train_denoiser': gdf.train_denoiser(args.image_dir, args.epochs, args.batch_size)
    elif args.command == 'construct': gdf.construct(args.image_dir)
    elif args.command == 'generate': 
        gdf._init_models('generate') 
        print("\n--- Galactic Diffusion Console ---")
        while not gdf.should_shutdown:
            try: prompt = input("\nYour Prompt> ")
            except EOFError: print("\n--- Exiting. ---"); break
            if prompt.lower() in ["exit","quit"]: break
            gdf.generate(prompt)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"); sys.exit(0)