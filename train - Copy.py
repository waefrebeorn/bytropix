# train.py
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, common_utils
from flax.jax_utils import replicate, unreplicate
import optax
import numpy as np
import pickle
from typing import Any, Tuple, Dict, List
import sys
import argparse
from collections import deque
import signal
from functools import partial
import random
import time
from pathlib import Path

# --- JAX Configuration: Optimized for Speed ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_default_matmul_precision', 'bfloat16')
jax.config.update('jax_threefry_partitionable', True)

# --- Dependency Checks ---
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError: print("[FATAL] `tensorflow` not found."), sys.exit(1)
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError: print("[FATAL] `scikit-learn` not found."), sys.exit(1)
try:
    from tqdm import tqdm
except ImportError: print("[FATAL] `tqdm` not found."), sys.exit(1)


# --- WUBU GEOMETRY (FROM YOUR RESEARCH) ---
class PoincareBall:
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

# --- MODEL DEFINITIONS (v7.0 - Hard Negative Mining) ---
class ComplexEmbedding(nn.Module):
    num_embeddings: int; features: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Embed(self.num_embeddings, self.features, name="real_embed", dtype=self.dtype)(x), nn.Embed(self.num_embeddings, self.features, name="imag_embed", dtype=self.dtype)(x)

class ComplexLayerNorm(nn.Module):
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x_complex: Tuple[jnp.ndarray, jnp.ndarray]):
        real, imag = x_complex
        real_norm = nn.LayerNorm(dtype=self.dtype, name="real_ln")(real)
        imag_norm = nn.LayerNorm(dtype=self.dtype, name="imag_ln")(imag)
        return nn.tanh(real_norm), nn.tanh(imag_norm)

class GalacticNavigator(nn.Module):
    d_model_total: int; num_patches: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, patch_embeddings: jnp.ndarray) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
        d_model_comp = self.d_model_total // 2
        patch_proj = nn.Dense(self.d_model_total, dtype=self.dtype, name="patch_proj")(patch_embeddings)
        pos_indices = jnp.arange(self.num_patches)
        pos_embed_r, pos_embed_i = ComplexEmbedding(self.num_patches, d_model_comp, name="pos_embed", dtype=self.dtype)(pos_indices)
        pos_embed_r = pos_embed_r.reshape((1,) * (patch_embeddings.ndim - 2) + (self.num_patches, d_model_comp))
        pos_embed_i = pos_embed_i.reshape((1,) * (patch_embeddings.ndim - 2) + (self.num_patches, d_model_comp))
        h_r = patch_proj[..., :d_model_comp] + pos_embed_r
        h_i = patch_proj[..., d_model_comp:] + pos_embed_i
        return {
            'syn': ComplexLayerNorm(name="norm_syn")((h_r, h_i)),
            'sem': ComplexLayerNorm(name="norm_sem")((h_r, h_i)),
            'exe': ComplexLayerNorm(name="norm_exe")((h_r, h_i))
        }

class SinusoidalPosEmb(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)

class HyperbolicDenoisingBlock(nn.Module):
    d_model: int; num_heads: int; mlp_dim: int; dtype: Any = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x, time_emb, cake_cond, curvature):
        h_tangent = PoincareBall.logmap0(x, curvature)
        h_norm = nn.LayerNorm(dtype=jnp.float32)(h_tangent)
        
        attn_out_tangent = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.d_model, dtype=self.dtype
        )(h_norm, h_norm)
        
        attn_out_poincare = PoincareBall.expmap0(attn_out_tangent, curvature)
        x = PoincareBall.mobius_add(x, attn_out_poincare, curvature)

        h_tangent = PoincareBall.logmap0(x, curvature)
        h_norm = nn.LayerNorm(dtype=jnp.float32)(h_tangent)

        time_cond_proj = nn.Dense(self.d_model, dtype=self.dtype)(nn.gelu(time_emb))
        cake_cond_proj = nn.Dense(self.d_model, dtype=self.dtype)(nn.gelu(cake_cond))
        h_norm = h_norm + time_cond_proj[:, None, :] + cake_cond_proj[:, None, :]
        
        mlp_out_tangent = nn.Dense(self.mlp_dim, dtype=self.dtype)(h_norm)
        mlp_out_tangent = nn.gelu(mlp_out_tangent)
        mlp_out_tangent = nn.Dense(self.d_model, dtype=self.dtype)(mlp_out_tangent)

        mlp_out_poincare = PoincareBall.expmap0(mlp_out_tangent, curvature)
        x = PoincareBall.mobius_add(x, mlp_out_poincare, curvature)
        
        return x

class HyperbolicDenoisingNetwork(nn.Module):
    d_model: int; num_layers: int; num_heads: int; mlp_dim: int
    
    @nn.compact
    def __call__(self, noisy_patches, time, cake_conditioning):
        curvature = self.param('curvature', nn.initializers.constant(1.0), (1,), jnp.float32)
        curvature = nn.softplus(curvature) + 1e-7

        h = PoincareBall.project(noisy_patches)
        cake_cond_norm = nn.LayerNorm(dtype=jnp.float32, name="cake_cond_ln")(cake_conditioning)
        time_emb = SinusoidalPosEmb(self.d_model)(time)
        time_emb = nn.Dense(self.d_model)(time_emb)

        for _ in range(self.num_layers):
            h = HyperbolicDenoisingBlock(
                d_model=self.d_model, num_heads=self.num_heads, mlp_dim=self.mlp_dim
            )(h, time_emb, cake_cond_norm, curvature)
        
        predicted_noise_tangent = PoincareBall.logmap0(h, curvature)
        predicted_noise = nn.Dense(self.d_model, dtype=jnp.float32, name='out_proj')(predicted_noise_tangent)
        
        return predicted_noise

class ImagePatchEncoder(nn.Module):
    patch_size: int; in_channels: int; embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): return nn.Conv(self.embed_dim,kernel_size=(self.patch_size,self.patch_size),strides=(self.patch_size,self.patch_size),dtype=self.dtype,name="conv_encoder")(x)

class StableImagePatchDecoder(nn.Module):
    patch_size: int; out_channels: int; embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x): # Input shape: (B, H_patches, W_patches, D)
        num_upsamples = int(np.log2(self.patch_size))
        ch = self.embed_dim
        
        def ResBlock(inner_x, out_ch):
            h_in = nn.gelu(nn.LayerNorm(dtype=jnp.float32)(inner_x))
            h = nn.Conv(out_ch, kernel_size=(3, 3), padding='SAME', dtype=self.dtype)(h_in)
            h = nn.gelu(nn.LayerNorm(dtype=jnp.float32)(h))
            h = nn.Conv(out_ch, kernel_size=(3, 3), padding='SAME', dtype=self.dtype)(h)
            if inner_x.shape[-1] != out_ch:
                inner_x = nn.Conv(out_ch, kernel_size=(1, 1), dtype=self.dtype)(inner_x)
            return inner_x + h

        h = nn.Conv(ch, kernel_size=(3,3), padding='SAME', dtype=self.dtype)(x)
        h = ResBlock(h, ch)

        for i in range(num_upsamples):
            B, H, W, C = h.shape
            h = jax.image.resize(h, (B, H * 2, W * 2, C), 'nearest')
            out_ch = max(ch // 2, self.out_channels*4) 
            h = nn.Conv(out_ch, kernel_size=(3, 3), padding='SAME', dtype=self.dtype)(h)
            h = ResBlock(h, out_ch)
            ch = out_ch
        
        h = nn.gelu(nn.LayerNorm(dtype=jnp.float32)(h))
        h = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME', dtype=jnp.float32)(h)
        return nn.tanh(h)

class DiffusionProcess:
    def __init__(self, timesteps, beta_schedule='cosine'):
        self.timesteps = timesteps
        if beta_schedule == "cosine":
            s = 0.008; x = jnp.linspace(0, timesteps, timesteps + 1)
            ac = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5)**2
            betas = jnp.clip(1 - (ac[1:] / ac[:-1]), 0.0001, 0.9999)
        else: betas = jnp.linspace(1e-4, 0.02, timesteps)
        self.betas = betas
        self.alphas = 1. - self.betas; self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = jnp.pad(self.alphas_cumprod[:-1], (1, 0), constant_values=1.)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod); self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

def _extract(a, t, x_shape):
    b, *_ = t.shape; res = a[t]; return res.reshape(b, *((1,) * (len(x_shape) - 1)))

# --- DATA PIPELINE ---
def _bytes_feature(value): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_to_tfrecords(image_dir: str, image_size: int):
    from tqdm import tqdm
    print("--- Converting dataset to TFRecord format with Hard Negative Mining... ---")
    image_paths_raw = sorted([p for p in Path(image_dir).rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    captions, image_paths = [], []
    for img_path in tqdm(image_paths_raw, desc="Reading captions"):
        txt_path = img_path.with_suffix('.txt')
        if txt_path.exists():
            captions.append(txt_path.read_text(encoding='utf-8', errors='ignore').strip())
            image_paths.append(str(img_path))

    if not image_paths: raise ValueError(f"No images with matching .txt files found in {image_dir}")

    print("--- Building TF-IDF matrix... ---")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
    caption_vectors = vectorizer.fit_transform(captions)

    num_neighbors = min(50, len(image_paths))
    print(f"--- Finding {num_neighbors} nearest neighbors for each image... ---")
    neighbors = NearestNeighbors(n_neighbors=num_neighbors, algorithm='brute', metric='cosine').fit(caption_vectors)
    distances, indices = neighbors.kneighbors(caption_vectors)
    
    def process_image(path):
        try:
            img_bytes = tf.io.read_file(path)
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img = tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC)
            img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
            return tf.io.encode_jpeg(img, quality=95).numpy()
        except Exception as e:
            print(f"Warning: Skipping corrupted image {path} due to error: {e}")
            return None

    triplet_tfrecord_path = Path(image_dir) / f"triplets_{image_size}.tfrecord"
    single_tfrecord_path = Path(image_dir) / f"singles_{image_size}.tfrecord"
    written_count = 0
    with tf.io.TFRecordWriter(str(triplet_tfrecord_path)) as triplet_writer, \
         tf.io.TFRecordWriter(str(single_tfrecord_path)) as single_writer:
        for i in tqdm(range(len(image_paths)), desc="Writing TFRecords with Hard Negatives"):
            anchor_path = image_paths[i]
            
            # The positive is always the closest neighbor (that isn't itself)
            positive_path = image_paths[indices[i][1]]

            # Hard Negative Mining:
            # The pool of hard negatives are the next closest neighbors (from index 2 to num_neighbors)
            hard_negative_pool_indices = indices[i][2:]
            if len(hard_negative_pool_indices) == 0:
                continue # Skip if there's no one else to choose from
            
            # Select a random negative from this much more challenging pool
            chosen_negative_idx = random.choice(hard_negative_pool_indices)
            negative_path = image_paths[chosen_negative_idx]

            anchor_bytes, pos_bytes, neg_bytes = process_image(anchor_path), process_image(positive_path), process_image(negative_path)
            
            if all((anchor_bytes, pos_bytes, neg_bytes)):
                triplet_feature = {'anchor': _bytes_feature(anchor_bytes), 'positive': _bytes_feature(pos_bytes), 'negative': _bytes_feature(neg_bytes)}
                triplet_example = tf.train.Example(features=tf.train.Features(feature=triplet_feature)); triplet_writer.write(triplet_example.SerializeToString())
                single_feature = {'image': _bytes_feature(anchor_bytes)}
                single_example = tf.train.Example(features=tf.train.Features(feature=single_feature)); single_writer.write(single_example.SerializeToString())
                written_count += 1
                
    with open(Path(image_dir) / "dataset_info.pkl", "wb") as f: pickle.dump({"num_samples": written_count}, f)
    print(f"--- TFRecord conversion complete. Wrote {written_count} valid triplets and singles. ---")

def create_dataset(image_dir: str, image_size: int, batch_size: int):
    record_file = Path(image_dir) / f"triplets_{image_size}.tfrecord"
    if not record_file.exists(): raise FileNotFoundError(f"TFRecord file not found. Run 'convert-to-tfrecords' command first.")
    print(f"--- Loading from optimized TFRecord: {record_file} ---")
    def _parse_and_normalize(proto):
        img = tf.io.decode_jpeg(proto, channels=3); img = (tf.cast(img, tf.float32) / 127.5) - 1.0; img.set_shape([image_size, image_size, 3]); return img
    feature_desc = {'anchor': tf.io.FixedLenFeature([], tf.string), 'positive': tf.io.FixedLenFeature([], tf.string), 'negative': tf.io.FixedLenFeature([], tf.string)}
    def parser(x):
        features = tf.io.parse_single_example(x, feature_desc)
        return (_parse_and_normalize(features['anchor']), _parse_and_normalize(features['positive']), _parse_and_normalize(features['negative']))
    ds = tf.data.TFRecordDataset(str(record_file), num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.shuffle(4096).repeat().map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

# --- DYNAMIC LOSS CONTROLLER ---
class DynamicLossController:
    def __init__(self, target_weights: Dict[str, float], history_len: int = 100, adjustment_strength: float = 0.01, min_weight: float = 0.1):
        self.loss_names = list(target_weights.keys())
        self.target_weights = target_weights
        target_sum = sum(self.target_weights.values())
        self.target_proportions = {name: w / target_sum for name, w in self.target_weights.items()}
        self.weights = {name: 1.0 for name in self.loss_names}
        self.loss_histories = {name: deque(maxlen=history_len) for name in self.loss_names}
        self.strength = adjustment_strength
        self.min_weight = min_weight
        print(f"--- Dynamic Loss Balancer initialized. ---")
        print(f"    Tracking: {self.loss_names}")
        print(f"    Target Proportions: {self.target_proportions}")

    def update(self, current_losses: Dict[str, float]):
        for name, loss in current_losses.items():
            if name in self.loss_histories and np.isfinite(loss):
                self.loss_histories[name].append(loss)
        weighted_losses = {name: np.mean(history) * self.weights.get(name, 1.0) for name, history in self.loss_histories.items() if history}
        if not weighted_losses: return self.weights
        total_weighted_loss = sum(weighted_losses.values())
        if total_weighted_loss < 1e-6: return self.weights
        for name in self.weights:
            current_proportion = weighted_losses.get(name, 0) / total_weighted_loss
            target_proportion = self.target_proportions.get(name, 0)
            error = current_proportion - target_proportion
            self.weights[name] -= self.strength * error
            self.weights[name] = max(self.min_weight, self.weights[name])
        current_sum = sum(self.weights.values())
        if current_sum > 0:
            target_sum = len(self.weights)
            self.weights = {name: w * (target_sum / current_sum) for name, w in self.weights.items()}
        return self.weights

class GalacticDiffusionFunnel:
    def __init__(self, config):
        self.config = config; self.key = jax.random.PRNGKey(config['seed']); self.params = {}
        self.should_shutdown = False; signal.signal(signal.SIGINT, self._handle_sigint)
        self.d_model = config['d_model']; self.num_devices = jax.local_device_count()
        self.num_patches_side = config['num_patches_side']; self.num_patches = self.num_patches_side ** 2
        self.models = {
            'patch_encoder': ImagePatchEncoder(patch_size=config['patch_size'], in_channels=config['channels'], embed_dim=self.d_model),
            'navigator': GalacticNavigator(d_model_total=self.d_model, num_patches=self.num_patches),
            'denoiser': HyperbolicDenoisingNetwork(d_model=self.d_model, num_layers=6, num_heads=8, mlp_dim=self.d_model*4),
            'patch_decoder': StableImagePatchDecoder(patch_size=config['patch_size'], out_channels=config['channels'], embed_dim=self.d_model)
        }
        self.diffusion = DiffusionProcess(timesteps=config['diffusion_timesteps'])

    def _handle_sigint(self, s, f): print("\n--- SIGINT received. Shutting down... ---"); self.should_shutdown = True

    def _init_params(self):
        if self.params: return
        print("--- Initializing model parameters on CPU... ---")
        with jax.default_device(CPU_DEVICE):
            dummy_image = jnp.zeros((1, self.config['image_size'], self.config['image_size'], 3), jnp.bfloat16)
            self.key, enc_key, nav_key, den_key, dec_key = jax.random.split(self.key, 5)
            self.params['patch_encoder'] = self.models['patch_encoder'].init(enc_key, dummy_image)['params']
            dummy_patch_grid = self.models['patch_encoder'].apply({'params': self.params['patch_encoder']}, dummy_image)
            dummy_patches_flat = dummy_patch_grid.reshape(1, -1, self.d_model)
            self.params['navigator'] = self.models['navigator'].init(nav_key, dummy_patches_flat)['params']
            dummy_time = jnp.array([0]); dummy_cond = jnp.zeros((1, self.d_model))
            self.params['denoiser'] = self.models['denoiser'].init(den_key, dummy_patches_flat, dummy_time, dummy_cond)['params']
            self.params['patch_decoder'] = self.models['patch_decoder'].init(dec_key, dummy_patch_grid)['params']
        print("--- CPU Initialization Complete. ---")

    def train(self, image_dir, epochs, batch_size, steps_per_epoch):
        dataset = create_dataset(image_dir, self.config['image_size'], batch_size)
        data_iterator = dataset.as_numpy_iterator()
        total_steps = epochs * steps_per_epoch
        warmup_steps = min(5000, int(0.1 * total_steps))
        lr_schedule_fn = optax.warmup_cosine_decay_schedule(1e-7, self.config['learning_rate'], warmup_steps, total_steps - warmup_steps, 1e-6)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_schedule_fn, weight_decay=1e-2, b1=0.9, b2=0.95))
        
        p_state, loss_controller, start_step = self.load_checkpoint(optimizer)
        if p_state is None:
            print("--- No checkpoint found. Initializing new training run. ---")
            self._init_params()
            state = train_state.TrainState.create(apply_fn=None, params=self.params, tx=optimizer)
            p_state = replicate(state)
            loss_priorities = {'triplet': 1.0, 'denoise': 2.0}
            loss_controller = DynamicLossController(loss_priorities)
            start_step = 0

        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,))
        def p_train_step(state, batch, key, weights):
            def loss_fn(params):
                anchor_img, positive_img, negative_img = batch
                B = anchor_img.shape[0]; 
                anchor_patches = self.models['patch_encoder'].apply({'params': params['patch_encoder']}, anchor_img).reshape(B, -1, self.d_model)
                positive_patches = self.models['patch_encoder'].apply({'params': params['patch_encoder']}, positive_img).reshape(B, -1, self.d_model)
                negative_patches = self.models['patch_encoder'].apply({'params': params['patch_encoder']}, negative_img).reshape(B, -1, self.d_model)
                all_patches = jnp.stack([anchor_patches, positive_patches, negative_patches])
                h_all = self.models['navigator'].apply({'params': params['navigator']}, all_patches)
                h_anchor, h_pos, h_neg = [jax.tree.map(lambda x: x[i], h_all) for i in range(3)]
                triplet_loss = 0.0; margin = 0.5
                for m in ['syn', 'sem', 'exe']:
                    ah_r, ah_i = jnp.mean(h_anchor[m][0], 1).astype(jnp.float32), jnp.mean(h_anchor[m][1], 1).astype(jnp.float32)
                    ph_r, ph_i = jnp.mean(h_pos[m][0], 1).astype(jnp.float32), jnp.mean(h_pos[m][1], 1).astype(jnp.float32)
                    nh_r, nh_i = jnp.mean(h_neg[m][0], 1).astype(jnp.float32), jnp.mean(h_neg[m][1], 1).astype(jnp.float32)
                    dist_pos = jnp.sum((ah_r - ph_r)**2 + (ah_i - ph_i)**2, axis=-1)
                    dist_neg = jnp.sum((ah_r - nh_r)**2 + (ah_i - nh_i)**2, axis=-1)
                    triplet_loss += jnp.mean(jnp.maximum(0, dist_pos - dist_neg + margin))
                key_t, key_noise = jax.random.split(key)
                t = jax.random.randint(key_t, (B,), 0, self.diffusion.timesteps)
                noise = jax.random.normal(key_noise, anchor_patches.shape)
                noisy_patches = self.diffusion.q_sample(jax.lax.stop_gradient(anchor_patches), t, noise)
                sem_r, sem_i = h_anchor['sem']
                cake_cond = jnp.concatenate([sem_r, sem_i], axis=-1).mean(axis=1)
                pred_noise = self.models['denoiser'].apply({'params': params['denoiser']}, noisy_patches, t, jax.lax.stop_gradient(cake_cond))
                denoising_loss = jnp.mean((pred_noise.astype(jnp.float32) - noise.astype(jnp.float32))**2)
                total_loss = (triplet_loss * weights['triplet']) + (denoising_loss * weights['denoise'])
                return total_loss, {'triplet': triplet_loss, 'denoise': denoising_loss}

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, axis_name='devices')
            new_state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss, 'triplet': aux['triplet'], 'denoise': aux['denoise'], 'grad_norm': optax.global_norm(grads), 'lr': lr_schedule_fn(state.step)}
            metrics = jax.lax.pmean(metrics, axis_name='devices')
            return new_state, metrics

        print("\n--- JIT Compiling unified training step... This will take a few minutes. ---")
        dummy_batch = next(data_iterator)
        sharded_dummy_batch = common_utils.shard(dummy_batch)
        self.key, *compile_keys = jax.random.split(self.key, self.num_devices + 1)
        sharded_keys = jnp.array(compile_keys)
        dummy_weights = replicate({'triplet': 1.0, 'denoise': 1.0})
        jit_start_time = time.time()
        p_state, _ = p_train_step(p_state, sharded_dummy_batch, sharded_keys, dummy_weights)
        jax.block_until_ready(p_state)
        print(f"--- Compilation finished in {time.time() - jit_start_time:.2f}s. Starting training loop. ---")

        global_step = start_step
        try:
            for epoch in range(start_step // steps_per_epoch, epochs):
                if self.should_shutdown: break
                print(f"\n--- Epoch {epoch+1}/{epochs} ---")
                
                last_metrics = {}
                with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}", initial=global_step % steps_per_epoch) as pbar:
                    for _ in range(steps_per_epoch - (global_step % steps_per_epoch)):
                        if self.should_shutdown: break
                        batch = next(data_iterator)
                        sharded_batch = common_utils.shard(batch)
                        self.key, *step_keys = jax.random.split(self.key, self.num_devices + 1)
                        sharded_keys = jnp.array(step_keys)
                        current_weights = loss_controller.update(last_metrics)
                        sharded_weights = replicate(current_weights)
                        p_state, metrics = p_train_step(p_state, sharded_batch, sharded_keys, sharded_weights)
                        
                        metrics = unreplicate(metrics)
                        metrics = {k: float(v) for k, v in metrics.items()}
                        last_metrics = {k: v for k, v in metrics.items() if k in loss_controller.loss_names}

                        if not np.isfinite(metrics['loss']):
                            pbar.write("\n[FATAL] Loss is NaN/Inf. Halting training.")
                            self.save_checkpoint(p_state, loss_controller, global_step)
                            return
                        
                        pbar.set_postfix(step=f"{global_step}/{total_steps}", loss=f"{metrics['loss']:.3f}", triplet=f"{metrics['triplet']:.3f}", denoise=f"{metrics['denoise']:.3f}", lr=f"{metrics['lr']:.1e}")
                        pbar.update(1)
                        global_step += 1
                
                if self.should_shutdown: break
                self.save_checkpoint(p_state, loss_controller, global_step)
        
        finally:
            if self.should_shutdown:
                print("\n--- Training interrupted. Saving final checkpoint... ---")
                self.save_checkpoint(p_state, loss_controller, global_step)
                print("--- Checkpoint saved. Exiting. ---")
                return

        print("\n--- Training finished. Saving final inference weights. ---")
        self.save_final_weights(p_state)

    def save_checkpoint(self, p_state, loss_controller, step):
        state_to_save = unreplicate(p_state)
        save_path = f"{self.config['basename']}.checkpoint.pkl"
        print(f"\n--- Saving checkpoint to {save_path} at step {step} ---")
        data_to_save = {
            'params': state_to_save.params,
            'opt_state': state_to_save.opt_state,
            'step': step,
            'loss_controller': loss_controller,
            'd_model': self.config['d_model']
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_checkpoint(self, optimizer):
        checkpoint_path = Path(f"{self.config['basename']}.checkpoint.pkl")
        if not checkpoint_path.exists():
            return None, None, 0
            
        print(f"--- Found checkpoint at {checkpoint_path}. Resuming training. ---")
        with open(checkpoint_path, 'rb') as f:
            saved_data = pickle.load(f)

        if saved_data.get('d_model') != self.config['d_model']:
            print(f"[FATAL] d_model mismatch! Checkpoint is for {saved_data.get('d_model')}, but config is for {self.config['d_model']}.")
            sys.exit(1)
        
        self.params = saved_data['params']
        restored_state = train_state.TrainState(
            step=saved_data['step'],
            apply_fn=None,
            params=self.params,
            tx=optimizer,
            opt_state=saved_data['opt_state']
        )
        
        return replicate(restored_state), saved_data['loss_controller'], saved_data['step']

    def save_final_weights(self, p_state):
        params = unreplicate(p_state).params
        save_path = f"{self.config['basename']}.weights.pkl"
        print(f"--- Saving final weights for inference to {save_path} ---")
        data_to_save = {'params': jax.device_get(params), 'd_model': self.config['d_model']}
        with open(save_path, 'wb') as f: pickle.dump(data_to_save, f)

def main():
    parser = argparse.ArgumentParser(description="GDF Trainer (WuBu Spheres v7.0)");
    parser.add_argument('command', choices=['convert-to-tfrecords', 'train'], default='train', nargs='?', help="Action to perform.")
    parser.add_argument('--image_dir', type=str, default="./images/", help="Path to image directory.")
    parser.add_argument('--basename', type=str, default="gdf_model", help="Basename for model files.")
    parser.add_argument('--d-model', type=int, default=256, help="Model embedding dimension.")
    parser.add_argument('--batch-size', type=int, default=None, help="Global batch size. Defaults to 4 * num_devices.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--learning-rate', type=float, default=2e-4, help="Peak learning rate for training.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    num_devices = jax.local_device_count()
    if args.batch_size is None:
        args.batch_size = max(1, 4 * num_devices)
        print(f"[INFO] --batch-size not set. Defaulting to {args.batch_size}.")
    patch_size = 32; image_size = 512
    MODEL_CONFIG = {
        'basename': f"{args.basename}_{args.d_model}d", 'image_size': image_size, 'patch_size': patch_size,
        'num_patches_side': image_size // patch_size, 'channels': 3, 'd_model': args.d_model,
        'diffusion_timesteps': 1000, 'learning_rate': args.learning_rate, 'seed': args.seed}
    print(f"--- GDF Trainer on {num_devices} device(s) | Mode: {args.command.upper()} ---")
    gdf = GalacticDiffusionFunnel(MODEL_CONFIG)
    dataset_info_path = Path(args.image_dir) / "dataset_info.pkl"
    if args.command == 'convert-to-tfrecords':
        convert_to_tfrecords(args.image_dir, MODEL_CONFIG['image_size'])
        return
    if args.command == 'train':
        if not dataset_info_path.exists():
            print(f"[FATAL] dataset_info.pkl not found. Run 'convert-to-tfrecords' first."); sys.exit(1)
        with open(dataset_info_path, "rb") as f: num_total_samples = pickle.load(f)['num_samples']
        print(f"--- Found {num_total_samples} samples in dataset info. ---")
        steps_per_epoch = num_total_samples // args.batch_size
        if steps_per_epoch == 0:
            print(f"[FATAL] Total samples ({num_total_samples}) is less than batch size ({args.batch_size}). Cannot train.")
            sys.exit(1)
        gdf.train(args.image_dir, args.epochs, args.batch_size, steps_per_epoch)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"); sys.exit(0)