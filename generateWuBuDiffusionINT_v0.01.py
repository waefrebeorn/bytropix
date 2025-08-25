# generate.py
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import numpy as np
import pickle
from typing import Any, Tuple, Dict, List
import sys
import argparse
from functools import partial
import time
from pathlib import Path
import torch
from PIL import Image
from collections import deque
from tqdm import tqdm, trange

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
except ImportError: print("[FATAL] `scikit-learn` not found."), sys.exit(1)
try:
    import clip
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError: print("[FATAL] `clip` and `torch` are required. Please `pip install git+https://github.com/openai/CLIP.git`"), sys.exit(1)


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


# --- MODEL DEFINITIONS (v9.0 - Sphere Alignment) ---
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
    def __call__(self, x, time_emb, cond_sequence, curvature):
        h_tangent = PoincareBall.logmap0(x, curvature)
        h_norm = nn.LayerNorm(dtype=jnp.float32, name="ln_1")(h_tangent)
        
        attn_out_tangent = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.d_model, dtype=self.dtype, name="self_attn"
        )(h_norm)
        
        attn_out_poincare = PoincareBall.expmap0(attn_out_tangent, curvature)
        x = PoincareBall.mobius_add(x, attn_out_poincare, curvature)

        h_tangent = PoincareBall.logmap0(x, curvature)
        h_norm_for_cross = nn.LayerNorm(dtype=jnp.float32, name="ln_2")(h_tangent)
        cond_norm = nn.LayerNorm(dtype=jnp.float32, name="ln_cond")(cond_sequence)

        cross_attn_out_tangent = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.d_model, dtype=self.dtype, name="cross_attn"
        )(inputs_q=h_norm_for_cross, inputs_kv=cond_norm)

        cross_attn_out_poincare = PoincareBall.expmap0(cross_attn_out_tangent, curvature)
        x = PoincareBall.mobius_add(x, cross_attn_out_poincare, curvature)

        h_tangent = PoincareBall.logmap0(x, curvature)
        h_norm_for_mlp = nn.LayerNorm(dtype=jnp.float32, name="ln_3")(h_tangent)

        time_cond_proj = nn.Dense(self.d_model, dtype=self.dtype)(nn.gelu(time_emb))
        h_norm_for_mlp = h_norm_for_mlp + time_cond_proj[:, None, :]
        
        mlp_out_tangent = nn.Dense(self.mlp_dim, dtype=self.dtype)(h_norm_for_mlp)
        mlp_out_tangent = nn.gelu(mlp_out_tangent)
        mlp_out_tangent = nn.Dense(self.d_model, dtype=self.dtype)(mlp_out_tangent)

        mlp_out_poincare = PoincareBall.expmap0(mlp_out_tangent, curvature)
        x = PoincareBall.mobius_add(x, mlp_out_poincare, curvature)
        
        return x

class HyperbolicDenoisingNetwork(nn.Module):
    d_model: int; num_layers: int; num_heads: int; mlp_dim: int
    
    @nn.compact
    def __call__(self, noisy_patches, time, cond_sequence):
        curvature = self.param('curvature', nn.initializers.constant(1.0), (1,), jnp.float32)
        curvature = nn.softplus(curvature) + 1e-7

        h = PoincareBall.project(noisy_patches)
        time_emb = SinusoidalPosEmb(self.d_model)(time)
        time_emb = nn.Dense(self.d_model)(time_emb)

        for i in range(self.num_layers):
            h = HyperbolicDenoisingBlock(
                d_model=self.d_model, num_heads=self.num_heads, mlp_dim=self.mlp_dim, name=f"block_{i}"
            )(h, time_emb, cond_sequence, curvature)
        
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
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

def _extract(a, t, x_shape):
    b, *_ = t.shape; res = a[t]; return res.reshape(b, *((1,) * (len(x_shape) - 1)))

# Class needed for unpickling checkpoint
class DynamicLossController:
    def __init__(self, target_weights: Dict[str, float], history_len: int = 100, adjustment_strength: float = 0.01, min_weight: float = 0.1):
        self.loss_names = list(target_weights.keys()); self.target_weights = target_weights; target_sum = sum(self.target_weights.values()); self.target_proportions = {name: w / target_sum for name, w in self.target_weights.items()}; self.weights = {name: 1.0 for name in self.loss_names}; self.loss_histories = {name: deque(maxlen=history_len) for name in self.loss_names}; self.strength = adjustment_strength; self.min_weight = min_weight

# --- DATA PIPELINE (for construct command) ---
def create_dataset_for_construct(image_dir: str, image_size: int, batch_size: int):
    record_file = Path(image_dir) / f"singles_{image_size}.tfrecord"
    if not record_file.exists():
        raise FileNotFoundError(f"TFRecord file not found. Run the trainer script with 'convert-to-tfrecords' command first.")
    print(f"--- Loading from optimized TFRecord: {record_file} ---")
    def _parse(proto):
        feature_desc = {'image': tf.io.FixedLenFeature([], tf.string)}
        img_bytes = tf.io.parse_single_example(proto, feature_desc)['image']
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img_for_model = (tf.cast(img, tf.float32) / 127.5) - 1.0
        img_for_model.set_shape([image_size, image_size, 3])
        return img_for_model, img
        
    ds = tf.data.TFRecordDataset(str(record_file), num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

class GDFInference:
    def __init__(self, config):
        self.config = config; self.key = jax.random.PRNGKey(config['seed']); self.params = {}
        self.d_model = config['d_model']
        self.num_patches_side = config['num_patches_side']; self.num_patches = self.num_patches_side ** 2
        self.models = {
            'patch_encoder': ImagePatchEncoder(patch_size=config['patch_size'], in_channels=config['channels'], embed_dim=self.d_model),
            'navigator': GalacticNavigator(d_model_total=self.d_model, num_patches=self.num_patches),
            'denoiser': HyperbolicDenoisingNetwork(d_model=self.d_model, num_layers=6, num_heads=8, mlp_dim=self.d_model*4),
            'patch_decoder': StableImagePatchDecoder(patch_size=config['patch_size'], out_channels=config['channels'], embed_dim=self.d_model)
        }
        self.diffusion = DiffusionProcess(timesteps=config['diffusion_timesteps'])
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=_clip_device)

    def _load_params(self):
        if self.params: return
        print("--- Loading model parameters from disk... ---")
        load_path = Path(f"{self.config['basename']}.weights.pkl")
        if not load_path.exists():
            load_path = Path(f"{self.config['basename']}.checkpoint.pkl")
        if not load_path.exists():
            print(f"[FATAL] No checkpoint or weights file found for basename '{self.config['basename']}'. Please train a model first.")
            sys.exit(1)
        
        print(f"--- Loading from: {load_path} ---")
        with open(load_path, 'rb') as f:
            saved_data = pickle.load(f)

        if saved_data.get('d_model') != self.config['d_model']:
            print(f"[FATAL] d_model mismatch! File is for {saved_data.get('d_model')}, but config is for {self.config['d_model']}.")
            sys.exit(1)
        
        self.params = saved_data.get('params')
        if not self.params:
            print(f"[FATAL] Could not find model parameters in {load_path}. Checkpoint might be from an old version."), sys.exit(1)
        print("--- Weights loaded successfully. ---")

    def construct(self, image_dir, batch_size, num_samples):
        self._load_params()

        @jax.jit
        def get_sem_sequence(params, img_batch):
            patches = self.models['patch_encoder'].apply({'params': params['patch_encoder']}, img_batch).reshape(img_batch.shape[0], -1, self.d_model)
            nav_states = self.models['navigator'].apply({'params': params['navigator']}, patches)
            sem_r, sem_i = nav_states['sem']
            return jnp.concatenate([sem_r, sem_i], axis=-1)

        dataset = create_dataset_for_construct(image_dir, self.config['image_size'], batch_size)
        
        all_sem_sequences = []
        all_clip_embeddings = []
        
        total_batches = (num_samples + batch_size - 1) // batch_size
        data_iterator = iter(dataset)

        for _ in tqdm(range(total_batches), desc="Baking cake", total=total_batches):
            try:
                model_batch, clip_batch_uint8 = next(data_iterator)
                sem_sequences = get_sem_sequence(self.params, model_batch.numpy())
                all_sem_sequences.append(jax.device_get(sem_sequences))
                
                pil_images = [Image.fromarray(img.numpy()) for img in clip_batch_uint8]
                processed_images = torch.stack([self.clip_preprocess(p) for p in pil_images]).to(_clip_device)
                with torch.no_grad():
                    embeddings = self.clip_model.encode_image(processed_images).cpu().numpy()
                all_clip_embeddings.append(embeddings)
            except tf.errors.OutOfRangeError:
                break

        sem_sequences_np = np.concatenate(all_sem_sequences)
        clip_embeddings_np = np.concatenate(all_clip_embeddings)
        
        print(f"--- Building NearestNeighbors index for {len(clip_embeddings_np)} CLIP embeddings... ---")
        clip_nn = NearestNeighbors(n_neighbors=16, algorithm='brute', metric='cosine')
        clip_nn.fit(clip_embeddings_np)
        
        cake_data = {'clip_nn': clip_nn, 'sem_sequences': sem_sequences_np}
        cake_path = f"{self.config['basename']}.cake.pkl"
        with open(cake_path, 'wb') as f:
            pickle.dump(cake_data, f)
        print(f"--- Funnel Cake saved to {cake_path} ---")

    def generate(self, prompt_text: str, num_samples: int, guidance_scale: float, steps: int):
        self._load_params()

        print(f"--- Generating: '{prompt_text}' | CFG: {guidance_scale} | Steps: {steps} ---")
        cake_path = Path(f"{self.config['basename']}.cake.pkl")
        if not cake_path.exists(): print(f"[ERROR] Cake file not found. Run 'construct' first."); return
        with open(cake_path, 'rb') as f: cake_data = pickle.load(f)

        print(f"--- Finding best match for prompt... ---")
        with torch.no_grad():
            text_features = self.clip_model.encode_text(clip.tokenize([prompt_text]).to(_clip_device)).cpu().numpy()
        
        distances, indices = cake_data['clip_nn'].kneighbors(text_features)
        best_match_idx = indices[0, 0]
        
        cond_sequence = jnp.array(cake_data['sem_sequences'][best_match_idx])
        cond_batch = jnp.stack([cond_sequence] * num_samples)
        
        # Create a zeroed sequence for unconditional guidance
        uncond_batch = jnp.zeros_like(cond_batch)
        
        full_cond_batch = jnp.concatenate([cond_batch, uncond_batch])
        self.key, sample_key = jax.random.split(self.key)
        latents = jax.random.normal(sample_key, (num_samples, self.num_patches, self.d_model))
        
        print("--- Starting sampling loop (No JIT)... ---")
        start_time = time.time()
        timesteps = jnp.arange(steps - 1, -1, -1)
        for i in trange(steps, desc="Sampling"):
            t = timesteps[i]
            t_batch = jnp.full((num_samples * 2,), t)
            
            combined_latents = jnp.concatenate([latents, latents])
            
            pred_noise_double = self.models['denoiser'].apply({'params': self.params['denoiser']}, combined_latents, t_batch, full_cond_batch)
            pred_noise_cond, pred_noise_uncond = jnp.split(pred_noise_double, 2)
            guided_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            
            t_batch_single = jnp.full((num_samples,), t)
            alpha_t = _extract(self.diffusion.alphas, t_batch_single, latents.shape)
            sqrt_one_minus_alpha_cumprod = _extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t_batch_single, latents.shape)
            pred_mean = (latents - (1. - alpha_t) / sqrt_one_minus_alpha_cumprod * guided_noise) / jnp.sqrt(alpha_t)
            
            if t > 0:
                posterior_variance_t = _extract(self.diffusion.posterior_variance, t_batch_single, latents.shape)
                self.key, step_key = jax.random.split(self.key)
                noise = jax.random.normal(step_key, latents.shape)
                latents = pred_mean + jnp.sqrt(posterior_variance_t) * noise
            else:
                latents = pred_mean
        
        print("--- Decoding latents (No JIT)... ---")
        num_patches_side = self.num_patches_side
        final_images = self.models['patch_decoder'].apply(
            {'params': self.params['patch_decoder']}, 
            latents.reshape(num_samples, num_patches_side, num_patches_side, -1)
        )
        final_images.block_until_ready()
        
        print(f"--- Generation executed in {time.time() - start_time:.2f}s ---")
        for i, img_tensor in enumerate(final_images):
            img_np = np.array((img_tensor * 0.5 + 0.5).clip(0,1) * 255, dtype=np.uint8)
            save_path = f"{prompt_text.replace(' ','_')[:30]}_{i}.png"; Image.fromarray(img_np).save(save_path)
            print(f"Image saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="GDF Inference Engine (WuBu Spheres v9.0)");
    parser.add_argument('command', choices=['construct', 'generate'], help="Action to perform.")
    parser.add_argument('--image_dir', type=str, default="./images/", help="Path to image directory for 'construct'.")
    parser.add_argument('--basename', type=str, default="gdf_model", help="Basename for model files.")
    parser.add_argument('--d-model', type=int, default=256, help="Model embedding dimension of the loaded weights.")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size for 'construct'. Defaults to 256.")
    parser.add_argument('--prompt', type=str, default="a beautiful landscape painting", help="Text prompt for generation.")
    parser.add_argument('--num-samples', type=int, default=1, help="Number of images to generate or use for cake construction.")
    parser.add_argument('--guidance-scale', '-g', type=float, default=7.5, help="Classifier-Free Guidance scale.")
    parser.add_argument('--steps', type=int, default=50, help="Number of denoising steps for generation.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed. If not set, a random seed will be used.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000)
        print(f"[INFO] No seed provided. Using random seed: {args.seed}")

    if args.batch_size is None:
        args.batch_size = 256
        if args.command == 'construct': print(f"[INFO] --batch-size not set. Defaulting to {args.batch_size}.")
    
    patch_size = 32; image_size = 512
    MODEL_CONFIG = {
        'basename': f"{args.basename}_{args.d_model}d", 'image_size': image_size, 'patch_size': patch_size,
        'num_patches_side': image_size // patch_size, 'channels': 3, 'd_model': args.d_model,
        'diffusion_timesteps': 1000, 'learning_rate': 0, 'seed': args.seed}
    
    print(f"--- GDF Inference on {jax.local_device_count()} device(s) | Mode: {args.command.upper()} ---")
    gdf_inference = GDFInference(MODEL_CONFIG)
    
    if args.command == 'construct':
        dataset_info_path = Path(args.image_dir) / "dataset_info.pkl"
        if not dataset_info_path.exists():
            print(f"[FATAL] dataset_info.pkl not found. Run 'convert-to-tfrecords' in train.py first."); sys.exit(1)
        with open(dataset_info_path, "rb") as f: num_total_samples = pickle.load(f)['num_samples']
        print(f"--- Found {num_total_samples} samples in dataset info. ---")
        num_to_construct = min(args.num_samples, num_total_samples) if args.num_samples > 1 else num_total_samples
        gdf_inference.construct(args.image_dir, args.batch_size, num_to_construct)
    elif args.command == 'generate':
        gdf_inference.generate(args.prompt, args.num_samples, args.guidance_scale, args.steps)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"); sys.exit(0)
