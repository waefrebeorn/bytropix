# =================================================================================================
#
#        TOPOLOGICAL AUTOENCODER: COMPRESSION & DECOMPRESSION UTILITY (PHASE 2.3)
#
# (Enhanced Small Image Handling: This version adds explicit user feedback when an
#  input image is smaller than the model's patch size, clarifying that it will be
#  padded to fit. The core logic robustly handles this case.)
#
# =================================================================================================

import os
# --- Environment Setup for JAX/TensorFlow ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Core Imports ---
import argparse
import pickle
import sys
from functools import partial
from pathlib import Path
from typing import Any, Tuple, Dict
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import jit
from PIL import Image
from tqdm import tqdm

# --- Additional Imports for Loading Checkpoint ---
from flax import struct
from flax.training import train_state
import chex # chex is used for type hints in the saved state objects

# Suppress TensorFlow GPU usage for this utility.
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass

# Auto-select the best JAX device.
if jax.devices("gpu"):
    print(f"--- JAX is using GPU: {jax.devices('gpu')[0].device_kind} ---")
elif jax.devices("tpu"):
     print(f"--- JAX is using TPU: {jax.devices('tpu')[0].device_kind} ---")
else:
    print("--- JAX is using CPU. Performance will be slower. ---")

# =================================================================================================
# 1. HELPER DEFINITIONS FOR LOADING THE CHECKPOINT
# =================================================================================================

class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; current_lr: jnp.ndarray
    exploration_rate: jnp.ndarray; step_count: jnp.ndarray; last_action_idx: jnp.ndarray; status_code: jnp.ndarray

class CustomTrainState(train_state.TrainState):
    ema_params: Any; q_controller_state: QControllerState

# =================================================================================================
# 2. MODEL DEFINITIONS (Copied directly from Trainer for perfect compatibility)
# =================================================================================================

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2); imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images_rgb: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x, features, current_dim, i = images_rgb, 32, self.input_image_size, 0
        context_vectors = []
        while (current_dim // 2) >= self.latent_grid_size and (current_dim // 2) > 0:
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            context_vectors.append(jnp.mean(x, axis=(1, 2)))
            features *= 2; current_dim //= 2; i += 1
        context_vector = jnp.concatenate(context_vectors, axis=-1)
        if current_dim != self.latent_grid_size:
            x = jax.image.resize(x, (x.shape[0], self.latent_grid_size, self.latent_grid_size, x.shape[-1]), 'bilinear')
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        path_params_raw = nn.Conv(3, (1, 1), name="path_params_head", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params_raw[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params_raw[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params_raw[..., 2]) * (jnp.pi / 2.0)
        path_params = jnp.stack([delta_c, chi_c, radius], axis=-1)
        return path_params, context_vector

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = path_params_grid.shape; L = H * W
        path_params = path_params_grid.reshape(B, L, 3)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
        chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path) + 1e-8
        path_real_mean = jnp.mean(t_co_steps.real, axis=-1); path_real_std = jnp.std(t_co_steps.real, axis=-1)
        path_imag_mean = jnp.mean(t_co_steps.imag, axis=-1); path_imag_std = jnp.std(t_co_steps.imag, axis=-1)
        complex_measurement = jnp.stack([path_real_mean, path_real_std, path_imag_mean, path_imag_std], axis=-1)
        feature_vectors = nn.Dense(self.d_model, name="feature_projector", dtype=self.dtype)(complex_measurement)
        return feature_vectors.reshape(B, H, W, self.d_model)

class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)

class FiLMLayer(nn.Module):
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x, context):
        context_proj = nn.Dense(x.shape[-1] * 2, name="context_proj", dtype=self.dtype)(context)
        gamma, beta = context_proj[..., :x.shape[-1]], context_proj[..., x.shape[-1]:]
        return x * (gamma[:, None, :] + 1) + beta[:, None, :]

class CoordinateDecoder(nn.Module):
    d_model: int; num_freqs: int = 10; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.remat
    def _mlp_block(self, h: jnp.ndarray, context_vector: jnp.ndarray) -> jnp.ndarray:
        film_layer = FiLMLayer(dtype=self.dtype)
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = film_layer(h, context_vector)
            h = nn.gelu(h)
        return nn.Dense(3, name="mlp_out", dtype=self.dtype, kernel_init=nn.initializers.zeros)(h)
    
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        B, H, W, _ = feature_grid.shape
        encoded_coords = PositionalEncoding(self.num_freqs, dtype=self.dtype)(coords)
        pyramid = [feature_grid] + [jax.image.resize(feature_grid, (B, H//(2**i), W//(2**i), self.d_model), 'bilinear') for i in range(1, 3)]
        all_sampled_features = []
        for level_grid in pyramid:
            level_shape = jnp.array(level_grid.shape[1:3], dtype=self.dtype)
            coords_rescaled = (coords + 1) / 2 * (level_shape - 1)
            def sample_one_image_level(single_level_grid):
                grid_chw = single_level_grid.transpose(2, 0, 1)
                return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw).T
            all_sampled_features.append(jax.vmap(sample_one_image_level)(level_grid))
        concatenated_features = jnp.concatenate(all_sampled_features, axis=-1)
        encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)
        mlp_input = jnp.concatenate([encoded_coords_tiled, concatenated_features], axis=-1)
        return nn.tanh(self._mlp_block(mlp_input, context_vector))

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(self.latent_grid_size, self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(self.d_model, name="coord_decoder", dtype=self.dtype)

    def encode(self, images_rgb: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.modulator(images_rgb)

    def decode(self, path_params: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        feature_grid = self.observer(path_params)
        pixels_rgb = self.coord_decoder(feature_grid, context_vector, coords)
        return pixels_rgb

# =================================================================================================
# 3. COMPRESSOR CLASS
# =================================================================================================

class Compressor:
    def __init__(self, model: TopologicalCoordinateGenerator, ema_params: Dict, args: argparse.Namespace):
        self.args = args
        self.model = model
        self.ema_params = ema_params
        self.patch_size = self.model.input_image_size
        self.dtype = self.model.dtype
        self._jit_functions()

    def _jit_functions(self):
        print("--- JIT-compiling model functions (one-time cost)... ---")
        self._encode_jit = jit(partial(self.model.apply, {'params': self.ema_params}, method=self.model.encode))
        self._decode_batch_jit = jit(partial(self.model.apply, {'params': self.ema_params}, method=self.model.decode))
        print("--- Compilation complete. ---")

    def compress_image(self, image_path: str, output_path: str):
        print(f"--- 🖼️  Compressing '{image_path}'... ---")
        try:
            img = Image.open(image_path).convert('RGB')
            original_width, original_height = img.size
        except FileNotFoundError:
            print(f"[FATAL] Input image not found at '{image_path}'.")
            sys.exit(1)

        # --- NEW: Add explicit user feedback for small images ---
        if original_width < self.patch_size or original_height < self.patch_size:
            print(f"--- NOTE: Input ({original_width}x{original_height}) is smaller than model patch size ({self.patch_size}x{self.patch_size}). Padding to fit. ---")
        
        img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0
        num_patches_y = math.ceil(original_height / self.patch_size)
        num_patches_x = math.ceil(original_width / self.patch_size)
        padded_height = num_patches_y * self.patch_size
        padded_width = num_patches_x * self.patch_size
        pad_y = padded_height - original_height
        pad_x = padded_width - original_width
        img_padded = np.pad(img_np, ((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
        
        print(f"--- Processing image as a {num_patches_x}x{num_patches_y} grid of {self.patch_size}x{self.patch_size} patches... ---")
        all_path_params, all_context_vectors = [], []
        for i in tqdm(range(num_patches_y), desc="Encoding Patches (Y)"):
            for j in range(num_patches_x):
                patch = img_padded[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, :]
                patch_batch = jnp.expand_dims(patch, axis=0).astype(self.dtype)
                path_params, context_vector = self._encode_jit(patch_batch)
                all_path_params.append(jax.device_get(path_params[0]))
                all_context_vectors.append(jax.device_get(context_vector[0]))

        print("--- Quantizing latent data for compression... ---")
        path_params_grid = np.array(all_path_params)
        context_vector_grid = np.array(all_context_vectors)
        delta_c_q = np.round((path_params_grid[..., 0] / np.pi + 1.0) * 32767.5).astype(np.uint16)
        chi_c_q = np.round((path_params_grid[..., 1] / (np.pi / 4.0) + 1.0) * 32767.5).astype(np.uint16)
        radius_q = np.round((path_params_grid[..., 2] / (np.pi / 2.0)) * 65535.0).astype(np.uint16)
        quantized_path_params = np.stack([delta_c_q, chi_c_q, radius_q], axis=-1)
        context_min, context_max = context_vector_grid.min(), context_vector_grid.max()
        quantized_context = np.round((context_vector_grid - context_min) / (context_max - context_min + 1e-8) * 65535.0).astype(np.uint16)

        compressed_data = {
            'model_config': {
                'd_model': self.model.d_model,
                'latent_grid_size': self.model.latent_grid_size,
                'image_size': self.model.input_image_size,
            },
            'original_size': (original_width, original_height),
            'patch_grid_size': (num_patches_x, num_patches_y),
            'quantized_path_params': quantized_path_params,
            'quantized_context': quantized_context,
            'context_min': np.float32(context_min),
            'context_max': np.float32(context_max),
        }

        print(f"--- 💾 Saving compressed file to '{output_path}'... ---")
        with open(output_path, 'wb') as f: pickle.dump(compressed_data, f)
        original_size_bytes = Path(image_path).stat().st_size
        compressed_size_bytes = Path(output_path).stat().st_size
        print(f"--- ✅ Compression successful! Ratio: {original_size_bytes / compressed_size_bytes:.2f}x ({original_size_bytes/1024:.1f} KB -> {compressed_size_bytes/1024:.1f} KB) ---")

    def decompress_image(self, compressed_path: str, output_path: str):
        print(f"--- 🗜️  Decompressing '{compressed_path}'... ---")
        try:
            with open(compressed_path, 'rb') as f:
                compressed_data = pickle.load(f)
        except FileNotFoundError: print(f"[FATAL] Compressed file not found at '{compressed_path}'."); sys.exit(1)
            
        print("--- De-quantizing latent data... ---")
        original_width, original_height = compressed_data['original_size']
        num_patches_x, num_patches_y = compressed_data['patch_grid_size']
        q_path = compressed_data['quantized_path_params'].astype(np.float32)
        delta_c = (q_path[..., 0] / 32767.5 - 1.0) * np.pi
        chi_c = (q_path[..., 1] / 32767.5 - 1.0) * (np.pi / 4.0)
        radius = (q_path[..., 2] / 65535.0) * (np.pi / 2.0)
        path_params_grid = np.stack([delta_c, chi_c, radius], axis=-1)
        q_context = compressed_data['quantized_context'].astype(np.float32)
        context_min, context_max = compressed_data['context_min'], compressed_data['context_max']
        context_vector_grid = (q_context / 65535.0) * (context_max - context_min) + context_min

        print(f"--- Reconstructing {num_patches_x}x{num_patches_y} patch grid... ---")
        padded_height = num_patches_y * self.patch_size
        padded_width = num_patches_x * self.patch_size
        coords = jnp.mgrid[-1:1:self.patch_size*1j, -1:1:self.patch_size*1j].transpose(1, 2, 0).reshape(-1, 2)
        coord_batch_size = 65536
        num_coord_batches = math.ceil(coords.shape[0] / coord_batch_size)
        full_image_canvas = np.zeros((padded_height, padded_width, 3), dtype=np.float32)

        for k in tqdm(range(num_patches_y * num_patches_x), desc="Decoding Patches"):
            path_params_batch = jnp.expand_dims(path_params_grid[k], axis=0).astype(self.dtype)
            context_vector_batch = jnp.expand_dims(context_vector_grid[k], axis=0).astype(self.dtype)
            
            pixel_batches = []
            for c_idx in range(num_coord_batches):
                coord_batch = coords[c_idx * coord_batch_size:(c_idx + 1) * coord_batch_size]
                pixel_batches.append(jax.device_get(self._decode_batch_jit(path_params_batch, context_vector_batch, coord_batch)))
            
            decoded_patch = np.concatenate([p[0] for p in pixel_batches], axis=0).reshape(self.patch_size, self.patch_size, 3)
            i, j = k // num_patches_x, k % num_patches_x
            full_image_canvas[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, :] = decoded_patch
        
        print("--- Finalizing image... ---")
        final_image = full_image_canvas[:original_height, :original_width, :]
        img_out_np = np.array((final_image * 0.5 + 0.5).clip(0, 1) * 255.0, dtype=np.uint8)
        print(f"--- ✨ Saving reconstructed image to '{output_path}'... ---")
        Image.fromarray(img_out_np).save(output_path)
        print("--- ✅ Decompression successful! ---")

def load_model_weights(args: argparse.Namespace) -> Tuple[Dict, str]:
    """Loads EMA parameters from a checkpoint."""
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        # Construct filename based on model params which are now present in args for both compress and decompress
        base_name_str = f"{args.basename}_{args.d_model}d_{args.image_size}"
        ckpt_path_best = Path(f"{base_name_str}_best.pkl")
        ckpt_path_final = Path(f"{base_name_str}.pkl")
        if ckpt_path_best.exists(): ckpt_path = ckpt_path_best
        elif ckpt_path_final.exists(): ckpt_path = ckpt_path_final
        else: print(f"[FATAL] No checkpoint found for basename '{args.basename}' with specified model config. Searched for '{ckpt_path_best}' and '{ckpt_path_final}'."); sys.exit(1)

    print(f"--- Loading model weights from: {ckpt_path} ---")
    try:
        with open(ckpt_path, 'rb') as f: data = pickle.load(f)
        ema_params = data.get('ema_params')
        if ema_params is None: print("[FATAL] 'ema_params' not found in checkpoint."); sys.exit(1)
        dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        return jax.tree_util.tree_map(lambda x: x.astype(dtype), ema_params), dtype
    except Exception as e: print(f"[FATAL] Failed to load checkpoint. Error: {e}"); sys.exit(1)

# =================================================================================================
# 4. MAIN EXECUTION BLOCK
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Topological AE Compressor/Decompressor Utility (v2.3 - Robust Small Image Handling)")
    subparsers = parser.add_subparsers(dest='command', required=True)

    weights_parser = argparse.ArgumentParser(add_help=False)
    weights_parser.add_argument('--basename', type=str, help="Basename of the trained model file (e.g., 'my_modelv2'). Required if --ckpt-path is not set.")
    weights_parser.add_argument('--ckpt-path', type=str, default=None, help="Direct path to the .pkl checkpoint file (overrides basename).")
    weights_parser.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 precision for model weights (must match training).")

    p_comp = subparsers.add_parser('compress', help="Compress an image of any size to a self-describing file.", parents=[weights_parser])
    p_comp.add_argument('--d-model', type=int, required=True, help="Model dimension used during training (e.g., 64).")
    p_comp.add_argument('--latent-grid-size', type=int, required=True, help="Latent grid size used during training (e.g., 64).")
    p_comp.add_argument('--image-size', type=int, required=True, help="The image resolution the model was trained on, used as the patch size (e.g., 512).")
    p_comp.add_argument('--image-path', type=str, required=True, help="Path to the input image (PNG, JPG, etc.).")
    p_comp.add_argument('--output-path', type=str, required=True, help="Path to save the compressed file (e.g., image.wubu).")

    p_dcomp = subparsers.add_parser('decompress', help="Decompress a file back to a full-resolution image.", parents=[weights_parser])
    p_dcomp.add_argument('--compressed-path', type=str, required=True, help="Path to the input compressed file.")
    p_dcomp.add_argument('--output-path', type=str, required=True, help="Path to save the reconstructed PNG image.")
    
    args = parser.parse_args()

    if args.command == 'compress':
        if not args.basename and not args.ckpt_path: parser.error("For 'compress', either --basename or --ckpt-path is required.")
        ema_params, dtype = load_model_weights(args)
        model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=dtype)
        compressor = Compressor(model, ema_params, args)
        compressor.compress_image(args.image_path, args.output_path)

    elif args.command == 'decompress':
        print("--- Reading metadata from compressed file... ---")
        try:
            with open(args.compressed_path, 'rb') as f: compressed_data = pickle.load(f)
            model_config = compressed_data['model_config']
            print(f"--- File metadata found: {model_config} ---")
        except Exception as e: print(f"[FATAL] Could not read metadata from '{args.compressed_path}'. Is it a valid .wubu file? Error: {e}"); sys.exit(1)
        
        args.d_model, args.latent_grid_size, args.image_size = model_config['d_model'], model_config['latent_grid_size'], model_config['image_size']
        if not args.basename and not args.ckpt_path: parser.error("For 'decompress', either --basename or --ckpt-path is required to load model weights.")
        
        ema_params, dtype = load_model_weights(args)
        model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=dtype)
        compressor = Compressor(model, ema_params, args)
        compressor.decompress_image(args.compressed_path, args.output_path)

if __name__ == "__main__":
    main()