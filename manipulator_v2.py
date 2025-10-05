# =================================================================================================
#
#        TOPOLOGICAL AUTOENCODER: LATENT SPACE MANIPULATOR (V2 - BATCHED VIDEO)
#
# (This utility provides tools for the deterministic manipulation of an image's latent
#  blueprint. It can interpolate between images, transfer lighting and color moods, and
#  surgically graft textures from one image onto another.)
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
from typing import Any, Tuple, Dict, List
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import jit
from PIL import Image
from tqdm import tqdm

# --- Additional Imports for Loading Checkpoint & Video Output ---
from flax import struct
from flax.training import train_state
import chex
import imageio  # Added for MP4 output

# Suppress TensorFlow GPU usage for this utility.
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    pass

if jax.devices("gpu"):
    print(f"--- JAX is using GPU: {jax.devices('gpu')[0].device_kind} ---")
else:
    print("--- JAX is using CPU. Performance will be slower. ---")

# =================================================================================================
# 1. HELPER DEFINITIONS & MODEL ARCHITECTURE (For compatibility with checkpoints)
# =================================================================================================

class QControllerState(struct.PyTreeNode):
    q_table: chex.Array; metric_history: chex.Array; current_lr: jnp.ndarray
    exploration_rate: jnp.ndarray; step_count: jnp.ndarray; last_action_idx: jnp.ndarray; status_code: jnp.ndarray

class CustomTrainState(train_state.TrainState):
    ema_params: Any; q_controller_state: QControllerState

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

    def decode(self, path_params: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        feature_grid = self.observer(path_params)
        pixels_rgb = self.coord_decoder(feature_grid, context_vector, coords)
        return pixels_rgb

# =================================================================================================
# 2. LATENT MANIPULATOR CLASS
# =================================================================================================

class LatentManipulator:
    def __init__(self, model: TopologicalCoordinateGenerator, ema_params: Dict, args: argparse.Namespace):
        self.args = args
        self.model = model
        self.ema_params = ema_params
        self.patch_size = self.model.input_image_size
        self.dtype = self.model.dtype
        self._jit_functions()

    def _jit_functions(self):
        print("--- JIT-compiling model functions (one-time cost)... ---")
        self._decode_batch_jit = jit(partial(self.model.apply, {'params': self.ema_params}, method=self.model.decode))
        print("--- Compilation complete. ---")

    def _load_and_dequantize_wubu(self, wubu_path: str) -> Dict[str, Any]:
        """Loads a .wubu file and returns its dequantized latent data."""
        print(f"--- Loading and dequantizing '{wubu_path}'... ---")
        try:
            with open(wubu_path, 'rb') as f:
                compressed_data = pickle.load(f)
        except FileNotFoundError:
            print(f"[FATAL] Compressed file not found at '{wubu_path}'."); sys.exit(1)
            
        q_path = compressed_data['quantized_path_params'].astype(np.float32)
        delta_c = (q_path[..., 0] / 32767.5 - 1.0) * np.pi
        chi_c = (q_path[..., 1] / 32767.5 - 1.0) * (np.pi / 4.0)
        radius = (q_path[..., 2] / 65535.0) * (np.pi / 2.0)
        path_params_grid = np.stack([delta_c, chi_c, radius], axis=-1)
        q_context = compressed_data['quantized_context'].astype(np.float32)
        context_min, context_max = compressed_data['context_min'], compressed_data['context_max']
        context_vector_grid = (q_context / 65535.0) * (context_max - context_min) + context_min

        return {
            "path_params": path_params_grid,
            "context_vectors": context_vector_grid,
            "original_size": compressed_data['original_size'],
            "patch_grid_size": compressed_data['patch_grid_size']
        }

    def _reconstruct_from_latents(self, path_params_grid, context_vector_grid, original_size, patch_grid_size, desc="Decoding Patches") -> np.ndarray:
        """Reconstructs a single image from the provided latent grids."""
        original_width, original_height = original_size
        num_patches_x, num_patches_y = patch_grid_size
        padded_height = num_patches_y * self.patch_size
        padded_width = num_patches_x * self.patch_size
        coords = jnp.mgrid[-1:1:self.patch_size*1j, -1:1:self.patch_size*1j].transpose(1, 2, 0).reshape(-1, 2)
        coord_batch_size = 65536
        num_coord_batches = math.ceil(coords.shape[0] / coord_batch_size)
        full_image_canvas = np.zeros((padded_height, padded_width, 3), dtype=np.float32)

        for k in tqdm(range(num_patches_y * num_patches_x), desc=desc, leave=False):
            context_vector = context_vector_grid[k] if context_vector_grid.ndim > 1 else context_vector_grid
            path_params_batch = jnp.expand_dims(path_params_grid[k], axis=0).astype(self.dtype)
            context_vector_batch = jnp.expand_dims(context_vector, axis=0).astype(self.dtype)
            
            pixel_batches = [jax.device_get(self._decode_batch_jit(path_params_batch, context_vector_batch, coords[c_idx * coord_batch_size:(c_idx + 1) * coord_batch_size])) for c_idx in range(num_coord_batches)]
            decoded_patch = np.concatenate([p[0] for p in pixel_batches], axis=0).reshape(self.patch_size, self.patch_size, 3)
            i, j = k // num_patches_x, k % num_patches_x
            full_image_canvas[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, :] = decoded_patch
        
        return full_image_canvas[:original_height, :original_width, :]

    def _reconstruct_batch_from_latents(self, batched_path_params: np.ndarray, batched_context_vectors: np.ndarray, batched_original_sizes: List[Tuple[int, int]], patch_grid_size: Tuple[int, int], desc: str) -> List[np.ndarray]:
        """Reconstructs a BATCH of images from batched latent grids for high performance."""
        batch_size = batched_path_params.shape[0]
        num_patches_x, num_patches_y = patch_grid_size
        
        # Prepare coordinates once for all patches
        coords = jnp.mgrid[-1:1:self.patch_size*1j, -1:1:self.patch_size*1j].transpose(1, 2, 0).reshape(-1, 2)
        coord_batch_size = 65536
        num_coord_batches = math.ceil(coords.shape[0] / coord_batch_size)
        
        # Prepare a canvas for each image in the batch
        canvases = [np.zeros((num_patches_y * self.patch_size, num_patches_x * self.patch_size, 3), dtype=np.float32) for _ in range(batch_size)]

        # Iterate through patches, but process all images in the batch for that patch simultaneously
        for k in tqdm(range(num_patches_y * num_patches_x), desc=desc, leave=False):
            # Select the k-th patch from ALL images in the batch
            path_params_for_patch_k = jnp.asarray(batched_path_params[:, k, ...], dtype=self.dtype)
            context_vectors_for_patch_k = jnp.asarray(batched_context_vectors[:, k, ...], dtype=self.dtype)
            
            # Decode the batch of patches (one from each image)
            pixel_batches = [jax.device_get(self._decode_batch_jit(path_params_for_patch_k, context_vectors_for_patch_k, coords[c_idx*coord_batch_size:(c_idx+1)*coord_batch_size])) for c_idx in range(num_coord_batches)]
            
            # The result is already batched, shape (B, num_pixels, 3)
            decoded_patches_flat = np.concatenate(pixel_batches, axis=1)
            decoded_patches = decoded_patches_flat.reshape(batch_size, self.patch_size, self.patch_size, 3)

            # Place each decoded patch onto its corresponding canvas
            i, j = k // num_patches_x, k % num_patches_x
            for b in range(batch_size):
                canvases[b][i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, :] = decoded_patches[b]

        # Crop each canvas to its final interpolated size
        final_images = [canvas[:h, :w, :] for canvas, (w, h) in zip(canvases, batched_original_sizes)]
        return final_images

    def _save_image(self, image_array, output_path):
        """Saves a numpy array as a PNG image."""
        print(f"--- Saving image to '{output_path}'... ---")
        img_out_np = np.array((image_array * 0.5 + 0.5).clip(0, 1) * 255.0, dtype=np.uint8)
        Image.fromarray(img_out_np).save(output_path)
    
    def _canvas_resize_latents(self, source_data, target_data, canvas_nx, canvas_ny):
        """Correctly places a source latent grid onto a larger canvas, padding with target data."""
        src_nx, src_ny = source_data["patch_grid_size"]
        
        # --- Handle path_params (high-dimensional tensors) ---
        latent_h, latent_w, latent_c = source_data["path_params"].shape[1:]
        
        # Create canvas from target data, then reshape into a grid of patches
        params_canvas = np.copy(target_data["path_params"])
        params_canvas_gridded = params_canvas.reshape((canvas_ny, canvas_nx, latent_h, latent_w, latent_c))
        
        # Reshape source to grid-of-patches
        source_params_gridded = source_data["path_params"].reshape((src_ny, src_nx, latent_h, latent_w, latent_c))
        
        # Place source onto canvas
        params_canvas_gridded[:src_ny, :src_nx, :, :, :] = source_params_gridded
        
        # Reshape back to flat list of patches for the renderer
        final_params = params_canvas_gridded.reshape((-1, latent_h, latent_w, latent_c))

        # --- Handle context_vectors (1D vectors) ---
        context_dim = source_data["context_vectors"].shape[-1]
        
        # Create canvas and reshape
        context_canvas = np.copy(target_data["context_vectors"])
        context_canvas_gridded = context_canvas.reshape((canvas_ny, canvas_nx, context_dim))
        
        # Reshape source
        source_context_gridded = source_data["context_vectors"].reshape((src_ny, src_nx, context_dim))
        
        # Place source onto canvas
        context_canvas_gridded[:src_ny, :src_nx, :] = source_context_gridded
        
        # Reshape back to flat list
        final_contexts = context_canvas_gridded.reshape((-1, context_dim))

        return final_params, final_contexts



    def perform_interpolation(self):
        """Generates an MP4 video by interpolating between two latent spaces using high-performance batching."""
        data_a = self._load_and_dequantize_wubu(self.args.start_wubu)
        data_b = self._load_and_dequantize_wubu(self.args.end_wubu)

        nx_a, ny_a = data_a["patch_grid_size"]
        nx_b, ny_b = data_b["patch_grid_size"]
        
        canvas_nx, canvas_ny = max(nx_a, nx_b), max(ny_a, ny_b)
        canvas_grid_size = (canvas_nx, canvas_ny)
        
        params_a, contexts_a = data_a["path_params"], data_a["context_vectors"]
        params_b, contexts_b = data_b["path_params"], data_b["context_vectors"]

        if (nx_a, ny_a) != canvas_grid_size:
            print(f"--- Resizing start latents from {nx_a}x{ny_a} to {canvas_nx}x{canvas_ny} canvas... ---")
            params_a, contexts_a = self._canvas_resize_latents(data_a, data_b, canvas_nx, canvas_ny)

        if (nx_b, ny_b) != canvas_grid_size:
            print(f"--- Resizing end latents from {nx_b}x{ny_b} to {canvas_nx}x{canvas_ny} canvas... ---")
            params_b, contexts_b = self._canvas_resize_latents(data_b, data_a, canvas_nx, canvas_ny)

        # --- START OF FIX ---
        # 1. Determine the maximum dimensions from both images
        max_w = max(data_a["original_size"][0], data_b["original_size"][0])
        max_h = max(data_a["original_size"][1], data_b["original_size"][1])
        
        # 2. Round up to be divisible by 16 for video codec compatibility
        # This prevents the `imageio` warning and ensures a fixed frame size.
        VIDEO_CODEC_DIVISOR = 16
        output_w = math.ceil(max_w / VIDEO_CODEC_DIVISOR) * VIDEO_CODEC_DIVISOR
        output_h = math.ceil(max_h / VIDEO_CODEC_DIVISOR) * VIDEO_CODEC_DIVISOR
        print(f"--- Unified video output resolution set to {output_w}x{output_h} for compatibility. ---")
        # --- END OF FIX ---

        output_path = Path(self.args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"--- Generating {self.args.frames} frames for '{output_path}' (batch size: {self.args.batch_size})... ---")

        with imageio.get_writer(output_path, fps=self.args.fps, macro_block_size=None) as writer: # Set macro_block_size to None as we handle it
            pbar = tqdm(total=self.args.frames, desc="Rendering Video Frames")
            for i in range(0, self.args.frames, self.args.batch_size):
                current_batch_size = min(self.args.batch_size, self.args.frames - i)
                if current_batch_size <= 0: continue

                t_values = np.linspace(i / (self.args.frames - 1), (i + current_batch_size - 1) / (self.args.frames - 1), current_batch_size)
                
                t_params_bcast = t_values.reshape(-1, 1, 1, 1, 1)
                t_context_bcast = t_values.reshape(-1, 1, 1)
                interp_path_params = (1 - t_params_bcast) * params_a[None, ...] + t_params_bcast * params_b[None, ...]
                interp_context_vectors = (1 - t_context_bcast) * contexts_a[None, ...] + t_context_bcast * contexts_b[None, ...]
                
                interp_sizes = [
                    (
                        int((1 - t) * data_a["original_size"][0] + t * data_b["original_size"][0]),
                        int((1 - t) * data_a["original_size"][1] + t * data_b["original_size"][1])
                    ) for t in t_values
                ]
                
                desc = f"Batch {i//self.args.batch_size + 1}"
                reconstructed_images = self._reconstruct_batch_from_latents(
                    interp_path_params, interp_context_vectors,
                    interp_sizes, canvas_grid_size, desc=desc
                )

                # --- START OF FIX ---
                # 3. Place each rendered frame onto a fixed-size canvas before writing
                for image_array in reconstructed_images:
                    # Create the final canvas for this frame (float for precision placement)
                    frame_canvas = np.zeros((output_h, output_w, 3), dtype=np.float32)
                    
                    # Get the current image's actual dimensions
                    h, w, _ = image_array.shape
                    
                    # Calculate top-left corner for centering
                    y_offset = (output_h - h) // 2
                    x_offset = (output_w - w) // 2
                    
                    # Place the rendered image on the canvas
                    frame_canvas[y_offset:y_offset+h, x_offset:x_offset+w, :] = image_array
                    
                    # Convert the final canvas to uint8 and write to video
                    img_uint8 = np.array((frame_canvas * 0.5 + 0.5).clip(0, 1) * 255.0, dtype=np.uint8)
                    writer.append_data(img_uint8)
                # --- END OF FIX ---
                pbar.update(current_batch_size)
            pbar.close()

        print(f"--- ✅ MP4 video saved! BOOM. ---")



    def perform_mood_transfer(self):
        """Applies the lighting/color mood from one image to the content of another."""
        content_data = self._load_and_dequantize_wubu(self.args.content_wubu)
        mood_data = self._load_and_dequantize_wubu(self.args.mood_wubu)
        alpha = self.args.alpha

        global_mood_vector = np.mean(mood_data["context_vectors"], axis=0)
        print(f"--- Extracted global mood vector. Blending with alpha={alpha}... ---")

        original_contexts = content_data["context_vectors"]
        blended_contexts = (1 - alpha) * original_contexts + alpha * global_mood_vector
        
        reconstructed_image = self._reconstruct_from_latents(
            content_data["path_params"], blended_contexts,
            content_data["original_size"], content_data["patch_grid_size"],
            desc="Decoding with new mood"
        )
        self._save_image(reconstructed_image, self.args.output_path)
        print("--- ✅ Mood transfer complete! ---")

    def perform_texture_graft(self):
        """Grafts a texture from a source image onto a region of a base image."""
        base_data = self._load_and_dequantize_wubu(self.args.base_wubu)
        source_data = self._load_and_dequantize_wubu(self.args.source_wubu)
        alpha = self.args.alpha
        
        try:
            px, py, pw, ph = [int(x) for x in self.args.rect.split(',')]
        except Exception:
            print(f"[FATAL] Invalid --rect format. Use 'x,y,width,height'."); sys.exit(1)

        modified_path_params = np.copy(base_data["path_params"])
        modified_context_vectors = np.copy(base_data["context_vectors"])
        num_patches_x, _ = base_data["patch_grid_size"]

        source_path_params = source_data["path_params"][0]
        source_context_vector = source_data["context_vectors"][0]
        
        start_j, end_j = px // self.patch_size, (px + pw - 1) // self.patch_size
        start_i, end_i = py // self.patch_size, (py + ph - 1) // self.patch_size
        
        print(f"--- Grafting texture onto patch grid from Y:{start_i}-{end_i}, X:{start_j}-{end_j}... ---")
        
        for i in range(start_i, end_i + 1):
            for j in range(start_j, end_j + 1):
                idx = i * num_patches_x + j
                if idx < len(modified_path_params):
                    modified_path_params[idx] = source_path_params
                    base_context = modified_context_vectors[idx]
                    modified_context_vectors[idx] = (1 - alpha) * base_context + alpha * source_context_vector
        
        reconstructed_image = self._reconstruct_from_latents(
            modified_path_params, modified_context_vectors,
            base_data["original_size"], base_data["patch_grid_size"],
            desc="Decoding grafted texture"
        )
        self._save_image(reconstructed_image, self.args.output_path)
        print("--- ✅ Texture graft complete! ---")

# =================================================================================================
# 3. UTILITY FUNCTIONS AND MAIN EXECUTION BLOCK
# =================================================================================================

def load_model_weights(args: argparse.Namespace) -> Tuple[Dict, str]:
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        base_name_str = f"{args.basename}_{args.d_model}d_{args.image_size}"
        ckpt_path_best = Path(f"{base_name_str}_best.pkl")
        ckpt_path_final = Path(f"{base_name_str}.pkl")
        if ckpt_path_best.exists(): ckpt_path = ckpt_path_best
        elif ckpt_path_final.exists(): ckpt_path = ckpt_path_final
        else: print(f"[FATAL] No checkpoint found. Searched for '{ckpt_path_best}' and '{ckpt_path_final}'."); sys.exit(1)

    print(f"--- Loading model weights from: {ckpt_path} ---")
    try:
        with open(ckpt_path, 'rb') as f: data = pickle.load(f)
        ema_params = data.get('ema_params')
        if ema_params is None: print("[FATAL] 'ema_params' not found in checkpoint."); sys.exit(1)
        dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        return jax.tree_util.tree_map(lambda x: x.astype(dtype), ema_params), dtype
    except Exception as e: print(f"[FATAL] Failed to load checkpoint. Error: {e}"); sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Topological AE Latent Space Manipulator (V2 - Batched Video)")
    subparsers = parser.add_subparsers(dest='command', required=True)
    weights_parser = argparse.ArgumentParser(add_help=False)
    weights_parser.add_argument('--basename', type=str, required=True, help="Basename of the trained model file (e.g., 'my_modelv2').")
    weights_parser.add_argument('--ckpt-path', type=str, default=None, help="Direct path to the .pkl checkpoint file (overrides basename).")
    weights_parser.add_argument('--use-bfloat16', action='store_true', help="Use BFloat16 precision.")

    p_interp = subparsers.add_parser('interpolate', help="Smoothly interpolate between two images and output an MP4.", parents=[weights_parser])
    p_interp.add_argument('--start-wubu', type=str, required=True, help="The starting .wubu file.")
    p_interp.add_argument('--end-wubu', type=str, required=True, help="The ending .wubu file.")
    p_interp.add_argument('--frames', type=int, default=120, help="Number of frames to generate for the video.")
    p_interp.add_argument('--fps', type=int, default=30, help="Frames per second for the output video.")
    p_interp.add_argument('--batch-size', type=int, default=16, help="Number of frames to render simultaneously. Increase for better GPU utilization.")
    p_interp.add_argument('--output-path', type=str, required=True, help="Path to save the output .mp4 video.")

    p_mood = subparsers.add_parser('mood-transfer', help="Transfer lighting/atmosphere from one image to another.", parents=[weights_parser])
    p_mood.add_argument('--content-wubu', type=str, required=True, help="Path to the .wubu file providing the structure/content.")
    p_mood.add_argument('--mood-wubu', type=str, required=True, help="Path to the .wubu file providing the mood (lighting, color).")
    p_mood.add_argument('--alpha', type=float, default=1.0, help="Blend factor for the mood (0.0=original, 1.0=full mood).")
    p_mood.add_argument('--output-path', type=str, required=True, help="Path to save the resulting PNG image.")

    p_graft = subparsers.add_parser('texture-graft', help="Graft a texture onto a region of a base image.", parents=[weights_parser])
    p_graft.add_argument('--base-wubu', type=str, required=True, help="The base .wubu file to be edited.")
    p_graft.add_argument('--source-wubu', type=str, required=True, help="The .wubu file providing the new texture/material.")
    p_graft.add_argument('--rect', type=str, required=True, help="Rectangle to replace, format: 'x,y,width,height'.")
    p_graft.add_argument('--alpha', type=float, default=0.75, help="How strongly the graft adopts the source lighting vs the base lighting (0.0=base, 1.0=source).")
    p_graft.add_argument('--output-path', type=str, required=True, help="Path to save the resulting PNG image.")
    
    args = parser.parse_args()
    
    config_source_path = ""
    if args.command == 'interpolate': config_source_path = args.start_wubu
    elif args.command == 'mood-transfer': config_source_path = args.content_wubu
    elif args.command == 'texture-graft': config_source_path = args.base_wubu

    print("--- Reading model metadata from compressed file... ---")
    with open(config_source_path, 'rb') as f: model_config = pickle.load(f)['model_config']
    print(f"--- File metadata found: {model_config} ---")
    
    args.d_model, args.latent_grid_size, args.image_size = model_config['d_model'], model_config['latent_grid_size'], model_config['image_size']
        
    ema_params, dtype = load_model_weights(args)
    model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=dtype)
    manipulator = LatentManipulator(model, ema_params, args)

    if args.command == 'interpolate': manipulator.perform_interpolation()
    elif args.command == 'mood-transfer': manipulator.perform_mood_transfer()
    elif args.command == 'texture-graft': manipulator.perform_texture_graft()

if __name__ == "__main__":
    main()