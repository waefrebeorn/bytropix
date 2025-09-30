#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio Compressor for Topological Coordinate Generator Model

This script provides command-line tools to encode and decode audio files using
a pre-trained model from PHASE1_AUDIO.PY.

It works by:
1.  (Encode): Loading an audio file, converting it segment-by-segment into the
    same kind of colormapped spectrogram image the model was trained on, and
    running it through the model's encoder to get a compressed latent vector.
2.  (Decode): Loading the compressed latent vectors, using the model's decoder
    to reconstruct the spectrogram image, and then converting that image back
    into an audio waveform using the Griffin-Lim algorithm for phase reconstruction.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import gzip
import math
import pickle
from functools import partial
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import jax.scipy.signal
import jax.image
import jax.scipy.ndimage
import numpy as np
import soundfile as sf
import torchaudio
from flax import linen as nn
from tqdm import tqdm

# JAX configuration
jax.config.update("jax_platform_name", "cuda" if jax.devices("gpu") else "cpu")

# =================================================================================================
# 1. MODEL DEFINITIONS (COPIED VERBATIM FROM PHASE1_AUDIO.PY FOR COMPATIBILITY)
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
        num_downsamples = int(math.log2(self.input_image_size / self.latent_grid_size))
        x, features = images, 32
        for i in range(num_downsamples):
            x = nn.Conv(features, (4, 4), (2, 2), name=f"downsample_conv_{i}", dtype=self.dtype)(x)
            x = nn.gelu(x); features *= 2
        x = nn.Conv(256, (3, 3), padding='SAME', name="final_feature_conv", dtype=self.dtype)(x)
        x = nn.gelu(x)
        path_params = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1)

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = path_params_grid.shape; L = H * W
        path_params = path_params_grid.reshape(B, L, C)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
        chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
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
        B = feature_grid.shape[0]; pos_encoder = PositionalEncoding(self.num_freqs, dtype=self.dtype)
        encoded_coords = pos_encoder(coords)
        coords_rescaled = (coords + 1) / 2 * (jnp.array(feature_grid.shape[1:3], dtype=self.dtype) - 1)
        def sample_one_image(single_feature_grid):
            grid_chw = single_feature_grid.transpose(2, 0, 1)
            return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw).T
        sampled_features = jax.vmap(sample_one_image)(feature_grid)
        encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)
        mlp_input = jnp.concatenate([encoded_coords_tiled, sampled_features], axis=-1)
        h = mlp_input
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = nn.gelu(h)
        output_pixels = nn.Dense(3, name="mlp_out", dtype=self.dtype)(h)
        return nn.tanh(output_pixels)

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(d_model=self.d_model, name="coord_decoder", dtype=self.dtype)

    def encode(self, images):
        path_params = self.modulator(images)
        feature_grid = self.observer(path_params)
        return feature_grid

    def decode(self, feature_grid, coords):
        return self.coord_decoder(feature_grid, coords)

# =================================================================================================
# 2. AUDIO & SPECTROGRAM UTILITIES
# =================================================================================================
# We need to import matplotlib here just for the colormap LUT
try:
    import matplotlib.cm
except ImportError:
    print("Matplotlib is required for the colormap. Please install it: pip install matplotlib")
    exit(1)

COLORMAP_LUT = jnp.array(matplotlib.cm.magma(np.linspace(0, 1, 256))[:, :3])

def audio_to_spec_image_and_params(audio_batch: jnp.ndarray, n_fft: int, hop_length: int, output_size: tuple[int, int]):
    """Converts audio to a spec image and returns normalization params needed for reconstruction."""
    specs_complex = jax.vmap(lambda w: jax.scipy.signal.stft(w, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)[2])(audio_batch.astype(jnp.float32))
    spec_mag_sq = jnp.abs(specs_complex)**2
    spec_db = 10.0 * jnp.log10(jnp.maximum(1e-6, spec_mag_sq))

    # Get p5 and p95 for each item in the batch for later reconstruction
    p5 = jnp.percentile(spec_db, 5, axis=(-1, -2))
    p95 = jnp.percentile(spec_db, 95, axis=(-1, -2))

    # Normalize each spec independently, the same way as in training
    def normalize_db(db_spec_single, p5_single, p95_single):
        db_spec_clipped = jnp.clip(db_spec_single, p5_single, p95_single)
        return (db_spec_clipped - db_spec_clipped.min()) / (db_spec_clipped.max() - db_spec_clipped.min() + 1e-6)

    spec_norm = jax.vmap(normalize_db)(spec_db, p5, p95)
    indices = (spec_norm * 255).astype(jnp.int32)
    spec_color = COLORMAP_LUT[indices]
    spec_resized = jax.image.resize(spec_color, (spec_color.shape[0], *output_size, 3), 'bilinear')
    return spec_resized * 2.0 - 1.0, p5, p95, jnp.max(jnp.abs(audio_batch), axis=-1)

@partial(jax.jit, static_argnames=('n_fft', 'hop_length', 'n_iter'))
def griffin_lim(mag_spec, n_fft, hop_length, n_iter):
    """Iteratively reconstruct phase from magnitude."""
    angle = jnp.zeros_like(mag_spec)
    for _ in range(n_iter):
        spec_complex = mag_spec * jnp.exp(1j * angle)
        _, audio = jax.scipy.signal.istft(spec_complex, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)
        _, _, spec_new = jax.scipy.signal.stft(audio, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)
        angle = jnp.angle(spec_new)
    spec_final = mag_spec * jnp.exp(1j * angle)
    _, audio_final = jax.scipy.signal.istft(spec_final, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)
    return audio_final

def spec_image_to_audio(spec_image, p5, p95, peak_val, n_fft, hop_length, original_spec_shape):
    """Converts a reconstructed spectrogram image back to audio."""
    # 1. Undo the tanh and resize
    spec_image_norm = (spec_image + 1.0) / 2.0
    spec_image_resized = jax.image.resize(spec_image_norm, (spec_image.shape[0], *original_spec_shape, 3), 'bilinear')

    # 2. Convert from color to grayscale to approximate normalized magnitude
    mag_norm = jnp.mean(spec_image_resized, axis=-1)

    # 3. Denormalize using the saved p5 and p95 values
    mag_db_clipped = mag_norm * (p95[:, None, None] - p5[:, None, None]) + p5[:, None, None]

    # 4. Convert from dB back to magnitude
    magnitude_spec = jnp.sqrt(10**(mag_db_clipped / 10.0))

    # 5. Reconstruct audio using Griffin-Lim
    audio_recon = jax.vmap(griffin_lim, in_axes=(0, None, None, None))(magnitude_spec, n_fft, hop_length, 32)
    
    # 6. Normalize to the original peak value
    current_peak = jnp.maximum(1e-6, jnp.max(jnp.abs(audio_recon), axis=-1))
    audio_renorm = audio_recon * (peak_val / current_peak)[:, None]
    
    return audio_renorm

# =================================================================================================
# 3. COMPRESSOR CLASS
# =================================================================================================

class AudioCompressor:
    def __init__(self, args):
        self.args = args
        self.dtype = jnp.float32 # Stick to float32 for inference stability
        self.N_FFT = 1024
        self.HOP_LENGTH = 256
        self.segment_len_samples = args.sample_rate * args.segment_len_secs

        print("--- Initializing Model for Inference ---")
        self.model = TopologicalCoordinateGenerator(
            d_model=args.d_model,
            latent_grid_size=args.latent_grid_size,
            input_image_size=args.image_size,
            dtype=self.dtype
        )
        self.params = self._load_model_params()
        
        # Determine original spectrogram shape before resizing
        dummy_audio = jnp.zeros((1, self.segment_len_samples))
        _, _, spec = jax.scipy.signal.stft(dummy_audio, nperseg=self.N_FFT, noverlap=self.N_FFT-self.HOP_LENGTH, nfft=self.N_FFT)
        self.original_spec_shape = spec.shape[1:3]

        self._jit_functions()

    def _load_model_params(self):
        ckpt_path = Path(f"{self.args.basename}.pkl")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}. Please train the model first.")
        print(f"--- Loading model checkpoint from {ckpt_path} ---")
        with open(ckpt_path, 'rb') as f:
            data = pickle.load(f)
        return data['params'] # Assuming params are stored under this key

    def _jit_functions(self):
        print("--- Compiling JAX functions (one-time cost)... ---")

        @jax.jit
        def _encode_jit(params, spec_image_batch):
            return self.model.apply({'params': params}, spec_image_batch, method=self.model.encode)
        self.encode_jit = _encode_jit

        @partial(jax.jit, static_argnames=('patch_size',))
        def _decode_patch_jit(params, feature_grid, coords_patch, patch_size):
            return self.model.apply({'params': params}, feature_grid, coords_patch, method=self.model.decode)
        self.decode_patch_jit = _decode_patch_jit

        # Dry run to trigger compilation
        dummy_image = jnp.zeros((1, self.args.image_size, self.args.image_size, 3), dtype=self.dtype)
        dummy_latents = self.encode_jit(self.params, dummy_image)
        dummy_coords = jnp.zeros((1024, 2), dtype=jnp.float32)
        self.decode_patch_jit(self.params, dummy_latents, dummy_coords, 1024)
        print("--- JAX functions compiled. ---")

    def encode_file(self):
        print(f"--- Loading audio from {self.args.input} ---")
        waveform, sr = torchaudio.load(self.args.input)
        if sr != self.args.sample_rate:
            print(f"Resampling from {sr} Hz to {self.args.sample_rate} Hz...")
            waveform = torchaudio.transforms.Resample(sr, self.args.sample_rate)(waveform)
        
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)

        num_segments = -(-waveform.shape[0] // self.segment_len_samples)
        padded_len = num_segments * self.segment_len_samples
        padding_needed = padded_len - waveform.shape[0]
        waveform_padded = jnp.pad(waveform.numpy(), (0, padding_needed))
        segments = waveform_padded.reshape(num_segments, self.segment_len_samples)
        
        all_latents, all_p5, all_p95, all_peaks = [], [], [], []
        
        print("--- Encoding audio segments... ---")
        batch_size = 8 # Smaller batch for potentially large spec images
        for i in tqdm(range(0, num_segments, batch_size), desc="Encoding"):
            segment_batch = segments[i:i+batch_size]
            spec_images, p5, p95, peaks = audio_to_spec_image_and_params(
                segment_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size)
            )
            latents = self.encode_jit(self.params, spec_images)
            all_latents.append(np.array(latents))
            all_p5.append(np.array(p5))
            all_p95.append(np.array(p95))
            all_peaks.append(np.array(peaks))

        final_latents = np.concatenate(all_latents, axis=0)
        final_p5 = np.concatenate(all_p5, axis=0)
        final_p95 = np.concatenate(all_p95, axis=0)
        final_peaks = np.concatenate(all_peaks, axis=0)

        metadata = {'original_len': waveform.shape[0]}
        
        print(f"--- Saving {final_latents.shape[0]} latent vectors to {self.args.output} ---")
        with gzip.open(self.args.output, 'wb') as f:
            pickle.dump({
                'latents': final_latents, 
                'p5': final_p5,
                'p95': final_p95,
                'peaks': final_peaks,
                'metadata': metadata
            }, f)
        print("--- ✅ Encoding complete. ---")

    def decode_file(self):
        print(f"--- Loading compressed data from {self.args.input} ---")
        with gzip.open(self.args.input, 'rb') as f:
            data = pickle.load(f)
        
        latents = jnp.array(data['latents'], dtype=self.dtype)
        p5_vals = jnp.array(data['p5'])
        p95_vals = jnp.array(data['p95'])
        peaks = jnp.array(data['peaks'])
        metadata = data['metadata']
        
        num_segments = latents.shape[0]
        reconstructed_segments = []

        print("--- Decoding latent vectors and reconstructing audio... ---")
        batch_size = 4 # Decoding can be memory intensive
        for i in tqdm(range(0, num_segments, batch_size), desc="Decoding"):
            latent_chunk = latents[i:i+batch_size]
            p5_chunk = p5_vals[i:i+batch_size]
            p95_chunk = p95_vals[i:i+batch_size]
            peak_chunk = peaks[i:i+batch_size]
            
            # Create coordinate grid for the full image
            coords_y = jnp.linspace(-1, 1, self.args.image_size)
            coords_x = jnp.linspace(-1, 1, self.args.image_size)
            full_coords = jnp.stack(jnp.meshgrid(coords_y, coords_x, indexing='ij'), axis=-1).reshape(-1, 2)
            
            # Decode in patches to save memory
            patch_size = 16384 
            num_patches = (full_coords.shape[0] + patch_size - 1) // patch_size
            
            pixel_patches = []
            for j in range(num_patches):
                start_idx = j * patch_size
                coord_patch = jax.lax.dynamic_slice_in_dim(full_coords, start_idx, patch_size, axis=0)
                pixel_patch = self.decode_patch_jit(self.params, latent_chunk, coord_patch, patch_size)
                pixel_patches.append(pixel_patch)

            spec_image_flat = jnp.concatenate(pixel_patches, axis=1)
            spec_image_recon = spec_image_flat.reshape(
                latent_chunk.shape[0], self.args.image_size, self.args.image_size, 3
            )
            
            audio_recon = spec_image_to_audio(
                spec_image_recon, p5_chunk, p95_chunk, peak_chunk, self.N_FFT, self.HOP_LENGTH, self.original_spec_shape
            )
            reconstructed_segments.extend([np.array(a) for a in audio_recon])

        print("--- Stitching audio segments... ---")
        full_audio = np.concatenate(reconstructed_segments)
        final_audio = full_audio[:metadata['original_len']]
        
        print(f"--- Saving reconstructed audio to {self.args.output} ---")
        sf.write(self.args.output, final_audio, self.args.sample_rate)
        print("--- ✅ Decoding complete. ---")

def main():
    parser = argparse.ArgumentParser(description="Compressor for Topological Coordinate Generator Audio Model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments for both encode and decode
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--basename', type=str, required=True, help="Basename of the trained model checkpoint (e.g., 'TopologicalAE_FMA_44k_4s_256px_64d_spec512').")
    parent_parser.add_argument('--d-model', type=int, default=64, help="Model dimension used during training.")
    parent_parser.add_argument('--latent-grid-size', type=int, default=64, help="Latent grid size used during training.")
    parent_parser.add_argument('--image-size', type=int, default=512, help="Spectrogram image resolution used during training.")
    parent_parser.add_argument('--sample-rate', type=int, default=44100, help="Sample rate of the audio.")
    parent_parser.add_argument('--segment-len-secs', type=int, default=4, help="Segment length in seconds used during training.")

    # Encode command
    p_encode = subparsers.add_parser("encode", help="Encode an audio file to a compressed format.", parents=[parent_parser])
    p_encode.add_argument('--input', type=str, required=True, help="Path to the input audio file (.wav, .flac, etc.).")
    p_encode.add_argument('--output', type=str, required=True, help="Path to save the compressed output file (.taac)") # Topological Audio Auto-encoder Codec :)

    # Decode command
    p_decode = subparsers.add_parser("decode", help="Decode a compressed file back to an audio file.", parents=[parent_parser])
    p_decode.add_argument('--input', type=str, required=True, help="Path to the compressed input file (.taac).")
    p_decode.add_argument('--output', type=str, required=True, help="Path to save the reconstructed audio file (.wav).")
    
    args = parser.parse_args()

    # Match the checkpoint name from your training logs
    # e.g., "TopologicalAE_FMA_44k_4s_256px_64d_spec512.pkl"
    args.basename = f"{args.basename}_{args.d_model}d_spec{args.image_size}"

    compressor = AudioCompressor(args)
    if args.command == "encode":
        compressor.encode_file()
    elif args.command == "decode":
        compressor.decode_file()

if __name__ == "__main__":
    main()