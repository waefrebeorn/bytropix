#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Audio Compressor for Topological Coordinate-Based Spectrogram Autoencoder (V5 "Image-First")

V5 UPDATE: This script is fully compatible with the single-headed, "image-first"
model. It encodes audio into latents and decodes by first reconstructing the
colormapped spectrogram image, then inverting the colormap back to a magnitude
spectrogram for audio synthesis via Griffin-Lim.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse, math, pickle, gzip, sys
from functools import partial
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
import soundfile as sf, torchaudio
from flax import linen as nn
from tqdm import tqdm
from typing import Any

# JAX configuration
try:
    if jax.lib.xla_bridge.get_backend().platform == 'gpu':
        jax.config.update("jax_platform_name", "cuda")
    else:
        jax.config.update("jax_platform_name", "cpu")
except Exception:
     jax.config.update("jax_platform_name", "cpu")

try:
    import matplotlib.cm
except ImportError:
    print("[FATAL] Matplotlib is required for the colormap. Please install it: pip install matplotlib")
    sys.exit(1)

# =================================================================================================
# 1. MODEL DEFINITIONS (COPIED VERBATIM FROM V5 TRAINER FOR COMPATIBILITY)
# =================================================================================================

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        num_downsamples = int(math.log2(self.input_image_size / self.latent_grid_size))
        x, context_vectors, features = images, [], 32
        for i in range(num_downsamples):
            x = nn.Conv(features, (4, 4), (2, 2), name=f"down_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x)
            context_vectors.append(jnp.mean(x, axis=(1, 2)))
            features *= 2
        context_vector = jnp.concatenate(context_vectors, axis=-1)
        x = nn.Conv(256, (3, 3), padding='SAME', name="feat_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        path_params_conv = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)
        delta_c, chi_c, radius = nn.tanh(path_params_conv[..., 0]) * jnp.pi, nn.tanh(path_params_conv[..., 1]) * (jnp.pi / 4.0), nn.sigmoid(path_params_conv[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1), context_vector

class TopologicalObserver(nn.Module):
    d_model: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = path_params_grid.shape
        features = nn.Dense(self.d_model, name="proj", dtype=self.dtype)(path_params_grid.reshape(B, H * W, C))
        return nn.gelu(features).reshape(B, H, W, self.d_model)

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
    d_model: int; mlp_width: int = 256; mlp_depth: int = 4; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, context_vector: jnp.ndarray, coords: jnp.ndarray):
        B = feature_grid.shape[0]
        pos_encoder = PositionalEncoding(num_freqs=10, dtype=self.dtype)
        encoded_coords = pos_encoder(coords)
        coords_rescaled = (coords + 1) / 2 * (jnp.array(feature_grid.shape[1:3], dtype=self.dtype) - 1)
        def sample_one_image(single_grid):
            return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(single_grid.transpose(2, 0, 1)).T
        sampled_features = jax.vmap(sample_one_image)(feature_grid)
        h = jnp.concatenate([jnp.repeat(encoded_coords[None, :, :], B, axis=0), sampled_features], axis=-1)
        film_layer = FiLMLayer(dtype=self.dtype)
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h)
            h = film_layer(h, context_vector)
            h = nn.gelu(h)
        return nn.tanh(nn.Dense(3, name="mlp_img_out", dtype=self.dtype)(h))

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(d_model=self.d_model, dtype=self.dtype)
    def __call__(self, images, coords_img):
        path_params, context = self.encode(images)
        return self.decode(path_params, context, coords_img)
    def encode(self, images): return self.modulator(images)
    def decode(self, path_params, context, coords):
        return self.coord_decoder(self.observer(path_params), context, coords)

# =================================================================================================
# 2. AUDIO & SPECTROGRAM UTILITIES
# =================================================================================================

COLORMAP_LUT = jnp.array(matplotlib.cm.magma(np.linspace(0, 1, 256))[:, :3])

def get_a_weighting(frequencies):
    f_sq = frequencies**2
    const = 12194**2
    num = const * (f_sq**2)
    den = (f_sq + 20.6**2) * jnp.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + const)
    weights = 20 * jnp.log10(jnp.where(den > 0, num / den, 0)) + 2.00
    return weights

@partial(jax.jit, static_argnames=('n_fft', 'hop_length', 'output_size', 'sample_rate'))
def audio_to_spec_image(audio_batch, n_fft: int, hop_length: int, output_size: tuple[int, int], sample_rate: int):
    specs_complex = jax.vmap(lambda w: jax.scipy.signal.stft(w, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft, boundary='constant', padded=True)[2])(audio_batch.astype(jnp.float32))
    spec_mag = jnp.abs(specs_complex)
    spec_db = 10.0 * jnp.log10(jnp.maximum(1e-12, spec_mag**2))
    fft_freqs = jnp.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    a_weights_db = get_a_weighting(fft_freqs)
    spec_db_a = spec_db + a_weights_db[None, :, None]
    DYNAMIC_RANGE = 120.0
    spec_norm = (jnp.clip(spec_db_a, -DYNAMIC_RANGE, 0.0) + DYNAMIC_RANGE) / DYNAMIC_RANGE
    indices = (spec_norm * 255).astype(jnp.int32)
    spec_color = COLORMAP_LUT[indices]
    spec_resized = jax.image.resize(spec_color, (spec_color.shape[0], *output_size, 3), 'bilinear')
    return spec_resized * 2.0 - 1.0

@partial(jax.jit, static_argnames=('output_shape',))
def invert_colormap_to_magnitude(pixel_batch, output_shape):
    pixels_reshaped = pixel_batch.reshape(pixel_batch.shape[0], -1, 1, 3)
    colormap_lut_reshaped = COLORMAP_LUT[None, None, :, :]
    indices = jnp.argmin(jnp.sum((pixels_reshaped - colormap_lut_reshaped)**2, axis=-1), axis=-1)
    spec_norm_db = indices / 255.0
    DYNAMIC_RANGE = 120.0
    spec_db = spec_norm_db * DYNAMIC_RANGE - DYNAMIC_RANGE
    magnitude = jnp.sqrt(10**(spec_db / 10.0))
    magnitude_image = magnitude.reshape(pixel_batch.shape[0], *pixel_batch.shape[1:3])
    return jax.image.resize(magnitude_image, (magnitude_image.shape[0], *output_shape), 'bilinear')

@partial(jax.jit, static_argnames=('n_fft', 'hop_length', 'n_iter'))
def griffin_lim(mag_spec, n_fft, hop_length, n_iter):
    angle = jnp.zeros_like(mag_spec)
    for _ in range(n_iter):
        spec_complex = mag_spec * jnp.exp(1j * angle)
        _, audio = jax.scipy.signal.istft(spec_complex, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)
        _, _, spec_new = jax.scipy.signal.stft(audio, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)
        angle = jnp.angle(spec_new)
    _, audio_final = jax.scipy.signal.istft(mag_spec * jnp.exp(1j * angle), nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)
    return audio_final

# =================================================================================================
# 3. COMPRESSOR CLASS
# =================================================================================================

class AudioCompressor:
    def __init__(self, args):
        self.args = args
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        self.segment_len_samples = args.sample_rate * args.segment_len_secs

        # Calculate STFT parameters to match the training setup
        self.N_FFT = (args.image_size - 1) * 2
        self.HOP_LENGTH = self.segment_len_samples // args.image_size
        print(f"--- STFT Config: N_FFT={self.N_FFT}, HOP_LENGTH={self.HOP_LENGTH} ---")

        print(f"--- Initializing Model for Inference (dtype: {self.dtype}) ---")
        self.model = TopologicalCoordinateGenerator(
            d_model=args.d_model, latent_grid_size=args.latent_grid_size,
            input_image_size=args.image_size, dtype=self.dtype
        )
        self.params = self._load_model_params()

        # Define the shape of the original (pre-resize) magnitude spectrogram
        self.original_spec_shape = (self.N_FFT // 2 + 1, self.args.image_size)

        # Coordinate grid for decoding the full image
        self.coords_img = jnp.stack(jnp.meshgrid(
            jnp.linspace(-1, 1, self.args.image_size),
            jnp.linspace(-1, 1, self.args.image_size),
            indexing='ij'
        ), axis=-1).reshape(-1, 2)
        
        self._jit_functions()

    def _load_model_params(self):
        ckpt_path = Path(self.args.model_path)
        if not ckpt_path.exists(): raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}.")
        print(f"--- Loading model checkpoint from {ckpt_path} ---")
        with open(ckpt_path, 'rb') as f: data = pickle.load(f)
        if 'ema_params' not in data: raise KeyError("Checkpoint must contain 'ema_params'.")
        return data['ema_params']

    def _jit_functions(self):
        print("--- Compiling JAX functions (one-time cost)... ---")
        @jax.jit
        def _encode_jit(spec_image_batch):
            return self.model.apply({'params': self.params}, spec_image_batch, method=self.model.encode)
        self.encode_jit = _encode_jit
        
        @partial(jax.jit, static_argnames=('img_size', 'spec_shape'))
        def _decode_jit(path_params_batch, context_batch, coords, img_size, spec_shape):
            # Decode to get the flat image pixels
            recon_pixels_flat = self.model.apply(
                {'params': self.params}, path_params_batch, context_batch, coords, method=self.model.decode
            )
            # Reshape to image and convert range from [-1, 1] to [0, 1]
            recon_image = recon_pixels_flat.reshape(path_params_batch.shape[0], img_size, img_size, 3)
            recon_image_0_1 = (recon_image + 1.0) / 2.0
            
            # Invert the colormap to get the magnitude spectrogram
            mag_spec_recon = invert_colormap_to_magnitude(recon_image_0_1, spec_shape)
            return mag_spec_recon
        self.decode_jit = _decode_jit

        # Trigger JIT compilation
        dummy_image = jnp.zeros((1, self.args.image_size, self.args.image_size, 3), dtype=self.dtype)
        dummy_params, dummy_context = self.encode_jit(dummy_image)
        self.decode_jit(dummy_params, dummy_context, self.coords_img, self.args.image_size, self.original_spec_shape)
        print("--- JAX functions compiled. ---")

    def encode_file(self):
        print(f"--- Loading audio from {self.args.input} ---")
        waveform, sr = torchaudio.load(self.args.input)
        if sr != self.args.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.args.sample_rate)(waveform)
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        original_len = waveform.shape[0]
        
        num_segments = -(-original_len // self.segment_len_samples)
        padded_len = num_segments * self.segment_len_samples
        segments = np.pad(waveform.numpy(), (0, padded_len - original_len)).reshape(num_segments, self.segment_len_samples)
        
        all_path_params, all_contexts, all_peaks = [], [], []
        batch_size = 16
        print("--- Encoding audio segments... ---")
        for i in tqdm(range(0, num_segments, batch_size), desc="Encoding"):
            segment_batch = segments[i:i+batch_size]
            spec_images = audio_to_spec_image(segment_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size), self.args.sample_rate)
            
            path_params, context = self.encode_jit(spec_images.astype(self.dtype))
            peaks = np.max(np.abs(segment_batch), axis=-1)
            
            all_path_params.append(np.array(path_params))
            all_contexts.append(np.array(context))
            all_peaks.append(peaks)
        
        with gzip.open(self.args.output, 'wb') as f:
            pickle.dump({
                'path_params': np.concatenate(all_path_params), 
                'contexts': np.concatenate(all_contexts),
                'peaks': np.concatenate(all_peaks), 
                'metadata': {'original_len': original_len}
            }, f)
        print(f"--- ✅ Encoding complete. Saved to {self.args.output} ---")

    def decode_file(self):
        print(f"--- Loading compressed data from {self.args.input} ---")
        with gzip.open(self.args.input, 'rb') as f: data = pickle.load(f)
        
        path_params = jnp.array(data['path_params'], dtype=self.dtype)
        contexts = jnp.array(data['contexts'], dtype=self.dtype)
        original_peaks, metadata = jnp.array(data['peaks']), data['metadata']
        reconstructed_segments = []

        vmapped_gl = jax.vmap(griffin_lim, in_axes=(0, None, None, None))
        batch_size = 8
        print("--- Decoding latent vectors and reconstructing audio... ---")
        for i in tqdm(range(0, path_params.shape[0], batch_size), desc="Decoding"):
            params_chunk = path_params[i:i+batch_size]
            context_chunk = contexts[i:i+batch_size]
            peak_chunk = original_peaks[i:i+batch_size]
            
            # Decode latents -> image -> magnitude spectrogram
            mag_spec_recon = self.decode_jit(params_chunk, context_chunk, self.coords_img, self.args.image_size, self.original_spec_shape)
            
            # Reconstruct audio from magnitude
            audio_recon_chunk = vmapped_gl(mag_spec_recon, self.N_FFT, self.HOP_LENGTH, 32)
            
            # Restore original peak amplitude
            current_peaks = jnp.maximum(1e-7, jnp.max(jnp.abs(audio_recon_chunk), axis=-1))
            audio_renorm = audio_recon_chunk * (peak_chunk / current_peaks)[:, None]
            reconstructed_segments.extend(np.array(a) for a in audio_renorm)

        final_audio = np.concatenate(reconstructed_segments)[:metadata['original_len']]
        sf.write(self.args.output, final_audio, self.args.sample_rate)
        print(f"--- ✅ Decoding complete. Saved to {self.args.output} ---")

def main():
    parser = argparse.ArgumentParser(description="Compressor for Topological Audio AE (V5 Image-First)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model checkpoint (.pkl).")
    parent_parser.add_argument('--d-model', type=int, default=128, help="Model dimension (d_model) from training.")
    parent_parser.add_argument('--latent-grid-size', type=int, default=16, help="Latent grid size from training.")
    parent_parser.add_argument('--image-size', type=int, default=256, help="Spectrogram image resolution from training.")
    parent_parser.add_argument('--sample-rate', type=int, default=44100, help="Sample rate from training.")
    parent_parser.add_argument('--segment-len-secs', type=int, default=4, help="Segment length from training.")
    parent_parser.add_argument('--use-bfloat16', action='store_true', help="MUST be used if the model was trained with this flag.")

    p_encode = subparsers.add_parser("encode", help="Encode an audio file.", parents=[parent_parser])
    p_encode.add_argument('--input', type=str, required=True, help="Input audio file (.wav, .flac, etc.).")
    p_encode.add_argument('--output', type=str, required=True, help="Output compressed file (e.g., audio.taac5).")

    p_decode = subparsers.add_parser("decode", help="Decode a compressed file to audio.", parents=[parent_parser])
    p_decode.add_argument('--input', type=str, required=True, help="Input compressed file (e.g., audio.taac5).")
    p_decode.add_argument('--output', type=str, required=True, help="Output reconstructed audio file (.wav).")
    
    args = parser.parse_args()

    try:
        compressor = AudioCompressor(args)
        if args.command == "encode": compressor.encode_file()
        elif args.command == "decode": compressor.decode_file()
    except (FileNotFoundError, KeyError) as e: print(f"\n[FATAL ERROR] {e}", file=sys.stderr); sys.exit(1)
    except Exception: import traceback; print("\n[FATAL ERROR] An unexpected error occurred:", file=sys.stderr); traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main()