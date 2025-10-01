#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Topological Coordinate-Based Spectrogram Autoencoder (High-Performance Version)

V5 "IMAGE-FIRST" UPDATE: This version embraces a single-headed architecture. The model is trained
exclusively to reconstruct the colormapped spectrogram image. All audio-centric losses (Mel, Linear)
are calculated by first inverting the reconstructed image's colormap back into a spectrogram,
forcing the model to learn audio-correctness through the visual domain. This simplifies the model
and increases stability.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math, jax, jax.numpy as jnp, numpy as np, pickle, time, sys, argparse, signal
from flax import linen as nn
from flax.training import train_state, common_utils
from flax.jax_utils import replicate, unreplicate
import optax
from functools import partial
from pathlib import Path
from collections import deque
from PIL import Image
import jax.scipy.ndimage, jax.scipy.signal, jax.image
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False)

try:
    import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
    import torchaudio, torch, tensorflow_datasets as tfds, matplotlib.cm, chex
    from rich.live import Live; from rich.table import Table; from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.layout import Layout; from rich.console import Console; from rich.align import Align; from rich.text import Text
    import pynvml; pynvml.nvmlInit()
    from tqdm import tqdm
except ImportError as e:
    print(f"[FATAL] A required dependency is missing: {e.name}. Please install it."); sys.exit(1)

try:
    from rich_pixels import Pixels
except ImportError:
    Pixels = None

# =================================================================================================
# 1. MODEL ARCHITECTURE (Single-Headed "Image-First" Version)
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

class TrainStateWithEMA(train_state.TrainState): ema_params: Any

# =================================================================================================
# 2. AUDIO & DATA HANDLING / LOSS FUNCTIONS
# =================================================================================================

COLORMAP_LUT = jnp.array(matplotlib.cm.magma(np.linspace(0, 1, 256))[:, :3])

def get_a_weighting(frequencies):
    """Calculate A-weighting in dB for a given set of frequencies."""
    f_sq = frequencies**2
    const = 12194**2
    num = const * (f_sq**2)
    den = (f_sq + 20.6**2) * jnp.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + const)
    # --- START FIX: Removed the non-existent `jnp.errstate` context manager. ---
    # The `jnp.where` call below is already sufficient to prevent division by zero.
    weights = 20 * jnp.log10(jnp.where(den > 0, num / den, 0)) + 2.00
    # --- END FIX ---
    return weights

@partial(jax.jit, static_argnames=('n_fft', 'hop_length', 'output_size', 'sample_rate'))
def audio_to_spec_image(audio_batch: chex.Array, n_fft: int, hop_length: int, output_size: tuple[int, int], sample_rate: int):
    specs_complex = jax.vmap(lambda w: jax.scipy.signal.stft(w, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft, boundary='constant', padded=True)[2])(audio_batch.astype(jnp.float32))
    spec_mag = jnp.abs(specs_complex)
    spec_db = 10.0 * jnp.log10(jnp.maximum(1e-12, spec_mag**2)) # Use a smaller epsilon

    # Apply A-weighting to make the spectrogram perceptually relevant
    fft_freqs = jnp.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    a_weights_db = get_a_weighting(fft_freqs)
    spec_db_a = spec_db + a_weights_db[None, :, None]
    
    # Normalize using a wider, 120dB dynamic range for better detail
    DYNAMIC_RANGE = 120.0
    spec_norm = (jnp.clip(spec_db_a, -DYNAMIC_RANGE, 0.0) + DYNAMIC_RANGE) / DYNAMIC_RANGE
    
    indices = (spec_norm * 255).astype(jnp.int32)
    spec_color = COLORMAP_LUT[indices]
    spec_resized = jax.image.resize(spec_color, (spec_color.shape[0], *output_size, 3), 'bilinear')
    return spec_resized * 2.0 - 1.0, spec_mag

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

def create_mel_filterbank(sr=44100, n_fft=1024, n_mels=128):
    f_max, num_bins = sr / 2.0, n_fft // 2 + 1
    weights, fft_freqs = np.zeros((n_mels, num_bins)), np.linspace(0, sr / 2.0, num_bins)
    mel_min, mel_max = 1127.0 * np.log(1.0 + 0.0 / 700.0), 1127.0 * np.log(1.0 + f_max / 700.0)
    mel_pts, hz_pts = np.linspace(mel_min, mel_max, n_mels + 2), 700.0 * (np.exp(np.linspace(mel_min, mel_max, n_mels + 2) / 1127.0) - 1.0)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i+1], hz_pts[i+2]
        weights[i, :] = np.maximum(0, np.minimum((fft_freqs - left) / (center - left), (right - fft_freqs) / (right - center)))
    return np.array(weights)

def magnitude_to_mel_db(mag_spec, mel_filterbank):
    return 10.0 * jnp.log10(jnp.maximum(1e-6, jnp.einsum('...ft,mf->...mt', mag_spec**2, mel_filterbank)))

@jax.jit
def contrastive_loss(img1, img2):
    def get_grads(img):
        img_lum = jnp.dot(img, jnp.array([0.299, 0.587, 0.114]))
        grads_x = jax.scipy.signal.convolve(img_lum, jnp.array([[1, 0, -1]]), mode='same')
        grads_y = jax.scipy.signal.convolve(img_lum, jnp.array([[1], [0], [-1]]), mode='same')
        return jnp.sqrt(grads_x**2 + grads_y**2)
    return jnp.mean(jnp.abs(jax.vmap(get_grads)(img1) - jax.vmap(get_grads)(img2)))

def prepare_audio_data(data_dir: str, sample_rate: int, segment_len_secs: int):
    base_path = Path(data_dir); record_file = base_path / f"audio_{sample_rate}hz_{segment_len_secs}s.tfrecord"
    if record_file.exists():
        print(f"✅ Audio data found. Skipping preparation."); return
    print(f"--- Preparing audio data from {data_dir} ---")
    file_list = [p for p in base_path.rglob('*') if p.suffix.lower() in ('.wav', '.flac', '.mp3')]
    if not file_list: print(f"[FATAL] No audio files found in {data_dir}."), sys.exit(1)
    num_samples, segment_len_samples = 0, sample_rate * segment_len_secs
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for file_path in tqdm(file_list, "Processing Audio", unit="file"):
            try:
                waveform, sr = torchaudio.load(file_path)
                if sr != sample_rate: waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
                if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
                for start in range(0, waveform.shape[1] - segment_len_samples + 1, segment_len_samples):
                    segment = waveform[:, start:start + segment_len_samples].numpy().squeeze()
                    example = tf.train.Example(features=tf.train.Features(feature={'audio': tf.train.Feature(float_list=tf.train.FloatList(value=segment))}))
                    writer.write(example.SerializeToString()); num_samples += 1
            except Exception as e: print(f"Skipping {file_path}: {e}")
    with open(base_path / "audio_dataset_info.pkl", 'wb') as f: pickle.dump({'num_samples': num_samples}, f)
    print(f"✅ Audio data preparation complete. Found {num_samples} segments.")

def create_audio_dataset(data_dir: str, segment_len_samples: int, sample_rate: int, segment_len_secs: int, is_training: bool = True):
    base_path = Path(data_dir); record_file = base_path / f"audio_{sample_rate}hz_{segment_len_secs}s.tfrecord"
    info_file = base_path / "audio_dataset_info.pkl"
    if not record_file.exists(): raise FileNotFoundError(f"{record_file} not found. Run 'prepare-audio' first.")
    with open(info_file, 'rb') as f: num_samples = pickle.load(f)['num_samples']
    def _parse(proto):
        return tf.io.parse_single_example(proto, {'audio': tf.io.FixedLenFeature([segment_len_samples], tf.float32)})['audio']
    ds = tf.data.TFRecordDataset(str(record_file)).map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training: ds = ds.shuffle(1024).repeat()
    return ds, num_samples

# =================================================================================================
# 4. HIGH-PERFORMANCE TRAINER
# =================================================================================================

class Trainer:
    def __init__(self, args):
        self.args = args; self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'should_shutdown', True))
        self.console = Console()
        self.num_devices = jax.local_device_count()
        self.dtype = jnp.bfloat16 if args.use_bfloat16 else jnp.float32
        
        self.SEGMENT_LEN_SAMPLES = args.sample_rate * args.segment_len_secs
        self.N_FFT = (args.image_size - 1) * 2
        self.HOP_LENGTH = self.SEGMENT_LEN_SAMPLES // args.image_size
        self.console.print(f"--- [bold cyan]Audio Config[/]: Sample Rate={args.sample_rate}, Segment Len={args.segment_len_secs}s")
        self.console.print(f"--- [bold cyan]STFT Config[/]: N_FFT=[green]{self.N_FFT}[/], HOP_LENGTH=[green]{self.HOP_LENGTH}[/] for target image size {args.image_size}x{args.image_size}")
        
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, dtype=self.dtype)
        self.loss_history = deque(maxlen=200)
        self.mel_filterbank = create_mel_filterbank(sr=args.sample_rate, n_fft=self.N_FFT, n_mels=128)
        self.steps_per_sec = 0.0; self.last_losses = {}; self.best_val_loss = float('inf')
        self.preview_orig_np, self.preview_recon_np = None, None
        
    def _get_gpu_stats(self):
        try:
            h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except Exception: return "N/A", "N/A"

    def _get_sparkline(self, data, width=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data))
        if len(hist)<2: return " "*width
        hist=hist[-width:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " "*width
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

    def _generate_layout(self) -> Layout:
        layout = Layout(name="root"); layout.split(Layout(Panel(f"🎤 [bold]Spectrogram Autoencoder (Image-First)[/] | Model: [cyan]{self.args.basename}[/]", expand=False), size=3), Layout(ratio=1, name="main"), Layout(self.progress, size=3))
        layout["main"].split_row(Layout(name="left", minimum_size=40), Layout(name="right"))
        mem, util = self._get_gpu_stats()
        stats_tbl = Table.grid(expand=True, padding=(0,1)); stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right")
        stats_tbl.add_row("Total Loss", f"[bold green]{self.last_losses.get('total', 0):.4f}[/]")
        stats_tbl.add_row("L1 Image", f"{self.last_losses.get('l1_img', 0):.4f}")
        stats_tbl.add_row("L1 Spec (Inv)", f"[yellow]{self.last_losses.get('l1_spec_inv', 0):.4f}[/]")
        stats_tbl.add_row("Mel Loss (Inv)", f"[bold yellow]{self.last_losses.get('mel_inv', 0):.4f}[/]")
        stats_tbl.add_row("Contrast Loss", f"{self.last_losses.get('contrast', 0):.4f}")
        stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/]")
        stats_tbl.add_row("GPU Mem", f"{mem}"); stats_tbl.add_row("GPU Util", f"{util}")
        layout["left"].update(Panel(stats_tbl, title="[bold]📊 Stats[/]", border_style="blue"))
        preview_content = Text("Waiting for first preview...", justify="center")
        if self.preview_orig_np is not None and self.preview_recon_np is not None:
            if Pixels:
                term_w = (self.console.width - 50) // 2 if self.console.width > 50 else 20
                h, w, _ = self.preview_orig_np.shape; term_h = int(term_w * (h / w) * 0.5) if term_w > 0 else 10
                orig_img = Image.fromarray(self.preview_orig_np).resize((term_w, term_h), Image.Resampling.LANCZOS)
                recon_img = Image.fromarray(self.preview_recon_np).resize((term_w, term_h), Image.Resampling.LANCZOS)
                prev_tbl = Table.grid(expand=True); prev_tbl.add_row(Pixels.from_image(orig_img), Pixels.from_image(recon_img))
                preview_content = prev_tbl
            else:
                preview_content = Text("Install rich-pixels for image previews", justify="center")
        layout["right"].update(Panel(preview_content, title="[bold]🖼️ Live Preview[/] (Original vs. Recon)", border_style="magenta"))
        return layout

    def _save_checkpoint(self, state, epoch, step, val_loss, path):
        unrep_state = unreplicate(state)
        data = {'params': jax.device_get(unrep_state.params), 'ema_params': jax.device_get(unrep_state.ema_params), 'opt_state': jax.device_get(unrep_state.opt_state), 'epoch': epoch, 'step': step, 'best_val_loss': val_loss}
        with open(path, 'wb') as f: pickle.dump(data, f)
        self.console.print(f"--- 💾 Checkpoint saved to {path.name} (Val Loss: {val_loss:.4f}) ---")
    
    def _update_preview_task(self, ema_params, audio_batch):
        spec_image, _ = audio_to_spec_image(audio_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size), self.args.sample_rate)
        coords_img = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, self.args.image_size), jnp.linspace(-1, 1, self.args.image_size), indexing='ij'), axis=-1).reshape(-1, 2)
        @jax.jit
        def predict(params, img, c_img): 
            return self.model.apply({'params': params}, img, c_img)
        recon_flat = predict(ema_params, spec_image, coords_img)
        recon_image = recon_flat.reshape(spec_image.shape)
        self.preview_orig_np = np.array((spec_image[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        self.preview_recon_np = np.array((recon_image[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

    def train(self):
        ckpt_path = Path(f"{self.args.basename}.pkl"); best_ckpt_path = Path(f"{self.args.basename}_best.pkl")
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.args.lr))
        start_epoch, global_step = 0, 0
        
        is_resuming = ckpt_path.exists()
        
        if is_resuming:
            self.console.print(f"--- Resuming training from {ckpt_path} ---")
            with open(ckpt_path, 'rb') as f: data = pickle.load(f)
            params = data.get('params'); ema_params = data.get('ema_params', params)
            state = TrainStateWithEMA.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=ema_params).replace(opt_state=data['opt_state'])
            start_epoch, global_step = data.get('epoch', 0), data.get('step', 0)
            self.best_val_loss = data.get('best_val_loss', float('inf'))
        else:
            self.console.print("--- Initializing new model ---")
            with jax.default_device(CPU_DEVICE):
                dummy_img = jnp.zeros((1, self.args.image_size, self.args.image_size, 3), dtype=self.dtype)
                dummy_coords = jnp.zeros((1024, 2), dtype=self.dtype)
                params = self.model.init(jax.random.PRNGKey(0), dummy_img, dummy_coords)['params']
            state = TrainStateWithEMA.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=params)

        p_state = replicate(state)
        
        coords_img = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, self.args.image_size), jnp.linspace(-1, 1, self.args.image_size), indexing='ij'), axis=-1).reshape(-1, 2)
        p_coords_img = replicate(coords_img)
        
        loss_weights = (0.5, 20.0, 0.1, 1.0) # l1_img, mel_inv, contrast, l1_spec_inv
        if is_resuming:
            loss_weights_recompile = (0.5, 20.0, 0.1, 1.000001)
        else:
            loss_weights_recompile = loss_weights

        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,), in_axes=(0, 0, 0, None))
        def train_step(state, audio_batch, c_img, current_loss_weights):
            spec_gt_img, mag_gt_spec = audio_to_spec_image(audio_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size), self.args.sample_rate)
            def loss_fn(params):
                recon_img = self.model.apply({'params': params}, spec_gt_img, c_img).reshape(spec_gt_img.shape)
                l1_weighted_img = jnp.mean(jnp.clip(jnp.mean((spec_gt_img + 1.0)/2.0, axis=-1, keepdims=True) + 0.5, 0.2, 5.0) * jnp.abs(recon_img - spec_gt_img))
                
                spec_gt_img_0_1 = (spec_gt_img + 1.0) / 2.0
                recon_img_0_1 = (recon_img + 1.0) / 2.0
                cont_loss = contrastive_loss(spec_gt_img_0_1, recon_img_0_1)
                
                recon_mag_from_img = invert_colormap_to_magnitude(recon_img_0_1, mag_gt_spec.shape[1:])
                
                mel_gt = magnitude_to_mel_db(mag_gt_spec, self.mel_filterbank)
                mel_recon_from_img = magnitude_to_mel_db(recon_mag_from_img, self.mel_filterbank)
                mel_loss_inv = jnp.mean(jnp.abs(mel_gt - mel_recon_from_img))
                l1_spec_loss_inv = jnp.mean(jnp.abs(mag_gt_spec - recon_mag_from_img))
                w_l1_img, w_mel_inv, w_cont, w_l1_spec_inv = current_loss_weights
                total_loss = w_l1_img * l1_weighted_img + w_mel_inv * mel_loss_inv + w_cont * cont_loss + w_l1_spec_inv * l1_spec_loss_inv
                return total_loss, {'total': total_loss, 'l1_img': l1_weighted_img, 'mel_inv': mel_loss_inv, 'contrast': cont_loss, 'l1_spec_inv': l1_spec_loss_inv}

            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, 'devices'); metrics = jax.lax.pmean(metrics, 'devices')
            new_state = state.apply_gradients(grads=grads)
            new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1.0 - 0.999), state.ema_params, new_state.params)
            return new_state.replace(ema_params=new_ema_params), metrics
        
        @partial(jax.pmap, axis_name='devices')
        def eval_step(ema_params, audio_batch, c_img):
            spec_gt_img, mag_gt_spec = audio_to_spec_image(audio_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size), self.args.sample_rate)
            recon_img = self.model.apply({'params': ema_params}, spec_gt_img, c_img).reshape(spec_gt_img.shape)
            recon_mag_from_img = invert_colormap_to_magnitude((recon_img + 1.0) / 2.0, mag_gt_spec.shape[1:])
            mel_gt = magnitude_to_mel_db(mag_gt_spec, self.mel_filterbank)
            mel_recon_from_img = magnitude_to_mel_db(recon_mag_from_img, self.mel_filterbank)
            return jax.lax.pmean(jnp.mean(jnp.abs(mel_gt - mel_recon_from_img)), 'devices')

        self.console.print("--- Setting up data pipelines ---")
        train_ds, num_samples = create_audio_dataset(self.args.audio_dir, self.SEGMENT_LEN_SAMPLES, self.args.sample_rate, self.args.segment_len_secs)
        val_ds, _ = create_audio_dataset(self.args.audio_dir, self.SEGMENT_LEN_SAMPLES, self.args.sample_rate, self.args.segment_len_secs, is_training=False)
        REBATCH_SIZE=200; super_batch_size = self.args.batch_size * self.num_devices * REBATCH_SIZE
        train_ds = train_ds.batch(super_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.args.batch_size * self.num_devices, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        train_iterator = iter(tfds.as_numpy(train_ds))
        
        preview_audio_tensor = next(iter(val_ds.take(1)), None)
        preview_audio_np = preview_audio_tensor.numpy() if preview_audio_tensor is not None else None
        if preview_audio_np is None: self.console.print("[bold red]Validation set is empty![/bold red]"); return

        steps_per_epoch = num_samples // (self.args.batch_size * self.num_devices) if self.args.batch_size > 0 else 0
        total_steps = self.args.epochs * steps_per_epoch if steps_per_epoch > 0 else 0
        
        self.progress = Progress(TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val Mel: {task.fields[val_loss]:.4f}[/]"), BarColumn(), TextColumn("Step {task.fields[step]}/{task.fields[total_steps]}"),"•",TimeRemainingColumn())
        epoch_task = self.progress.add_task("epochs", total=self.args.epochs, completed=start_epoch, val_loss=self.best_val_loss, step=global_step, total_steps=total_steps)
        
        self.console.print(f"--- Compiling JAX functions (one-time cost)... ---")
        compile_super_batch = next(train_iterator); compile_batch = compile_super_batch[:self.args.batch_size * self.num_devices]
        
        p_state, _ = train_step(p_state, common_utils.shard(compile_batch), p_coords_img, loss_weights_recompile)
        
        self.console.print("--- Compilation complete. Starting training. ---")

        last_step_time = time.time()
        with Live(self._generate_layout(), screen=True, auto_refresh=False) as live, ThreadPoolExecutor(max_workers=1) as executor:
            preview_future = None
            while global_step < total_steps and not self.should_shutdown:
                try:
                    super_batch_audio = next(train_iterator).reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, -1)
                    for audio_batch_np in super_batch_audio:
                        if self.should_shutdown: break
                        
                        p_state, metrics = train_step(p_state, common_utils.shard(audio_batch_np), p_coords_img, loss_weights)
                        
                        if global_step % 1 == 0:
                            self.last_losses = unreplicate(metrics)
                            if np.isfinite(self.last_losses.get('total', np.inf)): self.loss_history.append(self.last_losses['total'])
                        global_step += 1; current_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
                        now = time.time(); self.steps_per_sec = 1.0 / (now - last_step_time + 1e-6); last_step_time = now
                        self.progress.update(epoch_task, completed=current_epoch, step=global_step)
                        if global_step % 1 == 0:
                            if preview_future is None or preview_future.done():
                                if preview_future: preview_future.result()
                                preview_future = executor.submit(self._update_preview_task, unreplicate(p_state.ema_params), preview_audio_np)
                            live.update(self._generate_layout(), refresh=True)
                        if global_step > 0 and global_step % self.args.eval_every == 0:
                            self.console.print(f"\n[yellow]Running validation at step {global_step} with Inverted Mel Loss...[/yellow]")
                            val_losses = []
                            for val_batch_audio in val_ds.take(20).as_numpy_iterator():
                                val_loss = unreplicate(eval_step(p_state.ema_params, common_utils.shard(val_batch_audio), p_coords_img))
                                val_losses.append(val_loss)
                            current_val_loss = np.mean(val_losses) if val_losses else float('inf')
                            self.console.print(f"--- 📈 Validation Complete. Current Mel Loss: [bold cyan]{current_val_loss:.4f}[/], Best: [bold green]{self.best_val_loss:.4f}[/] ---")
                            if np.isfinite(current_val_loss) and current_val_loss < self.best_val_loss:
                                self.best_val_loss = current_val_loss
                                self._save_checkpoint(p_state, current_epoch, global_step, self.best_val_loss, best_ckpt_path)
                            self.progress.update(epoch_task, val_loss=self.best_val_loss)
                    if steps_per_epoch > 0 and (global_step // steps_per_epoch) > start_epoch:
                        epoch_just_finished = start_epoch; start_epoch = global_step // steps_per_epoch
                        if (epoch_just_finished + 1) % self.args.save_every == 0:
                            self._save_checkpoint(p_state, epoch_just_finished, global_step, self.best_val_loss, ckpt_path)
                except StopIteration:
                    train_iterator = iter(tfds.as_numpy(train_ds))

        self.console.print("\n--- Training finished. Saving final state... ---")
        if 'p_state' in locals():
            self._save_checkpoint(p_state, global_step // steps_per_epoch if steps_per_epoch > 0 else 0, global_step, self.best_val_loss, ckpt_path)
            
            
def main():
    parser = argparse.ArgumentParser(description="Topological Audio AE - High-Performance Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--basename', type=str, required=True, help="Base name for model checkpoints.")
    model_parser.add_argument('--d-model', type=int, default=128, help="Model dimension for the observer.")
    model_parser.add_argument('--latent-grid-size', type=int, default=16, help="Size of the encoder's output grid.")
    model_parser.add_argument('--image-size', type=int, default=256, help="Resolution of the input spectrogram image.")
    model_parser.add_argument('--sample-rate', type=int, default=44100, help="Target sample rate for all audio.")
    model_parser.add_argument('--segment-len-secs', type=int, default=4, help="Length of audio segments in seconds.")

    p_prep = subparsers.add_parser("prepare-audio", help="Convert raw audio to TFRecords.", parents=[model_parser])
    p_prep.add_argument('--audio-dir', type=str, required=True, help="Directory containing audio files.")
    
    p_train = subparsers.add_parser("train", help="Train the autoencoder.", parents=[model_parser])
    p_train.add_argument('--audio-dir', type=str, required=True, help="Directory containing pre-processed TFRecords.")
    p_train.add_argument('--epochs', type=int, default=100, help="Total number of epochs to train.")
    p_train.add_argument('--batch-size', type=int, default=8, help="Batch size PER DEVICE.")
    p_train.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    p_train.add_argument('--use-bfloat16', action='store_true', help="Use bfloat16 for mixed-precision training.")
    p_train.add_argument('--save-every', type=int, default=1, help="Save a checkpoint every N epochs.")
    p_train.add_argument('--eval-every', type=int, default=1000, help="Run validation every N global steps.")
    
    args = parser.parse_args()
    try:
        if args.command == "prepare-audio": prepare_audio_data(args.audio_dir, args.sample_rate, args.segment_len_secs)
        elif args.command == "train": Trainer(args).train()
    except KeyboardInterrupt:
        print("\n--- Program terminated by user. ---")
    except Exception:
        Console().print_exception(); sys.exit(1)

if __name__ == "__main__":
    main()