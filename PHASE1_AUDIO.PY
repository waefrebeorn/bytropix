#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Topological Coordinate-Based Spectrogram Autoencoder (High-Performance Version)

This script provides a fast, efficient, and stable pipeline for training a
neural audio autoencoder. It combines the simple and effective core model with a
high-performance training harness and an audio-centric multi-part loss function.

The loss function is designed for spectrograms:
1.  Weighted L1 Loss (Image): Focuses on high-energy "bloomed" parts of the colored spectrogram.
2.  Mel-Frequency Loss (Audio): Directly reconstructs the Mel spectrogram for perceptual accuracy.
3.  Contrastive Loss (Image): Preserves the sharpness of transients in the colored spectrogram.

Core Functionality:
1.  prepare-audio: Converts audio files into an efficient TFRecord dataset.
2.  train: Trains the model with a rich UI, live preview, and validation.
3.  compress/decompress: Encodes/decodes audio to/from its latent representation.
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
# --- FIX: Import `Any` ---
from typing import Any, Dict

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
# 1. MODEL ARCHITECTURE (With Dual Decoder Heads)
# =================================================================================================

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        return jnp.cos(delta_f32 / 2) * jnp.cos(chi_f32) + 1j * jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)

class PathModulator(nn.Module):
    latent_grid_size: int; input_image_size: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        num_downsamples = int(math.log2(self.input_image_size / self.latent_grid_size))
        x, features = images, 32
        for i in range(num_downsamples):
            x = nn.Conv(features, (4, 4), (2, 2), name=f"down_conv_{i}", dtype=self.dtype)(x); x = nn.gelu(x); features *= 2
        x = nn.Conv(256, (3, 3), padding='SAME', name="feat_conv", dtype=self.dtype)(x); x = nn.gelu(x)
        path_params = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)
        delta_c = nn.tanh(path_params[..., 0]) * jnp.pi; chi_c = nn.tanh(path_params[..., 1]) * (jnp.pi / 4.0); radius = nn.sigmoid(path_params[..., 2]) * (jnp.pi / 2.0)
        return jnp.stack([delta_c, chi_c, radius], axis=-1)

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = path_params_grid.shape
        path_params = path_params_grid.reshape(B, H * W, C)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta); chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path)
        complex_measurement = jnp.stack([jnp.cumprod(t_co_steps, axis=-1)[:, :, -1].real, jnp.cumprod(t_co_steps, axis=-1)[:, :, -1].imag], axis=-1)
        return nn.Dense(self.d_model, name="proj", dtype=self.dtype)(complex_measurement).reshape(B, H, W, self.d_model)

class PositionalEncoding(nn.Module):
    num_freqs: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        freqs = 2.**jnp.arange(self.num_freqs, dtype=self.dtype) * jnp.pi
        return jnp.concatenate([x] + [f(x * freq) for freq in freqs for f in (jnp.sin, jnp.cos)], axis=-1)

class CoordinateDecoder(nn.Module):
    # --- FIX: Move non-default arguments before default arguments ---
    d_model: int
    n_fft: int
    mlp_width: int = 128
    mlp_depth: int = 3
    dtype: Any = jnp.float32
    # --- END FIX ---
    
    @nn.compact
    def __call__(self, feature_grid: jnp.ndarray, coords_img: jnp.ndarray, coords_spec: jnp.ndarray):
        B = feature_grid.shape[0]
        pos_encoder = PositionalEncoding(num_freqs=10, dtype=self.dtype)
        
        def sample_features(coords):
            coords_rescaled = (coords + 1) / 2 * (jnp.array(feature_grid.shape[1:3], dtype=self.dtype) - 1)
            def sample_one_image(single_feature_grid):
                grid_chw = single_feature_grid.transpose(2, 0, 1)
                return jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw).T
            return jax.vmap(sample_one_image)(feature_grid)

        # Image Head (outputs RGB pixels)
        encoded_coords_img = pos_encoder(coords_img)
        sampled_features_img = sample_features(coords_img)
        mlp_input_img = jnp.concatenate([jnp.repeat(encoded_coords_img[None, :, :], B, axis=0), sampled_features_img], axis=-1)
        
        h_img = mlp_input_img
        for i in range(self.mlp_depth):
            h_img = nn.Dense(self.mlp_width, name=f"mlp_img_{i}", dtype=self.dtype)(h_img); h_img = nn.gelu(h_img)
        pixels_out = nn.Dense(3, name="mlp_img_out", dtype=self.dtype)(h_img)

        # Spectrogram Head (outputs magnitude bins)
        encoded_coords_spec = pos_encoder(coords_spec)
        sampled_features_spec = sample_features(coords_spec)
        mlp_input_spec = jnp.concatenate([jnp.repeat(encoded_coords_spec[None, :, :], B, axis=0), sampled_features_spec], axis=-1)
        
        h_spec = mlp_input_spec
        for i in range(self.mlp_depth):
            h_spec = nn.Dense(self.mlp_width, name=f"mlp_spec_{i}", dtype=self.dtype)(h_spec); h_spec = nn.gelu(h_spec)
        spec_bins_out = nn.Dense(1, name="mlp_spec_out", dtype=self.dtype)(h_spec)

        return nn.tanh(pixels_out), nn.relu(spec_bins_out)

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int; input_image_size: int; n_fft: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(latent_grid_size=self.latent_grid_size, input_image_size=self.input_image_size, dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(d_model=self.d_model, n_fft=self.n_fft, dtype=self.dtype)
    def __call__(self, images, coords_img, coords_spec):
        path_params = self.encode(images)
        return self.decode(path_params, coords_img, coords_spec)
    def encode(self, images):
        return self.modulator(images)
    def decode(self, path_params, coords_img, coords_spec):
        feature_grid = self.observer(path_params)
        return self.coord_decoder(feature_grid, coords_img, coords_spec)

class TrainStateWithEMA(train_state.TrainState):
    ema_params: Any

# =================================================================================================
# 2. AUDIO & DATA HANDLING
# =================================================================================================

COLORMAP_LUT = jnp.array(matplotlib.cm.magma(np.linspace(0, 1, 256))[:, :3])

@partial(jax.jit, static_argnames=('n_fft', 'hop_length', 'output_size'))
def audio_to_spec_image(audio_batch: chex.Array, n_fft: int, hop_length: int, output_size: tuple[int, int]):
    specs_complex = jax.vmap(lambda w: jax.scipy.signal.stft(w, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft)[2])(audio_batch.astype(jnp.float32))
    spec_mag = jnp.abs(specs_complex)
    spec_db = 10.0 * jnp.log10(jnp.maximum(1e-6, spec_mag**2))
    
    def normalize_db(db_spec):
        p5, p95 = jnp.percentile(db_spec, 5), jnp.percentile(db_spec, 95)
        db_spec_clipped = jnp.clip(db_spec, p5, p95)
        return (db_spec_clipped - db_spec_clipped.min()) / (jnp.maximum(db_spec_clipped.max() - db_spec_clipped.min(), 1e-6))
    spec_norm = jax.vmap(normalize_db)(spec_db)
    
    indices = (spec_norm * 255).astype(jnp.int32)
    spec_color = COLORMAP_LUT[indices]
    spec_resized = jax.image.resize(spec_color, (spec_color.shape[0], *output_size, 3), 'bilinear')
    return spec_resized, spec_mag

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
# 3. AUDIO-CENTRIC LOSS FUNCTIONS
# =================================================================================================

def create_mel_filterbank(sr=44100, n_fft=1024, n_mels=128, f_min=0.0, f_max=None):
    f_max = f_max or sr / 2.0
    num_bins = n_fft // 2 + 1
    weights = np.zeros((n_mels, num_bins))
    fft_freqs = np.linspace(0, sr / 2.0, num_bins)
    mel_min = 1127.0 * np.log(1.0 + f_min / 700.0)
    mel_max = 1127.0 * np.log(1.0 + f_max / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700.0 * (np.exp(mel_pts / 1127.0) - 1.0)

    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i+1], hz_pts[i+2]
        lower_slope = (fft_freqs - left) / (center - left)
        upper_slope = (right - fft_freqs) / (right - center)
        weights[i, :] = np.maximum(0, np.minimum(lower_slope, upper_slope))
    return np.array(weights)

def magnitude_to_mel_db(mag_spec, mel_filterbank):
    power_spec = mag_spec**2
    mel_spec = jnp.einsum('...ft,mf->...mt', power_spec, mel_filterbank)
    return 10.0 * jnp.log10(jnp.maximum(1e-6, mel_spec))

@jax.jit
def contrastive_loss(img1, img2):
    def get_grads(img):
        img_lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        grads_x = jax.scipy.signal.convolve(img_lum, jnp.array([[1, 0, -1]]), mode='same')
        grads_y = jax.scipy.signal.convolve(img_lum, jnp.array([[1], [0], [-1]]), mode='same')
        return jnp.sqrt(grads_x**2 + grads_y**2)
    return jnp.mean(jnp.abs(jax.vmap(get_grads)(img1) - jax.vmap(get_grads)(img2)))

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
        self.N_FFT = 1024; self.HOP_LENGTH = 256
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, n_fft=self.N_FFT, dtype=self.dtype)
        
        self.loss_history = deque(maxlen=200)
        self.SEGMENT_LEN_SAMPLES = args.sample_rate * args.segment_len_secs
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
        layout = Layout(name="root")
        layout.split(
            Layout(Panel(f"🎤 [bold]Spectrogram Autoencoder[/] | Model: [cyan]{self.args.basename}[/]", expand=False), size=3),
            Layout(ratio=1, name="main"), Layout(self.progress, size=3)
        )
        layout["main"].split_row(Layout(name="left", minimum_size=40), Layout(name="right"))
        
        mem, util = self._get_gpu_stats()
        stats_tbl = Table.grid(expand=True, padding=(0,1)); stats_tbl.add_column(style="dim", width=15); stats_tbl.add_column(justify="right")
        stats_tbl.add_row("Total Loss", f"[bold green]{self.last_losses.get('total', 0):.4f}[/]")
        stats_tbl.add_row("L1-Weighted", f"{self.last_losses.get('l1_w', 0):.4f}")
        stats_tbl.add_row("Mel Loss", f"{self.last_losses.get('mel', 0):.4f}")
        stats_tbl.add_row("Contrast Loss", f"{self.last_losses.get('contrast', 0):.4f}")
        stats_tbl.add_row("Steps/sec", f"[blue]{self.steps_per_sec:.2f}[/]")
        stats_tbl.add_row("GPU Mem", f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util", f"[yellow]{util}[/]")
        
        spark_w = max(10, self.console.width - 50) if self.console.width > 50 else 10
        loss_panel = Panel(Align.center(f"[cyan]{self._get_sparkline(self.loss_history, spark_w)}[/]"), title=f"Total Loss History", height=3, border_style="cyan")
        
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
        data = {
            'params': jax.device_get(unrep_state.params), 'ema_params': jax.device_get(unrep_state.ema_params),
            'opt_state': jax.device_get(unrep_state.opt_state), 'epoch': epoch, 'step': step, 'best_val_loss': val_loss
        }
        with open(path, 'wb') as f: pickle.dump(data, f)
        self.console.print(f"--- 💾 Checkpoint saved to {path.name} (Val Loss: {val_loss:.4f}) ---")
    
    def _update_preview_task(self, ema_params, audio_batch):
        spec_image, _ = audio_to_spec_image(audio_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size))
        
        coords_img = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, self.args.image_size), jnp.linspace(-1, 1, self.args.image_size), indexing='ij'), axis=-1).reshape(-1, 2)
        
        @jax.jit
        def predict(params, img, c_img): 
            pixels, _ = self.model.apply({'params': params}, img, c_img, c_img) # dummy spec coords
            return pixels
            
        recon_flat = predict(ema_params, spec_image, coords_img)
        recon_image = recon_flat.reshape(spec_image.shape)

        self.preview_orig_np = np.array((spec_image[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        self.preview_recon_np = np.array((recon_image[0] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)

    def train(self):
        ckpt_path = Path(f"{self.args.basename}.pkl"); best_ckpt_path = Path(f"{self.args.basename}_best.pkl")
        
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.args.lr))
        start_epoch, global_step = 0, 0
        
        if ckpt_path.exists():
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
                params = self.model.init(jax.random.PRNGKey(0), dummy_img, dummy_coords, dummy_coords)['params']
            state = TrainStateWithEMA.create(apply_fn=self.model.apply, params=params, tx=optimizer, ema_params=params)

        p_state = replicate(state)
        
        coords_img = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, self.args.image_size), jnp.linspace(-1, 1, self.args.image_size), indexing='ij'), axis=-1).reshape(-1, 2)
        
        num_time_steps = math.ceil((self.SEGMENT_LEN_SAMPLES) / self.HOP_LENGTH) + 1
        num_freq_bins = self.N_FFT // 2 + 1
        coords_spec = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, num_time_steps), jnp.linspace(-1, 1, num_freq_bins), indexing='ij'), axis=-1).reshape(-1, 2)
        
        p_coords_img = replicate(coords_img)
        p_coords_spec = replicate(coords_spec)

        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,), in_axes=(0, 0, 0, 0, None))
        def train_step(state, audio_batch, c_img, c_spec, loss_weights):
            spec_gt_img, mag_gt_spec = audio_to_spec_image(audio_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size))
            
            def loss_fn(params):
                recon_img_flat, recon_mag_flat = self.model.apply({'params': params}, spec_gt_img, c_img, c_spec)
                recon_img = recon_img_flat.reshape(spec_gt_img.shape)
                recon_mag = recon_mag_flat.reshape(mag_gt_spec.shape)
                
                weights = jnp.clip(jnp.mean((spec_gt_img + 1.0) / 2.0, axis=-1, keepdims=True) + 0.5, 0.2, 5.0)
                l1_weighted = jnp.mean(weights * jnp.abs(recon_img - spec_gt_img))
                
                mel_gt = magnitude_to_mel_db(mag_gt_spec, self.mel_filterbank)
                mel_recon = magnitude_to_mel_db(recon_mag, self.mel_filterbank)
                mel_loss = jnp.mean(jnp.abs(mel_gt - mel_recon))

                cont_loss = contrastive_loss(spec_gt_img, recon_img)
                
                w_l1, w_mel, w_cont = loss_weights
                total_loss = w_l1 * l1_weighted + w_mel * mel_loss + w_cont * cont_loss
                
                return total_loss, {'total': total_loss, 'l1_w': l1_weighted, 'mel': mel_loss, 'contrast': cont_loss}

            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, 'devices'); metrics = jax.lax.pmean(metrics, 'devices')
            
            new_state = state.apply_gradients(grads=grads)
            new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * 0.999 + p * (1.0 - 0.999), state.ema_params, new_state.params)
            
            return new_state.replace(ema_params=new_ema_params), metrics

        self.console.print("--- Setting up data pipelines ---")
        train_ds, num_samples = create_audio_dataset(self.args.audio_dir, self.SEGMENT_LEN_SAMPLES, self.args.sample_rate, self.args.segment_len_secs)
        val_ds, _ = create_audio_dataset(self.args.audio_dir, self.SEGMENT_LEN_SAMPLES, self.args.sample_rate, self.args.segment_len_secs, is_training=False)
        
        REBATCH_SIZE=200; super_batch_size = self.args.batch_size * self.num_devices * REBATCH_SIZE
        train_ds = train_ds.batch(super_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.args.batch_size * self.num_devices, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        train_iterator = iter(tfds.as_numpy(train_ds))
        
        preview_audio_tensor = next(iter(val_ds.take(1)), None)
        if preview_audio_tensor is not None:
            preview_audio_np = preview_audio_tensor.numpy()
        if preview_audio_np is None: self.console.print("[bold red]Validation set is empty![/bold red]"); return

        steps_per_epoch = num_samples // (self.args.batch_size * self.num_devices) if self.args.batch_size > 0 else 0
        total_steps = self.args.epochs * steps_per_epoch if steps_per_epoch > 0 else 0
        
        self.progress = Progress(TextColumn("[bold]Epoch {task.completed}/{task.total} [green]Best Val: {task.fields[val_loss]:.4f}[/]"), BarColumn(), TextColumn("Step {task.fields[step]}/{task.fields[total_steps]}"),"•",TimeRemainingColumn())
        epoch_task = self.progress.add_task("epochs", total=self.args.epochs, completed=start_epoch, val_loss=self.best_val_loss, step=global_step, total_steps=total_steps)
        
        self.console.print(f"--- Compiling JAX functions (one-time cost)... ---")
        compile_super_batch = next(train_iterator); compile_batch = compile_super_batch[:self.args.batch_size * self.num_devices]
        p_state, _ = train_step(p_state, common_utils.shard(compile_batch), p_coords_img, p_coords_spec, (1.0, 0.5, 0.5))
        self.console.print("--- Compilation complete. Starting training. ---")

        last_step_time = time.time()
        with Live(self._generate_layout(), screen=True, auto_refresh=False) as live, ThreadPoolExecutor(max_workers=1) as executor:
            preview_future = None
            while global_step < total_steps and not self.should_shutdown:
                try:
                    super_batch_audio = next(train_iterator).reshape(REBATCH_SIZE, self.args.batch_size * self.num_devices, -1)
                    
                    for audio_batch_np in super_batch_audio:
                        if self.should_shutdown: break
                        p_state, metrics = train_step(p_state, common_utils.shard(audio_batch_np), p_coords_img, p_coords_spec, (1.0, 0.5, 0.5))
                        
                        if global_step % 10 == 0:
                            self.last_losses = jax.device_get(unreplicate(metrics))
                            if np.isfinite(self.last_losses.get('total', np.inf)): self.loss_history.append(self.last_losses['total'])

                        global_step += 1; current_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
                        now = time.time(); self.steps_per_sec = 1.0 / (now - last_step_time + 1e-6); last_step_time = now
                        self.progress.update(epoch_task, completed=current_epoch, step=global_step)
                        
                        if global_step % 20 == 0:
                            if preview_future is None or preview_future.done():
                                if preview_future: preview_future.result()
                                preview_future = executor.submit(self._update_preview_task, jax.device_get(unreplicate(p_state.ema_params)), preview_audio_np)
                            live.update(self._generate_layout(), refresh=True)
                    
                        if global_step > 0 and global_step % self.args.eval_every == 0:
                            self.console.print(f"\n[yellow]Running validation at step {global_step}...[/yellow]")
                            val_losses = []
                            for val_batch in val_ds.take(20).as_numpy_iterator():
                                spec_img, _ = audio_to_spec_image(val_batch, self.N_FFT, self.HOP_LENGTH, (self.args.image_size, self.args.image_size))
                                recon, _ = p_state.apply_fn({'params': p_state.ema_params}, common_utils.shard(spec_img), p_coords_img, p_coords_spec)
                                val_losses.append(jnp.mean(jnp.abs(recon - spec_img.reshape(recon.shape))))
                            
                            current_val_loss = np.mean([unreplicate(l) for l in val_losses]) if val_losses else float('inf')
                            if np.isfinite(current_val_loss) and current_val_loss < self.best_val_loss:
                                self.best_val_loss = current_val_loss
                                self._save_checkpoint(p_state, current_epoch, global_step, self.best_val_loss, best_ckpt_path)
                            self.progress.update(epoch_task, val_loss=self.best_val_loss)

                    if steps_per_epoch > 0 and (global_step // steps_per_epoch) > start_epoch:
                        epoch_just_finished = start_epoch
                        start_epoch = global_step // steps_per_epoch
                        if (epoch_just_finished + 1) % self.args.save_every == 0:
                            self._save_checkpoint(p_state, epoch_just_finished, global_step, self.best_val_loss, ckpt_path)

                except StopIteration:
                    train_iterator = iter(tfds.as_numpy(train_ds))

        self.console.print("\n--- Training finished. Saving final state... ---")
        if 'p_state' in locals():
            final_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
            self._save_checkpoint(p_state, final_epoch, global_step, self.best_val_loss, ckpt_path)

def compress_audio(args):
    console = Console()
    console.print(f"--- Loading model from [cyan]{args.model_path}[/cyan] ---")
    with open(args.model_path, 'rb') as f: data = pickle.load(f)
    model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, n_fft=1024)
    
    @jax.jit
    def encode_fn(spec_image_batch):
        return model.apply({'params': data['ema_params']}, spec_image_batch, method=model.encode)

    console.print(f"--- Loading and processing audio file [cyan]{args.input_file}[/cyan] ---")
    waveform, sr = torchaudio.load(args.input_file)
    if sr != args.sample_rate: waveform = torchaudio.transforms.Resample(sr, args.sample_rate)(waveform)
    if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)

    segment_len_samples = args.sample_rate * args.segment_len_secs
    segments = list(waveform.split(segment_len_samples, dim=1))
    if segments[-1].shape[1] < segment_len_samples:
        padding = torch.zeros(1, segment_len_samples - segments[-1].shape[1])
        segments[-1] = torch.cat([segments[-1], padding], dim=1)
    
    audio_batch_np = np.stack([s.numpy().squeeze() for s in segments])
    spec_images, _ = audio_to_spec_image(audio_batch_np, 1024, 256, (args.image_size, args.image_size))

    path_params = jax.device_get(encode_fn(spec_images))

    np.savez_compressed(args.output_file, latents=path_params)
    console.print(f"✅ Compression complete. Saved to [cyan]{args.output_file}[/cyan]")

def decompress_to_image(args):
    console = Console()
    console.print(f"--- Loading model from [cyan]{args.model_path}[/cyan] ---")
    with open(args.model_path, 'rb') as f: data = pickle.load(f)
    model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size, input_image_size=args.image_size, n_fft=1024)
    
    latents = np.load(args.input_file)['latents']
    coords_img = jnp.mgrid[-1:1:args.image_size*1j, -1:1:args.image_size*1j].transpose(1, 2, 0).reshape(-1, 2)
    
    @jax.jit
    def decode_fn(path_params_batch):
        pixels, _ = model.apply({'params': data['ema_params']}, path_params_batch, coords_img, coords_img, method=model.decode)
        return pixels

    recon_images = []
    for i in tqdm(range(latents.shape[0]), desc="Decoding"):
        latents_batch = jnp.expand_dims(latents[i], axis=0)
        pixels_flat = decode_fn(latents_batch)
        recon_images.append(jax.device_get(pixels_flat.reshape(1, args.image_size, args.image_size, 3)))
    
    full_recon = np.concatenate(recon_images, axis=0)
    full_recon_uint8 = np.array((np.clip(full_recon * 0.5 + 0.5, 0, 1) * 255)).astype(np.uint8)
    
    grid_size = math.ceil(math.sqrt(full_recon_uint8.shape[0]))
    tile_img = Image.new('RGB', (grid_size * args.image_size, grid_size * args.image_size))
    for idx, frame_img in enumerate(full_recon_uint8):
        row, col = divmod(idx, grid_size)
        tile_img.paste(Image.fromarray(frame_img), (col * args.image_size, row * args.image_size))
        
    tile_img.save(args.output_file)
    console.print(f"✅ Decompression complete. Saved to [cyan]{args.output_file}[/cyan]")

def main():
    parser = argparse.ArgumentParser(description="Topological Audio AE - High-Performance Trainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--basename', type=str, required=True)
    model_parser.add_argument('--d-model', type=int, default=128)
    model_parser.add_argument('--latent-grid-size', type=int, default=16)
    model_parser.add_argument('--image-size', type=int, default=256)
    model_parser.add_argument('--sample-rate', type=int, default=44100)
    model_parser.add_argument('--segment-len-secs', type=int, default=4)

    p_prep = subparsers.add_parser("prepare-audio", help="Convert raw audio to TFRecords.", parents=[model_parser])
    p_prep.add_argument('--audio-dir', type=str, required=True)
    
    p_train = subparsers.add_parser("train", help="Train the autoencoder.", parents=[model_parser])
    p_train.add_argument('--audio-dir', type=str, required=True)
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--batch-size', type=int, default=8, help="Batch size PER DEVICE.")
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--use-bfloat16', action='store_true')
    p_train.add_argument('--save-every', type=int, default=1, help="Save checkpoint every N epochs.")
    p_train.add_argument('--eval-every', type=int, default=1000, help="Run validation every N global steps.")
    
    p_compress = subparsers.add_parser("compress", help="Encode audio to a latent file.", parents=[model_parser])
    p_compress.add_argument('--input-file', type=str, required=True); p_compress.add_argument('--output-file', type=str, required=True); p_compress.add_argument('--model-path', type=str, required=True)
    p_decompress = subparsers.add_parser("decompress", help="Decode a latent file to a spectrogram image.", parents=[model_parser])
    p_decompress.add_argument('--input-file', type=str, required=True); p_decompress.add_argument('--output-file', type=str, required=True); p_decompress.add_argument('--model-path', type=str, required=True)

    args = parser.parse_args()
    try:
        if args.command == "prepare-audio": prepare_audio_data(args.audio_dir, args.sample_rate, args.segment_len_secs)
        elif args.command == "train": Trainer(args).train()
        elif args.command == "compress": compress_audio(args)
        elif args.command == "decompress": decompress_to_image(args)
    except KeyboardInterrupt:
        print("\n--- Program terminated by user. ---")
    except Exception:
        Console().print_exception(); sys.exit(1)

if __name__ == "__main__":
    main()