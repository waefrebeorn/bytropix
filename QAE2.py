import os
# --- Environment Setup for JAX/TensorFlow ---
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
import time
from typing import Any, Sequence, Dict
import sys
import argparse
import signal
from functools import partial
from pathlib import Path
from collections import deque
from PIL import Image
import jax.scipy.ndimage

# --- JAX Configuration ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_threefry_partitionable', True)

# --- Dependency Checks and Imports ---
try:
    import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
    from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
    from rich.text import Text
    import pynvml; pynvml.nvmlInit()
    from tqdm import tqdm
except ImportError:
    print("[FATAL] A required dependency is missing (tensorflow, rich, pynvml, tqdm). Please install them.")
    sys.exit(1)

# --- Generative Command & GUI Preview Dependencies ---
try:
    import clip
    import torch
    _clip_device = "cuda" if "cuda" in str(jax.devices()[0]).lower() else "cpu"
except ImportError:
    print("[Warning] `clip-by-openai` or `torch` not found. Generative commands will not be available.")
    clip, torch = None, None
try:
    from rich_pixels import Pixels
except ImportError:
    print("[Warning] `rich-pixels` not found. Visual preview in GUI will be disabled. Run: pip install rich-pixels")
    Pixels = None

# =================================================================================================
# 1. MATHEMATICAL & PHYSICAL FOUNDATIONS (FROM TOPOLOGICAL AE)
# =================================================================================================

class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2) * jnp.cos(chi_f32)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

# =================================================================================================
# 2. MODEL ARCHITECTURE (HYBRID MODEL)
# =================================================================================================

class PathModulator(nn.Module):
    latent_grid_size: int = 16
    dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, (4, 4), (2, 2), name="conv1", dtype=self.dtype)(images); x = nn.gelu(x)
        x = nn.Conv(64, (4, 4), (2, 2), name="conv2", dtype=self.dtype)(x); x = nn.gelu(x)
        x = nn.Conv(128, (4, 4), (2, 2), name="conv3", dtype=self.dtype)(x); x = nn.gelu(x)
        x = nn.Conv(256, (4, 4), (2, 2), name="conv4", dtype=self.dtype)(x); x = nn.gelu(x)
        target_stride = 32 // self.latent_grid_size
        x = nn.Conv(256, (3, 3), (target_stride, target_stride), padding='SAME', name="conv5", dtype=self.dtype)(x); x = nn.gelu(x)
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
        B = feature_grid.shape[0]
        pos_encoder = PositionalEncoding(self.num_freqs, dtype=self.dtype)
        encoded_coords = pos_encoder(coords)
        coords_rescaled = (coords + 1) / 2 * (jnp.array(feature_grid.shape[1:3], dtype=self.dtype) - 1)
        
        def sample_one_image(single_feature_grid):
            grid_chw = single_feature_grid.transpose(2, 0, 1)
            sampled_channels = jax.vmap(lambda g: jax.scipy.ndimage.map_coordinates(g, coords_rescaled.T, order=1, mode='reflect'))(grid_chw)
            return sampled_channels.T
        sampled_features = jax.vmap(sample_one_image)(feature_grid)
        
        encoded_coords_tiled = jnp.repeat(encoded_coords[None, :, :], B, axis=0)
        mlp_input = jnp.concatenate([encoded_coords_tiled, sampled_features], axis=-1)
        h = mlp_input
        for i in range(self.mlp_depth):
            h = nn.Dense(self.mlp_width, name=f"mlp_{i}", dtype=self.dtype)(h); h = nn.gelu(h)
        output_pixels = nn.Dense(3, name="mlp_out", dtype=self.dtype)(h)
        return nn.tanh(output_pixels)

class TopologicalCoordinateGenerator(nn.Module):
    d_model: int; latent_grid_size: int = 16; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(latent_grid_size=self.latent_grid_size, name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, name="observer", dtype=self.dtype)
        self.coord_decoder = CoordinateDecoder(d_model=self.d_model, name="coord_decoder", dtype=self.dtype)
    def __call__(self, images, coords):
        path_params = self.modulator(images)
        feature_grid = self.observer(path_params)
        return self.coord_decoder(feature_grid, coords), path_params
    def decode(self, path_params, coords):
        feature_grid = self.observer(path_params)
        return self.coord_decoder(feature_grid, coords)

# =================================================================================================
# 3. DATA HANDLING & 4. TRAINING/VISUALIZATION LOGIC
# =================================================================================================
def prepare_data(image_dir: str):
    base_path = Path(image_dir); record_file = base_path/"data_512x512.tfrecord"; info_file=base_path/"dataset_info.pkl"
    if record_file.exists(): print(f"✅ Data files found in {image_dir}. Skipping preparation."); return
    print(f"--- Preparing 512x512 data from {image_dir} ---")
    image_paths = sorted([p for p in base_path.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
    if not image_paths: print(f"[FATAL] No images found in {image_dir}."), sys.exit(1)
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for path in tqdm(image_paths, "Processing Images"):
            try:
                img = Image.open(path).convert("RGB").resize((512,512),Image.Resampling.LANCZOS)
                img_bytes = tf.io.encode_jpeg(np.array(img),quality=95).numpy()
                ex=tf.train.Example(features=tf.train.Features(feature={'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
                writer.write(ex.SerializeToString())
            except Exception as e: print(f"Skipping {path}: {e}")
    with open(info_file,'wb') as f: pickle.dump({'num_samples':len(image_paths)},f)
    print(f"✅ Data preparation complete.")

def create_dataset(image_dir: str, batch_size: int, is_training: bool = True):
    record_file = Path(image_dir)/"data_512x512.tfrecord"
    if not record_file.exists(): raise FileNotFoundError(f"{record_file} not found. Run 'prepare-data' first.")
    def _parse(proto):
        f={'image':tf.io.FixedLenFeature([],tf.string)}; p=tf.io.parse_single_example(proto,f)
        img=(tf.cast(tf.io.decode_jpeg(p['image'],3),tf.float32)/127.5)-1.0; img.set_shape([512,512,3]); return img
    ds = tf.data.TFRecordDataset(str(record_file)).map(_parse,num_parallel_calls=tf.data.AUTOTUNE)
    if is_training: ds = ds.shuffle(512).repeat()
    return ds.batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)

class Trainer:
    def __init__(self, args):
        self.args = args; self.key = jax.random.PRNGKey(args.seed); self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.num_devices = jax.local_device_count()
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size)
        if Pixels is None: print("[Warning] `rich-pixels` not installed. Visual preview will be disabled. Run: pip install rich-pixels")
        self.recon_loss_history = deque(maxlen=200)

    def _get_gpu_stats(self):
        try:
            h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except Exception:
            return "N/A", "N/A"

    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)
        
    def _update_preview_panel(self, panel, original_img, recon_img):
        if Pixels is None:
            panel.renderable = Align.center(Text("Install `rich-pixels` for previews", style="yellow"))
            return panel
        term_width = 64
        h, w, _ = original_img.shape
        term_height = int(term_width * (h / w) * 0.5)
        original_pil = Image.fromarray(original_img).resize((term_width, term_height), Image.Resampling.LANCZOS)
        recon_pil = Image.fromarray(recon_img).resize((term_width, term_height), Image.Resampling.LANCZOS)
        original_pix = Pixels.from_image(original_pil); recon_pix = Pixels.from_image(recon_pil)
        preview_table = Table.grid(expand=True)
        preview_table.add_column(justify="center", ratio=1); preview_table.add_column(justify="center", ratio=1)
        preview_table.add_row(Text("Original", justify="center"), Text("Reconstruction", justify="center"))
        preview_table.add_row(original_pix, recon_pix)
        panel.renderable = preview_table
        return panel

    def train(self):
        ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.args.lr))

        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path,'rb') as f: data=pickle.load(f)
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=data['params'],tx=optimizer).replace(opt_state=data['opt_state'])
            start_epoch = data.get('epoch',0)+1
        else:
            print("--- Initializing new model ---")
            with jax.default_device(CPU_DEVICE):
                dummy_images = jnp.zeros((1, 512, 512, 3)); dummy_coords = jnp.zeros((1024, 2))
                params = self.model.init(jax.random.PRNGKey(0), dummy_images, dummy_coords)['params']
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=params,tx=optimizer); start_epoch=0

        p_state = replicate(state)

        # --- MODIFIED: Deterministic, Whole-Image Training Step ---
        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,1))
        def train_step(state, batch):
            image_res = batch.shape[1] # e.g., 512

            # 1. Create a deterministic, full grid of coordinates for the entire image.
            x = jnp.linspace(-1, 1, image_res)
            coords = jnp.stack(jnp.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)

            # 2. The target pixels are the entire image, reshaped.
            target_pixels = batch.reshape(batch.shape[0], -1, 3)

            def loss_fn(params):
                # 3. The encoder runs ONCE on the full image to get a stable latent representation.
                path_params = self.model.apply({'params': params}, batch, method=lambda m, i: m.modulator(i))
                feature_grid = self.model.apply({'params': params}, path_params, method=lambda m, p: m.observer(p))

                # --- 4. Memory Management: Decode in deterministic patches ---
                num_pixels = coords.shape[0]
                decoder_patch_size = 8192
                num_patches = (num_pixels + decoder_patch_size - 1) // decoder_patch_size

                def process_patch(i):
                    start_idx = i * decoder_patch_size
                    coord_patch = jax.lax.dynamic_slice_in_dim(coords, start_idx, decoder_patch_size, axis=0)
                    target_patch = jax.lax.dynamic_slice_in_dim(target_pixels, start_idx, decoder_patch_size, axis=1)
                    recon_patch = self.model.apply(
                        {'params': params}, feature_grid, coord_patch, method=lambda m, fg, c: m.coord_decoder(fg, c)
                    )
                    return jnp.sum(jnp.abs(target_patch - recon_patch))

                total_loss = jnp.sum(jax.vmap(process_patch)(jnp.arange(num_patches)))
                return total_loss / num_pixels

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            grads = jax.lax.pmean(grads, 'devices')
            return state.apply_gradients(grads=grads), jax.lax.pmean(loss, 'devices')

        # --- Preview function (unchanged) ---
        @partial(jax.jit, static_argnames=('resolution', 'patch_size'))
        def generate_preview(params, image_batch, resolution=128, patch_size=64):
            latent_params = self.model.apply({'params': params}, image_batch, method=lambda m, i: m.modulator(i))
            x = jnp.linspace(-1, 1, resolution); grid_x, grid_y = jnp.meshgrid(x, x, indexing='ij')
            full_coords = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
            num_patches = (resolution // patch_size)**2 if resolution >= patch_size else 1
            coords_patched = jnp.array_split(full_coords, num_patches)
            pixels = []
            for patch_coords in coords_patched:
                pixels.append(self.model.apply({'params': params}, latent_params, patch_coords, method=lambda m, p, c: m.decode(p, c)))
            return jnp.concatenate(pixels, axis=1).reshape(1, resolution, resolution, 3)

        # --- Dataset & UI Logic ---
        dataset = create_dataset(self.args.image_dir, self.args.batch_size*self.num_devices)
        preview_batch = next(dataset.as_numpy_iterator())[0:1]
        it = dataset.as_numpy_iterator()
        with open(Path(self.args.image_dir)/"dataset_info.pkl",'rb') as f:
            steps_per_epoch = pickle.load(f)['num_samples'] // (self.args.batch_size * self.num_devices)

        # --- MODIFIED: Corrected Compilation Order ---
        print("--- Compiling JAX functions (one-time cost)... ---")
        # 1. Compile the preview function first. It does NOT donate/invalidate its arguments.
        print("   Compiling preview function...")
        generate_preview(unreplicate(p_state.params), preview_batch)

        # 2. Now, compile the training step. This call WILL donate and invalidate the initial p_state.
        print("   Compiling train_step function...")
        compile_batch = common_utils.shard(np.repeat(preview_batch, self.num_devices, axis=0))
        # We need a new state variable here because the old `p_state` is consumed.
        p_state, _ = train_step(p_state, compile_batch)

        print("--- Compilation complete. Starting training. ---")

        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Topological Coordinate Generator (512x512)[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2)); layout["left"].split(Layout(name="stats"), Layout(name="preview"))
        preview_panel = Panel("...", title="[bold]🔎 Preview[/]", border_style="green", height=18)
        layout["left"]["preview"].update(preview_panel)
        progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn())
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs); layout['footer'].update(progress)

        epoch_for_save = start_epoch
        with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            try:
                for epoch in range(start_epoch, self.args.epochs):
                    epoch_for_save = epoch
                    progress.update(epoch_task, completed=0, epoch=epoch+1)

                    for step in range(steps_per_epoch):
                        if self.should_shutdown: break
                        batch_np = next(it)
                        p_state, loss = train_step(p_state, common_utils.shard(batch_np))

                        if step % 1 == 0:
                            loss_val = unreplicate(loss); self.recon_loss_history.append(loss_val)
                            stats_tbl=Table(show_header=False,box=None,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
                            stats_tbl.add_row("Image Loss",f"[bold green]{loss_val:.4f}[/]"); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util",f"[yellow]{util}[/]")
                            layout["left"]["stats"].update(Panel(stats_tbl,title="[bold]📊 Stats[/]",border_style="blue"))
                            spark_w = max(1, (live.console.width*2//3)-10)
                            recon_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.recon_loss_history,spark_w)}[/]"),title=f"Reconstruction Loss (Full 512x512 Image)",height=3, border_style="cyan")
                            layout["right"].update(Panel(recon_panel,title="[bold]📉 Losses[/]",border_style="magenta"))

                        if step > 0 and step % 25 == 0:
                            recon = generate_preview(unreplicate(p_state.params), preview_batch)
                            orig_np = np.array((preview_batch[0]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                            recon_np = np.array((recon[0]*0.5+0.5).clip(0,1)*255, dtype=np.uint8)
                            self._update_preview_panel(preview_panel, orig_np, recon_np)

                        progress.update(epoch_task, advance=1)
                    if self.should_shutdown: break

                    if jax.process_index() == 0:
                        state_to_save = unreplicate(p_state)
                        with open(ckpt_path, 'wb') as f: pickle.dump({'params':jax.device_get(state_to_save.params),'opt_state':jax.device_get(state_to_save.opt_state),'epoch':epoch}, f)
                        live.console.print(f"--- :floppy_disk: Checkpoint saved for epoch {epoch+1} ---")

            except Exception as e:
                live.stop()
                print("\n[bold red]FATAL: Training loop crashed![/bold red]")
                raise e

        if jax.process_index() == 0 and 'p_state' in locals():
            print("\n--- Training finished or interrupted. Saving final state... ---")
            state_to_save = unreplicate(p_state)
            with open(ckpt_path, 'wb') as f: pickle.dump({'params':jax.device_get(state_to_save.params),'opt_state':jax.device_get(state_to_save.opt_state),'epoch':epoch_for_save}, f)
            print("--- :floppy_disk: Final state saved. ---")




class Compressor:
    def __init__(self, args):
        self.args = args
        self.model = TopologicalCoordinateGenerator(d_model=args.d_model, latent_grid_size=args.latent_grid_size)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d_512.pkl")
        if not model_path.exists(): print(f"[FATAL] Model file not found at {model_path}. Train a model first."), sys.exit(1)
        if jax.process_index() == 0: print(f"--- Loading compressor model from {model_path} ---")
        with open(model_path, 'rb') as f: self.params = pickle.load(f)['params']

    @partial(jax.jit, static_argnames=('self',))
    def _encode(self, image_batch):
        return self.model.apply({'params': self.params}, image_batch, method=lambda module, images: module.modulator(images))

    @partial(jax.jit, static_argnames=('self', 'resolution', 'patch_size'))
    def _decode_batched(self, latent_batch, resolution=512, patch_size=128):
        x = jnp.linspace(-1, 1, resolution); grid_x, grid_y = jnp.meshgrid(x, x, indexing='ij')
        full_coords = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        feature_grid = self.model.apply({'params': self.params}, latent_batch, method=lambda module, latents: module.observer(latents))
        num_patches = (resolution // patch_size) ** 2 if resolution >= patch_size else 1
        coords_patched = jnp.array_split(full_coords, num_patches)
        pixels = [self.model.apply({'params': self.params}, feature_grid, p, method=lambda m, fg, c: m.coord_decoder(fg, c)) for p in coords_patched]
        return jnp.concatenate(pixels, axis=1).reshape(1, resolution, resolution, 3)

    def compress(self):
        image_path = Path(self.args.image_path);
        if not image_path.exists(): print(f"[FATAL] Image file not found at {image_path}"), sys.exit(1)
        img = Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        img_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
        image_batch = jnp.expand_dims(img_np, axis=0)
        latent_grid = self._encode(image_batch)
        latent_grid_uint8 = self._quantize_latents(latent_grid)
        output_path = Path(self.args.output_path); np.save(output_path, latent_grid_uint8)
        original_size = image_path.stat().st_size; compressed_size = output_path.stat().st_size
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        print(f"✅ Image compressed successfully to {output_path}"); print(f"   Original size:     {original_size / 1024:.2f} KB"); print(f"   Compressed size:   {compressed_size / 1024:.2f} KB ({self.args.latent_grid_size}x{self.args.latent_grid_size}x3 uint8 grid)"); print(f"   Compression Ratio: {ratio:.2f}x")

    def decompress(self):
        compressed_path = Path(self.args.compressed_path);
        if not compressed_path.exists(): print(f"[FATAL] Compressed file not found at {compressed_path}"), sys.exit(1)
        latent_grid_uint8 = np.load(compressed_path)
        latent_grid = self._dequantize_latents(latent_grid_uint8)
        latent_batch = jnp.expand_dims(latent_grid, axis=0)
        print("--- Decompressing (rendering 512x512 image)... This may take a moment. ---")
        reconstruction_batch = self._decode_batched(latent_batch, resolution=512, patch_size=256)
        recon_np = np.array(reconstruction_batch[0]); recon_np = ((recon_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        recon_img = Image.fromarray(recon_np)
        output_path = Path(self.args.output_path); recon_img.save(output_path)
        print(f"✅ Decompressed 512x512 image saved to {output_path}")

    def _quantize_latents(self, latent_grid_float):
        params = latent_grid_float[0]; delta, chi, radius = params[..., 0], params[..., 1], params[..., 2]
        delta_norm = (delta / jnp.pi) * 0.5 + 0.5; chi_norm = (chi / (jnp.pi / 4.0)) * 0.5 + 0.5; radius_norm = radius / (jnp.pi / 2.0)
        num_bins = 256
        delta_q = np.array(jnp.round(delta_norm * (num_bins - 1)), dtype=np.uint8)
        chi_q = np.array(jnp.round(chi_norm * (num_bins - 1)), dtype=np.uint8)
        radius_q = np.array(jnp.round(radius_norm * (num_bins - 1)), dtype=np.uint8)
        return np.stack([delta_q, chi_q, radius_q], axis=-1)

    def _dequantize_latents(self, latent_grid_uint8):
        num_bins = 256
        latent_grid_float_norm = jnp.asarray(latent_grid_uint8, dtype=jnp.float32) / (num_bins - 1)
        delta_norm, chi_norm, radius_norm = latent_grid_float_norm[..., 0], latent_grid_float_norm[..., 1], latent_grid_float_norm[..., 2]
        delta = (delta_norm - 0.5) * 2.0 * jnp.pi; chi = (chi_norm - 0.5) * 2.0 * (jnp.pi / 4.0); radius = radius_norm * (jnp.pi / 2.0)
        return jnp.stack([delta, chi, radius], axis=-1)

# =================================================================================================
# 5. GENERATIVE COMMANDS (INSPIRED BY CHIMERA)
# =================================================================================================
class Generator(Compressor):
    def __init__(self, args):
        super().__init__(args)
        if clip is None: print("[FATAL] CLIP and PyTorch are required for generative commands."), sys.exit(1)
        self.latent_db_path = Path(self.args.image_dir) / f"latent_database_{self.args.latent_grid_size}grid.pkl"
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)

    def _get_latent_for_text(self, text):
        if not self.latent_db_path.exists():
            print(f"[FATAL] Latent database not found at {self.latent_db_path}. Please run the 'build-db' command first."), sys.exit(1)
        with open(self.latent_db_path, 'rb') as f: db = pickle.load(f)
        image_features = db['clip_features'].to(_clip_device)
        with torch.no_grad(): text_features = self.clip_model.encode_text(clip.tokenize([text]).to(_clip_device))
        image_features /= image_features.norm(dim=-1, keepdim=True); text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=0)
        best_idx = similarity.argmax().item()
        print(f"--- Best match for '{text}' is image #{best_idx} in the dataset ---")
        return jnp.asarray(db['latents'][best_idx])

    def build_db(self):
        print("--- Building latent and CLIP feature database for generative tasks ---")
        dataset = create_dataset(self.args.image_dir, self.args.batch_size, is_training=False)
        all_latents, all_clip_features = [], []
        for batch_np in tqdm(dataset.as_numpy_iterator(), desc="Encoding Images"):
            latents = self._encode(jnp.asarray(batch_np)); all_latents.append(np.array(latents))
            with torch.no_grad():
                batch_torch = torch.from_numpy(batch_np).to(_clip_device).permute(0, 3, 1, 2)
                image_features = self.clip_model.encode_image(batch_torch); all_clip_features.append(image_features.cpu())
        db = {'latents': np.concatenate(all_latents), 'clip_features': torch.cat(all_clip_features)}
        with open(self.latent_db_path, 'wb') as f: pickle.dump(db, f)
        print(f"✅ Database with {len(db['latents'])} entries saved to {self.latent_db_path}")

    def generate(self):
        latent_grid = self._get_latent_for_text(self.args.prompt)
        print(f"--- Generating 512x512 image for prompt: '{self.args.prompt}' ---")
        reconstruction = self._decode_batched(jnp.expand_dims(latent_grid, 0), resolution=512, patch_size=256)
        recon_np = np.array(reconstruction[0]); recon_np = ((recon_np*0.5+0.5).clip(0,1)*255).astype(np.uint8)
        img = Image.fromarray(recon_np)
        save_path = f"GEN_{''.join(c for c in self.args.prompt if c.isalnum())[:50]}.png"; img.save(save_path)
        print(f"✅ Image saved to {save_path}")

    def animate(self):
        print(f"--- Creating animation from '{self.args.start}' to '{self.args.end}' ---")
        latent_start = self._get_latent_for_text(self.args.start); latent_end = self._get_latent_for_text(self.args.end)
        frames = []
        for i in tqdm(range(self.args.steps), desc="Generating Frames"):
            alpha = i / (self.args.steps - 1)
            interp_latent = latent_start * (1 - alpha) + latent_end * alpha
            reconstruction = self._decode_batched(jnp.expand_dims(interp_latent, 0))
            recon_np = np.array(reconstruction[0]); recon_np = ((recon_np*0.5+0.5).clip(0,1)*255).astype(np.uint8)
            frames.append(Image.fromarray(recon_np))
        start_name, end_name = ''.join(c for c in self.args.start if c.isalnum())[:20], ''.join(c for c in self.args.end if c.isalnum())[:20]
        save_path = f"ANIM_{start_name}_to_{end_name}.gif"
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
        print(f"✅ Animation saved to {save_path}")

# =================================================================================================
# 6. MAIN EXECUTION BLOCK (WITH ALL COMMANDS)
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Topological Coordinate Generator for High-Resolution Images")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # --- Infrastructure Commands ---
    p_prep = subparsers.add_parser("prepare-data", help="Convert 512x512 images to TFRecords.")
    p_prep.add_argument('--image-dir', type=str, required=True)
    p_train = subparsers.add_parser("train", help="Train the autoencoder.")
    p_train.add_argument('--image-dir', type=str, required=True); p_train.add_argument('--basename', type=str, required=True)
    p_train.add_argument('--d-model', type=int, default=128); p_train.add_argument('--latent-grid-size', type=int, default=16)
    p_train.add_argument('--epochs', type=int, default=100); p_train.add_argument('--batch-size', type=int, default=4, help="Batch size PER DEVICE.")
    p_train.add_argument('--lr', type=float, default=2e-4); p_train.add_argument('--seed', type=int, default=42)
    # --- Compression Commands ---
    p_comp = subparsers.add_parser("compress", help="Compress a single image to a file.")
    p_comp.add_argument('--image-path', type=str, required=True); p_comp.add_argument('--output-path', type=str, required=True)
    p_comp.add_argument('--basename', type=str, required=True); p_comp.add_argument('--d-model', type=int, default=128); p_comp.add_argument('--latent-grid-size', type=int, default=16)
    p_dcomp = subparsers.add_parser("decompress", help="Decompress a file to a 512x512 image.")
    p_dcomp.add_argument('--compressed-path', type=str, required=True); p_dcomp.add_argument('--output-path', type=str, required=True)
    p_dcomp.add_argument('--basename', type=str, required=True); p_dcomp.add_argument('--d-model', type=int, default=128); p_dcomp.add_argument('--latent-grid-size', type=int, default=16)
    # --- Generative Commands (Inspired by Chimera) ---
    p_db = subparsers.add_parser("build-db", help="Build a latent database for generative tasks.")
    p_db.add_argument('--image-dir', type=str, required=True); p_db.add_argument('--basename', type=str, required=True)
    p_db.add_argument('--d-model', type=int, default=128); p_db.add_argument('--latent-grid-size', type=int, default=16)
    p_db.add_argument('--batch-size', type=int, default=16, help="Batch size for DB creation.")
    p_gen = subparsers.add_parser("generate", help="Generate an image from a text prompt (via nearest latent).")
    p_gen.add_argument('--image-dir', type=str, required=True, help="Path to the original image dataset (for DB lookup)."); p_gen.add_argument('--prompt', type=str, required=True)
    p_gen.add_argument('--basename', type=str, required=True); p_gen.add_argument('--d-model', type=int, default=128); p_gen.add_argument('--latent-grid-size', type=int, default=16)
    p_anim = subparsers.add_parser("animate", help="Create a latent space interpolation between two prompts.")
    p_anim.add_argument('--image-dir', type=str, required=True); p_anim.add_argument('--start', type=str, required=True); p_anim.add_argument('--end', type=str, required=True)
    p_anim.add_argument('--basename', type=str, required=True); p_anim.add_argument('--d-model', type=int, default=128); p_anim.add_argument('--latent-grid-size', type=int, default=16)
    p_anim.add_argument('--steps', type=int, default=60)
    args = parser.parse_args()
    if args.command == "prepare-data": prepare_data(args.image_dir)
    elif args.command == "train": Trainer(args).train()
    elif args.command == "compress": Compressor(args).compress()
    elif args.command == "decompress": Compressor(args).decompress()
    elif args.command == "build-db": Generator(args).build_db()
    elif args.command == "generate": Generator(args).generate()
    elif args.command == "animate": Generator(args).animate()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"), sys.exit(0)