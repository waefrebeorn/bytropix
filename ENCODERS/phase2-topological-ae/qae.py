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
from typing import Any, Sequence, Dict
import sys
import argparse
import signal
from functools import partial
from pathlib import Path
from collections import deque
from PIL import Image
from jax.scipy.linalg import expm # Kept for reference, but not used in new model

# --- JAX Configuration ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_threefry_partitionable', True)

# --- Dependency Checks and Imports ---
try:
    import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
    from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
    import pynvml; pynvml.nvmlInit()
    from tqdm import tqdm
except ImportError:
    print("[FATAL] A required dependency is missing (tensorflow, rich, pynvml, tqdm). Please install them.")
    sys.exit(1)

# =================================================================================================
# 1. MATHEMATICAL & PHYSICAL FOUNDATIONS (UPDATED)
# =================================================================================================

class PoincareSphere:
    """
    A class for performing calculations on the Poincaré sphere, representing
    non-Euclidean geometry relevant to polarization optics and topological encoding.

    This class includes a method based on the findings from
    "Exploiting hidden singularity on the surface of the Poincaré sphere"
    (doi.org/10.1038/s41467-024-48611-y) to calculate a topologically
    protected geometric phase.
    """
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the complex transmittance for a co-polarized channel by
        exploiting a hidden singularity on the Poincaré sphere.

        This method implements the core principle from the research paper.
        The resulting phase of the complex output, arg(t_co), is a topologically
        protected geometric phase determined by the path taken in the parameter
        space (delta, chi) around a singularity. The singularity (zero
        transmittance) occurs at delta=pi, chi=0. Encircling this point in
        the parameter space results in a full 2π phase accumulation.

        Args:
            delta (jnp.ndarray): The eigen birefringence of the system (path parameter).
            chi (jnp.ndarray): The eigen ellipticity of the system (path parameter).

        Returns:
            jnp.ndarray: A complex array `t_co` representing the co-polarized
                         transmittance. The angle of this complex number is the geometric phase.
        """
        # This is the implementation of Equation (2) from the paper.
        delta_f32 = jnp.asarray(delta, dtype=jnp.float32)
        chi_f32 = jnp.asarray(chi, dtype=jnp.float32)
        
        real_part = jnp.cos(delta_f32 / 2) * jnp.cos(chi_f32)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        
        t_co = real_part + 1j * imag_part
        return t_co

# =================================================================================================
# 2. MODEL ARCHITECTURE (UPDATED TO TOPOLOGICAL ENCODING)
# =================================================================================================

class PathModulator(nn.Module):
    """
    Encodes an image into a GRID of topological path parameters.
    Each patch in the image gets its own path description.
    """
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        # Standard CNN to extract features at a per-patch level
        x = nn.Conv(32, (4, 4), (2, 2), name="conv1", dtype=self.dtype)(images)
        x = nn.gelu(x) # (B, 32, 32, 32)
        x = nn.Conv(64, (4, 4), (2, 2), name="conv2", dtype=self.dtype)(x)
        x = nn.gelu(x) # (B, 16, 16, 64)
        x = nn.Conv(128, (3, 3), padding='SAME', name="conv3", dtype=self.dtype)(x)
        x = nn.gelu(x) # (B, 16, 16, 128)

        # Output a 3-channel map. Each "pixel" in this map corresponds to an
        # image patch and contains the 3 parameters for its topological path.
        # Shape: (Batch, 16, 16, 3)
        path_params = nn.Conv(3, (1, 1), name="path_params", dtype=self.dtype)(x)

        # Split and constrain the parameters to sensible ranges
        # delta_center: [-pi, pi], chi_center: [-pi/4, pi/4], radius: [0, pi/2]
        delta_c = nn.tanh(path_params[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params[..., 2]) * (jnp.pi / 2.0)
        
        return jnp.stack([delta_c, chi_c, radius], axis=-1)

class TopologicalObserver(nn.Module):
    """
    Decodes a grid of path parameters by observing the geometric phase
    accumulated by traversing each path on the Poincaré sphere.
    """
    d_model: int
    num_path_steps: int = 16 # Number of points to sample along the circular path
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        # Reshape grid into a list of patches: (B, H, W, 3) -> (B, L, 3)
        B, H, W, C = path_params_grid.shape
        L = H * W
        path_params = path_params_grid.reshape(B, L, C)
        
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]

        # 1. Define the path for each patch.
        # We'll trace a simple circle in the parameter space.
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps)

        # Calculate delta and chi values along the path for EACH patch
        # Shapes: delta_c (B, L), radius (B, L), theta (S,) -> delta (B, L, S)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
        chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)

        # 2. Calculate the complex transmittance at each step of the path.
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path)

        # 3. Accumulate the effect of the path.
        # The total phase is the sum of individual phases, which corresponds
        # to the product of the complex numbers.
        # We take the final value which represents the end point of the integral.
        # Using cumprod gives the evolution along the path.
        accumulated_t_co = jnp.cumprod(t_co_steps, axis=-1)[:, :, -1]

        # 4. The result is a complex number for each patch. We represent it as
        # a 2D real vector (real, imag) for the downstream network.
        # Shape: (B, L, 2)
        complex_measurement = jnp.stack([accumulated_t_co.real, accumulated_t_co.imag], axis=-1)

        # 5. Project the 2D measurement vector up to the final patch dimension.
        output_patches = nn.Dense(self.d_model, name="patch_projector", dtype=self.dtype)(complex_measurement)
        return output_patches


class ImagePatchDecoder(nn.Module):
    """Standard Euclidean decoder to convert patches back to an image."""
    embed_dim: int; dtype: Any = jnp.float32
    @nn.compact
    def __call__(self, x):
        B, L, D = x.shape; H = W = int(np.sqrt(L)); x_reshaped = x.reshape(B, H, W, D); ch = self.embed_dim
        def ResBlock(inner_x, out_ch, name):
            h_in = nn.gelu(nn.LayerNorm(dtype=jnp.float32, epsilon=1e-5, name=f"{name}_ln1")(inner_x))
            h = nn.Conv(out_ch, (3, 3), padding='SAME', dtype=self.dtype, name=f"{name}_conv1")(h_in)
            h = nn.gelu(nn.LayerNorm(dtype=jnp.float32, epsilon=1e-5, name=f"{name}_ln2")(h))
            h = nn.Conv(out_ch, (3, 3), padding='SAME', dtype=self.dtype, name=f"{name}_conv2")(h)
            if inner_x.shape[-1] != out_ch: inner_x = nn.Conv(out_ch, (1, 1), dtype=self.dtype, name=f"{name}_skip")(inner_x)
            return inner_x + h
        h = nn.Conv(ch, (3, 3), padding='SAME', dtype=self.dtype, name="in_conv")(x_reshaped)
        h = ResBlock(h, ch, name="in_resblock")
        for i in range(2):
            B_up, H_up, W_up, C_up = h.shape
            h = jax.image.resize(h, (B_up, H_up * 2, W_up * 2, C_up), 'nearest')
            out_ch = max(ch // 2, 12)
            h = nn.Conv(out_ch, (3, 3), padding='SAME', dtype=self.dtype, name=f"up_{i}_conv")(h)
            h = ResBlock(h, out_ch, name=f"up_{i}_resblock"); ch = out_ch
        h = nn.gelu(nn.LayerNorm(dtype=jnp.float32, epsilon=1e-5, name="out_ln")(h))
        return nn.tanh(nn.Conv(3, (3, 3), padding='SAME', dtype=jnp.float32, name="out_conv")(h))

class TopologicalAutoencoder(nn.Module):
    d_model: int; dtype: Any = jnp.float32
    def setup(self):
        self.modulator = PathModulator(name="modulator", dtype=self.dtype)
        self.observer = TopologicalObserver(d_model=self.d_model, name="observer", dtype=self.dtype)
        self.image_decoder = ImagePatchDecoder(self.d_model, name="image_decoder", dtype=self.dtype)

    def __call__(self, images):
        # The compressed representation IS the grid of path parameters
        path_params = self.modulator(images)

        # The observer generates patches from the topological physics
        reconstructed_patches = self.observer(path_params)

        # The Euclidean decoder builds the final image
        reconstructed_images = self.image_decoder(reconstructed_patches)

        loss = jnp.mean(jnp.abs(images - reconstructed_images))
        aux = {'loss': loss, 'recon_loss': loss, 'reg_loss': jnp.array(0.0), 'reconstructions': reconstructed_images, 'latent': path_params}
        return loss, aux

# =================================================================================================
# 3. DATA HANDLING & 4. TRAINING/VISUALIZATION LOGIC (Unchanged)
# =================================================================================================
def prepare_data(image_dir: str):
    base_path = Path(image_dir); record_file = base_path/"data_64x64.tfrecord"; info_file=base_path/"dataset_info.pkl"
    if record_file.exists(): print(f"✅ Data files found in {image_dir}. Skipping preparation."); return
    print(f"--- Preparing data from {image_dir} ---")
    image_paths = sorted([p for p in base_path.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
    if not image_paths: print(f"[FATAL] No images found in {image_dir}."), sys.exit(1)
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for path in tqdm(image_paths, "Processing Images"):
            try:
                img = Image.open(path).convert("RGB").resize((64,64),Image.Resampling.BICUBIC)
                img_bytes = tf.io.encode_jpeg(np.array(img),quality=95).numpy()
                ex=tf.train.Example(features=tf.train.Features(feature={'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
                writer.write(ex.SerializeToString())
            except Exception as e: print(f"Skipping {path}: {e}")
    with open(info_file,'wb') as f: pickle.dump({'num_samples':len(image_paths)},f)
    print(f"✅ Data preparation complete.")

def create_dataset(image_dir: str, batch_size: int):
    record_file = Path(image_dir)/"data_64x64.tfrecord"
    if not record_file.exists(): raise FileNotFoundError(f"{record_file} not found. Run 'prepare-data' first.")
    def _parse(proto):
        f={'image':tf.io.FixedLenFeature([],tf.string)}; p=tf.io.parse_single_example(proto,f)
        img=(tf.cast(tf.io.decode_jpeg(p['image'],3),tf.float32)/127.5)-1.0; img.set_shape([64,64,3]); return img
    return tf.data.TFRecordDataset(str(record_file)).map(_parse,num_parallel_calls=tf.data.AUTOTUNE).shuffle(1024).repeat().batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)

class Trainer:
    def __init__(self, args):
        self.args = args; self.key = jax.random.PRNGKey(args.seed); self.should_shutdown = False
        signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.num_devices = jax.local_device_count()
        print("🚀 Training in TOPOLOGICAL AUTOENCODER mode.")
        self.model = TopologicalAutoencoder(d_model=args.d_model, dtype=jnp.float32)
        self.recon_loss_history = deque(maxlen=200)

    def _get_gpu_stats(self):
        try:
            h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except: return "N/A", "N/A"

    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v or np.isnan(min_v) or np.isnan(max_v): return " " * w
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

    def train(self):
        ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d.pkl")
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.args.lr))

        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path,'rb') as f: data=pickle.load(f)
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=data['params'],tx=optimizer).replace(opt_state=data['opt_state'])
            start_epoch = data.get('epoch',0)+1
        else:
            print("--- Initializing new model ---")
            with jax.default_device(CPU_DEVICE):
                init_data = jnp.zeros((1, 64, 64, 3), dtype=jnp.float32)
                params = self.model.init(jax.random.PRNGKey(0), init_data)['params']
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=params,tx=optimizer); start_epoch=0

        p_state = replicate(state)
        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,))
        def train_step(state, batch):
            (loss,aux),grads=jax.value_and_grad(state.apply_fn,has_aux=True)({'params':state.params},batch)
            return state.apply_gradients(grads=jax.lax.pmean(grads['params'],'devices')), jax.lax.pmean(aux,'devices')

        dataset = create_dataset(self.args.image_dir, self.args.batch_size*self.num_devices)
        it = dataset.as_numpy_iterator()
        with open(Path(self.args.image_dir)/"dataset_info.pkl",'rb') as f:
            num_samples = pickle.load(f)['num_samples']
        steps_per_epoch = num_samples // (self.args.batch_size * self.num_devices)

        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Topological Autoencoder[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2))
        progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn())
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs); layout['footer'].update(progress)

        epoch_loop = start_epoch
        with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            for epoch in range(start_epoch, self.args.epochs):
                progress.update(epoch_task, completed=0, epoch=epoch+1); epoch_loop=epoch
                for step in range(steps_per_epoch):
                    if self.should_shutdown: break
                    numpy_batch = next(it); jax_batch = jnp.asarray(numpy_batch, dtype=jnp.float32); sharded_batch = common_utils.shard(jax_batch)
                    p_state, metrics = train_step(p_state, sharded_batch)

                    if step%10==0 and jax.process_index() == 0:
                        m=unreplicate(metrics); self.recon_loss_history.append(m['recon_loss'])
                        stats_tbl=Table(show_header=False,box=None,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
                        stats_tbl.add_row("Total Loss",f"[bold green]{m['loss']:.4e}[/]"); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util",f"[yellow]{util}[/]")
                        layout["left"].update(Panel(stats_tbl,title="[bold]📊 Stats[/]",border_style="blue"))
                        loss_tbl=Table(show_header=False,box=None,padding=(0,1)); loss_tbl.add_column(style="dim",width=15); loss_tbl.add_column(justify="right")
                        loss_tbl.add_row("Reconstruction",f"{m['recon_loss']:.4f}")
                        spark_w = max(1, (live.console.width*2//3)-10)
                        recon_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.recon_loss_history,spark_w)}[/]"),title="Reconstruction Loss",height=3, border_style="cyan")
                        layout["right"].update(Panel(Group(loss_tbl,recon_panel),title="[bold]📉 Losses[/]",border_style="magenta"))
                    progress.update(epoch_task,advance=1)
                if self.should_shutdown: break

                if jax.process_index() == 0:
                    state_to_save=unreplicate(p_state)
                    with open(ckpt_path,'wb') as f: pickle.dump({'params':jax.device_get(state_to_save.params),'opt_state':jax.device_get(state_to_save.opt_state),'epoch':epoch},f)
                    live.console.print(f"--- :floppy_disk: Checkpoint saved for epoch {epoch+1} ---")

        if self.should_shutdown and jax.process_index() == 0:
            print("\n--- Shutdown detected. Saving final state... ---")
            state_to_save=unreplicate(p_state)
            with open(ckpt_path,'wb') as f: pickle.dump({'params':jax.device_get(state_to_save.params),'opt_state':jax.device_get(state_to_save.opt_state),'epoch':epoch_loop},f)

class Compressor:
    def __init__(self, args):
        self.args = args
        self.model = TopologicalAutoencoder(d_model=args.d_model, dtype=jnp.float32)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d.pkl")
        if not model_path.exists():
            print(f"[FATAL] Model file not found at {model_path}. Train a model first."), sys.exit(1)
        print(f"--- Loading compressor model from {model_path} ---")
        with open(model_path, 'rb') as f: self.params = pickle.load(f)['params']

    @partial(jax.jit, static_argnames=('self',))
    def _encode(self, image_batch):
        return self.model.apply({'params': self.params}, image_batch, method=lambda module, images: module.modulator(images))

    @partial(jax.jit, static_argnames=('self',))
    def _decode(self, latent_batch):
        reconstructed_patches = self.model.apply({'params': self.params}, latent_batch, method=lambda module, latents: module.observer(latents))
        return self.model.apply({'params': self.params}, reconstructed_patches, method=lambda module, patches: module.image_decoder(patches))

    def compress(self):
        image_path = Path(self.args.image_path);
        if not image_path.exists(): print(f"[FATAL] Image file not found at {image_path}"), sys.exit(1)
        img = Image.open(image_path).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
        img_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
        image_batch = jnp.expand_dims(img_np, axis=0)
        latent_grid = self._encode(image_batch)
        output_path = Path(self.args.output_path); np.save(output_path, np.array(latent_grid))
        original_size = image_path.stat().st_size; compressed_size = output_path.stat().st_size
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        print(f"✅ Image compressed successfully to {output_path}")
        print(f"   Original size:     {original_size / 1024:.2f} KB")
        print(f"   Compressed size:   {compressed_size / 1024:.2f} KB (16x16x3 float32 grid)")
        print(f"   Compression Ratio: {ratio:.2f}x")

    def decompress(self):
        compressed_path = Path(self.args.compressed_path);
        if not compressed_path.exists(): print(f"[FATAL] Compressed file not found at {compressed_path}"), sys.exit(1)
        latent_grid = jnp.asarray(np.load(compressed_path))
        reconstruction_batch = self._decode(latent_grid)
        recon_np = np.array(reconstruction_batch[0]); recon_np = ((recon_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        recon_img = Image.fromarray(recon_np)
        output_path = Path(self.args.output_path); recon_img.save(output_path)
        print(f"✅ Decompressed file saved to {output_path}")

# =================================================================================================
# 5. MAIN EXECUTION BLOCK (Unchanged)
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Topological Autoencoder based on Geometric Phase")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_prep = subparsers.add_parser("prepare-data", help="Convert images to TFRecords.")
    p_prep.add_argument('--image-dir', type=str, required=True)
    p_train = subparsers.add_parser("train", help="Train the autoencoder.")
    p_train.add_argument('--image-dir', type=str, required=True)
    p_train.add_argument('--basename', type=str, required=True)
    p_train.add_argument('--d-model', type=int, default=256, help="Dimension of the patch embedding and image decoder.")
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch-size', type=int, default=32, help="Batch size PER DEVICE.")
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--seed', type=int, default=42)
    p_comp = subparsers.add_parser("compress", help="Compress a single image to a file.")
    p_comp.add_argument('--image-path', type=str, required=True)
    p_comp.add_argument('--output-path', type=str, required=True)
    p_comp.add_argument('--basename', type=str, required=True)
    p_comp.add_argument('--d-model', type=int, default=256)
    p_dcomp = subparsers.add_parser("decompress", help="Decompress a file to an image.")
    p_dcomp.add_argument('--compressed-path', type=str, required=True)
    p_dcomp.add_argument('--output-path', type=str, required=True)
    p_dcomp.add_argument('--basename', type=str, required=True)
    p_dcomp.add_argument('--d-model', type=int, default=256)
    args = parser.parse_args()
    if args.command == "prepare-data": prepare_data(args.image_dir)
    elif args.command == "train": Trainer(args).train()
    elif args.command == "compress": Compressor(args).compress()
    elif args.command == "decompress": Compressor(args).decompress()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"), sys.exit(0)