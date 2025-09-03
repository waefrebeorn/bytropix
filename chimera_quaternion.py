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
from flax.core import freeze, unfreeze
import optax
import numpy as np
import pickle
from typing import Any
import sys
import argparse
import signal
from functools import partial
from pathlib import Path
from collections import deque
import torch
from PIL import Image
import time

# --- JAX Configuration & Dependency Checks ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update('jax_debug_nans', False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_threefry_partitionable', True)
try:
    import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
    from rich.live import Live; from rich.table import Table; from rich.panel import Panel; from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn; from rich.layout import Layout; from rich.console import Group, Console; from rich.align import Align
    import pynvml; pynvml.nvmlInit()
    from tqdm import tqdm
    import clip; _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError: print("[FATAL] A required dependency is missing. Please install tensorflow, rich, pynvml, tqdm, ftfy, clip."), sys.exit(1)

# =================================================================================================
# 1. MATHEMATICAL & PHYSICAL FOUNDATIONS
# =================================================================================================
class PoincareSphere:
    @staticmethod
    def calculate_co_polarized_transmittance(delta: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
        delta_f32, chi_f32 = jnp.asarray(delta, dtype=jnp.float32), jnp.asarray(chi, dtype=jnp.float32)
        real_part = jnp.cos(delta_f32 / 2)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        return real_part + 1j * imag_part

# =================================================================================================
# 2. CORE ARCHITECTURE - QUATERNION DYNAMICS
# =================================================================================================
DTYPE = jnp.float32

class ImageEncoder(nn.Module):
    d_model: int; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, images: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, (4, 4), (2, 2), name="conv1", dtype=self.dtype)(images)
        x = nn.gelu(x); x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32)(x)
        x = nn.Conv(64, (4, 4), (2, 2), name="conv2", dtype=self.dtype)(x)
        x = nn.gelu(x); x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32)(x)
        x = nn.Conv(128, (3, 3), padding='SAME', name="conv3", dtype=self.dtype)(x)
        x = nn.gelu(x); x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32)(x)
        x = jnp.mean(x, axis=(1, 2))
        z = nn.Dense(self.d_model, name="proj_out", dtype=self.dtype)(x)
        return z

class TextEncoder(nn.Module):
    d_model: int; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, text_embeds: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.d_model * 2, name="dense1", dtype=self.dtype)(text_embeds)
        h = nn.gelu(h); h = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32)(h)
        z = nn.Dense(self.d_model, name="dense2", dtype=self.dtype)(h)
        return z

class QuaternionComponentGenerator(nn.Module):
    d_model: int; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.d_model // 2, name="dense1", dtype=self.dtype)(z)
        h = nn.gelu(h)
        scalar_out = nn.Dense(1, name="dense_out", dtype=self.dtype)(h)
        return jnp.squeeze(scalar_out, axis=-1)

class TopologicalObserver(nn.Module):
    d_model: int; num_path_steps: int = 16; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, path_params_grid: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = path_params_grid.shape; L = H * W
        path_params = path_params_grid.reshape(B, L, C)
        delta_c, chi_c, radius = path_params[..., 0], path_params[..., 1], path_params[..., 2]
        theta = jnp.linspace(0, 2 * jnp.pi, self.num_path_steps, dtype=self.dtype)
        delta_path = delta_c[..., None] + radius[..., None] * jnp.cos(theta)
        chi_path = chi_c[..., None] + radius[..., None] * jnp.sin(theta)
        t_co_steps = PoincareSphere.calculate_co_polarized_transmittance(delta_path, chi_path)
        accumulated_t_co = jnp.cumprod(t_co_steps, axis=-1)[:, :, -1]
        complex_measurement = jnp.stack([accumulated_t_co.real, accumulated_t_co.imag], axis=-1)
        return nn.Dense(self.d_model, name="patch_projector", dtype=self.dtype)(complex_measurement)

class TopologicalGenerativeDecoder(nn.Module):
    d_model: int; dtype: Any = DTYPE
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(512, name="modulator_dense1", dtype=self.dtype)(z); h = nn.gelu(h)
        h = nn.Dense(16*16*32, name="modulator_dense2", dtype=self.dtype)(h); h = nn.gelu(h)
        h = h.reshape(h.shape[0], 16, 16, 32)
        path_params_map = nn.Conv(3, (1, 1), name="path_params_conv", dtype=self.dtype)(h)
        delta_c = nn.tanh(path_params_map[..., 0]) * jnp.pi
        chi_c = nn.tanh(path_params_map[..., 1]) * (jnp.pi / 4.0)
        radius = nn.sigmoid(path_params_map[..., 2]) * (jnp.pi / 2.0)
        path_params_grid = jnp.stack([delta_c, chi_c, radius], axis=-1)
        return TopologicalObserver(d_model=self.d_model, dtype=self.dtype, name="observer")(path_params_grid)

class ImagePatchDecoder(nn.Module):
    embed_dim: int; dtype: Any = DTYPE
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
        h = nn.Conv(ch, (3, 3), padding='SAME', dtype=self.dtype, name="in_conv")(x_reshaped); h = ResBlock(h, ch, name="in_resblock")
        for i in range(2):
            B_up, H_up, W_up, C_up = h.shape; h = jax.image.resize(h, (B_up, H_up * 2, W_up * 2, C_up), 'nearest')
            out_ch = max(ch // 2, 12)
            h = nn.Conv(out_ch, (3, 3), padding='SAME', dtype=self.dtype, name=f"up_{i}_conv")(h); h = ResBlock(h, out_ch, name=f"up_{i}_resblock"); ch = out_ch
        h = nn.gelu(nn.LayerNorm(dtype=jnp.float32, epsilon=1e-5, name="out_ln")(h)); return nn.tanh(nn.Conv(3, (3, 3), padding='SAME', dtype=self.dtype, name="out_conv")(h))

# =================================================================================================
# 3. PROJECT CHIMERA: The Quaternion Dynamics Model
# =================================================================================================
class ChimeraAE_Quaternion(nn.Module):
    d_model: int; dtype: Any = DTYPE
    def setup(self):
        self.image_encoder = ImageEncoder(d_model=self.d_model, name="image_encoder", dtype=self.dtype)
        self.text_encoder = TextEncoder(d_model=self.d_model, name="text_encoder", dtype=self.dtype)
        self.w_gen = QuaternionComponentGenerator(d_model=self.d_model, name="w_gen", dtype=self.dtype)
        self.x_gen = QuaternionComponentGenerator(d_model=self.d_model, name="x_gen", dtype=self.dtype)
        self.y_gen = QuaternionComponentGenerator(d_model=self.d_model, name="y_gen", dtype=self.dtype)
        self.z_gen = QuaternionComponentGenerator(d_model=self.d_model, name="z_gen", dtype=self.dtype)
        self.topological_decoder = TopologicalGenerativeDecoder(d_model=self.d_model, name="topological_decoder", dtype=self.dtype)
        self.image_decoder = ImagePatchDecoder(self.d_model, name="image_decoder", dtype=self.dtype)

    def apply_quaternion_rotation(self, q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        v_3d = v[:, :3] 
        v_quat_3d = jnp.concatenate([jnp.zeros((v_3d.shape[0], 1)), v_3d], axis=-1)
        q_conj = q * jnp.array([1., -1., -1., -1.])
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
            w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2; x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2; z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return jnp.stack([w, x, y, z], axis=-1)
        rotated_v_quat = quat_mul(quat_mul(q, v_quat_3d), q_conj)
        rotated_v_3d = rotated_v_quat[..., 1:]
        return v.at[:, :3].set(rotated_v_3d)
    
    def generate_quaternion(self, z_initial):
        qw = self.w_gen(z_initial); qx = self.x_gen(z_initial)
        qy = self.y_gen(z_initial); qz = self.z_gen(z_initial)
        q = jnp.stack([qw, qx, qy, qz], axis=-1)
        return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

    def decode_from_z(self, z):
        generated_patches = self.topological_decoder(z)
        return self.image_decoder(generated_patches)

    def __call__(self, batch):
        images, gt_text_embs = batch['images'], batch['clip_text_embeddings']
        z_initial = self.image_encoder(images)
        z_text = self.text_encoder(gt_text_embs)
        q_normalized = self.generate_quaternion(z_initial)
        z_rotated = self.apply_quaternion_rotation(q_normalized, z_initial)
        recon_images = self.decode_from_z(z_rotated)
        loss_align = jnp.mean((z_rotated - z_text)**2)
        loss_pixel = jnp.mean(jnp.abs(images - recon_images))
        total_loss = loss_align + loss_pixel
        aux = {'loss': total_loss, 'pixel_loss': loss_pixel, 'align_loss': loss_align}
        return total_loss, aux

# =================================================================================================
# 4. DATA HANDLING (Unchanged)
# =================================================================================================
def prepare_data(image_dir: str):
    base_path = Path(image_dir); record_file = base_path/"data_64x64.tfrecord"
    text_emb_file = base_path/"clip_text_embeddings.pkl"; info_file = base_path/"dataset_info.pkl"
    if record_file.exists() and text_emb_file.exists():
        print(f"✅ All necessary data files exist in {image_dir}. Skipping."); return
    print(f"--- Preparing data from {image_dir} ---")
    image_paths = sorted([p for p in base_path.rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')])
    text_paths = [p.with_suffix('.txt') for p in image_paths]
    valid_pairs = [(img, txt) for img, txt in zip(image_paths, text_paths) if txt.exists()]
    if not valid_pairs: print(f"[FATAL] No matching image/text pairs found in {image_dir}."), sys.exit(1)
    image_paths, text_paths = zip(*valid_pairs); print(f"Found {len(image_paths)} matching image-text pairs. Processing...")
    clip_model, _ = clip.load("ViT-B/32", device=_clip_device); all_clip_text_embs = []
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for i in tqdm(range(0, len(image_paths), 256), desc="Processing Batches"):
            for img_path in image_paths[i:i+256]:
                img = Image.open(img_path).convert("RGB").resize((64,64),Image.Resampling.BICUBIC)
                img_bytes = tf.io.encode_jpeg(np.array(img),quality=95).numpy()
                ex = tf.train.Example(features=tf.train.Features(feature={'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
                writer.write(ex.SerializeToString())
            captions = [p.read_text().strip() for p in text_paths[i:i+256]]
            text_tokens = clip.tokenize(captions, truncate=True).to(_clip_device)
            with torch.no_grad(): all_clip_text_embs.append(clip_model.encode_text(text_tokens).cpu().numpy())
    with open(text_emb_file,'wb') as f: pickle.dump(np.concatenate(all_clip_text_embs).astype(np.float32), f)
    with open(info_file,'wb') as f: pickle.dump({'num_samples':len(image_paths)}, f)
    print(f"✅ Data preparation complete.")

def create_dataset(image_dir: str, batch_size: int):
    record_file = Path(image_dir)/"data_64x64.tfrecord"; text_emb_file = Path(image_dir)/"clip_text_embeddings.pkl"
    if not all([f.exists() for f in [record_file, text_emb_file]]):
        raise FileNotFoundError("Dataset files not found. Run 'prepare-data' first.")
    with open(text_emb_file,'rb') as f: clip_text_embs = pickle.load(f)
    def _parse(proto):
        f={'image':tf.io.FixedLenFeature([],tf.string)}; p=tf.io.parse_single_example(proto,f)
        img=(tf.cast(tf.io.decode_jpeg(p['image'],3),tf.float32)/127.5)-1.0; img.set_shape([64,64,3]); return img
    img_ds = tf.data.TFRecordDataset(str(record_file)).map(_parse,num_parallel_calls=tf.data.AUTOTUNE)
    text_emb_ds = tf.data.Dataset.from_tensor_slices(clip_text_embs)
    full_ds = tf.data.Dataset.zip((img_ds, text_emb_ds))
    return full_ds.shuffle(1024).repeat().batch(batch_size,drop_remainder=True).map(
        lambda i, c_t: {'images': i, 'clip_text_embeddings': c_t}
    ).prefetch(tf.data.AUTOTUNE)

# =================================================================================================
# 5. THE MARKETPLACE TRAINER for QUATERNION DYNAMICS
# =================================================================================================
class MarketplaceTrainer_Quaternion:
    def __init__(self, args):
        self.args = args; self.num_devices = jax.local_device_count()
        self.should_shutdown=False; signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.vendor_count = args.vendor_count 
        self.market_depth = 4
        self.model = ChimeraAE_Quaternion(d_model=args.d_model)
        self.spec_names = ['w_gen', 'x_gen', 'y_gen', 'z_gen']
        self.avg_loss_history=deque(maxlen=200); self.min_loss_history=deque(maxlen=200)

    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except: return "N/A", "N/A"
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array([val for val in data if np.isfinite(val)])
        if len(hist)<2: return " "*w; hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v: return "".join([s[0]]*len(hist))
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1); return "".join(s[i] for i in indices)

    def train(self):
        ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d_marketplace_quat.pkl"); key = jax.random.PRNGKey(self.args.seed)
        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path, 'rb') as f: data = pickle.load(f)
            state = freeze({'base_params': data['base_params']}); start_epoch = data.get('epoch', -1) + 1
        else:
            print("--- Initializing new Quaternion model for Marketplace training ---")
            key, init_key = jax.random.split(key)
            dummy_batch = {'images': jnp.zeros((1, 64, 64, 3)), 'clip_text_embeddings': jnp.zeros((1, 512))}
            params = self.model.init(init_key, dummy_batch)['params']; state = freeze({'base_params': params}); start_epoch = 0
        p_state = replicate(state)
        
        @partial(jax.pmap, axis_name='devices')
        def train_step(state, key, batch):
            base_params = state['base_params']; lr = self.args.lr
            def generate_deltas(p): return jax.random.normal(key, (self.vendor_count,) + p.shape, dtype=p.dtype)
            
            spec_deltas = {name: jax.tree_util.tree_map(generate_deltas, base_params[name]) for name in self.spec_names}
            path_indices = jnp.stack(jnp.meshgrid(*[jnp.arange(self.vendor_count)]*self.market_depth),-1).reshape(-1, self.market_depth)
            
            def model_forward_on_path(path_idx_vector):
                mutable_params = unfreeze(base_params)
                for i, name in enumerate(self.spec_names):
                    vendor_idx = path_idx_vector[i]
                    delta_for_vendor = jax.tree_util.tree_map(lambda d: d[vendor_idx], spec_deltas[name])
                    perturbed_params = jax.tree_util.tree_map(lambda b, d: b + d, base_params[name], delta_for_vendor)
                    mutable_params[name] = perturbed_params
                loss, _ = self.model.apply({'params': mutable_params}, batch)
                return loss

            all_losses = jax.vmap(model_forward_on_path)(path_indices)
            
            mean_loss, std_loss = jnp.mean(all_losses), jnp.std(all_losses) + 1e-8
            attributions = -(all_losses - mean_loss) / std_loss
            
            reconciled_deltas = {}
            for i, name in enumerate(self.spec_names):
                def reconcile_leaf(delta_leaf):
                    vendor_indices = path_indices[:, i]
                    deltas_for_paths = delta_leaf[vendor_indices]
                    return jnp.einsum('n,n...->...', attributions, deltas_for_paths) / path_indices.shape[0]
                reconciled_deltas[name] = jax.tree_util.tree_map(reconcile_leaf, spec_deltas[name])
            
            updated_spec_params = {}
            for name in self.spec_names:
                grad_approx = reconciled_deltas[name]
                updated_spec_params[name] = jax.tree_util.tree_map(lambda p, g: p + g * lr, base_params[name], grad_approx)
            
            new_base_params = unfreeze(base_params)
            new_base_params.update(updated_spec_params)
            
            metrics = jax.lax.pmean({'avg_loss': mean_loss, 'min_loss': jnp.min(all_losses), 'std_loss': std_loss}, 'devices')
            
            # --- THE FIX IS HERE ---
            # Construct a new state dict explicitly instead of using .copy()
            new_state = {'base_params': freeze(new_base_params)}
            return new_state, metrics
            # --- END OF FIX ---

        dataset=create_dataset(self.args.image_dir,self.args.batch_size*self.num_devices); it=dataset.as_numpy_iterator()
        with open(Path(self.args.image_dir)/"dataset_info.pkl",'rb') as f: steps_per_epoch=(pickle.load(f)['num_samples'])//(self.args.batch_size*self.num_devices)
        
        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Project Chimera (Quaternion Dynamics)[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2))
        progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn())
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs); layout['footer'].update(progress)
        
        with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            for epoch in range(start_epoch, self.args.epochs):
                progress.update(epoch_task, completed=0, epoch=epoch+1)
                for step in range(steps_per_epoch):
                    if self.should_shutdown: break
                    key, *train_keys = jax.random.split(key, self.num_devices + 1)
                    sharded_keys = jax.device_put_sharded(train_keys, jax.local_devices())
                    p_state, metrics = train_step(p_state, sharded_keys, common_utils.shard(next(it)))
                    if step%5==0:
                        m=unreplicate(metrics); self.avg_loss_history.append(m['avg_loss'].item()); self.min_loss_history.append(m['min_loss'].item())
                        stats_tbl=Table(show_header=False,box=None,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
                        stats_tbl.add_row("Avg Loss",f"[bold green]{m['avg_loss'].item():.4e}[/]"); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util",f"[yellow]{util}[/]");
                        layout["left"].update(Panel(stats_tbl,title="[bold]📊 Stats[/]",border_style="blue"))
                        spark_w = max(1, (live.console.width * 2 // 3) - 25)
                        loss_tbl=Table(show_header=False,box=None,padding=(0,1)); loss_tbl.add_column(style="dim",width=15); loss_tbl.add_column(width=10, justify="right"); loss_tbl.add_column(ratio=1)
                        loss_tbl.add_row("Avg Loss", f"{m['avg_loss'].item():.4f}", f"[yellow]{self._get_sparkline(self.avg_loss_history, spark_w)}")
                        loss_tbl.add_row("Min Loss", f"[magenta]{m['min_loss'].item():.4f}[/magenta]", f"[magenta]{self._get_sparkline(self.min_loss_history, spark_w)}")
                        layout["right"].update(Panel(loss_tbl, title="[bold]📉 Losses[/]", border_style="blue"))
                    progress.update(epoch_task,advance=1)
                if self.should_shutdown: break
                state_to_save=unreplicate(p_state)
                with open(ckpt_path,'wb') as f: pickle.dump({'base_params':jax.device_get(state_to_save['base_params']),'epoch':epoch},f)
                live.console.print(f"--- :floppy_disk: Quaternion Marketplace checkpoint saved for epoch {epoch+1} ---")

# =================================================================================================
# 6. GENERATOR CLASS for DYNAMIC TRANSFORMATION
# =================================================================================================
class Generator:
    def __init__(self, args):
        self.args = args; self.d_model = args.d_model; self.model = ChimeraAE_Quaternion(d_model=self.d_model)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d_marketplace_quat.pkl")
        if not model_path.exists(): print(f"[FATAL] Model file not found at {model_path}. Please train first."), sys.exit(1)
        print(f"--- Loading model from {model_path} ---");
        with open(model_path, 'rb') as f: self.params = pickle.load(f)['base_params']
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        self.key = jax.random.PRNGKey(args.seed if 'seed' in args and not callable(args.seed) else int(time.time()))
    @partial(jax.jit, static_argnames=('self',))
    def _get_z_from_image(self, img_array): return self.model.apply({'params': self.params}, img_array, method=self.model.image_encoder)
    @partial(jax.jit, static_argnames=('self',))
    def _get_quaternion(self, z): return self.model.apply({'params': self.params}, z, method=self.model.generate_quaternion)
    @partial(jax.jit, static_argnames=('self',))
    def _rotate_z(self, q, z): return self.model.apply({'params': self.params}, q, z, method=self.model.apply_quaternion_rotation)
    @partial(jax.jit, static_argnames=('self',))
    def _decode_z(self, z): return self.model.apply({'params': self.params}, z, method=self.model.decode_from_z)
    def transform(self):
        print(f"--- Generating transformation from '{self.args.base_image}' towards '{self.args.target_prompt}' over {self.args.steps} steps ---")
        img = Image.open(self.args.base_image).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
        img_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
        current_z = self._get_z_from_image(jnp.expand_dims(img_np, axis=0))
        q_operator = self._get_quaternion(current_z)
        frames = []
        for step in tqdm(range(self.args.steps), desc="Animating frames"):
            img_tensor = self._decode_z(current_z)
            img_out_np = np.array((jax.device_get(img_tensor[0]) * 0.5 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
            frames.append(Image.fromarray(img_out_np).resize((256, 256), Image.Resampling.NEAREST))
            current_z = self._rotate_z(q_operator, current_z)
        base_slug = Path(self.args.base_image).stem
        target_slug = "".join(c if c.isalnum() else "_" for c in self.args.target_prompt).strip("_")[:25]
        save_path = f"transform_{base_slug}_to_{target_slug}_{self.args.seed}.gif"
        frames[0].save(save_path, save_all=True, append_images=frames[1:], optimize=False, duration=150, loop=0)
        print(f"✅ Animation saved to {save_path}")

# =================================================================================================
# 7. MAIN EXECUTION BLOCK
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Project Chimera: A Quaternion Dynamics Model trained with Marketplace V2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_prep = subparsers.add_parser("prepare-data", help="Prepare images and CLIP text embeddings.")
    p_prep.add_argument('--image-dir', type=str, required=True)
    p_train = subparsers.add_parser("train", help="Train the Quaternion Dynamics model using Marketplace.")
    p_train.add_argument('--image-dir', type=str, required=True); p_train.add_argument('--basename', type=str, required=True); p_train.add_argument('--d-model', type=int, default=256)
    p_train.add_argument('--epochs', type=int, default=100); p_train.add_argument('--batch-size', type=int, default=16, help="Per device. Keep this low due to high computation per step."); p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--vendor-count', type=int, default=8, help="Number of variations per component. 8^4=4096 paths.")
    p_train.add_argument('--seed', type=int, default=42)
    p_trans = subparsers.add_parser("transform", help="Generate an animation by applying a learned transformation.")
    p_trans.add_argument('--base-image', type=str, required=True, help="Path to the starting image."); p_trans.add_argument('--target-prompt', type=str, required=True, help="Text prompt to transform towards.")
    p_trans.add_argument('--steps', type=int, default=30, help="Number of frames in the animation.")
    p_trans.add_argument('--basename', type=str, required=True); p_trans.add_argument('--d-model', type=int, default=256)
    p_trans.add_argument('--seed', type=int, default=lambda: int(time.time()), help="Random seed.")
    args = parser.parse_args()
    if 'seed' in args and hasattr(args, 'seed') and callable(args.seed): args.seed = args.seed()

    if args.command == "prepare-data": prepare_data(args.image_dir)
    elif args.command == "train": MarketplaceTrainer_Quaternion(args).train()
    elif args.command == "transform": Generator(args).transform()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"), sys.exit(0)