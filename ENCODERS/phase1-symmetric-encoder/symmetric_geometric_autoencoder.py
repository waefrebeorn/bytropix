# =================================================================================================
#            SYMMETRIC GEOMETRIC AUTOENCODER: The True Wubu Manifold
#
# This script embodies the architectural pivot from a diffusion model to a true,
# symmetric autoencoder. It builds upon the insight that the WuBu manifold is a
# powerful encoder. To create a true autoencoder, we introduce a SYMMETRIC
# WuBuNestingDecoder that mirrors the encoder's architecture, including U-Net style
# geometric skip connections. This allows the model to unfold the latent space with
# the same geometric care it used to fold it.
#
# This is a pure autoencoder, designed to learn the fundamental manifold of image data.
#
# HOW TO USE:
#
# 1. Prepare Data:
#    python symmetric_geometric_autoencoder.py prepare-data --image-dir ./your_images/
#
# 2. Train the Symmetric Autoencoder:
#    python symmetric_geometric_autoencoder.py train --image-dir ./your_images/ --basename my_symmetric_ae --d-model 256
#
# 3. Visualize Reconstructions:
#    python symmetric_geometric_autoencoder.py visualize --image-dir ./your_images/ --basename my_symmetric_ae --d-model 256
#
# =================================================================================================

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

# --- JAX Configuration ---
CPU_DEVICE = jax.devices("cpu")[0]
jax.config.update("jax_debug_nans", False); jax.config.update('jax_disable_jit', False); jax.config.update('jax_default_matmul_precision', 'bfloat16'); jax.config.update('jax_threefry_partitionable', True)

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
# 1. MATHEMATICAL FOUNDATIONS: The Poincaré Ball
# =================================================================================================
class PoincareBall:
    EPS = 1e-7
    @staticmethod
    def _ensure_float32(*args): return [jnp.asarray(arg, dtype=jnp.float32) for arg in args]
    @staticmethod
    def project(x, c=1.0):
        x_f32, c_f32 = PoincareBall._ensure_float32(x, c); norm_sq = jnp.sum(x_f32*x_f32, -1, keepdims=True)
        max_norm_sq = (1./jnp.sqrt(c_f32).clip(PoincareBall.EPS))**2-PoincareBall.EPS; cond=norm_sq>max_norm_sq
        projected = x_f32/(jnp.sqrt(norm_sq*c_f32).clip(PoincareBall.EPS)*(1.+PoincareBall.EPS))
        return jnp.where(cond, projected, x_f32).astype(x.dtype)
    @staticmethod
    def expmap0(v, c):
        v_f32, c_f32 = PoincareBall._ensure_float32(v, c); sqrt_c=jnp.sqrt(c_f32).clip(PoincareBall.EPS); v_norm=jnp.linalg.norm(v_f32,-1,keepdims=True).clip(PoincareBall.EPS)
        direction=v_f32/v_norm; magnitude=jnp.tanh(sqrt_c*v_norm)/sqrt_c; result=magnitude*direction
        return jnp.where(v_norm < PoincareBall.EPS, jnp.zeros_like(v_f32), result).astype(v.dtype)
    @staticmethod
    def mobius_add(x, y, c):
        x_f32, y_f32, c_f32 = PoincareBall._ensure_float32(x,y,c); x2=jnp.sum(x_f32*x_f32,-1,keepdims=True); y2=jnp.sum(y_f32*y_f32,-1,keepdims=True); xy=jnp.sum(x_f32*y_f32,-1,keepdims=True)
        num=(1+2*c_f32*xy+c_f32*y2)*x_f32+(1-c_f32*x2)*y_f32; den=1+2*c_f32*xy+c_f32**2*x2*y2
        return PoincareBall.project(num/den.clip(PoincareBall.EPS), c_f32).astype(x.dtype)

# =================================================================================================
# 2. MODEL ARCHITECTURE: The Symmetric Autoencoder
# =================================================================================================
def safe_orthogonal_initializer(dtype=jnp.bfloat16):
    def init(key, shape, dtype=dtype):
        q = nn.initializers.orthogonal(dtype=jnp.float32)(key, shape); return q.astype(dtype)
    return init

class ImagePatchEncoder(nn.Module):
    embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x):
        conv = nn.Conv(self.embed_dim, kernel_size=(4, 4), strides=(4, 4), dtype=self.dtype, name="patch_conv")
        patches = conv(x.astype(self.dtype))
        patches_norm = nn.LayerNorm(dtype=jnp.float32, name="patch_ln")(patches)
        B, H, W, C = patches_norm.shape
        return patches_norm.reshape(B, H * W, C)

class WuBuLevel(nn.Module):
    dim: int; num_boundary_points: int; mlp_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, v_in, relative_vectors, ld_in, spread_in):
        v_in, rel, ld_in, spread_in = [t.astype(self.dtype) for t in (v_in, relative_vectors, ld_in, spread_in)]
        boundaries = self.param('boundaries', nn.initializers.normal(0.01), (self.num_boundary_points, self.dim), self.dtype)
        descriptor = self.param('descriptor', nn.initializers.normal(0.01), (self.dim,), self.dtype)
        spread = nn.softplus(self.param('spread', nn.initializers.zeros, (), jnp.float32))
        context = jnp.concatenate([v_in, jnp.mean(rel, axis=1), ld_in, spread_in], axis=-1)
        flow_mlp = nn.Dense(self.mlp_dim, dtype=self.dtype, name="flow_dense_1")(context)
        flow_mlp = nn.gelu(flow_mlp.astype(jnp.float32)).astype(self.dtype)
        flow_vector = nn.Dense(self.dim, dtype=self.dtype, kernel_init=nn.initializers.zeros, name="flow_dense_2")(flow_mlp)
        return v_in + flow_vector, boundaries, descriptor, spread

class WuBuInterLevelTransition(nn.Module):
    source_dim: int; target_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, v_in, boundary_vectors, ld_in):
        v_in, b_vecs, ld_in = [t.astype(self.dtype) for t in (v_in, boundary_vectors, ld_in)]
        rotation = self.param('rotation', safe_orthogonal_initializer(self.dtype), (self.source_dim, self.source_dim))
        v_rotated, b_rotated, ld_rotated = v_in @ rotation, b_vecs @ rotation, ld_in @ rotation
        def mapping_mlp(x, name):
            h = nn.Dense(self.source_dim * 2, name=f'{name}_dense1', dtype=self.dtype)(x)
            h = nn.gelu(h.astype(jnp.float32)).astype(self.dtype)
            return nn.Dense(self.target_dim, name=f'{name}_dense2', dtype=self.dtype)(h)
        v_out = mapping_mlp(v_rotated, 'v_map')
        b_out = jax.vmap(mapping_mlp, in_axes=(0, None), out_axes=0)(b_rotated, 'b_map')
        relative_vectors = v_out[:, None, :] - b_out
        return v_out, mapping_mlp(ld_rotated, 'ld_map'), relative_vectors

class WuBuNestingEncoder(nn.Module):
    d_model: int; num_levels: int; level_dims: Sequence[int]; num_boundary_points: Sequence[int]; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, patches):
        B, N, D = patches.shape
        v_patches = nn.Dense(self.level_dims[0], name="in_proj", dtype=self.dtype)(patches.astype(self.dtype))
        v_global = jnp.mean(v_patches, axis=1)
        relative_vectors = jnp.zeros((B, self.num_boundary_points[0], self.level_dims[0]), self.dtype)
        ld_in, spread_in = jnp.zeros((B, self.level_dims[0]), self.dtype), jnp.zeros((B, 1), self.dtype)
        encoder_states = []
        for i in range(self.num_levels):
            level = WuBuLevel(self.level_dims[i], self.num_boundary_points[i], self.d_model * 2, name=f"level_{i}", dtype=self.dtype)
            v_global, boundaries, ld, spread = level(v_global, relative_vectors, ld_in, spread_in)
            encoder_states.append({'v_global': v_global, 'spread': spread})
            if i < self.num_levels - 1:
                transition = WuBuInterLevelTransition(self.level_dims[i], self.level_dims[i+1], name=f"transition_{i}", dtype=self.dtype)
                v_global, ld_in, relative_vectors = transition(v_global, boundaries, jnp.tile(ld[None, :], (B, 1)))
                spread_in = jnp.full((B, 1), spread, jnp.float32)
        return encoder_states

class WuBuNestingDecoder(nn.Module):
    d_model: int; num_levels: int; level_dims: Sequence[int]; num_boundary_points: Sequence[int]; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, encoder_states, initial_patches):
        B = initial_patches.shape[0]
        v_global = encoder_states[-1]['v_global']
        
        rev_level_dims = self.level_dims[::-1]
        rev_num_boundary_points = self.num_boundary_points[::-1]
        
        relative_vectors = jnp.zeros((B, rev_num_boundary_points[0], rev_level_dims[0]), self.dtype)
        ld_in, spread_in = jnp.zeros((B, rev_level_dims[0]), self.dtype), jnp.zeros((B, 1), self.dtype)
        
        for i in range(self.num_levels):
            v_global += encoder_states[self.num_levels - 1 - i]['v_global']

            level = WuBuLevel(rev_level_dims[i], rev_num_boundary_points[i], self.d_model * 2, name=f"level_{i}", dtype=self.dtype)
            v_global, boundaries, ld, spread = level(v_global, relative_vectors, ld_in, spread_in)
            
            if i < self.num_levels - 1:
                transition = WuBuInterLevelTransition(rev_level_dims[i], rev_level_dims[i+1], name=f"transition_{i}", dtype=self.dtype)
                v_global, ld_in, relative_vectors = transition(v_global, boundaries, jnp.tile(ld[None, :], (B, 1)))
                spread_in = jnp.full((B, 1), spread, jnp.float32)
        
        final_v_global_proj = nn.Dense(self.d_model, name="final_decoder_proj", dtype=self.dtype)(v_global)
        return nn.Dense(self.d_model, name="out_proj", dtype=self.dtype)(initial_patches) + final_v_global_proj[:, None, :]

class ImagePatchDecoder(nn.Module):
    embed_dim: int; dtype: Any = jnp.bfloat16
    @nn.compact
    def __call__(self, x):
        B, L, D = x.shape; H = W = int(np.sqrt(L)); x_reshaped = x.reshape(B, H, W, D); ch = self.embed_dim
        def ResBlock(inner_x, out_ch, name):
            h_in = nn.gelu(nn.LayerNorm(dtype=jnp.float32, name=f"{name}_ln1")(inner_x))
            h = nn.Conv(out_ch, (3, 3), padding='SAME', dtype=self.dtype, name=f"{name}_conv1")(h_in)
            h = nn.gelu(nn.LayerNorm(dtype=jnp.float32, name=f"{name}_ln2")(h))
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
        h = nn.gelu(nn.LayerNorm(dtype=jnp.float32, name="out_ln")(h))
        return nn.tanh(nn.Conv(3, (3, 3), padding='SAME', dtype=jnp.float32, name="out_conv")(h))

class SymmetricGeometricAutoencoder(nn.Module):
    d_model: int; dtype: Any = jnp.bfloat16
    def setup(self):
        wubu_config = {'d_model': self.d_model, 'num_levels': 4, 'level_dims': [512, 256, 128, self.d_model], 'num_boundary_points': [16, 12, 8, 4], 'dtype': self.dtype}
        self.patch_encoder = ImagePatchEncoder(self.d_model, name="patch_encoder")
        self.wubu_encoder = WuBuNestingEncoder(**wubu_config, name="wubu_encoder")
        self.wubu_decoder = WuBuNestingDecoder(**wubu_config, name="wubu_decoder")
        self.image_decoder = ImagePatchDecoder(self.d_model, name="image_decoder")

    def __call__(self, images):
        initial_patches = self.patch_encoder(images)
        encoder_states = self.wubu_encoder(initial_patches)
        reconstructed_patches = self.wubu_decoder(encoder_states, initial_patches)
        reconstructed_images = self.image_decoder(reconstructed_patches)
        
        loss_recon = jnp.mean(jnp.abs(images - reconstructed_images))
        
        # --- THE DEFINITIVE FIX: Explicit, Unambiguous Summation ---
        # This removes all Python list comprehensions from the loss path,
        # guaranteeing that the JAX tracer can compile it robustly.
        loss_reg = jnp.asarray(0.0, dtype=jnp.float32)
        loss_reg += jnp.mean(jnp.sum(encoder_states[0]['v_global']**2, axis=-1))
        loss_reg += jnp.mean(jnp.sum(encoder_states[1]['v_global']**2, axis=-1))
        loss_reg += jnp.mean(jnp.sum(encoder_states[2]['v_global']**2, axis=-1))
        loss_reg += jnp.mean(jnp.sum(encoder_states[3]['v_global']**2, axis=-1))
        loss_reg *= 0.001
        
        total_loss = loss_recon + loss_reg
        aux = {'loss': total_loss, 'recon_loss': loss_recon, 'reg_loss': loss_reg, 'reconstructions': reconstructed_images}
        return total_loss, aux

# =================================================================================================
# 3. DATA HANDLING & 4. TRAINING/VISUALIZATION LOGIC
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
        self.model = SymmetricGeometricAutoencoder(d_model=args.d_model)
        self.recon_loss_history = deque(maxlen=200); self.reg_loss_history = deque(maxlen=200)

    def _get_gpu_stats(self):
        try:
            h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h)
            return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except: return "N/A", "N/A"
        
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array(list(data));
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v: return "".join([s[0]]*len(hist))
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1)
        return "".join(s[i] for i in indices)

    def train(self):
        ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d.pkl"); optimizer=optax.adamw(self.args.lr)
        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path,'rb') as f: data=pickle.load(f)
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=data['params'],tx=optimizer).replace(opt_state=data['opt_state'])
            start_epoch = data.get('epoch',0)+1
        else:
            print("--- Initializing new model ---")
            with jax.default_device(CPU_DEVICE):
                params=self.model.init(jax.random.PRNGKey(0),jnp.zeros((1,64,64,3)))['params']
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=params,tx=optimizer); start_epoch=0
        
        p_state = replicate(state)
        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,))
        def train_step(state, batch):
            (loss,aux),grads=jax.value_and_grad(state.apply_fn,has_aux=True)({'params':state.params},batch)
            return state.apply_gradients(grads=jax.lax.pmean(grads['params'],'devices')), jax.lax.pmean(aux,'devices')
            
        dataset = create_dataset(self.args.image_dir, self.args.batch_size*self.num_devices); it=dataset.as_numpy_iterator()
        with open(Path(self.args.image_dir)/"dataset_info.pkl",'rb') as f: steps_per_epoch=(pickle.load(f)['num_samples'])//(self.args.batch_size*self.num_devices)
        
        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Symmetric Geometric AE[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2))
        progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn())
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs); layout['footer'].update(progress)
        
        epoch_loop = start_epoch
        with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            for epoch in range(start_epoch, self.args.epochs):
                progress.update(epoch_task, completed=0, epoch=epoch+1); epoch_loop=epoch
                for step in range(steps_per_epoch):
                    if self.should_shutdown: break
                    p_state,metrics = train_step(p_state,common_utils.shard(next(it)))
                    if step%10==0:
                        m=unreplicate(metrics); self.recon_loss_history.append(m['recon_loss']); self.reg_loss_history.append(m['reg_loss'])
                        stats_tbl=Table(show_header=False,box=None,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
                        stats_tbl.add_row("Total Loss",f"[bold green]{m['loss']:.4e}[/]"); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util",f"[yellow]{util}[/]")
                        layout["left"].update(Panel(stats_tbl,title="[bold]📊 Stats[/]",border_style="blue"))
                        loss_tbl=Table(show_header=False,box=None,padding=(0,1)); loss_tbl.add_column(style="dim",width=15); loss_tbl.add_column(justify="right")
                        loss_tbl.add_row("Reconstruction",f"{m['recon_loss']:.4f}"); loss_tbl.add_row("Regularization",f"{m['reg_loss']:.4e}")
                        spark_w = max(1, (live.console.width*2//3)-10)
                        recon_panel=Panel(Align.center(f"[cyan]{self._get_sparkline(self.recon_loss_history,spark_w)}[/]"),title="Reconstruction Loss",height=3, border_style="cyan")
                        reg_panel=Panel(Align.center(f"[magenta]{self._get_sparkline(self.reg_loss_history,spark_w)}[/]"),title="Regularization Loss",height=3, border_style="magenta")
                        layout["right"].update(Panel(Group(loss_tbl,recon_panel,reg_panel),title="[bold]📉 Losses[/]",border_style="magenta"))
                    progress.update(epoch_task,advance=1)
                if self.should_shutdown: break
                state_to_save=unreplicate(p_state)
                with open(ckpt_path,'wb') as f: pickle.dump({'params':jax.device_get(state_to_save.params),'opt_state':jax.device_get(state_to_save.opt_state),'epoch':epoch},f)
                live.console.print(f"--- :floppy_disk: Checkpoint saved for epoch {epoch+1} ---")
        if self.should_shutdown:
            print("\n--- Shutdown detected. Saving final state... ---")
            state_to_save=unreplicate(p_state)
            with open(ckpt_path,'wb') as f: pickle.dump({'params':jax.device_get(state_to_save.params),'opt_state':jax.device_get(state_to_save.opt_state),'epoch':epoch_loop},f)

class Visualizer:
    def __init__(self, args):
        self.args = args
        self.model = SymmetricGeometricAutoencoder(d_model=args.d_model)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d.pkl")
        if not model_path.exists(): print(f"[FATAL] Model file not found at {model_path}."), sys.exit(1)
        print(f"--- Loading model from {model_path} ---")
        with open(model_path, 'rb') as f: self.params = pickle.load(f)['params']

    @partial(jax.jit, static_argnames=('self',))
    def _reconstruct(self, images):
        _, aux = self.model.apply({'params': self.params}, images)
        return aux['reconstructions']

    def visualize(self):
        dataset = create_dataset(self.args.image_dir, 8); it = dataset.as_numpy_iterator()
        originals = next(it)
        reconstructions = self._reconstruct(originals)
        
        originals_pil = [Image.fromarray(np.array((img*0.5+0.5).clip(0,1)*255,dtype=np.uint8)) for img in originals]
        recons_pil = [Image.fromarray(np.array((img*0.5+0.5).clip(0,1)*255,dtype=np.uint8)) for img in reconstructions]
        
        num_images = len(originals_pil)
        canvas = Image.new('RGB', (64*2 + 10, 64*num_images))
        for i in range(num_images):
            canvas.paste(originals_pil[i], (0, i*64))
            canvas.paste(recons_pil[i], (64 + 10, i*64))
        
        save_path = f"{self.args.basename}_reconstruction.png"
        canvas.save(save_path)
        print(f"✅ Visualization saved to {save_path}")
        print("Left column: Original images | Right column: Reconstructed images")

class Reconstructor:
    def __init__(self, args):
        self.args = args
        self.model = SymmetricGeometricAutoencoder(d_model=args.d_model)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d.pkl")
        if not model_path.exists():
            print(f"[FATAL] Model file not found at {model_path}. Please train a model first."), sys.exit(1)
        print(f"--- Loading model from {model_path} ---")
        with open(model_path, 'rb') as f:
            self.params = pickle.load(f)['params']

    @partial(jax.jit, static_argnames=('self',))
    def _reconstruct(self, image):
        _, aux = self.model.apply({'params': self.params}, image)
        return aux['reconstructions']

    def reconstruct_single_image(self):
        image_path = Path(self.args.image_path)
        if not image_path.exists():
            print(f"[FATAL] Image file not found at {image_path}"), sys.exit(1)

        # --- CRITICAL: Preprocess the image EXACTLY like the training data ---
        # 1. Load and resize
        original_img = Image.open(image_path).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
        
        # 2. Convert to numpy and normalize to [-1, 1]
        img_np = (np.array(original_img, dtype=np.float32) / 127.5) - 1.0
        
        # 3. Add a batch dimension and convert to JAX array
        image_batch = jnp.expand_dims(img_np, axis=0)

        # --- Run the model ---
        reconstruction_batch = self._reconstruct(image_batch)
        
        # --- Post-process the output ---
        # 1. Remove batch dimension and get numpy array
        recon_np = np.array(reconstruction_batch[0])
        
        # 2. De-normalize from [-1, 1] to [0, 255]
        recon_np = ((recon_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        recon_img = Image.fromarray(recon_np)

        # --- Create and save the comparison image ---
        canvas = Image.new('RGB', (64 * 2 + 10, 64))
        canvas.paste(original_img, (0, 0))
        canvas.paste(recon_img, (64 + 10, 0))
        
        save_path = f"{image_path.stem}_reconstruction.png"
        canvas.save(save_path)
        print(f"✅ Reconstruction saved to {save_path}")
        print("Left: Your original image | Right: Model's reconstruction")
# =================================================================================================
# 5. MAIN EXECUTION BLOCK
# =================================================================================================
# Find the main() function and add the new parser
def main():
    parser = argparse.ArgumentParser(description="Symmetric Geometric Autoencoder")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # (p_prep and p_train parsers are here, no changes needed)
    p_prep = subparsers.add_parser("prepare-data", help="Convert images to TFRecords.")
    p_prep.add_argument('--image-dir', type=str, required=True)
    
    p_train = subparsers.add_parser("train", help="Train the autoencoder.")
    p_train.add_argument('--image-dir', type=str, required=True)
    p_train.add_argument('--basename', type=str, required=True)
    p_train.add_argument('--d-model', type=int, default=256)
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch-size', type=int, default=32, help="Batch size PER DEVICE.")
    p_train.add_argument('--lr', type=float, default=1e-4)
    p_train.add_argument('--seed', type=int, default=42)
    
    # (p_vis parser is here, no changes needed)
    p_vis = subparsers.add_parser("visualize", help="Visualize model reconstructions from the dataset.")
    p_vis.add_argument('--image-dir', type=str, required=True)
    p_vis.add_argument('--basename', type=str, required=True)
    p_vis.add_argument('--d-model', type=int, default=256)

    # --- ADD THIS NEW PARSER ---
    p_recon = subparsers.add_parser("reconstruct", help="Reconstruct a single image file.")
    p_recon.add_argument('--image-path', type=str, required=True, help="Path to the image you want to reconstruct.")
    p_recon.add_argument('--basename', type=str, required=True, help="Basename of the trained model file.")
    p_recon.add_argument('--d-model', type=int, default=256, help="The d_model of the trained model.")

    args = parser.parse_args()
    if args.command == "prepare-data": prepare_data(args.image_dir)
    elif args.command == "train": Trainer(args).train()
    elif args.command == "visualize": Visualizer(args).visualize()
    # --- ADD THIS NEW COMMAND HANDLER ---
    elif args.command == "reconstruct": Reconstructor(args).reconstruct_single_image()


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"), sys.exit(0)