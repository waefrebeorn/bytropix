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
        
        # Implementation of Equation (2) from the Li et al. (2025) paper
        real_part = jnp.cos(delta_f32 / 2)
        imag_part = jnp.sin(delta_f32 / 2) * jnp.sin(2 * chi_f32)
        
        return real_part + 1j * imag_part

# =================================================================================================
# 2. CORE ARCHITECTURE - REBUILT FOR STABILITY
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
# 3. PROJECT CHIMERA: The Top-Level Model - REBUILT AND STABLE
# =================================================================================================
class ChimeraAE(nn.Module):
    d_model: int; dtype: Any = DTYPE
    def setup(self):
        self.image_encoder = ImageEncoder(d_model=self.d_model, name="image_encoder", dtype=self.dtype)
        self.text_encoder = TextEncoder(d_model=self.d_model, name="text_encoder", dtype=self.dtype)
        self.topological_decoder = TopologicalGenerativeDecoder(d_model=self.d_model, name="topological_decoder", dtype=self.dtype)
        self.image_decoder = ImagePatchDecoder(self.d_model, name="image_decoder", dtype=self.dtype)

    # --- THE FIX IS HERE: `__call__` now has a standard signature and always returns (loss, aux_dict) ---
    def __call__(self, batch):
        images, gt_text_embs = batch['images'], batch['clip_text_embeddings']
        z_image = self.image_encoder(images)
        z_text = self.text_encoder(gt_text_embs)
        recon_patches = self.topological_decoder(z_image)
        recon_images = self.image_decoder(recon_patches)
        loss_pixel = jnp.mean(jnp.abs(images - recon_images))
        loss_align = jnp.mean((z_image - z_text)**2)
        loss_reg = 0.0001 * (jnp.mean(z_image**2) + jnp.mean(z_text**2))
        w_pixel=1.0; w_align=1.0
        total_loss = (w_pixel * loss_pixel) + (w_align * loss_align) + loss_reg
        aux = {'loss': total_loss, 'pixel_loss': loss_pixel, 'align_loss': loss_align, 'reg_loss': loss_reg}
        return total_loss, aux
    
    # --- ADDED BACK: Inference methods required by the Generator class ---
    def encode_image(self, images):
        return self.image_encoder(images)
    def encode_text(self, text_embeds):
        return self.text_encoder(text_embeds)
    def decode_from_z(self, z):
        generated_patches = self.topological_decoder(z)
        return self.image_decoder(generated_patches)

# =================================================================================================
# 4. DATA HANDLING & TRAINING - Adapted for Simpler Model
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

class Trainer:
    def __init__(self, args):
        self.args = args; self.num_devices = jax.local_device_count()
        self.should_shutdown=False; signal.signal(signal.SIGINT, lambda s,f: setattr(self,'should_shutdown',True))
        self.pixel_loss_history=deque(maxlen=200); self.align_loss_history=deque(maxlen=200)
        self.model = ChimeraAE(d_model=args.d_model)

    def _get_gpu_stats(self):
        try: h=pynvml.nvmlDeviceGetHandleByIndex(0); m=pynvml.nvmlDeviceGetMemoryInfo(h); u=pynvml.nvmlDeviceGetUtilizationRates(h); return f"{m.used/1024**3:.2f}/{m.total/1024**3:.2f} GiB", f"{u.gpu}%"
        except: return "N/A", "N/A"
    def _get_sparkline(self, data: deque, w=50):
        s=" ▂▃▄▅▆▇█"; hist=np.array([val for val in data if np.isfinite(val)])
        if len(hist)<2: return " "*w
        hist=hist[-w:]; min_v,max_v=hist.min(),hist.max()
        if max_v==min_v: return "".join([s[0]]*len(hist))
        bins=np.linspace(min_v,max_v,len(s)); indices=np.clip(np.digitize(hist,bins)-1,0,len(s)-1); return "".join(s[i] for i in indices)

    def train(self):
        ckpt_path=Path(f"{self.args.basename}_{self.args.d_model}d.pkl")
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(self.args.lr))
        
        if ckpt_path.exists():
            print(f"--- Resuming from {ckpt_path} ---")
            with open(ckpt_path,'rb') as f: data=pickle.load(f)
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=data['params'],tx=optimizer)
            try: state = state.replace(opt_state=data['opt_state'])
            except (KeyError, ValueError): print("Warning: Optimizer state not compatible. Re-initializing.")
            start_epoch=data.get('epoch',-1)+1
        else:
            print("--- Initializing new model ---")
            dummy_batch={'images':jnp.zeros((1,64,64,3)), 'clip_text_embeddings': jnp.zeros((1,512))}
            params=self.model.init(jax.random.PRNGKey(0), dummy_batch)['params']
            state=train_state.TrainState.create(apply_fn=self.model.apply,params=params,tx=optimizer); start_epoch=0
        
        p_state = replicate(state)

        @partial(jax.pmap, axis_name='devices', donate_argnums=(0,))
        def train_step(state, batch):
            def loss_fn(params):
                # This now correctly calls the model which returns (loss, aux)
                return state.apply_fn({'params': params}, batch)
            # has_aux=True works because loss_fn returns a (value, dict) tuple
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, 'devices')
            aux = jax.lax.pmean(aux, 'devices')
            return state.apply_gradients(grads=grads), aux

        dataset=create_dataset(self.args.image_dir,self.args.batch_size*self.num_devices)
        it=dataset.as_numpy_iterator()
        with open(Path(self.args.image_dir)/"dataset_info.pkl",'rb') as f:
            steps_per_epoch=(pickle.load(f)['num_samples'])//(self.args.batch_size*self.num_devices)

        layout=Layout(name="root"); layout.split(Layout(Panel(f"[bold]Project Chimera (Reborn & Stable)[/] | Model: [cyan]{self.args.basename}_{self.args.d_model}d[/]", expand=False),size=3), Layout(ratio=1,name="main"), Layout(size=3,name="footer"))
        layout["main"].split_row(Layout(name="left"),Layout(name="right",ratio=2))
        progress=Progress(TextColumn("[bold]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"), BarColumn(),"[p.p.]{task.percentage:>3.1f}%","•",TimeRemainingColumn(),"•",TimeElapsedColumn())
        epoch_task=progress.add_task("epoch",total=steps_per_epoch,epoch=start_epoch+1,total_epochs=self.args.epochs); layout['footer'].update(progress)
        
        with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
            for epoch in range(start_epoch, self.args.epochs):
                progress.update(epoch_task, completed=0, epoch=epoch+1)
                for step in range(steps_per_epoch):
                    if self.should_shutdown: break
                    p_state, metrics = train_step(p_state, common_utils.shard(next(it)))
                    if step%5==0:
                        m=unreplicate(metrics)
                        self.pixel_loss_history.append(m['pixel_loss'].item())
                        self.align_loss_history.append(m['align_loss'].item())
                        stats_tbl=Table(show_header=False,box=None,padding=(0,1)); stats_tbl.add_column(style="dim",width=15); stats_tbl.add_column(justify="right")
                        stats_tbl.add_row("Total Loss",f"[bold green]{m['loss'].item():.4e}[/]"); mem,util=self._get_gpu_stats(); stats_tbl.add_row("GPU Mem",f"[yellow]{mem}[/]"); stats_tbl.add_row("GPU Util",f"[yellow]{util}[/]");
                        layout["left"].update(Panel(stats_tbl,title="[bold]📊 Stats[/]",border_style="blue"))
                        spark_w = max(1, (live.console.width * 2 // 3) - 25)
                        loss_tbl=Table(show_header=False,box=None,padding=(0,1)); loss_tbl.add_column(style="dim",width=15); loss_tbl.add_column(width=10, justify="right"); loss_tbl.add_column(ratio=1)
                        loss_tbl.add_row("Pixel Loss", f"{m['pixel_loss'].item():.4f}", f"[yellow]{self._get_sparkline(self.pixel_loss_history, spark_w)}")
                        loss_tbl.add_row("Align Loss", f"[magenta]{m['align_loss'].item():.4f}[/magenta]", f"[magenta]{self._get_sparkline(self.align_loss_history, spark_w)}")
                        loss_tbl.add_row("Reg", f"{m['reg_loss'].item():.4e}", "")
                        layout["right"].update(Panel(loss_tbl, title="[bold]📉 Losses[/]", border_style="blue"))
                    progress.update(epoch_task,advance=1)
                if self.should_shutdown: break
                state_to_save=unreplicate(p_state)
                with open(ckpt_path,'wb') as f: pickle.dump({'params':jax.device_get(state_to_save.params),'opt_state':jax.device_get(state_to_save.opt_state),'epoch':epoch},f)
                live.console.print(f"--- :floppy_disk: Checkpoint saved for epoch {epoch+1} ---")

# =================================================================================================
# 5. GENERATOR CLASS for Inference
# =================================================================================================
class Generator:
    def __init__(self, args):
        self.args = args; self.d_model = args.d_model
        self.model = ChimeraAE(d_model=self.d_model)
        model_path = Path(f"{self.args.basename}_{self.args.d_model}d.pkl")
        if not model_path.exists(): print(f"[FATAL] Model file not found at {model_path}. Please train first."), sys.exit(1)
        print(f"--- Loading model from {model_path} ---");
        with open(model_path, 'rb') as f: self.params = pickle.load(f)['params']
        self.clip_model, _ = clip.load("ViT-B/32", device=_clip_device)
        
    @partial(jax.jit, static_argnames=('self',))
    def _encode_image(self, images): return self.model.apply({'params': self.params}, images, method=self.model.encode_image)
    @partial(jax.jit, static_argnames=('self',))
    def _encode_text_embeds(self, embeds): return self.model.apply({'params': self.params}, embeds, method=self.model.encode_text)
    @partial(jax.jit, static_argnames=('self',))
    def _decode(self, z): return self.model.apply({'params': self.params}, z, method=self.model.decode_from_z)
    
    def _get_z_from_text(self, prompt: str):
        with torch.no_grad():
            text_tokens = clip.tokenize([prompt]).to(_clip_device)
            text_embed = self.clip_model.encode_text(text_tokens).cpu().numpy().astype(jnp.float32)
        return self._encode_text_embeds(text_embed)

    def _get_z_from_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
        img_np = (np.array(img, dtype=np.float32) / 127.5) - 1.0
        return self._encode_image(jnp.expand_dims(img_np, axis=0))

    def generate(self):
        print(f"--- Generating from prompt: '{self.args.prompt}' ---")
        z = self._get_z_from_text(self.args.prompt)
        generated_image = self._decode(z)
        img_np = np.array((jax.device_get(generated_image[0]) * 0.5 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
        prompt_slug = "".join(c if c.isalnum() else "_" for c in self.args.prompt).strip("_")[:50]
        save_path = f"{prompt_slug}_{self.args.seed}.png"; Image.fromarray(img_np).save(save_path); print(f"✅ Image saved to {save_path}")

    def refine(self):
        print(f"--- Refining prompt '{self.args.prompt}' over {self.args.steps} steps (Guidance: {self.args.guidance_strength}) ---")
        prompt_slug = "".join(c if c.isalnum() else "_" for c in self.args.prompt).strip("_")[:50]
        z_anchor_text = self._get_z_from_text(self.args.prompt); current_z = z_anchor_text
        for step in range(self.args.steps):
            print(f"  > Step {step+1}/{self.args.steps}...")
            current_image_tensor = self._decode(current_z)
            img_np = np.array((jax.device_get(current_image_tensor[0]) * 0.5 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
            save_path = f"{prompt_slug}_{self.args.seed}_step_{step:02d}.png"; Image.fromarray(img_np).save(save_path)
            if step < self.args.steps - 1:
                z_current_img = self._get_z_from_image(save_path)
                guidance_vector = z_anchor_text - z_current_img
                current_z = z_current_img + guidance_vector * self.args.guidance_strength
        print(f"✅ Refinement complete. Final image at: {save_path}")

    def blend(self):
        print(f"--- Blending base '{self.args.base}' with modifier '{self.args.modifier}' ---")
        p = Path(self.args.base)
        is_file = p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')
        z_a = self._get_z_from_image(self.args.base) if is_file else self._get_z_from_text(self.args.base)
        z_b = self._get_z_from_text(self.args.modifier)
        z_blended = z_a + (z_b - z_a) * self.args.strength
        blended_image = self._decode(z_blended)
        img_np = np.array((jax.device_get(blended_image[0]) * 0.5 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
        base_slug = Path(self.args.base).stem if is_file else "".join(c if c.isalnum() else "_" for c in self.args.base).strip("_")[:25]
        mod_slug = "".join(c if c.isalnum() else "_" for c in self.args.modifier).strip("_")[:25]
        save_path = f"blend_{base_slug}__{mod_slug}_{int(self.args.strength*100)}_{self.args.seed}.png"; Image.fromarray(img_np).save(save_path); print(f"✅ Blended image saved to {save_path}")

# =================================================================================================
# 6. MAIN EXECUTION BLOCK
# =================================================================================================
def main():
    parser = argparse.ArgumentParser(description="Project Chimera (Reborn): A Stable, Physics-Grounded Generative Model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_prep = subparsers.add_parser("prepare-data", help="Prepare images and CLIP text embeddings.")
    p_prep.add_argument('--image-dir', type=str, required=True)
    p_train = subparsers.add_parser("train", help="Train the Autoencoder.")
    p_train.add_argument('--image-dir', type=str, required=True); p_train.add_argument('--basename', type=str, required=True); p_train.add_argument('--d-model', type=int, default=256)
    p_train.add_argument('--epochs', type=int, default=100); p_train.add_argument('--batch-size', type=int, default=32, help="Per device."); p_train.add_argument('--lr', type=float, default=1e-4); p_train.add_argument('--seed', type=int, default=42)
    p_gen = subparsers.add_parser("generate", help="Generate an image from a text prompt.")
    p_gen.add_argument('--prompt', type=str, required=True); p_gen.add_argument('--basename', type=str, required=True); p_gen.add_argument('--d-model', type=int, default=256)
    p_gen.add_argument('--seed', type=int, default=lambda: int(time.time()), help="Random seed for generation.")
    p_refine = subparsers.add_parser("refine", help="Iteratively refine a prompt.")
    p_refine.add_argument('--prompt', type=str, required=True); p_refine.add_argument('--basename', type=str, required=True); p_refine.add_argument('--d-model', type=int, default=256)
    p_refine.add_argument('--steps', type=int, default=10); p_refine.add_argument('--guidance-strength', type=float, default=0.2)
    p_refine.add_argument('--seed', type=int, default=lambda: int(time.time()))
    p_blend = subparsers.add_parser("blend", help="Geometrically blend two concepts.")
    p_blend.add_argument('--base', type=str, required=True, help="Base concept (text or path to image)."); p_blend.add_argument('--modifier', type=str, required=True, help="Modifier concept (text).")
    p_blend.add_argument('--strength', type=float, default=0.5, help="Blend strength from 0 (base) to 1 (modifier).")
    p_blend.add_argument('--basename', type=str, required=True); p_blend.add_argument('--d-model', type=int, default=256); p_blend.add_argument('--seed', type=int, default=lambda: int(time.time()))
    args = parser.parse_args()
    if 'seed' in args and callable(args.seed): args.seed = args.seed()

    if args.command == "prepare-data": prepare_data(args.image_dir)
    elif args.command == "train": Trainer(args).train()
    elif args.command == "generate": Generator(args).generate()
    elif args.command == "refine": Generator(args).refine()
    elif args.command == "blend": Generator(args).blend()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n--- Program terminated by user. ---"), sys.exit(0)