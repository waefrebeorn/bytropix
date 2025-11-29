# Wubu_Smith_Architect.py
# THE GEOMETRIC BRAIN v4.2 [STABILITY PATCH]
#
# CONFIG: 2048 Neurons, 2048 Context.
# FIX: Added missing 'shutil' import.

import os
import sys
import time
import math
import pickle
import traceback
import multiprocessing as mp
import requests
import numpy as np
import shutil # <--- Fixed: Added missing import
from functools import partial
from typing import Any, Tuple, List

# --- MEMORY SETTINGS ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform' 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
from flax import linen as nn
from flax.training import train_state
import chex

# --- GUI ---
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich_pixels import Pixels
from PIL import Image, ImageDraw

# --- TOKENIZER ---
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# ==============================================================================
# 1. HYPER-PARAMETERS
# ==============================================================================
class Config:
    # GEOMETRY
    NUM_NEURONS = 2048      
    LOBE_SEP = 0.6          
    
    # TRANSFORMER
    SEQ_LEN = 2048          
    D_MODEL = 2048          
    NUM_LAYERS = 4          
    NUM_HEADS = 8
    D_FF = 2048             
    
    # OPTIMIZATION
    LR = 1e-4               
    BATCH_SIZE = 1          
    
    USE_SMITH_GATING = True 
    GEO_WEIGHT = 0.05
    ENTROPY_WEIGHT = 0.01
    
    # I/O
    VOCAB_SIZE = 8192

# ==============================================================================
# 2. SMITH CHART MATH
# ==============================================================================
def smith_transform(real_h, imag_h):
    r = jax.nn.softplus(real_h) 
    x = imag_h
    denominator = (r + 1.0)**2 + x**2
    u = (r**2 + x**2 - 1.0) / denominator
    v = (2.0 * x) / denominator
    gamma_mag = jnp.sqrt(u**2 + v**2)
    return u, v, gamma_mag

# ==============================================================================
# 3. GEOMETRY
# ==============================================================================
def generate_bicameral_brain():
    points = []
    n = Config.NUM_NEURONS
    half = n // 2
    phi = math.pi * (3. - math.sqrt(5.))
    
    for i in range(half):
        y = 1 - (i / float(half - 1)) * 2 
        r = math.sqrt(1 - y * y)
        theta = phi * i 
        points.append((math.cos(theta)*r*0.8 - Config.LOBE_SEP, y, math.sin(theta)*r*0.9))

    for i in range(half):
        y = 1 - (i / float(half - 1)) * 2 
        r = math.sqrt(1 - y * y)
        theta = phi * i 
        points.append((math.cos(theta)*r*0.8 + Config.LOBE_SEP, y, math.sin(theta)*r*0.9))
        
    points_np = np.array(points, dtype=np.float32)
    
    wires = []
    chunk_size = 512 
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        dists = np.linalg.norm(points_np[i:end, None] - points_np[None, :], axis=-1)
        for idx in range(end - i):
            global_idx = i + idx
            nearest = np.argsort(dists[idx])[1:3] 
            for t in nearest:
                if global_idx < t: wires.append((global_idx, t))

    dists_full = np.linalg.norm(points_np[:, None] - points_np[None, :], axis=-1)
    geo_weights = jnp.exp(-jnp.array(dists_full) * 2.0)
    geo_weights = geo_weights.at[jnp.diag_indices(n)].set(0.0)
            
    return points_np, wires, geo_weights

POINTS_3D, WIRES, GEO_WEIGHTS_GPU = generate_bicameral_brain()

# ==============================================================================
# 4. SMITH-ENHANCED TRANSFORMER
# ==============================================================================

class TransformerLayer(nn.Module):
    config: Config
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, _):
        residual = x
        h = nn.LayerNorm(dtype=self.dtype)(x)
        h = nn.SelfAttention(num_heads=Config.NUM_HEADS, dtype=self.dtype)(h)
        x = residual + h
        
        residual = x
        h = nn.LayerNorm(dtype=self.dtype)(x)
        h = nn.Dense(Config.D_FF, dtype=self.dtype)(h)
        h = nn.gelu(h)
        h = nn.Dense(Config.D_MODEL, dtype=self.dtype)(h)
        x = residual + h
        return x, None

class WubuSmithCore(nn.Module):
    vocab_size: int
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Embed(self.vocab_size, Config.D_MODEL, dtype=self.dtype)(x)
        
        pos = jnp.arange(0, x.shape[1])[None, :, None]
        div_term = jnp.exp(jnp.arange(0, Config.D_MODEL, 2) * -(math.log(10000.0) / Config.D_MODEL))
        pe = jnp.zeros_like(x)
        pe = pe.at[:, :, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, :, 1::2].set(jnp.cos(pos * div_term))
        x = x + pe.astype(self.dtype)

        ScannedLayer = nn.remat(TransformerLayer)
        x, _ = nn.scan(
            ScannedLayer,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            length=Config.NUM_LAYERS
        )(config=Config, dtype=self.dtype)(x, None)

        half_dim = Config.D_MODEL // 2
        x_f32 = x.astype(jnp.float32) 
        left_lobe = x_f32[..., :half_dim]
        right_lobe = x_f32[..., half_dim:]
        
        u, v, gamma = smith_transform(left_lobe, right_lobe)
        modulation = 1.0 - gamma
        
        modulated_left = left_lobe * modulation
        modulated_right = right_lobe * modulation
        
        brain_state = jnp.concatenate([modulated_left, modulated_right], axis=-1)
        logits = nn.Dense(self.vocab_size, dtype=self.dtype)(brain_state.astype(self.dtype))
        smith_stats = jnp.stack([u.mean(), v.mean(), gamma.mean()])
        
        return logits, brain_state, smith_stats

# ==============================================================================
# 5. TRAINING ENGINE
# ==============================================================================
def train_process(queue, stop_event):
    try:
        try: text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
        except: text = "Wubu " * 10000
        
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=Config.VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"])
        tokenizer.train_from_iterator([text], trainer=trainer)
        data = jnp.array(tokenizer.encode(text).ids)
        vocab_size = tokenizer.get_vocab_size()
        
        key = jax.random.PRNGKey(42)
        model = WubuSmithCore(vocab_size=vocab_size)
        dummy_x = jnp.ones((1, Config.SEQ_LEN), dtype=jnp.int32)
        params = model.init(key, dummy_x)['params']
        
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(Config.LR))
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        @jit
        def train_step(state, batch):
            def loss_fn(params):
                logits, brain_states, smith_stats = state.apply_fn({'params': params}, batch)
                
                # 1. CE Loss (The Content)
                targets = jnp.roll(batch, -1, axis=1) 
                one_hot = jax.nn.one_hot(targets, vocab_size)
                loss_ce = optax.softmax_cross_entropy(logits, one_hot).mean()
                
                # 2. Impedance Matching (The Tuner)
                gamma = smith_stats[2] 
                
                # DYNAMIC TUNING:
                # If Gamma is high (0.96), multiply the penalty massively.
                # Force the network to prioritize structure over content until it matches.
                # As Gamma drops, this penalty vanishes, letting it learn the text.
                gamma_penalty = (gamma ** 2) * 0.5 
                
                # 3. Geodesic Loss (The Shape)
                avg_act = jnp.mean(jnp.abs(brain_states), axis=1) 
                act_diff = jnp.abs(avg_act[0, :, None] - avg_act[0, None, :])
                loss_geo = jnp.sum(GEO_WEIGHTS_GPU * act_diff) / (Config.NUM_NEURONS**2)
                
                total_loss = loss_ce + gamma_penalty + (loss_geo * Config.GEO_WEIGHT)
                return total_loss, (loss_ce, loss_geo, gamma, avg_act[0], smith_stats)
                
            grad_fn = value_and_grad(loss_fn, has_aux=True)
            (loss, aux), grads = grad_fn(state.params)
            
            # ADAPTIVE GRADIENT SCALING
            # If reflection is high, drive the gradients harder to break the "Open Circuit"
            # Gamma is usually 0.0 to 1.0. 
            # If Gamma > 0.9, scale grads by 1.2x
            current_gamma = aux[2]
            grad_scale = 1.0 + (current_gamma * 0.5) 
            grads = jax.tree_util.tree_map(lambda g: g * grad_scale, grads)
            
            new_state = state.apply_gradients(grads=grads)
            return new_state, loss, aux

        @jit
        def generate(params, prompt, length=30):
            current = prompt
            for _ in range(length):
                logits, _, _ = model.apply({'params': params}, current)
                next_tok = jnp.argmax(logits[0, -1])
                current = jnp.concatenate([current[:, 1:], next_tok[None, None]], axis=1)
            return current

        ptr = 0
        step = 0
        queue.put({'type': 'READY'})
        
        while not stop_event.is_set():
            if ptr + Config.SEQ_LEN + 1 > len(data): ptr = 0
            batch = jax.lax.dynamic_slice(data, (ptr,), (Config.SEQ_LEN,)).reshape(1, Config.SEQ_LEN)
            state, loss, (ce, geo, gamma, activity, smith_coords) = train_step(state, batch)
            
            ptr += Config.SEQ_LEN
            step += 1
            
            if step % 2 == 0:
                txt = None
                if step % 10 == 0:
                    try:
                        seed = jnp.array(tokenizer.encode("The brain").ids).reshape(1, -1)
                        pad_len = Config.SEQ_LEN - seed.shape[1]
                        pad = jnp.zeros((1, pad_len), dtype=jnp.int32)
                        seed_padded = jnp.concatenate([pad, seed], axis=1)
                        out_tokens = generate(state.params, seed_padded)
                        full_txt = tokenizer.decode(np.array(out_tokens[0]))
                        txt = full_txt[-150:].replace("\n", " ")
                    except: pass
                
                packet = {
                    'step': step,
                    'loss': float(loss),
                    'gamma': float(gamma),
                    'smith': np.array(smith_coords), 
                    'activity': np.array(activity), 
                    'points': POINTS_3D, 
                    'wires': WIRES,
                    'text': txt
                }
                try: queue.put_nowait({'type': 'DATA', 'data': packet})
                except: pass
                
    except Exception:
        queue.put({'type': 'ERROR', 'msg': traceback.format_exc()})

# ==============================================================================
# 6. VISUALIZATION (ROBUST GUI)
# ==============================================================================
class WubuSmithGUI:
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(Layout(name="T", size=3), Layout(name="M", ratio=1), Layout(name="B", size=4))
        self.layout["M"].split_row(Layout(name="L", ratio=1), Layout(name="R", ratio=1))
        self.angle = 0.0
        self.last_txt = "Impedance Matching..."
        self.smith_history = []

    def draw_smith_chart(self, draw, w, h, current_uv):
        cx, cy = w//2, h//2
        radius = min(w, h) * 0.4
        if radius < 1: return # Safety for resize
        
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], outline=(50, 50, 50))
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=(100, 100, 100))
        draw.line([cx-radius, cy, cx+radius, cy], fill=(50, 50, 50))
        
        for r_val in [0.5, 1.0, 2.0]:
            u_c = r_val / (r_val + 1)
            rad_c = 1.0 / (r_val + 1)
            px = cx + u_c * radius
            pr = rad_c * radius
            draw.ellipse([px-pr, cy-pr, px+pr, cy+pr], outline=(30, 30, 80))

        self.smith_history.append(current_uv)
        if len(self.smith_history) > 50: self.smith_history.pop(0)
        
        for i, (u, v) in enumerate(self.smith_history):
            px = cx + u * radius
            py = cy - v * radius 
            dist = math.sqrt(u*u + v*v)
            r = int(dist * 255)
            g = int((1-dist) * 255)
            draw.rectangle([px-2, py-2, px+2, py+2], fill=(r, g, 50))

    def render(self, d):
        # 1. Safety Check for Dimensions
        term_w, term_h = shutil.get_terminal_size()
        w = max(20, (term_w // 2) - 4)
        h = max(20, (term_h - 8) * 2)
        
        try:
            # LEFT: 3D BRAIN
            img_3d = Image.new('RGB', (w, h), (5, 5, 10))
            draw_3d = ImageDraw.Draw(img_3d)
            cx, cy = w/2, h/2
            scale = min(w, h) * 0.35
            self.angle += 0.02
            ca, sa = math.cos(self.angle), math.sin(self.angle)
            
            act = d['activity']
            norm_act = (act - act.min()) / (act.max() - act.min() + 1e-6)
            
            proj = []
            for i, (x, y, z) in enumerate(d['points']):
                rx = x*ca - z*sa
                rz = x*sa + z*ca
                proj.append({'x': cx+rx*scale, 'y': cy+y*scale, 'z': rz, 'v': norm_act[i], 'i': i})
                
            proj.sort(key=lambda x: x['z'])
            for p in proj:
                if p['z'] > -0.8:
                    sz = max(1, int(scale * 0.015))
                    val = int(p['v']*255)
                    col = (val, 255-val, 255) if p['i'] < 1024 else (255, 255-val, val)
                    draw_3d.ellipse([p['x']-sz, p['y']-sz, p['x']+sz, p['y']+sz], fill=col)

            # RIGHT: SMITH CHART
            img_smith = Image.new('RGB', (w, h), (0, 0, 0))
            draw_smith = ImageDraw.Draw(img_smith)
            u, v, gamma = d['smith']
            self.draw_smith_chart(draw_smith, w, h, (u, v))

            if d.get('text'): self.last_txt = d['text']
            
            head = f"WUBU SMITH CORE | REFLECTION (GAMMA): {d['gamma']:.4f} | LOSS: {d['loss']:.4f}"
            self.layout["T"].update(Panel(head, style="bold white on blue"))
            self.layout["L"].update(Panel(Pixels.from_image(img_3d), title="CORTEX ACTIVITY"))
            self.layout["R"].update(Panel(Pixels.from_image(img_smith), title="SEMANTIC IMPEDANCE (SMITH CHART)"))
            self.layout["B"].update(Panel(f"[yellow]{self.last_txt}[/]", title="STREAM"))
        
        except Exception as e:
            # Fallback for rendering errors
            err_msg = f"[bold red]RENDER ERROR:[/bold red] {str(e)}"
            self.layout["T"].update(Panel(err_msg, style="red"))
            
        return self.layout

if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = mp.Queue(); stop = mp.Event()
    console = Console()
    p = mp.Process(target=train_process, args=(q, stop))
    p.start()
    
    gui = WubuSmithGUI()
    ready = False
    
    try:
        # Phase 1: Wait for Signal
        with console.status("Calibrating Neural Impedance...") as status:
            while not ready:
                if not p.is_alive():
                    raise RuntimeError("Training process died during initialization.")
                try:
                    m = q.get(timeout=0.1)
                    if m.get('type') == 'READY': ready = True
                    if m.get('type') == 'ERROR': 
                        raise RuntimeError(f"Training Error: {m['msg']}")
                except mp.queues.Empty:
                    pass

        # Phase 2: Live Loop
        with Live(gui.layout, refresh_per_second=10, screen=True) as live:
            while True:
                if not p.is_alive():
                    raise RuntimeError("Training process terminated unexpectedly.")
                    
                try:
                    msg = q.get_nowait()
                    if msg['type'] == 'DATA': 
                        live.update(gui.render(msg['data']))
                    elif msg['type'] == 'ERROR': 
                        raise RuntimeError(f"Engine Error: {msg['msg']}")
                except mp.queues.Empty:
                    time.sleep(0.01)

    except KeyboardInterrupt:
        stop.set()
    except Exception as e:
        stop.set()
        console.clear() 
        console.print(Panel(f"[bold red]FATAL CRASH[/bold red]\n\n{str(e)}", border_style="red"))
        while not q.empty():
            m = q.get()
            if m.get('type') == 'ERROR':
                console.print(f"\n[dim]{m['msg']}[/dim]")
    finally:
        if p.is_alive():
            p.terminate()
            p.join()